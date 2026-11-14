# Copyright 2026 PyDefi Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional NumPy containers with exact construction and checked int64 arithmetic.

Use ``array`` for numeric values and ``from_raw`` for integer encodings. External
writers to raw storage must preserve its layout, dtype, shape, and declared value
range. Arithmetic accepts fixed-point values and integer constants, not ndarrays.

Reductions (``sum``, ``dot``, ``matmul``) quantize in three stages: product policy
on each completed product, accumulator policy on each addition in logical order,
then sum policy on the final result.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from functools import reduce
from numbers import Number
import operator

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    raise ImportError("NumPy is required for chainfix.arrays; install 'chainfix[arrays]'") from error

from chainfix import _array_kernels as kernels
from chainfix.binary import Fixb, Ufixb, _operand, _selected_math
from chainfix.complex import CFixb
from chainfix.format import Format, _product_format, _sum_format, _unary_format
from chainfix.policy import FixedPointOverflowError, Math

__all__ = ["Array", "array", "from_raw", "empty", "add", "subtract", "multiply",
           "positive", "negative", "absolute", "conjugate", "mag2", "sum", "dot",
           "matmul", "equal", "not_equal", "less", "less_equal", "greater",
           "greater_equal"]

# Product terms materialized per component for one block of a dot/matmul
# reduction. Blocks bound temporary storage without changing any result.
_BLOCK_ELEMENTS = 1 << 16


def _complex_flag(value, *, optional=False):
    if not isinstance(value, bool) and not (optional and value is None):
        raise TypeError("complex must be a boolean" + (" or None" if optional else ""))


def _supported_layout(raw, complex):
    if not raw.flags.aligned or any(stride < 0 for stride in raw.strides):
        return False
    if raw.flags.c_contiguous:
        return True
    if complex:
        return False
    # A component of contiguous interleaved I/Q storage has twice the ordinary
    # C strides. Ignore singleton axes; they have no effect on accessed addresses.
    stride = 2 * raw.itemsize
    for size, actual in zip(reversed(raw.shape), reversed(raw.strides)):
        if size > 1 and actual != stride:
            return False
        stride *= size
    return True


def _check_storage(raw, format, complex):
    if type(raw) is not np.ndarray:
        raise TypeError("raw storage must be a plain ndarray, not a subclass")
    if raw.dtype.kind not in "iu" or not raw.dtype.isnative or raw.itemsize > 8:
        raise TypeError("raw storage must have a native integer dtype of at most 64 bits")
    limits = np.iinfo(raw.dtype)
    if limits.min > format.min_int or limits.max < format.max_int:
        raise ValueError("storage dtype must hold the entire declared format range")
    if complex and (raw.ndim == 0 or raw.shape[-1] != 2):
        raise ValueError("complex raw storage requires a trailing I/Q axis of length 2")
    if not _supported_layout(raw, complex):
        raise ValueError("unsupported layout; use from_raw(..., copy=True) to materialize")


class Array:
    """An integer ndarray and immutable binary format, with no arithmetic policy.

    Construct with numeric values, or use :func:`from_raw` to share storage.
    Only C-contiguous arrays and interleaved component views are supported.
    """

    __slots__ = ("_raw", "_format", "_complex")
    __hash__ = None

    def __init__(self, values, *, format: Format, complex: bool | None = None,
                 math: Math | None = None):
        result = _from_values(values, format, complex, math)
        object.__setattr__(self, "_raw", result.raw)
        object.__setattr__(self, "_format", result.format)
        object.__setattr__(self, "_complex", result.is_complex)

    def __setattr__(self, name, value):
        raise AttributeError("Array metadata is read-only; write encoded values through raw")

    def __delattr__(self, name):
        raise AttributeError("Array metadata is read-only")

    def __copy__(self):
        """Copy the wrapper, sharing its integer storage and immutable format."""
        return type(self)._from_storage(self.raw, self.format, self.is_complex)

    def __deepcopy__(self, memo):
        """Copy storage independently, preserving repeated references via memo."""
        raw = deepcopy(self.raw, memo)
        raw.flags.writeable = self.raw.flags.writeable
        result = type(self)._from_storage(raw, self.format, self.is_complex)
        memo[id(self)] = result
        return result

    def __reduce__(self):
        # Materialize component views, but preserve dtype and all raw bits.
        # Reconstruction must not quantize or scan an uninitialized output.
        return (_restore_array, (type(self), self.raw.copy(order="C"), self.format,
                                 self.is_complex, self.raw.flags.writeable))

    @classmethod
    def _from_storage(cls, raw, format, complex):
        self = object.__new__(cls)
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_format", format)
        object.__setattr__(self, "_complex", complex)
        return self

    @property
    def raw(self):
        """Encoded integer storage; complex storage has a trailing [I, Q] axis."""
        return self._raw

    @property
    def int(self):
        if self.is_complex:
            raise AttributeError("complex arrays have no int; use real.int and imag.int")
        return self.raw

    @property
    def format(self):
        return self._format

    @property
    def is_complex(self):
        return self._complex

    @property
    def shape(self):
        return self.raw.shape[:-1] if self.is_complex else self.raw.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return self.raw.size // 2 if self.is_complex else self.raw.size

    @property
    def real(self):
        if not self.is_complex:
            return self
        raw = self.raw[..., :1].reshape(self.shape)
        return Array._from_storage(raw, self.format, False)

    @property
    def imag(self):
        if self.is_complex:
            raw = self.raw[..., 1:].reshape(self.shape)
        else:
            raw = np.zeros(self.shape, dtype=self.raw.dtype)
            raw.flags.writeable = self.raw.flags.writeable
        return Array._from_storage(raw, self.format, False)

    def to_numpy(self):
        """Return an explicitly approximate float64/complex128 copy."""
        # ldexp avoids first constructing an underflowing/overflowing scale.
        # Beyond these exponents every nonzero int64 value over/underflows.
        exponent = max(-1139, min(1024, -self.format.precision))
        with np.errstate(over="ignore", under="ignore"):
            real = np.ldexp(self.real.raw.astype(np.float64), exponent)
            if not self.is_complex:
                return real
            result = np.empty(self.shape, dtype=np.complex128)
            result.real = real
            result.imag = np.ldexp(self.imag.raw.astype(np.float64), exponent)
            return result

    def quantize(self, format: Format, *, math: Math | None = None):
        kernels.check_format(format, self.is_complex)
        raw = kernels.convert(self.raw, self.format, format, _selected_math(math))
        # A value-preserving conversion can return this array's own int64 storage.
        # The result always owns independent, writable, C-contiguous storage; copy
        # again only when conversion did not already produce it.
        copy = True if raw is self.raw or np.may_share_memory(raw, self.raw) else None
        return Array._from_storage(np.array(raw, dtype=kernels.storage_dtype(format),
                                            copy=copy, order="C"), format, self.is_complex)

    def conjugate(self, *, math: Math | None = None):
        return conjugate(self, math=math)

    def mag2(self, *, math: Math | None = None):
        return mag2(self, math=math)

    def sum(self, axis=None, *, keepdims=False, out=None, math: Math | None = None):
        return sum(self, axis, keepdims=keepdims, out=out, math=math)

    def __len__(self):
        if not self.ndim:
            raise TypeError("len() of a zero-dimensional array")
        return self.shape[0]

    def __iter__(self):
        if not self.ndim:
            raise TypeError("iteration over a zero-dimensional array")
        return (self[index] for index in range(len(self)))

    def __bool__(self):
        raise TypeError("fixed-point array truth is ambiguous; compare elements explicitly")

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        for item in keys:
            if item is None or item is Ellipsis:
                continue
            if isinstance(item, slice):
                if item.step not in (None, 1):
                    raise ValueError("only unit-step basic slicing is supported")
            elif isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise TypeError("only basic integer/slice indexing is supported")
        # The module-level sum is the fixed-point reduction, not the builtin.
        ellipses = builtins.sum(item is Ellipsis for item in keys)
        count = builtins.sum(item is not None and item is not Ellipsis for item in keys)
        if ellipses > 1 or count > self.ndim:
            raise IndexError("too many indices or ellipses for logical array shape")
        # Keep even a zero-width ellipsis: NumPy then returns a shared 0-d
        # ndarray instead of detaching a scalar when all axes are selected.
        expanded = list(keys)
        if not ellipses:
            expanded.extend([slice(None)] * (self.ndim - count))
        if self.is_complex:
            expanded.append(slice(None))
        raw = self.raw[tuple(expanded)]
        if self.is_complex and raw.shape == (2,) and not ellipses:
            return CFixb.from_raw(i=int(raw[0]), q=int(raw[1]), format=self.format)
        if not self.is_complex and np.isscalar(raw):
            return (Fixb if self.format.signed else Ufixb).from_raw(int(raw), format=self.format)
        if not _supported_layout(raw, self.is_complex):
            raise ValueError("indexing produced an unsupported layout")
        return Array._from_storage(raw, self.format, self.is_complex)

    def __array__(self, dtype=None, copy=None):
        raise TypeError("implicit ndarray conversion discards format; use raw or to_numpy()")

    def __array_function__(self, func, types, args, kwargs):
        if func not in _FUNCTIONS:
            raise TypeError(f"NumPy function {func.__name__} is unsupported for fixed-point arrays")
        return _FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Leave foreign types a chance to dispatch, but keep recognized numeric
        # inputs strict so NumPy cannot silently coerce them to object arrays.
        outputs = kwargs.get("out", ())
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        if any(not _supports_operand(value) and not _recognized_operand(value)
               for value in inputs + tuple(out for out in outputs if out is not None)):
            return NotImplemented
        if ufunc is np.add and method == "reduce":
            return _numpy_reduce(*inputs, **kwargs)
        if method != "__call__" or ufunc not in _UFUNCS:
            raise TypeError(f"NumPy {ufunc.__name__}.{method} is unsupported for fixed-point arrays")
        if kwargs.pop("where", True) is not True:
            raise TypeError("fixed-point ufuncs require where=True")
        out = _normalize_out(kwargs.pop("out", None))
        if kwargs:
            raise TypeError(f"unsupported fixed-point ufunc keywords: {', '.join(kwargs)}")
        function = _UFUNCS[ufunc]
        if ufunc in _COMPARISONS:
            if out is not None:
                raise TypeError("comparison out is unsupported")
            return function(*inputs)
        return function(*inputs, out=out)

    def __add__(self, other):
        return add(self, other) if _supports_operand(other) else NotImplemented

    def __radd__(self, other):
        return add(other, self) if _supports_operand(other) else NotImplemented

    def __sub__(self, other):
        return subtract(self, other) if _supports_operand(other) else NotImplemented

    def __rsub__(self, other):
        return subtract(other, self) if _supports_operand(other) else NotImplemented

    def __mul__(self, other):
        return multiply(self, other) if _supports_operand(other) else NotImplemented

    def __rmul__(self, other):
        return multiply(other, self) if _supports_operand(other) else NotImplemented

    def __matmul__(self, other):
        return matmul(self, other) if isinstance(other, Array) else NotImplemented

    def __rmatmul__(self, other):
        return matmul(other, self) if isinstance(other, Array) else NotImplemented

    def __pos__(self):
        return positive(self)

    def __neg__(self):
        return negative(self)

    def __abs__(self):
        return absolute(self)

    def __eq__(self, other):
        return equal(self, other) if _comparison_dispatch(other) else NotImplemented

    def __ne__(self, other):
        return not_equal(self, other) if _comparison_dispatch(other) else NotImplemented

    def __lt__(self, other):
        return less(self, other) if _supports_operand(other) else NotImplemented

    def __le__(self, other):
        return less_equal(self, other) if _supports_operand(other) else NotImplemented

    def __gt__(self, other):
        return greater(self, other) if _supports_operand(other) else NotImplemented

    def __ge__(self, other):
        return greater_equal(self, other) if _supports_operand(other) else NotImplemented

    def __repr__(self):
        return f"Array(shape={self.shape}, format={self.format!r}, complex={self.is_complex})"


# Keep this module/name stable: saved pickles reference this reconstruction helper.
def _restore_array(cls, raw, format, complex, writeable):
    _complex_flag(complex)
    kernels.check_format(format, complex)
    _check_storage(raw, format, complex)
    result = cls._from_storage(raw, format, complex)
    result.raw.flags.writeable = writeable
    return result


def _native_value(value):
    return value.item() if isinstance(value, np.generic) and value.dtype.kind in "biufc" else value


def _supports_operand(value):
    # Only dispatch on type here. Supported operands still pass through all
    # format/domain validation, whose errors must not be swallowed by operators.
    return isinstance(_native_value(value), (Array, Fixb, Ufixb, CFixb, int))


def _recognized_operand(value):
    return isinstance(value, (Number, Decimal, list, tuple, np.ndarray, np.generic))


def _comparison_dispatch(value):
    # Equality's fallback is scalar False/True, so unsupported numeric inputs
    # must reach validation rather than silently looking like a comparison.
    return _supports_operand(value) or _recognized_operand(value)


def _from_values(values, format, complex, math):
    _complex_flag(complex, optional=True)
    policy = _selected_math(math)
    if isinstance(values, Array):
        has_complex = values.is_complex
    else:
        # Object storage is confined to setup: preserve heterogeneous exact
        # inputs rather than promoting large integers to float first.
        values = np.asarray(values, dtype=object)
        has_complex = any(isinstance(v, (builtins.complex, np.complexfloating, CFixb))
                          for v in values.flat)
    if complex is False and has_complex:
        raise TypeError("complex input cannot construct a real array")
    complex = has_complex if complex is None else complex
    kernels.check_format(format, complex)
    result = empty(values.shape, format=format, complex=complex)
    scalar_type = CFixb if complex else (Fixb if format.signed else Ufixb)
    for index in np.ndindex(values.shape):
        scalar = scalar_type(_native_value(values[index]), format=format, math=policy)
        result.raw[index] = (scalar.real.int, scalar.imag.int) if complex else scalar.int
    return result


def array(values, *, format: Format, complex: bool | None = None, math: Math | None = None) -> Array:
    """Convert numeric values exactly; infer complex storage unless specified."""
    return _from_values(values, format, complex, math)


def from_raw(raw, *, format: Format, complex: bool = False, copy: bool = False) -> Array:
    """Validate encodings, sharing supported native storage when copy=False.

    Storage must represent the entire format range. External raw writers must
    maintain the declared range; arithmetic does not rescan inputs for violations.
    """
    _complex_flag(complex)
    kernels.check_format(format, complex)
    if not isinstance(copy, bool):
        raise TypeError("copy must be a boolean")
    if isinstance(raw, np.ndarray) and type(raw) is not np.ndarray:
        raise TypeError("raw storage must be a plain ndarray, not a subclass")
    if not copy and not isinstance(raw, np.ndarray):
        raise ValueError("copy=False requires ndarray storage")
    if copy:
        raw = np.asarray(raw)
        raw = np.array(raw, dtype=raw.dtype.newbyteorder("="), order="C", copy=True)
    _check_storage(raw, format, complex)
    if np.any(raw < format.min_int) or np.any(raw > format.max_int):
        raise FixedPointOverflowError("raw array contains values outside the declared format")
    return Array._from_storage(raw, format, complex)


def empty(shape, *, format: Format, complex: bool = False) -> Array:
    """Allocate uninitialized contiguous storage; fill it before using as input."""
    _complex_flag(complex)
    kernels.check_format(format, complex)
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)
    else:
        shape = tuple(shape)
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) for n in shape):
        raise TypeError("shape dimensions must be integers, not booleans")
    raw = np.empty(shape + ((2,) if complex else ()), dtype=kernels.storage_dtype(format))
    return Array._from_storage(raw, format, complex)


def _scalar_operand(value):
    value = _native_value(value)
    if isinstance(value, CFixb):
        return value
    value = _operand(value)
    if value is NotImplemented:
        raise TypeError("operands must be fixed-point arrays/scalars or integer constants; "
                        "use array() or from_raw() for ndarrays")
    return value


def _as_operand(value):
    if isinstance(value, Array):
        return value
    value = _scalar_operand(value)
    if isinstance(value, CFixb):
        kernels.check_format(value.format, True)
        return Array._from_storage(np.array([value.real.int, value.imag.int], dtype=np.int64),
                                   value.format, True)
    kernels.check_format(value.format)
    return Array._from_storage(np.array(value.int, dtype=np.int64), value.format, False)


def _normalize_out(out):
    if isinstance(out, tuple):
        if len(out) != 1:
            raise TypeError("out must contain exactly one destination")
        out = out[0]
    if out is not None and not isinstance(out, Array):
        raise TypeError("out must be a fixed-point Array")
    return out


def _prepare(full, boundary, complex, shape, inputs, out):
    # Preflight all domains and destination properties before any output write.
    kernels.check_format(full, complex)
    target = full if boundary == "full" else boundary
    kernels.check_format(target, complex)
    kernels.check_conversion(full, target)
    out = _normalize_out(out)
    if out is not None:
        if out.shape != shape or out.is_complex != complex:
            raise ValueError("out shape and complexity must match the logical result")
        _check_storage(out.raw, out.format, out.is_complex)
        if not out.raw.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")
        if not out.raw.flags.writeable:
            raise ValueError("out must be writable")
        if any(np.may_share_memory(out.raw, value.raw) for value in inputs):
            raise ValueError("out may overlap an input; use a disjoint destination")
        kernels.check_conversion(target, out.format)
    return target, out


def _convert_result(raw, full, target, destination, policy, *, out=None, scratch=None):
    if full != target:
        raw = kernels.convert(raw, full, target, policy, out=out, scratch=scratch)
    # Preserve the operation boundary even when out has a different format.
    if destination != target:
        raw = kernels.convert(raw, target, destination, policy, out=out, scratch=scratch)
    return raw


def _finish(i, q, full, target, policy, shape, out):
    complex = q is not None
    destination = target if out is None else out.format
    if out is None:
        out = empty(shape, format=target, complex=complex)
    i = _convert_result(i, full, target, destination, policy)
    if complex:
        q = _convert_result(q, full, target, destination, policy)
    if complex:
        out.raw[..., 0], out.raw[..., 1] = i, q
    else:
        out.raw[...] = i
    return out


def _components(value):
    raw = value.raw.astype(np.int64, copy=False)
    return (raw[..., 0], raw[..., 1]) if value.is_complex else (raw, np.int64(0))


def _multiply_format(a, b, a_complex, b_complex):
    full = _product_format(a, b)
    return replace(full, wordlength=full.wordlength + 1) if a_complex and b_complex else full


def _binary(a, b, operation, out, math):
    a, b = _as_operand(a), _as_operand(b)
    policy = _selected_math(math)
    complex = a.is_complex or b.is_complex
    shape = np.broadcast_shapes(a.shape, b.shape)
    if operation == "multiply":
        full = _multiply_format(a.format, b.format, a.is_complex, b.is_complex)
        boundary = policy.product
    else:
        precision = max(a.format.precision, b.format.precision)
        kernels.check_shift(a.format, precision - a.format.precision)
        kernels.check_shift(b.format, precision - b.format.precision)
        full = _sum_format(a.format, b.format, subtract=operation == "subtract")
        boundary = policy.sum
    target, out = _prepare(full, boundary, complex, shape, (a, b), out)
    ai, aq = _components(a)
    bi, bq = _components(b)
    if operation == "multiply":
        if complex:
            i, q, work = (np.empty(shape, dtype=np.int64) for _ in range(3))
            kernels.multiply_complex(ai, aq, bi, bq, i=i, q=q, work=work)
            del work  # Release product scratch before _finish allocates its output.
        else:
            i, q = np.multiply(ai, bi), None
    else:
        ai, aq = ai << (full.precision - a.format.precision), aq << (full.precision - a.format.precision)
        bi, bq = bi << (full.precision - b.format.precision), bq << (full.precision - b.format.precision)
        op = np.subtract if operation == "subtract" else np.add
        i, q = op(ai, bi), op(aq, bq)
    return _finish(i, q if complex else None, full, target, policy, shape, out)


def add(a, b, *, out=None, math: Math | None = None):
    """Add, applying sum policy before any destination conversion."""
    return _binary(a, b, "add", out, math)


def subtract(a, b, *, out=None, math: Math | None = None):
    """Subtract, applying sum policy before any destination conversion."""
    return _binary(a, b, "subtract", out, math)


def multiply(a, b, *, out=None, math: Math | None = None):
    """Multiply, applying product policy before any destination conversion."""
    return _binary(a, b, "multiply", out, math)


def _unary(value, operation, out, math):
    value = _as_operand(value)
    policy = _selected_math(math)
    complex = value.is_complex
    if operation == "absolute" and complex:
        raise TypeError("complex absolute value is unsupported; use mag2()")
    if operation == "mag2":
        if not complex:
            raise TypeError("mag2 requires a complex array")
        full = _product_format(value.format, value.format)
        full = replace(full, wordlength=full.wordlength + 1)
        complex = False
    elif operation == "positive" or (operation == "conjugate" and not complex):
        full = value.format
    else:
        full = _unary_format(value.format, absolute=operation == "absolute")
    preserves_format = operation == "positive" or (operation == "conjugate" and not complex)
    boundary = "full" if preserves_format else policy.sum
    target, out = _prepare(full, boundary, complex, value.shape, (value,), out)
    i, q = _components(value)
    if operation == "mag2":
        i = i * i + q * q
    elif operation == "negative":
        i, q = -i, -q
    elif operation == "absolute":
        i = np.abs(i)
    elif operation == "conjugate" and complex:
        q = -q
    return _finish(i, q if complex else None, full, target, policy, value.shape, out)


def positive(value, *, out=None):
    """Copy with unchanged format; an out conversion uses the active policy."""
    return _unary(value, "positive", out, None)


def negative(value, *, out=None, math: Math | None = None):
    """Negate with declared-range growth, then apply sum policy."""
    return _unary(value, "negative", out, math)


def absolute(value, *, out=None, math: Math | None = None):
    """Real absolute value with declared-range growth, then sum policy."""
    return _unary(value, "absolute", out, math)


def conjugate(value, *, out=None, math: Math | None = None):
    """Conjugate complex inputs with sum policy; copy real inputs unchanged."""
    return _unary(value, "conjugate", out, math)


def mag2(value, *, out=None, math: Math | None = None):
    """Sum exact component squares, applying sum policy once (no product policy)."""
    return _unary(value, "mag2", out, math)


def _array_operand(value, name):
    if not isinstance(value, Array):
        raise TypeError(f"{name} operands must be fixed-point arrays; "
                        "use array() or from_raw() for ndarrays")
    return value


def _reduction_axis(axis, ndim):
    if axis is None:
        return None
    axis = _native_value(axis)
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("axis must be None or one integer; multi-axis reductions are unsupported")
    if not -ndim <= axis < ndim:
        raise np.exceptions.AxisError(axis, ndim)
    return axis % ndim


def _stage_format(format, complex, stage, hint=""):
    """Check a reduction stage's declared format, naming the stage on failure."""
    try:
        return kernels.check_format(format, complex)
    except ValueError as error:
        kind = "signed" if format.signed else "unsigned"
        raise ValueError(f"{error}; {stage} is {format.wordlength}-bit {kind}{hint}") from None


def _accumulation(term, count, policy, complex):
    """Return the format reached before sum policy, and the exact step format.

    Full accumulation adds the conservative ceil(log2(count)) guard bits to the
    term format and has no step. An explicit accumulator bounds every partial
    sum, so it gains no guard bits; each addition is exact in the step format.
    """
    if policy.accumulator == "full":
        guard = max(count - 1, 0).bit_length()
        full = replace(term, wordlength=term.wordlength + guard)
        return _stage_format(full, complex, f"the full sum of {count} terms",
                             ". An explicit accumulator format bounds each step instead"), None
    accumulator = _stage_format(policy.accumulator, complex, "the accumulator format")
    precision = max(accumulator.precision, term.precision)
    kernels.check_shift(accumulator, precision - accumulator.precision)
    kernels.check_shift(term, precision - term.precision)
    return accumulator, _stage_format(_sum_format(accumulator, term), False,
                                      "each exact accumulator step")


def _accumulate(blocks, lanes, parts, term, accumulated, step, policy):
    """Reduce the last axis of int64 term blocks, one total per component.

    Full accumulation is exact, so each block is summed by one NumPy kernel.
    An explicit accumulator quantizes after every addition and therefore visits
    terms one at a time, in order, vectorized across the independent lanes.
    """
    totals = [np.zeros(lanes, dtype=np.int64) for _ in range(parts)]
    if step is None:
        for block in blocks:
            for total, values in zip(totals, block):
                np.add(total, values.sum(axis=-1, dtype=np.int64), out=total)
        return totals
    scratch = kernels._ConversionScratch(lanes, lazy=True)
    work = np.empty(lanes, dtype=np.int64)
    shift = step.precision - term.precision
    for block in blocks:
        for index in range(block[0].shape[-1]):
            for total, values in zip(totals, block):
                kernels.accumulate(total, values[..., index], term_shift=shift, step=step,
                                   accumulator=accumulated, policy=policy, scratch=scratch,
                                   work=work)
    return totals


def _finish_reduction(totals, accumulated, target, policy, shape, out):
    i = totals[0].reshape(shape)
    q = totals[1].reshape(shape) if len(totals) == 2 else None
    return _finish(i, q, accumulated, target, policy, shape, out)


def sum(value, axis=None, *, keepdims=False, out=None, math: Math | None = None):
    """Sum all logical samples or one logical axis.

    Terms are added in logical C order under accumulator policy, then sum
    policy selects the result format, then any destination conversion applies.
    """
    value = _array_operand(value, "sum")
    policy = _selected_math(math)
    axis = _reduction_axis(axis, value.ndim)
    if not isinstance(keepdims, bool):
        raise TypeError("keepdims must be a boolean")
    if axis is None:
        count, lanes = value.size, ()
        shape = (1,) * value.ndim if keepdims else ()
    else:
        count, lanes = value.shape[axis], value.shape[:axis] + value.shape[axis + 1:]
        shape = value.shape[:axis] + ((1,) if keepdims else ()) + value.shape[axis + 1:]
    accumulated, step = _accumulation(value.format, count, policy, value.is_complex)
    target, out = _prepare(accumulated, policy.sum, value.is_complex, shape, (value,), out)
    parts = _components(value)[:1 + value.is_complex]
    block = tuple(part.reshape(-1) if axis is None else np.moveaxis(part, axis, -1)
                  for part in parts)
    totals = _accumulate((block,), lanes, len(parts), value.format, accumulated, step, policy)
    return _finish_reduction(totals, accumulated, target, policy, shape, out)


def _matrix_parts(value, shape):
    """int64 components as (..., rows, columns); a real operand has no imaginary part."""
    i, q = _components(value)
    return i.reshape(shape), q.reshape(shape) if value.is_complex else None


def _product_blocks(a, b, lanes, count, full, term, policy):
    """Yield completed products, quantized by product policy, as lanes + (block,).

    Operands are (..., m, k) and (..., k, n) component pairs. Each yielded tuple
    reuses the previous block's storage; consume it before requesting the next.
    """
    complex = a[1] is not None or b[1] is not None
    # Put the summed axis last and give each operand the other's lane axis.
    a = [None if part is None else part[..., :, None, :] for part in a]
    b = [None if part is None else np.swapaxes(part, -1, -2)[..., None, :, :] for part in b]
    width = max(1, min(count, _BLOCK_ELEMENTS // max(1, reduce(operator.mul, lanes, 1))))
    parts = ()
    for start in range(0, count, width):
        ai, aq, bi, bq = (None if part is None else part[..., start:start + width]
                          for part in a + b)
        shape = lanes + (ai.shape[-1],)
        if not parts or parts[0].shape != shape:
            parts = tuple(np.empty(shape, dtype=np.int64) for _ in range(1 + complex))
            work = np.empty(shape, dtype=np.int64) if complex else None
            scratch = kernels._ConversionScratch(shape, lazy=True)
        if complex:
            zero = np.int64(0)
            kernels.multiply_complex(ai, zero if aq is None else aq, bi, zero if bq is None else bq,
                                     i=parts[0], q=parts[1], work=work)
        else:
            np.multiply(ai, bi, out=parts[0])
        if full != term:
            for part in parts:
                kernels.convert(part, full, term, policy, out=part, scratch=scratch)
        yield parts


def _matmul(a, b, out, math, name):
    a, b = _array_operand(a, name), _array_operand(b, name)
    policy = _selected_math(math)
    if not a.ndim or not b.ndim:
        raise ValueError(f"{name} requires operands with at least one dimension")
    # NumPy's logical rules: a 1-D left operand is a row and a 1-D right operand
    # a column; the added axis is removed from the result. Leading axes broadcast.
    a_shape = a.shape if a.ndim > 1 else (1,) + a.shape
    b_shape = b.shape if b.ndim > 1 else b.shape + (1,)
    count = a_shape[-1]
    if count != b_shape[-2]:
        raise ValueError(f"{name} operands have mismatched lengths {count} and {b_shape[-2]} "
                         "along the summed axis")
    lanes = np.broadcast_shapes(a_shape[:-2], b_shape[:-2]) + (a_shape[-2], b_shape[-1])
    shape = (lanes[:-2] + ((a_shape[-2],) if a.ndim > 1 else ())
             + ((b_shape[-1],) if b.ndim > 1 else ()))
    complex = a.is_complex or b.is_complex
    full = _stage_format(_multiply_format(a.format, b.format, a.is_complex, b.is_complex),
                         complex, "the full product")
    term = (full if policy.product == "full"
            else _stage_format(policy.product, complex, "the product format"))
    kernels.check_conversion(full, term)
    accumulated, step = _accumulation(term, count, policy, complex)
    target, out = _prepare(accumulated, policy.sum, complex, shape, (a, b), out)
    if step is None and term == full:
        # Exact products and exact accumulation need no ordering or boundaries.
        totals = [part for part in kernels.matmul(*_matrix_parts(a, a_shape),
                                                  *_matrix_parts(b, b_shape))
                  if part is not None]
    else:
        blocks = _product_blocks(_matrix_parts(a, a_shape), _matrix_parts(b, b_shape),
                                 lanes, count, full, term, policy)
        totals = _accumulate(blocks, lanes, 1 + complex, term, accumulated, step, policy)
    return _finish_reduction(totals, accumulated, target, policy, shape, out)


def dot(a, b, *, out=None, math: Math | None = None):
    """Inner product of two one-dimensional arrays, without conjugation.

    Product policy applies to each completed product, accumulator policy to
    each ordered addition, and sum policy to the final result.
    """
    for value in (a, b):
        if _array_operand(value, "dot").ndim != 1:
            raise ValueError("dot supports one-dimensional inner products; use matmul for matrices")
    return _matmul(a, b, out, math, "dot")


def matmul(a, b, *, out=None, math: Math | None = None):
    """Matrix product over the last two logical axes, with stacked broadcasting.

    One-dimensional operands follow NumPy: a left vector is a row, a right
    vector a column, and the added axis is removed. The three quantization
    stages match ``dot``.
    """
    return _matmul(a, b, out, math, "matmul")


def _numpy_reduce(value, axis=0, keepdims=False, out=None, where=True, **unsupported):
    # np.add.reduce defaults to axis 0, unlike np.sum.
    if where is not True:
        raise TypeError("fixed-point reductions require where=True")
    if unsupported:
        raise TypeError(f"unsupported fixed-point reduction keywords: {', '.join(unsupported)}")
    return sum(value, axis, keepdims=keepdims, out=out)


def _numpy_sum(value, axis=None, dtype=None, out=None, keepdims=False, **unsupported):
    if dtype is not None:
        unsupported["dtype"] = dtype
    if unsupported:
        raise TypeError(f"unsupported fixed-point reduction keywords: {', '.join(unsupported)}")
    return sum(value, axis, keepdims=keepdims, out=out)


def _numpy_dot(a, b, out=None):
    return dot(a, b, out=out)


_REVERSED_COMPARISON = {"equal": "equal", "not_equal": "not_equal", "less": "greater",
                        "less_equal": "greater_equal", "greater": "less",
                        "greater_equal": "less_equal"}
_SCALAR_COMPARISON = {"equal": operator.eq, "not_equal": operator.ne, "less": operator.lt,
                      "less_equal": operator.le, "greater": operator.gt,
                      "greater_equal": operator.ge}


def _compare_constant(raw, format, constant, precision, operation):
    """Project an exact binary constant onto the array's raw grid, without
    restricting the scalar format or creating a potentially enormous shift.
    """
    shift = format.precision - precision
    if not constant:
        floor, exact = 0, True
    elif shift >= 0:
        if abs(constant).bit_length() + shift > 64:
            # Beyond every supported array range. Only the sign matters.
            above = constant > 0
            result = operation in (("not_equal", "less", "less_equal") if above
                                   else ("not_equal", "greater", "greater_equal"))
            return np.full(raw.shape, result, dtype=bool)
        floor, exact = constant << shift, True
    else:
        floor = constant >> -shift
        # Count trailing zero bits rather than constructing 2**(-shift).
        exact = -shift < (constant & -constant).bit_length()

    below = floor < format.min_int
    above = floor > format.max_int or (floor == format.max_int and not exact)
    if below or above:
        result = operation in (("not_equal", "less", "less_equal") if above
                               else ("not_equal", "greater", "greater_equal"))
        return np.full(raw.shape, result, dtype=bool)
    if operation in ("equal", "not_equal") and not exact:
        return np.full(raw.shape, operation == "not_equal", dtype=bool)
    # x < c uses ceil(c), x <= c uses floor(c); the mirrored relations
    # use the same thresholds. Range checks above keep both within int64.
    threshold = floor + (not exact and operation in ("less", "greater_equal"))
    return getattr(np, operation)(raw, threshold)


def _compare_array_scalar(a, b, operation):
    complex = a.is_complex or isinstance(b, CFixb)
    if complex and operation not in ("equal", "not_equal"):
        raise TypeError("complex ordered comparisons are unsupported")
    ai, aq = _components(a)
    bi, bq = (b.real.int, b.imag.int) if isinstance(b, CFixb) else (b.int, 0)
    if complex:
        result = (_compare_constant(ai, a.format, bi, b.format.precision, "equal") &
                  _compare_constant(aq, a.format, bq, b.format.precision, "equal"))
        return ~result if operation == "not_equal" else result
    return _compare_constant(ai, a.format, bi, b.format.precision, operation)


def _compare(a, b, operation):
    a = a if isinstance(a, Array) else _scalar_operand(a)
    b = b if isinstance(b, Array) else _scalar_operand(b)
    if not isinstance(a, Array):
        if not isinstance(b, Array):
            return np.asarray(_SCALAR_COMPARISON[operation](a, b), dtype=bool)
        a, b, operation = b, a, _REVERSED_COMPARISON[operation]
    if not isinstance(b, Array):
        return np.asarray(_compare_array_scalar(a, b, operation), dtype=bool)
    complex = a.is_complex or b.is_complex
    if complex and operation not in ("equal", "not_equal"):
        raise TypeError("complex ordered comparisons are unsupported")
    ai, aq = _components(a)
    bi, bq = _components(b)
    ap, bp = a.format.precision, b.format.precision
    if complex:
        result = kernels.compare(ai, ap, bi, bp, "equal") & kernels.compare(aq, ap, bq, bp, "equal")
        if operation == "not_equal":
            result = ~result
    else:
        result = kernels.compare(ai, ap, bi, bp, operation)
    return np.asarray(result, dtype=bool)


def equal(a, b):
    return _compare(a, b, "equal")


def not_equal(a, b):
    return _compare(a, b, "not_equal")


def less(a, b):
    return _compare(a, b, "less")


def less_equal(a, b):
    return _compare(a, b, "less_equal")


def greater(a, b):
    return _compare(a, b, "greater")


def greater_equal(a, b):
    return _compare(a, b, "greater_equal")


_COMPARISONS = {np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal}
_UFUNCS = {np.add: add, np.subtract: subtract, np.multiply: multiply,
           np.positive: positive, np.negative: negative, np.absolute: absolute,
           np.conjugate: conjugate, np.equal: equal, np.not_equal: not_equal,
           np.less: less, np.less_equal: less_equal, np.greater: greater,
           np.greater_equal: greater_equal, np.matmul: matmul}
_FUNCTIONS = {np.sum: _numpy_sum, np.dot: _numpy_dot}
