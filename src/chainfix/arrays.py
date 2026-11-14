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
"""

from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import replace

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
           "positive", "negative", "absolute", "conjugate", "mag2", "equal",
           "not_equal", "less", "less_equal", "greater", "greater_equal"]


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
        return Array._from_storage(np.array(raw, dtype=kernels.storage_dtype(format),
                                          copy=True, order="C"), format, self.is_complex)

    def conjugate(self, *, math: Math | None = None):
        return conjugate(self, math=math)

    def mag2(self, *, math: Math | None = None):
        return mag2(self, math=math)

    def __len__(self):
        if not self.ndim:
            raise TypeError("len() of a zero-dimensional array")
        return self.shape[0]

    def __bool__(self):
        raise TypeError("fixed-point array truth is ambiguous; compare elements explicitly")

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        if not self.ndim and len(keys) == 1 and keys[0] is Ellipsis:
            return Array._from_storage(self.raw.view(), self.format, self.is_complex)
        for item in keys:
            if item is None or item is Ellipsis:
                continue
            if isinstance(item, slice):
                if item.step not in (None, 1):
                    raise ValueError("only unit-step basic slicing is supported")
            elif isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise TypeError("only basic integer/slice indexing is supported")
        ellipses = sum(item is Ellipsis for item in keys)
        count = sum(item is not None and item is not Ellipsis for item in keys)
        if ellipses > 1 or count > self.ndim:
            raise IndexError("too many indices or ellipses for logical array shape")
        expanded = []
        for item in keys:
            if item is Ellipsis:
                expanded.extend([slice(None)] * (self.ndim - count))
            else:
                expanded.append(item)
        if not ellipses:
            expanded.extend([slice(None)] * (self.ndim - count))
        if self.is_complex:
            expanded.append(slice(None))
        raw = self.raw[tuple(expanded)]
        if self.is_complex and raw.shape == (2,):
            return CFixb.from_raw(i=int(raw[0]), q=int(raw[1]), format=self.format)
        if not self.is_complex and np.isscalar(raw):
            return (Fixb if self.format.signed else Ufixb).from_raw(int(raw), format=self.format)
        if not _supported_layout(raw, self.is_complex):
            raise ValueError("indexing produced an unsupported layout")
        return Array._from_storage(raw, self.format, self.is_complex)

    def __array__(self, dtype=None, copy=None):
        raise TypeError("implicit ndarray conversion discards format; use raw or to_numpy()")

    def __array_function__(self, func, types, args, kwargs):
        raise TypeError(f"NumPy function {func.__name__} is unsupported for fixed-point arrays")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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

    def __pos__(self):
        return positive(self)

    def __neg__(self):
        return negative(self)

    def __abs__(self):
        return absolute(self)

    def __eq__(self, other):
        return equal(self, other) if _supports_operand(other) else NotImplemented

    def __ne__(self, other):
        return not_equal(self, other) if _supports_operand(other) else NotImplemented

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
    result = cls._from_storage(raw, format, complex)
    result.raw.flags.writeable = writeable
    return result


def _native_value(value):
    return value.item() if isinstance(value, np.generic) else value


def _supports_operand(value):
    # Only dispatch on type here. Supported operands still pass through all
    # format/domain validation, whose errors must not be swallowed by operators.
    return isinstance(_native_value(value), (Array, Fixb, Ufixb, CFixb, int))


def _from_values(values, format, complex, math):
    _complex_flag(complex, optional=True)
    policy = _selected_math(math)
    if isinstance(values, Array):
        if complex is False and values.is_complex:
            raise TypeError("complex input cannot construct a real array")
        result = values.quantize(format, math=policy)
        if complex and not values.is_complex:
            kernels.check_format(format, True)
            raw = np.zeros(values.shape + (2,), dtype=kernels.storage_dtype(format))
            raw[..., 0] = result.raw
            return Array._from_storage(raw, format, True)
        return result
    # Object storage is confined to setup: preserve heterogeneous exact inputs
    # rather than allowing NumPy to promote large integers to float first.
    values = np.asarray(values, dtype=object)
    has_complex = any(isinstance(v, (builtins.complex, np.complexfloating, CFixb)) for v in values.flat)
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


def _as_operand(value):
    if isinstance(value, Array):
        return value
    value = _native_value(value)
    if isinstance(value, CFixb):
        kernels.check_format(value.format, True)
        return Array._from_storage(np.array([value.real.int, value.imag.int], dtype=np.int64),
                                   value.format, True)
    value = _operand(value)
    if value is NotImplemented:
        raise TypeError("operands must be fixed-point arrays/scalars or integer constants; "
                        "use array() or from_raw() for ndarrays")
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


def _finish(i, q, full, target, policy, shape, out):
    complex = q is not None
    destination = target if out is None else out.format
    if out is None:
        out = empty(shape, format=target, complex=complex)
    i = kernels.convert(i, full, target, policy)
    if complex:
        q = kernels.convert(q, full, target, policy)
    # Preserve the operation boundary even when out has a different format.
    if destination != target:
        i = kernels.convert(i, target, destination, policy)
        if complex:
            q = kernels.convert(q, target, destination, policy)
    if complex:
        out.raw[..., 0], out.raw[..., 1] = i, q
    else:
        out.raw[...] = i
    return out


def _components(value):
    raw = value.raw.astype(np.int64, copy=False)
    return (raw[..., 0], raw[..., 1]) if value.is_complex else (raw, np.int64(0))


def _binary(a, b, operation, out, math):
    a, b = _as_operand(a), _as_operand(b)
    policy = _selected_math(math)
    complex = a.is_complex or b.is_complex
    shape = np.broadcast_shapes(a.shape, b.shape)
    if operation == "multiply":
        full = _product_format(a.format, b.format)
        if a.is_complex and b.is_complex:
            full = replace(full, wordlength=full.wordlength + 1)
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
        i, q = ai * bi - aq * bq, ai * bq + aq * bi
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


def _compare(a, b, operation):
    a, b = _as_operand(a), _as_operand(b)
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
           np.greater_equal: greater_equal}
