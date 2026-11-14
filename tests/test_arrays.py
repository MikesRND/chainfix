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

"""Array construction, storage contracts, broadcasting and NumPy dispatch."""

from copy import copy, deepcopy
from decimal import Decimal
from fractions import Fraction
import operator
import pickle

import pytest

np = pytest.importorskip("numpy", exc_type=ImportError)

import chainfix as cf
from chainfix import arrays as cfa
from tests.test_binary_math import MODES, OVERFLOWS, fmt, oracle_raw


def test_exact_construction_and_explicit_conversion():
    values = [Fraction(1, 3), Fraction(-1, 3), np.nextafter(0.5, 0), True,
              cf.Fixb(0.25), cf.Ufixb(0.5), cf.Fixd(0.5), np.int64(1)]
    result = cfa.array(values, format=fmt(16, 8))
    assert result.raw.tolist() == [85, -85, 128, 256, 64, 128, 128, 256]
    assert result.raw.dtype == np.int16
    assert result.int is result.raw
    assert result.shape == (8,) and result.ndim == 1 and result.size == len(result) == 8
    assert cfa.Array([1], format=fmt()).raw.tolist() == [4]
    assert cfa.array([np.nextafter(0.5, 0)], format=fmt(8, 0)).raw.tolist() == [0]
    # Heterogeneous numeric setup must not round a large integer through float.
    large = cfa.array([2**60 + 1, 0.5], format=fmt(64, 0))
    assert large.raw.tolist() == [2**60 + 1, 0]
    clone = cfa.array(result, format=result.format)
    assert not np.shares_memory(clone.raw, result.raw)
    np.testing.assert_array_equal(clone.raw, result.raw)
    numeric = result.to_numpy()
    assert numeric.dtype == np.float64
    assert not np.shares_memory(numeric, result.raw)
    np.testing.assert_array_equal(numeric, result.raw / 256)
    with pytest.raises(TypeError, match="implicit ndarray"):
        np.asarray(result)
    with pytest.raises(TypeError):
        hash(result)
    with pytest.raises(TypeError, match="truth"):
        bool(result)
    for name in ("format", "_format", "_raw", "_complex"):
        with pytest.raises(AttributeError):
            setattr(result, name, None)
        with pytest.raises(AttributeError):
            delattr(result, name)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("overflow", OVERFLOWS)
def test_setup_conversion_matches_rational_oracle(mode, overflow):
    values = [Fraction(n, 8) for n in range(-40, 41)] + [Fraction(1, 3), Fraction(-1, 3)]
    for precision in (-2, 0, 2):
        target = fmt(4, precision)
        for value in values:
            try:
                expected = oracle_raw(value, target, mode, overflow)
            except ValueError:
                with pytest.raises(cf.FixedPointOverflowError):
                    cfa.array([value], format=target, math=cf.Math(rounding=mode, overflow=overflow))
            else:
                assert cfa.array([value], format=target,
                                 math=cf.Math(rounding=mode, overflow=overflow)).raw[0] == expected


def test_complex_construction_views_and_scalars():
    precise = cf.CFixb(real=Fraction(1, 3), imag=Fraction(-1, 3), format=fmt(60, 58))
    z = cfa.array([precise, 0.5 + 0.25j, 1], format=fmt(60, 58))
    assert z.is_complex and z.shape == (3,) and z.raw.shape == (3, 2)
    assert z[0] == precise
    assert z.real.format == z.imag.format == z.format
    assert np.shares_memory(z.real.raw, z.raw)
    assert np.shares_memory(z.imag.raw, z.raw)
    z.imag.raw[1] = 0
    assert z[1] == 0.5
    assert z.to_numpy().dtype == np.complex128
    with pytest.raises(AttributeError, match="no int"):
        z.int
    with pytest.raises(TypeError):
        cfa.array(z, format=z.format, complex=False)
    with pytest.raises(TypeError):
        cfa.array([1j], format=fmt(), complex=False)
    real = cfa.array([1, 2], format=fmt())
    assert real.real is real
    np.testing.assert_array_equal(real.imag.raw, [0, 0])
    promoted = cfa.array(real, format=real.format, complex=True)
    np.testing.assert_array_equal(promoted.raw, [[4, 0], [8, 0]])
    scalar = cfa.array(0.5 + 0.25j, format=fmt())
    assert scalar.shape == () and scalar.raw.shape == (2,)
    assert scalar.real.shape == scalar.imag.shape == ()
    assert np.shares_memory(scalar.real.raw, scalar.raw)
    assert np.shares_memory(scalar.imag.raw, scalar.raw)
    assert scalar[()] == 0.5 + 0.25j
    with pytest.raises(TypeError):
        len(scalar)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"),
                                 complex(0, float("nan")), Decimal("0.5"), "1"])
def test_invalid_numeric_construction(bad):
    with pytest.raises((TypeError, ValueError)):
        cfa.array([bad], format=fmt())


def test_extreme_floats_and_approximate_export():
    maxfloat = float.fromhex("0x1.fffffffffffffp+1023")
    for precision in (-1024, -1000, 0, 1074):
        target = fmt(64, precision)
        values = [maxfloat, -maxfloat, float.fromhex("0x0.0000000000001p-1022")]
        for mode in MODES:
            result = cfa.array(values, format=target, math=cf.Math(rounding=mode, overflow="wrap"))
            expected = [oracle_raw(Fraction(v), target, mode, "wrap") for v in values]
            assert result.raw.tolist() == expected
    small = cfa.from_raw(np.array([-(2**63), 2**63-1]), format=fmt(64, 10000))
    assert small.to_numpy().tolist() == [0.0, 0.0]
    large = cfa.array([0, 1, -1], format=fmt(8, 0))
    large = cfa.from_raw(large.raw, format=fmt(8, -10000))
    assert large.to_numpy().tolist() == [0, float("inf"), float("-inf")]
    z = cfa.from_raw(np.array([[1, 0], [0, 1]], dtype=np.int8), format=fmt(8, -10000), complex=True)
    assert z.to_numpy().real.tolist() == [float("inf"), 0]
    assert z.to_numpy().imag.tolist() == [0, float("inf")]


def test_raw_sharing_copy_and_readonly():
    raw = np.arange(12, dtype=np.int32).reshape(2, 3, 2)
    raw.flags.writeable = False
    z = cfa.from_raw(raw, format=fmt(8, 2), complex=True)
    assert z.raw is raw and z.raw.dtype == np.int32
    for view in (z, z.real, z.imag, z[0], z.real[0]):
        assert not view.raw.flags.writeable
    assert not z.real.imag.raw.flags.writeable
    copied = cfa.from_raw(raw, format=z.format, complex=True, copy=True)
    assert copied.raw.flags.writeable and copied.raw.flags.c_contiguous
    assert not np.shares_memory(copied.raw, raw)
    for component in (z.real, z.imag):
        wrapped = cfa.from_raw(component.raw, format=z.format)
        assert wrapped.raw is component.raw
    np.testing.assert_array_equal((z * 2).raw, raw * 2)
    for invalid in (raw[::-1], raw.transpose(1, 0, 2)):
        with pytest.raises(ValueError, match="layout"):
            cfa.from_raw(invalid, format=z.format, complex=True)
        materialized = cfa.from_raw(invalid, format=z.format, complex=True, copy=True)
        np.testing.assert_array_equal(materialized.raw, invalid)


@pytest.mark.parametrize("raw,format,complex,error", [
    ([1, 2], fmt(), False, ValueError),
    (np.array([1.0]), fmt(), False, TypeError),
    (np.array([True]), fmt(), False, TypeError),
    (np.array([1], dtype=object), fmt(), False, TypeError),
    (np.array([1], dtype=">i2"), fmt(), False, TypeError),
    (np.array([1], dtype=np.int8), fmt(16), False, ValueError),
    (np.array([128], dtype=np.int16), fmt(8), False, cf.FixedPointOverflowError),
    (np.array([-1], dtype=np.int16), fmt(8, 0, False), False, cf.FixedPointOverflowError),
    (np.array([1, 2, 3]), fmt(), True, ValueError),
    (np.array(1), fmt(), True, ValueError),
    (np.array([1, 2]), fmt(8, 0, False), True, ValueError),
])
def test_raw_validation(raw, format, complex, error):
    with pytest.raises(error):
        cfa.from_raw(raw, format=format, complex=complex)


def test_raw_materialization_and_unaligned_storage():
    assert cfa.from_raw([1, 2], format=fmt(), copy=True).raw.tolist() == [1, 2]
    big_endian = np.array([1, 2], dtype=">i2")
    assert cfa.from_raw(big_endian, format=fmt(), copy=True).raw.dtype.isnative
    raw = np.ndarray((2,), dtype=np.int16, buffer=bytearray(5), offset=1)
    with pytest.raises(ValueError, match="layout"):
        cfa.from_raw(raw, format=fmt())
    assert cfa.from_raw(raw, format=fmt(), copy=True).raw.flags.aligned
    for raw in (np.arange(12, dtype=np.int16)[::3],
                np.arange(12, dtype=np.int16).reshape(3, 4).T,
                np.broadcast_to(np.array([1], dtype=np.int16), (3,))):
        with pytest.raises(ValueError, match="layout"):
            cfa.from_raw(raw, format=fmt())
        np.testing.assert_array_equal(cfa.from_raw(raw, format=fmt(), copy=True).raw, raw)


def test_basic_logical_indexing():
    z = cfa.from_raw(np.arange(24, dtype=np.int16).reshape(3, 4, 2), format=fmt(), complex=True)
    for key in (0, slice(1, 3), (Ellipsis,), (None, Ellipsis), (1, None, Ellipsis)):
        view = z[key]
        assert isinstance(view, cfa.Array)
        assert np.shares_memory(view.raw, z.raw)
    assert z[1, 2] == cf.CFixb.from_raw(i=12, q=13, format=z.format)
    assert z.real[1, 2].int == 12
    assert z[0].real[2].int == 4
    assert z[:0].shape == (0, 4)
    for key in ((slice(None), 0), slice(None, None, -1), slice(None, None, 2)):
        with pytest.raises(ValueError):
            z[key]
    for key in ([0, 1], np.array([0]), True, False, "x"):
        with pytest.raises(TypeError):
            z[key]
    for key in ((0, 0, 0), (Ellipsis, Ellipsis)):
        with pytest.raises(IndexError):
            z[key]
    real = cfa.array(1, format=fmt())
    assert isinstance(real[()], cf.Fixb)
    assert isinstance(real[...], cfa.Array)


@pytest.mark.parametrize("complex", [False, True])
def test_broadcasting_empty_arrays_and_constants(complex):
    x = cfa.array([[1], [2]], format=fmt(), complex=complex)
    y = cfa.array([1, 2, 3], format=fmt(), complex=complex)
    assert (x + y).shape == (2, 3)
    assert (x * y).shape == (2, 3)
    for constant in (2, True, np.int16(2), np.uint64(2), cf.Fixb(2), cf.Ufixb(2), cf.CFixb(2)):
        scalar_constant = int(constant) if isinstance(constant, np.integer) else constant
        for operation in (operator.add, operator.sub, operator.mul):
            for a, b in ((x, constant), (constant, x)):
                result = operation(a, b)
                assert result.shape == (2, 1)
                expected = (operation(x[0, 0], scalar_constant) if a is x
                            else operation(scalar_constant, x[0, 0]))
                assert result[0, 0] == expected
                assert result.format == expected.format
    zero = cfa.empty((2, 0), format=fmt(), complex=complex)
    assert (zero * 2).shape == (2, 0)
    assert (zero == zero).shape == (2, 0)
    assert (+zero).raw.flags.c_contiguous
    with pytest.raises(ValueError):
        x + cfa.array([[1], [2], [3]], format=fmt())


def test_growth_context_overrides_and_rebinding():
    x = cfa.array([-32768, 1], format=fmt(16, 0))
    for grown in (-x, abs(x)):
        assert grown.format == fmt(17, 0)
        assert grown.raw.dtype == np.int32
        assert grown.raw[0] == 32768
    assert (-cfa.array([1], format=fmt(16, 0))).format == fmt(17, 0)
    u = cfa.array([1], format=fmt(16, 0, False))
    assert abs(u).format == u.format
    assert abs(u).raw.dtype == np.uint16
    source = cfa.array([1], format=fmt(8, 0))
    alias = source
    old_raw = source.raw
    source *= 2
    assert source is not alias and source.raw is not old_raw
    assert alias.raw.tolist() == [1] and source.raw.tolist() == [2]
    assert source.format == fmt(11, 0)
    bound = fmt(3, 0)
    with cf.local_binary_context(math=cf.Math(product=bound, sum=bound, overflow="saturate")):
        assert (source * 2).raw.tolist() == [3]
        assert cfa.multiply(source, 2, math=cf.Math()).raw.tolist() == [4]
        assert cfa.multiply(source, 2, math=cf.Math()).format != bound
        assert (+source).format == source.format
        assert cfa.conjugate(source).format == source.format
    assert (source * 2).raw.tolist() == [4]
    assert not hasattr(source, "math")
    with pytest.raises(TypeError):
        cfa.add(source, source, math="wrap")


def test_out_boundaries_reuse_and_double_rounding():
    x = cfa.array([Fraction(5, 16)], format=fmt(8, 4))
    target, destination = fmt(8, 2), fmt(8, 1)
    policy = cf.Math(product=target)
    out = cfa.empty(x.shape, format=destination)
    raw = out.raw
    assert cfa.multiply(x, 1, math=policy, out=out) is out
    assert out.raw is raw and out.raw.tolist() == [0]
    assert cfa.multiply(x, 1, math=policy).quantize(destination).raw.tolist() == [0]
    assert x.quantize(destination).raw.tolist() == [1]
    with cf.local_binary_context(math=policy):
        assert np.multiply(x, 1, out=out) is out
        assert np.multiply(x, 1, out=(out,)) is out
        assert np.multiply(x, 1, out=(None,)).format == target
    for function in (cfa.add, cfa.subtract, cfa.multiply):
        result = function(x, 2)
        dest = cfa.empty(x.shape, format=x.format)
        assert function(x, 2, out=(dest,)) is dest
        np.testing.assert_array_equal(dest.raw, result.quantize(dest.format).raw)


def test_out_rejects_aliases_layout_and_invalid_metadata_before_writes():
    buffer = np.arange(12, dtype=np.int16)
    x = cfa.from_raw(buffer[:8], format=fmt())
    overlap = cfa.from_raw(buffer[4:], format=x.format)
    for destination in (x, overlap):
        snapshot = buffer.copy()
        for multiply in (cfa.multiply, np.multiply):
            with pytest.raises(ValueError, match="overlap"):
                multiply(x, 1, out=destination)
        np.testing.assert_array_equal(buffer, snapshot)
    i = cfa.from_raw(buffer[::2], format=fmt())
    # A strided input with holes and a contiguous output in a hole: disjoint
    # addresses, overlapping memory bounds. The conservative check must reject.
    backing = np.zeros(64, dtype=np.uint8)
    input_raw = np.ndarray((4,), dtype=np.int64, buffer=backing, strides=(16,))
    output_raw = np.ndarray((4,), dtype=np.int8, buffer=backing, offset=8)
    a = cfa.from_raw(input_raw, format=fmt(8, 0))
    out = cfa.from_raw(output_raw, format=fmt(8, 0))
    assert not np.shares_memory(a.raw, out.raw) and np.may_share_memory(a.raw, out.raw)
    with pytest.raises(ValueError, match="overlap"):
        cfa.positive(a, out=out)
    with pytest.raises(ValueError, match="contiguous"):
        cfa.add(i, 1, out=cfa.from_raw(np.zeros((6, 2), dtype=np.int16)[:, 0], format=fmt()))
    readonly = cfa.empty(x.shape, format=x.format)
    readonly.raw.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        cfa.multiply(x, 1, out=readonly)
    for bad in (cfa.empty((2,), format=fmt()), cfa.empty(x.shape, format=fmt(), complex=True)):
        with pytest.raises(ValueError, match="shape and complexity"):
            cfa.multiply(x, 1, out=bad)
    for bad in (buffer, (x, x), (), 3):
        with pytest.raises(TypeError):
            cfa.multiply(x, 1, out=bad)


@pytest.mark.parametrize("function", [np.add, np.subtract, np.multiply])
def test_ufunc_matches_operator_and_explicit_policy(function):
    x = cfa.array([0.5 + 0.25j, -0.25 + 0.5j], format=fmt(16, 14))
    y = cfa.array(0.5 - 0.25j, format=x.format)
    policy = cf.Math(product=fmt(16, 12), sum=fmt(17, 10), rounding="floor", overflow="wrap")
    with cf.local_binary_context(math=policy):
        actual = function(x, y, where=True)
        expected = getattr(cfa, function.__name__)(x, y, math=policy)
    assert actual.format == expected.format
    np.testing.assert_array_equal(actual.raw, expected.raw)


def test_unary_and_comparison_ufuncs():
    x = cfa.array([-1, 0, 1], format=fmt())
    z = cfa.array([-1 + 1j, 0, 1 - 1j], format=fmt())
    np.testing.assert_array_equal(z.conjugate().raw, cfa.conjugate(z).raw)
    for ufunc, explicit in ((np.positive, cfa.positive), (np.negative, cfa.negative),
                            (np.absolute, cfa.absolute), (np.conjugate, cfa.conjugate)):
        for value in (x, z):
            if value.is_complex and ufunc is np.absolute:
                with pytest.raises(TypeError):
                    ufunc(value)
            else:
                np.testing.assert_array_equal(ufunc(value).raw, explicit(value).raw)
    for ufunc in (np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal):
        expected = ufunc(np.array([-1, 0, 1]), 0)
        np.testing.assert_array_equal(ufunc(x, 0), expected)
    np.testing.assert_array_equal(z == x, [False, True, False])
    np.testing.assert_array_equal(z != x, [True, False, True])
    for call in (lambda: z < x, lambda: x >= z, lambda: z <= 0, lambda: z > 0):
        with pytest.raises(TypeError, match="complex ordered"):
            call()
    scalar = cfa.array(1, format=fmt())
    assert isinstance(scalar == 1, np.ndarray) and (scalar == 1).shape == ()


def test_rejected_numpy_dispatch_and_operands():
    x = cfa.array([1, 2], format=fmt())
    for call in (lambda: np.sin(x), lambda: np.multiply.reduce(x), lambda: np.multiply.outer(x, x),
                 lambda: np.add.at(x, 0, 1), lambda: np.add.accumulate(x), lambda: np.prod(x),
                 lambda: np.vdot(x, x), lambda: np.concatenate([x, x]),
                 lambda: np.add(x, x, where=False), lambda: np.add(x, x, where=np.array([True, True])),
                 lambda: np.add(x, x, dtype=np.int64), lambda: np.add(x, x, casting="unsafe"),
                 lambda: np.add(x, x, order="C"), lambda: np.equal(x, x, out=np.empty(2, dtype=bool)),
                 lambda: np.equal(x, x, out=x)):
        with pytest.raises(TypeError):
            call()
    for other in (np.array([1, 2]), [1, 2], 1.5, 1j, Fraction(1, 2), Decimal("1"), object()):
        for operation in (operator.add, operator.sub, operator.mul):
            with pytest.raises(TypeError):
                operation(x, other)
            with pytest.raises(TypeError):
                operation(other, x)


@pytest.mark.parametrize("method", ("__add__", "__radd__", "__sub__", "__rsub__",
                                    "__mul__", "__rmul__", "__eq__", "__ne__",
                                    "__lt__", "__le__", "__gt__", "__ge__"))
def test_operators_return_notimplemented_for_unsupported_types(method):
    z = cfa.array([1j, 1], format=fmt())
    for other in (None, "raise", object()):
        assert getattr(z, method)(other) is NotImplemented
    for other in ([1, 2], 1.5, 1j, Fraction(1, 2),
                  Decimal("1"), np.zeros(2), np.float64(1.5), np.timedelta64(1, "D")):
        if method in ("__eq__", "__ne__"):
            with pytest.raises(TypeError):
                getattr(z, method)(other)
        else:
            assert getattr(z, method)(other) is NotImplemented


def test_unsupported_equality_and_sentinel_membership():
    z = cfa.array([1j, 1], format=fmt())
    for other in (None, "raise", object()):
        assert (z == other) is False
        assert (other == z) is False
        assert (z != other) is True
        assert (other != z) is True
    assert "raise" not in [z]
    assert z not in [None, "raise"]
    # Supported comparisons remain elementwise; membership between distinct
    # arrays does not turn into scalar equality as a side effect of dispatch.
    with pytest.raises(ValueError, match="ambiguous"):
        z in [cfa.array([1j, 1], format=fmt())]


def test_foreign_reflected_arithmetic_and_comparisons():
    class Foreign:
        def __radd__(self, other):
            return "radd", id(other)

        def __rsub__(self, other):
            return "rsub", id(other)

        def __rmul__(self, other):
            return "rmul", id(other)

        def __eq__(self, other):
            return "eq", id(other)

        def __ne__(self, other):
            return "ne", id(other)

        def __lt__(self, other):
            return "lt", id(other)

        def __le__(self, other):
            return "le", id(other)

        def __gt__(self, other):
            return "gt", id(other)

        def __ge__(self, other):
            return "ge", id(other)

    z = cfa.array([1j, 1], format=fmt())
    foreign = Foreign()
    for operation, reflected in ((operator.add, "radd"), (operator.sub, "rsub"),
                                 (operator.mul, "rmul"), (operator.eq, "eq"),
                                 (operator.ne, "ne"), (operator.lt, "gt"),
                                 (operator.le, "ge"), (operator.gt, "lt"),
                                 (operator.ge, "le")):
        assert operation(z, foreign) == (reflected, id(z))


@pytest.mark.parametrize("complex", (False, True))
@pytest.mark.parametrize("operation", (operator.add, operator.sub, operator.mul,
                                       operator.eq, operator.ne, operator.lt,
                                       operator.le, operator.gt, operator.ge))
def test_ndarray_operands_still_raise_in_both_directions(complex, operation):
    z = cfa.array([1, 2], format=fmt(), complex=complex)
    for nd in (np.zeros(2), np.zeros(2, dtype=object), np.array(1)):
        with pytest.raises(TypeError, match="operands must be"):
            operation(z, nd)
        with pytest.raises(TypeError, match="operands must be"):
            operation(nd, z)


@pytest.mark.parametrize("name", ("add", "subtract", "multiply", "equal", "not_equal",
                                  "less", "less_equal", "greater", "greater_equal"))
def test_explicit_and_ufunc_calls_keep_rejecting_unsupported_operands(name):
    z = cfa.array([1j, 1], format=fmt())
    for function in (getattr(cfa, name), getattr(np, name)):
        for other in (None, "raise", object(), 1.5):
            for a, b in ((z, other), (other, z)):
                with pytest.raises(TypeError):
                    function(a, b)


def test_supported_operand_validation_errors_are_not_swallowed(monkeypatch):
    z = cfa.array([1j, 1], format=fmt())
    with pytest.raises(TypeError, match="complex ordered"):
        z.__lt__(1)
    with pytest.raises(ValueError, match="kernel domain"):
        z.__add__(2**100)

    def invalid_supported_operation(*args):
        raise TypeError("supported operand validation failed")

    monkeypatch.setattr(cfa, "add", invalid_supported_operation)
    monkeypatch.setattr(cfa, "equal", invalid_supported_operation)
    for call in (lambda: z + 1, lambda: 1 + z, lambda: z == 1):
        with pytest.raises(TypeError, match="supported operand validation failed"):
            call()


@pytest.fixture(params=("real", "unsigned", "complex", "real_view", "imag_view",
                        "scalar", "complex_scalar", "empty", "complex_empty"))
def protocol_array(request):
    kind = request.param
    if kind == "unsigned":
        raw = np.array([2**63-1, 2**60+1], dtype=np.uint64)
        return cfa.from_raw(raw, format=fmt(63, 7, False))
    raw = np.array([[2**60+1, -(2**63)], [7, -3]], dtype=np.int64)
    z = cfa.from_raw(raw, format=fmt(64, 7), complex=True)
    if kind == "complex":
        return z
    if kind == "real_view":
        return z.real
    if kind == "imag_view":
        return z.imag
    if kind == "complex_scalar":
        return cfa.from_raw(raw[0], format=z.format, complex=True)
    if kind == "scalar":
        return cfa.from_raw(raw[:1, :1].reshape(()), format=z.format)
    if kind.endswith("empty"):
        return cfa.empty((2, 0), format=fmt(5, -2), complex=kind == "complex_empty")
    # The wider dtype must survive copying, including for a subword format.
    return cfa.from_raw(np.array([7, -3], dtype=np.int32), format=fmt(5, -2))


def assert_same_storage_values(restored, source):
    assert restored is not source
    assert restored.format == source.format
    assert restored.is_complex == source.is_complex
    assert restored.shape == source.shape
    assert restored.raw.dtype == source.raw.dtype
    assert restored.raw.flags.writeable == source.raw.flags.writeable
    np.testing.assert_array_equal(restored.raw, source.raw)
    with pytest.raises(AttributeError):
        restored._format = fmt()


@pytest.mark.parametrize("readonly", (False, True))
def test_shallow_and_deep_array_copies(protocol_array, readonly):
    source = protocol_array
    source.raw.flags.writeable = not readonly
    with cf.local_binary_context(wordlength=4, precision=0, math=cf.Math(overflow="wrap")):
        shallow, deep = copy(source), deepcopy(source)
    for restored in (shallow, deep):
        assert_same_storage_values(restored, source)
        assert restored.format is source.format
    assert shallow.raw is source.raw
    assert deep.raw is not source.raw
    assert deep.raw.flags.c_contiguous
    assert not np.shares_memory(deep.raw, source.raw)
    if source.size:
        original = source.raw.flat[0]
        if readonly:
            for restored in (shallow, deep):
                with pytest.raises(ValueError, match="read-only"):
                    restored.raw.flat[0] = 0
        else:
            deep.raw.flat[0] = 0
            assert source.raw.flat[0] == original
            shallow.raw.flat[0] = 0
            assert source.raw.flat[0] == 0


@pytest.mark.parametrize("readonly", (False, True))
def test_deepcopy_memo_preserves_repeated_array_and_raw_references(readonly):
    z = cfa.array([1j, 1], format=fmt()).imag
    z.raw.flags.writeable = not readonly
    # Include raw first so Array must honor the copy already in the memo.
    raw, first, repeated, shallow = deepcopy([z.raw, z, z, copy(z)])
    assert first is repeated and first is not shallow
    assert raw is first.raw is shallow.raw
    assert not np.shares_memory(raw, z.raw)
    assert raw.flags.c_contiguous and raw.flags.writeable == (not readonly)
    np.testing.assert_array_equal(raw, z.raw)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("readonly", (False, True))
def test_array_pickle_restores_exact_storage(protocol_array, readonly, protocol):
    source = protocol_array
    source.raw.flags.writeable = not readonly
    with cf.local_binary_context(wordlength=4, precision=0, math=cf.Math(overflow="wrap")):
        restored = pickle.loads(pickle.dumps(source, protocol=protocol))
    assert_same_storage_values(restored, source)
    assert restored.raw.flags.c_contiguous
    assert not np.shares_memory(restored.raw, source.raw)


def test_array_copy_and_pickle_bypass_quantization_and_value_scan(monkeypatch):
    source = cfa.array([0.5 + 0.25j], format=fmt(16, 14)).imag

    def reject(*args, **kwargs):
        raise AssertionError("copy/pickle must restore storage directly")

    for name in ("_from_values", "from_raw", "_selected_math"):
        monkeypatch.setattr(cfa, name, reject)
    monkeypatch.setattr(cfa.kernels, "convert", reject)
    for restored in (copy(source), deepcopy(source), pickle.loads(pickle.dumps(source))):
        assert_same_storage_values(restored, source)


def test_structural_validation():
    for kwargs in ({"format": fmt(65)}, {"format": fmt(64, 0, False)},
                   {"format": cf.Format(base=10, signed=True, wordlength=8, precision=2)},
                   {"format": fmt(8, 0, False), "complex": True}, {"format": "q14"}):
        with pytest.raises((TypeError, ValueError)):
            cfa.empty(1, **kwargs)
    for shape in (True, (True,), (1.5,), (-1,)):
        with pytest.raises((TypeError, ValueError)):
            cfa.empty(shape, format=fmt())
    for call in (lambda: cfa.array([1], format=fmt(), complex=1),
                 lambda: cfa.empty(1, format=fmt(), complex=None),
                 lambda: cfa.from_raw(np.array([1]), format=fmt(), copy=None)):
        with pytest.raises(TypeError):
            call()
