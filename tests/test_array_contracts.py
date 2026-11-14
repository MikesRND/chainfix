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

"""Exact setup/comparison boundaries and Python/NumPy container protocols."""

from decimal import Decimal
from fractions import Fraction
import operator
import pickle

import pytest

np = pytest.importorskip("numpy", exc_type=ImportError)

import chainfix as cf
from chainfix import arrays as cfa
from tests.test_binary_math import fmt, MODES, OVERFLOWS


COMPARISONS = (("equal", operator.eq), ("not_equal", operator.ne),
               ("less", operator.lt), ("less_equal", operator.le),
               ("greater", operator.gt), ("greater_equal", operator.ge))


@pytest.mark.parametrize("name,op", COMPARISONS)
def test_unsupported_numeric_comparisons_raise_in_both_orders(name, op):
    x = cfa.array([0, 0.5, 1], format=fmt())
    for constant in (0.5, Fraction(1, 2), Decimal("0.5"), 0.5 + 0j,
                     [0.5], (0.5,), np.float64(0.5), np.float32(0.5),
                     np.complex128(0.5), np.timedelta64(1, "ns"),
                     np.array(0.5), np.array([0.5])):
        for function in (op, getattr(cfa, name), getattr(np, name)):
            for a, b in ((x, constant), (constant, x)):
                with pytest.raises(TypeError):
                    function(a, b)
    with pytest.raises(TypeError):
        np.where(x == 0.5, 1, 0)


@pytest.mark.parametrize("name", ("add", "equal"))
def test_foreign_ufunc_dispatch_in_both_orders(name):
    class Foreign:
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            return ufunc.__name__, method, tuple(id(value) for value in inputs)

    x = cfa.array([1], format=fmt())
    foreign = Foreign()
    ufunc = getattr(np, name)
    for a, b in ((x, foreign), (foreign, x)):
        assert ufunc(a, b) == (name, "__call__", (id(a), id(b)))
    assert ufunc(x, 1, out=foreign) == (name, "__call__", (id(x), id(1)))


@pytest.mark.parametrize("name,op", COMPARISONS)
def test_wide_comparison_constants_against_fraction_oracle(name, op):
    constants = [-(2**200), -(2**64), -1, 0, 1, 2**64, 2**200, np.uint64(2**64-1)]
    constants += [cf.Fixb.from_raw(n, format=fmt(300, p))
                  for p in (-100, -3, 0, 3, 100)
                  for n in (-(2**200), -65, -64, -63, -1, 0, 1, 63, 64, 65, 2**200)]
    constants += [cf.Ufixb.from_raw(2**150, format=fmt(200, 160, False))]
    for signed, width in ((True, 4), (False, 4), (True, 64), (False, 63)):
        for precision in (-100, -3, 0, 3, 100):
            format = fmt(width, precision, signed)
            raw = sorted({format.min_int, format.min_int+1, 0, 1,
                          format.max_int-1, format.max_int})
            x = cfa.from_raw(np.array(raw, dtype=np.int64), format=format)
            exact = [Fraction(n) * Fraction(2)**-precision for n in raw]
            for constant in constants:
                c = (Fraction(constant.int) * Fraction(2)**-constant.format.precision
                     if isinstance(constant, (cf.Fixb, cf.Ufixb)) else Fraction(int(constant)))
                for function in (op, getattr(cfa, name), getattr(np, name)):
                    np.testing.assert_array_equal(function(x, constant), [op(v, c) for v in exact])
                    # NumPy's scalar-left comparison operators first convert
                    # themselves to a bare 0-d ndarray. Explicit ufunc calls
                    # preserve scalar operands; bare ndarrays remain rejected.
                    if function is op and isinstance(constant, np.generic):
                        continue
                    np.testing.assert_array_equal(function(constant, x), [op(c, v) for v in exact])


@pytest.mark.parametrize("name,op", COMPARISONS)
def test_constants_at_and_between_grid_boundaries(name, op):
    for signed in (False, True):
        format = fmt(4, 2, signed)
        raw = list(range(format.min_int, format.max_int+1))
        x = cfa.from_raw(np.array(raw, dtype=np.int8), format=format)
        for twice_raw in range(2*format.min_int-2, 2*format.max_int+3):
            c = cf.Fixb.from_raw(twice_raw, format=fmt(100, 3))
            expected = [op(Fraction(n, 4), Fraction(twice_raw, 8)) for n in raw]
            np.testing.assert_array_equal(getattr(cfa, name)(x, c), expected)


def test_extreme_precision_constants_zero_dimensional_and_empty():
    for precision, less, greater in ((10**9, [True, True, True], [False, False, False]),
                                     (-10**9, [True, True, False], [False, False, True])):
        x = cfa.from_raw(np.array([-1, 0, 1], dtype=np.int8), format=fmt(8, precision))
        np.testing.assert_array_equal(x < 1, less)
        np.testing.assert_array_equal(x > 1, greater)
        tiny = cf.Fixb.from_raw(1, format=fmt(100, precision))
        np.testing.assert_array_equal(x == tiny, [False, False, True])
        np.testing.assert_array_equal(x < 0, [True, False, False])
        # Exercise huge downshifts of both negative and positive constants.
        z = cfa.array([-1, 0, 1], format=fmt(8, 0))
        np.testing.assert_array_equal(z < tiny, [True, True, False] if precision > 0 else [True]*3)
        np.testing.assert_array_equal(z > -tiny, [False, True, True] if precision > 0 else [True]*3)
    for shape in ((), (2, 0)):
        x = cfa.empty(shape, format=fmt())
        x.raw.fill(0)
        for name, op in COMPARISONS:
            result = getattr(cfa, name)(x, 2**100)
            assert result.shape == shape and result.dtype == bool
            np.testing.assert_array_equal(result, np.full(shape, op(0, 2**100)))


def test_wide_complex_constants_and_scalar_only_comparisons():
    z = cfa.array([0, 0.5, 0.5 + 0.25j], format=fmt())
    c = cf.CFixb.from_raw(i=2, q=1, format=fmt(100, 2))
    for a, b in ((z, c), (c, z)):
        np.testing.assert_array_equal(cfa.equal(a, b), [False, False, True])
        np.testing.assert_array_equal(cfa.not_equal(a, b), [True, True, False])
        with pytest.raises(TypeError, match="complex ordered"):
            cfa.less(a, b)
    np.testing.assert_array_equal(z.real == c, [False]*3)
    huge = cf.CFixb.from_raw(i=2**90, q=0, format=fmt(100, 0))
    np.testing.assert_array_equal(z != huge, [True]*3)
    for name, op in COMPARISONS:
        result = getattr(cfa, name)(2**100, -(2**100))
        assert result.shape == () and result.item() == op(2**100, -(2**100))


@pytest.mark.parametrize("copy", (False, True))
def test_from_raw_rejects_ndarray_subclasses(copy):
    class Subclass(np.ndarray):
        pass

    masked = np.ma.array([100, 1], mask=[True, False], dtype=np.int16)
    with pytest.warns(PendingDeprecationWarning):
        matrix = np.matrix([[1, 2]], dtype=np.int16)
    subclass = np.array([1], dtype=np.int16).view(Subclass)
    for raw in (masked, matrix, subclass):
        with pytest.raises(TypeError, match="plain ndarray"):
            cfa.from_raw(raw, format=fmt(4, 0), copy=copy)
    with pytest.raises(cf.FixedPointOverflowError):
        cfa.from_raw(np.asarray(masked), format=fmt(4, 0), copy=copy)


@pytest.mark.parametrize("complex", (False, True))
def test_iteration_and_zero_width_ellipsis_views(complex):
    scalar = cfa.array(1, format=fmt(), complex=complex)
    for consume in (iter, list, tuple):
        with pytest.raises(TypeError, match="zero-dimensional"):
            consume(scalar)
    x = cfa.array([[1, 2], [3, 4]], format=fmt(), complex=complex)
    rows = list(x)
    assert len(rows) == 2 and all(np.shares_memory(row.raw, x.raw) for row in rows)
    assert list(cfa.empty((0,), format=fmt(), complex=complex)) == []
    for source, keys in ((scalar, (Ellipsis, (Ellipsis,))),
                         (x, ((0, 1, Ellipsis), (Ellipsis, 0, 1), (0, Ellipsis, 1))),
                         (x[0], ((0, Ellipsis), (Ellipsis, 0)))):
        source.raw.flags.writeable = False
        for key in keys:
            view = source[key]
            assert isinstance(view, cfa.Array) and view.shape == ()
            assert view.format == source.format and view.is_complex == complex
            assert np.shares_memory(view.raw, source.raw)
            assert not view.raw.flags.writeable
    assert isinstance(x[0, 1], cf.CFixb if complex else cf.Fixb)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("overflow", OVERFLOWS)
def test_array_setup_matches_scalar_setup_outside_vector_conversion_domain(mode, overflow):
    for complex in (False, True):
        source = cfa.from_raw(np.array([[1, -1], [3, -3]], dtype=np.int64),
                              format=fmt(64, 0), complex=complex)
        policy = cf.Math(rounding=mode, overflow=overflow)
        for target in (fmt(64, 1), fmt(4, 3), fmt(4, -1)):
            scalars = np.empty(source.shape, dtype=object)
            for index in np.ndindex(source.shape):
                scalars[index] = source[index]
            try:
                expected = cfa.array(scalars, format=target, math=policy)
            except cf.FixedPointOverflowError:
                with pytest.raises(cf.FixedPointOverflowError):
                    cfa.array(source, format=target, math=policy)
            else:
                actual = cfa.array(source, format=target, math=policy)
                np.testing.assert_array_equal(actual.raw, expected.raw)
                assert not np.shares_memory(actual.raw, source.raw)
        with pytest.raises(ValueError, match="kernel domain"):
            source.quantize(fmt(64, 1), math=policy)


def test_pickle_restore_checks_structure_but_not_values():
    good = np.array([1], dtype=np.int16)
    cases = [(good.astype(good.dtype.newbyteorder("S")), fmt(), False), (good.astype(float), fmt(), False),
             (good.astype(np.int8), fmt(16), False), (good, fmt(), True),
             (good, fmt(65), False), (good, fmt(), 1),
             (np.zeros((2, 2), dtype=np.int16).T, fmt(), False)]
    for raw, format, complex in cases:
        with pytest.raises((TypeError, ValueError)):
            cfa._restore_array(cfa.Array, raw, format, complex, True)
    # Uninitialized output buffers may contain any bits. Restore checks metadata
    # only; it must not reinterpret them as setup values or scan their range.
    output = cfa.empty((2,), format=fmt(4, 0))
    output.raw.fill(100)
    restored = pickle.loads(pickle.dumps(output))
    assert restored.raw.tolist() == [100, 100]
