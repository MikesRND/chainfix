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

"""Ordered reductions checked against an independent three-stage Fraction model."""

from dataclasses import replace
from fractions import Fraction
import itertools
import random

import pytest

np = pytest.importorskip("numpy", exc_type=ImportError)

import chainfix as cf
from chainfix import arrays as cfa
from tests.test_array_math import raw_array
from tests.test_binary_math import MODES, OVERFLOWS, fmt, oracle_raw


def val(raw, format):
    return Fraction(int(raw)) / Fraction(2) ** format.precision


def guard(count):
    """Smallest g with 2**g >= count, by search rather than bit tricks."""
    bits = 0
    while 2 ** bits < count:
        bits += 1
    return bits


def model(terms, full, *, rounding="nearest_even", overflow="raise", product="full",
          accumulator="full", sum="full"):
    """Reduce exact real terms in order through product, accumulator, and sum stages.

    ``terms`` are completed products (or plain samples, with ``full`` their
    format). Raises ValueError where the raise policy would detect overflow.
    """
    term = full if product == "full" else product
    values = [val(oracle_raw(v, term, rounding, overflow), term) for v in terms]
    total = Fraction(0)
    if accumulator == "full":
        for v in values:
            total += v
        reached = fmt(term.wordlength + guard(len(values)), term.precision, term.signed)
    else:
        for v in values:
            total = val(oracle_raw(total + v, accumulator, rounding, overflow), accumulator)
        reached = accumulator
    target = reached if sum == "full" else sum
    return oracle_raw(total, target, rounding, overflow), target


def policy_of(spec):
    return cf.Math(**spec)


def expect(call, lanes, full, spec, complex=False):
    """Compare a reduction with the model for every lane of exact terms.

    ``lanes`` is a list of term lists; complex terms are (real, imag) pairs and
    each component is reduced independently.
    """
    try:
        if complex:
            expected = [[model([t[part] for t in terms], full, **spec)[0] for part in (0, 1)]
                        for terms in lanes]
        else:
            expected = [model(terms, full, **spec)[0] for terms in lanes]
        target = model([], full, **spec)[1] if not lanes else model(
            [t[0] for t in lanes[0]] if complex else lanes[0], full, **spec)[1]
    except ValueError:
        with pytest.raises(cf.FixedPointOverflowError):
            call()
        return None
    result = call()
    assert isinstance(result, cfa.Array) and result.is_complex == complex
    assert result.format == target
    assert result.raw.reshape(-1, 2).tolist() == expected if complex else \
        result.raw.ravel().tolist() == expected
    return result


def products(a, af, b, bf):
    return [val(x, af) * val(y, bf) for x, y in zip(a, b)]


def complex_products(a, af, b, bf):
    """Completed complex products of raw (i, q) pairs; a real operand has q == 0."""
    out = []
    for (ar, ai), (br, bi) in zip(a, b):
        ar, ai, br, bi = val(ar, af), val(ai, af), val(br, bf), val(bi, bf)
        out.append((ar * br - ai * bi, ar * bi + ai * br))
    return out


SPECS = [
    {},
    {"sum": fmt(4, 1)},
    {"accumulator": fmt(4, 1)},
    {"accumulator": fmt(5, 0), "sum": fmt(3, -1)},
    {"accumulator": fmt(4, 3, False)},
]


# --- formats -----------------------------------------------------------------

@pytest.mark.parametrize("count,bits", [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 3),
                                        (8, 3), (9, 4), (16, 4), (17, 5)])
def test_full_sum_guard_bits_for_zero_one_and_many_terms(count, bits):
    for format, complex in ((fmt(8, 2), False), (fmt(8, 2, False), False), (fmt(8, 2), True)):
        raw = np.ones((count, 2) if complex else count, dtype=np.int64)
        result = cfa.sum(cfa.from_raw(raw, format=format, complex=complex))
        assert result.format == replace(format, wordlength=8 + bits)
        assert result.shape == () and result.is_complex == complex
        assert result.raw.tolist() == ([count, count] if complex else count)
        # Guard bits follow the declared count, so extreme terms always fit.
        low = np.full(raw.shape, format.min_int, dtype=np.int64)
        assert cfa.sum(cfa.from_raw(low, format=format, complex=complex)).raw.ravel().tolist() == \
            [count * format.min_int] * (2 if complex else 1)
        high = np.full(raw.shape, format.max_int, dtype=np.int64)
        assert cfa.sum(cfa.from_raw(high, format=format, complex=complex)).raw.ravel().tolist() == \
            [count * format.max_int] * (2 if complex else 1)


def test_dot_formats_start_from_product_format_and_one_term_adds_no_sum_guard():
    q, u = fmt(8, 4), fmt(6, 1, False)
    for count, bits in ((1, 0), (2, 1), (5, 3)):
        real = raw_array([3] * count, q)
        unsigned = raw_array([3] * count, u)
        z = raw_array([[3, -2]] * count, q, complex=True)
        assert cfa.dot(real, real).format == fmt(16 + bits, 8)
        assert cfa.dot(real, unsigned).format == fmt(14 + bits, 5)
        assert cfa.dot(unsigned, unsigned).format == fmt(12 + bits, 2, False)
        # Complex by complex carries the complex-product guard bit; complex by
        # real follows the real product format, as elementary multiplication does.
        assert cfa.dot(z, z).format == fmt(17 + bits, 8)
        assert cfa.dot(z, real).format == cfa.dot(real, z).format == fmt(16 + bits, 8)
        assert cfa.dot(z, unsigned).format == fmt(14 + bits, 5)
        assert cfa.dot(z, z).is_complex and cfa.dot(z, real).is_complex
    one = raw_array([[3, -2]], q, complex=True)
    assert cfa.dot(one, one).format == (one * one).format
    assert cfa.dot(one, one).raw.tolist() == (one * one).raw[0].tolist() == [5, -12]


def test_explicit_accumulator_bounds_the_result_without_guard_bits():
    x = raw_array([1, 2, 3, 4, 5], fmt(8, 2))
    acc, out = fmt(12, 1), fmt(6, 0)
    assert cfa.sum(x, math=cf.Math(accumulator=acc)).format == acc
    assert cfa.sum(x, math=cf.Math(accumulator=acc, sum=out)).format == out
    assert cfa.sum(x, math=cf.Math(sum=out)).format == out
    assert cfa.dot(x, x, math=cf.Math(accumulator=acc)).format == acc
    assert cfa.dot(x, x, math=cf.Math(product=fmt(10, 2))).format == fmt(13, 2)
    assert cfa.dot(x, x, math=cf.Math(product=fmt(10, 2), accumulator=acc, sum=out)).format == out


# --- sum values --------------------------------------------------------------

def test_per_add_saturation_and_final_saturation_are_distinct_policies():
    s8 = fmt(8, 0)
    x = raw_array([100, 100, -100], s8)
    per_add = cfa.sum(x, math=cf.Math(accumulator=s8, overflow="saturate"))
    final = cfa.sum(x, math=cf.Math(sum=s8, overflow="saturate"))
    assert per_add.raw.tolist() == 27 and per_add.format == s8
    assert final.raw.tolist() == 100 and final.format == s8
    assert cfa.sum(x, math=cf.Math(accumulator=s8, overflow="wrap")).raw.tolist() == 100
    assert cfa.sum(x, math=cf.Math(sum=s8)).raw.tolist() == 100
    with pytest.raises(cf.FixedPointOverflowError):
        cfa.sum(x, math=cf.Math(accumulator=s8))
    # The same distinction through a dot product with unit coefficients.
    ones = raw_array([1, 1, 1], fmt(2, 0))
    assert cfa.dot(x, ones, math=cf.Math(product=s8, accumulator=s8,
                                         overflow="saturate")).raw.tolist() == 27
    assert cfa.dot(x, ones, math=cf.Math(sum=s8, overflow="saturate")).raw.tolist() == 100


def test_accumulator_rounding_follows_logical_order():
    half = fmt(8, 1)
    acc = cf.Math(accumulator=fmt(8, 0))
    forward, backward = raw_array([1, 1, 2], half), raw_array([2, 1, 1], half)
    # 0.5, 0.5, 1: ties to even give 0, 0, 1. Reversed: 1, 1.5 -> 2, 2.5 -> 2.
    assert cfa.sum(forward, math=acc).raw.tolist() == 1
    assert cfa.sum(backward, math=acc).raw.tolist() == 2
    assert cfa.sum(forward).raw.tolist() == cfa.sum(backward).raw.tolist() == 4
    assert cfa.sum(forward, math=cf.Math(sum=fmt(8, 0))).raw.tolist() == 2
    # Axis=None visits logical samples in C order, not storage or column order.
    grid = raw_array([[1, 1], [2, 0]], half)
    assert cfa.sum(grid, math=acc).raw.tolist() == 1
    assert cfa.sum(raw_array([[2, 0], [1, 1]], half), math=acc).raw.tolist() == 2
    assert cfa.sum(grid, axis=0, math=acc).raw.tolist() == [1, 0]
    assert cfa.sum(grid, axis=1, math=acc).raw.tolist() == [0, 1]


@pytest.mark.parametrize("mode,overflow", list(itertools.product(MODES, OVERFLOWS)))
def test_exhaustive_three_term_sums_for_small_formats(mode, overflow):
    for source in (fmt(3, 2), fmt(3, -1), fmt(2, 1, False), fmt(3, 3, False)):
        raws = range(source.min_int, source.max_int + 1)
        rows = list(itertools.product(raws, repeat=3))
        x = raw_array(rows, source)
        lanes = [[val(r, source) for r in row] for row in rows]
        for spec in SPECS:
            spec = dict(spec, rounding=mode, overflow=overflow)
            # One lane at a time isolates raise: any overflowing lane fails the call.
            for index in (0, len(rows) // 2, len(rows) - 1):
                expect(lambda: cfa.sum(x[index], math=policy_of(spec)), [lanes[index]], source, spec)
            if overflow != "raise":
                expect(lambda: cfa.sum(x, axis=1, math=policy_of(spec)), lanes, source, spec)


@pytest.mark.parametrize("mode", MODES)
def test_complex_sums_reduce_components_independently(mode):
    rng = random.Random(7)
    source = fmt(6, 3)
    raw = [[[rng.randint(-32, 31), rng.randint(-32, 31)] for _ in range(5)] for _ in range(4)]
    z = raw_array(raw, source, complex=True)
    for spec in ({}, {"sum": fmt(6, 1)}, {"accumulator": fmt(7, 1)},
                 {"accumulator": fmt(8, 2), "sum": fmt(5, 0)}):
        spec = dict(spec, rounding=mode, overflow="saturate")
        rows = [[(val(i, source), val(q, source)) for i, q in row] for row in raw]
        columns = [list(column) for column in zip(*rows)]
        expect(lambda: cfa.sum(z, axis=1, math=policy_of(spec)), rows, source, spec, complex=True)
        expect(lambda: cfa.sum(z, axis=0, math=policy_of(spec)), columns, source, spec, complex=True)
        flat = [[term for row in rows for term in row]]
        result = expect(lambda: cfa.sum(z, math=policy_of(spec)), flat, source, spec, complex=True)
        assert result.shape == () and result.raw.shape == (2,)
    for bad in (cf.Math(accumulator=fmt(8, 2, False)), cf.Math(sum=fmt(8, 2, False))):
        with pytest.raises(ValueError, match="signed"):
            cfa.sum(z, math=bad)


def test_axes_keepdims_shapes_and_storage():
    source = fmt(8, 0)
    values = np.arange(24, dtype=np.int64).reshape(2, 3, 4) - 12
    x = cfa.from_raw(values.astype(np.int8), format=source)
    for axis in (0, 1, 2, -1, -3, np.int64(1)):
        for keepdims in (False, True):
            result = cfa.sum(x, axis, keepdims=keepdims)
            np.testing.assert_array_equal(result.raw, values.sum(axis=int(axis), keepdims=keepdims))
            assert result.format == fmt(8 + guard(values.shape[int(axis)]), 0)
            assert result.raw.dtype == np.int16
    for keepdims, shape in ((False, ()), (True, (1, 1, 1))):
        result = x.sum(keepdims=keepdims)
        assert result.shape == shape and result.raw.ravel().tolist() == [values.sum()]
        assert result.format == fmt(13, 0)
    # A full reduction is a zero-dimensional array; [()] extracts the scalar.
    scalar = cfa.sum(x)[()]
    assert isinstance(scalar, cf.Fixb) and scalar.format == fmt(13, 0) and scalar == values.sum()
    zero_d = cfa.sum(cfa.sum(x))
    assert zero_d.format == fmt(13, 0) and zero_d.raw.tolist() == values.sum()
    # As in NumPy, axis 0 or -1 names a zero-dimensional array's single sample.
    for axis in (0, -1):
        again = cfa.sum(zero_d, axis=axis, keepdims=True)
        assert again.shape == () and again.format == fmt(13, 0) and again.raw.tolist() == values.sum()
    for axis in (1, -2):
        with pytest.raises(np.exceptions.AxisError):
            cfa.sum(zero_d, axis=axis)
    # An exact sum reads stored samples directly, including unsigned 64-bit storage.
    unsigned = cfa.from_raw(np.array([[2 ** 60, 2 ** 60 - 1], [1, 0]], dtype=np.uint64),
                            format=fmt(61, 0, False))
    assert cfa.sum(unsigned).raw.tolist() == 2 ** 61 and cfa.sum(unsigned).format == fmt(63, 0, False)
    assert cfa.sum(unsigned, axis=1).raw.tolist() == [2 ** 61 - 1, 1]
    assert cfa.sum(unsigned, math=cf.Math(accumulator=fmt(62, 0, False))).raw.tolist() == 2 ** 61

    # Component views, read-only inputs, and wider storage are all read in place.
    z = raw_array([[[1, 10], [2, 20]], [[3, 30], [4, 40]]], source, complex=True)
    assert cfa.sum(z.imag, axis=0).raw.tolist() == [40, 60]
    assert cfa.sum(z.real).raw.tolist() == 10
    wide = np.array([5, -6, 7], dtype=np.int64)
    wide.flags.writeable = False
    assert cfa.sum(cfa.from_raw(wide, format=source)).raw.tolist() == 6
    assert cfa.sum(cfa.from_raw(wide, format=source), math=cf.Math(accumulator=source)).raw.dtype == np.int8


def test_empty_reductions_return_formatted_zero():
    source, acc, out = fmt(8, 2), fmt(12, 1), fmt(5, 0)
    for spec, target in (({}, source), ({"sum": out}, out), ({"accumulator": acc}, acc),
                         ({"accumulator": acc, "sum": out}, out),
                         ({"accumulator": fmt(4, 0, False)}, fmt(4, 0, False))):
        policy = cf.Math(**spec)
        result = cfa.sum(raw_array([], source), math=policy)
        assert result.format == target and result.shape == () and result.raw.tolist() == 0
        wide = cfa.sum(cfa.from_raw(np.empty((0, 3), dtype=np.int8), format=source), axis=0, math=policy)
        assert wide.format == target and wide.raw.tolist() == [0, 0, 0]
        kept = cfa.sum(cfa.from_raw(np.empty((3, 0), dtype=np.int8), format=source), axis=1,
                       keepdims=True, math=policy)
        assert kept.format == target and kept.raw.tolist() == [[0], [0], [0]]
        dot = cfa.dot(raw_array([], source), raw_array([], source),
                      math=replace(policy, product=source))
        assert dot.format == target and dot.raw.tolist() == 0
    # An empty lane axis still declares guard bits for its nonempty summed axis.
    lanes = cfa.sum(cfa.from_raw(np.empty((0, 3), dtype=np.int8), format=source), axis=1)
    assert lanes.format == fmt(10, 2) and lanes.shape == (0,)
    z = cfa.sum(cfa.from_raw(np.empty((0, 2), dtype=np.int8), format=source, complex=True))
    assert z.format == source and z.is_complex and z.raw.tolist() == [0, 0]
    empty_product = cfa.matmul(cfa.from_raw(np.empty((2, 0), dtype=np.int8), format=source),
                               cfa.from_raw(np.empty((0, 3), dtype=np.int8), format=source))
    assert empty_product.format == fmt(16, 4) and empty_product.raw.tolist() == [[0] * 3] * 2


def test_sum_argument_validation():
    x = raw_array([[1, 2], [3, 4]], fmt())
    for axis in (2, -3):
        with pytest.raises(np.exceptions.AxisError):
            cfa.sum(x, axis)
    for axis in (True, np.bool_(False), (0,), (0, 1), [0], 1.0, "0"):
        with pytest.raises(TypeError, match="axis"):
            cfa.sum(x, axis)
    for keepdims in (1, None, np.bool_(True)):
        with pytest.raises(TypeError, match="keepdims"):
            cfa.sum(x, keepdims=keepdims)
    for value in (np.array([1, 2]), [1, 2], 3, cf.Fixb(1, 8, 2)):
        with pytest.raises(TypeError, match="fixed-point arrays"):
            cfa.sum(value)


# --- destinations ------------------------------------------------------------

def test_out_converts_after_the_sum_policy():
    # 0.3125 -> precision 2 -> precision 1 is 0.0; direct conversion gives 0.5.
    x = raw_array([4, 1], fmt(8, 4))
    stage, destination = fmt(8, 2), fmt(8, 1)
    out = cfa.empty((), format=destination)
    assert cfa.sum(x, out=out, math=cf.Math(sum=stage)) is out and out.raw.tolist() == 0
    assert cfa.sum(x, out=out).raw.tolist() == 1
    assert cfa.sum(x, math=cf.Math(sum=stage)).quantize(destination).raw.tolist() == 0
    # The accumulator boundary is likewise preserved ahead of sum and out.
    acc = cf.Math(accumulator=fmt(8, 3), sum=stage)
    # 0.25 -> 0.25; + 0.0625 = 0.3125 -> 0.25 (tie to even at precision 3) -> 0.25 -> 0.0
    assert cfa.sum(x, out=out, math=acc).raw.tolist() == 0
    ones = raw_array([1, 1], fmt(2, 0))
    assert cfa.dot(x, ones, out=out, math=cf.Math(sum=stage)).raw.tolist() == 0
    assert cfa.matmul(x, ones, out=(out,)).raw.tolist() == 1
    with cf.local_binary_context(math=cf.Math(sum=stage)):
        assert np.sum(x, out=out) is out and out.raw.tolist() == 0
        assert np.add.reduce(x, out=out) is out and np.dot(x, ones, out=out) is out
        assert np.matmul(x, ones, out=out) is out and out.raw.tolist() == 0

    z = raw_array([[4, 1], [1, 4]], fmt(8, 4), complex=True)
    zout = cfa.empty((), format=destination, complex=True)
    assert cfa.sum(z, out=zout, math=cf.Math(sum=stage)) is zout and zout.raw.tolist() == [0, 0]
    assert cfa.sum(z, out=zout).raw.tolist() == [1, 1]


def test_out_validation_precedes_execution():
    source = fmt(8, 0)
    x = raw_array([[1, 2, 3]], source)
    sentinel = cfa.from_raw(np.full(3, 99, dtype=np.int16), format=fmt(16, 0))
    for bad in (cfa.empty((2,), format=source), cfa.empty((3,), format=source, complex=True)):
        with pytest.raises(ValueError, match="shape and complexity"):
            cfa.sum(x, axis=0, out=bad)
    with pytest.raises(TypeError, match="Array"):
        cfa.sum(x, axis=0, out=np.empty(3, dtype=np.int64))
    with pytest.raises(TypeError, match="exactly one"):
        cfa.sum(x, axis=0, out=(sentinel, sentinel))
    readonly = np.zeros(3, dtype=np.int16)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        cfa.sum(x, axis=0, out=cfa.from_raw(readonly, format=fmt(16, 0)))
    # A kept unit axis gives the result the input's shape: aliases are rejected.
    alias = cfa.from_raw(np.array([[1, 2, 3]], dtype=np.int64), format=source)
    with pytest.raises(ValueError, match="overlap"):
        cfa.sum(alias, axis=0, keepdims=True, out=alias)
    vector = cfa.from_raw(np.array([1, 2, 3], dtype=np.int64), format=source)
    with pytest.raises(ValueError, match="overlap"):
        cfa.matmul(alias, vector, out=vector[:1])
    with pytest.raises(ValueError, match="unsupported int64"):
        cfa.sum(x, axis=0, out=cfa.empty((3,), format=fmt(8, 70)))
    # Numeric failures in any stage raise without writing the destination.
    loud = raw_array([[100], [100]], source)
    for policy in (cf.Math(accumulator=source), cf.Math(sum=source), cf.Math()):
        target = cfa.from_raw(np.full(1, 99, dtype=np.int8), format=fmt(8, 0))
        with pytest.raises(cf.FixedPointOverflowError):
            cfa.sum(loud, axis=0, out=target, math=policy)
        assert target.raw.tolist() == [99]
    with pytest.raises(cf.FixedPointOverflowError):
        cfa.dot(loud[:, 0], loud[:, 0], math=cf.Math(product=source))
    assert sentinel.raw.tolist() == [99, 99, 99]


# --- dot and matmul ----------------------------------------------------------

def test_dot_is_unconjugated_and_applies_product_policy_per_completed_product():
    q = fmt(8, 1)
    z = raw_array([[0, 2]], q, complex=True)          # 1j
    assert cfa.dot(z, z).raw.tolist() == [-4, 0]      # 1j * 1j == -1, at precision 2
    whole = cf.Math(product=fmt(8, 0))
    halves, ones = raw_array([1, 1], q), raw_array([2, 2], q)
    # Each 0.5 product ties to even 0 before accumulation; the exact sum is 1.
    assert cfa.dot(halves, ones, math=whole).raw.tolist() == 0
    assert cfa.dot(halves, ones, math=cf.Math(sum=fmt(8, 0))).raw.tolist() == 1
    # (0.5+0.5j)(0.5-0.5j) == 0.5 exactly. Rounding the partial products 0.25
    # and -0.25 separately would give 0; the completed component is 0.5.
    a, b = raw_array([[1, 1]], q, complex=True), raw_array([[1, -1]], q, complex=True)
    assert cfa.dot(a, b, math=cf.Math(product=fmt(8, 1))).raw.tolist() == [1, 0]
    for policy in (cf.Math(product=fmt(8, 1, False)), cf.Math(accumulator=fmt(8, 1, False)),
                   cf.Math(sum=fmt(8, 1, False))):
        with pytest.raises(ValueError, match="signed"):
            cfa.dot(a, b, math=policy)
        assert cfa.dot(halves, ones, math=policy).format.signed is False


@pytest.mark.parametrize("mode,overflow", list(itertools.product(MODES, OVERFLOWS)))
def test_exhaustive_two_term_dots_for_small_formats(mode, overflow):
    af, bf = fmt(3, 2), fmt(2, 1, False)
    araws, braws = range(af.min_int, af.max_int + 1), range(bf.min_int, bf.max_int + 1)
    full = fmt(5, 3)
    specs = [{}, {"product": fmt(4, 1)}, {"product": fmt(3, 2), "accumulator": fmt(4, 1)},
             {"accumulator": fmt(5, 4)}, {"product": fmt(5, 0), "sum": fmt(2, 0)},
             {"product": fmt(4, 2), "accumulator": fmt(4, 2), "sum": fmt(4, 3)}]
    assert list(braws) == [0, 1, 2, 3]
    for a, b in itertools.product(itertools.product(araws, repeat=2), ((0, 3), (1, 2), (3, 1))):
        x, y = raw_array(a, af), raw_array(b, bf)
        for spec in specs:
            spec = dict(spec, rounding=mode, overflow=overflow)
            # Operand order changes neither the completed products nor their order.
            for left, right in ((x, y), (y, x)):
                expect(lambda: cfa.dot(left, right, math=policy_of(spec)),
                       [products(a, af, b, bf)], full, spec)


def random_complex(rng, shape, format):
    raw = np.array([rng.randint(format.min_int, format.max_int)
                    for _ in range(int(np.prod(shape)) * 2)], dtype=np.int64)
    return raw.reshape(tuple(shape) + (2,))


def matmul_lanes(a, af, a_complex, b, bf, b_complex):
    """Exact completed products for every lane of a broadcast matrix product."""
    def pairs(raw, complex):
        return raw if complex else np.stack([raw, np.zeros_like(raw)], axis=-1)
    a, b = pairs(a, a_complex), pairs(b, b_complex)
    if a.ndim == 2:
        a = a[None]               # a logical vector on the left is one row
    if b.ndim == 2:
        b = b[:, None]            # and on the right, one column
    stack = np.broadcast_shapes(a.shape[:-3], b.shape[:-3])
    a = np.broadcast_to(a, stack + a.shape[-3:])
    b = np.broadcast_to(b, stack + b.shape[-3:])
    lanes = []
    for index in np.ndindex(stack + (a.shape[-3], b.shape[-2])):
        row, column = a[index[:-2] + (index[-2],)], b[index[:-2] + (slice(None), index[-1])]
        lanes.append(complex_products(row.tolist(), af, column.tolist(), bf))
    return lanes


MATMUL_SHAPES = [((3,), (3,), ()), ((2, 3), (3,), (2,)), ((3,), (3, 2), (2,)),
                 ((2, 3), (3, 4), (2, 4)), ((5, 2, 3), (3, 4), (5, 2, 4)),
                 ((2, 1, 2, 3), (4, 3, 2), (2, 4, 2, 2)), ((4, 7), (7,), (4,)),
                 ((0, 3), (3, 2), (0, 2)), ((2, 3), (3, 0), (2, 0))]
MATMUL_SPECS = [
    {},
    {"product": fmt(10, 6)},
    {"accumulator": fmt(14, 9)},
    {"product": fmt(9, 5), "accumulator": fmt(11, 4), "sum": fmt(8, 2)},
    {"product": fmt(12, 12), "sum": fmt(10, 3)},
    {"accumulator": fmt(16, 12), "sum": fmt(9, 9)},
]


@pytest.mark.parametrize("block", [1, 4, 1 << 16])
@pytest.mark.parametrize("a_complex,b_complex", [(False, False), (True, True), (True, False),
                                                 (False, True)])
def test_matmul_shapes_and_policies_match_the_model(monkeypatch, block, a_complex, b_complex):
    # Small blocks force several product blocks and a shorter final block.
    monkeypatch.setattr(cfa, "_BLOCK_ELEMENTS", block)
    rng = random.Random(block + 2 * a_complex + b_complex)
    af, bf = fmt(6, 5), fmt(5, 4)
    complex = a_complex or b_complex
    full = fmt(11 + (a_complex and b_complex), 9)
    for (a_shape, b_shape, shape), spec in itertools.product(MATMUL_SHAPES, MATMUL_SPECS):
        spec = dict(spec, rounding=rng.choice(MODES), overflow=rng.choice(("saturate", "wrap")))
        araw = random_complex(rng, a_shape, af)
        braw = random_complex(rng, b_shape, bf)
        araw, braw = (araw if a_complex else araw[..., 0]), (braw if b_complex else braw[..., 0])
        a = cfa.from_raw(araw, format=af, complex=a_complex)
        b = cfa.from_raw(braw, format=bf, complex=b_complex)
        lanes = matmul_lanes(araw, af, a_complex, braw, bf, b_complex)
        if not complex:
            lanes = [[term[0] for term in terms] for terms in lanes]
        if not lanes:
            result = cfa.matmul(a, b, math=policy_of(spec))
            assert result.shape == shape and result.raw.size == 0
            continue
        result = expect(lambda: cfa.matmul(a, b, math=policy_of(spec)), lanes, full, spec, complex)
        assert result.shape == shape
        with cf.local_binary_context(math=policy_of(spec)):
            np.testing.assert_array_equal((a @ b).raw, result.raw)
            np.testing.assert_array_equal(np.matmul(a, b).raw, result.raw)
        if len(a_shape) == len(b_shape) == 1:
            np.testing.assert_array_equal(cfa.dot(a, b, math=policy_of(spec)).raw, result.raw)


def test_unsigned_real_operands_and_mixed_signedness():
    rng = random.Random(3)
    uf, sf = fmt(5, 2, False), fmt(6, 3)
    u = np.array([rng.randint(0, 31) for _ in range(12)], dtype=np.int64).reshape(3, 4)
    s = np.array([rng.randint(-32, 31) for _ in range(8)], dtype=np.int64).reshape(4, 2)
    z = random_complex(rng, (4, 2), sf)
    for spec in ({}, {"product": fmt(8, 2, False), "overflow": "wrap"},
                 {"accumulator": fmt(9, 3, False), "overflow": "saturate"}):
        lanes = [[term[0] for term in terms] for terms in matmul_lanes(u, uf, False, u.T, uf, False)]
        expect(lambda: cfa.matmul(raw_array(u, uf), raw_array(np.ascontiguousarray(u.T), uf), math=policy_of(spec)),
               lanes, fmt(10, 4, False), spec)
    lanes = [[term[0] for term in terms] for terms in matmul_lanes(u, uf, False, s, sf, False)]
    expect(lambda: raw_array(u, uf) @ raw_array(s, sf), lanes, fmt(11, 5), {})
    expect(lambda: raw_array(u, uf) @ raw_array(z, sf, complex=True),
           matmul_lanes(u, uf, False, z, sf, True), fmt(11, 5), {}, complex=True)


def test_dot_and_matmul_argument_validation():
    q = fmt(8, 2)
    v, m, s = raw_array([1, 2, 3], q), raw_array([[1, 2, 3], [4, 5, 6]], q), raw_array(1, q)
    for a, b in ((m, v), (v, m), (m, m), (s, v), (v, s)):
        with pytest.raises(ValueError, match="one-dimensional"):
            cfa.dot(a, b)
    with pytest.raises(ValueError, match="mismatched lengths 3 and 2"):
        cfa.dot(v, v[:2])
    for a, b in ((s, v), (v, s), (s, s)):
        with pytest.raises(ValueError, match="at least one dimension"):
            cfa.matmul(a, b)
    for a, b in ((m, m), (v, v[:2]), (m, raw_array([1, 2], q))):
        with pytest.raises(ValueError, match="mismatched lengths"):
            cfa.matmul(a, b)
    with pytest.raises(ValueError, match="broadcast"):
        cfa.matmul(raw_array(np.ones((2, 2, 3)), q), raw_array(np.ones((3, 3, 2)), q))
    for other in (np.array([1, 2, 3]), [1, 2, 3], 2, cf.Fixb(1, 8, 2), 1.5):
        for function in (cfa.dot, cfa.matmul):
            for a, b in ((v, other), (other, v)):
                with pytest.raises(TypeError, match="fixed-point arrays"):
                    function(a, b)
    for call in (lambda: v @ 2, lambda: 2 @ v, lambda: v @ [1, 2, 3], lambda: [1, 2, 3] @ v,
                 lambda: v @ np.array([1, 2, 3]), lambda: np.array([1, 2, 3]) @ v,
                 lambda: v @ cf.Fixb(1, 8, 2), lambda: cf.Fixb(1, 8, 2) @ v):
        with pytest.raises(TypeError):
            call()
    # Without an in-place operator, @= computes a new array and rebinds the name.
    left = original = raw_array([[1, 2], [3, 4]], q)
    left @= left
    assert left is not original and original.raw.tolist() == [[1, 2], [3, 4]]
    assert left.raw.tolist() == [[7, 10], [15, 22]] and left.format == fmt(17, 4)


# --- checked int64 domains ---------------------------------------------------

def test_unsupported_reduction_domains_raise_before_execution():
    domain = "unsupported int64 kernel domain"
    wide = raw_array([1, 1], fmt(64, 0))
    assert cfa.sum(wide[:1]).raw.tolist() == 1
    with pytest.raises(ValueError, match=domain):
        cfa.sum(wide)
    # An explicit accumulator needs each exact step, not the whole sum, in int64.
    assert cfa.sum(raw_array([1, 1], fmt(63, 0)), math=cf.Math(accumulator=fmt(63, 0))).raw.tolist() == 2
    for accumulator in (fmt(63, 0), fmt(64, 0), fmt(8, 0)):
        with pytest.raises(ValueError, match=domain):
            cfa.sum(wide, math=cf.Math(accumulator=accumulator, overflow="wrap"))  # 65-bit step
    with pytest.raises(ValueError, match=domain):
        cfa.sum(raw_array([1, 1], fmt(63, 0, False)))
    with pytest.raises(ValueError, match=domain):
        cfa.sum(raw_array([1] * 5, fmt(62, 0)))
    # The message names the stage whose declared range or alignment is too wide.
    hint = "An explicit accumulator format bounds each step instead"
    for call, stage in (
            (lambda: cfa.sum(raw_array([1] * 5, fmt(62, 0))),
             f"the full sum of 5 terms is 65-bit signed. {hint}"),
            (lambda: cfa.sum(raw_array([1] * 3, fmt(62, 0, False))),
             f"the full sum of 3 terms is 64-bit unsigned. {hint}"),
            (lambda: cfa.sum(wide), "the full sum of 2 terms is 65-bit signed$"),
            (lambda: cfa.sum(raw_array([1, 1], fmt(63, 0, False))),
             "the full sum of 2 terms is 64-bit unsigned$"),
            (lambda: cfa.sum(wide, math=cf.Math(accumulator=fmt(64, 0))),
             "each exact accumulator step is 65-bit signed"),
            (lambda: cfa.sum(wide, math=cf.Math(accumulator=fmt(70, 0))),
             "the accumulator format is 70-bit signed"),
            (lambda: cfa.sum(raw_array([1, 1], fmt(16, 0)), math=cf.Math(accumulator=fmt(16, 60))),
             "aligning each 16-bit signed term to accumulator precision 60"),
            (lambda: cfa.sum(raw_array([1, 1], fmt(8, 8)), math=cf.Math(accumulator=fmt(62, 0))),
             "aligning the 62-bit signed accumulator to term precision 8"),
            (lambda: cfa.dot(raw_array([[1, 1]], fmt(32, 0), complex=True),
                             raw_array([[1, 1]], fmt(32, 0), complex=True)),
             "the full product is 65-bit signed"),
            (lambda: cfa.dot(raw_array([1], fmt(8, 0)), raw_array([1], fmt(8, 0)),
                             math=cf.Math(product=fmt(64, 0, False))),
             "the product format is 64-bit unsigned"),
            (lambda: cfa.dot(raw_array([1], fmt(16, 0)), raw_array([1], fmt(16, 0)),
                             math=cf.Math(product=fmt(16, 80))),
             "converting each 32-bit signed product to product precision 80"),
            (lambda: cfa.sum(raw_array([[1, 1]], fmt(8, 0), complex=True),
                             math=cf.Math(accumulator=fmt(8, 0, False))),
             "signed binary format; the accumulator format is 8-bit unsigned")):
        with pytest.raises(ValueError, match=stage):
            call()
    # The hint appears exactly when one more bit than a term is representable.
    for width, signed in ((63, True), (62, False)):
        terms = raw_array([1] * 3, fmt(width, 0, signed))
        with pytest.raises(ValueError, match=hint):
            cfa.sum(terms)
        assert cfa.sum(terms, math=cf.Math(accumulator=fmt(width, 0, signed))).raw.tolist() == 3
    small = raw_array([1, 1], fmt(16, 0))
    with pytest.raises(ValueError, match=domain):
        cfa.sum(small, math=cf.Math(accumulator=fmt(16, 60)))     # term alignment
    with pytest.raises(ValueError, match=domain):
        # Accumulator alignment: 62 bits shifted up to the term's precision.
        cfa.sum(raw_array([1, 1], fmt(8, 8)), math=cf.Math(accumulator=fmt(62, 0)))
    with pytest.raises(ValueError, match=domain):
        cfa.sum(small, math=cf.Math(accumulator=fmt(65, 0)))
    with pytest.raises(ValueError, match=domain):
        cfa.sum(small, math=cf.Math(sum=fmt(16, 60)))
    # A narrow product or result never excuses an unrepresentable intermediate.
    half = raw_array([1, 1], fmt(32, 0))
    z = raw_array([[1, 1]], fmt(32, 0), complex=True)
    narrow = cf.Math(product=fmt(8, 0), accumulator=fmt(8, 0), sum=fmt(8, 0), overflow="wrap")
    assert cfa.dot(half[:1], half[:1]).format == fmt(64, 0)
    with pytest.raises(ValueError, match=domain):
        cfa.dot(half, half)
    assert cfa.dot(half, half, math=narrow).raw.tolist() == 2
    for policy in (cf.Math(), narrow):
        with pytest.raises(ValueError, match=domain):
            cfa.dot(z, z, math=policy)
    with pytest.raises(ValueError, match=domain):
        cfa.dot(small, small, math=cf.Math(product=fmt(16, 80)))


@pytest.mark.parametrize("overflow", OVERFLOWS)
def test_int64_boundaries(overflow):
    source = fmt(62, 0)
    for raw in (source.min_int, source.max_int):
        x = raw_array([raw] * 4, source)
        assert cfa.sum(x).format == fmt(64, 0) and cfa.sum(x).raw.tolist() == 4 * raw
    # A 63-bit accumulator and term meet in an exact 64-bit step.
    acc = fmt(63, 0)
    edges = [acc.min_int, acc.max_int, acc.max_int, -1, acc.min_int, acc.min_int, 1, acc.max_int]
    for terms in (edges, edges[::-1], [acc.max_int] * 3, [acc.min_int] * 3, [acc.min_int, acc.max_int]):
        spec = {"accumulator": acc, "overflow": overflow}
        expect(lambda: cfa.sum(raw_array(terms, acc), math=policy_of(spec)),
               [[Fraction(t) for t in terms]], acc, spec)
    # Rounding at the accumulator boundary near the int64 limits.
    fine = fmt(63, 1)
    for mode in MODES:
        spec = {"accumulator": fmt(62, 0), "overflow": overflow, "rounding": mode}
        terms = [fine.max_int, fine.min_int, -1, 1, fine.max_int, 3, -3, fine.min_int]
        expect(lambda: cfa.sum(raw_array(terms, fine), math=policy_of(spec)),
               [[val(t, fine) for t in terms]], fine, spec)
    # Full 31-bit complex products reach 63 bits and sum exactly in 64.
    q = fmt(31, 0)
    lo, hi = q.min_int, q.max_int
    z = raw_array([[lo, lo], [lo, hi]], q, complex=True)
    w = raw_array([[lo, hi], [hi, lo]], q, complex=True)
    result = cfa.dot(z, w)
    assert result.format == fmt(64, 0)
    expected = [lo * lo - lo * hi + lo * hi - hi * lo, lo * hi + lo * lo + lo * lo + hi * hi]
    assert result.raw.tolist() == expected


# --- NumPy dispatch and contexts ---------------------------------------------

def test_numpy_dispatch_follows_the_active_context():
    q = fmt(8, 1)
    x = raw_array([[1, 1, 2], [2, 1, 1]], q)
    v = raw_array([2, 2, 2], q)
    assert np.sum(x).format == fmt(11, 1) and np.sum(x).raw.tolist() == 8
    assert np.sum(x, 0).raw.tolist() == np.sum(x, axis=0).raw.tolist() == [3, 2, 3]
    assert np.sum(x, axis=1, keepdims=True).raw.tolist() == [[4], [4]]
    # np.add.reduce defaults to the first axis; np.sum reduces every axis.
    assert np.add.reduce(x).raw.tolist() == [3, 2, 3]
    assert np.add.reduce(x, 1).raw.tolist() == np.add.reduce(x, axis=-1).raw.tolist() == [4, 4]
    assert np.add.reduce(x, axis=None).raw.tolist() == 8
    assert np.add.reduce(x, axis=1, keepdims=True).shape == (2, 1)
    assert np.dot(v, v).raw.tolist() == 12 and np.dot(v, v).format == fmt(18, 2)
    assert np.matmul(x, v).raw.tolist() == (x @ v).raw.tolist() == [8, 8]
    assert x.sum(1).raw.tolist() == [4, 4]

    ordered = cf.Math(accumulator=fmt(8, 0), product=fmt(8, 1))
    with cf.local_binary_context(math=ordered):
        scoped = [np.sum(x, axis=1), np.add.reduce(x, 1), x.sum(axis=1), cfa.sum(x, 1)]
        products = [np.dot(x[0], v), x[0] @ v, np.matmul(x[0], v), cfa.dot(x[0], v)]
        # An explicit math argument overrides the scope for that call only.
        assert cfa.sum(x, 1, math=cf.Math()).raw.tolist() == [4, 4]
    for result in scoped:
        assert result.format == fmt(8, 0) and result.raw.tolist() == [1, 2]
    for result in products:
        assert result.format == fmt(8, 0) and result.raw.tolist() == 1
    assert np.sum(x, axis=1).raw.tolist() == [4, 4]                # scope restored
    with pytest.raises(cf.FixedPointOverflowError):
        with cf.local_binary_context(math=cf.Math(sum=fmt(2, 1))):
            np.sum(x)
    assert np.sum(x).raw.tolist() == 8                             # restored after an error


def test_unsupported_numpy_reduction_requests_fail_explicitly():
    x = raw_array([[1, 2], [3, 4]], fmt())
    out = cfa.empty((2,), format=fmt())
    for call in (lambda: np.sum(x, initial=0), lambda: np.sum(x, where=np.array([True, False])),
                 lambda: np.sum(x, dtype=np.int64), lambda: np.sum(x, axis=(0, 1)),
                 lambda: np.sum(x, None, np.int64), lambda: np.add.reduce(x, dtype=np.int64),
                 lambda: np.add.reduce(x, initial=1), lambda: np.add.reduce(x, where=False),
                 lambda: np.add.reduce(x, where=np.array([True, False])),
                 lambda: np.add.reduce(x, axis=(0, 1)), lambda: np.add.reduce(x, out=np.empty(2)),
                 lambda: np.sum(x, axis=0, out=np.empty(2)),
                 lambda: np.add.accumulate(x), lambda: np.add.reduceat(x, [0]),
                 lambda: np.multiply.reduce(x), lambda: np.subtract.reduce(x),
                 lambda: np.matmul(x, x, axes=[(0, 1), (0, 1), (0, 1)]),
                 lambda: np.matmul(x, x, dtype=np.int64), lambda: np.matmul(x, x, casting="unsafe"),
                 lambda: np.matmul(x, x, where=False), lambda: np.matmul(x, np.ones((2, 2))),
                 lambda: np.dot(x[0], np.ones(2)), lambda: np.dot(np.ones(2), x[0]),
                 lambda: np.dot(x[0], 2), lambda: np.vdot(x[0], x[0]), lambda: np.inner(x[0], x[0]),
                 lambda: np.cumsum(x), lambda: np.prod(x), lambda: np.mean(x),
                 lambda: np.einsum("ij,jk", x, x), lambda: np.tensordot(x, x)):
        with pytest.raises(TypeError):
            call()
    assert np.sum(x, axis=0, out=out) is out and out.raw.tolist() == [4, 6]
    assert np.sum(x, None, None, cfa.empty((), format=fmt())).raw.tolist() == 10


def test_numpy_reductions_accept_every_default_spelling():
    x = raw_array([[1, 2], [3, 4]], fmt())
    # NumPy forwards explicitly passed defaults, repeats array= as a keyword,
    # and turns a reduce's positional arguments into keywords.
    for call in (lambda: np.sum(a=x), lambda: np.sum(x, dtype=None), lambda: np.sum(x, None, None),
                 lambda: np.sum(x, where=True), lambda: np.sum(a=x, axis=None, dtype=None, out=None,
                                                             keepdims=False, where=True)):
        result = call()
        assert result.raw.tolist() == 10 and result.format == fmt(10)
    for call in (lambda: np.add.reduce(array=x), lambda: np.add.reduce(x, 0, None),
                 lambda: np.add.reduce(x, dtype=None), lambda: np.add.reduce(x, where=True),
                 lambda: np.add.reduce(x, 0, None, None, False),
                 lambda: np.add.reduce(array=x, axis=0, dtype=None, out=None, keepdims=False,
                                       where=True)):
        result = call()
        assert result.raw.tolist() == [4, 6] and result.format == fmt(9)
    assert np.add.reduce(array=x, axis=1).raw.tolist() == np.add.reduce(x, 1).raw.tolist() == [3, 7]
    assert np.sum(a=x, axis=1).raw.tolist() == np.sum(x, 1).raw.tolist() == [3, 7]
    out = cfa.empty((2,), format=fmt())
    assert np.add.reduce(x, 0, None, out) is out and np.sum(x, 0, None, out) is out
    assert np.dot(a=x[0], b=x[1]).raw.tolist() == 11
    for function in (np.sum, np.add.reduce):
        for options in ({"dtype": np.int64}, {"dtype": "int64"}, {"initial": 0}, {"initial": None},
                        {"where": False}, {"where": np.True_}, {"where": np.array([True, False])},
                        {"where": 1}):
            with pytest.raises(TypeError, match="keywords|where=True"):
                function(x, **options)
    with pytest.raises(TypeError, match="initial"):
        np.sum(x, None, None, None, False, 0)
    with pytest.raises(TypeError, match="dtype, initial"):
        np.add.reduce(x, 0, np.int64, None, False, 0)


class Duck:
    """A foreign array-like that answers every NumPy protocol call itself."""

    def __array_function__(self, func, types, args, kwargs):
        return "duck"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return "duck"

    def __matmul__(self, other):
        return "duck"

    def __rmatmul__(self, other):
        return "duck"


NO = np._NoValue
# Every way to spell a reduction of one operand, by entry point.
UNARY_SPELLINGS = {
    "np.sum(x)": lambda x: np.sum(x),
    "np.sum(a=x)": lambda x: np.sum(a=x),
    "np.sum(x, None)": lambda x: np.sum(x, None),
    "np.sum(x, 0)": lambda x: np.sum(x, 0),
    "np.sum(x, -1)": lambda x: np.sum(x, -1),
    "np.sum(x, axis=1)": lambda x: np.sum(x, axis=1),
    "np.sum(x, axis=np.int64(0))": lambda x: np.sum(x, axis=np.int64(0)),
    "np.sum(x, axis=2)": lambda x: np.sum(x, axis=2),
    "np.sum(x, axis=-3)": lambda x: np.sum(x, axis=-3),
    "np.sum(x, axis=0, keepdims=True)": lambda x: np.sum(x, axis=0, keepdims=True),
    "np.sum(x, keepdims=True)": lambda x: np.sum(x, keepdims=True),
    "np.sum(x, None, None)": lambda x: np.sum(x, None, None),
    "np.sum(x, dtype=None)": lambda x: np.sum(x, dtype=None),
    "np.sum(x, 0, None, None, False)": lambda x: np.sum(x, 0, None, None, False),
    "np.sum(x, where=True)": lambda x: np.sum(x, where=True),
    "np.sum(x, keepdims=NO)": lambda x: np.sum(x, keepdims=NO),
    "np.sum(x, initial=NO)": lambda x: np.sum(x, initial=NO),
    "np.sum(x, where=NO)": lambda x: np.sum(x, where=NO),
    "np.sum(x, keepdims=NO, initial=NO, where=NO)":
        lambda x: np.sum(x, keepdims=NO, initial=NO, where=NO),
    "np.sum(x, 1.0)": lambda x: np.sum(x, 1.0),
    "np.sum(x, True)": lambda x: np.sum(x, True),
    "np.add.reduce(x)": lambda x: np.add.reduce(x),
    "np.add.reduce(array=x)": lambda x: np.add.reduce(array=x),
    "np.add.reduce(x, 0)": lambda x: np.add.reduce(x, 0),
    "np.add.reduce(x, -1)": lambda x: np.add.reduce(x, -1),
    "np.add.reduce(x, 1)": lambda x: np.add.reduce(x, 1),
    "np.add.reduce(x, None)": lambda x: np.add.reduce(x, None),
    "np.add.reduce(x, axis=2)": lambda x: np.add.reduce(x, axis=2),
    "np.add.reduce(x, axis=None, keepdims=True)": lambda x: np.add.reduce(x, axis=None, keepdims=True),
    "np.add.reduce(array=x, axis=-1, keepdims=True)":
        lambda x: np.add.reduce(array=x, axis=-1, keepdims=True),
    "np.add.reduce(x, 0, None)": lambda x: np.add.reduce(x, 0, None),
    "np.add.reduce(x, 0, None, None, False)": lambda x: np.add.reduce(x, 0, None, None, False),
    "np.add.reduce(x, dtype=None)": lambda x: np.add.reduce(x, dtype=None),
    "np.add.reduce(x, where=True)": lambda x: np.add.reduce(x, where=True),
    "np.add.reduce(x, initial=NO)": lambda x: np.add.reduce(x, initial=NO),
    "np.add.reduce(x, where=NO)": lambda x: np.add.reduce(x, where=NO),
    "np.add.reduce(x, keepdims=NO)": lambda x: np.add.reduce(x, keepdims=NO),
    "x.sum()": lambda x: x.sum(),
    "x.sum(0)": lambda x: x.sum(0),
    "x.sum(axis=-1)": lambda x: x.sum(axis=-1),
    "x.sum(axis=1, keepdims=True)": lambda x: x.sum(axis=1, keepdims=True),
    "x.sum(axis=2)": lambda x: x.sum(axis=2),
}
BINARY_SPELLINGS = {
    "np.matmul(a, b)": lambda a, b: np.matmul(a, b),
    "a @ b": lambda a, b: a @ b,
}
UNARY_OPERANDS = {"matrix": np.arange(6).reshape(2, 3) - 2, "vector": np.array([3, -1, 2]),
                  "zero-dimensional": np.array(5), "empty": np.zeros((0, 3), dtype=np.int64)}
BINARY_OPERANDS = {
    "vector, vector": (np.array([3, -1, 2]), np.array([1, 4, -2])),
    "matrix, vector": (np.arange(6).reshape(2, 3), np.array([1, 4, -2])),
    "vector, matrix": (np.array([3, -1]), np.arange(6).reshape(2, 3)),
    "matrix, matrix": (np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4)),
    "stack, matrix": (np.arange(24).reshape(4, 2, 3), np.arange(6).reshape(3, 2)),
    "broadcast stacks": (np.arange(12).reshape(2, 1, 2, 3), np.arange(18).reshape(3, 3, 2)),
    "mismatched vectors": (np.array([3, -1, 2]), np.array([1, 4])),
    "mismatched matrices": (np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3)),
    "mismatched stacks": (np.arange(12).reshape(2, 2, 3), np.arange(18).reshape(3, 3, 2)),
    "zero-dimensional, vector": (np.array(5), np.array([1, 4, -2])),
    "vector, zero-dimensional": (np.array([1, 4, -2]), np.array(5)),
}
WIDE = fmt(12, 0)


def assert_same_outcome(call, reference, operands):
    """The fixed-point call accepts or rejects exactly as the ndarray call does."""
    try:
        expected = call(*reference)
    except (TypeError, ValueError):
        with pytest.raises((TypeError, ValueError)):
            call(*operands)
        return
    result = call(*operands)
    assert isinstance(result, cfa.Array) and result.shape == np.shape(expected)
    np.testing.assert_array_equal(result.raw, expected)


@pytest.mark.parametrize("spelling", UNARY_SPELLINGS)
@pytest.mark.parametrize("operand", UNARY_OPERANDS)
def test_unary_reduction_spellings_accept_and_reject_as_numpy_does(spelling, operand):
    raw = UNARY_OPERANDS[operand]
    assert_same_outcome(UNARY_SPELLINGS[spelling], (raw,), (raw_array(raw, WIDE),))


@pytest.mark.parametrize("spelling", BINARY_SPELLINGS)
@pytest.mark.parametrize("operands", BINARY_OPERANDS)
def test_matmul_spellings_accept_and_reject_as_numpy_does(spelling, operands):
    a, b = BINARY_OPERANDS[operands]
    assert_same_outcome(BINARY_SPELLINGS[spelling], (a, b), (raw_array(a, WIDE), raw_array(b, WIDE)))
    if a.ndim == b.ndim == 1:
        for dot in (np.dot, lambda a, b: np.dot(a=a, b=b), lambda a, b: np.dot(a, b, None)):
            assert_same_outcome(dot, (a, b), (raw_array(a, WIDE), raw_array(b, WIDE)))


def test_out_spellings_match_numpy():
    raw = np.arange(6).reshape(2, 3)
    x = raw_array(raw, WIDE)
    for call, shape in ((lambda x, out: np.sum(x, 0, None, out), (3,)),
                        (lambda x, out: np.sum(x, axis=1, out=out), (2,)),
                        (lambda x, out: np.sum(x, out=out), ()),
                        (lambda x, out: np.add.reduce(x, 0, None, out), (3,)),
                        (lambda x, out: np.add.reduce(x, axis=1, out=(out,)), (2,)),
                        (lambda x, out: x.sum(axis=0, out=out), (3,)),
                        (lambda x, out: np.matmul(x, x[0], out=out), (2,)),
                        (lambda x, out: np.matmul(x, x[0], out=(out,)), (2,)),
                        (lambda x, out: np.dot(x[0], x[1], out), ())):
        reference = np.empty(shape, dtype=np.int64)
        call(raw, reference)
        out = cfa.empty(shape, format=fmt(16, 0))
        assert call(x, out) is out
        np.testing.assert_array_equal(out.raw, reference)
        with pytest.raises(ValueError, match="shape and complexity"):
            call(x, cfa.empty(shape + (4,), format=fmt(16, 0)))
        with pytest.raises(TypeError):
            call(x, reference)                  # an ndarray destination has no format


def test_documented_differences_from_numpy_fail_explicitly():
    raw = np.arange(6).reshape(2, 3)
    x, zero_d = raw_array(raw, WIDE), raw_array(5, WIDE)
    # Each of these succeeds on a plain ndarray and is rejected here by design.
    for call in (lambda x: np.sum(x, dtype=np.int64), lambda x: np.sum(x, initial=0),
                 lambda x: np.sum(x, where=np.array([True, False, True])),
                 lambda x: np.sum(x, axis=(0, 1)), lambda x: np.sum(x, axis=()),
                 lambda x: np.sum(x, keepdims=1), lambda x: np.add.reduce(x, dtype=np.int64),
                 lambda x: np.add.reduce(x, initial=0), lambda x: np.add.reduce(x, initial=None),
                 lambda x: np.add.reduce(x, where=np.array([True, False, True])),
                 lambda x: np.add.reduce(x, axis=(0, 1)), lambda x: np.add.reduce(x, keepdims=1),
                 lambda x: x.sum(axis=(0, 1)), lambda x: np.dot(x, x[0]),
                 lambda x: np.matmul(x, x[0], axes=[(0, 1), (0,), (0,)]),
                 lambda x: np.matmul(x, x[0], dtype=np.int64),
                 lambda x: np.matmul(x, x[0], casting="unsafe")):
        call(raw)
        with pytest.raises((TypeError, ValueError)):
            call(x)
    with pytest.raises(ValueError, match="one-dimensional"):
        np.dot(zero_d, zero_d)                  # NumPy multiplies zero-dimensional operands
    assert np.dot(np.array(5), np.array(5)) == 25


def test_ndarrays_are_rejected_and_foreign_types_are_deferred_to():
    raw = np.arange(6).reshape(2, 3)
    x, duck = raw_array(raw, WIDE), Duck()
    masked = np.ma.array(raw[0])
    calls = {
        "np.dot": (lambda a, b: np.dot(a, b), 0),
        "np.matmul": (lambda a, b: np.matmul(a, b), 0),
        "@": (lambda a, b: a @ b, 0),
    }
    # Bare ndarrays and ndarray subclasses carry no format: both orders are
    # rejected with a TypeError, never coerced.
    for name, (call, row) in calls.items():
        for other in (raw[row], masked):
            for a, b in ((x[row], other), (other, x[row])):
                with pytest.raises(TypeError):
                    call(a, b)
    for call in (lambda: np.sum(x, out=np.empty(())), lambda: np.add.reduce(x, out=np.empty(3)),
                 lambda: np.concatenate([x, raw]), lambda: np.concatenate([raw, x]),
                 lambda: np.concatenate([x, masked[None]])):
        with pytest.raises(TypeError):
            call()
    # A foreign type's own handler answers in either operand order, for
    # supported and unsupported functions alike.
    for call in (lambda: np.dot(x[0], duck), lambda: np.dot(duck, x[0]),
                 lambda: np.matmul(x, duck), lambda: np.matmul(duck, x),
                 lambda: x @ duck, lambda: duck @ x,
                 lambda: np.sum(x, out=duck), lambda: np.add.reduce(x, out=duck),
                 lambda: np.sum(duck, out=cfa.empty((), format=WIDE)),
                 lambda: np.concatenate([x, duck]), lambda: np.concatenate([duck, x]),
                 lambda: np.vdot(x[0], duck), lambda: np.add(x, duck), lambda: np.add(duck, x)):
        assert call() == "duck"

    class Silent:
        def __array_function__(self, func, types, args, kwargs):
            return NotImplemented

    for call in (lambda: np.dot(x[0], Silent()), lambda: np.sum(x, out=Silent()),
                 lambda: np.concatenate([Silent(), x])):
        with pytest.raises(TypeError, match="no implementation found"):
            call()


# --- scalar reference and FIR-style fixture ----------------------------------

def scalar_of(raw, format, complex):
    if complex:
        return cf.CFixb.from_raw(i=int(raw[0]), q=int(raw[1]), format=format)
    return (cf.Fixb if format.signed else cf.Ufixb).from_raw(int(raw), format=format)


def scalar_reduce(pairs, policy, reached, complex):
    """Readable scalar loop: product, ordered accumulator, then sum policy.

    Scalar full additions grow one bit each, so an exact total is converted to
    the reduction's declared ``reached`` format before the sum policy applies.
    """
    exact = cf.Math(rounding=policy.rounding, overflow=policy.overflow)
    zero = cf.CFixb(0, format=reached) if complex else scalar_of(0, reached, False)
    total = zero
    for a, b in pairs:
        term = cf.multiply(a, b, math=policy)
        if policy.accumulator == "full":
            total = cf.add(total, term, math=exact)
        else:
            total = cf.add(total, term, math=replace(policy, sum=policy.accumulator))
    total = total.quantize(reached, math=exact)
    return total if policy.sum == "full" else total.quantize(policy.sum, math=policy)


FIR_POLICIES = [
    cf.Math(rounding="floor", overflow="wrap", product=fmt(24, 20), accumulator=fmt(26, 20),
            sum=fmt(18, 14)),
    cf.Math(rounding="nearest_even", overflow="saturate", product=fmt(20, 16),
            accumulator=fmt(21, 15), sum=fmt(16, 13)),
    cf.Math(rounding="nearest_away", overflow="saturate", product=fmt(33, 30),
            accumulator=fmt(18, 12)),
    cf.Math(rounding="zero", overflow="wrap", product=fmt(22, 18)),
    cf.Math(),
]


@pytest.mark.parametrize("complex", [False, True])
@pytest.mark.parametrize("policy", FIR_POLICIES)
def test_fir_style_fixture_matches_scalar_reference_and_model(policy, complex):
    """y[n] = sum_k h[k] x[n + K - 1 - k], with explicit stage formats.

    Sixteen-bit samples and coefficients, one contiguous row of history per
    output sample, coefficients applied newest-first.
    """
    rng = random.Random(2026 + complex)
    q15 = fmt(16, 15)
    taps, frame = 7, 24
    shape = (taps, 2) if complex else (taps,)
    h = np.array([rng.randint(-2 ** 12, 2 ** 12) for _ in range(taps * 2)], dtype=np.int16)
    h = h.reshape(taps, 2) if complex else h[:taps]
    x = np.array([rng.randint(-2 ** 15, 2 ** 15 - 1) for _ in range((frame + taps - 1) * 2)],
                 dtype=np.int16)
    x = x.reshape(-1, 2) if complex else x[:frame + taps - 1]
    assert h.shape == shape
    # Row n holds x[n + K - 1], ..., x[n]: the newest sample meets h[0].
    history = np.ascontiguousarray(np.stack([x[n:n + taps][::-1] for n in range(frame)]))
    windows = cfa.from_raw(history, format=q15, complex=complex)
    coefficients = cfa.from_raw(h, format=q15, complex=complex)

    result = cfa.matmul(windows, coefficients, math=policy)
    assert result.shape == (frame,) and result.is_complex == complex
    spec = {name: getattr(policy, name) for name in ("product", "accumulator", "sum")}
    spec.update(rounding=policy.rounding.value, overflow=policy.overflow.name.lower())
    full = fmt(32 + complex, 30)
    lanes = matmul_lanes(history, q15, complex, h, q15, complex)
    if not complex:
        lanes = [[term[0] for term in terms] for terms in lanes]
    expect(lambda: result, lanes, full, spec, complex)

    term = full if policy.product == "full" else policy.product
    reached = (replace(term, wordlength=term.wordlength + guard(taps))
               if policy.accumulator == "full" else policy.accumulator)
    for n in range(frame):
        pairs = [(scalar_of(history[n, k], q15, complex), scalar_of(h[k], q15, complex))
                 for k in range(taps)]
        scalar = scalar_reduce(pairs, policy, reached, complex)
        assert scalar == result[n] and scalar.format == result.format
        np.testing.assert_array_equal(cfa.dot(windows[n], coefficients, math=policy).raw,
                                      result.raw[n])

    # The explicit policy is independent of the caller's context, and a
    # destination conversion follows the completed sum stage.
    output = fmt(16, 13)
    out = cfa.empty((frame,), format=output, complex=complex)
    convert = cf.Math(rounding=policy.rounding, overflow=policy.overflow)
    with cf.local_binary_context(math=cf.Math(rounding="ceil", overflow="saturate",
                                              product=fmt(8, 0), accumulator=fmt(8, 0))):
        assert cfa.matmul(windows, coefficients, out=out, math=policy) is out
    np.testing.assert_array_equal(out.raw, result.quantize(output, math=convert).raw)
    # The same three stages spelled with ordinary scoped expressions: product
    # policy on each elementwise product, then an ordered sum along each row.
    with cf.local_binary_context(math=policy):
        np.testing.assert_array_equal((windows @ coefficients).raw, result.raw)
        np.testing.assert_array_equal(np.sum(windows * coefficients, axis=1).raw, result.raw)


@pytest.mark.parametrize("seed", range(4))
def test_random_sixteen_bit_iq_dot_parity(seed):
    rng = random.Random(seed)
    q = fmt(16, 14)
    count = 33
    a, b = random_complex(rng, (count,), q), random_complex(rng, (count,), q)
    za, zb = raw_array(a, q, complex=True), raw_array(b, q, complex=True)
    for policy in FIR_POLICIES:
        result = cfa.dot(za, zb, math=policy)
        term = fmt(33, 28) if policy.product == "full" else policy.product
        reached = (replace(term, wordlength=term.wordlength + guard(count))
                   if policy.accumulator == "full" else policy.accumulator)
        pairs = [(scalar_of(x, q, True), scalar_of(y, q, True)) for x, y in zip(a, b)]
        scalar = scalar_reduce(pairs, policy, reached, True)
        assert result[()] == scalar and result.format == scalar.format
        # Reversing both operands changes only the order of accumulation.
        reverse = cfa.dot(raw_array(a[::-1], q, complex=True), raw_array(b[::-1], q, complex=True),
                          math=policy)
        if policy.accumulator == "full":
            np.testing.assert_array_equal(reverse.raw, result.raw)
        else:
            assert reverse[()] == scalar_reduce(pairs[::-1], policy, reached, True)
