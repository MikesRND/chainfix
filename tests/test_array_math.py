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

"""Independent exact oracles and scalar parity for bounded integer kernels."""

from fractions import Fraction
import itertools
import operator

import pytest

np = pytest.importorskip("numpy", exc_type=ImportError)

import chainfix as cf
from chainfix import arrays as cfa
from tests.test_binary_math import MODES, OVERFLOWS, fmt, number, oracle_raw
from tests.test_complex_math import calculate, full_format


def raw_array(values, format, complex=False):
    return cfa.from_raw(np.array(values, dtype=np.int64), format=format, complex=complex)


def test_exhaustive_small_real_formats():
    formats = [fmt(w, p, s) for w, p, s in itertools.product((1, 2, 3), (-1, 0, 2), (False, True))]
    for af, bf in itertools.product(formats, repeat=2):
        av, bv = list(range(af.min_int, af.max_int + 1)), list(range(bf.min_int, bf.max_int + 1))
        a, b = raw_array(np.array(av)[:, None], af), raw_array(bv, bf)
        for operation in ("add", "subtract", "multiply"):
            result = getattr(cfa, operation)(a, b)
            arithmetic = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul}[operation]
            exact = [[arithmetic(Fraction(x) * Fraction(2)**-af.precision,
                                 Fraction(y) * Fraction(2)**-bf.precision) for y in bv] for x in av]
            if operation == "multiply":
                expected_format = fmt(af.wordlength + bf.wordlength, af.precision + bf.precision,
                                      af.signed or bf.signed)
            else:
                precision = max(af.precision, bf.precision)
                scaled = [v * Fraction(2)**precision for row in exact for v in row]
                signed = af.signed or bf.signed or operation == "subtract"
                width = 1
                while min(scaled) < (-(2**(width-1)) if signed else 0) or max(scaled) >= 2**(width-int(signed)):
                    width += 1
                expected_format = fmt(width, precision, signed)
            assert result.format == expected_format
            expected = [[oracle_raw(v, expected_format, "nearest_even", "raise") for v in row] for row in exact]
            np.testing.assert_array_equal(result.raw, expected)
            for i, x in enumerate(av):
                for j, y in enumerate(bv):
                    scalar = getattr(cf, operation)(number(x, af), number(y, bf))
                    assert scalar.int == result.raw[i, j] and scalar.format == result.format


@pytest.mark.parametrize("mode,overflow", list(itertools.product(MODES, OVERFLOWS)))
def test_bounded_real_and_unary_oracles(mode, overflow):
    target = fmt(3, 1)
    policy = cf.Math(rounding=mode, overflow=overflow, sum=target, product=target)
    for signed, precision in itertools.product((False, True), (-1, 2)):
        source = fmt(3, precision, signed)
        raws = list(range(source.min_int, source.max_int + 1))
        a = raw_array(raws, source)
        values = [Fraction(r) * Fraction(2)**-precision for r in raws]
        calls = [(lambda op=op: op(a, math=policy), [transform(v) for v in values])
                 for op, transform in ((cfa.negative, operator.neg), (cfa.absolute, abs))]
        for operation in ("add", "subtract", "multiply"):
            op = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul}[operation]
            calls.append((lambda operation=operation: getattr(cfa, operation)(a[:, None], a, math=policy),
                          [op(x, y) for x in values for y in values]))
        for call, exact in calls:
            try:
                expected = [oracle_raw(v, target, mode, overflow) for v in exact]
            except ValueError:
                with pytest.raises(cf.FixedPointOverflowError):
                    call()
            else:
                result = call()
                assert result.format == target
                assert result.raw.ravel().tolist() == expected


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("overflow", OVERFLOWS)
def test_quantize_int64_edges_and_all_shift_domains(mode, overflow):
    values = [-(2**63), -(2**63)+1, -(2**62)-1, -17, -9, -5, -3, -2, -1,
              0, 1, 2, 3, 5, 9, 17, 2**62-1, 2**62, 2**63-2, 2**63-1]
    a = raw_array(values, fmt(64, -3))
    policy = cf.Math(rounding=mode, overflow=overflow)
    for shift, width, signed in itertools.product((0, 1, 2, 3, 61, 62, 63, 64, 65, 1000),
                                                 (1, 3, 8, 32, 63, 64), (False, True)):
        if not signed and width == 64:
            continue
        target = fmt(width, -3-shift, signed)
        try:
            expected = [oracle_raw(Fraction(v) * 8, target, mode, overflow) for v in values]
        except ValueError:
            with pytest.raises(cf.FixedPointOverflowError):
                a.quantize(target, math=policy)
        else:
            result = a.quantize(target, math=policy)
            assert result.raw.tolist() == expected
            assert result.raw.dtype.kind != "O"
            scalar = [number(v, a.format).quantize(target, math=policy).int for v in values]
            assert result.raw.tolist() == scalar


@pytest.mark.parametrize("overflow", OVERFLOWS)
def test_left_conversion_machine_boundaries(overflow):
    for source in (fmt(1, 0), fmt(2, 0), fmt(8, -2), fmt(8, 3, False)):
        values = [source.min_int, 0, source.max_int]
        a = raw_array(values, source)
        largest = (64 if source.signed else 63) - source.wordlength
        for shift in (0, 1, largest):
            for target in (fmt(64, source.precision+shift), fmt(8, source.precision+shift)):
                policy = cf.Math(overflow=overflow)
                try:
                    expected = [oracle_raw(Fraction(v)*Fraction(2)**-source.precision,
                                           target, "nearest_even", overflow) for v in values]
                except ValueError:
                    with pytest.raises(cf.FixedPointOverflowError):
                        a.quantize(target, math=policy)
                else:
                    assert a.quantize(target, math=policy).raw.tolist() == expected
        for shift in (largest+1, 64, 1000000000):
            with pytest.raises(ValueError, match="kernel domain"):
                a.quantize(fmt(8, source.precision+shift), math=cf.Math(overflow=overflow))


@pytest.mark.parametrize("mode,overflow", list(itertools.product(MODES, OVERFLOWS)))
def test_complex_random16_oracle_scalar_parity_and_output_conversion(mode, overflow):
    rng = np.random.default_rng(7103)
    aiq = rng.integers(-32768, 32768, size=(64, 2), dtype=np.int16)
    biq = rng.integers(-32768, 32768, size=(64, 2), dtype=np.int16)
    aiq[:3] = [[-32768, -32768], [32767, 32767], [0, 0]]
    biq[:3] = [[-32768, -32768], [-32768, 32767], [0, 0]]
    a = cfa.from_raw(aiq, format=fmt(16, 14), complex=True)
    b = cfa.from_raw(biq, format=fmt(16, 12), complex=True)
    bounded = fmt(16, 11)
    for boundary in ("full", bounded):
        policy = cf.Math(rounding=mode, overflow=overflow, product=boundary, sum=boundary)
        for operation in ("add", "subtract", "multiply"):
            target = full_format(a[0], b[0], operation) if boundary == "full" else boundary
            exact = [calculate(a[index], b[index], operation) for index in range(len(a))]
            try:
                expected = [[oracle_raw(v, target, mode, overflow) for v in pair] for pair in exact]
            except ValueError:
                with pytest.raises(cf.FixedPointOverflowError):
                    getattr(cfa, operation)(a, b, math=policy)
                continue
            result = getattr(cfa, operation)(a, b, math=policy)
            assert result.format == target
            np.testing.assert_array_equal(result.raw, expected)
            for index in range(len(a)):
                scalar = getattr(cf, operation)(a[index], b[index], math=policy)
                assert scalar == result[index] and scalar.format == result.format
            destination = fmt(16, 9)
            out = cfa.empty(a.shape, format=destination, complex=True)
            try:
                encoded = [[oracle_raw(Fraction(raw)*Fraction(2)**-target.precision,
                                       destination, mode, overflow) for raw in pair] for pair in expected]
            except ValueError:
                with pytest.raises(cf.FixedPointOverflowError):
                    getattr(cfa, operation)(a, b, math=policy, out=out)
            else:
                assert getattr(cfa, operation)(a, b, math=policy, out=out) is out
                np.testing.assert_array_equal(out.raw, encoded)


def test_small_complex_exhaustive_and_mixed_signedness_formats():
    for precision in (-1, 1):
        cfmat = fmt(2, precision)
        pairs = list(itertools.product(range(-2, 2), repeat=2))
        a = raw_array(np.array(pairs)[:, None, :], cfmat, True)
        b = raw_array(pairs, cfmat, True)
        for operation in ("add", "subtract", "multiply"):
            result = getattr(cfa, operation)(a, b)
            target = full_format(a[0, 0], b[0], operation)
            assert result.format == target
            for i, j in itertools.product(range(len(pairs)), repeat=2):
                exact = calculate(a[i, 0], b[j], operation)
                assert result.raw[i, j].tolist() == [oracle_raw(v, target, "nearest_even", "raise") for v in exact]
    z = raw_array([[-4, -4], [3, 3]], fmt(3, 2), True)
    for signed, precision in itertools.product((False, True), (-1, 2)):
        rf = fmt(3, precision, signed)
        r = raw_array([rf.min_int, rf.max_int], rf)
        for a, b in ((z, r), (r, z)):
            for operation in ("add", "subtract", "multiply"):
                result = getattr(cfa, operation)(a, b)
                target = full_format(a[0], b[0], operation)
                assert result.format == target
                for index in range(2):
                    exact = calculate(a[index], b[index], operation)
                    assert result.raw[index].tolist() == [oracle_raw(v, target, "nearest_even", "raise") for v in exact]
                    assert result[index] == getattr(cf, operation)(a[index], b[index])


@pytest.mark.parametrize("mode,overflow", list(itertools.product(MODES, OVERFLOWS)))
def test_complex_unary_and_mag2_boundaries(mode, overflow):
    z = raw_array([[-8, -8], [-3, 5], [7, 7]], fmt(4, 2), True)
    for boundary in ("full", fmt(5, 1)):
        policy = cf.Math(rounding=mode, overflow=overflow, sum=boundary, product=fmt(1, 0))
        for name, target in (("negative", fmt(5, 2)), ("conjugate", fmt(5, 2)), ("mag2", fmt(9, 4))):
            target = target if boundary == "full" else boundary
            values = []
            for i, q in z.raw.tolist():
                ar, aq = Fraction(i, 4), Fraction(q, 4)
                values.append((ar*ar+aq*aq,) if name == "mag2" else ((-ar, -aq) if name == "negative" else (ar, -aq)))
            try:
                expected = [[oracle_raw(v, target, mode, overflow) for v in pair] for pair in values]
            except ValueError:
                with pytest.raises(cf.FixedPointOverflowError):
                    getattr(cfa, name)(z, math=policy)
            else:
                result = getattr(cfa, name)(z, math=policy)
                assert result.format == target
                assert result.raw.tolist() == ([p[0] for p in expected] if name == "mag2" else expected)
                for index in range(len(z)):
                    scalar = cf.negative(z[index], math=policy) if name == "negative" else getattr(z[index], name)(math=policy)
                    assert result[index] == scalar
    unsigned = fmt(10, 4, False)
    assert z.mag2(math=cf.Math(sum=unsigned)).format == unsigned
    for function in (cfa.negative, cfa.conjugate):
        with pytest.raises(ValueError, match="signed"):
            function(z, math=cf.Math(sum=unsigned))
    with pytest.raises(TypeError):
        cfa.mag2(z.real)


def test_full_real_int32_product_and_checked_intermediate_rejection():
    a = raw_array([-(2**31), 2**31-1], fmt(32, 16))
    result = a[:, None] * a
    assert result.format == fmt(64, 32)
    assert result.raw.tolist() == [[2**62, -(2**31)*(2**31-1)], [-(2**31)*(2**31-1), (2**31-1)**2]]
    z = raw_array([[0, 0]], fmt(32, 16), True)
    # Domain selection depends on metadata, including zero/empty samples, and
    # happens before execution regardless of a narrow destination or policy.
    for value in (z, cfa.empty((0,), format=z.format, complex=True)):
        out = cfa.empty(value.shape, format=fmt(8), complex=True)
        out.raw.fill(7)
        with pytest.raises(ValueError, match="kernel domain"):
            cfa.multiply(value, value, math=cf.Math(product=fmt(8), overflow="wrap"), out=out)
        assert np.all(out.raw == 7)
    real64 = raw_array([0], fmt(64, 0))
    unsigned63 = raw_array([0], fmt(63, 0, False))
    for call in (lambda: -real64, lambda: abs(real64), lambda: real64 + real64,
                 lambda: real64 * 1, lambda: unsigned63 * 1,
                 lambda: cfa.multiply(real64, real64, math=cf.Math(product=fmt(8))),
                 lambda: z.mag2(), lambda: cfa.array([0], format=fmt(8, 10**9)) + real64,
                 lambda: cfa.add(real64, cf.Fixb(0, format=fmt(100))),
                 lambda: cfa.add(real64, 2**80),
                 lambda: cfa.add(real64, cf.CFixb(0, format=fmt(100)))):
        with pytest.raises(ValueError, match="kernel domain"):
            call()
    assert (+real64).raw.tolist() == [0]
    assert (+unsigned63).raw.tolist() == [0]
    huge_target = fmt(8, 10**9)
    out = cfa.empty(a.shape, format=huge_target)
    out.raw.fill(3)
    with pytest.raises(ValueError, match="kernel domain"):
        cfa.positive(a, out=out)
    assert np.all(out.raw == 3)


def test_comparisons_are_exact_policy_independent_and_need_no_alignment():
    values = [-(2**63), -(2**63)+1, -3, -1, 0, 1, 3, 2**63-2, 2**63-1]
    operations = ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal")
    python_ops = (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge)
    for ap, bp in itertools.product((-1000, -64, -63, 0, 63, 64, 1000), repeat=2):
        a = raw_array(np.array(values)[:, None], fmt(64, ap))
        b = raw_array(values, fmt(64, bp))
        with cf.local_binary_context(math=cf.Math(sum=fmt(1, 0), product=fmt(1, 0), overflow="wrap")):
            for name, op in zip(operations, python_ops):
                expected = [[op(Fraction(x)*Fraction(2)**-ap, Fraction(y)*Fraction(2)**-bp)
                             for y in values] for x in values]
                np.testing.assert_array_equal(getattr(cfa, name)(a, b), expected)
    tiny = raw_array([-1, 0, 1], fmt(8, 10**9))
    np.testing.assert_array_equal(tiny < 0, [True, False, False])
    np.testing.assert_array_equal(tiny > 0, [False, False, True])


def test_rounding_induced_overflow_occurs_after_rounding():
    value = raw_array([15], fmt(8, 1))
    target = fmt(4, 0)
    assert value.quantize(target, math=cf.Math(rounding="floor")).raw.tolist() == [7]
    with pytest.raises(cf.FixedPointOverflowError):
        value.quantize(target)
    assert value.quantize(target, math=cf.Math(overflow="wrap")).raw.tolist() == [-8]
    assert value.quantize(target, math=cf.Math(overflow="saturate")).raw.tolist() == [7]
