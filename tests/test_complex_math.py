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

"""Complex scalars checked against independent exact rational calculations."""

from copy import copy, deepcopy
from decimal import Decimal
from fractions import Fraction
import itertools
import math
import operator
import pickle
import random
import sys
from typing import get_args, get_type_hints

import pytest

import chainfix as cf
from chainfix.format import _product_format, _sum_format
from tests.test_binary_math import MODES, OVERFLOWS, exact, fmt, number, oracle_raw


class TaggedComplex(cf.CFixb):
    __slots__ = ("tag", "__dict__")


def pair(value):
    if isinstance(value, cf.CFixb):
        return exact(value.real), exact(value.imag)
    return exact(value), Fraction(0)


def calculate(a, b, operation):
    ar, ai = pair(a)
    br, bi = pair(b)
    if operation == "multiply":
        return ar * br - ai * bi, ar * bi + ai * br
    op = operator.add if operation == "add" else operator.sub
    return op(ar, br), op(ai, bi)


def assert_complex(call, values, target, mode="nearest_even", overflow="raise"):
    try:
        expected = tuple(oracle_raw(v, target, mode, overflow) for v in values)
    except ValueError:
        with pytest.raises(cf.FixedPointOverflowError):
            call()
    else:
        result = call()
        assert type(result) is cf.CFixb
        assert (result.real.int, result.imag.int) == expected
        assert result.format == result.real.format == result.imag.format == target
        assert type(result.real) is type(result.imag) is cf.Fixb


def full_format(a, b, operation):
    """Infer interval bounds independently of Chainfix's format helpers."""
    af, bf = a.format, b.format
    if operation == "multiply":
        guard = int(isinstance(a, cf.CFixb) and isinstance(b, cf.CFixb))
        return fmt(af.wordlength + bf.wordlength + guard, af.precision + bf.precision)
    precision = max(af.precision, bf.precision)
    arange = [Fraction(n) * Fraction(2) ** -af.precision for n in (af.min_int, af.max_int)]
    brange = [Fraction(n) * Fraction(2) ** -bf.precision for n in (bf.min_int, bf.max_int)]
    op = operator.add if operation == "add" else operator.sub
    endpoints = [op(x, y) * Fraction(2) ** precision for x, y in itertools.product(arange, brange)]
    width = 1
    while min(endpoints) < -(2 ** (width - 1)) or max(endpoints) >= 2 ** (width - 1):
        width += 1
    return fmt(width, precision)


def test_construction_forms_and_exact_components():
    target = fmt(16, 14)
    values = [cf.CFixb(0.5 + 0.25j, 16, 14),
              cf.CFixb(0.5 + 0.25j, format=target),
              cf.CFixb(real=Fraction(1, 2), imag=Fraction(1, 4), format=target),
              cf.CFixb.from_raw(i=8192, q=4096, format=target)]
    for z in values:
        assert pair(z) == (Fraction(1, 2), Fraction(1, 4))
        assert complex(z) == 0.5 + 0.25j
        assert z.format == target
    assert cf.CFixb().format == fmt(32, 16)
    assert cf.CFixb() == 0
    assert cf.CFixb(imag=1) == 1j
    assert cf.CFixb(real=1) == 1
    assert cf.CFixb(True) == 1
    assert cf.CFixb(real=True, imag=False) == 1
    assert cf.CFixb(real=False, imag=True) == 1j
    for real in (3, Fraction(3, 8), 0.375, cf.Fixb(0.375), cf.Ufixb(0.375), cf.Fixd(0.375)):
        assert cf.CFixb(real).real == (exact(real) if hasattr(real, "as_integer_ratio") else real)
        assert cf.CFixb(real).imag == 0
    huge = 2 ** 10000 + 1
    z = cf.CFixb(real=huge, imag=Fraction(-huge, 8), format=fmt(10005, 3))
    assert pair(z) == (huge, Fraction(-huge, 8))
    copy = cf.CFixb(z, format=fmt(10008, 6))
    assert copy == z
    assert copy.real.int == z.real.int * 8
    assert copy.imag.int == z.imag.int * 8
    tiny = cf.CFixb(real=Fraction(1, 2 ** 2000), imag=Fraction(-3, 2 ** 2000),
                    format=fmt(4, 2000))
    assert (tiny.real.int, tiny.imag.int) == (1, -3)
    assert cf.CFixb(cf.CFixb(12 + 4j, 4, -2), 8, 0) == 12 + 4j


@pytest.mark.parametrize("mode,overflow,precision", list(itertools.product(MODES, OVERFLOWS, (-3, 0, 3))))
def test_component_conversion_and_quantize_oracle(mode, overflow, precision):
    target = fmt(5, precision)
    policy = cf.Math(rounding=mode, overflow=overflow)
    values = [Fraction(n, 8) for n in range(-67, 68)] + [Fraction(1, 3), Fraction(-1, 3)]
    for real in values:
        imag = -real * 3
        assert_complex(lambda: cf.CFixb(real=real, imag=imag, format=target, math=policy),
                       (real, imag), target, mode, overflow)
        if real.denominator != 3:
            source = cf.CFixb(real=real, imag=imag, format=fmt(12, 3))
            assert_complex(lambda: source.quantize(target, math=policy),
                           (real, imag), target, mode, overflow)
            assert_complex(lambda: cf.CFixb(source, format=target, math=policy),
                           (real, imag), target, mode, overflow)


@pytest.mark.parametrize("mode", MODES)
def test_float_components_use_exact_ratios(mode):
    target = fmt(1100, 0)
    values = [math.nextafter(0.5, 0), 0.5, math.nextafter(0.5, 1),
              sys.float_info.max, math.ulp(0.0)]
    for value in values:
        assert_complex(lambda: cf.CFixb(complex(value, -value), format=target,
                                        math=cf.Math(rounding=mode)),
                       (Fraction(value), Fraction(-value)), target, mode)
    assert cf.CFixb(complex(math.ulp(0.0), 0), 3, 1074).real.int == 1


@pytest.mark.parametrize("kwargs", [dict(value=0, real=0), dict(value=0, imag=0),
                                    dict(value=1j, real=0, imag=0)])
def test_component_argument_conflicts(kwargs):
    with pytest.raises(ValueError, match="value cannot"):
        cf.CFixb(**kwargs)


def test_format_and_policy_validation():
    z = cf.CFixb(0)
    for dimensions in (dict(wordlength=8), dict(precision=0), dict(wordlength=8, precision=0)):
        with pytest.raises(ValueError, match="format cannot"):
            cf.CFixb(format=fmt(), **dimensions)
    for bad, error in [(fmt(signed=False), ValueError),
                       (cf.Format(base=10, signed=True, wordlength=8, precision=2), ValueError),
                       ("invalid", TypeError)]:
        for call in (lambda: cf.CFixb(format=bad),
                     lambda: cf.CFixb.from_raw(i=0, q=0, format=bad),
                     lambda: z.quantize(bad)):
            with pytest.raises(error):
                call()
    for kwargs, error in [(dict(wordlength=True), TypeError), (dict(precision=True), TypeError),
                          (dict(wordlength=0), ValueError), (dict(precision=1.5), TypeError)]:
        with pytest.raises(error):
            cf.CFixb(**kwargs)
    for call in (lambda: cf.CFixb(math="wrap"), lambda: z.quantize(fmt(), math="wrap"),
                 lambda: z.conjugate(math="wrap"), lambda: z.mag2(math="wrap"),
                 lambda: cf.add(z, z, math="wrap"), lambda: cf.negative(z, math="wrap")):
        with pytest.raises(TypeError, match="math must"):
            call()
    with pytest.raises(TypeError):
        cf.CFixb.from_raw(0, 0, format=fmt())


@pytest.mark.parametrize("overflow", OVERFLOWS)
def test_nonfinite_decimal_and_raw_boundaries(overflow):
    policy = cf.Math(overflow=overflow)
    with cf.local_binary_context(math=policy):
        for value in (float("nan"), float("inf"), -float("inf")):
            for call in (lambda: cf.CFixb(value), lambda: cf.CFixb(imag=value),
                         lambda: cf.CFixb(complex(value, 0)), lambda: cf.CFixb(complex(0, value))):
                with pytest.raises(ValueError, match="finite"):
                    call()
        for value in (Decimal("0.5"), Decimal("NaN"), Decimal("Infinity"), "1", object()):
            for kwargs in (dict(value=value), dict(real=value), dict(imag=value)):
                with pytest.raises(TypeError):
                    cf.CFixb(**kwargs)
        for i, q in itertools.product((-8, 7), repeat=2):
            assert pair(cf.CFixb.from_raw(i=i, q=q, format=fmt(4, -2))) == (i * 4, q * 4)
        for bad in (-9, 8, True, 1.0, Fraction(1), "1"):
            error = cf.FixedPointOverflowError if type(bad) is int else TypeError
            for i, q in ((bad, 0), (0, bad)):
                with pytest.raises(error):
                    cf.CFixb.from_raw(i=i, q=q, format=fmt(4, -2))
    with pytest.raises(cf.FixedPointOverflowError):
        cf.CFixb(1.5j, 16, 15)
    assert cf.CFixb(real=Fraction(Decimal("0.5"))) == Decimal("0.5")


def test_small_formats_exhaustive_complex_arithmetic():
    formats = [fmt(w, p) for w, p in itertools.product((1, 2), (-1, 1))]
    for af, bf in itertools.product(formats, repeat=2):
        av = [cf.CFixb.from_raw(i=i, q=q, format=af)
              for i, q in itertools.product(range(af.min_int, af.max_int + 1), repeat=2)]
        bv = [cf.CFixb.from_raw(i=i, q=q, format=bf)
              for i, q in itertools.product(range(bf.min_int, bf.max_int + 1), repeat=2)]
        for a, b in itertools.product(av, bv):
            for operation, op in (("add", operator.add), ("subtract", operator.sub),
                                  ("multiply", operator.mul)):
                expected = calculate(a, b, operation)
                target = full_format(a, b, operation)
                assert_complex(lambda: op(a, b), expected, target)
                assert_complex(lambda: getattr(cf, operation)(a, b, math=cf.Math()), expected, target)


@pytest.mark.parametrize("signed,cp,rp", list(itertools.product((False, True), (-2, 0, 3), (-1, 0, 2))))
def test_mixed_signedness_formats_and_subtraction_order(signed, cp, rp):
    cformat, rformat = fmt(8, cp), fmt(8, rp, signed)
    for cr, ci, rr in itertools.product((-128, -1, 0, 127), (-128, 0, 127),
                                       (rformat.min_int, 0, rformat.max_int)):
        z = cf.CFixb.from_raw(i=cr, q=ci, format=cformat)
        real = number(rr, rformat)
        for a, b in ((z, real), (real, z)):
            for operation, op in (("add", operator.add), ("subtract", operator.sub),
                                  ("multiply", operator.mul)):
                result = op(a, b)
                target = full_format(a, b, operation)
                helper = (_product_format(a.format, b.format) if operation == "multiply" else
                          _sum_format(a.format, b.format, subtract=operation == "subtract"))
                assert result.format == target == helper
                assert pair(result) == calculate(a, b, operation)
                assert getattr(cf, operation)(a, b, math=cf.Math()) == result
                if operation == "multiply":
                    assert result.real.format == (z.real * real).format
                    assert result.imag.format == (z.imag * real).format
    z, u = cf.CFixb(0, 8, 2), cf.Ufixb(0, 8, 2)
    assert (z * u).format == (u * z).format == fmt(16, 4)
    assert (z + u).format == (u + z).format == fmt(10, 2)


@pytest.mark.parametrize("mode,overflow", list(itertools.product(MODES, OVERFLOWS)))
def test_bounded_arithmetic_and_compound_operations(mode, overflow):
    target = fmt(4, 1)
    policy = cf.Math(rounding=mode, overflow=overflow, sum=target, product=target)
    rng = random.Random(20261113)
    for _ in range(80):
        z, w = [cf.CFixb.from_raw(i=rng.randrange(-8, 8), q=rng.randrange(-8, 8),
                                  format=fmt(4, rng.choice((-1, 2)))) for _ in range(2)]
        real = number(rng.randrange(16), fmt(4, 2, False))
        with cf.local_binary_context(math=policy):
            for a, b in ((z, w), (z, real), (real, z), (z, True), (False, z), (2, z), (z, -3)):
                # Integer constants are exact real numbers in the independent oracle.
                oa = cf.Fixb(a, 4, 0) if isinstance(a, int) else a
                ob = cf.Fixb(b, 4, 0) if isinstance(b, int) else b
                for operation, op in (("add", operator.add), ("subtract", operator.sub),
                                      ("multiply", operator.mul)):
                    expected = calculate(oa, ob, operation)
                    assert_complex(lambda: op(a, b), expected, target, mode, overflow)
                    assert_complex(lambda: getattr(cf, operation)(a, b, math=policy),
                                   expected, target, mode, overflow)
            r, i = pair(z)
            for call, expected in ((lambda: -z, (-r, -i)),
                                   (lambda: cf.negative(z, math=policy), (-r, -i)),
                                   (lambda: z.conjugate(), (r, -i))):
                assert_complex(call, expected, target, mode, overflow)
            try:
                expected = oracle_raw(r * r + i * i, target, mode, overflow)
            except ValueError:
                with pytest.raises(cf.FixedPointOverflowError):
                    z.mag2()
            else:
                power = z.mag2()
                assert type(power) is cf.Fixb
                assert power.int == expected and power.format == target


def test_random_16_bit_iq_and_wide_intermediates():
    rng = random.Random(68153)
    for width in (16, 80):
        for _ in range(150):
            a, b = [cf.CFixb.from_raw(i=rng.randrange(-(2 ** (width - 1)), 2 ** (width - 1)),
                                      q=rng.randrange(-(2 ** (width - 1)), 2 ** (width - 1)),
                                      format=fmt(width, rng.choice((-2, 0, width - 2))))
                    for _ in range(2)]
            target = fmt(16, rng.choice((-2, 0, 14)))
            mode, overflow = rng.choice(MODES), rng.choice(OVERFLOWS)
            policy = cf.Math(rounding=mode, overflow=overflow, product=target, sum=target)
            for operation in ("add", "subtract", "multiply"):
                expected = calculate(a, b, operation)
                assert_complex(lambda: getattr(cf, operation)(a, b), expected,
                               full_format(a, b, operation))
                assert_complex(lambda: getattr(cf, operation)(a, b, math=policy),
                               expected, target, mode, overflow)


def test_full_guard_bits_minima_and_container_shape():
    for raw in (0, -128, 127):
        z = cf.CFixb.from_raw(i=raw, q=raw, format=fmt(8, 3))
        assert_complex(lambda: -z, tuple(-x for x in pair(z)), fmt(9, 3))
        assert_complex(z.conjugate, (pair(z)[0], -pair(z)[1]), fmt(9, 3))
        assert z.mag2().format == fmt(17, 6)
        assert exact(z.mag2()) == sum(x * x for x in pair(z))
        assert (z * z).format == fmt(17, 6)
    z = cf.CFixb(1, 8, 2)
    assert (z * cf.CFixb(2, 8, 2)).format == fmt(17, 4)
    assert (z * cf.Fixb(2, 8, 2)).format == fmt(16, 4)
    for integer, width in ((False, 1), (True, 2), (0, 1), (2, 3), (-1, 1), (-3, 3)):
        for result in (z * integer, integer * z):
            assert result.format == fmt(8 + width, 2)
            assert result == integer


def test_compound_operations_have_one_boundary_and_unsigned_mag2():
    z = cf.CFixb(0.5 + 0.5j, 4, 1)
    target = fmt(4, 0)
    policy = cf.Math(product=target, sum=fmt(4, 1))
    # Rounding the partial products separately would turn both quarters into zero.
    assert cf.multiply(z, z, math=cf.Math(product=fmt(4, 1))) == 0.5j
    assert z.mag2(math=policy) == Fraction(1, 2)
    # mag2 ignores product entirely, even if partial products would overflow it.
    assert cf.CFixb(2 + 2j).mag2(math=cf.Math(product=fmt(1, 0))) == 8
    unsigned = fmt(8, 1, False)
    power = z.mag2(math=cf.Math(sum=unsigned, product=fmt(1, 0)))
    assert type(power) is cf.Ufixb and power.format == unsigned and power == Fraction(1, 2)
    bad = cf.Math(sum=unsigned, product=unsigned)
    for call in (lambda: cf.add(z, 1, math=bad), lambda: cf.subtract(1, z, math=bad),
                 lambda: cf.multiply(z, z, math=bad), lambda: cf.multiply(z, 2, math=bad),
                 lambda: cf.negative(z, math=bad), lambda: z.conjugate(math=bad)):
        with pytest.raises(ValueError, match="signed"):
            call()
    with cf.local_binary_context(math=bad):
        assert (+z).format == z.format
        assert cf.positive(z) == z


def test_context_overrides_views_and_result_policy_independence():
    z = cf.CFixb(0.75 + 0.5j, 8, 2)
    full = cf.Math()
    bounded = cf.Math(rounding="floor", overflow="wrap", sum=fmt(3, 0), product=fmt(3, 0))
    original = cf.get_binary_context()
    with cf.local_binary_context(wordlength=10, precision=3, math=bounded):
        assert cf.CFixb().format == fmt(10, 3)
        assert cf.CFixb(wordlength=12).format == fmt(12, 3)
        assert cf.CFixb(precision=-2).format == fmt(10, -2)
        assert cf.multiply(z, z, math=full) == Fraction(5, 16) + 0.75j
        assert (z * z) == 0
        assert cf.add(z, z, math=full) == 1.5 + 1j
        assert cf.subtract(z, 1, math=full) == -0.25 + 0.5j
        assert cf.negative(z, math=full) == -0.75 - 0.5j
        assert z.conjugate(math=full) == 0.75 - 0.5j
        assert z.mag2(math=full) == Fraction(13, 16)
        assert z.quantize(fmt(8, 1), math=full) == 1 + 0.5j
        assert cf.CFixb(z, format=fmt(8, 1), math=full) == 1 + 0.5j
        view = z.real
        result = cf.multiply(z, z, math=full)
        assert view == 0.75 and view.format == z.format
        assert (+z).format == cf.positive(z).format == z.format
        assert not hasattr(z, "math") and not hasattr(view, "math") and not hasattr(z, "int")
        assert result + 0 == 0
    assert cf.get_binary_context() is original
    assert result + 0 == result
    assert view == z.real
    # Expression boundaries retain the rounded product before the final sum.
    with cf.local_binary_context(math=cf.Math(product=fmt(8, 2), sum=fmt(8, 1))):
        x = cf.CFixb(real=Fraction(5, 16), imag=Fraction(-5, 16), format=fmt(8, 4))
        assert x * 1 + 0 == 0
        assert x.quantize(fmt(8, 1)) == 0.5 - 0.5j


@pytest.mark.parametrize("other", [1.0, 1j, Fraction(1), Decimal(1), cf.Fixd(1), object()])
def test_unsupported_operands_and_reflection(other):
    z = cf.CFixb(1j)
    for method in (z.__add__, z.__radd__, z.__sub__, z.__rsub__, z.__mul__, z.__rmul__):
        assert method(other) is NotImplemented
    for a, b in ((z, other), (other, z)):
        for op in (operator.add, operator.sub, operator.mul, cf.add, cf.subtract, cf.multiply):
            with pytest.raises(TypeError):
                op(a, b)


def test_unsupported_operations_and_custom_reflection():
    z = cf.CFixb(1 + 1j)
    for op in (operator.lt, operator.le, operator.gt, operator.ge, operator.truediv,
               operator.floordiv, operator.lshift, operator.rshift):
        for a, b in ((z, z), (z, 1), (1, z)):
            with pytest.raises(TypeError):
                op(a, b)
    for op in (abs, cf.absolute, float, int):
        with pytest.raises(TypeError):
            op(z)
    with pytest.raises(TypeError):
        cf.positive(z, math=cf.Math())
    class Reflected:
        def __radd__(self, other):
            return "reflected"

        def __eq__(self, other):
            return True

    assert z + Reflected() == "reflected"
    assert z.__eq__(object()) is NotImplemented
    assert z == Reflected()
    assert z != object()


def test_exact_equality_hash_and_nonfinite_comparisons():
    values = (Fraction(0), Fraction(-1), Fraction(1, 2), Fraction(-3, 8), Fraction(2 ** 100 + 1))
    for r, i in itertools.product(values, repeat=2):
        z = cf.CFixb(real=r, imag=i, format=fmt(110, 3))
        equal = cf.CFixb(real=r, imag=i, format=fmt(120, 5))
        assert z == equal and not z != equal and hash(z) == hash(equal)
        assert z != cf.CFixb(real=r, imag=i + Fraction(1, 8), format=fmt(120, 5))
        combined = hash(r) + sys.hash_info.imag * hash(i)
        limit = 2 ** (sys.hash_info.width - 1)
        expected = combined % (2 * limit)
        if expected >= limit:
            expected -= 2 * limit
        assert hash(z) == (-2 if expected == -1 else expected)
        if Fraction(float(r)) == r and Fraction(float(i)) == i:
            native = complex(float(r), float(i))
            assert z == native and native == z and hash(z) == hash(native)
        if i == 0:
            for other in (r, cf.Fixb(r, 110, 3), Decimal(r.numerator) / Decimal(r.denominator)):
                # Use Decimal only where the default decimal context retains the value.
                if other == r:
                    assert z == other and other == z and hash(z) == hash(other)
                    assert {z: "found"}[other] == "found"
            if r.denominator == 1:
                assert z == int(r) and int(r) == z and hash(z) == hash(int(r))
        else:
            assert z != r
        for nonfinite in (float("nan"), float("inf"), -float("inf")):
            for other in (nonfinite, complex(nonfinite, 0), complex(0, nonfinite)):
                assert not z == other and not other == z
                assert z != other and other != z
        for other in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            assert z != other and other != z
    assert hash(cf.CFixb(real=2 * sys.hash_info.imag - 1, imag=-1, format=fmt(64, 0))) == -2
    assert cf.CFixb(True) == True and True == cf.CFixb(True)
    assert cf.CFixb(1) == cf.Ufixb(1) and cf.Ufixb(1) == cf.CFixb(1)
    assert cf.CFixb(Fraction(2 ** 60 + 1, 8), 64, 3) != complex(float(Fraction(2 ** 60 + 1, 8)))


def test_truthiness_repr_and_generic_results_without_context_lookup(monkeypatch):
    for r, i in ((0, 0), (1, 0), (0, 1), (1, -1)):
        assert bool(cf.CFixb(real=r, imag=i)) is bool(r or i)
    class Derived(cf.CFixb):
        pass

    z = Derived(real=Fraction(2 ** 80 + 1, 8), imag=Fraction(-2 ** 81 + 3, 8), format=fmt(85, 3))
    for result in (+z, cf.positive(z), -z, z + 0, z * 1, z.conjugate(), z.quantize(z.format)):
        assert type(result) is cf.CFixb
    assert type(z.mag2()) is cf.Fixb
    assert type(z.real) is type(z.imag) is cf.Fixb
    assert type(cf.CFixb(1j) + cf.Fixb32(1)) is cf.CFixb
    def reject_context():
        raise AssertionError("value inspection must not consult policy")

    monkeypatch.setattr("chainfix.complex.get_binary_context", reject_context)
    monkeypatch.setattr("chainfix.binary.get_binary_context", reject_context)
    assert z.real.int == 2 ** 80 + 1 and z.imag.int == -2 ** 81 + 3
    assert bool(z)
    assert +z == z and cf.positive(z) == z
    assert hash(z) == hash(+z)
    expected = (f"CFixb.from_raw(i={2 ** 80 + 1}, q={-2 ** 81 + 3}, "
                "format=Format(base=2, signed=True, wordlength=85, precision=3))")
    assert repr(z) == expected
    restored = eval(repr(z), {"CFixb": cf.CFixb, "Format": cf.Format})
    assert restored == z and restored.format == z.format


@pytest.mark.parametrize("name", ("add", "subtract", "multiply", "positive", "negative"))
def test_public_runtime_return_annotations(name):
    hints = get_type_hints(getattr(cf, name))
    assert set(get_args(hints["return"])) == {cf.Fixb, cf.Ufixb, cf.CFixb}


@pytest.mark.parametrize("real_type", (cf.Fixb, cf.Ufixb))
@pytest.mark.parametrize("operation,op", [("add", operator.add), ("subtract", operator.sub),
                                        ("multiply", operator.mul)])
def test_reflected_mixed_operators_and_explicit_policy(real_type, operation, op):
    real = real_type(Fraction(3, 4), 8, 2)
    z = cf.CFixb(real=Fraction(5, 8), imag=Fraction(-3, 8), format=fmt(8, 3))
    target = fmt(5, 1)
    explicit = cf.Math(rounding="nearest_away", product=target, sum=target)
    ambient = cf.Math(rounding="floor", product=fmt(4, 0), sum=fmt(4, 0))
    method = {"add": "__add__", "subtract": "__sub__", "multiply": "__mul__"}[operation]
    # The real method leaves complex operands to Python's reflected dispatch.
    assert getattr(real, method)(z) is NotImplemented
    with cf.local_binary_context(math=ambient):
        for a, b in ((real, z), (z, real)):
            expected = calculate(a, b, operation)
            assert_complex(lambda: op(a, b), expected, fmt(4, 0), "floor")
            assert_complex(lambda: getattr(cf, operation)(a, b, math=explicit),
                           expected, target, "nearest_away")
            assert_complex(lambda: getattr(cf, operation)(a, b, math=cf.Math()),
                           expected, full_format(a, b, operation))
    with pytest.raises(TypeError):
        op(real, object())
    with pytest.raises(TypeError):
        getattr(cf, operation)(real, object())


@pytest.mark.parametrize("construct", [lambda: cf.CFixb(0.5 + 0.25j),
                                      lambda: cf.CFixb(real=Fraction(1, 2), imag=Fraction(1, 4)),
                                      lambda: cf.CFixb.from_raw(i=2, q=1, format=fmt(4, 2))])
def test_complex_storage_cannot_change_a_dictionary_key(construct):
    z = construct()
    original = (z.real.int, z.imag.int, z.format, hash(z))
    lookup, members = {z: "found"}, {z}
    for name in ("_i", "_q", "_format", "real", "imag", "format"):
        with pytest.raises(AttributeError):
            setattr(z, name, fmt(4, 0) if "format" in name else 5)
        with pytest.raises(AttributeError):
            delattr(z, name)
        assert (z.real.int, z.imag.int, z.format, hash(z)) == original
        assert lookup[+z] == "found" and +z in members
    assert not hasattr(z, "__dict__")
    # Real components are independent values, not mutable access to complex storage.
    component = z.real
    component._int = 0
    assert z.real == Fraction(1, 2)


@pytest.mark.parametrize("cls", (cf.CFixb, TaggedComplex))
def test_copy_and_pickle_preserve_exact_storage_without_context(cls, monkeypatch):
    z = cls.from_raw(i=2 ** 100 + 1, q=-2 ** 100 + 3, format=fmt(104, -3))
    if cls is TaggedComplex:
        z.tag, z.metadata = ["slot"], ["dictionary"]
        z.temporary = True
        del z.temporary
        for name in ("_i", "_q", "_format"):
            with pytest.raises(AttributeError):
                setattr(z, name, 0)
            with pytest.raises(AttributeError):
                delattr(z, name)

    def reject_context():
        raise AssertionError("copying raw storage must not consult arithmetic policy")

    monkeypatch.setattr("chainfix.complex.get_binary_context", reject_context)
    monkeypatch.setattr("chainfix.binary.get_binary_context", reject_context)
    shallow, deep = copy(z), deepcopy(z)
    results = [shallow, deep] + [pickle.loads(pickle.dumps(z, protocol))
                                 for protocol in range(pickle.HIGHEST_PROTOCOL + 1)]
    for restored in results:
        assert type(restored) is cls
        assert restored == z and hash(restored) == hash(z)
        assert restored.format == z.format
        assert (restored.real.int, restored.imag.int) == (z.real.int, z.imag.int)
        with pytest.raises(AttributeError):
            restored._i = 0
        if cls is TaggedComplex:
            assert restored.tag == z.tag and restored.metadata == z.metadata
    if cls is TaggedComplex:
        assert shallow.tag is z.tag and shallow.metadata is z.metadata
        assert deep.tag is not z.tag and deep.metadata is not z.metadata
