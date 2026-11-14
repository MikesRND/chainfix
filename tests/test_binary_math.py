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

"""Exact binary arithmetic checked against independent Fraction oracles."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction
import itertools
import math
import operator
import random
import sys

import pytest

import chainfix as cf

MODES = ("nearest_even", "floor", "ceil", "zero", "nearest_up", "nearest_away")
OVERFLOWS = ("raise", "saturate", "wrap")


def fmt(width=8, precision=2, signed=True):
    return cf.Format(base=2, signed=signed, wordlength=width, precision=precision)


def number(raw, format):
    cls = cf.Fixb if format.signed else cf.Ufixb
    return cls.from_raw(raw, format=format)


def exact(value):
    return Fraction(*value.as_integer_ratio())


def oracle_round(value, mode):
    """Choose between the two surrounding integers by distance, not bit shifts."""
    low, high = math.floor(value), math.ceil(value)
    if mode == "floor":
        return low
    if mode == "ceil":
        return high
    if mode == "zero":
        return int(value)
    distances = {candidate: abs(value - candidate) for candidate in (low, high)}
    candidates = [n for n, distance in distances.items() if distance == min(distances.values())]
    if len(candidates) == 1:
        return candidates[0]
    if mode == "nearest_even":
        return next(n for n in candidates if n % 2 == 0)
    if mode == "nearest_up":
        return max(candidates)
    return max(candidates, key=abs)


def oracle_raw(value, format, rounding, overflow):
    raw = oracle_round(value * Fraction(2) ** format.precision, rounding)
    low = -(2 ** (format.wordlength - 1)) if format.signed else 0
    high = 2 ** (format.wordlength - int(format.signed)) - 1
    if low <= raw <= high:
        return raw
    if overflow == "raise":
        raise ValueError("outside destination")
    if overflow == "saturate":
        return min(high, max(low, raw))
    residue = raw % (2 ** format.wordlength)
    return residue - 2 ** format.wordlength if format.signed and residue > high else residue


def assert_conversion(call, value, format, mode, overflow):
    try:
        expected = oracle_raw(value, format, mode, overflow)
    except ValueError:
        with pytest.raises(cf.FixedPointOverflowError):
            call()
    else:
        result = call()
        assert result.int == expected
        assert result.format == format


@pytest.mark.parametrize("mode,overflow,signed,precision", list(itertools.product(
    MODES, OVERFLOWS, (False, True), (-4, -1, 0, 2, 5))))
def test_construction_and_quantize_against_rational_oracle(mode, overflow, signed, precision):
    target = fmt(4, precision, signed)
    policy = cf.Math(rounding=mode, overflow=overflow)
    cls = cf.Fixb if signed else cf.Ufixb
    values = [Fraction(n, 4) for n in range(-33, 34)] + [Fraction(1, 3), Fraction(-1, 3)]
    for value in values:
        assert_conversion(lambda: cls(value, format=target, math=policy), value,
                          target, mode, overflow)
        if value.denominator in (1, 2, 4):
            source = cf.Fixb(value, 16, 4)
            assert_conversion(lambda: source.quantize(target, math=policy), value,
                              target, mode, overflow)


def test_small_formats_exhaustive_full_arithmetic():
    formats = [fmt(w, p, s) for w, p, s in itertools.product((1, 2, 3), (-1, 0, 2), (False, True))]
    with cf.local_binary_context(math=cf.Math()):
        for aformat, bformat in itertools.product(formats, repeat=2):
            av = [(raw, Fraction(raw) * Fraction(2) ** -aformat.precision)
                  for raw in range(aformat.min_int, aformat.max_int + 1)]
            bv = [(raw, Fraction(raw) * Fraction(2) ** -bformat.precision)
                  for raw in range(bformat.min_int, bformat.max_int + 1)]
            for op in (operator.add, operator.sub, operator.mul):
                signed = aformat.signed or bformat.signed or op is operator.sub
                if op is operator.mul:
                    precision = aformat.precision + bformat.precision
                    width = aformat.wordlength + bformat.wordlength
                else:
                    precision = max(aformat.precision, bformat.precision)
                    possibilities = [op(x, y) * Fraction(2) ** precision
                                     for _, x in av for _, y in bv]
                    width = 1
                    lower, upper = min(possibilities), max(possibilities)
                    while True:
                        minimum = -(2 ** (width - 1)) if signed else 0
                        maximum = 2 ** (width - int(signed)) - 1
                        if minimum <= lower and upper <= maximum:
                            break
                        width += 1
                expected_format = fmt(width, precision, signed)
                for (araw, a), (braw, b) in itertools.product(av, bv):
                    result = op(number(araw, aformat), number(braw, bformat))
                    assert exact(result) == op(a, b)
                    assert result.format == expected_format
                    assert type(result) is (cf.Fixb if signed else cf.Ufixb)


@pytest.mark.parametrize("mode,overflow", list(itertools.product(MODES, OVERFLOWS)))
def test_bounded_operators_match_independent_oracle(mode, overflow):
    target = fmt(3, 1)
    policy = cf.Math(rounding=mode, overflow=overflow, product=target, sum=target)
    sources = [number(raw, fmt(3, p, s)) for p, s in itertools.product((-1, 2), (False, True))
               for raw in range(fmt(3, p, s).min_int, fmt(3, p, s).max_int + 1)]
    with cf.local_binary_context(math=policy):
        for a in sources:
            for op in (operator.neg, operator.abs):
                assert_conversion(lambda: op(a), op(exact(a)), target, mode, overflow)
        for a, b in itertools.product(sources, repeat=2):
            for op in (operator.add, operator.sub, operator.mul):
                value = op(exact(a), exact(b))
                assert_conversion(lambda: op(a, b), value, target, mode, overflow)


@pytest.mark.parametrize("mode", MODES)
def test_float_conversion_uses_exact_ratio(mode):
    target = fmt(1100, 4)
    policy = cf.Math(rounding=mode)
    values = [math.nextafter(0.5, 0), 0.5, math.nextafter(0.5, 1),
              math.nextafter(-0.5, 0), -0.5, math.nextafter(-0.5, -1),
              sys.float_info.max, -sys.float_info.max, math.ulp(0.0)]
    for precision in (-100, 0, 4):
        destination = replace(target, precision=precision)
        for value in values:
            assert cf.Fixb(value, format=destination, math=policy).int == oracle_raw(
                Fraction.from_float(value), destination, mode, "raise")
    assert cf.Fixb(math.ulp(0.0), 3, 1074).int == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("overflow", OVERFLOWS)
def test_nonfinite_is_invalid_even_with_overflow_handling(value, overflow):
    with pytest.raises(ValueError):
        cf.Fixb(value, math=cf.Math(overflow=overflow))


def test_overflow_type_and_valueerror_catch_compatibility():
    assert cf.FixedPointOverflowError.__bases__ == (ValueError,)
    with pytest.raises(cf.FixedPointOverflowError):
        cf.Fixb(1.5, 16, 15)
    caught = False
    try:
        cf.Fixb(1.5, 16, 15)
    except ValueError as error:
        caught = True
        assert isinstance(error, cf.FixedPointOverflowError)
        assert not isinstance(error, OverflowError)
    assert caught
    # Rounding may overflow even when the unrounded value is below the bound.
    with pytest.raises(cf.FixedPointOverflowError):
        cf.Fixb(Fraction(7, 2), 3, 0)
    assert cf.Fixb(Fraction(7, 2), 3, 0, math=cf.Math(overflow="saturate")).int == 3
    assert cf.Fixb(Fraction(7, 2), 3, 0, math=cf.Math(overflow="wrap")).int == -4


@pytest.mark.parametrize("signed", (False, True))
def test_from_raw_is_strict_in_every_context(signed):
    target = fmt(4, -2, signed)
    cls = cf.Fixb if signed else cf.Ufixb
    for overflow in OVERFLOWS:
        with cf.local_binary_context(overflow=overflow):
            for raw in (target.min_int, target.max_int):
                value = cls.from_raw(raw, format=target)
                assert value.int == raw
                assert exact(value) == raw * 4
            for raw in (target.min_int - 1, target.max_int + 1):
                with pytest.raises(cf.FixedPointOverflowError):
                    cls.from_raw(raw, format=target)
    for raw in (True, 1.0, "1", Fraction(1)):
        with pytest.raises(TypeError):
            cls.from_raw(raw, format=target)


def test_exact_large_construction_conversion_and_hashing():
    huge = 2 ** 10000 + 1
    x = cf.Fixb(huge, 10002, 0)
    assert int(x) == huge
    assert x == huge
    assert hash(x) == hash(huge)
    assert x != huge - 1
    with pytest.raises(cf.FixedPointOverflowError):
        cf.Fixb(huge, 8, 0)
    y = cf.Fixb(Fraction(2 ** 80 + 1, 8), 84, 3)
    assert y.as_integer_ratio() == (2 ** 80 + 1, 8)
    assert int(-y) == -int(Fraction(2 ** 80 + 1, 8))
    assert hash(y) == hash(Fraction(2 ** 80 + 1, 8))
    assert cf.Fixb(y, 84, 3).int == y.int
    assert cf.Ufixb(y, 84, 3).int == y.int
    assert cf.Fixb(cf.Fixd(0.125, 16, 3), 16, 3).int == 1
    # Existing decimal inspection is unchanged; binary construction still reads
    # an existing decimal's negative precision exactly from its stored data.
    decimal = cf.Fixd(100, 16, -2)
    assert cf.Fixb(decimal, 16, 0) == 100


@pytest.mark.parametrize("value", [Fraction(-1), Fraction(0), Fraction(1, 2),
                                   Fraction(-3, 8), Fraction(2 ** 60 + 1)])
def test_exact_numeric_comparison_and_hash(value):
    x = cf.Fixb(value, 80, 3)
    equal = [value, cf.Fixb(value, 90, 6), Decimal(value.numerator) / Decimal(value.denominator)]
    if value.denominator == 1:
        equal.append(value.numerator)
    if Fraction(float(value)) == value:
        equal.extend([float(value), complex(float(value), 0)])
    with cf.local_binary_context(math=cf.Math(overflow="wrap", sum=fmt(1, -10))):
        for other in equal:
            assert x == other and other == x
            assert not x != other
            assert hash(x) == hash(other)
            if not isinstance(other, complex):
                assert x <= other and x >= other
        assert x < value + Fraction(1, 128)
        assert x > value - Fraction(1, 128)
    assert {x: "found"}[value] == "found"
    assert x != complex(0, 1)
    assert x != float("nan")
    assert not x < float("nan")
    assert x < float("inf") and x > -float("inf")
    assert x.__eq__(object()) is NotImplemented
    with pytest.raises(TypeError):
        x < 1j


def test_reflected_integers_and_deterministic_constant_formats():
    x = cf.Fixb(Fraction(3, 4), 4, 2)
    for constant in (-9, -1, 0, 1, 8, 2 ** 80):
        for op in (operator.add, operator.sub, operator.mul):
            assert exact(op(x, constant)) == op(Fraction(3, 4), constant)
            assert exact(op(constant, x)) == op(constant, Fraction(3, 4))
    assert (x * 0).wordlength == x.wordlength + 1
    assert (x * -1).wordlength == x.wordlength + 1
    assert (x * 1).wordlength == x.wordlength + 2
    assert (x * 8).wordlength == x.wordlength + 5


@pytest.mark.parametrize("other", [0.5, 1j, Fraction(1, 2), Decimal("0.5"), cf.Fixd(0), object()])
def test_unsupported_arithmetic_does_not_silently_convert(other):
    x = cf.Fixb(0.5)
    for op in (operator.add, operator.sub, operator.mul):
        with pytest.raises(TypeError):
            op(x, other)
        with pytest.raises(TypeError):
            op(other, x)
    assert x.__add__(other) is NotImplemented
    assert x.__sub__(other) is NotImplemented
    assert x.__mul__(other) is NotImplemented
    for op in (cf.add, cf.subtract, cf.multiply):
        with pytest.raises(TypeError):
            op(x, other)


def test_reflected_protocol_is_available_to_other_types():
    class Other:
        def __radd__(self, value):
            return "reflected"
    assert cf.Fixb(0) + Other() == "reflected"


def test_unary_growth_uses_declared_ranges_and_generic_result_types():
    for cls in (cf.Fixb, cf.Fixb32):
        x = cls(0.5, precision=2)
        assert type(+x) is cf.Fixb
        assert (+x).format == x.format
        for result in (-x, abs(x)):
            assert type(result) is cf.Fixb
            assert result.wordlength == x.wordlength + 1
            assert result.precision == x.precision
    minimum = cf.Fixb.from_raw(-32768, format=fmt(16, 14))
    assert (-minimum).int == abs(minimum).int == 32768
    assert (-minimum).format == abs(minimum).format == fmt(17, 14)
    unsigned = cf.Ufixb.from_raw(255, format=fmt(8, 0, False))
    assert abs(unsigned).format == unsigned.format
    assert (-unsigned).format == fmt(9, 0, True)
    target = fmt(16, 14)
    assert cf.negative(minimum, math=cf.Math(sum=target, overflow="saturate")).int == 32767
    assert cf.absolute(minimum, math=cf.Math(sum=target, overflow="wrap")).int == -32768
    assert type(cf.Ufixb32(1, 0) + cf.Ufixb32(1, 0)) is cf.Ufixb
    assert type(cf.Fixb32(1, 0) * cf.Fixb32(1, 0)) is cf.Fixb
    assert cf.positive(3) == 3
    assert cf.negative(3) == -3
    assert cf.absolute(-3) == 3
    for op in (cf.positive, cf.negative, cf.absolute):
        with pytest.raises(TypeError):
            op(0.5)


def test_unary_all_small_raw_values():
    for signed, width, precision in itertools.product((False, True), (1, 2, 3), (-1, 2)):
        format = fmt(width, precision, signed)
        for raw in range(format.min_int, format.max_int + 1):
            x = number(raw, format)
            assert exact(-x) == -exact(x)
            assert exact(abs(x)) == abs(exact(x))


@pytest.mark.parametrize("overflow", OVERFLOWS)
def test_raw_shifts_are_separate_from_exact_scaling(overflow):
    target = fmt(4, 2)
    with cf.local_binary_context(math=cf.Math(rounding="ceil", overflow=overflow,
                                             sum=fmt(2, -1), product=fmt(2, -1))):
        for raw in range(-8, 8):
            x = number(raw, target)
            for count in (0, 1, 3, 4, 1000):
                assert (x >> count).int == raw >> count
                assert (x >> count).format == target
                assert_conversion(lambda: x << count, exact(x) * 2 ** count,
                                  target, "ceil", overflow)
            for exponent in (-100, -3, 0, 3, 100):
                scaled = x.scale_pow2(exponent)
                assert scaled.int == raw
                assert scaled.wordlength == x.wordlength
                assert scaled.precision == x.precision - exponent
                assert exact(scaled) == exact(x) * Fraction(2) ** exponent
    for count in (-1, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            x << count
        with pytest.raises((TypeError, ValueError)):
            x >> count
    with pytest.raises(TypeError):
        x.scale_pow2(True)
    with cf.local_binary_context(overflow="wrap"):
        assert (cf.Fixb(1, 8, 0) << 10 ** 100).int == 0
        assert (cf.Fixb(0) << 10 ** 100).int == 0
        assert (cf.Fixb(-1, 8, 0) >> 10 ** 100).int == -1


def test_no_policy_on_values_and_independent_operation_boundaries():
    target = fmt(8, 2)
    bounded = cf.Math(product=target, sum=fmt(8, 1))
    x = cf.Fixb(Fraction(5, 16), 8, 4)
    with cf.local_binary_context(math=bounded):
        product = x * 1
        assert product.int == 1  # 0.3125 -> 0.25
        assert (product + 0).int == 0  # 0.25 -> 0.0, nearest-even
        assert cf.multiply(x, 1, math=cf.Math()).int == 5
        assert cf.add(x, 0, math=cf.Math()).format.precision == 4
        assert cf.subtract(x, 0, math=cf.Math()).int == 5
        # Constructor targets do not come from product/sum formats.
        assert cf.Fixb(x, 16, 4, math=bounded).int == 5
        assert (x * 2).format == target
    assert x.quantize(fmt(8, 1)).int == 1  # direct conversion gives 0.5
    assert (product + 0).precision == 2  # policy did not travel with product
    for value in (x, product, +cf.Fixb32(0)):
        assert not hasattr(value, "math")
        with pytest.raises(AttributeError):
            value.math = bounded


def test_invalid_construction_and_fixed_width_helpers():
    for cls in (cf.Fixb, cf.Ufixb):
        signed = cls is cf.Fixb
        with pytest.raises(ValueError):
            cls(0, format=fmt(signed=not signed))
        with pytest.raises(ValueError):
            cls(0, 8, format=fmt(signed=signed))
        with pytest.raises(ValueError):
            cls(0, precision=2, format=fmt(signed=signed))
        with pytest.raises(TypeError):
            cls("1")
        with pytest.raises(TypeError):
            cls(0, math="wrap")
        with pytest.raises(ValueError):
            cls(0, format=cf.Format(base=10, signed=signed, wordlength=8, precision=2))
        with pytest.raises(TypeError):
            cls(0, format="q14")
    for cls in (cf.Fixb32, cf.Ufixb32):
        signed = cls is cf.Fixb32
        target = fmt(32, 0, signed)
        assert cls(2 ** 32, format=target, math=cf.Math(overflow="wrap")).int == 0
        assert cls.from_raw(1, format=target).format == target
        with pytest.raises(ValueError):
            cls(0, format=fmt(16, 0, signed))
        with pytest.raises(ValueError):
            cls.from_raw(0, format=fmt(16, 0, signed))
    for call in (lambda: cf.multiply(cf.Fixb(0), 1, math="full"),
                 lambda: cf.Fixb(0).quantize(fmt(), math="full")):
        with pytest.raises(TypeError):
            call()


def test_validated_frozen_descriptors_and_enum_compatibility():
    assert cf.Overflow.SATURATE.value == 1
    assert cf.Overflow.WRAP.value == 2
    assert cf.Math().overflow is cf.Overflow.RAISE
    assert cf.Math(rounding="floor", overflow="wrap").rounding is cf.Rounding.FLOOR
    assert cf.Math(overflow="wrap").overflow is cf.Overflow.WRAP
    for value, field, new_value in ((fmt(), "precision", 3), (cf.Math(), "rounding", "floor")):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, new_value)
    assert replace(fmt(), precision=-4).precision == -4
    for field, value in (("base", True), ("base", 2.0), ("wordlength", True),
                         ("wordlength", 3.0), ("precision", False), ("precision", 1.0),
                         ("signed", 1)):
        with pytest.raises(TypeError):
            replace(fmt(), **{field: value})
    for field, value in (("base", 1), ("wordlength", 0), ("wordlength", -1)):
        with pytest.raises(ValueError):
            replace(fmt(), **{field: value})
    for field in ("product", "sum", "accumulator"):
        for value in ("other", replace(fmt(), base=10)):
            with pytest.raises(ValueError):
                cf.Math(**{field: value})
        with pytest.raises(TypeError):
            cf.Math(**{field: 5})
    for field in ("rounding", "overflow"):
        with pytest.raises(ValueError):
            cf.Math(**{field: "unknown"})
        with pytest.raises(TypeError):
            cf.Math(**{field: 1})


def test_seeded_wider_formats_and_unsigned_bounded_results():
    rng = random.Random(21539)
    for _ in range(250):
        formats = [fmt(rng.choice((16, 32, 64)), rng.choice((-20, -3, 0, 14, 63)),
                       rng.choice((False, True))) for _ in range(2)]
        a, b = [number(rng.randint(f.min_int, f.max_int), f) for f in formats]
        destination = fmt(rng.choice((8, 16, 32)), rng.choice((-2, 0, 5)), False)
        mode, overflow = rng.choice(MODES), rng.choice(OVERFLOWS)
        policy = cf.Math(rounding=mode, overflow=overflow, product=destination, sum=destination)
        for op, function in ((operator.add, cf.add), (operator.sub, cf.subtract),
                             (operator.mul, cf.multiply)):
            value = op(exact(a), exact(b))
            assert exact(op(a, b)) == value
            assert_conversion(lambda: function(a, b, math=policy), value,
                              destination, mode, overflow)


def test_binary_inspection_preserves_existing_surface():
    x = cf.Fixb(Fraction(-3, 2), 16, 2)
    assert x.value == float(x) == -1.5
    assert int(x) == -1
    assert bool(x) and not cf.Fixb(0)
    assert x.bin == "0b1111111111111010"
    assert x.hex == "0xfffa"
    assert x.lsb == 0.25
    assert x.get_current_context() is cf.get_binary_context()
    assert x.base == 2 and x.signed is True
