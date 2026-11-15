"""Exact decimal construction, its context policy, and cross-base comparison."""

import copy
from contextvars import Context
import decimal
from decimal import Decimal
from fractions import Fraction
import operator
import pickle
import random

import pytest

import chainfix as cf
from tests.test_binary_math import MODES, OVERFLOWS, assert_conversion

# Python's decimal module has no counterpart for nearest_up (ties toward +inf).
DECIMAL_ROUNDING = {"nearest_even": decimal.ROUND_HALF_EVEN, "floor": decimal.ROUND_FLOOR,
                    "ceil": decimal.ROUND_CEILING, "zero": decimal.ROUND_DOWN,
                    "nearest_away": decimal.ROUND_HALF_UP}
# Wide enough that quantize never raises InvalidOperation for large coefficients.
WIDE = decimal.Context(prec=decimal.MAX_PREC, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)


def dfmt(width=16, precision=2, signed=True):
    return cf.Format(base=10, signed=signed, wordlength=width, precision=precision)


def decimal_raw(value, precision, mode):
    """The stored integer Decimal(value).quantize(10**-precision) implies."""
    # Build the quantum from a tuple: Decimal(10) ** 2 has exponent 0, not 2.
    quantum = Decimal((0, (1,), -precision))
    with decimal.localcontext(WIDE):
        return int(Decimal(value).quantize(quantum, rounding=mode).scaleb(precision))


def test_float_conversion_follows_python_decimal():
    assert cf.Fixd(2.675, 32, 2).int == 267
    assert cf.Fixd(67.685, 32, 2).int == 6769
    assert cf.Fixd(0.1).int == 10 ** 17 + 6
    # Exact decimal literals come from Fraction, as they come from strings for Decimal.
    assert cf.Fixd(Fraction("2.675"), 32, 2).int == 268
    assert cf.Fixd(Fraction("67.685"), 32, 2).int == 6768
    assert cf.Fixd(Fraction("0.1")).int == 10 ** 17
    with cf.local_decimal_context(rounding="nearest_away"):
        assert cf.Fixd(Fraction("67.685"), 32, 2).int == 6769
    for value, precision in ((2.675, 2), (67.685, 2), (0.1, 18), (-0.1, 18)):
        assert cf.Fixd(value, 128, precision).int == decimal_raw(
            value, precision, decimal.ROUND_HALF_EVEN)


@pytest.mark.parametrize("mode", sorted(DECIMAL_ROUNDING))
def test_randomized_floats_match_decimal_quantize(mode):
    rng = random.Random(f"chainfix-decimal-{mode}")
    math = cf.Math(rounding=mode)
    for _ in range(500):
        precision = rng.randint(-4, 12)
        value = rng.choice([
            rng.uniform(-1000, 1000),
            round(rng.uniform(-100, 100), rng.randint(0, 6)),  # near decimal ties
            rng.uniform(-1e15, 1e15),
            rng.randint(-10 ** 6, 10 ** 6) + 0.5,
        ])
        expected = decimal_raw(value, precision, DECIMAL_ROUNDING[mode])
        assert cf.Fixd(value, format=dfmt(256, precision), math=math).int == expected


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("overflow", OVERFLOWS)
def test_conversion_matches_the_rational_oracle(mode, overflow):
    rng = random.Random(f"chainfix-oracle-{mode}-{overflow}")
    math = cf.Math(rounding=mode, overflow=overflow)
    for _ in range(150):
        format = dfmt(rng.choice([4, 8, 12]), rng.randint(-2, 3), rng.random() < 0.7)
        cls = cf.Fixd if format.signed else cf.Ufixd
        value = Fraction(rng.randint(-5000, 5000), rng.choice([1, 2, 3, 7, 8, 10, 40, 1000]))
        assert_conversion(lambda: cls(value, format=format, math=math), value, format,
                          mode, overflow)


def test_negative_precision():
    assert cf.Fixd(123.4, 16, -2).int == 1
    assert cf.Fixd(150, 16, -2).int == 2
    assert cf.Fixd(250, 16, -2).int == 2
    assert cf.Fixd(-250, 16, -2).int == -2
    with cf.local_decimal_context(rounding="nearest_away"):
        assert cf.Fixd(250, 16, -2).int == 3
    assert cf.Fixd(12749, 8, -2).int == 127
    with pytest.raises(cf.FixedPointOverflowError):
        cf.Fixd(12850, 8, -2)
    for overflow, expected in (("saturate", 127), ("wrap", -128)):
        with cf.local_decimal_context(overflow=overflow):
            assert cf.Fixd(12850, 8, -2).int == expected
    limit = cf.Fixd(0, 8, -2)
    assert (limit.lower_bound, limit.upper_bound) == (-12800, 12700)


def test_decimal_context_policy_and_explicit_math():
    with cf.local_decimal_context(rounding="floor", overflow="wrap", wordlength=8, precision=1):
        assert cf.Fixd(1.99).int == 19
        assert cf.Fixd(12.8).int == -128
        assert cf.Fixd(1.99, math=cf.Math()).int == 20
        with pytest.raises(cf.FixedPointOverflowError):
            cf.Fixd(12.8, math=cf.Math())
    assert cf.Fixd(1.99, 8, 1).int == 20
    # Each base reads only its own context.
    with cf.local_binary_context(rounding="floor", overflow="saturate"):
        assert cf.Fixd(1.99, 8, 1).int == 20
        with pytest.raises(cf.FixedPointOverflowError):
            cf.Fixd(1000, 8, 0)
    with cf.local_decimal_context(rounding="floor", overflow="saturate"):
        assert cf.Fixb(1.75, 8, 1).int == 4
        with pytest.raises(cf.FixedPointOverflowError):
            cf.Fixb(1000, 8, 0)
    # An explicit policy follows the decimal context's rule: no result boundaries.
    for name in ("product", "sum", "accumulator"):
        bounded = cf.Math(**{name: cf.Format(base=2, signed=True, wordlength=8, precision=2)})
        for build in (lambda: cf.Fixd(1, 8, 1, math=bounded), lambda: cf.Ufixd32(1, 1, math=bounded),
                      lambda: cf.Ufixd(1, format=dfmt(8, 1, False), math=bounded)):
            with pytest.raises(ValueError, match=f"{name}='full'"):
                build()
        assert cf.Fixb(1, 8, 1, math=bounded).int == 2  # construction uses only modes


def test_explicit_format_and_math_never_consult_the_context(monkeypatch):
    def reject_context():
        raise AssertionError("must not consult the decimal context")

    monkeypatch.setattr("chainfix.decimal.get_decimal_context", reject_context)
    format = dfmt(32, 3)
    x = cf.Fixd(Fraction(-5, 8), format=format, math=cf.Math())
    assert x.int == -625
    assert cf.Fixd32(1.5, format=dfmt(32, 1), math=cf.Math()).int == 15
    assert cf.Fixd.from_raw(-625, format=format) == x
    assert x.as_integer_ratio() == (-5, 8)
    assert (x.value, float(x), int(x)) == (-0.625, -0.625, 0)
    assert (x.hex, repr(x), str(x)) == ("0xfffffd8f", "Fixd(-0.625, 32, 3)", "-0.625")
    assert hash(x) == hash(Fraction(-5, 8))
    assert x == Fraction(-5, 8) and x < 0 and bool(x)
    assert x.format == format
    with pytest.raises(AssertionError):
        cf.Fixd(0)
    with pytest.raises(AssertionError):
        cf.Fixd(0, format=format)


def test_decimal_validation():
    for cls in (cf.Fixd, cf.Ufixd):
        signed = cls is cf.Fixd
        with pytest.raises(ValueError):
            cls(0, format=dfmt(signed=not signed))
        with pytest.raises(ValueError):
            cls(0, 8, format=dfmt(signed=signed))
        with pytest.raises(ValueError, match="base-10"):
            cls(0, format=cf.Format(base=2, signed=signed, wordlength=8, precision=2))
        with pytest.raises(TypeError):
            cls(0, format="d2")
        with pytest.raises(TypeError):
            cls(0, math="wrap")
        for kwargs in ({"wordlength": 0}, {"wordlength": True}, {"precision": 1.5}):
            with pytest.raises((TypeError, ValueError)):
                cls(0, **kwargs)
        for value in ("1", Decimal("1"), None, 1j):
            with pytest.raises(TypeError):
                cls(value)
        for value in (float("nan"), float("inf")):
            with pytest.raises(ValueError):
                cls(value)
        with pytest.raises(cf.FixedPointOverflowError):
            cls.from_raw(2 ** 16, format=dfmt(16, 2, signed))
        with pytest.raises(TypeError):
            cls.from_raw(True, format=dfmt(16, 2, signed))
    with pytest.raises(ValueError, match="base-2"):
        cf.Fixb(0, format=dfmt())
    assert cf.Fixd(True, 8, 1).int == 10


def test_exact_equality_ordering_and_hash_across_bases():
    halves = [cf.Fixd(0.5, 8, 1), cf.Ufixd(0.5, 8, 1), cf.Fixb(0.5, 8, 4), cf.Ufixb(0.5, 8, 4),
              cf.CFixb(0.5, 8, 4), Fraction(1, 2), 0.5, Decimal("0.5")]
    for a in halves:
        for b in halves:
            assert a == b and not a != b
            assert hash(a) == hash(b)
    assert len(set(halves)) == 1
    assert cf.Fixd(0.25, 8, 2) < cf.Fixb(0.5) <= cf.Fixd(0.5, 8, 1) < 1
    assert cf.Fixd(0.5, 8, 1) >= cf.Fixb(0.25) > -1
    assert cf.Fixd(0.1, 32, 1) != 0.1
    assert cf.Fixd(0.1, 32, 1) == Fraction(1, 10)
    assert cf.Fixd(1) == cf.CFixb(1) and cf.CFixb(1) == cf.Fixd(1)
    assert cf.Fixd(1) != cf.CFixb(1 + 1j) and cf.CFixb(1 + 1j) != cf.Fixd(1)
    assert cf.Fixd(1) == 1 + 0j and cf.Fixd(1) != object()
    with pytest.raises(TypeError):
        cf.Fixd(1) < 1j
    # The same base and precision compare stored integers, whatever the width or sign.
    assert cf.Fixd(-2.5, 8, 1) < cf.Ufixd(2.5, 16, 1) == cf.Fixd(2.5, 8, 1)
    assert cf.Fixb(-1, 8, 2) < cf.Ufixb(0.25, 16, 2) <= cf.Ufixb(0.25, 4, 2)
    assert cf.Fixd(1, 8, 1) > cf.Fixd(Fraction(9, 100), 8, 2) != cf.Fixd(0.1, 8, 1)


def test_decimal_arithmetic_stays_unsupported():
    x, b = cf.Fixd(1.5, 16, 1), cf.Fixb(1.5)
    for left, right in ((x, x), (x, b), (b, x), (1, x), (x, 1)):
        for op in (operator.add, operator.sub, operator.mul):
            with pytest.raises(TypeError):
                op(left, right)
    for op in (operator.neg, operator.pos, abs, lambda v: v << 1, lambda v: v >> 1):
        with pytest.raises(TypeError):
            op(x)
    for function in (cf.add, cf.subtract, cf.multiply):
        with pytest.raises(TypeError):
            function(x, x)
    for function in (cf.positive, cf.negative, cf.absolute):
        with pytest.raises(TypeError):
            function(x)


@pytest.mark.parametrize("cls,args", [(cf.Fixd, (-1.25, 24, 2)), (cf.Ufixd, (1.25, 24, 2)),
                                      (cf.Fixd32, (-1.25, 2)), (cf.Ufixd32, (1.25, 2))])
def test_storage_copy_and_pickle_under_another_context(cls, args):
    value = cls(*args)
    assert not hasattr(value, "__dict__")
    with pytest.raises(AttributeError):
        value.tag = 1

    def check():
        cf.set_decimal_context(cf.DecimalContext(8, -3, rounding="floor", overflow="wrap"))
        results = [copy.copy(value), copy.deepcopy(value)] + [
            pickle.loads(pickle.dumps(value, protocol))
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)]
        for result in results:
            assert type(result) is cls
            assert (result.int, result.format) == (value.int, value.format)

    Context().run(check)


def test_decimal_fixed_width_helpers():
    for cls, signed in ((cf.Fixd32, True), (cf.Ufixd32, False)):
        assert cls(1.25, 2).format == dfmt(32, 2, signed)
        assert cls(2 ** 32, format=dfmt(32, 0, signed), math=cf.Math(overflow="wrap")).int == 0
        with pytest.raises(ValueError, match="32 bits"):
            cls(0, format=dfmt(16, 0, signed))
        with pytest.raises(cf.FixedPointOverflowError):
            cls(2 ** 32, 0)
    assert cf.Fixd32(0, 2).upper_bound == 21474836.47
    for cls in (cf.Fixd32, cf.Ufixd32, cf.Fixb32, cf.Ufixb32):
        # Argument errors name the class that was called.
        with pytest.raises(TypeError, match=rf"^{cls.__name__}\.__new__\(\) takes"):
            cls(1, 32, 16)
        with pytest.raises(TypeError, match=rf"^{cls.__name__}\.__new__\(\) got"):
            cls(1, wordlength=32)


def test_huge_values_are_exact():
    huge = 2 ** 10000 + 1
    x = cf.Fixd(huge, 10_010, 2)
    assert x.int == huge * 100
    assert x == huge and hash(x) == hash(huge)
    y = cf.Fixd(Fraction(huge, 1000), 10_010, 3)
    assert y.int == huge
    assert y.as_integer_ratio() == Fraction(huge, 1000).as_integer_ratio()
    assert cf.Fixb(y, 10_030, 10).as_integer_ratio() == cf.Fixb(
        Fraction(huge, 1000), 10_030, 10).as_integer_ratio()


def test_text_is_exact_and_repr_rebuilds_the_value_under_any_context():
    tenth, exact_tenth = cf.Fixd(0.1), cf.Fixd(Fraction("0.1"))
    assert tenth != exact_tenth
    cases = {
        tenth: ("0.100000000000000006", "Fixd(Fraction('0.100000000000000006'), 256, 18)"),
        exact_tenth: ("0.1", "Fixd(Fraction('0.1'), 256, 18)"),
        cf.Fixd(-0.625, 32, 3): ("-0.625", "Fixd(-0.625, 32, 3)"),
        cf.Fixd(Fraction(-1, 200), 16, 3): ("-0.005", "Fixd(Fraction('-0.005'), 16, 3)"),
        cf.Ufixd(3, 16, 2): ("3.0", "Ufixd(3.0, 16, 2)"),
        cf.Fixd(0, 8, 2): ("0.0", "Fixd(0.0, 8, 2)"),
        cf.Fixd(150, 16, -2): ("200.0", "Fixd(200.0, 16, -2)"),
        cf.Fixb(-12, 8, -2): ("-12.0", "Fixb(-12.0, 8, -2)"),
        cf.Fixb(1.5): ("1.5", "Fixb(1.5, 32, 16)"),
        # A value that is exactly a float uses the shortest float literal.
        cf.Ufixb(2 ** -30, 64, 30): ("0.000000000931322574615478515625",
                                     "Ufixb(9.313225746154785e-10, 64, 30)"),
        cf.Fixb(1 / 3, 64, 52): ("0.3333333333333332593184650249895639717578887939453125",
                                 "Fixb(0.33333333333333326, 64, 52)"),
        cf.Fixb(2 ** 60, 64, 0): ("1152921504606846976.0", "Fixb(1.152921504606847e+18, 64, 0)"),
        cf.Fixb32(-1.5, 16): ("-1.5", "Fixb32(-1.5, 16)"),
        cf.Ufixd32(Fraction("2.5"), 1): ("2.5", "Ufixd32(2.5, 1)"),
        cf.Fixd32(Fraction("2.1"), 1): ("2.1", "Fixd32(Fraction('2.1'), 1)"),
    }
    for value, (text, representation) in cases.items():
        assert (str(value), repr(value)) == (text, representation)
    huge = 10 ** 400 + 1
    values = [*cases, cf.Fixb(Fraction(1, 3), 80, 70), cf.Fixb(2 ** 2000, 4096, 0),
              cf.Fixd(-huge, 2000, 2), cf.Fixd(Fraction(1, 10 ** 400), 16, 400)]
    assert str(values[-2]) == f"-{huge}.0"
    assert str(values[-1]) == "0." + "0" * 399 + "1"
    namespace = {"Fraction": Fraction, **{cls.__name__: cls for cls in (
        cf.Fixd, cf.Ufixd, cf.Fixb, cf.Ufixb, cf.Fixd32, cf.Ufixd32, cf.Fixb32, cf.Ufixb32)}}

    def check():
        cf.set_decimal_context(cf.DecimalContext(8, -3, rounding="floor", overflow="wrap"))
        cf.set_binary_context(cf.BinaryContext(8, -3, rounding="ceil", overflow="wrap"))
        for value in values:
            rebuilt = eval(repr(value), namespace)
            assert type(rebuilt) is type(value)
            assert (rebuilt.int, rebuilt.format) == (value.int, value.format)

    Context().run(check)
