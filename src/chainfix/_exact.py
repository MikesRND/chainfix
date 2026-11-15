"""Unbounded integer conversion helpers for the exact scalar engine."""

from chainfix.format import Format
from chainfix.policy import FixedPointOverflowError, Math, Overflow, Rounding


def scaled_ratio(raw: int, precision: int, base: int) -> tuple[int, int]:
    if precision >= 0:
        return raw, base ** precision
    return raw * base ** -precision, 1


def round_ratio(numerator: int, denominator: int, rounding: Rounding) -> int:
    quotient, remainder = divmod(numerator, denominator)
    if not remainder or rounding is Rounding.FLOOR:
        return quotient
    if rounding is Rounding.CEIL:
        return quotient + 1
    if rounding is Rounding.ZERO:
        return quotient + (numerator < 0)
    twice = remainder * 2
    if twice != denominator:
        return quotient + (twice > denominator)
    if rounding is Rounding.NEAREST_EVEN:
        return quotient + (quotient & 1)
    if rounding is Rounding.NEAREST_UP:
        return quotient + 1
    return quotient + (numerator >= 0)  # nearest_away


def overflow_raw(raw: int, format: Format, overflow: Overflow) -> int:
    lower, upper = format.min_int, format.max_int
    if lower <= raw <= upper:
        return raw
    if overflow is Overflow.RAISE:
        raise FixedPointOverflowError(
            f"stored integer is outside the range of a {format.wordlength}-bit "
            f"{'signed' if format.signed else 'unsigned'} format")
    if overflow is Overflow.SATURATE:
        return min(max(raw, lower), upper)
    return (raw - lower) % (1 << format.wordlength) + lower


def convert_ratio(numerator: int, denominator: int, format: Format, math: Math) -> int:
    """Scale an exact ratio into a format, rounding once before checking overflow."""
    if format.base == 2:
        # Shifts keep the common binary path fast.
        if format.precision >= 0:
            numerator <<= format.precision
        else:
            denominator <<= -format.precision
    elif format.precision >= 0:
        numerator *= format.base ** format.precision
    else:
        denominator *= format.base ** -format.precision
    return overflow_raw(round_ratio(numerator, denominator, math.rounding), format, math.overflow)


def quantize_raw(raw: int, precision: int, format: Format, math: Math) -> int:
    shift = format.precision - precision
    if shift >= 0:
        rounded = raw << shift
    else:
        rounded = round_ratio(raw, 1 << -shift, math.rounding)
    return overflow_raw(rounded, format, math.overflow)
