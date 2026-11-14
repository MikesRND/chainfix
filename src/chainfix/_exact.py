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

"""Unbounded integer conversion helpers for the binary scalar engine."""

from fractions import Fraction
from math import isfinite

from chainfix.fixed_point import _FixedPoint
from chainfix.format import Format
from chainfix.policy import FixedPointOverflowError, Math, Overflow, Rounding


def scaled_ratio(raw: int, precision: int, base: int = 2) -> tuple[int, int]:
    if precision >= 0:
        return raw, base ** precision
    return raw * base ** -precision, 1


def value_ratio(value: int | float | Fraction | _FixedPoint) -> tuple[int, int]:
    if isinstance(value, _FixedPoint):
        return scaled_ratio(value.int, value.precision, value.base)
    if isinstance(value, int):
        return int(value), 1
    if isinstance(value, Fraction):
        return value.numerator, value.denominator
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("fixed-point inputs must be finite")
        return value.as_integer_ratio()
    raise TypeError("value must be an integer, finite float, Fraction, or fixed-point value")


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
    if format.precision >= 0:
        numerator <<= format.precision
    else:
        denominator <<= -format.precision
    return overflow_raw(round_ratio(numerator, denominator, math.rounding), format, math.overflow)


def quantize_raw(raw: int, precision: int, format: Format, math: Math) -> int:
    shift = format.precision - precision
    if shift >= 0:
        rounded = raw << shift
    else:
        rounded = round_ratio(raw, 1 << -shift, math.rounding)
    return overflow_raw(rounded, format, math.overflow)
