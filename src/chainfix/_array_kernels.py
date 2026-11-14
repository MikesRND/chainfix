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

"""Checked int64 kernels, imported only by the optional array adapter.

All bounds come from declared formats, never observed samples. No arithmetic
kernel falls back to float, object arrays, or unbounded elementwise arithmetic.
"""

import numpy as np

from chainfix.format import Format, _binary_format
from chainfix.policy import FixedPointOverflowError, Overflow, Rounding


def check_format(format: Format, complex: bool = False) -> Format:
    _binary_format(format)
    if complex and not format.signed:
        raise ValueError("complex arrays require a signed binary format")
    if format.wordlength > (64 if format.signed else 63):
        raise ValueError("unsupported int64 kernel domain: signed <=64 or unsigned <=63 bits required")
    return format


def storage_dtype(format: Format) -> np.dtype:
    return np.dtype(f"{'int' if format.signed else 'uint'}"
                    f"{next(n for n in (8, 16, 32, 64) if n >= format.wordlength)}")


def check_shift(format: Format, shift: int) -> None:
    # Check before shifting even Python bounds: precision can be arbitrarily large.
    if shift > 63 or format.wordlength + shift > (64 if format.signed else 63):
        raise ValueError("unsupported int64 kernel domain: aligned/scaled range exceeds int64")


def check_conversion(source: Format, target: Format) -> None:
    check_format(target)
    shift = target.precision - source.precision
    if shift > 0:
        check_shift(source, shift)


def _quotient(raw, shift):
    """Floor quotient and nonzero-remainder flag without a large divisor."""
    if shift >= 64:
        return np.where(raw < 0, np.int64(-1), np.int64(0)), raw != 0
    quotient = np.right_shift(raw, shift)
    remainder = np.bitwise_and(raw, (1 << shift) - 1)
    return quotient, remainder != 0


def _round_down(raw, shift, rounding):
    # shift >= 1. The quotient has headroom for a rounding increment, including
    # at INT64_MIN/MAX. Never negate min-int or add a half-LSB to the input.
    quotient, nonzero = _quotient(raw, shift)
    if rounding is Rounding.FLOOR:
        return quotient
    if rounding is Rounding.CEIL:
        return quotient + nonzero
    if rounding is Rounding.ZERO:
        return quotient + (nonzero & (raw < 0))
    if shift >= 64:
        # |raw| <= 2**63: only INT64_MIN / 2**64 can be a tie.
        if shift == 64 and rounding is Rounding.NEAREST_AWAY:
            return np.where(raw == -(1 << 63), np.int64(-1), np.int64(0))
        return np.zeros_like(raw)
    remainder = np.bitwise_and(raw, (1 << shift) - 1)
    half = 1 << (shift - 1)
    increment = remainder > half
    if rounding is Rounding.NEAREST_EVEN:
        increment |= (remainder == half) & ((quotient & 1) != 0)
    elif rounding is Rounding.NEAREST_UP:
        increment |= remainder == half
    else:  # nearest_away
        increment |= (remainder == half) & (raw >= 0)
    return quotient + increment


def convert(raw, source: Format, target: Format, policy):
    """Round first, then apply overflow. The caller also preflights conversions."""
    check_conversion(source, target)
    raw = np.asarray(raw, dtype=np.int64)
    shift = target.precision - source.precision
    if shift > 0:
        raw = np.left_shift(raw, shift)
    elif shift < 0:
        raw = _round_down(raw, -shift, policy.rounding)
    lower, upper = target.min_int, target.max_int
    if policy.overflow is Overflow.RAISE:
        if np.any(raw < lower) or np.any(raw > upper):
            raise FixedPointOverflowError("stored array integer is outside the destination format")
        return raw
    if policy.overflow is Overflow.SATURATE:
        return np.clip(raw, lower, upper)
    if target.signed and target.wordlength == 64:
        return raw  # Every int64 encoding is valid; no 2**64 mask is needed.
    raw = np.bitwise_and(raw, (1 << target.wordlength) - 1)
    if target.signed:
        sign = 1 << (target.wordlength - 1)
        # Both intermediates fit int64, even for a 63-bit target.
        raw = np.bitwise_xor(raw, sign) - sign
    return raw


def compare(a, ap, b, bp, operation):
    """Compare exactly without left alignment, even across enormous precisions."""
    if ap < bp:
        reverse = {"equal": "equal", "not_equal": "not_equal", "less": "greater",
                   "less_equal": "greater_equal", "greater": "less",
                   "greater_equal": "less_equal"}
        return compare(b, bp, a, ap, reverse[operation])
    if ap == bp:
        return getattr(np, operation)(a, b)
    quotient, nonzero = _quotient(a, ap - bp)
    equal = (quotient == b) & ~nonzero
    if operation == "equal":
        return equal
    if operation == "not_equal":
        return ~equal
    if operation == "less":
        return quotient < b
    if operation == "less_equal":
        return (quotient < b) | equal
    if operation == "greater":
        return (quotient > b) | ((quotient == b) & nonzero)
    return quotient >= b
