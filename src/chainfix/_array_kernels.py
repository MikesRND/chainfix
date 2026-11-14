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


class _ConversionScratch:
    """Serial scratch; buffers must be disjoint from conversion input/output."""

    __slots__ = ("remainder", "increment", "mask", "predicate")

    def __init__(self, shape, *, remainder=None):
        self.remainder = np.empty(shape, dtype=np.int64) if remainder is None else remainder
        self.increment = np.empty(shape, dtype=bool)
        self.mask = np.empty(shape, dtype=bool)
        self.predicate = np.empty(shape, dtype=bool)


def _round_down_into(raw, shift, rounding, out, scratch):
    # shift >= 1. Save the remainder/sign before overwriting raw: exact aliases
    # are used privately after multiplication. The quotient has increment room,
    # including at INT64_MIN/MAX; never bias or negate the original integer.
    remainder, increment, mask = scratch.remainder, scratch.increment, scratch.mask
    if rounding is Rounding.FLOOR:
        np.right_shift(raw, min(shift, 63), out=out)
        return
    if rounding in (Rounding.CEIL, Rounding.ZERO):
        if shift >= 64:
            np.not_equal(raw, 0, out=increment)
        else:
            np.bitwise_and(raw, (1 << shift) - 1, out=remainder)
            np.not_equal(remainder, 0, out=increment)
        if rounding is Rounding.ZERO:
            np.less(raw, 0, out=mask)
            np.logical_and(increment, mask, out=increment)
        np.right_shift(raw, min(shift, 63), out=out)
        np.add(out, 1, out=out, where=increment)
        return
    if shift >= 64:
        # |raw| <= 2**63: only INT64_MIN / 2**64 can be a tie.
        if shift == 64 and rounding is Rounding.NEAREST_AWAY:
            np.equal(raw, -(1 << 63), out=mask)
            np.copyto(out, mask)
            np.negative(out, out=out)
        else:
            out.fill(0)
        return
    np.bitwise_and(raw, (1 << shift) - 1, out=remainder)
    half = 1 << (shift - 1)
    np.greater(remainder, half, out=increment)
    np.equal(remainder, half, out=mask)
    if rounding is Rounding.NEAREST_AWAY:
        np.greater_equal(raw, 0, out=scratch.predicate)
        np.logical_and(mask, scratch.predicate, out=mask)
    np.right_shift(raw, shift, out=out)
    if rounding is Rounding.NEAREST_EVEN:
        np.bitwise_and(out, 1, out=remainder)
        np.not_equal(remainder, 0, out=scratch.predicate)
        np.logical_and(mask, scratch.predicate, out=mask)
    np.logical_or(increment, mask, out=increment)
    # Adding a boolean array directly would allocate an int64 iterator cast
    # buffer. A masked scalar increment uses the existing boolean storage.
    np.add(out, 1, out=out, where=increment)


def convert(raw, source: Format, target: Format, policy, *, out=None, scratch=None):
    """Round, then overflow, optionally into private int64 storage and scratch.

    The reusable caller supplies same-shaped buffers; raw and out may be exact
    aliases, while scratch is disjoint. Public output overlap remains forbidden.
    """
    check_conversion(source, target)
    raw = np.asarray(raw, dtype=np.int64)
    if out is None:
        out = np.empty(raw.shape, dtype=np.int64)
    if scratch is None:
        scratch = _ConversionScratch(raw.shape)
    shift = target.precision - source.precision
    if shift > 0:
        np.left_shift(raw, shift, out=out)
    elif shift < 0:
        _round_down_into(raw, -shift, policy.rounding, out, scratch)
    else:
        np.copyto(out, raw)
    lower, upper = target.min_int, target.max_int
    if policy.overflow is Overflow.RAISE:
        np.less(out, lower, out=scratch.mask)
        if scratch.mask.any():
            raise FixedPointOverflowError("stored array integer is outside the destination format")
        np.greater(out, upper, out=scratch.mask)
        if scratch.mask.any():
            raise FixedPointOverflowError("stored array integer is outside the destination format")
    elif policy.overflow is Overflow.SATURATE:
        np.clip(out, lower, upper, out=out)
    elif not (target.signed and target.wordlength == 64):
        # Every int64 encoding already fits signed 64 bits; avoid a 2**64 mask.
        np.bitwise_and(out, (1 << target.wordlength) - 1, out=out)
        if target.signed:
            sign = 1 << (target.wordlength - 1)
            np.bitwise_xor(out, sign, out=out)
            np.subtract(out, sign, out=out)
    return out


def multiply_complex(ai, aq, bi, bq, *, i, q, work):
    """Completed components in disjoint int64 outputs, before any quantization.

    Callers preflight declared product/sum bounds and promote operands to int64.
    The reusable path promotes into persistent storage to avoid cast buffers.
    """
    np.multiply(ai, bi, out=i)
    np.multiply(aq, bq, out=work)
    np.subtract(i, work, out=i)
    np.multiply(ai, bq, out=q)
    np.multiply(aq, bi, out=work)
    np.add(q, work, out=q)


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
