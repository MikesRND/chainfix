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
    """Serial scratch; buffers must be disjoint from conversion input/output.

    The reusable workspace preallocates every buffer once. A one-off conversion
    passes lazy=True so only the buffers its rounding/overflow branch needs exist.
    """

    __slots__ = ("_shape", "_remainder", "_increment", "_mask", "_predicate")

    def __init__(self, shape, *, remainder=None, lazy=False):
        self._shape = shape
        self._remainder = remainder
        self._increment = self._mask = self._predicate = None
        if not lazy:
            for name in ("remainder", "increment", "mask", "predicate"):
                getattr(self, name)

    @property
    def remainder(self):
        if self._remainder is None:
            self._remainder = np.empty(self._shape, dtype=np.int64)
        return self._remainder

    @property
    def increment(self):
        if self._increment is None:
            self._increment = np.empty(self._shape, dtype=bool)
        return self._increment

    @property
    def mask(self):
        if self._mask is None:
            self._mask = np.empty(self._shape, dtype=bool)
        return self._mask

    @property
    def predicate(self):
        if self._predicate is None:
            self._predicate = np.empty(self._shape, dtype=bool)
        return self._predicate


def _add_increment(out, increment, scratch):
    # Adding the boolean array directly allocates an iterator cast buffer, and a
    # masked add (where=increment) is several times slower on irregular masks.
    # The int64 remainder is dead here, so widen the increment into it.
    widened = scratch.remainder
    np.copyto(widened, increment)
    np.add(out, widened, out=out)


def _round_down_into(raw, shift, rounding, out, scratch):
    # shift >= 1. Save the remainder/sign before overwriting raw: exact aliases
    # are used privately after multiplication. The quotient has increment room,
    # including at INT64_MIN/MAX; never bias or negate the original integer.
    if rounding is Rounding.FLOOR:
        np.right_shift(raw, min(shift, 63), out=out)
        return
    # Access scratch lazily per branch: one-off conversions allocate only what
    # their rounding mode uses.
    increment = scratch.increment
    if rounding in (Rounding.CEIL, Rounding.ZERO):
        if shift >= 64:
            np.not_equal(raw, 0, out=increment)
        else:
            remainder = scratch.remainder
            np.bitwise_and(raw, (1 << shift) - 1, out=remainder)
            np.not_equal(remainder, 0, out=increment)
        if rounding is Rounding.ZERO:
            mask = scratch.mask
            np.less(raw, 0, out=mask)
            np.logical_and(increment, mask, out=increment)
        np.right_shift(raw, min(shift, 63), out=out)
        _add_increment(out, increment, scratch)
        return
    mask = scratch.mask
    if shift >= 64:
        # |raw| <= 2**63: only INT64_MIN / 2**64 can be a tie.
        if shift == 64 and rounding is Rounding.NEAREST_AWAY:
            np.equal(raw, -(1 << 63), out=mask)
            np.copyto(out, mask)
            np.negative(out, out=out)
        else:
            out.fill(0)
        return
    remainder = scratch.remainder
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
    _add_increment(out, increment, scratch)


def convert(raw, source: Format, target: Format, policy, *, out=None, scratch=None):
    """Round, then overflow, optionally into private int64 storage and scratch.

    The reusable caller supplies same-shaped buffers; raw and out may be exact
    aliases, while scratch is disjoint. Public output overlap remains forbidden.
    Without out, a conversion that changes no values (zero shift with raise, or
    wrap to signed 64 bits) returns the int64 input itself, which may be borrowed
    caller storage. Callers that need independent storage must copy.
    """
    check_conversion(source, target)
    raw = np.asarray(raw, dtype=np.int64)
    if scratch is None:
        scratch = _ConversionScratch(raw.shape, lazy=True)
    shift = target.precision - source.precision
    if shift == 0:
        value = raw
    else:
        if out is None:
            out = np.empty(raw.shape, dtype=np.int64)
        if shift > 0:
            np.left_shift(raw, shift, out=out)
        else:
            _round_down_into(raw, -shift, policy.rounding, out, scratch)
        value = out
    lower, upper = target.min_int, target.max_int
    # Every int64 encoding already fits signed 64 bits, so wrapping changes nothing.
    if policy.overflow is Overflow.RAISE or (policy.overflow is Overflow.WRAP and
                                              target.signed and target.wordlength == 64):
        if policy.overflow is Overflow.RAISE:
            mask = scratch.mask
            np.less(value, lower, out=mask)
            if mask.any():
                raise FixedPointOverflowError("stored array integer is outside the destination format")
            np.greater(value, upper, out=mask)
            if mask.any():
                raise FixedPointOverflowError("stored array integer is outside the destination format")
        if out is None:
            return value
        if out is not value:
            np.copyto(out, value)
        return out
    if out is None:
        out = np.empty(raw.shape, dtype=np.int64)
    if policy.overflow is Overflow.SATURATE:
        np.clip(value, lower, upper, out=out)
    else:
        np.bitwise_and(value, (1 << target.wordlength) - 1, out=out)
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
