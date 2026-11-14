# Copyright 2021 PyDefi Development Team
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

"""Exact real binary scalars and their elementary arithmetic engine."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import operator
from typing import Self

from chainfix._exact import overflow_raw, quantize_raw, scaled_ratio, convert_ratio, value_ratio
from chainfix.context import get_binary_context
from chainfix.fixed_point import _FixedPoint
from chainfix.format import (
    Format, _binary_format, _integer, _product_format, _range_format, _sum_format,
    _unary_format,
)
from chainfix.policy import FixedPointOverflowError, Math, Overflow

__all__ = ["Fixb", "Ufixb", "add", "subtract", "multiply", "positive", "negative", "absolute"]


def _selected_math(math: Math | None) -> Math:
    if math is None:
        return get_binary_context().math
    if not isinstance(math, Math):
        raise TypeError("math must be a Math")
    return math


class _Binary(_FixedPoint):
    __slots__ = ()
    _base = 2
    _fixed_wordlength = None

    def __new__(cls, value: int | float | Fraction | _FixedPoint = 0,
                wordlength: int | None = None, precision: int | None = None, *,
                format: Format | None = None, math: Math | None = None) -> Self:
        if format is not None:
            if wordlength is not None or precision is not None:
                raise ValueError("format cannot be combined with wordlength or precision")
        else:
            context = get_binary_context()
            format = Format(base=2, signed=cls._signed,
                            wordlength=wordlength if wordlength is not None else context.wordlength,
                            precision=precision if precision is not None else context.precision)
        cls._validate_format(format)
        policy = _selected_math(math)
        raw = convert_ratio(*value_ratio(value), format, policy)
        return cls._from_raw(raw, format)

    @classmethod
    def _validate_format(cls, format: Format) -> None:
        _binary_format(format)
        if format.signed != cls._signed:
            raise ValueError(f"format signedness is incompatible with {cls.__name__}")
        if cls._fixed_wordlength is not None and format.wordlength != cls._fixed_wordlength:
            raise ValueError(f"{cls.__name__} requires {cls._fixed_wordlength} bits")

    @classmethod
    def _from_raw(cls, raw: int, format: Format) -> Self:
        # Internal only: the caller has already validated format and raw range.
        self = object.__new__(cls)
        self._int = raw
        self._wordlength = format.wordlength
        self._precision = format.precision
        return self

    @classmethod
    def from_raw(cls, raw: int, *, format: Format) -> Self:
        """Construct from an encoded integer, strictly checking its range."""
        cls._validate_format(format)
        raw = _integer(raw, "raw")
        return cls._from_raw(overflow_raw(raw, format, Overflow.RAISE), format)

    @property
    def format(self) -> Format:
        return Format(base=2, signed=self.signed, wordlength=self.wordlength,
                      precision=self.precision)

    def quantize(self, format: Format, *, math: Math | None = None) -> Fixb | Ufixb:
        """Convert to a format, rounding before checking overflow."""
        _binary_format(format)
        raw = quantize_raw(self.int, self.precision, format, _selected_math(math))
        return _result(raw, format)

    def get_current_context(self):
        return get_binary_context()

    def as_integer_ratio(self) -> tuple[int, int]:
        return Fraction(*scaled_ratio(self.int, self.precision)).as_integer_ratio()

    @property
    def value(self) -> float:
        return float(Fraction(*self.as_integer_ratio()))

    @property
    def lower_bound(self) -> float:
        return float(Fraction(*scaled_ratio(self.min_int, self.precision)))

    @property
    def upper_bound(self) -> float:
        return float(Fraction(*scaled_ratio(self.max_int, self.precision)))

    def __int__(self) -> int:
        return int(Fraction(*self.as_integer_ratio()))

    def __hash__(self) -> int:
        return hash(Fraction(*self.as_integer_ratio()))

    def _compare(self, other, operation):
        if isinstance(other, _Binary):
            other = Fraction(*other.as_integer_ratio())
        elif not isinstance(other, (int, float, Fraction, Decimal, complex)):
            return NotImplemented
        if isinstance(other, complex) and operation not in (operator.eq, operator.ne):
            return NotImplemented
        return operation(Fraction(*self.as_integer_ratio()), other)

    def __eq__(self, other):
        return self._compare(other, operator.eq)

    def __ne__(self, other):
        return self._compare(other, operator.ne)

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    def __le__(self, other):
        return self._compare(other, operator.le)

    def __gt__(self, other):
        return self._compare(other, operator.gt)

    def __ge__(self, other):
        return self._compare(other, operator.ge)

    def __add__(self, other):
        return _binary_operation(self, other, "add")

    def __radd__(self, other):
        return _binary_operation(other, self, "add")

    def __sub__(self, other):
        return _binary_operation(self, other, "subtract")

    def __rsub__(self, other):
        return _binary_operation(other, self, "subtract")

    def __mul__(self, other):
        return _binary_operation(self, other, "multiply")

    def __rmul__(self, other):
        return _binary_operation(other, self, "multiply")

    def __pos__(self):
        return _result(self.int, self.format)

    def __neg__(self):
        return _unary(self, absolute=False, math=None)

    def __abs__(self):
        return _unary(self, absolute=True, math=None)

    def __lshift__(self, count: int):
        _shift_count(count)
        policy = get_binary_context().math
        if self.int == 0:
            raw = 0
        elif count >= self.wordlength:
            # Avoid constructing an enormous integer when the answer is known.
            if policy.overflow is Overflow.RAISE:
                raise FixedPointOverflowError("left shift exceeds the stored integer range")
            if policy.overflow is Overflow.SATURATE:
                raw = self.max_int if self.int > 0 else self.min_int
            else:
                raw = 0
        else:
            raw = overflow_raw(self.int << count, self.format, policy.overflow)
        return _result(raw, self.format)

    def __rshift__(self, count: int):
        _shift_count(count)
        return _result(self.int >> count, self.format)

    def scale_pow2(self, exponent: int) -> Fixb | Ufixb:
        """Scale exactly by changing precision, preserving raw bits and width."""
        _integer(exponent, "exponent")
        return _result(self.int, replace(self.format, precision=self.precision - exponent))


class Fixb(_Binary):
    """Signed binary fixed-point scalar, with exact conversion and arithmetic."""

    __slots__ = ()
    _signed = True


class Ufixb(_Binary):
    """Unsigned binary fixed-point scalar, with exact conversion and arithmetic."""

    __slots__ = ()
    _signed = False


def _shift_count(count: int) -> None:
    if _integer(count, "shift count") < 0:
        raise ValueError("shift count must be nonnegative")


def _result(raw: int, format: Format) -> Fixb | Ufixb:
    return (Fixb if format.signed else Ufixb)._from_raw(raw, format)


def _operand(value):
    if isinstance(value, _Binary):
        return value
    if isinstance(value, int):
        raw = int(value)
        format = _range_format(raw, raw, 0, True)
        return Fixb._from_raw(raw, format)
    return NotImplemented


def _binary_operation(a, b, operation: str, math: Math | None = None):
    a, b = _operand(a), _operand(b)
    if a is NotImplemented or b is NotImplemented:
        return NotImplemented
    policy = _selected_math(math)
    if operation == "multiply":
        full = _product_format(a.format, b.format)
        raw = a.int * b.int
        boundary = policy.product
    else:
        full = _sum_format(a.format, b.format, subtract=operation == "subtract")
        left = a.int << (full.precision - a.precision)
        right = b.int << (full.precision - b.precision)
        raw = left - right if operation == "subtract" else left + right
        boundary = policy.sum
    target = full if boundary == "full" else boundary
    return _result(quantize_raw(raw, full.precision, target, policy), target)


def _explicit(a, b, operation: str, math: Math | None):
    if isinstance(a, CFixb) or isinstance(b, CFixb):
        result = _complex_operation(a, b, operation, math)
    else:
        result = _binary_operation(a, b, operation, math)
    if result is NotImplemented:
        raise TypeError(f"{operation} requires binary fixed-point values or integer constants")
    return result


def add(a, b, *, math: Math | None = None) -> Fixb | Ufixb | CFixb:
    """Add exactly, then apply the sum result format."""
    return _explicit(a, b, "add", math)


def subtract(a, b, *, math: Math | None = None) -> Fixb | Ufixb | CFixb:
    """Subtract exactly, then apply the sum result format."""
    return _explicit(a, b, "subtract", math)


def multiply(a, b, *, math: Math | None = None) -> Fixb | Ufixb | CFixb:
    """Multiply original stored integers, then apply the product result format."""
    return _explicit(a, b, "multiply", math)


def positive(value) -> Fixb | Ufixb | CFixb:
    """Return a generic scalar with unchanged value and format."""
    if isinstance(value, CFixb):
        return +value
    value = _operand(value)
    if value is NotImplemented:
        raise TypeError("positive requires a binary fixed-point value or integer")
    return +value


def _unary(value, *, absolute: bool, math: Math | None):
    value = _operand(value)
    if value is NotImplemented:
        raise TypeError("unary arithmetic requires a binary fixed-point value or integer")
    policy = _selected_math(math)
    full = _unary_format(value.format, absolute=absolute)
    raw = abs(value.int) if absolute else -value.int
    target = full if policy.sum == "full" else policy.sum
    return _result(quantize_raw(raw, full.precision, target, policy), target)


def negative(value, *, math: Math | None = None) -> Fixb | Ufixb | CFixb:
    """Negate with declared-range growth, then apply the sum result format."""
    if isinstance(value, CFixb):
        return _complex_unary(value, conjugate=False, math=math)
    return _unary(value, absolute=False, math=math)


def absolute(value, *, math: Math | None = None) -> Fixb | Ufixb:
    """Take the real absolute value, then apply the sum result format."""
    if isinstance(value, CFixb):
        raise TypeError("complex absolute value is unsupported; use mag2()")
    return _unary(value, absolute=True, math=math)


# Complex scalars depend on the real helpers above. Import after defining them
# to resolve the cycle once, and bind CFixb for runtime annotation introspection.
# Only explicit functions dispatch here; real operators use Python reflection.
from chainfix.complex import CFixb, _complex_operation, _complex_unary
