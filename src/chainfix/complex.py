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

"""Exact complex binary scalars with a shared signed component format."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import sys
from typing import Self

from chainfix._exact import convert_ratio, overflow_raw, quantize_raw, scaled_ratio, value_ratio
from chainfix.binary import (
    Fixb, Ufixb, _Binary, _binary_operation, _operand, _result, _selected_math, negative,
)
from chainfix.context import get_binary_context
from chainfix.format import Format, _binary_format, _integer, _product_format, _sum_format, _unary_format
from chainfix.policy import Math, Overflow

__all__ = ["CFixb"]

_OMITTED = object()


def _complex_format(format: Format) -> None:
    _binary_format(format)
    if not format.signed:
        raise ValueError("CFixb requires a signed binary format")


class CFixb:
    """Two exact raw integers sharing one signed binary format, with no policy.

    Supply a value or keyword-only real/imaginary components, plus either
    wordlength/precision or a Format. Omitted values/components default to zero.
    """

    __slots__ = ("_i", "_q", "_format")

    def __new__(cls, value=_OMITTED, wordlength: int | None = None,
                precision: int | None = None, *, real=_OMITTED, imag=_OMITTED,
                format: Format | None = None, math: Math | None = None) -> Self:
        if value is not _OMITTED and (real is not _OMITTED or imag is not _OMITTED):
            raise ValueError("value cannot be combined with real or imag")
        if format is not None:
            if wordlength is not None or precision is not None:
                raise ValueError("format cannot be combined with wordlength or precision")
        else:
            context = get_binary_context()
            format = Format(base=2, signed=True,
                            wordlength=wordlength if wordlength is not None else context.wordlength,
                            precision=precision if precision is not None else context.precision)
        _complex_format(format)
        policy = _selected_math(math)
        if isinstance(value, CFixb):
            i_ratio = scaled_ratio(value._i, value.format.precision)
            q_ratio = scaled_ratio(value._q, value.format.precision)
        else:
            if isinstance(value, complex):
                real, imag = value.real, value.imag
            elif value is not _OMITTED:
                real = value
            i_ratio = value_ratio(0 if real is _OMITTED else real)
            q_ratio = value_ratio(0 if imag is _OMITTED else imag)
        return cls._from_raw(convert_ratio(*i_ratio, format, policy),
                             convert_ratio(*q_ratio, format, policy), format)

    @classmethod
    def _from_raw(cls, i: int, q: int, format: Format) -> Self:
        # Internal only: the caller has validated both components and the format.
        self = object.__new__(cls)
        self._i, self._q, self._format = i, q, format
        return self

    @classmethod
    def from_raw(cls, *, i: int, q: int, format: Format) -> Self:
        """Construct from two encoded integers, strictly checking both ranges."""
        _complex_format(format)
        i, q = _integer(i, "i"), _integer(q, "q")
        return cls._from_raw(overflow_raw(i, format, Overflow.RAISE),
                             overflow_raw(q, format, Overflow.RAISE), format)

    @property
    def format(self) -> Format:
        return self._format

    @property
    def real(self) -> Fixb:
        return Fixb._from_raw(self._i, self.format)

    @property
    def imag(self) -> Fixb:
        return Fixb._from_raw(self._q, self.format)

    def quantize(self, format: Format, *, math: Math | None = None) -> CFixb:
        """Round and apply overflow independently to each completed component."""
        return _complex_result(self._i, self._q, self.format.precision,
                               format, _selected_math(math))

    def conjugate(self, *, math: Math | None = None) -> CFixb:
        """Negate the imaginary component, then apply the sum policy once."""
        return _complex_unary(self, conjugate=True, math=math)

    def mag2(self, *, math: Math | None = None) -> Fixb | Ufixb:
        """Sum exact component squares, applying only the final sum policy."""
        policy = _selected_math(math)
        product = _product_format(self.format, self.format)
        full = replace(product, wordlength=product.wordlength + 1)
        target = full if policy.sum == "full" else policy.sum
        raw = self._i * self._i + self._q * self._q
        return _result(quantize_raw(raw, full.precision, target, policy), target)

    def __complex__(self) -> complex:
        return complex(float(self.real), float(self.imag))

    def __bool__(self) -> bool:
        return self._i != 0 or self._q != 0

    def __repr__(self) -> str:
        return f"CFixb.from_raw(i={self._i}, q={self._q}, format={self.format!r})"

    def __eq__(self, other):
        if isinstance(other, (CFixb, complex)):
            return self.real == other.real and self.imag == other.imag
        if isinstance(other, (_Binary, int, float, Fraction, Decimal)):
            return self._q == 0 and self.real == other
        return NotImplemented

    def __hash__(self) -> int:
        combined = hash(self.real) + sys.hash_info.imag * hash(self.imag)
        modulus = 1 << sys.hash_info.width
        signed = (combined + modulus // 2) % modulus - modulus // 2
        return -2 if signed == -1 else signed

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
        return CFixb._from_raw(self._i, self._q, self.format)

    def __neg__(self):
        return negative(self)

    def __abs__(self):
        raise TypeError("complex absolute value is unsupported; use mag2()")


def _complex_result(i: int, q: int, precision: int, target: Format, policy: Math) -> CFixb:
    _complex_format(target)
    return CFixb._from_raw(quantize_raw(i, precision, target, policy),
                           quantize_raw(q, precision, target, policy), target)


def _complex_unary(value: CFixb, *, conjugate: bool, math: Math | None) -> CFixb:
    policy = _selected_math(math)
    full = _unary_format(value.format)
    target = full if policy.sum == "full" else policy.sum
    i = value._i if conjugate else -value._i
    return _complex_result(i, -value._q, full.precision, target, policy)


def _complex_operation(a, b, operation: str, math: Math | None):
    a = a if isinstance(a, CFixb) else _operand(a)
    b = b if isinstance(b, CFixb) else _operand(b)
    if a is NotImplemented or b is NotImplemented:
        return NotImplemented
    policy = _selected_math(math)
    ai, aq = (a._i, a._q) if isinstance(a, CFixb) else (a.int, 0)
    bi, bq = (b._i, b._q) if isinstance(b, CFixb) else (b.int, 0)
    if operation == "multiply":
        full = _product_format(a.format, b.format)
        if isinstance(a, CFixb) and isinstance(b, CFixb):
            full = replace(full, wordlength=full.wordlength + 1)
        i, q = ai * bi - aq * bq, ai * bq + aq * bi
        boundary = policy.product
    else:
        full = _sum_format(a.format, b.format, subtract=operation == "subtract")
        ashift = full.precision - a.format.precision
        bshift = full.precision - b.format.precision
        ai, aq, bi, bq = ai << ashift, aq << ashift, bi << bshift, bq << bshift
        i, q = (ai - bi, aq - bq) if operation == "subtract" else (ai + bi, aq + bq)
        boundary = policy.sum
    target = full if boundary == "full" else boundary
    return _complex_result(i, q, full.precision, target, policy)
