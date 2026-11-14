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

"""Validated value formats and binary format inference."""

from dataclasses import dataclass


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not a boolean")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class Format:
    """A stored integer represents ``integer * base**(-precision)``."""

    base: int
    signed: bool
    wordlength: int
    precision: int

    def __post_init__(self):
        if _integer(self.base, "base") < 2:
            raise ValueError("base must be at least 2")
        if not isinstance(self.signed, bool):
            raise TypeError("signed must be a boolean")
        if _integer(self.wordlength, "wordlength") < 1:
            raise ValueError("wordlength must be positive")
        _integer(self.precision, "precision")

    @property
    def min_int(self) -> int:
        return -(1 << (self.wordlength - 1)) if self.signed else 0

    @property
    def max_int(self) -> int:
        return (1 << (self.wordlength - self.signed)) - 1


def _binary_format(format: Format) -> Format:
    if not isinstance(format, Format):
        raise TypeError("format must be a Format")
    if format.base != 2:
        raise ValueError("binary arithmetic requires a base-2 format")
    return format


def _range_format(lower: int, upper: int, precision: int, signed: bool) -> Format:
    if signed:
        width = max(1, (~lower).bit_length() + 1 if lower < 0 else 1,
                    upper.bit_length() + 1 if upper >= 0 else 1)
    else:
        width = max(1, upper.bit_length())
    return Format(base=2, signed=signed, wordlength=width, precision=precision)


def _sum_format(a: Format, b: Format, *, subtract: bool = False) -> Format:
    precision = max(a.precision, b.precision)
    alo, ahi = a.min_int << (precision - a.precision), a.max_int << (precision - a.precision)
    blo, bhi = b.min_int << (precision - b.precision), b.max_int << (precision - b.precision)
    lower, upper = (alo - bhi, ahi - blo) if subtract else (alo + blo, ahi + bhi)
    return _range_format(lower, upper, precision, subtract or a.signed or b.signed)


def _product_format(a: Format, b: Format) -> Format:
    return Format(base=2, signed=a.signed or b.signed,
                  wordlength=a.wordlength + b.wordlength,
                  precision=a.precision + b.precision)


def _unary_format(format: Format, *, absolute: bool = False) -> Format:
    if absolute:
        return _range_format(0, max(abs(format.min_int), abs(format.max_int)),
                             format.precision, format.signed)
    return _range_format(-format.max_int, -format.min_int, format.precision, True)
