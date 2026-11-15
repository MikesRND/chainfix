"""Exact real fixed-point scalars shared by the binary and decimal types."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from math import ceil, isfinite
import operator
from typing import TYPE_CHECKING, Self

from chainfix._exact import convert_ratio, overflow_raw, scaled_ratio
from chainfix.format import Format, _integer
from chainfix.policy import Math, Overflow


class _FixedPoint:
    """A stored integer representing ``integer * base**(-precision)``.

    Integers are stored in two's complement within ``wordlength`` bits. Values
    convert exactly from their numeric input and round once, using the explicit
    ``math`` or the active context of the type's base. Values carry no policy.
    """

    __slots__ = ("_int", "_wordlength", "_precision")

    # Concrete subclasses set the base, signedness, and (optionally) a fixed width.
    _base: int
    _signed: bool
    _fixed_wordlength: int | None = None
    # An extra check for an explicit math, raising if this type cannot apply it.
    _math_rule: Callable[[Math], None] | None = None

    if TYPE_CHECKING:
        _int: int
        _wordlength: int
        _precision: int

    def __new__(cls, value: int | float | Fraction | _FixedPoint = 0,
                wordlength: int | None = None, precision: int | None = None, *,
                format: Format | None = None, math: Math | None = None) -> Self:
        format, math = cls._settings(wordlength, precision, format, math)
        cls._validate_format(format)
        return cls._from_raw(convert_ratio(*value_ratio(value), format, math), format)

    @classmethod
    def _current_context(cls):
        """Return the active context for this type's base."""
        raise NotImplementedError

    @classmethod
    def _settings(cls, wordlength: int | None, precision: int | None,
                  format: Format | None, math: Math | None) -> tuple[Format, Math]:
        """Choose a construction's format and math, reading the context at most once.

        Explicit ``format`` and ``math`` never read it. The format is not yet
        validated against the type.
        """
        context = None
        if format is not None:
            if wordlength is not None or precision is not None:
                raise ValueError("format cannot be combined with wordlength or precision")
        else:
            context = cls._current_context()
            format = Format(base=cls._base, signed=cls._signed,
                            wordlength=wordlength if wordlength is not None else context.wordlength,
                            precision=precision if precision is not None else context.precision)
        if math is None:
            math = (context if context is not None else cls._current_context()).math
        elif not isinstance(math, Math):
            raise TypeError("math must be a Math")
        elif cls._math_rule is not None:
            cls._math_rule(math)
        return format, math

    @classmethod
    def _validate_format(cls, format: Format) -> None:
        if not isinstance(format, Format):
            raise TypeError("format must be a Format")
        if format.base != cls._base:
            raise ValueError(f"{cls.__name__} requires a base-{cls._base} format")
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

    # -----------------------------------------------------------------------
    # Data type inspection
    # -----------------------------------------------------------------------

    base = property(lambda self: self._base)
    wordlength = property(lambda self: self._wordlength)
    precision = property(lambda self: self._precision)
    signed = property(lambda self: self._signed)

    @property
    def format(self) -> Format:
        return Format(base=self._base, signed=self._signed, wordlength=self._wordlength,
                      precision=self._precision)

    @property
    def max_int(self) -> int:
        """Maximum possible stored integer for data type. """
        if self._signed:
            return int(2 ** (self._wordlength - 1) - 1)
        else:
            return int(2 ** self._wordlength - 1)

    @property
    def min_int(self) -> int:
        """Minimum possible stored integer for data type. """
        if self._signed:
            return int(-(2 ** (self._wordlength - 1)))
        else:
            return int(0)

    #: Data type resolution (i.e. value of one LSB)
    lsb = property(lambda self: self._base ** -self._precision)

    # -----------------------------------------------------------------------
    # Exact value
    # -----------------------------------------------------------------------

    def _scaled(self, raw: int) -> Fraction:
        return Fraction(*scaled_ratio(raw, self._precision, self._base))

    def as_integer_ratio(self) -> tuple[int, int]:
        """Return the exact real-world value as a ratio of integers."""
        return self._scaled(self._int).as_integer_ratio()

    @property
    def value(self) -> float:
        """Real-world value, rounded to the nearest float."""
        return float(self._scaled(self._int))

    @property
    def lower_bound(self) -> float:
        return float(self._scaled(self.min_int))

    @property
    def upper_bound(self) -> float:
        return float(self._scaled(self.max_int))

    # -----------------------------------------------------------------------
    # Stored Integer Properties
    # -----------------------------------------------------------------------

    #: Stored integer value
    int = property(lambda self: self._int)

    @property
    def hex(self) -> str:
        """Two's complement representation of stored integer (Hex value) """
        digits = ceil(self._wordlength / 4)
        if self._int >= 0:
            return "0x{num:0{digits}x}".format(num=self._int, digits=digits)
        else:
            return "0x{num:0{digits}x}".format(
                num=(2 ** self._wordlength + self._int), digits=digits
            )

    @property
    def bin(self) -> str:
        """Two's complement representation of stored integer
         (binary value) """
        digits = self._wordlength
        if self._int >= 0:
            return "0b{num:0{digits}b}".format(num=self._int, digits=digits)
        else:
            return "0b{num:0{digits}b}".format(
                num=(2 ** self._wordlength + self._int), digits=digits
            )

    # -----------------------------------------------------------------------
    # Representations, conversions, and exact comparison
    # -----------------------------------------------------------------------

    def _text(self) -> str:
        # Exact positional decimal text, without trailing fractional zeros. A
        # base-2 or base-10 value always has one: raw / base**p equals
        # raw * (10 // base)**p / 10**p.
        raw, precision = self._int, self._precision
        if precision <= 0:
            return f"{raw * self._base ** -precision}.0"
        whole, fraction = divmod(abs(raw) * (10 // self._base) ** precision, 10 ** precision)
        digits = f"{fraction:0{precision}d}".rstrip("0") or "0"
        return f"{'-' if raw < 0 else ''}{whole}.{digits}"

    def __repr__(self) -> str:
        # A constructor call that rebuilds this value exactly under any context: the
        # shortest float literal when the value is exactly a float, else a Fraction.
        text = self._text()
        number = float(text)  # correctly rounded; inf beyond the float range
        text = repr(number) if number == self._scaled(self._int) else f"Fraction('{text}')"
        if self._fixed_wordlength is None:
            text = f"{text}, {self._wordlength}"
        return f"{type(self).__name__}({text}, {self._precision})"

    def __str__(self) -> str:
        return self._text()

    def __bool__(self) -> bool:
        return self._int != 0

    def __float__(self) -> float:
        return self.value

    def __int__(self) -> int:
        return int(self._scaled(self._int))

    def __hash__(self) -> int:
        return hash(self._scaled(self._int))

    def _compare(self, other, operation):
        # Exact and policy-free, across bases and against Python numbers.
        if isinstance(other, _FixedPoint):
            if other._precision == self._precision and other._base == self._base:
                return operation(self._int, other._int)  # the same scale
            other = other._scaled(other._int)
        elif not isinstance(other, (int, float, Fraction, Decimal, complex)):
            return NotImplemented
        if isinstance(other, complex) and operation not in (operator.eq, operator.ne):
            return NotImplemented
        return operation(self._scaled(self._int), other)

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


def value_ratio(value: int | float | Fraction | _FixedPoint) -> tuple[int, int]:
    """Return a supported numeric input's exact value as an integer ratio."""
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
