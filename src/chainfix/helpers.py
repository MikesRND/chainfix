from fractions import Fraction
from typing import Self

from chainfix.binary import Fixb
from chainfix.binary import Ufixb
from chainfix.decimal import Fixd
from chainfix.decimal import Ufixd
from chainfix.fixed_point import _FixedPoint
from chainfix.format import Format
from chainfix.policy import Math


# --------------------------------------------------------------------------
# 32-bit Helper Types
# --------------------------------------------------------------------------
# Each type has its own __new__ (the second positional argument is precision),
# so argument errors name the class that was called.
class Fixb32(Fixb):
    __slots__ = ()
    _fixed_wordlength = 32

    def __new__(cls, value: int | float | Fraction | _FixedPoint = 0,
                precision: int | None = None, *,
                format: Format | None = None, math: Math | None = None) -> Self:
        return super().__new__(cls, value, 32 if format is None else None, precision,
                               format=format, math=math)


class Ufixb32(Ufixb):
    __slots__ = ()
    _fixed_wordlength = 32

    def __new__(cls, value: int | float | Fraction | _FixedPoint = 0,
                precision: int | None = None, *,
                format: Format | None = None, math: Math | None = None) -> Self:
        return super().__new__(cls, value, 32 if format is None else None, precision,
                               format=format, math=math)


class Fixd32(Fixd):
    __slots__ = ()
    _fixed_wordlength = 32

    def __new__(cls, value: int | float | Fraction | _FixedPoint = 0,
                precision: int | None = None, *,
                format: Format | None = None, math: Math | None = None) -> Self:
        return super().__new__(cls, value, 32 if format is None else None, precision,
                               format=format, math=math)


class Ufixd32(Ufixd):
    __slots__ = ()
    _fixed_wordlength = 32

    def __new__(cls, value: int | float | Fraction | _FixedPoint = 0,
                precision: int | None = None, *,
                format: Format | None = None, math: Math | None = None) -> Self:
        return super().__new__(cls, value, 32 if format is None else None, precision,
                               format=format, math=math)
