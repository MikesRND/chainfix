"""Exact decimal fixed-point scalars: construction, inspection, and comparison.

A float converts as its exact binary value, rounded once, as ``decimal.Decimal``
does; use ``Fraction("0.1")`` for an exact decimal literal. Decimal arithmetic is
unsupported.
"""

from chainfix.context import DecimalContext, get_decimal_context
from chainfix.fixed_point import _FixedPoint

__all__ = ["Fixd", "Ufixd"]


class _Decimal(_FixedPoint):
    __slots__ = ()
    _base = 10
    # An explicit policy follows the same rule as a decimal context's.
    _math_rule = DecimalContext._check_math

    @classmethod
    def _current_context(cls):
        # Look the getter up at call time so it can be patched in tests.
        return get_decimal_context()


class Fixd(_Decimal):
    """Signed decimal fixed-point scalar, with exact conversion."""

    __slots__ = ()
    _signed = True


class Ufixd(_Decimal):
    """Unsigned decimal fixed-point scalar, with exact conversion."""

    __slots__ = ()
    _signed = False
