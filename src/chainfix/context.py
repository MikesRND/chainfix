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

"""Immutable fixed-point contexts and their execution-context scoping.

A context holds construction defaults (word length and precision) and the
arithmetic policy (``Math``). Contexts are values: derive a changed one with
``replace()`` and activate it with ``set_*_context()`` or ``local_*_context()``.
Each ``ContextVar`` binding is local to its execution context, so asyncio tasks
and copied contexts can share a context object safely.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import replace as _replace_policy
from typing import Self

from chainfix.format import _integer
from chainfix.policy import Math, Overflow, ResultFormat, Rounding

_POLICY_FIELDS = ("rounding", "overflow", "product", "sum", "accumulator")
_FIELDS = frozenset(("wordlength", "precision", "math", *_POLICY_FIELDS))
_UNSET = object()


def _given(**settings) -> dict:
    """Drop settings left at None, which means "not specified"."""
    return {name: value for name, value in settings.items() if value is not None}


def _policy(current: Math, math, overrides: dict):
    """Select a whole Math or derive one from individual policy settings."""
    if math is _UNSET:
        return _replace_policy(current, **overrides) if overrides else current
    if overrides:
        raise ValueError("math cannot be combined with individual policy overrides")
    return math


class _Context:
    """Construction defaults and arithmetic policy for one base; immutable."""

    __slots__ = ("wordlength", "precision", "math")
    wordlength: int
    precision: int
    math: Math

    # Each concrete subclass supplies its defaults.
    _default_wordlength: int
    _default_precision: int
    _default_math = Math()

    def __init__(self, wordlength: int | None = None, precision: int | None = None,
                 overflow: Overflow | str | None = None, *, math: Math | None = None,
                 rounding: Rounding | str | None = None,
                 product: ResultFormat | None = None, sum: ResultFormat | None = None,
                 accumulator: ResultFormat | None = None):
        overrides = _given(rounding=rounding, overflow=overflow, product=product,
                           sum=sum, accumulator=accumulator)
        self._initialize(
            self._default_wordlength if wordlength is None else wordlength,
            self._default_precision if precision is None else precision,
            _policy(self._default_math, _UNSET if math is None else math, overrides))

    def _initialize(self, wordlength: int, precision: int, math: Math) -> None:
        if _integer(wordlength, "wordlength") < 1:
            raise ValueError("wordlength must be positive")
        _integer(precision, "precision")
        if not isinstance(math, Math):
            raise TypeError("math must be a Math")
        self._check_math(math)
        object.__setattr__(self, "wordlength", wordlength)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "math", math)

    def _check_math(self, math: Math) -> None:
        """Reject policies this kind of context cannot apply."""

    def replace(self, **changes) -> Self:
        """Return a context with the given settings changed; this one is unchanged.

        Accepts ``wordlength``, ``precision``, and either ``math`` or individual
        ``rounding``, ``overflow``, ``product``, ``sum``, and ``accumulator`` settings.
        """
        unknown = changes.keys() - _FIELDS
        if unknown:
            raise TypeError(f"unknown context setting(s): {', '.join(sorted(unknown))}")
        overrides = {name: changes[name] for name in _POLICY_FIELDS if name in changes}
        result = object.__new__(type(self))
        result._initialize(changes.get("wordlength", self.wordlength),
                           changes.get("precision", self.precision),
                           _policy(self.math, changes.get("math", _UNSET), overrides))
        return result

    __replace__ = replace  # copy.replace() on Python 3.13+

    @property
    def overflow(self) -> Overflow:
        return self.math.overflow

    @property
    def rounding(self) -> Rounding:
        return self.math.rounding

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable; use replace()")

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} is immutable; use replace()")

    def _key(self) -> tuple:
        return self.wordlength, self.precision, self.math

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self), self._key()))

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(wordlength={self.wordlength}, "
                f"precision={self.precision}, math={self.math!r})")

    def __reduce__(self):
        return _restore_context, (type(self), *self._key())

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo) -> Self:
        return self


def _restore_context(cls, wordlength, precision, math):
    # Unpickling validates through the public constructor.
    return cls(wordlength, precision, math=math)


class BinaryContext(_Context):
    """Binary construction defaults and arithmetic policy (immutable)."""

    __slots__ = ()
    _default_wordlength = 32
    _default_precision = 16


class DecimalContext(_Context):
    """Decimal construction defaults and policy (immutable).

    Decimal arithmetic is unsupported, so every result boundary must be "full".
    Decimal construction does not yet apply ``math``.
    """

    __slots__ = ()
    _default_wordlength = 256
    _default_precision = 18

    def _check_math(self, math: Math) -> None:
        for name in ("product", "sum", "accumulator"):
            if getattr(math, name) != "full":
                raise ValueError(f"decimal contexts require {name}='full'")


DefaultBinaryContext = BinaryContext()
DefaultDecimalContext = DecimalContext()

# An immutable default can serve as the ContextVar default directly: an empty
# execution context sees it, and nothing can change it in place.
_binary_context: ContextVar[BinaryContext] = ContextVar(
    "chainfix_binary", default=DefaultBinaryContext)
_decimal_context: ContextVar[DecimalContext] = ContextVar(
    "chainfix_decimal", default=DefaultDecimalContext)


def _checked(context, kind: type[_Context]):
    if not isinstance(context, kind):
        raise TypeError(f"context must be a {kind.__name__}")
    return context


@contextmanager
def _scope(variable: ContextVar, kind: type[_Context], context, changes: dict) -> Iterator:
    # Validate before binding so a rejected change leaves the variable untouched.
    base = variable.get() if context is None else _checked(context, kind)
    scoped = base.replace(**changes) if changes else base
    token = variable.set(scoped)
    try:
        yield scoped
    finally:
        variable.reset(token)


def get_binary_context() -> BinaryContext:
    """Return the binary context active in the current execution context."""
    return _binary_context.get()


def set_binary_context(context: BinaryContext) -> None:
    """Activate a binary context for the rest of the current execution context."""
    _binary_context.set(_checked(context, BinaryContext))


def local_binary_context(context: BinaryContext | None = None, *,
                         wordlength: int | None = None, precision: int | None = None,
                         math: Math | None = None, rounding: Rounding | str | None = None,
                         overflow: Overflow | str | None = None,
                         product: ResultFormat | None = None, sum: ResultFormat | None = None,
                         accumulator: ResultFormat | None = None
                         ) -> AbstractContextManager[BinaryContext]:
    """Activate a binary context, optionally changed, until the block exits.

    The supplied context (or the active one) is unchanged; the previous binding
    is restored even when the block raises.
    """
    return _scope(_binary_context, BinaryContext, context, _given(
        wordlength=wordlength, precision=precision, math=math, rounding=rounding,
        overflow=overflow, product=product, sum=sum, accumulator=accumulator))


def get_decimal_context() -> DecimalContext:
    """Return the decimal context active in the current execution context."""
    return _decimal_context.get()


def set_decimal_context(context: DecimalContext) -> None:
    """Activate a decimal context for the rest of the current execution context."""
    _decimal_context.set(_checked(context, DecimalContext))


def local_decimal_context(context: DecimalContext | None = None, *,
                          wordlength: int | None = None, precision: int | None = None,
                          math: Math | None = None, rounding: Rounding | str | None = None,
                          overflow: Overflow | str | None = None
                          ) -> AbstractContextManager[DecimalContext]:
    """Activate a decimal context, optionally changed, until the block exits.

    The supplied context (or the active one) is unchanged; the previous binding
    is restored even when the block raises.
    """
    return _scope(_decimal_context, DecimalContext, context, _given(
        wordlength=wordlength, precision=precision, math=math, rounding=rounding,
        overflow=overflow))
