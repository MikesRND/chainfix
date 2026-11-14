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

A context holds construction defaults (word length and precision) and, for
binary values, the arithmetic policy (``Math``). Contexts are final values:
derive a changed one with ``replace()`` and activate it with
``set_*_context()`` or ``local_*_context()``. Each ``ContextVar`` binding is
local to its execution context, so asyncio tasks and copied contexts can share
a context object safely. There is no process-wide default to reconfigure.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace as _replace_policy
from types import MappingProxyType
from typing import ClassVar, Self, final

from chainfix.format import _integer
from chainfix.policy import Math, Overflow, ResultFormat, Rounding

_POLICY_FIELDS = ("rounding", "overflow", "product", "sum", "accumulator")


class _Context:
    """Behavior shared by the final, immutable context classes.

    Instances are only built by the ``_create`` classmethod, which always returns
    a new object; no instance method writes a slot.
    """

    __slots__ = ("wordlength", "precision")
    wordlength: int
    precision: int

    base: ClassVar[int]
    _fields: ClassVar[tuple[str, ...]] = ("wordlength", "precision")
    _settings: ClassVar[frozenset[str]] = frozenset(_fields)
    _defaults: ClassVar[Mapping[str, object]]

    def __init_subclass__(cls, **kwargs):
        # replace() and pickling rebuild exactly the declared fields.
        if cls.__module__ != __name__:
            raise TypeError(f"{cls.__mro__[1].__name__} is final and cannot be subclassed")
        super().__init_subclass__(**kwargs)

    @classmethod
    def _specified(cls, changes: dict) -> dict:
        """Reject unknown settings and drop None, which means "not specified"."""
        if not cls._settings.issuperset(changes):
            unknown = ", ".join(sorted(changes.keys() - cls._settings))
            raise TypeError(f"unknown context setting(s): {unknown}")
        return {name: value for name, value in changes.items() if value is not None}

    @classmethod
    def _fold(cls, current: Mapping[str, object], changes: dict) -> dict:
        """Translate convenience settings into field values."""
        return changes

    @classmethod
    def _build(cls, current: Mapping[str, object], changes: dict) -> Self:
        return cls._create({**current, **cls._fold(current, changes)})

    @classmethod
    def _create(cls, settings: Mapping[str, object]) -> Self:
        """Validate complete field values and return a new context."""
        if _integer(settings["wordlength"], "wordlength") < 1:
            raise ValueError("wordlength must be positive")
        _integer(settings["precision"], "precision")
        context = object.__new__(cls)
        for name in cls._fields:
            object.__setattr__(context, name, settings[name])
        return context

    def _current(self) -> dict:
        return {name: getattr(self, name) for name in self._fields}

    def replace(self, **changes) -> Self:
        """Return a context with the given settings changed.

        Settings left at None are not changed; with no changes, this context is
        returned. The original context is never modified.
        """
        changes = self._specified(changes)
        if not changes:
            return self
        return self._build(self._current(), changes)

    __replace__ = replace  # copy.replace() on Python 3.13+

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable; use replace()")

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} is immutable; use replace()")

    def _key(self) -> tuple:
        return tuple(getattr(self, name) for name in self._fields)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self), self._key()))

    def __repr__(self) -> str:
        settings = ", ".join(f"{name}={getattr(self, name)!r}" for name in self._fields)
        return f"{type(self).__name__}({settings})"

    def __reduce__(self):
        return _restore_context, (type(self), self._current())

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo) -> Self:
        return self


def _restore_context(cls, settings):
    # Unpickling validates exactly as construction does.
    return cls._build(cls._defaults, cls._specified(dict(settings)))


@final
class BinaryContext(_Context):
    """Binary construction defaults and arithmetic policy (final, immutable)."""

    __slots__ = ("math",)
    math: Math

    base = 2
    _fields = ("wordlength", "precision", "math")
    _settings = frozenset((*_fields, *_POLICY_FIELDS))
    _defaults = MappingProxyType({"wordlength": 32, "precision": 16, "math": Math()})

    def __new__(cls, wordlength: int | None = None, precision: int | None = None,
                overflow: Overflow | str | None = None, *, math: Math | None = None,
                rounding: Rounding | str | None = None,
                product: ResultFormat | None = None, sum: ResultFormat | None = None,
                accumulator: ResultFormat | None = None) -> Self:
        return cls._build(cls._defaults, cls._specified({
            "wordlength": wordlength, "precision": precision, "math": math,
            "rounding": rounding, "overflow": overflow, "product": product,
            "sum": sum, "accumulator": accumulator}))

    @classmethod
    def _fold(cls, current: Mapping[str, object], changes: dict) -> dict:
        if changes.keys().isdisjoint(_POLICY_FIELDS):
            return changes
        if "math" in changes:
            raise ValueError("math cannot be combined with individual policy overrides")
        overrides = {name: changes.pop(name) for name in _POLICY_FIELDS if name in changes}
        changes["math"] = _replace_policy(current["math"], **overrides)
        return changes

    @classmethod
    def _create(cls, settings: Mapping[str, object]) -> Self:
        if not isinstance(settings["math"], Math):
            raise TypeError("math must be a Math")
        return super()._create(settings)

    @property
    def overflow(self) -> Overflow:
        return self.math.overflow

    @property
    def rounding(self) -> Rounding:
        return self.math.rounding


@final
class DecimalContext(_Context):
    """Decimal construction defaults: word length and precision (final, immutable)."""

    __slots__ = ()

    base = 10
    _defaults = MappingProxyType({"wordlength": 256, "precision": 18})

    def __new__(cls, wordlength: int | None = None, precision: int | None = None) -> Self:
        return cls._build(cls._defaults, cls._specified(
            {"wordlength": wordlength, "precision": precision}))


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


def _scoped(variable: ContextVar, kind: type[_Context], context, changes: dict):
    # Validate before binding so a rejected change leaves the variable untouched.
    base = variable.get() if context is None else _checked(context, kind)
    return base.replace(**changes)


def get_binary_context() -> BinaryContext:
    """Return the binary context active in the current execution context."""
    return _binary_context.get()


def set_binary_context(context: BinaryContext) -> None:
    """Activate a binary context for the rest of the current execution context."""
    _binary_context.set(_checked(context, BinaryContext))


@contextmanager
def local_binary_context(context: BinaryContext | None = None, *,
                         wordlength: int | None = None, precision: int | None = None,
                         math: Math | None = None, rounding: Rounding | str | None = None,
                         overflow: Overflow | str | None = None,
                         product: ResultFormat | None = None, sum: ResultFormat | None = None,
                         accumulator: ResultFormat | None = None) -> Iterator[BinaryContext]:
    """Activate a binary context, optionally changed, until the block exits.

    The supplied context (or the active one) is unchanged; the previous binding
    is restored even when the block raises. Also usable as a decorator.
    """
    scoped = _scoped(_binary_context, BinaryContext, context, {
        "wordlength": wordlength, "precision": precision, "math": math,
        "rounding": rounding, "overflow": overflow, "product": product,
        "sum": sum, "accumulator": accumulator})
    token = _binary_context.set(scoped)
    try:
        yield scoped
    finally:
        _binary_context.reset(token)


def get_decimal_context() -> DecimalContext:
    """Return the decimal context active in the current execution context."""
    return _decimal_context.get()


def set_decimal_context(context: DecimalContext) -> None:
    """Activate a decimal context for the rest of the current execution context."""
    _decimal_context.set(_checked(context, DecimalContext))


@contextmanager
def local_decimal_context(context: DecimalContext | None = None, *,
                          wordlength: int | None = None,
                          precision: int | None = None) -> Iterator[DecimalContext]:
    """Activate a decimal context, optionally changed, until the block exits.

    The supplied context (or the active one) is unchanged; the previous binding
    is restored even when the block raises. Also usable as a decorator.
    """
    scoped = _scoped(_decimal_context, DecimalContext, context,
                     {"wordlength": wordlength, "precision": precision})
    token = _decimal_context.set(scoped)
    try:
        yield scoped
    finally:
        _decimal_context.reset(token)
