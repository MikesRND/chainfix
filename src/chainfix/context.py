# Copyright 2021 PyDefi Development Team
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
#
# Some context code adapted from Python's decimal module
# Copyright (c) 2004 Python Software Foundation.

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator, Optional, Self

from chainfix.format import _integer
from chainfix.policy import Math, Overflow, ResultFormat, Rounding


class _Context:

    def __init__(self,
                 wordlength: Optional[int] = None,
                 precision: Optional[int] = None,
                 overflow: Optional[Overflow] = None):
        dc = self.get_default()
        self.precision = precision if precision is not None else dc.precision
        self.wordlength = wordlength if wordlength is not None else dc.wordlength
        self.overflow = overflow if overflow is not None else dc.overflow

    def copy(self):
        """Returns a deep copy from self."""
        nc = self.__class__(self.wordlength, self.precision, self.overflow)
        return nc

    __copy__ = copy

    def get_default(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return '{}(wordlength={}, precision={}, overflow={})'.format(
            self.__class__.__name__,
            self.wordlength,
            self.precision,
            self.overflow)


class DecimalContext(_Context):
    base = 10

    def get_default(self):
        try:
            dc = DefaultDecimalContext
        except NameError:
            dc = None

        return dc


class BinaryContext(_Context):
    """Mutable binary defaults; use local_binary_context for scoped isolation."""

    base = 2

    def __init__(self, wordlength: int | None = None, precision: int | None = None,
                 overflow: Overflow | str | None = None, *, math: Math | None = None,
                 rounding: Rounding | str | None = None,
                 product: ResultFormat | None = None, sum: ResultFormat | None = None,
                 accumulator: ResultFormat | None = None):
        default = self.get_default()
        self.wordlength = wordlength if wordlength is not None else (default.wordlength if default else 32)
        self.precision = precision if precision is not None else (default.precision if default else 16)
        overrides = {name: value for name, value in
                     (("rounding", rounding), ("overflow", overflow), ("product", product),
                      ("sum", sum), ("accumulator", accumulator)) if value is not None}
        if math is not None and overrides:
            raise ValueError("math cannot be combined with individual policy overrides")
        if math is not None:
            self.math = math
        else:
            self.math = replace(default.math if default else Math(), **overrides)

    @property
    def wordlength(self) -> int:
        return self._wordlength

    @wordlength.setter
    def wordlength(self, value: int):
        if _integer(value, "wordlength") < 1:
            raise ValueError("wordlength must be positive")
        self._wordlength = value

    @property
    def precision(self) -> int:
        return self._precision

    @precision.setter
    def precision(self, value: int):
        self._precision = _integer(value, "precision")

    @property
    def math(self) -> Math:
        return self._math

    @math.setter
    def math(self, value: Math):
        if not isinstance(value, Math):
            raise TypeError("math must be a Math")
        self._math = value

    @property
    def overflow(self) -> Overflow:
        return self.math.overflow

    @overflow.setter
    def overflow(self, value: Overflow | str):
        self.math = replace(self.math, overflow=value)

    @property
    def rounding(self) -> Rounding:
        return self.math.rounding

    @rounding.setter
    def rounding(self, value: Rounding | str):
        self.math = replace(self.math, rounding=value)

    def copy(self) -> Self:
        """Copy mutable defaults while sharing the immutable Math configuration."""
        return type(self)(self.wordlength, self.precision, math=self.math)

    # The base alias binds _Context.copy; it does not dispatch to this override.
    __copy__ = copy

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(wordlength={self.wordlength}, "
                f"precision={self.precision}, math={self.math!r})")

    def get_default(self):
        try:
            dc = DefaultBinaryContext
        except NameError:
            dc = None

        return dc


DefaultDecimalContext = DecimalContext(
    wordlength=256,
    precision=18,
    overflow=Overflow.SATURATE
)

DefaultBinaryContext = BinaryContext(
    wordlength=32,
    precision=16,
    overflow=Overflow.RAISE
)

# Context Functions

# The getcontext() and setcontext() function manage access to a thread-local
# current context.

import contextvars

_current_decimal_context_var = contextvars.ContextVar('chainfix_decimal')


def get_decimal_context():
    """Returns this thread's decimal context.

    If this thread does not yet have a context, returns
    a new context and sets this thread's context.
    New contexts are copies of DefaultDecimalContext.
    """
    try:
        return _current_decimal_context_var.get()
    except LookupError:
        context = DecimalContext()
        _current_decimal_context_var.set(context)
        return context


def set_decimal_context(context):
    """Set this thread's context to context."""
    if context in (DefaultDecimalContext,):
        context = context.copy()
    _current_decimal_context_var.set(context)


_current_binary_context_var = contextvars.ContextVar('chainfix_binary')


def get_binary_context():
    """Returns this thread's binary context.

    If this thread does not yet have a context, returns
    a new context and sets this thread's context.
    New contexts are copies of DefaultBinaryContext.
    """
    try:
        return _current_binary_context_var.get()
    except LookupError:
        context = BinaryContext()
        _current_binary_context_var.set(context)
        return context


def set_binary_context(context):
    """Validate and copy binary defaults into the current execution context."""
    if not isinstance(context, BinaryContext):
        raise TypeError("context must be a BinaryContext")
    _current_binary_context_var.set(context.copy())


@contextmanager
def local_binary_context(context: BinaryContext | None = None, *,
                         wordlength: int | None = None, precision: int | None = None,
                         math: Math | None = None, rounding: Rounding | str | None = None,
                         overflow: Overflow | str | None = None,
                         product: ResultFormat | None = None, sum: ResultFormat | None = None,
                         accumulator: ResultFormat | None = None) -> Iterator[BinaryContext]:
    """Install a copied context and restore the previous one even on failure."""
    if context is None:
        context = get_binary_context()
    if not isinstance(context, BinaryContext):
        raise TypeError("context must be a BinaryContext")
    scoped = context.copy()
    overrides = {name: value for name, value in
                 (("rounding", rounding), ("overflow", overflow), ("product", product),
                  ("sum", sum), ("accumulator", accumulator)) if value is not None}
    if math is not None and overrides:
        raise ValueError("math cannot be combined with individual policy overrides")
    if math is not None:
        scoped.math = math
    elif overrides:
        scoped.math = replace(scoped.math, **overrides)
    if wordlength is not None:
        scoped.wordlength = wordlength
    if precision is not None:
        scoped.precision = precision
    token = _current_binary_context_var.set(scoped)
    try:
        yield scoped
    finally:
        _current_binary_context_var.reset(token)


del contextvars  # Don't contaminate the namespace
