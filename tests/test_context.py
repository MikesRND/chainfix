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

"""Installing, scoping, and inheriting immutable binary and decimal contexts."""

import asyncio
import copy
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from types import SimpleNamespace

import pytest

import chainfix as cf
from chainfix import Fixb, Fixd, Overflow
from chainfix.context import (
    BinaryContext,
    DecimalContext,
    DefaultBinaryContext,
    DefaultDecimalContext,
    get_binary_context,
    get_decimal_context,
    local_binary_context,
    local_decimal_context,
    set_binary_context,
    set_decimal_context,
)

BINARY = SimpleNamespace(cls=BinaryContext, other=DecimalContext, default=DefaultBinaryContext,
                         library=(32, 16), get=get_binary_context, set=set_binary_context,
                         local=local_binary_context, scalar=Fixb)
DECIMAL = SimpleNamespace(cls=DecimalContext, other=BinaryContext, default=DefaultDecimalContext,
                          library=(256, 18), get=get_decimal_context, set=set_decimal_context,
                          local=local_decimal_context, scalar=Fixd)
KINDS = pytest.mark.parametrize("kind", [BINARY, DECIMAL], ids=["binary", "decimal"])


def sizes(scalar):
    return scalar.wordlength, scalar.precision


@KINDS
def test_default_is_installed_without_copy_and_is_frozen(kind):
    def check():
        assert kind.get() is kind.default
        kind.set(kind.default)
        assert kind.get() is kind.default
        for attr, value in (("wordlength", 20), ("precision", 5), ("overflow", Overflow.WRAP)):
            with pytest.raises(AttributeError):
                setattr(kind.default, attr, value)
        assert sizes(kind.scalar()) == kind.library
    Context().run(check)
    # Nothing can reconfigure the library defaults that an empty context sees.
    assert (kind.default.wordlength, kind.default.precision) == kind.library
    assert Context().run(lambda: sizes(kind.scalar())) == kind.library


@KINDS
def test_setting_custom_context(kind):
    def check():
        custom = kind.cls(wordlength=20, precision=5, overflow=Overflow.WRAP)
        kind.set(custom)
        current = kind.get()
        assert current is custom
        assert (current.wordlength, current.precision, current.overflow) == (20, 5, Overflow.WRAP)
        assert sizes(kind.scalar()) == (20, 5)
        with pytest.raises(AttributeError):
            custom.precision = 8
        assert sizes(kind.scalar()) == (20, 5)
    Context().run(check)


@KINDS
def test_setters_and_scopes_reject_other_kinds(kind):
    for value in (None, kind.other(), "context"):
        with pytest.raises(TypeError):
            kind.set(value)
        with pytest.raises(TypeError):
            with kind.local(value if value is not None else kind.other()):
                pass


@KINDS
@pytest.mark.parametrize("primed", [False, True], ids=["unprimed", "primed"])
def test_child_tasks_cannot_change_the_parent_context(kind, primed):
    async def child():
        with pytest.raises(AttributeError):
            kind.get().rounding = "floor"
        kind.set(kind.get().replace(rounding="floor"))
        return kind.get().rounding

    async def parent():
        if primed:
            kind.get()
        before = kind.get()
        assert await asyncio.create_task(child()) is cf.Rounding.FLOOR
        assert kind.get() is before
        return kind.get().rounding

    assert Context().run(asyncio.run, parent()) is cf.Rounding.NEAREST_EVEN


@KINDS
def test_local_scopes_restore_nest_and_validate_before_binding(kind):
    def check():
        before = kind.get()
        with pytest.raises(RuntimeError):
            with kind.local(precision=3) as outer:
                assert kind.get() is outer
                assert outer.precision == 3
                with kind.local(outer, wordlength=24, rounding="ceil") as inner:
                    assert (inner.wordlength, inner.precision) == (24, 3)
                    assert inner.rounding is cf.Rounding.CEIL
                    assert sizes(kind.scalar()) == (24, 3)
                assert kind.get() is outer
                raise RuntimeError("restore the entry binding")
        assert kind.get() is before
        for bad in ({"wordlength": 0}, {"precision": 1.5}, {"rounding": "invalid"},
                    {"math": cf.Math(), "overflow": "wrap"}, {"math": "full"}):
            with pytest.raises((TypeError, ValueError)):
                with kind.local(**bad):
                    pass
            assert kind.get() is before
    Context().run(check)


@KINDS
def test_contexts_are_values(kind):
    context = kind.cls(20, 5, Overflow.WRAP)
    same = kind.cls(20, 5, Overflow.WRAP)
    assert context == same and hash(context) == hash(same)
    assert context != kind.cls(20, 6, Overflow.WRAP)
    assert context != kind.other(20, 5, Overflow.WRAP)
    assert context != (20, 5)
    restored = pickle.loads(pickle.dumps(context))
    assert restored == context and type(restored) is kind.cls
    assert copy.copy(context) is context
    assert copy.deepcopy(context) is context
    for attr in ("wordlength", "precision", "math", "rounding", "overflow", "unrelated"):
        with pytest.raises(AttributeError, match="immutable; use replace"):
            delattr(context, attr)
    with pytest.raises(AttributeError, match="immutable; use replace"):
        context.unrelated = 1
    assert repr(context).startswith(f"{kind.cls.__name__}(wordlength=20, precision=5, math=Math(")
    if sys.version_info >= (3, 13):
        assert copy.replace(context, precision=6) == context.replace(precision=6)


@KINDS
def test_empty_and_copied_execution_contexts(kind):
    def configure_and_observe():
        kind.set(kind.cls(20, 5))
        caller = kind.get()
        copied = copy_context()
        assert copied.run(kind.get) is caller

        def rebind():
            kind.set(kind.cls(21, 6))
            return kind.get()

        assert Context().run(rebind) == kind.cls(21, 6)
        assert copied.run(rebind) == kind.cls(21, 6)
        assert kind.get() is caller
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(copy_context().run, kind.get).result() is caller
        return Context().run(kind.get)

    assert Context().run(configure_and_observe) is kind.default


@KINDS
@pytest.mark.skipif(sys.version_info < (3, 14), reason="Thread(context=...) requires Python 3.14")
def test_thread_context_argument(kind):
    def configure():
        custom = kind.cls(20, 5)
        kind.set(custom)
        seen = {}

        def observe(name):
            seen[name] = kind.get()
            kind.set(kind.cls(21, 6))

        for name, context in (("empty", Context()), ("copied", copy_context())):
            thread = threading.Thread(target=observe, args=(name,), context=context)
            thread.start()
            thread.join()
        assert seen["empty"] is kind.default
        assert seen["copied"] is custom
        assert kind.get() is custom

    Context().run(configure)
