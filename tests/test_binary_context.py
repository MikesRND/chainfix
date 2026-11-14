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

"""Policy precedence, immutable contexts with replace(), and scope isolation."""

import asyncio
from contextvars import Context
from copy import copy, deepcopy
from fractions import Fraction

import pytest

import chainfix as cf
from chainfix.context import DefaultBinaryContext, DefaultDecimalContext


def format_at(precision):
    return cf.Format(base=2, signed=True, wordlength=16, precision=precision)


@pytest.fixture
def configured():
    return cf.BinaryContext(20, -3, math=cf.Math(
        rounding="nearest_away", overflow="wrap", product=format_at(3),
        sum=format_at(4), accumulator=format_at(5)))


def test_replace_preserves_unchanged_settings_and_the_original(configured):
    resized = configured.replace(wordlength=21, precision=7)
    assert resized is not configured
    assert resized.math is configured.math
    assert (resized.wordlength, resized.precision) == (21, 7)
    assert (configured.wordlength, configured.precision) == (20, -3)
    modes = configured.replace(rounding="floor", overflow="saturate")
    assert modes.math == cf.Math(rounding="floor", overflow="saturate", product=format_at(3),
                                 sum=format_at(4), accumulator=format_at(5))
    assert modes.rounding is cf.Rounding.FLOOR
    assert modes.overflow is cf.Overflow.SATURATE
    assert configured.rounding is cf.Rounding.NEAREST_AWAY
    assert configured.overflow is cf.Overflow.WRAP
    stages = configured.replace(product="full", sum=format_at(1), accumulator="full")
    assert (stages.math.product, stages.math.sum, stages.math.accumulator) == (
        "full", format_at(1), "full")
    assert stages.rounding is cf.Rounding.NEAREST_AWAY
    reset = configured.replace(math=cf.Math())
    assert reset.overflow is cf.Overflow.RAISE
    assert reset.wordlength == 20
    assert configured.replace() == configured
    with pytest.raises(ValueError):
        configured.replace(math=cf.Math(), rounding="floor")
    with pytest.raises(TypeError, match="unknown context setting"):
        configured.replace(base=10)
    for kwargs in ({"wordlength": 0}, {"wordlength": None}, {"precision": 1.5},
                   {"math": None}, {"rounding": "invalid"}, {"overflow": None}):
        with pytest.raises((TypeError, ValueError)):
            configured.replace(**kwargs)
    assert copy(configured) is configured
    assert deepcopy(configured) is configured


def test_binary_setter_installs_the_given_context(configured):
    def check():
        cf.set_binary_context(configured)
        assert cf.get_binary_context() is configured
        with pytest.raises(AttributeError):
            configured.precision = 10
        assert cf.Fixb(0).format.precision == -3
        cf.set_binary_context(configured.replace(precision=10))
        assert cf.Fixb(0).format.precision == 10
        assert configured.precision == -3
    Context().run(check)


def test_nested_scopes_restore_on_exception_and_setter(configured):
    original = cf.get_binary_context()
    with cf.local_binary_context(configured, wordlength=24, precision=6) as outer:
        assert cf.get_binary_context() is outer
        assert outer == configured.replace(wordlength=24, precision=6)
        assert outer.math is configured.math
        assert cf.Fixb(0).format == cf.Format(base=2, signed=True, wordlength=24, precision=6)
        with pytest.raises(RuntimeError):
            with cf.local_binary_context(rounding="zero", sum="full") as inner:
                assert inner is not outer
                assert inner.rounding is cf.Rounding.ZERO
                assert inner.math.sum == "full"
                assert inner.math.product == outer.math.product
                cf.set_binary_context(cf.BinaryContext(10, 2))
                raise RuntimeError("restore the entry token")
        assert cf.get_binary_context() is outer
    assert cf.get_binary_context() is original
    with cf.local_binary_context(configured) as unchanged:
        assert unchanged is configured
    assert (configured.wordlength, configured.precision) == (20, -3)


def test_separately_scoped_async_tasks(configured):
    async def worker(precision, overflow):
        with cf.local_binary_context(configured, precision=precision, overflow=overflow) as context:
            await asyncio.sleep(0)
            assert cf.get_binary_context() is context
            assert context.precision == precision
            assert context.overflow.value == overflow.value
            # Rebinding inside one task never reaches its sibling or parent.
            cf.set_binary_context(context.replace(wordlength=context.wordlength + 1))
            await asyncio.sleep(0)
            return cf.Fixb(0).format, cf.get_binary_context().math
    async def run():
        parent = cf.get_binary_context()
        result = await asyncio.gather(worker(3, cf.Overflow.SATURATE), worker(7, cf.Overflow.WRAP))
        assert cf.get_binary_context() is parent
        return result
    result = Context().run(asyncio.run, run())
    assert [item[0].precision for item in result] == [3, 7]
    assert [item[0].wordlength for item in result] == [21, 21]
    assert [item[1].overflow for item in result] == [cf.Overflow.SATURATE, cf.Overflow.WRAP]
    assert (configured.wordlength, configured.precision) == (20, -3)


def test_explicit_conversion_and_reusable_function_policy():
    dsp = cf.Math(rounding="floor", overflow="wrap", product=format_at(2), sum=format_at(2))
    precise = cf.Math()
    def mix_add(x, gain, offset, *, math):
        with cf.local_binary_context(math=math):
            return x * gain + offset
    x = cf.Fixb(Fraction(5, 16), 16, 4)
    gain = cf.Fixb(Fraction(1, 2), 16, 4)
    expected = mix_add(x, gain, x, math=precise)
    with cf.local_binary_context(math=dsp):
        assert cf.Fixb(1.5, 16, 15).int == -16384
        with pytest.raises(cf.FixedPointOverflowError):
            cf.Fixb(1.5, 16, 15, math=precise)
        assert cf.Fixb(Fraction(7, 8), 8, 2).int == 3
        assert cf.Fixb(Fraction(7, 8), 8, 2, math=precise).int == 4
        assert cf.Fixb(Fraction(7, 8), 8, 3).quantize(format_at(2), math=precise).int == 4
        actual = mix_add(x, gain, x, math=precise)
        assert actual == expected and actual.format == expected.format
        assert cf.get_binary_context().math is dsp
        assert cf.multiply(x, gain, math=precise).precision == 8
        assert (x * gain).precision == 2
        assert (x * 2).precision == 2
    assert cf.get_binary_context().math == cf.Math()


def test_context_validation_and_conflicting_overrides(configured):
    assert cf.BinaryContext(8, 3, cf.Overflow.WRAP).overflow is cf.Overflow.WRAP
    assert cf.BinaryContext(rounding="ceil", product=format_at(2)).math.product == format_at(2)
    for kwargs in ({"rounding": "floor"}, {"overflow": "wrap"}, {"product": "full"},
                   {"sum": "full"}, {"accumulator": "full"}):
        with pytest.raises(ValueError):
            cf.BinaryContext(math=cf.Math(), **kwargs)
        before = cf.get_binary_context()
        with pytest.raises(ValueError):
            with cf.local_binary_context(math=cf.Math(), **kwargs):
                pass
        assert cf.get_binary_context() is before
    for value in (None, cf.DecimalContext(), "binary"):
        with pytest.raises(TypeError):
            cf.set_binary_context(value)
    with pytest.raises(TypeError):
        with cf.local_binary_context(cf.DecimalContext()):
            pass
    for kwargs in ({"wordlength": 0}, {"wordlength": True}, {"precision": 1.5}, {"math": "wrap"}):
        with pytest.raises((TypeError, ValueError)):
            cf.BinaryContext(**kwargs)
        with pytest.raises((TypeError, ValueError)):
            with cf.local_binary_context(**kwargs):
                pass
    snapshot = (configured.wordlength, configured.precision, configured.math)
    for attr, value in (("wordlength", 21), ("wordlength", -1), ("precision", True),
                        ("math", cf.Math()), ("math", None), ("rounding", "floor"),
                        ("overflow", "wrap"), ("unrelated", 1)):
        with pytest.raises(AttributeError, match="immutable; use replace"):
            setattr(configured, attr, value)
        assert (configured.wordlength, configured.precision, configured.math) == snapshot
    for attr in ("wordlength", "precision", "math", "rounding", "overflow"):
        with pytest.raises(AttributeError, match="immutable; use replace"):
            delattr(configured, attr)
    assert (configured.wordlength, configured.precision, configured.math) == snapshot


def test_decimal_defaults_and_unchanged_construction():
    assert DefaultDecimalContext.math == cf.Math()
    assert DefaultDecimalContext.overflow is cf.Overflow.RAISE
    assert DefaultBinaryContext.overflow is cf.Overflow.RAISE
    decimal = cf.DecimalContext(20, 3, cf.Overflow.WRAP)
    assert (decimal.wordlength, decimal.precision, decimal.overflow) == (20, 3, cf.Overflow.WRAP)
    changed = decimal.replace(precision=1)
    assert changed.precision == 1
    assert decimal.precision == 3
    for name in ("product", "sum", "accumulator"):
        with pytest.raises(ValueError, match="full"):
            cf.DecimalContext(**{name: format_at(2)})
        with pytest.raises(ValueError, match="full"):
            decimal.replace(**{name: format_at(2)})
    with pytest.raises(ValueError):
        cf.DecimalContext(wordlength=0)
    with pytest.raises(TypeError):
        cf.Ufixd(3.1) + cf.Ufixd(3.3)
    # Decimal construction does not apply the context's math yet.
    with cf.local_decimal_context(overflow="saturate"):
        with pytest.raises(ValueError) as error:
            cf.Fixd(1000, 8, 0)
    assert type(error.value) is ValueError
