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

"""Regression coverage for installing default and custom contexts."""

from contextvars import Context

import pytest

from chainfix import Fixb, Fixd
from chainfix.context import (
    BinaryContext,
    DecimalContext,
    DefaultBinaryContext,
    DefaultDecimalContext,
    Overflow,
    get_binary_context,
    get_decimal_context,
    set_binary_context,
    set_decimal_context,
)


@pytest.mark.parametrize(
    "default,get_context,set_context,scalar",
    [
        (DefaultBinaryContext, get_binary_context, set_binary_context, Fixb),
        (DefaultDecimalContext, get_decimal_context, set_decimal_context, Fixd),
    ],
    ids=["binary", "decimal"],
)
def test_setting_default_copies_and_preserves_it(default, get_context, set_context, scalar):
    def check():
        expected = (default.wordlength, default.precision, default.overflow)
        set_context(default)
        current = get_context()
        assert current is not default
        assert (current.wordlength, current.precision, current.overflow) == expected

        current.wordlength = 20
        current.precision = 5
        current.overflow = Overflow.WRAP
        assert (scalar().wordlength, scalar().precision) == (20, 5)
        assert (default.wordlength, default.precision, default.overflow) == expected

        set_context(default)
        restored = get_context()
        assert restored is not current
        assert restored is not default
        assert (restored.wordlength, restored.precision, restored.overflow) == expected
        assert (scalar().wordlength, scalar().precision) == expected[:2]

    # Keep this regression independent of the ambient context and test ordering.
    Context().run(check)


@pytest.mark.parametrize(
    "context_type,get_context,set_context,scalar",
    [
        (BinaryContext, get_binary_context, set_binary_context, Fixb),
        (DecimalContext, get_decimal_context, set_decimal_context, Fixd),
    ],
    ids=["binary", "decimal"],
)
def test_setting_custom_context(context_type, get_context, set_context, scalar):
    def check():
        custom = context_type(wordlength=20, precision=5, overflow=Overflow.WRAP)
        set_context(custom)
        current = get_context()
        assert (current.wordlength, current.precision, current.overflow) == (20, 5, Overflow.WRAP)
        assert (scalar().wordlength, scalar().precision) == (20, 5)

    Context().run(check)
