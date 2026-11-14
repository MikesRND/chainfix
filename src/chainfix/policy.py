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

"""Immutable binary arithmetic policies; values never carry a policy."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from chainfix.format import Format, _binary_format


class Overflow(Enum):
    SATURATE = 1
    WRAP = 2
    RAISE = 3


class Rounding(str, Enum):
    NEAREST_EVEN = "nearest_even"
    FLOOR = "floor"
    CEIL = "ceil"
    ZERO = "zero"
    NEAREST_UP = "nearest_up"
    NEAREST_AWAY = "nearest_away"


class FixedPointOverflowError(ValueError):
    """A rounded stored integer is outside its destination format's range."""


def _overflow(value: Overflow | str) -> Overflow:
    if isinstance(value, Overflow):
        return value
    if isinstance(value, str):
        try:
            return {"raise": Overflow.RAISE, "saturate": Overflow.SATURATE,
                    "wrap": Overflow.WRAP}[value]
        except KeyError:
            raise ValueError(f"unknown overflow mode: {value!r}") from None
    raise TypeError("overflow must be an Overflow or a mode name")


def _rounding(value: Rounding | str) -> Rounding:
    if not isinstance(value, (str, Rounding)):
        raise TypeError("rounding must be a Rounding or a mode name")
    return Rounding(value)


type ResultFormat = Literal["full"] | Format


@dataclass(frozen=True, slots=True, kw_only=True)
class Math:
    """Rounding, overflow, and independent operation quantization boundaries.

    ``product`` and ``sum`` bound elementary results. In array reductions
    ``product`` applies to each term, ``accumulator`` to every ordered addition,
    and ``sum`` to the final result.
    """

    rounding: Rounding | str = Rounding.NEAREST_EVEN
    overflow: Overflow | str = Overflow.RAISE
    product: ResultFormat = "full"
    sum: ResultFormat = "full"
    accumulator: ResultFormat = "full"

    def __post_init__(self):
        object.__setattr__(self, "rounding", _rounding(self.rounding))
        object.__setattr__(self, "overflow", _overflow(self.overflow))
        for name in ("product", "sum", "accumulator"):
            value = getattr(self, name)
            if isinstance(value, str):
                if value != "full":
                    raise ValueError(f"{name} must be 'full' or a binary Format")
            else:
                _binary_format(value)
