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

"""Private serial workspace for the standalone complex gain experiment.

This is not a supported external binding/preparation API. It accepts current raw
buffers on every call, trusts externally written encodings to fit their declared
formats, and checks storage metadata/overlap before execution. Rebuild when shape,
formats or policy change; concurrent execution requires independent instances.
"""

from chainfix import arrays as cfa  # Keep the optional dependency's import error.
from chainfix import _array_kernels as kernels
from chainfix.policy import Math

import numpy as np


class _ComplexMultiplyWorkspace:
    """One (channels, samples) complex frame times one complex scalar gain."""

    __slots__ = ("_shape", "_input_format", "_gain_format", "_output_format", "_math",
                 "_full", "_target", "_input", "_gain", "_i", "_q", "_work",
                 "_conversion", "_ai", "_aq")

    def __init__(self, shape, input_format, gain_format, output_format, *, math):
        shape = tuple(shape)
        if len(shape) != 2 or any(isinstance(n, (bool, np.bool_)) or
                                  not isinstance(n, (int, np.integer)) or n < 0 for n in shape):
            raise ValueError("workspace shape must be two nonnegative integer dimensions")
        if not isinstance(math, Math):
            raise TypeError("workspace requires an explicit Math configuration")
        for format in (input_format, gain_format, output_format):
            kernels.check_format(format, True)
        full = cfa._multiply_format(input_format, gain_format, True, True)
        target, _ = cfa._prepare(full, math.product, True, shape, (), None)
        kernels.check_conversion(target, output_format)
        self._shape = shape
        self._input_format, self._gain_format = input_format, gain_format
        self._output_format, self._math = output_format, math
        self._full, self._target = full, target
        # Promote before multiplying. Casting narrow operands inside a ufunc
        # can allocate iterator buffers, even when that ufunc has an out array.
        self._input = np.empty(shape + (2,), dtype=np.int64)
        self._gain = np.empty(2, dtype=np.int64)
        self._ai, self._aq = self._input[..., 0], self._input[..., 1]
        self._i, self._q, self._work = (np.empty(shape, dtype=np.int64) for _ in range(3))
        # Product scratch is dead before conversion; reuse it for remainders.
        self._conversion = kernels._ConversionScratch(shape, remainder=self._work)

    @property
    def buffers(self):
        """Distinct persistent allocations, exposed only for diagnostics/tests."""
        return (self._input, self._gain, self._i, self._q, self._work,
                self._conversion.increment, self._conversion.mask, self._conversion.predicate)

    def execute(self, raw, gain, out):
        """Write the current disjoint output and return that same raw ndarray.

        The private binding trusts value ranges, as array arithmetic does. Use
        public from_raw to validate external data before relying on this contract.
        No caller input/output buffer or wrapper is retained between calls.
        """
        for value, format in ((raw, self._input_format), (gain, self._gain_format),
                              (out, self._output_format)):
            cfa._check_storage(value, format, True)
        if raw.shape != self._shape + (2,) or out.shape != raw.shape or gain.shape != (2,):
            raise ValueError("workspace requires its configured frame shape and a scalar gain")
        if not out.flags.writeable:
            raise ValueError("out must be writable")
        if np.may_share_memory(out, raw) or np.may_share_memory(out, gain):
            raise ValueError("out may overlap an input; use a disjoint destination")
        buffers = self.buffers
        if any(np.may_share_memory(value, work) for value in (raw, gain, out) for work in buffers):
            raise ValueError("bindings must not overlap private workspace storage")
        np.copyto(self._input, raw)
        np.copyto(self._gain, gain)
        kernels.multiply_complex(self._ai, self._aq, self._gain[0], self._gain[1],
                                 i=self._i, q=self._q, work=self._work)
        for component in (self._i, self._q):
            cfa._convert_result(component, self._full, self._target, self._output_format,
                                self._math, out=component, scratch=self._conversion)
        # Both quantization boundaries have completed before narrowing storage.
        np.copyto(out[..., 0], self._i, casting="unsafe")
        np.copyto(out[..., 1], self._q, casting="unsafe")
        return out
