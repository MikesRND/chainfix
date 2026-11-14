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

"""Standalone complex gain fixture and diagnostic benchmark for Chainfix.

Run with a NumPy-enabled Chainfix interpreter:
    python benchmarks/complex_gain.py --cpu 2 --output /tmp/complex-gain.json

The reusable implementation deliberately exercises a PRIVATE workspace. This
fixture is not a public API or a timing gate for CI.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
import tracemalloc

import numpy as np

import chainfix as cf
from chainfix import arrays as cfa
from chainfix._array_workspace import _ComplexMultiplyWorkspace


class ComplexGain:
    """Explicit configuration with current input/output binding on every call."""

    def __init__(self, shape, input_format, gain_format, output_format, *, gain, math):
        self.input_format, self.output_format = input_format, output_format
        self.math = math
        # Coefficient setup has its own explicit conversion policy, outside any
        # processing scope whose product policy could affect setup expressions.
        self.gain = cfa.array(gain, format=gain_format, complex=True, math=cf.Math())
        if self.gain.shape:
            raise ValueError("standalone gain requires a scalar coefficient")
        self.workspace = _ComplexMultiplyWorkspace(
            shape, input_format, gain_format, output_format, math=math)

    def reference(self, raw):
        x = cfa.from_raw(raw, format=self.input_format, complex=True)
        with cf.local_binary_context(math=self.math):
            return (x * self.gain).quantize(self.output_format)

    def public_out(self, raw, out):
        x = cfa.from_raw(raw, format=self.input_format, complex=True)
        return cfa.multiply(x, self.gain, out=out, math=self.math)

    def process(self, raw, out):
        return self.workspace.execute(raw, self.gain.raw, out)

    def allocating_control(self, raw, out):
        # The frame-sized temporary is actually used and freed before returning.
        temporary = raw.copy()
        return self.process(temporary, out)


def representative_gain():
    q14 = cf.Format(base=2, signed=True, wordlength=16, precision=14)
    q12 = cf.Format(base=2, signed=True, wordlength=16, precision=12)
    return ComplexGain((4, 4096), q14, q14, q12, gain=0.5-0.25j,
                       math=cf.Math(product=q14, rounding="nearest_even", overflow="raise"))


def peak_bytes(function):
    """One warmed call's global peak above its own current traced baseline."""
    baseline = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    function()
    _, peak = tracemalloc.get_traced_memory()
    return peak - baseline


def retained_numpy_bytes(function):
    """Additional diagnostic only: freed temporaries disappear from snapshots."""
    domain = tracemalloc.DomainFilter(True, np.lib.tracemalloc_domain)
    before = tracemalloc.take_snapshot().filter_traces([domain])
    function()
    after = tracemalloc.take_snapshot().filter_traces([domain])
    return sum(stat.size_diff for stat in after.compare_to(before, "traceback"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--number", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cpu", type=int, help="pin execution to a CPU on Linux")
    parser.add_argument("--output", type=Path, help="also save the JSON report")
    args = parser.parse_args()
    if min(args.rounds, args.number, args.warmup) < 1:
        parser.error("rounds, number, and warmup must be positive")
    if tracemalloc.is_tracing():
        parser.error("run without active tracemalloc so throughput excludes tracing overhead")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})

    setup_samples = []
    for _ in range(args.rounds):
        start = perf_counter_ns()
        gain = representative_gain()
        setup_samples.append(perf_counter_ns() - start)
        del gain
    gain = representative_gain()
    raw = np.random.default_rng(20261114).integers(
        -32768, 32768, size=(4, 4096, 2), dtype=np.int16)
    raw.flags.writeable = False
    out = cfa.empty(raw.shape[:-1], format=gain.output_format, complex=True)
    expected = gain.reference(raw).raw
    functions = {
        "expression": lambda: gain.reference(raw),
        "public_out": lambda: gain.public_out(raw, out),
        "reusable": lambda: gain.process(raw, out.raw),
    }
    for function in functions.values():
        for _ in range(args.warmup):
            function()
    np.testing.assert_array_equal(out.raw, expected)
    identities = tuple(id(buffer) for buffer in gain.workspace.buffers)
    output_identity = id(out.raw)

    timings = {name: [] for name in functions}
    names = list(functions)
    for index in range(args.rounds):
        for name in names[index % len(names):] + names[:index % len(names)]:
            function = functions[name]
            start = perf_counter_ns()
            for _ in range(args.number):
                function()
            timings[name].append((perf_counter_ns() - start) / args.number)

    # Throughput above excludes tracing; this separate phase measures temporary
    # memory. Persistent storage is reported independently from ndarray.nbytes.
    functions["allocating_control"] = lambda: gain.allocating_control(raw, out.raw)
    tracemalloc.start()
    try:
        peaks, retained = {}, {}
        for name, function in functions.items():
            for _ in range(args.warmup):
                function()
            peaks[name] = [peak_bytes(function) for _ in range(5)]
            retained[name] = retained_numpy_bytes(function)
    finally:
        tracemalloc.stop()
    assert tuple(id(buffer) for buffer in gain.workspace.buffers) == identities
    assert id(out.raw) == output_identity
    np.testing.assert_array_equal(out.raw, expected)

    report = {
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "chainfix_module": cf.__file__, "cpu": args.cpu,
        "rounds": args.rounds, "number": args.number, "warmup": args.warmup,
        "shape": list(raw.shape), "dtype": str(raw.dtype), "seed": 20261114,
        "input_format": asdict(gain.input_format), "gain_format": asdict(gain.gain.format),
        "gain_raw": gain.gain.raw.tolist(), "output_format": asdict(gain.output_format),
        "math": {"product": asdict(gain.math.product), "rounding": gain.math.rounding.value,
                 "overflow": gain.math.overflow.name.lower()},
        "setup_ns": {"median": statistics.median(setup_samples), "samples": setup_samples,
                     "includes": "coefficient construction, validation and workspace allocation"},
        "frame_bytes": raw.nbytes, "output_bytes": out.raw.nbytes,
        "coefficient_bytes": gain.gain.raw.nbytes,
        "workspace_bytes": sum(buffer.nbytes for buffer in gain.workspace.buffers),
        "temporary_peak_bytes": peaks, "retained_numpy_bytes": retained,
        "nanoseconds_per_frame": {
            name: {"median": statistics.median(samples), "min": min(samples), "max": max(samples),
                   "samples": samples,
                   "complex_samples_per_second": raw[..., 0].size * 1e9 / statistics.median(samples)}
            for name, samples in timings.items()},
    }
    formatted = json.dumps(report, indent=2) + "\n"
    print(formatted, end="")
    if args.output:
        args.output.write_text(formatted)


if __name__ == "__main__":
    main()
