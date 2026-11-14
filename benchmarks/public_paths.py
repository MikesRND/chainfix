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

"""Public array paths on the complex gain frame, for comparison across revisions.

Uses only the public Chainfix API, so the same script runs against any checkout
that has chainfix.arrays. Point PYTHONPATH (or an installed wheel) at each
revision, keep the runtime and arguments identical, and alternate run order.
"""

import argparse
import json
import os
import platform
import statistics
import sys
from time import perf_counter_ns
import tracemalloc

import numpy as np

import chainfix as cf
from chainfix import arrays as cfa


def peak_bytes(function):
    """One warmed call's global peak above its own current traced baseline."""
    baseline = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    function()
    return tracemalloc.get_traced_memory()[1] - baseline


def paths():
    q14 = cf.Format(base=2, signed=True, wordlength=16, precision=14)
    q12 = cf.Format(base=2, signed=True, wordlength=16, precision=12)
    rng = np.random.default_rng(20261114)
    raw = rng.integers(-32768, 32768, size=(4, 4096, 2), dtype=np.int16)
    raw.flags.writeable = False
    real_raw = np.ascontiguousarray(raw[..., 0])
    real_raw.flags.writeable = False
    z = cfa.from_raw(raw, format=q14, complex=True)
    x = cfa.from_raw(real_raw, format=q14)
    gain = cfa.array(0.5 - 0.25j, format=q14, math=cf.Math())
    policy = cf.Math(product=q14, rounding="nearest_even", overflow="raise")
    out = cfa.empty(z.shape, format=q12, complex=True)

    def expression():
        with cf.local_binary_context(math=policy):
            return (z * gain).quantize(q12)

    return raw, {
        "complex_expression": expression,
        "complex_public_out": lambda: cfa.multiply(z, gain, out=out, math=policy),
        "complex_full_multiply": lambda: cfa.multiply(z, gain, math=cf.Math()),
        "real_full_multiply": lambda: cfa.multiply(x, x, math=cf.Math()),
        "real_quantize_q14_q12_nearest_even": lambda: x.quantize(q12, math=policy),
        "real_quantize_same_format": lambda: x.quantize(q14, math=policy),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--number", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cpu", type=int, help="pin execution to a CPU on Linux")
    args = parser.parse_args()
    if min(args.rounds, args.number, args.warmup) < 1:
        parser.error("rounds, number, and warmup must be positive")
    if tracemalloc.is_tracing():
        parser.error("run without active tracemalloc so throughput excludes tracing overhead")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})

    raw, functions = paths()
    for function in functions.values():
        for _ in range(args.warmup):
            function()
    names = list(functions)
    timings = {name: [] for name in names}
    for index in range(args.rounds):
        for name in names[index % len(names):] + names[:index % len(names)]:
            function = functions[name]
            start = perf_counter_ns()
            for _ in range(args.number):
                function()
            timings[name].append((perf_counter_ns() - start) / args.number)

    tracemalloc.start()
    try:
        peaks = {name: [peak_bytes(function) for _ in range(5)]
                 for name, function in functions.items()}
    finally:
        tracemalloc.stop()

    report = {
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "chainfix_module": cf.__file__, "cpu": args.cpu,
        "rounds": args.rounds, "number": args.number, "warmup": args.warmup,
        "frame_shape": list(raw.shape), "frame_bytes": raw.nbytes,
        "paths": {
            name: {"median_ns": statistics.median(samples), "min_ns": min(samples),
                   "max_ns": max(samples), "peak_bytes": peaks[name]}
            for name, samples in timings.items()},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
