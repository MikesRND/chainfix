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

"""Cost of the reduction stages on a FIR-style frame and a long inner product.

Compares exact reduction with a bounded product stage and with an ordered,
bounded accumulator. Policies are built once, outside the timed calls, and each
round repeats a call enough times to last about 20 ms, so microsecond-scale
cases are not dominated by one cold call. Diagnostic only: there is no timing
threshold.
"""

import argparse
import json
import os
import platform
import statistics
import sys
from time import perf_counter_ns

import numpy as np

import chainfix as cf
from chainfix import arrays as cfa


def workloads(frame, taps):
    q15 = cf.Format(base=2, signed=True, wordlength=16, precision=15)
    product = cf.Format(base=2, signed=True, wordlength=24, precision=20)
    accumulator = cf.Format(base=2, signed=True, wordlength=28, precision=20)
    rng = np.random.default_rng(20261114)

    def samples(*shape):
        return rng.integers(-32768, 32768, size=shape, dtype=np.int16)

    policies = {
        "exact": cf.Math(),
        "bounded product": cf.Math(product=product, rounding="floor", overflow="wrap"),
        "ordered accumulator": cf.Math(product=product, accumulator=accumulator,
                                       rounding="floor", overflow="wrap"),
    }
    cases = {}
    for kind, complex in (("real", False), ("complex", True)):
        tail = (2,) if complex else ()
        windows = cfa.from_raw(samples(frame, taps, *tail), format=q15, complex=complex)
        coefficients = cfa.from_raw(samples(taps, *tail), format=q15, complex=complex)
        a = cfa.from_raw(samples(frame, *tail), format=q15, complex=complex)
        b = cfa.from_raw(samples(frame, *tail), format=q15, complex=complex)
        for name, policy in policies.items():
            cases[f"{kind} matmul ({frame}, {taps}) @ ({taps},): {name}"] = (
                lambda w=windows, c=coefficients, p=policy: cfa.matmul(w, c, math=p))
            cases[f"{kind} dot ({frame},): {name}"] = (
                lambda a=a, b=b, p=policy: cfa.dot(a, b, math=p))
            if policy.product == "full" or policy.accumulator != "full":
                # A plain sum has no product stage to bound.
                plain = cf.Math(accumulator=policy.accumulator, rounding=policy.rounding,
                                overflow=policy.overflow)
                cases[f"{kind} sum ({frame},): {name}"] = lambda a=a, p=plain: cfa.sum(a, math=p)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=4096)
    parser.add_argument("--taps", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--round-ms", type=float, default=20.0,
                        help="approximate duration of one timed round")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    results = {}
    for name, function in workloads(args.frame, args.taps).items():
        function()
        # Size each round from the faster of two calls, so one slow sample
        # does not shorten every round.
        single = []
        for _ in range(2):
            start = perf_counter_ns()
            function()
            single.append(perf_counter_ns() - start)
        number = max(1, int(args.round_ms * 1e6 // max(1, min(single))))
        times = []
        for _ in range(args.rounds):
            start = perf_counter_ns()
            for _ in range(number):
                function()
            times.append((perf_counter_ns() - start) / number / 1e3)
        results[name] = {"median_us": statistics.median(times), "min_us": min(times),
                         "calls_per_round": number}
        print(f"{name:62s} {results[name]['median_us']:12.1f} us  ({number} calls/round)")
    report = {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
              "frame": args.frame, "taps": args.taps, "rounds": args.rounds,
              "round_ms": args.round_ms, "results": results}
    if args.output:
        with open(args.output, "w") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
