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

"""Compare scalar and array construction timings with a Git snapshot.

Example: python benchmarks/construction.py --baseline origin/main --cpu 2
Each case is timed with ambient context defaults (a context lookup) and, where
the API allows, with an explicit format and math (no lookup). Array cases run
only when NumPy is installed. This is a local diagnostic, not a CI timing gate;
imports, warmup, and source extraction are outside the timed regions.
"""

import argparse
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile

from real_dispatch import ROOT, snapshot

WORKER = """
import json, os, sys, timeit
from fractions import Fraction
from pathlib import Path
source, number, warmup, cpu = sys.argv[1:]
number, warmup = int(number), int(warmup)
if cpu != 'None':
    os.sched_setaffinity(0, {int(cpu)})
sys.path.insert(0, source)
import chainfix as cf
assert Path(cf.__file__).resolve().is_relative_to(Path(source).resolve())
F = cf.Format(base=2, signed=True, wordlength=16, precision=8)
F32 = cf.Format(base=2, signed=True, wordlength=32, precision=8)
M = cf.Math()
i, f, q, c = 3, 0.375, Fraction(3, 8), 0.375 - 0.25j
cases = {
    'Fixb(int)': 'cf.Fixb(i)',
    'Fixb(float)': 'cf.Fixb(f)',
    'Fixb(Fraction)': 'cf.Fixb(q)',
    'Fixb(int, format, math)': 'cf.Fixb(i, format=F, math=M)',
    'Fixb(float, format, math)': 'cf.Fixb(f, format=F, math=M)',
    'Fixb(Fraction, format, math)': 'cf.Fixb(q, format=F, math=M)',
    'Fixb32(float)': 'cf.Fixb32(f)',
    'Fixb32(float, format, math)': 'cf.Fixb32(f, format=F32, math=M)',
    'CFixb(complex)': 'cf.CFixb(c)',
    'CFixb(complex, format, math)': 'cf.CFixb(c, format=F, math=M)',
    'Fixd(float)': 'cf.Fixd(f)',
    'Fixd(float, 32, 4)': 'cf.Fixd(f, 32, 4)',
}
scale = {}
try:
    import numpy
except ImportError:
    pass
else:
    from chainfix import arrays as cfa
    values = [v for _ in range(1024) for v in (i, f, q, -i)]
    for name, expression in (('array(4096 mixed)', 'cfa.array(values, format=F)'),
                             ('array(4096 mixed, math)', 'cfa.array(values, format=F, math=M)')):
        cases[name] = expression
        scale[name] = 1000
results = {}
for name, expression in cases.items():
    timer = timeit.Timer(expression, globals=globals())
    count = max(1, number // scale.get(name, 1))
    timer.timeit(number=max(1, warmup // scale.get(name, 1)))
    results[name] = timer.timeit(number=count) * 1e9 / count
print(json.dumps(results))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="origin/main")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--number", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--cpu", type=int, help="pin each worker to one CPU on Linux")
    parser.add_argument("--output", type=Path, help="also save the JSON report")
    args = parser.parse_args()
    if min(args.rounds, args.number, args.warmup) < 1:
        parser.error("rounds, number, and warmup must be positive")
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "cpu": args.cpu,
              "rounds": args.rounds, "number": args.number, "warmup": args.warmup}
    with tempfile.TemporaryDirectory(prefix="chainfix-construction-") as temporary:
        root = Path(temporary)
        revisions = {"main": snapshot(args.baseline, root / "main"),
                     "candidate": "working source tree"}
        shutil.copytree(ROOT / "src", root / "candidate" / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        samples = {name: {} for name in revisions}
        names = list(revisions)
        for round_index in range(args.rounds):
            # Alternate process order to spread warmup and thermal drift across variants.
            order = names[round_index % len(names):] + names[:round_index % len(names)]
            for name in order:
                output = subprocess.check_output(
                    [sys.executable, "-I", "-c", WORKER, str(root / name / "src"),
                     str(args.number), str(args.warmup), str(args.cpu)], text=True)
                for case, nanoseconds in json.loads(output).items():
                    samples[name].setdefault(case, []).append(nanoseconds)
        report["revisions"] = revisions
        report["nanoseconds_per_call"] = {
            name: {case: {"median": statistics.median(values), "min": min(values),
                          "max": max(values)} for case, values in results.items()}
            for name, results in samples.items()}
        report["candidate_percent_vs_main"] = {
            case: round(100 * (statistics.median(samples["candidate"][case]) /
                               statistics.median(samples["main"][case]) - 1), 1)
            for case in samples["main"]}
    formatted = json.dumps(report, indent=2) + "\n"
    print(formatted, end="")
    if args.output:
        args.output.write_text(formatted)


if __name__ == "__main__":
    main()
