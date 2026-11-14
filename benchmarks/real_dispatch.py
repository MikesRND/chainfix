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

"""Compare real operator timings with Git snapshots using one Python executable.

Example: python benchmarks/real_dispatch.py --baseline origin/main --before f5f0b07
This is a local diagnostic, not a CI timing gate. Imports, construction, warmup,
and source extraction are outside the timed regions. No dependencies are needed.
"""

import argparse
import io
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORKER = """
import json, os, sys, timeit
from pathlib import Path
source, number, warmup, cpu = sys.argv[1:]
if cpu != 'None':
    os.sched_setaffinity(0, {int(cpu)})
sys.path.insert(0, source)
import chainfix as cf
assert Path(cf.__file__).resolve().is_relative_to(Path(source).resolve())
a, b = cf.Fixb(0.75, 16, 14), cf.Fixb(0.5, 16, 14)
assert a + b == 1.25 and -a == -0.75
results = {}
for expression in ('a + b', '-a'):
    timer = timeit.Timer(expression, globals=globals())
    timer.timeit(number=int(warmup))
    results[expression] = timer.timeit(number=int(number)) * 1e9 / int(number)
print(json.dumps(results))
"""


def snapshot(ref, destination):
    sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", sha, "src"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        contents.extractall(destination, filter="data")
    return sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="origin/main")
    parser.add_argument("--before", help="optional previous PR revision")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--number", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=5_000)
    parser.add_argument("--cpu", type=int, help="pin each worker to one CPU on Linux")
    parser.add_argument("--output", type=Path, help="also save the JSON report")
    args = parser.parse_args()
    if min(args.rounds, args.number, args.warmup) < 1:
        parser.error("rounds, number, and warmup must be positive")
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "cpu": args.cpu,
              "rounds": args.rounds, "number": args.number, "warmup": args.warmup}
    with tempfile.TemporaryDirectory(prefix="chainfix-dispatch-") as temporary:
        root = Path(temporary)
        revisions = {"main": snapshot(args.baseline, root / "main")}
        if args.before:
            revisions["before"] = snapshot(args.before, root / "before")
        revisions["candidate"] = "working source tree"
        shutil.copytree(ROOT / "src", root / "candidate" / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        samples = {name: {"a + b": [], "-a": []} for name in revisions}
        names = list(revisions)
        for round_index in range(args.rounds):
            # Rotate process order to distribute warmup/thermal drift across variants.
            order = names[round_index % len(names):] + names[:round_index % len(names)]
            for name in order:
                output = subprocess.check_output(
                    [sys.executable, "-I", "-c", WORKER, str(root / name / "src"),
                     str(args.number), str(args.warmup), str(args.cpu)], text=True)
                for expression, nanoseconds in json.loads(output).items():
                    samples[name][expression].append(nanoseconds)
        report["revisions"] = revisions
        report["nanoseconds_per_operation"] = {
            name: {expression: {"median": statistics.median(values),
                                "min": min(values), "max": max(values), "samples": values}
                   for expression, values in results.items()}
            for name, results in samples.items()}
        report["candidate_percent_vs_main"] = {
            expression: 100 * (statistics.median(samples["candidate"][expression]) /
                               statistics.median(samples["main"][expression]) - 1)
            for expression in samples["main"]}
    formatted = json.dumps(report, indent=2) + "\n"
    print(formatted, end="")
    if args.output:
        args.output.write_text(formatted)


if __name__ == "__main__":
    main()
