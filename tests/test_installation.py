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

"""Check the installed distribution and the optional-dependency CI split."""

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import chainfix


def test_distribution_metadata():
    metadata = importlib.metadata.metadata("chainfix")
    assert metadata["Requires-Python"] == ">=3.12"
    assert metadata["Version"] == chainfix.__version__
    # Intentional baseline tripwire: when arrays lands, check that NumPy is
    # required only by the arrays extra and that the scalar core stays independent.
    assert not metadata.get_all("Requires-Dist")


def test_core_import_is_numpy_independent():
    # A fresh isolated interpreter catches transitive imports as well as direct ones.
    subprocess.run(
        [sys.executable, "-I", "-c", """
import importlib.abc
import sys

class RejectNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.'):
            raise AssertionError('Core import must not load NumPy')

sys.meta_path.insert(0, RejectNumpy())
import chainfix
assert chainfix.Fixb(0.5).as_integer_ratio() == (1, 2)
with chainfix.local_binary_context(math=chainfix.Math()):
    assert (chainfix.Fixb(0.5) * 2 + chainfix.Fixb(0.25)).as_integer_ratio() == (5, 4)
    z = chainfix.CFixb(0.5 + 0.25j)
    assert z * z.conjugate() == z.mag2() == 0.3125
    assert chainfix.add(z, chainfix.Ufixb(0.5)) == 1 + 0.25j
    assert chainfix.multiply(z, 2) == 1 + 0.5j
assert not any(name == 'numpy' or name.startswith('numpy.') for name in sys.modules)
"""],
        check=True,
    )


def test_tox_dependency_environment():
    mode = os.environ.get("CHAINFIX_TEST_NUMPY")
    if mode is None:
        return  # Local developer runs need not use the CI dependency split.

    # Tox must test its installed wheel, rather than accidentally importing src/.
    assert Path(chainfix.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    if mode == "absent":
        assert importlib.util.find_spec("numpy") is None
    else:
        version = importlib.metadata.version("numpy")
        if mode == "min":
            assert version == "2.5.0"
        else:
            assert mode == "latest"
            assert tuple(map(int, version.split(".")[:2])) >= (2, 5)
