"""3tears-geo imports without core's optional NATS client.

3tears-geo builds on ``DerivedCollection`` and declares plain ``3tears``, so it
must import wherever the ``nats`` extra is not installed.
"""

from __future__ import annotations

import subprocess
import sys

BLOCK_CLIENT = """
import importlib.abc, sys
class _NoClient(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("nats", "nkeys"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NoClient())
"""


def test_geo_imports_without_the_nats_client() -> None:
    code = BLOCK_CLIENT + "import threetears.geo\nprint('imported')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"
