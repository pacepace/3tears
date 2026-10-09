"""Core imports without the optional NATS client.

The client is core's ``nats`` extra. Only code that takes a cross-pod lock needs
it; importing a module must not, or every consumer that builds on
``DerivedCollection`` (``threetears.geo`` among them) fails to import wherever
the extra is not installed.
"""

from __future__ import annotations

import subprocess
import sys

#: refuses the client's distributions, as an install without the extra would
BLOCK_CLIENT = """
import importlib.abc, sys
class _NoClient(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("nats", "nkeys"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NoClient())
"""


def test_derived_collection_imports_without_the_nats_client() -> None:
    code = BLOCK_CLIENT + "import threetears.core.collections.derived\nprint('imported')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_versioned_answers_import_without_the_nats_client() -> None:
    code = BLOCK_CLIENT + "import threetears.core.collections.versioned_answers\nprint('imported')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"
