"""a process that never imports the memory collections still knows the ``memories`` table's tenancy once it starts.

The hub imported no ``MemoriesCollection``, so ``tenancy_of_table("memories")`` answered undeclared and a user
merge's record of it went out under ``_all``. This package lists its collection module under the
``threetears.collections`` entry points, and the startup loader
(:func:`threetears.core.collections.load_declared_collections`) imports it. Run in a fresh interpreter: this test
process may already hold the module from another test.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_SCRIPT = """
import sys
from threetears.core.collections import load_declared_collections
from threetears.core.collections.generation import tenancy_of_table
assert "threetears.agent.memory.collections" not in sys.modules, "nothing here imports the memory collections"
loaded = load_declared_collections()
assert "threetears.agent.memory.collections" in loaded.modules
assert "memories" in loaded.tables
print(tenancy_of_table("memories").kind)
"""


@pytest.mark.uncacheable("a fresh interpreter is the point: this test process may hold the module already")
def test_tenancy_of_memories_is_known_in_a_process_that_never_imported_its_module() -> None:
    result = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().splitlines()[-1] == "schema"
