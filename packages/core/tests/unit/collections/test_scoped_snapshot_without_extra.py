"""the scoped snapshot module imports without the snapshot extra, and names it when a snapshot needs it.

duckdb and pyarrow are the ``snapshot`` extra's. The module imports them at its top under ``try/except
ImportError``, so a process that imports it without the extra still starts, and coding a chunk raises the
ImportError naming the extra. Run in a fresh interpreter in which neither can be imported.
"""

from __future__ import annotations

import subprocess
import sys

_SCRIPT = """
import sys
for name in ("duckdb", "pyarrow", "pyarrow.ipc"):
    sys.modules[name] = None
from threetears.core.collections import scoped_snapshot
try:
    scoped_snapshot.decode_chunk(b"")
except ImportError as exc:
    print("refused:", exc)
    sys.exit(0)
sys.exit(1)
"""


def test_the_module_imports_and_a_chunk_names_the_extra() -> None:
    result = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "3tears[snapshot]" in result.stdout
