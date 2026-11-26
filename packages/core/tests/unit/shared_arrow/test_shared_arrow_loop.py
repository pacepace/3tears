"""The module DuckDB scans Arrow through is imported when shared Arrow is, never by a worker's first query.

The serving process runs no DuckDB over the shared copy (no engine thread, no cursor): its queries run in query
workers (``collections.copy_queries``), so what its loop owed to threads querying the copy is gone with them.
"""

from __future__ import annotations

import subprocess
import sys


class TestTheScanModuleIsImportedWithSharedArrow:
    def test_importing_shared_arrow_imports_what_duckdb_scans_arrow_through(self) -> None:
        """DuckDB imports ``pyarrow.dataset`` at a process's first scan of a registered table (25-70 ms); importing
        shared Arrow does it first, so no worker's first query pays it."""
        code = "import sys, threetears.core.shared_arrow; print('pyarrow.dataset' in sys.modules)"
        answer = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert answer.stdout.strip() == "True"
