"""DuckDB's default database let down to one thread by every module that imports DuckDB
(:mod:`threetears.core.utils.duckdb_module`)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

#: a fresh process's DuckDB default database's threads, once it imports a module that opens DuckDB
_DEFAULT_THREADS = """
import duckdb
import threetears.core.cache.duckdb
[(setting,)] = duckdb.execute("SELECT current_setting('threads')").fetchall()
print(setting)
"""


#: the repository's packages, whose source is scanned for DuckDB imports
_PACKAGES = Path(__file__).resolve().parents[4]


def test_a_module_that_imports_duckdb_leaves_its_default_database_on_one_thread() -> None:
    """Flips if the default database DuckDB opens as it loads keeps its thread a core: idle in every process that
    imports DuckDB, 63 of them on a 64-core pool host, against its container's pid limit (2026-11-25)."""
    done = subprocess.run([sys.executable, "-c", _DEFAULT_THREADS], capture_output=True, text=True, timeout=60)

    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["1"]


def test_every_module_that_imports_duckdb_lets_its_default_threads_go() -> None:
    """Flips if a module of a package imports DuckDB without calling ``release_default_threads``: whichever module
    imports it first must be one that lets the default database's threads go."""
    imports = re.compile(r"^\s*(import duckdb\b|from duckdb\b)", re.MULTILINE)
    found = [
        str(path.relative_to(_PACKAGES))
        for path in sorted(_PACKAGES.glob("*/src/**/*.py")) + sorted(_PACKAGES.glob("*/*/src/**/*.py"))
        if path.name != "duckdb_module.py"
        and imports.search(text := path.read_text(encoding="utf-8"))
        and "release_default_threads()" not in text
    ]
    assert found == []
