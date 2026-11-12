"""``scripts/integration_marks.py`` stays readable to the oldest ``python3`` that runs it: 3.11.

``scripts/test-integration.sh`` runs it with whatever ``python3`` is on PATH, which on a laptop may be the system's
3.12 or 3.13, not the workspace's 3.14. Syntax newer than 3.11 crashes every run there; it happened once, when the
3.14 formatter removed the parentheses from a two-exception ``except``. ``pyproject.toml`` pins the file's ruff
target to py311, and this checks the result: the file parses as 3.11, and runs under every older interpreter this
machine has.

    ./scripts/test.sh tests/test_integration_marks_script.py
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "integration_marks.py"

#: interpreters older than the workspace's that a machine may put first on PATH
_OLDER = ("python3.11", "python3.12", "python3.13", "/usr/bin/python3")


def test_it_parses_as_python_3_11() -> None:
    ast.parse(_SCRIPT.read_text(encoding="utf-8"), feature_version=(3, 11))


def test_it_runs_under_every_older_python_on_this_machine(tmp_path: Path) -> None:
    """each one found lists the marked file, and none crashes (none found: the parse above is the check)."""
    (tmp_path / "test_live.py").write_text("import pytest\n\npytestmark = pytest.mark.integration\n")
    (tmp_path / "test_unit.py").write_text("def test_y():\n    pass\n")
    found = sorted({path for name in _OLDER if (path := shutil.which(name)) is not None})
    for interpreter in found:
        version = subprocess.run(
            [interpreter, "-c", "import sys; print(sys.version_info >= (3, 14))"],
            capture_output=True,
            text=True,
            check=True,
        )
        if version.stdout.strip() == "True":
            continue
        ran = subprocess.run(
            [interpreter, "-I", str(_SCRIPT), str(tmp_path)], capture_output=True, text=True, check=False
        )
        assert ran.returncode == 0, f"{interpreter}: {ran.stderr}"
        assert ran.stdout.splitlines() == [str(tmp_path / "test_live.py")], interpreter
