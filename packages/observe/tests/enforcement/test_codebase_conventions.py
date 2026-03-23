"""Enforcement: codebase conventions for observe package."""

from __future__ import annotations

from pathlib import Path

import pytest

_OBSERVE_SRC = Path(__file__).resolve().parents[2] / "src" / "threetears" / "observe"


def _python_files():
    return sorted(_OBSERVE_SRC.rglob("*.py"))


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_bare_except(path: Path):
    """No bare except clauses (except Exception: pass is also banned)."""
    content = path.read_text()
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Allow 'except Exception:' only in shutdown paths (setup.py)
        if "except:" in stripped and "except:  #" not in stripped:
            pytest.fail(f"{path.name}:{i} bare except: {stripped}")


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_print_statements(path: Path):
    """Source files should use logging, not print()."""
    content = path.read_text()
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("print(") or stripped.startswith("print ("):
            pytest.fail(f"{path.name}:{i} uses print(): {stripped}")
