"""Enforcement: observe package must not import from metallm."""

from __future__ import annotations

from pathlib import Path

import pytest

_OBSERVE_SRC = Path(__file__).resolve().parents[2] / "src" / "threetears" / "observe"


def _python_files():
    return sorted(_OBSERVE_SRC.rglob("*.py"))


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_metallm_imports(path: Path):
    """No source file in observe may import from metallm."""
    content = path.read_text()
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "from src." not in stripped, (
            f"{path.name}:{i} imports from metallm (src.*): {stripped}"
        )
        assert "import src." not in stripped, (
            f"{path.name}:{i} imports from metallm (src.*): {stripped}"
        )
        assert "from metallm" not in stripped, (
            f"{path.name}:{i} imports from metallm: {stripped}"
        )
