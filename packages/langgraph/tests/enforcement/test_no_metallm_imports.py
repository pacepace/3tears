"""Enforcement: langgraph package must not import from the parent product."""

from __future__ import annotations

from pathlib import Path

import pytest

_LANGGRAPH_SRC = Path(__file__).resolve().parents[2] / "src" / "threetears" / "langgraph"


def _python_files():
    return sorted(_LANGGRAPH_SRC.rglob("*.py"))


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_parent_product_imports(path: Path):
    """No source file in langgraph may import from the parent product."""
    content = path.read_text()
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "from src." not in stripped, f"{path.name}:{i} imports from the parent product (src.*): {stripped}"
        assert "import src." not in stripped, f"{path.name}:{i} imports from the parent product (src.*): {stripped}"
        assert "from metallm" not in stripped, f"{path.name}:{i} imports from the parent product: {stripped}"
