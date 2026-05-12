"""tests for ``Violation`` dataclass formatting."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.common.violations import Violation


def test_format_path_under_repo_root(tmp_path: Path) -> None:
    """relative paths are rendered with forward slashes and category in brackets."""
    repo = tmp_path
    file_path = repo / "src" / "pkg" / "mod.py"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("")
    violation = Violation(
        category="cache.missing_collection",
        file=file_path,
        line=42,
        symbol="orders",
        reason="no Collection class declared",
    )
    rendered = violation.format(repo)
    assert rendered == "[cache.missing_collection] src/pkg/mod.py:42:orders  -- no Collection class declared"


def test_format_path_outside_repo_root(tmp_path: Path) -> None:
    """paths outside the repo render as the absolute posix path."""
    repo = tmp_path / "consumer"
    repo.mkdir()
    other_repo = tmp_path / "neighbour"
    other_repo.mkdir()
    file_path = other_repo / "src" / "thing.py"
    file_path.parent.mkdir()
    file_path.write_text("")
    violation = Violation(
        category="underscore.A",
        file=file_path,
        line=1,
        symbol="_helper",
        reason="cross-package private import",
    )
    rendered = violation.format(repo)
    assert "underscore.A" in rendered
    assert rendered.endswith(":1:_helper  -- cross-package private import")
    assert "/neighbour/src/thing.py" in rendered


def test_format_shape_is_consistent() -> None:
    """the rendered string has the canonical shape regardless of inputs."""
    violation = Violation(
        category="x.y",
        file=Path("/abs/file.py"),
        line=3,
        symbol="Sym",
        reason="r",
    )
    rendered = violation.format(Path("/other"))
    assert rendered.startswith("[x.y] ")
    assert ":3:Sym  -- r" in rendered
