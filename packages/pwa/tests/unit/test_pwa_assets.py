"""The shipped browser files and the command that copies them into an app."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.pwa import STATIC_FILES, export_static
from threetears.pwa.__main__ import main


def test_every_shipped_file_is_exported(tmp_path: Path) -> None:
    assert set(STATIC_FILES) == {"pwa.js", "safe-area.css"}
    export_static(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(STATIC_FILES)
    assert b"export class MediaSessionControl" in (tmp_path / "pwa.js").read_bytes()
    assert b"env(safe-area-inset-bottom" in (tmp_path / "safe-area.css").read_bytes()


def test_export_writes_then_skips_what_is_current(tmp_path: Path) -> None:
    dest = tmp_path / "static" / "pwa"
    assert [p.name for p in export_static(dest)] == list(STATIC_FILES)
    shipped = (dest / "pwa.js").read_bytes()
    assert export_static(dest) == []
    (dest / "pwa.js").write_text("stale")
    assert [p.name for p in export_static(dest)] == ["pwa.js"]
    assert (dest / "pwa.js").read_bytes() == shipped


def test_the_command_exports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["export-static", str(tmp_path)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert main(["export-static", str(tmp_path)]) == 0
    assert "up to date" in capsys.readouterr().out
