"""tests for the CLI: --update idempotency and verify exit codes."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.integration_skill import SENTINEL
from threetears.enforcement.integration_skill.__main__ import main


def _minimal_skill(tmp_path: Path) -> Path:
    path = tmp_path / "SKILL.md"
    path.write_text(f"---\nname: x\n---\n\n> {SENTINEL}\n")
    return path


class TestCli:
    def test_update_then_verify_is_green(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = _minimal_skill(tmp_path)
        assert main([str(path), "--update"]) == 0
        assert main([str(path)]) == 0
        out = capsys.readouterr().out
        assert "STALE" not in out

    def test_update_is_idempotent(self, tmp_path: Path) -> None:
        path = _minimal_skill(tmp_path)
        main([str(path), "--update"])
        first = path.read_bytes()
        main([str(path), "--update"])
        assert path.read_bytes() == first

    def test_verify_flags_tampered_fingerprint(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = _minimal_skill(tmp_path)
        main([str(path), "--update"])
        tampered = path.read_text().replace("sha256:", "sha256:f00d", 1)
        path.write_text(tampered)
        assert main([str(path)]) == 1
        assert "STALE" in capsys.readouterr().out

    def test_verify_without_attestations_fails(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = _minimal_skill(tmp_path)
        assert main([str(path)]) == 1
        assert "nothing is verified" in capsys.readouterr().out
