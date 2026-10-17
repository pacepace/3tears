"""the directory form of the text-block check: the files every walker reads, decoded as utf-8."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.text_blocks import unbroken_text_in

_UNBROKEN = 'lines.append(f"      {excerpt}")  # café\n'


def test_a_module_under_a_root_is_reported_by_path_and_line(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    module = tmp_path / "pkg" / "listing.py"
    module.write_text('"""ünïcode in the docstring."""\n' + _UNBROKEN, encoding="utf-8")

    assert unbroken_text_in([tmp_path]) == [f'{module}:2: lines.append(f"      {{excerpt}}")  # café']


def test_a_directory_no_walker_reads_is_not_read(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "vendored.py").write_text(_UNBROKEN, encoding="utf-8")

    assert unbroken_text_in([tmp_path]) == []


def test_a_module_that_does_not_parse_is_skipped_not_raised(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def (:\n", encoding="utf-8")
    (tmp_path / "listing.py").write_text(_UNBROKEN, encoding="utf-8")

    assert [entry.split(":")[0] for entry in unbroken_text_in([tmp_path])] == [str(tmp_path / "listing.py")]
