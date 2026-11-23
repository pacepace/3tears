"""The start check of the delta files' directory: a process whose tables opt in to delta files does not start without
a directory it can write, and says how to fix it (threetears.core.collections.delta_files_check)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from threetears.core.collections.delta_files_check import DeltaFilesDirectoryError, require_delta_files_root
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema
from threetears.core.testing.l1_tables import TEST_L1_TABLE

ENV = "THREETEARS_DELTA_FILES_DIR"

FIXES = {
    "the cluster (the chart)": "toolPod.deltaFiles in the values file\nthen check it after the deploy",
    "the local replay pod": "scripts/replay.sh pod makes one",
}


def _schema(name: str, *, delta_files: bool) -> TableSchema:
    return TableSchema(
        name=name,
        primary_key="id",
        columns=[Column("id", STRING_TYPE)],
        l1_eviction=TEST_L1_TABLE,
        delta_files=delta_files,
    )


OPTED = [_schema("lsf_runs", delta_files=True), _schema("plain", delta_files=False), _schema("ap", delta_files=True)]


def _refused(environ: dict[str, str]) -> DeltaFilesDirectoryError:
    with pytest.raises(DeltaFilesDirectoryError) as caught:
        require_delta_files_root(OPTED, fixes=FIXES, environ=environ)
    return caught.value


def test_a_usable_directory_is_returned_and_left_as_it_was(tmp_path: Path) -> None:
    (tmp_path / "lsf_runs").mkdir()
    assert require_delta_files_root(OPTED, environ={ENV: str(tmp_path)}) == tmp_path
    # the probe is written, renamed and removed: nothing of it is left for a sweep or a fold to meet
    assert sorted(path.name for path in tmp_path.iterdir()) == ["lsf_runs"]


def test_tables_that_do_not_opt_in_are_never_checked() -> None:
    assert require_delta_files_root([_schema("plain", delta_files=False)], environ={}) is None
    assert require_delta_files_root([], environ={}) is None


def test_unset_is_refused_naming_the_tables_why_and_every_fix() -> None:
    error = _refused({})
    text = str(error)
    assert error.tables == ("ap", "lsf_runs")
    assert error.found == f"{ENV} is not set"
    assert "THIS PROCESS CANNOT START" in text
    assert "Found: THREETEARS_DELTA_FILES_DIR is not set." in text
    assert "ap, lsf_runs" in text and "plain" not in text
    assert "refused" in text and "ReadWriteMany" in text
    # each place the process runs, its fix on its own indented lines, and a test's fix always
    assert "  * the cluster (the chart):\n      toolPod.deltaFiles in the values file\n      then check it" in text
    assert "  * the local replay pod:\n      scripts/replay.sh pod makes one" in text
    assert "monkeypatch.setenv with tmp_path" in text


def test_empty_is_refused_as_unset() -> None:
    assert _refused({ENV: ""}).found == f"{ENV} is not set"


def test_a_relative_path_is_refused() -> None:
    assert "is not an absolute path" in _refused({ENV: "delta-files"}).found


def test_a_missing_directory_is_refused(tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    assert _refused({ENV: str(gone)}).found == f"{ENV}={gone} does not exist (nothing is mounted there)"


def test_a_file_is_refused(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("x")
    assert _refused({ENV: str(file)}).found == f"{ENV}={file} is not a directory"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a read-only directory")
def test_a_read_only_directory_is_refused(tmp_path: Path) -> None:
    tmp_path.chmod(0o555)
    try:
        found = _refused({ENV: str(tmp_path)}).found
    finally:
        tmp_path.chmod(0o755)
    assert found.startswith(f"{ENV}={tmp_path} is not writable by this process")
    assert list(tmp_path.iterdir()) == []
