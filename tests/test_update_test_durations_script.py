"""behavioural tests for ``scripts/update-test-durations.sh``, the estimator behind ``.test_durations``.

pytest-split balances the evals CI shards by the seconds in ``.test_durations``. The script reads
them from the testament pool's history: a measured test gets its median, every other collected
test of a file shares what the file's mean per test-run leaves over (never below zero), and a test
of a file the pool has no time for gets the median of those shares. Ids carry the
``@packages/evals`` suffix xdist adds before pytest-split reads them. These tests pin that
arithmetic, the file's shape, and the refusals.

Self-contained: the script is staged in a miniature repo under ``tmp_path``, and ``uv`` and
``testament`` are stand-ins: ``uv run pytest`` prints a canned collection, ``uv run python`` runs
the script's own estimator, and ``testament`` serves canned envelopes. Nothing here collects the
real suite or reaches the pool.

    ./scripts/test.sh tests/test_update_test_durations_script.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "update-test-durations.sh"

_A = "packages/evals/tests/test_a.py"
_B = "packages/evals/tests/test_b.py"
_C = "packages/evals/tests/test_c.py"
_D = "packages/evals/tests/test_d.py"

_COLLECTED = [f"{_A}::test_1", f"{_A}::test_2", f"{_A}::test_3", f"{_B}::test_1", f"{_B}::test_2"]
_COLLECTED += [f"{_C}::test_1", f"{_D}::test_1", f"{_D}::test_2"]

#: `uv run pytest` prints the collection (or fails); `uv run python` runs the estimator
_UV = """#!/bin/sh
printf '%s\\n' "$*" >> "$STAGE/uv.calls"
case "$1 $2" in
    "run pytest") [ -f "$STAGE/collect.fail" ] && exit 1; cat "$STAGE/ids.txt" ;;
    "run python") shift 2; exec python3 "$@" ;;
    *) exit 9 ;;
esac
"""

_TESTAMENT = """#!/bin/sh
printf '%s\\n' "$*" >> "$STAGE/testament.calls"
case "$1 $2" in
    "tests slowest") cat "$STAGE/slowest.json" ;;
    "time --by") cat "$STAGE/files.json" ;;
    *) exit 9 ;;
esac
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="the script is bash; none on this host")


@dataclass(frozen=True)
class _Run:
    """what one invocation of the staged script did.

    :param returncode: the script's exit status
    :ptype returncode: int
    :param stderr: what it printed to stderr
    :ptype stderr: str
    :param durations: the .test_durations it wrote, parsed, or None when it wrote none
    :ptype durations: dict[str, float] | None
    :param text: the file's text, or empty when it wrote none
    :ptype text: str
    :param testament: each testament call's arguments
    :ptype testament: list[str]
    """

    returncode: int
    stderr: str
    durations: dict[str, float] | None
    text: str
    testament: list[str]


def _ok(rows: list[dict[str, object]]) -> str:
    """A successful testament ``--json`` envelope holding ``rows``.

    :param rows: the result's rows
    :ptype rows: list[dict[str, object]]
    :return: the envelope as JSON
    :rtype: str
    """
    return json.dumps({"ok": True, "result": {"rows": rows}})


#: the pool's medians: a::test_1 and d::test_1 measured, and one test no longer collected
_SLOWEST = [
    {"item_id": f"{_A}::test_1", "median_s": 10.0},
    {"item_id": f"{_D}::test_1", "median_s": 9.0},
    {"item_id": "packages/evals/tests/test_gone.py::test_1", "median_s": 99.0},
]
#: each file's mean seconds per test-run is test_s / runs; test_c.py has none, and a file with no runs is skipped
_FILES = [
    {"folder": _A, "test_s": 60.0, "runs": 12},
    {"folder": _B, "test_s": 5.0, "runs": 10},
    {"folder": _D, "test_s": 10.0, "runs": 10},
    {"folder": _C, "test_s": 7.0, "runs": 0},
]


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    """A miniature repo with the script, stand-ins on PATH, and the canned answers.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: the directory holding repo/, bin/ and stage/
    :rtype: Path
    """
    (tmp_path / "repo" / "scripts").mkdir(parents=True)
    shutil.copy(_SCRIPT, tmp_path / "repo" / "scripts" / "update-test-durations.sh")
    (tmp_path / "bin").mkdir()
    (tmp_path / "stage").mkdir()
    for name, body in (("uv", _UV), ("testament", _TESTAMENT)):
        (tmp_path / "bin" / name).write_text(body)
        (tmp_path / "bin" / name).chmod(0o755)
    (tmp_path / "bin" / "python3").symlink_to(sys.executable)
    (tmp_path / "stage" / "ids.txt").write_text("\n".join([*_COLLECTED, "", "8 tests collected in 0.1s"]) + "\n")
    (tmp_path / "stage" / "slowest.json").write_text(_ok(_SLOWEST))
    (tmp_path / "stage" / "files.json").write_text(_ok(_FILES))
    return tmp_path


def _run(stage: Path, *args: str, testament: bool = True) -> _Run:
    """Run the staged script.

    :param stage: the fixture's directory
    :ptype stage: Path
    :param args: the script's arguments
    :ptype args: str
    :param testament: whether testament is on PATH
    :ptype testament: bool
    :return: what it did
    :rtype: _Run
    """
    if not testament:
        (stage / "bin" / "testament").unlink()
    result = subprocess.run(
        ["bash", str(stage / "repo" / "scripts" / "update-test-durations.sh"), *args],
        cwd=stage,
        env={"PATH": f"{stage / 'bin'}:/usr/bin:/bin", "STAGE": str(stage / "stage")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    out = stage / "repo" / ".test_durations"
    calls = stage / "stage" / "testament.calls"
    return _Run(
        returncode=result.returncode,
        stderr=result.stderr,
        durations=json.loads(out.read_text()) if out.exists() else None,
        text=out.read_text() if out.exists() else "",
        testament=calls.read_text().splitlines() if calls.exists() else [],
    )


def _ids(durations: dict[str, float]) -> dict[str, float]:
    """The durations with the xdist suffix taken off each id, asserting every id carried it.

    :param durations: the written file, parsed
    :ptype durations: dict[str, float]
    :return: node id to seconds
    :rtype: dict[str, float]
    """
    suffix = "@packages/evals"
    assert all(key.endswith(suffix) for key in durations), list(durations)
    return {key.removesuffix(suffix): value for key, value in durations.items()}


def test_the_estimator(stage: Path) -> None:
    """measured tests keep their medians; the rest share what their file's mean leaves over.

    test_a.py: mean 5 s a run, 3 tests, 15 s; test_1 measured 10, so test_2 and test_3 get 2.5 each.
    test_b.py: mean 0.5, nothing measured: 0.5 each. test_d.py: mean 1, 2 tests, 2 s, but test_1
    alone measured 9, so test_2 gets 0, never less. test_c.py has no runs: it gets the median
    share (of 2.5, 0.5 and 0), 0.5.

    :param stage: the fixture's directory
    :ptype stage: Path
    :return: nothing
    :rtype: None
    """
    run = _run(stage)
    assert run.returncode == 0, run.stderr
    assert run.durations is not None
    assert _ids(run.durations) == {
        f"{_A}::test_1": 10.0,
        f"{_A}::test_2": 2.5,
        f"{_A}::test_3": 2.5,
        f"{_B}::test_1": 0.5,
        f"{_B}::test_2": 0.5,
        f"{_C}::test_1": 0.5,
        f"{_D}::test_1": 9.0,
        f"{_D}::test_2": 0.0,
    }
    assert "8 tests, 2 measured" in run.stderr


def test_every_collected_test_and_nothing_else(stage: Path) -> None:
    """the file lists the collection: a measured test no longer collected is dropped.

    :param stage: the fixture's directory
    :ptype stage: Path
    :return: nothing
    :rtype: None
    """
    run = _run(stage)
    assert run.durations is not None
    assert sorted(_ids(run.durations)) == sorted(_COLLECTED)


def test_one_sorted_entry_a_line(stage: Path) -> None:
    """a refresh diffs line by line.

    :param stage: the fixture's directory
    :ptype stage: Path
    :return: nothing
    :rtype: None
    """
    lines = _run(stage).text.splitlines()
    assert lines[0] == "{" and lines[-1] == "}"
    entries = lines[1:-1]
    assert len(entries) == len(_COLLECTED)
    assert entries == sorted(entries)


def test_with_no_file_times_every_unmeasured_test_gets_a_tenth(stage: Path) -> None:
    """with no file has a mean, there is no share to take the median of: 0.1 s each.

    :param stage: the fixture's directory
    :ptype stage: Path
    :return: nothing
    :rtype: None
    """
    (stage / "stage" / "files.json").write_text(_ok([]))
    run = _run(stage)
    assert run.durations is not None
    ids = _ids(run.durations)
    assert ids[f"{_A}::test_1"] == 10.0
    assert ids[f"{_B}::test_1"] == 0.1 and ids[f"{_C}::test_1"] == 0.1


def test_the_window_reaches_both_queries(stage: Path) -> None:
    """``--since`` sets the window of both testament queries; 14d without it.

    :param stage: the fixture's directory
    :ptype stage: Path
    :return: nothing
    :rtype: None
    """
    run = _run(stage, "--since", "30d")
    assert len(run.testament) == 2
    assert all("--since 30d" in call and "--area evals" in call for call in run.testament)
    (stage / "stage" / "testament.calls").unlink()
    assert all("--since 14d" in call for call in _run(stage).testament)


def test_a_refused_query_writes_nothing(stage: Path) -> None:
    """an envelope that is not ok stops the script, naming the query.

    :param stage: the fixture's directory
    :ptype stage: Path
    :return: nothing
    :rtype: None
    """
    (stage / "stage" / "slowest.json").write_text(json.dumps({"ok": False, "error": {"code": "forbidden"}}))
    run = _run(stage)
    assert run.returncode != 0
    assert "testament refused slowest.json" in run.stderr
    assert run.durations is None


def test_a_failed_collection_writes_nothing(stage: Path) -> None:
    """no collection, no file.

    :param stage: the fixture's directory
    :ptype stage: Path
    :return: nothing
    :rtype: None
    """
    (stage / "stage" / "collect.fail").write_text("")
    run = _run(stage)
    assert run.returncode == 1
    assert "collection failed" in run.stderr
    assert run.durations is None and not run.testament


def test_without_testament_it_refuses(stage: Path) -> None:
    """the seconds come from the pool, so no client is exit 2.

    :param stage: the fixture's directory
    :ptype stage: Path
    :return: nothing
    :rtype: None
    """
    run = _run(stage, testament=False)
    assert run.returncode == 2
    assert "testament is not on PATH" in run.stderr
