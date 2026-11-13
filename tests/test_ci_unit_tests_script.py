"""behavioural tests for ``scripts/ci-unit-tests.sh``'s guard against a local whole-suite run.

Outside CI the script runs the unit suite on the machine it is on, for hours, which is the load the
testament pool exists to take. So, like ``test.sh``'s ``TESTAMENT=off``, a local run states why in
``TESTAMENT_OFF_REASON``; ``--loop-timing`` (six files, serial) needs no reason, and CI is unchanged.

Self-contained: the script is staged in a miniature repo under ``tmp_path``, and ``uv`` is a stand-in
on PATH that records that it ran, so nothing here runs a test.

    ./scripts/test.sh tests/test_ci_unit_tests_script.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci-unit-tests.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="ci-unit-tests.sh is bash; none on this host")


def _run(tmp_path: Path, args: list[str], env: dict[str, str]) -> tuple[int, str, bool]:
    """Run the staged script; return its status, its stderr, and whether ``uv`` ran.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :param args: the script's arguments
    :ptype args: list[str]
    :param env: extra environment for the run
    :ptype env: dict[str, str]
    :return: the exit status, stderr, and whether the ``uv`` stand-in ran
    :rtype: tuple[int, str, bool]
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "packages" / "core").mkdir(parents=True)
    (repo / "tests").mkdir()
    shutil.copy(_SCRIPT, repo / "scripts" / "ci-unit-tests.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "uv-ran"
    (bin_dir / "uv").write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 0\n')
    (bin_dir / "uv").chmod(0o755)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "ci-unit-tests.sh"), *args],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result.returncode, result.stderr, marker.exists()


@pytest.mark.parametrize("args", [pytest.param([], id="whole-suite"), pytest.param(["--shard", "core"], id="shard")])
def test_outside_ci_without_a_reason_it_runs_nothing(tmp_path: Path, args: list[str]) -> None:
    """no CI, no reason: exit 2, nothing run, and the runbook named.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :param args: the script's arguments
    :ptype args: list[str]
    :return: nothing
    :rtype: None
    """
    status, stderr, ran = _run(tmp_path, args, {})
    assert status == 2
    assert not ran
    assert "When the pool is down" in stderr


@pytest.mark.parametrize(
    ("args", "env"),
    [
        pytest.param([], {"TESTAMENT_OFF_REASON": "pool down"}, id="with-a-reason"),
        pytest.param(["--shard", "core"], {"CI": "true"}, id="ci"),
        pytest.param(["--loop-timing"], {}, id="loop-timing"),
    ],
)
def test_ci_a_stated_reason_and_loop_timing_run(tmp_path: Path, args: list[str], env: dict[str, str]) -> None:
    """CI, a stated reason, or the serial loop_timing session run as before.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :param args: the script's arguments
    :ptype args: list[str]
    :param env: extra environment for the run
    :ptype env: dict[str, str]
    :return: nothing
    :rtype: None
    """
    status, stderr, ran = _run(tmp_path, args, env)
    assert status == 0, stderr
    assert ran
