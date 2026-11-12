"""behavioural tests for ``scripts/check-all.sh``'s handling of a unit suite the pool gave no verdict on.

The manifest is ``fallback = "pool-only"``, so when the pool can't take the unit suite ``test.sh``
exits 2 instead of running it here. check-all then still runs the suites that never use the pool
(mcp 1.x, the OpenTelemetry floor, the sidecar) and fails at the end, saying the unit suite did not
run. A failing test still stops it at once.

Self-contained: check-all is staged in a miniature repo under ``tmp_path`` beside stand-in scripts
that record that they ran and exit as told, so nothing here runs a test or reaches the pool.

    ./scripts/test.sh tests/test_check_all_script.py
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check-all.sh"

#: the scripts check-all runs, in order
_STEPS = ("lint.sh", "typecheck.sh", "test.sh", "test-mcp-v1.sh", "test-otel-floor.sh", "test-sidecar.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="check-all.sh is bash; none on this host")


@dataclass(frozen=True)
class _Run:
    """what one invocation of the staged check-all did.

    :param returncode: its exit status
    :ptype returncode: int
    :param output: its stdout and stderr together
    :ptype output: str
    :param ran: the stand-in scripts that ran, in order
    :ptype ran: list[str]
    """

    returncode: int
    output: str
    ran: list[str]


def _check_all(tmp_path: Path, *, test_sh_exit: int) -> _Run:
    """Run a staged check-all whose ``test.sh`` exits ``test_sh_exit`` and every other step 0.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :param test_sh_exit: the unit step's exit status
    :ptype test_sh_exit: int
    :return: what the run did
    :rtype: _Run
    """
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(_SCRIPT, scripts / "check-all.sh")
    log = tmp_path / "ran.txt"
    for step in _STEPS:
        code = test_sh_exit if step == "test.sh" else 0
        (scripts / step).write_text(f'#!/bin/sh\necho "{step}" >> "{log}"\nexit {code}\n')
        (scripts / step).chmod(0o755)
    result = subprocess.run(
        ["bash", str(scripts / "check-all.sh")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    ran = log.read_text().splitlines() if log.exists() else []
    return _Run(result.returncode, result.stdout + result.stderr, ran)


def test_a_unit_suite_the_pool_gave_no_verdict_on_fails_the_gate_at_the_end(tmp_path: Path) -> None:
    """exit 2 from test.sh: every other suite still runs, and the gate fails naming the unit suite.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    run = _check_all(tmp_path, test_sh_exit=2)
    assert run.returncode == 2
    assert run.ran == list(_STEPS)
    assert "Unit suite NOT run" in run.output
    assert "When the pool is down" in run.output
    assert "All checks passed" not in run.output


def test_a_failing_test_stops_the_gate_at_once(tmp_path: Path) -> None:
    """a failure is the code's: check-all stops at the unit step with its status, as before.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    run = _check_all(tmp_path, test_sh_exit=1)
    assert run.returncode == 1
    assert run.ran == ["lint.sh", "typecheck.sh", "test.sh"]


def test_all_green_passes(tmp_path: Path) -> None:
    """every step passing ends in the usual line.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    run = _check_all(tmp_path, test_sh_exit=0)
    assert run.returncode == 0
    assert run.ran == list(_STEPS)
    assert "All checks passed" in run.output
