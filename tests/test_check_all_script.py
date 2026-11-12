"""behavioural tests for ``scripts/check-all.sh``'s handling of a suite the pool gave no verdict on.

The manifest is ``fallback = "pool-only"``, so when the pool can't take the unit suite ``test.sh``
exits 2 instead of running it here, and the integration suite (``test-integration.sh``, also on the
pool) does the same. check-all then still runs the suites that never use the pool (mcp 1.x, the
OpenTelemetry floor, the sidecar) and fails at the end, saying which suite did not run. A failing test
still stops it at once.

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
_STEPS = (
    "lint.sh",
    "typecheck.sh",
    "test.sh",
    "test-integration.sh",
    "test-mcp-v1.sh",
    "test-otel-floor.sh",
    "test-sidecar.sh",
)

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


def _check_all(
    tmp_path: Path,
    *,
    test_sh_exit: int,
    integration_exit: int = 0,
    testament: bool = True,
    env: dict[str, str] | None = None,
) -> _Run:
    """Run a staged check-all whose ``test.sh`` exits ``test_sh_exit``, its integration step ``integration_exit``,
    and every other step 0.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :param test_sh_exit: the unit step's exit status
    :ptype test_sh_exit: int
    :param integration_exit: the integration step's exit status
    :ptype integration_exit: int
    :param testament: whether a ``testament`` is on PATH (the pool route)
    :ptype testament: bool
    :param env: extra environment for the run
    :ptype env: dict[str, str] | None
    :return: what the run did
    :rtype: _Run
    """
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(_SCRIPT, scripts / "check-all.sh")
    log = tmp_path / "ran.txt"
    for step in _STEPS:
        code = {"test.sh": test_sh_exit, "test-integration.sh": integration_exit}.get(step, 0)
        (scripts / step).write_text(f'#!/bin/sh\necho "{step}" >> "{log}"\nexit {code}\n')
        (scripts / step).chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if testament:
        (bin_dir / "testament").write_text("#!/bin/sh\nexit 0\n")
        (bin_dir / "testament").chmod(0o755)
    result = subprocess.run(
        ["bash", str(scripts / "check-all.sh")],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", **(env or {})},
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


def test_an_integration_suite_the_pool_gave_no_verdict_on_fails_the_gate_at_the_end(tmp_path: Path) -> None:
    """exit 2 from the integration step: every other suite still runs, and the gate fails naming it.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    run = _check_all(tmp_path, test_sh_exit=0, integration_exit=2)
    assert run.returncode == 2
    assert run.ran == list(_STEPS)
    assert "Integration suite NOT run" in run.output
    assert "Unit suite NOT run" not in run.output
    assert "When the pool is down" in run.output


def test_both_suites_without_a_verdict_are_both_named(tmp_path: Path) -> None:
    """the pool down for both: both named, the rest still run.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    run = _check_all(tmp_path, test_sh_exit=2, integration_exit=2)
    assert run.returncode == 2
    assert run.ran == list(_STEPS)
    assert "Unit suite NOT run" in run.output and "Integration suite NOT run" in run.output


def test_a_failing_integration_test_stops_the_gate_at_once(tmp_path: Path) -> None:
    """a failure is the code's: check-all stops at the integration step with its status.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    run = _check_all(tmp_path, test_sh_exit=0, integration_exit=1)
    assert run.returncode == 1
    assert run.ran == ["lint.sh", "typecheck.sh", "test.sh", "test-integration.sh"]


@pytest.mark.parametrize(
    ("env", "testament", "why"),
    [
        pytest.param({}, False, "testament is not on PATH", id="no-testament"),
        pytest.param({"CI": "true"}, True, "CI is set", id="ci"),
        pytest.param({"TESTAMENT": "off"}, True, "TESTAMENT=off", id="testament-off"),
    ],
)
def test_off_the_pool_route_the_integration_step_is_not_run_and_the_gate_says_so(
    tmp_path: Path, env: dict[str, str], testament: bool, why: str
) -> None:
    """the suite would run here through testcontainers (an hour, or green on nothing without Docker): the gate
    leaves it out, runs the rest, and fails at the end naming why and what to run instead.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :param env: extra environment for the run
    :ptype env: dict[str, str]
    :param testament: whether a ``testament`` is on PATH
    :ptype testament: bool
    :param why: the reason the gate gives
    :ptype why: str
    :return: nothing
    :rtype: None
    """
    run = _check_all(tmp_path, test_sh_exit=0, testament=testament, env=env)
    assert run.returncode == 2
    assert run.ran == [step for step in _STEPS if step != "test-integration.sh"]
    assert f"Integration suite NOT run here: {why}" in run.output
    assert "./scripts/test-integration.sh -rs where the pool is reachable" in run.output
    assert "All checks passed" not in run.output
