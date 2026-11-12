"""Enforcement: the CI unit shards run the whole suite, and the gate waits for all of them.

``scripts/ci-unit-tests.sh --shard <name>`` runs one piece of the unit + enforcement suite;
``ci.yml`` and ``release.yml`` run the pieces as a ``unit`` matrix. A shard missing from a matrix
is a slice of the suite that silently never runs, with every job green. And the shards only gate
anything through the job that waits for them: ``check`` in ``ci.yml`` (the required status) and
``check`` in ``release.yml`` (which ``build`` needs). This module pins both.

``ci.yml`` has a second route: ``unit-pool`` runs the same suite on the testament pool when
``route`` chose it (``scripts/ci-unit-pool.sh route``: the key, a same-repo event, ``TESTAMENT_CI``
not off, and a pool that answers its probe). The shards run when route chose them, or when the pool
was lost during ``unit-pool``'s run (its outcome ``unreachable``), never after a test failed there.
``check`` is green only when the route that ran to a verdict passed; this module pins the wiring and
runs ``check``'s verdict step against every combination. tests/test_ci_unit_pool_script.py pins the
script's own decisions.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"
_SCRIPT = _REPO_ROOT / "scripts" / "ci-unit-tests.sh"


def _script_shards() -> list[str]:
    """Return the shard names the script accepts.

    :return: the names in its ``SHARDS`` list
    :rtype: list[str]
    """
    match = re.search(r'^SHARDS="([^"]+)"$', _SCRIPT.read_text(), re.MULTILINE)
    assert match is not None, "scripts/ci-unit-tests.sh no longer declares SHARDS; update this test with it"
    return match[1].split()


def _jobs(workflow: str) -> dict[str, dict[str, object]]:
    """Return a workflow's jobs.

    :param workflow: the file name under .github/workflows
    :ptype workflow: str
    :return: job name to job mapping
    :rtype: dict[str, dict[str, object]]
    """
    jobs: dict[str, dict[str, object]] = yaml.safe_load((_WORKFLOWS / workflow).read_text())["jobs"]
    return jobs


def _needs(job: dict[str, object]) -> list[str]:
    """Return a job's ``needs`` as a list.

    :param job: the job mapping
    :ptype job: dict[str, object]
    :return: the job names it needs
    :rtype: list[str]
    """
    needs = job.get("needs") or []
    if isinstance(needs, str):
        return [needs]
    assert isinstance(needs, list)
    return [str(name) for name in needs]


def test_the_shards_partition_the_suite() -> None:
    """the eleven shards are the eight evals groups, core, tests and rest -- what the no-selector run runs."""
    evals = [f"evals-{group}" for group in range(1, 9)]
    assert _script_shards() == [*evals, "core", "tests", "rest"]


def test_the_evals_shards_are_every_group_of_the_split() -> None:
    """pytest-split's groups partition packages/evals only when every group 1..N runs, N being ``--splits``.

    a shard list one short of ``--splits`` leaves a group, a slice of the evals tests, that no job runs.
    """
    script = _SCRIPT.read_text()
    splits = re.findall(r"--splits (\d+) --group \$\{shard#evals-\}", script)
    assert len(splits) == 1, "scripts/ci-unit-tests.sh no longer splits evals as --splits N --group <shard>"
    groups = [name.removeprefix("evals-") for name in _script_shards() if name.startswith("evals-")]
    assert groups == [str(group) for group in range(1, int(splits[0]) + 1)]
    assert "--durations-path .test_durations" in script
    assert (_REPO_ROOT / ".test_durations").is_file(), "the evals split balances by .test_durations"


@pytest.mark.parametrize("workflow", ["ci.yml", "release.yml"])
def test_the_unit_matrix_runs_every_shard(workflow: str) -> None:
    """every shard the script knows is in the matrix, and the job runs the selected one.

    :param workflow: the file name under .github/workflows
    :ptype workflow: str
    :return: nothing
    :rtype: None
    """
    unit = _jobs(workflow)["unit"]
    strategy = unit["strategy"]
    assert isinstance(strategy, dict)
    assert strategy.get("fail-fast") is False, f"{workflow}: a failing shard must not cancel the others"
    assert sorted(strategy["matrix"]["shard"]) == sorted(_script_shards()), (
        f"{workflow}'s unit matrix does not run every shard of scripts/ci-unit-tests.sh"
    )
    steps = unit["steps"]
    assert isinstance(steps, list)
    runs = [str(step.get("run", "")) for step in steps]
    assert any("./scripts/ci-unit-tests.sh --shard ${{ matrix.shard }}" in run for run in runs), (
        f"{workflow}'s unit job does not run ./scripts/ci-unit-tests.sh --shard ${{{{ matrix.shard }}}}"
    )


@pytest.mark.parametrize("workflow", ["ci.yml", "release.yml"])
def test_the_evals_shards_install_chromium_and_require_it(workflow: str) -> None:
    """the evals render tests (the ``browser`` marker) run in the evals shards: each installs headless Chromium
    before its test step, and that step sets THREETEARS_REQUIRE_CHROMIUM, so a failed install fails the tests rather
    than skipping them. tests/enforcement/test_browser_marker.py keeps every browser test in packages/evals.

    :param workflow: the file name under .github/workflows
    :ptype workflow: str
    :return: nothing
    :rtype: None
    """
    steps = _jobs(workflow)["unit"]["steps"]
    assert isinstance(steps, list)
    runs = [str(step.get("run", "")) for step in steps]
    install = [i for i, run in enumerate(runs) if "playwright install --with-deps --only-shell chromium" in run]
    test = [i for i, run in enumerate(runs) if "./scripts/ci-unit-tests.sh --shard" in run]
    assert install and test and install[0] < test[0], (
        f"{workflow}'s unit job does not install Chromium before its tests"
    )
    assert "startsWith(matrix.shard, 'evals-')" in str(steps[install[0]].get("if", "")), (
        f"{workflow}: the Chromium install should run on the evals shards, where the browser tests are"
    )
    env = steps[test[0]].get("env") or {}
    assert env.get("THREETEARS_REQUIRE_CHROMIUM") == "${{ startsWith(matrix.shard, 'evals-') && '1' || '' }}", (
        f"{workflow}: the evals shards' test step must require Chromium"
    )
    fonts = [i for i, run in enumerate(runs) if run.strip() == "./scripts/check-fallback-fonts.sh"]
    assert fonts and install[0] < fonts[0] < test[0], (
        f"{workflow}'s unit job must check the fallback fonts are DejaVu after installing them and before its tests"
    )
    assert "startsWith(matrix.shard, 'evals-')" in str(steps[fonts[0]].get("if", "")), workflow
    # the cache key's version: assigned before it is written, so a failed read fails the step rather than caching
    # under an empty version
    version = [step for step in steps if step.get("id") == "playwright"]
    assert len(version) == 1, f"{workflow}: one step reads the locked playwright version"
    read = str(version[0].get("run", ""))
    assert re.search(r'^version="\$\(uv run python .*\)"$', read, re.M) and 'test -n "$version"' in read, read
    assert read.index('test -n "$version"') < read.index('echo "version=${version}"'), read


@pytest.mark.parametrize("workflow", ["ci.yml", "release.yml"])
def test_check_waits_for_the_shards_and_fails_unless_all_passed(workflow: str) -> None:
    """``check`` needs ``unit``, always reports, and its first step refuses anything but success.

    :param workflow: the file name under .github/workflows
    :ptype workflow: str
    :return: nothing
    :rtype: None
    """
    check = _jobs(workflow)["check"]
    assert "unit" in _needs(check), f"{workflow}: check must need the unit shards"
    assert check.get("if") == "always()", f"{workflow}: check must report even when a shard failed"
    steps = check["steps"]
    assert isinstance(steps, list)
    first = steps[0]
    assert "needs.unit.result" in str(first.get("env", "")), f"{workflow}: check's first step must read the shards"
    assert '!= "success"' in str(first.get("run", "")) and "exit 1" in str(first.get("run", ""))


#: the key reaches a job only for a push or a pull request from a branch of this repo
_GATED_KEY = (
    "${{ (github.event_name == 'push' || github.event.pull_request.head.repo.full_name == github.repository)"
    " && secrets.TESTAMENT_KEY || '' }}"
)


def _step(job: str, script: str) -> dict[str, object]:
    """Return the one step of a ci.yml job that runs ``script``.

    :param job: the job name in ci.yml
    :ptype job: str
    :param script: the command the step's ``run`` must be
    :ptype script: str
    :return: the step mapping
    :rtype: dict[str, object]
    """
    steps = _jobs("ci.yml")[job]["steps"]
    assert isinstance(steps, list)
    found = [step for step in steps if str(step.get("run", "")).strip() == script]
    assert len(found) == 1, f"ci.yml's {job} job no longer runs {script} in one step"
    step: dict[str, object] = found[0]
    return step


def test_route_chooses_with_the_script_and_a_gated_key() -> None:
    """``route`` runs ``ci-unit-pool.sh route``, which probes the pool, and only same-repo events get the key."""
    route = _jobs("ci.yml")["route"]
    step = _step("route", "./scripts/ci-unit-pool.sh route")
    env = step["env"]
    assert isinstance(env, dict)
    assert env["TESTAMENT_KEY"] == _GATED_KEY, "route must withhold the key from anything but this repo's events"
    assert env["TESTAMENT_CI"] == "${{ vars.TESTAMENT_CI }}", "TESTAMENT_CI=off must reach route"
    assert env["TESTAMENT_COORDINATOR"] == "${{ vars.TESTAMENT_COORDINATOR }}"
    assert env["EVENT_NAME"] == "${{ github.event_name }}"
    assert env["HEAD_REPO"] == "${{ github.event.pull_request.head.repo.full_name }}"
    assert env["REPO"] == "${{ github.repository }}"
    outputs = route["outputs"]
    assert isinstance(outputs, dict)
    assert outputs == {
        "pool": f"${{{{ steps.{step['id']}.outputs.pool }}}}",
        "why": f"${{{{ steps.{step['id']}.outputs.why }}}}",
    }


def test_the_pool_runs_when_route_chose_it() -> None:
    """``unit-pool`` runs the pool script's ``run`` as the gated key, and reports its outcome."""
    jobs = _jobs("ci.yml")
    pool = jobs["unit-pool"]
    assert pool.get("if") == "needs.route.outputs.pool == 'true'"
    assert _needs(pool) == ["route"]
    step = _step("unit-pool", "./scripts/ci-unit-pool.sh run")
    env = step["env"]
    assert isinstance(env, dict)
    assert env["TESTAMENT_KEY"] == _GATED_KEY
    assert env["TESTAMENT_COORDINATOR"] == "${{ vars.TESTAMENT_COORDINATOR }}"
    assert env["TESTAMENT_BRANCH"] == "${{ github.head_ref || github.ref_name }}"
    outputs = pool["outputs"]
    assert isinstance(outputs, dict)
    assert outputs["outcome"] == f"${{{{ steps.{step['id']}.outputs.outcome }}}}"


#: the serial loop_timing session, as ci-unit-tests.sh runs it: every loop_timing test, one at a time
_LOOP_TIMING_SESSION = "pytest packages tests -m loop_timing and not integration -n 0"


def test_the_pool_job_runs_loop_timing_serially_unless_the_pool_was_lost() -> None:
    """the pool run leaves the loop_timing tests out, so ``unit-pool`` runs them itself, on its own runner.

    After a pool verdict, passed or failed (one failure must not hide the other), and never after
    ``unreachable``: the shards then run the whole suite, these tests in ``rest``, and running them
    here too would run them twice. Every step the serial run needs is gated the same way.
    """
    steps = _jobs("ci.yml")["unit-pool"]["steps"]
    assert isinstance(steps, list)
    pool = _step("unit-pool", "./scripts/ci-unit-pool.sh run")
    after = steps[steps.index(pool) + 1 :]
    runs = [str(step.get("run", "")).strip() for step in after]
    assert "./scripts/ci-unit-tests.sh --loop-timing -v" in runs, "unit-pool no longer runs loop_timing after the pool"
    gate = (
        f"!cancelled() && steps.scope.outputs.needed == 'true' && steps.{pool['id']}.outputs.outcome != 'unreachable'"
    )
    serial = after[: runs.index("./scripts/ci-unit-tests.sh --loop-timing -v") + 1]
    assert [step.get("run") for step in serial if step.get("if") != gate] == [], "a loop_timing step is gated otherwise"


@pytest.mark.skipif(shutil.which("bash") is None, reason="ci-unit-tests.sh is bash; none on this host")
@pytest.mark.parametrize(
    ("args", "sessions"),
    [
        pytest.param(["--loop-timing", "-v"], 0, id="loop-timing-alone"),
        pytest.param(["--shard", "rest", "-v"], 2, id="rest-shard"),
        pytest.param(["--shard", "core", "-v"], 1, id="core-shard"),
    ],
)
def test_loop_timing_runs_once_serially_and_only_where_it_should(
    args: list[str], sessions: int, tmp_path: Path
) -> None:
    """``--loop-timing`` runs the serial session and nothing else; ``rest`` runs it after its parallel sessions.

    Run against a stand-in ``uv`` that records each pytest call.

    :param args: the script's arguments
    :ptype args: list[str]
    :param sessions: how many parallel (``not loop_timing``) sessions run first
    :ptype sessions: int
    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    calls = tmp_path / "uv.calls"
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{calls}"\n')
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(_SCRIPT), *args],
        env={"PATH": f"{uv.parent}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ran = calls.read_text().splitlines()
    parallel = [call for call in ran if "-m not integration and not loop_timing" in call]
    serial = [call for call in ran if _LOOP_TIMING_SESSION in call]
    assert len(parallel) == sessions, ran
    assert len(serial) == (0 if args[:2] == ["--shard", "core"] else 1), ran
    assert len(ran) == len(parallel) + len(serial), ran


def test_the_shards_run_when_route_chose_them_or_the_pool_was_lost() -> None:
    """the shards wait for ``unit-pool`` and run for route's ``false`` or the pool's ``unreachable``, nothing else.

    A pool outcome of ``failed`` (a test failed) never runs them: the pool's verdict stands.
    """
    unit = _jobs("ci.yml")["unit"]
    assert set(_needs(unit)) == {"route", "unit-pool"}
    assert unit.get("if") == (
        "!cancelled() && needs.route.result == 'success' && "
        "(needs.route.outputs.pool == 'false' || needs.unit-pool.outputs.outcome == 'unreachable')"
    )


def _verdict_step() -> tuple[dict[str, str], str]:
    """Return ci.yml's ``check`` verdict step: its env and its shell text.

    :return: the env mapping and the run text
    :rtype: tuple[dict[str, str], str]
    """
    check = _jobs("ci.yml")["check"]
    assert {"route", "unit", "unit-pool"} <= set(_needs(check))
    steps = check["steps"]
    assert isinstance(steps, list)
    env, run = steps[0].get("env"), steps[0].get("run")
    assert isinstance(env, dict) and isinstance(run, str)
    return {str(k): str(v) for k, v in env.items()}, run


def test_check_reads_both_routes() -> None:
    """``check``'s verdict step reads route's choice, the pool's result and outcome, and the shards' result."""
    env, _ = _verdict_step()
    assert env == {
        "ROUTE": "${{ needs.route.outputs.pool }}",
        "ROUTE_WHY": "${{ needs.route.outputs.why }}",
        "UNIT_RESULT": "${{ needs.unit.result }}",
        "POOL_RESULT": "${{ needs.unit-pool.result }}",
        "POOL_OUTCOME": "${{ needs.unit-pool.outputs.outcome }}",
        "POOL_WHY": "${{ needs.unit-pool.outputs.why }}",
    }


#: (route's pool output, unit-pool's outcome, unit-pool's result, the shards' result) -> check green?
_VERDICTS = [
    pytest.param("true", "passed", "success", "skipped", True, "unit (testament pool)", id="pool-passed"),
    pytest.param("true", "", "success", "skipped", True, "unit (testament pool)", id="pool-doc-only"),
    pytest.param("true", "failed", "failure", "skipped", False, "unit (testament pool)", id="pool-test-failed"),
    pytest.param(
        "true", "failed", "failure", "success", False, "unit (testament pool)", id="pool-failed-shards-ignored"
    ),
    pytest.param("true", "", "failure", "skipped", False, "unit (testament pool)", id="pool-job-broke"),
    pytest.param("true", "", "cancelled", "skipped", False, "unit (testament pool)", id="pool-cancelled"),
    pytest.param("true", "unreachable", "success", "success", True, "the unit shards", id="pool-lost-shards-passed"),
    pytest.param("true", "unreachable", "success", "failure", False, "the unit shards", id="pool-lost-shards-failed"),
    pytest.param(
        "true", "unreachable", "success", "skipped", False, "the unit shards", id="pool-lost-shards-never-ran"
    ),
    pytest.param(
        "true", "unreachable", "success", "cancelled", False, "the unit shards", id="pool-lost-shards-cancelled"
    ),
    pytest.param("false", "", "skipped", "success", True, "the unit shards", id="shards-passed"),
    pytest.param("false", "", "skipped", "failure", False, "the unit shards", id="shards-failed"),
    pytest.param("false", "", "skipped", "skipped", False, "the unit shards", id="shards-skipped"),
    pytest.param("", "", "skipped", "skipped", False, "no unit route", id="route-decided-nothing"),
    pytest.param("", "", "skipped", "success", False, "no unit route", id="route-broke-shards-ran"),
]


@pytest.mark.skipif(shutil.which("bash") is None, reason="the workflow step is bash; none on this host")
@pytest.mark.parametrize(("route", "outcome", "pool", "shards", "green", "ran"), _VERDICTS)
def test_check_is_green_iff_the_route_that_ran_to_a_verdict_passed(
    route: str, outcome: str, pool: str, shards: str, green: bool, ran: str, tmp_path: Path
) -> None:
    """runs ``check``'s own verdict step: the pool's verdict, or the shards' when they ran in its place.

    :param route: route's pool output
    :ptype route: str
    :param outcome: unit-pool's outcome output
    :ptype outcome: str
    :param pool: unit-pool's job result
    :ptype pool: str
    :param shards: the unit matrix's job result
    :ptype shards: str
    :param green: whether check must pass
    :ptype green: bool
    :param ran: the route the step must name
    :ptype ran: str
    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    _, run = _verdict_step()
    summary = tmp_path / "summary"
    summary.write_text("")
    result = subprocess.run(
        ["bash", "-e", "-c", run],
        env={
            "PATH": "/usr/bin:/bin",
            "GITHUB_STEP_SUMMARY": str(summary),
            "ROUTE": route,
            "ROUTE_WHY": "the pool answered the probe" if route == "true" else "no TESTAMENT_KEY secret",
            "UNIT_RESULT": shards,
            "POOL_RESULT": pool,
            "POOL_OUTCOME": outcome,
            "POOL_WHY": "connection_lost: gone" if outcome == "unreachable" else "",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert (result.returncode == 0) is green, result.stdout + result.stderr
    if ran == "no unit route":
        assert "::error::the route job chose no unit route" in result.stdout
        return
    # the route that ran, and why, as an annotation and in the job summary: a fallback is never silent
    assert f"title=unit route::{ran}: " in result.stdout
    assert f"### unit route: {ran}" in summary.read_text()
    if outcome == "unreachable":
        assert "::warning title=unit route::the unit shards: pool unreachable during the run" in result.stdout


def test_the_release_build_waits_for_check() -> None:
    """``build`` publishes nothing until ``check`` -- and through it every shard -- passed."""
    assert "check" in _needs(_jobs("release.yml")["build"])
