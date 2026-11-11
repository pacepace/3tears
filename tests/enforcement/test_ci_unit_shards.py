"""Enforcement: the CI unit shards run the whole suite, and the gate waits for all of them.

``scripts/ci-unit-tests.sh --shard <name>`` runs one piece of the unit + enforcement suite;
``ci.yml`` and ``release.yml`` run the pieces as a ``unit`` matrix. A shard missing from a matrix
is a slice of the suite that silently never runs, with every job green. And the shards only gate
anything through the job that waits for them: ``check`` in ``ci.yml`` (the required status) and
``check`` in ``release.yml`` (which ``build`` needs). This module pins both.
"""

from __future__ import annotations

import re
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


def test_the_release_build_waits_for_check() -> None:
    """``build`` publishes nothing until ``check`` -- and through it every shard -- passed."""
    assert "check" in _needs(_jobs("release.yml")["build"])
