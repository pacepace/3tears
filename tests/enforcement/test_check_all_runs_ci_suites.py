"""Enforcement: ``scripts/check-all.sh`` runs every suite CI's ``check`` job runs.

check-all is the local gate before a PR. PR #1087 went red in CI on ``test-mcp-v1.sh`` after every
local gate was green, because check-all did not run it. This reads the ``check`` job of
``.github/workflows/ci.yml``, takes each ``./scripts/<name>.sh`` its steps run, and requires
check-all to run it too. A suite step added to that job turns this red until check-all runs it, or
until it is added to the CI-only list below with its reason.

CI-only plumbing, excluded on purpose (``_CI_ONLY``):

- ``check-attribution.sh``: reads the commits a pull request adds (``base..head``), which only a
  PR has; locally it runs as the commit-msg hook ``scripts/install-hooks.sh`` installs.
- ``ci-scope.sh``: decides whether a PR needs lint, typecheck and the suites at all; check-all
  always runs them.

The ``check`` job's Lint step runs ruff inline rather than through a script; check-all runs
``lint.sh``, which runs the same two ruff commands, and the last test pins that.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_CHECK_ALL = _REPO_ROOT / "scripts" / "check-all.sh"

#: scripts the ``check`` job runs that are CI plumbing rather than a suite; see the module docstring
_CI_ONLY = frozenset({"check-attribution.sh", "ci-scope.sh"})


def _check_job_runs() -> list[str]:
    """Return the ``run`` text of each step of ci.yml's ``check`` job.

    :return: each step's shell text, for the steps that run one
    :rtype: list[str]
    """
    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["check"]["steps"]
    return [str(step["run"]) for step in steps if "run" in step]


def _ci_scripts() -> set[str]:
    """Return the ``./scripts/*.sh`` names the ``check`` job runs.

    :return: script file names
    :rtype: set[str]
    """
    return {name for run in _check_job_runs() for name in re.findall(r"\./scripts/([\w.-]+\.sh)", run)}


def _check_all_scripts() -> set[str]:
    """Return the ``scripts/*.sh`` names check-all runs, from its non-comment lines.

    :return: script file names
    :rtype: set[str]
    """
    lines = [line for line in _CHECK_ALL.read_text().splitlines() if not line.lstrip().startswith("#")]
    return {name for line in lines for name in re.findall(r"scripts/([\w.-]+\.sh)", line)}


def test_check_all_runs_every_suite_the_check_job_runs() -> None:
    """every script step of ``check``, less the CI-only plumbing, is in check-all."""
    suites = _ci_scripts() - _CI_ONLY
    assert suites, "found no ./scripts/*.sh step in ci.yml's check job; update this test with the workflow"
    missing = sorted(suites - _check_all_scripts())
    assert not missing, (
        f"CI's check job runs {missing} and scripts/check-all.sh does not: run it there, or, if it is CI "
        "plumbing rather than a suite, add it to _CI_ONLY with its reason in this module's docstring"
    )


def test_every_ci_only_exclusion_is_still_in_the_check_job() -> None:
    """an exclusion for a step CI no longer runs is stale; drop it."""
    stale = sorted(_CI_ONLY - _ci_scripts())
    assert not stale, f"_CI_ONLY names {stale}, which ci.yml's check job no longer runs"


def test_check_all_lints_as_the_check_job_does() -> None:
    """the job's inline ruff step is covered by check-all running ``lint.sh``."""
    assert any("ruff check" in run for run in _check_job_runs()), "ci.yml's check job no longer lints with ruff"
    assert "lint.sh" in _check_all_scripts()
