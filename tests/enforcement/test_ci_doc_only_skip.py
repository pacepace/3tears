"""Enforcement: CI's doc-only skip runs the suite whenever code changed.

``.github/workflows/ci.yml`` skips lint, typecheck and tests when a pull
request touched nothing but documentation. The saving is real and the failure
mode is silent: a classifier that answers "documentation only" for a changeset
containing a ``.py`` file reports a green ``check`` having verified nothing,
and ``check`` is a REQUIRED status context on both develop and main. Nobody
reads a green tick to find out what it skipped.

The case that actually bites is the MIXED changeset -- a PR that edits a
README *and* a module. It is the one a hand-written negation gets wrong, and
it is the reason the workflow classifies with shell ``case`` rather than
``grep -qv``: "some line did not match" and "no line matched" read alike and
differ exactly here.

The classifier lives in ``scripts/ci-scope.sh``, the one file both the
``check`` job, every ``unit`` shard and the ``unit-pool`` job run as their
``scope`` step: a unit job that ran a different rule could skip a suite
``check`` believes ran. This test pins that those jobs' ``scope`` steps run that script, then runs the script
itself rather than restating its logic -- a copy of the rule here could agree
with itself forever while the script drifted. The git-diff line is swapped for
a fixture file, which is the only substitution made.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_SCOPE_SCRIPT = "scripts/ci-scope.sh"

#: the jobs whose suite the doc-only skip governs; each must classify with the shared script.
_SCOPED_JOBS = ("check", "unit", "unit-pool")

#: changesets and whether the suite MUST run for them. The mixed entries are
#: the point of the file; the pure-docs ones stop the skip from becoming a
#: no-op that quietly runs everything anyway.
_CASES = [
    pytest.param(["README.md"], False, id="root-readme"),
    pytest.param(["packages/scrape/README.md"], False, id="package-readme"),
    pytest.param(["docs/search-spec.md", "CHANGELOG.md"], False, id="docs-and-changelog"),
    pytest.param(["docs/notes.txt"], False, id="non-md-under-docs"),
    pytest.param(["LICENSE", "packages/search/LICENSE"], False, id="licences"),
    pytest.param(["packages/search/src/threetears/search/call.py"], True, id="module-only"),
    pytest.param(["README.md", "packages/search/src/threetears/search/call.py"], True, id="MIXED-docs-and-module"),
    pytest.param(["docs/search-spec.md", "uv.lock"], True, id="MIXED-docs-and-lockfile"),
    pytest.param(["pyproject.toml"], True, id="root-pyproject"),
    pytest.param([".github/workflows/ci.yml"], True, id="the-workflow-itself"),
    pytest.param(["tests/enforcement/_fake_parity_exemptions.txt"], True, id="enforcement-exemptions"),
    pytest.param(["packages/core/tests/test_smoke.py"], True, id="a-test"),
    pytest.param([], True, id="empty-changeset-defaults-to-running"),
]


def _scope_step(job: str) -> dict[str, object]:
    """Return *job*'s ``scope`` step from the workflow.

    :param job: the job name in ci.yml
    :ptype job: str
    :return: the step mapping
    :rtype: dict[str, object]
    :raises AssertionError: when the job or its ``scope`` step has gone
    """
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    assert job in workflow["jobs"], f"ci.yml has no job {job!r}; this test no longer checks the real classifier"
    steps = [step for step in workflow["jobs"][job]["steps"] if step.get("id") == "scope"]
    assert steps, f"ci.yml job {job!r} has no step with id 'scope'; this test no longer checks the real classifier"
    step: dict[str, object] = steps[0]
    return step


@pytest.mark.parametrize("job", _SCOPED_JOBS)
def test_every_scoped_job_classifies_with_the_shared_script(job: str) -> None:
    """``check``, every unit shard and the pool job run the one classifier, fed the event and the PR base.

    :param job: the job name in ci.yml
    :ptype job: str
    :return: nothing
    :rtype: None
    """
    step = _scope_step(job)
    assert str(step.get("run", "")).strip() == f"./{_SCOPE_SCRIPT}", (
        f"ci.yml job {job!r}'s scope step must run ./{_SCOPE_SCRIPT}, the classifier this test exercises"
    )
    env = step.get("env")
    assert isinstance(env, dict)
    assert env.get("EVENT_NAME") == "${{ github.event_name }}"
    assert env.get("BASE_SHA") == "${{ github.event.pull_request.base.sha }}"


def _scope_script() -> str:
    """Return the shared ``scope`` classifier, ready to run against a fixture.

    :return: the script's shell body with the git-diff line reading a fixture
    :rtype: str
    :raises AssertionError: when the line to swap has been renamed -- which
        means this test is no longer checking the real rule
    """
    body = (_REPO_ROOT / _SCOPE_SCRIPT).read_text()
    marker = "changed=$(git diff"
    assert marker in body, f"{_SCOPE_SCRIPT} no longer contains {marker!r}; update this test with the script"
    lines = ['changed=$(cat "$FIXTURE")' if line.startswith(marker) else line for line in body.splitlines()]
    return "\n".join(lines)


@pytest.mark.skipif(shutil.which("bash") is None, reason="the workflow step is bash; none on this host")
@pytest.mark.parametrize(("changed", "suite_required"), _CASES)
def test_the_ci_scope_step_classifies_a_changeset(changed: list[str], suite_required: bool, tmp_path: Path) -> None:
    """The real workflow step demands the suite for anything that is not a doc.

    :param changed: paths the fixture reports as changed
    :ptype changed: list[str]
    :param suite_required: whether lint/typecheck/tests must run
    :ptype suite_required: bool
    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    fixture = tmp_path / "changed.txt"
    fixture.write_text("\n".join(changed))
    output = tmp_path / "github_output"
    output.write_text("")

    result = subprocess.run(
        ["bash", "-c", _scope_script()],
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "FIXTURE": str(fixture),
            "GITHUB_OUTPUT": str(output),
            "EVENT_NAME": "pull_request",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"the scope step failed: {result.stderr}"

    written = [line for line in output.read_text().splitlines() if line.startswith("needed=")]
    assert written, f"the step wrote no 'needed=' output; it emitted: {result.stdout}"
    assert written[-1] == f"needed={str(suite_required).lower()}", (
        f"changeset {changed} classified as {written[-1]}, expected needed={str(suite_required).lower()}"
    )
