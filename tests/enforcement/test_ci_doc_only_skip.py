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

This test extracts the real ``scope`` step out of the workflow and runs it,
rather than restating its logic -- a copy of the rule here could agree with
itself forever while the workflow drifted. The git-diff line is swapped for a
fixture file, which is the only substitution made.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"

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


def _scope_script() -> str:
    """Return the workflow's ``scope`` step, ready to run against a fixture.

    :return: the step's shell body with the git-diff line reading a fixture
    :rtype: str
    :raises AssertionError: when the step, or the line to swap, has been
        renamed -- which means this test is no longer checking the real rule
    """
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    steps = [step for step in workflow["jobs"]["check"]["steps"] if step.get("id") == "scope"]
    assert steps, "ci.yml has no step with id 'scope'; this test no longer checks the real classifier"
    body: str = steps[0]["run"]

    marker = "changed=$(git diff"
    assert marker in body, f"the scope step no longer contains {marker!r}; update this test with the workflow"
    lines = ['changed=$(cat "$FIXTURE")' if line.startswith(marker) else line for line in body.splitlines()]
    # The event-name guard is a GitHub expression the shell cannot evaluate.
    return "\n".join(lines).replace("${{ github.event_name }}", "pull_request")


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
