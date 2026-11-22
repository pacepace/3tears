"""The GitHub Actions workflows stay hardened against a hostile pull request and a hijacked action.

Each rule here closes one route by which code nobody reviewed runs with this repository's token or
secrets, or ends up inside a published artifact:

- no workflow triggers on ``pull_request_target``, ``issue_comment`` or ``workflow_run``. Those run with a
  write token and the repository's secrets on an event an outsider can cause, and GitHub's own guidance
  is to drop them rather than guard them;
- every workflow declares its token permissions at the top and grants nothing writable there, so a
  change to the repository's default cannot widen them; a job that writes asks for it itself;
- every action is pinned to a full commit SHA. A tag or a branch can be moved to new code by whoever
  controls the action's repository, and the release jobs hold the publish credentials;
- every checkout sets ``persist-credentials: false``, so the token is not left in ``.git/config`` for
  any later step, or an uploaded artifact, to read;
- the release workflow restores no cache into anything it publishes. The one cache it keeps holds the
  test browser, which runs the render tests and never reaches an artifact;
- Dependabot keeps the pins current, so a pin does not become a way of staying on a vulnerable release.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_GITHUB = Path(__file__).resolve().parents[2] / ".github"
_WORKFLOWS = sorted([*(_GITHUB / "workflows").glob("*.yml"), *(_GITHUB / "workflows").glob("*.yaml")])
_DANGEROUS_TRIGGERS = frozenset({"pull_request_target", "issue_comment", "workflow_run"})
_PINNED = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
#: the release workflow's one cache: the test browser, never part of a published artifact
_RELEASE_CACHE_PATHS = frozenset({"~/.cache/ms-playwright"})


def _load(path: Path) -> dict[Any, Any]:
    """Parse one workflow file.

    :param path: the workflow file
    :ptype path: Path
    :return: the parsed workflow
    :rtype: dict[Any, Any]
    """
    workflow = yaml.safe_load(path.read_text())
    assert isinstance(workflow, dict), path.name
    return workflow


def _steps(workflow: dict[Any, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every step of every job, with the job's name.

    :param workflow: a parsed workflow
    :ptype workflow: dict[Any, Any]
    :return: (job name, step) pairs
    :rtype: list[tuple[str, dict[str, Any]]]
    """
    pairs: list[tuple[str, dict[str, Any]]] = []
    for name, job in workflow["jobs"].items():
        pairs.extend((name, step) for step in job.get("steps") or [] if isinstance(step, dict))
    return pairs


def _uses(step: dict[str, Any], action: str) -> bool:
    """Whether a step runs ``action`` at any ref.

    :param step: one workflow step
    :ptype step: dict[str, Any]
    :param action: ``owner/repo``
    :ptype action: str
    :return: whether the step's ``uses`` names that action
    :rtype: bool
    """
    return str(step.get("uses", "")).startswith(f"{action}@")


def test_there_are_workflows_to_check() -> None:
    """the glob reaches the workflows, so every parametrized check below has something to read.

    :return: nothing
    :rtype: None
    """
    assert {path.name for path in _WORKFLOWS} >= {"ci.yml", "release.yml"}


@pytest.mark.parametrize("path", _WORKFLOWS, ids=lambda path: path.name)
def test_no_workflow_triggers_on_an_event_an_outsider_controls_with_secrets(path: Path) -> None:
    """``pull_request_target``, ``issue_comment`` and ``workflow_run`` run with a write token and secrets.

    :param path: the workflow file
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    workflow = _load(path)
    # YAML 1.1 reads the bare key `on` as the boolean true
    triggers = workflow.get("on", workflow.get(True))
    names = {triggers} if isinstance(triggers, str) else set(triggers or ())
    assert not names & _DANGEROUS_TRIGGERS, f"{path.name} triggers on {sorted(names & _DANGEROUS_TRIGGERS)}"


@pytest.mark.parametrize("path", _WORKFLOWS, ids=lambda path: path.name)
def test_every_workflow_declares_a_token_with_nothing_writable(path: Path) -> None:
    """the top-level ``permissions`` is stated, and writes nothing; a job that writes asks for it itself.

    :param path: the workflow file
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    permissions = _load(path).get("permissions")
    assert isinstance(permissions, dict) and permissions, (
        f"{path.name} inherits the repository's default token permissions; declare them at the top"
    )
    writable = sorted(scope for scope, level in permissions.items() if level == "write")
    assert not writable, f"{path.name} grants {writable} to every job; grant it on the job that needs it"


@pytest.mark.parametrize("path", _WORKFLOWS, ids=lambda path: path.name)
def test_every_action_is_pinned_to_a_commit(path: Path) -> None:
    """a tag or a branch can be moved to new code; a full commit SHA cannot.

    :param path: the workflow file
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    workflow = _load(path)
    refs = [str(step["uses"]) for _, step in _steps(workflow) if "uses" in step]
    refs += [str(job["uses"]) for job in workflow["jobs"].values() if "uses" in job]
    unpinned = [ref for ref in refs if not ref.startswith(("./", "docker://")) and not _PINNED.match(ref)]
    assert not unpinned, f"{path.name} runs actions by a movable ref: {unpinned}"


@pytest.mark.parametrize("path", _WORKFLOWS, ids=lambda path: path.name)
def test_every_checkout_leaves_no_credentials_behind(path: Path) -> None:
    """``actions/checkout`` writes the token into ``.git/config`` unless told not to.

    :param path: the workflow file
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    checkouts = [(job, step) for job, step in _steps(_load(path)) if _uses(step, "actions/checkout")]
    persisting = [job for job, step in checkouts if (step.get("with") or {}).get("persist-credentials") is not False]
    assert not persisting, f"{path.name}: checkouts in {persisting} do not set persist-credentials: false"


def test_the_release_restores_no_cache_into_what_it_publishes() -> None:
    """uv and npm run uncached in the release workflow; its only cache is the test browser.

    :return: nothing
    :rtype: None
    """
    steps = _steps(_load(_GITHUB / "workflows" / "release.yml"))
    toggles = {"astral-sh/setup-uv": "enable-cache", "actions/setup-node": "package-manager-cache"}
    for action, toggle in toggles.items():
        cached = [
            job for job, step in steps if _uses(step, action) and (step.get("with") or {}).get(toggle) is not False
        ]
        assert not cached, f"release.yml: {action} in {cached} must set {toggle}: false"
    caches = [str((step.get("with") or {}).get("path")) for _, step in steps if _uses(step, "actions/cache")]
    assert set(caches) <= _RELEASE_CACHE_PATHS, f"release.yml restores a cache beyond the test browser: {caches}"


def test_dependabot_keeps_the_action_pins_current() -> None:
    """a pinned action is updated by Dependabot, or the pin holds a vulnerable release indefinitely.

    :return: nothing
    :rtype: None
    """
    config = yaml.safe_load((_GITHUB / "dependabot.yml").read_text())
    ecosystems = {update["package-ecosystem"] for update in config["updates"]}
    assert "github-actions" in ecosystems, "dependabot.yml does not watch github-actions"
