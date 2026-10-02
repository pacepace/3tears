"""the binding that keeps a gate-running CI job on a checkout with the release tags."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.release import untagged_checkout_findings

_WORKFLOW = """name: ci
on: [push]
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
{own}
      - uses: actions/checkout@v7
        with:
          repository: other/sibling
      - run: ./scripts/check-all.sh
  unrelated:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - run: echo hi
"""


def _findings(tmp_path: Path, own: str, runs: str = "check-all.sh") -> list[str]:
    """findings for one workflow whose gate job's own checkout carries *own*.

    :param tmp_path: scratch directory
    :ptype tmp_path: Path
    :param own: the ``with:`` block of the repo's own checkout, indented, or empty
    :ptype own: str
    :param runs: the gate command to look for
    :ptype runs: str
    :return: findings
    :rtype: list[str]
    """
    (tmp_path / "ci.yaml").write_text(_WORKFLOW.format(own=own))
    return untagged_checkout_findings(tmp_path, runs)


def test_a_full_history_checkout_passes(tmp_path: Path) -> None:
    """
    ``fetch-depth: 0`` on the repo's own checkout; the sibling checkout needs nothing.

    :param tmp_path: scratch directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    assert _findings(tmp_path, "        with:\n          fetch-depth: 0") == []


def test_a_default_checkout_is_named(tmp_path: Path) -> None:
    """
    no ``fetch-depth`` is depth 1, and no tags.

    :param tmp_path: scratch directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    findings = _findings(tmp_path, "")
    assert len(findings) == 1
    assert "ci.yaml job 'gate'" in findings[0] and "unset, so 1" in findings[0]


def test_a_command_no_job_runs_is_a_finding(tmp_path: Path) -> None:
    """
    a renamed gate command must not leave the binding matching nothing.

    :param tmp_path: scratch directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    findings = _findings(tmp_path, "        with:\n          fetch-depth: 0", runs="renamed.sh")
    assert len(findings) == 1 and "proves nothing" in findings[0]
