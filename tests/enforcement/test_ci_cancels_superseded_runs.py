"""CI cancels a run that a newer run of the same ref supersedes, and nothing else.

Two merges into develop seconds apart each ran the whole suite (2026-11-12): a fresh, uncached unit run
on the testament pool apiece, though only the newer head's result is ever read. ``ci.yml``'s workflow-level
``concurrency`` cancels the older run of the same event and ref. Its group must name both, so that runs
of different refs, or a PR run and a develop push, never cancel each other: a cancelled required check
on another PR would block it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_CI = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def test_a_newer_run_of_the_same_ref_cancels_the_older() -> None:
    """the workflow's concurrency group is per event and ref, and cancels the run in progress.

    :return: nothing
    :rtype: None
    """
    workflow = yaml.safe_load(_CI.read_text())
    concurrency = workflow["concurrency"]
    assert concurrency["cancel-in-progress"] is True
    group = concurrency["group"]
    assert "${{ github.ref }}" in group and "${{ github.event_name }}" in group, group
