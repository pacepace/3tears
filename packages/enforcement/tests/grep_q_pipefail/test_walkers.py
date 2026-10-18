"""the quiet-grep-under-pipefail walker: what it refuses, what it lets through, and that it refuses a scan of nothing."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.grep_q_pipefail import GrepQPipefailConfig, find_grep_q_pipefail_violations
from threetears.enforcement.grep_q_pipefail.walkers import logical_lines, quiet_grep_pipes

PIPEFAIL = "#!/usr/bin/env bash\nset -euo pipefail\n"


def found(tmp_path: Path, name: str, text: str) -> list[int]:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return [v.line for v in find_grep_q_pipefail_violations(GrepQPipefailConfig(repo_root=tmp_path))]


@pytest.mark.parametrize(
    "line",
    [
        "xprop -root X | grep -q 'window id'",
        "cmd |& grep -q x",
        "cmd | egrep -q x",
        "cmd | fgrep -q x",
        "cmd | grep -iq x",
        "cmd | grep -qi x",
        "cmd | grep -E -q x",
        "cmd | grep --quiet x",
        "cmd | grep --silent x",
        "cmd | /usr/bin/grep -q x",
        "if cmd | grep -q x; then :; fi",
    ],
)
def test_a_quiet_grep_at_the_end_of_a_pipe_is_refused(line: str) -> None:
    assert quiet_grep_pipes(line)


@pytest.mark.parametrize(
    "line",
    [
        "cmd || grep -q x file",  # || is not a pipe
        "grep -q x file",  # no pipe at all
        "cmd | grep x",  # not quiet
        "cmd | grep -c x",
        'grep -q x <<< "$out"',  # the fix
    ],
)
def test_what_is_not_a_quiet_grep_pipe_passes(line: str) -> None:
    assert not quiet_grep_pipes(line)


def test_a_pipeline_split_across_lines_is_seen_whole() -> None:
    joined = list(logical_lines("xprop -root X \\\n  | grep -q 'window id'\nother |\n  grep -q y\n# cmd | grep -q z\n"))
    assert [n for n, line in joined if quiet_grep_pipes(line)] == [1, 3]


def test_a_script_without_pipefail_is_not_refused(tmp_path: Path) -> None:
    assert found(tmp_path, "a.sh", "#!/bin/sh\nset -eu\ncmd | grep -q x\n") == []
    assert found(tmp_path, "b.sh", f"{PIPEFAIL}cmd | grep -q x\n") == [3]


def test_a_workflow_is_always_under_pipefail(tmp_path: Path) -> None:
    flow = "jobs:\n  a:\n    steps:\n      - run: |\n          curl -s x | grep -q ok\n"
    assert found(tmp_path, ".github/workflows/ci.yml", flow) == [5]


def test_a_scan_that_finds_no_files_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="found no files"):
        find_grep_q_pipefail_violations(GrepQPipefailConfig(repo_root=tmp_path))
