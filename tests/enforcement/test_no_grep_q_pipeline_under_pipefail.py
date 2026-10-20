"""no pipe into a quiet grep under pipefail in this repo's scripts and workflows (threetears.enforcement.grep_q_pipefail).

The scrape sidecar's entrypoint refused to boot with a running window manager this way (2026-10-18). The same gate
runs in the hub (#643), reports (#60) and geography (#14).
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.grep_q_pipefail import GrepQPipefailConfig, run_grep_q_pipefail_enforcement

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_grep_q_pipeline_under_pipefail() -> None:
    run_grep_q_pipefail_enforcement(GrepQPipefailConfig(repo_root=REPO_ROOT))
