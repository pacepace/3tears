"""configuration for the quiet-grep-under-pipefail enforcement domain.

``cmd | grep -q PATTERN`` under ``set -o pipefail`` fails when it matched: ``grep -q`` exits on its first match, the
command still writing into the pipe dies of SIGPIPE (141), and pipefail makes that the pipeline's status. Output past
the 64 KB pipe buffer makes it certain; shorter output makes it a race a loaded host sometimes loses. The 3tears scrape
sidecar refused to boot with a running window manager this way, and the same bug failed a deploy script's key check
in CI (2026-10-18). The fix is to read the output whole and match it without a pipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DEFAULT_EXCLUDED_DIRS", "DEFAULT_GLOBS", "GrepQPipefailConfig"]

#: what is scanned: shell scripts (checked when they set pipefail) and GitHub workflow files (whose bash steps always
#: run with ``-eo pipefail``)
DEFAULT_GLOBS: tuple[str, ...] = ("**/*.sh", ".github/workflows/*.yml", ".github/workflows/*.yaml")
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {".git", ".venv", "node_modules", "__pycache__", ".tox", "dist", "build"}
)


@dataclass(frozen=True)
class GrepQPipefailConfig:
    """per-repo configuration.

    :ivar repo_root: the repository's root; globs are relative to it
    :ivar globs: what is scanned
    :ivar excluded_dirs: directory names never descended into
    :ivar mode_env_var: the environment variable that switches the runner between strict and report
    """

    repo_root: Path
    globs: tuple[str, ...] = DEFAULT_GLOBS
    excluded_dirs: frozenset[str] = field(default=DEFAULT_EXCLUDED_DIRS)
    mode_env_var: str = "GREP_Q_PIPEFAIL_ENFORCEMENT_MODE"
