"""per-repo configuration for the memory-only KV gate.

Deliberately without an exemptions path, unlike every other domain in this package. The four
file-backed buckets this rule existed to remove were each exempted with a specific, honest
rationale naming the work that would remove them -- and that is how they stayed for months. The
work is done, so the mechanism goes with it: from here a file-backed bucket cannot be exempted,
only designed out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DEFAULT_SOURCE_GLOBS", "MemoryOnlyKvConfig"]

#: where a repo's importable source lives. Three shapes, because the estate has three: a
#: workspace of packages (``packages/nats/src``), a workspace with nested families
#: (``packages/agent/tools/src`` -- which a single ``packages/*/src`` glob silently missed, so
#: every agent package went unscanned while the gate read green), and a plain ``src`` layout.
DEFAULT_SOURCE_GLOBS: tuple[str, ...] = (
    "packages/*/src/**/*.py",
    "packages/*/*/src/**/*.py",
    "src/**/*.py",
)


@dataclass(frozen=True)
class MemoryOnlyKvConfig:
    """what the gate needs to know about one repo.

    :ivar repo_root: the repo's root, which every reported path is relative to
    :ivar source_globs: where to look for importable source
    """

    repo_root: Path
    source_globs: tuple[str, ...] = field(default=DEFAULT_SOURCE_GLOBS)
