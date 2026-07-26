"""Thin shell over the canonical no-silent-swallow walker.

An exception handler logs, re-raises, or carries a ``# NOSILENT: <reason>`` marker. A
handler whose body is only ``pass`` hides the failure until the failure is the outage, and
a ``contextlib.suppress`` is the same thing with better manners.

**Scoped to ``threetears.core`` for now, deliberately.** The walker reports 94 handlers
across the workspace; core's 11 are marked, and every one turned out to be legitimate --
optional-dependency probes, best-effort teardown, an awaited cancellation, and temp-file
cleanup on a path that re-raises two lines later. The remaining 83 are the same review
done thirteen more times, and marking them unread would produce exactly the rubber-stamp
this rule exists to prevent. Widening one package at a time keeps each batch small enough
to actually read.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.no_silent_swallow import (
    NoSilentSwallowConfig,
    run_no_silent_swallow_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFIG = NoSilentSwallowConfig(
    repo_root=_REPO_ROOT,
    src_roots=(_REPO_ROOT / "packages" / "core" / "src" / "threetears" / "core",),
)


def test_no_handler_swallows_without_saying_why() -> None:
    """Deliberate silence is fine and has to be stated; accidental silence is the bug."""
    run_no_silent_swallow_enforcement(_CONFIG)
