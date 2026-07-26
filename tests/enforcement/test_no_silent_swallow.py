"""Thin shell over the canonical no-silent-swallow walker.

An exception handler logs, re-raises, or carries a ``# NOSILENT: <reason>`` marker. A
handler whose body is only ``pass`` hides the failure until the failure is the outage, and
a ``contextlib.suppress`` is the same thing with better manners.

**RELEASE GATE: this must scan the whole workspace before v0.20.0 ships.** These are
security packages; a handler that swallows silently is an outage that reports nothing, and
shipping one is not a thing to trade against convenience.

**Scoped to ``threetears.core`` today, deliberately.** Every handler it reports there is
marked, and every one turned out to be legitimate -- optional-dependency probes, best-effort
teardown, an awaited cancellation, and temp-file cleanup on a path that re-raises two lines
later. The rest of the workspace is the same review repeated per package, and marking those
unread would produce exactly the rubber-stamp this rule exists to prevent, in the rule that
can least afford one.

Widen ``_CONFIG.src_roots`` one package at a time as each is actually read. Run the walker
for the current count; it prints one. The release ships when this file names every package.
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
    # Widen one package at a time, as each is actually read. A package appears here only
    # once every handler in it has been looked at and either marked with a reason or fixed.
    src_roots=(
        _REPO_ROOT / "packages" / "core" / "src" / "threetears" / "core",
        _REPO_ROOT / "packages" / "iam" / "src" / "threetears" / "iam",
        _REPO_ROOT / "packages" / "mcp" / "src" / "threetears" / "mcp",
    ),
)


def test_no_handler_swallows_without_saying_why() -> None:
    """Deliberate silence is fine and has to be stated; accidental silence is the bug."""
    run_no_silent_swallow_enforcement(_CONFIG)
