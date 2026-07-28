"""Thin shell over the canonical no-silent-swallow walker.

An exception handler logs, re-raises, or carries a ``# NOSILENT: <reason>`` marker. A
handler whose body is only ``pass`` hides the failure until the failure is the outage, and
a ``contextlib.suppress`` is the same thing with better manners.

**Workspace-wide.** ``src_roots`` is left unset so the runner discovers every
transitively-reachable src tree rather than reading a hand-maintained list. A new package
is covered the moment it is a path-dep, and cannot be added while quietly sitting outside
the gate.

The scope was widened one package at a time, each only after every handler in it had
actually been read and either fixed or given a reason, because marking them unread would
have produced exactly the rubber-stamp this rule exists to prevent, in the rule that can
least afford one. What that review kept turning up was not the bare ``pass`` but the
handler that returns a plausible value: a memory search answering "nothing found" when
three of its four branches had failed, a replay ending short of the stream head and
reporting a clean tail, a restore reporting success after feeding a partial archive. Those
read as ordinary results at every layer above them.

A marker is for the case where the failure is the expected outcome and there is nothing to
tell anyone: a probing loop trying each candidate encoding in turn, an awaited cancellation
that was requested one line above. Everything else logs.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.no_silent_swallow import (
    NoSilentSwallowConfig,
    run_no_silent_swallow_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFIG = NoSilentSwallowConfig(repo_root=_REPO_ROOT)


def test_no_handler_swallows_without_saying_why() -> None:
    """Deliberate silence is fine and has to be stated; accidental silence is the bug."""
    run_no_silent_swallow_enforcement(_CONFIG)
