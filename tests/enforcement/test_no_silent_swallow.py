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
tell anyone: a probing loop trying each candidate encoding in turn. Everything else logs.

**Broad and cancellation handlers are held to more, and no marker excuses them.** An
``except Exception`` re-raises or calls :func:`threetears.observe.contain`, which logs the
traceback and counts the site on ``contained_errors_total``; an ``except CancelledError``
re-raises, and stopping another task is :func:`threetears.observe.cancel_and_wait`. A log line
was never enough there: the caller still carried on with an answer it could not tell from a
real one. The packages in ``_UNCONTAINED_PENDING`` have not been converted yet. The list only
shrinks -- an entry with nothing left to convert fails the gate -- and a package not on it,
including every new one, is checked.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.no_silent_swallow import (
    NoSilentSwallowConfig,
    run_no_silent_swallow_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: packages whose broad and cancellation handlers are not converted yet; delete a line as its
#: package converts, and the list itself once it is empty.
_UNCONTAINED_PENDING = (
    "packages/agent/audit",
    "packages/agent/intention",
    "packages/agent/knowledge",
    "packages/agent/memory",
    "packages/agent/skills",
    "packages/agent/tools",
    "packages/agent/wake",
    "packages/agent/workspace",
    "packages/backup",
    "packages/channels",
    "packages/conversations",
    "packages/core",
    "packages/datasources",
    "packages/enforcement",
    "packages/epoch",
    "packages/evals",
    "packages/geo",
    "packages/iam",
    "packages/langgraph",
    "packages/mcp",
    "packages/models",
    "packages/nats",
    "packages/registry",
    "packages/scheduled-jobs",
    "packages/scrape",
    "packages/search",
)

_CONFIG = NoSilentSwallowConfig(repo_root=_REPO_ROOT, uncontained_pending=_UNCONTAINED_PENDING)


def test_no_handler_swallows_without_saying_why() -> None:
    """Deliberate silence is fine and has to be stated; accidental silence is the bug."""
    run_no_silent_swallow_enforcement(_CONFIG)
