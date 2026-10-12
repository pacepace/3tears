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
tell anyone: a probing loop trying each candidate encoding in turn, an optional import. And a
marker NAMES ITS KIND -- ``# NOSILENT(expected|attempt|cleanup|optional-import): <reason>`` --
which the walker checks against the code, because a bare reason was accepted for an ``except
ValueError: return False`` that dropped unreadable judge cases from an arm (issue #715). A
handler that answers ``None``, a constant, an empty collection or an assigned default is as
silent as one that ``pass``es, and is held to the same rule -- LOGGING FIRST DOES NOT CHANGE
THAT: the log tells the operator, the caller still gets the default. Where the caller's answer
is still true after a logged failure, ``# NOSILENT(logged): <why>`` says so, and the walker
checks the log is at WARNING or above. Everything else raises, or returns a failure the caller
can tell from an answer. A bare ``# NOSILENT: <reason>`` comment is flagged wherever it sits.

**The nodriver sidecar too.** It never enters the workspace venv (AGPL), so discovery cannot
see it; its modules are scanned by path, its own tests excepted.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.no_silent_swallow import (
    NoSilentSwallowConfig,
    find_silent_swallows,
    run_no_silent_swallow_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFIG = NoSilentSwallowConfig(repo_root=_REPO_ROOT)


def test_no_handler_swallows_without_saying_why() -> None:
    """Deliberate silence is fine and has to be stated; accidental silence is the bug."""
    run_no_silent_swallow_enforcement(_CONFIG)


_SIDECAR = _REPO_ROOT / "packages" / "scrape" / "sidecar"


def test_the_sidecar_swallows_nothing_either() -> None:
    """The sidecar sits outside the workspace, not outside the rule."""
    assert (_SIDECAR / "main.py").is_file(), f"the sidecar moved from {_SIDECAR}"
    found = [
        violation
        for violation in find_silent_swallows(
            (_SIDECAR,),
            _REPO_ROOT,
            _CONFIG.logger_names,
            _CONFIG.logger_methods,
            _CONFIG.nosilent_marker,
        )
        if "tests" not in violation.file.relative_to(_SIDECAR).parts and violation.file.name != "conftest.py"
    ]
    assert not found, "\n".join(f"{v.file}:{v.line} [{v.symbol}] {v.reason}" for v in found)
