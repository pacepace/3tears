"""Thin shell over the canonical dict-state walker, over every 3tears package's source.

Persistent state in an ``__init__`` goes in an L1 backend or NATS KV, not a raw dict: a dict is per-process, so two
pods disagree and a restart forgets. And serving code's shared or long-lived dict, set, list and deque state, on an
object or at module level, is never changed in place: its owner assigns the next immutable value whole
(threetears.core.frozen_state; decisions 45/46). The widened walk counts public attributes, lists and deques,
class-body fields and module-level state; function-local and call-scoped containers are exempt by the walker itself,
and an ``asyncio.Queue`` is the approved hand-off. The walker, its rules and the stale-entry meta-check live in
:mod:`threetears.enforcement.dict_state_detection`.

**There is no allowlist and no tracked debt** (Pace, 2026-11-23: redesign it, never waive it). Code that is not
serving code is declared by root, each with its why (Pace approved the three below).
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.dict_state_detection import (
    DictStateConfig,
    run_dict_state_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: every package's source: a package added later is walked without anyone remembering to list it
_SRC_ROOTS: tuple[Path, ...] = tuple(
    sorted(
        root
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for root in _REPO_ROOT.glob(pattern)
        if root.is_dir()
    )
)

#: the code that is not serving code, each with its why
_NON_SERVING: tuple[tuple[str, str], ...] = (
    (
        "packages/enforcement",
        "the rule engine: run by pytest in each repo's enforcement suite, never imported by a serving process",
    ),
    (
        "packages/core/src/threetears/core/testing",
        "test helpers (fakes, fixtures, harnesses) imported only by test suites",
    ),
    (
        "packages/evals",
        "the evaluation harness: run offline against recorded cases, never part of a serving pod",
    ),
)

_CONFIG = DictStateConfig(
    repo_root=_REPO_ROOT,
    src_roots=_SRC_ROOTS,
    immutable_roots=_SRC_ROOTS,
    immutable_widened=True,
    non_serving_roots=tuple(_REPO_ROOT / path for path, _why in _NON_SERVING),
)


class TestDictStateDetection:
    def test_no_unexplained_raw_dict_state(self) -> None:
        """No raw-dict state in an ``__init__``, and no shared or long-lived container changed in place, anywhere in
        3tears' serving code."""
        run_dict_state_enforcement(_CONFIG, walker="detect")

    def test_no_stale_allowlist_entries(self) -> None:
        """Guard the guard: with nothing listed, nothing can go stale; this stays so a list added later is checked."""
        run_dict_state_enforcement(_CONFIG, walker="allowlist_integrity")

    def test_every_source_root_and_non_serving_root_exists(self) -> None:
        """A root that moved would silently stop being walked (or excused): each one must be there, and the walk
        must cover the packages it says it does."""
        assert len(_SRC_ROOTS) > 20, _SRC_ROOTS
        missing = [path for path, _why in _NON_SERVING if not (_REPO_ROOT / path).exists()]
        assert not missing, missing
