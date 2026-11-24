"""Thin shell over the canonical dict-state walker, over every 3tears package's source.

Persistent state in an ``__init__`` goes in an L1 backend or NATS KV, not a raw dict: a dict is per-process, so two
pods disagree and a restart forgets. And serving code's shared or long-lived dict, set, list and deque state, on an
object or at module level, is never changed in place: its owner assigns the next immutable value whole
(threetears.core.frozen_state; decisions 45/46). The widened walk counts public attributes, lists and deques,
class-body fields and module-level state; function-local and call-scoped containers are exempt by the walker itself,
and an ``asyncio.Queue`` is the approved hand-off. The walker, its rules and the stale-entry meta-check live in
:mod:`threetears.enforcement.dict_state_detection`.

**There is no allowlist and no tracked debt** (Pace, 2026-11-23: redesign it, never waive it). Code that is not
serving code is declared by root, each with its why (:mod:`tests.enforcement.serving_code_support`, shared with the
await-race check).
"""

from __future__ import annotations

from threetears.enforcement.dict_state_detection import (
    DictStateConfig,
    run_dict_state_enforcement,
)

from tests.enforcement.serving_code_support import NON_SERVING, NON_SERVING_ROOTS, REPO_ROOT, SRC_ROOTS

_CONFIG = DictStateConfig(
    repo_root=REPO_ROOT,
    src_roots=SRC_ROOTS,
    immutable_roots=SRC_ROOTS,
    immutable_widened=True,
    non_serving_roots=NON_SERVING_ROOTS,
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
        assert len(SRC_ROOTS) > 20, SRC_ROOTS
        missing = [path for path, _why in NON_SERVING if not (REPO_ROOT / path).exists()]
        assert not missing, missing
