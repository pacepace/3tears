"""Thin shell over the await-race walker, over 3tears' serving code.

The dict rule's replace-whole shape has one failure mode: an async function reads a long-lived value, awaits, and
assigns back a value computed from the stale read, losing another coroutine's update made during the await (reports'
refresh_service, fixed 2026-11-23). :mod:`threetears.enforcement.await_race` finds that. The walk covers the
same serving code as the dict rule (:mod:`tests.enforcement.serving_code_support`). **No allowlist**: a finding is
fixed by reading after the await, by a compare-and-set, or by keeping the update with its one owner.
"""

from __future__ import annotations

from threetears.enforcement.await_race import AwaitRaceConfig, run_await_race_enforcement

from tests.enforcement.serving_code_support import NON_SERVING_ROOTS, REPO_ROOT, SRC_ROOTS


def test_no_long_lived_value_is_written_back_from_a_read_before_an_await() -> None:
    """No async function in 3tears' serving code assigns a long-lived target a value derived from a read of it taken
    before an await."""
    run_await_race_enforcement(
        AwaitRaceConfig(repo_root=REPO_ROOT, src_roots=SRC_ROOTS, non_serving_roots=NON_SERVING_ROOTS)
    )
