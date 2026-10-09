"""a pod's mark for a followed table: what it has accounted for, and what it must drop for.

The contract this pins:

- a table is judged against the mark only once it has one; the first generation a pod sees drops
  the table, because the pod cannot vouch for what it cached before it followed;
- an advance is accounted for when every row message of it has been heard, in any order, and the
  mark moves only across advances heard in full with none skipped;
- a generation the mark has not reached, under the mark's incarnation, is a missed broadcast;
- a generation under another incarnation is a replaced store, whatever its count;
- a pod's own advances need no broadcast;
- a table nobody follows records nothing.
"""

from __future__ import annotations

import pytest

from threetears.core.collections.generation import (
    GenerationMarks,
    GenerationVerdict,
    NoWriteGeneration,
    split_generation_token,
)

_TABLE = "role_assignments"


def _followed(start: str | None = "inc-a:0") -> GenerationMarks:
    """marks following the table, already past their first sight of it at ``start``."""
    marks = GenerationMarks()
    marks.follow(_TABLE)
    assert marks.settle(_TABLE, start) is GenerationVerdict.FIRST_SIGHT
    return marks


class TestATokenSplitsIntoIncarnationAndCount:
    def test_a_followable_token(self) -> None:
        assert split_generation_token("0190-abc:17") == ("0190-abc", 17)

    @pytest.mark.parametrize("token", ["", "no-separator", ":3", "inc:", "inc:x", "inc:-1"])
    def test_anything_else_does_not(self, token: str) -> None:
        assert split_generation_token(token) is None


class TestTheFirstGenerationSeenDropsTheTable:
    def test_a_mark_that_never_saw_a_generation_cannot_vouch_for_the_cache(self) -> None:
        marks = GenerationMarks()
        marks.follow(_TABLE)
        verdict = marks.settle(_TABLE, "inc-a:4")
        assert verdict is GenerationVerdict.FIRST_SIGHT
        assert verdict.drops
        assert marks.recorded(_TABLE) == "inc-a:4"

    def test_the_same_generation_again_is_current(self) -> None:
        marks = _followed("inc-a:4")
        verdict = marks.settle(_TABLE, "inc-a:4")
        assert verdict is GenerationVerdict.CURRENT
        assert not verdict.drops


class TestAnAdvanceIsAccountedForWhenEveryRowWasHeard:
    def test_one_row_of_one_moves_the_mark(self) -> None:
        marks = _followed()
        marks.hear(_TABLE, "inc-a:1", 1)
        assert marks.settle(_TABLE, "inc-a:1") is GenerationVerdict.CURRENT
        assert marks.recorded(_TABLE) == "inc-a:1"

    def test_some_rows_of_many_do_not(self) -> None:
        marks = _followed()
        for _ in range(4):
            marks.hear(_TABLE, "inc-a:1", 5)
        assert marks.judge(_TABLE, "inc-a:1") is GenerationVerdict.MISSED
        marks.hear(_TABLE, "inc-a:1", 5)
        assert marks.judge(_TABLE, "inc-a:1") is GenerationVerdict.CURRENT

    def test_advances_heard_out_of_order_are_all_counted(self) -> None:
        marks = _followed()
        marks.hear(_TABLE, "inc-a:3", 1)
        marks.hear(_TABLE, "inc-a:2", 1)
        assert marks.judge(_TABLE, "inc-a:3") is GenerationVerdict.MISSED
        marks.hear(_TABLE, "inc-a:1", 1)
        assert marks.settle(_TABLE, "inc-a:3") is GenerationVerdict.CURRENT
        assert marks.recorded(_TABLE) == "inc-a:3"

    def test_an_advance_never_heard_is_a_missed_broadcast(self) -> None:
        marks = _followed()
        marks.hear(_TABLE, "inc-a:1", 1)
        marks.hear(_TABLE, "inc-a:3", 1)
        verdict = marks.settle(_TABLE, "inc-a:3")
        assert verdict is GenerationVerdict.MISSED
        assert verdict.drops
        # the drop accounts for everything up to the generation read
        assert marks.settle(_TABLE, "inc-a:3") is GenerationVerdict.CURRENT

    def test_a_broadcast_that_outran_the_read_still_counts_after_a_drop(self) -> None:
        marks = _followed()
        marks.hear(_TABLE, "inc-a:3", 1)
        assert marks.settle(_TABLE, "inc-a:2") is GenerationVerdict.MISSED
        assert marks.settle(_TABLE, "inc-a:3") is GenerationVerdict.CURRENT

    def test_an_older_generation_than_the_mark_is_already_accounted_for(self) -> None:
        marks = _followed()
        marks.hear(_TABLE, "inc-a:1", 1)
        marks.hear(_TABLE, "inc-a:2", 1)
        assert marks.settle(_TABLE, "inc-a:1") is GenerationVerdict.CURRENT

    def test_a_pods_own_advance_needs_no_broadcast(self) -> None:
        marks = _followed()
        marks.account(_TABLE, "inc-a:1")
        assert marks.settle(_TABLE, "inc-a:1") is GenerationVerdict.CURRENT


class TestAnotherIncarnationIsAReplacedStore:
    def test_a_new_incarnation_drops_whatever_its_count(self) -> None:
        marks = _followed("inc-a:9")
        verdict = marks.settle(_TABLE, "inc-b:9")
        assert verdict is GenerationVerdict.REPLACED
        assert verdict.drops
        assert marks.recorded(_TABLE) == "inc-b:9"

    def test_hearing_every_row_of_the_new_incarnation_does_not_excuse_the_old_ones_missed(self) -> None:
        marks = _followed("inc-a:9")
        marks.hear(_TABLE, "inc-b:1", 1)
        assert marks.settle(_TABLE, "inc-b:1") is GenerationVerdict.REPLACED

    def test_a_store_that_lost_the_generation_is_a_replaced_store(self) -> None:
        marks = _followed("inc-a:9")
        assert marks.settle(_TABLE, None) is GenerationVerdict.REPLACED
        assert marks.recorded(_TABLE) is None
        assert marks.settle(_TABLE, None) is GenerationVerdict.CURRENT

    def test_the_first_generation_after_none_is_a_new_incarnation(self) -> None:
        marks = _followed(None)
        assert marks.settle(_TABLE, "inc-a:1") is GenerationVerdict.REPLACED

    def test_a_token_that_is_not_a_count_only_ever_equals_itself(self) -> None:
        marks = _followed("opaque-1")
        assert marks.settle(_TABLE, "opaque-1") is GenerationVerdict.CURRENT
        assert marks.settle(_TABLE, "opaque-2") is GenerationVerdict.REPLACED


class TestWhatIsNotFollowedCostsNothing:
    def test_hearing_a_table_nobody_follows_records_nothing(self) -> None:
        marks = GenerationMarks()
        marks.hear(_TABLE, "inc-a:1", 1)
        marks.account(_TABLE, "inc-a:2")
        assert marks.followed == ()
        assert not marks.follows(_TABLE)
        assert marks.recorded(_TABLE) is None

    def test_judging_a_table_nobody_follows_is_refused(self) -> None:
        with pytest.raises(KeyError):
            GenerationMarks().settle(_TABLE, "inc-a:1")

    def test_following_twice_keeps_the_mark(self) -> None:
        marks = _followed("inc-a:2")
        marks.follow(_TABLE)
        assert marks.recorded(_TABLE) == "inc-a:2"

    def test_a_follower_nobody_runs_a_pass_for_stops_recording_and_is_dropped_when_one_runs(self) -> None:
        marks = _followed()
        # advance 1 is never heard, so nothing after it is contiguous with the mark
        for count in range(2, 6000):
            marks.hear(_TABLE, f"inc-a:{count}", 1)
        assert marks.settle(_TABLE, "inc-a:5999") is GenerationVerdict.MISSED
        assert marks.settle(_TABLE, "inc-a:5999") is GenerationVerdict.CURRENT


class TestTheOptOutCarriesAReason:
    def test_a_reason_is_kept(self) -> None:
        assert NoWriteGeneration(reason="append-only; no pod reads a row by key").reason.startswith("append-only")

    @pytest.mark.parametrize("reason", ["", "   "])
    def test_an_empty_reason_is_refused(self, reason: str) -> None:
        with pytest.raises(ValueError, match="reason"):
            NoWriteGeneration(reason=reason)
