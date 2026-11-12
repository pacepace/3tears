"""The editor's clearing rule (owner ruling 2026-11-09, R1): which open findings keep a report from finishing.

A review clears its report when its scores pass (no item at 0 and at least 11 of 14 on the shipped rubric) and no
finding it holds blocks. Every ``must_fix`` blocks, whenever it was raised. A finding first raised in round 1 blocks
only while it was first raised as a ``must_fix``: one first raised there as a ``should_fix`` blocks nothing, whatever
round carries it, and one first raised as a ``must_fix`` and carried at ``should_fix`` still blocks. A ``should_fix``
first raised after round 1 blocks nothing (owner rulings 2026-10-29 and 2026-10-31, unchanged).

Held on :class:`~threetears.evals.ops.report_writer.editor.Review` itself, built as the editor's reading builds it: each
finding with its id, the round it was first raised (``since``) and, where a later review carries it at another
severity, the one it was first raised with (``raised_as``). The loop's tests hold the same rule through the editor's
answers (``test_report_writer_loop.py``).
"""

from __future__ import annotations

from typing import Literal

import pytest

from threetears.evals.kernel.report_doc import ReportFinding
from threetears.evals.ops.report_writer.editor import Review, RubricScore

Severity = Literal["must_fix", "should_fix"]


def scored(*scores: int) -> tuple[RubricScore, ...]:
    """The shipped rubric's seven items, scored as given."""
    return tuple(RubricScore(n=n, name=f"item {n}", score=score, reason="so") for n, score in enumerate(scores, 1))


#: Scores that pass: 13 of 14, no item at 0.
PASSING = scored(2, 2, 2, 2, 2, 1, 2)


def finding(severity: Severity, id: str, since: int, raised_as: Severity | None = None) -> ReportFinding:
    """A finding as the editor's reading keeps it: its id, the round it was first raised, and what it was raised as
    where that differs from its severity now."""
    return ReportFinding.model_validate(
        {
            "claim": "",
            "severity": severity,
            "finding": f"finding {id}",
            "id": id,
            "since": since,
            **({"raised_as": raised_as} if raised_as is not None else {}),
        }
    )


def test_a_should_fix_first_raised_in_round_one_and_carried_open_blocks_nothing() -> None:
    """The run-5 shape that kept 7 shipped and 3 candidate reports from finishing: round-1 should_fix findings carried
    open through the last review, on scores that pass."""
    last = Review(round=3, items=PASSING, findings=(finding("should_fix", "f1.1", 1), finding("should_fix", "f1.2", 1)))

    assert last.blocking == ()
    assert last.clears, "stored finished, its findings listed open"
    assert not last.settles, "a finding is open, so a revision is still given while one remains"


def test_a_round_one_review_whose_findings_are_all_should_fix_clears_and_still_revises() -> None:
    first = Review(round=1, items=PASSING, findings=(finding("should_fix", "f1.1", 1),))

    assert first.blocking == () and first.clears and not first.settles


@pytest.mark.parametrize("round", [1, 2, 3])
def test_every_must_fix_blocks_whenever_it_was_raised(round: int) -> None:
    review = Review(round=round, items=PASSING, findings=(finding("must_fix", f"f{round}.1", round),))

    assert [blocked.id for blocked in review.blocking] == [f"f{round}.1"]
    assert not review.clears


def test_a_round_one_must_fix_carried_as_a_should_fix_still_blocks() -> None:
    """Downgraded is not resolved: a finding first raised in round 1 as a must_fix blocks while a review carries it."""
    last = Review(
        round=3,
        items=PASSING,
        findings=(finding("should_fix", "f1.1", 1, raised_as="must_fix"), finding("should_fix", "f1.2", 1)),
    )

    assert [blocked.id for blocked in last.blocking] == ["f1.1"]
    assert not last.clears


def test_a_round_one_should_fix_carried_as_a_must_fix_blocks_as_a_must_fix() -> None:
    review = Review(round=2, items=PASSING, findings=(finding("must_fix", "f1.1", 1, raised_as="should_fix"),))

    assert [blocked.id for blocked in review.blocking] == ["f1.1"] and not review.clears


def test_a_must_fix_first_raised_after_round_one_and_carried_as_a_should_fix_blocks_nothing() -> None:
    """Only round 1's must_fix findings outlive a downgrade, as before the ruling only round 1's findings did."""
    last = Review(round=3, items=PASSING, findings=(finding("should_fix", "f2.1", 2, raised_as="must_fix"),))

    assert last.blocking == () and last.clears


@pytest.mark.parametrize(
    ("scores", "passes"),
    [
        ((2, 2, 2, 2, 1, 1, 1), True),  # 11 of 14, the bar
        ((2, 2, 2, 1, 1, 1, 1), False),  # 10 of 14
        ((2, 2, 2, 2, 2, 2, 0), False),  # 12 of 14 with an item at 0
    ],
)
def test_the_score_bar_is_unchanged_eleven_of_fourteen_with_no_zero(scores: tuple[int, ...], passes: bool) -> None:
    review = Review(round=3, items=scored(*scores), findings=(finding("should_fix", "f1.1", 1),))

    assert review.passes is passes and review.clears is passes and review.blocking == ()


def test_a_review_s_stored_record_names_what_a_finding_was_raised_as_only_where_it_moved() -> None:
    review = Review(
        round=2,
        items=PASSING,
        findings=(finding("should_fix", "f1.1", 1, raised_as="must_fix"), finding("should_fix", "f1.2", 1)),
    )

    record = review.to_json()

    assert [stored.get("raised_as") for stored in record["findings"]] == ["must_fix", None]
    assert "raised_as" not in record["findings"][1], "a finding at the severity it was raised with keeps its shape"
    assert record["blocking"] == ["f1.1"]
