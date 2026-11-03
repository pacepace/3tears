"""The pairwise preference instrument: its criterion, the evidence of one order of a pair, and one case's reading.

The criterion is built from the shipped editor rubric's items and digests stably: built twice it is one criterion,
and an item's text reaches it, so a changed item is another. One order's evidence lays out Report A then Report B, each page with its own facts
labelled, and swapping the pages swaps the labels and nothing else. A case's reading turns each reply into the
candidate's share from whichever slot it sat in, so a judge that answers the two orders as mirrors gives the candidate
the same share from both; a "can't tell" reply is in no share and is counted; order consistency and the lean to the
first slot are read beside the share.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from threetears.evals.ops.report_writer.preference import (
    PairJudgment,
    case_preference,
    pair_evidence,
    preference_rubric,
)
from threetears.evals.ops.report_writer.rubric import shipped_rubric
from threetears.evals.schema.models import JudgeEvidence, fingerprint_criterion, fingerprint_judged_output

_CONTROL = JudgeEvidence(case_material="Computed facts: control pass rate 0.62.", artifact="# Control page\n\nBody C.")
_CANDIDATE = JudgeEvidence(case_material="Computed facts: candidate pass rate 0.71.", artifact="# Candidate page\n\nB.")


def _share(candidate: str, score: int) -> Fraction:
    """What the spec says one reply gives the candidate: (s-1)/4 from B, (5-s)/4 from A."""
    return Fraction(score - 1, 4) if candidate == "B" else Fraction(5 - score, 4)


# --- the criterion --------------------------------------------------------------------------------------------------


def _item_lines() -> list[str]:
    """The shipped rubric's items, one line each, as the criterion lists them."""
    return [f"{item.n}. {item.name}. {item.definition}" for item in shipped_rubric().items]


def test_the_criterion_is_one_ordinal_builtin_dimension_ending_in_every_shipped_item_in_order() -> None:
    dim = preference_rubric()
    assert (dim.name, dim.scale, dim.judge) == ("report_writer.preferred", "ordinal", "builtin")
    assert sorted(dim.scoring_guide) == ["1", "3", "5"]
    assert "Report A" in dim.scoring_guide["1"] and "Report B" in dim.scoring_guide["5"]
    lines = dim.description.splitlines()
    items = _item_lines()
    assert lines[-len(items) :] == items
    # What comes before the items says the score is a direction between the two reports, A first.
    preamble = "\n".join(lines[: -len(items)])
    assert preamble.index("Report A") < preamble.index("Report B")


def test_the_criterion_digests_the_same_every_time_and_another_when_an_item_changes() -> None:
    dim = preference_rubric()
    assert fingerprint_criterion(preference_rubric()) == fingerprint_criterion(dim)
    first = _item_lines()[0]
    reworded = dim.model_copy(
        update={"description": dim.description.replace(first, first + " It also names the cost.")}
    )
    assert reworded.description != dim.description
    assert fingerprint_criterion(reworded) != fingerprint_criterion(dim)


# --- the evidence ---------------------------------------------------------------------------------------------------


def test_a_pair_lays_out_report_a_then_report_b_with_each_pages_facts_labelled() -> None:
    pair = pair_evidence(_CONTROL, _CANDIDATE)
    assert pair.subject is None
    art = pair.artifact
    assert art.index("Report A") < art.index(_CONTROL.artifact) < art.index("Report B") < art.index(_CANDIDATE.artifact)
    facts = pair.case_material
    assert (
        facts.index("Report A")
        < facts.index(_CONTROL.case_material)
        < facts.index("Report B")
        < facts.index(_CANDIDATE.case_material)
    )


def test_the_swapped_pair_is_the_mirror_and_a_judged_output_of_its_own() -> None:
    forward, back = pair_evidence(_CONTROL, _CANDIDATE), pair_evidence(_CANDIDATE, _CONTROL)

    def swapped(text: str, one: str, other: str) -> str:
        return text.replace(one, "\0").replace(other, one).replace("\0", other)

    assert back.artifact == swapped(forward.artifact, _CONTROL.artifact, _CANDIDATE.artifact)
    assert back.case_material == swapped(forward.case_material, _CONTROL.case_material, _CANDIDATE.case_material)
    assert fingerprint_judged_output(forward) != fingerprint_judged_output(back)


@pytest.mark.parametrize("first", [True, False])
def test_a_page_with_a_subject_is_refused_rather_than_dropped(first: bool) -> None:
    briefed = _CONTROL.model_copy(update={"subject": "a brief"})
    with pytest.raises(ValueError, match="subject"):
        pair_evidence(briefed, _CANDIDATE) if first else pair_evidence(_CANDIDATE, briefed)


# --- one case's reading ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("as_b", [1, 2, 3, 4, 5])
def test_a_judge_answering_the_two_orders_as_mirrors_gives_the_same_share_from_each_slot(as_b: int) -> None:
    as_a = 6 - as_b  # the same verdict on the same pages, read from the other slot
    reading = case_preference([PairJudgment("B", as_b), PairJudgment("A", as_a)] * 2)
    expected = float(_share("B", as_b))
    assert reading.share_as_b == expected and reading.share_as_a == float(_share("A", as_a)) == expected
    assert reading.share == expected
    assert (reading.scored, reading.cannot_tell) == (4, 0)
    assert reading.order_consistent is True
    assert reading.first_slot_bias == 0.0


def test_a_judge_that_always_prefers_report_a_reads_as_order_inconsistent_with_a_first_slot_lean() -> None:
    reading = case_preference([PairJudgment("A", 2), PairJudgment("B", 2), PairJudgment("A", 1), PairJudgment("B", 1)])
    as_a = (_share("A", 2) + _share("A", 1)) / 2
    as_b = (_share("B", 2) + _share("B", 1)) / 2
    assert (reading.share_as_a, reading.share_as_b) == (float(as_a), float(as_b))
    assert reading.share == float((as_a + as_b) / 2)
    assert reading.order_consistent is False
    # The page in A's mean share over the two orders, less one half.
    assert reading.first_slot_bias == float((as_a + (1 - as_b)) / 2 - Fraction(1, 2))
    assert reading.first_slot_bias is not None and reading.first_slot_bias > 0


def test_the_share_is_the_mean_of_the_two_orders_so_an_unequal_repeat_count_does_not_weight_one_slot() -> None:
    judgments = [PairJudgment("B", 5), PairJudgment("B", 4), PairJudgment("B", 5), PairJudgment("A", 3)]
    reading = case_preference(judgments)
    as_b = (_share("B", 5) + _share("B", 4) + _share("B", 5)) / 3
    as_a = _share("A", 3)
    assert reading.share == float((as_a + as_b) / 2)
    assert reading.order_consistent is False  # B says the candidate; A says neither


def test_cannot_tell_is_excluded_from_every_share_and_counted() -> None:
    scored = [PairJudgment("B", 4), PairJudgment("A", 2)]
    reading = case_preference([*scored, PairJudgment("B", None), PairJudgment("A", None), PairJudgment("A", None)])
    assert reading == replace(case_preference(scored), cannot_tell=3)
    assert reading.scored == 2


def test_an_order_with_only_cannot_tell_leaves_the_other_orders_share_and_no_consistency_or_lean() -> None:
    reading = case_preference([PairJudgment("B", 4), PairJudgment("B", 5), PairJudgment("A", None)])
    assert reading.share == reading.share_as_b == float((_share("B", 4) + _share("B", 5)) / 2)
    assert reading.share_as_a is None
    assert (reading.order_consistent, reading.first_slot_bias) == (None, None)
    assert (reading.scored, reading.cannot_tell) == (2, 1)


def test_a_case_the_judge_could_not_tell_at_all_has_no_share() -> None:
    reading = case_preference([PairJudgment("A", None), PairJudgment("B", None)])
    assert (reading.share, reading.share_as_a, reading.share_as_b) == (None, None, None)
    assert (reading.scored, reading.cannot_tell) == (0, 2)


@pytest.mark.parametrize("score", [0, 6, True])
def test_a_score_off_the_scale_is_refused(score: int) -> None:
    with pytest.raises(ValueError, match="1-5"):
        case_preference([PairJudgment("B", score)])


def test_a_slot_that_is_neither_a_nor_b_is_refused() -> None:
    with pytest.raises(ValueError, match="neither A nor B"):
        case_preference([PairJudgment("C", 3)])  # type: ignore[arg-type]


def test_a_case_with_no_judgment_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        case_preference([])
