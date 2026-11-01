"""The promotion read: a candidate arm is promoted only when the engine's corrected verdicts show it better.

Each campaign is the toy host's, scripted: a control and a candidate arm measured over its twelve documents, each
stored as a writer case its mint put on the held-out side, every run naming the held-out set and scored by one judge
on one rubric. What the candidate measures decides what the engine's verdicts say, and the read answers from those
verdicts alone: promote when every reading is decided and some is better, keep when one is worse, undecided when any
reading is not decided. It refuses a campaign run over the dev side, a run naming no case set, and arms scored by
different judges, and it never records anything.
"""

from __future__ import annotations

from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_DOCUMENTS, TOYHOST_SCOPE
from packages.evals.tests.writer_promotion_support import (
    BETTER,
    CANDIDATE,
    CONTROL,
    HELD_OUT,
    SCORING_JUDGE,
    WriterMeasures,
    promotion_campaign,
)
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.ops.report_writer.writer_contract import (
    BOUNCES_MEASURE,
    FINISHED_MEASURE,
    MUST_FIX_OPEN_MEASURE,
)
from threetears.evals.ops.report_writer.writer_promotion import WriterPromotion, writer_promotion

#: A candidate shown more accurate and shown worse on cost and time.
_TRADE: dict[str, Any] = {"cost_usd": 0.03, "total_ms": 1400.0, "field_accuracy": 0.91, "layout_fidelity": None}


def _writer(
    *, bounces: tuple[float, float] = (0.0, 0.0), must_fix: tuple[float, float] = (0.0, 0.0), unfinished: int = 0
) -> WriterMeasures:
    """A writer arm's measures on each document: bounces and open must_fix findings spread around a level by the
    document's place (so the arms differ by more than one amount), and the last ``unfinished`` documents not
    finished."""
    return {
        document: {
            BOUNCES_MEASURE: bounces[0] + bounces[1] * (index % 3),
            MUST_FIX_OPEN_MEASURE: must_fix[0] + must_fix[1] * (index % 2),
            FINISHED_MEASURE: index < len(TOYHOST_DOCUMENTS) - unfinished,
        }
        for index, document in enumerate(TOYHOST_DOCUMENTS)
    }


#: A writer that bounces little, leaves few must-fix findings open and finishes every report.
_STEADY_WRITER = _writer(bounces=(0.0, 0.5), must_fix=(0.0, 0.5))

#: The judged layout score of a control, and of a candidate the judge scores higher on every case.
_JUDGED_LOW, _JUDGED_HIGH = {**CONTROL, "layout_fidelity": 1}, {**BETTER, "layout_fidelity": 4}


def _read(candidate: dict[str, Any], control: dict[str, Any] = CONTROL, **campaign: Any) -> WriterPromotion:
    host, campaign_id = promotion_campaign(candidate, control, **campaign)
    return writer_promotion(host, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE)


def test_a_candidate_shown_better_on_every_reading_is_promoted_with_what_was_read() -> None:
    read = _read(BETTER)
    assert read.decision == "promote", read.why
    assert {verdict.outcome for verdict in read.verdicts if verdict.kind == "contrast"} == {"improved"}
    assert {verdict.arm for verdict in read.verdicts} == {CANDIDATE}
    assert read.candidate == CANDIDATE and read.control == "control"
    assert read.changed == ["chunk_tokens"]
    assert read.levers["chunk_tokens"] == CANDIDATE
    assert read.writer_model == read.levers["model"]
    assert read.cases == len(TOYHOST_DOCUMENTS)
    assert read.case_sets == [HELD_OUT]
    assert read.cost_usd is not None and read.control_cost_usd is not None
    assert read.cost_usd < read.control_cost_usd
    assert read.shipped.skill == load_skill().digest


def test_a_candidate_shown_worse_on_a_reading_is_kept_out_however_much_better_it_is_on_another() -> None:
    read = _read(_TRADE)
    assert read.decision == "keep", read.why
    assert {"improved", "regressed"} <= {verdict.outcome for verdict in read.verdicts}


def test_a_reading_left_undecided_never_promotes_even_beside_every_other_reading_improved() -> None:
    """The judged layout score is not separated while every measure improves: the read stays undecided."""
    read = _read({**BETTER, "layout_fidelity": 4}, {**CONTROL, "layout_fidelity": 2})
    outcomes = [verdict.outcome for verdict in read.verdicts if verdict.kind == "contrast"]
    assert "not_separated" in outcomes and "improved" in outcomes
    assert read.decision == "undecided", read.why


def test_a_candidate_measured_as_the_control_is_undecided() -> None:
    assert _read(CONTROL).decision == "undecided"


def test_a_campaign_over_the_dev_side_is_refused() -> None:
    with pytest.raises(ValidationFailedError, match="is on the dev side of its split"):
        _read(BETTER, side="dev")


def test_a_run_naming_no_case_set_is_refused() -> None:
    with pytest.raises(ValidationFailedError, match="named no case set"):
        _read(BETTER, case_sets=(HELD_OUT, None))


def test_arms_scored_by_different_judges_are_refused() -> None:
    with pytest.raises(ValidationFailedError, match="scored under 2 judges"):
        _read(BETTER, judges=(SCORING_JUDGE, "openai/gpt-mini"))


def test_the_control_is_no_candidate() -> None:
    host, campaign_id = promotion_campaign(BETTER)
    with pytest.raises(ValidationFailedError, match="is the campaign's control"):
        writer_promotion(host, campaign_id, TOYHOST_SCOPE, candidate="control")


def test_the_read_records_no_call() -> None:
    """Promotion is a person's act: the read leaves the campaign as it was."""
    host, campaign_id = promotion_campaign(BETTER)
    assert writer_promotion(host, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE).decision == "promote"
    stored = host.storage.load_campaign(campaign_id, TOYHOST_SCOPE)
    assert stored is not None and stored.call is None


def test_a_writer_better_on_its_own_measures_and_every_other_reading_is_promoted() -> None:
    control = _writer(bounces=(3.0, 1.0), must_fix=(2.0, 1.0), unfinished=len(TOYHOST_DOCUMENTS))
    read = _read(_JUDGED_HIGH, _JUDGED_LOW, writer=(control, _STEADY_WRITER))
    assert read.decision == "promote", read.why
    improved = {verdict.name for verdict in read.verdicts if verdict.outcome == "improved"}
    assert {BOUNCES_MEASURE, MUST_FIX_OPEN_MEASURE, FINISHED_MEASURE} <= improved


@pytest.mark.parametrize(
    ("worse", "regressed"),
    [
        pytest.param(_writer(bounces=(4.0, 2.0), must_fix=(0.0, 0.5)), BOUNCES_MEASURE, id="more bounces"),
        pytest.param(_writer(bounces=(0.0, 0.5), must_fix=(3.0, 2.0)), MUST_FIX_OPEN_MEASURE, id="more must-fix open"),
        pytest.param(
            _writer(bounces=(0.0, 0.5), must_fix=(0.0, 0.5), unfinished=len(TOYHOST_DOCUMENTS)),
            FINISHED_MEASURE,
            id="none finished",
        ),
    ],
)
def test_a_writer_the_judge_scores_higher_is_kept_out_when_its_own_measures_are_shown_worse(
    worse: WriterMeasures, regressed: str
) -> None:
    """The judge scores the candidate's pages higher and every reading but the writer's own improves; the writer
    bounces more, leaves more must-fix findings open, or finishes fewer reports than the control, and is kept out."""
    read = _read(_JUDGED_HIGH, _JUDGED_LOW, writer=(_STEADY_WRITER, worse))
    assert read.decision == "keep", read.why
    outcomes = {verdict.name: verdict.outcome for verdict in read.verdicts if verdict.kind == "contrast"}
    assert outcomes["extraction.layout_fidelity"] == "improved"
    assert outcomes[regressed] == "regressed"


def test_a_campaign_run_the_store_cannot_return_is_refused_by_name() -> None:
    host, campaign_id = promotion_campaign(BETTER)
    stored = host.storage.load_campaign(campaign_id, TOYHOST_SCOPE)
    assert stored is not None
    (aspect,) = stored.aspects
    gone = aspect.model_copy(update={"run_ids": [*aspect.run_ids, "run-gone"]})
    host.storage.save_campaign(stored.model_copy(update={"aspects": [gone]}))
    with pytest.raises(NotFoundError, match="run-gone"):
        writer_promotion(host, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE)
