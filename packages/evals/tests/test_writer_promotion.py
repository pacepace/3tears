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

from packages.evals.tests.fixtures.toyhost.campaign import (
    TOYHOST_NARROW,
    TOYHOST_WIDE,
    toyhost_aspect_design,
    toyhost_campaign_of_one_aspect,
)
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_DOCUMENTS,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.campaigns import set_campaign_control
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops.report_writer.writer_kind import (
    WriterCase,
    WriterCaseOrigin,
    WriterSide,
    writer_case_payload,
)
from threetears.evals.ops.report_writer.writer_promotion import WriterPromotion, writer_promotion
from threetears.evals.schema.models import CaseSetRef, EvalTestCase

#: The held-out set every run names unless a test says otherwise.
_HELD_OUT = CaseSetRef(name="writer-held-out", version=1)

#: The scoring judge every run names unless a test says otherwise.
_JUDGE = "google/gemini-flash"

#: What the control measures on each document.
_CONTROL: dict[str, Any] = {"cost_usd": 0.02, "total_ms": 900.0, "field_accuracy": 0.78, "layout_fidelity": None}

#: A candidate shown better than the control on every reading: cheaper, faster, more accurate.
_BETTER: dict[str, Any] = {"cost_usd": 0.01, "total_ms": 600.0, "field_accuracy": 0.99, "layout_fidelity": None}

#: A candidate shown more accurate and shown worse on cost and time.
_TRADE: dict[str, Any] = {"cost_usd": 0.03, "total_ms": 1400.0, "field_accuracy": 0.91, "layout_fidelity": None}

#: The candidate the read is asked about: the arm addressed from the wide batch, by its label.
_CANDIDATE = "1024tok"


def _campaign(
    candidate: dict[str, Any],
    control: dict[str, Any] = _CONTROL,
    *,
    side: WriterSide = "held_out",
    case_sets: tuple[CaseSetRef | None, CaseSetRef | None] = (_HELD_OUT, _HELD_OUT),
    judges: tuple[str, str] = (_JUDGE, _JUDGE),
) -> tuple[EvalHost, str]:
    """The toy host holding a campaign of a control and a candidate, its writer cases on ``side``; and its id."""
    profile = toyhost_profile()
    host = toyhost_host(profile=profile)
    run_ids = []
    for level, values, case_set, judge in zip(
        (TOYHOST_NARROW, TOYHOST_WIDE), (control, candidate), case_sets, judges, strict=True
    ):
        batch = toyhost_batch(
            chunk_tokens=level, retriever_top_k=3, extraction_schema="v1", ocr_engine_version="tess-5.3.1"
        ).model_copy(
            update={
                "case_set": case_set,
                "judge_model": judge,
                "judge_temperature": 0.0,
                "template_digest": "a" * 64,
            }
        )
        host.storage.save_eval_run(batch)
        for result in toyhost_measurements(batch, profile=profile, **values):
            host.storage.save_eval_result(result)
        run_ids.append(batch.id)
    for document in TOYHOST_DOCUMENTS:
        origin = WriterCaseOrigin(
            host_id="toyhost",
            minted_by="test",
            source="snapshot",
            campaign_id=None,
            analysis_id=None,
            group=document,
            side=side,
            case_set=_HELD_OUT.name if side == "held_out" else "writer-dev",
            held_out_fraction=0.3,
            skill=load_skill().digest,
        )
        host.storage.save_test_case(
            EvalTestCase(
                id=document,
                scope_id=TOYHOST_SCOPE,
                template_id="writer-template",
                host_payload=writer_case_payload(WriterCase(snapshot={}, origin=origin)),
            )
        )
    campaign = toyhost_campaign_of_one_aspect(
        id="5f1c2e3d-4b5a-4c6d-8e7f-90a1b2c3d4e5",
        scope_id=TOYHOST_SCOPE,
        name="writer promotion",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=run_ids,
        design=toyhost_aspect_design(bars=[]),
        created_by="test:fixture",
    )
    host.storage.save_campaign(campaign)
    set_campaign_control(host.storage, campaign.id, TOYHOST_SCOPE, run_ids[0], set_by="test", profile=profile)
    return host, campaign.id


def _read(candidate: dict[str, Any], control: dict[str, Any] = _CONTROL, **campaign: Any) -> WriterPromotion:
    host, campaign_id = _campaign(candidate, control, **campaign)
    return writer_promotion(host, campaign_id, TOYHOST_SCOPE, candidate=_CANDIDATE)


def test_a_candidate_shown_better_on_every_reading_is_promoted_with_what_was_read() -> None:
    read = _read(_BETTER)
    assert read.decision == "promote", read.why
    assert {verdict.outcome for verdict in read.verdicts if verdict.kind == "contrast"} == {"improved"}
    assert {verdict.arm for verdict in read.verdicts} == {_CANDIDATE}
    assert read.candidate == _CANDIDATE and read.control == "control"
    assert read.changed == ["chunk_tokens"]
    assert read.levers["chunk_tokens"] == _CANDIDATE
    assert read.writer_model == read.levers["model"]
    assert read.cases == len(TOYHOST_DOCUMENTS)
    assert read.case_sets == [_HELD_OUT]
    assert read.cost_usd is not None and read.control_cost_usd is not None
    assert read.cost_usd < read.control_cost_usd
    assert read.shipped.skill == load_skill().digest


def test_a_candidate_shown_worse_on_a_reading_is_kept_out_however_much_better_it_is_on_another() -> None:
    read = _read(_TRADE)
    assert read.decision == "keep", read.why
    assert {"improved", "regressed"} <= {verdict.outcome for verdict in read.verdicts}


def test_a_reading_left_undecided_never_promotes_even_beside_every_other_reading_improved() -> None:
    """The judged layout score is not separated while every measure improves: the read stays undecided."""
    read = _read({**_BETTER, "layout_fidelity": 4}, {**_CONTROL, "layout_fidelity": 2})
    outcomes = [verdict.outcome for verdict in read.verdicts if verdict.kind == "contrast"]
    assert "not_separated" in outcomes and "improved" in outcomes
    assert read.decision == "undecided", read.why


def test_a_candidate_measured_as_the_control_is_undecided() -> None:
    assert _read(_CONTROL).decision == "undecided"


def test_a_campaign_over_the_dev_side_is_refused() -> None:
    with pytest.raises(ValidationFailedError, match="is on the dev side of its split"):
        _read(_BETTER, side="dev")


def test_a_run_naming_no_case_set_is_refused() -> None:
    with pytest.raises(ValidationFailedError, match="named no case set"):
        _read(_BETTER, case_sets=(_HELD_OUT, None))


def test_arms_scored_by_different_judges_are_refused() -> None:
    with pytest.raises(ValidationFailedError, match="scored under 2 judges"):
        _read(_BETTER, judges=(_JUDGE, "openai/gpt-mini"))


def test_the_control_is_no_candidate() -> None:
    host, campaign_id = _campaign(_BETTER)
    with pytest.raises(ValidationFailedError, match="is the campaign's control"):
        writer_promotion(host, campaign_id, TOYHOST_SCOPE, candidate="control")


def test_the_read_records_no_call() -> None:
    """Promotion is a person's act: the read leaves the campaign as it was."""
    host, campaign_id = _campaign(_BETTER)
    assert writer_promotion(host, campaign_id, TOYHOST_SCOPE, candidate=_CANDIDATE).decision == "promote"
    stored = host.storage.load_campaign(campaign_id, TOYHOST_SCOPE)
    assert stored is not None and stored.call is None
