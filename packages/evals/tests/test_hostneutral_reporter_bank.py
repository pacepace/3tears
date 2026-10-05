"""The reporter case bank's pure reads, over inputs that carry no host vocabulary.

:mod:`threetears.evals.analysis.reporter_bank` states it is generic by construction: it reads a case
only through :func:`~threetears.evals.analysis.reporter_kind.reporter_case_of` and a result only
through fields the engine records on every result. These tests hold it to that. Each input is a
hand-built case over a neutral bundle document, or the toy host's assembled bundle, and every
dimension, model and label is invented for this file. Nothing here reaches the launch, which is
the host's; each function is driven directly.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.reporter_bank import (
    case_limits,
    criterion_drift,
    frozen_case_receipt,
    label_agrees,
    read_calibration,
    reporter_case_bank,
)
from threetears.evals.analysis.reporter_kind import (
    LABEL_BANDS,
    REPORTER_CASE_KEY,
    LabelCriterion,
    ReporterCase,
    ReporterLabel,
    reporter_case_payload,
)
from threetears.evals.analysis.generator import user_message_digest
from threetears.evals.contracts.identity import IDENTITY_VERSION
from threetears.evals.contracts.models import EvalResult, EvalTestCase, RubricScore
from packages.evals.tests.factories import result_capture_defaults


TEMPLATE = "tpl-summary-review"
SCOPE = "scope-a"
ACCURACY = "summary.accuracy"
CLARITY = "summary.clarity"


def _case(
    *,
    fingerprint: str = "fp-1",
    analysis_id: str | None = "memo-1",
    labels: tuple[ReporterLabel, ...] = (),
    supersedes: tuple[str, ...] = (),
    limits: tuple[str, ...] = (),
) -> ReporterCase:
    """A reporter case over a neutral bundle document, pinning ``analysis_id``'s memo as a freeze would.

    A pinned memo travels with the message its writer read and that message's digest, which the
    freeze takes from the analysis; here the message digests to the recorded value, so the check
    reads ``verified``.
    """
    message = None if analysis_id is None else f"message for {analysis_id}"
    return ReporterCase(
        bundle={"campaign_id": "camp-widgets"},
        bundle_fingerprint=fingerprint,
        bundle_assembled_at="2026-01-01T00:00:00+00:00",
        recorded_analysis_id=analysis_id,
        recorded_memo=None if analysis_id is None else f"memo of {analysis_id}",
        writer_message=message,
        recorded_writer_message_digest=None if message is None else user_message_digest(message),
        labels=list(labels),
        limits=list(limits),
        supersedes=list(supersedes),
    )


def _stored(case_id: str, case: ReporterCase | None = None, *, payload: Any = None) -> EvalTestCase:
    """A stored test case carrying ``case`` (or a raw ``payload``) the way a freeze stores it."""
    host_payload = reporter_case_payload(case) if case is not None else payload
    return EvalTestCase(id=case_id, scope_id=SCOPE, template_id=TEMPLATE, host_payload=host_payload)


#: The criterion a fixture label was frozen against, as a freeze stamps one from the template.
_FROZEN = LabelCriterion(description="as the template stated it", scale="ordinal", scoring_guide={})


def _label(dimension: str, direction: str, quote: str = "reads fine") -> ReporterLabel:
    return ReporterLabel(dimension=dimension, direction=direction, quote=quote, criterion=_FROZEN)


def _result(result_id: str, case_id: str, model: str, k: int, scores: dict[str, int], **extra: Any) -> EvalResult:
    return EvalResult(
        id=result_id,
        scope_id=SCOPE,
        eval_run_id="run-1",
        test_case_id=case_id,
        model=model,
        k_iteration=k,
        rubric_scores=[
            RubricScore(dim=dim, score=score, reasoning=f"why {dim}", scale="ordinal") for dim, score in scores.items()
        ],
        **{**result_capture_defaults(), **extra},
    )


# --- what a frozen bundle cannot support -----------------------------------------------------------


@pytest.fixture(scope="module")
def toy_bundle():
    """The toy host's assembled bundle: two described arms, each with its cell."""
    from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

    profile = toyhost_profile()
    from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle

    return toyhost_bundle(profile=profile)


class TestCaseLimits:
    def test_a_bundle_that_reproduces_with_every_arm_described_and_celled_has_no_limits(self, toy_bundle) -> None:
        assert len(toy_bundle.variant_index) == 2, "the fixture must carry arms, or 'no limits' is vacuous"
        assert case_limits(toy_bundle) == []
        assert case_limits(toy_bundle, recorded_fingerprint=toy_bundle.fingerprint()) == []

    def test_a_recorded_memo_read_from_another_bundle_is_stated_first(self, toy_bundle) -> None:
        stale = "sha256:not-this-bundle"
        limits = case_limits(toy_bundle, recorded_fingerprint=stale)
        assert len(limits) == 1
        assert stale in limits[0] and toy_bundle.fingerprint() in limits[0]

    def test_an_arm_without_its_levels_and_an_arm_without_a_cell_are_each_one_sentence(self, toy_bundle) -> None:
        first, second = toy_bundle.variant_index
        undescribed = first.model_copy(update={"levels_unavailable": "keyed by a predicate this build lacks"})
        bundle = toy_bundle.model_copy(
            update={
                "variant_index": [undescribed, second],
                "cell_measures": [c for c in toy_bundle.cell_measures if c.variant_key != second.variant_key],
            }
        )
        limits = case_limits(bundle)
        assert len(limits) == 2
        assert first.variant_key in limits[0] and "keyed by a predicate this build lacks" in limits[0]
        assert second.variant_key in limits[1] and "no per-arm cell" in limits[1]

    def test_a_bundle_keying_no_variant_says_so_rather_than_returning_nothing(self, toy_bundle) -> None:
        limits = case_limits(toy_bundle.model_copy(update={"variant_index": []}))
        assert len(limits) == 1
        assert "keys no variant" in limits[0]


# --- which case is live ----------------------------------------------------------------------------


class TestTheCaseBank:
    def test_a_successor_retires_the_case_it_supersedes(self) -> None:
        bank = reporter_case_bank([_stored("c-old", _case()), _stored("c-new", _case(supersedes=("c-old",)))])
        assert bank.is_live("c-new") and not bank.is_live("c-old")
        assert [stored.id for stored, _ in bank.live] == ["c-new"]
        assert dict(bank.superseded_by) == {"c-old": ("c-new",)}
        assert bank.ambiguous == []

    def test_two_live_cases_of_one_pair_are_named_as_ambiguous(self) -> None:
        bank = reporter_case_bank(
            [
                _stored("c-2", _case()),
                _stored("c-1", _case(fingerprint="fp-2")),
                _stored("c-other", _case(analysis_id="memo-2")),
            ]
        )
        # One memo is one pair whatever bundle it was frozen from, so c-1 (a later re-assembly of
        # memo-1) is a second live case of the same pair; c-other pins a different memo.
        assert len(bank.ambiguous) == 1
        (pair,) = bank.ambiguous
        assert (pair.campaign_id, pair.recorded_analysis_id, pair.live_case_ids) == (
            "camp-widgets",
            "memo-1",
            ("c-1", "c-2"),
        )
        assert [stored.id for stored, _ in bank.pair("camp-widgets", "memo-1")] == ["c-2", "c-1"]

    def test_an_unreadable_case_is_carried_and_a_case_with_no_reporter_payload_is_skipped(self) -> None:
        bank = reporter_case_bank(
            [
                _stored("c-bad", payload={REPORTER_CASE_KEY: {"bundle_fingerprint": "fp-1"}}),
                _stored("c-plain", payload={"something_else": 1}),
                _stored("c-ok", _case()),
            ]
        )
        assert [case_id for case_id, _ in bank.unreadable] == ["c-bad"]
        assert [stored.id for stored, _ in bank.cases] == ["c-ok"]


# --- the receipt a freeze answers with -------------------------------------------------------------


class TestTheFreezeReceipt:
    def test_the_receipt_projects_the_stored_case(self) -> None:
        label = _label(ACCURACY, "high")
        receipt = frozen_case_receipt(
            _stored("c-new", _case(labels=(label,), supersedes=("c-old",), limits=("an arm lacks its cell",)))
        )
        assert receipt.test_case_id == "c-new"
        assert (receipt.template_id, receipt.scope_id) == (TEMPLATE, SCOPE)
        assert (receipt.source_campaign_id, receipt.recorded_analysis_id) == ("camp-widgets", "memo-1")
        assert receipt.bundle_fingerprint == "fp-1"
        assert receipt.labels == [label]
        assert receipt.limits == ["an arm lacks its cell"]
        assert receipt.supersedes == ["c-old"]

    def test_a_case_carrying_no_reporter_payload_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries no reporter case"):
            frozen_case_receipt(_stored("c-plain", payload={}))


# --- reading a run against its labels --------------------------------------------------------------


class TestReadingCalibration:
    @pytest.mark.parametrize(("direction", "inside", "outside"), [("low", 2, 3), ("mid", 3, 4), ("high", 4, 3)])
    def test_a_label_band_is_inclusive_and_closed(self, direction, inside, outside) -> None:
        low, high = LABEL_BANDS[direction]

        def score(value: int) -> RubricScore:
            return RubricScore(dim="x.d", score=value, scale="ordinal")

        assert label_agrees(direction, score(low)) and label_agrees(direction, score(high))
        assert label_agrees(direction, score(inside))
        assert not label_agrees(direction, score(outside))

    def test_a_pass_fail_dimension_has_no_band_and_does_not_abort_the_read(self) -> None:
        """Five directions onto two answers would be a calibration rule nobody ratified, and a raise here
        would abandon every other dimension's reading after the run was paid for."""
        assert label_agrees("high", RubricScore(dim="x.d", scale="pass_fail", score=1)) is None
        assert label_agrees("high", RubricScore(dim="x.d", score=5, scale="ordinal")) is True

    def test_a_pass_fail_dimension_reads_as_pass_or_fail_beside_a_1_to_5_verdict(self) -> None:
        """One pass/fail dimension neither aborts the read nor borrows a band, and reads in its own words.

        The labelled accuracy dimension is pass/fail, the labelled tone dimension 1-5, and clarity an
        unlabelled pass/fail dimension; the 1-5 reading must keep its verdict whatever the others are.
        """
        result = EvalResult(
            id="r-1",
            scope_id=SCOPE,
            eval_run_id="run-1",
            test_case_id="c-1",
            model="writer-a",
            k_iteration=1,
            rubric_scores=[
                RubricScore(dim=ACCURACY, scale="pass_fail", score=1),
                RubricScore(dim="summary.tone", score=1, scale="ordinal"),
                RubricScore(dim=CLARITY, scale="pass_fail", score=0),
            ],
            termination="completed",
            cost_usd=0.0,
            cost_roles=["candidate", "inner_agent", "judge", "simulator"],
            usage=[],
            covariates={},
            phase_timings={},
            host_measures={},
            variant_key="vk-1",
            identity_version=IDENTITY_VERSION,
        )

        (cell,) = self._read([result]).cases[0].cells

        accuracy, tone = cell.labelled
        assert (accuracy.score_label, accuracy.agrees) == ("pass", None)
        assert (tone.score_label, tone.agrees) == ("1", True)
        assert [(d.dimension, d.score_label) for d in cell.unlabelled] == [(CLARITY, "fail")]

    def _read(self, results: list[EvalResult], **overrides: Any):
        case = _case(labels=(_label(ACCURACY, "high", "every figure checks out"), _label("summary.tone", "low")))
        call: dict[str, Any] = {
            "run_id": "run-1",
            "template_id": TEMPLATE,
            "status": "completed",
            "dimensions": [ACCURACY, CLARITY],
            "cases": [("c-1", case)],
            "results": results,
            "superseded_by": {},
            "archived": {},
            "live_criteria": None,
            "judge_model": "judge-x",
            "effective_judges": {ACCURACY: "judge-x", CLARITY: "judge-y"},
            "ratings": [],
        }
        call.update(overrides)
        return read_calibration(**call)

    def test_each_label_sits_beside_the_judges_score_and_says_whether_it_agrees(self) -> None:
        read = self._read([_result("r-1", "c-1", "writer-a", 1, {ACCURACY: 2, CLARITY: 5})])
        (case,) = read.cases
        (cell,) = case.cells
        accuracy, tone = cell.labelled
        assert (accuracy.dimension, accuracy.score, accuracy.agrees) == (ACCURACY, 2, False)
        assert accuracy.quote == "every figure checks out" and accuracy.reasoning == f"why {ACCURACY}"
        # A label on a dimension the judge scored nothing on: absent, never a disagreement.
        assert (tone.score, tone.agrees) == (None, None)
        assert [(d.dimension, d.score) for d in cell.unlabelled] == [(CLARITY, 5)]

    def test_cells_order_by_model_then_repeat_and_name_an_abnormal_end(self) -> None:
        read = self._read(
            [
                _result("r-3", "c-1", "writer-b", 1, {}),
                _result("r-2", "c-1", "writer-a", 2, {}, termination="cell_timeout"),
                _result("r-1", "c-1", "writer-a", 1, {}, termination="completed"),
                _result("r-x", "c-elsewhere", "writer-a", 1, {}),
            ]
        )
        cells = read.cases[0].cells
        assert [c.result_id for c in cells] == ["r-1", "r-2", "r-3"]
        assert [c.termination for c in cells] == [None, "cell_timeout", None]

    def test_the_read_names_its_judge_and_what_replaced_the_case(self) -> None:
        read = self._read([], superseded_by={"c-1": ["c-3", "c-2"]}, missing_case_ids=["c-gone"])
        assert read.judge_model == "judge-x"
        assert read.effective_judges == {ACCURACY: "judge-x", CLARITY: "judge-y"}
        assert read.cases[0].superseded_by == ["c-2", "c-3"]
        assert read.cases[0].source_campaign_id == "camp-widgets"
        assert read.missing_case_ids == ["c-gone"]

    def test_an_unattributed_run_reads_as_unattributed_not_as_empty(self) -> None:
        read = self._read([], judge_model=None, effective_judges=None)
        assert read.judge_model is None and read.effective_judges is None


# --- what a label was written against, and a retired case ------------------------------------------

_WORDS = LabelCriterion(
    description="every figure matches its source", scale="ordinal", scoring_guide={"5": "all match"}
)


class TestCriterionDrift:
    """A label's frozen criterion against the template's today — three states, each reachable."""

    @pytest.mark.parametrize(
        ("frozen", "live", "expected"),
        [
            pytest.param(_WORDS, {ACCURACY: _WORDS}, "unchanged", id="unchanged"),
            pytest.param(
                _WORDS,
                {ACCURACY: _WORDS.model_copy(update={"description": "every figure matches"})},
                "changed",
                id="reworded",
            ),
            pytest.param(
                _WORDS, {ACCURACY: _WORDS.model_copy(update={"scoring_guide": {}})}, "changed", id="guide-dropped"
            ),
            pytest.param(
                _WORDS, {ACCURACY: _WORDS.model_copy(update={"scale": "pass_fail"})}, "changed", id="rescaled"
            ),
            pytest.param(_WORDS, {CLARITY: _WORDS}, "no_live_criterion", id="dimension-dropped"),
            pytest.param(_WORDS, None, "no_live_criterion", id="template-gone"),
        ],
    )
    def test_each_state(self, frozen, live, expected) -> None:
        label = _label(ACCURACY, "high").model_copy(update={"criterion": frozen})
        assert criterion_drift(label, live) == expected

    def test_a_label_carrying_no_criterion_is_refused_rather_than_compared(self) -> None:
        """Only a stored label, which its freeze stamped, says what it was written against."""
        unstamped = _label(ACCURACY, "high").model_copy(update={"criterion": None})
        with pytest.raises(ValueError, match="carries no criterion"):
            criterion_drift(unstamped, {ACCURACY: _WORDS})

    def test_a_case_storing_a_label_with_no_criterion_is_refused(self) -> None:
        unstamped = _label(ACCURACY, "high").model_copy(update={"criterion": None})
        with pytest.raises(ValidationError, match="carry none"):
            _case(labels=(unstamped,))

    def test_the_read_carries_the_drift_beside_the_agreement_it_qualifies(self) -> None:
        reworded = _label(ACCURACY, "high").model_copy(update={"criterion": _WORDS})
        case = _case(labels=(reworded,))
        read = read_calibration(
            run_id="run-1",
            template_id=TEMPLATE,
            status="completed",
            dimensions=[ACCURACY],
            cases=[("c-1", case)],
            results=[_result("r-1", "c-1", "writer-a", 1, {ACCURACY: 1})],
            superseded_by={},
            archived={},
            live_criteria={ACCURACY: _WORDS.model_copy(update={"description": "now worded otherwise"})},
            judge_model="judge-x",
            effective_judges=None,
            ratings=[],
        )
        (reading,) = read.cases[0].cells[0].labelled
        assert (reading.agrees, reading.criterion_drift) == (False, "changed")


class TestARetiredCase:
    def test_the_bank_carries_it_as_retired_and_not_live(self) -> None:
        retired = _stored("c-1", _case()).model_copy(update={"archived": True, "archived_reason": "orphaned"})
        bank = reporter_case_bank([retired, _stored("c-2", _case(analysis_id="memo-2"))])
        assert dict(bank.archived) == {"c-1": "orphaned"}
        assert not bank.is_live("c-1") and bank.is_live("c-2")
        assert bank.unreadable == ()

    def test_the_read_says_retired_and_checks_the_message_each_recorded_memo_was_judged_against(self) -> None:
        differing = _case().model_copy(update={"writer_message": "a message the writer never read"})
        read = read_calibration(
            run_id="run-1",
            template_id=TEMPLATE,
            status="completed",
            dimensions=[ACCURACY],
            cases=[("c-1", differing), ("c-2", _case(analysis_id="memo-2")), ("c-3", _case(analysis_id=None))],
            results=[],
            superseded_by={},
            archived={"c-2": None},
            live_criteria=None,
            judge_model=None,
            effective_judges=None,
            ratings=[],
        )
        assert [(c.archived, c.archived_reason) for c in read.cases] == [(False, None), (True, None), (False, None)]
        assert [c.writer_message_check for c in read.cases] == ["differs", "verified", None]
