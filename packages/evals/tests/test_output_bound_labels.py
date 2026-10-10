"""Output-bound labels (#628): a person's rating holds for what they read, not only for the result they read it on.

A ``CalibrationRating`` was reusable only by judges scoring the same stored result. These tests pin the label key
that binds it to the judged output and the criterion — :func:`fingerprint_judged_output`, :func:`fingerprint_criterion`
— its stamping by the judge and by the rating write, its storage read, and agreement finding a label by either
route without counting it twice.

Done when: a label given on one result is found for a second result with byte-identical judged output and the
same criterion (:class:`TestAgreementFindsALabelByWhatWasRead`).

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``fingerprint_criterion``: hashing the dimension's name alone; dropping the scale.
- ``fingerprint_judged_output``: hashing the artifact alone.
- ``JudgeService._score``: not stamping the request's label key.
- ``rate_result``: not copying the score's key onto the rating.
- ``judge_agreement``: dropping the label-key route; dropping the once-per-judge guard.
- the bundle: reading ratings per run only.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import assemble_context_bundle, judge_agreement
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.run import rate_result
from threetears.evals.run.judge_service import JudgeContext, JudgeService
from threetears.evals.schema import (
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    CalibrationRating,
    EvalResult,
    JudgedArtifact,
    JudgeEvidence,
    LabelKey,
    RubricDim,
    RubricScore,
    fingerprint_criterion,
    fingerprint_judged_output,
    label_key_of,
)
from threetears.evals.schema.base import CoreDocumentModel
from packages.evals.tests.factories import make_calibration_rating, make_eval_result, memory_storage
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION, TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

TONE = "conversation.tone"
EVIDENCE = JudgeEvidence(subject="a support agent", case_material="the queue holds 2 tickets", artifact="Agent: two.")
DIM = RubricDim(name=TONE, description="Warm and on topic.", scale="ordinal", scoring_guide={"5": "warm"})
KEY = label_key_of(EVIDENCE, DIM)


def _scored(
    result_id: str, score: int, *, judge: str = "judge-a", key: LabelKey | None = KEY, **extra: Any
) -> EvalResult:
    """A result whose judge scored ``TONE``, stamped with ``key``."""
    stamp = {} if key is None else {"output_fingerprint": key[0], "criterion_fingerprint": key[1]}
    return make_eval_result(
        id=result_id,
        rubric_scores=[RubricScore(dim=TONE, score=score, scale="ordinal", served_model=judge, **stamp)],
        **extra,
    )


def _rating(result_id: str, score: int, *, rater: str = "host", key: LabelKey | None = KEY) -> CalibrationRating:
    stamp = {} if key is None else {"output_fingerprint": key[0], "criterion_fingerprint": key[1]}
    return make_calibration_rating(result_id=result_id, score=score, rater=rater, **stamp)


# --- the fingerprints --------------------------------------------------------------------------------


class TestTheOutputFingerprint:
    def test_byte_identical_evidence_is_one_output(self) -> None:
        again = JudgeEvidence(**EVIDENCE.model_dump())
        assert fingerprint_judged_output(again) == fingerprint_judged_output(EVIDENCE)
        assert re.fullmatch(r"[0-9a-f]{64}", fingerprint_judged_output(EVIDENCE))

    @pytest.mark.parametrize(
        "change",
        [
            {"artifact": "Agent: two. "},  # whitespace is part of what was read
            {"case_material": "the queue holds 3 tickets"},
            {"subject": None},
        ],
    )
    def test_any_byte_the_judge_reads_is_another_output(self, change: dict[str, Any]) -> None:
        assert fingerprint_judged_output(EVIDENCE.model_copy(update=change)) != fingerprint_judged_output(EVIDENCE)


class TestTheCriterionFingerprint:
    def test_the_same_definition_is_one_criterion_and_the_axis_is_not_part_of_it(self) -> None:
        assert fingerprint_criterion(DIM.model_copy()) == fingerprint_criterion(DIM)
        assert fingerprint_criterion(DIM.model_copy(update={"axis": "boundary"})) == fingerprint_criterion(DIM)

    @pytest.mark.parametrize(
        "change",
        [
            {"description": "Warm, polite and on topic."},
            {"scoring_guide": {"5": "very warm"}},
            {"name": "conversation.warmth"},
        ],
    )
    def test_a_reworded_or_renamed_dimension_is_another_criterion(self, change: dict[str, Any]) -> None:
        assert fingerprint_criterion(DIM.model_copy(update=change)) != fingerprint_criterion(DIM)

    def test_a_dimension_moved_to_another_scale_is_another_criterion(self) -> None:
        pass_fail = RubricDim(name=TONE, description=DIM.description, scale="pass_fail")
        ordinal = RubricDim(name=TONE, description=DIM.description, scale="ordinal")
        assert fingerprint_criterion(pass_fail) != fingerprint_criterion(ordinal)

    def test_a_reserved_axis_is_its_own_criterion_and_a_bare_name_is_refused(self) -> None:
        assert fingerprint_criterion(TRANSCRIPT_DIM_ID) != fingerprint_criterion(OUTCOME_DIM_ID)
        with pytest.raises(ValueError, match="not a reserved axis id"):
            fingerprint_criterion(TONE)


class TestTheKeyOnTheDocuments:
    def test_half_a_key_is_refused_on_a_score_and_on_a_rating(self) -> None:
        with pytest.raises(ValidationError, match="set together"):
            RubricScore(dim=TONE, score=3, scale="ordinal", output_fingerprint=KEY.output_fingerprint)
        with pytest.raises(ValidationError, match="set together"):
            make_calibration_rating(criterion_fingerprint=KEY.criterion_fingerprint)

    def test_a_fingerprint_that_is_not_a_digest_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="output_fingerprint"):
            make_calibration_rating(output_fingerprint="abc", criterion_fingerprint=KEY.criterion_fingerprint)

    def test_the_key_is_an_address_not_an_identity(self) -> None:
        keyed, bare = _rating("r-1", 4), _rating("r-1", 4, key=None)
        assert keyed.label_key == KEY and bare.label_key is None
        assert keyed.id == bare.id, "the id stays the result's: one rater's rating of one result is one row"

    def test_a_v8_rating_and_result_read_with_no_key(self) -> None:
        rating = {**make_calibration_rating().to_dict(), "schema_version": 8}
        result = {**make_eval_result().to_dict(), "schema_version": 8}
        for document in (rating, result):
            document.pop("output_fingerprint", None)
        read_rating = CalibrationRating.from_dict(rating)
        read_result = EvalResult.from_dict(result)
        assert read_rating.label_key is None
        assert all(score.label_key is None for score in read_result.rubric_scores)
        assert isinstance(read_rating, CoreDocumentModel)


# --- stamping: the judge, and the rating write --------------------------------------------------------


class _Client:
    """A judge client answering 4 on whatever single dimension it is asked."""

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        match = re.search(r'the single key "(.+?)"', system)
        dim = match.group(1) if match else "?"
        return SimpleNamespace(
            content=json.dumps({"reasoning": "ok", "criteria_scores": {dim: 4}}),
            served_model="judge-a",
            stop_reason="end_turn",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            model="judge-a",
            temperature=0.0,
        )


def _context() -> JudgeContext:
    return JudgeContext(
        case_id="tc-1",
        intent="answer politely",
        variation={},
        goal_outcomes=[],
        judged_artifact=JudgedArtifact.TRANSCRIPT,
        judge_evidence=EVIDENCE,
    )


class TestTheJudgeStampsWhatItRead:
    async def test_every_score_carries_the_key_of_its_evidence_and_criterion(self) -> None:
        service = JudgeService(client_factory=lambda *_: _Client(), failure_describer=withhold_failure_detail)

        dim = await service.score_dimension(DIM, _context())
        transcript = await service.score_transcript(_context())
        outcome = await service.score_outcome(_context())

        assert dim.score is not None and dim.score.label_key == KEY
        assert transcript.score is not None and transcript.score.label_key == label_key_of(EVIDENCE, TRANSCRIPT_DIM_ID)
        assert outcome.score is not None and outcome.score.label_key == label_key_of(EVIDENCE, OUTCOME_DIM_ID)
        assert service.dimension_request(DIM, _context()).label_key == KEY


class TestTheRatingWriteCopiesTheKey:
    def test_a_rating_of_a_stamped_score_carries_its_key(self) -> None:
        storage, _ = memory_storage()
        storage.save_eval_result(_scored("r-1", 3, eval_run_id="run-7"))

        rating = rate_result(
            storage,
            result_id="r-1",
            scope_id="uni-1",
            rubric_dim=TONE,
            rater="host",
            rater_kind="person",
            score=4,
            reason="warm",
        )

        assert rating.label_key == KEY
        assert storage.query_calibration_ratings("uni-1", label_key=KEY) == [rating]

    def test_a_rating_of_an_unstamped_score_carries_none_and_is_read_by_its_result(self) -> None:
        storage, _ = memory_storage()
        storage.save_eval_result(_scored("r-1", 3, key=None))

        rating = rate_result(
            storage,
            result_id="r-1",
            scope_id="uni-1",
            rubric_dim=TONE,
            rater="host",
            rater_kind="person",
            score=4,
            reason="warm",
        )

        assert rating.label_key is None
        assert storage.query_calibration_ratings("uni-1", label_key=KEY) == []
        assert storage.query_calibration_ratings("uni-1", result_id="r-1") == [rating]


class TestTheStoreReadsByKey:
    def test_every_label_of_one_output_on_one_criterion_and_nothing_else(self) -> None:
        storage, _ = memory_storage()
        other = label_key_of(EVIDENCE, DIM.model_copy(update={"description": "Reworded."}))
        here, there = _rating("r-1", 4), _rating("r-2", 2, rater="second")
        storage.save_calibration_rating(here)
        storage.save_calibration_rating(there)
        storage.save_calibration_rating(_rating("r-3", 4, key=other))
        storage.save_calibration_rating(_rating("r-4", 4, key=None))

        assert storage.query_calibration_ratings("uni-1", label_key=KEY) == [here, there]
        assert storage.query_calibration_ratings("uni-2", label_key=KEY) == []


# --- agreement ---------------------------------------------------------------------------------------


class TestAgreementFindsALabelByWhatWasRead:
    def test_a_label_on_one_result_is_found_for_a_second_with_the_same_output_and_criterion(self) -> None:
        """The done-when: a second judge scoring a byte-identical output reads the person's label."""
        agreement = judge_agreement([_rating("r-1", 4)], [_scored("r-1", 4), _scored("r-2", 2, judge="judge-b")])

        by_judge = {row.judge_model: row for row in agreement.dimensions}
        assert set(by_judge) == {"judge-a", "judge-b"}
        assert (by_judge["judge-b"].n, by_judge["judge-b"].exact_agreement) == (1, 0.0)
        assert agreement.unpaired == []

    def test_a_label_found_by_both_routes_enters_one_judge_once(self) -> None:
        agreement = judge_agreement([_rating("r-1", 4)], [_scored("r-1", 4), _scored("r-2", 4), _scored("r-3", 4)])

        (row,) = agreement.dimensions
        assert row.n == 1, "one person's one answer is one pair per judge, however many copies it scored"
        assert row.raters == ["host"]

    def test_a_label_whose_result_is_gone_still_pairs_with_the_same_output(self) -> None:
        agreement = judge_agreement([_rating("deleted", 4)], [_scored("r-2", 4)])

        (row,) = agreement.dimensions
        assert row.n == 1 and agreement.unpaired == []

    def test_a_label_does_not_follow_an_output_to_a_reworded_criterion(self) -> None:
        reworded = label_key_of(EVIDENCE, DIM.model_copy(update={"description": "Reworded."}))
        agreement = judge_agreement([_rating("deleted", 4)], [_scored("r-2", 4, key=reworded)])

        assert agreement.dimensions == []
        assert [u.reason for u in agreement.unpaired] == ["result_unresolved"]

    def test_a_label_does_not_follow_to_another_output(self) -> None:
        elsewhere = label_key_of(EVIDENCE.model_copy(update={"artifact": "Agent: three."}), DIM)
        agreement = judge_agreement([_rating("deleted", 4)], [_scored("r-2", 4, key=elsewhere)])

        assert agreement.dimensions == []

    def test_a_label_without_a_key_is_read_by_its_result_alone(self) -> None:
        agreement = judge_agreement([_rating("r-1", 4, key=None)], [_scored("r-1", 4), _scored("r-2", 2, judge="b")])

        (row,) = agreement.dimensions
        assert (row.judge_model, row.n) == ("judge-a", 1)

    def test_an_agents_label_is_never_paired_by_either_route(self) -> None:
        agent = make_calibration_rating(
            rater_kind="agent",
            output_fingerprint=KEY.output_fingerprint,
            criterion_fingerprint=KEY.criterion_fingerprint,
        )
        agreement = judge_agreement([agent], [_scored("r-1", 4), _scored("r-2", 4, judge="judge-b")])

        assert agreement.dimensions == []
        assert [u.reason for u in agreement.unpaired] == ["rated_by_an_agent"]


class _StampedStore:
    """The toy campaign's store, with every judged score stamped with one label key on the way out.

    Delegates everything else to the toy store; the stamp stands in for a judge that recorded what it read.
    """

    def __init__(self, inner: Any, key: LabelKey) -> None:
        self._inner = inner
        self._key = key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def query_eval_results_by_run(self, run_id: str, scope_id: str) -> list[EvalResult]:
        stamp = {"output_fingerprint": self._key[0], "criterion_fingerprint": self._key[1]}
        return [
            result.model_copy(
                update={"rubric_scores": [score.model_copy(update=stamp) for score in result.rubric_scores]}
            )
            for result in self._inner.query_eval_results_by_run(run_id, scope_id)
        ]


class TestTheBundleReadsLabelsOfWhatItsJudgeScored:
    def test_a_label_given_outside_the_campaign_on_the_same_output_is_read(self) -> None:
        campaign, toy = toyhost_campaign()
        criterion = RubricDim(name=TOYHOST_JUDGED_DIMENSION, description="layout kept", scale="ordinal")
        key = label_key_of(EVIDENCE, criterion)
        storage = _StampedStore(toy, key)
        toy.save_calibration_rating(
            CalibrationRating(
                scope_id=TOYHOST_SCOPE,
                run_id="a-run-outside-the-campaign",
                result_id="a-result-outside-the-campaign",
                rubric_dim=TOYHOST_JUDGED_DIMENSION,
                rater="reviewer-1",
                rater_kind="person",
                scale="ordinal",
                score=3,
                reason="read the same extraction elsewhere",
                output_fingerprint=key.output_fingerprint,
                criterion_fingerprint=key.criterion_fingerprint,
            )
        )

        bundle = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())  # type: ignore[arg-type]

        (row,) = bundle.judge_agreement.dimensions
        assert (row.rubric_dim, row.n, row.raters) == (TOYHOST_JUDGED_DIMENSION, 1, ["reviewer-1"])
        assert bundle.judge_agreement.unpaired == []
