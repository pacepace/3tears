"""Calibration ratings: people score judged results, and every reader sets the judge beside them.

A judge's agreement with people was promised by the schema (an embedded rating list on every
result) and written by nothing. These tests pin the standalone ``calibration_rating`` document that
replaces it, the one write that makes one (:func:`~threetears.evals.run.rate_result`), the kappa
arithmetic, and both readers: the bundle's ``judge_agreement`` and a reporter run's
``rating_agreement``.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``rate_result``: removing the not-found raise; removing the no-judge-score refusal; taking the
  scale from a caller default rather than the judge's score.
- ``CalibrationRating``: removing the on-scale check; removing the derived-id check.
- ``cohen_kappa``: returning 1.0 instead of ``None`` when chance predicts no disagreement; using
  ``abs(i - j)`` for the quadratic cost.
- ``judge_agreement``: grouping without the judge model; pairing a rating whose scale differs.
- the bundle: dropping the ``judge_agreement=`` argument from the assembly.
- the reporter read: passing ``ratings=[]`` in ``reporter_calibration``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import JudgeAgreement, assemble_context_bundle, judge_agreement, reporter_calibration
from threetears.evals.analysis.reporter_kind import ReporterCase, reporter_case_payload
from threetears.evals.analysis.stats import cohen_kappa
from threetears.evals.contracts import (
    TRANSCRIPT_DIM_ID,
    CalibrationRating,
    EvalResult,
    EvalTestCase,
    NotFoundError,
    RubricScore,
    ValidationFailedError,
)
from threetears.evals.run import rate_result
from packages.evals.tests.factories import make_calibration_rating, make_eval_result, make_eval_run, memory_storage
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION, TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

TONE = "conversation.tone"
HELPED = "conversation.helped"


def _rating(**fields: Any) -> CalibrationRating:
    """One rating of ``TONE`` on result ``r-1``, with ``fields`` over it."""
    return make_calibration_rating(**fields)


def _result(result_id: str, *scores: RubricScore, **extra: Any) -> EvalResult:
    return make_eval_result(id=result_id, rubric_scores=list(scores), **extra)


def _tone(score: int, judge: str | None = "judge-a") -> RubricScore:
    return RubricScore(dim=TONE, score=score, scale="ordinal", served_model=judge)


# --- the document ------------------------------------------------------------------------------------


class TestTheDocument:
    def test_one_rater_one_dimension_one_result_is_one_id(self) -> None:
        first = _rating(score=2, reason="cold")
        again = _rating(score=4, reason="on reflection, warm")
        other_rater = _rating(rater="second reader")
        same_rater_as_agent = _rating(rater_kind="agent")

        assert first.id == again.id, "a re-rating is a correction, and lands on the same row"
        assert other_rater.id != first.id, "two raters are two ratings"
        assert same_rater_as_agent.id != first.id, "one identity as a person and as an agent is two ratings"
        assert first.id.startswith("rating:") and first.id != first.result_id

    def test_an_id_its_fields_do_not_derive_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="is not the one"):
            _rating(id="rating:made-up")

    @pytest.mark.parametrize(("scale", "score"), [("ordinal", 0), ("ordinal", 6), ("pass_fail", 2), ("pass_fail", -1)])
    def test_a_score_off_its_scale_is_refused(self, scale: str, score: int) -> None:
        with pytest.raises(ValidationError, match="is not on the"):
            _rating(scale=scale, score=score)

    @pytest.mark.parametrize("score", [True, "4", 4.0])
    def test_a_score_that_is_not_an_integer_is_refused(self, score: Any) -> None:
        with pytest.raises(ValidationError, match="score"):
            _rating(score=score)

    @pytest.mark.parametrize("field", ["rater", "reason", "result_id", "run_id", "scope_id"])
    def test_a_blank_required_field_is_refused(self, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            _rating(**{field: "  "})

    def test_it_round_trips_through_the_store(self) -> None:
        storage, _ = memory_storage()
        rating = _rating()
        storage.save_calibration_rating(rating)

        assert storage.query_calibration_ratings("uni-1", run_id="run-1") == [rating]
        assert storage.query_calibration_ratings("uni-1", result_id="r-1") == [rating]
        assert storage.query_calibration_ratings("uni-1", run_id="run-2") == []
        assert storage.query_calibration_ratings("uni-2") == [], "a scope reads only its own ratings"


# --- the write ---------------------------------------------------------------------------------------


class TestRateResult:
    def _stored(self, *scores: RubricScore, **extra: Any) -> Any:
        storage, _ = memory_storage()
        storage.save_eval_result(_result("r-1", *scores, eval_run_id="run-7", **extra))
        return storage

    def test_the_run_and_the_scale_are_read_off_the_result(self) -> None:
        storage = self._stored(RubricScore(dim=HELPED, score=1, scale="pass_fail"))

        rating = rate_result(
            storage,
            result_id="r-1",
            scope_id="uni-1",
            rubric_dim=HELPED,
            rater="host",
            rater_kind="person",
            score=0,
            reason="it did not",
        )

        assert (rating.run_id, rating.scale, rating.score) == ("run-7", "pass_fail", 0)
        assert storage.query_calibration_ratings("uni-1", run_id="run-7") == [rating]

    def test_rating_again_replaces_the_raters_earlier_rating(self) -> None:
        storage = self._stored(_tone(3))
        rate_result(
            storage,
            result_id="r-1",
            scope_id="uni-1",
            rubric_dim=TONE,
            rater="host",
            rater_kind="person",
            score=2,
            reason="flat",
        )
        rate_result(
            storage,
            result_id="r-1",
            scope_id="uni-1",
            rubric_dim=TONE,
            rater="host",
            rater_kind="person",
            score=3,
            reason="fine",
        )
        rate_result(
            storage,
            result_id="r-1",
            scope_id="uni-1",
            rubric_dim=TONE,
            rater="guest",
            rater_kind="person",
            score=5,
            reason="lovely",
        )

        stored = storage.query_calibration_ratings("uni-1", result_id="r-1")
        assert sorted((r.rater, r.score, r.reason) for r in stored) == [("guest", 5, "lovely"), ("host", 3, "fine")]

    def test_a_dual_score_axis_can_be_rated(self) -> None:
        storage = self._stored(transcript_score=RubricScore(dim=TRANSCRIPT_DIM_ID, score=4, scale="ordinal"))

        rating = rate_result(
            storage,
            result_id="r-1",
            scope_id="uni-1",
            rubric_dim=TRANSCRIPT_DIM_ID,
            rater="host",
            rater_kind="person",
            score=4,
            reason="ok",
        )

        assert rating.rubric_dim == TRANSCRIPT_DIM_ID

    def test_an_unknown_result_is_refused(self) -> None:
        storage = self._stored(_tone(3))

        with pytest.raises(NotFoundError, match="result 'r-404'"):
            rate_result(
                storage,
                result_id="r-404",
                scope_id="uni-1",
                rubric_dim=TONE,
                rater="h",
                rater_kind="person",
                score=3,
                reason="r",
            )
        with pytest.raises(NotFoundError):
            rate_result(
                storage,
                result_id="r-1",
                scope_id="uni-2",
                rubric_dim=TONE,
                rater="h",
                rater_kind="person",
                score=3,
                reason="r",
            )

    def test_a_dimension_the_judge_did_not_score_is_refused_and_names_what_it_did(self) -> None:
        storage = self._stored(_tone(3))

        with pytest.raises(ValidationFailedError, match=r"no judge score on 'conversation.helped'.*conversation.tone"):
            rate_result(
                storage,
                result_id="r-1",
                scope_id="uni-1",
                rubric_dim=HELPED,
                rater="h",
                rater_kind="person",
                score=1,
                reason="r",
            )
        assert storage.query_calibration_ratings("uni-1") == [], "nothing is written on a refusal"

    @pytest.mark.parametrize(
        ("fields", "match"),
        [
            ({"score": 6}, "is not on the ordinal scale"),
            ({"score": 0}, "is not on the ordinal scale"),
            ({"rater": ""}, "rater"),
            ({"reason": "   "}, "reason"),
            ({"rater_kind": "machine"}, "rater_kind"),
        ],
    )
    def test_an_invalid_rating_is_refused_as_a_validation_failure(self, fields: dict[str, Any], match: str) -> None:
        storage = self._stored(_tone(3))
        call: dict[str, Any] = {"rater": "host", "rater_kind": "person", "score": 3, "reason": "fine", **fields}

        with pytest.raises(ValidationFailedError, match=match):
            rate_result(storage, result_id="r-1", scope_id="uni-1", rubric_dim=TONE, **call)
        assert storage.query_calibration_ratings("uni-1") == []

    def test_a_bare_dimension_name_is_refused(self) -> None:
        storage = self._stored(_tone(3))

        with pytest.raises(ValidationFailedError):
            rate_result(
                storage,
                result_id="r-1",
                scope_id="uni-1",
                rubric_dim="tone",
                rater="h",
                rater_kind="person",
                score=3,
                reason="r",
            )


# --- the arithmetic ----------------------------------------------------------------------------------


class TestCohenKappa:
    """A worked table: six items, three ordered categories.

    Marginals 2/2/2 against 3/2/1, so chance agreement is (6 + 4 + 2) / 36 = 1/3 and the observed 4/6
    gives kappa (2/3 - 1/3) / (2/3) = 0.5. Quadratically, the two near misses cost 1/4 each, so the
    observed disagreement is 1/12 against an expected 1/3: weighted kappa 0.75. Linear weights would
    give 0.625, so the table separates the two.
    """

    PAIRS = [(1, 1), (1, 1), (2, 2), (2, 1), (3, 3), (3, 2)]

    def test_the_worked_table(self) -> None:
        assert cohen_kappa(self.PAIRS, [1, 2, 3]) == pytest.approx(0.5)
        assert cohen_kappa(self.PAIRS, [1, 2, 3], weights="quadratic") == pytest.approx(0.75)

    def test_perfect_disagreement_on_two_categories(self) -> None:
        assert cohen_kappa([(0, 1), (1, 0)], [0, 1]) == pytest.approx(-1.0)
        assert cohen_kappa([(0, 1), (1, 0)], [0, 1], weights="quadratic") == pytest.approx(-1.0)

    def test_undefined_is_none_not_perfect(self) -> None:
        assert cohen_kappa([(3, 3), (3, 3)], [1, 2, 3, 4, 5]) is None, "both gave one score to everything"
        assert cohen_kappa([], [1, 2, 3, 4, 5]) is None

    def test_a_value_off_the_categories_or_a_single_category_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not among the categories"):
            cohen_kappa([(1, 6)], [1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="at least two categories"):
            cohen_kappa([(1, 1)], [1])


# --- pairing -----------------------------------------------------------------------------------------


class TestJudgeAgreement:
    def test_pairs_read_per_dimension_with_n_raters_and_both_kappas(self) -> None:
        results = [_result(f"r-{i}", _tone(judge)) for i, judge in enumerate([1, 1, 2, 2, 3, 3])]
        people = [1, 1, 2, 1, 3, 2]
        ratings = [
            _rating(result_id=f"r-{i}", score=score, rater="host" if i % 2 else "guest")
            for i, score in enumerate(people)
        ]

        agreement = judge_agreement(ratings, results)

        (tone,) = agreement.dimensions
        assert (tone.rubric_dim, tone.scale, tone.judge_model, tone.n) == (TONE, "ordinal", "judge-a", 6)
        assert tone.raters == ["guest", "host"]
        assert tone.exact_agreement == pytest.approx(4 / 6)
        judged = [1, 1, 2, 2, 3, 3]
        guest = [(judged[i], people[i]) for i in range(0, 6, 2)]
        host = [(judged[i], people[i]) for i in range(1, 6, 2)]
        scale = [1, 2, 3, 4, 5]
        # Each person's kappa over the results they rated, pooled weighted by each person's pairs (3 and 3 here).
        assert tone.kappa == pytest.approx((cohen_kappa(guest, scale) + cohen_kappa(host, scale)) / 2)
        assert tone.weighted_kappa == pytest.approx(
            (cohen_kappa(guest, scale, weights="quadratic") + cohen_kappa(host, scale, weights="quadratic")) / 2
        )
        assert (agreement.ratings_read, agreement.unpaired) == (6, [])

    def test_the_weighted_kappa_is_the_quadratic_one_over_the_full_scale(self) -> None:
        """Pinned to its value, on pairs where it differs from the unweighted kappa and from one over seen scores.

        The scores seen are 1, 2 and 5, unevenly spaced, so a kappa weighting the seen scores as if adjacent differs.
        """
        pairs = [(1, 1), (1, 1), (2, 2), (2, 1), (5, 5), (5, 2)]
        results = [_result(f"r-{i}", _tone(judge)) for i, (judge, _) in enumerate(pairs)]
        ratings = [_rating(result_id=f"r-{i}", score=person) for i, (_, person) in enumerate(pairs)]

        (tone,) = judge_agreement(ratings, results).dimensions

        full = cohen_kappa(pairs, [1, 2, 3, 4, 5], weights="quadratic")
        assert tone.weighted_kappa == pytest.approx(full)
        assert tone.weighted_kappa != pytest.approx(cohen_kappa(pairs, [1, 2, 3, 4, 5])), "not the unweighted one"
        assert tone.weighted_kappa != pytest.approx(cohen_kappa(pairs, [1, 2, 5], weights="quadratic")), (
            "not over the observed scores alone"
        )

    def test_two_people_rating_one_result_enter_no_kappa_twice(self) -> None:
        """Pooled, the judge's score would be counted twice and the people's disagreement read as the judge's."""
        judged = [1, 2, 3, 4, 5]
        alice = [1, 2, 3, 4, 5]
        bob = [2, 1]  # bob rated the first two results, which alice rated too
        results = [_result(f"r-{i}", _tone(judge)) for i, judge in enumerate(judged)]
        ratings = [
            *(_rating(result_id=f"r-{i}", score=score, rater="alice") for i, score in enumerate(alice)),
            *(_rating(result_id=f"r-{i}", score=score, rater="bob") for i, score in enumerate(bob)),
        ]

        (tone,) = judge_agreement(ratings, results).dimensions

        scale = [1, 2, 3, 4, 5]
        # Per person, then pooled by result: five results weigh 1 each, the two both people rated split between
        # them — alice carries 3 + 2 x 1/2 = 4, bob 2 x 1/2 = 1.
        per_person = (
            4 * cohen_kappa(list(zip(judged, alice)), scale) + 1 * cohen_kappa(list(zip(judged[:2], bob)), scale)
        ) / 5
        pooled = cohen_kappa(list(zip(judged + judged[:2], alice + bob)), scale)
        assert tone.n == 7 and tone.raters == ["alice", "bob"]
        assert tone.results == 5, "two people rating one result is one result"
        assert tone.kappa == pytest.approx(per_person)
        assert per_person != pytest.approx(pooled), "the fixture separates per-person pooling from one pooled table"

    def test_a_person_whose_kappa_is_undefined_is_left_out_of_the_mean(self) -> None:
        results = [
            _result("r-0", _tone(3)),
            _result("r-1", _tone(3)),
            _result("r-2", _tone(1)),
            _result("r-3", _tone(5)),
        ]
        ratings = [
            _rating(result_id="r-0", score=3, rater="same"),
            _rating(result_id="r-1", score=3, rater="same"),
            _rating(result_id="r-2", score=2, rater="varied"),
            _rating(result_id="r-3", score=5, rater="varied"),
        ]

        (tone,) = judge_agreement(ratings, results).dimensions

        assert cohen_kappa([(3, 3), (3, 3)], [1, 2, 3, 4, 5]) is None
        assert tone.kappa == pytest.approx(cohen_kappa([(1, 2), (5, 5)], [1, 2, 3, 4, 5]))

    def test_each_judge_is_read_separately(self) -> None:
        """Pooling two judges would credit one with the other's calibration — the judge-swap question."""
        results = [
            _result("r-a", _tone(4, "judge-a")),
            _result("r-b", _tone(2, "judge-b")),
            _result("r-n", _tone(3, None)),
        ]
        ratings = [
            _rating(result_id="r-a", score=4),
            _rating(result_id="r-b", score=5),
            _rating(result_id="r-n", score=3),
        ]

        agreement = judge_agreement(ratings, results)

        assert [(d.judge_model, d.n, d.exact_agreement) for d in agreement.dimensions] == [
            (None, 1, 1.0),
            ("judge-a", 1, 1.0),
            ("judge-b", 1, 0.0),
        ]

    def test_pass_fail_carries_no_weighted_kappa(self) -> None:
        results = [
            _result("r-1", RubricScore(dim=HELPED, score=1, scale="pass_fail")),
            _result("r-2", RubricScore(dim=HELPED, score=0, scale="pass_fail")),
        ]
        ratings = [
            _rating(result_id="r-1", rubric_dim=HELPED, scale="pass_fail", score=1),
            _rating(result_id="r-2", rubric_dim=HELPED, scale="pass_fail", score=1),
        ]

        (helped,) = judge_agreement(ratings, results).dimensions

        assert (helped.exact_agreement, helped.weighted_kappa) == (0.5, None)
        assert helped.kappa == pytest.approx(0.0)

    def test_a_rating_with_no_judge_score_to_meet_is_named_with_why(self) -> None:
        results = [
            _result("r-tone", _tone(3)),
            _result("r-moved", RubricScore(dim=TONE, score=1, scale="pass_fail")),
            _result("r-unscored", RubricScore(dim=HELPED, score=1, scale="pass_fail")),
        ]
        ratings = [
            _rating(result_id="r-gone", rater="a"),
            _rating(result_id="r-unscored", rater="b"),
            _rating(result_id="r-moved", rater="c"),
            _rating(result_id="r-tone", rater="d", score=3),
        ]

        agreement = judge_agreement(ratings, results)

        assert [(u.result_id, u.rater, u.reason) for u in agreement.unpaired] == [
            ("r-gone", "a", "result_unresolved"),
            ("r-unscored", "b", "dimension_unscored"),
            ("r-moved", "c", "scale_changed"),
        ]
        assert [(d.n, d.raters) for d in agreement.dimensions] == [(1, ["d"])]
        assert agreement.ratings_read == 4, "the pairs and the unpaired reconcile with what was read"

    def test_an_agents_rating_is_listed_and_never_pooled_into_agreement_with_people(self) -> None:
        """An agent rating through a tool is another model's opinion; read as a person's it would calibrate the judge."""
        results = [_result("r-1", _tone(4)), _result("r-2", _tone(2))]
        ratings = [
            _rating(result_id="r-1", rater="owner", score=4),
            # The agent agrees with the judge everywhere; pooled, it would lift the judge's agreement with people.
            _rating(result_id="r-1", rater="agent:scout", rater_kind="agent", score=4),
            _rating(result_id="r-2", rater="agent:scout", rater_kind="agent", score=2),
        ]

        agreement = judge_agreement(ratings, results)

        (tone,) = agreement.dimensions
        assert (tone.n, tone.raters) == (1, ["owner"]), "only the person's rating is a pair"
        assert [(u.result_id, u.rater, u.reason) for u in agreement.unpaired] == [
            ("r-1", "agent:scout", "rated_by_an_agent"),
            ("r-2", "agent:scout", "rated_by_an_agent"),
        ]
        assert agreement.ratings_read == 3
        people_only = judge_agreement([ratings[0]], results)
        assert people_only.dimensions == agreement.dimensions, "the agent's ratings move no human figure"

    def test_a_person_and_an_agent_under_one_identity_are_two_ratings_and_only_the_person_pairs(self) -> None:
        """An agent acting as the account it serves writes beside that person's rating, never over it.

        Both orders are driven through the write, so neither kind can silently replace the other, and the
        agent disagrees with the person so a pooled read would move the human figures.
        """
        for first, second in (("person", "agent"), ("agent", "person")):
            storage, _ = memory_storage()
            storage.save_eval_result(_result("r-1", _tone(4)))
            scores = {"person": 4, "agent": 1}
            for kind in (first, second):
                rate_result(
                    storage,
                    result_id="r-1",
                    scope_id="uni-1",
                    rubric_dim=TONE,
                    rater="owner",
                    rater_kind=kind,
                    score=scores[kind],
                    reason=f"as {kind}",
                )

            stored = storage.query_calibration_ratings("uni-1", result_id="r-1")
            assert sorted((r.rater, r.rater_kind, r.score) for r in stored) == [
                ("owner", "agent", 1),
                ("owner", "person", 4),
            ], f"{first} then {second}: both documents stand"
            agreement = judge_agreement(stored, [_result("r-1", _tone(4))])
            (tone,) = agreement.dimensions
            assert (tone.n, tone.raters, tone.exact_agreement) == (1, ["owner"], 1.0), "only the person pairs"
            assert [(u.rater, u.reason) for u in agreement.unpaired] == [("owner", "rated_by_an_agent")]

    def test_rate_result_records_who_kind_of_rater_wrote_it(self) -> None:
        storage, _ = memory_storage()
        storage.save_eval_result(_result("r-1", _tone(3)))

        agent = rate_result(
            storage,
            result_id="r-1",
            scope_id="uni-1",
            rubric_dim=TONE,
            rater="agent:scout",
            rater_kind="agent",
            score=3,
            reason="matches the transcript",
        )
        person = rate_result(
            storage,
            result_id="r-1",
            scope_id="uni-1",
            rubric_dim=TONE,
            rater="owner",
            rater_kind="person",
            score=3,
            reason="fine",
        )

        stored = {rating.rater: rating.rater_kind for rating in storage.query_calibration_ratings("uni-1")}
        assert stored == {"agent:scout": "agent", "owner": "person"}
        assert (agent.rater_kind, person.rater_kind) == ("agent", "person")
        with pytest.raises(TypeError, match="rater_kind"):
            rate_result(  # type: ignore[call-arg]
                storage, result_id="r-1", scope_id="uni-1", rubric_dim=TONE, rater="owner", score=3, reason="fine"
            )

    def test_nobody_rated_anything_is_an_empty_reading(self) -> None:
        assert judge_agreement([], [_result("r-1", _tone(3))]) == JudgeAgreement()


# --- the readers -------------------------------------------------------------------------------------


class TestTheBundleReadsRatingsWrittenThroughTheOperation:
    """Done when: ratings written through the operation appear as agreement in a toy bundle."""

    def test_ratings_of_the_toy_campaign_appear_as_judge_agreement(self) -> None:
        campaign, storage = toyhost_campaign()
        profile = toyhost_profile()
        before = assemble_context_bundle(campaign, storage=storage, profile=profile)
        assert before.judge_agreement == JudgeAgreement(), "nothing rated yet: uncalibrated, not zero"

        narrow, wide = campaign.run_ids
        rated = [
            *storage.query_eval_results_by_run(narrow, TOYHOST_SCOPE)[:2],
            *storage.query_eval_results_by_run(wide, TOYHOST_SCOPE)[:2],
        ]
        judged = [result.judge_score(TOYHOST_JUDGED_DIMENSION) for result in rated]
        assert all(score is not None for score in judged), "the toy results must be judged, or the pairs are vacuous"
        # Three people agree with the judge exactly and one is a point off it.
        for index, (result, score) in enumerate(zip(rated, judged, strict=True)):
            assert score is not None
            person = score.score if index else (score.score - 1 if score.score > 1 else score.score + 1)
            rate_result(
                storage,
                result_id=result.id,
                scope_id=TOYHOST_SCOPE,
                rubric_dim=TOYHOST_JUDGED_DIMENSION,
                rater="reviewer-1" if index % 2 else "reviewer-2",
                rater_kind="person",
                score=person,
                reason="read the extracted record against the invoice",
            )

        after = assemble_context_bundle(campaign, storage=storage, profile=profile)

        (layout,) = after.judge_agreement.dimensions
        assert (layout.rubric_dim, layout.scale, layout.judge_model) == (TOYHOST_JUDGED_DIMENSION, "ordinal", None)
        assert (layout.n, layout.exact_agreement) == (4, 0.75)
        assert layout.raters == ["reviewer-1", "reviewer-2"]
        assert after.judge_agreement.unpaired == []
        assert after.fingerprint() != before.fingerprint(), "a new rating is evidence that moved"
        assert after.schema_version == 48


class TestTheReporterReadCarriesTheRunsRatings:
    """The reporter kind's human labels sit beside people's ratings of the same run, read the same way."""

    def test_reporter_calibration_reads_the_runs_ratings(self) -> None:
        storage, _ = memory_storage()
        case = ReporterCase(
            bundle={"campaign_id": "camp-1"},
            bundle_fingerprint="fp-1",
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
        )
        storage.save_test_case(
            EvalTestCase(id="c-1", scope_id="uni-1", template_id="tpl-1", host_payload=reporter_case_payload(case))
        )
        storage.save_eval_run(make_eval_run(id="run-r", test_case_ids=["c-1"], status="completed"))
        storage.save_eval_result(_result("r-1", _tone(4), eval_run_id="run-r", test_case_id="c-1"))
        storage.save_eval_result(_result("r-other", _tone(4), eval_run_id="run-other", test_case_id="c-1"))
        rate_result(
            storage,
            result_id="r-1",
            scope_id="uni-1",
            rubric_dim=TONE,
            rater="owner",
            rater_kind="person",
            score=5,
            reason="sharp",
        )
        rate_result(
            storage,
            result_id="r-other",
            scope_id="uni-1",
            rubric_dim=TONE,
            rater="owner",
            rater_kind="person",
            score=4,
            reason="x",
        )

        read = reporter_calibration(storage, "run-r", "uni-1")

        (tone,) = read.rating_agreement.dimensions
        assert (tone.rubric_dim, tone.n, tone.exact_agreement, tone.raters) == (TONE, 1, 0.0, ["owner"])
        assert read.rating_agreement.ratings_read == 1, "another run's ratings are not this run's"
