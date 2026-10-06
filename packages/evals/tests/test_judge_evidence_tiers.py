"""PD-13: a judged reading's evidence tier is decided by code from the judge's measured reliability.

The owner's ruling (2026-10-06): ``calibrated`` = judge–human weighted kappa at least 0.6 over at least 20
person-rated pairs (held here as 20 distinct results, which a pile of pairs about a few results cannot fake);
``separation`` = the judge's agreement with its own repeated scores at least 0.8;
``incidental`` = a judged reading meeting neither. Too little evidence is ``undetermined`` — a state, never
a tier quietly filled in.

Each threshold is driven on both sides of its boundary, through the pure derivation and through the
assembled bundle, the code-only report and the generator's resolution of a finding, since a tier shown
anywhere must be the one the evidence decides there.

Mutations that turn this file red (each made in a scratch copy and the file restored from it, 2026-10-06):

- ``criterion_state``: ``>`` for ``>=`` on the agreement, ``<=`` for ``<`` on the pair floor (both caught by
  the at-boundary ``met`` case);
- ``tier_of``: returning ``incidental`` where it returns ``undetermined``; checking separation before
  calibration;
- ``agreement_statistic``: reading ``kappa`` on a 1-5 scale;
- ``weakest_judged_tier``: taking the strongest;
- ``judge_self_agreement``: pairing a repeat served by another model;
- ``judge_evidence_tiers``: dropping the ``judged`` keys (an unmeasured judge is then absent);
- the bundle: stamping every arm ``undetermined``; not copying the arm's tier onto the cell's reading;
- the generator: not carrying the resolved tier onto the evidence row;
- the code-only report: dropping the tier sentences.

And, for the review fixes (each applied to a scratch-backed copy and restored from it, 2026-10-06):

- ``_pooled_kappa`` weighting every rater 1, or by pairs, instead of by result (the small-round, small-person and
  two-people cases, and the verify round's shapes: two results repeated thirty more times; five annotators on three
  shared anchors), and splitting a result's weight with a rater left out of the mean;
- the floor reading pairs instead of distinct results (the two-results-ten-times cases, here and end to end), and
  counting results behind an undefined kappa;
- a "can't tell" repeat set aside as unpaired (both scales, here and end to end);
- ``tier_for_judges`` keyed without the scale; ``judge_key`` dropping the config; a repeat under another config paired;
- the old ``separation`` words; ``EVAL_SCHEMA_VERSION`` left at 7.
"""

from __future__ import annotations

import json
import typing
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import (
    JudgeKey,
    assemble_context_bundle,
    judge_agreement,
    judge_evidence_tiers,
    judge_self_agreement,
    tier_for_judges,
    tier_sentence,
)
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT
from threetears.evals.analysis.stats import cohen_kappa
from threetears.evals.analysis.report import DisclosureBlock, build_code_only_report
from threetears.evals.contracts.evidence_tiers import (
    CALIBRATION_MIN_AGREEMENT,
    CALIBRATION_MIN_RESULTS,
    SEPARATION_MIN_AGREEMENT,
    SEPARATION_MIN_RESULTS,
    CriterionState,
    JudgedEvidenceTier,
    JudgeEvidenceTier,
    TierCriterion,
    agreement_statistic,
    calibration_criterion,
    separation_criterion,
    tier_of,
)
from threetears.evals.contracts.models import (
    EvalResult,
    JudgeRepeat,
    RepeatedScore,
    RubricScore,
    utc_now_iso,
)
from threetears.evals.run import rate_result
from packages.evals.tests.factories import make_calibration_rating, make_eval_result
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION, TOYHOST_SCOPE, ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, alias_at, memo_payload

TONE = "conversation.tone"


class TestTheRuledConstants:
    def test_the_constants_are_the_ruling(self) -> None:
        assert (CALIBRATION_MIN_AGREEMENT, CALIBRATION_MIN_RESULTS) == (0.6, 20)
        assert SEPARATION_MIN_AGREEMENT == 0.8
        # Not ruled: calibration's floor, so the two agreements are read over comparable evidence.
        assert SEPARATION_MIN_RESULTS == CALIBRATION_MIN_RESULTS


class TestEachCriterionFiresOnBothSidesOfItsBoundary:
    @pytest.mark.parametrize(
        ("n", "agreement", "state"),
        [
            (CALIBRATION_MIN_RESULTS, CALIBRATION_MIN_AGREEMENT, "met"),
            (CALIBRATION_MIN_RESULTS, 0.5999, "not_met"),
            (CALIBRATION_MIN_RESULTS - 1, 1.0, "insufficient"),
            (CALIBRATION_MIN_RESULTS, None, "insufficient"),
            (0, None, "insufficient"),
        ],
    )
    def test_calibration(self, n: int, agreement: float | None, state: CriterionState) -> None:
        assert calibration_criterion(n, n, agreement).state == state

    @pytest.mark.parametrize(
        ("n", "agreement", "state"),
        [
            (SEPARATION_MIN_RESULTS, SEPARATION_MIN_AGREEMENT, "met"),
            (SEPARATION_MIN_RESULTS, 0.7999, "not_met"),
            # Calibration's bar is not separation's: 0.6 meets one and misses the other.
            (SEPARATION_MIN_RESULTS, CALIBRATION_MIN_AGREEMENT, "not_met"),
            (SEPARATION_MIN_RESULTS - 1, 1.0, "insufficient"),
            (SEPARATION_MIN_RESULTS, None, "insufficient"),
        ],
    )
    def test_separation(self, n: int, agreement: float | None, state: CriterionState) -> None:
        assert separation_criterion(n, n, agreement).state == state

    def test_the_floor_counts_results_not_pairs(self) -> None:
        # Forty pairs about nineteen results is nineteen pieces of evidence.
        assert separation_criterion(40, SEPARATION_MIN_RESULTS - 1, 1.0).state == "insufficient"
        assert calibration_criterion(40, CALIBRATION_MIN_RESULTS - 1, 1.0).state == "insufficient"
        assert separation_criterion(40, SEPARATION_MIN_RESULTS, 1.0).state == "met"

    def test_more_results_than_pairs_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="cannot cover"):
            separation_criterion(19, 20, 1.0)

    def test_a_state_its_numbers_contradict_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="is 'insufficient', not 'met'"):
            TierCriterion(
                state="met",
                n=CALIBRATION_MIN_RESULTS,
                results=CALIBRATION_MIN_RESULTS - 1,
                agreement=0.9,
                threshold=CALIBRATION_MIN_AGREEMENT,
                min_results=CALIBRATION_MIN_RESULTS,
            )


def _criterion(state: CriterionState, *, separation: bool) -> TierCriterion:
    build = separation_criterion if separation else calibration_criterion
    return {
        "met": build(20, 20, 1.0),
        "not_met": build(20, 20, 0.0),
        "insufficient": build(3, 3, 1.0),
    }[state]


class TestTheTierRule:
    @pytest.mark.parametrize(
        ("calibration", "separation", "tier"),
        [
            ("met", "met", "calibrated"),
            ("met", "not_met", "calibrated"),
            ("met", "insufficient", "calibrated"),
            ("not_met", "met", "separation"),
            ("insufficient", "met", "separation"),
            ("not_met", "not_met", "incidental"),
            # Too little evidence on either side decides nothing: never filed as incidental.
            ("not_met", "insufficient", "undetermined"),
            ("insufficient", "not_met", "undetermined"),
            ("insufficient", "insufficient", "undetermined"),
        ],
    )
    def test_every_combination(
        self, calibration: CriterionState, separation: CriterionState, tier: JudgedEvidenceTier
    ) -> None:
        assert tier_of(_criterion(calibration, separation=False), _criterion(separation, separation=True)) == tier

    def test_every_tier_is_reachable(self) -> None:
        states: tuple[CriterionState, ...] = ("met", "not_met", "insufficient")
        reached = {
            tier_of(_criterion(c, separation=False), _criterion(s, separation=True)) for c in states for s in states
        }
        assert reached == set(typing.get_args(JudgedEvidenceTier))

    def test_a_tier_the_criteria_do_not_decide_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="decide 'undetermined', not 'incidental'"):
            JudgeEvidenceTier(
                rubric_dim=TONE,
                scale="ordinal",
                judge_model="j",
                judge_config_id=None,
                tier="incidental",
                calibration=calibration_criterion(0, 0, None),
                separation=separation_criterion(20, 20, 0.1),
            )

    def test_a_criterion_held_to_another_bar_is_refused(self) -> None:
        lowered = TierCriterion(state="met", n=20, results=20, agreement=0.5, threshold=0.5, min_results=20)
        with pytest.raises(ValidationError, match="CALIBRATION_MIN_AGREEMENT"):
            JudgeEvidenceTier(
                rubric_dim=TONE,
                scale="ordinal",
                judge_model="j",
                judge_config_id=None,
                tier="calibrated",
                calibration=lowered,
                separation=separation_criterion(0, 0, None),
            )

    def test_the_agreement_figure_is_weighted_on_one_to_five_and_kappa_on_pass_fail(self) -> None:
        assert agreement_statistic("ordinal", kappa=0.1, weighted_kappa=0.7) == 0.7
        assert agreement_statistic("pass_fail", kappa=0.1, weighted_kappa=None) == 0.1


# --- the two agreements, from results ------------------------------------------------------------


def _scored(result_id: str, score: int, judge: str | None = "judge-a", **extra: Any) -> EvalResult:
    return make_eval_result(
        id=result_id, rubric_scores=[RubricScore(dim=TONE, scale="ordinal", score=score, served_model=judge)], **extra
    )


def _repeat(
    first: int,
    again: int | None,
    *,
    judge: str | None = "judge-a",
    served: str | None = "judge-a",
    cannot_tell: bool = False,
    scale: str = "ordinal",
    first_config: str | None = None,
    repeat_config: str | None = None,
) -> JudgeRepeat:
    """One repeat of :data:`TONE`: a score, a failure (``again`` None), or a "can't tell" (``cannot_tell``)."""
    return JudgeRepeat(
        judge_model="judge-a",
        scores=[
            RepeatedScore(
                dim=TONE,
                scale=scale,
                first_score=first,
                first_served_model=judge,
                first_judge_config_id=first_config,
                repeat=None if again is None else RubricScore(dim=TONE, scale=scale, score=again, served_model=served),
                error="cut short" if again is None and not cannot_tell else None,
                cannot_tell="the evidence does not show it" if cannot_tell else None,
            )
        ],
        judge_config_ids={} if repeat_config is None else {TONE: repeat_config},
    )


def _key(scale: str = "ordinal", judge: str | None = "judge-a", config: str | None = None) -> JudgeKey:
    return JudgeKey(TONE, scale, judge, config)  # type: ignore[arg-type]


def _cycle(n: int) -> list[int]:
    """``n`` scores cycling over 2..5, so both raters use several categories and kappa is defined."""
    return [2 + index % 4 for index in range(n)]


class TestSelfAgreementIsReadAsCalibrationIs:
    def test_the_same_pairs_read_the_same_numbers_either_way(self) -> None:
        # One person rating exactly what the repeat answered: the two reads must agree to the last digit,
        # or the two tiers' thresholds would not be on one scale.
        firsts, others = _cycle(24), list(reversed(_cycle(24)))
        rated = [_scored(f"r-{i}", first) for i, first in enumerate(firsts)]
        ratings = [
            make_calibration_rating(result_id=f"r-{i}", rubric_dim=TONE, score=other) for i, other in enumerate(others)
        ]
        repeated = [
            _scored(f"r-{i}", first, judge_repeats=[_repeat(first, other)])
            for i, (first, other) in enumerate(zip(firsts, others, strict=True))
        ]
        (people,) = judge_agreement(ratings, rated).dimensions
        (itself,) = judge_self_agreement(repeated).dimensions
        assert (itself.n, itself.exact_agreement, itself.kappa, itself.weighted_kappa) == (
            people.n,
            people.exact_agreement,
            people.kappa,
            people.weighted_kappa,
        )

    def test_unpairable_repeats_are_named_never_dropped(self) -> None:
        results = [
            _scored("ok", 4, judge_repeats=[_repeat(4, 4)]),
            _scored("failed", 4, judge_repeats=[_repeat(4, None)]),
            _scored("moved", 4, judge_repeats=[_repeat(4, 4, served="judge-b")]),
        ]
        read = judge_self_agreement(results)
        assert read.repeats_read == 3
        assert [(u.result_id, u.reason) for u in read.unpaired] == [
            ("failed", "repeat_failed"),
            ("moved", "judge_changed"),
        ]
        (dimension,) = read.dimensions
        assert dimension.n == 1

    def test_a_second_repeat_is_its_own_rater(self) -> None:
        result = _scored("r", 4, judge_repeats=[_repeat(4, 4), _repeat(4, 3)])
        (dimension,) = judge_self_agreement([result]).dimensions
        assert dimension.rounds == ["repeat 1", "repeat 2"]


class TestTheTiersFromResults:
    @staticmethod
    def _tiers(*, rated: int = 0, repeated: int = 0, agree: bool = True, rater_kind: str = "person"):
        scores = _cycle(max(rated, repeated, 1))
        results, ratings = [], []
        for index, score in enumerate(scores):
            other = score if agree else 2 + (index + 2) % 4
            repeats = [_repeat(score, other)] if index < repeated else []
            results.append(_scored(f"r-{index}", score, judge_repeats=repeats))
            if index < rated:
                ratings.append(
                    make_calibration_rating(result_id=f"r-{index}", rubric_dim=TONE, score=other, rater_kind=rater_kind)
                )
        (tier,) = judge_evidence_tiers(judge_agreement(ratings, results), judge_self_agreement(results), {_key()})
        return tier

    def test_twenty_agreeing_person_ratings_calibrate(self) -> None:
        assert self._tiers(rated=CALIBRATION_MIN_RESULTS).tier == "calibrated"

    def test_nineteen_ratings_do_not(self) -> None:
        tier = self._tiers(rated=CALIBRATION_MIN_RESULTS - 1)
        assert tier.tier == "undetermined"
        assert tier.calibration.state == "insufficient"

    def test_an_agent_rating_never_calibrates(self) -> None:
        # The engine pairs only a person's rating with the judge, and the tier reads that pairing.
        tier = self._tiers(rated=CALIBRATION_MIN_RESULTS, rater_kind="agent")
        assert tier.tier == "undetermined"
        assert tier.calibration.n == 0

    def test_twenty_agreeing_repeats_separate(self) -> None:
        assert self._tiers(repeated=SEPARATION_MIN_RESULTS).tier == "separation"

    def test_nineteen_repeats_do_not(self) -> None:
        assert self._tiers(repeated=SEPARATION_MIN_RESULTS - 1).tier == "undetermined"

    def test_both_measured_and_both_missed_is_incidental(self) -> None:
        tier = self._tiers(rated=CALIBRATION_MIN_RESULTS, repeated=SEPARATION_MIN_RESULTS, agree=False)
        assert (tier.calibration.state, tier.separation.state, tier.tier) == ("not_met", "not_met", "incidental")

    def test_a_judge_nothing_measured_is_listed_undetermined(self) -> None:
        (tier,) = judge_evidence_tiers(judge_agreement([], []), judge_self_agreement([]), {_key()})
        assert (tier.tier, tier.calibration.n, tier.separation.n) == ("undetermined", 0, 0)

    def test_a_reading_served_by_two_judges_bears_the_weaker(self) -> None:
        strong = self._tiers(rated=CALIBRATION_MIN_RESULTS)
        assert tier_for_judges([strong], [_key()]) == "calibrated"
        assert tier_for_judges([strong], [_key(), _key(judge="judge-unmeasured")]) == "undetermined"
        assert tier_for_judges([strong], []) == "undetermined"


# --- the shared rule: results, not pairs; raters pooled by their pairs (B1) -------------------------


def _shifted(count: int, disagreeing: int, shift: int) -> list[tuple[int, int]]:
    """``count`` (first, other) pairs cycling over 2..5, the first ``disagreeing`` of them moved ``shift`` along."""
    firsts = _cycle(count)
    return [(first, first if i >= disagreeing else 2 + (i + shift) % 4) for i, first in enumerate(firsts)]


class TestAFewResultsCannotEarnATier:
    def test_two_results_repeated_ten_times_do_not_separate(self) -> None:
        # The reviewer's end-to-end shape: twenty agreeing pairs about two results.
        results = [
            _scored("a", 2, judge_repeats=[_repeat(2, 2) for _ in range(10)]),
            _scored("b", 4, judge_repeats=[_repeat(4, 4) for _ in range(10)]),
        ]
        (itself,) = judge_self_agreement(results).dimensions
        assert (itself.n, itself.results, itself.weighted_kappa) == (20, 2, 1.0)
        (tier,) = judge_evidence_tiers(judge_agreement([], results), judge_self_agreement(results), {_key()})
        assert (tier.separation.state, tier.tier) == ("insufficient", "undetermined")

    def test_a_small_round_cannot_carry_a_large_one_over_the_bar(self) -> None:
        pairs = _shifted(20, disagreeing=4, shift=2)
        results = [
            _scored(f"r-{i}", first, judge_repeats=[_repeat(first, again)]) for i, (first, again) in enumerate(pairs)
        ]
        # A second round on two of them, agreeing perfectly in two categories.
        results[0] = _scored("r-0", 2, judge_repeats=[_repeat(2, pairs[0][1]), _repeat(2, 2)])
        results[3] = _scored("r-3", 5, judge_repeats=[_repeat(5, pairs[3][1]), _repeat(5, 5)])
        first_round = cohen_kappa(pairs, [1, 2, 3, 4, 5], weights="quadratic")
        assert first_round is not None and SEPARATION_MIN_AGREEMENT <= (first_round + 1) / 2, (
            "an unweighted mean of the two rounds would meet the bar — the fixture must make that so"
        )
        (itself,) = judge_self_agreement(results).dimensions
        # By result: the two re-repeated results weigh 1 each, split between the rounds — round 1 carries 19.
        assert itself.weighted_kappa == pytest.approx((19 * first_round + 1 * 1.0) / 20)
        (tier,) = judge_evidence_tiers(judge_agreement([], results), judge_self_agreement(results), {_key()})
        assert (tier.separation.state, tier.tier) == ("not_met", "undetermined")

    def test_a_small_person_cannot_carry_a_large_one_over_the_calibration_bar(self) -> None:
        # Twenty ratings at about 0.3 and two at 1.0: an unweighted mean reads 0.64 and "met" the bar.
        big = _shifted(20, disagreeing=12, shift=1)
        small = [(1, 1), (5, 5)]
        results = [_scored(f"big-{i}", first) for i, (first, _) in enumerate(big)]
        results += [_scored(f"small-{i}", first) for i, (first, _) in enumerate(small)]
        ratings = [
            *(
                make_calibration_rating(result_id=f"big-{i}", rubric_dim=TONE, score=o, rater="big")
                for i, (_, o) in enumerate(big)
            ),
            *(
                make_calibration_rating(result_id=f"small-{i}", rubric_dim=TONE, score=o, rater="small")
                for i, (_, o) in enumerate(small)
            ),
        ]
        big_kappa = cohen_kappa(big, [1, 2, 3, 4, 5], weights="quadratic")
        assert big_kappa is not None and big_kappa < 0.3 and (big_kappa + 1) / 2 >= CALIBRATION_MIN_AGREEMENT
        (people,) = judge_agreement(ratings, results).dimensions
        assert (people.n, people.results) == (22, 22)
        assert people.weighted_kappa == pytest.approx((20 * big_kappa + 2 * 1.0) / 22)
        (tier,) = judge_evidence_tiers(judge_agreement(ratings, results), judge_self_agreement([]), {_key()})
        assert (tier.calibration.state, tier.tier) == ("not_met", "undetermined")

    def test_results_behind_an_undefined_kappa_do_not_count_toward_the_floor(self) -> None:
        # Twenty results one person rated 3 where the judge said 3 (kappa undefined), and two another person
        # matched in two categories: the figure rests on the two, so the floor sees two.
        results = [_scored(f"flat-{i}", 3) for i in range(20)] + [_scored("lo", 1), _scored("hi", 5)]
        ratings = [
            make_calibration_rating(result_id=f"flat-{i}", rubric_dim=TONE, score=3, rater="flat") for i in range(20)
        ]
        ratings += [
            make_calibration_rating(result_id="lo", rubric_dim=TONE, score=1, rater="pair"),
            make_calibration_rating(result_id="hi", rubric_dim=TONE, score=5, rater="pair"),
        ]
        (people,) = judge_agreement(ratings, results).dimensions
        assert (people.n, people.results, people.weighted_kappa) == (22, 2, 1.0)
        (tier,) = judge_evidence_tiers(judge_agreement(ratings, results), judge_self_agreement([]), {_key()})
        assert (tier.calibration.state, tier.tier) == ("insufficient", "undetermined")


class TestEachResultWeighsOnceInTheFigure:
    """The figure weighs what the floor counts: each distinct result carries weight 1, split across its measurers.

    Pair-weighting closed a small rater outvoting a large one and left many small raters, or many rounds over a
    few results, free to carry the figure: the floor saw twenty results while the kappa saw mostly two or three.
    """

    @staticmethod
    def _first_round() -> list[tuple[int, int]]:
        # Twenty results at a weighted kappa well under either bar.
        return _shifted(20, disagreeing=10, shift=2)

    def test_two_results_repeated_thirty_more_times_do_not_carry_a_round_to_separation(self) -> None:
        pairs = self._first_round()
        first_round = cohen_kappa(pairs, [1, 2, 3, 4, 5], weights="quadratic")
        assert first_round is not None and first_round < 0.5
        results = []
        for i, (first, again) in enumerate(pairs):
            repeats = [_repeat(first, again)]
            if i in (0, 2):  # two results in different categories, repeated thirty more times, agreeing
                repeats += [_repeat(first, first) for _ in range(30)]
            results.append(_scored(f"r-{i}", first, judge_repeats=repeats))
        (itself,) = judge_self_agreement(results).dimensions
        assert (itself.n, itself.results) == (80, 20)
        (tier,) = judge_evidence_tiers(judge_agreement([], results), judge_self_agreement(results), {_key()})
        assert itself.weighted_kappa is not None and itself.weighted_kappa < 0.6, (
            "the many re-measurements weigh two results"
        )
        assert (tier.separation.state, tier.tier) == ("not_met", "undetermined")

    def test_five_annotators_on_three_shared_anchors_do_not_calibrate_a_judge_one_person_found_wanting(self) -> None:
        # Alice rates twenty results at about 0.44: every other one two points off the judge.
        firsts = [1, 2, 3, 4, 5] * 4
        pairs = [(f, f if i % 2 else (f + 2 if f <= 3 else f - 2)) for i, f in enumerate(firsts)]
        results = [_scored(f"r-{i}", first) for i, (first, _) in enumerate(pairs)]
        ratings = [
            make_calibration_rating(result_id=f"r-{i}", rubric_dim=TONE, score=other, rater="alice")
            for i, (_, other) in enumerate(pairs)
        ]
        anchors = [1, 3, 4]  # three of alice's results, each in a different category
        assert len({pairs[i][0] for i in anchors}) == 3
        for annotator in range(5):
            ratings += [
                make_calibration_rating(
                    result_id=f"r-{i}", rubric_dim=TONE, score=pairs[i][0], rater=f"annotator-{annotator}"
                )
                for i in anchors
            ]
        (people,) = judge_agreement(ratings, results).dimensions
        assert (people.n, people.results) == (35, 20)
        alice = cohen_kappa(pairs, [1, 2, 3, 4, 5], weights="quadratic")
        assert alice is not None and alice < CALIBRATION_MIN_AGREEMENT
        assert (20 * alice + 15 * 1.0) / 35 >= CALIBRATION_MIN_AGREEMENT, (
            "weighted by pairs the annotators would carry it over the bar — the fixture must make that so"
        )
        # Each result weighs 1: alice carries 17 + 3/6, the five annotators 3/6 each.
        assert people.weighted_kappa == pytest.approx((17.5 * alice + 5 * 0.5 * 1.0) / 20)
        (tier,) = judge_evidence_tiers(judge_agreement(ratings, results), judge_self_agreement([]), {_key()})
        assert (tier.calibration.state, tier.tier) == ("not_met", "undetermined")

    def test_a_rater_left_out_of_the_mean_takes_no_share_of_a_results_weight(self) -> None:
        # Someone who rated alice's five 3s as 3 has an undefined kappa and is left out; those five results stay
        # wholly alice's, so alice weighs her twenty and bob his two.
        pairs = _shifted(20, disagreeing=8, shift=1)
        results = [_scored(f"r-{i}", first) for i, (first, _) in enumerate(pairs)] + [
            _scored("lo", 1),
            _scored("hi", 5),
        ]
        ratings = [
            make_calibration_rating(result_id=f"r-{i}", rubric_dim=TONE, score=other, rater="alice")
            for i, (_, other) in enumerate(pairs)
        ]
        threes = [i for i, (first, _) in enumerate(pairs) if first == 3]
        assert len(threes) == 5
        ratings += [make_calibration_rating(result_id=f"r-{i}", rubric_dim=TONE, score=3, rater="flat") for i in threes]
        ratings += [
            make_calibration_rating(result_id="lo", rubric_dim=TONE, score=1, rater="bob"),
            make_calibration_rating(result_id="hi", rubric_dim=TONE, score=5, rater="bob"),
        ]
        alice = cohen_kappa(pairs, [1, 2, 3, 4, 5], weights="quadratic")
        assert alice is not None and alice < 1.0
        (people,) = judge_agreement(ratings, results).dimensions
        assert (people.n, people.results) == (27, 22)
        assert people.weighted_kappa == pytest.approx((20 * alice + 2 * 1.0) / 22)


# --- a "can't tell" repeat is the judge disagreeing with itself (W1) ---------------------------------


class TestACannotTellRepeatCountsAgainstTheJudge:
    @pytest.mark.parametrize("scale", ["ordinal", "pass_fail"])
    def test_a_judge_declining_a_third_of_its_repeats_does_not_separate(self, scale: str) -> None:
        firsts = _cycle(30) if scale == "ordinal" else [index % 2 for index in range(30)]
        results = []
        for index, first in enumerate(firsts):
            declined = index < 10
            score = RubricScore(dim=TONE, scale=scale, score=first, served_model="judge-a")
            results.append(
                make_eval_result(
                    id=f"r-{index}",
                    rubric_scores=[score],
                    judge_repeats=[_repeat(first, None if declined else first, cannot_tell=declined, scale=scale)],
                )
            )
        read = judge_self_agreement(results)
        assert read.unpaired == [], "a declined repeat is a pair, never set aside"
        (itself,) = read.dimensions
        assert (itself.n, itself.results, itself.n_cannot_tell) == (30, 30, 10)
        assert itself.exact_agreement == pytest.approx(20 / 30)
        (tier,) = judge_evidence_tiers(judge_agreement([], results), read, {_key(scale)})
        assert (tier.separation.state, tier.tier) == ("not_met", "undetermined")

    @pytest.mark.parametrize("first", [1, 3, 5])
    def test_cannot_tell_costs_the_most_a_disagreement_can_wherever_the_first_score_sat(self, first: int) -> None:
        # Off the scale: a declined repeat of a 3 costs as much as one of a 1 or a 5 — the full cost of 1,
        # which no pair of scores reaches except 1 against 5.
        pairs = [(1, 1), (5, 5), (first, -1)]
        kappa = cohen_kappa(pairs, [1, 2, 3, 4, 5], weights="quadratic", unordered=[-1])
        assert kappa == pytest.approx(1 - (1 / 3) / _expected_disagreement(pairs))


def _expected_disagreement(pairs: list[tuple[int, int]]) -> float:
    """Chance disagreement of ``pairs`` on 1-5, quadratic, a "can't tell" (-1) at cost 1 from everything — by hand."""
    n = len(pairs)
    firsts = [a for a, _ in pairs]
    seconds = [b for _, b in pairs]
    categories = sorted(set(firsts) | set(seconds))

    def cost(i: int, j: int) -> float:
        if i == j:
            return 0.0
        if -1 in (i, j):
            return 1.0
        return (i - j) ** 2 / 16

    return sum(firsts.count(i) * seconds.count(j) * cost(i, j) for i in categories for j in categories) / (n * n)


# --- a tier is the very judge's: scale and config are part of who judged (W3, W7) ---------------------


class TestATierBelongsToTheWholeJudge:
    def test_a_reading_on_one_scale_never_carries_the_tier_measured_on_another(self) -> None:
        # The pass/fail judge is separated; the ordinal readings of the same dimension and model are not.
        pass_fail = [
            make_eval_result(
                id=f"pf-{i}",
                rubric_scores=[RubricScore(dim=TONE, scale="pass_fail", score=i % 2, served_model="judge-a")],
                judge_repeats=[_repeat(i % 2, i % 2, scale="pass_fail")],
            )
            for i in range(SEPARATION_MIN_RESULTS)
        ]
        ordinal = [_scored(f"o-{i}", score) for i, score in enumerate(_cycle(5))]
        results = pass_fail + ordinal
        tiers = judge_evidence_tiers(
            judge_agreement([], results), judge_self_agreement(results), {_key(), _key("pass_fail")}
        )
        assert {(t.scale, t.tier) for t in tiers} == {("ordinal", "undetermined"), ("pass_fail", "separation")}
        assert tier_for_judges(tiers, [_key()]) == "undetermined"
        assert tier_for_judges(tiers, [_key("pass_fail")]) == "separation"

    def test_a_repeat_under_one_config_never_tiers_readings_under_another(self) -> None:
        under_v1 = [
            _scored(
                f"v1-{i}",
                s,
                judge_config_ids={TONE: "cfg-v1"},
                judge_repeats=[_repeat(s, s, first_config="cfg-v1", repeat_config="cfg-v1")],
            )
            for i, s in enumerate(_cycle(SEPARATION_MIN_RESULTS))
        ]
        under_v2 = [_scored(f"v2-{i}", s, judge_config_ids={TONE: "cfg-v2"}) for i, s in enumerate(_cycle(5))]
        results = under_v1 + under_v2
        from threetears.evals.analysis import judge_key

        judged = {judge_key(result, TONE) for result in results}
        assert judged == {_key(config="cfg-v1"), _key(config="cfg-v2")}
        tiers = judge_evidence_tiers(judge_agreement([], results), judge_self_agreement(results), judged)  # type: ignore[arg-type]
        assert tier_for_judges(tiers, [_key(config="cfg-v1")]) == "separation"
        assert tier_for_judges(tiers, [_key(config="cfg-v2")]) == "undetermined"

    def test_a_repeat_answered_under_another_config_is_not_paired(self) -> None:
        result = _scored(
            "r",
            4,
            judge_config_ids={TONE: "cfg-v1"},
            judge_repeats=[_repeat(4, 4, first_config="cfg-v1", repeat_config="cfg-v2")],
        )
        read = judge_self_agreement([result])
        assert [(u.result_id, u.reason) for u in read.unpaired] == [("r", "config_changed")]
        assert read.dimensions == []

    def test_the_tier_sentence_names_the_config(self) -> None:
        (tier,) = judge_evidence_tiers(judge_agreement([], []), judge_self_agreement([]), {_key(config="cfg-v1")})
        assert tier_sentence(tier).startswith(f"{TONE} (judge-a, config cfg-v1): undetermined")


# --- where the tier is shown ---------------------------------------------------------------------


def _repeated(result: EvalResult, *, agree: bool) -> EvalResult:
    """``result`` with one repeat of the toy judged dimension recorded on it — agreeing, or a long way off."""
    score = result.judge_score(TOYHOST_JUDGED_DIMENSION)
    assert score is not None, "the toy results must be judged, or the repeat is vacuous"
    again = score.score if agree else (1 if score.score >= 3 else 5)
    config = result.judge_config_ids.get(TOYHOST_JUDGED_DIMENSION)
    repeat = JudgeRepeat(
        judge_model="toy-judge",
        judge_config_ids={} if config is None else {TOYHOST_JUDGED_DIMENSION: config},
        scores=[
            RepeatedScore(
                dim=TOYHOST_JUDGED_DIMENSION,
                scale=score.scale,
                first_score=score.score,
                first_served_model=score.served_model,
                first_judge_config_id=config,
                repeat=RubricScore(
                    dim=TOYHOST_JUDGED_DIMENSION, scale=score.scale, score=again, served_model=score.served_model
                ),
            )
        ],
    )
    return result.model_copy(update={"judge_repeats": [repeat]})


def _bundle(*, repeats: int = 0, rated: int = 0, agree: bool = True, two_days: bool = False):
    """The toy campaign's bundle, with ``repeats`` of its results repeated and ``rated`` of them rated by a person.

    ``two_days`` starts the second arm a day after the first, which gives the bundle a time axis.
    """
    campaign, toy = toyhost_campaign()
    runs = toy.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
    if two_days:
        runs = [
            run.model_copy(update={"created_at": f"2026-03-1{4 + index}T09:30:00+00:00"})
            for index, run in enumerate(runs)
        ]
    ordered = [(run.id, result) for run in runs for result in toy.query_eval_results_by_run(run.id, TOYHOST_SCOPE)]
    assert len(ordered) >= max(repeats, rated), "the toy campaign must hold enough judged results"
    results_by_run: dict[str, list[EvalResult]] = {run.id: [] for run in runs}
    for index, (run_id, result) in enumerate(ordered):
        results_by_run[run_id].append(_repeated(result, agree=agree) if index < repeats else result)
    storage = ToyhostStorage(runs, results_by_run)
    for _, result in ordered[:rated]:
        score = result.judge_score(TOYHOST_JUDGED_DIMENSION)
        assert score is not None
        rate_result(
            storage,
            result_id=result.id,
            scope_id=TOYHOST_SCOPE,
            rubric_dim=TOYHOST_JUDGED_DIMENSION,
            rater="reviewer-1",
            rater_kind="person",
            score=score.score,
            reason="read it against the source",
        )
    return assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())


def _shown_tiers(bundle: Any) -> set[str]:
    arm_tiers = {arm.evidence_tier for measure in bundle.judged_measures for arm in measure.arms}
    cell_tiers = {reading.evidence_tier for cell in bundle.cell_measures for reading in cell.judged}
    assert arm_tiers == cell_tiers, "a cell's judged reading and its arm are one transposition"
    return arm_tiers


class TestTheBundleDerivesTheTierWhereItShowsIt:
    def test_an_unmeasured_judge_is_undetermined_everywhere(self) -> None:
        bundle = _bundle()
        (tier,) = bundle.judge_evidence_tiers
        assert (tier.rubric_dim, tier.tier) == (TOYHOST_JUDGED_DIMENSION, "undetermined")
        assert _shown_tiers(bundle) == {"undetermined"}

    def test_twenty_agreeing_repeats_put_every_reading_on_separation(self) -> None:
        bundle = _bundle(repeats=SEPARATION_MIN_RESULTS)
        (tier,) = bundle.judge_evidence_tiers
        assert (tier.separation.n, tier.tier) == (SEPARATION_MIN_RESULTS, "separation")
        assert bundle.judge_self_agreement.dimensions[0].n == SEPARATION_MIN_RESULTS
        assert _shown_tiers(bundle) == {"separation"}

    def test_nineteen_repeats_leave_them_undetermined(self) -> None:
        bundle = _bundle(repeats=SEPARATION_MIN_RESULTS - 1)
        assert _shown_tiers(bundle) == {"undetermined"}

    def test_twenty_agreeing_ratings_calibrate_them(self) -> None:
        assert _shown_tiers(_bundle(rated=CALIBRATION_MIN_RESULTS)) == {"calibrated"}

    def test_the_time_axis_cells_carry_the_campaigns_tier(self) -> None:
        bundle = _bundle(repeats=SEPARATION_MIN_RESULTS, two_days=True)
        assert bundle.time_axis is not None, "two days of runs give the bundle a time axis"
        tiers = {r.evidence_tier for p in bundle.time_axis.positions for c in p.cells for r in c.judged}
        assert tiers == {"separation"}

    def test_a_repeat_moves_the_fingerprint(self) -> None:
        assert _bundle().fingerprint() != _bundle(repeats=1).fingerprint()


class TestTheReportStatesEachTier:
    def test_the_code_only_report_states_the_tier_and_its_measurements(self) -> None:
        report = build_code_only_report(
            _bundle(repeats=SEPARATION_MIN_RESULTS),
            measures=toyhost_profile().measures,
            assembled_at="2026-10-06T00:00:00+00:00",
        )
        texts = [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]
        sentences = [text for text in texts if text.startswith("Judged evidence tier:")]
        assert sentences == [
            f"Judged evidence tier: {TOYHOST_JUDGED_DIMENSION} (an unnamed judge): separation — agreement with people "
            "not measured (bar 0.6 over at least 20 results); with its own repeats 1 over 20 pairs from 20 results, "
            "meets (bar 0.8 over at least 20 results)."
        ]


class TestTheWordsClaimOnlyWhatWasMeasured:
    def test_separation_never_says_the_judge_went_unchecked_against_people(self) -> None:
        # `separation` is also the tier of a judge checked against people over enough results and found wanting
        # (calibration `not_met`), so "not checked" would be false there — and kinder than the truth.
        from threetears.evals.analysis.report.words import EVIDENCE_TIER_WORDS

        words = EVIDENCE_TIER_WORDS["separation"]
        assert "not checked" not in words
        assert "not shown to agree with people" in words
        tier = JudgeEvidenceTier(
            rubric_dim=TONE,
            scale="ordinal",
            judge_model="j",
            judge_config_id=None,
            tier="separation",
            calibration=calibration_criterion(20, 20, 0.1),
            separation=separation_criterion(20, 20, 0.9),
        )
        assert tier.calibration.state == "not_met", "the case the words must stay true for"


class TestTheStoredShapeMovedTheSchemaVersion:
    def test_the_tier_fields_are_v8(self) -> None:
        # Judged rows and readings gained a required tier, `directional` left `EvidenceTier`, and a repeated
        # score records its first config — each a stored shape change, so a document written before it is v7.
        from threetears.evals.contracts import models

        assert models.EVAL_SCHEMA_VERSION == 8
        assert "**v8**" in _schema_version_doc(), "a bump says what changed, as v7 did"
        assert RepeatedScore.model_fields["first_judge_config_id"].is_required()


def _schema_version_doc() -> str:
    """The text documenting the schema versions, read from the module source beside the constant."""
    import inspect

    from threetears.evals.contracts import models

    source = inspect.getsource(models)
    start = source.index("EVAL_SCHEMA_VERSION: int")
    return source[start : source.index('"""', source.index('"""', start) + 3)]


class TestThePromptPointsAtTheTiers:
    def test_the_generator_is_told_where_the_tiers_are_and_what_each_bears(self) -> None:
        for name in ("judge_evidence_tiers", "judge_self_agreement", "judge_agreement", "evidence_tier"):
            assert f"`{name}`" in EVAL_ANALYSIS_GEN_DEFAULT
        for tier in typing.get_args(JudgedEvidenceTier):
            assert f"`{tier}`" in EVAL_ANALYSIS_GEN_DEFAULT
        assert "stands only on a `calibrated` reading" in EVAL_ANALYSIS_GEN_DEFAULT


class TestAFindingStandsOnTheTierItsCellsCarry:
    @staticmethod
    async def _generated(bundle: Any):
        payload = memo_payload(bundle)
        payload["findings"][0]["evidence"].append(
            {"cell": alias_at(bundle, TOYHOST_NARROW), "measure_id": TOYHOST_JUDGED_DIMENSION, "reading": "judged"}
        )
        analysis, _ = await generate_analysis(
            bundle,
            prompt=PROMPT,
            model=MODEL,
            client=FixturedClient(json.dumps(payload)),
            prompt_id=PROMPT_ID,
            bundle_assembled_at=utc_now_iso(),
            profile=toyhost_profile(),
        )
        return analysis.resolutions[0]

    async def test_a_finding_citing_a_separated_judge_stands_on_separation(self) -> None:
        resolution = await self._generated(_bundle(repeats=SEPARATION_MIN_RESULTS))
        assert [row.judged_tier for row in resolution.evidence] == [None, None, "separation"]
        assert resolution.evidence_tier == "separation"

    async def test_the_same_finding_over_an_unmeasured_judge_is_undetermined(self) -> None:
        resolution = await self._generated(_bundle())
        assert resolution.evidence_tier == "undetermined"
