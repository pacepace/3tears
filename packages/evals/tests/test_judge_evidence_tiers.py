"""PD-13: a judged reading's evidence tier is decided by code from the judge's measured reliability.

The owner's ruling (2026-10-06): ``calibrated`` = judge–human weighted kappa at least 0.6 over at least 20
person-rated pairs; ``separation`` = the judge's agreement with its own repeated scores at least 0.8;
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
"""

from __future__ import annotations

import json
import typing
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import (
    assemble_context_bundle,
    judge_agreement,
    judge_evidence_tiers,
    judge_self_agreement,
    tier_for_judges,
)
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT
from threetears.evals.analysis.report import DisclosureBlock, build_code_only_report
from threetears.evals.contracts.evidence_tiers import (
    CALIBRATION_MIN_AGREEMENT,
    CALIBRATION_MIN_PAIRS,
    SEPARATION_MIN_AGREEMENT,
    SEPARATION_MIN_PAIRS,
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
        assert (CALIBRATION_MIN_AGREEMENT, CALIBRATION_MIN_PAIRS) == (0.6, 20)
        assert SEPARATION_MIN_AGREEMENT == 0.8
        # Not ruled: calibration's floor, so the two agreements are read over comparable evidence.
        assert SEPARATION_MIN_PAIRS == CALIBRATION_MIN_PAIRS


class TestEachCriterionFiresOnBothSidesOfItsBoundary:
    @pytest.mark.parametrize(
        ("n", "agreement", "state"),
        [
            (CALIBRATION_MIN_PAIRS, CALIBRATION_MIN_AGREEMENT, "met"),
            (CALIBRATION_MIN_PAIRS, 0.5999, "not_met"),
            (CALIBRATION_MIN_PAIRS - 1, 1.0, "insufficient"),
            (CALIBRATION_MIN_PAIRS, None, "insufficient"),
            (0, None, "insufficient"),
        ],
    )
    def test_calibration(self, n: int, agreement: float | None, state: CriterionState) -> None:
        assert calibration_criterion(n, agreement).state == state

    @pytest.mark.parametrize(
        ("n", "agreement", "state"),
        [
            (SEPARATION_MIN_PAIRS, SEPARATION_MIN_AGREEMENT, "met"),
            (SEPARATION_MIN_PAIRS, 0.7999, "not_met"),
            # Calibration's bar is not separation's: 0.6 meets one and misses the other.
            (SEPARATION_MIN_PAIRS, CALIBRATION_MIN_AGREEMENT, "not_met"),
            (SEPARATION_MIN_PAIRS - 1, 1.0, "insufficient"),
            (SEPARATION_MIN_PAIRS, None, "insufficient"),
        ],
    )
    def test_separation(self, n: int, agreement: float | None, state: CriterionState) -> None:
        assert separation_criterion(n, agreement).state == state

    def test_a_state_its_numbers_contradict_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="is 'insufficient', not 'met'"):
            TierCriterion(
                state="met",
                n=CALIBRATION_MIN_PAIRS - 1,
                agreement=0.9,
                threshold=CALIBRATION_MIN_AGREEMENT,
                min_pairs=CALIBRATION_MIN_PAIRS,
            )


def _criterion(state: CriterionState, *, separation: bool) -> TierCriterion:
    build = separation_criterion if separation else calibration_criterion
    return {
        "met": build(20, 1.0),
        "not_met": build(20, 0.0),
        "insufficient": build(3, 1.0),
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
                tier="incidental",
                calibration=calibration_criterion(0, None),
                separation=separation_criterion(20, 0.1),
            )

    def test_a_criterion_held_to_another_bar_is_refused(self) -> None:
        lowered = TierCriterion(state="met", n=20, agreement=0.5, threshold=0.5, min_pairs=20)
        with pytest.raises(ValidationError, match="CALIBRATION_MIN_AGREEMENT"):
            JudgeEvidenceTier(
                rubric_dim=TONE,
                scale="ordinal",
                judge_model="j",
                tier="calibrated",
                calibration=lowered,
                separation=separation_criterion(0, None),
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
    first: int, again: int | None, *, judge: str | None = "judge-a", served: str | None = "judge-a"
) -> JudgeRepeat:
    return JudgeRepeat(
        judge_model="judge-a",
        scores=[
            RepeatedScore(
                dim=TONE,
                scale="ordinal",
                first_score=first,
                first_served_model=judge,
                repeat=None
                if again is None
                else RubricScore(dim=TONE, scale="ordinal", score=again, served_model=served),
                error="cut short" if again is None else None,
            )
        ],
    )


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
        judged = {(TONE, "ordinal", "judge-a")}
        (tier,) = judge_evidence_tiers(judge_agreement(ratings, results), judge_self_agreement(results), judged)
        return tier

    def test_twenty_agreeing_person_ratings_calibrate(self) -> None:
        assert self._tiers(rated=CALIBRATION_MIN_PAIRS).tier == "calibrated"

    def test_nineteen_ratings_do_not(self) -> None:
        tier = self._tiers(rated=CALIBRATION_MIN_PAIRS - 1)
        assert tier.tier == "undetermined"
        assert tier.calibration.state == "insufficient"

    def test_an_agent_rating_never_calibrates(self) -> None:
        # The engine pairs only a person's rating with the judge, and the tier reads that pairing.
        tier = self._tiers(rated=CALIBRATION_MIN_PAIRS, rater_kind="agent")
        assert tier.tier == "undetermined"
        assert tier.calibration.n == 0

    def test_twenty_agreeing_repeats_separate(self) -> None:
        assert self._tiers(repeated=SEPARATION_MIN_PAIRS).tier == "separation"

    def test_nineteen_repeats_do_not(self) -> None:
        assert self._tiers(repeated=SEPARATION_MIN_PAIRS - 1).tier == "undetermined"

    def test_both_measured_and_both_missed_is_incidental(self) -> None:
        tier = self._tiers(rated=CALIBRATION_MIN_PAIRS, repeated=SEPARATION_MIN_PAIRS, agree=False)
        assert (tier.calibration.state, tier.separation.state, tier.tier) == ("not_met", "not_met", "incidental")

    def test_a_judge_nothing_measured_is_listed_undetermined(self) -> None:
        (tier,) = judge_evidence_tiers(
            judge_agreement([], []), judge_self_agreement([]), {(TONE, "ordinal", "judge-a")}
        )
        assert (tier.tier, tier.calibration.n, tier.separation.n) == ("undetermined", 0, 0)

    def test_a_reading_served_by_two_judges_bears_the_weaker(self) -> None:
        strong = self._tiers(rated=CALIBRATION_MIN_PAIRS)
        assert tier_for_judges([strong], TONE, ["judge-a"]) == "calibrated"
        assert tier_for_judges([strong], TONE, ["judge-a", "judge-unmeasured"]) == "undetermined"
        assert tier_for_judges([strong], TONE, []) == "undetermined"


# --- where the tier is shown ---------------------------------------------------------------------


def _repeated(result: EvalResult, *, agree: bool) -> EvalResult:
    """``result`` with one repeat of the toy judged dimension recorded on it — agreeing, or a long way off."""
    score = result.judge_score(TOYHOST_JUDGED_DIMENSION)
    assert score is not None, "the toy results must be judged, or the repeat is vacuous"
    again = score.score if agree else (1 if score.score >= 3 else 5)
    repeat = JudgeRepeat(
        judge_model="toy-judge",
        scores=[
            RepeatedScore(
                dim=TOYHOST_JUDGED_DIMENSION,
                scale=score.scale,
                first_score=score.score,
                first_served_model=score.served_model,
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
        bundle = _bundle(repeats=SEPARATION_MIN_PAIRS)
        (tier,) = bundle.judge_evidence_tiers
        assert (tier.separation.n, tier.tier) == (SEPARATION_MIN_PAIRS, "separation")
        assert bundle.judge_self_agreement.dimensions[0].n == SEPARATION_MIN_PAIRS
        assert _shown_tiers(bundle) == {"separation"}

    def test_nineteen_repeats_leave_them_undetermined(self) -> None:
        bundle = _bundle(repeats=SEPARATION_MIN_PAIRS - 1)
        assert _shown_tiers(bundle) == {"undetermined"}

    def test_twenty_agreeing_ratings_calibrate_them(self) -> None:
        assert _shown_tiers(_bundle(rated=CALIBRATION_MIN_PAIRS)) == {"calibrated"}

    def test_the_time_axis_cells_carry_the_campaigns_tier(self) -> None:
        bundle = _bundle(repeats=SEPARATION_MIN_PAIRS, two_days=True)
        assert bundle.time_axis is not None, "two days of runs give the bundle a time axis"
        tiers = {r.evidence_tier for p in bundle.time_axis.positions for c in p.cells for r in c.judged}
        assert tiers == {"separation"}

    def test_a_repeat_moves_the_fingerprint(self) -> None:
        assert _bundle().fingerprint() != _bundle(repeats=1).fingerprint()


class TestTheReportStatesEachTier:
    def test_the_code_only_report_states_the_tier_and_its_measurements(self) -> None:
        report = build_code_only_report(
            _bundle(repeats=SEPARATION_MIN_PAIRS),
            measures=toyhost_profile().measures,
            assembled_at="2026-10-06T00:00:00+00:00",
        )
        texts = [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]
        sentences = [text for text in texts if text.startswith("Judged evidence tier:")]
        assert sentences == [
            f"Judged evidence tier: {TOYHOST_JUDGED_DIMENSION} (an unnamed judge): separation — agreement with people "
            "not measured (bar 0.6 over at least 20 pairs); with its own repeats 1 over 20 pairs, meets "
            "(bar 0.8 over at least 20 pairs)."
        ]


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
        resolution = await self._generated(_bundle(repeats=SEPARATION_MIN_PAIRS))
        assert [row.judged_tier for row in resolution.evidence] == [None, None, "separation"]
        assert resolution.evidence_tier == "separation"

    async def test_the_same_finding_over_an_unmeasured_judge_is_undetermined(self) -> None:
        resolution = await self._generated(_bundle())
        assert resolution.evidence_tier == "undetermined"
