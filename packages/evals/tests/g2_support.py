"""A G2-shaped report: the campaign the failed production memo was written over, as a fixture.

Two arms over 4 cases × 3 repeats, run through the quick path, whose bundle's census is then set to G2's: the
control lost 4 of its 12 trials to the test setup, the challenger 10 of its 12 — 7 to a judge whose reply was cut
off at the 10240-token cap it was sent, 3 to the test setup — and no question was declared. Its judge was new, with
no human labels yet, as every judge on a first campaign is. Shared by the golden test and the vocabulary test.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from threetears.evals.analysis import AnalysisContextBundle, Report, build_code_only_report
from threetears.evals.analysis.bundle.schema import ArmExclusions, ArmScoreboard, TrialRow
from threetears.evals.kernel.evidence_tiers import (
    CALIBRATION_MIN_AGREEMENT,
    CALIBRATION_MIN_RESULTS,
    SEPARATION_MIN_AGREEMENT,
    SEPARATION_MIN_RESULTS,
    JudgeEvidenceTier,
    TierCriterion,
)
from threetears.evals.quick import compare

__all__ = ["CASES", "JUDGE", "LOSSES", "g2_bundle", "g2_report"]


#: Four cases, each run three times on each arm.
CASES = [{"text": f"queue request {index}", "label": "queued"} for index in range(4)]

#: The trials each arm loses, by cause, in trial order: the control four to the test setup, the challenger seven to
#: a truncated judge and three to the test setup.
LOSSES = {"v4.1-flash": ["infra"] * 4, "v4-flash": ["truncated"] * 7 + ["infra"] * 3}


#: The campaign's judge: new, so nobody has labelled any of its outputs yet.
JUDGE = "judge-model-1"


def _unlabelled_judge() -> JudgeEvidenceTier:
    """The judge's standing on its one dimension: no human labels and no repeats, so no tier is decided."""
    return JudgeEvidenceTier(
        rubric_dim="conversation.queue_discipline",
        scale="ordinal",
        judge_model=JUDGE,
        judge_config_id=None,
        judge_temperature=0.0,
        tier="undetermined",
        calibration=TierCriterion(
            state="insufficient",
            n=0,
            results=0,
            agreement=None,
            threshold=CALIBRATION_MIN_AGREEMENT,
            min_results=CALIBRATION_MIN_RESULTS,
        ),
        separation=TierCriterion(
            state="insufficient",
            n=0,
            results=0,
            agreement=None,
            threshold=SEPARATION_MIN_AGREEMENT,
            min_results=SEPARATION_MIN_RESULTS,
        ),
    )


async def _queue(case: Mapping[str, Any]) -> str:
    """Queues every request."""
    return "queued"


def _census(variant: str, trials: list[TrialRow]) -> ArmScoreboard:
    """One arm's census, counted off its trials as the assembler counts it."""
    mine = [trial for trial in trials if trial.arm == variant]
    lost = Counter(trial.cause or "unknown" for trial in mine if trial.state == "lost")
    failed = Counter(trial.cause or "unknown" for trial in mine if trial.state == "failed")
    usable = [trial for trial in mine if trial.state != "lost"]
    return ArmScoreboard(
        variant_key=variant,
        ran=len(mine),
        lost=dict(lost),
        usable=len(usable),
        passed=sum(1 for trial in usable if trial.state == "passed"),
        failed=dict(failed),
        cases=len({trial.case_id for trial in mine}),
        usable_cases=len({trial.case_id for trial in usable}),
        test_cost_usd=0.07,
        reply_cost_usd=0.0028,
    )


def g2_bundle(bundle: AnalysisContextBundle, arms: Mapping[str, str]) -> AnalysisContextBundle:
    """The bundle with G2's losses: each arm's first trials lost by :data:`LOSSES`, and the census over them."""
    trials = []
    taken: Counter[str] = Counter()
    for trial in bundle.trials:
        name = next(arm for arm, variant in arms.items() if variant == trial.arm)
        index = taken[name]
        taken[name] += 1
        if index < len(LOSSES[name]):
            cause = LOSSES[name][index]
            trial = trial.model_copy(
                update={
                    "state": "lost",
                    "cause": cause,
                    "error_words": "judge: judge reply was cut short by the provider and was not retried"
                    if cause == "truncated"
                    else "test setup: the simulator failed",
                }
            )
        trials.append(trial)
    boards = [_census(arms[name], trials) for name in ("v4.1-flash", "v4-flash")]
    exclusions = [
        ArmExclusions(
            variant_key=board.variant_key,
            n_trials=board.ran,
            n_usable=board.usable,
            by_cause=board.lost,
            truncated_sent_caps=[10240] if "truncated" in board.lost else [],
        )
        for board in boards
    ]
    return bundle.model_copy(
        update={
            "trials": trials,
            "scoreboard": boards,
            "exclusions": exclusions,
            "judge_evidence_tiers": [_unlabelled_judge()],
        }
    )


async def g2_report() -> Report:
    """The G2-shaped campaign's code-only report, assembled at a fixed instant."""
    comparison = await compare(
        CASES,
        {"v4.1-flash": _queue, "v4-flash": _queue},
        expected=lambda case: str(case["label"]),
        control="v4.1-flash",
        k=3,
        name="Kairo queue discipline",
    )
    bundle = g2_bundle(comparison.report.bundle, comparison.arm_variants)
    return build_code_only_report(bundle, assembled_at="2026-10-14T01:31:58+00:00", campaign_name=comparison.name)
