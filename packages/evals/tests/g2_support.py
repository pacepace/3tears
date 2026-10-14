"""A G2-shaped report: the campaign the failed production memo was written over, as a fixture.

Two arms over 4 cases × 3 repeats, run through the quick path (an :class:`~threetears.evals.Eval` comparing two
runners), whose stored results are then given G2's losses:
the control lost 4 of its 12 trials to the test setup, the challenger 10 of its 12 — 7 to a judge whose reply was
cut off at the 10240-token cap it was sent (``judge_cut_short``, as the judge's cut-short path records it), 3 to the
test setup — and no question was declared. The bundle is then assembled again over them, so its ``exclusions``,
``differential_exclusions``, ``scoreboard`` and ``trials`` are the assembler's own, each trial classified by
``trial_exclusion``. Its judge was new, with no human labels yet, as every judge on a first campaign is. Shared by
the golden test and the vocabulary test.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from threetears.evals.analysis import (
    AnalysisContextBundle,
    Report,
    assemble_context_bundle,
    build_code_only_report,
    get_campaign,
)
from threetears.evals.kernel.evidence_tiers import (
    CALIBRATION_MIN_AGREEMENT,
    CALIBRATION_MIN_RESULTS,
    SEPARATION_MIN_AGREEMENT,
    SEPARATION_MIN_RESULTS,
    JudgeEvidenceTier,
    TierCriterion,
)
from threetears.evals import Eval, Result
from threetears.evals.run import JUDGE_REQUEST_SETTINGS
from threetears.evals.schema import EvalResult, JudgeCutShort
from packages.evals.tests.off_loop import off_loop

__all__ = ["CASES", "JUDGE", "LOSSES", "g2_bundle", "g2_report"]


#: Four cases, each run three times on each arm.
CASES = [{"text": f"queue request {index}", "label": "queued"} for index in range(4)]

#: The trials each arm loses, by cause, in trial order: the control four to the test setup, the challenger seven to
#: a truncated judge and three to the test setup.
LOSSES = {"v4.1-flash": ["infra"] * 4, "v4-flash": ["truncated"] * 7 + ["infra"] * 3}


#: The campaign's judge: new, so nobody has labelled any of its outputs yet.
JUDGE = "judge-model-1"

#: The dimension the judge's reply was cut off on.
_DIM = "conversation.queue_discipline"


def _unlabelled_judge() -> JudgeEvidenceTier:
    """The judge's standing on its one dimension: no human labels and no repeats, so no tier is decided."""
    return JudgeEvidenceTier(
        rubric_dim=_DIM,
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


def _lose(result: EvalResult, cause: str) -> EvalResult:
    """The result as the run would have stored it had it been lost to ``cause``."""
    if cause == "truncated":
        cut = JudgeCutShort(
            stop_reason="max_tokens",
            sent_settings=JUDGE_REQUEST_SETTINGS,
            output_tokens=JUDGE_REQUEST_SETTINGS.max_tokens,
            reasoning_tokens=9918,
        )
        return result.model_copy(
            update={
                "judge_error": f"{_DIM}: judge reply cut off at its output cap (stop_reason=max_tokens, sent "
                f"max_tokens={JUDGE_REQUEST_SETTINGS.max_tokens}); not re-bought",
                "judge_cut_short": {_DIM: cut},
            }
        )
    return result.model_copy(update={"infra_error": "the simulator failed"})


def g2_bundle(comparison: Result) -> AnalysisContextBundle:
    """The campaign's bundle once each arm's first trials, in trial order, are lost by :data:`LOSSES`.

    The losses are written onto the stored results and the bundle assembled again over them, so every count the
    report prints is the assembler's own; only the judge's standing is set by hand, since a quick run has no judge.
    """
    storage = comparison.host.storage
    for name, causes in LOSSES.items():
        run_id = comparison.arms[name].run_id
        results = sorted(
            storage.query_eval_results_by_run(run_id, comparison.scope_id),
            key=lambda result: (result.test_case_id, result.k_iteration, result.id),
        )
        for result, cause in zip(results, causes, strict=False):
            storage.replace_eval_result(_lose(result, cause), if_match=None)
    campaign = get_campaign(storage, comparison.campaign_id, comparison.scope_id)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=comparison.host.profile)
    return bundle.model_copy(update={"judge_evidence_tiers": [_unlabelled_judge()]})


async def g2_report() -> Report:
    """The G2-shaped campaign's code-only report, assembled at a fixed instant."""
    ev = Eval(CASES, {"v4.1-flash": _queue, "v4-flash": _queue}, ["label"], k=3, name="Kairo queue discipline")
    # The scope the quick path once defaulted to, so the fixture's stored documents are where they were.
    comparison = await ev.run(store=":memory:", scope="quick")
    bundle = off_loop(g2_bundle, comparison)
    return build_code_only_report(bundle, assembled_at="2026-10-14T01:31:58+00:00", campaign_name=comparison.name)
