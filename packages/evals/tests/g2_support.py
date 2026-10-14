"""A G2-shaped report: the campaign the failed production memo was written over, as a fixture.

Two arms over 4 cases × 3 repeats, run through the quick path (an :class:`~threetears.evals.Eval` comparing two
runners), whose stored results are then given G2's losses:
the control lost 4 of its 12 trials to the test setup, the challenger 10 of its 12 — 7 to a judge whose reply was
cut off at the 10240-token cap it was sent (``judge_cut_short``, as the judge's cut-short path records it), 3 to the
test setup — and no question was declared. The bundle is then assembled again over them, so its ``exclusions``,
``differential_exclusions``, ``scoreboard`` and ``trials`` are the assembler's own, each trial classified by
``trial_exclusion``. Its judge was new, with no human labels yet, as every judge on a first campaign is. Shared by
the golden test and the vocabulary test.

:func:`g2_ruled_report` is the later G2 run instead: its own numbers (production passed 10 of 12, the cheaper
candidate 2 of 12, at 0.003092 and 0.003301 per reply), its arms named as the host names them
(``deepseek-v4.1-flash (production)``), and its decision rule declared before the run. Shared by the decision-rule
and presentation tests.
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
from threetears.evals import Answer, Eval, Result
from threetears.evals.kernel.declaration import Question
from threetears.evals.run import JUDGE_REQUEST_SETTINGS
from threetears.evals.schema import EvalResult, JudgeCutShort
from packages.evals.tests.off_loop import off_loop

__all__ = [
    "CANDIDATE",
    "CASES",
    "G2_QUESTION",
    "JUDGE",
    "LOSSES",
    "PRODUCTION",
    "answering",
    "declaring",
    "g2_bundle",
    "g2_numbers",
    "g2_report",
    "g2_ruled_report",
]


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


# --- the G2 numbers, replayed with arms named as a host names them, and its decision rule declared -------

#: The G2 campaign's declared question, decision rule included, as its operator wrote it.
G2_QUESTION = (
    "On Kairo's queue-discipline regression set, does deepseek-v4-flash hold every pass/fail queue-discipline "
    "dimension within 10 points of deepseek-v4.1-flash's pass rate, at lower cost per turn? Decision rule: switch "
    "Kairo's model only if no dimension drops by more than 10 points AND the cost per turn is lower; otherwise keep "
    "deepseek-v4.1-flash."
)

PRODUCTION = "deepseek-v4.1-flash (production)"
CANDIDATE = "deepseek-v4-flash (cheaper candidate)"


def answering(passes: int, cost: float) -> Any:
    """A runner whose first ``passes`` calls answer right and the rest wrong, each costing ``cost``."""
    calls = 0

    async def run(case: Mapping[str, Any]) -> Answer:
        nonlocal calls
        calls += 1
        label = case["label"] if calls <= passes else "skipped"
        return Answer(label, model="deepseek", input_tokens=100, output_tokens=10, cost_usd=cost)

    return run


async def g2_numbers() -> Result:
    arms = {PRODUCTION: answering(10, 0.003092), CANDIDATE: answering(2, 0.003301)}
    return await Eval(CASES, arms, ["label"], k=3, name="Kairo queue discipline").run(store=":memory:")


def declaring(bundle: AnalysisContextBundle, text: str) -> AnalysisContextBundle:
    """The bundle with ``text`` declared before the run, as the campaign's one live question."""
    design = bundle.declared_design
    assert design is not None, "a quick comparison declares its design"
    asked = Question(id="q-g2", text=text, asked_at="2026-10-14T00:00:00+00:00")
    return bundle.model_copy(
        update={
            "declared_design": design.model_copy(update={"questions": [asked]}),
            "reading_scope": bundle.reading_scope.model_copy(update={"questions_declared": True}),
            "design_frozen_at": "2026-10-14T01:00:00+00:00",
        }
    )


async def g2_ruled_report() -> tuple[Result, Report]:
    """The G2 numbers through the quick path, with the G2 decision rule declared before the run."""
    comparison = await g2_numbers()
    bundle = declaring(comparison.report.bundle, G2_QUESTION)
    return comparison, build_code_only_report(
        bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name="G2: can the cheaper model replace production?"
    )
