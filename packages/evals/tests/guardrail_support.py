"""Test support for guardrails: a control and one contrast, judged on a capability and a boundary dimension.

The guardrail suites read the bundle the way every consumer does — through
:func:`~threetears.evals.analysis.assemble_context_bundle` — over a campaign of two arms (the control and
one contrast, one run each) that ran the same cases. Each result carries one score per dimension it is
given, stamped with the axis the judge would stamp: ``task.correct`` on the capability axis and
``boundary.correct`` on the boundary axis, both pass/fail unless a test asks otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.kernel import EvalCampaign, Question
from threetears.evals.schema import EvalResult, RubricScore
from threetears.evals.kernel.host import HostProfile
from threetears.evals.schema.models import RubricAxis, RubricScale
from packages.evals.tests.factories import fixture_variant_key, make_eval_result, make_eval_run, minimal_declaration
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

__all__ = [
    "BOUNDARY",
    "CAPABILITY",
    "CONTRAST",
    "CONTROL",
    "QUALITY_QUESTION",
    "two_arm_bundle",
]

CONTROL = "control-model"
CONTRAST = "contrast-model"
CAPABILITY = "task.correct"
BOUNDARY = "boundary.correct"

#: A question asking whether the contrast does the job better: it names the quality axis.
QUALITY_QUESTION = Question(id="q-better", text="does the contrast do the job better?", merit_axes=["quality"])

#: Per dimension: its name, axis and each arm's per-case scores (control, contrast).
Scored = tuple[str, RubricAxis | None, tuple[Sequence[int], Sequence[int]]]


def two_arm_bundle(
    dimensions: Sequence[Scored],
    *,
    scale: RubricScale = "pass_fail",
    questions: Sequence[Question] = (QUALITY_QUESTION,),
    control: bool = True,
    host_measures: tuple[Sequence[dict[str, float]], Sequence[dict[str, float]]] | None = None,
    profile: HostProfile | None = None,
) -> AnalysisContextBundle:
    """The control and one contrast over the same cases, each scored on every dimension in ``dimensions``.

    Args:
        dimensions: Each judged dimension: its name, the axis its scores are stamped with (None = judged
            before the axis was stamped) and each arm's per-case scores, one per case, control first.
        scale: The scale every dimension is judged on.
        questions: The campaign's live questions; empty declares none.
        control: Whether the control arm is declared the control.
        host_measures: Each arm's per-case host measures, control first, when the test reads a measure.
        profile: The host vocabulary; the toy host's when None.

    Returns:
        The assembled bundle.
    """
    n_cases = len(dimensions[0][2][0]) if dimensions else len(host_measures[0]) if host_measures else 0
    runs = []
    results: dict[str, list[EvalResult]] = {}
    for side, model in enumerate((CONTROL, CONTRAST)):
        run = make_eval_run(status="completed", candidate_model=model)
        runs.append(run)
        results[run.id] = [
            make_eval_result(
                id=f"{model}-{case}",
                eval_run_id=run.id,
                scope_id=run.scope_id,
                model=model,
                test_case_id=f"tc-{case:02d}",
                goal_state_outcomes=[],
                cost_usd=None,
                host_measures=dict(host_measures[side][case]) if host_measures is not None else {},
                rubric_scores=[
                    RubricScore(dim=name, scale=scale, axis=axis, score=scores[side][case])
                    for name, axis, scores in dimensions
                ],
            )
            for case in range(n_cases)
        ]
    declaration = minimal_declaration(control=fixture_variant_key(CONTROL) if control else None).model_copy(
        update={"questions": list(questions)}
    )
    campaign = EvalCampaign(
        scope_id=runs[0].scope_id,
        name="guardrails",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[run.id for run in runs],
        declared_design=declaration,
        created_by="test:fixture",
    )
    return assemble_context_bundle(
        campaign, storage=ToyhostStorage(runs, results), profile=profile if profile is not None else toyhost_profile()
    )
