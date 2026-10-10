"""The frontier's boundary pillar on a toy-host campaign (#613): a contestant failing only a guardrail is disqualified.

The campaign is the guardrail suites' two arms — a control and one contrast over the same twenty cases — read the way
every consumer reads it, through :func:`~threetears.evals.analysis.assemble_context_bundle`. The bundle's frontier
holds each contestant's boundary dimensions against the campaign's control arm by the guardrail rule the bundle's own
guardrails are decided by, so the two never disagree about which arm breached.
"""

from __future__ import annotations

from threetears.evals.analysis.bundle import bundle_decision_surface
from threetears.evals.schema.models import EvalTemplate, RubricDim
from packages.evals.tests.guardrail_support import BOUNDARY, CAPABILITY, CONTRAST, two_arm_bundle

_CASES = 20


def test_a_contrast_failing_only_the_boundary_dim_is_disqualified_and_annotated_with_it() -> None:
    bundle = two_arm_bundle(
        [
            (CAPABILITY, "capability", ([1] * _CASES, [1] * _CASES)),
            (BOUNDARY, "boundary", ([1] * _CASES, [0] * _CASES)),
        ]
    )

    (subject,) = bundle.frontier.subjects
    (contrast,) = [point for point in subject.points if point.model == CONTRAST]
    assert contrast.disqualified_by == [BOUNDARY]
    assert [(check.dimension, check.decision) for check in contrast.boundary_checks] == [(BOUNDARY, "breached")]
    assert subject.boundary_dimensions == [BOUNDARY] and subject.n_disqualified == 1
    # The frontier and the bundle's guardrail section read one rule over one pair of arms.
    (guardrail,) = [check for check in bundle.guardrails.checks if check.name == BOUNDARY]
    assert guardrail.decision == "breached"


def test_a_contrast_holding_the_boundary_dim_is_not_disqualified() -> None:
    bundle = two_arm_bundle(
        [
            (CAPABILITY, "capability", ([1] * _CASES, [0] * _CASES)),
            (BOUNDARY, "boundary", ([1] * _CASES, [1] * _CASES)),
        ]
    )

    (subject,) = bundle.frontier.subjects
    (contrast,) = [point for point in subject.points if point.model == CONTRAST]
    assert contrast.disqualified_by == []


def test_a_template_stored_before_dims_carried_an_axis_still_loads_as_capability() -> None:
    template = EvalTemplate(
        scope_id="s",
        name="t",
        intent="i",
        candidate_kind="k",
        rubric=[RubricDim(name="ctx.d", description="x", scale="ordinal")],
    )
    stored = template.to_dict()
    for dim in stored["rubric"]:
        dim.pop("axis", None)

    (dim,) = EvalTemplate.from_dict(stored).rubric
    assert dim.axis == "capability"


def test_the_decision_surface_carries_the_disqualification_a_frontier_chart_draws() -> None:
    bundle = two_arm_bundle(
        [
            (CAPABILITY, "capability", ([1] * _CASES, [1] * _CASES)),
            (BOUNDARY, "boundary", ([1] * _CASES, [0] * _CASES)),
        ]
    )

    surface = bundle_decision_surface(bundle)

    (contrast,) = [point for point in bundle.frontier.subjects[0].points if point.model == CONTRAST]
    assert surface.frontier_disqualified == {contrast.variant_key: [BOUNDARY]}
