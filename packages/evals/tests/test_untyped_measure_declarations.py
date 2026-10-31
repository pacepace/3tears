"""A host measure that declares a better end or a diagnostic declares its data type too.

The bundle's measure surfaces keep only a typed measure, so a directed or diagnostic measure declared with
no ``data_type`` would register and then reach no bundle, numbers included. The registry refuses it where
it is declared. A measure with neither stays legal untyped, and so does what :func:`describe_measure`
answers for a name nobody describes, which is no declaration at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.kernel import JUDGE_KIND_MEASURES, MetricDescriptor
from threetears.evals.kernel.host import MeasureRegistry
from threetears.evals.kernel.host.measures import MeasureRegistrationError
from threetears.evals.kernel.metrics import METRIC_DESCRIPTORS, describe_measure
from packages.evals.tests.fixtures.courierhost import courier_profile
from packages.evals.tests.fixtures.report_plans.plan_eval import PROFILE as PLAN_EVAL_PROFILE
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile


def _untyped(**declared: Any) -> MetricDescriptor:
    return MetricDescriptor(
        name="queue_jumps",
        reader_name="Queue jumps",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="How many places the candidate's turn jumped the queue.",
        **declared,
    )


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_a_directed_measure_with_no_data_type_is_refused_where_it_is_declared(higher_is_better: bool) -> None:
    with pytest.raises(MeasureRegistrationError) as refused:
        MeasureRegistry([_untyped(higher_is_better=higher_is_better)])
    assert "queue_jumps declares a better end and no data_type" in str(refused.value)
    assert "declare its data_type" in str(refused.value)


def test_a_diagnostic_with_no_data_type_is_refused_where_it_is_declared() -> None:
    with pytest.raises(MeasureRegistrationError) as refused:
        MeasureRegistry([_untyped(diagnostic=True)])
    assert "queue_jumps declares itself a diagnostic and no data_type" in str(refused.value)


def test_a_guardrail_with_no_data_type_is_refused_by_its_direction() -> None:
    with pytest.raises(MeasureRegistrationError, match="queue_jumps declares a better end and no data_type"):
        MeasureRegistry([_untyped(higher_is_better=False, guardrail=True)])


def test_the_same_measures_typed_register() -> None:
    measures = MeasureRegistry(
        [
            _untyped(data_type="numeric", higher_is_better=False).model_copy(
                update={"name": "jumps", "reader_name": "Jumps"}
            )
        ]
        + [_untyped(data_type="numeric", diagnostic=True)]
    )
    assert measures.names == ("jumps", "queue_jumps")


def test_an_undirected_measure_that_is_no_diagnostic_stays_legal_untyped() -> None:
    assert MeasureRegistry([_untyped()]).names == ("queue_jumps",)


def test_the_answer_for_a_name_nobody_describes_is_no_declaration_and_stays_legal() -> None:
    described = describe_measure("never_described", MeasureRegistry([]))
    assert (described.data_type, described.family, described.higher_is_better, described.diagnostic) == (
        None,
        None,
        None,
        False,
    )
    assert MeasureRegistry([described.model_copy(update={"reader_name": "Never described"})]).names == (
        "never_described",
    )


def test_every_in_tree_declaration_types_each_directed_or_diagnostic_measure() -> None:
    """The engine's core, the judge kind's measures, and every fixture host's catalogue pass the rule.

    The fixture hosts are built through their registries, which refuse a breach at construction; the core and
    the judge kind's measures are never registered by a host, so they are read here directly.
    """
    hosts = [toyhost_profile().measures, courier_profile().measures, PLAN_EVAL_PROFILE.measures]
    declared = [d for registry in hosts for name in registry.names if (d := registry.get(name)) is not None]
    declared += [*METRIC_DESCRIPTORS.values(), *JUDGE_KIND_MEASURES]
    assert len(declared) > len(METRIC_DESCRIPTORS)
    untyped = sorted(
        d.name for d in declared if d.data_type is None and (d.higher_is_better is not None or d.diagnostic)
    )
    assert untyped == []
    MeasureRegistry(JUDGE_KIND_MEASURES)
