"""A negative value on a measure declared non-negative is named, never only clamped.

A host bug that records ``-3`` on a count is hidden by the interval's floor at zero. The bundle counts the results
that did, per measure and per arm; the page states one line saying what the clamp did; the writer's input carries the
same line as a caveat; a warning names the measure and the result ids, capped. A fixture with no such value says
nothing anywhere.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import (
    AnalysisContextBundle,
    DisclosureBlock,
    assemble_context_bundle,
    build_code_only_report,
)
from threetears.evals.analysis.bundle.schema import NEGATIVE_VALUE_IDS_SHOWN
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.host import SHARED_CORE, HostProfile, MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.schema import EvalResult, GoalStateOutcome, LatencyMetrics
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import fixture_variant_key, make_campaign, make_eval_result, make_eval_run

_SCOPE = "uni-1"
_COUNT = "parcels_sent"
_CHECK = "state.answer.delivered == true"
_CASES = [f"case-{index}" for index in range(12)]

_PROFILE = HostProfile(
    host_id="negative-count",
    host_sweepables=SHARED_CORE,
    measures=MeasureRegistry(
        [
            MetricDescriptor(
                name=_COUNT,
                reader_name="Parcels sent",
                data_type="numeric",
                family="mechanical",
                transferability_class="mechanical",
                attribution_scope="end_to_end",
                description="Parcels the candidate sent in one turn.",
                higher_is_better=True,
                unit="count",
                nonnegative=True,
            )
        ]
    ),
)


def _result(model: str, case_id: str, items: float) -> EvalResult:
    return make_eval_result(
        scope_id=_SCOPE,
        test_case_id=case_id,
        model=model,
        variant_key=fixture_variant_key(model),
        latency=LatencyMetrics(total_ms=400.0),
        host_measures={_COUNT: items},
        goal_state_outcomes=[GoalStateOutcome(expression=_CHECK, passed=True)],
        rubric_scores=[],
    )


def _bundle(arms: Mapping[str, list[EvalResult]]) -> AnalysisContextBundle:
    storage = EvalStorage(InMemoryDocumentStore())
    run_ids = []
    for model, results in arms.items():
        run = make_eval_run(
            id=f"run-{model}",
            scope_id=_SCOPE,
            candidate_model=model,
            k_runs=1,
            test_case_ids=sorted({result.test_case_id for result in results}),
            status="completed",
        )
        storage.save_eval_run(run)
        for result in results:
            storage.save_eval_result(result.model_copy(update={"eval_run_id": run.id}))
        run_ids.append(run.id)
    campaign = make_campaign(scope_id=_SCOPE, run_ids=run_ids)
    storage.save_campaign(campaign)
    return assemble_context_bundle(campaign, storage=storage, profile=_PROFILE)


def _methods(bundle: AnalysisContextBundle) -> list[str]:
    report = build_code_only_report(bundle, assembled_at="2026-10-09T00:00:00+00:00")
    return [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]


def _arms(negative_at: Mapping[str, list[int]]) -> dict[str, list[EvalResult]]:
    return {
        model: [
            _result(model, case, -3.0 if index in negative_at.get(model, []) else 5.0)
            for index, case in enumerate(_CASES)
        ]
        for model in ("alpha", "beta")
    }


def test_one_negative_value_is_counted_printed_and_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        bundle = _bundle(_arms({"alpha": [2]}))
    (negative,) = bundle.negative_values
    assert (negative.measure, negative.variant_key, negative.n_results) == (_COUNT, fixture_variant_key("alpha"), 1)
    assert len(negative.result_ids) == 1
    lines = [text for text in _methods(bundle) if "negative value" in text]
    assert len(lines) == 1
    assert lines[0].startswith("Parcels sent, ")
    assert (
        "1 result recorded a negative value on a measure declared non-negative (a host bug); it is drawn at 0."
        in lines[0]
    )
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING and "negative value" in r.getMessage()]
    assert _COUNT in warning.getMessage()
    assert negative.result_ids[0] in warning.getMessage()


def test_the_ids_are_capped_but_the_count_is_whole(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        bundle = _bundle(_arms({"alpha": list(range(len(_CASES)))}))
    (negative,) = bundle.negative_values
    assert negative.n_results == len(_CASES)
    assert len(negative.result_ids) == NEGATIVE_VALUE_IDS_SHOWN < len(_CASES)
    assert f"{len(_CASES)} results recorded a negative value" in "\n".join(_methods(bundle))
    (warning,) = [r for r in caplog.records if "negative value" in r.getMessage()]
    assert sum(result_id in warning.getMessage() for result_id in negative.result_ids) == NEGATIVE_VALUE_IDS_SHOWN


def test_all_non_negative_values_say_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        bundle = _bundle(_arms({}))
    assert bundle.negative_values == []
    assert "negative_values" not in bundle.model_dump(mode="json")
    assert not [text for text in _methods(bundle) if "negative value" in text]
    assert not [r for r in caplog.records if "negative value" in r.getMessage()]


def test_the_clamp_is_unchanged_by_the_disclosure() -> None:
    bundle = _bundle(_arms({"alpha": [2]}))
    cell: Any = next(c for c in bundle.cell_measures if c.variant_key == fixture_variant_key("alpha"))
    summary = next(m for m in cell.measures.measures if m.name == _COUNT)
    assert summary.n == len(_CASES)


def test_a_negative_value_reaches_the_writers_input_as_a_caveat() -> None:
    from packages.evals.tests.report_shapes_support import SHAPES
    from threetears.evals.analysis.bundle.schema import NegativeValues
    from threetears.evals.analysis.campaign_bundle import build_campaign_bundle
    from threetears.evals.analysis.report_input_build import report_input

    shape = SHAPES["s1"]()
    bundle = build_campaign_bundle(
        shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings
    )
    first = bundle.aspects[0]
    key = first.design.contrasts[0].variant_key
    measure = next(iter(first.measure_catalog))
    negative = NegativeValues(measure=measure, variant_key=key, n_results=3, result_ids=["a", "b", "c"])
    built = report_input(
        bundle.model_copy(
            update={"aspects": [first.model_copy(update={"negative_values": [negative]}), *bundle.aspects[1:]]}
        )
    )
    assert [line for line in built.caveats.unreported if "3 results recorded a negative value" in line]
