"""A difference below a measure's declared materiality threshold is labelled immaterial — wherever one is stated.

``MetricDescriptor.materiality_threshold`` is the host's word for "smaller than this is not worth acting
on". It is read by one predicate, :func:`~threetears.evals.kernel.materiality`, at every place the
engine states a difference:

* **the analysis bundle**, where every :class:`~threetears.evals.analysis.MeasureMovement` between two
  levels of a lever carries ``materiality`` — the input a generator's caveats are written from — and so
  does every comparison a declared question's family tests (``FamilyComparison.materiality``, pinned in
  ``test_multiple_comparisons.py``);
* **the decision surface**, which freezes each measure's threshold beside its facts, so a delta table
  drawn from it labels a row ``immaterial`` in its values table and names it in a disclosure.

Silence is conservative: a measure that declared no threshold has every difference material.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import MeasureMovement, assemble_context_bundle
from threetears.evals.analysis.bundle.surface import cell_measure_facts
from threetears.evals.analysis.viz import chart_intent
from threetears.evals.kernel import MetricDescriptor, materiality
from threetears.evals.schema.models import LatencyMetrics
from threetears.evals.kernel.host import HostProfile, MeasureRegistry
from threetears.evals.kernel.surface import MeasureFacts
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, ToyhostStorage
from packages.evals.tests.fixtures.toyhost.kind import FIELD_ACCURACY
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.test_viz_refs import build, chart, measure_facts, ref, surface


def _latency(threshold: float | None) -> MetricDescriptor:
    return MetricDescriptor(
        name="total_ms",
        reader_name="Turn time",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Wall-clock.",
        higher_is_better=False,
        unit="ms",
        materiality_threshold=threshold,
    )


# --- the predicate -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("threshold", "delta", "expected"),
    [
        (50.0, 49.9, "immaterial"),
        (50.0, -49.9, "immaterial"),
        (50.0, 50.0, "material"),
        (50.0, -120.0, "material"),
        (None, 0.0001, "material"),
    ],
)
def test_the_predicate_reads_the_threshold_and_ignores_the_sign(
    threshold: float | None, delta: float, expected: str
) -> None:
    assert materiality(threshold, delta) == expected


def test_a_negative_threshold_is_refused() -> None:
    with pytest.raises(ValidationError):
        _latency(-1.0)
    with pytest.raises(ValidationError):
        MeasureFacts(materiality_threshold=-1.0)


# --- the bundle labels a movement ---------------------------------------------------------------------


#: The host-declared end-to-end latency the movement tests grade. A host may declare a threshold only
#: on its own measures — the engine core is consulted first — so the movement is read on this one.
_P95 = "p95_extract_ms"


def _with_p95_threshold(threshold: float | None) -> HostProfile:
    profile = toyhost_profile()
    measures = tuple(
        descriptor.model_copy(update={"materiality_threshold": threshold}) if descriptor.name == _P95 else descriptor
        for descriptor in TOYHOST_MEASURES
    )
    return replace(profile, measures=MeasureRegistry(measures, families=profile.measures.families))


def _divergent_movements(threshold: float | None) -> dict[str, MeasureMovement]:
    """Each end-to-end movement in the toy campaign's scope divergences, by measure name.

    The batches record ``p95_extract_ms`` as their wall-clock, which moves between the two chunk
    widths, and a ``tool_ms`` that holds flat — so the whole and the part disagree in direction and
    the bundle reports the pair, with the whole's movement graded against ``threshold``.
    """
    profile = _with_p95_threshold(threshold)
    campaign, storage = toyhost_campaign(profile=profile)
    runs = [storage.load_eval_run(run_id, TOYHOST_SCOPE) for run_id in campaign.run_ids]
    results = {
        run_id: [
            result.model_copy(
                update={
                    "host_measures": {**result.host_measures, _P95: result.latency.total_ms},
                    "latency": LatencyMetrics(total_ms=result.latency.total_ms, tool_ms=300.0 + result.k_iteration),
                }
            )
            for result in storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE)
        ]
        for run_id in campaign.run_ids
    }
    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=profile)
    return {divergence.end_to_end.name: divergence.end_to_end for divergence in bundle.scope_divergences}


def _p95_delta() -> float:
    movement = _divergent_movements(None)[_P95]
    assert movement.direction != "flat", "the fixture must move the measure clear of its noise"
    return abs(movement.delta)


def test_a_movement_that_clears_its_noise_and_not_its_threshold_is_immaterial() -> None:
    """The two readings are independent: improved against noise, and still too small to act on."""
    movements = _divergent_movements(_p95_delta() * 2)

    assert movements[_P95].direction == "improved"
    assert movements[_P95].materiality == "immaterial"
    # Read per measure: the core wall-clock moved by the same amount and declares no threshold.
    assert movements["total_ms"].materiality == "material"


def test_a_movement_on_a_measure_with_no_threshold_is_material() -> None:
    assert _divergent_movements(None)[_P95].materiality == "material"


def test_a_movement_at_or_beyond_the_threshold_is_material() -> None:
    delta = _p95_delta()
    assert _divergent_movements(delta)[_P95].materiality == "material"
    assert _divergent_movements(delta / 2)[_P95].materiality == "material"


# --- the decision surface carries the threshold, and a delta drawn from it is labelled ------------------


def test_the_decision_surface_freezes_each_measures_threshold() -> None:
    measures = tuple(
        descriptor.model_copy(update={"materiality_threshold": 0.02})
        if descriptor.name == FIELD_ACCURACY
        else descriptor
        for descriptor in TOYHOST_MEASURES
    )
    profile = replace(
        toyhost_profile(), measures=MeasureRegistry(measures, families=toyhost_profile().measures.families)
    )

    facts = cell_measure_facts(toyhost_bundle(profile=profile))

    assert facts[FIELD_ACCURACY].materiality_threshold == 0.02
    assert facts[FIELD_ACCURACY].population == "scored"


def _delta_table(facts: dict[str, MeasureFacts]):
    payload = build(chart("delta_table", [ref("A"), ref("B")], ["pass_rate", "total_ms"]), surface(facts=facts))
    return payload, chart_intent("delta_table", payload)


def test_a_delta_below_its_threshold_is_labelled_immaterial_on_the_surface() -> None:
    """``total_ms`` moves 200ms between A and B, under a 250ms threshold; ``pass_rate`` moves 0.2, over 0.05."""
    facts = measure_facts()
    facts["total_ms"] = facts["total_ms"].model_copy(update={"materiality_threshold": 250.0})
    facts["pass_rate"] = facts["pass_rate"].model_copy(update={"materiality_threshold": 0.05})

    payload, compiled = _delta_table(facts)

    rows = {row["metric"]: row for row in payload["rows"]}
    assert (rows["Turn time"]["materiality"], rows["pass_rate"]["materiality"]) == ("immaterial", "material")
    table = {row["metric"]: row for row in compiled.rows}
    assert table["Turn time"]["delta"].endswith("(immaterial)")
    assert not table["pass_rate"]["delta"].endswith("(immaterial)")
    (disclosure,) = [d for d in compiled.disclosures if "materiality threshold" in d]
    assert "1 of 2 changes" in disclosure and "Turn time" in disclosure


def test_a_surface_with_no_thresholds_labels_nothing_immaterial() -> None:
    payload, compiled = _delta_table(measure_facts())

    assert {row["materiality"] for row in payload["rows"]} == {"material"}
    assert not any("materiality threshold" in d for d in compiled.disclosures)
