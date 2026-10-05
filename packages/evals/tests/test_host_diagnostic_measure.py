"""A host declares a diagnostic on its measure's descriptor, and it reaches the bundle through the engine's predicate.

A directionless numeric measure is either a raw count, which the bundle keeps off every measure
surface (a per-role count pooled across roles is a distribution of nothing), or a diagnostic, which
it carries so a reader can explain a movement. Nothing about the VALUES tells the two apart, so the
descriptor says which: :attr:`MetricDescriptor.diagnostic`. The engine's own diagnostic is declared
the same way, so one predicate admits both — before this, the engine's diagnostics were a fixed set
of names inside the engine, and a host's signed error (forged XP minus the asked budget, say)
registered and then appeared on no surface.

The toy host declares ``field_count_error``: fields emitted minus fields graded, signed, with no
better end.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from threetears.evals.contracts import MetricDescriptor
from threetears.evals.contracts import UnreadableBarName, resolve_bar_name
from threetears.evals.contracts.host import Bar, BarRegistry, MeasureRegistry
from threetears.evals.contracts.host.bars import BarRegistrationError
from threetears.evals.contracts.metrics import describe_measure
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import FIELD_COUNT_ERROR
from packages.evals.tests.fixtures.toyhost.profile import (
    TOYHOST_BARS,
    TOYHOST_EXTRACTION_FAMILY,
    TOYHOST_MEASURES,
    toyhost_profile,
)
from packages.evals.tests.fixtures.toyhost.run import execute_toyhost_run, toyhost_run_bundle

#: A count the toy kind already emits on every cell, declared here as what it is: a raw count.
FIELDS_CORRECT = "fields_correct"

FIELDS_CORRECT_AS_A_RAW_COUNT = MetricDescriptor(
    name=FIELDS_CORRECT,
    data_type="numeric",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="subsystem",
    description="How many graded fields the extractor got exactly right, as a count.",
    unit="fields",
)


def _diagnostic(**overrides: object) -> MetricDescriptor:
    fields: dict[str, object] = {
        "name": "signed_error",
        "data_type": "numeric",
        "family": "mechanical",
        "transferability_class": "mechanical",
        "attribution_scope": "subsystem",
        "description": "What was produced minus what was asked.",
        "diagnostic": True,
    }
    return MetricDescriptor.model_validate(fields | overrides)


async def test_a_host_declared_diagnostic_reaches_the_bundles_measure_surfaces():
    path = await execute_toyhost_run(host=toyhost_host())
    bundle = toyhost_run_bundle(path)

    # The value the toy kind emits, so the absence below cannot be a measure that was never produced.
    assert all(FIELD_COUNT_ERROR in result.host_measures for result in path.results)
    assert bundle.measure_catalog[FIELD_COUNT_ERROR].diagnostic is True
    assert bundle.measure_catalog[FIELD_COUNT_ERROR].higher_is_better is None
    assert bundle.run_summaries, "the campaign resolved its runs"
    assert all(FIELD_COUNT_ERROR in {m.name for m in summary.measures.measures} for summary in bundle.run_summaries)


async def test_a_declared_raw_count_stays_out_while_the_diagnostic_beside_it_is_carried():
    """The stated rule holds: a directionless measure that does not declare itself a diagnostic is a count."""
    profile = toyhost_profile()
    profile = replace(
        profile,
        measures=MeasureRegistry(
            (*TOYHOST_MEASURES, FIELDS_CORRECT_AS_A_RAW_COUNT), families=(TOYHOST_EXTRACTION_FAMILY,)
        ),
    )
    path = await execute_toyhost_run(host=toyhost_host(profile=profile))
    bundle = toyhost_run_bundle(path)

    # Emitted and described — so its absence is the filter's decision, not a gap.
    assert all(FIELDS_CORRECT in result.host_measures for result in path.results)
    assert profile.measures.get(FIELDS_CORRECT) == FIELDS_CORRECT_AS_A_RAW_COUNT
    assert FIELDS_CORRECT not in bundle.measure_catalog
    assert all(FIELDS_CORRECT not in {m.name for m in summary.measures.measures} for summary in bundle.run_summaries)
    assert FIELD_COUNT_ERROR in bundle.measure_catalog


def test_the_engines_own_diagnostic_is_declared_on_its_descriptor_like_a_hosts():
    """One predicate for both: the engine's diagnostic carries the same field a host sets."""
    measures = toyhost_profile().measures
    assert describe_measure("candidate_output_tokens_per_s", measures).diagnostic is True
    assert describe_measure(FIELD_COUNT_ERROR, measures).diagnostic is True


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_a_diagnostic_declaring_a_better_end_is_refused(higher_is_better: bool):
    with pytest.raises(ValidationError, match="declared a diagnostic and higher_is_better"):
        _diagnostic(higher_is_better=higher_is_better)


def test_a_directionless_diagnostic_and_a_directional_non_diagnostic_are_both_accepted():
    """The refusal's two neighbours, so it is the conjunction that fires and not either half."""
    assert _diagnostic().diagnostic is True
    assert _diagnostic(diagnostic=False, higher_is_better=True).higher_is_better is True


def test_a_bar_on_a_diagnostic_is_refused_naming_it_a_diagnostic_and_a_count_naming_it_a_count():
    profile = replace(
        toyhost_profile(),
        measures=MeasureRegistry(
            (*TOYHOST_MEASURES, FIELDS_CORRECT_AS_A_RAW_COUNT), families=(TOYHOST_EXTRACTION_FAMILY,)
        ),
    )

    def resolve(name: str) -> UnreadableBarName:
        resolved = resolve_bar_name(name, rubric_dimensions={}, goal_state_checks=(), measures=profile.measures)
        assert isinstance(resolved, UnreadableBarName), resolved
        assert resolved.refusal == "no_better_end"
        return resolved

    assert "— a diagnostic —" in resolve(FIELD_COUNT_ERROR).reason
    assert "— a raw count —" in resolve(FIELDS_CORRECT).reason


def _bar_on(measure: str) -> Bar:
    return Bar(
        behavior="extract_invoice_fields",
        measure=measure,
        threshold=0.0,
        higher_is_better=True,
        rationale="a bar on a measure with no better end, which nothing could ever read",
    )


@pytest.mark.parametrize(("measure", "what"), [(FIELD_COUNT_ERROR, "a diagnostic"), (FIELDS_CORRECT, "a raw count")])
def test_registering_a_bar_on_a_measure_with_no_better_end_is_refused_naming_which_and_a_directional_one_is_not(
    measure: str, what: str
):
    """Registration refuses what the ratchet and the campaign gate already refuse, through their predicate.

    Before this ``validate_against`` checked only that the measure was declared and that the bar's
    direction did not contradict it — and a directionless descriptor contradicts nothing, so a bar
    on one registered and was never read. Both directions run on one profile's measures, so the
    refusal is the measure's lack of a better end and not something else about the fixture.
    """
    profile = replace(
        toyhost_profile(),
        measures=MeasureRegistry(
            (*TOYHOST_MEASURES, FIELDS_CORRECT_AS_A_RAW_COUNT), families=(TOYHOST_EXTRACTION_FAMILY,)
        ),
    )
    # The accepting direction: the toy's own bars, both on directional measures, register against
    # these same measures. Without it the refusal below could be any refusal at all.
    assert replace(profile, bars=BarRegistry(TOYHOST_BARS)).bars.bars == TOYHOST_BARS

    with pytest.raises(
        BarRegistrationError, match=f"{measure} names a measure that declares no better direction — {what} —"
    ):
        replace(profile, bars=BarRegistry([*TOYHOST_BARS, _bar_on(measure)]))


def test_the_ratchet_names_which_directionless_shape_it_refuses_the_way_registration_does():
    profile = replace(
        toyhost_profile(),
        measures=MeasureRegistry(
            (*TOYHOST_MEASURES, FIELDS_CORRECT_AS_A_RAW_COUNT), families=(TOYHOST_EXTRACTION_FAMILY,)
        ),
    )
    for measure, what in ((FIELD_COUNT_ERROR, "a diagnostic"), (FIELDS_CORRECT, "a raw count")):
        with pytest.raises(BarRegistrationError, match=f"declares no better direction — {what} —"):
            profile.bars.propose(
                behavior="extract_invoice_fields",
                measure=measure,
                observed=1.0,
                measures=profile.measures,
                rationale="no such thing as clearing this",
            )
