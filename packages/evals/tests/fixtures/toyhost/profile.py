"""The toy host's profile — six measures, two bars in opposite directions, a world, and a style unlike the default.

Each measure is chosen for a value-shape the engine has to handle: a bounded ratio, two unbounded quantities in opposite better-directions, an
**unbounded count** — the class where a confidence interval reached −7.7 on a quantity that
cannot go below zero and shipped green — and a **signed diagnostic** with no better end, which the
host declares as one so the bundle carries it rather than dropping it as a raw count.

All four merit axes are represented, which is the assertion behind the axis enum being
engine-owned and closed: a second product's measures land on the same four without being made to.
"""

from __future__ import annotations

from dataclasses import replace

from threetears.evals.contracts import MetricDescriptor
from threetears.evals.contracts.host import (
    CHART_FONT_CHARACTERS,
    Bar,
    BarRegistry,
    ChartFont,
    ChartPalette,
    HostProfile,
    MeasureRegistry,
    StyleProfile,
)
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT
from packages.evals.tests.fixtures.toyhost.kind import (
    FIELD_COUNT_ERROR,
    TOYHOST_EXTRACTION_FAMILY,
    field_accuracy,
    field_count_error,
)
from packages.evals.tests.fixtures.toyhost.sweepables import (
    TOYHOST_SWEEPABLE_REGISTRY,
    TOYHOST_TUNABLE_SWEEPABLE_REGISTRY,
)
from packages.evals.tests.fixtures.toyhost.variant import tunable_variant_levers, variant_levers
from packages.evals.tests.fixtures.toyhost.world import toyhost_world

#: Opaque to the engine, which never branches on it.
TOYHOST_ID = "toyhost"

#: The two measures the kind computes are declared on the functions computing them (``kind.py``); the four it
#: does not are declared here.
TOYHOST_MEASURES: tuple[MetricDescriptor, ...] = (
    field_accuracy.descriptor,
    MetricDescriptor(
        name="cost_per_document_usd",
        reader_name="Cost per document",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Provider spend attributable to one document, end to end.",
        reader_prose="what one document cost to extract",
        higher_is_better=False,
        unit="usd",
        merit_axis="cost",
        population="all_observed",
        # A DECLARED materiality threshold, present for the reason the two opposite-direction bars
        # are: without one, every caveat in every toy-host fixture attaches by the conservative
        # default and the branch of the caveat-placement rule that leaves a below-threshold caveat
        # unattached never executes. A tenth of a cent
        # per document is the scale below which nobody reroutes a pipeline.
        materiality_threshold=0.001,
    ),
    MetricDescriptor(
        name="p95_extract_ms",
        reader_name="95th-percentile extraction time",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="95th-percentile wall-clock time from document received to fields emitted.",
        reader_prose="how long the slow documents took",
        higher_is_better=False,
        unit="ms",
        merit_axis="latency",
        population="all_observed",
    ),
    MetricDescriptor(
        name="manual_review_rate",
        reader_name="Manual review rate",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Share of documents the pipeline escalated to a human rather than completing.",
        reader_prose="how often the pipeline gave up and asked a person",
        higher_is_better=False,
        value_range=(0.0, 1.0),
        merit_axis="reliability",
        population="all_observed",
    ),
    # An UNBOUNDED COUNT, present because that is the shape a normal-approximation interval gets
    # wrong: a t-interval on a quantity with a hard floor at zero reached -7.7 and rendered
    # without complaint. This fixture makes one unavoidable.
    MetricDescriptor(
        name="fields_stripped",
        reader_name="Fields stripped",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        description="Count of fields dropped by the schema validator before the record was stored.",
        reader_prose="how many fields the validator threw away",
        higher_is_better=False,
        merit_axis="quality",
        population="all_observed",
    ),
    field_count_error.descriptor,
)

#: The incumbent standards — what the ratchet and the vacuous-seed flag hold against.
#:
#: **Two bars, in opposite directions, deliberately.** A ratchet that only ever sees
#: higher-is-better never executes its own lower-is-better branch, and that branch is where an
#: inverted comparison hides — a latency bar "tightened" to 5000ms would register as stricter than
#: 2000ms and nothing would notice. One direction in the fixture is one direction under test.
TOYHOST_BARS: tuple[Bar, ...] = (
    Bar(
        behavior="extract_invoice_fields",
        measure="field_accuracy",
        threshold=0.92,
        higher_is_better=True,
        rationale="below this the manual-review queue costs more than the pipeline saves",
    ),
    Bar(
        behavior="extract_invoice_fields",
        measure="p95_extract_ms",
        threshold=2000.0,
        higher_is_better=False,
        rationale="beyond this the upload UI times out before the fields come back",
    ),
)

#: A style deliberately unlike the default on every axis, so the prompt-purity test is asserting
#: something: a different register, a palette and a font of its own. If host style
#: could leak into the prompt, this profile is what would show it — and the palette is what a renderer
#: built for this host draws in, so it is a palette no packaged one shares a colour with.
TOYHOST_PALETTE = ChartPalette(
    series=("#f4a259", "#5b8e7d", "#bc4b51", "#8cb8d6", "#9a6b3a", "#3d5f54", "#7e3337", "#58788e"),
    sequential=("#f6e3c9", "#e9b97c", "#c98a3d", "#8a5a22"),
    background="#0b1021",
    ink="#f2efe6",
    muted="#a8a49a",
    grid="#262b3d",
    rule="#4a4f63",
    context="#6d6a72",
    on_fill="#0b1021",
)

#: The toy host's own chart face, declared with its metrics as every host font must be. A uniform
#: table, wider than the packaged face's lowercase, so a layout taken from it differs visibly from one
#: taken from the packaged table — which is what lets a test tell which table a chart was laid out in.
#: The family is no real face, so a raster falls through to the generic family; a real host measures
#: its face with ``packages/evals/scripts/measure_font_metrics.py`` and ships the directory beside it.
TOYHOST_FONT = ChartFont(
    family="Toyface Grotesk, sans-serif",
    advances=dict.fromkeys(CHART_FONT_CHARACTERS, 0.7),
    fallback_advance=0.7,
)

TOYHOST_STYLE = StyleProfile(tone_register="executive", chart_palette=TOYHOST_PALETTE, chart_font=TOYHOST_FONT)

#: A caveat kind the ENGINE does not own, registered for the reason the two opposite-direction
#: bars are registered: the four engine kinds were derived from one product's caveats, and a
#: fixture that only ever uses them proves nothing about whether the field is open. Adjudication
#: scope is a real fifth kind for this domain — which invoices the human key covers is neither
#: apparatus, sampling, instrument nor scope — and forcing it into `scope` is exactly the
#: collapse a closed enum on a host-facing field causes.
TOYHOST_CAVEAT_KINDS: frozenset[str] = frozenset({"adjudication_scope"})


def toyhost_profile(
    *,
    with_world: bool = True,
    optional_capabilities: bool = True,
    tunable_retrieval: bool = False,
    every_seat: bool = False,
) -> HostProfile:
    """The toy host's profile.

    Every call builds a **fresh** world, so a test that seeds one cannot leak into the next.

    Args:
        with_world: When False, omit the world entirely — the shape a consumer that evaluates
            production traffic has, whose representability is ``inapplicable`` rather than
            ``uncovered``. Note this is not the same as registering an empty world, which says
            the host HAS one and seeds nothing in it. Both shapes are real and the engine must
            not collapse them.
        optional_capabilities: When False, the same world registered without the optional host
            capabilities conformance can use — no perturbation, no way to fire a condition. The
            *hosts-without* half of the obligations table, and a real shape rather than a
            degraded one. See :func:`~packages.evals.tests.fixtures.toyhost.world.toyhost_world`.
        tunable_retrieval: When True, register retrieval tuning — an open family and the resolved
            configuration it is merged into (see ``packages.evals.tests.fixtures.toyhost.sweepables``). Off by
            default, so a suite that never sweeps retrieval carries no coordinate for it.
        every_seat: When True, the extractor kind declares no seats and is held to every one — the shape a
            suite exercising the judged variant (``packages.evals.tests.fixtures.toyhost.judge``) needs, whose
            rubric a model scores. Off by default: the standard extractor is graded by a comparison rule and
            fills no seat.

    Returns:
        A profile the engine cannot distinguish from any other host's except by its contents.
    """
    world, _state = toyhost_world(optional_capabilities=optional_capabilities)
    return HostProfile(
        host_id=TOYHOST_ID,
        host_sweepables=TOYHOST_TUNABLE_SWEEPABLE_REGISTRY if tunable_retrieval else TOYHOST_SWEEPABLE_REGISTRY,
        measures=MeasureRegistry(TOYHOST_MEASURES, families=(TOYHOST_EXTRACTION_FAMILY,)),
        bars=BarRegistry(TOYHOST_BARS),
        style=TOYHOST_STYLE,
        caveat_kinds=TOYHOST_CAVEAT_KINDS,
        world=world if with_world else None,
        variant_levers=tunable_variant_levers if tunable_retrieval else variant_levers,
        kinds=(replace(TOY_EXTRACTOR_CONTRACT, seats=None) if every_seat else TOY_EXTRACTOR_CONTRACT,),
    )


__all__ = [
    "FIELD_COUNT_ERROR",
    "TOYHOST_BARS",
    "TOYHOST_EXTRACTION_FAMILY",
    "TOYHOST_FONT",
    "TOYHOST_CAVEAT_KINDS",
    "TOYHOST_ID",
    "TOYHOST_MEASURES",
    "TOYHOST_PALETTE",
    "TOYHOST_STYLE",
    "toyhost_profile",
]
