"""The toy host's profile — five measures, two bars in opposite directions, a world, and a style unlike the default.

Each measure is chosen for a value-shape the engine has to handle: a bounded ratio, two unbounded quantities in opposite better-directions, and an
**unbounded count** — the class where a confidence interval reached −7.7 on a quantity that
cannot go below zero and shipped green.

All four merit axes are represented, which is the assertion behind the axis enum being
engine-owned and closed: a second product's measures land on the same four without being made to.
"""

from __future__ import annotations

from threetears.evals.contracts import MetricDescriptor
from threetears.evals.contracts.host import Bar, BarRegistry, Coverage, HostProfile, MeasureRegistry, StyleProfile
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT
from packages.evals.tests.fixtures.toyhost.sweepables import (
    TOYHOST_SWEEPABLE_REGISTRY,
    TOYHOST_TUNABLE_SWEEPABLE_REGISTRY,
)
from packages.evals.tests.fixtures.toyhost.variant import tunable_variant_levers, variant_levers
from packages.evals.tests.fixtures.toyhost.world import toyhost_world

#: Opaque to the engine, which never branches on it.
TOYHOST_ID = "toyhost"

TOYHOST_MEASURES: tuple[MetricDescriptor, ...] = (
    MetricDescriptor(
        name="field_accuracy",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Share of invoice fields extracted exactly right, against the adjudicated key.",
        reader_prose="how often the extractor got a field exactly right",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        merit_axis="quality",
        population="scored",
    ),
    MetricDescriptor(
        name="cost_per_document_usd",
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
#: something: a different register, a different locale, and a palette of its own. If host style
#: could leak into the prompt, this profile is what would show it.
TOYHOST_STYLE = StyleProfile(
    tone_register="executive",
    locale="en-GB",
    vega_config={"background": "#0b1021", "range": {"category": ["#f4a259", "#5b8e7d", "#bc4b51"]}},
)

#: A caveat kind the ENGINE does not own, registered for the reason the two opposite-direction
#: bars are registered: the four engine kinds were derived from one product's caveats, and a
#: fixture that only ever uses them proves nothing about whether the field is open. Adjudication
#: scope is a real fifth kind for this domain — which invoices the human key covers is neither
#: apparatus, sampling, instrument nor scope — and forcing it into `scope` is exactly the
#: collapse a closed enum on a host-facing field causes.
TOYHOST_CAVEAT_KINDS: frozenset[str] = frozenset({"adjudication_scope"})


#: What this host does not have, in its own words — the invoice extractor grades with code and talks
#: to nobody.
#:
#: Five of the shared core's seven apparatus dimensions are not absences here, they are states this
#: host genuinely occupies. It DOES have a grader: `field_accuracy` is scored against an adjudicated
#: key, and `reviewer_pool` records which pool of humans adjudicated it. What it does not have is a
#: MODEL grader, and the core's `judge_model` holds a model id whose blank is documented to mean the
#: judge is unrecoverable — so there is nothing legal for this host to put there, and reading its
#: blank as an absence fabricated a confound in every bundle.
#:
#: `judge_config_ids` is deliberately NOT here either, and for a different reason from
#: `max_cost_usd`'s: its READER already answers correctly. `_judge_config_ids` returns the
#: `NO_JUDGE_CONFIGS` sentinel for results that were read and carried none, which is a recorded
#: level rather than an absence — it is the one core judge axis that never needed this map, and the
#: issue that produced this map names it as the control. Declaring it inapplicable would replace a
#: working observation with a claim.
#:
#: `max_cost_usd` is deliberately NOT here. The extractor calls a model, so it plausibly has a spend
#: ceiling; that one is a real fixture gap and is recorded by the observations instead. Declaring a
#: real gap inapplicable is how a detector gets quietly switched off — the same failure as reading a
#: recorded level as an absence, arriving from the other side.
TOYHOST_INAPPLICABLE_APPARATUS: dict[str, Coverage] = {
    "judge_model": Coverage(
        "inapplicable",
        "nothing this host produces is scored by a model — field accuracy is computed against an "
        "adjudicated key, and which pool adjudicated it is recorded as `reviewer_pool`",
    ),
    "judge_dim_divergence": Coverage(
        "inapplicable",
        "there is no judge pin to diverge from: the grader is a comparison rule, applied to every field the same way",
    ),
    "judge_request_settings": Coverage(
        "inapplicable",
        "no model grades anything here, so no judge request is ever sent with a cap or a reasoning budget",
    ),
    "simulator_model": Coverage(
        "inapplicable",
        "the subject reads a batch of scanned invoices; no conversation happens, so nobody plays the other side of one",
    ),
    "simulator_request_settings": Coverage(
        "inapplicable",
        "nobody plays the other side of a conversation here, so no simulated-user request is ever sent",
    ),
}


def toyhost_profile(
    *, with_world: bool = True, optional_capabilities: bool = True, tunable_retrieval: bool = False
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

    Returns:
        A profile the engine cannot distinguish from any other host's except by its contents.
    """
    world, _state = toyhost_world(optional_capabilities=optional_capabilities)
    return HostProfile(
        host_id=TOYHOST_ID,
        host_sweepables=TOYHOST_TUNABLE_SWEEPABLE_REGISTRY if tunable_retrieval else TOYHOST_SWEEPABLE_REGISTRY,
        measures=MeasureRegistry(TOYHOST_MEASURES),
        bars=BarRegistry(TOYHOST_BARS),
        style=TOYHOST_STYLE,
        caveat_kinds=TOYHOST_CAVEAT_KINDS,
        apparatus_applicability=TOYHOST_INAPPLICABLE_APPARATUS,
        world=world if with_world else None,
        variant_levers=tunable_variant_levers if tunable_retrieval else variant_levers,
        kinds=(TOY_EXTRACTOR_CONTRACT,),
    )


__all__ = [
    "TOYHOST_BARS",
    "TOYHOST_CAVEAT_KINDS",
    "TOYHOST_ID",
    "TOYHOST_MEASURES",
    "TOYHOST_STYLE",
    "toyhost_profile",
]
