"""Measure registry — the single source of truth for what an eval measure *is*.

Every number the eval system produces is a measure, and a surface that renders one
needs to know more than its value: is it numeric or categorical, does higher mean
better, what unit is it in, is it raw or derived (and from what formula), and — the
question that decides whether two runs may be pooled — how far its meaning travels.

Read surfaces render from these descriptors rather than switching on scenario type,
so a new measure becomes visible by being described here instead of by editing every
page that might show it.

Three axes, deliberately independent:

``family``
    What kind of measure it is. Families are distinct and non-mergeable: a mechanical
    timing and a judged rubric score are not the same kind of thing even when both are
    numbers on 0-1.

``transferability_class``
    How far the measure's meaning carries, ordered loosest to strictest as
    ``mechanical`` < ``judge_mediated`` < ``scenario_bound``. ``mechanical`` is measured
    the same way everywhere (milliseconds are milliseconds); ``judge_mediated`` depends
    on a model's opinion, so it is comparable only under the same judge configuration;
    ``scenario_bound`` is defined by the scenario itself, so it means nothing outside it.
    Where a measure qualifies for more than one, the strictest wins — see
    :func:`strictest_class`.

``attribution_scope``
    Whether the measure isolates one subsystem or reflects the whole end-to-end run.
    A regression in an ``end_to_end`` measure does not tell you which part moved.

Alongside the three, one relation between measures rather than a property of one:
``contained_by`` names the end-to-end whole a subsystem measure is a component of. It is
declared, never inferred from a shared unit — ``llm_ms`` nests inside the turn spans
``total_ms`` sums, while the drain wait and every async tool's phase are disjoint stretches of
wall-clock in the same unit. Only a declared part may be subtracted from its whole; the
default of "not known to be contained" is what keeps an unvetted pair from being differenced.
Declaring it is necessary and not sufficient: several measures may name the same whole, and a
remainder is only "unattributed" where the part EXHAUSTS the whole. ``total_ms`` partitions
into three components, so differencing one of them leaves the other two's movement — which
this catalog names, and is therefore attributed.

``family`` does not determine ``transferability_class``. Both ``pass_hat_k`` and
``mean_composite`` are ``composite`` family, but pass^k is scenario-defined while the
composite is judge-mediated — which is exactly why the class is recorded rather than
inferred.

**Part of the measure name space is open, so this module is a closed core plus a rule.**
Three name spaces are unbounded: goal-state checks (the measure name IS the expression) and
rubric dimension names, both operator-authored in the database, and phase-timing keys
(``<tool>_<phase>_ms``), which grow with each new tool or phase. Those are classified by
construction — :func:`describe_rubric_dim`, :func:`describe_goal_state`,
:func:`describe_phase_timing` — rather than enumerated here, because listing them would both
go stale and duplicate the authoring surface. Covariate keys are a fixed set the code emits
and ARE seeded.

Each of those takes the family from the CALLER, who knows it because of which field they
are iterating. Only :func:`describe_measure` resolves a bare name, and it guesses as little
as possible: an unrecognised name is held at the strictest class rather than sorted into a
family by its shape. Guessing from shape was tried and removed — run-summary statistics like
``async_delivery_p95_elapsed_ms`` look exactly like phase-timing keys, and pattern-matching them
produced a confident description of a phase that does not exist.

**The closed core is the ENGINE's half, and the host declares its own.** A measure whose
vocabulary belongs to a host's tool — what that tool calls a healthy conclusion, what its
grounding pass strips — is declared in that host's catalogue and reaches
:func:`describe_measure` through the host's measure registry, never from here. The split is not
cosmetic: this module is shared contract, so a name seeded here is a name every consumer of
the package inherits whether or not it has the thing being named.

What IS enumerated in :data:`METRIC_DESCRIPTORS`: the run-summary aggregates, the dimension
summaries, the three comparison surfaces, and the per-result measure carriers
(``RoleUsage``, ``LatencyMetrics``, ``AsyncDelivery``). That much is a *checked* property:
``TestTheCoreCoversWhatTheSurfacesPublish`` reads exactly those models and fails if one grows
a measure that neither this core nor the host's catalogue describes — so it is not a claim
maintained by hand, and it spans both halves rather than only this one.

One more is enumerated and is NOT covered by that guard, so it is named here: ``score``, the
raw 1-5 judge score for one rubric dimension. It is a field on ``RubricScore``, which the
guard skips as an open name space — but the open space is the DIMENSION names, not the
statistic recorded against them, exactly as with ``mean_score`` / ``min_score`` /
``max_score``. Seeding it is what makes ``score`` mean one thing on every surface that
mentions it, having been accepted by the pivot, emitted by the export, and absent here.

What is deliberately NOT enumerated, even though the code emits it: phase-timing keys, which
grow with every new tool and phase and are resolved by :func:`describe_phase_timing`. Read
this module as "complete for what the read surfaces publish", not "complete for every number
the system produces" — the second is not true and is not the goal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator

from threetears.evals.contracts.base import EvalBaseModel

# The reserved dual-score axis ids, re-exported so a caller describing a judged dimension
# has one import for both the ids and their descriptors.
from threetears.evals.contracts.models import OUTCOME_DIM_ID, SCALES, TRANSCRIPT_DIM_ID, RubricScale

if TYPE_CHECKING:
    # Type-only: the host's registry is BUILT from this module's descriptors, so importing it
    # at run time is a cycle. The annotation names the real class rather than a structural
    # stand-in, which would have been a second declaration of one shape.
    from threetears.evals.contracts.host.measures import MeasureRegistry

__all__ = [  # noqa: RUF022 — the sort deletes the note below, which is why the aliases are exported
    "OUTCOME_DIM_ID",
    "TRANSCRIPT_DIM_ID",
    "METRIC_DESCRIPTORS",
    "CODE_GRADED_FAMILIES",
    "CLASSIFIER_FAMILY",
    "CLASSIFIER_LABEL_MEASURE_PREFIX",
    "COMPOSITE_FAMILY",
    "ACCURACY_MEASURE",
    "CONFUSION_CELL_MEASURE",
    "CONFUSION_SEPARATOR",
    "DUAL_AXIS_FAMILY",
    "ENGINE_FAMILIES",
    "GOAL_STATE_FAMILY",
    "MATCH_MEASURE",
    "MECHANICAL_FAMILY",
    "RUBRIC_FAMILY",
    "ClassifierStatistic",
    "GradedBy",
    "MeasureFamily",
    "MeasurePopulation",
    "DELIVERED_AXES",
    "reads_turns",
    "summary_population",
    "Materiality",
    "classifier_label_measure",
    "classifier_label_of",
    "confusion_cell",
    "confusion_of",
    "describe_classifier_label",
    "is_code_graded",
    "materiality",
    "DERIVED_PER_RESULT_MEASURES",
    # The measure vocabulary is published, not internal: surfaces that carry a measure's
    # metadata onward (the analysis context bundle types these fields off the registry
    # rather than restating them as bare strings) need the aliases, not just the model.
    "AttributionScope",
    "MetricDataType",
    "MetricDescriptor",
    "MetricFamily",
    "TransferabilityClass",
    "GOAL_CHECK_MEASURE_PREFIX",
    "describe_goal_check_rate",
    "describe_goal_state",
    "describe_measure",
    "goal_check_measure",
    "goal_check_of",
    "describe_phase_timing",
    "containment_defects",
    "describe_rubric_dim",
    "list_metrics",
    "strictest_class",
]

#: What one observation of a measure is. ``boolean`` is a condition that held or did not, summarised
#: as a rate with an interval (never averaged into a percentile); ``text`` is words — a ruling, a
#: reason — listed as evidence and never aggregated at all.
MetricDataType = Literal["numeric", "categorical", "boolean", "text"]

#: A measure family's name. **Open**: the engine's own families are the named constants below, and a
#: host declares its own on its measure registry as a :class:`MeasureFamily`, saying whether code or a
#: judge produced the number. A descriptor naming a family neither declares is refused where the host
#: registers it, so a misspelled family is caught at startup rather than read as unclassified.
MetricFamily = str

#: Measured the same way everywhere by code: wall-clock, tokens, spend, counts.
MECHANICAL_FAMILY: MetricFamily = "mechanical"
#: A label compared against an expected one by code.
CLASSIFIER_FAMILY: MetricFamily = "classifier"
#: A goal-state check's verdict: code compared against what the candidate did.
GOAL_STATE_FAMILY: MetricFamily = "goal_state"
#: A judge's score on an authored rubric dimension.
RUBRIC_FAMILY: MetricFamily = "rubric"
#: The two reserved judge axes every judged run is scored on.
DUAL_AXIS_FAMILY: MetricFamily = "dual_axis"
#: A figure built over several results or runs: pass^k, the composite, a comparison's effect size.
COMPOSITE_FAMILY: MetricFamily = "composite"

#: Who produced a family's numbers. ``code`` is what the analysis generator may rank on and a bar on a
#: described measure may name; ``judge`` is a model's opinion and never ranks.
GradedBy = Literal["code", "judge"]

_FAMILY_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"

TransferabilityClass = Literal["mechanical", "judge_mediated", "scenario_bound"]

AttributionScope = Literal["subsystem", "end_to_end"]

#: The merit axes a measure can serve. **Engine-owned and closed**; which axis a given measure
#: serves is the host's declaration. Generic slot, host-declared content — the same split the
#: sweepables registry draws over levers. "Best" is never unqualified: a memo states position
#: along these, and a measure that names none contributes to no verdict.
MeritAxis = Literal["quality", "cost", "latency", "reliability"]

#: Which observations a measure is computed over. Two surfaces reporting the same measure name
#: over different populations produce figures that differ **by construction** over any corpus with
#: one excluded observation, so the name alone is not enough to license quoting them side by side.
#:
#: ``scored`` drops observations an apparatus fault produced; ``all_observed`` keeps every raw
#: row. The distinction is not a nicety: it is the difference between two ``mean_score`` figures
#: that a reader currently has to hold in their head, and that a second consumer would inherit
#: with no way to know.
#:
#: ``delivered`` is the turns the candidate took
#: (:func:`~threetears.evals.contracts.result_condition.delivered_a_turn`): it drops, beside the faults,
#: the candidate failures that took no turn — a call the model refused or errored on straight away. A
#: failure still counts against the arm wherever the arm is graded (a rate, a bar, a judged score), but a
#: refused call's round trip is not a turn's latency, and an arm whose every call was refused read as the
#: fastest and cheapest on the surface when it was averaged in. A failure that DID take a turn — one the
#: host's turn budget ended, the output cap cut, the cell's deadline struck mid-call, or a model failure
#: after turns the candidate had delivered (``EvalResult.turns_delivered``) — stays: its time and spend are
#: the arm's real cost of failing. Every cost or latency measure that declares no other
#: population is read over this one (:func:`summary_population`).
MeasurePopulation = Literal["scored", "all_observed", "delivered"]

#: The merit axes whose readings describe a turn the candidate took — what taking it cost, and how long it
#: took — and are therefore read over ``delivered`` unless the measure declares otherwise.
DELIVERED_AXES: frozenset[MeritAxis] = frozenset({"cost", "latency"})

#: The engine's blended spend, read over ``delivered`` like a cost-axis measure though it serves no axis. It
#: serves none only because it sums the judge's spend beside the candidate's — what it cost to MEASURE a
#: result, not what the candidate costs — and that does not make a refused call's spend a turn's.
_DELIVERED_SPEND = "cost_usd"


def reads_turns(descriptor: MetricDescriptor) -> bool:
    """Whether a measure describes a turn's time or spend — a cost or latency axis, or ``cost_usd``.

    The one membership test for the measures ``delivered`` is for: :func:`summary_population` reads them over
    it, and :class:`MetricDescriptor` refuses ``delivered`` declared on any other.
    """
    return descriptor.merit_axis in DELIVERED_AXES or descriptor.name == _DELIVERED_SPEND


# Loosest to strictest. `strictest_class` relies on this ordering, and the lint that
# refuses a declaration looser than its family's structural floor (a separate concern,
# not built here) will read the same order.
_CLASS_STRICTNESS: dict[TransferabilityClass, int] = {
    "mechanical": 0,
    "judge_mediated": 1,
    "scenario_bound": 2,
}


class MeasureFamily(EvalBaseModel):
    """One family of measures and who produces its numbers — the engine's six, or one a host declares.

    A host declares a family when its measures are a kind of number none of the engine's families
    names: a table's play-quality reading, a reviewer's tally. Declaring it on the host's
    :class:`~threetears.evals.contracts.host.measures.MeasureRegistry` is what lets a descriptor name
    it; ``graded_by`` is what decides whether its measures may be ranked on.
    """

    name: MetricFamily = Field(
        pattern=_FAMILY_NAME_PATTERN, description="The family's name, as a descriptor's `family` spells it."
    )
    graded_by: GradedBy = Field(
        description="`code` when code produced every number in the family; `judge` when a model's opinion did."
    )
    description: str = Field(min_length=1, description="What the family's measures are, in one sentence.")


#: The engine's own families, by name. A host family may not reuse one of these names.
ENGINE_FAMILIES: Mapping[MetricFamily, MeasureFamily] = {
    family.name: family
    for family in (
        MeasureFamily(
            name=MECHANICAL_FAMILY, graded_by="code", description="Measured by code the same way everywhere."
        ),
        MeasureFamily(
            name=CLASSIFIER_FAMILY, graded_by="code", description="A label compared against an expected one."
        ),
        MeasureFamily(name=GOAL_STATE_FAMILY, graded_by="code", description="A goal-state check's verdict."),
        MeasureFamily(name=RUBRIC_FAMILY, graded_by="judge", description="A judge's score on a rubric dimension."),
        MeasureFamily(name=DUAL_AXIS_FAMILY, graded_by="judge", description="The two reserved judged axes."),
        MeasureFamily(
            name=COMPOSITE_FAMILY,
            graded_by="judge",
            description="A figure built over several results — pass^k, the composite, an effect size.",
        ),
    )
}


class MetricDescriptor(EvalBaseModel):
    """What a single measure is, independent of any particular value of it."""

    name: str = Field(min_length=1, description="The measure's key as it appears on results and summaries.")
    data_type: MetricDataType | None = Field(
        default=None,
        description="None when the measure has never been described, so its type is genuinely unknown rather than assumed numeric.",
    )
    family: MetricFamily | None = Field(
        default=None,
        pattern=_FAMILY_NAME_PATTERN,
        description=(
            "One of the engine's families or one the host declares on its measure registry. None when the measure "
            "has never been described. Missing is not 'mechanical' — see the unclassified arm of describe_measure."
        ),
    )
    transferability_class: TransferabilityClass
    attribution_scope: AttributionScope
    description: str = Field(min_length=1, description="One sentence an operator can read at point of use.")
    higher_is_better: bool | None = Field(
        default=None,
        description="None when the measure has no better direction (a coordinate, a condition, or a raw count).",
    )
    value_range: tuple[float, float] | None = Field(
        default=None,
        description="Inclusive numeric bounds, when the measure has them. None for unbounded or non-numeric measures.",
    )
    categories: tuple[str, ...] | None = Field(
        default=None,
        description="The closed value set for a categorical measure. None when the value set is open or the measure is not categorical.",
    )
    unit: str | None = Field(
        default=None, description="Unit of the value (e.g. 'ms', 'usd', 'tokens'). None when unitless."
    )
    materiality_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "The magnitude, in this measure's own units, below which a difference in it is too small to act on. "
            "A difference below it is labelled immaterial wherever the engine states one (`materiality`). It is "
            "also the measure's one declared margin: a bar on it is read against its threshold less this, and a "
            "run-history step reads `equivalent` only when an equivalence test shows the move inside it. None "
            "means the host declared none, and every difference is then material — a bar is held at its "
            "threshold and no step can read `equivalent`. Silence stays conservative, and the cost of not "
            "declaring one is paid in attention rather than banked as a permanent banner."
        ),
    )
    formula: str | None = Field(
        default=None,
        description="How a derived measure is computed, shown at point of use. None for raw captured measures.",
    )
    reader_prose: str | None = Field(
        default=None,
        description=(
            "The reader-facing sentence rendered prose prefers to the measure's key. None when the measure's "
            "'description' already reads as that sentence, which rendered prose then uses."
        ),
    )
    merit_axis: MeritAxis | None = Field(
        default=None,
        description=(
            "Which merit axis this measure serves. The axis vocabulary is the engine's and closed; the assignment "
            "is the declarer's — the host for its own measures, the engine for the core ones it emits. None when "
            "the measure serves no axis and contributes to no verdict."
        ),
    )
    population: MeasurePopulation | None = Field(
        default=None,
        description=(
            "Which observations this measure is computed over. Every summary of it — a cell's, a bar's, a run's — "
            "is computed over exactly that population and states which, so two populations are never pooled under "
            "one name. None leaves it to the surface: the decision surface's cells and bars read `scored`, a run's "
            "summary and the rollups `all_observed`, and each summary says which it used. A cost or latency measure, "
            "and the engine's blended spend `cost_usd`, is read over `delivered` on every surface unless it declares "
            "`all_observed`: the turns the candidate took, leaving out the failures that took none (a refusal, a "
            "model error), whose round trip and spend are no turn's. `delivered` may be declared only on such a "
            "measure."
        ),
    )
    contained_by: str | None = Field(
        default=None,
        description=(
            "The end-to-end measure this one is a COMPONENT of, when it is. Sharing a unit is not "
            "containment: a part and a whole in milliseconds may still be disjoint stretches of "
            "wall-clock, in which case subtracting one from the other describes nothing. None means "
            "'not known to be contained', which is where every measure starts — a subtraction is "
            "earned by declaring the whole, never by matching units."
        ),
    )
    diagnostic: bool = Field(
        default=False,
        description=(
            "True for a measure reported so a reader can EXPLAIN a movement — the conditions a measurement was "
            "taken under, or a signed error with no better end — and never ranked on or held to a bar. A "
            "directionless numeric measure reaches the analysis surfaces only when it says this: one that does "
            "not is a raw count, and a raw count stays out. Requires `higher_is_better` to be None, since a "
            "direction is exactly what would let a ranking read the diagnostic as a merit."
        ),
    )

    @model_validator(mode="after")
    def _delivered_is_for_a_turns_time_or_spend(self) -> MetricDescriptor:
        """Refuse ``population="delivered"`` on a measure that is not a turn's time or spend.

        ``delivered`` leaves out the failures that took no turn. On a quality or reliability reading that
        is exactly the inflation it exists to prevent elsewhere: a refusing arm's refusals would vanish from
        its accuracy, and it would read better for refusing.
        """
        if self.population == "delivered" and not reads_turns(self):
            raise ValueError(
                f"{self.name} declares population 'delivered', which is for a turn's time or spend — a cost or "
                f"latency measure, or cost_usd — and {self.name} is on {self.merit_axis or 'no'} axis; a failure "
                "must count against the arm on any other reading, so declare `scored` or `all_observed`"
            )
        return self

    @model_validator(mode="after")
    def _a_diagnostic_has_no_better_end(self) -> MetricDescriptor:
        """Refuse a diagnostic that declares a better end, which would let a ranking read it as a merit."""
        if self.diagnostic and self.higher_is_better is not None:
            raise ValueError(
                f"{self.name} is declared a diagnostic and higher_is_better={self.higher_is_better}: a diagnostic "
                "explains a movement and has no better end, so either drop the direction or drop `diagnostic`"
            )
        return self


def strictest_class(*classes: TransferabilityClass) -> TransferabilityClass:
    """Return the strictest of the given transferability classes.

    A measure can qualify for more than one class — the dual-score outcome axis is both
    judge-produced and scenario-defined. Taking the strictest is the conservative choice:
    it can only ever refuse a pooling that would have been allowed, never permit one that
    should have been refused.

    Args:
        *classes: One or more transferability classes.

    Returns:
        The strictest class among them.

    Raises:
        ValueError: If no classes are given — there is no honest answer to return.
    """
    if not classes:
        raise ValueError("strictest_class() requires at least one class")
    return max(classes, key=lambda c: _CLASS_STRICTNESS[c])


#: The range any rubric score can take, over every scale — derived from :data:`SCALES`, so a
#: statistic pooled across dimensions (``score``, ``mean_score`` …) is bounded by the scales that
#: exist rather than by the 1-5 one alone. A single dimension's own range is
#: :func:`describe_rubric_dim`'s, which knows its scale.
_ANY_SCALE_RANGE: tuple[float, float] = (
    min(spec.value_range[0] for spec in SCALES.values()),
    max(spec.value_range[1] for spec in SCALES.values()),
)

#: How a raw score reads, per scale — derived from :data:`SCALES` for every per-dimension descriptor.
_ON_ITS_OWN_SCALE = (
    "on the dimension's own scale (" + "; ".join(f"{name}: {spec.reads_as}" for name, spec in SCALES.items()) + ")"
)

#: The composite's per-scale normalisation, derived from :data:`SCALES` so the formula names every scale.
_NORMALISED = "; ".join(
    f"{name}: (score - {spec.scores[0]}) / {spec.scores[1] - spec.scores[0]}" for name, spec in SCALES.items()
)


def _d(**kwargs: Any) -> MetricDescriptor:
    """Build a descriptor; a local alias so the seed table below stays readable."""
    return MetricDescriptor(**kwargs)


# =============================================================================
# The closed core — every measure the code itself emits.
#
# What is NOT here: the two open NAME spaces — rubric dimension names and goal-state
# expressions — resolved by `describe_rubric_dim` / `describe_goal_state`, and phase-timing
# keys, resolved by `describe_phase_timing`. The five covariate keys ARE seeded below (they
# are a fixed set the code emits), so there is no covariate resolver.
# =============================================================================


def _compare_trio(base: str, label: str, cls: TransferabilityClass) -> tuple[MetricDescriptor, ...]:
    """Build the A / B / delta descriptors the run-comparison surface publishes for one measure.

    Enumerated at import rather than inferred from an ``_a`` / ``_b`` / ``_delta`` suffix at
    call time: the names end up in :data:`METRIC_DESCRIPTORS` as ordinary static entries, so a
    lookup never has to guess what a suffix means, and an operator-authored measure that
    happens to end in ``_delta`` cannot be swept into this family.

    The delta is deliberately NOT a copy of the base: it is signed, its range is the base's
    span rather than the base's bounds, and "higher is better" only holds because it is
    oriented B minus A.

    Args:
        base: The underlying measure name (e.g. ``composite``).
        label: Human phrase for the measure, used in the generated descriptions.
        cls: The base measure's transferability class, carried to all three.

    Returns:
        The ``_a``, ``_b`` and ``_delta`` descriptors.
    """
    side = tuple(
        MetricDescriptor(
            name=f"{base}_{suffix}",
            data_type="numeric",
            family="composite",
            transferability_class=cls,
            attribution_scope="end_to_end",
            higher_is_better=True,
            value_range=(0.0, 1.0),
            description=f"{label} for run {run}. Null when that run did not exercise the model or template.",
        )
        for suffix, run in (("a", "A"), ("b", "B"))
    )
    delta = MetricDescriptor(
        name=f"{base}_delta",
        data_type="numeric",
        family="composite",
        transferability_class=cls,
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(-1.0, 1.0),
        formula=f"{base}_b - {base}_a",
        description=(
            f"Change in {label.lower()} from run A to run B. Signed and oriented B minus A, so positive "
            "means B improved; it says nothing about whether the change is larger than noise (see significant)."
        ),
    )
    return (*side, delta)


_SEED: tuple[MetricDescriptor, ...] = (
    # ---- Latency (mechanical, from OTel spans) ------------------------------
    # The turn partition — total_ms and its three parts — carries the latency merit axis: it is the
    # candidate's own wall-clock. judge_ms and async_wait_ms do not: the judge is measurement, not
    # the candidate, and the drain wait is disjoint from the turns by construction.
    _d(
        name="total_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        merit_axis="latency",
        higher_is_better=False,
        unit="ms",
        description=(
            "Wall-clock across the turn roots. Partitions exactly into llm_ms + tool_ms + "
            "orchestration_ms, the third being the named remainder rather than an unexplained gap. "
            "Covers the candidate's turns only: the judge scores after the turns end, so judge_ms is "
            "measured beside this rather than inside it, and neither it nor async_wait_ms belongs to "
            "the partition."
        ),
    ),
    _d(
        name="llm_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        merit_axis="latency",
        higher_is_better=False,
        unit="ms",
        contained_by="total_ms",
        description="Time inside model calls — the model-attributable share of latency.",
    ),
    _d(
        name="tool_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        merit_axis="latency",
        higher_is_better=False,
        unit="ms",
        contained_by="total_ms",
        description="Time inside tool executions.",
    ),
    _d(
        name="orchestration_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        merit_axis="latency",
        higher_is_better=False,
        unit="ms",
        contained_by="total_ms",
        formula="total_ms - llm_ms - tool_ms",
        description=(
            "Turn wall-clock spent neither inside a model call nor inside a tool execution — "
            "assembling perceptions, building the prompt, parsing the decision, updating and "
            "persisting state. The named remainder that closes the total_ms partition: it exists so "
            "a whole-run latency movement can be PLACED, since a total that moves while both named "
            "parts stay flat is otherwise indistinguishable from a measurement error. Derived per "
            "result and never stored — a persisted copy would be a second answer to a question the "
            "three components already settle. Absent unless all three are measured, and it is not a "
            "home for time that fell outside the turns: the drain wait, the judge phase and every "
            "tool's phase timing are disjoint from total_ms, and folding any of them in here is "
            "the arithmetic that once produced a ~95-second remainder describing no stretch of "
            "wall-clock at all."
        ),
    ),
    _d(
        name="async_wait_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=False,
        unit="ms",
        formula="summed across the result's drain windows",
        description=(
            "Wall-clock the runner spent waiting on in-flight background tool work before the "
            "candidate could deliver it. Deliberately NOT declared a component of total_ms: the wait "
            "sits between turns, outside every turn-root span, so it is disjoint from turn latency "
            "rather than part of it. Zero is a real observation (nothing was in flight); None means "
            "the cell never reached a drain."
        ),
    ),
    _d(
        name="judge_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=False,
        unit="ms",
        description=(
            "Wall-clock the judge phase took to score one result — every axis and dimension, "
            "including retries. Like the drain wait and for the same reason, NOT declared a "
            "component of total_ms: the judge scores a finished transcript after the turns end, so "
            "it is disjoint from turn latency rather than a share of it, and the two must never be "
            "differenced. Note the asymmetry with cost_usd, which DOES include judge spend: the "
            "run's two end-to-end measures cover different stretches of the same cell. Measured whenever "
            "the phase ran, a failed judge included (judge_error says the scoring failed, not the "
            "clock); None when no judge phase was timed."
        ),
    ),
    # ---- Spend --------------------------------------------------------------
    # Only production_replicating_cost carries the cost merit axis — it is the spend belonging to the
    # roles production runs. cost_usd and program_cost both include the judge, which is what it cost to
    # MEASURE the candidate, not what the candidate costs.
    _d(
        name="cost_usd",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False,
        unit="usd",
        description=(
            "Blended program spend for the result. Authoritative for spend; the per-role rows decompose "
            "it but never re-total it. WHAT IT SUMS VARIES: metered third-party spend is included only for a "
            "run whose operator declared a rate for it (cost_roles names what each result summed). So two of "
            "these are comparable only if they summed the same roles — the "
            "pooled cost surfaces disclose which sets they spanned, and a mean over a mixed pool is not "
            "one distribution."
        ),
    ),
    _d(
        name="production_replicating_cost",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        merit_axis="cost",
        higher_is_better=False,
        unit="usd",
        formula="sum of cost_usd over the candidate, inner_agent and external roles",
        description=(
            "Observed dollars belonging to the production roles — measurement-only roles excluded. This is "
            "the spent subset, NOT the counterfactual 'what running this candidate for real would cost': the "
            "two coincide only when the run substituted nothing, and a swept model understates while a "
            "memory-stripped candidate that does background work where production would recall overstates. "
            "Absent for a result carrying a seeded or replayed async DELIVERY, whose background-model dollars "
            "and metered units were never spent — unknown rather than free. A cassette replay at the ACTION "
            "seam (a synchronous metered tool) still reports a figure, and correctly: that tool's spend never "
            "enters these rows live either, so the replayed number equals the live one — what it saved is "
            "provider units this metric does not count in either mode. Metered calls contribute dollars only "
            "where the run resolved a rate for them, and only through an async delivery: units a synchronous "
            "tool spent are outside this figure entirely."
        ),
    ),
    _d(
        name="program_cost",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False,
        unit="usd",
        formula="sum of cost_usd over every role",
        description=(
            "What the eval program spent to measure the candidate, including judge and simulator. A floor for "
            "the same reason as production_replicating_cost, and for one more: a requeued background delivery "
            "drops its dollars."
        ),
    ),
    # ---- Tokens (raw counts: no better direction) ---------------------------
    _d(
        name="prompt_tokens",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="tokens",
        description="Input tokens a role consumed. None rather than zero when the provider reported no count.",
    ),
    _d(
        name="completion_tokens",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="tokens",
        description="Output tokens a role produced.",
    ),
    _d(
        name="reasoning_tokens",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="tokens",
        description="The reasoning share of completion_tokens. Absent when the provider reported no split; zero only when it reported zero.",
    ),
    _d(
        name="call_count",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="calls",
        description=(
            "How many calls a role made — and it counts a different thing per role: LLM rounds for the "
            "candidate, attempts including parse retries for the judge, searches for external."
        ),
    ),
    _d(
        name="provider_units",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="provider-defined",
        formula="summed from what each caller REPORTED consuming; never re-derived from a rate",
        description=(
            "Units of a provider's own metered quantity that an external role consumed — a search API's "
            "credits, a video API's quota units — named by the row's provider_unit. The unit those "
            "providers actually meter in, so it is comparable across runs whose accounts pay "
            "different dollar rates, and present even on a run whose rate was never declared, "
            "where it is the only quantity the spend can be reconstructed from by hand. "
            "Comparable ONLY within one (provider, provider_unit): two providers' units are not "
            "one quantity and a figure adding them is fabricated, not approximate. Absent on "
            "token-metered roles and wherever the count was not knowable."
        ),
    ),
    _d(
        name="context_tokens_in",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="tokens",
        formula="sum of prompt_tokens over the candidate role's rows",
        description="How much context the candidate was carrying — a stratification covariate, not a quality measure.",
    ),
    _d(
        name="reasoning_ratio",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        value_range=(0.0, 1.0),
        formula="candidate reasoning_tokens / candidate completion_tokens",
        description="How much of the candidate's output was reasoning. Absent unless both halves were measured.",
    ),
    _d(
        name="candidate_output_tokens_per_s",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        # No better end, deliberately: a faster provider hour is not a better arm, and a direction
        # here is what would let a superlative or a ranking read provider drift as a lever's merit —
        # the misreading this measure exists to expose. It is a DIAGNOSTIC, declared on the
        # descriptor exactly as a host declares one of its own.
        higher_is_better=None,
        diagnostic=True,
        unit="tokens/s",
        formula="sum of completion_tokens over the candidate role's rows / (llm_ms / 1000)",
        description=(
            "How fast the candidate's provider produced output, over the time spent inside its model calls. "
            "A provider-side signal rather than a property of any lever: a model served slower this hour "
            "stretches llm_ms, and therefore total_ms, for every arm run in that hour. So when a whole-run "
            "latency difference between arms is carried by llm_ms, compare this first — arms measured at "
            "different times whose throughput differs by the same factor moved with the provider, not with "
            "what they tuned. Derived per result and never stored; absent unless both the candidate's output "
            "tokens and a non-zero llm_ms were measured. Reasoning tokens are output and count here. "
            "A diagnostic: reported so a latency movement can be explained, never ranked on and never a bar."
        ),
    ),
    _d(
        name="dropped_tool_calls",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="calls",
        formula="count of TurnRecord.dropped_tool_calls over the cell's candidate turns",
        description=(
            "How many times the candidate emitted a tool call the client dropped before dispatch — a "
            "name that did not parse, so nothing ran and the model was not told. Reaching for a tool a "
            "template's tools_allowed withheld is signal about the candidate, and 0 is a real "
            "observation here: it says the turns were watched and nothing was dropped."
        ),
    ),
    _d(
        name="refused_tool_attaches",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="requests",
        formula="count of TurnRecord.refused_tool_attaches over the cell's candidate turns",
        description=(
            "How many times the candidate asked to attach a tool outside the run's tools_allowed "
            "and the harness refused. The refusal is not neutral under measurement — the candidate "
            "spends a turn asking, is told no, and re-plans — so scoring the re-plan as ordinary "
            "conduct without it hides that the harness intervened. 0 is a real observation here: "
            "it says the turns were watched and nothing was refused."
        ),
    ),
    _d(
        name="truncated_rounds",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=False,
        unit="rounds",
        formula="sum of TurnRecord.truncated_rounds over the cell's candidate turns",
        description=(
            "How many of the candidate's LLM rounds the provider cut off at its output cap "
            "(its output reached the cap, or the provider reported finish_reason=length) — the turn's silence or half-answer is the cap's, not the model's "
            "decision, so every quality measure on the cell is confounded by it; a reasoning model that "
            "spends the cap thinking shows here with reasoning_ratio near 1. Lower is better because a "
            "cut round is never the candidate's answer. 0 is a real observation here: it says the turns "
            "were watched and none was cut."
        ),
    ),
    _d(
        name="turns_ended_by_budget",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False,
        unit="turns",
        formula="candidate turns the host's turn budget cancelled before they finished",
        description=(
            "How many of the candidate's turns outlived the host's per-turn budget — the bound the host "
            "runs every turn under in production — and were ended by it. An ended turn delivers nothing, "
            "whatever its rounds had decided, so any count above 0 makes the result a "
            "candidate failure. 0 is a real observation: the turns ran under the budget and none outlived "
            "it; absent means no budget bounded them."
        ),
    ),
    # ---- Conditions (categorical covariates) --------------------------------
    _d(
        name="execution_mode",
        data_type="categorical",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        categories=("serial", "concurrent"),
        description="Whether the observation was made while other eval jobs were running — a condition, not a result.",
    ),
    # ---- Run-summary aggregates (RunSummaryRow / DimensionSummaryRow) -------
    # These are the names the run-summary surfaces actually publish. They are
    # code-emitted, so they belong in the closed core — a renderer that meets one and
    # falls through to the unclassified arm is the closed-core contract (every name the engine
    # emits is classified) failing in the one place it is most visible.
    _d(
        name="n_results",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="results",
        description="How many results the group aggregates — the denominator behind its means.",
    ),
    _d(
        name="n_total_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="results",
        description=(
            "How many results contributed a measured total_ms — the denominator behind mean/median/p95 total_ms, "
            "which can be smaller than n_results because latency components are nulled independently."
        ),
    ),
    _d(
        name="n_llm_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="results",
        description="How many results contributed a measured llm_ms — the denominator behind mean_llm_ms.",
    ),
    _d(
        name="n_tool_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="results",
        description="How many results contributed a measured tool_ms — the denominator behind mean_tool_ms.",
    ),
    _d(
        name="n_cost_usd",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="results",
        description=(
            "How many results' spend was priced — the denominator behind mean_cost_usd. Smaller than "
            "n_results by however many went unpriced (a model call whose client reported no price), which "
            "is the number to read before comparing two configs' program cost."
        ),
    ),
    _d(
        name="n_prod_cost_usd",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="results",
        description=(
            "How many results measured a production-replicating cost — the denominator behind "
            "mean_prod_cost_usd. Smaller than n_results by however many contributed nothing, which is the "
            "number to read before comparing two configs' prod cost."
        ),
    ),
    _d(
        name="n_test_cases",
        data_type="numeric",
        family="mechanical",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        unit="cases",
        description="How many test cases the model was run against. Scenario-bound: two runs' counts are comparable only over the same template set.",
    ),
    _d(
        name="k",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="iterations",
        description=(
            "The depth the headline pass_hat_k is read at. On a run summary it is the deepest iteration this "
            "model attempted in the run (the run's planned k once its matrix finished), counting cells excluded "
            "as harness failures; a comparison reads every side at one shared depth. The cases actually scored "
            "that deep are n_cases_at_k."
        ),
    ),
    _d(
        name="n_cannot_tell_excluded",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="iterations",
        description=(
            "Iterations left out of pass^k because the judge answered it could not score a rubric "
            "dimension from the evidence. Not a harness fault: counted apart so a smaller denominator "
            "says why it shrank."
        ),
    ),
    _d(
        name="n_cases_at_k",
        data_type="numeric",
        family="mechanical",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        unit="cases",
        description=(
            "The cases pass_hat_k averages over: those scored at least k times. The estimator is undefined "
            "for a case measured fewer times, so a run stopped part-way leaves its shallow cases out of the "
            "deep points rather than letting one passed attempt stand in for k."
        ),
    ),
    _d(
        name="mean_total_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False,
        unit="ms",
        formula="mean of total_ms over results whose total_ms was measured, excluding cells a harness failure produced",
        description=(
            "Average end-to-end wall-clock over the cells that MEASURE the candidate — a cell an apparatus fault "
            "produced is excluded, since a clock stopped by a cassette miss or a judge error times the harness. "
            "A result whose turn-root span was never harvested is likewise absent from the mean, not counted as "
            "zero — components are nulled independently, so this mean and mean_llm_ms can rest on different "
            "sample sizes."
        ),
    ),
    _d(
        name="median_total_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False,
        unit="ms",
        formula="nearest-rank median of total_ms, excluding cells a harness failure produced",
        description=(
            "Typical end-to-end wall-clock, less sensitive to one slow outlier than the mean. Over the same "
            "candidate-measuring population as mean_total_ms: a cell an apparatus fault produced is excluded."
        ),
    ),
    _d(
        name="p95_total_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False,
        unit="ms",
        formula="nearest-rank 95th percentile of total_ms, excluding cells a harness failure produced",
        description=(
            "Tail end-to-end wall-clock. Nearest-rank, so on small n it is an actual observed value. Over the same "
            "candidate-measuring population as mean_total_ms: a cell an apparatus fault produced is excluded."
        ),
    ),
    _d(
        name="mean_llm_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=False,
        unit="ms",
        formula="mean of llm_ms over results whose llm_ms was measured, excluding cells a harness failure produced",
        description=(
            "Average model-attributable latency — the share a model swap can move. Over the same "
            "candidate-measuring population as mean_total_ms: a cell an apparatus fault produced is "
            "excluded, since the clock it stopped was timing the harness. Measured independently of "
            "total_ms, so the two can still rest on different numbers of results."
        ),
    ),
    _d(
        name="mean_tool_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=False,
        unit="ms",
        formula="mean of tool_ms over results whose tool_ms was measured, excluding cells a harness failure produced",
        description=(
            "Average time inside tool executions. Over the same candidate-measuring population as "
            "mean_total_ms: a cell an apparatus fault produced is excluded."
        ),
    ),
    _d(
        name="total_cost_usd",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False,
        unit="usd",
        formula="sum of cost_usd over the group's priced results",
        description=(
            "What the group's priced results cost in total: each result's cost_usd is a sum over the roles "
            "its cost_roles names. A result whose spend went unpriced is OMITTED rather than counted as zero "
            "— read n_cost_usd for how many this sums over; absent entirely when none was priced. A result "
            "the cell deadline stopped holds none of the call in flight when it struck, so for those the "
            "total is a floor; termination says which they are."
        ),
    ),
    _d(
        name="mean_cost_usd",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False,
        unit="usd",
        formula="total_cost_usd / n_cost_usd",
        description=(
            "Average program spend per PRICED result (blended, incl. judge + simulator). Its denominator is "
            "n_cost_usd, not n_results: a result whose spend went unpriced is absent from this mean rather than "
            "dragging it toward a zero nobody paid."
        ),
    ),
    _d(
        name="total_prod_cost_usd",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False,
        unit="usd",
        formula="sum of production_replicating_cost over the results that measured one",
        description=(
            "Observed spend on the production roles — candidate + inner_agent + external, excluding the "
            "judge/simulator measurement scaffolding. NOT what the group would cost to run in production: "
            "it sums dollars actually spent under whatever this run substituted, and a run that swept a "
            "model, overrode a tool config or a prompt, stripped learned memory, or seeded a delivery moved "
            "a knob production does not. The gap has no reliable sign, so this is not a floor either. A "
            "result with no usage decomposition, or one carrying a substituted delivery, is OMITTED rather "
            "than counted as zero — read n_prod_cost_usd for how many results this sums over; absent "
            "entirely when none did."
        ),
    ),
    _d(
        name="mean_prod_cost_usd",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False,
        unit="usd",
        formula="total_prod_cost_usd / n_prod_cost_usd",
        description=(
            "Average production-replicating spend per MEASURED result — the reporting default (what a config "
            "costs to run). Its denominator is n_prod_cost_usd, not n_results: a result with no usage "
            "decomposition is absent from this mean rather than dragging it toward a zero nobody observed, "
            "which would rank the least-measured config cheapest."
        ),
    ),
    _d(
        name="score",
        data_type="numeric",
        family="rubric",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=_ANY_SCALE_RANGE,
        description=(
            f"What one judged score on ONE rubric dimension of ONE result counts as, {_ON_ITS_OWN_SCALE}: the judge's "
            "score, the scale's floor where the candidate failed (a turn the output cap ended included), or null "
            "where the harness faulted the cell — the per-observation "
            "measure mean_score / min_score / max_score aggregate, emitted per judged dimension by "
            "export_results and accepted as a pivot metric. Its dimension is a COORDINATE "
            "(rubric_dim), never part of the name, so pivoting it without rubric_dim on an axis pools "
            "every dimension into one number: two dims that disagree average to a score neither was "
            f"given. NOT the composite, which rescales it to 0-1 ({_NORMALISED}) and averages across dims "
            "on 0-1 — a 4 here is not a 4 there. And not a history measure: a series carries one "
            "value per contestant per run, and this is per dimension, so history series mean_composite "
            "instead."
        ),
    ),
    _d(
        name="mean_score",
        data_type="numeric",
        family="rubric",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=_ANY_SCALE_RANGE,
        formula=f"mean of the counted scores for one rubric dimension, {_ON_ITS_OWN_SCALE} (the floor where the candidate failed; a harness fault excluded)",
        description=(
            "Average judged score for a single dimension — on its raw scale, NOT the 0-1 composite scale; "
            "on a pass/fail dimension it is the pass rate. "
            "Every surface averages the same counted values: a cell an apparatus fault produced is left out "
            "(a judge reading a broken transcript is not measuring the candidate), and a candidate failure "
            "counts at the scale's floor, whatever the judge made of what it left."
        ),
    ),
    _d(
        name="min_score",
        data_type="numeric",
        family="rubric",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=_ANY_SCALE_RANGE,
        description=f"Worst judged score observed for one rubric dimension, {_ON_ITS_OWN_SCALE}.",
    ),
    _d(
        name="max_score",
        data_type="numeric",
        family="rubric",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=_ANY_SCALE_RANGE,
        description=f"Best judged score observed for one rubric dimension, {_ON_ITS_OWN_SCALE}.",
    ),
    _d(
        name="n",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="scores",
        description="How many judged scores a dimension row aggregates.",
    ),
    # ---- Async-delivery lifecycle (run-summary rollup) -----------------------
    # The generic half of what an async PERCEIVE+ACT tool's deliveries can be counted for: how
    # many there were, how many of those a harness stood in for, and how long they took. What a
    # delivery's OUTCOMES are called — concluded cleanly, salvaged, timed out — is the tool's own
    # taxonomy, so the host declares those in its own catalogue and this module never sees them.
    _d(
        name="async_deliveries",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="deliveries",
        description=(
            "How many async deliveries the group made that describe a real run of the tool — not every "
            "delivery that happened, and the denominator any rate over them is taken against. A "
            "substituted delivery (one whose payload a harness supplied) describes no run at all, so it "
            "is excluded here and counted in async_deliveries_substituted; the two summed are the full "
            "population."
        ),
    ),
    _d(
        name="async_deliveries_substituted",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        # No better direction, and the registry reserves None to say so. Declaring
        # False would make a correctly-configured seeded eval read as a REGRESSION
        # against a live one on the movement comparison — the arm is a property of
        # the apparatus, not a quality the run is trying to minimise.
        higher_is_better=None,
        unit="deliveries",
        description=(
            "Deliveries EXCLUDED from async_deliveries because a harness supplied their payload (a "
            "seeded payload or a replayed capture). Never summed with a host's outcome buckets: "
            "these describe no run of the tool at all, so counting them would report a mechanism nobody "
            "chose. Present so async_deliveries reads as 'the deliveries those rates describe' rather "
            "than 'the deliveries that happened' — the gap between the two is exactly this number."
        ),
    ),
    _d(
        name="async_delivery_mean_elapsed_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=False,
        unit="ms",
        formula="mean of each delivery's elapsed_ms, over the deliveries that measured one",
        description="Average async delivery duration. A statistic over deliveries, NOT a phase timing.",
    ),
    _d(
        name="async_delivery_median_elapsed_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=False,
        unit="ms",
        formula="nearest-rank median of each delivery's elapsed_ms",
        description="Typical async delivery duration.",
    ),
    _d(
        name="async_delivery_p95_elapsed_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=False,
        unit="ms",
        formula="nearest-rank 95th percentile of each delivery's elapsed_ms",
        description="Tail async delivery duration.",
    ),
    _d(
        name="async_delivery_elapsed_n",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="deliveries",
        description="How many deliveries had a measured duration — absent rather than zero when none did.",
    ),
    # ---- Significance (stats.py) --------------------------------------------
    _d(
        name="hedges_g",
        data_type="numeric",
        family="composite",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        formula=(
            "J(df) x paired mean(diff)/sd(diff) (g_z, df = pairs - 1), else J(df) x pooled-SD standardized difference "
            "(g, df = n_a + n_b - 2); J is Hedges' small-sample factor; signed B minus A"
        ),
        description=(
            "Effect size of a composite difference between two runs: Hedges' g, Cohen's d with its small-sample "
            "upward bias removed. Unbounded and signed; None when undefined, including at two pairs, where no "
            "unbiased estimate exists."
        ),
    ),
    _d(
        name="significant",
        data_type="boolean",
        family="composite",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        description="Whether the composite difference cleared p < 0.05. Says the difference is unlikely to be noise, not that it is large.",
    ),
    _d(
        name="p",
        data_type="numeric",
        family="composite",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        value_range=(0.0, 1.0),
        formula="two-sided t-test p on the two runs' per-case composite samples",
        description="The p-value the significance verdict was thresholded against. Carried with the verdict so a reader can check it rather than take it; absent when no test was defined, never 1.0 as a stand-in for that.",
    ),
    _d(
        name="paired",
        data_type="boolean",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Whether the significance test paired the two runs' composites by test case. A shared template makes pairing possible; only cases scored in both runs make it happen, and the two effect sizes are not interchangeable.",
    ),
    _d(
        name="n_pairs",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="cases",
        description="How many test cases were scored in BOTH runs — the paired test's sample size. Absent when the samples were not paired, because an unpaired test has two sample sizes and no pair count.",
    ),
    _d(
        name="n_cases",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="cases",
        description="How many test cases contributed a composite — the pairing atom behind significance.",
    ),
    # ---- Composites ---------------------------------------------------------
    _d(
        name="pass_hat_k",
        data_type="numeric",
        family="composite",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        formula="mean over cases scored at least k times of C(c, k) / C(n, k), n scored attempts and c passes",
        description=(
            "pass^k (τ-bench): the chance that k attempts at a case ALL pass — never pass@k, the chance that at "
            "least one does. An attempt passes only if it cleared every rubric dimension and every goal-state "
            "check. Unbiased at any depth; infra-excluded attempts count toward no case's n."
        ),
    ),
    _d(
        name="mean_composite",
        data_type="numeric",
        family="composite",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        formula=f"mean over cases of the per-case mean, across the result's rubric dims, of each score normalised to 0-1 ({_NORMALISED})",
        description=(
            "Average judged quality, threshold-free — a regression often shows here before cases start "
            "failing pass^k. Comparable only across runs judged the same way."
        ),
    ),
    # ---- Dual-score axes -------------------------------------------------------
    _d(
        name=TRANSCRIPT_DIM_ID,
        data_type="numeric",
        family="dual_axis",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(1.0, 5.0),
        description=(
            "Decision quality given the context the candidate actually had. Stable when the outside world "
            "drifts, so a fall here is the candidate's own."
        ),
    ),
    _d(
        name=OUTCOME_DIM_ID,
        data_type="numeric",
        family="dual_axis",
        # Judge-produced AND scenario-defined; strictest wins.
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(1.0, 5.0),
        description=(
            "Whether the final state satisfied the scenario's intent. Drifts when externals drift — its "
            "divergence from the transcript axis is what separates a worse agent from a changed world."
        ),
    ),
    _d(
        name="mean_transcript_score",
        data_type="numeric",
        family="dual_axis",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(1.0, 5.0),
        formula="mean of the counted 1-5 transcript-axis scores (the floor where the candidate failed; a harness fault excluded)",
        description=(
            "Average decision quality given the context the candidate actually had — the aggregate an "
            "aggregating surface reports over __transcript__ rows, on the raw 1-5 scale and NOT the 0-1 "
            "composite scale. Read it beside mean_outcome_score: the two moving together is a candidate "
            "that changed, and this one holding while that one falls is a world that changed."
        ),
    ),
    _d(
        name="mean_outcome_score",
        data_type="numeric",
        # Judge-produced AND scenario-defined, as the per-observation axis is; strictest wins.
        family="dual_axis",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(1.0, 5.0),
        formula="mean of the counted 1-5 outcome-axis scores (the floor where the candidate failed; a harness fault excluded)",
        description=(
            "Average intent satisfaction — the aggregate an aggregating surface reports over __outcome__ "
            "rows, on the raw 1-5 scale. Comparable across runs only where the scenarios and the externals "
            "they touch held still, which is what separates it from mean_transcript_score."
        ),
    ),
    # The goal-state family's two FIXED keys, on the rubric family's terms: a check's own measure is its
    # expression and stays open (``describe_goal_state``); what the code emits is the per-observation row
    # (``goal_state``, 1 passed / 0 not, keyed by the check) and the statistic over it. Seeding a check's
    # expression would be the violation; seeding the observation and its aggregate is not.
    _d(
        name="goal_state",
        data_type="boolean",
        family="goal_state",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        description=(
            "Whether one goal-state check passed on one observation — 1 or 0, keyed by the check it reports "
            "(goal_check). Code-graded: no judge is in it. Defined by the check's scenario."
        ),
    ),
    _d(
        name="goal_state_pass_rate",
        data_type="numeric",
        family="goal_state",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        formula="passed goal_state rows / goal_state rows, per check",
        description=(
            "How often a goal-state check passed — the aggregate an aggregating surface reports over goal_state "
            "rows, one check at a time (key the rows on goal_check). Code-graded, and defined by the check's "
            "scenario, so two checks' rates do not pool and neither means anything outside its scenario."
        ),
    ),
    # ---- Classifier track ---------------------------------------------------
    _d(
        name="accuracy",  # ACCURACY_MEASURE, derived from MATCH_MEASURE — see the pair's definition below
        data_type="numeric",
        family="classifier",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        merit_axis="quality",
        formula="1 when the observation's match is true, 0 when it is false; its mean is correct / total",
        reader_prose="how often a classification matched its expected label",
        description=(
            "Share of classifications matching the expected label — the classifier family's quality measure, "
            "derived by the engine from each observation's `match` (a host lands `match` and never this). "
            "Scenario-bound: it means only what the labelled set means."
        ),
    ),
    _d(
        name="precision",
        data_type="numeric",
        family="classifier",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        formula="true positives / (true positives + false positives)",
        description="Per class: of the times this label was predicted, how often it was right.",
    ),
    _d(
        name="recall",
        data_type="numeric",
        family="classifier",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        formula="true positives / (true positives + false negatives)",
        description="Per class: of the times this label was correct, how often it was predicted.",
    ),
    _d(
        name="f1",
        data_type="numeric",
        family="classifier",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        formula="2 * precision * recall / (precision + recall)",
        description="Harmonic mean of precision and recall for a class.",
    ),
    _d(
        name="support",
        data_type="numeric",
        family="classifier",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        unit="cases",
        description="How many cases carried this expected label — the denominator behind its precision and recall.",
    ),
    _d(
        name="match",  # MATCH_MEASURE
        data_type="boolean",
        family="classifier",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        description=(
            "Whether one classification matched its expected label. The one a classifier kind lands; the engine "
            "derives `accuracy`, the quality measure, from it."
        ),
    ),
    _d(
        name="confusion_cell",  # CONFUSION_CELL_MEASURE, defined below beside the cell's format
        data_type="categorical",
        family="classifier",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        formula="the expected label and the predicted one, as `expected → predicted` (see confusion_cell)",
        description=(
            "Which cell of the confusion matrix one classification landed in. Counted, never averaged: the counts "
            "over a cell's results are its confusion matrix, from which per-label precision, recall and F1 are derived."
        ),
    ),
    # ---- Async-delivery lifecycle (per delivery) ----------------------------
    # The leaves of ``AsyncDelivery``, the engine's record of one piece of background work. What
    # the tool's work consisted of — its stop reasons, its own outcome buckets — is the kind's,
    # carried on ``EvalResult.kind_payload`` and described, if anywhere, in the host's catalogue.
    _d(
        name="status",
        data_type="categorical",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        categories=("delivered", "failed", "undelivered"),
        description=(
            "Where one piece of background work stood when the cell ended: delivered, failed with an error, or "
            "acknowledged and never delivered before the cell ended."
        ),
    ),
    _d(
        name="acknowledged_turn",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="turn index",
        description="The 0-based turn whose call a background tool acknowledged. A position in the conversation, not a merit.",
    ),
    _d(
        name="delivered_turn",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        unit="turn index",
        description="The 0-based turn a background tool's payload reached the conversation on. A position, not a merit.",
    ),
    _d(
        name="elapsed_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=False,
        unit="ms",
        description="Wall-clock of one piece of background work, acknowledgement to delivery or failure.",
    ),
    _d(
        name="delivered_items",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=True,
        unit="items",
        description="How many items one async delivery's payload carried. Zero is a real delivery of nothing, not a missing measurement.",
    ),
    _d(
        name="substituted",
        data_type="categorical",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        description=(
            "Whether a harness supplied this delivery's payload instead of a background run producing it "
            "(a seeded payload or a replayed capture). Such a delivery describes NO run of the tool, so "
            "the host's outcome rollup excludes it from its partition and reports the excluded count as "
            "async_deliveries_substituted. It also withholds the result's production-replicating cost, "
            "since the dollars and metered units it stands in for were never spent."
        ),
    ),
    # ---- Run-comparison surface (ResultsCompareArmRow / PerTemplateRow) -
    _d(
        name="count_a",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="cases",
        description="How many cases contributed a composite in run A — the sample size behind its score.",
    ),
    _d(
        name="count_b",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        unit="cases",
        description="How many cases contributed a composite in run B.",
    ),
    _d(
        name="comparison_basis",
        data_type="categorical",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        categories=("shared-template-intersection", "independent"),
        description="Whether the two runs were compared over the templates they share or taken as independent — the caveat that decides how much the deltas mean.",
    ),
)

_SEED = _SEED + _compare_trio("pass_hat_k", "Reliability (pass^k at the shared depth k)", "scenario_bound")
_SEED = _SEED + _compare_trio("composite", "Mean composite quality", "judge_mediated")

METRIC_DESCRIPTORS: dict[str, MetricDescriptor] = {d.name: d for d in _SEED}

# Guard against a copy-paste duplicate silently winning the dict comprehension above.
if len(METRIC_DESCRIPTORS) != len(_SEED):  # pragma: no cover - import-time invariant
    _dupes = sorted({d.name for d in _SEED if sum(1 for o in _SEED if o.name == d.name) > 1})
    raise RuntimeError(f"Duplicate measure name(s) in the metric seed: {_dupes}")


#: The ENGINE's measure families a code path grades, with no judge between the candidate and the
#: number — derived from :data:`ENGINE_FAMILIES`. Together with every host family declared
#: ``graded_by="code"`` (:func:`is_code_graded`, the one predicate both readers call), these are the
#: only families the analysis generator may rank on, and the only ones a campaign bar on a DESCRIBED
#: measure may name.
#:
#: Admitting a family is a
#: change to what every generated memo ranks on and to what every bar may be read on, so it is a
#: decision to record and not an edit to make. The analysis bundle's ``_is_reportable`` carries which
#: exclusion rests on what: ``rubric``/``dual_axis`` permanently (they are judged), ``composite``
#: because no producer puts one on a bundle surface.
#:
#: **``goal_state`` is admitted** on ``classifier``'s argument: a
#: check's verdict is code compared against what the candidate did, with no judge in it. Its
#: exclusion rested only on no producer putting one on a bundle surface; the bundle's measure walk
#: now yields each check's 0/1 per observation under :func:`goal_check_measure`, so every check's
#: pass rate is a ranked measure whether or not a bar names it. Held here, in the registry, because
#: two readers must agree on it — the bundle's ranking filter and the bar-name resolver in
#: ``contracts/declaration.py`` — and the second cannot import the first; both ask
#: :func:`is_code_graded`.
CODE_GRADED_FAMILIES: frozenset[str] = frozenset(
    name for name, family in ENGINE_FAMILIES.items() if family.graded_by == "code"
)


def is_code_graded(descriptor: MetricDescriptor, measures: MeasureRegistry) -> bool:
    """Whether code, with no judge between the candidate and the number, produced this measure.

    The engine's families answer from :data:`ENGINE_FAMILIES`; a host's own family answers from the
    :class:`MeasureFamily` the host declared on ``measures``. A measure with no family — one nobody
    described — is not code-graded: there is nothing to say it is.

    Args:
        descriptor: The measure's descriptor.
        measures: The host's measure registry, which holds its declared families.

    Returns:
        True for a code-graded family.
    """
    if descriptor.family is None:
        return False
    if descriptor.family in ENGINE_FAMILIES:
        return descriptor.family in CODE_GRADED_FAMILIES
    declared = measures.family(descriptor.family)
    return declared is not None and declared.graded_by == "code"


def summary_population(descriptor: MetricDescriptor, undeclared: MeasurePopulation) -> MeasurePopulation:
    """The population one summary of a measure is computed over — the ONE answer every summary asks.

    A measure describing a turn's time or spend (:func:`reads_turns`) is read over ``delivered`` on EVERY
    surface unless it declares ``all_observed`` — the decision surface's cells and bars, a run's summary,
    the telemetry rollup and the scope divergences alike — so no lens can average a refused call's round
    trip in as a fast turn while another leaves it out. ``scored`` keeps a candidate's failures because a
    failure is a result of the arm, and on a quality or reliability reading it must count against it; on a
    cost or latency reading a failure that took no turn has no turn's time or spend to contribute. An
    explicit ``all_observed`` is not narrowed: the declarer asked for every raw row, and the summary says so.

    Every other measure is read over its declared population, else ``undeclared``, the population of the
    surface asking.

    Args:
        descriptor: The measure's descriptor.
        undeclared: The population of the surface asking, for a measure that declares none.

    Returns:
        The population the summary is computed over, and states.
    """
    if reads_turns(descriptor) and descriptor.population != "all_observed":
        return "delivered"
    return descriptor.population or undeclared


#: How a difference in a measure reads against the measure's declared materiality threshold.
Materiality = Literal["material", "immaterial"]


def materiality(threshold: float | None, delta: float) -> Materiality:
    """Whether a difference of ``delta`` in a measure is large enough to act on.

    The ONE predicate over :attr:`MetricDescriptor.materiality_threshold`, called with the
    descriptor's threshold by the analysis bundle and with the threshold a decision surface froze
    (:class:`~threetears.evals.contracts.surface.MeasureFacts`) by every surface drawn from it, so
    the two cannot disagree about one difference. A difference whose size is below the threshold is
    ``immaterial``; at or above it, or when no threshold was declared, it is ``material``. Silence is
    conservative — a host that has not said what is too small to matter gets every difference
    treated as one that might.

    Args:
        threshold: The measure's declared threshold, in its own units, or None when it declared none.
        delta: The difference, in the measure's own units. Its sign is ignored.

    Returns:
        ``immaterial`` only below a declared threshold.
    """
    return "immaterial" if threshold is not None and abs(delta) < threshold else "material"


#: Described measures a result IMPLIES rather than carries — computed from its own fields each time
#: it is read, and never stored. The analysis bundle's measure walk yields exactly these on top of
#: the result's carried fields, so a reader asking "can a result ever carry this name" must count
#: them too, or it refuses a measure every result produces.
DERIVED_PER_RESULT_MEASURES: frozenset[str] = frozenset({"orchestration_ms", "candidate_output_tokens_per_s"})


def containment_defects(descriptor: MetricDescriptor, catalog: Mapping[str, MetricDescriptor]) -> list[str]:
    """Say why a descriptor's ``contained_by`` is not a claim the registry can honour.

    A containment declaration licenses a SUBTRACTION downstream, so it is checked rather than
    trusted: the three ways it can be wrong all produce a plausible number describing nothing.
    Read as "a part, inside a named whole, measured in the same unit" — each clause below is one
    of those three words.

    **``catalog`` is a parameter because a host declares measures too.** This ran only over the
    engine's own seed at import until a host catalogue existed; a host part naming a host whole
    would then have failed on "no seeded measure defines it" while a host part naming a nonsense
    whole would never have been asked. A registry validates its own catalogue, and the engine
    validates its own — same rule, each over what it can actually see.

    Args:
        descriptor: The descriptor to check.
        catalog: The catalogue its ``contained_by`` must resolve within.

    Returns:
        Human-readable defects, empty when the declaration holds.
    """
    if descriptor.contained_by is None:
        return []
    defects: list[str] = []
    if descriptor.attribution_scope != "subsystem":
        defects.append(f"declares containment but is {descriptor.attribution_scope}, not a part")
    whole = catalog.get(descriptor.contained_by)
    if whole is None:
        defects.append(f"names {descriptor.contained_by!r}, which no measure in the same catalogue defines")
        return defects
    if whole.attribution_scope != "end_to_end":
        defects.append(f"names {whole.name!r} as its whole, but that measure is {whole.attribution_scope}")
    if whole.unit != descriptor.unit:
        defects.append(f"is in {descriptor.unit!r} while {whole.name!r} is in {whole.unit!r}")
    return defects


if _bad := {
    d.name: defects for d in _SEED if (defects := containment_defects(d, METRIC_DESCRIPTORS))
}:  # pragma: no cover - import-time
    raise RuntimeError(f"Unsound containment declaration(s) in the metric seed: {_bad}")


# A goal-state DSL path is rooted at one of these, so a name so rooted is a scenario's own
# definition rather than a measure name. Not every CHECK is: the DSL's call forms
# (`call_count(...)`, `any(... for it in calls(...))`) have no such root, and a candidate kind's
# mechanical facts need not be DSL at all. A check's measure on a report is therefore named
# through `goal_check_measure`, which every check has, and this recognises only the raw path form.
_GOAL_STATE_ROOTS = ("state.", "variation.")


def _require_name(name: str) -> None:
    """Refuse a blank measure name.

    The resolvers below are total over real names, but the empty string is not a measure
    that failed to be recognised — it is the absence of one. Minting a nameless descriptor
    for it would push the problem downstream to a surface that renders a blank row, so it
    fails here where the caller can still see why.

    Raises:
        ValueError: If the name is empty or whitespace-only.
    """
    if not name.strip():
        raise ValueError("measure name must be a non-empty string")


def describe_rubric_dim(name: str, *, scale: RubricScale) -> MetricDescriptor:
    """Describe a rubric dimension by construction, without enumerating dimension names.

    Rubric dimensions are operator-authored and live in the database — as template-embedded
    dims or shared catalog records — so the registry classifies them structurally instead of
    listing them. Every rubric dimension is judge-mediated by construction: its value is a
    model's opinion, comparable only under the same judge configuration.

    The two reserved dual-score axis ids are not ordinary dimensions and are returned from
    the seeded core instead, which is what keeps the outcome axis at its strictest class.

    The scale is required, not defaulted: a pass/fail dimension's values are 1/0 and its mean is a
    pass rate, so describing it as 1-5 would mislabel every axis and bar drawn from it.

    Args:
        name: The dimension name, or a reserved dual-score axis id.
        scale: How the dimension is answered. Ignored for a reserved axis, which is 1-5.

    Returns:
        The descriptor for that dimension.

    Raises:
        ValueError: If the name is blank.
    """
    _require_name(name)
    seeded = METRIC_DESCRIPTORS.get(name)
    if seeded is not None and seeded.family == "dual_axis":
        return seeded
    return MetricDescriptor(
        name=name,
        data_type="numeric",
        family="rubric",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=SCALES[scale].value_range,
        description=f"Judged rubric dimension {name!r}, {SCALES[scale].reads_as}.",
    )


def describe_goal_state(expression: str) -> MetricDescriptor:
    """Describe a goal-state check, whose measure name is the expression itself.

    Args:
        expression: The goal-state DSL expression as declared on the template.

    Returns:
        The descriptor for that check.

    Raises:
        ValueError: If the expression is blank.
    """
    _require_name(expression)
    return MetricDescriptor(
        name=expression,
        data_type="boolean",
        family="goal_state",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        description=f"Goal-state check {expression!r} — defined by its scenario, so it means nothing outside it.",
    )


#: The prefix a goal-state check's PASS-RATE measure is named under on every report surface.
#:
#: A check's text is free — a DSL call form, a candidate kind's ``classifier.label == expected``, a
#: host's ``field_accuracy >= 0.92`` — so nothing in the text says it is a check, and a surface that
#: resolves a measure by name alone (the bundle's catalog, the chart compiler) would describe one as
#: unclassified. The registry mints the name and is the one reader of it, so the namespace is a
#: structural marker rather than a convention an author has to follow.
GOAL_CHECK_MEASURE_PREFIX = "goal_state:"

#: The characters a memo's figure reference (``{{<cell>|<measure_id>|<reading>|<stat>}}``) reserves,
#: percent-encoded in a minted name so any check a memo cites stays citable; ``%`` first, so an
#: authored ``%7C`` survives the round trip. A check's text is authored and free, and refusing one
#: here would fail the bundle of a campaign that already ran.
_REFERENCE_ESCAPES: tuple[tuple[str, str], ...] = (("%", "%25"), ("|", "%7C"), ("{", "%7B"), ("}", "%7D"))
_REFERENCE_DECODED = {code: char for char, code in _REFERENCE_ESCAPES}
_REFERENCE_UNESCAPE = re.compile("|".join(re.escape(code) for code in _REFERENCE_DECODED))


def goal_check_measure(expression: str) -> str:
    """The measure name a goal-state check's pass rate is reported under.

    Args:
        expression: The check, verbatim as its template or candidate kind wrote it.

    Returns:
        ``goal_state:<expression>``.

    Raises:
        ValueError: If the expression is blank.
    """
    _require_name(expression)
    escaped = expression
    for char, code in _REFERENCE_ESCAPES:
        escaped = escaped.replace(char, code)
    return GOAL_CHECK_MEASURE_PREFIX + escaped


def goal_check_of(name: str) -> str | None:
    """The check a measure name was minted for by :func:`goal_check_measure`, or None.

    Args:
        name: Any measure name.

    Returns:
        The check's verbatim text, or None when the name is not a check's measure.
    """
    if not name.startswith(GOAL_CHECK_MEASURE_PREFIX):
        return None
    # One pass, so a decoded character is never read again as part of another escape.
    expression = _REFERENCE_UNESCAPE.sub(
        lambda m: _REFERENCE_DECODED[m.group(0)], name[len(GOAL_CHECK_MEASURE_PREFIX) :]
    )
    return expression if expression.strip() else None


def describe_goal_check_rate(expression: str) -> MetricDescriptor:
    """Describe how often one goal-state check passed — the measure a report carries for it.

    Each observation of a check is a 1 (passed) or a 0, and a report pools them, so the measure is
    their mean: a pass rate, numeric on 0-1, higher better. Code-graded with no judge anywhere in it,
    which is why the ``goal_state`` family is ranked on (:data:`CODE_GRADED_FAMILIES`). Defined by
    the check's own scenario, so two checks' rates never pool and neither transfers.

    Args:
        expression: The check, verbatim.

    Returns:
        The descriptor, named by :func:`goal_check_measure`.

    Raises:
        ValueError: If the expression is blank.
    """
    return MetricDescriptor(
        name=goal_check_measure(expression),
        data_type="numeric",
        family="goal_state",
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        merit_axis="quality",
        formula="observations on which the check held / observations that evaluated it",
        reader_prose=f"how often the check {expression!r} held",
        description=(
            f"Pass rate of goal-state check {expression!r}: code compared what the candidate did against "
            "the check, with no judge in it. Defined by its scenario, so it means nothing outside it."
        ),
    )


#: The core measure a classification's confusion-matrix cell is reported under.
CONFUSION_CELL_MEASURE = "confusion_cell"

#: The core measure one classification's verdict is reported under — the one a classifier kind lands on
#: ``host_measures``, a bool.
MATCH_MEASURE = "match"

#: The classifier family's quality measure: 1.0 or 0.0 per observation, DERIVED by the engine from
#: :data:`MATCH_MEASURE` and never landed by a kind. It is the classifier's reading on the ``quality`` merit
#: axis, so it is what a bar, a verdict tier and a family of comparisons rank a classifier on. Derived rather
#: than carried because a host carrying both would put one comparison into every family twice — the duplicate
#: that made hosts mint their own ``<thing>_accuracy`` while ``match`` had no axis. A boolean has no per-case
#: mean for a family to test, so the axis lives on the numeric reading, as a goal check's does.
ACCURACY_MEASURE = "accuracy"

#: The separator between the two labels of a confusion cell — the expected one, then the predicted one.
CONFUSION_SEPARATOR = " → "

#: Characters escaped inside a label so a cell always splits into exactly the two labels it was made of.
_CONFUSION_ESCAPES: tuple[tuple[str, str], ...] = (("%", "%25"), ("→", "%E2%86%92"))

#: A run of percent-encoded UTF-8 bytes: how every escape in a minted label is written.
_PERCENT_RUN = re.compile(r"(?:%[0-9A-F]{2})+")


def _percent_encoded(text: str) -> str:
    return "".join(f"%{byte:02X}" for byte in text.encode())


def _percent_decoded(text: str) -> str:
    """``text`` with every percent-encoded run decoded; a run that is not UTF-8 is left as written.

    One pass, so a decoded ``%`` is never read again as the start of another escape. Every escape a
    minted label carries is the percent-encoding of the character it stands for, so this one decoder
    reads all of them.
    """

    def decoded(run: re.Match[str]) -> str:
        try:
            return bytes.fromhex(run.group(0).replace("%", "")).decode()
        except UnicodeDecodeError:
            return run.group(0)

    return _PERCENT_RUN.sub(decoded, text)


def _keeping_edge_whitespace(text: str) -> str:
    """``text`` with the whitespace at either end percent-encoded, so a model that strips its strings keeps it.

    Every eval model strips the strings it holds
    (:class:`~threetears.evals.contracts.base.EvalBaseModel`), a stored result's measures included, so a
    label ending in a space would be read back without it: ``"positive "`` counted as ``"positive"``, the
    label it is not. Encoded, the label comes back exactly. Interior whitespace is never stripped and is
    left as written.
    """
    start = len(text) - len(text.lstrip())
    if start == len(text):
        return _percent_encoded(text)
    end = len(text.rstrip())
    return _percent_encoded(text[:start]) + text[start:end] + _percent_encoded(text[end:])


def _escape_label(label: str) -> str:
    for char, code in _CONFUSION_ESCAPES:
        label = label.replace(char, code)
    return _keeping_edge_whitespace(label)


def confusion_cell(expected: str, predicted: str) -> str:
    """The value a classification reports under ``confusion_cell``: ``expected → predicted``.

    A label is free text, so the separator's arrow is escaped inside each label, and so is any whitespace
    at a label's ends, which a stored result would otherwise strip; :func:`confusion_of` reads the two
    labels back exactly.

    Args:
        expected: The label the case expected.
        predicted: The label the candidate gave.

    Returns:
        The cell, as one categorical value.

    Raises:
        ValueError: Either label is blank.
    """
    if not expected.strip() or not predicted.strip():
        raise ValueError("a confusion cell needs both an expected and a predicted label")
    return _escape_label(expected) + CONFUSION_SEPARATOR + _escape_label(predicted)


def confusion_of(cell: str) -> tuple[str, str] | None:
    """The ``(expected, predicted)`` labels of a value :func:`confusion_cell` made, or None for anything else.

    Args:
        cell: A ``confusion_cell`` value.

    Returns:
        The two labels, or None when the value is not one confusion cell.
    """
    parts = cell.split(CONFUSION_SEPARATOR)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        return None
    return _percent_decoded(parts[0]), _percent_decoded(parts[1])


#: The per-label statistics a classifier's confusion matrix yields, each minted per label.
ClassifierStatistic = Literal["precision", "recall", "f1"]

#: The prefix a per-label classifier statistic is named under: ``classifier:<statistic>:<label>``.
CLASSIFIER_LABEL_MEASURE_PREFIX = "classifier:"

_CLASSIFIER_STATISTICS: tuple[ClassifierStatistic, ...] = ("precision", "recall", "f1")


def classifier_label_measure(statistic: ClassifierStatistic, label: str) -> str:
    """The measure name one label's precision, recall or F1 is reported under.

    Minted rather than authored, like :func:`goal_check_measure`: a label set is the host's and open,
    so the registry names each label's statistic and is the one reader of the name.

    Args:
        statistic: ``precision``, ``recall`` or ``f1``.
        label: The label.

    Returns:
        ``classifier:<statistic>:<label>``, the label escaped as a figure reference requires and with the
        whitespace at its ends encoded, so a summary's stripped name still names this label and no other.

    Raises:
        ValueError: The label is blank.
    """
    _require_name(label)
    escaped = label
    for char, code in _REFERENCE_ESCAPES:
        escaped = escaped.replace(char, code)
    return f"{CLASSIFIER_LABEL_MEASURE_PREFIX}{statistic}:{_keeping_edge_whitespace(escaped)}"


def classifier_label_of(name: str) -> tuple[ClassifierStatistic, str] | None:
    """The ``(statistic, label)`` a name was minted for by :func:`classifier_label_measure`, or None.

    Args:
        name: Any measure name.

    Returns:
        The statistic and the label, verbatim, or None when the name is not a per-label measure.
    """
    if not name.startswith(CLASSIFIER_LABEL_MEASURE_PREFIX):
        return None
    statistic, _, escaped = name[len(CLASSIFIER_LABEL_MEASURE_PREFIX) :].partition(":")
    label = _percent_decoded(escaped)
    if statistic not in _CLASSIFIER_STATISTICS or not label.strip():
        return None
    return statistic, label


def describe_classifier_label(statistic: ClassifierStatistic, label: str) -> MetricDescriptor:
    """Describe one label's precision, recall or F1 — derived from a cell's confusion matrix.

    Args:
        statistic: ``precision``, ``recall`` or ``f1``.
        label: The label.

    Returns:
        The descriptor, named by :func:`classifier_label_measure`, carrying the core statistic's meaning.
    """
    core = METRIC_DESCRIPTORS[statistic]
    return core.model_copy(
        update={
            "name": classifier_label_measure(statistic, label),
            "description": f"{core.description} For the label {label!r}.",
            "reader_prose": f"the {statistic} of the label {label!r}",
            "merit_axis": "quality",
        }
    )


def partition_components(whole: str, *extra: dict[str, MetricDescriptor], measures: MeasureRegistry) -> list[str]:
    """Every measure declared a component of ``whole``, sorted.

    The authority on whether a part EXHAUSTS its whole. Containment alone does not license
    subtracting a part from a whole and calling the leftover "unattributed" — that word
    claims nothing accounts for the difference, which is false as soon as the whole has
    other declared components. ``total_ms`` has three, so differencing any one of them
    leaves the other two's movement, attributed to measures this registry names.

    **Read from the whole describable measure space, deliberately, and not from a caller's
    local catalog.** Partition membership is a fact about the MEASURE SPACE, not about what one
    campaign happened to record — and that space is the engine's core plus whatever the
    host declares, since a host measure can name an engine whole. Reading the core alone would
    narrow the partition silently, which is the same unsound subtraction from the other side.
    A bundle's ``measure_catalog`` holds only the measures appearing in that bundle, so a
    campaign that recorded ``total_ms`` and ``orchestration_ms`` but not
    ``llm_ms``/``tool_ms`` would look like a one-component partition and earn back exactly
    the unsound subtraction this exists to refuse — and an unmeasured component is MORE
    reason to withhold, not less, since nothing can say where the movement went.

    ``extra`` catalogs are unioned in rather than replacing the registry, so a caller
    carrying descriptors the registry does not seed (phase timings, resolved at read time)
    can contribute components without ever narrowing the set.

    Args:
        whole: The end-to-end measure name to find components of.
        *extra: Additional descriptor maps to consider alongside the registry.
        measures: The host's measure registry, whose declarations join the core's.

    Returns:
        The component names in sorted order; empty when nothing declares ``whole`` its container.
    """
    found = {name for name, d in _describable_measures(measures).items() if d.contained_by == whole}
    for catalog in extra:
        found |= {name for name, d in catalog.items() if d.contained_by == whole}
    return sorted(found)


#: The distinguishing clause of each containment reason a remainder is withheld for. Named here, with
#: the predicate that uses them, so the tests pin the clause and the sentence around it stays free.
WITHHELD_NOT_CONTAINED = "is not declared a component of"
WITHHELD_PARTITION_INCOMPLETE = "is only one of several declared components of"


def remainder_withheld_reason(
    part: str,
    whole: str,
    described: MetricDescriptor | None,
    *extra: dict[str, MetricDescriptor],
    measures: MeasureRegistry,
) -> str | None:
    """Say why ``whole`` minus ``part`` may not be published as an "unattributed" remainder, or None.

    THE containment rule, in one place, for every site that decides whether a remainder may be
    stated: the bundle's divergence lens and the attribution chart compiler. Two sites deciding it
    separately is how a chart once stated a remainder on containment alone that the generator then
    refused on exhaustion, discarding a paid analysis.

    Two clauses, and they are different questions. The part must be DECLARED inside the whole
    (sharing a unit makes two measures comparable, not nested — undeclared fails closed). And it
    must EXHAUST it: ``total_ms`` partitions into three components, so the leftover of subtracting
    one is the other two's movement, attributed to measures this registry names, and calling it
    "unattributed" states the wrong conclusion.

    Args:
        part: The subsystem measure's name.
        whole: The end-to-end measure's name.
        described: ``part``'s descriptor as the caller resolves it, or None when it has none.
        *extra: Descriptor maps whose components join the registry's (see :func:`partition_components`).
        measures: The host's measure registry.

    Returns:
        The sentence a reader is owed in place of the number, or None when the remainder is sound.
    """
    if described is None or described.contained_by != whole:
        return (
            f"{part} {WITHHELD_NOT_CONTAINED} {whole} — they share a unit but are not known to be nested, so "
            "what looks like an unexplained remainder may be two disjoint stretches of the same measure."
        )
    siblings = partition_components(whole, *extra, measures=measures)
    if len(siblings) > 1:
        rest = [name for name in siblings if name != part]
        return (
            f"{part} {WITHHELD_PARTITION_INCOMPLETE} {whole}, which the catalog partitions into "
            f"{' + '.join(siblings)}. Subtracting one component leaves {' + '.join(rest)}, which is movement "
            "that IS attributed — to those measures — so calling it unattributed would name the wrong "
            "conclusion. Report the other components instead of a remainder."
        )
    return None


def describe_reported_measure(name: str, measures: MeasureRegistry) -> MetricDescriptor:
    """Describe a measure a report carries, resolving a phase timing the way its summary did.

    :func:`describe_measure` deliberately has no phase-timing branch — a phase key is
    indistinguishable by name from a run-summary statistic — so a measure that reached a report
    through the phase-timing walk must be resolved as one, or the catalog would describe it as
    unclassified while the summary reports it as measured. The bundle's catalog and the chart
    compiler both resolve through this, so they describe one measure one way.

    Args:
        name: A measure name appearing in a report.
        measures: The host's measure registry.

    Returns:
        Its descriptor.
    """
    seeded = describe_measure(name, measures)
    return seeded if seeded.family is not None else describe_phase_timing(name)


def describe_phase_timing(key: str) -> MetricDescriptor:
    """Describe a phase-timing key, for a caller iterating ``EvalResult.phase_timings``.

    Phase-timing keys are written ``<tool>_<phase>_ms`` and their space is open — a new tool
    or a new phase adds keys without touching this module. Deliberately NOT pattern-matched
    from a bare name: run-summary statistics like ``async_delivery_p95_elapsed_ms`` are shaped
    identically and are not phases, so guessing from the name alone invented a "Wall-clock
    inside the async delivery p95 elapsed phase" that does not exist. Only the caller holding the
    ``phase_timings`` mapping knows a key really is a phase, so only it may say so.

    Args:
        key: A key from ``EvalResult.phase_timings``.

    Returns:
        The descriptor for that phase.

    Raises:
        ValueError: If the key is blank.
    """
    _require_name(key)
    seeded = METRIC_DESCRIPTORS.get(key)
    if seeded is not None:
        return seeded
    phase = key.removesuffix("_ms")
    return MetricDescriptor(
        name=key,
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        higher_is_better=False,
        unit="ms",
        formula="summed across repeat deliveries within the result",
        description=f"Wall-clock inside the {phase.replace('_', ' ')} phase.",
    )


def _host_declared(name: str, measures: MeasureRegistry) -> MetricDescriptor | None:
    """The host's descriptor for ``name``, or None when nothing declares one.

    A direct registry lookup rather than a filter over :func:`_describable_measures`, because
    this runs once per measure leaf on every bundle walk and that one composes a whole
    dictionary. The two are held in agreement by a test that resolves every listable name
    through :func:`describe_measure` — a shared helper would make them agree by construction
    but would put the composition in the hot path, and the drift this guards against is in
    WHICH catalogues are consulted, which a test can see.

    Args:
        name: The measure to ask the host about.
        measures: The host's measure registry.

    Returns:
        The host's descriptor, or ``None``.
    """
    return measures.get(name)


def _describable_measures(measures: MeasureRegistry) -> dict[str, MetricDescriptor]:
    """Every measure a host can describe: the engine core plus the host's own.

    The core is applied last so it wins a tie, matching :func:`describe_measure`'s resolution
    order. A collision cannot arise from a well-formed host — the profile refuses one at
    registration — so this is the same rule stated where it can be read rather than a second
    defence.

    Args:
        measures: The host's measure registry.

    Returns:
        ``{name: descriptor}``. The core alone for a host that declares no measures.
    """
    host = {name: d for name in measures.names if (d := measures.get(name)) is not None}
    return {**host, **METRIC_DESCRIPTORS}


def describe_measure(name: str, measures: MeasureRegistry) -> MetricDescriptor:
    """Describe any measure name, seeded or not. Total by construction.

    Resolution order, most specific first: the seeded core, then a goal-state check's pass-rate
    measure (named by :func:`goal_check_measure`, a namespace the registry mints), then a per-label
    classifier statistic (named by :func:`classifier_label_measure`, another), then **the
    host's own catalogue** (``measures``), then a raw goal-state path expression (recognised by its DSL
    root — ``state.`` or ``variation.``; a call-form check has no such root, which is why reports
    carry a check under its minted name). Anything left is unclassified and is
    treated as ``scenario_bound`` — the strictest class — so an unrecognised name can never
    widen a pooling decision it was never vetted for.

    **The host step is not a convenience, and the core wins ties deliberately.** A measure
    whose vocabulary belongs to one host's tool — the outcome buckets of an async delivery,
    say — is declared in that host's catalogue and not here, so without this step every one
    of them resolves to the unclassified arm: silently, in the analysis bundle's own
    ``measure_catalog``, which is the surface a paid generator reads. The core is consulted
    first so a host cannot redefine what ``mean_score`` is. A host that declares no measures
    passes an empty registry, and resolution continues past the step.

    There is deliberately NO phase-timing branch here: a phase-timing key is indistinguishable
    by name from a run-summary statistic, so pass such keys to :func:`describe_phase_timing`
    instead. A phase key reaching this function lands in the unclassified arm, which is the
    honest answer for a bare string but not the one you want.

    Callers that already know a name's family should say so by calling
    :func:`describe_rubric_dim`, :func:`describe_goal_state` or
    :func:`describe_phase_timing` directly; a bare rubric
    dimension name is indistinguishable from a novel measure and so resolves conservatively
    here rather than being guessed into the rubric family.

    That is not only a nicety: **measure names are unique within a family, not across
    them.** Nothing stops an operator naming a rubric dimension ``accuracy`` while the
    classifier track already emits one, and this function sees only a string, so the seeded
    measure wins the name. Rendering a judged 1-5 dimension as a 0-1 classifier rate would
    be silently wrong — so resolve through the family-specific function whenever the family
    is known, and treat this one as the fallback for names arriving without context.

    Args:
        name: Any measure name.
        measures: The host's measure registry.

    Returns:
        A descriptor. Never ``None``, and never raises merely because a name is unknown.

    Raises:
        ValueError: If the name is blank — see :func:`_require_name`.
    """
    _require_name(name)
    seeded = METRIC_DESCRIPTORS.get(name)
    if seeded is not None:
        return seeded

    # Before the host, for the reason the core is: the namespace is the core's, so a host cannot
    # redefine what a check's measure is by declaring a name inside it.
    if (expression := goal_check_of(name)) is not None:
        return describe_goal_check_rate(expression)
    if (per_label := classifier_label_of(name)) is not None:
        return describe_classifier_label(*per_label)

    declared = _host_declared(name, measures)
    if declared is not None:
        return declared

    if name.startswith(_GOAL_STATE_ROOTS):
        return describe_goal_state(name)

    # `family` and `data_type` are left None rather than guessed. There is no honest value
    # for a name nobody has described, and defaulting them to "mechanical"/"numeric" would
    # state the loosest family as a fact — in the module that is the SSoT for what a measure
    # is, and against the missing-is-not-zero rule the rest of this subsystem enforces.
    # `transferability_class` is the one field that must still answer, because it gates
    # pooling; strictest makes the unknown safe.
    return MetricDescriptor(
        name=name,
        data_type=None,
        family=None,
        transferability_class="scenario_bound",
        attribution_scope="end_to_end",
        description=(
            f"Unclassified measure {name!r}. Held at the strictest transferability class until it is "
            "described, so it is never pooled on an assumption nobody made."
        ),
    )


def list_metrics(
    measures: MeasureRegistry,
    family: MetricFamily | None = None,
    attribution_scope: AttributionScope | None = None,
) -> list[MetricDescriptor]:
    """List every measure a host can describe, optionally filtered by family and/or scope.

    The engine's closed core AND the host's own catalogue, merged the way
    :func:`describe_measure` resolves them — because this is the operator-facing catalogue behind
    the ``/metrics`` surface, and a measure a host publishes on its run summaries while this list
    omits it is the same defect from the other end: a picker that cannot offer a number the
    surface beside it renders. The core is listed even for a host that declares no measures.

    Three things are NOT listed, all because no honest complete list of them exists: rubric
    dimension names and goal-state expressions (authored in the database), and phase-timing keys
    (one per tool and phase). Resolve those through :func:`describe_rubric_dim`,
    :func:`describe_goal_state` and :func:`describe_phase_timing` respectively.

    ``attribution_scope`` is the seam behind the subsystem-view defaults: a subsystem
    tuning view (one background tool's, say) populates its measure picker with ``attribution_scope="subsystem"``
    so an end-to-end score — which a downstream failure can tank without the subsystem moving —
    is not the default read. The end-to-end measures are one dropped filter away, never hidden:
    the unfiltered call still returns every measure. The two filters compose, so a caller can ask
    for a family's subsystem-isolated measures alone.

    Args:
        measures: The host's measure registry.
        family: Restrict to a single family, or None for all.
        attribution_scope: Restrict to ``subsystem``-isolated or ``end_to_end`` measures, or
            None for both.

    Returns:
        Descriptors sorted by name.
    """
    values = list(_describable_measures(measures).values())
    if family is not None:
        values = [d for d in values if d.family == family]
    if attribution_scope is not None:
        values = [d for d in values if d.attribution_scope == attribution_scope]
    return sorted(values, key=lambda d: d.name)
