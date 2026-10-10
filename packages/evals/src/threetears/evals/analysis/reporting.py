"""Query-time projection of eval runs + results into flat, comparable rows.

This module is the read tier's foundation: one projection,
:func:`project_score_records`, that every downstream reporting surface consumes
(pivots, export, and the estimation engine), plus :func:`compute_comparison_sets`, which
answers *which of these runs may honestly be compared with each other*.

**Nothing here is stored.** Records are recomputed per query, exactly as
:func:`~threetears.evals.kernel.scoring.compute_pass_hat_k` is. That keeps the row shape free to
evolve while the surfaces that consume it are still being learned — a persisted
projection would freeze it against every future consumer. Aggregation is
calibrated to a corpus of dozens-to-hundreds of cells (one operator, one
scope); :func:`project_score_records` is the seam to push down into SQL if
that ever stops holding.

**The subject is a first-class coordinate, keyed on a stable id.** Composite quality is
comparable *within* a subject and never across one: rubric dimensions are derived
from each subject's own self-description and tools, so a support agent's 0.8 and a
coding assistant's 0.8 are different measurements wearing the same number, and pooling
them produces a figure that nothing downstream can falsify. Carrying
``subject_id`` on the row is what makes such pooling *detectable* — a surface
that ignores the coordinate is doing so visibly, in code a reviewer can point at,
rather than by omission nobody can see. It does not make pooling impossible: no
data model can stop a caller from averaging two numbers. It keys on
``SubjectSnapshot.subject_id`` and never on a display label: a name is not a stable
identity, so keying on it would split one renamed subject into two and merge two
same-named subjects into one.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pydantic import Field, model_validator

from threetears.evals.analysis.numbers import format_number
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.hashing import canonical_digest, canonical_json
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.host.sweepables import CANDIDATE_MODEL_LEVER
from threetears.evals.kernel.identity import IDENTITY_VERSION, resolve_context_identity
from threetears.evals.kernel.metrics import MetricDescriptor, describe_measure
from threetears.evals.schema.models import (
    OUTCOME_DIM_ID,
    RESERVED_DIM_IDS,
    TRANSCRIPT_DIM_ID,
    RubricScale,
)
from threetears.evals.kernel.surface import FrontierDominance
from threetears.evals.kernel.result_condition import (
    JUDGE_CANNOT_TELL_OUTCOME,
    classify_result,
    counted_goal_verdicts,
    counted_rubric_scores,
    counted_score,
    trial_exclusion,
)
from threetears.evals.kernel.scoring import (
    CompositeBasis,
    composite_basis,
    pool_composite_bases,
    result_composite,
)
from threetears.observe import get_logger

from threetears.evals.analysis.completeness import degraded_run_disclosures

if TYPE_CHECKING:
    from threetears.evals.schema.models import (
        EvalResult,
        EvalRun,
        GoalStateOutcome,
        LatencyMetrics,
    )


log = get_logger(__name__)


# The measures this projection emits today.
#
# These are OBSERVATION-level names and are deliberately not the registry's
# aggregate names: `threetears.evals.kernel.metrics` describes `mean_composite`,
# `composite_a/_b/_delta` — all of which are aggregates OVER these rows, not the
# rows themselves. Do not "fix" the mismatch by renaming these to a registry key:
# `describe_measure` is a total function, so a wrong name resolves to the
# unclassified arm (`family=None`, `transferability_class="scenario_bound"`)
# silently rather than raising. The aggregation layer maps a row metric to its
# aggregate descriptor; the mapping is that layer's job, not a naming coincidence.
#
# Scoped to these two. `METRIC_SCORE` below IS a registry key — the registry describes
# the raw per-observation judge score under that exact name — and it still maps to
# `mean_score` in the aggregation layer, because the mapping is about what a CELL holds,
# never about whether the row name resolves.
#: The observation-level measure holding a result's composite quality score.
METRIC_COMPOSITE = "composite"
METRIC_COST_USD = "cost_usd"

# The per-dimension raw judge score, on the 1-5 rubric scale. Described in the measure
# registry under this same name, so the one an operator meets here, in `export_results`'
# `metric` column and in `list_metrics` is one measure rather than three spellings.
#
# ONE metric whose dimension is a coordinate (`ScoreRecord.rubric_dim`), never a
# family of metric names like `character` / `grounding`. Three reasons, and the
# first is the one that would cost most to discover later:
#
# 1. A dim name in the metric namespace can COLLIDE. `TRANSCRIPT_DIM_ID` already
#    carries a `__` prefix for exactly this reason; a template dim named
#    `composite` would merge dim rows and composite rows into one aggregate,
#    because `pivot` selects on `r.metric == metric`. Wrong number, no error.
# 2. `row_factor='rubric_dim', column_factor='model'` answers "compare these runs
#    by dimension" in ONE table. A metric per dim needs N pivots and shows the
#    dims nowhere side by side.
# 3. Dim names are moving to a dotted `<context>.<dim>` form so a dimension
#    carries the context it was scored against, and a host's levers are often
#    dotted too (`<kind>.<field>`). The pivot routes neither by shape (#664), but
#    a reader would: a dotted name in the metric namespace reads as a lever.
#    Stated as the direction it is: no dotted dim name exists in the tree today,
#    and this reason holds the shape open for one rather than describing one.
#: The observation-level measure holding one judged rubric dimension's score: one row per dimension.
METRIC_SCORE = "score"

# The per-result wall-clock, read straight off `EvalResult.latency` rather than
# projected into a `ScoreRecord` — so it is NOT in `PROJECTED_METRICS` and `pivot`
# refuses it. It lives up here with the other names because it is the same KIND of
# name (observation-level, one result's value) and because `_AGGREGATE_OF_OBSERVATION`
# below has to map it: `history` series it, and a history point is a mean.
METRIC_TOTAL_MS = "total_ms"

# The two dual-score axes, as METRIC names. Deliberately the SAME strings as
# the reserved rubric-dim ids, aliased rather than re-spelled so one axis has one name
# wherever it appears — on `EvalResult.transcript_score.dim`, in a `JudgeConfig`
# binding, in the measure registry, and in this `metric` column.
#
# This is the case reason 1 above anticipates: the `__` prefix exists so a reserved id
# cannot collide with a template dim in the metric namespace, and these two are the ids
# it was minted for. They are NOT rubric dims on the `rubric_dim` coordinate, and that is
# the other half of the same decision — `compute_dimension_summary` does not aggregate
# them, so putting them on that coordinate would make the projection and the run summary
# disagree about which dimensions exist. A metric of its own is also the shape the
# question wants: their DIVERGENCE is what the catalog says separates a worse agent from
# a changed world, and a divergence between two metrics is one pivot rather than two
# levels of a coordinate.
#: The observation-level measure holding the dual-score transcript axis.
METRIC_TRANSCRIPT = TRANSCRIPT_DIM_ID
#: The observation-level measure holding the dual-score outcome axis.
METRIC_OUTCOME = OUTCOME_DIM_ID
# One row per goal-state check a result evaluated, keyed by `ScoreRecord.goal_check`: 1.0 passed, 0.0 not.
# A projected metric because this projection is the one producer behind export, pivot and every pooled
# read, and a code-graded verdict a reader cannot key by check is one they cannot act on.
METRIC_GOAL_STATE = "goal_state"

#: Every measure `project_score_records` can emit. Unlike the factor set, this one
#: is genuinely CLOSED — the projection is the only producer of score records, so
#: a name outside it can only be a typo. That distinction is why an unknown metric
#: is refused while a registered lever no run set is not: such an axis names a key a
#: run might not have set, but a mistyped metric names nothing that exists, and
#: answering it with an empty grid is indistinguishable from an empty scope.
#: A metric added to the projection adds itself here, in the same edit.
PROJECTED_METRICS = frozenset(
    {METRIC_COMPOSITE, METRIC_COST_USD, METRIC_SCORE, METRIC_TRANSCRIPT, METRIC_OUTCOME, METRIC_GOAL_STATE}
)

# The projected metrics that emit several rows per result, each keyed by a coordinate naming what the row is
# about: metric -> (that coordinate, what pooling spans in words, the short noun for one row's member). A cell
# of one of these is the statistic the
# registry describes only while the coordinate is pinned by an axis or filter; unpinned, it pools every value
# of it. Read by the pivot's scope check and POOLED caveat, the pivot's unknown-metric hint, the history
# refusal, and — through SCOPED_METRICS_HELP — the REST and MCP pivot and export help, so a metric added here is qualified on each of them at once,
# and a per-row metric NOT added here pools silently, which is the defect this table exists to make
# impossible to write without deciding.
SCOPED_METRICS: dict[str, tuple[str, str, str]] = {
    METRIC_SCORE: ("rubric_dim", "judged rubric dimension", "dimension"),
    METRIC_GOAL_STATE: ("goal_check", "goal-state check", "check"),
}

#: How each scoped metric must be read, in one sentence per metric, for every surface's help text (REST and MCP
#: alike) — rendered from the table rather than written beside it, so no surface can describe a subset.
SCOPED_METRICS_HELP = " ".join(
    f"'{metric}' has one row per {noun}: put '{field}' on an axis, or each cell pools every {noun}."
    for metric, (field, noun, _row) in SCOPED_METRICS.items()
)


# How far the parts may overrun the whole before the split is refused rather than
# reported.
#
# The parts nest inside the whole structurally — ``llm.call`` and ``tool.execute``
# spans are children of the ``agent.invoke`` turn root, and both the rounds loop and
# the action loop are serial — so a correct measurement can only leave the remainder
# at or above zero. The tolerance is therefore NOT a modelling allowance for
# overlapping work; it absorbs one thing only: the three numbers are summed in
# different groupings, so binary floating-point addition can put the difference a few
# units in the last place below zero on an exact partition.
#
# **Sized to that noise and not to a round number**, because a threshold generous
# enough to swallow a real overlap would silently convert a capture fault into a
# plausible-looking split. Durations are exact integer nanoseconds divided by 1e6, so
# the only error is the summation itself: bounded by roughly n·ε relative, and a cell
# emitting ~100 bucketed spans across a ~17-minute total puts that near 1e-8 ms. One
# nanosecond sits about two orders of magnitude above the noise and six below the
# shortest overlap any real span pair could produce. An overrun larger than this means
# a part was measured outside its whole — a span reaching a bucket from outside the
# turn roots — and that is a capture fault to report, not a negative component to
# publish.
PARTITION_TOLERANCE_MS = 1e-6

# The distinguishing clause of each reason the partition is withheld. Split out for the
# same reason the divergence lens splits its own: a test pins the CLAUSE, so the
# surrounding sentence can be rewritten for a reader without breaking the pin.
WITHHELD_UNMEASURED_COMPONENT = "not measured"
WITHHELD_PARTS_EXCEED_WHOLE = "sum to more than"


class LatencyPartition(EvalBaseModel):
    """One result's ``total_ms`` split into its named parts and a named remainder.

    ``total_ms`` is the wall-clock of the candidate's turn-root spans, and the two
    measured parts nest inside it. What is left over — whatever the candidate does
    between its model and tool calls: building its input, parsing its output, updating
    and persisting state — had no name, so a whole-run latency movement that lived there could not be placed. It
    could only be reported as a total that moved while both named parts stayed flat,
    which reads exactly like a measurement error. Naming the remainder is what turns
    that into an attribution.

    **Derived, never stored.** ``orchestration_ms`` is ``total_ms`` minus the two
    parts, recomputed per query exactly as :func:`project_score_records` and
    :func:`~threetears.evals.kernel.scoring.compute_pass_hat_k` are. A persisted copy would be a
    second answer to a question the three captured components already settle.

    **What is deliberately NOT in here.** The drain wait, the judge phase and every
    background phase timing are milliseconds measured on the same clock that fall
    *outside* the turn roots — background work on a detached trace root, and scoring
    that happens after the turns end. They are disjoint from ``total_ms``, not
    components of it, which the registry records by leaving their ``contained_by``
    unset. Folding any of them into the remainder is the arithmetic that once produced
    a ~95-second unattributed swing describing no stretch of wall-clock at all, and
    the reason this class takes a :class:`~threetears.evals.schema.models.LatencyMetrics` and
    reads three fields of it rather than summing whatever it holds.

    Exactly one of the split and ``withheld`` is present, enforced below: a caller
    that gets no numbers gets a sentence saying why, and never a silent ``None`` it
    has to invent an explanation for.
    """

    #: The whole being partitioned — wall-clock across the result's turn-root spans.
    total_ms: float | None = None
    #: The share inside model calls.
    llm_ms: float | None = None
    #: The share inside tool executions.
    tool_ms: float | None = None
    #: The named remainder: turn wall-clock inside neither a model call nor a tool
    #: execution. Clamped to exactly 0.0 when the parts overrun the whole by less than
    #: the float-noise tolerance, since a negative share of a whole describes nothing.
    orchestration_ms: float | None = None
    #: Why no split is published, as a sentence for a reader — set exactly when the
    #: split is absent. A partition needs all three components measured, and needs the
    #: parts to fit inside the whole.
    withheld: str | None = None

    @model_validator(mode="after")
    def _split_or_reason(self) -> LatencyPartition:
        """Refuse a record that is neither a whole split nor a whole refusal.

        Checks all four numbers rather than the remainder alone: a record carrying a
        total beside a withheld sentence would be a partition that half-published, and a
        reader has no way to tell which half to believe.
        """
        numbers = (self.total_ms, self.llm_ms, self.tool_ms, self.orchestration_ms)
        present = [value is not None for value in numbers]
        if any(present) != all(present) or all(present) == (self.withheld is not None):
            raise ValueError(
                "exactly one of the complete split / withheld must be set — got "
                f"total_ms={self.total_ms!r}, llm_ms={self.llm_ms!r}, tool_ms={self.tool_ms!r}, "
                f"orchestration_ms={self.orchestration_ms!r}, withheld={self.withheld!r}"
            )
        return self


def decompose_total_ms(latency: LatencyMetrics | None) -> LatencyPartition:
    """Split one result's ``total_ms`` into its parts and the named remainder.

    Args:
        latency: The result's captured latency, or ``None`` for a cell that timed
            nothing whatever.

    Returns:
        A :class:`LatencyPartition` carrying either the four numbers or a sentence
        naming why they were withheld.
    """
    if latency is None:
        return LatencyPartition(withheld="this cell timed nothing, so there is no total to partition.")

    components = {"total_ms": latency.total_ms, "llm_ms": latency.llm_ms, "tool_ms": latency.tool_ms}
    unmeasured = [name for name, value in components.items() if value is None]
    if unmeasured:
        # Not zero-filled. A cell can measure an `llm.call` while producing no turn
        # root — the candidate failing outside the turn wrapper does exactly that —
        # and treating the missing one as zero would publish a remainder computed
        # from a number nobody observed, in the same units as ones somebody did.
        verb = "was" if len(unmeasured) == 1 else "were"
        return LatencyPartition(
            withheld=(
                f"{', '.join(unmeasured)} {verb} {WITHHELD_UNMEASURED_COMPONENT}, so the parts cannot account for the whole."
            )
        )

    # Read back through the same dict the absence check walked, so the two can't
    # diverge into checking one set of fields and partitioning another.
    measured = {name: value for name, value in components.items() if value is not None}
    total, llm, tool = (float(measured[name]) for name in ("total_ms", "llm_ms", "tool_ms"))
    remainder = total - llm - tool
    if remainder < -PARTITION_TOLERANCE_MS:
        return LatencyPartition(
            withheld=(
                f"llm_ms and tool_ms {WITHHELD_PARTS_EXCEED_WHOLE} total_ms by "
                # The one number rule, and both ends of the range are why. A fixed
                # `.1f` renders every overrun below 50 microseconds as "by 0.0ms" — a
                # refusal whose own sentence says nothing happened — and the threshold
                # is a nanosecond, so those are reachable. A capped `%g` breaks the
                # other end: it goes scientific once the exponent reaches the precision,
                # so `.3g` prints a 464ms overrun, the size actually measured here, as
                # "by 4.64e+02ms". `format_number` keeps a small value non-zero and a
                # large one whole. This sentence is read by a person.
                f"{format_number(-remainder)}ms, so something reached a component bucket from outside the turn roots."
            )
        )
    return LatencyPartition(
        total_ms=total,
        llm_ms=llm,
        tool_ms=tool,
        # Clamped rather than published negative: within the tolerance the overrun is
        # float-addition noise on an exact partition, and a share of a whole that reads
        # as less than none of it would be a worse answer than the zero it really is.
        orchestration_ms=max(remainder, 0.0),
    )


#: The token a served-model coordinate (:attr:`ScoreRecord.served_model`) carries for candidate calls whose
#: response named no model, or that were stored before served models were recorded. Never the requested id.
SERVED_MODEL_UNRECORDED = "unrecorded"

#: How a served-model coordinate joins the models one result's candidate calls named. A result answered by two
#: models is itself a mixture, and its coordinate says so rather than picking one.
_SERVED_MODEL_JOIN = " + "

#: How many models answered a pooled set of candidate calls, as the provider's responses named them.
ServedModelState = Literal["one", "pooled", "unrecorded"]


def served_model_state(served_models: Collection[str], n_unrecorded: int) -> ServedModelState:
    """The one rule every served-model reading's state follows (#684).

    Args:
        served_models: The distinct models the responses named.
        n_unrecorded: The results with at least one candidate call whose response named none.

    Returns:
        ``pooled`` where two or more models were named, ``one`` where exactly one was and every call named it,
        ``unrecorded`` otherwise: whether one model answered cannot be established, and unknown is never one.
    """
    if len(served_models) > 1:
        return "pooled"
    return "one" if served_models and not n_unrecorded else "unrecorded"


class ResultServedReading(NamedTuple):
    """What one result's candidate calls say about the model that answered them.

    Attributes:
        requested: The candidate model the result's run asked for — the id an arm is keyed by.
        served: Every model the provider's responses named for its candidate calls.
        unrecorded: Some candidate row names no served model — a response that named none, or a row
            stored before served models were recorded.
    """

    requested: str
    served: frozenset[str]
    unrecorded: bool


def served_reading(result: EvalResult) -> ResultServedReading | None:
    """Read which model answered one result's candidate calls, off its candidate usage rows.

    ``RoleUsage.served_model`` only — what the provider's response named — and never ``RoleUsage.model``
    or the run's ``candidate_model``, which are what the launch asked for and, for a floating alias, name
    the pointer rather than the model behind it.

    Args:
        result: The result.

    Returns:
        The reading, or ``None`` when the candidate left no usage row: nothing was called, so nothing
        answered, and no claim is made about it.
    """
    rows = [row for row in result.usage if row.role == "candidate"]
    if not rows:
        return None
    return ResultServedReading(
        requested=result.model,
        served=frozenset(row.served_model for row in rows if row.served_model),
        unrecorded=any(not row.served_model for row in rows),
    )


def served_model_coordinate(reading: ResultServedReading | None) -> str | None:
    """One result's served models as the coordinate a pivot groups on and an export carries.

    Args:
        reading: The result's reading, from :func:`served_reading`.

    Returns:
        The named models, sorted and joined with `` + ``, with :data:`SERVED_MODEL_UNRECORDED` among them where
        some call named none (alone where none did). ``None`` where the candidate made no call.
    """
    if reading is None:
        return None
    return _SERVED_MODEL_JOIN.join(sorted(reading.served) + ([SERVED_MODEL_UNRECORDED] if reading.unrecorded else []))


class ServedModelReading(EvalBaseModel):
    """Which models answered the candidate calls a contestant, cell or row pooled (#684).

    A contestant is keyed by the model id its launch ASKED for, and a floating alias resolves on the provider's
    side, so runs of one contestant can have been answered by different models and still pool as one. The mixture
    is disclosed, not split: the key is fixed at launch, before any response names a model. The bundle's per-arm
    reading (``AnalysisContextBundle.arm_served_models``) is this one, keyed by arm.
    """

    served_models: list[str] = Field(
        default_factory=list,
        description=(
            "Every distinct model the provider's responses named as having answered these candidate calls, sorted. "
            "Never the requested id: a call whose response named no model adds nothing here and is counted in "
            "n_unrecorded."
        ),
    )
    n_results: int = Field(ge=1, description="The pooled results whose candidate calls left a usage row.")
    n_unrecorded: int = Field(
        ge=0,
        description=(
            "Of those, the results with at least one candidate call whose response named no model — or stored "
            "before served models were recorded. Not recorded, never a match with the requested id."
        ),
    )
    state: ServedModelState = Field(
        description=(
            "one = every candidate call named one and the same model. pooled = two or more models answered, so the "
            "numbers are a mixture of them. unrecorded = at most one model was named and some call named none, so "
            "whether one model answered cannot be established."
        )
    )

    @model_validator(mode="after")
    def _state_follows_the_counts(self) -> ServedModelReading:
        """The state is the one the served models and the unrecorded count imply, never a second opinion.

        Returns:
            The validated reading.

        Raises:
            ValueError: ``state`` disagrees with ``served_models`` and ``n_unrecorded``.
        """
        expected = served_model_state(self.served_models, self.n_unrecorded)
        if self.state != expected or self.n_unrecorded > self.n_results:
            raise ValueError(
                f"a pool with these served models and unrecorded calls is {expected!r}, not {self.state!r}"
            )
        return self

    def disclosure(self) -> str | None:
        """The reading in words, or ``None`` where one named model answered every call.

        Returns:
            A sentence for a pooled or unrecorded reading.
        """
        if self.state == "pooled":
            return (
                f"pools results answered by {len(self.served_models)} models ({', '.join(self.served_models)}) under "
                "one requested model, so its numbers mix those models"
            )
        if self.state == "unrecorded":
            named = f"; the rest named {self.served_models[0]}" if self.served_models else ""
            return (
                f"{self.n_unrecorded} of {self.n_results} result(s) made candidate calls whose response named no "
                f"model{named}, so whether one model answered them cannot be established"
            )
        return None


def pool_served_readings(readings: Iterable[ResultServedReading | None]) -> ServedModelReading | None:
    """Pool per-result readings into one, the rule every surface that groups by contestant reads.

    Args:
        readings: The pooled results' readings; ``None`` (no candidate call) contributes nothing.

    Returns:
        The pooled reading, or ``None`` when no result's candidate left a usage row.
    """
    own = [reading for reading in readings if reading is not None]
    if not own:
        return None
    models = sorted(set().union(*(reading.served for reading in own)))
    n_unrecorded = sum(1 for reading in own if reading.unrecorded)
    return ServedModelReading(
        served_models=models,
        n_results=len(own),
        n_unrecorded=n_unrecorded,
        state=served_model_state(models, n_unrecorded),
    )


def pooled_served_models(results: Sequence[EvalResult] | Sequence[ScoreRecord]) -> ServedModelReading | None:
    """Say which models answered the candidate calls a pooled figure rests on (#684).

    Shared, as :func:`pooled_cost_compositions` is, so the frontier, history, pivot and two-run comparison read
    one predicate, and the bundle's ``arm_served_models`` the same one (:func:`pool_served_readings`).

    Args:
        results: The results the caller pooled, or the rows a pivot cell pooled — read through
            :attr:`ScoreRecord.served_model`, once per result however many rows it put in the cell.

    Returns:
        The pooled reading, or ``None`` when no pooled result's candidate made a call.
    """
    readings: dict[str, ResultServedReading | None] = {}
    for result in results:
        if isinstance(result, ScoreRecord):
            if result.result_id not in readings:
                readings[result.result_id] = _reading_of_coordinate(result.served_model, result.model)
        else:
            readings[result.id] = served_reading(result)
    return pool_served_readings(readings.values())


def _reading_of_coordinate(coordinate: str | None, requested: str) -> ResultServedReading | None:
    """Read a served-model coordinate back into the reading :func:`served_model_coordinate` wrote it from."""
    if coordinate is None:
        return None
    tokens = coordinate.split(_SERVED_MODEL_JOIN)
    return ResultServedReading(
        requested=requested,
        served=frozenset(token for token in tokens if token != SERVED_MODEL_UNRECORDED),
        unrecorded=SERVED_MODEL_UNRECORDED in tokens,
    )


class ScoreRecord(EvalBaseModel):
    """One measurement at one fully-specified cell.

    A record exists only for a result that ran. The absence of a cell is not
    representable here and must not be: "did not run" is a property of a query's
    expected grid, not of an observation, and collapsing the two is how a missing
    cell comes to render as a zero.

    Coordinates are stable ids, never rendered strings. ``subject_label`` is
    carried as a display label only and must not be grouped on.

    **The measurement is ``metric``/``value``, except for a kind that grades itself.**
    A code-graded candidate has no composite and no judged dimension, so its grade
    rides on ``host_measures`` — the result's own mechanical measures, on its
    composite row alone. A judge-graded row and a code-graded row are then told
    apart by which of the two is populated, rather than by one of them being
    quietly empty everywhere the other is not.

    **Each identity key travels with the predicate version that produced it.**
    ``variant_key`` and ``context_key`` are derived by separate predicates that
    today share one :data:`~threetears.evals.kernel.identity.IDENTITY_VERSION` counter — so a
    bump for either moves the number stamped beside *both*, and a row can carry a
    version whose change did not touch its own predicate. Each key still needs its
    own column: a single ``identity_version`` on the row could only qualify one of
    them and would silently mis-qualify the other. A consumer that
    groups on one key gates comparability on *that* key's version: keys from
    different predicates are queryably distinct rather than silently regrouped,
    which is the whole reason the versions are stored beside the keys instead of
    only hashed into them.

    **The sentence above is an obligation, not a description of every surface that
    reads one.** The lenses that gate are ``frontier`` and ``history``, which group
    through :func:`_contestant_key` and carry the version in the key — but neither of
    those reads a ``ScoreRecord`` at all. The surface that reads this row is ``pivot``,
    whose axis set is deliberately open: ``_axis_value`` reads any declared coordinate by
    name, so ``row_factor="variant_key"`` groups the raw digest. The pivot does not gate
    there, because gating means deciding that one particular axis implies a partition,
    which the open-axis design does not let this module assert; it **discloses** instead
    (#672). A cell grouped on an identity key that pools more than one version of that
    key's predicate names the versions (:attr:`PivotCell.identity_versions`), and the
    table says which cells do (:attr:`PivotTable.identity_pooling_disclosure`).
    ``variant_identity_version`` is itself a declared coordinate, so pivoting the key
    against it separates the versions.

    The sentence stood here unqualified for some time while nothing did it, which is how
    the wrong merge it forbids reached production on the two lenses that now gate.

    **``context_identity_version`` is deliberately still consumerless, and that is not the
    same gap.** The variant side had a live wrong merge because two lenses GROUPED on
    ``variant_key``. Nothing groups on ``context_key`` anywhere: ``comparison_sets``
    partitions on ``(subject_id, template_id)`` and reads the key only to raise
    ``BADGE_CONTEXT_DIFFERS``, which is a per-pair comparison that already fails closed when
    the keys differ for any reason, version included. So there is no pooling decision for a
    context-side version gate to correct — the field waits for a consumer that groups, and
    stamping one today would be a gate over nothing.
    """

    # --- The comparability partition ---
    subject_id: str
    subject_label: str = ""

    # --- Location ---
    #
    # `scope_id` is the storage scope the observation was read under, and the engine
    # never interprets it: nothing branches on its value, it is not an identity
    # component, and no pooling or comparability rule reads it. **A caller may still
    # pivot on it** — the axis set is deliberately open, so `_axis_value` reads any
    # declared coordinate by name and "cost by scope" is a legitimate question. Opaque
    # means the engine assigns it no meaning, not that it is unqueryable.
    scope_id: str
    run_id: str
    result_id: str

    # --- Cell coordinates ---
    #
    # The two identity keys resolve at different granularities and that is a
    # property of what they measure, not an inconsistency to normalize away: a
    # variant is the resolved contestant stack, which varies per result because a
    # run carries several candidate models, while the measurement context is
    # pinned once for the whole run.
    template_id: str | None = None
    test_case_id: str
    model: str
    k_iteration: int
    variant_key: str | None = None
    variant_identity_version: int | None = None
    context_key: str | None = None
    context_identity_version: int | None = None

    # The run-pinned judge and simulator roles, carried as first-class
    # coordinates rather than only hashed into `context_key`'s roles component.
    # A digest is enough to gate comparability (badge that the roles differ,
    # per `comparison_sets`) but cannot be grouped ON: "which runs were pinned
    # to the same judge" is a pivot the hash cannot answer. Sourced run-level —
    # the roles are pinned once per run, exactly as `context_key` composes them
    # — so a pivot on `judge_model` partitions on the same value the
    # comparability badge reads, and the two surfaces cannot disagree about
    # which runs used the same judge. `None` means the run was not judged (judge) or ran
    # no simulated user, or recorded none (simulator). Clean-snapshot provenance stays
    # in the digest, un-pivotable, until a query needs it; cassette mode is carried below,
    # because a cost pivot needed it.
    #
    # **`judge_model` is the run's PIN and is NOT the model that scored this
    # row's dimension.** Under the judge-model cascade (role default < run pin <
    # per-dim `JudgeConfig.model`) a dim whose config names a model is scored by
    # that instead, so on a `score` row the pin is a fallback the dimension may
    # never have used — reading it as the judge of record attributed a
    # gemini-scored dimension to the run's gpt pin, on the same export whose
    # `run_summary` correctly reported two judges. `rubric_dim_judge_model`
    # below is the per-row answer; this one stays run-level so the badge
    # agreement above survives, and the two coordinates are kept apart rather
    # than one of them being made to mean both things.
    judge_model: str | None = None
    simulator_model: str | None = None

    # The cassette mode the run RECORDED (`off`, `capture` or `replay`), run-level like the
    # judge pin above. Carried because a replayed result did not spend what a live one does:
    # a replayed background delivery spent no inner-agent dollars, so its `cost_usd` is
    # smaller for a reason in the apparatus, not the configuration (#658). A cost pivot reads
    # it to withhold a cell that pools replayed with live results, the export carries it as a
    # column, and a caller can pivot on it like any other coordinate. `None` only on a row
    # built without a run.
    cassette_mode: str | None = None
    # How many of this result's async deliveries a harness supplied rather than the candidate's
    # background work producing them — a replayed capture or a seeded finding
    # (`count_substituted_deliveries`). The per-result half of the same fact: a seeded finding
    # substitutes in a run that recorded `off`, so the run's mode alone cannot say which results
    # spent less. Non-zero is what withholds the result's production-replicating cost.
    substituted_deliveries: int = 0

    # The model the provider's responses NAMED as having answered this result's candidate calls (#684) —
    # `served_model_coordinate`, read off `RoleUsage.served_model`, never off `model`, which is the id the
    # launch asked for and, for a floating alias, the pointer rather than the model behind it. A declared
    # coordinate so a pivot can split by it and the export carries it as a column. Several models a result's
    # calls named are joined with ` + `; `unrecorded` stands for calls whose response named none (or rows stored
    # before served models were recorded) and is never read as the requested id. `None` where the candidate
    # made no call at all.
    served_model: str | None = None

    created_at: str = ""

    # Which rubric dimension this row measures — a coordinate like `model` or
    # `test_case_id`, so it can go on either axis or into a filter with no new
    # machinery, and `_reads_a_lever` reads it as the declared field it is.
    #
    # Populated only on `score` rows, and `None` there too when the result
    # carried no dims at all. A composite or cost row leaves it `None` because it
    # measures the result as a whole: those metrics have no dimension, and
    # inventing one would make a `rubric_dim` pivot of composites look
    # per-dimension when it is not.
    rubric_dim: str | None = None

    # The scale `rubric_dim` was judged on, set exactly where `rubric_dim` is: `value` is 1-5 on
    # `ordinal` and 1/0 on `pass_fail`, so a mean over rows of one dimension is a level or a pass
    # rate, and a pooling across the two is neither.
    rubric_scale: RubricScale | None = None

    # The goal-state check a `goal_state` row reports, as its expression — set on those rows and on no other.
    # A coordinate of its own rather than `rubric_dim`, which is pinned to template rubric dimensions: a check
    # is code-graded and a dimension is judged, and a `rubric_dim` pivot that pooled the two would mean
    # neither. A pivot or export can key on it like any other coordinate.
    goal_check: str | None = None

    # Which dimensions the COMPOSITE on this result was meaned over, sorted.
    #
    # Populated on `composite` rows alone, because it qualifies that number and no
    # other. A composite is a mean across whatever dims the result happened to
    # carry, so two composites are comparable only when they were meaned over the
    # SAME dims — otherwise one averages three things and the other five, and
    # their difference is partly a difference in what was being averaged rather
    # than in what was measured. Nothing recorded this before, so every pooled
    # quality comparison made that assumption silently and none of them could be
    # checked; a consumer can now gate on it instead of trusting that a template
    # never changed mid-campaign.
    #
    # `None` on a row whose composite is itself None (infra-excluded, or no dims
    # at all) — there is no basis for a number that does not exist, and an empty
    # list would read as "meaned over nothing", which is a different claim.
    dimension_basis: list[str] | None = None

    @model_validator(mode="after")
    def _a_composite_states_what_it_was_meaned_over(self) -> ScoreRecord:
        """A quality number with a value must name its basis; one without must not claim a basis.

        The structural half of the disclosure, and it lives HERE rather than on a companion type
        because this is the row that ships. A second model carrying the rule would be the concept
        landing twice with the guarantee on the copy nothing is on the path of — which is exactly
        how a pooled number with no basis stays constructible while a docstring says it cannot be.

        Returns:
            The validated record.

        **Three states, and the empty list is one of them.** ``None`` is "nobody recorded a
        basis", which is the defect. ``[]`` is "meaned over nothing", which is a real and common
        answer: a candidate failure scores 0.0 by policy rather than by averaging, so it has a
        value and legitimately no dimensions behind it. A non-empty list is a real mean. Refusing
        ``[]`` alongside ``None`` would reject every candidate-failure row and force the producer
        to lie about one of the two.

        Raises:
            ValueError: A composite carries a value and no basis was recorded at all, so nothing
                downstream can tell whether it is comparable with the next one; or it records a
                basis beside no value, which claims a mean that was never taken.
        """
        if self.metric != METRIC_COMPOSITE:
            return self
        if self.value is not None and self.dimension_basis is None:
            raise ValueError("a composite with a value must state the dimensions it was meaned over")
        if self.value is None and self.dimension_basis is not None:
            raise ValueError("a composite with no value cannot have been meaned over anything")
        return self

    # WHICH MODEL SCORED `rubric_dim` — the per-row half of the judge question
    # `judge_model` above cannot answer. Declared as a field, so it is pivotable,
    # filterable and exported by exactly the same machinery `rubric_dim` is, with
    # no per-coordinate work anywhere.
    #
    # `None` wherever no SINGLE model scored the row. A composite or cost row
    # measures the result as a whole, spanning every dim the judge scored, so
    # naming one model would make a `rubric_dim_judge_model` pivot of composites
    # look per-dimension when it is not; a dimensionless `score` row (the judge
    # scored nothing) has nothing to attribute either.
    #
    # That rule is NOT "null exactly where `rubric_dim` is null", which it used to
    # say and which the dual-score axes falsify: a `__transcript__` / `__outcome__`
    # row carries no `rubric_dim` (the metric IS the axis) and was scored by exactly
    # one judge, which the run's own map can name. Withholding it there would drop a
    # real attribution to preserve a coincidence between two fields.
    #
    # `None` ALSO where the run cannot say — see `dim_judge_model`, which is the
    # one definition of this value and reads the same map the run summary's
    # "Judged by" block and `bisect_runs`' `judge_dim_divergence` read. A cell in
    # a pivot or a CSV carries no caveat channel, so a reconstruction is not
    # published here as though it were a record.
    rubric_dim_judge_model: str | None = None

    # --- Open coordinates: every lever the host's registry resolves for the run ---
    #
    # A kind's overlays (`gm.difficulty`), an open family's members, and anything a
    # host registers next. These live in a map rather than as declared fields
    # because the factor set is deliberately OPEN: a new lever has to become
    # pivotable with no change to this model, no migration, and no per-key
    # rendering work. Enumerating them as fields would mean every new lever is a
    # schema change plus an edit in each read surface, which is the coupling the
    # open set exists to prevent.
    #
    # The *identity rule* stays closed even though the set does not: anything
    # that can change an outcome is either a recorded coordinate or explicitly
    # declared noise. A key absent here is absent because the run carried no
    # level for it, not because reporting declined to carry it.
    factors: dict[str, str] = {}

    # --- The measurement ---
    metric: str
    value: float | None = None

    # The roles a `cost_usd` row's dollars were summed over (`EvalResult.cost_roles`), set on
    # cost rows alone (#625). What the sum covers differs per run — metered third-party spend
    # enters only for a run whose operator declared a rate — so two equal totals can cover
    # different things, and a cost pivot reads this to say which compositions each cell pooled
    # (`pooled_cost_compositions`, the reading every pooled cost surface shares). `None` on every
    # other row, which measures no dollars.
    cost_roles: list[str] | None = None

    @model_validator(mode="after")
    def _only_a_cost_row_names_its_cost_roles(self) -> ScoreRecord:
        """A cost composition qualifies a cost and nothing else.

        Returns:
            The validated record.

        Raises:
            ValueError: A non-cost row carries a cost composition.
        """
        if self.metric != METRIC_COST_USD and self.cost_roles is not None:
            raise ValueError(f"a {self.metric!r} row measures no dollars, so it has no cost composition")
        return self

    # --- The kind's own grade, for a kind whose scoring is code rather than a judge ---
    #
    # `EvalResult.host_measures` carried straight through: the mechanical grade the
    # candidate kind computed itself, by registered measure name. An extractor reports
    # `field_accuracy` here (a classifier reports `match` and `confusion_cell`, from which the
    # engine derives accuracy itself); a judge-graded kind reports nothing and the map is empty.
    #
    # It exists because this projection was judge-centric — composite, cost, per-dimension
    # score — and a code-graded run has none of those. Its composite is null (there are no
    # rubric dims to mean), its score row is the dimensionless null, and its actual grade
    # was nowhere on the export at all, so "which case did this arm miss" could not be
    # asked of the surface built to answer it.
    #
    # AN OPEN MAP, for the same reason `factors` is one and NOT for the same reason: a
    # host declares whatever it measures, so a kind that grades on two axes rather than one
    # adds a key here and a column to the CSV, with no edit to this model and no second
    # shape for the two-grade case. What stays closed is the *metric* namespace — these are
    # deliberately not rows of their own, because `PROJECTED_METRICS` is closed precisely so
    # an unknown metric can be refused as a typo, and an open host catalogue in that
    # namespace would end that. They are also not `value`s on the composite row: a 0/1
    # mechanical grade and a 0-1 judge composite are different quantities, and pooling them
    # under one metric is exactly the silent difference this field exists to end.
    #
    # CARRIED ON THE COMPOSITE ROW ALONE — the result's quality row, enforced by
    # `_only_the_quality_row_carries_the_hosts_grade` below. Repeating it on every row of a
    # result would make a spreadsheet's mean over the column count each observation as many
    # times as the result has rows, which is the same inflation the per-observation
    # composite comment above refuses one field over. One row per result carries the grade,
    # so averaging the column is the arm's accuracy and filtering it to 0 names the cases
    # that arm missed.
    host_measures: dict[str, bool | float | str] = {}

    @model_validator(mode="after")
    def _only_the_quality_row_carries_the_hosts_grade(self) -> ScoreRecord:
        """A result's mechanical grade rides on its quality row, and on no other.

        Structural rather than a convention held by the producer, for the reason the
        composite's own validator is: the property a reader relies on is "one row per result
        carries this", and a second row carrying it makes every mean over the column silently
        wrong by a factor of however many rows that result emitted. Nothing downstream could
        detect that, because each individual value would be right.

        Returns:
            The validated record.

        Raises:
            ValueError: A non-composite row carries a host grade.
        """
        if self.metric != METRIC_COMPOSITE and self.host_measures:
            raise ValueError(
                f"a {self.metric!r} row cannot carry the host's grade — it belongs to the result, and the "
                "composite row is the one row per result that carries it"
            )
        return self

    # --- How this observation participates in scoring ---
    outcome: str


class ProjectionExclusions(EvalBaseModel):
    """What the projection dropped on the way to producing rows, and why.

    Carried on the wire rather than only logged. A corpus whose results cannot be placed
    projects to zero rows, and a surface that reports only "no
    observations" renders it identically to a scope nobody has ever run —
    which is the single most likely misreading of this whole tier, because the
    honest answer ("the data exists but cannot be grouped") and the wrong one
    ("there is no data") differ by a fact the operator has no other way to get.

    The classes are kept apart because they mean opposite things to the reader.
    ``results_without_run`` is data that *cannot* be placed — an integrity gap
    worth investigating.
    ``results_outside_queried_runs`` and ``results_from_archived_runs`` are data
    deliberately left out because the corpus was narrowed, which is the answer
    working as asked. Pooling a deliberate class with an unplaceable one was the
    original defect: the caller's own ``status="completed"`` default filters out
    every in-flight run, so the routine case reported its results under a counter
    documented as unplaceable.

    **The two deliberate classes are kept apart for a further reason — they have
    different remedies.** A run leaves the reported cohort either because the
    operator archived it or because the caller's ``status`` filter did not match
    it, and the fix differs: ``run_archive(archived=false)`` versus a wider
    ``status``. Counting both as ``results_outside_queried_runs`` let every read
    surface tell an operator to pass ``status='all'`` for observations that
    ``status='all'`` provably cannot reach — worse than saying nothing, because
    the advice can be followed exactly and return the identical answer.

    **Archival takes precedence when both apply**, because un-archiving is the
    step that is *necessary* in that case while a status widening alone would
    still not reach the run. The surfaces word the archival disclosure to stay
    true there: it says un-archiving returns the run to the reported cohort,
    where the ``status`` filter then applies as usual, rather than promising the
    observations reappear.
    """

    results_without_run: int = 0
    results_outside_queried_runs: int = 0
    results_from_archived_runs: int = 0

    @property
    def total(self) -> int:
        """Total observations dropped before any aggregation.

        Returns:
            The sum of every exclusion class. Deliberately includes the
            deliberately-filtered classes: the count exists to answer "is this
            scope empty, or did I just not ask for what is in it", and every
            cause makes an empty table non-empty underneath.
        """
        return self.results_without_run + self.results_outside_queried_runs + self.results_from_archived_runs


class ScoreProjection(EvalBaseModel):
    """The projected rows plus an account of everything that did not become one.

    The two travel together because they are only meaningful together: ``n=0``
    means something different depending on whether anything was excluded, and
    separating them is what lets a caller forget to ask.

    ``completeness_disclosures`` is the third of the same kind, and it is about
    rows that ARE here rather than rows that are not: a run that measured less
    than the matrix it promised still projects rows, and pooling them with a full
    run's computes a rate over two different populations. It rides on the
    projection so an aggregator cannot render the rows without the sentence.
    """

    records: list[ScoreRecord] = []
    exclusions: ProjectionExclusions = ProjectionExclusions()
    #: ``run_id -> DEGRADED sentence`` for every placed run that came up short of
    #: its matrix, from :func:`degraded_run_disclosures`. Restricted to runs that
    #: actually contributed a row: a degraded run whose every result was excluded
    #: qualifies nothing here, and disclosing it would warn about a population the
    #: answer does not contain.
    completeness_disclosures: dict[str, str] = {}


def _subject_id_of(run: EvalRun, result: EvalResult | None = None) -> str:
    """Resolve the real subject a measurement is attributed to.

    Two fields carry the same fact, populated by different generations of the
    runner: ``EvalResult.subject_id`` is per-result attribution and is ``None``
    on stub slots and pre-isolation rows, while ``SubjectSnapshot.subject_id`` is
    per-run and always present. The result-level field is preferred as the more
    specific one; the run's snapshot is the fallback. The fallback cannot
    misattribute, because it is read from the run that produced this very result.

    Callers differ in what they can offer, and the asymmetry is structural rather
    than an oversight: :func:`project_score_records` works per result and passes
    both, while :func:`compute_comparison_sets` groups *runs* and has no results to pass,
    so it always resolves through the snapshot. The two agree because both fields
    are written from the same capture — but if they ever disagreed, the row and
    the grouping would partition on different keys, which is why the preference
    order is defined in one place instead of at each call site.

    Args:
        run: The run supplying the snapshot fallback.
        result: The result whose attribution is preferred, when available.

    Returns:
        The subject key, never blank. ``SubjectSnapshot`` refuses a blank one, so the fallback
        below always answers — which is what retired the whole family of "excluded: no captured
        identity" counters this function used to feed. A blank subject was never a group of its
        own; it is now not a state.
    """
    attributed = (result.subject_id or "").strip() if result is not None else ""
    if attributed:
        return attributed
    return run.subject_snapshot.subject_id


class PlacedResult(NamedTuple):
    """One result matched to its run and its resolved subject, ready to group.

    The unit of :func:`place_results`. Carrying the ``run`` beside the result
    spares each caller a second lookup for the coordinates it reads off the run
    (subject name, template, ``created_at``, the frozen case set), and pins the
    ``subject_id`` the placement resolved so no caller re-derives it — resolving
    it twice is how two surfaces would drift on what "same subject" means.
    """

    run: EvalRun
    result: EvalResult
    subject_id: str


def place_results(
    runs: list[EvalRun],
    results: list[EvalResult],
    known_run_ids: set[str] | None,
    *,
    source: str,
    archived_run_ids: set[str] | None,
) -> tuple[list[PlacedResult], ProjectionExclusions]:
    """Match each result to its run and resolved subject, accounting for every drop.

    The prologue shared by :func:`project_score_records`, :func:`compute_frontier`, and
    :func:`compute_history`: a result cannot be placed without its run's coordinates, and
    a run with no captured subject identity cannot be grouped, because every
    consumer partitions on subject and a blank identity would pool unrelated
    subjects into one bucket. Both skips are **returned**, not merely logged —
    they are the two exclusions that turn a real corpus into an empty result set,
    so a surface that renders only the placed rows reports "nothing here" for data
    that exists and simply cannot be grouped.

    Extracted so the three surfaces cannot diverge on which observations they
    drop or how they classify the drop — an on-request narrowing must never be
    counted as an integrity gap, and vice versa. Callers apply their own further
    filtering (a subject restriction, a metric selection) over the placed rows.

    Args:
        runs: The runs supplying cell coordinates. Order is irrelevant.
        results: The observations to place. Output preserves this order.
        known_run_ids: Every run id that exists in the corpus, including ones a
            caller's own filter kept out of ``runs``. A result whose run is in
            this set but not in ``runs`` was excluded *on request* and is counted
            ``results_outside_queried_runs``; one whose run is in neither is
            genuinely unplaceable and counted ``results_without_run``. ``None``
            asserts ``runs`` is the whole corpus.
        source: The calling function's name, used only in the skip log line so a
            reader can tell which surface dropped rows.
        archived_run_ids: The corpus run ids the operator has archived. A result
            whose run is in this set is counted ``results_from_archived_runs``
            instead of ``results_outside_queried_runs`` — the two narrowings have
            different remedies, and only one of them is a wider ``status``.
            ``None`` asserts no archival narrowing was applied, which is correct
            only for a caller whose cohort already includes archived runs; it is
            never a licence to leave archival exclusions attributed to a filter
            that did not make them. Required with no default, here and on the
            three surfaces that forward it, so a caller that never thought about
            archival fails with ``TypeError`` instead of blaming ``status``.

    Returns:
        The placed results in input order — each carrying a non-blank
        ``subject_id`` — plus the exclusion accounting.
    """
    runs_by_id = {run.id: run for run in runs}

    placed: list[PlacedResult] = []
    skipped_no_run = 0
    skipped_outside_query = 0
    skipped_archived = 0

    for result in results:
        run = runs_by_id.get(result.eval_run_id)
        if run is None:
            if known_run_ids is not None and result.eval_run_id in known_run_ids:
                # Archival is checked FIRST, so a run that is both archived and
                # outside the caller's status narrowing is reported as archived:
                # un-archiving is the necessary step there, and a wider `status`
                # on its own would still not reach it.
                if archived_run_ids is not None and result.eval_run_id in archived_run_ids:
                    skipped_archived += 1
                else:
                    skipped_outside_query += 1
            else:
                skipped_no_run += 1
            continue

        placed.append(PlacedResult(run=run, result=result, subject_id=_subject_id_of(run, result)))

    if skipped_no_run:
        # Only the unplaceable class is logged. A result filtered out on
        # request is not an anomaly, and logging it would make the routine
        # `status="completed"` query emit a warning-shaped line every time.
        log.info("%s skipped rows: %d without a supplied run", source, skipped_no_run)
    return placed, ProjectionExclusions(
        results_without_run=skipped_no_run,
        results_outside_queried_runs=skipped_outside_query,
        results_from_archived_runs=skipped_archived,
    )


#: The types :func:`lever_level` renders through ``str()`` rather than as canonical JSON.
SCALAR_LEVEL_TYPES: tuple[type, ...] = (str, int, float, bool)

#: The level a lever sits at when the launch NAMED it with the value ``null`` — the operator set it
#: to nothing, which is a level, and is not the inherited ``'—'`` of a run that never named it.
#: JSON's own spelling, so it is the token the structured branch of :func:`lever_level` already
#: produced for ``None`` and the pivot and the export (which render every resolved value through
#: :func:`lever_level`) agree with the coverage map about it. Like every scalar it can collide with a
#: string level spelled the same (``"1024"`` and ``1024`` share one already); that is the rendering's
#: accepted limit, not a reason to spell a null as something JSON does not.
NULL_LEVEL = "null"


def lever_level(value: Any) -> str:
    """Render one lever's resolved value as the level a cohort is keyed on.

    A scalar renders as itself — ``1024``, not ``"1024"`` — matching the overlay spelling every
    surface downstream already carries. Anything structured renders as canonical JSON, which is a
    stable comparable key rather than a Python repr whose ordering could split two runs that
    resolved alike.

    **Content, never a display.** A host's own display for a level is more readable (``"20 chars ·
    3f9a1c"``) and is the wrong thing to bin on: a display is an abbreviation, two different levels
    can share one, and cohorts keyed on it would MERGE two variants — the direction nothing
    downstream can undo. :attr:`~threetears.evals.kernel.campaign.VariantIndexEntry.levers` carries
    the display beside the content hash for a reader who needs to recognise the level.

    **``None`` is** :data:`NULL_LEVEL`, **deliberately.** A caller hands one in only for a lever the
    launch named as ``null`` — a reader returning nothing is "the lever does not apply", and the
    caller skips it before reaching here — so the null is a level the operator set.

    Args:
        value: The resolved value, JSON-safe by the host's contract.

    Returns:
        The level.
    """
    if value is None:
        return NULL_LEVEL
    return str(value) if isinstance(value, SCALAR_LEVEL_TYPES) else canonical_json(value)


def _run_factors(
    placed: Sequence[tuple[EvalRun, EvalResult, str]], *, profile: HostProfile
) -> dict[str, dict[str, str]]:
    """Each placed run's open coordinates: every lever its host's registry resolves for it.

    Read through the registry and nowhere else — the fixed levers, the members of every open family
    the run carried, and a kind's overlays among them (each a lever its kind contract derives) — so a
    lever reaches a pivot and the export under the name the host gave it, with no per-lever work
    here. The candidate model is not among them: it is a declared coordinate of every record.

    Absence is preserved as absence: a lever whose reader returns nothing for this run (a lever of
    another kind, or one the run never recorded) has no entry, so a pivot groups those runs under the
    visible ``"—"`` level rather than inventing one.

    Resolved once per run, not per result: the host's reader pass is one unit of work, and asking it
    again per row would pay for it k × cases times for one answer.

    Args:
        placed: The ``(run, result, subject_id)`` triples being projected.
        profile: The host whose vocabulary this reads.

    Returns:
        Run id -> its coordinate map.
    """
    results_by_run: dict[str, list[EvalResult]] = {}
    runs: dict[str, EvalRun] = {}
    for run, result, _ in placed:
        runs.setdefault(run.id, run)
        results_by_run.setdefault(run.id, []).append(result)
    sweepables = profile.sweepables
    return {
        run_id: {
            lever: lever_level(value)
            for lever, value in sorted(sweepables.resolve_levers(run, results_by_run[run_id]).values.items())
            if lever != CANDIDATE_MODEL_LEVER and value is not None
        }
        for run_id, run in runs.items()
    }


def dim_judge_model(run: EvalRun, rubric_dim: str | None) -> str | None:
    """Name the model that scored ONE rubric dimension of one run.

    A lookup on the run's own attribution map, deliberately not a second
    evaluation of the judge-model cascade. The cascade is applied once, at launch,
    by :func:`~threetears.evals.schema.models.resolve_effective_judges`, and its answer is
    stored on :attr:`~threetears.evals.schema.models.EvalRun.effective_judges` — the same
    map the run summary's "Judged by" block renders and
    the ``judge_dim_divergence`` sweepable bisects on. Re-deriving it here
    would put a fourth reader on a rule that has already had three, and the whole
    point of storing the resolved attribution was that no reader has to re-run the
    cascade against today's configs.

    **Reads the HASHABLE map, so a reconstruction never answers.** A ``derived``
    attribution infers each dim's config from the records stored as of the run's
    start; it is displayable with a caveat, which is why
    a host's judge listing may show it, saying so. A pivot axis level and a CSV cell have no caveat channel — an inferred
    judge published there is indistinguishable from a recorded one, and a reader
    would group observations under a model that may never have scored them. The
    same reasoning keeps a reconstruction out of the context key and out of the
    comparison badge, and this function reads the same property they do rather
    than restating the eligibility rule.

    Args:
        run: The run that produced the observation.
        rubric_dim: The dimension being attributed, or ``None`` for a
            whole-result measure that spans every dim and has none.

    Returns:
        The model that scored ``rubric_dim``, or ``None`` when there is no single
        dimension to attribute, when the run recorded no attribution (or only a
        reconstruction), or when its recorded attribution has no entry for this
        dim — a dim the run never committed to a judge for is one nobody can
        name, and falling back to the run pin here is the exact
        misattribution this coordinate exists to end.
    """
    if rubric_dim is None:
        return None
    judges = run.hashable_effective_judges
    if judges is None:
        return None
    return judges.get(rubric_dim)


def project_score_records(
    runs: list[EvalRun],
    results: list[EvalResult],
    *,
    known_run_ids: set[str] | None = None,
    archived_run_ids: set[str] | None,
    profile: HostProfile,
) -> ScoreProjection:
    """Flatten runs + results into one row per (cell, measure).

    Results whose run is absent from ``runs`` are skipped — a row cannot be
    placed without its run's coordinates, and inventing them would fabricate a
    cell. Results whose run has no captured subject identity are also skipped,
    because every consumer of this projection partitions on subject and a blank
    identity would pool unrelated subjects into one bucket.

    Both skips are **returned**, not merely logged. A log line is invisible to
    the operator reading the answer, and these two exclusions are exactly the
    ones that turn a corpus into an empty result set — so a caller that renders
    the rows without the counts reports "nothing here" for data that exists and
    simply cannot be grouped.

    Args:
        runs: The runs supplying cell coordinates. Order is irrelevant.
        results: The observations to project.
        known_run_ids: Every run id that exists in the corpus, including ones a
            caller's own filter kept out of ``runs``. A result whose run is in
            this set but not in ``runs`` was excluded *on request* and is
            counted as ``results_outside_queried_runs``; one whose run is in
            neither is genuinely unplaceable and counted as
            ``results_without_run``. ``None`` asserts that ``runs`` is the whole
            corpus — correct for an unfiltered caller, and the only case where
            every absent run really is missing.
        archived_run_ids: The corpus run ids the operator has archived, so an
            archival exclusion is counted as ``results_from_archived_runs``
            rather than under the caller's ``status`` filter. See
            :func:`place_results`.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`ScoreProjection` — the rows, plus the exclusion counts and the
        completeness disclosures of every run that contributed a row while having
        measured less than its matrix. Every
        projected result contributes a row for every measure, carrying
        ``value=None`` when that measure has no value for it. A row therefore
        records *that the cell was attempted* independently of whether it was
        measured, which is what lets a consumer tell "tried and unmeasured" from
        "never tried". The metrics in :data:`SCOPED_METRICS` emit several rows for one result, each
        keyed by its scope coordinate. :data:`METRIC_SCORE` emits one per judged rubric dimension, keyed by
        :attr:`ScoreRecord.rubric_dim` and attributed to the model that scored
        *that* dimension on :attr:`ScoreRecord.rubric_dim_judge_model`, which is
        the run's judge pin only when the dimension carried no judge config of
        its own — and emits a single dimensionless null row when the judge scored
        none.
    """
    from threetears.evals.kernel.usage_capture import count_substituted_deliveries

    placed, exclusions = place_results(
        runs, results, known_run_ids, source="project_score_records", archived_run_ids=archived_run_ids
    )

    records: list[ScoreRecord] = []
    factors_by_run = _run_factors(placed, profile=profile)

    for run, result, subject_id in placed:
        coordinates: dict[str, Any] = {
            "subject_id": subject_id,
            "subject_label": run.subject_snapshot.subject_label,
            "scope_id": result.scope_id,
            "run_id": result.eval_run_id,
            "result_id": result.id,
            "template_id": run.template_id,
            "test_case_id": result.test_case_id,
            "model": result.model,
            "k_iteration": result.k_iteration,
            "variant_key": result.variant_key,
            "variant_identity_version": result.identity_version,
            "context_key": run.context_key,
            "context_identity_version": run.identity_version,
            "judge_model": run.judge_model,
            "simulator_model": run.simulator_model,
            "cassette_mode": run.cassette_mode,
            "substituted_deliveries": count_substituted_deliveries(result),
            "served_model": served_model_coordinate(served_reading(result)),
            "created_at": run.created_at,
            "factors": factors_by_run[run.id],
            "outcome": classify_result(result).value,
        }

        # Per-OBSERVATION composite, not the per-case mean over k. A row is one
        # measurement at one cell, and its coordinates carry `k_iteration` and
        # `result_id` — so writing a k-averaged value into each of them would
        # emit N identical rows claiming to be N distinct measurements, which
        # inflates every downstream `n` and drives within-case dispersion to
        # zero. Dispersion across k is precisely what the aggregation layer must
        # be able to compute from these rows.
        #
        # `result_composite` returns None twice over — for an infra-excluded
        # result, and for an ok result carrying no rubric dims. Both are
        # *unmeasured*, not zero, and the row is emitted anyway with a null
        # value. Aggregators drop nulls from the mean, so a harness failure
        # still cannot be absorbed into a quality figure; what the row adds is
        # the record that the cell was attempted at all.
        #
        # Emitting nothing was the earlier shape, and it made "tried and
        # unmeasured" indistinguishable from "never tried": a cell whose every
        # observation failed in the harness had no composite record, fell to the
        # empty-cell branch, and reported that the combination was never
        # attempted. It also silently zeroed the per-cell unmeasured count,
        # since a row that does not exist cannot be counted as dropped.
        # The basis travels with the number it qualifies, derived from the same
        # `rubric_scores` the composite is meaned over rather than from the
        # template's declared dims — a dim the template declares and this result
        # never scored is not in the mean, so naming it here would describe a
        # basis the arithmetic did not use.
        #
        # The host's own grade rides on THIS row, and the two are the same question asked of
        # two kinds of candidate: what this cell was worth. A judge-graded result answers it
        # in `value`; a code-graded one has no rubric dims to mean and answers it in
        # `host_measures`, which the export flattens to a column per measure. Carried
        # verbatim — an unregistered measure name is the registry's defect to report, not
        # this projection's to filter — and `{}` for every kind that grades with a judge, so
        # a judge-graded export gains no column at all.
        composite = result_composite(result)
        # A composite the judge could not complete is unmeasured for a reason of its own, and the
        # row says which, so a counter can tell it from a fault rather than from nothing.
        composite_coordinates = (
            {**coordinates, "outcome": JUDGE_CANNOT_TELL_OUTCOME}
            if trial_exclusion(result) == "judge_cannot_tell"
            else coordinates
        )
        records.append(
            ScoreRecord(
                metric=METRIC_COMPOSITE,
                value=composite,
                dimension_basis=composite_basis(result),
                host_measures=result.host_measures,
                **composite_coordinates,
            )
        )
        records.append(
            ScoreRecord(
                metric=METRIC_COST_USD, value=result.cost_usd, cost_roles=list(result.cost_roles), **coordinates
            )
        )

        # One row per judged dimension, carrying the RAW score on the dimension's own scale
        # (1-5, or 1/0 for pass/fail, named by `rubric_scale`) — not the 0-1 composite scale. Read straight off
        # `rubric_scores`, which holds the template dims alone: the reserved
        # `transcript_score` / `outcome_score` axes live in their own fields and
        # are emitted just below under METRIC NAMES of their own, so no reserved id
        # ever reaches `rubric_dim`.
        #
        # Excluding them is what lets these rows reproduce
        # `compute_dimension_summary`'s per-dimension means — under three
        # conditions, all of which are properties of the aggregation rather than
        # of these rows, and none of which this comment may assert away:
        # `compute_dimension_summary` divides by OBSERVATIONS, so the pivot
        # matches it under `sample_weighted` and not under the
        # `equal_per_scenario` default that gives each test case one vote; this
        # projection drops results it cannot place (see `exclusions`), so the two
        # agree only over a corpus where every result is placeable.
        #
        # **The value is what the dim counts as, from `counted_rubric_scores`** — the rule
        # `compute_dimension_summary` reads too. A harness-faulted result's rows are null (the
        # judge read a transcript the rig broke); a candidate failure's count at the
        # scale's floor (a turn the user got nothing from is not a
        # pass on any dimension, whatever the judge made of the silence). The judge's raw
        # reading stays on the result, where `get_result` shows it.
        #
        # Each dim row is also attributed to the model that scored THAT dim,
        # which the run-level `judge_model` in `coordinates` cannot express: the
        # pin is what a dim falls back to, and a dim whose `JudgeConfig` names a
        # model was scored by that instead. It is set per row rather than in
        # `coordinates` for the same reason `rubric_dim` is — the composite and
        # cost rows above span every dim and have no one judge to name.
        #
        # A result with no dims still emits one row, null-valued and with no
        # dimension. Same reason the composite row is emitted for an unmeasured
        # result: the row records that the cell was ATTEMPTED, and dropping it
        # would make "the judge scored no dimensions here" indistinguishable
        # from "nobody ran this combination".
        #
        # A dim the judge answered it could not tell on gets its own null row, outcome
        # ``judge_cannot_tell``: the trial is out of that dim's mean, and the row is how a
        # counter learns it is out and why, instead of the dim simply having fewer rows.
        cannot_tell_dims = [dim for dim in result.judge_cannot_tell if dim not in RESERVED_DIM_IDS]
        counted_by_dim = {score.dim: value for score, value in counted_rubric_scores(result) or ()}
        for rubric_score in result.rubric_scores:
            counted_value = counted_by_dim.get(rubric_score.dim)
            records.append(
                ScoreRecord(
                    metric=METRIC_SCORE,
                    rubric_dim=rubric_score.dim,
                    rubric_scale=rubric_score.scale,
                    rubric_dim_judge_model=dim_judge_model(run, rubric_score.dim),
                    value=None if counted_value is None else float(counted_value),
                    **coordinates,
                )
            )
        for dim in cannot_tell_dims:
            records.append(
                ScoreRecord(
                    metric=METRIC_SCORE,
                    rubric_dim=dim,
                    rubric_scale=run.rubric_scale(dim),
                    rubric_dim_judge_model=dim_judge_model(run, dim),
                    value=None,
                    **{**coordinates, "outcome": JUDGE_CANNOT_TELL_OUTCOME},
                )
            )
        if not result.rubric_scores and not cannot_tell_dims:
            records.append(ScoreRecord(metric=METRIC_SCORE, value=None, **coordinates))

        # One row per goal-state check the result carries, 1.0 passed and 0.0 not, keyed by the check —
        # including the template's checks a candidate-charged deadline stamps unevaluated, which count as
        # failed. No null row for a result that carries none: the rows above already say the cell was
        # attempted, and a check the result does not carry has no verdict to be null about. The value is the verdict
        # `counted_goal_verdicts` says every rate counts, which the bundle reads too: null for a
        # harness-faulted result (the check read the harness, so it stays out of every pass rate while the
        # row still records that it was attempted), and 0.0 for every check on a candidate failure.
        counted = counted_goal_verdicts(result)
        verdicts: list[tuple[GoalStateOutcome, float | None]]
        if counted is None:
            verdicts = [(goal, None) for goal in result.goal_state_outcomes]
        else:
            verdicts = [(goal, 1.0 if passed else 0.0) for goal, passed in counted]
        for goal, value in verdicts:
            records.append(
                ScoreRecord(
                    metric=METRIC_GOAL_STATE,
                    goal_check=goal.expression,
                    value=value,
                    **coordinates,
                )
            )

        # The dual-score axes. They live in their own fields rather than in
        # `rubric_scores`, so every export was structurally missing them — while
        # `list_metrics` publishes the pair as the one whose DIVERGENCE "separates a worse
        # agent from a changed world". An operator doing local analysis on the CSV had the
        # rubric dims and the composite and not that axis at all.
        #
        # EACH IS ITS OWN METRIC, and the reserved id never reaches `rubric_dim`. Two
        # recorded decisions meet here and this is the shape that honours both. The
        # `rubric_dim` coordinate is pinned to template dims (a test says so, and its
        # reason is that `compute_dimension_summary` does not aggregate these two, so a
        # reserved dim on that coordinate would make the two surfaces disagree about which
        # dims exist); and the `__` prefix on these ids exists, as `METRIC_SCORE`'s own
        # note says, precisely so they can live in the metric namespace without colliding
        # with a template dim. So a pooled `score` cell still averages template dims alone,
        # and the axes are addressable directly rather than as two of N levels on a
        # coordinate — which is what asking about their divergence actually needs.
        #
        # NO NULL ROW when an axis was not scored, unlike the dims above. That null row
        # exists to say the cell was ATTEMPTED and the judge scored no dimension; two more
        # nulls would say the same thing twice over, on axes the first row already covers.
        #
        # `dim` is read off the score rather than paired with a constant here: the judge
        # service stamps the reserved id it scored under, so a row can only be keyed by the
        # axis that produced it. It carries `rubric_dim_judge_model` even though
        # `rubric_dim` is None — see that field, where the rule is "no attribution where no
        # SINGLE model scored the row", which a per-axis judge answers.
        for reserved in (result.transcript_score, result.outcome_score):
            if reserved is None:
                continue
            # `RubricScore.dim` is a stored free-text field, and this is the one row whose
            # `metric` comes off stored data rather than off a module constant. Guarded so
            # the closed-set claim above `PROJECTED_METRICS` stays true by construction:
            # a mis-stamped or hand-written row would otherwise export a metric column
            # value `pivot` refuses as unknown and no descriptor describes.
            #
            # LOUD, because the absence it leaves already means something else. An unjudged
            # result gains no rows here either, so a silent skip would make a mis-stamped
            # axis read exactly like an unjudged one — the
            # make-an-absence-ambiguous failure this module refuses everywhere else, and
            # worse than the unguarded state, where the odd metric value was at least
            # visible and `pivot` refused it by name.
            if reserved.dim not in RESERVED_DIM_IDS:
                log.warning(
                    "eval.reporting dropping a dual-score row for result=%s run=%s: dim %r is not a reserved axis "
                    "(%s). The axis was scored and is NOT in this projection — its absence here is a defect in "
                    "what stamped it, not a result that went unjudged.",
                    result.id,
                    run.id,
                    reserved.dim,
                    ", ".join(sorted(RESERVED_DIM_IDS)),
                )
                continue
            # What the axis counts as, by the rule every judged score follows (`counted_score`):
            # null for a harness fault, the scale's floor for a candidate failure.
            counted_axis = counted_score(result, reserved)
            records.append(
                ScoreRecord(
                    metric=reserved.dim,
                    rubric_dim_judge_model=dim_judge_model(run, reserved.dim),
                    value=None if counted_axis is None else float(counted_axis),
                    **coordinates,
                )
            )
        # The one null axis row: an axis the judge answered it could not tell on. Unlike an
        # unscored axis this is a fact about the axis alone, and its count is owed wherever the
        # axis's exclusions are counted.
        for axis in sorted(RESERVED_DIM_IDS & set(result.judge_cannot_tell)):
            records.append(
                ScoreRecord(
                    metric=axis,
                    rubric_dim_judge_model=dim_judge_model(run, axis),
                    value=None,
                    **{**coordinates, "outcome": JUDGE_CANNOT_TELL_OUTCOME},
                )
            )

    # Over the runs that PLACED a row, not over every run supplied: a run whose
    # results were all excluded contributes nothing this table averages, and a
    # disclosure about it would qualify a population the answer does not contain.
    placed_run_ids = {placement.run.id for placement in placed}
    return ScoreProjection(
        records=records,
        exclusions=exclusions,
        completeness_disclosures=degraded_run_disclosures(run for run in runs if run.id in placed_run_ids),
    )


class CaseSetIdentity(EvalBaseModel):
    """One distinct case set inside a group, and which runs executed it.

    A ``template_id`` does not establish case-set identity while templates are
    mutable in place: a template can gain or lose cases between runs, so two runs
    of "the same suite" may have scored different denominators. The identity is
    already computed — it is ``ContextComponents.case_basis``, the digest of
    ``(template_id, sorted test_case_ids)`` stamped on every run at launch — but
    nothing rendered it, so the fact was recorded and unreadable.

    ``n_cases`` is the size of the set this fingerprint stands for, which is the
    number the group's ``shared_test_case_ids`` intersection cannot give you: an
    intersection of 3 is equally consistent with runs of 3 cases and runs of 31.
    """

    #: Full ``case_basis`` digest — the run's own stamped value where present,
    #: re-derived from its stored fields otherwise (a run its host assembled without
    #: the launch), so every run is identified on the same predicate. ``None`` when the run's basis could
    #: not be resolved at all: the entry then stands for that ONE run (``run_ids`` has
    #: one member), because an unresolved basis says the set is unknown and merging
    #: unknowns would assert a sameness nothing established. It is a state, not a
    #: digest, so a renderer branches on ``None`` rather than displaying anything.
    fingerprint: str | None
    n_cases: int
    run_ids: list[str]


# =============================================================================
# Measurement windows — WHEN a run was measured, and whether two runs overlap
# =============================================================================


class MeasurementWindow(EvalBaseModel):
    """The wall-clock span a run's cells were actually measured over — DERIVED.

    **Deliberately not called a "window" bare**, because that word is already
    taken one level up: ``CampaignWindow`` is a campaign's span, derived from its
    member runs' ``created_at``. This is a different quantity on a different
    basis, and the two disagree by hours on the same data, so they carry different
    names and different derivations rather than one name meaning two things.

    **The basis is every result's ``scored_at``, never the run's
    ``created_at``/``completed_at``.** A run document is saved — stamping
    ``created_at`` — *before* its task is created, and the task then waits on a
    semaphore that admits two jobs at a time, so ``created_at`` is when the run
    was enqueued and not when anything was measured. The gap is not academic: a
    ten-arm sweep whose arms were enqueued within half a minute of each other
    executed across four hours, so on a ``created_at`` basis every arm's span
    contains every other arm's and no pair can ever read as disjoint. A window
    derived that way would be silent on precisely the runs it exists to describe.
    ``scored_at`` is stamped when a cell's result object is built, at the end of
    that cell, which makes min/max over a run's results the span its cells really
    occupied.

    Both bounds are the ISO-8601 UTC strings the models stamp, compared as strings
    throughout: every eval timestamp comes from one stamper that always emits a
    ``+00:00`` offset, so lexicographic order is chronological order. Nothing here
    parses a datetime, which is what keeps a malformed stamp from raising on a
    read surface.
    """

    run_id: str
    #: Earliest ``scored_at`` among the run's results.
    start: str
    #: Latest ``scored_at`` among the run's results.
    end: str


def measurement_window(run_id: str, results: Sequence[EvalResult]) -> MeasurementWindow | None:
    """Derive one run's measurement window from the results it produced.

    Args:
        run_id: The run the window describes.
        results: That run's stored results. Callers filter to one run; nothing
            here checks ``eval_run_id``, so passing another run's results
            produces a window describing a population that never existed.

    Returns:
        The window, or ``None`` when the run produced no results. ``None`` is an
        honest absence and never a zero-length span at some arbitrary instant: a
        run that measured nothing did not measure it "at" a time, and a synthetic
        point would compare against other windows as though it had.
    """
    stamps = [r.scored_at for r in results if r.scored_at]
    if not stamps:
        return None
    return MeasurementWindow(run_id=run_id, start=min(stamps), end=max(stamps))


class WindowGap(NamedTuple):
    """One pair of runs whose measurement spans do not overlap, and how far apart they were.

    The pair is oriented — ``earlier`` finished before ``later`` began — so the gap
    is a forward duration rather than a signed difference a reader has to interpret.
    """

    earlier: MeasurementWindow
    later: MeasurementWindow
    #: Seconds between ``earlier.end`` and ``later.start``, or ``None`` when the
    #: recorded stamps cannot be read as instants. See :func:`_gap_seconds`.
    seconds: float | None


def _gap_seconds(earlier_end: str, later_start: str) -> float | None:
    """How long a run's span sat idle before the next one began, if that is knowable.

    **The lexicographic discipline is unchanged and this does not weaken it.** Ordering
    and the disjointness predicate still compare the ISO-8601 strings directly, so a
    malformed stamp cannot make a comparison raise. What is added is a *guarded*
    parse used for magnitude only: a stamp that will not parse, or a pair that mixes an
    offset-aware stamp with a naive one, yields no duration and the disclosure says so
    rather than failing.

    Args:
        earlier_end: The last ``scored_at`` of the run that finished first.
        later_start: The first ``scored_at`` of the run that began after it.

    Returns:
        The non-negative gap in seconds, or ``None`` when it cannot be computed
        honestly — an unparseable stamp, a naive/aware mix, or a parsed order that
        contradicts the lexicographic one (which means the two stamps are not on a
        shared clock and no duration between them is meaningful).
    """
    try:
        end = datetime.fromisoformat(earlier_end)
        start = datetime.fromisoformat(later_start)
    except TypeError, ValueError:
        # NOSILENT: None IS the report -- format_window_gap renders it as the uncomputable-gap clause
        return None
    try:
        delta = (start - end).total_seconds()
    except TypeError:
        # NOSILENT: One stamp carried a UTC offset and the other did not. Subtracting those is
        # not a duration anybody measured, so it is reported as uncomputable (None renders as
        # the uncomputable-gap clause).
        return None
    return delta if delta >= 0 else None


class OverlappingWindows(NamedTuple):
    """One pair of runs whose measurement spans DID overlap.

    Unoriented, unlike :class:`WindowGap`: there is no earlier and later to name when
    two spans share wall-clock time, and no duration between them to report. It carries
    the two windows and nothing else, because the only claim it supports is that these
    two runs were not measured apart.
    """

    first: MeasurementWindow
    second: MeasurementWindow


class WindowPairs(NamedTuple):
    """Every pair of a run set, split by whether the two spans overlapped.

    Both halves come off one enumeration. The disclosure counts the disjoint pairs
    against the total, and the total includes the overlapping half. Complementing one half
    somewhere else would put the overlap predicate in two places, which is the drift this
    type exists to prevent.
    """

    disjoint: list[WindowGap]
    overlapping: list[OverlappingWindows]

    @property
    def total(self) -> int:
        """How many pairs the run set has.

        Returns:
            The P a partial disclosure counts its disjoint pairs against.
        """
        return len(self.disjoint) + len(self.overlapping)


def classify_window_pairs(windows: Sequence[MeasurementWindow]) -> WindowPairs:
    """Split every pair of these runs by whether their measurement spans overlapped.

    The single derivation behind both the predicate and the prose: the badge, the
    quantifier the sentence states and the magnitudes it names are all read off this one
    enumeration, so they cannot come apart about which pairs are disjoint.

    Only windows that RESOLVED are passed in, and callers must keep it that way. A run
    with no results has no window, and letting its absence contribute would report a
    difference on the strength of what one run could not say — the same rule the
    case-set and attribution comparisons follow.

    **Any pair, not all pairs**, and the prose a caller renders from this must state
    that quantifier and not a stronger one. Three arms where two ran together and the
    third ran the next morning are still a set nothing held fixed across, so requiring
    every pair to be disjoint would report nothing for the common sweep shape of a few
    arms at a time — but printing the existential as "these runs were measured over
    non-overlapping spans" is the falsehood the generator escalated into a false
    headline. Fewer than two windows yields both halves
    empty: one run cannot be measured apart from itself, and zero runs assert nothing.

    Bounds are treated as closed and touching counts as overlap: two runs where one's
    last cell and the other's first share an instant were running against the same
    conditions, and the whole point of the question is whether they were.

    Args:
        windows: The resolved windows of the runs being compared.

    Returns:
        The pairs, split. Both halves are in the order the pairs are enumerated from
        ``windows``, and both are empty for fewer than two windows — one run cannot be
        measured apart from itself, and it cannot overlap itself either.
    """
    # Checked exhaustively over every pair. A sorted single pass is the shape that
    # answers "do ALL of them overlap", which is a different question and the one
    # a comparison does not want asked; over the handful of runs a comparison set
    # holds, the pairwise loop costs nothing and says what it means.
    disjoint: list[WindowGap] = []
    overlapping: list[OverlappingWindows] = []
    for i, a in enumerate(windows):
        for b in windows[i + 1 :]:
            if a.end < b.start:
                earlier, later = a, b
            elif b.end < a.start:
                earlier, later = b, a
            else:
                overlapping.append(OverlappingWindows(first=a, second=b))
                continue
            disjoint.append(WindowGap(earlier=earlier, later=later, seconds=_gap_seconds(earlier.end, later.start)))
    return WindowPairs(disjoint=disjoint, overlapping=overlapping)


def disjoint_window_pairs(windows: Sequence[MeasurementWindow]) -> list[WindowGap]:
    """Every pair of these runs whose measurement spans do not overlap, with its gap.

    The disjoint half of :func:`classify_window_pairs`, which holds the predicate and
    the reasoning. Kept as its own name because the disclosure and the badge ask only
    this question, and reading ``.disjoint`` at every call site would say less.

    Args:
        windows: The resolved windows of the runs being compared.

    Returns:
        One :class:`WindowGap` per non-overlapping pair, in the order the pairs are
        enumerated from ``windows``. Empty when every pair overlaps, and for fewer than
        two windows — one run cannot be measured apart from itself.
    """
    return classify_window_pairs(windows).disjoint


# The distinguishing clause of the disjoint-window disclosure, split out for the
# same reason :data:`DEGRADED_RUN_CLAUSE` is: a test pins the CLAUSE, so the
# sentence around it stays free to be rewritten for a reader.
DISJOINT_WINDOWS_CLAUSE = "measured over non-overlapping spans"


#: Above this many windows the disclosure summarises instead of listing every span.
#:
#: The listing form is the better disclosure and stays the default for the sizes an
#: operator actually reads. It stops being a disclosure at scale: a 27-run group
#: renders every span inline as one unbroken paragraph, and the sentence directing the
#: reader to "read it rather than the badge" becomes advice nobody can take. Four keeps
#: the pairwise and small-sweep cases — the ones where every span is the point —
#: verbatim.
MAX_INLINE_MEASUREMENT_WINDOWS = 4

#: How many non-overlapping pairs the disclosure names before it counts the rest.
#:
#: Six is exactly the pair count of :data:`MAX_INLINE_MEASUREMENT_WINDOWS` runs, so the
#: sizes an operator actually reads never truncate. The cap exists for the other end: a
#: ``full=true`` read of a 22-run campaign has 231 pairs, and a paragraph of them is the
#: same non-disclosure the uncapped span list was.
MAX_RENDERED_WINDOW_GAPS = 6

#: What a gap renders as when the recorded stamps cannot be read as instants. Said
#: rather than omitted: a pair silently missing its magnitude reads as a pair with no
#: gap, which is the opposite of what it means.
UNCOMPUTABLE_GAP_CLAUSE = "gap not computable from the recorded stamps"


def format_window_gap(seconds: float | None) -> str:
    """Render a gap between two measurement spans as a magnitude, in its own units.

    Two significant units and no unit words: ``50m31s``, ``1h04m``, ``5d23h``. The
    compact form is deliberate — this is a magnitude the reader weighs, not a sentence,
    and it sits inside a list of pairs where spelled-out units would bury the numbers.

    **It states the size and never what the size means.** Each disclosure computes a magnitude
    in the measure's own units; deciding whether that magnitude is material is the descriptor's
    threshold to declare and is not this function's.

    Args:
        seconds: The gap, or ``None`` when it could not be computed.

    Returns:
        The magnitude, or :data:`UNCOMPUTABLE_GAP_CLAUSE` when there is none.
    """
    if seconds is None:
        return UNCOMPUTABLE_GAP_CLAUSE
    total = round(seconds)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _gap_phrase(gap: WindowGap) -> str:
    """Name one non-overlapping pair and how far apart it was."""
    if gap.seconds is None:
        return f"{gap.earlier.run_id} and {gap.later.run_id}, {UNCOMPUTABLE_GAP_CLAUSE}"
    return f"{gap.earlier.run_id} and {gap.later.run_id}, {format_window_gap(gap.seconds)} apart"


def _widest_gaps_first(pairs: Sequence[WindowGap]) -> list[WindowGap]:
    """Order non-overlapping pairs so the biggest gap is read first.

    Widest first because that is the pair a reader most needs to see, and because
    truncating the tail then drops the least, never the most. Pairs whose gap could not
    be computed sort last — they are an admission rather than a measurement, and putting
    an admission where the widest gap belongs would read as a ranking. Ties break on run
    id so one set of windows always renders one way.
    """
    return sorted(
        pairs,
        key=lambda p: (0 if p.seconds is not None else 1, -(p.seconds or 0.0), p.earlier.run_id, p.later.run_id),
    )


def measurement_window_disclosure(windows: Sequence[MeasurementWindow], *, full: bool = False) -> str | None:
    """The sentence a comparison must carry when its runs did not share a clock.

    Descriptive, never a verdict: it states when each run was measured, which pairs did
    not overlap and by how much, and stops there. No threshold decides how far apart is
    too far, no severity is assigned, and nothing is corrected — how much a gap matters
    depends on what else moved in it, which this surface cannot see and the operator can.

    **It states the quantifier it can defend.** The condition is existential — see
    :func:`disjoint_window_pairs` — so the universal form ("these runs were measured over
    non-overlapping spans") is printed only when every pair really is disjoint. Otherwise
    the sentence counts: *D of the P pairs among these N runs*, with the remainder named
    as overlapping and the closing attribution scoped to the pairs that earned it.
    Rendering the existential as a universal is the defect this exists for — on the
    campaign that motivated the fix, three of six pairs overlapped while the sentence
    denied it, and the generator escalated the denial into a false headline.

    **Each non-overlapping pair carries its gap magnitude**, widest first, because
    a disclosure computes a magnitude in the measure's own
    units. An earlier form deliberately reported bounds and no durations, on the grounds
    that no read surface may parse a datetime; that guarantee is kept — ordering and the
    overlap predicate are still purely lexicographic — and only the magnitude is parsed,
    under a guard that yields :data:`UNCOMPUTABLE_GAP_CLAUSE` rather than raising.

    Above :data:`MAX_INLINE_MEASUREMENT_WINDOWS` the span list is replaced by the group's
    outer bounds and its two extreme windows, and the gap list collapses to the widest
    pair; both say how much they did not name and where to get it.

    Args:
        windows: The resolved windows of the runs being compared.
        full: List every span regardless of count. For surfaces whose consumer
            can collapse the text itself, and for an operator who asked. The pair
            list still caps at :data:`MAX_RENDERED_WINDOW_GAPS`, because pairs grow
            quadratically where spans grow linearly.

    Returns:
        The disclosure, rendered verbatim by every surface, or ``None`` when every
        pair of runs overlaps or there are too few windows to ask.
    """
    pairs = disjoint_window_pairs(windows)
    if not pairs:
        return None
    ordered = sorted(windows, key=lambda w: (w.start, w.end))
    n_runs = len(ordered)
    n_pairs = n_runs * (n_runs - 1) // 2
    inline = full or n_runs <= MAX_INLINE_MEASUREMENT_WINDOWS
    if inline:
        detail = "; ".join(f"{w.run_id} {w.start} to {w.end}" for w in ordered)
    else:
        first, last = ordered[0], ordered[-1]
        # The lower bound is `first.start` by construction — the sort is by start.
        # The upper bound is NOT `last.end` for the same reason: sorting by start
        # says nothing about which member ends last, so a run that began early and
        # overran holds the latest end while sitting first. Taking `last.end` would
        # understate the group's span in exactly that case.
        detail = (
            f"{n_runs} runs, measured between {first.start} "
            f"and {max(w.end for w in ordered)}; earliest {first.run_id} {first.start} to {first.end}; "
            f"latest {last.run_id} {last.start} to {last.end}; "
            f"{n_runs - 2} further span(s) not shown — pass full=true for every span"
        )

    widest_first = _widest_gaps_first(pairs)
    if inline:
        shown = widest_first[:MAX_RENDERED_WINDOW_GAPS]
        gap_detail = "; ".join(_gap_phrase(gap) for gap in shown)
        withheld = len(pairs) - len(shown)
        if withheld:
            gap_detail += f"; +{withheld} further non-overlapping pair(s), all narrower than these"
        gaps = f"Non-overlapping pairs, widest first: {gap_detail}."
    else:
        gaps = f"Widest non-overlapping pair: {_gap_phrase(widest_first[0])}"
        if len(pairs) > 1:
            gaps += f"; {len(pairs) - 1} further non-overlapping pair(s) not named"
        gaps += "."

    if len(pairs) == n_pairs:
        lead = f"These runs were {DISJOINT_WINDOWS_CLAUSE} of wall-clock time — every pair of them ({detail})."
        subject = "a difference between them"
    else:
        lead = (
            f"{len(pairs)} of the {n_pairs} pairs among the {n_runs} runs named here were "
            f"{DISJOINT_WINDOWS_CLAUSE} of wall-clock time; the other {n_pairs - len(pairs)} overlap ({detail})."
        )
        subject = "a difference between the two runs of a non-overlapping pair"
    return (
        f"{lead} {gaps} "
        "Anything that changed on the machine, the providers or the account between those spans "
        f"varies with the runs, so {subject} is not attributable to the runs alone."
    )


# The distinguishing clause of the cassette-span disclosure, split out for the same
# reason :data:`DISJOINT_WINDOWS_CLAUSE` is: a test pins the CLAUSE, so the sentence
# around it stays free to be rewritten for a reader.
CASSETTE_SPAN_CLAUSE = "did not all record the same cassette mode"

#: The one cassette mode that SUBSTITUTES third-party output. ``off`` and ``capture``
#: both run the third party live — capture additionally records what came back — so a
#: span across those two is a difference in recording, not in measurement. Naming the
#: substituting mode once keeps the disclosure's two branches from drifting apart.
SUBSTITUTING_CASSETTE_MODE = "replay"


def cassette_mode_disclosure(modes_by_run: Mapping[str, str]) -> str | None:
    """The sentence a comparison must carry when its runs recorded different cassette modes.

    **One vocabulary, two layers.** ``cassette_mode`` is already a declared apparatus
    confound in the host's sweepable declarations — "tool output was
    live for some of these runs and replayed for others, which changes both latency and
    content" — which is what ``bisect_runs`` and the analysis bundle's confound scan read,
    and the generated analysis already names it correctly. The badge derived from this
    sentence is therefore ``cassette_mode_differs``: the declared dimension name plus the
    suffix its sibling ``tool_config_differs`` already uses, so the reporting layer and the
    analysis layer name one thing once. What follows is the same claim said in the detail a
    comparison surface needs and a confound catalogue does not carry.

    Reports what the runs RECORDED and says so in those words, because nothing on a run
    corroborates the mode it recorded. Three candidate corroborators were checked and each
    is the same claim one level down or weaker than it: :attr:`~threetears.evals.schema.models.EvalRun.cassette_corpus_id` is
    set exactly when the run claims replay, and
    :func:`~threetears.evals.kernel.usage_capture.count_substituted_deliveries` counts seeded case
    findings alongside replayed cassettes, so a non-zero count does not evidence replay
    and a zero one cannot separate "did not replay" from "replayed a template with no
    async delivery". So this asserts only the record and names the ambiguity, the choice
    the run-detail badge already makes.

    **Two spans, one badge.** Only ``replay`` substitutes: an arm that replayed re-served
    a recording rather than measuring the third party, which confounds its quality — a
    replayed arm is served only the asks its capture made (any other ask stops the cell), so
    it is measured on the questions the capturing candidate chose. **Its cost is confounded at
    one of the two seams, not both**, which is why the sentence names them separately: a
    replayed background DELIVERY spent no inner-agent dollars and its
    production-replicating cost is withheld, while a replayed tool ACTION (a search, say) saved
    provider credits that never entered the per-role rows on a live arm either, so its cost
    is reported and is the same figure the live arm would report. Saying "a replayed arm's
    cost is withheld" flatly was wrong for the second seam and told a reader to expect a
    blank where a number correctly appears — see
    :func:`~threetears.evals.kernel.usage_capture.production_replicating_cost`. A span that stays
    within ``off``/``capture`` is a difference in what was *recorded* while both arms ran
    live. Both are disclosed, because both are a difference in a condition that should have
    been holding still; the branch decides only what the sentence says the difference costs.

    Descriptive, never a verdict, on the rule :func:`measurement_window_disclosure`
    follows: no comparison is refused, no number is adjusted and no severity is assigned.
    Capture-versus-replay is a legitimate thing to run.

    Args:
        modes_by_run: One recorded ``cassette_mode`` per run being compared, keyed by run id.

    Returns:
        The disclosure, rendered verbatim by every surface, or ``None`` when fewer than
        two runs were supplied or every run recorded the same mode. Two arms that both
        replayed are uniform and disclose nothing: the substitution is on both sides, so
        the delta between them is honest.
    """
    if len(modes_by_run) < 2:
        return None
    recorded = set(modes_by_run.values())
    if len(recorded) < 2:
        return None
    detail = "; ".join(f"{run_id} recorded {mode}" for run_id, mode in sorted(modes_by_run.items()))
    if SUBSTITUTING_CASSETTE_MODE in recorded:
        cost = (
            "A replayed arm did not measure the third party — it re-served a recording. Where it "
            "replayed a background DELIVERY its production-replicating cost is withheld as unknown "
            "rather than reported smaller; where it only replayed a tool ACTION the cost is "
            "still reported, and is not understated, because that tool's spend never reaches the "
            "per-role rows on a live arm either — what replay saved there is search credits, not "
            "dollars. Its QUALITY is confounded either way: a replayed arm is served only the asks its "
            "capture made, so it is measured on the questions the capturing candidate chose to ask."
        )
    else:
        cost = (
            "Both 'off' and 'capture' run the third party live — capture additionally records what "
            "came back — so this is a difference in what was recorded rather than in what was measured."
        )
    return f"These runs {CASSETTE_SPAN_CLAUSE} ({detail}). {cost}"


#: The one origin value that means *this launch said so*. The other tiers — a value the
#: subject carried, or one inherited from a system default — supplied it from below the launch,
#: so a difference across them is one nobody asked this comparison for.
#:
#: The vocabulary of origins is a host's; what the engine owns is the RULE beneath it, and the
#: surface that renders a particular lever's sentence lives with that lever's vocabulary, in the host.
DECLARED_INPUT_ORIGIN = "chosen"


def difference_was_declared_at_launch(origins: Iterable[str | None]) -> bool:
    """Whether every one of these runs got its value from its own launch declaration.

    **The generic half of a lever-origin disclosure, and the part that stayed.** The rule is
    not about any one lever: a difference between arms is a *swept axis* when each arm's launch
    declared the value, and a *confound* when any arm inherited it from a tier the comparison
    never named. Only the vocabulary (``chosen`` / a subject-carried tier / an inherited
    default) is a host's; the rule is the same one a campaign's declared design will state
    formally once it declares one, at which point this reads the declaration instead of
    the origins. A host calls it to write, for example, the sentence about which model its
    arms delegated background work to.

    Conservative by construction: one arm that inherited its value is enough to make the
    whole difference undeclared, because the comparison cannot claim to be sweeping an
    axis one of its arms was never pointed at.

    Args:
        origins: One origin per compared run, as the run recorded it. ``None`` means the
            run recorded no origin and therefore declared nothing that can be read.

    Returns:
        ``True`` when every origin is :data:`DECLARED_INPUT_ORIGIN`. ``False`` for an
        empty set — nothing declared anything — so a caller can never read silence as a
        declaration.
    """
    listed = list(origins)
    return bool(listed) and all(origin == DECLARED_INPUT_ORIGIN for origin in listed)


class ComparisonSet(EvalBaseModel):
    """A group of runs that may be compared with each other, and on what basis.

    ``badges`` names every respect in which the group is *less* than fully
    comparable. An empty list means the group shares template, subject, context
    key, and test-case set — the only case where run-vs-run subtraction needs no
    caveat.

    ``shared_test_case_ids`` is the INTERSECTION across the group, and on its own
    it misleads in exactly the direction that matters: a sweep whose runs
    report 2 shared cases can have runs that each executed 6, and the intersection
    was small *because* the sets differed. ``case_sets`` names each distinct set
    and who ran it, so the reader sees how the group is split rather than only
    what survives the overlap.
    """

    subject_id: str
    subject_label: str = ""
    template_id: str | None = None
    run_ids: list[str]
    shared_test_case_ids: list[str]
    #: Distinct case sets within the group, most-used first. More than one entry is
    #: the ``case_set_differs`` badge's substance — the badge says *that* they
    #: differ, these say *how*.
    case_sets: list[CaseSetIdentity] = []
    badges: list[str]
    #: The ``measurement_windows_disjoint`` badge's substance: the sentence naming
    #: when each run was measured. The badge says the group did not share a clock;
    #: only this says whether that means four minutes or four hours, which is the
    #: whole of the judgement being handed to the reader. ``None`` whenever the
    #: badge is absent — the two are derived from one predicate and cannot disagree.
    measurement_window_disclosure: str | None = None
    #: The ``cassette_mode_differs`` badge's substance: the sentence naming what each run
    #: recorded and what the span costs. The badge says the group's arms did not all
    #: record one mode; only this says whether that means a recording difference between
    #: two live arms or a replayed arm that never measured the third party at all.
    #: ``None`` whenever the badge is absent — one predicate, so the two cannot disagree.
    cassette_mode_disclosure: str | None = None
    #: The runs that share this group's (subject, template) and that the caller's scope
    #: left out. Empty when unscoped, which is every group's ordinary state.
    #:
    #: A scope makes the badges honest about the set under analysis, and in doing so it
    #: hides how much else the group holds — so a reader meeting a badge-clean scoped
    #: group cannot tell whether the group really is three replicates or three of seven runs
    #: spanning several days. Naming the excluded runs is what keeps the narrowing visible:
    #: the argument is that a set quietly reduced reads identically to a set that was always
    #: that size.
    out_of_scope_run_ids: list[str] = []


# Badge vocabulary. ``context_differs`` extends the existing ``comparison_basis``
# vocabulary used by run comparison; ``case_set_differs`` records the drift that
# makes a longitudinal series dishonest when a suite silently gained or lost
# cases between runs.
BADGE_CONTEXT_DIFFERS = "context_differs"
BADGE_CASE_SET_DIFFERS = "case_set_differs"
# The pinned roles: models held fixed so the candidate is the only thing varying.
# When they differ between runs, a quality delta is confounded by a judge or
# simulator change and the comparison measures two things at once.
BADGE_ROLES_DIFFER = "roles_differ"
# At least one run's measurement context could not be fully reconstructed — a role pin
# it never recorded. Distinct from every other badge here, which reports a difference
# that WAS observed: this one reports that the comparison cannot be decided. It has to
# exist separately because an unrecorded pin makes the other badges come out CLEAN —
# two runs that both stored a blank compare equal on the blank — so the group would
# otherwise render as the one case needing no caveat at all.
BADGE_CONTEXT_INCOMPLETE = "context_incomplete"
# At least one run's case basis could not be resolved. The sibling of
# ``context_incomplete``, one axis over, and separate from ``case_set_differs`` for
# the reason that badge cannot carry: an unresolved basis means the sets are
# UNKNOWN, not known-different. Asserting a difference we did not observe is the
# same dishonesty as asserting a match. See ``CaseSetIdentity.fingerprint`` for
# why such runs never merge into one entry.
BADGE_CASE_SET_UNRESOLVED = "case_set_unresolved"
# The candidate's own resolved tool config (search depth, token budgets, call caps,
# the inner-agent model). Unlike the three above it is not a caveat on the
# comparison — sweeping it is usually the POINT — but it has to be said out loud,
# because it is invisible in the one place an operator looks for it: tool config
# composes the VARIANT key, so runs that differ across it still share a context key
# and, badged only on the conditions, render as one configuration — runs differing
# in three tool caps at once can read as one.
BADGE_TOOL_CONFIG_DIFFERS = "tool_config_differs"
# At least one pair of the group's runs was measured over spans that do not
# overlap, so anything that moved on the box, the providers or the account between
# those spans varies with the runs. It is the one badge here whose evidence is not
# in the run documents at all — it is derived from the results' ``scored_at``, so a
# group whose results were not supplied cannot be asked and is silent rather than
# clean. Descriptive only: how much a gap costs depends on what else changed in it,
# which this surface cannot see and the operator can.
BADGE_MEASUREMENT_WINDOWS_DISJOINT = "measurement_windows_disjoint"
# The group's runs did not all record the same cassette mode. It sits with
# ``roles_differ`` and ``tool_config_differs`` as a per-component badge over a condition
# the context key already hashes but cannot name: ``cassette`` is a context component
# (``identity.py``), so a capture-versus-replay pair fires ``context_differs`` — "something
# about the conditions moved" — and the operator is left to bisect which. That is the exact
# opacity these badges exist to remove, and it went unnamed on the one condition where the
# arms may not have measured the same thing at all. Like the disjoint-window badge it is
# gated on its DISCLOSURE rather than on a predicate of its own, so the flag can never
# appear with no sentence naming the modes behind it: the flag cannot tell an
# ``off``/``capture`` recording difference from a live-versus-replayed substitution, and
# that distinction is the whole of what a reader needs.
BADGE_CASSETTE_MODE_DIFFERS = "cassette_mode_differs"


class ComparisonSetsResult(EvalBaseModel):
    """Comparability groups plus the runs the caller's own scope left out.

    There is no blank-subject exclusion count, and its absence is the point: every run carries a
    non-blank subject key, so "a scope whose runs cannot be grouped" is not a state. A counter
    that could only ever read zero is worse than none — a reader takes zero as evidence.

    ``out_of_scope_run_ids`` names every supplied run the scope left out, so a reader meeting a
    small clean group can tell whether it is the whole story. Empty when unscoped.
    """

    comparison_sets: list[ComparisonSet] = []
    out_of_scope_run_ids: list[str] = []


def compute_comparison_sets(
    runs: list[EvalRun],
    *,
    results: Sequence[EvalResult] = (),
    full_windows: bool = False,
    scope_run_ids: Iterable[str] | None = None,
    profile: HostProfile,
) -> ComparisonSetsResult:
    """Group runs into sets that may honestly be compared, badging every caveat.

    Runs are grouped by (subject, template): those are the two coordinates whose
    difference makes a comparison meaningless rather than merely noisy. Within a
    group, differences that weaken — but do not invalidate — the comparison are
    reported as badges, never by silently dropping runs.

    **A badge speaks for the runs it was computed over, and a reader is usually
    asking about a smaller set than the group holds.** A campaign can hold
    three runs on one template while the group holds seven, spanning several days,
    and badge ``cassette_mode_differs`` because some non-members recorded with a
    cassette while every member recorded ``off`` — a caveat earned by runs
    the analysed set does not contain. ``scope_run_ids`` is how a caller
    asks about the set actually under analysis: the grouping key is unchanged and
    the badge logic is unchanged, and only the population they run over narrows.

    Two badges are not "a difference was observed". ``tool_config_differs``
    reports one that usually IS the point of the runs — it is here because it
    lives in the *variant* key rather than the context key, so a swept set is
    otherwise badge-silent and reads as one configuration. ``context_incomplete``
    reports the opposite: a comparison that cannot be decided, because a run
    never recorded a role pin. That one cannot be inferred from the others —
    an unrecorded pin makes every value-equality here come out clean.

    ``measurement_windows_disjoint`` is the one badge whose evidence is not in the
    run documents. When two runs occupied non-overlapping spans of wall-clock
    time, whatever moved on the box or at the providers between them varies with
    the runs — a real confound that every field on an :class:`EvalRun` compares
    equal on. The spans come from the results' ``scored_at``, which is why
    ``results`` exists as a parameter at all.

    Runs with no captured subject identity are excluded, for the reason given in
    :func:`project_score_records`.

    Args:
        runs: Runs to group. Order is irrelevant; output is sorted for stability.
        results: The results those runs produced, in any order and from any run —
            they are indexed by ``eval_run_id`` here. **A caller that omits them
            asks a narrower question**: the measurement-window badge cannot be
            decided without them, and an undecidable badge is silent, so a group
            reads as sharing a clock when nothing checked. Every production caller
            supplies the scope's results; the default exists for callers whose
            question really is only about the run documents.
        full_windows: List every measurement span rather than summarising above
            :data:`MAX_INLINE_MEASUREMENT_WINDOWS`. Off by default because this
            surface groups a whole scope — the group that most needs the
            disclosure is the one with the most runs, and so the one the listing
            form renders unreadable.
        scope_run_ids: Narrow every group to these runs before badging — the run
            ids of the set actually under analysis, typically a campaign's members
            resolved by the caller. ``None`` groups and badges every supplied run,
            which is the unscoped behaviour and is unchanged. **Run ids rather than
            a campaign id deliberately**: this module knows nothing about
            campaigns and must not learn — new eval machinery lands host-agnostic,
            and resolving a grouping concept
            to its members is the caller's one lookup. A scope naming runs that are
            not here is logged rather than refused: unlike an unknown run *status*,
            which names nothing that can exist, a member absent from the scope
            is a real and reachable state.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`ComparisonSetsResult` — one :class:`ComparisonSet` per
        (subject, template) group, sorted by subject then template, plus the
        count of runs excluded for having no captured subject identity, plus
        ``out_of_scope_run_ids`` naming what a scope left out — at the result
        level every supplied run the scope excluded, and per group the comparable
        runs it excluded from THAT group. Both are empty when unscoped. Under a
        scope, a group whose every member is out of scope is not emitted at all,
        because a group with no in-scope run has nothing to say about the set the
        caller asked about — its members are still named at the result level, so
        the narrowing stays visible rather than becoming a group that vanished.
    """
    results_by_run: dict[str, list[EvalResult]] = {}
    for result in results:
        results_by_run.setdefault(result.eval_run_id, []).append(result)

    # Narrowed BEFORE grouping, so every derived quantity below — the shared case
    # intersection, the case-set fingerprints, every badge and both disclosures —
    # is computed over one population. Filtering afterwards would leave a badge
    # earned by a run the returned group no longer lists, which is the defect one
    # level down from the one this parameter exists to fix.
    scoped: frozenset[str] | None = frozenset(scope_run_ids) if scope_run_ids is not None else None
    # What the scope left out, captured BEFORE the narrowing and keyed the same way the
    # groups are, so a scoped group can say what a reader would otherwise have to re-read
    # the scope to discover. Grouped here rather than derived by the caller because
    # only this function knows the grouping rule, and a second derivation of it is how a
    # group and its own exclusion list come to disagree about which runs are comparable.
    out_of_scope_run_ids: list[str] = []
    out_of_scope_by_key: dict[tuple[str, str | None], list[str]] = {}
    if scoped is not None:
        present = {run.id for run in runs}
        out_of_scope = [run for run in runs if run.id not in scoped]
        out_of_scope_run_ids = sorted(run.id for run in out_of_scope)
        for run in out_of_scope:
            out_of_scope_by_key.setdefault((_subject_id_of(run), run.template_id), []).append(run.id)
        if missing := sorted(scoped - present):
            # Logged rather than raised, and never silent: a scope that resolves to
            # nothing returns no groups, which reads exactly like "these runs are
            # not comparable with anything".
            log.info(
                "comparison_sets scope named %d run(s) not among the supplied runs: %s",
                len(missing),
                ", ".join(missing),
            )
        runs = [run for run in runs if run.id in scoped]

    # No blank-subject exclusion, and nothing to count: every run carries a non-blank subject key,
    # because ``SubjectSnapshot`` refuses a blank one. A counter that could only ever read zero is
    # a disclosure that tells a reader the opposite of the truth.
    groups: dict[tuple[str, str | None], list[EvalRun]] = {}
    for run in runs:
        groups.setdefault((_subject_id_of(run), run.template_id), []).append(run)

    sets: list[ComparisonSet] = []
    for (subject_id, template_id), grouped in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        member_case_sets = [set(run.test_case_ids) for run in grouped]
        shared = set.intersection(*member_case_sets) if member_case_sets else set()

        # Through the resolver, not off the raw field. `resolve_context_identity` is
        # the single place the stamped-or-derived decision is made, and reading
        # `run.context_key` directly opted this surface out of it: a run stamped under
        # an older predicate kept comparing on a key that predicate no longer produces,
        # and a run whose roles were never recorded reported a key as if it were whole.
        identities = {run.id: resolve_context_identity(run, profile) for run in grouped}

        # The group's distinct case sets, keyed by the run's own `case_basis` digest.
        # Computed BEFORE the badges because the case-set badge is derived from it:
        # one notion of "the sets differ", so the badge and the fingerprints cannot
        # come apart. Deciding the badge separately — over `set(test_case_ids)` — was
        # a second derivation that really did disagree, since a set comparison cannot
        # see a duplicate id that the sorted-list digest does.
        by_fingerprint: dict[str, list[EvalRun]] = {}
        unresolved: list[EvalRun] = []
        for run in grouped:
            basis = identities[run.id].context_components.case_basis
            # An unresolved basis is held apart, one entry PER RUN. Keying every such
            # run on a shared ``""`` merged them into one bucket, which broke two things
            # at once: the case-set badge could not fire between them (they compared as
            # a single set), and ``n_cases`` below — read from the first member as the
            # set's representative, sound only while a bucket really IS one set —
            # reported that one member's denominator as the whole bucket's. Keeping them
            # out of the digest-keyed buckets makes the merge structurally impossible
            # rather than merely unlikely, which matters because the state is latent
            # today: nothing stamps an identity without ``case_basis`` at the current
            # IDENTITY_VERSION, so the reachable version would arrive unnoticed on the
            # next bump.
            if basis:
                by_fingerprint.setdefault(basis, []).append(run)
            else:
                unresolved.append(run)

        badges: list[str] = []
        # Only RESOLVED fingerprints can evidence a difference. Counting the
        # per-run unresolved entries here would report "the sets differ" on the
        # strength of not knowing what they were.
        if len(by_fingerprint) > 1:
            badges.append(BADGE_CASE_SET_DIFFERS)
        if unresolved:
            badges.append(BADGE_CASE_SET_UNRESOLVED)
        if len({identities[run.id].context_key for run in grouped}) > 1:
            badges.append(BADGE_CONTEXT_DIFFERS)
        # Equality on the pins is only meaningful once both are recorded. Two runs that
        # each inherited a judge before the pin was resolved both carry None, and every
        # comparison here — the tuple below, the key above — comes out EQUAL, so the group
        # ships with no badge at all, which this surface defines as needing no caveat.
        # The absence has to be its own badge; nothing about the values can express it.
        if any(identities[run.id].partial for run in grouped):
            badges.append(BADGE_CONTEXT_INCOMPLETE)
        # Per-dim attribution joins the pins here for the same reason it joined the roles
        # component: two runs can agree on every pin and still have had their rubric scored
        # by different models, which is a difference in the apparatus wearing the appearance
        # of a replicate. Hashed rather than compared as a mapping so the tuple stays
        # hashable, and sorted so dim resolution order cannot fake a difference.
        #
        # RECORDED attribution only, via the same accessor ``derive_context_identity``
        # gates on, and compared SEPARATELY from the pins rather than folded into one
        # tuple per run. Both halves matter. Reading the map whenever it was merely
        # present let a DERIVED reconstruction speak with the authority of a record —
        # asserting or suppressing a comparability claim that ``identity.py`` refuses to
        # hash in the same breath. Folding it into the per-run tuple then reproduced the
        # absence-as-difference bug one level down: a run with no recorded attribution
        # contributes a different tuple element from one that has it, so merely MIXING
        # them fired the badge on the strength of what one run could not say. This is the
        # shape the case-basis arm above already uses — only resolved values evidence a
        # difference, and the undecidable case is BADGE_CONTEXT_INCOMPLETE's to carry.
        # The pinned CONFIG set is the third arm, on the argument above carried one step
        # further: two runs can agree on every pin AND on every effective judge model and
        # still have been scored by different judge PROMPTS, which is what a judge A/B is
        # made of. Without this arm such a pair fires only ``context_differs`` — "something
        # about the conditions moved" — which is precisely the opaque key mismatch the
        # per-component badges exist to save the operator from bisecting by hand.
        #
        # THE PINS COME FROM THE REGISTRY, through the declared readers, so this badge and the two
        # surfaces that print the delta cannot disagree about one pair of runs. It was a hand-built
        # ``{(run.judge_model, run.simulator_model)}``, which was wrong twice over: it spoke only
        # for the ENGINE's two roles, so a host that grades with code and declares its own role got
        # no arm at all however registry-driven everything upstream had become; and a tuple of raw
        # fields reproduced the absence-as-difference bug the attribution arm below had already
        # been fixed for — one arm recording a pin and another recording nothing is not an observed
        # difference, and ``BADGE_CONTEXT_INCOMPLETE`` is what carries it. ``comparability`` applies
        # that rule per input rather than per tuple.
        #
        # ``omits_apparatus`` takes every arm's VALUE rather than the declaration alone: a host that
        # left a seat unfilled and whose runs recorded one is contradicting itself, and the
        # runs win.
        #
        # **The two digest arms below stay, and are not redundant with it.** Both read a RUN-LEVEL
        # declaration that no ``Sweepable`` reads: ``judge_config_ids`` is declared here at launch,
        # while the core declaration of the same name reads the set the results were actually scored
        # WITH — the reader's own docstring says those are different questions. A judge A/B moves
        # the launch pin, and a run whose results were never scored would stop badging if this arm
        # were folded into the observed one. ``effective_judges`` is the same shape: the declaration
        # isolates DIVERGENCE from the pin, which is deliberately narrower than the whole map.
        sweepables = profile.sweepables
        pins_by_run = {run.id: sweepables.read_role_pins(run, results_by_run.get(run.id, ())) for run in grouped}
        pin_values = {name: [pins_by_run[run.id].get(name) for run in grouped] for name in sweepables.role_pins}
        pins_differ = any(
            sweepables.comparability(
                name,
                [profile.apparatus_level(run, name, value) for run, value in zip(grouped, values, strict=True)],
            )
            == "differs"
            for name, values in pin_values.items()
            if not profile.omits_apparatus(name, zip(grouped, values, strict=True))
        )
        recorded_attributions = {
            canonical_digest(dict(sorted(judges.items())))
            for run in grouped
            if (judges := run.hashable_effective_judges)
        }
        # ``is not None``, never truthiness: an empty set is the recording "no scored dim
        # carried a config", and two runs that both recorded it agree. Reading it as absent
        # would drop them out of the comparison and let a genuinely-differing third run in
        # the group pass unbadged.
        recorded_config_sets = {
            canonical_digest(dict(sorted(run.judge_config_ids.items())))
            for run in grouped
            if run.judge_config_ids is not None
        }
        if pins_differ or len(recorded_attributions) > 1 or len(recorded_config_sets) > 1:
            badges.append(BADGE_ROLES_DIFFER)
        # Digest the RESOLVED configs, not the overrides: an override restating the
        # subject's own value changes nothing the candidate faced, and comparing
        # overrides would badge that as a difference. This mirrors what the variant
        # key hashes, so the badge and the key can never disagree about what moved.
        if (
            len(
                {
                    canonical_digest(run.subject_snapshot.component_hashes().get("resolved_tool_configs"))
                    for run in grouped
                }
            )
            > 1
        ):
            badges.append(BADGE_TOOL_CONFIG_DIFFERS)
        # Only RESOLVED windows are collected, on the rule the case-basis and
        # attribution arms already follow: a run that cannot say when it was
        # measured is not thereby evidence that it was measured somewhere else.
        # The badge is gated on the DISCLOSURE rather than on the predicate
        # directly, so a group can never carry the flag with no sentence naming
        # the spans behind it — the flag alone cannot tell four minutes from four
        # hours, and that judgement is the reader's.
        windows = [
            w for run in grouped if (w := measurement_window(run.id, results_by_run.get(run.id, ()))) is not None
        ]
        window_disclosure = measurement_window_disclosure(windows, full=full_windows)
        if window_disclosure:
            badges.append(BADGE_MEASUREMENT_WINDOWS_DISJOINT)
        # Gated on the disclosure, never on a predicate of its own — the discipline the
        # window arm above follows, and for a sharper reason here: the flag alone cannot
        # tell an `off`/`capture` recording difference from a live-versus-replayed
        # substitution, and only the second makes the group's QUALITY numbers a mixture.
        # A badge with no sentence behind it would be the less useful half of the pair.
        cassette_disclosure = cassette_mode_disclosure({run.id: run.cassette_mode for run in grouped})
        if cassette_disclosure:
            badges.append(BADGE_CASSETTE_MODE_DIFFERS)

        case_sets = [
            CaseSetIdentity(
                fingerprint=fingerprint,
                n_cases=len(set(members[0].test_case_ids)),
                run_ids=sorted(run.id for run in members),
            )
            # Most-used set first, so the group's dominant denominator leads and the
            # odd run out is visibly the exception rather than one row among equals.
            for fingerprint, members in sorted(by_fingerprint.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ]
        # The unresolved runs last, one entry each: a set of one is the least-used there
        # is, and nothing established that any two of them ran the same cases.
        case_sets.extend(
            CaseSetIdentity(fingerprint=None, n_cases=len(set(run.test_case_ids)), run_ids=[run.id])
            for run in sorted(unresolved, key=lambda run: run.id)
        )

        sets.append(
            ComparisonSet(
                subject_id=subject_id,
                subject_label=grouped[0].subject_snapshot.subject_label,
                template_id=template_id,
                run_ids=sorted(run.id for run in grouped),
                shared_test_case_ids=sorted(shared),
                case_sets=case_sets,
                badges=badges,
                measurement_window_disclosure=window_disclosure,
                cassette_mode_disclosure=cassette_disclosure,
                out_of_scope_run_ids=sorted(out_of_scope_by_key.get((subject_id, template_id), [])),
            )
        )
    return ComparisonSetsResult(
        comparison_sets=sets,
        out_of_scope_run_ids=out_of_scope_run_ids,
    )


# =============================================================================
# Pivot — any two factors as axes, with the disclosures that make it honest.
# =============================================================================

# Weighting modes for rolling a cell's observations into one number.
#
# The default is equal-per-scenario: case counts in this corpus reflect budget
# history rather than importance, and k is unbalanced (k=1 vs k=3) across the
# same suite, so sample-weighting lets an artifact of how much we could afford
# to run drive the ranking. Sample-weighted stays available as an explicit,
# labeled toggle — it is the right answer when the question really is "what did
# the whole population do", and the cost of the default (a thin scenario gets an
# equal vote) is exactly what the per-cell `n` is displayed to expose.
#: Weighting mode: every scenario (case) gets an equal vote in a cell's number, however many observations it has.
WEIGHTING_EQUAL_PER_SCENARIO = "equal_per_scenario"
#: Weighting mode: every observation gets an equal vote, so a case with more repeats counts for more.
WEIGHTING_SAMPLE_WEIGHTED = "sample_weighted"
WEIGHTINGS = (WEIGHTING_EQUAL_PER_SCENARIO, WEIGHTING_SAMPLE_WEIGHTED)
#: The weighting a pivot uses when none is named: equal per scenario.
DEFAULT_WEIGHTING = WEIGHTING_EQUAL_PER_SCENARIO

# Observation-level metric name -> the registry name of the AGGREGATE that an
# aggregating surface (`pivot` cell, `history` series point) actually reports.
#
# It has a second job: inverted, it is the operator-facing ALIAS table
# (`_OBSERVATION_OF_AGGREGATE`), so every name `list_metrics` publishes for one of
# these quantities is a name the surfaces accept. Both jobs need the same pairs,
# which is why they are one table and not two that can drift.
#
# A row metric is never its own cell's descriptor: a row carries one observation,
# while `mean_composite` / `mean_cost_usd` / `mean_score` describe the mean OVER
# rows, which is what a cell holds. That is true whether or not the row metric is
# itself a registry key — `score` is one (the registry describes the raw 1-5 judge
# score) and `composite` / `cost_usd` are not, and all three map here regardless,
# because what a cell needs described is the aggregate.
#
# The mapping has to be explicit because `describe_measure` is total — an unmapped
# name resolves silently to whatever the registry happens to hold for it, which for
# an unseeded name is the unclassified arm (family=None, held at the strictest
# transferability class) and for `score` would be the per-observation 1-5 descriptor
# on a cell holding a mean. Adding a row metric means adding its aggregate here, or
# accepting the fallthrough knowingly.
_AGGREGATE_OF_OBSERVATION = {
    METRIC_COMPOSITE: "mean_composite",
    METRIC_COST_USD: "mean_cost_usd",
    # The registry declares `mean_score` `judge_mediated` on a (1.0, 5.0) range.
    # The cross-subject pooling refusal would fire without this entry — the unclassified arm holds
    # at `scenario_bound`, which is stricter still — but it would fire naming a
    # class nothing vetted, and every honest cell would report `family=None` with
    # no scale beside a number whose whole point is that it is 1-5 and not 0-1.
    METRIC_SCORE: "mean_score",
    # `history` only — `pivot` refuses `total_ms` (it is not projected). Without
    # this entry a latency series described itself with the `total_ms` descriptor,
    # which describes ONE result's wall-clock and its llm/tool/orchestration
    # partition — a partition a mean over cases does not have.
    METRIC_TOTAL_MS: "mean_total_ms",
    # The dual-score axes. Their per-observation descriptors are the registry entries
    # under these same names (1-5, family `dual_axis`); a cell holds the MEAN of those,
    # which is a different quantity and needs its own entry — the fallthrough would
    # otherwise describe a mean with the descriptor of one observation, the exact defect
    # the `score` entry above exists to avoid. Not mapped to `mean_score`, though the
    # arithmetic is identical: this table is inverted into the operator-facing alias map,
    # so three metrics sharing one aggregate name would make the inverse answer with
    # whichever came last.
    METRIC_TRANSCRIPT: "mean_transcript_score",
    METRIC_OUTCOME: "mean_outcome_score",
    # A cell of 1/0 check rows means a pass rate — of ONE check when the cell is keyed on goal_check.
    METRIC_GOAL_STATE: "goal_state_pass_rate",
}

# The inverse: registry aggregate name -> the observation-level name the surfaces
# key rows on. This is what closes the seam an operator hits when they read
# `list_metrics` and use what it says.
#
# `list_metrics` publishes the AGGREGATE names (`mean_composite`, `mean_cost_usd`,
# `mean_total_ms`, `mean_score`) because those describe what an aggregating surface
# returns. The `metric` parameter of `pivot` / `history` names the ROW being
# aggregated, so it took the short observation names — and `mean_composite`, the one
# name the catalog gives that quantity, was refused as unknown. Accepting both is the
# fix, in this direction and not the other:
#
#   - Renaming the row constants to the registry keys is forbidden at their definition
#     site (see the comment above `METRIC_COMPOSITE`) and would be wrong anyway: an
#     export row IS one observation, so `metric=composite` on a row is its honest name.
#   - Seeding a bare `composite` measure into the registry would publish a second
#     composite-quality entry describing a row nobody aggregates, and would leave
#     `mean_cost_usd` / `mean_total_ms` still refused.
#
# So the catalog name resolves to the row name at the surface boundary, and the row
# name keeps working. Derived rather than written out, so a new pair cannot be added
# to one direction only; `_metric_vocabulary` renders both names in every refusal, so
# an operator who guesses wrong is told the other one exists.
_OBSERVATION_OF_AGGREGATE = {aggregate: observation for observation, aggregate in _AGGREGATE_OF_OBSERVATION.items()}


def resolve_measure_name(metric: str) -> str:
    """Resolve a catalog (aggregate) measure name to the observation name rows carry.

    Args:
        metric: A measure name as an operator supplied it — either the
            observation-level row name (``composite``) or the registry name of its
            aggregate as ``list_metrics`` publishes it (``mean_composite``).

    Returns:
        The observation-level name, unchanged for anything this table does not
        cover — an unknown name must reach its surface's own refusal so the caller
        is told what IS accepted, rather than being rewritten into something else.
    """
    return _OBSERVATION_OF_AGGREGATE.get(metric, metric)


def _metric_vocabulary(accepted: frozenset[str]) -> str:
    """Render an accepted-measure set naming each row name beside its catalog name.

    Args:
        accepted: The observation-level names a surface accepts.

    Returns:
        A sorted, comma-joined list in which every measure that has a published
        aggregate name shows it too — a refusal that named only the row names is
        what sent an operator reading ``list_metrics`` in a circle.
    """
    return ", ".join(
        f"{name} (or {_AGGREGATE_OF_OBSERVATION[name]})" if name in _AGGREGATE_OF_OBSERVATION else name
        for name in sorted(accepted)
    )


# What each surface-accepted metric is, in a clause, for every help text that lists them. Keyed on
# exactly the metrics some surface accepts, and checked at import, so a metric added to one of the
# accepted sets without a gloss fails loudly instead of reaching a help text that never mentions it.
_METRIC_GLOSS: dict[str, str] = {
    METRIC_COMPOSITE: "0-1 quality",
    METRIC_COST_USD: "measuring spend in USD: the candidate's and the judge's and simulator's",
    METRIC_SCORE: "the raw judge score: 1-5, or 1/0 on a pass/fail dimension",
    METRIC_TOTAL_MS: "one result's wall-clock",
    METRIC_TRANSCRIPT: "dual-score axis, 1-5 and NOT a dimension",
    METRIC_OUTCOME: (
        "dual-score axis, 1-5 and NOT a dimension; its divergence from the transcript axis is what separates a "
        "worse agent from a changed world"
    ),
    METRIC_GOAL_STATE: "1 passed / 0 not, code-graded",
}


def metric_help(accepted: frozenset[str]) -> str:
    """Render the metrics a surface accepts, each glossed and beside its catalog name, for its help text.

    The one producer of that list for the pivot and history help on REST and MCP alike, so a help
    text cannot name a metric its surface refuses or leave out one it accepts.

    Args:
        accepted: The observation-level names the surface accepts.

    Returns:
        A semicolon-joined list, in name order.
    """
    return "; ".join(
        f"'{name}' ({_METRIC_GLOSS[name]}"
        + (f"; or its catalog name '{_AGGREGATE_OF_OBSERVATION[name]}'" if name in _AGGREGATE_OF_OBSERVATION else "")
        + ")"
        for name in sorted(accepted)
    )


def _describe_aggregate(metric: str, *, profile: HostProfile) -> MetricDescriptor:
    """Describe what a pivot CELL holds, which is an aggregate over rows.

    Args:
        metric: The observation-level metric name carried on the rows.
        profile: The host whose vocabulary this reads.

    Returns:
        The aggregate's descriptor when one is registered, else the honest
        unclassified answer for the observation name itself.
    """
    return describe_measure(_AGGREGATE_OF_OBSERVATION.get(metric, metric), profile.measures)


def _effective_formula(metric: str, weighting: str, *, scoped: bool = True) -> str:
    """State exactly what the reported number is, under this weighting and scope.

    Two things can make the registry's static formula describe a different number
    than the cell shows, and both are stated here rather than left to the reader.
    The weighting is the first. The second is *scope*: the registry describes
    ``mean_score`` as the mean "for one rubric dimension", which is what a cell
    holds only while ``rubric_dim`` is pinned. With it pinned by neither axis nor
    filter, a cell averages every dimension the judge scored — a real question,
    and roughly the composite on the 1-5 scale, but not the one the descriptor
    names. Saying so is the same obligation the weighting clause discharges: an
    unqualified formula beside a qualified number is how a reader checks the
    wrong arithmetic and concludes the table is right.

    Args:
        metric: The observation-level metric being aggregated.
        weighting: The weighting mode the value was computed under.
        scoped: Whether the metric's scope coordinate (:data:`SCOPED_METRICS`) is pinned by an axis or
            filter. Only meaningful for a scoped metric; every other measure is whole-result and has
            nothing to pool. Both pinnings count here because both scope the cell, but only the axis is
            reachable from the REST and MCP surfaces — see the emitted caveat below.

    Returns:
        A one-line formula an operator can read at point of use.
    """
    if weighting == WEIGHTING_EQUAL_PER_SCENARIO:
        formula = f"mean over test cases of the per-case mean of {metric} (each case weighted equally)"
    else:
        formula = f"mean of {metric} over every contributing observation (cases weighted by their observation count)"
    if metric in SCOPED_METRICS and not scoped:
        scope_field, scope_noun, row_noun = SCOPED_METRICS[metric]
        # Names the AXIS only, though `scoped` also honours a filter.
        # This string ships to operators, and no operator can reach the filter:
        # `reads.pivot` composes `filters` itself from `subject_id`, and
        # neither surface accepts a `rubric_dim` filter — MCP would swallow it in
        # its kwarg union and REST declares no such query parameter, so the
        # advice would be followed, silently ignored, and answered with the same
        # pooled table still carrying this caveat. A remedy a reader cannot
        # perform is the false-capability claim this measure was added to end.
        formula += f", POOLED ACROSS EVERY {scope_noun.upper()} (put '{scope_field}' on an axis for one)"
        # The pooling also changes what the cell's EVIDENCE fields count, and
        # qualifying the value while leaving `n` unqualified is the same defect
        # one field over. Pooled, a row is a (result, dimension) pair: a judged
        # result contributes one row per dimension and an unjudged result exactly
        # one unmeasured row. So four judged 3-dim results beside four unjudged
        # ones report n=12 with 4 unmeasured — which reads as 75% measured when
        # half the results were never scored. With `rubric_dim` on an axis each
        # cell holds one dimension and `n` is a result count again.
        #
        # The DISPERSION is named per weighting rather than lumped in, because
        # `_aggregate` branches: under `equal_per_scenario` (the default) the SEM
        # is over test-case means — pooling changes those means, and how many
        # observations each averages, but not how MANY MEANS the SEM is taken
        # over — so claiming the dispersion
        # rides on rows would send an operator reconstructing an interval to `n`
        # instead of the case count and hand them one too narrow by ~sqrt(n/n_cases).
        # `sample_weighted` averages the flat rows, but its SEM is clustered by test case
        # (`_aggregate`), so its dispersion does not ride on rows either.
        formula += f"; n and the outcome counts are over (result x {row_noun}) rows here, not results"
        if weighting == WEIGHTING_SAMPLE_WEIGHTED:
            formula += "; the dispersion is clustered by test case, so the rows of one case are not independent draws"
        else:
            formula += "; the dispersion is over test-case means — a BASIS pooling does not change, though the means themselves do"
    return formula


def _aggregate(values_by_case: dict[str, list[float]], weighting: str) -> tuple[float, float | None]:
    """Roll one cell's observations into a value plus its dispersion.

    Args:
        values_by_case: Observed values grouped by test case.
        weighting: Which of :data:`WEIGHTINGS` to apply.

    Returns:
        ``(value, sem)``. The SEM is over whatever the value averages, so the
        two always describe the same estimate, and it counts cases, not observations:
        under ``sample_weighted`` the flat mean takes the cluster-robust SEM over the
        cases (:func:`~threetears.evals.analysis.stats.clustered_standard_error`), since
        a case's repeats are not independent draws.
    """
    from threetears.evals.analysis.stats import clustered_standard_error, standard_error_of_mean

    if weighting == WEIGHTING_EQUAL_PER_SCENARIO:
        case_means = [sum(vals) / len(vals) for vals in values_by_case.values()]
        return sum(case_means) / len(case_means), standard_error_of_mean(case_means)
    flat = [v for vals in values_by_case.values() for v in vals]
    cases = [case for case, vals in values_by_case.items() for _ in vals]
    return sum(flat) / len(flat), clustered_standard_error(flat, cases)


#: What both lenses group on. The predicate version is IN the key rather than a filter
#: applied beside it, because the join this gate protects *is* the grouping dict — a gate
#: keyed on a projection of the group key admits exactly the class it exists to refuse,
#: and the projected-away field is the gap.
ContestantKey = tuple[str, int]


def _contestant_key(result: EvalResult) -> ContestantKey:
    """Key a result to its contestant, and to the identity version stamped beside that key.

    One key, both lenses. ``frontier`` and ``history`` rank and trend the same
    contestants, so a grouping policy that lived in each of them separately would be
    two authorings of one rule — which is how the version gate came to exist on the
    bundle path and not here.

    The comparison this key performs is **equality, and there is no ``>=`` reading**.
    Keys stamped at different identity versions are incomparable in both directions: an
    older key is not "good enough" for a newer reader, and a newer key is not admissible
    to an older series. That is what makes this a partition rather than a compatibility
    check, and it is why a version mismatch is never resolved in one side's favour.

    **The stamp is one counter over two predicates, so it is a conservative
    discriminator rather than a precise one.** ``IDENTITY_VERSION`` backs the variant and
    context predicates together, so a bump on the CONTEXT side moves the number stamped
    beside a ``variant_key`` whose own predicate did not change — v9 and v10 are both that
    case. Such a pair carries a byte-identical digest under two stamps and is partitioned
    here anyway. That is deliberate and errs the safe way: the stamp cannot say WHICH
    predicate moved, and merging on a digest whose provenance is unverifiable is the wrong
    direction, since nothing downstream undoes a wrong merge. What it costs is a split that
    is sometimes narrower than necessary, which the disclosures state honestly rather than
    dressing up as a predicate change.

    Args:
        result: The observation to place.

    Returns:
        ``(variant_key, identity_version)`` — the key the runner stamped and the predicate
        version stamped beside it.
    """
    return (result.variant_key, result.identity_version)


def _identity_version_disclosure(version: int) -> str | None:
    """Say so when a contestant's key was minted by a predicate this build does not use.

    Derived from the same equality :func:`_contestant_key` groups on, never from a
    correlate: a sentence that agreed with the grouping on the observations we happen
    to have and disagreed on the one it exists for would be worse than none.

    Args:
        version: The predicate version stamped on the contestant's key.

    Returns:
        The sentence, or ``None`` when the key is this build's.
    """
    if version == IDENTITY_VERSION:
        return None
    return (
        f"stamped at identity version v{version}, not this build's v{IDENTITY_VERSION} — "
        "ranked separately from contestants stamped at a different version rather than pooled with them. "
        "The stamp does not say which predicate moved, so this may be a narrower split than the change warranted"
    )


def _identity_version_span(results: Iterable[EvalResult]) -> list[int]:
    """The distinct identity versions the KEYED observations in an answer were stamped at.

    Args:
        results: Every observation the answer is built over. Pass the placed rows
            rather than the assembled points: an answer whose subjects were all
            filtered out still spanned what it spanned, and a span derived downstream
            goes missing exactly when the reader most needs it.

    Returns:
        The versions, ascending.
    """
    return sorted({r.identity_version for r in results})


def _identity_span_disclosure(versions: Sequence[int]) -> str | None:
    """Explain a corpus that spans identity versions, so a doubled row is not a mystery.

    Partitioning stops two stampings of one stack being ranked as rival
    contestants; it does not stop them APPEARING as two rows. Without this sentence a
    reader meets one model twice, under two digests, with nothing saying why — which is
    the second half of the same defect and the one a per-row flag cannot answer.

    Args:
        versions: The span from :func:`_identity_version_span`.

    Returns:
        The sentence, or ``None`` when the answer rests on a single predicate — where
        there is no split to explain.
    """
    if len(versions) < 2:
        return None
    named = ", ".join(f"v{v}" for v in versions)
    return (
        f"spans identity versions {named}. Two keys stamped at different versions cannot be shown FROM THE "
        "STAMP ALONE to describe the same contestant, so they are listed separately — one model appearing "
        "more than once here is that split, not two rival configurations. The stamp covers both the variant "
        "and the context predicate, so some of these splits are narrower than the change that caused them."
    )


def pooled_composite_basis(results: Sequence[EvalResult] | Sequence[ScoreRecord]) -> CompositeBasis | None:
    """Say what a pooled composite was meaned over, and whether the pool is ragged (#638).

    Every surface that means composites across results pools numbers each meaned over whatever dimensions
    its result carried, so two pools that read alike can average different questions. Shared, as
    :func:`pooled_cost_compositions` is, so every surface reads one predicate
    (:func:`~threetears.evals.kernel.scoring.pool_composite_bases`).

    Args:
        results: The results whose composites the caller pooled, or the composite rows a pivot cell pooled
            (:attr:`ScoreRecord.dimension_basis`). A row that is not a composite row contributes nothing.

    Returns:
        The pooled basis, or None when nothing pooled carried a composite.
    """
    return pool_composite_bases(
        (result.dimension_basis if result.metric == METRIC_COMPOSITE else None)
        if isinstance(result, ScoreRecord)
        else composite_basis(result)
        for result in results
    )


def pooled_cost_compositions(results: Sequence[EvalResult] | Sequence[ScoreRecord]) -> list[list[str]]:
    """Say what a pooled cost total is made of.

    Every surface that adds ``cost_usd`` across results is adding numbers whose composition
    can differ per run, since metered calls become dollars only for a run whose operator
    declared a credit rate. The sums look identical either way, so the disclosure is the only thing
    that makes the difference readable. Shared rather than re-derived per surface for the
    usual reason: three copies of this predicate is three chances to disagree about what a
    total covers.

    Args:
        results: The results whose ``cost_usd`` the caller pooled, or the cost rows a pivot cell
            pooled (:attr:`ScoreRecord.cost_roles`, which carries the result's composition
            verbatim). A row carrying no composition — any row but a cost row — contributes none.

    Returns:
        The distinct compositions, sorted.
    """
    return sorted(list(c) for c in {tuple(result.cost_roles) for result in results if result.cost_roles is not None})


__all__ = [
    "BADGE_CASE_SET_DIFFERS",
    "BADGE_CASE_SET_UNRESOLVED",
    "BADGE_CASSETTE_MODE_DIFFERS",
    "BADGE_CONTEXT_DIFFERS",
    "BADGE_CONTEXT_INCOMPLETE",
    "BADGE_MEASUREMENT_WINDOWS_DISJOINT",
    "BADGE_ROLES_DIFFER",
    "BADGE_TOOL_CONFIG_DIFFERS",
    "CaseSetIdentity",
    "cassette_mode_disclosure",
    "CASSETTE_SPAN_CLAUSE",
    "classify_window_pairs",
    "ComparisonSet",
    "ComparisonSetsResult",
    "compute_comparison_sets",
    "ContestantKey",
    "DECLARED_INPUT_ORIGIN",
    "decompose_total_ms",
    "DEFAULT_WEIGHTING",
    "difference_was_declared_at_launch",
    "dim_judge_model",
    "disjoint_window_pairs",
    "DISJOINT_WINDOWS_CLAUSE",
    "format_window_gap",
    "FrontierDominance",
    "LatencyPartition",
    "lever_level",
    "MAX_INLINE_MEASUREMENT_WINDOWS",
    "MAX_RENDERED_WINDOW_GAPS",
    "measurement_window",
    "measurement_window_disclosure",
    "MeasurementWindow",
    "METRIC_COMPOSITE",
    "METRIC_COST_USD",
    "METRIC_GOAL_STATE",
    "metric_help",
    "METRIC_OUTCOME",
    "METRIC_SCORE",
    "METRIC_TOTAL_MS",
    "METRIC_TRANSCRIPT",
    "NULL_LEVEL",
    "OverlappingWindows",
    "PARTITION_TOLERANCE_MS",
    "place_results",
    "PlacedResult",
    "pooled_composite_basis",
    "pooled_cost_compositions",
    "project_score_records",
    "PROJECTED_METRICS",
    "ProjectionExclusions",
    "resolve_measure_name",
    "SCALAR_LEVEL_TYPES",
    "SCOPED_METRICS",
    "SCOPED_METRICS_HELP",
    "ScoreProjection",
    "ScoreRecord",
    "SUBSTITUTING_CASSETTE_MODE",
    "UNCOMPUTABLE_GAP_CLAUSE",
    "WEIGHTING_EQUAL_PER_SCENARIO",
    "WEIGHTING_SAMPLE_WEIGHTED",
    "WEIGHTINGS",
    "WindowGap",
    "WindowPairs",
    "WITHHELD_PARTS_EXCEED_WHOLE",
    "WITHHELD_UNMEASURED_COMPONENT",
]
