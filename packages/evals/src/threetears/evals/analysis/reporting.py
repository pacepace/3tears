"""Query-time projection of eval runs + results into flat, comparable rows.

This module is the read tier's foundation: one projection,
:func:`project_score_records`, that every downstream reporting surface consumes
(pivots, export, and the estimation engine). Those surfaces are the read lenses, one module
each under :mod:`threetears.evals.analysis.lenses` — among them
:mod:`~threetears.evals.analysis.lenses.comparison_sets`, which answers *which of these runs
may honestly be compared with each other*. The disclosures the lenses and the context bundle
share sit beside this module: :mod:`~threetears.evals.analysis.completeness`,
:mod:`~threetears.evals.analysis.significance`, :mod:`~threetears.evals.analysis.measurement_windows`
and :mod:`~threetears.evals.analysis.cassette_mode`, with the per-result latency split in
:mod:`~threetears.evals.analysis.latency_partition`.

**Nothing here is stored.** Records are recomputed per query, exactly as
:func:`~threetears.evals.kernel.scoring.compute_pass_hat_k` is. That keeps the row shape free to
evolve while the surfaces that consume it are still being learned — a persisted
projection would freeze it against every future consumer. Aggregation is
calibrated to a corpus of dozens-to-hundreds of cells (one operator, one
scope); :func:`project_score_records` is the seam to push down into SQL if
that ever stops holding. ``pivot`` and ``export_results``
(:mod:`threetears.evals.analysis.reads`) log each call's rows in, records and
cells out and wall time, at WARNING past ``READ_TIER_ROW_BUDGET``, so a host
sees the premise being crossed (#652).

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

from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pydantic import Field, model_validator

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.hashing import canonical_json
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.host.sweepables import CANDIDATE_MODEL_LEVER
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
    "dim_judge_model",
    "FrontierDominance",
    "lever_level",
    "METRIC_COMPOSITE",
    "METRIC_COST_USD",
    "METRIC_GOAL_STATE",
    "METRIC_OUTCOME",
    "METRIC_SCORE",
    "METRIC_TOTAL_MS",
    "METRIC_TRANSCRIPT",
    "NULL_LEVEL",
    "place_results",
    "PlacedResult",
    "pooled_composite_basis",
    "pooled_cost_compositions",
    "project_score_records",
    "PROJECTED_METRICS",
    "ProjectionExclusions",
    "SCALAR_LEVEL_TYPES",
    "SCOPED_METRICS",
    "SCOPED_METRICS_HELP",
    "ScoreProjection",
    "ScoreRecord",
]
