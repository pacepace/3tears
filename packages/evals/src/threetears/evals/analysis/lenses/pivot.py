"""Pivot: any two factors as axes over the score projection, with the disclosures that make it honest.

:func:`compute_pivot` rolls the projection's rows into cells under a weighting mode, states each cell as measured,
not run, unmeasured or withheld, and flags Simpson's-paradox reversals, pooled identity versions, served-model
substitutions and withheld cost beside the table.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.host.sweepables import CANDIDATE_MODEL_LEVER
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.schema.models import SCALES
from threetears.evals.kernel.scoring import CompositeBasis
from threetears.evals.analysis.reporting import (
    _aggregate,
    _describe_aggregate,
    _effective_formula,
    _metric_vocabulary,
    cassette_mode_disclosure,
    DEFAULT_WEIGHTING,
    METRIC_COMPOSITE,
    METRIC_COST_USD,
    pooled_composite_basis,
    pooled_cost_compositions,
    pooled_served_models,
    PROJECTED_METRICS,
    ProjectionExclusions,
    resolve_measure_name,
    SCOPED_METRICS,
    ScoreRecord,
    ServedModelReading,
    SUBSTITUTING_CASSETTE_MODE,
    WEIGHTINGS,
)
from threetears.evals.analysis.lenses.cost_estimate import CostEstimate, PlannedCost, PredictedValue


# A cell's three states, which a renderer must keep visually distinct.
#
# `not_run` and a value of zero are different facts about the world, and
# collapsing them is the failure this whole projection is shaped to prevent: an
# unrun combination that renders as 0.0 reads as a catastrophic result rather
# than as an absent one.
#
# `unmeasured` is the third case — observations exist at this cell but none
# carried a value for this measure, which is neither "we never tried" nor "we
# measured zero". It is reached from real data by the projection emitting a
# null-valued row for an observation with no value: an infra-excluded result, or
# an ok result carrying no rubric dims. A projection that emitted nothing
# instead produced the exact collapse the first paragraph forbids, one state
# over — the cell fell to the empty-cell branch and claimed nobody tried a
# combination that was tried and failed in the harness.
#: A pivot cell state: observations landed here and carried a value for the measure.
CELL_MEASURED = "measured"
#: A pivot cell state: no observation landed here — the combination was never run, which is not a zero.
CELL_NOT_RUN = "not_run"
#: A pivot cell state: observations landed here, but none carried a value for the measure.
CELL_UNMEASURED = "unmeasured"
#: A fourth state, and not a kind of the other three: the cell HAS measured observations, and its
#: mean is withheld because it would pool two quantities that are not one distribution — today a
#: cost cell pooling replayed results with live ones (#658). `PivotCell.withheld` says why, and
#: `n` / `n_cases` / `outcomes` still say what the cell held, so withheld never reads as empty.
CELL_WITHHELD = "withheld"


# Transferability class that may be pooled across subjects.
#
# Composite quality is comparable within a subject and never across one,
# because rubric dimensions derive from each subject's own description and
# tools. That binds judge-mediated and scenario-bound measures. It does NOT bind
# mechanical ones — a dollar is a dollar whoever spent it — and refusing those
# too would put the cross-subject budget view out of reach for no gain.
_POOLABLE_ACROSS_SUBJECTS = "mechanical"


class PivotError(ValueError):
    """A pivot was requested that cannot be answered honestly.

    Raised rather than returned as an empty or caveated table: every case that
    reaches it is a question whose honest answer is "not like that" — an axis the
    rows do not carry, an unknown weighting mode, or a cross-subject pooling.
    Returning an empty table instead would be indistinguishable from "no data",
    which is how a refusal becomes a silent zero.
    """


class PivotCell(EvalBaseModel):
    """One (row, column) cell: the number, and everything needed to trust it."""

    row: str
    column: str
    status: str
    value: float | None = None
    # What the cell was predicted to cost per observation when it was planned — set only on a cost
    # pivot handed the estimate made before launch (`compute_pivot(predicted_cost=...)`). Kept beside
    # `value` rather than folded into it so a prediction can never be mistaken for an observation.
    predicted: PredictedValue | None = None
    # Beside a prediction, how many of the cell's observations came from runs the plan did not make — the earlier
    # history its prediction may have been drawn from among them. None when there is no prediction, or when the
    # plan names no launched runs (it had not launched), so every observation here is one it did not make.
    n_unplanned: int | None = None
    # Observations that carried a value, and the distinct test cases they span.
    # Under equal-per-scenario weighting `n_cases` is the denominator the value
    # was divided by, so showing both is what lets a reader see that a cell's
    # mean rests on three cases or on thirty.
    n: int = 0
    n_cases: int = 0
    # Standard error of whatever the value averages — case means under
    # equal-per-scenario, raw observations under sample-weighted — so the spread
    # always describes the estimate actually reported.
    sem: float | None = None
    # Observations present at this cell that carried no value for this measure.
    # Counted rather than dropped: an aggregate over 3 of 12 observations is a
    # different claim from an aggregate over 12.
    n_unmeasured: int = 0
    # Observation counts per scoring outcome, so a cell whose mean rests largely
    # on candidate failures cannot look like one that rests on clean passes.
    outcomes: dict[str, int] = {}
    #: Why the cell's value is withheld, set exactly when ``status`` is ``withheld``: on a cost pivot,
    #: the cell pools results from runs that replayed their third party with results from runs that
    #: ran it live, and a replayed result did not spend what a live one does, so their mean is neither
    #: one's spend (#658). The counts above still say what the cell held.
    withheld: str | None = None
    #: The cassette modes the runs behind the cell's valued observations recorded, sorted. One entry is
    #: a uniform cell; ``replay`` beside another mode is the mix a cost cell withholds.
    cassette_modes: list[str] = []
    #: On a cost pivot, the role sets the cell's dollars were summed over
    #: (:func:`pooled_cost_compositions`), from the observations that carried a value (#625). More than
    #: one entry means the cell's own mean pools totals that covered different things; two cells whose
    #: entries differ are not comparable on cost, which :attr:`PivotTable.cost_compositions_differ`
    #: flags at the table. Empty on any other metric.
    cost_compositions: list[list[str]] = []
    #: On a composite pivot, what the cell's composites were meaned over (:func:`pooled_composite_basis`):
    #: the union of the observations' bases, and ``ragged`` when they were meaned over different dimension
    #: sets, so the cell's mean averages different questions (#638). ``None`` on any other metric and on a
    #: cell with no valued observation.
    composite_basis: CompositeBasis | None = None
    #: Of the ``n`` valued observations, how many carried a background delivery a harness supplied — seeded
    #: or replayed (:func:`~threetears.evals.kernel.usage_capture.count_substituted_deliveries`). Counted on
    #: every metric, because a substituted delivery is what the candidate read as well as what it did not pay
    #: for. ``0`` on a cell whose observations ran every delivery live.
    n_substituted: int = 0
    #: On a cost pivot, the sentence a cell carries when ``n_substituted`` is above zero: a substituted delivery
    #: spent none of its dollars, so the cell's spend leaves them out, and a cell built ONLY from such
    #: observations is no live run's spend at all. ``None`` on any other metric and on a cell with none.
    #: Stated on the cell rather than left to the export's ``substituted_deliveries`` column, because the cell
    #: is what is read.
    substitution_disclosure: str | None = None
    #: Identity key -> the predicate versions its observations here were stamped at, for each identity
    #: key (``variant_key``, ``context_key``) the table groups or filters on, and only where the cell
    #: pools more than one (#672). Two keys stamped at different versions cannot be shown FROM THE STAMP
    #: ALONE to describe one contestant — the frontier ranks them apart for that reason — so a cell
    #: pooling them may be averaging two conditions as repeats of one. Disclosed rather than split,
    #: because the pivot's axes are open and no one axis may imply a partition.
    identity_versions: dict[str, list[int]] = {}
    #: On a table grouped by contestant (``variant_key`` or ``model`` on an axis), which models the provider's
    #: responses named as having answered the candidate calls of the cell's observations (#684): ``one``,
    #: ``pooled`` — one requested model answered by several, so the cell's number mixes them — or
    #: ``unrecorded``. Disclosed rather than split, since the contestant is keyed at launch; ``served_model`` is a
    #: coordinate, so pivoting on it separates them. ``None`` on a table grouped on neither, and on a cell none of
    #: whose observations' candidate made a call.
    served_models: ServedModelReading | None = None


class SimpsonsFlag(EvalBaseModel):
    """A pooled column ranking that the per-row rankings mostly contradict.

    The pooled table says ``leader`` beats the other column overall, while more
    rows than not rank them the other way — the pooled number is being driven by
    which rows each column was measured on rather than by the columns
    themselves. Descriptive: the flag says the comparison is unsafe to read as a
    ranking, not which answer is correct.
    """

    column_a: str
    column_b: str
    pooled_leader: str
    rows_agreeing: int
    rows_disagreeing: int
    disagreeing_rows: list[str]


class PivotTable(EvalBaseModel):
    """A two-factor pivot over one measure, with its disclosures attached.

    ``formula`` states what was computed under the *requested weighting* rather
    than repeating the registry's single static formula, which is itself
    weighting-specific and would otherwise describe a different number than the
    one displayed. Everything not weighting-dependent — family, unit, direction —
    still comes from ``measure``, the registry descriptor.
    """

    metric: str
    measure: MetricDescriptor
    formula: str
    weighting: str
    row_factor: str
    column_factor: str
    rows: list[str]
    columns: list[str]
    cells: list[PivotCell]
    simpsons_flags: list[SimpsonsFlag] = []
    # Corpus-level accounting, so a table that answers over a fraction of the
    # supplied rows says so rather than looking complete. `n_filtered_out` counts
    # rows this query's own filters removed; `exclusions` counts observations
    # that never became rows at all, which is the difference between "you asked
    # for a subset" and "this data cannot be placed".
    n_observations: int = 0
    n_filtered_out: int = 0
    exclusions: ProjectionExclusions = ProjectionExclusions()
    #: ``run_id -> DEGRADED sentence`` for the runs behind these cells that measured
    #: less than the matrix they promised. The counterpart of ``exclusions`` on the
    #: other side of the seam: that field accounts for observations the table does
    #: NOT contain, this one qualifies observations it does. Every cell is affected,
    #: not a nameable subset — a pivot aggregates over every coordinate that is not
    #: an axis, and ``run_id`` is usually one of them — so the disclosure is stated
    #: at the table rather than marked per cell.
    completeness_disclosures: dict[str, str] = {}
    #: How many of ``n_observations`` came from those runs. The weight the caveat
    #: carries: two of two hundred is a footnote and two of four is the answer.
    n_degraded_observations: int = 0
    #: On a cost pivot, whether the table's valued observations were summed over more than one role set
    #: (#625) — within one cell or between cells. True means some cost here covered roles another did not,
    #: so a cheaper cell may only have priced fewer things; each cell's ``cost_compositions`` says which.
    cost_compositions_differ: bool = False
    #: On a composite pivot, whether the table's valued composites were meaned over more than one dimension
    #: set (#638) — within one cell or between cells. True means a difference between two cells may be a
    #: difference in what was averaged rather than in what was measured; each cell's ``composite_basis`` says
    #: which sets it pooled.
    composite_bases_differ: bool = False
    #: The sentence a comparison carries when its runs recorded different cassette modes
    #: (:func:`cassette_mode_disclosure`, the words ``runs_compare`` and ``comparison_sets`` use), over the
    #: runs behind this table's observations, or ``None`` when they all recorded one (#658). It qualifies
    #: every metric, not only cost: a replayed arm was also measured on the questions its capture asked.
    cassette_mode_disclosure: str | None = None
    #: One sentence naming every cell that pools more than one identity version of the key it is grouped
    #: or filtered on (:attr:`PivotCell.identity_versions`), or ``None`` when none does (#672).
    identity_pooling_disclosure: str | None = None
    #: One sentence naming every cell whose observations were answered by more than one served model, and
    #: counting those that cannot say (:attr:`PivotCell.served_models`), or ``None`` when every cell names one
    #: model or the table is not grouped by contestant (#684).
    served_model_disclosure: str | None = None
    #: Plans the cost estimate made that no cell describes — a model no level of the model axis carries, or a
    #: template no cell at that model holds alone — each as ``model`` or ``model (template)``. Named rather than
    #: dropped, since a prediction with nowhere to sit is still a fact about the plan: an arm that was priced and
    #: did not run, or ran where this table does not separate it. Empty when no estimate was given.
    unplaced_predicted_models: list[str] = []


def _reads_a_lever(factor: str, *, records: Sequence[ScoreRecord], profile: HostProfile) -> bool:
    """Decide where an axis or filter name is read: off the run's levers, or off a declared field.

    **The registry decides, and the name's shape decides nothing** (#664). This once routed on
    ``"." in factor`` — dotted read ``factors``, undotted had to be a ``ScoreRecord`` field — on the
    assumption that a host's levers are dotted. They are not: ``factors`` holds every lever
    :func:`_run_factors` resolves, under the name the host gave it, so a plain ``chunk_tokens`` reached
    the rows and was refused at the pivot while a campaign could declare it as an axis.

    A name is a **lever** when the host's registry admits it as one
    (:meth:`~threetears.evals.kernel.host.sweepables.SweepableRegistry.refuse_as_axis`: a fixed lever,
    or a member some open family's membership test claims — the same decision a campaign's declared axis
    goes through), or when some row carries it in ``factors``, which only the registry's own resolution
    writes. The second half covers a family that declares no membership test: its members are knowable
    only from the runs that carried them. A lever no run set still pivots, as a table of ``"—"``, because
    "nobody set this" is an answer; a name neither half admits is a typo.

    **One name, two meanings, is refused rather than resolved by branch order.** A host lever that takes
    a declared coordinate's name (``template_id``, ``scope_id``, …) would be read off whichever branch
    ran first, so it is refused, naming the rename. Refused here rather than at registration because
    the registry lives in ``kernel.host`` and the coordinate set is this module's record — the kernel
    does not import ``analysis``. The one exception is the engine's own candidate-model lever: the
    projection writes it into the ``model`` coordinate and keeps it out of ``factors`` (see
    :func:`_run_factors`), so the two names are one quantity and the field is read.

    Args:
        factor: An axis or filter key a caller sent.
        records: The rows the pivot reads, for the members a family names only at run time.
        profile: The host whose registry decides.

    Returns:
        True to read ``factors[factor]``, False to read the declared field.

    Raises:
        PivotError: The open-coordinate or host-measure map itself, a name that is both a lever and a
            declared coordinate, or a name that is neither — the last naming the host's actual levers.
    """
    if factor == "factors":
        # The map is the container for open coordinates, never one itself —
        # stringifying it would collapse every run into one dict-shaped bucket.
        raise PivotError("'factors' is the open-coordinate map, not a coordinate — pivot on one of the levers it holds")
    if factor == "host_measures":
        # Same defect, and one step further from being an axis: these are the host's own
        # MEASUREMENTS of a cell, so grouping by one would partition the corpus by its own
        # answer. Refused here rather than left to `_axis_value`, which would stringify the
        # whole map and bucket every result under one dict.
        raise PivotError(
            "'host_measures' is the host's own grade of a cell, not a coordinate — read it from the export, "
            "where each measure is its own column"
        )
    if factor == CANDIDATE_MODEL_LEVER:
        return False
    # Checked against the declared fields, not `hasattr`: every model also
    # exposes methods, so `hasattr` would accept `model_dump` as an axis and
    # bucket the whole corpus under one stringified bound method.
    declared = factor in ScoreRecord.model_fields
    registry = profile.sweepables
    lever = registry.refuse_as_axis(factor) is None or any(factor in record.factors for record in records)
    if declared and lever:
        raise PivotError(
            f"factor {factor!r} is both a declared score-record coordinate and a lever this host registers — "
            "the pivot will not pick one by rule of thumb; rename the lever"
        )
    if lever:
        return True
    if declared:
        return False
    raise PivotError(
        f"unknown factor {factor!r} — score records carry no such coordinate and this host registers no such "
        f"lever. {registry.axis_remedy}; or a declared coordinate (model, template_id, test_case_id, ...)"
    )


def _axis_value(record: ScoreRecord, factor: str, *, lever: bool) -> str:
    """Read one factor off a row, as the string an axis is keyed on.

    Two coordinate spaces, resolved in one place. A **declared** field
    (``model``, ``template_id``, …) is read by name; a **lever** is read off the
    open ``factors`` map, which carries every lever the host's registry resolves
    for the run — a kind's overlays (``gm.difficulty``) among them — at their
    resolved levels. Which space a name belongs to is decided once per pivot by
    :func:`_reads_a_lever`, from the host's registry, never from the name's shape.

    An absent value becomes a visible ``"—"`` level rather than being dropped,
    so a corpus where half the runs carry no override for a key shows that as a
    row of its own instead of quietly shrinking. That makes "ran without this
    override" a comparable cohort rather than missing data — which is exactly
    what a bake-off is asking about.

    Args:
        record: The row to read.
        factor: A declared coordinate name or a lever name.
        lever: What :func:`_reads_a_lever` decided for ``factor``.

    Returns:
        The coordinate's value as a string.
    """
    if lever:
        return record.factors.get(factor) or "—"
    value = getattr(record, factor)
    return "—" if value is None or value == "" else str(value)


def _simpsons_flags(
    cells: dict[tuple[str, str], PivotCell],
    pooled: dict[str, float],
    rows: list[str],
    columns: list[str],
) -> list[SimpsonsFlag]:
    """Flag every column pair whose pooled order most rows contradict.

    ``pooled`` must be recomputed from the underlying observations with the row
    axis collapsed — the figure an operator sees when they *don't* break the
    comparison down — and emphatically not averaged from the cell values above.
    An unweighted mean of per-row cells cannot reverse the per-row order at all,
    so a guard built that way is arithmetically incapable of firing. The reversal
    lives precisely in the row weighting: a column measured mostly on easy rows
    outranks one measured mostly on hard rows even while losing every row.

    Pooling collapses rows, so the caller puts the thing being compared on the
    columns and the suspected confounder on the rows. Per-row leaders are read
    only from rows where both columns were measured — a row carrying just one of
    them has no order to contribute.

    Args:
        cells: Measured cells keyed by ``(row, column)``.
        pooled: Column level -> its value with the row axis collapsed.
        rows: Row levels, in display order.
        columns: Column levels, in display order.

    Returns:
        One flag per reversed pair, in column order. Empty when nothing reverses.
    """
    flags: list[SimpsonsFlag] = []
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            if col_a not in pooled or col_b not in pooled or pooled[col_a] == pooled[col_b]:
                continue
            pooled_leader = col_a if pooled[col_a] > pooled[col_b] else col_b

            per_row_leaders = []
            for row in rows:
                cell_a, cell_b = cells.get((row, col_a)), cells.get((row, col_b))
                if cell_a is None or cell_b is None or cell_a.value is None or cell_b.value is None:
                    continue
                if cell_a.value == cell_b.value:
                    continue
                per_row_leaders.append((row, col_a if cell_a.value > cell_b.value else col_b))

            if len(per_row_leaders) < 2:
                # One row cannot be a "majority of per-cell rankings"; flagging on
                # it would report every ordinary sampling difference as a paradox.
                continue

            disagreeing = [row for row, leader in per_row_leaders if leader != pooled_leader]
            agreeing = len(per_row_leaders) - len(disagreeing)
            if len(disagreeing) > agreeing:
                flags.append(
                    SimpsonsFlag(
                        column_a=col_a,
                        column_b=col_b,
                        pooled_leader=pooled_leader,
                        rows_agreeing=agreeing,
                        rows_disagreeing=len(disagreeing),
                        disagreeing_rows=disagreeing,
                    )
                )
    return flags


#: Each identity key a pivot can group on, and the coordinate carrying the predicate version that minted it.
_IDENTITY_KEY_VERSION_FIELDS: dict[str, str] = {
    "variant_key": "variant_identity_version",
    "context_key": "context_identity_version",
}


def _pooled_identity_versions(records: Sequence[ScoreRecord], keys: Iterable[str]) -> dict[str, list[int]]:
    """The identity versions a cell pools, for each identity key it is grouped or filtered on (#672).

    Args:
        records: The cell's observations.
        keys: The identity keys among the table's axes and filters.

    Returns:
        Key -> its versions, ascending, only for a key whose observations here carry more than one.
    """
    pooled: dict[str, list[int]] = {}
    for key in keys:
        field = _IDENTITY_KEY_VERSION_FIELDS[key]
        versions = sorted({version for record in records if (version := getattr(record, field)) is not None})
        if len(versions) > 1:
            pooled[key] = versions
    return pooled


def _identity_pooling_disclosure(cells: Sequence[PivotCell]) -> str | None:
    """Name every cell that pools more than one identity version of the key it is grouped on.

    The frontier's reason in the pivot's terms: two keys stamped at different versions cannot be shown from
    the stamp alone to describe one contestant, so the frontier ranks them apart (:func:`_contestant_key`).
    The pivot's axes are open, so it does not split; it says which cells pool and how to read them apart.

    Args:
        cells: The table's cells.

    Returns:
        The sentence, or ``None`` when no cell pools versions.
    """
    pooled = [cell for cell in cells if cell.identity_versions]
    if not pooled:
        return None
    named = "; ".join(
        f"({cell.row}, {cell.column}): "
        + ", ".join(
            f"{key} stamped at {', '.join(f'v{version}' for version in versions)}"
            for key, versions in cell.identity_versions.items()
        )
        for cell in pooled
    )
    return (
        f"{len(pooled)} cell(s) pool observations stamped at more than one identity version of the key they are "
        f"grouped on — {named}. Two keys stamped at different versions cannot be shown FROM THE STAMP ALONE to "
        "describe the same contestant (the frontier ranks them apart), so such a cell may average two conditions "
        "as repeats of one. Pivot the key against its version coordinate (variant_identity_version or "
        "context_identity_version) to read each version alone."
    )


#: The axes that group a pivot by contestant, on which each cell names the models that answered it (#684).
_CONTESTANT_FACTORS = frozenset({"variant_key", "model"})


def _served_model_disclosure(cells: Sequence[PivotCell]) -> str | None:
    """Name every cell answered by more than one served model, and count those that cannot say (#684).

    Args:
        cells: The table's cells.

    Returns:
        The sentence, or ``None`` when every cell that carries a reading names one model.
    """
    pooled = [cell for cell in cells if cell.served_models is not None and cell.served_models.state == "pooled"]
    unrecorded = [cell for cell in cells if cell.served_models is not None and cell.served_models.state == "unrecorded"]
    parts: list[str] = []
    if pooled:
        named = "; ".join(
            f"({cell.row}, {cell.column}): {', '.join(cell.served_models.served_models)}"
            for cell in pooled
            if cell.served_models is not None
        )
        parts.append(
            f"{len(pooled)} cell(s) pool observations one requested model was answered by several models — {named} — "
            "so each such number mixes them; pivot on served_model to read each model alone"
        )
    if unrecorded:
        parts.append(
            f"{len(unrecorded)} cell(s) rest on candidate calls whose response named no model, so whether one model "
            "answered them cannot be established"
        )
    return ". ".join(parts) + "." if parts else None


def _substitution_disclosure(n_substituted: int, n_valued: int) -> str | None:
    """The sentence a cost cell carries when some of its observations had a delivery a harness supplied.

    Disclosed rather than withheld, for the reason :func:`_cost_withheld` gives: two arms over a seeded
    template carry the same substitutions, so the comparison between their cells is honest. But a reader of
    one cell's dollars needs to know they leave the substituted deliveries' spend out — and when every
    observation substituted, that the figure describes no live run.

    Args:
        n_substituted: Valued observations carrying at least one substituted delivery.
        n_valued: Valued observations in the cell.

    Returns:
        The sentence, or ``None`` when nothing was substituted.
    """
    if n_substituted == 0:
        return None
    share = "every one" if n_substituted == n_valued else f"{n_substituted}"
    return (
        f"{share} of the {n_valued} observation(s) behind this spend carried a background delivery a harness "
        "supplied (seeded or replayed), which spent none of its dollars, so they are not in this figure"
        + (": it is no live run's spend." if n_substituted == n_valued else ".")
    )


def _cost_withheld(records: Sequence[ScoreRecord]) -> str | None:
    """Why a cost mean over these valued observations is withheld, or ``None`` when it is reported (#658).

    **A cost mean that pools replayed with live results is withheld rather than disclosed**, because a
    caveat beside a number does not stop it being read, and this number is neither population's spend. The
    observations come from runs that recorded ``replay`` and runs that ran the third party live: a replayed
    background delivery spent none of its dollars, and a replay serves only the asks its capture made (any
    other ask stops the cell), so even where nothing was substituted the replayed conversations are a
    selected population whose spend is not the live one's. A mix of ``off`` and ``capture`` is not withheld:
    both run the third party live.

    A uniform cell — every observation replayed, or every one live — is reported: its mean is one
    population's, and :attr:`PivotTable.cassette_mode_disclosure` says when the table's cells differ.

    **A substituted delivery within one mode is not a reason to withhold.** A seeded finding substitutes in
    a run that recorded ``off``, but it does so on the template's own cases, so two arms over those cases
    carry the same substitutions and the comparison between their cells is honest; the dollars it did not
    spend were never measuring spend either. The cell says so (:attr:`PivotCell.substitution_disclosure`), and
    each row's ``substituted_deliveries`` is an export column. Withholding on it would blank the cost of every
    arm over a seeded template.

    Args:
        records: The observations a cost mean would be taken over.

    Returns:
        The reason, or ``None``.
    """
    modes = sorted({record.cassette_mode for record in records if record.cassette_mode is not None})
    if SUBSTITUTING_CASSETTE_MODE in modes and len(modes) > 1:
        live = ", ".join(mode for mode in modes if mode != SUBSTITUTING_CASSETTE_MODE)
        return (
            f"pools results from runs that recorded cassette mode {SUBSTITUTING_CASSETTE_MODE} with results from "
            f"runs that recorded {live}. A replayed result re-served a recording rather than running its third "
            "party: where it replayed a background delivery it spent none of that delivery's dollars, and a "
            "replay serves only the asks its capture made, so the mean of the two is neither one's spend. Put "
            "'cassette_mode' on an axis to read each alone."
        )
    return None


def compute_pivot(
    records: list[ScoreRecord],
    *,
    row_factor: str,
    column_factor: str,
    metric: str,
    weighting: str = DEFAULT_WEIGHTING,
    filters: dict[str, str] | None = None,
    exclusions: ProjectionExclusions | None = None,
    completeness_disclosures: Mapping[str, str] | None = None,
    predicted_cost: CostEstimate | Sequence[PlannedCost] | None = None,
    profile: HostProfile,
) -> PivotTable:
    """Aggregate score records over any two factors, disclosing every caveat.

    The grid is the full cross-product of the axis levels actually observed, so
    a combination that was never run appears as a ``not_run`` cell rather than
    being absent from the table — an absent cell is indistinguishable from a
    zero once a renderer lays the grid out.

    **Pooling across subjects is refused for anything but a mechanical measure.**
    A pivot aggregates over every factor that is not an axis, so a
    ``model × template`` pivot over a multi-subject corpus would average one
    subject's composites with another's — different measurements wearing the same
    number, because rubric dimensions derive from each subject's own description
    and tools. Cost and latency are mechanical and are pooled freely. To pivot a
    judge-mediated measure over several subjects, put ``subject_id`` on an axis
    or filter to one.

    **A run that measured less than its matrix is aggregated and disclosed, never
    refused**, on the position :func:`compute_frontier` states: its rows are real
    measurements, and dropping them silently is the failure this tier exists to
    avoid. Every cell is affected rather than a nameable subset — a pivot pools over
    every coordinate that is not an axis, and ``run_id`` usually is not one — so the
    caveat is carried at the table in ``completeness_disclosures`` rather than marked
    per cell. The predicate is the completeness record, never the run's status.

    **What a cell pools that is not one quantity is said, or its number withheld.** On a cost pivot each
    cell names the role sets its dollars covered (``cost_compositions``) and the table flags when they
    differ (#625); a cost cell pooling replayed results with live ones is ``withheld`` with the reason,
    since a caveat does not stop a mean being read (#658); the table carries the comparison surfaces' cassette-mode sentence when its runs recorded
    different modes; and a cell grouped or filtered on an identity key that pools more than one version of
    its predicate names them (#672).

    Args:
        records: Rows from :func:`project_score_records`.
        row_factor: Coordinate to use as the row axis.
        column_factor: Coordinate to use as the column axis. This is the axis the
            Simpson's guard pools over rows, so put the thing being compared here.
        metric: Which measure to aggregate; rows carrying others are ignored. A
            measure in :data:`PROJECTED_METRICS`, or the registry name of its
            aggregate as ``list_metrics`` publishes it — see
            :func:`resolve_measure_name`, which holds the pairing. Stated
            relationally rather than as a list: the set has grown once already,
            and an enumeration here is one more place that would not have moved
            with it.
        weighting: One of :data:`WEIGHTINGS`.
        filters: Optional exact-match coordinate filters applied before
            aggregating, e.g. ``{"subject_id": "subj-1"}``.
        exclusions: What the projection dropped before these records existed,
            carried onto the table so the answer discloses what it could not
            see. Omitting it renders an all-excluded corpus as an empty one.
        completeness_disclosures: ``run_id -> DEGRADED sentence`` from
            :func:`degraded_run_disclosures`, for the runs behind these records.
            Narrowed here to the runs that survived ``filters``, so the table
            never carries a caveat about a run it did not aggregate. **Omitting
            it pools a short run's rows into every rate with nothing saying so**
            — the defect this parameter exists to end, and the reason it is a
            parameter rather than something re-derived per surface.
        predicted_cost: The estimate made BEFORE these observations, when the cells were planned
            (:func:`compute_estimate_cost`), or the planned costs of a launch priced by its host's pricer
            (:class:`PlannedCost`, one per priced arm). Each cell at a planned model carries that model's
            prediction in ``predicted``, beside the cost it observed and never in place of it. Passed
            in rather than computed here because a prediction drawn from a history that already
            holds the observations it predicts would be a restatement of them. Only a cost pivot
            with the candidate model on an axis has a cell a planned model's cost describes.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`PivotTable` whose every cell carries ``n``, dispersion, and its
        measured/unmeasured/not-run/withheld status, plus the completeness disclosures of
        the short runs its numbers were pooled from and the pooling disclosures above.

    Raises:
        PivotError: Unknown weighting or metric, an axis or filter that is neither a
            declared coordinate nor a lever of this host (or is both), a cross-subject pooling, or a predicted cost handed
            to a pivot of another metric or one with no model axis. Every one
            is refused here rather than at an adapter, so both surfaces refuse
            identically instead of one of them answering an empty grid.
    """
    if weighting not in WEIGHTINGS:
        raise PivotError(f"unknown weighting {weighting!r} — expected one of {', '.join(WEIGHTINGS)}")
    # A cell holds `mean_composite`, so `mean_composite` is a name an operator can
    # legitimately arrive with — it is what `list_metrics` publishes for the
    # quantity. Resolved to the row name before the closed-set check, never after:
    # the check is what makes an unknown name a refusal instead of an empty grid.
    # Keep what the caller actually sent. Resolution maps a catalog name onto the row
    # name, and `mean_total_ms` is IN the alias table (it has to be, for `history`)
    # while `total_ms` is not a measure `pivot` accepts — so refusing from the rebound
    # name told an operator "unknown metric 'total_ms'" about a string they never
    # typed, and about the one name the docs say this surface does not take. The
    # refusal names their input; the accepted set names both vocabularies.
    requested = metric
    metric = resolve_measure_name(metric)
    if metric not in PROJECTED_METRICS:
        refusal = f"unknown metric {requested!r} — expected one of {_metric_vocabulary(PROJECTED_METRICS)}"
        # Naming what is available is not enough when the caller asked for a
        # DIMENSION: they wanted a real capability, reached for it under the
        # wrong noun, and a bare list of measure names reads as "per-dimension
        # means are unavailable" — the false-absence answer that sends a reader
        # away from a route that exists. Checked against the rows rather than
        # guessed from the name, so the sentence is only added when it is true.
        #
        # Names the AXIS alone. A `rubric_dim` filter scopes the same cell one
        # layer down, but `reads.pivot` composes `filters` itself and
        # neither surface accepts one — so of the two remedies, only the axis is
        # reachable by whoever reads this. Advice a reader cannot act on is the
        # false capability claim this refusal exists to prevent, and it would
        # land here of all places: this string fires precisely when someone has
        # already reached for the capability under the wrong noun.
        for scoped_metric, (scope_field, scope_noun, _row_noun) in SCOPED_METRICS.items():
            if any(getattr(record, scope_field) == metric for record in records):
                refusal += f"; {metric!r} is a {scope_noun}, not a measure — aggregate {scoped_metric!r} with '{scope_field}' on an axis"
                break
        raise PivotError(refusal)
    # Decided before any row is grouped: grouping runs only once there are rows, so over an empty
    # selection a typo'd axis would otherwise answer an empty grid instead of a refusal.
    reads_lever = {
        name: _reads_a_lever(name, records=records, profile=profile)
        for name in (row_factor, column_factor, *(filters or {}))
    }
    predictions = _planned_cost_per_observation(predicted_cost, metric, (row_factor, column_factor))

    measure = _describe_aggregate(metric, profile=profile)

    of_metric = [r for r in records if r.metric == metric]
    selected = of_metric
    for factor, wanted in (filters or {}).items():
        selected = [r for r in selected if _axis_value(r, factor, lever=reads_lever[factor]) == wanted]
    # A per-dimension score's catalogue range spans every scale, since the catalogue cannot know
    # which dimensions a table holds. The table can: when every row was judged on one scale, its
    # measure carries that scale's range, so a 1-5 table does not claim a floor of 0.
    if len(table_scales := {r.rubric_scale for r in selected if r.rubric_scale is not None}) == 1:
        measure = measure.model_copy(update={"value_range": SCALES[table_scales.pop()].value_range})

    # The no-pooling rule, enforced before any number is computed rather than as a badge after:
    # a table that has already averaged across subjects cannot be un-averaged by
    # a caveat, and the reader most likely to miss the caveat is the one reading
    # a single headline figure.
    if (
        measure.transferability_class != _POOLABLE_ACROSS_SUBJECTS
        and row_factor != "subject_id"
        and column_factor != "subject_id"
    ):
        subjects = {r.subject_id for r in selected}
        if len(subjects) > 1:
            raise PivotError(
                f"{measure.name!r} is {measure.transferability_class} and spans {len(subjects)} subjects — "
                "put subject_id on an axis or filter to one; pooling it across subjects compares "
                "measurements derived from different rubrics"
            )

    grouped: dict[tuple[str, str], list[ScoreRecord]] = {}
    for record in selected:
        grouped.setdefault(
            (
                _axis_value(record, row_factor, lever=reads_lever[row_factor]),
                _axis_value(record, column_factor, lever=reads_lever[column_factor]),
            ),
            [],
        ).append(record)

    # A mean of 1-5 levels and 1/0 pass/fail answers is neither a level nor a pass rate, so a cell
    # pooling both is refused before it is computed, like the subject pooling above.
    for (row, column), members in grouped.items():
        if len(scales := {r.rubric_scale for r in members if r.rubric_scale is not None}) > 1:
            raise PivotError(
                f"cell ({row}, {column}) pools rubric dimensions judged on different scales ({', '.join(sorted(scales))})"
                " — put 'rubric_dim' on an axis so each cell holds one dimension"
            )

    rows = sorted({row for row, _ in grouped})
    columns = sorted({column for _, column in grouped})
    # The identity keys this table groups or filters on, each of which must not silently pool versions.
    identity_keys = sorted(
        {name for name in (row_factor, column_factor, *(filters or {})) if name in _IDENTITY_KEY_VERSION_FIELDS}
    )
    # A table grouped by contestant names, per cell, the models that answered it (#684).
    by_contestant = bool({row_factor, column_factor} & _CONTESTANT_FACTORS)

    cells: list[PivotCell] = []
    measured: dict[tuple[str, str], PivotCell] = {}
    placed: set[str] = set()
    for row in rows:
        for column in columns:
            at_cell = grouped.get((row, column), [])
            plan = _plan_for(predictions, row, column, at_cell, (row_factor, column_factor))
            planned = plan.predicted if plan is not None else None
            unplanned = (
                sum(1 for record in at_cell if record.run_id not in plan.run_ids)
                if plan is not None and plan.run_ids
                else None
            )
            if plan is not None:
                placed.add(plan.label)
            if not at_cell:
                cells.append(
                    PivotCell(row=row, column=column, status=CELL_NOT_RUN, predicted=planned, n_unplanned=unplanned)
                )
                continue

            outcomes: dict[str, int] = {}
            for record in at_cell:
                outcomes[record.outcome] = outcomes.get(record.outcome, 0) + 1

            values_by_case: dict[str, list[float]] = {}
            for record in at_cell:
                if record.value is not None:
                    values_by_case.setdefault(record.test_case_id, []).append(record.value)

            n_valued = sum(len(v) for v in values_by_case.values())
            identity_versions = _pooled_identity_versions(at_cell, identity_keys)
            served = pooled_served_models(at_cell) if by_contestant else None
            if not values_by_case:
                cells.append(
                    PivotCell(
                        predicted=planned,
                        n_unplanned=unplanned,
                        row=row,
                        column=column,
                        status=CELL_UNMEASURED,
                        n_unmeasured=len(at_cell),
                        outcomes=outcomes,
                        identity_versions=identity_versions,
                        served_models=served,
                    )
                )
                continue

            # What the value is drawn over is the valued observations, so the qualifiers below read those.
            valued = [record for record in at_cell if record.value is not None]
            n_substituted = sum(1 for record in valued if record.substituted_deliveries > 0)
            qualifiers: dict[str, Any] = {
                "cassette_modes": sorted({r.cassette_mode for r in valued if r.cassette_mode is not None}),
                "cost_compositions": pooled_cost_compositions(valued) if metric == METRIC_COST_USD else [],
                "composite_basis": pooled_composite_basis(valued) if metric == METRIC_COMPOSITE else None,
                "identity_versions": identity_versions,
                "served_models": served,
                "n_substituted": n_substituted,
                "substitution_disclosure": (
                    _substitution_disclosure(n_substituted, len(valued)) if metric == METRIC_COST_USD else None
                ),
            }
            withheld = _cost_withheld(valued) if metric == METRIC_COST_USD else None
            if withheld is not None:
                cells.append(
                    PivotCell(
                        predicted=planned,
                        n_unplanned=unplanned,
                        row=row,
                        column=column,
                        status=CELL_WITHHELD,
                        withheld=withheld,
                        n=n_valued,
                        n_cases=len(values_by_case),
                        n_unmeasured=len(at_cell) - n_valued,
                        outcomes=outcomes,
                        **qualifiers,
                    )
                )
                continue

            value, sem = _aggregate(values_by_case, weighting)
            cell = PivotCell(
                predicted=planned,
                n_unplanned=unplanned,
                row=row,
                column=column,
                status=CELL_MEASURED,
                value=value,
                n=n_valued,
                n_cases=len(values_by_case),
                sem=sem,
                n_unmeasured=len(at_cell) - n_valued,
                outcomes=outcomes,
                **qualifiers,
            )
            cells.append(cell)
            measured[(row, column)] = cell

    # The column figures with the row axis collapsed — what the operator reads
    # when they stop breaking the comparison down. Computed from the observations
    # rather than from the cells above, because the row weighting is the entire
    # mechanism the Simpson's guard exists to catch.
    # A cost column whose observations the cells' own rule would withhold has no pooled figure either.
    pooled: dict[str, float] = {}
    for column in columns:
        in_column = [
            r
            for r in selected
            if _axis_value(r, column_factor, lever=reads_lever[column_factor]) == column and r.value is not None
        ]
        if metric == METRIC_COST_USD and _cost_withheld(in_column) is not None:
            continue
        by_case: dict[str, list[float]] = {}
        for record in in_column:
            assert record.value is not None
            by_case.setdefault(record.test_case_id, []).append(record.value)
        if by_case:
            pooled[column], _ = _aggregate(by_case, weighting)

    return PivotTable(
        metric=metric,
        measure=measure,
        formula=_effective_formula(
            metric,
            weighting,
            scoped=metric not in SCOPED_METRICS
            or SCOPED_METRICS[metric][0] in (row_factor, column_factor, *(filters or {})),
        ),
        weighting=weighting,
        row_factor=row_factor,
        column_factor=column_factor,
        rows=rows,
        columns=columns,
        cells=cells,
        simpsons_flags=_simpsons_flags(measured, pooled, rows, columns),
        n_observations=len(selected),
        n_filtered_out=len(of_metric) - len(selected),
        exclusions=exclusions or ProjectionExclusions(),
        # Narrowed to the runs that reached a cell. A subject filter or a metric
        # selection can remove a short run entirely, and a caveat about a run the
        # table never averaged is one the reader cannot act on or check.
        completeness_disclosures={
            run_id: sentence
            for run_id, sentence in (completeness_disclosures or {}).items()
            if any(record.run_id == run_id for record in selected)
        },
        n_degraded_observations=sum(1 for record in selected if record.run_id in (completeness_disclosures or {})),
        cost_compositions_differ=metric == METRIC_COST_USD
        and len(pooled_cost_compositions([r for r in selected if r.value is not None])) > 1,
        composite_bases_differ=metric == METRIC_COMPOSITE
        and (table_basis := pooled_composite_basis([r for r in selected if r.value is not None])) is not None
        and table_basis.ragged,
        cassette_mode_disclosure=cassette_mode_disclosure(
            {record.run_id: record.cassette_mode for record in selected if record.cassette_mode is not None}
        ),
        identity_pooling_disclosure=_identity_pooling_disclosure(cells),
        served_model_disclosure=_served_model_disclosure(cells),
        unplaced_predicted_models=sorted({plan.label for plan in predictions} - placed),
    )


#: The coordinate a pivot axis names the template by.
_TEMPLATE_FACTOR = "template_id"


class _Plan(NamedTuple):
    """One plan as a pivot places it: whose cells it describes, and its prediction per observation."""

    model: str
    template_id: str | None
    run_ids: frozenset[str]
    predicted: PredictedValue

    @property
    def label(self) -> str:
        """The plan as an unplaced list names it."""
        return self.model if self.template_id is None else f"{self.model} ({self.template_id})"


def _planned_cost_per_observation(
    estimate: CostEstimate | Sequence[PlannedCost] | None, metric: str, axes: tuple[str, str]
) -> list[_Plan]:
    """Each plan's predicted cost per observation, read off the estimate made before the run.

    A planned cell's prediction is its sweep's TOTAL (``n_observations`` draws), and a pivot cell's
    value is a mean per observation, so the prediction is divided by the planned observation count —
    and so is its band. That is not a rescaling of convenience: the mean of the ``n`` planned
    observations is their total over ``n``, so the band on the total, divided by ``n``, is exactly the
    band on that mean, at the same level and on the same assumptions.
    One prediction, read two ways.

    Args:
        estimate: The estimate, or the planned costs, or None.
        metric: The pivot's resolved metric.
        axes: The pivot's row and column factors.

    Returns:
        One plan per priced model and template; empty when no estimate was given.

    Raises:
        PivotError: An estimate was given to a pivot of another metric, or to one with no model axis — a
            planned model's cost describes neither, so its prediction would sit beside a number it does not
            predict — or it plans one model on one template twice, which leaves no answer to which prediction
            a cell carries.
    """
    if estimate is None:
        return []
    if metric != METRIC_COST_USD:
        raise PivotError(
            f"a predicted cost sits beside an observed cost, and this pivot aggregates {metric!r} — "
            f"pivot {METRIC_COST_USD!r} to set the estimate beside what was spent"
        )
    if CANDIDATE_MODEL_LEVER not in axes:
        raise PivotError(
            f"the estimate predicts cost per planned model, and neither axis is {CANDIDATE_MODEL_LEVER!r} — "
            "put the model on an axis so each prediction has the cells it planned"
        )
    planned = estimate.planned_costs() if isinstance(estimate, CostEstimate) else list(estimate)
    keys = [(cell.model, cell.template_id) for cell in planned]
    if repeated := sorted({key for key in keys if keys.count(key) > 1}, key=str):
        named = ", ".join(model if template is None else f"{model} on {template}" for model, template in repeated)
        raise PivotError(
            f"the estimate plans {named} more than once, so no cell can say which prediction it carries — hand "
            "the pivot one plan per model and template"
        )
    plans: list[_Plan] = []
    for cell in planned:
        if cell.predicted is None:
            continue
        n = cell.n_observations
        plans.append(
            _Plan(
                model=cell.model,
                template_id=cell.template_id,
                run_ids=frozenset(cell.run_ids),
                predicted=cell.predicted.model_copy(
                    update={
                        "value": cell.predicted.value / n,
                        "interval_low": None
                        if cell.predicted.interval_low is None
                        else cell.predicted.interval_low / n,
                        "interval_high": None
                        if cell.predicted.interval_high is None
                        else cell.predicted.interval_high / n,
                    }
                ),
            )
        )
    return plans


def _plan_for(
    plans: list[_Plan], row: str, column: str, records: list[ScoreRecord], axes: tuple[str, str]
) -> _Plan | None:
    """The plan a pivot cell's observations are, or None — matched on the model and the template the cell holds.

    The cell's template is the template axis's level when one axis is the template, else the one template every
    observation in it shares; a cell holding several, or an empty cell with no template axis, holds none a
    template's plan can claim. A plan that names no template matches on the model alone, and one naming the
    cell's template is preferred to it.
    """
    row_factor, column_factor = axes
    model = row if row_factor == CANDIDATE_MODEL_LEVER else column
    templates: set[str | None]
    if _TEMPLATE_FACTOR in axes:
        templates = {row if row_factor == _TEMPLATE_FACTOR else column}
    else:
        templates = {record.template_id for record in records}
    template = next(iter(templates)) if len(templates) == 1 else None
    at_model = [plan for plan in plans if plan.model == model]
    return next(
        (plan for plan in at_model if plan.template_id is not None and plan.template_id == template), None
    ) or next((plan for plan in at_model if plan.template_id is None), None)


__all__ = [
    "CELL_MEASURED",
    "CELL_NOT_RUN",
    "CELL_UNMEASURED",
    "CELL_WITHHELD",
    "compute_pivot",
    "PivotCell",
    "PivotError",
    "PivotTable",
    "SimpsonsFlag",
]
