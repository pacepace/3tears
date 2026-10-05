"""The arm table — joined from what the analysis references, never authored as rows.

An analysis's central table answers one question: *which contestant won, which are out, and
what did the incumbent do?* Every number in it is already an
:class:`~threetears.evals.contracts.campaign.EvidenceRow` resolved under a finding, and every status
is already a claim a decision makes about the cells it names. So the table is DERIVED here
rather than authored: an analysis that re-stated the rows would re-author numbers the store
already holds, and it would do it through an LLM.

**The join needs a coordinate table, and that is the whole reason the analysis carries one.**
A decision names cells (``<variant_key>:<apparatus_class_id>``), and an arm is keyed by a
*variant*, a digest over the whole resolved contestant stack, whose levers a reader pivots on.
Nothing turns a digest back into levers, so :attr:`~threetears.evals.contracts.campaign.EvalAnalysis.variant_index`
carries the map each digest was computed from, and everything below reads coordinates through
it.

**Persisted: the coordinates. Derived: the report.** Rows, statuses and ordering are computed
on every read, so a reader can pivot a multivariate campaign on any axis without a stored table
deciding for them. Nothing here is written back onto the analysis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import Field, computed_field

from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.contracts.authored import Decision
from threetears.evals.contracts.campaign import (
    EvalAnalysis,
    FindingResolution,
    VariantIndexEntry,
)
from threetears.evals.contracts.host.values import SweepableValue
from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.surface import CellFacts
from threetears.observe import get_logger

log = get_logger(__name__)

#: Where an arm stands, according to this analysis.
#:
#: Four states, and the fourth is not a failure: a campaign that swept an arm and reached no
#: verdict on it has said something, and rendering that arm as though it had been ruled out
#: would be the analysis claiming a result it did not reach.
ArmStatus = Literal["winner", "ruled_out", "replaced_incumbent", "unresolved"]

#: Render order — a reader wants the answer, then what it beat, then what it replaced.
#:
#: An ordering rather than a sort key on the row, because the order is a rendering decision and
#: rendering decisions are not persisted.
_STATUS_ORDER: dict[ArmStatus, int] = {"winner": 0, "ruled_out": 1, "replaced_incumbent": 2, "unresolved": 3}


class ArmLevel(EvalDocumentModel):
    """One coordinate of an arm — which lever, and the level it carried."""

    axis_id: str = Field(min_length=1, description="The lever this level is on, in the host's own vocabulary.")
    display: str = Field(
        min_length=1, description="What a reader is shown — the level's own rendering, never its digest."
    )
    content_hash: str = Field(
        min_length=1,
        description=(
            "The level's identity, carried so a consumer can join this coordinate to an axis point or a "
            "declared level without re-resolving anything. Not for display: a digest has no rendering, "
            "which is why `display` exists beside it."
        ),
    )


class ArmMeasurement(EvalDocumentModel):
    """One measurement that placed at this arm, and the finding it was read from.

    ``finding_id`` rides along because a number in this table is a projection of a finding's
    evidence rather than a fact of its own: a reader who doubts it must be able to reach the
    finding it was recorded under, and a table that severed that link would be a copy of the
    evidence with no author.
    """

    measure_id: str = Field(min_length=1, description="Which measure the value is of.")
    value: float = Field(description="The measurement, in that measure's own units.")
    n: int = Field(ge=0, description="Samples behind the value.")
    dispersion: str = Field(description="Spread of the value.")
    finding_id: str = Field(min_length=1, description="The position of the finding whose evidence carried this row.")


class ArmRow(EvalDocumentModel):
    """One contestant, where it stands, and the evidence that placed there."""

    variant_key: str = Field(min_length=1, description="The arm's variant coordinate — its identity, not its label.")
    levels: list[ArmLevel] = Field(
        default_factory=list,
        description=(
            "What this arm carried, axis by axis, sorted by axis — the knobs it SWEPT in place of any resolved "
            "surface they explain (`VariantIndexEntry.named_levers`). EMPTY means this analysis cannot say "
            "what the arm ran — a state to render as such rather than one to fill in from somewhere "
            "else. `levels_unavailable` carries why when the index knows, and is None for the two cases "
            "it does not: the declared control, which gets a row whether or not the index holds it, and "
            "a host that resolved no levers, whose arm is described exactly by carrying none."
        ),
    )
    levels_unavailable: str | None = Field(
        default=None,
        description=(
            "Why this arm's levels cannot be described, when the index says so. Carried onto the row "
            "rather than left to the reader to infer from empty `levels`: an arm nobody can describe "
            "and an arm described by a predicate this build does not have are the same SHAPE and "
            "different facts, and only one of them is a version boundary the reader can act on."
        ),
    )
    status: ArmStatus = Field(description="Where this arm stands, read off the decisions that name its cells.")
    is_control: bool = Field(
        default=False,
        description=(
            "Whether the campaign declared this arm as its control. Carried beside `status` rather than "
            "folded into it: a control that WON is still the incumbent, and a reader is owed both facts."
        ),
    )
    finding_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Positions of the findings the decisions naming this arm rest on — the 'why' column's source. "
            "Empty for an arm no decision names, which is precisely what `unresolved` means."
        ),
    )
    measurements: list[ArmMeasurement] = Field(
        default_factory=list,
        description=(
            "Every evidence row that placed at this arm, in findings order and then in the order "
            "that finding's payload lists its rows — the generator's own ordering, not a re-sort."
        ),
    )

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "One reading per measure, in first-cited order: the first finding's evidence row for each "
            "`measure_id` in `measurements`. Computed on the server so the page and the MCP table show one "
            "number for a measure several findings cite, rather than each choosing its own."
        )
    )
    @property
    def readings(self) -> list[ArmMeasurement]:
        """This arm's measurements, one per measure — the first finding's reading of each."""
        first: dict[str, ArmMeasurement] = {}
        for measurement in self.measurements:
            first.setdefault(measurement.measure_id, measurement)
        return list(first.values())

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "What this arm is called — `arm_label` over the row's own `variant_key`, `levels` and "
            "`levels_unavailable`, computed on the server and served so every surface prints one spelling. "
            "Without the control marker, which each table spells for itself."
        )
    )
    @property
    def label(self) -> str:
        """The arm's name, as :func:`arm_label` spells it from this row's own fields."""
        return arm_label(self.variant_key, self.levels, levels_unavailable=self.levels_unavailable)


class ArmTable(EvalDocumentModel):
    """The derived comparison — every arm the campaign observed, and where it stands."""

    rows: list[ArmRow] = Field(
        default_factory=list,
        description=(
            "The arms, ordered winner → ruled out → replaced incumbent → unresolved. Empty when the "
            "analysis carries no variant index and declared no control — which is not the same fact as a "
            "campaign with no arms. `EvalAnalysis.variant_index` names what produces an empty index; read "
            "an empty table through that field rather than through a second account here."
        ),
    )
    unplaced_coordinates: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence coordinates that named no arm in this table, rendered for a reader, sorted and "
            "de-duplicated. BOTH coordinate kinds: an axis point whose level no indexed variant carried, "
            "and a cell reference naming an unknown variant. Reported rather than dropped — a measurement "
            "at a coordinate nothing here can place is evidence the reader is not seeing, and silence "
            "would render as though it had been weighed. The axis branch is the one production takes, so "
            "accounting for only the other would state the rule where it costs nothing."
        ),
    )
    unplaced_decision_cells: list[str] = Field(
        default_factory=list,
        description=(
            "Cells a DECISION names whose arm has no row here — a verdict the join could not place, so the "
            "table reads `unresolved` while a decision names a winner. Empty is the ordinary case: generation "
            "refuses a cell the surface does not hold."
        ),
    )


# --- How an arm is named — the one construction every table and chart reads ---------------------
#
# The arm table, the decision-surface table and the charts compiled from references each name arms,
# and a reader who meets one arm under two names on one page reads two arms. So the three facts a
# name is built from — which levels, in what order; whether the arm spans several rigs; how much of a
# digest tells two apart — are derived here once, and every one of those surfaces calls these.

#: How many characters of a digest — an arm's variant key or a rig's apparatus class id — tell two
#: apart where one has to be shown. One width for both, so a rig reads alike in every table and chart.
DIGEST_CHARS = 12


def arm_levers(entry: VariantIndexEntry | None) -> list[tuple[str, SweepableValue]]:
    """The levers an arm is NAMED by, as ``(axis, level)`` pairs sorted by axis.

    Read through :attr:`~threetears.evals.contracts.campaign.VariantIndexEntry.named_levers`, so an arm is
    named by the knob it swept rather than by the resolved surface that knob was written into. The
    full level is returned — scale included — for the one consumer that needs more than its
    rendering (a sweep chart declares each lever's orderedness from its scale).

    Args:
        entry: The arm's index entry, or ``None`` for an arm the index does not hold.

    Returns:
        The pairs; empty for an arm the index does not hold, for one whose levels are unavailable, and
        for one whose host resolved no levers.
    """
    return [] if entry is None else sorted(entry.named_levers.items())


def distinguishing_axes(entries: Iterable[VariantIndexEntry]) -> frozenset[str]:
    """The axes on which an analysis's arms differ — the only ones that tell one arm from another.

    An arm is named by what separates it from the other arms of its campaign. A lever every arm
    carried at the same level (a subject's description, a model no arm moved) names none of them, and
    printing it on every row buries the one setting that differs under ones that do not. An axis one
    arm carries and another lacks differs: absence is a level here, which is how the control of a
    one-knob sweep stays distinct from the arm that swept it.

    Read over the entries whose levels are known; an entry with ``levels_unavailable`` carries no
    levers and says nothing either way. With fewer than two such entries nothing can be compared,
    so every axis the one arm carried names it.

    Args:
        entries: The analysis's variant index.

    Returns:
        The axis names whose level is not the same on every described arm.
    """
    described = [entry for entry in entries if entry.levels_unavailable is None]
    named = [entry.named_levers for entry in described]
    axes = {axis for levers in named for axis in levers}
    if len(described) < 2:
        return frozenset(axes)
    return frozenset(
        axis for axis in axes if len({levers[axis].content_hash if axis in levers else None for levers in named}) > 1
    )


def arm_levels(entry: VariantIndexEntry | None, distinguishing: frozenset[str]) -> list[ArmLevel]:
    """The levels that tell an arm apart, axis by axis — what every table row and chart label names it by.

    Args:
        entry: The arm's index entry, or ``None`` for an arm the index does not hold.
        distinguishing: The analysis's :func:`distinguishing_axes`, read over its whole variant index so
            one arm is named alike on every surface.

    Returns:
        One :class:`ArmLevel` per :func:`arm_levers` pair on a distinguishing axis, in that order.
    """
    return [
        ArmLevel(axis_id=axis, display=level.display, content_hash=level.content_hash)
        for axis, level in arm_levers(entry)
        if axis in distinguishing
    ]


def writer_arms(entries: Sequence[VariantIndexEntry]) -> dict[str, object]:
    """The report writer's view of a campaign's arms — interpreted facts, never the raw index.

    The index is the key's pre-image, so it carries every resolved lever as a content hash,
    folded surfaces included. A writer handed that reads a folded surface's hash as a second
    moved setting and reports it as a confound, and finds the one setting that differs among a
    dozen that do not. So the writer gets what code has already decided: each arm named by the
    levels that tell it apart (the same :func:`distinguishing_axes` every table and chart names
    arms by), and the settings every arm shared, once. Display strings only — a hash says nothing
    a writer can use.

    Args:
        entries: The bundle's variant index.

    Returns:
        ``{"arms": [...], "shared_levels": {...}}``. Each arm carries ``variant_key``, ``levels``
        (axis → display) and ``levels_unavailable`` (why its levels cannot be described, or None).
    """
    distinguishing = distinguishing_axes(entries)
    arms: list[dict[str, object]] = []
    for entry in entries:
        arm: dict[str, object] = {
            "variant_key": entry.variant_key,
            "levels": {level.axis_id: level.display for level in arm_levels(entry, distinguishing)},
            "levels_unavailable": entry.levels_unavailable,
        }
        arms.append(arm)
    # A non-distinguishing axis carries one level on every described arm, so any one of them
    # states it; an axis some arm lacks is distinguishing by construction and never lands here.
    described = next((entry for entry in entries if entry.levels_unavailable is None), None)
    shared = (
        {}
        if described is None
        else {axis: level.display for axis, level in arm_levers(described) if axis not in distinguishing}
    )
    return {"arms": arms, "shared_levels": shared}


def multi_rig_variants(cells: Iterable[CellFacts]) -> frozenset[str]:
    """The variants measured under more than one rig among ``cells``.

    One stack under two rigs is two cells with one name, so a surface names the rig beside exactly
    these arms and no others. The population is the caller's: a table asks it of every cell it
    lays out, a chart of the cells it draws.

    Args:
        cells: The cells to consider.

    Returns:
        Every variant key those cells carry under two or more distinct apparatus class ids.
    """
    rigs: dict[str, set[str]] = {}
    for cell in cells:
        rigs.setdefault(cell.variant_key, set()).add(cell.apparatus_class_id)
    return frozenset(variant for variant, seen in rigs.items() if len(seen) > 1)


def short_digest(digest: str) -> str:
    """Cut a variant key or an apparatus class id to the :data:`DIGEST_CHARS` that tell it apart.

    Args:
        digest: The full id.

    Returns:
        Its first :data:`DIGEST_CHARS` characters.
    """
    return digest[:DIGEST_CHARS]


def arm_label(
    variant_key: str,
    levels: Sequence[ArmLevel],
    *,
    levels_unavailable: str | None,
    placed: bool = True,
    rig: str | None = None,
) -> str:
    """Name one arm — or one cell of it — in the words every python surface prints.

    Its levels as ``axis=display`` when it has any — only the distinguishing ones its caller passes
    (:func:`arm_levels`); a described arm with none of those runs the shared settings and says so.
    Otherwise a LABELLED digest: an arm printed as a
    bare hash reads as an arm with a strange name, when what the reader needs to know is that this
    analysis cannot say what it ran. The label says which of the two reasons applies — the arm is
    not in the variant index at all (``placed=False``), or the index holds it and cannot describe
    its levels, with the index's reason when it has one, because naming the version boundary turns
    a gap into a fact about when the arm ran. A label is never borrowed from a neighbouring arm.

    The rig rides on the end when the caller says the arm was measured under more than one; the
    control marker does not, because each table spells its own.

    The browser does not spell these words at all: :class:`ArmRow` and
    :class:`~threetears.evals.analysis.surface_table.SurfaceRow` serve the result as ``label``, and the
    kit prints what it is served — a second spelling in a second language is a second name.

    Args:
        variant_key: The arm's variant coordinate.
        levels: Its levels, as :func:`arm_levels` returns them.
        levels_unavailable: Why the index cannot describe it, when it says.
        placed: Whether the variant index holds this arm at all.
        rig: The rig's short digest, only when the arm spans more than one rig.

    Returns:
        The label.
    """
    digest = short_digest(variant_key)
    if not placed:
        name = f"unplaced ({digest}) — not in this analysis's variant index"
    elif levels:
        name = ", ".join(f"{level.axis_id}={level.display}" for level in levels)
    elif levels_unavailable:
        name = f"levels unavailable ({digest}) — {levels_unavailable}"
    else:
        # Placed and described, with no level on a distinguishing axis: this arm moved nothing the
        # others did, so it runs exactly the settings every arm shares.
        name = "the shared settings (no lever moved)"
    return f"{name} @ rig {rig}" if rig else name


def cell_label(
    variant_key: str,
    apparatus_class_id: str,
    *,
    index: Mapping[str, VariantIndexEntry],
    multi_rig: frozenset[str],
) -> str:
    """Name one cell — an arm under one rig — from the index and the multi-rig population.

    The composition every surface naming a cell makes: the arm's levels (or the fact that the index
    cannot place or describe it), and the rig's short digest exactly when the arm was measured under
    more than one rig in the caller's population.

    Args:
        variant_key: The cell's variant coordinate.
        apparatus_class_id: The rig it was measured under.
        index: Variant key → index entry.
        multi_rig: The variants measured under more than one rig, as :func:`multi_rig_variants` returns them.

    Returns:
        The label, as :func:`arm_label` spells it.
    """
    entry = index.get(variant_key)
    return arm_label(
        variant_key,
        arm_levels(entry, distinguishing_axes(index.values())),
        levels_unavailable=entry.levels_unavailable if entry else None,
        placed=entry is not None,
        rig=short_digest(apparatus_class_id) if variant_key in multi_rig else None,
    )


def _status_and_why(
    decisions: Sequence[Decision], variant_key: str, *, is_control: bool, a_winner_exists: bool
) -> tuple[ArmStatus, list[str]]:
    """Where one arm stands, read off the decisions that name its cells.

    An adopted decision naming one of the arm's cells makes it the winner; a rejected one rules it
    out; a declared control is replaced when some other arm won. Everything else is unresolved —
    including an arm only a deferred decision names, since deferring is not a verdict.

    Returns:
        ``(status, finding_ids)`` — where the arm stands, and the positions of the findings the
        deciding decision rests on.
    """
    rulings: tuple[tuple[str, ArmStatus], ...] = (("adopted", "winner"), ("rejected", "ruled_out"))
    for disposition, status in rulings:
        for decision in decisions:
            if decision.disposition == disposition and any(
                variant_of_cell_ref(cell) == variant_key for cell in decision.cells
            ):
                return status, [str(position) for position in decision.rests_on]
    if is_control and a_winner_exists:
        return "replaced_incumbent", []
    return "unresolved", []


def _measurements(
    resolutions: Sequence[FindingResolution], arms: list[str]
) -> tuple[dict[str, list[ArmMeasurement]], list[str]]:
    """Place every resolved evidence row onto the arm its cell belongs to.

    Returns:
        ``(by_arm, unplaced)`` — the measurements at each arm, and the cells that named no arm in
        the table.
    """
    placed: dict[str, list[ArmMeasurement]] = {key: [] for key in arms}
    unplaced: set[str] = set()
    for position, resolution in enumerate(resolutions):
        for row in resolution.evidence:
            measurement = ArmMeasurement(
                measure_id=row.measure_id, value=row.value, n=row.n, dispersion=row.dispersion, finding_id=str(position)
            )
            variant = variant_of_cell_ref(row.cell_ref or "")
            if variant is not None and variant in placed:
                placed[variant].append(measurement)
            else:
                unplaced.add(row.cell_ref or "")
    return placed, sorted(unplaced)


def _unplaced_decision_cells(decisions: Sequence[Decision], arms: list[str]) -> list[str]:
    """Every cell a decision names whose arm has no row in this table — a verdict with nowhere to land."""
    return sorted({cell for decision in decisions for cell in decision.cells if variant_of_cell_ref(cell) not in arms})


def arm_table(analysis: EvalAnalysis) -> ArmTable:
    """Derive the arm comparison for one stored analysis.

    Args:
        analysis: The analysis to read. Nothing is written back to it.

    Returns:
        The table, as :func:`arm_table_of` derives it from the analysis's frozen variant index, its
        declared control, and its decisions and resolutions.
    """
    return arm_table_of(
        analysis.variant_index,
        control=analysis.design_snapshot.control if analysis.design_snapshot else None,
        decisions=analysis.document.decisions,
        resolutions=analysis.resolutions,
        source=f"eval.analysis {analysis.id}",
    )


def arm_table_of(
    variant_index: Sequence[VariantIndexEntry],
    *,
    control: str | None,
    decisions: Sequence[Decision],
    resolutions: Sequence[FindingResolution],
    source: str,
) -> ArmTable:
    """Derive the arm comparison from its inputs — a stored analysis's, or a campaign's evidence alone.

    With no decisions every arm is ``unresolved`` (a verdict is a decision's to give), and with no
    resolutions no arm carries a measurement: that is what the table says about a campaign nothing has
    analysed, and it is the truth rather than a gap.

    Args:
        variant_index: One entry per keyed variant.
        control: The declared control's variant key, or None.
        decisions: The decisions whose cells place a verdict on an arm.
        resolutions: The findings' resolved evidence, in finding order.
        source: What the table is of, named in the log line when a coordinate cannot be placed.

    Returns:
        The table. Every arm the variant index holds gets a row, plus the declared control when the
        index does not hold it — an incumbent whose levels are unavailable is still an incumbent, and
        dropping its row is what the shape was chosen to prevent.
    """
    index = {entry.variant_key: entry for entry in variant_index}
    keys = sorted(index) + ([control] if control and control not in index else [])
    distinguishing = distinguishing_axes(variant_index)

    placed, unplaced = _measurements(resolutions, keys)
    # Answered over the whole set before any row is built: "was this incumbent replaced" asks
    # whether ANY arm won, which no single row can see.
    a_winner_exists = any(
        _status_and_why(decisions, key, is_control=False, a_winner_exists=False)[0] == "winner" for key in keys
    )

    rows = []
    for key in keys:
        entry = index.get(key)
        status, finding_ids = _status_and_why(
            decisions, key, is_control=key == control, a_winner_exists=a_winner_exists
        )
        rows.append(
            ArmRow(
                variant_key=key,
                levels=arm_levels(entry, distinguishing),
                levels_unavailable=entry.levels_unavailable if entry else None,
                status=status,
                is_control=key == control,
                finding_ids=finding_ids,
                measurements=placed.get(key, []),
            )
        )
    # Sorted on the RENDERED level rather than the key, so a reader scanning a status band meets
    # the arms in the order the report names them; the key breaks ties.
    rows.sort(key=lambda row: (_STATUS_ORDER[row.status], [level.display for level in row.levels], row.variant_key))
    stranded = _unplaced_decision_cells(decisions, keys)
    if unplaced or stranded:
        # A cell that resolves to no arm renders as a neutral "no verdict" — the state an operator
        # is least able to distinguish from a real one — so it is logged as well as disclosed.
        log.warning(
            "%s: arm table could not place %d evidence cell(s) %s and %d decision cell(s) %s",
            source,
            len(unplaced),
            unplaced,
            len(stranded),
            stranded,
        )
    return ArmTable(rows=rows, unplaced_coordinates=unplaced, unplaced_decision_cells=stranded)


__all__ = [
    "DIGEST_CHARS",
    "ArmLevel",
    "ArmMeasurement",
    "ArmRow",
    "ArmStatus",
    "ArmTable",
    "arm_label",
    "arm_levels",
    "arm_levers",
    "arm_table",
    "arm_table_of",
    "cell_label",
    "distinguishing_axes",
    "multi_rig_variants",
    "short_digest",
    "writer_arms",
]
