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
from threetears.evals.contracts.host.sweepables import CANDIDATE_KIND_LEVER, CANDIDATE_MODEL_LEVER
from threetears.evals.contracts.host.values import SweepableValue
from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.surface import CellFacts
from threetears.observe import get_logger

log = get_logger(__name__)

#: Where an arm stands, according to this analysis.
#:
#: Five states. ``unresolved`` is not a failure: a campaign that swept an arm and reached no
#: verdict on it has said something, and rendering that arm as though it had been ruled out
#: would be the analysis claiming a result it did not reach. ``contradicted`` is an arm one
#: decision adopts and another rejects: generation refuses that memo, so only an analysis stored
#: before the refusal carries one, and neither verdict is shown as standing.
ArmStatus = Literal["winner", "contradicted", "ruled_out", "replaced_incumbent", "unresolved"]

#: Render order — a reader wants the answer, then any verdict the memo contradicts, then what it
#: beat, then what it replaced.
#:
#: An ordering rather than a sort key on the row, because the order is a rendering decision and
#: rendering decisions are not persisted.
_STATUS_ORDER: dict[ArmStatus, int] = {
    "winner": 0,
    "contradicted": 1,
    "ruled_out": 2,
    "replaced_incumbent": 3,
    "unresolved": 4,
}


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
    label: str = Field(
        min_length=1,
        description=(
            "What this arm is called — `arm_names` read over the analysis's whole variant index, so every "
            "surface prints one spelling and no two arms print the same one: its `levels` as `axis=level`, "
            "each level on one line and cut in the middle past `LABEL_LEVEL_CHARS`, with the variant key's "
            "digest added where that would read like another arm's. Without the control marker, which each "
            "table spells for itself."
        ),
    )
    levels: list[ArmLevel] = Field(
        default_factory=list,
        description=(
            "What this arm is NAMED by, sorted by axis: the levers on which the analysis's arms differ, read "
            "through `VariantIndexEntry.named_levers` (the knobs it SWEPT in place of any resolved surface they "
            "explain), never one that does not apply to its kind — or, where it differs on none, its candidate "
            "kind and model. Displays whole; `label` is where they are cut. EMPTY for an arm this analysis "
            "cannot describe (`levels_unavailable` says why when the index knows; the declared control gets a "
            "row whether or not the index holds it), and for a described arm carrying no lever at all."
        ),
    )
    settings: list[ArmLevel] = Field(
        default_factory=list,
        description=(
            "Every lever this arm ran, at its level, sorted by axis — the full set `levels` is a selection "
            "from, displays whole, stated once per arm here rather than in every row and chart that names it. "
            "A lever that does not apply to the arm's kind is not one it ran and is left out. Empty where "
            "`levels` is empty for want of a description."
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


class ArmTable(EvalDocumentModel):
    """The derived comparison — every arm the campaign observed, and where it stands."""

    rows: list[ArmRow] = Field(
        default_factory=list,
        description=(
            "The arms, ordered winner → contradicted → ruled out → replaced incumbent → unresolved. Empty when the "
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
# and a reader who meets one arm under two names on one page reads two arms. So the facts a name is
# built from — which levels, in what order; how a long one is cut; whether the arm spans several rigs;
# how much of a digest tells two apart — are derived here once, and every one of those surfaces calls
# these. The rule, in full:
#
# - An arm is named by the levers on which the report's arms DIFFER (:func:`distinguishing_axes`), and
#   by nothing else: a lever every arm carried at one level names none of them.
# - A lever that does not apply to the arm's kind (its level is
#   :meth:`~threetears.evals.contracts.host.values.SweepableValue.not_this_kind`'s) is never named.
# - An arm that differs on nothing it carries is named by its candidate kind and model.
# - Each level is put on one line and cut in the middle past :data:`LABEL_LEVEL_CHARS` (:func:`elide_level`).
# - Two arms the above would name alike are told apart by their variant keys' digests (:func:`arm_names`).
# - A cell names its rig too, exactly when its arm was measured under more than one.
#
# The full set of levers an arm ran is stated once, on its arm-table row (:attr:`ArmRow.settings`).

#: How many characters of a digest — an arm's variant key or a rig's apparatus class id — tell two
#: apart where one has to be shown. One width for both, so a rig reads alike in every table and chart.
DIGEST_CHARS = 12

#: The longest a level runs inside an arm's name before :func:`elide_level` cuts it. Long enough for a
#: model id whole; a prompt or a long description is what it cuts.
LABEL_LEVEL_CHARS = 48

#: What stands in for the characters :func:`elide_level` cut.
ELISION = "…"

#: What a described arm is called when it carries no lever that names it — no distinguishing lever, and
#: neither a candidate kind nor a model to fall back on.
NO_LEVER_MOVED = "the shared settings (no lever moved)"


def elide_level(display: str) -> str:
    """A level's display as an arm's name carries it: on one line, and cut in the middle when long.

    The middle rather than the end, because the end is where two model ids, two builds or two versions
    of a prompt usually part: a head-only cut eats exactly the characters that tell them apart. The cut
    can still leave two levels reading alike — :func:`arm_names` tells such arms apart, so nothing here
    has to.

    Args:
        display: The level's display.

    Returns:
        The display with its whitespace collapsed, cut to :data:`LABEL_LEVEL_CHARS` characters (the
        :data:`ELISION` mark included) when longer.
    """
    text = " ".join(display.split())
    if len(text) <= LABEL_LEVEL_CHARS:
        return text
    kept = LABEL_LEVEL_CHARS - len(ELISION)
    tail = kept // 3
    return f"{text[: kept - tail]}{ELISION}{text[-tail:]}"


def arm_levers(entry: VariantIndexEntry | None) -> list[tuple[str, SweepableValue]]:
    """The levers an arm is NAMED by, as ``(axis, level)`` pairs sorted by axis.

    Read through :attr:`~threetears.evals.contracts.campaign.VariantIndexEntry.named_levers`, so an arm is
    named by the knob it swept rather than by the resolved surface that knob was written into. The
    full level is returned — scale included — for the one consumer that needs more than its
    rendering (a sweep chart declares each lever's orderedness from its scale). Levers that do not
    apply to the arm's kind are included; :func:`applicable_levers` is this without them.

    Args:
        entry: The arm's index entry, or ``None`` for an arm the index does not hold.

    Returns:
        The pairs; empty for an arm the index does not hold, for one whose levels are unavailable, and
        for one whose host resolved no levers.
    """
    return [] if entry is None else sorted(entry.named_levers.items())


def applicable_levers(entry: VariantIndexEntry | None) -> list[tuple[str, SweepableValue]]:
    """:func:`arm_levers` without the levers that do not apply to the arm's kind.

    A run of one kind carries every other kind's levers at their
    :meth:`~threetears.evals.contracts.host.values.SweepableValue.not_this_kind` level, because that is
    what keeps the variant key honest. It is not something the arm RAN, so no name and no list of what
    an arm ran states it.

    Args:
        entry: The arm's index entry, or ``None`` for an arm the index does not hold.

    Returns:
        The pairs, sorted by axis.
    """
    return [(axis, level) for axis, level in arm_levers(entry) if level.not_of_kind is None]


def distinguishing_axes(entries: Iterable[VariantIndexEntry]) -> frozenset[str]:
    """The axes on which an analysis's arms differ — the only ones that tell one arm from another.

    An arm is named by what separates it from the other arms of its campaign. A lever every arm
    carried at the same level (a subject's description, a model no arm moved) names none of them, and
    printing it on every row buries the one setting that differs under ones that do not. An axis one
    arm carries and another lacks differs: absence is a level here, which is how the control of a
    one-knob sweep stays distinct from the arm that swept it.

    A lever is compared only across the arms it APPLIES to. On an arm of another kind it sits at the
    engine's "not a run of this kind" level, and that difference is the arms' kinds differing, which
    the candidate kind already says: counting it would name an arm by every lever of its kind because
    an arm of another kind has none of them.

    Read over the entries whose levels are known; an entry with ``levels_unavailable`` carries no
    levers and says nothing either way. With fewer than two such entries nothing differs, so no axis
    distinguishes: one arm measured under several rigs is told apart by its rigs, not by every lever
    it carried.

    Args:
        entries: The analysis's variant index.

    Returns:
        The axis names whose level is not the same on every described arm it applies to.
    """
    described = [entry for entry in entries if entry.levels_unavailable is None]
    if len(described) < 2:
        return frozenset()
    named = [entry.named_levers for entry in described]
    axes = {axis for levers in named for axis in levers}
    return frozenset(
        axis
        for axis in axes
        if len(
            {
                levers[axis].content_hash if axis in levers else None
                for levers in named
                if axis not in levers or levers[axis].not_of_kind is None
            }
        )
        > 1
    )


def _arm_level(axis: str, level: SweepableValue) -> ArmLevel:
    return ArmLevel(axis_id=axis, display=level.display, content_hash=level.content_hash)


def arm_levels(entry: VariantIndexEntry | None, distinguishing: frozenset[str]) -> list[ArmLevel]:
    """The levels that tell an arm apart, axis by axis — its applicable levers on a distinguishing axis.

    Args:
        entry: The arm's index entry, or ``None`` for an arm the index does not hold.
        distinguishing: The analysis's :func:`distinguishing_axes`, read over its whole variant index so
            one arm is named alike on every surface.

    Returns:
        One :class:`ArmLevel` per :func:`applicable_levers` pair on a distinguishing axis, in that order.
    """
    return [_arm_level(axis, level) for axis, level in applicable_levers(entry) if axis in distinguishing]


def naming_levels(entry: VariantIndexEntry | None, distinguishing: frozenset[str]) -> list[ArmLevel]:
    """The levels an arm is NAMED by — what every table row and chart label prints for it.

    Its :func:`arm_levels` when it has any. A described arm with none — the only arm of its report, or
    one whose every difference is a lever that does not apply to it — falls back to its candidate kind
    and model, which say what ran without repeating every setting it shares.

    Args:
        entry: The arm's index entry, or ``None`` for an arm the index does not hold.
        distinguishing: The analysis's :func:`distinguishing_axes`.

    Returns:
        The levels, sorted by axis; empty for an arm the index does not hold or cannot describe, and for a
        described arm with neither a distinguishing lever nor a kind or model.
    """
    levels = arm_levels(entry, distinguishing)
    if levels or entry is None or entry.levels_unavailable is not None:
        return levels
    return [
        _arm_level(axis, level)
        for axis, level in applicable_levers(entry)
        if axis in (CANDIDATE_KIND_LEVER, CANDIDATE_MODEL_LEVER)
    ]


def arm_settings(entry: VariantIndexEntry | None) -> list[ArmLevel]:
    """Every lever an arm ran, at its level — the full set its name is a selection from.

    Args:
        entry: The arm's index entry, or ``None`` for an arm the index does not hold.

    Returns:
        One :class:`ArmLevel` per :func:`applicable_levers` pair, displays whole; empty where the index
        cannot say what the arm ran.
    """
    return [_arm_level(axis, level) for axis, level in applicable_levers(entry)]


def writer_arms(entries: Sequence[VariantIndexEntry]) -> dict[str, object]:
    """The report writer's view of a campaign's arms — interpreted facts, never the raw index.

    The index is the key's pre-image, so it carries every resolved lever as a content hash,
    folded surfaces included. A writer handed that reads a folded surface's hash as a second
    moved setting and reports it as a confound, and finds the one setting that differs among a
    dozen that do not. So the writer gets what code has already decided: the settings every arm
    shared, once, and each arm's other levers — in a campaign of one kind, exactly the levels that
    tell it apart (:func:`distinguishing_axes`); across kinds, also the levers only its kind has.
    Neither ever names a lever that does not apply to the arm's kind. Display strings only — a hash
    says nothing a writer can use.

    Args:
        entries: The bundle's variant index.

    Returns:
        ``{"arms": [...], "shared_levels": {...}}``. Each arm carries ``variant_key``, ``levels``
        (axis → display) and ``levels_unavailable`` (why its levels cannot be described, or None).
    """
    described = [dict(applicable_levers(entry)) for entry in entries if entry.levels_unavailable is None]
    # Shared: carried by every described arm, at one level — so any one of them states it.
    shared = (
        {
            axis: level
            for axis, level in described[0].items()
            if all(axis in levers and levers[axis].content_hash == level.content_hash for levers in described)
        }
        if described
        else {}
    )
    arms: list[dict[str, object]] = [
        {
            "variant_key": entry.variant_key,
            "levels": {axis: level.display for axis, level in applicable_levers(entry) if axis not in shared},
            "levels_unavailable": entry.levels_unavailable,
        }
        for entry in entries
    ]
    return {"arms": arms, "shared_levels": {axis: level.display for axis, level in shared.items()}}


def arm_names(entries: Sequence[VariantIndexEntry]) -> dict[str, str]:
    """Name every arm of a variant index — distinct names, read over the whole index at once.

    Each described arm is named by its :func:`naming_levels`, as ``axis=level`` pairs with every level
    cut by :func:`elide_level`; an arm whose levels are unavailable is named as such, with its digest
    and the index's reason, because naming the version boundary turns a gap into a fact about when the
    arm ran. A label is never borrowed from a neighbouring arm.

    **Distinct by construction.** Two arms can read alike: a cut can stop before the characters that
    differ, and a host's own display can abbreviate two values to one. Every arm whose name another
    arm shares then carries its variant key's digest as well, so no two arms of one index print the
    same name — and a reader who sees the digest knows the levels shown do not tell the arms apart.

    Args:
        entries: The analysis's variant index.

    Returns:
        Variant key -> name, for every entry. The name carries no rig and no control marker: a cell
        adds its rig (:func:`arm_label`) and each table spells the control for itself.
    """
    distinguishing = distinguishing_axes(entries)
    names: dict[str, str] = {}
    for entry in entries:
        if entry.levels_unavailable is not None:
            reason = f" — {entry.levels_unavailable}" if entry.levels_unavailable else ""
            names[entry.variant_key] = f"levels unavailable ({short_digest(entry.variant_key)}){reason}"
            continue
        levels = naming_levels(entry, distinguishing)
        names[entry.variant_key] = (
            ", ".join(f"{level.axis_id}={elide_level(level.display)}" for level in levels) if levels else NO_LEVER_MOVED
        )
    for digest in (short_digest, lambda key: key):
        shared = {name for name in names.values() if list(names.values()).count(name) > 1}
        if not shared:
            break
        # Twelve characters of two different keys can agree; the whole key cannot.
        names = {key: f"{name} (arm {digest(key)})" if name in shared else name for key, name in names.items()}
    return names


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


def arm_label(variant_key: str, names: Mapping[str, str], *, rig: str | None = None) -> str:
    """Name one arm — or one cell of it — in the words every python surface prints.

    The arm's name from :func:`arm_names` when the variant index holds it. Otherwise a LABELLED
    digest: an arm printed as a bare hash reads as an arm with a strange name, when what the reader
    needs to know is that this analysis cannot say what it ran.

    The rig rides on the end when the caller says the arm was measured under more than one; the
    control marker does not, because each table spells its own.

    The browser does not spell these words at all: :class:`ArmRow` and
    :class:`~threetears.evals.analysis.surface_table.SurfaceRow` serve the result as ``label``, and the
    kit prints what it is served — a second spelling in a second language is a second name.

    Args:
        variant_key: The arm's variant coordinate.
        names: The analysis's :func:`arm_names`.
        rig: The rig's short digest, only when the arm spans more than one rig.

    Returns:
        The label.
    """
    if variant_key in names:
        name = names[variant_key]
    else:
        name = f"unplaced ({short_digest(variant_key)}) — not in this analysis's variant index"
    return f"{name} @ rig {rig}" if rig else name


def cell_label(
    variant_key: str,
    apparatus_class_id: str,
    *,
    index: Mapping[str, VariantIndexEntry],
    multi_rig: frozenset[str],
) -> str:
    """Name one cell — an arm under one rig — from the index and the multi-rig population.

    The composition every surface naming a cell makes: the arm's name over the whole index (or the
    fact that the index cannot place it), and the rig's short digest exactly when the arm was measured
    under more than one rig in the caller's population.

    Args:
        variant_key: The cell's variant coordinate.
        apparatus_class_id: The rig it was measured under.
        index: Variant key → index entry.
        multi_rig: The variants measured under more than one rig, as :func:`multi_rig_variants` returns them.

    Returns:
        The label, as :func:`arm_label` spells it.
    """
    return arm_label(
        variant_key,
        arm_names(list(index.values())),
        rig=short_digest(apparatus_class_id) if variant_key in multi_rig else None,
    )


def _status_and_why(
    decisions: Sequence[Decision], variant_key: str, *, is_control: bool, a_winner_exists: bool
) -> tuple[ArmStatus, list[str]]:
    """Where one arm stands, read off the decisions that name its cells.

    An adopted decision naming one of the arm's cells makes it the winner; a rejected one rules it
    out; a declared control is replaced when some other arm won. Everything else is unresolved —
    including an arm only a deferred decision names, since deferring is not a verdict.

    **An arm both adopted and rejected is contradicted, never the winner.** Generation refuses such a
    memo (:func:`contradicted_arms`), so only an analysis stored before that refusal carries one, and
    settling it by which disposition is looked at first would show a winner the memo also rules out.
    Its findings are those of the first adopting and the first rejecting decision, in that order.

    Returns:
        ``(status, finding_ids)`` — where the arm stands, and the positions of the findings the
        deciding decision rests on.
    """
    first: dict[str, Decision] = {}
    for decision in decisions:
        if decision.disposition in ("adopted", "rejected") and any(
            variant_of_cell_ref(cell) == variant_key for cell in decision.cells
        ):
            first.setdefault(decision.disposition, decision)
    if "adopted" in first and "rejected" in first:
        rests_on = [*first["adopted"].rests_on, *first["rejected"].rests_on]
        return "contradicted", [str(position) for position in dict.fromkeys(rests_on)]
    rulings: tuple[tuple[str, ArmStatus], ...] = (("adopted", "winner"), ("rejected", "ruled_out"))
    for disposition, status in rulings:
        if disposition in first:
            return status, [str(position) for position in first[disposition].rests_on]
    if is_control and a_winner_exists:
        return "replaced_incumbent", []
    return "unresolved", []


def contradicted_arms(decisions: Sequence[Decision]) -> list[tuple[str, int, int]]:
    """Every arm one decision adopts and another rejects — a memo contradicting itself.

    Structural: read off each decision's disposition and the variants of the cells it names
    (:func:`~threetears.evals.analysis.cells.variant_of_cell_ref`, as the arm table reads them), never
    its prose. A cell that names no variant names no arm, and a deferred decision is no verdict.

    Args:
        decisions: The analysis's decisions, in document order.

    Returns:
        ``(variant_key, adopting position, rejecting position)`` per contradicted arm — the first
        decision of each disposition naming it — in the order the arms are first adopted.
    """
    adopted: dict[str, int] = {}
    rejected: dict[str, int] = {}
    for position, decision in enumerate(decisions):
        if decision.disposition not in ("adopted", "rejected"):
            continue
        seen = adopted if decision.disposition == "adopted" else rejected
        for cell in decision.cells:
            variant = variant_of_cell_ref(cell)
            if variant is not None:
                seen.setdefault(variant, position)
    return [(variant, at, rejected[variant]) for variant, at in adopted.items() if variant in rejected]


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
    names = arm_names(variant_index)

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
                label=arm_label(key, names),
                levels=naming_levels(entry, distinguishing),
                settings=arm_settings(entry),
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
    "ELISION",
    "LABEL_LEVEL_CHARS",
    "NO_LEVER_MOVED",
    "ArmLevel",
    "ArmMeasurement",
    "ArmRow",
    "ArmStatus",
    "ArmTable",
    "applicable_levers",
    "arm_label",
    "arm_levels",
    "arm_levers",
    "arm_names",
    "arm_settings",
    "arm_table",
    "arm_table_of",
    "cell_label",
    "contradicted_arms",
    "distinguishing_axes",
    "elide_level",
    "multi_rig_variants",
    "naming_levels",
    "short_digest",
    "writer_arms",
]
