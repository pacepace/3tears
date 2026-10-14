"""How an arm is named — the one construction every table of a report, and the writer's view, reads.

An arm is keyed by a *variant*, a digest over the whole resolved contestant stack, and nothing turns a digest
back into levers; :attr:`~threetears.evals.kernel.campaign.EvalAnalysis.variant_index` carries the map each
digest was computed from, and everything here reads coordinates through it. An arm's name is the levers on which
the campaign's arms differ, never a status: no arm is ever called a winner, and where an arm stands is the code
verdicts' to say (:mod:`threetears.evals.analysis.report`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from pydantic import Field

from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.kernel.authored import Decision
from threetears.evals.kernel.campaign import EvalAnalysis, VariantIndexEntry
from threetears.evals.kernel.host.sweepables import CANDIDATE_KIND_LEVER, CANDIDATE_MODEL_LEVER
from threetears.evals.schema.values import SweepableValue
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.kernel.surface import CellFacts


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


# --- How an arm is named — the one construction every table and chart reads ---------------------
#
# Every table of a report names arms, and the writer's view does too, and a reader who meets one arm under two
# names on one page reads two arms. So the facts a name is
# built from — which levels, in what order; how a long one is cut; whether the arm spans several rigs;
# how much of a digest tells two apart — are derived here once, and every one of those surfaces calls
# these. The rule, in full:
#
# - An arm is named by the levers on which the report's arms DIFFER (:func:`distinguishing_axes`), and
#   by nothing else: a lever every arm carried at one level names none of them.
# - A lever that does not apply to the arm's kind (its level is
#   :meth:`~threetears.evals.schema.values.SweepableValue.not_this_kind`'s) is never named.
# - An arm that differs on nothing it carries is named by its candidate kind and model.
# - Each level is put on one line and cut in the middle past :data:`LABEL_LEVEL_CHARS` (:func:`elide_level`).
# - Two arms the above would name alike are told apart by their variant keys' digests (:func:`arm_names`).
# - A cell names its rig too, exactly when its arm was measured under more than one.

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

    Read through :attr:`~threetears.evals.kernel.campaign.VariantIndexEntry.named_levers`, so an arm is
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
    :meth:`~threetears.evals.schema.values.SweepableValue.not_this_kind` level, because that is
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

    A browser does not spell these words at all: it reads the report the package renders, so there is one
    spelling.

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


def surface_order(cells: Iterable[CellFacts], *, control: str | None, names: Mapping[str, str]) -> list[CellFacts]:
    """The cells in the decision surface's one row order — the reference, then every other arm by name.

    The control's cells come first because every other arm is read against them: they are the
    reference, never the pick. The other arms follow in alphabetical order of the names a reader sees
    (:func:`arm_label` over ``names``, case-folded), each arm's rigs by id, the variant key breaking a
    tie. Alphabetical because it is checkable on the page and says nothing about the evidence: an order
    by a raw score would read as a ranking the comparisons may not support, and an order by digest
    would be one nobody can state. Every surface that lays out the decision surface — the report's
    strata table and the writer's ``cell_measures`` — orders its cells here, and the table
    states the rule (:data:`~threetears.evals.analysis.report.figures.SURFACE_ORDER`).

    Args:
        cells: The cells to order.
        control: The declared control's variant key, or None.
        names: The analysis's :func:`arm_names`.

    Returns:
        The cells, in that order.
    """

    def key(cell: CellFacts) -> tuple[bool, str, str, str, str]:
        label = arm_label(cell.variant_key, names)
        return (cell.variant_key != control, label.casefold(), label, cell.variant_key, cell.apparatus_class_id)

    return sorted(cells, key=key)


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


def contradicted_arms(decisions: Sequence[Decision]) -> list[tuple[str, int, int]]:
    """Every arm one decision adopts and another rejects — a memo contradicting itself.

    Structural: read off each decision's disposition and the variants of the cells it names
    (:func:`~threetears.evals.analysis.cells.variant_of_cell_ref`), never
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


__all__ = [
    "DIGEST_CHARS",
    "ELISION",
    "LABEL_LEVEL_CHARS",
    "NO_LEVER_MOVED",
    "ArmLevel",
    "applicable_levers",
    "arm_label",
    "arm_levels",
    "arm_levers",
    "arm_names",
    "cell_label",
    "contradicted_arms",
    "distinguishing_axes",
    "elide_level",
    "multi_rig_variants",
    "naming_levels",
    "short_digest",
    "surface_order",
    "writer_arms",
]
