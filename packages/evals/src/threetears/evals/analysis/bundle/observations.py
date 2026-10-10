"""Which arm and which cell each result belongs to.

:func:`variant_key_of_run` and :func:`_variant_key_of` read the variant key a run or result was stamped with,
:func:`_observations` turns a campaign's results into the observations the cell algebra pools, and
:func:`_apparatus_classes` groups the apparatus a cell id is neutral to (:data:`CELL_ID_NEUTRAL`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING


from threetears.evals.analysis.cells import (
    ApparatusClass,
    Observation,
    apparatus_class_of,
)
from threetears.evals.kernel.campaign import VariantIndexEntry
from threetears.evals.schema.hashing import canonical_json
from threetears.evals.kernel.host.profile import UNSEATED_LEVEL, HostProfile
from threetears.evals.schema.values import SweepableValue
from threetears.evals.kernel.identity import IDENTITY_VERSION, resolve_variant_identity

# At runtime for its field set, which tells a result-level measure from a row-level one.
from threetears.evals.schema.models import EvalResult


if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun


#: Apparatus dimensions that joined the rig after cells were minted under ids that never digested them, each
#: mapped to the dimension whose seat it shares. Such a dimension stays out of a class's id at the levels that
#: say nothing about it the class does not already say: UNRECORDED (``None``) — every run stored before the
#: dimension existed — or the unseated level where its owner reads unseated too. At any recorded level it is
#: digested like every other dimension. So a stored run's cell keeps the id a stored analysis cites, and a run
#: that recorded the dimension gets a cell of its own, which never pools with the unrecorded one: the class
#: still lists the dimension (``unknown_dimensions``), so the merge rule refuses the pair, and the confound scan
#: reads it ``undecided``.
#:
#: **Why no two different classes can share an id.** Within one bundle every class is built over one dimension
#: set, so a class's unknown set is fixed by its recorded map, and two classes the id cannot tell apart differ
#: only in this dimension's level, which is neutral in both. Unrecorded beside unrecorded is the same class.
#: Unrecorded beside unseated cannot happen with the owner agreeing: unseated here needs the owner unseated
#: (the condition below), while unrecorded here means the run filled the seat, so its owner reads a recorded or
#: an unrecorded level, never unseated — the owner's own level tells the two classes apart. A dimension's
#: unseated level paired with a recorded owner (a run that filled no judge seat yet recorded a judge, which
#: :meth:`~threetears.evals.kernel.host.profile.HostProfile.omits_apparatus` reports as a contradiction) is
#: therefore digested, not neutral.
CELL_ID_NEUTRAL: Mapping[str, str] = {"judge_temperature": "judge_model"}


def _cell_id_neutral(run_id: str, apparatus_levels: dict[str, dict[str, str | None]]) -> frozenset[str]:
    """The :data:`CELL_ID_NEUTRAL` dimensions this run's class id leaves out, at the levels where it says nothing new.

    Args:
        run_id: The run.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.

    Returns:
        The dimensions to leave out of the run's class id; empty when every one is recorded, or absent from the
        bundle's apparatus altogether.
    """
    unseated = canonical_json(UNSEATED_LEVEL)
    neutral: set[str] = set()
    for dimension, owner in CELL_ID_NEUTRAL.items():
        if dimension not in apparatus_levels:
            continue
        level = apparatus_levels[dimension].get(run_id)
        if level is None or (level == unseated and apparatus_levels.get(owner, {}).get(run_id) == unseated):
            neutral.add(dimension)
    return frozenset(neutral)


def _apparatus_classes(
    runs: list[EvalRun],
    apparatus_levels: dict[str, dict[str, str | None]],
) -> dict[str, ApparatusClass]:
    """Classify each run's rig, so every observation it carries shares one class.

    A launching host declares its apparatus at launch, so a batch's observations were all measured
    under the same rig and reading it per run is exact rather than an approximation. A host
    whose apparatus genuinely moves within a batch supplies it per observation instead — the
    :class:`~threetears.evals.analysis.cells.Observation` carries the coordinate, and nothing
    downstream can tell which route produced it.

    Args:
        runs: The campaign's resolved runs.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`. A
            ``None`` level means the run never recorded that dimension.

    Returns:
        Run id → the class that run's observations belong to.
    """
    dimensions = set(apparatus_levels)
    return {
        run.id: apparatus_class_of(
            {dim: apparatus_levels.get(dim, {}).get(run.id) for dim in dimensions},
            dimensions=dimensions,
            id_neutral=_cell_id_neutral(run.id, apparatus_levels),
            # Read off the run, never assumed: the launch path stamps `commissioned`, and a host
            # capturing traffic it did not control writes `witnessed`. It enters the class id, so a
            # captured session beside a launched arm of the same variant is two cells everywhere.
            provenance=run.apparatus_provenance,
        )
        for run in runs
    }


def _observations(
    runs: list[EvalRun],
    results_by_run: dict[str, list[EvalResult]],
    classes: dict[str, ApparatusClass],
    *,
    scope_id: str,
    profile: HostProfile,
) -> tuple[list[Observation], list[VariantIndexEntry]]:
    """Turn this campaign's results into the observations the cell algebra pools.

    One observation per result, because that is where the variant resolves: a run carries
    several candidate models and the candidate is part of what the variant IS, so a run-level
    observation would pool contestants that never competed under one key.

    The key is the one the runner stamped on the result — every result carries one, since the
    engine resolves every run's candidate model and kind into its variant map.

    Args:
        runs: The campaign's resolved runs, in bundle order.
        results_by_run: Each run's results, keyed by run id.
        classes: Run id → apparatus class, from :func:`_apparatus_classes`.
        scope_id: The scope these were read under.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(observations, variant_index)`` — the observations in ``(run, result id)`` order, and
        one index entry per variant, carrying the lever map its key was
        digested from. The index is built HERE rather than beside the cells because this is the
        one place that holds both halves at once: a cell knows its variant key and nothing
        about the run and candidate model the levels came from.
    """
    observations = []
    index: dict[str, VariantIndexEntry] = {}
    for run in runs:
        for result in results_by_run.get(run.id, []):
            key, levers, unavailable = _variant_key_of(result, run, profile=profile)
            if key not in index:
                # An arm whose levels cannot be described is indexed anyway, saying why. It was
                # measured and it pools; what nothing today can supply is a DESCRIPTION of it.
                # Leaving it out of the index instead is what made such an arm vanish: every
                # consumer of the index — the arm table, the generator prompt, the coverage gate —
                # reads arms from here, so an unindexed arm is one no surface can report and no
                # reader can miss.
                index[key] = VariantIndexEntry(variant_key=key, levers=levers, levels_unavailable=unavailable)
            observations.append(
                Observation(
                    id=result.id,
                    scope_id=scope_id,
                    variant_key=key,
                    apparatus_class_id=classes[run.id].apparatus_class_id,
                    # The batch this was commissioned under. Provenance for a reader, never an
                    # input to the cell — two observations from different batches with identical
                    # apparatus are one cell, which is the whole point of pooling across them.
                    apparatus_ref=run.id,
                    case_ref=result.test_case_id,
                )
            )
    return observations, sorted(index.values(), key=lambda entry: entry.variant_key)


def _variant_key_of(
    result: EvalResult, run: EvalRun, *, profile: HostProfile
) -> tuple[str, dict[str, SweepableValue], str | None]:
    """This result's variant key, and the lever map that key was digested from.

    The key is the one the runner STAMPED on the result. The map behind it comes from the run's RECORDED
    pre-image when the run carries one, and only from a fresh derivation for a run its host
    assembled without the launch — see
    :func:`~threetears.evals.kernel.identity.resolve_variant_identity`, which owns that choice. The
    difference is the whole of this function's behaviour across an
    :data:`~threetears.evals.kernel.identity.IDENTITY_VERSION` bump: a recorded map needs no predicate to
    read it, so the arm stays described, while a derivation replays today's predicate and cannot.

    Args:
        result: The observation.
        run: The batch it belongs to.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(key, levers, unavailable)``. ``unavailable`` is the reason ``levers`` is empty, or ``None`` when it is empty because the
        host genuinely resolved no levers. **An empty map has two causes and they are not the same
        fact** — a host that registers no levers describes its arms exactly, with nothing; a key
        nothing here can place describes its arm not at all. Deciding that here, where
        ``reproduces`` is already known, is what stops each caller re-deriving it from the shape of
        the map and getting it wrong in its own way. :func:`_undescribable_arm_reason` says which
        of the two undescribable cases it is, in the numbers that tell an operator what to do.

    Raises:
        LeverCoordinateError: The host's variant map disagrees with its own registry.
            Reachable only on the DERIVED path — a run carrying its own pre-image is never
            checked against today's registry, because the registry says what a lever is now and
            the map records what one was. On that path it is loud: a campaign over a mis-registered
            host fails assembly rather than assembling with no coordinates, and the same registry
            would refuse every new observation anyway.

    Note:
        ``levers`` is the map the returned key was digested from, and is EMPTY with
        ``unavailable`` set for a stamped key nothing here can place — a result from a run that
        recorded no pre-image, read by a build whose predicate no longer reproduces the key. Its
        stored key stays authoritative for pooling (that is what it was measured under), and
        today's levels are not a description of it: indexing them would label the arm with a
        stack it never ran. The arm is still INDEXED, carrying that reason instead of levels — an
        arm that pooled and then appeared in no index was one the arm table, the generator prompt
        and the coverage gate alike could not see, which is a silence rather than a disclosure.
    """
    resolved = resolve_variant_identity(run=run, profile=profile)
    if resolved.variant_key == result.variant_key:
        return result.variant_key, resolved.levers, None
    return result.variant_key, {}, _undescribable_arm_reason(result, run)


def _undescribable_arm_reason(result: EvalResult, run: EvalRun) -> str:
    """Say why this arm's stamped key has no lever map behind it, in numbers an operator can act on.

    The two causes need different actions and the text is the only channel that carries the
    difference out to the arm table and to the pre-generation refusal that reads it. A run that
    recorded a map which does not digest to this key is a broken record; a run that recorded none
    (its host assembled it without the launch) is being read by a predicate other than the one it
    ran under, and re-running the campaign on this build fixes it.

    The version NUMBERS are what make the second actionable. "A predicate this build does not
    reproduce" is true of a run measured this morning across one bump and of a run from a year
    ago, and the remedy differs entirely.

    Args:
        result: The observation whose stamped key cannot be described.
        run: The batch it belongs to.

    Returns:
        The reason, for :attr:`~threetears.evals.kernel.campaign.VariantIndexEntry.levels_unavailable`.
    """
    if run.variant_levers is not None:
        return (
            "the run recorded a lever map for this candidate and it does not digest to the key stamped on this "
            "observation, so it is not this arm's pre-image and describing the arm with it would label it with a "
            "stack it never ran"
        )
    return (
        f"the key was stamped under identity predicate v{result.identity_version}, this build derives "
        f"v{IDENTITY_VERSION}, which does "
        f"not reproduce it, and the run recorded no lever map for this candidate — so nothing here can say what the "
        f"arm ran"
    )


def variant_key_of_run(results: Sequence[EvalResult]) -> str | None:
    """Which variant a run's observations carried — the authoring side of the control.

    A control is a variant key and nobody types a sha256, so the authoring surface points at a
    run and this addresses it. A run is one arm, so it names one variant.

    Args:
        results: One run's results, which carry the key the runner stamped on each.

    Returns:
        The variant key, or ``None`` for a run with no results yet — a run with no observation
        is in no arm.

    Note:
        **The key is taken from the first result.** Every result of a run is stamped from the one
        lever map the run carries, so two results differ in their stamped key only if they were
        written under different ``IDENTITY_VERSION``s, which a single runner pass cannot produce.
        Stated rather than defended against, because the check would cost every bundle assembly a
        comparison for a state no writer can reach — if one ever can (a backfill that re-stamps part
        of a run, say), this is where it would go unnoticed.
    """
    return results[0].variant_key if results else None


__all__: list[str] = []
