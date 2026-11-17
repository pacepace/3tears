"""Which arm and which cell each result belongs to.

:func:`variant_key_of_run` and :func:`_variant_key_of` read the variant key a run or result was stamped with,
:func:`observations_of` turns a campaign's results into the observations the cell algebra pools,
:func:`classify_apparatus` classifies each run's rig, :func:`measure_definition_levels` reads the definitions of
its host's measures each run's values were read under, and :func:`split_coordinates` says which coordinates of its
identity one arm's results, read together, are split on, for the lenses that read results outside a campaign's cells.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, get_args

from threetears.evals.analysis.cells import (
    MEASURE_DEFINITIONS_DIMENSION,
    NO_MEASURE_DEFINITIONS,
    ApparatusClass,
    Observation,
    apparatus_class_of,
    pool_observations,
    split_on_served_model,
)
from threetears.evals.analysis.reporting import (
    attested_served_models,
    served_level,
    served_reading,
    through_attestation,
)
from threetears.evals.kernel.campaign import VariantIndexEntry
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.schema.values import SweepableValue
from threetears.evals.kernel.identity import IDENTITY_VERSION, resolve_variant_identity
from threetears.evals.kernel.metrics import holds_host_measure
from threetears.evals.kernel.result_reading import ResultView
from threetears.evals.kernel.scoring import definitions_unrecorded, remeasure_unfinished
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.reading_states import Coordinate

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun


def classify_apparatus(
    runs: list[EvalRun],
    apparatus_levels: dict[str, dict[str, str | None]],
) -> dict[str, ApparatusClass]:
    """Classify each run's rig, so every observation it carries shares one class.

    A launching host declares its apparatus at launch, so a batch's observations were all measured
    under the same rig and reading it per run is exact rather than an approximation. A host
    whose apparatus genuinely moves within a batch supplies it per observation instead — the
    :class:`~threetears.evals.analysis.cells.Observation` carries the coordinate, and nothing
    downstream can tell which route produced it.

    :param runs: The campaign's resolved runs.
    :param apparatus_levels: Dimension → run id → level key, from :func:`read_apparatus_levels`. A
        ``None`` level means the run never recorded that dimension.
    :return: Run id → the class that run's observations belong to.
    """
    dimensions = set(apparatus_levels)
    definitions = apparatus_levels.get(MEASURE_DEFINITIONS_DIMENSION, {})
    return {
        run.id: apparatus_class_of(
            {dim: apparatus_levels.get(dim, {}).get(run.id) for dim in dimensions},
            dimensions=dimensions,
            # Measure definitions nothing recorded match nothing, not even another run's silence: the class is
            # alone to the run, where a host's own unrecorded dimension pools with the same silence.
            alone=run.id if run.id in definitions and definitions[run.id] is None else None,
            # Read off the run, never assumed: the launch path stamps `commissioned`, and a host
            # capturing traffic it did not control writes `witnessed`. It enters the class id, so a
            # captured session beside a launched arm of the same variant is two cells everywhere.
            provenance=run.apparatus_provenance,
        )
        for run in runs
    }


def measure_definition_levels(
    runs: Sequence[EvalRun], results_by_run: dict[str, list[ResultView]]
) -> dict[str, str | None]:
    """The definitions of its host's measures each run's values were read under, as the rig's level of
    :data:`~threetears.evals.analysis.cells.MEASURE_DEFINITIONS_DIMENSION`, where the runs do not all say one thing.

    Read as pass^k reads it (:func:`~threetears.evals.kernel.scoring.pass_hat_k_cell`), off what each run RECORDED
    (``EvalRun.measure_definitions``: its launch's, a re-measure's or its host's attestation) and what its results
    hold, never off the host's registry as it is today. The assembly adds the answer to the runs' apparatus levels,
    so the cells, the confound scans, the coverage rows, the divergences and the factor aliasing all read it as they
    read any dimension of the rig.

    :param runs: The campaign's resolved runs.
    :param results_by_run: Each run's results, keyed by run id.
    :return: Run id → one digest of the run's recorded map, in any key order;
        :data:`~threetears.evals.analysis.cells.NO_MEASURE_DEFINITIONS` where it has none to record (an empty map,
        or none recorded and no host measure held); ``None`` where nothing says which definition its values were
        read under: its results hold a host measure and it recorded none, or a re-measure of it began and did not
        finish. EMPTY where every run says one known thing: the dimension separates nothing, so it is no dimension
        of this campaign's rig and no class id moves.
    """
    levels: dict[str, str | None] = {}
    for run in runs:
        holds = any(holds_host_measure(result.host_measures) for result in results_by_run.get(run.id, []))
        if remeasure_unfinished(run) or definitions_unrecorded(run, holds_host_measures=holds):
            levels[run.id] = None
        elif run.measure_definitions:
            levels[run.id] = canonical_digest(run.measure_definitions)
        else:
            levels[run.id] = NO_MEASURE_DEFINITIONS
    said = set(levels.values())
    # One silent run beside nothing else separates nothing either: there is no other run to pool or contrast it with.
    return levels if len(said) > 1 or (None in said and len(levels) > 1) else {}


def split_coordinates(runs: Sequence[EvalRun], results: Sequence[ResultView]) -> list[Coordinate]:
    """The coordinates of its identity one arm's results, read together, are split on, as a campaign's cells split
    them: what a lens that pools them into one figure, or reads a step between them, asks first.

    Read through the cell algebra itself, never beside it: each run's rig is classified on the definitions of its
    host's measures alone (:func:`measure_definition_levels`, :func:`classify_apparatus`), split on the model that
    answered (:func:`~threetears.evals.analysis.cells.split_on_served_model`, a silence its host stated read through
    the statement) and pooled (:func:`~threetears.evals.analysis.cells.pool_observations`), and the answer is its
    cells' ``Cell.split_on``. So an unrecorded coordinate matches nothing, not even another run's silence, exactly
    as it does in a campaign.

    :param runs: The runs the results belong to; others are ignored.
    :param results: One arm's results: a contestant's, in one series point, one step or one pivot cell.
    :return: The coordinates split on, in the order :data:`~threetears.evals.schema.reading_states.Coordinate`
        lists them; empty where the results are one cell on every coordinate the package reads.
    """
    results_by_run: dict[str, list[ResultView]] = {}
    for result in results:
        results_by_run.setdefault(result.eval_run_id, []).append(result)
    held = [run for run in runs if run.id in results_by_run]
    levels = measure_definition_levels(held, results_by_run)
    classes = classify_apparatus(held, {MEASURE_DEFINITIONS_DIMENSION: levels} if levels else {})
    attested = attested_served_models(held)
    observations = [
        Observation(
            id=result.id,
            scope_id=run.scope_id,
            variant_key="arm",
            apparatus_class_id=classes[run.id].apparatus_class_id,
            apparatus_ref=run.id,
        )
        for run in held
        for result in results_by_run[run.id]
    ]
    served = {
        result.id: served_level(through_attestation(reading, attested.get(result.id)))
        for result in results
        if (reading := served_reading(result)) is not None
    }
    placed, by_class = split_on_served_model(
        observations, {klass.apparatus_class_id: klass for klass in classes.values()}, served
    )
    cells, _refused, _next = pool_observations(placed, by_class)
    return [coordinate for coordinate in get_args(Coordinate) if any(coordinate in cell.split_on for cell in cells)]


def observations_of(
    runs: list[EvalRun],
    results_by_run: dict[str, list[ResultView]],
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

    :param runs: The campaign's resolved runs, in bundle order.
    :param results_by_run: Each run's results, keyed by run id.
    :param classes: Run id → apparatus class, from :func:`classify_apparatus`.
    :param scope_id: The scope these were read under.
    :param profile: The host whose vocabulary this reads.
    :return: ``(observations, variant_index)`` — the observations in ``(run, result id)`` order, and
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
                # consumer of the index — the arm table, the writer's input, the coverage gate —
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
    result: ResultView, run: EvalRun, *, profile: HostProfile
) -> tuple[str, dict[str, SweepableValue], str | None]:
    """This result's variant key, and the lever map that key was digested from.

    The key is the one the runner STAMPED on the result. The map behind it comes from the run's RECORDED
    pre-image when the run carries one, and only from a fresh derivation for a run its host
    assembled without the launch — see
    :func:`~threetears.evals.kernel.identity.resolve_variant_identity`, which owns that choice. The
    difference is the whole of this function's behaviour across an
    :data:`~threetears.evals.kernel.identity.IDENTITY_VERSION` bump: a recorded map needs no predicate to
    read it, so the arm stays described, while a derivation replays today's predicate and cannot.

    :param result: The observation.
    :param run: The batch it belongs to.
    :param profile: The host whose vocabulary this reads.
    :return: ``(key, levers, unavailable)``. ``unavailable`` is the reason ``levers`` is empty, or ``None`` when it is
        empty because the host genuinely resolved no levers. **An empty map has two causes and they are not the same
        fact** — a host that registers no levers describes its arms exactly, with nothing; a key
        nothing here can place describes its arm not at all. Deciding that here, where
        ``reproduces`` is already known, is what stops each caller re-deriving it from the shape of
        the map and getting it wrong in its own way. :func:`_undescribable_arm_reason` says which
        of the two undescribable cases it is, in the numbers that tell an operator what to do.
    :raises LeverCoordinateError: The host's variant map disagrees with its own registry.
        Reachable only on the DERIVED path — a run carrying its own pre-image is never
        checked against today's registry, because the registry says what a lever is now and
        the map records what one was. On that path it is loud: a campaign over a mis-registered
        host fails assembly rather than assembling with no coordinates, and the same registry
        would refuse every new observation anyway.

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


def _undescribable_arm_reason(result: ResultView, run: EvalRun) -> str:
    """Say why this arm's stamped key has no lever map behind it, in numbers an operator can act on.

    The two causes need different actions and the text is the only channel that carries the
    difference out to the arm table and to the pre-generation refusal that reads it. A run that
    recorded a map which does not digest to this key is a broken record; a run that recorded none
    (its host assembled it without the launch) is being read by a predicate other than the one it
    ran under, and re-running the campaign on this build fixes it.

    The version NUMBERS are what make the second actionable. "A predicate this build does not
    reproduce" is true of a run measured this morning across one bump and of a run from a year
    ago, and the remedy differs entirely.

    :param result: The observation whose stamped key cannot be described.
    :param run: The batch it belongs to.
    :return: The reason, for :attr:`~threetears.evals.kernel.campaign.VariantIndexEntry.levels_unavailable`.
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


def variant_key_of_run(results: Sequence[ResultView]) -> str | None:
    """Which variant a run's observations carried — the authoring side of the control.

    A control is a variant key and nobody types a sha256, so the authoring surface points at a
    run and this addresses it. A run is one arm, so it names one variant.

    :param results: One run's results, which carry the key the runner stamped on each.
    :return: The variant key, or ``None`` for a run with no results yet — a run with no observation
        is in no arm.

    **The key is taken from the first result.** Every result of a run is stamped from the one
    lever map the run carries, so two results differ in their stamped key only if they were
    written under different ``IDENTITY_VERSION``s, which a single runner pass cannot produce.
    Stated rather than defended against, because the check would cost every bundle assembly a
    comparison for a state no writer can reach — if one ever can (a backfill that re-stamps part
    of a run, say), this is where it would go unnoticed.
    """
    return results[0].variant_key if results else None


__all__ = [
    "classify_apparatus",
    "observations_of",
    "split_coordinates",
    "variant_key_of_run",
]
