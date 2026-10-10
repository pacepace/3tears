"""The time axis: the campaign's runs placed in time, or why they cannot be.

Each position's cells are the decision surface's own algebra over that position's runs, grouped by the host's
release label or, without one, by UTC day (:func:`_time_positions`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from threetears.evals.kernel.evidence_tiers import JudgeEvidenceTier
from threetears.evals.analysis.cells import (
    ApparatusClass,
    Observation,
    pool_observations,
)
from threetears.evals.analysis.reporting import ScoreRecord
from threetears.evals.kernel.declaration import CampaignDesign
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.schema.models import EvalResult
from threetears.evals.kernel.surface import (
    TimeAxis,
    TimeAxisBasis,
    TimePosition,
)
from threetears.evals.analysis.bundle.cell_reads import (
    _cell_measures,
    _judged_measures,
    _results_by_cell,
)

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun


def _time_axis(
    runs: list[EvalRun],
    results_by_run: dict[str, list[EvalResult]],
    observations: list[Observation],
    classes: dict[str, ApparatusClass],
    records: list[ScoreRecord],
    design: CampaignDesign | None,
    *,
    tiers: list[JudgeEvidenceTier],
    short_runs: dict[str, str],
    incomplete_runs: dict[str, str],
    profile: HostProfile,
    names: Mapping[str, str],
) -> tuple[TimeAxis | None, str | None]:
    """Place the campaign's runs in time, or say why they cannot be.

    **Each position's cells are the decision surface's own algebra over that position's runs**: the
    observations re-pooled (:func:`~threetears.evals.analysis.cells.pool_observations`), the judged
    dimensions summarised (:func:`_judged_measures`) and the cells measured (:func:`_cell_measures`) by the
    same functions the whole surface is, so a cell's figure at one build and its figure over the campaign
    differ only in which observations they read. A run that measured nothing has no place in time — it
    cannot say what was measured when — and is left out of every position.

    Args:
        runs: The resolved member runs, in creation order.
        results_by_run: Each run's results.
        observations: Every observation the cell algebra pooled.
        classes: Apparatus class id → the class, as the campaign's pooling read them.
        records: The score projection, for judged dimensions.
        design: The campaign's declaration, for the bar a judged dimension carries.
        tiers: The judges' evidence tiers, which every judged reading carries — the campaign's own, since a
            judge's reliability is measured over the whole campaign, not one position of it.
        short_runs: The bundle's short-run sentences, by run id.
        incomplete_runs: The bundle's incomplete-run statuses, by run id.
        profile: The host, whose ``release_label`` names its builds.
        names: The bundle's arm names, which order each position's cells as the whole surface's are.

    Returns:
        ``(axis, None)`` when the measuring runs span two or more positions, else ``(None, why)``.
    """
    measuring = [run for run in runs if results_by_run[run.id]]
    if not measuring:
        return None, "no run produced an observation, so nothing was measured at any time"
    basis, grouped, release_why = _time_positions(measuring, results_by_run, profile=profile)
    if len(grouped) < 2:
        return None, f"every run started on one day ({grouped[0][0]}) and {release_why}"
    positions = []
    for key, members in grouped:
        member_ids = {run.id for run in members}
        slice_cells, _, _ = pool_observations([obs for obs in observations if obs.apparatus_ref in member_ids], classes)
        slice_results = [result for run in members for result in results_by_run[run.id]]
        result_ids = {result.id for result in slice_results}
        by_cell = _results_by_cell(slice_cells, slice_results)
        judged = _judged_measures(
            [record for record in records if record.result_id in result_ids], by_cell, design, tiers=tiers
        )
        positions.append(
            TimePosition(
                key=key,
                first_run_at=members[0].created_at,
                last_run_at=members[-1].created_at,
                run_ids=sorted(member_ids),
                # A position's cells are not broken down by stratum: the breakdown is read over the whole
                # campaign, and per position it would multiply the bundle by every stratum at every build.
                cells=_cell_measures(
                    slice_cells,
                    by_cell,
                    judged,
                    strata={},
                    short_runs=short_runs,
                    incomplete_runs=incomplete_runs,
                    profile=profile,
                    control=design.control if design else None,
                    names=names,
                ),
            )
        )
    if basis == "release":
        return TimeAxis(basis=basis, release_label=profile.release_label, positions=positions), None
    return TimeAxis(basis=basis, basis_reason=release_why, positions=positions), None


def _time_positions(
    runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
) -> tuple[TimeAxisBasis, list[tuple[str, list[EvalRun]]], str]:
    """Group the measuring runs into time positions, earliest first.

    By the host's release label when it declares one, every run recorded it and the runs span two values of
    it; otherwise by the UTC day each run was created on. Either way the positions are ordered by when their
    earliest run was created, which is the one order every run records — a label is the host's string and
    nothing here can sort it.

    Args:
        runs: The runs that measured something, in creation order.
        results_by_run: Each run's results, which a label reader may read.
        profile: The host, whose ``release_label`` names its builds.

    Returns:
        ``(basis, positions, why_not_builds)``: each position's key and its runs in creation order, and — on a
        ``date`` basis — why the positions are not builds, in the words of the branch that found it, naming
        the runs that recorded no label where that is the reason. Empty on a ``release`` basis.

    Raises:
        RuntimeError: The release label names no registered input — registration refuses that, so the
            profile was built around its own check.
    """
    release_why = "the host labels no build"
    if profile.release_label is not None:
        declared = profile.sweepables.get(profile.release_label)
        if declared is None:
            raise RuntimeError(f"release_label {profile.release_label!r} names no registered input")
        # Normalised ONCE, and every check below reads the normalised label: a position's key is stripped
        # where it is stored (the base stance), so two labels that differ only in whitespace — a version read
        # from a file with its trailing newline — are one build, and grouping them as two would hand the axis
        # two positions it then refuses as repeated.
        values = {run.id: _release_label(declared.read(run, results_by_run[run.id])) for run in runs}
        unrecorded = [run.id for run in runs if values[run.id] is None]
        if unrecorded:
            named = ", ".join(sorted(unrecorded))
            release_why = f"{len(unrecorded)} of {len(runs)} runs recorded no {profile.release_label} ({named})"
        elif len(set(values.values())) > 1:
            return "release", _group_in_order(runs, lambda run: values[run.id] or ""), ""
        else:
            release_why = f"every run recorded one {profile.release_label} ({next(iter(values.values()))})"
    return "date", _group_in_order(runs, _utc_day), release_why


def _release_label(value: object) -> str | None:
    """A run's release label as a time position keys it: stripped, and ``None`` when it recorded none or only blanks."""
    if value is None:
        return None
    return str(value).strip() or None


def _group_in_order(runs: list[EvalRun], key: Callable[[EvalRun], str]) -> list[tuple[str, list[EvalRun]]]:
    """Group runs by ``key``, each group in creation order and the groups ordered by their earliest run."""
    groups: dict[str, list[EvalRun]] = {}
    for run in sorted(runs, key=lambda r: (r.created_at, r.id)):
        groups.setdefault(key(run), []).append(run)
    return list(groups.items())


def _utc_day(run: EvalRun) -> str:
    """The UTC calendar day a run was created on, as YYYY-MM-DD."""
    created = datetime.fromisoformat(run.created_at)
    if created.tzinfo is not None:
        created = created.astimezone(UTC)
    return created.date().isoformat()


__all__: list[str] = []
