"""The cell notes a reader must see beside the numbers: short cells, unmeasured spend, contended latency, failed arms.

:func:`_short_cells` names every cell holding fewer repetitions than declared, :func:`_cost_unmeasured` and
:func:`_latency_contended` name the cells whose spend or latency cannot be read, :func:`_all_failed` the arms
whose every result failed, and :func:`_held_fixed_reading` what the campaign held fixed.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from threetears.evals.analysis.contention import contended_latency_sentence
from threetears.evals.analysis.cells import Cell
from threetears.evals.kernel.declaration import CampaignDesign
from threetears.evals.schema.models import EvalResult
from threetears.evals.kernel.result_condition import delivered_a_turn
from threetears.evals.kernel.surface import (
    CellFacts,
    all_failed_sentence,
)
from threetears.evals.kernel.usage_capture import spend_observed
from threetears.evals.analysis.bundle.schema import (
    CellCoordinate,
    HeldFixedReading,
    ShortCell,
)
from threetears.evals.analysis.bundle.cell_reads import _CellKey

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun


def _short_cells(cells: list[Cell], design: CampaignDesign | None) -> list[ShortCell]:
    """Name every cell holding fewer repetitions than the declaration intended.

    A cell is counted by its least-repeated case, since that case is the cell's weakest replication
    and pooling more of the others does not repair it.

    Args:
        cells: The pooled cells, in coordinate order.
        design: The campaign's declaration, or None.

    Returns:
        One entry per short cell, in coordinate order. Empty when nothing was declared — an unstated
        intention cannot be fallen short of — or when every cell met it. A cell with no case count
        has no repetitions to compare and is not listed: its own facts already say its cases were
        unrecorded, which is the honest answer (unknown, not short). The bundle's observations all
        name their case, since ``EvalResult.test_case_id`` is required.
    """
    intended = design.intended_repetitions if design is not None else None
    if intended is None:
        return []
    short = []
    for cell in sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id)):
        observed = cell.repeats_per_case_min
        if observed is None or observed >= intended:
            continue
        short.append(
            ShortCell(
                variant_key=cell.variant_key,
                apparatus_class_id=cell.apparatus_class_id,
                intended=intended,
                observed=observed,
                # Names no coordinate: the entry carries the cell's, and the writer's view renames those
                # to the cell's alias, which a digest spelled into prose would bypass.
                sentence=(
                    f"This cell ran its least-repeated case {observed} times against the {intended} repetitions "
                    "the campaign declared it intends per case in each cell, so its estimates rest on less "
                    "replication than the design set out to buy."
                ),
            )
        )
    return short


#: How a reader learns to report spend from the one candidate the engine cannot see into.
_HOW_TO_REPORT_SPEND = "A quick candidate reports its spend by returning an Answer."


def _cost_unmeasured(results_by_cell: dict[_CellKey, list[EvalResult]]) -> tuple[list[CellCoordinate], str | None]:
    """Name every cell where no result observed spend, with the one sentence that says what that means.

    A cell is listed when it holds a result storing a ``cost_usd`` and no result in it observed spend
    (:func:`~threetears.evals.kernel.usage_capture.spend_observed`): every number it stores is the sum of
    nothing, so the measure walk read none of them and the cell carries no ``cost_usd`` reading. A cell whose
    every result went unpriced is not listed — its cost is unknown for a reason ``cost_usd`` null already
    states — and neither is one where any result observed spend, whose own ``n`` discloses the rest. Read
    over the turns the candidate took (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`),
    the population a cell's ``cost_usd`` reading is read over (``delivered``), so a billed refusal cannot
    make a cell whose turns reported no spend look measured. A cell where no result took a turn is not
    listed: it has no cost because every call failed, which :func:`_all_failed` says, and "nobody reported
    spend" would be the wrong reason.

    Args:
        results_by_cell: Each cell's results, from :func:`_results_by_cell`.

    Returns:
        The unmeasured cells in coordinate order, and the sentence — None when there are none.
    """
    unmeasured: list[CellCoordinate] = []
    for (variant_key, apparatus_class_id), members in sorted(results_by_cell.items()):
        counted = [result for result in members if delivered_a_turn(result)]
        stores_a_cost = any(result.cost_usd is not None for result in counted)
        if stores_a_cost and not any(spend_observed(result.usage, result.cost_roles) for result in counted):
            unmeasured.append(CellCoordinate(variant_key=variant_key, apparatus_class_id=apparatus_class_id))
    if not unmeasured:
        return [], None
    if len(unmeasured) == len(results_by_cell):
        sentence = (
            "Cost was not measured: no result reported its spend, so the $0 each one stores is not a measurement "
            "and cost is neither charted nor tested."
        )
    else:
        sentence = (
            f"Cost was not measured in {len(unmeasured)} of {len(results_by_cell)} cells: no result there reported "
            "its spend, so the $0 those results store is not a measurement and cost is charted and tested only "
            "where it was measured."
        )
    return unmeasured, f"{sentence} {_HOW_TO_REPORT_SPEND}"


def _latency_contended(
    results_by_cell: dict[_CellKey, list[EvalResult]], contended_ids: Collection[str], *, declared: bool = False
) -> tuple[list[CellCoordinate], str | None]:
    """Name every cell whose latency read under concurrency was left out, with the one sentence that says so.

    Args:
        results_by_cell: Each cell's results, from :func:`_results_by_cell`.
        contended_ids: The results whose latency :func:`~threetears.evals.analysis.contention.withhold_contended_latency`
            removed before anything read them.
        declared: The campaign's design declares latency under test, so a run read under concurrency is one it
            cannot read its question from — said, with the remedy, rather than left to a reader to infer.

    Returns:
        The cells in coordinate order, and the sentence — None when nothing was left out.
    """
    cells: list[CellCoordinate] = []
    withheld = 0
    total = 0
    for (variant_key, apparatus_class_id), members in sorted(results_by_cell.items()):
        total += len(members)
        here = sum(1 for result in members if result.id in contended_ids)
        if here:
            withheld += here
            cells.append(CellCoordinate(variant_key=variant_key, apparatus_class_id=apparatus_class_id))
    where = "" if len(cells) == len(results_by_cell) else f" in {len(cells)} of {len(results_by_cell)} cells"
    return cells, contended_latency_sentence(withheld, total, where=where, declared=declared)


def _all_failed(cells: list[CellFacts]) -> tuple[list[CellCoordinate], str | None]:
    """Name every cell where no counted result took a turn, with the one sentence that says so.

    Read off the cells' own counts (:attr:`~threetears.evals.kernel.surface.CellFacts.all_failed`), so the
    list and the surface cannot disagree about which cell took no turn. Such a cell has no cost or latency
    reading, and without this the absence reads as "not measured" beside the arms that were.

    Args:
        cells: The decision surface's cells, from :func:`_cell_measures`.

    Returns:
        The cells in coordinate order, and the sentence
        (:func:`~threetears.evals.kernel.surface.all_failed_sentence`) — None when there are none.
    """
    failed = [
        CellCoordinate(variant_key=cell.variant_key, apparatus_class_id=cell.apparatus_class_id)
        for cell in sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id))
        if cell.all_failed
    ]
    if not failed:
        return [], None
    return failed, all_failed_sentence(len(failed), len(cells))


def _held_fixed_reading(runs: list[EvalRun], design: CampaignDesign | None) -> HeldFixedReading:
    """Compare what the campaign declared held fixed with the provenance every resolved run recorded.

    Args:
        runs: The resolved member runs.
        design: The campaign's declaration, or None.

    Returns:
        The reading. Its disclosure is composed from the branches actually taken: a declared apparatus
        the runs contradict, a mix of provenances nobody declared, and an uncontrolled stimulus each
        add their own sentence, and none of them adds one that did not happen.
    """
    provenance = {run.id: run.apparatus_provenance for run in sorted(runs, key=lambda r: r.id)}
    held_fixed = design.held_fixed if design is not None else None
    declared = held_fixed.apparatus if held_fixed is not None else None
    contradicting = sorted(run_id for run_id, found in provenance.items() if declared is not None and found != declared)
    sentences = []
    if contradicting:
        sentences.append(
            f"This campaign declares its apparatus {declared}, but {len(contradicting)} of its {len(provenance)} "
            f"resolved runs recorded otherwise ({', '.join(contradicting)}); their observations sit in cells of "
            "their own, and a finding drawn from them is drawn from a different kind of evidence than the "
            "declaration describes."
        )
    elif declared is None and len(set(provenance.values())) > 1:
        sentences.append(
            "This campaign declares nothing held fixed, and its runs mix commissioned and witnessed apparatus; the two "
            "never share a cell, so an arm measured both ways is reported as two cells."
        )
    if held_fixed is not None and held_fixed.stimulus == "uncontrolled":
        sentences.append(f"The stimulus was not held fixed: {held_fixed.stimulus_reason.strip()}")
    return HeldFixedReading(
        declared_stimulus=held_fixed.stimulus if held_fixed is not None else None,
        stimulus_reason=held_fixed.stimulus_reason if held_fixed is not None else "",
        declared_apparatus=declared,
        run_provenance=provenance,
        contradicting_run_ids=contradicting,
        disclosure=" ".join(sentences) or None,
    )


__all__: list[str] = []
