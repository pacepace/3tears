"""A declared design can mark cells as deliberately not run, apart from cells that went missing (#654).

A design lists levels per axis; a cell is one level of every axis. ``crossing`` (``full`` or ``star``) and
``skipped_cells`` say which cells the design meant to run, so the bundle's ``declared_crossing`` reads an unrun
off-star cell as ``skipped_by_design`` (never a gap) and an unrun cell it meant to run as ``not_run``. A design
that says nothing about crossings keeps today's reading: no cell coverage at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle, build_code_only_report
from threetears.evals.analysis.report import DisclosureBlock
from threetears.evals.contracts import CampaignDesign, ControlDeclaration, EvalCampaign, SweptAxis
from threetears.evals.contracts.host import IntervalScale, SweepableValue
from packages.evals.tests.factories import memory_storage
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_INSTANT,
    TOYHOST_SCOPE,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

CHUNKS = (256, 512, 1024)
TOP_KS = (3, 5, 8)
#: The star's cells: the centre (256, 3) and every cell moving one axis from it.
STAR = [(256, 3), (512, 3), (1024, 3), (256, 5), (256, 8)]
OFF_STAR = {(512, 5), (512, 8), (1024, 5), (1024, 8)}


def _axis(axis_id: str, levels: Sequence[int], unit: str) -> SweptAxis:
    return SweptAxis(
        axis_id=axis_id,
        values=[
            SweepableValue.of(level, scale=IntervalScale(value=level, unit=unit), keep_raw=True) for level in levels
        ],
    )


def _design(**declared: object) -> CampaignDesign:
    return CampaignDesign(
        axes=[_axis("chunk_tokens", CHUNKS, "tok"), _axis("retriever_top_k", TOP_KS, "chunks")],
        declared_at=TOYHOST_INSTANT,
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="witnessed"),
        **declared,  # type: ignore[arg-type]
    )


def _bundle(design: CampaignDesign, cells: Sequence[tuple[int, int]] = STAR) -> AnalysisContextBundle:
    profile = toyhost_profile()
    storage, _ = memory_storage()
    runs = []
    for chunk, top_k in cells:
        run = toyhost_batch(chunk_tokens=chunk, retriever_top_k=top_k, grader_version="grade-2.1.0")
        storage.save_eval_run(run)
        for result in toyhost_measurements(run, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8):
            storage.save_eval_result(result)
        runs.append(run)
    campaign = EvalCampaign(
        id="crossing",
        scope_id=TOYHOST_SCOPE,
        name="crossing",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        run_ids=[run.id for run in runs],
        created_by="test:fixture",
        declared_design=design,
    )
    storage.save_campaign(campaign)
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _number(display: str) -> int:
    """A declared level's number, off its display (``256tok``)."""
    return int("".join(character for character in display if character.isdigit()))


def _states(bundle: AnalysisContextBundle) -> dict[tuple[int, int], str]:
    assert bundle.declared_crossing is not None
    return {
        (_number(cell.levels["chunk_tokens"]), _number(cell.levels["retriever_top_k"])): cell.state
        for cell in bundle.declared_crossing.cells
    }


def test_a_star_design_reads_its_off_star_cells_as_skipped_by_design() -> None:
    bundle = _bundle(_design(crossing="star"))

    states = _states(bundle)

    assert {cell for cell, state in states.items() if state == "skipped_by_design"} == OFF_STAR
    assert {cell for cell, state in states.items() if state == "ran"} == set(STAR)
    assert bundle.declared_crossing is not None and bundle.declared_crossing.n_not_run == 0
    assert "4 were skipped by design, which is no gap" in bundle.declared_crossing.sentence


def test_the_same_runs_under_a_full_crossing_read_the_off_star_cells_as_gaps() -> None:
    states = _states(_bundle(_design(crossing="full")))

    assert {cell for cell, state in states.items() if state == "not_run"} == OFF_STAR
    assert "skipped_by_design" not in states.values()


def test_a_star_cell_that_never_ran_is_a_gap_and_the_report_names_both_apart() -> None:
    bundle = _bundle(_design(crossing="star"), cells=[cell for cell in STAR if cell != (256, 8)])
    report = build_code_only_report(bundle, measures=toyhost_profile().measures, assembled_at="2026-10-10T00:00:00Z")
    texts = [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]

    assert _states(bundle)[(256, 8)] == "not_run"
    (gaps,) = [text for text in texts if text.startswith("Declared cells never run:")]
    (skipped,) = [text for text in texts if text.startswith("Cells skipped by design:")]
    assert gaps == "Declared cells never run: 256tok × 8chunks."
    assert skipped.count("×") == len(OFF_STAR)


def test_named_skipped_cells_are_skipped_and_the_rest_of_the_full_crossing_is_meant() -> None:
    skipped = [{"chunk_tokens": "1024tok", "retriever_top_k": "8chunks"}]

    states = _states(_bundle(_design(skipped_cells=skipped)))

    assert states[(1024, 8)] == "skipped_by_design"
    assert {cell for cell, state in states.items() if state == "not_run"} == OFF_STAR - {(1024, 8)}


def test_a_design_that_declares_no_crossing_keeps_todays_reading() -> None:
    assert _bundle(_design()).declared_crossing is None


def test_a_skipped_cell_naming_an_undeclared_level_is_refused() -> None:
    with pytest.raises(ValidationError, match="none of its declared levels"):
        _design(skipped_cells=[{"chunk_tokens": "2048tok", "retriever_top_k": "3chunks"}])
    with pytest.raises(ValidationError, match="one level of EVERY declared axis"):
        _design(skipped_cells=[{"chunk_tokens": "256tok"}])
