"""A declared level that never ran is named in coverage, apart from a level nobody declared (#690).

``LeverCoverageInput.levels`` lists only levels that ran, so a campaign declaring three levels and
running two looked exactly like one that declared two: the design declaration's promise that such a
level is "named" was false, and a reader had to diff the declaration against the row by hand.

Each declared axis row now carries ``declared_levels``: every declared level, joined to the runs on
content identity, marked ``ran`` / ``not_run`` / ``undetermined``. The code-only report names a level
declared and never run.
"""

from __future__ import annotations

from threetears.evals.analysis import (
    AnalysisContextBundle,
    LeverCoverageInput,
    assemble_context_bundle,
    build_code_only_report,
)
from threetears.evals.analysis.report import DisclosureBlock
from threetears.evals.kernel import SweptAxis
from threetears.evals.schema import IntervalScale, SweepableValue
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_AXIS, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage, toyhost_batch, toyhost_measurements
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: A third chunk width the campaign declares and never runs.
_UNRUN = 2048


def _bundle(*, extra_level: bool) -> AnalysisContextBundle:
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    if extra_level:
        design = campaign.declared_design
        assert design is not None
        (axis,) = design.axes
        unrun = SweepableValue.of(_UNRUN, scale=IntervalScale(value=_UNRUN, unit="tok"), keep_raw=True)
        widened = axis.model_copy(update={"values": [*axis.values, unrun]})
        campaign = campaign.model_copy(update={"declared_design": design.model_copy(update={"axes": [widened]})})
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _row(bundle: AnalysisContextBundle) -> LeverCoverageInput:
    (row,) = [row for row in bundle.coverage if row.name == TOYHOST_AXIS]
    return row


def _report_texts(bundle: AnalysisContextBundle) -> list[str]:
    report = build_code_only_report(
        bundle, measures=toyhost_profile().measures, assembled_at="2026-10-10T00:00:00+00:00"
    )
    return [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]


def test_three_declared_two_run_names_the_third_as_not_run() -> None:
    row = _row(_bundle(extra_level=True))

    states = {level.display: level.state for level in row.declared_levels}

    assert len(row.levels) == 2, "levels lists only what ran"
    unrun = SweepableValue.of(_UNRUN, scale=IntervalScale(value=_UNRUN, unit="tok")).display
    assert states.pop(unrun) == "not_run"
    assert set(states.values()) == {"ran"} and len(states) == 2


def test_the_code_only_report_names_the_unrun_level() -> None:
    bundle = _bundle(extra_level=True)
    (level,) = [level for level in _row(bundle).declared_levels if level.state == "not_run"]

    texts = _report_texts(bundle)

    assert f"Declared on axis {TOYHOST_AXIS} and never run: {level.display}." in texts


def test_every_declared_level_run_reports_none_missing() -> None:
    bundle = _bundle(extra_level=False)

    assert [level.state for level in _row(bundle).declared_levels] == ["ran", "ran"]
    assert not any(text.startswith("Declared on axis") for text in _report_texts(bundle))


def test_an_undeclared_row_carries_no_declared_levels() -> None:
    bundle = _bundle(extra_level=True)

    assert all(row.declared_levels == [] for row in bundle.coverage if row.name != TOYHOST_AXIS)


def test_a_level_no_run_matched_is_undetermined_where_a_run_inherited() -> None:
    """A control that named no retrieval knob may sit at any declared level: 'not run' cannot be claimed."""
    profile = toyhost_profile(tunable_retrieval=True)
    campaign, storage = toyhost_campaign(profile=profile)
    runs = storage.load_eval_runs(campaign.run_ids, campaign.scope_id)
    tuned = toyhost_batch(
        chunk_tokens=1024,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
        retrieval_overrides={"rerank_depth": 8},
        retrieval_config={"rerank_depth": 8, "dedupe_threshold": 0.9},
    )
    results = {run.id: storage.query_eval_results_by_run(run.id, run.scope_id) for run in runs}
    results[tuned.id] = toyhost_measurements(tuned, profile=profile, cost_usd=0.002, total_ms=900.0, field_accuracy=0.9)
    axis = SweptAxis(
        axis_id="retrieval.rerank_depth",
        values=[SweepableValue.of(depth, display=str(depth), keep_raw=True) for depth in (8, 16)],
    )
    design = campaign.declared_design
    assert design is not None
    campaign = campaign.model_copy(
        update={
            "run_ids": [*campaign.run_ids, tuned.id],
            "declared_design": design.model_copy(update={"axes": [*design.axes, axis]}),
        }
    )

    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage([*runs, tuned], results), profile=profile)

    (row,) = [row for row in bundle.coverage if row.name == "retrieval.rerank_depth"]
    assert {level.display: level.state for level in row.declared_levels} == {"8": "ran", "16": "undetermined"}
