"""Factors that move together are grouped, and every co-varying pair is shown as a pivot (#596).

The bundle's coverage lists what else varied behind each lever, one factor at a time; when four factors moved
in lockstep across every run, a reader saw four rows each confounded by the other three and no statement that
nothing could separate them. The bundle now:

- groups factors whose partitions of the runs are identical into one ``aliased_factors`` entry with its run
  count and ONE sentence, and the report says that sentence, not one per factor;
- checks every pair of varying factors, carrying a two-factor pivot (unrun combinations as ``not_run``) for each
  co-varying pair outside a group, and says how many pairs it examined;
- states that aliasing with an interaction is not checked.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle, build_code_only_report
from threetears.evals.analysis.bundle import INTERACTION_ALIASING_UNCHECKED
from threetears.evals.analysis.report import DisclosureBlock
from threetears.evals.kernel import EvalCampaign
from packages.evals.tests.factories import memory_storage
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, toyhost_batch, toyhost_measurements
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: Four factors set in lockstep: three swept levers and one apparatus dimension.
LOCKSTEP = ("chunk_tokens", "extraction_schema", "ocr_engine_version", "retriever_top_k")
LOW = {"chunk_tokens": 256, "retriever_top_k": 3, "extraction_schema": "v1", "ocr_engine_version": "tess-5.3.1"}
HIGH = {"chunk_tokens": 512, "retriever_top_k": 5, "extraction_schema": "v2", "ocr_engine_version": "tess-5.4.0"}


def _bundle(batches: Sequence[dict[str, Any]]) -> AnalysisContextBundle:
    """A campaign of one toy batch per entry, every batch recording each value it names."""
    profile = toyhost_profile()
    storage, _ = memory_storage()
    runs = []
    for values in batches:
        run = toyhost_batch(grader_version="grade-2.1.0", **values)
        storage.save_eval_run(run)
        for result in toyhost_measurements(run, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8):
            storage.save_eval_result(result)
        runs.append(run)
    campaign = EvalCampaign(
        id="aliasing",
        scope_id=TOYHOST_SCOPE,
        name="aliasing",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        run_ids=[run.id for run in runs],
        created_by="test:fixture",
    )
    storage.save_campaign(campaign)
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _lockstep(*, split_ocr: bool = False) -> AnalysisContextBundle:
    """Four runs: the four factors low on two and high on two, the reviewer pool crossed with them."""
    second_low = {**LOW, "ocr_engine_version": HIGH["ocr_engine_version"]} if split_ocr else LOW
    return _bundle(
        [
            {**LOW, "reviewer_pool": "pool-a"},
            {**second_low, "reviewer_pool": "pool-b"},
            {**HIGH, "reviewer_pool": "pool-a"},
            {**HIGH, "reviewer_pool": "pool-b"},
        ]
    )


def _disclosures(bundle: AnalysisContextBundle) -> list[str]:
    report = build_code_only_report(bundle, measures=toyhost_profile().measures, assembled_at="2026-10-10T00:00:00Z")
    return [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]


class TestLockstepFactorsAreOneGroup:
    def test_four_factors_with_one_partition_are_one_group_with_the_run_count(self) -> None:
        bundle = _lockstep()

        (group,) = bundle.aliased_factors

        assert tuple(group.factors) == LOCKSTEP
        assert (group.n_runs, group.n_levels) == (4, 2)
        assert group.sentence.startswith(
            "chunk_tokens, extraction_schema, ocr_engine_version and retriever_top_k move together across all 4 runs"
        )
        assert "no comparison separates them" in group.sentence

    def test_the_group_is_said_once_and_no_pair_of_its_factors_is_said_apart(self) -> None:
        """Ungrouped, the four would co-vary as six pairs, each with a pivot and a sentence of its own."""
        bundle = _lockstep()
        (group,) = bundle.aliased_factors

        texts = _disclosures(bundle)

        assert texts.count(group.sentence) == 1
        assert bundle.factor_pairs is not None and bundle.factor_pairs.pivots == []
        assert not [text for text in texts if "co-vary;" in text], "a pair of lockstep factors said on its own"

    def test_the_bundle_states_that_interaction_aliasing_is_not_checked(self) -> None:
        bundle = _lockstep()

        assert bundle.factor_pairs is not None
        assert bundle.factor_pairs.interaction_aliasing == INTERACTION_ALIASING_UNCHECKED
        assert any(INTERACTION_ALIASING_UNCHECKED in text for text in _disclosures(bundle))

    def test_splitting_one_factors_partition_removes_it_from_the_group(self) -> None:
        bundle = _lockstep(split_ocr=True)

        (group,) = bundle.aliased_factors

        assert "ocr_engine_version" not in group.factors
        assert tuple(group.factors) == ("chunk_tokens", "extraction_schema", "retriever_top_k")

    def test_a_crossed_factor_is_neither_grouped_nor_pivoted(self) -> None:
        bundle = _lockstep()

        assert bundle.factor_pairs is not None
        assert "reviewer_pool" in bundle.factor_pairs.factors
        assert bundle.factor_pairs.pivots == []
        # Five factors: ten pairs, the six inside the group co-vary, and the four with the crossed pool do not.
        assert (bundle.factor_pairs.n_pairs_examined, bundle.factor_pairs.n_covarying) == (10, 6)
        assert bundle.factor_pairs.n_covarying_in_groups == 6


class TestACoVaryingPairOutsideAnyGroupIsPivoted:
    def _nested(self) -> AnalysisContextBundle:
        """chunk_tokens is nested in retriever_top_k: top_k never moves with chunk_tokens held."""
        return _bundle(
            [
                {"chunk_tokens": 256, "retriever_top_k": 3, "reviewer_pool": "pool-a"},
                {"chunk_tokens": 256, "retriever_top_k": 3, "reviewer_pool": "pool-b"},
                {"chunk_tokens": 512, "retriever_top_k": 5, "reviewer_pool": "pool-a"},
                {"chunk_tokens": 1024, "retriever_top_k": 5, "reviewer_pool": "pool-a"},
            ]
        )

    def test_its_pivot_shows_the_unrun_combinations_and_the_count_of_pairs_examined(self) -> None:
        bundle = self._nested()

        assert bundle.aliased_factors == []
        scan = bundle.factor_pairs
        assert scan is not None
        (pivot,) = scan.pivots
        assert (pivot.row_factor, pivot.column_factor) == ("chunk_tokens", "retriever_top_k")
        assert len(pivot.cells) == 6
        assert sorted(cell.n_runs for cell in pivot.cells if cell.status == "ran") == [1, 1, 2]
        assert sum(cell.status == "not_run" for cell in pivot.cells) == 3
        assert all(cell.n_runs == 0 for cell in pivot.cells if cell.status == "not_run")
        assert (scan.n_pairs_examined, scan.n_covarying) == (3, 1)
        assert scan.completeness.startswith("3 factor pair(s) examined over the 3 factors that varied; 1 co-vary")

    def test_the_report_lists_the_pairs_holes(self) -> None:
        texts = _disclosures(self._nested())

        assert any(
            text.startswith("chunk_tokens and retriever_top_k co-vary; 3 of their 6 combinations never ran")
            for text in texts
        )
