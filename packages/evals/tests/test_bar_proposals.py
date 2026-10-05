"""Bar proposals: a baseline campaign's incumbent is measured, and each measure with a better end gets a proposed bar.

:func:`~threetears.evals.analysis.propose_bars` is the ratchet's caller. It reads the baseline
campaign's one cell the way an analysis does, proposes a bar per declared directional measure, flags a
seed nothing could fail, and registers nothing.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``propose_bars``: removing the not-found raise; removing the one-cell refusal (a two-cell campaign
  then proposes from whichever cell came first); proposing on a directionless measure; reading the
  threshold off the cell's median rather than its mean.
- ``BarRegistry._vacuity``: removing the permissive-end branch (a flat baseline then reads as a
  discriminating bar).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from threetears.evals.analysis import BaselineBarProposals, propose_bars
from threetears.evals.contracts import EvalCampaign, EvalResult, EvalRun, NotFoundError, ValidationFailedError
from threetears.evals.contracts.host import EvalHost
from packages.evals.tests.factories import memory_storage
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, toyhost_batch, toyhost_measurements
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import FIELD_COUNT_ERROR, toyhost_profile

BEHAVIOR = "extract_invoice_fields"


def _batch(chunk_tokens: int) -> EvalRun:
    return toyhost_batch(
        chunk_tokens=chunk_tokens,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
        grader_version="grade-2.1.0",
    )


def _host(batches: Sequence[tuple[EvalRun, list[EvalResult]]], campaign_id: str = "baseline") -> EvalHost:
    storage, _ = memory_storage()
    for run, results in batches:
        storage.save_eval_run(run)
        for result in results:
            storage.save_eval_result(result)
    storage.save_campaign(
        EvalCampaign(
            id=campaign_id,
            scope_id=TOYHOST_SCOPE,
            name="incumbent baseline",
            subject_id=batches[0][0].subject_snapshot.subject_id,
            subject_kind="extractor_config",
            behavior=BEHAVIOR,
            run_ids=[run.id for run, _ in batches],
            created_by="test:fixture",
        )
    )
    return toyhost_host(storage=storage)


def _measured(chunk_tokens: int, *, field_accuracy: float) -> tuple[EvalRun, list[EvalResult]]:
    batch = _batch(chunk_tokens)
    return batch, toyhost_measurements(
        batch, profile=toyhost_profile(), cost_usd=0.02, total_ms=900.0, field_accuracy=field_accuracy
    )


def _flat(chunk_tokens: int) -> tuple[EvalRun, list[EvalResult]]:
    """A baseline whose quality measure sat at zero on every observation — bottomed out."""
    batch, results = _measured(chunk_tokens, field_accuracy=0.5)
    return batch, [
        result.model_copy(update={"host_measures": {**result.host_measures, "field_accuracy": 0.0}})
        for result in results
    ]


def _proposals(host: EvalHost) -> BaselineBarProposals:
    return propose_bars(host, "baseline", scope_id=TOYHOST_SCOPE)


class TestAFlatBaselineIsFlaggedVacuous:
    """Done when: a proposal against a flat baseline is flagged vacuous."""

    def test_a_quality_measure_bottomed_out_at_its_floor_proposes_a_bar_nothing_can_fail(self) -> None:
        result = _proposals(_host([_flat(256)]))

        accuracy = next(p for p in result.proposals if p.bar.measure == "field_accuracy")
        assert accuracy.bar.threshold == 0.0
        assert accuracy.vacuous and accuracy.bar.vacuous_seed
        assert "cleared by every value" in accuracy.reason
        assert accuracy in result.vacuous

    def test_a_baseline_that_discriminates_is_not_flagged(self) -> None:
        # 0.95 tightens the registered 0.92, and sits well off the measure's floor.
        result = _proposals(_host([_measured(256, field_accuracy=0.95)]))

        accuracy = next(p for p in result.proposals if p.bar.measure == "field_accuracy")
        assert accuracy.bar.threshold == pytest.approx(0.95, abs=0.02)
        assert not accuracy.vacuous and accuracy.reason == ""
        assert accuracy.bar.behavior == BEHAVIOR and accuracy.bar.higher_is_better


class TestWhatAProposalReads:
    def test_the_threshold_is_the_cells_own_mean_and_the_rationale_says_where_it_came_from(self) -> None:
        run, results = _measured(256, field_accuracy=0.8)
        # Skewed on purpose — one observation far below the rest — so the mean and the median differ
        # and a threshold read off the wrong statistic cannot pass by coincidence.
        values = [0.0, *[0.9] * (len(results) - 1)]
        results = [
            result.model_copy(update={"host_measures": {**result.host_measures, "field_accuracy": value}})
            for result, value in zip(results, values, strict=True)
        ]
        host = _host([(run, results)])

        accuracy = next(p for p in _proposals(host).proposals if p.bar.measure == "field_accuracy")

        assert accuracy.bar.threshold == pytest.approx(sum(values) / len(values))
        assert accuracy.bar.threshold != pytest.approx(0.9), "the median would be 0.9"
        assert "baseline" in accuracy.bar.rationale and f"{len(values)} observations" in accuracy.bar.rationale

    def test_a_baseline_below_the_registered_bar_is_flagged_as_loosening_it(self) -> None:
        result = _proposals(_host([_measured(256, field_accuracy=0.8)]))

        accuracy = next(p for p in result.proposals if p.bar.measure == "field_accuracy")
        assert accuracy.vacuous and "does not tighten it" in accuracy.reason

    def test_measures_no_bar_could_be_proposed_on_are_named_with_why(self) -> None:
        run, results = _measured(256, field_accuracy=0.95)
        # A declared diagnostic: carried onto the cell, and with no better end a bar could clear.
        results = [
            result.model_copy(update={"host_measures": {**result.host_measures, FIELD_COUNT_ERROR: 1.0}})
            for result in results
        ]
        result = _proposals(_host([(run, results)]))

        proposed = {proposal.bar.measure for proposal in result.proposals}
        assert proposed, "the incumbent was measured on declared measures; a proposal set must not be empty"
        assert not proposed & set(result.not_proposed), "a measure is either proposed or named, never both"
        assert "cost_usd" in result.not_proposed, "an engine-core measure this host does not declare"
        assert "does not declare" in result.not_proposed["cost_usd"]
        assert FIELD_COUNT_ERROR in result.not_proposed, "the directionless measure the cell carries"
        assert "a diagnostic" in result.not_proposed[FIELD_COUNT_ERROR]
        assert result.behavior == BEHAVIOR and result.campaign_id == "baseline"

    def test_nothing_is_registered(self) -> None:
        host = _host([_flat(256)])
        before = host.profile.bars.bars
        _proposals(host)
        assert host.profile.bars.bars == before


class TestRefusals:
    def test_an_unknown_campaign_is_refused(self) -> None:
        host = _host([_measured(256, field_accuracy=0.95)])
        with pytest.raises(NotFoundError):
            propose_bars(host, "no-such-campaign", scope_id=TOYHOST_SCOPE)

    def test_a_campaign_of_two_arms_is_not_a_baseline(self) -> None:
        host = _host([_measured(256, field_accuracy=0.95), _measured(1024, field_accuracy=0.97)])
        with pytest.raises(ValidationFailedError, match="measured 2 cells, and a baseline is one configuration"):
            _proposals(host)
