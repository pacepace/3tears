"""A campaign's bundle past the read tier's budget is refused before any result is read, as the aggregates are.

The bundle holds a reading of every result of the runs it reads at once (``test_bundle_store_reads``). Until the
store computes its aggregates (#996), a campaign whose runs' matrices (cases × k) promise more than
``READ_TIER_ROW_BUDGET`` results is refused before that read, in the aggregates' refusal shape, naming how to narrow
it; a campaign at the budget assembles. The estimate is read off the runs the batch run read already returned.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``assemble.assemble_context_bundle``: the budget held after the results are read (the refusal reads them).
- ``reads.hold_to_read_budget``: ``>=`` for ``>`` (a campaign at the budget refused); the check skipped.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from threetears.evals.analysis import READ_TIER_ROW_BUDGET
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis import reads
from threetears.evals.analysis.reads import hold_to_read_budget
from threetears.evals.kernel import CampaignArm, ValidationFailedError
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

_SCOPE = "uni-1"
#: The large run's cases: at k = 3, 49,998 results by its matrix, two under the budget.
LARGE_CASES = 16_666


class _ResultReadCounter(InMemoryDocumentStore):
    """The reference store, counting every read of results it serves, by method."""

    def __init__(self) -> None:
        super().__init__()
        self.result_reads: Counter[str] = Counter()

    def by_doc_type(
        self, doc_type: str, scope_id: str, *, field_in: Mapping[str, Sequence[Any]] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        if doc_type == "eval_result":
            self.result_reads["by_doc_type"] += 1
        return super().by_doc_type(doc_type, scope_id, field_in=field_in, **kwargs)

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        if doc_type == "eval_result":
            self.result_reads["get_many"] += 1
        return super().get_many(doc_type, doc_ids, scope_id, **kwargs)

    def iter_narrowed(self, doc_type: str, scope_id: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        if doc_type == "eval_result":
            self.result_reads["iter_narrowed"] += 1
        return super().iter_narrowed(doc_type, scope_id, **kwargs)


def _campaign(*, small_cases: int) -> tuple[_ResultReadCounter, EvalStorage, EvalCampaign]:
    """A campaign of two runs: one of ``small_cases`` cases, each result stored; one whose matrix is
    :data:`LARGE_CASES` × 3 with three results stored. The estimate is ``small_cases + 49,998``, however few of the
    large run's results are stored."""
    store = _ResultReadCounter()
    storage = EvalStorage(store)
    small = make_eval_run(
        id="run-small", status="completed", k_runs=1, test_case_ids=[f"tc-{case}" for case in range(small_cases)]
    )
    large = make_eval_run(
        id="run-large", status="completed", k_runs=3, test_case_ids=[f"tc-{case}" for case in range(LARGE_CASES)]
    )
    for run, stored in ((small, small_cases), (large, 3)):
        storage.save_eval_run(run)
        for case in range(stored):
            storage.save_eval_result(
                make_eval_result(id=f"{run.id}-{case}", eval_run_id=run.id, test_case_id=f"tc-{case}", scope_id=_SCOPE)
            )
    store.result_reads.clear()
    return store, storage, make_aspect_campaign(run_ids=[small.id, large.id])


class TestABundlePastTheReadBudgetIsRefused:
    def test_one_result_past_the_budget_is_refused_naming_how_to_narrow_it(self) -> None:
        # 3 + 16,666 × 3 = 50,001: one past the budget.
        store, storage, campaign = _campaign(small_cases=3)

        with pytest.raises(ValidationFailedError) as refused:
            assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())

        assert refused.value.details == {
            "runs": 2,
            "results_estimated": READ_TIER_ROW_BUDGET + 1,
            "result_budget": READ_TIER_ROW_BUDGET,
        }
        assert "assemble_context_bundle would aggregate about 50001 results" in refused.value.message
        assert refused.value.remedy.startswith(
            "assemble a campaign of fewer member runs, or archive the members it need not read"
        )

    def test_the_refusal_reads_no_result(self) -> None:
        store, storage, campaign = _campaign(small_cases=3)

        with pytest.raises(ValidationFailedError):
            assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())

        assert store.result_reads == Counter(), "refused before any result was read"

    def test_a_campaign_at_the_budget_assembles_in_its_one_read_of_results(self) -> None:
        # 2 + 16,666 × 3 = 50,000: at the budget, which a call is answered within.
        store, storage, campaign = _campaign(small_cases=2)

        bundle = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())

        assert sorted(bundle.run_ids) == ["run-large", "run-small"]
        # The store narrows its own rows, so the results are one read (D7, test_bundle_store_reads); the trials the
        # bundle keeps are then read whole, by id, in one more.
        assert store.result_reads == Counter({"iter_narrowed": 1, "get_many": 1}), "the one read of results"


class TestTheBudgetHoldsAReadThatReturnedMoreThanItsRunsPromised:
    def test_a_read_past_the_budget_is_refused_naming_what_it_returned(self) -> None:
        run = make_eval_run(k_runs=1, test_case_ids=["tc-0"])
        hold_to_read_budget("assemble_context_bundle", _SCOPE, [run], how="narrow it", returned=READ_TIER_ROW_BUDGET)

        with pytest.raises(ValidationFailedError) as refused:
            hold_to_read_budget(
                "assemble_context_bundle", _SCOPE, [run], how="narrow it", returned=READ_TIER_ROW_BUDGET + 1
            )

        assert refused.value.details == {
            "runs": 1,
            "results_estimated": 1,
            "result_budget": READ_TIER_ROW_BUDGET,
            "results_returned": READ_TIER_ROW_BUDGET + 1,
        }


def _stored(storage: EvalStorage, run_id: str, *, cases: int, k: int, archived: bool = False) -> str:
    """A completed run of ``cases`` × ``k``, every cell stored."""
    run = make_eval_run(
        id=run_id,
        status="completed",
        k_runs=k,
        archived=archived,
        test_case_ids=[f"tc-{case}" for case in range(cases)],
    )
    storage.save_eval_run(run)
    for case in range(cases):
        for iteration in range(1, k + 1):
            storage.save_eval_result(
                make_eval_result(
                    id=f"{run_id}-{case}-{iteration}",
                    eval_run_id=run_id,
                    test_case_id=f"tc-{case}",
                    k_iteration=iteration,
                    scope_id=_SCOPE,
                )
            )
    return run_id


class TestTheBundleCountsWhatItReads:
    def test_an_archived_member_read_for_a_declared_control_counts_and_one_not_read_does_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _ResultReadCounter()
        storage = EvalStorage(store)
        member = _stored(storage, "run-member", cases=3, k=1)
        curated = _stored(storage, "run-curated", cases=3, k=1, archived=True)
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 5)
        control = CampaignArm(label="control", role="control", variant_key=fixture_variant_key("sonnet"))
        store.result_reads.clear()

        # A declared control may have been curated out, so the archived member is read too: 6 results, past 5.
        with pytest.raises(ValidationFailedError) as refused:
            assemble_context_bundle(
                make_aspect_campaign(run_ids=[member, curated], control=control),
                storage=storage,
                profile=toyhost_profile(),
            )
        assert refused.value.details["runs"] == 2 and refused.value.details["results_estimated"] == 6
        assert store.result_reads == Counter()

        # With no declared control the archived member is not read, and the 3 the bundle reads are within 5.
        bundle = assemble_context_bundle(
            make_aspect_campaign(run_ids=[member, curated]), storage=storage, profile=toyhost_profile()
        )
        assert bundle.run_ids == [member]

    def test_a_read_returning_more_than_its_runs_promised_past_the_budget_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storage = EvalStorage(InMemoryDocumentStore())
        # A matrix of one cell, which stored five: the estimate (1) is within the budget, the read (5) is not.
        run_id = _stored(storage, "run-overran", cases=1, k=1)
        for iteration in range(2, 6):
            storage.save_eval_result(
                make_eval_result(
                    id=f"{run_id}-extra-{iteration}",
                    eval_run_id=run_id,
                    test_case_id="tc-0",
                    k_iteration=iteration,
                    scope_id=_SCOPE,
                )
            )
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 3)

        with pytest.raises(ValidationFailedError) as refused:
            assemble_context_bundle(make_aspect_campaign(run_ids=[run_id]), storage=storage, profile=toyhost_profile())

        assert refused.value.details == {
            "runs": 1,
            "results_estimated": 1,
            "result_budget": 3,
            "results_returned": 5,
        }
