"""A run's stored results are never read whole: its summary, a take-up of it and an offer of one read it a page at a time.

A stored result carries its judge's reasoning, its kind's payload and its world events, so a read of every result of
a large run held the run's whole volume at once. Pinned here through a store that records how many results each read
of a run hands out:

- **summarize_run** reads its run a page of whole cases at a time, never more than a page (or one case) in one read,
  and its summary is byte for byte the one a whole read of the run in one call gives, at one case a page, a few, or
  more cases than a page holds: on a run much larger than a page, and on the toy host's stored corpus. At the real
  page size, unpatched, no read passes it.
- **A take-up** plans from pages: what it holds of each stored row (done or run again) and every row its cap counts
  are the same at any page size, no read of the run hands out more than a page, and a take-up paged one case at a time
  ends with every cell stored once, completed.
- **resumable_jobs** offers a large run read a page at a time, as the whole read offers it.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``summary.summarize_run``: the run read whole (``query_eval_results_by_run``); the page size ignored.
- ``summary._ExactSum``: a plain running ``+=`` (the summary moves with the page size).
- ``resume._stored_rows``: the run read whole; a row run again kept as done.
- ``run_pages.run_case_pages``: one page of every case; ``per_page`` left at the page size whatever ``k``.
- ``run_pages.read_case_page``: the page left in the store's order.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from threetears.evals.analysis import summary
from threetears.evals.analysis.summary import summarize_run
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.run_pages import RUN_RESULTS_PAGE
from threetears.evals.ops import OpsHost, run_job_id
from threetears.evals.run import resumable_jobs, resume_run
from threetears.evals.run import resume
from threetears.evals.run.resume import take_up_arms
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.schema.models import RubricScore
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_test_case
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.lease_support import ClockedStore, held_extraction, launched_arm, until
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture, settled

SCOPE = "uni-run-reads-paged"
#: The large run's cases and repeats: 120 results, many pages at any patched page size.
CASES = 40
K = 3
#: Repeats of the toy host's take-up, and the cells its first holding stores before it is cancelled.
TAKE_UP_K = 3
DONE_BEFORE = 3


class BatchRecordingStore(ClockedStore):
    """The engine's in-memory store, recording how many results each read naming a run hands out."""

    def __init__(self) -> None:
        super().__init__()
        #: ``(run ids the read named, whether it named cases too, results it handed out)``, one per read of results that
        #: names a run.
        self.run_reads: list[tuple[tuple[str, ...], bool, int]] = []

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        rows = super().by_doc_type(
            doc_type,
            scope_id,
            order_by=order_by,
            descending=descending,
            limit=limit,
            exclude=exclude,
            field_in=field_in,
            **field_eq,
        )
        named = (
            *([field_eq["eval_run_id"]] if "eval_run_id" in field_eq else []),
            *(field_in or {}).get("eval_run_id", ()),
        )
        if doc_type == "eval_result" and named:
            self.run_reads.append((tuple(named), "test_case_id" in (field_in or {}), len(rows)))
        return rows

    def batches_of(self, run_id: str) -> list[int]:
        """How many results each read naming ``run_id`` handed out, in call order."""
        return [rows for named, _, rows in self.run_reads if run_id in named]

    def page_batches_of(self, run_id: str) -> list[int]:
        """How many results each read of some of ``run_id``'s cases handed out, in call order."""
        return [rows for named, by_case, rows in self.run_reads if run_id in named and by_case]


def _results(run: EvalRun, *, cases: int, k: int) -> list[EvalResult]:
    """A row per cell of ``run``, varied in every field a summary reads, in an order the store keeps (shuffled)."""
    rows = []
    for case in range(cases):
        for iteration in range(1, k + 1):
            failed = (case + iteration) % 11 == 0
            rows.append(
                make_eval_result(
                    id=f"{run.id}-{case}-{iteration}",
                    eval_run_id=run.id,
                    scope_id=run.scope_id,
                    model=run.candidate_model,
                    test_case_id=f"tc-{case}",
                    k_iteration=iteration,
                    # Sums an order-dependent running total rounds differently from one exact sum.
                    cost_usd=(0.1, 0.2, 1 / 3, 1e-17, 7.25)[(case * k + iteration) % 5],
                    host_measures={
                        "field_accuracy": ((case * iteration) % 7) / 7,
                        "p95_extract_ms": 100.0 + case / 3 + iteration * 0.1,
                        "manual_review_rate": (case + iteration) % 3 == 0,
                    },
                    rubric_scores=[
                        RubricScore(
                            dim="conversation.tone",
                            score=1 + (case + iteration) % 5,
                            scale="ordinal",
                            axis="capability",
                            judge_temperature=0.0,
                        )
                    ],
                    candidate_error="the candidate refused" if failed else None,
                    runner_error=f"cell {case}/{iteration} broke" if failed else None,
                )
            )
    random.Random(7).shuffle(rows)
    return rows


def _large_run(storage: EvalStorage, *, cases: int = CASES, k: int = K) -> EvalRun:
    """A completed run of ``cases`` × ``k`` stored results, its even cases in a stratum."""
    run = make_eval_run(
        scope_id=SCOPE, status="completed", k_runs=k, test_case_ids=[f"tc-{case}" for case in range(cases)]
    )
    storage.save_eval_run(run)
    for case in range(cases):
        storage.save_test_case(
            make_test_case(id=f"tc-{case}", scope_id=SCOPE, stratum="even" if case % 2 == 0 else None)
        )
    for row in _results(run, cases=cases, k=k):
        storage.save_eval_result(row)
    return run


def _host() -> tuple[BatchRecordingStore, EvalHost]:
    store = BatchRecordingStore()
    return store, toyhost_host(storage=EvalStorage(store))


class TestASummaryReadsItsRunAPageAtATime:
    def test_at_any_page_size_it_is_the_summary_a_whole_read_gives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store, host = _host()
        run = _large_run(host.storage)
        monkeypatch.setattr(summary, "RUN_RESULTS_PAGE", CASES * K)
        whole = summarize_run(host, run.id, SCOPE)
        assert store.batches_of(run.id) == [CASES * K], "one read of the whole run"
        assert whole.n_results == CASES * K and whole.errors and whole.strata, "the fixture exercises the summary"

        for page in (1, K, 4 * K, 7):
            monkeypatch.setattr(summary, "RUN_RESULTS_PAGE", page)
            store.run_reads.clear()

            paged = summarize_run(host, run.id, SCOPE)

            assert paged.model_dump_json() == whole.model_dump_json(), f"at {page} a page"
            assert max(store.batches_of(run.id)) <= max(page, K), "no read past a page, or one case"
            assert sum(store.batches_of(run.id)) == CASES * K, "each result read once"
            assert len(store.batches_of(run.id)) == -(-CASES // max(1, page // K)), "a read per page of cases"

    def test_at_the_real_page_no_read_hands_out_more(self) -> None:
        store, host = _host()
        cases = RUN_RESULTS_PAGE + RUN_RESULTS_PAGE // 2
        run = _large_run(host.storage, cases=cases, k=2)

        summarized = summarize_run(host, run.id, SCOPE)

        assert summarized.n_results == cases * 2
        assert max(store.batches_of(run.id)) <= RUN_RESULTS_PAGE
        assert len(store.batches_of(run.id)) == 3

    def test_the_toy_hosts_stored_runs_summarise_alike_at_one_case_a_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = BatchRecordingStore()
        host = off_loop(ops_fixture, store=store).host.eval_host
        runs = host.storage.query_eval_runs(TOYHOST_SCOPE)
        assert runs, "the toy corpus stores runs"
        for run in runs:
            monkeypatch.setattr(summary, "RUN_RESULTS_PAGE", 10_000)
            whole = summarize_run(host, run.id, TOYHOST_SCOPE).model_dump_json()
            monkeypatch.setattr(summary, "RUN_RESULTS_PAGE", 1)
            store.run_reads.clear()

            paged = summarize_run(host, run.id, TOYHOST_SCOPE).model_dump_json()

            assert paged == whole
            assert len(store.batches_of(run.id)) == len(dict.fromkeys(run.test_case_ids)), "one read per case"
            assert max(store.batches_of(run.id)) <= run.k_runs


async def _cancelled(monkeypatch: pytest.MonkeyPatch, store: BatchRecordingStore) -> tuple[OpsHost, str]:
    """The toy host's run of three cases × :data:`TAKE_UP_K`, cancelled once :data:`DONE_BEFORE` cells are stored."""
    gate = held_extraction(monkeypatch, passing=DONE_BEFORE)
    holder = off_loop(ops_fixture, store=store).host
    launch_id, run_id = await launched_arm(holder, TAKE_UP_K)
    rows = holder.eval_host.storage.query_eval_results_by_run
    await until(lambda: len(off_loop(rows, run_id, TOYHOST_SCOPE)) == DONE_BEFORE and gate.held >= 1)
    manager = holder.launch.job_manager
    assert manager.cancel_job(run_id, reason="shipping a fix")
    await until(lambda: not manager.is_active(run_id))
    gate.set()
    await until(lambda: not manager.is_task_active(launch_id))
    assert "cancelled" in {row.termination for row in off_loop(rows, run_id, TOYHOST_SCOPE)}, "cells were cut short"
    return holder, run_id


class TestATakeUpPlansFromPages:
    async def test_its_plan_is_the_same_at_any_page_size_and_no_read_passes_a_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = BatchRecordingStore()
        holder, run_id = await _cancelled(monkeypatch, store)
        monkeypatch.setattr(resume, "RUN_RESULTS_PAGE", 10_000)
        whole, refused = await take_up_arms(holder.launch, [("arm", run_id)], TOYHOST_SCOPE)
        assert not refused

        monkeypatch.setattr(resume, "RUN_RESULTS_PAGE", 1)
        store.run_reads.clear()
        paged, refused = await take_up_arms(holder.launch, [("arm", run_id)], TOYHOST_SCOPE)

        assert not refused
        assert paged["arm"].stored == whole["arm"].stored
        assert paged["arm"].rows == whole["arm"].rows
        assert paged["arm"].stored.done and paged["arm"].stored.replaced, "done cells and cells run again alike"
        stored = off_loop(holder.eval_host.storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)
        assert len(paged["arm"].rows) == len(stored)
        # The replayed launch's history pricer reads each earlier run of the template by run, this one too: the take-up's
        # own read of the run is its pages.
        pages = store.page_batches_of(run_id)
        assert len(pages) == 3 and max(pages) <= TAKE_UP_K and sum(pages) == len(stored), "a case a page"

    async def test_a_take_up_paged_one_case_at_a_time_ends_with_every_cell_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = BatchRecordingStore()
        holder, run_id = await _cancelled(monkeypatch, store)
        monkeypatch.setattr(resume, "RUN_RESULTS_PAGE", 1)

        await resume_run(holder.launch, run_id, TOYHOST_SCOPE)
        assert (await settled(holder, run_job_id(run_id))).state == "completed"

        rows = off_loop(holder.eval_host.storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)
        assert len(rows) == 3 * TAKE_UP_K
        assert len({(row.test_case_id, row.k_iteration) for row in rows}) == 3 * TAKE_UP_K
        assert "cancelled" not in {row.termination for row in rows}


class TestAnOfferReadsItsRunAPageAtATime:
    def test_it_offers_what_a_whole_read_offers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = BatchRecordingStore()
        storage = EvalStorage(store)
        run = make_eval_run(
            scope_id=SCOPE,
            status="cancelled",
            cancellation_reason="shipping a fix",
            template_id="tpl-1",
            k_runs=K,
            test_case_ids=[f"tc-{case}" for case in range(CASES)],
        )
        storage.save_eval_run(run)
        rows = _results(run, cases=CASES, k=K)
        for index, row in enumerate(rows):
            cut_short = index % 4 == 0
            storage.save_eval_result(row.model_copy(update={"termination": "cancelled"}) if cut_short else row)
        monkeypatch.setattr(resume, "RUN_RESULTS_PAGE", CASES * K)
        (whole,) = resumable_jobs(storage, [SCOPE]).jobs

        monkeypatch.setattr(resume, "RUN_RESULTS_PAGE", 2 * K)
        store.run_reads.clear()
        (paged,) = resumable_jobs(storage, [SCOPE]).jobs

        assert paged == whole
        assert paged.cells_done == CASES * K - len(range(0, CASES * K, 4))
        assert max(store.batches_of(run.id)) <= 2 * K
        assert sum(store.batches_of(run.id)) == CASES * K
