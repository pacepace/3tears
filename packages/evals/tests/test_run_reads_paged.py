"""A run's stored results are never read whole: its summary, a take-up of it and an offer of one read it a page at a time.

A stored result carries its judge's reasoning, its kind's payload and its world events, so a read of every result of
a large run held the run's whole volume at once. Each reader now passes over the run once, reading its results' keys
alone, then reads its results by id a page at a time. Pinned here through a store that records every read of results
it serves:

- **summarize_run** makes one read of the run's keys and reads its results by id, none past a page, each once, with
  no read that hands out the run's rows; its summary is byte for byte the one a single page holding the whole run
  gives, at one result a page, a few, or more: on a run much larger than a page, and on the toy host's stored corpus.
  At the real page size, unpatched, no read passes it.
- **A take-up** plans from pages: what it holds of each stored row (done or run again) and every row its cap counts
  are the same at any page size. Every read it makes, its replayed launch's history pricer included, is a read of
  keys or of results by id, a page at a time; none hands out a run's rows, nor the scope's. A take-up paged one
  result at a time ends with every cell stored once, completed.
- **resumable_jobs** offers a large run read a page at a time, as a single page holding it offers it.
- **Both stores the engine ships** (in memory, SQLite) serve the two reads: the keys, with what grows with a cell left
  out, then the results by id.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``summary.summarize_run``: the run read whole (``query_eval_results_by_run``); the page size ignored.
- ``summary._ExactSum``: a plain running ``+=`` (the summary moves with the page size).
- ``resume._stored_rows``: the run read whole; a row run again kept as done.
- ``lenses.history_launch_pricer``: a run of the template's history read whole.
- ``run_pages.iter_run_results``: a page past ``page_results``.
- ``storage.query_eval_result_keys_by_run``: ``usage``, ``rubric_scores`` or ``world_events`` no longer left out.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path


import pytest

from threetears.evals.analysis import summary
from threetears.evals.analysis.summary import summarize_run
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.run_pages import RUN_RESULTS_PAGE
from threetears.evals.ops import OpsHost, history_launch_pricer, lenses, run_job_id
from threetears.evals.run import resumable_jobs, resume_run
from threetears.evals.run import resume
from threetears.evals.run.resume import take_up_arms
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.schema.models import RubricScore
from threetears.evals.storage import InMemoryDocumentStore, SqliteDocumentStore
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_test_case
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.lease_support import held_extraction, launched_arm, until
from packages.evals.tests.run_pages_support import BatchRecordingStore
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture, settled

SCOPE = "uni-run-reads-paged"
#: The large run's cases and repeats: 120 results, many pages at any patched page size.
CASES = 40
K = 3
#: Repeats of the toy host's take-up, and the cells its first holding stores before it is cancelled.
TAKE_UP_K = 3
DONE_BEFORE = 3


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
        store.clear()
        whole = summarize_run(host, run.id, SCOPE)
        assert store.id_reads == [CASES * K], "the whole run in one read"
        assert whole.n_results == CASES * K and whole.errors and whole.strata, "the fixture exercises the summary"

        for page in (1, K, 4 * K, 7):
            monkeypatch.setattr(summary, "RUN_RESULTS_PAGE", page)
            store.clear()

            paged = summarize_run(host, run.id, SCOPE)

            assert paged.model_dump_json() == whole.model_dump_json(), f"at {page} a page"
            assert store.key_reads_of(run.id) == [CASES * K], "one pass over the run, for its keys"
            assert store.row_reads() == [], "no read of the run hands out its rows"
            assert max(store.id_reads) <= page and sum(store.id_reads) == CASES * K, "a page at a time, each once"
            assert len(store.id_reads) == -(-CASES * K // page)

    def test_at_the_real_page_no_read_hands_out_more(self) -> None:
        store, host = _host()
        cases = RUN_RESULTS_PAGE + RUN_RESULTS_PAGE // 2
        run = _large_run(host.storage, cases=cases, k=2)

        store.clear()

        summarized = summarize_run(host, run.id, SCOPE)

        assert summarized.n_results == cases * 2
        assert store.row_reads() == [] and store.key_reads_of(run.id) == [cases * 2]
        assert store.id_reads == [RUN_RESULTS_PAGE] * 3

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
            store.clear()

            paged = summarize_run(host, run.id, TOYHOST_SCOPE).model_dump_json()

            assert paged == whole
            assert store.row_reads() == [] and len(store.key_reads_of(run.id)) == 1
            assert set(store.id_reads) == {1}, "one result a read"


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
        # Priced from the template's history, as Discodon's launches are: the replayed launch reads it, this run among it.
        launch = dataclasses.replace(holder.launch, launch_pricer=history_launch_pricer(holder.eval_host))
        monkeypatch.setattr(resume, "RUN_RESULTS_PAGE", 10_000)
        whole, refused = await take_up_arms(launch, [("arm", run_id)], TOYHOST_SCOPE)
        assert not refused

        # Every walk the take-up makes one result a page: its own read of the run, and the replayed launch's pricer
        # reading the template's history, this run among it.
        monkeypatch.setattr(resume, "RUN_RESULTS_PAGE", 1)
        monkeypatch.setattr(lenses, "RUN_RESULTS_PAGE", 1)
        stored = off_loop(holder.eval_host.storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)
        store.clear()
        paged, refused = await take_up_arms(launch, [("arm", run_id)], TOYHOST_SCOPE)

        assert not refused
        assert paged["arm"].stored == whole["arm"].stored
        assert paged["arm"].rows == whole["arm"].rows
        assert paged["arm"].stored.done and paged["arm"].stored.replaced, "done cells and cells run again alike"
        assert len(paged["arm"].rows) == len(stored)
        assert store.row_reads() == [], "no read the take-up makes hands out a run's rows by type, nor the scope's"
        assert len(store.key_reads_of(run_id)) == 2, "its keys read once by the take-up, once by the pricer"
        assert store.id_reads and set(store.id_reads) == {1}, "every result it reads, one at a time"
        assert store.id_reads.count(1) >= 2 * len(stored)

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
        store.clear()
        (paged,) = resumable_jobs(storage, [SCOPE]).jobs

        assert paged == whole
        assert paged.cells_done == CASES * K - len(range(0, CASES * K, 4))
        assert store.row_reads() == [] and store.key_reads_of(run.id) == [CASES * K]
        assert max(store.id_reads) <= 2 * K and sum(store.id_reads) == CASES * K


class TestEveryShippedStoreServesTheTwoReads:
    @pytest.mark.parametrize("kind", ["memory", "sqlite"])
    def test_the_keys_then_the_results_by_id_summarise_as_one_read(
        self, kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = InMemoryDocumentStore() if kind == "memory" else SqliteDocumentStore(tmp_path / "evals.sqlite")
        host = toyhost_host(storage=EvalStorage(store))
        run = _large_run(host.storage, cases=12)

        keys = host.storage.query_eval_result_keys_by_run(run.id, SCOPE)

        assert sorted(key.id for key in keys) == sorted(
            f"{run.id}-{case}-{k}" for case in range(12) for k in range(1, K + 1)
        )
        assert {key.k_iteration for key in keys} == set(range(1, K + 1)) and all(key.scored_at for key in keys)
        assert len(host.storage.load_eval_results([key.id for key in keys[:5]], SCOPE)) == 5
        monkeypatch.setattr(summary, "RUN_RESULTS_PAGE", 12 * K)
        whole = summarize_run(host, run.id, SCOPE).model_dump_json()
        monkeypatch.setattr(summary, "RUN_RESULTS_PAGE", 5)
        assert summarize_run(host, run.id, SCOPE).model_dump_json() == whole
