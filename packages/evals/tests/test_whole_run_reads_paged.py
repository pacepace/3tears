"""The rest of the readers of a whole run read it a page at a time: none hands out more than a page, and each gives
what one read of the whole run gave (#998).

#1000 paged a run's summary, a take-up of it and an export of one; these are the readers it left reading every stored
result of a run at once. Each now makes one read of the run's result keys (a few scalars a result) and reads its
results by id, :data:`~threetears.evals.kernel.run_pages.RUN_RESULTS_PAGE` at a time, in the store's order, so what
it gives is what the whole read gave. Pinned here through a store that records every read of results it serves, at a
page of :data:`PAGE` against a page holding the whole run:

- **runs_compare and runs_bisect** read each run a page at a time; the comparison's clock sentence reads the runs'
  keys alone.
- **A run written lost** counts what it stored a page at a time.
- **results_list** reads only the results its page lists when unfiltered, and every result a page at a time when
  filtered by condition; **list_results** reads the run a page at a time.
- **delete_run** reads the run's result ids off the key read, never a row.
- **The run_summary lens** reads its run a page at a time.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``reads.compare_two_runs`` / ``reads.bisect_runs``: either run read whole (``query_eval_results_by_run``).
- ``lenses._pair_disclosures``: the window read off the runs' results rather than their keys.
- ``job_leases._run_lost``: the completeness counted from a whole read.
- ``results.results_list``: a whole read; the unfiltered page reading every result.
- ``curation.delete_run``: the ids read off the run's results.
- ``run_pages.iter_keyed_results``: a page past ``RUN_RESULTS_PAGE``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from typing import Any

import pytest

from threetears.evals.analysis.reads import run_summary
from threetears.evals.kernel import EvalStorage, ResultOutcome, classify_result
from threetears.evals.kernel import run_pages
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.scoring import reconstruct_completeness
from threetears.evals.ops import job_poll, results_list, run_job_id, runs_bisect, runs_compare
from threetears.evals.run import delete_run, list_results
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.schema.models import RubricScore
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.run_pages_support import BatchRecordingStore

SCOPE = "uni-whole-run-reads-paged"
#: Each run's cases and repeats: 75 results, many pages at :data:`PAGE`.
CASES = 25
K = 3
#: The page every reader is held to here; a page holding a whole run is :data:`WHOLE`.
PAGE = 4
WHOLE = 10_000


def _results(run: EvalRun, *, shift: int) -> list[EvalResult]:
    """A row per cell of ``run``, varied in what the readers read, carrying a kind's payload, shuffled as stored."""
    rows = []
    for case in range(CASES):
        for iteration in range(1, K + 1):
            seed = case + iteration + shift
            failed = seed % 7 == 0
            rows.append(
                make_eval_result(
                    id=f"{run.id}-{case}-{iteration}",
                    eval_run_id=run.id,
                    scope_id=run.scope_id,
                    model=run.candidate_model,
                    test_case_id=f"tc-{case}",
                    k_iteration=iteration,
                    # Sums an order-dependent running total rounds differently from one exact sum.
                    cost_usd=(0.1, 0.2, 1 / 3, 1e-17, 7.25)[seed % 5],
                    rubric_scores=[
                        RubricScore(
                            dim="conversation.tone",
                            score=1 + seed % 5,
                            scale="ordinal",
                            axis="capability",
                            judge_temperature=0.0,
                        )
                    ],
                    kind_payload={"transcript": f"cell {case}/{iteration} " * 40},
                    candidate_error="the candidate refused" if failed else None,
                    scored_at=f"2026-10-{20 + shift:02d}T{case % 24:02d}:{iteration * 10:02d}:00+00:00",
                )
            )
    random.Random(shift).shuffle(rows)
    return rows


def _run(storage: EvalStorage, run_id: str, *, shift: int = 0) -> EvalRun:
    """A completed run of :data:`CASES` × :data:`K` stored results, of the template every run here shares."""
    run = make_eval_run(
        id=run_id,
        scope_id=SCOPE,
        status="completed",
        k_runs=K,
        test_case_ids=[f"tc-{case}" for case in range(CASES)],
    )
    storage.save_eval_run(run)
    for row in _results(run, shift=shift):
        storage.save_eval_result(row)
    return run


def _host() -> tuple[BatchRecordingStore, EvalHost]:
    store = BatchRecordingStore()
    return store, toyhost_host(storage=EvalStorage(store))


def _whole_then_paged[T](
    monkeypatch: pytest.MonkeyPatch, store: BatchRecordingStore, read: Callable[[], T]
) -> tuple[T, T]:
    """What ``read`` gives with the whole run in one page, then at :data:`PAGE`, the store's record cleared between."""
    monkeypatch.setattr(run_pages, "RUN_RESULTS_PAGE", WHOLE)
    whole = read()
    monkeypatch.setattr(run_pages, "RUN_RESULTS_PAGE", PAGE)
    store.clear()
    return whole, read()


def _held_to_a_page(store: BatchRecordingStore) -> None:
    assert store.row_reads() == [], "no read hands out a run's rows"
    assert store.id_reads and max(store.id_reads) <= PAGE, f"a page at a time: {store.id_reads}"


def _dumped(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


class TestTwoRunsAreReadSideBySideAPageAtATime:
    def test_a_comparison_reads_each_run_a_page_at_a_time_and_its_clock_off_their_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, host = _host()
        a, b = _run(host.storage, "run-a"), _run(host.storage, "run-b", shift=3)

        whole, paged = _whole_then_paged(monkeypatch, store, lambda: runs_compare(host, a.id, b.id, SCOPE))

        assert paged.model_dump_json() == whole.model_dump_json()
        arm = whole.comparison["comparison"]["arm"]
        assert arm["paired"] and arm["n_pairs"] == CASES and whole.measurement_window_disclosure, "the fixture reads"
        _held_to_a_page(store)
        assert sum(store.id_reads) == 2 * CASES * K, "each run's results read once"
        # One pass for the comparison's figures and one for the clock sentence, which reads no result.
        assert store.key_reads_of(a.id) == [CASES * K] * 2 and store.key_reads_of(b.id) == [CASES * K] * 2

    def test_a_bisection_reads_each_run_a_page_at_a_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store, host = _host()
        a, b = _run(host.storage, "run-a"), _run(host.storage, "run-b", shift=3)

        whole, paged = _whole_then_paged(monkeypatch, store, lambda: runs_bisect(host, a.id, b.id, SCOPE))

        assert paged.model_dump_json() == whole.model_dump_json()
        _held_to_a_page(store)
        assert sum(store.id_reads) == 2 * CASES * K


class TestARunWrittenLostCountsWhatItStoredAPageAtATime:
    async def test_its_completeness_is_the_whole_reads_and_no_read_passes_a_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = BatchRecordingStore()
        fixture = off_loop(ops_fixture, store=store)
        storage = fixture.host.eval_host.storage
        run = off_loop(storage.load_eval_run, fixture.campaign.run_ids[0], TOYHOST_SCOPE)
        assert run is not None
        stored = off_loop(storage.query_eval_results_by_run, run.id, TOYHOST_SCOPE)
        assert len(stored) > 1, "the run is more than a page"
        # A run left running with no lease reads lost, and is written lost with what it stored counted.
        off_loop(
            storage.save_eval_run,
            run.model_copy(update={"status": "running", "completed_at": None, "completeness": None}),
        )
        monkeypatch.setattr(run_pages, "RUN_RESULTS_PAGE", 1)
        store.clear()

        lost = await job_poll(fixture.host, run_job_id(run.id), TOYHOST_SCOPE)

        assert lost.state == "lost"
        assert store.row_reads() == [] and store.key_reads_of(run.id) == [len(stored)]
        assert store.id_reads == [1] * len(stored), "each result read once, one a read"
        written = off_loop(storage.load_eval_run, run.id, TOYHOST_SCOPE)
        assert written is not None and written.completeness == reconstruct_completeness(run, stored)


class TestAResultListingReadsOnlyWhatItLists:
    def test_unfiltered_it_reads_only_the_results_of_its_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store, host = _host()
        run = _run(host.storage, "run-a")

        whole, paged = _whole_then_paged(
            monkeypatch, store, lambda: results_list(host, run.id, SCOPE, offset=5, limit=9)
        )

        assert paged == whole and len(whole.results) == 9 and whole.total == CASES * K
        _held_to_a_page(store)
        assert sum(store.id_reads) == 9, "only the page's results are read"

    def test_filtered_it_reads_every_result_a_page_at_a_time_and_keeps_its_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, host = _host()
        run = _run(host.storage, "run-a")
        failed = [
            row
            for row in host.storage.query_eval_results_by_run(run.id, SCOPE)
            if classify_result(row) is ResultOutcome.CANDIDATE_FAIL
        ]
        assert len(failed) > 4, "the fixture holds a few failed cells"

        whole, paged = _whole_then_paged(
            monkeypatch,
            store,
            lambda: results_list(host, run.id, SCOPE, condition=ResultOutcome.CANDIDATE_FAIL, offset=1, limit=3),
        )

        assert paged == whole and whole.total == len(failed) and len(whole.results) == 3
        assert whole.next_offset == 4
        expected = sorted(failed, key=lambda row: (row.test_case_id, row.k_iteration, row.id))[1:4]
        assert [line.id for line in whole.results] == [row.id for row in expected]
        _held_to_a_page(store)
        assert sum(store.id_reads) == CASES * K

    def test_list_results_reads_the_run_a_page_at_a_time_in_the_stores_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, host = _host()
        run = _run(host.storage, "run-a")
        stored = host.storage.query_eval_results_by_run(run.id, SCOPE)
        store.clear()
        monkeypatch.setattr(run_pages, "RUN_RESULTS_PAGE", PAGE)

        listed = list_results(host.storage, run.id, SCOPE)

        assert listed == stored, "the results a whole read gives, in its order"
        _held_to_a_page(store)


class TestADeleteReadsOnlyTheRunsIds:
    def test_it_reads_the_ids_off_the_key_read_and_no_result(self) -> None:
        store, host = _host()
        run = _run(host.storage, "run-a")
        store.clear()

        removed = delete_run(host.storage, run, SCOPE, confirm=run.id)

        assert removed["results_deleted"] == CASES * K
        assert store.row_reads() == [] and store.id_reads == [], "no result read, whole or by id"
        assert store.key_reads_of(run.id) == [CASES * K], "the ids read with what grows with a cell left out"
        assert host.storage.query_eval_results_by_run(run.id, SCOPE) == []

    def test_a_result_this_build_cannot_reconstruct_is_deleted_with_its_readable_run(self) -> None:
        """The ids are read off the rows, so a row the model has outlived no longer refuses its run's delete."""
        store, host = _host()
        run = _run(host.storage, "run-a")
        (row,) = host.storage.load_eval_results([f"{run.id}-0-1"], SCOPE)
        store.upsert(row.to_dict() | {"id": "res-outlived", "turns_delivered": "several"})

        removed = delete_run(host.storage, run, SCOPE, confirm=run.id)

        assert removed["results_deleted"] == CASES * K + 1
        assert store.get("res-outlived", SCOPE) is None, "the unreadable row went with its run"
        assert host.storage.query_eval_result_ids_by_run(run.id, SCOPE) == []
        assert host.storage.load_eval_run(run.id, SCOPE) is None


class TestTheRunSummaryLensReadsItsRunAPageAtATime:
    def test_it_is_the_summary_a_whole_read_gives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store, host = _host()
        run = _run(host.storage, "run-a")

        def summarised() -> str:
            def listed(run_id: str, scope_id: str) -> EvalRun:
                found = host.storage.load_eval_run(run_id, scope_id)
                assert found is not None
                return found

            return _dumped(
                run_summary(
                    host.storage,
                    run.id,
                    SCOPE,
                    load_run_listed=listed,
                    row_columns=lambda results: {},
                    profile=toyhost_profile(),
                )
            )

        whole, paged = _whole_then_paged(monkeypatch, store, summarised)

        assert paged == whole and json.loads(whole)["rows"], "the fixture summarises"
        _held_to_a_page(store)
