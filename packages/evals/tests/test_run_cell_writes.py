"""A run's cell is one row, written once per holding of the run: the store's rule, not the runner's discipline.

A run's cell result takes an id derived from its coordinates (:func:`cell_result_id`), so the same cell written twice
is the same row, and :meth:`EvalStorage.save_run_cell` writes a run's cell by one rule:

- **Created only if absent**: a second create of the cell is refused, and the first row stands with its trace.
- **Replaced only from a lower epoch**: a row written under a lower lease epoch is replaced by a writer at a higher
  one, trace and all; a higher-epoch row is never replaced by a lower-epoch writer, and its trace is untouched.
- **Never into a run that is no longer the writer's**: a run written lost, or held at a later epoch than the
  result's, takes no row.

And the runner writes every cell of a run under the derived id.
"""

from __future__ import annotations

import pytest

from threetears.evals.kernel import EvalStorage, JobFenced
from threetears.evals.ops import LaunchArguments, run_job_id, run_launch
from threetears.evals.schema import EvalResult, EvalTrace, JobLease, JobLoss
from threetears.evals.schema.models import cell_result_id, eval_trace_doc_id
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_eval_trace
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture, settled

#: The one cell every storage-level test writes.
RUN_ID, CASE_ID, MODEL, K = "run-1", "tc-1", "sonnet", 1

#: Its derived id.
CELL_ID = cell_result_id(RUN_ID, CASE_ID, MODEL, K)


def _cell(*, lease_epoch: int, cost_usd: float, turn: str) -> tuple[EvalResult, EvalTrace]:
    """The cell's result at ``lease_epoch``, told apart by its cost, and a trace told apart by its one turn."""
    result = make_eval_result(id=CELL_ID, lease_epoch=lease_epoch, cost_usd=cost_usd)
    return result, make_eval_trace(result_id=CELL_ID, trace=[{"turn": 1, "candidate": turn}])


def _stored(storage: EvalStorage) -> tuple[EvalResult | None, EvalTrace | None]:
    return storage.load_eval_result(CELL_ID, "uni-1"), storage.load_eval_trace(CELL_ID, "uni-1")


def test_a_cells_result_id_is_the_same_for_the_same_cell_and_differs_for_any_other() -> None:
    assert cell_result_id(RUN_ID, CASE_ID, MODEL, K) == CELL_ID, "derived, so the same cell names the same row"
    others = {
        cell_result_id("run-2", CASE_ID, MODEL, K),
        cell_result_id(RUN_ID, "tc-2", MODEL, K),
        cell_result_id(RUN_ID, CASE_ID, "opus", K),
        cell_result_id(RUN_ID, CASE_ID, MODEL, 2),
        # Coordinates that would run together if they were joined with a separator.
        cell_result_id("run-1,tc-1", "", MODEL, K),
    }
    assert len(others) == 5 and CELL_ID not in others


def test_a_second_create_of_a_cell_is_refused_and_the_first_row_stands() -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    first, first_trace = _cell(lease_epoch=0, cost_usd=0.01, turn="first")
    storage.save_run_cell(first, first_trace)

    second, second_trace = _cell(lease_epoch=0, cost_usd=0.99, turn="second")
    with pytest.raises(JobFenced, match="already stored under lease epoch 0"):
        storage.save_run_cell(second, second_trace)

    row, trace = _stored(storage)
    assert row is not None and row.cost_usd == 0.01 and row.has_trace, "the first row stands"
    assert trace is not None and trace.trace == first_trace.trace, "and so does its trace"
    assert [result.id for result in storage.query_eval_results_by_run(RUN_ID, "uni-1")] == [CELL_ID], "one row"


def test_a_lower_epoch_row_is_replaced_by_a_higher_epoch_writer() -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    earlier, earlier_trace = _cell(lease_epoch=0, cost_usd=0.01, turn="cancelled mid-cell")
    storage.save_run_cell(earlier, earlier_trace)

    later, later_trace = _cell(lease_epoch=1, cost_usd=0.02, turn="ran to its end")
    storage.save_run_cell(later, later_trace)

    row, trace = _stored(storage)
    assert row is not None and (row.lease_epoch, row.cost_usd) == (1, 0.02), "the later holding's row replaced it"
    assert trace is not None and trace.trace == later_trace.trace, "with its trace"
    assert len(storage.query_eval_results_by_run(RUN_ID, "uni-1")) == 1


def test_a_higher_epoch_row_is_not_replaced_by_a_lower_epoch_writer_and_its_trace_is_untouched() -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    later, later_trace = _cell(lease_epoch=1, cost_usd=0.02, turn="the later holding's")
    storage.save_run_cell(later, later_trace)

    superseded, superseded_trace = _cell(lease_epoch=0, cost_usd=0.01, turn="the superseded holding's")
    with pytest.raises(JobFenced, match="already stored under lease epoch 1; a writer at epoch 0"):
        storage.save_run_cell(superseded, superseded_trace)

    row, trace = _stored(storage)
    assert row is not None and (row.lease_epoch, row.cost_usd) == (1, 0.02)
    assert trace is not None and trace.trace == later_trace.trace, "the refused result wrote no trace"


@pytest.mark.parametrize(
    "run_fields",
    [
        {"status": "cancelled", "lost": JobLoss(at="2026-10-10T12:00:00+00:00", cause="process shut down")},
        {
            "status": "running",
            "lease": JobLease(holder="the-later-holding", expires_at="2999-01-01T00:00:00+00:00", epoch=1),
        },
    ],
    ids=["written-lost", "held-at-a-later-epoch"],
)
def test_a_cell_is_not_written_into_a_run_that_is_no_longer_the_writers(run_fields: dict[str, object]) -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    run = make_eval_run(id=RUN_ID, **run_fields)
    storage.save_job_record(run.to_dict(), if_match=None)
    result, trace = _cell(lease_epoch=0, cost_usd=0.01, turn="finished after the run moved on")

    with pytest.raises(JobFenced, match=f"eval run '{RUN_ID}'"):
        storage.save_run_cell(result, trace)

    assert _stored(storage) == (None, None), "neither the row nor its trace was written"


def test_an_upsert_outside_a_run_cell_still_replaces_by_id() -> None:
    """A host re-recording a witnessed cell under its own id replaces the record: the cell rule is a run's alone."""
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_eval_result(
        make_eval_result(id="witnessed-1", cost_usd=0.01), make_eval_trace(result_id="witnessed-1")
    )
    storage.save_eval_result(make_eval_result(id="witnessed-1", cost_usd=0.02), None)

    row = storage.load_eval_result("witnessed-1", "uni-1")
    assert row is not None and row.cost_usd == 0.02
    assert storage.load_eval_trace("witnessed-1", "uni-1") is not None


async def test_the_runner_writes_every_cell_under_its_derived_id_and_its_runs_lease_epoch() -> None:
    fixture = off_loop(ops_fixture)
    arguments = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=[RUN_MODELS[0]]
    )
    (job,) = (await run_launch(fixture.host, arguments, TOYHOST_SCOPE)).jobs
    launched = await settled(fixture.host, job.job_id)
    (run_id,) = launched.run_ids
    assert (await settled(fixture.host, run_job_id(run_id))).status == "completed"

    storage = fixture.host.eval_host.storage
    run = off_loop(storage.load_eval_run, run_id, TOYHOST_SCOPE)
    results = off_loop(storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)
    assert run is not None and results and len(results) == len(run.test_case_ids) * run.k_runs
    for result in results:
        assert result.id == cell_result_id(run_id, result.test_case_id, result.model, result.k_iteration)
        assert result.lease_epoch == 0, "written under the run's first holding"
        if result.has_trace:
            trace = off_loop(storage.load_eval_trace, result.id, TOYHOST_SCOPE)
            assert trace is not None and trace.id == eval_trace_doc_id(result.id)
