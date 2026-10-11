"""A boundary that contains an error writes it where an operator reads it, when there is a record to write it on.

Some handlers are isolation boundaries: they keep an error from changing an outcome that is not theirs to change (a
run that executed its matrix, a job whose lease machinery failed). Each used to put the error in the log and nowhere
else. Where a run or job record exists at that point, the error is now also written on it — ``EvalRun.error_details``
for a run, the loss's cause for a job record ended lost — naming where it happened, its kind and its message. What
the caller receives is unchanged, and a record write that fails as well is logged with both errors.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from threetears.evals.kernel.errors import ConflictError
from threetears.evals.kernel.scoring import CellSummary
from threetears.evals.run.lifecycle import record_completeness
from threetears.evals.schema.models import EvalRun
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.job_support import InMemoryRunStore

# =============================================================================
# record_completeness: a record that is not written says so on the run
# =============================================================================


class _CompletenessRefusingStore(InMemoryRunStore):
    """Refuses every write of a run carrying a completeness record, as a store another writer keeps winning against."""

    def __init__(self, *, refuse_everything: bool = False) -> None:
        """Start with no runs; ``refuse_everything`` refuses the follow-up write too."""
        super().__init__()
        self.refuse_everything = refuse_everything

    def save_eval_run(self, run: EvalRun, *, if_match: str | None = None) -> None:
        """Refuse a completeness write (or every write); keep anything else."""
        if self.refuse_everything or run.completeness is not None:
            raise ConflictError(f"run {run.id} moved under the write")
        super().save_eval_run(run, if_match=if_match)


def _cells(run: EvalRun) -> list[CellSummary]:
    return [CellSummary.from_result(make_eval_result(eval_run_id=run.id, test_case_id="tc-1"), persisted=True)]


def test_a_completeness_record_that_is_not_written_says_so_on_the_run() -> None:
    """Before: the run's completeness read unknown, and why was only in the log. Now it is an ``error_details`` entry."""
    storage = _CompletenessRefusingStore()
    run = make_eval_run()
    storage.runs[run.id] = run

    record_completeness(storage, run.id, run.scope_id, _cells(run))

    stored = storage.runs[run.id]
    assert stored.completeness is None, "the record itself was refused, as the store was made to"
    [entry] = stored.error_details
    assert entry.startswith("completeness record not written"), entry
    assert "no attempt of its conditional write completed" in entry


def test_a_completeness_record_that_raises_names_the_error_on_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside the record's own write is named by its type and message on the run."""
    storage = InMemoryRunStore()
    run = make_eval_run()
    storage.runs[run.id] = run

    def broken(current: EvalRun, cells: Any) -> Any:
        raise ValueError("a cell summary this build cannot count")

    monkeypatch.setattr("threetears.evals.run.lifecycle.summarize_completeness", broken)

    record_completeness(storage, run.id, run.scope_id, _cells(run))

    [entry] = storage.runs[run.id].error_details
    assert entry == (
        "completeness record not written, so the run's completeness reads unknown: ValueError: a cell summary this "
        "build cannot count"
    )


def test_a_completeness_fault_that_cannot_be_written_either_is_logged_with_both_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The run is left as it was, nothing is raised into the work function, and the log carries both failures."""
    storage = _CompletenessRefusingStore(refuse_everything=True)
    run = make_eval_run()
    storage.runs[run.id] = run

    with caplog.at_level(logging.ERROR):
        record_completeness(storage, run.id, run.scope_id, _cells(run))

    assert storage.runs[run.id].error_details == []
    [line] = [r for r in caplog.records if "could not be written on the run either" in r.getMessage()]
    assert line.levelno == logging.ERROR
    assert "refused" in line.getMessage() and "completeness record not written" in line.getMessage()
