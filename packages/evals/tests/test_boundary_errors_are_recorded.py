"""A boundary that contains an error writes it where an operator reads it, when there is a record to write it on.

Some handlers are isolation boundaries: they keep an error from changing an outcome that is not theirs to change (a
run that executed its matrix, a job whose lease machinery failed). Each used to put the error in the log and nowhere
else. Where a run or job record exists at that point, the error is now also written on it — ``EvalRun.error_details``
for a run, the loss's cause for a job record ended lost — naming where it happened, its kind and its message. What
the caller receives is unchanged, and a record write that fails as well is logged with both errors.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel

from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink, VariantConfig
from threetears.evals.kernel.cassettes import ActionSeam, CellCassettes, DeliverySeam, ToolWrap
from threetears.evals.kernel.errors import ConflictError, StorageError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.scoring import CellSummary
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.lifecycle import record_completeness
from threetears.evals.run.runner import RunnerOptions, execute_run
from threetears.evals.schema.models import EvalCassette, EvalRun, JudgedArtifact
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import DEFAULT_KIND, make_eval_result, make_eval_run, make_template, make_test_case
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
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


# =============================================================================
# execute_run: what its loop contained is written on the run as the loop ends
# =============================================================================


class _Roll(BaseModel):
    total: int


# parity-with: threetears.evals.kernel.cassettes.ToolLike
class _FakeDice:
    """One recordable tool: a roll."""

    @property
    def name(self) -> str:
        return "roll"

    def can_dispatch(self, action: str) -> bool:
        return action == "roll"

    async def act(self, action: str, parameters: dict[str, Any]) -> _Roll:
        await asyncio.sleep(0)
        return _Roll(total=7)


# parity-with: threetears.evals.kernel.cassettes.CassetteSeams
class _FakeSession:
    """A prepared candidate exposing its one tool through an action seam."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {"roll": _FakeDice()}

    @property
    def action_seam(self) -> ActionSeam | None:
        return self  # type: ignore[return-value]

    @property
    def delivery_seams(self) -> Mapping[str, DeliverySeam]:
        return {}

    @property
    def recorded_tools(self) -> Mapping[str, type[_Roll]]:
        return {"roll": _Roll}

    def arm_tools(self, wrap: ToolWrap) -> None:
        self.tools = wrap(self.tools)


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeRollingKind:
    """Wires whatever cassettes it is handed, then rolls once per cell."""

    judged_artifact = JudgedArtifact.UNJUDGED

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: Any,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> _FakeSession:
        session = _FakeSession()
        if cassettes is not None:
            cassettes.wire(session)
        return session

    async def invoke(self, instance: _FakeSession, test_case: Any, sink: CellSink) -> CandidateOutput:
        roll = await instance.tools["roll"].act("roll", {"dice": "1d20"})
        return CandidateOutput(output=[{"roll": roll.total}])


class _CassetteRefusingStorage(EvalStorage):
    """An eval store that refuses every cassette recording, and keeps everything else."""

    def save_cassette(self, cassette: EvalCassette) -> None:
        """Refuse the recording."""
        raise StorageError(f"the store refused recording {cassette.id}")


_TEMPLATE = make_template(id="tpl-roll", candidate_kind=DEFAULT_KIND, goal_state_checks=[])
_CASES = [make_test_case(id="case-1", template_id=_TEMPLATE.id)]


async def _execute(host: EvalHost, run: EvalRun, options: RunnerOptions) -> EvalRun:
    host.storage.save_eval_run(run)
    await execute_run(host, run=run, template=_TEMPLATE, test_cases=_CASES, judge_service=None, options=options)
    # The etag read, which settles nothing: the run here carries no lease, which a settling read would write lost.
    stored, _ = host.storage.load_eval_run_with_etag(run.id, run.scope_id)
    assert stored is not None
    return stored


def _rolling_run(**fields: Any) -> EvalRun:
    return make_eval_run(
        template_id=_TEMPLATE.id, test_case_ids=[case.id for case in _CASES], candidate_kind=DEFAULT_KIND, **fields
    )


async def test_a_capture_recording_that_is_not_written_is_named_on_the_capture_run() -> None:
    """Before: the corpus hole was a log line; a later replay found it as a miss. Now the capture run says so."""
    host = toyhost_host(storage=_CassetteRefusingStorage(InMemoryDocumentStore()))
    run = _rolling_run(cassette_mode="capture")

    stored = await _execute(host, run, RunnerOptions(candidate_kinds={DEFAULT_KIND: lambda _cell: _FakeRollingKind()}))

    [entry] = stored.error_details
    assert entry.startswith("cassette capture: recording ")
    assert "this run's corpus is incomplete" in entry
    recording = entry.split()[3]
    assert f"re-capture before replaying it: StorageError: the store refused recording {recording}" in entry
    assert stored.status == run.status, "the run's status is the job manager's, untouched"


async def test_a_raising_concurrency_probe_is_named_on_the_run_with_the_samples_it_cost() -> None:
    """Before: each raise was logged and the cells read execution_mode unmeasured, with nothing on the run."""
    host = toyhost_host()
    run = _rolling_run()

    def probe() -> int:
        raise RuntimeError("the job manager is gone")

    options = RunnerOptions(
        candidate_kinds={DEFAULT_KIND: lambda _cell: _FakeRollingKind()}, concurrent_eval_jobs_probe=probe
    )
    stored = await _execute(host, run, options)

    [entry] = stored.error_details
    assert entry.startswith("concurrency probe (RunnerOptions.concurrent_eval_jobs_probe) raised")
    assert ": RuntimeError: the job manager is gone (" in entry and entry.endswith(" samples)")
    [result] = host.storage.query_eval_results_by_run(run.id, run.scope_id)
    assert "execution_mode" not in result.covariates, "the cell still reads unmeasured, as it did"


async def test_a_run_whose_loop_contained_nothing_carries_no_entry() -> None:
    host = toyhost_host()
    run = _rolling_run(cassette_mode="capture")

    stored = await _execute(host, run, RunnerOptions(candidate_kinds={DEFAULT_KIND: lambda _cell: _FakeRollingKind()}))

    assert stored.error_details == []
