"""A generation's spend is admitted, ledgered and read like every other call the engine makes outside a run.

``analysis_generate`` is a ``spend`` action. It runs the report writer, every turn admitted against the host's report
budget (``AnalysisGeneration.report_budget``), the only admission: the out-of-run cap is not read. Pinned here:

- **The report writer's spend is ledgered once**, when the writing ends, under purpose ``analysis`` and the campaign,
  so ``scope_out_of_run_spend`` (operation and action) reads it; its preparation refusals end the job before any
  turn, and the start returns before the job has done anything.
- **The estimate**: ``analysis_estimate`` states the report budget and the most turns, making no call.
- **No store call runs on the event loop**: the freeze, the analysis, its record, its insights and every ledger row
  are written on the host's blocking executor; so are a run cancel's reads and repair.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.schema import JobLease, JobRefusal
from threetears.evals.ops import (
    ANALYSIS_PREPARATION_BUDGET_S,
    AnalysisGeneration,
    WriterLimits,
    analysis_estimate,
    format_number,
    generation_key,
    job_cancel,
    job_poll,
    run_job_id,
    scope_out_of_run_spend,
)
from packages.evals.tests.doc_writer_support import FakeAnalystClient
from packages.evals.tests.ops_support import (
    CALLER,
    REPORT_BUDGET_USD,
    TOYHOST_SCOPE,
    generate,
    ops_fixture,
    settled,
)
from packages.evals.tests.off_loop import off_loop
from threetears.evals.kernel.inner_ceilings import writer_turns


def _ledger(fixture: Any) -> list[Any]:
    return scope_out_of_run_spend(fixture.host.eval_host, TOYHOST_SCOPE, purpose="analysis").rows


def _attempts(fixture: Any) -> list[Any]:
    return off_loop(
        fixture.host.eval_host.storage.list_analysis_attempts_by_campaign, fixture.campaign.id, TOYHOST_SCOPE
    )


# =============================================================================
# The report writer: admitted against the report budget, ledgered once
# =============================================================================


async def test_a_generation_is_ledgered_once_and_read_by_the_spend_lens_and_action() -> None:
    fixture = off_loop(ops_fixture)
    (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"

    (row,) = off_loop(_ledger, fixture)
    assert (row.purpose, row.campaign_id, row.subject_id, row.outcome) == (
        "analysis",
        fixture.campaign.id,
        fixture.campaign.subject_id,
        "completed",
    )
    assert row.cap_usd is None, "the report budget admits the turns; the out-of-run cap is not read"
    assert row.cost_usd == pytest.approx(0.02), "the analyst's turn and the editor's, each priced at a cent"
    report = off_loop(scope_out_of_run_spend, fixture.host.eval_host, TOYHOST_SCOPE)
    assert report.by_purpose["analysis"].priced_usd == pytest.approx(row.cost_usd)

    evals = eval_catalogue().mount_all(standard_tools())[0]
    outcome = await evals.call(
        {"action": "scope_out_of_run_spend", "purpose_filter": "analysis"}, host=fixture.host, caller=CALLER
    )
    assert not outcome.is_error, outcome.text
    assert "purpose analysis:" in outcome.text and f"campaign {fixture.campaign.id}" in outcome.text


async def test_a_start_returns_its_job_before_the_job_has_built_a_client_or_frozen_anything() -> None:
    gate = asyncio.Event()
    fixture = off_loop(ops_fixture, gate=gate)

    started = await asyncio.wait_for(generate(fixture.host, fixture.campaign.id), timeout=5)

    (job,) = started.jobs
    running = await job_poll(fixture.host, job.job_id, TOYHOST_SCOPE)
    assert (running.state, running.done) == ("running", False)
    gate.set()
    assert (await settled(fixture.host, job.job_id)).state == "completed"
    assert fixture.writers == [], "no planner client was built"


async def test_a_pricing_refusal_ends_the_job_without_an_error_record(caplog: pytest.LogCaptureFixture) -> None:
    """A refusal is a designed ending the attempt records: the manager's task-failure log is for what nobody designed."""
    fixture = off_loop(ops_fixture, analyst_unpriced="no rate for toy-analyst-model")

    with caplog.at_level(logging.DEBUG):
        (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
        status = await settled(fixture.host, job.job_id)
        await fixture.host.launch.job_manager.wait_for([job.job_id.rpartition(":")[2]])

    assert status.state == "failed" and status.detail is not None
    assert status.detail.startswith("refused before any call:") and "no rate for toy-analyst-model" in status.detail
    assert status.refusal == JobRefusal(kind="validation_failed", status=422)
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR] == []
    assert not [record for record in caplog.records if record.exc_info], "no traceback was logged for a refusal"
    (analyst,) = fixture.analysts
    assert analyst.sessions == {} and analyst.released, "nothing was sent, and the client was released"
    assert off_loop(_ledger, fixture) == []


async def test_a_preparation_that_never_finishes_ends_the_attempt_failed_and_releases_its_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = off_loop(ops_fixture)
    budget_s = 0.2
    assert fixture.host.generation is not None
    host = dataclasses.replace(
        fixture.host, generation=dataclasses.replace(fixture.host.generation, preparation_budget_s=budget_s)
    )

    async def never_priced(self: FakeAnalystClient) -> str | None:
        await asyncio.Event().wait()
        return None

    monkeypatch.setattr(FakeAnalystClient, "turn_pricing_refusal", never_priced)

    (job,) = (await generate(host, fixture.campaign.id)).jobs
    status = await settled(host, job.job_id)

    assert status.state == "failed"
    assert status.detail == f"refused before any call: preparation did not finish in {format_number(budget_s)} s"
    assert [(analyst.sessions, analyst.released) for analyst in fixture.analysts] == [({}, True)]
    await fixture.host.launch.job_manager.wait_for([job.job_id.rpartition(":")[2]])
    assert fixture.host.launch.job_manager.active_task_ids(generation_key(fixture.campaign.id, TOYHOST_SCOPE)) == []


async def test_a_generation_without_a_preparation_budget_runs_under_the_default() -> None:
    fixture = off_loop(ops_fixture)
    assert fixture.host.generation is not None
    assert fixture.host.generation.preparation_budget_s == ANALYSIS_PREPARATION_BUDGET_S == 900.0
    (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"


@pytest.mark.parametrize("budget_s", [0.0, -1.0, float("inf"), float("nan")])
def test_a_preparation_budget_that_is_not_positive_and_finite_is_refused(budget_s: float) -> None:
    with pytest.raises(ValueError, match="preparation_budget_s must be positive and finite"):
        AnalysisGeneration(budget_s=1.0, preparation_budget_s=budget_s)


async def test_an_ended_attempt_keeps_the_start_its_call_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = off_loop(ops_fixture)
    release = asyncio.Event()
    priced = FakeAnalystClient.turn_pricing_refusal

    async def held(self: FakeAnalystClient) -> str | None:
        await release.wait()
        return await priced(self)

    monkeypatch.setattr(FakeAnalystClient, "turn_pricing_refusal", held)

    (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
    (running,) = off_loop(_attempts, fixture)
    await asyncio.sleep(0.05)  # the preparation takes time the start must not be moved by
    release.set()
    assert (await settled(fixture.host, job.job_id)).state == "completed"

    (ended,) = off_loop(_attempts, fixture)
    assert (ended.id, ended.outcome) == (running.id, "stored")
    assert ended.started_at == running.started_at


async def test_a_client_is_released_when_the_report_budget_cannot_be_built() -> None:
    fixture = off_loop(ops_fixture)
    assert fixture.host.generation is not None

    def cannot_build() -> Any:
        raise RuntimeError("no budget")

    host = dataclasses.replace(
        fixture.host, generation=dataclasses.replace(fixture.host.generation, report_budget=cannot_build)
    )
    (job,) = (await generate(host, fixture.campaign.id)).jobs
    status = await settled(host, job.job_id)

    assert status.state == "failed" and status.detail is not None and "no budget" in status.detail
    assert [analyst.released for analyst in fixture.analysts] == [True]


async def test_the_estimate_states_the_report_budget_and_the_most_turns_and_makes_no_call() -> None:
    fixture = off_loop(ops_fixture)

    estimate = await analysis_estimate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)

    assert (estimate.first_call_ceiling_usd, estimate.cap_usd, estimate.would_start) == (None, REPORT_BUDGET_USD, True)
    assert estimate.max_calls == writer_turns(WriterLimits())
    (analyst,) = fixture.analysts
    assert analyst.sessions == {} and analyst.released, "asked, released, never sent a turn"
    assert off_loop(_ledger, fixture) == []


async def test_an_estimate_refuses_in_the_jobs_own_words() -> None:
    fixture = off_loop(ops_fixture, analyst_unpriced="no rate for toy-analyst-model")
    estimate = await analysis_estimate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)
    assert not estimate.would_start and estimate.refusal is not None
    (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
    status = await settled(fixture.host, job.job_id)
    assert status.detail is not None and estimate.refusal in status.detail


async def test_the_estimate_action_renders_the_budget_and_the_most_turns() -> None:
    fixture = off_loop(ops_fixture)
    evals = eval_catalogue().mount_all(standard_tools())[0]

    outcome = await evals.call(
        {"action": "analysis_estimate", "campaign_id": fixture.campaign.id}, host=fixture.host, caller=CALLER
    )

    assert not outcome.is_error, outcome.text
    assert {
        f"cap_usd: {format_number(REPORT_BUDGET_USD)}",
        f"max_calls: {writer_turns(WriterLimits())}",
        "would_start: True",
    } <= set(outcome.text.splitlines())


# =============================================================================
# No store call on the event loop
# =============================================================================


def _record_threads(storage: Any, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> list[tuple[str, int]]:
    seen: list[tuple[str, int]] = []
    for name in names:
        original = getattr(storage, name)

        def recording(*args: Any, __name: str = name, __original: Any = original, **kwargs: Any) -> Any:
            seen.append((__name, threading.get_ident()))
            return __original(*args, **kwargs)

        monkeypatch.setattr(storage, name, recording)
    return seen


async def test_a_generation_makes_no_store_call_on_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = off_loop(ops_fixture)
    names = (
        "load_campaign",
        "load_eval_runs",
        "iter_result_readings",
        "load_eval_results",
        "query_insights",
        "save_report_snapshot",
        "load_report_snapshot",
        "save_report_doc",
        "save_analysis",
        "save_analysis_attempt",
        "load_job_record",
        "save_job_record",
        "save_insight",
        "save_out_of_run_spend",
    )
    seen = _record_threads(fixture.host.eval_host.storage, monkeypatch, names)
    loop_thread = threading.get_ident()

    (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"

    called = {name for name, _ in seen}
    assert set(names) <= called, "every read and write the generation makes was recorded"
    assert [name for name, thread in seen if thread == loop_thread] == []


async def test_cancelling_a_run_another_process_holds_makes_no_store_call_on_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    run = off_loop(storage.load_eval_run, fixture.campaign.run_ids[0], TOYHOST_SCOPE)
    assert run is not None
    held = JobLease(holder="another-process", expires_at="2999-01-01T00:00:00+00:00")
    off_loop(storage.save_eval_run, run.model_copy(update={"status": "running", "completed_at": None, "lease": held}))
    names = ("load_job_record", "merge_job_record_fields", "store_time")
    seen = _record_threads(storage, monkeypatch, names)
    loop_thread = threading.get_ident()

    asked = await job_cancel(fixture.host, run_job_id(run.id), TOYHOST_SCOPE, reason="stale")

    assert asked.state == "running", "the holder, elsewhere, ends it at its next renewal"
    assert {name for name, _ in seen} == set(names)
    assert [name for name, thread in seen if thread == loop_thread] == []
