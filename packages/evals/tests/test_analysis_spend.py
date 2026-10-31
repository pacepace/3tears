"""A generation's spend is admitted, ledgered and read like every other call the engine makes outside a run.

``analysis_generate`` is a ``spend`` action. It runs the report writer, every turn admitted against the host's report
budget (``AnalysisGeneration.report_budget``), the only admission: the out-of-run cap is not read. The report planner,
which the reporter kind still runs (``prepare_analysis_generation`` and ``run_analysis_generation``), is held to the
host's out-of-run cap (``LaunchSettings.max_out_of_run_cost_usd``) by the one out-of-run rule (``OutOfRunBudget``).
Pinned here:

- **The report writer's spend is ledgered once**, when the writing ends, under purpose ``analysis`` and the campaign,
  so ``scope_out_of_run_spend`` (operation and action) reads it; its preparation refusals end the job before any
  turn, and the start returns before the job has done anything.
- **The planner's first call is priced before it is sent**: over the cap, or unpriceable under it, the generation is
  refused with nothing spent and the client it built released.
- **Every planner call after the first is priced when it exists**: its structure and then its prose, and a refused
  reply of either buys a repair; each is refused before it is sent when it would pass what is left of the cap — the
  attempt records only the calls that were made. Every call is ledgered; with enforcement off, under no cap.
- **The estimates**: ``analysis_estimate`` states the report budget and the most turns, making no call; the planner's
  (``estimate_analysis_generation``) prices its first call against the cap.
- **No store call runs on the event loop**: the freeze, the analysis, its record, its insights and every ledger row
  are written on the host's blocking executor; so are a run cancel's reads and repair.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import estimate_analysis_generation
from threetears.evals.analysis.service import APPROX_INPUT_TOKENS_BASIS, CHARS_PER_TOKEN
from threetears.evals.kernel import ValidationFailedError
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
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.ops_support import (
    CALLER,
    REPORT_BUDGET_USD,
    TOYHOST_SCOPE,
    generate,
    ops_fixture,
    planned_generation,
    settled,
)
from packages.evals.tests.toyhost_memo import FIXTURED_CALL_CEILING_USD
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.toyhost_plan import PLAN_CALLS
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.analysis.planner import REPORT_INPUT_HEADING
from threetears.evals.kernel.inner_ceilings import MAX_GENERATION_CALLS, writer_turns

#: The out-of-run cap the toy host holds the planner to.
CAP = TOYHOST_LAUNCH_SETTINGS.max_out_of_run_cost_usd


async def _planner_prompt() -> str:
    return EVAL_REPORT_PLAN_DEFAULT


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
# The report planner (the reporter kind's path): its first call priced before it is sent
# =============================================================================


@pytest.mark.parametrize(
    ("ceiling", "said"),
    [(CAP + 1.0, "above the out-of-run cap"), (None, "cannot be priced before they are made")],
    ids=["over-the-cap", "unpriceable"],
)
async def test_a_planned_generation_over_the_cap_is_refused_before_any_call(ceiling: float | None, said: str) -> None:
    fixture = off_loop(ops_fixture, ceiling_usd=ceiling)

    with pytest.raises(ValidationFailedError, match=said):
        await planned_generation(fixture)

    (writer,) = fixture.writers
    assert writer.calls == [], "nothing was sent"
    assert writer.closed == 1, "the client it built was released"
    assert off_loop(_ledger, fixture) == []


async def test_a_planned_generation_within_the_cap_ledgers_every_call() -> None:
    fixture = off_loop(ops_fixture)
    await planned_generation(fixture)

    # The planner's two calls, its structure and its prose, each priced, admitted and ledgered.
    rows = off_loop(_ledger, fixture)
    assert len(rows) == PLAN_CALLS
    for row in rows:
        assert (row.purpose, row.campaign_id, row.subject_id, row.outcome) == (
            "analysis",
            fixture.campaign.id,
            fixture.campaign.subject_id,
            "completed",
        )
        assert (row.priced_ceiling_usd, row.cap_usd) == (FIXTURED_CALL_CEILING_USD, CAP)
        assert row.cost_usd is not None and row.cost_usd > 0


async def test_the_logged_input_size_is_a_ballpark_over_the_exact_first_request_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The size a planned generation discloses before its first call counts what it actually sends (#827): the system
    prompt, user message and response schema, at two characters a token, and the line says so.

    The old figure was ``chars/4`` over a differently rendered bundle, half or less of what a JSON-heavy call bills.
    """
    fixture = off_loop(ops_fixture)
    with caplog.at_level(logging.INFO, logger="threetears.evals"):
        await planned_generation(fixture)

    (writer,) = [each for each in fixture.writers if each.calls]
    sent, response_format = writer.calls[0], writer.response_formats[0]
    assert sent["user"].startswith(f"{REPORT_INPUT_HEADING}\n") and response_format is not None
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("eval.generate_analysis campaign=")]
    logged = int(line.split("approx_input_tokens=")[1].split(" ", 1)[0])

    sent_chars = len(sent["system"]) + len(sent["user"]) + len(json.dumps(response_format))
    assert logged == sent_chars // CHARS_PER_TOKEN
    assert f"({APPROX_INPUT_TOKENS_BASIS})" in line


async def test_with_enforcement_off_a_planned_generation_is_still_ledgered_under_no_cap() -> None:
    off = TOYHOST_LAUNCH_SETTINGS.model_copy(update={"enforcement_enabled": False})
    fixture = off_loop(ops_fixture, ceiling_usd=None, settings=off)
    await planned_generation(fixture)

    rows = off_loop(_ledger, fixture)
    assert [(row.cap_usd, row.priced_ceiling_usd) for row in rows] == [(None, None)] * PLAN_CALLS


async def test_a_prose_call_that_would_pass_the_cap_is_refused_before_it_is_sent() -> None:
    """The structure call fits and is accepted; its prose call would pass what is left, so it is never sent."""
    fixture = off_loop(ops_fixture, ceiling_usd=CAP * 0.6)
    with pytest.raises(ValidationFailedError) as refused:
        await planned_generation(fixture)

    (writer,) = fixture.writers
    assert len(writer.calls) == 1, "the prose call was refused before it was sent"
    assert "call 2 of the plan was refused before it was sent" in str(refused.value)
    assert "above the out-of-run cap" in str(refused.value)
    (attempt,) = off_loop(_attempts, fixture)
    assert (attempt.outcome, attempt.calls, attempt.analysis_id) == ("failed", 1, None)
    assert [row.outcome for row in off_loop(_ledger, fixture)] == ["completed"]
    assert off_loop(fixture.host.eval_host.storage.list_analyses_by_campaign, fixture.campaign.id, TOYHOST_SCOPE) == []


async def test_a_repair_that_would_pass_the_cap_is_refused_before_it_is_sent() -> None:
    """The first call fits; its output is refused; the repair would pass what is left, so it is never sent."""
    fixture = off_loop(ops_fixture, ceiling_usd=CAP * 0.6, first_output="not json at all")
    with pytest.raises(ValidationFailedError, match="call 2 of the plan was refused before it was sent"):
        await planned_generation(fixture)

    (writer,) = fixture.writers
    assert len(writer.calls) == 1, "the repair was refused before it was sent"
    (attempt,) = off_loop(_attempts, fixture)
    assert attempt.calls == 1 and attempt.unpriced_calls == 0, "only the call that was made is counted"
    assert [row.outcome for row in off_loop(_ledger, fixture)] == ["completed"]


async def test_a_repair_within_what_is_left_is_sent_and_every_call_is_ledgered() -> None:
    """A refused structure buys its repair; the repaired structure's prose follows: three calls, all within the cap."""
    fixture = off_loop(ops_fixture, ceiling_usd=CAP * 0.3, first_output="not json at all")
    await planned_generation(fixture)

    assert len(fixture.writers[0].calls) == PLAN_CALLS + 1
    assert [row.outcome for row in off_loop(_ledger, fixture)] == ["completed"] * (PLAN_CALLS + 1)


@pytest.mark.parametrize(("ceiling", "would_start"), [(FIXTURED_CALL_CEILING_USD, True), (CAP + 1.0, False)])
async def test_the_planners_estimate_prices_the_first_call_against_the_cap_and_makes_no_call(
    ceiling: float, would_start: bool
) -> None:
    fixture = off_loop(ops_fixture, ceiling_usd=ceiling)

    estimate = await estimate_analysis_generation(
        fixture.host.eval_host,
        fixture.campaign.id,
        TOYHOST_SCOPE,
        model=None,
        resolve_prompt=_planner_prompt,
        out_of_run_cap_usd=fixture.host.out_of_run_cap(),
    )

    assert (estimate.first_call_ceiling_usd, estimate.cap_usd, estimate.would_start) == (ceiling, CAP, would_start)
    assert (estimate.refusal is None) is would_start
    assert estimate.max_calls == MAX_GENERATION_CALLS == 4
    (writer,) = fixture.writers
    assert writer.calls == [] and writer.closed == 1, "priced, released, never called"
    assert off_loop(_ledger, fixture) == []


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
