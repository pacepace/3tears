"""An analysis generation is priced, capped and ledgered like every other call the engine makes outside a run.

``analysis_generate`` is a ``spend`` action, and its calls run after the runs it reads, under no run's cost
cap. So the generation is held to the host's out-of-run cap (``LaunchSettings.max_out_of_run_cost_usd``) by
the one out-of-run rule (``OutOfRunBudget``): every call priced on the client before it is sent, refused
when the price would pass what is left of the cap, and ledgered under purpose ``analysis`` however it ends.
Pinned here:

- **The first call is priced in the job, before it is sent**: over the cap, or unpriceable under it, the job ends
  refused with nothing spent and the client it built released (a behaviour change: the start used to refuse it in its
  call, which meant assembling the whole bundle inside the host's call budget). The start itself returns without
  assembling anything.
- **The repair round-trip is priced when it exists**: a refused first output buys a repair, which is
  refused before it is sent when it would pass what is left of the cap — the attempt records only the call
  that was made.
- **Every call is ledgered**, so ``scope_out_of_run_spend`` (operation and action) reads the generation's
  spend, under purpose ``analysis`` and the campaign it was for; with enforcement off the call is still
  ledgered, under no cap.
- **``analysis_estimate`` is the start's own answer**, priced the same way, making no call.
- **No store call runs on the event loop**: assembling the bundle, the analysis, its record, its insights
  and every ledger row are written on the host's blocking executor; so are a run cancel's reads and repair.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import service as service_module
from threetears.evals.schema import JobLease, JobRefusal
from threetears.evals.ops import (
    analysis_estimate,
    analysis_generate,
    format_number,
    generation_key,
    job_cancel,
    job_poll,
    run_job_id,
    scope_out_of_run_spend,
)
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, MemoWriter, ops_fixture, settled
from packages.evals.tests.toyhost_memo import FIXTURED_CALL_CEILING_USD
from packages.evals.tests.off_loop import off_loop

#: The out-of-run cap the toy host holds a generation to.
CAP = TOYHOST_LAUNCH_SETTINGS.max_out_of_run_cost_usd


def _ledger(fixture: Any) -> list[Any]:
    return scope_out_of_run_spend(fixture.host.eval_host, TOYHOST_SCOPE, purpose="analysis").rows


# =============================================================================
# The first call: priced in the job, before it is sent
# =============================================================================


@pytest.mark.parametrize(
    ("ceiling", "said"),
    [(CAP + 1.0, "above the out-of-run cap"), (None, "cannot be priced before they are made")],
    ids=["over-the-cap", "unpriceable"],
)
async def test_a_generation_over_the_cap_ends_its_job_refused_before_any_call(ceiling: float | None, said: str) -> None:
    fixture = off_loop(ops_fixture, ceiling_usd=ceiling)

    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    status = await settled(fixture.host, job.job_id)

    assert status.state == "failed" and status.detail is not None
    assert "refused before any call" in status.detail and said in status.detail
    assert status.refusal == JobRefusal(kind="validation_failed", status=422)
    (writer,) = fixture.writers
    assert writer.calls == [], "nothing was sent"
    assert writer.closed == 1, "the client it built was released"
    await fixture.host.launch.job_manager.wait_for([job.job_id.rpartition(":")[2]])
    assert fixture.host.launch.job_manager.active_task_ids(generation_key(fixture.campaign.id, TOYHOST_SCOPE)) == []
    (attempt,) = off_loop(
        fixture.host.eval_host.storage.list_analysis_attempts_by_campaign, fixture.campaign.id, TOYHOST_SCOPE
    )
    assert (attempt.outcome, attempt.calls, attempt.token_cost) == ("failed", 0, 0.0)
    assert attempt.bundle_fingerprint, "the bundle was assembled before the price refused it"
    assert off_loop(_ledger, fixture) == []


async def test_a_start_returns_its_job_without_assembling_the_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bundle is assembled in the job: a start whose assembly cannot finish still returns, and the job then runs."""
    fixture = off_loop(ops_fixture)
    release = threading.Event()
    assembling: list[str] = []
    assemble = service_module.assemble_context_bundle

    def held_assembly(campaign: Any, **kwargs: Any) -> Any:
        assembling.append(campaign.id)
        assert release.wait(timeout=10), "the test never released the assembly"
        return assemble(campaign, **kwargs)

    monkeypatch.setattr(service_module, "assemble_context_bundle", held_assembly)

    started = await asyncio.wait_for(analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE), timeout=5)

    assert assembling == [], "the start assembled the bundle before it returned"
    assert fixture.writers == [], "the start built a client before it returned"
    (job,) = started.jobs
    running = await job_poll(fixture.host, job.job_id, TOYHOST_SCOPE)
    assert (running.state, running.done) == ("running", False)
    release.set()
    assert (await settled(fixture.host, job.job_id)).state == "completed"
    assert assembling == [fixture.campaign.id]


async def test_a_generation_within_the_cap_is_ledgered_and_read_by_the_spend_lens_and_action() -> None:
    fixture = off_loop(ops_fixture)
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"

    (row,) = off_loop(_ledger, fixture)
    assert (row.purpose, row.campaign_id, row.subject_id, row.outcome) == (
        "analysis",
        fixture.campaign.id,
        fixture.campaign.subject_id,
        "completed",
    )
    assert (row.priced_ceiling_usd, row.cap_usd) == (FIXTURED_CALL_CEILING_USD, CAP)
    assert row.cost_usd is not None and row.cost_usd > 0
    report = off_loop(scope_out_of_run_spend, fixture.host.eval_host, TOYHOST_SCOPE)
    assert report.by_purpose["analysis"].priced_usd == pytest.approx(row.cost_usd)

    evals = eval_catalogue().mount_all(standard_tools())[0]
    outcome = await evals.call(
        {"action": "scope_out_of_run_spend", "purpose_filter": "analysis"}, host=fixture.host, caller=CALLER
    )
    assert not outcome.is_error, outcome.text
    assert "purpose analysis:" in outcome.text and f"campaign {fixture.campaign.id}" in outcome.text


async def test_with_enforcement_off_a_generation_is_still_ledgered_under_no_cap() -> None:
    off = TOYHOST_LAUNCH_SETTINGS.model_copy(update={"enforcement_enabled": False})
    fixture = off_loop(ops_fixture, ceiling_usd=None, settings=off)
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"

    (row,) = off_loop(_ledger, fixture)
    assert (row.cap_usd, row.priced_ceiling_usd) == (None, None)


# =============================================================================
# The repair round-trip: priced when its prompt exists, against what is left
# =============================================================================


async def test_a_repair_that_would_pass_the_cap_is_refused_before_it_is_sent() -> None:
    """The first call fits; its output is refused; the repair would pass what is left, so it is never sent."""
    ceiling = CAP * 0.6
    fixture = off_loop(ops_fixture, ceiling_usd=ceiling, first_output="not json at all")
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    status = await settled(fixture.host, job.job_id)

    (writer,) = fixture.writers
    assert len(writer.calls) == 1, "the repair was refused before it was sent"
    assert status.state == "failed" and status.detail is not None
    assert "the repair round-trip was refused before it was sent" in status.detail
    assert "above the out-of-run cap" in status.detail
    (attempt,) = off_loop(
        fixture.host.eval_host.storage.list_analysis_attempts_by_campaign, fixture.campaign.id, TOYHOST_SCOPE
    )
    assert attempt.calls == 1 and attempt.unpriced_calls == 0, "only the call that was made is counted"
    assert [row.outcome for row in off_loop(_ledger, fixture)] == ["completed"]


async def test_a_repair_within_what_is_left_is_sent_and_both_calls_are_ledgered() -> None:
    fixture = off_loop(ops_fixture, ceiling_usd=CAP * 0.4, first_output="not json at all")
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"

    assert len(fixture.writers[0].calls) == 2
    assert [row.outcome for row in off_loop(_ledger, fixture)] == ["completed", "completed"]


# =============================================================================
# The estimate: the start's own answer, making no call
# =============================================================================


@pytest.mark.parametrize(("ceiling", "would_start"), [(FIXTURED_CALL_CEILING_USD, True), (CAP + 1.0, False)])
async def test_an_estimate_prices_the_first_call_against_the_cap_and_makes_no_call(
    ceiling: float, would_start: bool
) -> None:
    fixture = off_loop(ops_fixture, ceiling_usd=ceiling)

    estimate = await analysis_estimate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)

    assert (estimate.first_call_ceiling_usd, estimate.cap_usd, estimate.would_start) == (ceiling, CAP, would_start)
    assert (estimate.refusal is None) is would_start
    assert estimate.max_calls == 2
    (writer,) = fixture.writers
    assert writer.calls == [] and writer.closed == 1, "priced, released, never called"
    assert off_loop(_ledger, fixture) == []
    if not would_start:
        (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
        status = await settled(fixture.host, job.job_id)
        assert estimate.refusal is not None
        assert status.state == "failed" and status.detail == f"refused before any call: {estimate.refusal}", (
            "the estimate refuses in the job's own words"
        )


async def test_the_estimate_action_renders_the_price_and_the_cap() -> None:
    fixture = off_loop(ops_fixture)
    evals = eval_catalogue().mount_all(standard_tools())[0]

    outcome = await evals.call(
        {"action": "analysis_estimate", "campaign_id": fixture.campaign.id}, host=fixture.host, caller=CALLER
    )

    assert not outcome.is_error, outcome.text
    assert {
        f"first_call_ceiling_usd: {format_number(FIXTURED_CALL_CEILING_USD)}",
        f"cap_usd: {format_number(CAP)}",
        "would_start: True",
    } <= set(outcome.text.splitlines())


# =============================================================================
# No store call on the event loop
# =============================================================================


#: Set while the fixture's writer composes its memo: it re-reads the store to write over the bundle, and that
#: read is the test double's, not the engine's, so it is not what this file is about.
_IN_THE_WRITER = threading.local()


def _record_threads(storage: Any, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> list[tuple[str, int]]:
    seen: list[tuple[str, int]] = []
    for name in names:
        original = getattr(storage, name)

        def recording(*args: Any, __name: str = name, __original: Any = original, **kwargs: Any) -> Any:
            if not getattr(_IN_THE_WRITER, "active", False):
                seen.append((__name, threading.get_ident()))
            return __original(*args, **kwargs)

        monkeypatch.setattr(storage, name, recording)
    return seen


async def test_a_generation_makes_no_store_call_on_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = off_loop(ops_fixture)
    names = (
        "load_campaign",
        "load_eval_runs",
        "query_eval_results_by_runs",
        "query_insights",
        "save_analysis",
        "save_analysis_attempt",
        "load_job_record",
        "save_job_record",
        "save_insight",
        "save_out_of_run_spend",
    )
    seen = _record_threads(fixture.host.eval_host.storage, monkeypatch, names)
    loop_thread = threading.get_ident()
    composing = MemoWriter.generate

    async def writer(self: MemoWriter, **kwargs: Any) -> Any:
        _IN_THE_WRITER.active = True
        try:
            return await composing(self, **kwargs)
        finally:
            _IN_THE_WRITER.active = False

    monkeypatch.setattr(MemoWriter, "generate", writer)

    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"

    called = {name for name, _ in seen}
    assert set(names) - {"save_insight"} <= called, "every read and write the generation makes was recorded"
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
