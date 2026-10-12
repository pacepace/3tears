"""A job refused by an engine error keeps the refusal's kind and status on its record, and a poll serves them.

A refused CALL reaches a client as ``{"refusal": {"kind", "status"}}`` beside its text, so a host branches on the
kind. A job refused AFTER its call returned — a provider refusing the account its work calls on, a store failing —
ended ``failed`` with the refusal's words alone in ``detail``, so a host retrying a throttled job had to parse prose.
Its record now keeps the same pair (``refusal``, core v11), and ``job_poll`` serves it, in its JSON and its text.
"""

from __future__ import annotations

import asyncio

from pydantic import Field

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.kernel import ProviderRefusedError, StorageError
from threetears.evals.kernel.campaign import EvalAnalysisAttempt
from threetears.evals.ops import analysis_generate
from threetears.evals.run import EvalActionJob, action_job_work, start_job_work
from threetears.evals.schema import JobRefusal
from threetears.evals.schema.base import EvalBaseModel
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture, settled
from packages.evals.tests.off_loop import off_loop


class _Draft(EvalBaseModel):
    """What a drafting action produces."""

    text: str = Field(description="The draft.")


async def _failing_job(raised: Exception) -> tuple[OpsFixture, str]:
    fixture = off_loop(ops_fixture)
    storage, executor = fixture.host.eval_host.storage, fixture.host.eval_host.blocking_executor

    async def draft() -> _Draft:
        raise raised

    work = action_job_work(TOYHOST_SCOPE, action="rubric_propose", run=draft, budget_s=60.0)
    await start_job_work(fixture.host.launch.job_manager, work, store=storage, executor=executor)
    return fixture, f"action:{work.record_id}"


async def test_a_job_its_provider_refused_serves_the_refusals_kind_and_status() -> None:
    fixture, job_id = await _failing_job(ProviderRefusedError("the provider throttled the account"))

    status = await settled(fixture.host, job_id)
    assert (status.state, status.refusal) == ("failed", JobRefusal(kind="provider_refused", status=502))
    stored, _ = off_loop(
        fixture.host.eval_host.storage.load_job_record, EvalActionJob, job_id.removeprefix("action:"), TOYHOST_SCOPE
    )
    assert stored is not None and stored.refusal == JobRefusal(kind="provider_refused", status=502)

    evals, _admin = eval_catalogue().mount_all(standard_tools())
    polled = await evals.call({"action": "job_poll", "job_id": job_id}, host=fixture.host, caller=CALLER)
    assert polled.structured is not None and polled.structured["refusal"] == {"kind": "provider_refused", "status": 502}
    assert "refusal: provider_refused (502)" in polled.text


async def test_each_engine_refusal_keeps_its_own_kind() -> None:
    fixture, job_id = await _failing_job(StorageError("the store is down"))
    status = await settled(fixture.host, job_id)
    assert status.refusal == JobRefusal(kind="storage_failed", status=503)


async def test_a_job_broken_by_a_defect_carries_no_refusal() -> None:
    """A defect is no refusal of the job's work: its words are the detail, and it has no kind to branch on."""
    fixture, job_id = await _failing_job(RuntimeError("a bug in the draft"))
    status = await settled(fixture.host, job_id)
    assert status.state == "failed" and status.refusal is None and "RuntimeError" in (status.detail or "")


async def test_a_generation_its_provider_refused_keeps_the_refusal_on_its_attempt() -> None:
    gate = asyncio.Event()
    fixture = off_loop(ops_fixture, gate=gate)
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    (writer,) = fixture.writers
    writer.raises = ConnectionResetError("status 529: overloaded, account acct-123")
    gate.set()

    status = await settled(fixture.host, job.job_id)
    assert (status.state, status.refusal) == ("failed", JobRefusal(kind="provider_refused", status=502))
    attempt_id = job.job_id.rpartition(":")[2]
    stored, _ = off_loop(fixture.host.eval_host.storage.load_job_record, EvalAnalysisAttempt, attempt_id, TOYHOST_SCOPE)
    assert stored is not None and stored.refusal == status.refusal
    assert "acct-123" not in (stored.error or ""), "the provider's payload stays withheld from the record"
