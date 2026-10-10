"""The operations and their one job contract, driven over the toy host.

A launch and an analysis generation both start long work and return jobs; :func:`job_poll` reads each
job from the record its work writes, and :func:`job_cancel` asks it to stop. Pinned here:

- **A launch is one job per arm**, polled to ``completed`` with the run's progress, then summarised.
- **A generation is one job**, polled to ``completed`` naming the analysis it stored, whose report reads
  three ways.
- **A job outlives the process that ran it as an answer**: a run that reads ``running`` with no live job,
  and a generation that recorded no attempt, both read ``lost``.
- **Cancel converges**: a held generation cancelled lands ``cancelled``; a held live run cancelled lands
  ``cancelled`` through its own boundary; a run with no live job is repaired to ``cancelled``.
- **A generation is answered only in its scope and campaign**: polled from another scope, or under another
  campaign, it reads ``lost``; cancelled from there, it is refused and keeps running.
- **Every refusal fires** — a job id of neither shape, cancelling a job that has ended, a generation on
  a host without generation settings, a second generation of a campaign while one runs (releasing the
  client it built when it loses the race), and generation settings that cannot bound a generation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.contracts import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.ops import (
    AnalysisGeneration,
    CampaignDefinition,
    LaunchArguments,
    analyses_list,
    analysis_archive,
    analysis_generate,
    analysis_job_id,
    campaign_archive,
    campaign_create,
    campaigns_list,
    job_cancel,
    job_poll,
    parse_job_id,
    report_read,
    run_delete,
    run_get,
    run_job_id,
    run_launch,
)
from threetears.evals.analysis import judge_agreement
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION
from packages.evals.tests.fixtures.toyhost.kind import ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import (
    CALLER,
    RUN_MODELS,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ops_fixture,
    settled,
)


def _launch(*models: str) -> LaunchArguments:
    return LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=list(models)
    )


# =============================================================================
# Launch: one job per arm
# =============================================================================


async def test_a_launch_is_a_job_per_arm_polled_to_completed() -> None:
    fixture = ops_fixture()
    started = await run_launch(fixture.host, _launch(*RUN_MODELS), TOYHOST_SCOPE)

    assert [job.label for job in started.jobs] == list(RUN_MODELS)
    for job in started.jobs:
        assert job.kind == "run" and job.job_id == run_job_id(job.target_id)
        status = await settled(fixture.host, job.job_id)
        assert (status.state, status.status, status.done) == ("completed", "completed", True)
        assert status.progress["completed"] == status.progress["total"] > 0
        assert status.detail is None
        summary = run_get(fixture.host.eval_host, job.target_id, TOYHOST_SCOPE)
        assert summary.status == "completed" and summary.n_results > 0


async def test_a_run_reading_running_with_no_live_job_is_lost_and_cancel_repairs_it() -> None:
    fixture = ops_fixture()
    storage = fixture.host.eval_host.storage
    run = storage.load_eval_run(fixture.campaign.run_ids[0], TOYHOST_SCOPE)
    assert run is not None
    storage.save_eval_run(run.model_copy(update={"status": "running", "completed_at": None}))

    lost = await job_poll(fixture.host, run_job_id(run.id), TOYHOST_SCOPE)
    assert (lost.state, lost.status, lost.done) == ("lost", "running", True)
    assert lost.detail is not None and "no job in this process is running it" in lost.detail

    cancelled = await job_cancel(fixture.host, run_job_id(run.id), TOYHOST_SCOPE, reason="stale")
    assert (cancelled.state, cancelled.status) == ("cancelled", "cancelled")


async def test_a_run_its_wall_clock_budget_stopped_reads_stopped_and_says_why() -> None:
    """Every reader of a run's status names the clock stop as a stop, never as a failure."""
    fixture = ops_fixture()
    storage = fixture.host.eval_host.storage
    run = storage.load_eval_run(fixture.campaign.run_ids[0], TOYHOST_SCOPE)
    assert run is not None
    reason = "wall-clock budget reached — the run's 660s time budget ran out after 660s"
    storage.save_eval_run(run.model_copy(update={"status": "budget_stopped", "budget_stop_reason": reason}))

    status = await job_poll(fixture.host, run_job_id(run.id), TOYHOST_SCOPE)
    assert (status.state, status.status, status.done) == ("stopped", "budget_stopped", True)
    assert status.detail == reason

    summary = run_get(fixture.host.eval_host, run.id, TOYHOST_SCOPE)
    assert summary.stopped_because == reason
    assert f"  stopped: {reason}" in summary.render().splitlines()
    assert not [error for error in summary.errors if "wall-clock" in error], "a stop is not among the faults"


async def test_cancelling_a_run_that_has_ended_is_refused() -> None:
    fixture = ops_fixture()
    with pytest.raises(ValidationFailedError, match="is completed — only pending/running runs can be cancelled"):
        await job_cancel(fixture.host, run_job_id(fixture.campaign.run_ids[0]), TOYHOST_SCOPE)


# =============================================================================
# Analysis: one job, its record the generation attempt
# =============================================================================


async def test_a_generation_is_a_job_polled_to_the_analysis_it_stored() -> None:
    fixture = ops_fixture()
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert job.kind == "analysis" and parse_job_id(job.job_id)[:2] == ("analysis", fixture.campaign.id)

    status = await settled(fixture.host, job.job_id)
    assert (status.state, status.status) == ("completed", "stored") and status.analysis_id is not None

    (line,) = analyses_list(fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
    assert line.id == status.analysis_id
    markdown = report_read(fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="markdown")
    assert (markdown.basis, markdown.analysis_id) == ("analysis", line.id)
    assert markdown.body.startswith(f"# {line.headline}")
    html = report_read(fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="html")
    assert "<script" not in html.body.lower() and line.headline in html.body
    canonical = report_read(fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="json")
    assert f'"analysis_id":"{line.id}"' in canonical.body.replace(" ", "")


async def test_a_held_generation_is_running_refuses_a_second_and_cancels_to_cancelled() -> None:
    gate = asyncio.Event()
    fixture = ops_fixture(gate=gate)
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    await asyncio.sleep(0)

    running = await job_poll(fixture.host, job.job_id, TOYHOST_SCOPE)
    assert (running.state, running.done) == ("running", False)
    with pytest.raises(ConflictError, match="is already running"):
        await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)
    assert len(fixture.writers) == 1, "the refused second generation built no client"

    await job_cancel(fixture.host, job.job_id, TOYHOST_SCOPE)
    cancelled = await settled(fixture.host, job.job_id)
    assert (cancelled.state, cancelled.status) == ("cancelled", "cancelled")

    with pytest.raises(ValidationFailedError, match="is not running here \\(it reads cancelled\\)"):
        await job_cancel(fixture.host, job.job_id, TOYHOST_SCOPE)


@pytest.mark.parametrize(
    ("scope_id", "forged_campaign"),
    [("some-other-scope", "not-my-campaign"), ("some-other-scope", None), (TOYHOST_SCOPE, "not-my-campaign")],
    ids=["other-scope-forged-campaign", "other-scope-real-campaign", "same-scope-forged-campaign"],
)
async def test_a_generation_is_neither_polled_nor_cancelled_from_outside_its_campaign_and_scope(
    scope_id: str, forged_campaign: str | None
) -> None:
    """The job manager is process-wide; a generation is live only to the scope and campaign that started it."""
    gate = asyncio.Event()
    fixture = ops_fixture(gate=gate)
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    await asyncio.sleep(0)
    _, _, attempt = parse_job_id(job.job_id)
    assert attempt is not None
    probe = analysis_job_id(forged_campaign or fixture.campaign.id, attempt)

    polled = await job_poll(fixture.host, probe, scope_id)
    assert (polled.state, polled.status) == ("lost", "unrecorded"), "another scope's live generation reads as none"
    with pytest.raises(ValidationFailedError, match="is not running here"):
        await job_cancel(fixture.host, probe, scope_id)

    owner = await job_poll(fixture.host, job.job_id, TOYHOST_SCOPE)
    assert owner.state == "running", "the refused cancel left the owner's generation running"
    gate.set()
    finished = await settled(fixture.host, job.job_id)
    assert (finished.state, finished.status) == ("completed", "stored")


async def test_a_live_run_is_cancelled_through_the_job_contract_and_converges(monkeypatch: pytest.MonkeyPatch) -> None:
    """``job_cancel`` asks the live job to stop; its own boundary writes ``cancelled``, which ``job_poll`` then reads."""
    gate = asyncio.Event()
    extract = ScriptedExtractionClient.extract

    async def held(self: ScriptedExtractionClient, request: Any) -> Any:
        await gate.wait()
        return await extract(self, request)

    monkeypatch.setattr(ScriptedExtractionClient, "extract", held)
    fixture = ops_fixture()
    (job,) = (await run_launch(fixture.host, _launch(RUN_MODELS[0]), TOYHOST_SCOPE)).jobs
    async with asyncio.timeout(10):
        while (await job_poll(fixture.host, job.job_id, TOYHOST_SCOPE)).status != "running":
            await asyncio.sleep(0.01)

    await job_cancel(fixture.host, job.job_id, TOYHOST_SCOPE, reason="operator stop")
    cancelled = await settled(fixture.host, job.job_id)

    assert (cancelled.state, cancelled.status) == ("cancelled", "cancelled")
    assert cancelled.detail is not None and "operator stop" in cancelled.detail
    assert not gate.is_set(), "the run was stopped while held, not after it finished"


async def test_a_generation_that_loses_the_race_releases_its_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Between the check and the start another generation took the key: the client this one built is released."""
    fixture = ops_fixture()

    def taken(*_: Any, **__: Any) -> str:
        raise ConflictError("task 'other' is already running")

    monkeypatch.setattr(fixture.host.launch.job_manager, "start_task", taken)
    with pytest.raises(ConflictError):
        await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)
    (writer,) = fixture.writers
    assert writer.closed == 1 and writer.calls == []


async def test_a_generation_that_recorded_no_attempt_is_lost() -> None:
    fixture = ops_fixture()
    status = await job_poll(fixture.host, analysis_job_id(fixture.campaign.id, "0193-never"), TOYHOST_SCOPE)
    assert (status.state, status.status, status.done) == ("lost", "unrecorded", True)


async def test_a_host_without_generation_settings_refuses_a_generation() -> None:
    fixture = ops_fixture(generation=False)
    with pytest.raises(ValidationFailedError, match="does not generate analyses here"):
        await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)
    assert fixture.writers == []


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"prompt_id": " "}, "prompt_id is blank"),
        ({"max_output_tokens": 0}, "max_output_tokens must be positive"),
        ({"budget_s": 0.0}, "budget_s must be positive"),
    ],
)
def test_generation_settings_that_cannot_bound_a_generation_are_refused(fields: dict[str, Any], message: str) -> None:
    async def prompt() -> str:
        return "p"

    valid: dict[str, Any] = {"prompt_id": "p", "resolve_prompt": prompt, "max_output_tokens": 10, "budget_s": 1.0}
    AnalysisGeneration(**valid)
    with pytest.raises(ValueError, match=message):
        AnalysisGeneration(**(valid | fields))


# =============================================================================
# Job ids
# =============================================================================


@pytest.mark.parametrize(
    ("job_id", "parsed"),
    [
        ("run:r-1", ("run", "r-1", None)),
        ("analysis:c-1:a-1", ("analysis", "c-1", "a-1")),
        ("analysis:c:with:colons:a-1", ("analysis", "c:with:colons", "a-1")),
    ],
)
def test_a_job_id_reads_back_into_what_it_names(job_id: str, parsed: tuple[str, str, str | None]) -> None:
    assert parse_job_id(job_id) == parsed


@pytest.mark.parametrize("job_id", ["r-1", "run:", "analysis:", "analysis:c-1", "analysis::a-1", "analysis:c-1:"])
def test_a_job_id_of_neither_shape_is_refused(job_id: str) -> None:
    with pytest.raises(ValidationFailedError, match="names no job"):
        parse_job_id(job_id)


# =============================================================================
# Campaign curation
# =============================================================================


def test_a_campaign_is_created_archived_and_listed_as_typed_lines() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    created = campaign_create(
        host,
        CampaignDefinition(
            name="widths", subject_id=TOYHOST_SUBJECT.subject_id, behavior="speed", run_ids=fixture.campaign.run_ids
        ),
        TOYHOST_SCOPE,
        created_by=CALLER.identity,
    )
    assert created.run_count == len(fixture.campaign.run_ids)
    stored = host.storage.load_campaign(created.id, TOYHOST_SCOPE)
    assert stored is not None and stored.created_by == CALLER.identity

    assert campaign_archive(host, created.id, TOYHOST_SCOPE, archived=True).archived
    active = {line.id for line in campaigns_list(host, TOYHOST_SCOPE, archived=False).campaigns}
    assert created.id not in active and fixture.campaign.id in active


async def test_an_analysis_is_archived_and_restored_through_the_action_its_delete_points_to() -> None:
    """``analysis_delete`` says "archive instead"; ``analysis_archive`` is that action, and it is reversible."""
    fixture = ops_fixture()
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    analysis_id = (await settled(fixture.host, job.job_id)).analysis_id
    assert analysis_id is not None
    tools = {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}
    deleting = tools["evals_admin"].action("analysis_delete")
    assert deleting is not None and "archive instead" in deleting.summary
    assert tools["evals"].action("analysis_archive") is not None, "the action the delete's advice names exists"

    archived = await tools["evals"].call(
        {"action": "analysis_archive", "analysis_id": analysis_id, "archived": True, "archive_reason": "superseded"},
        host=fixture.host,
        caller=CALLER,
    )
    assert not archived.is_error and archived.structured is not None and archived.structured["archived"] is True
    stored = fixture.host.eval_host.storage.load_analysis(analysis_id, TOYHOST_SCOPE)
    assert stored is not None and stored.archived and stored.archived_reason == "superseded"

    restored = analysis_archive(fixture.host.eval_host, analysis_id, TOYHOST_SCOPE, archived=False)
    assert not restored.archived
    with pytest.raises(NotFoundError):
        analysis_archive(fixture.host.eval_host, "no-such-analysis", TOYHOST_SCOPE, archived=True)


async def test_an_agent_lists_reads_and_deletes_the_insights_an_analysis_minted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#651: holding only the standard tools, an agent reaches the insight ledger end to end."""
    import packages.evals.tests.ops_support as support

    memo = support.memo_payload

    def durable_memo(bundle: Any) -> dict[str, Any]:
        payload = memo(bundle)
        payload["findings"][0]["durable"] = "the wide chunk is slower per document than the narrow one"
        return payload

    monkeypatch.setattr(support, "memo_payload", durable_memo)
    fixture = ops_fixture()
    tools = {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}
    evals, admin = tools["evals"], tools["evals_admin"]
    started = await evals.call(
        {"action": "analysis_generate", "campaign_id": fixture.campaign.id}, host=fixture.host, caller=CALLER
    )
    assert not started.is_error and started.structured is not None
    (job,) = started.structured["jobs"]
    analysis_id = (await settled(fixture.host, job["job_id"])).analysis_id
    assert analysis_id is not None

    listed = await evals.call(
        {"action": "insights_list", "campaign_filter": fixture.campaign.id}, host=fixture.host, caller=CALLER
    )
    assert not listed.is_error and listed.structured is not None
    (line,) = listed.structured["insights"]
    assert line["source_analysis_id"] == analysis_id and line["standing"] == "live"
    assert line["statement"] in listed.text

    elsewhere = await evals.call(
        {"action": "insights_list", "campaign_filter": "no-such-campaign"}, host=fixture.host, caller=CALLER
    )
    assert not elsewhere.is_error and elsewhere.structured is not None and elsewhere.structured["insights"] == []
    assert "source_campaign_id='no-such-campaign'" in elsewhere.text, "an empty filtered read names what it searched"

    read = await evals.call({"action": "insight_get", "insight_id": line["id"]}, host=fixture.host, caller=CALLER)
    assert not read.is_error and read.structured is not None
    stored = fixture.host.eval_host.storage.load_insight(line["id"], TOYHOST_SCOPE)
    assert stored is not None and read.structured["insight"] == stored.model_dump(mode="json")
    assert f"analysis {analysis_id}" in read.text

    assert analysis_archive(fixture.host.eval_host, analysis_id, TOYHOST_SCOPE, archived=True).archived
    retracted = await evals.call({"action": "insight_get", "insight_id": line["id"]}, host=fixture.host, caller=CALLER)
    assert retracted.structured is not None and retracted.structured["standing"] == "retracted"

    assert evals.action("insight_delete") is None, "a destructive action is only on the admin tool"
    refused = await admin.call(
        {"action": "insight_delete", "insight_id": line["id"], "confirm": "wrong"}, host=fixture.host, caller=CALLER
    )
    assert refused.is_error and fixture.host.eval_host.storage.load_insight(line["id"], TOYHOST_SCOPE) is not None
    deleted = await admin.call(
        {"action": "insight_delete", "insight_id": line["id"], "confirm": line["id"]}, host=fixture.host, caller=CALLER
    )
    assert not deleted.is_error and deleted.structured == {
        "insight_id": line["id"],
        "source_campaign_id": fixture.campaign.id,
    }
    assert fixture.host.eval_host.storage.load_insight(line["id"], TOYHOST_SCOPE) is None
    gone = await evals.call({"action": "insight_get", "insight_id": line["id"]}, host=fixture.host, caller=CALLER)
    assert gone.is_error


async def test_an_agent_rates_through_the_action_and_its_rating_is_never_a_persons() -> None:
    """``result_rate`` fixes ``rater_kind`` to agent: the judge's agreement with people lists it, never pools it."""
    fixture = ops_fixture()
    storage = fixture.host.eval_host.storage
    (judged,) = [
        result
        for run_id in fixture.campaign.run_ids[:1]
        for result in storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE)[:1]
    ]
    assert judged.judge_score(TOYHOST_JUDGED_DIMENSION) is not None, "the toy result is judged on that dimension"
    tools = {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}

    outcome = await tools["evals"].call(
        {
            "action": "result_rate",
            "result_id": judged.id,
            "rubric_dim": TOYHOST_JUDGED_DIMENSION,
            "score": 4,
            "rating_reason": "faithful layout",
        },
        host=fixture.host,
        caller=CALLER,
    )

    assert not outcome.is_error, outcome.text
    (rating,) = storage.query_calibration_ratings(TOYHOST_SCOPE, result_id=judged.id)
    assert (rating.rater, rating.rater_kind, rating.score) == (CALLER.identity, "agent", 4)
    agreement = judge_agreement([rating], [judged])
    assert agreement.dimensions == [] and [u.reason for u in agreement.unpaired] == ["rated_by_an_agent"]
    action = tools["evals"].action("result_rate")
    assert action is not None and "rater_kind" not in action.params.model_fields, "the caller cannot claim a person"


def test_the_launch_action_offers_exactly_the_launchs_own_arguments() -> None:
    """The action's parameters are derived from the operation's one declaration, so neither can gain a field alone."""
    actions = {action.name: action for action in eval_catalogue().actions}
    for name in ("run_launch", "launch_estimate"):
        params = actions[name].params
        assert issubclass(params, LaunchArguments)
        for field_name, declared in LaunchArguments.model_fields.items():
            assert params.model_fields[field_name].description == declared.description


def test_a_run_is_deleted_when_confirm_echoes_its_id() -> None:
    fixture = ops_fixture()
    run_id = fixture.campaign.run_ids[0]
    deleted = run_delete(fixture.host.eval_host, run_id, TOYHOST_SCOPE, confirm=run_id)
    assert deleted.run_id == run_id and deleted.results_deleted > 0
    assert fixture.campaign.id in deleted.campaigns_detached
    assert fixture.host.eval_host.storage.load_eval_run(run_id, TOYHOST_SCOPE) is None
