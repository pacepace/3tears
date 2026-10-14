"""The FastMCP transport, driven by an in-process MCP client over the toy host.

The chunk's Done-when, end to end through a real client and server: an agent **lists** the tools and
what the scope holds, **launches** a run on the toy host, **polls** its job to completion, starts an
analysis generation and polls that, and **reads the report** — and an **undeclared parameter is refused
with the valid set**. Then what the transport adds over the catalogue: each tool's hints, schema and
description as a client sees them, a refusal as an MCP error result carrying its kind and status, a host's own tool cut (read-only,
own prefix), and the caller resolved per call — sync or async.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP

from threetears.evals.actions import Caller, eval_catalogue, read_only_tools
from threetears.evals.transports.fastmcp import CallTimedOut, mount_fastmcp
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import CALLER, RUN_MODELS, TOYHOST_SUBJECT, OpsFixture, ops_fixture
from packages.evals.tests.off_loop import off_loop


def _served(fixture: OpsFixture, **mount: Any) -> FastMCP:
    server = FastMCP("toyhost")
    mount.setdefault("caller", lambda: CALLER)
    mount_fastmcp(server, eval_catalogue(), host=fixture.host, **mount)
    return server


async def _text(client: Client, tool: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any] | None, bool]:
    result = await client.call_tool(tool, arguments, raise_on_error=False)
    return result.content[0].text, result.structured_content, result.is_error


async def _done(client: Client, job_id: str) -> dict[str, Any]:
    async with asyncio.timeout(10):
        while True:
            _, status, is_error = await _text(client, "evals", {"action": "job_poll", "job_id": job_id})
            assert not is_error and status is not None
            if status["done"]:
                return status
            await asyncio.sleep(0.01)


async def test_an_agent_lists_launches_polls_and_reads_a_report() -> None:
    """The Done-when, through an in-process MCP client."""
    fixture = off_loop(ops_fixture)
    async with Client(_served(fixture)) as client:
        assert sorted(tool.name for tool in await client.list_tools()) == ["evals", "evals_admin"]

        listed, _, _ = await _text(client, "evals", {"action": "templates_list"})
        assert listed.startswith("templates (1)") and toyhost_template().id in listed

        launch = {
            "action": "run_launch",
            "template_id": toyhost_template().id,
            "subject_id": TOYHOST_SUBJECT.subject_id,
            "models": [RUN_MODELS[0]],
        }
        started, jobs, is_error = await _text(client, "evals", launch)
        assert not is_error and jobs is not None and started.startswith("started 1 job(s):")
        assert jobs["jobs"][0]["kind"] == "launch"
        launch_status = await _done(client, jobs["jobs"][0]["job_id"])
        assert launch_status["state"] == "completed"
        (run_id,) = launch_status["run_ids"]
        summary, _, _ = await _text(client, "evals", {"action": "run_get", "run_id": run_id})
        assert f"run {run_id} completed: {RUN_MODELS[0]}" in summary

        _, generation, _ = await _text(
            client, "evals", {"action": "analysis_generate", "campaign_id": fixture.campaign.id}
        )
        assert generation is not None
        analysis_status = await _done(client, generation["jobs"][0]["job_id"])
        assert analysis_status["state"] == "completed"

        report, document, is_error = await _text(
            client, "evals", {"action": "report_read", "campaign_id": fixture.campaign.id}
        )
        assert not is_error and document is not None and document["format"] == "markdown"
        assert (document["basis"], document["analysis_id"]) == ("analysis", analysis_status["analysis_id"])
        title = report.splitlines()[0]
        assert title.startswith("# chunk width sweep · ") and "analysis by" in title


async def test_an_undeclared_parameter_is_refused_with_the_valid_set() -> None:
    async with Client(_served(off_loop(ops_fixture))) as client:
        text, structured, is_error = await _text(
            client, "evals", {"action": "job_poll", "job_id": "run:r", "analysis_id": "a"}
        )
    assert is_error and structured == {"refusal": {"kind": "validation_failed", "status": 422}}
    assert text.startswith(
        "refused: job_poll does not take analysis_id.\njob_poll accepts:\n- job_id (string, required)"
    )
    assert 'Example: {"action": "job_poll", "job_id": "run:' in text


async def test_an_engine_refusal_reaches_the_client_with_its_kind_and_status() -> None:
    async with Client(_served(off_loop(ops_fixture))) as client:
        text, structured, is_error = await _text(client, "evals", {"action": "run_get", "run_id": "no-such-run"})
    assert is_error and structured == {"refusal": {"kind": "not_found", "status": 404}}
    assert text.startswith("refused: run_get:") and "no-such-run" in text, "the teaching text is unchanged"


async def test_each_tool_shows_a_client_its_hints_schema_and_help() -> None:
    async with Client(_served(off_loop(ops_fixture))) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        help_text, _, _ = await _text(client, "evals", {"action": "help"})
    evals, admin = tools["evals"], tools["evals_admin"]
    assert evals.annotations is not None and admin.annotations is not None
    assert (evals.annotations.readOnlyHint, evals.annotations.destructiveHint, evals.annotations.openWorldHint) == (
        False,
        False,
        True,
    )
    assert admin.annotations.destructiveHint is True
    assert evals.inputSchema["properties"]["action"]["enum"][:2] == ["help", "templates_list"]
    assert set(admin.inputSchema["properties"]) == {"action", "topic", "run_id", "analysis_id", "insight_id", "confirm"}
    assert evals.description is not None and "action='help'" in evals.description
    assert "## Run and watch" in help_text


async def test_a_host_mounts_a_read_only_tool_under_its_own_prefix() -> None:
    async with Client(_served(off_loop(ops_fixture), tools=read_only_tools("lab"))) as client:
        (tool,) = await client.list_tools()
        text, _, is_error = await _text(client, "lab", {"action": "run_launch", "template_id": "t", "subject_id": "s"})
    assert tool.name == "lab" and tool.annotations is not None and tool.annotations.readOnlyHint is True
    assert is_error and "run_launch is a spend action, and tool lab carries read actions only" in text


async def test_the_caller_is_resolved_for_every_call_and_may_be_async() -> None:
    fixture = off_loop(ops_fixture)
    asked: list[str] = []

    async def caller() -> Caller:
        asked.append("asked")
        return Caller(scope_id="another-scope", identity="agent:other")

    async with Client(_served(fixture, caller=caller)) as client:
        first, _, _ = await _text(client, "evals", {"action": "campaigns_list"})
        second, _, _ = await _text(client, "evals", {"action": "templates_list"})
    assert asked == ["asked", "asked"]
    assert first == "campaigns (0)" and second == "templates (0)", "the resolved scope, not the toy host's, was read"


async def test_a_cap_above_the_hosts_is_refused_through_the_transport() -> None:
    """An agent over MCP cannot raise the host's per-run ceiling by naming a larger ``max_cost_usd``."""
    fixture = off_loop(ops_fixture)
    above = fixture.host.launch.settings().max_cost_usd * 10
    async with Client(_served(fixture)) as client:
        text, _, is_error = await _text(
            client,
            "evals",
            {
                "action": "run_launch",
                "template_id": toyhost_template().id,
                "subject_id": TOYHOST_SUBJECT.subject_id,
                "models": [RUN_MODELS[0]],
                "max_cost_usd": above,
            },
        )
    assert is_error and "a launch may only lower the host's ceiling" in text
    assert fixture.host.launch.job_manager.admitted_count == 0


async def test_every_call_is_held_to_the_one_fixed_call_timeout() -> None:
    """A call outliving the server's one timeout is answered as an error saying so, never left holding the client."""
    fixture = off_loop(ops_fixture)

    async def slow_caller() -> Caller:
        await asyncio.sleep(1)
        return CALLER

    async with Client(_served(fixture, caller=slow_caller, call_timeout_s=0.05)) as client:
        text, structured, is_error = await _text(client, "evals", {"action": "templates_list"})
    assert is_error and structured is None
    assert "outlived this server's 0.05s call timeout" in text and "No action's call waits on a model" in text
    # A run_launch stopped after its job record was written, retried blind, is a second paid launch.
    assert "may have completed, or begun a write, before it was stopped, so read before calling it again" in text
    assert "job_poll for a job it may have started" in text and "runs_list" in text


async def test_a_call_timed_out_says_to_read_before_calling_again() -> None:
    stopped = CallTimedOut(0.5)
    assert str(stopped).startswith("the call outlived its 0.5s call timeout and was stopped; the action may have")
    assert "read before calling it again: job_poll" in str(stopped)


async def test_a_timeout_the_call_raises_of_its_own_is_not_reported_as_the_call_timeout() -> None:
    """``except TimeoutError`` caught the call's own timeout too, and answered it as the server's."""
    fixture = off_loop(ops_fixture)

    async def timing_out_caller() -> Caller:
        raise TimeoutError("the identity service did not answer")

    async with Client(_served(fixture, caller=timing_out_caller, call_timeout_s=30.0)) as client:
        text, _, is_error = await _text(client, "evals", {"action": "templates_list"})
    assert is_error and "the identity service did not answer" in text
    assert "outlived this server's" not in text


def test_a_call_timeout_that_is_not_positive_is_refused_at_mount() -> None:
    with pytest.raises(ValueError, match="call_timeout_s must be positive"):
        _served(ops_fixture(), call_timeout_s=0)


async def test_a_model_calling_actions_refusal_in_the_call_reaches_the_client_as_data() -> None:
    """A spend action that runs as a job is still admitted in the call; refused there, its kind and status are data."""
    async with Client(_served(off_loop(ops_fixture))) as client:
        text, structured, is_error = await _text(
            client, "evals", {"action": "judge_second", "run_id": "no-such-run", "second_judge_model": "judge/other"}
        )
    assert is_error and structured == {"refusal": {"kind": "not_found", "status": 404}}
    assert "no-such-run" in text


def test_importing_the_transport_leaves_fastmcps_server_stack_unloaded() -> None:
    """``FastMCP`` is an annotation here, so importing the transport does not load the server stack it lives in.

    That stack (its context, sampling, HTTP app and event store) cost about as long again as the tool classes this
    module subclasses, and an annotation-only import of it ran at every import of the transport. A fresh interpreter,
    since this test process already loaded it.
    """
    probe = (
        "import sys\n"
        "import threetears.evals.transports.fastmcp\n"
        "assert 'fastmcp.tools' in sys.modules, 'non-vacuity: the transport loads the tool classes it subclasses'\n"
        "print('fastmcp.server.server' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        # A guard against a hung interpreter, not a speed bound: the transport imports the whole engine.
        timeout=300,
        cwd=Path(__file__).resolve().parents[3],
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", "importing the transport loaded fastmcp.server.server"
