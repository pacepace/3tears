"""The FastMCP transport, driven by an in-process MCP client over the toy host.

The chunk's Done-when, end to end through a real client and server: an agent **lists** the tools and
what the scope holds, **launches** a run on the toy host, **polls** its job to completion, starts an
analysis generation and polls that, and **reads the report** — and an **undeclared parameter is refused
with the valid set**. Then what the transport adds over the catalogue: each tool's hints, schema and
description as a client sees them, a refusal as an MCP error result, a host's own tool cut (read-only,
own prefix), and the caller resolved per call — sync or async.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import Client, FastMCP

from threetears.evals.actions import Caller, eval_catalogue, read_only_tools
from threetears.evals.transports.fastmcp import mount_fastmcp
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import CALLER, RUN_MODELS, TOYHOST_SUBJECT, OpsFixture, ops_fixture


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
    fixture = ops_fixture()
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
        run_status = await _done(client, jobs["jobs"][0]["job_id"])
        assert run_status["state"] == "completed"
        summary, _, _ = await _text(client, "evals", {"action": "run_get", "run_id": run_status["run_id"]})
        assert f"run {run_status['run_id']} completed: {RUN_MODELS[0]}" in summary

        _, generation, _ = await _text(
            client, "evals", {"action": "analysis_generate", "campaign_id": fixture.campaign.id}
        )
        assert generation is not None
        analysis_status = await _done(client, generation["jobs"][0]["job_id"])
        assert analysis_status["state"] == "completed"

        report, document, is_error = await _text(
            client, "evals", {"action": "report_read", "analysis_id": analysis_status["analysis_id"]}
        )
        assert not is_error and document is not None and document["format"] == "markdown"
        assert report.startswith("# the wider chunk costs half again as much wall-clock per document.")


async def test_an_undeclared_parameter_is_refused_with_the_valid_set() -> None:
    async with Client(_served(ops_fixture())) as client:
        text, structured, is_error = await _text(
            client, "evals", {"action": "job_poll", "job_id": "run:r", "analysis_id": "a"}
        )
    assert is_error and structured is None
    assert text.startswith(
        "refused: job_poll does not take analysis_id.\njob_poll accepts:\n- job_id (string, required)"
    )
    assert 'Example: {"action": "job_poll", "job_id": "run:' in text


async def test_each_tool_shows_a_client_its_hints_schema_and_help() -> None:
    async with Client(_served(ops_fixture())) as client:
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
    assert set(admin.inputSchema["properties"]) == {"action", "topic", "run_id", "analysis_id", "confirm"}
    assert evals.description is not None and "action='help'" in evals.description
    assert "## Run and watch" in help_text


async def test_a_host_mounts_a_read_only_tool_under_its_own_prefix() -> None:
    async with Client(_served(ops_fixture(), tools=read_only_tools("lab"))) as client:
        (tool,) = await client.list_tools()
        text, _, is_error = await _text(client, "lab", {"action": "run_launch", "template_id": "t", "subject_id": "s"})
    assert tool.name == "lab" and tool.annotations is not None and tool.annotations.readOnlyHint is True
    assert is_error and "run_launch is a spend action, and tool lab carries read actions only" in text


async def test_the_caller_is_resolved_for_every_call_and_may_be_async() -> None:
    fixture = ops_fixture()
    asked: list[str] = []

    async def caller() -> Caller:
        asked.append("asked")
        return Caller(scope_id="another-scope", identity="agent:other")

    async with Client(_served(fixture, caller=caller)) as client:
        first, _, _ = await _text(client, "evals", {"action": "campaigns_list"})
        second, _, _ = await _text(client, "evals", {"action": "templates_list"})
    assert asked == ["asked", "asked"]
    assert first == "campaigns (0)" and second == "templates (0)", "the resolved scope, not the toy host's, was read"
