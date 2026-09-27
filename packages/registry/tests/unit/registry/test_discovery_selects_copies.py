"""unit -- discovery shows each caller the copy it would be routed to, and a pod its own copy's state.

Discovery used to read the one entry-level definition the last registration wrote. It now asks
:meth:`CatalogEntry.select_copies`, the same function the proxy routes by, so an agent is shown
the definition of the copy its call would land on, with the confirmation gate OR'd across every
copy it can see, and the input-schema digest it hands back to be routed only to copies still
serving that schema. A polling pod names itself (``pod_id``) and is told the state of its OWN
copy, which is what its readiness waits on.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from threetears.nats import IncomingMessage, Subjects, set_default_namespace
from threetears.registry.catalog import ToolCatalog
from threetears.registry.discovery import DiscoverRequest, DiscoverResponse, DiscoverToolEntry, DiscoveryHandler

from ._copies import definition, endpoint, entry

__all__: list[str] = []

_AGENT_A = UUID("01948a00-aaaa-7000-8000-00000000000a")
_AGENT_B = UUID("01948a00-aaaa-7000-8000-00000000000b")
_A_POD = Subjects.agent_inprocess_pod_id(_AGENT_A, "inst-1")


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """bind a deterministic subject namespace."""
    set_default_namespace("test")


async def _discover(catalog: ToolCatalog, request: DiscoverRequest) -> DiscoverResponse:
    """drive one discovery and return the reply.

    :param catalog: the catalog to answer from
    :ptype catalog: ToolCatalog
    :param request: the request
    :ptype request: DiscoverRequest
    :return: the reply
    :rtype: DiscoverResponse
    """
    handler = DiscoveryHandler(catalog, namespace="test")
    nc = AsyncMock()
    await handler.start(nc)
    await handler.handle_discover(
        IncomingMessage(
            data=request.model_dump_json().encode("utf-8"), reply_subject="r", subject="test.tools.discover"
        )
    )
    reply = nc.publish_reply.await_args.kwargs["message"]
    assert isinstance(reply, DiscoverResponse)
    return reply


def _calc() -> list[DiscoverToolEntry]:
    """the one pinned tool these tests ask about.

    :return: the manifest
    :rtype: list[DiscoverToolEntry]
    """
    return [DiscoverToolEntry(name="threetears.calculator", version="1.0.0")]


class TestDiscoveryShowsTheSelectedCopy:
    """what a caller reads is what select_copies chose for it."""

    @pytest.mark.asyncio
    async def test_two_differing_copies_show_one_stable_definition_with_its_digest(self) -> None:
        """the newer announcement is shown, the same way on every ask, with its schema digest."""
        now = datetime.now(UTC)
        newer = definition("newer", input_schema={"type": "object", "properties": {"y": {}}})
        catalog = ToolCatalog()
        await catalog.register(
            entry(
                "threetears.calculator",
                "1.0.0",
                endpoint("pod-old", tool_definition=definition("older"), first=now - timedelta(seconds=30)),
                endpoint("pod-new", tool_definition=newer, first=now - timedelta(seconds=10)),
            )
        )
        answers = [
            (await _discover(catalog, DiscoverRequest(agent_id="unknown", tool_manifest=_calc()))).tools[0]
            for _ in range(3)
        ]
        assert {a.description for a in answers} == {"newer"}
        assert {a.input_schema_digest for a in answers} == {newer.schema_digest}
        assert answers[0].input_schema == newer.input_schema
        assert answers[0].endpoint_count == 1

    @pytest.mark.asyncio
    async def test_the_gate_is_ored_across_every_visible_copy(self) -> None:
        """one gated copy gates the tool for every caller who can see it -- even when not shown."""
        now = datetime.now(UTC)
        catalog = ToolCatalog()
        await catalog.register(
            entry(
                "threetears.calculator",
                "1.0.0",
                endpoint(
                    "pod-gated",
                    tool_definition=definition("gated", requires_confirmation=True),
                    first=now - timedelta(seconds=20),
                ),
                endpoint("pod-open", tool_definition=definition("open"), first=now - timedelta(seconds=5)),
            )
        )
        (result,) = (await _discover(catalog, DiscoverRequest(agent_id="unknown", tool_manifest=_calc()))).tools
        assert result.description == "open"
        assert result.requires_confirmation is True

    async def test_an_agent_is_shown_its_own_copy_and_others_are_not(self) -> None:
        """agent A sees its in-process definition; agent B and an unnamed caller see the shared one.

        A's copy is the older announcement, so it is the tier that shows it, not recency.
        """
        now = datetime.now(UTC)
        catalog = ToolCatalog()
        await catalog.register(
            entry(
                "threetears.calculator",
                "1.0.0",
                endpoint(_A_POD, tool_definition=definition("AGENT-A ONLY"), first=now - timedelta(seconds=20)),
                endpoint("pod-shared", tool_definition=definition("shared"), first=now - timedelta(seconds=5)),
            )
        )
        seen = {}
        for caller in (str(_AGENT_A), str(_AGENT_B), "unknown"):
            (result,) = (await _discover(catalog, DiscoverRequest(agent_id=caller, tool_manifest=_calc()))).tools
            seen[caller] = result.description
        assert seen == {str(_AGENT_A): "AGENT-A ONLY", str(_AGENT_B): "shared", "unknown": "shared"}

    async def test_the_list_all_path_asks_the_same_question(self) -> None:
        """an empty manifest lists what select_copies leaves each caller."""
        catalog = ToolCatalog()
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint(_A_POD, tool_definition=definition("A")))
        )
        mine = await _discover(catalog, DiscoverRequest(agent_id=str(_AGENT_A), tool_manifest=[]))
        theirs = await _discover(catalog, DiscoverRequest(agent_id=str(_AGENT_B), tool_manifest=[]))
        assert [(t.name, t.description) for t in mine.tools] == [("threetears.calculator", "A")]
        assert theirs.tools == []


class TestAPodIsToldItsOwnCopysState:
    """``pod_id`` on the request, ``requester_copy_status`` on each result."""

    @pytest.mark.asyncio
    async def test_a_pod_whose_copy_is_absent_is_told_so_even_when_another_copy_serves(self) -> None:
        """another pod's copy being available says nothing about this pod's."""
        catalog = ToolCatalog()
        await catalog.register(entry("threetears.calculator", "1.0.0", endpoint("builtin-tool-server")))
        (result,) = (
            await _discover(catalog, DiscoverRequest(agent_id="stray", pod_id="stray", tool_manifest=_calc()))
        ).tools
        assert result.status == "available"
        assert result.requester_copy_status == "absent"

    @pytest.mark.asyncio
    async def test_a_pod_is_told_its_copy_is_available_or_pending(self) -> None:
        """the two states readiness distinguishes."""
        catalog = ToolCatalog()
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint("pod-ready"), endpoint("pod-wait", "pending"))
        )
        (ready,) = (
            await _discover(catalog, DiscoverRequest(agent_id="pod-ready", pod_id="pod-ready", tool_manifest=_calc()))
        ).tools
        (waiting,) = (
            await _discover(catalog, DiscoverRequest(agent_id="pod-wait", pod_id="pod-wait", tool_manifest=_calc()))
        ).tools
        assert ready.requester_copy_status == "available"
        assert waiting.requester_copy_status == "pending"

    @pytest.mark.asyncio
    async def test_a_request_naming_no_pod_gets_no_copy_status(self) -> None:
        """an agent's discovery is not a readiness poll."""
        catalog = ToolCatalog()
        await catalog.register(entry("threetears.calculator", "1.0.0", endpoint("pod-ready")))
        (result,) = (await _discover(catalog, DiscoverRequest(agent_id="unknown", tool_manifest=_calc()))).tools
        assert result.requester_copy_status is None

    @pytest.mark.asyncio
    async def test_an_unknown_tool_reports_the_pods_copy_absent(self) -> None:
        """a tool nobody registered: unavailable, and this pod holds no copy of it."""
        (result,) = (
            await _discover(ToolCatalog(), DiscoverRequest(agent_id="p", pod_id="p", tool_manifest=_calc()))
        ).tools
        assert result.status == "unavailable"
        assert result.requester_copy_status == "absent"


class TestTheNewRequestFieldNeverCrossesAsNull:
    """``DiscoverRequest.pod_id`` is omitted when unset, so it is invisible to an older registry."""

    def test_an_unset_pod_id_is_not_serialized(self) -> None:
        """no key at all, not ``"pod_id": null``."""
        data = json.loads(DiscoverRequest(agent_id="a", tool_manifest=[]).model_dump_json())
        assert "pod_id" not in data

    def test_a_set_pod_id_is_serialized(self) -> None:
        """the twin: a real value crosses."""
        data = json.loads(DiscoverRequest(agent_id="a", tool_manifest=[], pod_id="p").model_dump_json())
        assert data["pod_id"] == "p"
