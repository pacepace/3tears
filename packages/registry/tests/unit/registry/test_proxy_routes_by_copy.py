"""unit -- the proxy routes by the same copy selection discovery shows, and keeps old peers working.

A caller is routed only to copies serving the input schema it was shown: by default the shown
definition's, or exactly the digest it names on ``ProxyCallRequest.input_schema_digest``. When
copies are visible but none serves that schema any more, the call is refused
``TOOL_DEFINITION_CHANGED`` so the caller re-discovers instead of sending arguments shaped for a
schema nobody serves. The timeout a call runs under is the ROUTED copy's.

Two wire properties are pinned beside that, because each would break live traffic at deploy:
the new request field never crosses as an explicit null (an older registry forbids unknown keys,
null included), and the call the proxy forwards to a pod gains no key at all.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid7

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from threetears.agent.tools.context_envelope import CallContext
from threetears.nats import IncomingMessage, Subjects, set_default_namespace
from threetears.registry.catalog import ToolCatalog
from threetears.registry.proxy import CallProxy, ProxyCallRequest, ProxyCallResponse

from .copy_entries import definition, endpoint, entry
from .dispatch_auth import make_authed_request, make_proxy

__all__: list[str] = []

_NS = "test"
_AGENT_A = UUID("01948a00-aaaa-7000-8000-00000000000a")
_A_POD = Subjects.agent_inprocess_pod_id(_AGENT_A, "inst-1")
_NARROW = definition("narrow", timeout_seconds=7.0)
_WIDE = definition("wide", input_schema={"type": "object", "properties": {"extra": {}}}, timeout_seconds=11.0)


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """bind a deterministic subject namespace."""
    set_default_namespace(_NS)


async def _two_schemas() -> ToolCatalog:
    """a tool served under two input schemas; the wide one was announced more recently.

    :return: the catalog
    :rtype: ToolCatalog
    """
    now = datetime.now(UTC)
    catalog = ToolCatalog()
    await catalog.register(
        entry(
            "threetears.calculator",
            "1.0.0",
            endpoint("pod-narrow", tool_definition=_NARROW, first=now - timedelta(seconds=20)),
            endpoint("pod-wide", tool_definition=_WIDE, first=now - timedelta(seconds=5)),
        )
    )
    return catalog


async def _proxy(catalog: ToolCatalog) -> tuple[CallProxy, AsyncMock]:
    """a started proxy whose forwards are recorded and always succeed.

    :param catalog: the catalog to route against
    :ptype catalog: ToolCatalog
    :return: the proxy and its transport
    :rtype: tuple[CallProxy, AsyncMock]
    """
    proxy = make_proxy(catalog, namespace=_NS, timeout=5.0)
    nc = AsyncMock()
    ok = ProxyCallResponse(success=True, content="ok", context=CallContext()).model_dump_json().encode("utf-8")
    nc.request_raw = AsyncMock(return_value=ok)
    await proxy.start(nc)
    return proxy, nc


async def _call(proxy: CallProxy, nc: AsyncMock, request: ProxyCallRequest) -> ProxyCallResponse:
    """drive one call and return the answer.

    :param proxy: the proxy
    :ptype proxy: CallProxy
    :param nc: its transport
    :ptype nc: AsyncMock
    :param request: the call
    :ptype request: ProxyCallRequest
    :return: the answer
    :rtype: ProxyCallResponse
    """
    before = nc.publish_reply.await_count
    await proxy.handle_call(
        IncomingMessage(data=request.model_dump_json().encode("utf-8"), reply_subject="_INBOX.c", subject="t")
    )
    for _ in range(100):
        if nc.publish_reply.await_count > before:
            break
        await asyncio.sleep(0)
    answer: ProxyCallResponse = nc.publish_reply.await_args.kwargs["message"]
    return answer


def _targets(nc: AsyncMock) -> list[str]:
    """the pod id of every forward, in order.

    :param nc: the proxy's transport
    :ptype nc: AsyncMock
    :return: pod ids
    :rtype: list[str]
    """
    prefix = f"{_NS}.tools.internal."
    return [call.kwargs["subject"].path.removeprefix(prefix) for call in nc.request_raw.await_args_list]


class TestRoutingFollowsTheSelectedSchema:
    """no digest routes by the shown definition; a digest routes by itself."""

    @pytest.mark.asyncio
    async def test_no_digest_routes_to_the_shown_copy_only(self) -> None:
        """the wide copy is shown, so only the wide copy is called."""
        proxy, nc = await _proxy(await _two_schemas())
        for _ in range(20):
            answer = await _call(proxy, nc, make_authed_request())
            assert answer.success is True, answer.error
        assert set(_targets(nc)) == {"pod-wide"}

    @pytest.mark.asyncio
    async def test_a_named_digest_routes_to_the_copy_serving_it(self) -> None:
        """a caller still holding the narrow schema is routed to the narrow copy."""
        proxy, nc = await _proxy(await _two_schemas())
        for _ in range(20):
            answer = await _call(proxy, nc, make_authed_request(input_schema_digest=_NARROW.schema_digest))
            assert answer.success is True, answer.error
        assert set(_targets(nc)) == {"pod-narrow"}

    @pytest.mark.asyncio
    async def test_a_digest_nobody_serves_is_a_changed_definition(self) -> None:
        """copies are visible, none serves the schema: refused before any forward."""
        proxy, nc = await _proxy(await _two_schemas())
        answer = await _call(proxy, nc, make_authed_request(input_schema_digest="0" * 64))
        assert answer.success is False
        assert answer.error_code == "TOOL_DEFINITION_CHANGED"
        assert _targets(nc) == []

    @pytest.mark.asyncio
    async def test_the_routed_copys_timeout_is_the_one_applied(self) -> None:
        """narrow declares 7s, wide 11s; each call waits its own copy's declaration."""
        proxy, nc = await _proxy(await _two_schemas())
        await _call(proxy, nc, make_authed_request(input_schema_digest=_NARROW.schema_digest))
        await _call(proxy, nc, make_authed_request(input_schema_digest=_WIDE.schema_digest))
        waits = [call.kwargs["timeout"].total_seconds() for call in nc.request_raw.await_args_list]
        assert waits == [7.0, 11.0]

    @pytest.mark.asyncio
    async def test_an_agent_is_routed_to_its_own_copy_over_the_shared_one(self) -> None:
        """the caller's own in-process copy is its tier -- older, same schema, and still chosen."""
        now = datetime.now(UTC)
        catalog = ToolCatalog()
        await catalog.register(
            entry(
                "threetears.calculator",
                "1.0.0",
                endpoint(_A_POD, tool_definition=_NARROW, first=now - timedelta(seconds=20)),
                endpoint("pod-shared", tool_definition=definition("shared", timeout_seconds=7.0), first=now),
            )
        )
        proxy, nc = await _proxy(catalog)
        for _ in range(20):
            answer = await _call(proxy, nc, make_authed_request(agent_id=_AGENT_A))
            assert answer.success is True, answer.error
        assert set(_targets(nc)) == {_A_POD}

    @pytest.mark.asyncio
    async def test_replicas_under_one_pod_id_with_two_schemas_resolve_deterministically(self) -> None:
        """mid-rollout, one pod id announces two schemas: the newer is shown and routed, every time."""
        now = datetime.now(UTC)
        copy = endpoint("evd-tools", tool_definition=_NARROW, first=now - timedelta(seconds=30))
        copy.announce(_WIDE, now - timedelta(seconds=3))
        held = entry("threetears.calculator", "1.0.0", copy)
        selections = [held.select_copies(None, now=now, ttl=timedelta(seconds=45)) for _ in range(5)]
        assert {s.shown for s in selections} == {_WIDE}
        assert {s.routed_definitions["evd-tools"] for s in selections} == {_WIDE}
        # and once the old replica stops announcing, the narrow definition simply lapses
        later = now + timedelta(seconds=40)
        assert held.select_copies(None, _NARROW.schema_digest, now=later, ttl=timedelta(seconds=45)).routable == ()


class _OldRegistryProxyCallRequest(BaseModel):
    """a registry that predates ``input_schema_digest``, written out by hand.

    Not derived from the current model: the point is to read the wire the way an older
    deployment reads it.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_version: str
    arguments: dict[str, Any]
    context: dict[str, Any] | None = None
    pop: str | None = None
    result_subject: str | None = None
    deadline_seconds: float | None = None


class TestTheNewFieldNeverCrossesAsNull:
    """``input_schema_digest`` is omitted when unset, by the model itself, for every sender."""

    def test_an_unset_digest_is_not_serialized(self) -> None:
        """no key, not a null -- whichever way the sender serializes."""
        request = ProxyCallRequest(tool_name="t", tool_version="1", arguments={})
        assert "input_schema_digest" not in json.loads(request.model_dump_json())
        assert "input_schema_digest" not in request.model_dump()
        assert "input_schema_digest" not in request.model_dump(mode="json")

    def test_an_older_registry_accepts_a_request_without_a_digest(self) -> None:
        """the property as the lagging reader experiences it."""
        request = ProxyCallRequest(
            tool_name="t", tool_version="1", arguments={}, context=CallContext(agent_id=uuid7(), correlation_id=uuid7())
        )
        _OldRegistryProxyCallRequest.model_validate_json(request.model_dump_json())

    def test_an_older_registry_refuses_a_request_carrying_one(self) -> None:
        """the rollout constraint, pinned: the registry ships before any sender sets the field."""
        request = ProxyCallRequest(tool_name="t", tool_version="1", arguments={}, input_schema_digest="d")
        with pytest.raises(ValidationError):
            _OldRegistryProxyCallRequest.model_validate_json(request.model_dump_json())

    def test_an_unset_deadline_is_omitted_the_same_way(self) -> None:
        """the older optional follows the same rule, so no sender has to prune either by hand."""
        request = ProxyCallRequest(tool_name="t", tool_version="1", arguments={})
        assert "deadline_seconds" not in json.loads(request.model_dump_json())

    def test_unknown_fields_are_still_refused(self) -> None:
        """omitting nulls is not tolerating extras."""
        with pytest.raises(ValidationError):
            ProxyCallRequest.model_validate({"tool_name": "t", "tool_version": "1", "arguments": {}, "stray": 1})


class TestTheForwardedCallGainsNoKey:
    """a pod on any older release still parses what the proxy forwards."""

    @pytest.mark.asyncio
    async def test_the_forwarded_key_set_is_the_same_with_and_without_a_digest(self) -> None:
        """the digest is checked at the proxy and never forwarded."""
        catalog = ToolCatalog()
        await catalog.register(entry("threetears.calculator", "1.0.0", endpoint("pod-A", tool_definition=_NARROW)))
        proxy, nc = await _proxy(catalog)
        await _call(proxy, nc, make_authed_request())
        await _call(proxy, nc, make_authed_request(input_schema_digest=_NARROW.schema_digest))
        without, with_digest = (json.loads(call.kwargs["payload"]) for call in nc.request_raw.await_args_list)
        assert set(with_digest) == set(without)
        assert "input_schema_digest" not in with_digest
        assert set(without) <= {
            "tool_name",
            "tool_version",
            "arguments",
            "context",
            "proxy_assertion",
            "deadline_seconds",
        }
