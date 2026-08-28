"""regression guard: the pod writes no ``namespaces`` row for its own tools.

The original namespace-task-01 phase 2 + three-tier-task-01 phase F contract
had ``ToolServer`` materialize one ``namespaces`` row of type ``tool`` per
registered tool, and delete it again on deregister, through a
``namespace_collection`` threaded in at construction. Both halves are gone,
for two independent reasons:

1. **The write could never land.** The agent's L3 proxy routes platform-scoped
   writes to the agent's own ``agent_<hex>`` schema, which has no
   ``namespaces`` table, so the emit failed with ``relation "namespaces" does
   not exist`` on every agent tool pod. The hub-side
   ``aibots.hub.tools.namespace_emitter.ToolNamespaceEmitter``, subscribed to
   ``{ns}.tools.register``, became the SOLE writer of ``tool``-type rows --
   it reconciles removals off the same manifest and requires a verified
   signature, which a pod cannot produce about itself.
2. **The delete RAISED on every call.** It passed a bare ``UUID`` to
   ``NamespaceCollection.delete``, whose ``primary_key_column`` is the
   composite ``("row_scope", "namespace_id")``; ``normalize_pk`` refuses that
   arity. Three live call sites reach ``deregister_tool`` through stale-tool
   pruning, so a pod dropping a tool threw. The only test covering it replaced
   the collection with an ``AsyncMock``, which accepts any arity and hid it --
   which is why the tests below drive the real public methods and assert on
   what reaches NATS, rather than on a mock's call log.

Detailed coverage of the hub-side emitter and its post-emit access
re-materialization lives in
``14-eng-ai-bot/tests/unit/hub/tools/test_namespace_emitter.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.server import ToolServer


class _StubTool(TearsTool):
    """stub TearsTool for the regression-guard tests."""

    def __init__(self, name: str = "test.ns_stub", version: str = "1.0") -> None:
        """initialize stub tool.

        :param name: namespaced tool name
        :ptype name: str
        :param version: version string
        :ptype version: str
        """
        self._name = name
        self._version = version

    async def execute(self, **kwargs: Any) -> ToolResult:
        """execute stub tool.

        :param kwargs: arguments
        :ptype kwargs: Any
        :return: empty result
        :rtype: ToolResult
        """
        return ToolResult(success=True, content="")

    def mcp_schema(self) -> MCPToolDefinition:
        """return mcp schema.

        :return: schema
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self._name,
            version=self._version,
            description="stub tool for regression-guard tests",
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        """return mcp name.

        :return: name
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """return version.

        :return: version
        :rtype: str
        """
        return self._version


def _server_with_nats() -> tuple[ToolServer, AsyncMock]:
    """build a pod attached to a recording NATS client.

    :return: the pod and the NATS double it publishes through
    :rtype: tuple[ToolServer, AsyncMock]
    """
    nats = AsyncMock()
    server = ToolServer(
        agent_id=uuid7(),
        customer_id=uuid7(),
        nats_client=nats,
    )
    return server, nats


class TestThePodHoldsNoNamespaceCollection:
    """construction refuses the handle the retired emitter needed."""

    def test_constructor_rejects_a_namespace_collection(self) -> None:
        """``ToolServer`` no longer accepts ``namespace_collection``.

        the parameter's removal is the point: while it existed, every
        production caller had to decide whether to pass a Collection, and the
        two callers that did got a delete that raised.

        :return: nothing
        :rtype: None
        """
        with pytest.raises(TypeError, match="namespace_collection"):
            ToolServer(nats_url="nats://test:4222", namespace_collection=None)  # type: ignore[call-arg]


class TestDeregisterPublishesAndNothingElse:
    """the manifest publish is the whole of what deregistration does."""

    @pytest.mark.asyncio
    async def test_deregister_of_a_registered_tool_does_not_raise(self) -> None:
        """dropping a tool completes and reports the removal.

        this is the defect itself: the removed ``_delete_tool_namespace``
        raised ``ValueError: primary key arity mismatch`` here on every call
        against a real composite-keyed Collection.

        :return: nothing
        :rtype: None
        """
        server, _nats = _server_with_nats()
        await server.register_tool(_StubTool())

        removed = await server.deregister_tool("test.ns_stub")

        assert removed is True
        assert server.tool_names == ()

    @pytest.mark.asyncio
    async def test_deregister_publishes_exactly_one_reduced_manifest(self) -> None:
        """the only wire traffic a deregister produces is the manifest.

        :return: nothing
        :rtype: None
        """
        server, nats = _server_with_nats()
        await server.register_tool(_StubTool())
        nats.reset_mock()

        await server.deregister_tool("test.ns_stub")

        assert nats.publish.await_count == 1
        published = nats.publish.await_args
        subject = published.args[0] if published.args else published.kwargs["subject"]
        assert str(subject).endswith("tools.register")

    @pytest.mark.asyncio
    async def test_deregister_of_an_unknown_tool_is_a_reported_no_op(self) -> None:
        """an unregistered name removes nothing and publishes nothing.

        :return: nothing
        :rtype: None
        """
        server, nats = _server_with_nats()

        removed = await server.deregister_tool("test.never_registered")

        assert removed is False
        assert nats.publish.await_count == 0
