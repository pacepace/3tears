"""regression guard: agent-side ToolServer no longer emits tool namespaces.

The original namespace-task-01 phase 2 + three-tier-task-01 phase F
contract had the agent-side ``ToolServer.register`` path call
``NamespaceCollection.save_entity`` to materialize one
``platform.namespaces`` row of type ``tool`` per registered tool.
That contract was retired in this v0.5.0 wave: the agent's L3 proxy
routes platform-scoped writes to the agent's own ``agent_<hex>``
schema, which has no ``namespaces`` table -- the write reliably
failed with ``relation "namespaces" does not exist`` whenever an
agent's tool pod tried to register.

The hub-side :class:`3tears.hub.tools.namespace_emitter
.ToolNamespaceEmitter` (subscribed to ``{ns}.tools.register``) is now
the SOLE writer of ``tool``-type namespace rows on the platform.

This test file exists to guard against an accidental re-introduction
of the agent-side write. Detailed coverage of the hub-side emitter
+ its post-emit access re-materialization dispatcher lives in
``14-eng-ai-bot/tests/unit/hub/tools/test_namespace_emitter.py`` and
``14-eng-ai-bot/tests/unit/hub/agents/test_access_materialization.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
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


class _RecordingNamespaceCollection:
    """recording stand-in for NamespaceCollection.

    save_entity / delete are AsyncMocks so the regression-guard
    assertions can confirm they were NEVER called from the agent-side
    ``ToolServer.register`` / ``deregister`` paths.
    """

    def __init__(self) -> None:
        """initialize the collection with empty call recorders."""
        self.entity_class = MagicMock(side_effect=lambda *_args, **_kw: MagicMock())
        self.save_entity = AsyncMock()
        self.delete = AsyncMock()


class TestAgentSideEmissionRetired:
    """regression guard: register / deregister never write through
    ``namespace_collection`` from the agent side. all tool-type
    namespace mutations are owned by the hub-side
    :class:`3tears.hub.tools.namespace_emitter.ToolNamespaceEmitter`
    going forward.
    """

    @pytest.mark.asyncio
    async def test_register_does_not_call_save_entity(self) -> None:
        """``ToolServer.register`` MUST NOT call
        ``namespace_collection.save_entity``. the hub-side emitter
        listens on ``{ns}.tools.register`` for the manifest the
        ``ToolServer`` already publishes and writes the row from
        there.
        """
        agent_id = uuid7()
        customer_id = uuid7()
        ns_collection = _RecordingNamespaceCollection()
        server = ToolServer(
            agent_id=agent_id,
            customer_id=customer_id,
            namespace_collection=ns_collection,
            nats_url="nats://test:4222",
        )

        server.register(_StubTool())

        ns_collection.save_entity.assert_not_awaited()
        ns_collection.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_register_with_no_namespace_collection_still_works(
        self,
    ) -> None:
        """passing ``namespace_collection=None`` is the standalone /
        test-fixture path and must continue to work without any
        emission attempt.
        """
        agent_id = uuid7()
        server = ToolServer(
            agent_id=agent_id,
            customer_id=None,
            namespace_collection=None,
            nats_url="nats://test:4222",
        )

        # no-raise + no recorder to inspect; success here means
        # ``register`` did not try to dereference the (None)
        # namespace_collection on its happy path.
        server.register(_StubTool())
