"""tests for ``threetears.workspace.use`` -- WorkspaceUseTool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition

from threetears.agent.workspace.tools import workspace_use as workspace_use_module
from threetears.agent.workspace.tools.workspace_use import WorkspaceUseTool


@dataclass
class _FakeWorkspaceEntity:
    """minimal stand-in for :class:`Workspace` exposing id and name."""

    id: UUID
    name: str


class _FakeCollection:
    """records lookup calls and serves entities keyed by (agent_id, name)."""

    def __init__(self, entities: list[_FakeWorkspaceEntity]) -> None:
        self._entities = entities
        self.find_by_agent_calls: list[UUID] = []
        self.find_by_agent_and_name_calls: list[tuple[UUID, str]] = []

    async def find_by_agent(self, agent_id: UUID) -> list[_FakeWorkspaceEntity]:
        self.find_by_agent_calls.append(agent_id)
        return list(self._entities)

    async def find_by_agent_and_name(
        self, agent_id: UUID, name: str
    ) -> _FakeWorkspaceEntity | None:
        self.find_by_agent_and_name_calls.append((agent_id, name))
        for entity in self._entities:
            if entity.name == name:
                return entity
        return None


class _RecordingPin:
    """records calls to set_pin so we can assert correct arguments."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def set_pin(
        self,
        context: Any,
        workspace_id: UUID,
        workspace_name: str,
        pinned_by_actor_id: UUID,
    ) -> None:
        self.calls.append(
            {
                "context": context,
                "workspace_id": workspace_id,
                "workspace_name": workspace_name,
                "pinned_by_actor_id": pinned_by_actor_id,
            }
        )


class _FakeContext:
    """sentinel object returned by the context provider closure."""


@pytest.fixture()
def patch_set_pin(monkeypatch: pytest.MonkeyPatch) -> _RecordingPin:
    """replace pin.set_pin in the tool's module namespace with a recorder."""
    recorder = _RecordingPin()
    monkeypatch.setattr(workspace_use_module.pin, "set_pin", recorder.set_pin)
    return recorder


@pytest.mark.asyncio
async def test_execute_happy_path_pins_workspace(patch_set_pin: _RecordingPin) -> None:
    """found workspace pins via pin.set_pin with correct args; returns success."""
    agent_id = uuid4()
    workspace_id = uuid4()
    entity = _FakeWorkspaceEntity(id=workspace_id, name="main")
    coll = _FakeCollection([entity])
    fake_ctx = _FakeContext()
    tool = WorkspaceUseTool(
        workspace_collection=coll,
        agent_id=agent_id,
        context_provider=lambda: fake_ctx,
    )

    result = await tool.execute(name="main")

    assert result.success is True
    assert result.error is None
    assert result.content == "pinned workspace 'main'"
    assert coll.find_by_agent_and_name_calls == [(agent_id, "main")]
    assert coll.find_by_agent_calls == []
    assert len(patch_set_pin.calls) == 1
    call = patch_set_pin.calls[0]
    assert call["context"] is fake_ctx
    assert call["workspace_id"] == workspace_id
    assert call["workspace_name"] == "main"
    assert call["pinned_by_actor_id"] == agent_id


@pytest.mark.asyncio
async def test_execute_not_found_returns_error_with_available_names(
    patch_set_pin: _RecordingPin,
) -> None:
    """missing workspace yields error mentioning name and the available list."""
    agent_id = uuid4()
    available = [
        _FakeWorkspaceEntity(id=uuid4(), name="alpha"),
        _FakeWorkspaceEntity(id=uuid4(), name="beta"),
    ]
    coll = _FakeCollection(available)
    tool = WorkspaceUseTool(
        workspace_collection=coll,
        agent_id=agent_id,
        context_provider=lambda: _FakeContext(),
    )

    result = await tool.execute(name="missing")

    assert result.success is False
    assert result.error is not None
    assert "'missing'" in result.error
    assert "alpha" in result.error
    assert "beta" in result.error
    assert coll.find_by_agent_calls == [agent_id]
    assert patch_set_pin.calls == []


@pytest.mark.asyncio
async def test_execute_traps_collection_errors_as_data(
    patch_set_pin: _RecordingPin,
) -> None:
    """collection failures surface as ToolResult(success=False, error=...)."""

    class _Failing:
        async def find_by_agent_and_name(self, agent_id: UUID, name: str) -> Any:
            raise RuntimeError("db down")

        async def find_by_agent(self, agent_id: UUID) -> list[_FakeWorkspaceEntity]:
            return []

    tool = WorkspaceUseTool(
        workspace_collection=_Failing(),
        agent_id=uuid4(),
        context_provider=lambda: _FakeContext(),
    )

    result = await tool.execute(name="x")

    assert result.success is False
    assert result.error is not None
    assert "use failed" in result.error
    assert "db down" in result.error
    assert patch_set_pin.calls == []


@pytest.mark.asyncio
async def test_execute_traps_set_pin_errors_as_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """failures inside pin.set_pin surface as errors-as-data, not raise."""

    async def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("kv unavailable")

    monkeypatch.setattr(workspace_use_module.pin, "set_pin", _boom)
    coll = _FakeCollection([_FakeWorkspaceEntity(id=uuid4(), name="m")])
    tool = WorkspaceUseTool(
        workspace_collection=coll,
        agent_id=uuid4(),
        context_provider=lambda: _FakeContext(),
    )

    result = await tool.execute(name="m")

    assert result.success is False
    assert result.error is not None
    assert "use failed" in result.error
    assert "kv unavailable" in result.error


def test_mcp_name_is_exact_string() -> None:
    """mcp_name must equal ``threetears.workspace.use`` exactly."""
    tool = WorkspaceUseTool(
        workspace_collection=_FakeCollection([]),
        agent_id=uuid4(),
        context_provider=lambda: _FakeContext(),
    )

    assert tool.mcp_name() == "threetears.workspace.use"


def test_mcp_version_is_semver_string() -> None:
    """mcp_version returns a non-empty version string."""
    tool = WorkspaceUseTool(
        workspace_collection=_FakeCollection([]),
        agent_id=uuid4(),
        context_provider=lambda: _FakeContext(),
    )

    assert tool.mcp_version() == "1.0"


def test_mcp_schema_declares_required_name_string() -> None:
    """mcp_schema input requires a ``name`` string and forbids extra properties."""
    tool = WorkspaceUseTool(
        workspace_collection=_FakeCollection([]),
        agent_id=uuid4(),
        context_provider=lambda: _FakeContext(),
    )

    definition = tool.mcp_schema()

    assert isinstance(definition, MCPToolDefinition)
    assert definition.name == "threetears.workspace.use"
    assert definition.input_schema["type"] == "object"
    assert "name" in definition.input_schema["properties"]
    assert definition.input_schema["properties"]["name"]["type"] == "string"
    assert definition.input_schema["required"] == ["name"]
    assert definition.input_schema["additionalProperties"] is False
