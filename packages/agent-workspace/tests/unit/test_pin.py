"""tests for workspace pin module (ContextItem-backed)."""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.context import ToolContextManager
from threetears.agent.workspace.pin import (
    PinnedWorkspace,
    clear_pin,
    get_pin,
    set_pin,
)
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

# share the agent-tools test utilities (FakePool, make_context_metadata, make_nats_mock)
_AGENT_TOOLS_TESTS = Path(__file__).resolve().parent.parent.parent.parent / "agent-tools" / "tests"
if str(_AGENT_TOOLS_TESTS) not in sys.path:
    sys.path.insert(0, str(_AGENT_TOOLS_TESTS))

from testing_utils import FakePool, make_context_metadata, make_nats_mock  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    """per-test SQLite backend with context_items table registered."""
    backend = SQLiteBackend(db_name=f"test_pin_{uuid.uuid4().hex[:8]}")
    backend.initialize(make_context_metadata())
    yield backend
    from threetears.core._bridge import drain, shutdown

    drain()
    shutdown()
    backend.reset()


@pytest.fixture()
def registry(l1_backend: SQLiteBackend) -> CollectionRegistry:
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1_backend)
    return reg


@pytest.fixture()
def config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


@pytest.fixture()
def pool() -> FakePool:
    return FakePool()


@pytest.fixture()
def collection(
    registry: CollectionRegistry,
    config: DefaultCoreConfig,
    pool: FakePool,
) -> ContextItemCollection:
    nats = make_nats_mock()
    coll = ContextItemCollection(registry, config, nats_client=nats)
    coll.l3_pool = pool
    return coll


@pytest.fixture()
def conversation_id() -> UUID:
    return UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture()
def user_id() -> UUID:
    return UUID("00000000-0000-0000-0000-0000000000aa")


@pytest.fixture()
def ctx(
    collection: ContextItemCollection,
    conversation_id: UUID,
    user_id: UUID,
) -> ToolContextManager:
    return ToolContextManager(collection, conversation_id, user_id)


# ---------------------------------------------------------------------------
# set_pin / get_pin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_then_get_returns_pinned_workspace(ctx: ToolContextManager) -> None:
    """set_pin + get_pin round-trip yields PinnedWorkspace with correct types."""
    workspace_id = UUID("11111111-1111-1111-1111-111111111111")
    actor_id = UUID("22222222-2222-2222-2222-222222222222")

    await set_pin(ctx, workspace_id, "main-workspace", actor_id)
    pin = await get_pin(ctx)

    assert pin is not None
    assert isinstance(pin, PinnedWorkspace)
    assert isinstance(pin.workspace_id, UUID)
    assert isinstance(pin.workspace_name, str)
    assert isinstance(pin.date_pinned, datetime)
    assert isinstance(pin.pinned_by_actor_id, UUID)
    assert pin.workspace_id == workspace_id
    assert pin.workspace_name == "main-workspace"
    assert pin.pinned_by_actor_id == actor_id


@pytest.mark.asyncio
async def test_get_pin_empty_returns_none(ctx: ToolContextManager) -> None:
    """get_pin on untouched context returns None."""
    result = await get_pin(ctx)

    assert result is None


@pytest.mark.asyncio
async def test_set_pin_replaces_existing_pin(ctx: ToolContextManager) -> None:
    """calling set_pin twice replaces the first pin (only one current pin)."""
    first_workspace = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    second_workspace = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    actor_id = UUID("22222222-2222-2222-2222-222222222222")

    await set_pin(ctx, first_workspace, "first", actor_id)
    await set_pin(ctx, second_workspace, "second", actor_id)

    pin = await get_pin(ctx)
    pin_items = [item for item in ctx.items if item["context_type"] == "workspace_pin" and item["key"] == "current"]

    assert pin is not None
    assert pin.workspace_id == second_workspace
    assert pin.workspace_name == "second"
    assert len(pin_items) == 1


# ---------------------------------------------------------------------------
# clear_pin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_pin_removes_pin(ctx: ToolContextManager) -> None:
    """clear_pin removes the pin entry; subsequent get_pin returns None."""
    workspace_id = UUID("11111111-1111-1111-1111-111111111111")
    actor_id = UUID("22222222-2222-2222-2222-222222222222")
    await set_pin(ctx, workspace_id, "workspace", actor_id)

    await clear_pin(ctx)

    assert await get_pin(ctx) is None
    assert not any(item["context_type"] == "workspace_pin" for item in ctx.items)


@pytest.mark.asyncio
async def test_clear_pin_noop_when_absent(ctx: ToolContextManager) -> None:
    """clear_pin is idempotent when no pin exists."""
    await clear_pin(ctx)

    assert await get_pin(ctx) is None


@pytest.mark.asyncio
async def test_clear_pin_idempotent_twice(ctx: ToolContextManager) -> None:
    """clear_pin called twice on a set pin does not raise on second call."""
    workspace_id = UUID("11111111-1111-1111-1111-111111111111")
    actor_id = UUID("22222222-2222-2222-2222-222222222222")
    await set_pin(ctx, workspace_id, "workspace", actor_id)

    await clear_pin(ctx)
    await clear_pin(ctx)

    assert await get_pin(ctx) is None


# ---------------------------------------------------------------------------
# border conversion round-trips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uuid_round_trip_preserves_identity(ctx: ToolContextManager) -> None:
    """workspace_id and pinned_by_actor_id preserve UUID identity through storage."""
    workspace_id = UUID("01912345-6789-7abc-8def-0123456789ab")
    actor_id = UUID("01912345-6789-7abc-8def-fedcba987654")

    await set_pin(ctx, workspace_id, "ws", actor_id)
    pin = await get_pin(ctx)

    assert pin is not None
    assert pin.workspace_id == workspace_id
    assert pin.pinned_by_actor_id == actor_id


@pytest.mark.asyncio
async def test_datetime_round_trip_preserves_utc(ctx: ToolContextManager) -> None:
    """date_pinned returns with tzinfo set to UTC."""
    workspace_id = UUID("11111111-1111-1111-1111-111111111111")
    actor_id = UUID("22222222-2222-2222-2222-222222222222")
    before = datetime.now(UTC)

    await set_pin(ctx, workspace_id, "ws", actor_id)
    pin = await get_pin(ctx)
    after = datetime.now(UTC)

    assert pin is not None
    assert pin.date_pinned.tzinfo is not None
    assert pin.date_pinned.utcoffset() == UTC.utcoffset(None)
    assert before <= pin.date_pinned <= after


@pytest.mark.asyncio
async def test_stored_content_is_string_at_border(ctx: ToolContextManager) -> None:
    """UUIDs and datetime are stored as strings (JSON-safe) at the storage border."""
    workspace_id = UUID("11111111-1111-1111-1111-111111111111")
    actor_id = UUID("22222222-2222-2222-2222-222222222222")

    await set_pin(ctx, workspace_id, "ws", actor_id)
    item = next(i for i in ctx.items if i["context_type"] == "workspace_pin" and i["key"] == "current")

    assert isinstance(item["content"], str)
    assert item["content"] == str(workspace_id)
    assert isinstance(item["metadata"]["pinned_by_actor_id"], str)
    assert item["metadata"]["pinned_by_actor_id"] == str(actor_id)
    assert isinstance(item["metadata"]["date_pinned"], str)
    # parseable as ISO
    datetime.fromisoformat(item["metadata"]["date_pinned"])


# ---------------------------------------------------------------------------
# cross-pod survival
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_pod_survival_via_shared_collection(
    collection: ContextItemCollection,
    conversation_id: UUID,
    user_id: UUID,
) -> None:
    """pin written on pod A is visible to pod B after load_context.

    uses the same underlying :class:`ContextItemCollection` (shared L3 via
    ``FakePool`` + shared L2 via the in-memory NATS KV stub) to simulate two
    pods, then constructs a fresh :class:`ToolContextManager` with cold L1
    local projection to confirm the pin round-trips end-to-end.
    """
    workspace_id = UUID("11111111-1111-1111-1111-111111111111")
    actor_id = UUID("22222222-2222-2222-2222-222222222222")

    # pod A: sets the pin, persists through collection (L1 + L2 stub + L3 FakePool)
    ctx_a = ToolContextManager(collection, conversation_id, user_id)
    await set_pin(ctx_a, workspace_id, "shared-ws", actor_id)

    # pod B: fresh manager with empty _items; hydrates via load_context
    ctx_b = ToolContextManager(collection, conversation_id, user_id)
    assert ctx_b.items == []

    await ctx_b.load_context()
    pin = await get_pin(ctx_b)

    assert pin is not None
    assert pin.workspace_id == workspace_id
    assert pin.workspace_name == "shared-ws"
    assert pin.pinned_by_actor_id == actor_id
    assert pin.date_pinned.tzinfo is not None
