"""Unit tests for the schedule + webhook tool factories.

Uses in-memory fakes for both Collections and the registry so the
per-tool behaviour (validation, ACL, cross-user isolation, cap-of-10,
cycle detection on context_from) is exercised without spinning up
Postgres. End-to-end lifecycle lands in the integration suite.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from uuid_utils import uuid7

from threetears.agent.wake.entities import WakeScheduleEntity, WebhookSubscriptionEntity
from threetears.agent.wake.tools.schedule_tools import (
    DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
    PROTECTED_REFUSAL,
    WakeRegistryClient,
    load_wake_schedule_create_tool,
    load_wake_schedule_delete_tool,
    load_wake_schedule_list_tool,
    load_wake_schedule_pause_tool,
    load_wake_schedule_resume_tool,
    load_wake_schedule_update_tool,
    load_wake_yield_tool,
)
from threetears.agent.wake.tools.webhook_tools import (
    load_webhook_subscription_create_tool,
    load_webhook_subscription_delete_tool,
    load_webhook_subscription_list_tool,
    load_webhook_subscription_pause_tool,
    load_webhook_subscription_resume_tool,
    load_webhook_subscription_rotate_secret_tool,
    load_webhook_subscription_update_tool,
)


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


# Minimal advisory-lock + count seam create_schedule_serialized drives:
# execute(advisory-lock) -> fetchval(count-active) -> save_entity(conn=self).
# Not a ``_Fake*`` (so it is exempt from the fake-parity walker): it mocks
# only the three-method txn-connection slice, not the full asyncpg.Connection
# surface, which the parity check would (correctly) demand in full. The unit
# tests are single-threaded so the in-memory store IS the serialization point.
class _CapLockConn:
    """asyncpg-connection slice over the schedule collection's rows."""

    def __init__(self, collection: _FakeScheduleCollection) -> None:
        self._collection = collection

    def transaction(self) -> _CapLockConn:
        return self

    async def __aenter__(self) -> _CapLockConn:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def execute(self, sql: str, *args: Any) -> str:
        # Advisory-lock acquisition is a no-op in the single-threaded test.
        del sql, args
        return "SELECT 1"

    async def fetchval(self, sql: str, *args: Any) -> int:
        del sql
        agent_id = args[0]
        # resume_schedule_serialized passes (agent_id, schedule_id) and counts
        # active rows EXCLUDING the schedule being resumed;
        # create_schedule_serialized passes (agent_id,) only. Protected rows
        # never count.
        exclude_schedule_id = args[1] if len(args) > 1 else None
        return sum(
            1
            for sid, r in self._collection.rows.items()
            if r["agent_id"] == agent_id
            and r["status"] == "active"
            and not r.get("protected", False)
            and (exclude_schedule_id is None or sid[1] != exclude_schedule_id)
        )


# asyncpg.Pool ``acquire()`` slice (see _CapLockConn note re: not _Fake*).
class _CapLockPool:
    """asyncpg-pool slice yielding a :class:`_CapLockConn`."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def acquire(self) -> _CapLockConn:
        return _CapLockConn(self._collection)


# parity-with: threetears.agent.wake.collections.WakeScheduleCollection
# parity-exempt: WakeScheduleCollection subset for the schedule tool factory unit tests; the tools call only create/save_entity/find_for_agent/list_for_agent/delete/pause/resume so the cache+l2/l3 SQL methods on the production class are not part of the tool API contract
class _FakeScheduleCollection:
    """In-memory stand-in for the public surface of WakeScheduleCollection."""

    def __init__(self) -> None:
        self.rows: dict[tuple[UUID, UUID], dict[str, Any]] = {}
        # create_schedule_serialized takes the cap-lock against this pool.
        self.l3_pool = _CapLockPool(self)

    def create(self, data: dict[str, Any]) -> WakeScheduleEntity:
        return WakeScheduleEntity(dict(data), is_new=True, collection=None)

    async def save_entity(self, entity: Any, *, conn: Any = None) -> int:
        del conn
        data = entity.to_dict()
        self.rows[(data["conversation_id"], data["schedule_id"])] = dict(data)
        return 1

    async def find_for_agent(self, agent_id: UUID, schedule_id: UUID) -> WakeScheduleEntity | None:
        for (_conv, sid), row in self.rows.items():
            if sid == schedule_id and row["agent_id"] == agent_id:
                return WakeScheduleEntity(dict(row), is_new=False, collection=None)
        return None

    async def list_for_agent(self, agent_id: UUID, *, include_paused: bool = True) -> list[WakeScheduleEntity]:
        statuses = {"active", "paused"} if include_paused else {"active"}
        return [
            WakeScheduleEntity(dict(r), is_new=False, collection=None)
            for r in self.rows.values()
            if r["agent_id"] == agent_id and r["status"] in statuses
        ]

    async def delete(self, entity_id: Any) -> bool:
        self.rows.pop(entity_id, None)
        return True

    async def pause(self, conversation_id: UUID, schedule_id: UUID) -> None:
        row = self.rows.get((conversation_id, schedule_id))
        if row is None:
            return
        row["status"] = "paused"
        row["next_fire_at"] = None

    async def resume(
        self,
        conversation_id: UUID,
        schedule_id: UUID,
        *,
        next_fire_at: datetime,
        conn: Any = None,
    ) -> None:
        del conn
        row = self.rows.get((conversation_id, schedule_id))
        if row is None:
            return
        row["status"] = "active"
        row["next_fire_at"] = next_fire_at


# parity-with: threetears.agent.wake.collections.WebhookSubscriptionCollection
# parity-exempt: WebhookSubscriptionCollection subset for the webhook tool factory unit tests; the tools call only create/save_entity/find_for_agent/list_for_agent/delete/pause/resume/rotate_secret and the cache+l2/l3 + find_by_id methods are exercised in the integration suite instead
class _FakeSubscriptionsCollection:
    """In-memory stand-in for WebhookSubscriptionCollection."""

    def __init__(self) -> None:
        self.rows: dict[tuple[UUID, UUID], dict[str, Any]] = {}
        self.saved_conns: list[Any] = []
        # a new wake conversation is made in a transaction on this pool
        self.l3_pool = _CapLockPool(self)

    def create(self, data: dict[str, Any]) -> WebhookSubscriptionEntity:
        return WebhookSubscriptionEntity(dict(data), is_new=True, collection=None)

    async def save_entity(self, entity: Any, *, conn: Any = None) -> int:
        self.saved_conns.append(conn)
        data = entity.to_dict()
        self.rows[(data["conversation_id"], data["subscription_id"])] = dict(data)
        return 1

    async def find_for_agent(self, agent_id: UUID, subscription_id: UUID) -> WebhookSubscriptionEntity | None:
        for (_conv, sid), row in self.rows.items():
            if sid == subscription_id and row["agent_id"] == agent_id:
                return WebhookSubscriptionEntity(dict(row), is_new=False, collection=None)
        return None

    async def list_for_agent(self, agent_id: UUID) -> list[WebhookSubscriptionEntity]:
        return [
            WebhookSubscriptionEntity(dict(r), is_new=False, collection=None)
            for r in self.rows.values()
            if r["agent_id"] == agent_id
        ]

    async def delete(self, entity_id: Any) -> bool:
        self.rows.pop(entity_id, None)
        return True

    async def pause(self, conversation_id: UUID, subscription_id: UUID) -> None:
        row = self.rows.get((conversation_id, subscription_id))
        if row is not None:
            row["status"] = "paused"

    async def resume(self, conversation_id: UUID, subscription_id: UUID) -> None:
        row = self.rows.get((conversation_id, subscription_id))
        if row is not None:
            row["status"] = "active"

    async def rotate_secret(
        self,
        conversation_id: UUID,
        subscription_id: UUID,
        *,
        new_ciphertext: bytes,
    ) -> None:
        row = self.rows.get((conversation_id, subscription_id))
        if row is not None:
            row["secret_ciphertext"] = bytes(new_ciphertext)


# parity-with: threetears.agent.wake.types.WakeConversations
class _FakeWakeConversations:
    """Records the wake conversations it makes; knows which ones are the agent's."""

    def __init__(self, agent_id: UUID, existing: set[UUID] | None = None) -> None:
        self.agent_id = agent_id
        self.known: set[UUID] = set(existing or ())
        self.made: list[dict[str, Any]] = []

    async def create(
        self,
        *,
        parent_conversation_id: UUID,
        user_id: UUID,
        agent_id: UUID,
        name: str | None,
        conn: Any,
    ) -> UUID:
        new_id = _new_uuid()
        self.made.append(
            {
                "conversation_id": new_id,
                "parent_conversation_id": parent_conversation_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "name": name,
                "conn": conn,
            }
        )
        self.known.add(new_id)
        return new_id

    async def is_wake_conversation(self, *, agent_id: UUID, conversation_id: UUID) -> bool:
        return agent_id == self.agent_id and conversation_id in self.known


# parity-with: threetears.agent.wake.tools.schedule_tools.WakeRegistryClient
# parity-exempt: in-memory WakeRegistryClient implementation subclassing the production protocol; the fake declares its parity via subclass declaration but the strict walker also requires this entry to satisfy the cross-file Protocol lookup performed during walker analysis
class _FakeRegistry(WakeRegistryClient):
    """In-memory ACL + skill-name registry."""

    def __init__(
        self,
        *,
        permitted_skills: set[UUID] | None = None,
        names: dict[UUID, str] | None = None,
    ) -> None:
        self.permitted = permitted_skills or set()
        self.names = names or {}

    async def acl_permits_skill(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        skill_id: UUID,
    ) -> bool:
        del user_id, agent_id
        return skill_id in self.permitted

    async def skill_name_for_id(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        skill_id: UUID,
    ) -> str | None:
        del user_id, agent_id
        return self.names.get(skill_id)


# parity-with: threetears.agent.wake.entities.EncryptionService
# parity-exempt: identity-encryption stand-in matching the two-method EncryptionService Protocol; production parity is declared via the parity-with marker on the class but the strict walker requires this entry for the cross-file lookup
class _FakeEncryption:
    """Identity-encryption stand-in (returns plaintext unchanged)."""

    def encrypt(self, plaintext: bytes) -> bytes:
        return bytes(plaintext)

    def decrypt(self, ciphertext: bytes) -> str:
        return ciphertext.decode("utf-8")


# ---------------------------------------------------------------------------
# Schedule tool tests
# ---------------------------------------------------------------------------


@pytest.fixture
def actor() -> tuple[UUID, UUID, UUID]:
    return _new_uuid(), _new_uuid(), _new_uuid()


@pytest.mark.asyncio
async def test_schedule_create_persists_and_returns_catalog_line(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    registry = _FakeRegistry()
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=registry,
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "name": "every-10",
        },
    )
    assert result.startswith("[schedule:")
    assert "every-10" in result
    assert len(coll.rows) == 1


@pytest.mark.asyncio
async def test_schedule_create_rejects_invalid_schedule_config(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 0},  # invalid
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "positive int" in result
    assert len(coll.rows) == 0


@pytest.mark.asyncio
async def test_schedule_create_enforces_cap(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    # pre-seed up to the cap
    for _ in range(DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT):
        sid = _new_uuid()
        coll.rows[(conv_id, sid)] = {
            "conversation_id": conv_id,
            "schedule_id": sid,
            "user_id": user_id,
            "agent_id": agent_id,
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "execution_mode": "inline",
            "status": "active",
            "next_fire_at": datetime.now(UTC) + timedelta(hours=1),
            "missed_fire_policy": "coalesce",
            "date_created": datetime.now(UTC),
            "date_updated": datetime.now(UTC),
        }
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "the most allowed" in result


@pytest.mark.asyncio
async def test_schedule_create_rejects_unauthorized_skill(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    forbidden_skill = _new_uuid()
    registry = _FakeRegistry(permitted_skills=set())  # nothing allowed
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=registry,
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "skill_id": str(forbidden_skill),
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "not authorized" in result


@pytest.mark.asyncio
async def test_schedule_create_accepts_permitted_skill(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    skill_id = _new_uuid()
    registry = _FakeRegistry(
        permitted_skills={skill_id},
        names={skill_id: "diagnostic"},
    )
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=registry,
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "skill_id": f"[skill:{skill_id}]",
        },
    )
    assert "skill: diagnostic" in result
    assert len(coll.rows) == 1


@pytest.mark.asyncio
async def test_schedule_create_rejects_context_from_cycle(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    # seed A and B with A -> B -> A
    a_id = _new_uuid()
    b_id = _new_uuid()
    base_row = {
        "conversation_id": conv_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "schedule_type": "interval",
        "schedule_config": {"seconds": 600},
        "execution_mode": "inline",
        "status": "active",
        "next_fire_at": datetime.now(UTC) + timedelta(hours=1),
        "missed_fire_policy": "coalesce",
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    coll.rows[(conv_id, a_id)] = {
        **base_row,
        "schedule_id": a_id,
        "context_from_schedule_id": b_id,
    }
    coll.rows[(conv_id, b_id)] = {
        **base_row,
        "schedule_id": b_id,
        "context_from_schedule_id": a_id,
    }
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "context_from_schedule_id": str(a_id),
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "cycle" in result


@pytest.mark.asyncio
async def test_schedule_list_leaves_out_other_agents(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    other_agent = _new_uuid()
    coll = _FakeScheduleCollection()
    # row owned by another agent, in the same conversation
    other_id = _new_uuid()
    coll.rows[(conv_id, other_id)] = {
        "conversation_id": conv_id,
        "schedule_id": other_id,
        "user_id": user_id,
        "agent_id": other_agent,
        "schedule_type": "interval",
        "schedule_config": {"seconds": 600},
        "execution_mode": "inline",
        "status": "active",
        "next_fire_at": datetime.now(UTC),
        "missed_fire_policy": "coalesce",
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    tools = load_wake_schedule_list_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke({})
    assert "no wake schedules" in result


@pytest.mark.asyncio
async def test_schedule_pause_and_resume(actor: tuple[UUID, UUID, UUID]) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    schedule_id = _new_uuid()
    coll.rows[(conv_id, schedule_id)] = {
        "conversation_id": conv_id,
        "schedule_id": schedule_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "schedule_type": "interval",
        "schedule_config": {"seconds": 600},
        "execution_mode": "inline",
        "status": "active",
        "next_fire_at": datetime.now(UTC),
        "missed_fire_policy": "coalesce",
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    pause_tools = load_wake_schedule_pause_tool(
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
    )
    result = await pause_tools[0].ainvoke({"schedule_id": str(schedule_id)})
    assert "Paused" in result
    assert coll.rows[(conv_id, schedule_id)]["status"] == "paused"

    resume_tools = load_wake_schedule_resume_tool(
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
    )
    result = await resume_tools[0].ainvoke({"schedule_id": str(schedule_id)})
    assert "Resumed" in result
    assert coll.rows[(conv_id, schedule_id)]["status"] == "active"


def _active_schedule_row(
    *,
    conv_id: UUID,
    user_id: UUID,
    agent_id: UUID,
    status: str = "active",
) -> dict[str, Any]:
    return {
        "conversation_id": conv_id,
        "schedule_id": _new_uuid(),
        "user_id": user_id,
        "agent_id": agent_id,
        "schedule_type": "interval",
        "schedule_config": {"seconds": 600},
        "execution_mode": "inline",
        "status": status,
        "next_fire_at": datetime.now(UTC) if status == "active" else None,
        "missed_fire_policy": "coalesce",
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }


@pytest.mark.asyncio
async def test_schedule_resume_rejected_at_cap(actor: tuple[UUID, UUID, UUID]) -> None:
    """Re-activating a paused schedule respects the active-schedule cap.

    pause -> create-to-fill -> resume must NOT push the active count over
    the cap (PLACEMENT §1.9 / LR-50). With ``cap`` active schedules already
    present, resuming a paused one is rejected.
    """
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    cap = 2
    # Fill the conversation to the cap with active schedules.
    for _ in range(cap):
        row = _active_schedule_row(conv_id=conv_id, user_id=user_id, agent_id=agent_id)
        coll.rows[(conv_id, row["schedule_id"])] = row
    # A paused schedule the user now tries to resume.
    paused = _active_schedule_row(conv_id=conv_id, user_id=user_id, agent_id=agent_id, status="paused")
    paused_id = paused["schedule_id"]
    coll.rows[(conv_id, paused_id)] = paused

    resume_tools = load_wake_schedule_resume_tool(
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        max_active_schedules_per_agent=cap,
    )
    result = await resume_tools[0].ainvoke({"schedule_id": str(paused_id)})
    assert result.startswith("[TOOL ERROR]")
    assert "the most allowed" in result
    # Still paused -- the cap rejection blocked the flip.
    assert coll.rows[(conv_id, paused_id)]["status"] == "paused"


@pytest.mark.asyncio
async def test_schedule_resume_succeeds_under_cap(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Resuming a paused schedule works when the conversation is under cap."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    cap = 3
    # One active schedule -- under the cap of 3.
    row = _active_schedule_row(conv_id=conv_id, user_id=user_id, agent_id=agent_id)
    coll.rows[(conv_id, row["schedule_id"])] = row
    paused = _active_schedule_row(conv_id=conv_id, user_id=user_id, agent_id=agent_id, status="paused")
    paused_id = paused["schedule_id"]
    coll.rows[(conv_id, paused_id)] = paused

    resume_tools = load_wake_schedule_resume_tool(
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        max_active_schedules_per_agent=cap,
    )
    result = await resume_tools[0].ainvoke({"schedule_id": str(paused_id)})
    assert "Resumed" in result
    assert coll.rows[(conv_id, paused_id)]["status"] == "active"


@pytest.mark.asyncio
async def test_schedule_delete_blocks_other_agent(actor: tuple[UUID, UUID, UUID]) -> None:
    conv_id, user_id, agent_id = actor
    other_agent = _new_uuid()
    coll = _FakeScheduleCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = {
        "conversation_id": conv_id,
        "schedule_id": sid,
        "user_id": user_id,
        "agent_id": other_agent,  # another agent owns
        "schedule_type": "interval",
        "schedule_config": {"seconds": 600},
        "execution_mode": "inline",
        "status": "active",
        "next_fire_at": datetime.now(UTC),
        "missed_fire_policy": "coalesce",
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    tools = load_wake_schedule_delete_tool(
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
    )
    result = await tools[0].ainvoke({"schedule_id": str(sid)})
    assert "not found" in result
    # row still present -- delete refused for another agent's wake
    assert (conv_id, sid) in coll.rows


@pytest.mark.asyncio
async def test_schedule_update_changes_name_and_skill(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = {
        "conversation_id": conv_id,
        "schedule_id": sid,
        "user_id": user_id,
        "agent_id": agent_id,
        "schedule_type": "interval",
        "schedule_config": {"seconds": 600},
        "execution_mode": "inline",
        "status": "active",
        "next_fire_at": datetime.now(UTC),
        "missed_fire_policy": "coalesce",
        "name": "old",
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    new_skill = _new_uuid()
    registry = _FakeRegistry(
        permitted_skills={new_skill},
        names={new_skill: "summarise"},
    )
    tools = load_wake_schedule_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=registry,
    )
    result = await tools[0].ainvoke(
        {
            "schedule_id": str(sid),
            "name": "new",
            "skill_id": str(new_skill),
        },
    )
    assert "new" in result
    assert "skill: summarise" in result


def _seeded_schedule_row(
    *,
    conv_id: UUID,
    sid: UUID,
    user_id: UUID,
    agent_id: UUID,
    skill_id: UUID | None = None,
    name: str | None = "seed",
    context_from_schedule_id: UUID | None = None,
) -> dict[str, Any]:
    """Compact helper for the detach/no-op unit fixtures."""
    return {
        "conversation_id": conv_id,
        "schedule_id": sid,
        "user_id": user_id,
        "agent_id": agent_id,
        "skill_id": skill_id,
        "schedule_type": "interval",
        "schedule_config": {"seconds": 600},
        "execution_mode": "inline",
        "status": "active",
        "next_fire_at": datetime.now(UTC),
        "missed_fire_policy": "coalesce",
        "name": name,
        "context_from_schedule_id": context_from_schedule_id,
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }


@pytest.mark.asyncio
async def test_schedule_update_detach_skill_clears_attachment(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """detach_skill=true must clear an existing skill attachment."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    sid = _new_uuid()
    attached_skill = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_schedule_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
        skill_id=attached_skill,
    )
    tools = load_wake_schedule_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_id": str(sid),
            "detach_skill": True,
        },
    )
    assert not result.startswith("[TOOL ERROR]"), result
    assert coll.rows[(conv_id, sid)]["skill_id"] is None


@pytest.mark.asyncio
async def test_schedule_update_omitting_both_is_no_op_on_skill(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Neither skill_id nor detach_skill: leave the existing attachment alone."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    sid = _new_uuid()
    attached_skill = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_schedule_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
        skill_id=attached_skill,
    )
    tools = load_wake_schedule_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    # Only update name; skill_id/detach_skill both unset.
    result = await tools[0].ainvoke(
        {
            "schedule_id": str(sid),
            "name": "renamed",
        },
    )
    assert not result.startswith("[TOOL ERROR]"), result
    assert coll.rows[(conv_id, sid)]["skill_id"] == attached_skill


@pytest.mark.asyncio
async def test_schedule_update_rejects_skill_id_plus_detach_skill(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Passing skill_id AND detach_skill=true is contradictory; reject."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_schedule_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
        skill_id=None,
    )
    new_skill = _new_uuid()
    tools = load_wake_schedule_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(permitted_skills={new_skill}),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_id": str(sid),
            "skill_id": str(new_skill),
            "detach_skill": True,
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "cannot be combined" in result


@pytest.mark.asyncio
async def test_schedule_update_attach_new_skill(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Passing skill_id with no existing attachment attaches the skill."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_schedule_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
        skill_id=None,
    )
    new_skill = _new_uuid()
    registry = _FakeRegistry(
        permitted_skills={new_skill},
        names={new_skill: "summarise"},
    )
    tools = load_wake_schedule_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=registry,
    )
    result = await tools[0].ainvoke(
        {
            "schedule_id": str(sid),
            "skill_id": str(new_skill),
        },
    )
    assert "skill: summarise" in result
    assert coll.rows[(conv_id, sid)]["skill_id"] == new_skill


@pytest.mark.asyncio
async def test_schedule_update_clear_name(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """clear_name=true clears the schedule name."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_schedule_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
        name="old-name",
    )
    tools = load_wake_schedule_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_id": str(sid),
            "clear_name": True,
        },
    )
    assert not result.startswith("[TOOL ERROR]"), result
    assert coll.rows[(conv_id, sid)]["name"] is None


@pytest.mark.asyncio
async def test_schedule_update_rejects_name_plus_clear_name(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """name + clear_name=true is contradictory; reject."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_schedule_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
    )
    tools = load_wake_schedule_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_id": str(sid),
            "name": "new",
            "clear_name": True,
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "cannot be combined" in result


@pytest.mark.asyncio
async def test_schedule_update_detach_context_from(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """detach_context_from=true clears an existing context_from chain."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    sid = _new_uuid()
    upstream = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_schedule_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
        context_from_schedule_id=upstream,
    )
    tools = load_wake_schedule_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_id": str(sid),
            "detach_context_from": True,
        },
    )
    assert not result.startswith("[TOOL ERROR]"), result
    assert coll.rows[(conv_id, sid)]["context_from_schedule_id"] is None


@pytest.mark.asyncio
async def test_schedule_create_rejects_schedule_tag_as_skill_id(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Passing a [schedule:<uuid>] as skill_id surfaces a tag-confusion error."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    a_schedule = _new_uuid()
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "skill_id": f"[schedule:{a_schedule}]",
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "schedule tag" in result
    assert "skill_list" in result
    assert len(coll.rows) == 0


@pytest.mark.asyncio
async def test_schedule_update_rejects_schedule_tag_as_skill_id(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Same tag-confusion guard on the update tool's skill_id arg."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_schedule_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
    )
    tools = load_wake_schedule_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    a_schedule = _new_uuid()
    result = await tools[0].ainvoke(
        {
            "schedule_id": str(sid),
            "skill_id": f"[schedule:{a_schedule}]",
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "schedule tag" in result


# ---------------------------------------------------------------------------
# Webhook tool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_create_returns_plaintext_once(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    tools = load_webhook_subscription_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        encryption_service=_FakeEncryption(),
        registry=_FakeRegistry(),
        endpoint_base_url="https://example.test/webhooks",
    )
    result = await tools[0].ainvoke(
        {
            "task_prompt_template": "event: {{event.type}}",
            "name": "github-push",
        },
    )
    assert "[webhook:" in result
    assert "secret (copy now" in result
    assert "endpoint: https://example.test/webhooks/" in result
    assert len(coll.rows) == 1


@pytest.mark.asyncio
async def test_webhook_create_rejects_bad_template(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    tools = load_webhook_subscription_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        encryption_service=_FakeEncryption(),
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "task_prompt_template": "{% if unclosed",  # bad jinja
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "task_prompt_template" in result


@pytest.mark.asyncio
async def test_webhook_rotate_secret_replaces_ciphertext(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = {
        "conversation_id": conv_id,
        "subscription_id": sid,
        "user_id": user_id,
        "agent_id": agent_id,
        "default_skill_id": None,
        "name": "test",
        "secret_ciphertext": b"old",
        "allowed_source_pattern": None,
        "execution_mode": "inline",
        "task_prompt_template": "x",
        "verification_scheme": "generic_hmac_sha256",
        "status": "active",
        "rate_limit_per_minute": None,
        "last_fired_at": None,
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    tools = load_webhook_subscription_rotate_secret_tool(
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        encryption_service=_FakeEncryption(),
    )
    result = await tools[0].ainvoke({"subscription_id": str(sid)})
    assert "Rotated" in result
    assert "shown only once" in result
    assert coll.rows[(conv_id, sid)]["secret_ciphertext"] != b"old"


@pytest.mark.asyncio
async def test_webhook_pause_resume_delete_cycle(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = {
        "conversation_id": conv_id,
        "subscription_id": sid,
        "user_id": user_id,
        "agent_id": agent_id,
        "default_skill_id": None,
        "name": "test",
        "secret_ciphertext": b"sec",
        "allowed_source_pattern": None,
        "execution_mode": "inline",
        "task_prompt_template": "x",
        "verification_scheme": "generic_hmac_sha256",
        "status": "active",
        "rate_limit_per_minute": None,
        "last_fired_at": None,
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    pause = load_webhook_subscription_pause_tool(
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
    )
    await pause[0].ainvoke({"subscription_id": str(sid)})
    assert coll.rows[(conv_id, sid)]["status"] == "paused"

    resume = load_webhook_subscription_resume_tool(
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
    )
    await resume[0].ainvoke({"subscription_id": str(sid)})
    assert coll.rows[(conv_id, sid)]["status"] == "active"

    delete = load_webhook_subscription_delete_tool(
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
    )
    await delete[0].ainvoke({"subscription_id": str(sid)})
    assert (conv_id, sid) not in coll.rows


@pytest.mark.asyncio
async def test_webhook_update_changes_template_and_pattern(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = {
        "conversation_id": conv_id,
        "subscription_id": sid,
        "user_id": user_id,
        "agent_id": agent_id,
        "default_skill_id": None,
        "name": "test",
        "secret_ciphertext": b"sec",
        "allowed_source_pattern": None,
        "execution_mode": "inline",
        "task_prompt_template": "x",
        "verification_scheme": "generic_hmac_sha256",
        "status": "active",
        "rate_limit_per_minute": None,
        "last_fired_at": None,
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    tools = load_webhook_subscription_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "subscription_id": str(sid),
            "task_prompt_template": "new: {{event.type}}",
            "allowed_source_pattern": r"^10\.0\.",
        },
    )
    assert "[webhook:" in result
    row = coll.rows[(conv_id, sid)]
    assert row["task_prompt_template"] == "new: {{event.type}}"
    assert row["allowed_source_pattern"] == r"^10\.0\."


@pytest.mark.asyncio
async def test_webhook_list_leaves_out_other_agents(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    other = _new_uuid()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = {
        "conversation_id": conv_id,
        "subscription_id": sid,
        "user_id": user_id,
        "agent_id": other,
        "default_skill_id": None,
        "name": "other-agent",
        "secret_ciphertext": b"sec",
        "allowed_source_pattern": None,
        "execution_mode": "inline",
        "task_prompt_template": "x",
        "verification_scheme": "generic_hmac_sha256",
        "status": "active",
        "rate_limit_per_minute": None,
        "last_fired_at": None,
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    tools = load_webhook_subscription_list_tool(
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke({})
    assert "no webhook subscriptions" in result


def _seeded_subscription_row(
    *,
    conv_id: UUID,
    sid: UUID,
    user_id: UUID,
    agent_id: UUID,
    default_skill_id: UUID | None = None,
    name: str | None = "sub",
    allowed_source_pattern: str | None = None,
) -> dict[str, Any]:
    """Compact helper for the webhook detach/no-op unit fixtures."""
    return {
        "conversation_id": conv_id,
        "subscription_id": sid,
        "user_id": user_id,
        "agent_id": agent_id,
        "default_skill_id": default_skill_id,
        "name": name,
        "secret_ciphertext": b"sec",
        "allowed_source_pattern": allowed_source_pattern,
        "execution_mode": "inline",
        "task_prompt_template": "x",
        "verification_scheme": "generic_hmac_sha256",
        "status": "active",
        "rate_limit_per_minute": None,
        "last_fired_at": None,
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }


@pytest.mark.asyncio
async def test_webhook_update_detach_default_skill_clears_attachment(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """detach_default_skill=true clears the default_skill_id."""
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    sid = _new_uuid()
    attached = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_subscription_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
        default_skill_id=attached,
    )
    tools = load_webhook_subscription_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "subscription_id": str(sid),
            "detach_default_skill": True,
        },
    )
    assert not result.startswith("[TOOL ERROR]"), result
    assert coll.rows[(conv_id, sid)]["default_skill_id"] is None


@pytest.mark.asyncio
async def test_webhook_update_omitting_both_is_no_op_on_skill(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Neither default_skill_id nor detach_default_skill: leave attachment alone."""
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    sid = _new_uuid()
    attached = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_subscription_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
        default_skill_id=attached,
    )
    tools = load_webhook_subscription_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "subscription_id": str(sid),
            "name": "renamed",
        },
    )
    assert not result.startswith("[TOOL ERROR]"), result
    assert coll.rows[(conv_id, sid)]["default_skill_id"] == attached


@pytest.mark.asyncio
async def test_webhook_update_rejects_default_skill_id_plus_detach(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Passing both default_skill_id and detach_default_skill is rejected."""
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_subscription_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
    )
    new_skill = _new_uuid()
    tools = load_webhook_subscription_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(permitted_skills={new_skill}),
    )
    result = await tools[0].ainvoke(
        {
            "subscription_id": str(sid),
            "default_skill_id": str(new_skill),
            "detach_default_skill": True,
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "cannot be combined" in result


@pytest.mark.asyncio
async def test_webhook_update_attach_new_default_skill(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Passing default_skill_id with ACL allow attaches the new skill."""
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_subscription_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
    )
    new_skill = _new_uuid()
    registry = _FakeRegistry(
        permitted_skills={new_skill},
        names={new_skill: "named-skill"},
    )
    tools = load_webhook_subscription_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        registry=registry,
    )
    result = await tools[0].ainvoke(
        {
            "subscription_id": str(sid),
            "default_skill_id": str(new_skill),
        },
    )
    assert "skill: named-skill" in result
    assert coll.rows[(conv_id, sid)]["default_skill_id"] == new_skill


@pytest.mark.asyncio
async def test_webhook_update_clear_name(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """clear_name=true clears the subscription name."""
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_subscription_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
        name="old-name",
    )
    tools = load_webhook_subscription_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "subscription_id": str(sid),
            "clear_name": True,
        },
    )
    assert not result.startswith("[TOOL ERROR]"), result
    assert coll.rows[(conv_id, sid)]["name"] is None


@pytest.mark.asyncio
async def test_webhook_update_clear_allowed_source_pattern(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """clear_allowed_source_pattern=true clears the field."""
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    sid = _new_uuid()
    coll.rows[(conv_id, sid)] = _seeded_subscription_row(
        conv_id=conv_id,
        sid=sid,
        user_id=user_id,
        agent_id=agent_id,
        allowed_source_pattern=r"^10\.0\.",
    )
    tools = load_webhook_subscription_update_tool(
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "subscription_id": str(sid),
            "clear_allowed_source_pattern": True,
        },
    )
    assert not result.startswith("[TOOL ERROR]"), result
    assert coll.rows[(conv_id, sid)]["allowed_source_pattern"] is None


# ---------------------------------------------------------------------------
# wake_yield tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wake_yield_fires_setter() -> None:
    flag = {"set": False}

    def setter() -> None:
        flag["set"] = True

    tools = load_wake_yield_tool(is_wake_turn=lambda: True, set_yield_requested=setter)
    result = await tools[0].ainvoke({})
    assert "yielded" in result
    assert flag["set"] is True


def test_wake_yield_refuses_load_on_user_turn() -> None:
    with pytest.raises(RuntimeError):
        load_wake_yield_tool(
            is_wake_turn=lambda: False,
            set_yield_requested=lambda: None,
        )


# ---------------------------------------------------------------------------
# Agent scope, wake conversations, protected wakes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_create_makes_a_new_wake_conversation_whose_parent_is_the_caller(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    hooks = _FakeWakeConversations(agent_id)
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
        wake_conversations=hooks,
    )
    result = await tools[0].ainvoke(
        {"schedule_type": "interval", "schedule_config": {"seconds": 600}, "name": "mail"},
    )
    assert result.startswith("[schedule:"), result
    assert len(hooks.made) == 1
    made = hooks.made[0]
    assert made["parent_conversation_id"] == conv_id
    assert made["agent_id"] == agent_id
    assert made["name"] == "mail"
    # made on the create's own transaction, the one holding the agent's lock
    assert isinstance(made["conn"], _CapLockConn)
    ((row_conv, _sid),) = coll.rows
    assert row_conv == made["conversation_id"]
    assert f"[conversation:{made['conversation_id']}]" in result
    assert coll.rows[(row_conv, _sid)]["execution_mode"] == "spawn"


@pytest.mark.asyncio
async def test_schedule_create_adds_to_a_named_wake_conversation(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    existing = _new_uuid()
    coll = _FakeScheduleCollection()
    hooks = _FakeWakeConversations(agent_id, existing={existing})
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
        wake_conversations=hooks,
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "wake_conversation_id": f"[conversation:{existing}]",
        },
    )
    assert result.startswith("[schedule:"), result
    assert hooks.made == []
    ((row_conv, _sid),) = coll.rows
    assert row_conv == existing


@pytest.mark.asyncio
async def test_schedule_create_refuses_a_conversation_that_is_not_the_agents_wake_conversation(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
        wake_conversations=_FakeWakeConversations(agent_id),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "wake_conversation_id": str(conv_id),
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "not one of your wake conversations" in result
    assert coll.rows == {}


@pytest.mark.asyncio
async def test_schedule_create_without_hooks_refuses_a_named_wake_conversation(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "wake_conversation_id": str(_new_uuid()),
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert coll.rows == {}


@pytest.mark.asyncio
async def test_schedule_cap_counts_across_the_agents_conversations(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Two wakes in two conversations fill a cap of 2; protected wakes do not count."""
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    for conversation in (_new_uuid(), _new_uuid()):
        row = _active_schedule_row(conv_id=conversation, user_id=user_id, agent_id=agent_id)
        coll.rows[(conversation, row["schedule_id"])] = row
    life = _active_schedule_row(conv_id=_new_uuid(), user_id=user_id, agent_id=agent_id)
    life["protected"] = True
    coll.rows[(life["conversation_id"], life["schedule_id"])] = life
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
        max_active_schedules_per_agent=3,
    )
    first = await tools[0].ainvoke({"schedule_type": "interval", "schedule_config": {"seconds": 600}})
    assert first.startswith("[schedule:"), first
    second = await tools[0].ainvoke({"schedule_type": "interval", "schedule_config": {"seconds": 600}})
    assert second.startswith("[TOOL ERROR]")
    assert "3 active wakes" in second


@pytest.mark.asyncio
async def test_schedule_tools_reach_the_agents_wake_in_another_conversation(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    """Pause, resume and delete act on a wake that lives in a different conversation of the same agent."""
    _conv_id, user_id, agent_id = actor
    elsewhere = _new_uuid()
    coll = _FakeScheduleCollection()
    row = _active_schedule_row(conv_id=elsewhere, user_id=user_id, agent_id=agent_id)
    sid = row["schedule_id"]
    coll.rows[(elsewhere, sid)] = row

    pause = load_wake_schedule_pause_tool(agent_id=agent_id, schedules_collection=coll)  # type: ignore[arg-type]
    assert "Paused" in await pause[0].ainvoke({"schedule_id": str(sid)})
    assert coll.rows[(elsewhere, sid)]["status"] == "paused"

    resume = load_wake_schedule_resume_tool(agent_id=agent_id, schedules_collection=coll)  # type: ignore[arg-type]
    assert "Resumed" in await resume[0].ainvoke({"schedule_id": str(sid)})
    assert coll.rows[(elsewhere, sid)]["status"] == "active"

    delete = load_wake_schedule_delete_tool(agent_id=agent_id, schedules_collection=coll)  # type: ignore[arg-type]
    assert "Deleted" in await delete[0].ainvoke({"schedule_id": str(sid)})
    assert coll.rows == {}


@pytest.mark.asyncio
async def test_schedule_tools_refuse_another_agents_wake(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    row = _active_schedule_row(conv_id=conv_id, user_id=user_id, agent_id=agent_id)
    sid = row["schedule_id"]
    coll.rows[(conv_id, sid)] = row
    stranger = _new_uuid()

    pause = load_wake_schedule_pause_tool(agent_id=stranger, schedules_collection=coll)  # type: ignore[arg-type]
    resume = load_wake_schedule_resume_tool(agent_id=stranger, schedules_collection=coll)  # type: ignore[arg-type]
    update = load_wake_schedule_update_tool(
        user_id=user_id,
        agent_id=stranger,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    for tool_obj, payload in (
        (pause[0], {"schedule_id": str(sid)}),
        (resume[0], {"schedule_id": str(sid)}),
        (update[0], {"schedule_id": str(sid), "name": "taken"}),
    ):
        result = await tool_obj.ainvoke(payload)
        assert "not found" in result, result
    assert coll.rows[(conv_id, sid)]["status"] == "active"
    assert coll.rows[(conv_id, sid)].get("name") is None


@pytest.mark.asyncio
async def test_schedule_list_covers_every_conversation_of_the_agent(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    _conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    conversations = [_new_uuid(), _new_uuid()]
    for conversation in conversations:
        row = _active_schedule_row(conv_id=conversation, user_id=user_id, agent_id=agent_id)
        coll.rows[(conversation, row["schedule_id"])] = row
    tools = load_wake_schedule_list_tool(
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke({})
    assert "Found 2 schedules" in result
    for conversation in conversations:
        assert f"[conversation:{conversation}]" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["pause", "delete", "update"])
async def test_schedule_tools_refuse_a_protected_wake(
    actor: tuple[UUID, UUID, UUID],
    tool_name: str,
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    row = _active_schedule_row(conv_id=conv_id, user_id=user_id, agent_id=agent_id)
    row["protected"] = True
    sid = row["schedule_id"]
    coll.rows[(conv_id, sid)] = row
    before = dict(row)
    if tool_name == "pause":
        tools = load_wake_schedule_pause_tool(agent_id=agent_id, schedules_collection=coll)  # type: ignore[arg-type]
        payload: dict[str, Any] = {"schedule_id": str(sid)}
    elif tool_name == "delete":
        tools = load_wake_schedule_delete_tool(agent_id=agent_id, schedules_collection=coll)  # type: ignore[arg-type]
        payload = {"schedule_id": str(sid)}
    else:
        tools = load_wake_schedule_update_tool(
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=coll,  # type: ignore[arg-type]
            registry=_FakeRegistry(),
        )
        payload = {"schedule_id": str(sid), "schedule_config": {"seconds": 60}}
    result = await tools[0].ainvoke(payload)
    assert result == f"[TOOL ERROR] wake_schedule_{tool_name}: {PROTECTED_REFUSAL}"
    assert coll.rows[(conv_id, sid)] == before


@pytest.mark.asyncio
async def test_schedule_create_accepts_context_from_a_wake_in_another_conversation(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    upstream = _active_schedule_row(conv_id=_new_uuid(), user_id=user_id, agent_id=agent_id)
    coll.rows[(upstream["conversation_id"], upstream["schedule_id"])] = upstream
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "context_from_schedule_id": f"[schedule:{upstream['schedule_id']}]",
        },
    )
    assert result.startswith("[schedule:"), result
    created = [r for (c, _s), r in coll.rows.items() if c == conv_id]
    assert created[0]["context_from_schedule_id"] == upstream["schedule_id"]


@pytest.mark.asyncio
async def test_schedule_create_refuses_context_from_another_agents_wake(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeScheduleCollection()
    foreign = _active_schedule_row(conv_id=conv_id, user_id=user_id, agent_id=_new_uuid())
    coll.rows[(conv_id, foreign["schedule_id"])] = foreign
    tools = load_wake_schedule_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        schedules_collection=coll,  # type: ignore[arg-type]
        registry=_FakeRegistry(),
    )
    result = await tools[0].ainvoke(
        {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "context_from_schedule_id": str(foreign["schedule_id"]),
        },
    )
    assert result.startswith("[TOOL ERROR]")
    assert "not found" in result
    assert len(coll.rows) == 1


@pytest.mark.asyncio
async def test_webhook_create_makes_a_new_wake_conversation_in_the_same_transaction(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    conv_id, user_id, agent_id = actor
    coll = _FakeSubscriptionsCollection()
    hooks = _FakeWakeConversations(agent_id)
    tools = load_webhook_subscription_create_tool(
        conversation_id=conv_id,
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=coll,  # type: ignore[arg-type]
        encryption_service=_FakeEncryption(),
        registry=_FakeRegistry(),
        wake_conversations=hooks,
    )
    result = await tools[0].ainvoke({"task_prompt_template": "event: {{event.type}}", "name": "mail"})
    assert "[webhook:" in result, result
    assert len(hooks.made) == 1
    made = hooks.made[0]
    assert made["parent_conversation_id"] == conv_id
    ((row_conv, _sid),) = coll.rows
    assert row_conv == made["conversation_id"]
    # the row was saved on the connection the conversation was made on
    assert coll.saved_conns == [made["conn"]]


@pytest.mark.asyncio
async def test_webhook_tools_reach_the_agents_subscription_in_another_conversation(
    actor: tuple[UUID, UUID, UUID],
) -> None:
    _conv_id, user_id, agent_id = actor
    elsewhere = _new_uuid()
    coll = _FakeSubscriptionsCollection()
    sid = _new_uuid()
    coll.rows[(elsewhere, sid)] = _seeded_subscription_row(
        conv_id=elsewhere, sid=sid, user_id=user_id, agent_id=agent_id
    )

    pause = load_webhook_subscription_pause_tool(agent_id=agent_id, subscriptions_collection=coll)  # type: ignore[arg-type]
    assert "Paused" in await pause[0].ainvoke({"subscription_id": str(sid)})
    assert coll.rows[(elsewhere, sid)]["status"] == "paused"

    stranger_delete = load_webhook_subscription_delete_tool(agent_id=_new_uuid(), subscriptions_collection=coll)  # type: ignore[arg-type]
    assert "not found" in await stranger_delete[0].ainvoke({"subscription_id": str(sid)})
    assert (elsewhere, sid) in coll.rows
