"""Unit tests for the seven tool-factory functions in ``skills.tools``.

Uses in-memory fakes for both Collections and the registry client so
the per-tool behaviour (validation, ACL, cross-user isolation,
first-invoke-wins, prompt_mode rejection) is exercised without
spinning up Postgres. The full happy-path lifecycle (create -> list ->
introspect -> update -> delete) lands in the integration suite where
real Collections run.

Fake parity: ``_FakeSkillsCollection`` and ``_FakeInvocationsCollection``
declare their parity contract via subclass declaration so the canonical
fake-parity walker accepts them in ``strict`` mode.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from uuid_utils import uuid7

from threetears.agent.skills.entities import (
    AgentSkillEntity,
    AgentSkillInvocationEntity,
)
from threetears.agent.skills.tools import (
    SkillEligibleTool,
    SkillToolIntrospect,
    load_skill_create_tool,
    load_skill_delete_tool,
    load_skill_get_tool,
    load_skill_introspect_tool,
    load_skill_invoke_tool,
    load_skill_list_tool,
    load_skill_report_outcome_tool,
    load_skill_update_tool,
)


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


# --- Fakes ---


# parity-with: threetears.agent.skills.collections.AgentSkillCollection
class _FakeSkillsCollection:
    """In-memory stand-in for the public surface of :class:`AgentSkillCollection`.

    Implements the slice the tool factories call: ``create`` /
    ``save_entity`` / ``get`` / ``delete`` / ``find_by_name_for_user``
    / ``list_for_user`` / ``count_for_user``. Constructs entities with
    ``collection=None`` so the cache-write path in
    :meth:`BaseEntity.__init__` falls back to transient dict storage
    -- no L1 / L2 / L3 wiring needed for unit tests.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[UUID, UUID], dict[str, Any]] = {}

    def create(self, data: dict[str, Any]) -> AgentSkillEntity:
        return AgentSkillEntity(dict(data), is_new=True, collection=None)

    async def save_entity(self, entity: Any, **kwargs: Any) -> int:
        data = entity.to_dict()
        self.rows[(data["agent_id"], data["skill_id"])] = dict(data)
        return 1

    async def get(self, entity_id: Any) -> AgentSkillEntity | None:
        agent_id, skill_id = entity_id
        row = self.rows.get((agent_id, skill_id))
        if row is None:
            return None
        return AgentSkillEntity(dict(row), is_new=False, collection=None)

    async def delete(self, entity_id: Any) -> bool:
        agent_id, skill_id = entity_id
        self.rows.pop((agent_id, skill_id), None)
        return True

    async def find_by_name_for_user(
        self,
        agent_id: UUID,
        user_id: UUID,
        name: str,
    ) -> AgentSkillEntity | None:
        for row in self.rows.values():
            if row["agent_id"] == agent_id and row["user_id"] == user_id and row["name"] == name:
                return AgentSkillEntity(dict(row), is_new=False, collection=None)
        return None

    async def list_for_user(
        self,
        agent_id: UUID,
        user_id: UUID,
        *,
        enabled_only: bool = True,
        tag_filter: Any = None,
        query: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[AgentSkillEntity]:
        results: list[AgentSkillEntity] = []
        needle = (query or "").lower().strip() if query else None
        for row in self.rows.values():
            if row["agent_id"] != agent_id or row["user_id"] != user_id:
                continue
            if enabled_only and not row.get("enabled", True):
                continue
            if tag_filter:
                row_tags = list(row.get("tags") or [])
                if not any(t in row_tags for t in tag_filter):
                    continue
            if needle:
                hay = f"{row.get('name', '')} {row.get('summary', '')} {row.get('body', '') or ''}".lower()
                if needle not in hay:
                    continue
            results.append(AgentSkillEntity(dict(row), is_new=False, collection=None))
        return results[offset : offset + limit]

    async def count_for_user(
        self,
        agent_id: UUID,
        user_id: UUID,
        *,
        enabled_only: bool = True,
        tag_filter: Any = None,
        query: str | None = None,
    ) -> int:
        needle = (query or "").lower().strip() if query else None
        count = 0
        for row in self.rows.values():
            if row["agent_id"] != agent_id or row["user_id"] != user_id:
                continue
            if enabled_only and not row.get("enabled", True):
                continue
            if tag_filter:
                row_tags = list(row.get("tags") or [])
                if not any(t in row_tags for t in tag_filter):
                    continue
            if needle:
                hay = f"{row.get('name', '')} {row.get('summary', '')} {row.get('body', '') or ''}".lower()
                if needle not in hay:
                    continue
            count += 1
        return count

    async def increment_outcome_counts(
        self,
        agent_id: UUID,
        skill_id: UUID,
        outcome: str,
    ) -> None:
        row = self.rows.get((agent_id, skill_id))
        if row is None:
            return
        if outcome == "success":
            row["success_count"] = row.get("success_count", 0) + 1
        elif outcome == "failure":
            row["failure_count"] = row.get("failure_count", 0) + 1
            row["last_failure_at"] = datetime.now(UTC)
        else:
            raise ValueError(f"increment_outcome_counts: outcome must be 'success' or 'failure'; got {outcome!r}")


# parity-with: threetears.agent.skills.collections.AgentSkillInvocationCollection
class _FakeInvocationsCollection:
    """In-memory stand-in for :class:`AgentSkillInvocationCollection`."""

    def __init__(self) -> None:
        self.rows: dict[tuple[UUID, UUID], dict[str, Any]] = {}

    def create(self, data: dict[str, Any]) -> AgentSkillInvocationEntity:
        return AgentSkillInvocationEntity(dict(data), is_new=True, collection=None)

    async def save_entity(self, entity: Any, **kwargs: Any) -> int:
        data = entity.to_dict()
        self.rows[(data["agent_id"], data["invocation_id"])] = dict(data)
        return 1

    async def record(
        self,
        agent_id: UUID,
        invocation: AgentSkillInvocationEntity,
    ) -> None:
        await self.save_entity(invocation)

    async def list_for_conversation(
        self,
        agent_id: UUID,
        conversation_id: UUID,
        *,
        limit: int = 20,
    ) -> list[AgentSkillInvocationEntity]:
        matches = [
            AgentSkillInvocationEntity(dict(row), is_new=False, collection=None)
            for row in self.rows.values()
            if row["agent_id"] == agent_id and row["conversation_id"] == conversation_id
        ]
        matches.sort(key=lambda e: e.invoked_at, reverse=True)
        return matches[:limit]

    async def set_outcome(
        self,
        agent_id: UUID,
        invocation_id: UUID,
        *,
        outcome: str,
        source: str,
    ) -> None:
        row = self.rows.get((agent_id, invocation_id))
        if row is None:
            return
        row["outcome"] = outcome
        row["outcome_source"] = source

    def latest(self) -> dict[str, Any] | None:
        if not self.rows:
            return None
        return list(self.rows.values())[-1]


# parity-with: threetears.agent.skills.tools.SkillRegistryClient
class _FakeRegistry:
    """In-memory implementation of :class:`SkillRegistryClient`."""

    def __init__(
        self,
        *,
        permitted_tools: set[str] | None = None,
        skill_eligible: list[SkillEligibleTool] | None = None,
        introspect_payloads: dict[str, SkillToolIntrospect] | None = None,
    ) -> None:
        self._permitted = permitted_tools or set()
        self._skill_eligible = list(skill_eligible or [])
        self._introspect = dict(introspect_payloads or {})
        self.acl_calls: list[tuple[UUID, UUID, str]] = []

    async def acl_permits(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        tool_name: str,
    ) -> bool:
        self.acl_calls.append((user_id, agent_id, tool_name))
        return tool_name in self._permitted

    async def list_skill_eligible_tools(
        self,
        *,
        actor_user_id: UUID,
        actor_agent_id: UUID,
    ) -> list[SkillEligibleTool]:
        return list(self._skill_eligible)

    async def get_tool_introspect(
        self,
        *,
        actor_user_id: UUID,
        actor_agent_id: UUID,
        mcp_name: str,
    ) -> SkillToolIntrospect | None:
        return self._introspect.get(mcp_name)


# --- skill_create ---


class TestSkillCreate:
    """``skill_create`` validates payload, ACL, cap, name uniqueness."""

    @pytest.fixture
    def agent_id(self) -> UUID:
        return _new_uuid()

    @pytest.fixture
    def user_id(self) -> UUID:
        return _new_uuid()

    async def test_happy_path_body_only(
        self,
        agent_id: UUID,
        user_id: UUID,
    ) -> None:
        coll = _FakeSkillsCollection()
        reg = _FakeRegistry()
        [tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke(
            {
                "name": "deploy",
                "summary": "Ship the service",
                "body": "Run helm install",
            }
        )
        assert isinstance(out, str)
        assert out.startswith("[skill:")
        assert "deploy" in out
        assert "Ship the service" in out
        # one row persisted
        assert len(coll.rows) == 1

    async def test_at_least_one_payload_enforced(
        self,
        agent_id: UUID,
        user_id: UUID,
    ) -> None:
        coll = _FakeSkillsCollection()
        reg = _FakeRegistry()
        [tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({"name": "empty", "summary": "no payload"})
        assert "[TOOL ERROR]" in out
        assert "at least one" in out
        assert len(coll.rows) == 0

    async def test_acl_rejects_unauthorized_tool(
        self,
        agent_id: UUID,
        user_id: UUID,
    ) -> None:
        coll = _FakeSkillsCollection()
        reg = _FakeRegistry()  # nothing permitted
        [tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke(
            {
                "name": "shell-skill",
                "summary": "uses shell",
                "tool_additions": ["mcp.shell"],
            }
        )
        assert "[TOOL ERROR]" in out
        assert "not authorized" in out
        assert "mcp.shell" in out
        assert len(coll.rows) == 0

    async def test_acl_passes_when_permitted(
        self,
        agent_id: UUID,
        user_id: UUID,
    ) -> None:
        coll = _FakeSkillsCollection()
        reg = _FakeRegistry(permitted_tools={"mcp.shell"})
        [tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke(
            {
                "name": "shell-skill",
                "summary": "uses shell",
                "tool_additions": ["mcp.shell"],
            }
        )
        assert out.startswith("[skill:")
        assert len(coll.rows) == 1

    async def test_name_uniqueness(
        self,
        agent_id: UUID,
        user_id: UUID,
    ) -> None:
        coll = _FakeSkillsCollection()
        reg = _FakeRegistry()
        [tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        await tool.ainvoke(
            {"name": "deploy", "summary": "one", "body": "first"},
        )
        out = await tool.ainvoke(
            {"name": "deploy", "summary": "two", "body": "second"},
        )
        assert "[TOOL ERROR]" in out
        assert "already exists" in out
        assert len(coll.rows) == 1

    async def test_cap_enforcement(
        self,
        agent_id: UUID,
        user_id: UUID,
    ) -> None:
        coll = _FakeSkillsCollection()
        reg = _FakeRegistry()
        [tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
            max_prose_skills_per_user=2,
        )
        await tool.ainvoke({"name": "a", "summary": "x", "body": "1"})
        await tool.ainvoke({"name": "b", "summary": "x", "body": "2"})
        out = await tool.ainvoke({"name": "c", "summary": "x", "body": "3"})
        assert "[TOOL ERROR]" in out
        assert "max 2 prose skills" in out
        assert len(coll.rows) == 2


# --- skill_list ---


class TestSkillList:
    """``skill_list`` UNIONs prose-skill rows + tool-skill registry entries."""

    async def test_empty_returns_message(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        reg = _FakeRegistry()
        [tool] = load_skill_list_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({})
        assert "No skills available" in out

    async def test_union_prose_and_tool(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        # seed one prose skill
        skill_id = _new_uuid()
        coll.rows[(agent_id, skill_id)] = {
            "skill_id": skill_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "name": "manual-deploy",
            "summary": "manual deploy procedure",
            "body": "steps",
            "prompt_mode": "additive",
            "tool_additions": [],
            "tool_restrictions": [],
            "trigger_keywords": "",
            "tags": [],
            "source": "manual",
            "enabled": True,
            "use_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "last_used_at": None,
            "last_failure_at": None,
            "date_created": datetime.now(UTC),
            "date_updated": datetime.now(UTC),
        }
        reg = _FakeRegistry(
            skill_eligible=[
                SkillEligibleTool(mcp_name="loki.query", summary="Query Loki logs"),
            ],
        )
        [tool] = load_skill_list_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({})
        assert "manual-deploy" in out
        assert "kind=prose" in out
        assert "loki.query" in out
        assert "kind=tool" in out

    async def test_kind_filter_prose_only(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        reg = _FakeRegistry(
            skill_eligible=[SkillEligibleTool(mcp_name="loki.query", summary="x")],
        )
        [tool] = load_skill_list_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({"kind_filter": "prose"})
        # tool-skill suppressed
        assert "loki.query" not in out

    async def test_cross_user_isolation(self) -> None:
        """A skill belonging to a different user is hidden."""
        agent_id = _new_uuid()
        user_a = _new_uuid()
        user_b = _new_uuid()
        coll = _FakeSkillsCollection()
        skill_id = _new_uuid()
        coll.rows[(agent_id, skill_id)] = {
            "skill_id": skill_id,
            "agent_id": agent_id,
            "user_id": user_b,  # different user
            "name": "secret",
            "summary": "private",
            "body": "...",
            "prompt_mode": "additive",
            "tool_additions": [],
            "tool_restrictions": [],
            "trigger_keywords": "",
            "tags": [],
            "source": "manual",
            "enabled": True,
            "use_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "last_used_at": None,
            "last_failure_at": None,
            "date_created": datetime.now(UTC),
            "date_updated": datetime.now(UTC),
        }
        reg = _FakeRegistry()
        [tool] = load_skill_list_tool(
            agent_id=agent_id,
            user_id=user_a,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({})
        assert "secret" not in out

    async def test_tool_skills_surface_when_prose_dominates(self) -> None:
        """SK-16 discoverability: prose can't saturate limit and silently hide tools.

        With ``limit=4`` and 10 eligible prose rows + 2 tool-skills,
        the merge must still surface at least one tool-skill. The
        prior implementation queried prose with the full limit then
        truncated, hiding tools entirely.
        """
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        # Seed 10 prose skills
        for i in range(10):
            sid = _new_uuid()
            coll.rows[(agent_id, sid)] = {
                "skill_id": sid,
                "agent_id": agent_id,
                "user_id": user_id,
                "name": f"prose-{i}",
                "summary": f"prose number {i}",
                "body": "x",
                "prompt_mode": "additive",
                "tool_additions": [],
                "tool_restrictions": [],
                "trigger_keywords": "",
                "tags": [],
                "source": "manual",
                "enabled": True,
                "use_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "last_used_at": None,
                "last_failure_at": None,
                "date_created": datetime.now(UTC),
                "date_updated": datetime.now(UTC),
            }
        reg = _FakeRegistry(
            skill_eligible=[
                SkillEligibleTool(mcp_name="loki.query", summary="Query Loki logs"),
                SkillEligibleTool(mcp_name="mcp.shell", summary="Shell access"),
            ],
        )
        [tool] = load_skill_list_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({"limit": 4})
        # Result honors limit ...
        # one header line + N entry lines; count entry lines
        entry_lines = [line for line in out.splitlines() if line.startswith("- [skill:")]
        assert len(entry_lines) == 4
        # ... AND at least one tool-skill is present
        assert "kind=tool" in out


# --- skill_get / skill_update / skill_delete ---


async def _seed_skill(
    coll: _FakeSkillsCollection,
    *,
    agent_id: UUID,
    user_id: UUID,
    name: str = "example",
    body: str | None = "do the thing",
    prompt_mode: str = "additive",
    enabled: bool = True,
) -> UUID:
    skill_id = _new_uuid()
    now = datetime.now(UTC)
    coll.rows[(agent_id, skill_id)] = {
        "skill_id": skill_id,
        "agent_id": agent_id,
        "user_id": user_id,
        "name": name,
        "summary": "one-liner",
        "body": body,
        "prompt_mode": prompt_mode,
        "tool_additions": [],
        "tool_restrictions": [],
        "trigger_keywords": "",
        "tags": [],
        "source": "manual",
        "enabled": enabled,
        "use_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "last_used_at": None,
        "last_failure_at": None,
        "date_created": now,
        "date_updated": now,
    }
    return skill_id


class TestSkillGet:
    async def test_happy_path(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        [tool] = load_skill_get_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"skill_id": f"[skill:{skill_id}]"})
        assert f"[skill:{skill_id}]" in out
        assert "kind: prose" in out

    async def test_cross_user_returns_not_found(self) -> None:
        agent_id = _new_uuid()
        owner = _new_uuid()
        other = _new_uuid()
        coll = _FakeSkillsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=owner)
        [tool] = load_skill_get_tool(
            agent_id=agent_id,
            user_id=other,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert "[TOOL ERROR]" in out
        assert "not found" in out

    async def test_invalid_id_returns_error(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        [tool] = load_skill_get_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"skill_id": "not-a-uuid"})
        assert "[TOOL ERROR]" in out
        assert "invalid skill_id" in out


class TestSkillUpdate:
    async def test_partial_update(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        reg = _FakeRegistry()
        [tool] = load_skill_update_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke(
            {"skill_id": str(skill_id), "summary": "updated summary"},
        )
        assert "[TOOL ERROR]" not in out
        assert coll.rows[(agent_id, skill_id)]["summary"] == "updated summary"
        # other fields untouched
        assert coll.rows[(agent_id, skill_id)]["name"] == "example"

    async def test_cross_user_not_found(self) -> None:
        agent_id = _new_uuid()
        owner = _new_uuid()
        other = _new_uuid()
        coll = _FakeSkillsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=owner)
        reg = _FakeRegistry()
        [tool] = load_skill_update_tool(
            agent_id=agent_id,
            user_id=other,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id), "summary": "x"})
        assert "[TOOL ERROR]" in out
        assert "not found" in out

    async def test_acl_recheck_on_tool_additions_change(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        reg = _FakeRegistry()  # nothing permitted
        [tool] = load_skill_update_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke(
            {"skill_id": str(skill_id), "tool_additions": ["mcp.shell"]},
        )
        assert "[TOOL ERROR]" in out
        assert "not authorized" in out

    async def test_removing_body_then_no_tools_rejected(self) -> None:
        """Clearing body while tool lists are empty triggers the CHECK."""
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        reg = _FakeRegistry()
        [tool] = load_skill_update_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id), "body": ""})
        assert "[TOOL ERROR]" in out
        assert "at least one" in out


class TestSkillDelete:
    async def test_happy_path(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        [tool] = load_skill_delete_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert out.startswith("Deleted")
        assert (agent_id, skill_id) not in coll.rows

    async def test_cross_user_not_found(self) -> None:
        agent_id = _new_uuid()
        owner = _new_uuid()
        other = _new_uuid()
        coll = _FakeSkillsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=owner)
        [tool] = load_skill_delete_tool(
            agent_id=agent_id,
            user_id=other,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert "[TOOL ERROR]" in out
        # row remains
        assert (agent_id, skill_id) in coll.rows


# --- skill_invoke ---


class _ActiveState:
    """Trivial active-skill state holder used by ``skill_invoke`` tests."""

    def __init__(self) -> None:
        self.active: UUID | None = None

    def probe(self) -> UUID | None:
        return self.active

    def setter(self, skill_id: UUID) -> None:
        self.active = skill_id


class TestSkillInvoke:
    async def test_happy_path_additive(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        conv_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        state = _ActiveState()
        [tool] = load_skill_invoke_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            invocations_collection=inv,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=state.probe,
            active_skill_setter=state.setter,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert "[ACTIVE SKILL: example]" in out
        assert "prompt_mode: additive" in out
        assert state.active == skill_id
        latest = inv.latest()
        assert latest is not None
        assert latest["skill_id"] == skill_id
        assert latest["conversation_id"] == conv_id
        assert latest["invocation_source"] == "invoke"

    async def test_first_invoke_wins(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        conv_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        already = _new_uuid()
        state = _ActiveState()
        state.active = already
        [tool] = load_skill_invoke_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            invocations_collection=inv,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=state.probe,
            active_skill_setter=state.setter,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert "[TOOL ERROR]" in out
        assert "already active" in out
        assert state.active == already

    async def test_replace_mode_rejected_mid_turn(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        conv_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        skill_id = await _seed_skill(
            coll,
            agent_id=agent_id,
            user_id=user_id,
            prompt_mode="replace",
        )
        state = _ActiveState()
        [tool] = load_skill_invoke_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            invocations_collection=inv,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=state.probe,
            active_skill_setter=state.setter,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert "[TOOL ERROR]" in out
        assert "replace" in out
        assert "wake" in out
        assert state.active is None

    async def test_disabled_rejected(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        conv_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        skill_id = await _seed_skill(
            coll,
            agent_id=agent_id,
            user_id=user_id,
            enabled=False,
        )
        state = _ActiveState()
        [tool] = load_skill_invoke_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            invocations_collection=inv,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=state.probe,
            active_skill_setter=state.setter,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert "[TOOL ERROR]" in out
        assert "disabled" in out
        assert state.active is None

    async def test_no_conversation_id_rejected(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        state = _ActiveState()
        [tool] = load_skill_invoke_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            invocations_collection=inv,
            conversation_id_resolver=lambda: None,
            active_skill_probe=state.probe,
            active_skill_setter=state.setter,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert "[TOOL ERROR]" in out
        assert "conversation_id_resolver" in out
        assert state.active is None

    async def test_setter_failure_leaves_no_invocation_row(self) -> None:
        """Setter-first ordering: if the setter raises, NO invocation row persists.

        The previous order (persist-then-setter) left a row in the DB
        when the setter raised, with the in-process state unset --
        the next skill_invoke this turn would pass the probe and
        record a second row (first-invoke-wins violated at the row
        count). Reversing to setter-first means a setter raise
        short-circuits before the persist.
        """
        agent_id = _new_uuid()
        user_id = _new_uuid()
        conv_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)

        def boom(_: UUID) -> None:
            raise RuntimeError("setter exploded")

        [tool] = load_skill_invoke_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            invocations_collection=inv,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=lambda: None,
            active_skill_setter=boom,
        )
        out = await tool.ainvoke({"skill_id": str(skill_id)})
        assert "[TOOL ERROR]" in out
        assert "state setter failed" in out
        # No invocation row persisted -- this is the load-bearing
        # assertion for the setter-first ordering fix.
        assert inv.latest() is None
        assert len(inv.rows) == 0


class TestSkillReportOutcome:
    async def test_happy_path_success(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        conv_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        state = _ActiveState()
        [invoke_tool] = load_skill_invoke_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            invocations_collection=inv,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=state.probe,
            active_skill_setter=state.setter,
        )
        await invoke_tool.ainvoke({"skill_id": str(skill_id)})

        [tool] = load_skill_report_outcome_tool(
            agent_id=agent_id,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=state.probe,
            invocations_collection=inv,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"outcome": "success"})
        assert f"[skill:{skill_id}]" in out
        assert "success" in out
        latest = inv.latest()
        assert latest is not None
        assert latest["outcome"] == "success"
        assert latest["outcome_source"] == "agent_tool"
        assert coll.rows[(agent_id, skill_id)]["success_count"] == 1
        assert coll.rows[(agent_id, skill_id)]["failure_count"] == 0

    async def test_happy_path_failure_with_notes(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        conv_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        state = _ActiveState()
        [invoke_tool] = load_skill_invoke_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            invocations_collection=inv,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=state.probe,
            active_skill_setter=state.setter,
        )
        await invoke_tool.ainvoke({"skill_id": str(skill_id)})

        [tool] = load_skill_report_outcome_tool(
            agent_id=agent_id,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=state.probe,
            invocations_collection=inv,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"outcome": "failure", "notes": "target host unreachable"})
        assert "failure" in out
        assert coll.rows[(agent_id, skill_id)]["failure_count"] == 1
        assert coll.rows[(agent_id, skill_id)]["success_count"] == 0
        # notes are observability-only in v1 -- never written to the row
        assert inv.latest()["notes"] is None

    async def test_no_marker_leak_in_output(self) -> None:
        """The whole point of a tool over a text marker: no [SUCCESS]/[FAILED] leaks."""
        agent_id = _new_uuid()
        user_id = _new_uuid()
        conv_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        state = _ActiveState()
        [invoke_tool] = load_skill_invoke_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            invocations_collection=inv,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=state.probe,
            active_skill_setter=state.setter,
        )
        await invoke_tool.ainvoke({"skill_id": str(skill_id)})
        [tool] = load_skill_report_outcome_tool(
            agent_id=agent_id,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=state.probe,
            invocations_collection=inv,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"outcome": "success"})
        assert "[SUCCESS]" not in out
        assert "[FAILED]" not in out

    async def test_no_active_skill_rejected(self) -> None:
        agent_id = _new_uuid()
        conv_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        [tool] = load_skill_report_outcome_tool(
            agent_id=agent_id,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=lambda: None,
            invocations_collection=inv,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"outcome": "success"})
        assert "[TOOL ERROR]" in out
        assert "no skill is active" in out

    async def test_no_conversation_id_rejected(self) -> None:
        agent_id = _new_uuid()
        active_skill_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        [tool] = load_skill_report_outcome_tool(
            agent_id=agent_id,
            conversation_id_resolver=lambda: None,
            active_skill_probe=lambda: active_skill_id,
            invocations_collection=inv,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"outcome": "success"})
        assert "[TOOL ERROR]" in out
        assert "conversation_id_resolver" in out

    async def test_no_matching_invocation_rejected(self) -> None:
        """The active-skill probe says a skill is active, but no invocation row
        for it exists in this conversation (e.g. a wake-attach path that never
        recorded one) -- the tool must not attribute the outcome to the wrong row.
        """
        agent_id = _new_uuid()
        conv_id = _new_uuid()
        active_skill_id = _new_uuid()
        coll = _FakeSkillsCollection()
        inv = _FakeInvocationsCollection()
        [tool] = load_skill_report_outcome_tool(
            agent_id=agent_id,
            conversation_id_resolver=lambda: conv_id,
            active_skill_probe=lambda: active_skill_id,
            invocations_collection=inv,
            skills_collection=coll,
        )
        out = await tool.ainvoke({"outcome": "success"})
        assert "[TOOL ERROR]" in out
        assert "no active invocation found" in out


# --- skill_introspect ---


class TestSkillIntrospect:
    async def test_prose_skill_by_uuid(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        skill_id = await _seed_skill(coll, agent_id=agent_id, user_id=user_id)
        reg = _FakeRegistry()
        [tool] = load_skill_introspect_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({"name_or_id": str(skill_id)})
        assert f"[skill:{skill_id}]" in out
        assert "kind: prose" in out
        # minimal-token shape: NO use_count etc
        assert "use_count" not in out

    async def test_prose_skill_by_name(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        await _seed_skill(coll, agent_id=agent_id, user_id=user_id, name="manual-deploy")
        reg = _FakeRegistry()
        [tool] = load_skill_introspect_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({"name_or_id": "manual-deploy"})
        assert "kind: prose" in out

    async def test_tool_skill_via_registry(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        reg = _FakeRegistry(
            introspect_payloads={
                "loki.query": SkillToolIntrospect(
                    mcp_name="loki.query",
                    summary="Query Loki logs by container + time range",
                    args={
                        "container": "str  # container name",
                        "query": "str  # LogQL",
                    },
                    example={"container": "api", "query": 'level="ERROR"'},
                )
            }
        )
        [tool] = load_skill_introspect_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({"name_or_id": "loki.query"})
        assert "kind: tool" in out
        assert "Query Loki logs" in out
        assert "container:" in out
        assert "example:" in out

    async def test_prose_wins_on_name_collision(self) -> None:
        """Per Implementation note 7: prose-skill takes precedence."""
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        await _seed_skill(coll, agent_id=agent_id, user_id=user_id, name="loki.query")
        reg = _FakeRegistry(
            introspect_payloads={
                "loki.query": SkillToolIntrospect(
                    mcp_name="loki.query",
                    summary="tool variant",
                    args={},
                    example={},
                )
            }
        )
        [tool] = load_skill_introspect_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({"name_or_id": "loki.query"})
        assert "kind: prose" in out
        assert "tool variant" not in out

    async def test_not_found(self) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        coll = _FakeSkillsCollection()
        reg = _FakeRegistry()
        [tool] = load_skill_introspect_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=coll,
            registry=reg,
        )
        out = await tool.ainvoke({"name_or_id": "ghost"})
        assert "[TOOL ERROR]" in out
        assert "no skill or tool" in out
