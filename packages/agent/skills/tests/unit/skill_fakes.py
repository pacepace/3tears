"""in-memory fakes for the skills collections and the registry client.

shared by every skills unit test that drives a tool factory, so a test
module builds its fakes from here rather than importing a sibling test
module's private helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from threetears.agent.skills.entities import (
    AgentSkillEntity,
    AgentSkillInvocationEntity,
)
from threetears.agent.skills.tools import (
    SkillEligibleTool,
    SkillToolIntrospect,
)

__all__ = ["FakeInvocationsCollection", "FakeRegistry", "FakeSkillsCollection"]


# parity-with: threetears.agent.skills.collections.AgentSkillCollection
# parity-exempt: AgentSkillCollection subset for the eight tool factory unit tests; the tools call only create/save_entity/get/delete/find_by_name_for_user/list_for_user/count_for_user/increment_outcome_counts and the three-tier-cache + l2/l3 SQL methods on the production class are not part of the tool API contract
class FakeSkillsCollection:
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
# parity-exempt: AgentSkillInvocationCollection subset for skill_invoke + skill_report_outcome unit coverage; the tools call only create/save_entity/record/list_for_conversation/set_outcome so the cache/persistence methods on the production class are out of scope
class FakeInvocationsCollection:
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
# parity-exempt: in-memory SkillRegistryClient implementation; the production protocol is the SkillRegistryClient defined in tools.py and the fake declares parity-with via the marker on the class, but the strict walker also requires this entry to satisfy the cross-file Protocol lookup
class FakeRegistry:
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
