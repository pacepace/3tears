"""in-memory fakes for the skills collections and the registry client.

shared by every skills unit test that drives a tool factory, so a test
module builds its fakes from here rather than importing a sibling test
module's private helpers.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from threetears.agent.skills.collections import (
    AgentSkillCollection,
    AgentSkillInvocationCollection,
    InvocationPage,
    OutcomeRecorded,
)
from threetears.agent.skills.types import OutcomeSource, SkillOutcome
from threetears.core.entities.base import BaseEntity
from threetears.core.keyed import BoundedCount
from threetears.agent.skills.entities import (
    AgentSkillEntity,
    AgentSkillInvocationEntity,
)
from threetears.agent.skills.tools import (
    SkillEligibleTool,
    SkillToolIntrospect,
)

__all__ = ["FakeInvocationsCollection", "FakeRegistry", "FakeSkillsCollection"]


class FakeSkillsCollection(AgentSkillCollection):
    """The production :class:`AgentSkillCollection` holding its rows in memory.

    Built without registry, config, pool or cache; overrides the slice the tool factories call:
    ``create`` / ``save_entity`` / ``get`` / ``delete`` / ``find_by_name_for_user`` /
    ``list_for_user`` / ``count_for_user`` / ``increment_outcome_counts``. Constructs entities
    with ``collection=None`` so the cache-write path in :meth:`BaseEntity.__init__` falls back
    to transient dict storage -- no L1 / L2 / L3 wiring needed for unit tests.
    """

    def __init__(self) -> None:
        """build the fake without the production wiring: no registry, config, pool or L1 table."""
        self.rows: dict[tuple[UUID, UUID], dict[str, Any]] = {}

    def create(self, data: dict[str, Any]) -> AgentSkillEntity:
        return AgentSkillEntity(dict(data), is_new=True, collection=None)

    async def save_entity(self, entity: BaseEntity, *, conn: Any = None) -> None:
        _ = conn
        data = entity.to_dict()
        self.rows[(data["agent_id"], data["skill_id"])] = dict(data)

    async def get(self, entity_id: Any) -> AgentSkillEntity | None:
        agent_id, skill_id = entity_id
        row = self.rows.get((agent_id, skill_id))
        if row is None:
            return None
        return AgentSkillEntity(dict(row), is_new=False, collection=None)

    async def delete(self, entity_id: Any, *, conn: Any = None) -> bool:
        agent_id, skill_id = entity_id
        self.rows.pop((agent_id, skill_id), None)
        return True

    async def invalidate_cache(self, entity_id: Any) -> None:
        """Nothing is cached here; the rows are the store."""
        del entity_id

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
        tag_filter: Sequence[str] | None = None,
        query: str | None = None,
        limit: int = 20,
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
        return results[:limit]

    async def count_for_user(
        self,
        agent_id: UUID,
        user_id: UUID,
        *,
        enabled_only: bool = True,
        tag_filter: Sequence[str] | None = None,
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
        outcome: SkillOutcome,
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


class FakeInvocationsCollection(AgentSkillInvocationCollection):
    """The production :class:`AgentSkillInvocationCollection` holding its rows in memory.

    Built without registry, config, pool or cache; overrides the slice skill_invoke,
    skill_report_outcome and skill_history call. record_outcome's one-transaction write is
    covered on real Postgres in tests/integration/test_tools_e2e.py.
    """

    def __init__(self, skills: FakeSkillsCollection | None = None) -> None:
        """build the fake without the production wiring, moving ``skills``' counts on a recorded outcome."""
        self.rows: dict[tuple[UUID, UUID], dict[str, Any]] = {}
        #: the skills whose counts a recorded outcome moves, as the production statement does
        self.skills = skills

    def create(self, data: dict[str, Any]) -> AgentSkillInvocationEntity:
        return AgentSkillInvocationEntity(dict(data), is_new=True, collection=None)

    async def save_entity(self, entity: BaseEntity, *, conn: Any = None) -> None:
        _ = conn
        data = entity.to_dict()
        self.rows[(data["agent_id"], data["invocation_id"])] = dict(data)

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
        outcome: SkillOutcome,
        source: OutcomeSource,
    ) -> None:
        row = self.rows.get((agent_id, invocation_id))
        if row is None:
            return
        row["outcome"] = outcome
        row["outcome_source"] = source

    async def record_outcome(
        self,
        agent_id: UUID,
        invocation_id: UUID,
        *,
        skill_id: UUID,
        outcome: SkillOutcome,
        source: OutcomeSource,
        notes: str | None = None,
    ) -> OutcomeRecorded:
        """The production rule in memory: a use counts once; a flip moves one count to the other side."""
        row = self.rows.get((agent_id, invocation_id))
        if row is None or row["skill_id"] != skill_id:
            return OutcomeRecorded(found=False)
        was, was_notes = row.get("outcome"), row.get("outcome_notes")
        row["outcome"], row["outcome_source"] = outcome, source
        row["outcome_notes"] = notes if notes is not None else was_notes
        counted = was != outcome
        target = self.skills.rows.get((agent_id, skill_id)) if self.skills is not None else None
        if counted and target is not None:
            if was is not None:
                target[f"{was}_count"] = target.get(f"{was}_count", 0) - 1
            target[f"{outcome}_count"] = target.get(f"{outcome}_count", 0) + 1
            if outcome == "failure":
                target["last_failure_at"] = datetime.now(UTC)
        return OutcomeRecorded(found=True, previous=was, previous_notes=was_notes, counted=counted)

    def _uses(
        self, agent_id: UUID, skill_id: UUID, outcome_filter: SkillOutcome | None
    ) -> list[AgentSkillInvocationEntity]:
        matches = [
            AgentSkillInvocationEntity(dict(row), is_new=False, collection=None)
            for row in self.rows.values()
            if row["agent_id"] == agent_id
            and row["skill_id"] == skill_id
            and (outcome_filter is None or row.get("outcome") == outcome_filter)
        ]
        matches.sort(key=lambda e: (e.invoked_at, str(e.invocation_id)), reverse=True)
        return matches

    async def list_for_skill(
        self,
        agent_id: UUID,
        skill_id: UUID,
        *,
        limit: int = 20,
        cursor: str | None = None,
        outcome_filter: SkillOutcome | None = None,
    ) -> InvocationPage:
        # the cursor is the last row's key here: opaque to the tool, as the real one is
        matches = self._uses(agent_id, skill_id, outcome_filter)
        start = 0 if cursor is None else 1 + [str(e.invocation_id) for e in matches].index(cursor)
        rows = matches[start : start + limit]
        more = start + limit < len(matches)
        return InvocationPage(rows=rows, next_cursor=str(rows[-1].invocation_id) if more and rows else None)

    async def count_for_skill(
        self, agent_id: UUID, skill_id: UUID, *, outcome_filter: SkillOutcome | None = None
    ) -> int:
        return len(self._uses(agent_id, skill_id, outcome_filter))

    async def count_for_skill_at_most(self, agent_id: UUID, skill_id: UUID, *, at_most: int) -> BoundedCount:
        held = len(self._uses(agent_id, skill_id, None))
        return BoundedCount(count=min(held, at_most), more=held > at_most)

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
