"""Conversation-scoped context manager backed by three-tier collection."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from threetears.agent.memory.ledger import MemoryLedger
from threetears.agent.tools.collections import ContextItemCollection


class ToolContextManager:
    """Manages conversation context: variables, tool results, media slots, and workflows.

    Backed by a :class:`ContextItemCollection` for three-tier persistence
    (L1 SQLite → L2 NATS KV → L3 PostgreSQL).  Workflow state is transient
    (in-memory, single-turn only).

    Call :meth:`load_context` after construction to populate from storage.
    """

    def __init__(
        self,
        collection: ContextItemCollection,
        conversation_id: str,
        user_id: str,
        *,
        var_limit: int = 50,
        var_max_chars: int = 50_000,
        result_limit: int | None = None,
        ledger: MemoryLedger | None = None,
        l3_pool: Any = None,
    ) -> None:
        self._collection = collection
        self.conversation_id = conversation_id
        self.user_id = user_id
        self._var_limit = var_limit
        self._var_max_chars = var_max_chars
        self._result_limit = result_limit
        self._ledger = ledger or MemoryLedger()
        self._l3_pool = l3_pool

        # Local projection of collection data for this conversation.
        # Populated by load_context(), updated by write methods.
        self._items: list[dict[str, Any]] = []

        # Workflow state (transient, single-turn, in-memory only)
        self._workflow: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def load_context(self) -> None:
        """Load all context items and ledger for this conversation from storage."""
        entities = await self._collection.find_by_conversation(self.conversation_id)
        self._items = []
        for entity in entities:
            self._items.append(entity.to_dict())
        if self._l3_pool is not None:
            conv_uuid = (
                uuid.UUID(self.conversation_id)
                if isinstance(self.conversation_id, str)
                else self.conversation_id
            )
            await self._ledger.load(self._l3_pool, conv_uuid)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    async def set_variable(self, key: str, value: str, value_type: str = "string") -> str:
        """Set or update a variable.  Returns the context_id."""
        var_count = sum(1 for i in self._items if i["context_type"] == "variable")
        existing = next(
            (i for i in self._items if i["context_type"] == "variable" and i["key"] == key),
            None,
        )
        if existing is None and var_count >= self._var_limit:
            raise ValueError(
                f"Variable limit reached ({self._var_limit}). Delete unused variables before adding new ones."
            )
        if len(value) > self._var_max_chars:
            value = value[: self._var_max_chars]

        now = datetime.now(UTC)
        data = {
            "context_id": uuid.uuid4(),
            "conversation_id": uuid.UUID(self.conversation_id)
            if isinstance(self.conversation_id, str)
            else self.conversation_id,
            "context_type": "variable",
            "key": key,
            "short_desc": value[:200],
            "long_desc": "",
            "content": value,
            "metadata": {"value_type": value_type},
            "date_accessed": now,
            "date_created": now,
            "date_updated": now,
        }

        returned_id = await self._collection.upsert_variable(data)
        context_id_str = str(returned_id)

        # Update local projection
        self._items = [i for i in self._items if not (i["context_type"] == "variable" and i["key"] == key)]
        data["context_id"] = returned_id
        self._items.append(data)

        return context_id_str

    async def get_variable(self, key: str) -> dict[str, Any] | None:
        """Get a variable by key, or ``None`` if not found."""
        for item in self._items:
            if item["context_type"] == "variable" and item["key"] == key:
                vtype = (item.get("metadata") or {}).get("value_type", "string")
                return {"value": item["content"], "value_type": vtype}
        return None

    async def get_all_variables(self) -> list[dict[str, Any]]:
        """Return all variables as a list of dicts."""
        return [i for i in self._items if i["context_type"] == "variable"]

    async def delete_variable(self, key: str) -> bool:
        """Delete a variable by key.  Returns ``True`` if it existed."""
        target = next(
            (i for i in self._items if i["context_type"] == "variable" and i["key"] == key),
            None,
        )
        if target is None:
            return False

        self._items = [i for i in self._items if i is not target]
        await self._collection.delete(target["context_id"])
        return True

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    async def save_context_item(
        self,
        context_type: str,
        key: str,
        short_desc: str,
        content: str,
        long_desc: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a context item with full 3-tier descriptions.

        General-purpose save that covers variables, tool results, and
        any other context type. Returns the context_id.

        :param context_type: type of context item (variable, tool_result, scan_result, etc.)
        :ptype context_type: str
        :param key: unique key within the conversation
        :ptype key: str
        :param short_desc: token-efficient summary (≤200 chars)
        :ptype short_desc: str
        :param content: full content for on-demand retrieval
        :ptype content: str
        :param long_desc: expanded description (≤1000 chars)
        :ptype long_desc: str
        :param metadata: optional metadata dict
        :ptype metadata: dict[str, Any] | None
        :return: context_id of the saved item
        :rtype: str
        """
        return await self.save_tool_result(
            tool_name=key,
            result=content,
            short_desc=short_desc,
            long_desc=long_desc,
            context_type=context_type,
            metadata=metadata,
        )

    async def save_tool_result(
        self,
        tool_name: str,
        result: str,
        *,
        short_desc: str | None = None,
        long_desc: str = "",
        context_type: str = "tool_result",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a tool result.  Returns the generated context_id.

        If ``result_limit`` is set and the count of tool_result items
        exceeds it, the least-recently-accessed items are evicted.
        """
        now = datetime.now(UTC)
        context_id = uuid.uuid4()
        data = {
            "context_id": context_id,
            "conversation_id": uuid.UUID(self.conversation_id)
            if isinstance(self.conversation_id, str)
            else self.conversation_id,
            "context_type": context_type,
            "key": tool_name,
            "short_desc": short_desc or result[:200],
            "long_desc": long_desc[:1000] if long_desc else "",
            "content": result,
            "metadata": metadata or {},
            "date_accessed": now,
            "date_created": now,
            "date_updated": now,
        }

        await self._collection.save_entity(
            self._collection.entity_class(data, is_new=True, collection=self._collection)
        )
        self._items.append(data)

        # LRU eviction
        if self._result_limit is not None:
            evicted = await self._collection.evict_lru(self.conversation_id, self._result_limit)
            if evicted:
                # Refresh local projection to drop evicted items
                evicted_ids = {
                    str(i["context_id"])
                    for i in self._items
                    if i["context_type"] == "tool_result"
                    and not self._collection._exists_in_cache_sync(i["context_id"])
                }
                if evicted_ids:
                    self._items = [i for i in self._items if str(i["context_id"]) not in evicted_ids]

        return str(context_id)

    async def get_context_item(self, context_id: str) -> dict[str, Any] | None:
        """Retrieve a context item by id.

        Accessing an item updates its ``date_accessed`` for LRU tracking.
        """
        cid = str(context_id)
        if cid.startswith("ctx:"):
            cid = cid[4:]
        for item in self._items:
            if str(item["context_id"]) == cid:
                await self._collection.touch(cid)
                item["date_accessed"] = datetime.now(UTC)
                return item
        return None

    def build_context_detail(self, context_id: str) -> str | None:
        """Return the long_desc for a specific context item.

        Used by two-tier enrichment to get more detail without
        loading the full content.

        :param context_id: UUID string of the context item
        :ptype context_id: str
        :return: long_desc or None if not found
        :rtype: str | None
        """
        cid = str(context_id)
        for item in self._items:
            if str(item["context_id"]) == cid:
                return item.get("long_desc", "")
        return None

    def get_by_context_id(self, context_id: str) -> dict[str, Any] | None:
        """Retrieve a context item by its context_id from the local projection.

        :param context_id: UUID string of the context item
        :ptype context_id: str
        :return: context item data or None
        :rtype: dict[str, Any] | None
        """
        cid = str(context_id)
        for item in self._items:
            if str(item["context_id"]) == cid:
                return item
        return None

    # ------------------------------------------------------------------
    # Media slots
    # ------------------------------------------------------------------

    async def register_media(self, slot_name: str, **kwargs: Any) -> str:
        """Register a media slot.  Returns the context_id."""
        now = datetime.now(UTC)
        context_id = uuid.uuid4()
        data = {
            "context_id": context_id,
            "conversation_id": uuid.UUID(self.conversation_id)
            if isinstance(self.conversation_id, str)
            else self.conversation_id,
            "context_type": "media_slot",
            "key": slot_name,
            "short_desc": kwargs.get("description", slot_name),
            "long_desc": "",
            "content": "",
            "metadata": kwargs,
            "date_accessed": now,
            "date_created": now,
            "date_updated": now,
        }

        await self._collection.save_entity(
            self._collection.entity_class(data, is_new=True, collection=self._collection)
        )
        self._items.append(data)
        return str(context_id)

    def get_slots(self) -> dict[str, dict[str, Any]]:
        """Return all registered media slots."""
        return {i["key"]: i for i in self._items if i["context_type"] == "media_slot"}

    def build_media_context(self) -> str | None:
        """Format media slots into a prompt string, or ``None`` if empty."""
        slots = [i for i in self._items if i["context_type"] == "media_slot"]
        if not slots:
            return None
        lines = ["[Active Media Slots]"]
        for slot in slots:
            lines.append(f"- {slot['key']}: {slot.get('metadata', {})}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def declare_workflow(self, plan: str, steps: list[str]) -> dict[str, Any]:
        """Declare a new workflow with a plan and steps."""
        self._workflow = {
            "plan": plan,
            "steps": steps,
            "current_step": 0,
            "status": "active",
            "declared_at": datetime.now(UTC).isoformat(),
        }
        return self._workflow

    def advance_workflow_step(self) -> dict[str, Any] | None:
        """Advance to the next workflow step.  Returns updated state or ``None``."""
        if self._workflow is None or self._workflow["status"] != "active":
            return None
        self._workflow["current_step"] += 1
        if self._workflow["current_step"] >= len(self._workflow["steps"]):
            self._workflow["status"] = "completed"
        return self._workflow

    def complete_workflow(self) -> dict[str, Any] | None:
        """Mark the workflow as completed.  Returns final state or ``None``."""
        if self._workflow is None:
            return None
        self._workflow["status"] = "completed"
        return self._workflow

    @property
    def has_active_workflow(self) -> bool:
        """Whether there is an active workflow."""
        return self._workflow is not None and self._workflow["status"] == "active"

    @property
    def workflow_state(self) -> dict[str, Any] | None:
        """Current workflow state, or ``None``."""
        return self._workflow

    def build_workflow_prompt(self) -> str:
        """Format the active workflow as a system prompt section.

        Returns empty string if no workflow is active.

        :return: formatted workflow prompt section
        :rtype: str
        """
        if not self.has_active_workflow:
            return ""
        lines = [f"## Active Workflow: {self._workflow['plan']}"]
        for i, step in enumerate(self._workflow["steps"]):
            if isinstance(step, dict):
                desc = step.get("description", str(step))
                status = step.get("status", "pending")
            else:
                desc = str(step)
                status = "completed" if i < self._workflow["current_step"] else "pending"
            if status == "completed":
                check = "[x]"
            elif status == "skipped":
                check = "[-]"
            else:
                check = "[ ]"
            lines.append(f"{i + 1}. {check} {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Memory ledger (tracks surfaced items to prevent re-retrieval)
    # ------------------------------------------------------------------

    @property
    def ledger(self) -> MemoryLedger:
        """Return the conversation's memory ledger.

        :return: memory ledger instance
        :rtype: MemoryLedger
        """
        return self._ledger

    async def add_ledger_ref(self, item_id: str, item_type: str, short_desc: str) -> None:
        """Track a surfaced item in the memory ledger.

        :param item_id: UUID string of the surfaced item
        :ptype item_id: str
        :param item_type: type of the item (memory, media, chunk, finding, scan)
        :ptype item_type: str
        :param short_desc: short description (≤150 chars)
        :ptype short_desc: str
        """
        conv_uuid = (
            uuid.UUID(self.conversation_id)
            if isinstance(self.conversation_id, str)
            else self.conversation_id
        )
        if self._l3_pool is not None:
            await self._ledger.add_ref(self._l3_pool, conv_uuid, item_id, item_type, short_desc)

    def is_known(self, item_id: str) -> bool:
        """Check if an item is already tracked in the ledger.

        :param item_id: UUID string to check
        :ptype item_id: str
        :return: True if already in ledger
        :rtype: bool
        """
        return item_id in self._ledger.ledgered_ids

    def build_ledger_prompt(self) -> str:
        """Format the memory ledger for inclusion in the system prompt.

        :return: formatted ledger section or empty string
        :rtype: str
        """
        return self._ledger.build_context()

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_conversation_context(self) -> str | None:
        """Format variables and tool results into a prompt string.

        Returns ``None`` if there is no context to include.
        """
        variables = [i for i in self._items if i["context_type"] == "variable"]
        tool_results = [i for i in self._items if i["context_type"] == "tool_result"]

        if not variables and not tool_results:
            return None

        sections: list[str] = []

        if variables:
            lines = ["[Conversation Variables]"]
            for var in variables:
                vtype = (var.get("metadata") or {}).get("value_type", "string")
                lines.append(f"- {var['key']} ({vtype}): {var['content']}")
            sections.append("\n".join(lines))

        if tool_results:
            lines = ["[Tool Results]"]
            for item in tool_results:
                lines.append(f"- [{item['context_id']}] {item['key']}: {item['short_desc']}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    @property
    def has_context(self) -> bool:
        """Whether there is any context (variables, tool results, or media)."""
        return bool(self._items)
