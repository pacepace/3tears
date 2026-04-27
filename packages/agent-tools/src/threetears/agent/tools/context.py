"""Conversation-scoped context manager backed by three-tier collection."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid7

from threetears.agent.memory.collections import MemoryRefsCollection
from threetears.agent.memory.entities import MemoryRefEntity
from threetears.agent.tools.collections import ContextItemCollection

__all__ = [
    "ToolContextManager",
]

_MEMORY_REF_FIFO_CAPACITY = 50


class ToolContextManager:
    """Manages conversation context: variables, tool results, media slots, and workflows.

    Backed by a :class:`ContextItemCollection` for three-tier persistence
    (L1 SQLite -> L2 NATS KV -> L3 PostgreSQL) and -- for memory-ref
    surfacing tracking -- by a :class:`MemoryRefsCollection` on the
    composite-pk ``conversation_memory_refs`` table. workflow state is
    transient (in-memory, single-turn only).

    the surfaced-items FIFO projection lives in memory (capped at 50)
    and is backed by the collection: cold-start rehydrates via
    :meth:`MemoryRefsCollection.find_by_conversation`; writes go through
    :meth:`MemoryRefsCollection.save_entity` so L1 / L2 / L3 and cross-
    pod invalidation all stay coherent. retires the bespoke
    :class:`MemoryLedger` wrapper under namespace-task-01 phase 8.5l-2.

    Call :meth:`load_context` after construction to populate from storage.
    """

    def __init__(
        self,
        collection: ContextItemCollection,
        conversation_id: UUID,
        user_id: UUID,
        *,
        var_limit: int = 50,
        var_max_chars: int = 50_000,
        result_limit: int | None = None,
        memory_refs_collection: MemoryRefsCollection | None = None,
    ) -> None:
        """initialize context manager with required context collection.

        :param collection: three-tier ContextItemCollection for variables
            / tool results / media slots
        :ptype collection: ContextItemCollection
        :param conversation_id: conversation scope (first pk column of
            memory refs)
        :ptype conversation_id: UUID
        :param user_id: invoking user UUID for ownership checks
        :ptype user_id: UUID
        :param var_limit: maximum variables before set rejects new keys
        :ptype var_limit: int
        :param var_max_chars: truncation cap for variable values
        :ptype var_max_chars: int
        :param result_limit: optional LRU cap for tool_result items
        :ptype result_limit: int | None
        :param memory_refs_collection: optional three-tier memory-refs
            collection for cross-pod surfacing tracking. when absent,
            surfacing writes through :meth:`add_ledger_ref` become
            no-ops (tests + tool smoke fixtures without an agent pod
            configured do not need surfacing persistence)
        :ptype memory_refs_collection: MemoryRefsCollection | None
        :return: nothing
        :rtype: None
        """
        self._collection = collection
        self.conversation_id = conversation_id
        self.user_id = user_id
        self._var_limit = var_limit
        self._var_max_chars = var_max_chars
        self._result_limit = result_limit
        self._memory_refs = memory_refs_collection

        # Local projection of collection data for this conversation.
        # Populated by load_context(), updated by write methods.
        self.items: list[dict[str, Any]] = []

        # Ordered projection of surfaced memory-ref rows (FIFO), mirror
        # of the rows persisted via :class:`MemoryRefsCollection`. The
        # list preserves insertion order so the prompt renders items
        # chronologically and FIFO eviction drops the oldest first.
        # Each entry is a dict with keys
        # ``{item_id, item_type, short_desc, date_added}``.
        self._memory_refs_projection: list[dict[str, Any]] = []

        # Workflow state (transient, single-turn, in-memory only)
        self._workflow: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def load_context(self) -> None:
        """Load all context items and surfaced-refs projection for this conversation.

        pulls the persisted ``conversation_memory_refs`` rows through
        :meth:`MemoryRefsCollection.find_by_conversation` into the
        in-memory FIFO projection (capped at 50, oldest first). when
        no collection is wired (unit tests, smoke fixtures) the
        projection stays empty and surfacing operations become no-ops.
        """
        entities = await self._collection.find_by_conversation(self.conversation_id)
        self.items = []
        for entity in entities:
            self.items.append(entity.to_dict())
        self._memory_refs_projection = []
        if self._memory_refs is not None:
            refs = await self._memory_refs.find_by_conversation(
                self.conversation_id,
            )
            for ref in refs:
                self._memory_refs_projection.append(
                    {
                        "item_id": str(ref.item_id),
                        "item_type": ref.item_type,
                        "short_desc": ref.short_desc,
                        "date_added": ref.date_added.isoformat() if ref.date_added else "",
                    },
                )

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    async def set_variable(self, key: str, value: str, value_type: str = "string") -> str:
        """Set or update a variable.  Returns the context_id."""
        var_count = sum(1 for i in self.items if i["context_type"] == "variable")
        existing = next(
            (i for i in self.items if i["context_type"] == "variable" and i["key"] == key),
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
            "context_id": uuid7(),
            "conversation_id": self.conversation_id,
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

        returned_id = await self._collection.upsert_variable(self.conversation_id, data)
        context_id_str = str(returned_id)

        # Update local projection
        self.items = [i for i in self.items if not (i["context_type"] == "variable" and i["key"] == key)]
        data["context_id"] = returned_id
        self.items.append(data)

        return context_id_str

    async def get_variable(self, key: str) -> dict[str, Any] | None:
        """Get a variable by key, or ``None`` if not found."""
        for item in self.items:
            if item["context_type"] == "variable" and item["key"] == key:
                vtype = (item.get("metadata") or {}).get("value_type", "string")
                return {"value": item["content"], "value_type": vtype}
        return None

    async def get_all_variables(self) -> list[dict[str, Any]]:
        """Return all variables as a list of dicts."""
        return [i for i in self.items if i["context_type"] == "variable"]

    async def delete_variable(self, key: str) -> bool:
        """Delete a variable by key.  Returns ``True`` if it existed."""
        target = next(
            (i for i in self.items if i["context_type"] == "variable" and i["key"] == key),
            None,
        )
        if target is None:
            return False

        self.items = [i for i in self.items if i is not target]
        await self._collection.delete((self.conversation_id, target["context_id"]))
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
        context_id = uuid7()
        data = {
            "context_id": context_id,
            "conversation_id": self.conversation_id,
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
        self.items.append(data)

        # LRU eviction
        if self._result_limit is not None:
            evicted = await self._collection.evict_lru(self.conversation_id, self._result_limit)
            if evicted:
                # Refresh local projection to drop evicted items
                evicted_ids = {
                    str(i["context_id"])
                    for i in self.items
                    if i["context_type"] == "tool_result"
                    and not self._collection.exists_in_cache_sync(
                        (self.conversation_id, i["context_id"]),
                    )
                }
                if evicted_ids:
                    self.items = [i for i in self.items if str(i["context_id"]) not in evicted_ids]

        return str(context_id)

    async def get_context_item(self, context_id: str) -> dict[str, Any] | None:
        """Retrieve a context item by id.

        Accessing an item updates its ``date_accessed`` for LRU tracking.
        """
        cid = str(context_id)
        if cid.startswith("ctx:"):
            cid = cid[4:]
        for item in self.items:
            if str(item["context_id"]) == cid:
                await self._collection.touch(self.conversation_id, cid)
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
        for item in self.items:
            if str(item["context_id"]) == cid:
                val: str | None = item.get("long_desc", "")
                return val
        return None

    def get_by_context_id(self, context_id: str) -> dict[str, Any] | None:
        """Retrieve a context item by its context_id from the local projection.

        :param context_id: UUID string of the context item
        :ptype context_id: str
        :return: context item data or None
        :rtype: dict[str, Any] | None
        """
        cid = str(context_id)
        for item in self.items:
            if str(item["context_id"]) == cid:
                return item
        return None

    # ------------------------------------------------------------------
    # Arbitrary context items keyed by (context_type, key)
    # ------------------------------------------------------------------

    async def get_item_by_type_and_key(
        self,
        context_type: str,
        key: str,
    ) -> dict[str, Any] | None:
        """Fetch a single context item by ``(context_type, key)`` pair.

        General-purpose lookup for callers that store arbitrary items
        (workspace pin, per-agent bookmarks, etc.) outside the variable
        / tool_result taxonomies. Reads the local projection; returns
        ``None`` when no item matches.

        :param context_type: context taxonomy tag (e.g. ``"workspace_pin"``)
        :ptype context_type: str
        :param key: unique key within ``context_type`` scope
        :ptype key: str
        :return: item dict or ``None`` if absent
        :rtype: dict[str, Any] | None
        """
        result: dict[str, Any] | None = None
        for item in self.items:
            if item["context_type"] == context_type and item["key"] == key:
                result = item
                break
        return result

    async def save_item_by_type_and_key(
        self,
        *,
        context_type: str,
        key: str,
        content: str,
        short_desc: str = "",
        long_desc: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save (or replace) a context item by ``(context_type, key)`` pair.

        Mirrors :meth:`set_variable`'s delete-then-save pattern so any
        existing item with the same ``(context_type, key)`` is replaced
        atomically in the local projection, L1, L2, and L3 tiers.
        Returns the fresh ``context_id`` as a string.

        :param context_type: context taxonomy tag (e.g. ``"workspace_pin"``)
        :ptype context_type: str
        :param key: unique key within ``context_type`` scope
        :ptype key: str
        :param content: item content
        :ptype content: str
        :param short_desc: optional token-efficient summary
        :ptype short_desc: str
        :param long_desc: optional expanded description
        :ptype long_desc: str
        :param metadata: optional metadata dict
        :ptype metadata: dict[str, Any] | None
        :return: context_id of the saved item as string
        :rtype: str
        """
        # drop any prior entry so we enforce single-item semantics across
        # every storage tier (local projection, L1, L2, L3).
        existing = await self.get_item_by_type_and_key(context_type, key)
        if existing is not None:
            self.items = [i for i in self.items if i is not existing]
            await self._collection.delete(
                (self.conversation_id, existing["context_id"]),
            )

        now = datetime.now(UTC)
        context_id = uuid7()
        data: dict[str, Any] = {
            "context_id": context_id,
            "conversation_id": self.conversation_id,
            "context_type": context_type,
            "key": key,
            "short_desc": short_desc,
            "long_desc": long_desc,
            "content": content,
            "metadata": metadata or {},
            "date_accessed": now,
            "date_created": now,
            "date_updated": now,
        }
        await self._collection.save_entity(
            self._collection.entity_class(
                data,
                is_new=True,
                collection=self._collection,
            )
        )
        self.items.append(data)
        return str(context_id)

    async def delete_item_by_type_and_key(
        self,
        context_type: str,
        key: str,
    ) -> bool:
        """Delete a context item by ``(context_type, key)`` pair.

        Idempotent: returns ``False`` when no item matches rather than
        raising, so callers can invoke it unconditionally.

        :param context_type: context taxonomy tag (e.g. ``"workspace_pin"``)
        :ptype context_type: str
        :param key: unique key within ``context_type`` scope
        :ptype key: str
        :return: ``True`` if an item was deleted, ``False`` if absent
        :rtype: bool
        """
        existing = await self.get_item_by_type_and_key(context_type, key)
        result: bool
        if existing is None:
            result = False
        else:
            self.items = [i for i in self.items if i is not existing]
            await self._collection.delete(
                (self.conversation_id, existing["context_id"]),
            )
            result = True
        return result

    # ------------------------------------------------------------------
    # Media slots
    # ------------------------------------------------------------------

    async def register_media(self, slot_name: str, **kwargs: Any) -> str:
        """Register a media slot.  Returns the context_id."""
        now = datetime.now(UTC)
        context_id = uuid7()
        data = {
            "context_id": context_id,
            "conversation_id": self.conversation_id,
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
        self.items.append(data)
        return str(context_id)

    def get_slots(self) -> dict[str, dict[str, Any]]:
        """Return all registered media slots."""
        return {i["key"]: i for i in self.items if i["context_type"] == "media_slot"}

    def build_media_context(self) -> str | None:
        """Format media slots into a prompt string, or ``None`` if empty."""
        slots = [i for i in self.items if i["context_type"] == "media_slot"]
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
        if not self.has_active_workflow or self._workflow is None:
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
    # Memory refs (tracks surfaced items to prevent re-retrieval)
    # ------------------------------------------------------------------

    @property
    def memory_refs_count(self) -> int:
        """return the number of surfaced memory refs in the projection.

        replaces ``len(self.ledger)`` — graph-node gating keeps its
        shape (``memory_refs_count >= min_coverage``) without reaching
        through an intermediate object.

        :return: current surfaced-refs count
        :rtype: int
        """
        return len(self._memory_refs_projection)

    @property
    def surfaced_item_ids(self) -> set[str]:
        """return the set of surfaced item IDs (as strings).

        callers passing this to :meth:`MemoryRetriever.retrieve` as
        ``surfaced_ids`` get dedup on already-shown items without
        reaching into the internal projection.

        :return: surfaced item IDs
        :rtype: set[str]
        """
        return {ref["item_id"] for ref in self._memory_refs_projection}

    async def add_ledger_ref(
        self,
        item_id: str,
        item_type: str,
        short_desc: str,
    ) -> None:
        """Track a surfaced item by persisting through MemoryRefsCollection.

        FIFO eviction at capacity 50. truncates ``short_desc`` to 150
        chars at the write boundary so the L1 / L3 rows satisfy the
        migration-v002 VARCHAR(150) bound. when
        ``memory_refs_collection`` was not wired, the projection still
        updates (so in-conversation dedup works) but no cross-pod
        persistence happens.

        :param item_id: UUID string of the surfaced item
        :ptype item_id: str
        :param item_type: type of the item (memory / media / chunk /
            finding / scan)
        :ptype item_type: str
        :param short_desc: short description (truncated to 150 chars)
        :ptype short_desc: str
        :return: nothing
        :rtype: None
        """
        if any(ref["item_id"] == item_id for ref in self._memory_refs_projection):
            return

        desc = short_desc[:150] if len(short_desc) > 150 else short_desc
        now = datetime.now(UTC)

        if len(self._memory_refs_projection) >= _MEMORY_REF_FIFO_CAPACITY:
            oldest = self._memory_refs_projection.pop(0)
            if self._memory_refs is not None:
                await self._memory_refs.delete(
                    (self.conversation_id, UUID(oldest["item_id"])),
                )

        self._memory_refs_projection.append(
            {
                "item_id": item_id,
                "item_type": item_type,
                "short_desc": desc,
                "date_added": now.isoformat(),
            },
        )

        if self._memory_refs is not None:
            entity = self._memory_refs.create(
                {
                    "conversation_id": self.conversation_id,
                    "item_id": UUID(item_id),
                    "item_type": item_type,
                    "short_desc": desc,
                    "date_added": now,
                },
            )
            await self._memory_refs.save_entity(entity)

    def is_known(self, item_id: str) -> bool:
        """Check if an item is already tracked as surfaced.

        :param item_id: UUID string to check
        :ptype item_id: str
        :return: True if already surfaced this conversation
        :rtype: bool
        """
        return any(ref["item_id"] == item_id for ref in self._memory_refs_projection)

    def build_ledger_prompt(self) -> str:
        """Format surfaced memory refs for inclusion in system prompt.

        :return: formatted section or empty string
        :rtype: str
        """
        if not self._memory_refs_projection:
            return ""
        lines = ["Previously recalled in this conversation (use recall_memory with the ID and type shown):"]
        for ref in self._memory_refs_projection:
            itype = ref["item_type"]
            tag = f"[{itype}:{ref['item_id']}]"
            lines.append(f"- {tag} type: {itype} — {ref['short_desc']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context_prompt(self) -> str:
        """Return a non-None prompt suitable for system-prompt injection.

        Compatibility shim for the canonical
        :func:`threetears.langgraph.agent_node`, which expects a sync
        ``build_context_prompt() -> str`` on the ``context_manager``
        slot in ``configurable``. Returns the empty string when there
        is no context, instead of ``None``, so callers can concatenate
        the result into the system prompt unconditionally.

        :return: formatted context string, or "" when no context exists
        :rtype: str
        """
        raw = self.build_conversation_context()
        return raw or ""

    def build_conversation_context(self) -> str | None:
        """Format variables and tool results into a prompt string.

        Returns ``None`` if there is no context to include.
        """
        variables = [i for i in self.items if i["context_type"] == "variable"]
        tool_results = [i for i in self.items if i["context_type"] == "tool_result"]

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
        return bool(self.items)
