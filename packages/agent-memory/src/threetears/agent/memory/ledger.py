"""Memory ledger -- tracks surfaced items per conversation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import Column, MetaData, String, Table, Text

__all__ = [
    "MemoryLedger",
]

if TYPE_CHECKING:
    from threetears.core.cache.sqlite import SQLiteBackend

_MEMORY_REFS_METADATA = MetaData()

_memory_refs_table = Table(
    "memory_refs",
    _MEMORY_REFS_METADATA,
    Column("key", String, primary_key=True),
    Column("value", Text, nullable=True),
    Column("date_updated", String, nullable=True),
)


class MemoryLedger:
    """Tracks which memories/media/chunks have been surfaced in a conversation.

    FIFO eviction at capacity 50. when l1_backend is provided, refs are
    persisted to SQLite for crash recovery. dict is retained as fast
    lookup index and fallback when l1_backend is None.
    """

    MAX_SIZE = 50

    def __init__(self, l1_backend: SQLiteBackend | None = None) -> None:
        """initialize memory ledger with optional L1 persistence.

        :param l1_backend: optional SQLiteBackend for persistent memory refs
        :ptype l1_backend: SQLiteBackend | None
        """
        self._l1 = l1_backend
        if self._l1 is not None and not self._l1.is_initialized():
            self._l1.initialize(_MEMORY_REFS_METADATA)
        self._refs: dict[str, dict[str, Any]] = {}

    async def load(self, pool: Any, conversation_id: UUID) -> None:
        """Load from conversation_memory_refs table."""
        rows = await pool.fetch(
            """
            SELECT item_id, item_type, short_desc, date_added
            FROM conversation_memory_refs
            WHERE conversation_id = $1
            ORDER BY date_added ASC
            """,
            conversation_id,
        )
        for row in rows:
            ref_data = {
                "item_type": row["item_type"],
                "short_desc": row["short_desc"],
                "date_added": row["date_added"].isoformat() if row["date_added"] else "",
            }
            item_id_str = str(row["item_id"])
            self._refs[item_id_str] = ref_data
            self._persist_ref_to_l1(item_id_str, ref_data)

    async def add_ref(
        self,
        pool: Any,
        conversation_id: UUID,
        item_id: str,
        item_type: str,
        short_desc: str,
    ) -> None:
        """Add ref. Truncate desc to 150 chars. Evict oldest if at capacity."""
        if item_id in self._refs:
            return

        short_desc = short_desc[:150] if len(short_desc) > 150 else short_desc

        if len(self._refs) >= self.MAX_SIZE:
            oldest_key = next(iter(self._refs))
            del self._refs[oldest_key]
            if self._l1 is not None:
                self._l1.delete_by_id("memory_refs", oldest_key, primary_key="key")
            await pool.execute(
                "DELETE FROM conversation_memory_refs WHERE conversation_id = $1 AND item_id = $2",
                conversation_id,
                UUID(oldest_key),
            )

        now = datetime.now(timezone.utc)
        ref_data = {
            "item_type": item_type,
            "short_desc": short_desc,
            "date_added": now.isoformat(),
        }
        self._refs[item_id] = ref_data
        self._persist_ref_to_l1(item_id, ref_data)

        await pool.execute(
            """
            INSERT INTO conversation_memory_refs (conversation_id, item_id, item_type, short_desc, date_added)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (conversation_id, item_id) DO NOTHING
            """,
            conversation_id,
            UUID(item_id),
            item_type,
            short_desc,
            now,
        )

    @property
    def ledgered_ids(self) -> set[str]:
        """Set of item IDs in the ledger."""
        return set(self._refs.keys())

    def build_context(self) -> str:
        """Format ledger for system prompt."""
        if not self._refs:
            return ""

        lines = ["Previously recalled in this conversation (use recall_memory with the ID and type shown):"]
        for item_id, ref in self._refs.items():
            itype = ref["item_type"]
            tag = f"[{itype}:{item_id}]"
            lines.append(f"- {tag} type: {itype} — {ref['short_desc']}")

        return "\n".join(lines)

    def _persist_ref_to_l1(self, item_id: str, ref_data: dict[str, Any]) -> None:
        """persist single memory ref to L1 SQLiteBackend if available.

        :param item_id: unique item identifier
        :ptype item_id: str
        :param ref_data: ref metadata (item_type, short_desc, date_added)
        :ptype ref_data: dict[str, Any]
        """
        if self._l1 is None:
            return
        self._l1.upsert(
            "memory_refs",
            {
                "key": item_id,
                "value": json.dumps(ref_data),
                "date_updated": ref_data.get("date_added", ""),
            },
            primary_key="key",
        )

    def __len__(self) -> int:
        return len(self._refs)
