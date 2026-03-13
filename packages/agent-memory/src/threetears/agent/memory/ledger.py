"""Memory ledger -- tracks surfaced items per conversation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID


class MemoryLedger:
    """Tracks which memories/media/chunks have been surfaced in a conversation.

    FIFO eviction at capacity 50.
    """

    MAX_SIZE = 50

    def __init__(self) -> None:
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
            self._refs[str(row["item_id"])] = {
                "item_type": row["item_type"],
                "short_desc": row["short_desc"],
                "date_added": row["date_added"].isoformat() if row["date_added"] else "",
            }

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
            await pool.execute(
                "DELETE FROM conversation_memory_refs WHERE conversation_id = $1 AND item_id = $2",
                conversation_id,
                UUID(oldest_key),
            )

        now = datetime.now(timezone.utc)
        self._refs[item_id] = {
            "item_type": item_type,
            "short_desc": short_desc,
            "date_added": now.isoformat(),
        }

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

    def __len__(self) -> int:
        return len(self._refs)
