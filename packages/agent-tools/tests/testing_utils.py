"""Shared test utilities for agent-tools tests."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import MetaData

from threetears.agent.tools.collections import context_items_table


def _naive(dt: datetime) -> datetime:
    """Strip timezone info for safe comparison in test mocks."""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def make_context_metadata() -> MetaData:
    """Create SA metadata with the context_items table."""
    metadata = MetaData()
    context_items_table(metadata)
    return metadata


def make_nats_mock() -> AsyncMock:
    """typed-wrapper NATS mock with in-memory KV bucket.

    matches :class:`threetears.nats.NatsClient` /
    :class:`threetears.nats.NatsKvBucket` shapes -- ``kv_bucket`` is
    awaited, returns a bucket whose ``get`` / ``put`` / ``delete`` are
    kw-only. ``publish`` / ``subscribe_typed`` accept the typed
    Pydantic-message kw-only signatures of the wrapper.
    """
    store: dict[str, bytes] = {}

    async def _get(*, key: str) -> bytes | None:
        return store.get(key)

    async def _put(*, key: str, value: bytes) -> int:
        store[key] = value
        return len(store)

    async def _delete(*, key: str, revision: int | None = None) -> bool:  # noqa: ARG001
        existed = key in store
        store.pop(key, None)
        return existed or revision is None

    bucket = AsyncMock()
    bucket.get = AsyncMock(side_effect=_get)
    bucket.put = AsyncMock(side_effect=_put)
    bucket.delete = AsyncMock(side_effect=_delete)

    nats = AsyncMock()
    nats.kv_bucket = AsyncMock(return_value=bucket)
    nats.publish = AsyncMock()
    nats.subscribe_typed = AsyncMock()
    nats.store = store
    nats.bucket = bucket
    return nats


class FakePool:
    """In-memory mock of asyncpg.Pool for testing."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def fetchrow(self, sql: str, *args: object) -> dict[str, Any] | None:
        sql_lower = sql.strip().lower()
        if "returning context_id" in sql_lower:
            # upsert_variable: (context_id, conversation_id, context_type, key, ...)
            context_id, conversation_id = args[0], args[1]
            key = args[3]
            for row in self.rows.values():
                if (
                    str(row["conversation_id"]) == str(conversation_id)
                    and row["key"] == key
                    and row["context_type"] == "variable"
                ):
                    row["short_desc"] = args[4]
                    row["long_desc"] = args[5]
                    row["content"] = args[6]
                    row["metadata"] = json.loads(args[7]) if args[7] else None
                    row["date_accessed"] = args[8]
                    row["date_updated"] = args[10]
                    return {"context_id": row["context_id"]}
            row_data = {
                "context_id": context_id,
                "conversation_id": conversation_id,
                "context_type": "variable",
                "key": key,
                "short_desc": args[4],
                "long_desc": args[5],
                "content": args[6],
                "metadata": json.loads(args[7]) if args[7] else None,
                "date_accessed": args[8],
                "date_created": args[9],
                "date_updated": args[10],
            }
            self.rows[str(context_id)] = row_data
            return {"context_id": context_id}

        # composite-pk fetch: WHERE conversation_id = $1 AND context_id = $2
        if "select * from context_items" in sql_lower and "where conversation_id" in sql_lower and "and context_id" in sql_lower:
            cid = str(args[1])
            return self.rows.get(cid)

        if "select count" in sql_lower:
            cid = str(args[0])
            cnt = sum(
                1
                for r in self.rows.values()
                if str(r["conversation_id"]) == cid and r["context_type"] == "tool_result"
            )
            return {"cnt": cnt}

        return None

    async def fetch(self, sql: str, *args: object) -> list[dict[str, Any]]:
        sql_lower = sql.strip().lower()

        if "select * from context_items" in sql_lower and "where conversation_id" in sql_lower:
            cid = str(args[0])
            rows = [r for r in self.rows.values() if str(r["conversation_id"]) == cid]
            rows.sort(key=lambda r: _naive(r.get("date_created", datetime.min)))
            return rows

        if "select context_id from context_items" in sql_lower and "order by date_accessed asc" in sql_lower:
            cid = str(args[0])
            limit = int(args[1])
            tool_results = [
                r
                for r in self.rows.values()
                if str(r["conversation_id"]) == cid and r["context_type"] == "tool_result"
            ]
            tool_results.sort(key=lambda r: _naive(r.get("date_accessed", datetime.min)))
            return [{"context_id": r["context_id"]} for r in tool_results[:limit]]

        return []

    async def execute(self, sql: str, *args: object) -> str:
        sql_lower = sql.strip().lower()

        if "insert into context_items" in sql_lower:
            # composite-pk schema column order: conversation_id, context_id,
            # context_type, key, short_desc, long_desc, content, metadata,
            # date_accessed, date_created, date_updated
            conversation_id = args[0]
            context_id = args[1]
            metadata_raw = args[7]
            row_data = {
                "conversation_id": conversation_id,
                "context_id": context_id,
                "context_type": args[2],
                "key": args[3],
                "short_desc": args[4],
                "long_desc": args[5],
                "content": args[6],
                "metadata": json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw,
                "date_accessed": args[8],
                "date_created": args[9],
                "date_updated": args[10],
            }
            self.rows[str(context_id)] = row_data
            return "INSERT 0 1"

        if "delete from context_items" in sql_lower:
            # composite-pk delete: conversation_id = $1 AND context_id = $2
            cid = str(args[1])
            if cid in self.rows:
                del self.rows[cid]
                return "DELETE 1"
            return "DELETE 0"

        if "update context_items set date_accessed" in sql_lower:
            # composite-pk: WHERE conversation_id = $1 AND context_id = $2
            # SET date_accessed = $3
            cid = str(args[1])
            if cid in self.rows:
                self.rows[cid]["date_accessed"] = args[2]
                return "UPDATE 1"
            return "UPDATE 0"

        if "update context_items set" in sql_lower:
            # CAS-style update keyed by composite pk
            cid = str(args[1])
            if cid in self.rows:
                self.rows[cid]["short_desc"] = args[2]
                self.rows[cid]["long_desc"] = args[3]
                self.rows[cid]["content"] = args[4]
                return "UPDATE 1"
            return "UPDATE 0"

        return "0"
