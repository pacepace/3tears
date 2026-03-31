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
    """Create a NATS client mock with in-memory KV store."""
    store: dict[str, bytes] = {}
    nats = AsyncMock()
    nats.bucket_name = MagicMock(return_value="test_collections")

    async def _get(bucket: str, key: str) -> bytes | None:
        return store.get(key)

    async def _put(bucket: str, key: str, value: bytes) -> bool:
        store[key] = value
        return True

    async def _delete(bucket: str, key: str) -> bool:
        store.pop(key, None)
        return True

    nats.get = AsyncMock(side_effect=_get)
    nats.put = AsyncMock(side_effect=_put)
    nats.delete = AsyncMock(side_effect=_delete)
    nats.publish = AsyncMock()
    nats.subscribe = AsyncMock()
    nats._store = store
    return nats


class FakePool:
    """In-memory mock of asyncpg.Pool for testing."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    async def fetchrow(self, sql: str, *args: object) -> dict[str, Any] | None:
        sql_lower = sql.strip().lower()
        if "returning context_id" in sql_lower:
            context_id, conversation_id = args[0], args[1]
            key = args[3]
            for row in self._rows.values():
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
            self._rows[str(context_id)] = row_data
            return {"context_id": context_id}

        if "select * from context_items where context_id" in sql_lower:
            cid = str(args[0])
            return self._rows.get(cid)

        if "select count" in sql_lower:
            cid = str(args[0])
            cnt = sum(
                1
                for r in self._rows.values()
                if str(r["conversation_id"]) == cid and r["context_type"] == "tool_result"
            )
            return {"cnt": cnt}

        return None

    async def fetch(self, sql: str, *args: object) -> list[dict[str, Any]]:
        sql_lower = sql.strip().lower()

        if "select * from context_items" in sql_lower and "where conversation_id" in sql_lower:
            cid = str(args[0])
            rows = [r for r in self._rows.values() if str(r["conversation_id"]) == cid]
            rows.sort(key=lambda r: _naive(r.get("date_created", datetime.min)))
            return rows

        if "select context_id from context_items" in sql_lower and "order by date_accessed asc" in sql_lower:
            cid = str(args[0])
            limit = int(args[1])
            tool_results = [
                r
                for r in self._rows.values()
                if str(r["conversation_id"]) == cid and r["context_type"] == "tool_result"
            ]
            tool_results.sort(key=lambda r: _naive(r.get("date_accessed", datetime.min)))
            return [{"context_id": r["context_id"]} for r in tool_results[:limit]]

        return []

    async def execute(self, sql: str, *args: object) -> str:
        sql_lower = sql.strip().lower()

        if "insert into context_items" in sql_lower:
            context_id = args[0]
            metadata_raw = args[7]
            row_data = {
                "context_id": context_id,
                "conversation_id": args[1],
                "context_type": args[2],
                "key": args[3],
                "short_desc": args[4],
                "long_desc": args[5],
                "content": args[6],
                "metadata": json.loads(metadata_raw) if metadata_raw else None,
                "date_accessed": args[8],
                "date_created": args[9],
                "date_updated": args[10],
            }
            self._rows[str(context_id)] = row_data
            return "INSERT 0 1"

        if "delete from context_items" in sql_lower:
            cid = str(args[0])
            if cid in self._rows:
                del self._rows[cid]
                return "DELETE 1"
            return "DELETE 0"

        if "update context_items set date_accessed" in sql_lower:
            cid = str(args[0])
            if cid in self._rows:
                self._rows[cid]["date_accessed"] = args[1]
                return "UPDATE 1"
            return "UPDATE 0"

        if "update context_items set" in sql_lower:
            cid = str(args[0])
            if cid in self._rows:
                self._rows[cid]["short_desc"] = args[1]
                self._rows[cid]["long_desc"] = args[2]
                self._rows[cid]["content"] = args[3]
                return "UPDATE 1"
            return "UPDATE 0"

        return "0"
