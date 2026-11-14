"""copy_keyed: a prefix's rows as COPY CSV, the statement keyset_rows reads, through asyncpg's copy_from_query."""

from __future__ import annotations

from typing import Any

import pytest

from threetears.core.keyed import Selected, Sort, copy_keyed


class _Pool:
    """an asyncpg pool's copy_from_query, answering in two chunks and keeping what it was asked."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def copy_from_query(self, query: str, *args: Any, output: Any, **options: Any) -> str:
        self.asked.append((query, args, options))
        await output(b'"a",1\n')
        await output(b'"b",2\n')
        return "COPY 2"


async def test_the_prefix_s_rows_are_copied_in_order_as_csv_and_joined() -> None:
    """Flips if copy_keyed pages (a LIMIT), drops the prefix or its order, loses a chunk, or asks another format."""
    pool = _Pool()
    text = await copy_keyed(
        pool,
        "nats_pod_connections",
        columns=["principal_id", Selected("to_char(t.date_admitted, 'YYYY')", "date_admitted")],
        by={"principal_kind": "agent", "principal_id": "p1"},
        order=[Sort("date_admitted", cast="timestamptz"), Sort("connection_id", cast="uuid")],
        force_quote=["principal_id"],
    )
    assert text == b'"a",1\n"b",2\n'
    ((query, args, options),) = pool.asked
    assert args == ("agent", "p1") and options == {"format": "csv", "force_quote": ["principal_id"]}
    assert 'WHERE t."principal_kind" = $1 AND t."principal_id" = $2' in query
    # the pinned columns are constants on every row: never in the ORDER BY
    assert query.endswith('ORDER BY t."date_admitted" ASC, t."connection_id" ASC')
    assert "LIMIT" not in query


async def test_an_empty_order_is_refused() -> None:
    """Flips if a copy with no order is sent (its rows would come in no order)."""
    with pytest.raises(ValueError, match="names its sort"):
        await copy_keyed(_Pool(), "t", columns=["a"], by={}, order=[])
