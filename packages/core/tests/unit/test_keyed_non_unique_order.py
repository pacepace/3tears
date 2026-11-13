"""A keyset read in an order that, with its pins, holds none of its table's declared keys is refused at run time.

Each page after the first starts past the last row read, so the rows sharing that row's order values are never read:
the P0 of 2026-11-13 came in through an order built at run time, which no static rule sees. Every ``TableSchema``
registers its keys on construction (``threetears.interfaces.keyed.keys``); ``keyset_rows`` and ``keyset_page`` past a
cursor refuse an order holding none of them, before any statement. A table no declaration names passes.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    Index,
    TableSchema,
)
from threetears.core.keyed import AnyOf, Sort, keyset_page, keyset_rows, keyset_cursor
from threetears.interfaces.keyed.keys import refuse_non_unique_order

#: tables named for this module alone, so no other declaration's keys stand beside theirs
_EVENTS = "o64_order_events"
_GROUPS = "o64_order_groups"

TableSchema(
    name=_EVENTS,
    primary_key=("agent_id", "event_id"),
    columns=[Column("agent_id", UUID_TYPE), Column("event_id", UUID_TYPE), Column("created_at", DATETIMETZ_TYPE)],
)
TableSchema(
    name=_GROUPS,
    primary_key=("row_scope", "group_id"),
    columns=[
        Column("row_scope", STRING_TYPE, partition=True),
        Column("group_id", UUID_TYPE),
        Column("customer_id", UUID_TYPE, nullable=True),
        Column("name", STRING_TYPE),
    ],
    indexes=(
        Index("o64_order_groups_customer_name", "customer_id", "name", unique=True, where="customer_id IS NOT NULL"),
    ),
)


class _NoStatement:
    """a connection a refused read must never reach."""

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        raise AssertionError(f"a refused read sent a statement: {sql}")


class _Empty:
    """a connection answering every statement with no rows."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        self.sent.append(sql)
        return []


async def test_a_registered_table_refuses_a_non_unique_order_before_any_statement() -> None:
    agent = uuid4()
    with pytest.raises(ValueError, match="holds none of its keys"):
        await keyset_rows(
            _NoStatement(),
            _EVENTS,
            columns=["event_id", "created_at"],
            by={"agent_id": agent},
            order=[Sort("created_at")],
        )
    order = [Sort("created_at", cast="timestamptz")]
    cursor = keyset_cursor(order, {"created_at": "2026-11-13T00:00:00+00:00"})
    with pytest.raises(ValueError, match="holds none of its keys"):
        await keyset_page(
            _NoStatement(), _EVENTS, columns=["created_at"], by={"agent_id": agent}, order=order, limit=5, cursor=cursor
        )


async def test_an_order_ending_in_the_key_reads_and_a_first_page_reads_without_a_cursor() -> None:
    conn = _Empty()
    agent = uuid4()
    await keyset_rows(
        conn,
        _EVENTS,
        columns=["event_id", "created_at"],
        by={"agent_id": agent},
        order=[Sort("created_at"), Sort("event_id")],
    )
    # one page, past nothing: a tie can reorder it, never skip a row
    await keyset_page(
        conn, _EVENTS, columns=["created_at"], by={"agent_id": agent}, order=[Sort("created_at")], limit=5
    )
    assert len(conn.sent) == 2


def test_a_set_pin_holds_no_column_of_the_key() -> None:
    with pytest.raises(ValueError, match="holds none of its keys"):
        refuse_non_unique_order(_EVENTS, {"agent_id": AnyOf([uuid4(), uuid4()])}, [Sort("event_id")])
    refuse_non_unique_order(_EVENTS, {"agent_id": uuid4()}, [Sort("event_id")])


def test_a_partial_unique_index_is_a_key_where_its_pins_hold_its_predicate() -> None:
    # (customer_id, name) WHERE customer_id IS NOT NULL: a key of one customer's groups
    refuse_non_unique_order(_GROUPS, {"row_scope": "customer", "customer_id": uuid4()}, [Sort("name")])
    with pytest.raises(ValueError, match="holds none of its keys"):
        refuse_non_unique_order(_GROUPS, {"customer_id": None}, [Sort("name")])
    with pytest.raises(ValueError, match="holds none of its keys"):
        refuse_non_unique_order(_GROUPS, {"customer_id": AnyOf([uuid4()])}, [Sort("name")])


def test_a_partition_column_of_the_key_places_rows_and_need_not_be_held() -> None:
    refuse_non_unique_order(_GROUPS, {}, [Sort("group_id")])


def test_a_table_no_declaration_names_passes() -> None:
    refuse_non_unique_order("o64_order_undeclared", {}, [Sort("created_at")])
    refuse_non_unique_order(object(), {}, [Sort("created_at")])
