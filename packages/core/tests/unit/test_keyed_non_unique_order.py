"""A keyset read in an order that, with its pins, holds none of its table's declared keys is refused at run time.

Each page after the first starts past the last row read, so the rows sharing that row's order values are never read:
the P0 of 2026-11-13 came in through an order built at run time, which no static rule sees. Every ``TableSchema``
registers its keys on construction (``threetears.interfaces.keyed.keys``); ``keyset_rows`` and ``keyset_page`` past a
cursor refuse an order holding none of them, before any statement. A table no declaration names passes.

A table is known by its identity: an L3 read of the relation a declaration names. A warehouse relation, or one in
another schema, that shares a bare name with a declared L3 table is another table, judged by none of its keys.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import duckdb
import pytest

from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    Index,
    TableSchema,
)
from threetears.core.keyed import AnyOf, Dialect, InlineDuckDB, Sort, keyset_page, keyset_rows, keyset_cursor
from threetears.interfaces.keyed.keys import TableKey, refuse_non_unique_order, register_table_keys

#: tables named for this module alone, so no other declaration's keys stand beside theirs
_EVENTS = "o64_order_events"
_GROUPS = "o64_order_groups"
_AUDIT = "o65_audit.o65_order_audit"
_SOURCES = "o65_order_sources"

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


TableSchema(
    name=_AUDIT,
    primary_key="event_id",
    columns=[Column("event_id", UUID_TYPE), Column("created_at", DATETIMETZ_TYPE)],
)


TableSchema(
    name=_SOURCES,
    primary_key="id",
    columns=[Column("id", UUID_TYPE), Column("owner_agent_id", UUID_TYPE, nullable=True), Column("name", STRING_TYPE)],
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
        refuse_non_unique_order(_EVENTS, {"agent_id": AnyOf([uuid4(), uuid4()])}, [Sort("event_id")], l3=True)
    refuse_non_unique_order(_EVENTS, {"agent_id": uuid4()}, [Sort("event_id")], l3=True)


def test_a_partial_unique_index_is_a_key_where_its_pins_hold_its_predicate() -> None:
    # (customer_id, name) WHERE customer_id IS NOT NULL: a key of one customer's groups
    refuse_non_unique_order(_GROUPS, {"row_scope": "customer", "customer_id": uuid4()}, [Sort("name")], l3=True)
    with pytest.raises(ValueError, match="holds none of its keys"):
        refuse_non_unique_order(_GROUPS, {"customer_id": None}, [Sort("name")], l3=True)
    with pytest.raises(ValueError, match="holds none of its keys"):
        refuse_non_unique_order(_GROUPS, {"customer_id": AnyOf([uuid4()])}, [Sort("name")], l3=True)


def test_a_partition_column_of_the_key_places_rows_and_need_not_be_held() -> None:
    refuse_non_unique_order(_GROUPS, {}, [Sort("group_id")], l3=True)


def test_a_table_no_declaration_names_passes() -> None:
    refuse_non_unique_order("o64_order_undeclared", {}, [Sort("created_at")], l3=True)
    refuse_non_unique_order(object(), {}, [Sort("created_at")], l3=True)
    # a subquery or a three-part name is no declared relation
    refuse_non_unique_order(f"(SELECT * FROM {_EVENTS})", {}, [Sort("created_at")], l3=True)
    refuse_non_unique_order(f"db.public.{_EVENTS}", {}, [Sort("created_at")], l3=True)


class _Warehouse(_Empty):
    """a datasource's reader: it names its engine's dialect, and its relations are the warehouse's, not L3's."""

    sql_dialect = Dialect.REDSHIFT


class _PostgresDatasource(_Empty):
    """a datasource's reader on a PostgreSQL warehouse: its dialect is L3's, its tables are not."""

    @property
    def sql_dialect(self) -> Dialect:
        return Dialect.POSTGRES


async def test_a_warehouse_relation_sharing_a_declared_tables_bare_name_is_read_in_its_own_order() -> None:
    # the warehouse's o64_order_events is not the L3 table: (agent_id, event_id) is no key of it, and a refusal
    # would stop a read that holds its own
    for reader in (_Warehouse(), _PostgresDatasource()):
        await keyset_rows(
            reader, _EVENTS, columns=["event_id", "created_at"], by={"agent_id": uuid4()}, order=[Sort("created_at")]
        )
        await keyset_rows(
            reader,
            f"warehouse.{_EVENTS}",
            columns=["event_id", "created_at"],
            by={},
            order=[Sort("created_at")],
        )
        assert len(reader.sent) == 2
    # an in-process DuckDB's relation of that name is its own as well
    with duckdb.connect() as conn:
        conn.execute(f"CREATE TABLE {_EVENTS} (agent_id UUID, event_id UUID, created_at TIMESTAMPTZ)")
        rows = await keyset_rows(
            InlineDuckDB(conn), _EVENTS, columns=["event_id", "created_at"], by={}, order=[Sort("created_at")]
        )
    assert rows == []
    refuse_non_unique_order(_EVENTS, {}, [Sort("created_at")], l3=False)


async def test_the_same_bare_name_in_another_schema_is_another_table() -> None:
    conn = _Empty()
    await keyset_rows(
        conn, f"elsewhere.{_EVENTS}", columns=["event_id", "created_at"], by={}, order=[Sort("created_at")]
    )
    # a table declared with its schema is that schema's: read bare, the search path may find another
    await keyset_rows(conn, "o65_order_audit", columns=["event_id", "created_at"], by={}, order=[Sort("created_at")])
    assert len(conn.sent) == 2


async def test_the_declared_l3_table_still_refuses_however_its_name_is_spelled() -> None:
    for name in (_EVENTS, f'"{_EVENTS}"', _EVENTS.upper()):
        with pytest.raises(ValueError, match="holds none of its keys"):
            await keyset_rows(
                _NoStatement(), name, columns=["event_id", "created_at"], by={}, order=[Sort("created_at")]
            )
    for name in (_AUDIT, '"o65_audit"."o65_order_audit"', "O65_AUDIT . o65_order_audit"):
        with pytest.raises(ValueError, match="holds none of its keys"):
            await keyset_rows(
                _NoStatement(), name, columns=["event_id", "created_at"], by={}, order=[Sort("created_at")]
            )
    # a quoted name keeps its case: "O64_ORDER_EVENTS" is another relation
    await keyset_rows(
        _Empty(), f'"{_EVENTS.upper()}"', columns=["event_id", "created_at"], by={}, order=[Sort("created_at")]
    )


def test_a_consumer_adds_its_own_migrations_keys_to_a_table_another_package_declares() -> None:
    # the consumer's migration adds (owner_agent_id, name) WHERE owner_agent_id IS NOT NULL to the declared table
    pinned = {"owner_agent_id": uuid4()}
    with pytest.raises(ValueError, match="holds none of its keys"):
        refuse_non_unique_order(_SOURCES, pinned, [Sort("name")], l3=True)
    register_table_keys(_SOURCES, (TableKey(("owner_agent_id", "name"), not_null=("owner_agent_id",)),))
    refuse_non_unique_order(_SOURCES, pinned, [Sort("name")], l3=True)
    # the declaration's own key still stands beside it; the consumer's holds only where its predicate does
    refuse_non_unique_order(_SOURCES, {}, [Sort("name"), Sort("id")], l3=True)
    with pytest.raises(ValueError, match="holds none of its keys"):
        refuse_non_unique_order(_SOURCES, {"owner_agent_id": None}, [Sort("name")], l3=True)


@pytest.mark.parametrize("expression", ["lower(name)", "(lower(name))", "name || id", "", "metadata ->> 'mcp_name'"])
def test_an_expression_index_is_never_a_column_key(expression: str) -> None:
    with pytest.raises(ValueError, match="column"):
        TableKey((expression,))
    with pytest.raises(ValueError, match="column"):
        TableKey(("customer_id", expression))
    with pytest.raises(ValueError, match="column"):
        TableKey(("name",), not_null=(expression,))
    with pytest.raises(ValueError, match="column"):
        register_table_keys(_SOURCES, (TableKey((expression,)),))


def test_a_key_names_a_column() -> None:
    with pytest.raises(ValueError, match="at least one column"):
        TableKey(())
