"""On YugabyteDB a keyed read names the declared index that serves its pins and its order.

YugabyteDB read one agent's conversations of one status through the primary key ``(agent_id, conversation_id)``,
filtering the status, on every one of 15 fresh worlds (3tears #87's live7: 13-17 rows for a page of 9), rather than the
declared ``(agent_id, status, conversation_id)``; statistics did not move it. Every ``TableSchema`` registers its plain
B-tree indexes (``threetears.interfaces.keyed.keys``); ``keyset_page``, ``keyset_rows`` and an ordered ``read_by``
start their statement with ``/*+ IndexOnlyScan(t <index>) */`` there, and nowhere else.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.core.testing.captured import CapturedL3
from threetears.core.collections.schema_backed import STRING_TYPE, UUID_TYPE, Column, Index, TableSchema
from threetears.core.keyed import (
    AnyOf,
    DeclaredIndex,
    Sort,
    index_hint,
    keyset_cursor,
    keyset_page,
    keyset_rows,
    read_by,
    serving_index,
)

#: a table named for this module alone, so no other declaration's indexes stand beside its own
_CONV = "o88_hint_conversations"

TableSchema(
    name=_CONV,
    primary_key=("agent_id", "conversation_id"),
    columns=[
        Column("agent_id", UUID_TYPE),
        Column("conversation_id", UUID_TYPE),
        Column("user_id", UUID_TYPE),
        Column("status", STRING_TYPE),
        Column("name", STRING_TYPE),
    ],
    indexes=(
        Index("o88_conv_agent_status", "agent_id", "status", "conversation_id"),
        Index("o88_conv_agent_user_status", "agent_id", "user_id", "status", "conversation_id"),
        Index("o88_conv_open", "agent_id", "conversation_id", where="status = 'open'"),
        Index("o88_conv_name_c", "agent_id", "name", "conversation_id", collate=(("name", "C"),)),
    ),
)

_BY_STATUS = [Sort("status", descending=True), Sort("conversation_id", cast="uuid", descending=True)]
_BY_ID = [Sort("conversation_id", cast="uuid", descending=True)]


class _Reader:
    """an L3 reader that records each statement, answers its server's version, and no rows."""

    def __init__(self, version: str) -> None:
        self.version = version
        self.asked: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.asked.append(query)
        if query.startswith("SELECT version()"):
            return [{"v": self.version}]
        return []


def _yugabyte() -> _Reader:
    return _Reader("PostgreSQL 15.12-YB-2026.1.0.0-b0 on x86_64")


def _postgres() -> _Reader:
    return _Reader("PostgreSQL 16.4 on x86_64")


class TestServingIndex:
    def test_the_index_holding_the_pins_then_the_order(self) -> None:
        """Flips if a read pinning the agent sorted by status, or pinning the agent and a status (a pinned sort column
        orders nothing), is served by anything but (agent_id, status, conversation_id), or one pinning the agent, a
        user and a status by anything but the index holding all three."""
        assert serving_index(_CONV, {"agent_id": 1}, _BY_STATUS).name == "o88_conv_agent_status"
        assert serving_index(_CONV, {"agent_id": 1, "status": "a"}, _BY_STATUS).name == "o88_conv_agent_status"
        held = serving_index(_CONV, {"agent_id": 1, "user_id": 2, "status": "a"}, _BY_ID)
        assert held is not None and held.name == "o88_conv_agent_user_status"

    def test_the_primary_key_serving_as_many_pins_wins_and_needs_no_hint(self) -> None:
        """Flips if a read the primary key serves (the agent, by id) is handed a secondary index, or its hint is
        anything but nothing."""
        index = serving_index(_CONV, {"agent_id": 1}, _BY_ID)
        assert index is not None and index.primary
        assert index_hint(index) == ""

    def test_no_index_where_none_is_declared_to_serve(self) -> None:
        """Flips if an index is named for a set pin (AnyOf), a sort by an expression, an order no index holds after
        the pins, a direction mixed against the index, a relation no declaration registers, a derived relation, or a
        partial or collated index (a read must state the one's predicate; the other orders by another collation)."""
        assert serving_index(_CONV, {"agent_id": AnyOf([1, 2])}, _BY_STATUS) is None
        assert serving_index(_CONV, {"agent_id": 1}, [Sort("status", expression='lower(t."status")')]) is None
        assert serving_index(_CONV, {"agent_id": 1}, [Sort("name"), Sort("conversation_id")]) is None
        assert serving_index(_CONV, {"agent_id": 1}, [Sort("status"), Sort("conversation_id", descending=True)]) is None
        assert serving_index("o88_never_declared", {"agent_id": 1}, _BY_STATUS) is None
        assert serving_index(_CONV, {"status": "a"}, _BY_ID) is None
        assert serving_index(_CONV, {"agent_id": 1, "status": "a"}, [Sort("status")]) is None

    def test_a_declared_index_names_plain_identifiers(self) -> None:
        """Flips if an index a hint could not name unquoted, or one sorting a column it lacks, is accepted."""
        with pytest.raises(ValueError, match="plain identifiers"):
            DeclaredIndex('idx "x"', ("a",))
        with pytest.raises(ValueError, match="descending"):
            DeclaredIndex("idx", ("a",), frozenset({"b"}))


class TestTheHintOnYugabyteOnly:
    async def test_a_page_past_a_cursor_names_the_index_on_yugabyte(self) -> None:
        """Flips if a page of one status on YugabyteDB does not start with the IndexOnlyScan hint naming
        (agent_id, status, conversation_id), or a page on PostgreSQL carries any hint."""
        cursor = keyset_cursor(_BY_STATUS, {"status": "a", "conversation_id": "00000000-0000-0000-0000-000000000001"})
        for reader, hinted in ((_yugabyte(), True), (_postgres(), False)):
            await keyset_page(
                reader,
                _CONV,
                columns=("conversation_id", "status"),
                by={"agent_id": 1, "status": "a"},
                order=_BY_STATUS,
                limit=8,
                cursor=cursor,
            )
            (page,) = [sql for sql in reader.asked if not sql.startswith("SELECT version()")]
            assert page.startswith("/*+ IndexOnlyScan(t o88_conv_agent_status) */ ") is hinted, page
            assert ("/*+" in page) is hinted, page

    async def test_a_first_page_and_a_drain_and_an_ordered_read_by_name_it(self) -> None:
        """Flips if a first page, every statement of a drain, or an ordered read_by on YugabyteDB starts with
        anything but the hint, or the database is asked more than once."""
        reader = _yugabyte()
        await keyset_page(
            reader, _CONV, columns=("conversation_id", "status"), by={"agent_id": 1}, order=_BY_STATUS, limit=8
        )
        await keyset_rows(reader, _CONV, columns=("conversation_id", "status"), by={"agent_id": 1}, order=_BY_STATUS)
        await read_by(reader, _CONV, by={"agent_id": 1, "user_id": 2, "status": "a"}, order=_BY_ID)
        reads = [sql for sql in reader.asked if not sql.startswith("SELECT version()")]
        assert [sql.split(" */ ")[0] for sql in reads] == [
            "/*+ IndexOnlyScan(t o88_conv_agent_status)",
            "/*+ IndexOnlyScan(t o88_conv_agent_status)",
            "/*+ IndexOnlyScan(t o88_conv_agent_user_status)",
        ]
        assert len(reader.asked) - len(reads) == 1

    async def test_a_read_no_secondary_index_serves_asks_nothing_more(self) -> None:
        """Flips if a read the primary key serves, or an unordered read_by, asks the database which it is, or carries
        a hint."""
        reader = _yugabyte()
        await keyset_page(reader, _CONV, columns=("conversation_id",), by={"agent_id": 1}, order=_BY_ID, limit=8)
        await read_by(reader, _CONV, by={"agent_id": 1, "status": "a"})
        assert len(reader.asked) == 2 and not any("/*+" in sql or "version()" in sql for sql in reader.asked)


class TestARecorderAnswersForItsConnection:
    async def test_a_captured_pool_spells_for_its_real_connection_and_records_no_probe(self) -> None:
        """Flips if a statement recorder in front of a YugabyteDB connection records the dialect probe, or records the
        page without the hint its connection would be sent; or one with no connection is spelled for YugabyteDB."""
        behind = _yugabyte()
        captured = CapturedL3(real=behind, passes=lambda sql: "o88_hint" not in sql)
        await keyset_page(
            captured, _CONV, columns=("conversation_id", "status"), by={"agent_id": 1}, order=_BY_STATUS, limit=8
        )
        ((page, _),) = captured.asked
        assert page.startswith("/*+ IndexOnlyScan(t o88_conv_agent_status) */ "), page
        alone = CapturedL3()
        await keyset_page(
            alone, _CONV, columns=("conversation_id", "status"), by={"agent_id": 1}, order=_BY_STATUS, limit=8
        )
        ((page, _),) = alone.asked
        assert "/*+" not in page, page
