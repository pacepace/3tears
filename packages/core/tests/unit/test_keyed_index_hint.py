"""On YugabyteDB a keyed read names the declared index that serves its pins and its order.

YugabyteDB read one agent's conversations of one status through the primary key ``(agent_id, conversation_id)``,
filtering the status, on every one of 15 fresh worlds (3tears #87's live7: 13-17 rows for a page of 9), rather than the
declared ``(agent_id, status, conversation_id)``; statistics did not move it. Every ``TableSchema`` registers its plain
B-tree indexes (``threetears.interfaces.keyed.keys``); ``keyset_page``, ``keyset_rows`` and an ordered ``read_by``
start their statement with ``/*+ IndexOnlyScan(t <index>) */`` there, and nowhere else; so does an unordered
read_by when a declared index binds more of its pins than the primary key (coordinator ruling, #96).
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa

from threetears.core.testing.captured import CapturedL3
from threetears.core.collections.schema_backed import (
    STRING_TYPE,
    UUID_TYPE,
    Column,
    Index,
    TableSchema,
    UniqueConstraint,
    register_sa_table_indexes,
)
from threetears.core.keyed import (
    AnyOf,
    Fragment,
    DeclaredIndex,
    counts_by,
    Sort,
    index_hint,
    keyset_cursor,
    keyset_page,
    keyset_rows,
    read_by,
    register_table_indexes,
    serving_index,
)
from threetears.core.ddl.row_epoch import ROW_STAMPED
from threetears.interfaces.keyed.keys import partial_predicate, stated_literals
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

from packages.core.tests.optional_support import present

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
        Index("o88_conv_open_named", "agent_id", "name", "conversation_id", where="status = 'open'"),
        Index("o88_conv_kept", "agent_id", "user_id", "name", where="status = 'open' AND name IS NOT NULL"),
    ),
    l1_eviction=TEST_L1_TABLE,
)

#: a conversation's schedules, as agent-wake declares them: the primary key (conversation_id, schedule_id) and a
#: partial index of the conversation's active ones (idx_wake_schedules_conv_active), which binds no more leading
#: columns than the primary key; only its predicate's column says it holds fewer rows
_SCHED = "a100_hint_schedules"

TableSchema(
    name=_SCHED,
    primary_key=("conversation_id", "schedule_id"),
    columns=[
        Column("conversation_id", UUID_TYPE),
        Column("schedule_id", UUID_TYPE),
        Column("status", STRING_TYPE),
    ],
    indexes=(Index("a100_sched_conv_active", "conversation_id", where="status = 'active'"),),
    l1_eviction=TEST_L1_TABLE,
)

#: one agent's conversations by ref, as conversations declares them (v013, v015): a wide index of the ref in the
#: database's collation, and a narrower one holding it in code-point order for a prefix range
_REFS = "d7_hint_refs"

TableSchema(
    name=_REFS,
    primary_key=("agent_id", "conversation_id"),
    columns=[
        Column("agent_id", UUID_TYPE),
        Column("conversation_id", UUID_TYPE),
        Column("channel_type", STRING_TYPE),
        Column("conversation_ref", STRING_TYPE),
        Column("date_created", STRING_TYPE),
    ],
    indexes=(
        Index("d7_refs_created", "agent_id", "channel_type", "conversation_ref", "date_created", "conversation_id"),
        Index("d7_refs_c", "agent_id", "channel_type", "conversation_ref", collate=(("conversation_ref", "C"),)),
    ),
    l1_eviction=TEST_L1_TABLE,
)

#: one user's memories, as agent-memory declares them: by date, and the named ones (alias IS NOT NULL), each led by
#: the agent and the user
_ALIASES = "d7_hint_memories"

TableSchema(
    name=_ALIASES,
    primary_key=("agent_id", "memory_id"),
    columns=[
        Column("agent_id", UUID_TYPE),
        Column("memory_id", UUID_TYPE),
        Column("user_id", UUID_TYPE),
        Column("alias", STRING_TYPE, nullable=True),
        Column("date_created", STRING_TYPE),
    ],
    indexes=(
        Index("d7_mem_date", "agent_id", "user_id", "date_created"),
        Index("d7_mem_alias", "agent_id", "user_id", "alias", where="alias IS NOT NULL"),
    ),
    l1_eviction=TEST_L1_TABLE,
)

#: datasources as declared: a unique name (a UNIQUE constraint) and an index of the status
_NAMED = "d7_hint_datasources"

TableSchema(
    name=_NAMED,
    primary_key="id",
    columns=[Column("id", UUID_TYPE), Column("name", STRING_TYPE), Column("status", STRING_TYPE)],
    indexes=(Index("d7_ds_status", "status"),),
    unique_constraints=(UniqueConstraint("d7_ds_name", "name"),),
    l1_eviction=TEST_L1_TABLE,
)

#: datasources as declared since live-only names (Pace 2026-11-25): stamped, the name unique among live rows only
#: (``WHERE NOT gone``), and an index of the status
_LIVE_NAMED = "lu_hint_datasources"

TableSchema(
    name=_LIVE_NAMED,
    primary_key="id",
    columns=[Column("id", UUID_TYPE), Column("name", STRING_TYPE), Column("status", STRING_TYPE)],
    indexes=(Index("lu_ds_name", "name", unique=True, where="NOT gone"), Index("lu_ds_status", "status")),
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

#: a datasource's tables, as the hub declares them since v198: a name unique among a datasource's live rows only
#: (``WHERE NOT gone``) and, beside it, a whole index of the datasource for the database's foreign-key check during
#: cleanup, which must find gone children too
_FK_CHILD = "lt_hint_datasource_tables"

TableSchema(
    name=_FK_CHILD,
    primary_key="id",
    columns=[Column("id", UUID_TYPE), Column("datasource_id", UUID_TYPE), Column("name", STRING_TYPE)],
    indexes=(
        Index("lt_tables_ds", "datasource_id"),
        Index("lt_tables_ds_name", "datasource_id", "name", unique=True, where="NOT gone"),
    ),
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

#: one agent's skills, as agent-skills declares them (v001): the primary key (agent_id, skill_id), a unique name per
#: user and an index of the user's enabled ones; none of them holds the listing's recency order
_SKILLS = "yb1_hint_skills"

TableSchema(
    name=_SKILLS,
    primary_key=("agent_id", "skill_id"),
    columns=[
        Column("agent_id", UUID_TYPE),
        Column("skill_id", UUID_TYPE),
        Column("user_id", UUID_TYPE),
        Column("name", STRING_TYPE),
        Column("enabled", STRING_TYPE),
        Column("last_used_at", STRING_TYPE, nullable=True),
        Column("date_created", STRING_TYPE),
    ],
    indexes=(
        Index("yb1_skills_agent_user_name", "agent_id", "user_id", "name", unique=True),
        Index("yb1_skills_agent_user_enabled", "agent_id", "user_id", "enabled"),
    ),
    l1_eviction=TEST_L1_TABLE,
)

#: a group's grants, as agent-acl declares role_assignments: by group and scope, a namespace's (its id IS NOT NULL),
#: by scope type, and a namespace grant's natural key, unique among the namespace scope's rows
_GRANTS = "ai_hint_grants"

TableSchema(
    name=_GRANTS,
    primary_key=("row_scope", "grant_id"),
    columns=[
        Column("row_scope", STRING_TYPE),
        Column("grant_id", UUID_TYPE),
        Column("group_id", UUID_TYPE),
        Column("role_id", UUID_TYPE),
        Column("scope_type", STRING_TYPE),
        Column("scope_ns", UUID_TYPE, nullable=True),
        Column("managed_by", STRING_TYPE),
        Column("date_granted", STRING_TYPE),
    ],
    indexes=(
        Index("ai_grants_group_scope", "group_id", "scope_type"),
        Index("ai_grants_ns", "scope_ns", where="scope_ns IS NOT NULL"),
        Index("ai_grants_scope_type", "scope_type", "grant_id"),
        Index(
            "ai_grants_ns_key",
            "group_id",
            "role_id",
            "scope_ns",
            "managed_by",
            unique=True,
            where="scope_type = 'namespace'",
        ),
    ),
    l1_eviction=TEST_L1_TABLE,
)

#: the skills listing's order (agent-skills' _RECENT_FIRST_PAGE): an expression, then columns no index holds
_RECENT_FIRST = [
    Sort("last_used_or_never", descending=True, expression="COALESCE(t.\"last_used_at\", '-infinity'::timestamptz)"),
    Sort("date_created", descending=True),
    Sort("skill_id", cast="uuid", descending=True),
]

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
        assert present(serving_index(_CONV, {"agent_id": 1}, _BY_STATUS)).name == "o88_conv_agent_status"
        assert present(serving_index(_CONV, {"agent_id": 1, "status": "a"}, _BY_STATUS)).name == "o88_conv_agent_status"
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
        assert serving_index(_CONV, {"status": "a"}, []) is None
        # a read not pinning the partial index's text does not state its predicate
        for by in ({"agent_id": 1}, {"agent_id": 1, "status": "closed"}, {"agent_id": 1, "status": AnyOf(["open"])}):
            served = serving_index(_CONV, by, [Sort("name"), Sort("conversation_id")])
            assert served is None or served.name != "o88_conv_open_named", by
        assert serving_index(_CONV, {"agent_id": 1}, [Sort("name"), Sort("conversation_id")]) is None
        assert serving_index("o88_never_declared", {"agent_id": 1, "status": "a"}, []) is None

    def test_an_unordered_read_takes_the_index_binding_more_pins_than_the_primary_key(self) -> None:
        """Flips if an unordered read (no sort, or one sorting only pinned columns) pinning more of a declared index's
        leading columns than of the primary key's is not served by that index (the one binding most, then the
        narrower), or one the primary key binds as far, or a point read by the whole primary key, is handed a secondary
        index (coordinator ruling, #96)."""
        for order in ([], [Sort("status")]):
            served = serving_index(_CONV, {"agent_id": 1, "status": "a"}, order)
            assert served is not None and served.name == "o88_conv_agent_status", order
        held = serving_index(_CONV, {"agent_id": 1, "user_id": 2, "status": "a"}, [])
        assert held is not None and held.name == "o88_conv_agent_user_status"
        for by in ({"agent_id": 1}, {"agent_id": 1, "conversation_id": 2, "status": "a"}):
            index = serving_index(_CONV, by, [])
            assert index is not None and index.primary and index_hint(index) == "", by

    def test_a_partial_index_serves_only_a_read_pinning_its_predicate_s_text(self) -> None:
        """Flips if a partial index whose predicate is ``column = 'text'`` conjuncts does not serve a read pinning each
        to that text (ordered or not), or serves one pinning another text, a set, or nothing; or if a partial index
        with any other predicate (``name IS NOT NULL``) is ever named (3tears #98: identity's pending queue, unhinted on
        YugabyteDB, read 400 rows and sorted them for a page of 10)."""
        served = serving_index(_CONV, {"agent_id": 1, "status": "open"}, [Sort("name"), Sort("conversation_id")])
        assert served is not None and served.name == "o88_conv_open_named", served
        assert served.stated == frozenset({("status", "open")})
        for by in ({"agent_id": 1, "status": "closed"}, {"agent_id": 1, "status": AnyOf(["open"])}, {"agent_id": 1}):
            assert serving_index(_CONV, by, [Sort("name"), Sort("conversation_id")]) is None, by
        unordered = serving_index(_CONV, {"agent_id": 1, "user_id": 2, "status": "open", "name": "n"}, [])
        assert unordered is not None and unordered.name == "o88_conv_agent_user_status", unordered

    def test_a_partial_index_s_stated_column_counts_as_bound_for_an_unordered_read(self) -> None:
        """Flips if an unordered read pinning a conversation and its partial index's exact text (status = 'active') is
        not served by that index, which binds the conversation as the primary key does and the status besides
        (coordinator ruling, 3tears #100: YugabyteDB counted a conversation's 10 active schedules of 400 through the
        primary key, 6 of 6 fresh ANALYZEs)."""
        for order in ([], [Sort("status")]):
            served = serving_index(_SCHED, {"conversation_id": 1, "status": "active"}, order)
            assert served is not None and served.name == "a100_sched_conv_active", (order, served)

    def test_a_partial_index_is_never_named_for_a_read_not_pinning_its_exact_text(self) -> None:
        """Flips if the partial index is named for a read pinning another status, a set of statuses (even the one
        text), or no status: it holds only the active rows, so the read would miss every other."""
        for by in (
            {"conversation_id": 1, "status": "paused"},
            {"conversation_id": 1, "status": AnyOf(["active"])},
            {"conversation_id": 1, "status": None},
            {"conversation_id": 1},
        ):
            served = serving_index(_SCHED, by, [])
            assert served is not None and served.primary, (by, served)

    def test_a_read_sorted_by_an_expression_takes_the_index_binding_most_pins(self) -> None:
        """Flips if a read sorting by an expression (no plain index holds its order) is not served by the index binding
        most of its pins, as an unordered read is: YugabyteDB read a user's enabled skills, recency first, through the
        user's prefix of the unique name index, the disabled ones too, when the read named no index (D5, 2026-11-15).
        Where only the primary key binds the pins, none is named (the planner's own choice)."""
        served = serving_index(_SKILLS, {"agent_id": 1, "user_id": 2, "enabled": "t"}, _RECENT_FIRST)
        assert served is not None and served.name == "yb1_skills_agent_user_enabled", served
        assert index_hint(served) == "/*+ IndexOnlyScan(t yb1_skills_agent_user_enabled) */ "
        assert serving_index(_SKILLS, {"agent_id": 1}, _RECENT_FIRST) is None
        assert serving_index(_SKILLS, {"user_id": 2}, _RECENT_FIRST) is None

    def test_a_plain_order_no_declared_index_holds_names_none(self) -> None:
        """Flips if a read sorting by plain columns that no declared index holds after its pins is handed the index
        binding most of them: an index the declarations do not name may hold that order, and the hint would override
        it (the hub's provider model pages read v160's undeclared (provider_id, status, id) indexes; named the declared
        (provider_id, model_name) instead, YugabyteDB read 333 and 1,021 rows for a page of 21; YB1, 2026-11-16)."""
        assert serving_index(_SKILLS, {"agent_id": 1, "user_id": 2, "enabled": "t"}, _RECENT_FIRST[1:]) is None

    def test_a_declared_index_names_plain_identifiers(self) -> None:
        """Flips if an index a hint could not name unquoted, or one sorting a column it lacks, is accepted."""
        with pytest.raises(ValueError, match="plain identifiers"):
            DeclaredIndex('idx "x"', ("a",))
        with pytest.raises(ValueError, match="descending"):
            DeclaredIndex("idx", ("a",), frozenset({"b"}))


class TestASqlalchemyTableRegistersItsIndexes:
    def test_its_plain_b_tree_indexes_register_as_a_table_schema_s_do(self) -> None:
        """Flips if a SQLAlchemy ``Table``'s primary key, a plain index (columns, ``DESC`` written as text or as
        ``.desc()``) or a partial index whose predicate is ``column = 'text'`` is not registered as served, or one with
        another access method, an operator class, an expression or another predicate is (3tears #100: wake, skills and
        scheduled-jobs declare their tables so, and none of their indexes was ever named to YugabyteDB)."""
        table = sa.Table(
            "a100_sa_schedules",
            sa.MetaData(),
            sa.Column("conversation_id", sa.Uuid, primary_key=True),
            sa.Column("schedule_id", sa.Uuid, primary_key=True),
            sa.Column("status", sa.Text),
            sa.Column("name", sa.Text),
            sa.Column("date_created", sa.DateTime),
            info={L1_EVICTION_INFO: TEST_L1_TABLE},
        )
        sa.Index("a100_sa_active", table.c.conversation_id, postgresql_where=sa.text("status = 'active'"))
        sa.Index("a100_sa_created", table.c.conversation_id, sa.text("date_created DESC"), table.c.schedule_id.desc())
        # each of these would serve one conversation's rows by name, were it admitted
        named = (table.c.conversation_id, table.c.name)
        sa.Index("a100_sa_named", *named, postgresql_where=sa.text("name IS NOT NULL"))
        sa.Index("a100_sa_gin", *named, postgresql_using="gin")
        sa.Index("a100_sa_ops", *named, postgresql_ops={"name": "text_pattern_ops"})
        sa.Index("a100_sa_lower", table.c.conversation_id, sa.func.lower(table.c.name))
        assert len(table.indexes) == 6
        register_sa_table_indexes(table)
        active = serving_index("a100_sa_schedules", {"conversation_id": 1, "status": "active"}, [])
        assert active is not None and active.name == "a100_sa_active", active
        assert active.stated == frozenset({("status", "active")})
        created = serving_index(
            "a100_sa_schedules",
            {"conversation_id": 1},
            [Sort("date_created", descending=True), Sort("schedule_id", descending=True)],
        )
        assert created is not None and created.name == "a100_sa_created", created
        assert created.descending == frozenset({"date_created", "schedule_id"})
        point = serving_index("a100_sa_schedules", {"conversation_id": 1, "schedule_id": 2}, [])
        assert point is not None and point.primary, point
        for by, order in (
            ({"conversation_id": 1, "status": "paused"}, []),
            ({"conversation_id": 1}, [Sort("name")]),
        ):
            served = serving_index("a100_sa_schedules", by, order)
            assert served is None or served.primary, (by, served)
        assert ("status", "active") in stated_literals("a100_sa_schedules")


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
        """Flips if a read the primary key serves, ordered or not, asks the database which it is, or carries a hint."""
        reader = _yugabyte()
        await keyset_page(reader, _CONV, columns=("conversation_id",), by={"agent_id": 1}, order=_BY_ID, limit=8)
        await read_by(reader, _CONV, by={"agent_id": 1})
        await read_by(reader, _CONV, by={"agent_id": 1, "conversation_id": 2, "status": "a"})
        assert len(reader.asked) == 3 and not any("/*+" in sql or "version()" in sql for sql in reader.asked)

    async def test_a_read_by_stating_a_partial_index_names_it_on_yugabyte(self) -> None:
        """Flips if an ordered read_by pinning a partial index's text does not start with the hint naming it on
        YugabyteDB (its literal written as the predicate's), or one pinning another text names it."""
        reader = _yugabyte()
        await read_by(
            reader, _CONV, by={"agent_id": 1, "status": "open"}, order=[Sort("name"), Sort("conversation_id")]
        )
        await read_by(reader, _CONV, by={"agent_id": 1, "status": "x"}, order=[Sort("name"), Sort("conversation_id")])
        stated, other = [sql for sql in reader.asked if not sql.startswith("SELECT version()")]
        assert stated.startswith("/*+ IndexOnlyScan(t o88_conv_open_named) */ "), stated
        assert "t.\"status\" = 'open'" in stated, stated
        assert "/*+" not in other, other

    async def test_an_unordered_read_by_names_the_index_on_yugabyte_only(self) -> None:
        """Flips if an unordered read_by pinning the agent and a status does not start with the hint naming
        (agent_id, status, conversation_id) on YugabyteDB, or carries any hint on PostgreSQL."""
        for reader, hinted in ((_yugabyte(), True), (_postgres(), False)):
            await read_by(reader, _CONV, by={"agent_id": 1, "status": "a"})
            (read,) = [sql for sql in reader.asked if not sql.startswith("SELECT version()")]
            assert read.startswith("/*+ IndexOnlyScan(t o88_conv_agent_status) */ ") is hinted, read
            assert ("/*+" in read) is hinted, read


class TestACountNamesItsIndexOnYugabyte:
    async def test_a_count_by_value_stating_a_partial_index_names_it_on_yugabyte_only(self) -> None:
        """Flips if a count of one conversation's active schedules (the counted column bound to each value named) does
        not start with the hint naming the partial index on YugabyteDB, or carries any hint on PostgreSQL, or a count
        of another status is hinted to it."""
        for reader, hinted in ((_yugabyte(), True), (_postgres(), False)):
            await counts_by(
                reader, _SCHED, column="conversation_id", values=AnyOf([1], "uuid"), by={"status": "active"}
            )
            await counts_by(
                reader, _SCHED, column="conversation_id", values=AnyOf([1], "uuid"), by={"status": "paused"}
            )
            active, paused = [sql for sql in reader.asked if not sql.startswith("SELECT version()")]
            assert active.startswith("/*+ IndexOnlyScan(r a100_sched_conv_active) */ ") is hinted, active
            assert ("/*+" in active) is hinted, active
            assert "/*+" not in paused, paused

    async def test_a_paged_count_of_every_value_names_the_index_holding_its_order_on_yugabyte(self) -> None:
        """Flips if a count of every status within an agent (paged in the status's order) does not name
        (agent_id, status, conversation_id) on YugabyteDB."""
        reader = _yugabyte()
        await counts_by(reader, _CONV, column="status", by={"agent_id": 1}, nullable=False)
        (paged,) = [sql for sql in reader.asked if not sql.startswith("SELECT version()")]
        assert paged.startswith("/*+ IndexOnlyScan(r o88_conv_agent_status) */ "), paged


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


class TestACollatedOrGuardedIndexServesWhatItCan:
    def test_a_collated_index_serves_a_read_by_the_plain_columns_before_its_collated_one(self) -> None:
        """Flips if an index holding a column in another collation is skipped whole: an unordered read pinning the
        columns before that one is then handed a wider index of the same lead (YugabyteDB read every ref of a channel
        for a prefix, D7 2026-11-15); or if the collated column, or an order on it, is ever served by it."""
        served = serving_index(_REFS, {"agent_id": 1, "channel_type": "web"}, [])
        assert served is not None and served.name == "d7_refs_c", served
        assert served.columns == ("agent_id", "channel_type")
        # a ref pinned in the database's collation, newest first: the wide index, never the collated one
        ordered = serving_index(
            _REFS,
            {"agent_id": 1, "channel_type": "web", "conversation_ref": "r"},
            [Sort("date_created", descending=True)],
        )
        assert ordered is not None and ordered.name == "d7_refs_created", ordered
        assert serving_index(_REFS, {"agent_id": 1, "channel_type": "web"}, [Sort("conversation_ref")]) is None or (
            present(serving_index(_REFS, {"agent_id": 1, "channel_type": "web"}, [Sort("conversation_ref")])).name
            == "d7_refs_created"
        )

    def test_a_not_null_partial_index_serves_only_a_read_stating_that_guard(self) -> None:
        """Flips if a partial index whose predicate is ``alias IS NOT NULL`` is not served to an unordered read that
        states exactly that guard (a user merge's alias collisions, hinted on YugabyteDB to the user's every memory by
        date, D7 2026-11-15), or is served to one stating no guard, or a guard on another column."""
        by = {"agent_id": 1, "user_id": 2}
        guarded = serving_index(_ALIASES, by, [], guarded=frozenset({"alias"}))
        assert guarded is not None and guarded.name == "d7_mem_alias", guarded
        for guards in (frozenset(), frozenset({"date_created"})):
            served = serving_index(_ALIASES, by, [], guarded=guards)
            assert served is not None and served.name == "d7_mem_date", (guards, served)

    async def test_read_by_states_its_not_null_guard_to_the_hint(self) -> None:
        """Flips if read_by's ``where=`` guard ``t.alias IS NOT NULL`` does not name the partial index on YugabyteDB,
        or names it without that guard, or any hint is sent on PostgreSQL."""
        for reader, hinted in ((_yugabyte(), True), (_postgres(), False)):
            await read_by(reader, _ALIASES, by={"agent_id": 1, "user_id": 2}, where=[Fragment("t.alias IS NOT NULL")])
            await read_by(reader, _ALIASES, by={"agent_id": 1, "user_id": 2}, where=[Fragment("t.alias = $1", ("a",))])
            named, other = [sql for sql in reader.asked if not sql.startswith("SELECT version()")]
            assert named.startswith("/*+ IndexOnlyScan(t d7_mem_alias) */ ") is hinted, named
            assert "d7_mem_alias" not in other, other
            assert ("/*+" in other) is hinted, other


class TestASetOfAUniqueKeyIsBound:
    def test_a_set_of_names_reads_the_unique_name_index(self) -> None:
        """Flips if an unordered read binding a unique key's every column, a set of values among them, is not served by
        that key's index but by an index binding one pinned value (datasources' schema priming, name = ANY and status
        = 'active', hinted on YugabyteDB to the status index, D8 2026-11-15); or if a set binding only part of a
        unique key, or a set of a column no unique key holds, is served by it."""
        served = serving_index(_NAMED, {"name": AnyOf(["a", "b"]), "status": "active"}, [])
        assert served is not None and served.name == "d7_ds_name" and served.unique, served
        single = serving_index(_NAMED, {"name": "a", "status": "active"}, [])
        assert single is not None and single.name == "d7_ds_name", single
        other = serving_index(_NAMED, {"id": AnyOf([1]), "status": "active"}, [])
        assert other is not None and other.name == "d7_ds_status", other


class TestALiveOnlyIndexServesLiveReadsOnly:
    """A stamped table's unique index among live rows only (``WHERE NOT gone``, core ``make_unique_index_live_only``)
    holds no gone row. Every shared keyed read of a stamped L3 table answers live rows alone (``live_only`` adds
    ``NOT t."gone"``), so it states that predicate; a read of the table as stored (``with_gone``: the restore,
    quiet cleanup) does not, and named to it would miss every gone row.

    WHY (datasources' schema priming on YugabyteDB, 2026-11-26): with ``idx_datasources_name`` live-only, the predicate
    was no shape a read could state, so the name index was never named and the read was hinted to the status index
    again, every active datasource read and filtered (4,900 rows for 2 names), the D8 regression back."""

    def test_the_predicate_is_the_live_rows_guard(self) -> None:
        """Flips if ``NOT gone`` (alone, or beside an equality or a guard) is not read as the live-rows guard."""
        assert partial_predicate("NOT gone") == (frozenset(), frozenset(), frozenset({"gone"}))
        assert partial_predicate("(customer_id IS NOT NULL) AND (NOT gone)") == (
            frozenset(),
            frozenset({"customer_id"}),
            frozenset({"gone"}),
        )
        assert partial_predicate("status = 'a' AND NOT \"gone\"") == (
            frozenset({("status", "a")}),
            frozenset(),
            frozenset({"gone"}),
        )

    def test_served_to_a_live_read_never_to_a_with_gone_one(self) -> None:
        """Flips if a set of names read live is not served by the live-only unique name index, or a read stating no
        live guard (with_gone, quiet cleanup) is served by it."""
        by = {"name": AnyOf(["a", "b"]), "status": "active"}
        live = serving_index(_LIVE_NAMED, by, [], live=frozenset({"gone"}))
        assert live is not None and live.name == "lu_ds_name" and live.unique, live
        stored = serving_index(_LIVE_NAMED, by, [])
        assert stored is not None and stored.name == "lu_ds_status", stored
        assert serving_index(_LIVE_NAMED, {"name": "a"}, [], live=frozenset({"other"})) is None or (
            present(serving_index(_LIVE_NAMED, {"name": "a"}, [], live=frozenset({"other"}))).name != "lu_ds_name"
        )

    async def test_a_live_read_is_hinted_to_it_a_with_gone_read_never(self) -> None:
        """Flips if read_by (live) of a set of names is not hinted on YugabyteDB to the live-only name index, or a
        keyset page read with_gone names it, or any hint is sent on PostgreSQL."""
        for reader, hinted in ((_yugabyte(), True), (_postgres(), False)):
            await read_by(reader, _LIVE_NAMED, by={"name": AnyOf(["a", "b"]), "status": "active"})
            await keyset_page(
                reader,
                _LIVE_NAMED,
                columns=("id",),
                by={"name": AnyOf(["a", "b"]), "status": "active"},
                order=[Sort("id", cast="uuid")],
                limit=5,
                with_gone=True,
            )
            live, stored = [sql for sql in reader.asked if not sql.startswith("SELECT version()")]
            assert live.startswith("/*+ IndexOnlyScan(t lu_ds_name) */ ") is hinted, live
            assert 'NOT t."gone"' in live, live
            assert "lu_ds_name" not in stored and '"gone"' not in stored.split("WHERE", 1)[-1], stored


class TestALiveReadKeepsItsLiveOnlyIndexBesideAWholeOne:
    """A whole index beside a live-only one leading with the same column (hub v198: ``datasource_tables.datasource_id``
    and the rest, for the database's foreign-key check during cleanup) never takes a live read: the live-only index
    holds only the live rows it asks. It wins a tie of binding before the narrower index, and a same-shape pair is
    decided by that, never by the order a set registered them in; a read stating no live guard never sees it.

    WHY (2026-11-26): ranked by binding and width alone, a live read of a datasource's tables was hinted to the whole
    (datasource_id), every gone row read and filtered, and a same-shape whole/live pair went to whichever the set
    named first."""

    _LIVE = frozenset({"gone"})
    _WHOLE = DeclaredIndex(name="lt_whole_ds", columns=("ds",))
    _LIVE_SAME = DeclaredIndex(name="lt_live_ds", columns=("ds",), live=frozenset({"gone"}))
    _LIVE_WIDE = DeclaredIndex(name="lt_live_ds_name", columns=("ds", "name"), unique=True, live=frozenset({"gone"}))
    _KEY = DeclaredIndex(name="lt_pk", columns=("id",), primary=True)

    def _registered(self, table: str, *indexes: DeclaredIndex) -> str:
        """``table`` with ``indexes`` registered one call each, so the registry holds them in exactly that order."""
        for index in (self._KEY, *indexes):
            register_table_indexes(table, [index])
        return table

    def test_a_same_shape_pair_goes_to_the_live_only_index_in_either_order(self) -> None:
        """Flips if a live read pinning the lead is served by the whole index of the same shape, in either order the
        two were registered, or a read stating no live guard is served by the live-only one."""
        for table, indexes in (
            ("lt_hint_same_whole_first", (self._WHOLE, self._LIVE_SAME)),
            ("lt_hint_same_live_first", (self._LIVE_SAME, self._WHOLE)),
        ):
            self._registered(table, *indexes)
            live = serving_index(table, {"ds": 1}, [], live=self._LIVE)
            assert live is not None and live.name == "lt_live_ds", (table, live)
            stored = serving_index(table, {"ds": 1}, [])
            assert stored is not None and stored.name == "lt_whole_ds", (table, stored)

    def test_a_narrower_whole_index_never_takes_a_live_read(self) -> None:
        """Flips if a live read pinning only the lead is served by the narrower whole index rather than the live-only
        one, in either registration order, or a read stating no live guard (with_gone, quiet cleanup) by the live-only
        one, which would answer it without its gone rows."""
        for table, indexes in (
            ("lt_hint_wide_whole_first", (self._WHOLE, self._LIVE_WIDE)),
            ("lt_hint_wide_live_first", (self._LIVE_WIDE, self._WHOLE)),
        ):
            self._registered(table, *indexes)
            live = serving_index(table, {"ds": 1}, [], live=self._LIVE)
            assert live is not None and live.name == "lt_live_ds_name", (table, live)
            for guard in (frozenset(), frozenset({"other"})):
                stored = serving_index(table, {"ds": 1}, [], live=guard)
                assert stored is not None and stored.name == "lt_whole_ds", (table, guard, stored)

    def test_an_ordered_same_shape_pair_goes_to_the_live_only_index_in_either_order(self) -> None:
        """Flips if an ordered live read is served by the whole index of the live-only one's shape in either
        registration order, or an ordered read stating no live guard by the live-only one."""
        whole = DeclaredIndex(name="lt_whole_ds_name", columns=("ds", "name"))
        for table, indexes in (
            ("lt_hint_order_whole_first", (whole, self._LIVE_WIDE)),
            ("lt_hint_order_live_first", (self._LIVE_WIDE, whole)),
        ):
            self._registered(table, *indexes)
            live = serving_index(table, {"ds": 1}, [Sort("name")], live=self._LIVE)
            assert live is not None and live.name == "lt_live_ds_name", (table, live)
            stored = serving_index(table, {"ds": 1}, [Sort("name")])
            assert stored is not None and stored.name == "lt_whole_ds_name", (table, stored)

    async def test_read_by_of_a_datasources_tables_is_hinted_to_the_live_only_index(self) -> None:
        """Flips if read_by (live) of one datasource's tables is not hinted on YugabyteDB to the live-only index, the
        whole one named instead, or a keyset page read with_gone names the live-only one, or PostgreSQL gets a hint."""
        for reader, hinted in ((_yugabyte(), True), (_postgres(), False)):
            await read_by(reader, _FK_CHILD, by={"datasource_id": 1})
            await keyset_page(
                reader,
                _FK_CHILD,
                columns=("id",),
                by={"datasource_id": 1},
                order=[Sort("id", cast="uuid")],
                limit=5,
                with_gone=True,
            )
            live, stored = [sql for sql in reader.asked if not sql.startswith("SELECT version()")]
            assert live.startswith("/*+ IndexOnlyScan(t lt_tables_ds_name) */ ") is hinted, live
            assert ("/*+" in live) is hinted, live
            assert "lt_tables_ds_name" not in stored, stored


class TestAWholeUniqueKeyServesAnyOrder:
    """An ordered read binding every column of a unique index (each to a value or a set) finds at most one row a value
    through it: its order sorts that handful, so no index need hold it.

    WHY (agent-acl's grant_winner, oldest first by (date_granted, assignment_id)): no declared index held that order,
    so the read was left to the planner rather than named to the natural-key unique index that holds the one row."""

    _KEY = {"group_id": 1, "role_id": 2, "scope_ns": 3, "managed_by": "manual"}
    _OLDEST = [Sort("date_granted"), Sort("grant_id")]

    def test_an_ordered_read_binding_the_whole_key_takes_it(self) -> None:
        """Flips if an ordered read binding the whole natural key, its predicate stated, is not served by the unique
        index, or one with the row scope a set beside it."""
        for by in (
            {**self._KEY, "scope_type": "namespace"},
            {**self._KEY, "scope_type": "namespace", "row_scope": AnyOf(["platform", "customer"])},
        ):
            served = serving_index(_GRANTS, by, self._OLDEST)
            assert served is not None and served.name == "ai_grants_ns_key", (by, served)

    def test_not_for_a_null_pin_another_scope_or_part_of_the_key(self) -> None:
        """Flips if the unique index is named for a read pinning a key column to NULL (two NULLs are distinct: it binds
        no single row), stating another scope's text (it holds none of those rows), or binding part of the key."""
        for by in (
            {**self._KEY, "scope_ns": None, "scope_type": "namespace"},
            {**self._KEY, "scope_type": "all"},
            {"group_id": 1, "role_id": 2, "scope_ns": 3, "scope_type": "namespace"},
        ):
            served = serving_index(_GRANTS, by, self._OLDEST)
            assert served is None or served.name != "ai_grants_ns_key", (by, served)

    def test_a_live_only_unique_index_serves_an_ordered_live_read_never_a_with_gone_one(self) -> None:
        """Flips if the live-only unique name index is not named for an ordered live read of one name, or is named for
        one stating no live guard (with_gone, quiet cleanup), which it would answer without its gone rows."""
        live = serving_index(_LIVE_NAMED, {"name": "a"}, [Sort("id")], live=frozenset({"gone"}))
        assert live is not None and live.name == "lu_ds_name", live
        stored = serving_index(_LIVE_NAMED, {"name": "a"}, [Sort("id")])
        assert stored is None or stored.name != "lu_ds_name", stored


class TestASetOnALeadingColumnBindsIt:
    """An unordered read pinning an index's leading column to a set of values (``col = ANY(...)``) narrows that index
    as a value does, one seek a value, so the set counts as binding the column when the indexes are ranked.

    WHY (the hub's outside grants, 2026-11-26): an agent's grants on its namespaces, ``scope_namespace_id = ANY(...)``
    and ``scope_type = 'namespace'`` with the namespace index's ``IS NOT NULL`` stated, were hinted on YugabyteDB to
    (scope_type, assignment_id), every namespace grant read and filtered, rather than the namespace index; the hub
    read once a namespace to reach it."""

    _OUTSIDE = {"scope_ns": AnyOf([1, 2, 3]), "scope_type": "namespace"}
    _STATED = frozenset({"scope_ns"})

    def test_the_index_whose_lead_a_set_binds_wins_over_one_matching_a_constant(self) -> None:
        """Flips if the read of a set of namespaces, its partial index's guard stated (both partitions as a set too, or
        not), is served by the scope-type index, which matches only the constant scope type; or if the partial index is
        named for the read that does not state its guard, which it would answer without the other rows."""
        for by in (self._OUTSIDE, {**self._OUTSIDE, "row_scope": AnyOf(["platform", "customer"])}):
            served = serving_index(_GRANTS, by, [], guarded=self._STATED)
            assert served is not None and served.name == "ai_grants_ns", (by, served)
        unstated = serving_index(_GRANTS, self._OUTSIDE, [])
        assert unstated is not None and unstated.name == "ai_grants_scope_type", unstated

    def test_a_set_the_primary_key_leads_with_names_no_secondary_index(self) -> None:
        """Flips if a read binding by a set only what the primary key leads with too (a set of agents) is handed a
        secondary index, or an ordered read by a set takes the index holding the order after that column (a set of
        leads reads several runs of it, unordered across them), or the read of the namespaces is not hinted on
        YugabyteDB to the index whose lead its set binds."""
        assert serving_index(_CONV, {"agent_id": AnyOf([1, 2])}, []) is None
        assert serving_index(_CONV, {"agent_id": AnyOf([1, 2])}, _BY_STATUS) is None
        assert serving_index(_GRANTS, {"scope_ns": AnyOf([1, 2])}, [Sort("grant_id")], guarded=self._STATED) is None
        served = serving_index(_GRANTS, self._OUTSIDE, [], guarded=self._STATED)
        assert served is not None and served.name == "ai_grants_ns", served

    async def test_read_by_a_set_of_namespaces_names_the_namespace_index_on_yugabyte_only(self) -> None:
        """Flips if read_by of a set of namespaces with the guard stated is not hinted on YugabyteDB to the namespace
        index, or carries any hint on PostgreSQL."""
        for reader, hinted in ((_yugabyte(), True), (_postgres(), False)):
            await read_by(
                reader,
                _GRANTS,
                by={**self._OUTSIDE, "scope_ns": AnyOf([1, 2, 3], "uuid")},
                where=[Fragment('t."scope_ns" IS NOT NULL')],
            )
            (read,) = [sql for sql in reader.asked if not sql.startswith("SELECT version()")]
            assert read.startswith("/*+ IndexOnlyScan(t ai_grants_ns) */ ") is hinted, read
            assert ("/*+" in read) is hinted, read
