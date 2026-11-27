"""integration: the acl and skills reads moved onto the keyed helpers read and write what their statements did.

Each conversion is held by the equivalence harness (:func:`threetears.core.testing.same_results.assert_same_results`):
every seed case (empty, one, many, NULLs, ties, a tenant boundary, a page boundary and one past it), the old statement
(written out here as it read) against the collection method as the module now makes it, on the session's migrated
schema (``conn`` on PostgreSQL, ``yugabyte_conn`` on YugabyteDB; each skips on the other engine).

- acl ``NamespaceCollection.list_ids_under_name`` (``NAMESPACE_IDS_UNDER_NAME_SQL`` -> ``read_by`` over both
  partitions, the node or its byte range as one guard), ``list_tool_namespaces_for_actor`` /
  ``list_skill_eligible_tool_namespaces`` (``TOOL_ELIGIBLE_NAMESPACES_SQL`` / ``SKILL_ELIGIBLE_NAMESPACES_SQL`` ->
  ``read_by`` with the flag a literal guard), ``GroupCollection.read_cascade``'s members (a raw ``keyset_pages`` head
  -> ``keyset_rows`` by the group).
- skills ``AgentSkillCollection.list_for_user`` (its interpolated ``ORDER BY ... LIMIT ... OFFSET 0`` -> a
  ``keyset_page`` first page, or ``read_nearest`` with a ``TextRank`` whose ties come in the recency order),
  ``AgentSkillInvocationCollection.list_for_skill`` (``OFFSET`` pages -> ``keyset_page`` pages by cursor, walked
  to the end against the whole ordered history) and ``record_outcome`` (one CTE statement -> one transaction: the
  use's row locked as read, then each table's row by its key), the tables after each compared.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.agent.acl.collections import GroupCollection, NamespaceCollection
from threetears.agent.skills.collections import AgentSkillCollection, AgentSkillInvocationCollection
from threetears.agent.skills.types import SkillOutcome
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import next_epoch
from threetears.core.testing.same_results import Seeded, assert_same_results
from threetears.interfaces.keyed.keys import declared_columns

pytestmark = pytest.mark.integration

_NS = uuid.UUID("5d0c2b6e-0d55-4c55-9a5e-0d5d0c2b6e55")
_BASE = datetime(2026, 11, 1, tzinfo=UTC)
#: the page each paged read asks (the harness's page boundary sits at it)
_PAGE = 5


@pytest.fixture(params=["conn", "yugabyte_conn"])
def db(request: pytest.FixtureRequest, db_container: str) -> Any:
    """the session's migrated schema on whichever engine it is (the other skips)."""
    # ``db_container`` is named so the session database is in this fixture's closure: the engine fixture comes
    # through ``getfixturevalue``, which pytest cannot see while collecting, and a run on YugabyteDB deselects a
    # test whose closure does not reach the session database (``tests/require_services.py``)
    return request.getfixturevalue(request.param)


class _OnePool:
    """the harness's connection as a pool: ``acquire`` yields the connection itself, so a method's transaction nests
    in the case's (a savepoint, rolled back with it)."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        yield self._conn


class _Stop(Exception):  # noqa: N818 - a control-flow signal carrying the rows, not an error
    """raised by :class:`_FirstRows` once it holds the rows of the first statement a method sends."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__("read")
        self.rows = rows


class _FirstRows:
    """a connection answering the first statement a method sends, then stopping the method there."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def fetch(self, sql: str, *args: Any) -> Any:
        raise _Stop([dict(row) for row in await self._conn.fetch(sql, *args)])


def _collection(db: Any, kind: type[Any]) -> Any:
    registry = CollectionRegistry()
    registry.configure(l3_pool=db)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return kind(registry=registry, config=config)


# --- acl -----------------------------------------------------------------------------------------------------------

#: names at, beneath, beside and above the node ``a.b`` (``-`` and ``.`` sort as ignorable under a linguistic
#: collation, ``/`` is the range's end, ``_`` a LIKE wildcard)
_NAMES = ("a.b", "a.b.c", "a.b.c.d", "a.bc", "a.b-c", "a.b/c", "a.b_c", "ab.c", "a.a.z", "a.c", "a.b.x", "a.b.x.y")


def _namespace(seed: Seeded) -> dict[str, Any]:
    # tenant 1's rows are customer rows of another customer (the read spans both partitions); every other row is a
    # tool, its flags turned by the index
    customer = seed.tenant == 1 or seed.index % 3 == 2
    return {
        "row_scope": "customer" if customer else "platform",
        "namespace_id": uuid.uuid5(_NS, f"ns-{seed.case}-{seed.index}"),
        "name": _NAMES[seed.tie % len(_NAMES)] + ("" if seed.tie < len(_NAMES) else f".n{seed.index}"),
        "namespace_type": "tool" if seed.index % 4 != 3 else ("agent" if customer else "model"),
        "customer_id": uuid.uuid5(_NS, f"customer-{seed.tenant}") if customer else None,
        "tool_eligible": not seed.null and seed.index % 2 == 0,
        "skill_eligible": seed.index % 3 == 0,
        "date_created": _BASE,
        "date_updated": _BASE,
        # namespaces is ROW_STAMPED: a deleted row stays, marked gone, and every read skips it
        "gone": seed.index % 5 == 4,
    }


def _every_column() -> str:
    """what a read of every column of the stamped ``namespaces`` selects: its declared columns, never its gone flag or
    restore bucket (``keys.default_projection``)."""
    declared = declared_columns("namespaces")
    assert declared is not None, "namespaces is no longer declared ROW_STAMPED"
    return ", ".join(f'"{column}"' for column in declared)


async def test_the_names_under_a_node_are_the_rows_the_statement_read(db: Any) -> None:
    """Flips if ``list_ids_under_name`` answers other namespaces than the node and its byte range did, in either
    partition."""

    async def helper(c: Any) -> list[dict[str, Any]]:
        ids = await _collection(c, NamespaceCollection).list_ids_under_name("a.b")
        return [{"namespace_id": found} for found in ids]

    await assert_same_results(
        db,
        table="namespaces",
        row=_namespace,
        old_sql=(
            "SELECT namespace_id FROM namespaces WHERE row_scope IN ('platform', 'customer')"
            ' AND (name = $1 OR (name COLLATE "C" >= $2 AND name COLLATE "C" < $3)) AND NOT gone'
        ),
        params=["a.b", "a.b.", "a.b/"],
        helper=helper,
        ordered=False,
    )


@pytest.mark.parametrize("flag", ["tool_eligible", "skill_eligible"])
async def test_the_flagged_tools_are_the_rows_the_statement_read(db: Any, flag: str) -> None:
    """Flips if a flagged-tool listing reads other namespaces than its written-out statement did: another type, the
    other flag, one partition."""

    async def helper(c: Any) -> list[dict[str, Any]]:
        namespaces = _collection(_FirstRows(c), NamespaceCollection)
        listing = (
            namespaces.list_tool_namespaces_for_actor
            if flag == "tool_eligible"
            else namespaces.list_skill_eligible_tool_namespaces
        )
        try:
            await listing(actor_user_id=uuid.uuid4(), actor_agent_id=uuid.uuid4(), cache=None)
        except _Stop as stopped:
            return stopped.rows
        raise AssertionError("the listing read nothing")

    await assert_same_results(
        db,
        table="namespaces",
        row=_namespace,
        old_sql=(
            f"SELECT {_every_column()} FROM namespaces WHERE row_scope IN ('platform', 'customer')"
            f" AND namespace_type = 'tool' AND {flag} = TRUE AND NOT gone"
        ),
        helper=helper,
        ordered=False,
    )


_GROUPS = (uuid.uuid5(_NS, "group-0"), uuid.uuid5(_NS, "group-1"))


@pytest.fixture
async def groups(db: Any) -> AsyncIterator[None]:
    """the two groups the memberships hang on (committed: a membership's foreign key names its group)."""
    # groups is ROW_STAMPED: each row carries its epoch, as every write stamps it
    await db.executemany(
        "INSERT INTO groups (row_scope, group_id, customer_id, name, epoch) VALUES ('platform', $1, NULL, $2, $3)",
        [(group, f"same-results-{group}", next_epoch()) for group in _GROUPS],
    )
    try:
        yield
    finally:
        await db.execute("DELETE FROM groups WHERE group_id = ANY($1::uuid[])", list(_GROUPS))


def _member(seed: Seeded) -> dict[str, Any]:
    return {
        "id": uuid.uuid5(_NS, f"member-{seed.case}-{seed.index}"),
        "group_id": _GROUPS[seed.tenant],
        "member_type": "agent" if seed.null else "user",
        "member_id": uuid.uuid5(_NS, f"who-{seed.tie}"),
    }


async def test_a_group_s_cascade_reads_the_members_its_pages_read(db: Any, groups: None) -> None:
    """Flips if ``read_cascade`` reads other members of the group than its key-ordered pages did, or in another
    order."""

    async def helper(c: Any) -> list[dict[str, Any]]:
        cascade = await _collection(c, GroupCollection).read_cascade(_GROUPS[0])
        return list(cascade.members)

    await assert_same_results(
        db,
        table="group_members",
        row=_member,
        old_sql="SELECT member_type, member_id, id FROM group_members WHERE group_id = $1 ORDER BY id",
        params=[_GROUPS[0]],
        helper=helper,
        order_by=["id"],
    )


# --- skills --------------------------------------------------------------------------------------------------------

_AGENT = uuid.uuid5(_NS, "agent")
_USERS = (uuid.uuid5(_NS, "user-0"), uuid.uuid5(_NS, "user-1"))
_SKILLS = (uuid.uuid5(_NS, "skill-0"), uuid.uuid5(_NS, "skill-1"))
_LISTED = ("skill_id", "name", "last_used_at", "date_created", "enabled")


def _skill(seed: Seeded) -> dict[str, Any]:
    # never used when NULL; recency by the tie (ties share it), creation by the tie too, so full ties fall to the key
    return {
        "agent_id": _AGENT,
        "skill_id": uuid.uuid5(_NS, f"skill-{seed.case}-{seed.index}"),
        "user_id": _USERS[seed.tenant],
        "name": f"s{seed.index}",
        "summary": "s",
        "body": "alpha beta" if seed.index % 4 else "alpha alpha beta",
        "enabled": seed.index % 5 != 4,
        "last_used_at": None if seed.null else _BASE + timedelta(hours=seed.tie),
        "date_created": _BASE + timedelta(minutes=seed.tie % 4),
    }


@pytest.mark.parametrize("enabled_only", [True, False])
async def test_a_user_s_skills_are_the_page_the_statement_read(db: Any, enabled_only: bool) -> None:
    """Flips if ``list_for_user`` (no query) reads other skills, or in another order, than its ``ORDER BY last_used_at
    DESC NULLS LAST, date_created DESC LIMIT n OFFSET 0`` did: never-used skills last, the user's alone."""

    async def helper(c: Any) -> list[dict[str, Any]]:
        listed = await _collection(c, AgentSkillCollection).list_for_user(
            _AGENT, _USERS[0], enabled_only=enabled_only, limit=_PAGE
        )
        return [{column: getattr(skill, column) for column in _LISTED} for skill in listed]

    enabled = " AND enabled = true" if enabled_only else ""
    await assert_same_results(
        db,
        table="agent_skills",
        row=_skill,
        old_sql=(
            f"SELECT {', '.join(_LISTED)} FROM agent_skills WHERE agent_id = $1 AND user_id = $2{enabled}"
            " ORDER BY last_used_at DESC NULLS LAST, date_created DESC LIMIT $3 OFFSET 0"
        ),
        params=[_AGENT, _USERS[0], _PAGE],
        helper=helper,
        columns=_LISTED,
        order_by=["last_used_at", "date_created"],
        limited=True,
        page_size=_PAGE,
    )


async def test_a_user_s_searched_skills_are_the_page_the_statement_ranked(db: Any) -> None:
    """Flips if ``list_for_user`` with a query reads other skills, or in another order, than its ``ORDER BY fts_rank
    DESC, last_used_at DESC NULLS LAST, date_created DESC`` did: the better match first (``alpha`` twice), and the
    rows tied on the rank in the recency order, never the engine's."""

    async def helper(c: Any) -> list[dict[str, Any]]:
        listed = await _collection(c, AgentSkillCollection).list_for_user(_AGENT, _USERS[0], query="alpha", limit=_PAGE)
        return [{column: getattr(skill, column) for column in (*_LISTED, "body")} for skill in listed]

    # the body stands in for the rank (two bodies, two ranks), so the comparison orders by it as the rank did
    await assert_same_results(
        db,
        table="agent_skills",
        row=_skill,
        old_sql=(
            f"SELECT {', '.join(_LISTED)}, body FROM agent_skills WHERE agent_id = $1 AND user_id = $2"
            " AND enabled = true AND search_vector @@ websearch_to_tsquery('english', $3)"
            " ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', $3)) DESC,"
            " last_used_at DESC NULLS LAST, date_created DESC LIMIT $4 OFFSET 0"
        ),
        params=[_AGENT, _USERS[0], "alpha", _PAGE],
        helper=helper,
        columns=(*_LISTED, "body"),
        order_by=["body", "last_used_at", "date_created"],
        limited=True,
        page_size=_PAGE,
    )


@pytest.fixture
async def skills(db: Any) -> AsyncIterator[None]:
    """the two skills the uses hang on (committed: a use's foreign key names its skill)."""
    await db.executemany(
        "INSERT INTO agent_skills (agent_id, skill_id, user_id, name, summary, body) VALUES ($1, $2, $3, $4, 's', 'b')",
        [(_AGENT, skill, _USERS[0], f"same-results-{skill}") for skill in _SKILLS],
    )
    try:
        yield
    finally:
        await db.execute("DELETE FROM agent_skills WHERE skill_id = ANY($1::uuid[])", list(_SKILLS))


def _use(seed: Seeded) -> dict[str, Any]:
    return {
        "agent_id": _AGENT,
        "invocation_id": uuid.uuid5(_NS, f"use-{seed.index}"),
        "skill_id": _SKILLS[seed.tenant],
        "user_id": _USERS[0],
        "conversation_id": _NS,
        "invocation_source": "invoke",
        "invoked_at": _BASE + timedelta(hours=seed.tie),
        "outcome": None if seed.null else ("failure" if seed.tie % 2 else "success"),
        "outcome_notes": None if seed.null else f"was {seed.index}",
    }


_USE_COLUMNS = ("invocation_id", "skill_id", "invoked_at", "outcome")


@pytest.mark.parametrize("outcome", [None, "success"])
async def test_a_skill_s_history_pages_are_the_rows_the_statement_ordered(
    db: Any, skills: None, outcome: str | None
) -> None:
    """Flips if ``list_for_skill``'s pages, walked by cursor to the end, read other uses of the skill (another skill's,
    another outcome's) or in another order than its ``ORDER BY invoked_at DESC``: none twice, none missed."""

    async def helper(c: Any) -> list[dict[str, Any]]:
        invocations = _collection(c, AgentSkillInvocationCollection)
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            page = await invocations.list_for_skill(
                _AGENT, _SKILLS[0], limit=_PAGE, cursor=cursor, outcome_filter=outcome
            )
            rows += [{column: getattr(use, column) for column in _USE_COLUMNS} for use in page.rows]
            if page.next_cursor is None:
                return rows
            cursor = page.next_cursor

    filtered = "" if outcome is None else " AND outcome = $3"
    await assert_same_results(
        db,
        table="agent_skill_invocations",
        row=_use,
        old_sql=(
            f"SELECT {', '.join(_USE_COLUMNS)} FROM agent_skill_invocations WHERE agent_id = $1 AND skill_id = $2"
            f"{filtered} ORDER BY invoked_at DESC"
        ),
        params=[_AGENT, _SKILLS[0]] if outcome is None else [_AGENT, _SKILLS[0], outcome],
        helper=helper,
        columns=_USE_COLUMNS,
        order_by=["invoked_at"],
        page_size=_PAGE,
    )


#: the one-statement outcome report ``record_outcome`` sent before it took one transaction
_OLD_RECORD_OUTCOME_SQL = """
WITH prior AS (
    SELECT outcome AS was, outcome_notes AS was_notes
      FROM agent_skill_invocations
     WHERE agent_id = $1 AND invocation_id = $2 AND skill_id = $3
       FOR UPDATE
), marked AS (
    UPDATE agent_skill_invocations AS i
       SET outcome = $4::text,
           outcome_source = $5::text,
           outcome_notes = COALESCE($6::text, prior.was_notes)
      FROM prior
     WHERE i.agent_id = $1 AND i.invocation_id = $2
    RETURNING prior.was, prior.was_notes
), counted AS (
    UPDATE agent_skills AS s
       SET success_count = s.success_count
               + (CASE WHEN $4::text = 'success' THEN 1 ELSE 0 END)
               - (CASE WHEN m.was = 'success' THEN 1 ELSE 0 END),
           failure_count = s.failure_count
               + (CASE WHEN $4::text = 'failure' THEN 1 ELSE 0 END)
               - (CASE WHEN m.was = 'failure' THEN 1 ELSE 0 END),
           last_failure_at = CASE WHEN $4::text = 'failure' THEN now() ELSE s.last_failure_at END,
           date_updated = now()
      FROM marked AS m
     WHERE s.agent_id = $1 AND s.skill_id = $3 AND m.was IS DISTINCT FROM $4::text
    RETURNING 1
)
SELECT m.was, m.was_notes, EXISTS (SELECT 1 FROM counted) AS counted FROM marked AS m
"""


@pytest.mark.parametrize(("outcome", "notes"), [("success", None), ("failure", "it broke"), ("success", "fine")])
async def test_an_outcome_report_writes_what_the_statement_wrote(
    db: Any, skills: None, outcome: SkillOutcome, notes: str | None
) -> None:
    """Flips if ``record_outcome`` writes other rows or values than its one CTE statement did: the use by its key and
    skill (another skill's use is not found), its notes kept when none are given, the skill's counts moved once and
    only when the outcome changes, its failure stamped on a failure; and answers other than it did."""
    use = uuid.uuid5(_NS, "use-1")

    async def helper(c: Any) -> list[dict[str, Any]]:
        registry = CollectionRegistry()
        registry.configure(l3_pool=_OnePool(c))
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        recorded = await AgentSkillInvocationCollection(registry=registry, config=config).record_outcome(
            _AGENT, use, skill_id=_SKILLS[0], outcome=outcome, source="agent_tool", notes=notes
        )
        if not recorded.found:
            return []
        return [{"was": recorded.previous, "was_notes": recorded.previous_notes, "counted": recorded.counted}]

    skills_of = ", ".join(f"'{skill}'" for skill in _SKILLS)
    await assert_same_results(
        db,
        table="agent_skill_invocations",
        row=_use,
        old_sql=_OLD_RECORD_OUTCOME_SQL,
        params=[_AGENT, use, _SKILLS[0], outcome, "agent_tool", notes],
        helper=helper,
        state_sql=(
            "SELECT 'use' AS kind, invocation_id::text AS id, outcome, outcome_source, outcome_notes,"
            " NULL::int AS success_count, NULL::int AS failure_count, NULL::bool AS failed_at"
            " FROM agent_skill_invocations"
            " UNION ALL SELECT 'skill', skill_id::text, NULL, NULL, NULL, success_count, failure_count,"
            f" last_failure_at IS NOT NULL FROM agent_skills WHERE skill_id IN ({skills_of})"
        ),
    )
