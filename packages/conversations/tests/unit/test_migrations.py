"""
unit tests for 3tears-conversations migration registration and DDL shape.

these tests mirror the pattern used by agent-workspace and agent-memory:
a :class:`_FakeDataStore` captures executed SQL so the tests can assert
statement shape + idempotent re-apply without touching a real database.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from threetears.conversations.migrations import (
    PACKAGE_NAME,
    add_conversation_language_column,
    add_conversation_search_vector,
    add_folder_referential_integrity,
    add_name_column,
    create_conversations_table,
    create_folders_and_conversation_folder_id,
    datetime_to_datetimetz,
    register,
)
from threetears.core.data.migrations import (
    DuplicateVersionError,
    MigrationRunner,
    MigrationScope,
)


# parity-exempt: narrow migration-capture stub — emulates only the execute/fetch subset the MigrationRunner calls, not a DataStore substitute.
class _FakeDataStore:
    """
    in-memory DataStore stub capturing every executed statement.

    :ivar executed: list of (sql, params) tuples in execution order
    :ptype executed: list[tuple[str, tuple[Any, ...]]]
    :ivar migrations_rows: emulated ``_schema_migrations`` rows
    :ptype migrations_rows: list[dict[str, Any]]
    :ivar migrations_table_created: whether the bookkeeping table has
        been materialized
    :ptype migrations_table_created: bool
    """

    def __init__(self) -> None:
        """
        initialize an empty in-memory store.
        """
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.migrations_rows: list[dict[str, Any]] = []
        self.migrations_table_created = False

    async def execute(self, sql: str, *params: Any) -> str:
        """
        record an execute call and emulate ``_schema_migrations`` writes.

        :param sql: SQL statement text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: synthetic status string (``EXECUTE`` or an insert tag)
        :rtype: str
        """
        self.executed.append((sql, params))
        normalized = " ".join(sql.split()).upper()
        result: str
        if "CREATE TABLE IF NOT EXISTS _SCHEMA_MIGRATIONS" in normalized:
            self.migrations_table_created = True
            result = "CREATE TABLE"
            return result
        if normalized.startswith("INSERT INTO _SCHEMA_MIGRATIONS"):
            self.migrations_rows.append(
                {
                    "version": params[0],
                    "package": params[1],
                    "description": params[2],
                }
            )
            result = "INSERT 0 1"
            return result
        result = "EXECUTE"
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        emulate the two ``_schema_migrations`` queries the runner issues.

        :param sql: SQL query text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        normalized = " ".join(sql.split()).upper()
        result: list[dict[str, Any]]
        if "SELECT VERSION, PACKAGE FROM _SCHEMA_MIGRATIONS" in normalized:
            result = [{"version": row["version"], "package": row["package"]} for row in self.migrations_rows]
            return result
        if "COALESCE(MAX(VERSION)" in normalized:
            max_version = max((row["version"] for row in self.migrations_rows), default=0)
            result = [{"max_version": max_version}]
            return result
        result = []
        return result


def _joined_executed_sql(store: _FakeDataStore) -> str:
    """
    join every executed statement into a single normalized string.

    :param store: fake data store with execution history
    :ptype store: _FakeDataStore
    :return: joined, whitespace-normalized statements
    :rtype: str
    """
    return "\n".join(" ".join(sql.split()) for sql, _params in store.executed)


class TestRegisterConversationsMigrations:
    """tests for the register factory and apply flow."""

    async def test_register_returns_agent_scoped_package(self) -> None:
        """register produces a PackageMigrations scoped to AGENT."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.name == PACKAGE_NAME
        assert pkg.scope == MigrationScope.AGENT

    async def test_register_has_no_depends_on_edges(self) -> None:
        """conversations is the root of the agent dependency graph."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.depends_on == ()

    async def test_register_populates_versions_one_through_nine(self) -> None:
        """register wires v001 (create), v002 (message_count), v003
        (name), v004 (datetimetz), v005 (search_vector + trigger),
        v006 (language column + trigger update), v007 (rename id
        -> conversation_id), v008 (folders table + conversation
        folder_id), v009 (folder referential integrity: folder_id
        unique + conversation->folder FK ON DELETE SET NULL)."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert set(pkg.versions.keys()) == {1, 2, 3, 4, 5, 6, 7, 8, 9}

    async def test_apply_runs_nine_versions_then_idempotent(self) -> None:
        """apply records v1..v9 and re-running is a no-op."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        first_count = await runner.apply_for_agent_schema(store)
        assert first_count == 9
        assert store.migrations_table_created is True
        assert [row["version"] for row in store.migrations_rows] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        second_count = await runner.apply_for_agent_schema(store)
        assert second_count == 0

    async def test_apply_emits_name_column_add(self) -> None:
        """v003 emits an ADD COLUMN IF NOT EXISTS for ``name``."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert re.search(
            r"ALTER TABLE conversations\s+ADD COLUMN IF NOT EXISTS name TEXT",
            joined,
            re.IGNORECASE,
        )

    async def test_apply_emits_conversations_create_statement(self) -> None:
        """the CREATE TABLE statement carries every column and type."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert re.search(r"CREATE TABLE IF NOT EXISTS conversations", joined)
        assert re.search(r"agent_id UUID NOT NULL", joined)
        assert re.search(r"id UUID NOT NULL", joined)
        assert "PRIMARY KEY (agent_id, id)" in joined
        assert re.search(r"customer_id UUID NOT NULL", joined)
        assert re.search(r"user_id UUID NOT NULL", joined)
        assert "channel_type VARCHAR(50) NOT NULL" in joined
        assert "conversation_ref VARCHAR(500)" in joined
        assert "status VARCHAR(20) NOT NULL" in joined
        assert "summary TEXT" in joined
        assert re.search(r"date_created TIMESTAMP NOT NULL", joined)
        assert re.search(r"date_updated TIMESTAMP NOT NULL", joined)
        assert "date_last_message TIMESTAMP" in joined
        assert "metadata JSONB" in joined

    async def test_apply_emits_three_indexes(self) -> None:
        """lookup indexes on user, customer, and status are emitted."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_conv_user "
            r"ON conversations \(user_id, date_created\)",
            joined,
        )
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_conv_customer "
            r"ON conversations \(customer_id, date_created\)",
            joined,
        )
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_conv_status "
            r"ON conversations \(status\)",
            joined,
        )

    async def test_apply_does_not_qualify_with_schema_name(self) -> None:
        """
        DDL statements stay unqualified so ``search_path`` governs.

        protects against accidental hard-coded ``agent_<32hex>.`` prefixes
        that would break the L3 broker contract.
        """
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert not re.search(r"agent_[0-9a-f]{32}\.", joined)


class TestDirectMigrationFunction:
    """tests exercising create_conversations_table directly."""

    async def test_direct_call_issues_four_statements(self) -> None:
        """one CREATE TABLE plus three CREATE INDEX statements."""
        store = _FakeDataStore()
        await create_conversations_table(store)  # type: ignore[arg-type]
        assert len(store.executed) == 4

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch ``_schema_migrations``."""
        store = _FakeDataStore()
        await create_conversations_table(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestDatetimeToDatetimetzMigration:
    """
    tests for v004: TIMESTAMP -> TIMESTAMPTZ promotion of every
    datetime column on ``conversations``.

    collections-task-05 requires every per-column ALTER to appear as a
    literal SQL string (not a templated DO block iterating a list) so
    the column-type-alignment AST walker in
    ``packages/core/tests/enforcement/test_column_type_alignment.py``
    can match each ``(table, column) -> TIMESTAMPTZ`` pair against its
    ``Column(..., DATETIMETZ_TYPE, ...)`` declaration in
    ``collection.py``. these tests pin that pattern so a future
    refactor cannot regress it.
    """

    async def test_direct_call_issues_three_per_column_alters(self) -> None:
        """one DO block per (table, column) pair: 3 statements."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        assert len(store.executed) == 3

    async def test_direct_call_targets_every_datetime_column(self) -> None:
        """every datetime column on conversations has its own ALTER."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "ALTER TABLE conversations ALTER COLUMN date_created TYPE TIMESTAMPTZ" in joined
        assert "ALTER TABLE conversations ALTER COLUMN date_updated TYPE TIMESTAMPTZ" in joined
        assert "ALTER TABLE conversations ALTER COLUMN date_last_message TYPE TIMESTAMPTZ" in joined

    async def test_direct_call_uses_at_time_zone_utc(self) -> None:
        """every ALTER asserts UTC semantics on the bare TIMESTAMP cell."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "USING date_created AT TIME ZONE 'UTC'" in joined
        assert "USING date_updated AT TIME ZONE 'UTC'" in joined
        assert "USING date_last_message AT TIME ZONE 'UTC'" in joined

    async def test_direct_call_is_guarded_by_information_schema(self) -> None:
        """each ALTER lives inside a DO block that probes data_type."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert joined.count("information_schema.columns") == 3
        assert joined.count("'timestamp without time zone'") == 3

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch ``_schema_migrations``."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestAddConversationSearchVectorMigration:
    """tests for v005: search_vector tsvector + trigger + GIN index for
    postgres FTS on conversation display titles. mirrors the metallm
    alembic-057 conversation-side DDL shape, lifted to 3tears so other
    consumers don't reinvent the column + trigger pair locally.
    """

    async def test_direct_call_issues_six_statements(self) -> None:
        """one ALTER + one CREATE INDEX + one CREATE FUNCTION + one
        DROP TRIGGER + one CREATE TRIGGER + one backfill UPDATE."""
        store = _FakeDataStore()
        await add_conversation_search_vector(store)  # type: ignore[arg-type]
        assert len(store.executed) == 6

    async def test_direct_call_adds_search_vector_column(self) -> None:
        """ADD COLUMN IF NOT EXISTS search_vector tsvector."""
        store = _FakeDataStore()
        await add_conversation_search_vector(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert re.search(
            r"ALTER TABLE conversations\s+ADD COLUMN IF NOT EXISTS search_vector tsvector",
            joined,
            re.IGNORECASE,
        )

    async def test_direct_call_creates_gin_index(self) -> None:
        """GIN index on the new tsvector column for FTS lookups."""
        store = _FakeDataStore()
        await add_conversation_search_vector(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_conversations_search_vector "
            r"ON conversations USING gin\(search_vector\)",
            joined,
        )

    async def test_direct_call_uses_drop_create_trigger_pattern(self) -> None:
        """postgres has no ``CREATE TRIGGER IF NOT EXISTS``; the
        migration uses DROP-then-CREATE for replay safety."""
        store = _FakeDataStore()
        await add_conversation_search_vector(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "DROP TRIGGER IF EXISTS trg_conversations_search_vector ON conversations" in joined
        assert "CREATE TRIGGER trg_conversations_search_vector" in joined
        # The trigger fires on INSERT or UPDATE OF name -- pinning the
        # column dependency so a future schema change that renames
        # ``name`` breaks the build instead of silently breaking FTS.
        assert "BEFORE INSERT OR UPDATE OF name ON conversations" in joined

    async def test_direct_call_uses_setweight_a_on_name(self) -> None:
        """the trigger weights ``name`` at 'A' so future multi-source
        FTS (e.g. summary at 'B') can layer in without changing the
        read query shape."""
        store = _FakeDataStore()
        await add_conversation_search_vector(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "setweight(to_tsvector('english', coalesce(NEW.name, '')), 'A')" in joined

    async def test_direct_call_backfill_is_idempotent(self) -> None:
        """backfill UPDATE carries a replay guard so a re-run on a
        fully-populated table is a clean no-op."""
        store = _FakeDataStore()
        await add_conversation_search_vector(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "UPDATE conversations SET search_vector" in joined
        assert "WHERE search_vector IS NULL" in joined

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch ``_schema_migrations``."""
        store = _FakeDataStore()
        await add_conversation_search_vector(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestAddConversationLanguageColumnMigration:
    """tests for v006: per-row language column + trigger update.

    Future polyglot consumers set conversations.language to whatever
    pg_ts_config supports (``simple``, ``spanish``, ``french``, ...)
    and the trigger rebuilds the search_vector with that tokenizer.
    """

    async def test_direct_call_issues_four_statements(self) -> None:
        """ADD COLUMN + CREATE FUNCTION + DROP TRIGGER + CREATE TRIGGER."""
        store = _FakeDataStore()
        await add_conversation_language_column(store)  # type: ignore[arg-type]
        assert len(store.executed) == 4

    async def test_direct_call_adds_language_column_with_english_default(self) -> None:
        store = _FakeDataStore()
        await add_conversation_language_column(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert re.search(
            r"ALTER TABLE conversations\s+ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'english'",
            joined,
            re.IGNORECASE,
        )

    async def test_direct_call_trigger_reads_new_language(self) -> None:
        """Trigger function uses ``NEW.language`` (with COALESCE to
        'english' as defensive fallback) instead of hard-coding
        'english'. This is the load-bearing change behind v0.7.0
        review item #7."""
        store = _FakeDataStore()
        await add_conversation_language_column(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "COALESCE(NEW.language, 'english')" in joined
        # The function MUST use the per-row config now, not the bare
        # 'english' literal of v005.
        assert "to_tsvector(cfg, coalesce(NEW.name, ''))" in joined

    async def test_direct_call_trigger_fires_on_language_updates(self) -> None:
        """Updating ``language`` on an existing row must re-tokenize
        the search_vector. Pin the trigger's UPDATE OF list."""
        store = _FakeDataStore()
        await add_conversation_language_column(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "BEFORE INSERT OR UPDATE OF name, language ON conversations" in joined

    async def test_direct_call_uses_drop_create_trigger_pattern(self) -> None:
        store = _FakeDataStore()
        await add_conversation_language_column(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "DROP TRIGGER IF EXISTS trg_conversations_search_vector ON conversations" in joined
        assert "CREATE TRIGGER trg_conversations_search_vector" in joined

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        store = _FakeDataStore()
        await add_conversation_language_column(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestCreateFoldersAndConversationFolderIdMigration:
    """tests for v008: create the app-agnostic ``folders`` table plus
    the mutable ``conversations.folder_id`` FK column.

    a folder is a per-owner named container grouping conversations,
    lifted from metallm so multiple apps reuse one canonical entity.
    app-specific presentation lives in ``metadata`` so the canonical
    shape stays column-stable.
    """

    async def test_direct_call_issues_four_statements(self) -> None:
        """CREATE TABLE + CREATE UNIQUE INDEX + CREATE INDEX + ALTER."""
        store = _FakeDataStore()
        await create_folders_and_conversation_folder_id(store)  # type: ignore[arg-type]
        assert len(store.executed) == 4

    async def test_direct_call_creates_folders_table(self) -> None:
        """CREATE TABLE IF NOT EXISTS folders with composite PK + scope columns."""
        store = _FakeDataStore()
        await create_folders_and_conversation_folder_id(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert re.search(r"CREATE TABLE IF NOT EXISTS folders", joined)
        assert re.search(r"agent_id UUID NOT NULL", joined)
        assert re.search(r"folder_id UUID NOT NULL", joined)
        assert re.search(r"customer_id UUID NOT NULL", joined)
        assert re.search(r"user_id UUID NOT NULL", joined)
        assert "name TEXT NOT NULL" in joined
        assert "metadata JSONB" in joined
        assert "PRIMARY KEY (agent_id, folder_id)" in joined

    async def test_direct_call_declares_timestamptz_columns(self) -> None:
        """fresh table declares date columns as TIMESTAMPTZ so the
        column-type-alignment enforcement matches the DATETIMETZ_TYPE
        Column declarations on FolderCollection."""
        store = _FakeDataStore()
        await create_folders_and_conversation_folder_id(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "date_created TIMESTAMPTZ NOT NULL" in joined
        assert "date_updated TIMESTAMPTZ NOT NULL" in joined

    async def test_direct_call_enforces_unique_agent_user_name(self) -> None:
        """UNIQUE(agent_id, user_id, name) -- folders are scoped per user,
        so a folder name is unique within one owner. Expressed as a
        unique index for ``IF NOT EXISTS`` idempotency."""
        store = _FakeDataStore()
        await create_folders_and_conversation_folder_id(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert re.search(
            r"CREATE UNIQUE INDEX IF NOT EXISTS uq_folders_agent_user_name "
            r"ON folders \(agent_id, user_id, name\)",
            joined,
        )

    async def test_direct_call_creates_user_lookup_index(self) -> None:
        """btree index on (agent_id, user_id) backing find_by_user."""
        store = _FakeDataStore()
        await create_folders_and_conversation_folder_id(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_folders_user ON folders \(agent_id, user_id\)",
            joined,
        )

    async def test_direct_call_adds_conversation_folder_id_column(self) -> None:
        """ALTER conversations ADD COLUMN IF NOT EXISTS folder_id UUID
        (mutable, nullable -- conversations start unfiled and move
        between folders)."""
        store = _FakeDataStore()
        await create_folders_and_conversation_folder_id(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert re.search(
            r"ALTER TABLE conversations\s+ADD COLUMN IF NOT EXISTS folder_id UUID",
            joined,
            re.IGNORECASE,
        )

    async def test_direct_call_does_not_qualify_with_schema_name(self) -> None:
        """DDL stays unqualified so search_path governs the target schema."""
        store = _FakeDataStore()
        await create_folders_and_conversation_folder_id(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert not re.search(r"agent_[0-9a-f]{32}\.", joined)

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch ``_schema_migrations``."""
        store = _FakeDataStore()
        await create_folders_and_conversation_folder_id(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestAddFolderReferentialIntegrityMigration:
    """tests for v009: complete the folder<->conversation referential
    integrity v008 left as a bare nullable column -- a single-column
    UNIQUE on ``folders.folder_id`` (a valid FK target) plus the
    ``conversations.folder_id -> folders.folder_id ON DELETE SET NULL``
    FK so deleting a folder auto-unfiles its conversations.
    """

    async def test_direct_call_issues_two_statements(self) -> None:
        """one CREATE UNIQUE INDEX + one guarded ADD CONSTRAINT DO block."""
        store = _FakeDataStore()
        await add_folder_referential_integrity(store)  # type: ignore[arg-type]
        assert len(store.executed) == 2

    async def test_direct_call_creates_single_column_folder_id_unique(self) -> None:
        """a standalone single-column UNIQUE on folder_id so a
        single-column FK can target it (the composite PK cannot be)."""
        store = _FakeDataStore()
        await add_folder_referential_integrity(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert re.search(
            r"CREATE UNIQUE INDEX IF NOT EXISTS uq_folders_folder_id ON folders \(folder_id\)",
            joined,
        )

    async def test_direct_call_adds_on_delete_set_null_fk(self) -> None:
        """the FK references folders(folder_id) with ON DELETE SET NULL
        so deleting a folder nulls referencing conversations' folder_id
        rather than orphaning them."""
        store = _FakeDataStore()
        await add_folder_referential_integrity(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "ADD CONSTRAINT conversations_folder_id_fkey" in joined
        assert "FOREIGN KEY (folder_id)" in joined
        assert "REFERENCES folders (folder_id)" in joined
        assert "ON DELETE SET NULL" in joined

    async def test_direct_call_guards_fk_with_pg_constraint_probe(self) -> None:
        """ADD CONSTRAINT has no IF NOT EXISTS form; the FK is guarded by
        a pg_constraint existence probe scoped to current_schema() so
        replays no-op and a multi-schema test host stays isolated."""
        store = _FakeDataStore()
        await add_folder_referential_integrity(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "pg_constraint" in joined
        assert "conname = 'conversations_folder_id_fkey'" in joined
        assert "current_schema()::regnamespace" in joined

    async def test_direct_call_does_not_qualify_with_schema_name(self) -> None:
        """DDL stays unqualified so search_path governs the target schema."""
        store = _FakeDataStore()
        await add_folder_referential_integrity(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert not re.search(r"agent_[0-9a-f]{32}\.", joined)

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch ``_schema_migrations``."""
        store = _FakeDataStore()
        await add_folder_referential_integrity(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestDuplicateVersionGuard:
    """tests confirming the runner rejects duplicate version registration."""

    async def test_duplicate_version_registration_raises(self) -> None:
        """registering a second callable at v1 raises DuplicateVersionError."""
        runner = MigrationRunner()
        pkg = register(runner)
        with pytest.raises(DuplicateVersionError):
            pkg.version(1)(create_conversations_table)
