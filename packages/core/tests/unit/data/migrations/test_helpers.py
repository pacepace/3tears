"""
unit tests for ``threetears.core.data.migrations.helpers``.

every helper produces SQL via ``store.execute`` calls. these tests
assert that:

- each helper splits DDL and DML across SEPARATE ``execute`` calls
  (the v054-class footgun: yugabyte silently no-ops DML inside a DO
  block that also performs DDL)
- backfill ``UPDATE`` statements carry a replay-guard predicate so
  re-running is a clean no-op
- ``replace_check_constraint`` emits a ``pg_get_constraintdef`` compare
  so an interrupted run resumes idempotently
- ``replace_primary_key`` emits the FK-preservation dance: drop inbound
  FKs, drop old PK, add new composite PK, add ``UNIQUE`` on the
  preserved id column, recreate inbound FKs
- generic SQL types only -- no 3tears-specific imports leak through
- helpers accept an optional ``schema=`` kwarg that propagates to the
  emitted SQL as a schema-qualified prefix

every test uses :class:`FakeDataStore` so the suite stays under the
15-second enforcement budget.
"""

from __future__ import annotations

import pytest

from threetears.core.data.migrations.helpers import (
    InboundFk,
    add_check_constraint,
    add_column_with_backfill,
    add_index,
    add_partition_column,
    create_policy_if_not_exists,
    enable_row_level_security,
    replace_check_constraint,
    replace_primary_key,
)

from ._fake_store import FakeDataStore

__all__: list[str] = []


# ---------------------------------------------------------------------------
# add_column_with_backfill
# ---------------------------------------------------------------------------


class TestAddColumnWithBackfill:
    """``add_column_with_backfill`` must split DDL + DML across executes."""

    async def test_ddl_and_dml_emitted_as_separate_executes(self) -> None:
        """ADD COLUMN, UPDATE backfill, and CHECK go via distinct executes."""
        store = FakeDataStore()
        await add_column_with_backfill(
            store,
            table="t",
            column="c",
            column_type="VARCHAR(8)",
            default="'a'",
            not_null=True,
            backfill_value_sql="'b'",
            backfill_predicate="x IS NULL",
        )
        # at least 2 executes: ADD COLUMN + backfill UPDATE.
        assert len(store.executed) >= 2
        # no single execute mixes ALTER TABLE and UPDATE.
        for sql, _ in store.executed:
            upper = sql.upper()
            has_alter = "ALTER TABLE" in upper
            has_update = "UPDATE " in upper and "SET " in upper
            assert not (has_alter and has_update), f"DDL+DML mixing in single execute: {sql!r}"

    async def test_add_column_uses_if_not_exists(self) -> None:
        """ADD COLUMN clause includes ``IF NOT EXISTS`` for idempotency."""
        store = FakeDataStore()
        await add_column_with_backfill(
            store,
            table="t",
            column="c",
            column_type="VARCHAR(8)",
            default="'a'",
        )
        joined = "\n".join(sql for sql, _ in store.executed).upper()
        assert "ADD COLUMN IF NOT EXISTS" in joined

    async def test_backfill_includes_replay_guard(self) -> None:
        """backfill UPDATE carries the predicate plus replay-safe AND clause."""
        store = FakeDataStore()
        await add_column_with_backfill(
            store,
            table="t",
            column="c",
            column_type="VARCHAR(8)",
            default="'a'",
            backfill_value_sql="'b'",
            backfill_predicate="x IS NULL",
        )
        update_sqls = [sql for sql, _ in store.executed if "UPDATE " in sql.upper()]
        assert len(update_sqls) == 1
        update_sql = update_sqls[0]
        # custom predicate present.
        assert "x IS NULL" in update_sql
        # replay guard present (column = default form).
        assert "c = 'a'" in update_sql or "c = $1" in update_sql

    async def test_no_backfill_without_value_sql(self) -> None:
        """omitting ``backfill_value_sql`` skips the UPDATE entirely."""
        store = FakeDataStore()
        await add_column_with_backfill(
            store,
            table="t",
            column="c",
            column_type="INTEGER",
            default="0",
        )
        update_sqls = [sql for sql, _ in store.executed if "UPDATE " in sql.upper()]
        assert update_sqls == []

    async def test_schema_qualifies_target(self) -> None:
        """passing ``schema=`` qualifies every emitted statement."""
        store = FakeDataStore()
        await add_column_with_backfill(
            store,
            table="t",
            column="c",
            column_type="INTEGER",
            default="0",
            schema="myschema",
        )
        assert any("myschema.t" in sql for sql, _ in store.executed)


# ---------------------------------------------------------------------------
# add_check_constraint
# ---------------------------------------------------------------------------


class TestAddCheckConstraint:
    """``add_check_constraint`` is idempotent via ``information_schema``."""

    async def test_emits_information_schema_existence_probe(self) -> None:
        """idempotency probe queries ``pg_constraint``."""
        store = FakeDataStore()
        await add_check_constraint(
            store,
            table="t",
            constraint_name="t_x_ck",
            expression="x > 0",
        )
        joined = "\n".join(sql for sql, _ in store.executed)
        assert "pg_constraint" in joined or "information_schema" in joined
        assert "t_x_ck" in joined
        assert "x > 0" in joined

    async def test_single_execute_call(self) -> None:
        """check-constraint add is one execute (DDL only)."""
        store = FakeDataStore()
        await add_check_constraint(
            store,
            table="t",
            constraint_name="t_x_ck",
            expression="x > 0",
        )
        assert len(store.executed) == 1


# ---------------------------------------------------------------------------
# replace_check_constraint
# ---------------------------------------------------------------------------


class TestReplaceCheckConstraint:
    """``replace_check_constraint`` uses pg_get_constraintdef compare-then-swap."""

    async def test_emits_pg_get_constraintdef_compare(self) -> None:
        """compare-then-swap pattern surfaces in the SQL."""
        store = FakeDataStore()
        await replace_check_constraint(
            store,
            table="namespaces",
            constraint_name="namespaces_namespace_type_ck",
            new_expression=("namespace_type IN ('agent', 'shared', 'system')"),
        )
        joined = "\n".join(sql for sql, _ in store.executed)
        assert "pg_get_constraintdef" in joined
        assert "namespaces_namespace_type_ck" in joined
        assert "namespace_type IN ('agent', 'shared', 'system')" in joined

    async def test_single_execute_call(self) -> None:
        """replace-check is one execute (DDL only)."""
        store = FakeDataStore()
        await replace_check_constraint(
            store,
            table="t",
            constraint_name="t_x_ck",
            new_expression="x > 0",
        )
        assert len(store.executed) == 1

    async def test_drop_and_add_present(self) -> None:
        """generated DO block contains both DROP CONSTRAINT and ADD CONSTRAINT."""
        store = FakeDataStore()
        await replace_check_constraint(
            store,
            table="t",
            constraint_name="t_x_ck",
            new_expression="x > 0",
        )
        sql = store.executed[0][0]
        assert "DROP CONSTRAINT" in sql
        assert "ADD CONSTRAINT" in sql


# ---------------------------------------------------------------------------
# replace_primary_key
# ---------------------------------------------------------------------------


class TestReplacePrimaryKey:
    """``replace_primary_key`` encodes the v054 PK-swap dance."""

    async def test_basic_pk_swap_emits_drop_add_unique(self) -> None:
        """no inbound FKs: DROP old PK, ADD new PK, ADD UNIQUE on id column."""
        store = FakeDataStore()
        await replace_primary_key(
            store,
            table="api_keys",
            new_columns=("customer_id", "id"),
        )
        sql = store.executed[0][0]
        assert "DROP CONSTRAINT" in sql
        assert "PRIMARY KEY (customer_id, id)" in sql
        assert "UNIQUE (id)" in sql

    async def test_pk_swap_with_inbound_fks(self) -> None:
        """inbound FKs are dropped + recreated around the PK swap."""
        store = FakeDataStore()
        await replace_primary_key(
            store,
            table="agents",
            new_columns=("customer_id", "id"),
            inbound_fks=(
                InboundFk(
                    source_table="channel_configs",
                    constraint_name="channel_configs_agent_id_fkey",
                    source_column="agent_id",
                    on_delete="",
                ),
            ),
        )
        sql = store.executed[0][0]
        assert "ALTER TABLE channel_configs DROP CONSTRAINT IF EXISTS channel_configs_agent_id_fkey" in sql
        assert (
            "ALTER TABLE channel_configs ADD CONSTRAINT "
            "channel_configs_agent_id_fkey FOREIGN KEY (agent_id) "
            "REFERENCES agents(id)" in sql
        )

    async def test_on_delete_clause_preserved_verbatim(self) -> None:
        """ON DELETE CASCADE on inbound FKs round-trips into the recreate."""
        store = FakeDataStore()
        await replace_primary_key(
            store,
            table="datasources",
            new_columns=("customer_id", "id"),
            inbound_fks=(
                InboundFk(
                    source_table="datasource_tables",
                    constraint_name="datasource_tables_datasource_id_fkey",
                    source_column="datasource_id",
                    on_delete="ON DELETE CASCADE",
                ),
            ),
        )
        sql = store.executed[0][0]
        assert "ON DELETE CASCADE" in sql

    async def test_idempotency_probe_on_definition_not_just_name(self) -> None:
        """guard inspects ``pg_get_constraintdef`` (definition compare)."""
        store = FakeDataStore()
        await replace_primary_key(
            store,
            table="users",
            new_columns=("customer_id", "id"),
        )
        sql = store.executed[0][0]
        assert "pg_get_constraintdef" in sql

    async def test_single_execute_call(self) -> None:
        """PK swap is one execute (DDL only -- no DML in the DO block)."""
        store = FakeDataStore()
        await replace_primary_key(
            store,
            table="t",
            new_columns=("p", "id"),
        )
        assert len(store.executed) == 1


# ---------------------------------------------------------------------------
# add_partition_column
# ---------------------------------------------------------------------------


class TestAddPartitionColumn:
    """``add_partition_column`` composes column-with-backfill + check."""

    async def test_emits_column_backfill_and_check(self) -> None:
        """adds column, backfills, and adds CHECK constraint as separate executes."""
        store = FakeDataStore()
        await add_partition_column(
            store,
            table="namespaces",
            column="row_scope",
            column_type="VARCHAR(8)",
            default="'customer'",
            backfill_value_sql="'platform'",
            backfill_predicate="customer_id IS NULL",
            add_check_allowed_values=("platform", "customer"),
        )
        assert len(store.executed) >= 3
        joined = "\n".join(sql for sql, _ in store.executed).upper()
        assert "ADD COLUMN" in joined
        assert "UPDATE " in joined
        assert "CHECK" in joined

    async def test_no_ddl_dml_mixing(self) -> None:
        """no execute mixes ALTER TABLE and UPDATE simultaneously."""
        store = FakeDataStore()
        await add_partition_column(
            store,
            table="t",
            column="c",
            column_type="VARCHAR(8)",
            default="'a'",
            backfill_value_sql="'b'",
            backfill_predicate="x IS NULL",
            add_check_allowed_values=("a", "b"),
        )
        for sql, _ in store.executed:
            upper = sql.upper()
            has_ddl = "ALTER TABLE" in upper or "ADD CONSTRAINT" in upper
            has_dml = ("UPDATE " in upper and "SET " in upper) or "INSERT INTO" in upper
            # add_check is a single ALTER TABLE under a DO block; that
            # block does not perform any DML. the test is precisely
            # "no DML in any execute that contains DDL".
            assert not (has_ddl and has_dml), f"DDL+DML mixing in single execute: {sql!r}"


# ---------------------------------------------------------------------------
# add_index
# ---------------------------------------------------------------------------


class TestAddIndex:
    """``add_index`` is a thin ``CREATE INDEX IF NOT EXISTS`` wrapper."""

    async def test_emits_create_index_if_not_exists(self) -> None:
        """idempotency clause present."""
        store = FakeDataStore()
        await add_index(
            store,
            table="t",
            name="idx_t_c",
            columns=("c",),
        )
        sql = store.executed[0][0]
        assert "CREATE INDEX IF NOT EXISTS" in sql
        assert "idx_t_c" in sql
        assert "(c)" in sql

    async def test_unique_flag_widens_clause(self) -> None:
        """unique=True emits CREATE UNIQUE INDEX."""
        store = FakeDataStore()
        await add_index(
            store,
            table="t",
            name="ux_t_c",
            columns=("c",),
            unique=True,
        )
        sql = store.executed[0][0]
        assert "CREATE UNIQUE INDEX IF NOT EXISTS" in sql

    async def test_partial_index_carries_where(self) -> None:
        """``where=`` produces a partial index."""
        store = FakeDataStore()
        await add_index(
            store,
            table="t",
            name="idx_t_active",
            columns=("c",),
            where="status = 'active'",
        )
        sql = store.executed[0][0]
        assert "WHERE" in sql
        assert "status = 'active'" in sql


# ---------------------------------------------------------------------------
# generic-SQL discipline (no 3tears-specific imports)
# ---------------------------------------------------------------------------


class TestGenericSqlDiscipline:
    """helpers must reference no 3tears-specific tables in their generated SQL."""

    async def test_no_3tears_table_names_baked_in(self) -> None:
        """smoke test: passing a generic table name yields generic SQL."""
        store = FakeDataStore()
        await replace_check_constraint(
            store,
            table="my_app_widgets",
            constraint_name="my_app_widgets_color_ck",
            new_expression="color IN ('red', 'blue')",
        )
        joined = "\n".join(sql for sql, _ in store.executed)
        assert "namespaces" not in joined
        assert "audit_events" not in joined
        assert "my_app_widgets" in joined


# ---------------------------------------------------------------------------
# create_policy_if_not_exists (RLS)
# ---------------------------------------------------------------------------


class TestCreatePolicyIfNotExists:
    """``create_policy_if_not_exists`` is idempotent via a ``pg_policies`` probe."""

    async def test_emits_pg_policies_probe(self) -> None:
        """one DO block consults pg_policies and creates the policy when absent."""
        store = FakeDataStore()
        await create_policy_if_not_exists(
            store,
            table="memories",
            policy_name="memories_tenant",
            using="customer_id = current_setting('app.customer_id', true)::uuid",
        )
        assert len(store.executed) == 1  # single statement (DO block)
        sql = store.executed[0][0]
        assert "pg_policies" in sql
        assert "IF NOT EXISTS" in sql
        assert "CREATE POLICY memories_tenant ON memories" in sql
        assert "FOR ALL" in sql
        assert "USING (customer_id = current_setting('app.customer_id', true)::uuid)" in sql

    async def test_with_check_clause_for_write_policy(self) -> None:
        store = FakeDataStore()
        await create_policy_if_not_exists(
            store, table="t", policy_name="p", command="INSERT", using="true", check="customer_id = '1'"
        )
        sql = store.executed[0][0]
        assert "FOR INSERT" in sql
        assert "WITH CHECK (customer_id = '1')" in sql

    async def test_schema_qualified_and_probe_filtered(self) -> None:
        store = FakeDataStore()
        await create_policy_if_not_exists(store, table="t", policy_name="p", using="true", schema="agent_abc")
        sql = store.executed[0][0]
        assert "ON agent_abc.t" in sql
        assert "schemaname = 'agent_abc'" in sql

    async def test_to_role_and_restrictive(self) -> None:
        store = FakeDataStore()
        await create_policy_if_not_exists(
            store, table="t", policy_name="p", using="true", to_role="app_subject", permissive=False
        )
        sql = store.executed[0][0]
        assert "TO app_subject" in sql
        assert "AS RESTRICTIVE" in sql

    async def test_unknown_command_rejected(self) -> None:
        store = FakeDataStore()
        with pytest.raises(ValueError):
            await create_policy_if_not_exists(store, table="t", policy_name="p", using="true", command="GRANT")

    async def test_check_on_read_command_rejected(self) -> None:
        # WITH CHECK is invalid SQL for FOR SELECT/DELETE -> reject early, not at the DB.
        store = FakeDataStore()
        with pytest.raises(ValueError):
            await create_policy_if_not_exists(
                store, table="t", policy_name="p", using="true", command="SELECT", check="x"
            )

    async def test_no_product_table_names_baked_in(self) -> None:
        store = FakeDataStore()
        await create_policy_if_not_exists(store, table="my_widgets", policy_name="p", using="true")
        joined = "\n".join(sql for sql, _ in store.executed)
        assert "namespaces" not in joined
        assert "my_widgets" in joined


# ---------------------------------------------------------------------------
# enable_row_level_security
# ---------------------------------------------------------------------------


class TestEnableRowLevelSecurity:
    """``enable_row_level_security`` enables + FORCEs RLS as separate idempotent DDL."""

    async def test_enable_and_force_by_default(self) -> None:
        store = FakeDataStore()
        await enable_row_level_security(store, table="memories")
        sqls = [sql for sql, _ in store.executed]
        assert sqls == [
            "ALTER TABLE memories ENABLE ROW LEVEL SECURITY",
            "ALTER TABLE memories FORCE ROW LEVEL SECURITY",
        ]

    async def test_force_false_enables_only(self) -> None:
        # without FORCE the table owner bypasses policies -- only use force=False deliberately.
        store = FakeDataStore()
        await enable_row_level_security(store, table="t", force=False)
        sqls = [sql for sql, _ in store.executed]
        assert len(sqls) == 1
        assert "FORCE" not in sqls[0]
        assert "ENABLE ROW LEVEL SECURITY" in sqls[0]

    async def test_schema_qualified(self) -> None:
        store = FakeDataStore()
        await enable_row_level_security(store, table="t", schema="agent_abc")
        assert all("agent_abc.t" in sql for sql, _ in store.executed)
