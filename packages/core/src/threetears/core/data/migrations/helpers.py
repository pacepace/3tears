"""
yugabyte-safe migration helpers as free async functions.

every helper splits DDL and DML across separate ``store.execute`` calls
so yugabyte's auto-committing DDL semantics never silently no-op a
``UPDATE`` that runs against a stale schema snapshot. v054's hand-rolled
``_composite_pk_swap_sql`` / ``_composite_pk_swap_with_drop_fks_sql``
are the canonical reference; this module promotes those into reusable
primitives any 3tears app's migrations can consume.

design points:

- helpers are FREE FUNCTIONS, not methods on :class:`DataStore`. data
  layer / migration layer separation keeps the runtime surface lean
  and lets aibots / metallm / future apps share the same primitives.
- signatures use generic SQL types and reference NO aibots-specific
  table names. nothing in this module knows about ``namespaces`` /
  ``audit_events`` / ``memories``.
- every helper is :func:`@traced <threetears.observe.tracing.traced>`
  and emits a single ``log.info`` describing the operation so the
  migration log is auditable.
- backfill ``UPDATE`` statements include a replay-guard predicate so
  re-running a migration is a clean no-op even after partial failure.
- DDL idempotency uses ``IF NOT EXISTS`` clauses where the engine
  supports them (ADD COLUMN, CREATE INDEX) and ``pg_get_constraintdef``
  compare-then-swap for CHECK / PRIMARY KEY constraints whose names
  alone do not encode the desired definition.

the v054-class footgun this module forecloses: any ``DO`` block that
mixes ``ALTER TABLE`` and ``UPDATE`` silently loses the UPDATE on
yugabyte because the ALTER auto-commits before the UPDATE runs. every
helper guarantees the split at construction time so a migration author
cannot reintroduce the bug by accident.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from threetears.observe import get_logger, traced

__all__ = [
    "InboundFk",
    "MigrationStore",
    "add_check_constraint",
    "add_column_with_backfill",
    "add_index",
    "add_partition_column",
    "create_policy_if_not_exists",
    "enable_row_level_security",
    "replace_check_constraint",
    "replace_primary_key",
]

log = get_logger(__name__)


@runtime_checkable
class MigrationStore(Protocol):
    """
    minimal store surface every migration helper consumes.

    matches the :class:`~threetears.core.data.store.DataStore` shape used
    by :class:`~threetears.core.data.migrations.runner.MigrationRunner`,
    but typed as a structural protocol so test fakes (and any
    third-party migration runner) can be passed without inheriting from
    DataStore.
    """

    async def execute(self, sql: str, *params: object) -> str:
        """
        execute ``sql`` against the bound schema.

        :param sql: SQL statement text
        :ptype sql: str
        :param params: positional parameter values
        :ptype params: object
        :return: status string from the underlying engine
        :rtype: str
        """
        ...


@dataclass(frozen=True)
class InboundFk:
    """
    descriptor for an inbound foreign key that depends on the table whose
    PK is being swapped.

    postgres refuses to drop a primary key while a foreign key on a
    sibling table depends on it; ``replace_primary_key`` drops every
    inbound FK before swapping the PK and recreates each one against
    the post-swap ``UNIQUE`` index. ``on_delete`` is preserved verbatim
    because changing cascade semantics during a PK swap would silently
    alter delete behaviour.

    :param source_table: name of table holding the FK
    :ptype source_table: str
    :param constraint_name: FK constraint name (the
        ``ALTER TABLE ... DROP CONSTRAINT`` target)
    :ptype constraint_name: str
    :param source_column: column on ``source_table`` carrying the FK
    :ptype source_column: str
    :param on_delete: full ``ON DELETE`` clause to reattach (e.g.
        ``"ON DELETE CASCADE"`` / ``"ON DELETE SET NULL"`` / ``""``);
        empty string preserves engine-default ``NO ACTION`` semantics
    :ptype on_delete: str
    """

    source_table: str
    constraint_name: str
    source_column: str
    on_delete: str = ""


def _qualify(table: str, schema: str | None) -> str:
    """
    return ``schema.table`` when schema is supplied, else bare ``table``.

    :param table: bare table name
    :ptype table: str
    :param schema: optional schema name
    :ptype schema: str | None
    :return: qualified or unqualified table reference
    :rtype: str
    """
    result = f"{schema}.{table}" if schema else table
    return result


def _backfill_replay_guard_clause(
    column: str,
    default: str | None,
    explicit: bool,
) -> str:
    """
    build the ``AND <column> = <default>`` clause appended to backfills.

    when ``explicit`` is true and a non-NULL default is known, the helper
    emits a guard clause restricting the UPDATE to rows still at the
    pre-backfill state. re-running the migration becomes a clean no-op
    because the second run sees no rows matching the guard.

    when ``explicit`` is false, the caller has signalled the predicate
    they passed already encodes a replay-safe state (e.g. an ``IS NULL``
    check on the new column) and no extra clause is added.

    :param column: column being backfilled
    :ptype column: str
    :param default: column default expression, if any
    :ptype default: str | None
    :param explicit: whether to append a default-equality guard
    :ptype explicit: bool
    :return: SQL fragment, including a leading ``AND`` when present, or
        empty string when no guard is needed
    :rtype: str
    """
    result: str
    if explicit and default is not None:
        result = f" AND {column} = {default}"
    else:
        result = ""
    return result


@traced
async def add_column_with_backfill(
    store: MigrationStore,
    *,
    table: str,
    column: str,
    column_type: str,
    default: str | None = None,
    not_null: bool = False,
    backfill_value_sql: str | None = None,
    backfill_predicate: str | None = None,
    backfill_replay_guard: bool = True,
    schema: str | None = None,
) -> None:
    """
    add ``column`` to ``table`` and optionally backfill it -- yugabyte-safe.

    emits up to two separate ``store.execute`` calls so the ALTER and
    the UPDATE never share a transaction. the ALTER uses
    ``ADD COLUMN IF NOT EXISTS`` for idempotency; the UPDATE carries
    the caller's ``backfill_predicate`` plus an optional default-equality
    replay guard so a re-run after partial failure is a clean no-op.

    :param store: migration-time store; only ``execute`` is consumed
    :ptype store: MigrationStore
    :param table: target table (unqualified)
    :ptype table: str
    :param column: column being added
    :ptype column: str
    :param column_type: SQL type expression (e.g. ``"VARCHAR(8)"``,
        ``"INTEGER"``); passed through verbatim
    :ptype column_type: str
    :param default: optional column default expression (passed through
        verbatim); when supplied alongside the backfill flags, used as
        the replay-guard value
    :ptype default: str | None
    :param not_null: emit ``NOT NULL`` on the column declaration
    :ptype not_null: bool
    :param backfill_value_sql: when supplied, value SQL fragment used in
        the backfill UPDATE (e.g. ``"'platform'"``); ``None`` skips the
        backfill entirely
    :ptype backfill_value_sql: str | None
    :param backfill_predicate: WHERE clause restricting the backfill to
        the rows that need updating (e.g. ``"customer_id IS NULL"``);
        the helper appends a default-equality replay guard when both
        ``backfill_replay_guard=True`` and a ``default`` are supplied
    :ptype backfill_predicate: str | None
    :param backfill_replay_guard: append ``AND <column> = <default>`` to
        the backfill predicate so re-running is a no-op; default true
    :ptype backfill_replay_guard: bool
    :param schema: optional schema name; when supplied, every emitted
        statement is schema-qualified
    :ptype schema: str | None
    :return: nothing
    :rtype: None
    """
    qualified = _qualify(table, schema)
    null_clause = " NOT NULL" if not_null else ""
    default_clause = f" DEFAULT {default}" if default is not None else ""
    add_column_sql = (
        f"ALTER TABLE {qualified} ADD COLUMN IF NOT EXISTS {column} {column_type}{null_clause}{default_clause}"
    )
    log.info(
        "migration helper: add column %s.%s (%s)%s%s",
        qualified,
        column,
        column_type,
        " NOT NULL" if not_null else "",
        f" DEFAULT {default}" if default is not None else "",
    )
    await store.execute(add_column_sql)

    if backfill_value_sql is not None:
        guard_tail = _backfill_replay_guard_clause(
            column,
            default,
            backfill_replay_guard,
        )
        predicate = backfill_predicate or "TRUE"
        backfill_sql = f"UPDATE {qualified} SET {column} = {backfill_value_sql} WHERE {predicate}{guard_tail}"
        log.info(
            "migration helper: backfill %s.%s where %s",
            qualified,
            column,
            predicate,
        )
        await store.execute(backfill_sql)


@traced
async def add_check_constraint(
    store: MigrationStore,
    *,
    table: str,
    constraint_name: str,
    expression: str,
    schema: str | None = None,
    if_not_exists: bool = True,
) -> None:
    """
    add a CHECK constraint guarded by an ``information_schema`` probe.

    emits one execute carrying a DO block. the block consults
    ``pg_constraint`` to skip the ADD when the constraint already
    exists, so a re-run is a clean no-op.

    :param store: migration-time store
    :ptype store: MigrationStore
    :param table: target table (unqualified)
    :ptype table: str
    :param constraint_name: name to use for the new CHECK constraint
    :ptype constraint_name: str
    :param expression: SQL boolean expression (without leading ``CHECK``)
    :ptype expression: str
    :param schema: optional schema name; when supplied, the ALTER TABLE
        statement is schema-qualified and the existence probe
        constrains by ``ns.nspname``
    :ptype schema: str | None
    :param if_not_exists: when true, wrap the ALTER in an existence
        probe so re-runs are no-ops; when false, the helper emits a
        bare ALTER (caller takes responsibility for replay safety)
    :ptype if_not_exists: bool
    :return: nothing
    :rtype: None
    """
    qualified = _qualify(table, schema)
    log.info(
        "migration helper: add check constraint %s on %s",
        constraint_name,
        qualified,
    )
    if not if_not_exists:
        sql = f"ALTER TABLE {qualified} ADD CONSTRAINT {constraint_name} CHECK ({expression})"
        await store.execute(sql)
        return

    schema_filter = f"\n           AND ns.nspname = '{schema}'" if schema else ""
    sql = f"""
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint pc
          JOIN pg_class cls ON cls.oid = pc.conrelid
          JOIN pg_namespace ns ON ns.oid = cls.relnamespace
         WHERE cls.relname = '{table}'
           AND pc.conname = '{constraint_name}'{schema_filter}
    ) THEN
        ALTER TABLE {qualified}
          ADD CONSTRAINT {constraint_name} CHECK ({expression});
    END IF;
END
$$;
"""
    await store.execute(sql)


_RLS_COMMANDS = frozenset({"ALL", "SELECT", "INSERT", "UPDATE", "DELETE"})


@traced
async def create_policy_if_not_exists(
    store: MigrationStore,
    *,
    table: str,
    policy_name: str,
    using: str,
    command: str = "ALL",
    check: str | None = None,
    schema: str | None = None,
    to_role: str | None = None,
    permissive: bool = True,
) -> None:
    """
    create an RLS policy guarded by a ``pg_policies`` probe.

    ``CREATE POLICY`` has NO native ``IF NOT EXISTS`` form, so a re-run would raise
    "policy already exists" and break migration replay. This emits one ``store.execute``
    carrying a DO block that consults ``pg_policies`` and skips the CREATE when a policy of the
    same name already exists on the table -- so a re-run is a clean no-op, mirroring
    :func:`add_check_constraint`.

    :param store: migration-time store
    :ptype store: MigrationStore
    :param table: target table (unqualified)
    :ptype table: str
    :param policy_name: policy name (unique per table)
    :ptype policy_name: str
    :param using: row-visibility expression for the ``USING`` clause, WITHOUT the keyword
    :ptype using: str
    :param command: the command the policy applies to: ``ALL``/``SELECT``/``INSERT``/
        ``UPDATE``/``DELETE``
    :ptype command: str
    :param check: optional ``WITH CHECK`` expression (write policies: INSERT/UPDATE/ALL),
        WITHOUT the keyword; ``None`` omits the clause
    :ptype check: str | None
    :param schema: optional schema name; qualifies the table and constrains the probe by
        ``schemaname``
    :ptype schema: str | None
    :param to_role: optional role the policy applies to (``TO <role>``); ``None`` = all roles
    :ptype to_role: str | None
    :param permissive: ``PERMISSIVE`` (default, OR-combined) vs ``RESTRICTIVE`` (AND-combined)
    :ptype permissive: bool
    :return: nothing
    :rtype: None
    :raises ValueError: if ``command`` is not a recognized RLS command, or ``check`` is supplied
        for a read-only command (SELECT/DELETE), which would be invalid SQL
    """
    command_norm = command.strip().upper()
    if command_norm not in _RLS_COMMANDS:
        raise ValueError(
            f"create_policy_if_not_exists: unknown command {command!r}; expected one of {sorted(_RLS_COMMANDS)}"
        )
    if check is not None and command_norm in {"SELECT", "DELETE"}:
        raise ValueError(
            f"create_policy_if_not_exists: WITH CHECK is invalid for FOR {command_norm}; "
            "use USING for read/delete visibility"
        )
    qualified = _qualify(table, schema)
    log.info("migration helper: create policy %s on %s", policy_name, qualified)
    as_clause = "PERMISSIVE" if permissive else "RESTRICTIVE"
    to_clause = f"\n          TO {to_role}" if to_role else ""
    check_clause = f"\n          WITH CHECK ({check})" if check is not None else ""
    schema_filter = f"\n           AND schemaname = '{schema}'" if schema else ""
    sql = f"""
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE tablename = '{table}'
           AND policyname = '{policy_name}'{schema_filter}
    ) THEN
        CREATE POLICY {policy_name} ON {qualified}
          AS {as_clause}
          FOR {command_norm}{to_clause}
          USING ({using}){check_clause};
    END IF;
END
$$;
"""
    await store.execute(sql)


@traced
async def enable_row_level_security(
    store: MigrationStore,
    *,
    table: str,
    force: bool = True,
    schema: str | None = None,
) -> None:
    """
    enable (and by default ``FORCE``) row-level security on ``table``.

    ``ENABLE``/``FORCE ROW LEVEL SECURITY`` are idempotent in Postgres -- a re-run does not
    error -- so these are emitted as bare ALTERs, replay-safe without a probe. ``force=True``
    is the default AND the security-relevant choice: without ``FORCE`` the table OWNER bypasses
    every policy, so RLS silently does nothing for the owning role. Emitted as SEPARATE execute
    calls (each its own DDL statement; never mixed with DML).

    :param store: migration-time store
    :ptype store: MigrationStore
    :param table: target table (unqualified)
    :ptype table: str
    :param force: also issue ``FORCE ROW LEVEL SECURITY`` so the table owner is subject to
        policies (the secure default)
    :ptype force: bool
    :param schema: optional schema name; qualifies the table
    :ptype schema: str | None
    :return: nothing
    :rtype: None
    """
    qualified = _qualify(table, schema)
    log.info("migration helper: enable RLS on %s (force=%s)", qualified, force)
    await store.execute(f"ALTER TABLE {qualified} ENABLE ROW LEVEL SECURITY")
    if force:
        await store.execute(f"ALTER TABLE {qualified} FORCE ROW LEVEL SECURITY")


@traced
async def replace_check_constraint(
    store: MigrationStore,
    *,
    table: str,
    constraint_name: str,
    new_expression: str,
    schema: str | None = None,
    only_if_changed: bool = True,
    engine_normalized_def: str | None = None,
) -> None:
    """
    replace an existing CHECK constraint via compare-then-swap.

    emits one execute carrying a DO block that:

    1. captures the current constraint definition with
       ``pg_get_constraintdef``
    2. compares it against the target form (the literal
       ``CHECK ({new_expression})`` string by default, or
       ``engine_normalized_def`` when supplied)
    3. drops + re-adds the constraint only when the definitions differ

    when ``engine_normalized_def`` is ``None``, the target_def is the
    literal ``CHECK ({new_expression})``. postgres canonicalises
    expressions on storage (e.g. ``c IN ('a', 'b')`` -> ``((c)::text =
    ANY ((ARRAY['a'::character varying, 'b'::character varying]
    )::text[]))``), so a literal target_def does NOT match the stored
    form and the helper will DROP+ADD on every replay -- functionally
    idempotent but with OID churn. callers who need the OID-stable
    short-circuit MUST supply ``engine_normalized_def`` matching the
    output of ``pg_get_constraintdef`` exactly.

    :param store: migration-time store
    :ptype store: MigrationStore
    :param table: target table (unqualified)
    :ptype table: str
    :param constraint_name: name of the existing CHECK constraint
    :ptype constraint_name: str
    :param new_expression: target SQL boolean expression (without
        leading ``CHECK``); used to construct the ``ALTER TABLE ...
        ADD CONSTRAINT`` clause
    :ptype new_expression: str
    :param schema: optional schema name; when supplied, every emitted
        statement is schema-qualified
    :ptype schema: str | None
    :param only_if_changed: when true (default), the compare-then-swap
        pattern is used; when false, an unconditional DROP + ADD is
        emitted (caller takes responsibility for replay safety)
    :ptype only_if_changed: bool
    :param engine_normalized_def: optional pre-computed engine-form
        string for OID-stable replay; when supplied, the DO block
        compares the stored ``pg_get_constraintdef`` output against
        this value rather than the literal ``CHECK ({new_expression})``
    :ptype engine_normalized_def: str | None
    :return: nothing
    :rtype: None
    """
    qualified = _qualify(table, schema)
    log.info(
        "migration helper: replace check constraint %s on %s",
        constraint_name,
        qualified,
    )
    if not only_if_changed:
        sql = f"""
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint pc
          JOIN pg_class cls ON cls.oid = pc.conrelid
         WHERE cls.relname = '{table}'
           AND pc.conname = '{constraint_name}'
    ) THEN
        ALTER TABLE {qualified}
          DROP CONSTRAINT {constraint_name};
    END IF;
    ALTER TABLE {qualified}
      ADD CONSTRAINT {constraint_name} CHECK ({new_expression});
END
$$;
"""
        await store.execute(sql)
        return

    # the target_def SQL-string literal nests inside the outer DO block
    # body, so any single quote must be doubled to round-trip through
    # pl/pgsql parsing. without this escape an expression like ``c IN
    # ('a', 'b')`` produces ``'CHECK (c IN ('a', 'b'))'`` -- a syntax
    # error because pl/pgsql terminates the string at the first
    # unescaped quote inside the value.
    target_literal_value = engine_normalized_def if engine_normalized_def is not None else f"CHECK ({new_expression})"
    escaped_for_target_literal = target_literal_value.replace("'", "''")
    sql = f"""
DO $$
DECLARE
    current_def TEXT;
    target_def TEXT;
BEGIN
    target_def := '{escaped_for_target_literal}';

    SELECT pg_get_constraintdef(pc.oid)
      INTO current_def
      FROM pg_constraint pc
      JOIN pg_class cls ON cls.oid = pc.conrelid
     WHERE cls.relname = '{table}'
       AND pc.conname = '{constraint_name}';

    IF current_def IS NOT NULL AND current_def = target_def THEN
        RETURN;
    END IF;

    IF current_def IS NOT NULL THEN
        ALTER TABLE {qualified}
          DROP CONSTRAINT {constraint_name};
    END IF;

    ALTER TABLE {qualified}
      ADD CONSTRAINT {constraint_name} CHECK ({new_expression});
END
$$;
"""
    await store.execute(sql)


@dataclass(frozen=True)
class _PkSwapPlan:
    """
    plan describing the SQL fragments composed by :func:`replace_primary_key`.

    factored out as a dataclass so the helper body builds the plan in
    one pass and emits the assembled DO block in another -- single-return
    discipline preserved.

    :ivar drop_fks: list of ``ALTER TABLE ... DROP CONSTRAINT IF EXISTS`` lines
    :ivar recreate_fks: list of ``ALTER TABLE ... ADD CONSTRAINT ...`` lines
    :ivar pkey_name: the table's primary key constraint name
    :ivar unique_name: the ``UNIQUE (id)`` constraint name
    """

    drop_fks: list[str] = field(default_factory=list)
    recreate_fks: list[str] = field(default_factory=list)
    pkey_name: str = ""
    unique_name: str = ""


def _build_pk_swap_plan(
    table: str,
    pk_name: str | None,
    preserve_unique_id_column: str,
    inbound_fks: Sequence[InboundFk],
) -> _PkSwapPlan:
    """
    assemble the SQL fragments :func:`replace_primary_key` will emit.

    :param table: target table (unqualified)
    :ptype table: str
    :param pk_name: primary-key constraint name; defaults to
        ``{table}_pkey`` when ``None``
    :ptype pk_name: str | None
    :param preserve_unique_id_column: column name to receive the
        ``UNIQUE`` constraint that preserves inbound FK targets
    :ptype preserve_unique_id_column: str
    :param inbound_fks: descriptors of inbound FKs to drop and recreate
    :ptype inbound_fks: Sequence[InboundFk]
    :return: assembled plan
    :rtype: _PkSwapPlan
    """
    pkey_name = pk_name or f"{table}_pkey"
    unique_name = f"{table}_{preserve_unique_id_column}_unique"
    drop_fks = [f"ALTER TABLE {fk.source_table} DROP CONSTRAINT IF EXISTS {fk.constraint_name};" for fk in inbound_fks]
    recreate_fks = []
    for fk in inbound_fks:
        on_delete = f" {fk.on_delete}" if fk.on_delete else ""
        recreate_fks.append(
            f"ALTER TABLE {fk.source_table} "
            f"ADD CONSTRAINT {fk.constraint_name} "
            f"FOREIGN KEY ({fk.source_column}) "
            f"REFERENCES {table}({preserve_unique_id_column})"
            f"{on_delete};"
        )
    return _PkSwapPlan(
        drop_fks=drop_fks,
        recreate_fks=recreate_fks,
        pkey_name=pkey_name,
        unique_name=unique_name,
    )


@traced
async def replace_primary_key(
    store: MigrationStore,
    *,
    table: str,
    new_columns: Sequence[str],
    pk_name: str | None = None,
    preserve_unique_id_column: str = "id",
    inbound_fks: Sequence[InboundFk] = (),
    schema: str | None = None,
) -> None:
    """
    replace ``table``'s primary key, preserving inbound FK references.

    emits one execute carrying a DO block that performs the v054 PK-swap
    dance:

    1. probe ``pg_get_constraintdef`` to short-circuit if the table is
       already on the target composite PK form
    2. drop every inbound FK named in ``inbound_fks``
    3. drop the old primary key (named ``pk_name`` or
       ``{table}_pkey`` by default)
    4. add the new composite primary key on ``new_columns``
    5. add ``UNIQUE (preserve_unique_id_column)`` so future inbound
       FKs can target the preserved id column without dragging the
       partition column into every reference
    6. recreate each inbound FK with the original ``ON DELETE`` clause

    the DO block contains DDL only -- no DML, no UPDATE, nothing the
    yugabyte DDL/DML separation rule cares about.

    :param store: migration-time store
    :ptype store: MigrationStore
    :param table: target table (unqualified)
    :ptype table: str
    :param new_columns: column names forming the new composite primary
        key (order matters for index layout)
    :ptype new_columns: Sequence[str]
    :param pk_name: optional explicit primary-key constraint name;
        defaults to ``{table}_pkey``
    :ptype pk_name: str | None
    :param preserve_unique_id_column: column to receive ``UNIQUE``;
        default ``"id"`` matches the convention every aibots /
        agent-memory partitioned table uses
    :ptype preserve_unique_id_column: str
    :param inbound_fks: descriptors of inbound FKs to drop and recreate
        around the swap; postgres refuses to drop a PK while an FK
        depends on it, so each FK on the list is dropped explicitly
    :ptype inbound_fks: Sequence[InboundFk]
    :param schema: optional schema name; when supplied, every emitted
        statement on ``table`` is schema-qualified (the inbound FK
        statements remain unqualified because ``InboundFk.source_table``
        carries its own qualification)
    :ptype schema: str | None
    :return: nothing
    :rtype: None
    """
    qualified = _qualify(table, schema)
    plan = _build_pk_swap_plan(
        table=table,
        pk_name=pk_name,
        preserve_unique_id_column=preserve_unique_id_column,
        inbound_fks=inbound_fks,
    )
    log.info(
        "migration helper: replace primary key on %s -> (%s)",
        qualified,
        ", ".join(new_columns),
    )
    drop_fks_block = "\n        ".join(plan.drop_fks)
    recreate_fks_block = "\n        ".join(plan.recreate_fks)
    new_pk_columns_csv = ", ".join(new_columns)

    # current_schema() guard: every per-agent invocation lands in its
    # own ``agent_<hex>`` schema (search_path is set by the runner
    # caller). without filtering pg_constraint to ``cls.relnamespace
    # = current_schema()``, the second agent's run sees the FIRST
    # agent's already-swapped media_pkey row in the global catalog,
    # the inner ``NOT EXISTS (PK matching new shape)`` short-circuits
    # to FALSE, and the swap silently skips. result: the second
    # agent's media keeps the original ``PRIMARY KEY (media_id)`` and
    # the v010 composite FK fails with "no unique constraint matching
    # given keys for referenced table 'media'". one-database-many-
    # schemas (vanilla Postgres testcontainer) exposes this; YB's
    # one-database-per-tenant deployment shape masks it in
    # production.
    sql = f"""
DO $$
DECLARE
    cur_ns oid := (SELECT oid FROM pg_namespace WHERE nspname = current_schema());
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint pc
          JOIN pg_class cls ON cls.oid = pc.conrelid
         WHERE cls.relname = '{table}'
           AND cls.relnamespace = cur_ns
           AND pc.conname = '{plan.pkey_name}'
           AND pc.contype = 'p'
    ) AND NOT EXISTS (
        SELECT 1 FROM pg_constraint pc
          JOIN pg_class cls ON cls.oid = pc.conrelid
         WHERE cls.relname = '{table}'
           AND cls.relnamespace = cur_ns
           AND pc.contype = 'p'
           AND pg_get_constraintdef(pc.oid)
               LIKE 'PRIMARY KEY ({new_pk_columns_csv})%'
    ) THEN
        {drop_fks_block}
        ALTER TABLE {qualified} DROP CONSTRAINT {plan.pkey_name};
        ALTER TABLE {qualified}
          ADD CONSTRAINT {plan.pkey_name}
          PRIMARY KEY ({new_pk_columns_csv});
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint pc
              JOIN pg_class cls ON cls.oid = pc.conrelid
             WHERE cls.relname = '{table}'
               AND cls.relnamespace = cur_ns
               AND pc.conname = '{plan.unique_name}'
        ) THEN
            ALTER TABLE {qualified}
              ADD CONSTRAINT {plan.unique_name}
              UNIQUE ({preserve_unique_id_column});
        END IF;
        {recreate_fks_block}
    END IF;
END
$$;
"""
    await store.execute(sql)


@traced
async def add_partition_column(
    store: MigrationStore,
    *,
    table: str,
    column: str,
    column_type: str,
    default: str,
    backfill_value_sql: str,
    backfill_predicate: str | None = None,
    add_check_allowed_values: Sequence[str] | None = None,
    schema: str | None = None,
) -> None:
    """
    add a partition column to ``table``, backfill it, and gate it with a CHECK.

    composes :func:`add_column_with_backfill` and
    :func:`add_check_constraint` so the partition column lands with
    its closed-set value gate in one declarative call. each underlying
    helper emits its own ``execute`` calls, so the combined operation
    yields three executes (ADD COLUMN, UPDATE backfill, CHECK).

    :param store: migration-time store
    :ptype store: MigrationStore
    :param table: target table (unqualified)
    :ptype table: str
    :param column: partition column being added
    :ptype column: str
    :param column_type: SQL type expression (e.g. ``"VARCHAR(8)"``)
    :ptype column_type: str
    :param default: column default expression; required for partition
        columns since they must be NOT NULL
    :ptype default: str
    :param backfill_value_sql: value SQL fragment for rows matching
        ``backfill_predicate``
    :ptype backfill_value_sql: str
    :param backfill_predicate: WHERE clause restricting the backfill;
        ``None`` updates every row whose new column still equals the
        default
    :ptype backfill_predicate: str | None
    :param add_check_allowed_values: when supplied, install a
        CHECK constraint pinning the column to the closed set; the
        constraint is named ``{table}_{column}_ck``
    :ptype add_check_allowed_values: Sequence[str] | None
    :param schema: optional schema name
    :ptype schema: str | None
    :return: nothing
    :rtype: None
    """
    log.info(
        "migration helper: add partition column %s.%s (%s)",
        _qualify(table, schema),
        column,
        column_type,
    )
    await add_column_with_backfill(
        store,
        table=table,
        column=column,
        column_type=column_type,
        default=default,
        not_null=True,
        backfill_value_sql=backfill_value_sql,
        backfill_predicate=backfill_predicate,
        backfill_replay_guard=True,
        schema=schema,
    )
    if add_check_allowed_values is not None:
        values_csv = ", ".join(f"'{v}'" for v in add_check_allowed_values)
        await add_check_constraint(
            store,
            table=table,
            constraint_name=f"{table}_{column}_ck",
            expression=f"{column} IN ({values_csv})",
            schema=schema,
        )


@traced
async def add_index(
    store: MigrationStore,
    *,
    table: str,
    name: str,
    columns: Sequence[str],
    unique: bool = False,
    where: str | None = None,
    schema: str | None = None,
) -> None:
    """
    create an index using ``CREATE INDEX IF NOT EXISTS`` for idempotency.

    :param store: migration-time store
    :ptype store: MigrationStore
    :param table: target table (unqualified)
    :ptype table: str
    :param name: index name
    :ptype name: str
    :param columns: column names participating in the index (order
        matters for the index layout)
    :ptype columns: Sequence[str]
    :param unique: when true, emit ``CREATE UNIQUE INDEX``
    :ptype unique: bool
    :param where: optional partial-index predicate
    :ptype where: str | None
    :param schema: optional schema name; qualifies the table reference
    :ptype schema: str | None
    :return: nothing
    :rtype: None
    """
    qualified = _qualify(table, schema)
    unique_clause = "UNIQUE " if unique else ""
    columns_csv = ", ".join(columns)
    where_clause = f" WHERE {where}" if where else ""
    sql = f"CREATE {unique_clause}INDEX IF NOT EXISTS {name} ON {qualified} ({columns_csv}){where_clause}"
    log.info(
        "migration helper: add index %s on %s (%s)%s",
        name,
        qualified,
        columns_csv,
        " (partial)" if where else "",
    )
    await store.execute(sql)
