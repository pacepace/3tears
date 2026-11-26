"""The keyed reads the no-whole-reads walk sees through, each beside the whole read through the same shape it still
names: a read in a table's place (``live_read``, ``versions_read``, ``newest_read``) and the relation builders over
one; one range of a row-epoch table's declared index; a sweep by a ``TableSchema``'s own key prefix; and a CTE a
caller names by a module constant."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, no_whole_reads_findings

#: the ledger's name, beside each repo's gate
_EXEMPTIONS_NAME = "_no_whole_reads_exemptions.txt"

#: a finding naming one read: ``<path>:<line> in <function>: <kind> <what> -- ...``
_HIT: re.Pattern[str] = re.compile(r"^\S+:\d+ in (\S+): (\S+ \S+) -- ")


def _flagged(tmp_path: Path, modules: dict[str, str]) -> set[str]:
    """the functions the gate names a read in, over synthesised modules.

    :param tmp_path: the repo root
    :ptype tmp_path: Path
    :param modules: module path under ``src`` onto its source
    :ptype modules: dict[str, str]
    :return: the functions
    :rtype: set[str]
    """
    src = tmp_path / "src"
    for name, source in modules.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / _EXEMPTIONS_NAME).write_text("", encoding="utf-8")
    config = NoWholeReadsConfig(repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False)
    return {
        match.group(1)
        for match in (_HIT.match(finding) for finding in no_whole_reads_findings(config))
        if match is not None
    }


# -- a read in a table's place, inside the keyed package --------------------------------------------------------------

_LIVE_READS = """
    from threetears.core.keyed._filters import live_read, newest_read, versions_read
    from threetears.interfaces.keyed.filters import relation


    async def aggregate_keyed(conn, table, *, where=()):
        read_from, guarded = live_read(conn, table, where)
        return await _aggregate(conn, read_from, guarded)


    async def _aggregate(conn, table, guarded):
        return await conn.fetch(f"SELECT count(*) FROM {relation(table, 't', [])} WHERE {guarded}")


    async def read_after(conn, table, *, by=None):
        return await conn.fetch(f"SELECT a FROM {relation(newest_read(conn, table, by=by), 'r', [])} WHERE {by}")


    async def counts_by_relations(conn, tables, *, label):
        return await conn.fetch(f"SELECT count(*) FROM {relation(versions_read(conn, tables[label]), 'r', [])}")


    async def group_hashes(conn, relation_, *, where=()):
        read_from, guarded = live_read(conn, relation_, where)
        return await conn.fetch(f"SELECT h FROM (SELECT h FROM {relation(read_from, 't', [])}) AS s LIMIT 10")


    async def any_rows(conn, table):
        return await _over_a_table(conn)


    async def _over_a_table(conn):
        read_from, guarded = live_read(conn, "accounts", ())
        return await conn.fetch(f"SELECT a FROM {relation(read_from, 't', [])} WHERE {guarded}")


    async def holds_any(conn, table):
        guarded, read_from = live_read(conn, table, ())
        return await conn.fetch(f"SELECT a FROM {relation(read_from, 't', [])} WHERE {guarded}")


    async def count_at_most(conn, table):
        read_from, guarded = live_read(conn, table, ())
        read_from = "accounts"
        return await conn.fetch(f"SELECT a FROM {relation(read_from, 't', [])} WHERE {guarded}")


    async def read_value(conn, table):
        return await conn.fetch(f"SELECT a FROM {relation(newest_read(conn, 'accounts'), 'r', [])} WHERE {table}")
"""


def test_a_read_in_a_keyed_helper_s_table_s_place_is_judged_at_the_helper_s_callers(tmp_path: Path) -> None:
    """Flips if a keyed helper's own statement over its relation read in its place stays a finding: the first of
    ``live_read``'s pair (handed on to a private function too), ``newest_read`` and ``versions_read`` written into
    the statement, one relation of a mapping the helper is handed; or if one read in a table's place it is not handed
    is taken as its relation: a literal table, the second of the pair, a local bound twice, ``newest_read`` of a
    literal table."""
    flagged = _flagged(tmp_path, {"threetears/core/keyed/counts.py": _LIVE_READS})
    keyed = {"aggregate_keyed", "_aggregate", "read_after", "counts_by_relations", "group_hashes"}
    assert not flagged & keyed, flagged
    assert {"_over_a_table", "holds_any", "count_at_most", "read_value"} <= flagged, flagged


# -- the relation builders over a table, inside the keyed package -----------------------------------------------------

_BUILDERS = """
    from threetears.interfaces.keyed.filters import quoted_table


    def newest_relation(table, read_as, *, within=()):
        chosen = f"(SELECT DISTINCT ON (k) t.* FROM {quoted_table(read_as)} AS t ORDER BY k, epoch DESC) AS n"
        return _kept(table, read_as, chosen)


    def newest_lateral(table, read_as, *, keys):
        chosen = f"{keys} CROSS JOIN LATERAL (SELECT t.* FROM {quoted_table(read_as)} AS t WHERE {keys}) AS n"
        return _kept(table, read_as, chosen)


    def _kept(table, read_as, chosen):
        return f"SELECT n.* FROM {chosen} WHERE NOT n.{table}"


    def live_relation(declared, table):
        return f'SELECT {declared} FROM {quoted_table(table)} WHERE NOT "gone"'


    def other_relation(declared, table):
        return f'SELECT {declared} FROM {quoted_table(table)} WHERE NOT "gone"'


    def newest_of_accounts(table, read_as):
        chosen = f"(SELECT t.* FROM accounts AS t WHERE {read_as}) AS n"
        return _kept_whole(table, chosen)


    def _kept_whole(table, chosen):
        return f"SELECT n.* FROM {chosen} WHERE NOT n.{table}"
"""


def test_a_relation_builder_s_text_over_the_table_it_is_handed_is_judged_where_it_is_read(tmp_path: Path) -> None:
    """Flips if a builder of the relation a read reads in a table's place (``newest_relation``, ``newest_lateral``,
    ``live_relation``), or its private part handed the derived relation it writes over that table (``_kept``), stays
    a finding; or if a function of another name, or a private part handed a derived relation over a literal table, is
    taken as one."""
    flagged = _flagged(tmp_path, {"threetears/interfaces/keyed/keys.py": _BUILDERS})
    assert not flagged & {"newest_relation", "newest_lateral", "_kept", "live_relation"}, flagged
    assert {"other_relation", "_kept_whole"} <= flagged, flagged


_STEPS = """
    from threetears.core.keyed.steps import _Statement


    def latest_per_key(rel, *, key, order, where=(), gone=None):
        heads = latest_per_key(rel, key=key, order=order, where=where)
        inner = _Statement()
        newest = inner.source(heads, "n")
        return inner.step(f"SELECT a FROM {newest} WHERE NOT n.{gone}")


    def windowed(rel, *, windows, keep=(), where=()):
        heads = latest_per_key(rel, key=windows, order=keep)
        heads = latest_per_key("accounts", key=windows, order=keep)
        inner = _Statement()
        newest = inner.source(heads, "n")
        return inner.step(f"SELECT a FROM {newest} WHERE NOT n.{keep}")
"""


def test_a_step_over_a_keyed_step_s_relation_is_that_relation(tmp_path: Path) -> None:
    """Flips if a keyed step's statement over a step of one relation it built over its own relation (``heads =
    latest_per_key(rel, ...)``) stays a finding, or if one whose local is bound twice (once over a literal table) is
    taken as its relation."""
    flagged = _flagged(tmp_path, {"threetears/core/keyed/steps.py": _STEPS})
    assert "latest_per_key" not in flagged, flagged
    assert "windowed" in flagged, flagged


# -- one range of a row-epoch table's index ---------------------------------------------------------------------------

_RANGES = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.collections.table_restore import RestoredTable, StampedTable, VersionedTable
    from threetears.core.keyed import AnyOf, Fragment, Sort, copy_keyed, keyset_page
    from threetears.interfaces.keyed import page_statement

    # as row_epoch declares them (the walk reads them where 3tears' own modules are walked)
    EPOCH_COLUMN = "epoch"
    BUCKET_COLUMN = "epoch_bucket"


    async def scope_range(pool, table: VersionedTable, value, floor):
        return await copy_keyed(
            pool,
            table.relation,
            by={table.scope: value},
            order=[Sort(EPOCH_COLUMN, cast="bigint")],
            where=[Fragment(f't."{EPOCH_COLUMN}" >= $1', (floor,))],
        )


    async def bucket_range(pool, table: StampedTable, bucket, floor):
        return await copy_keyed(
            pool, table.relation, by={table.bucket: bucket}, where=[Fragment('t."epoch" >= $1', (floor,))]
        )


    async def either_range(pool, table: RestoredTable, bucket, scope, floor, order, after, filters):
        bounds = [Fragment('t."epoch" >= $1', (floor,)), Fragment('t."epoch" <= $1', (floor,))]
        by = {table.bucket: bucket} if isinstance(table, StampedTable) else {table.scope: scope}
        return await keyset_page(
            pool, table.relation, by=by, order=order, cursor=after, where=[*bounds, *filters], with_gone=True
        )


    def stamped_page(schema: TableSchema, bucket, since):
        return page_statement(
            schema.name,
            (*schema.pk_columns, EPOCH_COLUMN),
            {BUCKET_COLUMN: bucket},
            (Sort(EPOCH_COLUMN, cast="bigint"),),
            10,
            None,
            False,
            (),
            (Fragment('t."gone"'), Fragment(f't."{EPOCH_COLUMN}" >= $1', (since,))),
            None,
        )


    def versioned_page(schema: TableSchema, value, since, order):
        by = {schema.row_epoch.scope: value}
        return page_statement(
            schema.name, None, by, order, 10, None, False, (), (Fragment('t."epoch" >= $1', (since,)),), None
        )


    async def bounded_above(pool, table: VersionedTable, value, floor):
        return await copy_keyed(
            pool, table.relation, by={table.scope: value}, where=[Fragment('t."epoch" < $1', (floor,))]
        )


    async def no_lead(pool, table: VersionedTable, floor):
        return await copy_keyed(pool, table.relation, where=[Fragment('t."epoch" >= $1', (floor,))])


    async def another_record_s_lead(pool, table: VersionedTable, bucket, floor):
        return await copy_keyed(
            pool, table.relation, by={table.bucket: bucket}, where=[Fragment('t."epoch" >= $1', (floor,))]
        )


    async def unannotated(pool, table, value, floor):
        return await copy_keyed(
            pool, table.relation, by={table.scope: value}, where=[Fragment('t."epoch" >= $1', (floor,))]
        )


    async def two_pins(pool, table: VersionedTable, value, kind, floor):
        return await copy_keyed(
            pool,
            table.relation,
            by={table.scope: value, "kind": kind},
            where=[Fragment('t."epoch" >= $1', (floor,))],
        )


    async def every_scope(pool, table: VersionedTable, values, floor):
        return await copy_keyed(
            pool, table.relation, by={table.scope: AnyOf(values)}, where=[Fragment('t."epoch" >= $1', (floor,))]
        )


    async def off_the_index_order(pool, table: VersionedTable, value, floor):
        return await copy_keyed(
            pool,
            table.relation,
            by={table.scope: value},
            order=[Sort("name")],
            where=[Fragment('t."epoch" >= $1', (floor,))],
        )


    def with_a_visibility_clause(schema: TableSchema, bucket, since, visible):
        return page_statement(
            schema.name,
            None,
            {BUCKET_COLUMN: bucket},
            (Sort(EPOCH_COLUMN),),
            10,
            None,
            False,
            (Fragment(visible),),
            (Fragment('t."epoch" >= $1', (since,)),),
            None,
        )


    def another_column(schema: TableSchema, state, since):
        return page_statement(
            schema.name, None, {"state": state}, (Sort("epoch"),), 10, None, False, (),
            (Fragment('t."epoch" >= $1', (since,)),), None,
        )
"""


def test_one_range_of_a_row_epoch_table_s_declared_index_is_keyed(tmp_path: Path) -> None:
    """Flips if a read of one range of the row-epoch index a versioned or stamped table's declaration gives it -- its
    leading column (a restore record's scope or bucket, a ``TableSchema``'s ``row_epoch.scope`` or ``BUCKET_COLUMN``)
    pinned to one value, the epoch bounded from below -- stays unknown; or if one bounded only above, pinning no lead,
    another record's lead, a lead of an unannotated parameter, more than the lead, a set of leads, ordered off the
    index, beside a visibility clause, or pinning another column, is taken as keyed."""
    flagged = _flagged(tmp_path, {"pkg/ranges.py": _RANGES})
    keyed = {"scope_range", "bucket_range", "either_range", "stamped_page", "versioned_page"}
    assert not flagged & keyed, flagged
    whole = {
        "bounded_above",
        "no_lead",
        "another_record_s_lead",
        "unannotated",
        "two_pins",
        "every_scope",
        "off_the_index_order",
        "with_a_visibility_clause",
        "another_column",
    }
    assert whole <= flagged, whole - flagged


# -- a sweep by a TableSchema's own key prefix ------------------------------------------------------------------------

_SWEEPS = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import delete_before


    async def run_rows(pool, schema: TableSchema, values, epoch):
        return await delete_before(
            schema.name,
            collection=None,
            key=schema.stored_key,
            column="epoch",
            before=epoch,
            page=10,
            by=dict(zip(schema.pk_columns[: max(1, len(values))], values, strict=True)),
            conn=pool,
        )


    async def may_pin_nothing(pool, schema: TableSchema, values, epoch):
        return await delete_before(
            schema.name,
            collection=None,
            key=schema.stored_key,
            column="epoch",
            before=epoch,
            page=10,
            by=dict(zip(schema.pk_columns[: len(values)], values, strict=True)),
            conn=pool,
        )


    async def not_strict(pool, schema: TableSchema, values, epoch):
        return await delete_before(
            schema.name,
            collection=None,
            key=schema.stored_key,
            column="epoch",
            before=epoch,
            page=10,
            by=dict(zip(schema.pk_columns[: max(1, len(values))], values)),
            conn=pool,
        )


    async def at_least_none(pool, schema: TableSchema, values, epoch):
        return await delete_before(
            schema.name,
            collection=None,
            key=schema.stored_key,
            column="epoch",
            before=epoch,
            page=10,
            by=dict(zip(schema.pk_columns[: max(0, len(values))], values, strict=True)),
            conn=pool,
        )


    async def a_prefix_mapping(pool, schema: TableSchema, prefix, epoch):
        return await delete_before(
            schema.name, collection=None, key=schema.stored_key, column="epoch", before=epoch, page=10, by=prefix,
            conn=pool,
        )


    async def another_key(pool, schema: TableSchema, values, epoch):
        return await delete_before(
            schema.name,
            collection=None,
            key=("id",),
            column="epoch",
            before=epoch,
            page=10,
            by=dict(zip(schema.pk_columns[: max(1, len(values))], values, strict=True)),
            conn=pool,
        )


    async def not_a_schema(pool, schema, values, epoch):
        return await delete_before(
            schema.name,
            collection=None,
            key=schema.stored_key,
            column="epoch",
            before=epoch,
            page=10,
            by=dict(zip(schema.pk_columns[: max(1, len(values))], values, strict=True)),
            conn=pool,
        )
"""


def test_a_sweep_by_a_table_schema_s_own_key_prefix_is_keyed(tmp_path: Path) -> None:
    """Flips if a ``delete_before`` of a ``TableSchema`` parameter's table, its rows named by its own key, pinning at
    least the key's first column (``by=dict(zip(pk_columns[:max(1, n)], values, strict=True))``) stays unknown; or if
    one that may pin nothing (``[:n]``, ``max(0, ...)``), may pin short (no ``strict``), pins a run-time mapping, names
    its rows by another key, or sweeps an unannotated parameter's table, is taken as keyed."""
    flagged = _flagged(tmp_path, {"pkg/sweeps.py": _SWEEPS})
    assert "run_rows" not in flagged, flagged
    whole = {"may_pin_nothing", "not_strict", "at_least_none", "a_prefix_mapping", "another_key", "not_a_schema"}
    assert whole <= flagged, whole - flagged


# -- a CTE named by a module constant ---------------------------------------------------------------------------------

_MOVES = """
    _MOVED = "moved_rows"


    class Record:
        def append_sql(self, source):
            return f"INSERT INTO moves (a) SELECT t.a FROM {source} AS t WHERE t.moved"


    def with_moves(statement, record):
        return f"WITH {_MOVED} AS ({statement}), noted AS ({record.append_sql(_MOVED)}) SELECT 1 FROM {_MOVED}"
"""

_A_TABLE_PASSED = """
    _ACCOUNTS = "accounts"


    def copied(record):
        return record.append_sql(_ACCOUNTS)
"""


def test_a_cte_a_caller_names_by_a_module_constant_is_the_cte(tmp_path: Path) -> None:
    """Flips if a statement reading a parameter every caller fills with a CTE its own statement defines, both named by
    one module constant (``WITH {_MOVED} AS (...)``, ``append_sql(_MOVED)``), stays a finding; or if a caller passing
    a constant naming no CTE it defines leaves the parameter read as a CTE."""
    assert "Record.append_sql" not in _flagged(tmp_path / "cte", {"pkg/moves.py": _MOVES})
    flagged = _flagged(tmp_path / "table", {"pkg/moves.py": _MOVES, "pkg/copied.py": _A_TABLE_PASSED})
    assert "Record.append_sql" in flagged, flagged
