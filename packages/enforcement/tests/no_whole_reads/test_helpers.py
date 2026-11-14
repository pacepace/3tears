"""A call to a keyed-read helper is judged by the statement it sends: keyed when its literal key is, an
``unknown-predicate`` when the walk cannot read its arguments, a whole read when the key it names is not one."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from threetears.enforcement.no_whole_reads.helpers import (
    UNSEEN,
    helper_statements,
    is_keyed_fragment,
    narrows,
)
from threetears.enforcement.no_whole_reads.schema import COPY_PREFIX, SchemaCatalog, collect_schema
from threetears.enforcement.no_whole_reads.sql import Reader, judge_sql
from threetears.enforcement.no_whole_reads.walker import DEFAULT_NARROWING_KEYWORDS

#: the function the purposed sweep index names
_SWEEP = "src/app.py::sweep"

_DDL = f"""
CREATE TABLE results (state TEXT, geo TEXT, id BIGINT, stamp TIMESTAMPTZ, kind TEXT, PRIMARY KEY (state, geo, id));
CREATE TABLE leases (purpose TEXT, key TEXT, expires_at TIMESTAMPTZ, PRIMARY KEY (purpose, key));
CREATE TABLE tenant_leases (customer_id UUID, key TEXT, expires_at TIMESTAMPTZ, PRIMARY KEY (customer_id, key));
-- purpose: {_SWEEP}
-- plan_test: tests/test_app.py::test_sweep_plan
CREATE INDEX tenant_leases_by_expiry ON tenant_leases (customer_id, expires_at);
CREATE TABLE grants (who TEXT, state TEXT, PRIMARY KEY (who, state));
CREATE TABLE fence (signal TEXT PRIMARY KEY, run_id BIGINT);
-- purpose: src/app.py::reader
-- plan_test: tests/test_app.py::test_fence_plan
CREATE INDEX fence_by_signal_c ON fence (signal COLLATE "C" ASC);
CREATE TABLE unindexed (name TEXT, run_id BIGINT PRIMARY KEY);
CREATE TABLE threads (thread_id TEXT, checkpoint_id TEXT, PRIMARY KEY (thread_id, checkpoint_id));
CREATE TABLE relations (id UUID PRIMARY KEY, customer_id UUID, name TEXT);
-- purpose: src/app.py::reader
-- plan_test: tests/test_app.py::test_relations_plan
CREATE INDEX relations_by_customer ON relations (customer_id, name, id);
-- purpose: src/app.py::reader
-- plan_test: tests/test_app.py::test_relations_plan
CREATE INDEX relations_by_lower_name ON relations (lower(name), id);
-- purpose: src/app.py::reader
-- plan_test: tests/test_app.py::test_relations_plan
CREATE INDEX relations_by_folded_name ON relations ((lower(name) COLLATE "C") ASC, customer_id ASC);
CREATE TABLE nodes (id UUID PRIMARY KEY, kind TEXT, metadata JSONB);
-- purpose: src/app.py::reader
-- plan_test: tests/test_app.py::test_nodes_plan
CREATE INDEX nodes_by_kind_pod ON nodes (kind, (metadata->>'pod'));
-- purpose: src/app.py::reader
-- plan_test: tests/test_app.py::test_nodes_plan
CREATE INDEX nodes_by_pod ON nodes ((metadata->>'pod'));
"""


@pytest.fixture
def schema(tmp_path: Path) -> SchemaCatalog:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "v001.sql").write_text(_DDL, encoding="utf-8")
    return collect_schema(tmp_path, (tmp_path / "src",))


def _call(code: str) -> ast.Call:
    node = ast.parse(code).body[0]
    assert isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    return node.value


def _judged(code: str, schema: SchemaCatalog, name: str = "src/app.py::reader") -> list[list[tuple[str, str]]]:
    judged = []
    for statement in helper_statements(_call(code)):
        assert statement.sql is not UNSEEN, statement
        reader = Reader(schema, frozenset({name}))
        judged.append([(finding.kind, finding.table.strip('"')) for finding in judge_sql(statement.sql, reader)])
    return judged


@pytest.mark.parametrize(
    "code",
    [
        # a keyed batch by a leading part of the key, or by the key's rest after a pinned prefix
        'read_keyed(c, "results", keys=KeyBatch(("state", "geo"), ks), columns=cols)',
        'read_keyed_arrow(c, "results", keys=KeyBatch(("id",), ks), by={"state": s, "geo": g}, types=t)',
        # a page after a bound prefix, ordered by the key's next column; one across the whole key
        'keyset_page(c, "results", columns=cols, by={"state": s}, order=[Sort("geo"), Sort("id", "bigint")], limit=5)',
        'keyset_page(c, "results", columns=cols, by={}, order=[Sort("state"), Sort("geo"), Sort("id")], limit=5)',
        # its visibility a keyed probe, or a reviewed caller-visibility builder
        'keyset_page(c, "results", columns=cols, by={"state": s}, order=[Sort("geo"), Sort("id")], limit=5, '
        'visible=[exists_keyed("grants", on={"state": "state"}, by={"who": w})])',
        'keyset_page(c, "results", columns=cols, by={"state": s}, order=[Sort("geo"), Sort("id")], limit=5, '
        "visible=[Fragment.of(caller_visible_customer_clause(user_id=u, customer_id_column=x, "
        "scope_namespace_type=n))])",
        # the caller's visible customers as a bound set (VisibleCustomers.fragment): the page is theirs, by their index
        'keyset_page(c, "relations", columns=cols, by={}, order=[Sort("name"), Sort("id", "uuid")], limit=5, '
        "visible=[customers.fragment('t.\"customer_id\"', platform_rows=True)])",
        # a sort by an expression an index holds
        'keyset_page(c, "relations", columns=cols, by={}, '
        'order=[Sort("name_key", expression=\'lower(t."name")\'), Sort("id", "uuid")], limit=5)',
        # ... in an explicit collation the index holds it in (the hub's name-search pages, v158), page after page;
        # a collation written quoted or bare is the same collation
        'keyset_page(c, "relations", columns=cols, by={}, '
        'order=[Sort("name_folded", expression=\'lower(t."name") COLLATE "C"\'), Sort("customer_id", "uuid")], '
        "limit=5, cursor=k)",
        'keyset_page(c, "relations", columns=cols, by={}, '
        'order=[Sort("name_folded", expression=\'lower(t."name") COLLATE C\'), Sort("customer_id", "uuid")], '
        "limit=5, cursor=k)",
        'keyset_rows(c, "relations", columns=cols, by={}, '
        'order=[Sort("name_folded", expression=\'lower(t."name") COLLATE "C"\'), Sort("customer_id", "uuid")], '
        "page=500)",
        # every row of a prefix, page after page
        'keyset_rows(c, "results", columns=cols, by={"state": s}, order=[Sort("geo"), Sort("id", "bigint")], page=500)',
        # a scope's changed rows by hash: its keys and hashes paged in the key's order, then those keys read
        'read_changed(c, "results", key=["state", "geo", Sort("id", "bigint")], columns=cols, held=h, '
        'hash_columns=hc, by={"state": s})',
        # ... its key handed at run time (a source's own key): phase 1 judged by its pins, a leading key's here
        # (R-D, coordinator 2026-11-12)
        'read_changed(c, "results", key=list(source.key), columns=cols, held=h, hash_columns=hc, by={"state": s})',
        'count_at_most(c, "results", by={"state": s}, at_most=3, order=["state", "geo", "id"])',
        # many scopes' group hashes in one statement: the scopes' rows, bound by the key's leading column
        'group_hashes(c, "results", scope="state", scopes=ss, group="geo", hash_columns=hc)',
        'group_hashes(c, "results", scope=Sort("state"), scopes=ss, group="geo", hash_columns=hc, '
        "where=[Fragment('t.\"geo\" = ANY($1)', (g,))])",
        # a delete of the rows a prefix names, through the C-ordered index its reader's purpose names
        'delete_keyed(c, "fence", prefix={"signal": p}, returning=("signal",))',
        'update_keyed(c, "fence", values={"run_id": r}, prefix={"signal": p})',
        # texts of a key prefix's rows, packed several to a row
        'read_packed(c, "results", "geo", by={"state": s}, packed_rows=5, items=100)',
        'read_packed(c, "results", "geo", where=[Fragment(\'r."state" = $1\', (s,))], packed_rows=5, items=9)',
        # a delete or update of the rows a key prefix or a batch of keys names
        'delete_keyed(c, "results", by={"state": s})',
        'delete_keyed(c, "results", keys=KeyBatch(("state", "geo", "id"), ks), returning=("id",))',
        'update_keyed(c, "results", values={"kind": k}, by={"state": s, "geo": g})',
        'update_keyed(c, "results", values={"kind": k}, keys=KeyBatch(("state", "geo", "id"), ks))',
        # a bounded look: any rows up to its limit, of the table or of a key's rows
        'any_rows(c, "results", ["state", "geo"], limit=500)',
        'any_rows(c, "results", cols, limit=1, by={"state": s})',
        'holds_any(c, "results", by={"state": s}, order=["state", "geo", "id"])',
        'holds_any(c, "results", order=("state", "geo", "id"))',
        'read_after(c, "results", columns=cols, id_column="id", after=0, limit=3, by={"state": s, "geo": g})',
        # an export's SELECT written out with literals: keyed as the bound read it writes out
        'literal_select("results", columns=cols, by={"state": AnyOf(ss)})',
        'literal_select("results", columns=cols, by={"state": s}, where=[Fragment(\'t."geo" IS NOT NULL\')])',
        # a change record read past its marks within the key prefix its by pins
        'read_past_marks(c, "results", columns=cols, marks={"up": m}, by={"state": s})',
        'read_past_marks(c, "results", columns=cols, marks={"up": m}, by={"state": s}, where=[Fragment(\'"geo" IS NULL\')])',
    ],
)
def test_a_helper_naming_a_key_is_keyed(code: str, schema: SchemaCatalog) -> None:
    """Flips if a keyed helper call is judged a whole read, or synthesizes no statement at all."""
    judged = _judged(code, schema)
    assert judged and all(findings == [] for findings in judged), judged


def test_a_sweep_of_a_key_prefix_is_keyed_anywhere(schema: SchemaCatalog) -> None:
    """Flips if a sweep binding a key prefix (purpose leads leases' key) is refused outside a purposed index."""
    code = 'delete_before("leases", collection=c, key=["purpose", "key"], column="expires_at", before=t, page=5, by={"purpose": p})'
    assert _judged(code, schema, "src/app.py::elsewhere") == [[]]


def test_a_tenant_sweep_is_keyed_only_through_the_index_naming_its_caller(schema: SchemaCatalog) -> None:
    """Flips if a sweep bound only by its tenant passes without its purposed index, or fails with it."""
    code = (
        'delete_before("tenant_leases", collection=c, key=["customer_id", "key"], column="expires_at", before=t, page=5, '
        'by={"customer_id": p})'
    )
    assert _judged(code, schema, _SWEEP) == [[]]
    assert _judged(code, schema, "src/app.py::elsewhere") != [[]]


def test_a_keyed_count_is_a_keyed_count(schema: SchemaCatalog) -> None:
    code = 'counts_by(c, "results", column="geo", values=AnyOf(g), by={"state": s})'
    assert _judged(code, schema) == [[("keyed-count", "results")]]


@pytest.mark.parametrize(
    "code",
    [
        # a count grouped by a column named at run time, its rows bound by a literal key pin (reports #55 ask 1)
        'counts_by(c, "results", column=column, by={"state": s})',
        'counts_by(c, "results", column=part.column, by={"state": AnyOf(ss)}, where=required)',
        # a literal key pin beside a mapping merged in at run time: the merge only narrows it (reports #55 ask 2)
        'counts_by(c, "results", column="geo", by={"state": AnyOf(ss), **others})',
        'counts_by(c, "results", column=column, by={"state": AnyOf(ss), **others}, where=required)',
        'counts_by_relations(c, {"a": "results"}, column=column, by={"state": AnyOf(ss), **others})',
    ],
)
def test_a_count_judged_by_its_literal_pins_beside_run_time_parts(code: str, schema: SchemaCatalog) -> None:
    """Flips if a grouped count whose literal ``by`` entry pins the key is left unknown because its grouped column or
    the rest of its pins are named at run time (reports #55 asks 1 and 2)."""
    assert _judged(code, schema) == [[("keyed-count", "results")]]


@pytest.mark.parametrize(
    "code",
    [
        # the literal entry pins no key: the merged pins may, but the walk cannot see them
        'counts_by(c, "results", column=column, by={"kind": k, **others})',
        'counts_by(c, "results", column="geo", by={"kind": AnyOf(ks), **others})',
    ],
)
def test_a_count_whose_literal_pins_bind_no_key_is_still_found(code: str, schema: SchemaCatalog) -> None:
    """Flips if a count beside run-time parts passes when its literal pins bind no key, or if its statement does not
    say its pins were merged (the walk then names it unknown-predicate, never a guessed non-key read)."""
    judged = _judged(code, schema)
    assert judged != [[("keyed-count", "results")]] and judged != [[]], judged
    assert [statement.merged for statement in helper_statements(_call(code))] == ["by"]


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        # a batch of a later key column alone: every row is read to find them
        ('read_keyed(c, "results", keys=KeyBatch(("geo",), ks), columns=cols)', "whole-select"),
        ('holds_any(c, "results", by={"kind": k}, order=["state", "geo", "id"])', "non-key-where"),
        ('count_at_most(c, "results", by={"kind": k}, at_most=3, order=["state", "geo", "id"])', "non-key-where"),
        # a page past a collated expression no index holds in that collation (only lower(name) COLLATE "C" is)
        (
            'keyset_page(c, "relations", columns=cols, by={}, order=[Sort("name_folded", '
            'expression=\'upper(t."name") COLLATE "C"\'), Sort("customer_id", "uuid")], limit=5, cursor=k)',
            "non-key-where",
        ),
        # a prefix delete on a column no index ranges
        ('delete_keyed(c, "unindexed", prefix={"name": p})', "non-key-where"),
        # ... or one only the primary key leads with: the prefix is a code-point (COLLATE "C") range, which a key in
        # the column's own collation does not order (and YugabyteDB hashes a key's first column): every row is read
        ('delete_keyed(c, "threads", prefix={"thread_id": p})', "non-key-where"),
        ('update_keyed(c, "threads", values={"checkpoint_id": r}, prefix={"thread_id": p})', "non-key-where"),
        ('update_keyed(c, "unindexed", values={"run_id": r}, prefix={"name": p})', "non-key-where"),
        # a packed read pinning a column no key leads with, or nothing
        ('read_packed(c, "results", "geo", by={"kind": k}, packed_rows=5, items=100)', "non-key-where"),
        ('read_packed(c, "results", "geo", packed_rows=5, items=100)', "whole-select"),
        # a relation ordered whole: every row read to order it
        ('ordered("results", [Sort("geo")])', "whole-select"),
        # a delete or update pinning a column no key leads with: every row is read to find them
        ('delete_keyed(c, "results", by={"kind": k})', "non-key-where"),
        ('update_keyed(c, "results", values={"geo": g}, by={"kind": k})', "non-key-where"),
        # a bounded look filtered on a column no key leads with
        ('any_rows(c, "results", cols, limit=5, by={"kind": k})', "non-key-where"),
        # the same listing with its customers' fragment missing: every customer's rows, by no index
        (
            'keyset_page(c, "relations", columns=cols, by={}, order=[Sort("name"), Sort("id", "uuid")], limit=5)',
            "non-key-where",
        ),
        # a sort by an expression no index holds
        (
            'keyset_page(c, "relations", columns=cols, by={}, '
            'order=[Sort("name_key", expression=\'upper(t."name")\'), Sort("id", "uuid")], limit=5)',
            "non-key-where",
        ),
        # a drain ordered by no index: each page reads and sorts every row
        ('keyset_rows(c, "results", columns=cols, by={}, order=[Sort("kind"), Sort("id")], page=500)', "non-key-where"),
        # group hashes scoped by a column no key leads with: every row is read to find the scopes'
        ('group_hashes(c, "results", scope="kind", scopes=ss, group="geo", hash_columns=hc)', "non-key-where"),
        # a literal select pinning no key column: every row is read for its filter
        ('literal_select("results", columns=cols, by={"kind": AnyOf(ks)})', "non-key-where"),
        # a change record read past its marks pinning no key: every row's stamps are read
        ('read_past_marks(c, "results", columns=cols, marks={"up": m}, by={"kind": k})', "non-key-where"),
        # a changed-rows read keyed by no key of its relation: its hash pages read and sort every row
        (
            'read_changed(c, "results", key=["kind", "id"], columns=cols, held=h, hash_columns=hc, by={})',
            "non-key-where",
        ),
        # ... its key handed at run time: judged by its pins alone (R-D, coordinator 2026-11-12), none a whole read,
        # a column no key leads with read for its filter
        ('read_changed(c, "results", key=key, columns=cols, held=h, hash_columns=hc, by={})', "whole-select"),
        ('read_changed(c, "results", key=key, columns=cols, held=h, hash_columns=hc, by={"kind": k})', "non-key-where"),
    ],
)
def test_a_helper_naming_no_key_is_a_whole_read(code: str, kind: str, schema: SchemaCatalog) -> None:
    """Flips if a helper call passes on its name alone, whatever key it names."""
    judged = _judged(code, schema)
    assert any(found[0] == kind for findings in judged for found in findings), judged


@pytest.mark.parametrize(
    ("code", "argument"),
    [
        ('read_keyed(c, f(), keys=KeyBatch(("state",), ks), columns=cols)', "table"),
        ('read_keyed(c, table, keys=KeyBatch(("state",), ks), columns=cols)', "table"),
        ('read_keyed(c, "results", keys=batch, columns=cols)', "keys"),
        ('read_keyed(c, "results", keys=KeyBatch(f(), ks), columns=cols)', "keys"),
        ('keyset_page(c, "results", columns=cols, by=pinned, order=[Sort("geo")], limit=5)', "by"),
        ('keyset_page(c, "results", columns=cols, by={"state": s}, order=order, limit=5)', "order"),
        ('keyset_page(c, "results", columns=cols, by={"state": s}, order=[Sort(column)], limit=5)', "order"),
        ('keyset_page(c, "results", columns=cols, by={}, order=[Sort("geo")], limit=5, visible=[clause])', "visible"),
        # the caller's customers passed without their fragment: no reviewed builder says what it admits
        (
            'keyset_page(c, "results", columns=cols, by={"state": s}, order=[Sort("geo")], limit=5, visible=[customers])',
            "visible",
        ),
        ('counts_by(c, "results", column=name, values=AnyOf(g), by={"state": s})', "column"),
        # a run-time column bound to a set of values: a predicate no one can see
        ('counts_by(c, "results", column=name, values=AnyOf(g), by={"state": s, **others})', "column"),
        # a merged mapping with no literal entry
        ('counts_by(c, "results", column="geo", by={**others})', "by"),
        ('holds_any(c, "results", by={**pinned}, order=["state"])', "by"),
        ('holds_any(c, "results", by={"state": s})', "order"),
        ('count_at_most(c, "results", by={"state": s}, at_most=3)', "order"),
        ('count_at_most(c, "results", at_most=3, order=key)', "order"),
        ('delete_before("leases", collection=c, key=key, column="expires_at", before=t, page=5)', "key"),
        ('read_past_marks(c, "rel", columns=cols, marks=marks)', "marks"),
        # a run-time key whose by= the walk cannot read either
        ('read_changed(c, "results", key=key, columns=cols, held=h, hash_columns=hc, by=pinned)', "by"),
        ('group_hashes(c, "results", scope=column, scopes=ss, group="geo", hash_columns=hc)', "scope"),
        ('group_hashes(c, "results", scope="state", group="geo", hash_columns=hc)', "scopes"),
        # a look with no bound is no bounded look
        ('any_rows(c, "results", cols)', "limit"),
        ('delete_keyed(c, "results", by=pinned)', "by"),
        # a table named at run time read by a literal column: the walk cannot match it to any table's key
        ('read_row(c, table, by={"id": i})', "table"),
        ('read_row(c, table, by={pk_column: i, "id": j})', "table"),
    ],
)
def test_a_helper_whose_arguments_the_walk_cannot_read_is_unseen(code: str, argument: str) -> None:
    """Flips if a call the walk cannot read is passed (or judged on a guess) instead of named unknown-predicate."""
    statements = list(helper_statements(_call(code)))
    assert [(statement.sql, statement.unseen) for statement in statements] == [(UNSEEN, argument)]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # a generic collection's own row by its key column: keyed when the walk hands the helper ``pk`` (the column
        # flows from a declared key source, :mod:`~threetears.enforcement.no_whole_reads.keys`), as its SQL over
        # {pk} would be (``"{}"``: a table its caller names, as the walk writes it where the module declares no
        # template); a column whose NAME alone says key is no declared source, a predicate the walk cannot see
        ('delete_keyed(c, "{}", by={pk: i})', []),
        ('delete_keyed(c, "{}", by={pk_column: i})', ["unknown-schema"]),
        ('read_row(c, "{}", by={self._pk_column: i})', ["unknown-schema"]),
        # a write pinning a column named at run time that says nothing of the key (merge.repoint_user_rows' owner
        # column): a predicate the walk cannot see on a table it does not know
        ('update_keyed(c, "{}", values=v, by={user_column: u}, returning=pk)', ["unknown-schema"]),
        ('read_row(c, "results", by={column: v})', ["non-key-where"]),
        # every key column of a table the caller reads at run time, each bound to the row's value (SelectiveRestore's
        # live twin): keyed when the walk hands ``pk`` (a declared source), as SQL over {pk} would be; a list named
        # ``pk_columns`` the walk has not traced to one is not
        ('read_row(c, "{}", by=dict(zip(pk, key, strict=True)))', []),
        ('read_row(c, "{}", by=dict(zip(pk_columns, key, strict=True)))', ["unknown-schema"]),
        ('read_row(c, "results", by=dict(zip(columns, values)))', ["non-key-where"]),
        # a generic store's row by the primary key mapping its caller hands (SqlL3Backend.fetch_one / delete with no
        # schema registered; a generated collection's fetch_from_store): keyed, as its SQL over {table} / {pk} would
        # be; a mapping whose name says nothing of the key is a predicate the walk cannot see
        ("read_row(c, table, by=dict(pk))", []),
        ("delete_keyed(c, table, by=dict(pk))", []),
        ("read_row(c, tbl_name, by={pk: entity_id})", []),
        ("read_row(c, tbl_name, by={pk_column: entity_id})", ["unknown-schema"]),
        ("read_row(c, table, by=dict(filters))", ["unknown-schema"]),
        ('read_row(c, "results", by=dict(values))', ["non-key-where"]),
    ],
)
def test_a_by_column_named_at_run_time_is_judged_as_its_interpolation(
    code: str, expected: list[str], schema: SchemaCatalog
) -> None:
    """Flips if a ``by`` column the walk hands as the declared key (``pk``) stays unknown where SQL interpolating it
    would be judged, or a column named at run time passes on its name alone (``pk_column``, ``self._pk_column``)."""
    assert [[kind for kind, _ in findings] for findings in _judged(code, schema)] == [expected]


def test_an_ordered_step_is_judged_by_the_step_it_orders() -> None:
    """Flips if an ordered step over another step sends a statement of its own (the step it orders is judged where it
    is written), or one over a relation reads nothing."""
    assert list(helper_statements(_call('ordered(scan(c, "results"), [Sort("geo")], limit=3)'))) == []
    assert [s.sql for s in helper_statements(_call('ordered(change, [Sort("geo")])'))] == [
        "SELECT * FROM {change} AS t"
    ]


def test_any_other_call_sends_nothing() -> None:
    assert list(helper_statements(_call('fetch("SELECT 1")'))) == []
    assert list(helper_statements(_call('empty_tables(c, ["results"])'))) == []


def test_the_acl_visibility_builders_are_keyed_fragments() -> None:
    assert is_keyed_fragment(
        _call("caller_visible_customer_clause(user_id=u, customer_id_column=x, scope_namespace_type=n)")
    )
    assert not is_keyed_fragment(_call("build_clause(u)"))


@pytest.mark.parametrize(
    ("code", "narrowed"),
    [
        ("keyset_pages(c, head, key, page_size=10, after=last)", False),
        ("read_l3_rows(c, t, cols, key, since=mark)", False),
        ('keyset_pages(c, head, key, page_size=10, where={"state": s}, after=last)', True),
        ("read_l3_rows(c, t, cols, key, where_any=sets)", True),
    ],
)
def test_where_a_pager_starts_does_not_narrow_it(code: str, narrowed: bool) -> None:
    """Flips if a drain narrowed only by ``after=`` (every row from a place on) counts as keyed."""
    assert narrows(_call(code), DEFAULT_NARROWING_KEYWORDS) is narrowed


def test_a_reap_is_judged_as_the_paged_update_it_sends(schema: SchemaCatalog) -> None:
    """Flips if a paged update passes on a column no key or purposed index leads, or fails on a key prefix."""
    keyed = (
        'update_before("leases", collection=c, key=["purpose", "key"], column="expires_at", before=t, set_to={"purpose": x}, '
        'page=5, by={"purpose": p})'
    )
    assert _judged(keyed, schema) == [[]]
    unkeyed = (
        'update_before("tenant_leases", collection=c, key=["customer_id", "key"], column="expires_at", before=t, '
        'set_to={"customer_id": x}, page=5, by={"customer_id": p})'
    )
    assert _judged(unkeyed, schema, "src/app.py::elsewhere") != [[]]
    assert [
        s.unseen for s in helper_statements(_call('update_before("leases", collection=c, key=k, column="e", set_to=s)'))
    ] == ["key"]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # a generic reader's keyed batch over the table it is handed, by that table's key: keyed when the walk hands
        # the helper ``pk`` (the columns flow from a declared key source), as its SQL would be
        ("read_keyed_arrow(l3, table.name, keys=KeyBatch(pk, chunk, casts), types=types)", []),
        ("delete_keyed(c, rows_stage, keys=KeyBatch(tuple(pk), named, as_list=True))", []),
        # columns the walk has not traced to a declared source, whatever they are named: a predicate it cannot see
        ("read_keyed_arrow(l3, table.name, keys=KeyBatch(table.columns, chunk), types=types)", ["unknown-predicate"]),
        (
            "read_keyed_arrow(l3, table.name, keys=KeyBatch(table.key, chunk, casts), types=types)",
            ["unknown-predicate"],
        ),
        ("delete_keyed(c, rows_stage, keys=KeyBatch(tuple(key), named, as_list=True))", ["unknown-schema"]),
        ("read_keyed(c, table, keys=KeyBatch(list(self.key_columns), ks))", ["unknown-predicate"]),
        ("read_keyed(c, table, keys=KeyBatch(tuple(columns), ks))", ["unknown-predicate"]),
    ],
)
def test_a_generic_keyed_batch_is_judged_as_its_interpolated_sql(
    code: str, expected: list[str], schema: SchemaCatalog
) -> None:
    """Flips if a generic reader's batch by its table's key is refused, or one by another column passes."""
    assert [[kind for kind, _ in findings] for findings in _judged(code, schema)] == [expected]


def test_a_page_s_filters_are_judged_with_it(schema: SchemaCatalog) -> None:
    """Flips if a page's literal where= fragments are dropped from the statement judged, or written outside their own
    parentheses (as the helper writes them)."""
    code = (
        'keyset_page(c, "results", columns=cols, by={"state": s}, where=[Fragment(\'t."geo" <= $1\', (g,))], '
        'order=[Sort("geo"), Sort("id")], limit=5)'
    )
    (statement,) = [s.sql for s in helper_statements(_call(code)) if s.sql]
    assert '(t."geo" <= $2)' in statement
    assert _judged(code, schema) == [[]]


@pytest.mark.parametrize(
    "call",
    [
        'literal_select("results", columns=cols, by={by}, where={where})',
        'read_keyed(c, "results", keys=KeyBatch(("state",), ks), columns=cols, by={by}, where={where})',
        'counts_by(c, "results", column="geo", by={by}, where={where})',
        'keyset_page(c, "results", columns=cols, by={by}, where={where}, order=[Sort("geo")], limit=5)',
    ],
)
@pytest.mark.parametrize(
    "where",
    ["where", "[Fragment(condition, tuple(params))]", "[*narrowing, Fragment(text)]"],
)
def test_a_run_time_where_only_narrows_so_the_call_is_judged_by_its_pins(
    call: str, where: str, schema: SchemaCatalog
) -> None:
    """Flips if a where= the walk cannot read is named unknown-predicate, or lets a call pinning nothing pass: every
    helper ANDs each fragment after its pins in its own parentheses (``keyed._filters.narrowing``), so a run-time
    fragment only narrows, and the call is keyed exactly when its by= is."""
    keyed = _call(call.replace("{by}", '{"state": s}').replace("{where}", where))
    statements = list(helper_statements(keyed))
    assert statements and not any(s.unseen for s in statements), [s.unseen for s in statements]
    # a keyed count is named so (a count by a key prefix); anything else keyed is clean
    keyed_kinds = {"keyed-count"}
    judged = _judged(ast.unparse(keyed), schema)
    assert judged and all(found[0] in keyed_kinds for findings in judged for found in findings), judged
    if call.startswith("read_keyed"):
        return  # its key batch keys it whatever it pins
    whole = ast.unparse(_call(call.replace("{by}", "{}").replace("{where}", where)))
    assert any(found[0] not in keyed_kinds for findings in _judged(whole, schema) for found in findings), whole


def test_joined_rows_are_judged_as_the_keyed_read_they_send(schema: SchemaCatalog) -> None:
    """Flips if a joined read by a non-key column passes, or one by the key fails."""
    assert _judged('read_joined(c, rows, "results", on={"state": "s", "geo": "g"}, columns=cols)', schema) == [[]]
    assert _judged('read_joined(c, rows, "results", on={"geo": "g"}, columns=cols)', schema) != [[]]


@pytest.mark.parametrize(
    "code",
    [
        # a bounded probe of a table's first rows: at most its bound, bound as a parameter
        'count_at_most(c, "results", at_most=50, order=["state", "geo", "id"])',
        # a cache's trim by SQLite's rowid, the key its b-tree is stored by
        'delete_oldest(c, "results", count=n)',
        # several relations counted in one statement, one of them derived and judged on its own text
        'counts_by_relations(c, {"a": "results", "b": Derived("SELECT state, geo FROM results WHERE state = \'TX\'")}, '
        'column="geo", values=AnyOf(v), by={"state": s})',
        # a schema-qualified table, and a derived relation read by key
        'read_after(c, "public.results", columns=cols, id_column="id", after=0, limit=3, by={"state": s, "geo": g})',
        'read_keyed(c, Derived("SELECT * FROM results WHERE state = \'TX\'"), keys=KeyBatch(("state", "geo"), ks), '
        "columns=cols)",
    ],
)
def test_bounded_probes_rowid_trims_and_relations_are_keyed(code: str, schema: SchemaCatalog) -> None:
    """Flips if a bounded probe, a rowid trim, a multi-relation count or a derived/qualified relation is refused."""
    judged = _judged(code, schema)
    assert judged and all(not [kind for kind, _ in findings if kind != "keyed-count"] for findings in judged), judged


def test_a_derived_relation_reading_whole_is_named(schema: SchemaCatalog) -> None:
    """Flips if a derived relation is trusted without its own text being judged."""
    judged = _judged('holds_any(c, Derived("SELECT state FROM results WHERE kind = \'x\'"), order=["state"])', schema)
    assert any(findings for findings in judged), judged


def test_a_pinned_expression_is_judged_against_the_expression_index_holding_it(schema: SchemaCatalog) -> None:
    """Flips if a read_by pinning an expression (Pin.expr) is not keyed by the purposed index on that expression, or
    is keyed outside the read its purpose names, or an expression of another shape passes."""
    pinned = 'read_by(c, "nodes", by={Pin.expr("metadata->>\'pod\'"): pod})'
    assert _judged(pinned, schema) == [[]]
    assert _judged('read_by(c, "nodes", by={Pin.expr("t.\\"metadata\\"->>\'pod\'"): pod, "kind": k})', schema) == [[]]
    assert _judged(pinned, schema, "src/app.py::elsewhere") == [[("non-key-where", "nodes")]]
    assert _judged('read_by(c, "nodes", by={Pin.expr("lower(kind)"): v})', schema) == [[("non-key-where", "nodes")]]


@pytest.mark.parametrize(
    ("builder", "runner"),
    [
        (
            'read_row_statement("results", by={"state": s, "geo": g}, columns=cols)',
            'read_row(c, "results", by={"state": s, "geo": g}, columns=cols)',
        ),
        ('read_by_statement("results", by={"kind": k})', 'read_by(c, "results", by={"kind": k})'),
        (
            'read_keyed_statements("results", keys=KeyBatch(("geo",), ks), columns=cols)',
            'read_keyed(c, "results", keys=KeyBatch(("geo",), ks), columns=cols)',
        ),
        (
            'keyset_page_statement("results", columns=cols, by={"state": s}, order=[Sort("geo"), Sort("id")], limit=5)',
            'keyset_page(c, "results", columns=cols, by={"state": s}, order=[Sort("geo"), Sort("id")], limit=5)',
        ),
        (
            'keyset_rows_statement("results", columns=cols, by={}, order=[Sort("kind"), Sort("id")], page=500)',
            'keyset_rows(c, "results", columns=cols, by={}, order=[Sort("kind"), Sort("id")], page=500)',
        ),
        (
            'delete_keyed_statements("unindexed", prefix={"name": p})',
            'delete_keyed(c, "unindexed", prefix={"name": p})',
        ),
        (
            'update_keyed_statements("unindexed", values={"run_id": r}, prefix={"name": p})',
            'update_keyed(c, "unindexed", values={"run_id": r}, prefix={"name": p})',
        ),
        (
            'delete_before_statement("tenant_leases", key=["customer_id", "key"], column="expires_at", before=t, '
            'page=5, by={"customer_id": p})',
            'delete_before("tenant_leases", collection=c, key=["customer_id", "key"], column="expires_at", '
            'before=t, page=5, by={"customer_id": p})',
        ),
        (
            'update_before_statement("leases", key=["purpose", "key"], column="expires_at", before=t, '
            'set_to={"purpose": q}, page=5, by={"purpose": p})',
            'update_before("leases", collection=c, key=["purpose", "key"], column="expires_at", before=t, '
            'set_to={"purpose": q}, page=5, by={"purpose": p})',
        ),
    ],
)
def test_a_builder_s_call_is_judged_as_its_runner_s(builder: str, runner: str, schema: SchemaCatalog) -> None:
    """Flips if a pure builder's call (threetears.interfaces.keyed, run on a caller's own connection) is judged
    otherwise than its runner's: keyed where the runner's is, the same whole read where it is not."""
    assert [statement.sql for statement in helper_statements(_call(builder))] == [
        statement.sql for statement in helper_statements(_call(runner))
    ]
    assert _judged(builder, schema, _SWEEP) == _judged(runner, schema, _SWEEP)


def test_a_builder_naming_no_key_is_a_whole_read(schema: SchemaCatalog) -> None:
    """Flips if a builder's call passes on its name alone (a point read pinning a column no key leads with)."""
    judged = _judged('read_row_statement("results", by={"kind": k})', schema)
    assert any(found[0] == "non-key-where" for findings in judged for found in findings), judged


@pytest.mark.parametrize(
    ("code", "argument"),
    [
        ('page_statement("results", cols, {}, order, 6, None, False, (), (), None)', "order"),
        ('keyed_read_statement("results", cols, KeyBatch(("geo",), ks), {})', "keys"),
    ],
)
def test_a_runners_positional_builder_called_elsewhere_is_unseen(code: str, argument: str) -> None:
    """Flips if a builder the runners call positionally (a page's statement, a keyed batch's), called outside the
    keyed packages, passes unjudged rather than being named unknown-predicate."""
    statements = list(helper_statements(_call(code)))
    assert [(statement.sql, statement.unseen) for statement in statements] == [(UNSEEN, argument)]


@pytest.mark.parametrize(
    "code",
    [
        'apply_stage(c, f"main.{name}", "_chunk", key=key)',
        'update_from(c, f"main.{name}", source=Derived("SELECT id FROM _chunk"), key=["id"])',
    ],
)
def test_a_stage_or_source_target_written_as_an_f_string_is_the_relation_it_names(code: str) -> None:
    """Flips if apply_stage's / update_from's target written as an f-string (``f"main.{name}"``: a local copy named at
    run time, R9) is written as an unknown ``"{}"`` instead of the relation it names, the default key every copy
    declares (``local_relations["*"]``), as the same f-string handed to any other helper is."""
    written = [statement.sql for statement in helper_statements(_call(code))]
    assert any(f"{COPY_PREFIX}{{}}" in sql and sql.startswith(("DELETE", "UPDATE")) for sql in written), written


def test_a_keyed_aggregate_over_a_prefix_and_a_range_of_the_next_column_is_keyed(schema: SchemaCatalog) -> None:
    """Flips if aggregate_keyed over the key's lead and a range of its next column (a set of states, a window of
    geographies) is refused, or one whose range leads no index (expires_at alone) is passed."""
    code = (
        'aggregate_keyed(c, "results", aggregates={"n": Count(), "s": Sum("t.id")}, by={"state": AnyOf(ss)}, '
        'ranges=[Range("geo", lo, hi), Range("geo", lo2, hi2)], group=[Selected("t.kind", "kind")])'
    )
    judged = _judged(code, schema, "src/app.py::elsewhere")
    assert judged and all(kind not in {"whole-select", "non-key-where", "whole-aggregate"} for kind, _ in judged[0])
    whole = 'aggregate_keyed(c, "tenant_leases", aggregates={"n": Count()}, ranges=[Range("expires_at", high=t)])'
    assert _judged(whole, schema, _SWEEP) != [[]]


def test_a_keyed_aggregate_through_an_index_is_keyed_only_for_the_caller_it_names(schema: SchemaCatalog) -> None:
    """Flips if a tenant's window (customer_id, expires_at) is keyed outside the index's purpose, or not inside it."""
    code = (
        'aggregate_keyed(c, "tenant_leases", aggregates={"n": Count()}, by={"customer_id": p}, '
        'ranges=[Range("expires_at", lo, hi, high_inclusive=True)])'
    )
    inside = _judged(code, schema, _SWEEP)
    assert inside and all(kind not in {"whole-select", "non-key-where", "whole-aggregate"} for kind, _ in inside[0])
    assert any(kind in {"whole-select", "non-key-where", "whole-aggregate"} for kind, _ in _judged(code, schema)[0])


@pytest.mark.parametrize(
    "code",
    [
        'aggregate_keyed(c, "results", aggregates={"n": Count()}, by={"state": s}, ranges=rs)',
        'aggregate_keyed(c, "results", aggregates={"n": Count()}, by={"state": s}, ranges=[Range(col, lo)])',
        'aggregate_keyed(c, "results", aggregates={"n": Count()}, by={"state": s}, ranges=[Range("geo", lo), Range("id", lo)])',
        'aggregate_keyed(c, "results", aggregates={"n": Count()}, by={"state": s}, ranges=[r])',
    ],
)
def test_a_keyed_aggregate_whose_ranges_the_walk_cannot_read_is_unseen(code: str) -> None:
    """Flips if ranges named at run time, a run-time range column, or ranges of two columns are passed on a guess."""
    statements = list(helper_statements(_call(code)))
    assert [(statement.sql, statement.unseen) for statement in statements] == [(UNSEEN, "ranges")]


@pytest.mark.parametrize(
    ("code", "found"),
    [
        # a correlated read of another table by its key: keyed, as the hand-written select list's was
        (
            'read_row(c, "results", by={"state": s, "geo": g, "id": i}, '
            'columns=("kind", Selected("(SELECT f.run_id FROM fence f WHERE f.signal = t.kind)", "run")))',
            [],
        ),
        # one reading another table by no key: found, as it was in the hand-written statement
        (
            'keyset_page(c, "results", by={"state": s}, order=[Sort("geo"), Sort("id")], limit=5, '
            'columns=("kind", Selected("(SELECT count(*) FROM unindexed u WHERE u.name = t.kind)", "n")))',
            [("whole-aggregate", "unindexed")],
        ),
    ],
)
def test_a_selected_sub_select_is_judged_with_its_read(
    code: str, found: list[tuple[str, str]], schema: SchemaCatalog
) -> None:
    """Flips if a ``Selected`` expression in a read's literal ``columns=`` is dropped from the statement judged: a
    sub-select there reads another table, and the walk must see it (and credit the index it uses) as it saw it in the
    hand-written statement the helper replaced."""
    (statement,) = [s.sql for s in helper_statements(_call(code)) if s.sql]
    assert " AS run" in statement or " AS n" in statement, statement
    assert _judged(code, schema)[0] == found
