"""the keyed-read helpers, judged by the statements a call to one sends: keyed by construction, never on trust.

:mod:`threetears.core.keyed` writes each shape of keyed read once (a keyed batch, a listing's page, a bounded count, a
change record past a mark, an expiry sweep). The helper builds its SQL at run time from the table and columns its
caller names, so the walk sees no statement in it. What the walk can see is the call: the helper's name and the
literal table, key, ``by``, sort and column its caller passes. From those, :func:`helper_statements` writes the
statement the helper will send, in its canonical shape, and the walk judges that statement under the caller's own
name (:func:`~threetears.enforcement.no_whole_reads.sql.judge_sql`), so a purposed index names the caller as it would
for SQL the caller wrote. A helper call is keyed when the read it makes is keyed, by the same rule, against the same
declared keys.

A call whose table, key, ``by``, sort or column is not a literal the walk can read (a name bound at run time, a
comprehension) answers :data:`UNSEEN`: the walk reports it ``unknown-predicate``, never a pass. One exception, as
for SQL a caller writes: a keyed batch over a table a generic reader is handed (``read_keyed(l3, table.name,
keys=KeyBatch(table.key, ...))``, its table and its key both names) is written with the names as interpolations
(``{name}``, ``{key}``), and the rule reads a key named as one (``{key}``, ``{pk}``) as the key, and any other as a
predicate it cannot see. ``empty_tables`` and
``estimated_rows`` read one row of each table, or the catalog, whatever they are passed: nothing of theirs is judged.

Visibility: a listing's ``visible=`` clauses are judged each on its own, as the correlated probe it is
(:func:`~threetears.core.keyed.exists_keyed`'s ``EXISTS``, its outer columns bound). A clause built by one of
:data:`KEYED_FRAGMENT_BUILDERS` (the ACL's caller-visibility ``EXISTS``, reviewed once where it is written) is a keyed
predicate wherever it is placed.

Pagers: ``after=`` and ``since=`` say where a read starts, not which rows it reads. A pager narrowed by them alone
reads every row from that place on: :func:`narrows` counts them only beside a ``where`` / ``where_any`` / ``keys``.

**Wiring** (the walker, where it visits a call)::

    for statement in helper_statements(call):
        if statement.sql is None:
            hit("unknown-predicate", statement.helper)      # the helper's arguments are not literal
        else:
            judge(statement.sql)                             # as SQL written in the calling function
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from threetears.enforcement.no_whole_reads.schema import PRODUCED_PREFIX

__all__ = [
    "KEYED_FRAGMENT_BUILDERS",
    "KEYED_HELPER_CALLS",
    "START_KEYWORDS",
    "STEP_CALLS",
    "STEP_RELATIONS",
    "UNSEEN",
    "HelperStatement",
    "helper_statements",
    "is_keyed_fragment",
    "narrows",
]

#: the helpers judged by the statement a call sends; each reads (or deletes) only what its arguments key
KEYED_HELPER_CALLS: Final[frozenset[str]] = frozenset(
    {
        "read_keyed",
        "read_keyed_arrow",
        "read_by",
        "read_row",
        "read_value",
        "read_joined",
        "any_rows",
        "keyset_page",
        "keyset_rows",
        "copy_keyed",
        "read_changed",
        "group_hashes",
        "counts_by",
        "count_at_most",
        "holds_any",
        "read_after",
        "read_past_marks",
        "literal_select",
        "delete_before",
        "delete_oldest",
        "delete_keyed",
        "update_keyed",
        "counts_by_relations",
        "update_before",
        "apply_stage",
        "update_from",
    }
)

#: the relational steps over change relations (:mod:`threetears.core.keyed.steps`), judged by the relations a call
#: names: each one read whole, as the step reads it, so a step over a change relation (one a module registers or
#: declares with ``change_relation``) passes and one over a data table is the whole read it is; a step a step reads
#: is judged at its own call. Judged only where the call is the keyed helper's (imported from ``threetears.core.keyed``)
STEP_CALLS: Final[frozenset[str]] = frozenset(
    {
        "scan",
        "semi",
        "anti",
        "distinct",
        "grouped",
        "latest_per_key",
        "windowed",
        "except_rows",
        "union",
        "read_step",
        "read_step_arrow",
        "count_step",
        "delete_matching",
        "first_difference",
    }
)

#: where each step names the relations it reads: a positional index or a keyword; ``*`` every positional
STEP_RELATIONS: Final[Mapping[str, tuple[tuple[int | None, str], ...]]] = {
    "scan": ((0, "rel"),),
    "semi": ((0, "rel"), (1, "keys")),
    "anti": ((0, "rel"), (1, "keys")),
    "distinct": ((0, "rel"),),
    "grouped": ((0, "rel"),),
    "latest_per_key": ((0, "rel"),),
    "windowed": ((0, "rel"),),
    "except_rows": ((0, "rel"), (1, "other")),
    "read_step": ((1, "step"),),
    "read_step_arrow": ((1, "step"),),
    "count_step": ((1, "step"),),
    "delete_matching": ((2, "keys"),),
    "first_difference": ((1, "rel"), (2, "other")),
}

#: builders of a visibility clause reviewed once where it is written: a keyed predicate wherever it is placed
KEYED_FRAGMENT_BUILDERS: Final[frozenset[str]] = frozenset(
    {
        "caller_visible_customer_clause",
        "customer_scope_visibility_clause",
        "three_scope_visibility_clause",
    }
)

#: methods building a reviewed visibility fragment for a keyed read's ``visible=``
#: (``threetears.agent.acl.VisibleCustomers.fragment``): the caller's customers bound as a set, a keyed predicate
_KEYED_FRAGMENT_METHODS: Final[frozenset[str]] = frozenset({"fragment"})

#: keywords that say where a pager starts, not which rows it reads
START_KEYWORDS: Final[frozenset[str]] = frozenset({"after", "since"})

#: what :attr:`HelperStatement.sql` is when the walk cannot read the call's arguments
UNSEEN: Final = None

#: the rows a synthesized page or probe asks for: any bound; the rule reads only that one is there
_LIMIT: Final = 100


@dataclass(frozen=True)
class HelperStatement:
    """one statement a helper call sends, as the walk judges it.

    :ivar helper: the helper's name (``read_keyed``)
    :ivar sql: the statement in the helper's canonical shape; :data:`UNSEEN` when an argument it needs is not a
        literal the walk can read
    :ivar unseen: which argument was not read, when ``sql`` is :data:`UNSEEN`
    """

    helper: str
    sql: str | None
    unseen: str = ""


class _Unseen(Exception):
    """an argument the walk cannot read as a literal."""

    def __init__(self, argument: str) -> None:
        super().__init__(argument)
        self.argument = argument


def _name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _argument(call: ast.Call, keyword: str, position: int | None) -> ast.expr | None:
    for given in call.keywords:
        if given.arg == keyword:
            return given.value
    if (
        position is not None
        and len(call.args) > position
        and not any(isinstance(arg, ast.Starred) for arg in call.args[: position + 1])
    ):
        return call.args[position]
    return None


def _text(node: ast.expr | None, argument: str) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise _Unseen(argument)


def _texts(node: ast.expr | None, argument: str) -> list[str]:
    if isinstance(node, ast.Tuple | ast.List):
        return [_text(element, argument) for element in node.elts]
    raise _Unseen(argument)


def _by(node: ast.expr | None, argument: str = "by") -> list[tuple[str, str]]:
    """each pinned column and how it is bound: ``=``, ``= ANY`` (an ``AnyOf``) or ``IS NULL``."""
    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return []
    if not isinstance(node, ast.Dict) or any(key is None for key in node.keys):
        raise _Unseen(argument)
    pinned = []
    for key, value in zip(node.keys, node.values, strict=True):
        column = _text(key, argument)
        if isinstance(value, ast.Call) and _name(value) == "AnyOf":
            pinned.append((column, "any"))
        elif isinstance(value, ast.Constant) and value.value is None:
            pinned.append((column, "null"))
        else:
            pinned.append((column, "eq"))
    return pinned


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _conditions(pinned: Sequence[tuple[str, str]], *, first: int = 1, alias: str = "") -> tuple[list[str], int]:
    prefix = f"{alias}." if alias else ""
    conditions = []
    place = first
    for column, how in pinned:
        if how == "null":
            conditions.append(f"{prefix}{_quoted(column)} IS NULL")
            continue
        conditions.append(
            f"{prefix}{_quoted(column)} = ANY(${place})" if how == "any" else f"{prefix}{_quoted(column)} = ${place}"
        )
        place += 1
    return conditions, place


def _named(node: ast.expr | None, argument: str) -> str:
    """a relation or key the caller names: a literal, quoted; a name or attribute, as the interpolation it would be in
    SQL the caller wrote (``{table}``, ``{key}``), so the rule reads it as it reads that SQL (a name that says it is
    the key binds the key; any other is a predicate it cannot see)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _quoted(node.value)
    if isinstance(node, ast.Attribute):
        return "{" + node.attr + "}"
    if isinstance(node, ast.Name):
        return "{" + node.id + "}"
    raise _Unseen(argument)


def _key_batch_columns(node: ast.expr | None) -> list[str]:
    """the key columns a ``KeyBatch`` names, each as the statement spells it."""
    if isinstance(node, ast.Call) and _name(node) == "KeyBatch":
        columns = _argument(node, "columns", 0)
        if isinstance(columns, ast.Tuple | ast.List):
            return [_quoted(column) for column in _texts(columns, "keys")]
        if isinstance(columns, ast.Attribute | ast.Name):
            return [_named(columns, "keys")]
    raise _Unseen("keys")


def _sort_columns(node: ast.expr | None) -> list[str]:
    """each sort's term as the page's statement orders by it: the column, quoted, or the sort's expression."""
    if not isinstance(node, ast.Tuple | ast.List) or not node.elts:
        raise _Unseen("order")
    columns = []
    for element in node.elts:
        if not (isinstance(element, ast.Call) and _name(element) == "Sort"):
            raise _Unseen("order")
        expression = _argument(element, "expression", None)
        if expression is None:
            columns.append(_quoted(_text(_argument(element, "column", 0), "order")))
        else:
            # a sort by an expression of the paged row's columns (``lower(t."name")``), as an expression index reads it
            written = re.sub(r'\bt\.(?="|\w)', "", _text(expression, "order"))
            columns.append(re.sub(r'"(\w+)"', r"\1", written))
    return columns


def _where(conditions: Sequence[str]) -> str:
    return f" WHERE {' AND '.join(conditions)}" if conditions else ""


def _read_keyed(call: ast.Call) -> list[str]:
    table_node = _argument(call, "table", 1)
    keys = _key_batch_columns(_argument(call, "keys", None))
    own: list[str] = []
    if isinstance(table_node, ast.Constant) or (isinstance(table_node, ast.Call) and _name(table_node) == "Derived"):
        table, own = _relation(table_node, "table")
    elif keys[0].startswith("{"):
        # a generic reader's own table by its own key: judged as the interpolated SQL it would write
        table = _named(table_node, "table")
    else:
        # an interpolated table read by literal columns: the walk cannot match the columns to any table's key
        raise _Unseen("table")
    pinned, _ = _conditions(_by(_argument(call, "by", None)), first=len(keys) + 1, alias="t")
    unnested = ", ".join(f"${index + 1}" for index in range(len(keys)))
    named = ", ".join(f"k{index}" for index in range(len(keys)))
    joined = " AND ".join(f"t.{column} = k.k{index}" for index, column in enumerate(keys))
    return [f"SELECT * FROM {table} AS t JOIN unnest({unnested}) AS k({named}) ON {joined}{_where(pinned)}", *own]


def _read_by(call: ast.Call) -> list[str]:
    """a point read (``read_by``, ``read_row``, ``read_value``): the rows its ``by`` pins, each ``= $n``; one pinning
    nothing is the whole read it is."""
    table, own = _relation(_argument(call, "table", 1), "table")
    pinned, _ = _conditions(_by(_argument(call, "by", None)), alias="t")
    return [f"SELECT * FROM {table} AS t{_where(pinned)}", *own]


def _literal_select(call: ast.Call) -> list[str]:
    """a keyed read written out with literals (``literal_select``, an export's ``SELECT``): the rows its ``by`` pins
    (``=`` a value, ``IN`` a set) and its ``where`` narrows, as the bound read it writes out; one pinning nothing is
    the whole read it is."""
    table, own = _relation(_argument(call, "relation", 0), "relation")
    pinned, place = _conditions(_by(_argument(call, "by", None)), alias="t")
    filtered, _ = _filters(call, place)
    return [f"SELECT * FROM {table} AS t{_where([*pinned, *filtered])}", *own]


def _visible(call: ast.Call) -> list[str]:
    """each ``exists_keyed`` probe a listing's ``visible=`` names, as the keyed read it is; reviewed builders (and a
    ``VisibleCustomers.fragment(...)``, the caller's customers as a bound set) pass."""
    node = _argument(call, "visible", None)
    if node is None:
        return []
    if not isinstance(node, ast.Tuple | ast.List):
        raise _Unseen("visible")
    probes = []
    for element in node.elts:
        built = (
            element.args[0] if isinstance(element, ast.Call) and _name(element) == "of" and element.args else element
        )
        if isinstance(built, ast.Call) and _name(built) in KEYED_FRAGMENT_BUILDERS:
            continue
        if (
            isinstance(built, ast.Call)
            and isinstance(built.func, ast.Attribute)
            and built.func.attr in _KEYED_FRAGMENT_METHODS
        ):
            continue
        if not (isinstance(built, ast.Call) and _name(built) == "exists_keyed"):
            raise _Unseen("visible")
        probed = _text(_argument(built, "table", 0), "visible")
        on = _by(_argument(built, "on", None), "visible")
        if not on:
            raise _Unseen("visible")
        conditions, _ = _conditions([(column, "eq") for column, _ in on] + _by(_argument(built, "by", None), "visible"))
        probes.append(f"SELECT 1 FROM {_quoted(probed)}{_where(conditions)}")
    return probes


def _visible_columns(call: ast.Call) -> list[str]:
    """the columns a listing's ``visible=`` binds to the caller's customers (``customers.fragment('t."customer_id"')``,
    a :class:`threetears.agent.acl.VisibleCustomers` fragment): each a set the page is narrowed to, as its statement
    reads it (``"customer_id" = ANY($n)``)."""
    node = _argument(call, "visible", None)
    if not isinstance(node, ast.Tuple | ast.List):
        return []
    columns = []
    for element in node.elts:
        if (
            isinstance(element, ast.Call)
            and isinstance(element.func, ast.Attribute)
            and element.func.attr in _KEYED_FRAGMENT_METHODS
        ):
            written = _text(_argument(element, "column", 0), "visible")
            columns.append(written.split(".")[-1].strip('"'))
    return columns


def _filters(call: ast.Call, first: int) -> tuple[list[str], int]:
    """a page's ``where=`` fragments as written (``Fragment("t.due_at <= $1", (now,))``), renumbered past ``first``."""
    node = _argument(call, "where", None)
    if node is None:
        return [], first
    if not isinstance(node, ast.Tuple | ast.List):
        raise _Unseen("where")
    written = []
    place = first
    for element in node.elts:
        if not (isinstance(element, ast.Call) and _name(element) == "Fragment"):
            raise _Unseen("where")
        text = _text(_argument(element, "sql", 0), "where")
        numbers = [int(found) for found in re.findall(r"\$(\d+)", text)]
        written.append(re.sub(r"\$(\d+)", lambda match: f"${int(match.group(1)) + place - 1}", text))
        place += max(numbers, default=0)
    return written, place


def _keyset_page(call: ast.Call) -> list[str]:
    table, own = _relation(_argument(call, "table", 1), "table")
    order = _sort_columns(_argument(call, "order", None))
    pinned, place = _conditions(
        _by(_argument(call, "by", None)) + [(column, "any") for column in _visible_columns(call)]
    )
    filtered, place = _filters(call, place)
    page = [*pinned, *filtered, f"{order[0]} > ${place}"]
    ordered = ", ".join(order)
    return [f"SELECT * FROM {table} AS t{_where(page)} ORDER BY {ordered} LIMIT {_LIMIT}", *_visible(call), *own]


def _read_changed(call: ast.Call) -> list[str]:
    """phase 1, the scope's keys and hashes, is a keyset page in the key's order (``relation`` / ``key`` as
    ``keyset_rows`` takes ``table`` / ``order``; a key column named by its string a sort of it); phase 2 reads by
    those keys, keyed whenever phase 1 is."""
    key = _argument(call, "key", None)
    if isinstance(key, ast.Tuple | ast.List):
        key = ast.List(
            elts=[
                ast.Call(func=ast.Name("Sort", ast.Load()), args=[element], keywords=[])
                if isinstance(element, ast.Constant)
                else element
                for element in key.elts
            ],
            ctx=ast.Load(),
        )
    page = ast.Call(
        func=call.func,
        args=[call.args[0]] if call.args else [],
        keywords=[
            *(keyword for keyword in call.keywords if keyword.arg not in {"relation", "key"}),
            ast.keyword(arg="table", value=_argument(call, "relation", 1) or ast.Name("_unseen", ast.Load())),
            *([ast.keyword(arg="order", value=key)] if key is not None else []),
        ],
    )
    return _keyset_page(page)


def _group_hashes(call: ast.Call) -> list[str]:
    """every scope's group hashes (``group_hashes``): the scopes' rows read to hash them, bound by ``scope = ANY``
    (``scopes``) and the pins and filters every scope shares, so keyed only where the scope column leads a key (or the
    pins before it do)."""
    relation, own = _relation(_argument(call, "relation", 1), "relation")
    scope = _argument(call, "scope", None)
    if isinstance(scope, ast.Call) and _name(scope) == "Sort":
        scope = _argument(scope, "column", 0)
    column = _text(scope, "scope")
    if _argument(call, "scopes", None) is None:
        raise _Unseen("scopes")
    pinned, place = _conditions(_by(_argument(call, "by", None)), alias="t")
    conditions = [*pinned, f"t.{_quoted(column)} = ANY(${place})"]
    filtered, _ = _filters(call, place + 1)
    return [f"SELECT * FROM {relation} AS t{_where([*conditions, *filtered])}", *own]


def _read_joined(call: ast.Call) -> list[str]:
    table = _named(_argument(call, "table", 2), "table")
    on = _by(_argument(call, "on", None), "on")
    if not on:
        raise _Unseen("on")
    pinned, _ = _conditions(_by(_argument(call, "by", None)), first=len(on) + 1, alias="t")
    unnested = ", ".join(f"${index + 1}" for index in range(len(on)))
    named = ", ".join(f"k{index}" for index in range(len(on)))
    joined = " AND ".join(f"t.{_quoted(column)} = k.k{index}" for index, (column, _) in enumerate(on))
    return [f"SELECT * FROM {table} AS t JOIN unnest({unnested}) AS k({named}) ON {joined}{_where(pinned)}"]


def _relation(node: ast.expr | None, argument: str) -> tuple[str, list[str]]:
    """a relation a helper reads, as its statement names it, and the statements it holds of its own: a table literal
    (``schema.table`` quoted a segment at a time), or a ``Derived("SELECT ...")`` judged on its own text."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(PRODUCED_PREFIX):
        # a declared producer's run-time relation (the walk wrote it): the catalog's own name for it, as declared
        return node.value, []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ".".join(_quoted(part) for part in node.value.split(".")), []
    if isinstance(node, ast.Call) and _name(node) == "Derived":
        derived = _text(_argument(node, "sql", 0), argument)
        return f"({derived})", [derived]
    raise _Unseen(argument)


def _count_of(relation: str, column: str, call: ast.Call, *, values: bool) -> str:
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    conditions = [*pinned]
    if values:
        conditions.append(f"{_quoted(column)} = ANY(${place})")
        place += 1
    filtered, _ = _filters(call, place)
    conditions += filtered
    return f"SELECT {_quoted(column)}, count(*) FROM {relation} AS r{_where(conditions)} GROUP BY {_quoted(column)}"


def _counts_by(call: ast.Call) -> list[str]:
    relation, own = _relation(_argument(call, "table", 1), "table")
    column = _text(_argument(call, "column", None), "column")
    values = _argument(call, "values", None) is not None
    if not values and _argument(call, "where", None) is None:
        raise _Unseen("values")
    return [_count_of(relation, column, call, values=values), *own]


def _counts_by_relations(call: ast.Call) -> list[str]:
    tables = _argument(call, "tables", 1)
    if not isinstance(tables, ast.Dict) or not tables.values:
        raise _Unseen("tables")
    column = _text(_argument(call, "column", None), "column")
    values = _argument(call, "values", None) is not None
    if not values and _argument(call, "where", None) is None:
        raise _Unseen("values")
    statements: list[str] = []
    for node in tables.values:
        relation, own = _relation(node, "tables")
        statements += [_count_of(relation, column, call, values=values), *own]
    return statements


def _probe(call: ast.Call) -> list[str]:
    relation, own = _relation(_argument(call, "table", 1), "table")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    # holds_any reads one row; count_at_most at most its bound, bound as a parameter (a probe of a table's first rows
    # when it names no key)
    limit = "1" if _name(call) == "holds_any" else f"${place}"
    # the probe walks the key it names (order=): with no order it is a scan stopped at the LIMIT, never keyed
    order = ", ".join(f"r.{column}" for column in _texts(_argument(call, "order", None), "order"))
    return [f"SELECT 1 FROM {relation} AS r{_where(pinned)} ORDER BY {order} LIMIT {limit}", *own]


def _any_rows(call: ast.Call) -> list[str]:
    """a bounded look (``any_rows``): its rows up to its ``limit``, which it must name; ``by`` and ``where`` judged."""
    if _argument(call, "limit", None) is None:
        raise _Unseen("limit")
    relation, own = _relation(_argument(call, "table", 1), "table")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    filtered, place = _filters(call, place)
    return [f"SELECT * FROM {relation} AS t{_where([*pinned, *filtered])} LIMIT ${place}", *own]


def _keyed_write(call: ast.Call) -> list[str]:
    """a delete or update by key (``delete_keyed`` / ``update_keyed``): judged only where it writes by a prefix
    (``prefix={"signal": p}``), as the code-point range it sends (``"signal" >= $n AND "signal" < $m``, judged as a
    page of that range); one by ``by``/``keys`` sends nothing here to judge. A table its caller names (a fence's,
    ``clear_signals(conn, table)``) is the module's template (``TableSchema(name=<parameter>)``), as a generic read's
    ``FROM {table}`` is: keyed by the template's index whose purpose names the caller, and a whole read where the
    module declares no template."""
    node = _argument(call, "prefix", None)
    if node is None:
        return []
    started = _by(node, "prefix")
    if not started:
        raise _Unseen("prefix")
    table_node = _argument(call, "table", 1)
    table = _relation(table_node, "table")[0] if isinstance(table_node, ast.Constant) else '"{}"'
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    # the prefix's rows are a range read in the column's order: keyed exactly when an index ranges that column for
    # the reader (a C-ordered, ASC one), as a page of the range is
    column = _quoted(started[0][0])
    return [f"SELECT 1 FROM {table}{_where([*pinned, f'{column} > ${place}'])} ORDER BY {column} LIMIT {_LIMIT}"]


def _delete_oldest(call: ast.Call) -> list[str]:
    table, _ = _relation(_argument(call, "table", 1), "table")
    key_node = _argument(call, "key", None)
    key = "rowid" if key_node is None else _text(key_node, "key")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    return [
        f"DELETE FROM {table} WHERE {_quoted(key)} IN (SELECT {_quoted(key)} FROM {table}{_where(pinned)} "
        f"ORDER BY {_quoted(key)} LIMIT ${place})"
    ]


def _read_after(call: ast.Call) -> list[str]:
    table, own = _relation(_argument(call, "table", 1), "table")
    column = _text(_argument(call, "id_column", None), "id_column")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    conditions = [*pinned, f"{_quoted(column)} > ${place}"]
    return [f"SELECT * FROM {table} AS r{_where(conditions)} ORDER BY {_quoted(column)} LIMIT {_LIMIT}", *own]


def _read_past_marks(call: ast.Call) -> list[str]:
    """the rows past the marks, as the helper sends it: its ``by`` pins and ``where`` filters, then the stamps."""
    relation, own = _relation(_argument(call, "relation", 1), "relation")
    marks = _by(_argument(call, "marks", None), "marks")
    if not marks:
        raise _Unseen("marks")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    filtered, place = _filters(call, place)
    past = " OR ".join(f"{_quoted(stamp)} > ${number}" for number, (stamp, _) in enumerate(marks, start=place))
    return [f"SELECT * FROM {relation}{_where([*pinned, *filtered, f'({past})'])}", *own]


def _delete_before(call: ast.Call) -> list[str]:
    table, _ = _relation(_argument(call, "table", 0), "table")
    key = _texts(_argument(call, "key", None), "key")
    column = _text(_argument(call, "column", None), "column")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    conditions = [*pinned, f"{_quoted(column)} < ${place}"]
    named = ", ".join(_quoted(part) for part in key)
    return [
        f"DELETE FROM {table} WHERE ({named}) IN (SELECT {named} FROM {table}"
        f"{_where(conditions)} ORDER BY {_quoted(column)} LIMIT {_LIMIT})"
    ]


def _update_before(call: ast.Call) -> list[str]:
    table, _ = _relation(_argument(call, "table", 0), "table")
    key = _texts(_argument(call, "key", None), "key")
    column = _text(_argument(call, "column", None), "column")
    assigned = _by(_argument(call, "set_to", None), "set_to")
    if not assigned:
        raise _Unseen("set_to")
    sets = ", ".join(f"{_quoted(name)} = ${place}" for place, (name, _) in enumerate(assigned, start=1))
    pinned, place = _conditions(_by(_argument(call, "by", None)), first=len(assigned) + 1)
    conditions = [*pinned, f"{_quoted(column)} < ${place}"]
    named = ", ".join(_quoted(part) for part in key)
    return [
        f"UPDATE {table} SET {sets} WHERE ({named}) IN (SELECT {named} FROM {table}"
        f"{_where(conditions)} ORDER BY {_quoted(column)} LIMIT {_LIMIT})"
    ]


def _step_relation(node: ast.expr | None, argument: str) -> list[str]:
    """the read a step makes of one relation it names: the relation whole; nothing for a step or derived select (each
    judged where it is written)."""
    if isinstance(node, ast.Call) and _name(node) == "change_relation":
        node = _argument(node, "name", 0)
    if isinstance(node, ast.Call) and (_name(node) in STEP_CALLS or _name(node) == "Derived"):
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [f"SELECT * FROM {'.'.join(_quoted(part) for part in node.value.split('.'))} AS t"]
    return [f"SELECT * FROM {_named(node, argument)} AS t"]


def _match_columns(node: ast.expr | None) -> list[str]:
    """the step relation's columns a ``Match`` list names."""
    if not isinstance(node, ast.Tuple | ast.List) or not node.elts:
        raise _Unseen("on")
    columns = []
    for element in node.elts:
        if not (isinstance(element, ast.Call) and _name(element) == "Match"):
            raise _Unseen("on")
        columns.append(_text(_argument(element, "column", 0), "on"))
    return columns


def _step(call: ast.Call) -> list[str]:
    """each relation a step names, read whole; a delete by a change relation's keys as the keyed delete it is."""
    helper = _name(call)
    statements: list[str] = []
    if helper == "union":
        for index, node in enumerate(call.args):
            if isinstance(node, ast.Starred):
                raise _Unseen("rels")
            statements += _step_relation(node, f"rels[{index}]")
    for position, keyword in STEP_RELATIONS.get(helper, ()):
        statements += _step_relation(_argument(call, keyword, position), keyword)
    if helper == "delete_matching":
        table, _ = _relation(_argument(call, "table", 1), "table")
        pinned = [
            f"{_quoted(column)} = ANY(${place})"
            for place, column in enumerate(_match_columns(_argument(call, "on", None)), 1)
        ]
        statements.append(f"DELETE FROM {table}{_where(pinned)}")
    return statements


def _apply_stage(call: ast.Call) -> list[str]:
    """the stage read whole, as the helper reads it to write the target by its key: a stage that is a change relation
    (registered, declared, a transient producer's) passes, and one that is a data table is the whole read it is.
    The target is written by the stage's key (``replace``/``difference``: its rows of the stage's keys deleted, as a
    delete by those keys is; ``upsert``: by its conflict key); a scope (``by``) deletes the target's rows of that
    scope. A target its caller names is the caller module's template (``TableSchema(name=<parameter>)``), as
    ``delete_keyed``'s is; an ``append`` names no target row."""
    statements = _step_relation(_argument(call, "stage", 2), "stage")
    mode = _argument(call, "mode", None)
    spelled = mode.value if isinstance(mode, ast.Constant) else mode.attr if isinstance(mode, ast.Attribute) else None
    if mode is not None and spelled is None:
        raise _Unseen("mode")
    if str(spelled).lower() == "append":
        return statements
    target = _argument(call, "target", 1)
    table = _relation(target, "target")[0] if isinstance(target, ast.Constant) else '"{}"'
    keyed = [
        f"{_quoted(column)} = ANY(${place})"
        for place, column in enumerate(_texts(_argument(call, "key", None), "key"), 1)
    ]
    statements.append(f"DELETE FROM {table}{_where(keyed)}")
    scope, _ = _conditions(_by(_argument(call, "by", None)))
    if scope:
        statements.append(f"DELETE FROM {table}{_where(scope)}")
    return statements


def _update_from(call: ast.Call) -> list[str]:
    """the source read whole where it is a relation (a step or derived select is judged where it is written), and the
    target written by the key it joins on, as a keyed update by those keys is. A target its caller names is the
    caller module's template (``TableSchema(name=<parameter>)``), as ``apply_stage``'s is."""
    statements = _step_relation(_argument(call, "source", None), "source")
    target = _argument(call, "table", 1)
    table = _relation(target, "table")[0] if isinstance(target, ast.Constant) else '"{}"'
    keyed = [
        f"{_quoted(column)} = ANY(${place})"
        for place, column in enumerate(_texts(_argument(call, "key", None), "key"), 1)
    ]
    pinned, place = _conditions(_by(_argument(call, "by", None)), first=len(keyed) + 1)
    statements.append(f"UPDATE {table} SET {_quoted('_')} = ${place}{_where([*keyed, *pinned])}")
    return statements


_SYNTHESIZERS: Final = {
    **dict.fromkeys(STEP_CALLS, _step),
    "read_keyed": _read_keyed,
    "read_keyed_arrow": _read_keyed,
    "read_by": _read_by,
    "read_row": _read_by,
    "read_value": _read_by,
    "read_joined": _read_joined,
    "any_rows": _any_rows,
    "keyset_page": _keyset_page,
    "keyset_rows": _keyset_page,
    "copy_keyed": _keyset_page,
    "read_changed": _read_changed,
    "group_hashes": _group_hashes,
    "counts_by": _counts_by,
    "count_at_most": _probe,
    "holds_any": _probe,
    "read_after": _read_after,
    "read_past_marks": _read_past_marks,
    "literal_select": _literal_select,
    "delete_before": _delete_before,
    "delete_oldest": _delete_oldest,
    "delete_keyed": _keyed_write,
    "update_keyed": _keyed_write,
    "counts_by_relations": _counts_by_relations,
    "update_before": _update_before,
    "apply_stage": _apply_stage,
    "update_from": _update_from,
}


def helper_statements(call: ast.Call) -> Iterator[HelperStatement]:
    """the statements a call to a keyed helper sends, for the walk to judge in the calling function; nothing for any
    other call.

    :param call: a call the walk visits
    :ptype call: ast.Call
    :return: each statement, or one :data:`UNSEEN` statement naming the argument the walk could not read
    :rtype: Iterator[HelperStatement]
    """
    helper = _name(call)
    synthesize = _SYNTHESIZERS.get(helper)
    if synthesize is None:
        return
    try:
        statements = synthesize(call)
    except _Unseen as unseen:
        yield HelperStatement(helper=helper, sql=UNSEEN, unseen=unseen.argument)
        return
    for sql in statements:
        yield HelperStatement(helper=helper, sql=sql)


def is_keyed_fragment(call: ast.Call) -> bool:
    """whether ``call`` builds a reviewed visibility clause: interpolated into a ``WHERE``, a keyed predicate.

    :param call: a call the walk visits
    :ptype call: ast.Call
    :return: whether it does
    :rtype: bool
    """
    return _name(call) in KEYED_FRAGMENT_BUILDERS


def narrows(call: ast.Call, narrowing: frozenset[str] | Mapping[str, object]) -> bool:
    """whether a pager call names the rows it reads: a narrowing keyword other than where it starts.

    ``after=`` / ``since=`` count only beside another narrowing keyword (a ``where``, a ``where_any``, ``keys``):
    alone they read every row from a place on.

    :param call: the pager call
    :ptype call: ast.Call
    :param narrowing: the keywords that narrow a pager
    :ptype narrowing: frozenset[str] | Mapping[str, object]
    :return: whether it narrows
    :rtype: bool
    """
    named = {given.arg for given in call.keywords if given.arg is not None and given.arg in narrowing}
    return bool(named - START_KEYWORDS)
