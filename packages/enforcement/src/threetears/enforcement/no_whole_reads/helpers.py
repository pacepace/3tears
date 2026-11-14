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
keys=KeyBatch(table.key, ...))``, its table and its key both names) is written with the names as interpolations. The
walk hands the helper ``pk`` for columns that flow from a declared key source (:mod:`.keys`), written ``{pk}``, the
key; any other run-time column, whatever it is named, is a predicate the rule cannot see. ``empty_tables`` and
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

from threetears.enforcement.no_whole_reads.schema import COPY_PREFIX, PRODUCED_PREFIX

__all__ = [
    "COPY_CALLS",
    "KEYED_FRAGMENT_BUILDERS",
    "BUILDER_CALLS",
    "KEYED_HELPER_CALLS",
    "START_KEYWORDS",
    "STEP_CALLS",
    "STEP_RELATIONS",
    "STEP_TARGETS",
    "UNSEEN",
    "HelperStatement",
    "PagerOrder",
    "helper_statements",
    "is_keyed_fragment",
    "narrows",
    "pager_order",
    "templated",
    "with_relation",
]

#: the calls answering a local copy's relation by its table's name (``snapshot.relation(NAME)``)
COPY_CALLS: Final[frozenset[str]] = frozenset({"relation", "relation_reading"})

#: the helpers judged by the statement a call sends; each reads (or deletes) only what its arguments key
KEYED_HELPER_CALLS: Final[frozenset[str]] = frozenset(
    {
        "read_keyed",
        "read_keyed_arrow",
        "read_by",
        "read_row",
        "read_value",
        "read_joined",
        "read_by_joined",
        "read_link_chain",
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
        "read_packed",
        "literal_select",
        "delete_before",
        "delete_oldest",
        "delete_keyed",
        "update_keyed",
        "counts_by_relations",
        "aggregate_keyed",
        # :func:`threetears.datasources.query_client.read_counted`: the part of a warehouse relation its filters name,
        # paged by its key (reports #61 R4: judged by those filters against the relation's declared keys)
        "read_counted",
        "update_before",
        "apply_stage",
        "update_from",
        "visible_rows",
        # the pure builders (threetears.interfaces.keyed) a runner sends, and a package that may not depend on core
        # runs on its own connection: each judged as its runner's call (:data:`BUILDER_CALLS`)
        *(
            "read_by_statement",
            "read_row_statement",
            "read_keyed_statements",
            "keyset_page_statement",
            "keyset_rows_statement",
            "update_keyed_statements",
            "delete_keyed_statements",
            "delete_before_statement",
            "update_before_statement",
            "page_statement",
            "keyed_read_statement",
        ),
    }
)

#: each pure builder a caller may run on its own connection -> the runner whose statements it writes: its call is judged
#: as that runner's, its arguments where the runner takes them (the runner's connection, its first argument, not passed)
BUILDER_CALLS: Final[Mapping[str, str]] = {
    "read_by_statement": "read_by",
    "read_row_statement": "read_row",
    "read_keyed_statements": "read_keyed",
    "keyset_page_statement": "keyset_page",
    "keyset_rows_statement": "keyset_rows",
    "update_keyed_statements": "update_keyed",
    "delete_keyed_statements": "delete_keyed",
    "delete_before_statement": "delete_before",
    "update_before_statement": "update_before",
    # a page's statement as its runners write it, positionally: read by no synthesizer's keywords, so a call outside
    # the keyed packages is unseen (a finding), never a pass
    "page_statement": "keyset_page",
    # a keyed batch's statement as its runners write it, positionally: its keys unseen outside the keyed packages
    "keyed_read_statement": "read_keyed",
}

#: the runners that take no connection first (the table is their first argument): their builders' calls stand as written
_TABLE_FIRST: Final[frozenset[str]] = frozenset({"delete_before", "update_before"})

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
        "joined",
        "any_step",
        "materialize",
        "ordered",
        "ranked_order",
        "value_ranks",
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
    "joined": ((0, "rel"), (1, "other")),
    "any_step": ((1, "step"),),
    "materialize": ((2, "step"),),
    "ordered": ((0, "step"),),
    # an order by a costly key per distinct value: bounded exactly as the relation it orders
    "ranked_order": ((0, "rel"),),
    "value_ranks": ((0, "rel"),),
    # the keyed writes from a relation: what they read is a step's or a stage's rows
    "apply_stage": ((2, "stage"),),
    "update_from": ((None, "source"),),
}

#: where a step names the table it writes (``materialize``'s, replaced whole by the step's rows; ``apply_stage``'s and
#: ``update_from``'s, written by the stage's key): a change set's stage passes, a data table is the write it is. A
#: target a forwarder is handed (``Stages._add_rows``' ``stage``) is judged at each call with what the call passes (R9)
STEP_TARGETS: Final[Mapping[str, tuple[tuple[int | None, str], ...]]] = {
    "materialize": ((1, "table"),),
    "apply_stage": ((1, "target"),),
    "update_from": ((1, "table"),),
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

#: a local database's table named at run time: ``main.`` and nothing but interpolations
_LOCAL_RUN_TIME: Final[re.Pattern[str]] = re.compile(r"main\.(?:\{\})+")

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
    :ivar merged: the argument holding pins merged in at run time beside its literal ones (``by={"run_id": ids,
        **others}``): the statement holds only the literal ones, so it passes when they key it, and a read they do not
        key is unknown (the merged pins may key it), never judged a non-key read on a guess
    """

    helper: str
    sql: str | None
    unseen: str = ""
    merged: str = ""


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
    """each pinned column and how it is bound: ``=``, ``= ANY`` (an ``AnyOf``) or ``IS NULL``. A ``by`` column named
    at run time (``by={pk_column: i}``) is the interpolation it would be in SQL the caller wrote: the key when the walk
    hands it as ``pk`` (a declared key source), any other a predicate the rule cannot see."""
    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return []
    zipped = _zipped_columns(node)
    if argument == "by" and zipped is not None:
        # ``by=dict(zip(pk_columns, key))`` / ``by=dict(pk)``: every column a run-time list or mapping names, each
        # bound, as the interpolation in SQL the caller wrote: the key when the walk hands it as ``pk`` (a declared
        # key source), else a predicate the rule cannot see
        return [("{" + _named(zipped, argument)[1:-1] + "}", "eq")]
    if not isinstance(node, ast.Dict):
        raise _Unseen(argument)
    if any(key is None for key in node.keys) and (
        argument != "by" or all(key is None or not isinstance(key, ast.Constant) for key in node.keys)
    ):
        # a mapping merged in at run time (``{**others}``) names columns the walk cannot see; beside a literal entry
        # (``{"run_id": AnyOf(ids), **others}``) it only narrows that pin -- every entry is ANDed -- so the call is
        # judged by its literal entries (reports #55 ask 2); with none, nothing is seen
        raise _Unseen(argument)
    pinned = []
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            continue
        if isinstance(key, ast.Call):
            column = _pinned_expression(key, argument)
        elif argument == "by" and not isinstance(key, ast.Constant):
            column = "{" + _named(key, argument)[1:-1] + "}"
        else:
            column = _text(key, argument)
        if isinstance(value, ast.Call) and _name(value) == "AnyOf":
            pinned.append((column, "any"))
        elif isinstance(value, ast.Constant) and value.value is None:
            pinned.append((column, "null"))
        elif argument == "by" and isinstance(value, ast.Constant) and isinstance(value.value, bool):
            # a boolean pin is sent as its literal (``threetears.interfaces.keyed.filters.by_conditions``), so it
            # states a partial index's ``WHERE flag = false`` under any plan
            pinned.append((column, _LITERAL + ("TRUE" if value.value else "FALSE")))
        elif argument == "by" and isinstance(value, ast.Constant) and isinstance(value.value, str):
            # a literal pin (``by={"status": "proposed"}``) is the literal the statement compares: a partial index
            # whose predicate names it (``WHERE status = 'proposed'``) holds the rows asked for
            pinned.append((column, _LITERAL + "'" + value.value.replace("'", "''") + "'"))
        else:
            pinned.append((column, "eq"))
    return pinned


#: what marks a pin bound to a literal: the rest of its ``how`` is the literal as SQL spells it
_LITERAL: str = "lit:"


def _zipped_columns(node: ast.expr) -> ast.expr | None:
    """the column list of a ``dict(zip(<columns>, <values>))`` pin, or the mapping of a ``dict(<mapping>)`` one: a
    name or attribute the code binds at run time, each of its columns bound to its value; None for any other
    expression."""
    if not (isinstance(node, ast.Call) and _name(node) == "dict" and len(node.args) == 1 and not node.keywords):
        return None
    inner = node.args[0]
    if isinstance(inner, ast.Name | ast.Attribute):
        # ``dict(pk)``: a copy of a run-time column -> value mapping, every column of it pinned
        return inner
    if not (isinstance(inner, ast.Call) and _name(inner) == "zip" and len(inner.args) == 2):
        return None
    columns = inner.args[0]
    return columns if isinstance(columns, ast.Name | ast.Attribute) else None


#: what marks a pinned column as a ``Pin.expr`` expression, spelled as written rather than quoted
_EXPRESSION: str = "\x00"


def _pinned_expression(key: ast.Call, argument: str) -> str:
    """a ``Pin.expr("<expression>")`` key, as the statement spells it: the expression of the row's columns with its
    ``t.`` and quotes dropped, as an expression index reads it (a :class:`Sort` expression's form)."""
    if not (isinstance(key.func, ast.Attribute) and key.func.attr == "expr" and _name(key) == "expr"):
        raise _Unseen(argument)
    written = re.sub(r'\bt\.(?="|\w)', "", _text(_argument(key, "expression", 0), argument))
    return _EXPRESSION + re.sub(r'"(\w+)"', r"\1", written)


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _pinned(column: str) -> str:
    """a pinned column as the statement spells it: quoted, or a column named at run time as its interpolation."""
    return column if column.startswith("{") and column.endswith("}") else _quoted(column)


def _conditions(pinned: Sequence[tuple[str, str]], *, first: int = 1, alias: str = "") -> tuple[list[str], int]:
    prefix = f"{alias}." if alias else ""
    conditions = []
    place = first
    for column, how in pinned:
        spelled = column.removeprefix(_EXPRESSION) if column.startswith(_EXPRESSION) else f"{prefix}{_pinned(column)}"
        if how == "null":
            conditions.append(f"{spelled} IS NULL")
            continue
        if how.startswith(_LITERAL):
            conditions.append(f"{spelled} = {how.removeprefix(_LITERAL)}")
            continue
        conditions.append(f"{spelled} = ANY(${place})" if how == "any" else f"{spelled} = ${place}")
        place += 1
    return conditions, place


def _named(node: ast.expr | None, argument: str) -> str:
    """a relation or key the caller names: a literal, quoted; a name or attribute, as the interpolation it would be in
    SQL the caller wrote (``{table}``), so the rule reads it as it reads that SQL: ``pk`` (the walk's name for columns
    from a declared key source) binds the key; any other is a predicate it cannot see."""
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
        if (
            isinstance(columns, ast.Call)
            and isinstance(columns.func, ast.Name)
            and columns.func.id in {"tuple", "list"}
            and len(columns.args) == 1
            and not columns.keywords
        ):
            # the columns copied (``KeyBatch(tuple(key), ...)``): the same columns, judged as the name it copies
            columns = columns.args[0]
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


def _selected_list(call: ast.Call) -> str:
    """the select list a read sends: every column, and each ``Selected("<expression>", "<alias>")`` its ``columns=``
    names as a literal, written over the row ``t`` as the helper writes it, so a sub-select in it (a correlated read
    of another table, an index it uses) is judged as the hand-written statement's was. A ``columns=`` bound elsewhere
    reads as every column."""
    node = _argument(call, "columns", None)
    listed = ["*"]
    if isinstance(node, ast.Tuple | ast.List):
        for element in node.elts:
            if (
                isinstance(element, ast.Call)
                and _name(element) == "Selected"
                and len(element.args) == 2
                and all(isinstance(arg, ast.Constant) and isinstance(arg.value, str) for arg in element.args)
            ):
                expression, alias = (str(arg.value) for arg in element.args if isinstance(arg, ast.Constant))
                listed.append(f"{expression} AS {alias}")
    return ", ".join(listed)


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
    """a point read (``read_by``, ``read_row``, ``read_value``): the rows its ``by`` pins, each ``= $n``, and its
    ``where=`` fragments after them; one pinning nothing is the whole read it is. A table the code names at run time read by columns it also names at run time
    (``by=dict(pk)``, ``by={pk_column: i}``) is a generic reader's, judged as the interpolated SQL it would write
    (``FROM {table} WHERE {pk} = $1``), as :func:`_read_keyed` judges one; by literal columns, the walk cannot match
    them to any table's key."""
    table_node = _argument(call, "table", 1)
    pins = _by(_argument(call, "by", None))
    own: list[str] = []
    if isinstance(table_node, ast.Name | ast.Attribute) and pins and all(column.startswith("{") for column, _ in pins):
        table = _named(table_node, "table")
    else:
        table, own = _relation(table_node, "table")
    pinned, place = _conditions(pins, alias="t")
    # its ``where=`` fragments, ANDed after the pins as the helper sends them: a partial index's predicate a fragment
    # states (``where=[Fragment("kind = 'live'")]``) is stated by the read
    filtered, _ = _filters(call, place)
    order = _written_order(_argument(call, "order", None))
    return [f"SELECT {_selected_list(call)} FROM {table} AS t{_where([*pinned, *filtered])}{order}", *own]


def _written_order(node: ast.expr | None) -> str:
    """`` ORDER BY ...`` for a read's ``order=`` written as a list of ``Sort`` terms, as the helper sends it; ``""``
    for none, or one the walk cannot read (a name bound elsewhere): such a read is judged by its pins, as before."""
    if node is None:
        return ""
    try:
        columns = _sort_columns(node)
    except _Unseen:  # NOSILENT: an order the walk cannot read is left out; the pins alone judge the read
        return ""
    assert isinstance(node, ast.Tuple | ast.List)
    terms = []
    for column, element in zip(columns, node.elts, strict=True):
        descending = _argument(element, "descending", None) if isinstance(element, ast.Call) else None
        down = isinstance(descending, ast.Constant) and descending.value is True
        terms.append(f"{column} DESC" if down else column)
    return f" ORDER BY {', '.join(terms)}"


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
    reads it (``"customer_id" = ANY($n)``). :func:`threetears.agent.acl.visible_rows` binds its ``customer_column`` so
    itself."""
    columns = []
    if _name(call) == "visible_rows":
        customer = _argument(call, "customer_column", None)
        if not (isinstance(customer, ast.Constant) and isinstance(customer.value, str)):
            raise _Unseen("customer_column")
        columns.append(customer.value)
    node = _argument(call, "visible", None)
    if not isinstance(node, ast.Tuple | ast.List):
        return columns
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
    """a helper's ``where=`` fragments as written (``Fragment("t.due_at <= $1", (now,))``), renumbered past ``first``.

    A fragment the walk cannot read (a run-time list, a ``Fragment`` of run-time text) only narrows: every helper ANDs
    each fragment after its pins inside its own parentheses (``threetears.core.keyed._filters.narrowing``), so it can
    never widen past them, and the call is judged by its pins (``by``) alone: keyed when they are, a whole read when
    they pin nothing. A fragment the walk reads is judged with them, as it may pin a key itself."""
    node = _argument(call, "where", None)
    if node is None or not isinstance(node, ast.Tuple | ast.List):
        return [], first
    written = []
    place = first
    for element in node.elts:
        if not (isinstance(element, ast.Call) and _name(element) == "Fragment"):
            continue
        sql = _argument(element, "sql", 0)
        if not (isinstance(sql, ast.Constant) and isinstance(sql.value, str)):
            continue
        text = sql.value
        numbers = [int(found) for found in re.findall(r"\$(\d+)", text)]
        # in its own parentheses, as the helper writes it
        written.append("(" + re.sub(r"\$(\d+)", lambda match: f"${int(match.group(1)) + place - 1}", text) + ")")
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
    return [
        f"SELECT {_selected_list(call)} FROM {table} AS t{_where(page)} ORDER BY {ordered} LIMIT {_LIMIT}",
        *_visible(call),
        *own,
    ]


def _read_changed(call: ast.Call) -> list[str]:
    """phase 1, the scope's keys and hashes, is a keyset page in the key's order (``relation`` / ``key`` as
    ``keyset_rows`` takes ``table`` / ``order``; a key column named by its string a sort of it); phase 2 reads by
    those keys, keyed whenever phase 1 is.

    A key the walk cannot read (``key=list(source.key)``, handed at run time) leaves the page's order unknown: phase 1
    is judged by its ``by`` and ``where`` alone, a page ordered by ``{}``, so pins of a leading key of the relation
    (its table's, or the key a declared relation producer gives it) are keyed and a page pinning none is found."""
    key = _argument(call, "key", None)
    if key is not None and not isinstance(key, ast.Tuple | ast.List):
        table, own = _relation(_argument(call, "relation", 1), "relation")
        pinned, place = _conditions(_by(_argument(call, "by", None)))
        filtered, _ = _filters(call, place)
        return [f"SELECT * FROM {table} AS t{_where([*pinned, *filtered])} ORDER BY {{}} LIMIT {_LIMIT}", *own]
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


def _read_by_joined(call: ast.Call) -> list[str]:
    """a read with joins (``read_by_joined``): its read by its pins, as :func:`_read_by` judges one, then each joined
    table's keyed batch by the columns it is joined on and its own pins, as :func:`_read_joined` judges one; joins the
    walk cannot read as a literal list of ``Joined(...)`` are unseen."""
    joins = _argument(call, "joins", None)
    if not isinstance(joins, ast.List | ast.Tuple):
        raise _Unseen("joins")
    statements = _read_by(call)
    for join in joins.elts:
        if not (isinstance(join, ast.Call) and _name(join) == "Joined"):
            raise _Unseen("joins")
        joined = ast.Call(
            func=ast.Name("read_joined", ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg="table", value=_argument(join, "table", 0) or ast.Name("_unseen", ast.Load())),
                ast.keyword(arg="on", value=_argument(join, "on", 1) or ast.Name("_unseen", ast.Load())),
                *([ast.keyword(arg="by", value=by)] if (by := _argument(join, "by", None)) is not None else []),
            ],
        )
        statements += _read_joined(joined)
    return statements


def _read_link_chain(call: ast.Call) -> list[str]:
    """a chain of links (``read_link_chain``): the recursive statement it sends, the first row by its ``key`` and each
    next one by the ``key`` the row before links to, so keyed exactly when ``key`` leads the table's key."""
    table = _named(_argument(call, "table", 1), "table")
    key = _quoted(_text(_argument(call, "key", None), "key"))
    link = _quoted(_text(_argument(call, "link", None), "link"))
    return [
        f"WITH RECURSIVE chain({key}, {link}, depth) AS ( SELECT t.{key}, t.{link}, 1 FROM {table} AS t WHERE "
        f"t.{key} = $1 UNION ALL SELECT t.{key}, t.{link}, chain.depth + 1 FROM {table} AS t JOIN chain ON "
        f"t.{key} = chain.{link} WHERE chain.depth < $2 ) SELECT {key}, {link} FROM chain"
    ]


def _relation(node: ast.expr | None, argument: str) -> tuple[str, list[str]]:
    """a relation a helper reads, as its statement names it, and the statements it holds of its own: a table literal
    (``schema.table`` quoted a segment at a time), or a ``Derived("SELECT ...")`` judged on its own text."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(PRODUCED_PREFIX):
        # a declared producer's run-time relation (the walk wrote it): the catalog's own name for it, as declared
        return node.value, []
    if isinstance(node, ast.JoinedStr):
        # ``f"main.{name}"``: its literal parts as written, each interpolation ``{}``
        node = ast.Constant(
            "".join(
                part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else "{}"
                for part in node.values
            )
        )
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if _LOCAL_RUN_TIME.fullmatch(node.value):
            # a table of the local database (DuckDB's or SQLite's ``main``) named at run time: a local copy's
            # relation, judged against the default key every copy declares (``local_relations["*"]``)
            return f"{COPY_PREFIX}{{}}", []
        return ".".join(_quoted(part) for part in node.value.split(".")), []
    if isinstance(node, ast.Call) and _name(node) == "Derived":
        derived = _text(_argument(node, "sql", 0), argument)
        return f"({derived})", [derived]
    raise _Unseen(argument)


def _count_of(relation: str, column: str, call: ast.Call, *, values: bool, own: ast.expr | None = None) -> str:
    """a grouped count's statement; ``column`` as the statement spells it (quoted, or a run-time interpolation);
    ``own``: the relation's own pins (``counts_by_relations``' ``pins=`` for its label), ANDed after the rest."""
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    conditions = [*pinned]
    if values:
        conditions.append(f"{column} = ANY(${place})")
        place += 1
    filtered, place = _filters(call, place)
    conditions += filtered
    if own is not None:
        conditions += _conditions(_by(own, "pins"), first=place)[0]
    return f"SELECT {column}, count(*) FROM {relation} AS r{_where(conditions)} GROUP BY {column}"


def _counted_column(call: ast.Call, *, values: bool) -> str:
    """the column a count groups by, as the statement spells it: a literal, quoted; one named at run time, its
    interpolation, but only when no ``values=`` binds it (the count's rows are then bound by its pins and filters
    alone, whatever column they are grouped by; a set of values of an unseen column is a predicate no one can see)
    (reports #55 ask 1)."""
    node = _argument(call, "column", None)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _quoted(node.value)
    if values or not isinstance(node, ast.Name | ast.Attribute):
        raise _Unseen("column")
    return "{" + (node.id if isinstance(node, ast.Name) else node.attr) + "}"


def _counts_by(call: ast.Call) -> list[str]:
    relation, own = _relation(_argument(call, "table", 1), "table")
    values = _argument(call, "values", None) is not None
    column = _counted_column(call, values=values)
    if not values and _argument(call, "where", None) is None and _argument(call, "by", None) is None:
        # counted with nothing narrowing it: every value of the column, whatever the relation
        raise _Unseen("values")
    return [_count_of(relation, column, call, values=values), *own]


def _read_counted(call: ast.Call) -> list[str]:
    """``read_counted(client, datasource, columns=, relation=, key=, rows=, where=, where_in=, not_null=)``: every row
    of the part its filters name, page after page by its key, so judged as one read of that part -- equalities
    (``where``), sets (``where_in``, each ``= ANY``) and ``IS NOT NULL`` columns -- against the relation's declared
    keys; the pages' ``ORDER BY key LIMIT`` read the part whole, so they key nothing of their own."""
    relation, own = _relation(_argument(call, "relation", None), "relation")
    conditions, place = _conditions(_by(_argument(call, "where", None), "where"))
    sets = _argument(call, "where_in", None)
    if sets is not None and not (isinstance(sets, ast.Constant) and sets.value is None):
        if not isinstance(sets, ast.Dict) or any(key is None for key in sets.keys):
            raise _Unseen("where_in")
        conditions += _conditions([(_text(key, "where_in"), "any") for key in sets.keys if key], first=place)[0]
    present = _argument(call, "not_null", None)
    if present is not None:
        if not isinstance(present, ast.Tuple | ast.List):
            raise _Unseen("not_null")
        conditions += [f"{_quoted(_text(column, 'not_null'))} IS NOT NULL" for column in present.elts]
    return [f"SELECT * FROM {relation} AS r{_where(conditions)}", *own]


def _counts_by_relations(call: ast.Call) -> list[str]:
    tables = _argument(call, "tables", 1)
    if not isinstance(tables, ast.Dict) or not tables.values:
        raise _Unseen("tables")
    values = _argument(call, "values", None) is not None
    column = _counted_column(call, values=values)
    if not values and _argument(call, "where", None) is None and _argument(call, "by", None) is None:
        # counted with nothing narrowing it: every value of the column, whatever the relation
        raise _Unseen("values")
    pins = _argument(call, "pins", None)
    mine: dict[str, ast.expr] = {}
    if pins is not None and not (isinstance(pins, ast.Constant) and pins.value is None):
        if not isinstance(pins, ast.Dict):
            raise _Unseen("pins")
        for key, value in zip(pins.keys, pins.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise _Unseen("pins")
            mine[key.value.lower()] = value
    statements: list[str] = []
    for key, node in zip(tables.keys, tables.values, strict=True):
        relation, own = _relation(node, "tables")
        label = str(key.value).lower() if isinstance(key, ast.Constant) and isinstance(key.value, str) else None
        statements += [_count_of(relation, column, call, values=values, own=mine.get(label or "")), *own]
    return statements


def _ranges(node: ast.expr | None) -> list[str]:
    """a keyed aggregate's ``ranges=`` as the conditions it writes: each ``Range`` of its one literal column, by the
    edges it names (``low`` at or above, ``high`` below). Unseen when the list, a range or its column is run-time, or
    the ranges name two columns (the helper refuses those: one column, the one after the prefix)."""
    if node is None:
        return []
    if not isinstance(node, ast.Tuple | ast.List):
        raise _Unseen("ranges")
    columns: set[str] = set()
    edges: list[tuple[str, bool, bool]] = []
    for element in node.elts:
        if not (isinstance(element, ast.Call) and _name(element) == "Range"):
            raise _Unseen("ranges")
        column = _argument(element, "column", 0)
        if not (isinstance(column, ast.Constant) and isinstance(column.value, str)):
            raise _Unseen("ranges")
        columns.add(column.value)

        def given(keyword: str, position: int, range_: ast.Call = element) -> bool:
            value = _argument(range_, keyword, position)
            return value is not None and not (isinstance(value, ast.Constant) and value.value is None)

        edges.append((column.value, given("low", 1), given("high", 2)))
    if len(columns) > 1:
        raise _Unseen("ranges")
    # every range is of the one column: the rows read are those of its widest edges, the edges any range names
    if not edges:
        return []
    ranged = _quoted(edges[0][0])
    return [
        *([f"{ranged} >= $0"] if all(low for _, low, _ in edges) else []),
        *([f"{ranged} < $0"] if all(high for _, _, high in edges) else []),
    ]


def _aggregate_keyed(call: ast.Call) -> list[str]:
    """a keyed aggregate (``aggregate_keyed``): the rows its ``by`` pins (an ``AnyOf`` as ``= ANY``), its ranges of
    the next column bound and its ``where`` narrows, counted. Its group (a column or a time bucket) is no predicate."""
    relation, own = _relation(_argument(call, "table", 1), "table")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    ranged = []
    for condition in _ranges(_argument(call, "ranges", None)):
        ranged.append(condition.replace("$0", f"${place}"))
        place += 1
    filtered, _ = _filters(call, place)
    return [f"SELECT count(*) FROM {relation} AS r{_where([*pinned, *ranged, *filtered])}", *own]


def _probe(call: ast.Call) -> list[str]:
    relation, own = _relation(_argument(call, "table", 1), "table")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    # holds_any reads one row; count_at_most at most its bound, bound as a parameter (a probe of a table's first rows
    # when it names no key)
    limit = "1" if _name(call) == "holds_any" else f"${place}"
    # the probe walks the key it names (order=): with no order it is a scan stopped at the LIMIT, never keyed
    order = ", ".join(f"r.{column}" for column in _texts(_argument(call, "order", None), "order"))
    return [f"SELECT 1 FROM {relation} AS r{_where(pinned)} ORDER BY {order} LIMIT {limit}", *own]


def _read_packed(call: ast.Call) -> list[str]:
    """a packed read (``read_packed``): the texts of the rows its ``by`` pins and ``where`` narrows, as the read of
    those rows it sends; one pinning nothing is the whole read it is."""
    relation, own = _relation(_argument(call, "relation", 1), "relation")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    filtered, _ = _filters(call, place)
    return [f"SELECT * FROM {relation} AS r{_where([*pinned, *filtered])}", *own]


def _any_rows(call: ast.Call) -> list[str]:
    """a bounded look (``any_rows``): its rows up to its ``limit``, which it must name; ``by`` and ``where`` judged."""
    if _argument(call, "limit", None) is None:
        raise _Unseen("limit")
    relation, own = _relation(_argument(call, "table", 1), "table")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    filtered, place = _filters(call, place)
    return [f"SELECT * FROM {relation} AS t{_where([*pinned, *filtered])} LIMIT ${place}", *own]


def _keyed_write(call: ast.Call) -> list[str]:
    """a delete or update by key (``delete_keyed`` / ``update_keyed``), as the statement it sends: the rows its
    ``keys`` batch (``= ANY`` each column) and ``by`` pin, its ``where`` guards beside them; a write by a prefix
    (``prefix={"signal": p}``) as the code-point range it sends (``"signal" >= $n AND "signal" < $m``, judged as a page
    of that range). A table its caller names (a fence's, ``clear_signals(conn, table)``) is the module's template
    (``TableSchema(name=<parameter>)``), as a generic read's ``FROM {table}`` is: keyed by the template's index whose
    purpose names the caller, and a whole read where the module declares no template. A pinned column named at run
    time (``by={user_column: u}``) is a predicate the walk cannot see."""
    table_node = _argument(call, "table", 1)
    table = _relation(table_node, "table")[0] if isinstance(table_node, ast.Constant) else '"{}"'
    node = _argument(call, "prefix", None)
    if node is None:
        keys_node = _argument(call, "keys", None)
        keyed = (
            []
            if keys_node is None
            else [f"{column} = ANY(${place})" for place, column in enumerate(_key_batch_columns(keys_node), 1)]
        )
        pinned, place = _conditions(_by(_argument(call, "by", None)), first=len(keyed) + 1)
        filtered, place = _filters(call, place)
        conditions = [*keyed, *pinned, *filtered]
        if _name(call) == "update_keyed":
            return [f"UPDATE {table} SET {_quoted('_')} = ${place}{_where(conditions)}"]
        return [f"DELETE FROM {table}{_where(conditions)}"]
    started = _by(node, "prefix")
    if not started:
        raise _Unseen("prefix")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    # the prefix's rows are a range read in code-point order, as prefix_conditions sends it (``COLLATE "C"``): keyed
    # exactly when an index ranges that column in that collation for the reader (a C-ordered, ASC one), as a page of
    # the range is. A key in the column's own collation (a primary key's lead) does not order it so, and YugabyteDB
    # hashes a key's first column: such a prefix reads every row
    column = f'{_quoted(started[0][0])} COLLATE "C"'
    return [f"SELECT 1 FROM {table}{_where([*pinned, f'{column} > ${place}'])} ORDER BY {column} LIMIT {_LIMIT}"]


def _delete_oldest(call: ast.Call) -> list[str]:
    """a cache's trim (``delete_oldest``), as the statement it sends: its ``count`` oldest keys within ``by``, oldest
    by ``key``, or by ``order`` then ``key`` when the age is a column many rows share (a ``date_updated``), so the read
    is keyed (and credits an index's purpose) exactly when an index leads with ``order`` (then ``key``)."""
    table, _ = _relation(_argument(call, "table", 1), "table")
    key_node = _argument(call, "key", None)
    key = "rowid" if key_node is None else _text(key_node, "key")
    order_node = _argument(call, "order", None)
    unset = order_node is None or (isinstance(order_node, ast.Constant) and order_node.value is None)
    aged = [] if unset else [_quoted(_text(order_node, "order"))]
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    return [
        f"DELETE FROM {table} WHERE {_quoted(key)} IN (SELECT {_quoted(key)} FROM {table}{_where(pinned)} "
        f"ORDER BY {', '.join([*aged, _quoted(key)])} LIMIT ${place})"
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


def _step_relation(node: ast.expr | None, argument: str, where: Sequence[str] = ()) -> list[str]:
    """the read a step makes of one relation it names: the relation whole, or the rows its ``where=`` on ``t`` names;
    nothing for a step or derived select (each judged where it is written). A name already quoted
    (``Quoted(snapshot.relation(NAME))``, the walk writing it as the relation it names: ``copy.<name>``) is read as
    that relation, never trusted for being quoted."""
    if isinstance(node, ast.Call) and _name(node) == "change_relation":
        node = _argument(node, "name", 0)
    if isinstance(node, ast.Call) and (_name(node) in STEP_CALLS or _name(node) == "Derived"):
        return []
    if isinstance(node, ast.Call) and _name(node) == "Quoted":
        named = _argument(node, "name", 0)
        if isinstance(named, ast.Constant) and isinstance(named.value, str):
            return [f"SELECT * FROM {named.value} AS t{_where(where)}"]
        return [f"SELECT * FROM {_named(named, argument)} AS t{_where(where)}"]
    if isinstance(node, ast.Constant | ast.JoinedStr):
        # a literal, a run-time name of the local database's (``main.{}``: a local copy's), an f-string's text
        return [f"SELECT * FROM {_relation(node, argument)[0]} AS t{_where(where)}"]
    return [f"SELECT * FROM {_named(node, argument)} AS t{_where(where)}"]


#: the steps whose ``where=`` is on the relation they read first (``t``): its read is the rows those name
_WHERE_ON_T: Final[frozenset[str]] = frozenset(
    {"scan", "semi", "anti", "distinct", "grouped", "latest_per_key", "windowed"}
)


def _key_columns(node: ast.expr | None) -> list[str]:
    """the key a stage or source writes its target by, each column as the statement spells it: a literal list's
    columns quoted, one named at run time (``key=self._keys[table]``) as one interpolation (``{}``), a key the walk
    cannot see, as ``delete_matching``'s run-time ``on=`` is (R9)."""
    if isinstance(node, ast.Tuple | ast.List) and all(
        isinstance(element, ast.Constant) and isinstance(element.value, str) for element in node.elts
    ):
        return [_quoted(_text(element, "key")) for element in node.elts]
    if isinstance(node, ast.Name | ast.Attribute | ast.Subscript | ast.Call):
        return ["{}"]
    raise _Unseen("key")


def _match_columns(node: ast.expr | None) -> list[str]:
    """the step relation's columns a ``Match`` list names; a column named at run time (``Match(group_column(t))``)
    as an interpolation (``{}``), a predicate on no column the walk can see, beside the ones it can."""
    if not isinstance(node, ast.Tuple | ast.List) or not node.elts:
        raise _Unseen("on")
    columns = []
    for element in node.elts:
        if not (isinstance(element, ast.Call) and _name(element) == "Match"):
            raise _Unseen("on")
        column = _argument(element, "column", 0)
        if column is None:
            raise _Unseen("on")
        if isinstance(column, ast.Constant) and isinstance(column.value, str):
            columns.append(column.value)
        else:
            # the walk hands ``pk`` for a declared key named at run time: the key, as in SQL the caller wrote
            columns.append("{pk}" if isinstance(column, ast.Name) and column.id == "pk" else "{}")
    if all(column == "{}" for column in columns):
        raise _Unseen("on")
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
    narrowed = _filters(call, 1)[0] if helper in _WHERE_ON_T else []
    for position, keyword in STEP_RELATIONS.get(helper, ()):
        statements += _step_relation(_argument(call, keyword, position), keyword, narrowed if position == 0 else ())
    for position, keyword in STEP_TARGETS.get(helper, ()):
        replaced = _argument(call, "replace", None)
        if helper == "materialize" and isinstance(replaced, ast.Constant) and replaced.value is False:
            # a table made new (``CREATE TABLE ... AS``, no drop first): no row of one there is written
            continue
        target = _argument(call, keyword, position)
        if isinstance(target, ast.Call) and _name(target) == "change_relation":
            target = _argument(target, "name", 0)
        written, _ = _relation(target, keyword)
        statements.append(f"DELETE FROM {written}")
    if helper == "delete_matching":
        table, _ = _relation(_argument(call, "table", 1), "table")
        pinned = [
            f"{column if column in {'{}', '{pk}'} else _quoted(column)} = ANY(${place})"
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
    # a literal or an f-string target (``f"main.{name}"``: a local copy named at run time) as _relation names it (R9)
    table = _relation(target, "target")[0] if isinstance(target, ast.Constant | ast.JoinedStr) else '"{}"'
    keyed = [f"{column} = ANY(${place})" for place, column in enumerate(_key_columns(_argument(call, "key", None)), 1)]
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
    table = _relation(target, "table")[0] if isinstance(target, ast.Constant | ast.JoinedStr) else '"{}"'
    keyed = [f"{column} = ANY(${place})" for place, column in enumerate(_key_columns(_argument(call, "key", None)), 1)]
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
    "read_by_joined": _read_by_joined,
    "read_link_chain": _read_link_chain,
    "any_rows": _any_rows,
    "keyset_page": _keyset_page,
    "keyset_rows": _keyset_page,
    #: :func:`threetears.agent.acl.visible_rows`: keyset_rows' pages over ``table`` in ``order``, its ``where``
    #: beside, the caller's customers filtering them
    "visible_rows": _keyset_page,
    "copy_keyed": _keyset_page,
    "read_changed": _read_changed,
    "group_hashes": _group_hashes,
    "counts_by": _counts_by,
    "count_at_most": _probe,
    "holds_any": _probe,
    "read_after": _read_after,
    "read_past_marks": _read_past_marks,
    "read_packed": _read_packed,
    "literal_select": _literal_select,
    "delete_before": _delete_before,
    "delete_oldest": _delete_oldest,
    "delete_keyed": _keyed_write,
    "update_keyed": _keyed_write,
    "counts_by_relations": _counts_by_relations,
    "aggregate_keyed": _aggregate_keyed,
    "read_counted": _read_counted,
    "update_before": _update_before,
    "apply_stage": _apply_stage,
    "update_from": _update_from,
}


#: the argument naming the relation a keyed read reads, by helper: keyword and position
_READ_RELATION: Final[Mapping[str, tuple[str, int]]] = {
    **dict.fromkeys(
        (
            "read_keyed",
            "read_keyed_arrow",
            "read_by",
            "read_by_joined",
            "read_link_chain",
            "read_row",
            "read_value",
            "any_rows",
            "keyset_page",
            "keyset_rows",
            "copy_keyed",
            "counts_by",
            "aggregate_keyed",
            "count_at_most",
            "holds_any",
            "read_after",
        ),
        ("table", 1),
    ),
    **dict.fromkeys(("read_changed", "group_hashes", "read_past_marks", "read_packed"), ("relation", 1)),
    # keyword-only: no position names it
    "read_counted": ("relation", -1),
    # the keyed writes, and the pure builders a package that may not depend on core sends on its own connection
    **dict.fromkeys(("delete_keyed", "update_keyed"), ("table", 1)),
    **dict.fromkeys(
        (
            "delete_before",
            "update_before",
            "read_row_statement",
            "read_by_statement",
            "read_keyed_statements",
            "keyset_page_statement",
            "keyset_rows_statement",
            "delete_keyed_statements",
            "update_keyed_statements",
            "delete_before_statement",
            "update_before_statement",
        ),
        ("table", 0),
    ),
}


def templated(call: ast.Call) -> ast.Call | None:
    """the keyed read (or write, or builder) with the table its caller names at run time (a name or an attribute:
    ``self.table_name``, perhaps through one spelling call) as the
    calling module's template (``"{}"``, judged against ``TableSchema(name=<parameter>)``, as ``FROM {table}`` in SQL
    the caller writes is); None when the read names its table any other way.

    :param call: a keyed read whose table the walk could not read
    :ptype call: ast.Call
    :return: the call reading the template, or None
    :rtype: ast.Call | None
    """
    where = _READ_RELATION.get(_name(call))
    if where is None:
        return None
    named = _argument(call, *where)
    if isinstance(named, ast.Call) and len(named.args) == 1 and not named.keywords:
        # the name passed through one spelling call (``_relation(self._table)``, a quoter or a lower-caser)
        named = named.args[0]
    if not isinstance(named, ast.Name | ast.Attribute):
        return None
    return with_relation(call, "{}")


def with_relation(call: ast.Call, relation: str) -> ast.Call | None:
    """the keyed read (or write, or builder) with the relation it reads named ``relation``, whatever its caller
    passes; None for a helper whose relation argument is not known.

    :param call: a keyed helper call
    :ptype call: ast.Call
    :param relation: the relation's name as a literal
    :ptype relation: str
    :return: the call reading ``relation``, or None
    :rtype: ast.Call | None
    """
    where = _READ_RELATION.get(_name(call))
    if where is None:
        return None
    keyword, position = where
    named = ast.Constant(relation)
    by_keyword = any(given.arg == keyword for given in call.keywords)
    return ast.Call(
        func=call.func,
        args=[named if index == position and not by_keyword else argument for index, argument in enumerate(call.args)],
        keywords=[
            ast.keyword(arg=given.arg, value=named if given.arg == keyword else given.value) for given in call.keywords
        ],
    )


def helper_statements(call: ast.Call) -> Iterator[HelperStatement]:
    """the statements a call to a keyed helper sends, for the walk to judge in the calling function; nothing for any
    other call.

    :param call: a call the walk visits
    :ptype call: ast.Call
    :return: each statement, or one :data:`UNSEEN` statement naming the argument the walk could not read
    :rtype: Iterator[HelperStatement]
    """
    helper = _name(call)
    runner = BUILDER_CALLS.get(helper)
    if runner is not None and runner not in _TABLE_FIRST:
        # a builder's call, its arguments where its runner takes them: the connection the runner takes first stands
        # in as a constant no synthesizer reads
        call = ast.Call(func=call.func, args=[ast.Constant(None), *call.args], keywords=call.keywords)
    synthesize = _SYNTHESIZERS.get(runner or helper)
    if synthesize is None:
        return
    try:
        statements = synthesize(call)
    except _Unseen as unseen:
        yield HelperStatement(helper=helper, sql=UNSEEN, unseen=unseen.argument)
        return
    pins = _argument(call, "by", None)
    merged = "by" if isinstance(pins, ast.Dict) and any(key is None for key in pins.keys) else ""
    for sql in statements:
        yield HelperStatement(helper=helper, sql=sql, merged=merged)


#: the pagers whose every page after the first starts past the last row read, in their order: the argument naming
#: the relation paged and the one naming the order, by helper. ``copy_keyed`` (one ``COPY`` of every row) steps past
#: nothing
_ORDERED_PAGERS: Final[Mapping[str, tuple[str, str]]] = {
    "keyset_page": ("table", "order"),
    "keyset_rows": ("table", "order"),
    "visible_rows": ("table", "order"),
    "read_changed": ("relation", "key"),
}

#: the pagers that read one page unless the call hands them a cursor (``keyset_page(cursor=...)``): one called
#: without reads the first rows of its order, which a tie can reorder but never skip
_ONE_PAGE_UNLESS_CURSOR: Final[frozenset[str]] = frozenset({"keyset_page"})


@dataclass(frozen=True)
class PagerOrder:
    """what a keyset pager's call steps by: a page after the first starts past the last row read in ``order``, so a
    row sharing every order value with that row (and every pinned value, which all its rows share) is skipped unless
    the pinned and ordered columns together hold a key of the relation.

    :ivar helper: the pager's name
    :ivar relation: the relation paged, as its statement names it
    :ivar fixed: the columns the call pins to one value each (``by={"agent_id": a}``, ``IS NULL``); never a column
        pinned to a set (``AnyOf``), whose rows hold many values of it
    :ivar order: the plain columns of the order, as written; a sort by an expression (``lower(name)``) holds none
    :ivar nulls: those of ``fixed`` pinned ``IS NULL`` (every other is pinned to a value, never NULL)
    """

    helper: str
    relation: str
    fixed: frozenset[str]
    order: tuple[str, ...]
    nulls: frozenset[str] = frozenset()

    def covers(self, key: Sequence[str]) -> bool:
        """whether the pinned and ordered columns hold every column of ``key``: then no two rows share a place.

        :param key: a key of the relation
        :ptype key: Sequence[str]
        :return: whether they do
        :rtype: bool
        """
        held = {column.lower() for column in (*self.fixed, *self.order)}
        return bool(key) and all(column.lower() in held for column in key)


def pager_order(call: ast.Call) -> PagerOrder | None:
    """the columns a keyset pager call steps past (:class:`PagerOrder`), as a call to :func:`helper_statements`
    reads its arguments; None for any other call, for a ``keyset_page`` handed no cursor (it reads one page, past
    nothing), and for one whose relation, pins or order the walk cannot read as literals (a column named at run
    time): that call's own statements say it is unknown.

    :param call: a call the walk visits, its arguments resolved
    :ptype call: ast.Call
    :return: what it steps past, or None
    :rtype: PagerOrder | None
    """
    helper = _name(call)
    runner = BUILDER_CALLS.get(helper)
    if runner is not None and runner not in _TABLE_FIRST:
        call = ast.Call(func=call.func, args=[ast.Constant(None), *call.args], keywords=call.keywords)
    arguments = _ORDERED_PAGERS.get(runner or helper)
    if arguments is None:
        return None
    relation_argument, order_argument = arguments
    if (runner or helper) in _ONE_PAGE_UNLESS_CURSOR:
        cursor = _argument(call, "cursor", None)
        if cursor is None or (isinstance(cursor, ast.Constant) and cursor.value is None):
            return None
    try:
        relation, _ = _relation(_argument(call, relation_argument, 1), relation_argument)
        pins = _by(_argument(call, "by", None))
        order_node = _argument(call, order_argument, None)
        if not isinstance(order_node, ast.Tuple | ast.List) or not order_node.elts:
            return None
        order: list[str] = []
        for element in order_node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                # a key column named by its string (``read_changed(key=["id"])``): a sort of it
                order.append(element.value)
                continue
            if not (isinstance(element, ast.Call) and _name(element) == "Sort"):
                return None
            if _argument(element, "expression", None) is None:
                order.append(_text(_argument(element, "column", 0), order_argument))
    except _Unseen:
        # NOSILENT: an argument the walk cannot read is the answer here (no order to judge); the same call's
        # statements (helper_statements) report it as unknown-predicate, so nothing is lost by returning None
        return None
    if any(column.startswith(("{", _EXPRESSION)) for column, _ in pins):
        return None
    fixed = frozenset(column for column, how in pins if how in {"eq", "null"} or how.startswith(_LITERAL))
    nulls = frozenset(column for column, how in pins if how == "null")
    return PagerOrder(helper=helper, relation=relation, fixed=fixed, order=tuple(order), nulls=nulls)


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
