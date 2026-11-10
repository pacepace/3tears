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
comprehension) answers :data:`UNSEEN`: the walk reports it ``unknown-predicate``, never a pass. ``empty_tables`` and
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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "KEYED_FRAGMENT_BUILDERS",
    "KEYED_HELPER_CALLS",
    "START_KEYWORDS",
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
        "keyset_page",
        "counts_by",
        "count_at_most",
        "holds_any",
        "read_after",
        "read_past_marks",
        "delete_before",
    }
)

#: builders of a visibility clause reviewed once where it is written: a keyed predicate wherever it is placed
KEYED_FRAGMENT_BUILDERS: Final[frozenset[str]] = frozenset(
    {
        "caller_visible_customer_clause",
        "customer_scope_visibility_clause",
        "three_scope_visibility_clause",
    }
)

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


def _key_batch_columns(node: ast.expr | None) -> list[str]:
    if isinstance(node, ast.Call) and _name(node) == "KeyBatch":
        return _texts(_argument(node, "columns", 0), "keys")
    raise _Unseen("keys")


def _sort_columns(node: ast.expr | None) -> list[str]:
    if not isinstance(node, ast.Tuple | ast.List) or not node.elts:
        raise _Unseen("order")
    columns = []
    for element in node.elts:
        if not (isinstance(element, ast.Call) and _name(element) == "Sort"):
            raise _Unseen("order")
        columns.append(_text(_argument(element, "column", 0), "order"))
    return columns


def _where(conditions: Sequence[str]) -> str:
    return f" WHERE {' AND '.join(conditions)}" if conditions else ""


def _read_keyed(call: ast.Call) -> list[str]:
    table = _text(_argument(call, "table", 1), "table")
    keys = _key_batch_columns(_argument(call, "keys", None))
    pinned, _ = _conditions(_by(_argument(call, "by", None)), first=len(keys) + 1, alias="t")
    unnested = ", ".join(f"${index + 1}" for index in range(len(keys)))
    named = ", ".join(f"k{index}" for index in range(len(keys)))
    joined = " AND ".join(f"t.{_quoted(column)} = k.k{index}" for index, column in enumerate(keys))
    return [f"SELECT * FROM {_quoted(table)} AS t JOIN unnest({unnested}) AS k({named}) ON {joined}{_where(pinned)}"]


def _visible(call: ast.Call) -> list[str]:
    """each ``exists_keyed`` probe a listing's ``visible=`` names, as the keyed read it is; reviewed builders pass."""
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
        if not (isinstance(built, ast.Call) and _name(built) == "exists_keyed"):
            raise _Unseen("visible")
        probed = _text(_argument(built, "table", 0), "visible")
        on = _by(_argument(built, "on", None), "visible")
        if not on:
            raise _Unseen("visible")
        conditions, _ = _conditions([(column, "eq") for column, _ in on] + _by(_argument(built, "by", None), "visible"))
        probes.append(f"SELECT 1 FROM {_quoted(probed)}{_where(conditions)}")
    return probes


def _keyset_page(call: ast.Call) -> list[str]:
    table = _text(_argument(call, "table", 1), "table")
    order = _sort_columns(_argument(call, "order", None))
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    page = [*pinned, f"{_quoted(order[0])} > ${place}"]
    ordered = ", ".join(_quoted(column) for column in order)
    return [f"SELECT * FROM {_quoted(table)}{_where(page)} ORDER BY {ordered} LIMIT {_LIMIT}", *_visible(call)]


def _counts_by(call: ast.Call) -> list[str]:
    table = _text(_argument(call, "table", 1), "table")
    column = _text(_argument(call, "column", None), "column")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    conditions = [*pinned, f"{_quoted(column)} = ANY(${place})"]
    return [f"SELECT {_quoted(column)}, count(*) FROM {_quoted(table)}{_where(conditions)} GROUP BY {_quoted(column)}"]


def _probe(call: ast.Call) -> list[str]:
    table = _text(_argument(call, "table", 1), "table")
    by = _argument(call, "by", None)
    if by is None and _name(call) == "count_at_most":
        raise _Unseen("by")
    pinned, _ = _conditions(_by(by))
    # holds_any reads one row; count_at_most a bound of them
    limit = 1 if _name(call) == "holds_any" else _LIMIT
    return [f"SELECT 1 FROM {_quoted(table)}{_where(pinned)} LIMIT {limit}"]


def _read_after(call: ast.Call) -> list[str]:
    table = _text(_argument(call, "table", 1), "table")
    column = _text(_argument(call, "id_column", None), "id_column")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    conditions = [*pinned, f"{_quoted(column)} > ${place}"]
    return [f"SELECT * FROM {_quoted(table)}{_where(conditions)} ORDER BY {_quoted(column)} LIMIT {_LIMIT}"]


def _read_past_marks(call: ast.Call) -> list[str]:
    relation = _text(_argument(call, "relation", 1), "relation")
    marks = _by(_argument(call, "marks", None), "marks")
    if not marks:
        raise _Unseen("marks")
    past = " OR ".join(f"{_quoted(stamp)} > ${place}" for place, (stamp, _) in enumerate(marks, start=1))
    return [f"SELECT * FROM {_quoted(relation)} WHERE {past}"]


def _delete_before(call: ast.Call) -> list[str]:
    table = _text(_argument(call, "table", 1), "table")
    key = _texts(_argument(call, "key", None), "key")
    column = _text(_argument(call, "column", None), "column")
    pinned, place = _conditions(_by(_argument(call, "by", None)))
    conditions = [*pinned, f"{_quoted(column)} < ${place}"]
    named = ", ".join(_quoted(part) for part in key)
    return [
        f"DELETE FROM {_quoted(table)} WHERE ({named}) IN (SELECT {named} FROM {_quoted(table)}"
        f"{_where(conditions)} ORDER BY {_quoted(column)} LIMIT {_LIMIT})"
    ]


_SYNTHESIZERS: Final = {
    "read_keyed": _read_keyed,
    "read_keyed_arrow": _read_keyed,
    "keyset_page": _keyset_page,
    "counts_by": _counts_by,
    "count_at_most": _probe,
    "holds_any": _probe,
    "read_after": _read_after,
    "read_past_marks": _read_past_marks,
    "delete_before": _delete_before,
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
