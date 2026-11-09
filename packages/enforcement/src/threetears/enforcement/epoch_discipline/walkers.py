"""the five epoch-discipline detectors: static AST walks, no imports, no execution.

Every hit is keyed by module, enclosing function (its qualified name) and a short symbol naming
what it did (``drop_table``, ``except KvError``, ``ttl=``, ``INSERT conversations``,
``SELECT memories``). The key carries no line number, so a ledger line survives code moving
around it; its count pins how many hits the key has.

SQL is read where it is statically known: a string literal, an f-string (each placeholder read as
``{}``), a concatenation, ``.format`` or ``dedent`` of one, or a name or ``self.``/``cls.``
attribute bound to one in the module, the function or a class body. SQL assembled at run time
elsewhere is not seen; the gate can miss a site, never invent one.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from threetears.enforcement.common import dotted, iter_python_files, parse_python_path, relative_posix_path
from threetears.observe import BuildOnce
from threetears.enforcement.epoch_discipline.config import (
    RULE_EPOCH_BYPASS,
    RULE_L2_SWALLOW,
    RULE_TTL_SAFETY_NET,
    RULE_WHOLE_DROP,
    RULE_WHOLE_REREAD,
)

__all__ = [
    "EpochHit",
    "EpochScan",
    "scan_epoch_discipline",
]

#: methods that drop a whole table, scope or L1 copy wherever they are called.
_WHOLE_DROP_CALLS: Final[frozenset[str]] = frozenset(
    {
        "drop_table",
        "drop_cached_table",
        "drop_for_table",
        "drop_scope",
        "drop_all",
        "invalidate_all",
        "invalidate_table",
        "clear_table",
        "clear_cache",
        "reset_cache",
        "replace_all",
        "replace_partitions",
        "reload_all",
        "rebuild_all",
        # a derived cache's whole-table hook, called by anything but the registry's own drop
        "on_table_dropped",
    }
)

#: a receiver segment that names a cache, so its ``.clear()`` drops one whole.
_CACHE_SEGMENT: Final[re.Pattern[str]] = re.compile(r"(?i)(cache|^_?l1$|^_?l1_)")

#: calls whose first argument is SQL text.
_SQL_CALLS: Final[frozenset[str]] = frozenset(
    {"execute", "executemany", "execute_batch", "fetch", "fetchrow", "fetchval", "fetch_arrow", "fetchall", "fetchone"}
)

#: driver COPY calls; their first argument is the table.
_COPY_CALLS: Final[frozenset[str]] = frozenset({"copy_records_to_table", "copy_to_table", "copy_from_query"})

#: backend bulk-write methods, distinctive by name wherever they are called.
_BACKEND_BULK_WRITES: Final[frozenset[str]] = frozenset(
    {"truncate", "load_files", "copy_many", "upsert_many", "delete_many", "upsert_ordered"}
)

#: backend row writes, counted only on a receiver that names a backend or L3.
_BACKEND_ROW_WRITES: Final[frozenset[str]] = frozenset({"upsert", "delete", "insert", "update"})

#: a receiver segment naming L3 or a backend.
_BACKEND_SEGMENT: Final[re.Pattern[str]] = re.compile(r"(?i)(backend|(^|_)l3($|_))")

#: raw KV writes: a put, create, update, delete or purge on a receiver that names a KV bucket, inside a
#: collection class (one whose name ends in ``Collection``). Outside the collection write path, an L2 row or
#: state written this way reaches no follower. Coordination primitives' own buckets (leases, counters) hold no
#: collection's rows and are not walked for it.
_KV_WRITES: Final[frozenset[str]] = frozenset({"put", "create", "update", "delete", "purge"})

#: a receiver segment naming a KV bucket: ``kv``, ``..._kv``, ``kv_...``, ``bucket``.
_KV_WRITE_SEGMENT: Final[re.Pattern[str]] = re.compile(r"(?i)(^_?kv$|_kv$|^_?kv_|bucket$)")

#: a collection's own L2 primitives; called anywhere but the collection write path, they bypass it.
_L2_PRIMITIVES: Final[frozenset[str]] = frozenset({"_save_to_l2", "_delete_from_l2", "_put_l2", "_l2_put"})

#: the calls that announce an out-of-band change; a function making one may write around the collection.
_ANNOUNCE_CALLS: Final[frozenset[str]] = frozenset({"announce_change", "advance_with_change"})

#: the call that opens a cache-bypassing write whose keys the block names.
_BYPASSING_WRITE: Final[str] = "bypassing_write"

#: an exception type in the L2 family: KV, object store, JetStream, or the client's base error.
_L2_ERROR: Final[re.Pattern[str]] = re.compile(
    r"^_?(Nats)?(Kv\w*|ObjectStore\w*|NatsClientError|JetStream\w*Error|APIError|ServiceUnavailableError)$"
)

#: an L2-family type whose catch is an expected condition (a miss, a lost race), not a failure.
_L2_EXPECTED: Final[re.Pattern[str]] = re.compile(
    r"(NotFound|Exists|NotDeclared|Mismatch|WrongLastSequence|KeyDeleted|NoKeys)"
)

#: base error types that cover more than L2; their catch counts only around a KV-looking call.
_L2_BROAD: Final[frozenset[str]] = frozenset({"NatsClientError", "APIError", "ServiceUnavailableError"})

#: calls that read or write a KV bucket or object store.
_KV_CALLS: Final[frozenset[str]] = frozenset(
    {
        "get",
        "put",
        "create",
        "update",
        "delete",
        "purge",
        "keys",
        "history",
        "watch",
        "kv_bucket",
        "ensure_kv_bucket",
        "key_value",
        "create_key_value",
        "get_entry",
        "kv_get",
        "kv_put",
        "kv_create",
        "kv_update",
        "kv_delete",
        "object_store",
        "put_object",
        "get_object",
    }
)

#: a receiver segment that names a KV bucket, object store or L2.
_KV_SEGMENT: Final[re.Pattern[str]] = re.compile(r"(?i)(kv|bucket|(^|_)l2($|_)|object_store)")

#: the helper every degrading L2 handler routes through.
_DEGRADE_HELPER: Final[str] = "degrade_l2"

#: keyword arguments that set an expiry.
_TTL_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "ttl",
        "ttl_s",
        "ttl_seconds",
        "max_age",
        "max_age_s",
        "max_age_seconds",
        "expiry",
        "expire",
        "expires",
        "expire_after",
        "cache_ttl",
        "l1_ttl",
        "l2_ttl",
    }
)

#: a class-body attribute that sets an expiry.
_TTL_ATTRIBUTE: Final[re.Pattern[str]] = re.compile(r"(?i)(^|_)(ttl|max_age|expiry|expire_after)(_s|_seconds)?$")

#: a call that takes an expiry for a cache, a collection or a bucket.
_TTL_CALLEE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(^(kv_bucket|key_value|create_key_value|keyvalueconfig|streamconfig|objectstoreconfig|object_store|"
    r"put|create|update|kv_put|kv_create|kv_update|ttlcache|cached)$|cache$|collection$|snapshot$|bucket$)"
)

#: a function name that says it rebuilds, recovers, reloads, warms or re-proves.
_RECOVERY_NAME: Final[re.Pattern[str]] = re.compile(
    r"(rebuild|recover|reload|resync|warm|refill|repair|reseed|catch_up|reread|retained_whole|left_out|full_check|"
    r"full_scan|resnapshot|refresh_all|load_all)"
)

#: L3 reads inside a recovery-named function.
_L3_READS: Final[frozenset[str]] = frozenset(
    {
        "fetch",
        "fetch_arrow",
        "fetchrow",
        "fetchval",
        "fetch_from_store",
        "fetch_all",
        "fetch_scope",
        "scan",
        "list_entities",
        "list_all",
        "load_all",
        "read_all",
        "rows_at",
        "l3_pages",
        "_l3_pages",
        "keyset_pages",
    }
)

#: helpers that read a whole table, scope or group set from L3, by the shape of their name.
_WHOLE_READ_HELPER: Final[re.Pattern[str]] = re.compile(
    r"^_?(read_(table|tables|scope|scopes|group|groups|rows|all|whole|unpaged|pages|keyset)|read_and_publish\w*|"
    r"copy_table|load_all\w*|fetch_all\w*)$"
)

#: a receiver segment that names an L3 handle.
_L3_SEGMENT: Final[re.Pattern[str]] = re.compile(r"(?i)(pool|conn|backend|(^|_)l3($|_)|store|(^|_)db($|_)|driver)")

#: L3 reads distinctive enough to count on any receiver.
_L3_DISTINCT_READS: Final[frozenset[str]] = frozenset(
    {"fetch_from_store", "list_entities", "rows_at", "fetch_scope", "l3_pages", "_l3_pages", "keyset_pages"}
)

_SQL_COMMENT: Final[re.Pattern[str]] = re.compile(r"--[^\n]*")
_NAME: Final[str] = r"([\w.\"{}]+)"
_SQL_WRITE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:(INSERT)(?:\s+OR\s+\w+)?\s+INTO|(DELETE)\s+FROM|(TRUNCATE)(?:\s+TABLE)?|(MERGE)\s+INTO|^(COPY))"
    rf"\s+(?:ONLY\s+)?{_NAME}",
    re.IGNORECASE,
)
_SQL_UPDATE: Final[re.Pattern[str]] = re.compile(
    rf"(?<!FOR )(?<!DO )\bUPDATE\s+(?:ONLY\s+)?{_NAME}(?:\s+(?:AS\s+)?\w+)?\s+SET\b", re.IGNORECASE
)
_SQL_SELECT_FROM: Final[re.Pattern[str]] = re.compile(
    rf"^(?:WITH\b.*?\)\s*)?SELECT\b.*?\bFROM\s+{_NAME}", re.IGNORECASE
)
_SQL_NARROWS: Final[re.Pattern[str]] = re.compile(r"\b(WHERE|LIMIT|FETCH\s+FIRST)\b", re.IGNORECASE)
_SQL_VERB: Final[re.Pattern[str]] = re.compile(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE|TRUNCATE|COPY|MERGE)\b", re.I)
_CATALOG: Final[re.Pattern[str]] = re.compile(
    r"(?i)^(pg_|information_schema|sqlite_|duckdb_|\(|unnest|generate_series)"
)


@dataclass(frozen=True)
class EpochHit:
    """one site a rule's detector found.

    :ivar rule: the rule
    :ivar path: repo-relative posix path of the module
    :ivar line: line of the site
    :ivar function: qualified name of the enclosing function or class, or ``<module>``
    :ivar symbol: what the site did, as the ledgers key it
    """

    rule: str
    path: str
    line: int
    function: str
    symbol: str

    @property
    def key(self) -> tuple[str, str, str]:
        """the ledger key: module, function, symbol.

        :return: the key
        :rtype: tuple[str, str, str]
        """
        return (self.path, self.function, self.symbol)


@dataclass
class EpochScan:
    """everything one walk found.

    :ivar hits: every hit of every rule, in walk order
    :ivar files_scanned: modules parsed
    :ivar definitions: ``<path>::<qualified name>`` of every function and class seen, so a
        primitive entry naming none is stale
    :ivar unmatched_exclusions: excluded prefixes that matched no module
    """

    hits: list[EpochHit] = field(default_factory=list)
    files_scanned: int = 0
    definitions: set[str] = field(default_factory=set)
    unmatched_exclusions: list[str] = field(default_factory=list)


def _strings_bound(body: list[ast.stmt], env: dict[str, str]) -> None:
    """record the string each simple assignment in ``body`` binds, in order.

    :param body: statements to read
    :ptype body: list[ast.stmt]
    :param env: name onto bound string, updated in place
    :ptype env: dict[str, str]
    :return: nothing
    :rtype: None
    """
    for statement in body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target, value = statement.target, statement.value
        if isinstance(target, ast.Name) and value is not None:
            text = _text(value, env, {})
            if text is not None:
                env[target.id] = text


def _text(node: ast.expr, env: dict[str, str], attributes: dict[str, str]) -> str | None:
    """the statically known text of a string expression.

    :param node: the expression
    :ptype node: ast.expr
    :param env: names bound to strings in scope
    :ptype env: dict[str, str]
    :param attributes: class-body names bound to strings in the module
    :ptype attributes: dict[str, str]
    :return: the text, each unknown interpolation read as ``{}``; ``None`` when it is not known
    :rtype: str | None
    """
    result: str | None = None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        result = node.value
    elif isinstance(node, ast.JoinedStr):
        result = "".join(str(part.value) if isinstance(part, ast.Constant) else "{}" for part in node.values)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _text(node.left, env, attributes)
        right = _text(node.right, env, attributes)
        if left is not None:
            result = left + (right if right is not None else "{}")
    elif isinstance(node, ast.Name):
        result = env.get(node.id)
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        result = attributes.get(node.attr)
    elif isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"format", "strip", "lstrip"}:
            result = _text(func.value, env, attributes)
        elif node.args and (
            (isinstance(func, ast.Name) and func.id in {"dedent", "SQL", "text"})
            or (isinstance(func, ast.Attribute) and func.attr in {"dedent", "SQL", "text"})
        ):
            result = _text(node.args[0], env, attributes)
    return result


def _normal(sql: str) -> str:
    """SQL with comments removed and whitespace collapsed.

    :param sql: the text
    :ptype sql: str
    :return: one-line SQL
    :rtype: str
    """
    return " ".join(_SQL_COMMENT.sub(" ", sql).split())


def _sql_writes(sql: str) -> list[tuple[str, str]]:
    """every write a statement makes, as ``(verb, table)``.

    :param sql: one-line SQL
    :ptype sql: str
    :return: the writes, in text order
    :rtype: list[tuple[str, str]]
    """
    writes: list[tuple[str, str]] = []
    for match in _SQL_WRITE.finditer(sql):
        verb = next(group for group in match.groups()[:5] if group)
        writes.append((verb.upper(), match.group(6)))
    for match in _SQL_UPDATE.finditer(sql):
        if match.group(1).upper() != "SET":
            writes.append(("UPDATE", match.group(1)))
    return writes


def _whole_delete(sql: str, verb: str, table: str) -> bool:
    """whether a write removes every row of its table.

    :param sql: one-line SQL
    :ptype sql: str
    :param verb: the write's verb
    :ptype verb: str
    :param table: the table it names
    :ptype table: str
    :return: ``True`` for a TRUNCATE, or a DELETE with no WHERE after its table
    :rtype: bool
    """
    whole = verb == "TRUNCATE"
    if verb == "DELETE":
        tail = sql.split(table, 1)[1] if table in sql else sql
        whole = re.search(r"\bWHERE\b", tail, re.IGNORECASE) is None
    return whole


def _whole_select(sql: str) -> str | None:
    """the table a read takes whole: a SELECT ... FROM with no WHERE, LIMIT or FETCH FIRST.

    :param sql: one-line SQL
    :ptype sql: str
    :return: the table, or ``None`` when the read is narrowed, may be narrowed by text the walk cannot
        see, or is not a SELECT of a table
    :rtype: str | None
    """
    table: str | None = None
    match = _SQL_SELECT_FROM.match(sql)
    unknown_tail = match is not None and "{}" in sql[match.end() :]
    if (
        match is not None
        and not unknown_tail
        and _SQL_NARROWS.search(sql) is None
        and _CATALOG.match(match.group(1)) is None
    ):
        table = match.group(1)
    return table


def _segments(spelling: str | None) -> list[str]:
    """the dotted segments of a receiver, without ``self`` / ``cls``.

    :param spelling: dotted receiver, or ``None``
    :ptype spelling: str | None
    :return: its segments
    :rtype: list[str]
    """
    return [part for part in (spelling or "").split(".") if part and part not in {"self", "cls"}]


def _type_names(node: ast.expr | None) -> list[str]:
    """the final names of the types an ``except`` clause catches.

    :param node: the clause's type expression
    :ptype node: ast.expr | None
    :return: final names, in order
    :rtype: list[str]
    """
    names: list[str] = []
    elements = node.elts if isinstance(node, ast.Tuple) else ([node] if node is not None else [])
    for element in elements:
        if isinstance(element, ast.Name):
            names.append(element.id)
        elif isinstance(element, ast.Attribute):
            names.append(element.attr)
    return names


def _own_nodes(body: list[ast.stmt]) -> Iterator[ast.AST]:
    """every node under ``body``, not descending into nested functions, lambdas or classes.

    :param body: statements
    :ptype body: list[ast.stmt]
    :return: the nodes
    :rtype: Iterator[ast.AST]
    """
    stack: list[ast.AST] = list(reversed(body))
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                stack.append(child)


def _final_name(call: ast.Call) -> str | None:
    """the final name of a call's callee.

    :param call: the call
    :ptype call: ast.Call
    :return: ``f`` for ``f(...)`` and ``x.f(...)``, else ``None``
    :rtype: str | None
    """
    func = call.func
    name: str | None = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    return name


def _is_unset(value: ast.expr) -> bool:
    """whether a keyword value leaves the expiry unset: ``None``, ``0`` or ``False``.

    :param value: the value
    :ptype value: ast.expr
    :return: whether it sets nothing
    :rtype: bool
    """
    return isinstance(value, ast.Constant) and value.value in (None, 0, False)


class _ModuleWalk(ast.NodeVisitor):
    """walks one module, recording each rule's hits under their enclosing function."""

    def __init__(self, path: str, tree: ast.Module) -> None:
        """prepare a walk of one module.

        :param path: repo-relative posix path of the module
        :ptype path: str
        :param tree: its parse
        :ptype tree: ast.Module
        :return: nothing
        :rtype: None
        """
        self.path = path
        self.hits: list[EpochHit] = []
        self.definitions: set[str] = set()
        self._stack: list[str] = []
        self._envs: list[dict[str, str]] = [{}]
        self._announcing: list[bool] = [False]
        self._bypassing: int = 0
        self._attributes: dict[str, str] = {}
        _strings_bound(tree.body, self._envs[0])
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                _strings_bound(node.body, self._attributes)
        self._announcing[0] = self._calls_any(tree.body, _ANNOUNCE_CALLS)

    @property
    def _function(self) -> str:
        """the qualified name of the scope the walk is in.

        :return: dotted qualified name, or ``<module>``
        :rtype: str
        """
        return ".".join(self._stack) if self._stack else "<module>"

    @staticmethod
    def _calls_any(body: list[ast.stmt], names: frozenset[str]) -> bool:
        """whether ``body`` itself calls any of ``names``.

        :param body: statements
        :ptype body: list[ast.stmt]
        :param names: callee final names
        :ptype names: frozenset[str]
        :return: whether it does
        :rtype: bool
        """
        return any(isinstance(node, ast.Call) and _final_name(node) in names for node in _own_nodes(body))

    def _hit(self, rule: str, node: ast.AST, symbol: str) -> None:
        """record one hit at ``node``.

        :param rule: the rule
        :ptype rule: str
        :param node: the site
        :ptype node: ast.AST
        :param symbol: what it did
        :ptype symbol: str
        :return: nothing
        :rtype: None
        """
        self.hits.append(EpochHit(rule, self.path, getattr(node, "lineno", 0), self._function, symbol))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """walk a class body, reading its expiry attributes.

        :param node: the class
        :ptype node: ast.ClassDef
        :return: nothing
        :rtype: None
        """
        self._stack.append(node.name)
        self.definitions.add(f"{self.path}::{self._function}")
        for statement in node.body:
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value = statement.targets[0], statement.value
            elif isinstance(statement, ast.AnnAssign):
                target, value = statement.target, statement.value
            if (
                isinstance(target, ast.Name)
                and value is not None
                and _TTL_ATTRIBUTE.search(target.id)
                and not _is_unset(value)
            ):
                self._hit(RULE_TTL_SAFETY_NET, statement, f"{target.id}=")
        self.generic_visit(node)
        self._stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """walk a function with its own string bindings and announce status.

        :param node: the function
        :ptype node: ast.FunctionDef | ast.AsyncFunctionDef
        :return: nothing
        :rtype: None
        """
        self._stack.append(node.name)
        self.definitions.add(f"{self.path}::{self._function}")
        env = dict(self._envs[-1])
        for inner in _own_nodes(node.body):
            if isinstance(inner, (ast.Assign, ast.AnnAssign)):
                _strings_bound([inner], env)
        self._envs.append(env)
        self._announcing.append(self._calls_any(node.body, _ANNOUNCE_CALLS))
        self.generic_visit(node)
        self._announcing.pop()
        self._envs.pop()
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """walk a function.

        :param node: the function
        :ptype node: ast.FunctionDef
        :return: nothing
        :rtype: None
        """
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """walk a coroutine function.

        :param node: the function
        :ptype node: ast.AsyncFunctionDef
        :return: nothing
        :rtype: None
        """
        self._visit_function(node)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        """walk a ``with``, noting whether it opens a cache-bypassing write.

        :param node: the statement
        :ptype node: ast.With | ast.AsyncWith
        :return: nothing
        :rtype: None
        """
        opens = any(
            isinstance(item.context_expr, ast.Call) and _final_name(item.context_expr) == _BYPASSING_WRITE
            for item in node.items
        )
        self._bypassing += int(opens)
        self.generic_visit(node)
        self._bypassing -= int(opens)

    def visit_With(self, node: ast.With) -> None:
        """walk a ``with``.

        :param node: the statement
        :ptype node: ast.With
        :return: nothing
        :rtype: None
        """
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """walk an ``async with``.

        :param node: the statement
        :ptype node: ast.AsyncWith
        :return: nothing
        :rtype: None
        """
        self._visit_with(node)

    def _visit_try(self, node: ast.Try | ast.TryStar) -> None:
        """hold each L2-family catch to re-raising or degrading through the helper.

        A specific KV / object-store / JetStream type always counts. The client's base error, which
        a request or a publish raises too, counts only when the ``try`` body makes a KV-looking call.

        :param node: the statement
        :ptype node: ast.Try | ast.TryStar
        :return: nothing
        :rtype: None
        """
        touches_kv = any(
            isinstance(inner, ast.Call)
            and (
                _final_name(inner) in _KV_CALLS
                and (
                    not isinstance(inner.func, ast.Attribute)
                    or any(_KV_SEGMENT.search(part) for part in _segments(dotted(inner.func.value)))
                    or _final_name(inner) not in {"get", "put", "create", "update", "delete"}
                )
            )
            for inner in _own_nodes(node.body)
        )
        for handler in node.handlers:
            caught = [
                name
                for name in _type_names(handler.type)
                if _L2_ERROR.match(name)
                and not _L2_EXPECTED.search(name)
                and (touches_kv or name.lstrip("_") not in _L2_BROAD)
            ]
            settled = any(
                isinstance(inner, ast.Raise) or (isinstance(inner, ast.Call) and _final_name(inner) == _DEGRADE_HELPER)
                for inner in _own_nodes(handler.body)
            )
            if caught and not settled:
                self._hit(RULE_L2_SWALLOW, handler, "except " + "|".join(caught))
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """walk a ``try``.

        :param node: the statement
        :ptype node: ast.Try
        :return: nothing
        :rtype: None
        """
        self._visit_try(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        """walk a ``try`` with ``except*``.

        :param node: the statement
        :ptype node: ast.TryStar
        :return: nothing
        :rtype: None
        """
        self._visit_try(node)

    def visit_Call(self, node: ast.Call) -> None:
        """run every call-shaped detector over one call.

        :param node: the call
        :ptype node: ast.Call
        :return: nothing
        :rtype: None
        """
        name = _final_name(node)
        if name is not None:
            self._whole_drop_call(node, name)
            self._ttl_call(node, name)
            self._bypass_call(node, name)
            self._sql_call(node, name)
            self._recovery_read(node, name)
        self.generic_visit(node)

    def _whole_drop_call(self, node: ast.Call, name: str) -> None:
        """a call that drops a whole table, scope or cache.

        :param node: the call
        :ptype node: ast.Call
        :param name: its final callee name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        receiver = dotted(node.func.value) if isinstance(node.func, ast.Attribute) else None
        segments = _segments(receiver)
        if self._announcing[-1]:
            return
        if name in _WHOLE_DROP_CALLS:
            self._hit(RULE_WHOLE_DROP, node, name)
        elif name == "clear" and not node.args and segments and _CACHE_SEGMENT.search(segments[-1]):
            self._hit(RULE_WHOLE_DROP, node, f"{segments[-1]}.clear")

    def _ttl_call(self, node: ast.Call, name: str) -> None:
        """a call that sets an expiry on a cache, collection or bucket.

        :param node: the call
        :ptype node: ast.Call
        :param name: its final callee name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        if not _TTL_CALLEE.search(name):
            return
        for keyword in node.keywords:
            if keyword.arg in _TTL_KEYWORDS and not _is_unset(keyword.value):
                self._hit(RULE_TTL_SAFETY_NET, node, f"{name}({keyword.arg}=)")
            elif keyword.arg is None:
                # a ``**`` splat may carry an expiry under any of those names: the gate cannot see inside it
                self._hit(RULE_TTL_SAFETY_NET, node, f"{name}(**)")

    def _bypass_call(self, node: ast.Call, name: str) -> None:
        """a backend write or a collection L2 put outside the write path.

        :param node: the call
        :ptype node: ast.Call
        :param name: its final callee name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        if self._announcing[-1] or self._bypassing:
            return
        segments = _segments(dotted(node.func.value) if isinstance(node.func, ast.Attribute) else None)
        backend = any(_BACKEND_SEGMENT.search(segment) for segment in segments)
        if name in _L2_PRIMITIVES:
            self._hit(RULE_EPOCH_BYPASS, node, name)
        elif (
            name in _KV_WRITES
            and segments
            and _KV_WRITE_SEGMENT.search(segments[-1])
            and any(part.endswith("Collection") for part in self._stack)
        ):
            self._hit(RULE_EPOCH_BYPASS, node, f"kv.{name}")
        elif isinstance(node.func, ast.Attribute) and (
            name in _BACKEND_BULK_WRITES or (name in _BACKEND_ROW_WRITES and backend)
        ):
            self._hit(RULE_EPOCH_BYPASS, node, f"backend.{name}")
            if name == "truncate":
                self._hit(RULE_WHOLE_DROP, node, "backend.truncate")
        elif name in _COPY_CALLS:
            table = _text(node.args[0], self._envs[-1], self._attributes) if node.args else None
            self._hit(RULE_EPOCH_BYPASS, node, f"COPY {table or '?'}")

    def _sql_call(self, node: ast.Call, name: str) -> None:
        """SQL text a call runs: a raw write, a whole-table delete, a whole-table read.

        :param node: the call
        :ptype node: ast.Call
        :param name: its final callee name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        if name not in _SQL_CALLS or not isinstance(node.func, ast.Attribute):
            return
        argument = (
            node.args[0]
            if node.args
            else next(
                (keyword.value for keyword in node.keywords if keyword.arg in {"query", "sql", "statement"}), None
            )
        )
        text = _text(argument, self._envs[-1], self._attributes) if argument is not None else None
        if text is None or not _SQL_VERB.match(_SQL_COMMENT.sub(" ", text)):
            return
        sql = _normal(text)
        for verb, table in _sql_writes(sql):
            if _whole_delete(sql, verb, table) and not self._announcing[-1]:
                self._hit(RULE_WHOLE_DROP, node, f"{verb} {table}")
            if not self._announcing[-1] and not self._bypassing:
                self._hit(RULE_EPOCH_BYPASS, node, f"{verb} {table}")
        whole = _whole_select(sql)
        if whole is not None:
            self._hit(RULE_WHOLE_REREAD, node, f"SELECT {whole}")

    def _recovery_read(self, node: ast.Call, name: str) -> None:
        """an L3 read inside a function whose name says it rebuilds, recovers or warms.

        :param node: the call
        :ptype node: ast.Call
        :param name: its final callee name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        if not self._stack or not _RECOVERY_NAME.search(self._stack[-1]):
            return
        segments = _segments(dotted(node.func.value) if isinstance(node.func, ast.Attribute) else None)
        reads = name in _L3_READS and (
            name in _L3_DISTINCT_READS or any(_L3_SEGMENT.search(segment) for segment in segments)
        )
        if reads or _WHOLE_READ_HELPER.match(name):
            self._hit(RULE_WHOLE_REREAD, node, f"recovery read {name}")


_SCANS: BuildOnce[tuple[Path, tuple[Path, ...], tuple[str, ...]], EpochScan] = BuildOnce()


def _walk(repo_root: Path, src_roots: tuple[Path, ...], excluded_prefixes: tuple[str, ...]) -> EpochScan:
    """walk every module under ``src_roots`` once and record every rule's hits.

    :param repo_root: the repo root; paths are relative to it
    :ptype repo_root: Path
    :param src_roots: source trees to walk
    :ptype src_roots: tuple[Path, ...]
    :param excluded_prefixes: repo-relative directory prefixes left unwalked
    :ptype excluded_prefixes: tuple[str, ...]
    :return: the walk
    :rtype: EpochScan
    """
    scan = EpochScan()
    used: set[str] = set()
    for root in src_roots:
        for path in iter_python_files(root):
            relative = relative_posix_path(path, repo_root)
            excluded = [prefix for prefix in excluded_prefixes if relative.startswith(prefix)]
            used.update(excluded)
            if excluded:
                continue
            tree = parse_python_path(path)
            walk = _ModuleWalk(relative, tree)
            walk.visit(tree)
            scan.hits.extend(walk.hits)
            scan.definitions.update(walk.definitions)
            scan.files_scanned += 1
    scan.unmatched_exclusions = [prefix for prefix in excluded_prefixes if prefix not in used]
    return scan


def scan_epoch_discipline(
    repo_root: Path, src_roots: tuple[Path, ...], excluded_prefixes: tuple[str, ...] = ()
) -> EpochScan:
    """every rule's hits under ``src_roots``.

    The walk is kept for the process, keyed by its inputs, so the five rule gates of one repo
    read one walk.

    :param repo_root: the repo root; paths are relative to it
    :ptype repo_root: Path
    :param src_roots: source trees to walk
    :ptype src_roots: tuple[Path, ...]
    :param excluded_prefixes: repo-relative directory prefixes left unwalked
    :ptype excluded_prefixes: tuple[str, ...]
    :return: the walk
    :rtype: EpochScan
    """
    return _SCANS.get((repo_root, src_roots, excluded_prefixes), lambda: _walk(repo_root, src_roots, excluded_prefixes))
