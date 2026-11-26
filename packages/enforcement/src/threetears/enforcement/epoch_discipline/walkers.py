"""the five epoch-discipline detectors: static AST walks, no imports, no execution.

Every hit is keyed by module, enclosing function (its qualified name) and a short symbol naming
what it did (``drop_table``, ``except KvError``, ``ttl=``, ``INSERT conversations``,
``SELECT memories``). The key carries no line number, so a ledger line survives code moving
around it; its count pins how many hits the key has.

SQL is read where it is statically known: a string literal, an f-string (each placeholder read as
``{}``), a concatenation, ``.format`` or ``dedent`` of one, or a name or ``self.``/``cls.``
attribute bound to one in the module, the function or a class body. SQL assembled at run time
elsewhere is not seen; the gate can miss a site, never invent one.

A module's private function (a top-level ``def _name``) that writes is judged where its own module calls it: each
epoch-bypass hit in its body moves to every call site, under the caller's function and its announce and
``bypassing_write`` state (``UPDATE wake_fires via _fenced_update``), so a helper the collection write path alone
calls is the write path, and the same helper called from anywhere else is the bypass it is. A private writer its
module never calls keeps its hit in place.

A keyed update in a migration module (one the migration runner runs, ``EpochDisciplineConfig.migration_modules``) that
stamps every row it writes is heard, not a bypass: its ``values`` (``update_keyed``) or ``set_to`` (``update_before``)
is a dict literal, inline or bound once in the function, whose ``epoch`` key (``EPOCH_COLUMN`` or ``"epoch"``) holds
the shared stamp: ``CallerTransaction.stamp_for(...)``, ``next_epoch()``, or ``rising_stamp(<one of those>, ...)``,
each inline or through a name bound once in the function to one. The runner (``MigrationRunner(announcer=...)``)
announces every row stamped within its run's window, so such a write reaches every follower. An unstamped write in a
migration, a stamped one outside a migration, and a write whose values stamp other columns but not ``epoch`` are each
still the bypass they were.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from threetears.enforcement.common import dotted, iter_python_files, parse_python_path, relative_posix_path
from threetears.enforcement.no_whole_reads.walker import (
    DECLARATION_BLOCK,
    DECLARATION_CALL,
    DECLARED_KINDS,
    DeclaredWhole,
    declared_whole_read,
)
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

#: driver COPY calls that write a table; their first argument is the table. ``copy_from_query`` is not one: it runs
#: ``COPY (query) TO STDOUT``, a read of the query's rows (a change-note restore's keyed notes, L1 2026-11-15)
_COPY_CALLS: Final[frozenset[str]] = frozenset({"copy_records_to_table", "copy_to_table"})

#: backend bulk-write methods, distinctive by name wherever they are called. ``delete_prefixes`` is a store's delete of
#: whole key prefixes: called anywhere but the collection's own (``SchemaBackedCollection.delete_prefixes``, which
#: settles the prefixes when the caller's transaction ends), no pod learns the rows went
_BACKEND_BULK_WRITES: Final[frozenset[str]] = frozenset(
    {"truncate", "load_files", "copy_many", "upsert_many", "delete_many", "delete_prefixes", "upsert_ordered"}
)

#: the bulk writes a collection itself offers as its epoch-advancing primitive, the call the epoch-bypass advice sends
#: callers to: ``SchemaBackedCollection.delete_prefixes`` settles the prefixes it deletes when the caller's
#: transaction ends. A call of one is that primitive, not a bypass, only as ``self.<name>(...)`` written inside a
#: collection class (its name ends ``Collection``) whose own body defines ``<name>``; on any other receiver, or in any
#: other class, the same name is a store's raw bulk write and still fires
_COLLECTION_OWN_BULK_WRITES: Final[frozenset[str]] = frozenset({"delete_prefixes"})

#: backend row writes, counted only on a receiver that names a backend or L3.
_BACKEND_ROW_WRITES: Final[frozenset[str]] = frozenset({"upsert", "delete", "insert", "update"})

#: 3tears' keyed write helpers (``threetears.core.keyed``), each a write of the table it is handed: the verb it is
#: counted as, and the position of its table argument (or its ``table=``). A call is judged where it is written, as
#: the SQL it sends would be; the helper's own body is a primitive. ``delete_before`` / ``update_before`` handed a
#: collection run each page in its ``bypassing_write`` (announced): only ``collection=None`` is a bypass.
_KEYED_WRITES: Final[dict[str, tuple[str, int, str]]] = {
    "update_keyed": ("UPDATE", 1, "table"),
    "delete_keyed": ("DELETE", 1, "table"),
    # a stamped table's keyed delete, sent as its gone mark (threetears.core.collections.gone_marks)
    "mark_gone_keyed": ("UPDATE", 1, "table"),
    "update_from": ("UPDATE", 1, "table"),
    "apply_stage": ("INSERT", 1, "target"),
    "delete_matching": ("DELETE", 1, "table"),
    "delete_before": ("DELETE", 0, "table"),
    "update_before": ("UPDATE", 0, "table"),
}

#: the keyed write helpers that run inside the collection's ``bypassing_write`` when handed one
_KEYED_COLLECTION_WRITES: Final[frozenset[str]] = frozenset({"delete_before", "update_before"})

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

#: the keyed updates a migration may stamp, each onto the keyword holding the columns it sets
_STAMPABLE_UPDATES: Final[dict[str, str]] = {"update_keyed": "values", "update_before": "set_to"}

#: a dict key naming the epoch column: the constant (``row_epoch.EPOCH_COLUMN``) or its text
_EPOCH_KEY_NAME: Final[str] = "EPOCH_COLUMN"
_EPOCH_KEY_TEXT: Final[str] = "epoch"

#: the shared stamp's sources, each a fresh stamp every stamped write draws: ``next_epoch()`` and
#: ``CallerTransaction.stamp_for(...)``; ``rising_stamp(<stamp>, ...)`` carries one into a keyed update's value
_NEXT_EPOCH: Final[str] = "next_epoch"
_STAMP_FOR: Final[str] = "stamp_for"
_STAMP_FOR_OWNER: Final[str] = "CallerTransaction"
_RISING_STAMP: Final[str] = "rising_stamp"

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
#: a database's own catalog, never a table of rows: ``dbstat`` is SQLite's virtual table of a database's pages
#: (``threetears.core.cache.l1_sizes`` measures an L1's tables from it), bounded by the database as ``pg_class`` is
_CATALOG: Final[re.Pattern[str]] = re.compile(
    r"(?i)^(pg_|information_schema|sqlite_|duckdb_|dbstat$|\(|unnest|generate_series)"
)


@dataclass(frozen=True)
class EpochHit:
    """one site a rule's detector found.

    :ivar rule: the rule
    :ivar path: repo-relative posix path of the module
    :ivar line: line of the site
    :ivar function: qualified name of the enclosing function or class, or ``<module>``
    :ivar symbol: what the site did, as the ledgers key it
    :ivar declared: the ``whole_read`` declaration wrapping the site whose kind lets this read through (a whole select
        under a kind letting ``whole-select``, a recovery read under one letting ``calls-whole-read``), or None; it
        lets the hit go only when the declaration is sound (``no_whole_reads.declaration_defects``)
    """

    rule: str
    path: str
    line: int
    function: str
    symbol: str
    declared: DeclaredWhole | None = field(default=None, compare=False)

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
    :ivar unmatched_migrations: migration module globs that matched no module
    """

    hits: list[EpochHit] = field(default_factory=list)
    files_scanned: int = 0
    definitions: set[str] = field(default_factory=set)
    unmatched_exclusions: list[str] = field(default_factory=list)
    unmatched_migrations: list[str] = field(default_factory=list)


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

    def __init__(self, path: str, tree: ast.Module, *, migration: bool = False) -> None:
        """prepare a walk of one module.

        :param path: repo-relative posix path of the module
        :ptype path: str
        :param tree: its parse
        :ptype tree: ast.Module
        :param migration: whether the module is one the migration runner runs (its stamped keyed updates are heard)
        :ptype migration: bool
        :return: nothing
        :rtype: None
        """
        self.path = path
        self._migration = migration
        #: the functions the walk is inside, innermost last
        self._function_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.hits: list[EpochHit] = []
        self.definitions: set[str] = set()
        self._stack: list[str] = []
        self._envs: list[dict[str, str]] = [{}]
        self._announcing: list[bool] = [False]
        self._bypassing: int = 0
        self._attributes: dict[str, str] = {}
        #: per enclosing class, innermost last: (its name, the methods its own body defines)
        self._classes: list[tuple[str, frozenset[str]]] = []
        #: the ``whole_read`` declarations whose wrapped read the walk is inside, innermost last
        self._declared: list[DeclaredWhole] = []
        #: the module's private top-level functions, judged at their callers for epoch-bypass
        self._private: frozenset[str] = frozenset(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("_")
        )
        #: each call of one: (callee, caller's qualified name, line, caller announces or is inside bypassing_write)
        self._private_calls: list[tuple[str, str, int, bool]] = []
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

    def _hit(self, rule: str, node: ast.AST, symbol: str, *, read: str = "") -> None:
        """record one hit at ``node``.

        :param rule: the rule
        :ptype rule: str
        :param node: the site
        :ptype node: ast.AST
        :param symbol: what it did
        :ptype symbol: str
        :param read: for a whole read, its kind as a ``whole_read`` declaration's kind names what it lets through
            (``whole-select``, ``calls-whole-read``): the innermost declaration wrapping the site is kept with the hit
            when its kind lets it; empty for any other hit
        :ptype read: str
        :return: nothing
        :rtype: None
        """
        declared = self._declared[-1] if read and self._declared else None
        if declared is not None and read not in DECLARED_KINDS.get(declared.kind, frozenset()):
            declared = None
        self.hits.append(EpochHit(rule, self.path, getattr(node, "lineno", 0), self._function, symbol, declared))

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
        methods = frozenset(
            each.name for each in node.body if isinstance(each, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        self._classes.append((node.name, methods))
        self.generic_visit(node)
        self._classes.pop()
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
        self._function_nodes.append(node)
        self.generic_visit(node)
        self._function_nodes.pop()
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
        """walk a ``with``, noting whether it opens a cache-bypassing write, and walking the body of a
        ``whole_read_block`` declaration inside it as a wrapped read is walked inside ``whole_read``.

        :param node: the statement
        :ptype node: ast.With | ast.AsyncWith
        :return: nothing
        :rtype: None
        """
        opens = any(
            isinstance(item.context_expr, ast.Call) and _final_name(item.context_expr) == _BYPASSING_WRITE
            for item in node.items
        )
        blocks = [
            declared_whole_read(self.path, self._function, item.context_expr, node.body)
            for item in node.items
            if isinstance(item.context_expr, ast.Call) and _final_name(item.context_expr) == DECLARATION_BLOCK
        ]
        self._bypassing += int(opens)
        for item in node.items:
            self.visit(item)
        self._declared.extend(blocks)
        for statement in node.body:
            self.visit(statement)
        del self._declared[len(self._declared) - len(blocks) :]
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
        if isinstance(node.func, ast.Name) and node.func.id in self._private:
            self._private_calls.append(
                (node.func.id, self._function, node.lineno, self._announcing[-1] or bool(self._bypassing))
            )
        if name is not None:
            self._whole_drop_call(node, name)
            self._ttl_call(node, name)
            self._bypass_call(node, name)
            self._sql_call(node, name)
            self._recovery_read(node, name)
        if name == DECLARATION_CALL and node.args:
            # the read it wraps is walked inside the declaration; its other arguments are literals
            self._declared.append(declared_whole_read(self.path, self._function, node))
            self.visit(node.args[0])
            self._declared.pop()
            for child in (node.func, *node.args[1:], *node.keywords):
                self.visit(child)
        else:
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
        elif self._calls_own_primitive(node, name):
            return
        elif isinstance(node.func, ast.Attribute) and (
            name in _BACKEND_BULK_WRITES or (name in _BACKEND_ROW_WRITES and backend)
        ):
            self._hit(RULE_EPOCH_BYPASS, node, f"backend.{name}")
            if name == "truncate":
                self._hit(RULE_WHOLE_DROP, node, "backend.truncate")
        elif name in _COPY_CALLS:
            table = _text(node.args[0], self._envs[-1], self._attributes) if node.args else None
            self._hit(RULE_EPOCH_BYPASS, node, f"COPY {table or '?'}")
        elif (
            name in _KEYED_WRITES
            and not _handed_a_collection(node, name)
            and not self._stamped_migration_write(node, name)
        ):
            verb, position, keyword = _KEYED_WRITES[name]
            argument = node.args[position] if len(node.args) > position else None
            if argument is None:
                argument = next((each.value for each in node.keywords if each.arg == keyword), None)
            table = _text(argument, self._envs[-1], self._attributes) if argument is not None else None
            self._hit(RULE_EPOCH_BYPASS, node, f"{verb} {table or '{}'}")

    def _stamped_migration_write(self, node: ast.Call, name: str) -> bool:
        """whether ``node`` is a keyed update in a migration module that stamps every row it writes.

        :param node: the call
        :ptype node: ast.Call
        :param name: its final callee name
        :ptype name: str
        :return: whether the module is a migration, ``name`` one of :data:`_STAMPABLE_UPDATES`, and the columns it sets
            are a dict whose ``epoch`` is the shared stamp (:meth:`_is_stamp`)
        :rtype: bool
        """
        if not self._migration or name not in _STAMPABLE_UPDATES:
            return False
        handed = next((each.value for each in node.keywords if each.arg == _STAMPABLE_UPDATES[name]), None)
        columns = self._bound(handed)
        if not isinstance(columns, ast.Dict):
            return False
        epoch = next(
            (value for key, value in zip(columns.keys, columns.values, strict=True) if _names_epoch(key)), None
        )
        return epoch is not None and self._is_stamp(epoch)

    def _bound(self, node: ast.expr | None) -> ast.expr | None:
        """``node``, or for a bare name the one value the enclosing function binds it to.

        :param node: an expression
        :ptype node: ast.expr | None
        :return: the expression; None for a name bound other than exactly once by a plain assignment in the function
        :rtype: ast.expr | None
        """
        if not isinstance(node, ast.Name):
            return node
        if not self._function_nodes:
            return None
        values: list[ast.expr | None] = []
        for inner in _own_nodes(self._function_nodes[-1].body):
            if isinstance(inner, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == node.id for target in inner.targets):
                    values.append(inner.value if len(inner.targets) == 1 else None)
                elif any(node.id in _target_names(target) for target in inner.targets):
                    values.append(None)
            elif isinstance(inner, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                if node.id in _target_names(inner.target):
                    values.append(inner.value if isinstance(inner, ast.AnnAssign) else None)
            elif isinstance(inner, (ast.For, ast.AsyncFor, ast.comprehension)):
                if node.id in _target_names(inner.target):
                    values.append(None)
            elif isinstance(inner, (ast.With, ast.AsyncWith)):
                if any(
                    item.optional_vars is not None and node.id in _target_names(item.optional_vars)
                    for item in inner.items
                ):
                    values.append(None)
        return values[0] if len(values) == 1 else None

    def _is_stamp(self, node: ast.expr) -> bool:
        """whether ``node`` is the shared stamp: ``CallerTransaction.stamp_for(...)``, ``next_epoch()``, or
        ``rising_stamp(<a stamp>, ...)``, inline or through a name the function binds once to one.

        :param node: an expression
        :ptype node: ast.expr
        :return: whether it is
        :rtype: bool
        """
        value = self._bound(node) if isinstance(node, ast.Name) else node
        if not isinstance(value, ast.Call):
            return False
        func = value.func
        name = _final_name(value)
        if name == _NEXT_EPOCH:
            return True
        if name == _STAMP_FOR:
            owner = _segments(dotted(func.value)) if isinstance(func, ast.Attribute) else []
            return bool(owner) and owner[-1] == _STAMP_FOR_OWNER
        if name == _RISING_STAMP:
            stamp = (
                value.args[0]
                if value.args
                else next((each.value for each in value.keywords if each.arg == "stamp"), None)
            )
            return stamp is not None and self._is_stamp(stamp)
        return False

    def _calls_own_primitive(self, node: ast.Call, name: str) -> bool:
        """whether ``node`` is a collection calling its own epoch-advancing bulk write.

        :param node: the call
        :ptype node: ast.Call
        :param name: its final callee name
        :ptype name: str
        :return: whether it is ``self.<name>(...)``, ``name`` one of :data:`_COLLECTION_OWN_BULK_WRITES`, inside a
            collection class whose own body defines ``name``
        :rtype: bool
        """
        if name not in _COLLECTION_OWN_BULK_WRITES or not self._classes:
            return False
        func = node.func
        class_name, methods = self._classes[-1]
        return (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
            and class_name.endswith("Collection")
            and name in methods
        )

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
            self._hit(RULE_WHOLE_REREAD, node, f"SELECT {whole}", read="whole-select")

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
            self._hit(RULE_WHOLE_REREAD, node, f"recovery read {name}", read="calls-whole-read")

    def judged_at_callers(self) -> None:
        """move each epoch-bypass hit in a private top-level function to its module's calls of it, under each
        caller (none where the caller announces or writes inside ``bypassing_write``); a function its module never
        calls keeps its hit. Repeated until no hit sits in a called private function, so a chain of private helpers
        is judged at its first public or write-path caller.

        :return: nothing
        :rtype: None
        """
        for _ in range(len(self._private) + 1):
            moved = False
            kept: list[EpochHit] = []
            for hit in self.hits:
                callers = [call for call in self._private_calls if call[0] == hit.function and call[1] != hit.function]
                if hit.rule != RULE_EPOCH_BYPASS or hit.function not in self._private or not callers:
                    kept.append(hit)
                    continue
                moved = True
                symbol = hit.symbol if " via " in hit.symbol else f"{hit.symbol} via {hit.function}"
                kept.extend(
                    EpochHit(hit.rule, hit.path, line, caller, symbol)
                    for _, caller, line, covered in callers
                    if not covered
                )
            self.hits = kept
            if not moved:
                return


def _names_epoch(key: ast.expr | None) -> bool:
    """whether a dict key names the epoch column: ``EPOCH_COLUMN`` (bare or ``row_epoch.EPOCH_COLUMN``) or
    ``"epoch"``.

    :param key: the key; None for a ``**`` splat
    :ptype key: ast.expr | None
    :return: whether it does
    :rtype: bool
    """
    if isinstance(key, ast.Constant):
        return key.value == _EPOCH_KEY_TEXT
    segments = _segments(dotted(key)) if isinstance(key, (ast.Name, ast.Attribute)) else []
    return bool(segments) and segments[-1] == _EPOCH_KEY_NAME


def _target_names(target: ast.expr) -> set[str]:
    """the names an assignment target binds.

    :param target: the target
    :ptype target: ast.expr
    :return: the names
    :rtype: set[str]
    """
    return {each.id for each in ast.walk(target) if isinstance(each, ast.Name)}


def _handed_a_collection(node: ast.Call, name: str) -> bool:
    """whether a keyed paged write (``delete_before`` / ``update_before``) is handed a collection, whose
    ``bypassing_write`` each page runs in: any ``collection=`` but a literal ``None``.

    :param node: the call
    :ptype node: ast.Call
    :param name: its final callee name
    :ptype name: str
    :return: whether it runs through the collection
    :rtype: bool
    """
    if name not in _KEYED_COLLECTION_WRITES:
        return False
    handed = next((each.value for each in node.keywords if each.arg == "collection"), None)
    return handed is not None and not (isinstance(handed, ast.Constant) and handed.value is None)


_SCANS: BuildOnce[tuple[Path, tuple[Path, ...], tuple[str, ...], tuple[str, ...]], EpochScan] = BuildOnce()


def _walk(
    repo_root: Path, src_roots: tuple[Path, ...], excluded_prefixes: tuple[str, ...], migration_modules: tuple[str, ...]
) -> EpochScan:
    """walk every module under ``src_roots`` once and record every rule's hits.

    :param repo_root: the repo root; paths are relative to it
    :ptype repo_root: Path
    :param src_roots: source trees to walk
    :ptype src_roots: tuple[Path, ...]
    :param excluded_prefixes: repo-relative directory prefixes left unwalked
    :ptype excluded_prefixes: tuple[str, ...]
    :param migration_modules: repo-relative ``fnmatch`` globs of the modules the migration runner runs
    :ptype migration_modules: tuple[str, ...]
    :return: the walk
    :rtype: EpochScan
    """
    scan = EpochScan()
    used: set[str] = set()
    matched: set[str] = set()
    for root in src_roots:
        for path in iter_python_files(root):
            relative = relative_posix_path(path, repo_root)
            excluded = [prefix for prefix in excluded_prefixes if relative.startswith(prefix)]
            used.update(excluded)
            if excluded:
                continue
            migrations = [glob for glob in migration_modules if fnmatch.fnmatchcase(relative, glob)]
            matched.update(migrations)
            tree = parse_python_path(path)
            walk = _ModuleWalk(relative, tree, migration=bool(migrations))
            walk.visit(tree)
            walk.judged_at_callers()
            scan.hits.extend(walk.hits)
            scan.definitions.update(walk.definitions)
            scan.files_scanned += 1
    scan.unmatched_exclusions = [prefix for prefix in excluded_prefixes if prefix not in used]
    scan.unmatched_migrations = [glob for glob in migration_modules if glob not in matched]
    return scan


def scan_epoch_discipline(
    repo_root: Path,
    src_roots: tuple[Path, ...],
    excluded_prefixes: tuple[str, ...] = (),
    migration_modules: tuple[str, ...] = (),
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
    :param migration_modules: repo-relative ``fnmatch`` globs of the modules the migration runner runs, whose keyed
        updates stamping every row they write are heard
    :ptype migration_modules: tuple[str, ...]
    :return: the walk
    :rtype: EpochScan
    """
    key = (repo_root, src_roots, excluded_prefixes, migration_modules)
    return _SCANS.get(key, lambda: _walk(*key))
