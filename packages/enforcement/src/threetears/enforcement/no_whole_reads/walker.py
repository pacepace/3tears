"""the code half of the no-whole-reads rule: a static walk of every module, no imports, no execution.

Each module is walked once. What it records, each under the qualified name of the function it sits in:

- every SQL statement written in the code (a string literal, an f-string with each placeholder read as ``{}``, a
  concatenation of them), judged by :func:`~threetears.enforcement.no_whole_reads.sql.judge_sql` where it is
  written, so a statement built by a helper is found in the helper. A bare ``SELECT ... FROM <table>`` (a head) is
  judged where it is run: passed straight to ``execute`` / ``fetch`` (or through a name bound to it) it is a
  ``whole-select``; handed to a pager, the pager call is judged;
- a pager or fingerprint helper called with no narrowing keyword (``where=``, ``keys=``, ...), or with one that
  may be None (``where=None if states is None else ...``): ``unkeyed-pager`` / ``may-read-whole``; one narrowed
  only by the key of a loop over every key of a module constant (``for source in SOURCES: read(where=source)``):
  ``whole-by-loop``; a helper that always reads whole (``copy_table``): ``whole-read``;
- in the code a repo names as load code (``verify_prefixes``), a count or fingerprint even of one key's rows:
  ``verify-aggregate``;
- a flag asking for a whole read (``whole=True``, ``whole_table=True``), or a table put in a set of tables to read
  whole (``self._whole.add(t)``, ``self._whole |= ...``): ``whole-flag`` / ``marks-whole``;
- a backend's ``truncate``: ``truncate-call``.

Then the call graph, once every module is walked:

- a call into a function of the repo that itself reads whole (one level: it holds a read hit above), with no
  narrowing argument: ``calls-whole-read``. So ``ScopeEpochs.settled`` -> ``self.snapshot()`` and
  ``_check_counts`` -> ``self._store.count(table)`` are found where they are called, not only where the SQL is;
- a call whose name says it reads, rebuilds or warms whole (``_stage_whole``, ``warm``, ``_rebuild_scopes``,
  ``settle_whole``, ``refill_from_l3``) whose target reaches any hit through the call graph, or which the walk
  cannot find in the repo: ``whole-call``. A ``_rebuild`` that touches no table is not named.

A call resolves to a ``self`` method of its own class, a module function (its own, or the repo module it was
imported from), or the repo's methods of that name (narrowed to the classes named for the receiver:
``self._store`` reaches ``*Store*``); Protocol and abstract stubs are no target. A name that may reach more than one
function counts only when every one of them does. Migrations (a ``migrations`` directory) are schema changes and
are not walked; DuckDB relations a module registers from in-memory rows are the rows being written, not a table.

The walk can miss SQL assembled at run time from pieces it cannot see; it never invents a statement.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from threetears.enforcement.common import dotted, iter_python_files, parse_python_path, relative_posix_path
from threetears.enforcement.no_whole_reads.sql import HEAD, KEYED_COUNT, is_sql, judge_sql

__all__ = [
    "DEFAULT_NARROWING_KEYWORDS",
    "DEFAULT_PAGER_CALLS",
    "WholeReadHit",
    "WholeReadScan",
    "scan_whole_reads",
]

#: 3tears' helpers that read every row they are not narrowed to: a keyset pager over a head, a whole-table copy, a
#: fingerprint. Called with no narrowing keyword, they read the table whole.
DEFAULT_PAGER_CALLS: Final[frozenset[str]] = frozenset(
    {
        "keyset_pages",
        "keyset_arrow_pages",
        "read_keyset_pages",
        "read_l3_rows",
        "l3_row_pages",
        "read_l3_arrow",
        "_l3_arrow_pages",
        "l3_fingerprint",
        "relation_fingerprint",
        "relation_fingerprint_groups",
        "read_counted",
        "export_select",
        "export_part",
        "l3_pages",
        "_l3_pages",
        "read_all",
    }
)

#: helpers that read a whole table whatever they are passed
_ALWAYS_WHOLE_CALLS: Final[frozenset[str]] = frozenset({"copy_table"})

#: keywords that narrow a pager or a reading helper to the rows a change named
DEFAULT_NARROWING_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "where",
        "where_in",
        "where_any",
        "keys",
        "key_values",
        "ids",
        "scope",
        "scopes",
        "partition",
        "partitions",
        "states",
        "state",
        "run_ids",
        "rows",
        "changed",
        "since",
        "after",
    }
)

#: the kinds that read data whole: a function holding one is a whole-read function for ``calls-whole-read``
_READ_KINDS: Final[frozenset[str]] = frozenset(
    {
        "whole-select",
        "whole-aggregate",
        "unbound-where",
        "unkeyed-pager",
        "may-read-whole",
        "whole-read",
        "whole-by-loop",
        "verify-aggregate",
    }
)

#: calls whose first argument is SQL text that is run as written
_EXEC_CALLS: Final[frozenset[str]] = frozenset(
    {
        "execute",
        "executemany",
        "fetch",
        "fetchrow",
        "fetchval",
        "fetchall",
        "fetchone",
        "fetch_arrow",
        "fetch_page",
        "fetch_arrow_page",
        "fetch_df",
        "sql",
        "query",
        "l3_fetch_arrow",
    }
)

#: a callee name that says it reads, rebuilds or warms whole
_WHOLE_CALL: Final[re.Pattern[str]] = re.compile(
    r"(?:^|_)(whole|rebuild|rebuilds|rebuilt|warm|warms|warm_all|reload|reloads|resync|refill|reseed|resnapshot|"
    r"reread|full_check|full_scan|full_copy|full_reload|full_rebuild|refresh_all|load_all|replace_all)(?:_|$)"
)

#: a callee name that only asks about wholeness, or hears of it, or is not about a table: a composite key read
#: whole (``build_whole_key_select_sql``), pydantic's ``model_rebuild``
_WHOLE_QUESTION: Final[re.Pattern[str]] = re.compile(
    r"^_?(is|has|was|on|when|whether|why|count|say|log)_|whole_key|^model_rebuild$"
)

#: a receiver a pager is called on: a store, a handle, a datasource client
_PAGER_RECEIVER: Final[re.Pattern[str]] = re.compile(
    r"(?i)(store|backend|(^|_)l[13]($|_)|pool|conn|(^|_)db($|_)|warehouse|storage|client|proxy|reader|support)"
)

#: a positional argument's name that says it filters a pager's rows (not ``key``, which it pages on)
_PAGER_FILTER: Final[re.Pattern[str]] = re.compile(r"(?i)(filter|where|partition|scopes?$|states$|ids$)")

#: an argument's name that says it narrows: filters, keys, a scope
_NARROWING_NAME: Final[re.Pattern[str]] = re.compile(
    r"(?i)(filter|where|keys?$|ids?$|scope|partition|states?$|rows$|changed)"
)

#: keywords asking a callee for a whole read
_WHOLE_FLAGS: Final[frozenset[str]] = frozenset(
    {"whole", "whole_table", "full_reload", "rebuild", "reload", "force_full", "read_whole"}
)

#: a receiver segment naming a set of tables to read whole
_WHOLE_SET: Final[re.Pattern[str]] = re.compile(r"(?i)(^|_)whole($|_)")

#: a receiver segment naming a store, an L3 or L1 handle, or a copy
_STORE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(store|backend|(^|_)l[13]($|_)|pool|conn|(^|_)db($|_)|warehouse|storage|(^|_)copy($|_)|follower|epochs|"
    r"keeper|repo$|registry)"
)

#: calls that say something, whose string arguments are prose, not SQL
_PROSE_CALLS: Final[re.Pattern[str]] = re.compile(
    r"^(debug|info|warning|warn|error|exception|critical|log|print|_say|say|fail|skip|xfail|contain)$|"
    r"(Error|Exception|Warning)$"
)


@dataclass(frozen=True)
class WholeReadHit:
    """one site the walk found.

    :ivar path: repo-relative posix path of the module
    :ivar line: line of the site
    :ivar function: qualified name of the enclosing function or class, or ``<module>``
    :ivar symbol: ``<kind> <what>``, e.g. ``whole-aggregate {}`` or ``unkeyed-pager keyset_arrow_pages``
    :ivar statement: the statement or the call, one line, for the reader
    """

    path: str
    line: int
    function: str
    symbol: str
    statement: str

    @property
    def kind(self) -> str:
        """the kind, the symbol's first word.

        :return: the kind
        :rtype: str
        """
        return self.symbol.split(" ", 1)[0]


@dataclass(frozen=True)
class _CallSite:
    """a call that may reach a whole-read function of the repo, resolved after every module is walked.

    :ivar path: the caller's module
    :ivar line: the call's line
    :ivar function: the caller's qualified name
    :ivar name: the callee's final name
    :ivar receiver: ``self`` (a method of the caller's class), ``bare`` (a module function), ``store`` (a method
        on a store, L3, L1 or copy handle), ``other`` (a method on anything else)
    :ivar owner: the caller's class qualified name, for a ``self`` call
    :ivar spelled: the call as written, for the reader
    :ivar narrowed: whether the call passes a narrowing argument (a filter, keys, a scope)
    :ivar whole_named: whether the callee's name says it reads, rebuilds or warms whole
    :ivar receiver_name: the receiver's last segment (``_store``), empty for a bare or ``self`` call
    """

    path: str
    line: int
    function: str
    name: str
    receiver: str
    owner: str
    spelled: str
    narrowed: bool
    whole_named: bool
    receiver_name: str


@dataclass
class WholeReadScan:
    """everything one walk found.

    :ivar hits: every hit, in walk order, ``calls-whole-read`` last
    :ivar files_scanned: modules parsed
    :ivar definitions: ``<path>::<qualified name>`` of every function and class seen
    :ivar unmatched_exclusions: excluded prefixes that matched no module
    """

    hits: list[WholeReadHit] = field(default_factory=list)
    files_scanned: int = 0
    definitions: set[str] = field(default_factory=set)
    unmatched_exclusions: list[str] = field(default_factory=list)


#: callees that only quote the name they are given
_QUOTERS: Final[frozenset[str]] = frozenset({"quote_identifier", "_quote_ident", "quote_ident", "_quoted", "quoted"})


def _placeholder(value: ast.expr, constants: dict[str, str]) -> str:
    """what an f-string placeholder reads as: a module constant's text (``{_MARK_TABLE}``, or quoted), else ``{}``.

    :param value: the placeholder's expression
    :ptype value: ast.expr
    :param constants: the module's names bound to strings
    :ptype constants: dict[str, str]
    :return: the text
    :rtype: str
    """
    name = value
    if isinstance(value, ast.Call) and len(value.args) == 1 and _final(value) in _QUOTERS:
        name = value.args[0]
    return constants.get(name.id, "{}") if isinstance(name, ast.Name) else "{}"


def _string_text(node: ast.expr, env: dict[str, str]) -> str | None:
    """the statically known text of a string expression.

    :param node: the expression
    :ptype node: ast.expr
    :param env: names bound to strings in scope
    :ptype env: dict[str, str]
    :return: the text, each unknown interpolation as ``{}``; None when it is not a string expression
    :rtype: str | None
    """
    text: str | None = None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
    elif isinstance(node, ast.JoinedStr):
        text = "".join(
            str(part.value)
            if isinstance(part, ast.Constant)
            else _placeholder(part.value, env)
            if isinstance(part, ast.FormattedValue)
            else "{}"
            for part in node.values
        )
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_text(node.left, env)
        right = _string_text(node.right, env)
        if left is not None or right is not None:
            text = (left if left is not None else "{}") + (right if right is not None else "{}")
    elif isinstance(node, ast.Name):
        text = env.get(node.id)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"format", "strip"}:
        text = _string_text(node.func.value, env)
    elif isinstance(node, ast.Call) and node.args and _final(node) in {"dedent", "SQL", "text"}:
        text = _string_text(node.args[0], env)
    return text


def _final(call: ast.Call) -> str | None:
    """the final name of a call's callee.

    :param call: the call
    :ptype call: ast.Call
    :return: ``f`` for ``f(...)`` and ``x.f(...)``, else None
    :rtype: str | None
    """
    func = call.func
    name: str | None = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    return name


def _segments(call: ast.Call) -> list[str]:
    """the dotted segments of a call's receiver.

    :param call: the call
    :ptype call: ast.Call
    :return: the segments, ``self`` and ``cls`` included; empty for a bare call
    :rtype: list[str]
    """
    receiver = dotted(call.func.value) if isinstance(call.func, ast.Attribute) else None
    return [part for part in (receiver or "").split(".") if part]


def _maybe_none(value: ast.expr) -> bool:
    """whether a keyword's value is or may be None: ``None``, or a conditional with a None branch.

    :param value: the value
    :ptype value: ast.expr
    :return: whether it may leave the call unnarrowed
    :rtype: bool
    """
    none = isinstance(value, ast.Constant) and value.value is None
    if isinstance(value, ast.IfExp):
        none = _maybe_none(value.body) or _maybe_none(value.orelse)
    return none


def _one_line(text: str, limit: int = 200) -> str:
    """text on one line, cut to ``limit`` characters.

    :param text: the text
    :ptype text: str
    :param limit: the most characters kept
    :ptype limit: int
    :return: the line
    :rtype: str
    """
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


class _ModuleWalk(ast.NodeVisitor):
    """walks one module, recording hits and the calls that may reach a whole-read function."""

    def __init__(
        self,
        path: str,
        tree: ast.Module,
        source: str,
        *,
        pagers: frozenset[str],
        always_whole: frozenset[str],
        narrowing: frozenset[str],
        transient: re.Pattern[str] | None,
        verify: bool,
    ) -> None:
        """prepare a walk of one module.

        :param path: repo-relative posix path of the module
        :ptype path: str
        :param tree: its parse
        :ptype tree: ast.Module
        :param source: its text, for the statement shown with a call
        :ptype source: str
        :param pagers: pager names (:data:`DEFAULT_PAGER_CALLS` and the repo's own)
        :ptype pagers: frozenset[str]
        :param always_whole: helpers that read whole whatever they are passed
        :ptype always_whole: frozenset[str]
        :param narrowing: keywords that narrow a pager
        :ptype narrowing: frozenset[str]
        :param transient: relations that are a change set, never a data table
        :ptype transient: re.Pattern[str] | None
        :param verify: whether the module is load code, where a count or fingerprint even of one key's rows is a
            verification scan
        :ptype verify: bool
        :return: nothing
        :rtype: None
        """
        self.path = path
        self.source = source
        self.hits: list[WholeReadHit] = []
        self.calls: list[_CallSite] = []
        self.definitions: set[str] = set()
        self.module_functions: set[str] = set()
        #: a name imported into the module, onto ``<module>.<name>`` it came from
        self.imports: dict[str, str] = {}
        self._pagers = pagers
        self._always_whole = always_whole
        self._narrowing = narrowing
        self._transient = transient
        self._verify = verify
        self._loops: list[set[str]] = []
        self.stubs: set[str] = set()
        self._stack: list[str] = []
        self._classes: list[str] = []
        self._envs: list[dict[str, str]] = [{}]
        self._parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self._parents[child] = node
        #: module-level names (and names imported into the module): a loop over one walks a fixed set of keys
        self._constants: set[str] = set()
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.module_functions.add(statement.name)
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                self._constants.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                self._constants.update((alias.asname or alias.name).split(".")[0] for alias in statement.names)
            if isinstance(statement, ast.ImportFrom) and statement.module and not statement.level:
                self.imports.update(
                    {alias.asname or alias.name: f"{statement.module}.{alias.name}" for alias in statement.names}
                )
        #: relations the module registers from in-memory rows (DuckDB ``register("_bulk_rows", arrow)``): the rows
        #: being written, never a data table
        self._registered = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _final(node) == "register"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        self._bind(tree.body, self._envs[0])

    @property
    def _function(self) -> str:
        """the qualified name of the scope the walk is in.

        :return: dotted qualified name, or ``<module>``
        :rtype: str
        """
        return ".".join(self._stack) if self._stack else "<module>"

    def _bind(self, body: list[ast.stmt], env: dict[str, str]) -> None:
        """record the string each simple assignment under ``body`` binds (not in nested functions).

        :param body: statements
        :ptype body: list[ast.stmt]
        :param env: name onto bound string, updated in place
        :ptype env: dict[str, str]
        :return: nothing
        :rtype: None
        """
        for node in _own_nodes(body):
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            if isinstance(target, ast.Name) and value is not None:
                text = _string_text(value, env)
                if text is not None:
                    env[target.id] = text

    def _hit(self, node: ast.AST, symbol: str, statement: str) -> None:
        """record one hit at ``node``.

        :param node: the site
        :ptype node: ast.AST
        :param symbol: ``<kind> <what>``
        :ptype symbol: str
        :param statement: the statement or call
        :ptype statement: str
        :return: nothing
        :rtype: None
        """
        self.hits.append(
            WholeReadHit(self.path, getattr(node, "lineno", 0), self._function, symbol, _one_line(statement))
        )

    @staticmethod
    def _spelled(node: ast.AST) -> str:
        """a node as source text.

        :param node: the node
        :ptype node: ast.AST
        :return: its text
        :rtype: str
        """
        return ast.unparse(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """walk a class body.

        :param node: the class
        :ptype node: ast.ClassDef
        :return: nothing
        :rtype: None
        """
        self._stack.append(node.name)
        self._classes.append(self._function)
        self.definitions.add(f"{self.path}::{self._function}")
        self.generic_visit(node)
        self._classes.pop()
        self._stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """walk a function with its own string bindings.

        :param node: the function
        :ptype node: ast.FunctionDef | ast.AsyncFunctionDef
        :return: nothing
        :rtype: None
        """
        self._stack.append(node.name)
        self.definitions.add(f"{self.path}::{self._function}")
        if _is_stub(node):
            self.stubs.add(f"{self.path}::{self._function}")
        env = dict(self._envs[-1])
        self._bind(node.body, env)
        self._envs.append(env)
        owner = self._classes[-1] if self._classes and self._function.startswith(f"{self._classes[-1]}.") else ""
        self._classes.append(owner)
        self.generic_visit(node)
        self._classes.pop()
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

    def _maximal_string(self, node: ast.expr) -> bool:
        """whether ``node`` is a whole string expression: not a piece of a larger one, a docstring or prose.

        :param node: a string constant, f-string or concatenation
        :ptype node: ast.expr
        :return: whether to judge it
        :rtype: bool
        """
        parent = self._parents.get(node)
        piece = isinstance(parent, ast.JoinedStr) or (
            isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add) and _string_text(parent, {}) is not None
        )
        prose = isinstance(parent, ast.Expr) or (
            isinstance(parent, ast.Call) and _PROSE_CALLS.search(_final(parent) or "") is not None
        )
        return not piece and not prose

    def _judge_string(self, node: ast.expr) -> None:
        """judge a string expression that is SQL.

        :param node: the expression
        :ptype node: ast.expr
        :return: nothing
        :rtype: None
        """
        text = _string_text(node, self._envs[0]) if self._maximal_string(node) else None
        for finding in judge_sql(text) if text is not None and is_sql(text) else []:
            if self._transient_table(finding.table):
                continue
            parent = self._parents.get(node)
            if finding.kind == KEYED_COUNT:
                if self._verify:
                    self._hit(node, f"verify-aggregate {finding.table}", text or "")
            elif finding.kind != HEAD:
                self._hit(node, f"{finding.kind} {finding.table}", text or "")
            elif isinstance(parent, ast.Call) and self._runs(parent, node):
                self._hit(node, f"whole-select {finding.table}", text or "")

    def _transient_table(self, table: str) -> bool:
        """whether a relation is a change set (a registered relation, or one the repo names transient).

        :param table: the relation as written
        :ptype table: str
        :return: whether it is no data table
        :rtype: bool
        """
        bare = table.strip('"')
        return bare in self._registered or (self._transient is not None and self._transient.search(bare) is not None)

    def _runs(self, call: ast.Call, argument: ast.expr) -> bool:
        """whether ``call`` runs ``argument`` as SQL, as written.

        :param call: the call
        :ptype call: ast.Call
        :param argument: one of its arguments
        :ptype argument: ast.expr
        :return: whether the call executes it
        :rtype: bool
        """
        sql_keyword = any(k.value is argument and k.arg in {"query", "sql", "statement"} for k in call.keywords)
        first = bool(call.args) and call.args[0] is argument
        second = len(call.args) > 1 and call.args[1] is argument and _final(call) in {"l3_fetch_arrow", "query"}
        return _final(call) in _EXEC_CALLS and (first or second or sql_keyword)

    def visit_Constant(self, node: ast.Constant) -> None:
        """judge a string literal.

        :param node: the literal
        :ptype node: ast.Constant
        :return: nothing
        :rtype: None
        """
        if isinstance(node.value, str):
            self._judge_string(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """judge an f-string, and walk its placeholders.

        :param node: the f-string
        :ptype node: ast.JoinedStr
        :return: nothing
        :rtype: None
        """
        self._judge_string(node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """judge a concatenation of strings, then walk its parts (each a piece, not judged alone).

        :param node: the operation
        :ptype node: ast.BinOp
        :return: nothing
        :rtype: None
        """
        if isinstance(node.op, ast.Add) and _string_text(node, {}) is not None:
            self._judge_string(node)
        self.generic_visit(node)

    def _visit_loop(self, node: ast.For | ast.AsyncFor | ast.comprehension, body: list[ast.AST]) -> None:
        """walk a loop, noting the names it binds when it walks every key of a module constant (``for source in
        SOURCES``): a read narrowed only by them reads every key, so the whole table.

        :param node: the loop or comprehension
        :ptype node: ast.For | ast.AsyncFor | ast.comprehension
        :param body: what runs per key
        :ptype body: list[ast.AST]
        :return: nothing
        :rtype: None
        """
        constant = any(
            isinstance(name, ast.Name) and name.id.lstrip("_").isupper() and name.id in self._constants
            for name in ast.walk(node.iter)
        )
        bound = {name.id for name in ast.walk(node.target) if isinstance(name, ast.Name)} if constant else set()
        self._loops.append(bound | (self._loops[-1] if self._loops else set()))
        for child in body:
            self.visit(child)
        self._loops.pop()

    def visit_For(self, node: ast.For) -> None:
        """walk a ``for`` loop.

        :param node: the loop
        :ptype node: ast.For
        :return: nothing
        :rtype: None
        """
        self.visit(node.iter)
        self._visit_loop(node, [*node.body, *node.orelse])

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """walk an ``async for`` loop.

        :param node: the loop
        :ptype node: ast.AsyncFor
        :return: nothing
        :rtype: None
        """
        self.visit(node.iter)
        self._visit_loop(node, [*node.body, *node.orelse])

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """a table put in the set of tables to read whole: ``self._whole |= ...``.

        :param node: the statement
        :ptype node: ast.AugAssign
        :return: nothing
        :rtype: None
        """
        target = dotted(node.target) or ""
        if isinstance(node.op, (ast.BitOr, ast.Add)) and _WHOLE_SET.search(target.rsplit(".", 1)[-1]):
            self._hit(node, f"marks-whole {target.rsplit('.', 1)[-1]}", self._spelled(node))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """judge one call.

        :param node: the call
        :ptype node: ast.Call
        :return: nothing
        :rtype: None
        """
        name = _final(node)
        if name is not None:
            self._judge_call(node, name)
        self.generic_visit(node)

    def _judge_call(self, node: ast.Call, name: str) -> None:
        """every call-shaped check over one call.

        :param node: the call
        :ptype node: ast.Call
        :param name: its final callee name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        segments = _segments(node)
        narrowing = [k for k in node.keywords if k.arg in self._narrowing]
        narrowed = any(not _maybe_none(k.value) for k in narrowing) or any(
            _PAGER_FILTER.search((dotted(value) or "").rsplit(".", 1)[-1]) is not None for value in node.args[1:]
        )
        spelled = self._spelled(node)
        pager = name in self._pagers and bool(node.args or node.keywords)
        pager = pager and (
            isinstance(node.func, ast.Name)
            or segments[:1] in (["self"], ["cls"])
            or _PAGER_RECEIVER.search(segments[-1] if segments else "") is not None
        )
        looped = (
            pager
            and narrowed
            and bool(self._loops and self._loops[-1])
            and all(
                {n.id for n in ast.walk(k.value) if isinstance(n, ast.Name)} <= self._loops[-1] | self._constants
                and any(isinstance(n, ast.Name) and n.id in self._loops[-1] for n in ast.walk(k.value))
                for k in narrowing
                if not _maybe_none(k.value)
            )
        )
        if looped:
            self._hit(node, f"whole-by-loop {name}", spelled)
        elif name in self._always_whole and isinstance(node.func, ast.Name):
            self._hit(node, f"whole-read {name}", spelled)
        elif pager and narrowing and not narrowed:
            self._hit(node, f"may-read-whole {name}", spelled)
        elif pager and not narrowed and not any(k.arg is None for k in node.keywords):
            self._hit(node, f"unkeyed-pager {name}", spelled)
        for keyword in node.keywords:
            if keyword.arg in _WHOLE_FLAGS and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                self._hit(node, f"whole-flag {keyword.arg}=", spelled)
        if name in {"add", "update", "extend"} and segments and _WHOLE_SET.search(segments[-1]):
            self._hit(node, f"marks-whole {segments[-1]}", spelled)
        if name == "truncate" and segments and _STORE.search(segments[-1]):
            self._hit(node, "truncate-call truncate", spelled)
        if name in _EXEC_CALLS:
            self._judge_named_sql(node)
        narrowed = narrowed or any(
            isinstance(value, ast.Dict)
            or (isinstance(value, ast.Call) and _final(value) == "dict")
            or _NARROWING_NAME.search((dotted(value) or "").rsplit(".", 1)[-1]) is not None
            for value in [*node.args[1:], *(k.value for k in node.keywords)]
        )
        whole_named = (
            name not in self._pagers
            and name not in self._always_whole
            and _WHOLE_CALL.search(name.lower()) is not None
            and _WHOLE_QUESTION.search(name) is None
        )
        receiver = "other"
        if segments[:1] in (["self"], ["cls"]) and len(segments) == 1:
            receiver = "self"
        elif not segments and isinstance(node.func, ast.Name):
            receiver = "bare"
        elif segments and _STORE.search(segments[-1]):
            receiver = "store"
        if receiver != "other" or whole_named:
            self._call(node, name, receiver, spelled, narrowed=narrowed, whole_named=whole_named)

    def _judge_named_sql(self, node: ast.Call) -> None:
        """a head run through a name bound to it: ``sql = f"SELECT ... FROM {t}"; conn.fetch(sql)``.

        :param node: an executing call
        :ptype node: ast.Call
        :return: nothing
        :rtype: None
        """
        argument = node.args[0] if node.args else None
        if isinstance(argument, ast.Name) and argument.id in self._envs[-1]:
            text = self._envs[-1][argument.id]
            if is_sql(text):
                for finding in judge_sql(text):
                    if finding.kind == HEAD and not self._transient_table(finding.table):
                        self._hit(node, f"whole-select {finding.table}", text)

    def _call(
        self, node: ast.Call, name: str, receiver: str, spelled: str, *, narrowed: bool, whole_named: bool
    ) -> None:
        """keep a call for the call graph, resolved once every module is walked.

        :param node: the call
        :ptype node: ast.Call
        :param name: its final name
        :ptype name: str
        :param receiver: ``self``, ``bare``, ``store`` or ``other``
        :ptype receiver: str
        :param spelled: the call as written
        :ptype spelled: str
        :param narrowed: whether it passes a narrowing argument
        :ptype narrowed: bool
        :param whole_named: whether its name says whole
        :ptype whole_named: bool
        :return: nothing
        :rtype: None
        """
        segments = _segments(node)
        receiver_name = segments[-1] if segments and segments[-1] not in {"self", "cls"} else ""
        self.calls.append(
            _CallSite(
                self.path,
                getattr(node, "lineno", 0),
                self._function,
                name,
                receiver,
                self._classes[-1] if self._classes else "",
                _one_line(spelled),
                narrowed,
                whole_named,
                receiver_name,
            )
        )


def _is_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """whether a function is a stub (a Protocol's or an abstract method): a docstring and ``...``, ``pass`` or a
    ``raise``, nothing else. A stub is not a target a call resolves to.

    :param node: the function
    :ptype node: ast.FunctionDef | ast.AsyncFunctionDef
    :return: whether it is one
    :rtype: bool
    """
    body = node.body[1:] if node.body and isinstance(node.body[0], ast.Expr) else node.body
    return all(
        isinstance(statement, (ast.Pass, ast.Raise))
        or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
        for statement in body
    )


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


def _targets(
    call: _CallSite,
    module_functions: dict[str, set[str]],
    by_name: dict[str, list[tuple[str, str]]],
    imports: dict[str, dict[str, str]],
) -> list[tuple[str, str]]:
    """the repo's functions a call may reach.

    A ``self`` call reaches its own class's method; a bare call its module's function, else the function of the
    repo module it was imported from; a call on any other receiver the repo's methods of that name (those of the
    classes named for the receiver, when there are some).

    :param call: the call
    :ptype call: _CallSite
    :param module_functions: module path onto the names of its module-level functions
    :ptype module_functions: dict[str, set[str]]
    :param by_name: final name onto ``(path, qualified name)`` of every function of the repo
    :ptype by_name: dict[str, list[tuple[str, str]]]
    :param imports: module path onto its imported names, each onto ``<module>.<name>``
    :ptype imports: dict[str, dict[str, str]]
    :return: the targets, the caller itself left out
    :rtype: list[tuple[str, str]]
    """
    targets: list[tuple[str, str]] = []
    if call.receiver == "self" and call.owner:
        targets = [(call.path, f"{call.owner}.{call.name}")]
    elif call.receiver == "bare" and call.name in module_functions.get(call.path, set()):
        targets = [(call.path, call.name)]
    elif call.receiver == "bare":
        # an imported name reaches the function of the module it came from, when the repo holds that module; a
        # name the module neither defines nor imports from the repo reaches nothing the walk can name
        origin = imports.get(call.path, {}).get(call.name)
        module = origin.rsplit(".", 1)[0].replace(".", "/") + ".py" if origin else None
        targets = [
            (path, name)
            for path, name in by_name.get(call.name, [])
            if name == call.name and module is not None and path.endswith(module)
        ]
    else:
        targets = [(path, name) for path, name in by_name.get(call.name, []) if name.count(".") == 1]
        # a receiver named for its kind (``self._store``, ``self._warehouse``) reaches the classes named for it
        token = call.receiver_name.strip("_").lower().removesuffix("s")
        named = [(path, name) for path, name in targets if token and token in name.split(".", 1)[0].lower()]
        targets = named or targets
    return [target for target in targets if target != (call.path, call.function)]


def _resolve(
    calls: list[_CallSite],
    hits: list[WholeReadHit],
    module_functions: dict[str, set[str]],
    definitions: set[str],
    stubs: set[str],
    imports: dict[str, dict[str, str]],
) -> list[WholeReadHit]:
    """the call graph's hits: ``calls-whole-read`` and ``whole-call``.

    - ``calls-whole-read``: an unnarrowed call into a function that itself reads whole (one level, a read kind);
    - ``whole-call``: a call whose name says whole, rebuild or warm, when its target reaches any hit through the
      call graph (any depth), or when the walk cannot find its target in the repo.

    A call that may reach more than one function counts only when every one of them does: the walk never guesses.

    :param calls: every kept call
    :ptype calls: list[_CallSite]
    :param hits: every hit the module walks found
    :ptype hits: list[WholeReadHit]
    :param module_functions: module path onto the names of its module-level functions
    :ptype module_functions: dict[str, set[str]]
    :param definitions: ``<path>::<qualified name>`` of every function and class the walk found
    :ptype definitions: set[str]
    :param stubs: the definitions that are stubs, never a call's target
    :ptype stubs: set[str]
    :param imports: module path onto its imported names
    :ptype imports: dict[str, dict[str, str]]
    :return: the hits
    :rtype: list[WholeReadHit]
    """
    by_name: dict[str, list[tuple[str, str]]] = {}
    for entry in definitions - stubs:
        path, _, qualified = entry.partition("::")
        by_name.setdefault(qualified.rsplit(".", 1)[-1], []).append((path, qualified))
    readers: dict[tuple[str, str], WholeReadHit] = {}
    for hit in hits:
        if hit.kind in _READ_KINDS:
            readers.setdefault((hit.path, hit.function), hit)
    resolved = [(call, _targets(call, module_functions, by_name, imports)) for call in calls]
    reaches = {(hit.path, hit.function) for hit in hits}
    grew = True
    while grew:
        grew = False
        for call, targets in resolved:
            caller = (call.path, call.function)
            if caller not in reaches and targets and all(target in reaches for target in targets):
                reaches.add(caller)
                grew = True
    found: list[WholeReadHit] = []
    for call, targets in resolved:
        if not call.narrowed and targets and all(target in readers for target in targets):
            path, qualified = sorted(targets)[0]
            reached = readers[(path, qualified)]
            found.append(
                WholeReadHit(
                    call.path,
                    call.line,
                    call.function,
                    f"calls-whole-read {qualified}",
                    f"{call.spelled} -> {path}:{reached.line} {reached.symbol}",
                )
            )
        elif call.whole_named and (not targets or all(target in reaches for target in targets)):
            found.append(WholeReadHit(call.path, call.line, call.function, f"whole-call {call.name}", call.spelled))
    return found


def scan_whole_reads(
    repo_root: Path,
    src_roots: tuple[Path, ...],
    *,
    excluded_prefixes: tuple[str, ...] = (),
    pagers: frozenset[str] = DEFAULT_PAGER_CALLS,
    always_whole: frozenset[str] = _ALWAYS_WHOLE_CALLS,
    narrowing: frozenset[str] = DEFAULT_NARROWING_KEYWORDS,
    transient: str = "",
    schema_change_dirs: tuple[str, ...] = ("migrations",),
    verify_prefixes: tuple[str, ...] = (),
) -> WholeReadScan:
    """walk every module under ``src_roots`` and record every whole read, write and call.

    :param repo_root: the repo root; paths are relative to it
    :ptype repo_root: Path
    :param src_roots: source trees to walk
    :ptype src_roots: tuple[Path, ...]
    :param excluded_prefixes: repo-relative directory prefixes left unwalked
    :ptype excluded_prefixes: tuple[str, ...]
    :param pagers: pager and fingerprint helper names
    :ptype pagers: frozenset[str]
    :param always_whole: helpers that read whole whatever they are passed
    :ptype always_whole: frozenset[str]
    :param narrowing: keywords that narrow a pager
    :ptype narrowing: frozenset[str]
    :param transient: a regular expression of relations that are a change set (a stage), never a data table;
        empty for none
    :ptype transient: str
    :param schema_change_dirs: directory names whose modules are schema changes (a migration runs once per schema
        version, under the migration runner), left unwalked
    :ptype schema_change_dirs: tuple[str, ...]
    :param verify_prefixes: repo-relative prefixes of load code, where a count or fingerprint even of one key's rows
        is a verification scan (``verify-aggregate``)
    :ptype verify_prefixes: tuple[str, ...]
    :return: the walk
    :rtype: WholeReadScan
    """
    scan = WholeReadScan()
    used: set[str] = set()
    calls: list[_CallSite] = []
    module_functions: dict[str, set[str]] = {}
    stubs: set[str] = set()
    imports: dict[str, dict[str, str]] = {}
    pattern = re.compile(transient) if transient else None
    for root in src_roots:
        for path in iter_python_files(root):
            relative = relative_posix_path(path, repo_root)
            excluded = [prefix for prefix in excluded_prefixes if relative.startswith(prefix)]
            used.update(excluded)
            if excluded or any(part in schema_change_dirs for part in relative.split("/")[:-1]):
                continue
            tree = parse_python_path(path)
            walk = _ModuleWalk(
                relative,
                tree,
                path.read_text(encoding="utf-8"),
                pagers=pagers,
                always_whole=always_whole,
                narrowing=narrowing,
                transient=pattern,
                verify=any(relative.startswith(prefix) for prefix in verify_prefixes),
            )
            walk.visit(tree)
            stubs.update(walk.stubs)
            scan.hits.extend(walk.hits)
            scan.definitions.update(walk.definitions)
            calls.extend(walk.calls)
            module_functions[relative] = walk.module_functions
            imports[relative] = walk.imports
            scan.files_scanned += 1
    scan.hits.extend(_resolve(calls, scan.hits, module_functions, scan.definitions, stubs, imports))
    scan.unmatched_exclusions = [prefix for prefix in excluded_prefixes if prefix not in used]
    return scan
