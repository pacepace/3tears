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
imported from), a class of the repo it constructs (its ``__init__``; a class with none, a dataclass, reaches
nothing), or the repo's methods of that name (narrowed to the classes named for the receiver:
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
from threetears.enforcement.no_whole_reads.schema import COPY_PREFIX, SchemaCatalog, bare_table, purposes
from threetears.enforcement.no_whole_reads.sql import (
    HEAD,
    KEYED_COUNT,
    PK_TOKEN,
    Reader,
    condition_shape,
    is_sql,
    judge_sql,
)

__all__ = [
    "DEFAULT_NARROWING_KEYWORDS",
    "DEFAULT_PAGER_CALLS",
    "OperatorMark",
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
        "non-key-where",
        "unknown-schema",
        "unknown-predicate",
        "unkeyed-sort",
        "unkeyed-pager",
        "may-read-whole",
        "whole-read",
        "whole-by-loop",
        "verify-aggregate",
    }
)

#: the kinds an operator's reset (``@operator_reset``) may hold: whole writes it orders, never a read
_OPERATOR_RESET_KINDS: Final[frozenset[str]] = frozenset({"truncate", "truncate-call"})

#: the kinds an operator's explicit one-off command (``@operator_command``) may hold: the reads its completeness needs
_OPERATOR_COMMAND_KINDS: Final[frozenset[str]] = _READ_KINDS | frozenset({"calls-whole-read"})

#: the marks, by decorator name, onto the kinds each lets through inside the marked function. A security backstop's
#: whole pass is accepted (and printed); a source change read lets nothing through by kind: the judge passes only its
#: reads past the source's own stamps
_OPERATOR_MARKS: Final[dict[str, frozenset[str]]] = {
    "operator_reset": _OPERATOR_RESET_KINDS,
    "operator_command": _OPERATOR_COMMAND_KINDS,
    "security_backstop": _OPERATOR_COMMAND_KINDS,
    "source_change_read": frozenset(),
}

#: a function no source change read marks
_NO_SOURCE: Final[tuple[frozenset[str], frozenset[str], bool]] = (frozenset(), frozenset(), False)

#: the marks reachable only from their own entry: every caller in the repo carries the same mark
_ENTRY_ONLY_MARKS: Final[frozenset[str]] = frozenset({"operator_reset", "operator_command"})

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
        # 3tears' own query backends (DuckDB, SQLite, the query clients)
        "execute_query",
        "execute_arrow",
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
class OperatorMark:
    """a function marked as an operator's own command (``threetears.core.operator``).

    :ivar path: repo-relative posix path of the module
    :ivar function: qualified name of the marked function
    :ivar line: line of the function
    :ivar mark: ``operator_reset`` or ``operator_command``
    :ivar reason: the mark's reason, empty when it is not a literal
    :ivar proof: the mark's ``proof=``, ``<test file>::<test function>``, empty when it is not a literal
    :ivar speed: a security backstop's ``speed=``, its calibrated speed test
    :ivar stamps: a source change read's ``stamps=``
    :ivar ids: a source change read's ``ids=``
    :ivar unit: a source change read's ``unit=``
    """

    path: str
    function: str
    line: int
    mark: str
    reason: str
    proof: str
    speed: str = ""
    stamps: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()
    unit: str = ""

    @property
    def kinds(self) -> frozenset[str]:
        """the hit kinds the mark lets through inside the function.

        :return: the kinds
        :rtype: frozenset[str]
        """
        return _OPERATOR_MARKS[self.mark]


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
    :ivar operators: every function marked as an operator's own command
    :ivar operator_callers: each call in the repo that reaches a marked function, one line each: a marked function
        is reachable only from its own entry, so any caller is a finding
    :ivar constants: ``<path>::<qualified name>`` of every module or class constant bound to a string, which an
        index's ``purpose`` may name when the SQL is written there
    :ivar schema: the declared keys and indexes every statement was judged against
    """

    hits: list[WholeReadHit] = field(default_factory=list)
    files_scanned: int = 0
    definitions: set[str] = field(default_factory=set)
    unmatched_exclusions: list[str] = field(default_factory=list)
    operators: list[OperatorMark] = field(default_factory=list)
    operator_callers: list[str] = field(default_factory=list)
    constants: set[str] = field(default_factory=set)
    schema: SchemaCatalog = field(default_factory=SchemaCatalog)


#: callees that only quote the name they are given: checked before a template function, since a quoter returning
#: ``'"' + name + '"'`` is one too, and its template (``"{}"``) would hide the name it quotes
_QUOTERS: Final[frozenset[str]] = frozenset({"quote_identifier", "_quote_ident", "quote_ident", "_quoted", "quoted"})


#: a name in a placeholder that says it is the key column(s): the statement reads by the table's key, whatever table
_KEY_NAME: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:^|_)(?:pk\w*|primary_keys?|keys?|key_?col\w*|key_?columns?|key_?names?|id_?col\w*)$"
)


#: a local copy's relation accessors: ``snapshot.relation(NAME)`` names the copy's own table, never the L3 one
_COPY_CALLS: Final[frozenset[str]] = frozenset({"relation", "relation_reading"})


#: the key under which a function returning a string template is kept beside the names bound to strings
_TEMPLATE: Final[str] = "{}()"


def _constant_text(value: ast.expr, constants: dict[str, str], depth: int = 0) -> str | None:
    """the string a placeholder argument statically is: a literal, a name bound to one (a module constant, an
    imported one, or a string the function binds once), a call of a function returning a template
    (``note_table(index)`` -> ``threetears_change_notes_{}``), a one-argument call on any of these
    (``quote_identifier(NAME)``), or a sum of them.

    :param value: the expression
    :ptype value: ast.expr
    :param constants: names onto their strings, and template functions under ``<name>()``
    :ptype constants: dict[str, str]
    :param depth: how deep the walk is
    :ptype depth: int
    :return: the string, or None
    :rtype: str | None
    """
    text: str | None = None
    if depth > 5:
        text = None
    elif isinstance(value, ast.Constant) and isinstance(value.value, str):
        text = value.value
    elif isinstance(value, ast.Name):
        text = constants.get(value.id)
    elif isinstance(value, ast.Call) and len(value.args) == 1 and _final(value) in _QUOTERS:
        text = _constant_text(value.args[0], constants, depth + 1)
    elif isinstance(value, ast.Call) and _TEMPLATE.format(_final(value)) in constants:
        text = constants[_TEMPLATE.format(_final(value))]
        # a helper returning a statement is judged once where it is written, with its own module's constants; in its
        # caller (``FROM ({helper()}) AS d``, ``x AS ({helper()})``) it is an already-judged subquery or CTE
        text = None if is_sql(text) else text
    elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        left = _constant_text(value.left, constants, depth + 1)
        right = _constant_text(value.right, constants, depth + 1)
        text = None if left is None or right is None else left + right
    return text


def _placeholder(value: ast.expr, constants: dict[str, str]) -> str:
    """what an f-string placeholder reads as: a constant's text (``{_MARK_TABLE}``, an imported one, or one handed
    to a call: ``{quote(TABLE)}``), ``copy.<name>`` for a local copy's relation (``{snapshot.relation(GEOS)}``),
    ``{pk}`` when a name in it says it is the key (``{pk}``, ``{self._key_column}``, ``{", ".join(pk_cols)}``), else
    ``{}``.

    :param value: the placeholder's expression
    :ptype value: ast.expr
    :param constants: the module's names bound to strings
    :ptype constants: dict[str, str]
    :return: the text
    :rtype: str
    """
    known = _constant_text(value, constants)
    if known is not None:
        return known
    if isinstance(value, ast.Call) and value.args:
        # ``{quote(TABLE)}``, ``{snapshot.relation(GEOS)}``: a call on a constant reads as the constant; a local copy's
        # relation (``.relation(NAME)``, ``.relation_reading(NAME, ...)``) as ``copy.<name>``, its own key namespace
        argument = _constant_text(value.args[0], constants)
        if argument is not None and (_final(value) in _COPY_CALLS or not argument.count("{")):
            return f"{COPY_PREFIX}{argument}" if _final(value) in _COPY_CALLS else argument
        if _final(value) in _COPY_CALLS:
            # a copy relation named at run time: judged against the default key every copy declares
            return f"{COPY_PREFIX}{{}}"
    names = [
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(value)
        if isinstance(node, (ast.Name, ast.Attribute))
    ]
    return PK_TOKEN if any(_KEY_NAME.search(found) for found in names) else "{}"


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
        schema: SchemaCatalog | None = None,
        imported: dict[str, str] | None = None,
        exec_calls: frozenset[str] = frozenset(),
        cte_arguments: dict[tuple[str, str], list[tuple[str, bool]]] | None = None,
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
        :param schema: every table's declared key and indexes, the statements are judged against
        :ptype schema: SchemaCatalog | None
        :param imported: the strings the module imports from the repo's other modules, by the name it imports them as
        :ptype imported: dict[str, str] | None
        :param exec_calls: the repo's own calls that run their first argument as SQL, beside :data:`_EXEC_CALLS`
        :ptype exec_calls: frozenset[str]
        :param cte_arguments: ``(function, parameter)`` onto each call site's argument: ``(site, whether it is a string
            literal its calling function defines as a CTE)`` (:func:`_cte_arguments`)
        :ptype cte_arguments: dict[tuple[str, str], list[tuple[str, bool]]] | None
        :return: nothing
        :rtype: None
        """
        self.path = path
        self.source = source
        self.hits: list[WholeReadHit] = []
        self.calls: list[_CallSite] = []
        self.definitions: set[str] = set()
        self.module_functions: set[str] = set()
        #: the module's top-level classes: a call to one constructs it, reaching only its ``__init__``
        self.module_classes: set[str] = {
            statement.name for statement in tree.body if isinstance(statement, ast.ClassDef)
        }
        self.operators: list[OperatorMark] = []
        #: a name imported into the module, onto ``<module>.<name>`` it came from
        self.imports: dict[str, str] = {}
        self._pagers = pagers
        self._always_whole = always_whole
        self._narrowing = narrowing
        self._transient = transient
        self._verify = verify
        self._exec_calls = _EXEC_CALLS | exec_calls
        self._cte_arguments = cte_arguments or {}
        self._schema = schema if schema is not None else SchemaCatalog()
        #: module and class constants bound to strings, ``<path>::<qualified name>``
        self.constants: set[str] = set()
        self._loops: list[set[str]] = []
        #: per enclosing function, the CTE names its statements define
        self._ctes: list[frozenset[str]] = []
        self._function_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        #: per enclosing function, the terms a ``source_change_read`` mark names: stamps, ids, a whole static unit
        self._source: list[tuple[frozenset[str], frozenset[str], bool]] = []
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
        # a temp view over only the rows being written (``CREATE TEMP VIEW typed AS SELECT ... FROM raw``) is those rows
        self._registered |= _transient_views(tree, self._registered, transient)
        #: the module's sort vocabularies (``_SORT_COLUMNS: Final = {Field.NAME: SortColumn(column=..., index=...)}``)
        self._sort_mappings: dict[str, list[ast.Call]] = _sort_mappings(tree)
        #: each module function onto the vocabularies it, or a function it calls, uses
        self._mapping_users: dict[str, set[str]] = _mapping_users(tree, set(self._sort_mappings))
        self._envs[0].update(imported or {})
        self._envs[0].update(_module_strings(tree))
        self._bind(tree.body, self._envs[0])
        self._constants_of(tree.body, "")

    def _constants_of(self, body: list[ast.stmt], scope: str) -> None:
        """record the module's and its classes' names bound to a string (an index's purpose may name one).

        :param body: a module or class body
        :ptype body: list[ast.stmt]
        :param scope: the class's qualified name and a dot, empty for the module
        :ptype scope: str
        :return: nothing
        :rtype: None
        """
        for statement in body:
            if isinstance(statement, ast.ClassDef):
                self._constants_of(statement.body, f"{scope}{statement.name}.")
                continue
            target = (
                statement.targets[0]
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                else (
                    statement.target if isinstance(statement, ast.AnnAssign) and statement.value is not None else None
                )
            )
            value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
            if isinstance(target, ast.Name) and value is not None and _string_text(value, {}) is not None:
                self.constants.add(f"{self.path}::{scope}{target.id}")

    @property
    def _function_node(self) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """the innermost function the walk is in.

        :return: it, or None at module or class level
        :rtype: ast.FunctionDef | ast.AsyncFunctionDef | None
        """
        return self._function_nodes[-1] if self._function_nodes else None

    @property
    def _function(self) -> str:
        """the qualified name of the scope the walk is in.

        :return: dotted qualified name, or ``<module>``
        :rtype: str
        """
        return ".".join(self._stack) if self._stack else "<module>"

    def _bind(self, body: list[ast.stmt], env: dict[str, str]) -> None:
        """record the string each simple assignment under ``body`` binds (not in nested functions). A name the body
        binds more than once (``where = ""`` then ``where = "WHERE ..."``, or ``where += ...``) has no one text, so it
        is left unbound: a statement built from it reads it as an interpolation the walk cannot see.

        :param body: statements
        :ptype body: list[ast.stmt]
        :param env: name onto bound string, updated in place
        :ptype env: dict[str, str]
        :return: nothing
        :rtype: None
        """
        stores: dict[str, int] = {}
        for node in _own_nodes(body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                stores[node.id] = stores.get(node.id, 0) + 1
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                stores[node.target.id] = stores.get(node.target.id, 0) + 2
        for node in _own_nodes(body):
            # ``condition, params = filter_conditions(where_in={"run_id": ...}, not_null=...)``: the condition is the
            # columns it binds and the filters it adds
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)
                and node.targets[0].elts
                and isinstance(node.targets[0].elts[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and _final(node.value) in _FILTER_BUILDERS
            ):
                built = _filter_text(node.value)
                name = node.targets[0].elts[0].id
                if built is not None and stores.get(name, 0) == 1:
                    env[name] = built
                else:
                    env.pop(name, None)
                continue
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            if isinstance(target, ast.Name) and value is not None:
                text = _string_text(value, env)
                if text is not None and stores.get(target.id, 0) == 1:
                    env[target.id] = text
                else:
                    env.pop(target.id, None)

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
        # a source change read's terms hold in the function and every function nested in it
        self._source.append(self._source[-1] if self._source else _NO_SOURCE)
        self._mark_operator(node)
        if _is_stub(node):
            self.stubs.add(f"{self.path}::{self._function}")
        env = dict(self._envs[-1])
        self._bind(node.body, env)
        self._envs.append(env)
        self._ctes.append(_function_ctes(node.body))
        self._function_nodes.append(node)
        owner = self._classes[-1] if self._classes and self._function.startswith(f"{self._classes[-1]}.") else ""
        self._classes.append(owner)
        self.generic_visit(node)
        self._classes.pop()
        self._envs.pop()
        self._ctes.pop()
        self._function_nodes.pop()
        self._source.pop()
        self._stack.pop()

    def _mark_operator(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """record the function's operator mark, if it carries one (``@operator_reset(...)``).

        :param node: the function, already on the stack
        :ptype node: ast.FunctionDef | ast.AsyncFunctionDef
        :return: nothing
        :rtype: None
        """
        for decorator in node.decorator_list:
            mark = _final(decorator) if isinstance(decorator, ast.Call) else None
            if isinstance(decorator, ast.Call) and mark in _OPERATOR_MARKS:
                reason = decorator.args[0] if decorator.args else None
                literal = {
                    k.arg: k.value.value
                    for k in decorator.keywords
                    if k.arg and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)
                }
                stamps, ids = (
                    tuple(
                        element.value
                        for element in (node_.elts if isinstance(node_, (ast.Tuple, ast.List)) else [])
                        if isinstance(element, ast.Constant) and isinstance(element.value, str)
                    )
                    for node_ in (_keyword_value(decorator, "stamps"), _keyword_value(decorator, "ids"))
                )
                self.operators.append(
                    OperatorMark(
                        self.path,
                        self._function,
                        node.lineno,
                        mark,
                        reason.value if isinstance(reason, ast.Constant) and isinstance(reason.value, str) else "",
                        literal.get("proof", ""),
                        literal.get("speed", ""),
                        stamps,
                        ids,
                        literal.get("unit", ""),
                    )
                )
                if mark == "source_change_read":
                    self._source[-1] = (
                        frozenset(stamp.lower() for stamp in stamps),
                        frozenset(column.lower() for column in ids),
                        literal.get("unit", "") == "table",
                    )

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

    def _cte_parameters(self, function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        """the function's parameters that every call site in the repo fills with a string literal its own statements
        define as a CTE (``kept_runs_sql("runs")`` inside ``WITH runs AS (...)``).

        :param function: the function
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
        :return: the parameter names
        :rtype: set[str]
        """
        found: set[str] = set()
        for argument in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]:
            sites = self._cte_arguments.get((function.name, argument.arg), [])
            if sites and all(ok for _, ok in sites):
                found.add(argument.arg)
        return found

    def _reader(self, node: ast.AST) -> Reader:
        """what a statement at ``node`` is judged against: the schema, and the names of the code it is written in
        (its function, and the constant it is bound to).

        :param node: the statement's expression
        :ptype node: ast.AST
        :return: the reader
        :rtype: Reader
        """
        names = {f"{self.path}::{self._function}"}
        parent = self._parents.get(node)
        targets = (
            parent.targets
            if isinstance(parent, ast.Assign)
            else [parent.target]
            if isinstance(parent, ast.AnnAssign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name):
                scope = "" if self._function == "<module>" else f"{self._function}."
                names.add(f"{self.path}::{scope}{target.id}")
        stamps, ids, unit = self._source[-1] if self._source else _NO_SOURCE
        return Reader(
            schema=self._schema,
            names=frozenset(names),
            transient=self._transient_table,
            ctes=(self._ctes[-1] if self._ctes else frozenset())
            | frozenset(
                f"__cte_{name}" for name in (self._cte_parameters(self._function_node) if self._function_node else ())
            ),
            stamps=stamps,
            template=self._schema.templates.get(self.path),
            source_ids=ids,
            source_unit=unit,
        )

    def _judge_string(self, node: ast.expr) -> None:
        """judge a string expression that is SQL.

        :param node: the expression
        :ptype node: ast.expr
        :return: nothing
        :rtype: None
        """
        if isinstance(node, ast.JoinedStr) and self._maximal_string(node) and self._keyset_listing(node):
            return
        # a placeholder resolves to a module constant, or to a string the enclosing function binds once; a parameter
        # every caller fills with a CTE its own statement defines is that CTE (judged where the caller builds it)
        env = self._envs[-1]
        function = self._function_node
        cte_params = self._cte_parameters(function) if function is not None else set()
        if cte_params:
            env = {**env, **{name: f"__cte_{name}" for name in cte_params}}
        text = _string_text(node, env) if self._maximal_string(node) else None
        for finding in judge_sql(text, self._reader(node)) if text is not None and is_sql(text) else []:
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

    def _keyset_listing(self, node: ast.JoinedStr) -> bool:
        """judge a keyset listing (``SELECT ... FROM t {where_sql(conditions)} ORDER BY {order_by_clause(spec)}
        LIMIT $1``) once per column of its sort vocabulary: each ``SortColumn`` names an ``index`` declared on the
        table, whose purpose is this function and whose columns are a bound equality prefix then the sort column; the
        listing's conditions may only be that prefix, the index's partial ``WHERE``, or the page's range on the sort
        column (``keyset_condition``). A column failing any of it is ``unkeyed-sort``.

        :param node: an f-string
        :ptype node: ast.JoinedStr
        :return: whether it is such a listing (judged here), False to judge it as any statement
        :rtype: bool
        """
        order = next(
            (
                part.value
                for part in node.values
                if isinstance(part, ast.FormattedValue)
                and isinstance(part.value, ast.Call)
                and _final(part.value) == "order_by_clause"
            ),
            None,
        )
        top = self._stack[0] if self._stack else ""
        text = _string_text(node, self._envs[-1]) or ""
        table_match = re.search(r"\bFROM\s+([\w.\"]+)", text, re.IGNORECASE)
        if order is None or table_match is None:
            return False
        table = table_match.group(1)
        declared = self._schema.table(table)
        # the vocabularies handed to this listing; failing that, those whose sorts are indexes of its table
        mappings = sorted(self._mapping_users.get(top, set())) or sorted(
            name
            for name, sorts in self._sort_mappings.items()
            if declared is not None
            and any(
                (_string_text(_keyword_value(sort, "index") or ast.Constant(""), {}) or "").lower() in declared.indexes
                for sort in sorts
            )
        )
        if not mappings:
            return False
        conditions = self._listing_conditions(node)
        reader = self._reader(node)
        for mapping in mappings:
            for sort in self._sort_mappings[mapping]:
                column = (
                    (_string_text(_keyword_value(sort, "column") or sort.args[0], {}) or "?")
                    if (sort.args or _keyword_value(sort, "column"))
                    else "?"
                )
                index = _string_text(_keyword_value(sort, "index") or ast.Constant(""), self._envs[0]) or ""
                problem = self._sort_problem(table, column.rsplit(".", 1)[-1].lower(), index, conditions, reader)
                if problem is not None:
                    self._hit(node, f"unkeyed-sort {table}.{column}", f"{text} -- {mapping}: {problem}")
        return True

    def _listing_conditions(self, node: ast.JoinedStr) -> list[str | None]:
        """the conditions a listing's ``WHERE`` placeholder joins: each a string (``kind = $9``), None for the page's
        range on the sort column (``keyset_condition``), or ``{}`` for one the walk cannot see.

        :param node: the listing's f-string
        :ptype node: ast.JoinedStr
        :return: the conditions
        :rtype: list[str | None]
        """
        function = self._function_node
        names = {
            part.value.id
            for part in node.values
            if isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name)
        }
        listed: set[str] = set()
        for child in _own_nodes(function.body if function is not None else []):
            if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                value = child.value
                if child.targets[0].id in names and isinstance(value, ast.Call) and _final(value) == "where_sql":
                    listed.update(arg.id for arg in value.args if isinstance(arg, ast.Name))
        found: list[str | None] = []
        for child in _own_nodes(function.body if function is not None else []):
            elements: list[ast.expr] = []
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                target = child.targets[0] if isinstance(child, ast.Assign) else child.target
                if (
                    isinstance(target, ast.Name)
                    and target.id in listed
                    and isinstance(child.value, (ast.List, ast.Tuple))
                ):
                    elements = list(child.value.elts)
            elif isinstance(child, ast.Call) and _final(child) in {"append", "extend"}:
                receiver = child.func.value if isinstance(child.func, ast.Attribute) else None
                if isinstance(receiver, ast.Name) and receiver.id in listed and child.args:
                    argument = child.args[0]
                    elements = (
                        list(argument.elts)
                        if _final(child) == "extend" and isinstance(argument, (ast.List, ast.Tuple))
                        else [argument]
                    )
            for element in elements:
                if isinstance(element, ast.Call) and _final(element) == "keyset_condition":
                    found.append(None)
                    continue
                written = _string_text(element, self._envs[-1])
                found.append(re.sub(r"\$\{\}", "$9", written) if written is not None else "{}")
        return found

    def _sort_problem(
        self, table: str, column: str, index: str, conditions: list[str | None], reader: Reader
    ) -> str | None:
        """what keeps one sort column of a listing from being a keyset page through its index, if anything.

        :param table: the listing's table
        :ptype table: str
        :param column: the sort column, unqualified
        :ptype column: str
        :param index: the index its ``SortColumn`` names, empty for none
        :ptype index: str
        :param conditions: the listing's conditions (:meth:`_listing_conditions`)
        :ptype conditions: list[str | None]
        :param reader: the listing's reader
        :ptype reader: Reader
        :return: the problem, or None when every page is keyed
        :rtype: str | None
        """
        declared = self._schema.table(table)
        found = declared.indexes.get(index.lower()) if declared is not None and index else None
        problem: str | None = None
        if not index:
            problem = f"sort {column} names no index"
        elif found is None:
            problem = f"sort {column} names index {index}, which {table} does not declare"
        elif not set(purposes(found.purpose)) & reader.names:
            problem = f"index {index}'s purpose is {found.purpose or 'none'}, not this listing"
        elif column not in found.columns:
            problem = f"index {index} does not order by {column}"
        else:
            partial = " ".join(found.where.lower().split())
            prefix = found.columns[: found.columns.index(column)]
            bound: set[str] = set()
            for condition in conditions:
                shape = condition_shape(condition) if condition is not None else ("range", column)
                if condition is not None and " ".join(condition.lower().split()) == partial:
                    continue
                if condition is not None and _keyed_exists(condition, table, reader):
                    # a correlated EXISTS whose own read is keyed filters each row of the page, reading no more
                    continue
                if shape is not None and (shape == ("range", column) or (shape[0] == "eq" and shape[1] in prefix)):
                    bound.add(shape[1])
                    continue
                problem = (
                    f"condition {condition!r} is neither {index}'s equality prefix {list(prefix)}, its partial WHERE, "
                    f"nor the page's range on {column}"
                )
                break
            if problem is None and not set(prefix) <= bound:
                problem = f"no condition binds {sorted(set(prefix) - bound)}, the columns {index} orders {column} under"
            bare = bare_table(table)
            if problem is None and bare is not None:
                self._schema.use(bare, index.lower(), reader.names)
        return problem

    def _transient_table(self, table: str) -> bool:
        """whether a relation is a change set (a registered relation, or one the repo names transient), by its
        resolved name (a constant's text, a local copy's relation without its ``copy.``).

        :param table: the relation as written, its constants resolved
        :ptype table: str
        :return: whether it is no data table
        :rtype: bool
        """
        bare = table.strip('"').removeprefix(COPY_PREFIX)
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
        return _final(call) in self._exec_calls and (first or second or sql_keyword)

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
        # a record saying the whole table changed (``NotedChange(whole_table=True)``) is built, not read: the flag is
        # judged where it is consumed
        constructs = name[:1].isupper()
        for keyword in node.keywords:
            if (
                not constructs
                and keyword.arg in _WHOLE_FLAGS
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                self._hit(node, f"whole-flag {keyword.arg}=", spelled)
        if name in {"add", "update", "extend"} and segments and _WHOLE_SET.search(segments[-1]):
            self._hit(node, f"marks-whole {segments[-1]}", spelled)
        if name == "truncate" and segments and _STORE.search(segments[-1]):
            self._hit(node, "truncate-call truncate", spelled)
        if name in self._exec_calls:
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
                for finding in judge_sql(text, self._reader(node)):
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


def _keyed_exists(condition: str, table: str, reader: Reader) -> bool:
    """whether a listing's condition is a correlated ``EXISTS (SELECT ... WHERE inner.k = outer.k ...)`` whose own read
    is keyed: a per-row filter of the page (the caller-visibility clause), judged on its own.

    :param condition: the condition
    :ptype condition: str
    :param table: the listing's table, the outer row the subquery correlates with
    :ptype table: str
    :param reader: the listing's reader
    :ptype reader: Reader
    :return: whether it is
    :rtype: bool
    """
    stripped = condition.strip()
    if re.match(r"(?:NOT\s+)?EXISTS\s*\(", stripped, re.IGNORECASE) is None:
        return False
    # the subquery is judged inside a statement reading the listing's table by its key, so only its own read counts
    probe = f"SELECT 1 FROM {table} WHERE {{pk}} = $1 AND {stripped}"
    return not judge_sql(probe, reader)


#: 3tears' filter builders (``threetears.core.sql_fragments``), whose condition the walk renders from their arguments
_FILTER_BUILDERS: Final[frozenset[str]] = frozenset({"filter_conditions", "equality_conditions"})


def _filter_text(call: ast.Call) -> str | None:
    """the condition a 3tears filter builder returns, from its arguments: ``where`` columns as equalities, ``where_in``
    columns as ``IN`` a bound set, ``not_null`` as a filter on no key.

    :param call: the ``filter_conditions(...)`` / ``equality_conditions(...)`` call
    :ptype call: ast.Call
    :return: the condition, or None when a mapping of columns is not written out (its columns are unknown)
    :rtype: str | None
    """
    positions = ("where", "where_in", "not_null")
    given = dict(zip(positions, call.args, strict=False)) | {k.arg: k.value for k in call.keywords if k.arg}
    parts: list[str] = []
    for argument, form in (("where", "{} = $9"), ("where_in", "{} IN ($9)")):
        mapping = given.get(argument)
        if mapping is None or (isinstance(mapping, ast.Constant) and mapping.value is None):
            continue
        if not isinstance(mapping, ast.Dict) or not all(
            isinstance(key, ast.Constant) and isinstance(key.value, str) for key in mapping.keys
        ):
            return None
        parts.extend(form.format(key.value) for key in mapping.keys if isinstance(key, ast.Constant))
    if given.get("not_null") is not None:
        parts.append("__not_null__ IS NOT NULL")
    return " AND ".join(parts) or None


def _keyword_value(call: ast.Call, name: str) -> ast.expr | None:
    """a call's keyword argument.

    :param call: the call
    :ptype call: ast.Call
    :param name: the keyword
    :ptype name: str
    :return: its value, or None
    :rtype: ast.expr | None
    """
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _sort_mappings(tree: ast.Module) -> dict[str, list[ast.Call]]:
    """a module's sort vocabularies: each module-level dict whose values are all ``SortColumn(...)``.

    :param tree: the module
    :ptype tree: ast.Module
    :return: the vocabulary's name onto its ``SortColumn`` calls
    :rtype: dict[str, list[ast.Call]]
    """
    found: dict[str, list[ast.Call]] = {}
    for statement in tree.body:
        target = (
            statement.targets[0]
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1
            else (statement.target if isinstance(statement, ast.AnnAssign) else None)
        )
        value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Dict)
            and value.values
            and all(isinstance(entry, ast.Call) and _final(entry) == "SortColumn" for entry in value.values)
        ):
            found[target.id] = [entry for entry in value.values if isinstance(entry, ast.Call)]
    return found


def _mapping_users(tree: ast.Module, mappings: set[str]) -> dict[str, set[str]]:
    """each module function onto the sort vocabularies actually handed to it: an endpoint resolving
    ``spec = _SORT_COLUMNS[sort]`` and passing ``spec`` (or ``_SORT_COLUMNS[sort]`` itself) to the listing, or the
    listing resolving it itself.

    :param tree: the module
    :ptype tree: ast.Module
    :param mappings: the module's vocabulary names
    :ptype mappings: set[str]
    :return: function name onto vocabulary names
    :rtype: dict[str, set[str]]
    """

    def subscripted(node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in mappings:
            return node.value.id
        return None

    users: dict[str, set[str]] = {}
    for function in (s for s in tree.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))):
        resolved: dict[str, str] = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                mapping = subscripted(node.value)
                if mapping is not None:
                    resolved[node.targets[0].id] = mapping
                    users.setdefault(function.name, set()).add(mapping)
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                    mapping = subscripted(argument) or (
                        resolved.get(argument.id) if isinstance(argument, ast.Name) else None
                    )
                    if mapping is not None:
                        users.setdefault(node.func.id, set()).add(mapping)
    return users


#: a CTE a statement defines: ``WITH name AS (``, ``, name AS (``
_CTE_DEFINED: Final[re.Pattern[str]] = re.compile(
    r"(?:\bWITH(?:\s+RECURSIVE)?|,)\s*([A-Za-z_]\w*)\s*(?:\([^()]*\)\s*)?AS\s*(?:NOT\s+)?(?:MATERIALIZED\s+)?\(",
    re.IGNORECASE,
)


#: a temporary view a statement creates, and its body
_TEMP_VIEW: Final[re.Pattern[str]] = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?TEMP(?:ORARY)?\s+VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?\"?(\w+)\"?\s+AS\s+(.*)",
    re.IGNORECASE | re.DOTALL,
)

#: a relation a ``FROM`` or ``JOIN`` names
_READS_FROM: Final[re.Pattern[str]] = re.compile(r"\b(?:FROM|JOIN)\s+\"?([\w{}]+)\"?", re.IGNORECASE)


def _transient_views(tree: ast.Module, registered: set[str], transient: re.Pattern[str] | None) -> set[str]:
    """the temporary views a module creates over only transient relations (rows registered from memory, a stage):
    each is those rows too, never a data table.

    :param tree: the module
    :ptype tree: ast.Module
    :param registered: the relations the module registers from memory
    :ptype registered: set[str]
    :param transient: the repo's transient relations
    :ptype transient: re.Pattern[str] | None
    :return: the views
    :rtype: set[str]
    """
    views: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        text = _string_text(node, {}) if isinstance(node, (ast.Constant, ast.JoinedStr)) else None
        match = _TEMP_VIEW.search(text) if isinstance(text, str) else None
        if match is not None:
            # an f-string's own fragments match too, reading nothing: merge, never overwrite
            views.setdefault(match.group(1), []).extend(_READS_FROM.findall(match.group(2)))
    found: set[str] = set()
    grew = True
    while grew:
        grew = False
        for view, reads in views.items():
            known = registered | found
            if (
                view not in found
                and reads
                and all(name in known or (transient is not None and transient.search(name)) for name in reads)
            ):
                found.add(view)
                grew = True
    return found


def _function_ctes(body: list[ast.stmt]) -> frozenset[str]:
    """the CTE names a function's own statements define, lower-cased: a fragment another of its statements reads
    from (``FROM stage0{}``) is that CTE, not a table.

    :param body: the function's statements
    :ptype body: list[ast.stmt]
    :return: the names
    :rtype: frozenset[str]
    """
    return frozenset(
        match.group(1).lower()
        for node in _own_nodes(body)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        for match in _CTE_DEFINED.finditer(node.value)
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


def _class_module(
    call: _CallSite, module_classes: dict[str, set[str]], imports: dict[str, dict[str, str]]
) -> str | None:
    """the module defining the class a bare call constructs, when the call names one of the repo's classes.

    :param call: a bare call
    :ptype call: _CallSite
    :param module_classes: module path onto the names of its top-level classes
    :ptype module_classes: dict[str, set[str]]
    :param imports: module path onto its imported names, each onto ``<module>.<name>``
    :ptype imports: dict[str, dict[str, str]]
    :return: the module's path, or None when the call constructs none of the repo's classes
    :rtype: str | None
    """
    found: str | None = None
    if call.name in module_classes.get(call.path, set()):
        found = call.path
    else:
        origin = imports.get(call.path, {}).get(call.name)
        module = origin.rsplit(".", 1)[0].replace(".", "/") + ".py" if origin else None
        found = next(
            (path for path, names in module_classes.items() if module and path.endswith(module) and call.name in names),
            None,
        )
    return found


def _targets(
    call: _CallSite,
    module_functions: dict[str, set[str]],
    by_name: dict[str, list[tuple[str, str]]],
    imports: dict[str, dict[str, str]],
    module_classes: dict[str, set[str]] | None = None,
) -> list[tuple[str, str]] | None:
    """the repo's functions a call may reach.

    A ``self`` call reaches its own class's method; a bare call its module's function, else the function of the
    repo module it was imported from; a bare call constructing one of the repo's classes that class's
    ``__init__`` (a class with none, a dataclass, reaches nothing and reads nothing); a call on any other receiver
    the repo's methods of that name (those of the classes named for the receiver, when there are some).

    :param call: the call
    :ptype call: _CallSite
    :param module_functions: module path onto the names of its module-level functions
    :ptype module_functions: dict[str, set[str]]
    :param by_name: final name onto ``(path, qualified name)`` of every function of the repo
    :ptype by_name: dict[str, list[tuple[str, str]]]
    :param imports: module path onto its imported names, each onto ``<module>.<name>``
    :ptype imports: dict[str, dict[str, str]]
    :param module_classes: module path onto the names of its top-level classes
    :ptype module_classes: dict[str, set[str]] | None
    :return: the targets, the caller itself left out; None for a constructor of a class with no ``__init__``,
        which the walk found and which reaches nothing
    :rtype: list[tuple[str, str]] | None
    """
    targets: list[tuple[str, str]] = []
    constructed = (
        _class_module(call, module_classes or {}, imports)
        if call.receiver == "bare" and call.name not in module_functions.get(call.path, set())
        else None
    )
    if constructed is not None:
        init = (constructed, f"{call.name}.__init__")
        return [init] if init in by_name.get("__init__", []) else None
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
    module_classes: dict[str, set[str]] | None = None,
    operators: list[OperatorMark] | None = None,
) -> tuple[list[WholeReadHit], list[str]]:
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
    :param module_classes: module path onto the names of its top-level classes
    :ptype module_classes: dict[str, set[str]] | None
    :param operators: the functions marked as an operator's own command
    :ptype operators: list[OperatorMark] | None
    :return: the hits, and each call reaching a marked function from a function without the same mark
    :rtype: tuple[list[WholeReadHit], list[str]]
    """
    by_name: dict[str, list[tuple[str, str]]] = {}
    for entry in definitions - stubs:
        path, _, qualified = entry.partition("::")
        by_name.setdefault(qualified.rsplit(".", 1)[-1], []).append((path, qualified))
    # a security backstop's pass is accepted where it is marked: its caller (the loop running it) reads nothing whole
    backstops = {
        (mark.path, mark.function)
        for mark in operators or []
        if mark.mark == "security_backstop" or (mark.mark == "source_change_read" and mark.unit == "table")
    }
    readers: dict[tuple[str, str], WholeReadHit] = {}
    for hit in hits:
        if hit.kind in _READ_KINDS and (hit.path, hit.function) not in backstops:
            readers.setdefault((hit.path, hit.function), hit)
    constructors_found: set[int] = set()
    resolved: list[tuple[_CallSite, list[tuple[str, str]]]] = []
    for index, call in enumerate(calls):
        targets = _targets(call, module_functions, by_name, imports, module_classes)
        if targets is None:
            constructors_found.add(index)
        resolved.append((call, targets or []))
    reaches = {(hit.path, hit.function) for hit in hits} - backstops
    grew = True
    while grew:
        grew = False
        for call, targets in resolved:
            caller = (call.path, call.function)
            if caller not in reaches and targets and all(target in reaches for target in targets):
                reaches.add(caller)
                grew = True
    found: list[WholeReadHit] = []
    marks = {(mark.path, mark.function): mark.mark for mark in operators or [] if mark.mark in _ENTRY_ONLY_MARKS}
    callers = [
        f"{call.path}:{call.line} in {call.function}: {call.spelled} calls {target[1]}, marked @{marks[target]}: an "
        "operator's command is reached only from its own entry, so every caller carries the same mark"
        for call, targets in resolved
        for target in targets
        if target in marks and marks.get((call.path, call.function)) != marks[target]
    ]
    for index, (call, targets) in enumerate(resolved):
        if index in constructors_found:
            continue
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
    return found, callers


def _cte_arguments(modules: list[tuple[str, ast.Module]]) -> dict[tuple[str, str], list[tuple[str, bool]]]:
    """for every function the repo calls by name, each call site's argument per parameter: whether it is a string
    literal its calling function's own statements define as a CTE (``WITH <name> AS``).

    :param modules: each module's path and parse
    :ptype modules: list[tuple[str, ast.Module]]
    :return: ``(function, parameter)`` onto ``[(site, is a CTE the caller defines)]``
    :rtype: dict[tuple[str, str], list[tuple[str, bool]]]
    """
    definitions: dict[str, list[str]] = {}
    for _, tree in modules:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.setdefault(node.name, [a.arg for a in [*node.args.posonlyargs, *node.args.args]])
    found: dict[tuple[str, str], list[tuple[str, bool]]] = {}
    for relative, tree in modules:
        for caller in ast.walk(tree):
            if not isinstance(caller, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                continue
            body = caller.body
            ctes = _function_ctes(body) | _formatted_ctes(body)
            for call in _own_nodes(body):
                if not isinstance(call, ast.Call):
                    continue
                name = _final(call) or ""
                parameters = definitions.get(name)
                if parameters is None:
                    continue
                params = parameters[1:] if parameters[:1] in (["self"], ["cls"]) else parameters
                passed = [*zip(params, call.args, strict=False), *((k.arg, k.value) for k in call.keywords if k.arg)]
                for parameter, value in passed:
                    literal = value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None
                    site = f"{relative}:{call.lineno}"
                    found.setdefault((name, parameter), []).append(
                        (site, literal is not None and literal.lower() in ctes)
                    )
    return found


def _formatted_ctes(body: list[ast.stmt]) -> frozenset[str]:
    """the CTE names a function's f-strings define (``f"WITH runs AS ({...})"``).

    :param body: the function's statements
    :ptype body: list[ast.stmt]
    :return: the names, lower-cased
    :rtype: frozenset[str]
    """
    return frozenset(
        match.group(1).lower()
        for node in _own_nodes(body)
        if isinstance(node, ast.JoinedStr)
        for match in _CTE_DEFINED.finditer(_string_text(node, {}) or "")
    )


def _module_strings(tree: ast.Module) -> dict[str, str]:
    """the strings a module's top level binds once, by name: what another module importing one of them reads.

    :param tree: the module
    :ptype tree: ast.Module
    :return: name onto string
    :rtype: dict[str, str]
    """
    found: dict[str, str] = {}
    twice: set[str] = set()
    for statement in tree.body:
        if (
            isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.body
            and isinstance(statement.body[-1], ast.Return)
            and statement.body[-1].value is not None
        ):
            # a function returning a string: a name it builds at run time is its template
            template = _string_text(statement.body[-1].value, found)
            if template is not None:
                found[_TEMPLATE.format(statement.name)] = template
            continue
        target = (
            statement.targets[0]
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1
            else (statement.target if isinstance(statement, ast.AnnAssign) and statement.value is not None else None)
        )
        value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(target, ast.Name) and value is not None:
            text = _string_text(value, found)
            if target.id in found:
                twice.add(target.id)
            if text is not None:
                found[target.id] = text
    return {name: text for name, text in found.items() if name not in twice}


def _imported_strings(tree: ast.Module, strings: dict[str, dict[str, str]]) -> dict[str, str]:
    """the strings a module imports from the repo's own modules (``from pkg.tables import GEOS``), by the name it
    imports each as.

    :param tree: the module
    :ptype tree: ast.Module
    :param strings: every walked module's path onto its once-bound strings (:func:`_module_strings`)
    :ptype strings: dict[str, dict[str, str]]
    :return: name onto string
    :rtype: dict[str, str]
    """
    found: dict[str, str] = {}
    for statement in tree.body:
        if not (isinstance(statement, ast.ImportFrom) and statement.module and not statement.level):
            continue
        base = statement.module.replace(".", "/")
        source = next(
            (
                names
                for path, names in strings.items()
                if path.endswith((f"/{base}.py", f"/{base}/__init__.py"))
                or path in {f"{base}.py", f"{base}/__init__.py"}
            ),
            None,
        )
        for alias in statement.names:
            if source is not None and alias.name in source:
                found[alias.asname or alias.name] = source[alias.name]
            if source is not None and _TEMPLATE.format(alias.name) in source:
                found[_TEMPLATE.format(alias.asname or alias.name)] = source[_TEMPLATE.format(alias.name)]
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
    schema: SchemaCatalog | None = None,
    exec_calls: frozenset[str] = frozenset(),
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
    :param schema: every table's declared key and indexes; None judges against none, so every predicate on a table
        is ``unknown-schema``
    :ptype schema: SchemaCatalog | None
    :param exec_calls: the repo's own calls that run their first argument as SQL (``Snapshot.ask``), beside 3tears'
    :ptype exec_calls: frozenset[str]
    :return: the walk
    :rtype: WholeReadScan
    """
    scan = WholeReadScan(schema=schema if schema is not None else SchemaCatalog())
    used: set[str] = set()
    calls: list[_CallSite] = []
    module_functions: dict[str, set[str]] = {}
    module_classes: dict[str, set[str]] = {}
    stubs: set[str] = set()
    imports: dict[str, dict[str, str]] = {}
    pattern = re.compile(transient) if transient else None
    modules = [(relative_posix_path(path, repo_root), path) for root in src_roots for path in iter_python_files(root)]
    strings = {relative: _module_strings(parse_python_path(path)) for relative, path in modules}
    cte_arguments = _cte_arguments([(relative, parse_python_path(path)) for relative, path in modules])
    for relative, path in modules:
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
            schema=scan.schema,
            imported=_imported_strings(tree, strings),
            exec_calls=exec_calls,
            cte_arguments=cte_arguments,
        )
        walk.visit(tree)
        stubs.update(walk.stubs)
        scan.hits.extend(walk.hits)
        scan.definitions.update(walk.definitions)
        scan.constants.update(walk.constants)
        calls.extend(walk.calls)
        module_functions[relative] = walk.module_functions
        module_classes[relative] = walk.module_classes
        scan.operators.extend(walk.operators)
        imports[relative] = walk.imports
        scan.files_scanned += 1
    resolved, callers = _resolve(
        calls, scan.hits, module_functions, scan.definitions, stubs, imports, module_classes, scan.operators
    )
    scan.hits.extend(resolved)
    scan.operator_callers = callers
    scan.unmatched_exclusions = [prefix for prefix in excluded_prefixes if prefix not in used]
    return scan
