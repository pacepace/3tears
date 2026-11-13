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
import itertools
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

from threetears.enforcement.common import dotted, iter_python_files, parse_python_path, relative_posix_path
from threetears.enforcement.no_whole_reads.bounded import BoundBy, Bounded, bounded_relations
from threetears.enforcement.no_whole_reads.flow import (
    CLEAN,
    HANDED,
    Callee,
    RelationFlow,
    RepoForwarders,
    step_builders,
    default_of,
    local_names,
    step_relation_arguments,
)
from threetears.enforcement.no_whole_reads.helpers import (
    COPY_CALLS,
    KEYED_FRAGMENT_BUILDERS,
    BUILDER_CALLS,
    KEYED_HELPER_CALLS,
    STEP_CALLS,
    STEP_RELATIONS,
    STEP_TARGETS,
    START_KEYWORDS,
    HelperStatement,
    UNSEEN,
    helper_statements,
    is_keyed_fragment,
    narrows,
    templated,
)
from threetears.enforcement.no_whole_reads.sentinels import SentinelTemplates, declared_templates
from threetears.enforcement.no_whole_reads.schema import (
    COPY_PREFIX,
    PRODUCED_PREFIX,
    SchemaCatalog,
    bare_table,
    purposes,
)
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
    "DECLARED_KINDS",
    "DECLARED_OUTSIDE_KINDS",
    "DECLARED_SPEED_KINDS",
    "DEFAULT_NARROWING_KEYWORDS",
    "DEFAULT_PAGER_CALLS",
    "DECLARATION_CALL",
    "DeclaredWhole",
    "OperatorMark",
    "WholeReadHit",
    "WholeReadScan",
    "WrittenStatement",
    "declared_whole_read",
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

#: what a reviewed visibility clause (``caller_visible_customer_clause(...)``) reads as where it is placed: a bound
#: predicate on no key, a per-row filter
_KEYED_FRAGMENT_TEXT: Final[str] = "__keyed_fragment__ = $9"

#: a relation a declared transient producer returned (``Stages._stage(stage)``, ``work.put(rows)``): a change set
#: being applied, never a data table
TRANSIENT_PRODUCED: Final[str] = "__transient_produced__"

#: the most combinations of resolved parameter values a keyed helper call is judged under; past it, unresolved
_MAX_PARAMETER_COMBINATIONS: Final[int] = 16

#: the most calls a declared parameter's values make of one helper call (``declared_parameters``): each value is one
#: relation or column of a family the repo names, judged in turn, so a family is linear, never a cross of sites
_MAX_DECLARED_COMBINATIONS: Final[int] = 512

#: how many calls up a parameter's value is followed (``run`` -> ``_write`` -> ``_held`` -> the helper)
_PARAMETER_DEPTH: Final[int] = 6

#: a declared object a module tuple holds (``Declared(name=COUNTIES)``): its keyword arguments that read as strings,
#: sorted, so ``table.name`` reads as the object's ``name``
Declared = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Spelled:
    """what a call site passes a parameter that is no string, rebuilt in the caller's own terms as a literal a keyed
    helper's arguments read: a tuple of strings (a module tuple of key columns), or a mapping whose keys are strings
    (``{"conversation_id": conv}``), its values opaque (:data:`_PASSED_VALUE`) unless literals.

    :ivar text: the literal's source text
    """

    text: str

    def node(self) -> ast.expr:
        """the literal, parsed afresh.

        :return: the expression
        :rtype: ast.expr
        """
        return ast.parse(self.text, mode="eval").body


#: a value a call site passes in a mapping that is not a literal: a bound value, read as a parameter is
_PASSED_VALUE: Final[str] = "__passed_value__"

#: what a call site passes a parameter: a string, a declared object, a rebuilt literal, or None for anything the walk
#: cannot read
Passed = str | Declared | Spelled | None


@dataclass(frozen=True)
class Site:
    """one call site of a function: where it is, and what it passes each parameter.

    :ivar path: the calling module's repo-relative path
    :ivar line: the call's line
    :ivar function: the calling scope's qualified name (``<module>`` at a top level)
    :ivar values: each parameter onto what the site passes it
    """

    path: str
    line: int
    function: str
    values: dict[str, list[Passed]]


#: the calls that quote a relation's name, read through to the name they quote
_QUOTING_CALLS: Final = frozenset({"quoted_table", "quote_identifier"})

#: where 3tears' keyed helpers live: the runners in core, the pure builders in the dependency-free interfaces package; a
#: helper there may declare its own parameter a change relation
_KEYED_PACKAGES: Final = ("threetears/core/keyed/", "threetears/interfaces/keyed/")

#: the modules 3tears' keyed helpers are imported from: core's (the runners, every builder re-exported) and the
#: interfaces package's (the builders, for a package that may not depend on core)
_KEYED_MODULES: Final = ("threetears.core.keyed", "threetears.interfaces.keyed")

#: the modules holding a keyed helper in :data:`KEYED_HELPER_CALLS`: :data:`_KEYED_PACKAGES` and the one the
#: caller's grants filter (``visible_rows``, :mod:`threetears.agent.acl.visible_customers`); a statement over a
#: helper's relation parameter there is judged at the helper's call sites (:func:`_helper_relations`)
_KEYED_HELPER_MODULES: Final = (*_KEYED_PACKAGES, "threetears/agent/acl/visible_customers.py")

#: what an expression reads as when it names a relation no statement is judged by where it is written (a declared
#: transient producer's return, :data:`TRANSIENT_PRODUCED`; a keyed helper's relation parameter,
#: :data:`HELPER_RELATION`), None for any other; None when the module has neither
Produced = Callable[[ast.expr], str | None] | None

#: the relation a 3tears keyed helper is handed (a parameter of a helper in :data:`KEYED_HELPER_CALLS` or
#: :data:`STEP_CALLS`, or of a private renderer every in-module caller hands one): a statement over it is judged at
#: the helper's call sites, by the relation each hands it, never in the helper's body
HELPER_RELATION: Final[str] = "__keyed_helper_relation__"

#: the generic containers whose one type argument is the type of each element they yield
_ELEMENT_CONTAINERS: Final = frozenset(
    {"Sequence", "Iterable", "Iterator", "Collection", "list", "set", "frozenset", "AbstractSet", "MutableSequence"}
)
#: the generic mappings, whose ``.items()`` yields ``tuple[key, value]`` and ``.values()`` the value
_MAPPINGS: Final = frozenset({"Mapping", "dict", "MutableMapping"})


def produced_relation(entry: str) -> str:
    """the relation name a declared relation producer's run-time relations are judged as.

    :param entry: the declaration (``SourceTable.relation``, ``derived``)
    :ptype entry: str
    :return: ``produced.<entry>``, a dot written as ``__``, lower case
    :rtype: str
    """
    return PRODUCED_PREFIX + entry.replace(".", "__").lower()


def _generic(node: ast.expr) -> tuple[str, list[ast.expr]]:
    """an annotation's outer name and its type arguments (``Sequence[tuple[A, B]]`` -> ``Sequence``, ``[tuple[A,
    B]]``); a bare name has none.

    :param node: the annotation
    :ptype node: ast.expr
    :return: the name (empty when it is neither a name nor a subscript of one) and the arguments
    :rtype: tuple[str, list[ast.expr]]
    """
    if isinstance(node, ast.Subscript):
        name = node.value.attr if isinstance(node.value, ast.Attribute) else getattr(node.value, "id", "")
        given = node.slice
        return name, list(given.elts) if isinstance(given, ast.Tuple) else [given]
    if isinstance(node, ast.Attribute):
        return node.attr, []
    if isinstance(node, ast.Name):
        return node.id, []
    return "", []


def _optional_of(node: ast.expr) -> ast.expr:
    """an annotation with any ``| None`` / ``Optional[...]`` taken off (``C | None`` -> ``C``); a string annotation
    parsed.

    :param node: the annotation
    :ptype node: ast.expr
    :return: the annotation it allows besides None
    :rtype: ast.expr
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _optional_of(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return node
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        sides = [
            side for side in (node.left, node.right) if not (isinstance(side, ast.Constant) and side.value is None)
        ]
        return _optional_of(sides[0]) if len(sides) == 1 else node
    name, arguments = _generic(node)
    if name == "Optional" and len(arguments) == 1:
        return _optional_of(arguments[0])
    return node


def _element_type(node: ast.expr) -> ast.expr | None:
    """the type each element of an iterable of annotation ``node`` has: ``Sequence[T]`` / ``list[T]`` -> ``T``;
    ``tuple[T, ...]`` -> ``T``; None when the walk cannot tell.

    :param node: the iterable's annotation
    :ptype node: ast.expr
    :return: the element's annotation, or None
    :rtype: ast.expr | None
    """
    name, arguments = _generic(_optional_of(node))
    if name in _ELEMENT_CONTAINERS and len(arguments) == 1:
        return arguments[0]
    if name == "tuple" and len(arguments) == 2 and isinstance(arguments[1], ast.Constant) and arguments[1].value is ...:
        return arguments[0]
    return None


def _target_type(target: ast.expr, name: str, annotation: ast.expr | None) -> ast.expr | None:
    """the annotation a loop target binds ``name`` to, its element's annotation ``annotation``: the element itself
    for a bare name; a tuple target's matching position in a ``tuple[...]`` element, nested as written.

    :param target: the loop's (or comprehension's) target
    :ptype target: ast.expr
    :param name: the name looked for
    :ptype name: str
    :param annotation: the element's annotation, or None
    :ptype annotation: ast.expr | None
    :return: the annotation, or None when the target does not bind the name or the walk cannot tell its type
    :rtype: ast.expr | None
    """
    if annotation is None:
        return None
    if isinstance(target, ast.Name):
        return annotation if target.id == name else None
    if isinstance(target, (ast.Tuple, ast.List)):
        outer, parts = _generic(_optional_of(annotation))
        if outer != "tuple" or len(parts) != len(target.elts):
            return None
        for element, part in zip(target.elts, parts, strict=True):
            found = _target_type(element, name, part)
            if found is not None:
                return found
    return None


#: the calls of ``threetears/core/keyed`` whose first argument is the relation a statement names (``quoted_table``,
#: ``relation(table, "t")``, a step statement's ``source(rel, "t")``, ``_as_step(step)``)
_RELATION_CALLS: Final = frozenset({"quoted_table", "relation", "source", "_as_step"})

#: the calls that quote any name: a relation only where the statement names one there (:data:`_RELATION_POSITION`)
_NAME_QUOTERS: Final = frozenset({"quote_identifier", "quote"})

#: the text a placeholder follows where it names the relation a statement reads or writes
_RELATION_POSITION: Final = re.compile(r"\b(?:FROM|JOIN|UPDATE|INTO)\s*$", re.IGNORECASE)

#: the keyset pagers whose pages are written out to record the purposed index they read through
_CREDITED_PAGERS: Final[frozenset[str]] = frozenset({"keyset_pages", "read_keyset_pages", "keyset_arrow_pages"})

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

#: the call declaring a read whole by nature (:func:`threetears.core.declared.whole_read`)
DECLARATION_CALL: Final[str] = "whole_read"

#: each kind a read may be declared whole in (Pace, 2026-11-12) onto the hit kinds it lets through inside the wrapped
#: expression; anything else there is still reported. A new kind is a rule change Pace approves.
DECLARED_KINDS: Final[dict[str, frozenset[str]]] = {
    # the one fill of an empty copy: its whole select or pager, and a call into a function declared the same
    "cold_start": frozenset(
        {
            "whole-select",
            "unkeyed-pager",
            "unkeyed-sort",
            "may-read-whole",
            "whole-read",
            "whole-call",
            "calls-whole-read",
        }
    ),
    # an answer for every state (a national panel): its rows are every state's, filtered or summed
    "every_state_answer": frozenset(
        {
            "whole-select",
            "whole-aggregate",
            "non-key-where",
            "unkeyed-sort",
            "unkeyed-pager",
            "may-read-whole",
            "whole-read",
            "calls-whole-read",
        }
    ),
    # a backup's own count and checksum, taken inside its snapshot, of every table it holds (named at run time, so
    # read as unknown, cut at the backup's instant where a column says when a row was written)
    "backup_verification": frozenset(
        {
            "whole-select",
            "whole-aggregate",
            "verify-aggregate",
            "unknown-schema",
            "unknown-predicate",
            "calls-whole-read",
        }
    ),
    # a warehouse relation holding no key to read by (never an L3 table)
    "unbounded_by_warehouse": frozenset(
        {
            "whole-select",
            "whole-aggregate",
            "non-key-where",
            "unknown-schema",
            "unknown-predicate",
            "unkeyed-sort",
            "unkeyed-pager",
            "may-read-whole",
            "whole-read",
            "calls-whole-read",
        }
    ),
    # a schema change rewriting the table (named at run time, as a migration names the schema it runs in, so read as
    # unknown), and a call into a function declared the same (a rebuild's call into its duplicate check)
    "schema_change": frozenset(
        {
            "whole-select",
            "whole-aggregate",
            "whole-maintenance",
            "whole-update",
            "unkeyed-pager",
            "unknown-schema",
            "whole-call",
            "calls-whole-read",
        }
    ),
    # the drain of a process-local buffer (never an L3 table), and a call into a function declared the same
    "drain": frozenset({"whole-select", "whole-delete", "calls-whole-read"}),
    # an operator's reset
    "operator_reset": frozenset({"truncate", "truncate-call", "whole-delete"}),
    # a read serving writers older than a rolling upgrade, until the release it names (``until=``): the gate refuses it
    # once the package reaches that release (Pace 2026-11-12: rolling_upgrade until 0.96.0)
    "rolling_upgrade": frozenset(
        {
            "whole-select",
            "unkeyed-pager",
            "unkeyed-sort",
            "may-read-whole",
            "whole-read",
            "whole-call",
            "calls-whole-read",
        }
    ),
}

#: the kinds whose declaration names a speed test: they read our own tables at size. A ``backup_verification`` whose
#: every statement is a bare ``count(*)`` or row checksum over one table (no join, filter or ordering) may omit it: the
#: statement IS the bare operation (coordinator ruling 2026-11-12)
DECLARED_SPEED_KINDS: Final[frozenset[str]] = frozenset(
    {"cold_start", "every_state_answer", "backup_verification", "schema_change", "rolling_upgrade"}
)

#: the kinds whose relation is never a declared L3 table: a process-local buffer, a warehouse relation
DECLARED_OUTSIDE_KINDS: Final[frozenset[str]] = frozenset({"drain", "unbounded_by_warehouse"})

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
    :ivar reaches: for a ``whole-call``, the functions (path, qualified name) its call reaches when the walk found
        them; empty otherwise. Not part of the hit's identity
    """

    path: str
    line: int
    function: str
    symbol: str
    statement: str
    reaches: tuple[tuple[str, str], ...] = field(default=(), compare=False)

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


def module_constant_spans(tree: ast.Module) -> dict[str, tuple[int, int]]:
    """each module-level name bound once, onto the lines of its value: where a statement written as a module constant
    is judged (its hits are the module's, at those lines).

    :param tree: the module
    :ptype tree: ast.Module
    :return: name onto its value's first and last line
    :rtype: dict[str, tuple[int, int]]
    """
    spans: dict[str, tuple[int, int]] = {}
    seen: set[str] = set()
    for statement in tree.body:
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
            if isinstance(statement, ast.AnnAssign) and statement.value is not None
            else []
        )
        value = getattr(statement, "value", None)
        for target in targets:
            if isinstance(target, ast.Name) and value is not None:
                if target.id in seen:
                    spans.pop(target.id, None)
                    continue
                seen.add(target.id)
                spans[target.id] = (value.lineno, value.end_lineno or value.lineno)
    return spans


class _ConstantUses(ast.NodeVisitor):
    """each function (qualified as the walk names it) and the module constants it names."""

    def __init__(self, constants: Mapping[str, tuple[int, int]]) -> None:
        """prepare to find the uses of ``constants``.

        :param constants: each module constant onto its value's lines
        :ptype constants: Mapping[str, tuple[int, int]]
        """
        self._constants = constants
        self._stack: list[str] = []
        self.uses: set[tuple[str, int, int]] = set()

    def _scope(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """walk a scope with its name on the stack.

        :param node: the scope
        :ptype node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        :return: nothing
        :rtype: None
        """
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """a class's scope.

        :param node: the class
        :ptype node: ast.ClassDef
        :return: nothing
        :rtype: None
        """
        self._scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """a function's scope.

        :param node: the function
        :ptype node: ast.FunctionDef
        :return: nothing
        :rtype: None
        """
        self._scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """a coroutine function's scope.

        :param node: the function
        :ptype node: ast.AsyncFunctionDef
        :return: nothing
        :rtype: None
        """
        self._scope(node)

    def visit_Name(self, node: ast.Name) -> None:
        """a name: a module constant's use by the function the walk is in.

        :param node: the name
        :ptype node: ast.Name
        :return: nothing
        :rtype: None
        """
        span = self._constants.get(node.id)
        # a function's own use of a module constant; a constant naming another is the module's, not a function's
        if span is not None and isinstance(node.ctx, ast.Load) and self._stack:
            self.uses.add((".".join(self._stack), *span))


def constant_readers(path: str, tree: ast.Module) -> set[tuple[str, str, int, int]]:
    """``(path, function, first, last)`` of each function naming a module constant of ``path``, with the lines of
    that constant's value: a whole read written there is the function's for its callers (``calls-whole-read``).

    :param path: the module's repo-relative path
    :ptype path: str
    :param tree: the module
    :ptype tree: ast.Module
    :return: the uses
    :rtype: set[tuple[str, str, int, int]]
    """
    finder = _ConstantUses(module_constant_spans(tree))
    finder.visit(tree)
    return {(path, function, first, last) for function, first, last in finder.uses}


@dataclass(frozen=True)
class DeclaredWhole:
    """a read declared whole by nature where it is written (``whole_read(<read>, kind=..., ...)``).

    :ivar path: repo-relative posix path of the module
    :ivar function: qualified name of the enclosing function
    :ivar line: line of the declaration
    :ivar first: first line of the wrapped expression
    :ivar last: last line of the wrapped expression
    :ivar kind: the declared kind, empty when not a literal
    :ivar reason: the reason, empty when not a literal
    :ivar ruling: Pace's ruling, empty when not a literal
    :ivar proof: ``<test file>::<test function>``, empty when not a literal
    :ivar speed: the speed test, empty when not given or not a literal
    :ivar until: the release a ``rolling_upgrade`` read is gone by, empty when not given or not a literal
    :ivar constants: each module constant the wrapped expression names and nothing else in the module does, with the
        lines of its value: its statement's hits are written there
    :ivar locals: each local of the enclosing function bound once and named only in the wrapped expression (the SQL
        built just before the call), with the lines of its value
    """

    path: str
    function: str
    line: int
    first: int
    last: int
    kind: str
    reason: str
    ruling: str
    proof: str
    speed: str = ""
    constants: tuple[tuple[str, int, int], ...] = ()
    locals: tuple[tuple[str, int, int], ...] = ()
    until: str = ""

    def covers(self, path: str, function: str, line: int) -> bool:
        """whether a site (a hit, a written statement) is the declared read: in the wrapped expression, or the value
        of a module constant only it names.

        :param path: the site's module
        :ptype path: str
        :param function: the site's enclosing function
        :ptype function: str
        :param line: the site's line
        :ptype line: int
        :return: whether it is
        :rtype: bool
        """
        if path != self.path:
            return False
        if function == self.function and (
            self.first <= line <= self.last or any(first <= line <= last for _, first, last in self.locals)
        ):
            return True
        return function == "<module>" and any(first <= line <= last for _, first, last in self.constants)


def declared_whole_read(path: str, function: str, node: ast.Call) -> DeclaredWhole:
    """a ``whole_read(<read>, kind=..., ...)`` declaration as written: the wrapped expression's lines and the literal
    arguments (one not a literal reads empty, and the gate refuses it). The one reading every gate judging a declared
    read uses (no_whole_reads; epoch_discipline's whole-reread). Spans of a module constant or a local the wrapped
    expression names are the caller's to add.

    :param path: repo-relative posix path of the module
    :ptype path: str
    :param function: qualified name of the enclosing function
    :ptype function: str
    :param node: the ``whole_read`` call
    :ptype node: ast.Call
    :return: the declaration
    :rtype: DeclaredWhole
    """
    wrapped = node.args[0] if node.args else None
    literal = {
        k.arg: k.value.value
        for k in node.keywords
        if k.arg and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)
    }
    return DeclaredWhole(
        path,
        function,
        node.lineno,
        wrapped.lineno if wrapped is not None else node.lineno,
        (wrapped.end_lineno or wrapped.lineno) if wrapped is not None else node.lineno,
        literal.get("kind", ""),
        literal.get("reason", ""),
        literal.get("ruling", ""),
        literal.get("proof", ""),
        literal.get("speed", ""),
        until=literal.get("until", ""),
    )


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
    :ivar receiver_class: the class the receiver is known to be (a parameter or variable annotated with it, a name
        or ``self`` attribute assigned one of it), empty when unknown
    :ivar whole_flags: the whole flags (``whole=True``) a bare or ``self`` call passes: each a finding unless the
        callee resolves to functions of the repo none of which reaches a whole read
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
    receiver_class: str = ""
    whole_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class WrittenStatement:
    """one SQL statement a module writes as text, where it writes it.

    :ivar path: repo-relative posix path of the module
    :ivar line: line of the string
    :ivar function: qualified name of the enclosing function, or ``<module>``
    :ivar text: the statement, each unknown interpolation as ``{}``
    :ivar reader: what the walk judged it against (the schema, the code's names, its transient relations)
    """

    path: str
    line: int
    function: str
    text: str
    reader: Reader


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
    :ivar statements: every SQL statement the walked modules write as text
    :ivar declarations: every read declared whole by nature (``whole_read``)
    :ivar bounded_findings: each failed check of a declared bounded producer (:mod:`.bounded`)
    """

    hits: list[WholeReadHit] = field(default_factory=list)
    files_scanned: int = 0
    definitions: set[str] = field(default_factory=set)
    unmatched_exclusions: list[str] = field(default_factory=list)
    operators: list[OperatorMark] = field(default_factory=list)
    operator_callers: list[str] = field(default_factory=list)
    constants: set[str] = field(default_factory=set)
    schema: SchemaCatalog = field(default_factory=SchemaCatalog)
    statements: list[WrittenStatement] = field(default_factory=list)
    declared_matched: set[str] = field(default_factory=set)
    declarations: list[DeclaredWhole] = field(default_factory=list)
    bounded_findings: tuple[str, ...] = ()


#: callees that only quote the name they are given: checked before a template function, since a quoter returning
#: ``'"' + name + '"'`` is one too, and its template (``"{}"``) would hide the name it quotes
_QUOTERS: Final[frozenset[str]] = frozenset({"quote_identifier", "_quote_ident", "quote_ident", "_quoted", "quoted"})


#: a name in a placeholder that says it is the key column(s): the statement reads by the table's key, whatever table
_KEY_NAME: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:^|_)(?:pk\w*|primary_keys?|keys?|key_?col\w*|key_?columns?|key_?names?|id_?col\w*)$"
)


#: a local copy's relation accessors: ``snapshot.relation(NAME)`` names the copy's own table, never the L3 one
_COPY_CALLS: Final[frozenset[str]] = COPY_CALLS


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


def _placeholder(value: ast.expr, constants: dict[str, str], produced: Produced = None) -> str:
    """what an f-string placeholder reads as: a constant's text (``{_MARK_TABLE}``, an imported one, or one handed
    to a call: ``{quote(TABLE)}``), ``copy.<name>`` for a local copy's relation (``{snapshot.relation(GEOS)}``),
    ``{pk}`` when a name in it says it is the key (``{pk}``, ``{self._key_column}``, ``{", ".join(pk_cols)}``), else
    ``{}``.

    :param value: the placeholder's expression
    :ptype value: ast.expr
    :param constants: the module's names bound to strings
    :ptype constants: dict[str, str]
    :param produced: what an expression reads as when it is a declared transient producer's return
        (:data:`TRANSIENT_PRODUCED`) or a keyed helper's relation parameter (:data:`HELPER_RELATION`)
    :ptype produced: Produced
    :return: the text
    :rtype: str
    """
    named = produced(value) if produced is not None else None
    if named is not None:
        return named
    known = _constant_text(value, constants)
    if known is not None:
        return known
    if isinstance(value, ast.Call) and is_keyed_fragment(value):
        return _KEYED_FRAGMENT_TEXT
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


#: 3tears' keyed-helper connections over a local database (DuckDB in process): a helper handed one reads its tables
_LOCAL_CONNECTIONS: Final[frozenset[str]] = frozenset({"InlineDuckDB"})

#: 3tears' keyed-helper connections over a database in this process (DuckDB, SQLite), which hashes no index: a helper
#: handed one is asked no ``index-hashed-range`` (a SQLite L1 table's ``delete_oldest(InlineSQLite(l1), ...)``)
_IN_PROCESS_CONNECTIONS: Final[frozenset[str]] = _LOCAL_CONNECTIONS | {"InlineSQLite"}


def _on_a_local_database(call: ast.Call) -> bool:
    """whether a helper call's connection is one over a database in this process (``InlineSQLite(conn)``).

    :param call: the helper call
    :ptype call: ast.Call
    :return: whether it is
    :rtype: bool
    """
    return bool(call.args) and isinstance(call.args[0], ast.Call) and _final(call.args[0]) in _IN_PROCESS_CONNECTIONS


def _target_binds(target: ast.expr, name: str) -> bool:
    """whether an assignment target binds ``name`` (itself, or within a tuple or list it unpacks).

    :param target: the target
    :ptype target: ast.expr
    :param name: the name
    :ptype name: str
    :return: whether it does
    :rtype: bool
    """
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(target))


def _binds_in(node: ast.For | ast.AsyncFor | ast.With | ast.AsyncWith, name: str) -> bool:
    """whether a loop's target or a ``with``'s ``as`` binds ``name``.

    :param node: the statement
    :ptype node: ast.For | ast.AsyncFor | ast.With | ast.AsyncWith
    :param name: the name
    :ptype name: str
    :return: whether it does
    :rtype: bool
    """
    if isinstance(node, ast.For | ast.AsyncFor):
        return _target_binds(node.target, name)
    return any(item.optional_vars is not None and _target_binds(item.optional_vars, name) for item in node.items)


#: how many bindings deep a family name is followed through a function's own names (``_declared_calls``)
_INLINE_DEPTH: Final[int] = 3


#: the most scopes a function's if/else branch bindings are judged in
_MOST_BRANCH_ENVS: Final[int] = 8


def _branch_binding(body: list[ast.stmt]) -> dict[str, ast.expr]:
    """the names a branch binds by a plain assignment among its own statements (not nested ones), onto the value.

    :param body: the branch's statements
    :ptype body: list[ast.stmt]
    :return: name onto value
    :rtype: dict[str, ast.expr]
    """
    return {
        statement.targets[0].id: statement.value
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }


def _joined_statements(node: ast.AST | None) -> bool:
    """whether ``node`` joins the statements a comprehension writes with a literal separator (``" UNION ALL ".join(
    f"SELECT ..." for t in held)``): a body built of one statement's parts, each judged where it is written.

    :param node: the expression, or None
    :ptype node: ast.AST | None
    :return: whether it is
    :rtype: bool
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], (ast.GeneratorExp, ast.ListComp))
    ):
        return False
    part = _string_text(node.args[0].elt, {})
    return part is not None and is_sql(part)


def _string_text(node: ast.expr, env: dict[str, str], produced: Produced = None) -> str | None:
    """the statically known text of a string expression.

    :param node: the expression
    :ptype node: ast.expr
    :param env: names bound to strings in scope
    :ptype env: dict[str, str]
    :param produced: what an expression reads as when it is a declared transient producer's return
        (:data:`TRANSIENT_PRODUCED`, so a name bound to one is too) or a keyed helper's relation parameter
        (:data:`HELPER_RELATION`)
    :ptype produced: Produced
    :return: the text, each unknown interpolation as ``{}``; None when it is not a string expression
    :rtype: str | None
    """
    text: str | None = None
    named = produced(node) if produced is not None and isinstance(node, (ast.Call, ast.Attribute)) else None
    if named is not None:
        text = named
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
    elif isinstance(node, ast.JoinedStr):
        text = "".join(
            str(part.value)
            if isinstance(part, ast.Constant)
            else _placeholder(part.value, env, produced)
            if isinstance(part, ast.FormattedValue)
            else "{}"
            for part in node.values
        )
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_text(node.left, env, produced)
        right = _string_text(node.right, env, produced)
        if left is not None or right is not None:
            text = (left if left is not None else "{}") + (right if right is not None else "{}")
    elif isinstance(node, ast.Name):
        text = env.get(node.id)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"format", "strip"}:
        text = _string_text(node.func.value, env, produced)
    elif isinstance(node, ast.Call) and node.args and _final(node) in {"dedent", "SQL", "text"}:
        text = _string_text(node.args[0], env, produced)
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


def _definitions(tree: ast.Module) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.ClassDef | None]]:
    """the module's functions and its top-level classes' methods, each with its class (None for a function).

    :param tree: the module
    :ptype tree: ast.Module
    :return: each definition and its class
    :rtype: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.ClassDef | None]]
    """
    found: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.ClassDef | None]] = []
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append((statement, None))
        elif isinstance(statement, ast.ClassDef):
            found += [
                (member, statement)
                for member in statement.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
    return found


def _binds(function: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """whether a function binds ``name`` itself: a parameter, or a store in its own body.

    :param function: the function
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param name: the name
    :ptype name: str
    :return: whether it does
    :rtype: bool
    """
    arguments = function.args
    if name in {a.arg for a in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]}:
        return True
    return any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name
        for node in _own_nodes(function.body)
    )


def _family_folded(hits: dict[str, str], members: int) -> dict[str, str]:
    """a call judged once per declared value (``declared_parameters``) with one kind of finding on every value: that
    kind once, naming the family, not once per member (a whole read of each of a copy's tables is one finding of
    the read); a kind on only some members is kept per member, each named.

    :param hits: symbol onto its statement
    :ptype hits: dict[str, str]
    :param members: how many calls the declared values made
    :ptype members: int
    :return: the hits, folded
    :rtype: dict[str, str]
    """
    if members < 2:
        return hits
    kinds: dict[str, list[str]] = {}
    for symbol in hits:
        kinds.setdefault(symbol.split(" ", 1)[0], []).append(symbol)
    folded: dict[str, str] = {}
    for kind, symbols in kinds.items():
        if len(symbols) >= members:
            folded[f"{kind} <each-declared-value>"] = f"{hits[symbols[0]]} (and each of {len(symbols)} declared values)"
        else:
            folded.update({symbol: hits[symbol] for symbol in symbols})
    return folded


def _local_relation(call: ast.Call, declared: frozenset[str]) -> ast.Call:
    """a helper call on a local database's connection (``InlineDuckDB(cursor)``) naming a table a declared parameter
    holds (``declared_parameters``: a copy's table) bare, written as the table of that database's ``main`` schema: a
    local copy's relation, never the L3 table of the same name. Any other bare name (a scratch relation the code
    registers) is left as it is.

    :param call: the helper call, its arguments resolved
    :ptype call: ast.Call
    :param declared: every value a declared parameter holds
    :ptype declared: frozenset[str]
    :return: the call, its relation local where it is
    :rtype: ast.Call
    """
    args = list(call.args)
    if not (
        len(args) > 1
        and isinstance(args[0], ast.Call)
        and _final(args[0]) in _LOCAL_CONNECTIONS
        and isinstance(args[1], ast.Constant)
        and isinstance(args[1].value, str)
        and args[1].value in declared
    ):
        return call
    args[1] = ast.Constant(f"main.{args[1].value}")
    return ast.Call(func=call.func, args=args, keywords=call.keywords)


def _declared_of(
    function: ast.FunctionDef | ast.AsyncFunctionDef, qualified: str, declared: Mapping[str, tuple[str, ...]]
) -> dict[str, tuple[list[Passed], str]]:
    """the function's parameters the repo declares the values of (``declared_parameters``, keyed
    ``<qualified function>.<parameter>``), each onto its values and whether it holds one of them (a ``str`` or
    unannotated parameter) or a collection of them (a loop over it binds each).

    :param function: the function
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param qualified: its qualified name (``Class.method``)
    :ptype qualified: str
    :param declared: the repo's declarations
    :ptype declared: Mapping[str, tuple[str, ...]]
    :return: name onto its values and how it holds them (``one``, ``many``, ``loop``)
    :rtype: dict[str, tuple[list[Passed], str]]
    """
    if not declared:
        return {}
    found: dict[str, tuple[list[Passed], str]] = {}
    arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    for argument in arguments:
        values = declared.get(f"{qualified}.{argument.arg}")
        if values is None:
            continue
        annotation = argument.annotation
        one = annotation is None or (isinstance(annotation, ast.Name) and annotation.id == "str")
        found[argument.arg] = (list(values), "one" if one else "many")
    # a name a loop of the function binds (``for table, pieces in small.items()``): each value, in every loop
    for node in _own_nodes(function.body):
        target = node.target if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) else None
        if isinstance(target, ast.Tuple) and target.elts:
            target = target.elts[0]
        if isinstance(target, ast.Name) and target.id not in found:
            values = declared.get(f"{qualified}.{target.id}")
            if values is not None:
                found[target.id] = (list(values), "loop")
    return found


def _lookup_functions(
    tree: ast.Module, module_values: Mapping[str, ast.expr], constants: dict[str, str]
) -> dict[str, tuple[dict[str, str], str | None]]:
    """the module's top-level lookup functions: one parameter, and one statement after any docstring returning a
    module mapping of strings looked up by it (``return _STATE_COLUMNS.get(table, STATE_COLUMN)``, ``return
    _COLUMNS[table]``), each onto the mapping and its default (None for a subscript).

    :param tree: the module
    :ptype tree: ast.Module
    :param module_values: the module's names onto what its top level binds them to
    :ptype module_values: Mapping[str, ast.expr]
    :param constants: the module's names bound to strings
    :ptype constants: dict[str, str]
    :return: name onto mapping and default
    :rtype: dict[str, tuple[dict[str, str], str | None]]
    """
    found: dict[str, tuple[dict[str, str], str | None]] = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or len(function.args.args) != 1 or function.args.kwonlyargs:
            continue
        body = [
            statement
            for statement in function.body
            if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
        ]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            continue
        parameter = function.args.args[0].arg
        value = body[0].value
        mapping: ast.expr | None = None
        default: str | None = None
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get"
            and len(value.args) == 2
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == parameter
        ):
            mapping = value.func.value
            default = _constant_text(value.args[1], constants)
            if default is None:
                continue
        elif isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Name) and value.slice.id == parameter:
            mapping = value.value
        held = module_values.get(mapping.id) if isinstance(mapping, ast.Name) else None
        if (
            isinstance(held, ast.Call)
            and isinstance(held.func, ast.Name)
            and held.func.id in {"MappingProxyType", "dict"}
        ):
            held = held.args[0] if len(held.args) == 1 else None
        if not isinstance(held, ast.Dict):
            continue
        pairs = [
            (None if key is None else _constant_text(key, constants), _constant_text(item, constants))
            for key, item in zip(held.keys, held.values, strict=True)
        ]
        if any(key is None or item is None for key, item in pairs):
            continue
        found[function.name] = ({key: item for key, item in pairs if key is not None and item is not None}, default)
    return found


def _relation_wrappers(tree: ast.Module) -> frozenset[str]:
    """the module's top-level functions of one parameter that hand a step that parameter as its relation: every
    ``return`` is ``Quoted(<the parameter>)`` or a ``Derived(...)`` over nothing but it (a snapshot's relation, or the
    parenthesized select a limited snapshot reads it through, made a step's relation). A call of one reads as
    ``Quoted(<its argument>)``: the relation named, never the narrower derived select trusted.

    :param tree: the module
    :ptype tree: ast.Module
    :return: the functions' names
    :rtype: frozenset[str]
    """
    found: set[str] = set()
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = function.args
        if len(arguments.args) != 1 or arguments.posonlyargs or arguments.kwonlyargs or arguments.vararg:
            continue
        parameter = arguments.args[0].arg
        returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
        quoted = False
        for returned in returns:
            value = returned.value
            if not (isinstance(value, ast.Call) and _final(value) in {"Quoted", "Derived"} and len(value.args) == 1):
                break
            names = {node.id for node in ast.walk(value.args[0]) if isinstance(node, ast.Name)}
            if names != {parameter} or value.keywords:
                break
            quoted = quoted or _final(value) == "Quoted"
        else:
            if returns and quoted:
                found.add(function.name)
    return frozenset(found)


def _bound_argument(call: ast.Call, name: str, position: int | None) -> ast.expr | None:
    """the argument a call binds a parameter to: by its keyword, else at its position (none past a ``*args``).

    :param call: the call
    :ptype call: ast.Call
    :param name: the parameter
    :ptype name: str
    :param position: its position among the call's positional arguments, None for keyword-only
    :ptype position: int | None
    :return: the argument, or None where the call binds none the walk can see
    :rtype: ast.expr | None
    """
    for given in call.keywords:
        if given.arg == name:
            return given.value
    if position is None or any(isinstance(arg, ast.Starred) for arg in call.args[: position + 1]):
        return None
    return call.args[position] if len(call.args) > position else None


def _helper_relations(tree: ast.Module) -> tuple[dict[int, frozenset[str]], dict[int, frozenset[str]]]:
    """the parameters (per function) and fields (per class) of a :data:`_KEYED_HELPER_MODULES` module that hold a keyed
    helper's relation parameter: every parameter of a helper in :data:`KEYED_HELPER_CALLS` or :data:`STEP_CALLS`
    (its call sites judge the relation each hands it); a parameter of a private function or method, or a field of a
    private class, only where it has in-module callers (constructions) and every one hands it straight (a bare name)
    from a parameter or field that holds one, as the walk reads them, never as asserted.

    :param tree: the module
    :ptype tree: ast.Module
    :return: function node id onto its parameters, class node id onto its fields
    :rtype: tuple[dict[int, frozenset[str]], dict[int, frozenset[str]]]
    """
    definitions = _definitions(tree)
    roots = KEYED_HELPER_CALLS | STEP_CALLS
    parameters: dict[int, frozenset[str]] = {}
    fields: dict[int, frozenset[str]] = {}
    for function, owner in definitions:
        if owner is None and function.name in roots:
            arguments = function.args
            parameters[id(function)] = frozenset(
                argument.arg for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
            )
    calls = [
        (function, owner, node)
        for function, owner in definitions
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]

    def holds(argument: ast.expr | None, function: ast.AST, owner: ast.ClassDef | None) -> bool:
        if isinstance(argument, ast.Name):
            return argument.id in parameters.get(id(function), frozenset())
        return (
            isinstance(argument, ast.Attribute)
            and isinstance(argument.value, ast.Name)
            and argument.value.id == "self"
            and owner is not None
            and argument.attr in fields.get(id(owner), frozenset())
        )

    def handed(
        sites: Sequence[tuple[ast.AST, ast.ClassDef | None, ast.Call]], names: Sequence[tuple[str, int | None]]
    ) -> set[str]:
        return {
            name
            for name, position in names
            if all(holds(_bound_argument(call, name, position), caller, owner) for caller, owner, call in sites)
        }

    classes = [
        statement for statement in tree.body if isinstance(statement, ast.ClassDef) and statement.name.startswith("_")
    ]
    changed = True
    while changed:
        changed = False
        for function, owner in definitions:
            if owner is None and function.name in roots:
                continue
            if not (function.name.startswith("_") or (owner is not None and owner.name.startswith("_"))):
                continue
            sites = [
                (caller, caller_owner, call)
                for caller, caller_owner, call in calls
                if (owner is None and isinstance(call.func, ast.Name) and call.func.id == function.name)
                or (owner is not None and isinstance(call.func, ast.Attribute) and call.func.attr == function.name)
            ]
            if not sites:
                continue
            arguments = function.args
            positional = [*arguments.posonlyargs, *arguments.args][1 if owner is not None else 0 :]
            names: list[tuple[str, int | None]] = [(argument.arg, index) for index, argument in enumerate(positional)]
            names += [(argument.arg, None) for argument in arguments.kwonlyargs]
            found = frozenset(handed(sites, names)) | parameters.get(id(function), frozenset())
            if found != parameters.get(id(function), frozenset()):
                parameters[id(function)] = found
                changed = True
        for cls in classes:
            sites = [
                (caller, caller_owner, call)
                for caller, caller_owner, call in calls
                if isinstance(call.func, ast.Name) and call.func.id == cls.name
            ]
            if not sites:
                continue
            declared = [
                statement.target.id
                for statement in cls.body
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
            ]
            found = frozenset(handed(sites, [(name, index) for index, name in enumerate(declared)]))
            found |= fields.get(id(cls), frozenset())
            if found != fields.get(id(cls), frozenset()):
                fields[id(cls)] = found
                changed = True
    return parameters, fields


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
        producers: frozenset[str] = frozenset(),
        parameter_values: dict[tuple[str, str], list[Passed]] | None = None,
        parameter_sites: dict[str, list[Site]] | None = None,
        view_producers: Mapping[str, tuple[tuple[str, ...], ...]] | None = None,
        tuples: dict[str, list[Passed]] | None = None,
        relation_producers: frozenset[str] = frozenset(),
        match_builders: Mapping[str, MatchBuilder] | None = None,
        forwarders: RepoForwarders | None = None,
        builders: frozenset[str] = frozenset(),
        never_none: frozenset[tuple[str, str]] = frozenset(),
        declared: Mapping[str, tuple[str, ...]] | None = None,
        bounded: Bounded | None = None,
        sentinels: SentinelTemplates | None = None,
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
        :param producers: the repo's functions whose returned relation is a change set (``Stages._stage``,
            ``_Work.put``): ``Class.method``, or a module function's name
        :ptype producers: frozenset[str]
        :param parameter_values: ``(function, parameter)`` onto what each call site passes it, None where the walk
            cannot read it (:func:`_parameter_values`)
        :ptype parameter_values: dict[tuple[str, str], list[Passed]] | None
        :param parameter_sites: each function's call sites, what each passes each parameter (:func:`_parameter_values`)
        :ptype parameter_sites: dict[str, list[Site]] | None
        :param tuples: the module tuples this module reads (its own and the ones it imports), onto their elements
            (:func:`_module_tuples`)
        :ptype tuples: dict[str, list[Passed]] | None
        :param view_producers: each function that builds a view under a name it is handed onto the keys its views
            declare
        :ptype view_producers: Mapping[str, tuple[tuple[str, ...], ...]] | None
        :param relation_producers: the declared relation producers (``NoWholeReadsConfig.relation_producers``)
        :ptype relation_producers: frozenset[str]
        :param match_builders: the repo's functions returning a step's ``Match`` list (:func:`_match_builders`)
        :ptype match_builders: Mapping[str, MatchBuilder] | None
        :param forwarders: the repo's public forwarders, followed across modules; None follows only each module's
            private ones
        :ptype forwarders: RepoForwarders | None
        :param builders: the repo's step builders (:func:`~threetears.enforcement.no_whole_reads.flow.step_builders`)
        :ptype builders: frozenset[str]
        :param never_none: ``(function, parameter)`` every call site passes something that is never None
            (:func:`_never_none_parameters`)
        :ptype never_none: frozenset[tuple[str, str]]
        :param bounded: what a copy accessor's relation reads as under a checked bounded producer
        :ptype bounded: Bounded | None
        :return: nothing
        :rtype: None
        """
        self.path = path
        self._bounded = bounded if bounded is not None else Bounded()
        self._never_none = never_none
        self._match_builders = dict(match_builders or {})
        #: the names the module's imports bind: ``x.read_by()`` on any other name is a method of an object
        self._module_names = _imported_names(tree)
        self._parameter_values = parameter_values or {}
        self._parameter_sites = parameter_sites or {}
        self._tuples = tuples or {}
        #: ``<qualified function>.<parameter>`` onto the values the repo declares it holds (``declared_parameters``)
        self._declared = dict(declared or {})
        self._declared_values = frozenset(value for values in self._declared.values() for value in values)
        #: the declarations a function of this module has the parameter of
        self.declared_matched: set[str] = set()
        #: the module's lookup functions (:func:`_lookup_functions`), read once its strings are known
        self._lookups: dict[str, tuple[dict[str, str], str | None]] | None = None
        #: each nested function onto the function it is nested in, read once
        self._enclosing: dict[int, ast.FunctionDef | ast.AsyncFunctionDef] | None = None
        self.source = source
        self.hits: list[WholeReadHit] = []
        #: every SQL statement the module writes as text, as the walk reads it
        self.statements: list[WrittenStatement] = []
        self.calls: list[_CallSite] = []
        self.definitions: set[str] = set()
        self.module_functions: set[str] = set()
        #: the module's top-level classes: a call to one constructs it, reaching only its ``__init__``
        self.module_classes: set[str] = {
            statement.name for statement in tree.body if isinstance(statement, ast.ClassDef)
        }
        self.operators: list[OperatorMark] = []
        self.declarations: list[DeclaredWhole] = []
        self._tree = tree
        #: each module constant bound to a string expression once, onto its value's lines (for a declaration)
        self._string_constants: dict[str, tuple[int, int]] | None = None
        #: a name imported into the module, onto ``<module>.<name>`` it came from
        self.imports: dict[str, str] = {}
        self._pagers = pagers
        self._always_whole = always_whole
        self._narrowing = narrowing
        self._transient = transient
        #: each declared producer as ``(class, method)``, the class empty for a module function
        self._view_producers = dict(view_producers or {})
        #: each declared relation producer as ``(class, name, entry)``: a function (the class empty for a module
        #: function) or a class's attribute, whose run-time relation is judged as :func:`produced_relation` of it
        self._relation_producers: tuple[tuple[str, str, str], ...] = tuple(
            (owner.rsplit(".", 1)[-1], member, entry)
            for entry in sorted(relation_producers)
            for owner, _, member in (entry.rpartition("."),)
        )
        self._producers: tuple[tuple[str, str], ...] = tuple(
            (owner.rsplit(".", 1)[-1], method)
            for owner, _, method in (producer.rpartition(".") for producer in sorted(producers))
        )
        #: per class, the ``self`` attributes every assignment of which is a producer's return
        self._produced_attributes: dict[int, frozenset[str]] = {}
        #: the module's transient producers' returns as written (:meth:`_producer_spellings`), read once
        self._spellings: frozenset[str] | None = None
        self._tree = tree
        #: the module's templates over a declared closed map, judged once per entry (:mod:`.sentinels`)
        self._sentinels = sentinels
        self._verify = verify
        self._exec_calls = _EXEC_CALLS | exec_calls
        self._cte_arguments = cte_arguments or {}
        self._schema = schema if schema is not None else SchemaCatalog()
        #: module and class constants bound to strings, ``<path>::<qualified name>``
        self.constants: set[str] = set()
        self._loops: list[set[str]] = []
        #: per enclosing function, the CTE names its statements define
        self._ctes: list[frozenset[str]] = []
        #: per definition, the CTE names the statements it takes part in define (:func:`_statement_ctes`)
        self._statement_ctes: dict[int, frozenset[str]] = {}
        self._function_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        #: per enclosing function, the terms a ``source_change_read`` mark names: stamps, ids, a whole static unit
        self._source: list[tuple[frozenset[str], frozenset[str], bool]] = []
        self.stubs: set[str] = set()
        self._stack: list[str] = []
        self._classes: list[str] = []
        self._class_nodes: list[ast.ClassDef] = []
        self._envs: list[dict[str, str]] = [{}]
        #: per function on the stack, the scopes of its if/else branch bindings (:meth:`_branch_envs`)
        self._branch_scopes: list[list[dict[str, str]]] = [[]]
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
            and _final(node) in {"register", "change_relation"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            # a change relation declared over a data table is flagged where it is declared, never made a change set
            and not (_final(node) == "change_relation" and self._schema.table(node.args[0].value) is not None)
        }
        # a temp view over only the rows being written (``CREATE TEMP VIEW typed AS SELECT ... FROM raw``) is those rows
        self._registered |= _transient_views(tree, self._registered, transient)
        #: the module's sort vocabularies (``_SORT_COLUMNS: Final = {Field.NAME: SortColumn(column=..., index=...)}``)
        self._sort_mappings: dict[str, list[ast.Call]] = _sort_mappings(tree)
        #: each module function onto the vocabularies it, or a function it calls, uses
        self._mapping_users: dict[str, set[str]] = _mapping_users(tree, set(self._sort_mappings))
        #: each top-level name's value, for a helper argument that is a module constant of its shape
        self._module_values: dict[str, ast.expr] = {
            statement.targets[0].id: statement.value
            for statement in tree.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        } | {
            statement.target.id: statement.value
            for statement in tree.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        }
        #: in a module of :data:`_KEYED_HELPER_MODULES`, the parameters and fields that hold a keyed helper's relation
        #: parameter (:func:`_helper_relations`): a statement over one is judged at the helper's call sites
        self._helper_parameters, self._helper_fields = (
            _helper_relations(tree) if any(module in str(path) for module in _KEYED_HELPER_MODULES) else ({}, {})
        )
        #: the module's functions that hand a step their one parameter as its relation (:func:`_relation_wrappers`)
        self._relation_wrappers = _relation_wrappers(tree)
        #: where each relation a step names comes from (:mod:`~threetears.enforcement.no_whole_reads.flow`): a change
        #: relation through a name, an attribute or a container; a forwarder's parameter, judged at its call sites
        self._flow = RelationFlow(
            tree,
            keyed_step=self._keyed_step,
            produces=self._produces,
            produced=self._attributes_produced,
            producer=lambda function, owner: (
                (owner.name if owner is not None else "", function.name) in self._producers
            ),
            path=path,
            imports=self.imports,
            repo=forwarders,
            builders=builders,
        )
        self._envs[0].update(imported or {})
        self._envs[0].update(_module_strings(tree))
        self._bind(tree.body, self._envs[0])
        self._constants_of(tree.body, "")
        self._statement_ctes = _statement_ctes(tree, self._envs[0])

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

    def _bind(self, body: list[ast.stmt], env: dict[str, str], fixed: Mapping[str, str] | None = None) -> None:
        """record the string each simple assignment under ``body`` binds (not in nested functions). A name the body
        binds more than once (``where = ""`` then ``where = "WHERE ..."``, or ``where += ...``) has no one text, so it
        is left unbound: a statement built from it reads it as an interpolation the walk cannot see. A ``fixed`` name
        is bound to its given text throughout (one branch's binding: :meth:`_branch_envs`).

        :param body: statements
        :ptype body: list[ast.stmt]
        :param env: name onto bound string, updated in place
        :ptype env: dict[str, str]
        :param fixed: names onto the text each is read as, whatever the body binds it to
        :ptype fixed: Mapping[str, str] | None
        :return: nothing
        :rtype: None
        """
        fixed = fixed or {}
        env.update(fixed)
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
                and (_final(node.value) in _FILTER_BUILDERS | KEYED_FRAGMENT_BUILDERS)
            ):
                built = _KEYED_FRAGMENT_TEXT if is_keyed_fragment(node.value) else _filter_text(node.value)
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
            if isinstance(target, ast.Name) and target.id in fixed:
                continue
            if isinstance(target, ast.Name) and value is not None:
                text = _string_text(self._view_part(value), env, self._produced_by)
                if text is not None and stores.get(target.id, 0) == 1:
                    env[target.id] = text
                else:
                    env.pop(target.id, None)

    def _view_part(self, value: ast.expr) -> ast.expr:
        """in a declared view producer, the statement a body joined from one statement per table is written as
        (``union = " UNION ALL ".join(f"SELECT ... FROM {t}" for t in held)``: the f-string); else ``value``.

        :param value: an assignment's value
        :ptype value: ast.expr
        :return: the expression its text is read from
        :rtype: ast.expr
        """
        if self._function in self._view_producers and _joined_statements(value):
            assert isinstance(value, ast.Call) and isinstance(value.args[0], (ast.GeneratorExp, ast.ListComp))
            return value.args[0].elt
        return value

    def _branch_envs(self, body: list[ast.stmt], env: dict[str, str]) -> list[dict[str, str]]:
        """the scopes a function's statements are also judged in: one per branch choice of each name it binds once in
        each branch of an ``if``/``else`` (``select = f"{a}t.state{b}"`` / ``else: select = f"{a}t.state{c}"``), its
        known text in each, so a statement built from it is judged as each branch writes it (at most
        :data:`_MOST_BRANCH_ENVS`); empty when it binds none.

        :param body: the function's statements
        :ptype body: list[ast.stmt]
        :param env: the scope the function's own bindings start from
        :ptype env: dict[str, str]
        :return: the scopes, the function's own bindings read in each
        :rtype: list[dict[str, str]]
        """
        stores: dict[str, int] = {}
        for node in _own_nodes(body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                stores[node.id] = stores.get(node.id, 0) + 1
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                stores[node.target.id] = stores.get(node.target.id, 0) + 2
        choices: dict[str, list[ast.expr]] = {}
        for node in _own_nodes(body):
            if not isinstance(node, ast.If) or not node.orelse:
                continue
            for name in {*_branch_binding(node.body)} & {*_branch_binding(node.orelse)}:
                if stores.get(name) == 2 and name not in choices:
                    choices[name] = [_branch_binding(node.body)[name], _branch_binding(node.orelse)[name]]
        envs: list[dict[str, str]] = []
        names = sorted(choices)
        for picked in itertools.product(*(choices[name] for name in names)):
            scope = dict(env)
            texts = {name: _string_text(value, scope, self._produced_by) for name, value in zip(names, picked)}
            if any(text is None for text in texts.values()):
                continue
            self._bind(body, scope, fixed={name: text for name, text in texts.items() if text is not None})
            envs.append(scope)
            if len(envs) >= _MOST_BRANCH_ENVS:
                break
        return envs

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
        self._class_nodes.append(node)
        self.definitions.add(f"{self.path}::{self._function}")
        self.generic_visit(node)
        self._class_nodes.pop()
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
        self.declared_matched.update(
            f"{self._function}.{name}" for name in _declared_of(node, self._function, self._declared)
        )
        # a source change read's terms hold in the function and every function nested in it
        self._source.append(self._source[-1] if self._source else _NO_SOURCE)
        self._mark_operator(node)
        if _is_stub(node):
            self.stubs.add(f"{self.path}::{self._function}")
        # the function is on the stack before its bindings are read: a name bound to a producer's return is known by
        # its receiver's class, which the function's parameters and assignments say
        self._function_nodes.append(node)
        env = dict(self._envs[-1])
        branches = self._branch_envs(node.body, env)
        self._bind(node.body, env)
        self._envs.append(env)
        self._branch_scopes.append(branches)
        self._ctes.append(_function_ctes(node.body))
        owner = self._classes[-1] if self._classes and self._function.startswith(f"{self._classes[-1]}.") else ""
        self._classes.append(owner)
        self.generic_visit(node)
        self._classes.pop()
        self._envs.pop()
        self._branch_scopes.pop()
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
        # in a declared view producer, a statement a comprehension writes for a joined body (``" UNION ALL ".join(
        # f"SELECT ..." for t in held)``) is a part of the view it builds, judged there by the producer's keys
        part = (
            self._function in self._view_producers
            and isinstance(parent, (ast.GeneratorExp, ast.ListComp))
            and parent.elt is node
            and _joined_statements(self._parents.get(parent))
        )
        return not piece and not prose and not part

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
        # a statement interpolating a module constant (``{_MEMORY_MARK_COLUMNS}``) is written in it too, so an index's
        # purpose may name that constant once rather than every function placing it
        if isinstance(node, ast.JoinedStr):
            names.update(
                f"{self.path}::{part.value.id}"
                for part in node.values
                if isinstance(part, ast.FormattedValue)
                and isinstance(part.value, ast.Name)
                and part.value.id in self._module_values
            )
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
            view_keys=self._view_producers.get(self._function, ()),
            local=_run_locally(self._parents.get(node)),
            transient=self._transient_table,
            ctes=(self._ctes[-1] if self._ctes else frozenset())
            | (
                self._statement_ctes.get(id(self._function_nodes[0]), frozenset())
                if self._function_nodes
                else frozenset()
            )
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
        self._record_statement(node)
        if isinstance(node, ast.JoinedStr) and self._maximal_string(node) and self._keyset_listing(node):
            return
        # a placeholder resolves to a module constant, or to a string the enclosing function binds once; a parameter
        # every caller fills with a CTE its own statement defines is that CTE (judged where the caller builds it)
        function = self._function_node
        cte_params = self._cte_parameters(function) if function is not None else set()
        judged: set[tuple[str, str]] = set()
        # the function's scope, or one per if/else branch choice when it has them (a statement built from a name each
        # branch binds is judged as each writes it; every other name is bound in each as in the function's); a finding
        # counts once
        for scope in self._branch_scopes[-1] or [self._envs[-1]]:
            env = {**scope, **{name: f"__cte_{name}" for name in cte_params}} if cte_params else scope
            text = _string_text(node, env, self._produced_by) if self._maximal_string(node) else None
            for each in self._instances(node, text):
                self._judge_text(node, each, judged)

    def _instances(self, node: ast.expr, text: str | None) -> list[str | None]:
        """``text`` as it is judged: once per entry of the closed map a declared template ranges over
        (:mod:`.sentinels`), else as written.

        :param node: the string expression
        :ptype node: ast.expr
        :param text: its text in the scope judged, or None
        :ptype text: str | None
        :return: each text to judge
        :rtype: list[str | None]
        """
        if self._sentinels is None or text is None or self._sentinels.table not in text:
            return [text]
        child: ast.AST = node
        parent = self._parents.get(child)
        while parent is not None and not isinstance(parent, ast.Module):
            child, parent = parent, self._parents.get(parent)
        target = child.targets[0] if isinstance(child, ast.Assign) and len(child.targets) == 1 else None
        target = child.target if isinstance(child, ast.AnnAssign) else target
        if not (isinstance(target, ast.Name) and target.id in self._sentinels.templates):
            return [text]
        made = self._sentinels.instances(text)
        return [text] if made is None else list(made)

    def _record_statement(self, node: ast.expr) -> None:
        """record the SQL statement a maximal string expression writes, as the function's own scope reads it (the
        first branch scope when it has them), for the rule that statements are written by the helpers
        (:mod:`~threetears.enforcement.no_whole_reads.use_the_helper`).

        :param node: the expression
        :ptype node: ast.expr
        :return: nothing
        :rtype: None
        """
        if not self._maximal_string(node):
            return
        scope = (self._branch_scopes[-1] or [self._envs[-1]])[0]
        text = _string_text(node, scope, self._produced_by)
        if text is not None and is_sql(text):
            self.statements.append(
                WrittenStatement(self.path, getattr(node, "lineno", 0), self._function, text, self._reader(node))
            )

    def _judge_text(self, node: ast.expr, text: str | None, judged: set[tuple[str, str]]) -> None:
        """record the findings of ``node``'s statement as one scope reads it, each finding once.

        :param node: the expression
        :ptype node: ast.expr
        :param text: its text in that scope, or None
        :ptype text: str | None
        :param judged: the findings already recorded for ``node``, as (kind, table); updated in place
        :ptype judged: set[tuple[str, str]]
        :return: nothing
        :rtype: None
        """
        for finding in judge_sql(text, self._reader(node)) if text is not None and is_sql(text) else []:
            if (finding.kind, finding.table) in judged:
                continue
            judged.add((finding.kind, finding.table))
            if self._transient_table(finding.table):
                continue
            parent = self._parents.get(node)
            if finding.kind == KEYED_COUNT:
                # a count bound by a key prefix or a bound id set reads its key's rows: no verification scan
                continue
            if finding.kind != HEAD:
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
        text = _string_text(node, self._envs[-1], self._produced_by) or ""
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
                if condition == _KEYED_FRAGMENT_TEXT or (
                    condition is not None and _keyed_exists(condition, table, reader)
                ):
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
        """whether a relation is a change set (a registered relation, or one the repo names transient), or a keyed
        helper's relation parameter (:data:`HELPER_RELATION`, judged at the helper's call sites), by its resolved name
        (a constant's text, a local copy's relation without its ``copy.``).

        :param table: the relation as written, its constants resolved
        :ptype table: str
        :return: whether it is no data table
        :rtype: bool
        """
        bare = table.strip('"').removeprefix(COPY_PREFIX)
        return (
            bare in {TRANSIENT_PRODUCED, HELPER_RELATION}
            or bare in self._registered
            or (self._transient is not None and self._transient.search(bare) is not None)
        )

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
        if name == DECLARATION_CALL:
            self._declare(node)
        if name is not None:
            self._judge_call(node, name)
            self._judge_helper(node)
            self._judge_change_relation(node)
            self._judge_handed(node)
        self.generic_visit(node)

    def _declare(self, node: ast.Call) -> None:
        """record a read declared whole by nature: ``whole_read(<read>, kind=..., reason=..., ruling=..., proof=...,
        speed=...)``. Its arguments are read as written (literals); the gate judges them.

        :param node: the declaration
        :ptype node: ast.Call
        :return: nothing
        :rtype: None
        """
        wrapped = node.args[0] if node.args else None
        if self._string_constants is None:
            self._string_constants = module_constant_spans(self._tree)
        constants: list[tuple[str, int, int]] = []
        if wrapped is not None:
            inside = [n.id for n in ast.walk(wrapped) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
            everywhere = [n.id for n in ast.walk(self._tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
            for name in sorted(set(inside)):
                span = self._string_constants.get(name)
                # a constant named anywhere else in the module is not this declaration's alone
                if span is not None and everywhere.count(name) == inside.count(name):
                    constants.append((name, *span))
        own = self._declared_locals(node, wrapped) if wrapped is not None else ()
        self.declarations.append(
            replace(declared_whole_read(self.path, self._function, node), constants=tuple(constants), locals=own)
        )

    def _declared_locals(self, node: ast.Call, wrapped: ast.expr) -> tuple[tuple[str, int, int], ...]:
        """the locals of the declaration's enclosing function the wrapped expression names: each bound once there and
        named nowhere else in it, with its value's lines.

        :param node: the declaration
        :ptype node: ast.Call
        :param wrapped: the wrapped read
        :ptype wrapped: ast.expr
        :return: ``(name, first, last)`` each
        :rtype: tuple[tuple[str, int, int], ...]
        """
        enclosing = [
            function
            for function in ast.walk(self._tree)
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            and function.lineno <= node.lineno <= (function.end_lineno or function.lineno)
        ]
        if not enclosing:
            return ()
        function = max(enclosing, key=lambda found: found.lineno)
        inside = [n.id for n in ast.walk(wrapped) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
        loads = [n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
        bound: dict[str, list[ast.stmt]] = {}
        for statement in ast.walk(function):
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        bound.setdefault(target.id, []).append(statement)
            elif isinstance(statement, (ast.AugAssign, ast.AnnAssign)) and isinstance(statement.target, ast.Name):
                # a second spelling of the name: never the declaration's alone
                bound.setdefault(statement.target.id, []).append(statement)
        found: list[tuple[str, int, int]] = []
        for name in sorted(set(inside)):
            assigned = bound.get(name, [])
            only = assigned[0] if len(assigned) == 1 else None
            if isinstance(only, ast.Assign) and loads.count(name) == inside.count(name):
                value = only.value
                found.append((name, value.lineno, value.end_lineno or value.lineno))
        return tuple(found)

    def _relation_produced(self, value: ast.expr) -> str | None:
        """the relation an expression names when a declared relation producer yields it: a call of a declared
        function (a module function by its name; a method on ``self`` in its class, on the class, or on a receiver
        known to be the class), or a declared attribute on ``self`` in its class or on a receiver known to be the
        class (:meth:`_name_class`).

        :param value: the expression
        :ptype value: ast.expr
        :return: :func:`produced_relation` of the declaration, or None
        :rtype: str | None
        """
        if isinstance(value, ast.Call):
            member = _final(value)
            receiver = value.func.value if isinstance(value.func, ast.Attribute) else None
        elif isinstance(value, ast.Attribute):
            member, receiver = value.attr, value.value
        else:
            return None
        for owner, name, entry in self._relation_producers:
            if name != member:
                continue
            if not owner:
                if isinstance(value, ast.Call) and receiver is None:
                    return produced_relation(entry)
                continue
            if self._receiver_is(receiver, owner):
                return produced_relation(entry)
        return None

    def _receiver_is(self, receiver: ast.expr | None, owner: str) -> bool:
        """whether a receiver is known to be an instance (or the class) of ``owner``.

        :param receiver: the expression before the attribute, or None
        :ptype receiver: ast.expr | None
        :param owner: the class's name
        :ptype owner: str
        :return: whether it is
        :rtype: bool
        """
        if not isinstance(receiver, ast.Name):
            return False
        if receiver.id == owner:
            return True
        if receiver.id == "self":
            return bool(self._class_nodes) and self._class_nodes[-1].name == owner
        known = self._name_class(receiver.id)
        return known is not None and _generic(_optional_of(known))[0] == owner

    def _name_class(self, name: str) -> ast.expr | None:
        """the annotation a name of the enclosing function is known to hold: its parameter's or an annotated
        assignment's (``source: SourceTable``, ``marked: dict[str, tuple[int, SourceTable]] = ...``), or a loop's or
        comprehension's target over an iterable whose annotation the walk knows (a ``Sequence[...]``, a mapping's
        ``.items()`` / ``.values()``, ``enumerate(...)`` of one), its position in a ``tuple[...]`` element.

        :param name: the name
        :ptype name: str
        :return: its annotation, or None when the walk cannot tell
        :rtype: ast.expr | None
        """
        function = self._function_node
        if function is None:
            return None
        declared: dict[str, ast.expr] = {
            argument.arg: argument.annotation
            for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
            if argument.annotation is not None
        }
        for node in ast.walk(function):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                declared.setdefault(node.target.id, node.annotation)
        if name in declared:
            return declared[name]

        def element(iterable: ast.expr, depth: int = 0) -> ast.expr | None:
            if isinstance(iterable, ast.Name) and iterable.id in declared:
                return _element_type(declared[iterable.id])
            if not isinstance(iterable, ast.Call) or depth > 2:
                return None
            called = _final(iterable)
            if called == "enumerate" and iterable.args:
                inner = element(iterable.args[0], depth + 1)
                return (
                    None
                    if inner is None
                    else ast.Subscript(
                        value=ast.Name("tuple", ast.Load()),
                        slice=ast.Tuple(elts=[ast.Name("int", ast.Load()), inner], ctx=ast.Load()),
                        ctx=ast.Load(),
                    )
                )
            on = iterable.func.value if isinstance(iterable.func, ast.Attribute) else None
            if called in {"items", "values"} and isinstance(on, ast.Name) and on.id in declared:
                mapping, parts = _generic(_optional_of(declared[on.id]))
                if mapping not in _MAPPINGS or len(parts) != 2:
                    return None
                if called == "values":
                    return parts[1]
                return ast.Subscript(
                    value=ast.Name("tuple", ast.Load()), slice=ast.Tuple(elts=parts, ctx=ast.Load()), ctx=ast.Load()
                )
            return None

        for node in ast.walk(function):
            loops: list[tuple[ast.expr, ast.expr]] = []
            if isinstance(node, (ast.For, ast.AsyncFor)):
                loops.append((node.target, node.iter))
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                loops += [(generator.target, generator.iter) for generator in node.generators]
            for target, iterable in loops:
                found = _target_type(target, name, element(iterable))
                if found is not None:
                    return found
        return None

    def _resolved_call(self, node: ast.Call) -> ast.Call:
        """the call with each argument that is a known string (a module or imported constant, a string the function
        binds once) written as that string, so a helper's table named by a constant (``LINK_REDEMPTIONS_TABLE``)
        reads as the literal.

        :param node: the call
        :ptype node: ast.Call
        :return: a copy with its known strings as literals
        :rtype: ast.Call
        """

        def literal(value: ast.expr) -> ast.expr:
            text = _constant_text(value, self._envs[-1])
            if text is not None and not isinstance(value, ast.Constant):
                return ast.Constant(text)
            # a collection's own schema name (``self.schema.name``): the table its class declares
            if (schema_name := self._schema_name_text(value)) is not None:
                return ast.Constant(schema_name)
            # ... in a base class declaring none, whose same-module subclasses each declare one: each of their tables,
            # one or another, judged in turn (:func:`_branch_calls`)
            if (subclass_names := self._subclass_schema_names(value)) is not None:
                chained: ast.expr = ast.Constant(subclass_names[-1])
                for name in reversed(subclass_names[:-1]):
                    chained = ast.IfExp(test=ast.Constant(True), body=ast.Constant(name), orelse=chained)
                return chained
            # a mapping keyed by constants (``by={STATE_COLUMN: state}``): each key the string it names
            if isinstance(value, ast.Dict):
                keys = [
                    key if key is None else literal(key) if _constant_text(key, self._envs[-1]) is not None else key
                    for key in value.keys
                ]
                return ast.Dict(keys=keys, values=value.values)
            # a sequence of constants (``order=(STATE_COLUMN,)``): each element the string it names
            if isinstance(value, (ast.Tuple, ast.List)):
                elements = [literal(element) for element in value.elts]
                return (
                    ast.Tuple(elts=elements, ctx=ast.Load())
                    if isinstance(value, ast.Tuple)
                    else ast.List(elts=elements, ctx=ast.Load())
                )
            # a module constant declaring a change relation (``_CHANGED = keyed.change_relation("changed")``), or a
            # name the function binds once to a step (``step = semi(...)``): the call itself, judged where it is made
            # a mapping the function binds once and adds literal keys to (``by = {...}``; ``by["status"] = ...``): each
            # shape it can take, judged in turn (:func:`_branch_calls`)
            if isinstance(value, ast.Name) and (shaped := self._local_mapping(value.id)) is not None:
                return _mapped(shaped, literal)
            declared = self._module_values.get(value.id) if isinstance(value, ast.Name) else None
            if isinstance(declared, ast.Call) and _final(declared) == "change_relation":
                return declared
            if isinstance(value, ast.Name) and (stepped := self._bound_step(value.id)) is not None:
                return literal(stepped) if _final(stepped) in {"Quoted", "Derived"} else stepped
            # a local the function binds once to the argument's shape and never changes (``by = {"id": i}``,
            # ``order = [Sort("at")]``, ``pinned = Fragment(...)``): that shape, as a step local is read
            if isinstance(value, ast.Name) and (shaped := self._bound_shape(value.id)) is not None:
                return literal(shaped)
            # one literal or the other by a condition: each arm resolved, judged in turn (:meth:`_branch_calls`)
            if _literal_branches(value):
                assert isinstance(value, ast.IfExp)
                return ast.IfExp(test=value.test, body=literal(value.body), orelse=literal(value.orelse))
            # a sequence copied (``list(where)``, ``tuple(order)``): what it copies
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in {"list", "tuple"}
                and len(value.args) == 1
                and not value.keywords
            ):
                copied = literal(value.args[0])
                if isinstance(copied, ast.Tuple | ast.List):
                    return copied
            # a relation written as an f-string (``f"main.{table}"``, ``f"{_SCRATCH}.{table}"``): its text, each
            # constant resolved, each other interpolation ``{}``; the spelling of a declared transient producer's
            # return (``_scratch``: ``f"{_SCRATCH}.{quote(table)}"``) is that producer's change set (R6)
            if isinstance(value, ast.JoinedStr):
                text = _string_text(value, self._envs[-1], self._produced_by)
                if text is not None and not is_sql(text):
                    return ast.Constant(TRANSIENT_PRODUCED if text in self._producer_spellings() else text)
            # a fragment written as an f-string (``Fragment(f"t.{STATE_COLUMN} IN ({marks})")``): its text, as raw
            # SQL interpolating the same parts is read (R7)
            if (
                isinstance(value, ast.Call)
                and _final(value) == "Fragment"
                and value.args
                and isinstance(value.args[0], ast.JoinedStr)
                and (text := _string_text(value.args[0], self._envs[-1], self._produced_by)) is not None
            ):
                return ast.Call(func=value.func, args=[ast.Constant(text), *value.args[1:]], keywords=value.keywords)
            # a module constant holding the argument's shape (``_NEWEST_FIRST = (Sort("date_started"), ...)``)
            if isinstance(value, ast.Name) and isinstance(
                self._module_values.get(value.id), (ast.Tuple, ast.List, ast.Dict)
            ):
                return self._module_values[value.id]
            # a name already quoted (``Quoted(snapshot.relation(NAME))``): the relation it names, as a statement
            # interpolating it reads (``copy.<name>``, a constant's text, a producer's change set), never trusted
            if isinstance(value, ast.Call) and _final(value) == "Quoted" and len(value.args) == 1:
                text = _placeholder(value.args[0], self._envs[-1], self._produced_by)
                return ast.Call(func=value.func, args=[ast.Constant(text)], keywords=[])
            # a derived select the code does not write, but is handed (``Derived(snapshot.relation(NAME))``, a limited
            # copy's ``(SELECT ... WHERE state IN ...)``): the relation it names, as a quoted name is read (R10);
            # one naming nothing the walk can tell stays the derived select it is
            if (
                isinstance(value, ast.Call)
                and _final(value) == "Derived"
                and len(value.args) == 1
                and not value.keywords
                and isinstance(value.args[0], ast.Call)
                and _final(value.args[0]) in COPY_CALLS
            ):
                text = _placeholder(value.args[0], self._envs[-1], self._produced_by)
                if text.startswith(COPY_PREFIX) or not re.fullmatch(r"\{[^}]*\}", text):
                    return ast.Call(func=ast.Name("Quoted", ast.Load()), args=[ast.Constant(text)], keywords=[])
            # a module function handing a step its one parameter as its relation (``_step_relation(snapshot.relation(t))``
            # returning ``Quoted(relation)`` or ``Derived(...)`` of it): the name quoted, the stricter reading of the two
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in self._relation_wrappers
                and len(value.args) == 1
                and not value.keywords
            ):
                return literal(ast.Call(func=ast.Name("Quoted", ast.Load()), args=[value.args[0]], keywords=[]))
            # a relation a declared producer yields at run time (``source.relation``, ``derived(part.relation)``):
            # the name its declaration is judged as; a mapping every value of which is one, that one relation
            produced = self._relation_produced(value)
            if produced is not None:
                return ast.Constant(produced)
            if isinstance(value, ast.DictComp) and (produced := self._relation_produced(value.value)) is not None:
                return ast.Dict(keys=[ast.Constant("")], values=[ast.Constant(produced)])
            return value

        return ast.Call(
            func=node.func,
            args=[literal(argument) for argument in node.args],
            keywords=[ast.keyword(arg=k.arg, value=literal(k.value)) for k in node.keywords],
        )

    def _judge_helper(self, node: ast.Call) -> None:
        """judge each statement a keyed helper call sends (``read_keyed``, ``keyset_page``, ...), as SQL written in the
        calling function: :mod:`~threetears.enforcement.no_whole_reads.helpers` writes it from the call's arguments.

        :param node: the call
        :ptype node: ast.Call
        :return: nothing
        :rtype: None
        """
        name = _final(node)
        if (
            _on_an_attribute(node)
            or _on_an_object(node, self._module_names)
            or not (name in KEYED_HELPER_CALLS or (name in STEP_CALLS and self._keyed_step(node)))
        ):
            # a call on an object's attribute (``self._store.holds_any``) or on an object a name holds
            # (``readings.read_by(clause)``) is the repo's own method, judged in its body
            return
        if self._hands_on_helper_relation(node, name):
            # a helper calling a helper on its own relation parameter: judged where the outer helper is called
            return
        # a step's relations, and what a keyed write from a relation reads (a forwarder's step as update_from's
        # source, R8), as the flow says they are
        flowed = name in STEP_CALLS or name in STEP_RELATIONS
        self._judge_statements(node, self._flowed(node) if flowed else node)

    def _flowed(self, node: ast.Call) -> ast.Call:
        """a step call with each relation it names that the flow says is a change relation, a step judged where it
        is written, or a forwarder's parameter judged at its call sites written as a change set
        (:data:`TRANSIENT_PRODUCED`); every other argument as it is.

        :param node: the step call
        :ptype node: ast.Call
        :return: the call to judge
        :rtype: ast.Call
        """
        function = self._function_node
        owner = self._flow.owner_of(function)
        settled: set[int] = set()
        for argument, mode in step_relation_arguments(node):
            judged = (
                self._flow.kind(argument, function, owner)
                if mode == "rel"
                else self._flow.elements(argument, function, owner)
            )
            if judged in {CLEAN, HANDED}:
                settled.add(id(argument))
        if not settled:
            return node

        def written(argument: ast.expr) -> ast.expr:
            if isinstance(argument, ast.Starred) and id(argument.value) in settled:
                return ast.Constant(TRANSIENT_PRODUCED)
            return ast.Constant(TRANSIENT_PRODUCED) if id(argument) in settled else argument

        return ast.Call(
            func=node.func,
            args=[written(argument) for argument in node.args],
            keywords=[ast.keyword(arg=k.arg, value=written(k.value)) for k in node.keywords],
        )

    def _judge_handed(self, node: ast.Call) -> None:
        """judge what a call of a forwarder hands it, where the call is: each argument the forwarder hands a step,
        read as that step reads it (a relation whole; a container's every value), unless it is a change relation, a
        step judged where it is written, or the calling forwarder's own parameter.

        :param node: the call
        :ptype node: ast.Call
        :return: nothing
        :rtype: None
        """
        callee = self._flow.callee(node)
        if callee is None:
            return
        function = self._function_node
        owner = self._flow.owner_of(function)
        for argument, mode in self._flow.handed_arguments(node, callee):
            if mode == "rel":
                if self._flow.kind(argument, function, owner) not in {CLEAN, HANDED}:
                    inlined = self._inlined_steps(node, callee, argument)
                    for step in inlined or [ast.Call(func=ast.Name("scan", ast.Load()), args=[argument], keywords=[])]:
                        self._judge_statements(node, step)
                continue
            if self._flow.elements(argument, function, owner) in {CLEAN, HANDED}:
                continue
            values = (
                argument.values
                if isinstance(argument, ast.Dict)
                else argument.elts
                if isinstance(argument, (ast.List, ast.Tuple, ast.Set))
                else None
            )
            if values is None or any(isinstance(value, ast.Starred) for value in values):
                self._hit(node, f"unknown-predicate {callee.function.name}(each)", self._spelled(node))
                continue
            for value in values:
                if self._flow.kind(value, function, owner) not in {CLEAN, HANDED}:
                    self._judge_statements(node, ast.Call(func=ast.Name("scan", ast.Load()), args=[value], keywords=[]))

    def _inlined_steps(self, call: ast.Call, callee: Callee, argument: ast.expr) -> list[ast.Call]:
        """the forwarder's own steps over the relation a call hands it, as the call makes them: each step in the
        forwarder's body naming the parameter ``argument`` binds as a relation, every parameter in it written as
        what the call passes (its relation, its state, a ``where`` list), so the step's own pins are judged with
        the relation handed; through a forwarder it hands the parameter on to (``runs_step`` ->
        ``annotated_step``), the same way. None when the forwarder hands the parameter on only some other way, or a
        step reads a parameter the call leaves to its default or the forwarder's ``self``.

        :param call: the forwarder's call
        :ptype call: ast.Call
        :param callee: the forwarder
        :ptype callee: Callee
        :param argument: the relation the call hands it
        :ptype argument: ast.expr
        :return: the steps to judge, or none
        :rtype: list[ast.Call]
        """
        passed = _passed_arguments(call, callee)
        if passed is None:
            return []
        # a parameter the call leaves to its default holds the default (``flow.handed_arguments`` hands it)
        for name, _ in callee.parameters():
            if name not in passed and (default := default_of(callee.function, name)) is not None:
                passed[name] = default
        handed = next((name for name, value in passed.items() if value is argument), None)
        if handed is None:
            return []
        # each other relation the call hands that the flow says is a change set or a step judged where it is
        # written stays one in the steps inlined, as the call's own judgement of it says
        function = self._function_node
        owner = self._flow.owner_of(function)
        settled = {
            id(value)
            for value, mode in self._flow.handed_arguments(call, callee)
            if value is not argument
            and (
                self._flow.kind(value, function, owner)
                if mode == "rel"
                else self._flow.elements(value, function, owner)
            )
            in {CLEAN, HANDED}
        }
        passed = {
            name: ast.Constant(TRANSIENT_PRODUCED) if id(value) in settled else value for name, value in passed.items()
        }
        return self._forwarded_steps(callee, passed, handed, 0) or []

    def _forwarded_steps(
        self, callee: Callee, passed: dict[str, ast.expr], handed: str, depth: int
    ) -> list[ast.Call] | None:
        """:meth:`_inlined_steps` for one forwarder, its parameters bound to ``passed``; None where it cannot.

        :param callee: the forwarder
        :ptype callee: Callee
        :param passed: each parameter the call binds, onto what it passes
        :ptype passed: dict[str, ast.expr]
        :param handed: the parameter holding the relation
        :ptype handed: str
        :param depth: forwarders followed so far
        :ptype depth: int
        :return: the steps, or None
        :rtype: list[ast.Call] | None
        """
        if depth > _FORWARD_DEPTH:
            return None
        parameters = {name for name, _ in callee.parameters()}
        # the forwarder's own locals: never read as the caller's names of the same spelling
        local = {
            node.id
            for node in ast.walk(callee.function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id not in parameters
        }
        receivers = {a.arg for a in [*callee.function.args.posonlyargs, *callee.function.args.args]} - parameters
        unbound: list[str] = []
        # a local the forwarder binds once by a plain assignment (``within = Fragment(...)``): what it binds
        stored: dict[str, int] = {}
        for node in ast.walk(callee.function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                stored[node.id] = stored.get(node.id, 0) + 1
        once = {
            node.targets[0].id: node.value
            for node in ast.walk(callee.function)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and stored.get(node.targets[0].id) == 1
        }
        inlining: list[str] = []

        class _Bind(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.expr:
                if node.id in receivers or (node.id in parameters and node.id not in passed):
                    # the forwarder's ``self`` (nothing the call site can read), or a parameter left to its default
                    unbound.append(node.id)
                    return node
                if node.id in parameters:
                    return passed[node.id]
                if node.id in once and isinstance(node.ctx, ast.Load) and len(inlining) < _INLINE_DEPTH:
                    inlining.append(node.id)
                    try:
                        inlined: ast.expr = self.visit(ast.parse(ast.unparse(once[node.id]), mode="eval").body)
                    finally:
                        inlining.pop()
                    return inlined
                if node.id in local:
                    return ast.Name(id=f"__forwarded__{node.id}", ctx=node.ctx)
                return node

        def bind(node: ast.expr) -> ast.expr:
            bound: ast.expr = _Bind().visit(ast.parse(ast.unparse(node), mode="eval").body)
            return bound

        steps: list[ast.Call] = []
        for node in ast.walk(callee.function):
            if not isinstance(node, ast.Call):
                continue
            if (_final(node) in STEP_CALLS or _final(node) in STEP_TARGETS) and self._keyed_step(node):
                if any(
                    isinstance(relation, ast.Name) and relation.id == handed
                    for relation, _ in step_relation_arguments(node)
                ):
                    # a write's connection (``InlineDuckDB(self._db)``) is nothing its statements are judged by
                    bound = bind(_target_write(node, parameters, receivers) if _final(node) in STEP_TARGETS else node)
                    if unbound or not isinstance(bound, ast.Call):
                        return None
                    steps.append(bound)
                continue
            inner = self._flow.callee(node)
            if inner is None:
                continue
            given = _passed_arguments(node, inner)
            if given is None:
                return None
            onward = next(
                (name for name, value in given.items() if isinstance(value, ast.Name) and value.id == handed), None
            )
            if onward is None:
                continue
            rebound = {name: bind(value) for name, value in given.items() if name != onward}
            if unbound:
                return None
            rebound[onward] = passed[handed]
            found = self._forwarded_steps(inner, rebound, onward, depth + 1)
            if found is None:
                return None
            steps += found
        return steps

    def _match_calls(self, call: ast.Call) -> list[ast.Call]:
        """a ``delete_matching`` call once for each ``Match`` list its ``on=`` may be, where a repo function builds it
        (:class:`MatchBuilder`): called there (``on=_alike(SETTLE_KEYS.get(name, _PANEL_KEY))``, each of the module
        mapping's values in turn), or bound once from one (``on, also = let_go_on(table)``); the call itself
        otherwise.

        :param call: the helper call, its known strings already literals
        :ptype call: ast.Call
        :return: the calls to judge
        :rtype: list[ast.Call]
        """
        given = next((keyword for keyword in call.keywords if keyword.arg == "on"), None)
        if _final(call) != "delete_matching" or given is None or not self._match_builders:
            return [call]
        built: ast.expr = given.value
        first = False
        if isinstance(built, ast.Name) and self._function_node is not None:
            bound = [
                (target, node.value)
                for node in ast.walk(self._function_node)
                if isinstance(node, ast.Assign)
                for target in node.targets
                if (isinstance(target, ast.Name) and target.id == built.id)
                or (
                    isinstance(target, ast.Tuple)
                    and any(isinstance(e, ast.Name) and e.id == built.id for e in target.elts)
                )
            ]
            if len(bound) != 1:
                return [call]
            target, built = bound[0]
            if isinstance(target, ast.Tuple):
                if not (isinstance(target.elts[0], ast.Name) and target.elts[0].id == given.value.id):  # type: ignore[attr-defined]
                    return [call]
                first = True
        if not isinstance(built, ast.Call):
            return [call]
        builder = self._match_builders.get(_final(built) or "")
        if builder is None or builder.first != first:
            return [call]
        if builder.parameter is None:
            choices: list[list[str | None]] = [list(builder.columns)]
        else:
            if len(built.args) <= builder.parameter:
                return [call]
            found = self._column_choices(built.args[builder.parameter])
            if found is None:
                return [call]
            choices = found
        calls = []
        for columns in choices[:_MAX_PARAMETER_COMBINATIONS]:
            matches = ast.List(
                elts=[
                    ast.Call(
                        func=ast.Name("Match", ast.Load()),
                        args=[ast.Constant(column) if column is not None else ast.Name("column", ast.Load())],
                        keywords=[],
                    )
                    for column in columns
                ],
                ctx=ast.Load(),
            )
            calls.append(
                ast.Call(
                    func=call.func,
                    args=call.args,
                    keywords=[k if k.arg != "on" else ast.keyword(arg="on", value=matches) for k in call.keywords],
                )
            )
        return calls

    def _column_choices(self, node: ast.expr) -> list[list[str | None]] | None:
        """the column lists an expression may be: a sequence of column names (a literal, a module constant), or a pick
        from a module mapping of them (``SETTLE_KEYS[name]``, ``SETTLE_KEYS.get(name, _PANEL_KEY)``: each value, and the
        default); None when the walk cannot tell."""
        env = self._envs[-1]

        def columns(value: ast.expr | None) -> list[str | None] | None:
            if isinstance(value, ast.Name) and isinstance(self._module_values.get(value.id), (ast.Tuple, ast.List)):
                value = self._module_values[value.id]
            if not isinstance(value, (ast.Tuple, ast.List)) or not value.elts:
                return None
            return [_constant_text(element, env) for element in value.elts]

        picked: list[ast.expr] = []
        mapping: ast.expr | None = None
        if isinstance(node, ast.Subscript):
            mapping = node.value
        elif isinstance(node, ast.Call) and _final(node) == "get" and isinstance(node.func, ast.Attribute):
            mapping = node.func.value
            picked += node.args[1:2]
        if isinstance(mapping, ast.Name):
            held = self._module_values.get(mapping.id)
            if not isinstance(held, ast.Dict) or not held.values:
                return None
            choices = [columns(value) for value in [*held.values, *picked]]
            return None if any(choice is None for choice in choices) else [c for c in choices if c is not None]
        single = columns(node)
        return None if single is None else [single]

    def _judge_statements(self, site: ast.Call, node: ast.Call) -> None:
        """judge each statement a keyed helper call sends, its findings recorded at ``site``.

        :param site: where the findings are recorded
        :ptype site: ast.Call
        :param node: the helper call judged
        :ptype node: ast.Call
        :return: nothing
        :rtype: None
        """
        statement: HelperStatement
        hits: dict[str, str] = {}
        # a judged finding of a call one site's values make is that site's: recorded where the caller passes them
        passed: dict[tuple[str, int, str, str], str] = {}
        expanded = self._declared_calls(node)
        for call, at in (
            (matched, at)
            for declared in expanded
            for branched in _branch_calls(self._none_folded(self._resolved_call(declared)))
            for picked in self._mapping_calls(branched)
            for each, at in self._parameter_calls(picked)
            for matched in self._match_calls(each)
        ):
            for statement in self._helper_statements(_local_relation(call, self._declared_values)):
                if statement.sql is UNSEEN or statement.sql is None:
                    hits.setdefault(f"unknown-predicate {statement.helper}({statement.unseen})", self._spelled(site))
                    continue
                reader = self._reader(site)
                if _on_a_local_database(call) or self._bound_to_a_local_database(call):
                    reader = replace(reader, local=True)
                for finding in judge_sql(statement.sql, reader):
                    if finding.kind in {HEAD, KEYED_COUNT} or self._transient_table(finding.table):
                        continue
                    symbol = f"{finding.kind} {finding.table}"
                    if at is None or finding.kind.startswith("unknown-"):
                        hits.setdefault(symbol, statement.sql)
                    else:
                        passed.setdefault((at.path, at.line, at.function, symbol), statement.sql)
        for symbol, statement_text in _family_folded(hits, len(expanded)).items():
            self._hit(site, symbol, statement_text)
        for (path, line, function, symbol), statement_text in passed.items():
            self.hits.append(
                WholeReadHit(path, line, function, symbol, _one_line(f"{statement_text} (passed to {self._function})"))
            )

    def _bound_to_a_local_database(self, call: ast.Call) -> bool:
        """whether a helper call's connection is a name the enclosing function binds only to a connection over a
        database in this process (``db = InlineSQLite(conn)`` then ``delete_oldest(db, ...)``): never a parameter of
        it, and every assignment to it in the function's own statements such a connection.

        :param call: the helper call
        :ptype call: ast.Call
        :return: whether it is
        :rtype: bool
        """
        function = self._function_node
        if function is None or not call.args or not isinstance(call.args[0], ast.Name):
            return False
        name = call.args[0].id
        arguments = function.args
        given = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]
        if any(argument is not None and argument.arg == name for argument in given):
            return False
        bound: list[ast.expr | None] = []
        for node in _own_nodes(function.body):
            if isinstance(node, ast.Assign) and any(_target_binds(target, name) for target in node.targets):
                bound.append(node.value if all(isinstance(t, ast.Name) for t in node.targets) else None)
            elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr) and _target_binds(node.target, name):
                bound.append(node.value if not isinstance(node, ast.AugAssign) else None)
            elif isinstance(node, ast.For | ast.AsyncFor | ast.With | ast.AsyncWith) and _binds_in(node, name):
                bound.append(None)
        return bool(bound) and all(
            isinstance(value, ast.Call) and _final(value) in _IN_PROCESS_CONNECTIONS for value in bound
        )

    def _helper_statements(self, call: ast.Call) -> list[HelperStatement]:
        """the statements a keyed helper call sends; a read of a table its caller names at run time
        (``read_row(conn, self.table_name, ...)``) read as the module's template where the module declares one
        (ruling #13), as SQL the caller wrote over ``{table}`` is, and unknown where it declares none.

        :param call: the helper call, its arguments resolved
        :ptype call: ast.Call
        :return: each statement
        :rtype: list[HelperStatement]
        """
        statements = list(helper_statements(call))
        if self._schema.templates.get(self.path) is None or not any(
            statement.sql is UNSEEN and statement.unseen in {"table", "relation"} for statement in statements
        ):
            return statements
        named = templated(call)
        return statements if named is None else list(helper_statements(named))

    def _hands_on_helper_relation(self, node: ast.Call, name: str | None) -> bool:
        """whether a keyed helper call inside a keyed helper's body reads only the outer helper's relation parameter
        (``read_keyed(conn, table, ...)`` in ``read_joined``; ``counts_by_relations(conn, {"": table})`` in
        ``counts_by``; a step whose every relation is one).

        :param node: the call
        :ptype node: ast.Call
        :param name: the helper's name
        :ptype name: str | None
        :return: whether it does
        :rtype: bool
        """
        function = self._function_node
        held = self._helper_parameters.get(id(function), frozenset()) if function is not None else frozenset()
        if not held:
            return False

        def holds(argument: ast.expr | None) -> bool:
            if isinstance(argument, ast.Dict):
                return bool(argument.values) and all(holds(value) for value in argument.values)
            return isinstance(argument, ast.Name) and argument.id in held

        if name in STEP_CALLS:
            places = STEP_RELATIONS.get(name, ())
            return bool(places) and all(holds(_bound_argument(node, keyword, position)) for position, keyword in places)
        for keyword in ("table", "relation", "tables"):
            given = next((k.value for k in node.keywords if k.arg == keyword), None)
            if given is not None:
                return holds(given)
        # a builder takes no connection: its relation is its first argument
        place = 0 if name in BUILDER_CALLS else 1
        return len(node.args) > place and holds(node.args[place])

    def _local_mapping(self, name: str) -> ast.expr | None:
        """the shapes a mapping the enclosing function binds to ``name`` can take: bound once to a literal mapping (or
        a conditional of two), then only given keys by ``name[<string>] = value``; a key given in the function's own
        body is always there, one given under a condition or loop either there or not. None when the function binds
        or changes ``name`` any other way (``name.pop(...)``, ``del name[...]``, a second binding), reads it through
        an attribute, or one given key is no string.

        :param name: the name
        :ptype name: str
        :return: the shapes as a conditional of mappings (or the one mapping), or None
        :rtype: ast.expr | None
        """
        function = self._function_node
        if function is None:
            return None
        env = self._envs[-1]
        bound: list[ast.expr] = []
        given: list[tuple[str, ast.expr, bool]] = []
        top = {id(statement) for statement in function.body}
        for node in _own_nodes(function.body):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        if node.value is None or len(targets) != 1:
                            return None
                        bound.append(node.value)
                    elif (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == name
                    ):
                        key = _constant_text(target.slice, env)
                        if key is None or len(targets) != 1 or node.value is None:
                            return None
                        given.append((key, node.value, id(node) in top))
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == name:
                return None
            elif isinstance(node, (ast.Delete, ast.AugAssign)) and any(
                isinstance(each, ast.Name) and each.id == name for each in ast.walk(node)
            ):
                return None
        # a store other than the one binding (a loop target, ``del``, a walrus) leaves it no one shape
        stores = [
            node
            for node in _own_nodes(function.body)
            if isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load)
        ]
        if len(bound) != 1 or len(stores) != 1 or name in _function_bindings(function, {}).parameters:
            return None
        value = bound[0]
        arms = [value.body, value.orelse] if isinstance(value, ast.IfExp) else [value]
        if not all(isinstance(arm, ast.Dict) and None not in arm.keys for arm in arms):
            return None
        always = [(key, put) for key, put, at_top in given if at_top]
        every = [(key, put) for key, put, _ in given]

        def shaped(arm: ast.expr) -> ast.expr:
            assert isinstance(arm, ast.Dict)
            shapes = {
                ast.unparse(made): made
                for made in (
                    ast.Dict(
                        keys=[*arm.keys, *(ast.Constant(key) for key, _ in added)],
                        values=[*arm.values, *(put for _, put in added)],
                    )
                    for added in (always, every)
                )
            }
            *rest, last = shapes.values()
            chained: ast.expr = last
            for shape in reversed(rest):
                chained = ast.IfExp(test=ast.Constant(True), body=shape, orelse=chained)
            return chained

        # a conditional binding keeps its test, so an arm its callers never take is folded away as where it is
        # written in the call (:meth:`_none_folded`)
        if isinstance(value, ast.IfExp):
            return ast.IfExp(test=value.test, body=shaped(value.body), orelse=shaped(value.orelse))
        return shaped(value)

    def _bound_step(self, name: str) -> ast.Call | None:
        """the step (or change relation, or derived select) the enclosing function binds ``name`` to once, if it does.

        :param name: the local name
        :ptype name: str
        :return: the call it is bound to, or None
        :rtype: ast.Call | None
        """
        if not self._function_nodes:
            return None
        bound = [
            node.value
            for node in ast.walk(self._function_nodes[-1])
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        if len(bound) == 1 and isinstance(bound[0], ast.Call):
            callee = _final(bound[0])
            if callee in STEP_CALLS or callee in {"change_relation", "Derived", "Quoted"}:
                return bound[0]
        return None

    def _bound_shape(self, name: str) -> ast.Dict | ast.Tuple | ast.List | ast.Call | ast.IfExp | None:
        """the literal mapping or sequence the enclosing function binds ``name`` to, if it binds it exactly once and
        never changes it after (no item stored or deleted, no method called on it, no other binding): a ``by`` or
        ``order`` held in a local reads as the literal it holds, and one the function may change reads as unknown.

        :param name: the local name
        :ptype name: str
        :return: the literal (a mapping, a sequence, a ``Fragment``, or one of two such literals by a condition), or
            None
        :rtype: ast.Dict | ast.Tuple | ast.List | ast.Call | ast.IfExp | None
        """
        if not self._function_nodes:
            return None
        bound: list[ast.expr] = []
        for node in ast.walk(self._function_nodes[-1]):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    named = [element for element in ast.walk(target) if isinstance(element, ast.Name)]
                    if any(element.id == name for element in named):
                        if not (isinstance(target, ast.Name) and target.id == name):
                            return None
                        bound.append(node.value)
            elif isinstance(node, ast.AugAssign | ast.AnnAssign | ast.NamedExpr | ast.For | ast.AsyncFor):
                if any(isinstance(element, ast.Name) and element.id == name for element in ast.walk(node.target)):
                    return None
            elif isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load):
                if isinstance(node.value, ast.Name) and node.value.id == name:
                    return None
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == name:
                    return None
            elif isinstance(node, ast.arg) and node.arg == name:
                return None
        if len(bound) == 1 and isinstance(bound[0], ast.Dict | ast.Tuple | ast.List):
            return bound[0]
        if len(bound) == 1 and isinstance(bound[0], ast.Call) and _final(bound[0]) == "Fragment":
            return bound[0]
        if len(bound) == 1 and _literal_branches(bound[0]):
            # one literal or the other (``by = {...} if agent_id is None else {...}``): each judged in turn
            return bound[0]  # type: ignore[return-value]
        return None

    def _producer_spellings(self) -> frozenset[str]:
        """the text each declared transient producer of this module returns (``_scratch``: ``_export.{}``), an
        interpolation's every name ``{}``: a relation spelled the same is one of the change sets the producer names.
        A text naming nothing but interpolations (``{}``, ``"{}"``) is no spelling: it would match any relation.

        :return: the spellings
        :rtype: frozenset[str]
        """
        if self._spellings is None:
            found: set[str] = set()
            for function, owner in _definitions(self._tree):
                if (owner.name if owner is not None else "", function.name) not in self._producers:
                    continue
                for node in ast.walk(function):
                    if isinstance(node, ast.Return) and node.value is not None:
                        text = _string_text(node.value, self._envs[0])
                        if text is not None and re.sub(r"\{[^}]*\}|[\s\"'.]", "", text):
                            found.add(text)
            self._spellings = frozenset(found)
        return self._spellings

    def _keyed_step(self, node: ast.Call) -> bool:
        """whether ``node`` calls one of 3tears' relational steps (``semi``, ``union``, ...): a name imported from
        ``threetears.core.keyed``, or an attribute of a module imported as it (``keyed.semi``); a set's ``union`` or a
        repo's own ``scan`` is not one.

        :param node: the call
        :ptype node: ast.Call
        :return: whether it is
        :rtype: bool
        """
        func = node.func
        if isinstance(func, ast.Name):
            return self.imports.get(func.id, "").startswith(_KEYED_MODULES)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return self.imports.get(func.value.id, "").startswith(_KEYED_MODULES)
        return False

    def _judge_change_relation(self, node: ast.Call) -> None:
        """a change relation declared over a data table is a whole read laundered: flag it where it is declared.

        :param node: the call
        :ptype node: ast.Call
        :return: nothing
        :rtype: None
        """
        if _final(node) != "change_relation" or not node.args:
            return
        name = _constant_text(node.args[0], self._envs[-1])
        if name is not None and self._schema.table(name.strip('"')) is not None:
            self._hit(node, f"change-relation-is-a-table {name}", self._spelled(node))

    def _mapping_calls(self, call: ast.Call) -> list[ast.Call]:
        """the call once per combination of the entries its arguments pick from module mappings of literal sequences
        (``order=DATASOURCE_ORDERS[field]`` with ``DATASOURCE_ORDERS = {"name": [Sort("name"), Sort("id")], ...}``),
        picked in the argument itself or through a local bound to the pick once (``order = DATASOURCE_ORDERS[field]``):
        any entry may be the one sent, so each is judged; the call itself when no argument picks, or the combinations
        pass :data:`_MAX_PARAMETER_COMBINATIONS`.

        :param call: the helper call, its known strings already literals
        :ptype call: ast.Call
        :return: the calls to judge
        :rtype: list[ast.Call]
        """
        function = self._function_node
        picks: dict[str, ast.expr] = {}
        if function is not None:
            assigned: dict[str, list[ast.expr]] = {}
            for node in ast.walk(function):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        for name in (n for n in ast.walk(target) if isinstance(n, ast.Name)):
                            value = node.value if isinstance(target, ast.Name) and node.value is not None else None
                            assigned.setdefault(name.id, []).append(value if value is not None else target)
            picks = {name: values[0] for name, values in assigned.items() if len(values) == 1}

        def entries(value: ast.expr) -> list[ast.expr] | None:
            if isinstance(value, ast.Name) and value.id in picks:
                value = picks[value.id]
            if not (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)):
                return None
            mapping = self._module_values.get(value.value.id)
            if not isinstance(mapping, ast.Dict) or not mapping.values:
                return None
            values = list(mapping.values)
            # a pick by a literal key (``CHANNEL_ORDERS[("date_created", False)]``) is that entry alone, when the
            # mapping spells that key; a pick by a run-time key may be any entry
            literal = _literal(value.slice)
            if literal is not _NOT_LITERAL:
                chosen = [
                    v for k, v in zip(mapping.keys, values, strict=True) if k is not None and _literal(k) == literal
                ]
                if len(chosen) == 1:
                    values = chosen
            picked = [
                self._module_values.get(entry.id, entry) if isinstance(entry, ast.Name) else entry for entry in values
            ]
            if not all(isinstance(entry, (ast.Tuple, ast.List)) for entry in picked):
                return None
            return picked

        arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
        choices = {index: found for index, value in enumerate(arguments) if (found := entries(value)) is not None}
        combinations = list(itertools.product(*choices.values()))
        if not choices or len(combinations) > _MAX_PARAMETER_COMBINATIONS:
            return [call]
        calls = []
        for combination in combinations:
            given = dict(zip(choices, combination, strict=True))
            values = [given.get(index, value) for index, value in enumerate(arguments)]
            calls.append(
                ast.Call(
                    func=call.func,
                    args=values[: len(call.args)],
                    keywords=[
                        ast.keyword(arg=keyword.arg, value=value)
                        for keyword, value in zip(call.keywords, values[len(call.args) :], strict=True)
                    ],
                )
            )
        return calls

    def _none_folded(self, call: ast.Call) -> ast.Call:
        """the call with each argument that is one literal or another by whether a parameter of the calling function
        is None (``{...} if agent_id is None else {...}``) read as the one arm it takes, where every caller passes that
        parameter something never None (:func:`_never_none_parameters`): the other arm is never sent.

        :param call: the call, its arguments resolved
        :ptype call: ast.Call
        :return: the call, folded
        :rtype: ast.Call
        """
        function = self._function_node
        if function is None or not self._never_none:
            return call

        def folded(value: ast.expr) -> ast.expr:
            if not _literal_branches(value):
                return value
            assert isinstance(value, ast.IfExp)
            tested = _none_test(value.test)
            if tested is None or (function.name, tested[0]) not in self._never_none:
                return value
            return value.body if tested[1] else value.orelse

        return ast.Call(
            func=call.func,
            args=[folded(argument) for argument in call.args],
            keywords=[ast.keyword(arg=keyword.arg, value=folded(keyword.value)) for keyword in call.keywords],
        )

    def _parameter_calls(self, call: ast.Call) -> list[tuple[ast.Call, Site | None]]:
        """the call once per combination of what its arguments are passed: a name in an argument, at any depth, that
        is a parameter of the calling function never rebound in it reads as what each call site in the repo fills it
        with -- a literal or a module constant (``_held(conn, GEOS, key)``), a tuple of strings (a module tuple of key
        columns), a mapping with string keys -- followed up through callers that hand on their own parameter, one
        site at a time (one site's table never judged with another's key); so does one bound by a loop over a module
        tuple (``for name in NAMES``), and an attribute of either when it holds declared objects (``table.name``,
        ``for table in TABLES``). A comprehension over a tuple so passed reads as the mapping it builds
        (``{c: data[c] for c in pk_cols}``). A name one site passes something the walk cannot read stays as written
        at every site; the call itself when nothing is passed, or past :data:`_MAX_PARAMETER_COMBINATIONS`. Each call
        comes with the site whose values it reads when that site passes any value read, so a finding it judges (one
        the call's structure decides: never an unknown) is the site's, what that caller passes; None otherwise.

        :param call: the helper call, its known strings already literals
        :ptype call: ast.Call
        :return: the calls to judge, each with its site or None
        :rtype: list[tuple[ast.Call, Site | None]]
        """
        function = self._function_node
        declared = _declared_of(function, self._function, self._declared) if function is not None else {}
        if function is None or not (self._parameter_values or self._tuples or declared):
            return [(call, None)]
        bindings = _function_bindings(function, self._tuples, declared)
        # a declared value (``declared_parameters``) is one of a family judged member by member: its own bound
        most = (
            _MAX_DECLARED_COMBINATIONS
            if declared or any(self._declared_site(name) for name in self._parameter_sites_of(function))
            else _MAX_PARAMETER_COMBINATIONS
        )
        arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
        named = _argument_names(arguments)
        # names every site reads alike: a loop over a module tuple, an attribute kept from the constructor
        shared: dict[str, list[ast.expr]] = {}
        parameters: dict[str, tuple[str, str | None]] = {}
        for spelled, (name, attribute) in named.items():
            if name == "self" and attribute is not None:
                held = self._held_parameter(attribute)
                passed = self._parameter_values.get(held) if held is not None else None
                texts = _texts_of(passed, None) if passed and all(v is not None for v in passed) else None
                if texts:
                    shared[spelled] = [ast.Constant(text) for text in sorted(set(texts))]
            elif name in bindings.loops:
                nodes = _passed_nodes(bindings.loops[name], attribute)
                if nodes:
                    shared[spelled] = nodes
            elif name in bindings.parameters:
                parameters[spelled] = (name, attribute)
        passings = self._parameter_sites.get(function.name, []) if parameters else []
        read = [
            {
                spelled: _passed_nodes(passing.values.get(name, [None]), attribute)
                for spelled, (name, attribute) in parameters.items()
            }
            for passing in passings
        ]
        # a parameter one site passes what the walk cannot read stays as written at every site (as one statement it
        # is unknown; the sites it can read never judged alone)
        readable = {spelled for spelled in parameters if read and all(each[spelled] for each in read)}
        sites: list[tuple[dict[str, list[ast.expr]], Site | None]] = []
        for passing, of_site in zip(passings, read, strict=True):
            put = {spelled: of_site[spelled] for spelled in readable}
            sites.append((put, passing if put else None))
        if not any(put for put, _ in sites) and not shared:
            return [(call, None)]
        combinations: list[tuple[dict[str, ast.expr], Site | None]] = []
        for given, at in sites or [({}, None)]:
            choices = {**shared, **given}
            for chosen in itertools.product(*choices.values()):
                combinations.append((dict(zip(choices, chosen, strict=True)), at))
                if len(combinations) > most:
                    return [(call, None)]
        calls: list[tuple[ast.Call, Site | None]] = []
        seen: set[str] = set()
        for combination, site in combinations:
            bound = _Substituted(combination)
            made = ast.Call(
                func=call.func,
                args=[self._looked_up(_folded_literals(bound.visit(_copied(argument)))) for argument in call.args],
                keywords=[
                    ast.keyword(arg=k.arg, value=self._looked_up(_folded_literals(bound.visit(_copied(k.value)))))
                    for k in call.keywords
                ],
            )
            text = f"{ast.unparse(made)} {None if site is None else (site.path, site.line)}"
            if text not in seen:
                seen.add(text)
                calls.append((made, site))
        return calls

    def _parameter_sites_of(self, function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """the qualified scopes of each call site of ``function``.

        :param function: the function
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
        :return: each site's calling scope
        :rtype: list[str]
        """
        return [site.function for site in self._parameter_sites.get(function.name, [])]

    def _declared_site(self, scope: str) -> bool:
        """whether a calling scope has a declared parameter (``declared_parameters``), whose values its call hands on.

        :param scope: the calling scope's qualified name
        :ptype scope: str
        :return: whether it has one
        :rtype: bool
        """
        return any(key.rsplit(".", 1)[0] == scope for key in self._declared)

    def _declared_calls(self, call: ast.Call) -> list[ast.Call]:
        """the call once per value of each name in it bound to a declared family (``declared_parameters``): the
        declared parameter itself, a name a loop over it binds (``for table in tables``), and a name the function
        binds once from either (``held_in = f"main.{table}"``, ``column = quote_identifier(_state_column(table))``),
        put in place, each value a literal, a lookup function over it read (:func:`_lookup_functions`). Every value
        is judged, so a key the read binds must key every member; a call naming no such name is itself.

        :param call: the helper call, as written
        :ptype call: ast.Call
        :return: the calls to judge
        :rtype: list[ast.Call]
        """
        function = self._function_node
        declared = _declared_of(function, self._function, self._declared) if function is not None else {}
        if function is None or not declared:
            return [call]
        family = _function_bindings(function, self._tuples, declared).loops
        plain = _function_bindings(function, self._tuples).loops
        named = {name: values for name, values in family.items() if name not in plain}
        if not named:
            return [call]
        inlined = self._inlined_locals(call, function, frozenset(named))
        used = sorted({node.id for node in ast.walk(inlined) if isinstance(node, ast.Name) and node.id in named})
        if not used:
            return [call]
        texts = [[value for value in named[name] if isinstance(value, str)] for name in used]
        if any(len(values) != len(named[name]) for values, name in zip(texts, used, strict=True)):
            return [call]
        calls: list[ast.Call] = []
        seen: set[str] = set()
        for chosen in itertools.product(*texts):
            if len(calls) >= _MAX_DECLARED_COMBINATIONS:
                return [call]
            bound = _Substituted({name: ast.Constant(value) for name, value in zip(used, chosen, strict=True)})
            made = self._looked_up(bound.visit(_copied(inlined)))
            assert isinstance(made, ast.Call)
            text = ast.unparse(made)
            if text not in seen:
                seen.add(text)
                calls.append(made)
        return calls

    def _inlined_locals(
        self, call: ast.Call, function: ast.FunctionDef | ast.AsyncFunctionDef, family: frozenset[str]
    ) -> ast.Call:
        """the call with each name the function binds once (a plain assignment) to an expression reading a family
        name written as that expression, a few bindings deep.

        :param call: the call
        :ptype call: ast.Call
        :param function: the function it is in
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
        :param family: the names bound to a declared family
        :ptype family: frozenset[str]
        :return: the call, inlined
        :rtype: ast.Call
        """
        stores: dict[str, int] = {}
        bound: dict[str, ast.expr] = {}
        for node in _own_nodes(function.body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                stores[node.id] = stores.get(node.id, 0) + 1
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                bound[node.targets[0].id] = node.value
        reading = set(family)
        local: dict[str, ast.expr] = {}
        for _ in range(_INLINE_DEPTH):
            for name, value in bound.items():
                if stores.get(name) == 1 and name not in local and name not in family:
                    if any(isinstance(node, ast.Name) and node.id in reading for node in ast.walk(value)):
                        local[name] = value
                        reading.add(name)
        made: ast.expr = _copied(call)
        for _ in range(_INLINE_DEPTH):
            made = _Substituted(local).visit(made)
        assert isinstance(made, ast.Call)
        return made

    def _looked_up(self, node: ast.expr) -> ast.expr:
        """``node`` with each call of a lookup function of the module on a literal (``_state_column("lsf_rows")``)
        written as the string it returns (:func:`_lookup_functions`).

        :param node: the expression
        :ptype node: ast.expr
        :return: it, looked up
        :rtype: ast.expr
        """
        if self._lookups is None:
            self._lookups = _lookup_functions(self._tree, self._module_values, self._envs[0])
        if not self._lookups:
            return node
        lookups = self._lookups

        class _LookedUp(ast.NodeTransformer):
            def visit_Call(self, call: ast.Call) -> ast.AST:
                self.generic_visit(call)
                found = lookups.get(call.func.id) if isinstance(call.func, ast.Name) else None
                argument = call.args[0] if len(call.args) == 1 and not call.keywords else None
                if found is None or not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
                    return call
                mapping, default = found
                value = mapping.get(argument.value, default)
                return call if value is None else ast.Constant(value)

        looked = _LookedUp().visit(node)
        assert isinstance(looked, ast.expr)
        return looked

    def _held_parameter(self, attribute: str) -> tuple[str, str] | None:
        """the constructor parameter the enclosing class keeps as ``self.<attribute>``: assigned once in the whole
        class, in its ``__init__``'s own body, straight from a parameter ``__init__`` never rebinds; None otherwise.

        :param attribute: the attribute
        :ptype attribute: str
        :return: ``(<Class>.__init__, parameter)``, as the constructions' values are kept, or None
        :rtype: tuple[str, str] | None
        """
        owner = self._class_nodes[-1] if self._class_nodes else None
        if owner is None:
            return None
        init = next(
            (
                statement
                for statement in owner.body
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef) and statement.name == "__init__"
            ),
            None,
        )
        if init is None:
            return None
        stores = [
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Attribute)
            and node.attr == attribute
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and not isinstance(node.ctx, ast.Load)
        ]
        assigned = [
            statement
            for statement in init.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and any(statement.targets[0] is store for store in stores)
        ]
        if len(stores) != 1 or len(assigned) != 1 or not isinstance(assigned[0].value, ast.Name):
            return None
        parameter = assigned[0].value.id
        if parameter not in _function_bindings(init, {}).parameters:
            return None
        return f"{owner.name}.__init__", parameter

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
        # ``after=`` / ``since=`` say where a page starts, not which rows: alone they read on to the table's end
        narrowing = [k for k in node.keywords if k.arg in self._narrowing and k.arg not in START_KEYWORDS]
        narrowed = (narrows(node, self._narrowing) and any(not _maybe_none(k.value) for k in narrowing)) or any(
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
        flags = [
            keyword.arg
            for keyword in node.keywords
            if not constructs
            and keyword.arg in _WHOLE_FLAGS
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
        ]
        # a flag to one of the module's functions or the caller's own method is judged once the call graph says
        # whether that callee reads anything whole (a number format's ``whole=`` reads nothing); any other callee's
        # flag is a finding here
        local_callee = (isinstance(node.func, ast.Name) and not segments) or (
            segments in (["self"], ["cls"]) and isinstance(node.func, ast.Attribute)
        )
        for flag in flags if not local_callee else []:
            self._hit(node, f"whole-flag {flag}=", spelled)
        if name in {"add", "update", "extend"} and segments and _WHOLE_SET.search(segments[-1]):
            self._hit(node, f"marks-whole {segments[-1]}", spelled)
        if name == "truncate" and segments and _STORE.search(segments[-1]):
            self._hit(node, "truncate-call truncate", spelled)
        if name in self._exec_calls:
            self._judge_named_sql(node)
        if name in _CREDITED_PAGERS:
            self._credit_pager(node)
        narrowed = narrowed or any(
            isinstance(value, ast.Dict)
            or (isinstance(value, ast.Call) and _final(value) == "dict")
            or _NARROWING_NAME.search((dotted(value) or "").rsplit(".", 1)[-1]) is not None
            for value in [*node.args[1:], *(k.value for k in node.keywords)]
        )
        whole_named = (
            name != DECLARATION_CALL
            and name not in self._pagers
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
        # a receiver of any other name is kept when the walk knows its class: a call on it reaches that class
        if receiver != "other" or whole_named or self._receiver_class(node):
            self._call(
                node,
                name,
                receiver,
                spelled,
                narrowed=narrowed,
                whole_named=whole_named,
                whole_flags=tuple(flags) if local_callee else (),
            )

    def _credit_pager(self, node: ast.Call) -> None:
        """record the purposed index a keyset pager's pages read through (``keyset_pages(l3, head, key, where=...,
        conditions=[Fragment(...)])``): its page is written out and judged in the calling function for the index it
        uses only; whether the pager is narrowed is the pager rules' to say.

        :param node: the pager call
        :ptype node: ast.Call
        :return: nothing
        :rtype: None
        """
        head = _string_text(node.args[1], self._envs[-1], self._produced_by) if len(node.args) > 1 else None
        key_node = node.args[2] if len(node.args) > 2 else _keyword_value(node, "key")
        key = [
            e.value for e in getattr(key_node, "elts", []) if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        if head is None or not key:
            return
        conditions: list[str] = []
        for argument, form in (("where", "{} = $9"), ("where_any", "{} = ANY($9)")):
            mapping = _keyword_value(node, argument)
            if isinstance(mapping, ast.Dict):
                conditions.extend(
                    form.format(k.value)
                    for k in mapping.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                )
        listed = _keyword_value(node, "conditions")
        for element in getattr(listed, "elts", []):
            fragment = element.args[0] if isinstance(element, ast.Call) and element.args else element
            written = _string_text(fragment, self._envs[-1])
            conditions.append(written if written is not None else "{}")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        judge_sql(f"{head}{where} ORDER BY {key[0]} LIMIT 100", self._reader(node))

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
        self,
        node: ast.Call,
        name: str,
        receiver: str,
        spelled: str,
        *,
        narrowed: bool,
        whole_named: bool,
        whole_flags: tuple[str, ...] = (),
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
        :param whole_flags: the whole flags (``whole=True``) it passes, judged once its callee is resolved
        :ptype whole_flags: tuple[str, ...]
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
                self._receiver_class(node),
                whole_flags,
            )
        )

    def _enclosing_function(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """the function ``function`` is nested in directly; None for one at a module's or class's top level.

        :param function: the function
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
        :return: the enclosing function, or None
        :rtype: ast.FunctionDef | ast.AsyncFunctionDef | None
        """
        if self._enclosing is None:
            self._enclosing = {}
            for outer in ast.walk(self._tree):
                if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for inner in _own_nodes(outer.body):
                        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self._enclosing[id(inner)] = outer
        return self._enclosing.get(id(function))

    def _receiver_class(
        self,
        node: ast.Call,
        function: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
        owner: ast.ClassDef | None = None,
    ) -> str:
        """the class a method call's receiver is known to be: a name annotated with a class or assigned an instance of
        one in the calling function (its parameters included), or a ``self`` attribute annotated or assigned so in the
        calling class (from a parameter annotated with it, too); empty when nothing says.

        :param node: the call
        :ptype node: ast.Call
        :param function: the calling function; None for the one the walk is in
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef | None
        :param owner: the calling class; None for the one the walk is in
        :ptype owner: ast.ClassDef | None
        :return: the class's name, its final segment; empty when unknown
        :rtype: str
        """
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        function = function if function is not None else self._function_node
        owner = owner if owner is not None else (self._class_nodes[-1] if self._class_nodes else None)
        known = ""
        if isinstance(receiver, ast.Name) and function is not None:
            arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
            for argument in arguments:
                if argument.arg == receiver.id and argument.annotation is not None:
                    known = _annotated_class(argument.annotation)
            known = (
                _assigned_class(function.body, lambda target: isinstance(target, ast.Name) and target.id == receiver.id)
                or known
            )
            if not known and owner is not None:
                # ``work = self.work``, ``work, settled = self.work, self.settled``: the attribute's class
                held = _self_attribute_bound(function, receiver.id)
                if held is not None:
                    known = self._receiver_class(
                        ast.Call(func=ast.Attribute(value=held, attr="_", ctx=ast.Load()), args=[], keywords=[]),
                        function,
                        owner,
                    )
            enclosing = self._enclosing_function(function)
            if not known and enclosing is not None and not _binds(function, receiver.id):
                # a name a nested function reads from the function around it (a closure): that function's
                known = self._receiver_class(node, enclosing, owner)
        elif (
            isinstance(receiver, ast.Attribute)
            and isinstance(receiver.value, ast.Name)
            and receiver.value.id == "self"
            and owner is not None
        ):
            attribute = receiver.attr
            known = _assigned_class(
                owner.body,
                lambda target: (
                    (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == attribute
                    )
                    or (isinstance(target, ast.Name) and target.id == attribute)
                ),
            )
        return known

    def _schema_name_text(self, node: ast.expr) -> str | None:
        """the table a collection's own schema name reads as: ``<schema>.name`` where ``<schema>`` is the walked class's
        declared ``TableSchema`` (``self.schema``, ``type(self).schema``, ``cls.schema``, ``<Class>.schema``), a
        ``dataclasses.replace`` of it naming the same table (``name=f"{audit_schema}.audit_events"``), or a local the
        function binds once to either. It reads as the declared name (``platform_audit.audit_events``), so the catalog
        finds the table by its last segment and judges the read with that table's declared indexes.

        :param node: the expression
        :ptype node: ast.expr
        :return: the declared table name, or None when the expression is no such name
        :rtype: str | None
        """
        if not (isinstance(node, ast.Attribute) and node.attr == "name"):
            return None
        return self._declared_schema(node.value, 0)

    def _declared_schema(self, node: ast.expr, depth: int) -> str | None:
        """the declared table name of an expression that is the walked class's ``TableSchema`` (see
        :meth:`_schema_name_text`).

        :param node: the expression
        :ptype node: ast.expr
        :param depth: how many local bindings and replaces the walk followed
        :ptype depth: int
        :return: the declared table name, or None
        :rtype: str | None
        """
        owner = self._class_nodes[-1] if self._class_nodes else None
        if depth > 3 or owner is None:
            return None
        if isinstance(node, ast.Attribute) and node.attr == "schema":
            receiver = node.value
            own = (isinstance(receiver, ast.Name) and receiver.id in {"self", "cls", owner.name}) or (
                isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Name)
                and receiver.func.id == "type"
                and len(receiver.args) == 1
                and isinstance(receiver.args[0], ast.Name)
                and receiver.args[0].id == "self"
            )
            return self._class_schema_name(owner, 0) if own else None
        if isinstance(node, ast.Call) and _final(node) == "replace" and node.args:
            declared = self._declared_schema(node.args[0], depth + 1)
            renamed = _keyword_value(node, "name")
            if declared is None or renamed is None:
                return declared
            # a copy renamed onto the same table in another schema (``f"{audit_schema}.audit_events"``) is that table;
            # one renamed onto any other table is not the declaration's
            text = _string_text(renamed, self._envs[-1], self._produced_by)
            return declared if text is not None and bare_table(text) == bare_table(declared) else None
        function = self._function_node
        if isinstance(node, ast.Name) and function is not None:
            bound = [
                statement.value
                for statement in ast.walk(function)
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == node.id
            ]
            return self._declared_schema(bound[0], depth + 1) if len(bound) == 1 else None
        return None

    def _subclass_schema_names(self, node: ast.expr) -> list[str] | None:
        """the tables a base class's own schema name (``self.schema.name``) may be: the walked class declares no
        ``TableSchema`` (nor does a same-module base), and its same-module subclasses (direct or further down) declare
        one each. A shared method there (``CoordinationCollection.sweep_expired``) runs on each subclass's table.

        :param node: the expression
        :ptype node: ast.expr
        :return: the subclasses' declared names, in module order; None when the expression is no such name, the class
            declares its own, or fewer than one subclass declares one
        :rtype: list[str] | None
        """
        owner = self._class_nodes[-1] if self._class_nodes else None
        if owner is None or not (isinstance(node, ast.Attribute) and node.attr == "name"):
            return None
        receiver = node.value
        if not (
            isinstance(receiver, ast.Attribute)
            and receiver.attr == "schema"
            and isinstance(receiver.value, ast.Name)
            and receiver.value.id in {"self", "cls"}
        ):
            return None
        if self._class_schema_name(owner, 0) is not None:
            return None
        classes = [statement for statement in self._tree.body if isinstance(statement, ast.ClassDef)]
        below = {owner.name}
        changed = True
        while changed:
            changed = False
            for candidate in classes:
                if candidate.name not in below and any(
                    isinstance(base, ast.Name) and base.id in below for base in candidate.bases
                ):
                    below.add(candidate.name)
                    changed = True
        names = [
            found
            for candidate in classes
            if candidate.name in below - {owner.name} and (found := self._class_schema_name(candidate, 0)) is not None
        ]
        unique = list(dict.fromkeys(names))
        return unique or None

    def _class_schema_name(self, owner: ast.ClassDef, depth: int) -> str | None:
        """the table name a class declares as its ``schema = TableSchema(name=...)`` (or a module constant bound to
        one), or that a base class of the same module declares.

        :param owner: the class
        :ptype owner: ast.ClassDef
        :param depth: how many base classes the walk followed
        :ptype depth: int
        :return: the declared name, or None
        :rtype: str | None
        """
        for statement in owner.body:
            value: ast.expr | None = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                value = statement.value if isinstance(target, ast.Name) and target.id == "schema" else None
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                value = statement.value if statement.target.id == "schema" else None
            if isinstance(value, ast.Name):
                value = self._module_values.get(value.id)
            if isinstance(value, ast.Call) and _final(value) == "TableSchema":
                named = _keyword_value(value, "name")
                return _constant_text(named, self._envs[0]) if named is not None else None
        if depth > 3:
            return None
        bases = {base.id for base in owner.bases if isinstance(base, ast.Name)}
        for node in self._tree.body:
            if isinstance(node, ast.ClassDef) and node.name in bases:
                found = self._class_schema_name(node, depth + 1)
                if found is not None:
                    return found
        return None

    def _produced_by(self, node: ast.expr) -> str | None:
        """what an expression reads as when it names a relation judged elsewhere, where the walk is: a declared
        transient producer's return (a change set), or a keyed helper's relation parameter (judged at its call sites);
        a collection's own schema name (``self.schema.name``) reads as the table its class declares.

        :param node: the expression
        :ptype node: ast.expr
        :return: :data:`TRANSIENT_PRODUCED`, :data:`HELPER_RELATION`, the declared table name, a bounded copy relation's
            cut (:meth:`Bounded.relation`), or None for any other expression
        :rtype: str | None
        """
        declared = self._schema_name_text(node)
        if declared is not None:
            return declared
        if self._bounded.key and isinstance(node, ast.Call) and _final(node) in _COPY_CALLS:
            # a copy accessor's relation, cut by a checked bounded producer's key (``limited_snapshot``): the cut it
            # reads, one the producer leaves whole as the copy relation it is
            cut = self._bounded.relation(_placeholder(node, self._envs[-1]))
            if cut is not None:
                return cut
        if self._helper_change_parameter(node):
            return TRANSIENT_PRODUCED
        if self._helper_relation(node):
            return HELPER_RELATION
        owner = self._class_nodes[-1] if self._class_nodes else None
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and owner is not None
        ):
            return TRANSIENT_PRODUCED if node.attr in self._attributes_produced(owner) else None
        return TRANSIENT_PRODUCED if self._produces(node, self._function_node, owner) else None

    def _helper_relation(self, node: ast.expr, depth: int = 0) -> bool:
        """whether an expression names a keyed helper's relation parameter as the relation a statement reads or
        writes: through :data:`_RELATION_CALLS` anywhere (of the parameter, or of one value of a mapping parameter);
        a bare or quoted parameter only right after ``FROM`` /
        ``JOIN`` / ``UPDATE`` / ``INTO``; a local name the function binds once to a :data:`_RELATION_CALLS` call of
        one (``target = quoted_table(table)``, ``source = statement.source(rel, "t")``) anywhere.

        :param node: the expression (an f-string placeholder's)
        :ptype node: ast.expr
        :param depth: how many local bindings the walk followed
        :ptype depth: int
        :return: whether it is
        :rtype: bool
        """
        function = self._function_node
        if function is None or depth > 3 or not (self._helper_parameters or self._helper_fields):
            return False
        place = node
        named = False
        while isinstance(node, ast.Call) and node.args:
            callee = _final(node)
            if callee in _RELATION_CALLS:
                named = True
            elif not (callee in _NAME_QUOTERS and len(node.args) == 1):
                break
            node = node.args[0]
        if named and isinstance(node, ast.Subscript):
            # one relation of a mapping the helper is handed (``relation(tables[label], "r")``)
            node = node.value
        owner = self._class_nodes[-1] if self._class_nodes else None
        held = (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and owner is not None
            and node.attr in self._helper_fields.get(id(owner), frozenset())
        ) or (isinstance(node, ast.Name) and node.id in self._helper_parameters.get(id(function), frozenset()))
        if held:
            return named or (depth == 0 and self._relation_position(place))
        if named or depth > 0 or not isinstance(node, ast.Name):
            return False
        bound = [
            statement.value
            for statement in ast.walk(function)
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == node.id
        ]
        return (
            len(bound) == 1
            and isinstance(bound[0], ast.Call)
            and _final(bound[0]) in _RELATION_CALLS
            and self._helper_relation(bound[0], depth + 1)
        )

    def _relation_position(self, node: ast.expr) -> bool:
        """whether an f-string placeholder's expression stands where the statement names its relation.

        :param node: the expression
        :ptype node: ast.expr
        :return: whether the text before it ends ``FROM`` / ``JOIN`` / ``UPDATE`` / ``INTO``
        :rtype: bool
        """
        formatted = self._parents.get(node)
        joined = self._parents.get(formatted) if isinstance(formatted, ast.FormattedValue) else None
        if not isinstance(formatted, ast.FormattedValue) or not isinstance(joined, ast.JoinedStr):
            return False
        index = joined.values.index(formatted)
        before = joined.values[index - 1] if index else None
        return (
            isinstance(before, ast.Constant)
            and isinstance(before.value, str)
            and _RELATION_POSITION.search(before.value) is not None
        )

    def _helper_change_parameter(self, node: ast.expr) -> bool:
        """whether an expression is a 3tears keyed helper's own parameter declared a change relation
        (``quoted_table(change_relation(stage))`` in ``apply_stage``): inside a helper of ``threetears/core/keyed``
        whose every call site the rule judges by the relation it is handed, so a caller handing it a data table is
        flagged where it calls; ``change_relation`` of a parameter anywhere else is no change set.

        :param node: the expression, through ``quoted_table`` / ``quote_identifier``
        :ptype node: ast.expr
        :return: whether it is
        :rtype: bool
        """
        while isinstance(node, ast.Call) and _final(node) in _QUOTING_CALLS and len(node.args) == 1:
            node = node.args[0]
        if not (isinstance(node, ast.Call) and _final(node) == "change_relation" and len(node.args) == 1):
            return False
        function = self._function_node
        if (
            function is None
            or function.name not in KEYED_HELPER_CALLS
            or not any(package in str(self.path) for package in _KEYED_PACKAGES)
        ):
            return False
        arguments = function.args
        named = {argument.arg for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]}
        return isinstance(node.args[0], ast.Name) and node.args[0].id in named

    def _produces(
        self, node: ast.expr, function: ast.FunctionDef | ast.AsyncFunctionDef | None, owner: ast.ClassDef | None
    ) -> bool:
        """whether an expression is a call of a declared transient producer: a module function by its name, or a
        method called on ``self`` or ``cls`` in its class, on the class itself, or on a receiver known to be the class.

        :param node: the expression
        :ptype node: ast.expr
        :param function: the calling function
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef | None
        :param owner: the calling class
        :ptype owner: ast.ClassDef | None
        :return: whether it is a producer's call
        :rtype: bool
        """
        if not self._producers or not isinstance(node, ast.Call):
            return False
        name = _final(node)
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        for producer_class, method in self._producers:
            if method != name:
                continue
            if not producer_class:
                if isinstance(node.func, ast.Name):
                    return True
                continue
            if isinstance(receiver, ast.Name) and (
                receiver.id == producer_class
                or (receiver.id in ("self", "cls") and owner is not None and owner.name == producer_class)
            ):
                return True
            if receiver is not None and self._receiver_class(node, function, owner) == producer_class:
                return True
        return False

    def _attributes_produced(self, owner: ast.ClassDef) -> frozenset[str]:
        """the ``self`` attributes of a class every assignment of which (in any of its methods) is a declared
        transient producer's return: ``self.new = work.put(rows)``.

        :param owner: the class
        :ptype owner: ast.ClassDef
        :return: the attributes' names
        :rtype: frozenset[str]
        """
        cached = self._produced_attributes.get(id(owner))
        if cached is not None:
            return cached
        produced: set[str] = set()
        other: set[str] = set()
        for method in owner.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(method):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                    if isinstance(node, (ast.AnnAssign, ast.AugAssign))
                    else []
                )
                value = getattr(node, "value", None)
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        made = (
                            isinstance(node, ast.Assign) and value is not None and self._produces(value, method, owner)
                        )
                        (produced if made else other).add(target.attr)
        found = frozenset(produced - other)
        self._produced_attributes[id(owner)] = found
        return found


def _annotated_class(annotation: ast.expr) -> str:
    """the class an annotation names: its final segment, ``None`` left out of a union (``Tool | None``); empty for
    anything else.

    :param annotation: the annotation
    :ptype annotation: ast.expr
    :return: the class's name, or empty
    :rtype: str
    """
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        left, right = _annotated_class(annotation.left), _annotated_class(annotation.right)
        return left or right
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value.rsplit(".", 1)[-1]
    if isinstance(annotation, (ast.Name, ast.Attribute)):
        name = (dotted(annotation) or "").rsplit(".", 1)[-1]
        return "" if name == "None" else name
    return ""


def _assigned_class(body: list[ast.stmt], matches: Callable[[ast.expr], bool]) -> str:
    """the class a target is annotated with or assigned an instance of anywhere in ``body`` (nested statements
    included): ``x: Tool``, ``x = Tool(...)``, ``self._x = Tool(...)``; empty when none says.

    :param body: the statements
    :ptype body: list[ast.stmt]
    :param matches: whether an assignment target is the receiver
    :ptype matches: Callable[[ast.expr], bool]
    :return: the class's name, or empty
    :rtype: str
    """
    known = ""
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # ``self.work = work`` where the method's parameter is annotated ``work: _Work``
                annotated = {
                    argument.arg: _annotated_class(argument.annotation)
                    for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                    if argument.annotation is not None
                }
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Assign)
                        and isinstance(inner.value, ast.Name)
                        and annotated.get(inner.value.id)
                        and any(matches(target) for target in inner.targets)
                    ):
                        known = annotated[inner.value.id]
            if isinstance(node, ast.AnnAssign) and matches(node.target):
                known = _annotated_class(node.annotation) or known
            elif (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and any(matches(target) for target in node.targets)
            ):
                name = (dotted(node.value.func) or "").rsplit(".", 1)[-1]
                known = name if name.lstrip("_")[:1].isupper() else known
    return known


def _self_attribute_bound(function: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> ast.Attribute | None:
    """the ``self`` attribute a function binds a name to, the one binding of it: ``work = self.work``, or a place in a
    tuple assignment ``work, settled = self.work, self.settled``.

    :param function: the function
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param name: the name
    :ptype name: str
    :return: the attribute, or None when the function binds the name to anything else, or more than once
    :rtype: ast.Attribute | None
    """
    bound: list[ast.expr | None] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                bound.append(node.value)
            elif isinstance(target, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == name for e in target.elts):
                values = node.value.elts if isinstance(node.value, ast.Tuple) else None
                index = next(i for i, e in enumerate(target.elts) if isinstance(e, ast.Name) and e.id == name)
                bound.append(values[index] if values is not None and len(values) == len(target.elts) else None)
    if len(bound) != 1:
        return None
    value = bound[0]
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "self":
        return value
    return None


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
        if not isinstance(mapping, ast.Dict):
            return None
        written = [key.value for key in mapping.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
        if not written:
            return None
        parts.extend(form.format(column) for column in written)
        if len(written) < len(mapping.keys):
            # sets spread in beside the written ones (``**part.filters``) only narrow: what the walk cannot see is an
            # interpolation, unknown unless a written column keys the read
            parts.append("{}")
    if given.get("not_null") is not None:
        parts.append("__not_null__ IS NOT NULL")
    return " AND ".join(parts) or None


def _literal_branches(value: ast.expr) -> bool:
    """whether an expression is one of two literal mappings or sequences by a condition (``{...} if c else {...}``).

    :param value: the expression
    :ptype value: ast.expr
    :return: whether it is
    :rtype: bool
    """
    if not isinstance(value, ast.IfExp):
        return False
    # a table named one way or another (a base class's method on each subclass's table): strings, chained
    arms = (value.body, value.orelse)
    if all(_string_arm(arm) for arm in arms):
        return True
    return isinstance(value.body, ast.Dict | ast.Tuple | ast.List) and isinstance(
        value.orelse, ast.Dict | ast.Tuple | ast.List
    )


def _string_arm(arm: ast.expr) -> bool:
    """whether an arm of a condition is a literal string, or another condition whose arms all are."""
    if isinstance(arm, ast.Constant):
        return isinstance(arm.value, str)
    return isinstance(arm, ast.IfExp) and _string_arm(arm.body) and _string_arm(arm.orelse)


def _arms(value: ast.IfExp) -> list[ast.expr]:
    """every arm of a condition, a chained condition's arms flattened."""
    found: list[ast.expr] = []
    for arm in (value.body, value.orelse):
        found.extend(_arms(arm) if isinstance(arm, ast.IfExp) and _string_arm(arm) else [arm])
    return found


def _none_test(test: ast.expr) -> tuple[str, bool] | None:
    """the name a condition asks is None or not (``x is None`` -> ``("x", False)``, ``x is not None`` -> ``("x",
    True)``: whether the condition holds when the name is not None); None for any other condition.

    :param test: the condition
    :ptype test: ast.expr
    :return: the name and whether the condition holds for a value that is not None, or None
    :rtype: tuple[str, bool] | None
    """
    if (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is | ast.IsNot)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    ):
        return test.left.id, isinstance(test.ops[0], ast.IsNot)
    return None


def _admits_none(argument: ast.arg) -> bool:
    """whether a parameter's annotation lets it hold None: none written, ``Any``, ``object``, or one naming
    ``None`` / ``Optional``.

    :param argument: the parameter
    :ptype argument: ast.arg
    :return: whether it may hold None
    :rtype: bool
    """
    if argument.annotation is None:
        return True
    text = ast.unparse(argument.annotation)
    return any(word in re.findall(r"\w+", text) for word in ("None", "Optional", "Any", "object"))


def _never_none_parameters(modules: list[tuple[str, ast.Module]]) -> frozenset[tuple[str, str]]:
    """for every function the repo defines once by its name, each parameter every call site passes something never
    None: a literal that is not None, or the calling function's own parameter, never rebound, whose annotation admits
    no None and which has no default of None; a site leaving the parameter to its default counts only for a default
    that is a literal other than None. A parameter no site calls is not among them.

    :param modules: each module's path and parse
    :ptype modules: list[tuple[str, ast.Module]]
    :return: ``(function, parameter)`` for each
    :rtype: frozenset[tuple[str, str]]
    """
    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for _, tree in modules:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _stub(node):
                definitions.setdefault(node.name, []).append(node)
    verdicts: dict[tuple[str, str], bool] = {}

    def never_none(value: ast.expr, caller: ast.FunctionDef | ast.AsyncFunctionDef | None) -> bool:
        if isinstance(value, ast.Constant):
            return value.value is not None
        if not isinstance(value, ast.Name) or caller is None:
            return False
        arguments = [*caller.args.posonlyargs, *caller.args.args, *caller.args.kwonlyargs]
        held = next((argument for argument in arguments if argument.arg == value.id), None)
        if held is None or value.id not in _function_bindings(caller, {}).parameters or _admits_none(held):
            return False
        defaults = dict(
            zip(
                [argument.arg for argument in [*caller.args.posonlyargs, *caller.args.args]][
                    len(caller.args.posonlyargs) + len(caller.args.args) - len(caller.args.defaults) :
                ],
                caller.args.defaults,
                strict=True,
            )
        )
        defaults.update(
            {
                argument.arg: default
                for argument, default in zip(caller.args.kwonlyargs, caller.args.kw_defaults, strict=True)
                if default is not None
            }
        )
        default = defaults.get(value.id)
        return default is None or not (isinstance(default, ast.Constant) and default.value is None)

    for _, tree in modules:
        # a module's or a class's own statements, then each function's: each scope its own
        definers = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        callers: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, list[ast.stmt]]] = [
            (None, [statement for statement in tree.body if not isinstance(statement, definers)])
        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                callers.append((None, [statement for statement in node.body if not isinstance(statement, definers)]))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                callers.append((node, node.body))
        for caller, body in callers:
            for call in _own_nodes(body):
                if not isinstance(call, ast.Call):
                    continue
                defined = definitions.get(_final(call) or "", [])
                if len(defined) != 1:
                    continue
                function = defined[0]
                positional = [*function.args.posonlyargs, *function.args.args]
                if positional[:1] and positional[0].arg in ("self", "cls") and isinstance(call.func, ast.Attribute):
                    positional = positional[1:]
                if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
                    keyword.arg is None for keyword in call.keywords
                ):
                    passed: dict[str, ast.expr | None] = {}
                    unreadable = True
                else:
                    passed = dict(zip([argument.arg for argument in positional], call.args, strict=False))
                    passed.update({keyword.arg: keyword.value for keyword in call.keywords if keyword.arg})
                    unreadable = False
                defaults = dict(
                    zip(
                        [argument.arg for argument in positional][len(positional) - len(function.args.defaults) :],
                        function.args.defaults,
                        strict=False,
                    )
                )
                defaults.update(
                    {
                        argument.arg: default
                        for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults, strict=True)
                        if default is not None
                    }
                )
                for argument in [*positional, *function.args.kwonlyargs]:
                    value = passed.get(argument.arg)
                    if unreadable:
                        ok = False
                    elif value is not None:
                        ok = never_none(value, caller)
                    else:
                        default = defaults.get(argument.arg)
                        ok = isinstance(default, ast.Constant) and default.value is not None
                    key = (function.name, argument.arg)
                    verdicts[key] = verdicts.get(key, True) and ok
    return frozenset(key for key, ok in verdicts.items() if ok)


def _branch_calls(call: ast.Call) -> list[ast.Call]:
    """a helper call once for each arm of every argument that is one literal or another by a condition, every
    combination, so each statement the call may send is judged; the call itself when it has none.

    :param call: the call, its arguments resolved
    :ptype call: ast.Call
    :return: the calls
    :rtype: list[ast.Call]
    """
    calls = [call]
    for index, argument in enumerate(call.args):
        if _literal_branches(argument):
            assert isinstance(argument, ast.IfExp)
            calls = [
                ast.Call(
                    func=each.func, args=[*each.args[:index], arm, *each.args[index + 1 :]], keywords=each.keywords
                )
                for each in calls
                for arm in _arms(argument)
            ]
    for index, keyword in enumerate(call.keywords):
        if _literal_branches(keyword.value):
            assert isinstance(keyword.value, ast.IfExp)
            calls = [
                ast.Call(
                    func=each.func,
                    args=each.args,
                    keywords=[
                        *each.keywords[:index],
                        ast.keyword(arg=keyword.arg, value=arm),
                        *each.keywords[index + 1 :],
                    ],
                )
                for each in calls
                for arm in _arms(keyword.value)
            ]
    return calls


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

#: a CTE one of a module's strings defines, as a statement assembled from several of its strings writes it: after
#: ``WITH``, a comma or the string's start, its name perhaps templated (``o_{race} AS (``), its materialization
#: perhaps interpolated (``stage0 AS {inlined}(``)
_CTE_PART_DEFINED: Final[re.Pattern[str]] = re.compile(
    r"(?:\bWITH(?:\s+RECURSIVE)?|,|^)\s*([A-Za-z_](?:\w|\{\})*)\s*(?:\([^()]*\)\s*)?AS\s*(?:\{\}\s*)?"
    r"(?:NOT\s+)?(?:MATERIALIZED\s+)?\(",
    re.IGNORECASE,
)


def _defined_ctes(node: ast.AST, env: dict[str, str]) -> set[str]:
    """the CTE names a definition's strings define (:data:`_CTE_PART_DEFINED`), lower-cased, its constants resolved
    (``f"WITH {_NEEDLE_READ} AS MATERIALIZED ("``), a templated one's interpolations dropped (``o_{race}`` as ``o_``)."""
    names: set[str] = set()
    for inner in ast.walk(node):
        text = (
            inner.value
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
            else _string_text(inner, env)
            if isinstance(inner, ast.JoinedStr)
            else None
        )
        for match in _CTE_PART_DEFINED.finditer(text or ""):
            name = match.group(1).lower()
            names.add(name.replace("{}", "") if "{}" in name else name)
    return names


def _statement_ctes(tree: ast.Module, env: dict[str, str]) -> dict[int, frozenset[str]]:
    """for each of a module's functions and methods, the CTE names the statements it takes part in define: its own,
    and those of every definition it is joined to by a call in the module (a bare call of a module function, a
    method called on ``self`` or ``cls`` in its class), either way. One statement is assembled across them
    (``WITH stage0 AS (...), {self._fixed()}, base AS (SELECT * FROM fixed)``, ``_fixed`` reading ``stage0``), so
    each reads the others' CTEs by name; a definition no call joins to one defining a CTE reads it as a table.

    :param tree: the module
    :ptype tree: ast.Module
    :param env: the module's names bound to strings
    :ptype env: dict[str, str]
    :return: each definition's node id onto the names
    :rtype: dict[int, frozenset[str]]
    """
    definitions = _definitions(tree)
    parent = {id(function): id(function) for function, _ in definitions}

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    functions = {function.name: function for function, owner in definitions if owner is None}
    methods = {(owner.name, function.name): function for function, owner in definitions if owner is not None}
    for function, owner in definitions:
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            callee: ast.FunctionDef | ast.AsyncFunctionDef | None = None
            if isinstance(node.func, ast.Name):
                callee = functions.get(node.func.id)
            elif (
                owner is not None
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"self", "cls"}
            ):
                callee = methods.get((owner.name, node.func.attr))
            if callee is not None:
                parent[root(id(callee))] = root(id(function))
    own = {id(function): _defined_ctes(function, env) for function, _ in definitions}
    joined: dict[int, set[str]] = {}
    for function, _ in definitions:
        joined.setdefault(root(id(function)), set()).update(own[id(function)])
    return {id(function): frozenset(joined[root(id(function))]) for function, _ in definitions}


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


def _calls_marked(call: _CallSite, target: tuple[str, str]) -> bool:
    """whether a call that may reach an entry-only marked function is known to: a ``self`` or bare call resolved to
    it, or a method call on a receiver known to be its class. A receiver of unknown type that only shares the method's
    name (``self._db.execute`` on a database connection) is no caller of it.

    :param call: the call
    :ptype call: _CallSite
    :param target: the marked function, ``(path, qualified name)``
    :ptype target: tuple[str, str]
    :return: whether it is a caller
    :rtype: bool
    """
    owner = target[1].rsplit(".", 1)[0] if "." in target[1] else ""
    return call.receiver in ("self", "bare") or (bool(owner) and call.receiver_class == owner.rsplit(".", 1)[-1])


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
    # a bare call to a function defined in the caller, or in a function enclosing it, reaches that function: the
    # name is the nearest enclosing function scope's (a class body's names are not in its methods' scope)
    nested = [
        (call.path, f"{scope}.{call.name}")
        for scope in (call.function.rsplit(".", depth)[0] for depth in range(call.function.count(".") + 1))
        if call.receiver == "bare" and scope != call.owner and scope not in (module_classes or {}).get(call.path, set())
    ]
    local = next((target for target in nested if target in by_name.get(call.name, [])), None)
    if local is not None:
        targets = [local]
    elif call.receiver == "self" and call.owner:
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
        # a receiver known to be one of the repo's classes reaches that class's method alone
        typed = [
            (path, name)
            for path, name in targets
            if call.receiver_class and name.split(".", 1)[0] == call.receiver_class
        ]
        targets = typed or targets
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
    constant_uses: set[tuple[str, str, int, int]] | None = None,
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
    # a whole read written as a module constant is judged where the constant is written (the module's); the function
    # running it reads whole all the same, so its callers are named too
    at_module: dict[str, list[WholeReadHit]] = {}
    for hit in hits:
        if hit.function == "<module>":
            at_module.setdefault(hit.path, []).append(hit)
    constant_reach: set[tuple[str, str]] = set()
    for path, function, first, last in sorted(constant_uses or ()):
        for hit in at_module.get(path, []):
            if first <= hit.line <= last and (path, function) not in backstops:
                constant_reach.add((path, function))
                if hit.kind in _READ_KINDS:
                    readers.setdefault((path, function), hit)
    constructors_found: set[int] = set()
    resolved: list[tuple[_CallSite, list[tuple[str, str]]]] = []
    for index, call in enumerate(calls):
        targets = _targets(call, module_functions, by_name, imports, module_classes)
        if targets is None:
            constructors_found.add(index)
        resolved.append((call, targets or []))
    reaches = ({(hit.path, hit.function) for hit in hits} | constant_reach) - backstops
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
        if target in marks and marks.get((call.path, call.function)) != marks[target] and _calls_marked(call, target)
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
            found.append(
                WholeReadHit(
                    call.path, call.line, call.function, f"whole-call {call.name}", call.spelled, tuple(sorted(targets))
                )
            )
        if call.whole_flags and (not targets or any(target in reaches for target in targets)):
            found.extend(
                WholeReadHit(call.path, call.line, call.function, f"whole-flag {flag}=", call.spelled)
                for flag in call.whole_flags
            )
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


#: the calls that run their statement on a local backend (DuckDB, SQLite): nothing they read is hashed
_LOCAL_EXEC_CALLS: Final[frozenset[str]] = frozenset({"execute_query", "execute_arrow"})


def _run_locally(parent: ast.AST | None) -> bool:
    """whether a statement is the argument of a call that runs it on a local backend (``l1.execute_query(sql)``).

    :param parent: the statement's parent node
    :ptype parent: ast.AST | None
    :return: whether it is
    :rtype: bool
    """
    return isinstance(parent, ast.Call) and _final(parent) in _LOCAL_EXEC_CALLS


def _on_an_attribute(call: ast.Call) -> bool:
    """whether ``call`` is a method of an object an attribute holds (``self._store.f()``), never a module's function
    (``keyed.f()``, ``f()``).

    :param call: the call
    :ptype call: ast.Call
    :return: whether it is
    :rtype: bool
    """
    return isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Attribute)


def _on_an_object(call: ast.Call, module_names: frozenset[str]) -> bool:
    """whether ``call`` is a method of an object a plain name holds (``readings.read_by(clause)``, a parameter or a
    local), never a function of a module the file imports (``keyed.read_by(...)``).

    :param call: the call
    :ptype call: ast.Call
    :param module_names: the names the module's imports bind (:func:`_imported_names`)
    :ptype module_names: frozenset[str]
    :return: whether it is
    :rtype: bool
    """
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id not in module_names
    )


def _imported_names(tree: ast.Module) -> frozenset[str]:
    """the names a module's imports bind (``keyed`` for ``from threetears.core import keyed``): a call on one is a
    module's function, never a method the repo defines under the same name.

    :param tree: the module
    :ptype tree: ast.Module
    :return: the names
    :rtype: frozenset[str]
    """
    names: set[str] = set()
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Import):
            names.update((alias.asname or alias.name).split(".")[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in statement.names)
    return frozenset(names)


def _bound_name(value: ast.expr) -> tuple[str, str | None] | None:
    """the name an argument reads, and the attribute it reads of it: ``table`` -> ``("table", None)``,
    ``table.name`` -> ``("table", "name")``; None for anything else.

    :param value: the argument
    :ptype value: ast.expr
    :return: the name and attribute, or None
    :rtype: tuple[str, str | None] | None
    """
    if isinstance(value, ast.Name):
        return value.id, None
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
        return value.value.id, value.attr
    return None


def _argument_names(arguments: list[ast.expr]) -> dict[str, tuple[str, str | None]]:
    """each name, or attribute of a name, a call's arguments read at any depth (``table``, ``table.name``,
    ``pk_cols`` in ``{c: data[c] for c in pk_cols}``, ``key`` in ``KeyBatch((key,), ...)``), by its spelling; a name a
    comprehension inside the argument binds is its own, never read.

    :param arguments: the arguments
    :ptype arguments: list[ast.expr]
    :return: spelling onto the name and attribute
    :rtype: dict[str, tuple[str, str | None]]
    """
    found: dict[str, tuple[str, str | None]] = {}
    for argument in arguments:
        bound = {
            target.id
            for node in ast.walk(argument)
            if isinstance(node, ast.comprehension)
            for target in ast.walk(node.target)
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(argument):
            read = _bound_name(node) if isinstance(node, (ast.Name, ast.Attribute)) else None
            if read is not None and read[0] not in bound:
                found.setdefault(ast.unparse(node), read)
    return found


def _passed_nodes(passed: list[Passed], attribute: str | None) -> list[ast.expr]:
    """what each value one site (or one loop) passes reads as in a call: a string, a declared object's ``attribute``,
    a rebuilt literal; empty when any one cannot be read.

    :param passed: the values
    :ptype passed: list[Passed]
    :param attribute: the attribute read of each, or None for the value itself
    :ptype attribute: str | None
    :return: the expressions, in order, without repeats
    :rtype: list[ast.expr]
    """
    nodes: dict[str, ast.expr] = {}
    for value in passed:
        node: ast.expr | None = None
        if attribute is None and isinstance(value, str):
            node = ast.Constant(value)
        elif attribute is None and isinstance(value, Spelled):
            node = value.node()
        elif attribute is not None and isinstance(value, tuple):
            text = dict(value).get(attribute)
            node = None if text is None else ast.Constant(text)
        if node is None:
            return []
        nodes.setdefault(ast.unparse(node), node)
    return list(nodes.values())


def _copied(node: ast.expr) -> ast.expr:
    """a fresh copy of an argument, so substituting in it never touches the module's own tree.

    :param node: the argument
    :ptype node: ast.expr
    :return: the copy
    :rtype: ast.expr
    """
    return ast.parse(ast.unparse(node), mode="eval").body


class _Substituted(ast.NodeTransformer):
    """an argument with each name (or attribute) a site passes replaced by what it passes; a name a comprehension
    binds is left as it is."""

    def __init__(self, given: dict[str, ast.expr]) -> None:
        """:param given: spelling onto the expression put in its place
        :ptype given: dict[str, ast.expr]
        """
        self._given = given
        self._bound: list[set[str]] = []

    def _replaced(self, node: ast.expr) -> ast.expr | None:
        read = _bound_name(node)
        if read is None or any(read[0] in bound for bound in self._bound):
            return None
        put = self._given.get(ast.unparse(node))
        return None if put is None else _copied(put)

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """a name a site passes."""
        return self._replaced(node) or node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        """an attribute of a name a site passes, else its parts."""
        replaced = self._replaced(node)
        return replaced if replaced is not None else self.generic_visit(node)  # type: ignore[return-value]

    def _comprehension(self, node: ast.AST) -> ast.AST:
        generators = getattr(node, "generators", [])
        for generator in generators:
            generator.iter = self.visit(generator.iter)
        self._bound.append(
            {
                target.id
                for generator in generators
                for target in ast.walk(generator.target)
                if isinstance(target, ast.Name)
            }
        )
        for field_name in ("key", "value", "elt"):
            if hasattr(node, field_name):
                setattr(node, field_name, self.visit(getattr(node, field_name)))
        for generator in generators:
            generator.ifs = [self.visit(test) for test in generator.ifs]
        self._bound.pop()
        return node

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        """a mapping comprehension: its own names kept."""
        return self._comprehension(node)

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        """a list comprehension: its own names kept."""
        return self._comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        """a set comprehension: its own names kept."""
        return self._comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        """a generator expression: its own names kept."""
        return self._comprehension(node)


class _LiteralFolder(ast.NodeTransformer):
    """a mapping comprehension over a literal tuple of strings read as the mapping it builds
    (``{c: data[c] for c in ("a", "b")}`` -> ``{"a": data["a"], "b": data["b"]}``), and a literal mapping spread into
    another (``{**{"a": x}, "b": y}``) read as its entries."""

    def visit_DictComp(self, node: ast.DictComp) -> ast.expr:
        """the mapping a comprehension over a literal tuple builds."""
        self.generic_visit(node)
        if len(node.generators) != 1:
            return node
        generator = node.generators[0]
        over = generator.iter
        if (
            generator.ifs
            or generator.is_async
            or not isinstance(generator.target, ast.Name)
            or not isinstance(over, (ast.Tuple, ast.List))
            or not all(isinstance(each, ast.Constant) and isinstance(each.value, str) for each in over.elts)
        ):
            return node
        keys: list[ast.expr | None] = []
        values: list[ast.expr] = []
        for each in over.elts:
            given = {generator.target.id: each}
            keys.append(_Substituted(given).visit(_copied(node.key)))
            values.append(_Substituted(given).visit(_copied(node.value)))
        return ast.Dict(keys=keys, values=values)

    def visit_Dict(self, node: ast.Dict) -> ast.expr:
        """a mapping with literal mappings spread into it, as their entries."""
        self.generic_visit(node)
        keys: list[ast.expr | None] = []
        values: list[ast.expr] = []
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None and isinstance(value, ast.Dict):
                keys.extend(value.keys)
                values.extend(value.values)
            else:
                keys.append(key)
                values.append(value)
        return ast.Dict(keys=keys, values=values)


def _folded_literals(node: ast.expr) -> ast.expr:
    """``node`` with its comprehensions over literal tuples and its spread literal mappings read as literals
    (:class:`_LiteralFolder`).

    :param node: the argument
    :ptype node: ast.expr
    :return: the folded argument
    :rtype: ast.expr
    """
    folded = _LiteralFolder().visit(node)
    assert isinstance(folded, ast.expr)
    return folded


def _texts_of(passed: list[Passed], attribute: str | None) -> list[str] | None:
    """the strings ``passed`` reads as: each string itself, or each declared object's ``attribute``; None when any one
    does not read as a string.

    :param passed: what is passed
    :ptype passed: list[Passed]
    :param attribute: the attribute read of it, or None for the value itself
    :ptype attribute: str | None
    :return: the strings, or None
    :rtype: list[str] | None
    """
    texts: list[str] = []
    for value in passed:
        if attribute is None:
            text = value if isinstance(value, str) else None
        else:
            text = dict(value).get(attribute) if isinstance(value, tuple) else None
        if text is None:
            return None
        texts.append(text)
    return texts


@dataclass(frozen=True)
class _Bindings:
    """what a function's own names hold: its parameters never rebound, and its names bound only by a loop over a
    module tuple (``for table in TABLES``) onto that tuple's elements."""

    parameters: frozenset[str]
    loops: dict[str, list[Passed]]
    tuples: dict[str, list[Passed]] = field(default_factory=dict)


def _function_bindings(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    tuples: dict[str, list[Passed]],
    declared: Mapping[str, tuple[list[Passed], str]] | None = None,
) -> _Bindings:
    """the parameters and loop names of ``function`` whose values the walk can follow: a declared parameter
    (``declared_parameters``) holds each of its values when it is a string, and a loop over it binds each of them.

    :param function: the function
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param tuples: the module tuples it may loop over
    :ptype tuples: dict[str, list[Passed]]
    :param declared: its declared names onto their values, and how each holds them: ``one`` (a string parameter),
        ``many`` (a collection parameter a loop over binds each), ``loop`` (a loop target, each value in turn)
    :ptype declared: Mapping[str, tuple[list[Passed], str]] | None
    :return: its bindings
    :rtype: _Bindings
    """
    declared = declared or {}
    tuples = {**tuples, **{name: list(values) for name, (values, how) in declared.items() if how == "many"}}
    stores: dict[str, int] = {}
    for node in _own_nodes(function.body):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores[node.id] = stores.get(node.id, 0) + 1
    # a name every binding of which is a loop over known elements (``for table in tables`` twice): each of them
    looped_over: dict[str, list[list[Passed] | None]] = {}
    for node in _own_nodes(function.body):
        looped = (
            (node.target, node.iter) if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) else (None, None)
        )
        target, over = looped
        if isinstance(target, ast.Name) and over is not None:
            looped_over.setdefault(target.id, []).append(_iterated(over, tuples))
    loops: dict[str, list[Passed]] = {}
    for name, iterations in looped_over.items():
        if stores.get(name) == len(iterations) and all(elements is not None for elements in iterations):
            union: list[Passed] = []
            for elements in iterations:
                union += [element for element in elements or [] if element not in union]
            loops[name] = union
    for name, (values, how) in declared.items():
        if (how == "one" and name not in stores) or how == "loop":
            loops[name] = list(values)
    arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    return _Bindings(
        parameters=frozenset(argument.arg for argument in arguments if argument.arg not in stores),
        loops=loops,
        tuples=tuples,
    )


def _iterated(over: ast.expr, tuples: dict[str, list[Passed]]) -> list[Passed] | None:
    """the elements a loop over ``over`` binds: a module tuple's; an empty literal's (none); either branch's of a
    conditional (``() if behind else TABLES``); None for anything else.

    :param over: what the loop iterates
    :ptype over: ast.expr
    :param tuples: the module tuples
    :ptype tuples: dict[str, list[Passed]]
    :return: the elements, or None
    :rtype: list[Passed] | None
    """
    if isinstance(over, ast.Name):
        found = tuples.get(over.id)
        return None if found is None else list(found)
    if isinstance(over, (ast.Tuple, ast.List)) and not over.elts:
        return []
    if isinstance(over, ast.IfExp):
        body, orelse = _iterated(over.body, tuples), _iterated(over.orelse, tuples)
        return None if body is None or orelse is None else [*body, *orelse]
    if isinstance(over, ast.Call) and not over.keywords:
        # ``sorted(tables)``, ``list(tables)``, ``tables.keys()``: the same elements
        if isinstance(over.func, ast.Name) and over.func.id in {"sorted", "list", "tuple"} and len(over.args) == 1:
            return _iterated(over.args[0], tuples)
        if isinstance(over.func, ast.Attribute) and over.func.attr == "keys" and not over.args:
            return _iterated(over.func.value, tuples)
    return None


def _passed_values(
    function: str, name: str, bindings: _Bindings, found: dict[tuple[str, str], list[Passed]]
) -> list[Passed] | None:
    """what ``name`` holds in ``function``: a loop's elements, or what every caller passes the parameter; None when
    the walk cannot follow it.

    :param function: the function's name
    :ptype function: str
    :param name: the name
    :ptype name: str
    :param bindings: the function's bindings
    :ptype bindings: _Bindings
    :param found: what each parameter is passed, as followed so far
    :ptype found: dict[tuple[str, str], list[Passed]]
    :return: the values, or None
    :rtype: list[Passed] | None
    """
    if name in bindings.loops:
        return bindings.loops[name]
    if name in bindings.parameters:
        passed = found.get((function, name))
        if passed and all(value is not None for value in passed):
            return passed
    return None


def _site_values(
    value: ast.expr,
    known: dict[str, str],
    caller: tuple[str, _Bindings] | None,
    found: dict[tuple[str, str], list[Passed]],
) -> list[Passed]:
    """what one call site passes: a literal or a module constant; a name a loop over a module tuple binds (its
    elements) or an attribute of one (``table.name``); the caller's own parameter or its attribute, as its callers
    pass it; ``[None]`` for anything else.

    :param value: the argument
    :ptype value: ast.expr
    :param known: the calling module's strings
    :ptype known: dict[str, str]
    :param caller: the calling function's name and bindings, None at a module's or class's top level
    :ptype caller: tuple[str, _Bindings] | None
    :param found: what each parameter is passed, as followed so far
    :ptype found: dict[tuple[str, str], list[Passed]]
    :return: the values
    :rtype: list[Passed]
    """
    text = _constant_text(value, known)
    if text is not None:
        return [text]
    spelled = _spelled_literal(value, known, caller)
    if spelled is not None:
        return [spelled]
    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and caller is not None:
        looked = caller[1].tuples.get(f"{value.value.id}[]")
        return [None] if looked is None else list(looked)
    named = _bound_name(value)
    if named is None or caller is None:
        return [None]
    passed = _passed_values(caller[0], named[0], caller[1], found)
    if passed is None:
        return [None]
    if named[1] is None:
        return list(passed)
    texts = _texts_of(passed, named[1])
    return [None] if texts is None else list(texts)


def _mapped(shaped: ast.expr, literal: Callable[[ast.expr], ast.expr]) -> ast.expr:
    """each mapping of a conditional of mappings (:meth:`_ModuleWalk._local_mapping`) with its keys as literals.

    :param shaped: the mapping, or a conditional of them
    :ptype shaped: ast.expr
    :param literal: what writes a mapping's known keys as literals
    :ptype literal: Callable[[ast.expr], ast.expr]
    :return: the same shape, keys as literals
    :rtype: ast.expr
    """
    if isinstance(shaped, ast.IfExp):
        return ast.IfExp(test=shaped.test, body=_mapped(shaped.body, literal), orelse=_mapped(shaped.orelse, literal))
    return literal(shaped)


def _qualified_names(tree: ast.Module) -> dict[int, str]:
    """each function of a module onto its qualified name as the walk spells it (``Class.method``,
    ``outer.inner``).

    :param tree: the module
    :ptype tree: ast.Module
    :return: ``id`` of each function node onto its name
    :rtype: dict[int, str]
    """
    names: dict[int, str] = {}

    def visit(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names[id(node)] = ".".join((*prefix, node.name))
                visit(node.body, (*prefix, node.name))
            elif isinstance(node, ast.ClassDef):
                visit(node.body, (*prefix, node.name))
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.stmt):
                        visit([child], prefix)

    visit(tree.body, ())
    return names


def _spelled_literal(value: ast.expr, known: dict[str, str], caller: tuple[str, _Bindings] | None) -> Spelled | None:
    """what a call site passes as a literal a keyed helper reads, rebuilt in its own terms: a tuple or list of
    strings, or a module tuple of them (``SCHEDULE_PK``); a mapping whose every key is a string (none, ``{}``), its
    values kept where literals and opaque otherwise; None for anything else.

    :param value: the argument
    :ptype value: ast.expr
    :param known: the calling module's strings
    :ptype known: dict[str, str]
    :param caller: the calling function's name and bindings, None at a top level
    :ptype caller: tuple[str, _Bindings] | None
    :return: the literal, or None
    :rtype: Spelled | None
    """
    texts: list[str | None] | None = None
    if isinstance(value, (ast.Tuple, ast.List)) and value.elts:
        texts = [_constant_text(element, known) for element in value.elts]
    elif isinstance(value, ast.Name) and caller is not None and value.id in caller[1].tuples:
        held = caller[1].tuples[value.id]
        texts = [each if isinstance(each, str) else None for each in held] if held else None
    if texts is not None:
        if any(text is None for text in texts):
            return None
        return Spelled(ast.unparse(ast.Tuple(elts=[ast.Constant(text) for text in texts], ctx=ast.Load())))
    if isinstance(value, ast.Dict):
        keys = [None if key is None else _constant_text(key, known) for key in value.keys]
        if any(key is None for key in keys):
            return None
        values: list[ast.expr] = [
            each if isinstance(each, ast.Constant) else ast.Name(id=_PASSED_VALUE, ctx=ast.Load())
            for each in value.values
        ]
        return Spelled(ast.unparse(ast.Dict(keys=[ast.Constant(key) for key in keys], values=values)))
    return None


@dataclass(frozen=True)
class MatchBuilder:
    """a repo function returning a step's ``Match`` list (``on=``): one ``Match`` of each column its argument names
    (``_alike(columns)``: ``parameter`` its position), or a fixed list (``columns``, None for one named at run time),
    the first of a returned pair where ``first`` (``on, also = let_go_on(table)``).

    :ivar parameter: the position of the parameter naming the columns, or None for a fixed list
    :ivar columns: a fixed list's columns
    :ivar first: whether the list is the first of a returned tuple
    """

    parameter: int | None
    columns: tuple[str | None, ...] = ()
    first: bool = False


def _match_list(
    value: ast.expr, function: ast.FunctionDef | ast.AsyncFunctionDef, known: dict[str, str]
) -> MatchBuilder | None:
    """what one returned expression says of the ``Match`` list a function returns."""
    if isinstance(value, ast.Name):
        bound = [
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == value.id
        ]
        return (
            _match_list(bound[0], function, known) if len(bound) == 1 and not isinstance(bound[0], ast.Name) else None
        )
    if isinstance(value, ast.Tuple) and value.elts and not isinstance(value.elts[0], ast.Tuple):
        inner = _match_list(value.elts[0], function, known)
        return None if inner is None or inner.first else MatchBuilder(inner.parameter, inner.columns, first=True)
    if isinstance(value, ast.ListComp) and len(value.generators) == 1:
        generator = value.generators[0]
        element = value.elt
        positional = [argument.arg for argument in [*function.args.posonlyargs, *function.args.args]]
        if (
            isinstance(element, ast.Call)
            and _final(element) == "Match"
            and element.args
            and isinstance(element.args[0], ast.Name)
            and isinstance(generator.target, ast.Name)
            and element.args[0].id == generator.target.id
            and isinstance(generator.iter, ast.Name)
            and generator.iter.id in positional
        ):
            return MatchBuilder(positional.index(generator.iter.id))
        return None
    if isinstance(value, ast.List) and value.elts:
        columns: list[str | None] = []
        for element in value.elts:
            if not (isinstance(element, ast.Call) and _final(element) == "Match" and element.args):
                return None
            columns.append(_constant_text(element.args[0], known))
        return MatchBuilder(None, tuple(columns))
    return None


def _match_builders(
    modules: list[tuple[str, ast.Module]], strings: dict[str, dict[str, str]]
) -> dict[str, MatchBuilder]:
    """every module function of the repo, defined once by its name, that returns a step's ``Match`` list
    (:class:`MatchBuilder`): ``on=`` built by one is judged by the columns it names.

    :param modules: each module's path and parse
    :ptype modules: list[tuple[str, ast.Module]]
    :param strings: every walked module's path onto its once-bound strings (:func:`_module_strings`)
    :ptype strings: dict[str, dict[str, str]]
    :return: the function's name onto what it builds
    :rtype: dict[str, MatchBuilder]
    """
    found: dict[str, MatchBuilder | None] = {}
    for relative, tree in modules:
        known = {**_imported_strings(tree, strings), **strings.get(relative, {})}
        for statement in tree.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            returned = [node.value for node in ast.walk(statement) if isinstance(node, ast.Return) and node.value]
            built = _match_list(returned[0], statement, known) if len(returned) == 1 else None
            if built is None:
                continue
            found[statement.name] = built if statement.name not in found else None
    return {name: built for name, built in found.items() if built is not None}


def _parameter_values(
    modules: list[tuple[str, ast.Module]],
    strings: dict[str, dict[str, str]],
    tuples: dict[str, dict[str, list[Passed]]] | None = None,
    sites: dict[str, list[Site]] | None = None,
    declared: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[tuple[str, str], list[Passed]]:
    """for every function the repo defines once by its name, what each call site passes each of its parameters: a
    literal, or a module constant of the calling module (its own, or one it imports from the repo); a default the
    call leaves in place reads the same way; the elements of a module tuple a loop binds the argument to (or the
    attribute of each it reads); the calling function's own parameter, as its own callers pass it, followed
    :data:`_PARAMETER_DEPTH` calls up; None for anything else.

    :param modules: each module's path and parse
    :ptype modules: list[tuple[str, ast.Module]]
    :param strings: every walked module's path onto its once-bound strings (:func:`_module_strings`)
    :ptype strings: dict[str, dict[str, str]]
    :param tuples: every walked module's path onto its tuples (:func:`_module_tuples`); none when None
    :ptype tuples: dict[str, dict[str, list[Passed]]] | None
    :param sites: filled, when given, with each function's call sites, one at a time: what each passes each
        parameter, so a call is judged by one site's values together, never one site's crossed with another's
    :ptype sites: dict[str, list[Site]] | None
    :param declared: the parameters the repo declares the values of (``declared_parameters``): a site passing one,
        or a name a loop over one binds, passes each value
    :ptype declared: Mapping[str, tuple[str, ...]] | None
    :return: ``(function, parameter)`` onto each site's value, or None
    :rtype: dict[tuple[str, str], list[Passed]]
    """
    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for _, tree in modules:
        for node in ast.walk(tree):
            # a protocol's stub (a docstring and ``...``) declares a signature; only a body says what it does
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _stub(node):
                definitions.setdefault(node.name, []).append(node)
    constructors = _constructors(modules)
    contexts: list[tuple[dict[str, str], tuple[str, _Bindings] | None, list[ast.Call], tuple[str, str]]] = []
    for relative, tree in modules:
        known = {**_imported_strings(tree, strings), **strings.get(relative, {})}
        imported = _imported_names(tree)
        module_tuples = _imported_tuples(tree, relative, tuples or {})
        definers = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        # a module's or a class's own statements: each function's are its own scope, with its bindings
        scopes: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, list[ast.stmt]]] = [
            (None, [statement for statement in tree.body if not isinstance(statement, definers)])
        ]
        qualified = _qualified_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                scopes.append((None, [statement for statement in node.body if not isinstance(statement, definers)]))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scopes.append((node, node.body))
        for function, body in scopes:
            where = (relative, "<module>" if function is None else qualified.get(id(function), function.name))
            caller = (
                None
                if function is None
                else (
                    function.name,
                    _function_bindings(function, module_tuples, _declared_of(function, where[1], declared or {})),
                )
            )
            # a call on an imported module (``keyed.holds_any``) is that module's, not a method of the same name
            calls = [
                node
                for node in _own_nodes(body)
                if isinstance(node, ast.Call)
                and not (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in imported
                )
            ]
            contexts.append((known, caller, calls, where))
    found: dict[tuple[str, str], list[Passed]] = {}
    for _ in range(_PARAMETER_DEPTH):
        followed: dict[tuple[str, str], list[Passed]] = {}
        followed_sites: dict[str, list[Site]] = {}
        for known, caller, calls, where in contexts:
            for call in calls:
                callee = _final(call) or ""
                defined = definitions.get(callee, [])
                if not defined and callee in constructors:
                    # a construction of a repo class: what it passes its ``__init__``, kept as ``<Class>.__init__``
                    _record_site(
                        call,
                        constructors[callee],
                        known,
                        caller,
                        found,
                        followed,
                        name=f"{callee}.__init__",
                        sites=followed_sites,
                        where=where,
                    )
                    continue
                if len(defined) != 1:
                    # a name defined twice could be either: no site of it says what its parameters hold
                    continue
                _record_site(call, defined[0], known, caller, found, followed, sites=followed_sites, where=where)
        if sites is not None:
            sites.clear()
            sites.update(followed_sites)
        if followed == found:
            break
        found = followed
    return found


def _record_site(
    call: ast.Call,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    known: dict[str, str],
    caller: tuple[str, _Bindings] | None,
    found: dict[tuple[str, str], list[Passed]],
    followed: dict[tuple[str, str], list[Passed]],
    *,
    name: str | None = None,
    sites: dict[str, list[Site]] | None = None,
    where: tuple[str, str] = ("", "<module>"),
) -> None:
    """add what one call of ``function`` passes each of its parameters to ``followed``.

    :param call: the call
    :ptype call: ast.Call
    :param function: the function it calls
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param known: the calling module's strings
    :ptype known: dict[str, str]
    :param caller: the calling function's name and bindings, None at a top level
    :ptype caller: tuple[str, _Bindings] | None
    :param found: what each parameter is passed, as followed so far
    :ptype found: dict[tuple[str, str], list[Passed]]
    :param followed: what this pass finds
    :ptype followed: dict[tuple[str, str], list[Passed]]
    :param name: the name its parameters are kept under (a constructor's ``<Class>.__init__``, its ``self`` not
        passed); the function's own when None
    :ptype name: str | None
    :param sites: each function's call sites so far, this one added; none kept when None
    :ptype sites: dict[str, list[Site]] | None
    :param where: the calling module's path and the calling scope's qualified name
    :ptype where: tuple[str, str]
    :return: nothing
    :rtype: None
    """
    positional = [*function.args.posonlyargs, *function.args.args]
    if (
        positional[:1]
        and positional[0].arg in ("self", "cls")
        and (isinstance(call.func, ast.Attribute) or name is not None)
    ):
        positional = positional[1:]
    defaults = dict(
        zip(
            [argument.arg for argument in positional][len(positional) - len(function.args.defaults) :],
            function.args.defaults,
            strict=True,
        )
    )
    defaults.update(
        {
            argument.arg: default
            for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults, strict=True)
            if default is not None
        }
    )
    # a ``*args`` leaves every positional from it on unknown; a ``**kwargs`` every parameter not passed by name or
    # place, which it may fill
    star = next((place for place, argument in enumerate(call.args) if isinstance(argument, ast.Starred)), None)
    spread = any(keyword.arg is None for keyword in call.keywords)
    passed: dict[str, ast.expr] = dict(
        zip([a.arg for a in positional], call.args if star is None else call.args[:star], strict=False)
    )
    passed.update({keyword.arg: keyword.value for keyword in call.keywords if keyword.arg})
    site: dict[str, list[Passed]] = {}
    for place, argument in enumerate([*positional, *function.args.kwonlyargs]):
        value = passed.get(argument.arg)
        unknown = value is None and ((star is not None and place >= star) or spread)
        value = value if value is not None else defaults.get(argument.arg)
        values: list[Passed] = [None] if unknown or value is None else _site_values(value, known, caller, found)
        followed.setdefault((name or function.name, argument.arg), []).extend(values)
        site[argument.arg] = values
    if sites is not None:
        sites.setdefault(name or function.name, []).append(Site(where[0], call.lineno, where[1], site))


#: a value a forwarder reads off its ``self`` (``self._keys[table]``), as its caller's judgement of the write names it
_RUN_TIME_VALUE: Final[str] = "__forwarded__self_value"


#: what :func:`_literal` answers for an expression that is no literal
_NOT_LITERAL: Final = object()


def _literal(node: ast.expr) -> object:
    """the value a literal expression spells (a constant, or a tuple of them), or :data:`_NOT_LITERAL`.

    :param node: the expression
    :ptype node: ast.expr
    :return: its value
    :rtype: object
    """
    try:
        return ast.literal_eval(node)
    except ValueError, TypeError, SyntaxError, MemoryError, RecursionError:  # NOSILENT: not a literal: no pick
        return _NOT_LITERAL


def _target_write(call: ast.Call, parameters: set[str], receivers: set[str]) -> ast.Call:
    """a write step's call (``materialize``, ``apply_stage``, ``update_from``) as its caller judges it when the
    forwarder is handed its target: its connection (the first positional, or ``conn=``) ``None`` -- nothing its
    statements are judged by, and a forwarder's ``self``'s; each relation it reads that the forwarder names itself (a
    literal, a local step) a change set's, since that read is judged in the forwarder's body where it is written. A
    relation the forwarder is handed too stays, for the call to bind. Any other argument read off the forwarder's
    ``self`` (``key=self._keys[table]``) is a value named at run time, as the forwarder's own body judges it.

    :param call: the call
    :ptype call: ast.Call
    :param parameters: the forwarder's parameters
    :ptype parameters: set[str]
    :param receivers: the forwarder's ``self`` / ``cls``
    :ptype receivers: set[str]
    :return: the call to bind at the caller
    :rtype: ast.Call
    """
    reads = {(position, keyword) for position, keyword in STEP_RELATIONS.get(_final(call) or "", ())}

    def own(node: ast.expr) -> bool:
        return not (isinstance(node, ast.Name) and node.id in parameters)

    def written(index: int | None, keyword: str | None, node: ast.expr) -> ast.expr:
        if index == 0 or keyword == "conn":
            return ast.Constant(None)
        read = any((index is not None and index == p) or keyword == k for p, k in reads)
        if read and own(node):
            return ast.Constant(TRANSIENT_PRODUCED)
        if any(isinstance(part, ast.Name) and part.id in receivers for part in ast.walk(node)):
            return ast.Name(id=_RUN_TIME_VALUE, ctx=ast.Load())
        return node

    if any(isinstance(arg, ast.Starred) for arg in call.args):
        return call
    return ast.Call(
        func=call.func,
        args=[written(index, None, arg) for index, arg in enumerate(call.args)],
        keywords=[ast.keyword(arg=k.arg, value=written(None, k.arg, k.value)) for k in call.keywords],
    )


#: the most forwarders a relation handed on is followed through to the steps reading it
_FORWARD_DEPTH: Final[int] = 4


def _passed_arguments(call: ast.Call, callee: Callee) -> dict[str, ast.expr] | None:
    """each parameter of a forwarder a call binds, onto what it passes; None where a ``*args`` / ``**kwargs`` hides
    which.

    :param call: the call
    :ptype call: ast.Call
    :param callee: the forwarder it reaches
    :ptype callee: Callee
    :return: parameter onto argument, or None
    :rtype: dict[str, ast.expr] | None
    """
    if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        return None
    passed: dict[str, ast.expr] = {}
    for name, position in callee.parameters():
        given = next((keyword.value for keyword in call.keywords if keyword.arg == name), None)
        if given is None and position is not None and position < len(call.args):
            given = call.args[position]
        if given is not None:
            passed[name] = given
    return passed


def _constructors(modules: list[tuple[str, ast.Module]]) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """each repo class whose every construction the walk can see, onto its ``__init__``: defined once in the repo,
    with an ``__init__`` of its own, no repo class built on it (a subclass's construction passes the same
    ``__init__`` under another name), never built through ``cls(...)`` / ``type(self)(...)`` inside it, and its name
    never used as a value (handed to a factory, a ``partial``) outside a call, an annotation or a base list.

    :param modules: each module's path and parse
    :ptype modules: list[tuple[str, ast.Module]]
    :return: class name onto its constructor
    :rtype: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
    """
    classes: dict[str, list[ast.ClassDef]] = {}
    based: set[str] = set()
    loose: set[str] = set()
    for _, tree in modules:
        annotations: set[int] = set()
        called: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, []).append(node)
                for base in node.bases:
                    spelled = dotted(base)
                    if spelled:
                        based.add(spelled.rsplit(".", 1)[-1])
                annotations.update(id(element) for base in node.bases for element in ast.walk(base))
            elif isinstance(node, ast.arg | ast.AnnAssign) and node.annotation is not None:
                annotations.update(id(element) for element in ast.walk(node.annotation))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
                annotations.update(id(element) for element in ast.walk(node.returns))
            elif isinstance(node, ast.Call):
                called.add(id(node.func))
        local = local_names(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name | ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and id(node) not in called
                and id(node) not in annotations
                and id(node) not in local
            ):
                loose.add(node.id if isinstance(node, ast.Name) else node.attr)
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for name, defined in classes.items():
        if len(defined) != 1 or name in based or name in loose:
            continue
        (owner,) = defined
        if any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "cls")
                or (
                    isinstance(node.func, ast.Call)
                    and isinstance(node.func.func, ast.Name)
                    and node.func.func.id == "type"
                )
            )
            for node in ast.walk(owner)
        ):
            continue
        init = next(
            (
                statement
                for statement in owner.body
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef) and statement.name == "__init__"
            ),
            None,
        )
        if init is not None:
            found[name] = init
    return found


def _stub(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """whether a function's body is only a docstring and ``...`` (or ``pass``): a protocol's declared signature.

    :param function: the function
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :return: whether it is
    :rtype: bool
    """
    return all(
        isinstance(statement, ast.Pass)
        or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
        for statement in function.body
    )


def _module_tuples(
    tree: ast.Module, known: dict[str, str], imported: dict[str, list[Passed]] | None = None
) -> dict[str, list[Passed]]:
    """the tuples a module's top level binds once, by name, onto their elements: a string (a literal or a constant)
    or a declared object (``Declared(name=COUNTIES)``: its keyword arguments that read as strings); a starred tuple
    (``*FEED_TABLES``, the module's own or an imported one) adds its elements; an element none of these reads as
    None. A dict built by a comprehension over one (``{t.name: t for t in TABLES}``) is bound as ``<name>[]`` onto the
    same elements, so a lookup in it (``BY_NAME[name]``) reads as any one of them.

    :param tree: the module
    :ptype tree: ast.Module
    :param known: the module's strings, its imported ones among them
    :ptype known: dict[str, str]
    :param imported: the tuples it imports (:func:`_imported_tuples`), for a starred one; none when None
    :ptype imported: dict[str, list[Passed]] | None
    :return: name onto elements
    :rtype: dict[str, list[Passed]]
    """
    found: dict[str, list[Passed]] = {}
    twice: set[str] = set()
    for statement in tree.body:
        target = (
            statement.targets[0]
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1
            else (statement.target if isinstance(statement, ast.AnnAssign) and statement.value is not None else None)
        )
        value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.DictComp)
            and len(value.generators) == 1
            and isinstance(value.generators[0].target, ast.Name)
            and isinstance(value.value, ast.Name)
            and value.value.id == value.generators[0].target.id
            and isinstance(value.generators[0].iter, ast.Name)
        ):
            over = value.generators[0].iter.id
            looked = found.get(over, (imported or {}).get(over))
            if looked is not None:
                found[f"{target.id}[]"] = list(looked)
            continue
        if not isinstance(target, ast.Name) or not isinstance(value, (ast.Tuple, ast.List)):
            continue
        if target.id in found:
            twice.add(target.id)
        elements: list[Passed] = []
        for element in value.elts:
            text = _constant_text(element, known)
            spliced = (
                found.get(element.value.id, (imported or {}).get(element.value.id))
                if isinstance(element, ast.Starred) and isinstance(element.value, ast.Name)
                else None
            )
            if spliced is not None:
                elements.extend(spliced)
            elif text is not None:
                elements.append(text)
            elif isinstance(element, ast.Call):
                attributes = {
                    keyword.arg: _constant_text(keyword.value, known)
                    for keyword in element.keywords
                    if keyword.arg is not None
                }
                elements.append(tuple(sorted((k, v) for k, v in attributes.items() if v is not None)))
            else:
                elements.append(None)
        found[target.id] = elements
    return {name: elements for name, elements in found.items() if name not in twice}


def _imported_tuples(
    tree: ast.Module, relative: str, tuples: dict[str, dict[str, list[Passed]]]
) -> dict[str, list[Passed]]:
    """the tuples a module reads: the ones it binds and the ones it imports from the repo's own modules, by the name it
    imports each as.

    :param tree: the module
    :ptype tree: ast.Module
    :param relative: its repo-relative path
    :ptype relative: str
    :param tuples: every walked module's path onto its tuples (:func:`_module_tuples`)
    :ptype tuples: dict[str, dict[str, list[Passed]]]
    :return: name onto elements
    :rtype: dict[str, list[Passed]]
    """
    found: dict[str, list[Passed]] = {}
    for statement in tree.body:
        if not (isinstance(statement, ast.ImportFrom) and statement.module and not statement.level):
            continue
        source = _module_named(statement.module, tuples)
        for alias in statement.names:
            if source is not None and alias.name in source:
                found[alias.asname or alias.name] = source[alias.name]
    return {**found, **tuples.get(relative, {})}


def _module_named[T](module: str, per_module: dict[str, dict[str, T]]) -> dict[str, T] | None:
    """what a walked module binds, found by its dotted name (``pkg.tables``), or None when no walked module is it.

    :param module: the dotted name
    :ptype module: str
    :param per_module: every walked module's path onto what it binds
    :ptype per_module: dict[str, dict[str, T]]
    :return: what it binds, or None
    :rtype: dict[str, T] | None
    """
    base = module.replace(".", "/")
    return next(
        (
            names
            for path, names in per_module.items()
            if path.endswith((f"/{base}.py", f"/{base}/__init__.py")) or path in {f"{base}.py", f"{base}/__init__.py"}
        ),
        None,
    )


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
        source = _module_named(statement.module, strings)
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
    producers: frozenset[str] = frozenset(),
    view_producers: Mapping[str, tuple[tuple[str, ...], ...]] | None = None,
    relation_producers: frozenset[str] = frozenset(),
    public_forwarders: bool = False,
    declared_parameters: Mapping[str, tuple[str, ...]] | None = None,
    bounded_producers: Mapping[str, BoundBy] | None = None,
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
    :param producers: the repo's functions whose returned relation is a change set (``Class.method`` or a module
        function's name)
    :ptype producers: frozenset[str]
    :param view_producers: each function that builds a view under a name it is handed onto the keys its views declare
    :ptype view_producers: Mapping[str, tuple[tuple[str, ...], ...]] | None
    :param relation_producers: the functions and class attributes whose run-time relation is judged by the keys
        its declaration names (:func:`produced_relation`)
    :ptype relation_producers: frozenset[str]
    :param public_forwarders: whether a public function, method or constructor that hands a parameter to a step is
        judged at each of its calls in the repo, as a private one is in its module (:class:`RepoForwarders`)
    :ptype public_forwarders: bool
    :param declared_parameters: ``<qualified function>.<parameter>`` onto the values the repo declares it holds: a
        relation or column named at run time (a copy's every table), judged once per value
    :ptype declared_parameters: Mapping[str, tuple[str, ...]] | None
    :param bounded_producers: the copy accessor onto its checked bounded producer: a read through it is judged as
        reading its relation cut by the producer's key (:mod:`~threetears.enforcement.no_whole_reads.bounded`)
    :ptype bounded_producers: Mapping[str, BoundBy] | None
    :return: the walk
    :rtype: WholeReadScan
    """
    scan = WholeReadScan(schema=schema if schema is not None else SchemaCatalog())
    used: set[str] = set()
    calls: list[_CallSite] = []
    constant_uses: set[tuple[str, str, int, int]] = set()
    module_functions: dict[str, set[str]] = {}
    module_classes: dict[str, set[str]] = {}
    stubs: set[str] = set()
    imports: dict[str, dict[str, str]] = {}
    pattern = re.compile(transient) if transient else None
    modules = [(relative_posix_path(path, repo_root), path) for root in src_roots for path in iter_python_files(root)]
    # one tree per module for the whole walk: the passes below key what they find by the nodes they found it at
    # (a builder, a forwarder, a module's tuples) and later passes look those nodes up again, so each pass must see
    # the same tree. the shared parse cache keeps only some files' trees (its memory bound) and parses the rest
    # anew on each call, so the walk holds its own
    trees = {relative: parse_python_path(path) for relative, path in modules}
    strings = {relative: _module_strings(trees[relative]) for relative, _ in modules}
    tuples: dict[str, dict[str, list[Passed]]] = {}
    # twice: a module's tuple may splice one another module binds
    for _ in range(2):
        tuples = {
            relative: _module_tuples(
                trees[relative],
                {**_imported_strings(trees[relative], strings), **strings.get(relative, {})},
                _imported_tuples(trees[relative], relative, tuples),
            )
            for relative, _ in modules
        }
    cte_arguments = _cte_arguments([(relative, trees[relative]) for relative, _ in modules])
    parameter_sites: dict[str, list[Site]] = {}
    parameter_values = _parameter_values(
        [(relative, trees[relative]) for relative, _ in modules],
        strings,
        tuples,
        parameter_sites,
        declared_parameters,
    )
    match_builders = _match_builders([(relative, trees[relative]) for relative, _ in modules], strings)
    never_none = _never_none_parameters([(relative, trees[relative]) for relative, _ in modules])
    sentinel_templates = declared_templates([(relative, trees[relative]) for relative, _ in modules], strings)
    walked: list[tuple[str, Path]] = []
    for relative, path in modules:
        excluded = [prefix for prefix in excluded_prefixes if relative.startswith(prefix)]
        used.update(excluded)
        if excluded or any(part in schema_change_dirs for part in relative.split("/")[:-1]):
            continue
        walked.append((relative, path))
    bounded = bounded_relations(
        bounded_producers or {},
        [(relative, trees[relative]) for relative, _ in walked],
        {
            relative: {**_imported_strings(trees[relative], strings), **strings.get(relative, {})}
            for relative, _ in walked
        },
    )
    scan.bounded_findings = bounded.findings

    def walk_of(relative: str, path: Path, forwarders: RepoForwarders | None) -> _ModuleWalk:
        tree = trees[relative]
        return _ModuleWalk(
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
            producers=producers,
            parameter_values=parameter_values,
            parameter_sites=parameter_sites,
            view_producers=view_producers,
            tuples=_imported_tuples(tree, relative, tuples),
            relation_producers=relation_producers,
            match_builders=match_builders,
            forwarders=forwarders,
            builders=builders,
            never_none=never_none,
            declared=declared_parameters,
            bounded=bounded,
            sentinels=sentinel_templates.get(relative),
        )

    # the repo's step builders, by name across every walked module: a call of one is a step judged where it is
    # written (R11)
    builders = step_builders([trees[relative] for relative, _ in walked])
    forwarders: RepoForwarders | None = None
    if public_forwarders:
        # each module's flow settles into the repo's shared state; once a pass over every module adds nothing, a
        # forwarder's every caller is judged by what the forwarder hands a step
        forwarders = RepoForwarders.collect([(relative, trees[relative]) for relative, _ in walked])
        for _ in range(_PARAMETER_DEPTH):
            before = forwarders.size()
            for relative, path in walked:
                walk_of(relative, path, forwarders)
            if forwarders.size() == before:
                break
    for relative, path in walked:
        tree = trees[relative]
        walk = walk_of(relative, path, forwarders)
        walk.visit(tree)
        stubs.update(walk.stubs)
        scan.hits.extend(walk.hits)
        scan.statements.extend(walk.statements)
        scan.definitions.update(walk.definitions)
        scan.constants.update(walk.constants)
        calls.extend(walk.calls)
        module_functions[relative] = walk.module_functions
        module_classes[relative] = walk.module_classes
        scan.operators.extend(walk.operators)
        scan.declarations.extend(walk.declarations)
        imports[relative] = walk.imports
        scan.declared_matched.update(walk.declared_matched)
        constant_uses |= constant_readers(relative, tree)
        scan.files_scanned += 1
    resolved, callers = _resolve(
        calls,
        scan.hits,
        module_functions,
        scan.definitions,
        stubs,
        imports,
        module_classes,
        scan.operators,
        constant_uses,
    )
    scan.hits.extend(resolved)
    scan.operator_callers = callers
    scan.unmatched_exclusions = [prefix for prefix in excluded_prefixes if prefix not in used]
    return scan
