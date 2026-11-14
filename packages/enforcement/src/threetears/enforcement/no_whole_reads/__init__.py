"""no-whole-reads enforcement domain: load, cache and follower code reads by key, never a whole table.

An epoch IS its change set. A pod learns what changed from the change log and its change notes and reads exactly
those rows by key; data from outside comes in by the source's own change records (its runs, its upload stamps).
Nothing reads a whole table to find out what changed, re-verifies its own data by scanning it, or falls back to a
full reload when something goes wrong. A whole read is right in exactly two places: the one cold start that fills
an empty copy, and a schema change that rewrites the table.

**The rule, statically** (:func:`~threetears.enforcement.no_whole_reads.walker.scan_whole_reads`, AST plus the SQL
the code writes, judged by :func:`~threetears.enforcement.no_whole_reads.sql.judge_sql`):

- every SQL statement against a data table reads it **by key** (:mod:`~threetears.enforcement.no_whole_reads.sql`):
  its bound equality or ``= ANY`` predicates cover a leading prefix of the table's primary key, or the leading
  column of an index whose ``purpose`` names the function, or it is a keyset page (an ordered key range and a
  ``LIMIT``). No ``SELECT ... FROM <table>`` without ``WHERE`` (``whole-select``), no ``WHERE`` without a bound value
  (``unbound-where``), no filter on anything but the key (a status, a kind, a flag: ``non-key-where``), no predicate
  on a table whose key is declared nowhere (``unknown-schema``). Keys and indexes are read from the repo's
  declarations (:mod:`~threetears.enforcement.no_whole_reads.schema`: ``TableSchema``, SQLAlchemy ``Table``, the
  migrations' DDL, ``tables.yaml``) and from 3tears' own;
- no ``COUNT``, ``GROUP BY``, ``DISTINCT``, sum or fingerprint over a whole table (``whole-aggregate``);
- no ``TRUNCATE``, no ``DELETE`` / ``UPDATE`` without ``WHERE``, no ``ANALYZE`` / ``VACUUM`` of a table, no
  backend ``truncate()``;
- no pager or fingerprint helper called without a narrowing keyword (``unkeyed-pager``, ``may-read-whole``), no
  helper that always reads whole (``whole-read``);
- no call whose name says whole, rebuild, warm, reload or resync (``whole-call``), no flag asking for one
  (``whole-flag``), no table put in a set of tables to read whole (``marks-whole``);
- no call into a function of the repo that does any of the reads above (``calls-whole-read``), so the caller is
  named, not only the helper;
- every index the repo declares names its ``purpose`` (the function whose keyed read uses it) and a ``plan_test``
  proving that read uses it, and a wide index (``INCLUDE``) or one of more than three on a table gives its
  ``reason`` (:func:`~threetears.enforcement.no_whole_reads.schema.index_findings`); an index no read can name is a
  candidate for removal.

**An operator's own command is marked where it is written** (:mod:`threetears.core.operator`), never listed:

- ``@operator_reset(reason, proof=...)``: an explicit reset or truncate an operator orders. Its ``truncate`` and
  ``truncate-call`` hits pass; every read in it is judged as anywhere else;
- ``@operator_command(reason, proof=...)``: an operator's explicit one-off command that must read whole to be
  complete (a restore check over its scratch copy, a person's erasure across every agent), never a load or cache
  path. Its reads pass.

A mark is checked: every caller of a marked function in the repo must carry the same mark, so the chain ends at an
entry nothing in the repo calls (a route, a CLI command); ``proof`` names ``<test file>::<test function>``, a test
that names the function and proves its entry needs the operator; the reason is 30 characters or more.

**A read whole by nature is declared where it is written** (:func:`threetears.core.declared.whole_read`, Pace
2026-11-12), never listed: ``await whole_read(conn.fetch(...), kind=..., reason=..., ruling=..., proof=..., speed=...)``.
The kind (:data:`~threetears.enforcement.no_whole_reads.walker.DECLARED_KINDS`) lets through only its own whole reads
in the wrapped expression, or in a module constant only it names; the reason is 30 characters or more; the ruling
names Pace and a date; ``proof`` names a test that names the enclosing function (it proves the statement is reached
only in that situation); ``speed``, for the kinds reading our own tables at size, names a test calling
``budget_problem``; a ``drain`` or ``unbounded_by_warehouse`` read is never of a declared L3 table. A call into a
function whose every whole read is declared (or exempt in the ledger) is not flagged again at the caller (coordinator
ruling 2026-11-13: the callee's declaration, proof and speed test cover it wherever it is reached); a call into a
function holding any undeclared whole read is still ``calls-whole-read``. A call declared with another kind than the
callee's covers nothing. A declaration covering no whole read is a finding. The gate prints every one it accepts, and the use-the-helper rule
does not count its statement (no keyed helper answers it).

**The one way out for a cold start or schema change: the exemptions ledger.** A function that is a true cold start, or a schema change, is listed
in the repo's ``_no_whole_reads_exemptions.txt``, one line each::

    <path> :: <qualified function> :: <hits> :: <cold-start|schema-change|source-gap> :: <proving test> :: <speed test> :: <reason>

- ``<hits>`` pins the function's hit count exactly: a new whole read added inside an exempt function fails, and so
  does a line whose hits went;
- ``<proving test>`` is ``<repo-relative test file>::<test function>``: a test that proves the function is reached
  ONLY on a cold start (an empty copy, no checkpoint) or a schema change, never on a steady-state load, a retry or
  a recovery. The gate checks the file defines that test and that the test names the function;
- ``<speed test>`` is ``<repo-relative test file>::<test function>`` too: a real-size speed test of the function,
  held to a multiple of the bare operation timed on the same database
  (:func:`threetears.core.testing.calibrate.budget_problem`), never a figure in seconds. "It only runs once" is no
  reason for a cold start to be slow. The gate checks the file defines that test, that it names the function and
  that it calls ``budget_problem``;
- ``<reason>`` says which cold start or schema change, in 30 characters or more; a ``source-gap`` (a whole read kept
  only because an upstream source keeps no change record) opens with its gap id from the source-gaps list and the
  condition it fires on (``G7: ...``), and the gate prints every one it accepts.

A line is never "it's small", "it's an admin path", "it's being fixed" or "pre-existing": those are violations,
fixed, not listed. There is no allowlist of violations; the gate is red until each is fixed.

**The runtime half** is :class:`threetears.core.testing.plans.WholeScanGuard`: a load-path test wraps its
connection, and every statement whose plan scans a table past a row threshold, or whose pages add up to most of
one, fails the test.
"""

from __future__ import annotations

import ast
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Final

import threetears

from threetears.enforcement.common import dotted
from threetears.enforcement.no_whole_reads.bounded import BoundBy
from threetears.enforcement.no_whole_reads.declarations import DECLARATIONS_NAME, apply_index_declarations
from threetears.enforcement.no_whole_reads.schema import (
    COPY_PREFIX,
    PRODUCED_PREFIX,
    SchemaCatalog,
    collect_schema,
    index_findings,
)
from threetears.enforcement.no_whole_reads.use_the_helper import statement_counts, statement_findings
from threetears.enforcement.no_whole_reads.walker import (
    DECLARED_KINDS,
    DECLARED_OUTSIDE_KINDS,
    DECLARED_SPEED_KINDS,
    DEFAULT_NARROWING_KEYWORDS,
    DEFAULT_PAGER_CALLS,
    DeclaredWhole,
    OperatorMark,
    WholeReadHit,
    WholeReadScan,
    WrittenStatement,
    produced_relation,
    scan_whole_reads,
)

__all__ = [
    "BoundBy",
    "NoWholeReadsConfig",
    "declaration_defects",
    "no_whole_reads_findings",
    "use_the_helper_counts",
    "use_the_helper_findings",
]

#: the ledger's file name, beside the repo's gate test
_EXEMPTIONS_NAME: Final[str] = "_no_whole_reads_exemptions.txt"

#: the only reasons a function may read whole: a true cold start, a schema change, and a gap an upstream source's
#: missing change record leaves (``source-gap``, its reason naming the gap id from the source-gaps list)
_EXEMPTION_KINDS: Final[frozenset[str]] = frozenset({"cold-start", "schema-change", "source-gap"})

#: a source gap's reason opens with its id from the source-gaps list, then the condition it fires on: ``G7: ...``
_SOURCE_GAP: Final = re.compile(r"^G\d+:\s*\S.{20,}")

_SEPARATOR: Final[str] = " :: "
_MIN_REASON: Final[int] = 30

#: what each finding is told to do instead
_ADVICE: Final[str] = (
    "read the rows the epoch named, by key (the change log and change notes; ScopeEpochs for a scope), or for an "
    "external source its own change records; a true cold start or schema change goes in "
    f"{_EXEMPTIONS_NAME} with the test proving it is reached only then"
)


@dataclass(frozen=True)
class NoWholeReadsConfig:
    """per-repo config for the no-whole-reads gate.

    :ivar repo_root: absolute path of the repo root; every path in a finding or the ledger is relative to it
    :ivar src_roots: production source trees to walk
    :ivar ledger_dir: directory holding :data:`_EXEMPTIONS_NAME`
    :ivar excluded_prefixes: repo-relative posix directory prefixes left unwalked (test doubles shipped under src,
        the gates themselves); each must match a module
    :ivar extra_pagers: the repo's own helpers that read every row they are not narrowed to
    :ivar extra_narrowing: the repo's own keywords that narrow them
    :ivar transient_relations: regular expressions of relations that are a change set being applied (a stage of
        rows a change named), never a data table: a tuple, one pattern per alternative (a single ``str`` is still
        read as one pattern); empty for none. A pattern no relation of the walk matches, or one matching only what
        another matches, is a finding
    :ivar transient_producers: the repo's functions whose returned relation is such a change set (``Stages._stage``,
        ``_Work.put``): ``Class.method``, or a module function's name. A statement over what one returns -- the call in
        the placeholder, a name bound to it, a ``self`` attribute every assignment of which is one -- reads a change
        set; a method counts called on ``self`` in its class, on the class, or on a receiver known to be the class
    :ivar view_producers: the repo's functions that build a view under a name they are handed (``_view_sql``,
        ``Parts.copy_view``: ``Class.method`` or a module function's name), onto the key (or keys) every view each
        builds declares: its ``CREATE VIEW {}`` is judged by them, as a declared view is by its own (passed through)
    :ivar verify_prefixes: repo-relative prefixes of load code (a writer, a loader, a follower), where a count or
        fingerprint even of one key's rows is a verification scan (``verify-aggregate``): the epoch and the source's
        own change records say what changed, so nothing counts rows to find out
    :ivar minimum_files_scanned: the walk must reach at least this many modules
    :ivar schema_roots: more trees to read table declarations from (migrations kept outside the source trees)
    :ivar table_lists: YAML table lists to read; None finds every ``tables.yaml`` / ``agent.yaml`` in the repo
    :ivar local_relations: each local copy's relation the repo reads (``snapshot.relation(NAME)``, judged as
        ``copy.<name>``) onto its key columns: the only keys of the ``copy.`` namespace, so a copy never loosens the
        L3 table that shares its name, nor an L3 declaration a copy; one key, or several (a read covering a leading
        prefix of any is keyed); ``"*"`` declares the default key of every copy relation, and of one named at run time
        (``snapshot.relation(name)``)
    :ivar foreign_relations: each relation the repo reads that another service owns and declares there (the hub
        broker's synthesized ``knowledge_readiness``), onto the key that owner answers it by; one key or several
    :ivar extra_exec_calls: the repo's own calls that run their first argument as SQL (reports' ``Snapshot.ask``,
        the pod's ``work.arrow``): a bare ``SELECT ... FROM t`` handed to one is a whole read where it is run
    :ivar relation_producers: the repo's run-time relations a keyed helper is handed, onto the key (or keys) every
        relation each yields declares: a function returning one (``derived``, a module function's name, or
        ``Class.method``), or a class's attribute holding one (``SourceTable.relation``), named on ``self`` in its
        class or on a receiver the walk knows is the class (a parameter's or annotated assignment's annotation, a
        loop target over an annotated iterable). A helper reading one is judged as reading ``produced.<entry>`` (a
        dot ``__``) with those keys, as a declared table is by its own; ``()`` declares one read whole
    :ivar helper_modules: repo-relative prefixes of the repo's own reviewed modules that write a data table's SQL
        for the rest of it (``use-the-helper``: :mod:`~threetears.enforcement.no_whole_reads.use_the_helper`), beside
        3tears' own helpers
    :ivar caller_judged_modules: repo-relative prefixes of the repo's declared helper modules whose statements over
        a relation their caller names at run time (a parameter, a field, a template the module fills: a relation
        spelled with a ``{}`` in its table part) are the module's callers' to key, as 3tears' keyed helpers' are:
        ``use-the-helper`` leaves those statements to the callers (the module is the helper they call), while a
        statement over a table the module names itself (a literal table, a fixed table in a run-time schema) is
        still judged where it is written. ``no_whole_reads`` still judges every statement in them. A prefix
        matching no module that writes SQL is a finding
    :ivar public_forwarders: whether a public function, method or constructor of the repo that hands a parameter on
        to a keyed step (or returns a step over it: ``settle_rules.annotated_step(table, relation)``) is judged at
        each of its calls in the repo, by the argument the call hands it, as a private one always is in its own
        module. Only for a repo whose public functions its own walked code alone calls (an application): a library's
        are called from repos this walk never reads, so there a step over a public function's parameter stays judged
        in its body. A definition referenced other than by a call the walk matches (handed on as a value, its module
        imported whole, a method name another definition shares) is never one
    :ivar declared_parameters: ``<qualified function>.<parameter>`` (``_apply_slice.tables``,
        ``Backend.export_partition.table``) onto the values the repo declares it holds where no caller the walk reads
        says (a run-time mapping of a copy's tables, a protocol method 3tears calls): a ``str`` (or unannotated)
        parameter holds one of them, any other a collection a loop over it binds each of; a name a loop of the
        function binds (``apply_states.table`` for ``for table, pieces in small.items()``) each of them. Every read naming one is
        judged once per value, through names the function binds from it once and its module's lookup functions
        (``_state_column(table)``), so its key must key every member. Keep the values a module constant the repo's
        own test holds equal to what runs; a declaration naming no parameter of the walk is a finding
    :ivar dependency_schema: whether 3tears' own declarations (its change log, notes and collections, which a
        consumer reads) key reads too; its indexes are never the repo's to give a purpose
    :ivar threetears_tables_only: whether only the tables 3tears declares are data tables (a ``TableSchema``, a
        collection class, a table list, a relation this config declares, any declaration in 3tears' own sources):
        a table only a SQLAlchemy ``Table`` or the repo's own migration DDL declares is no data table, and a relation
        named at run time counts only when it resolves to a 3tears one. For a repo that keeps a side database of its
        own, unchecked; off by default, every declared or written relation a data table
    :ivar frozen_migrations: repo-relative globs (``fnmatch``) of the repo's frozen migrations: shipped and
        checksum-tracked, so their text is never edited again (the boundary moves forward each release). Only an index
        whose DDL is in one may be declared in the index declarations ledger (``<ledger_dir>/_index_declarations.txt``,
        :mod:`~threetears.enforcement.no_whole_reads.declarations`) instead of by comments beside its DDL
    :ivar consumer_purposes: path prefixes of the repos that read this repo's tables through its declarations (the
        hub's ``src/aibots/hub/``): an index purpose under one names a consumer's read, which keys that read in the
        consumer's own gate (its indexes are the dependency's there); here such an entry is neither unknown nor unused.
        The plan test, in this repo, still proves the index serves the consumer's statement
    :ivar local_relations_checked: whether a ``local_relations`` entry no read of the walk is judged by is a finding,
        as every other unused loosening declaration is (``transient_producers``, ``view_producers``,
        ``relation_producers``, ``bounded_producers``, ``transient_relations``); a repo whose entries record the
        copy's key facts whether or not a read names them today turns it off
    :ivar bounded_producers: the copy accessor (``Snapshot.relation``) onto the producer that cuts every relation it
        answers to a key (:class:`~threetears.enforcement.no_whole_reads.bounded.BoundBy`: reports'
        ``limited_snapshot``, each relation to the reader's states): a read through the accessor is judged as reading
        that cut, with no condition of its own. Checked, never trusted: the producer binds the key in every relation
        it returns, every other source of the relations is a snapshot's own or a declared one, and a reader of every
        state is the one place the producer is skipped (:mod:`~threetears.enforcement.no_whole_reads.bounded`)
    :ivar clause_builders: the repo's clause builders the use-the-helper rewrite turns into a helper's ``where=``
        (``module:name`` of the builder -> ``module:name`` of its proof subjects): a builder answers `` AND (...)`` or
        ``""``, called ``build(<subject>, "<text>", ..., <args>)``, binding its values onto the end of ``args``; a
        statement interpolating its answer is rewritten with ``where=[*clause_guards(build, <subject>, ...)]``. The
        subjects are a zero-argument callable answering the subjects the generated proof binds the clause to, beside
        ``None``: each must restrict the seeded rows, or the proof shows nothing about the clause. Not a judging
        option: the gate's findings are the same with or without it
    :ivar clause_parents: per clause builder whose clause reads rows of another table (a chunk's parent memory),
        ``module:name`` of the coroutine seeding them in the generated proof, handed the connection and the rows seeded
        in the read's own table: without those rows a subject's clause would match nothing on either side and prove
        nothing. Not a judging option either
    """

    repo_root: Path
    src_roots: tuple[Path, ...]
    ledger_dir: Path
    excluded_prefixes: tuple[str, ...] = ()
    extra_pagers: frozenset[str] = field(default_factory=frozenset)
    extra_narrowing: frozenset[str] = field(default_factory=frozenset)
    transient_relations: str | tuple[str, ...] = ""
    transient_producers: frozenset[str] = field(default_factory=frozenset)
    view_producers: Mapping[str, tuple[str, ...] | tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    verify_prefixes: tuple[str, ...] = ()
    minimum_files_scanned: int = 1
    schema_roots: tuple[Path, ...] = ()
    table_lists: tuple[Path, ...] | None = None
    dependency_schema: bool = True
    extra_exec_calls: frozenset[str] = field(default_factory=frozenset)
    local_relations: Mapping[str, tuple[str, ...] | tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    foreign_relations: Mapping[str, tuple[str, ...] | tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    helper_modules: tuple[str, ...] = ()
    caller_judged_modules: tuple[str, ...] = ()
    relation_producers: Mapping[str, tuple[str, ...] | tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    public_forwarders: bool = False
    declared_parameters: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    threetears_tables_only: bool = False
    consumer_purposes: tuple[str, ...] = ()
    frozen_migrations: tuple[str, ...] = ()
    bounded_producers: Mapping[str, BoundBy] = field(default_factory=dict)
    local_relations_checked: bool = True
    clause_builders: Mapping[str, str] = field(default_factory=dict)
    clause_parents: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Exemption:
    """one ledger line.

    :ivar file_line: its line in the ledger
    :ivar path: the module
    :ivar function: the qualified function
    :ivar count: the hits it pins
    :ivar kind: ``cold-start`` or ``schema-change``
    :ivar test: ``<test file>::<test function>``
    :ivar reason: why
    :ivar speed_test: ``<test file>::<test function>`` of its calibrated speed test
    """

    file_line: int
    path: str
    function: str
    count: int
    kind: str
    test: str
    reason: str
    speed_test: str = ""


def _parse(path: Path) -> tuple[list[_Exemption], list[str]]:
    """read the ledger.

    :param path: the ledger; a missing file reads as empty
    :ptype path: Path
    :return: the lines, and every malformed line as a finding
    :rtype: tuple[list[_Exemption], list[str]]
    """
    lines: list[_Exemption] = []
    defects: list[str] = []
    seen: set[tuple[str, str]] = set()
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split(_SEPARATOR)]
        where = f"{path.name}:{number}"
        if len(parts) != 7:
            defects.append(
                f"{where}: expected 7 fields separated by '{_SEPARATOR.strip()}' (path, function, hits, kind, proving "
                f"test, speed test, reason), got {len(parts)}"
            )
            continue
        module, function, count, kind, test, speed, reason = parts
        if kind not in _EXEMPTION_KINDS:
            defects.append(f"{where}: {kind!r} is not a reason to read whole; only {sorted(_EXEMPTION_KINDS)} are")
        elif not count.isdigit() or int(count) < 1:
            defects.append(f"{where}: hits {count!r} must be a positive integer")
        elif len(reason) < _MIN_REASON:
            defects.append(f"{where}: the reason must name the cold start or schema change ({_MIN_REASON}+ characters)")
        elif kind == "source-gap" and _SOURCE_GAP.match(reason) is None:
            defects.append(
                f"{where}: a source gap's reason names its gap id from the source-gaps list and the condition it "
                "fires on: 'G7: <when it reads whole>'"
            )
        elif (module, function) in seen:
            defects.append(f"{where}: {module} :: {function} is listed twice")
        else:
            seen.add((module, function))
            lines.append(_Exemption(number, module, function, int(count), kind, test, reason, speed))
    return lines, defects


def _proving_test_defect(
    repo_root: Path, exemption: _Exemption, *, test: str | None = None, calls: str | None = None
) -> str | None:
    """what is wrong with an exemption's proving test (or its speed test), if anything.

    :param repo_root: the repo root
    :ptype repo_root: Path
    :param exemption: the line
    :ptype exemption: _Exemption
    :param test: the test to judge, ``<file>::<function>``; the line's proving test when None
    :ptype test: str | None
    :param calls: a name the test must call as well as naming the function (the speed test's ``budget_problem``)
    :ptype calls: str | None
    :return: the defect, or None when the file defines the test and the test names the function (and calls
        ``calls``)
    :rtype: str | None
    """
    module, _, name = (exemption.test if test is None else test).partition("::")
    test_path = repo_root / module
    defect: str | None = None
    if not name or not test_path.is_file():
        defect = f"names no test file and function ({exemption.test!r}); give <tests/...py>::<test_function>"
    else:
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
        tests = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ]
        target = exemption.function.rsplit(".", 1)[-1]
        if not tests:
            defect = f"{module} defines no {name}"
        elif not any(target in "\n".join(ast.unparse(statement) for statement in test.body) for test in tests):
            defect = f"{module}::{name} never names {target}: it cannot prove where {target} is reached"
        elif calls is not None and not any(
            isinstance(node, ast.Call) and (dotted(node.func) or "").rsplit(".", 1)[-1] == calls
            for test in tests
            for node in ast.walk(test)
        ):
            defect = f"{module}::{name} never calls {calls}: a speed budget is a multiple of a calibrated baseline"
    return defect


def _dependency_roots(repo_root: Path) -> tuple[Path, ...]:
    """the source trees of the installed 3tears packages outside the repo, read for their table declarations.

    :param repo_root: the repo root
    :ptype repo_root: Path
    :return: the trees
    :rtype: tuple[Path, ...]
    """
    root = repo_root.resolve()
    return tuple(
        Path(entry) for entry in getattr(threetears, "__path__", []) if not Path(entry).resolve().is_relative_to(root)
    )


def _keys_of(declared: tuple[str, ...] | tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
    """a declaration's keys: one key (its columns), or several.

    :param declared: the declaration
    :ptype declared: tuple[str, ...] | tuple[tuple[str, ...], ...]
    :return: each key's columns
    :rtype: tuple[tuple[str, ...], ...]
    """
    if all(isinstance(column, str) for column in declared):
        return (tuple(str(column) for column in declared),)
    return tuple(tuple(str(column) for column in key) for key in declared if not isinstance(key, str))


def _schema_of(config: NoWholeReadsConfig) -> SchemaCatalog:
    """every table key and index the repo (and the 3tears it reads) declares.

    :param config: the repo's config
    :ptype config: NoWholeReadsConfig
    :return: the catalog
    :rtype: SchemaCatalog
    """
    catalog = collect_schema(
        config.repo_root,
        (*config.src_roots, *config.schema_roots),
        dependency_roots=_dependency_roots(config.repo_root) if config.dependency_schema else (),
        excluded_prefixes=config.excluded_prefixes,
        table_lists=config.table_lists,
    )
    # a relation this config declares is one the repo reads through 3tears (a local copy, another service's, a
    # produced relation): a data table with the switch on too
    catalog.marking = True
    for prefix, relations in ((COPY_PREFIX, config.local_relations), ("", config.foreign_relations)):
        for written, declared in relations.items():
            # ``local_relations["*"]``: the default key of every copy relation the repo declares no key for
            name = "{}" if written == "*" and prefix else written
            for key in _keys_of(declared):
                catalog.add_key(f"{prefix}{name.lower()}", key, "NoWholeReadsConfig")
    for entry, declared in config.relation_producers.items():
        for key in _keys_of(declared):
            catalog.add_key(produced_relation(entry), key, "NoWholeReadsConfig.relation_producers")
    catalog.marking = False
    catalog.threetears_only = config.threetears_tables_only
    # an index in a frozen migration is declared in the ledger; applied before the walk credits its purpose's read
    catalog.ledger_findings = apply_index_declarations(
        catalog, config.ledger_dir / DECLARATIONS_NAME, config.frozen_migrations
    )
    return catalog


def _hits(config: NoWholeReadsConfig, *, discovering: bool = False) -> tuple[list[WholeReadHit], WholeReadScan]:
    """every hit of the walk, ledger ignored: the repo's list of whole reads.

    :param config: the repo's config
    :ptype config: NoWholeReadsConfig
    :param discovering: the index-purpose tool's walk (:attr:`SchemaCatalog.discovering`): every index counts as
        purposed, so each read is credited to every index that serves it
    :ptype discovering: bool
    :return: the hits, and the walk
    :rtype: tuple[list[WholeReadHit], WholeReadScan]
    """
    schema = _schema_of(config)
    schema.discovering = discovering
    scan = scan_whole_reads(
        config.repo_root,
        config.src_roots,
        excluded_prefixes=config.excluded_prefixes,
        pagers=DEFAULT_PAGER_CALLS | config.extra_pagers,
        narrowing=DEFAULT_NARROWING_KEYWORDS | config.extra_narrowing,
        transient=config.transient_relations,
        verify_prefixes=config.verify_prefixes,
        schema=schema,
        exec_calls=config.extra_exec_calls,
        producers=config.transient_producers,
        view_producers={producer: _keys_of(declared) for producer, declared in config.view_producers.items()},
        relation_producers=frozenset(config.relation_producers),
        public_forwarders=config.public_forwarders,
        declared_parameters=config.declared_parameters,
        bounded_producers=config.bounded_producers,
    )
    return scan.hits, scan


def _inside(hit: WholeReadHit, exemption: _Exemption) -> bool:
    """whether a hit sits in an exempt function or one nested in it.

    :param hit: the hit
    :ptype hit: WholeReadHit
    :param exemption: the line
    :ptype exemption: _Exemption
    :return: whether it does
    :rtype: bool
    """
    return hit.path == exemption.path and (
        hit.function == exemption.function or hit.function.startswith(f"{exemption.function}.")
    )


def _operator_mark_findings(config: NoWholeReadsConfig, scan: WholeReadScan) -> list[str]:
    """what is wrong with each operator mark: a short reason, a proof that names no test of the function, a caller.

    :param config: the repo's config
    :ptype config: NoWholeReadsConfig
    :param scan: the walk
    :ptype scan: WholeReadScan
    :return: findings
    :rtype: list[str]
    """
    findings: list[str] = list(scan.operator_callers)
    for mark in scan.operators:
        where = f"{mark.path}:{mark.line} {mark.function} @{mark.mark}"
        if mark.mark == "source_change_read" and not (mark.stamps or mark.ids or mark.unit == "table"):
            findings.append(f"{where}: name the source's stamps, its ids, or unit='table'")
        elif mark.mark != "source_change_read" and len(mark.reason) < _MIN_REASON:
            what = "the risk the backstop covers" if mark.mark == "security_backstop" else "which operator command"
            findings.append(f"{where}: the reason must say {what} ({_MIN_REASON}+ characters)")
        line = _Exemption(mark.line, mark.path, mark.function, 1, mark.mark, mark.proof, mark.reason, mark.speed)
        defect = _proving_test_defect(config.repo_root, line)
        if defect is not None:
            findings.append(f"{where}: the proof {defect}")
        if mark.mark == "security_backstop":
            speed = _proving_test_defect(config.repo_root, line, test=mark.speed, calls="budget_problem")
            if speed is not None:
                findings.append(f"{where}: the speed test {speed}")
    return findings


class _WholeReadAccepted(UserWarning):
    """a whole pass the gate accepted by its mark: a security backstop, or a source change read. Each is printed
    and warned, so every one is seen on every run, never only listed."""


def _announce(scan: WholeReadScan) -> list[str]:
    """print and warn each security backstop and source change read the gate accepts: rare, and always visible.

    :param scan: the walk
    :ptype scan: WholeReadScan
    :return: the lines printed
    :rtype: list[str]
    """
    lines = [
        f"no_whole_reads accepts @{mark.mark} {mark.path}:{mark.line} {mark.function}: "
        + (
            mark.reason
            if mark.mark == "security_backstop"
            else ", ".join((*mark.stamps, *mark.ids, mark.unit)).strip(", ")
        )
        for mark in scan.operators
        if mark.mark in {"security_backstop", "source_change_read"}
    ]
    for line in lines:
        sys.stderr.write(f"{line}\n")
        warnings.warn(line, _WholeReadAccepted, stacklevel=2)
    return lines


def _marked(hit: WholeReadHit, marks: list[OperatorMark], schema: SchemaCatalog | None = None) -> bool:
    """whether a hit sits in a function (or one nested in it) whose operator mark lets its kind through. A source
    change read of a whole static input (``unit="table"``) lets through its whole reads and pagers of relations no
    declaration names, never of an L3 or catalog table.

    :param hit: the hit
    :ptype hit: WholeReadHit
    :param marks: the operator marks
    :ptype marks: list[OperatorMark]
    :param schema: the declared tables, which a static input's mark never covers
    :ptype schema: SchemaCatalog | None
    :return: whether it does
    :rtype: bool
    """
    relation = hit.symbol.split(" ", 1)[1] if " " in hit.symbol else ""
    declared = schema is not None and hit.kind == "whole-select" and schema.table(relation) is not None
    return any(
        hit.path == mark.path
        and (hit.function == mark.function or hit.function.startswith(f"{mark.function}."))
        and (hit.kind in mark.kinds or (mark.unit == "table" and hit.kind in _STATIC_UNIT_KINDS and not declared))
        for mark in marks
    )


#: the whole reads a source change read of a static input (``unit="table"``) is: a whole select, a pager over it
_STATIC_UNIT_KINDS: Final[frozenset[str]] = frozenset(
    {"whole-select", "whole-aggregate", "unkeyed-pager", "may-read-whole", "whole-read", "calls-whole-read"}
)


#: Pace's ruling, as a declaration cites it: his name and the ruling's date
_RULING: Final[re.Pattern[str]] = re.compile(r"\bPace\b.*\b\d{4}-\d{2}-\d{2}\b|\b\d{4}-\d{2}-\d{2}\b.*\bPace\b")

#: the bare operation a backup verifies with: ``count(*)`` (and the order-free row checksum) over one relation, no join,
#: filter or ordering. A ``backup_verification`` declaration whose every statement is this needs no speed test (it IS
#: the bare operation; coordinator ruling 2026-11-12); anything richer does.
_BARE_COUNT: Final[re.Pattern[str]] = re.compile(
    r"^\s*SELECT\s+count\(\*\)(?:\s+AS\s+\w+)?"
    r"(?:\s*,\s*coalesce\(sum\(hashtext\(\w+::text\)::bigint\),\s*0\)(?:\s+AS\s+\w+)?)?"
    r"\s+FROM\s+(?:\"[^\"]*\"|[\w{}]+)(?:\.(?:\"[^\"]*\"|[\w{}]+))?(?:\s+\w+)?\s*;?\s*$",
    re.IGNORECASE,
)

#: the bare operation a schema change maintains a table with: ``ANALYZE`` or ``REINDEX`` of one relation, nothing else
#: in the statement. A ``schema_change`` declaration whose every statement is this needs no speed test (it IS the
#: maintenance operation, as a backup's bare count is; coordinator ruling 2026-11-13, for the hub)
_BARE_MAINTENANCE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:ANALYZE|REINDEX\s+(?:\(\s*\w+\s*\)\s+)?(?:TABLE|INDEX)(?:\s+CONCURRENTLY)?)"
    r"\s+(?:\"[^\"]*\"|[\w{}]+)(?:\.(?:\"[^\"]*\"|[\w{}]+))?\s*;?\s*$",
    re.IGNORECASE,
)

#: each kind whose declaration may omit its speed test when every statement it covers is that kind's bare operation
_BARE_OPERATIONS: Final[dict[str, re.Pattern[str]]] = {
    "backup_verification": _BARE_COUNT,
    "schema_change": _BARE_MAINTENANCE,
}

#: a ``calls-whole-read`` hit's statement: ``<call> -> <path>:<line> <symbol>``
_REACHED: Final[re.Pattern[str]] = re.compile(r" -> (?P<path>[^ :]+):\d+ ")


def _declared_relation_outside(hit: WholeReadHit, schema: SchemaCatalog) -> bool:
    """whether a hit's relation is no declared L3 table (a process-local buffer, a warehouse relation). A run-time
    relation a declared producer yields (``produced.<entry>``, the repo's ``relation_producers``) is none: the config
    declares only the key it is read by (a warehouse relation's ``()``), never a table of the data layer.

    :param hit: the hit
    :ptype hit: WholeReadHit
    :param schema: the declared tables
    :ptype schema: SchemaCatalog
    :return: whether it is outside the declared tables
    :rtype: bool
    """
    relation = hit.symbol.split(" ", 1)[1] if " " in hit.symbol else ""
    found = schema.table(relation) if relation else None
    return found is None or not found.threetears or relation.startswith(PRODUCED_PREFIX)


#: a release as ``until=`` names it
_RELEASE: Final[re.Pattern[str]] = re.compile(r"(\d+)\.(\d+)\.(\d+)")

#: a package's own release in its ``pyproject.toml``
_PROJECT_VERSION: Final[re.Pattern[str]] = re.compile(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)[^"]*"', re.MULTILINE)


def _package_release(repo_root: Path, path: str) -> tuple[int, int, int] | None:
    """the release of the package holding a module: the ``version`` of the nearest ``pyproject.toml`` above it.

    :param repo_root: the repo root
    :ptype repo_root: Path
    :param path: the module, repo-relative
    :ptype path: str
    :return: the release, or None when no ``pyproject.toml`` above it names one
    :rtype: tuple[int, int, int] | None
    """
    root = repo_root.resolve()
    folder = (root / path).parent
    while folder.is_relative_to(root):
        project = folder / "pyproject.toml"
        found = _PROJECT_VERSION.search(project.read_text(encoding="utf-8")) if project.is_file() else None
        if found is not None:
            major, minor, patch = (int(part) for part in found.groups())
            return (major, minor, patch)
        if folder == root:
            break
        folder = folder.parent
    return None


def _until_defects(repo_root: Path, declared: DeclaredWhole) -> list[str]:
    """what is wrong with a declaration's ``until``: a ``rolling_upgrade`` names the release it is gone by, and is
    refused from that release on; no other kind names one.

    :param repo_root: the repo root
    :ptype repo_root: Path
    :param declared: the declaration
    :ptype declared: DeclaredWhole
    :return: its defects
    :rtype: list[str]
    """
    if declared.kind != "rolling_upgrade":
        return ["until= is a rolling_upgrade's alone"] if declared.until else []
    until = _RELEASE.fullmatch(declared.until)
    if until is None:
        return ['a rolling_upgrade names until="<major>.<minor>.<patch>", the release it is gone by, as a literal']
    release = _package_release(repo_root, declared.path)
    if release is None:
        return ["no pyproject.toml above the module names its release, so its until= cannot be judged"]
    major, minor, patch = (int(part) for part in until.groups())
    if release >= (major, minor, patch):
        return [
            f"the roll is over: the package is at {'.'.join(map(str, release))}, at or past until={declared.until!r}; "
            "delete the read that served the older writers"
        ]
    return []


def declaration_defects(repo_root: Path, declared: DeclaredWhole) -> list[str]:
    """what is wrong with a whole-read declaration's own arguments: its kind, reason, ruling, proof, speed test and
    ``until=``. Every gate letting a declared read through judges the declaration by this (no_whole_reads;
    epoch_discipline's whole-reread), so a declaration one gate refuses lets nothing through in another.

    :param repo_root: the repo root (proof and speed tests are read under it)
    :ptype repo_root: Path
    :param declared: the declaration
    :ptype declared: DeclaredWhole
    :return: its defects; empty when it is sound
    :rtype: list[str]
    """
    defects: list[str] = []
    if declared.kind not in DECLARED_KINDS:
        return [f"kind must be one of {sorted(DECLARED_KINDS)}, as a literal"]
    if len(declared.reason) < _MIN_REASON:
        defects.append(f"the reason must say which situation reads whole ({_MIN_REASON}+ characters, a literal)")
    if _RULING.search(declared.ruling) is None:
        defects.append('the ruling must cite Pace\'s ruling with its date ("Pace 2026-11-12: ..."), a literal')
    line = _Exemption(
        declared.line, declared.path, declared.function, 1, declared.kind, declared.proof, declared.reason
    )
    proof = _proving_test_defect(repo_root, line)
    if proof is not None:
        defects.append(f"the proof {proof}")
    defects.extend(_until_defects(repo_root, declared))
    if declared.kind in DECLARED_SPEED_KINDS and not (declared.kind in _BARE_OPERATIONS and not declared.speed):
        # a backup_verification or schema_change with no speed test is judged once its statements are known
        speed = _proving_test_defect(repo_root, line, test=declared.speed, calls="budget_problem")
        if speed is not None:
            defects.append(f"the speed test {speed}")
    return defects


def _reached_function(hit: WholeReadHit) -> tuple[str, str]:
    """the function a ``calls-whole-read`` hit reaches.

    :param hit: the hit
    :ptype hit: WholeReadHit
    :return: its path and qualified name
    :rtype: tuple[str, str]
    """
    reached = _REACHED.search(hit.statement)
    return (reached.group("path") if reached else "", hit.symbol.split(" ", 1)[-1])


class _Declared:
    """the repo's declarations judged: which are sound, and which hits each covers."""

    def __init__(self, config: NoWholeReadsConfig, scan: WholeReadScan, hits: list[WholeReadHit]) -> None:
        """judge every declaration against its hits.

        :param config: the repo's config
        :ptype config: NoWholeReadsConfig
        :param scan: the walk
        :ptype scan: WholeReadScan
        :param hits: every hit
        :ptype hits: list[WholeReadHit]
        """
        self.findings: list[str] = []
        self.accepted: list[DeclaredWhole] = []
        self.covered: set[WholeReadHit] = set()
        sound: list[DeclaredWhole] = []
        for declared in scan.declarations:
            defects = declaration_defects(config.repo_root, declared)
            where = f"{declared.path}:{declared.line} {declared.function} whole_read(kind={declared.kind!r})"
            self.findings.extend(f"{where}: {defect}" for defect in defects)
            if not defects:
                sound.append(declared)
        # a function's declared kinds, for a call into it: the same kind must be declared at the call
        kinds_of: dict[tuple[str, str], set[str]] = {}
        undeclared: set[tuple[str, str]] = set()
        for hit in hits:
            owners = [declared for declared in sound if declared.covers(hit.path, hit.function, hit.line)]
            if owners and all(self._lets(declared, hit, scan.schema) for declared in owners):
                # a constant's hit is the declaring function's read (its callers declare the same kind)
                for owner in owners:
                    kinds_of.setdefault((owner.path, owner.function), set()).add(owner.kind)
            elif hit.kind != "calls-whole-read":
                undeclared.add((hit.path, hit.function))
        used: set[int] = set()
        for hit in hits:
            for index, declared in enumerate(sound):
                if not declared.covers(hit.path, hit.function, hit.line) or not self._lets(declared, hit, scan.schema):
                    continue
                if hit.kind == "calls-whole-read":
                    target = self._target(hit)
                    if target in undeclared or kinds_of.get(target) != {declared.kind}:
                        continue
                if hit.kind == "whole-call" and any(
                    target in undeclared or kinds_of.get(target) != {declared.kind} for target in hit.reaches
                ):
                    # a call into the repo's own functions: each must be declared there with the same kind (one the
                    # walk cannot find in the repo has no declaration to check, and is the call's alone)
                    continue
                self.covered.add(hit)
                used.add(index)
        bare = {kind: self._bare_functions(sound, hits, pattern) for kind, pattern in _BARE_OPERATIONS.items()}
        for index, declared in enumerate(sound):
            if index in used and declared.kind in _BARE_OPERATIONS and not declared.speed:
                pattern = _BARE_OPERATIONS[declared.kind]
                mine = [hit for hit in self.covered if declared.covers(hit.path, hit.function, hit.line)]
                if not all(
                    (hit.kind == "calls-whole-read" and self._target(hit) in bare[declared.kind])
                    or (hit.kind != "calls-whole-read" and pattern.match(hit.statement) is not None)
                    for hit in mine
                ):
                    self.covered.difference_update(mine)
                    operation = (
                        "a bare count(*) or row checksum over one table (no join, filter or ordering)"
                        if declared.kind == "backup_verification"
                        else "a bare ANALYZE or REINDEX of one relation"
                    )
                    self.findings.append(
                        f"{declared.path}:{declared.line} {declared.function} whole_read(kind={declared.kind!r}): "
                        f"names no speed test, which only {operation} may omit: give speed=<test file>::<test "
                        "function> calling budget_problem"
                    )
                    continue
            if index in used:
                self.accepted.append(declared)
            else:
                self.findings.append(
                    f"{declared.path}:{declared.line} {declared.function} whole_read(kind={declared.kind!r}): covers "
                    "no whole read of its kind (a keyed read needs no declaration; a call into a function must be "
                    "declared there with the same kind): remove it or fix what it wraps"
                )

    @staticmethod
    def _target(hit: WholeReadHit) -> tuple[str, str]:
        """the function a ``calls-whole-read`` hit reaches.

        :param hit: the hit
        :ptype hit: WholeReadHit
        :return: its path and qualified name
        :rtype: tuple[str, str]
        """
        return _reached_function(hit)

    def _bare_functions(
        self, sound: list[DeclaredWhole], hits: list[WholeReadHit], pattern: re.Pattern[str]
    ) -> set[tuple[str, str]]:
        """the functions whose every whole read is a declared bare operation (``pattern``: :data:`_BARE_COUNT`,
        :data:`_BARE_MAINTENANCE`).

        :param sound: the sound declarations
        :ptype sound: list[DeclaredWhole]
        :param hits: every hit
        :ptype hits: list[WholeReadHit]
        :param pattern: the bare operation
        :ptype pattern: re.Pattern[str]
        :return: their paths and qualified names
        :rtype: set[tuple[str, str]]
        """
        reads: dict[tuple[str, str], list[bool]] = {}
        for hit in hits:
            if hit.kind == "calls-whole-read":
                continue
            owners = [declared for declared in sound if declared.covers(hit.path, hit.function, hit.line)]
            places = {(owner.path, owner.function) for owner in owners} or {(hit.path, hit.function)}
            for place in places:
                reads.setdefault(place, []).append(bool(owners) and pattern.match(hit.statement) is not None)
        return {place for place, bare in reads.items() if all(bare)}

    @staticmethod
    def _lets(declared: DeclaredWhole, hit: WholeReadHit, schema: SchemaCatalog) -> bool:
        """whether a declaration's kind lets a hit through (before a call's callee is checked).

        :param declared: the declaration
        :ptype declared: DeclaredWhole
        :param hit: the hit it wraps
        :ptype hit: WholeReadHit
        :param schema: the declared tables
        :ptype schema: SchemaCatalog
        :return: whether it does
        :rtype: bool
        """
        if hit.kind not in DECLARED_KINDS[declared.kind]:
            return False
        if declared.kind in DECLARED_OUTSIDE_KINDS and hit.kind != "calls-whole-read":
            return _declared_relation_outside(hit, schema)
        return True


class _Settled:
    """which functions' whole reads are all settled: each covered by a sound declaration, inside a sound exemption,
    or a call into a function so settled. A call into one is not flagged again at its caller (coordinator ruling,
    2026-11-13): the callee's declaration or exemption, with its proof and speed tests, covers the read wherever it
    is reached from. A call into a function holding any read neither declared nor exempt is still flagged."""

    def __init__(self, hits: list[WholeReadHit], covered: set[WholeReadHit], exemptions: list[_Exemption]) -> None:
        """index the hits by module.

        :param hits: every hit
        :ptype hits: list[WholeReadHit]
        :param covered: the hits sound declarations cover
        :ptype covered: set[WholeReadHit]
        :param exemptions: the sound exemptions
        :ptype exemptions: list[_Exemption]
        """
        self._by_path: dict[str, list[WholeReadHit]] = {}
        for hit in hits:
            self._by_path.setdefault(hit.path, []).append(hit)
        self._covered = covered
        self._exemptions = exemptions
        self._known: dict[tuple[str, str], bool] = {}

    def __call__(self, target: tuple[str, str], seen: frozenset[tuple[str, str]] = frozenset()) -> bool:
        """whether every whole read of a function (and of the functions defined inside it) is settled.

        :param target: the function's path and qualified name
        :ptype target: tuple[str, str]
        :param seen: the functions on the call chain being judged (a cycle settles nothing)
        :ptype seen: frozenset[tuple[str, str]]
        :return: whether it is
        :rtype: bool
        """
        if target in self._known:
            return self._known[target]
        path, function = target
        own = [
            hit
            for hit in self._by_path.get(path, [])
            if hit.function == function or hit.function.startswith(f"{function}.")
        ]
        answer = bool(own) and all(self._settled(hit, seen | {target}) for hit in own)
        self._known[target] = answer
        return answer

    def _settled(self, hit: WholeReadHit, seen: frozenset[tuple[str, str]]) -> bool:
        """whether one hit is settled.

        :param hit: the hit
        :ptype hit: WholeReadHit
        :param seen: the call chain being judged
        :ptype seen: frozenset[tuple[str, str]]
        :return: whether it is
        :rtype: bool
        """
        if hit in self._covered or any(_inside(hit, exemption) for exemption in self._exemptions):
            return True
        if hit.kind != "calls-whole-read":
            return False
        target = _reached_function(hit)
        return target not in seen and self(target, seen)


def _announce_declared(accepted: list[DeclaredWhole]) -> list[str]:
    """print and warn each declaration the gate accepts, so every one is seen on every run.

    :param accepted: the sound declarations covering a read
    :ptype accepted: list[DeclaredWhole]
    :return: the lines printed
    :rtype: list[str]
    """
    lines = [
        f"declared-whole {declared.kind} {declared.path}:{declared.line} {declared.function} -- {declared.ruling}"
        for declared in accepted
    ]
    for line in lines:
        sys.stderr.write(f"{line}\n")
        warnings.warn(line, _WholeReadAccepted, stacklevel=2)
    return lines


def _unused_declarations(config: NoWholeReadsConfig, scan: WholeReadScan) -> list[str]:
    """each loosening declaration of the config no judgement of the walk went through, and each transient pattern
    matching only what another matches: a dead allowance would let a new read through unjudged (reports #59).

    :param config: the repo's config
    :ptype config: NoWholeReadsConfig
    :param scan: the walk
    :ptype scan: WholeReadScan
    :return: the findings
    :rtype: list[str]
    """
    used = scan.declarations_used
    findings = [
        f"{field} {entry} matches nothing the walk read; delete it"
        for field, declared in (
            ("transient_producers", config.transient_producers),
            ("view_producers", config.view_producers),
            ("relation_producers", config.relation_producers),
        )
        for entry in sorted(declared)
        if (field, entry) not in used
    ]
    if config.bounded_producers and ("bounded_producers", "*") not in used:
        findings.extend(
            f"bounded_producers {entry} matches nothing the walk read; delete it"
            for entry in sorted(config.bounded_producers)
        )
    if config.local_relations_checked:
        findings.extend(
            f"local_relations {written} matches nothing the walk read; delete it"
            for written in sorted(config.local_relations)
            if f"{COPY_PREFIX}{'{}' if written == '*' else written.lower()}" not in scan.schema.copies_read
        )
    matched = scan.transient_matched
    for pattern, names in sorted(matched.items()):
        if not names:
            findings.append(f"transient_relations {pattern} matches nothing the walk read; delete it")
            continue
        covering = next(
            (other for other, theirs in sorted(matched.items()) if other != pattern and names <= theirs), None
        )
        if covering is not None and (pattern > covering or matched[covering] != names):
            findings.append(f"transient_relations {pattern} matches only what {covering} matches; delete it")
    return findings


def no_whole_reads_findings(config: NoWholeReadsConfig) -> list[str]:
    """every finding: each whole read outside an exempt function, and every ledger or walk defect.

    :param config: the repo's config
    :ptype config: NoWholeReadsConfig
    :return: findings, empty when the rule holds
    :rtype: list[str]
    """
    hits, scan = _hits(config)
    findings: list[str] = []
    if scan.files_scanned < config.minimum_files_scanned:
        findings.append(
            f"the walk reached {scan.files_scanned} modules, under the floor of {config.minimum_files_scanned}: "
            "check src_roots, since a walk that reads nothing reports nothing"
        )
    findings.extend(f"excluded prefix {prefix!r} matches no module; delete it" for prefix in scan.unmatched_exclusions)
    findings.extend(scan.bounded_findings)
    findings.extend(
        f"declared parameter {key!r} names no parameter of a function the walk reads (qualified as "
        "<Class>.<method>.<parameter>); fix or delete the declaration"
        for key in sorted(set(config.declared_parameters) - scan.declared_matched)
    )
    findings.extend(_unused_declarations(config, scan))
    exemptions, defects = _parse(config.ledger_dir / _EXEMPTIONS_NAME)
    findings.extend(defects)
    sound: list[_Exemption] = []
    for exemption in exemptions:
        where = f"{_EXEMPTIONS_NAME}:{exemption.file_line}"
        before = len(findings)
        if f"{exemption.path}::{exemption.function}" not in scan.definitions:
            findings.append(f"{where}: {exemption.path} :: {exemption.function} names no function the walk found")
        have = sum(1 for hit in hits if _inside(hit, exemption))
        if have != exemption.count:
            findings.append(
                f"{where}: {exemption.function} pins {exemption.count} hits but the code has {have}: "
                + ("a new whole read is not exempt by sitting beside one" if have > exemption.count else "lower it")
            )
        defect = _proving_test_defect(config.repo_root, exemption)
        if defect is not None:
            findings.append(f"{where}: the proving test {defect}")
        speed = _proving_test_defect(config.repo_root, exemption, test=exemption.speed_test, calls="budget_problem")
        if speed is not None:
            findings.append(f"{where}: the speed test {speed}")
        if len(findings) == before:
            sound.append(exemption)
    findings.extend(_operator_mark_findings(config, scan))
    declared = _Declared(config, scan, hits)
    findings.extend(declared.findings)
    _announce(scan)
    _announce_declared(declared.accepted)
    for exemption in exemptions:
        if exemption.kind == "source-gap":
            line = f"no_whole_reads accepts source-gap {exemption.path} :: {exemption.function}: {exemption.reason}"
            sys.stderr.write(f"{line}\n")
            warnings.warn(line, _WholeReadAccepted, stacklevel=2)
    findings.extend(scan.schema.ledger_findings)
    findings.extend(
        index_findings(
            scan.schema,
            config.repo_root,
            scan.definitions | scan.constants,
            consumer_purposes=config.consumer_purposes,
        )
    )
    settled = _Settled(hits, declared.covered, sound)
    for hit in hits:
        if _marked(hit, scan.operators, scan.schema) or hit in declared.covered:
            continue
        if hit.kind == "calls-whole-read" and settled(_reached_function(hit)):
            # the callee's whole reads are each declared or exempt, with their proofs and speed tests: its caller
            # is not flagged again for reaching them
            continue
        if not any(_inside(hit, exemption) for exemption in exemptions):
            findings.append(f"{hit.path}:{hit.line} in {hit.function}: {hit.symbol} -- {hit.statement} -- {_ADVICE}")
    return findings


def use_the_helper_findings(config: NoWholeReadsConfig) -> list[str]:
    """every SQL statement the repo writes as text against a data table outside the helpers, outside its
    ``_use_the_helper_exemptions.txt`` (:mod:`~threetears.enforcement.no_whole_reads.use_the_helper`).

    :param config: the repo's config (the same walk as :func:`no_whole_reads_findings`)
    :ptype config: NoWholeReadsConfig
    :return: the findings
    :rtype: list[str]
    """
    return statement_findings(
        _helper_statements(config),
        ledger_dir=config.ledger_dir,
        helper_modules=config.helper_modules,
        producers=_producers(config),
        caller_judged=config.caller_judged_modules,
    )


def use_the_helper_counts(config: NoWholeReadsConfig) -> dict[str, int]:
    """the repo's use-the-helper backlog by shape (``point``, ``list``, ``count``, ``join``, ``write``), with how many
    name a table only at run time and the total.

    :param config: the repo's config
    :ptype config: NoWholeReadsConfig
    :return: shape onto count
    :rtype: dict[str, int]
    """
    return statement_counts(
        _helper_statements(config),
        config.helper_modules,
        _producers(config),
        caller_judged=config.caller_judged_modules,
    )


def _helper_statements(config: NoWholeReadsConfig) -> list[WrittenStatement]:
    """the statements the use-the-helper rule judges: every one the repo writes as text, but a read declared whole by
    nature (no keyed helper answers it; the no-whole-reads gate judges its declaration).

    :param config: the repo's config
    :ptype config: NoWholeReadsConfig
    :return: the statements
    :rtype: list[WrittenStatement]
    """
    hits, scan = _hits(config)
    return _undeclared_statements(config, scan, hits)


def _undeclared_statements(
    config: NoWholeReadsConfig, scan: WholeReadScan, hits: list[WholeReadHit]
) -> list[WrittenStatement]:
    """a walk's statements but those a sound, accepted whole-read declaration covers: what the use-the-helper rule
    judges, and so all its codemod may plan (a declared read is the one no keyed helper answers).

    :param config: the repo's config
    :ptype config: NoWholeReadsConfig
    :param scan: the walk
    :ptype scan: WholeReadScan
    :param hits: every hit of the walk
    :ptype hits: list[WholeReadHit]
    :return: the statements
    :rtype: list[WrittenStatement]
    """
    accepted = _Declared(config, scan, hits).accepted
    return [
        statement
        for statement in scan.statements
        if not any(declared.covers(statement.path, statement.function, statement.line) for declared in accepted)
    ]


def _producers(config: NoWholeReadsConfig) -> frozenset[str]:
    """the repo's declared view and transient producers: the functions whose run-time relation names are local.

    :param config: the repo's config
    :ptype config: NoWholeReadsConfig
    :return: their names
    :rtype: frozenset[str]
    """
    return frozenset(config.view_producers) | config.transient_producers
