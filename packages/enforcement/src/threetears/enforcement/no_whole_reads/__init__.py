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
from threetears.enforcement.no_whole_reads.schema import COPY_PREFIX, SchemaCatalog, collect_schema, index_findings
from threetears.enforcement.no_whole_reads.walker import (
    DEFAULT_NARROWING_KEYWORDS,
    DEFAULT_PAGER_CALLS,
    OperatorMark,
    WholeReadHit,
    WholeReadScan,
    scan_whole_reads,
)

__all__ = [
    "NoWholeReadsConfig",
    "no_whole_reads_findings",
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
    :ivar transient_relations: a regular expression of relations that are a change set being applied (a stage of
        rows a change named), never a data table; empty for none
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
    :ivar dependency_schema: whether 3tears' own declarations (its change log, notes and collections, which a
        consumer reads) key reads too; its indexes are never the repo's to give a purpose
    """

    repo_root: Path
    src_roots: tuple[Path, ...]
    ledger_dir: Path
    excluded_prefixes: tuple[str, ...] = ()
    extra_pagers: frozenset[str] = field(default_factory=frozenset)
    extra_narrowing: frozenset[str] = field(default_factory=frozenset)
    transient_relations: str = ""
    verify_prefixes: tuple[str, ...] = ()
    minimum_files_scanned: int = 1
    schema_roots: tuple[Path, ...] = ()
    table_lists: tuple[Path, ...] | None = None
    dependency_schema: bool = True
    extra_exec_calls: frozenset[str] = field(default_factory=frozenset)
    local_relations: Mapping[str, tuple[str, ...] | tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    foreign_relations: Mapping[str, tuple[str, ...] | tuple[tuple[str, ...], ...]] = field(default_factory=dict)


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
    for prefix, relations in ((COPY_PREFIX, config.local_relations), ("", config.foreign_relations)):
        for written, declared in relations.items():
            # ``local_relations["*"]``: the default key of every copy relation the repo declares no key for
            name = "{}" if written == "*" and prefix else written
            keys: list[tuple[str, ...]] = (
                [tuple(str(column) for column in declared)]
                if all(isinstance(column, str) for column in declared)
                else [tuple(str(column) for column in key) for key in declared if not isinstance(key, str)]
            )
            for key in keys:
                catalog.add_key(f"{prefix}{name.lower()}", key, "NoWholeReadsConfig")
    return catalog


def _hits(config: NoWholeReadsConfig) -> tuple[list[WholeReadHit], WholeReadScan]:
    """every hit of the walk, ledger ignored: the repo's list of whole reads.

    :param config: the repo's config
    :ptype config: NoWholeReadsConfig
    :return: the hits, and the walk
    :rtype: tuple[list[WholeReadHit], WholeReadScan]
    """
    scan = scan_whole_reads(
        config.repo_root,
        config.src_roots,
        excluded_prefixes=config.excluded_prefixes,
        pagers=DEFAULT_PAGER_CALLS | config.extra_pagers,
        narrowing=DEFAULT_NARROWING_KEYWORDS | config.extra_narrowing,
        transient=config.transient_relations,
        verify_prefixes=config.verify_prefixes,
        schema=_schema_of(config),
        exec_calls=config.extra_exec_calls,
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


def _marked(hit: WholeReadHit, marks: list[OperatorMark]) -> bool:
    """whether a hit sits in a function (or one nested in it) whose operator mark lets its kind through.

    :param hit: the hit
    :ptype hit: WholeReadHit
    :param marks: the operator marks
    :ptype marks: list[OperatorMark]
    :return: whether it does
    :rtype: bool
    """
    return any(
        hit.path == mark.path
        and (hit.function == mark.function or hit.function.startswith(f"{mark.function}."))
        and hit.kind in mark.kinds
        for mark in marks
    )


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
    exemptions, defects = _parse(config.ledger_dir / _EXEMPTIONS_NAME)
    findings.extend(defects)
    for exemption in exemptions:
        where = f"{_EXEMPTIONS_NAME}:{exemption.file_line}"
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
    findings.extend(_operator_mark_findings(config, scan))
    _announce(scan)
    for exemption in exemptions:
        if exemption.kind == "source-gap":
            line = f"no_whole_reads accepts source-gap {exemption.path} :: {exemption.function}: {exemption.reason}"
            sys.stderr.write(f"{line}\n")
            warnings.warn(line, _WholeReadAccepted, stacklevel=2)
    findings.extend(index_findings(scan.schema, config.repo_root, scan.definitions | scan.constants))
    for hit in hits:
        if _marked(hit, scan.operators):
            continue
        if not any(_inside(hit, exemption) for exemption in exemptions):
            findings.append(f"{hit.path}:{hit.line} in {hit.function}: {hit.symbol} -- {hit.statement} -- {_ADVICE}")
    return findings
