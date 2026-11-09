"""no-whole-reads enforcement domain: load, cache and follower code reads by key, never a whole table.

An epoch IS its change set. A pod learns what changed from the change log and its change notes and reads exactly
those rows by key; data from outside comes in by the source's own change records (its runs, its upload stamps).
Nothing reads a whole table to find out what changed, re-verifies its own data by scanning it, or falls back to a
full reload when something goes wrong. A whole read is right in exactly two places: the one cold start that fills
an empty copy, and a schema change that rewrites the table.

**The rule, statically** (:func:`~threetears.enforcement.no_whole_reads.walker.scan_whole_reads`, AST plus the SQL
the code writes, judged by :func:`~threetears.enforcement.no_whole_reads.sql.judge_sql`):

- every SQL statement against a data table carries a key, partition or scope predicate, or is one ``LIMIT``-ed
  statement: no ``SELECT ... FROM <table>`` without ``WHERE`` (``whole-select``), no ``WHERE`` without a bound
  value (``unbound-where``);
- no ``COUNT``, ``GROUP BY``, ``DISTINCT``, sum or fingerprint over a whole table (``whole-aggregate``);
- no ``TRUNCATE``, no ``DELETE`` / ``UPDATE`` without ``WHERE``, no ``ANALYZE`` / ``VACUUM`` of a table, no
  backend ``truncate()``;
- no pager or fingerprint helper called without a narrowing keyword (``unkeyed-pager``, ``may-read-whole``), no
  helper that always reads whole (``whole-read``);
- no call whose name says whole, rebuild, warm, reload or resync (``whole-call``), no flag asking for one
  (``whole-flag``), no table put in a set of tables to read whole (``marks-whole``);
- no call into a function of the repo that does any of the reads above (``calls-whole-read``), so the caller is
  named, not only the helper.

**The one way out: the exemptions ledger.** A function that is a true cold start, or a schema change, is listed
in the repo's ``_no_whole_reads_exemptions.txt``, one line each::

    <repo-relative path> :: <qualified function> :: <hits> :: <cold-start|schema-change> :: <proving test> :: <reason>

- ``<hits>`` pins the function's hit count exactly: a new whole read added inside an exempt function fails, and so
  does a line whose hits went;
- ``<proving test>`` is ``<repo-relative test file>::<test function>``: a test that proves the function is reached
  ONLY on a cold start (an empty copy, no checkpoint) or a schema change, never on a steady-state load, a retry or
  a recovery. The gate checks the file defines that test and that the test names the function;
- ``<reason>`` says which cold start or schema change, in 30 characters or more.

A line is never "it's small", "it's an admin path", "it's being fixed" or "pre-existing": those are violations,
fixed, not listed. There is no allowlist of violations; the gate is red until each is fixed.

**The runtime half** is :class:`threetears.core.testing.plans.WholeScanGuard`: a load-path test wraps its
connection, and every statement whose plan scans a table past a row threshold, or whose pages add up to most of
one, fails the test.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from threetears.enforcement.no_whole_reads.walker import (
    DEFAULT_NARROWING_KEYWORDS,
    DEFAULT_PAGER_CALLS,
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

#: the only reasons a function may read whole
_EXEMPTION_KINDS: Final[frozenset[str]] = frozenset({"cold-start", "schema-change"})

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
    """

    file_line: int
    path: str
    function: str
    count: int
    kind: str
    test: str
    reason: str


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
        if len(parts) != 6:
            defects.append(f"{where}: expected 6 fields separated by '{_SEPARATOR.strip()}', got {len(parts)}")
            continue
        module, function, count, kind, test, reason = parts
        if kind not in _EXEMPTION_KINDS:
            defects.append(f"{where}: {kind!r} is not a reason to read whole; only {sorted(_EXEMPTION_KINDS)} are")
        elif not count.isdigit() or int(count) < 1:
            defects.append(f"{where}: hits {count!r} must be a positive integer")
        elif len(reason) < _MIN_REASON:
            defects.append(f"{where}: the reason must name the cold start or schema change ({_MIN_REASON}+ characters)")
        elif (module, function) in seen:
            defects.append(f"{where}: {module} :: {function} is listed twice")
        else:
            seen.add((module, function))
            lines.append(_Exemption(number, module, function, int(count), kind, test, reason))
    return lines, defects


def _proving_test_defect(repo_root: Path, exemption: _Exemption) -> str | None:
    """what is wrong with an exemption's proving test, if anything.

    :param repo_root: the repo root
    :ptype repo_root: Path
    :param exemption: the line
    :ptype exemption: _Exemption
    :return: the defect, or None when the file defines the test and the test names the function
    :rtype: str | None
    """
    module, _, name = exemption.test.partition("::")
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
    return defect


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
    for hit in hits:
        if not any(_inside(hit, exemption) for exemption in exemptions):
            findings.append(f"{hit.path}:{hit.line} in {hit.function}: {hit.symbol} -- {hit.statement} -- {_ADVICE}")
    return findings
