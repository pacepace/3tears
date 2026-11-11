"""use-the-helper: a read or write of a data table is written once, in 3tears' helpers, never as SQL text anywhere else.

Every keyed read has one home (:mod:`threetears.core.keyed`: a point read, a page, a count, a probe, a delete or
update by key; :mod:`threetears.core.keyset`: a walk of a table in key order; :class:`threetears.epoch.RowFollower`:
a cache that follows a table's changes by key), and the migration and DDL runners write the schema. A statement a
module writes as text against a data table anywhere else is a second way of reading it, and every second way is
one more place a whole read, a wrong index or a missed tenant can hide.

**The rule** (:func:`statement_findings`; per repo, ``threetears.enforcement.no_whole_reads.use_the_helper_findings``): every SQL statement a module writes as text -- a ``SELECT`` (or
``WITH ... SELECT``), an ``UPDATE``, a ``DELETE``, a ``MERGE``, an ``INSERT ... SELECT`` -- whose relations include
a data table (one the repo's or 3tears' declarations key: a ``TableSchema``, SQLAlchemy ``Table``, migration DDL,
``tables.yaml``), a table named at run time (``FROM {table}``) or a change relation (``change:<name>``: rows a
change registers, a stage; their steps are :mod:`threetears.core.keyed.steps`) is a finding,
``use-the-helper <shape> <table>``,
unless the module is one of the sanctioned helper modules (:data:`_HELPER_MODULES`, plus the repo's reviewed
``NoWholeReadsConfig.helper_modules``). A relation of a local copy (``copy.<name>``), a CTE, a catalog or a function
is no data table. Migrations are never walked.

**Its shapes** (:func:`~threetears.enforcement.no_whole_reads.sql.statement_shape`), the backlog's buckets: ``point``
(the rows a ``WHERE`` names), ``list`` (ordered, limited past a row, or unfiltered), ``count`` (a count, an
existence, a sum, a group), ``join``, ``write``.

**The one way out: the ledger.** A statement no helper can write yet is listed in the repo's
``_use_the_helper_exemptions.txt`` beside the no-whole-reads ledger, one line per function::

    <path> :: <qualified function> :: <statements> :: <reason>

``<statements>`` pins the function's count exactly (a new statement in it fails, and so does a line whose
statements went); the reason (30+ characters) says why no helper writes it and which helper should. There is no
allowlist of shapes or tables: where no helper fits, the helper set grows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from threetears.enforcement.no_whole_reads.schema import COPY_PREFIX
from threetears.enforcement.no_whole_reads.sql import change_relations, data_relations, statement_shape
from threetears.enforcement.no_whole_reads.walker import WrittenStatement

__all__ = [
    "statement_counts",
    "statement_findings",
]

#: the ledger's name, beside ``_no_whole_reads_exemptions.txt``
_LEDGER: Final[str] = "_use_the_helper_exemptions.txt"

#: the modules that write a data table's SQL for everyone else, as a fragment of their repo-relative path: the keyed
#: helpers, the keyset walk, the row follower, the migration and DDL runners
_HELPER_MODULES: Final[tuple[str, ...]] = (
    "threetears/core/keyed/",
    # the keyed helpers' pure builders, in the dependency-free interfaces package
    "threetears/interfaces/keyed/",
    "threetears/core/keyset.py",
    "threetears/epoch/row_follower.py",
    "threetears/core/data/migrations/",
    "threetears/core/ddl/",
    # the L1 storage layer itself: its backends write the local tables every cache reads through (no_whole_reads still
    # judges each statement in them)
    "threetears/core/cache/",
)

_SEPARATOR: Final[str] = " :: "
_MIN_REASON: Final[int] = 30

#: what each finding is told to do instead
_ADVICE: Final[str] = (
    "write it through threetears.core.keyed (read_keyed, keyset_page, counts_by, holds_any, delete_before, "
    "update_before, read_after; over a change relation its steps: semi, anti, distinct, grouped, latest_per_key, "
    "windowed, except_rows, union, read_step, delete_matching), threetears.core.keyset or a RowFollower; where none "
    "fits, add the helper to 3tears, "
    f"never the SQL here; until then it goes in {_LEDGER} with the reason"
)


@dataclass(frozen=True)
class _Hit:
    """one statement written outside the helpers.

    :ivar statement: where and what
    :ivar shape: ``point``, ``list``, ``count``, ``join`` or ``write``
    :ivar table: the first data table it names (``{}`` for one named at run time)
    :ivar declared: whether that table is declared (False for a run-time name)
    """

    statement: WrittenStatement
    shape: str
    table: str
    declared: bool


@dataclass(frozen=True)
class _Line:
    """one ledger line."""

    number: int
    path: str
    function: str
    count: int
    reason: str


def _sanctioned(path: str, extra: tuple[str, ...]) -> bool:
    """whether ``path`` is a helper module: a :data:`_HELPER_MODULES` fragment, or one of the repo's reviewed prefixes.

    :param path: repo-relative posix path
    :ptype path: str
    :param extra: the repo's own helper module prefixes
    :ptype extra: tuple[str, ...]
    :return: whether it is
    :rtype: bool
    """
    return any(fragment in f"/{path}" for fragment in _HELPER_MODULES) or any(path.startswith(p) for p in extra)


#: the schema a local DuckDB copy holds its tables in
_LOCAL_SCHEMA: Final[str] = "main."


def _local(name: str, statement: WrittenStatement, producers: frozenset[str]) -> bool:
    """whether a relation is a local copy's, not L3 or shared data: ``copy.<name>``; a ``main.<name>`` the repo's
    copy declarations key (``local_relations``, ``"*"`` for one named at run time); a run-time name written in a
    declared view or transient producer (the copy relation it builds or the change set it names).

    :param name: the relation as written
    :ptype name: str
    :param statement: the statement naming it
    :ptype statement: WrittenStatement
    :param producers: the repo's declared view and transient producers
    :ptype producers: frozenset[str]
    :return: whether it is local
    :rtype: bool
    """
    local = name.startswith(COPY_PREFIX)
    if not local and name.startswith(_LOCAL_SCHEMA):
        local = statement.reader.schema.table(f"{COPY_PREFIX}{name.removeprefix(_LOCAL_SCHEMA)}") is not None
    if not local and "{" in name:
        local = statement.function in producers
    return local


def _data_table(statement: WrittenStatement, producers: frozenset[str] = frozenset()) -> tuple[str, bool] | None:
    """the first data table a statement names, and whether it is declared; None when it names none.

    :param statement: the statement
    :ptype statement: WrittenStatement
    :param producers: the repo's declared view and transient producers (:func:`_local`)
    :ptype producers: frozenset[str]
    :return: ``(table, declared)``, or None
    :rtype: tuple[str, bool] | None
    """
    for name in data_relations(statement.text, statement.reader):
        if _local(name, statement, producers):
            continue
        if statement.reader.schema.table(name) is not None:
            return name, True
        if "{" in name:
            return name, False
    for name in change_relations(statement.text, statement.reader):
        # a relational step over a change relation (its produced rows, a stage, rows registered from memory) has one
        # home too: threetears.core.keyed.steps
        return f"change:{name}", True
    return None


def _helper_hits(
    statements: list[WrittenStatement], helper_modules: tuple[str, ...] = (), producers: frozenset[str] = frozenset()
) -> list[_Hit]:
    """every statement written as text against a data table outside the helper modules.

    :param statements: the statements a walk found
    :ptype statements: list[WrittenStatement]
    :param helper_modules: the repo's own reviewed helper module prefixes
    :ptype helper_modules: tuple[str, ...]
    :param producers: the repo's declared view and transient producers, whose run-time names are local relations
    :ptype producers: frozenset[str]
    :return: the hits, in walk order
    :rtype: list[_Hit]
    """
    hits: list[_Hit] = []
    for statement in statements:
        if _sanctioned(statement.path, helper_modules):
            continue
        shape = statement_shape(statement.text)
        table = _data_table(statement, producers) if shape is not None else None
        if shape is not None and table is not None:
            hits.append(_Hit(statement, shape, table[0], table[1]))
    return hits


def _parse(path: Path) -> tuple[list[_Line], list[str]]:
    """read the ledger.

    :param path: the ledger; a missing file reads as empty
    :ptype path: Path
    :return: the lines, and every malformed line as a finding
    :rtype: tuple[list[_Line], list[str]]
    """
    lines: list[_Line] = []
    defects: list[str] = []
    seen: set[tuple[str, str]] = set()
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split(_SEPARATOR)]
        where = f"{path.name}:{number}"
        if len(parts) != 4:
            defects.append(
                f"{where}: expected 4 fields separated by '{_SEPARATOR.strip()}' (path, function, statements, "
                f"reason), got {len(parts)}"
            )
            continue
        module, function, count, reason = parts
        if not count.isdigit() or int(count) < 1:
            defects.append(f"{where}: statements {count!r} must be a positive integer")
        elif len(reason) < _MIN_REASON:
            defects.append(f"{where}: the reason must say why no helper writes it ({_MIN_REASON}+ characters)")
        elif (module, function) in seen:
            defects.append(f"{where}: {module} :: {function} is listed twice")
        else:
            seen.add((module, function))
            lines.append(_Line(number, module, function, int(count), reason))
    return lines, defects


def _finding(hit: _Hit) -> str:
    """one hit as a finding line.

    :param hit: the hit
    :ptype hit: _Hit
    :return: ``<path>:<line> in <function>: use-the-helper <shape> <table> -- <statement> -- <advice>``
    :rtype: str
    """
    statement = " ".join(hit.statement.text.split())
    shown = statement if len(statement) <= 200 else f"{statement[:197]}..."
    where = f"{hit.statement.path}:{hit.statement.line} in {hit.statement.function}"
    return f"{where}: use-the-helper {hit.shape} {hit.table} -- {shown} -- {_ADVICE}"


def statement_findings(
    statements: list[WrittenStatement],
    *,
    ledger_dir: Path,
    helper_modules: tuple[str, ...] = (),
    producers: frozenset[str] = frozenset(),
) -> list[str]:
    """every statement written as text against a data table outside the helpers, and every ledger defect.

    :param statements: the statements a walk found (``WholeReadScan.statements``)
    :ptype statements: list[WrittenStatement]
    :param ledger_dir: the directory holding :data:`_LEDGER`
    :ptype ledger_dir: Path
    :param helper_modules: the repo's own reviewed helper module prefixes
    :ptype helper_modules: tuple[str, ...]
    :param producers: the repo's declared view and transient producers, whose run-time names are local
    :ptype producers: frozenset[str]
    :return: the findings: hits outside the ledger, a ledger line whose count differs from the function's
        statements, a line naming a function with none, and malformed lines
    :rtype: list[str]
    """
    hits = _helper_hits(statements, helper_modules, producers)
    lines, findings = _parse(ledger_dir / _LEDGER)
    by_function: Counter[tuple[str, str]] = Counter((hit.statement.path, hit.statement.function) for hit in hits)
    ledgered = {(line.path, line.function): line for line in lines}
    for hit in hits:
        if (hit.statement.path, hit.statement.function) not in ledgered:
            findings.append(_finding(hit))
    for key, line in ledgered.items():
        found = by_function.get(key, 0)
        if found != line.count:
            findings.append(
                f"{_LEDGER}:{line.number}: {line.path} :: {line.function} pins {line.count} "
                f"statement(s), the walk finds {found}: update the line, or drop it once a helper writes them"
            )
    return findings


def statement_counts(
    statements: list[WrittenStatement], helper_modules: tuple[str, ...] = (), producers: frozenset[str] = frozenset()
) -> dict[str, int]:
    """the backlog by shape: how many statements of each shape are written outside the helpers, ledgered or not,
    and how many of them name a table only at run time (``named-at-run-time``).

    :param statements: the statements a walk found
    :ptype statements: list[WrittenStatement]
    :param helper_modules: the repo's own reviewed helper module prefixes
    :ptype helper_modules: tuple[str, ...]
    :param producers: the repo's declared view and transient producers, whose run-time names are local
    :ptype producers: frozenset[str]
    :return: shape onto count, plus ``named-at-run-time`` and ``total``
    :rtype: dict[str, int]
    """
    hits = _helper_hits(statements, helper_modules, producers)
    counts: dict[str, int] = {shape: 0 for shape in ("point", "list", "count", "join", "write")}
    for hit in hits:
        counts[hit.shape] += 1
    counts["named-at-run-time"] = sum(1 for hit in hits if not hit.declared)
    counts["total"] = len(hits)
    return counts
