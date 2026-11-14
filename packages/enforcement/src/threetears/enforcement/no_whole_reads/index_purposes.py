"""the index-purpose tool: give each declared index the purpose, plan test and reason the purpose rule asks of it, from
the reads the walk sees use it.

``python -m threetears.enforcement.no_whole_reads.index_purposes <repo root> <support module> <config name> [--write]``

The walk runs as the gate's does, but with every declared index counted as purposed for every read
(:attr:`~.schema.SchemaCatalog.discovering`), so each keyed read is credited to every index that would serve it, with
the statement it was judged through (:attr:`~.schema.SchemaCatalog.matches`). For each index of the repo the rule
finds no purpose for, the tool picks the reads whose statement binds the longest leading run of its columns (then
orders by the next one): their functions are its purpose. An index no read is credited to, but whose leading columns
are a declared foreign key's referencing columns, serves the database's lookup of those rows (``fk:<constraint>``).
A unique index the rule finds no purpose for holds a data rule, so it is never a drop candidate: it is declared
``purpose=constraint``, its test a generated duplicate insert the database must refuse
(:func:`threetears.core.testing.rewrites.insert_duplicate` under ``pytest.raises``), a partial one's rows held to its
``WHERE``'s literals (one it cannot read so is left for a hand-written test). An index neither serves is a drop
candidate: listed for the owner, never given a purpose.

Its plan test is generated: one test per index in ``tests/integration/test_index_plans_generated.py``, run on
PostgreSQL and YugabyteDB alike, which seeds the table on the migrated test database and explains the read's shape --
the columns its purpose's read binds, in the index's order, then the column it orders by, under the index's own
partial ``WHERE`` -- asserting the plan reads through the index
(:func:`threetears.core.testing.rewrites.explain_index_read`, then ``plan_uses_index``). Each test's statement and
bindings sit in ``tests/integration/index_plan_manifest.json``.

A crowded table's (more than :data:`~.schema.MAX_PLAIN_INDEXES` indexes) or a wide index's reason is generated from the
reads it serves. Every declaration is written to the repo's index declarations ledger
(:mod:`~.declarations`, frozen migrations' indexes); an index whose purpose sits beside its DDL already gets a
reason-only supplement line. Approximate-nearest-neighbour and GIN indexes, and an index whose purpose names a read the
walk does not credit it with, are listed for a hand-written test: the tool writes nothing for them.

Without ``--write`` it prints what it would write.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, _hits
from threetears.enforcement.no_whole_reads.declarations import DECLARATIONS_NAME
from threetears.enforcement.no_whole_reads.schema import (
    CONSTRAINT_PURPOSE,
    MAX_PLAIN_INDEXES,
    MIN_REASON,
    NEAREST_METHODS,
    SchemaCatalog,
    _DeclaredIndex,
    _DeclaredTable,
    _not_null_of,
    purposes,
)

__all__ = ["main"]

#: the generated plan-test module and its manifest, repo-relative
_MODULE: Final[str] = "tests/integration/test_index_plans_generated.py"
_MANIFEST: Final[str] = "tests/integration/index_plan_manifest.json"

#: the most reads one purpose names: the strongest few, every one of which the plan test's shape stands for
_MOST_PURPOSES: Final[int] = 3

#: access methods whose reads are no equality/order shape the generated test can explain
_HAND_METHODS: Final[frozenset[str]] = NEAREST_METHODS | {"gin", "gist", "brin"}


@dataclass
class IndexDecision:
    """what the tool writes for one index.

    :ivar table: the bare table
    :ivar index: the index
    :ivar origin: ``<path>:<line>`` of its first declaration
    :ivar purpose: the purpose to write; empty for a supplement (the purpose is inline already)
    :ivar plan_test: the generated plan test, ``<module>::<test>``; empty for a supplement
    :ivar reason: the reason to write; empty when none is asked
    :ivar bound: the columns the plan test's read binds, in the index's order
    :ivar ordered: the column it orders by after them, or ""
    :ivar descending: whether its read orders by that column descending
    :ivar also: the other columns its read binds to a parameter by equality (not the index's), as the read sends them
    :ivar where: the index's partial ``WHERE``, kept in the read so the planner may use it
    :ivar refuses: a unique index declared by its constraint: its test inserts a duplicate the database must refuse
    :ivar literals: a partial unique index's ``WHERE`` as ``(column, literal)`` equalities its two rows hold
    """

    table: str
    index: str
    origin: str
    purpose: str = ""
    plan_test: str = ""
    reason: str = ""
    bound: tuple[str, ...] = ()
    ordered: str = ""
    descending: bool = False
    also: tuple[str, ...] = ()
    where: str = ""
    refuses: bool = False
    literals: tuple[tuple[str, Any], ...] = ()


@dataclass
class Decisions:
    """the tool's verdict over a repo.

    :ivar written: what it writes, per index
    :ivar drop: ``(table, index, origin)`` of each index no read and no foreign key uses: candidates for removal
    :ivar hand: ``(table, index, why)`` of each index left for a hand-written plan test or purpose
    """

    written: list[IndexDecision] = field(default_factory=list)
    drop: list[tuple[str, str, str]] = field(default_factory=list)
    hand: list[tuple[str, str, str]] = field(default_factory=list)


def _column(column: str) -> str:
    """an index column as a read names it: its name, without ``ASC``/``DESC``/``NULLS ...``."""
    return column.split()[0].strip('"').lower()


def _where_part(statement: str) -> str:
    """the statement from its first ``WHERE`` on (an ``UPDATE``'s ``SET`` binds nothing)."""
    found = re.search(r"\bWHERE\b", statement, re.IGNORECASE)
    return statement[found.start() :] if found is not None else ""


def _binds(statement: str, column: str) -> bool:
    """whether the statement binds ``column`` to a value or a set: ``=``, ``IN``, ``= ANY``, null-safe equality."""
    pattern = rf'(?<![\w."]){_qualified(column)}\s*(?:=(?!\s*\w+\s*\.)|IN\s*\(|IS\s+NOT\s+DISTINCT\s+FROM)'
    return re.search(pattern, _where_part(statement), re.IGNORECASE) is not None


def _orders(statement: str, column: str) -> bool:
    """whether the statement orders by ``column`` (first)."""
    return re.search(rf"ORDER\s+BY\s+{_qualified(column)}(?![\w\"])", statement, re.IGNORECASE) is not None


def _qualified(column: str) -> str:
    """a column, optionally qualified and quoted, as a pattern."""
    return rf'(?:\w+\.)?"?{re.escape(column)}"?'


def _shape(statement: str, columns: Sequence[str]) -> tuple[tuple[str, ...], str]:
    """the leading index columns the statement binds, and the next one when it orders by it."""
    names = [_column(column) for column in columns]
    bound: list[str] = []
    for name in names:
        if not _binds(statement, name):
            break
        bound.append(name)
    following = names[len(bound)] if len(bound) < len(names) else ""
    return tuple(bound), following if following and _orders(statement, following) else ""


def _also_bound(statement: str, bound: Sequence[str]) -> tuple[str, ...]:
    """the columns other than ``bound`` the statement binds to a parameter by plain equality (``"col" = $n``), in
    the statement's order: the plan test's read sends them too, as the read does."""
    found = re.findall(r'(?<![\w."])(?:\w+\.)?"?(\w+)"?\s*=\s*\$\d+', _where_part(statement))
    return tuple(dict.fromkeys(column.lower() for column in found if column.lower() not in bound))


def _descending(statement: str, column: str) -> bool:
    """whether the statement orders by ``column`` (first) descending."""
    return re.search(rf"ORDER\s+BY\s+{_qualified(column)}\s+DESC\b", statement, re.IGNORECASE) is not None


def _preferred(reader: str) -> tuple[int, str]:
    """a function before a SQL constant (``_NAME``) as a purpose."""
    last = reader.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    return (1 if last.upper() == last else 0, reader)


def _test_name(index: str) -> str:
    return f"test_{re.sub(r'[^0-9a-zA-Z_]', '_', index)}"


def _crowded(table: _DeclaredTable | None) -> bool:
    return table is not None and len(table.indexes) > MAX_PLAIN_INDEXES


def _reason(site: _DeclaredIndex, readers: Sequence[str], bound: Sequence[str], ordered: str, fk: str) -> str:
    """why the index stands beside its table's others (or carries its INCLUDE): the read it serves and its shape."""
    shape = f"by ({', '.join(bound)})" + (f" ordered by {ordered}" if ordered else "")
    served = fk if fk else ", ".join(reader.rsplit("::", 1)[-1] for reader in readers)
    reason = f"the only index serving {served}'s read {shape}"
    if site.include:
        reason += f", returning {', '.join(site.include)} from the index itself"
    return reason if len(reason) >= MIN_REASON else f"{reason} (no other index leads with these columns)"


#: one conjunct of a partial index's ``WHERE`` a seeded row can hold: ``<column> = '<text>'`` or ``= <integer>``
_LITERAL_EQUALITY: Final = re.compile(
    r"""^\(*\s*"?(\w+)"?\s*=\s*(?:'((?:[^']|'')*)'|(-?\d+)|(true|false))\s*\)*$""", re.I
)


def _literals(where: str) -> tuple[tuple[str, Any], ...] | None:
    """a partial index's ``WHERE`` as the column equalities a row holds to be in it; None when it is anything else."""
    if not where.strip():
        return ()
    held: list[tuple[str, Any]] = []
    for conjunct in re.split(r"\bAND\b", where, flags=re.IGNORECASE):
        found = _LITERAL_EQUALITY.match(conjunct.strip())
        if found is None:
            return None
        column, text, number, truth = found.groups()
        value: Any = text.replace("''", "'") if text is not None else int(number) if number is not None else None
        held.append((column.lower(), truth.lower() == "true" if truth is not None else value))
    return tuple(held)


def _foreign_key(table: _DeclaredTable | None, columns: Sequence[str]) -> tuple[str, tuple[str, ...]] | None:
    """the foreign key whose referencing columns the index leads with, as ``(constraint, columns)``."""
    if table is None:
        return None
    names = [_column(column) for column in columns]
    for constraint, referencing in sorted(table.foreign_keys.items()):
        if len(names) >= len(referencing) and set(names[: len(referencing)]) == set(referencing):
            return constraint, tuple(names[: len(referencing)])
    return None


def _constraint_reason(columns: Sequence[str]) -> str:
    """why a unique index stands: the duplicate the database refuses through it."""
    return f"a unique constraint over ({', '.join(columns)}): the database refuses a duplicate through it"


def decide(catalog: SchemaCatalog, module: str = _MODULE) -> Decisions:
    """what to write for every index of the repo the purpose rule faults, from a discovering walk's catalog.

    :param catalog: the catalog after a walk with :attr:`~.schema.SchemaCatalog.discovering` set
    :ptype catalog: SchemaCatalog
    :param module: the repo-relative generated plan-test module the plan tests are named in
    :ptype module: str
    :return: the decisions
    :rtype: Decisions
    """
    decisions = Decisions()
    owned: dict[tuple[str, str], _DeclaredIndex] = {}
    for site in sorted(catalog.sites, key=lambda s: (s.origin, s.name)):
        if site.local and site.table and not site.pattern and (site.table, site.name) not in catalog.dependency_indexes:
            owned.setdefault((site.table, site.name), site)
    for (table_name, name), site in list(owned.items()):
        if "{" in table_name or "{" in name:
            # a table its module names at run time (a template's): its reads are judged against the template
            decisions.hand.append((table_name, name, "named at run time: declared on its template"))
            del owned[(table_name, name)]
    for (table_name, name), site in owned.items():
        table = catalog.tables.get(table_name)
        merged = table.indexes.get(name) if table is not None else None
        purpose = site.purpose or (merged.purpose if merged is not None else "")
        reason = site.reason or (merged.reason if merged is not None else "")
        needs_reason = (bool(site.include) or _crowded(table)) and len(reason) < MIN_REASON
        if purpose and not needs_reason:
            continue
        method = site.method or (merged.method if merged is not None else "")
        if method in _HAND_METHODS:
            decisions.hand.append((table_name, name, f"a {method} index: its read is no equality/order shape"))
            continue
        credited = catalog.matches.get((table_name, name), {})
        where = site.where or (merged.where if merged is not None else "")
        if not purpose and site.unique:
            # a unique index holds a data rule whether or not a read uses it: its purpose is that rule
            literals = _literals(where)
            if literals is None:
                decisions.hand.append((table_name, name, f"a partial unique index on {where!r}: test it by hand"))
                continue
            columns = tuple(_column(column) for column in site.columns)
            decisions.written.append(
                IndexDecision(
                    table_name,
                    name,
                    site.origin,
                    purpose=CONSTRAINT_PURPOSE,
                    plan_test=f"{module}::{_test_name(name)}",
                    reason=_constraint_reason(columns) if needs_reason else "",
                    bound=columns,
                    refuses=True,
                    literals=literals,
                )
            )
            continue
        if purpose.strip() == CONSTRAINT_PURPOSE:
            reason = _constraint_reason(tuple(_column(c) for c in site.columns))
            decisions.written.append(IndexDecision(table_name, name, site.origin, reason=reason))
            continue
        if purpose:
            # inline purpose and plan test: only the reason is missing (a supplement line)
            named = [entry for entry in purposes(purpose) if entry in credited]
            shapes = [_shape(credited[entry], site.columns) for entry in named]
            bound, ordered = max(shapes, key=lambda s: (len(s[0]), bool(s[1])), default=((), ""))
            decisions.written.append(
                IndexDecision(
                    table_name, name, site.origin, reason=_reason(site, named or purposes(purpose), bound, ordered, "")
                )
            )
            continue
        scored = {reader: _shape(statement, site.columns) for reader, statement in credited.items()}
        best = max(((len(bound), bool(ordered)) for bound, ordered in scored.values()), default=(0, False))
        readers = sorted(
            (reader for reader, shape in scored.items() if (len(shape[0]), bool(shape[1])) == best), key=_preferred
        )
        if best[0] > 0 or best[1]:
            chosen = [reader for reader in readers if _preferred(reader)[0] == 0][:_MOST_PURPOSES] or readers[:1]
            bound, ordered = scored[chosen[0]]
            read = credited[chosen[0]]
            decision = IndexDecision(
                table_name,
                name,
                site.origin,
                purpose="; ".join(chosen),
                plan_test=f"{module}::{_test_name(name)}",
                bound=bound,
                ordered=ordered,
                descending=bool(ordered) and _descending(read, ordered),
                also=_also_bound(read, bound),
                where=where,
            )
            if needs_reason:
                decision.reason = _reason(site, chosen, bound, ordered, "")
            decisions.written.append(decision)
            continue
        # a partial index holds only the rows its WHERE admits: the lookup of every referencing row only when it says
        # no more than that they are not NULL
        fk = _foreign_key(table, site.columns)
        if fk is not None and site.partial and not _not_null_of(site.where, fk[1]):
            fk = None
        if fk is not None:
            constraint, columns = fk
            decision = IndexDecision(
                table_name,
                name,
                site.origin,
                purpose=f"fk:{constraint}",
                plan_test=f"{module}::{_test_name(name)}",
                bound=columns,
                where=where,
            )
            if needs_reason:
                decision.reason = _reason(site, (), columns, "", f"foreign key {constraint}")
            decisions.written.append(decision)
            continue
        decisions.drop.append((table_name, name, site.origin))
    return decisions


def _statement(decision: IndexDecision) -> tuple[str, list[dict[str, Any]]]:
    """the read the plan test explains, and its pins (each bound column a parameter, in order): the index's columns
    the read binds after the read's other equalities (seeded coarsest: they narrow no tenant the index tells apart), its
    order as it sends it; the table read as ``t``, as the keyed helpers read it (and name the index to YugabyteDB)."""
    bound = (*decision.also, *decision.bound)
    conditions = [f't."{column}" = ${number}' for number, column in enumerate(bound, start=1)]
    if decision.where:
        conditions.append(f"({decision.where})")
    sql = f'SELECT * FROM "{decision.table}" AS t'
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    if decision.ordered:
        sql += f' ORDER BY t."{decision.ordered}"{" DESC" if decision.descending else ""} LIMIT 10'
    pins = [
        {"column": column, "kind": "param", "param": number, "literal": None}
        for number, column in enumerate(bound, start=1)
    ]
    return sql, pins


def _manifest(decisions: Decisions) -> list[dict[str, Any]]:
    entries = []
    for decision in sorted(decisions.written, key=lambda d: d.index):
        if not decision.plan_test:
            continue
        if decision.refuses:
            pins = [
                {"column": column, "kind": "param", "param": number, "literal": None}
                for number, column in enumerate(decision.bound, start=1)
            ] + [
                {"column": column, "kind": "literal", "param": None, "literal": value}
                for column, value in decision.literals
            ]
            entries.append(
                {
                    "index": decision.index,
                    "table": decision.table,
                    "purpose": decision.purpose,
                    "refuses": True,
                    "pins": pins,
                }
            )
            continue
        sql, pins = _statement(decision)
        entries.append(
            {"index": decision.index, "table": decision.table, "purpose": decision.purpose, "sql": sql, "pins": pins}
        )
    return entries


_HEADER: Final[
    str
] = '''"""GENERATED by ``python -m threetears.enforcement.no_whole_reads.index_purposes``: do not edit by hand.

Each index the tool gave a purpose is proven used by its read's shape, on PostgreSQL and on YugabyteDB:
:func:`threetears.core.testing.rewrites.explain_index_read` seeds the index's table on the migrated test database and
explains the read ``tests/integration/index_plan_manifest.json`` records for it (the columns its purpose's read binds,
in the index's order, then the column it orders by), and the plan must read through the index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from threetears.core.testing.plans import plan_uses_index
from threetears.core.testing.rewrites import explain_index_read, insert_duplicate, load_manifest

__all__: list[str] = []

#: its own mark puts it in the repo's integration suite (it needs a database) and out of the unit run
pytestmark = pytest.mark.integration

_ENTRIES = {entry["index"]: entry for entry in load_manifest(Path(__file__).with_name("index_plan_manifest.json"))}


@pytest.fixture(params=["conn", "yugabyte_conn"])
def engine_conn(request: pytest.FixtureRequest) -> Any:
    """each engine's connection (``conftest.py``): the run's engine is proven, the other deselected."""
    return request.getfixturevalue(request.param)
'''


def _module(entries: list[dict[str, Any]]) -> str:
    parts = [_HEADER]
    for entry in entries:
        index = entry["index"]
        if entry.get("refuses"):
            parts.append(
                f"""

async def {_test_name(index)}(engine_conn: Any) -> None:
    \"\"\"Flips if {entry["table"]} may hold two rows alike on {index}'s columns: the index refuses the second.\"\"\"
    with pytest.raises(asyncpg.UniqueViolationError, match="{index}"):
        await insert_duplicate(engine_conn, _ENTRIES["{index}"])
"""
            )
            continue
        parts.append(
            f"""

async def {_test_name(index)}(engine_conn: Any) -> None:
    \"\"\"Flips if {entry["purpose"]}'s read stops reading {index}.\"\"\"
    plan = await explain_index_read(engine_conn, _ENTRIES["{index}"])
    assert plan_uses_index(plan, "{index}"), json.dumps(plan.plan)
"""
        )
    return "".join(parts)


def _credit_consumer(catalog: SchemaCatalog, consumer: SchemaCatalog, prefixes: Sequence[str]) -> None:
    """credit the repo's indexes with a consumer's reads of them: each read the consumer's discovering walk credited
    to an index of a table the repo declares, written under one of the repo's ``consumer_purposes`` prefixes."""
    for (table, index), credited in consumer.matches.items():
        if table not in catalog.tables or index not in catalog.tables[table].indexes:
            continue
        mine = catalog.matches.setdefault((table, index), {})
        for reader, statement in credited.items():
            if reader.startswith(tuple(prefixes)):
                mine.setdefault(reader, statement)


def _ledger_lines(decisions: Decisions) -> list[str]:
    lines = []
    for decision in sorted(decisions.written, key=lambda d: (d.table, d.index)):
        parts = [decision.index, f"on {decision.table}"]
        if decision.purpose:
            parts += [f"purpose={decision.purpose}", f"plan_test={decision.plan_test}"]
        if decision.reason:
            parts.append(f"reason={decision.reason}")
        lines.append(" :: ".join(parts))
    return lines


def _load_config(repo_root: Path, support: str, name: str) -> NoWholeReadsConfig:
    path = repo_root / support
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, name)
    if not isinstance(config, NoWholeReadsConfig):
        raise SystemExit(f"{support}:{name} is no NoWholeReadsConfig")
    return config


def main(argv: Sequence[str] | None = None) -> int:
    """run the tool.

    :param argv: the command line (``sys.argv[1:]`` when None)
    :ptype argv: Sequence[str] | None
    :return: the exit status
    :rtype: int
    """
    parser = argparse.ArgumentParser(prog="index_purposes")
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("support", help="the repo-relative module holding the gate's NoWholeReadsConfig")
    parser.add_argument("config", help="its name there")
    parser.add_argument("--write", action="store_true", help="write the ledger, manifest and test module")
    parser.add_argument(
        "--module",
        default=_MODULE,
        help="the repo-relative generated plan-test module (its manifest beside it); a repo whose engine fixtures "
        f"(conn, yugabyte_conn) live elsewhere names its own (default {_MODULE})",
    )
    parser.add_argument(
        "--index",
        action="append",
        default=[],
        help="write only this index's declaration and test (repeatable); the other verdicts are printed",
    )
    parser.add_argument("--drop-list", type=Path, help="where to write the drop candidates (default: printed)")
    parser.add_argument(
        "--consumer",
        nargs=3,
        action="append",
        default=[],
        metavar=("ROOT", "SUPPORT", "CONFIG"),
        help="a consumer repo (its config's consumer_purposes prefix) whose reads of this repo's tables count too",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config = _load_config(repo_root, args.support, args.config)
    _, scan = _hits(config, discovering=True)
    for root, support, name in args.consumer:
        _, consumed = _hits(_load_config(Path(root).resolve(), support, name), discovering=True)
        _credit_consumer(scan.schema, consumed.schema, config.consumer_purposes)
    module = args.module
    manifest_path = str(Path(module).with_name(Path(_MANIFEST).name))
    decisions = decide(scan.schema, module)
    if args.index:
        skipped = [d for d in decisions.written if d.index not in args.index]
        decisions.written = [d for d in decisions.written if d.index in args.index]
        sys.stdout.write("".join(f"not written (--index): {d.table}.{d.index}\n" for d in skipped))
    lines = _ledger_lines(decisions)
    entries = _manifest(decisions)
    manifest = repo_root / manifest_path
    if manifest.is_file():
        # a test generated before stays while the ledger still names it as its index's plan test
        made = {entry["index"] for entry in entries}
        ledger_text = (config.ledger_dir / DECLARATIONS_NAME).read_text(encoding="utf-8")
        entries += [
            entry
            for entry in json.loads(manifest.read_text(encoding="utf-8"))
            if entry["index"] not in made and f"plan_test={module}::{_test_name(entry['index'])}" in ledger_text
        ]
        entries.sort(key=lambda entry: entry["index"])
    drops = [f"{table}.{index}  ({origin})" for table, index, origin in sorted(decisions.drop)]
    hand = [f"{table}.{index}: {why}" for table, index, why in sorted(decisions.hand)]
    if args.write:
        ledger = config.ledger_dir / DECLARATIONS_NAME
        kept = ledger.read_text(encoding="utf-8").splitlines() if ledger.is_file() else []
        written = {line.split(" :: ", 1)[0] for line in lines}
        kept = [line for line in kept if line.split(" :: ", 1)[0] not in written]
        ledger.write_text("\n".join([*kept, *lines]) + "\n", encoding="utf-8")
        (repo_root / manifest_path).write_text(json.dumps(entries, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        (repo_root / module).write_text(_module(entries), encoding="utf-8")
        if args.drop_list is not None:
            args.drop_list.write_text("\n".join(drops) + "\n", encoding="utf-8")
    else:
        sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"{len(lines)} declarations, {len(entries)} plan tests, {len(drops)} drop candidates\n")
    if not args.write or args.drop_list is None:
        sys.stdout.write("".join(f"drop candidate: {drop}\n" for drop in drops))
    sys.stdout.write("".join(f"hand: {item}\n" for item in hand))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
