"""the index declarations ledger: an index's purpose, plan test and reason, for an index whose DDL sits in a frozen
migration.

A shipped migration is checksum-tracked: editing its text to add the ``-- purpose:`` / ``-- plan_test:`` /
``-- reason:`` comments an index needs (:mod:`~threetears.enforcement.no_whole_reads.schema`) would change a migration
every deployed database has already run. Such an index is declared in the repo's ``_index_declarations.txt``
instead (beside its other ledgers, :attr:`NoWholeReadsConfig.ledger_dir`), one line each::

    <index> :: on <table> :: purpose=<path>::<function>[; <path>::<function>] :: plan_test=<file>::<test> [:: reason=<why>]

A line is a declaration, not an exception: it is applied to the index exactly as its inline comments would be, so the
same checks judge it (each purpose names a function whose keyed read uses the index, the plan test exists and names
the index, a crowded or wide index gives its reason). Three findings are the ledger's own:

- ``index-ledger-not-frozen``: the index's DDL is in no frozen migration (``NoWholeReadsConfig.frozen_migrations``):
  it is declared inline, where its DDL is;
- ``index-declared-twice``: the index carries inline notes too; one declaration each;
- ``index-ledger-stale``: no migration the walk read creates the index (dropped, renamed, never there), or it is on
  another table than the line says.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from threetears.enforcement.no_whole_reads.schema import SchemaCatalog

__all__ = ["DECLARATIONS_NAME", "apply_index_declarations"]

#: the ledger's file name, beside the repo's other no-whole-reads ledgers
DECLARATIONS_NAME: Final[str] = "_index_declarations.txt"

_SEPARATOR: Final[str] = " :: "


@dataclass(frozen=True)
class _Declaration:
    """one ledger line: what an index in a frozen migration is for.

    :ivar index: the index name, as its DDL creates it (lower case)
    :ivar table: the bare table it is on (lower case)
    :ivar purpose: ``<path>::<function>`` entries, ``; ``-separated, as an inline ``-- purpose:`` names them
    :ivar plan_test: ``<test file>::<test>`` proving the purpose's read uses it
    :ivar reason: why it is wide or its table crowded; empty when it gives none
    :ivar line: the line in the ledger
    """

    index: str
    table: str
    purpose: str
    plan_test: str
    reason: str = ""
    line: int = 0


def _parse(path: Path) -> tuple[dict[str, _Declaration], list[str]]:
    """read the ledger: its declarations by index, and each malformed or repeated line.

    :param path: the ledger
    :ptype path: Path
    :return: the declarations, and the defects
    :rtype: tuple[dict[str, _Declaration], list[str]]
    """
    declarations: dict[str, _Declaration] = {}
    defects: list[str] = []
    if not path.is_file():
        return declarations, defects
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        where = f"{DECLARATIONS_NAME}:{number}"
        parts = [part.strip() for part in text.split(_SEPARATOR)]
        fields: dict[str, str] = {}
        for part in parts[2:]:
            name, equals, value = part.partition("=")
            if not equals or name.strip() not in {"purpose", "plan_test", "reason"} or name.strip() in fields:
                fields = {}
                break
            fields[name.strip()] = value.strip()
        if (
            len(parts) < 4
            or not parts[0]
            or not parts[1].startswith("on ")
            or not parts[1][3:].strip()
            or not fields.get("purpose")
            or not fields.get("plan_test")
        ):
            defects.append(
                f"{where}: index-ledger-malformed -- write <index> :: on <table> :: purpose=<path>::<function> :: "
                "plan_test=<file>::<test> [:: reason=<why>]"
            )
            continue
        index = parts[0].strip('"').lower()
        if index in declarations:
            defects.append(
                f"{where}: index-declared-twice -- {index} is declared on line {declarations[index].line} too"
            )
            continue
        declarations[index] = _Declaration(
            index=index,
            table=parts[1][3:].strip().strip('"').lower(),
            purpose=fields["purpose"],
            plan_test=fields["plan_test"],
            reason=fields.get("reason", ""),
            line=number,
        )
    return declarations, defects


def _frozen(origin: str, frozen: Sequence[str]) -> bool:
    """whether a declaration's ``<path>:<line>`` lies in a frozen migration."""
    path = origin.rsplit(":", 1)[0]
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in frozen)


def apply_index_declarations(catalog: SchemaCatalog, path: Path, frozen: Sequence[str]) -> list[str]:
    """give each index the ledger declares the purpose, plan test and reason its line carries, as its inline comments
    would; what is wrong with a line is a finding.

    Applied before the walk, so a keyed read the purpose names is credited as it is for an inline purpose.

    :param catalog: the catalog, as collected (every migration applied)
    :ptype catalog: SchemaCatalog
    :param path: the ledger
    :ptype path: Path
    :param frozen: repo-relative globs of the frozen (shipped, checksum-tracked) migrations
    :ptype frozen: Sequence[str]
    :return: the ledger's findings
    :rtype: list[str]
    """
    declarations, findings = _parse(path)
    for declaration in declarations.values():
        where = f"{DECLARATIONS_NAME}:{declaration.line} index {declaration.index}"
        sites = [site for site in catalog.sites if site.local and not site.pattern and site.name == declaration.index]
        tables = {site.table for site in sites}
        if not sites:
            findings.append(
                f"{where}: index-ledger-stale -- no migration the walk read creates it (dropped, renamed, or never "
                "there): delete the line"
            )
            continue
        if declaration.table not in tables:
            findings.append(
                f"{where}: index-ledger-stale -- it is on {', '.join(sorted(tables))}, not {declaration.table}: correct "
                "the line"
            )
            continue
        sites = [site for site in sites if site.table == declaration.table]
        declared = catalog.tables.get(declaration.table)
        merged = declared.indexes.get(declaration.index) if declared is not None else None
        if not any(_frozen(site.origin, frozen) for site in sites):
            findings.append(
                f"{where}: index-ledger-not-frozen -- its DDL ({sites[0].origin}) is in no frozen migration: declare "
                "it inline, where its DDL is (-- purpose: / -- plan_test: / -- reason:)"
            )
            continue
        inline = [*sites, *([merged] if merged is not None else [])]
        if any(index.purpose or index.plan_test or index.reason for index in inline):
            findings.append(
                f"{where}: index-declared-twice -- it carries inline notes too: keep one declaration, the inline one "
                "where its DDL is not frozen, else this line"
            )
            continue
        for index in inline:
            index.purpose = declaration.purpose
            index.plan_test = declaration.plan_test
            index.reason = declaration.reason
    return findings
