"""the index declarations ledger: an index's purpose, plan test and reason, for an index whose DDL sits in a frozen
migration.

A shipped migration is checksum-tracked: editing its text to add the ``-- purpose:`` / ``-- plan_test:`` /
``-- reason:`` comments an index needs (:mod:`~threetears.enforcement.no_whole_reads.schema`) would change a migration
every deployed database has already run. Such an index is declared in the repo's ``_index_declarations.txt``
instead (beside its other ledgers, :attr:`NoWholeReadsConfig.ledger_dir`), one line each::

    <index> :: on <table> :: purpose=<path>::<function>[; <path>::<function>] :: plan_test=<file>::<test> [:: reason=<why>]

A line is a declaration, not an exception: it is applied to the index exactly as its inline comments would be, so the
same checks judge it (each purpose names a function whose keyed read uses the index, the plan test exists and names
the index, a crowded or wide index gives its reason).

A frozen index whose purpose and plan test sit beside its DDL, but which needs a reason its migration cannot be edited
to give (its table grew crowded later), takes that reason from a supplement line, the reason and nothing else::

    <index> :: on <table> :: reason=<why, 30+ characters>

A consumer reading a dependency's table through the dependency's index adds its own reads to that index's purpose
with a purpose-only line (coordinator ruling 2026-11-13, for the hub)::

    <index> :: on <table> :: purpose=<path>::<function>[; <path>::<function>]

It re-declares nothing (the dependency keeps the index's DDL, purpose and plan test), so it is never
``index-declared-twice``; each read it adds is held to a function the walk found whose read uses the index
(``index-purpose-unknown`` / ``index-purpose-unused``). A purpose-only line for an index the repo declares itself is
``index-addition-refused``: declare it whole.

The ledger's own findings:

- ``index-ledger-not-frozen``: the index's DDL is in no frozen migration (``NoWholeReadsConfig.frozen_migrations``):
  it is declared inline, where its DDL is;
- ``index-declared-twice``: the index carries inline notes too; one declaration each;
- ``index-ledger-stale``: no migration the walk read creates the index (dropped, renamed, never there), or it is on
  another table than the line says;
- ``index-supplement-refused``: a reason-only line whose reason is under 30 characters, or whose index has no inline
  purpose and plan test to supplement (declare it whole on its line). A supplement carrying ``purpose=`` or
  ``plan_test=`` is malformed; one for an index already giving an inline reason is ``index-declared-twice``.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from threetears.enforcement.no_whole_reads.schema import MIN_REASON, SchemaCatalog, purposes

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
    :ivar plan_test: ``<test file>::<test>`` proving the purpose's read uses it; with ``purpose``, empty on a
        supplement (a reason-only line for an index declared inline)
    :ivar reason: why it is wide or its table crowded; empty when it gives none
    :ivar line: the line in the ledger
    """

    index: str
    table: str
    purpose: str
    plan_test: str
    reason: str = ""
    line: int = 0

    @property
    def supplement(self) -> bool:
        """whether the line only adds a reason to an index whose purpose and plan test are inline."""
        return not self.purpose and not self.plan_test

    @property
    def addition(self) -> bool:
        """whether the line only adds the repo's own reads to the purpose of an index a dependency declares."""
        return bool(self.purpose) and not self.plan_test and not self.reason


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
        whole = bool(fields.get("purpose")) and bool(fields.get("plan_test"))
        supplement = set(fields) == {"reason"} and bool(fields["reason"])
        addition = set(fields) == {"purpose"} and bool(fields["purpose"])
        if (
            len(parts) < 3
            or not parts[0]
            or not parts[1].startswith("on ")
            or not parts[1][3:].strip()
            or not (whole or supplement or addition)
        ):
            defects.append(
                f"{where}: index-ledger-malformed -- write <index> :: on <table> :: purpose=<path>::<function> :: "
                "plan_test=<file>::<test> [:: reason=<why>], or, for an index declared inline, the reason alone: "
                "<index> :: on <table> :: reason=<why>"
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
            purpose=fields.get("purpose", ""),
            plan_test=fields.get("plan_test", ""),
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
        if declaration.addition:
            refusal = _addition_refusal(catalog, declaration, where)
            if refusal is not None:
                findings.append(refusal)
            continue
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
        if declaration.supplement:
            refusal = _supplement_refusal(
                where,
                declaration.reason,
                declared=any(index.purpose and index.plan_test for index in inline),
                reasoned=any(index.reason for index in inline),
            )
            if refusal is not None:
                findings.append(refusal)
                continue
            for index in inline:
                index.reason = declaration.reason
            continue
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


def _addition_refusal(catalog: SchemaCatalog, declaration: _Declaration, where: str) -> str | None:
    """add a purpose-only line's reads to the purpose of the dependency's index it names, or say why it is refused.

    A consumer reads a dependency's table through the dependency's index; the dependency declares the index (its DDL,
    purpose and plan test) and cannot name every consumer's read. The consumer's line adds its own reads to that
    index's purpose, so the consumer's gate credits them as keyed by it (each still held to a read that uses it,
    :attr:`SchemaCatalog.added_purposes`). It re-declares nothing, so it is never ``index-declared-twice``; an index
    the repo itself declares is declared whole instead.

    :param catalog: the catalog
    :ptype catalog: SchemaCatalog
    :param declaration: the purpose-only line
    :ptype declaration: _Declaration
    :param where: the line, as a finding names it
    :ptype where: str
    :return: the refusal, or None when the reads were added
    :rtype: str | None
    """
    sites = [
        site
        for site in catalog.sites
        if not site.pattern and site.name == declaration.index and site.table == declaration.table
    ]
    if not sites:
        return (
            f"{where}: index-ledger-stale -- no declaration the walk read creates it on {declaration.table}: delete "
            "the line"
        )
    if any(site.local for site in sites):
        return (
            f"{where}: index-addition-refused -- {declaration.index} is the repo's own: a purpose-only line adds a "
            "consumer's reads to a dependency's index; declare this one whole (purpose= :: plan_test=)"
        )
    declared = catalog.tables.get(declaration.table)
    merged = declared.indexes.get(declaration.index) if declared is not None else None
    if merged is None:
        return f"{where}: index-ledger-stale -- {declaration.table} carries no index {declaration.index}"
    added = purposes(declaration.purpose)
    merged.purpose = "; ".join(dict.fromkeys([*purposes(merged.purpose), *added]))
    catalog.added_purposes.extend((where, declaration.table, declaration.index, entry) for entry in added)
    return None


def _supplement_refusal(where: str, reason: str, *, declared: bool, reasoned: bool) -> str | None:
    """what refuses a reason-only line for a frozen index, or None when its reason applies.

    :param where: the line, for the finding
    :ptype where: str
    :param reason: the line's reason
    :ptype reason: str
    :param declared: whether the index names its purpose and plan test beside its DDL
    :ptype declared: bool
    :param reasoned: whether it gives a reason beside its DDL already
    :ptype reasoned: bool
    :return: the finding, or None
    :rtype: str | None
    """
    if reasoned:
        return (
            f"{where}: index-declared-twice -- it gives a reason beside its DDL already: keep that one, delete the line"
        )
    if not declared:
        return (
            f"{where}: index-supplement-refused -- a reason-only line supplements an index whose purpose and plan "
            "test sit beside its DDL, and this one has none: declare it whole on the line (purpose= :: plan_test= "
            ":: reason=)"
        )
    if len(reason) < MIN_REASON:
        return (
            f"{where}: index-supplement-refused -- its reason is {len(reason)} characters: say why the index is wide "
            f"or its table crowded in {MIN_REASON}+"
        )
    return None
