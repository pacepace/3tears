"""what each table's key and indexes are: the schema half of the no-whole-reads rule.

A read is keyed only by the table's own key, or by an index declared for that read. So the rule needs every
table's primary key and its indexes, and it reads them from where a repo declares them, statically, never by
importing:

- 3tears' ``TableSchema(name=..., primary_key=..., indexes=(Index(...), ...))`` and SQLAlchemy's
  ``Table("name", metadata, Column(..., primary_key=True), Index(...))``, in every module under the walked roots;
- SQL DDL written anywhere in them (a migration's ``CREATE TABLE`` / ``CREATE INDEX`` / ``ALTER TABLE ... ADD
  PRIMARY KEY`` / ``DROP INDEX``, in ``.py`` strings and ``.sql`` files), applied in path order so a later
  migration's drop or rename wins;
- a table list in YAML (a tool pod's ``tables.yaml``, an ``agent.yaml`` data section): every mapping with a
  ``name``, ``columns`` and a ``primary_key`` or ``indexes``.

A declared ``UNIQUE`` (a ``UniqueConstraint``, an ``Index(..., unique=True)``, ``UNIQUE (...)`` in a ``CREATE
TABLE``, a ``CREATE UNIQUE INDEX``, ``unique: true`` in a table list) is a key as the primary key is; a partial one
(``WHERE ...``) holds only for some rows and is not. A table declared nowhere has no known key: a read of it fails as
``unknown-schema``, never passes.

**Every declared index names its purpose** (:func:`index_findings`): the one function whose keyed read uses it
(``purpose``, ``<path>::<qualified function>``) and a plan test proving that read uses it (``plan_test``,
``<test file>::<test function>``: the test names the index and calls
:func:`threetears.core.testing.plans.plan_uses_index`). An index with ``INCLUDE`` columns, and every index of a
table with more than three, also carries a written ``reason``: many or wide indexes are how a whole scan hides
behind an index. In SQL the three are a comment just before the ``CREATE INDEX``::

    -- purpose: src/app/store.py::Store.by_owner
    -- plan_test: tests/test_store_plans.py::test_by_owner_uses_its_index
    -- reason: the owner's page lists every column it shows from the index alone
    CREATE INDEX store_by_owner ON store (owner_id) INCLUDE (title)
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from threetears.core.authored_yaml import safe_load_authored
from threetears.enforcement.common import dotted, iter_python_files, parse_python_path, relative_posix_path

__all__ = [
    "COPY_PREFIX",
    "TENANT_COLUMNS",
    "SchemaCatalog",
    "bare_table",
    "expression_key",
    "purposes",
    "collect_schema",
    "index_findings",
]

#: the namespace of a local copy's relations (``snapshot.relation(NAME)``): keyed only by the repo's declared local
#: relations, never by the L3 table that shares the name
COPY_PREFIX: Final[str] = "copy."

#: the columns that name a tenant on every table: a key prefix of only these reads a whole tenant
TENANT_COLUMNS: Final[frozenset[str]] = frozenset({"row_scope", "customer_id"})

#: more indexes than this on one table needs a written reason on each
MAX_PLAIN_INDEXES: Final[int] = 3

#: the purpose of a unique index no read uses: it enforces a constraint, proven by a test that a duplicate is refused
CONSTRAINT_PURPOSE: Final[str] = "constraint"

#: the shortest reason the rule accepts
_MIN_REASON: Final[int] = 30

#: the names a declared index is made with: 3tears' factory and dataclass, SQLAlchemy's ``Index``
_INDEX_CALLS: Final[frozenset[str]] = frozenset({"Index", "IndexDef"})

#: YAML files read as table lists when found under the repo
_TABLE_LIST_FILES: Final[frozenset[str]] = frozenset({"tables.yaml", "tables.yml", "agent.yaml", "agent.yml"})

#: directories never searched for a table list
_SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache", "dist", "build"}
)

_IDENT: Final[str] = r"(?:\"[^\"]+\"|[\w{}]+)"
_QNAME: Final[str] = rf"({_IDENT}(?:\.{_IDENT})*)"
_CREATE_TABLE: Final[re.Pattern[str]] = re.compile(
    rf"\bCREATE\s+(?:UNLOGGED\s+|TEMP\w*\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{_QNAME}\s*\(", re.IGNORECASE
)
_CREATE_INDEX: Final[re.Pattern[str]] = re.compile(
    rf"\bCREATE\s+(UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?(?:{_QNAME}\s+)?ON\s+(?:ONLY\s+)?"
    rf"{_QNAME}\s*(?:USING\s+\w+\s*)?\(",
    re.IGNORECASE,
)
_DROP_INDEX: Final[re.Pattern[str]] = re.compile(
    rf"\bDROP\s+INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+EXISTS\s+)?({_QNAME}(?:\s*,\s*{_QNAME})*)", re.IGNORECASE
)
_DROP_TABLE: Final[re.Pattern[str]] = re.compile(rf"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?{_QNAME}", re.IGNORECASE)
_ADD_PK: Final[re.Pattern[str]] = re.compile(
    rf"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?{_QNAME}\s+ADD\s+(?:CONSTRAINT\s+{_IDENT}\s+)?PRIMARY\s+KEY\s*\(",
    re.IGNORECASE,
)
_RENAME_TABLE: Final[re.Pattern[str]] = re.compile(
    rf"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?{_QNAME}\s+RENAME\s+TO\s+{_QNAME}", re.IGNORECASE
)
_RENAME_COLUMN: Final[re.Pattern[str]] = re.compile(
    rf"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?{_QNAME}\s+RENAME\s+(?:COLUMN\s+)?(?!TO\b)({_IDENT})\s+TO\s+"
    rf"({_IDENT})",
    re.IGNORECASE,
)
_RENAME_INDEX: Final[re.Pattern[str]] = re.compile(
    rf"\bALTER\s+INDEX\s+(?:IF\s+EXISTS\s+)?{_QNAME}\s+RENAME\s+TO\s+{_QNAME}", re.IGNORECASE
)
_PK_ITEM: Final[re.Pattern[str]] = re.compile(r"^(?:CONSTRAINT\s+\S+\s+)?PRIMARY\s+KEY\s*\(", re.IGNORECASE)
_UNIQUE_ITEM: Final[re.Pattern[str]] = re.compile(
    r"^(?:CONSTRAINT\s+\S+\s+)?UNIQUE\s*(?:NULLS\s+(?:NOT\s+)?DISTINCT\s*)?\(", re.IGNORECASE
)
_INLINE_UNIQUE: Final[re.Pattern[str]] = re.compile(r"\bUNIQUE\b", re.IGNORECASE)
_ADD_UNIQUE: Final[re.Pattern[str]] = re.compile(
    rf"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?{_QNAME}\s+ADD\s+(?:CONSTRAINT\s+{_IDENT}\s+)?UNIQUE\s*"
    r"(?:NULLS\s+(?:NOT\s+)?DISTINCT\s*)?\(",
    re.IGNORECASE,
)
_PARTIAL: Final[re.Pattern[str]] = re.compile(r"^\s*(?:WITH\s*\([^)]*\)\s*)?WHERE\b", re.IGNORECASE)
_INLINE_PK: Final[re.Pattern[str]] = re.compile(r"\bPRIMARY\s+KEY\b", re.IGNORECASE)
_NOT_A_COLUMN: Final[re.Pattern[str]] = re.compile(
    r"^(?:CONSTRAINT|UNIQUE|CHECK|FOREIGN|EXCLUDE|LIKE|PRIMARY)\b", re.IGNORECASE
)
_INCLUDE: Final[re.Pattern[str]] = re.compile(r"^\s*INCLUDE\s*\(", re.IGNORECASE)
_DDL_COMMENT: Final[re.Pattern[str]] = re.compile(r"--\s*(purpose|plan_test|reason)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_DDL_HINT: Final[re.Pattern[str]] = re.compile(r"\b(?:CREATE|ALTER|DROP)\b", re.IGNORECASE)


def purposes(purpose: str) -> tuple[str, ...]:
    """the reads an index's ``purpose`` names: ``<path>::<name>`` entries separated by ``; ``.

    :param purpose: the purpose
    :ptype purpose: str
    :return: its entries
    :rtype: tuple[str, ...]
    """
    return tuple(entry.strip() for entry in purpose.split(";") if entry.strip())


def expression_key(text: str) -> str:
    """an indexed expression as the catalog and the judge both key it: each quoted literal ``'_'``, no whitespace,
    lower-cased, outer parentheses gone (``replace(name, '.', '-')`` -> ``replace(name,'_','_')``).

    :param text: the expression
    :ptype text: str
    :return: its key
    :rtype: str
    """
    key = re.sub(r"\s+", "", re.sub(r"'(?:[^']|'')*'", "'_'", text)).lower()
    while key.startswith("(") and key.endswith(")") and key.count("(") == key.count(")"):
        inner = key[1:-1]
        depth = 0
        balanced = True
        for char in inner:
            depth += {"(": 1, ")": -1}.get(char, 0)
            if depth < 0:
                balanced = False
                break
        if not balanced:
            break
        key = inner
    return key


def bare_table(name: str) -> str | None:
    """the table a relation names, as the catalog keys it: its last dotted part, unquoted, lower-cased.

    :param name: the relation as written (``public."Results"``, ``{}.change_log``, ``copy.geos``)
    :ptype name: str
    :return: the bare name (a local copy's keeps its ``copy.``; one numbered at run time is its template,
        ``threetears_change_notes_{}``), or None when the table part is only an interpolation (``{}``, ``"{}"``)
    :rtype: str | None
    """
    parts = re.findall(r"\"[^\"]+\"|[^.\"]+", name.strip())
    last = parts[-1].strip('"').strip() if parts else ""
    copied = len(parts) > 1 and parts[-2].strip('"').strip().lower() == COPY_PREFIX.rstrip(".")
    # a name numbered at run time (``threetears_change_notes_{}``) is its template, as its declaration is
    template = re.sub(r"\{[^{}]*\}", "{}", last).lower()
    literal = template.replace("{}", "").strip("_")
    if copied and not literal:
        # a local copy's relation named at run time: judged against the default key every copy declares
        return f"{COPY_PREFIX}{{}}"
    return None if not last or not literal else f"{COPY_PREFIX if copied else ''}{template}"


@dataclass
class _DeclaredIndex:
    """one index a repo declares.

    :ivar table: the bare table, or ``""`` when the declaration's table cannot be read statically
    :ivar name: the index name (as written, when it is built at run time)
    :ivar columns: its key columns, in order
    :ivar include: its ``INCLUDE`` columns
    :ivar purpose: ``<path>::<qualified function>`` whose keyed read uses it; empty when it names none
    :ivar plan_test: ``<test file>::<test function>`` proving that read uses it
    :ivar reason: why it is wide, or why its table carries more than three
    :ivar origin: ``<path>:<line>`` of the declaration
    :ivar local: whether the repo being judged declares it (not a dependency's source)
    :ivar pattern: for a name built at run time, a regular expression of the names it makes
    :ivar unique: whether it is a unique index, a key of its table
    :ivar partial: whether it covers only the rows a ``WHERE`` names (a partial unique index is no key)
    :ivar where: a partial index's predicate, as written
    """

    table: str
    name: str
    columns: tuple[str, ...]
    include: tuple[str, ...] = ()
    purpose: str = ""
    plan_test: str = ""
    reason: str = ""
    origin: str = ""
    local: bool = True
    pattern: str = ""
    unique: bool = False
    partial: bool = False
    where: str = ""


@dataclass
class _DeclaredTable:
    """one table's key and indexes, merged across its declarations.

    :ivar name: the bare table
    :ivar primary_keys: each declaration's key; a read covering a leading prefix of any is keyed
    :ivar unique_keys: each declared ``UNIQUE``'s columns, keys as the primary key is
    :ivar tenant_columns: its tenant and partition columns beyond :data:`TENANT_COLUMNS` (a ``Column(...,
        partition=True)``, a collection's ``tenancy = tenant_column(...)``): a key prefix of only these is no key
    :ivar indexes: its indexes by name
    :ivar origins: ``<path>:<line>`` of each declaration
    """

    name: str
    primary_keys: list[tuple[str, ...]] = field(default_factory=list)
    unique_keys: list[tuple[str, ...]] = field(default_factory=list)
    tenant_columns: set[str] = field(default_factory=set)
    indexes: dict[str, _DeclaredIndex] = field(default_factory=dict)
    origins: list[str] = field(default_factory=list)


@dataclass
class SchemaCatalog:
    """every table the walk can find a declaration of.

    :ivar tables: bare table name onto its declaration
    :ivar sites: every index declaration the purpose rule judges, one per site
    :ivar used: ``(table, index, purpose entry)`` of each purposed index a keyed read named in its purpose used
    :ivar dependency_indexes: ``(table, index)`` a dependency declares (3tears' own): a copy of one in the repo's
        table list is the dependency's, judged by its gate
    :ivar templates: each module's template tables (a ``TableSchema(name=<parameter>)``), merged: the keys and
        indexes of every table built from them, which the module's generic reads (``FROM {table}``) are judged against
    """

    tables: dict[str, _DeclaredTable] = field(default_factory=dict)
    sites: list[_DeclaredIndex] = field(default_factory=list)
    used: set[tuple[str, str, str]] = field(default_factory=set)
    dependency_indexes: set[tuple[str, str]] = field(default_factory=set)
    templates: dict[str, _DeclaredTable] = field(default_factory=dict)

    def use(self, table: str, index: str, readers: frozenset[str]) -> None:
        """record that a keyed read written in ``readers`` used ``index`` of ``table``, for each of its purpose entries
        naming one of them.

        :param table: the bare table
        :ptype table: str
        :param index: the index
        :ptype index: str
        :param readers: ``<path>::<name>`` of the code the read is written in
        :ptype readers: frozenset[str]
        :return: nothing
        :rtype: None
        """
        entry = self.tables.get(table)
        declared = entry.indexes.get(index) if entry is not None else None
        for purpose in purposes(declared.purpose) if declared is not None else ():
            if purpose in readers:
                self.used.add((table, index, purpose))

    def table(self, name: str) -> _DeclaredTable | None:
        """a table's declaration.

        :param name: the relation as written
        :ptype name: str
        :return: the declaration, or None when none is found (or the table is interpolated)
        :rtype: _DeclaredTable | None
        """
        bare = bare_table(name)
        found = self.tables.get(bare) if bare is not None else None
        if found is None and bare is not None and not bare.startswith(COPY_PREFIX):
            # a view a copy declares (``local_relations``), read by its own name: no L3 table shares it
            found = self.tables.get(f"{COPY_PREFIX}{bare}")
        if found is None and bare is not None and bare.startswith(COPY_PREFIX):
            # a copy the repo declares no key for has the default every copy declares (``local_relations["*"]``)
            found = self.tables.get(f"{COPY_PREFIX}{{}}")
        return found

    def _entry(self, table: str, origin: str) -> _DeclaredTable:
        entry = self.tables.setdefault(table, _DeclaredTable(table))
        if origin not in entry.origins:
            entry.origins.append(origin)
        return entry

    def add_key(self, table: str, key: tuple[str, ...], origin: str) -> None:
        """record a table's primary key.

        :param table: the bare table
        :ptype table: str
        :param key: its key columns
        :ptype key: tuple[str, ...]
        :param origin: where it is declared
        :ptype origin: str
        :return: nothing
        :rtype: None
        """
        entry = self._entry(table, origin)
        key = tuple(column.lower() for column in key)
        if key and key not in entry.primary_keys:
            entry.primary_keys.append(key)

    def add_tenant(self, table: str, column: str, origin: str) -> None:
        """record a table's tenant or partition column: a key prefix of only such columns reads a whole tenant.

        :param table: the bare table
        :ptype table: str
        :param column: the column
        :ptype column: str
        :param origin: where it is declared
        :ptype origin: str
        :return: nothing
        :rtype: None
        """
        self._entry(table, origin).tenant_columns.add(column.lower())

    def add_unique(self, table: str, key: tuple[str, ...], origin: str) -> None:
        """record a table's declared ``UNIQUE`` columns: a key of it.

        :param table: the bare table
        :ptype table: str
        :param key: the columns, in order
        :ptype key: tuple[str, ...]
        :param origin: where it is declared
        :ptype origin: str
        :return: nothing
        :rtype: None
        """
        entry = self._entry(table, origin)
        key = tuple(column.lower() for column in key)
        if key and "?" not in key and key not in entry.unique_keys:
            entry.unique_keys.append(key)

    def add_index(self, index: _DeclaredIndex) -> None:
        """record an index of a table, merging a second declaration of the same name (the purpose of either wins).

        :param index: the index
        :ptype index: _DeclaredIndex
        :return: nothing
        :rtype: None
        """
        entry = self._entry(index.table, index.origin)
        if index.unique and not index.partial:
            self.add_unique(index.table, index.columns, index.origin)
        known = entry.indexes.get(index.name)
        if known is None:
            entry.indexes[index.name] = index
        else:
            known.purpose = known.purpose or index.purpose
            known.plan_test = known.plan_test or index.plan_test
            known.reason = known.reason or index.reason
            known.include = known.include or index.include

    def drop_index(self, name: str) -> None:
        """forget an index a later migration drops.

        :param name: its name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        for entry in self.tables.values():
            entry.indexes.pop(name, None)
        self.sites = [site for site in self.sites if not (site.name == name and not site.pattern and site.table)]

    def rename_index(self, old: str, new: str) -> None:
        """follow an index a later migration renames.

        :param old: its old name
        :ptype old: str
        :param new: its new name
        :ptype new: str
        :return: nothing
        :rtype: None
        """
        for entry in self.tables.values():
            if old in entry.indexes:
                index = entry.indexes.pop(old)
                index.name = new
                declared = entry.indexes.get(new)
                if declared is not None:
                    # renamed onto a name declared elsewhere (the collection's): it is that index, notes and all
                    index.purpose = index.purpose or declared.purpose
                    index.plan_test = index.plan_test or declared.plan_test
                    index.reason = index.reason or declared.reason
                entry.indexes[new] = index

    def rename_column(self, table: str, old: str, new: str) -> None:
        """follow a column a later migration renames, in the table's keys, indexes and tenant columns.

        :param table: the bare table
        :ptype table: str
        :param old: the column's old name
        :ptype old: str
        :param new: its new name
        :ptype new: str
        :return: nothing
        :rtype: None
        """
        entry = self.tables.get(table)
        if entry is None:
            return

        def renamed(columns: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(new if column == old else column for column in columns)

        entry.primary_keys = list(dict.fromkeys(renamed(key) for key in entry.primary_keys))
        entry.unique_keys = list(dict.fromkeys(renamed(key) for key in entry.unique_keys))
        entry.tenant_columns = {new if column == old else column for column in entry.tenant_columns}
        for index in entry.indexes.values():
            index.columns = renamed(index.columns)
            index.include = renamed(index.include)

    def rename_table(self, old: str, new: str) -> None:
        """follow a table a later migration renames.

        :param old: its old bare name
        :ptype old: str
        :param new: its new bare name
        :ptype new: str
        :return: nothing
        :rtype: None
        """
        entry = self.tables.pop(old, None)
        if entry is not None:
            entry.name = new
            for index in entry.indexes.values():
                index.table = new
            merged = self.tables.setdefault(new, _DeclaredTable(new))
            merged.primary_keys.extend(key for key in entry.primary_keys if key not in merged.primary_keys)
            merged.unique_keys.extend(key for key in entry.unique_keys if key not in merged.unique_keys)
            merged.indexes.update(entry.indexes)
            merged.origins.extend(entry.origins)
        for site in self.sites:
            if site.table == old:
                site.table = new


def _split_top(text: str, separator: str = ",") -> list[str]:
    """split ``text`` on ``separator`` outside parentheses and quotes.

    :param text: the text
    :ptype text: str
    :param separator: one character
    :ptype separator: str
    :return: the trimmed parts
    :rtype: list[str]
    """
    parts: list[str] = []
    depth = 0
    quoted = False
    current: list[str] = []
    for char in text:
        if char == "'":
            quoted = not quoted
        elif not quoted and char == "(":
            depth += 1
        elif not quoted and char == ")":
            depth -= 1
        if char == separator and depth == 0 and not quoted:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _paren_body(text: str, opening: int) -> tuple[str, int]:
    """the text inside the parenthesis opening at ``opening``, and the index after its close.

    :param text: the text
    :ptype text: str
    :param opening: index of the ``(``
    :ptype opening: int
    :return: ``(body, end)``
    :rtype: tuple[str, int]
    """
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index], index + 1
    return text[opening + 1 :], len(text)


def collated(column: str, collation: str) -> str:
    """a column as compared in an explicit collation, as the catalog and the judge both name it: ``name`` in
    ``"C"`` -> ``name collate "c"``. An index on the column in one collation serves only comparisons in it.

    :param column: the column, lower-cased
    :ptype column: str
    :param collation: the collation as written (quoted or bare)
    :ptype collation: str
    :return: the collated column's name
    :rtype: str
    """
    return f'{column} collate "{collation.strip(chr(34)).lower()}"'


def _column_of(item: str) -> str:
    """the column a key or index item names (``(a) HASH`` -> ``a``, ``"b" DESC`` -> ``b``, ``lower(c)`` as written).

    :param item: one item of a key or index column list
    :ptype item: str
    :return: the column, lower-cased
    :rtype: str
    """
    stripped = item.strip()
    while stripped.startswith("(") and stripped.endswith(")") and "," not in stripped:
        stripped = stripped[1:-1].strip()
    collation = re.fullmatch(r"(.+?)\s+COLLATE\s+(\"[^\"]+\"|\w+)", stripped, re.IGNORECASE | re.DOTALL)
    if collation is not None:
        # ``(name COLLATE "C")``: the column in that collation, which serves only comparisons made in it
        return collated(_column_of(collation.group(1)), collation.group(2))
    match = re.match(r"^\(?\s*(\"[^\"]+\"|[\w{}]+)\s*\)?(?:\s|$)", stripped)
    if match is None or stripped[len(match.group(1)) :].lstrip().startswith("("):
        # an expression index's item (``(replace(name, '.', '-'))``): keyed by the expression itself
        return expression_key(re.sub(r"\s+(?:ASC|DESC)(?:\s+NULLS\s+(?:FIRST|LAST))?$", "", stripped, flags=re.I))
    return match.group(1).strip('"').lower()


def _ddl_events(text: str) -> Iterator[tuple[str, Any]]:
    """the schema events one SQL text holds, in order.

    :param text: the SQL (comments kept, so a ``-- purpose:`` line before an index is read)
    :ptype text: str
    :return: ``("table", (name, key))``, ``("index", (name, table, columns, include, unique, notes))``,
        ``("drop_index", name)``, ``("drop_table", name)``, ``("rename_table", (old, new))``,
        ``("rename_index", (old, new))``
    :rtype: Iterator[tuple[str, Any]]
    """
    events: list[tuple[int, str, Any]] = []
    for match in _CREATE_TABLE.finditer(text):
        body, _ = _paren_body(text, match.end() - 1)
        key: tuple[str, ...] = ()
        for item in _split_top(_strip_comments(body)):
            if _PK_ITEM.match(item):
                inner, _ = _paren_body(item, item.index("("))
                key = tuple(_column_of(part) for part in _split_top(inner))
            elif _UNIQUE_ITEM.match(item):
                inner, _ = _paren_body(item, item.index("("))
                unique = tuple(_column_of(part) for part in _split_top(inner))
                events.append((match.start(), "unique", (match.group(1), unique)))
            elif _NOT_A_COLUMN.match(item) is None and _INLINE_PK.search(item):
                key = (_column_of(item.split()[0]),)
            elif _NOT_A_COLUMN.match(item) is None and _INLINE_UNIQUE.search(item):
                events.append((match.start(), "unique", (match.group(1), (_column_of(item.split()[0]),))))
        events.append((match.start(), "table", (match.group(1), key)))
    for match in _ADD_UNIQUE.finditer(text):
        inner, _ = _paren_body(text, match.end() - 1)
        events.append((match.start(), "unique", (match.group(1), tuple(_column_of(p) for p in _split_top(inner)))))
    for match in _ADD_PK.finditer(text):
        inner, _ = _paren_body(text, match.end() - 1)
        events.append((match.start(), "table", (match.group(1), tuple(_column_of(p) for p in _split_top(inner)))))
    for match in _CREATE_INDEX.finditer(text):
        inner, after = _paren_body(text, match.end() - 1)
        include: tuple[str, ...] = ()
        rest = text[after:]
        if _INCLUDE.match(rest):
            inc, inc_end = _paren_body(rest, rest.index("("))
            include = tuple(_column_of(part) for part in _split_top(inc))
            rest = rest[inc_end:]
        partial_match = _PARTIAL.match(rest)
        partial = partial_match is not None
        where_text = _strip_comments(rest[partial_match.end() :]).split(";")[0].strip() if partial_match else ""
        notes = _notes_before(text, match.start())
        columns = tuple(_column_of(part) for part in _split_top(_strip_comments(inner)))
        events.append(
            (
                match.start(),
                "index",
                (
                    match.group(2) or "",
                    match.group(3),
                    columns,
                    include,
                    (bool(match.group(1)), partial, where_text),
                    notes,
                ),
            )
        )
    for match in _DROP_INDEX.finditer(text):
        for name in _split_top(match.group(1)):
            events.append((match.start(), "drop_index", name))
    for match in _DROP_TABLE.finditer(text):
        events.append((match.start(), "drop_table", match.group(1)))
    for match in _RENAME_TABLE.finditer(text):
        events.append((match.start(), "rename_table", (match.group(1), match.group(2))))
    for match in _RENAME_COLUMN.finditer(text):
        events.append((match.start(), "rename_column", (match.group(1), match.group(2), match.group(3))))
    for match in _RENAME_INDEX.finditer(text):
        events.append((match.start(), "rename_index", (match.group(1), match.group(2))))
    for _, kind, payload in sorted(events, key=lambda event: event[0]):
        yield kind, payload


def _strip_comments(text: str) -> str:
    return re.sub(r"--[^\n]*", " ", text)


def _notes_before(text: str, start: int) -> dict[str, str]:
    """the ``-- purpose:`` / ``-- plan_test:`` / ``-- reason:`` comment lines just before a statement.

    :param text: the SQL
    :ptype text: str
    :param start: where the statement starts
    :ptype start: int
    :return: the notes by key
    :rtype: dict[str, str]
    """
    notes: dict[str, str] = {}
    for line in reversed(text[:start].splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        match = _DDL_COMMENT.match(stripped)
        if match is None:
            if stripped.startswith("--"):
                continue
            break
        notes.setdefault(match.group(1).lower(), match.group(2))
    return notes


def _apply_ddl(catalog: SchemaCatalog, text: str, origin: str, *, local: bool) -> None:
    """apply one SQL text's DDL to the catalog.

    :param catalog: the catalog
    :ptype catalog: SchemaCatalog
    :param text: the SQL
    :ptype text: str
    :param origin: ``<path>:<line>`` of the text
    :ptype origin: str
    :param local: whether the judged repo holds it
    :ptype local: bool
    :return: nothing
    :rtype: None
    """
    for kind, payload in _ddl_events(text):
        if kind in {"table", "unique"}:
            table = bare_table(payload[0])
            if table is not None and kind == "table":
                catalog.add_key(table, payload[1], origin)
            elif table is not None:
                catalog.add_unique(table, payload[1], origin)
        elif kind == "index":
            name, table_written, columns, include, (unique, partial, where_text), notes = payload
            table = bare_table(table_written)
            index_name = bare_table(name) if name else None
            if table is None or index_name is None:
                continue
            index = _DeclaredIndex(
                table,
                index_name,
                columns,
                include,
                notes.get("purpose", ""),
                notes.get("plan_test", ""),
                notes.get("reason", ""),
                origin,
                local,
                unique=unique,
                partial=partial,
                where=where_text,
            )
            catalog.add_index(index)
            if not local:
                catalog.dependency_indexes.add((table, index_name))
            catalog.sites = [s for s in catalog.sites if not (s.name == index_name and s.table == table)]
            catalog.sites.append(index)
        elif kind == "drop_index":
            name = bare_table(payload)
            if name is not None:
                catalog.drop_index(name)
        elif kind == "drop_table":
            table = bare_table(payload)
            if table is not None:
                catalog.tables.pop(table, None)
                catalog.sites = [site for site in catalog.sites if site.table != table]
        elif kind == "rename_table":
            old, new = bare_table(payload[0]), bare_table(payload[1])
            if old is not None and new is not None:
                catalog.rename_table(old, new)
        elif kind == "rename_column":
            table = bare_table(payload[0])
            if table is not None:
                catalog.rename_column(table, payload[1].strip('"').lower(), payload[2].strip('"').lower())
        elif kind == "rename_index":
            old, new = bare_table(payload[0]), bare_table(payload[1])
            if old is not None and new is not None:
                catalog.rename_index(old, new)
                for site in catalog.sites:
                    if site.name == old:
                        site.name = new


class _Values:
    """the statically known values of a module's names: module and class constants, and what a module imports."""

    def __init__(self, tree: ast.Module) -> None:
        self.names: dict[str, ast.expr] = {}
        #: each module function returning a string, onto that string as a template (``note_table`` ->
        #: ``threetears_change_notes_{}``): a name it builds at run time is that template
        self.templates: dict[str, ast.expr] = {
            statement.name: statement.body[-1].value
            for statement in tree.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.body
            and isinstance(statement.body[-1], ast.Return)
            and isinstance(statement.body[-1].value, (ast.JoinedStr, ast.Constant, ast.BinOp))
        }
        self.index_aliases: set[str] = set(_INDEX_CALLS)
        self.table_schema_aliases: set[str] = {"TableSchema"}
        bodies: list[list[ast.stmt]] = [tree.body]
        while bodies:
            for statement in bodies.pop():
                target: ast.expr | None = None
                value: ast.expr | None = None
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target, value = statement.targets[0], statement.value
                elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                    target, value = statement.target, statement.value
                elif isinstance(statement, ast.ClassDef):
                    bodies.append(statement.body)
                elif isinstance(statement, ast.ImportFrom):
                    for alias in statement.names:
                        if alias.name in _INDEX_CALLS and alias.asname:
                            self.index_aliases.add(alias.asname)
                        if alias.name == "TableSchema" and alias.asname:
                            self.table_schema_aliases.add(alias.asname)
                if isinstance(target, ast.Name) and value is not None:
                    self.names.setdefault(target.id, value)

    def text(self, node: ast.expr | None, depth: int = 0) -> str | None:
        """a string the expression statically is.

        :param node: the expression
        :ptype node: ast.expr | None
        :param depth: how many names were followed
        :ptype depth: int
        :return: the string, or None
        :rtype: str | None
        """
        if node is None or depth > 5:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.names:
            return self.text(self.names[node.id], depth + 1)
        if isinstance(node, ast.Attribute) and node.attr in self.names and dotted(node.value) in {"cls", "self"}:
            return self.text(self.names[node.attr], depth + 1)
        if isinstance(node, ast.JoinedStr):
            return "".join(
                str(part.value)
                if isinstance(part, ast.Constant)
                else (self.text(part.value, depth + 1) if isinstance(part, ast.FormattedValue) else None) or "{}"
                for part in node.values
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.text(node.left, depth + 1), self.text(node.right, depth + 1)
            return None if left is None or right is None else left + right
        if isinstance(node, ast.Call):
            final = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if len(node.args) == 1 and "quote" in (final or "").lower():
                # a quoting helper (``quote_ident(NAME)``) reads as the name it quotes, whatever template it returns
                return self.text(node.args[0], depth + 1)
            if final in self.templates:
                return self.text(self.templates[final], depth + 1)
            # ``quote_identifier(NAME)``: a call on a known string reads as it
            return self.text(node.args[0], depth + 1) if len(node.args) == 1 else None
        return None

    def texts(self, node: ast.expr | None, depth: int = 0) -> tuple[str, ...] | None:
        """the strings a str, tuple or list expression statically is.

        :param node: the expression
        :ptype node: ast.expr | None
        :param depth: how many names were followed
        :ptype depth: int
        :return: the strings, or None when any is unknown
        :rtype: tuple[str, ...] | None
        """
        single = self.text(node, depth)
        if single is not None:
            return (single,)
        if isinstance(node, ast.Name) and node.id in self.names and depth <= 5:
            return self.texts(self.names[node.id], depth + 1)
        if isinstance(node, (ast.Tuple, ast.List)):
            found: list[str] = []
            for element in node.elts:
                inner = self.texts(element.value if isinstance(element, ast.Starred) else element, depth + 1)
                if inner is None:
                    return None
                found.extend(inner)
            return tuple(found)
        return None

    def elements(self, node: ast.expr | None, depth: int = 0) -> list[ast.expr]:
        """the elements of a tuple or list expression, following names.

        :param node: the expression
        :ptype node: ast.expr | None
        :param depth: how many names were followed
        :ptype depth: int
        :return: the elements, empty when unknown
        :rtype: list[ast.expr]
        """
        if isinstance(node, ast.Name) and node.id in self.names and depth <= 5:
            return self.elements(self.names[node.id], depth + 1)
        if isinstance(node, (ast.Tuple, ast.List)):
            return list(node.elts)
        return []


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _name_pattern(node: ast.expr) -> str:
    """a regular expression of the names an f-string index name makes (``f"{self.name}_by_state"``).

    :param node: the name expression
    :ptype node: ast.expr
    :return: the pattern, empty when it is not an f-string
    :rtype: str
    """
    if not isinstance(node, ast.JoinedStr):
        return ""
    return "".join(re.escape(str(part.value)) if isinstance(part, ast.Constant) else r"\w+" for part in node.values)


def _template_names(
    tree: ast.Module, function: ast.FunctionDef | ast.AsyncFunctionDef, parameter: str, values: _Values
) -> list[str]:
    """the concrete tables a template function builds: its parameter's default (``name: str = SCOPE_EPOCHS_TABLE``)
    and each constant the module's calls of it pass.

    :param tree: the module
    :ptype tree: ast.Module
    :param function: the function building the template
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param parameter: the parameter naming the table
    :ptype parameter: str
    :param values: the module's known values
    :ptype values: _Values
    :return: the table names
    :rtype: list[str]
    """
    arguments = [*function.args.posonlyargs, *function.args.args]
    position = next((i for i, arg in enumerate(arguments) if arg.arg == parameter), None)
    found: list[str] = []
    defaults = function.args.defaults
    if position is not None and position >= len(arguments) - len(defaults):
        default = values.text(defaults[position - (len(arguments) - len(defaults))])
        found.extend([default] if default is not None else [])
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function.name:
            passed = _keyword(node, parameter) or (
                node.args[position] if position is not None and position < len(node.args) else None
            )
            text = values.text(passed)
            found.extend([text] if text is not None else [])
    return found


def _declare_template(
    catalog: SchemaCatalog, call: ast.Call, values: _Values, relative: str, *, local: bool
) -> _DeclaredTable:
    """record a template table (``TableSchema(name=name, ...)`` inside a function taking ``name``): its keys and
    indexes hold for every table built from it, so the module's generic reads are judged against them, and a
    default name (``name: str = SCOPE_EPOCHS_TABLE``) is a table of its own.

    :param catalog: the catalog
    :ptype catalog: SchemaCatalog
    :param call: the ``TableSchema(...)`` call
    :ptype call: ast.Call
    :param values: the module's known values
    :ptype values: _Values
    :param relative: the module's repo-relative path
    :ptype relative: str
    :param local: whether the judged repo holds it
    :ptype local: bool
    :return: the module's template, merged
    :rtype: _DeclaredTable
    """
    origin = f"{relative}:{call.lineno}"
    template = catalog.templates.setdefault(relative, _DeclaredTable(f"{{}}@{relative}"))
    key = values.texts(_keyword(call, "primary_key"))
    if key:
        lowered = tuple(column.lower() for column in key)
        if lowered not in template.primary_keys:
            template.primary_keys.append(lowered)
    for element in values.elements(_keyword(call, "indexes")):
        if isinstance(element, ast.Call):
            index = _index_from_call(element, values, "", f"{relative}:{element.lineno}", local=local)
            if index is not None and not index.pattern:
                template.indexes.setdefault(index.name, index)
    template.origins.append(origin)
    return template


def _class_tenancy(catalog: SchemaCatalog, tree: ast.Module, values: _Values, relative: str) -> None:
    """record each collection's declared tenant column (``tenancy = tenant_column("source_customer_id")``) on the
    table the class names (``table_name = ...``, or ``schema = TableSchema(name=...)``).

    :param catalog: the catalog
    :ptype catalog: SchemaCatalog
    :param tree: the module
    :ptype tree: ast.Module
    :param values: the module's known values
    :ptype values: _Values
    :param relative: its repo-relative path
    :ptype relative: str
    :return: nothing
    :rtype: None
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        assigned = {
            statement.targets[0].id: statement.value
            for statement in node.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        }
        tenancy = assigned.get("tenancy")
        if not (
            isinstance(tenancy, ast.Call) and (dotted(tenancy.func) or "").endswith("tenant_column") and tenancy.args
        ):
            continue
        column = values.text(tenancy.args[0])
        schema = assigned.get("schema")
        if isinstance(schema, ast.Name):
            schema = values.names.get(schema.id)
        name = values.text(assigned.get("table_name")) or (
            values.text(_keyword(schema, "name")) if isinstance(schema, ast.Call) else None
        )
        table = bare_table(name) if name is not None else None
        if table is not None and column is not None:
            catalog.add_tenant(table, column, f"{relative}:{node.lineno}")


def _info(call: ast.Call, values: _Values, key: str) -> str:
    """a note a SQLAlchemy ``Index`` carries in its ``info`` (``info={"purpose": ...}``): SQLAlchemy takes no other
    keyword, so that is where the purpose, plan test and reason of an index declared through it go.

    :param call: the ``Index(...)`` call
    :ptype call: ast.Call
    :param values: the module's known values
    :ptype values: _Values
    :param key: ``purpose``, ``plan_test`` or ``reason``
    :ptype key: str
    :return: the note, empty when it has none
    :rtype: str
    """
    info = _keyword(call, "info")
    if isinstance(info, ast.Dict):
        for name, value in zip(info.keys, info.values, strict=True):
            if isinstance(name, ast.Constant) and name.value == key:
                return values.text(value) or ""
    return ""


def _index_from_call(call: ast.Call, values: _Values, table: str, origin: str, *, local: bool) -> _DeclaredIndex | None:
    """the index an ``Index(...)`` / ``IndexDef(...)`` call declares.

    :param call: the call
    :ptype call: ast.Call
    :param values: the module's known values
    :ptype values: _Values
    :param table: the bare table it belongs to, ``""`` when unknown
    :ptype table: str
    :param origin: ``<path>:<line>``
    :ptype origin: str
    :param local: whether the judged repo holds it
    :ptype local: bool
    :return: the index, or None when the call copies another index (``name=index.name``)
    :rtype: _DeclaredIndex | None
    """
    name_node = call.args[0] if call.args else _keyword(call, "name")
    if name_node is None or (isinstance(name_node, ast.Attribute) and name_node.attr == "name"):
        return None
    written = values.text(name_node)
    # a name built at run time (``f"{self.name}_by_state"``) is a pattern of the names it makes
    written = None if written is not None and "{" in written else written
    columns_node = _keyword(call, "columns")
    if columns_node is not None:
        columns = values.texts(columns_node) or ()
    else:
        found: list[str] = []
        for argument in call.args[1:]:
            known = values.texts(argument.value if isinstance(argument, ast.Starred) else argument)
            # a SQLAlchemy column (``table.c.owner``) is its last part; anything else unknown
            spelled = (dotted(argument) or "?").rsplit(".", 1)[-1] if not isinstance(argument, ast.Starred) else "?"
            found.extend(known if known is not None else [spelled])
        columns = tuple(found)
    return _DeclaredIndex(
        table,
        written.lower() if written is not None else ast.unparse(name_node),
        tuple(column.lower() for column in columns),
        tuple(column.lower() for column in (values.texts(_keyword(call, "include")) or ())),
        values.text(_keyword(call, "purpose")) or _info(call, values, "purpose"),
        values.text(_keyword(call, "plan_test")) or _info(call, values, "plan_test"),
        values.text(_keyword(call, "reason")) or _info(call, values, "reason"),
        origin,
        local,
        "" if written is not None else _name_pattern(name_node),
        unique=getattr(_keyword(call, "unique"), "value", False) is True,
        partial=_keyword(call, "where") is not None or _keyword(call, "postgresql_where") is not None,
        where=values.text(_keyword(call, "where")) or "",
    )


def _apply_python(catalog: SchemaCatalog, tree: ast.Module, relative: str, *, local: bool) -> None:
    """apply one module's declarations: its TableSchemas, SQLAlchemy Tables, Index calls and DDL strings.

    :param catalog: the catalog
    :ptype catalog: SchemaCatalog
    :param tree: the module
    :ptype tree: ast.Module
    :param relative: its repo-relative path
    :ptype relative: str
    :param local: whether the judged repo holds it
    :ptype local: bool
    :return: nothing
    :rtype: None
    """
    values = _Values(tree)
    claimed: set[int] = set()
    #: each call's enclosing function's parameters: a declaration named by one is a factory, not a declaration
    params_of: dict[int, set[str]] = {}
    enclosing: dict[int, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {arg.arg for arg in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]}
            for inner in ast.walk(function):
                if isinstance(inner, ast.Call):
                    params_of.setdefault(id(inner), set()).update(names)
                    enclosing[id(inner)] = function
    calls: list[ast.Call] = []
    strings: list[ast.Constant] = []
    #: DDL written as an f-string, its placeholders resolved to the module's constants (``{TURN_DEDUP_ROWS_TABLE}``)
    formatted: list[tuple[ast.JoinedStr, str]] = []
    pieces: set[int] = set()
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            docstrings.add(id(node.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _DDL_HINT.search(node.value):
            strings.append(node)
        elif isinstance(node, ast.JoinedStr):
            pieces.update(id(part) for part in node.values)
            written = values.text(node) or ""
            if _DDL_HINT.search(written):
                formatted.append((node, written))
    for call in calls:
        final = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
        origin = f"{relative}:{call.lineno}"
        if final in values.table_schema_aliases:
            name_node = _keyword(call, "name")
            if isinstance(name_node, ast.Name) and name_node.id in params_of.get(id(call), set()):
                # a template: its keys and indexes are every table built from it (C6)
                template = _declare_template(catalog, call, values, relative, local=local)
                builder = enclosing.get(id(call))
                for concrete in _template_names(tree, builder, name_node.id, values) if builder is not None else []:
                    built = bare_table(concrete)
                    for template_key in template.primary_keys if built is not None else []:
                        catalog.add_key(built or "", template_key, origin)
                claimed.update(id(element) for element in values.elements(_keyword(call, "indexes")))
                continue
            table_text = values.text(name_node)
            table = bare_table(table_text) if table_text is not None else None
            key = values.texts(_keyword(call, "primary_key"))
            if table is not None and key:
                catalog.add_key(table, key, origin)
            for element in values.elements(_keyword(call, "indexes")):
                if isinstance(element, ast.Call):
                    index = _index_from_call(element, values, table or "", f"{relative}:{element.lineno}", local=local)
                    claimed.add(id(element))
                    if index is not None:
                        _add_site(catalog, index)
            for element in values.elements(_keyword(call, "columns")):
                if (
                    table is not None
                    and isinstance(element, ast.Call)
                    and element.args
                    and getattr(_keyword(element, "partition"), "value", False) is True
                ):
                    partition = values.text(element.args[0])
                    if partition is not None:
                        catalog.add_tenant(table, partition, origin)
            for element in values.elements(_keyword(call, "unique_constraints")):
                columns = _unique_columns(element, values, named_first=True)
                if table is not None and columns:
                    catalog.add_unique(table, columns, origin)
        elif final == "Table" and call.args and values.text(call.args[0]) is not None:
            table = bare_table(values.text(call.args[0]) or "")
            if table is None:
                continue
            key = tuple(
                values.text(column.args[0]) or ""
                for column in call.args[1:]
                if isinstance(column, ast.Call)
                and column.args
                and any(k.arg == "primary_key" and getattr(k.value, "value", False) is True for k in column.keywords)
            )
            if key:
                catalog.add_key(table, key, origin)
            for element in call.args[1:]:
                columns = _unique_columns(element, values, named_first=False)
                if columns:
                    catalog.add_unique(table, columns, origin)
                elif (
                    isinstance(element, ast.Call)
                    and element.args
                    and any(k.arg == "unique" and getattr(k.value, "value", False) is True for k in element.keywords)
                    and (getattr(element.func, "attr", None) or getattr(element.func, "id", None)) == "Column"
                ):
                    catalog.add_unique(table, (values.text(element.args[0]) or "?",), origin)
            for element in call.args[1:]:
                if (
                    isinstance(element, ast.Call)
                    and (getattr(element.func, "attr", None) or getattr(element.func, "id", None)) == "Index"
                ):
                    index = _index_from_call(element, values, table, f"{relative}:{element.lineno}", local=local)
                    claimed.add(id(element))
                    if index is not None:
                        _add_site(catalog, index)
    # a SQLAlchemy Table bound to a module name: an Index on ``that.c.column`` belongs to it
    tables_by_name = {
        name: bare_table(values.text(value.args[0]) or "") or ""
        for name, value in values.names.items()
        if isinstance(value, ast.Call)
        and (getattr(value.func, "attr", None) or getattr(value.func, "id", None)) == "Table"
        and value.args
        and values.text(value.args[0]) is not None
    }
    _class_tenancy(catalog, tree, values, relative)
    for call in calls:
        final = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
        name_node = (call.args[0] if call.args else _keyword(call, "name")) if final in values.index_aliases else None
        if isinstance(name_node, ast.Name) and name_node.id in params_of.get(id(call), set()):
            # a factory (``Index(name, ...)`` building whatever its caller names): its call sites declare (C3)
            continue
        if final in values.index_aliases and id(call) not in claimed and (call.args or _keyword(call, "name")):
            owner = (dotted(call.args[1]) or "").split(".") if len(call.args) > 1 else []
            table = tables_by_name.get(owner[0], "") if len(owner) == 3 and owner[1] == "c" else ""
            index = _index_from_call(call, values, table, f"{relative}:{call.lineno}", local=local)
            if index is not None:
                _add_site(catalog, index)
    # a docstring's example DDL declares nothing
    for node in strings:
        if id(node) not in docstrings and id(node) not in pieces:
            _apply_ddl(catalog, str(node.value), f"{relative}:{node.lineno}", local=local)
    for joined, written in formatted:
        _apply_ddl(catalog, written, f"{relative}:{joined.lineno}", local=local)
    # DDL written as a ``.format(...)`` template: its placeholders filled from the module's constants (C4)
    for call in calls:
        layout = (
            values.text(call.func.value)
            if isinstance(call.func, ast.Attribute) and call.func.attr == "format"
            else None
        )
        if layout is not None and _DDL_HINT.search(layout):
            positional = [values.text(argument) or "{}" for argument in call.args]
            named = {keyword.arg: values.text(keyword.value) or "{}" for keyword in call.keywords if keyword.arg}
            try:
                filled = layout.format(*positional, **named)
            except IndexError, KeyError, ValueError:
                filled = ""
            if filled:
                _apply_ddl(catalog, filled, f"{relative}:{call.lineno}", local=local)


def _unique_columns(node: ast.expr, values: _Values, *, named_first: bool) -> tuple[str, ...]:
    """the columns a unique-constraint call declares: 3tears' ``UniqueConstraint(name, *columns)`` in a
    ``TableSchema``'s ``unique_constraints``, SQLAlchemy's ``UniqueConstraint(*columns, name=...)`` in a ``Table``.

    :param node: an expression
    :ptype node: ast.expr
    :param values: the module's known values
    :ptype values: _Values
    :param named_first: whether the call names the constraint first (3tears' factory)
    :ptype named_first: bool
    :return: the columns, empty when it is no such call or a column is unknown
    :rtype: tuple[str, ...]
    """
    if not isinstance(node, ast.Call):
        return ()
    final = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
    if not (final or "").endswith(("UniqueConstraint", "UniqueConstraintDef")):
        return ()
    named = _keyword(node, "columns")
    if named is not None:
        return values.texts(named) or ()
    texts = [values.text(argument) for argument in node.args[1 if named_first else 0 :]]
    return tuple(text for text in texts if text is not None) if texts and None not in texts else ()


def _add_site(catalog: SchemaCatalog, index: _DeclaredIndex) -> None:
    if not index.local and index.table:
        catalog.dependency_indexes.add((index.table, index.name))
    catalog.sites.append(index)
    if index.table and not index.pattern:
        catalog.add_index(index)


def _apply_yaml(catalog: SchemaCatalog, path: Path, relative: str, *, local: bool) -> None:
    """apply a YAML table list: every mapping with ``name``, ``columns`` and a ``primary_key`` or ``indexes``.

    :param catalog: the catalog
    :ptype catalog: SchemaCatalog
    :param path: the file
    :ptype path: Path
    :param relative: its repo-relative path
    :ptype relative: str
    :param local: whether the judged repo holds it
    :ptype local: bool
    :return: nothing
    :rtype: None
    """
    text = path.read_text(encoding="utf-8")
    # an unreadable table list hides every key and index it declares: it raises, naming the file, never skips
    with path.open(encoding="utf-8") as handle:
        document = safe_load_authored(handle)
    lines = text.splitlines()

    def line_of(name: str) -> int:
        needle = re.compile(rf"^\s*-?\s*name:\s*['\"]?{re.escape(name)}['\"]?\s*$")
        return next((number for number, line in enumerate(lines, start=1) if needle.match(line)), 1)

    def walk(node: object) -> None:
        if isinstance(node, dict):
            name = node.get("name")
            if (
                isinstance(name, str)
                and isinstance(node.get("columns"), list)
                and ("primary_key" in node or "indexes" in node)
            ):
                table = bare_table(name)
                if table is not None:
                    key = node.get("primary_key")
                    keys = (key,) if isinstance(key, str) else tuple(str(k) for k in key or ())
                    if keys:
                        catalog.add_key(table, keys, f"{relative}:{line_of(name)}")
                    for index in node.get("indexes") or ():
                        if isinstance(index, dict) and isinstance(index.get("name"), str):
                            declared = _DeclaredIndex(
                                table,
                                index["name"].lower(),
                                tuple(str(c).lower() for c in index.get("columns") or ()),
                                tuple(str(c).lower() for c in index.get("include") or ()),
                                str(index.get("purpose") or ""),
                                str(index.get("plan_test") or ""),
                                str(index.get("reason") or ""),
                                f"{relative}:{line_of(index['name'])}",
                                local,
                                unique=index.get("unique") is True,
                                partial=bool(index.get("where")),
                            )
                            _add_site(catalog, declared)
                    for constraint in node.get("unique_constraints") or ():
                        if isinstance(constraint, dict):
                            columns = tuple(str(c) for c in constraint.get("columns") or ())
                            catalog.add_unique(table, columns, f"{relative}:{line_of(name)}")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)


def _table_lists(root: Path) -> Iterator[Path]:
    """every YAML table list under ``root``, skipping virtualenvs and build output.

    :param root: the directory
    :ptype root: Path
    :return: the files
    :rtype: Iterator[Path]
    """
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name not in _SKIP_DIRS and not entry.name.startswith("."):
            yield from _table_lists(entry)
        elif entry.is_file() and entry.name in _TABLE_LIST_FILES:
            yield entry


def _sql_files(root: Path) -> Iterator[Path]:
    """every ``.sql`` file under ``root``.

    :param root: the directory
    :ptype root: Path
    :return: the files
    :rtype: Iterator[Path]
    """
    yield from (path for path in sorted(root.rglob("*.sql")) if not set(path.parts) & _SKIP_DIRS)


def collect_schema(
    repo_root: Path,
    roots: tuple[Path, ...],
    *,
    dependency_roots: tuple[Path, ...] = (),
    excluded_prefixes: tuple[str, ...] = (),
    table_lists: tuple[Path, ...] | None = None,
) -> SchemaCatalog:
    """every table key and index declared under the repo's roots (and its dependencies' sources).

    :param repo_root: the repo root
    :ptype repo_root: Path
    :param roots: the repo's source trees (migrations included)
    :ptype roots: tuple[Path, ...]
    :param dependency_roots: source trees of what the repo depends on (3tears' own tables a consumer reads); their
        declarations key reads, but their indexes are not the repo's to give a purpose
    :ptype dependency_roots: tuple[Path, ...]
    :param excluded_prefixes: repo-relative prefixes whose modules declare nothing real (test doubles, the gates)
    :ptype excluded_prefixes: tuple[str, ...]
    :param table_lists: YAML table lists to read; None finds every ``tables.yaml`` / ``agent.yaml`` in the repo
    :ptype table_lists: tuple[Path, ...] | None
    :return: the catalog
    :rtype: SchemaCatalog
    """
    catalog = SchemaCatalog()
    seen: set[Path] = set()
    entries: list[tuple[str, Path, bool]] = []
    for local, group in ((False, dependency_roots), (True, roots)):
        for root in group:
            for path in [*iter_python_files(root), *_sql_files(root)]:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                inside = resolved.is_relative_to(repo_root.resolve())
                relative = relative_posix_path(path, repo_root) if inside else path.as_posix()
                if any(relative.startswith(prefix) for prefix in excluded_prefixes):
                    continue
                entries.append((relative, path, local and inside))
    # a dependency's tables first, then the repo's, each in path order: a later migration's drop or rename wins
    entries.sort(key=lambda entry: (entry[2], entry[0]))
    for relative, path, local in entries:
        if path.suffix == ".sql":
            _apply_ddl(catalog, path.read_text(encoding="utf-8"), f"{relative}:1", local=local)
        else:
            _apply_python(catalog, parse_python_path(path), relative, local=local)
    for path in table_lists if table_lists is not None else tuple(_table_lists(repo_root)):
        _apply_yaml(catalog, path, relative_posix_path(path, repo_root), local=True)
    _resolve_patterns(catalog)
    return catalog


def _resolve_patterns(catalog: SchemaCatalog) -> None:
    """give each index declared under a name built at run time (``f"{self.name}_by_state"``) the purpose, plan test
    and reason its declaration carries, on every table index the name makes; the table list's copies of it are then
    no sites of their own.

    :param catalog: the catalog
    :ptype catalog: SchemaCatalog
    :return: nothing
    :rtype: None
    """
    patterns = [site for site in catalog.sites if site.pattern]
    covered: set[int] = set()
    for site in catalog.sites:
        if site.pattern or not site.table:
            continue
        for pattern in patterns:
            if re.fullmatch(pattern.pattern, site.name) and pattern.local == site.local:
                covered.add(id(site))
                table = catalog.tables.get(site.table)
                index = table.indexes.get(site.name) if table is not None else None
                if index is not None:
                    index.purpose = index.purpose or pattern.purpose
                    index.plan_test = index.plan_test or pattern.plan_test
                    index.reason = index.reason or pattern.reason
    catalog.sites = [site for site in catalog.sites if id(site) not in covered]


def _test_defect(repo_root: Path, test: str, *, names: tuple[str, ...], calls: str) -> str | None:
    """what is wrong with a plan test, if anything.

    :param repo_root: the repo root
    :ptype repo_root: Path
    :param test: ``<test file>::<test function>``
    :ptype test: str
    :param names: what the test must name (the index)
    :ptype names: tuple[str, ...]
    :param calls: what it must call
    :ptype calls: str
    :return: the defect, or None
    :rtype: str | None
    """
    module, _, name = test.partition("::")
    path = repo_root / module
    if not name or not path.is_file():
        return f"names no test file and function ({test!r}); give <tests/...py>::<test_function>"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tests = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if not tests:
        return f"{module} defines no {name}"
    # the body only: a test named for the index does not name it by its name alone
    body = "\n".join(ast.unparse(statement) for test in tests for statement in test.body)
    missing = [needed for needed in names if needed not in body]
    if missing:
        return f"{module}::{name} never names {missing[0]}: it cannot prove that index is used"
    if not any(
        isinstance(node, ast.Call) and (dotted(node.func) or "").rsplit(".", 1)[-1] == calls
        for test in tests
        for node in ast.walk(test)
    ):
        return f"{module}::{name} never calls {calls}: the database, not the test's word, shows the index does its work"
    return None


def _names_a_match(repo_root: Path, test: str, pattern: str) -> bool:
    """whether a plan test's body names a concrete index a pattern-named declaration makes
    (``f"{self.name}_by_state"`` -> ``latest_results_by_state``).

    :param repo_root: the repo root
    :ptype repo_root: Path
    :param test: ``<test file>::<test function>``
    :ptype test: str
    :param pattern: the regular expression of the names the declaration makes
    :ptype pattern: str
    :return: whether it does
    :rtype: bool
    """
    module, _, name = test.partition("::")
    tree = ast.parse((repo_root / module).read_text(encoding="utf-8"))
    body = "\n".join(
        ast.unparse(statement)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for statement in node.body
    )
    return re.search(rf"\b{pattern}\b", body) is not None


def index_findings(catalog: SchemaCatalog, repo_root: Path, definitions: set[str]) -> list[str]:
    """every declared index of the repo that names no purpose, no plan test, no reason it needs, or a purpose no
    keyed read uses.

    :param catalog: the catalog, after the walk judged every read (so :attr:`SchemaCatalog.used` is filled)
    :ptype catalog: SchemaCatalog
    :param repo_root: the repo root
    :ptype repo_root: Path
    :param definitions: ``<path>::<qualified name>`` of every function, class and SQL constant the walk found
    :ptype definitions: set[str]
    :return: findings
    :rtype: list[str]
    """
    findings: list[str] = []
    # an index a dependency declares too (3tears' own, rendered into the repo's table list) is the dependency's; one
    # (table, index) declared twice (a TableSchema and its migration) is one index, with the notes either carries
    owned: dict[tuple[str, str] | int, _DeclaredIndex] = {}
    for site in catalog.sites:
        if site.local and (site.table, site.name) not in catalog.dependency_indexes:
            owned.setdefault((site.table, site.name) if site.table and not site.pattern else id(site), site)
    for site in sorted(owned.values(), key=lambda s: (s.origin, s.name)):
        table = catalog.tables.get(site.table) if site.table else None
        merged = table.indexes.get(site.name) if table is not None else None
        purpose = site.purpose or (merged.purpose if merged is not None else "")
        plan_test = site.plan_test or (merged.plan_test if merged is not None else "")
        reason = site.reason or (merged.reason if merged is not None else "")
        findings.extend(_index_site_findings(catalog, repo_root, definitions, site, (purpose, plan_test, reason)))
    return findings


def _index_site_findings(
    catalog: SchemaCatalog,
    repo_root: Path,
    definitions: set[str],
    site: _DeclaredIndex,
    notes: tuple[str, str, str],
) -> list[str]:
    """what is wrong with one declared index.

    :param catalog: the catalog, after the walk
    :ptype catalog: SchemaCatalog
    :param repo_root: the repo root
    :ptype repo_root: Path
    :param definitions: ``<path>::<qualified name>`` of every function, class and SQL constant the walk found
    :ptype definitions: set[str]
    :param site: the declaration
    :ptype site: _DeclaredIndex
    :param notes: its purpose, plan test and reason, from any declaration of it
    :ptype notes: tuple[str, str, str]
    :return: findings
    :rtype: list[str]
    """
    purpose, plan_test, reason = notes
    findings: list[str] = []
    where = f"{site.origin} index {site.name}" + (f" on {site.table}" if site.table else "")
    table = catalog.tables.get(site.table) if site.table else None
    crowded = table is not None and len(table.indexes) > MAX_PLAIN_INDEXES
    if not purpose:
        findings.append(
            f"{where}: index-no-purpose -- names no purpose: give purpose='<path>::<function>' (the keyed read that "
            "uses it) and plan_test='<test file>::<test>'; an index no read can name is a candidate for removal"
        )
    constraint = purpose.strip() == CONSTRAINT_PURPOSE
    if constraint and not site.unique:
        findings.append(
            f"{where}: index-constraint-not-unique -- only a unique index may serve a constraint and no read; give "
            "the read that uses it"
        )
    for entry in () if constraint else purposes(purpose):
        if "::" not in entry or entry not in definitions:
            findings.append(
                f"{where}: index-purpose-unknown -- purpose {entry!r} names no function the walk found; give "
                "<path>::<qualified function>"
            )
        elif site.table and not site.pattern and (site.table, site.name, entry) not in catalog.used:
            findings.append(
                f"{where}: index-purpose-unused -- {entry} holds no read of {site.table} keyed by its leading "
                f"column {site.columns[0] if site.columns else '?'}: each purpose names a read that uses it"
            )
    if purpose and not plan_test:
        findings.append(f"{where}: index-no-plan-test -- names no plan_test proving its read uses it")
    elif plan_test:
        # a constraint's test proves the database refuses a duplicate; a read's, that its plan uses the index
        calls = "raises" if constraint else "plan_uses_index"
        defect = _test_defect(repo_root, plan_test, names=() if site.pattern else (site.name,), calls=calls)
        if defect is None and site.pattern and not _names_a_match(repo_root, plan_test, site.pattern):
            defect = f"{plan_test} never names an index {site.name} makes: it cannot prove one is used"
        if defect is not None:
            findings.append(f"{where}: index-plan-test -- the plan test {defect}")
    if site.include and len(reason) < _MIN_REASON:
        findings.append(
            f"{where}: index-wide-no-reason -- carries INCLUDE {list(site.include)} with no written reason "
            f"({_MIN_REASON}+ characters): a wide index is how a whole scan hides behind one"
        )
    if crowded and len(reason) < _MIN_REASON and table is not None:
        findings.append(
            f"{where}: index-crowded-no-reason -- {site.table} carries {len(table.indexes)} indexes, more than "
            f"{MAX_PLAIN_INDEXES}, and this one gives no written reason ({_MIN_REASON}+ characters)"
        )
    return findings
