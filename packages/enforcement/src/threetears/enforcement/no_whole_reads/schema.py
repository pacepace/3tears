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

A table declared nowhere has no known key: a read of it fails as ``unknown-schema``, never passes.

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
    "SchemaCatalog",
    "bare_table",
    "collect_schema",
    "index_findings",
]

#: more indexes than this on one table needs a written reason on each
MAX_PLAIN_INDEXES: Final[int] = 3

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
_RENAME_INDEX: Final[re.Pattern[str]] = re.compile(
    rf"\bALTER\s+INDEX\s+(?:IF\s+EXISTS\s+)?{_QNAME}\s+RENAME\s+TO\s+{_QNAME}", re.IGNORECASE
)
_PK_ITEM: Final[re.Pattern[str]] = re.compile(r"^(?:CONSTRAINT\s+\S+\s+)?PRIMARY\s+KEY\s*\(", re.IGNORECASE)
_INLINE_PK: Final[re.Pattern[str]] = re.compile(r"\bPRIMARY\s+KEY\b", re.IGNORECASE)
_NOT_A_COLUMN: Final[re.Pattern[str]] = re.compile(
    r"^(?:CONSTRAINT|UNIQUE|CHECK|FOREIGN|EXCLUDE|LIKE|PRIMARY)\b", re.IGNORECASE
)
_INCLUDE: Final[re.Pattern[str]] = re.compile(r"^\s*INCLUDE\s*\(", re.IGNORECASE)
_DDL_COMMENT: Final[re.Pattern[str]] = re.compile(r"--\s*(purpose|plan_test|reason)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_DDL_HINT: Final[re.Pattern[str]] = re.compile(r"\b(?:CREATE|ALTER|DROP)\b", re.IGNORECASE)


def bare_table(name: str) -> str | None:
    """the table a relation names, as the catalog keys it: its last dotted part, unquoted, lower-cased.

    :param name: the relation as written (``public."Results"``, ``{}.change_log``)
    :ptype name: str
    :return: the bare name, or None when the table part is interpolated (``{}``, ``"{}"``)
    :rtype: str | None
    """
    parts = re.findall(r"\"[^\"]+\"|[^.\"]+", name.strip())
    last = parts[-1].strip('"').strip() if parts else ""
    return None if not last or "{" in last else last.lower()


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


@dataclass
class _DeclaredTable:
    """one table's key and indexes, merged across its declarations.

    :ivar name: the bare table
    :ivar primary_keys: each declaration's key; a read covering a leading prefix of any is keyed
    :ivar indexes: its indexes by name
    :ivar origins: ``<path>:<line>`` of each declaration
    """

    name: str
    primary_keys: list[tuple[str, ...]] = field(default_factory=list)
    indexes: dict[str, _DeclaredIndex] = field(default_factory=dict)
    origins: list[str] = field(default_factory=list)


@dataclass
class SchemaCatalog:
    """every table the walk can find a declaration of.

    :ivar tables: bare table name onto its declaration
    :ivar sites: every index declaration the purpose rule judges, one per site
    :ivar used: ``(table, index)`` of each purposed index a keyed read of its purpose function used
    """

    tables: dict[str, _DeclaredTable] = field(default_factory=dict)
    sites: list[_DeclaredIndex] = field(default_factory=list)
    used: set[tuple[str, str]] = field(default_factory=set)

    def table(self, name: str) -> _DeclaredTable | None:
        """a table's declaration.

        :param name: the relation as written
        :ptype name: str
        :return: the declaration, or None when none is found (or the table is interpolated)
        :rtype: _DeclaredTable | None
        """
        bare = bare_table(name)
        return self.tables.get(bare) if bare is not None else None

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

    def add_index(self, index: _DeclaredIndex) -> None:
        """record an index of a table, merging a second declaration of the same name (the purpose of either wins).

        :param index: the index
        :ptype index: _DeclaredIndex
        :return: nothing
        :rtype: None
        """
        entry = self._entry(index.table, index.origin)
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
                entry.indexes[new] = index

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
    match = re.match(r"^\(?\s*(\"[^\"]+\"|[\w{}]+)\s*\)?(?:\s|$)", stripped)
    if match is None or stripped[len(match.group(1)) :].lstrip().startswith("("):
        return stripped.lower()
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
            elif _NOT_A_COLUMN.match(item) is None and _INLINE_PK.search(item):
                key = (_column_of(item.split()[0]),)
        events.append((match.start(), "table", (match.group(1), key)))
    for match in _ADD_PK.finditer(text):
        inner, _ = _paren_body(text, match.end() - 1)
        events.append((match.start(), "table", (match.group(1), tuple(_column_of(p) for p in _split_top(inner)))))
    for match in _CREATE_INDEX.finditer(text):
        inner, after = _paren_body(text, match.end() - 1)
        include: tuple[str, ...] = ()
        rest = text[after:]
        if _INCLUDE.match(rest):
            inc, _ = _paren_body(rest, rest.index("("))
            include = tuple(_column_of(part) for part in _split_top(inc))
        notes = _notes_before(text, match.start())
        columns = tuple(_column_of(part) for part in _split_top(_strip_comments(inner)))
        events.append(
            (
                match.start(),
                "index",
                (match.group(2) or "", match.group(3), columns, include, bool(match.group(1)), notes),
            )
        )
    for match in _DROP_INDEX.finditer(text):
        for name in _split_top(match.group(1)):
            events.append((match.start(), "drop_index", name))
    for match in _DROP_TABLE.finditer(text):
        events.append((match.start(), "drop_table", match.group(1)))
    for match in _RENAME_TABLE.finditer(text):
        events.append((match.start(), "rename_table", (match.group(1), match.group(2))))
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
        if kind == "table":
            table = bare_table(payload[0])
            if table is not None:
                catalog.add_key(table, payload[1], origin)
        elif kind == "index":
            name, table_written, columns, include, _, notes = payload
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
            )
            catalog.add_index(index)
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
        values.text(_keyword(call, "purpose")) or "",
        values.text(_keyword(call, "plan_test")) or "",
        values.text(_keyword(call, "reason")) or "",
        origin,
        local,
        "" if written is not None else _name_pattern(name_node),
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
    calls: list[ast.Call] = []
    strings: list[ast.Constant] = []
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            docstrings.add(id(node.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _DDL_HINT.search(node.value):
            strings.append(node)
    for call in calls:
        final = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
        origin = f"{relative}:{call.lineno}"
        if final in values.table_schema_aliases:
            table_text = values.text(_keyword(call, "name"))
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
                if (
                    isinstance(element, ast.Call)
                    and (getattr(element.func, "attr", None) or getattr(element.func, "id", None)) == "Index"
                ):
                    index = _index_from_call(element, values, table, f"{relative}:{element.lineno}", local=local)
                    claimed.add(id(element))
                    if index is not None:
                        _add_site(catalog, index)
    for call in calls:
        final = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
        if final in values.index_aliases and id(call) not in claimed and (call.args or _keyword(call, "name")):
            index = _index_from_call(call, values, "", f"{relative}:{call.lineno}", local=local)
            if index is not None:
                _add_site(catalog, index)
    # a docstring's example DDL declares nothing
    for node in strings:
        if id(node) not in docstrings:
            _apply_ddl(catalog, str(node.value), f"{relative}:{node.lineno}", local=local)


def _add_site(catalog: SchemaCatalog, index: _DeclaredIndex) -> None:
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
                            )
                            _add_site(catalog, declared)
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
        return f"{module}::{name} never calls {calls}: the plan, not the test's word, shows the index is used"
    return None


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
    for site in sorted((s for s in catalog.sites if s.local), key=lambda s: (s.origin, s.name)):
        where = f"{site.origin} index {site.name}" + (f" on {site.table}" if site.table else "")
        table = catalog.tables.get(site.table) if site.table else None
        crowded = table is not None and len(table.indexes) > MAX_PLAIN_INDEXES
        if not site.purpose:
            findings.append(
                f"{where}: index-no-purpose -- names no purpose: give purpose='<path>::<function>' (the keyed read that "
                "uses it) and plan_test='<test file>::<test>'; an index no read can name is a candidate for removal"
            )
        elif "::" not in site.purpose or site.purpose not in definitions:
            findings.append(
                f"{where}: index-purpose-unknown -- purpose {site.purpose!r} names no function the walk found; give "
                "<path>::<qualified function>"
            )
        elif site.table and (site.table, site.name) not in catalog.used and not site.pattern:
            findings.append(
                f"{where}: index-purpose-unused -- {site.purpose} holds no read of {site.table} keyed by its leading "
                f"column {site.columns[0] if site.columns else '?'}: the purpose names the read that uses it"
            )
        if site.purpose and not site.plan_test:
            findings.append(f"{where}: index-no-plan-test -- names no plan_test proving its read uses it")
        elif site.plan_test:
            defect = _test_defect(repo_root, site.plan_test, names=(site.name,), calls="plan_uses_index")
            if defect is not None:
                findings.append(f"{where}: index-plan-test -- the plan test {defect}")
        if site.include and len(site.reason) < _MIN_REASON:
            findings.append(
                f"{where}: index-wide-no-reason -- carries INCLUDE {list(site.include)} with no written reason "
                f"({_MIN_REASON}+ characters): a wide index is how a whole scan hides behind one"
            )
        if crowded and len(site.reason) < _MIN_REASON and table is not None:
            findings.append(
                f"{where}: index-crowded-no-reason -- {site.table} carries {len(table.indexes)} indexes, more than "
                f"{MAX_PLAIN_INDEXES}, and this one gives no written reason ({_MIN_REASON}+ characters)"
            )
    return findings
