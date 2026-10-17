"""where a row's file lives: the layout half of the plug-in boundary.

A :class:`Layout` maps a table and a primary key (pk) to a path in the working tree, a filter to a
glob of candidate files, and a changed path back to the table and pk it holds, so a commit made
elsewhere can be turned into cache drops. It also owns how a row is put into its file and read
back, through the table's codec.

:class:`TemplateLayout` builds one from path templates (:class:`TableLayout`), which is what a
product whose files are named by their keys uses. A product whose file names are not its keys
(scriob names files by slug, its ids live in frontmatter) writes its own :class:`Layout`, returning
``None`` from :meth:`Layout.path_for` so the store finds a row by scanning.

Path values are escaped: ``%``, ``/``, ``\\``, control characters, a leading ``.`` and the
character that ends the value in its template (the ``-`` of ``{tile}-{n}.json``) are
percent-encoded, and an empty value, ``.`` and ``..`` are refused, so a pk can never point outside
its folder and every path reads back to exactly one pk.
"""

from __future__ import annotations

import glob
import re
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from threetears.git_store.codec import RowCodec

__all__ = ["Layout", "TableLayout", "TemplateLayout", "escape_value", "locate", "same_key", "unescape_value"]


@runtime_checkable
class Layout(Protocol):
    """where each table's rows live in a working tree, and how a row is put in its file.

    :ivar reads_content: whether :meth:`identify` needs a file's bytes; ``False`` lets a changed path
        be mapped to its row without reading it
    """

    reads_content: bool

    def tables(self) -> Sequence[str]:
        """every table the layout places.

        :return: the table names
        :rtype: Sequence[str]
        """
        ...

    def pk_columns(self, table: str) -> tuple[str, ...]:
        """the table's primary key columns, in order.

        :param table: the table
        :ptype table: str
        :return: the column names
        :rtype: tuple[str, ...]
        """
        ...

    def version_column(self, table: str) -> str | None:
        """the column a compare-and-swap reads, or ``None`` when the table has none.

        :param table: the table
        :ptype table: str
        :return: the column name, or ``None``
        :rtype: str | None
        """
        ...

    def path_for(self, table: str, pk: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> PurePosixPath | None:
        """the path of the row with key ``pk``, or ``None`` when the key alone cannot say.

        :param table: the table
        :ptype table: str
        :param pk: the key, column to value
        :ptype pk: Mapping[str, Any]
        :param row: the whole row, when a write is placing it: a layout whose file names come from
            the row's content (a slug) places a new row from it
        :ptype row: Mapping[str, Any] | None
        :return: the path, relative to the working tree's root; ``None`` makes the store look for
            the row's existing file among :meth:`candidates`
        :rtype: PurePosixPath | None
        """
        ...

    def candidates(self, table: str, filters: Mapping[str, Any] | None) -> str:
        """a glob of every file that may hold a row of ``table`` matching ``filters``.

        :param table: the table
        :ptype table: str
        :param filters: equality filters; any on a path column narrow the glob
        :ptype filters: Mapping[str, Any] | None
        :return: a glob, relative to the working tree's root
        :rtype: str
        """
        ...

    def identify(self, path: PurePosixPath, data: bytes | None) -> tuple[str, dict[str, Any]] | None:
        """the table and key a file holds, or ``None`` when it holds no row.

        :param path: the path, relative to the working tree's root
        :ptype path: PurePosixPath
        :param data: the file's bytes, when :attr:`reads_content`; else ``None``
        :ptype data: bytes | None
        :return: ``(table, pk)``, or ``None``
        :rtype: tuple[str, dict[str, Any]] | None
        """
        ...

    def encode(self, table: str, row: Mapping[str, Any]) -> bytes:
        """the bytes of the file holding ``row``.

        :param table: the table
        :ptype table: str
        :param row: the whole row, key columns included
        :ptype row: Mapping[str, Any]
        :return: the file's bytes
        :rtype: bytes
        """
        ...

    def decode(self, table: str, path: PurePosixPath, data: bytes) -> dict[str, Any]:
        """the whole row a file holds, key columns included.

        :param table: the table
        :ptype table: str
        :param path: the file's path, relative to the working tree's root
        :ptype path: PurePosixPath
        :param data: the file's bytes
        :ptype data: bytes
        :return: the row
        :rtype: dict[str, Any]
        """
        ...


def escape_value(value: Any, *, stop: str | None = None) -> str:
    """a key value as one path component.

    :param value: the value; written with ``str()``
    :ptype value: Any
    :param stop: the character that ends this value in its template, encoded too
    :ptype stop: str | None
    :return: the component
    :rtype: str
    :raises ValueError: for an empty value, ``.`` or ``..``
    """
    text = str(value)
    if text in ("", ".", ".."):
        raise ValueError(f"a git-store key value cannot be {text!r}: it would not name one file")
    out: list[str] = []
    for index, char in enumerate(text):
        code = ord(char)
        if (
            char in "%/\\"
            or code < 0x20
            or code == 0x7F
            or (index == 0 and char == ".")
            or (stop is not None and char == stop)
        ):
            out.append("".join(f"%{byte:02X}" for byte in char.encode("utf-8")))
        else:
            out.append(char)
    return "".join(out)


def unescape_value(text: str) -> str:
    """reverse :func:`escape_value`.

    :param text: the path component
    :ptype text: str
    :return: the value
    :rtype: str
    """
    return re.sub(
        r"(?:%[0-9A-F]{2})+",
        lambda match: bytes.fromhex(match.group(0).replace("%", "")).decode("utf-8"),
        text,
    )


@dataclass(frozen=True, slots=True)
class _Field:
    """one ``{column}`` of a path template.

    :ivar column: the pk column
    :ivar stop: the character right after it in its path segment, or ``None`` at the segment's end
    """

    column: str
    stop: str | None


@dataclass(frozen=True)
class TableLayout:
    """one table's place in the working tree.

    :ivar table: the table name
    :ivar path: the path template, relative to the root: ``"{character}/keepers/{batch}/{tile}.json"``.
        Every pk column appears in it exactly once, and two fields never touch
    :ivar pk: the primary key columns, in order
    :ivar codec: how a row is put in its file
    :ivar version_column: the column a compare-and-swap reads; ``None`` turns compare-and-swap off
        for the table (a write that asks for one is refused, never applied unfenced)
    :ivar pk_in_file: ``False`` (the default) keeps the pk columns out of the file: they live in the
        path and are added back on read
    :ivar pk_types: how a key column read from a path is turned back into its value (``int``);
        ``str`` when not named
    """

    table: str
    path: str
    pk: tuple[str, ...]
    codec: RowCodec
    version_column: str | None = "date_updated"
    pk_in_file: bool = False
    pk_types: Mapping[str, Callable[[str], Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """refuse a template that cannot name every key exactly once.

        :return: None
        :rtype: None
        :raises ValueError: for a template missing a pk column, naming a column twice, naming a
            column outside the pk, with two fields touching, or that is absolute or climbs out
        """
        if not self.pk:
            raise ValueError(f"{self.table}: a table needs at least one pk column")
        parsed = list(string.Formatter().parse(self.path))
        names = [name for _, name, _, _ in parsed if name is not None]
        if sorted(names) != sorted(self.pk) or len(set(names)) != len(names):
            raise ValueError(f"{self.table}: the path {self.path!r} must name each pk column {self.pk} exactly once")
        for (_, name, _, _), (literal, _, _, _) in zip(parsed, parsed[1:], strict=False):
            if name is not None and literal == "":
                raise ValueError(f"{self.table}: two fields touch in {self.path!r}; put a separator between them")
        if self.path.startswith("/") or ".." in PurePosixPath(self.path).parts:
            raise ValueError(f"{self.table}: the path {self.path!r} must stay inside the working tree")

    def fields(self) -> list[_Field]:
        """the template's fields, each with the character that ends it.

        :return: the fields in template order
        :rtype: list[_Field]
        """
        parsed = list(string.Formatter().parse(self.path))
        found: list[_Field] = []
        for index, (_, name, _, _) in enumerate(parsed):
            if name is None:
                continue
            following = parsed[index + 1][0] if index + 1 < len(parsed) else ""
            stop = following[0] if following and following[0] != "/" else None
            found.append(_Field(name, stop))
        return found


class TemplateLayout:
    """a :class:`Layout` built from one :class:`TableLayout` per table.

    A changed path is matched against the tables in the order given; the first whose template
    matches names it. Templates that can match the same path should therefore be avoided, or listed
    most specific first.

    :ivar reads_content: ``False``: the path alone names the row
    """

    reads_content = False

    def __init__(self, tables: Sequence[TableLayout]) -> None:
        """index the tables.

        :param tables: each table's layout
        :ptype tables: Sequence[TableLayout]
        :return: None
        :rtype: None
        :raises ValueError: when a table is named twice
        """
        self._tables: dict[str, TableLayout] = {}
        self._patterns: list[tuple[TableLayout, re.Pattern[str]]] = []
        self._by_table: dict[str, re.Pattern[str]] = {}
        for layout in tables:
            if layout.table in self._tables:
                raise ValueError(f"table {layout.table!r} is laid out twice")
            self._tables[layout.table] = layout
            self._by_table[layout.table] = self._pattern(layout)
            self._patterns.append((layout, self._by_table[layout.table]))

    @staticmethod
    def _pattern(layout: TableLayout) -> re.Pattern[str]:
        """the regular expression a path of ``layout`` matches, one named group per pk column.

        :param layout: the table
        :ptype layout: TableLayout
        :return: the compiled expression
        :rtype: re.Pattern[str]
        """
        stops = {item.column: item.stop for item in layout.fields()}
        parts: list[str] = []
        for literal, name, _, _ in string.Formatter().parse(layout.path):
            parts.append(re.escape(literal))
            if name is not None:
                stop = stops[name]
                excluded = "/" + (re.escape(stop) if stop is not None else "")
                parts.append(f"(?P<{name}>[^{excluded}]+)")
        return re.compile("".join(parts))

    def table(self, table: str) -> TableLayout:
        """one table's layout.

        :param table: the table
        :ptype table: str
        :return: its layout
        :rtype: TableLayout
        :raises KeyError: for a table this layout does not place
        """
        try:
            return self._tables[table]
        except KeyError:
            raise KeyError(f"table {table!r} has no place in this git-store layout") from None

    def tables(self) -> Sequence[str]:
        """every table the layout places.

        :return: the table names, in the order given
        :rtype: Sequence[str]
        """
        return list(self._tables)

    def pk_columns(self, table: str) -> tuple[str, ...]:
        """the table's primary key columns.

        :param table: the table
        :ptype table: str
        :return: the columns
        :rtype: tuple[str, ...]
        """
        return self.table(table).pk

    def version_column(self, table: str) -> str | None:
        """the table's compare-and-swap column.

        :param table: the table
        :ptype table: str
        :return: the column, or ``None``
        :rtype: str | None
        """
        return self.table(table).version_column

    def path_for(self, table: str, pk: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> PurePosixPath:
        """the path of the row with key ``pk``.

        :param table: the table
        :ptype table: str
        :param pk: the key; every pk column must be present
        :ptype pk: Mapping[str, Any]
        :param row: unused; the key alone names the file
        :ptype row: Mapping[str, Any] | None
        :return: the path
        :rtype: PurePosixPath
        :raises KeyError: when a pk column is missing
        :raises ValueError: for a key value that cannot name a file
        """
        del row
        layout = self.table(table)
        values = {item.column: escape_value(pk[item.column], stop=item.stop) for item in layout.fields()}
        return PurePosixPath(layout.path.format_map(values))

    def candidates(self, table: str, filters: Mapping[str, Any] | None) -> str:
        """a glob of the files that may hold matching rows; a filter on a pk column narrows it.

        :param table: the table
        :ptype table: str
        :param filters: equality filters
        :ptype filters: Mapping[str, Any] | None
        :return: the glob
        :rtype: str
        """
        layout = self.table(table)
        stops = {item.column: item.stop for item in layout.fields()}
        wanted = dict(filters or {})
        parts: list[str] = []
        for literal, name, _, _ in string.Formatter().parse(layout.path):
            parts.append(glob.escape(literal))
            if name is not None:
                if name in wanted:
                    parts.append(glob.escape(escape_value(wanted[name], stop=stops[name])))
                else:
                    parts.append("*")
        return "".join(parts)

    def identify(self, path: PurePosixPath, data: bytes | None) -> tuple[str, dict[str, Any]] | None:
        """the table and key ``path`` holds, by its template.

        :param path: the path, relative to the root
        :ptype path: PurePosixPath
        :param data: unused; the path alone names the row
        :ptype data: bytes | None
        :return: ``(table, pk)``, or ``None`` when no template matches
        :rtype: tuple[str, dict[str, Any]] | None
        """
        del data
        for layout, pattern in self._patterns:
            pk = self._pk_of(layout, pattern, path)
            if pk is not None:
                return layout.table, pk
        return None

    @staticmethod
    def _pk_of(layout: TableLayout, pattern: re.Pattern[str], path: PurePosixPath) -> dict[str, Any] | None:
        """the key ``path`` names under one table's template, or ``None`` when it is not that table's.

        :param layout: the table
        :ptype layout: TableLayout
        :param pattern: its compiled template
        :ptype pattern: re.Pattern[str]
        :param path: the path
        :ptype path: PurePosixPath
        :return: the key, or ``None``
        :rtype: dict[str, Any] | None
        """
        match = pattern.fullmatch(path.as_posix())
        if match is None:
            return None
        try:
            return {
                column: layout.pk_types.get(column, str)(unescape_value(match.group(column))) for column in layout.pk
            }
        except ValueError:  # NOSILENT: a value of the wrong type (a file a person named): not this table's row
            return None

    def encode(self, table: str, row: Mapping[str, Any]) -> bytes:
        """the file's bytes for ``row``, the pk columns left out unless the table keeps them.

        :param table: the table
        :ptype table: str
        :param row: the whole row
        :ptype row: Mapping[str, Any]
        :return: the bytes
        :rtype: bytes
        """
        layout = self.table(table)
        content = dict(row) if layout.pk_in_file else {k: v for k, v in row.items() if k not in layout.pk}
        return layout.codec.encode(content)

    def decode(self, table: str, path: PurePosixPath, data: bytes) -> dict[str, Any]:
        """the whole row a file holds, the pk columns put back from the path.

        :param table: the table
        :ptype table: str
        :param path: the file's path
        :ptype path: PurePosixPath
        :param data: its bytes
        :ptype data: bytes
        :return: the row, pk columns first
        :rtype: dict[str, Any]
        :raises ValueError: when the path is not one of ``table``'s
        """
        layout = self.table(table)
        content = layout.codec.decode(data)
        if layout.pk_in_file:
            return content
        pk = self._pk_of(layout, self._by_table[table], path)
        if pk is None:
            raise ValueError(f"{path} is not a {table} file")
        return {**pk, **{k: v for k, v in content.items() if k not in layout.pk}}


def same_key(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """whether two keys name the same row, comparing values as text (a path gives text back).

    :param left: one key
    :ptype left: Mapping[str, Any]
    :param right: the other
    :ptype right: Mapping[str, Any]
    :return: ``True`` when every column agrees
    :rtype: bool
    """
    return left.keys() == right.keys() and all(str(left[column]) == str(right[column]) for column in left)


def locate(
    layout: Layout, root: Path, table: str, pk: Mapping[str, Any], row: Mapping[str, Any] | None = None
) -> PurePosixPath | None:
    """where a row's file is, or would be: its layout's path, else the existing file holding it.

    BLOCKING when the layout cannot name the path from the key: it reads the candidates.

    :param layout: the layout
    :ptype layout: Layout
    :param root: the working tree's root
    :ptype root: Path
    :param table: the table
    :ptype table: str
    :param pk: the row's key
    :ptype pk: Mapping[str, Any]
    :param row: the whole row, when placing a write
    :ptype row: Mapping[str, Any] | None
    :return: the path relative to the root, or ``None`` when the row has no file and none can be named
    :rtype: PurePosixPath | None
    """
    path = layout.path_for(table, pk, row)
    if path is not None:
        return path
    for candidate in sorted(root.glob(layout.candidates(table, pk))):
        if not candidate.is_file():
            continue
        relative = PurePosixPath(candidate.relative_to(root).as_posix())
        found = layout.identify(relative, candidate.read_bytes() if layout.reads_content else None)
        if found is not None and found[0] == table and same_key(found[1], pk):
            return relative
    return None
