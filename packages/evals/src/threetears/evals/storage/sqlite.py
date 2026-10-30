"""A durable :class:`~threetears.evals.schema.store_port.DocumentStore` in one SQLite file.

The store for keeping runs past the process without running a database: the standard library's
:mod:`sqlite3`, one file, no dependency. Hand it to ``Eval(...).run(store=...)`` (an eval opens one under
``.evals/`` by default), and every run, result and campaign is in the file the next process opens.

Every rule the port states holds as the in-memory reference store holds it, and the store conformance
kit (``threetears.evals.testing``) proves it case by case:

- **One table**, keyed by ``(scope_id, id)``. ``doc_type`` is a column of its own, so every typed read
  selects on it, and the document itself is stored as its JSON text. The etag lives in a column beside
  the document, never in it, so a read has nothing to strip.
- **Predicates and ordering run in SQLite**, on the stored JSON type: ``json_type`` tells ``true`` from
  ``1`` and a number from its text, as the port requires, and a document without the order field (or
  with it null) sorts below every document that has it. A projection (``exclude``, ``keep``) is applied
  on the way out with the port's own :func:`~threetears.evals.schema.store_port.omit_paths` and
  :func:`~threetears.evals.schema.store_port.keep_fields`.
- **Optimistic concurrency** is one statement: a conditional write is an ``UPDATE ... WHERE etag = ?``
  that lands or touches no row, so its compare and its write cannot be split, between threads or between
  processes. Every write mints a random etag, so a document deleted and written again never honours a
  token from before. A create-only write is a plain ``INSERT`` the ``(scope_id, id)`` key refuses, one
  statement too.
- **Narrowed and grouped reads** (:class:`~threetears.evals.schema.store_port.ResultReadings`, which this store
  offers): a narrowed read is one statement with :meth:`SqliteDocumentStore.by_doc_type`'s predicates and order,
  run on a read-only connection of its own (for a file) and its rows narrowed in Python as they are fetched; a
  grouped read is one ``GROUP BY`` statement over ``json_extract``, with its refusals counted by ``json_type`` in
  the same statement. Both use only JSON1 functions, ``FILTER`` on an aggregate (SQLite 3.30) and nothing newer.

**Threads and processes.** The port is synchronous; the engine calls it from a blocking-I/O executor, so
one connection is shared across threads under a lock. The file is opened in WAL mode with a busy timeout,
so a second process can read the runs while another writes them, and a writer waits for a lock rather
than failing at once. A network filesystem is outside what SQLite's locking promises; keep the file on a
local disk.

**The file's layout is versioned** (``PRAGMA user_version``): a file written by a later layout is refused,
never read as this one. The documents inside it are the engine's, read as from any store: a later release reads the evidence
core it holds (runs, results, cases, campaigns and the definitions they name) through the core's upgraders,
and refuses a regenerable document written under another version: an analysis or an insight, which the host
regenerates from the core, or a cassette, whose format has a version of its own and which the host re-captures
(:mod:`~threetears.evals.schema.versioning`).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from threetears.evals.schema.store_port import (
    GroupMeasure,
    StoreConflict,
    field_in_values,
    keep_fields,
    narrow_items,
    omit_paths,
)

__all__ = ["SQLITE_STORE_LAYOUT", "SqliteDocumentStore"]

#: The file layout this store writes and reads, stamped as the file's ``user_version``.
SQLITE_STORE_LAYOUT = 1

#: The fields that locate a document, which :meth:`SqliteDocumentStore.merge_fields` refuses to set.
_LOCATING_FIELDS = frozenset({"id", "scope_id", "doc_type"})

#: The kinds :meth:`SqliteDocumentStore.grouped` computes; ``count`` alone reads no path.
_MEASURE_KINDS = frozenset({"count", "count_null", "sum_int", "min", "max", "values"})

#: Rows :meth:`SqliteDocumentStore.iter_narrowed` fetches from its one statement at a time, under the lock.
_STREAM_PAGE = 256

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_id TEXT NOT NULL,
    id TEXT NOT NULL,
    doc_type TEXT,
    etag TEXT NOT NULL,
    body TEXT NOT NULL,
    UNIQUE (scope_id, id)
);
CREATE INDEX IF NOT EXISTS documents_by_type ON documents (scope_id, doc_type);
"""


def _path(field: str) -> str:
    """The SQLite JSON path naming one top-level field, quoted so any key but one holding ``"`` is a key."""
    if '"' in field:
        raise ValueError(f"a field name cannot hold a double quote: {field!r}")
    return f'$."{field}"'


def _predicate(field: str, value: Any) -> tuple[str, list[Any]]:
    """One equality predicate on the stored JSON type, as SQL and its parameters.

    :raises TypeError: ``value`` is not a scalar the port compares (``str``, ``int``, ``float``, ``bool``, ``None``).
    """
    path = _path(field)
    if value is None:
        return "(json_type(body, ?) IS NULL OR json_type(body, ?) = 'null')", [path, path]
    if isinstance(value, bool):
        return "json_type(body, ?) = ?", [path, "true" if value else "false"]
    if isinstance(value, int | float):
        return "(json_type(body, ?) IN ('integer', 'real') AND json_extract(body, ?) = ?)", [path, path, value]
    if isinstance(value, str):
        return "(json_type(body, ?) = 'text' AND json_extract(body, ?) = ?)", [path, path, value]
    raise TypeError(f"by_doc_type compares a field to a scalar, not {type(value).__name__}: {field}={value!r}")


def _membership(field: str, values: Sequence[Any]) -> tuple[str, list[Any]]:
    """One membership predicate on the stored JSON type, as SQL and its parameters: the field equals any of ``values``.

    The values are grouped by the JSON type they can equal, and each group travels as ONE JSON array read back with
    ``json_each``, so the parameter count is fixed whatever the number of values and no list outgrows SQLite's
    bound-parameter limit. The values are scalars: :func:`~threetears.evals.schema.store_port.field_in_values`
    refused any other before this is reached.
    """
    path = _path(field)
    clauses: list[str] = []
    params: list[Any] = []
    if any(value is None for value in values):
        clauses.append("(json_type(body, ?) IS NULL OR json_type(body, ?) = 'null')")
        params += [path, path]
    flags = sorted({"true" if value else "false" for value in values if isinstance(value, bool)})
    if flags:
        clauses.append("json_type(body, ?) IN (SELECT value FROM json_each(?))")
        params += [path, json.dumps(flags)]
    numbers = [value for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
    if numbers:
        clauses.append(
            "(json_type(body, ?) IN ('integer', 'real') AND json_extract(body, ?) IN (SELECT value FROM json_each(?)))"
        )
        params += [path, path, json.dumps(numbers)]
    texts = [value for value in values if isinstance(value, str)]
    if texts:
        clauses.append("(json_type(body, ?) = 'text' AND json_extract(body, ?) IN (SELECT value FROM json_each(?)))")
        params += [path, path, json.dumps(texts)]
    return f"({' OR '.join(clauses)})", params


def _where(
    doc_type: str, scope_id: str, field_in: Mapping[str, Sequence[Any]] | None, field_eq: Mapping[str, Any]
) -> tuple[str, list[Any]] | None:
    """The ``WHERE`` clause of a typed read and its parameters, or ``None`` when a ``field_in`` entry is empty.

    :raises ValueError: A field is named in both ``field_in`` and ``field_eq``.
    :raises TypeError: A predicate's value is not a scalar, or a ``field_in`` entry is a string.
    """
    members = field_in_values(field_in, field_eq)
    clauses = ["scope_id = ?", "doc_type = ?"]
    params: list[Any] = [scope_id, doc_type]
    for field, value in field_eq.items():
        clause, values = _predicate(field, value)
        clauses.append(clause)
        params += values
    for field, listed in members.items():
        clause, values = _membership(field, listed)
        clauses.append(clause)
        params += values
    if any(not listed for listed in members.values()):
        return None
    return " AND ".join(clauses), params


def _dotted(path: str) -> str:
    """The SQLite JSON path naming a dotted path through objects, each key quoted as :func:`_path` quotes one."""
    if '"' in path:
        raise ValueError(f"a path cannot hold a double quote: {path!r}")
    return "$" + "".join(f'."{key}"' for key in path.split("."))


def _check_grouping(by: Sequence[str], measures: Mapping[str, GroupMeasure]) -> None:
    """Refuse a grouping the port does not define, before SQLite is asked.

    :raises TypeError: ``by`` is a string rather than a sequence of paths.
    :raises ValueError: A measure's kind is unknown, ``count`` names a path or another kind names none, or a measure
        name is also a ``by`` path.
    """
    if isinstance(by, str):
        raise TypeError(f"grouped takes a sequence of paths to group by, not a string: by={by!r}")
    if clash := sorted(set(by) & measures.keys()):
        raise ValueError(f"a measure name cannot also be a by path: {clash}")
    for name, (kind, path) in measures.items():
        if kind not in _MEASURE_KINDS:
            raise ValueError(f"measure {name!r} has kind {kind!r}; the kinds are {sorted(_MEASURE_KINDS)}")
        if (kind == "count") != (path is None):
            raise ValueError(f"measure {name!r}: count reads no path and every other kind reads one, got {path!r}")


#: A stored value's JSON type, with absent read as null.
_TYPE = "coalesce(json_type(body, ?), 'null')"


def _measure_columns(kind: str, path: str | None) -> tuple[list[str], list[Any]]:
    """The aggregate columns one measure needs, and their parameters.

    ``sum_int`` adds a count of the values it refuses, and ``min``/``max`` the extreme of the numbers, the extreme of
    the strings and a count of every other value, so the refusals are read from the same statement.
    """
    if kind == "count" or path is None:
        return ["count(*)"], []
    json_path = _dotted(path)
    if kind == "count_null":
        return [f"count(*) FILTER (WHERE {_TYPE} = 'null')"], [json_path]
    if kind == "sum_int":
        return (
            [
                f"coalesce(sum(json_extract(body, ?)) FILTER (WHERE {_TYPE} = 'integer'), 0)",
                f"count(*) FILTER (WHERE {_TYPE} NOT IN ('integer', 'null'))",
            ],
            [json_path, json_path, json_path],
        )
    extreme = "min" if kind == "min" else "max"
    return (
        [
            f"{extreme}(json_extract(body, ?)) FILTER (WHERE {_TYPE} IN ('integer', 'real'))",
            f"{extreme}(json_extract(body, ?)) FILTER (WHERE {_TYPE} = 'text')",
            f"count(*) FILTER (WHERE {_TYPE} IN ('true', 'false', 'object', 'array'))",
        ],
        [json_path, json_path, json_path, json_path, json_path],
    )


def _stored_value(json_type: str | None, value: Any) -> Any:
    """A value as stored, from its JSON type (``None`` when absent) and what ``json_extract`` read.

    ``json_extract`` reads ``true`` as ``1`` and an object or a list as its JSON text; a real comes back as the stored
    double exactly, and an integer as an integer.
    """
    if json_type == "true":
        return True
    if json_type == "false":
        return False
    if json_type in ("object", "array") and isinstance(value, str):
        return json.loads(value)
    return value


def _measure_answer(name: str, kind: str, path: str | None, columns: Sequence[Any]) -> Any:
    """One measure's answer from its columns, raising the refusal its kind states.

    :raises TypeError: ``sum_int`` met a non-integer, or ``min``/``max`` a value that is neither a number nor a
        string, or both numbers and strings.
    """
    if kind in ("count", "count_null"):
        return columns[0]
    if kind == "sum_int":
        total, refused = columns
        if refused:
            raise TypeError(f"measure {name!r} sums JSON integers, and {path!r} holds {refused} value(s) that are not")
        return total
    numbers, texts, refused = columns
    if refused:
        raise TypeError(
            f"measure {name!r} compares numbers or strings, and {path!r} holds {refused} value(s) that are neither"
        )
    if numbers is not None and texts is not None:
        raise TypeError(f"measure {name!r} compares numbers or strings, and {path!r} holds both in one group")
    return numbers if numbers is not None else texts


def _group_value(path: str, json_type: str, value: Any) -> Any:
    """A ``by`` value from its JSON type and its extracted value: SQLite hands back ``true`` as ``1``.

    :raises TypeError: The value is an object or a list.
    """
    if json_type in ("object", "array"):
        raise TypeError(f"grouped groups by scalars, and {path!r} holds a JSON {json_type}: {value}")
    if json_type == "true":
        return True
    if json_type == "false":
        return False
    return value


def _narrowed_rows(
    cursor: sqlite3.Cursor,
    exclude: Sequence[str],
    exclude_in_items: Mapping[str, Sequence[str]],
    lock: AbstractContextManager[Any],
    reader: sqlite3.Connection | None,
) -> Iterator[dict[str, Any]]:
    """Yield the cursor's rows parsed and narrowed, a page fetched at a time under ``lock``.

    The cursor is closed at the end, and so is ``reader``, the connection opened for this read alone, when there is
    one. Fetching a page continues the one statement already running; it is not another query.
    """
    try:
        while True:
            with lock:
                page = cursor.fetchmany(_STREAM_PAGE)
            if not page:
                return
            for (body,) in page:
                yield narrow_items(omit_paths(json.loads(body), exclude), exclude_in_items)
    finally:
        # Closing a connection finalises its cursors, and when the iterator is collected the connection may already
        # be closed (its finaliser runs first), where closing the cursor would raise: so a read's own connection is
        # closed, and only the shared connection's cursor.
        if reader is not None:
            reader.close()
        else:
            with lock:
                cursor.close()


class SqliteDocumentStore:
    """A :class:`~threetears.evals.schema.store_port.DocumentStore` over one SQLite file.

    :param path: The database file, created with its table when absent. ``":memory:"`` keeps it in this
        process's memory, which is the in-memory store with SQLite's semantics.
    :ptype path: str | os.PathLike[str]
    :raises ValueError: The file was written by a layout this store does not read.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open (or create) the file and its table."""
        self.path = os.fspath(path)
        # Resolved once, so a narrowed read opens this file even after the process changes directory.
        self._reader_uri = f"{Path(os.path.abspath(self.path)).as_uri()}?mode=ro"
        self._lock = threading.RLock()
        # Autocommit, so each statement is its own transaction and a read-modify-write opens one explicitly.
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30.0)
        with self._lock:
            layout = self._db.execute("PRAGMA user_version").fetchone()[0]
            if layout not in (0, SQLITE_STORE_LAYOUT):
                self._db.close()
                raise ValueError(
                    f"{self.path} holds eval documents in store layout {layout}, and this version reads layout "
                    f"{SQLITE_STORE_LAYOUT}; open it with the version of 3tears-evals that wrote it"
                )
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self._db.execute(f"PRAGMA user_version = {SQLITE_STORE_LAYOUT}")

    def close(self) -> None:
        """Close the connection; the store is unusable after."""
        with self._lock:
            self._db.close()

    def __enter__(self) -> SqliteDocumentStore:
        """Use the store as a context manager that closes it on exit."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the store."""
        self.close()

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        """Return one document by id within a scope, or ``None`` if absent.

        :param doc_id: The document's own id.
        :ptype doc_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :return: The stored document, or ``None``.
        :rtype: dict[str, Any] | None
        """
        document, _etag = self.get_with_etag(doc_id, scope_id)
        return document

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Return one document plus the token a conditional write must present.

        :param doc_id: The document's own id.
        :ptype doc_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :return: ``(document, etag)``, or ``(None, None)`` when there is no such document.
        :rtype: tuple[dict[str, Any] | None, str | None]
        """
        with self._lock:
            row = self._db.execute(
                "SELECT body, etag FROM documents WHERE scope_id = ? AND id = ?", (scope_id, doc_id)
            ).fetchone()
        if row is None:
            return None, None
        return json.loads(row[0]), row[1]

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Return the documents of one type whose ids are in ``doc_ids``, within a scope.

        :param doc_type: The discriminator every returned document carries.
        :ptype doc_type: str
        :param doc_ids: The ids to fetch; an absent one is skipped, and one named twice is returned once.
        :ptype doc_ids: Sequence[str]
        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param exclude: Dotted paths left out of every returned document.
        :ptype exclude: Sequence[str]
        :param keep: When given, the only top-level fields each returned document carries.
        :ptype keep: Sequence[str]
        :return: The matching documents, projected as asked.
        :rtype: list[dict[str, Any]]
        :raises ValueError: Both ``exclude`` and ``keep`` were given.
        """
        if exclude and keep:
            raise ValueError("get_many takes exclude or keep, never both")
        wanted = list(dict.fromkeys(doc_ids))
        if not wanted:
            return []
        documents: list[dict[str, Any]] = []
        # Under SQLite's bound-parameter limit however many ids are asked for; each row projected as it is read.
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            marks = ", ".join("?" * len(chunk))
            with self._lock:
                for row in self._db.execute(
                    f"SELECT body FROM documents WHERE scope_id = ? AND doc_type = ? AND id IN ({marks})",  # noqa: S608
                    (scope_id, doc_type, *chunk),
                ):
                    document = json.loads(row[0])
                    documents.append(keep_fields(document, keep) if keep else omit_paths(document, exclude))
        return documents

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        """Return documents of one type within a scope, matching ANDed equality and membership predicates.

        A predicate compares on the stored JSON type: ``True`` matches only ``true``, never ``1``, and a number
        never matches its text. A ``None`` predicate matches a field that is absent or null. A membership predicate
        (``field_in``) matches a field equal, by the same comparison, to any of its values; one with no values matches
        nothing, and SQLite is not asked. Under ``order_by``, documents without the field (or with it null) sort as
        lower than every document that has it.

        :param doc_type: The discriminator to select on.
        :ptype doc_type: str
        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param order_by: The field to sort by, or ``None`` for insertion order.
        :ptype order_by: str | None
        :param descending: Sort direction when ``order_by`` is given.
        :ptype descending: bool
        :param limit: Maximum documents to return; ``None`` is unbounded.
        :ptype limit: int | None
        :param exclude: Dotted paths left out of every returned document.
        :ptype exclude: Sequence[str]
        :param field_in: Field-membership predicates: field → the values it may equal.
        :ptype field_in: Mapping[str, Sequence[Any]] | None
        :param **field_eq: Field-equality predicates.
        :ptype field_eq: Any
        :return: The matching documents.
        :rtype: list[dict[str, Any]]
        :raises ValueError: A field is named in both ``field_in`` and ``field_eq``.
        :raises TypeError: A predicate's value is not a scalar, or a ``field_in`` entry is a string rather than a
            sequence of values.
        """
        where = _where(doc_type, scope_id, field_in, field_eq)
        if where is None:
            return []
        clause, params = where
        sql = f"SELECT body FROM documents WHERE {clause}"  # noqa: S608
        if order_by is not None:
            # SQLite sorts NULL lowest: first ascending, last descending, which is the port's rule for a
            # missing or null order field. Ties keep insertion order.
            sql += f" ORDER BY json_extract(body, ?) {'DESC' if descending else 'ASC'}, seq"
            params.append(_path(order_by))
        else:
            sql += " ORDER BY seq"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        # Each row projected as it is read, so a projected read of many large documents never holds their bodies at
        # once: the run's result keys, read before its results are read a page at a time, are a few scalars a row.
        with self._lock:
            return [omit_paths(json.loads(row[0]), exclude) for row in self._db.execute(sql, params)]

    def iter_narrowed(
        self,
        doc_type: str,
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        exclude_in_items: Mapping[str, Sequence[str]] = {},
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield the rows :meth:`by_doc_type` would return for the same predicates, each narrowed.

        One statement, with :meth:`by_doc_type`'s predicates and order, run when this is called. Its rows are fetched
        a page at a time, and each is parsed and narrowed in Python as it is yielded, so the bodies are never all
        held at once.

        For a file, the statement runs on a read-only connection opened for this read alone and closed when the rows
        run out, when the iterator is closed, or when it is collected. A paused read holds that connection's
        snapshot, so it sees the rows as they were at the call, and the store's own connection, which writes, holds
        none: neither this store's writes nor another process's wait on a read left part way. A ``":memory:"`` store
        has no second process and no second connection to the same database, so its statement runs on the store's
        own connection, under the lock; a write the store makes while its rows are still being yielded may or may
        not be among them. See :meth:`~threetears.evals.schema.ResultReadings.iter_narrowed`.

        :param doc_type: The discriminator to select on.
        :ptype doc_type: str
        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param exclude: Dotted paths left out of every row.
        :ptype exclude: Sequence[str]
        :param exclude_in_items: List-field path → dotted paths left out of each object item of that list.
        :ptype exclude_in_items: Mapping[str, Sequence[str]]
        :param field_in: Field-membership predicates: field → the values it may equal.
        :ptype field_in: Mapping[str, Sequence[Any]] | None
        :param **field_eq: Field-equality predicates.
        :ptype field_eq: Any
        :return: The narrowed rows, in insertion order.
        :rtype: Iterator[dict[str, Any]]
        :raises ValueError: A field is named in both ``field_in`` and ``field_eq``.
        :raises TypeError: A predicate's value is not a scalar, or a ``field_in`` entry is a string rather than a
            sequence of values.
        """
        where = _where(doc_type, scope_id, field_in, field_eq)
        if where is None:
            return iter(())
        clause, params = where
        sql = f"SELECT body FROM documents WHERE {clause} ORDER BY seq"  # noqa: S608
        if self.path == ":memory:":
            with self._lock:
                cursor = self._db.execute(sql, params)
            return _narrowed_rows(cursor, exclude, exclude_in_items, self._lock, None)
        reader = sqlite3.connect(
            self._reader_uri,
            uri=True,
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )
        try:
            cursor = reader.execute(sql, params)
        except BaseException:
            reader.close()
            raise
        rows = _narrowed_rows(cursor, exclude, exclude_in_items, nullcontext(), reader)
        # A read never iterated, or dropped part way, still closes its connection when the iterator is collected.
        weakref.finalize(rows, reader.close)
        return rows

    def grouped(
        self,
        doc_type: str,
        scope_id: str,
        *,
        by: Sequence[str],
        measures: Mapping[str, GroupMeasure],
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        """Group the rows :meth:`by_doc_type` would return and compute each measure per group, in one statement.

        SQLite groups and aggregates: each ``by`` path by its JSON type and its value (``json_extract`` reads
        ``true`` as ``1``, so the type keeps them apart), ``sum_int`` over values whose ``json_type`` is
        ``integer``, and ``min``/``max`` with the binary collation (code-point order for text). The refusals are
        counted in the same statement and raised here. ``values`` rides the same statement as a ``UNION ALL`` of one
        row per matching document (its seq, id, group key, and each path's ``json_type`` and ``json_extract``),
        ordered by seq and grouped here: never through SQLite's JSON text, which writes a real at 15 significant
        digits, so every number comes back exactly as stored.
        See :meth:`~threetears.evals.schema.ResultReadings.grouped`.

        :param doc_type: The discriminator to select on.
        :ptype doc_type: str
        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param by: Dotted paths to group by.
        :ptype by: Sequence[str]
        :param measures: Measure name → the measure.
        :ptype measures: Mapping[str, GroupMeasure]
        :param field_in: Field-membership predicates: field → the values it may equal.
        :ptype field_in: Mapping[str, Sequence[Any]] | None
        :param **field_eq: Field-equality predicates.
        :ptype field_eq: Any
        :return: One dict per group, in no promised order.
        :rtype: list[dict[str, Any]]
        :raises ValueError: The grouping is refused, or a field is named in both ``field_in`` and ``field_eq``.
        :raises TypeError: A measure met a value its kind refuses, a ``by`` value is an object or a list, or a
            predicate is refused.
        """
        _check_grouping(by, measures)
        where = _where(doc_type, scope_id, field_in, field_eq)
        if where is None:
            return []
        clause, where_params = where
        # The layout both halves share: a tag, the row count or seq, the row id, two columns per by path, the
        # aggregate measures' columns, then a JSON type and a value per values measure.
        key_columns: list[str] = []
        key_params: list[Any] = []
        group_by: list[str] = []
        for index, path in enumerate(by):
            json_path = _dotted(path)
            key_columns += [
                f"CASE {_TYPE} WHEN 'real' THEN 'integer' ELSE {_TYPE} END AS k{index}t",
                f"json_extract(body, ?) AS k{index}v",
            ]
            key_params += [json_path, json_path, json_path]
            group_by += [f"k{index}t", f"k{index}v"]
        aggregates: list[str] = []
        aggregate_params: list[Any] = []
        spans: list[tuple[str, str, str | None, int]] = []
        listed: list[tuple[str, str]] = []
        for name, (kind, measure_path) in measures.items():
            if kind == "values" and measure_path is not None:
                listed.append((name, _dotted(measure_path)))
                continue
            measure_columns, measure_params = _measure_columns(kind, measure_path)
            spans.append((name, kind, measure_path, len(measure_columns)))
            aggregates += measure_columns
            aggregate_params += measure_params
        listed_padding = ["NULL"] * (2 * len(listed))
        sql = (
            f"SELECT 'group', count(*), NULL, {', '.join([*key_columns, *aggregates, *listed_padding]) or 'NULL'} "  # noqa: S608
            f"FROM documents WHERE {clause}"
        )
        if group_by:
            sql += f" GROUP BY {', '.join(group_by)}"
        params = [*key_params, *aggregate_params, *where_params]
        if listed:
            # Each value as a row of its own, never through SQLite's JSON text, which writes a real at 15 digits:
            # json_extract hands back the stored double exactly, and the seq puts the rows in insertion order here.
            value_columns = [f"{column}(body, ?)" for _, _ in listed for column in ("json_type", "json_extract")]
            sql += (
                f" UNION ALL SELECT 'value', seq, id, "  # noqa: S608
                f"{', '.join([*key_columns, *(['NULL'] * len(aggregates)), *value_columns])} "
                f"FROM documents WHERE {clause}"
            )
            params += [*key_params, *[path for _, path in listed for _ in range(2)], *where_params]
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        answers: dict[tuple[Any, ...], dict[str, Any]] = {}
        listed_rows: list[tuple[Any, ...]] = []
        at_values = 3 + 2 * len(by) + len(aggregates)
        for row in rows:
            key = tuple(row[3 : 3 + 2 * len(by)])
            if row[0] == "value":
                listed_rows.append(row)
                continue
            if not row[1]:
                # Without a GROUP BY, an aggregate answers one row even when nothing matched.
                continue
            answer: dict[str, Any] = {
                path: _group_value(path, row[3 + 2 * index], row[4 + 2 * index]) for index, path in enumerate(by)
            }
            at = 3 + 2 * len(by)
            for name, span_kind, span_path, width in spans:
                answer[name] = _measure_answer(name, span_kind, span_path, row[at : at + width])
                at += width
            for name, _ in listed:
                answer[name] = []
            answers[key] = answer
        for row in sorted(listed_rows, key=lambda listed_row: listed_row[1]):
            answer = answers[tuple(row[3 : 3 + 2 * len(by)])]
            for index, (name, _) in enumerate(listed):
                json_type, value = row[at_values + 2 * index], row[at_values + 2 * index + 1]
                answer[name].append([row[2], _stored_value(json_type, value)])
        return [{name: answer[name] for name in [*by, *measures]} for answer in answers.values()]

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        """Yield the id of every document of one type within a scope.

        :param doc_type: The discriminator to sweep.
        :ptype doc_type: str
        :param scope_id: The scope to sweep.
        :ptype scope_id: str
        :yield: Document ids, from a snapshot taken when iteration starts, so deleting each as it is
            yielded is safe.
        :rtype: Iterator[str]
        """
        with self._lock:
            ids = [
                row[0]
                for row in self._db.execute(
                    "SELECT id FROM documents WHERE scope_id = ? AND doc_type = ? ORDER BY seq", (scope_id, doc_type)
                )
            ]
        yield from ids

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Write a document under the scope and id it carries, creating or replacing it.

        :param document: The document to store; it supplies its own ``scope_id`` and ``id``.
        :ptype document: dict[str, Any]
        :param if_match: A token from :meth:`get_with_etag`; when given, the write lands only if the
            stored document still carries it.
        :ptype if_match: str | None
        :param create_only: When True, the write lands only if nothing is stored under its scope and id: a
            plain ``INSERT``, which the table's unique key refuses, so the check and the write are one
            statement between threads and between processes.
        :ptype create_only: bool
        :raises StoreConflict: ``if_match`` was given and the stored document no longer carries it, or is gone;
            or ``create_only`` was set and a document is stored there.
        :raises ValueError: Both ``if_match`` and ``create_only`` were given.
        :raises KeyError: The document carries no ``scope_id`` or no ``id``.
        :raises TypeError: The document holds a value JSON cannot.
        """
        if if_match is not None and create_only:
            raise ValueError("a write is conditional on an etag or on absence, never both")
        scope_id, doc_id = document["scope_id"], document["id"]
        body = json.dumps(document)
        etag = uuid.uuid7().hex
        with self._lock:
            if create_only:
                try:
                    self._db.execute(
                        "INSERT INTO documents (scope_id, id, doc_type, etag, body) VALUES (?, ?, ?, ?, ?)",
                        (scope_id, doc_id, document.get("doc_type"), etag, body),
                    )
                except sqlite3.IntegrityError as exists:
                    raise StoreConflict(f"document {doc_id!r} in scope {scope_id!r} already exists") from exists
                return
            if if_match is not None:
                written = self._db.execute(
                    "UPDATE documents SET doc_type = ?, etag = ?, body = ? WHERE scope_id = ? AND id = ? AND etag = ?",
                    (document.get("doc_type"), etag, body, scope_id, doc_id, if_match),
                ).rowcount
                if written == 0:
                    raise StoreConflict(
                        f"document {doc_id!r} in scope {scope_id!r} no longer carries etag {if_match!r}"
                    )
                return
            self._db.execute(
                "INSERT INTO documents (scope_id, id, doc_type, etag, body) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (scope_id, id) DO UPDATE SET doc_type = excluded.doc_type, etag = excluded.etag, "
                "body = excluded.body",
                (scope_id, doc_id, document.get("doc_type"), etag, body),
            )

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        """Set top-level fields on one stored document, leaving every other field as stored.

        :param doc_id: The document's own id.
        :ptype doc_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :param fields: Top-level fields and their new values.
        :ptype fields: dict[str, Any]
        :return: ``True`` when the document was written; ``False`` when there is no such document.
        :rtype: bool
        :raises ValueError: ``fields`` is empty or names a field that locates the document.
        """
        if not fields:
            raise ValueError("merge_fields needs at least one field")
        if locating := sorted(_LOCATING_FIELDS & fields.keys()):
            raise ValueError(f"merge_fields cannot set the fields that locate a document: {locating}")
        with self._lock:
            # One write transaction from the read to the write, so another process's write cannot land between.
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT body FROM documents WHERE scope_id = ? AND id = ?", (scope_id, doc_id)
                ).fetchone()
                if row is None:
                    self._db.execute("COMMIT")
                    return False
                document = json.loads(row[0])
                document.update(fields)
                self._db.execute(
                    "UPDATE documents SET etag = ?, body = ? WHERE scope_id = ? AND id = ?",
                    (uuid.uuid7().hex, json.dumps(document), scope_id, doc_id),
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            return True

    def delete(self, doc_id: str, scope_id: str) -> bool:
        """Delete one document by id within a scope.

        :param doc_id: The document's own id.
        :ptype doc_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :return: ``True`` when a document was deleted, ``False`` when there was none.
        :rtype: bool
        """
        with self._lock:
            deleted = self._db.execute("DELETE FROM documents WHERE scope_id = ? AND id = ?", (scope_id, doc_id))
            return deleted.rowcount > 0
