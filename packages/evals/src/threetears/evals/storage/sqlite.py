"""A durable :class:`~threetears.evals.schema.store_port.DocumentStore` in one SQLite file.

The store for keeping runs past the process without running a database: the standard library's
:mod:`sqlite3`, one file, no dependency. Hand it to ``run_eval(..., store=...)`` or
``compare(..., store=...)``, and every run, result and campaign is in the file the next process opens.

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
  token from before.

**Threads and processes.** The port is synchronous; the engine calls it from a blocking-I/O executor, so
one connection is shared across threads under a lock. The file is opened in WAL mode with a busy timeout,
so a second process can read the runs while another writes them, and a writer waits for a lock rather
than failing at once. A network filesystem is outside what SQLite's locking promises; keep the file on a
local disk.

**The file's layout is versioned** (``PRAGMA user_version``): a file written by a later layout is refused,
never read as this one. The documents inside it are the engine's, read as from any store: a later release reads the evidence
core it holds (runs, results, cases and the definitions they name) through the core's upgraders, and
refuses a regenerable document (a campaign, an analysis, an insight) written under another version, which
the host regenerates from the core (:mod:`~threetears.evals.schema.versioning`).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from typing import Any

from threetears.evals.schema.store_port import StoreConflict, keep_fields, omit_paths

__all__ = ["SQLITE_STORE_LAYOUT", "SqliteDocumentStore"]

#: The file layout this store writes and reads, stamped as the file's ``user_version``.
SQLITE_STORE_LAYOUT = 1

#: The fields that locate a document, which :meth:`SqliteDocumentStore.merge_fields` refuses to set.
_LOCATING_FIELDS = frozenset({"id", "scope_id", "doc_type"})

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

    Raises:
        TypeError: ``value`` is not a scalar the port compares (``str``, ``int``, ``float``, ``bool``, ``None``).
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


class SqliteDocumentStore:
    """A :class:`~threetears.evals.schema.store_port.DocumentStore` over one SQLite file.

    Args:
        path: The database file, created with its table when absent. ``":memory:"`` keeps it in this
            process's memory, which is the in-memory store with SQLite's semantics.

    Raises:
        ValueError: The file was written by a layout this store does not read.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open (or create) the file and its table."""
        self.path = os.fspath(path)
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

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.

        Returns:
            The stored document, or ``None``.
        """
        document, _etag = self.get_with_etag(doc_id, scope_id)
        return document

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Return one document plus the token a conditional write must present.

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.

        Returns:
            ``(document, etag)``, or ``(None, None)`` when there is no such document.
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

        Args:
            doc_type: The discriminator every returned document carries.
            doc_ids: The ids to fetch; an absent one is skipped, and one named twice is returned once.
            scope_id: The scope to read.
            exclude: Dotted paths left out of every returned document.
            keep: When given, the only top-level fields each returned document carries.

        Returns:
            The matching documents, projected as asked.

        Raises:
            ValueError: Both ``exclude`` and ``keep`` were given.
        """
        if exclude and keep:
            raise ValueError("get_many takes exclude or keep, never both")
        wanted = list(dict.fromkeys(doc_ids))
        if not wanted:
            return []
        bodies: list[str] = []
        # Under SQLite's bound-parameter limit however many ids are asked for.
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            marks = ", ".join("?" * len(chunk))
            with self._lock:
                bodies += [
                    row[0]
                    for row in self._db.execute(
                        f"SELECT body FROM documents WHERE scope_id = ? AND doc_type = ? AND id IN ({marks})",  # noqa: S608
                        (scope_id, doc_type, *chunk),
                    )
                ]
        documents = [json.loads(body) for body in bodies]
        return [keep_fields(d, keep) if keep else omit_paths(d, exclude) for d in documents]

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        """Return documents of one type within a scope, matching ANDed equality predicates.

        A predicate compares on the stored JSON type: ``True`` matches only ``true``, never ``1``, and a number
        never matches its text. A ``None`` predicate matches a field that is absent or null. Under ``order_by``,
        documents without the field (or with it null) sort as lower than every document that has it.

        Args:
            doc_type: The discriminator to select on.
            scope_id: The scope to read.
            order_by: The field to sort by, or ``None`` for insertion order.
            descending: Sort direction when ``order_by`` is given.
            limit: Maximum documents to return; ``None`` is unbounded.
            exclude: Dotted paths left out of every returned document.
            **field_eq: Field-equality predicates.

        Returns:
            The matching documents.

        Raises:
            TypeError: A predicate's value is not a scalar.
        """
        clauses = ["scope_id = ?", "doc_type = ?"]
        params: list[Any] = [scope_id, doc_type]
        for field, value in field_eq.items():
            clause, values = _predicate(field, value)
            clauses.append(clause)
            params += values
        sql = f"SELECT body FROM documents WHERE {' AND '.join(clauses)}"  # noqa: S608
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
        with self._lock:
            bodies = [row[0] for row in self._db.execute(sql, params)]
        return [omit_paths(json.loads(body), exclude) for body in bodies]

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        """Yield the id of every document of one type within a scope.

        Args:
            doc_type: The discriminator to sweep.
            scope_id: The scope to sweep.

        Yields:
            Document ids, from a snapshot taken when iteration starts, so deleting each as it is
            yielded is safe.
        """
        with self._lock:
            ids = [
                row[0]
                for row in self._db.execute(
                    "SELECT id FROM documents WHERE scope_id = ? AND doc_type = ? ORDER BY seq", (scope_id, doc_type)
                )
            ]
        yield from ids

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        """Write a document under the scope and id it carries, creating or replacing it.

        Args:
            document: The document to store; it supplies its own ``scope_id`` and ``id``.
            if_match: A token from :meth:`get_with_etag`; when given, the write lands only if the
                stored document still carries it.

        Raises:
            StoreConflict: ``if_match`` was given and the stored document no longer carries it, or is gone.
            KeyError: The document carries no ``scope_id`` or no ``id``.
            TypeError: The document holds a value JSON cannot.
        """
        scope_id, doc_id = document["scope_id"], document["id"]
        body = json.dumps(document)
        etag = uuid.uuid4().hex
        with self._lock:
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

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.
            fields: Top-level fields and their new values.

        Returns:
            ``True`` when the document was written; ``False`` when there is no such document.

        Raises:
            ValueError: ``fields`` is empty or names a field that locates the document.
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
                    (uuid.uuid4().hex, json.dumps(document), scope_id, doc_id),
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            return True

    def delete(self, doc_id: str, scope_id: str) -> bool:
        """Delete one document by id within a scope.

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.

        Returns:
            ``True`` when a document was deleted, ``False`` when there was none.
        """
        with self._lock:
            deleted = self._db.execute("DELETE FROM documents WHERE scope_id = ? AND id = ?", (scope_id, doc_id))
            return deleted.rowcount > 0
