"""A scoped snapshot's copy published as shared-memory Arrow, rewriting only the scopes that moved.

**Why.** A pod whose serving process and worker processes read one shared copy (:mod:`threetears.core.shared_arrow`)
publishes a version of its scoped snapshot's L1 every time the copy changes. Writing every table whole each time
costs the whole copy in writes and, until the old version is let go, in memory. A scoped snapshot changes a scope at a
time (:mod:`~threetears.core.collections.scoped_snapshot`), so each table is published in parts, one per scope
(:class:`~threetears.core.shared_arrow.Parts`), and a version rewrites only the scopes whose epoch moved since the
version this publisher published last; every other scope's part is that version's own file, hard-linked.

**Exactly its epochs.** Each version is written from one read of the copy (:meth:`ScopedSnapshot.read_versioned`),
whose epochs name every scope's rows it reads: the parts written and the parts kept are those epochs' rows, never a
mix. A scope held at an epoch holds the same rows wherever and whenever it is read, so a part kept from the last
version is the part a fresh write would make. The scopes are those the read's tables hold (each table's scope values,
read every publish): one the epochs do not name has nothing to prove it unchanged and is written every time, and one
a table no longer holds is dropped from it.

Blocking: :meth:`SharedSnapshot.publish` reads the copy and writes files; call it from a worker thread.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from threetears.observe import get_logger

from threetears.core.cache.duckdb import fetch_arrow
from threetears.core.shared_arrow import (
    DEFAULT_BATCH_ROWS,
    Parts,
    SharedArrowStore,
    SharedVersion,
    epochs_version,
    plain_name,
    query_batches,
)
from threetears.core.sql_fragments import quote_identifier

if TYPE_CHECKING:
    from threetears.core.collections.scoped_snapshot import SnapshotTable, VersionedRead

__all__ = ["SharedSnapshot"]

log = get_logger(__name__)

#: the prefix of a part named by its scope's digest, which a scope named as it is never takes
_DIGEST_PREFIX: Final = "h-"


def _scope_part(scope: str) -> str:
    """the part a scope's rows are published as: the scope itself when it is a plain name, else its digest.

    :param scope: the scope
    :ptype scope: str
    :return: the part's name
    :rtype: str
    """
    plain = plain_name(scope) and not scope.startswith(_DIGEST_PREFIX)
    return scope if plain else _DIGEST_PREFIX + hashlib.sha256(scope.encode()).hexdigest()[:40]


def _scopes(cursor: Any, table: SnapshotTable) -> frozenset[Any]:
    """the scopes a table holds rows of.

    :return: the scope values
    :rtype: frozenset[Any]
    """
    column, name = quote_identifier(table.scope_column), quote_identifier(table.name)
    return frozenset(row[0] for row in cursor.execute(f"SELECT DISTINCT {column} FROM {name}").fetchall())  # noqa: S608


def _moved(scope: str, epochs: Mapping[str, int], held: Mapping[str, int]) -> bool:
    """whether a scope's rows may differ from the last version's: its epoch moved, or no epoch names it.

    :return: True when its part must be written
    :rtype: bool
    """
    epoch = epochs.get(scope)
    return epoch is None or held.get(scope) != epoch


def _schema(cursor: Any, table: str) -> Any:
    """a table's columns as Arrow types them.

    :return: the schema
    :rtype: pa.Schema
    """
    answer = cursor.execute(f"SELECT * FROM {quote_identifier(table)} LIMIT 0")  # noqa: S608 - a trusted name
    return fetch_arrow(answer, 1).schema


class SharedSnapshot:
    """publishes a scoped snapshot's copy into a :class:`SharedArrowStore`, a scope's part at a time.

    :param store: the store, in this process
    :ptype store: SharedArrowStore
    :param tables: the snapshot's tables (each published in parts by its scope column)
    :ptype tables: Sequence[SnapshotTable]
    :param version_name: names the version a read holds; the read's epochs by default (:func:`epochs_version`). Two
        reads named alike must hold the same rows
    :ptype version_name: Callable[[VersionedRead], str] | None
    :param extra_tables: tables written whole into every version beside the snapshot's (what the read's epochs are,
        say), from the read
    :ptype extra_tables: Callable[[VersionedRead], Mapping[str, Any]] | None
    :param batch_rows: rows per batch read from DuckDB
    :ptype batch_rows: int
    """

    def __init__(
        self,
        store: SharedArrowStore,
        tables: Sequence[SnapshotTable],
        *,
        version_name: Callable[[VersionedRead], str] | None = None,
        extra_tables: Callable[[VersionedRead], Mapping[str, Any]] | None = None,
        batch_rows: int = DEFAULT_BATCH_ROWS,
    ) -> None:
        if not tables:
            raise ValueError("a shared snapshot publishes at least one table")
        self._store = store
        self._tables = tuple(tables)
        self._name = version_name or (lambda read: epochs_version(read.epochs))
        self._extra = extra_tables
        self._batch_rows = batch_rows
        #: one publish at a time, so each finds the version the last one made
        self._lock = threading.Lock()
        #: the version this published last, the epochs it holds, and each table's scopes in it
        self._last: tuple[str, Mapping[str, int], Mapping[str, frozenset[Any]]] | None = None
        self.published = 0
        self.scopes_written = 0
        self.last_seconds = 0.0

    def publish(self, read: VersionedRead) -> SharedVersion:
        """make the version ``read`` holds current: written when the store does not hold it, from the last version
        this published with only the scopes whose epoch moved written again.

        :param read: an open read of the copy, held for the call
        :ptype read: VersionedRead
        :return: the version
        :rtype: SharedVersion
        :raises SharedArrowFullError: when the store has no room for it; the current version keeps serving
        """
        name = self._name(read)
        epochs = MappingProxyType(dict(read.epochs))
        with self._lock:
            current = self._store.current
            if current is not None and current.name == name:
                return current
            started = time.monotonic()
            present = {table.name: _scopes(read.cursor, table) for table in self._tables}
            if name in self._store.versions():
                made = self._store.publish(name, {})
                self._last = (name, epochs, present)
                return made
            last = self._last
            base = last[0] if last is not None and current is not None and current.name == last[0] else None
            held_epochs: Mapping[str, int] = last[1] if base is not None and last is not None else {}
            held_parts: Mapping[str, frozenset[Any]] = last[2] if base is not None and last is not None else {}
            tables: dict[str, Any] = {}
            written: set[str] = set()
            dropped: set[str] = set()
            for table in self._tables:
                kept = held_parts.get(table.name, frozenset())
                # a scope is written when its epoch moved, when no epoch names it, or when the base lacks it
                writes = sorted(
                    (
                        scope
                        for scope in present[table.name]
                        if _moved(str(scope), epochs, held_epochs) or scope not in kept
                    ),
                    key=str,
                )
                drops = kept - present[table.name]
                rows = (
                    f"SELECT * FROM {quote_identifier(table.name)} "  # noqa: S608 - trusted names
                    f"WHERE {quote_identifier(table.scope_column)} = ?"
                )
                parts: dict[str, Any] = {
                    _scope_part(str(scope)): query_batches(read.cursor, rows, [scope], batch_rows=self._batch_rows)
                    for scope in writes
                }
                parts.update({_scope_part(str(scope)): None for scope in drops})
                tables[table.name] = Parts(schema=_schema(read.cursor, table.name), parts=parts)
                written.update(str(scope) for scope in writes)
                dropped.update(str(scope) for scope in drops)
            if self._extra is not None:
                tables.update(self._extra(read))
            made = self._store.publish(name, tables, base=base)
            self._last = (name, epochs, present)
            self.published += 1
            self.scopes_written += len(written)
            self.last_seconds = time.monotonic() - started
        log.info(
            "scoped snapshot published to shared memory",
            extra={
                "extra_data": {
                    "version": name,
                    "base": base,
                    "scopes_written": len(written),
                    "scopes_dropped": len(dropped),
                    "seconds": round(self.last_seconds, 3),
                }
            },
        )
        return made

    def status(self) -> dict[str, Any]:
        """the versions published, the scopes written, and the last publish's seconds, as plain JSON values.

        :return: the values
        :rtype: dict[str, Any]
        """
        with self._lock:
            last = None if self._last is None else self._last[0]
        return {
            "last_version": last,
            "published": self.published,
            "scopes_written": self.scopes_written,
            "last_seconds": round(self.last_seconds, 3),
        }
