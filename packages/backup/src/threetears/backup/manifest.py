"""Backup manifests — the durable identity and inventory of a backup set.

A backup used to be one opaque encrypted blob whose only metadata was string-packed into its
object key. That shape has no stable identity (nothing to address a restore by), no inventory
(nothing to check coverage against), and no record of which driver wrote it (so a restore had to
guess). The manifest fixes all three: one small JSON document per backup set, written through the
same encrypted store as the dumps themselves, holding the set's uuid7 id, its driver, every
database dumped with a per-table row inventory, and the key of the globals dump when one was taken.

The manifest is written LAST, after every dump it names has landed — so a manifest's existence
asserts a complete set, and a crashed backup leaves dangling dumps but never a manifest that lies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

__all__ = [
    "BackupManifest",
    "DatabaseDump",
    "TableCount",
    "manifest_key",
]

#: manifests live under their own prefix segment so a listing of dumps never mistakes one for a dump.
_MANIFEST_SEGMENT = "manifests"


def manifest_key(prefix: str, backup_id: UUID) -> str:
    """The object key a manifest is stored under.

    :param prefix: the engine's configured key prefix.
    :param backup_id: the backup set's uuid7 id.
    :return: ``<prefix>/manifests/<hex>.json.enc``.
    """
    return f"{prefix}/{_MANIFEST_SEGMENT}/{backup_id.hex}.json.enc"


@dataclass(frozen=True, slots=True)
class TableCount:
    """One table's row count at dump time — the unit of the coverage inventory."""

    schema: str
    table: str
    row_count: int


@dataclass(frozen=True, slots=True)
class DatabaseDump:
    """One database's dump within a backup set."""

    database: str
    key: str
    size_bytes: int
    #: SHA-256 of the PLAINTEXT dump stream (pre-encryption), hex — proves a decrypted restore
    #: stream is byte-identical to what the dump tool produced.
    sha256: str
    tables: tuple[TableCount, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BackupManifest:
    """The identity and inventory of one backup set.

    :param backup_id: uuid7 — stable across every listing, the address every restore uses.
    :param created_at: when the set was taken (UTC).
    :param driver: the dump driver's name (``postgres`` / ``yugabyte``) — restores construct the
        SAME driver from this rather than trusting whatever the restoring process was built with.
    :param databases: every database dumped, with its inventory.
    :param globals_key: the key of the cluster-globals dump (roles, grants), or ``None`` when the
        set is a single-database backup that took none.
    """

    backup_id: UUID
    created_at: datetime
    driver: str
    databases: tuple[DatabaseDump, ...]
    globals_key: str | None = None

    @property
    def total_size_bytes(self) -> int:
        """Sum of every database dump's stored size."""
        return sum(dump.size_bytes for dump in self.databases)

    @property
    def table_total(self) -> int:
        """Total table count across the set — the number a coverage check compares against."""
        return sum(len(dump.tables) for dump in self.databases)

    def to_json(self) -> bytes:
        """Serialize for storage (stable field order, UTF-8)."""
        payload = {
            "version": 1,
            "backup_id": str(self.backup_id),
            "created_at": self.created_at.isoformat(),
            "driver": self.driver,
            "globals_key": self.globals_key,
            "databases": [
                {
                    "database": dump.database,
                    "key": dump.key,
                    "size_bytes": dump.size_bytes,
                    "sha256": dump.sha256,
                    "tables": [{"schema": t.schema, "table": t.table, "row_count": t.row_count} for t in dump.tables],
                }
                for dump in self.databases
            ],
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> BackupManifest:
        """Parse a stored manifest.

        :raises ValueError: on a version this reader does not understand — refusing beats
            silently misreading a future shape.
        """
        payload = json.loads(raw.decode("utf-8"))
        version = payload.get("version")
        if version != 1:
            raise ValueError(f"unknown manifest version {version!r}")
        created = datetime.fromisoformat(payload["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return cls(
            backup_id=UUID(payload["backup_id"]),
            created_at=created,
            driver=payload["driver"],
            globals_key=payload.get("globals_key"),
            databases=tuple(
                DatabaseDump(
                    database=dump["database"],
                    key=dump["key"],
                    size_bytes=dump["size_bytes"],
                    sha256=dump["sha256"],
                    tables=tuple(
                        TableCount(schema=t["schema"], table=t["table"], row_count=t["row_count"])
                        for t in dump.get("tables", ())
                    ),
                )
                for dump in payload["databases"]
            ),
        )
