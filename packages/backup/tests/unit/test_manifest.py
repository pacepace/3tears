"""Manifest identity and round-trip — the durable half of a backup set."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid7

import pytest

from threetears.backup.manifest import BackupManifest, DatabaseDump, TableCount, manifest_key


def _manifest() -> BackupManifest:
    return BackupManifest(
        backup_id=uuid7(),
        created_at=datetime(2026, 9, 1, 4, 0, tzinfo=UTC),
        driver="yugabyte",
        globals_key="backups/2026/09/01/x/globals.sql.gz.enc",
        databases=(
            DatabaseDump(
                database="fourteenaibots_hub",
                key="backups/2026/09/01/x/fourteenaibots_hub.yugabyte.dump.gz.enc",
                size_bytes=1234,
                sha256="ab" * 32,
                tables=(
                    TableCount(schema="public", table="agents", row_count=7),
                    TableCount(schema="agent_ab12", table="notes", row_count=3),
                ),
            ),
            DatabaseDump(
                database="dipp",
                key="backups/2026/09/01/x/dipp.yugabyte.dump.gz.enc",
                size_bytes=99,
                sha256="cd" * 32,
            ),
        ),
    )


def test_round_trips_through_json_identically() -> None:
    original = _manifest()
    assert BackupManifest.from_json(original.to_json()) == original


def test_totals_sum_across_the_set() -> None:
    manifest = _manifest()
    assert manifest.total_size_bytes == 1333
    assert manifest.table_total == 2


def test_a_future_version_is_refused_rather_than_misread() -> None:
    raw = _manifest().to_json().replace(b'"version": 1', b'"version": 2')
    with pytest.raises(ValueError, match="unknown manifest version"):
        BackupManifest.from_json(raw)


def test_manifest_key_lives_under_its_own_segment() -> None:
    backup_id = uuid7()
    key = manifest_key("backups", backup_id)
    assert key == f"backups/manifests/{backup_id.hex}.json.enc"


def test_naive_created_at_is_read_as_utc() -> None:
    raw = _manifest().to_json().replace(b"+00:00", b"")
    parsed = BackupManifest.from_json(raw)
    assert parsed.created_at.tzinfo is not None
