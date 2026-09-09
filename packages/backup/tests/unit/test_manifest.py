"""Manifest identity and round-trip — the durable half of a backup set."""

from __future__ import annotations

import json
from dataclasses import replace

from datetime import UTC, datetime
from uuid import uuid7

import pytest

from threetears.backup.manifest import BackupManifest, DatabaseFailure, DatabaseDump, TableCount, manifest_key


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
    """a version this reader does not know must not be guessed at.

    The number is deliberately far past anything real rather than "current + 1":
    this test named version 2 as the future to refuse, and then version 2 became
    the present, so the test failed while asserting nothing it meant.
    """
    raw = json.loads(_manifest().to_json())
    raw["version"] = 9999
    with pytest.raises(ValueError, match="unknown manifest version"):
        BackupManifest.from_json(json.dumps(raw).encode("utf-8"))


def test_a_version_one_manifest_still_reads_and_counts_as_complete() -> None:
    """sets written before partial backups existed are complete, truthfully.

    A version-1 writer aborted the whole set on any failure, so a v1 manifest
    that exists at all covered every database. Reading its absent
    `failed_databases` as "none" is the fact, not a default.
    """
    raw = json.loads(_manifest().to_json())
    raw["version"] = 1
    del raw["failed_databases"]

    manifest = BackupManifest.from_json(json.dumps(raw).encode("utf-8"))

    assert manifest.failed_databases == ()
    assert manifest.is_complete


def test_a_partial_set_reports_itself_incomplete_across_the_round_trip() -> None:
    """the manifest is what a restore reads, so the gap has to survive storage."""
    partial = replace(_manifest(), failed_databases=(DatabaseFailure(database="dipp", error="OSError: catalog gone"),))

    reread = BackupManifest.from_json(partial.to_json())

    assert not reread.is_complete
    assert reread.failed_databases[0].database == "dipp"
    assert "catalog gone" in reread.failed_databases[0].error


def test_manifest_key_lives_under_its_own_segment() -> None:
    backup_id = uuid7()
    key = manifest_key("backups", backup_id)
    assert key == f"backups/manifests/{backup_id.hex}.json.enc"


def test_naive_created_at_is_read_as_utc() -> None:
    raw = _manifest().to_json().replace(b"+00:00", b"")
    parsed = BackupManifest.from_json(raw)
    assert parsed.created_at.tzinfo is not None


def test_a_synchronized_inventory_survives_the_round_trip() -> None:
    """the flag decides how strictly a verifier may compare, so it has to reach the reader.

    Without this test, dropping the key from `to_json` keeps the whole suite green while every
    stored backup silently reads back as unsynchronized -- and every dry run quietly weakens
    from an exact comparison to a shortfall check, which is the failure that hides a loss.
    """
    manifest = _manifest()
    synchronized = replace(
        manifest,
        databases=tuple(replace(d, inventory_snapshot_consistent=True) for d in manifest.databases),
    )

    reread = BackupManifest.from_json(synchronized.to_json())

    assert [d.inventory_snapshot_consistent for d in reread.databases] == [True] * len(manifest.databases)


def test_a_manifest_written_before_the_flag_existed_reads_as_unsynchronized() -> None:
    """absent must mean the cautious answer, never the flattering one.

    Every backup taken before this field shipped WAS counted outside the dump's snapshot, so
    False is not a fallback here -- it is the truth about those sets.
    """
    raw = _manifest().to_json()
    assert b"inventory_snapshot_consistent" in raw

    stripped = json.loads(raw.decode("utf-8"))
    for dump in stripped["databases"]:
        del dump["inventory_snapshot_consistent"]

    reread = BackupManifest.from_json(json.dumps(stripped).encode("utf-8"))

    assert not any(d.inventory_snapshot_consistent for d in reread.databases)
