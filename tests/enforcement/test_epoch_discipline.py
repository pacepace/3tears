"""enforcement: no code in 3tears bypasses epochs or degrades quietly.

An epoch IS its change set: the writer names the rows it changed, and every pod applies exactly
those. Five rules hold that, one gate each, over every package's ``src``:

- ``whole-drop``: no dropping, clearing or reloading a whole table, scope or L1 copy, except where
  a real whole-table epoch record is applied;
- ``l2-swallow``: an L2 failure re-raises or goes through ``degrade_l2``, never a log line and a
  fall-through;
- ``ttl-safety-net``: no expiry standing in for an epoch on a cache or bucket;
- ``epoch-bypass``: no raw L3 write or collection L2 put outside the collection write path or an
  ``announce_change``;
- ``whole-reread``: no whole table or scope read from L3 to recover, warm or prove.

The detectors and the ledger grammar live in :mod:`threetears.enforcement.epoch_discipline`. Each
rule's allowlist (``_epoch_<rule>_allowlist.txt`` beside this file) lists today's violations, each
with the fix that removes it, pinned by count; ``_epoch_waivers.txt`` lists the sites a detector
matches that are not what the rule forbids, each with its reason. A new site fails, and so does a
line whose site shrank or went, so both lists only shrink.

The sanctioned whole-table-epoch apply site, ``CollectionRegistry.apply_epoch_change``, is not a
primitive: its one ``drop_table`` (for a record naming the whole table) is a waiver pinned by count,
so a second drop there fails the gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.common import find_local_src_roots
from threetears.enforcement.epoch_discipline import (
    RULES,
    EpochDisciplineConfig,
    epoch_discipline_findings,
    epoch_violation_counts,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE = "packages/core/src/threetears/core"

#: the functions that IMPLEMENT what each rule polices; what they do inside is the operation itself.
_PRIMITIVES: dict[str, frozenset[str]] = {
    "whole-drop": frozenset(
        {
            # the drop itself; every caller is what the rule holds
            f"{_CORE}/collections/registry.py::CollectionRegistry.drop_table",
            # the L3 backend's truncate; its callers announce or are listed
            f"{_CORE}/backends/sql.py::SqlL3Backend.truncate",
            # the DuckDB L1 backend: its partition swaps and staged changes are the primitives callers reach
            f"{_CORE}/cache/duckdb.py::DuckDBBackend",
        }
    ),
    "epoch-bypass": frozenset(
        {
            # the collection write path: every save, delete and L2 put the rule routes writes into, method by
            # method, so a raw write added anywhere else in BaseCollection is a finding
            f"{_CORE}/collections/base.py::BaseCollection.save_entity",
            f"{_CORE}/collections/base.py::BaseCollection.delete",
            f"{_CORE}/collections/base.py::BaseCollection._write_l2_at",
            f"{_CORE}/collections/base.py::BaseCollection._cache_committed_row",
            f"{_CORE}/collections/base.py::BaseCollection._async_propagate_write",
            f"{_CORE}/collections/base.py::BaseCollection._drop_unread_row",
            f"{_CORE}/collections/base.py::BaseCollection._clear_marks",
            f"{_CORE}/collections/base.py::BaseCollection._withdraw_from_caches",
            f"{_CORE}/collections/base.py::BaseCollection.invalidate_cache_many",
            # the row's L2 put, behind save_entity
            f"{_CORE}/collections/base.py::BaseCollection._save_to_l2",
            # the row's L2 delete, behind delete
            f"{_CORE}/collections/base.py::BaseCollection._delete_from_l2",
            # a revision-fenced L2 delete on the write path
            f"{_CORE}/collections/base.py::BaseCollection._delete_l2_at",
            # seeding L2 from a read, revision-fenced, the read path's own fill
            f"{_CORE}/collections/base.py::BaseCollection._seed_l2",
            # an L2-first commit: its epoch is appended with the row
            f"{_CORE}/collections/base.py::BaseCollection._l2_first_apply",
            # the receiver's eviction of a key an invalidation named
            f"{_CORE}/collections/base.py::BaseCollection.delete_l2_entry",
            # the L1+L2-only collection's compare-and-swap write path
            f"{_CORE}/collections/base.py::BaseCollection.l2_cas_mutate",
            # taking back a won compare-and-swap whose persist failed
            f"{_CORE}/collections/base.py::BaseCollection._withdraw_unpersisted_cas",
            # the L2 write generation: the epoch machinery itself
            f"{_CORE}/collections/base.py::BaseCollection._advance_l2_generation",
            # the L2 write generation: the epoch machinery itself
            f"{_CORE}/collections/base.py::BaseCollection._advance_l2_generation_key",
            # the L2 write generation: the epoch machinery itself
            f"{_CORE}/collections/base.py::BaseCollection.l2_write_generation",
            # the schema-backed collection's keyed bulk writes, which name every row they change
            f"{_CORE}/collections/schema_backed.py::SchemaBackedCollection.save_rows",
            f"{_CORE}/collections/schema_backed.py::SchemaBackedCollection.delete_rows",
            f"{_CORE}/collections/schema_backed.py::SchemaBackedCollection._copy_rows",
            # the backends: the SQL they run IS the write a collection asked for
            f"{_CORE}/backends/sql.py::SqlL3Backend",
            f"{_CORE}/backends/git/repos.py::RoutedGitStore",
            f"{_CORE}/cache/duckdb.py::DuckDBBackend",
            f"{_CORE}/cache/duckdb.py::_insert_through_arrow",
            f"{_CORE}/cache/sqlite.py::SQLiteBackend",
            # the epoch machinery's own ledgers: the write marks, the scope epochs and the source fence
            f"{_CORE}/collections/write_marks.py::WriteMarkReconciler",
            f"{_CORE}/collections/scope_epochs.py::ScopeEpochs",
            f"{_CORE}/coordination/source_token.py::SourceTokenFence",
            # the change-note ring in L3: its ring row, its note tables and its triggers (installed, emptied)
            f"{_CORE}/collections/change_notes.py::ChangeNoteKeeper",
            # the migration runner's own version ledger, _schema_migrations
            f"{_CORE}/data/migrations/runner.py::MigrationRunner",
            # the epoch client's own counter table, config_epochs: it IS the epoch
            "packages/epoch/src/threetears/epoch/client.py::EpochClient",
        }
    ),
    "whole-reread": frozenset(
        {
            # the L3 backend's scan and the L1 backend's own reads of the copy it holds
            f"{_CORE}/backends/sql.py::SqlL3Backend",
            f"{_CORE}/cache/duckdb.py::DuckDBBackend",
        }
    ),
    "ttl-safety-net": frozenset(
        {
            # the KV layer that carries a caller's expiry into the bucket config; the callers are held
            "packages/nats/src/threetears/nats/kv.py::build_kv_stream_config",
            "packages/nats/src/threetears/nats/kv.py::_bind_kv_handle",
        }
    ),
}

_CONFIG = EpochDisciplineConfig(
    repo_root=_REPO_ROOT,
    src_roots=find_local_src_roots(_REPO_ROOT),
    ledger_dir=Path(__file__).parent,
    primitives=_PRIMITIVES,
    excluded_prefixes=(
        # the gates themselves: their parse caches hold no data
        "packages/enforcement/",
        # test doubles shipped under src; no deployed process runs them
        f"{_CORE}/testing/",
    ),
    minimum_files_scanned=900,
)


@pytest.mark.parametrize("rule", RULES)
def test_epoch_discipline(rule: str) -> None:
    """no new violation of ``rule``, and no ledger line its code has outgrown.

    :param rule: the rule
    :ptype rule: str
    :return: nothing
    :rtype: None
    """
    findings = epoch_discipline_findings(_CONFIG, rule)
    assert not findings, f"{len(findings)} {rule} finding(s):\n" + "\n".join(findings)


def test_every_detector_sees_the_code() -> None:
    """each rule's detector finds sites in 3tears today, so a green gate is a walk that read the code.

    When a rule's last site is fixed this floor drops to zero for it, with the change that fixes it.

    :return: nothing
    :rtype: None
    """
    counts = epoch_violation_counts(_CONFIG)
    assert all(counts[rule] > 0 for rule in RULES), f"a detector found nothing: {counts}"
