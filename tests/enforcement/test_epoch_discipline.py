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

The sanctioned whole-table-epoch apply site, ``CollectionRegistry.apply_epoch_change``, is a
:data:`_PRIMITIVES` entry under ``whole-drop``.
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
            # the DuckDB L1 backend: its replace_all and partition swaps are the primitives callers reach
            f"{_CORE}/cache/duckdb.py::DuckDBBackend",
            # the sanctioned whole-table-epoch apply: it drops a table only for a record naming the whole table
            f"{_CORE}/collections/registry.py::CollectionRegistry.apply_epoch_change",
        }
    ),
    "epoch-bypass": frozenset(
        {
            # the collection write path: every save, delete and L2 put the rule routes writes into
            f"{_CORE}/collections/base.py::BaseCollection",
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
            # the L2-first epoch log in L3: the saved record of each epoch, the ledger the saver walks
            f"{_CORE}/collections/l2_first.py::SqlEpochLogStore",
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
