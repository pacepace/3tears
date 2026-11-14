"""the no-whole-reads walk 3tears' gates share: the no-whole-reads gate and the use-the-helper gate read the same
modules, declarations and transient relations."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.common import find_local_src_roots
from threetears.enforcement.no_whole_reads import NoWholeReadsConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE = "packages/core/src/threetears/core"

NO_WHOLE_READS_CONFIG = NoWholeReadsConfig(
    repo_root=_REPO_ROOT,
    src_roots=find_local_src_roots(_REPO_ROOT),
    ledger_dir=Path(__file__).parent,
    excluded_prefixes=(
        # the gates themselves: they read source files, no table
        "packages/enforcement/",
        # test doubles shipped under src; no deployed process runs them
        f"{_CORE}/testing/",
    ),
    # a copy follower's stages (STAGE_PREFIX) and its applied mark: the rows a change named, beside the copy
    transient_relations=r"^__stage",
    # the collections, caches and follower: load code, where even one key's count or fingerprint re-verifies by
    # scanning what the epoch already said
    verify_prefixes=(f"{_CORE}/collections/", f"{_CORE}/cache/"),
    minimum_files_scanned=900,
    # the hub reads 3tears' per-agent tables through their declarations (the admin conversations listing): an index
    # purpose may name its read, which the hub's own gate keys by that index
    consumer_purposes=("src/aibots/hub/",),
    # 3tears' declared helper modules, judged at their callers: each writes the SQL over a relation its caller names
    # at run time (a collection's table, a copy's table, a warehouse table, the change-note ring a store owns), so
    # use-the-helper leaves those statements to the callers, as it does 3tears' keyed helpers'; a table one names
    # itself is still judged where it is written, and no_whole_reads still judges every statement in them
    caller_judged_modules=(
        # the L3 storage layer: the backend and its statement builders, over the table each collection names
        f"{_CORE}/backends/",
        # a collection's base and declared-table writer, over the table its subclass declares
        f"{_CORE}/collections/base.py",
        f"{_CORE}/collections/declared_table.py",
        # the change-note ring and the scope epochs: the epoch machinery every keyed read is told its keys by
        f"{_CORE}/collections/change_notes.py",
        f"{_CORE}/collections/scope_epochs.py",
        # the copy machinery: a follower, its checkpoint and a complete copy, over the copied tables it is handed
        f"{_CORE}/collections/copy_follower.py",
        f"{_CORE}/collections/copy_checkpoint.py",
        f"{_CORE}/collections/complete_copy.py",
        # a table's fingerprint, over the table its caller names
        f"{_CORE}/fingerprint.py",
        # the warehouse readers: pages, partitions, exports and coverage over the table a datasource names
        "packages/datasources/src/threetears/datasources/partitioned_read.py",
        "packages/datasources/src/threetears/datasources/query_client.py",
        "packages/datasources/src/threetears/datasources/export_read.py",
        "packages/datasources/src/threetears/datasources/drivers/base.py",
        # a bounded look (count to a cap, a few rows) over the warehouse table a binding names
        "packages/datasources/src/threetears/datasources/bounded_look.py",
    ),
    # every shipped migration: its text is never edited again, so an index in one is declared in the ledger
    # (_index_declarations.txt, written by python -m threetears.enforcement.no_whole_reads.index_purposes)
    frozen_migrations=("packages/*/migrations/v*.py", "packages/scrape/src/threetears/scrape/migrations.py"),
    # memory's read filter, written by its clause builders: the use-the-helper rewrite turns a statement interpolating
    # one's answer into where=[*clause_guards(<builder>, ...)], proven bound to tests/integration/clause_subjects
    clause_builders={
        f"threetears.agent.memory.{builder}": "tests.integration.clause_subjects:read_filters"
        for builder in (
            "read_filter:filter_clause",
            "collections:_chunk_parent_clause",
            "collections:_media_parent_clause",
        )
    },
    # the clause builders deciding on a parent table's rows: their proof seeds those rows beside the read's own
    clause_parents={
        "threetears.agent.memory.collections:_chunk_parent_clause": "tests.integration.clause_subjects:chunk_parents",
        "threetears.agent.memory.collections:_media_parent_clause": "tests.integration.clause_subjects:media_parents",
    },
)
