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
    # every shipped migration: its text is never edited again, so an index in one is declared in the ledger
    # (_index_declarations.txt, written by python -m threetears.enforcement.no_whole_reads.index_purposes)
    frozen_migrations=("packages/*/migrations/v*.py", "packages/scrape/src/threetears/scrape/migrations.py"),
)
