"""configuration for the epoch-discipline enforcement domain.

The domain holds five rules, each one a way code can step around 3tears epochs or degrade
quietly. An epoch IS its change set: the writer names the rows it changed, every pod applies
exactly those, and nothing reloads a whole table to be safe.

- ``whole-drop``: nothing drops, clears or reloads a whole cached table, scope or L1 copy --
  ``drop_table``, ``drop_cached_table``, ``drop_for_table``, ``invalidate_all``, ``.clear()`` on a
  cache, ``TRUNCATE`` or an unqualified ``DELETE FROM`` -- except where a real whole-table epoch
  record is applied, or in a function that announces the whole-table change it makes.
- ``l2-swallow``: an ``except`` catching a KV / object-store / JetStream error either re-raises or
  routes through ``degrade_l2``; a log line and a fall-through to L3 is a silent degrade.
- ``ttl-safety-net``: no ``ttl`` / ``max_age`` / expiry stands in for an epoch on a cache or a
  bucket. A value that IS the data's own lifetime (a lease, a nonce, a session) takes a waiver
  with its reason.
- ``epoch-bypass``: no raw ``INSERT`` / ``UPDATE`` / ``DELETE`` / ``TRUNCATE`` / ``COPY`` through a
  backend's ``execute``, no backend bulk write, and no direct put into a collection's L2 bucket,
  outside the collection write path, a ``bypassing_write`` block, or a function that calls
  ``announce_change``.
- ``whole-reread``: no whole-table read from L3 (a ``SELECT`` with no ``WHERE`` and no ``LIMIT``),
  and no L3 read inside a function whose name says it rebuilds, recovers, reloads, warms or
  re-proves.

Each repo keeps one allowlist per rule of the violations it still has, each tagged with the fix
that removes it, and one waiver file for the sites where a detector's shape is legitimately not
the thing the rule forbids. Both are pinned by count in both directions: a new site fails, and a
line whose site shrank or vanished fails until it is lowered or deleted, so the lists only shrink.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__all__ = [
    "RULES",
    "RULE_ADVICE",
    "RULE_EPOCH_BYPASS",
    "RULE_L2_SWALLOW",
    "RULE_TTL_SAFETY_NET",
    "RULE_WHOLE_DROP",
    "RULE_WHOLE_REREAD",
    "EpochDisciplineConfig",
    "allowlist_path_for",
]

RULE_WHOLE_DROP: Final[str] = "whole-drop"
RULE_L2_SWALLOW: Final[str] = "l2-swallow"
RULE_TTL_SAFETY_NET: Final[str] = "ttl-safety-net"
RULE_EPOCH_BYPASS: Final[str] = "epoch-bypass"
RULE_WHOLE_REREAD: Final[str] = "whole-reread"

#: every rule, in the order the gates report them.
RULES: Final[tuple[str, ...]] = (
    RULE_WHOLE_DROP,
    RULE_L2_SWALLOW,
    RULE_TTL_SAFETY_NET,
    RULE_EPOCH_BYPASS,
    RULE_WHOLE_REREAD,
)

#: what a NEW violation of each rule is told to do instead.
RULE_ADVICE: Final[Mapping[str, str]] = {
    RULE_WHOLE_DROP: (
        "apply the epoch's change set (the rows the writer named) instead of dropping the whole table; "
        "a deliberate whole-table change calls announce_change(table, whole_table=True) in the same function"
    ),
    RULE_L2_SWALLOW: (
        "route the failure through degrade_l2(exc, table=..., op=..., what=...) so it is counted and "
        "classified, or re-raise it"
    ),
    RULE_TTL_SAFETY_NET: (
        "drop the expiry and let epochs carry the change; if the value is the data's own lifetime "
        "(a lease, a nonce, a session), add a waiver line naming that lifetime"
    ),
    RULE_EPOCH_BYPASS: (
        "write through the collection (save_entity / bypassing_write), or call "
        "announce_change(table, rows) in the same function so every pod learns which rows changed"
    ),
    RULE_WHOLE_REREAD: (
        "catch up by epochs (fetch mine+1..head and apply their rows) instead of re-reading the whole "
        "table or scope from L3"
    ),
}


#: the store hooks :class:`~threetears.core.collections.base.BaseCollection`'s write path calls.
_DEFAULT_WRITE_PATH_HOOKS: Final[frozenset[str]] = frozenset(
    {
        "save_to_store",
        "save_many_to_store",
        "delete_from_store",
        "save_ordered_to_store",
        "persist_to_store",
        "persist_to_postgres",
    }
)


def allowlist_path_for(ledger_dir: Path, rule: str) -> Path:
    """the allowlist file of ``rule`` in a repo's ledger directory.

    :param ledger_dir: the directory that holds the repo's epoch-discipline ledgers
    :ptype ledger_dir: Path
    :param rule: one of :data:`RULES`
    :ptype rule: str
    :return: ``<ledger_dir>/_epoch_<rule with underscores>_allowlist.txt``
    :rtype: Path
    """
    return ledger_dir / f"_epoch_{rule.replace('-', '_')}_allowlist.txt"


@dataclass(frozen=True)
class EpochDisciplineConfig:
    """per-repo config for the epoch-discipline domain.

    :ivar repo_root: absolute path of the repo root; every path in a finding or a ledger is
        relative to it
    :ivar src_roots: production source trees to walk
    :ivar ledger_dir: directory holding one allowlist per rule (:func:`allowlist_path_for`) and
        :attr:`waivers_name`
    :ivar waivers_name: file name of the waiver ledger inside :attr:`ledger_dir`
    :ivar primitives: rule onto the functions that IMPLEMENT the operation the rule polices, each
        ``<repo-relative path>::<qualified name>``; a class name covers its methods. Each must
        name a function or class the walk finds, so a stale entry is a finding
    :ivar excluded_prefixes: repo-relative posix directory prefixes, each ending ``/``, left
        unwalked; each must match at least one module
    :ivar write_path_hooks: method names a collection's write path calls to reach its store; a raw
        write inside one is the write path, not a bypass of it
    :ivar minimum_files_scanned: the walk must reach at least this many modules, or it reports
        nothing about anything
    """

    repo_root: Path
    src_roots: tuple[Path, ...]
    ledger_dir: Path
    waivers_name: str = "_epoch_waivers.txt"
    primitives: Mapping[str, frozenset[str]] = field(default_factory=dict)
    excluded_prefixes: tuple[str, ...] = ()
    write_path_hooks: frozenset[str] = _DEFAULT_WRITE_PATH_HOOKS
    minimum_files_scanned: int = 1
