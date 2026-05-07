"""bind conflict policy enum shared by bind primitives and config.

the enum captures which side wins when L3 head-state and the on-disk
``bind_root`` disagree during a bind window. two scopes of disagreement
exist and the policy applies to both:

- **on enter** (before the watcher spawns): ``_seed_l3_from_disk``
  decides whether disk contents seed L3 unconditionally or only when L3
  is empty.
- **during the bind window** (via the live watcher):
  ``_handle_watch_batch`` decides whether externally-observed disk
  events propagate into L3 or are ignored in favor of the L3 head that
  bind already projected onto disk.

the two members differ in which side owns the authority:

- :attr:`L3_WINS` treats L3 as the source of truth. the enter phase
  imports from disk only when L3 is empty (the historical "seed from
  disk if empty" contract). the watcher imports only genuinely-new
  paths the agent would never have created itself; modifications and
  deletions observed on disk are discarded because the L3 copy is
  authoritative and will be re-projected onto disk via ``atomic_write``
  on the next bind enter.
- :attr:`DISK_WINS` treats disk as the source of truth. the enter
  phase walks disk unconditionally: every disk file becomes a ``create``
  or ``update`` row, every L3-only path (present in head-state, absent
  from disk) becomes a ``delete`` row. the watcher imports every
  ``added`` / ``modified`` / ``deleted`` event wholesale, matching the
  contract that shipped with the initial live-sync implementation.

the just-wrote guard (bind's own L3 -> disk sync echoing back through
the watcher) applies in both modes: the bind process always suppresses
its own round-trip writes.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "BindConflictPolicy",
]


class BindConflictPolicy(StrEnum):
    """
    policy governing L3 vs disk authority during :func:`bind`.

    :cvar L3_WINS: L3 is source of truth. disk seeds L3 only when empty.
        watcher imports only net-new paths; modifications and deletions
        on disk are ignored.
    :cvar DISK_WINS: disk is source of truth. disk clobbers L3 on enter
        (L3-only paths get delete journal rows). watcher imports every
        observed add / modify / delete event.
    """

    L3_WINS = "l3_wins"
    DISK_WINS = "disk_wins"
