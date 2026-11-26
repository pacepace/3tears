"""the whys a table is reloaded whole without reading L3, each counted: ``collection_whole_reloads_total{table,why}``.

A copy that starts from the delta-files volume reloads a table from the writer's full on the volume, never from L3
(:meth:`~threetears.core.collections.copy_follower.CopyFollower._wait_for_reload`, ``_wait_for_recovery``). Its
start does so for every opted-in table, as does an operator's reset or a schema change of one. Every other reload is a
recovery: the copy lost the table's epoch log or a delta file, and the writer's recovered full stands in for the
changes it can no longer walk. The writer's lost-write backstop (:mod:`~threetears.core.collections.lost_writes`)
records whole changes of a dead writer's window again, which every copy then reloads.

Recovery reloads stay, but are never silent (Pace's ruling of 2026-11-27): each is counted here by table and why and
logged at WARNING with both, and a replay fails on any (:attr:`WholeReads.gated` of :data:`WHOLE_RELOADS`). A reset or
a schema change the replay itself caused is allowed, and named in its report.

**When a reload is counted.** When the copy starts waiting for the full it reloads from, not when the reload lands: a
reload still waiting at a replay's end is counted as well. Once a reload: a second lost file or a second reset joining
a wait already counted under the same why is the same reload; one of another kind (a lost file while a reset waits)
counts too.
"""

from __future__ import annotations

from typing import Final

from threetears.core.keyed import WholeReads

__all__ = ["LOST_FILE", "LOST_LOG", "LOST_WRITES", "WHOLE_RELOADS"]

#: the copy can no longer walk the table's epoch log from its mark (replaced, a record gone, aged out)
LOST_LOG: Final = "lost_log"

#: a delta file the copy waited for never comes (lost, or swept by a fold because a full covers it)
LOST_FILE: Final = "lost_file"

#: the writer's backstop recorded a whole change of a dead writer's window again
LOST_WRITES: Final = "lost_writes"

#: every whole reload with no L3 read, counted by table and why; only a start, a reset and a schema change are expected
WHOLE_RELOADS: Final = WholeReads(
    "collection_whole_reloads_total",
    subject="tables reloaded whole from the writer's full on the delta-files volume, or recorded whole again by the "
    "lost-write backstop",
    whys={
        "cold_start": "a copy's start from the volume: every opted-in table loaded from the writer's full",
        "reset": "an operator's reset of the table: reloaded from the writer's full past the reset",
        "schema_change": "a table dropped or recreated: reloaded from the writer's full past the change",
        LOST_LOG: "the copy cannot walk the table's epoch log from its mark: reloaded from the writer's recovered full",
        LOST_FILE: "a delta file the copy waited for never comes: reloaded from a full holding its change",
        LOST_WRITES: "a dead writer's window held a whole change (a reset, a run deleted): the backstop recorded it again",
        "held_past_cap": "a table waiting for a delta file held more changes than its cap: reloaded from the next full",
        "partition_refused": "a record naming no delta file (an older writer's) of an opted-in table: reloaded from "
        "the next full",
    },
    expected={"cold_start", "reset", "schema_change"},
    said="reloaded whole",
)
