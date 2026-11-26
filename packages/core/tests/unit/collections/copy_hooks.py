"""A product's hooks the shared copy's tests hand its apply worker: module-level, so the worker imports them by name."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.collections.shared_copy import ROWS, AppliedTables, ChangePage

#: the derived table: each state's votes, from the results' parts a change leaves
TOTALS = "totals"
TOTALS_SCHEMA = pa.schema([pa.field("state", pa.string()), pa.field("votes", pa.int64())])

#: a derived table held as one part, as big as a test asks: a file no change's own tables foretell
BULK = "bulk"
BULK_SCHEMA = pa.schema([pa.field("id", pa.int64()), pa.field("value", pa.int64())])


@dataclass(frozen=True, slots=True)
class StateTotals:
    """derives each touched state's total votes from the results' parts as the change leaves them (one per state)."""

    def derive(self, applied: AppliedTables) -> Sequence[ChangePage]:
        """the totals of each results part the change leaves.

        :param applied: what the change did
        :ptype applied: AppliedTables
        :return: one rows page of totals
        :rtype: Sequence[ChangePage]
        """
        parts = applied.parts.get("results", {})
        states = sorted(parts)
        votes = [pc.sum(parts[state]["votes"]).as_py() or 0 for state in states]
        rows = pa.table({"state": pa.array(states, pa.string()), "votes": pa.array(votes, pa.int64())})
        return [ChangePage(ROWS, TOTALS, rows)] if states else []


@dataclass(frozen=True, slots=True)
class WaitFor:
    """holds each change in the apply worker until a file exists: a lane's change applying as long as a test wants.

    :ivar path: the file
    """

    path: str

    def derive(self, applied: AppliedTables) -> Sequence[ChangePage]:
        """wait for the file (60 s at most), then derive nothing.

        :param applied: what the change did (not read)
        :ptype applied: AppliedTables
        :return: no pages
        :rtype: Sequence[ChangePage]
        """
        deadline = time.monotonic() + 60
        while not os.path.exists(self.path) and time.monotonic() < deadline:
            time.sleep(0.01)
        return []


@dataclass(frozen=True, slots=True)
class DiesWhile:
    """kills the apply worker applying a change while a file exists; with ``once``, the dying worker removes it first.

    :ivar path: the file
    :ivar once: whether the first death removes the file
    """

    path: str
    once: bool

    def derive(self, applied: AppliedTables) -> Sequence[ChangePage]:
        """die while the file exists, else derive nothing.

        :param applied: what the change did (not read)
        :ptype applied: AppliedTables
        :return: no pages
        :rtype: Sequence[ChangePage]
        """
        if os.path.exists(self.path):
            if self.once:
                os.remove(self.path)
            os._exit(9)
        return []


@dataclass(frozen=True, slots=True)
class WaitThenBulk:
    """holds each change in the apply worker until a file exists (:class:`WaitFor`), then derives ``rows`` rows of
    :data:`BULK` (16 bytes each): one part bigger than the change's own tables say its result needs.

    :ivar path: the file
    :ivar rows: the derived rows
    """

    path: str
    rows: int

    def derive(self, applied: AppliedTables) -> Sequence[ChangePage]:
        """wait for the file, then derive the bulk table's rows.

        :param applied: what the change did
        :ptype applied: AppliedTables
        :return: one rows page of :data:`BULK`
        :rtype: Sequence[ChangePage]
        """
        WaitFor(self.path).derive(applied)
        ids = pa.array(np.arange(self.rows, dtype=np.int64))
        return [ChangePage(ROWS, BULK, pa.table({"id": ids, "value": ids}, schema=BULK_SCHEMA))]
