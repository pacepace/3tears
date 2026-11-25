"""A product's hooks the shared copy's tests hand its apply worker: module-level, so the worker imports them by name."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.collections.shared_copy import ROWS, AppliedTables, ChangePage

#: the derived table: each state's votes, from the results' parts a change leaves
TOTALS = "totals"
TOTALS_SCHEMA = pa.schema([pa.field("state", pa.string()), pa.field("votes", pa.int64())])


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
