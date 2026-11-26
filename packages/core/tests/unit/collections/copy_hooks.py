"""A product's hooks the shared copy's tests hand its apply worker: module-level, so the worker imports them by name."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.collections.copy_parts import PartRule
from threetears.core.collections.shared_copy import ROWS, AppliedTables, ChangePage, DerivedPages, LetGoRows

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
class LetsGo:
    """lets go, after each change, of the rows of ``table`` equal to one of ``values`` on ``columns`` (a retention's
    let-go, named by the test).

    :ivar table: the table
    :ivar columns: the columns matched
    :ivar values: the rows let go, a tuple of values a row, in ``columns``' order
    """

    table: str
    columns: tuple[str, ...]
    values: tuple[tuple[str | int, ...], ...]

    def derive(self, applied: AppliedTables) -> DerivedPages:
        """the rows let go, nothing derived.

        :param applied: what the change did (not read)
        :ptype applied: AppliedTables
        :return: the let-go rows
        :rtype: DerivedPages
        """
        schema = applied.schemas[self.table]
        rows = pa.table(
            {
                column: pa.array([row[at] for row in self.values], schema.field(column).type)
                for at, column in enumerate(self.columns)
            }
        )
        return DerivedPages(let_go=[LetGoRows(self.table, rows)])


#: the settle-inputs tests' derived tables: the rows a change deleted as they were held, and each touched state's rows
#: of an unchanged table read from the version with the change applied
GONE = "gone"
SEEN = "seen"
SEEN_SCHEMA = pa.schema([pa.field("state", pa.string()), pa.field("rows", pa.int64())])


@dataclass(frozen=True, slots=True)
class SettleInputs:
    """derives :data:`GONE` (``results``' deleted rows as the version held them, every column) and :data:`SEEN` (each
    touched state's rows of ``lookup``, a table the change did not carry, read from the applied version)."""

    def derive(self, applied: AppliedTables) -> Sequence[ChangePage]:
        """the two derived tables' pages.

        :param applied: what the change did
        :ptype applied: AppliedTables
        :return: their pages
        :rtype: Sequence[ChangePage]
        """
        pages: list[ChangePage] = []
        gone = applied.deleted_rows.get("results")
        if gone is not None:
            pages.append(ChangePage(ROWS, GONE, gone))
        if applied.touched:
            lookup = applied.table("lookup", scopes=applied.touched)
            counts = lookup.group_by("state").aggregate([("state", "count")])
            rows = pa.table({"state": counts["state"], "rows": pc.cast(counts["state_count"], pa.int64())})
            pages.append(ChangePage(ROWS, SEEN, rows))
        return pages


@dataclass(frozen=True, slots=True)
class WholeTotals:
    """derives :data:`TOTALS` of each touched state on a change, and of every state at a whole settle, which also lets
    go of ``results``' rows of no votes."""

    def derive(self, applied: AppliedTables) -> Sequence[ChangePage]:
        """the touched states' totals.

        :param applied: what the change did
        :ptype applied: AppliedTables
        :return: one rows page of totals
        :rtype: Sequence[ChangePage]
        """
        return [ChangePage(ROWS, TOTALS, _totals(applied.table("results", scopes=applied.touched)))]

    def derive_whole(self, applied: AppliedTables) -> DerivedPages:
        """every state's totals, and the rows of no votes let go.

        :param applied: the version read whole
        :ptype applied: AppliedTables
        :return: the totals, replacing the table whole, and the let-go
        :rtype: DerivedPages
        """
        results = applied.table("results")
        zero = results.filter(pc.equal(results["votes"], 0)).select(["state", "office_key", "candidate_id"])
        return DerivedPages(
            pages=[ChangePage(ROWS, TOTALS, _totals(results))],
            whole=frozenset({TOTALS}),
            let_go=[LetGoRows("results", zero)],
        )


def _totals(results: pa.Table) -> pa.Table:
    summed = results.group_by("state").aggregate([("votes", "sum")])
    return pa.table({"state": summed["state"], "votes": pc.cast(summed["votes_sum"], pa.int64())}, schema=TOTALS_SCHEMA)


@dataclass(frozen=True, slots=True)
class OlderThan:
    """merges a scope's single-run parts whose runs are of ``model`` and more than ``seconds`` behind its newest run,
    by the runs' times the copy's held runs give (a retention merging by age, as reports' ``LetGo`` does).

    :ivar model: the model
    :ivar seconds: the age
    """

    model: str
    seconds: float

    def due(
        self, rule: PartRule, parts: Sequence[tuple[int, int, int, str]], runs: pa.Table | None
    ) -> tuple[int, Sequence[str]] | None:
        """the old runs' parts, two at least.

        :param rule: the table's rule (not read)
        :ptype rule: PartRule
        :param parts: one scope's parts
        :ptype parts: Sequence[tuple[int, int, int, str]]
        :param runs: the scope's held runs
        :ptype runs: pa.Table | None
        :return: level 0 and the group, or None
        :rtype: tuple[int, Sequence[str]] | None
        """
        if runs is None or not runs.num_rows:
            return None
        newest = pc.max(runs["at"]).as_py()
        old = runs.filter(pc.and_(pc.equal(runs["model"], self.model), pc.less(runs["at"], newest - self.seconds)))[
            "run"
        ]
        olds = frozenset(old.to_pylist())
        group = [part for first, last, level, part in parts if level == 0 and first == last and first in olds]
        return (0, group) if len(group) >= 2 else None


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
