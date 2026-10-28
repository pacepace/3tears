"""A store that records every read of results it serves: what a test of a paged walk of a run reads it through.

A walk of a run reads its keys once (a typed read that leaves out what grows with a cell), then its results by id a
page at a time; any other typed read of results hands out whole rows. The store tells the three apart, so a test can
hold a reader to no read past a page and no read of a run's rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from threetears.evals.kernel.run_pages import ResultKey
from packages.evals.tests.lease_support import ClockedStore

#: Fields of a stored result that grow with what its cell did, which a read of its run's keys leaves out. Every result
#: the fixtures store carries ``usage`` and ``rubric_scores``, so a whole read of a run always hands them out.
HEAVY_RESULT_FIELDS = frozenset({"usage", "rubric_scores", "world_events", "judge_reasoning", "kind_payload"})


class BatchRecordingStore(ClockedStore):
    """The engine's in-memory store, recording every read of results it serves and how many it hands out."""

    def __init__(self) -> None:
        super().__init__()
        #: ``(run ids the read named, whether it read keys and left out what grows with a cell, results handed out)``,
        #: one per typed read of results.
        self.typed_reads: list[tuple[tuple[str, ...], bool, int]] = []
        #: How many results each read by id handed out, in call order.
        self.id_reads: list[int] = []

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        rows = super().by_doc_type(
            doc_type,
            scope_id,
            order_by=order_by,
            descending=descending,
            limit=limit,
            exclude=exclude,
            field_in=field_in,
            **field_eq,
        )
        if doc_type == "eval_result":
            named = (
                *([field_eq["eval_run_id"]] if "eval_run_id" in field_eq else []),
                *(field_in or {}).get("eval_run_id", ()),
            )
            # A read of a run's keys leaves out what grows with a cell: a stored result always carries its usage rows.
            keys_only = bool(rows) and all(
                set(ResultKey._fields) <= set(row) and HEAVY_RESULT_FIELDS.isdisjoint(row) for row in rows
            )
            self.typed_reads.append((tuple(named), keys_only, len(rows)))
        return rows

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        rows = super().get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep)
        if doc_type == "eval_result":
            self.id_reads.append(len(rows))
        return rows

    def clear(self) -> None:
        self.typed_reads.clear()
        self.id_reads.clear()

    def key_reads_of(self, run_id: str) -> list[int]:
        """How many keys each read of ``run_id``'s result keys handed out."""
        return [rows for named, keys_only, rows in self.typed_reads if run_id in named and keys_only]

    def row_reads(self) -> list[int]:
        """How many whole results each typed read handed out: every one but a read of keys."""
        return [rows for _, keys_only, rows in self.typed_reads if not keys_only and rows]


__all__ = ["HEAVY_RESULT_FIELDS", "BatchRecordingStore"]
