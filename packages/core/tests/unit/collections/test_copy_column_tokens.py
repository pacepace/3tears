"""The copy's apply worker stamps each part it writes with its column tokens (``shared_arrow.stamped``) against the
base version's same part: a vote-only change keeps the token of every static column and moves the counts' alone, on a
scope's own part, on a pack of small scopes, and on a tier a merge wrote (reports R-1: its page memo skips a statement
reading no column whose token moved, ``build_memo``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from threetears.core.collections.copy_parts import PartRule, is_pack
from threetears.core.collections.shared_copy import ROWS, ChangePage, SharedCopyBackend, held_parts
from threetears.core.shared_arrow import FileTokens, SharedArrowStore, file_tokens

SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("office_key", pa.string()),
        pa.field("candidate_id", pa.int64()),
        pa.field("votes", pa.int64()),
    ]
)
RUN_SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("run_id", pa.int64()),
        pa.field("candidate_id", pa.int64()),
        pa.field("votes", pa.int64()),
    ]
)
BY_STATE = PartRule("results", ("state", "office_key", "candidate_id"), scope_column="state")
WHOLE = PartRule("results", ("state", "office_key", "candidate_id"))
BY_RUN = PartRule("runs", ("state", "run_id", "candidate_id"), scope_column="state", run_column="run_id")

#: the columns a vote-only change leaves as they were
_STATIC = ("state", "office_key", "candidate_id")


def _rows(state: str, count: int, votes: int) -> pa.Table:
    """``count`` rows of one state, over a few races, every count ``votes``."""
    return pa.table(
        {
            "state": pa.array([state] * count, pa.string()),
            "office_key": pa.array([f"{state}-{i % 3}" for i in range(count)], pa.string()),
            "candidate_id": pa.array(np.arange(count), pa.int64()),
            "votes": pa.array(np.full(count, votes), pa.int64()),
        },
        schema=SCHEMA,
    )


def _run_rows(state: str, run: int, count: int, votes: int) -> pa.Table:
    return pa.table(
        {
            "state": pa.array([state] * count, pa.string()),
            "run_id": pa.array([run] * count, pa.int64()),
            "candidate_id": pa.array(np.arange(count), pa.int64()),
            "votes": pa.array(np.full(count, votes), pa.int64()),
        },
        schema=RUN_SCHEMA,
    )


def _tokens(store: SharedArrowStore, table: str, part: str) -> FileTokens:
    """the tokens of a part of the current version's ``table``; never none."""
    current = store.current
    assert current is not None
    tokens = file_tokens(held_parts(store.root, current.name, table)[part])
    assert tokens is not None, f"{table}@{part} holds no column tokens"
    return tokens


def _vote_only(before: FileTokens, after: FileTokens, static: tuple[str, ...]) -> None:
    """every static column's token and the row token kept, the counts' moved."""
    assert after.rows == before.rows
    assert {c: after.columns[c] for c in static} == {c: before.columns[c] for c in static}
    assert after.columns["votes"] != before.columns["votes"]


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    return SharedArrowStore(tmp_path / "shm", max_bytes=1 << 30)


@pytest.fixture
async def by_state(store: SharedArrowStore) -> AsyncIterator[SharedCopyBackend]:
    backend = SharedCopyBackend(store, {"results": SCHEMA}, {"results": BY_STATE}, pack_below=0)
    yield backend
    await backend.close()


class TestAVoteOnlyChangeKeepsStaticTokens:
    async def test_on_a_scopes_own_part(self, store: SharedArrowStore, by_state: SharedCopyBackend) -> None:
        """Flips if the apply writes a part unstamped: VA's part rewritten with some counts moved keeps the token of
        every column but the counts; a part the change never touched keeps its own file, tokens and all."""
        rows = pa.concat_tables([_rows(state, 50, 1) for state in ("AZ", "GA", "VA")])
        await by_state.apply([ChangePage(ROWS, "results", rows)])
        va, ga = _tokens(store, "results", "VA"), _tokens(store, "results", "GA")
        await by_state.apply([ChangePage(ROWS, "results", _rows("VA", 50, 9).slice(10, 5))])
        _vote_only(va, _tokens(store, "results", "VA"), _STATIC)
        assert _tokens(store, "results", "GA") == ga

    async def test_on_a_table_held_as_one_part(self, store: SharedArrowStore) -> None:
        """the same on a table never split: its one part keeps every static column's token."""
        backend = SharedCopyBackend(store, {"results": SCHEMA}, {"results": WHOLE}, pack_below=0)
        try:
            await backend.apply([ChangePage(ROWS, "results", _rows("VA", 40, 1))])
            before = _tokens(store, "results", "all")
            await backend.apply([ChangePage(ROWS, "results", _rows("VA", 40, 2).slice(0, 3))])
            _vote_only(before, _tokens(store, "results", "all"), _STATIC)
        finally:
            await backend.close()

    async def test_on_a_pack_of_small_scopes(self, store: SharedArrowStore) -> None:
        """Flips if a pack is written unstamped: a vote in one small state rewrites its pack, which keeps every static
        column's token (the pack's other states' rows unmoved); the counts' token moves."""
        backend = SharedCopyBackend(store, {"results": SCHEMA}, {"results": BY_STATE})
        try:
            states = [f"S{n:02d}" for n in range(20)]
            await backend.apply(
                [ChangePage(ROWS, "results", pa.concat_tables([_rows(s, 30, 1) for s in states]))], whole={"results"}
            )
            current = store.current
            assert current is not None
            packs = sorted(part for part in held_parts(store.root, current.name, "results") if is_pack(part))
            assert packs, "the small states are packed"
            before = {pack: _tokens(store, "results", pack) for pack in packs}
            applied = await backend.apply([ChangePage(ROWS, "results", _rows("S07", 30, 5).slice(0, 4))])
            [written] = [part for part, wrote in applied.parts["results"].items() if wrote]
            assert is_pack(written)
            _vote_only(before[written], _tokens(store, "results", written), _STATIC)
        finally:
            await backend.close()

    async def test_on_a_tier_a_merge_wrote(self, store: SharedArrowStore) -> None:
        """Flips if a merge writes its tier unstamped: the merged tier holds tokens, and a vote-only change to a run in
        it rewrites the tier keeping every static column's token."""
        backend = SharedCopyBackend(store, {"runs": RUN_SCHEMA}, {"runs": BY_RUN}, pack_below=0)
        try:
            for run in range(1, 13):
                await backend.apply([ChangePage(ROWS, "runs", _run_rows("NC", run, 20, run))])
            merged = await backend.merge()
            assert merged is not None
            before = _tokens(store, "runs", merged.into)
            applied = await backend.apply([ChangePage(ROWS, "runs", _run_rows("NC", 3, 20, -1).slice(5, 2))])
            assert applied.parts == {"runs": {merged.into: True}}
            _vote_only(before, _tokens(store, "runs", merged.into), ("state", "run_id", "candidate_id"))
        finally:
            await backend.close()
