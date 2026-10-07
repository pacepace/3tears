"""the scoped snapshot's names and chunk codec, without NATS."""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")

from threetears.core.collections.scoped_snapshot import (  # noqa: E402
    ScopedSnapshot,
    SnapshotTable,
    decode_chunk,
    encode_chunk,
    scope_token,
)


@pytest.mark.parametrize(
    ("scope", "token"),
    [("TX", "TX"), ("state-1_a", "state-1_a"), ("a b", "a=20b"), ("a.b", "a=2Eb"), ("x=y", "x=3Dy"), (None, "=")],
)
def test_a_scope_becomes_one_literal_token(scope: str | None, token: str) -> None:
    assert scope_token(scope) == token


def test_distinct_scopes_never_share_a_token() -> None:
    scopes = ["a.b", "a=2Eb", "a b", "=", "", "é"]
    assert len({scope_token(s) for s in scopes}) == len(scopes)


def test_a_chunk_round_trips_and_is_compressed() -> None:
    table = pa.table({"state": ["TX"] * 5000, "votes": list(range(5000))})
    data = encode_chunk(table)
    assert decode_chunk(data).equals(table)
    assert len(data) < table.nbytes


@pytest.mark.parametrize("name", ["", "a.b", "a b", "a/b"])
def test_a_snapshot_name_is_one_token(name: str) -> None:
    with pytest.raises(ValueError):
        ScopedSnapshot(
            name=name,
            tables=[SnapshotTable(name="t", scope_column="s", key=("k",))],
            backend=None,  # type: ignore[arg-type]
            store=None,
            pointers=None,
            l3=None,
            epochs=None,  # type: ignore[arg-type]
        )
