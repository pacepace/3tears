"""Integration proof: the JetStream op-log primitive against a real broker.

This is the output-first proof test for the op-log (WAL) primitive. It asserts the
three load-bearing behaviours from the scriob write-path spec against a real
JetStream-enabled NATS server (no fakes):

1. **expected-last-seq CAS (fence #1).** An ``append`` carrying a *stale*
   ``expected_last_seq`` is rejected with :class:`OpLogSequenceConflict` -- the
   in-band fence that stops a split-brain second writer.
2. **op-id dedup (at-most-once).** A retried ``append`` carrying an ``op_id`` already
   in the dedup window is a no-op: it returns the *original* sequence with
   ``deduplicated=True`` and appends no second message.
3. **ordered replay.** ``replay(from_seq)`` yields the logged ops in sequence order
   (payload + op_id + seq preserved), for both a full replay and a tail replay
   (the failover-resume cursor).

The stream is per ``(repo, branch)``, in-memory, R3 -- durability rides replication,
not disk (write-path.md). Uses the canonical session-scoped ``nats_container`` fixture;
a checkout without docker skips cleanly.

run with::

    uv run pytest -m integration packages/nats/tests/integration/test_oplog_round_trip.py
"""

from __future__ import annotations

import pytest

from threetears.nats import (
    NatsClient,
    OpLog,
    OpLogSequenceConflict,
    set_default_namespace,
)

pytestmark = pytest.mark.integration


async def _connect(nats_url: str) -> NatsClient:
    """connect a wrapper client bound to the op-log test namespace."""
    set_default_namespace("oplogtest")
    return await NatsClient.connect(
        nats_url=nats_url,
        nats_subject_namespace="oplogtest",
        client_name="oplog-proof",
    )


async def test_append_returns_monotonic_sequence(nats_container: str) -> None:
    """Three successive appends return sequence 1, 2, 3 on a fresh stream."""
    async with await _connect(nats_container) as nc:
        oplog = await OpLog.open(client=nc, repo="proof-mono", branch="main")

        r1 = await oplog.append(payload=b"op-1", op_id="m1", expected_last_seq=0)
        r2 = await oplog.append(payload=b"op-2", op_id="m2", expected_last_seq=r1.seq)
        r3 = await oplog.append(payload=b"op-3", op_id="m3", expected_last_seq=r2.seq)

        assert (r1.seq, r2.seq, r3.seq) == (1, 2, 3)
        assert (r1.deduplicated, r2.deduplicated, r3.deduplicated) == (False, False, False)


async def test_stale_expected_seq_is_rejected(nats_container: str) -> None:
    """A stale expected_last_seq append (fence #1) is rejected in-band.

    The first writer advances the log to seq 1; a second writer that still believes the
    log is empty (``expected_last_seq=0``) must be fenced out -- never silently appended.
    """
    async with await _connect(nats_container) as nc:
        oplog = await OpLog.open(client=nc, repo="proof-cas", branch="main")

        first = await oplog.append(payload=b"winner", op_id="c1", expected_last_seq=0)
        assert first.seq == 1

        # A different op (distinct op_id, so dedup cannot mask it) with a stale
        # expected_last_seq must be rejected, not appended.
        with pytest.raises(OpLogSequenceConflict):
            await oplog.append(payload=b"loser", op_id="c2", expected_last_seq=0)

        # The rejected op left no trace: the log still holds exactly the winner.
        records = [r async for r in oplog.replay(from_seq=1)]
        assert [r.payload for r in records] == [b"winner"]


async def test_duplicate_op_id_is_at_most_once(nats_container: str) -> None:
    """A retried append (same op_id within the window) is an at-most-once no-op.

    Models the client/transport retry: the identical op is resent. It must NOT produce a
    second log entry; the call reports the original sequence and ``deduplicated=True``.
    """
    async with await _connect(nats_container) as nc:
        oplog = await OpLog.open(client=nc, repo="proof-dedup", branch="main")

        original = await oplog.append(payload=b"edit", op_id="dup-1", expected_last_seq=0)
        assert original.seq == 1
        assert original.deduplicated is False

        retry = await oplog.append(payload=b"edit", op_id="dup-1", expected_last_seq=0)
        assert retry.deduplicated is True
        assert retry.seq == original.seq

        # At-most-once: exactly one message landed for that op_id.
        records = [r async for r in oplog.replay(from_seq=1)]
        assert len(records) == 1
        assert records[0].op_id == "dup-1"


async def test_replay_is_ordered_full_and_tail(nats_container: str) -> None:
    """replay(from_seq) returns ops in order, for a full replay and a failover tail.

    Full replay drives materialise; tail replay (from a committed-through seq) is the
    failover-resume cursor -- a takeover pod replays only the ops after HEAD's Op-Seq.
    """
    async with await _connect(nats_container) as nc:
        oplog = await OpLog.open(client=nc, repo="proof-replay", branch="main")

        last = 0
        for i in range(1, 6):
            result = await oplog.append(
                payload=f"op-{i}".encode(),
                op_id=f"r{i}",
                expected_last_seq=last,
            )
            last = result.seq

        full = [r async for r in oplog.replay(from_seq=1)]
        assert [r.seq for r in full] == [1, 2, 3, 4, 5]
        assert [r.payload for r in full] == [b"op-1", b"op-2", b"op-3", b"op-4", b"op-5"]
        assert [r.op_id for r in full] == ["r1", "r2", "r3", "r4", "r5"]

        # Failover tail: resume after seq 3 yields only the uncommitted remainder, in order.
        tail = [r async for r in oplog.replay(from_seq=4)]
        assert [r.seq for r in tail] == [4, 5]
        assert [r.payload for r in tail] == [b"op-4", b"op-5"]
