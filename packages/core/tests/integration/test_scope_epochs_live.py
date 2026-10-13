"""Integration: per-scope epochs kept in L3, advanced after a write commits, never back.

What a fake cannot prove: that the epochs are rows in a real Postgres table moved in the caller's
own transaction (so a rolled-back commit moves nothing), that the record a copy builder reads says
"a write is in progress" from the moment one begins, and that a commit on one replica reaches a
second replica's listener over a real NATS bus -- and, when that broadcast is missed, that the
second replica still learns the new epochs by reading them.

Uses the session-scoped ``db_container`` and ``nats_container`` fixtures; a checkout without docker
skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.complete_copy import Unsettled
from threetears.core.collections.scope_epochs import (
    STALLED_WRITE_SECONDS,
    WHOLE,
    EpochSnapshot,
    ScopeEpochs,
    scope_epochs_collection,
)
from threetears.core.config import DefaultCoreConfig
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_TABLE = "scope_epochs"


@dataclass
class _Held:
    pool: asyncpg.Pool
    epochs: ScopeEpochs


async def _schema_pool(db_container: str) -> asyncpg.Pool:
    schema = f"epochs_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    await pool.execute(
        f"""
        CREATE TABLE {_TABLE} (
            scope TEXT PRIMARY KEY,
            epoch BIGINT NOT NULL,
            previous_epoch BIGINT,
            writing BIGINT,
            pending BIGINT,
            date_created TIMESTAMPTZ NOT NULL,
            date_updated TIMESTAMPTZ
        )
        """
    )
    await pool.execute("CREATE TABLE results (state TEXT PRIMARY KEY, votes BIGINT)")
    return pool


def _replica(
    pool: asyncpg.Pool, nats: NatsClient | None = None, *, idle_timeout: Any = None
) -> tuple[CollectionRegistry, ScopeEpochs]:
    registry = CollectionRegistry()
    l1 = SQLiteBackend(db_name=f"epochs_{uuid.uuid4().hex[:8]}")
    collection_class = scope_epochs_collection(_TABLE)
    metadata = MetaData()
    collection_class.schema.to_sqlalchemy_table(metadata)
    l1.initialize(metadata)
    # replicas of one deployment share one key scope, as tool pods of one ``tool_pods.id`` do
    registry.configure(l1_backend=l1, l3_pool=SqlL3Backend(pool), kv_key_scope="epochs-pod")
    collection = collection_class(registry, DefaultCoreConfig(), nats)
    return registry, ScopeEpochs(collection, idle_timeout=idle_timeout)


@pytest.fixture
async def held(db_container: str) -> AsyncIterator[_Held]:
    pool = await _schema_pool(db_container)
    try:
        yield _Held(pool, _replica(pool)[1])
    finally:
        await pool.close()


async def _in_transaction(pool: asyncpg.Pool, step: Any) -> Any:
    async with pool.acquire() as conn, CallerTransaction(conn):
        return await step(conn)


async def _write(held: _Held, scopes: set[str]) -> int:
    version: int = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    await _in_transaction(held.pool, lambda conn: held.epochs.commit(version, scopes, conn=conn))
    return version


async def test_nothing_written_is_version_zero_settled_with_no_scopes(held: _Held) -> None:
    snapshot = await held.epochs.snapshot()

    assert snapshot == EpochSnapshot(version=0, writing=None, epochs={}, previous={}, writing_since=None)
    assert await held.epochs.settled() == snapshot
    assert snapshot.epoch("state:VA") == 0


async def test_a_write_begun_is_in_progress_until_it_commits(held: _Held) -> None:
    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))

    assert version == 1
    begun = await held.epochs.snapshot()
    assert begun.writing == 1
    assert begun.writing_since is not None, "a write in progress does not say when it began"
    assert isinstance(await held.epochs.settled(), Unsettled)

    await _in_transaction(held.pool, lambda conn: held.epochs.commit(version, {"state:VA", "race:va-sen"}, conn=conn))

    settled = await held.epochs.settled()
    assert isinstance(settled, EpochSnapshot)
    assert (settled.version, settled.writing) == (1, None)
    assert dict(settled.epochs) == {"state:VA": 1, "race:va-sen": 1}


async def test_a_commit_moves_only_the_scopes_it_names_and_records_what_each_replaced(held: _Held) -> None:
    await _write(held, {"state:VA", "state:MD", "race:va-sen"})

    await _write(held, {"state:VA"})

    snapshot = await held.epochs.snapshot()
    assert snapshot.version == 2
    assert dict(snapshot.epochs) == {"state:VA": 2, "state:MD": 1, "race:va-sen": 1}
    assert snapshot.previous["state:VA"] == 1
    assert snapshot.previous["state:MD"] is None
    assert snapshot.previous[WHOLE] == 1


async def test_a_write_that_changed_nothing_moves_the_version_and_no_scope(held: _Held) -> None:
    await _write(held, {"state:VA"})

    await _write(held, set())

    snapshot = await held.epochs.snapshot()
    assert (snapshot.version, dict(snapshot.epochs)) == (2, {"state:VA": 1})


async def test_a_commit_rolled_back_with_its_transaction_moves_nothing(held: _Held) -> None:
    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))

    async def commit_then_fail(conn: Any) -> None:
        await held.epochs.commit(version, {"state:VA"}, conn=conn)
        raise ConnectionError("the data write after it failed")

    with pytest.raises(ConnectionError):
        await _in_transaction(held.pool, commit_then_fail)

    snapshot = await held.epochs.snapshot()
    assert (snapshot.version, snapshot.writing, dict(snapshot.epochs)) == (0, version, {})


async def test_a_write_begun_after_one_that_never_committed_takes_a_new_number(held: _Held) -> None:
    abandoned = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))

    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))

    assert version == abandoned + 1, "a number a partial write may have stamped rows with was used again"


async def test_a_commit_of_a_write_not_in_progress_is_refused(held: _Held) -> None:
    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))

    with pytest.raises(ValueError, match="in progress"):
        await _in_transaction(held.pool, lambda conn: held.epochs.commit(version + 1, {"state:VA"}, conn=conn))
    assert (await held.epochs.snapshot()).epoch("state:VA") == 0


async def test_a_writer_whose_lease_lapsed_cannot_commit_over_the_writer_that_followed(held: _Held) -> None:
    """a later begin supersedes: its number is the fencing token, and only it may commit."""
    stale = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    current = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))

    with pytest.raises(ValueError, match="in progress"):
        await _in_transaction(held.pool, lambda conn: held.epochs.commit(stale, {"state:VA"}, conn=conn))
    snapshot = await held.epochs.snapshot()
    assert snapshot.writing == current, "the stale writer cleared the marker of the write that followed"

    await _in_transaction(held.pool, lambda conn: held.epochs.commit(current, {"state:VA"}, conn=conn))
    with pytest.raises(ValueError, match="in progress"):
        await _in_transaction(held.pool, lambda conn: held.epochs.commit(stale, {"state:MD"}, conn=conn))
    after = await held.epochs.snapshot()
    assert (after.version, after.epoch("state:VA"), after.epoch("state:MD")) == (current, current, 0)


async def test_writes_begun_together_never_share_a_number(held: _Held) -> None:
    """two begins at once (a lapsed lease): each takes its own number, or is refused because the
    other holds the record; never the same number, and never a wait."""
    from threetears.core.collections.scope_epochs import EpochRecordBusyError

    # the record's row exists, as after any first write, so each begin takes its lock with NOWAIT
    # (with no row there is nothing to lock, and the begins only meet at the insert)
    await _write(held, set())
    outcomes = await asyncio.gather(
        *(_in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn)) for _ in range(4)),
        return_exceptions=True,
    )

    numbers = [outcome for outcome in outcomes if isinstance(outcome, int)]
    refused = [outcome for outcome in outcomes if not isinstance(outcome, int)]
    assert numbers, "every begin was refused"
    assert len(set(numbers)) == len(numbers)
    assert all(isinstance(outcome, EpochRecordBusyError) for outcome in refused)
    assert (await held.epochs.snapshot()).writing == max(numbers)


async def test_a_scope_already_at_or_past_the_version_is_refused_and_nothing_moves(held: _Held) -> None:
    await _write(held, {"state:VA"})
    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    await held.pool.execute(f"UPDATE {_TABLE} SET epoch = $1 WHERE scope = 'state:VA'", version + 5)

    with pytest.raises(ValueError, match="already at or past"):
        await _in_transaction(held.pool, lambda conn: held.epochs.commit(version, {"state:VA", "state:MD"}, conn=conn))

    snapshot = await held.epochs.snapshot()
    assert (snapshot.version, snapshot.writing, snapshot.epoch("state:MD")) == (1, version, 0)


async def test_scopes_an_abandoned_write_touched_move_at_the_next_commit(held: _Held) -> None:
    """an abandoned write committed rows for VA; the next write changes only MD, and VA still moves."""
    abandoned = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    # its data transaction for VA committed, with VA recorded as touched in the same transaction
    await _in_transaction(held.pool, lambda conn: held.epochs.touch(abandoned, {"state:VA"}, conn=conn))

    version = await _write(held, {"state:MD"})

    snapshot = await held.epochs.snapshot()
    assert (snapshot.epoch("state:VA"), snapshot.epoch("state:MD")) == (version, version)
    await _write(held, {"state:MD"})
    assert (await held.epochs.snapshot()).epoch("state:VA") == version, "a touched scope moved twice"


async def test_a_touch_rolled_back_with_its_data_leaves_nothing_pending(held: _Held) -> None:
    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))

    async def touch_then_fail(conn: Any) -> None:
        await held.epochs.touch(version, {"state:VA"}, conn=conn)
        raise ConnectionError("the data write failed")

    with pytest.raises(ConnectionError):
        await _in_transaction(held.pool, touch_then_fail)
    await _in_transaction(held.pool, lambda conn: held.epochs.commit(version, set(), conn=conn))

    assert (await held.epochs.snapshot()).epoch("state:VA") == 0


async def test_the_whole_record_is_not_a_scope_a_caller_may_name(held: _Held) -> None:
    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))

    with pytest.raises(ValueError, match="reserved"):
        await _in_transaction(held.pool, lambda conn: held.epochs.commit(version, {WHOLE}, conn=conn))


async def test_more_scopes_than_one_page_are_all_read(held: _Held) -> None:
    scopes = {f"race:r{index:04d}" for index in range(1500)}

    await _write(held, scopes)

    assert set((await held.epochs.snapshot()).epochs) == scopes


async def test_a_commit_on_one_replica_reaches_the_others_listener_and_a_missed_one_is_read(
    db_container: str, nats_container: str
) -> None:
    namespace = f"epochs{uuid.uuid4().hex[:8]}"
    set_default_namespace(namespace)
    pool = await _schema_pool(db_container)
    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="a") as a,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="b") as b,
    ):
        await a.ensure_kv_bucket(name="collections", create_if_missing=True)
        registry_a, writer = _replica(pool, a)
        registry_b, reader = _replica(pool, b)
        await registry_a.start_invalidation_listener(a)
        await registry_b.start_invalidation_listener(b)
        heard = asyncio.Event()
        reader.on_change(heard.set)
        try:
            held = _Held(pool, writer)
            await _write(held, {"state:VA"})
            await asyncio.wait_for(heard.wait(), timeout=10)
            assert (await reader.snapshot()).epoch("state:VA") == 1

            # replica b stops listening (a dropped subscription, a pod restarting): it misses the
            # next broadcast, and a read of the epochs is still the truth
            await registry_b.stop_invalidation_listener()
            heard.clear()
            await _write(held, {"state:MD"})
            await asyncio.sleep(0.5)
            assert not heard.is_set()
            converged = await reader.snapshot()
            assert (converged.version, converged.epoch("state:MD")) == (2, 2)
        finally:
            await registry_a.stop_invalidation_listener()
            await registry_b.stop_invalidation_listener()
            await pool.close()


async def test_a_reader_refused_by_a_write_in_progress_is_told_how_long_it_has_been_going(
    held: _Held, caplog: pytest.LogCaptureFixture
) -> None:
    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))

    with caplog.at_level("INFO", logger="threetears.core.collections.scope_epochs"):
        assert isinstance(await held.epochs.settled(), Unsettled)

    [record] = [r for r in caplog.records if r.getMessage() == "a write is in progress; readers keep what they hold"]
    data = record.extra_data  # type: ignore[attr-defined]
    assert data["version"] == version
    assert 0 <= data["seconds"] < 60


async def test_a_superseded_writers_late_data_write_is_refused_and_leaves_nothing(held: _Held) -> None:
    """A's lease lapsed: B began and committed; A's late data transaction must not land under VA's epoch."""
    stale = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    current = await _write(held, {"state:MD"})
    assert current > stale

    async def late_data_write(conn: Any) -> None:
        await conn.execute("INSERT INTO results VALUES ('VA', 100)")
        await held.epochs.touch(stale, {"state:VA"}, conn=conn)

    with pytest.raises(ValueError, match="no longer in progress"):
        await _in_transaction(held.pool, late_data_write)

    assert await held.pool.fetch("SELECT * FROM results") == []
    pending = await held.pool.fetch(f"SELECT scope FROM {_TABLE} WHERE pending IS NOT NULL")
    assert pending == []
    settled = await held.epochs.settled()
    assert isinstance(settled, EpochSnapshot) and settled.epoch("state:VA") == 0


async def test_a_write_in_progress_far_too_long_is_logged_as_stalled_and_described(
    held: _Held, caplog: pytest.LogCaptureFixture
) -> None:
    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    await held.pool.execute(
        f"UPDATE {_TABLE} SET date_updated = now() - make_interval(secs => $1) WHERE scope = '*'",
        STALLED_WRITE_SECONDS + 60,
    )

    with caplog.at_level("INFO", logger="threetears.core.collections.scope_epochs"):
        assert isinstance(await held.epochs.settled(), Unsettled)

    [record] = [r for r in caplog.records if "write is in progress" in r.getMessage()]
    assert record.levelname == "WARNING"
    refused = await held.epochs.settled()
    assert isinstance(refused, Unsettled)
    assert f"write {version}" in refused.reason
    assert "in progress for" in refused.reason


async def test_a_begin_behind_a_hung_writers_open_data_transaction_is_refused_at_once(held: _Held) -> None:
    """a stale writer whose data transaction hangs open (holding the record's lock) cannot stall the next."""
    import time

    from threetears.core.collections.scope_epochs import EpochRecordBusyError

    stale = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    hung = await held.pool.acquire()
    hang = CallerTransaction(hung)
    await hang.__aenter__()
    try:
        await held.epochs.touch(stale, {"state:VA"}, conn=hung)  # its lock now held, never committed

        started = time.monotonic()
        with pytest.raises(EpochRecordBusyError):
            await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
        assert time.monotonic() - started < 2.0, "begin waited on the hung writer's lock"
    finally:
        await hang.__aexit__(ConnectionError, ConnectionError("hung writer abandoned"), None)
        await held.pool.release(hung)

    # once the hung transaction is gone, the next write begins
    assert await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn)) == stale + 1


async def test_writes_through_one_record_take_turns_and_each_turn_is_short(held: _Held) -> None:
    """every data transaction of a write passes the record's lock: measure what that costs per turn."""
    import time

    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    started = time.monotonic()
    turns = 20
    for index in range(turns):
        await _in_transaction(held.pool, lambda conn, i=index: held.epochs.touch(version, {f"state:S{i}"}, conn=conn))
    per_turn = (time.monotonic() - started) / turns
    assert per_turn < 0.05, f"a touch took {per_turn:.3f} s on a local Postgres"


async def test_two_data_transactions_of_one_write_run_alongside_and_its_commit_waits_for_both(held: _Held) -> None:
    """the touch fence is a share lock: one write's data transactions overlap; begin and commit still wait."""
    import time

    from threetears.core.collections.scope_epochs import EpochRecordBusyError

    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    first = await held.pool.acquire()
    open_first = CallerTransaction(first)
    await open_first.__aenter__()
    committed = False
    try:
        await held.epochs.touch(version, {"state:VA"}, conn=first)  # held open

        started = time.monotonic()
        await asyncio.wait_for(
            _in_transaction(held.pool, lambda conn: held.epochs.touch(version, {"state:MD"}, conn=conn)), timeout=2.0
        )
        overlap = time.monotonic() - started
        assert overlap < 0.5, f"a second data transaction of the write waited {overlap:.2f} s on the first"

        with pytest.raises(EpochRecordBusyError):
            await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
        commit = asyncio.create_task(
            _in_transaction(held.pool, lambda conn: held.epochs.commit(version, set(), conn=conn))
        )
        await asyncio.sleep(0.3)
        assert not commit.done(), "the commit did not wait for a data transaction still open"

        await open_first.__aexit__(None, None, None)
        committed = True
        await asyncio.wait_for(commit, timeout=5.0)
    finally:
        if not committed:
            await open_first.__aexit__(ConnectionError, ConnectionError("abandoned"), None)
        await held.pool.release(first)
    snapshot = await held.epochs.snapshot()
    assert (snapshot.version, snapshot.epoch("state:VA"), snapshot.epoch("state:MD")) == (version, version, version)


async def test_a_stalled_writers_hold_ends_within_the_idle_bound_and_the_next_begin_succeeds(held: _Held) -> None:
    """direct Postgres: the server ends a data transaction idle past ``idle_timeout``, releasing the record."""
    import time
    from datetime import timedelta

    from threetears.core.collections.scope_epochs import EpochRecordBusyError

    bound = timedelta(seconds=1)
    # another replica over the same pool shares the table, bounded where the fixture's is not
    _registry, epochs = _replica(held.pool, idle_timeout=bound)
    stale = await _in_transaction(held.pool, lambda conn: epochs.begin(conn=conn))
    hung = await held.pool.acquire()
    hang = CallerTransaction(hung)
    await hang.__aenter__()
    try:
        await epochs.touch(stale, {"state:VA"}, conn=hung)  # then the writer stalls, its connection alive
        with pytest.raises(EpochRecordBusyError):
            await _in_transaction(held.pool, lambda conn: epochs.begin(conn=conn))

        started = time.monotonic()
        version = None
        while version is None and time.monotonic() - started < bound.total_seconds() * 5:
            await asyncio.sleep(0.2)
            try:
                version = await _in_transaction(held.pool, lambda conn: epochs.begin(conn=conn))
            except EpochRecordBusyError:
                version = None
        released = time.monotonic() - started
        assert version == stale + 1, "the next begin never succeeded: the stalled hold was not bounded"
        assert released < bound.total_seconds() + 1.5, f"the hold lasted {released:.1f} s past a {bound} bound"
        assert await held.pool.fetchval(f"SELECT pending FROM {_TABLE} WHERE scope = 'state:VA'") is None, (
            "the stalled transaction's touch survived it"
        )
    finally:
        try:
            await hang.__aexit__(ConnectionError, ConnectionError("stalled writer abandoned"), None)
        except asyncpg.PostgresError, asyncpg.InterfaceError, ConnectionError:
            pass  # NOSILENT: the server already ended this transaction; that is what the test proves
        await held.pool.release(hung)
