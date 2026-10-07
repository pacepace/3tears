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
from threetears.core.collections.scope_epochs import WHOLE, EpochSnapshot, ScopeEpochs, scope_epochs_collection
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
            date_created TIMESTAMPTZ NOT NULL,
            date_updated TIMESTAMPTZ
        )
        """
    )
    return pool


def _replica(pool: asyncpg.Pool, nats: NatsClient | None = None) -> tuple[CollectionRegistry, ScopeEpochs]:
    registry = CollectionRegistry()
    l1 = SQLiteBackend(db_name=f"epochs_{uuid.uuid4().hex[:8]}")
    collection_class = scope_epochs_collection(_TABLE)
    metadata = MetaData()
    collection_class.schema.to_sqlalchemy_table(metadata)
    l1.initialize(metadata)
    # replicas of one deployment share one key scope, as tool pods of one ``tool_pods.id`` do
    registry.configure(l1_backend=l1, l3_pool=SqlL3Backend(pool), kv_key_scope="epochs-pod")
    collection = collection_class(registry, DefaultCoreConfig(), nats)
    return registry, ScopeEpochs(collection)


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
    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    await _in_transaction(held.pool, lambda conn: held.epochs.commit(version, scopes, conn=conn))
    return version


async def test_nothing_written_is_version_zero_settled_with_no_scopes(held: _Held) -> None:
    snapshot = await held.epochs.snapshot()

    assert snapshot == EpochSnapshot(version=0, writing=None, epochs={}, previous={})
    assert await held.epochs.settled() == snapshot
    assert snapshot.epoch("state:VA") == 0


async def test_a_write_begun_is_in_progress_until_it_commits(held: _Held) -> None:
    version = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))

    assert version == 1
    assert (await held.epochs.snapshot()).writing == 1
    assert await held.epochs.settled() is None

    await _in_transaction(held.pool, lambda conn: held.epochs.commit(version, {"state:VA", "race:va-sen"}, conn=conn))

    settled = await held.epochs.settled()
    assert settled is not None
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
