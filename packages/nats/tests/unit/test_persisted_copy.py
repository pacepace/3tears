"""the one owner of a bucket that is the persisted copy of an in-memory structure.

The tool registry and the hub's agent router each kept a catalog in memory and its copy in a bucket,
each through its own persistence class holding a raw nats-py handle -- the handle that did not follow
a move to a successor connection, so every write after a NATS rolling restart failed until the pod
was deleted by hand. These pin what the shared owner does in their place: declare under the exact
name, retry a start the broker did not answer, load once, write memory back after the first load and
on every refill, and stop when told.

Run over the shipped in-memory client; the live counterpart, against a real broker, is
``tests/integration/test_a_persisted_copy_is_refilled_live.py``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any

import pytest

from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.nats import PersistedCopyBucket

_FAST = timedelta(milliseconds=1)


class _UnansweredClient(FakeNatsClient):
    """the shipped client, whose first ``unanswered`` declarations fail as a broker still starting does."""

    def __init__(self, *, unanswered: int) -> None:
        super().__init__()
        self.unanswered = unanswered
        self.declarations: list[dict[str, Any]] = []

    async def ensure_kv_bucket(self, **kwargs: Any) -> FakeKvBucket:  # type: ignore[override]
        self.declarations.append(kwargs)
        if self.unanswered > 0:
            self.unanswered -= 1
            raise TimeoutError("nats: timeout")
        return await super().ensure_kv_bucket(**kwargs)


class _Catalog:
    """an in-memory structure and its two callbacks, recording every call."""

    def __init__(self, entries: dict[str, bytes] | None = None) -> None:
        self.entries: dict[str, bytes] = dict(entries or {})
        self.loads = 0
        self.write_backs = 0
        self.load_failures: list[Exception] = []

    async def load(self, bucket: Any) -> None:
        self.loads += 1
        if self.load_failures:
            raise self.load_failures.pop(0)
        for key in await bucket.list_keys():
            value = await bucket.get(key=key)
            if value is not None:
                self.entries.setdefault(key, value)

    async def write_back(self, bucket: Any) -> None:
        self.write_backs += 1
        for key, value in self.entries.items():
            await bucket.put(key=key, value=value)


async def _until(condition: Any) -> None:
    """wait until ``condition()`` holds, failing the test if it never does.

    :param condition: the state the background declaration should reach
    :ptype condition: Callable[[], bool]
    :return: nothing
    :rtype: None
    """
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the owner did not reach the expected state in time")


def _owner(client: FakeNatsClient, catalog: _Catalog) -> PersistedCopyBucket:
    return PersistedCopyBucket(
        client=client,
        bucket="tool_catalog",
        load=catalog.load,
        write_back=catalog.write_back,
        retry_first_delay=_FAST,
        retry_max_delay=_FAST,
    )


@pytest.mark.asyncio
async def test_start_returns_at_once_and_declares_the_exact_name_in_the_background() -> None:
    client = _UnansweredClient(unanswered=0)
    catalog = _Catalog()
    owner = _owner(client, catalog)

    await owner.start()
    assert owner.bucket is None, "start does not wait on the broker"
    await _until(lambda: owner.bucket is not None)

    assert client.remembered_declarations == frozenset({"tool_catalog"})
    (declared,) = client.declarations
    assert declared["name"] == "tool_catalog"
    assert declared["prefix_namespace"] is False
    assert declared["storage"] == "file"
    assert declared["on_restored"] == catalog.write_back
    await owner.stop()


@pytest.mark.asyncio
async def test_a_start_the_broker_did_not_answer_is_retried_with_an_error_each_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _UnansweredClient(unanswered=2)
    owner = _owner(client, _Catalog())

    with caplog.at_level(logging.ERROR, logger="threetears.nats.persisted_copy"):
        await owner.start()
        await _until(lambda: owner.bucket is not None)

    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2, errors
    assert all("tool_catalog" in message and "nats: timeout" in message for message in errors), errors
    assert len(client.declarations) == 3
    await owner.stop()


@pytest.mark.asyncio
async def test_the_first_declaration_loads_what_an_earlier_process_persisted_and_writes_memory_back() -> None:
    client = _UnansweredClient(unanswered=0)
    persisted = await client.ensure_kv_bucket(name="tool_catalog", storage="file")
    await persisted.put(key="old", value=b"from-an-earlier-registry")
    catalog = _Catalog({"new": b"registered-before-the-bucket-was-declared"})
    owner = _owner(client, catalog)

    await owner.start()
    await _until(lambda: catalog.write_backs == 1)

    assert catalog.loads == 1
    assert catalog.entries == {"old": b"from-an-earlier-registry", "new": b"registered-before-the-bucket-was-declared"}
    assert sorted(persisted.keys()) == ["new", "old"], "a change made before the declaration reached the bucket"
    await owner.stop()


@pytest.mark.asyncio
async def test_a_load_that_failed_is_tried_again_before_anything_is_written() -> None:
    client = _UnansweredClient(unanswered=0)
    catalog = _Catalog()
    catalog.load_failures.append(RuntimeError("nats: connection closed"))
    owner = _owner(client, catalog)

    await owner.start()
    await _until(lambda: catalog.write_backs == 1)

    assert catalog.loads == 2
    await owner.stop()


@pytest.mark.asyncio
async def test_a_refill_writes_memory_back_and_never_loads_again() -> None:
    """a load after a restart could re-add an entry whose removal never reached the bucket."""
    client = _UnansweredClient(unanswered=0)
    catalog = _Catalog({"tool-a": b"a"})
    owner = _owner(client, catalog)
    await owner.start()
    await _until(lambda: catalog.write_backs == 1)
    bucket = owner.bucket
    assert isinstance(bucket, FakeKvBucket)

    await client.restart_broker()

    assert catalog.loads == 1
    assert catalog.write_backs == 2
    assert bucket.keys() == ("tool-a",)
    await owner.stop()


@pytest.mark.asyncio
async def test_stop_ends_a_declaration_still_retrying() -> None:
    client = _UnansweredClient(unanswered=10_000)
    owner = _owner(client, _Catalog())
    await owner.start()
    await _until(lambda: len(client.declarations) >= 2)

    await owner.stop()
    attempts = len(client.declarations)
    await asyncio.sleep(0.05)

    assert len(client.declarations) == attempts
    assert owner.bucket is None


@pytest.mark.asyncio
async def test_a_second_start_is_refused() -> None:
    owner = _owner(_UnansweredClient(unanswered=0), _Catalog())
    await owner.start()

    with pytest.raises(RuntimeError, match="already started"):
        await owner.start()
    await owner.stop()


def test_a_retry_schedule_it_cannot_back_off_on_is_refused() -> None:
    catalog = _Catalog()
    with pytest.raises(ValueError, match="retry_first_delay"):
        PersistedCopyBucket(
            client=FakeNatsClient(),
            bucket="tool_catalog",
            load=catalog.load,
            write_back=catalog.write_back,
            retry_first_delay=timedelta(seconds=2),
            retry_max_delay=timedelta(seconds=1),
        )
