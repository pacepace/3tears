"""unit tests for :meth:`threetears.nats.NatsKvBucket.watch_key`.

A grant narrowed to one key admits one consumer shape: a NAMED push consumer whose filter rides in
the create subject. nats-py's ``KeyValue.watch`` creates an unnamed one and is refused -- silently,
by blocking to its deadline -- so the watch must build its own consumer, name each one afresh, and
notice by missed heartbeats when the server has lost it. These tests pin that shape against a
scripted stand-in for the nats-py client; the live-broker proof under the real grant is
``tests/integration/test_kv_watch_key_grant_live.py``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import timedelta
from typing import Any

import nats.errors
import pytest
from nats.aio.msg import Msg
from nats.js.api import AckPolicy, ConsumerConfig, DeliverPolicy

from threetears.nats import KvError, NatsKvBucket
from threetears.nats.kv_watch import KvKeyUpdate, KvKeyWatching

_BUCKET = "3tears-dv"
_STREAM = f"KV_{_BUCKET}"
_KEY = "019470a8b5c37def8123456789abcdef"
_SUBJECT = f"$KV.{_BUCKET}.{_KEY}"
_FAST = timedelta(milliseconds=20)


def _delivered(data: bytes, *, sequence: int, operation: str | None = None) -> Msg:
    """a message as a push consumer delivers it, its stream sequence in the ack reply.

    :param data: the payload
    :ptype data: bytes
    :param sequence: the stream sequence -- the key's revision
    :ptype sequence: int
    :param operation: the ``KV-Operation`` header, for a delete or purge marker
    :ptype operation: str | None
    :return: the message
    :rtype: Msg
    """
    headers = {"KV-Operation": operation} if operation is not None else None
    reply = f"$JS.ACK.{_STREAM}.kw_x.1.{sequence}.{sequence}.1700000000000000000.0"
    return Msg(_client=None, subject=_SUBJECT, reply=reply, data=data, headers=headers)  # type: ignore[arg-type]


def _heartbeat() -> Msg:
    """an idle heartbeat: no payload, a ``Status: 100`` header.

    :return: the heartbeat
    :rtype: Msg
    """
    return Msg(_client=None, subject="", reply="", data=b"", headers={"Status": "100", "Description": "Idle Heartbeat"})  # type: ignore[arg-type]


# parity-exempt: stands in for a nats-py core Subscription on a deliver inbox; the watch calls only next_msg and unsubscribe
class _Inbox:
    def __init__(self, subject: str, script: list[Msg], *, closed_after: bool = False) -> None:
        self.subject = subject
        self._script = list(script)
        self._closed_after = closed_after
        self.unsubscribed = False

    async def next_msg(self, timeout: float) -> Msg:
        if self._script:
            return self._script.pop(0)
        if self._closed_after:
            raise nats.errors.ConnectionClosedError
        await asyncio.sleep(timeout)
        raise nats.errors.TimeoutError

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


# parity-exempt: stands in for the nats-py client behind NatsClient.raw; the watch reads is_closed and calls new_inbox and subscribe
class _Raw:
    def __init__(self, scripts: list[list[Msg]], log: list[str], *, closed_after_scripts: bool = False) -> None:
        self._scripts = scripts
        self._log = log
        self._closed_after_scripts = closed_after_scripts
        self.is_closed = False
        self.inboxes: list[_Inbox] = []
        self._count = 0

    def new_inbox(self) -> str:
        self._count += 1
        return f"_INBOX_pod.{self._count}"

    async def subscribe(self, subject: str) -> _Inbox:
        self._log.append(f"subscribe {subject}")
        script = self._scripts.pop(0) if self._scripts else []
        inbox = _Inbox(subject, script, closed_after=self._closed_after_scripts and not self._scripts)
        self.inboxes.append(inbox)
        return inbox


# parity-exempt: stands in for the nats-py JetStream context; records add_consumer and refuses the stock KV watch path outright
class _JetStream:
    def __init__(self, log: list[str], *, failures: int = 0) -> None:
        self._log = log
        self._failures = failures
        self.creates: list[tuple[str, ConsumerConfig]] = []

    async def add_consumer(self, stream: str, config: ConsumerConfig) -> object:
        self._log.append(f"create {config.name}")
        if self._failures:
            self._failures -= 1
            raise nats.errors.TimeoutError
        self.creates.append((stream, config))
        return object()

    async def key_value(self, bucket: str) -> Any:
        raise AssertionError("watch_key must not open nats-py's KeyValue: its watch() is refused under a key grant")


# parity-exempt: stands in for NatsClient; the watch calls only raw and jetstream_context
class _Client:
    def __init__(self, raw: _Raw, js: _JetStream) -> None:
        self.raw = raw
        self._js = js

    def jetstream_context(self) -> _JetStream:
        return self._js


# parity-exempt: stands in for nats-py's KeyValue handle; watch_key must never reach it
class _ForbiddenKv:
    async def watch(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("watch_key must not call KeyValue.watch")


def _bucket(
    scripts: list[list[Msg]], *, failures: int = 0, closed_after_scripts: bool = False
) -> tuple[NatsKvBucket, _Raw, _JetStream, list[str]]:
    """a bucket over scripted deliveries, one script per consumer the watch creates.

    :param scripts: the messages each successive consumer delivers
    :ptype scripts: list[list[Msg]]
    :param failures: how many consumer creates fail first
    :ptype failures: int
    :param closed_after_scripts: close the connection once the last script is drained
    :ptype closed_after_scripts: bool
    :return: the bucket, the raw client, the JetStream context and the call log
    :rtype: tuple[NatsKvBucket, _Raw, _JetStream, list[str]]
    """
    log: list[str] = []
    raw = _Raw(scripts, log, closed_after_scripts=closed_after_scripts)
    js = _JetStream(log, failures=failures)
    bucket = NatsKvBucket(
        client=_Client(raw, js),  # type: ignore[arg-type]
        full_name=_BUCKET,
        kv=_ForbiddenKv(),  # type: ignore[arg-type]
        ttl=None,
    )
    return bucket, raw, js, log


async def _take(watch: AsyncGenerator[KvKeyUpdate], count: int) -> list[KvKeyUpdate]:
    """the first ``count`` updates a watch yields, then close it.

    :param watch: the watch
    :ptype watch: AsyncGenerator[KvKeyUpdate]
    :param count: how many
    :ptype count: int
    :return: the updates
    :rtype: list[KvKeyUpdate]
    """
    taken: list[KvKeyUpdate] = []
    async with aclosing(watch) as updates:
        async for update in updates:
            taken.append(update)
            if len(taken) == count:
                break
    return taken


class TestTheConsumer:
    """every consumer is named, filtered in its subject, last-per-subject, unacknowledged."""

    async def test_the_consumer_is_named_filtered_on_the_key_and_delivers_to_a_fresh_inbox(self) -> None:
        bucket, raw, js, log = _bucket([[_delivered(b"3", sequence=7)]])

        await asyncio.wait_for(_take(bucket.watch_key(key=_KEY, heartbeat=_FAST), 1), timeout=2)

        stream, config = js.creates[0]
        assert stream == _STREAM
        assert config.name is not None and config.name.startswith("kw_")
        assert config.filter_subject == _SUBJECT
        assert config.deliver_subject == raw.inboxes[0].subject
        assert config.deliver_policy is DeliverPolicy.LAST_PER_SUBJECT
        assert config.ack_policy is AckPolicy.NONE
        assert config.idle_heartbeat == _FAST.total_seconds()
        assert config.durable_name is None
        # the inbox listens before the consumer exists, so nothing it delivers can be missed
        assert log[0] == f"subscribe {raw.inboxes[0].subject}"
        assert log[1] == f"create {config.name}"

    async def test_closing_the_watch_drops_the_inbox(self) -> None:
        bucket, raw, _js, _log = _bucket([[_delivered(b"3", sequence=7)]])

        await asyncio.wait_for(_take(bucket.watch_key(key=_KEY, heartbeat=_FAST), 1), timeout=2)

        assert raw.inboxes[0].unsubscribed

    async def test_a_bucket_satisfies_the_watching_protocol(self) -> None:
        bucket, _raw, _js, _log = _bucket([])

        assert isinstance(bucket, KvKeyWatching)


class TestWhatItYields:
    """the key's latest message, then every later one, with its revision; removals as ``None``."""

    async def test_the_current_value_then_an_update(self) -> None:
        bucket, _raw, _js, _log = _bucket([[_delivered(b"3", sequence=7), _delivered(b"4", sequence=9)]])

        updates = await asyncio.wait_for(_take(bucket.watch_key(key=_KEY, heartbeat=_FAST), 2), timeout=2)

        assert updates == [
            KvKeyUpdate(key=_KEY, value=b"3", revision=7),
            KvKeyUpdate(key=_KEY, value=b"4", revision=9),
        ]

    async def test_a_delete_and_a_purge_arrive_as_removals(self) -> None:
        bucket, _raw, _js, _log = _bucket(
            [[_delivered(b"", sequence=4, operation="DEL"), _delivered(b"", sequence=5, operation="PURGE")]]
        )

        updates = await asyncio.wait_for(_take(bucket.watch_key(key=_KEY, heartbeat=_FAST), 2), timeout=2)

        assert [(u.value, u.revision, u.deleted) for u in updates] == [(None, 4, True), (None, 5, True)]

    async def test_a_heartbeat_is_proof_of_life_not_an_update(self) -> None:
        bucket, _raw, js, _log = _bucket([[_heartbeat(), _heartbeat(), _delivered(b"3", sequence=7)]])

        updates = await asyncio.wait_for(_take(bucket.watch_key(key=_KEY, heartbeat=_FAST), 1), timeout=2)

        assert updates == [KvKeyUpdate(key=_KEY, value=b"3", revision=7)]
        assert len(js.creates) == 1


class TestReplacingAConsumer:
    """a consumer that stops heartbeating is replaced under a fresh name, and nothing repeats."""

    async def test_missed_heartbeats_replace_the_consumer_under_a_fresh_name(self) -> None:
        bucket, raw, js, _log = _bucket(
            [
                [_delivered(b"3", sequence=7)],
                # the replacement redelivers the latest message, then a new one arrives
                [_delivered(b"3", sequence=7), _delivered(b"4", sequence=8)],
            ]
        )

        updates = await asyncio.wait_for(_take(bucket.watch_key(key=_KEY, heartbeat=_FAST), 2), timeout=2)

        assert [u.value for u in updates] == [b"3", b"4"]
        names = [config.name for _stream, config in js.creates]
        assert len(names) == 2
        assert len(set(names)) == 2
        assert raw.inboxes[0].unsubscribed

    async def test_a_consumer_the_server_deleted_is_replaced_without_waiting_out_its_heartbeats(self) -> None:
        deleted = Msg(_client=None, data=b"", headers={"Status": "409", "Description": "Consumer Deleted"})  # type: ignore[arg-type]
        bucket, raw, js, _log = _bucket([[_delivered(b"3", sequence=7), deleted], [_delivered(b"4", sequence=8)]])

        # a heartbeat far longer than the test's budget: only the status can trigger the replacement
        updates = await asyncio.wait_for(
            _take(bucket.watch_key(key=_KEY, heartbeat=timedelta(seconds=30)), 2), timeout=2
        )

        assert [u.value for u in updates] == [b"3", b"4"]
        assert len(js.creates) == 2
        assert raw.inboxes[0].unsubscribed

    async def test_a_redelivery_after_a_wipe_that_changed_the_value_is_yielded(self) -> None:
        bucket, _raw, _js, _log = _bucket([[_delivered(b"3", sequence=7)], [_delivered(b"5", sequence=1)]])

        updates = await asyncio.wait_for(_take(bucket.watch_key(key=_KEY, heartbeat=_FAST), 2), timeout=2)

        assert updates[1] == KvKeyUpdate(key=_KEY, value=b"5", revision=1)

    async def test_a_failed_create_is_logged_naming_the_grant_and_retried(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bucket, raw, js, _log = _bucket([[], [_delivered(b"3", sequence=7)]], failures=1)

        with caplog.at_level(logging.WARNING, logger="threetears.nats.kv"):
            updates = await asyncio.wait_for(
                _take(bucket.watch_key(key=_KEY, heartbeat=_FAST, retry=_FAST), 1), timeout=2
            )

        assert updates == [KvKeyUpdate(key=_KEY, value=b"3", revision=7)]
        assert raw.inboxes[0].unsubscribed
        assert len(js.creates) == 1
        assert f"$JS.API.CONSUMER.CREATE.{_STREAM}.*.{_SUBJECT}" in caplog.text


class TestRefusals:
    """a key that is not one literal key, and a closed connection, are refused."""

    @pytest.mark.parametrize("key", ["", "a.*", "a.>", ">", "a b"])
    async def test_a_key_that_is_not_literal_is_refused(self, key: str) -> None:
        bucket, _raw, _js, _log = _bucket([])

        with pytest.raises(ValueError, match="literal key"):
            await anext(bucket.watch_key(key=key))

    async def test_a_closed_connection_ends_the_watch_with_an_error(self) -> None:
        bucket, raw, _js, _log = _bucket([])
        raw.is_closed = True

        with pytest.raises(KvError, match="closed"):
            await anext(bucket.watch_key(key=_KEY))

    async def test_a_connection_closed_mid_watch_ends_it_with_an_error(self) -> None:
        bucket, _raw, _js, _log = _bucket([[_delivered(b"3", sequence=7)]], closed_after_scripts=True)
        watch = bucket.watch_key(key=_KEY, heartbeat=_FAST)

        async with aclosing(watch) as updates:
            assert await anext(updates) == KvKeyUpdate(key=_KEY, value=b"3", revision=7)
            with pytest.raises(KvError, match="closed"):
                await anext(updates)
