"""an absent KV bucket raises its own typed error on every 3tears path, and a raw handle's failures classify.

Consumers could not tell "this bucket does not exist" from any other KV failure: every path raised
a plain :class:`~threetears.nats.KvError`, and the hub told them apart by matching nats-py's
exception class NAMES on ``__cause__``, because its enforcement keeps ``nats.*`` imports out of its
production code. Absent and refused need different responses -- wait for the declarer, or fix the
grant -- so the difference is a type here.

The nats-py errors in this file are never constructed by hand. A real nats-py ``JetStreamContext``
runs over :class:`_FakeWire`, which answers each request the way a JetStream server does on the wire --
an API error document, a 404 status, no responders, or no answer at all -- so every exception the
wrapper sees is the one nats-py itself raises from that answer.
"""

from __future__ import annotations

import asyncio
import logging
import json
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import nats.errors
import pytest
from nats.aio.client import ServerVersion
from nats.js.client import JetStreamContext

from threetears.nats import (
    KvBucketNotFoundError,
    KvError,
    NatsClient,
    NatsClientError,
    NatsKvBucket,
    is_bucket_not_found,
    is_key_not_found,
    is_nats_error,
)
from threetears.nats.errors import KvConfigMismatch, StreamSubjectsOverlapError
from threetears.nats.kv import KvTimings

#: a bind wait shrunk so a test of an absent bucket spends milliseconds, not the production 30s.
_FAST = KvTimings(
    op_timeout_seconds=2.0,
    bind_wait_for_declarer_seconds=0.05,
    bind_retry_first_delay_seconds=0.001,
    bind_retry_max_delay_seconds=0.004,
)

_NS = "3tears"
_BUCKET = f"{_NS}-nonces"
_STREAM = f"KV_{_BUCKET}"


def _kv_stream_config(stream: str) -> dict[str, Any]:
    """the stream config a JetStream server reports for a KV bucket, as wire JSON.

    :param stream: the backing stream's name
    :ptype stream: str
    :return: the ``config`` object of a ``STREAM.INFO`` response
    :rtype: dict[str, Any]
    """
    bucket = stream.removeprefix("KV_")
    return {
        "name": stream,
        "subjects": [f"$KV.{bucket}.>"],
        "max_msgs_per_subject": 1,
        "allow_direct": True,
        "allow_msg_ttl": True,
        "storage": "memory",
        "max_age": 0,
    }


# parity-exempt: the server-facing request seam nats-py's JetStreamContext drives, answering as a JetStream server does on the wire; it is the transport under nats-py, not a stand-in for a 3tears protocol
class _FakeWire:
    """a JetStream server's answers on the wire, under a real nats-py ``JetStreamContext``.

    ``streams`` maps a stream name to its wire config. A request it does not script is never
    answered -- the shape a refused JetStream request takes -- and so is every request while
    ``unanswered`` is set.

    :ivar streams: the streams the server holds
    :ivar unanswered: when set, every request dies on its deadline, as a refused or unreachable one does
    :ivar vanish_after_infos: drop every stream once this many ``STREAM.INFO`` requests were answered
    :ivar create_reply: the answer to a ``STREAM.CREATE``; ``None`` leaves it unanswered
    :ivar unacknowledged_publishes: how many KV writes in a row nothing acknowledges, as a broker in
        the middle of losing a stream does
    """

    def __init__(self) -> None:
        self.streams: dict[str, dict[str, Any]] = {}
        self.unanswered = False
        self.vanish_after_infos: int | None = None
        self.infos = 0
        self.seq = 0
        # the wrapper reads this to record which connection a handle was bound through
        self.is_closed = False
        # nats-py names a consumer in its create subject only for a server that says it can
        self.connected_server_version = ServerVersion("2.11.0")
        self.create_reply: _WireMsg | None = None
        self.unacknowledged_publishes = 0
        # when set, a STREAM.CREATE creates the stream it names, as a granted create does
        self.creates_streams = False
        # when set, the declarer puts a lost stream back as soon as anyone looks for it
        self.declarer_returns = False

    def new_inbox(self) -> str:
        return "_INBOX.listing"

    async def subscribe(self, subject: str) -> Any:
        del subject
        return MagicMock(unsubscribe=AsyncMock())

    def jetstream(self, **_kwargs: Any) -> JetStreamContext:
        return JetStreamContext(self)  # type: ignore[arg-type]

    async def request(
        self, subject: str, payload: bytes = b"", timeout: float | None = None, headers: Any = None
    ) -> _WireMsg:
        del payload, timeout, headers
        if self.unanswered:
            raise nats.errors.TimeoutError
        reply: _WireMsg | None = None
        if subject.startswith("$JS.API.STREAM.INFO."):
            reply = self._stream_info(subject.removeprefix("$JS.API.STREAM.INFO."))
        elif subject.startswith("$JS.API.DIRECT.GET."):
            reply = self._direct_get(subject.removeprefix("$JS.API.DIRECT.GET.").split(".", 1)[0])
        elif subject.startswith("$JS.API.STREAM.CREATE."):
            reply = self.create_reply
            if self.creates_streams:
                stream = subject.removeprefix("$JS.API.STREAM.CREATE.")
                self.streams[stream] = _kv_stream_config(stream)
                reply = self._stream_info(stream)
        elif subject.startswith("$KV."):
            reply = self._publish(subject.split(".")[1])
        elif subject.startswith("$JS.API.CONSUMER.CREATE."):
            stream = subject.removeprefix("$JS.API.CONSUMER.CREATE.").split(".", 1)[0]
            if stream not in self.streams:
                reply = _api_error(404, 10059, "stream not found")
            else:
                reply = _consumer_created(stream)
        if reply is None:
            raise nats.errors.TimeoutError
        return reply

    def _stream_info(self, stream: str) -> _WireMsg:
        self.infos += 1
        if self.declarer_returns:
            self.streams.setdefault(stream, _kv_stream_config(stream))
        if stream not in self.streams:
            return _api_error(404, 10059, "stream not found")
        reply = _WireMsg(
            json.dumps(
                {
                    "type": "io.nats.jetstream.api.v1.stream_info_response",
                    "config": self.streams[stream],
                    "state": {"messages": 0, "bytes": 0, "first_seq": 0, "last_seq": 0, "consumer_count": 0},
                    "created": "2026-10-02T00:00:00Z",
                }
            ).encode()
        )
        if self.vanish_after_infos is not None and self.infos >= self.vanish_after_infos:
            self.streams.clear()
        return reply

    def _direct_get(self, stream: str) -> _WireMsg:
        if stream not in self.streams:
            # no stream serves direct gets for it, so nothing answers: the server's 503 status
            raise nats.errors.NoRespondersError
        # the key has no message: the server answers a direct get with a 404 status and no body
        return _WireMsg(b"", headers={"Status": "404", "Description": "Message Not Found"})

    def _publish(self, bucket: str) -> _WireMsg:
        stream = f"KV_{bucket}"
        if stream not in self.streams or self.unacknowledged_publishes > 0:
            # no stream captures the subject, so nothing acknowledges it
            self.unacknowledged_publishes = max(0, self.unacknowledged_publishes - 1)
            raise nats.errors.NoRespondersError
        self.seq += 1
        return _WireMsg(json.dumps({"stream": stream, "seq": self.seq}).encode())


class _WireMsg:
    """one reply as nats-py's request returns it: a payload and its headers."""

    def __init__(self, data: bytes, *, headers: dict[str, str] | None = None) -> None:
        self.data = data
        self.headers = headers
        self.header = headers


def _consumer_created(stream: str) -> _WireMsg:
    """the server's answer to a consumer create on a stream holding no messages.

    :param stream: the stream the consumer was created on
    :ptype stream: str
    :return: the reply
    :rtype: _WireMsg
    """
    body = {
        "type": "io.nats.jetstream.api.v1.consumer_create_response",
        "name": "listing",
        "stream_name": stream,
        "config": {"ack_policy": "none", "deliver_policy": "last_per_subject"},
        "created": "2026-10-02T00:00:00Z",
        "num_pending": 0,
    }
    return _WireMsg(json.dumps(body).encode())


def _api_error(code: int, err_code: int, description: str) -> _WireMsg:
    """a JetStream API error document, as the server answers a refused API request.

    :param code: the HTTP-style status
    :ptype code: int
    :param err_code: JetStream's own error code
    :ptype err_code: int
    :param description: the server's description
    :ptype description: str
    :return: the reply
    :rtype: _WireMsg
    """
    body = {
        "type": "io.nats.jetstream.api.v1.error",
        "error": {"code": code, "err_code": err_code, "description": description},
    }
    return _WireMsg(json.dumps(body).encode())


def _client(wire: _FakeWire) -> NatsClient:
    """a wrapper client whose JetStream context is a real nats-py one over ``wire``.

    :param wire: the scripted server
    :ptype wire: _FakeWire
    :return: the client
    :rtype: NatsClient
    """
    return NatsClient(raw=wire, namespace=_NS, client_name="kv-not-found", kv_timings=_FAST)  # type: ignore[arg-type]


async def _raised_by(coro: Any) -> BaseException:
    """await ``coro`` and return what it raised.

    :param coro: the awaitable to run
    :ptype coro: Any
    :return: the exception
    :rtype: BaseException
    :raises AssertionError: when it raised nothing
    """
    try:
        await coro
    except BaseException as exc:  # noqa: BLE001 -- the exception IS the result under test
        return exc
    raise AssertionError("expected an exception, got none")


async def _bound_handle(wire: _FakeWire) -> NatsKvBucket:
    """bind the bucket on a server that holds it, through the client's real bind-only open.

    :param wire: the scripted server, which is given the bucket's stream
    :ptype wire: _FakeWire
    :return: the bound handle
    :rtype: NatsKvBucket
    """
    wire.streams[_STREAM] = _kv_stream_config(_STREAM)
    return await _client(wire).kv_bucket(name="nonces", create_if_missing=False)


class TestTheTypedError:
    """one error, a narrower kind of the one every existing catch already names."""

    def test_it_is_a_kv_error(self) -> None:
        error = KvBucketNotFoundError("absent", bucket=_BUCKET)
        assert isinstance(error, KvError)
        assert isinstance(error, NatsClientError)
        assert error.bucket == _BUCKET

    def test_it_is_exported_from_the_package_and_the_errors_module(self) -> None:
        from threetears.nats import errors

        assert errors.KvBucketNotFoundError is KvBucketNotFoundError
        assert "KvBucketNotFoundError" in errors.__all__


class TestABindOnlyOpenOfAnAbsentBucket:
    """the server answers not-found; once the wait for the declarer is spent, the error says so by type."""

    @pytest.mark.asyncio
    async def test_the_client_open_raises_the_typed_error(self) -> None:
        raised = await _raised_by(_client(_FakeWire()).kv_bucket(name="nonces", create_if_missing=False))
        assert isinstance(raised, KvBucketNotFoundError), repr(raised)
        assert raised.bucket == _BUCKET
        assert is_bucket_not_found(raised.__cause__ or raised), "the server's answer rides as the cause"

    @pytest.mark.asyncio
    async def test_a_bind_only_declaration_raises_the_typed_error(self) -> None:
        raised = await _raised_by(_client(_FakeWire()).ensure_kv_bucket(name="nonces", create_if_missing=False))
        assert isinstance(raised, KvBucketNotFoundError), repr(raised)
        assert raised.bucket == _BUCKET

    @pytest.mark.asyncio
    async def test_the_bucket_opener_raises_the_typed_error(self) -> None:
        raised = await _raised_by(
            NatsKvBucket.open(
                client=_client(_FakeWire()),
                full_name=_BUCKET,
                ttl=None,
                storage="memory",
                create_if_missing=False,
                history=1,
                timings=_FAST,
            )
        )
        assert isinstance(raised, KvBucketNotFoundError), repr(raised)

    @pytest.mark.asyncio
    async def test_a_bucket_that_vanishes_between_bind_and_config_check_raises_the_typed_error(self) -> None:
        # a declaration that binds reads the live config after the bind; the stream is gone by then.
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)
        wire.vanish_after_infos = 1
        raised = await _raised_by(_client(wire).ensure_kv_bucket(name="nonces", create_if_missing=False))
        assert isinstance(raised, KvBucketNotFoundError), repr(raised)

    @pytest.mark.asyncio
    async def test_a_refused_bind_stays_a_plain_kv_error(self) -> None:
        # never answered: an ungranted bucket and an unreachable broker look alike, and neither is absent.
        wire = _FakeWire()
        wire.unanswered = True
        raised = await _raised_by(_client(wire).kv_bucket(name="nonces", create_if_missing=False))
        assert isinstance(raised, KvError), repr(raised)
        assert not isinstance(raised, KvBucketNotFoundError), "a refusal is not an absence"


class TestADeclaringOpenThatCouldNotCreate:
    """a create nobody answered falls through to the bind; a bind that finds nothing means absent."""

    @pytest.mark.asyncio
    async def test_an_unanswered_create_and_an_absent_bucket_raise_the_typed_error(self) -> None:
        raised = await _raised_by(_client(_FakeWire()).ensure_kv_bucket(name="nonces"))
        assert isinstance(raised, KvBucketNotFoundError), repr(raised)
        assert raised.bucket == _BUCKET

    @pytest.mark.asyncio
    async def test_an_unanswered_create_and_an_unanswered_bind_stay_a_plain_kv_error(self) -> None:
        wire = _FakeWire()
        wire.unanswered = True
        raised = await _raised_by(_client(wire).ensure_kv_bucket(name="nonces"))
        assert isinstance(raised, KvError), repr(raised)
        assert not isinstance(raised, KvBucketNotFoundError)


class TestAnOperationOnAVanishedBucket:
    """a bound handle whose stream is gone re-binds once; when the bucket is still absent, it says so."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(lambda b: b.get(key="k"), id="get"),
            pytest.param(lambda b: b.get_entry(key="k"), id="get_entry"),
            pytest.param(lambda b: b.get_latest(key="k"), id="get_latest"),
            pytest.param(lambda b: b.put(key="k", value=b"v"), id="put"),
            pytest.param(lambda b: b.put(key="k", value=b"v", ttl=timedelta(seconds=5)), id="put_with_ttl"),
            pytest.param(lambda b: b.create(key="k", value=b"v"), id="create"),
            pytest.param(lambda b: b.update(key="k", value=b"v", revision=1), id="update"),
            pytest.param(
                lambda b: b.update(key="k", value=b"v", revision=1, ttl=timedelta(seconds=5)), id="update_with_ttl"
            ),
            pytest.param(lambda b: b.delete(key="k"), id="delete"),
            pytest.param(lambda b: b.date_created(), id="date_created"),
        ],
    )
    async def test_a_bind_only_handle_raises_the_typed_error(self, operation: Any) -> None:
        wire = _FakeWire()
        bucket = await _bound_handle(wire)
        wire.streams.clear()  # a NATS restart wiped memory storage, and the declarer is not back

        raised = await _raised_by(operation(bucket))

        assert isinstance(raised, KvBucketNotFoundError), repr(raised)
        assert raised.bucket == _BUCKET

    @pytest.mark.asyncio
    async def test_an_operation_that_fails_for_another_reason_stays_a_plain_kv_error(self) -> None:
        wire = _FakeWire()
        bucket = await _bound_handle(wire)
        wire.unanswered = True  # the broker stopped answering: not an absence

        raised = await _raised_by(bucket.date_created())

        assert isinstance(raised, KvError), repr(raised)
        assert not isinstance(raised, KvBucketNotFoundError)

    @pytest.mark.asyncio
    async def test_a_key_listing_over_a_vanished_stream_raises_the_typed_error(self) -> None:
        wire = _FakeWire()
        client = MagicMock()
        client.jetstream_context = MagicMock(return_value=wire.jetstream())
        client.raw = MagicMock(is_closed=False)
        client.raw.new_inbox = MagicMock(return_value="_INBOX.listing")
        client.raw.subscribe = AsyncMock(return_value=MagicMock(unsubscribe=AsyncMock()))
        bucket = NatsKvBucket(client=client, full_name=_BUCKET, kv=MagicMock(), ttl=None, timings=_FAST)

        raised = await _raised_by(bucket.list_keys())

        assert isinstance(raised, KvBucketNotFoundError), repr(raised)


class TestAKeyListingSelfHealsLikeEveryOtherOperation:
    """a listing whose stream vanished re-binds once, as get and put do, before it reports an absence."""

    @pytest.mark.asyncio
    async def test_a_declaring_handle_recreates_its_bucket_and_lists(self) -> None:
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)
        bucket = NatsKvBucket(
            client=_client(wire),
            full_name=_BUCKET,
            kv=await wire.jetstream().key_value(_BUCKET),
            ttl=None,
            timings=_FAST,
        )
        wire.streams.clear()  # a NATS restart took the stream
        wire.creates_streams = True

        assert await bucket.list_keys() == []
        assert _STREAM in wire.streams, "the declaring handle put its bucket back"

    @pytest.mark.asyncio
    async def test_a_bind_only_handle_lists_once_its_declarer_is_back(self) -> None:
        wire = _FakeWire()
        bucket = await _bound_handle(wire)
        wire.streams.clear()
        wire.declarer_returns = True

        assert await bucket.list_keys() == []


class TestAListingsReopenRunsOnItsOwnDeadline:
    """the re-open between two listing attempts waits for the declarer on its own clock.

    It used to run inside the listing's own deadline, and both default to 30s, so the listing's
    deadline always fired first: a bind-only handle whose declarer stayed away raised a plain
    ``KvError`` blaming the consumer-create grant -- a grant that was fine -- instead of saying the
    bucket is absent.
    """

    @pytest.mark.asyncio
    async def test_a_declarer_that_stays_away_is_reported_as_an_absence(self) -> None:
        listing_shorter_than_the_bind_wait = KvTimings(
            op_timeout_seconds=2.0,
            bind_wait_for_declarer_seconds=0.6,
            bind_retry_first_delay_seconds=0.01,
            bind_retry_max_delay_seconds=0.05,
            key_listing_timeout_seconds=0.2,
        )
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)
        client = NatsClient(
            raw=wire,  # type: ignore[arg-type]
            namespace=_NS,
            client_name="listing",
            kv_timings=listing_shorter_than_the_bind_wait,
        )
        bucket = await client.kv_bucket(name="nonces", create_if_missing=False)
        wire.streams.clear()  # the restart took it, and the declarer is not coming back

        raised = await _raised_by(bucket.list_keys())

        assert isinstance(raised, KvBucketNotFoundError), repr(raised)


class TestKvTimingsRefusesAScheduleItsBindCannotRun:
    """a bad timing is refused where the host builds it, naming the field -- never at a bind.

    The bind's wait for its declarer runs on ``retry_bounded``, which refuses a schedule it cannot
    back off on before its first attempt; a client carrying such timings would refuse every bind of a
    bucket that is right there.
    """

    @pytest.mark.parametrize(
        ("field", "overrides"),
        [
            ("bind_retry_first_delay_seconds", {"bind_retry_first_delay_seconds": 0.0}),
            (
                "bind_retry_max_delay_seconds",
                {"bind_retry_first_delay_seconds": 1.0, "bind_retry_max_delay_seconds": 0.5},
            ),
            ("bind_wait_for_declarer_seconds", {"bind_wait_for_declarer_seconds": -1.0}),
            ("op_timeout_seconds", {"op_timeout_seconds": 0.0}),
            ("key_listing_timeout_seconds", {"key_listing_timeout_seconds": 0.0}),
            ("timeout_remedy_log_interval_seconds", {"timeout_remedy_log_interval_seconds": -1.0}),
        ],
    )
    def test_it_is_refused_naming_the_field(self, field: str, overrides: dict[str, float]) -> None:
        with pytest.raises(ValueError, match=field):
            KvTimings(**overrides)

    def test_a_bind_wait_of_zero_binds_once(self) -> None:
        assert KvTimings(bind_wait_for_declarer_seconds=0.0).bind_wait_for_declarer_seconds == 0.0


class TestOneSlowOpenDoesNotStallTheOthers:
    """a bind-only open waiting for its declarer holds up callers of THAT bucket only."""

    @pytest.mark.asyncio
    async def test_an_unrelated_open_completes_while_an_absent_bucket_is_waited_for(self) -> None:
        wire = _FakeWire()
        wire.streams["KV_3tears-present"] = _kv_stream_config("KV_3tears-present")
        patient = KvTimings(
            op_timeout_seconds=2.0,
            bind_wait_for_declarer_seconds=3.0,
            bind_retry_first_delay_seconds=0.01,
            bind_retry_max_delay_seconds=0.05,
        )
        client = NatsClient(raw=wire, namespace=_NS, client_name="lock", kv_timings=patient)  # type: ignore[arg-type]
        waiting = asyncio.create_task(client.kv_bucket(name="absent", create_if_missing=False))
        await asyncio.sleep(0.1)
        try:
            bound = await asyncio.wait_for(client.kv_bucket(name="present", create_if_missing=False), timeout=1.0)
            again = await asyncio.wait_for(client.kv_bucket(name="present", create_if_missing=False), timeout=1.0)
            assert again is bound, "a cached handle is answered without waiting either"
            assert not waiting.done(), "the absent bucket is still being waited for"
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)


async def _until_refilled(refills: list[Any], count: int) -> None:
    """wait until the client's background refill has run ``count`` times, failing the test if it never does.

    :param refills: the buckets each refill was handed, appended as it runs
    :ptype refills: list[Any]
    :param count: how many refills to wait for
    :ptype count: int
    :return: nothing
    :rtype: None
    """
    for _ in range(500):
        if len(refills) >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected {count} refills, saw {len(refills)}")


class TestASelfHealThatCreatedTheBucketOwesItsRefill:
    """the path no reconnect runs: an operation finds the stream gone and its declaring handle creates it.

    A put after a wipe, or the first operation after a move to a successor connection, recreates the
    bucket through the handle's own re-open, and nothing else tells the declarer its entries are gone.
    """

    @pytest.mark.asyncio
    async def test_a_put_after_a_wipe_recreates_the_bucket_and_runs_the_refill_with_the_handle(self) -> None:
        wire = _FakeWire()
        wire.creates_streams = True
        refills: list[Any] = []

        async def _write_back(bucket: Any) -> None:
            refills.append(bucket)

        bucket = await _client(wire).ensure_kv_bucket(name="nonces", on_restored=_write_back)
        wire.streams.clear()  # a NATS restart took the stream, and no reconnect ran

        await bucket.put(key="k", value=b"v")

        assert _STREAM in wire.streams, "the declaring handle put its bucket back"
        await _until_refilled(refills, 1)
        assert refills == [bucket]

    @pytest.mark.asyncio
    async def test_a_reopen_that_found_the_stream_live_owes_nothing(self) -> None:
        """a write nothing acknowledged once is a transport failure, not a lost bucket."""
        wire = _FakeWire()
        wire.creates_streams = True
        refills: list[Any] = []

        async def _write_back(bucket: Any) -> None:
            refills.append(bucket)

        bucket = await _client(wire).ensure_kv_bucket(name="nonces", on_restored=_write_back)
        wire.unacknowledged_publishes = 1

        await bucket.put(key="k", value=b"v")
        for _ in range(50):
            await asyncio.sleep(0)

        assert refills == []

    @pytest.mark.asyncio
    async def test_a_raising_refill_after_a_self_heal_is_retried(self, caplog: pytest.LogCaptureFixture) -> None:
        wire = _FakeWire()
        wire.creates_streams = True
        refills: list[Any] = []

        async def _write_back(bucket: Any) -> None:
            refills.append(bucket)
            if len(refills) == 1:
                raise RuntimeError("nats: connection closed")

        bucket = await _client(wire).ensure_kv_bucket(name="nonces", on_restored=_write_back)
        wire.streams.clear()

        with caplog.at_level(logging.ERROR, logger="threetears.nats.client"):
            await bucket.put(key="k", value=b"v")
            await _until_refilled(refills, 2)

        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any(_BUCKET in message and "nats: connection closed" in message for message in errors), errors


class TestAKeyListingFollowsAMovedConnection:
    """a listing, like every other operation, binds its handle on the client's current connection first."""

    @staticmethod
    def _moved_handle(wire: _FakeWire) -> NatsKvBucket:
        """a declaring handle bound on a connection the client has since replaced with ``wire``.

        :param wire: the client's current connection, and the server behind it
        :ptype wire: _FakeWire
        :return: the handle
        :rtype: NatsKvBucket
        """
        client = MagicMock()
        client.jetstream_context = MagicMock(return_value=wire.jetstream())
        client.raw = wire
        return NatsKvBucket(
            client=client, full_name=_BUCKET, kv=MagicMock(), ttl=None, timings=_FAST, bound_to=object()
        )

    @pytest.mark.asyncio
    async def test_the_handle_is_bound_once_on_the_successor(self) -> None:
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)
        bucket = self._moved_handle(wire)
        infos_before = wire.infos

        assert await bucket.list_keys() == []
        assert wire.infos == infos_before + 1, "the listing bound the handle on the current connection"

        assert await bucket.list_keys() == []
        assert wire.infos == infos_before + 1, "a handle already on the current connection is not bound again"

    @pytest.mark.asyncio
    async def test_a_bucket_absent_on_the_successor_is_recreated_by_a_declaring_handle(self) -> None:
        wire = _FakeWire()
        wire.creates_streams = True
        bucket = self._moved_handle(wire)

        assert await bucket.list_keys() == []
        assert _STREAM in wire.streams

    @pytest.mark.asyncio
    async def test_a_bind_on_the_successor_that_is_never_answered_is_a_kv_error(self) -> None:
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)
        wire.unanswered = True
        bucket = self._moved_handle(wire)

        raised = await _raised_by(bucket.list_keys())

        assert isinstance(raised, KvError), repr(raised)
        assert not isinstance(raised, KvBucketNotFoundError), "an unanswered bind is not an absence"


async def _nats_py_raises(coro: Any) -> BaseException:
    """what nats-py itself raised for one request, answered by :class:`_FakeWire`.

    :param coro: a nats-py call
    :ptype coro: Any
    :return: the exception nats-py raised
    :rtype: BaseException
    """
    return await _raised_by(coro)


class TestTheRawHandleClassifiers:
    """a consumer holding a raw nats-py handle classifies its failures by type, without importing nats-py."""

    @pytest.mark.asyncio
    async def test_binding_an_absent_bucket_is_bucket_not_found(self) -> None:
        raised = await _nats_py_raises(_FakeWire().jetstream().key_value(_BUCKET))
        assert is_bucket_not_found(raised)
        assert is_nats_error(raised)
        assert not is_key_not_found(raised)

    @pytest.mark.asyncio
    async def test_reading_an_absent_streams_info_is_bucket_not_found(self) -> None:
        raised = await _nats_py_raises(_FakeWire().jetstream().stream_info(_STREAM))
        assert is_bucket_not_found(raised)
        assert is_nats_error(raised)

    @pytest.mark.asyncio
    async def test_writing_to_an_absent_bucket_is_bucket_not_found(self) -> None:
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)
        kv = await wire.jetstream().key_value(_BUCKET)
        wire.streams.clear()
        raised = await _nats_py_raises(kv.put("k", b"v"))
        assert is_bucket_not_found(raised)
        assert is_nats_error(raised)

    @pytest.mark.asyncio
    async def test_a_missing_key_is_key_not_found_and_not_bucket_not_found(self) -> None:
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)
        kv = await wire.jetstream().key_value(_BUCKET)
        raised = await _nats_py_raises(kv.get("k"))
        assert is_key_not_found(raised)
        assert not is_bucket_not_found(raised), "a missing key says the bucket exists"
        assert is_nats_error(raised)

    @pytest.mark.asyncio
    async def test_an_unanswered_request_is_a_nats_error_and_not_an_absence(self) -> None:
        wire = _FakeWire()
        wire.unanswered = True
        raised = await _nats_py_raises(wire.jetstream().key_value(_BUCKET))
        assert is_nats_error(raised)
        assert not is_bucket_not_found(raised), "a refused bucket and a dead broker are not absent ones"
        assert not is_key_not_found(raised)

    @pytest.mark.asyncio
    async def test_a_direct_get_nobody_answers_is_not_claimed_as_an_absence(self) -> None:
        # a vanished direct-get bucket answers this way, and so does any subject nobody serves.
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)
        kv = await wire.jetstream().key_value(_BUCKET)
        wire.streams.clear()
        raised = await _nats_py_raises(kv.get("k"))
        assert is_nats_error(raised)
        assert not is_bucket_not_found(raised)

    @pytest.mark.asyncio
    async def test_another_not_found_api_answer_is_not_a_missing_stream(self) -> None:
        # a 404 with a different JetStream code -- a consumer the server does not have -- shares the
        # class with a missing stream and must not be read as one.
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)

        async def _consumer_missing(subject: str, payload: bytes = b"", timeout: float | None = None) -> _WireMsg:
            del subject, payload, timeout
            return _api_error(404, 10014, "consumer not found")

        wire.request = _consumer_missing  # type: ignore[method-assign]
        raised = await _nats_py_raises(wire.jetstream().consumer_info(_STREAM, "durable"))
        assert is_nats_error(raised)
        assert not is_bucket_not_found(raised)
        assert not is_key_not_found(raised)

    def test_the_wrappers_own_absence_is_bucket_not_found_and_not_a_nats_error(self) -> None:
        error = KvBucketNotFoundError("absent", bucket=_BUCKET)
        assert is_bucket_not_found(error)
        assert not is_nats_error(error), "the wrapper already translated it"
        assert not is_key_not_found(error)

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(KvError("kv down"), id="plain-kv-error"),
            pytest.param(RuntimeError("stream not found"), id="unrelated-with-a-matching-message"),
            pytest.param(TimeoutError("nats: timeout"), id="builtin-timeout"),
            pytest.param(KeyError("KeyNotFoundError"), id="unrelated-named-like-one"),
        ],
    )
    def test_unrelated_exceptions_are_none_of_these(self, error: BaseException) -> None:
        assert not is_bucket_not_found(error)
        assert not is_key_not_found(error)
        assert not is_nats_error(error)


class TestARefusalDuringAnOperationsRebindKeepsItsType:
    """an operation that re-binds its bucket and is refused there surfaces the refusal, never a KvError.

    ``KvConfigMismatch`` and ``StreamSubjectsOverlapError`` are deliberately NOT ``KvError``s: the L2
    accessors catch ``KvError`` and degrade, so a refusal raised as one is downgraded to a warning and
    the process runs on against a bucket it refuses. The open raises them as themselves; an operation
    whose self-heal re-binds the bucket used to wrap them into a plain ``KvError``.
    """

    @pytest.mark.asyncio
    async def test_a_config_mismatch_found_on_rebind_is_raised_as_itself(self) -> None:
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)
        bucket = await _client(wire).ensure_kv_bucket(name="nonces", create_if_missing=False, direct=True)
        # the bucket was recreated without direct gets, and this write is the first to notice
        wire.streams[_STREAM] = {**_kv_stream_config(_STREAM), "allow_direct": False}
        wire.unacknowledged_publishes = 1

        raised = await _raised_by(bucket.put(key="k", value=b"v"))

        assert isinstance(raised, KvConfigMismatch), repr(raised)
        assert not isinstance(raised, KvError)

    @pytest.mark.asyncio
    async def test_a_subjects_overlap_found_on_recreate_is_raised_as_itself(self) -> None:
        wire = _FakeWire()
        wire.streams[_STREAM] = _kv_stream_config(_STREAM)
        # a handle that may declare (create_if_missing defaults to True), bound while the stream exists
        bucket = NatsKvBucket(
            client=_client(wire),
            full_name=_BUCKET,
            kv=await wire.jetstream().key_value(_BUCKET),
            ttl=None,
            timings=_FAST,
        )
        # the stream is gone, and another stream now owns its subjects
        wire.streams.clear()
        wire.create_reply = _api_error(400, 10065, "subjects overlap with an existing stream")

        raised = await _raised_by(bucket.put(key="k", value=b"v"))

        assert isinstance(raised, StreamSubjectsOverlapError), repr(raised)
        assert not isinstance(raised, KvError)
