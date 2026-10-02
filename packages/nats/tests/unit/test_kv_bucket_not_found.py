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
        elif subject.startswith("$KV."):
            reply = self._publish(subject.split(".")[1])
        elif subject.startswith("$JS.API.CONSUMER.CREATE."):
            stream = subject.removeprefix("$JS.API.CONSUMER.CREATE.").split(".", 1)[0]
            if stream not in self.streams:
                reply = _api_error(404, 10059, "stream not found")
        if reply is None:
            raise nats.errors.TimeoutError
        return reply

    def _stream_info(self, stream: str) -> _WireMsg:
        self.infos += 1
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
        if stream not in self.streams:
            # no stream captures the subject, so nothing acknowledges it
            raise nats.errors.NoRespondersError
        self.seq += 1
        return _WireMsg(json.dumps({"stream": stream, "seq": self.seq}).encode())


class _WireMsg:
    """one reply as nats-py's request returns it: a payload and its headers."""

    def __init__(self, data: bytes, *, headers: dict[str, str] | None = None) -> None:
        self.data = data
        self.headers = headers
        self.header = headers


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
