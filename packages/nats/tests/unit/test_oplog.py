"""unit tests for :mod:`threetears.nats.oplog` (no broker).

Covers everything testable without a live JetStream server: subject + stream
name construction and sanitization, the published header dict (CAS + dedup),
the :class:`AppendResult` / :class:`OpRecord` value shapes, and input
validation. The full broker-backed contract (CAS fence, dedup, ordered
terminating replay) is the integration proof
``tests/integration/test_oplog_round_trip.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from nats.errors import ConnectionClosedError as NatsConnectionClosedError
from nats.errors import TimeoutError as NatsTimeoutError
from nats.aio.msg import Msg
from nats.js.api import Header, StorageType
from nats.js.errors import APIError

from threetears.nats import (
    AppendResult,
    OpLog,
    OpLogError,
    OpLogSequenceConflict,
    OpRecord,
    Subjects,
    set_default_namespace,
)


@pytest.fixture(autouse=True)
def _reset_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """each test starts on the documented default namespace."""
    monkeypatch.delenv("THREETEARS_NATS_SUBJECT_NAMESPACE", raising=False)
    set_default_namespace("3tears")


# ----------------------------------------------------------------------
# fakes — capture what the wrapper hands to nats-py without a broker
# ----------------------------------------------------------------------


# parity-exempt: fakes the external nats.js.api.PubAck ack object (seq + duplicate fields only); not a threetears protocol
class _FakePubAck:
    def __init__(self, seq: int, duplicate: bool | None) -> None:
        self.seq = seq
        self.duplicate = duplicate


# parity-exempt: fakes the external nats JetStreamContext (add_stream/publish subset with scripted acks); external lib, no threetears parity
class _FakeJetStream:
    """records add_stream/publish calls; returns scripted acks/errors."""

    def __init__(self) -> None:
        self.add_stream_config: Any = None
        self.publish_calls: list[dict[str, Any]] = []
        self._next_ack: _FakePubAck | None = None
        self._raise_on_publish: BaseException | None = None

    async def add_stream(self, *, config: Any) -> Any:
        self.add_stream_config = config
        return config

    def script_publish(self, *, ack: _FakePubAck | None = None, error: BaseException | None = None) -> None:
        self._next_ack = ack
        self._raise_on_publish = error

    async def publish(
        self, subject: str, payload: bytes, *, headers: dict[str, str], stream: str, timeout: float
    ) -> Any:
        # `timeout` is required rather than defaulted: the real call always carries one now, and a
        # default here would let the append lose its bound without a single test noticing.
        self.publish_calls.append(
            {"subject": subject, "payload": payload, "headers": headers, "stream": stream, "timeout": timeout}
        )
        if self._raise_on_publish is not None:
            raise self._raise_on_publish
        assert self._next_ack is not None
        return self._next_ack


# parity-exempt: fakes the external nats Client wrapper handing back the fake jetstream; external lib, no threetears parity
class _FakeClient:
    def __init__(self, js: _FakeJetStream) -> None:
        self._js = js

    def jetstream_context(self) -> _FakeJetStream:
        return self._js


def _make_oplog(js: _FakeJetStream, *, repo: str = "story-alpha", branch: str = "main") -> OpLog:
    """construct an OpLog over the fake client without going through open()."""
    subject = str(Subjects.oplog(repo, branch))
    return OpLog(client=_FakeClient(js), stream=subject.replace(".", "-"), subject=subject)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# subject + stream name construction / sanitization
# ----------------------------------------------------------------------


def test_oplog_subject_is_namespace_prefixed() -> None:
    sub = Subjects.oplog("story-alpha", "main")
    assert str(sub) == "3tears.oplog.story-alpha.main"
    assert sub.kind == "point"


def test_oplog_subject_sanitizes_dotted_segments() -> None:
    """dots in repo/branch are replaced so they don't overload the separator."""
    sub = Subjects.oplog("my.org/repo", "feature.x")
    assert str(sub) == "3tears.oplog.my-org/repo.feature-x"


@pytest.mark.parametrize(("repo", "branch"), [("", "main"), ("repo", "")])
def test_oplog_subject_rejects_empty_segments(repo: str, branch: str) -> None:
    with pytest.raises(ValueError):
        Subjects.oplog(repo, branch)


async def test_open_builds_dot_free_stream_name_and_memory_r3_config() -> None:
    """stream name has no dots; config is in-memory, R3, single-subject, dedup window."""
    js = _FakeJetStream()
    oplog = await OpLog.open(client=_FakeClient(js), repo="story.alpha", branch="release.1")  # type: ignore[arg-type]

    assert oplog.subject == "3tears.oplog.story-alpha.release-1"
    assert oplog.stream == "3tears-oplog-story-alpha-release-1"
    assert "." not in oplog.stream

    cfg = js.add_stream_config
    assert cfg.name == oplog.stream
    assert cfg.subjects == [oplog.subject]
    assert cfg.storage == StorageType.MEMORY
    assert cfg.num_replicas == 3
    assert cfg.duplicate_window == 300.0


@pytest.mark.parametrize(("repo", "branch"), [("", "main"), ("repo", "")])
async def test_open_rejects_empty_repo_or_branch(repo: str, branch: str) -> None:
    js = _FakeJetStream()
    with pytest.raises(ValueError):
        await OpLog.open(client=_FakeClient(js), repo=repo, branch=branch)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# append — header assembly + result shape + validation
# ----------------------------------------------------------------------


async def test_append_sends_cas_and_dedup_headers() -> None:
    js = _FakeJetStream()
    js.script_publish(ack=_FakePubAck(seq=1, duplicate=None))
    oplog = _make_oplog(js)

    result = await oplog.append(payload=b"edit", op_id="op-1", expected_last_seq=0)

    assert len(js.publish_calls) == 1
    call = js.publish_calls[0]
    assert call["payload"] == b"edit"
    assert call["headers"][Header.EXPECTED_LAST_SEQUENCE] == "0"
    assert call["headers"][Header.MSG_ID] == "op-1"
    assert call["stream"] == oplog.stream
    assert result == AppendResult(seq=1, deduplicated=False)


async def test_append_reports_native_dedup_as_deduplicated() -> None:
    """when the server returns PubAck(duplicate=True) the result flags it."""
    js = _FakeJetStream()
    js.script_publish(ack=_FakePubAck(seq=7, duplicate=True))
    oplog = _make_oplog(js)

    result = await oplog.append(payload=b"edit", op_id="op-1", expected_last_seq=7)

    assert result == AppendResult(seq=7, deduplicated=True)


async def test_append_rejects_empty_op_id() -> None:
    js = _FakeJetStream()
    oplog = _make_oplog(js)
    with pytest.raises(ValueError):
        await oplog.append(payload=b"x", op_id="", expected_last_seq=0)
    assert js.publish_calls == []  # validated before any publish


async def test_append_rejects_negative_expected_last_seq() -> None:
    js = _FakeJetStream()
    oplog = _make_oplog(js)
    with pytest.raises(ValueError):
        await oplog.append(payload=b"x", op_id="op-1", expected_last_seq=-1)
    assert js.publish_calls == []


async def test_append_wraps_non_cas_apierror_as_oplogerror() -> None:
    """a JetStream error that is NOT wrong-last-sequence is not mistaken for a CAS conflict."""
    from nats.js.errors import APIError

    js = _FakeJetStream()
    # err_code 10059 is e.g. a stream-not-found style error -- anything but 10071.
    js.script_publish(error=APIError(code=404, err_code=10059, description="stream not found"))
    oplog = _make_oplog(js)

    with pytest.raises(OpLogError) as exc_info:
        await oplog.append(payload=b"x", op_id="op-1", expected_last_seq=0)
    # specifically NOT the CAS-conflict subclass
    assert not isinstance(exc_info.value, OpLogSequenceConflict)


async def test_append_wraps_transport_failure_as_oplogerror() -> None:
    js = _FakeJetStream()
    js.script_publish(error=TimeoutError("no response from stream"))
    oplog = _make_oplog(js)
    with pytest.raises(OpLogError):
        await oplog.append(payload=b"x", op_id="op-1", expected_last_seq=0)


# ----------------------------------------------------------------------
# replay — input validation + no-silent-truncation
# ----------------------------------------------------------------------


async def test_replay_rejects_from_seq_below_one() -> None:
    js = _FakeJetStream()
    oplog = _make_oplog(js)
    with pytest.raises(ValueError):
        async for _ in oplog.replay(from_seq=0):
            pass


# parity-exempt: fakes the external nats.js.api.StreamState (last_seq field only); not a threetears protocol
class _FakeStreamState:
    def __init__(self, last_seq: int, first_seq: int = 1) -> None:
        self.last_seq = last_seq
        self.first_seq = first_seq


# parity-exempt: fakes the external nats.js.api.StreamInfo (first_seq and last_seq fields only); not a threetears protocol
class _FakeStreamInfo:
    def __init__(self, last_seq: int, first_seq: int = 1) -> None:
        self.state = _FakeStreamState(last_seq, first_seq)


# parity-exempt: fakes the external nats pull subscription raising a scripted fetch error; external lib, no threetears parity
class _FakePullSub:
    def __init__(self, fetch_error: BaseException) -> None:
        self._fetch_error = fetch_error
        self.unsubscribed = False

    async def fetch(self, batch: int, timeout: float) -> list[Any]:
        raise self._fetch_error

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


class _ReplayJetStream:
    """fake JS that drives replay to the fetch loop with a scripted fetch error."""

    def __init__(self, *, last_seq: int, fetch_error: BaseException, first_seq: int = 1) -> None:
        self._last_seq = last_seq
        self._first_seq = first_seq
        self.psub: Any = _FakePullSub(fetch_error)
        self.created: list[tuple[str, Any]] = []
        self.bound: list[tuple[str, str]] = []

    async def stream_info(self, name: str) -> _FakeStreamInfo:
        return _FakeStreamInfo(self._last_seq, self._first_seq)

    async def add_consumer(self, stream: str, *, config: Any) -> Any:
        self.created.append((stream, config))
        return object()

    async def pull_subscribe_bind(self, *, consumer: str, stream: str) -> Any:
        self.bound.append((consumer, stream))
        return self.psub


# parity-exempt: fakes the external nats pull subscription handing back one scripted batch, then the drained-tail timeout; external lib, no threetears parity
class _ScriptedPullSub:
    def __init__(self, batch: list[Any]) -> None:
        self._batches = [batch]
        self.unsubscribed = False

    async def fetch(self, batch: int, timeout: float) -> list[Any]:
        if self._batches:
            return self._batches.pop(0)
        raise TimeoutError("nats: timeout")

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


def _fetched(seq: int, payload: bytes, *, consumer: str, stream: str) -> Msg:
    """a message as a pull fetch hands it over, its stream sequence in the ack reply."""
    reply = f"$JS.ACK.{stream}.{consumer}.1.{seq}.{seq}.1700000000000000000.0"
    return Msg(_client=None, subject="oplog", reply=reply, data=payload, headers={"Nats-Msg-Id": f"op-{seq}"})  # type: ignore[arg-type]


def _published(payload: bytes) -> Msg:
    """bytes a plain publish put on the fetch inbox: no reply at all."""
    return Msg(_client=None, subject="oplog", reply="", data=payload, headers=None)  # type: ignore[arg-type]


class _OwnConsumerJetStream(_ReplayJetStream):
    """fake JS whose fetch inbox receives forgeries beside the replay consumer's own deliveries."""

    async def pull_subscribe_bind(self, *, consumer: str, stream: str) -> Any:
        self.bound.append((consumer, stream))
        self.psub = _ScriptedPullSub(
            [
                _fetched(1, b"forged by another consumer", consumer="forger-consumer", stream=stream),
                _fetched(1, b"forged on another stream", consumer=consumer, stream="OBJ_forger"),
                _published(b"forged by a plain publish"),
                _fetched(1, b"op one", consumer=consumer, stream=stream),
                _fetched(2, b"op two", consumer=consumer, stream=stream),
            ]
        )
        return self.psub


async def test_replay_reads_only_its_own_randomly_named_consumer() -> None:
    """a forged op on the fetch inbox is never replayed; the logged ops are.

    any client in the account can get bytes delivered onto the fetch inbox. a fetched message is an
    op only when it carries the ack reply of the replay's own consumer, whose name is random.
    """
    js = _OwnConsumerJetStream(last_seq=2, fetch_error=TimeoutError("unused"))
    oplog = _make_oplog(js)  # type: ignore[arg-type]

    replayed = [record async for record in oplog.replay(from_seq=1)]

    assert [(r.seq, r.payload) for r in replayed] == [(1, b"op one"), (2, b"op two")]
    ((created_stream, config),) = js.created
    assert created_stream == oplog.stream
    assert config.name.startswith("oplog-replay-") and len(config.name.removeprefix("oplog-replay-")) == 32
    assert config.filter_subject == oplog.subject
    assert js.bound == [(config.name, oplog.stream)]
    assert js.psub.unsubscribed


async def test_replay_surfaces_transport_error_never_silently_truncates() -> None:
    """A non-timeout fetch failure mid-replay raises OpLogError, not a clean (truncated) stop.

    Silent truncation of a WAL replay would corrupt materialise / failover -- the durability
    invariant the write path must never break. A drained tail (TimeoutError) is the only clean
    terminator; anything else must surface.
    """
    js = _ReplayJetStream(last_seq=5, fetch_error=ConnectionResetError("broker dropped mid-replay"))
    oplog = _make_oplog(js)  # type: ignore[arg-type]

    with pytest.raises(OpLogError):
        async for _ in oplog.replay(from_seq=1):
            pass
    # the ephemeral replay consumer is still torn down on the error path
    assert js.psub.unsubscribed is True


class _ShortTailJetStream(_ReplayJetStream):
    """fake JS whose replay consumer delivers ops 1 and 2 of a five-op stream, then times out."""

    def __init__(self, *, last_seq: int) -> None:
        super().__init__(last_seq=last_seq, fetch_error=TimeoutError("unused"))
        self.publish_error: BaseException | None = None

    async def pull_subscribe_bind(self, *, consumer: str, stream: str) -> Any:
        self.bound.append((consumer, stream))
        self.psub = _ScriptedPullSub(
            [
                _fetched(1, b"op one", consumer=consumer, stream=stream),
                _fetched(2, b"op two", consumer=consumer, stream=stream),
            ]
        )
        return self.psub

    async def publish(
        self, subject: str, payload: bytes, *, headers: dict[str, str], stream: str, timeout: float
    ) -> Any:
        assert self.publish_error is not None
        raise self.publish_error


async def test_a_replay_that_times_out_short_of_the_head_raises_rather_than_ending_clean() -> None:
    """a fetch timeout before ``last_seq`` arrives ended the iteration, and a short replay read as complete.

    the ops delivered before the timeout are still yielded; the iteration then raises instead of ending.
    """
    js = _ShortTailJetStream(last_seq=5)
    oplog = _make_oplog(js)  # type: ignore[arg-type]

    replayed: list[OpRecord] = []
    with pytest.raises(OpLogError, match="short of the stream head") as exc_info:
        async for record in oplog.replay(from_seq=1):
            replayed.append(record)

    assert [r.seq for r in replayed] == [1, 2]
    assert "last_yielded_seq=2 last_seq=5" in str(exc_info.value)
    assert js.psub.unsubscribed is True


async def test_a_replay_that_delivers_nothing_before_timing_out_raises() -> None:
    """a stream whose head is 5 and a replay that reads none of it is not an empty replay."""
    js = _ReplayJetStream(last_seq=5, fetch_error=TimeoutError("nats: timeout"))
    oplog = _make_oplog(js)  # type: ignore[arg-type]

    with pytest.raises(OpLogError, match="short of the stream head"):
        _ = [r async for r in oplog.replay(from_seq=1)]
    assert js.psub.unsubscribed is True


async def test_a_cas_retry_whose_op_id_scan_ends_short_is_not_reported_as_a_fence() -> None:
    """append scans the log for its op_id on a wrong-last-sequence error; a scan that stopped short
    reported "not logged", so a retry of an op the log already holds was raised as a genuine fence."""
    from nats.js.errors import APIError

    js = _ShortTailJetStream(last_seq=5)
    js.publish_error = APIError(code=400, err_code=10071, description="wrong last sequence: 5")
    oplog = _make_oplog(js)  # type: ignore[arg-type]

    with pytest.raises(OpLogError, match="short of the stream head") as exc_info:
        await oplog.append(payload=b"x", op_id="op-5", expected_last_seq=2)
    assert not isinstance(exc_info.value, OpLogSequenceConflict)


# parity-exempt: fakes the external nats pull subscription answering each fetch with a scripted batch or timeout; external lib, no threetears parity
class _SteppedPullSub:
    def __init__(self, steps: list[list[Any] | BaseException]) -> None:
        self._steps = steps
        self.unsubscribed = False

    async def fetch(self, batch: int, timeout: float) -> list[Any]:
        step = self._steps.pop(0) if self._steps else TimeoutError("nats: timeout")
        if isinstance(step, BaseException):
            raise step
        return step

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


class _SteppedJetStream(_ReplayJetStream):
    """fake JS whose replay consumer answers fetches from a script of ``(seqs | timeout)`` steps."""

    def __init__(self, *, last_seq: int, first_seq: int = 1, steps: list[list[int] | BaseException]) -> None:
        super().__init__(last_seq=last_seq, first_seq=first_seq, fetch_error=TimeoutError("unused"))
        self._steps = steps
        self.publish_error: BaseException | None = None

    async def pull_subscribe_bind(self, *, consumer: str, stream: str) -> Any:
        self.bound.append((consumer, stream))
        self.psub = _SteppedPullSub(
            [
                step
                if isinstance(step, BaseException)
                else [_fetched(seq, f"op {seq}".encode(), consumer=consumer, stream=stream) for seq in step]
                for step in self._steps
            ]
        )
        return self.psub

    async def publish(
        self, subject: str, payload: bytes, *, headers: dict[str, str], stream: str, timeout: float
    ) -> Any:
        assert self.publish_error is not None
        raise self.publish_error


async def test_one_fetch_timeout_is_ridden_out_before_a_replay_is_called_short() -> None:
    """a single slow 2s fetch mid-replay is a broker under load, not a gap; two in a row raise."""
    js = _SteppedJetStream(last_seq=5, steps=[[1, 2], TimeoutError("nats: timeout"), [3, 4, 5]])
    oplog = _make_oplog(js)  # type: ignore[arg-type]

    replayed = [record.seq async for record in oplog.replay(from_seq=1)]

    assert replayed == [1, 2, 3, 4, 5]


async def test_a_replay_from_before_the_streams_first_seq_names_the_purged_gap() -> None:
    """a purged or trimmed start timed out as "short of the head", or replayed only what was left."""
    js = _SteppedJetStream(last_seq=9, first_seq=6, steps=[[6, 7, 8, 9]])
    oplog = _make_oplog(js)  # type: ignore[arg-type]

    with pytest.raises(OpLogError, match=r"holds seqs 6\.\.9; 2\.\.5 were purged or trimmed"):
        _ = [r async for r in oplog.replay(from_seq=2)]
    assert js.created == [], "a consumer was opened on a replay that could not start where asked"


async def test_the_append_op_id_scan_reads_from_the_first_seq_the_stream_still_holds() -> None:
    """a trimmed start is not a gap for "is this op already logged": the scan covers what is left."""
    from nats.js.errors import APIError

    js = _SteppedJetStream(last_seq=4, first_seq=3, steps=[[3, 4]])
    js.publish_error = APIError(code=400, err_code=10071, description="wrong last sequence: 4")
    oplog = _make_oplog(js)  # type: ignore[arg-type]

    result = await oplog.append(payload=b"x", op_id="op-4", expected_last_seq=2)

    assert result == AppendResult(seq=4, deduplicated=True)
    ((_stream, config),) = js.created
    assert config.opt_start_seq == 3


# ----------------------------------------------------------------------
# value-object shapes
# ----------------------------------------------------------------------


def test_append_result_is_frozen_value() -> None:
    r = AppendResult(seq=3, deduplicated=False)
    assert (r.seq, r.deduplicated) == (3, False)
    with pytest.raises((AttributeError, Exception)):
        r.seq = 4  # type: ignore[misc]


def test_op_record_carries_seq_payload_op_id() -> None:
    rec = OpRecord(seq=2, payload=b"op", op_id="op-2")
    assert (rec.seq, rec.payload, rec.op_id) == (2, b"op", "op-2")
    with pytest.raises((AttributeError, Exception)):
        rec.payload = b"changed"  # type: ignore[misc]


# ----------------------------------------------------------------------
# last_seq — the stream head (for fence / committed-through reconciliation)
# ----------------------------------------------------------------------


class _StreamInfoJetStream:
    """fake JS exposing only ``stream_info`` — the surface ``last_seq()`` reads."""

    def __init__(self, *, last_seq: int | None = None, error: BaseException | None = None) -> None:
        self._last_seq = last_seq
        self._error = error

    async def stream_info(self, name: str) -> _FakeStreamInfo:
        if self._error is not None:
            raise self._error
        assert self._last_seq is not None
        return _FakeStreamInfo(self._last_seq)


async def test_last_seq_returns_the_stream_head() -> None:
    oplog = _make_oplog(_StreamInfoJetStream(last_seq=8))  # type: ignore[arg-type]
    assert await oplog.last_seq() == 8


async def test_last_seq_is_zero_for_an_empty_stream() -> None:
    """A fresh/empty stream reports head 0 — the value a reset op-log carries (the desync case)."""
    oplog = _make_oplog(_StreamInfoJetStream(last_seq=0))  # type: ignore[arg-type]
    assert await oplog.last_seq() == 0


async def test_last_seq_wraps_stream_info_failure_as_oplogerror() -> None:
    oplog = _make_oplog(_StreamInfoJetStream(error=ConnectionResetError("broker down")))  # type: ignore[arg-type]
    with pytest.raises(OpLogError):
        await oplog.last_seq()


# ----------------------------------------------------------------------
# a bug is not a broker failure
# ----------------------------------------------------------------------

_BUG = TypeError("a bug in the call")


def _wrong_last_sequence() -> APIError:
    return APIError(code=400, err_code=10071, description="wrong last sequence: 5")


# parity-exempt: fakes the external nats JetStreamContext; each call the op-log makes raises what the test scripts, to tell nats-py's own failures from a bug
class _ScriptedFailureJetStream:
    def __init__(
        self,
        *,
        add_stream: list[BaseException | None] | None = None,
        stream_info: BaseException | None = None,
        add_consumer: BaseException | None = None,
        publish: BaseException | None = None,
    ) -> None:
        self._add_stream = list(add_stream or [])
        self._stream_info = stream_info
        self._add_consumer = add_consumer
        self._publish = publish
        self.infos = 0

    async def add_stream(self, *, config: Any) -> Any:
        step = self._add_stream.pop(0) if self._add_stream else None
        if step is not None:
            raise step
        return config

    async def stream_info(self, name: str) -> _FakeStreamInfo:
        self.infos += 1
        if self._stream_info is not None:
            raise self._stream_info
        return _FakeStreamInfo(5)

    async def add_consumer(self, stream: str, *, config: Any) -> Any:
        if self._add_consumer is not None:
            raise self._add_consumer
        return object()

    async def pull_subscribe_bind(self, *, consumer: str, stream: str) -> Any:
        return _FakePullSub(NatsTimeoutError())

    async def publish(
        self, subject: str, payload: bytes, *, headers: dict[str, str], stream: str, timeout: float
    ) -> Any:
        assert self._publish is not None
        raise self._publish


async def _open(js: _ScriptedFailureJetStream) -> OpLog:
    return await OpLog.open(client=_FakeClient(js), repo="story-alpha", branch="main")  # type: ignore[arg-type]


class TestOpeningTellsABugFromABrokerFailure:
    async def test_an_unanswered_create_falls_through_to_the_bind(self) -> None:
        js = _ScriptedFailureJetStream(add_stream=[NatsTimeoutError()])
        await _open(js)
        assert js.infos == 1

    async def test_a_bug_in_the_create_propagates_without_a_bind(self) -> None:
        js = _ScriptedFailureJetStream(add_stream=[_BUG])
        with pytest.raises(TypeError, match="a bug in the call"):
            await _open(js)
        assert js.infos == 0

    async def test_a_bug_in_the_single_replica_retry_propagates_without_a_bind(self) -> None:
        unclustered = APIError(code=500, err_code=10074, description="replicas > 1 not supported in non-clustered mode")
        js = _ScriptedFailureJetStream(add_stream=[unclustered, _BUG])
        with pytest.raises(TypeError, match="a bug in the call"):
            await _open(js)
        assert js.infos == 0

    async def test_a_refused_single_replica_retry_falls_through_to_the_bind(self) -> None:
        unclustered = APIError(code=500, err_code=10074, description="replicas > 1 not supported in non-clustered mode")
        js = _ScriptedFailureJetStream(add_stream=[unclustered, NatsTimeoutError()])
        await _open(js)
        assert js.infos == 1

    async def test_an_unanswered_bind_after_a_failed_create_is_an_oplogerror(self) -> None:
        js = _ScriptedFailureJetStream(add_stream=[NatsTimeoutError()], stream_info=NatsTimeoutError())
        with pytest.raises(OpLogError, match="open op-log failed"):
            await _open(js)

    async def test_a_bind_reply_that_is_not_json_is_an_oplogerror(self) -> None:
        # nats-py's own json.loads, on a JetStream API reply something other than JetStream answered
        js = _ScriptedFailureJetStream(
            add_stream=[NatsTimeoutError()], stream_info=json.JSONDecodeError("Expecting value", "<html>", 0)
        )
        with pytest.raises(OpLogError, match="open op-log failed"):
            await _open(js)

    async def test_a_bug_in_the_bind_propagates(self) -> None:
        js = _ScriptedFailureJetStream(add_stream=[NatsTimeoutError()], stream_info=_BUG)
        with pytest.raises(TypeError, match="a bug in the call"):
            await _open(js)


class TestAppendingTellsABugFromABrokerFailure:
    async def test_a_closed_connection_is_an_oplogerror(self) -> None:
        oplog = _make_oplog(_ScriptedFailureJetStream(publish=NatsConnectionClosedError()))  # type: ignore[arg-type]
        with pytest.raises(OpLogError, match="op-log append failed"):
            await oplog.append(payload=b"x", op_id="op-1", expected_last_seq=0)

    async def test_a_bug_in_the_publish_propagates(self) -> None:
        oplog = _make_oplog(_ScriptedFailureJetStream(publish=_BUG))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="a bug in the call"):
            await oplog.append(payload=b"x", op_id="op-1", expected_last_seq=0)

    async def test_an_unanswered_op_id_scan_is_an_oplogerror(self) -> None:
        js = _ScriptedFailureJetStream(publish=_wrong_last_sequence(), stream_info=NatsTimeoutError())
        with pytest.raises(OpLogError, match="op_id scan failed"):
            await _make_oplog(js).append(payload=b"x", op_id="op-1", expected_last_seq=2)  # type: ignore[arg-type]

    async def test_a_bug_in_the_op_id_scan_propagates(self) -> None:
        js = _ScriptedFailureJetStream(publish=_wrong_last_sequence(), stream_info=_BUG)
        with pytest.raises(TypeError, match="a bug in the call"):
            await _make_oplog(js).append(payload=b"x", op_id="op-1", expected_last_seq=2)  # type: ignore[arg-type]


class TestReadingTellsABugFromABrokerFailure:
    async def test_a_bug_in_the_head_read_propagates(self) -> None:
        oplog = _make_oplog(_StreamInfoJetStream(error=_BUG))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="a bug in the call"):
            await oplog.last_seq()

    async def test_an_unanswered_head_read_is_an_oplogerror(self) -> None:
        oplog = _make_oplog(_StreamInfoJetStream(error=NatsTimeoutError()))  # type: ignore[arg-type]
        with pytest.raises(OpLogError, match="last_seq failed"):
            await oplog.last_seq()

    async def test_an_unanswered_replay_stream_info_is_an_oplogerror(self) -> None:
        oplog = _make_oplog(_ScriptedFailureJetStream(stream_info=NatsTimeoutError()))  # type: ignore[arg-type]
        with pytest.raises(OpLogError, match="replay failed \\(stream_info\\)"):
            async for _ in oplog.replay(from_seq=1):
                pass

    async def test_a_bug_in_the_replay_stream_info_propagates(self) -> None:
        oplog = _make_oplog(_ScriptedFailureJetStream(stream_info=_BUG))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="a bug in the call"):
            async for _ in oplog.replay(from_seq=1):
                pass

    async def test_an_unanswered_replay_consumer_create_is_an_oplogerror(self) -> None:
        oplog = _make_oplog(_ScriptedFailureJetStream(add_consumer=NatsTimeoutError()))  # type: ignore[arg-type]
        with pytest.raises(OpLogError, match="replay failed \\(subscribe\\)"):
            async for _ in oplog.replay(from_seq=1):
                pass

    async def test_a_bug_in_the_replay_consumer_create_propagates(self) -> None:
        oplog = _make_oplog(_ScriptedFailureJetStream(add_consumer=_BUG))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="a bug in the call"):
            async for _ in oplog.replay(from_seq=1):
                pass

    async def test_a_closed_connection_mid_replay_is_an_oplogerror(self) -> None:
        js = _ReplayJetStream(last_seq=5, fetch_error=NatsConnectionClosedError())
        with pytest.raises(OpLogError, match="replay failed \\(fetch\\)"):
            async for _ in _make_oplog(js).replay(from_seq=1):  # type: ignore[arg-type]
                pass
        assert js.psub.unsubscribed is True

    async def test_a_bug_in_the_fetch_propagates_and_the_consumer_is_still_torn_down(self) -> None:
        js = _ReplayJetStream(last_seq=5, fetch_error=_BUG)
        with pytest.raises(TypeError, match="a bug in the call"):
            async for _ in _make_oplog(js).replay(from_seq=1):  # type: ignore[arg-type]
                pass
        assert js.psub.unsubscribed is True
