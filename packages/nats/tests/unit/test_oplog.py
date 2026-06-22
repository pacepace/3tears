"""unit tests for :mod:`threetears.nats.oplog` (no broker).

Covers everything testable without a live JetStream server: subject + stream
name construction and sanitization, the published header dict (CAS + dedup),
the :class:`AppendResult` / :class:`OpRecord` value shapes, and input
validation. The full broker-backed contract (CAS fence, dedup, ordered
terminating replay) is the integration proof
``tests/integration/test_oplog_round_trip.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from nats.js.api import Header, StorageType

from threetears.nats import (
    DEFAULT_NAMESPACE,
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
    monkeypatch.delenv("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", raising=False)
    set_default_namespace(DEFAULT_NAMESPACE)


# ----------------------------------------------------------------------
# fakes — capture what the wrapper hands to nats-py without a broker
# ----------------------------------------------------------------------


class _FakePubAck:
    def __init__(self, seq: int, duplicate: bool | None) -> None:
        self.seq = seq
        self.duplicate = duplicate


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

    async def publish(self, subject: str, payload: bytes, *, headers: dict[str, str], stream: str) -> Any:
        self.publish_calls.append({"subject": subject, "payload": payload, "headers": headers, "stream": stream})
        if self._raise_on_publish is not None:
            raise self._raise_on_publish
        assert self._next_ack is not None
        return self._next_ack


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
    assert str(sub) == "aibots.oplog.story-alpha.main"
    assert sub.kind == "point"


def test_oplog_subject_sanitizes_dotted_segments() -> None:
    """dots in repo/branch are replaced so they don't overload the separator."""
    sub = Subjects.oplog("my.org/repo", "feature.x")
    assert str(sub) == "aibots.oplog.my-org/repo.feature-x"


@pytest.mark.parametrize(("repo", "branch"), [("", "main"), ("repo", "")])
def test_oplog_subject_rejects_empty_segments(repo: str, branch: str) -> None:
    with pytest.raises(ValueError):
        Subjects.oplog(repo, branch)


async def test_open_builds_dot_free_stream_name_and_memory_r3_config() -> None:
    """stream name has no dots; config is in-memory, R3, single-subject, dedup window."""
    js = _FakeJetStream()
    oplog = await OpLog.open(client=_FakeClient(js), repo="story.alpha", branch="release.1")  # type: ignore[arg-type]

    assert oplog.subject == "aibots.oplog.story-alpha.release-1"
    assert oplog.stream == "aibots-oplog-story-alpha-release-1"
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


class _FakeStreamState:
    def __init__(self, last_seq: int) -> None:
        self.last_seq = last_seq


class _FakeStreamInfo:
    def __init__(self, last_seq: int) -> None:
        self.state = _FakeStreamState(last_seq)


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

    def __init__(self, *, last_seq: int, fetch_error: BaseException) -> None:
        self._last_seq = last_seq
        self.psub = _FakePullSub(fetch_error)

    async def stream_info(self, name: str) -> _FakeStreamInfo:
        return _FakeStreamInfo(self._last_seq)

    async def pull_subscribe(self, subject: str, *, stream: str, config: Any) -> _FakePullSub:
        return self.psub


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


async def test_replay_timeout_is_a_clean_tail_not_an_error() -> None:
    """A fetch TimeoutError (drained tail) ends replay cleanly -- it is the normal terminator."""
    js = _ReplayJetStream(last_seq=5, fetch_error=TimeoutError("nats: timeout"))
    oplog = _make_oplog(js)  # type: ignore[arg-type]

    records = [r async for r in oplog.replay(from_seq=1)]
    assert records == []  # nothing delivered before the tail timeout; no error raised
    assert js.psub.unsubscribed is True


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
