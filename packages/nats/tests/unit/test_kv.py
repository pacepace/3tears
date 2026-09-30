"""unit tests for :class:`threetears.nats.NatsKvBucket`.

these tests substitute fakes for the underlying nats-py JetStream KV
api so the bucket wrapper logic (CAS semantics, error mapping,
namespace-prefix) can be exercised without a live broker. integration
tests against a real JetStream KV bucket live in tests/integration/.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nats.js.errors import KeyNotFoundError, KeyWrongLastSequenceError

from threetears.nats import KvError, NatsKvBucket
from threetears.nats.errors import PublishTimeoutError
from threetears.nats.kv import _last_timeout_remedy_log  # noqa: SLF001 - module-level throttle state under test


# parity-exempt: minimal Entry dataclass for the NATS-KV wrapper unit tests carrying only value+revision
class _FakeEntry:
    def __init__(self, value: bytes | None, revision: int | None) -> None:
        self.value = value
        self.revision = revision


# parity-exempt: subset shim for nats.js.KeyValue exposing the get/put/create/update/delete surface tested at the wrapper level; full KeyValue carries history/watch/purge methods unrelated to wrapper behaviour
class _FakeKv:
    """fake KeyValue handle storing entries in a dict."""

    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, int]] = {}
        # the revision of each key's deletion marker, as nats-py reports it on KeyNotFoundError.entry
        self.markers: dict[str, int] = {}
        self.next_revision = 0
        self.fail_next: BaseException | None = None

    def _maybe_fail(self) -> None:
        if self.fail_next is not None:
            exc = self.fail_next
            self.fail_next = None
            raise exc

    async def get(self, key: str) -> _FakeEntry:
        self._maybe_fail()
        entry = self.store.get(key)
        if entry is None:
            marker = self.markers.get(key)
            if marker is not None:
                raise KeyNotFoundError(_FakeEntry(value=b"", revision=marker), "DEL")
            raise KeyNotFoundError()
        value, rev = entry
        return _FakeEntry(value=value, revision=rev)

    async def put(self, key: str, value: bytes) -> int:
        self._maybe_fail()
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        return self.next_revision

    async def create(self, key: str, value: bytes) -> int:
        self._maybe_fail()
        if key in self.store:
            raise KeyWrongLastSequenceError()
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        return self.next_revision

    async def update(self, key: str, value: bytes, revision: int) -> int:
        self._maybe_fail()
        existing = self.store.get(key)
        if existing is None or existing[1] != revision:
            raise KeyWrongLastSequenceError()
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        return self.next_revision

    async def delete(self, key: str) -> None:
        self._maybe_fail()
        if key not in self.store:
            raise KeyNotFoundError()
        del self.store[key]
        self.next_revision += 1
        self.markers[key] = self.next_revision


def _make_bucket() -> tuple[NatsKvBucket, _FakeKv]:
    """construct a NatsKvBucket backed by a fake KV handle."""
    kv = _FakeKv()
    bucket = NatsKvBucket(
        client=None,  # type: ignore[arg-type]
        full_name="3tears-tests",
        kv=kv,  # type: ignore[arg-type]
        ttl=timedelta(seconds=60),
    )
    return bucket, kv


# parity-exempt: minimal JetStream stand-in exposing only create_key_value/key_value for the bucket self-heal re-open path; full nats.js JetStreamContext surface is huge and unrelated to wrapper behaviour
class _FakeJetStream:
    """Returns a pre-seeded healed KV from create_key_value / key_value (re-open path)."""

    def __init__(self, healed_kv: _FakeKv) -> None:
        self._healed_kv = healed_kv

    async def create_key_value(self, _config: Any) -> _FakeKv:
        return self._healed_kv

    async def key_value(self, _name: str) -> _FakeKv:
        return self._healed_kv


# parity-exempt: minimal NatsClient stand-in exposing only jetstream_context() for the bucket self-heal re-open path; full NatsClient parity would be over-mocking
class _FakeClient:
    """Minimal NatsClient stand-in whose jetstream re-open yields ``healed_kv``."""

    def __init__(self, healed_kv: _FakeKv) -> None:
        self._js = _FakeJetStream(healed_kv)

    def jetstream_context(self) -> _FakeJetStream:
        return self._js


def _make_self_healing_bucket(broken_kv: _FakeKv, healed_kv: _FakeKv) -> NatsKvBucket:
    """A bucket whose initial handle is ``broken_kv`` and whose re-open yields ``healed_kv``."""
    return NatsKvBucket(
        client=_FakeClient(healed_kv),  # type: ignore[arg-type]
        full_name="3tears-tests",
        kv=broken_kv,  # type: ignore[arg-type]
        ttl=timedelta(seconds=60),
    )


# parity-exempt: minimal JetStream stand-in recording the StreamConfig an open sends to add_stream; the full JetStreamContext surface is unrelated to what the opener builds
class _CapturingJetStream:
    """Captures the StreamConfig an open passes to add_stream so its shape can be asserted."""

    def __init__(self) -> None:
        self.config: Any = None

    async def add_stream(self, config: Any) -> Any:
        self.config = config
        return MagicMock()

    async def key_value(self, _name: str) -> _FakeKv:
        return _FakeKv()


class _CapturingClient:
    def __init__(self, js: _CapturingJetStream) -> None:
        self._js = js

    def jetstream_context(self) -> _CapturingJetStream:
        return self._js


def test_kv_default_storage_is_memory() -> None:
    """NATS is the L2 tier in 3tears: the storage default is ``"memory"``, never file.

    Asserted at the signature level (both the public ``NatsClient.kv_bucket`` entry point and
    ``NatsKvBucket.__init__``) so the default can't silently drift back to file.
    """
    import inspect

    from threetears.nats import NatsClient

    assert inspect.signature(NatsKvBucket.__init__).parameters["storage"].default == "memory"
    assert inspect.signature(NatsClient.kv_bucket).parameters["storage"].default == "memory"


@pytest.mark.asyncio
async def test_open_memory_storage_maps_to_memory() -> None:
    """``storage="memory"`` (the default) maps to the JetStream MEMORY storage type, not FILE."""
    from nats.js.api import StorageType

    js = _CapturingJetStream()
    await NatsKvBucket.open(
        client=_CapturingClient(js),  # type: ignore[arg-type]
        full_name="3tears-tests",
        ttl=None,
        storage="memory",
        create_if_missing=True,
        history=1,
    )
    assert js.config.storage == StorageType.MEMORY


@pytest.mark.asyncio
async def test_open_file_storage_is_explicit_opt_in() -> None:
    """``storage="file"`` still works as a deliberate opt-in (maps to FILE)."""
    from nats.js.api import StorageType

    js = _CapturingJetStream()
    await NatsKvBucket.open(
        client=_CapturingClient(js),  # type: ignore[arg-type]
        full_name="3tears-tests",
        ttl=None,
        storage="file",
        create_if_missing=True,
        history=1,
    )
    assert js.config.storage == StorageType.FILE


@pytest.mark.asyncio
async def test_get_returns_none_on_miss() -> None:
    bucket, _ = _make_bucket()
    assert await bucket.get(key="absent") is None


@pytest.mark.asyncio
async def test_get_returns_value() -> None:
    bucket, kv = _make_bucket()
    await kv.put("k", b"v")
    assert await bucket.get(key="k") == b"v"


@pytest.mark.asyncio
async def test_get_entry_returns_value_and_revision() -> None:
    bucket, kv = _make_bucket()
    rev = await kv.put("k", b"v")
    entry = await bucket.get_entry(key="k")
    assert entry == (b"v", rev)


@pytest.mark.asyncio
async def test_get_entry_returns_none_on_miss() -> None:
    bucket, _ = _make_bucket()
    assert await bucket.get_entry(key="absent") is None


@pytest.mark.asyncio
async def test_get_latest_returns_a_live_value_and_its_revision() -> None:
    bucket, kv = _make_bucket()
    rev = await kv.put("k", b"v")
    assert await bucket.get_latest(key="k") == (b"v", rev)


@pytest.mark.asyncio
async def test_get_latest_reports_a_deleted_key_by_its_markers_revision() -> None:
    # get_entry reports a deleted key as absent; the marker's revision is what lets a writer land
    # only if nothing has happened to the key since it looked.
    bucket, kv = _make_bucket()
    await kv.put("k", b"v")
    await kv.delete("k")
    assert await bucket.get_entry(key="k") is None
    assert await bucket.get_latest(key="k") == (None, kv.markers["k"])


@pytest.mark.asyncio
async def test_get_latest_reports_a_never_written_key_as_revision_zero() -> None:
    bucket, _ = _make_bucket()
    assert await bucket.get_latest(key="absent") == (None, 0)


@pytest.mark.asyncio
async def test_put_returns_new_revision() -> None:
    bucket, _ = _make_bucket()
    rev = await bucket.put(key="k", value=b"v1")
    assert rev > 0


@pytest.mark.asyncio
async def test_create_returns_revision_when_absent() -> None:
    bucket, _ = _make_bucket()
    rev = await bucket.create(key="k", value=b"v")
    assert rev is not None and rev > 0


@pytest.mark.asyncio
async def test_create_returns_none_on_conflict() -> None:
    bucket, _ = _make_bucket()
    await bucket.create(key="k", value=b"v")
    second = await bucket.create(key="k", value=b"v2")
    assert second is None


@pytest.mark.asyncio
async def test_update_cas_succeeds_with_correct_revision() -> None:
    bucket, _ = _make_bucket()
    rev1 = await bucket.put(key="k", value=b"v1")
    assert rev1 is not None
    rev2 = await bucket.update(key="k", value=b"v2", revision=rev1)
    assert rev2 is not None and rev2 != rev1


@pytest.mark.asyncio
async def test_update_cas_returns_none_on_revision_mismatch() -> None:
    bucket, _ = _make_bucket()
    await bucket.put(key="k", value=b"v1")
    result = await bucket.update(key="k", value=b"v2", revision=999)
    assert result is None


@pytest.mark.asyncio
async def test_delete_succeeds() -> None:
    bucket, _ = _make_bucket()
    await bucket.put(key="k", value=b"v")
    assert await bucket.delete(key="k") is True
    assert await bucket.get(key="k") is None


@pytest.mark.asyncio
async def test_delete_idempotent_on_missing_key() -> None:
    bucket, _ = _make_bucket()
    assert await bucket.delete(key="never-existed") is True


@pytest.mark.asyncio
async def test_get_wraps_transport_failure() -> None:
    bucket, kv = _make_bucket()
    kv.fail_next = RuntimeError("transport down")
    with pytest.raises(KvError):
        await bucket.get(key="k")


@pytest.mark.asyncio
async def test_put_wraps_transport_failure() -> None:
    bucket, kv = _make_bucket()
    kv.fail_next = RuntimeError("transport down")
    with pytest.raises(KvError):
        await bucket.put(key="k", value=b"v")


@pytest.mark.asyncio
async def test_create_self_heals_after_stream_vanishes() -> None:
    """A vanished stream (NATS restart on ephemeral storage) is re-opened + retried.

    Regression for the production wake outage: the cached bucket handle failed forever on
    "nats: no response from stream" until a process restart. The op now re-opens once and
    retries against the recreated bucket.
    """
    broken_kv = _FakeKv()
    broken_kv.fail_next = RuntimeError("nats: no response from stream")
    healed_kv = _FakeKv()
    bucket = _make_self_healing_bucket(broken_kv, healed_kv)

    rev = await bucket.create(key="agent_wake_tick", value=b"1")

    assert rev is not None and rev > 0
    assert "agent_wake_tick" in healed_kv.store  # the retry wrote to the recreated bucket


@pytest.mark.asyncio
async def test_put_self_heals_after_stream_vanishes() -> None:
    broken_kv = _FakeKv()
    broken_kv.fail_next = RuntimeError("nats: no response from stream")
    healed_kv = _FakeKv()
    bucket = _make_self_healing_bucket(broken_kv, healed_kv)

    rev = await bucket.put(key="k", value=b"v")

    assert rev > 0
    assert healed_kv.store["k"][0] == b"v"


@pytest.mark.asyncio
async def test_miss_does_not_trigger_reopen() -> None:
    """A normal KeyNotFound miss is control flow, NOT a transport failure -- no re-open."""
    kv = _FakeKv()
    healed_kv = _FakeKv()
    bucket = _make_self_healing_bucket(kv, healed_kv)

    assert await bucket.get(key="absent") is None
    # The handle was never swapped: a miss must not pay a re-open round trip.
    assert bucket._kv is kv  # noqa: SLF001 - asserting no self-heal on a normal miss


@pytest.mark.asyncio
async def test_persistent_failure_after_reopen_surfaces_kverror() -> None:
    """If the op still fails AFTER a re-open (NATS genuinely down), surface KvError."""
    broken_kv = _FakeKv()
    broken_kv.fail_next = RuntimeError("down")
    healed_kv = _FakeKv()
    healed_kv.fail_next = RuntimeError("still down")  # the retry fails too
    bucket = _make_self_healing_bucket(broken_kv, healed_kv)

    with pytest.raises(KvError):
        await bucket.put(key="k", value=b"v")


# parity-exempt: minimal JetStream stand-in exposing only stream_info for the date_created read; the full JetStreamContext surface is unrelated to it
class _StreamInfoJetStream:
    """Answers stream_info with a fixed object, or raises what it was given."""

    def __init__(self, *, info: Any = None, error: BaseException | None = None) -> None:
        self._info = info
        self._error = error
        self.asked: list[str] = []

    async def stream_info(self, name: str) -> Any:
        self.asked.append(name)
        if self._error is not None:
            raise self._error
        return self._info


# parity-exempt: minimal NatsClient stand-in exposing only jetstream_context() for the date_created read; full NatsClient parity would be over-mocking
class _StreamInfoClient:
    def __init__(self, js: _StreamInfoJetStream) -> None:
        self._js = js

    def jetstream_context(self) -> _StreamInfoJetStream:
        return self._js


def _bucket_over(js: _StreamInfoJetStream) -> NatsKvBucket:
    """a bucket whose stream-info reads go to ``js``."""
    return NatsKvBucket(
        client=_StreamInfoClient(js),  # type: ignore[arg-type]
        full_name="3tears-tests",
        kv=_FakeKv(),  # type: ignore[arg-type]
        ttl=timedelta(seconds=60),
    )


class TestDateCreated:
    """the creation time a wipe check trusts must fail closed, never hand back a non-time."""

    @pytest.mark.asyncio
    async def test_returns_the_backing_streams_creation_time(self) -> None:
        created = datetime(2026, 9, 15, 20, 0, tzinfo=UTC)
        js = _StreamInfoJetStream(info=MagicMock(created=created))
        assert await _bucket_over(js).date_created() == created
        assert js.asked == ["KV_3tears-tests"]

    @pytest.mark.asyncio
    async def test_a_stream_reporting_no_creation_time_raises_kverror(self) -> None:
        # without this, `None + skew` would surface in a guard as a TypeError no caller denies on.
        js = _StreamInfoJetStream(info=MagicMock(created=None))
        with pytest.raises(KvError, match="no creation time"):
            await _bucket_over(js).date_created()

    @pytest.mark.asyncio
    async def test_a_failing_stream_info_raises_kverror_after_one_reopen(self) -> None:
        js = _StreamInfoJetStream(error=RuntimeError("nats: no response from stream"))
        bucket = _bucket_over(js)
        with patch.object(NatsKvBucket, "_reopen", autospec=True) as reopen:
            with pytest.raises(KvError, match="stream info failed"):
                await bucket.date_created()
        reopen.assert_awaited_once()
        assert len(js.asked) == 2  # the self-heal retried once before surfacing


@pytest.mark.asyncio
async def test_bucket_name_property() -> None:
    bucket, _ = _make_bucket()
    assert bucket.name == "3tears-tests"


@pytest.mark.asyncio
async def test_ttl_property() -> None:
    bucket, _ = _make_bucket()
    assert bucket.ttl == timedelta(seconds=60)


@pytest.fixture(autouse=True)
def _clear_timeout_remedy_throttle() -> Iterator[None]:
    """Reset the per-bucket remedy throttle around every test in this module.

    The throttle is module-level state with a 300s window, so without this the
    FIRST test to wedge a given bucket logs the remedy and every later one is
    silently suppressed. That failure is ordering-dependent, which is the kind
    that shows up in CI and not locally.

    :return: nothing
    :rtype: Iterator[None]
    """
    _last_timeout_remedy_log.clear()
    yield
    _last_timeout_remedy_log.clear()


class TestAKvOperationThatNeverAnswers:
    """The publish wedge at its other call site.

    `KeyValue.put` is literally `await self._js.publish(...)` with no timeout, and every KV
    write ends there. So an unresponsive broker hangs a KV call exactly the way it hung
    `jetstream_publish`, through the same flush path that discards `CancelledError` -- which
    means a caller's own `asyncio.wait_for` cannot break it either.

    This is the path the distributed lock is built on, so a wedge here is what lets one stuck
    pod hold a lock against a whole fleet.
    """

    @pytest.mark.asyncio
    async def test_a_wedged_kv_operation_raises_instead_of_hanging(self) -> None:
        """The caller gets an error and its loop back.

        :return: nothing
        :rtype: None
        """
        from threetears.nats.kv import _KV_OP_TIMEOUT_SECONDS  # noqa: SLF001

        kv = MagicMock()

        async def _never_answers(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(3600)

        kv.put = _never_answers
        bucket = NatsKvBucket(client=None, full_name="itest-b", kv=kv, ttl=None)  # type: ignore[arg-type]

        with patch("threetears.nats.kv._KV_OP_TIMEOUT_SECONDS", 0.05):
            assert _KV_OP_TIMEOUT_SECONDS > 0  # the real bound is a real number, not a sentinel
            with pytest.raises((PublishTimeoutError, KvError)):
                await bucket.put(key="k", value=b"v")

    @pytest.mark.asyncio
    async def test_a_wedged_operation_is_not_retried_through_reopen(self) -> None:
        """A wedge is not a vanished bucket, and retrying doubles the caller's wait.

        `_run_with_reopen` self-heals a transport failure by re-opening and running the op
        again. Treating a timeout as that kind of failure runs a second KV call against the
        same unresponsive broker, so the retry wedges too and the caller waits twice as long
        to learn the same thing.

        :return: nothing
        :rtype: None
        """

        kv = MagicMock()
        attempts = 0

        async def _never_answers(*_args: object, **_kwargs: object) -> None:
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(3600)

        kv.put = _never_answers
        bucket = NatsKvBucket(client=None, full_name="itest-b", kv=kv, ttl=None)  # type: ignore[arg-type]

        with patch("threetears.nats.kv._KV_OP_TIMEOUT_SECONDS", 0.05):
            with pytest.raises((PublishTimeoutError, KvError)):
                await bucket.put(key="k", value=b"v")

        assert attempts == 1, f"the wedged operation was retried {attempts} times through reopen"

    @pytest.mark.asyncio
    async def test_the_timeout_log_names_the_bucket_and_the_grant(self, caplog: pytest.LogCaptureFixture) -> None:
        """A deadline cannot tell an ungranted bucket from a dead broker; the log must say so.

        The two produce the identical timeout, because a JetStream request the server
        refuses is never answered at all. This frame is the only one that knows which
        bucket to name, so leaving the reader with "the broker did not answer" sends them
        to the network while a healthy connection carries every other subject.

        :param caplog: pytest log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """

        kv = MagicMock()

        async def _never_answers(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(3600)

        kv.put = _never_answers
        bucket = NatsKvBucket(client=None, full_name="prod-epochs", kv=kv, ttl=None)  # type: ignore[arg-type]

        with caplog.at_level(logging.ERROR), patch("threetears.nats.kv._KV_OP_TIMEOUT_SECONDS", 0.05):
            with pytest.raises((PublishTimeoutError, KvError)):
                await bucket.put(key="k", value=b"v")

        messages = [record.getMessage() for record in caplog.records]
        assert any("'prod-epochs'" in message and "js_resources" in message for message in messages), messages

    @pytest.mark.asyncio
    async def test_the_remedy_is_not_repeated_for_every_wedged_operation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The condition lasts; the explanation should not be re-printed per operation.

        A wedged broker times out every KV call, and the remedy is long and identical
        each time. Emitting it unthrottled buries the diagnosis inside its own
        repetitions. Each call still raises, which is the signal the caller acts on.

        :param caplog: pytest log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """

        kv = MagicMock()

        async def _never_answers(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(3600)

        kv.put = _never_answers
        bucket = NatsKvBucket(client=None, full_name="throttle-probe", kv=kv, ttl=None)  # type: ignore[arg-type]

        with caplog.at_level(logging.ERROR), patch("threetears.nats.kv._KV_OP_TIMEOUT_SECONDS", 0.05):
            for _ in range(3):
                with pytest.raises((PublishTimeoutError, KvError)):
                    await bucket.put(key="k", value=b"v")

        remedies = [
            r for r in caplog.records if "throttle-probe" in r.getMessage() and "js_resources" in r.getMessage()
        ]
        assert len(remedies) == 1, f"remedy logged {len(remedies)} times across 3 wedged operations"


class TestOpeningAnUngrantedBucket:
    """The FIRST symptom, for a bucket a component opens lazily on first use.

    Both halves of the opener are refused alike -- `STREAM.CREATE` and the `key_value`
    bind -- and neither is answered, so both die on their own deadline and the caller
    gets a `KvError` naming two transport failures. That message is what reaches the
    operator, so the fix has to be in it rather than only in a log line.
    """

    @pytest.mark.asyncio
    async def test_a_failed_open_carries_the_grant_it_probably_needs(self) -> None:
        """The raised error names the bucket's grant, not just the two failures.

        :return: nothing
        :rtype: None
        """
        js = MagicMock()

        async def _refused(*_args: object, **_kwargs: object) -> None:
            raise TimeoutError("nats: timeout")

        # A refusal is never ANSWERED -- the server drops the request and the call
        # dies on its own deadline. Both halves of the opener look like this, which
        # is why neither can be told apart from an unreachable broker on its own.
        js.add_stream = _refused
        js.key_value = _refused
        client = MagicMock()
        client.jetstream_context = MagicMock(return_value=js)

        with pytest.raises(KvError) as caught:
            await NatsKvBucket.open(
                client=client,
                full_name="prod-epochs",
                ttl=None,
                storage="memory",
                create_if_missing=True,
                history=1,
            )

        assert "js_resources" in str(caught.value)
        assert '"$KV.prod-epochs.>"' in str(caught.value)
        # Unhedged: create_if_missing was asked for, so a merely-absent bucket would
        # have been created. Reaching the bind at all rules that cause out.
        assert "FIX: grant" in str(caught.value)
        assert "never created" not in str(caught.value)

    @pytest.mark.asyncio
    async def test_a_failed_bind_only_open_hedges_between_the_two_causes(self) -> None:
        """`create_if_missing=False` never attempts a create, so it cannot rule one out.

        A bucket nobody has created yet fails this branch exactly the way an ungranted
        one does, so asserting the grant would send half of these readers to change a
        permission that was already correct.

        :return: nothing
        :rtype: None
        """
        js = MagicMock()

        async def _refused(*_args: object, **_kwargs: object) -> None:
            raise TimeoutError("nats: timeout")

        js.key_value = _refused
        client = MagicMock()
        client.jetstream_context = MagicMock(return_value=js)

        with pytest.raises(KvError) as caught:
            await NatsKvBucket.open(
                client=client,
                full_name="prod-epochs",
                ttl=None,
                storage="memory",
                create_if_missing=False,
                history=1,
            )

        assert "js_resources" in str(caught.value)
        assert "never created" in str(caught.value)


class TestTheRemedyIsNotSuppressedOnAFreshlyBootedMachine:
    """`time.monotonic()` is time since BOOT on Linux, not since the epoch.

    The throttle compared `now - last` against its window with `0.0` standing in for
    "never logged". On a machine whose uptime is under that window the arithmetic
    suppresses the FIRST remedy -- the one that matters, because a missing KV grant
    is most likely to bite a process that has just started.

    It reads as correct on any developer machine (uptime in days) and fails only
    where it counts. CI found it: a GitHub runner is about a minute old.

    Probed by inflating the INTERVAL rather than faking the clock. Patching
    `time.monotonic` patches it for asyncio too and hangs the event loop; making the
    window larger than the machine's uptime reproduces the same arithmetic on any
    host, and is what the old code fails.
    """

    @pytest.mark.asyncio
    async def test_the_first_remedy_logs_however_young_the_machine_is(self, caplog: pytest.LogCaptureFixture) -> None:
        """An absent key means never logged, whatever the clock reads.

        :param caplog: pytest log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        kv = MagicMock()

        async def _never_answers(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(3600)

        kv.put = _never_answers
        bucket = NatsKvBucket(client=None, full_name="fresh-boot", kv=kv, ttl=None)  # type: ignore[arg-type]

        with (
            caplog.at_level(logging.ERROR),
            patch("threetears.nats.kv._KV_OP_TIMEOUT_SECONDS", 0.05),
            # Larger than any real uptime, so `now - 0.0` is inside the window and the
            # old `0.0` sentinel suppresses. Absence must still mean "never logged".
            patch("threetears.nats.kv._TIMEOUT_REMEDY_LOG_INTERVAL_SECONDS", 1e12),
        ):
            with pytest.raises((PublishTimeoutError, KvError)):
                await bucket.put(key="k", value=b"v")

        assert any("'fresh-boot'" in r.getMessage() and "js_resources" in r.getMessage() for r in caplog.records), (
            "the first remedy was suppressed because the machine had not been up long enough"
        )


# parity-exempt: minimal JetStream stand-in for a bind-only open -- key_value binds, stream_info reports a live config
class _BindOnlyJs:
    """binds any bucket and reports one live stream config for it."""

    def __init__(self, *, max_age: float, allow_msg_ttl: bool) -> None:
        self.config = MagicMock(max_age=max_age, allow_msg_ttl=allow_msg_ttl, allow_direct=True)
        self.published: list[dict[str, Any]] = []

    async def key_value(self, name: str) -> Any:
        handle = MagicMock(name=f"kv:{name}")
        handle.put = AsyncMock(return_value=3)
        return handle

    async def stream_info(self, name: str) -> Any:
        return MagicMock(config=self.config)

    async def publish(self, subject: str, payload: bytes, **kwargs: Any) -> Any:
        self.published.append({"subject": subject, **kwargs})
        return MagicMock(seq=7)


class TestABindOnlyOpenKeepsItsEntryLifetime:
    """a pod never creates a bucket, so a lifetime it asks for must ride each entry instead.

    The hub declares every bucket a pod binds with no bucket-wide expiry, because it cannot know the
    lifetime each primitive over the bucket wants. A replay nonce, a reset ticket or a resume handle
    that stopped expiring would grow without bound or outlive its purpose, so a bind-only open that
    asks for ``ttl`` writes every entry with it -- and refuses a bucket whose own expiry disagrees.
    """

    async def _open(self, js: _BindOnlyJs, *, ttl: timedelta | None) -> NatsKvBucket:
        client = MagicMock()
        client.jetstream_context = MagicMock(return_value=js)
        return await NatsKvBucket.open(
            client=client,
            full_name="ns-agent_pod-x-nonces",
            ttl=ttl,
            storage="memory",
            create_if_missing=False,
            history=1,
        )

    @pytest.mark.asyncio
    async def test_a_bucket_with_no_expiry_gets_the_lifetime_on_every_write(self) -> None:
        js = _BindOnlyJs(max_age=0.0, allow_msg_ttl=True)
        bucket = await self._open(js, ttl=timedelta(minutes=5))
        await bucket.put(key="nonce-1", value=b"1")
        assert js.published == [{"subject": "$KV.ns-agent_pod-x-nonces.nonce-1", "msg_ttl": 300.0}]

    @pytest.mark.asyncio
    async def test_an_explicit_entry_lifetime_still_wins(self) -> None:
        js = _BindOnlyJs(max_age=0.0, allow_msg_ttl=True)
        bucket = await self._open(js, ttl=timedelta(minutes=5))
        await bucket.put(key="nonce-1", value=b"1", ttl=timedelta(seconds=30))
        assert js.published[0]["msg_ttl"] == 30.0

    @pytest.mark.asyncio
    async def test_a_bucket_already_expiring_at_that_age_writes_plainly(self) -> None:
        js = _BindOnlyJs(max_age=300.0, allow_msg_ttl=True)
        bucket = await self._open(js, ttl=timedelta(minutes=5))
        await bucket.put(key="nonce-1", value=b"1")
        assert js.published == []

    @pytest.mark.asyncio
    async def test_a_bucket_expiring_at_another_age_is_refused(self) -> None:
        from threetears.nats.errors import KvConfigMismatch

        with pytest.raises(KvConfigMismatch, match="expires entries after 60s"):
            await self._open(_BindOnlyJs(max_age=60.0, allow_msg_ttl=True), ttl=timedelta(minutes=5))

    @pytest.mark.asyncio
    async def test_a_bucket_refusing_entry_lifetimes_is_refused(self) -> None:
        from threetears.nats.errors import KvConfigMismatch

        with pytest.raises(KvConfigMismatch, match="allow_msg_ttl"):
            await self._open(_BindOnlyJs(max_age=0.0, allow_msg_ttl=False), ttl=timedelta(minutes=5))

    @pytest.mark.asyncio
    async def test_no_lifetime_asked_binds_whatever_the_declarer_set(self) -> None:
        js = _BindOnlyJs(max_age=60.0, allow_msg_ttl=False)
        bucket = await self._open(js, ttl=None)
        await bucket.put(key="cell-1", value=b"1")
        assert js.published == []


# parity-exempt: minimal JetStream stand-in whose bucket is absent for the first N binds, then declared
class _DeclaredLateJs:
    """a bucket nobody has declared yet: ``key_value`` answers not-found until ``absent_for`` binds have run."""

    def __init__(self, *, absent_for: int, healed: Any) -> None:
        self.absent_for = absent_for
        self.binds = 0
        self.healed = healed

    async def key_value(self, name: str) -> Any:
        from nats.js.errors import BucketNotFoundError

        self.binds += 1
        if self.binds <= self.absent_for:
            raise BucketNotFoundError
        return self.healed


def _client_over(js: Any) -> MagicMock:
    client = MagicMock()
    client.jetstream_context = MagicMock(return_value=js)
    return client


@pytest.fixture
def _fast_rebind() -> Iterator[None]:
    """shrink the bind wait so a test that waits for a declarer spends milliseconds, not seconds."""
    with (
        patch("threetears.nats.kv._BIND_RETRY_FIRST_DELAY_SECONDS", 0.001),
        patch("threetears.nats.kv._BIND_RETRY_MAX_DELAY_SECONDS", 0.004),
        patch("threetears.nats.kv._BIND_WAIT_FOR_DECLARER_SECONDS", 0.5),
    ):
        yield


class TestABindOnlyOpenWaitsForItsDeclarer:
    """a pod never creates a bucket, so one missing right now is one its declarer has not declared YET.

    A NATS restart wipes every memory-backed bucket. The hub re-declares them all once it reconnects,
    but a pod can reach the bus first -- and a bind-only open that failed on that first miss left the
    primitive over it unusable until something re-opened it, which for a guard bound once at startup
    meant until the pod restarted. So a bind that finds the bucket ABSENT (the server answered
    not-found) waits with bounded backoff for the declarer; one that is REFUSED (never answered -- an
    ungranted bucket) is not retried, because no amount of waiting grants it.
    """

    @pytest.mark.asyncio
    async def test_a_bucket_declared_while_the_pod_waits_is_bound(self, _fast_rebind: None) -> None:
        js = _DeclaredLateJs(absent_for=3, healed=_FakeKv())
        bucket = await NatsKvBucket.open(
            client=_client_over(js),
            full_name="ns-proxy_assertion_nonces",
            ttl=None,
            storage="memory",
            create_if_missing=False,
            history=1,
        )
        assert js.binds == 4
        assert bucket.name == "ns-proxy_assertion_nonces"

    @pytest.mark.asyncio
    async def test_a_bucket_never_declared_fails_once_the_wait_is_spent(self, _fast_rebind: None) -> None:
        js = _DeclaredLateJs(absent_for=10_000, healed=_FakeKv())
        with (
            patch("threetears.nats.kv._BIND_WAIT_FOR_DECLARER_SECONDS", 0.05),
            pytest.raises(KvError, match="declar"),
        ):
            await NatsKvBucket.open(
                client=_client_over(js),
                full_name="ns-ratelimits",
                ttl=None,
                storage="memory",
                create_if_missing=False,
                history=1,
            )
        assert js.binds > 1

    @pytest.mark.asyncio
    async def test_a_refused_bind_is_not_retried(self, _fast_rebind: None) -> None:
        calls = 0

        async def _refused(*_args: object, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1
            raise TimeoutError("nats: timeout")

        js = MagicMock()
        js.key_value = _refused
        with pytest.raises(KvError):
            await NatsKvBucket.open(
                client=_client_over(js),
                full_name="ns-epochs",
                ttl=None,
                storage="memory",
                create_if_missing=False,
                history=1,
            )
        assert calls == 1, "an ungranted bucket was retried as if waiting could grant it"

    @pytest.mark.asyncio
    async def test_a_handle_whose_bucket_was_wiped_recovers_once_the_declarer_is_back(self, _fast_rebind: None) -> None:
        """the handle a primitive holds outlives the wipe; its next operation re-binds and succeeds."""
        stale = _FakeKv()

        async def _stream_gone(key: str) -> Any:
            raise RuntimeError("nats: no response from stream")

        stale.get = _stream_gone  # type: ignore[method-assign]
        healed = _FakeKv()
        await healed.put("nonce", b"1")
        js = _DeclaredLateJs(absent_for=2, healed=healed)
        bucket = NatsKvBucket(
            client=_client_over(js),  # type: ignore[arg-type]
            full_name="ns-proxy_assertion_nonces",
            kv=stale,  # type: ignore[arg-type]
            ttl=None,
            create_if_missing=False,
        )
        assert await bucket.get(key="nonce") == b"1"
        assert js.binds == 3
