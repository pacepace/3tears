"""unit tests for the KV create-or-reconcile primitive (coll-task-04a).

The defect these cover is that opening a bucket that ALREADY EXISTS used to bind
to it and throw the caller's requested configuration away, with a ``log.debug``
as the only trace. Proven live before the fix: a second opener asking for
``ttl=7200s, history=5`` against a bucket created with ``ttl=60s, history=1`` got
a handle reporting ``ttl=2:00:00`` while the server still said ``max_age=60``.

Everything here runs against fakes; the shape of the KV stream itself is
compared against what nats-py actually builds, on a live broker, in
``tests/integration/test_kv_bucket_reconcile_live.py`` -- a hand-kept field list
is exactly what goes stale.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from nats.js.api import DiscardPolicy, StorageType, StreamConfig

from threetears.nats.errors import KvConfigMismatch, KvError, NatsClientError, StreamSubjectsOverlapError
from threetears.nats.kv import (
    RECONCILED_KV_STREAM_FIELDS,
    NatsKvBucket,
    build_kv_stream_config,
    kv_stream_differences,
)


class _ApiError(Exception):
    """Stand-in for a nats-py ``APIError`` carrying the server's own error code.

    Not a fake of a protocol: the production classifier reads ``err_code`` off
    whatever ``add_stream`` raised, so an exception with that attribute IS the
    input shape.
    """

    def __init__(self, err_code: int, description: str) -> None:
        super().__init__(f"nats: code=400 err_code={err_code} description={description!r}")
        self.err_code = err_code


# parity-exempt: JetStream stand-in scripted per-test for the add/update/info/bind calls the KV opener makes; the real JetStreamContext surface is an order of magnitude larger and unrelated
class _ScriptedJetStream:
    """A JetStream context whose four opener-facing calls are scripted per test."""

    def __init__(
        self,
        *,
        add_raises: Exception | None = None,
        live: StreamConfig | None = None,
        bind_raises: Exception | None = None,
    ) -> None:
        self.add_raises = add_raises
        self.live = live
        self.bind_raises = bind_raises
        self.added: list[StreamConfig] = []
        self.updated: list[StreamConfig] = []
        self.info_calls: list[str] = []

    async def add_stream(self, config: StreamConfig) -> Any:
        if self.add_raises is not None:
            raise self.add_raises
        self.added.append(config)
        return object()

    async def update_stream(self, config: StreamConfig) -> Any:
        self.updated.append(config)
        return object()

    async def stream_info(self, name: str) -> Any:
        self.info_calls.append(name)
        if self.live is None:
            raise AssertionError("stream_info called with no live stream scripted")
        return type("_Info", (), {"config": self.live})()

    async def key_value(self, _name: str) -> Any:
        if self.bind_raises is not None:
            raise self.bind_raises
        return object()


# parity-exempt: NatsClient stand-in exposing only jetstream_context(), the one method NatsKvBucket.open calls on its client
class _ScriptedClient:
    def __init__(self, js: _ScriptedJetStream) -> None:
        self._js = js

    def jetstream_context(self) -> _ScriptedJetStream:
        return self._js


def _live(**overrides: Any) -> StreamConfig:
    """A server-side KV stream config, defaulting to what nats-py creates today."""
    base: dict[str, Any] = {
        "name": "KV_probe",
        "subjects": ["$KV.probe.>"],
        "max_age": 0.0,
        "max_msgs_per_subject": 1,
        "storage": StorageType.MEMORY,
        "allow_direct": False,
        "discard": DiscardPolicy.NEW,
    }
    base.update(overrides)
    return StreamConfig(**base)


class TestTheMismatchTypeIsNotDegradable:
    """`KvError` is what the L2 accessors catch and degrade on.

    Raising a config mismatch as one would turn "this bucket is misconfigured"
    into a per-operation warning with L2 silently off fleet-wide, which is the
    exact degradation the mismatch exists to refuse. The type has to sit OUTSIDE
    that catch, and only the type keeps it there.
    """

    def test_mismatch_is_not_a_kv_error(self) -> None:
        assert not issubclass(KvConfigMismatch, KvError)

    def test_mismatch_is_still_a_nats_client_error(self) -> None:
        """It is a wrapper error, so `except NatsClientError` at a boundary still sees it."""
        assert issubclass(KvConfigMismatch, NatsClientError)

    def test_a_kv_error_catch_does_not_swallow_it(self) -> None:
        """The property stated behaviourally, not by subclass arithmetic.

        `issubclass` would still pass if somebody made ``KvError`` the base of a
        common parent that ``except KvError`` also matched.
        """
        caught = False
        try:
            try:
                raise KvConfigMismatch("drift")
            except KvError:
                caught = True
        except KvConfigMismatch:
            pass
        assert not caught, "an `except KvError` handler swallowed the config mismatch"


class TestTheComparedFieldSetIsNarrow:
    """A full comparison would raise on every open, forever.

    The requested config and the server-normalised one differ on something
    almost always -- `max_bytes`, `retention`, `max_msg_size` all default one way
    in the dataclass and another on the server.
    """

    def test_only_direct_is_reconciled_for_this_landing(self) -> None:
        assert RECONCILED_KV_STREAM_FIELDS == ("allow_direct",)

    def test_a_field_the_caller_did_not_request_is_not_a_difference(self) -> None:
        requested = build_kv_stream_config(
            bucket="probe", ttl_seconds=0, history=1, storage_type=StorageType.MEMORY, direct=None
        )
        assert kv_stream_differences(requested=requested, actual=_live()) == {}

    def test_a_requested_field_the_server_does_not_match_is_a_difference(self) -> None:
        requested = build_kv_stream_config(
            bucket="probe", ttl_seconds=60, history=5, storage_type=StorageType.MEMORY, direct=True
        )
        differences = kv_stream_differences(requested=requested, actual=_live())
        assert differences["max_age"] == (60, 0.0)
        assert differences["max_msgs_per_subject"] == (5, 1)
        assert differences["allow_direct"] == (True, False)

    def test_an_unset_server_direct_reads_as_false_not_as_a_difference_in_kind(self) -> None:
        """`allow_direct` is Optional on the wire and absent means false.

        Comparing ``True != None`` and ``True != False`` both flag drift, but
        ``False != None`` would flag drift where there is none -- a declarer
        asking for ``direct=False`` against a stream that never set the field
        would update forever.
        """
        requested = build_kv_stream_config(
            bucket="probe", ttl_seconds=0, history=1, storage_type=StorageType.MEMORY, direct=False
        )
        assert kv_stream_differences(requested=requested, actual=_live(allow_direct=None)) == {}

    def test_an_unset_server_max_age_reads_as_unlimited_not_as_a_difference(self) -> None:
        """``max_age`` carries the same absent-means-something encoding.

        ``ttl=None`` builds ``max_age=0`` (unlimited) while a stream nats-py
        created with no TTL can report the field as absent. Comparing raw values
        would report drift on every such open, and a report that fires every time
        is a report nobody reads.
        """
        requested = build_kv_stream_config(
            bucket="probe", ttl_seconds=0, history=1, storage_type=StorageType.MEMORY, direct=True
        )
        assert kv_stream_differences(requested=requested, actual=_live(max_age=None, allow_direct=True)) == {}


class TestTheDeclarerReconciles:
    """`create_if_missing=True` is the declaring identity: it fixes what it finds."""

    @pytest.mark.asyncio
    async def test_an_absent_bucket_is_created_with_the_requested_direct(self) -> None:
        js = _ScriptedJetStream()
        await NatsKvBucket.open(
            client=_ScriptedClient(js),  # type: ignore[arg-type]
            full_name="probe",
            ttl=None,
            storage="memory",
            create_if_missing=True,
            history=1,
            direct=True,
        )
        assert js.added[0].allow_direct is True
        assert js.updated == []

    @pytest.mark.asyncio
    async def test_a_live_bucket_with_the_wrong_direct_is_updated_in_place(self) -> None:
        js = _ScriptedJetStream(
            add_raises=_ApiError(10058, "stream name already in use with a different configuration"),
            live=_live(allow_direct=False),
        )
        await NatsKvBucket.open(
            client=_ScriptedClient(js),  # type: ignore[arg-type]
            full_name="probe",
            ttl=None,
            storage="memory",
            create_if_missing=True,
            history=1,
            direct=True,
        )
        assert len(js.updated) == 1, "a declarer must reconcile allow_direct in place"
        assert js.updated[0].allow_direct is True

    @pytest.mark.asyncio
    async def test_a_live_bucket_already_carrying_it_is_not_updated(self) -> None:
        """Idempotence, and the reason it matters: `update_stream` is a write.

        `coll-task-05a` removes it from pod principals, so an open that updated
        unconditionally would start failing for every pod.
        """
        js = _ScriptedJetStream(
            add_raises=_ApiError(10058, "stream name already in use with a different configuration"),
            live=_live(allow_direct=True),
        )
        await NatsKvBucket.open(
            client=_ScriptedClient(js),  # type: ignore[arg-type]
            full_name="probe",
            ttl=None,
            storage="memory",
            create_if_missing=True,
            history=1,
            direct=True,
        )
        assert js.updated == []

    @pytest.mark.asyncio
    async def test_drift_outside_the_reconciled_set_is_reported_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The half of the defect that is not about `direct` at all.

        The old opener bound to whatever existed and said so at DEBUG, so a
        bucket carrying somebody else's TTL was indistinguishable from one
        carrying yours. It still binds -- reconciling every field would let two
        processes fight over one bucket -- but it no longer does so in silence.
        """
        js = _ScriptedJetStream(
            add_raises=_ApiError(10058, "stream name already in use with a different configuration"),
            live=_live(max_age=60.0, allow_direct=None),
        )
        with caplog.at_level("WARNING"):
            await NatsKvBucket.open(
                client=_ScriptedClient(js),  # type: ignore[arg-type]
                full_name="probe",
                ttl=timedelta(seconds=7200),
                storage="memory",
                create_if_missing=True,
                history=1,
                direct=None,
            )
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "a dropped request must not be reported below WARNING"
        assert "max_age" in warnings[0].getMessage()
        assert js.updated == [], "an undeclared field must not be reconciled"


class TestTheReaderRefuses:
    """`create_if_missing=False` is a reader: it has no authority to change a shared bucket."""

    @pytest.mark.asyncio
    async def test_a_live_bucket_with_the_wrong_direct_raises_the_mismatch(self) -> None:
        js = _ScriptedJetStream(live=_live(allow_direct=False))
        with pytest.raises(KvConfigMismatch) as caught:
            await NatsKvBucket.open(
                client=_ScriptedClient(js),  # type: ignore[arg-type]
                full_name="probe",
                ttl=None,
                storage="memory",
                create_if_missing=False,
                history=1,
                direct=True,
            )
        assert "allow_direct" in str(caught.value)
        assert js.updated == [], "a reader must not repair the bucket it refused"

    @pytest.mark.asyncio
    async def test_a_matching_bucket_binds(self) -> None:
        js = _ScriptedJetStream(live=_live(allow_direct=True))
        bucket = await NatsKvBucket.open(
            client=_ScriptedClient(js),  # type: ignore[arg-type]
            full_name="probe",
            ttl=None,
            storage="memory",
            create_if_missing=False,
            history=1,
            direct=True,
        )
        assert bucket.name == "probe"

    @pytest.mark.asyncio
    async def test_a_reader_that_states_no_direct_does_not_even_look(self) -> None:
        """Today's callers pass no `direct`, and must not pay a round trip for it."""
        js = _ScriptedJetStream()
        await NatsKvBucket.open(
            client=_ScriptedClient(js),  # type: ignore[arg-type]
            full_name="probe",
            ttl=None,
            storage="memory",
            create_if_missing=False,
            history=1,
            direct=None,
        )
        assert js.info_calls == []


class TestARefusalIsNotAnExistingBucket:
    """`ensure_jetstream_stream` conflates them; this arm had to be built.

    That method types only subjects-overlap and falls through to
    ``update_stream`` for every other add failure -- "already in use" and a
    permissions refusal alike. A refusal is not answered at all, so it arrives as
    a deadline rather than as an API error, and updating in response just spends
    a second deadline learning the same thing.
    """

    @pytest.mark.asyncio
    async def test_an_unanswered_create_never_reaches_update_stream(self) -> None:
        js = _ScriptedJetStream(add_raises=TimeoutError("nats: timeout"), bind_raises=TimeoutError("nats: timeout"))
        with pytest.raises(KvError):
            await NatsKvBucket.open(
                client=_ScriptedClient(js),  # type: ignore[arg-type]
                full_name="probe",
                ttl=None,
                storage="memory",
                create_if_missing=True,
                history=1,
                direct=True,
            )
        assert js.updated == []
        assert js.info_calls == []

    @pytest.mark.asyncio
    async def test_a_create_refused_but_a_bind_allowed_still_yields_a_bucket(self) -> None:
        """A principal granted STREAM.INFO but not STREAM.CREATE reads fine.

        This is what pods look like once `coll-task-05a` narrows their grant, so
        turning an unanswered create into a hard failure would break them.
        """
        js = _ScriptedJetStream(add_raises=TimeoutError("nats: timeout"))
        bucket = await NatsKvBucket.open(
            client=_ScriptedClient(js),  # type: ignore[arg-type]
            full_name="probe",
            ttl=None,
            storage="memory",
            create_if_missing=True,
            history=1,
            direct=True,
        )
        assert bucket.name == "probe"

    @pytest.mark.asyncio
    async def test_subjects_overlap_is_still_its_own_typed_error(self) -> None:
        js = _ScriptedJetStream(add_raises=_ApiError(10065, "subjects overlap with an existing stream"))
        with pytest.raises(StreamSubjectsOverlapError):
            await NatsKvBucket.open(
                client=_ScriptedClient(js),  # type: ignore[arg-type]
                full_name="probe",
                ttl=None,
                storage="memory",
                create_if_missing=True,
                history=1,
                direct=True,
            )


class TestTheSelfHealCarriesDirect:
    """`_reopen` runs after a NATS restart wipes JetStream.

    If it forgot `direct`, the bucket would come back with the field unset --
    every read silently back on the body-carried form no key-scoped grant can
    constrain, and racing whatever startup hook reconciles it.
    """

    @pytest.mark.asyncio
    async def test_a_reopen_recreates_with_the_declared_direct(self) -> None:
        js = _ScriptedJetStream()
        bucket = await NatsKvBucket.open(
            client=_ScriptedClient(js),  # type: ignore[arg-type]
            full_name="probe",
            ttl=None,
            storage="memory",
            create_if_missing=True,
            history=1,
            direct=True,
        )
        js.added.clear()
        await bucket._reopen()  # noqa: SLF001 - the self-heal under test
        assert js.added[0].allow_direct is True
