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
from nats.js.errors import APIError, BucketNotFoundError, NotFoundError

from threetears.nats.errors import KvConfigMismatch, KvError, NatsClientError, StreamSubjectsOverlapError
from threetears.nats.kv import (
    RECONCILED_KV_STREAM_FIELDS,
    KvTimings,
    NatsKvBucket,
    build_kv_stream_config,
    kv_stream_differences,
    open_kv_stream,
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
        self.update_raises: Exception | None = None

    async def add_stream(self, config: StreamConfig) -> Any:
        if self.add_raises is not None:
            raise self.add_raises
        self.added.append(config)
        return object()

    async def update_stream(self, config: StreamConfig) -> Any:
        if self.update_raises is not None:
            raise self.update_raises
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
        # the connection an opened bucket records, and follows across a credential renewal
        self.raw = object()

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
        # both nats-py's create_key_value and this package's build_kv_stream_config set it on a
        # stream they create; a legacy bucket without it is spelled out where a test needs one.
        "allow_msg_ttl": True,
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

    def test_only_direct_and_per_entry_ttl_are_reconciled(self) -> None:
        # allow_direct for read scoping; allow_msg_ttl so entries nothing deletes can carry a
        # server-side lifetime. every other requestable field stays set-at-create.
        assert RECONCILED_KV_STREAM_FIELDS == ("allow_direct", "allow_msg_ttl")

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
    async def test_a_legacy_bucket_without_per_entry_ttl_is_enabled_in_place(self) -> None:
        """Buckets created before this package set allow_msg_ttl carry it off.

        Without the in-place enable, every TTL'd write to such a bucket is refused by the server
        for as long as the bucket lives -- which, for a memory bucket nobody deletes, is forever.
        """
        js = _ScriptedJetStream(
            add_raises=_ApiError(10058, "stream name already in use with a different configuration"),
            live=_live(allow_direct=True, allow_msg_ttl=None),
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
        assert len(js.updated) == 1, "a declarer must enable allow_msg_ttl on a legacy bucket"
        assert js.updated[0].allow_msg_ttl is True

    @pytest.mark.asyncio
    async def test_a_legacy_file_bucket_is_enabled_in_place_without_asking_to_change_storage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The server refuses to change a stream's storage, so an update that asked for it would
        fail the in-place enable along with it -- and the opener with it, for every guard on that
        bucket. The update carries the live storage and the drift is reported, not requested."""
        js = _ScriptedJetStream(
            add_raises=_ApiError(10058, "stream name already in use with a different configuration"),
            live=_live(storage=StorageType.FILE, allow_direct=True, allow_msg_ttl=None, max_age=30.0),
        )
        with caplog.at_level("WARNING"):
            await NatsKvBucket.open(
                client=_ScriptedClient(js),  # type: ignore[arg-type]
                full_name="probe",
                ttl=None,
                storage="memory",
                create_if_missing=True,
                history=1,
                direct=True,
            )
        assert len(js.updated) == 1
        assert js.updated[0].allow_msg_ttl is True
        assert js.updated[0].storage == StorageType.FILE, "the update asked the server to change storage"
        assert js.updated[0].max_age == 30.0, "the update carried a value that was not being reconciled"
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("storage" in w for w in warnings), "the storage drift went unreported"

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


def _name_in_use() -> _ApiError:
    """the server's answer to a create of a bucket that is live with another configuration."""
    return _ApiError(10058, "stream name already in use with a different configuration")


def _memory_declaration(*, ttl_seconds: int = 0, history: int = 1) -> StreamConfig:
    """the shape the hub declares every pod bucket with: memory, direct reads, no bucket-wide expiry."""
    return build_kv_stream_config(
        bucket="probe", ttl_seconds=ttl_seconds, history=history, storage_type=StorageType.MEMORY, direct=True
    )


class TestADeclarerThatOwnsItsBucket:
    """`owns_bucket=True` lets the bucket's ONE declarer reconcile its whole shape.

    Found live on cobalt-dev: the shared rate-limit bucket was created long ago with a bucket-wide
    ``max_age`` of 300s, and the shared nonce bucket on FILE storage with a ``max_age`` of 60s. Their
    declarer now asks for memory and no bucket-wide expiry, so every bind-only opener can give its
    own entries their own lifetime -- but neither field was reconciled, so the stale values were
    logged as dropped and never removed, and every opener asking for a 60s per-entry TTL was
    refused with ``KvConfigMismatch``.
    """

    @pytest.mark.asyncio
    async def test_a_stale_bucket_wide_expiry_is_removed_in_place(self, caplog: pytest.LogCaptureFixture) -> None:
        js = _ScriptedJetStream(add_raises=_name_in_use(), live=_live(allow_direct=True, max_age=300.0))
        with caplog.at_level("INFO", logger="threetears.nats.kv"):
            await open_kv_stream(
                js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True, owns_bucket=True
            )
        assert len(js.updated) == 1, "a declarer that owns its bucket must reconcile max_age in place"
        assert js.updated[0].max_age == 0
        assert js.updated[0].allow_msg_ttl is True
        reconciled = [r for r in caplog.records if r.getMessage() == "JetStream KV bucket reconciled in place"]
        assert reconciled, "the in-place reconcile went unlogged"
        applied = reconciled[0].extra_data["applied"]  # type: ignore[attr-defined]
        assert "max_age" in applied and "300" in applied, applied
        assert not [r for r in caplog.records if r.levelname == "WARNING"], "nothing was dropped"

    @pytest.mark.asyncio
    async def test_without_ownership_the_stale_expiry_is_reported_and_left(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """the default is today's behaviour exactly, with the warning now naming the way out."""
        js = _ScriptedJetStream(add_raises=_name_in_use(), live=_live(allow_direct=True, max_age=300.0))
        with caplog.at_level("WARNING"):
            await open_kv_stream(js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True)
        assert js.updated == [], "max_age is reconciled only by a declarer that owns the bucket"
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("max_age" in w and "owns_bucket" in w for w in warnings), warnings

    @pytest.mark.asyncio
    async def test_an_expiry_that_already_matches_is_not_updated(self) -> None:
        """idempotence: ``update_stream`` is a write, and an owner re-declares on every start."""
        js = _ScriptedJetStream(add_raises=_name_in_use(), live=_live(allow_direct=True, max_age=None))
        await open_kv_stream(
            js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True, owns_bucket=True
        )
        assert js.updated == []

    @pytest.mark.asyncio
    async def test_a_requested_expiry_matching_the_live_one_is_not_updated(self) -> None:
        js = _ScriptedJetStream(add_raises=_name_in_use(), live=_live(allow_direct=True, max_age=60.0))
        await open_kv_stream(
            js=js,
            full_name="probe",
            config=_memory_declaration(ttl_seconds=60),
            create_if_missing=True,
            owns_bucket=True,
        )
        assert js.updated == []

    @pytest.mark.asyncio
    async def test_a_requested_expiry_shorter_than_the_duplicate_window_brings_the_window_with_it(self) -> None:
        """JetStream refuses a duplicate window longer than ``max_age``.

        A bucket with no expiry carries the two-minute window; reconciling it to 60s alone would be
        refused, so the update carries the window the requested shape carries.
        """
        js = _ScriptedJetStream(
            add_raises=_name_in_use(), live=_live(allow_direct=True, max_age=0.0, duplicate_window=120.0)
        )
        await open_kv_stream(
            js=js,
            full_name="probe",
            config=_memory_declaration(ttl_seconds=60),
            create_if_missing=True,
            owns_bucket=True,
        )
        assert len(js.updated) == 1
        assert js.updated[0].max_age == 60
        assert js.updated[0].duplicate_window == 60

    @pytest.mark.asyncio
    async def test_the_update_changes_only_the_owned_fields_of_the_live_config(self) -> None:
        """built from the LIVE config: the owner takes expiry and history, and asks for nothing else."""
        js = _ScriptedJetStream(
            add_raises=_name_in_use(),
            live=_live(allow_direct=True, max_age=300.0, max_msgs_per_subject=5, max_consumers=7),
        )
        await open_kv_stream(
            js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True, owns_bucket=True
        )
        assert len(js.updated) == 1
        assert js.updated[0].max_age == 0
        assert js.updated[0].max_msgs_per_subject == 1, "the owner owns the declared history too"
        assert js.updated[0].max_consumers == 7, "the update carried a value nobody requested"

    @pytest.mark.asyncio
    async def test_an_update_the_server_answers_with_a_refusal_does_not_blame_a_grant(self) -> None:
        """an answered refusal names a configuration the server will not take; a missing grant is never answered."""
        js = _ScriptedJetStream(add_raises=_name_in_use(), live=_live(allow_direct=True, max_age=300.0))
        js.update_raises = _ApiError(10052, "duplicates window can not be larger then max age")
        with pytest.raises(KvError) as caught:
            await open_kv_stream(
                js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True, owns_bucket=True
            )
        assert "duplicates window" in str(caught.value)
        assert "the server refused it itself" in str(caught.value)
        assert "grant this principal" not in str(caught.value)

    @pytest.mark.asyncio
    async def test_an_unanswered_update_still_names_the_grant(self) -> None:
        js = _ScriptedJetStream(add_raises=_name_in_use(), live=_live(allow_direct=False))
        js.update_raises = TimeoutError("nats: timeout")
        with pytest.raises(KvError, match="grant this principal"):
            await open_kv_stream(js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True)

    @pytest.mark.asyncio
    async def test_a_binder_cannot_own_the_bucket(self) -> None:
        """a process that only binds has no authority over the bucket."""
        js = _ScriptedJetStream(live=_live(allow_direct=True, max_age=300.0))
        with pytest.raises(ValueError, match="owns_bucket"):
            await NatsKvBucket.open(
                client=_ScriptedClient(js),  # type: ignore[arg-type]
                full_name="probe",
                ttl=None,
                storage="memory",
                create_if_missing=False,
                history=1,
                direct=True,
                owns_bucket=True,
            )
        assert js.updated == []

    @pytest.mark.asyncio
    async def test_a_file_bucket_cannot_be_owned(self) -> None:
        """a recreate drops every entry, which is a restart for a memory bucket and data loss for a file one."""
        js = _ScriptedJetStream(add_raises=_name_in_use(), live=_live(allow_direct=True))
        with pytest.raises(ValueError, match="storage='memory'"):
            await open_kv_stream(
                js=js,
                full_name="probe",
                config=build_kv_stream_config(
                    bucket="probe", ttl_seconds=0, history=1, storage_type=StorageType.FILE, direct=True
                ),
                create_if_missing=True,
                owns_bucket=True,
            )
        assert js.added == [] and js.updated == []

    @pytest.mark.asyncio
    async def test_a_self_heal_reopen_keeps_owning_the_bucket(self) -> None:
        """the re-open after a vanished stream declares again, ownership of the bucket included.

        Without it a bucket another process recreated with its own expiry in the meantime would be
        bound as-is by the owner's own self-heal.
        """
        js = _RecreatedElsewhereJetStream()
        bucket = await NatsKvBucket.open(
            client=_ScriptedClient(js),  # type: ignore[arg-type]
            full_name="probe",
            ttl=None,
            storage="memory",
            create_if_missing=True,
            history=1,
            direct=True,
            owns_bucket=True,
        )
        js.recreate_elsewhere(_live(allow_direct=True, max_age=300.0))

        assert await bucket.get(key="k") == b"healed"

        assert len(js.updated) == 1, "the self-heal did not reconcile the bucket it owns"
        assert js.updated[0].max_age == 0


def _stream_not_found() -> NotFoundError:
    """the server's answer to a delete of a stream that is already gone."""
    return NotFoundError(code=404, err_code=10059, description="stream not found")


class _OwnedRecreateJetStream(_ScriptedJetStream):
    """a JetStream holding a FILE bucket an owner declares on memory, with a concurrent owner scripted in.

    The declaration's own create is refused as a name in use; the recreate's create succeeds unless
    ``peer_live`` is set, when a concurrent owner has created the stream again between this
    declarer's delete and its create, and ``peer_live`` is what the server then reports.
    """

    def __init__(
        self,
        *,
        live: StreamConfig,
        delete_raises: Exception | None = None,
        peer_live: StreamConfig | None = None,
        recreate_raises: Exception | None = None,
    ) -> None:
        super().__init__(live=live)
        self.deleted: list[str] = []
        self.delete_raises = delete_raises
        self.peer_live = peer_live
        self.recreate_raises = recreate_raises
        self.creates = 0
        # binds answered "bucket not found" before one succeeds: a concurrent owner's delete landing
        # between this owner's create and its bind, and its create a moment later
        self.absent_binds = 0
        self.binds = 0

    async def key_value(self, _name: str) -> Any:
        self.binds += 1
        if self.binds <= self.absent_binds:
            raise BucketNotFoundError()
        return object()

    async def delete_stream(self, name: str) -> bool:
        self.deleted.append(name)
        if self.delete_raises is not None:
            raise self.delete_raises
        return True

    async def add_stream(self, config: StreamConfig) -> Any:
        self.creates += 1
        if self.creates == 1:
            raise _name_in_use()
        if self.recreate_raises is not None:
            raise self.recreate_raises
        if self.peer_live is not None:
            self.live = self.peer_live
            raise _name_in_use()
        self.added.append(config)
        self.live = config
        return object()


def _nonce_bucket_on_file() -> StreamConfig:
    """the nonce bucket as cobalt-dev had it: file storage, a 60s bucket-wide expiry."""
    return _live(storage=StorageType.FILE, allow_direct=True, max_age=60.0)


class TestAnOwnerRecreatesWhatJetStreamCannotChangeInPlace:
    """storage cannot be updated on a live stream; the owner deletes the stream and creates it again."""

    @pytest.mark.asyncio
    async def test_an_owner_without_drop_file_storage_refuses_a_file_bucket_and_leaves_it(self) -> None:
        """a restart keeps a file bucket's entries, so dropping them needs the caller to say so."""
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file())
        with pytest.raises(KvConfigMismatch) as caught:
            await open_kv_stream(
                js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True, owns_bucket=True
            )
        message = str(caught.value)
        assert "probe" in message and "file storage" in message and "drop_file_storage=True" in message, message
        assert js.deleted == [] and js.added == [] and js.updated == [], "a refused owner touched the bucket"

    @pytest.mark.asyncio
    async def test_dropping_file_storage_needs_ownership(self) -> None:
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file())
        with pytest.raises(ValueError, match="drop_file_storage=True needs owns_bucket=True"):
            await open_kv_stream(
                js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True, drop_file_storage=True
            )
        assert js.deleted == []

    @pytest.mark.asyncio
    async def test_the_bind_after_a_recreate_waits_out_a_concurrent_owners_delete(self) -> None:
        """race three: this owner recreated the bucket, and the other owner deleted it before this bind.

        The other owner read the bucket on file before this create, so its delete takes this fresh
        bucket and its create puts it back a moment later. A single bind in that window answered
        not-found and was reported as a bucket this principal could not create; the bind waits for
        the declarer instead, as every binder does.
        """
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file())
        js.absent_binds = 2
        await open_kv_stream(
            js=js,
            full_name="probe",
            config=_memory_declaration(),
            create_if_missing=True,
            owns_bucket=True,
            drop_file_storage=True,
            timings=KvTimings(bind_retry_first_delay_seconds=0.01, bind_retry_max_delay_seconds=0.01),
        )
        assert js.binds == 3, "the bind did not wait for the bucket to come back"

    @pytest.mark.asyncio
    async def test_a_file_bucket_is_recreated_on_memory_with_the_declared_shape(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file())
        with caplog.at_level("WARNING", logger="threetears.nats.kv"):
            await open_kv_stream(
                js=js,
                full_name="probe",
                config=_memory_declaration(),
                create_if_missing=True,
                owns_bucket=True,
                drop_file_storage=True,
            )
        assert js.deleted == ["KV_probe"]
        assert len(js.added) == 1
        assert js.added[0].storage == StorageType.MEMORY
        assert js.added[0].max_age == 0
        assert js.added[0].allow_msg_ttl is True
        assert js.updated == [], "a recreated bucket already carries the declared shape"
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, warnings
        assert "probe" in warnings[0] and "storage" in warnings[0] and "dropped" in warnings[0], warnings
        assert "file" in warnings[0].lower() and "memory" in warnings[0].lower(), warnings

    @pytest.mark.asyncio
    async def test_without_ownership_a_file_bucket_is_reported_and_left(self, caplog: pytest.LogCaptureFixture) -> None:
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file())
        with caplog.at_level("WARNING", logger="threetears.nats.kv"):
            await open_kv_stream(js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True)
        assert js.deleted == [] and js.added == [] and js.updated == []
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("storage" in w and "owns_bucket" in w for w in warnings), warnings

    @pytest.mark.asyncio
    async def test_a_stream_a_concurrent_owner_already_deleted_is_not_an_error(self) -> None:
        """race one: the other owner's delete landed first, so this delete is answered not-found."""
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file(), delete_raises=_stream_not_found())
        await open_kv_stream(
            js=js,
            full_name="probe",
            config=_memory_declaration(),
            create_if_missing=True,
            owns_bucket=True,
            drop_file_storage=True,
        )
        assert js.deleted == ["KV_probe"]
        assert js.added[0].storage == StorageType.MEMORY

    @pytest.mark.asyncio
    async def test_a_bucket_a_concurrent_owner_already_recreated_is_not_deleted_again(self) -> None:
        """race two: the other owner's create landed between this delete and this create.

        Its bucket carries the declared shape; deleting it again would drop what it holds by now for
        nothing, and two owners doing so to each other would never converge.
        """
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file(), peer_live=_live(allow_direct=True))
        await open_kv_stream(
            js=js,
            full_name="probe",
            config=_memory_declaration(),
            create_if_missing=True,
            owns_bucket=True,
            drop_file_storage=True,
        )
        assert js.deleted == ["KV_probe"], "the concurrent owner's fresh bucket was deleted a second time"
        assert js.added == [] and js.updated == []

    @pytest.mark.asyncio
    async def test_a_concurrent_recreate_differing_only_in_place_is_reconciled_in_place(self) -> None:
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file(), peer_live=_live(allow_direct=True, max_age=300.0))
        await open_kv_stream(
            js=js,
            full_name="probe",
            config=_memory_declaration(),
            create_if_missing=True,
            owns_bucket=True,
            drop_file_storage=True,
        )
        assert js.deleted == ["KV_probe"]
        assert len(js.updated) == 1 and js.updated[0].max_age == 0

    @pytest.mark.asyncio
    async def test_a_concurrent_recreate_with_another_storage_is_refused_not_deleted_again(self) -> None:
        """a process recreating it on file meanwhile is a second declarer with another shape: say so."""
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file(), peer_live=_nonce_bucket_on_file())
        with pytest.raises(KvError, match="another process created the bucket again"):
            await open_kv_stream(
                js=js,
                full_name="probe",
                config=_memory_declaration(),
                create_if_missing=True,
                owns_bucket=True,
                drop_file_storage=True,
            )
        assert js.deleted == ["KV_probe"]

    @pytest.mark.asyncio
    async def test_an_unanswered_delete_names_the_grant(self) -> None:
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file(), delete_raises=TimeoutError("nats: timeout"))
        with pytest.raises(KvError, match="grant this principal"):
            await open_kv_stream(
                js=js,
                full_name="probe",
                config=_memory_declaration(),
                create_if_missing=True,
                owns_bucket=True,
                drop_file_storage=True,
            )
        assert js.added == []

    @pytest.mark.asyncio
    async def test_a_refused_recreate_says_the_stream_is_gone(self) -> None:
        js = _OwnedRecreateJetStream(
            live=_nonce_bucket_on_file(), recreate_raises=_ApiError(10074, "insufficient resources")
        )
        with pytest.raises(KvError, match="its stream was deleted and creating it again was refused"):
            await open_kv_stream(
                js=js,
                full_name="probe",
                config=_memory_declaration(),
                create_if_missing=True,
                owns_bucket=True,
                drop_file_storage=True,
            )


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
    async def test_a_reader_binds_a_legacy_bucket_without_per_entry_ttl(self) -> None:
        """Only the declarer enables allow_msg_ttl; a reader must not refuse the bucket meanwhile.

        Refusing would take L2 offline on every reader until the declaring identity rolled. What
        the reader loses instead is per-entry TTL writes, which the server refuses loudly.
        """
        js = _ScriptedJetStream(live=_live(allow_direct=True, allow_msg_ttl=None))
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
        assert js.updated == []

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
    """The self-heal re-open runs after a NATS restart wipes JetStream.

    If it forgot `direct`, the bucket would come back with the field unset --
    every read silently back on the body-carried form no key-scoped grant can
    constrain, and racing whatever startup hook reconciles it.
    """

    @pytest.mark.asyncio
    async def test_a_reopen_recreates_with_the_declared_direct(self) -> None:
        """An operation that hits the vanished stream re-opens the bucket, declaring ``direct`` again.

        :return: nothing
        :rtype: None
        """
        js = _VanishingStreamJetStream()
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

        assert await bucket.get(key="k") == b"healed"

        assert len(js.added) == 1, "the vanished stream did not trigger exactly one re-open"
        assert js.added[0].allow_direct is True


# parity-exempt: nats-py KeyValue stand-in exposing only get, the one call the self-heal test drives
class _VanishedKv:
    """a handle whose stream a broker restart wiped: every read fails as nats-py reports it."""

    async def get(self, _key: str) -> Any:
        raise RuntimeError("nats: no response from stream")


# parity-exempt: nats-py KeyValue stand-in exposing only get, the one call the self-heal test drives
class _HealedKv:
    """the handle a re-open binds, on the recreated stream."""

    async def get(self, _key: str) -> Any:
        return type("_Entry", (), {"value": b"healed", "revision": 1})()


class _VanishingStreamJetStream(_ScriptedJetStream):
    """binds a vanished handle first, and a healed one on every bind after it."""

    def __init__(self) -> None:
        super().__init__()
        self.binds = 0

    async def key_value(self, _name: str) -> Any:
        self.binds += 1
        return _VanishedKv() if self.binds == 1 else _HealedKv()


class _RecreatedElsewhereJetStream(_VanishingStreamJetStream):
    """a vanishing stream that, once :meth:`recreate_elsewhere` runs, another process has created again.

    Every create after that is refused as a name already in use, and the live config is the other
    process's, as the server answers when a stream a restart wiped was put back by someone else first.
    """

    def recreate_elsewhere(self, live: StreamConfig) -> None:
        self.live = live
        self.add_raises = _name_in_use()


def _server_answer(*, code: int, err_code: int, description: str) -> APIError:
    """the exception nats-py raises for a JetStream API error body the server sent.

    built through nats-py's own ``APIError.from_error``, the path every JetStream API reply and
    publish ack takes, so the type (``ServerError`` for a 500, ``ServiceUnavailableError`` for a
    503) and the code are exactly what production sees.

    :param code: the HTTP-like status in the error body
    :ptype code: int
    :param err_code: the server's JetStream error code
    :ptype err_code: int
    :param description: the server's description
    :ptype description: str
    :return: the exception nats-py raised
    :rtype: APIError
    """
    try:
        APIError.from_error({"code": code, "err_code": err_code, "description": description})
    except APIError as raised:
        return raised
    raise AssertionError("nats-py's APIError.from_error returned instead of raising")


def _stream_offline() -> APIError:
    """the server's answer while a NATS restart has the stream offline (``JSStreamOfflineErr``).

    :return: the exception nats-py raises for it
    :rtype: APIError
    """
    return _server_answer(code=500, err_code=10118, description="stream is offline")


class TestAStreamManagementCallTheServerAnsweredIsNotBlamedOnAGrant:
    """the update, delete and create an owner sends: an answered failure is never a missing grant.

    a refused request is never answered, so only a deadline may name the grant. A stream the server
    says is offline is a restart in progress and recovers on its own; any other answer names its own
    cause, which the message must carry.
    """

    @pytest.mark.asyncio
    async def test_an_update_answered_stream_offline_is_reported_as_an_outage(self) -> None:
        js = _ScriptedJetStream(add_raises=_name_in_use(), live=_live(allow_direct=True, max_age=300.0))
        js.update_raises = _stream_offline()
        with pytest.raises(KvError) as caught:
            await open_kv_stream(
                js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True, owns_bucket=True
            )
        message = str(caught.value)
        assert "stream is offline" in message
        assert "temporarily unavailable" in message, message
        assert "grant this principal" not in message, message

    @pytest.mark.asyncio
    async def test_a_delete_answered_stream_offline_is_reported_as_an_outage(self) -> None:
        js = _OwnedRecreateJetStream(live=_nonce_bucket_on_file(), delete_raises=_stream_offline())
        with pytest.raises(KvError) as caught:
            await open_kv_stream(
                js=js,
                full_name="probe",
                config=_memory_declaration(),
                create_if_missing=True,
                owns_bucket=True,
                drop_file_storage=True,
            )
        message = str(caught.value)
        assert "stream is offline" in message
        assert "temporarily unavailable" in message, message
        assert "grant this principal" not in message, message

    @pytest.mark.asyncio
    async def test_a_create_answered_with_its_own_error_then_an_absent_bind_names_that_error(self) -> None:
        """the create is the call that decides: the server answered it, so the grant is not the cause."""
        js = _ScriptedJetStream(
            add_raises=_server_answer(code=503, err_code=10023, description="insufficient resources"),
            bind_raises=BucketNotFoundError(),
        )
        with pytest.raises(KvError) as caught:
            await open_kv_stream(js=js, full_name="probe", config=_memory_declaration(), create_if_missing=True)
        message = str(caught.value)
        assert "insufficient resources" in message, "the server's own answer to the create must survive"
        assert "grant this principal" not in message, message
        assert "temporarily unavailable" not in message, "a capacity answer is not an outage"
