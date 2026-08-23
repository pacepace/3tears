"""JetStream Key-Value bucket primitive.

:class:`NatsKvBucket` is the canonical wrapper around one JetStream KV
bucket. consumers never call ``js.create_key_value`` /
``js.key_value`` directly; they go through
:meth:`threetears.nats.NatsClient.kv_bucket` which returns a
:class:`NatsKvBucket` bound to the connected client's namespace prefix.

design notes
------------

- bucket names are auto-prefixed with the connected client's
  namespace (``{namespace}-{name}``). callers pass the unprefixed
  suffix; the wrapper produces the full bucket name.
- CAS semantics: :meth:`update` returns the new revision on success,
  ``None`` on revision mismatch. transport / bucket-existence
  failures raise :class:`KvError` (distinct from CAS-conflict).
- :meth:`get_entry` returns ``(value, revision)`` for read-modify-write
  patterns; :meth:`get` returns just the value for read-only sites.
- ``ttl=None`` means entries never expire. :class:`timedelta` (not
  raw seconds) for self-documentation.
- opening is CREATE-OR-RECONCILE, not create-or-bind. the bucket's
  JetStream stream shape is built here (:func:`build_kv_stream_config`)
  rather than delegated to ``js.create_key_value``, because
  ``create_key_value`` cannot express every field the platform needs and
  offers no way to reconcile a bucket that already exists. see
  :func:`open_kv_stream`.
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from nats.js.api import DiscardPolicy, StorageType, StreamConfig
from nats.js.errors import KeyNotFoundError, KeyWrongLastSequenceError
from threetears.observe import get_logger

from threetears.nats._diagnostics import kv_grant_remedy, kv_timeout_remedy
from threetears.nats._publish import run_bounded
from threetears.nats.errors import (
    KvConfigMismatch,
    KvError,
    PublishTimeoutError,
    StreamSubjectsOverlapError,
)

if TYPE_CHECKING:
    from nats.js.kv import KeyValue

    from threetears.nats.client import NatsClient

__all__ = [
    "RECONCILED_KV_STREAM_FIELDS",
    "REQUESTABLE_KV_STREAM_FIELDS",
    "KvDeclaring",
    "NatsKvBucket",
    "build_kv_stream_config",
    "kv_stream_differences",
    "open_kv_stream",
]


log = get_logger(__name__)

#: Name of the JetStream stream backing a KV bucket. Mirrors nats-py's
#: ``KV_STREAM_TEMPLATE``, restated rather than imported because that constant is
#: module-level in ``nats.js.client`` and importing it would bind this wrapper to
#: a name nats-py does not document as public.
_KV_STREAM_PREFIX = "KV_"

#: Subject tree a KV bucket owns. Mirrors nats-py's ``KV_PRE_TEMPLATE``.
_KV_SUBJECT_TEMPLATE = "$KV.{bucket}.>"

#: JetStream's own duplicate-tracking window for a KV stream, in seconds.
#: nats-py hardcodes two minutes and narrows it to the bucket TTL when the TTL is
#: shorter; both halves are mirrored by :func:`build_kv_stream_config`.
_KV_DUPLICATE_WINDOW_SECONDS = 120.0

#: JetStream API error code for "stream name already in use with a different
#: configuration". The server answers with this ONLY when it looked at the
#: request and disagreed, which is what makes it usable to tell an existing
#: bucket from a REFUSED one: a refusal is never answered at all, so it arrives
#: as a deadline rather than as an API error. ``ensure_jetstream_stream`` does
#: not make that distinction -- it falls through to ``update_stream`` for every
#: non-overlap failure, refusals included -- so the arm is built here.
_JS_ERR_STREAM_NAME_IN_USE = 10058

#: JetStream API error code for "subjects overlap with an existing stream". A
#: subject belongs to exactly one stream, so this names a DIFFERENT stream
#: squatting ``$KV.{bucket}.>`` and is never fixed by updating this one.
#: Duplicated from ``client.py``'s private copy on purpose: the underscore there
#: is a stability contract, and importing across it is the thing the contract
#: forbids.
_JS_ERR_SUBJECTS_OVERLAP = 10065

#: Stream-config fields a caller can actually ASK for through
#: :meth:`threetears.nats.NatsClient.kv_bucket`, and therefore the only ones
#: whose server-side value is worth reporting when an existing bucket carries
#: something else. Everything outside this set differs between the requested and
#: the server-normalised config on essentially every open (``max_bytes``,
#: ``retention``, ``max_msg_size`` ... all default one way in the dataclass and
#: another on the server), so reporting them would bury the real drift.
REQUESTABLE_KV_STREAM_FIELDS: tuple[str, ...] = (
    "max_age",
    "max_msgs_per_subject",
    "storage",
    "allow_direct",
)

#: The subset of :data:`REQUESTABLE_KV_STREAM_FIELDS` an open RECONCILES -- by
#: updating the stream in place when the opener declares the bucket, and by
#: raising :class:`~threetears.nats.errors.KvConfigMismatch` when it only binds.
#:
#: Deliberately narrow (coll-task-04a KVC-07). A full comparison would raise on
#: every open forever. ``allow_direct`` is here because it is load-bearing for
#: security, not merely for performance: with it false nats-py reads a key by
#: publishing ``$JS.API.STREAM.MSG.GET.KV_{bucket}`` with the key in the request
#: BODY, and NATS authorises on subjects, so no key-scoped ``$KV.`` grant can
#: constrain a read. With it true the read is
#: ``$JS.API.DIRECT.GET.{stream}.{subject}`` and the key is pinnable.
#:
#: Consequence worth stating: anything outside this tuple is set at CREATE and
#: never reconciled afterwards.
RECONCILED_KV_STREAM_FIELDS: tuple[str, ...] = ("allow_direct",)

#: Ceiling on one KV operation, ack included.
#:
#: Module-private, and that is deliberate rather than an oversight: exposing it
#: (as a constant or as a per-call parameter) would grow this package's public
#: surface, which on a patch line is the defect
#: ``tests/enforcement/test_api_growth_requires_a_minor_bump.py`` exists to
#: refuse. A deployment needing a different number is a reason to widen the
#: surface in a MINOR release, not to widen it quietly here.
#:
#: Matches the JetStream publish ceiling because it bounds the same round trip:
#: every KV write ends in ``js.publish`` (``KeyValue.put`` / ``_update`` /
#: ``delete`` / ``purge`` all do), reached through the flush path that discards
#: ``CancelledError``. Reads travel the same path and get the same bound.
_KV_OP_TIMEOUT_SECONDS: float = 10.0

#: How often the KV-timeout remedy may be logged, per bucket.
#:
#: The condition it explains (an ungranted bucket, or an unreachable broker) persists
#: for as long as it persists, producing one timeout per operation. The remedy does not
#: change between them, so repeating it verbatim buries itself.
_TIMEOUT_REMEDY_LOG_INTERVAL_SECONDS: float = 300.0

#: Last time the remedy was logged, keyed by fully-qualified bucket name.
#:
#: Module-level because :class:`NatsKvBucket` declares ``__slots__`` and because the
#: throttle should hold across bucket handles for the same name -- a re-open mints a new
#: instance, and a reconnect loop must not reset the throttle on every attempt.
_last_timeout_remedy_log: dict[str, float] = {}


def build_kv_stream_config(
    *,
    bucket: str,
    ttl_seconds: float,
    history: int,
    storage_type: StorageType,
    direct: bool | None,
) -> StreamConfig:
    """build the JetStream stream shape that MAKES a stream a KV bucket.

    mirrors ``nats.js.client.JetStreamContext.create_key_value`` field for
    field, and exists because that method cannot express the whole shape the
    platform needs: it never sets ``allow_direct`` from anything but its own
    ``KeyValueConfig.direct``, it offers no reconcile for a bucket that already
    exists, and it drops the caller's config entirely on the "already exists"
    path.

    **mirroring, not enumerating.** a hand-kept list of fields goes stale
    silently and the symptom is not a config difference but a stream that stops
    being a bucket -- nats-py's ``key_value()`` validates
    ``max_msgs_per_subject >= 1`` and raises ``BadBucketError``, and everything
    else degrades even more quietly. ``test_kv_stream_shape_matches_nats_py``
    (integration) creates one bucket each way against a live broker and compares
    the resulting server-side stream configs, so drift in nats-py fails a test
    rather than a deployment.

    :param bucket: fully-qualified bucket name, namespace prefix included
    :ptype bucket: str
    :param ttl_seconds: per-entry expiry in seconds; ``0`` means never expire
    :ptype ttl_seconds: float
    :param history: per-key historical revision count
    :ptype history: int
    :param storage_type: JetStream storage backing the bucket
    :ptype storage_type: StorageType
    :param direct: request ``allow_direct``; ``None`` leaves the field unsent,
        so the server decides and no reconcile is attempted on it
    :ptype direct: bool | None
    :return: stream config equivalent to what ``create_key_value`` would send
    :rtype: StreamConfig
    """
    duplicate_window = _KV_DUPLICATE_WINDOW_SECONDS
    if ttl_seconds and ttl_seconds < duplicate_window:
        duplicate_window = ttl_seconds
    return StreamConfig(
        name=f"{_KV_STREAM_PREFIX}{bucket}",
        description=None,
        subjects=[_KV_SUBJECT_TEMPLATE.format(bucket=bucket)],
        allow_direct=direct,
        allow_rollup_hdrs=True,
        allow_msg_ttl=True,
        deny_delete=True,
        discard=DiscardPolicy.NEW,
        duplicate_window=duplicate_window,
        max_age=ttl_seconds,
        max_bytes=None,
        max_consumers=-1,
        # nats-py sends ``max_value_size`` here, which defaults to None and is
        # therefore omitted from the request. the StreamConfig default is -1, so
        # this has to be spelled out to match what create_key_value produces.
        max_msg_size=None,
        max_msgs=-1,
        max_msgs_per_subject=history,
        num_replicas=1,
        storage=storage_type,
        republish=None,
        subject_delete_marker_ttl=None,
    )


def kv_stream_differences(*, requested: StreamConfig, actual: StreamConfig) -> dict[str, tuple[Any, Any]]:
    """compare the fields a KV caller can request, ignoring the ones it cannot.

    a field the caller left unset (``None``) is not a request and is skipped:
    the server fills it and disagreeing with a value nobody asked for is noise.

    :param requested: config this process asked for
    :ptype requested: StreamConfig
    :param actual: config the server reports for the live stream
    :ptype actual: StreamConfig
    :return: field name -> ``(requested, actual)`` for every requested field the
        server does not match
    :rtype: dict[str, tuple[Any, Any]]
    """
    differences: dict[str, tuple[Any, Any]] = {}
    for field in REQUESTABLE_KV_STREAM_FIELDS:
        want = getattr(requested, field, None)
        if want is None:
            continue
        have = getattr(actual, field, None)
        if _normalised(field, want) != _normalised(field, have):
            differences[field] = (want, have)
    return differences


def _normalised(field: str, value: Any) -> Any:
    """map a stream-config value onto the form the server means by it.

    Two fields carry an "absent means something specific" encoding that a naive
    ``!=`` reads as drift: an unsent ``allow_direct`` IS false, and an unsent
    ``max_age`` IS zero (unlimited). Comparing the raw values would report a
    difference on every open of a bucket nats-py created, and the report would
    then be ignored -- which is how a real difference gets missed.

    :param field: the stream-config field name
    :ptype field: str
    :param value: the raw value from either side of the comparison
    :ptype value: Any
    :return: the value in its normalised form
    :rtype: Any
    """
    if field == "allow_direct":
        return bool(value)
    if field == "max_age":
        return float(value or 0.0)
    return value


async def open_kv_stream(
    *,
    js: Any,
    full_name: str,
    config: StreamConfig,
    create_if_missing: bool,
) -> KeyValue:
    """create, reconcile or bind the JetStream stream behind a KV bucket.

    the two modes are the declarer's and the reader's, and they differ on what
    they do about a bucket that already carries a different config:

    - ``create_if_missing=True`` DECLARES. the stream is created when absent;
      when it exists carrying a different value for one of
      :data:`RECONCILED_KV_STREAM_FIELDS`, it is updated in place. drift on any
      other requested field is bound to as-is and logged at WARNING -- the
      previous behaviour dropped it with nothing above DEBUG.
    - ``create_if_missing=False`` BINDS. a reader has no authority to change a
      shared bucket, so drift on the reconciled set raises
      :class:`~threetears.nats.errors.KvConfigMismatch`, which the L2 accessors
      deliberately do not catch.

    a create failure the SERVER answered is classified from its API error code;
    a failure the server never answered (a permissions refusal reads as a
    deadline, not as an error) falls through to the bind, whose own failure is
    what proves the bucket is ungranted rather than merely present.

    :param js: connected JetStream context
    :ptype js: Any
    :param full_name: fully-qualified bucket name, namespace prefix included
    :ptype full_name: str
    :param config: the KV stream shape from :func:`build_kv_stream_config`
    :ptype config: StreamConfig
    :param create_if_missing: declare the bucket when absent, rather than bind
    :ptype create_if_missing: bool
    :return: bound nats-py KeyValue handle
    :rtype: KeyValue
    :raises StreamSubjectsOverlapError: a different stream already owns ``$KV.{bucket}.>``
    :raises KvConfigMismatch: bind-only open found drift on the reconciled set
    :raises KvError: creation or binding failed
    """
    if not create_if_missing:
        return await _bind_kv_stream(js=js, full_name=full_name, config=config)

    add_exc: Exception | None = None
    try:
        await js.add_stream(config)
    except Exception as exc:  # noqa: BLE001 -- classified below, never swallowed
        _raise_if_subjects_overlap(exc, full_name=full_name, config=config)
        add_exc = exc
    if add_exc is None:
        log.info(
            "JetStream KV bucket created",
            extra={"extra_data": {"bucket": full_name, "allow_direct": config.allow_direct}},
        )
    elif getattr(add_exc, "err_code", None) == _JS_ERR_STREAM_NAME_IN_USE:
        await _reconcile_existing_kv_stream(js=js, full_name=full_name, config=config)
    # else: the server never answered the create. That is what a permissions
    # refusal looks like -- and what an unreachable broker looks like -- so it is
    # NOT reconcilable and must not fall through to update_stream, which would be
    # refused in turn and turn one deadline into two. Fall through to the bind
    # instead: a bucket this principal may read but not create binds fine, and a
    # bind that fails too is what proves the grant is missing.
    kv: KeyValue
    try:
        kv = await js.key_value(full_name)
    except Exception as bind_exc:
        raise KvError(
            f"open KV bucket failed: bucket={full_name}: create={add_exc!r} bind={bind_exc!r}. "
            f"{kv_grant_remedy(full_name)}"
        ) from bind_exc
    return kv


async def _bind_kv_stream(*, js: Any, full_name: str, config: StreamConfig) -> KeyValue:
    """bind to an existing KV bucket, refusing one whose reconciled config differs.

    :param js: connected JetStream context
    :ptype js: Any
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param config: the config this reader requires
    :ptype config: StreamConfig
    :return: bound nats-py KeyValue handle
    :rtype: KeyValue
    :raises KvConfigMismatch: the live bucket differs on the reconciled field set
    :raises KvError: binding failed
    """
    kv: KeyValue
    try:
        kv = await js.key_value(full_name)
    except Exception as exc:
        # Hedged: this branch never attempts a create, so a bucket nobody has
        # created yet fails here exactly the way an ungranted one does.
        raise KvError(
            f"bind KV bucket failed: bucket={full_name}: {exc}. {kv_grant_remedy(full_name, certain=False)}"
        ) from exc
    if any(getattr(config, field, None) is not None for field in RECONCILED_KV_STREAM_FIELDS):
        live = await _live_stream_config(js=js, full_name=full_name, stream=config.name)
        drift = {
            field: value
            for field, value in kv_stream_differences(requested=config, actual=live).items()
            if field in RECONCILED_KV_STREAM_FIELDS
        }
        if drift:
            raise KvConfigMismatch(
                f"KV bucket {full_name!r} is live with a configuration this process refuses to run "
                f"against: {_render_differences(drift)}. this process opened the bucket read-only "
                f"(create_if_missing=False), so it cannot reconcile it. the DECLARING identity -- "
                f"the one that calls ensure_kv_bucket at startup -- must run first and reconcile "
                f"the bucket, or the bucket must be recreated with the declared configuration."
            )
    return kv


async def _reconcile_existing_kv_stream(*, js: Any, full_name: str, config: StreamConfig) -> None:
    """update a live KV stream in place, or say loudly which request was dropped.

    the previous opener bound to whatever existed and logged the fact at DEBUG,
    so a bucket carrying somebody else's config was indistinguishable from one
    carrying yours. everything outside :data:`RECONCILED_KV_STREAM_FIELDS` is
    still bound to as-is -- reconciling every field would let two processes with
    different requests fight over one bucket -- but it is now reported at
    WARNING rather than dropped in silence.

    :param js: connected JetStream context
    :ptype js: Any
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param config: the config this process asked for
    :ptype config: StreamConfig
    :return: nothing
    :rtype: None
    :raises KvError: the live config could not be read, or the update was refused
    """
    live = await _live_stream_config(js=js, full_name=full_name, stream=config.name)
    differences = kv_stream_differences(requested=config, actual=live)
    reconcilable = {field: value for field, value in differences.items() if field in RECONCILED_KV_STREAM_FIELDS}
    if reconcilable:
        # update_stream sends the whole requested config, so this applies every
        # difference, not only the reconciled ones. That is the declarer's job.
        try:
            await js.update_stream(config)
        except Exception as exc:
            # A principal may be granted STREAM.CREATE and refused STREAM.UPDATE --
            # that is exactly the shape coll-task-05a gives pods. Say which grant is
            # missing rather than letting a raw nats-py error out of the opener.
            raise KvError(
                f"reconciling KV bucket {full_name!r} failed: {exc}. it is live with "
                f"{_render_differences(reconcilable)} and this principal could not update it. "
                f"{kv_grant_remedy(full_name)}"
            ) from exc
        log.info(
            "JetStream KV bucket reconciled in place",
            extra={"extra_data": {"bucket": full_name, "applied": _render_differences(differences)}},
        )
    elif differences:
        log.warning(
            "JetStream KV bucket bound to an existing stream whose configuration differs from the "
            "requested one; the requested values were NOT applied: bucket=%s %s",
            full_name,
            _render_differences(differences),
            extra={"extra_data": {"bucket": full_name, "dropped": _render_differences(differences)}},
        )
    else:
        log.debug(
            "JetStream KV bucket bound (already existed)",
            extra={"extra_data": {"bucket": full_name}},
        )


async def _live_stream_config(*, js: Any, full_name: str, stream: str | None) -> StreamConfig:
    """read the server's config for a KV bucket's backing stream.

    wraps the lookup so a refused or unreachable ``STREAM.INFO`` leaves the
    opener as a typed :class:`KvError` naming the grant, rather than as whatever
    nats-py happened to raise.

    :param js: connected JetStream context
    :ptype js: Any
    :param full_name: fully-qualified bucket name, for the message
    :ptype full_name: str
    :param stream: the backing stream's name
    :ptype stream: str | None
    :return: the live stream config
    :rtype: StreamConfig
    :raises KvError: the lookup failed
    """
    try:
        info = await js.stream_info(stream)
    except Exception as exc:
        raise KvError(
            f"reading the live configuration of KV bucket {full_name!r} failed: {exc}. {kv_grant_remedy(full_name)}"
        ) from exc
    config: StreamConfig = info.config
    return config


def _raise_if_subjects_overlap(exc: Exception, *, full_name: str, config: StreamConfig) -> None:
    """re-raise a subjects-overlap add failure as its own typed error.

    a subject belongs to exactly one stream, so this names a DIFFERENT stream
    squatting the bucket's subject tree. updating ``KV_{bucket}`` cannot fix it
    and would report a confusing not-found instead.

    :param exc: the exception ``add_stream`` raised
    :ptype exc: Exception
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param config: the config that was refused
    :ptype config: StreamConfig
    :return: nothing
    :rtype: None
    :raises StreamSubjectsOverlapError: when ``exc`` is the overlap refusal
    """
    if getattr(exc, "err_code", None) != _JS_ERR_SUBJECTS_OVERLAP and "subjects overlap" not in str(exc).lower():
        return
    raise StreamSubjectsOverlapError(
        f"cannot create KV bucket {full_name!r} over subjects {config.subjects}: they overlap "
        f"subjects already claimed by a different stream on this NATS account (a subject belongs "
        f"to exactly one stream). the usual cause is another connection using the wrong subject "
        f"namespace; resolve the conflicting stream or correct the namespace."
    ) from exc


def _render_differences(differences: dict[str, tuple[Any, Any]]) -> str:
    """format a drift map for a log line or an exception message.

    :param differences: field name -> ``(requested, actual)``
    :ptype differences: dict[str, tuple[Any, Any]]
    :return: human-readable one-line summary
    :rtype: str
    """
    return ", ".join(
        f"{field}: requested={want!r} live={have!r}" for field, (want, have) in sorted(differences.items())
    )


class NatsKvBucket:
    """one JetStream KV bucket.

    instances are produced by :meth:`NatsClient.kv_bucket`; the bare
    constructor is internal. instances are reusable for the client's
    lifetime; do not cache across client recreations.

    :param client: connected wrapper client owning this bucket
    :ptype client: NatsClient
    :param full_name: fully-qualified bucket name (``{namespace}-{suffix}``)
    :ptype full_name: str
    :param kv: underlying nats-py KeyValue handle
    :ptype kv: KeyValue
    :param ttl: configured time-to-live, or ``None`` for no expiry
    :ptype ttl: timedelta | None
    """

    __slots__ = ("_client", "_create_if_missing", "_direct", "_full_name", "_history", "_kv", "_storage", "_ttl")

    def __init__(
        self,
        *,
        client: NatsClient,
        full_name: str,
        kv: KeyValue,
        ttl: timedelta | None,
        storage: str = "memory",
        create_if_missing: bool = True,
        history: int = 1,
        direct: bool | None = None,
    ) -> None:
        self._client = client
        self._full_name = full_name
        self._kv = kv
        self._ttl = ttl
        # Retained for self-heal: if the underlying stream/bucket vanishes (a NATS restart
        # on ephemeral storage wipes JetStream), an op can re-open the bucket with its
        # original config and retry instead of failing forever on a dead cached handle.
        self._storage = storage
        self._create_if_missing = create_if_missing
        self._history = history
        # Retained for the same reason, and load-bearing for security rather than
        # for shape: a re-open that forgot ``direct`` would recreate the wiped
        # bucket with allow_direct unset, silently putting every read back on the
        # body-carried form no key-scoped grant can constrain.
        self._direct = direct

    @property
    def name(self) -> str:
        """fully-qualified bucket name (with namespace prefix).

        :return: bucket name as registered with JetStream
        :rtype: str
        """
        return self._full_name

    @property
    def ttl(self) -> timedelta | None:
        """configured time-to-live for entries in this bucket.

        :return: TTL or ``None`` for no expiry
        :rtype: timedelta | None
        """
        return self._ttl

    # ------------------------------------------------------------------
    # opener (internal — used by NatsClient.kv_bucket)
    # ------------------------------------------------------------------

    @classmethod
    async def open(
        cls,
        *,
        client: NatsClient,
        full_name: str,
        ttl: timedelta | None,
        storage: str,
        create_if_missing: bool,
        history: int,
        direct: bool | None = None,
    ) -> NatsKvBucket:
        """open, create or reconcile a JetStream KV bucket.

        called by :meth:`NatsClient.kv_bucket` and
        :meth:`NatsClient.ensure_kv_bucket`. excluded from
        ``threetears.nats.__all__`` because callers should not bypass
        the client's bucket cache; the public path is the client.

        the create branch goes through :func:`open_kv_stream` rather than
        ``js.create_key_value``, and that matters most for :meth:`_reopen`:
        after a NATS restart wipes JetStream, a re-open on
        ``create_key_value`` would silently recreate the bucket with
        ``allow_direct`` unset -- putting every read back on the body-carried
        form and racing whatever startup hook reconciles it.

        :param client: connected wrapper client
        :ptype client: NatsClient
        :param full_name: fully-qualified bucket name
        :ptype full_name: str
        :param ttl: TTL for entries; ``None`` for no expiry
        :ptype ttl: timedelta | None
        :param storage: ``"file"`` or ``"memory"``
        :ptype storage: str
        :param create_if_missing: create (and reconcile) bucket rather than bind read-only
        :ptype create_if_missing: bool
        :param history: per-key historical revision count
        :ptype history: int
        :param direct: request ``allow_direct`` on the backing stream; ``None``
            neither requests nor compares it
        :ptype direct: bool | None
        :return: ready bucket
        :rtype: NatsKvBucket
        :raises KvError: if bucket creation or binding fails
        :raises KvConfigMismatch: if a bind-only open finds a reconciled field differing
        :raises StreamSubjectsOverlapError: if a different stream owns the bucket's subjects
        """
        js = client.jetstream_context()
        storage_type = StorageType.FILE if storage == "file" else StorageType.MEMORY
        ttl_seconds = int(ttl.total_seconds()) if ttl is not None else 0

        kv = await open_kv_stream(
            js=js,
            full_name=full_name,
            config=build_kv_stream_config(
                bucket=full_name,
                ttl_seconds=ttl_seconds,
                history=history,
                storage_type=storage_type,
                direct=direct,
            ),
            create_if_missing=create_if_missing,
        )

        return cls(
            client=client,
            full_name=full_name,
            kv=kv,
            ttl=ttl,
            storage=storage,
            create_if_missing=create_if_missing,
            history=history,
            direct=direct,
        )

    # ------------------------------------------------------------------
    # self-heal
    # ------------------------------------------------------------------

    async def _reopen(self) -> None:
        """Rebind ``self._kv`` after the underlying stream/bucket vanished.

        A single-node NATS restart on ephemeral JetStream storage wipes every stream and
        KV bucket. The client caches this bucket handle, so without a re-open every op on
        it fails forever ("nats: no response from stream") until the process restarts --
        which is what silenced the wake scheduler in production. Re-running the opener with
        the bucket's original config recreates the bucket (when ``create_if_missing``) and
        refreshes the handle in place, so the cached bucket self-heals; no client-cache
        flush is needed because the same object is mutated.

        ``direct`` rides along with the rest of the stored config, so a self-heal
        recreates the bucket with the same ``allow_direct`` it was declared with
        rather than with the field unset.
        """
        rebound = await NatsKvBucket.open(
            client=self._client,
            full_name=self._full_name,
            ttl=self._ttl,
            storage=self._storage,
            create_if_missing=self._create_if_missing,
            history=self._history,
            direct=self._direct,
        )
        self._kv = rebound._kv  # noqa: SLF001 - sibling instance of the same class

    async def _run_with_reopen(self, op: Any, *, passthrough: tuple[type[BaseException], ...]) -> Any:
        """Run a KV op; on a TRANSPORT failure, re-open the bucket once and retry.

        ``passthrough`` exceptions (KeyNotFound / CAS-mismatch) are normal control flow and
        are re-raised immediately -- only an unexpected failure (a vanished stream, a
        transient transport error) triggers the single re-open + retry. A second failure
        propagates to the caller's ``KvError`` wrap.
        """
        try:
            return await self._bounded(op)
        except passthrough:
            raise
        except PublishTimeoutError:
            # A wedged operation is not a vanished bucket. Re-opening runs another KV call
            # against the same unresponsive broker, so the retry wedges too -- one deadline
            # becomes two, and the caller waits twice as long to learn the same thing.
            #
            # Logged here rather than left to the caller because the deadline is where the
            # ambiguity lives: an ungranted bucket and a dead broker produce the identical
            # timeout, and only this frame knows which bucket to name in the fix.
            #
            # Rate-limited per bucket. The remedy is long and it is the same remedy every
            # time; the condition that produces it produces one per operation, for as long
            # as it lasts. Emitting it unthrottled would bury the diagnosis inside its own
            # repetitions -- the failure the caller still hits every time is the raised
            # PublishTimeoutError, not this line.
            self._log_timeout_remedy()
            raise
        except Exception:  # noqa: BLE001 - transport failure: self-heal once, then let it surface
            await self._reopen()
            return await self._bounded(op)

    def _log_timeout_remedy(self) -> None:
        """Emit the ungranted-bucket-or-dead-broker remedy, at most once per window per bucket.

        :return: nothing
        :rtype: None
        """
        now = time.monotonic()
        last = _last_timeout_remedy_log.get(self._full_name)
        # ABSENCE means never logged, not ``0.0``. ``time.monotonic()`` is time since
        # BOOT on Linux, so on a freshly-started machine it is a small number and
        # ``now - 0.0`` is under the interval -- which suppressed the FIRST remedy for
        # the first five minutes of a process's life, exactly when a missing grant is
        # most likely to be the thing that is wrong. It reads as correct on any
        # long-lived developer machine and fails only where it matters.
        if last is not None and now - last < _TIMEOUT_REMEDY_LOG_INTERVAL_SECONDS:
            log.debug("KV operation timed out on %s (remedy already logged)", self._full_name)
            return
        _last_timeout_remedy_log[self._full_name] = now
        log.error(kv_timeout_remedy(self._full_name), extra={"extra_data": {"bucket": self._full_name}})

    async def _bounded(self, op: Any) -> Any:
        """Run one KV op under a deadline the operation cannot swallow.

        **Every KV operation reaches the broker through the same path a JetStream publish
        does**, and that path discards ``CancelledError`` (see
        :mod:`threetears.nats._publish`). So an unresponsive broker hangs a KV call forever
        and the caller's own ``asyncio.wait_for`` cannot break it -- the identical defect
        that froze a downstream fleet through ``jetstream_publish``, at a different call
        site. ``KeyValue.put`` is literally ``await self._js.publish(...)`` with no timeout.

        This matters most for :func:`~threetears.nats.distributed_lock.nats_distributed_lock`,
        which is KV-backed: a wedged heartbeat or release holds the lock for the length of
        the wedge, so ONE stuck pod blocks every other pod's turn at it.

        :param op: builds the KV coroutine to run
        :ptype op: Any
        :return: whatever the operation returned
        :rtype: Any
        :raises PublishTimeoutError: the operation blew its deadline or ignored cancellation
        """
        return await run_bounded(op, timeout=_KV_OP_TIMEOUT_SECONDS, what=f"kv operation on {self._full_name}")

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------

    async def get(self, *, key: str) -> bytes | None:
        """get value for a key.

        returns ``None`` on miss. transport failures raise
        :class:`KvError`.

        :param key: key to read
        :ptype key: str
        :return: stored bytes or ``None`` if absent
        :rtype: bytes | None
        :raises KvError: on transport failure
        """
        try:
            entry = await self._run_with_reopen(lambda: self._kv.get(key), passthrough=(KeyNotFoundError,))
        except KeyNotFoundError:
            # NOSILENT: a miss is this method's documented result, reported to the caller as None
            return None
        except Exception as exc:
            raise KvError(f"KV get failed: bucket={self._full_name} key={key}: {exc}") from exc
        return bytes(entry.value) if entry.value is not None else None

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        """get value + revision for CAS read-modify-write.

        :param key: key to read
        :ptype key: str
        :return: tuple of (value bytes, revision) or ``None`` if absent
        :rtype: tuple[bytes, int] | None
        :raises KvError: on transport failure
        """
        try:
            entry = await self._run_with_reopen(lambda: self._kv.get(key), passthrough=(KeyNotFoundError,))
        except KeyNotFoundError:
            # NOSILENT: a miss is this method's documented result, reported to the caller as None
            return None
        except Exception as exc:
            raise KvError(f"KV get_entry failed: bucket={self._full_name} key={key}: {exc}") from exc
        if entry.value is None or entry.revision is None:
            return None
        return (bytes(entry.value), int(entry.revision))

    async def put(self, *, key: str, value: bytes) -> int:
        """unconditional write. returns new revision.

        :param key: key to write
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :return: new revision number
        :rtype: int
        :raises KvError: on transport failure
        """
        try:
            revision = await self._run_with_reopen(lambda: self._kv.put(key, value), passthrough=())
        except Exception as exc:
            raise KvError(f"KV put failed: bucket={self._full_name} key={key}: {exc}") from exc
        return int(revision)

    async def create(self, *, key: str, value: bytes) -> int | None:
        """create-if-absent (SET NX). returns new revision or ``None`` on conflict.

        :param key: key to create
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :return: new revision number, or ``None`` if key already exists
        :rtype: int | None
        :raises KvError: on transport failure
        """
        try:
            revision = await self._run_with_reopen(
                lambda: self._kv.create(key, value), passthrough=(KeyWrongLastSequenceError,)
            )
        except KeyWrongLastSequenceError:
            # A lost create is a documented result (None), but a burst of them is contention on
            # one key -- two writers racing a lock or a leader election -- which only shows up
            # here. The value is not logged; the key is structural, the same identifier already
            # carried by the KvError below.
            log.debug(
                "KV create lost: key already exists",
                extra={"extra_data": {"bucket": self._full_name, "key": key}},
            )
            return None
        except Exception as exc:
            raise KvError(f"KV create failed: bucket={self._full_name} key={key}: {exc}") from exc
        return int(revision)

    async def update(self, *, key: str, value: bytes, revision: int) -> int | None:
        """compare-and-swap update. returns new revision or ``None`` on revision-mismatch.

        :param key: key to update
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :param revision: expected current revision
        :ptype revision: int
        :return: new revision number, or ``None`` if expected revision did not match
        :rtype: int | None
        :raises KvError: on transport failure
        """
        try:
            new_revision = await self._run_with_reopen(
                lambda: self._kv.update(key, value, revision), passthrough=(KeyWrongLastSequenceError,)
            )
        except KeyWrongLastSequenceError:
            # A lost CAS is a documented result (None), but a burst of them is contention on one
            # key, and a caller that never retries would otherwise drop the write in silence.
            log.debug(
                "KV update lost: revision mismatch",
                extra={"extra_data": {"bucket": self._full_name, "key": key, "expected_revision": revision}},
            )
            return None
        except Exception as exc:
            raise KvError(f"KV update failed: bucket={self._full_name} key={key} rev={revision}: {exc}") from exc
        return int(new_revision)

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        """delete a key, optionally guarded by a CAS revision.

        when ``revision`` is supplied the underlying nats-py call
        becomes a compare-and-swap delete: the delete only succeeds if
        the stored revision matches. on revision mismatch the method
        returns ``False`` (analogous to :meth:`update` returning
        ``None``); on a missing key it returns ``True`` (delete is
        idempotent). transport failures raise :class:`KvError`.

        :param key: key to delete
        :ptype key: str
        :param revision: expected current revision for CAS delete; ``None`` performs an unconditional delete
        :ptype revision: int | None
        :return: ``True`` on success or absent key, ``False`` only on revision mismatch
        :rtype: bool
        :raises KvError: on transport failure
        """

        async def _do_delete() -> None:
            if revision is None:
                await self._kv.delete(key)
            else:
                await self._kv.delete(key, last=revision)

        try:
            await self._run_with_reopen(_do_delete, passthrough=(KeyNotFoundError, KeyWrongLastSequenceError))
        except KeyNotFoundError:
            return True
        except KeyWrongLastSequenceError:
            return False
        except Exception as exc:
            raise KvError(f"KV delete failed: bucket={self._full_name} key={key} revision={revision}: {exc}") from exc
        return True


@runtime_checkable
class KvBucketLike(Protocol):
    """The bucket surface a KV consumer actually uses.

    Structurally identical to :class:`NatsKvBucket`'s public operations, and to the
    in-memory ``FakeKvBucket`` the testing package ships. It exists so a consumer can
    declare the slice it needs instead of the concrete class: the two are interchangeable
    at every call site in the platform, and a signature naming the concrete one forces a
    ``cast`` or a ``type: ignore`` on every test that passes the double.
    """

    @property
    def name(self) -> str: ...

    @property
    def ttl(self) -> timedelta | None: ...

    async def get(self, *, key: str) -> bytes | None: ...

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None: ...

    async def put(self, *, key: str, value: bytes) -> int: ...

    async def create(self, *, key: str, value: bytes) -> int | None: ...

    async def update(self, *, key: str, value: bytes, revision: int) -> int | None: ...

    async def delete(self, *, key: str, revision: int | None = None) -> bool: ...


@runtime_checkable
class KvCapable(Protocol):
    """A client that can open KV buckets -- the slice of :class:`NatsClient` that KV code needs.

    Most consumers of a NATS client only ever call ``kv_bucket``: a rate limiter, a replay
    guard, a distributed lock. Naming this instead of ``NatsClient`` says what is actually
    required, and lets the shipped in-memory double satisfy the signature by construction
    rather than by exemption.

    The full ``kv_bucket`` signature is declared rather than a two-argument subset: several
    coordination primitives pass ``storage`` / ``create_if_missing`` / ``history``, and a
    Protocol that omitted them would reject those callers. The shipped in-memory double
    mirrors the same signature, so both satisfy this by construction.
    """

    async def kv_bucket(
        self,
        *,
        name: str,
        ttl: timedelta | None = None,
        storage: str = "memory",
        create_if_missing: bool = True,
        history: int = 1,
    ) -> KvBucketLike: ...


@runtime_checkable
class KvDeclaring(Protocol):
    """A client that can DECLARE or BIND a KV bucket -- the other half of the KV slice.

    Sits beside :class:`KvCapable` and exists for the same stated reason: naming the slice a
    consumer actually requires, rather than the whole :class:`NatsClient`, is what lets an
    in-memory double satisfy the signature by construction instead of by exemption.

    SEPARATE from :class:`KvCapable` rather than folded into it. ``KvCapable`` declares
    ``kv_bucket`` only -- the ordinary open -- and its own docstring rests on that ("most
    consumers of a NATS client only ever call ``kv_bucket``"). Adding a second required method
    there would un-satisfy every double that implements the one method it advertises, across
    every consuming repo, to serve the few callers whose subject is the declaration.
    """

    async def ensure_kv_bucket(
        self,
        *,
        name: str,
        ttl: timedelta | None = None,
        storage: str = "memory",
        history: int = 1,
        direct: bool = True,
        create_if_missing: bool = True,
    ) -> KvBucketLike:
        """declare a KV bucket's configuration, or bind to one somebody else declared.

        :param name: bucket name suffix, namespace-prefixed by the implementation
        :ptype name: str
        :param ttl: time-to-live for entries, or ``None`` for no expiry
        :ptype ttl: timedelta | None
        :param storage: ``"memory"`` or ``"file"``
        :ptype storage: str
        :param history: historical revisions kept per key
        :ptype history: int
        :param direct: the ``allow_direct`` value the bucket must carry
        :ptype direct: bool
        :param create_if_missing: ``True`` declares (create + reconcile); ``False`` binds and
            REFUSES a bucket whose reconciled config differs
        :ptype create_if_missing: bool
        :return: ready bucket handle
        :rtype: KvBucketLike
        :raises KvError: if bucket creation or binding fails
        :raises KvConfigMismatch: if ``create_if_missing`` is ``False`` and the live bucket differs
        """
        ...


@runtime_checkable
class JetStreamPublisher(Protocol):
    """The slice of :class:`NatsClient` a durable-publish caller needs.

    One method, because that is genuinely all an audit emitter or any other
    fire-and-persist producer uses. Sits beside :class:`KvCapable` for the same reason:
    a signature naming the whole client forces every test that passes a recording double
    to launder it through a cast, which is a hole in exactly the tests that exist to prove
    the right thing was published.
    """

    async def jetstream_publish(self, *, subject: Any, payload: bytes) -> None: ...
