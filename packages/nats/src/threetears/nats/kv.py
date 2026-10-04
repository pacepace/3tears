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
  failures raise :class:`KvError` (distinct from CAS-conflict); a bucket
  the server answered is absent raises the narrower
  :class:`~threetears.nats.errors.KvBucketNotFoundError`, still a ``KvError``.
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
- a key is watched through :meth:`NatsKvBucket.watch_key`, never through
  nats-py's ``KeyValue.watch``: the stock watch creates an UNNAMED consumer,
  which a grant narrowed to one key cannot admit. see
  :mod:`threetears.nats.kv_watch`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol, runtime_checkable

from nats.errors import ConnectionClosedError as _NatsConnectionClosedError
from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    DeliverPolicy,
    DiscardPolicy,
    Header,
    StorageType,
    StreamConfig,
    StreamInfo,
)
from nats.js.errors import APIError, KeyNotFoundError, KeyWrongLastSequenceError
from threetears.observe import get_logger
from threetears.observe.resilience import retry_bounded

from threetears.nats.diagnostics import kv_grant_remedy, kv_timeout_remedy
from threetears.nats._publish import run_bounded
from threetears.nats.errors import (
    KvBucketNotFoundError,
    KvConfigMismatch,
    KvError,
    NatsClientError,
    PublishTimeoutError,
    StreamSubjectsOverlapError,
)
from threetears.nats.kv_watch import DEFAULT_KEY_WATCH_HEARTBEAT, DEFAULT_KEY_WATCH_RETRY, KvKeyUpdate
from threetears.nats.raw_errors import is_bucket_not_found

if TYPE_CHECKING:
    from nats.aio.msg import Msg
    from nats.aio.subscription import Subscription as _NatsSubscription
    from nats.js.kv import KeyValue

    from threetears.nats.client import NatsClient

__all__ = [
    "DEFAULT_KV_TIMINGS",
    "RECONCILED_KV_STREAM_FIELDS",
    "REQUESTABLE_KV_STREAM_FIELDS",
    "KvDeclaring",
    "KvTimings",
    "NatsKvBucket",
    "build_kv_stream_config",
    "kv_stream_differences",
    "open_kv_stream",
    "reconcile_kv_stream",
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

#: JetStream's refusals of a publish whose expected last subject sequence did not match --
#: ``JSStreamWrongLastSequenceErrF`` and ``JSStreamWrongLastSequenceConstantErr``. Both mean a lost
#: compare-and-swap, the same codes nats-py maps to ``KeyWrongLastSequenceError``.
_JS_ERR_WRONG_LAST_SEQUENCE: frozenset[int] = frozenset({10071, 10164})

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
    "allow_msg_ttl",
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
#: ``allow_msg_ttl`` is here because per-entry lifetimes are what bound L2 memory for entries
#: nothing ever deletes -- a collection's negative-cache markers -- and buckets created before
#: this module set it carry it off. Enabling it is safe on a live stream and cannot be undone,
#: which is fine: it only permits a header, it changes no existing entry.
#:
#: ``max_age``, ``max_msgs_per_subject`` and ``storage`` are NOT here, and stay out for every
#: declarer that does not claim the bucket: a bucket with several declarers asking for different
#: shapes would have them fight over it, each open rewriting the other's. A bucket whose shape has
#: exactly ONE owner has no such fight, so that owner may opt in (``owns_bucket=True`` on
#: :meth:`threetears.nats.NatsClient.ensure_kv_bucket`) and its declaration then reconciles the
#: whole requested shape -- see :data:`_OWNED_IN_PLACE_KV_STREAM_FIELDS` and
#: :data:`_OWNED_RECREATE_KV_STREAM_FIELDS`.
#:
#: Consequence worth stating: anything outside this tuple, for a declarer that does not own the
#: bucket, is set at CREATE and never reconciled afterwards.
RECONCILED_KV_STREAM_FIELDS: tuple[str, ...] = ("allow_direct", "allow_msg_ttl")

#: The fields a declaration that owns its bucket (``owns_bucket=True``) reconciles IN PLACE, in
#: addition to :data:`RECONCILED_KV_STREAM_FIELDS`: the ones JetStream accepts an update of on a
#: live stream.
#:
#: ``max_age`` because a bind-only opener refuses a bucket whose bucket-wide expiry differs from the
#: lifetime it asks for (:func:`_entry_ttl_for_bound_bucket`): a bucket created long ago with one
#: keeps refusing every such opener for as long as nothing removes it, and only the bucket's one
#: declarer may remove it. ``max_msgs_per_subject`` (the declared history) because the owner of the
#: whole shape owns that too.
_OWNED_IN_PLACE_KV_STREAM_FIELDS: tuple[str, ...] = ("max_age", "max_msgs_per_subject")

#: The fields a declaration that owns its bucket reconciles by DELETING the backing stream and
#: creating it again with the declared config: the ones JetStream refuses to change on a live stream.
#:
#: Every entry the bucket held is dropped. That is acceptable only because an owner must declare
#: the bucket on memory storage: its contents are ephemeral by design, a NATS restart already wipes
#: them, and every binder already survives finding the bucket emptied -- so a recreate is the
#: restart the platform already survives, applied to one bucket.
_OWNED_RECREATE_KV_STREAM_FIELDS: tuple[str, ...] = ("storage",)

#: The subset of :data:`RECONCILED_KV_STREAM_FIELDS` a BIND-only open refuses to run against.
#:
#: ``allow_msg_ttl`` is reconciled by the declarer but not refused by a binder. A process that
#: binds a bucket its declarer has not yet reconciled loses nothing but per-entry TTL writes, which
#: the server refuses loudly and every caller treats as an ordinary failed write; refusing the bind
#: instead would take L2 offline on every such process until the declarer rolled.
_BIND_REFUSED_KV_STREAM_FIELDS: tuple[str, ...] = ("allow_direct",)


@dataclasses.dataclass(frozen=True, slots=True)
class KvTimings:
    """the deadlines and paces a KV bucket runs its operations and binds under.

    One value per client: :meth:`threetears.nats.NatsClient.connect` takes it as ``kv_timings`` and
    every bucket the client opens carries it. The defaults are the production values; a host
    changes one only for a reason it can name (a test that must reach a deadline in milliseconds, a
    deployment whose declarer is known to come up slower).

    :ivar op_timeout_seconds: ceiling on one KV operation, ack included. Matches the JetStream
        publish ceiling because it bounds the same round trip: every KV write ends in
        ``js.publish`` (``KeyValue.put`` / ``_update`` / ``delete`` / ``purge`` all do), reached
        through the flush path that discards ``CancelledError``. Reads travel the same path and get
        the same bound.
    :ivar timeout_remedy_log_interval_seconds: how often the KV-timeout remedy may be logged, per
        bucket. The condition it explains (an ungranted bucket, or an unreachable broker) persists
        for as long as it persists, producing one timeout per operation; the remedy does not change
        between them, so repeating it verbatim buries itself.
    :ivar bind_wait_for_declarer_seconds: how long a BIND-only open waits for an absent bucket to be
        declared before it fails. A process that only binds a bucket never creates it, so a bucket
        missing at bind time is one its declarer has not declared YET: at first boot before the hub
        has run, or after a NATS restart wiped every memory-backed bucket and before the hub's
        reconnect re-declared them. Failing on the first miss left a primitive bound once at
        startup unusable until the process restarted. Long enough to cover the hub reconnecting and
        re-declaring after a broker restart; short enough that a bucket nobody will ever declare
        surfaces as an error an operator can read. Every later operation on a handle re-binds
        through the same wait, so an expiry here is never permanent.
    :ivar bind_retry_first_delay_seconds: the first pause between two binds of an absent bucket; it
        doubles up to ``bind_retry_max_delay_seconds``
    :ivar bind_retry_max_delay_seconds: the longest pause between two binds of an absent bucket
    :ivar key_listing_timeout_seconds: how long ONE key listing attempt may take, end to end, before
        it raises rather than hangs. A listing whose consumer create is ungranted is never answered,
        so without a bound it would block forever. A re-open between two attempts runs outside it,
        on ``bind_wait_for_declarer_seconds`` of its own
    """

    op_timeout_seconds: float = 10.0
    timeout_remedy_log_interval_seconds: float = 300.0
    bind_wait_for_declarer_seconds: float = 30.0
    bind_retry_first_delay_seconds: float = 0.1
    bind_retry_max_delay_seconds: float = 2.0
    key_listing_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """refuse timings the KV paths cannot run on, naming the field, where the host builds them.

        the bind's wait for its declarer runs on
        :func:`threetears.observe.resilience.retry_bounded`, which refuses a schedule it cannot back
        off on before its first attempt -- so a bad pause carried to a bind would refuse every bind,
        of a bucket that is right there. a zero bind wait is allowed: it binds once.

        :return: nothing
        :rtype: None
        :raises ValueError: when a deadline or pause is not positive, the bind wait is negative, or
            the longest pause is shorter than the first
        """
        for name in (
            "op_timeout_seconds",
            "timeout_remedy_log_interval_seconds",
            "bind_retry_first_delay_seconds",
            "key_listing_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"KvTimings.{name} must be positive, got {getattr(self, name)!r}")
        if self.bind_wait_for_declarer_seconds < 0:
            raise ValueError(
                f"KvTimings.bind_wait_for_declarer_seconds must not be negative, got {self.bind_wait_for_declarer_seconds!r}"
            )
        if self.bind_retry_max_delay_seconds < self.bind_retry_first_delay_seconds:
            raise ValueError(
                f"KvTimings.bind_retry_max_delay_seconds ({self.bind_retry_max_delay_seconds!r}) must not be shorter "
                f"than bind_retry_first_delay_seconds ({self.bind_retry_first_delay_seconds!r})"
            )


#: the production timings every client and bucket uses unless its host passes others.
DEFAULT_KV_TIMINGS: Final[KvTimings] = KvTimings()

#: Heartbeats a key watch's consumer may miss before the watch replaces it. One missed beat is
#: ordinary scheduling jitter; three is a consumer the server no longer has.
_KEY_WATCH_MISSED_HEARTBEATS: Final[int] = 3

#: The status an idle heartbeat carries. Any OTHER status on a key watch's inbox is the server
#: ending the consumer.
_STATUS_IDLE_HEARTBEAT: Final[str] = "100"

#: How long the server keeps a key watch's consumer once nothing is subscribed to its deliver subject.
#:
#: This, not a delete, is what removes a closed watch's consumer: a grant narrowed to one key
#: (``JsCapability.KV_KEY_READ``) carries ``CONSUMER.CREATE`` and nothing else, so a
#: ``CONSUMER.DELETE`` would be refused -- and a refused JetStream call does not raise, it blocks to
#: its deadline, which would turn every close into a ten-second stall. Long enough to ride out a
#: client reconnect without losing the consumer; short enough that an abandoned one is gone soon.
_KEY_WATCH_INACTIVE_THRESHOLD_SECONDS: Final[float] = 30.0

#: The header nats KV stamps on a delete or purge marker, and the values that mean one.
_KV_OPERATION_HEADER: Final[str] = "KV-Operation"
_KV_REMOVAL_OPERATIONS: Final[frozenset[str]] = frozenset({"DEL", "PURGE"})

#: Consumer names a key watch mints carry this prefix, so an operator listing a stream's consumers
#: can tell a watch from anything else.
_KEY_WATCH_CONSUMER_PREFIX: Final[str] = "kw_"

#: Consumer names a key listing mints carry this prefix, beside the key watch's ``kw_``.
_KEY_LISTING_CONSUMER_PREFIX: Final[str] = "kl_"

#: Characters that make a subject token a wildcard or split it. A watched key must be literal: the
#: grant names it literally, and a wildcard filter would be a different consumer from the one granted.
_KEY_WATCH_FORBIDDEN: Final[frozenset[str]] = frozenset({"*", ">", " ", "\t", "\r", "\n"})

#: Last time the remedy was logged, keyed by fully-qualified bucket name.
#:
#: Module-level because :class:`NatsKvBucket` declares ``__slots__`` and because the
#: throttle should hold across bucket handles for the same name -- a re-open mints a new
#: instance, and a reconnect loop must not reset the throttle on every attempt.
_last_timeout_remedy_log: dict[str, float] = {}


def _msg_ttl_seconds(ttl: timedelta | None) -> float | None:
    """convert a per-entry TTL to the whole seconds the ``Nats-TTL`` header carries.

    :param ttl: the requested lifetime, or ``None`` for none
    :ptype ttl: timedelta | None
    :return: whole seconds, or ``None``
    :rtype: float | None
    :raises ValueError: when ``ttl`` is under one second, which the header cannot express
    """
    if ttl is None:
        return None
    seconds = int(ttl.total_seconds())
    if seconds < 1:
        raise ValueError(f"a per-entry KV TTL must be at least one second, got {ttl}")
    return float(seconds)


#: Refusals an open raises that are deliberately NOT :class:`KvError`s, because the L2 accessors catch
#: ``KvError`` and degrade: a bind-only open finding config it refuses, and a create whose subjects
#: another stream owns. An operation whose self-heal re-binds or recreates its bucket can meet either,
#: and must raise it as itself rather than through :func:`_kv_error` -- wrapped, it would be caught
#: and downgraded to a per-operation warning, and the process would run on against the bucket.
_OPEN_REFUSALS: Final[tuple[type[NatsClientError], ...]] = (KvConfigMismatch, StreamSubjectsOverlapError)


def _kv_error(message: str, *, bucket: str, cause: BaseException) -> KvError:
    """the typed error a failed KV call surfaces as: an absent bucket gets its own type.

    Built here for every operation that runs through :meth:`NatsKvBucket._run_op` -- the reads, the
    writes and ``date_created`` -- and for a key listing's consumer create. Other paths build their
    own error, each naming a remedy only it knows: the opens (:func:`open_kv_stream`,
    :func:`_bind_when_declared`, :func:`_live_stream_config`) name the declarer or the grant, a key
    listing that ran out its bound names the consumer grant, and a key watch never raises a create
    failure at all. Those that classify an absence use the same predicate. ``cause`` is classified
    by type
    (:func:`~threetears.nats.raw_errors.is_bucket_not_found`), never by its message, and a
    ``KvBucketNotFoundError`` already raised further down -- a re-bind that found nothing -- keeps
    its type on the way out.

    :param message: the error's message, naming the bucket and the operation
    :ptype message: str
    :param bucket: fully-qualified bucket name
    :ptype bucket: str
    :param cause: the exception the KV call raised; the caller chains it with ``from``
    :ptype cause: BaseException
    :return: the error to raise
    :rtype: KvError
    """
    if is_bucket_not_found(cause):
        return KvBucketNotFoundError(message, bucket=bucket)
    return KvError(message)


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
    if field in ("allow_direct", "allow_msg_ttl"):
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
    timings: KvTimings = DEFAULT_KV_TIMINGS,
    owns_bucket: bool = False,
) -> KeyValue:
    """create, reconcile or bind the JetStream stream behind a KV bucket.

    the two modes are the declarer's and the reader's, and they differ on what
    they do about a bucket that already carries a different config:

    - ``create_if_missing=True`` DECLARES. the stream is created when absent;
      when it exists carrying a different value for one of
      :data:`RECONCILED_KV_STREAM_FIELDS`, it is updated in place. drift on any
      other requested field is bound to as-is and logged at WARNING -- the
      previous behaviour dropped it with nothing above DEBUG -- unless the
      declarer ``owns_bucket``, when the whole requested shape is reconciled
      (:func:`reconcile_kv_stream`).
    - ``create_if_missing=False`` BINDS. a reader has no authority to change a
      shared bucket, so drift on the reconciled set raises
      :class:`~threetears.nats.errors.KvConfigMismatch`, which the L2 accessors
      deliberately do not catch. an ABSENT bucket is waited for, with bounded
      backoff, since only its declarer can create it (:func:`_bind_when_declared`).

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
    :param timings: how long a bind waits for an absent bucket's declarer, and how it paces itself
    :ptype timings: KvTimings
    :param owns_bucket: this declarer is the ONE owner of the bucket's whole shape, so drift on
        any requested field is reconciled -- in place where JetStream allows it, by deleting and
        recreating the stream where it does not -- rather than reported and left. only a
        declaration of a memory-storage bucket may own it
    :ptype owns_bucket: bool
    :return: bound nats-py KeyValue handle
    :rtype: KeyValue
    :raises ValueError: ``owns_bucket`` on a bind-only open, which has no authority over the bucket,
        or on a bucket declared on file storage, whose contents are not ephemeral
    :raises StreamSubjectsOverlapError: a different stream already owns ``$KV.{bucket}.>``
    :raises KvConfigMismatch: bind-only open found drift on the reconciled set
    :raises KvError: creation or binding failed
    """
    _refuse_unownable_kv_bucket(
        full_name=full_name,
        owns_bucket=owns_bucket,
        create_if_missing=create_if_missing,
        storage="file" if config.storage == StorageType.FILE else "memory",
    )
    if not create_if_missing:
        return await _bind_kv_stream(js=js, full_name=full_name, config=config, timings=timings)

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
        await reconcile_kv_stream(js=js, full_name=full_name, config=config, owns_bucket=owns_bucket)
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
        if is_bucket_not_found(bind_exc):
            # the server ANSWERED the bind: the bucket is absent, and the create before it was never
            # answered -- the shape a principal refused STREAM.CREATE produces.
            raise KvBucketNotFoundError(
                f"open KV bucket failed: bucket={full_name} does not exist and this principal could not "
                f"create it: create={add_exc!r} bind={bind_exc!r}. {kv_grant_remedy(full_name)}",
                bucket=full_name,
            ) from bind_exc
        raise KvError(
            f"open KV bucket failed: bucket={full_name}: create={add_exc!r} bind={bind_exc!r}. "
            f"{kv_grant_remedy(full_name)}"
        ) from bind_exc
    return kv


async def _bind_when_declared(*, js: Any, full_name: str, timings: KvTimings) -> KeyValue:
    """bind a KV bucket, waiting with bounded backoff while it is absent.

    **Absent and refused are two different answers, and only one is worth waiting for.** A bucket
    that does not exist is ANSWERED -- the server replies not-found -- and for a process that only
    binds, that means its declarer has not declared it yet; the hub re-declares every pod bucket
    when it reconnects after a broker restart, so waiting is what recovers. A bucket this principal
    may not read is NEVER answered: the request dies on its deadline, and no amount of waiting grants
    it, so that failure is raised at once.

    :param js: connected JetStream context
    :ptype js: Any
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param timings: the wait for the declarer and the pace of the binds within it
    :ptype timings: KvTimings
    :return: bound nats-py KeyValue handle
    :rtype: KeyValue
    :raises KvBucketNotFoundError: the bucket stayed absent for
        :attr:`KvTimings.bind_wait_for_declarer_seconds`
    :raises KvError: the bind failed for any other reason
    """
    binds = 1

    def _absent_again(exc: Exception, attempt: int, pause: float) -> None:
        nonlocal binds
        del exc, pause
        binds = attempt + 1
        if attempt == 1:
            log.warning(
                "KV bucket %s does not exist yet; waiting up to %gs for its declarer to declare it",
                full_name,
                timings.bind_wait_for_declarer_seconds,
                extra={"extra_data": {"bucket": full_name}},
            )

    try:
        kv: KeyValue = await retry_bounded(
            lambda: js.key_value(full_name),
            retry_on=is_bucket_not_found,
            first_delay=timings.bind_retry_first_delay_seconds,
            max_delay=timings.bind_retry_max_delay_seconds,
            deadline_seconds=timings.bind_wait_for_declarer_seconds,
            on_retry=_absent_again,
        )
    except Exception as exc:
        if is_bucket_not_found(exc):
            raise KvBucketNotFoundError(
                f"bind KV bucket failed: bucket={full_name} does not exist, and its declarer did not "
                f"declare it within {timings.bind_wait_for_declarer_seconds:g}s ({binds} binds). this process "
                f"only binds it; the declaring identity (the hub, for every pod bucket) creates it at "
                f"startup and after every NATS reconnect -- check that it is running and connected.",
                bucket=full_name,
            ) from exc
        # Hedged: an unanswered bind is what a refused one looks like, and what an unreachable
        # broker looks like too.
        raise KvError(
            f"bind KV bucket failed: bucket={full_name}: {exc}. {kv_grant_remedy(full_name, certain=False)}"
        ) from exc
    if binds > 1:
        log.info(
            "KV bucket %s bound once its declarer declared it",
            full_name,
            extra={"extra_data": {"bucket": full_name, "binds": binds}},
        )
    return kv


async def _bind_kv_stream(*, js: Any, full_name: str, config: StreamConfig, timings: KvTimings) -> KeyValue:
    """bind to an existing KV bucket, refusing one whose reconciled config differs.

    A bucket that is ABSENT is waited for (:func:`_bind_when_declared`): a process that only binds
    never creates one, so it waits for the declarer rather than failing the first time it looks.

    :param js: connected JetStream context
    :ptype js: Any
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param config: the config this reader requires
    :ptype config: StreamConfig
    :param timings: the wait for an absent bucket's declarer
    :ptype timings: KvTimings
    :return: bound nats-py KeyValue handle
    :rtype: KeyValue
    :raises KvConfigMismatch: the live bucket differs on the reconciled field set
    :raises KvError: binding failed
    """
    kv = await _bind_when_declared(js=js, full_name=full_name, timings=timings)
    if any(getattr(config, field, None) is not None for field in _BIND_REFUSED_KV_STREAM_FIELDS):
        live = await _live_stream_config(js=js, full_name=full_name, stream=config.name)
        drift = {
            field: value
            for field, value in kv_stream_differences(requested=config, actual=live).items()
            if field in _BIND_REFUSED_KV_STREAM_FIELDS
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


async def _entry_ttl_for_bound_bucket(*, js: Any, full_name: str, ttl: timedelta) -> timedelta | None:
    """the per-entry lifetime a BIND-only opener must write with, so its entries still expire.

    A bucket's TTL is a property of its stream, set by whoever CREATES it. A pod never creates
    one: the hub declares every bucket a pod binds, uniformly, with no bucket-wide expiry and
    per-entry TTLs allowed -- because the hub cannot know which lifetime each primitive a pod runs
    over its bucket wants. The opener still knows. So a bind-only open that asks for ``ttl`` on a
    bucket with no bucket-wide expiry carries ``ttl`` onto every entry it writes instead, and its
    entries expire exactly as they would in a bucket created with that TTL.

    Anything else is refused rather than bound: a live bucket-wide expiry that differs from the
    request would expire this opener's entries early or late -- a quota count forgotten, a
    replay nonce remembered past its window, a credential outliving its lifetime -- with nothing
    to say so.

    :param js: connected JetStream context
    :ptype js: Any
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param ttl: the lifetime this opener wants every entry to have
    :ptype ttl: timedelta
    :return: ``None`` when the live bucket already expires entries at ``ttl``, else ``ttl`` itself
        as the default per-entry lifetime
    :rtype: timedelta | None
    :raises KvConfigMismatch: when the live bucket expires entries at another age, or has no
        bucket-wide expiry and refuses per-entry TTLs
    :raises KvError: when the live configuration cannot be read
    """
    live = await _live_stream_config(js=js, full_name=full_name, stream=f"{_KV_STREAM_PREFIX}{full_name}")
    requested = float(int(ttl.total_seconds()))
    live_age = float(_normalised("max_age", live.max_age))
    entry_ttl: timedelta | None = None
    if live_age == requested:
        log.debug(
            "KV bucket bound with the bucket-wide lifetime it asked for", extra={"extra_data": {"bucket": full_name}}
        )
    elif live_age != 0.0:
        raise KvConfigMismatch(
            f"KV bucket {full_name!r} expires entries after {live_age:g}s and this process opened it "
            f"read-only (create_if_missing=False) expecting {requested:g}s; its entries would expire at "
            f"the wrong age. the bucket's declarer (the hub) must create it with no bucket-wide expiry, "
            f"so each opener's entries carry their own lifetime; a bucket created earlier with one keeps "
            f"it until that declarer declares it with ensure_kv_bucket(ttl=None, owns_bucket=True), "
            f"which removes it in place."
        )
    elif not live.allow_msg_ttl:
        raise KvConfigMismatch(
            f"KV bucket {full_name!r} has no bucket-wide expiry and refuses per-entry TTLs "
            f"(allow_msg_ttl is off), so this read-only opener cannot give its entries the {requested:g}s "
            f"lifetime it needs. the declarer (the hub) must reconcile the bucket with allow_msg_ttl."
        )
    else:
        entry_ttl = ttl
        log.info(
            "KV bucket bound with a per-entry lifetime standing in for a bucket-wide one",
            extra={"extra_data": {"bucket": full_name, "entry_ttl_seconds": requested}},
        )
    return entry_ttl


async def reconcile_kv_stream(*, js: Any, full_name: str, config: StreamConfig, owns_bucket: bool = False) -> None:
    """bring a live KV stream to its declaration, or say loudly which request was dropped.

    the declaring half of :func:`open_kv_stream`, run when the create found the stream live with
    another configuration; also run by :class:`~threetears.nats.NatsClient`'s restoration after a
    reconnect for a bucket whose declarer owns it, so the restoration declares the same shape the
    declaration did.

    the previous opener bound to whatever existed and logged the fact at DEBUG,
    so a bucket carrying somebody else's config was indistinguishable from one
    carrying yours. everything outside :data:`RECONCILED_KV_STREAM_FIELDS` is
    still bound to as-is -- reconciling every field would let two declarers with
    different requests fight over one bucket -- but it is now reported at
    WARNING rather than dropped in silence.

    **a declarer that owns the bucket** (``owns_bucket=True``) has nobody to fight, so the whole
    requested shape is reconciled:

    - drift JetStream can update on a live stream (:data:`_OWNED_IN_PLACE_KV_STREAM_FIELDS`:
      ``max_age``, the history) is updated in place with the rest, logged at INFO naming the old and
      new values. the duplicate window comes with ``max_age``, because JetStream refuses a window
      longer than it.
    - drift JetStream refuses to update (:data:`_OWNED_RECREATE_KV_STREAM_FIELDS`: the storage) is
      reconciled by deleting the backing stream and creating it again with the declared config,
      logged at WARNING naming the bucket, the old and new values, and that its entries were
      dropped. that is safe only because an owner declares a MEMORY bucket, whose contents are
      ephemeral by design: a NATS restart already wipes them and every binder already survives an
      emptied bucket, so a recreate is that restart, for one bucket.

    **two owners may declare at once** (two replicas of the declaring service). a delete of a stream
    a concurrent owner already deleted is not an error; a create that finds the name taken re-reads
    the live config, and a bucket that now carries the declared storage is a concurrent owner's
    finished recreate -- it is never deleted a second time, and anything else it differs on is
    reconciled in place.

    :param js: connected JetStream context
    :ptype js: Any
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param config: the config this process asked for
    :ptype config: StreamConfig
    :param owns_bucket: this declarer is the one owner of the bucket's whole shape, so drift on
        every requested field is reconciled rather than reported and left
    :ptype owns_bucket: bool
    :return: nothing
    :rtype: None
    :raises KvError: the live config could not be read, or the update, delete or create was refused,
        or a concurrent process recreated the bucket with a storage other than the declared one
    :raises StreamSubjectsOverlapError: a recreate found the bucket's subjects owned by another stream
    """
    live = await _live_stream_config(js=js, full_name=full_name, stream=config.name)
    differences = kv_stream_differences(requested=config, actual=live)
    if owns_bucket and any(field in differences for field in _OWNED_RECREATE_KV_STREAM_FIELDS):
        live = await _recreate_owned_kv_stream(js=js, full_name=full_name, config=config, differences=differences)
        differences = kv_stream_differences(requested=config, actual=live)
    reconciled_fields = RECONCILED_KV_STREAM_FIELDS + (_OWNED_IN_PLACE_KV_STREAM_FIELDS if owns_bucket else ())
    await _reconcile_in_place(
        js=js,
        full_name=full_name,
        config=config,
        live=live,
        differences=differences,
        reconciled_fields=reconciled_fields,
    )


async def _recreate_owned_kv_stream(
    *, js: Any, full_name: str, config: StreamConfig, differences: dict[str, tuple[Any, Any]]
) -> StreamConfig:
    """delete an owned bucket's backing stream and create it again with the declared config.

    Only for drift JetStream will not update on a live stream (:data:`_OWNED_RECREATE_KV_STREAM_FIELDS`),
    and only for an owner, whose memory bucket's entries are ephemeral by design -- see
    :func:`reconcile_kv_stream`. Safe against a concurrent owner doing the same: a stream it already
    deleted is not an error, and a name it already took again is re-read rather than deleted again.

    :param js: connected JetStream context
    :ptype js: Any
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param config: the declared config
    :ptype config: StreamConfig
    :param differences: the live drift that called for the recreate, for the log
    :ptype differences: dict[str, tuple[Any, Any]]
    :return: the stream's config once it carries the declared storage: the declared one when this
        call created it, the live one when a concurrent owner did
    :rtype: StreamConfig
    :raises KvError: the delete or create was refused, or a concurrent process recreated the bucket
        with a storage other than the declared one
    :raises StreamSubjectsOverlapError: another stream owns the bucket's subjects
    """
    recreated = {field: value for field, value in differences.items() if field in _OWNED_RECREATE_KV_STREAM_FIELDS}
    try:
        await js.delete_stream(config.name)
    except Exception as exc:
        if not is_bucket_not_found(exc):
            raise _kv_error(
                f"recreating KV bucket {full_name!r} failed: deleting its stream was refused: {exc}. it is "
                f"live with {_render_differences(recreated)}, which JetStream cannot change in place. "
                f"{_refusal_remedy(exc, full_name=full_name)}",
                bucket=full_name,
                cause=exc,
            ) from exc
        # answered "not found": a concurrent owner deleted it first, and is recreating it.
        log.debug(
            "JetStream KV bucket's stream was already deleted by a concurrent declarer",
            extra={"extra_data": {"bucket": full_name}},
        )
    result = config
    try:
        await js.add_stream(config)
    except Exception as exc:
        _raise_if_subjects_overlap(exc, full_name=full_name, config=config)
        if getattr(exc, "err_code", None) != _JS_ERR_STREAM_NAME_IN_USE:
            raise _kv_error(
                f"recreating KV bucket {full_name!r} failed: its stream was deleted and creating it again "
                f"was refused: {exc}. {_refusal_remedy(exc, full_name=full_name)}",
                bucket=full_name,
                cause=exc,
            ) from exc
        # a concurrent owner created it between this delete and this create. its recreate is the
        # one that stands: re-read it, and never delete a freshly correct bucket a second time.
        result = await _live_stream_config(js=js, full_name=full_name, stream=config.name)
        still = {
            field: value
            for field, value in kv_stream_differences(requested=config, actual=result).items()
            if field in _OWNED_RECREATE_KV_STREAM_FIELDS
        }
        if still:
            raise KvError(
                f"recreating KV bucket {full_name!r} failed: between this declarer's delete and its create, "
                f"another process created the bucket again with {_render_differences(still)}. this "
                f"declaration does not delete it again, which two disagreeing declarers would do forever; "
                f"find the process that declares {full_name!r} with another shape -- this bucket should "
                f"have one declarer -- and the next declaration will reconcile it."
            )
        log.info(
            "JetStream KV bucket was recreated with its declared configuration by a concurrent declarer",
            extra={"extra_data": {"bucket": full_name, "recreated": _render_differences(recreated)}},
        )
        return result
    log.warning(
        "JetStream KV bucket %s recreated with its declared configuration (%s); JetStream cannot change "
        "that on a live stream, so its stream was deleted and created again and every entry it held was "
        "dropped",
        full_name,
        _render_differences(recreated),
        extra={
            "extra_data": {"bucket": full_name, "recreated": _render_differences(recreated), "entries_dropped": True}
        },
    )
    return result


async def _reconcile_in_place(
    *,
    js: Any,
    full_name: str,
    config: StreamConfig,
    live: StreamConfig,
    differences: dict[str, tuple[Any, Any]],
    reconciled_fields: tuple[str, ...],
) -> None:
    """update the reconciled fields of a live KV stream in place, reporting every other difference.

    :param js: connected JetStream context
    :ptype js: Any
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param config: the declared config
    :ptype config: StreamConfig
    :param live: the stream's live config
    :ptype live: StreamConfig
    :param differences: requested-versus-live drift, from :func:`kv_stream_differences`
    :ptype differences: dict[str, tuple[Any, Any]]
    :param reconciled_fields: the fields this declaration updates in place
    :ptype reconciled_fields: tuple[str, ...]
    :return: nothing
    :rtype: None
    :raises KvError: the update was refused
    """
    reconcilable = {field: value for field, value in differences.items() if field in reconciled_fields}
    dropped = {field: value for field, value in differences.items() if field not in reconciled_fields}
    if dropped:
        remedy = ""
        if any(field in dropped for field in _OWNED_IN_PLACE_KV_STREAM_FIELDS + _OWNED_RECREATE_KV_STREAM_FIELDS):
            remedy = (
                " -- the bucket's expiry, history and storage are reconciled only by a declarer that owns "
                "the bucket; when this declarer is its only one, declare it with "
                "ensure_kv_bucket(owns_bucket=True)"
            )
        log.warning(
            "JetStream KV bucket bound to an existing stream whose configuration differs from the "
            "requested one; the requested values were NOT applied: bucket=%s %s%s",
            full_name,
            _render_differences(dropped),
            remedy,
            extra={"extra_data": {"bucket": full_name, "dropped": _render_differences(dropped)}},
        )
    if reconcilable:
        # Built from the LIVE config with only the reconciled fields changed. Sending the
        # requested config would also ask for every other difference, and some of those the
        # server refuses to change at all: a legacy file-backed bucket opened by a declarer that
        # asks for memory would fail the whole update over storage, and with it the in-place
        # enable this update exists for.
        changes: dict[str, Any] = {field: want for field, (want, _have) in reconcilable.items()}
        if "max_age" in changes:
            # the window the declared shape carries (build_kv_stream_config): the server refuses a
            # duplicate window longer than a non-zero max_age, so the old window cannot stay.
            changes["duplicate_window"] = config.duplicate_window
        update = dataclasses.replace(live, **changes)
        try:
            await js.update_stream(update)
        except Exception as exc:
            raise _kv_error(
                f"reconciling KV bucket {full_name!r} failed: {exc}. it is live with "
                f"{_render_differences(reconcilable)} and this principal could not update it. "
                f"{_refusal_remedy(exc, full_name=full_name)}",
                bucket=full_name,
                cause=exc,
            ) from exc
        log.info(
            "JetStream KV bucket reconciled in place",
            extra={"extra_data": {"bucket": full_name, "applied": _render_differences(reconcilable)}},
        )
    elif not dropped:
        log.debug(
            "JetStream KV bucket bound (already existed)",
            extra={"extra_data": {"bucket": full_name}},
        )


def _refusal_remedy(exc: BaseException, *, full_name: str) -> str:
    """what to do about a stream-management call that failed, read from whether the server answered.

    A principal may be granted ``STREAM.CREATE`` and refused ``STREAM.UPDATE`` or ``STREAM.DELETE``
    -- exactly the shape a pod's grant has -- and an ungranted request is never answered: it arrives
    as a deadline, so the remedy is the grant. A refusal the server ANSWERED, with its own error
    code, is never that: it names a configuration the server will not accept, and the grant remedy
    would send an operator the wrong way.

    :param exc: what the call raised
    :ptype exc: BaseException
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :return: the remedy sentence for the error message
    :rtype: str
    """
    if getattr(exc, "err_code", None) is not None:
        return "the server refused it itself, so the declared configuration is one it will not apply"
    return kv_grant_remedy(full_name)


def _refuse_unownable_kv_bucket(*, full_name: str, owns_bucket: bool, create_if_missing: bool, storage: str) -> None:
    """refuse an ownership of a bucket that no open may claim.

    Only a declaration may own a bucket: a bind-only open has no authority over it. And only a
    MEMORY bucket may be owned, because an owner reconciles storage drift by deleting the stream and
    dropping every entry -- which is the restart a memory bucket already survives, and data loss for
    a file bucket that was declared durable on purpose.

    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param owns_bucket: whether the open claims ownership
    :ptype owns_bucket: bool
    :param create_if_missing: whether the open declares
    :ptype create_if_missing: bool
    :param storage: the declared storage, ``"memory"`` or ``"file"``
    :ptype storage: str
    :return: nothing
    :rtype: None
    :raises ValueError: when the open claims an ownership it may not have
    """
    if not owns_bucket:
        return
    if not create_if_missing:
        raise ValueError(
            f"KV bucket {full_name!r}: owns_bucket=True needs create_if_missing=True -- only the bucket's "
            f"declarer may own it, and a bind-only open declares nothing"
        )
    if storage != "memory":
        raise ValueError(
            f"KV bucket {full_name!r}: owns_bucket=True needs storage='memory' -- an owner reconciles "
            f"storage drift by deleting the stream and its entries, which is safe only for a bucket whose "
            f"contents are ephemeral by design"
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
    :raises KvBucketNotFoundError: the server answered that the stream does not exist
    :raises KvError: the lookup failed for any other reason
    """
    try:
        info = await js.stream_info(stream)
    except Exception as exc:
        if is_bucket_not_found(exc):
            # answered: the stream is gone, which no grant would fix.
            raise KvBucketNotFoundError(
                f"reading the live configuration of KV bucket {full_name!r} failed: it does not exist ({exc})",
                bucket=full_name,
            ) from exc
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


class _KvHandleBinding(NamedTuple):
    """what one open of a bucket produces, before it is put on a :class:`NatsKvBucket`.

    :ivar kv: the nats-py handle
    :ivar entry_ttl: the per-entry TTL a bind-only open stamps on its writes, or ``None``
    :ivar bound_to: the connection the handle was bound through
    """

    kv: KeyValue
    entry_ttl: timedelta | None
    bound_to: Any


async def _bind_kv_handle(
    *,
    client: NatsClient,
    full_name: str,
    ttl: timedelta | None,
    storage: str,
    create_if_missing: bool,
    history: int,
    direct: bool | None,
    timings: KvTimings,
    owns_bucket: bool = False,
) -> _KvHandleBinding:
    """open, create or reconcile a bucket and return the handle it yields.

    The one opener behind :meth:`NatsKvBucket.open`, which wraps the result in a new bucket, and
    :meth:`NatsKvBucket._reopen`, which refreshes an existing bucket in place with it.

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
    :param direct: request ``allow_direct`` on the backing stream; ``None`` neither requests nor compares it
    :ptype direct: bool | None
    :param timings: the bucket's deadlines, of which a bind uses the wait for its declarer
    :ptype timings: KvTimings
    :param owns_bucket: the declarer owns the bucket's whole shape and reconciles every requested
        field (:func:`reconcile_kv_stream`); only with ``create_if_missing`` on memory storage
    :ptype owns_bucket: bool
    :return: the handle, its per-entry TTL, and the connection it was bound through
    :rtype: _KvHandleBinding
    :raises ValueError: if ``owns_bucket`` is asked of a bind-only open or a file bucket
    :raises KvError: if bucket creation or binding fails
    :raises KvConfigMismatch: if a bind-only open finds a reconciled field differing
    :raises StreamSubjectsOverlapError: if a different stream owns the bucket's subjects
    """
    # the connection is read with the context, with no await between, so the handle records
    # the connection it was actually bound on even when a renewal lands while it opens.
    bound_to = client.raw
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
        timings=timings,
        owns_bucket=owns_bucket,
    )
    entry_ttl: timedelta | None = None
    if not create_if_missing and ttl is not None and ttl_seconds > 0:
        entry_ttl = await _entry_ttl_for_bound_bucket(js=js, full_name=full_name, ttl=ttl)
    return _KvHandleBinding(kv=kv, entry_ttl=entry_ttl, bound_to=bound_to)


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
    :param bound_to: the client's nats-py connection ``kv`` was bound through, which the handle
        follows across a credential renewal; ``None`` for a handle the client does not move
    :ptype bound_to: Any
    :param timings: the deadlines its operations and re-binds run under
    :ptype timings: KvTimings
    :param owns_bucket: the declaration this handle was opened by owns the bucket's whole shape,
        which a self-heal re-open declares again
    :ptype owns_bucket: bool
    """

    __slots__ = (
        "_bound_to",
        "_client",
        "_create_if_missing",
        "_direct",
        "_entry_ttl",
        "_full_name",
        "_history",
        "_kv",
        "_owns_bucket",
        "_storage",
        "_timings",
        "_ttl",
    )

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
        entry_ttl: timedelta | None = None,
        bound_to: Any = None,
        timings: KvTimings = DEFAULT_KV_TIMINGS,
        owns_bucket: bool = False,
    ) -> None:
        self._client = client
        self._timings = timings
        self._full_name = full_name
        self._kv = kv
        # the nats-py connection ``kv`` issues its operations on. a credential renewal replaces the
        # client's connection and later retires this one, so every operation first checks it is
        # still the current one (:meth:`_follow_connection`). :meth:`open` -- the only way the
        # client builds a bucket -- always records it; ``None`` is a handle built around a caller's
        # own KeyValue, which is not the client's to move.
        self._bound_to = bound_to
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
        # Retained for the same self-heal: a re-open that forgot it would bind a bucket another
        # process put back first with its own shape -- an expiry, a storage -- and every bind-only
        # opener asking for a per-entry lifetime would be refused until this process restarted.
        self._owns_bucket = owns_bucket
        # the lifetime every write carries when its caller names none: a bind-only open's stand-in
        # for a bucket-wide TTL the declarer did not set (``_entry_ttl_for_bound_bucket``).
        self._entry_ttl = entry_ttl

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
        timings: KvTimings = DEFAULT_KV_TIMINGS,
        owns_bucket: bool = False,
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
        :param timings: the deadlines the bucket's operations and binds run under
        :ptype timings: KvTimings
        :param owns_bucket: the declarer owns the bucket's whole shape and reconciles every
            requested field (:func:`reconcile_kv_stream`), here and on every self-heal re-open. only
            :meth:`NatsClient.ensure_kv_bucket` passes it
        :ptype owns_bucket: bool
        :return: ready bucket
        :rtype: NatsKvBucket
        :raises ValueError: if ``owns_bucket`` is asked of a bind-only open or a file bucket
        :raises KvBucketNotFoundError: if the bucket does not exist -- a bind-only open once its wait for
            the declarer is spent, or a declaring open whose create was not answered
        :raises KvError: if bucket creation or binding fails for any other reason
        :raises KvConfigMismatch: if a bind-only open finds a reconciled field differing, or a
            bucket-wide expiry that would expire its entries at the wrong age
        :raises StreamSubjectsOverlapError: if a different stream owns the bucket's subjects
        """
        binding = await _bind_kv_handle(
            client=client,
            full_name=full_name,
            ttl=ttl,
            storage=storage,
            create_if_missing=create_if_missing,
            history=history,
            direct=direct,
            timings=timings,
            owns_bucket=owns_bucket,
        )
        return cls(
            client=client,
            full_name=full_name,
            kv=binding.kv,
            ttl=ttl,
            storage=storage,
            create_if_missing=create_if_missing,
            history=history,
            direct=direct,
            entry_ttl=binding.entry_ttl,
            bound_to=binding.bound_to,
            timings=timings,
            owns_bucket=owns_bucket,
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
        rather than with the field unset. so does ``owns_bucket``: a bucket another
        process put back first, with an expiry or storage of its own, is reconciled
        to the declared shape rather than bound as-is.
        """
        binding = await _bind_kv_handle(
            client=self._client,
            full_name=self._full_name,
            ttl=self._ttl,
            storage=self._storage,
            create_if_missing=self._create_if_missing,
            history=self._history,
            direct=self._direct,
            timings=self._timings,
            owns_bucket=self._owns_bucket,
        )
        self._kv = binding.kv
        self._entry_ttl = binding.entry_ttl
        self._bound_to = binding.bound_to

    async def _follow_connection(self) -> None:
        """rebind the handle when a credential renewal has replaced the connection it was bound on.

        The nats-py handle issues every operation on the connection it was bound through, and a
        renewal retires that connection once the work it carries has finished. Rebinding before
        the next operation moves the bucket without an operation ever failing on the retired
        connection. A bind is one ``STREAM.INFO``, which every principal that opened the bucket
        holds, and it changes nothing about the bucket -- unlike :meth:`_reopen`, which declares.

        :return: nothing
        :rtype: None
        :raises Exception: when the bind fails; :meth:`_run_with_reopen` treats it as the transport
            failure it is
        """
        if self._bound_to is None:
            return
        current = self._client.raw
        if current is self._bound_to:
            return
        js = self._client.jetstream_context()
        self._kv = await self._bounded(lambda: js.key_value(self._full_name))
        self._bound_to = current

    async def _run_with_reopen(self, op: Any, *, passthrough: tuple[type[BaseException], ...]) -> Any:
        """Run a KV op; on a TRANSPORT failure, re-open the bucket once and retry.

        ``passthrough`` exceptions (KeyNotFound / CAS-mismatch) are normal control flow and
        are re-raised immediately -- only an unexpected failure (a vanished stream, a
        transient transport error) triggers the single re-open + retry. A second failure
        propagates to the caller's ``KvError`` wrap.
        """
        try:
            await self._follow_connection()
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

    async def _run_op(self, op: Any, *, passthrough: tuple[type[BaseException], ...], failure: str) -> Any:
        """Run a KV op through :meth:`_run_with_reopen` and translate what it raises: the ONE owner.

        Every operation's failure leaves through here, so a rule about what a caller sees is applied
        once rather than in each operation's own ``except`` chain -- the refusal arm this module once
        had to add to nine methods, one at a time, is that rule:

        - ``passthrough`` exceptions are the operation's own documented outcomes (a missing key, a
          lost compare-and-swap), raised unchanged for the operation to turn into its result;
        - an open's refusals (:data:`_OPEN_REFUSALS`) are raised as themselves, never as a
          ``KvError`` the L2 accessors would catch and degrade;
        - anything else becomes the typed error :func:`_kv_error` builds, chained to its cause.

        :param op: builds the KV coroutine to run
        :ptype op: Any
        :param passthrough: exceptions the operation handles itself
        :ptype passthrough: tuple[type[BaseException], ...]
        :param failure: what failed, naming the operation and the bucket; the cause is appended
        :ptype failure: str
        :return: whatever the operation returned
        :rtype: Any
        :raises KvError: on any failure that is neither passed through nor an open's refusal
        """
        try:
            return await self._run_with_reopen(op, passthrough=passthrough)
        except passthrough:
            raise
        except _OPEN_REFUSALS:
            # NOSILENT: re-raised as itself -- a refusal the re-bind found is not a KvError
            raise
        except Exception as exc:
            raise _kv_error(f"{failure}: {exc}", bucket=self._full_name, cause=exc) from exc

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
        if last is not None and now - last < self._timings.timeout_remedy_log_interval_seconds:
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
        return await run_bounded(
            op, timeout=self._timings.op_timeout_seconds, what=f"kv operation on {self._full_name}"
        )

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
            entry = await self._run_op(
                lambda: self._kv.get(key),
                passthrough=(KeyNotFoundError,),
                failure=f"KV get failed: bucket={self._full_name} key={key}",
            )
        except KeyNotFoundError:
            # NOSILENT: a miss is this method's documented result, reported to the caller as None
            return None
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
            entry = await self._run_op(
                lambda: self._kv.get(key),
                passthrough=(KeyNotFoundError,),
                failure=f"KV get_entry failed: bucket={self._full_name} key={key}",
            )
        except KeyNotFoundError:
            # NOSILENT: a miss is this method's documented result, reported to the caller as None
            return None
        if entry.value is None or entry.revision is None:
            return None
        return (bytes(entry.value), int(entry.revision))

    async def get_latest(self, *, key: str) -> tuple[bytes | None, int]:
        """the key's latest message: its value, and its revision even when that message is a deletion.

        :meth:`get_entry` answers "is there a live value" and reports a deleted key as absent,
        dropping the revision of the deletion marker. That revision is what a writer needs to
        write ONLY IF NOTHING HAS HAPPENED to the key since it looked: an :meth:`update` at the
        revision this returns lands only while the key's history is unchanged, where
        :meth:`create` would also land over a deletion made in between. A read that seeds a value
        from another tier depends on exactly that difference.

        :param key: key to read
        :ptype key: str
        :return: ``(value, revision)`` for a live value; ``(None, revision)`` for a key whose latest
            message is a delete or purge marker; ``(None, 0)`` for a key with no message at all.
            An :meth:`update` at the returned revision lands only if no message has been written
            to the key since
        :rtype: tuple[bytes | None, int]
        :raises KvError: on transport failure
        """
        try:
            entry = await self._run_op(
                lambda: self._kv.get(key),
                passthrough=(KeyNotFoundError,),
                failure=f"KV get_latest failed: bucket={self._full_name} key={key}",
            )
        except KeyNotFoundError as exc:
            # nats-py raises this for a missing key AND for a deleted one; only the second carries
            # the marker entry, whose revision is the key's latest.
            marker = getattr(exc, "entry", None)
            marker_revision = getattr(marker, "revision", None)
            return (None, int(marker_revision) if marker_revision else 0)
        revision = int(entry.revision) if entry.revision is not None else 0
        return (bytes(entry.value) if entry.value is not None else None, revision)

    async def put(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int:
        """unconditional write. returns new revision.

        :param key: key to write
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :param ttl: a server-side lifetime for THIS entry, after which the server removes it;
            ``None`` keeps the bucket's own expiry -- or, for a bucket bound read-only whose
            declarer set none, the lifetime the opener asked for. Whole seconds, at least one.
            Needs the stream's ``allow_msg_ttl``: a stream without it refuses the write, which raises
        :ptype ttl: timedelta | None
        :return: new revision number
        :rtype: int
        :raises KvError: on transport failure, or when the stream does not allow per-entry TTLs
        :raises ValueError: when ``ttl`` is under one second
        """
        msg_ttl = _msg_ttl_seconds(ttl if ttl is not None else self._entry_ttl)
        if msg_ttl is not None:
            return await self._put_with_ttl(key=key, value=value, msg_ttl=msg_ttl)
        revision = await self._run_op(
            lambda: self._kv.put(key, value),
            passthrough=(),
            failure=f"KV put failed: bucket={self._full_name} key={key}",
        )
        return int(revision)

    async def _put_with_ttl(self, *, key: str, value: bytes, msg_ttl: float) -> int:
        """unconditional write carrying a per-entry TTL.

        nats-py's public ``KeyValue.put`` takes no TTL, so this sends what it would -- a publish
        to the key's subject -- with the ``Nats-TTL`` header added. The sibling of
        :meth:`_update_with_ttl`, without the expected-sequence header, because an unconditional
        write fences on nothing.

        :param key: key to write
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :param msg_ttl: server-side lifetime in whole seconds
        :ptype msg_ttl: float
        :return: new revision number
        :rtype: int
        :raises KvError: on transport failure or any refusal
        """
        js = self._client.jetstream_context()
        subject = f"$KV.{self._full_name}.{key}"
        ack = await self._run_op(
            lambda: js.publish(subject, value, msg_ttl=msg_ttl),
            passthrough=(),
            failure=f"KV put failed: bucket={self._full_name} key={key}",
        )
        return int(ack.seq)

    async def create(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int | None:
        """create-if-absent (SET NX). returns new revision or ``None`` on conflict.

        :param key: key to create
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :param ttl: a server-side lifetime for THIS entry, after which the server removes it;
            ``None`` keeps the bucket's own expiry. Whole seconds, at least one. Needs the stream's
            ``allow_msg_ttl``: a stream without it refuses the write, which raises
        :ptype ttl: timedelta | None
        :return: new revision number, or ``None`` if key already exists
        :rtype: int | None
        :raises KvError: on transport failure, or when the stream does not allow per-entry TTLs
        :raises ValueError: when ``ttl`` is under one second
        """
        msg_ttl = _msg_ttl_seconds(ttl if ttl is not None else self._entry_ttl)

        def _do_create() -> Any:
            # the keyword is sent only when a lifetime was asked for, so the call an untimed
            # create makes is exactly the call it has always made.
            if msg_ttl is None:
                return self._kv.create(key, value)
            return self._kv.create(key, value, msg_ttl=msg_ttl)

        try:
            revision = await self._run_op(
                _do_create,
                passthrough=(KeyWrongLastSequenceError,),
                failure=f"KV create failed: bucket={self._full_name} key={key}",
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
        return int(revision)

    async def update(self, *, key: str, value: bytes, revision: int, ttl: timedelta | None = None) -> int | None:
        """compare-and-swap update. returns new revision or ``None`` on revision-mismatch.

        :param key: key to update
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :param revision: expected current revision -- the key's latest message, a deletion marker
            included (:meth:`get_latest`); ``0`` expects the key to have no message at all
        :ptype revision: int
        :param ttl: a server-side lifetime for the new entry; ``None`` keeps the bucket's own
            expiry. Whole seconds, at least one. Needs the stream's ``allow_msg_ttl``
        :ptype ttl: timedelta | None
        :return: new revision number, or ``None`` if expected revision did not match
        :rtype: int | None
        :raises KvError: on transport failure, or when the stream does not allow per-entry TTLs
        :raises ValueError: when ``ttl`` is under one second
        """
        msg_ttl = _msg_ttl_seconds(ttl if ttl is not None else self._entry_ttl)
        if msg_ttl is not None:
            return await self._update_with_ttl(key=key, value=value, revision=revision, msg_ttl=msg_ttl)
        try:
            new_revision = await self._run_op(
                lambda: self._kv.update(key, value, revision),
                passthrough=(KeyWrongLastSequenceError,),
                failure=f"KV update failed: bucket={self._full_name} key={key} rev={revision}",
            )
        except KeyWrongLastSequenceError:
            # A lost CAS is a documented result (None), but a burst of them is contention on one
            # key, and a caller that never retries would otherwise drop the write in silence.
            log.debug(
                "KV update lost: revision mismatch",
                extra={"extra_data": {"bucket": self._full_name, "key": key, "expected_revision": revision}},
            )
            return None
        return int(new_revision)

    async def _update_with_ttl(self, *, key: str, value: bytes, revision: int, msg_ttl: float) -> int | None:
        """compare-and-swap update carrying a per-entry TTL.

        nats-py's public ``KeyValue.update`` takes no TTL, so this sends what it would -- a
        publish to the key's subject expecting ``revision`` as the subject's last sequence -- with
        the ``Nats-TTL`` header added. The server's two wrong-last-sequence codes are the lost
        compare-and-swap; anything else it refuses is a failure.

        **This path deliberately does not self-heal a vanished stream, and :meth:`update` does.**
        ``update`` reopens and retries on any exception it did not pass through, so a stream wiped
        and recreated under it heals silently. Here the whole of ``APIError`` is passed through --
        ``_run_with_reopen`` selects by exception TYPE, and the lost compare-and-swap is an
        ``APIError`` distinguished only by its ``err_code``, so passing the code's class through
        is the only way to reach the branch below. ``stream not found`` rides the same class and
        therefore surfaces as a ``KvError`` rather than reopening.

        Accepted rather than overlooked. The window is the gap after the ``get_entry`` that
        produced ``revision`` succeeded, and a compare-and-swap against a recreated stream has
        lost by definition -- so the caller retrying is the correct outcome either way. Closing it
        would mean giving ``_run_with_reopen`` a predicate instead of a type tuple, which is a
        change to the machinery every KV operation runs through.

        :param key: key to update
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :param revision: expected current revision
        :ptype revision: int
        :param msg_ttl: server-side lifetime in whole seconds
        :ptype msg_ttl: float
        :return: new revision number, or ``None`` if expected revision did not match
        :rtype: int | None
        :raises KvError: on transport failure or any other refusal
        """
        js = self._client.jetstream_context()
        subject = f"$KV.{self._full_name}.{key}"
        headers = {Header.EXPECTED_LAST_SUBJECT_SEQUENCE: str(revision)}
        failure = f"KV update failed: bucket={self._full_name} key={key} rev={revision}"
        try:
            ack = await self._run_op(
                lambda: js.publish(subject, value, headers=headers, msg_ttl=msg_ttl),
                passthrough=(APIError,),
                failure=failure,
            )
        except APIError as exc:
            if exc.err_code in _JS_ERR_WRONG_LAST_SEQUENCE:
                log.debug(
                    "KV update lost: revision mismatch",
                    extra={"extra_data": {"bucket": self._full_name, "key": key, "expected_revision": revision}},
                )
                return None
            raise _kv_error(f"{failure}: {exc}", bucket=self._full_name, cause=exc) from exc
        return int(ack.seq)

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
            await self._run_op(
                _do_delete,
                passthrough=(KeyNotFoundError, KeyWrongLastSequenceError),
                failure=f"KV delete failed: bucket={self._full_name} key={key} revision={revision}",
            )
        except KeyNotFoundError:
            return True
        except KeyWrongLastSequenceError:
            return False
        return True

    async def date_created(self) -> datetime:
        """when the bucket's backing stream was created, read fresh from the server.

        Never cached, deliberately. A handle carries only names, so after another pod
        recreates a wiped stream every operation on this handle keeps succeeding against
        the new, empty stream without raising -- a creation time remembered at open would
        go stale with nothing to say so. Asking the server each time is what lets a caller
        detect that the bucket it wrote to is younger than something it trusts.

        A vanished stream takes the same self-heal as every other operation, so the answer
        describes the stream the next write will land in.

        :return: timezone-aware UTC creation time of the backing stream
        :rtype: datetime
        :raises KvError: on transport failure, or when the server reports no creation time
        """
        js = self._client.jetstream_context()
        stream = f"KV_{self._full_name}"
        info: StreamInfo = await self._run_op(
            lambda: js.stream_info(stream), passthrough=(), failure=f"KV stream info failed: bucket={self._full_name}"
        )
        if info.created is None:
            raise KvError(f"KV stream info carries no creation time: bucket={self._full_name}")
        return info.created

    async def watch_key(
        self,
        *,
        key: str,
        heartbeat: timedelta = DEFAULT_KEY_WATCH_HEARTBEAT,
        retry: timedelta = DEFAULT_KEY_WATCH_RETRY,
    ) -> AsyncGenerator[KvKeyUpdate]:
        """the key's latest message, then every later one, until the caller stops iterating.

        Watches through a push consumer created by NAME with its filter in the create subject
        (``$JS.API.CONSUMER.CREATE.{stream}.{name}.$KV.{bucket}.{key}``) -- the one consumer shape a
        grant narrowed to a single key can admit. nats-py's ``KeyValue.watch`` creates an unnamed
        consumer, which such a grant refuses by blocking to its deadline.

        The consumer delivers the key's last message first and acknowledges nothing. Every consumer
        gets a FRESH name: a name still held by a consumer the server has not yet reaped would
        refuse the create. It sends a heartbeat while the key is quiet, and when three go missing --
        a broker restart emptied the stream, or the consumer was reaped during a long disconnect --
        or the server says it ended the consumer, the watch replaces it. The replacement redelivers the key's latest
        message, so nothing written in between is missed; a redelivery of the message this watch
        last yielded is not yielded again.

        A consumer create that fails is logged, naming the grant to check, and retried after
        ``retry``; it is never raised, because the watch's whole job is to outlast the conditions
        that make a create fail. A closed consumer's server-side state is left to the server to
        reap -- a key-scoped grant carries no ``CONSUMER.DELETE``.

        Close it by stopping iteration -- ``aclose()``, or leaving an ``aclosing`` block -- which
        drops the subscription behind the current consumer.

        :param key: the key to watch; one literal key, never a wildcard
        :ptype key: str
        :param heartbeat: how often a quiet consumer proves it is alive
        :ptype heartbeat: timedelta
        :param retry: the pause after a consumer create failed
        :ptype retry: timedelta
        :return: the key's messages, in order; a delete or purge arrives with ``value=None``
        :rtype: AsyncGenerator[KvKeyUpdate]
        :raises ValueError: when ``key`` is empty or not a literal subject token sequence
        :raises KvError: when the NATS connection is closed, so no consumer can ever deliver again
        """
        if not key or any(char in _KEY_WATCH_FORBIDDEN for char in key):
            raise ValueError(f"watch_key needs one literal key, got {key!r}")
        subject = f"$KV.{self._full_name}.{key}"
        stream = f"{_KV_STREAM_PREFIX}{self._full_name}"
        last: KvKeyUpdate | None = None
        while True:
            consumer = await _KeyWatchConsumer.open(
                client=self._client,
                stream=stream,
                subject=subject,
                heartbeat=heartbeat,
                retry=retry,
                op_timeout_seconds=self._timings.op_timeout_seconds,
            )
            if consumer is None:
                await asyncio.sleep(retry.total_seconds())
                continue
            try:
                async with aclosing(consumer.updates(key=key)) as updates:
                    async for update in updates:
                        if update == last:
                            continue
                        last = update
                        yield update
            finally:
                await consumer.close()

    async def list_keys(self, *, prefix: str = "") -> list[str]:
        """every live key in the bucket that starts with ``prefix``.

        Lists through a push consumer created by NAME with its filter in the create subject
        (``$JS.API.CONSUMER.CREATE.{stream}.{name}.$KV.{bucket}.{prefix}>``), the one consumer shape
        a pod's grant on a bucket admits. nats-py's ``KeyValue.keys`` and ``watchall`` create an
        UNNAMED consumer, whose filter rides only in the request body, and a pod is never granted
        that -- the call blocks to its deadline instead of raising.

        A prefix that ends on a token boundary (``""``, or ending in ``.``) narrows the filter
        itself, so the server delivers only matching keys; any other prefix filters the whole
        bucket and keeps the matching keys here. The consumer delivers each key's latest message,
        headers only, and a key whose latest message is a delete or purge is not listed. It
        acknowledges nothing, and the server reaps it after its inactivity threshold -- a pod's
        grant carries no ``CONSUMER.DELETE``.

        :param prefix: keep keys starting with this; ``""`` lists every key
        :ptype prefix: str
        :return: the live keys, in stream order
        :rtype: list[str]
        :raises ValueError: when ``prefix`` carries a wildcard or whitespace
        :raises KvBucketNotFoundError: when the bucket's stream is gone and could not be bound again --
            a declaring handle could not recreate it, or a bind-only handle's declarer did not return
            within ``KvTimings.bind_wait_for_declarer_seconds``
        :raises KvError: when the consumer cannot be created or the listing does not finish within
            its bound -- an ungranted create is never answered, so it arrives here as a timeout
        """
        if any(char in _KEY_WATCH_FORBIDDEN for char in prefix):
            raise ValueError(f"list_keys needs a literal prefix, got {prefix!r}")
        subject_prefix = f"$KV.{self._full_name}."
        narrowed = prefix == "" or prefix.endswith(".")
        filter_subject = f"{subject_prefix}{prefix}>" if narrowed else f"{subject_prefix}>"
        stream = f"{_KV_STREAM_PREFIX}{self._full_name}"
        try:
            found = await self._list_keys_within_bound(stream=stream, filter_subject=filter_subject)
        except KvBucketNotFoundError:
            # the stream is gone: re-open once, as every other operation does -- a declaring handle
            # recreates its bucket, a bind-only one waits for its declarer -- then list again. the
            # re-open runs OUTSIDE the listing's bound, on its own wait for the declarer. a second
            # absence is raised.
            await self._reopen()
            found = await self._list_keys_within_bound(stream=stream, filter_subject=filter_subject)
        return [key[len(subject_prefix) :] for key in found if key[len(subject_prefix) :].startswith(prefix)]

    async def _list_keys_within_bound(self, *, stream: str, filter_subject: str) -> list[str]:
        """one listing attempt, bounded by :attr:`KvTimings.key_listing_timeout_seconds`.

        :param stream: the bucket's backing stream
        :ptype stream: str
        :param filter_subject: the subject filter, inside this bucket
        :ptype filter_subject: str
        :return: the subjects of the live keys, in stream order
        :rtype: list[str]
        :raises KvBucketNotFoundError: the server answered that the stream does not exist
        :raises KvError: the attempt did not finish within its bound -- an ungranted consumer create
            is never answered -- or the consumer create failed for another reason
        """
        bound = self._timings.key_listing_timeout_seconds
        try:
            async with asyncio.timeout(bound):
                found = await self._list_keys_through(stream=stream, filter_subject=filter_subject)
        except TimeoutError as exc:
            raise KvError(
                f"listing keys of {self._full_name} did not finish within {bound:g}s. "
                f"an ungranted consumer create blocks to its deadline -- check this principal's grant on "
                f"$JS.API.CONSUMER.CREATE.{stream}.*.{filter_subject}"
            ) from exc
        return found

    async def _list_keys_through(self, *, stream: str, filter_subject: str) -> list[str]:
        """create one named, headers-only consumer and read each key's latest message through it.

        :param stream: the bucket's backing stream
        :ptype stream: str
        :param filter_subject: the subject filter, inside this bucket
        :ptype filter_subject: str
        :return: the subjects of the live keys, in stream order
        :rtype: list[str]
        :raises KvError: when the NATS connection is closed or the consumer create fails
        """
        raw = self._client.raw
        if raw.is_closed:
            raise KvError(f"cannot list keys of {self._full_name}: the NATS connection is closed")
        inbox = raw.new_inbox()
        subscription = await raw.subscribe(inbox)
        subjects: list[str] = []
        try:
            config = ConsumerConfig(
                name=f"{_KEY_LISTING_CONSUMER_PREFIX}{uuid.uuid7().hex}",
                deliver_subject=inbox,
                filter_subject=filter_subject,
                deliver_policy=DeliverPolicy.LAST_PER_SUBJECT,
                ack_policy=AckPolicy.NONE,
                headers_only=True,
                inactive_threshold=_KEY_WATCH_INACTIVE_THRESHOLD_SECONDS,
                mem_storage=True,
            )
            try:
                info = await self._client.jetstream_context().add_consumer(stream, config=config)
            except Exception as exc:
                raise _kv_error(
                    f"key listing consumer on {stream} could not be created: {exc}", bucket=self._full_name, cause=exc
                ) from exc
            pending = int(info.num_pending or 0)
            while pending > 0:
                msg = await subscription.next_msg(timeout=self._timings.key_listing_timeout_seconds)
                pending = int(msg.metadata.num_pending)
                if (msg.headers or {}).get(_KV_OPERATION_HEADER) not in _KV_REMOVAL_OPERATIONS:
                    subjects.append(msg.subject)
        finally:
            await _drop_subscription(subscription, subject=filter_subject)
        return subjects


class _KeyWatchConsumer:
    """one named push consumer on one key, and the inbox subscription it delivers to.

    Built by :meth:`open` and consumed once by :meth:`updates`; :meth:`NatsKvBucket.watch_key`
    replaces it when it goes quiet.

    :param subscription: the core subscription on the consumer's deliver inbox
    :ptype subscription: Any
    :param name: the consumer's name
    :ptype name: str
    :param subject: the watched key's subject
    :ptype subject: str
    :param heartbeat: the consumer's idle heartbeat
    :ptype heartbeat: timedelta
    :param client: the wrapper client, asked whether it outlives the consumer's connection
    :ptype client: NatsClient
    """

    __slots__ = ("_client", "_heartbeat", "_name", "_subject", "_subscription")

    def __init__(
        self,
        *,
        subscription: _NatsSubscription,
        name: str,
        subject: str,
        heartbeat: timedelta,
        client: NatsClient,
    ) -> None:
        self._subscription = subscription
        self._name = name
        self._subject = subject
        self._heartbeat = heartbeat
        self._client = client

    @classmethod
    async def open(
        cls,
        *,
        client: NatsClient,
        stream: str,
        subject: str,
        heartbeat: timedelta,
        retry: timedelta,
        op_timeout_seconds: float,
    ) -> _KeyWatchConsumer | None:
        """subscribe a fresh inbox, then create a freshly named consumer delivering to it.

        The inbox is subscribed FIRST so nothing the consumer delivers can arrive before anything
        is listening for it.

        :param client: the connected wrapper client
        :ptype client: NatsClient
        :param stream: the bucket's backing stream
        :ptype stream: str
        :param subject: the watched key's subject
        :ptype subject: str
        :param heartbeat: the consumer's idle heartbeat
        :ptype heartbeat: timedelta
        :param retry: the pause the caller takes after a failure, for the log line
        :ptype retry: timedelta
        :param op_timeout_seconds: the bucket's ceiling on one KV operation, which bounds the create
        :ptype op_timeout_seconds: float
        :return: the consumer, or ``None`` when it could not be created
        :rtype: _KeyWatchConsumer | None
        :raises KvError: when the NATS connection is closed
        """
        raw = client.raw
        if raw.is_closed:
            raise KvError(f"cannot watch {subject}: the NATS connection is closed")
        name = f"{_KEY_WATCH_CONSUMER_PREFIX}{uuid.uuid7().hex}"
        inbox = raw.new_inbox()
        subscription = await raw.subscribe(inbox)
        config = ConsumerConfig(
            name=name,
            deliver_subject=inbox,
            filter_subject=subject,
            deliver_policy=DeliverPolicy.LAST_PER_SUBJECT,
            ack_policy=AckPolicy.NONE,
            idle_heartbeat=heartbeat.total_seconds(),
            inactive_threshold=_KEY_WATCH_INACTIVE_THRESHOLD_SECONDS,
            mem_storage=True,
        )
        js = client.jetstream_context()
        consumer: _KeyWatchConsumer | None = None
        try:
            await run_bounded(
                lambda: js.add_consumer(stream, config=config),
                timeout=op_timeout_seconds,
                what=f"key watch consumer create on {stream}",
            )
            consumer = cls(subscription=subscription, name=name, subject=subject, heartbeat=heartbeat, client=client)
        # NOSILENT: logged naming the grant to check; the caller pauses and creates another
        except Exception as exc:  # noqa: BLE001 -- a failed create is retried, never raised
            log.warning(
                "key watch consumer %s on %s could not be created; retrying in %.0fs. an ungranted "
                "create blocks to its deadline -- check this principal's grant on "
                "$JS.API.CONSUMER.CREATE.%s.*.%s: %s",
                name,
                subject,
                retry.total_seconds(),
                stream,
                subject,
                exc,
                extra={"extra_data": {"subject": subject, "stream": stream, "consumer": name, "error": str(exc)}},
            )
            await _drop_subscription(subscription, subject=subject)
        if consumer is not None:
            log.debug(
                "key watch consumer created",
                extra={"extra_data": {"subject": subject, "stream": stream, "consumer": name}},
            )
        return consumer

    async def updates(self, *, key: str) -> AsyncGenerator[KvKeyUpdate]:
        """every message the consumer delivers, until its heartbeats stop.

        :param key: the watched key, carried onto each update
        :ptype key: str
        :return: the key's messages, in order
        :rtype: AsyncGenerator[KvKeyUpdate]
        :raises KvError: when the NATS connection is closed
        """
        silence = self._heartbeat.total_seconds() * _KEY_WATCH_MISSED_HEARTBEATS
        while True:
            try:
                msg = await self._subscription.next_msg(timeout=silence)
            except _NatsConnectionClosedError as exc:
                if self._client.is_closed:
                    raise KvError(f"key watch on {self._subject} ended: the NATS connection is closed") from exc
                # NOSILENT: the consumer's connection was retired by a credential renewal while the
                # client lives on a successor; the caller replaces the consumer there, and the
                # replacement redelivers the key's latest message.
                log.info(
                    "key watch consumer %s on %s lost its connection to a credential renewal; replacing it",
                    self._name,
                    self._subject,
                    extra={"extra_data": {"subject": self._subject, "consumer": self._name}},
                )
                return
            except TimeoutError:
                # NOSILENT: a consumer that stopped heartbeating is replaced by the caller
                log.info(
                    "key watch consumer %s on %s missed %d heartbeats; replacing it",
                    self._name,
                    self._subject,
                    _KEY_WATCH_MISSED_HEARTBEATS,
                    extra={"extra_data": {"subject": self._subject, "consumer": self._name}},
                )
                return
            headers = msg.headers or {}
            status = headers.get(Header.STATUS) if not msg.data else None
            if status == _STATUS_IDLE_HEARTBEAT:
                # proof of life, and nothing to yield
                continue
            if status is not None:
                # the server ended the consumer (``409 Consumer Deleted``, and its kin): replace it
                # now rather than wait out the heartbeats it will never send.
                log.info(
                    "key watch consumer %s on %s ended by the server (%s %s); replacing it",
                    self._name,
                    self._subject,
                    status,
                    headers.get(Header.DESCRIPTION, ""),
                    extra={"extra_data": {"subject": self._subject, "consumer": self._name, "status": status}},
                )
                return
            yield self._update_of(msg, key=key)

    def _update_of(self, msg: Msg, *, key: str) -> KvKeyUpdate:
        """the update a delivered data message carries.

        :param msg: the delivered message
        :ptype msg: Msg
        :param key: the watched key
        :ptype key: str
        :return: the update
        :rtype: KvKeyUpdate
        """
        headers = msg.headers or {}
        try:
            revision = int(msg.metadata.sequence.stream)
        except Exception as exc:  # noqa: BLE001 -- the value is still delivered; only its revision is unknown
            log.warning(
                "key watch message on %s carries no stream sequence; delivering it with revision 0: %s",
                self._subject,
                exc,
                extra={"extra_data": {"subject": self._subject, "consumer": self._name, "error": str(exc)}},
            )
            revision = 0
        removed = headers.get(_KV_OPERATION_HEADER) in _KV_REMOVAL_OPERATIONS
        return KvKeyUpdate(key=key, value=None if removed else bytes(msg.data), revision=revision)

    async def close(self) -> None:
        """drop the inbox subscription; the server reaps the consumer after its inactivity threshold.

        :return: nothing
        :rtype: None
        """
        await _drop_subscription(self._subscription, subject=self._subject)


async def _drop_subscription(subscription: _NatsSubscription, *, subject: str) -> None:
    """unsubscribe a key watch's inbox, logging rather than raising a failure.

    :param subscription: the core subscription on the deliver inbox
    :ptype subscription: _NatsSubscription
    :param subject: the watched key's subject, for the log line
    :ptype subject: str
    :return: nothing
    :rtype: None
    """
    try:
        await subscription.unsubscribe()
    # NOSILENT: logged; the consumer behind it is reaped by the server once nothing listens
    except Exception as exc:  # noqa: BLE001 -- teardown continues regardless
        log.debug("key watch unsubscribe on %s failed: %s", subject, exc)


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

    async def get_latest(self, *, key: str) -> tuple[bytes | None, int]: ...

    async def put(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int: ...

    async def create(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int | None: ...

    async def update(self, *, key: str, value: bytes, revision: int, ttl: timedelta | None = None) -> int | None: ...

    async def delete(self, *, key: str, revision: int | None = None) -> bool: ...

    async def date_created(self) -> datetime: ...


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
        direct: bool | None = None,
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
        owns_bucket: bool = False,
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
        :param owns_bucket: this declarer is the bucket's one owner of its whole shape, so a live
            bucket-wide expiry or history other than the declared one is reconciled in place, and a
            live storage other than the declared one by recreating the bucket empty; only with
            ``create_if_missing`` on memory storage
        :ptype owns_bucket: bool
        :return: ready bucket handle
        :rtype: KvBucketLike
        :raises ValueError: if ``owns_bucket`` is asked of a bind (``create_if_missing=False``) or of a
            bucket on file storage
        :raises KvBucketNotFoundError: if the bucket does not exist and this call could not create it
        :raises KvError: if bucket creation or binding fails for any other reason
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
