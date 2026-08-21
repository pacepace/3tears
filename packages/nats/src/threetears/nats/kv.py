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
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from nats.js.api import KeyValueConfig, StorageType
from nats.js.errors import KeyNotFoundError, KeyWrongLastSequenceError
from threetears.observe import get_logger

from threetears.nats._diagnostics import kv_grant_remedy, kv_timeout_remedy
from threetears.nats._publish import run_bounded
from threetears.nats.errors import KvError, PublishTimeoutError

if TYPE_CHECKING:
    from nats.js.kv import KeyValue

    from threetears.nats.client import NatsClient

__all__ = ["NatsKvBucket"]


log = get_logger(__name__)

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

    __slots__ = ("_client", "_create_if_missing", "_full_name", "_history", "_kv", "_storage", "_ttl")

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
    ) -> NatsKvBucket:
        """open or create a JetStream KV bucket.

        called by :meth:`NatsClient.kv_bucket`. excluded from
        ``threetears.nats.__all__`` because callers should not bypass
        the client's bucket cache; the public path is
        :meth:`NatsClient.kv_bucket`.

        :param client: connected wrapper client
        :ptype client: NatsClient
        :param full_name: fully-qualified bucket name
        :ptype full_name: str
        :param ttl: TTL for entries; ``None`` for no expiry
        :ptype ttl: timedelta | None
        :param storage: ``"file"`` or ``"memory"``
        :ptype storage: str
        :param create_if_missing: create bucket if absent
        :ptype create_if_missing: bool
        :param history: per-key historical revision count
        :ptype history: int
        :return: ready bucket
        :rtype: NatsKvBucket
        :raises KvError: if bucket creation or binding fails
        """
        js = client.jetstream_context()
        storage_type = StorageType.FILE if storage == "file" else StorageType.MEMORY
        ttl_seconds = int(ttl.total_seconds()) if ttl is not None else 0

        kv: KeyValue
        if create_if_missing:
            try:
                kv = await js.create_key_value(
                    KeyValueConfig(
                        bucket=full_name,
                        ttl=ttl_seconds,
                        history=history,
                        storage=storage_type,
                    )
                )
                log.info(
                    "JetStream KV bucket created",
                    extra={
                        "extra_data": {
                            "bucket": full_name,
                            "ttl_seconds": ttl_seconds,
                            "storage": storage,
                            "history": history,
                        }
                    },
                )
            except Exception as exc:
                # bucket likely exists already; bind to it.
                try:
                    kv = await js.key_value(full_name)
                except Exception as bind_exc:
                    # Both halves failed, which is the shape an ungranted bucket produces:
                    # STREAM.CREATE and STREAM.INFO are refused alike and neither is answered,
                    # so both calls die on their own deadline. Carry the fix in the exception
                    # rather than only in a log line -- this one propagates to the caller, and
                    # for the bucket a component opens on first use it is the FIRST symptom.
                    #
                    # Unhedged here, unlike the bind-only branch below: `create_if_missing` was
                    # asked for, so a bucket that merely does not exist would have been CREATED.
                    # Reaching the bind at all means the create failed for some other reason,
                    # and the bind failing too rules out "it already existed".
                    raise KvError(
                        f"open KV bucket failed: bucket={full_name}: create={exc!r} bind={bind_exc!r}. "
                        f"{kv_grant_remedy(full_name)}"
                    ) from bind_exc
                log.debug(
                    "JetStream KV bucket bound (already existed)",
                    extra={"extra_data": {"bucket": full_name}},
                )
        else:
            try:
                kv = await js.key_value(full_name)
            except Exception as exc:
                # Hedged: this branch never attempts a create, so a bucket nobody has
                # created yet fails here exactly the way an ungranted one does.
                raise KvError(
                    f"bind KV bucket failed: bucket={full_name}: {exc}. {kv_grant_remedy(full_name, certain=False)}"
                ) from exc

        return cls(
            client=client,
            full_name=full_name,
            kv=kv,
            ttl=ttl,
            storage=storage,
            create_if_missing=create_if_missing,
            history=history,
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
        """
        rebound = await NatsKvBucket.open(
            client=self._client,
            full_name=self._full_name,
            ttl=self._ttl,
            storage=self._storage,
            create_if_missing=self._create_if_missing,
            history=self._history,
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
class JetStreamPublisher(Protocol):
    """The slice of :class:`NatsClient` a durable-publish caller needs.

    One method, because that is genuinely all an audit emitter or any other
    fire-and-persist producer uses. Sits beside :class:`KvCapable` for the same reason:
    a signature naming the whole client forces every test that passes a recording double
    to launder it through a cast, which is a hole in exactly the tests that exist to prove
    the right thing was published.
    """

    async def jetstream_publish(self, *, subject: Any, payload: bytes) -> None: ...
