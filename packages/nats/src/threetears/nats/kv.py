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

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

from nats.js.api import KeyValueConfig, StorageType
from nats.js.errors import KeyNotFoundError, KeyWrongLastSequenceError
from threetears.observe import get_logger

from threetears.nats.errors import KvError

if TYPE_CHECKING:
    from nats.js.kv import KeyValue

    from threetears.nats.client import NatsClient

__all__ = ["NatsKvBucket"]


log = get_logger(__name__)


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
        storage: str = "file",
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
                    raise KvError(
                        f"open KV bucket failed: bucket={full_name}: create={exc!r} bind={bind_exc!r}"
                    ) from bind_exc
                log.debug(
                    "JetStream KV bucket bound (already existed)",
                    extra={"extra_data": {"bucket": full_name}},
                )
        else:
            try:
                kv = await js.key_value(full_name)
            except Exception as exc:
                raise KvError(f"bind KV bucket failed: bucket={full_name}: {exc}") from exc

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
            return await op()
        except passthrough:
            raise
        except Exception:  # noqa: BLE001 - transport failure: self-heal once, then let it surface
            await self._reopen()
            return await op()

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
