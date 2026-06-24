"""multi-bucket fail-open KV facade over :class:`threetears.nats.NatsClient`.

:class:`NatsKvClient` is the multi-bucket KV facade L2 cache callers
bind to. it composes over the canonical
:class:`threetears.nats.NatsClient` wrapper -- one
:class:`~threetears.nats.NatsKvBucket` per registered suffix, fail-open
semantics layered on top so :class:`BaseCollection` consumers do not
need try/except guards.

design notes
------------

- **fail-open**: every public method swallows transport failures and
  returns a safe default (``None`` / ``False``). cache misses and
  broker outages must not bubble up into the calling collection.
- **multi-bucket**: callers register several bucket suffixes under one
  shared connection (``collections``, ``ratelimits``, ...) and address
  reads / writes by ``(bucket_name, key)``. the bucket suffix is the
  one that was passed to :meth:`connect` /
  :meth:`register_extra_bucket`; the wrapper resolves it to the full
  prefixed name internally.
- **renamed from cache.nats.NatsClient**: the canonical name
  :class:`threetears.nats.NatsClient` now belongs to the wrapper (full
  pub/sub + request/reply + KV + lifecycle); this class is the
  KV-only facade that L2 cache callers depend on.

callers do NOT import nats-py here -- the wrapper owns every nats-py
dependency. this file deliberately has no ``from nats import`` or
``from nats.aio`` lines so the per-repo enforcement walker stays
clean for ``packages/core/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from threetears.nats import NatsClient as _NatsTransport
from threetears.nats import NatsKvBucket
from threetears.observe import get_logger

__all__ = [
    "BucketConfig",
    "NatsKvClient",
]

log = get_logger(__name__)


@dataclass
class BucketConfig:
    """configuration for one KV bucket managed by :class:`NatsKvClient`.

    :param name: bucket suffix (e.g. ``"ratelimits"``); fully-qualified bucket name is ``{prefix}-{name}``
    :ptype name: str
    :param ttl_seconds: per-entry time-to-live; ``0`` disables expiry
    :ptype ttl_seconds: int
    :param storage: ``"memory"`` (default — NATS is L2) or ``"file"`` (opt-in)
    :ptype storage: str
    """

    name: str
    ttl_seconds: int
    storage: str = "memory"


class NatsKvClient:
    """multi-bucket fail-open KV facade over :class:`NatsClient`.

    constructed with a bucket prefix; :meth:`connect` opens the
    underlying :class:`threetears.nats.NatsClient` and registers a
    default ``collections`` bucket (TTL 7200s, file storage).
    additional buckets ride alongside via the ``extra_buckets``
    argument. every public read/write returns a safe default on error
    so :class:`~threetears.core.collections.BaseCollection` callers can
    treat the L2 tier as fail-open.

    :param bucket_prefix: namespace prefix prepended to every bucket name
    :ptype bucket_prefix: str
    """

    def __init__(self, bucket_prefix: str = "threetears") -> None:
        self._prefix = bucket_prefix
        self._transport: _NatsTransport | None = None
        self._buckets: dict[str, NatsKvBucket] = {}

    @property
    def transport(self) -> _NatsTransport | None:
        """connected :class:`NatsClient` transport, or ``None`` before :meth:`connect`.

        exposed (no leading underscore) so callers that already hold a
        :class:`NatsKvClient` can reach the underlying wrapper for
        pub/sub or request/reply work without opening a second
        connection. the property name is part of the public api per
        the underscore-stability-contract rule.

        :return: connected transport or ``None``
        :rtype: NatsClient | None
        """
        return self._transport

    @property
    def buckets(self) -> dict[str, NatsKvBucket]:
        """fully-qualified bucket name -> :class:`NatsKvBucket` mapping.

        exposed for legacy callers that look up buckets by full name
        (e.g. ``client.buckets[client.bucket_name("collections")]``).

        :return: registered buckets keyed by full name
        :rtype: dict[str, NatsKvBucket]
        """
        return self._buckets

    def bucket_name(self, suffix: str) -> str:
        """resolve fully-qualified bucket name for a registered suffix.

        :param suffix: bucket suffix passed to :class:`BucketConfig.name`
        :ptype suffix: str
        :return: full bucket name (``{prefix}-{suffix}``)
        :rtype: str
        """
        return f"{self._prefix}-{suffix}"

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def connect(
        self,
        url: str,
        extra_buckets: list[BucketConfig] | None = None,
    ) -> None:
        """connect to NATS and ensure the default + requested buckets exist.

        always registers the ``collections`` bucket (TTL 7200s, file
        storage) so every :class:`BaseCollection` finds an L2 tier.
        additional bucket suffixes ride on ``extra_buckets``.

        :param url: NATS server URL
        :ptype url: str
        :param extra_buckets: additional bucket suffixes to ensure exist
        :ptype extra_buckets: list[BucketConfig] | None
        :return: nothing
        :rtype: None
        """
        from datetime import timedelta

        self._transport = await _NatsTransport.connect(
            nats_url=url,
            nats_subject_namespace=self._prefix,
            client_name=f"{self._prefix}-kv",
        )

        all_buckets = [BucketConfig("collections", 7200)]  # memory: NATS is L2 (durability = R3 + L3)
        if extra_buckets:
            all_buckets.extend(extra_buckets)

        for cfg in all_buckets:
            ttl = timedelta(seconds=cfg.ttl_seconds) if cfg.ttl_seconds > 0 else None
            full_name = self.bucket_name(cfg.name)
            try:
                bucket = await self._transport.kv_bucket(
                    name=cfg.name,
                    ttl=ttl,
                    storage=cfg.storage,
                    create_if_missing=True,
                    history=1,
                )
                self._buckets[full_name] = bucket
            except Exception as exc:
                # fail-open: log and continue. callers reading this
                # bucket will hit the unknown-bucket fail-open path
                # below and observe a None / False return.
                log.warning(
                    "KV bucket open failed",
                    extra={"extra_data": {"bucket": full_name, "error": str(exc)}},
                )

    async def close(self) -> None:
        """drain and close the underlying NATS connection.

        idempotent -- second call is a no-op.

        :return: nothing
        :rtype: None
        """
        if self._transport is not None:
            await self._transport.shutdown()
            self._transport = None

    # ------------------------------------------------------------------
    # pub/sub passthroughs
    # ------------------------------------------------------------------

    async def publish(self, subject: str, data: bytes) -> bool:
        """publish raw bytes to subject; fail-open.

        :param subject: NATS subject string (already-formatted; not a :class:`Subject`)
        :ptype subject: str
        :param data: payload bytes
        :ptype data: bytes
        :return: ``True`` on success, ``False`` on transport failure
        :rtype: bool
        """
        if self._transport is None:
            return False
        try:
            await self._transport.raw.publish(subject, data)
            return True
        except Exception as exc:
            log.warning(
                "NATS publish failed",
                extra={"extra_data": {"subject": subject, "error": str(exc)}},
            )
            return False

    async def subscribe(self, subject: str, callback: Any) -> Any:
        """subscribe to subject; callback receives raw bytes.

        :param subject: NATS subject string (already-formatted)
        :ptype subject: str
        :param callback: async callback ``(bytes) -> None``
        :ptype callback: Any
        :return: opaque subscription handle, or ``None`` on transport failure
        :rtype: Any
        """
        if self._transport is None:
            return None
        try:
            sub = await self._transport.raw.subscribe(subject)

            async def _dispatch() -> None:
                async for msg in sub.messages:
                    try:
                        await callback(msg.data)
                    except Exception as exc:
                        log.warning(
                            "NATS subscription callback error",
                            extra={"extra_data": {"subject": subject, "error": str(exc)}},
                        )

            import asyncio

            asyncio.create_task(_dispatch())
            return sub
        except Exception as exc:
            log.warning(
                "NATS subscribe failed",
                extra={"extra_data": {"subject": subject, "error": str(exc)}},
            )
            return None

    # ------------------------------------------------------------------
    # KV operations -- ALL fail-open
    # ------------------------------------------------------------------

    async def get(self, bucket: str, key: str) -> bytes | None:
        """get value for key. returns ``None`` on miss or error.

        :param bucket: fully-qualified bucket name
        :ptype bucket: str
        :param key: key to read
        :ptype key: str
        :return: stored bytes, or ``None`` on miss / error / unknown bucket
        :rtype: bytes | None
        """
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning(
                "get called for unknown bucket",
                extra={"extra_data": {"bucket": bucket, "key": key}},
            )
            return None
        try:
            return await kv.get(key=key)
        except Exception as exc:
            log.warning(
                "KV get failed",
                extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}},
            )
            return None

    async def put(self, bucket: str, key: str, value: bytes) -> bool:
        """store value. returns ``True`` on success, ``False`` on error.

        :param bucket: fully-qualified bucket name
        :ptype bucket: str
        :param key: key to write
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :return: success indicator
        :rtype: bool
        """
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning(
                "put called for unknown bucket",
                extra={"extra_data": {"bucket": bucket, "key": key}},
            )
            return False
        try:
            await kv.put(key=key, value=value)
            return True
        except Exception as exc:
            log.warning(
                "KV put failed",
                extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}},
            )
            return False

    async def delete(self, bucket: str, key: str) -> bool:
        """delete key. returns ``True`` on success or absent, ``False`` on error.

        :param bucket: fully-qualified bucket name
        :ptype bucket: str
        :param key: key to delete
        :ptype key: str
        :return: success indicator
        :rtype: bool
        """
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning(
                "delete called for unknown bucket",
                extra={"extra_data": {"bucket": bucket, "key": key}},
            )
            return False
        try:
            return await kv.delete(key=key)
        except Exception as exc:
            log.warning(
                "KV delete failed",
                extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}},
            )
            return False

    async def create(self, bucket: str, key: str, value: bytes) -> bool:
        """create key only if it does not exist (SET NX). ``True`` if created, else ``False``.

        :param bucket: fully-qualified bucket name
        :ptype bucket: str
        :param key: key to create
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :return: ``True`` if newly created, ``False`` if exists / error
        :rtype: bool
        """
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning(
                "create called for unknown bucket",
                extra={"extra_data": {"bucket": bucket, "key": key}},
            )
            return False
        try:
            revision = await kv.create(key=key, value=value)
            return revision is not None
        except Exception as exc:
            log.warning(
                "KV create failed",
                extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}},
            )
            return False

    async def update(self, bucket: str, key: str, value: bytes, revision: int) -> int | None:
        """compare-and-swap write. returns new revision on success, ``None`` on mismatch / error.

        :param bucket: fully-qualified bucket name
        :ptype bucket: str
        :param key: key to update
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :param revision: expected current revision for CAS
        :ptype revision: int
        :return: new revision, or ``None`` on conflict / error
        :rtype: int | None
        """
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning(
                "update called for unknown bucket",
                extra={"extra_data": {"bucket": bucket, "key": key}},
            )
            return None
        try:
            return await kv.update(key=key, value=value, revision=revision)
        except Exception as exc:
            log.warning(
                "KV update failed",
                extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}},
            )
            return None

    async def get_entry(self, bucket: str, key: str) -> tuple[bytes, int] | None:
        """get value + revision for CAS reads. returns ``(value, revision)`` or ``None``.

        :param bucket: fully-qualified bucket name
        :ptype bucket: str
        :param key: key to read
        :ptype key: str
        :return: tuple of (value, revision), or ``None`` on miss / error
        :rtype: tuple[bytes, int] | None
        """
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning(
                "get_entry called for unknown bucket",
                extra={"extra_data": {"bucket": bucket, "key": key}},
            )
            return None
        try:
            return await kv.get_entry(key=key)
        except Exception as exc:
            log.warning(
                "KV get_entry failed",
                extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}},
            )
            return None

    async def ping(self) -> bool:
        """health check. returns ``True`` if JetStream is reachable.

        :return: reachability indicator
        :rtype: bool
        """
        if self._transport is None:
            return False
        try:
            js = self._transport.jetstream_context()
            await js.account_info()
            return True
        except Exception:
            return False
