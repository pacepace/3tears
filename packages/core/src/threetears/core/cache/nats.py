"""NATS JetStream Key-Value client for threetears cache operations.

All public methods return safe defaults (None/False) on error — they never
raise so callers do not need try/except guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import nats
from nats.aio.client import Client as NATSClient
from nats.js.api import KeyValueConfig, StorageType
from nats.js.errors import KeyNotFoundError, KeyWrongLastSequenceError
from nats.js.kv import KeyValue

from threetears.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class BucketConfig:
    """Configuration for an additional NATS KV bucket."""

    name: str  # suffix (e.g., "ratelimits")
    ttl_seconds: int  # 0 = no expiry
    storage: str = "memory"  # "memory" or "file"


class NatsClient:
    """NATS JetStream Key-Value client with configurable bucket prefix.

    All public KV operations are fail-open: they return None/False on error
    and never raise exceptions.
    """

    def __init__(self, bucket_prefix: str = "threetears") -> None:
        self._prefix = bucket_prefix
        self._nc: NATSClient | None = None
        self._js: Any = None  # JetStreamContext
        self._buckets: dict[str, KeyValue] = {}

    def bucket_name(self, suffix: str) -> str:
        """Return the full bucket name for a given suffix."""
        return f"{self._prefix}-{suffix}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, url: str, extra_buckets: list[BucketConfig] | None = None) -> None:
        """Connect to NATS, init JetStream, ensure buckets exist.

        Always creates the 'collections' bucket (TTL 7200s).
        Additional buckets can be registered via extra_buckets.
        """
        self._nc = await nats.connect(url, allow_reconnect=True, max_reconnect_attempts=-1)
        self._js = self._nc.jetstream()

        # Default buckets — collections uses FILE storage so data survives restarts
        all_buckets = [BucketConfig("collections", 7200, storage="file")]
        if extra_buckets:
            all_buckets.extend(extra_buckets)

        await self._ensure_buckets(all_buckets)

    async def _ensure_buckets(self, configs: list[BucketConfig]) -> None:
        """Create or bind KV buckets."""
        assert self._js is not None
        for cfg in configs:
            full_name = self.bucket_name(cfg.name)
            kv: KeyValue | None = None
            storage = StorageType.FILE if cfg.storage == "file" else StorageType.MEMORY
            try:
                kv = await self._js.create_key_value(
                    KeyValueConfig(
                        bucket=full_name,
                        ttl=cfg.ttl_seconds,
                        history=1,
                        storage=storage,
                    )
                )
            except Exception:
                log.warning(
                    "KV bucket already exists — binding existing bucket",
                    extra={"extra_data": {"bucket": full_name}},
                )
                kv = await self._js.key_value(full_name)
            self._buckets[full_name] = kv

    async def close(self) -> None:
        """Drain and close the NATS connection."""
        if self._nc is not None:
            await self._nc.drain()
            await self._nc.close()

    # ------------------------------------------------------------------
    # Pub/sub operations — fail-open
    # ------------------------------------------------------------------

    async def publish(self, subject: str, data: bytes) -> bool:
        """Publish a message to a NATS subject. Returns True on success."""
        if self._nc is None:
            return False
        try:
            await self._nc.publish(subject, data)
            return True
        except Exception as exc:
            log.warning(
                "NATS publish failed",
                extra={"extra_data": {"subject": subject, "error": str(exc)}},
            )
            return False

    async def subscribe(self, subject: str, callback: Any) -> Any:
        """Subscribe to a NATS subject. Callback receives raw bytes for each message.

        Returns the NATS subscription object for later unsubscribe.
        """
        if self._nc is None:
            return None
        try:
            sub = await self._nc.subscribe(subject)

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
    # Public KV operations — ALL fail-open
    # ------------------------------------------------------------------

    async def get(self, bucket: str, key: str) -> bytes | None:
        """Get value for key. Returns None on miss or error."""
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning("get called for unknown bucket", extra={"extra_data": {"bucket": bucket, "key": key}})
            return None
        try:
            entry = await kv.get(key)
            return entry.value
        except KeyNotFoundError:
            return None
        except Exception as exc:
            log.warning("KV get failed", extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}})
            return None

    async def put(self, bucket: str, key: str, value: bytes) -> bool:
        """Store value. Returns True on success, False on error."""
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning("put called for unknown bucket", extra={"extra_data": {"bucket": bucket, "key": key}})
            return False
        try:
            await kv.put(key, value)
            return True
        except Exception as exc:
            log.warning("KV put failed", extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}})
            return False

    async def delete(self, bucket: str, key: str) -> bool:
        """Delete key. Returns True on success or if key absent, False on error."""
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning("delete called for unknown bucket", extra={"extra_data": {"bucket": bucket, "key": key}})
            return False
        try:
            await kv.delete(key)
            return True
        except KeyNotFoundError:
            return True
        except Exception as exc:
            log.warning("KV delete failed", extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}})
            return False

    async def create(self, bucket: str, key: str, value: bytes) -> bool:
        """Create key only if it doesn't exist (SET NX). Returns True if created, False if exists or error."""
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning("create called for unknown bucket", extra={"extra_data": {"bucket": bucket, "key": key}})
            return False
        try:
            await kv.create(key, value)
            return True
        except KeyWrongLastSequenceError:
            return False
        except Exception as exc:
            log.warning("KV create failed", extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}})
            return False

    async def update(self, bucket: str, key: str, value: bytes, revision: int) -> int | None:
        """Compare-and-swap write. Returns new revision on success, None on mismatch or error."""
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning("update called for unknown bucket", extra={"extra_data": {"bucket": bucket, "key": key}})
            return None
        try:
            new_revision = await kv.update(key, value, revision)
            return int(new_revision)
        except KeyWrongLastSequenceError:
            return None
        except Exception as exc:
            log.warning("KV update failed", extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}})
            return None

    async def get_entry(self, bucket: str, key: str) -> tuple[bytes, int] | None:
        """Get value and revision for CAS reads. Returns (value, revision) or None."""
        kv = self._buckets.get(bucket)
        if kv is None:
            log.warning("get_entry called for unknown bucket", extra={"extra_data": {"bucket": bucket, "key": key}})
            return None
        try:
            entry = await kv.get(key)
            if entry.value is None or entry.revision is None:
                return None
            return (entry.value, entry.revision)
        except KeyNotFoundError:
            return None
        except Exception as exc:
            log.warning(
                "KV get_entry failed", extra={"extra_data": {"bucket": bucket, "key": key, "error": str(exc)}}
            )
            return None

    async def ping(self) -> bool:
        """Health check. Returns True if JetStream is reachable."""
        try:
            await self._js.account_info()
            return True
        except Exception:
            return False
