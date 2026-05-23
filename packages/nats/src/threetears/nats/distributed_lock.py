"""TTL-based distributed NATS lock primitive.

Lifted from metallm's ``api/src/services/scheduler.py:scheduler_lock``
(which has run the production backup job for months) so any 3tears app
needing single-active-holder semantics across pods can pick it up
without re-implementing the heartbeat lifecycle. The agent-wake tick
engine (``threetears.agent.wake.tick``) is the second canonical
consumer; metallm's existing ``scheduler_lock`` becomes a one-line
re-export of this primitive.

design notes
------------

- **NATS JetStream KV-backed.** Acquisition is an atomic
  ``bucket.create(key, value)`` (put-if-absent). The KV bucket's
  per-entry TTL bounds the worst-case orphan-lock window when the
  holder dies between heartbeats.
- **Background heartbeat.** A dedicated task refreshes the KV entry
  every ``heartbeat`` seconds so a long-running body stays the
  authoritative holder. ``heartbeat`` must be strictly less than
  ``ttl``; the contextmanager raises ``ValueError`` otherwise.
- **Clean cancellation on exit.** Normal exit and exception both
  cancel + await the heartbeat task before releasing the key. The
  ``finally`` block uses ``asyncio.gather(..., return_exceptions=True)``
  so a heartbeat death does not mask the body's own exception.
- **Graceful single-pod fallback.** ``client=None`` yields
  immediately without acquiring anything -- matches the existing
  metallm behaviour for dev environments that do not run NATS.
- **Bucket name namespacing.** The default bucket ``"scheduler-locks"``
  rides through :meth:`NatsClient.kv_bucket` and picks up the
  client's ``nats_subject_namespace`` prefix automatically (the
  resulting bucket is ``{namespace}-scheduler-locks``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Final

from threetears.observe import get_logger

from threetears.nats.client import NatsClient
from threetears.nats.errors import KvError

__all__ = ["LockHeld", "nats_distributed_lock"]


log = get_logger(__name__)


class LockHeld(Exception):
    """Raised by :func:`nats_distributed_lock` when another holder owns the key.

    Distinct from :class:`threetears.nats.KvError`, which signals a
    transport / bucket-creation failure -- callers can treat ``LockHeld``
    as the expected "another pod is already running this job" branch
    and re-raise / surface ``KvError`` separately.
    """


_DEFAULT_TTL: Final[timedelta] = timedelta(seconds=60)
_DEFAULT_HEARTBEAT: Final[timedelta] = timedelta(seconds=20)
_DEFAULT_BUCKET: Final[str] = "scheduler-locks"


@asynccontextmanager
async def nats_distributed_lock(
    client: NatsClient | None,
    key: str,
    *,
    bucket_name: str = _DEFAULT_BUCKET,
    ttl: timedelta = _DEFAULT_TTL,
    heartbeat: timedelta = _DEFAULT_HEARTBEAT,
) -> AsyncIterator[None]:
    """Acquire a TTL-based distributed lock backed by NATS JetStream KV.

    Usage::

        try:
            async with nats_distributed_lock(nats, "my_job"):
                ... # body runs while we hold the lock
        except LockHeld:
            return  # another holder owns this key; skip this turn

    A background heartbeat task refreshes the KV entry every
    ``heartbeat`` seconds so the lock stays alive for arbitrarily
    long bodies. On normal exit or exception the heartbeat is
    cancelled and the key is deleted. If the process dies the
    heartbeat stops, the TTL expires the key within ``ttl`` seconds,
    and another claimer can win on the next attempt.

    When ``client`` is ``None`` the context manager yields immediately
    without acquiring -- safe for single-pod dev environments without
    NATS available.

    :param client: connected NATS client, or ``None`` to no-op
    :ptype client: NatsClient | None
    :param key: lock key (per-resource identifier, e.g. ``"backup"``
        for the metallm backup job or ``"agent_wake_tick"`` for the
        wake tick engine)
    :ptype key: str
    :param bucket_name: KV bucket suffix; the connected client's
        namespace is prefixed automatically. Defaults to
        ``"scheduler-locks"`` so existing metallm prod state continues
        to bind to the same bucket post-lift.
    :ptype bucket_name: str
    :param ttl: KV entry TTL; bounds the orphan-lock window after a
        holder dies between heartbeats
    :ptype ttl: timedelta
    :param heartbeat: KV refresh interval; must be strictly less than
        ``ttl`` or the lock will expire under a live holder
    :ptype heartbeat: timedelta
    :return: async iterator yielding ``None`` while the lock is held
    :rtype: AsyncIterator[None]
    :raises ValueError: when ``heartbeat >= ttl`` (invalid invariant)
    :raises LockHeld: when the key is already owned by another holder
    :raises KvError: on transport / bucket failures (distinct from
        ``LockHeld``)
    """
    if client is None:
        yield
        return
    if heartbeat >= ttl:
        msg = f"heartbeat {heartbeat} must be less than ttl {ttl}"
        raise ValueError(msg)

    bucket = await client.kv_bucket(name=bucket_name, ttl=ttl)
    acquired = await bucket.create(key=key, value=b"1")
    if acquired is None:
        raise LockHeld(f"lock already held: {key}")

    heartbeat_seconds = heartbeat.total_seconds()

    async def _heartbeat() -> None:
        """Refresh the KV entry until cancelled.

        ``CancelledError`` is the normal exit path (the contextmanager
        cancels us on body completion). Any other exception is logged
        at WARNING so the orphan-lock-after-heartbeat-death case is
        diagnosable; the lock then auto-expires after ``ttl``.
        """
        try:
            while True:
                await asyncio.sleep(heartbeat_seconds)
                await bucket.put(key=key, value=b"1")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - boundary: log + let TTL expire the lock
            log.warning(
                "nats_distributed_lock: heartbeat failed; lock will expire after ttl",
                extra={
                    "extra_data": {
                        "key": key,
                        "bucket": bucket_name,
                        "ttl_seconds": ttl.total_seconds(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                },
            )

    hb_task = asyncio.create_task(_heartbeat(), name=f"nats-lock-heartbeat:{key}")
    try:
        yield
    finally:
        hb_task.cancel()
        # return_exceptions=True so heartbeat-death does not mask a
        # body exception that we are about to re-raise via the context
        # manager protocol.
        await asyncio.gather(hb_task, return_exceptions=True)
        try:
            await bucket.delete(key=key)
        except KvError as exc:
            log.debug(
                "nats_distributed_lock: cleanup delete failed (key already gone?)",
                extra={"extra_data": {"key": key, "bucket": bucket_name, "error": str(exc)}},
            )
