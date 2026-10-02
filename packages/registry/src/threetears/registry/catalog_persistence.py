"""owner of the registry's catalog bucket: declared at start, and again after every NATS reconnect.

the in-memory :class:`~threetears.registry.catalog.ToolCatalog` is what the registry routes from.
the catalog bucket (``tool_catalog`` by default) is its persisted copy, read only to warm-load a
starting registry -- but every registration also writes to it, and answers ``CATALOG_UNAVAILABLE``
when that write fails. a NATS restart that loses its storage -- memory storage always, and on
Kubernetes file storage too, since a restarted NATS pod can come back without its JetStream
volume -- takes the bucket under a registry that keeps running, and nothing else recreates it.

this bucket does not go through :meth:`threetears.nats.NatsClient.ensure_kv_bucket`, whose
remembered declarations the client re-creates after a reconnect on its own, for two reasons. that
path names every bucket ``{namespace}-{name}``, and this one has always been the bare name every
deployment's NATS permissions grant (``$KV.tool_catalog.>``, ``KV_tool_catalog``), so moving it
would orphan the persisted catalog and break the grants. and a bucket the client re-creates comes
back EMPTY: a registry that keeps routing from its in-memory catalog would leave the copy short of
every tool registered before the restart until each pod happened to register again. so this owner
re-declares the bucket itself and writes the in-memory catalog back into it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Final, Protocol

from threetears.nats import is_bucket_not_found
from threetears.observe import get_logger
from threetears.observe.resilience import retry_with_backoff

from threetears.registry.catalog import ToolCatalog

__all__ = ["CatalogBucketClient", "CatalogPersistence", "CatalogRestoreError"]

log = get_logger(__name__)

#: first pause before a restore after a reconnect tries again; doubles each failure up to the cap
_RESTORE_FIRST_DELAY_SECONDS: Final[float] = 0.5

#: longest pause between two attempts of a restore that keeps failing. it never gives up on its own:
#: every registration fails until the bucket is back
_RESTORE_MAX_DELAY_SECONDS: Final[float] = 30.0


class CatalogRestoreError(RuntimeError):
    """some catalog entries could not be written back into the bucket after a reconnect."""


class CatalogBucketClient(Protocol):
    """the slice of the registry's NATS client the catalog bucket's owner needs."""

    def jetstream_context(self) -> Any:
        """the raw JetStream context the bucket is bound and created through.

        :return: nats-py JetStream context
        :rtype: Any
        """
        ...

    def add_reconnect_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """register an argument-less coroutine function run after each successful reconnect.

        :param callback: the hook
        :ptype callback: Callable[[], Awaitable[None]]
        :return: nothing
        :rtype: None
        """
        ...


class CatalogPersistence:
    """owns the catalog bucket: declared and read at start, declared and rewritten after every reconnect.

    **at start** the bucket is declared and the catalog warm-loads from it, best-effort as it always
    was: a registry that cannot load serves from an empty catalog its pods fill by registering.

    **after every reconnect** the same declaration runs again and every entry of the in-memory
    catalog is written back into the bucket, in the background -- the reconnect path must not wait
    on a broker that may still be recovering -- retried with capped backoff until the bucket is
    declared and every entry landed, each failed attempt logged at ERROR. a later reconnect replaces
    a restore still running; :meth:`stop` ends it. a bucket that survived is bound, not created.

    :param catalog: the in-memory catalog the bucket is a copy of
    :ptype catalog: ToolCatalog
    :param nc: the registry's NATS client
    :ptype nc: CatalogBucketClient
    :param bucket: the bucket's name, used exactly as given
    :ptype bucket: str
    """

    def __init__(self, *, catalog: ToolCatalog, nc: CatalogBucketClient, bucket: str) -> None:
        """bind the catalog, the client and the bucket name; no I/O.

        :param catalog: the in-memory catalog the bucket is a copy of
        :ptype catalog: ToolCatalog
        :param nc: the registry's NATS client
        :ptype nc: CatalogBucketClient
        :param bucket: the bucket's name, used exactly as given
        :ptype bucket: str
        :return: nothing
        :rtype: None
        """
        self._catalog = catalog
        self._nc = nc
        self._bucket = bucket
        self._restoration: asyncio.Task[None] | None = None

    async def declare(self) -> Any:
        """bind the bucket, creating it only when the broker answered that it does not exist.

        a bucket that exists is bound with its entries kept. one the server answers is absent is
        created with nats-py's defaults -- file storage, history 1 -- the config this bucket has
        always been created with. any other bind failure (a deadline, a refusal) is raised, not
        answered with a create: a create would fail the same way, and a bucket that is merely
        unreachable must not be replaced.

        :return: raw nats-py KeyValue handle bound to the bucket
        :rtype: Any
        :raises Exception: whatever the bind raised when it was not an absent bucket, or whatever
            the create raised
        """
        js = self._nc.jetstream_context()
        try:
            kv = await js.key_value(bucket=self._bucket)
        except Exception as exc:
            if not is_bucket_not_found(exc):
                raise
            kv = await js.create_key_value(bucket=self._bucket)
            log.info("created catalog KV bucket", extra={"extra_data": {"bucket": self._bucket}})
        return kv

    async def start(self) -> None:
        """declare the bucket, warm-load the catalog from it, and hook every later reconnect.

        :return: nothing
        :rtype: None
        """
        loaded = await retry_with_backoff(self._declare_and_load, "registry.kv_catalog_load")
        if not loaded:
            log.warning(
                "registry catalog KV warm-load did not complete after retries; serving with an empty catalog "
                "that pods fill by registering, and nothing is persisted until a NATS reconnect declares %s",
                self._bucket,
            )
        self._nc.add_reconnect_callback(self._restore_after_reconnect)

    async def restore(self) -> None:
        """one attempt: declare the bucket again, then write the in-memory catalog into it.

        :return: nothing
        :rtype: None
        :raises CatalogRestoreError: when an entry could not be written
        :raises Exception: whatever the declaration raised
        """
        kv = await self.declare()
        failed = await self._catalog.restore_to_kv(kv)
        if failed:
            raise CatalogRestoreError(f"{len(failed)} catalog entries were not written to {self._bucket}: {failed}")

    async def stop(self) -> None:
        """stop a restore still retrying.

        :return: nothing
        :rtype: None
        """
        task = self._restoration
        self._restoration = None
        if task is not None and not task.done():
            task.cancel()
            # gather absorbs the task's own cancellation and still raises when THIS caller is
            # cancelled while it waits.
            await asyncio.gather(task, return_exceptions=True)

    async def _declare_and_load(self) -> None:
        """one attempt at start: declare the bucket and load the catalog from it.

        :return: nothing
        :rtype: None
        :raises Exception: whatever the declaration raised
        """
        kv = await self.declare()
        await self._catalog.load_from_kv(kv)
        log.info("catalog loaded from KV", extra={"extra_data": {"bucket": self._bucket}})

    async def _restore_after_reconnect(self) -> None:
        """the reconnect hook: start the restore in the background, replacing one still running.

        :return: nothing
        :rtype: None
        """
        previous = self._restoration
        if previous is not None and not previous.done():
            previous.cancel()
        log.info(
            "NATS reconnected: declaring the registry catalog bucket %s again and writing the catalog back",
            self._bucket,
        )
        self._restoration = asyncio.create_task(
            self._restore_until_complete(), name=f"registry-catalog-restore:{self._bucket}"
        )

    async def _restore_until_complete(self) -> None:
        """run :meth:`restore` with capped exponential backoff until one attempt succeeds.

        :return: nothing
        :rtype: None
        """
        delay = _RESTORE_FIRST_DELAY_SECONDS
        attempts = 0
        done = False
        while not done:
            attempts += 1
            try:
                await self.restore()
                done = True
            except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- logged at ERROR and retried; every registration fails until the bucket is back, so this never gives up
                log.error(
                    "restoring the registry catalog bucket %s after a NATS reconnect failed (attempt %d): %s: %s "
                    "-- registrations answer CATALOG_UNAVAILABLE until it is back; retrying in %.1fs",
                    self._bucket,
                    attempts,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RESTORE_MAX_DELAY_SECONDS)
        log.info(
            "registry catalog bucket %s declared again and the catalog written back after a NATS reconnect",
            self._bucket,
            extra={"extra_data": {"bucket": self._bucket, "attempts": attempts}},
        )
