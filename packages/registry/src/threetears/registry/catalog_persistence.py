"""owner of the registry's catalog bucket: declared at start, and again after every NATS reconnect.

the in-memory :class:`~threetears.registry.catalog.ToolCatalog` is what the registry routes from.
the catalog bucket (``tool_catalog`` by default) is its persisted copy, read only to warm-load a
starting registry -- but every registration also writes to it, and answers ``CATALOG_UNAVAILABLE``
when that write fails. a NATS restart that loses its storage -- memory storage always, and on
Kubernetes file storage too, since a restarted NATS pod can come back without its JetStream
volume -- takes the bucket under a registry that keeps running, and nothing else recreates it.

**why this is its own owner, and what it shares.** the NATS client already puts back every stream
and bucket declared through :meth:`threetears.nats.NatsClient.ensure_kv_bucket` after a reconnect.
this bucket cannot ride that, for two reasons: that path names every bucket ``{namespace}-{name}``,
and this one has always been the bare name every deployment's NATS permissions grant
(``$KV.tool_catalog.>``, ``KV_tool_catalog``), so moving it would orphan the persisted catalog and
break the grants; and a bucket the client re-creates comes back EMPTY, while this one must be
written back from the in-memory catalog. so what is left here is only what is particular to the
catalog -- the bare-name declaration and the write-back. the "retry until done, with capped
backoff" engine is the one the client's own restoration runs,
:func:`threetears.observe.resilience.retry_until_done`, with the same schedule.

**one recovery policy.** a start whose bucket is unreachable does not leave persistence off: the
start returns at once, says so at ERROR, and hands over to the same background restore a reconnect
starts, which declares the bucket, loads what an earlier registry persisted (never replacing an
entry registered since), and writes the catalog back. a reconnect's restore only declares and writes
back: a load then could re-add a tool whose deregistration failed to reach the bucket.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Final, Protocol

from threetears.nats import is_bucket_not_found
from threetears.observe import get_logger
from threetears.observe.resilience import retry_until_done

from threetears.registry.catalog import ToolCatalog

__all__ = ["CatalogBucketClient", "CatalogPersistence", "CatalogRestoreError"]

log = get_logger(__name__)

#: first pause before a restore tries again; doubles each failure up to the cap. the NATS client's
#: own restoration after a reconnect runs on the same schedule
_RESTORE_FIRST_DELAY_SECONDS: Final[float] = 0.5

#: longest pause between two attempts of a restore that keeps failing. it never gives up on its own:
#: every registration fails, or is not persisted, until the bucket is back
_RESTORE_MAX_DELAY_SECONDS: Final[float] = 30.0


class CatalogRestoreError(RuntimeError):
    """some catalog entries could not be written back into the bucket."""


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
    """owns the catalog bucket: declared and loaded at start, declared and rewritten after every reconnect.

    see the module docstring for why this is its own owner and for its one recovery policy. every
    restore runs in the background -- the reconnect path, and a registry's start, must not wait on a
    broker that may still be recovering -- retried until it lands, each failed attempt logged at
    ERROR. a later restore replaces one still running; :meth:`stop` ends it. a bucket that survived
    is bound, not created.

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
        # whether a start's load of the bucket has not landed yet, so a restore replacing the one
        # that owes it must load too (:meth:`_begin_restore`)
        self._load_owed = False

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
        """declare the bucket and warm-load the catalog from it, and hook every later reconnect.

        one attempt, in line. when it fails the registry still starts -- it serves from an in-memory
        catalog its pods fill by registering -- but never with persistence quietly off: the failure is
        logged at ERROR and the background restore takes over until the bucket is declared, loaded
        and written.

        :return: nothing
        :rtype: None
        """
        self._nc.add_reconnect_callback(self._restore_after_reconnect)
        try:
            await self._declare_and_load()
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- logged at ERROR and handed to the background restore, which retries until the bucket is back
            log.error(
                "registry catalog bucket %s could not be declared at start: %s: %s -- serving from the "
                "in-memory catalog; registrations are NOT persisted until the background restore declares "
                "the bucket, loads it and writes the catalog back",
                self._bucket,
                type(exc).__name__,
                exc,
            )
            self._begin_restore(load_first=True)

    async def restore(self, *, load_first: bool = False) -> None:
        """one attempt: declare the bucket, optionally load it, then write the in-memory catalog into it.

        :param load_first: load what the bucket holds before writing -- a start whose own load failed;
            the load never replaces an entry the catalog already holds
        :ptype load_first: bool
        :return: nothing
        :rtype: None
        :raises CatalogRestoreError: when an entry could not be written
        :raises Exception: whatever the declaration or the load raised
        """
        kv = await self.declare()
        if load_first:
            await self._catalog.load_from_kv(kv)
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
        """the start's one in-line attempt: declare the bucket and load the catalog from it.

        :return: nothing
        :rtype: None
        :raises Exception: whatever the declaration or the load raised
        """
        kv = await self.declare()
        await self._catalog.load_from_kv(kv)
        log.info("catalog loaded from KV", extra={"extra_data": {"bucket": self._bucket}})

    async def _restore_after_reconnect(self) -> None:
        """the reconnect hook: declare the bucket again and write the catalog back, in the background.

        :return: nothing
        :rtype: None
        """
        log.info(
            "NATS reconnected: declaring the registry catalog bucket %s again and writing the catalog back",
            self._bucket,
        )
        self._begin_restore(load_first=False)

    def _begin_restore(self, *, load_first: bool) -> None:
        """start a background restore, replacing one still running.

        a start's restore that has not landed yet is replaced by a reconnect's only once it has
        loaded: the replacement keeps ``load_first`` while that load is still owed.

        :param load_first: whether this restore must load the bucket before writing it
        :ptype load_first: bool
        :return: nothing
        :rtype: None
        """
        previous = self._restoration
        owed_load = load_first
        if previous is not None and not previous.done():
            previous.cancel()
            owed_load = owed_load or self._load_owed
        self._load_owed = owed_load
        self._restoration = asyncio.create_task(
            self._restore_until_complete(load_first=owed_load), name=f"registry-catalog-restore:{self._bucket}"
        )

    async def _restore_until_complete(self, *, load_first: bool) -> None:
        """run :meth:`restore` until one attempt lands, on the shared capped-backoff engine.

        :param load_first: whether the restore loads the bucket before writing it
        :ptype load_first: bool
        :return: nothing
        :rtype: None
        """
        attempts = await retry_until_done(
            lambda: self._attempt_restore(load_first=load_first),
            first_delay=_RESTORE_FIRST_DELAY_SECONDS,
            max_delay=_RESTORE_MAX_DELAY_SECONDS,
        )
        self._load_owed = False
        log.info(
            "registry catalog bucket %s declared and the catalog written back",
            self._bucket,
            extra={"extra_data": {"bucket": self._bucket, "attempts": attempts, "loaded": load_first}},
        )

    async def _attempt_restore(self, *, load_first: bool) -> bool:
        """one restore attempt, its failure logged at ERROR rather than raised.

        :param load_first: whether the restore loads the bucket before writing it
        :ptype load_first: bool
        :return: ``True`` when the restore landed
        :rtype: bool
        """
        done = False
        try:
            await self.restore(load_first=load_first)
            done = True
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- logged at ERROR and retried; registrations are not persisted until the bucket is back, so this never gives up
            log.error(
                "restoring the registry catalog bucket %s failed: %s: %s -- registrations are not persisted "
                "(or answer CATALOG_UNAVAILABLE) until it is back; retrying",
                self._bucket,
                type(exc).__name__,
                exc,
            )
        return done
