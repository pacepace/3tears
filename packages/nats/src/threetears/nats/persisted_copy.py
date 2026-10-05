"""a KV bucket that is the persisted copy of an in-memory structure, and the one owner of it.

Some services serve from memory and keep a copy in a KV bucket: the tool registry's catalog, the
hub's agent-router catalog. The copy is read once, to warm a starting process, and written on every
change. Two things go wrong with such a bucket that nothing else in the client puts right, and this
owner exists for both:

- **a start that cannot declare the bucket is otherwise never retried.**
  :meth:`threetears.nats.NatsClient.ensure_kv_bucket` remembers a declaration only once it
  succeeds, so a broker that was still coming up when the service started leaves the bucket
  undeclared for the life of the process. This owner declares in the background and retries,
  with capped backoff, until the declaration lands, logging each failed attempt at ERROR.
- **a bucket created again comes back empty.** The client puts a declared bucket back after a NATS
  restart, and a handle recreates a vanished one on its next operation; either way its entries are
  gone, and only the service holds them. This owner declares the bucket with ``on_restored`` set to
  its ``write_back``, so the client writes the copy back from memory every time that happens --
  through the live handle, which follows the client across a credential renewal or a move off a
  lame-duck server, rather than a raw nats-py handle left bound to a retired connection.

The bucket is declared under its exact name (``prefix_namespace=False``): these buckets have always
carried a bare name, and every deployment's grants name it so.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING

from threetears.observe import get_logger
from threetears.observe.resilience import retry_until_done

if TYPE_CHECKING:
    from threetears.nats.kv import KvBucketLike, KvDeclaring

__all__ = ["PersistedCopyBucket"]

log = get_logger(__name__)


class PersistedCopyBucket:
    """owns one KV bucket that is the persisted copy of an in-memory structure.

    :meth:`start` begins declaring the bucket in the background; the first declaration that lands
    runs ``load`` (bucket into memory) ONCE, then ``write_back`` (memory into bucket), and from then
    on the client runs ``write_back`` whenever it creates the bucket again, empty. :attr:`bucket` is
    the live handle once declared, and ``None`` before. :meth:`stop` ends a declaration still
    retrying.

    ``load`` must never replace an entry memory already holds: it runs after the service has
    started serving, and an entry registered meanwhile is newer than the copy. It runs only at the
    first declaration and never on a refill, because a refill after a restart must not re-add an
    entry whose removal never reached the bucket. ``write_back`` must write whatever memory holds
    when it runs; it is also run once after the first ``load``, so changes made before the bucket
    was declared reach it.

    :param client: the NATS client that declares the bucket
    :ptype client: KvDeclaring
    :param bucket: the bucket's exact name
    :ptype bucket: str
    :param load: reads the bucket into memory, never replacing an entry memory already holds
    :ptype load: Callable[[KvBucketLike], Awaitable[None]]
    :param write_back: writes memory into the bucket
    :ptype write_back: Callable[[KvBucketLike], Awaitable[None]]
    :param storage: ``"file"`` or ``"memory"``
    :ptype storage: str
    :param ttl: per-entry expiry; ``None`` for none
    :ptype ttl: timedelta | None
    :param history: revisions kept per key
    :ptype history: int
    :param direct: the ``allow_direct`` value the bucket must carry
    :ptype direct: bool
    :param retry_first_delay: the pause after the first failed declaration; it doubles each failure
    :ptype retry_first_delay: timedelta
    :param retry_max_delay: the longest pause between two declarations
    :ptype retry_max_delay: timedelta
    """

    def __init__(
        self,
        *,
        client: KvDeclaring,
        bucket: str,
        load: Callable[[KvBucketLike], Awaitable[None]],
        write_back: Callable[[KvBucketLike], Awaitable[None]],
        storage: str = "file",
        ttl: timedelta | None = None,
        history: int = 1,
        direct: bool = True,
        retry_first_delay: timedelta = timedelta(seconds=0.5),
        retry_max_delay: timedelta = timedelta(seconds=30),
    ) -> None:
        """bind the client, the bucket's name and shape, and the two callbacks; no I/O.

        :param client: the NATS client that declares the bucket
        :ptype client: KvDeclaring
        :param bucket: the bucket's exact name
        :ptype bucket: str
        :param load: reads the bucket into memory, never replacing an entry memory already holds
        :ptype load: Callable[[KvBucketLike], Awaitable[None]]
        :param write_back: writes memory into the bucket
        :ptype write_back: Callable[[KvBucketLike], Awaitable[None]]
        :param storage: ``"file"`` or ``"memory"``
        :ptype storage: str
        :param ttl: per-entry expiry; ``None`` for none
        :ptype ttl: timedelta | None
        :param history: revisions kept per key
        :ptype history: int
        :param direct: the ``allow_direct`` value the bucket must carry
        :ptype direct: bool
        :param retry_first_delay: the pause after the first failed declaration; it doubles each failure
        :ptype retry_first_delay: timedelta
        :param retry_max_delay: the longest pause between two declarations
        :ptype retry_max_delay: timedelta
        :return: nothing
        :rtype: None
        :raises ValueError: when the retry schedule is not ``0 < retry_first_delay <= retry_max_delay``
        """
        if retry_first_delay <= timedelta(0) or retry_max_delay < retry_first_delay:
            raise ValueError(
                f"PersistedCopyBucket {bucket!r} needs 0 < retry_first_delay <= retry_max_delay, got "
                f"{retry_first_delay} and {retry_max_delay}"
            )
        self._client = client
        self._name = bucket
        self._load = load
        self._write_back = write_back
        self._storage = storage
        self._ttl = ttl
        self._history = history
        self._direct = direct
        self._retry_first_delay = retry_first_delay
        self._retry_max_delay = retry_max_delay
        # the live handle, from the latest declaration that landed; None until one has.
        self._bucket: KvBucketLike | None = None
        # whether ``load`` has run to completion; it runs once, at the first declaration that lands.
        self._loaded = False
        # the background declaration :meth:`start` began, held so it is not collected mid-flight and
        # so :meth:`stop` can end it.
        self._declaring: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        """the bucket's exact name.

        :return: the bucket name
        :rtype: str
        """
        return self._name

    @property
    def bucket(self) -> KvBucketLike | None:
        """the live handle on the bucket, once a declaration has landed.

        :return: the handle, or ``None`` while the bucket is not yet declared
        :rtype: KvBucketLike | None
        """
        return self._bucket

    async def start(self) -> None:
        """begin declaring the bucket in the background, and return at once.

        The service starts serving from memory whether or not the broker answers; each failed
        declaration is logged at ERROR, and the declaration is retried until it lands.

        :return: nothing
        :rtype: None
        :raises RuntimeError: when it was already started
        """
        if self._declaring is not None:
            raise RuntimeError(f"PersistedCopyBucket {self._name!r} was already started")
        self._declaring = asyncio.create_task(self._declare_until_done(), name=f"persisted-copy-declare:{self._name}")

    async def stop(self) -> None:
        """end a declaration still retrying, and wait for it to end.

        A refill the client still owes the bucket is the client's to stop, at its own shutdown.

        :return: nothing
        :rtype: None
        """
        task = self._declaring
        if task is not None and not task.done():
            task.cancel()
            # gather absorbs the task's own cancellation and still raises when THIS caller is
            # cancelled while it waits.
            await asyncio.gather(task, return_exceptions=True)

    async def _declare_until_done(self) -> None:
        """declare, load and write back, retrying with capped backoff until one attempt lands.

        :return: nothing
        :rtype: None
        """
        attempts = await retry_until_done(
            self._declare_once,
            first_delay=self._retry_first_delay.total_seconds(),
            max_delay=self._retry_max_delay.total_seconds(),
        )
        log.info(
            "persisted-copy KV bucket %s declared, loaded and written back",
            self._name,
            extra={"extra_data": {"bucket": self._name, "attempts": attempts}},
        )

    async def _declare_once(self) -> bool:
        """one attempt: declare the bucket, load it the first time, and write memory back into it.

        :return: ``True`` when the attempt landed
        :rtype: bool
        """
        landed = False
        try:
            bucket = await self._client.ensure_kv_bucket(
                name=self._name,
                ttl=self._ttl,
                storage=self._storage,
                history=self._history,
                direct=self._direct,
                prefix_namespace=False,
                on_restored=self._write_back,
            )
            # exposed before the load: a change made from now on writes straight to the bucket,
            # and the load never replaces an entry memory holds.
            self._bucket = bucket
            if not self._loaded:
                await self._load(bucket)
                self._loaded = True
            await self._write_back(bucket)
            landed = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- logged at ERROR and retried until the bucket is declared
            log.error(
                "persisted-copy KV bucket %s could not be declared, loaded and written back: %s: %s -- the "
                "service serves from memory and changes are not persisted until it is; retrying",
                self._name,
                type(exc).__name__,
                exc,
                extra={"extra_data": {"bucket": self._name, "loaded": self._loaded, "error": str(exc)}},
            )
        return landed
