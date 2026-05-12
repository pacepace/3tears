"""epoch client -- atomic Postgres bump plus best-effort NATS broadcast.

:class:`EpochClient` is the publish-side companion to
:class:`~threetears.epoch.listener.EpochListener`. it owns one pair of
operations against the ``config_epochs`` table:

- :meth:`current` -- read the latest epoch for a subject (used by
  listeners on cold start and by periodic catch-up ticks)
- :meth:`bump` -- atomically increment the epoch for a subject, then
  publish an :class:`~threetears.epoch.wire.EpochBumpMessage` on the
  same subject so sibling pods notice immediately

postgres is the source of truth. nats is the fast notify. a missed
broadcast is recovered by either the next periodic tick (calls
:meth:`current`) or by the next response that echoes a higher epoch
(per-message echo, consumer-side).

the row PK in ``config_epochs`` is the subject path string. the
publisher always knows its own current epoch (just returned by the
``RETURNING epoch`` clause); subscribers learn it from broadcasts and
from echoes. this is the etcd ``mod_revision`` shape minus the
multi-key transaction support: every domain is independent.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from threetears.nats import NatsClient
from threetears.nats.errors import PublishError
from threetears.nats.subjects import Subject
from threetears.observe import get_logger, traced

from threetears.epoch.wire import EpochBumpMessage

__all__ = [
    "EpochClient",
    "PoolLike",
]

log = get_logger(__name__)


class PoolLike(Protocol):
    """minimal pool surface :class:`EpochClient` needs.

    matches the top-level ``fetchrow``/``fetchval`` methods that
    :class:`asyncpg.Pool` exposes (the pool acquires + releases a
    connection internally). callers pass an :class:`asyncpg.Pool`
    directly in production; tests pass a duck-typed mock.

    typed as a :class:`Protocol` so the package does not take a
    runtime dependency on asyncpg.
    """

    async def fetchrow(self, query: str, *args: object) -> Any:
        """run query and return the first row (or None if empty).

        :param query: parameterized SQL with ``$N`` placeholders
        :ptype query: str
        :param args: positional parameter values
        :ptype args: object
        :return: row record or None
        :rtype: Any
        """
        ...

    async def fetchval(self, query: str, *args: object) -> Any:
        """run query and return the first column of the first row.

        :param query: parameterized SQL with ``$N`` placeholders
        :ptype query: str
        :param args: positional parameter values
        :ptype args: object
        :return: scalar value or None
        :rtype: Any
        """
        ...


_BUMP_SQL = (
    "INSERT INTO config_epochs (subject_path, epoch, payload) "
    "VALUES ($1, 1, $2::jsonb) "
    "ON CONFLICT (subject_path) DO UPDATE SET "
    "epoch = config_epochs.epoch + 1, "
    "payload = EXCLUDED.payload, "
    "date_updated = now() "
    "RETURNING epoch"
)

_CURRENT_SQL = "SELECT epoch FROM config_epochs WHERE subject_path = $1"


class EpochClient:
    """publish-side client for cross-pod config-epoch coherence.

    one instance per process; safe to call from multiple admin
    handlers concurrently (the bump statement serializes on the row
    lock). the client never caches the last-seen epoch -- it always
    round-trips Postgres on :meth:`bump` because the
    ``RETURNING`` value is the only guaranteed-monotonic answer
    available to a single writer in a multi-writer system.

    :param pool: asyncpg-compatible pool exposing ``fetchrow`` /
        ``fetchval``; production passes :class:`asyncpg.Pool`
    :ptype pool: PoolLike
    :param nats_client: connected typed NATS wrapper for broadcast
    :ptype nats_client: NatsClient
    """

    def __init__(self, pool: PoolLike, nats_client: NatsClient) -> None:
        """capture pool + nats client; no I/O.

        :param pool: postgres pool implementing :class:`PoolLike`
        :ptype pool: PoolLike
        :param nats_client: connected NatsClient
        :ptype nats_client: NatsClient
        :return: nothing
        :rtype: None
        """
        self._pool = pool
        self._nats = nats_client

    @traced
    async def current(self, subject: Subject) -> int:
        """read the latest epoch recorded for a subject.

        used by :class:`~threetears.epoch.listener.EpochListener` on
        cold start to prime its last-seen, and by periodic catch-up
        ticks. returns ``0`` when no row exists yet -- the bump-side
        ``ON CONFLICT`` clause guarantees the first successful
        :meth:`bump` returns ``1``, so a returned ``0`` here means
        "nobody has bumped this domain in this database."

        :param subject: target subject; the subject's ``path`` is
            the row PK
        :ptype subject: Subject
        :return: latest epoch, or ``0`` if no row exists
        :rtype: int
        """
        value = await self._pool.fetchval(_CURRENT_SQL, subject.path)
        if value is None:
            result = 0
        else:
            result = int(value)
        return result

    @traced
    async def bump(
        self,
        subject: Subject,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """atomically increment the epoch for a subject, then broadcast.

        the upsert is serialized on the row lock; concurrent bumps
        from different writers wait briefly and each receive a
        unique strictly-increasing epoch. broadcast is best-effort:
        :class:`~threetears.nats.errors.PublishError` is logged and
        swallowed because the row commit is the source of truth and
        any subscriber that missed the broadcast catches up via
        :meth:`current` on the next periodic tick or via a per-
        message epoch echo.

        callers MUST invoke after the row mutation that motivates
        the bump has committed. bumping inside an open transaction
        broadcasts a phantom epoch if the transaction rolls back.

        :param subject: target subject; the subject's ``path`` is
            the row PK and the broadcast subject
        :ptype subject: Subject
        :param payload: opaque hint forwarded to subscribers in the
            broadcast envelope; framework never inspects
        :ptype payload: dict[str, Any] | None
        :return: the new epoch returned by ``RETURNING``
        :rtype: int
        :raises RuntimeError: if the upsert returns no row (should
            never happen on a healthy database -- the ``RETURNING``
            clause is unconditional)
        """
        # asyncpg does not auto-encode dict to jsonb without a per-pool
        # type codec; serialize at the call site so callers do not have
        # to register codecs to use this client. the ``$2::jsonb`` cast
        # in the SQL parses the resulting text back to jsonb.
        payload_json = json.dumps(payload) if payload is not None else None
        row = await self._pool.fetchrow(_BUMP_SQL, subject.path, payload_json)
        if row is None:
            raise RuntimeError(
                f"config_epochs upsert returned no row for subject={subject.path!r}",
            )
        new_epoch = int(row["epoch"])

        message = EpochBumpMessage(
            subject_path=subject.path,
            epoch=new_epoch,
            payload=payload,
        )
        try:
            await self._nats.publish(subject=subject, message=message)
        except PublishError as exc:
            log.warning(
                "epoch bump broadcast failed; row commit is durable, "
                "subscribers will catch up via current() or per-message echo",
                extra={
                    "extra_data": {
                        "subject": subject.path,
                        "epoch": new_epoch,
                        "error": str(exc),
                    },
                },
            )

        return new_epoch
