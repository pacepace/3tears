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

import hashlib
import json
import re
from typing import Any, Final, Protocol

from threetears.core.coordination.distributed_counter import DistributedCounter
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


#: KV bucket holding every EPHEMERAL epoch counter.
#:
#: ``ttl=None`` deliberately. ``NatsKvBucket``'s ttl is per-BUCKET and becomes
#: the stream's ``max_age``, so any value here would expire the counters (and,
#: once chunk 03 lands, the bucket-identity key) on a timer -- turning a reset
#: from an event into a scheduled fleet-wide cache flush.
_EPOCH_BUCKET: Final = "epochs"

#: Subject families whose epoch value ESCAPES this cluster and therefore cannot
#: live on a counter that resets.
#:
#: ``datasource_tile_epoch``'s value is the ``v{n}`` segment of a tile URL, and
#: the geo collection puts that version in its cache key, so the number reaches
#: browser and CDN caches this system cannot reach. A memory-backed counter
#: resets on a broker restart and would re-issue ``v1..vN`` for DIFFERENT
#: content while those edge caches still hold the old generation keyed on the
#: same version. No amount of in-process detection fixes a stale CDN.
#:
#: Matched on the subject's shape rather than declared per call site: durability
#: is a property of what the number MEANS, not of who happens to bump it, and a
#: per-call flag is one a caller can forget.
_DURABLE_SUBJECT_MARKER: Final = ".tiles."


#: The key grammar ``nats-server`` enforces on a KV key. A subject path is
#: usually already legal (dots are permitted), but a path segment can carry a
#: caller-supplied value, and one space turns a working bump into an
#: ``InvalidKeyError`` raised in production rather than in review.
_KV_KEY_GRAMMAR: Final = re.compile(r"^[-/_=.a-zA-Z0-9]+$")


def _key_for(subject: Subject) -> str:
    """derive this subject's KV counter key.

    The path verbatim where it is already a legal key, because a readable key
    is worth having when someone is staring at a bucket wondering which counter
    is which. A path that is not legal is digested rather than rejected: the
    caller cannot fix a subject built from a user-supplied layer name, and a
    hard failure at ``bump`` would surface as an outage rather than a rename.
    The digest is deterministic, so every pod derives the same key.

    :param subject: the epoch subject
    :ptype subject: Subject
    :return: a key that satisfies the KV grammar
    :rtype: str
    """
    if _KV_KEY_GRAMMAR.match(subject.path):
        return subject.path
    return f"digest.{hashlib.sha256(subject.path.encode()).hexdigest()}"


def _is_durable(subject: Subject) -> bool:
    """whether ``subject``'s epoch must survive a broker restart.

    :param subject: the epoch subject being bumped or read
    :ptype subject: Subject
    :return: ``True`` when the value escapes the cluster and must stay durable
    :rtype: bool
    """
    return _DURABLE_SUBJECT_MARKER in subject.path and subject.path.endswith(".epoch")


def _is_wildcard(subject: Subject) -> bool:
    """whether ``subject`` is a pattern rather than a concrete path.

    KV keys admit only ``[-/_=.a-zA-Z0-9]``, so ``*`` and ``>`` are illegal and
    ``get`` on one raises rather than missing. The listener's wildcard-priming
    path depends on :meth:`EpochClient.current` answering ``0`` for a pattern,
    which under Postgres happened for free (no row matched) and under KV has to
    be said.

    :param subject: the epoch subject
    :ptype subject: Subject
    :return: ``True`` when the path carries a NATS wildcard token
    :rtype: bool
    """
    return "*" in subject.path or subject.path.endswith(">")


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
        self._counter = DistributedCounter(nats_client, bucket_name=_EPOCH_BUCKET, ttl=None)

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
        if _is_wildcard(subject):
            # a pattern names no single counter. answering 0 without a lookup
            # is what the listener's wildcard priming expects, and asking KV
            # would raise on an illegal key rather than miss.
            return 0
        if _is_durable(subject):
            value = await self._pool.fetchval(_CURRENT_SQL, subject.path)
            return 0 if value is None else int(value)
        return await self._counter.get(_key_for(subject))

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
        if _is_durable(subject):
            payload_json = json.dumps(payload) if payload is not None else None
            row = await self._pool.fetchrow(_BUMP_SQL, subject.path, payload_json)
            if row is None:
                raise RuntimeError(
                    f"config_epochs upsert returned no row for subject={subject.path!r}",
                )
            new_epoch = int(row["epoch"])
        else:
            new_epoch = await self._counter.increment(_key_for(subject))

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
