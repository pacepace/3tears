"""epoch client -- atomic counter bump plus best-effort NATS broadcast.

:class:`EpochClient` is the publish-side companion to
:class:`~threetears.epoch.listener.EpochListener`. it owns one pair of
operations over whichever substrate the subject belongs to:

- :meth:`current` -- read the latest epoch for a subject (used by
  listeners on cold start and by periodic catch-up ticks)
- :meth:`bump` -- atomically increment the epoch for a subject, then
  publish an :class:`~threetears.epoch.wire.EpochBumpMessage` on the
  same subject so sibling pods notice immediately

**two substrates, routed by what the number means.** an epoch is a
coherence signal, not a durable fact, so the counter for one lives in a
memory-backed NATS KV bucket (via
:class:`~threetears.core.coordination.distributed_counter.DistributedCounter`)
and resets with the broker -- which is harmless, because a restart that
loses the counter also loses every cache it was sequencing.

the exception is an epoch whose VALUE escapes this cluster. a tile
epoch is the ``v{n}`` in a tile URL and reaches browser and CDN caches
nothing here can reach, so re-issuing ``v1..vN`` for different content
would serve the old generation from an address that looks current. that
family keeps its durable ``config_epochs`` row, keyed on the subject
path.

nats is the fast notify either way. a missed broadcast is recovered by
the next periodic tick (calls :meth:`current`) or by the next response
that echoes a higher epoch. the publisher always knows its own current
epoch (the counter returns it); subscribers learn it from broadcasts and
echoes. this is the etcd ``mod_revision`` shape minus the multi-key
transaction support: every domain is independent.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Final, Protocol

from threetears.core.coordination.distributed_counter import DistributedCounter
from threetears.nats import NatsClient
from threetears.nats.kv import KvBucketLike
from threetears.nats.errors import PublishError
from threetears.nats.subjects import Subject
from uuid_utils import uuid7
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
#: the stream's ``max_age``, so any value here would expire the counters -- and
#: the bucket-identity key that detects a recreated bucket -- on a timer,
#: turning a reset from an event into a scheduled fleet-wide cache flush.
_EPOCH_BUCKET: Final = "epochs"

#: Reserved key holding the epoch bucket's opaque identity.
#:
#: A leading underscore keeps it out of the subject-path keyspace: every
#: ``Subjects`` builder prefixes the namespace, so no epoch key can begin with
#: one. The value is a ``uuid7`` string, compared for EQUALITY only -- it says
#: "this is a different bucket", never "this is a later one".
_IDENTITY_KEY: Final = "_bucket_identity"

#: How many times to retry the create/read pair before giving up on learning an
#: identity. The pair is not atomic: a bucket wiped between the two leaves the
#: read empty. Bounded because a broker recreating the bucket faster than two
#: round trips is not a race to win, it is an outage.
_IDENTITY_ATTEMPTS: Final = 3

#: Subject families whose epoch value ESCAPES this cluster and therefore cannot
#: live on a counter that resets.
#:
#: **There is no durable epoch in NATS, and that is a decision, not a gap.**
#: ``NatsKvBucket`` does accept ``storage="file"``, so a file-backed counter is
#: constructible and would survive a broker process restart. It is not used
#: here because file-backed JetStream survives only if the store DIRECTORY
#: survives, and the failure this whole design answers is a broker on ephemeral
#: storage whose restart wipes JetStream wholesale -- on that deployment
#: ``file`` is exactly as durable as ``memory``. That makes NATS durability
#: CONDITIONAL on how someone provisioned a volume. For a value whose only
#: consumer is a CDN we cannot purge, conditional is the wrong guarantee, so
#: the one family that needs it keeps an unconditional Postgres row.
#:
#: **Durability is a property of the SUBJECT, declared here, per family.** Not
#: a per-call flag: two call sites bumping the same subject could then disagree,
#: and a caller can forget one.
#:
#: The DECLARATION is the table; the marker beside each entry is only how the
#: table is applied to a concrete path, and :func:`_is_durable` still matches on
#: it. That matcher is unchanged in behaviour from the bare ``.tiles.`` constant
#: it replaced, and on its own it would classify a subject nobody had thought
#: about exactly as before. What closes that is not the matcher but the
#: enumeration: every epoch subject must appear in this table or in
#: :data:`_EPHEMERAL_FAMILIES` below;
#: ``packages/epoch/tests/unit/test_durability_policy.py`` enumerates the real
#: ``Subjects`` factory and fails when a new ``*_epoch`` builder matches
#: neither, so adding one forces the decision instead of defaulting to
#: ephemeral -- which is the direction that cannot be repaired.
#: Each entry is ``(builder name, path marker, reason)``. The reason is a FIELD
#: rather than a comment so the policy test can require one; a rationale the
#: enforcement cannot read is a rationale nobody has to write.
_DURABLE_FAMILIES: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "datasource_tile_epoch",
        ".tiles.",
        "the value is the v{n} in a tile URL and the geo collection puts it in its cache key, "
        "so it reaches browser and CDN caches this system cannot purge; a counter that reset "
        "would re-issue v1..vN for DIFFERENT content at a version those caches still hold",
    ),
)

#: Epoch families deliberately left on the ephemeral counter, with the reason.
#:
#: Listed rather than left implicit so the enumeration test can tell "decided
#: ephemeral" from "nobody looked". Every one of these resolves a reset by
#: reloading from a lower tier that IS the source of truth, so a replaced
#: counter costs one extra reload and fails safe.
_EPHEMERAL_FAMILIES: Final[tuple[tuple[str, str], ...]] = (
    ("capabilities_epoch", "pods reload the capabilities registry from its Postgres row"),
    ("gateway_catalog_epoch", "pods re-run _load_catalog from the gateway tables"),
    ("mcp_rbac_epoch", "pods reload the RBAC view from mcp_tool_grants"),
    ("identity_epoch", "pods drop cached principal status and re-read it"),
)


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
    if not subject.path.endswith(".epoch"):
        return False
    return any(marker in subject.path for _name, marker, _why in _DURABLE_FAMILIES)


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
    handlers concurrently -- an ephemeral bump serializes through the
    counter's CAS loop, a durable one on its row lock. the client never caches the last-seen epoch -- it always
    round-trips the counter on :meth:`bump` because the value it
    returns is the only guaranteed-monotonic answer available to a
    single writer in a multi-writer system.

    :param pool: asyncpg-compatible pool exposing ``fetchrow`` /
        ``fetchval``; production passes :class:`asyncpg.Pool`. Used ONLY by
        the durable subject family -- every other epoch counts in NATS KV
        and never touches it.
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

    async def bucket_identity(self) -> str | None:
        """the epoch bucket's opaque identity, minting one if it has none.

        **This is what tells a live pod that its counter was replaced rather
        than merely rewound.** The ephemeral counters live in a memory-backed
        bucket, so a broker restart recreates it empty and every operation then
        SUCCEEDS while reading zero -- there is no error to catch, no gap in
        the sequence, and nothing else in the system that changes.

        Every caller attempts the create unconditionally; the return
        distinguishes winner from loser, so no branch on "did I make this
        bucket" is needed -- and that branch would be wrong anyway, since
        ``STREAM.CREATE`` succeeds for a bucket that already exists with the
        same config.

        Compared for EQUALITY only, never for order. A changed identity says
        the old numbers are meaningless, which is more useful than knowing they
        were merely lower, and it needs no clock: seeding a replacement counter
        ABOVE the old one cannot be done without remembering what the old one
        reached, which is the durability this design removes.

        :return: the identity, or ``None`` when it could not be established
            within :data:`_IDENTITY_ATTEMPTS` (a bucket being recreated faster
            than two round trips is an outage, not a race to win)
        :rtype: str | None
        """
        try:
            bucket = await self._nats.kv_bucket(name=_EPOCH_BUCKET, ttl=None)
            return await self._identity_from(bucket)
        # prawduct:allow prawduct/broad-except -- an unreachable broker is an
        # outage, not a replaced counter. raising here would fail a catch-up
        # that the durable path does not even need KV for, and reporting a
        # changed identity would flush every cache in the fleet on a blip.
        except Exception:  # noqa: BLE001
            log.warning("epoch bucket identity unavailable; not treating as a replacement", exc_info=True)
            return None

    async def _identity_from(self, bucket: KvBucketLike) -> str | None:
        """read or mint the identity on an already-open bucket.

        :param bucket: the opened epoch KV bucket
        :ptype bucket: Any
        :return: the identity, or ``None`` when it could not be established
        :rtype: str | None
        """
        for _ in range(_IDENTITY_ATTEMPTS):
            minted = str(uuid7())  # convert at border: opaque KV value, never parsed back
            if await bucket.create(key=_IDENTITY_KEY, value=minted.encode()) is not None:
                return minted
            existing = await bucket.get(key=_IDENTITY_KEY)
            if existing is not None:
                decoded: str = existing.decode()
                return decoded
            # lost the create AND read nothing: the bucket was wiped between
            # the two. try again rather than reporting an identity we do not
            # have -- reporting one would be indistinguishable from stability.
        log.warning("could not establish epoch bucket identity", extra={"extra_data": {"attempts": _IDENTITY_ATTEMPTS}})
        return None

    @traced
    async def current(self, subject: Subject) -> int:
        """read the latest epoch recorded for a subject.

        used by :class:`~threetears.epoch.listener.EpochListener` on
        cold start to prime its last-seen, and by periodic catch-up
        ticks. returns ``0`` when the subject has never been bumped --
        both substrates count per subject from zero, so the first
        :meth:`bump` returns ``1`` and a returned ``0`` here means
        "nobody has bumped this domain".

        Reads the KV counter for an ephemeral subject and the durable
        row for the tile family. A wildcard path short-circuits to ``0``
        without touching either.

        :param subject: target subject; its ``path`` keys the
            counter (digested when outside the KV key grammar), or is
            the row PK on the durable path
        :ptype subject: Subject
        :return: latest epoch, or ``0`` when nothing has bumped this subject
            (an absent KV key, or an absent row on the durable path)
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

        an ephemeral subject counts through a CAS loop in NATS KV; the
        durable tile family serializes on its Postgres row lock. Either
        way concurrent bumps from different writers each receive a
        unique strictly-increasing epoch. broadcast is best-effort:
        :class:`~threetears.nats.errors.PublishError` is logged and
        swallowed because the counter increment already happened and
        any subscriber that missed the broadcast catches up via
        :meth:`current` on the next periodic tick or via a per-
        message epoch echo.

        callers MUST invoke after the mutation that motivates the bump
        has committed. bumping inside an open transaction broadcasts a
        phantom epoch if the transaction rolls back.

        :param subject: target subject; its ``path`` keys the
            counter (digested when outside the KV key grammar), or is
            the row PK on the durable path and the broadcast subject
        :ptype subject: Subject
        :param payload: opaque hint forwarded to subscribers in the
            broadcast envelope; framework never inspects
        :ptype payload: dict[str, Any] | None
        :return: the new epoch the counter reports
        :rtype: int
        :raises RuntimeError: if the DURABLE path's upsert returns no
            row (should never happen on a healthy database -- the
            ``RETURNING`` clause is unconditional). The ephemeral path
            raises :class:`~threetears.nats.errors.KvError` instead.
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
                "epoch bump broadcast failed; the counter increment already happened, "
                "subscribers catch up via the next catch-up pass or a per-message echo",
                extra={
                    "extra_data": {
                        "subject": subject.path,
                        "epoch": new_epoch,
                        "error": str(exc),
                    },
                },
            )

        return new_epoch
