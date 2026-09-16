"""durable denylists: standing revocations, and single-use redemptions.

Both answer "has this been revoked or spent", both must survive a broker restart, and both are
read far more often than they are written -- so both live in L3 through a collection, with L1 and
L2 in front so the common answer costs no database query.

:class:`RevocationGuard` records the moment a key is revoked FROM. A revoked ``sub`` (principal)
or ``customer_id`` must block every session that STARTED BEFORE the revocation while leaving a
session that starts after it alone -- otherwise every future login to a once-revoked principal is
permanently denylisted, which is not what recovery or offboarding mean. So the entry carries a
value and the check is a comparison::

    guard = RevocationGuard(registry, purpose="standing", ttl_seconds=...)
    await guard.record_revocation("sub:<principal_id>", revoked_at=cutoff)
    blocked = await guard.is_revoked_before("sub:<principal_id>", moment=session_started_at)

:class:`RedemptionLedger` answers the other shape: has this exact artifact been spent. It is what
a refresh-token ``jti`` ledger needs -- one sighting is legitimate, a second is reuse -- and it is
NOT a :class:`~threetears.core.coordination.replay_guard.ReplayGuard`. A replay guard remembers a
nonce for the seconds its artifact is acceptable and fails closed on a wipe by refusing anything
issued before its bucket existed; a ledger remembers for the artifact's whole life, which for a
refresh token is a month, and refusing everything issued before the last broker restart would deny
every outstanding token::

    ledger = RedemptionLedger(registry, purpose="refresh_jti", ttl_seconds=30 * 86400)
    if not await ledger.record_unique(jti):
        raise <reuse detected>

**Both write L3 synchronously.** Losing a revocation makes a revoked session valid again, and
losing a redemption makes a spent token spendable; neither is worth a flush interval's exposure.

**What "fail closed" means now that L3 holds the truth.** While these lived in a KV bucket, L2 was
the only copy, so any transport failure had to deny. It no longer is:

- a **revocation read or write** survives an L2 outage. The write commits to L3 and the read falls
  through to it, so the caller gets the right answer rather than a refusal. An **L3** failure
  propagates, because then there is no answer, and "no answer" must never read as "not revoked".
- a **redemption** propagates an L2 failure too, because there the compare-and-swap IS the fence:
  without L2 two callers could both be told they were first, which is the one thing a single-use
  ledger may not do.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.coordination.tables import (
    CoordinationRedemptionsCollection,
    CoordinationRevocationsCollection,
    coordination_collection,
)
from threetears.observe import get_logger

__all__ = ["RedemptionLedger", "RevocationGuard"]

log = get_logger(__name__)


def _hashed(key: str) -> str:
    """the stored form of a denylist key.

    Hashed so the raw identifier -- a principal id, a customer id, a token id -- is never a stored
    key, and so any key shape fits one column.

    :param key: the caller's key
    :ptype key: str
    :return: its hex digest
    :rtype: str
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class RevocationGuard:
    """timestamped revocation entries: "denylist everything that started before this moment"."""

    def __init__(
        self,
        registry: CollectionRegistry,
        *,
        purpose: str,
        ttl_seconds: int,
        config: CoreConfig | None = None,
    ) -> None:
        """configure the guard over its registry's coordination tables.

        :param registry: the collection registry this guard reads and writes through. L3 is what
            makes a revocation survive a restart, and the absence of a revocation -- the common
            answer -- is served from L1 or L2 without a database query
        :ptype registry: CollectionRegistry
        :param purpose: which denylist these keys belong to, carried in the row key so two
            denylists never collide. This is what the KV bucket name used to be
        :ptype purpose: str
        :param ttl_seconds: how long a recorded revocation is remembered. MUST be at least as long
            as the longest session lifetime it has to outlive: an entry that expires while a
            session it denylists is still running fails OPEN. MUST be positive
        :ptype ttl_seconds: int
        :param config: core config forwarded when this process first builds the collection
        :ptype config: CoreConfig | None
        :raises ValueError: when ``ttl_seconds`` is not positive, or ``purpose`` is empty
        """
        if ttl_seconds <= 0:
            raise ValueError(f"RevocationGuard ttl_seconds must be positive, got {ttl_seconds}")
        if not purpose.strip():
            raise ValueError("RevocationGuard purpose must be a non-empty name, e.g. 'standing'")
        self._purpose = purpose
        self._ttl = timedelta(seconds=ttl_seconds)
        self._collection = coordination_collection(registry, CoordinationRevocationsCollection, config)

    @property
    def purpose(self) -> str:
        """which denylist this guard's keys belong to.

        :return: the purpose
        :rtype: str
        """
        return self._purpose

    async def record_revocation(self, key: str, *, revoked_at: datetime) -> None:
        """record (or overwrite) the revoked-from moment for ``key``.

        Unconditional, not create-if-absent: a second call for the same key -- an operator
        re-running offboarding, or narrowing an earlier cutoff -- replaces the effective moment
        rather than being refused as a duplicate.

        :param key: the denylist key, e.g. ``f"sub:{principal_id}"``. Hashed before storage
        :ptype key: str
        :param revoked_at: the moment the key is revoked from. MUST be timezone-aware: a naive
            value cannot be compared against a caller's moment from another clock assumption
        :ptype revoked_at: datetime
        :return: nothing
        :rtype: None
        :raises ValueError: when ``revoked_at`` is timezone-naive
        :raises threetears.nats.KvError: on an L2 failure -- the caller MUST deny
        """
        if revoked_at.tzinfo is None:
            raise ValueError("RevocationGuard.record_revocation requires a timezone-aware revoked_at")
        row_id = self._row_id(key)
        existing = await self._collection.get(row_id)
        row = {
            "purpose": self._purpose,
            "key": row_id[1],
            "revoked_at": revoked_at,
            # measured from the revocation, not from the write: re-recording a revocation must not
            # extend how long it is remembered past the sessions it exists to block.
            "expires_at": revoked_at + self._ttl,
        }
        if existing is None:
            await self._collection.save_entity(self._collection.create(row))
            return
        existing.set_data({**existing.to_dict(), **row})
        await self._collection.save_entity(existing)

    async def revoked_at(self, key: str) -> datetime | None:
        """the recorded revocation moment for ``key``, or ``None`` when never revoked.

        :param key: the denylist key to look up
        :ptype key: str
        :return: the recorded moment, or ``None``
        :rtype: datetime | None
        :raises threetears.nats.KvError: on an L2 failure -- the caller MUST deny
        """
        entity = await self._collection.get(self._row_id(key))
        if entity is None:
            return None
        stored: datetime = entity.to_dict()["revoked_at"]
        return stored

    async def is_revoked_before(self, key: str, *, moment: datetime) -> bool:
        """whether ``key``'s revocation denylists something that started at ``moment``.

        ``True`` iff a revocation is recorded AND ``moment < revoked_at``: something that started
        before the revocation is denylisted, something that starts at or after it is unaffected.
        Returns ``False`` for a key with no entry -- the legitimate "not revoked" case, not a
        failure. A storage failure still propagates, and the caller MUST treat that as deny.

        :param key: the denylist key to check
        :ptype key: str
        :param moment: the comparison timestamp, e.g. a session's start. MUST be timezone-aware
        :ptype moment: datetime
        :return: whether the key is revoked as of ``moment``
        :rtype: bool
        :raises ValueError: when ``moment`` is timezone-naive
        :raises threetears.nats.KvError: on an L2 failure -- the caller MUST deny
        """
        if moment.tzinfo is None:
            raise ValueError("RevocationGuard.is_revoked_before requires a timezone-aware moment")
        stored = await self.revoked_at(key)
        if stored is None:
            return False
        return moment < stored

    def _row_id(self, key: str) -> tuple[str, str]:
        """the row this key addresses.

        :param key: the caller's denylist key
        :ptype key: str
        :return: ``(purpose, hashed key)`` in declared column order
        :rtype: tuple[str, str]
        """
        return (self._purpose, _hashed(key))


class RedemptionLedger:
    """a durable single-use ledger: the first sighting of a key is fresh, every later one is reuse."""

    def __init__(
        self,
        registry: CollectionRegistry,
        *,
        purpose: str,
        ttl_seconds: int,
        config: CoreConfig | None = None,
    ) -> None:
        """configure the ledger over its registry's coordination tables.

        :param registry: the collection registry this ledger reads and writes through
        :ptype registry: CollectionRegistry
        :param purpose: which ledger these keys belong to (``"refresh_jti"``), carried in the row
            key so two ledgers never collide
        :ptype purpose: str
        :param ttl_seconds: how long a spent key is remembered -- at least the artifact's whole
            lifetime, since a key forgotten while its artifact is still valid is a key that can be
            spent twice. MUST be positive
        :ptype ttl_seconds: int
        :param config: core config forwarded when this process first builds the collection
        :ptype config: CoreConfig | None
        :raises ValueError: when ``ttl_seconds`` is not positive, or ``purpose`` is empty
        """
        if ttl_seconds <= 0:
            raise ValueError(f"RedemptionLedger ttl_seconds must be positive, got {ttl_seconds}")
        if not purpose.strip():
            raise ValueError("RedemptionLedger purpose must be a non-empty name, e.g. 'refresh_jti'")
        self._purpose = purpose
        self._ttl = timedelta(seconds=ttl_seconds)
        self._collection = coordination_collection(registry, CoordinationRedemptionsCollection, config)

    @property
    def purpose(self) -> str:
        """which ledger this instance's keys belong to.

        :return: the purpose
        :rtype: str
        """
        return self._purpose

    async def record_unique(self, key: str) -> bool:
        """record ``key`` as spent; report whether this call was the first to do so.

        The record is a create-if-absent compare-and-swap against L2 with the row written to L3
        before this returns, so exactly one caller across every replica is told ``True`` and a
        broker restart does not make a spent key spendable again.

        :param key: the artifact identifier, e.g. a refresh-token ``jti``. Hashed before storage
        :ptype key: str
        :return: ``True`` on the first sighting, ``False`` on every later one
        :rtype: bool
        :raises threetears.nats.KvError: on an L2 failure -- the caller MUST deny
        """
        now = datetime.now(UTC)
        row_id = self._row_id(key)

        def _claim_if_absent(
            current: dict[str, Any] | None,
        ) -> tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]:
            # an entry past its own expiry is absent to this callback at every tier, which is
            # correct: the artifact it recorded can no longer be presented either.
            if current is None:
                return "upsert", {
                    "purpose": self._purpose,
                    "key": row_id[1],
                    "expires_at": now + self._ttl,
                }
            return "noop", None

        outcome = await self._collection.l2_cas_mutate(row_id, _claim_if_absent)
        await self._collection.sweep_expired_if_due()
        return outcome.action == "created"

    async def was_redeemed(self, key: str) -> bool:
        """whether ``key`` has already been spent, recording nothing.

        :param key: the artifact identifier to look up
        :ptype key: str
        :return: whether a live entry exists
        :rtype: bool
        :raises threetears.nats.KvError: on an L2 failure -- the caller MUST deny
        """
        return await self._collection.get(self._row_id(key)) is not None

    def _row_id(self, key: str) -> tuple[str, str]:
        """the row this key addresses.

        :param key: the caller's artifact identifier
        :ptype key: str
        :return: ``(purpose, hashed key)`` in declared column order
        :rtype: tuple[str, str]
        """
        return (self._purpose, _hashed(key))
