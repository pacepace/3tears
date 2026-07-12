"""Identity self-evolution lifecycle: propose -> consent/reject -> apply + rollback.

These functions orchestrate the :class:`IdentityVersionsCollection` (which
stays a pure DB store) + emit the FrameworkEvents. They are the write
authority the ``identity_propose`` tool + the host's consent / rollback API
call. Each op authorizes via the owner short-circuit
(:func:`authorize_identity_access`) and returns the affected version (or
``None`` on a not-found / not-owned / wrong-state no-op, which the caller
maps to a 404 / error).

**The one-active invariant.** The partial UNIQUE index forbids two
``active`` rows per block, so an apply supersedes the prior active FIRST,
then activates the new one. The ordering (not a DB transaction -- the L3
abstraction exposes no connection-level one) is the guarantee: the brief
window between the two writes has ZERO active rows (allowed), never two. A
crash between them leaves no active version, which is recoverable (fall
back to the default; a re-propose / rollback restores an active).

**The consent tiers** (Pace: tier from day one). Tier-1 blocks are born
``proposed`` and await :func:`consent`; tier-2 blocks :func:`propose`
auto-applies immediately (with an async veto via :func:`reject` of a later
proposal / :func:`rollback`). :data:`IDENTITY_BLOCK_TIERS` is the map.

**Cycle-freedom (not strict linearity).** Every new version parents to a
version that already exists at creation time (the current active head, or
``None`` for the first), so a cycle is impossible -- no origin walk / guard
is needed (unlike the knowledge shadow-chain's arbitrary origin pointers).
The chain is NOT guaranteed strictly linear, though: two concurrent tier-1
proposals both parent to the same head, and consent does not re-parent, so
sibling branches can exist among proposed/superseded rows. That is
harmless here -- the one-active invariant still holds and nothing walks
parents -- but a history UI must not assume a single linear path.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID

from uuid_utils import uuid7

from threetears.langgraph.events import FrameworkEvent, dispatch_event
from threetears.observe import get_logger

from threetears.agent.identity.authorize import (
    ACTION_IDENTITY_WRITE,
    IdentityAuthorizerDependencies,
    authorize_identity_access,
)
from threetears.agent.identity.collections import IdentityVersionsCollection
from threetears.agent.identity.entities import IdentityVersionEntity
from threetears.agent.identity.events import (
    IdentityAppliedEvent,
    IdentityConsentedEvent,
    IdentityProposedEvent,
    IdentityRolledBackEvent,
)
from threetears.agent.identity.types import (
    IDENTITY_BLOCK_KEY_VALUES,
    IDENTITY_BLOCK_TIERS,
    IdentityBlockKey,
    IdentityTier,
    IdentityVersionStatus,
)

log = get_logger(__name__)

__all__ = [
    "content_hash",
    "consent",
    "propose",
    "reject",
    "rollback",
    "seed_active",
]

#: rationale-preview length carried on the proposed event.
_RATIONALE_PREVIEW_LEN = 120


def content_hash(content: str) -> str:
    """Return the sha256 hex digest of a block's content (dedup + integrity)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def _emit(event: FrameworkEvent) -> None:
    """Dispatch a framework event; best-effort outside a langgraph run.

    ``dispatch_event`` needs a run manager (a langgraph run context). The
    consent / rollback API path has none, so a ``RuntimeError`` is swallowed
    -- the durable DB state is the contract; the event is an aid.

    :param event: the event to dispatch
    :ptype event: FrameworkEvent
    :rtype: None
    """
    try:
        await dispatch_event(event)
    except RuntimeError:
        log.debug("identity event dispatched outside a run; skipped", extra={"extra_data": {}})


def _tier_for(block_key: str) -> IdentityTier:
    """Resolve the consent tier for a block key, validating the key."""
    if block_key not in IDENTITY_BLOCK_KEY_VALUES:
        raise ValueError(f"unknown identity block_key {block_key!r}")
    return IDENTITY_BLOCK_TIERS[IdentityBlockKey(block_key)]


async def _apply(
    collection: IdentityVersionsCollection,
    *,
    new_version: IdentityVersionEntity,
    prior_active: IdentityVersionEntity | None,
) -> None:
    """Supersede the prior active, then persist the new active.

    The **ordering** (supersede first, activate second) is what keeps the
    one-active partial-unique index satisfied -- the brief window between
    the two writes has ZERO active rows (allowed), never two. A crash
    between them leaves the block with no active version, which is
    recoverable (the consumer falls back to its default and a re-propose /
    rollback restores an active). The L3 abstraction does not expose a
    connection-level transaction, so this ordering is the atomicity
    guarantee rather than a single DB transaction.
    """
    if prior_active is not None and prior_active.version_id != new_version.version_id:
        prior_active.status = IdentityVersionStatus.SUPERSEDED.value
        await collection.save_entity(prior_active)
    await collection.save_entity(new_version)


async def propose(
    collection: IdentityVersionsCollection,
    authorizer: IdentityAuthorizerDependencies,
    *,
    agent_id: UUID,
    customer_id: UUID | None,
    user_id: UUID | None,
    block_key: str,
    content: str,
    rationale: str | None,
    proposer_agent_id: UUID,
    caller_agent_id: UUID,
    now: datetime | None = None,
) -> IdentityVersionEntity | None:
    """Propose a new version of an identity block.

    Tier-1 (identity-shaping) blocks are born ``proposed`` and await
    :func:`consent`. Tier-2 (routine) blocks auto-apply immediately. Identical
    content to the current active is a no-op (returns the active).

    :return: the new version (or the active on a dedup no-op)
    :rtype: IdentityVersionEntity | None
    """
    await authorize_identity_access(
        action=ACTION_IDENTITY_WRITE,
        agent_id=agent_id,
        customer_id=_scope(customer_id),
        caller_agent_id=caller_agent_id,
        deps=authorizer,
    )
    tier = _tier_for(block_key)
    digest = content_hash(content)
    active = await collection.resolve_active(
        agent_id=agent_id, customer_id=customer_id, user_id=user_id, block_key=block_key
    )
    if active is not None and active.content_hash == digest:
        return active  # dedup: proposing identical content is a no-op

    stamp = now if now is not None else datetime.now(UTC)
    is_auto = tier is IdentityTier.TIER_2
    new_version: IdentityVersionEntity = collection.create(
        {
            "version_id": UUID(str(uuid7())),
            "agent_id": agent_id,
            "customer_id": customer_id,
            "user_id": user_id,
            "block_key": block_key,
            "content": content,
            "rationale": rationale,
            "content_hash": digest,
            "parent_version_id": active.version_id if active is not None else None,
            "status": (
                IdentityVersionStatus.ACTIVE.value
                if is_auto
                else IdentityVersionStatus.PROPOSED.value
            ),
            "proposer_agent_id": proposer_agent_id,
            "consenter_user_id": None,
            "date_created": stamp,
            "date_updated": stamp,
        }
    )
    if is_auto:
        await _apply(collection, new_version=new_version, prior_active=active)
        await _emit(_applied_event(new_version, auto_applied=True))
    else:
        await collection.save_entity(new_version)
        await _emit(_proposed_event(new_version, rationale))
    return new_version


async def seed_active(
    collection: IdentityVersionsCollection,
    authorizer: IdentityAuthorizerDependencies,
    *,
    agent_id: UUID,
    customer_id: UUID | None,
    user_id: UUID | None,
    block_key: str,
    content: str,
    now: datetime | None = None,
) -> IdentityVersionEntity:
    """Import existing content as the ``active`` root version for a block.

    The baseline-import op a consumer adopting identity versioning over
    pre-existing prompt state calls once per block: it materialises the
    current content as an ``active`` root (``parent_version_id=None``,
    ``proposer_agent_id=None``, ``consenter_user_id=None``, ``rationale=None``)
    so the block's history starts at what was already live and rollback has a
    floor. **Idempotent**: if the block already has an active version this is a
    no-op returning it -- so a lazy "seed on first access" caller is safe to
    call every access. Emits no event (a silent baseline, not a proposal /
    apply -- it must never appear in the veto queue).

    Concurrency: two first-access callers can race; the ``active`` partial-
    unique index lets exactly one insert win and the loser re-resolves to it.

    :param collection: the identity-versions collection
    :ptype collection: IdentityVersionsCollection
    :param authorizer: the owner-short-circuit authorizer bundle
    :ptype authorizer: IdentityAuthorizerDependencies
    :param agent_id: partition column (the namespace owner)
    :ptype agent_id: UUID
    :param customer_id: scope grain (nullable)
    :ptype customer_id: UUID | None
    :param user_id: owning user (isolation boundary; nullable grain)
    :ptype user_id: UUID | None
    :param block_key: the identity block being seeded
    :ptype block_key: str
    :param content: the existing content to import as the active root
    :ptype content: str
    :param now: clock injection point; defaults to ``datetime.now(UTC)``
    :ptype now: datetime | None
    :return: the seeded root (or the pre-existing active on a no-op)
    :rtype: IdentityVersionEntity
    """
    await authorize_identity_access(
        action=ACTION_IDENTITY_WRITE,
        agent_id=agent_id,
        customer_id=_scope(customer_id),
        caller_agent_id=agent_id,
        deps=authorizer,
    )
    _tier_for(block_key)  # validate the key
    existing = await collection.resolve_active(
        agent_id=agent_id, customer_id=customer_id, user_id=user_id, block_key=block_key
    )
    if existing is not None:
        return existing  # already seeded; idempotent no-op

    stamp = now if now is not None else datetime.now(UTC)
    root: IdentityVersionEntity = collection.create(
        {
            "version_id": UUID(str(uuid7())),
            "agent_id": agent_id,
            "customer_id": customer_id,
            "user_id": user_id,
            "block_key": block_key,
            "content": content,
            "rationale": None,
            "content_hash": content_hash(content),
            "parent_version_id": None,
            "status": IdentityVersionStatus.ACTIVE.value,
            "proposer_agent_id": None,
            "consenter_user_id": None,
            "date_created": stamp,
            "date_updated": stamp,
        }
    )
    try:
        await collection.save_entity(root)
    except Exception:  # prawduct:allow prawduct/broad-except -- concurrent first-access seed races on the active partial-unique index; re-resolve returns the winner
        winner = await collection.resolve_active(
            agent_id=agent_id, customer_id=customer_id, user_id=user_id, block_key=block_key
        )
        if winner is not None:
            return winner
        raise
    return root


async def consent(
    collection: IdentityVersionsCollection,
    authorizer: IdentityAuthorizerDependencies,
    *,
    agent_id: UUID,
    customer_id: UUID | None,
    user_id: UUID | None,
    version_id: UUID,
    consenter_user_id: UUID,
    caller_agent_id: UUID,
) -> IdentityVersionEntity | None:
    """Consent to a proposed version -> it becomes active (tier-1 apply).

    :return: the applied version, or ``None`` if not found / not owned / not
        currently ``proposed``
    :rtype: IdentityVersionEntity | None
    """
    await authorize_identity_access(
        action=ACTION_IDENTITY_WRITE,
        agent_id=agent_id,
        customer_id=_scope(customer_id),
        caller_agent_id=caller_agent_id,
        deps=authorizer,
    )
    version = await collection.get((agent_id, version_id))
    if (
        version is None
        or version.user_id != user_id
        or version.status != IdentityVersionStatus.PROPOSED.value
    ):
        return None
    prior_active = await collection.resolve_active(
        agent_id=agent_id, customer_id=customer_id, user_id=user_id, block_key=version.block_key
    )
    version.status = IdentityVersionStatus.ACTIVE.value
    version.consenter_user_id = consenter_user_id
    await _apply(collection, new_version=version, prior_active=prior_active)
    await _emit(_consented_event(version))
    await _emit(_applied_event(version, auto_applied=False))
    return version


async def reject(
    collection: IdentityVersionsCollection,
    authorizer: IdentityAuthorizerDependencies,
    *,
    agent_id: UUID,
    customer_id: UUID | None,
    user_id: UUID | None,
    version_id: UUID,
    caller_agent_id: UUID,
) -> IdentityVersionEntity | None:
    """Reject a proposed version (kept for history; no active change).

    :return: the rejected version, or ``None`` if not found / not owned / not
        currently ``proposed``
    :rtype: IdentityVersionEntity | None
    """
    await authorize_identity_access(
        action=ACTION_IDENTITY_WRITE,
        agent_id=agent_id,
        customer_id=_scope(customer_id),
        caller_agent_id=caller_agent_id,
        deps=authorizer,
    )
    version = await collection.get((agent_id, version_id))
    if (
        version is None
        or version.user_id != user_id
        or version.status != IdentityVersionStatus.PROPOSED.value
    ):
        return None
    version.status = IdentityVersionStatus.REJECTED.value
    await collection.save_entity(version)
    return version


async def rollback(
    collection: IdentityVersionsCollection,
    authorizer: IdentityAuthorizerDependencies,
    *,
    agent_id: UUID,
    customer_id: UUID | None,
    user_id: UUID | None,
    target_version_id: UUID,
    consenter_user_id: UUID,
    caller_agent_id: UUID,
    now: datetime | None = None,
) -> IdentityVersionEntity | None:
    """Restore a prior version's content as a NEW active version (non-destructive).

    Clones ``target``'s content into a fresh ``active`` version parented to
    the current head; the prior active is superseded. History is never
    mutated.

    :return: the new active (clone) version, or ``None`` if the target is not
        found / not owned
    :rtype: IdentityVersionEntity | None
    """
    await authorize_identity_access(
        action=ACTION_IDENTITY_WRITE,
        agent_id=agent_id,
        customer_id=_scope(customer_id),
        caller_agent_id=caller_agent_id,
        deps=authorizer,
    )
    target = await collection.get((agent_id, target_version_id))
    if target is None or target.user_id != user_id:
        return None
    prior_active = await collection.resolve_active(
        agent_id=agent_id, customer_id=customer_id, user_id=user_id, block_key=target.block_key
    )
    stamp = now if now is not None else datetime.now(UTC)
    clone: IdentityVersionEntity = collection.create(
        {
            "version_id": UUID(str(uuid7())),
            "agent_id": agent_id,
            "customer_id": customer_id,
            "user_id": user_id,
            "block_key": target.block_key,
            "content": target.content,
            "rationale": f"rollback to {target_version_id}",
            "content_hash": target.content_hash,
            "parent_version_id": prior_active.version_id if prior_active is not None else None,
            "status": IdentityVersionStatus.ACTIVE.value,
            "proposer_agent_id": None,
            "consenter_user_id": consenter_user_id,
            "date_created": stamp,
            "date_updated": stamp,
        }
    )
    await _apply(collection, new_version=clone, prior_active=prior_active)
    await _emit(_rolled_back_event(clone, target_version_id))
    return clone


def _scope(customer_id: UUID | None) -> UUID:
    """The authorizer needs a customer UUID; a null grain uses the nil UUID.

    The owner short-circuit only reads the descriptor's owner_agent_id, so
    the customer value is not load-bearing for the allow decision; the nil
    UUID is a stable stand-in for the agent-internal (null-customer) case.
    """
    return customer_id if customer_id is not None else UUID(int=0)


# --- event builders (border: FrameworkEvent payloads carry str-form uuids) ---


def _proposed_event(version: IdentityVersionEntity, rationale: str | None) -> IdentityProposedEvent:
    return IdentityProposedEvent(
        agent_id=str(version.agent_id),  # convert at border: event payload wire uuid
        version_id=str(version.version_id),  # convert at border: event payload wire uuid
        block_key=version.block_key,
        user_id=_uid(version),
        rationale_preview=(rationale or "")[:_RATIONALE_PREVIEW_LEN],
    )


def _consented_event(version: IdentityVersionEntity) -> IdentityConsentedEvent:
    return IdentityConsentedEvent(
        agent_id=str(version.agent_id),  # convert at border: event payload wire uuid
        version_id=str(version.version_id),  # convert at border: event payload wire uuid
        block_key=version.block_key,
        user_id=_uid(version),
    )


def _applied_event(version: IdentityVersionEntity, *, auto_applied: bool) -> IdentityAppliedEvent:
    return IdentityAppliedEvent(
        agent_id=str(version.agent_id),  # convert at border: event payload wire uuid
        version_id=str(version.version_id),  # convert at border: event payload wire uuid
        block_key=version.block_key,
        user_id=_uid(version),
        auto_applied=auto_applied,
    )


def _rolled_back_event(
    version: IdentityVersionEntity, target_version_id: UUID
) -> IdentityRolledBackEvent:
    return IdentityRolledBackEvent(
        agent_id=str(version.agent_id),  # convert at border: event payload wire uuid
        version_id=str(version.version_id),  # convert at border: event payload wire uuid
        block_key=version.block_key,
        user_id=_uid(version),
        target_version_id=str(target_version_id),  # convert at border: event payload wire uuid
    )


def _uid(version: IdentityVersionEntity) -> str | None:
    uid = version.user_id
    return str(uid) if uid is not None else None  # convert at border: event payload wire uuid
