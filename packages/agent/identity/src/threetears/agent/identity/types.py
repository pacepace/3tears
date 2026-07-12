"""Identity value types -- the block-key set, the version status enum, and
the per-block consent tier map.

Two *constrained value sets* (not transition state machines, mirroring
``agent-intention``'s ``IntentionStatus``): ``IdentityBlockKey`` bounds
which identity block a version belongs to, and ``IdentityVersionStatus``
bounds which lifecycle value a version's ``status`` column may hold.
Nothing guards a transition here; the enums only bound the value space.

``IDENTITY_BLOCK_TIERS`` encodes the consent model (Pace, 2026-07-12,
"tier from day one"): tier 1 blocks are identity-shaping and require
human consent before a proposed version applies; tier 2 blocks are
routine and auto-apply with an async veto. The tier is intrinsic to the
block (a constant map), NOT a column.

Each ``*_VALUES`` tuple is the single source of truth the schema Column's
``enum_type`` imports AND the migration ``CREATE TYPE`` DDL mirrors, so a
drift between the DSL declaration and the hand-written enum surfaces as a
test failure rather than at runtime.
"""

from __future__ import annotations

from enum import Enum, IntEnum

__all__ = [
    "IDENTITY_BLOCK_KEY_VALUES",
    "IDENTITY_BLOCK_TIERS",
    "IDENTITY_STATUS_VALUES",
    "IdentityBlockKey",
    "IdentityTier",
    "IdentityVersionStatus",
]


class IdentityBlockKey(str, Enum):
    """Which identity block a version belongs to.

    These mirror metallm's ``users.config_*`` prompt columns: the
    system-prefix persona (``PERSONALITY``), the post-history reinforcement
    (``REINFORCEMENT``), the anti-sycophancy guard (``ANTI_SYCOPHANT``),
    the self-improvement scratch prompt (``SELF_IMPROVEMENT``), and the
    presence/aliveness voice (``PRESENCE``). Subclasses :class:`str` so a
    member compares equal to its wire value and serialises transparently.
    """

    PERSONALITY = "personality"
    REINFORCEMENT = "reinforcement"
    ANTI_SYCOPHANT = "anti_sycophant"
    SELF_IMPROVEMENT = "self_improvement"
    PRESENCE = "presence"


class IdentityVersionStatus(str, Enum):
    """Lifecycle status for one identity-block version.

    Conventional flow (not enforced): ``PROPOSED`` on an agent proposal;
    ``PROPOSED -> ACTIVE`` on user consent (tier 1) or immediately (tier
    2); the prior active flips ``ACTIVE -> SUPERSEDED``; ``PROPOSED ->
    REJECTED`` when the user declines. Exactly one ``ACTIVE`` version per
    ``(agent_id, customer_id, user_id, block_key)`` (a partial unique
    index enforces it).
    """

    PROPOSED = "proposed"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class IdentityTier(IntEnum):
    """Consent tier for an identity block.

    ``TIER_1`` (identity-shaping) requires human consent before a proposal
    applies. ``TIER_2`` (routine) auto-applies with an async veto.
    """

    TIER_1 = 1
    TIER_2 = 2


#: declaration-order tuples of the wire values -- the schema enum_type and
#: the migration ``CREATE TYPE`` DDL share this ordering.
IDENTITY_BLOCK_KEY_VALUES: tuple[str, ...] = tuple(block.value for block in IdentityBlockKey)
IDENTITY_STATUS_VALUES: tuple[str, ...] = tuple(status.value for status in IdentityVersionStatus)


#: The consent tier for each block (Pace: tier from day one). Tier 1 =
#: consent-gated identity-shaping; tier 2 = auto-apply + async veto.
IDENTITY_BLOCK_TIERS: dict[IdentityBlockKey, IdentityTier] = {
    IdentityBlockKey.PERSONALITY: IdentityTier.TIER_1,
    IdentityBlockKey.ANTI_SYCOPHANT: IdentityTier.TIER_1,
    IdentityBlockKey.PRESENCE: IdentityTier.TIER_1,
    IdentityBlockKey.SELF_IMPROVEMENT: IdentityTier.TIER_2,
    IdentityBlockKey.REINFORCEMENT: IdentityTier.TIER_2,
}
