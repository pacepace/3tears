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

import re
from enum import Enum, IntEnum

__all__ = [
    "IDENTITY_BLOCK_KEY_VALUES",
    "IDENTITY_BLOCK_TIERS",
    "IDENTITY_STATUS_VALUES",
    "IdentityBlockKey",
    "IdentityTier",
    "IdentityVersionStatus",
    "check_variant",
    "variant_label",
]


class IdentityBlockKey(str, Enum):
    """Which identity block a version belongs to.

    These are the prompt blocks an agent's system prompt is built from: the
    system-prefix persona (``PERSONALITY``), the post-history reinforcement
    (``REINFORCEMENT``), the anti-sycophancy guard (``ANTI_SYCOPHANT``),
    the self-improvement scratch prompt (``SELF_IMPROVEMENT``), the
    presence/aliveness voice (``PRESENCE``), and the standing appetite and
    regulatory conditions the blocks layout renders ahead of the
    conversation (``WANTS`` / ``NEEDS``). Subclasses :class:`str` so a
    member compares equal to its wire value and serialises transparently.
    """

    PERSONALITY = "personality"
    REINFORCEMENT = "reinforcement"
    ANTI_SYCOPHANT = "anti_sycophant"
    SELF_IMPROVEMENT = "self_improvement"
    PRESENCE = "presence"
    WANTS = "wants"
    NEEDS = "needs"


class IdentityVersionStatus(str, Enum):
    """Lifecycle status for one identity-block version.

    Conventional flow (not enforced): ``PROPOSED`` on an agent proposal;
    ``PROPOSED -> ACTIVE`` on user consent (tier 1) or immediately (tier
    2); the prior active flips ``ACTIVE -> SUPERSEDED``; ``PROPOSED ->
    REJECTED`` when the user declines. Exactly one ``ACTIVE`` version per
    ``(agent_id, customer_id, user_id, block_key, variant)`` (a partial unique
    index enforces it), so each variant of a block has its own.
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
    # TIER_2 (Pace, 2026-08-22). Wants and Needs are identity-shaping in the
    # sense PERSONALITY is, which would argue tier 1 — the ruling went the
    # other way deliberately: these are the agent's own homeostatic
    # conditions, regulated rather than requested, so they auto-apply and the
    # human vetoes after the fact. Wider latitude than the persona block gets,
    # and chosen knowing that.
    IdentityBlockKey.WANTS: IdentityTier.TIER_2,
    IdentityBlockKey.NEEDS: IdentityTier.TIER_2,
}


#: the longest variant name a block takes.
_VARIANT_MAX_LEN = 32

_VARIANT_PATTERN = re.compile(rf"[a-z0-9_]{{0,{_VARIANT_MAX_LEN}}}")


def check_variant(variant: str) -> str:
    """Return ``variant`` when it is a name a block takes, else raise.

    A variant is a second copy of a block with its own version chain and its
    own active version, named by the consumer: the platform gives a name no
    meaning, so a consumer can keep several texts of one block and choose
    which applies where. The empty string is the block itself. A name is up
    to 32 lowercase letters, digits or
    underscores, so it reads the same in a log, a URL and a tool argument.

    :param variant: the variant name
    :ptype variant: str
    :return: the same name
    :rtype: str
    :raises ValueError: when the name is not one a block takes
    """
    if not isinstance(variant, str) or _VARIANT_PATTERN.fullmatch(variant) is None:
        raise ValueError(
            f"identity variant {variant!r} is not a name a block takes: up to "
            f"{_VARIANT_MAX_LEN} lowercase letters, digits or underscores"
        )
    return variant


def variant_label(block_key: str, variant: str) -> str:
    """The block as a person or a model reads it: ``block_key``, or ``block_key/variant`` for a variant.

    :param block_key: the identity block
    :ptype block_key: str
    :param variant: the block's variant; ``""`` is the block itself
    :ptype variant: str
    :return: the label
    :rtype: str
    """
    return f"{block_key}/{variant}" if variant else block_key
