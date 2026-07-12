"""agent-identity value-type tests: enum members, wire values, tier map."""

from __future__ import annotations

from threetears.agent.identity.types import (
    IDENTITY_BLOCK_KEY_VALUES,
    IDENTITY_BLOCK_TIERS,
    IDENTITY_STATUS_VALUES,
    IdentityBlockKey,
    IdentityTier,
    IdentityVersionStatus,
)


def test_block_key_values_are_declaration_ordered() -> None:
    assert IDENTITY_BLOCK_KEY_VALUES == (
        "personality",
        "reinforcement",
        "anti_sycophant",
        "self_improvement",
        "presence",
    )


def test_status_values_are_declaration_ordered() -> None:
    assert IDENTITY_STATUS_VALUES == ("proposed", "active", "superseded", "rejected")


def test_enum_members_equal_wire_values() -> None:
    assert IdentityBlockKey.PERSONALITY == "personality"
    assert IdentityVersionStatus.ACTIVE == "active"


def test_every_block_has_a_tier() -> None:
    """The tier map must cover exactly the block-key set (no gaps / extras)."""
    assert set(IDENTITY_BLOCK_TIERS) == set(IdentityBlockKey)


def test_tier_assignment_matches_signed_off_model() -> None:
    """Tier from day one: identity-shaping = tier 1; routine = tier 2."""
    tier1 = {IdentityBlockKey.PERSONALITY, IdentityBlockKey.ANTI_SYCOPHANT, IdentityBlockKey.PRESENCE}
    tier2 = {IdentityBlockKey.SELF_IMPROVEMENT, IdentityBlockKey.REINFORCEMENT}
    assert {b for b, t in IDENTITY_BLOCK_TIERS.items() if t is IdentityTier.TIER_1} == tier1
    assert {b for b, t in IDENTITY_BLOCK_TIERS.items() if t is IdentityTier.TIER_2} == tier2
