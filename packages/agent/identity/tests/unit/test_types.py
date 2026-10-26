"""agent-identity value-type tests: enum members, wire values, tier map."""

from __future__ import annotations

import pytest

from threetears.agent.identity.types import (
    IDENTITY_BLOCK_KEY_VALUES,
    IDENTITY_BLOCK_TIERS,
    IDENTITY_STATUS_VALUES,
    IdentityBlockKey,
    IdentityTier,
    IdentityVersionStatus,
    check_variant,
    variant_label,
)


def test_block_key_values_are_declaration_ordered() -> None:
    assert IDENTITY_BLOCK_KEY_VALUES == (
        "personality",
        "reinforcement",
        "anti_sycophant",
        "self_improvement",
        "presence",
        "wants",
        "needs",
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
    tier2 = {
        IdentityBlockKey.SELF_IMPROVEMENT,
        IdentityBlockKey.REINFORCEMENT,
        # Wants / needs are the agent's own homeostatic conditions --
        # regulated, not requested -- so they auto-apply under a human
        # veto rather than waiting on consent. Pace's ruling, 2026-08-22.
        IdentityBlockKey.WANTS,
        IdentityBlockKey.NEEDS,
    }
    assert {b for b, t in IDENTITY_BLOCK_TIERS.items() if t is IdentityTier.TIER_1} == tier1
    assert {b for b, t in IDENTITY_BLOCK_TIERS.items() if t is IdentityTier.TIER_2} == tier2


@pytest.mark.parametrize("name", ["", "alt", "second_copy", "v2", "a" * 32])
def test_check_variant_takes_a_name_a_block_takes(name: str) -> None:
    assert check_variant(name) == name


@pytest.mark.parametrize(
    "name",
    ["Alt", "with-dash", "with space", " ", "a" * 33, "dot.ted", "ü", None, 3],
)
def test_check_variant_refuses_any_other_name(name: object) -> None:
    with pytest.raises(ValueError, match="not a name a block takes"):
        check_variant(name)  # type: ignore[arg-type]


def test_variant_label_names_the_block_and_its_variant() -> None:
    assert variant_label("presence", "") == "presence"
    assert variant_label("presence", "alt") == "presence/alt"
