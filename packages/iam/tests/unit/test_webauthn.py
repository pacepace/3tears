"""WebAuthn/passkey helpers."""

from __future__ import annotations

import pytest

from threetears.iam.webauthn import (
    CHALLENGE_BYTES,
    generate_challenge,
    is_signature_counter_regression,
    origin_allowed,
)


def test_challenges_are_long_and_unique() -> None:
    challenges = {generate_challenge() for _ in range(100)}
    assert len(challenges) == 100
    assert all(len(challenge) == CHALLENGE_BYTES for challenge in challenges)
    # WebAuthn requires at least 16 bytes of entropy.
    assert CHALLENGE_BYTES >= 16


@pytest.mark.parametrize(
    ("stored", "new"),
    [(1, 2), (0, 1), (41, 42), (0, 100)],
)
def test_an_increasing_counter_is_not_a_regression(stored: int, new: int) -> None:
    assert not is_signature_counter_regression(stored_sign_count=stored, new_sign_count=new)


@pytest.mark.parametrize(
    ("stored", "new"),
    [(2, 1), (2, 2), (100, 0), (1, 0), (42, 41)],
)
def test_a_non_increasing_counter_is_a_regression(stored: int, new: int) -> None:
    # Either the credential has been copied out of the authenticator -- the thing the counter
    # exists to detect -- or something is badly wrong. Both mean lock it.
    assert is_signature_counter_regression(stored_sign_count=stored, new_sign_count=new)


def test_a_constant_zero_counter_is_exempt() -> None:
    # Synced passkeys (iCloud Keychain, Google Password Manager) keep no counter and report
    # zero forever. Treating that as a clone locks out every passkey user on the platform.
    assert not is_signature_counter_regression(stored_sign_count=0, new_sign_count=0)


def test_a_previously_nonzero_counter_reporting_zero_is_not_exempt() -> None:
    # A real authenticator's counter does not reset. That transition IS the clone signal, and
    # exempting it would turn the synced-passkey carve-out into a universal bypass.
    assert is_signature_counter_regression(stored_sign_count=5, new_sign_count=0)


def test_origin_must_match_exactly() -> None:
    allowed = frozenset({"https://app.example.com"})
    assert origin_allowed("https://app.example.com", allowed)
    assert not origin_allowed("https://app.example.com.attacker.net", allowed)
    assert not origin_allowed("http://app.example.com", allowed)
    assert not origin_allowed("https://app.example.com/", allowed)


def test_an_empty_origin_allow_list_accepts_nothing() -> None:
    # Failing closed is the correct behaviour for an unconfigured relying party.
    assert not origin_allowed("https://app.example.com", frozenset())


def test_several_origins_are_supported() -> None:
    allowed = frozenset({"https://app.example.com", "https://admin.example.com"})
    assert origin_allowed("https://admin.example.com", allowed)
    assert not origin_allowed("https://other.example.com", allowed)
