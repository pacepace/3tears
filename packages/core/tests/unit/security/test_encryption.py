"""Encryption-at-rest util: AES-256-GCM seal/open under a master key.

The contract a stored secret depends on: a sealed token round-trips ONLY with the
same master key, any tamper or wrong key is rejected (never silently returns garbage),
the plaintext never appears in the token, and the opened value comes back inside a
``SecretStr`` so it cannot leak through a repr/log.
"""

from __future__ import annotations

import base64

import pytest
from pydantic import SecretStr

from threetears.core.security.encryption import DecryptionError, open_secret, seal

_KEY = SecretStr("an-operator-provided-master-key-with-plenty-of-entropy")
_OTHER_KEY = SecretStr("a-different-master-key-entirely-also-high-entropy")
_PLAINTEXT = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk...\n-----END OPENSSH PRIVATE KEY-----\n"


def test_round_trips_under_the_same_key() -> None:
    token = seal(_PLAINTEXT, _KEY)
    opened = open_secret(token, _KEY)
    assert isinstance(opened, SecretStr)
    assert opened.get_secret_value() == _PLAINTEXT


def test_round_trips_unicode() -> None:
    value = "ключ-🔑-naïve"
    assert open_secret(seal(value, _KEY), _KEY).get_secret_value() == value


def test_wrong_key_is_rejected_not_garbage() -> None:
    token = seal(_PLAINTEXT, _KEY)
    with pytest.raises(DecryptionError):
        open_secret(token, _OTHER_KEY)


def test_tampered_token_is_rejected() -> None:
    token = seal(_PLAINTEXT, _KEY)
    raw = bytearray(base64.urlsafe_b64decode(token))
    raw[-1] ^= 0x01  # flip a bit in the GCM tag
    tampered = base64.urlsafe_b64encode(bytes(raw)).decode()
    with pytest.raises(DecryptionError):
        open_secret(tampered, _KEY)


def test_garbage_token_raises_decryption_error_not_a_crash() -> None:
    for junk in ("", "not-base64-!!!", base64.urlsafe_b64encode(b"too-short").decode()):
        with pytest.raises(DecryptionError):
            open_secret(junk, _KEY)


def test_non_canonical_token_is_rejected() -> None:
    # A valid token with trailing junk / a stray char must NOT decode to the same secret —
    # base64 leniency would otherwise give one secret many string forms (b64-malleability).
    token = seal(_PLAINTEXT, _KEY)
    for variant in (token + "A", token + "extra", token[:-1] + "!" + token[-1]):
        with pytest.raises(DecryptionError):
            open_secret(variant, _KEY)


def test_unsupported_version_is_rejected() -> None:
    # A token whose version byte isn't this build's is refused (not mis-parsed). The version
    # is also AAD, so flipping it would fail the tag even without the explicit check.
    raw = bytearray(base64.urlsafe_b64decode(seal(_PLAINTEXT, _KEY)))
    raw[0] = 0x02
    bumped = base64.urlsafe_b64encode(bytes(raw)).decode()
    with pytest.raises(DecryptionError):
        open_secret(bumped, _KEY)


def test_nonce_is_random_so_two_seals_differ() -> None:
    # Same plaintext + key seals to different tokens (fresh nonce each call) — no
    # deterministic ciphertext that would leak equality of stored secrets.
    assert seal(_PLAINTEXT, _KEY) != seal(_PLAINTEXT, _KEY)


def test_token_never_contains_the_plaintext() -> None:
    token = seal(_PLAINTEXT, _KEY)
    assert _PLAINTEXT not in token
    assert "BEGIN OPENSSH" not in base64.urlsafe_b64decode(token).decode("latin-1")


def test_opened_secret_repr_does_not_leak() -> None:
    opened = open_secret(seal(_PLAINTEXT, _KEY), _KEY)
    assert _PLAINTEXT not in repr(opened)
    assert "BEGIN OPENSSH" not in str(opened)


def test_token_is_versioned_for_future_rotation() -> None:
    # First byte of the decoded token is a format-version marker, so the algorithm /
    # key-derivation can evolve without orphaning already-stored ciphertext.
    raw = base64.urlsafe_b64decode(seal(_PLAINTEXT, _KEY))
    assert raw[0] == 0x01
