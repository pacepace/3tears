"""API-key secret generation, hashing, and presentation."""

from __future__ import annotations

import hashlib

from threetears.iam.apikeys import (
    DEFAULT_KEY_PREFIX,
    KEY_PREFIX_LEN,
    generate_api_key_secret,
    hash_api_key_secret,
    key_prefix,
    secrets_match,
)


def test_generated_secret_carries_the_marker() -> None:
    assert generate_api_key_secret().startswith(DEFAULT_KEY_PREFIX)


def test_marker_is_overridable() -> None:
    assert generate_api_key_secret(prefix="acme_").startswith("acme_")


def test_generated_secrets_are_unique_and_high_entropy() -> None:
    secrets_seen = {generate_api_key_secret() for _ in range(200)}
    assert len(secrets_seen) == 200
    # 32 random bytes base64url-encoded is 43 characters, plus the marker.
    assert all(len(value) >= len(DEFAULT_KEY_PREFIX) + 43 for value in secrets_seen)


def test_hash_is_a_plain_sha256_hex_digest() -> None:
    raw = generate_api_key_secret()
    assert hash_api_key_secret(raw) == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert len(hash_api_key_secret(raw)) == 64


def test_hash_is_deterministic() -> None:
    # Deterministic on purpose: it is what lets a verifier resolve a key by equality lookup
    # instead of running a KDF against every stored key on every request.
    raw = generate_api_key_secret()
    assert hash_api_key_secret(raw) == hash_api_key_secret(raw)


def test_distinct_secrets_hash_differently() -> None:
    assert hash_api_key_secret(generate_api_key_secret()) != hash_api_key_secret(generate_api_key_secret())


def test_key_prefix_is_short_and_derived_from_the_secret() -> None:
    raw = generate_api_key_secret()
    prefix = key_prefix(raw)
    assert len(prefix) == KEY_PREFIX_LEN
    assert raw.startswith(prefix)
    # Short enough that the displayed prefix is not a meaningful head start on the secret.
    assert len(prefix) < len(raw) // 4


def test_key_prefix_length_is_overridable() -> None:
    assert len(key_prefix(generate_api_key_secret(), length=12)) == 12


def test_secrets_match_compares_equal_hashes() -> None:
    digest = hash_api_key_secret(generate_api_key_secret())
    assert secrets_match(digest, digest)


def test_secrets_match_rejects_different_hashes() -> None:
    assert not secrets_match(
        hash_api_key_secret(generate_api_key_secret()),
        hash_api_key_secret(generate_api_key_secret()),
    )


def test_secrets_match_rejects_a_length_mismatch() -> None:
    assert not secrets_match("short", hash_api_key_secret(generate_api_key_secret()))
