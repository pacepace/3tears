"""TOTP second factors and backup codes."""

from __future__ import annotations

import base64

import pyotp
import pytest
from pydantic import SecretStr

from threetears.iam.totp import (
    BACKUP_CODE_COUNT,
    DecryptionError,
    generate_backup_codes,
    generate_seed,
    hash_backup_code,
    match_backup_code,
    provisioning_uri,
    seal_seed,
    unseal_seed,
    verify_code,
)

_KEY = SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode())
_OTHER_KEY = SecretStr(base64.urlsafe_b64encode(b"j" * 32).decode())


def test_generated_seeds_are_base32_and_unique() -> None:
    seeds = {generate_seed() for _ in range(50)}
    assert len(seeds) == 50
    for seed in seeds:
        # Must be decodable as base32, or no authenticator app can accept it.
        base64.b32decode(seed)


def test_a_current_code_verifies() -> None:
    seed = generate_seed()
    assert verify_code(seed, pyotp.TOTP(seed).now())


def test_a_wrong_code_does_not_verify() -> None:
    seed = generate_seed()
    wrong = "000000" if pyotp.TOTP(seed).now() != "000000" else "111111"
    assert not verify_code(seed, wrong)


def test_a_code_from_another_seed_does_not_verify() -> None:
    assert not verify_code(generate_seed(), pyotp.TOTP(generate_seed()).now())


@pytest.mark.parametrize("code", ["", "abcdef", "12345", "1234567", "  ", "!!!!!!"])
def test_malformed_codes_fail_closed_without_raising(code: str) -> None:
    # A malformed code is an authentication failure, not an exception -- otherwise the
    # exception shape distinguishes rejection reasons.
    assert not verify_code(generate_seed(), code)


def test_one_step_of_skew_is_tolerated_each_side() -> None:
    # Authenticator apps and verifiers conventionally allow this; narrowing it mostly
    # produces support tickets from users with drifting phone clocks.
    seed = generate_seed()
    totp = pyotp.TOTP(seed)
    import time as _time

    now = int(_time.time())
    assert verify_code(seed, totp.at(now - 30))
    assert verify_code(seed, totp.at(now + 30))
    assert not verify_code(seed, totp.at(now - 120))


def test_provisioning_uri_names_the_issuer_and_account() -> None:
    uri = provisioning_uri(generate_seed(), account_name="ada@example.com", issuer_name="Acme")
    assert uri.startswith("otpauth://totp/")
    assert "issuer=Acme" in uri
    assert "ada%40example.com" in uri


def test_backup_codes_are_unique_and_counted() -> None:
    codes = generate_backup_codes()
    assert len(codes) == BACKUP_CODE_COUNT
    assert len(set(codes)) == BACKUP_CODE_COUNT


def test_backup_code_count_is_overridable() -> None:
    assert len(generate_backup_codes(3)) == 3


def test_backup_code_hash_is_stable_and_whitespace_tolerant() -> None:
    code = generate_backup_codes(1)[0]
    # Read off a screen and pasted: a trailing space that invalidates a user's last backup
    # code is a lockout, not a security control.
    assert hash_backup_code(code) == hash_backup_code(f"  {code}  ")
    assert len(hash_backup_code(code)) == 64


def test_matching_a_backup_code_returns_the_stored_hash() -> None:
    codes = generate_backup_codes(3)
    hashes = tuple(hash_backup_code(code) for code in codes)
    # Returns the hash rather than a bool so the caller can burn exactly that entry.
    assert match_backup_code(codes[1], hashes) == hashes[1]


def test_an_unknown_backup_code_matches_nothing() -> None:
    hashes = tuple(hash_backup_code(code) for code in generate_backup_codes(3))
    assert match_backup_code("nope-nope", hashes) is None


def test_matching_against_no_codes_is_safe() -> None:
    assert match_backup_code("anything", ()) is None


def test_seed_seals_and_unseals() -> None:
    seed = generate_seed()
    assert unseal_seed(seal_seed(seed, _KEY), _KEY) == seed


def test_sealing_is_non_deterministic() -> None:
    # A deterministic ciphertext would leak which users share a seed, and more usefully to an
    # attacker, when a seed was rotated.
    seed = generate_seed()
    assert seal_seed(seed, _KEY) != seal_seed(seed, _KEY)


def test_a_sealed_seed_does_not_contain_the_plaintext() -> None:
    seed = generate_seed()
    assert seed not in seal_seed(seed, _KEY)


def test_unsealing_under_the_wrong_key_fails() -> None:
    with pytest.raises(DecryptionError):
        unseal_seed(seal_seed(generate_seed(), _KEY), _OTHER_KEY)


def test_unsealing_a_tampered_token_fails() -> None:
    sealed = seal_seed(generate_seed(), _KEY)
    tampered = sealed[:-4] + ("AAAA" if not sealed.endswith("AAAA") else "BBBB")
    with pytest.raises(DecryptionError):
        unseal_seed(tampered, _KEY)


@pytest.mark.parametrize("garbage", ["", "not-a-token", "!!!!"])
def test_unsealing_garbage_fails(garbage: str) -> None:
    with pytest.raises(DecryptionError):
        unseal_seed(garbage, _KEY)
