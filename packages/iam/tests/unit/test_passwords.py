"""Password hashing, verification, and set-time policy."""

from __future__ import annotations

import unicodedata

import bcrypt
import pytest

from threetears.iam.breach import BreachCorpus
from threetears.iam.passwords import (
    DEFAULT_MAX_PASSWORD_LENGTH,
    DEFAULT_MIN_PASSWORD_LENGTH,
    PasswordPolicyError,
    equalize_verify_cost,
    equalize_verify_cost_async,
    hash_password,
    hash_password_async,
    normalize_password,
    resolve_password_length_bounds,
    validate_new_password,
    verify_password,
    verify_password_async,
)

_GOOD = "a-perfectly-reasonable-passphrase"


def test_hash_then_verify_round_trips() -> None:
    assert verify_password(_GOOD, hash_password(_GOOD)).matched


def test_hash_is_argon2id_and_salted() -> None:
    first = hash_password(_GOOD)
    assert first.startswith("$argon2id$")
    # Two hashes of the same password must differ: a shared salt would make the store
    # rainbow-table-able and would leak which users share a password.
    assert first != hash_password(_GOOD)


def test_wrong_password_does_not_verify() -> None:
    assert not verify_password("something else entirely", hash_password(_GOOD)).matched


def test_nfkc_normalization_is_applied_on_both_sides() -> None:
    # The same passphrase in composed (NFC) and decomposed (NFD) form. Without
    # normalization these are different byte sequences and one cannot verify the other.
    composed = unicodedata.normalize("NFC", "café-passphrase-that-is-long")
    decomposed = unicodedata.normalize("NFD", "café-passphrase-that-is-long")
    assert composed != decomposed
    assert verify_password(decomposed, hash_password(composed)).matched


def test_normalize_password_is_nfkc() -> None:
    # U+FB01 LATIN SMALL LIGATURE FI folds to "fi" under NFKC.
    assert normalize_password("ﬁ") == "fi"


@pytest.mark.parametrize(
    "stored",
    [
        "",
        "not-a-hash",
        "$argon2id$truncated",
        "$2b$not-really-bcrypt",
        "$unknown$scheme$xyz",
    ],
)
def test_malformed_stored_hash_is_a_failure_not_an_exception(stored: str) -> None:
    # A corrupt stored hash is an auth failure, not a 500 -- and never an exception-shaped
    # side channel distinguishing it from an ordinary wrong password.
    assert not verify_password(_GOOD, stored).matched


def test_bcrypt_hash_verifies_and_yields_an_argon2_upgrade() -> None:
    legacy = bcrypt.hashpw(normalize_password(_GOOD).encode("utf-8"), bcrypt.gensalt(rounds=4)).decode("utf-8")
    result = verify_password(_GOOD, legacy)
    assert result.matched
    assert result.upgraded_hash is not None
    assert result.upgraded_hash.startswith("$argon2id$")
    # The upgrade must be a usable replacement for the credential it replaces.
    assert verify_password(_GOOD, result.upgraded_hash).matched


def test_bcrypt_mismatch_yields_no_upgrade() -> None:
    legacy = bcrypt.hashpw(b"the-real-one", bcrypt.gensalt(rounds=4)).decode("utf-8")
    result = verify_password("not-the-real-one", legacy)
    assert not result.matched
    assert result.upgraded_hash is None


def test_argon2_hash_yields_no_upgrade() -> None:
    # Already on the current scheme -- nothing for the caller to write back.
    assert verify_password(_GOOD, hash_password(_GOOD)).upgraded_hash is None


def test_length_bounds_default_to_the_floors() -> None:
    assert resolve_password_length_bounds(None) == (DEFAULT_MIN_PASSWORD_LENGTH, DEFAULT_MAX_PASSWORD_LENGTH)
    assert resolve_password_length_bounds({}) == (DEFAULT_MIN_PASSWORD_LENGTH, DEFAULT_MAX_PASSWORD_LENGTH)


def test_length_bounds_may_be_raised_but_never_lowered() -> None:
    assert resolve_password_length_bounds({"min_length": 32, "max_length": 128}) == (32, 128)
    # Below the floor, both clamp back up: a caller cannot weaken the platform policy.
    assert resolve_password_length_bounds({"min_length": 4, "max_length": 8}) == (
        DEFAULT_MIN_PASSWORD_LENGTH,
        DEFAULT_MAX_PASSWORD_LENGTH,
    )


def test_length_bounds_ignore_non_integer_config() -> None:
    assert resolve_password_length_bounds({"min_length": "32", "max_length": None}) == (
        DEFAULT_MIN_PASSWORD_LENGTH,
        DEFAULT_MAX_PASSWORD_LENGTH,
    )


def test_validate_rejects_too_short_and_too_long() -> None:
    with pytest.raises(PasswordPolicyError, match="at least"):
        validate_new_password("x" * (DEFAULT_MIN_PASSWORD_LENGTH - 1))
    with pytest.raises(PasswordPolicyError, match="at most"):
        validate_new_password("x" * (DEFAULT_MAX_PASSWORD_LENGTH + 1))


def test_validate_measures_length_after_normalization() -> None:
    # Sixteen ligatures normalize to thirty-two characters, clearing a 16-character floor
    # that the pre-normalization string would also have cleared -- but the hash is computed
    # on the normalized form, so the normalized length is the one that must be measured.
    candidate = "ﬁ" * (DEFAULT_MIN_PASSWORD_LENGTH // 2)
    assert len(candidate) < DEFAULT_MIN_PASSWORD_LENGTH
    assert len(normalize_password(candidate)) == DEFAULT_MIN_PASSWORD_LENGTH
    validate_new_password(candidate)


def test_validate_screens_the_breach_corpus() -> None:
    breached = "correct-horse-battery-staple"
    corpus = BreachCorpus(seed_passwords=[breached])
    with pytest.raises(PasswordPolicyError) as excinfo:
        validate_new_password(breached, breach_corpus=corpus)
    # The message must not confirm corpus membership specifically -- doing so turns this
    # call into an oracle an attacker can query against the corpus.
    message = str(excinfo.value).lower()
    assert "breach" not in message
    assert "pwned" not in message


def test_validate_without_a_corpus_skips_screening() -> None:
    validate_new_password("correct-horse-battery-staple")


def test_validate_accepts_a_compliant_password() -> None:
    validate_new_password(_GOOD, breach_corpus=BreachCorpus(seed_passwords=["something-else"]))


def test_equalize_verify_cost_swallows_the_expected_mismatch() -> None:
    # It is called on the no-such-user branch, where raising would turn a normal login
    # miss into a 500 -- and would itself be the enumeration signal it exists to remove.
    equalize_verify_cost()


async def test_async_wrappers_match_their_sync_counterparts() -> None:
    stored = await hash_password_async(_GOOD)
    assert stored.startswith("$argon2id$")
    assert (await verify_password_async(_GOOD, stored)).matched
    assert not (await verify_password_async("wrong", stored)).matched
    await equalize_verify_cost_async()
