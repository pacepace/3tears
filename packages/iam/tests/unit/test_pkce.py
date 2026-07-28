"""PKCE (RFC 7636) verifier/challenge handling."""

from __future__ import annotations

import pytest

from threetears.iam.pkce import (
    S256_METHOD,
    PkceError,
    compute_s256_challenge,
    create_challenge,
    generate_code_verifier,
    validate_code_challenge_method,
    validate_code_verifier,
    verify,
)

# RFC 7636 appendix B's worked example -- the one published pair that pins this
# implementation against the spec rather than against itself.
_RFC_VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
_RFC_CHALLENGE = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_matches_the_rfc_7636_worked_example() -> None:
    assert compute_s256_challenge(_RFC_VERIFIER) == _RFC_CHALLENGE


def test_challenge_is_unpadded_base64url() -> None:
    challenge = compute_s256_challenge(generate_code_verifier())
    assert "=" not in challenge
    assert "+" not in challenge
    assert "/" not in challenge


def test_generated_verifier_satisfies_the_charset_rule() -> None:
    for _ in range(20):
        validate_code_verifier(generate_code_verifier())


def test_generated_verifiers_are_unique() -> None:
    assert len({generate_code_verifier() for _ in range(50)}) == 50


def test_create_challenge_round_trips() -> None:
    challenge = create_challenge()
    assert challenge.code_challenge_method == S256_METHOD
    verify(
        code_verifier=challenge.code_verifier,
        code_challenge=challenge.code_challenge,
        code_challenge_method=challenge.code_challenge_method,
    )


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "short",
        "a" * 42,  # one under the RFC minimum
        "a" * 129,  # one over the RFC maximum
        "has spaces in it" + "a" * 40,
        "has+plus+which+is+reserved" + "a" * 20,
        "has/slash/which/is/reserved" + "a" * 20,
    ],
)
def test_malformed_verifiers_are_rejected(bad: str) -> None:
    with pytest.raises(PkceError):
        validate_code_verifier(bad)


@pytest.mark.parametrize("length", [43, 128])
def test_boundary_lengths_are_accepted(length: int) -> None:
    validate_code_verifier("a" * length)


@pytest.mark.parametrize("method", ["plain", "PLAIN", "s256", "S512", "", "none"])
def test_only_s256_is_accepted(method: str) -> None:
    # "plain" in particular: accepting it means an intercepted authorization request
    # carries everything needed to redeem the code, which is the attack PKCE prevents.
    with pytest.raises(PkceError):
        validate_code_challenge_method(method)


def test_verify_rejects_a_mismatched_verifier() -> None:
    other = generate_code_verifier()
    with pytest.raises(PkceError, match="does not match"):
        verify(code_verifier=other, code_challenge=_RFC_CHALLENGE, code_challenge_method=S256_METHOD)


def test_verify_rejects_plain_even_when_verifier_equals_challenge() -> None:
    # The downgrade attempt: under "plain" these would match. The method pin runs first.
    value = generate_code_verifier()
    with pytest.raises(PkceError, match="unsupported code_challenge_method"):
        verify(code_verifier=value, code_challenge=value, code_challenge_method="plain")


def test_verify_checks_the_method_before_the_verifier_shape() -> None:
    # Ordering matters: a malformed verifier must not be hashed under an unsupported method.
    with pytest.raises(PkceError, match="unsupported code_challenge_method"):
        verify(code_verifier="!!", code_challenge="x", code_challenge_method="plain")


def test_error_never_carries_the_secret_values() -> None:
    verifier = generate_code_verifier()
    with pytest.raises(PkceError) as excinfo:
        verify(code_verifier=verifier, code_challenge="not-the-challenge", code_challenge_method=S256_METHOD)
    assert verifier not in str(excinfo.value)
