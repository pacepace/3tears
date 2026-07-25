"""Session token claims, signing, and verification."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from threetears.core.security.identity_token import build_jwks
from threetears.iam.tokens import (
    Ed25519JwksVerifier,
    Ed25519Signer,
    HmacSigner,
    HmacVerifier,
    SessionClaims,
    TokenError,
    TokenType,
    mint_session_token,
    mint_token_pair,
    new_session_id,
    verify_session_token,
)

_ISSUER = "https://issuer.example"
_AUDIENCE = "platform:internal"
_SECRET = "a" * 48


@pytest.fixture
def signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture
def ed_signer(signing_key: Ed25519PrivateKey) -> Ed25519Signer:
    return Ed25519Signer(signing_key, kid="key-1")


@pytest.fixture
def ed_verifier(signing_key: Ed25519PrivateKey) -> Ed25519JwksVerifier:
    return Ed25519JwksVerifier(
        jwks=build_jwks({"key-1": signing_key.public_key()}),
        issuer=_ISSUER,
        audience=_AUDIENCE,
    )


def _claims(**overrides: object) -> SessionClaims:
    now = int(datetime.now(UTC).timestamp())
    base: dict[str, object] = {
        "sub": "user-1",
        "sid": new_session_id(),
        "jti": "token-1",
        "iss": _ISSUER,
        "aud": (_AUDIENCE,),
        "iat": now,
        "exp": now + 900,
        "type": TokenType.ACCESS,
        "auth_time": now,
        "step_up_window": 300,
        "session_started_at": now,
    }
    base.update(overrides)
    return SessionClaims(**base)  # type: ignore[arg-type]


def _wire_payload(claims: SessionClaims, **extra: object) -> dict[str, object]:
    """The on-the-wire claim mapping, for tests that must forge a token by hand."""
    return {
        "sub": claims.sub,
        "sid": claims.sid,
        "jti": claims.jti,
        "iss": claims.iss,
        "aud": claims.aud[0],
        "iat": claims.iat,
        "exp": claims.exp,
        "type": claims.type.value,
        "auth_time": claims.auth_time,
        "step_up_window": claims.step_up_window,
        "session_started_at": claims.session_started_at,
        **extra,
    }


def test_eddsa_round_trip(ed_signer: Ed25519Signer, ed_verifier: Ed25519JwksVerifier) -> None:
    claims = _claims(customer_id="tenant-1")
    verified = verify_session_token(mint_session_token(claims, signer=ed_signer), verifier=ed_verifier)
    assert verified.sub == "user-1"
    assert verified.customer_id == "tenant-1"
    assert verified.aud == (_AUDIENCE,)
    assert verified.type is TokenType.ACCESS


def test_hs256_round_trip() -> None:
    signer = HmacSigner(_SECRET)
    verifier = HmacVerifier(_SECRET, issuer=_ISSUER, audience=_AUDIENCE)
    verified = verify_session_token(mint_session_token(_claims(), signer=signer), verifier=verifier)
    assert verified.sub == "user-1"


def test_both_schemes_produce_the_same_claims(ed_signer: Ed25519Signer, ed_verifier: Ed25519JwksVerifier) -> None:
    # The signing choice must change the signature and nothing else, or the two paths drift
    # into different security postures.
    claims = _claims(customer_id="tenant-1", act="admin-1", act_reason="support")
    hmac_verifier = HmacVerifier(_SECRET, issuer=_ISSUER, audience=_AUDIENCE)
    from_ed = verify_session_token(mint_session_token(claims, signer=ed_signer), verifier=ed_verifier)
    from_hs = verify_session_token(mint_session_token(claims, signer=HmacSigner(_SECRET)), verifier=hmac_verifier)
    assert from_ed == from_hs


def test_impersonation_claims_survive_the_round_trip(
    ed_signer: Ed25519Signer, ed_verifier: Ed25519JwksVerifier
) -> None:
    claims = _claims(act="admin-1", act_reason="impersonation", act_restriction="view")
    verified = verify_session_token(mint_session_token(claims, signer=ed_signer), verifier=ed_verifier)
    assert verified.is_impersonation
    assert verified.act == "admin-1"
    assert verified.act_restriction == "view"
    # sub stays the impersonated subject so downstream checks see the session as the user does.
    assert verified.sub == "user-1"


def test_a_plain_token_is_not_an_impersonation(ed_signer: Ed25519Signer, ed_verifier: Ed25519JwksVerifier) -> None:
    verified = verify_session_token(mint_session_token(_claims(), signer=ed_signer), verifier=ed_verifier)
    assert not verified.is_impersonation
    assert verified.act is None


def test_dpop_binding_survives_the_round_trip(ed_signer: Ed25519Signer, ed_verifier: Ed25519JwksVerifier) -> None:
    verified = verify_session_token(
        mint_session_token(_claims(cnf="thumbprint-value"), signer=ed_signer), verifier=ed_verifier
    )
    assert verified.cnf == "thumbprint-value"


def test_tampered_token_is_rejected(ed_signer: Ed25519Signer, ed_verifier: Ed25519JwksVerifier) -> None:
    token = mint_session_token(_claims(), signer=ed_signer)
    header, payload, signature = token.split(".")
    with pytest.raises(TokenError):
        verify_session_token(f"{header}.{payload}.{signature[:-4]}AAAA", verifier=ed_verifier)


def test_expired_token_is_rejected(ed_signer: Ed25519Signer, ed_verifier: Ed25519JwksVerifier) -> None:
    past = int((datetime.now(UTC) - timedelta(hours=2)).timestamp())
    with pytest.raises(TokenError):
        verify_session_token(
            mint_session_token(_claims(iat=past, exp=past + 60), signer=ed_signer), verifier=ed_verifier
        )


def test_wrong_issuer_is_rejected(ed_signer: Ed25519Signer, signing_key: Ed25519PrivateKey) -> None:
    verifier = Ed25519JwksVerifier(
        jwks=build_jwks({"key-1": signing_key.public_key()}),
        issuer="https://somewhere-else.example",
        audience=_AUDIENCE,
    )
    with pytest.raises(TokenError):
        verify_session_token(mint_session_token(_claims(), signer=ed_signer), verifier=verifier)


def test_wrong_audience_is_rejected(ed_signer: Ed25519Signer, signing_key: Ed25519PrivateKey) -> None:
    # The check that stops a token which has legitimately left one trust boundary from being
    # accepted at another.
    verifier = Ed25519JwksVerifier(
        jwks=build_jwks({"key-1": signing_key.public_key()}),
        issuer=_ISSUER,
        audience="platform:external",
    )
    with pytest.raises(TokenError):
        verify_session_token(mint_session_token(_claims(), signer=ed_signer), verifier=verifier)


def test_unknown_kid_is_rejected(ed_signer: Ed25519Signer) -> None:
    other = Ed25519PrivateKey.generate()
    verifier = Ed25519JwksVerifier(jwks=build_jwks({"key-2": other.public_key()}), issuer=_ISSUER, audience=_AUDIENCE)
    with pytest.raises(TokenError, match="no JWKS key matches"):
        verify_session_token(mint_session_token(_claims(), signer=ed_signer), verifier=verifier)


def test_empty_jwks_fails_closed(ed_signer: Ed25519Signer) -> None:
    # The normal state of a verifier whose key set has not warmed yet: reject, and say so.
    verifier = Ed25519JwksVerifier(jwks={"keys": []}, issuer=_ISSUER, audience=_AUDIENCE)
    with pytest.raises(TokenError, match="no keys"):
        verify_session_token(mint_session_token(_claims(), signer=ed_signer), verifier=verifier)


def test_alg_none_is_rejected(ed_verifier: Ed25519JwksVerifier) -> None:
    # The classic unsigned-token attack.
    unsigned = jwt.encode(_wire_payload(_claims()), key="", algorithm="none")
    with pytest.raises(TokenError, match="only EdDSA is accepted"):
        verify_session_token(unsigned, verifier=ed_verifier)


def test_hs256_token_is_rejected_by_the_eddsa_verifier(ed_verifier: Ed25519JwksVerifier) -> None:
    # Algorithm confusion: the header must not get to choose how verification happens.
    token = mint_session_token(_claims(), signer=HmacSigner(_SECRET))
    with pytest.raises(TokenError, match="only EdDSA is accepted"):
        verify_session_token(token, verifier=ed_verifier)


def test_eddsa_token_is_rejected_by_the_hs256_verifier(ed_signer: Ed25519Signer) -> None:
    verifier = HmacVerifier(_SECRET, issuer=_ISSUER, audience=_AUDIENCE)
    with pytest.raises(TokenError, match="only HS256 is accepted"):
        verify_session_token(mint_session_token(_claims(), signer=ed_signer), verifier=verifier)


def test_a_smuggled_role_claim_is_rejected(signing_key: Ed25519PrivateKey, ed_verifier: Ed25519JwksVerifier) -> None:
    # The invariant that makes "identity only" enforceable rather than aspirational.
    payload = _wire_payload(_claims(), role="platform-admin")
    smuggled = jwt.encode(payload, key=signing_key, algorithm="EdDSA", headers={"kid": "key-1"})
    with pytest.raises(TokenError, match="unexpected claims"):
        verify_session_token(smuggled, verifier=ed_verifier)


def test_type_confusion_is_rejected(ed_signer: Ed25519Signer, ed_verifier: Ed25519JwksVerifier) -> None:
    # A refresh token accepted where an access token was expected is an escalation: refresh
    # tokens live far longer and are handled far more casually.
    refresh = mint_session_token(_claims(type=TokenType.REFRESH), signer=ed_signer)
    with pytest.raises(TokenError, match="expected a access token"):
        verify_session_token(refresh, verifier=ed_verifier, expected_type=TokenType.ACCESS)


def test_mint_rejects_an_empty_audience(ed_signer: Ed25519Signer) -> None:
    # An empty audience encodes as `"aud": []`, which no verifier can ever match -- a token
    # broken on arrival. Refuse to mint it rather than shipping it.
    with pytest.raises(TokenError, match="audience"):
        mint_session_token(_claims(aud=()), signer=ed_signer)


def test_signers_reject_empty_key_material(signing_key: Ed25519PrivateKey) -> None:
    with pytest.raises(TokenError, match="kid"):
        Ed25519Signer(signing_key, kid="")
    with pytest.raises(TokenError, match="secret"):
        HmacSigner("")


def test_token_pair_shares_a_session_and_differs_by_type(
    ed_signer: Ed25519Signer, ed_verifier: Ed25519JwksVerifier
) -> None:
    pair = mint_token_pair(
        subject="user-1",
        session_id=new_session_id(),
        issuer=_ISSUER,
        audience=_AUDIENCE,
        signer=ed_signer,
        step_up_window=300,
    )
    access = verify_session_token(pair.access_token, verifier=ed_verifier, expected_type=TokenType.ACCESS)
    refresh = verify_session_token(pair.refresh_token, verifier=ed_verifier, expected_type=TokenType.REFRESH)
    assert access.sid == refresh.sid
    assert access.auth_time == refresh.auth_time
    assert access.session_started_at == refresh.session_started_at
    # Distinct jti: rotation marks the refresh token used, which needs its own identifier.
    assert access.jti != refresh.jti
    assert refresh.exp > access.exp


def test_token_pair_defaults_auth_time_and_session_start_to_now(ed_signer: Ed25519Signer) -> None:
    pair = mint_token_pair(
        subject="user-1", session_id=new_session_id(), issuer=_ISSUER, audience=_AUDIENCE, signer=ed_signer
    )
    assert pair.claims.auth_time == pair.claims.iat
    assert pair.claims.session_started_at == pair.claims.iat


def test_token_pair_carries_a_supplied_auth_time_forward(ed_signer: Ed25519Signer) -> None:
    # What a rotation must do: a refresh moves iat and must NOT move auth_time, or every
    # refresh silently re-satisfies step-up.
    original = int(datetime.now(UTC).timestamp()) - 3600
    pair = mint_token_pair(
        subject="user-1",
        session_id=new_session_id(),
        issuer=_ISSUER,
        audience=_AUDIENCE,
        signer=ed_signer,
        auth_time=original,
        session_started_at=original,
    )
    assert pair.claims.auth_time == original
    assert pair.claims.iat > original


def test_multiple_audiences_encode_as_a_list(ed_signer: Ed25519Signer, signing_key: Ed25519PrivateKey) -> None:
    verifier = Ed25519JwksVerifier(
        jwks=build_jwks({"key-1": signing_key.public_key()}),
        issuer=_ISSUER,
        audience=["platform:internal", "platform:external"],
    )
    token = mint_session_token(_claims(aud=("platform:internal", "platform:external")), signer=ed_signer)
    assert verify_session_token(token, verifier=verifier).aud == ("platform:internal", "platform:external")


def test_error_never_leaks_the_token(ed_signer: Ed25519Signer, signing_key: Ed25519PrivateKey) -> None:
    verifier = Ed25519JwksVerifier(
        jwks=build_jwks({"key-1": signing_key.public_key()}), issuer="other", audience=_AUDIENCE
    )
    token = mint_session_token(_claims(), signer=ed_signer)
    with pytest.raises(TokenError) as excinfo:
        verify_session_token(token, verifier=verifier)
    assert token not in str(excinfo.value)
