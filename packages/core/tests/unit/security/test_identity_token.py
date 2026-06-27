"""Hub-issued identity token: EdDSA-signed compact JWS the registry verifies before RBAC.

The contract the platform-auth layer depends on, and that these tests pin:

- a token round-trips ONLY under a key present in the JWKS (right key in, same claims out);
- the algorithm is PINNED to EdDSA, so an ``alg=none`` token or an HMAC-confusion forgery
  (HS256 signed with the public key bytes) is rejected, never accepted;
- an expired / tampered / wrong-key / wrong-issuer / missing-or-wrong-kid token is REJECTED
  with :class:`IdentityTokenError`, never silently parsed into a trusted identity;
- clock skew is tolerated within an explicit leeway, and no further;
- unknown FUTURE claims are tolerated (forward-compat for a receiver-first rollout) while
  the required identity claims must be present;
- the error never echoes the token string back (no token material in logs).
"""

from __future__ import annotations

import time
from typing import Any

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from jwt.algorithms import ECAlgorithm

from threetears.core.security.identity_token import (
    IdentityClaims,
    IdentityTokenError,
    build_jwks,
    canonical_call_hash,
    generate_signing_keypair,
    sign_identity_token,
    verify_identity_token,
)

_ISS = "hub"


def _claims(*, exp_delta: int = 600, iat_delta: int = 0, **over: object) -> IdentityClaims:
    now = int(time.time())
    base: dict[str, object] = {
        "sub": "0190a000-0000-7000-8000-000000000001",
        "customer_id": "0190a000-0000-7000-8000-0000000000c1",
        "user_id": "0190a000-0000-7000-8000-0000000000a1",
        "sid": "session-1",
        "pod_id": "pod-7",
        "iss": _ISS,
        "iat": now + iat_delta,
        "exp": now + exp_delta,
    }
    base.update(over)
    return IdentityClaims(**base)  # type: ignore[arg-type]


@pytest.fixture
def keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    return generate_signing_keypair()


@pytest.fixture
def signed(keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey]) -> tuple[str, dict[str, Any]]:
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    token = sign_identity_token(_claims(), signing_key=priv, kid="kid-1")
    return token, jwks


def test_round_trips_under_a_key_in_the_jwks(signed: tuple[str, dict[str, Any]]) -> None:
    token, jwks = signed
    claims = verify_identity_token(token, jwks=jwks, issuer=_ISS)
    assert claims.sub == "0190a000-0000-7000-8000-000000000001"
    assert claims.customer_id == "0190a000-0000-7000-8000-0000000000c1"
    assert claims.user_id == "0190a000-0000-7000-8000-0000000000a1"
    assert claims.sid == "session-1"
    assert claims.pod_id == "pod-7"
    assert claims.iss == _ISS


def test_tampered_token_is_rejected(signed: tuple[str, dict[str, Any]]) -> None:
    token, jwks = signed
    # flip a char in the signature segment
    head, payload, sig = token.split(".")
    tampered = f"{head}.{payload}.{sig[:-2]}{'AA' if sig[-2:] != 'AA' else 'BB'}"
    with pytest.raises(IdentityTokenError):
        verify_identity_token(tampered, jwks=jwks, issuer=_ISS)


def test_wrong_key_for_kid_is_rejected(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    priv, _pub = keypair
    token = sign_identity_token(_claims(), signing_key=priv, kid="kid-1")
    # a JWKS that maps kid-1 to a DIFFERENT public key
    _other_priv, other_pub = generate_signing_keypair()
    wrong_jwks = build_jwks({"kid-1": other_pub})
    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks=wrong_jwks, issuer=_ISS)


def test_expired_token_is_rejected(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    token = sign_identity_token(_claims(exp_delta=-10), signing_key=priv, kid="kid-1")
    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks=jwks, issuer=_ISS)


def test_leeway_tolerates_small_skew_but_not_large(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    token = sign_identity_token(_claims(exp_delta=-5), signing_key=priv, kid="kid-1")
    # within leeway -> accepted
    claims = verify_identity_token(token, jwks=jwks, issuer=_ISS, leeway_seconds=30)
    assert claims.pod_id == "pod-7"
    # outside leeway -> rejected
    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks=jwks, issuer=_ISS, leeway_seconds=1)


def test_alg_none_forgery_is_rejected(signed: tuple[str, dict[str, Any]]) -> None:
    _token, jwks = signed
    now = int(time.time())
    payload = {
        "sub": "attacker",
        "customer_id": "c",
        "sid": "s",
        "pod_id": "p",
        "iss": _ISS,
        "iat": now,
        "exp": now + 600,
    }
    # an unsigned alg=none token carrying the same kid
    forged = pyjwt.encode(payload, key=None, algorithm="none", headers={"kid": "kid-1"})  # type: ignore[arg-type]
    with pytest.raises(IdentityTokenError):
        verify_identity_token(forged, jwks=jwks, issuer=_ISS)


def test_hmac_confusion_forgery_is_rejected(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    # classic alg-confusion: sign HS256 using the Ed25519 PUBLIC key bytes as the HMAC
    # secret. A verifier that doesn't pin the alg would treat the public key as a shared
    # secret and accept it. EdDSA pinning must reject.
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    pub_raw = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
    now = int(time.time())
    payload = {
        "sub": "attacker",
        "customer_id": "c",
        "sid": "s",
        "pod_id": "p",
        "iss": _ISS,
        "iat": now,
        "exp": now + 600,
    }
    forged = pyjwt.encode(payload, key=pub_raw, algorithm="HS256", headers={"kid": "kid-1"})
    with pytest.raises(IdentityTokenError):
        verify_identity_token(forged, jwks=jwks, issuer=_ISS)


def test_wrong_issuer_is_rejected(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    token = sign_identity_token(_claims(iss="evil"), signing_key=priv, kid="kid-1")
    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks=jwks, issuer=_ISS)


def test_missing_kid_is_rejected(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    now = int(time.time())
    payload = {"sub": "a", "customer_id": "c", "sid": "s", "pod_id": "p", "iss": _ISS, "iat": now, "exp": now + 600}
    token = pyjwt.encode(payload, key=priv, algorithm="EdDSA")  # no kid header
    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks=jwks, issuer=_ISS)


def test_kid_absent_from_jwks_is_rejected(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    priv, pub = keypair
    token = sign_identity_token(_claims(), signing_key=priv, kid="kid-1")
    jwks = build_jwks({"kid-2": pub})  # token's kid not present
    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks=jwks, issuer=_ISS)


@pytest.mark.parametrize("missing", ["iss", "sub", "customer_id", "sid", "pod_id", "iat", "exp"])
def test_missing_required_claim_is_rejected(keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey], missing: str) -> None:
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    now = int(time.time())
    payload: dict[str, object] = {
        "sub": "a",
        "customer_id": "c",
        "sid": "s",
        "pod_id": "p",
        "iss": _ISS,
        "iat": now,
        "exp": now + 600,
    }
    del payload[missing]
    token = pyjwt.encode(payload, key=priv, algorithm="EdDSA", headers={"kid": "kid-1"})
    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks=jwks, issuer=_ISS)


@pytest.mark.parametrize(
    ("claim", "bad"),
    [("sub", ""), ("customer_id", None), ("sid", ""), ("pod_id", None), ("customer_id", "")],
)
def test_empty_or_null_identity_claim_is_rejected(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey], claim: str, bad: object
) -> None:
    # `require` accepts present-but-empty/null; the verifier must still reject it so RBAC never
    # sees customer_id="" / None as a trusted identity.
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    now = int(time.time())
    payload: dict[str, object] = {
        "sub": "a",
        "customer_id": "c",
        "sid": "s",
        "pod_id": "p",
        "iss": _ISS,
        "iat": now,
        "exp": now + 600,
    }
    payload[claim] = bad
    token = pyjwt.encode(payload, key=priv, algorithm="EdDSA", headers={"kid": "kid-1"})
    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks=jwks, issuer=_ISS)


def test_build_jwks_rejects_a_private_key(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    # the maximal-blast-radius guard: a private key must never reach the JWKS (it would emit
    # the private scalar `d`). The type hint is the contract; this is the runtime backstop.
    priv, _pub = keypair
    with pytest.raises(IdentityTokenError):
        build_jwks({"kid-1": priv})  # type: ignore[dict-item]


def test_jwks_entry_that_is_not_ed25519_is_rejected(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    # kid matches but the key is an EC key, not Ed25519 -> reject (don't verify against it).
    priv, _pub = keypair
    token = sign_identity_token(_claims(), signing_key=priv, kid="kid-1")
    ec_pub = ec.generate_private_key(ec.SECP256R1()).public_key()
    ec_jwk = ECAlgorithm.to_jwk(ec_pub, as_dict=True)
    ec_jwk["kid"] = "kid-1"
    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks={"keys": [ec_jwk]}, issuer=_ISS)


@pytest.mark.parametrize("bad_jwks", [{"keys": []}, {"no_keys": 1}, []])
def test_malformed_or_empty_jwks_is_rejected(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey], bad_jwks: object
) -> None:
    priv, _pub = keypair
    token = sign_identity_token(_claims(), signing_key=priv, kid="kid-1")
    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks=bad_jwks, issuer=_ISS)  # type: ignore[arg-type]


def test_unknown_future_claims_are_tolerated(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    # forward-compat: a future minter adds a claim; today's verifier must still accept it.
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    now = int(time.time())
    payload = {
        "sub": "a",
        "customer_id": "c",
        "user_id": "u",
        "sid": "s",
        "pod_id": "p",
        "iss": _ISS,
        "iat": now,
        "exp": now + 600,
        "future_field": "tolerated",
    }
    token = pyjwt.encode(payload, key=priv, algorithm="EdDSA", headers={"kid": "kid-1"})
    claims = verify_identity_token(token, jwks=jwks, issuer=_ISS)
    assert claims.sub == "a"


def test_user_id_is_optional(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    token = sign_identity_token(_claims(user_id=None), signing_key=priv, kid="kid-1")
    claims = verify_identity_token(token, jwks=jwks, issuer=_ISS)
    assert claims.user_id is None


def test_multi_kid_jwks_supports_overlap_window_rotation(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    # rotation runs two valid kids at once; a token signed by the NEW key verifies against a
    # JWKS that still also publishes the OLD key.
    old_priv, old_pub = keypair
    new_priv, new_pub = generate_signing_keypair()
    jwks = build_jwks({"kid-old": old_pub, "kid-new": new_pub})
    token = sign_identity_token(_claims(), signing_key=new_priv, kid="kid-new")
    claims = verify_identity_token(token, jwks=jwks, issuer=_ISS)
    assert claims.sub == "0190a000-0000-7000-8000-000000000001"


def test_build_jwks_emits_okp_ed25519_entries(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    _priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    assert list(jwks.keys()) == ["keys"]
    (entry,) = jwks["keys"]
    assert entry["kty"] == "OKP"
    assert entry["crv"] == "Ed25519"
    assert entry["kid"] == "kid-1"
    assert entry["alg"] == "EdDSA"
    assert entry["use"] == "sig"
    assert entry["x"]  # the public key, base64url, present
    assert "d" not in entry  # NEVER the private scalar


def test_garbage_token_raises_identity_error_not_a_crash(signed: tuple[str, dict[str, Any]]) -> None:
    _token, jwks = signed
    for junk in ("", "not-a-jwt", "a.b", "a.b.c.d"):
        with pytest.raises(IdentityTokenError):
            verify_identity_token(junk, jwks=jwks, issuer=_ISS)


def test_error_does_not_echo_the_token(signed: tuple[str, dict[str, Any]]) -> None:
    token, jwks = signed
    bad = build_jwks({"kid-2": generate_signing_keypair()[1]})  # kid mismatch
    with pytest.raises(IdentityTokenError) as exc:
        verify_identity_token(token, jwks=bad, issuer=_ISS)
    assert token not in str(exc.value)


def test_cnf_holder_key_thumbprint_round_trips(
    keypair: tuple[Ed25519PrivateKey, Ed25519PublicKey],
) -> None:
    priv, pub = keypair
    jwks = build_jwks({"kid-1": pub})
    token = sign_identity_token(_claims(cnf="thumb-print-jkt"), signing_key=priv, kid="kid-1")
    assert verify_identity_token(token, jwks=jwks, issuer=_ISS).cnf == "thumb-print-jkt"


def test_cnf_is_none_when_absent(signed: tuple[str, dict[str, Any]]) -> None:
    token, jwks = signed
    assert verify_identity_token(token, jwks=jwks, issuer=_ISS).cnf is None


def test_canonical_call_hash_is_arg_order_independent() -> None:
    a = canonical_call_hash("pentest.nmap", {"target": "x", "ports": "1-1024"}, "corr-1")
    b = canonical_call_hash("pentest.nmap", {"ports": "1-1024", "target": "x"}, "corr-1")
    assert a == b  # argument key order must not change the body hash


def test_canonical_call_hash_changes_with_any_field() -> None:
    base = canonical_call_hash("t", {"a": 1}, "c")
    assert canonical_call_hash("t2", {"a": 1}, "c") != base  # tool_name
    assert canonical_call_hash("t", {"a": 2}, "c") != base  # arguments
    assert canonical_call_hash("t", {"a": 1}, "c2") != base  # correlation_id
    assert canonical_call_hash("t", {"a": 1}, None) != base  # None vs a value


def test_canonical_call_hash_is_unpadded_base64url() -> None:
    h = canonical_call_hash("t", {}, None)
    assert "=" not in h and "+" not in h and "/" not in h
    assert len(h) == 43  # SHA-256 (32 bytes) -> 43 base64url chars, unpadded
