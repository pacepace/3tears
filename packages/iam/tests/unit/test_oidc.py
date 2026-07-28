"""OIDC discovery, id_token verification, and identity extraction."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from joserfc import jwt as jose_jwt
from joserfc.jwk import KeySet, RSAKey

from threetears.iam.oidc import (
    OidcDiscoveryClient,
    OidcError,
    OidcTokenError,
    coerce_email_verified,
    extract_identity,
    verify_id_token,
)

_ISSUER = "https://idp.example"
_CLIENT_ID = "client-abc"

_SIGNING_KEY = RSAKey.generate_key(2048, parameters={"kid": "k1"})
_JWKS = KeySet([_SIGNING_KEY])
_OTHER_KEY = RSAKey.generate_key(2048, parameters={"kid": "k1"})


def _id_token(**overrides: Any) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": _ISSUER,
        "aud": _CLIENT_ID,
        "sub": "user-123",
        "iat": now,
        "exp": now + 300,
    }
    claims.update(overrides)
    return jose_jwt.encode({"alg": "RS256", "kid": "k1"}, claims, _SIGNING_KEY)


def _discovery_handler(
    *,
    issuer: str = _ISSUER,
    omit: str | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/.well-known/openid-configuration"):
            document = {
                "issuer": issuer,
                "authorization_endpoint": f"{_ISSUER}/authorize",
                "token_endpoint": f"{_ISSUER}/token",
                "jwks_uri": f"{_ISSUER}/jwks",
            }
            if omit is not None:
                document.pop(omit)
            return httpx.Response(200, json=document)
        return httpx.Response(200, json=_JWKS.as_dict())

    return handler


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> OidcDiscoveryClient:
    return OidcDiscoveryClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_discovery_resolves_metadata() -> None:
    metadata = await _client(_discovery_handler()).discover(_ISSUER)
    assert metadata.issuer == _ISSUER
    assert metadata.authorization_endpoint == f"{_ISSUER}/authorize"
    assert metadata.token_endpoint == f"{_ISSUER}/token"


async def test_discovery_rejects_an_issuer_mismatch() -> None:
    # OIDC Discovery 1.0 SS4.3, and load-bearing: without it a compromised endpoint asserts a
    # different issuer and every later check is made against the attacker's value.
    with pytest.raises(OidcError, match="issuer mismatch"):
        await _client(_discovery_handler(issuer="https://attacker.example")).discover(_ISSUER)


async def test_discovery_caches_and_can_be_invalidated() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _discovery_handler()(request)

    client = _client(handler)
    await client.discover(_ISSUER)
    await client.discover(_ISSUER)
    assert len([path for path in calls if "well-known" in path]) == 1

    # The escape hatch for signing-key rotation.
    client.forget(_ISSUER)
    await client.discover(_ISSUER)
    assert len([path for path in calls if "well-known" in path]) == 2


@pytest.mark.parametrize("missing", ["authorization_endpoint", "token_endpoint", "jwks_uri"])
async def test_discovery_rejects_an_incomplete_document(missing: str) -> None:
    with pytest.raises(OidcError):
        await _client(_discovery_handler(omit=missing)).discover(_ISSUER)


async def test_discovery_surfaces_a_transport_failure() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    with pytest.raises(OidcError, match="fetch failed"):
        await _client(handler).discover(_ISSUER)


async def test_discovery_rejects_a_non_json_document() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>nope</html>")

    with pytest.raises(OidcError, match="not valid JSON"):
        await _client(handler).discover(_ISSUER)


def test_valid_id_token_verifies() -> None:
    claims = verify_id_token(_id_token(), jwks=_JWKS, issuer=_ISSUER, audience=_CLIENT_ID)
    assert claims["sub"] == "user-123"


def test_wrong_signing_key_is_a_signature_failure() -> None:
    forged = jose_jwt.encode(
        {"alg": "RS256", "kid": "k1"},
        {"iss": _ISSUER, "aud": _CLIENT_ID, "sub": "u", "exp": int(time.time()) + 300},
        _OTHER_KEY,
    )
    with pytest.raises(OidcTokenError) as excinfo:
        verify_id_token(forged, jwks=_JWKS, issuer=_ISSUER, audience=_CLIENT_ID)
    assert excinfo.value.reason == "signature"


def test_wrong_issuer_is_an_issuer_failure() -> None:
    with pytest.raises(OidcTokenError) as excinfo:
        verify_id_token(_id_token(iss="https://elsewhere.example"), jwks=_JWKS, issuer=_ISSUER, audience=_CLIENT_ID)
    assert excinfo.value.reason == "issuer"


def test_wrong_audience_is_an_audience_failure() -> None:
    # A token minted for a DIFFERENT client of the same provider must not verify here --
    # otherwise a less-trusted sibling client's token is a complete bypass.
    with pytest.raises(OidcTokenError) as excinfo:
        verify_id_token(_id_token(aud="some-other-client"), jwks=_JWKS, issuer=_ISSUER, audience=_CLIENT_ID)
    assert excinfo.value.reason == "audience"


def test_expired_token_is_an_expiry_failure() -> None:
    past = int(time.time()) - 7200
    with pytest.raises(OidcTokenError) as excinfo:
        verify_id_token(_id_token(iat=past, exp=past + 60), jwks=_JWKS, issuer=_ISSUER, audience=_CLIENT_ID)
    assert excinfo.value.reason == "expiry"


@pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512", "none"])
def test_symmetric_and_unsigned_algorithms_are_refused(algorithm: str) -> None:
    # In OIDC the client secret IS the HMAC key, so permitting HS256 means anyone holding
    # that secret can mint a token the provider never issued.
    with pytest.raises(OidcTokenError) as excinfo:
        verify_id_token(_id_token(), jwks=_JWKS, issuer=_ISSUER, audience=_CLIENT_ID, algorithms=[algorithm])
    assert excinfo.value.reason == "configuration"


def test_an_empty_algorithm_list_is_refused() -> None:
    with pytest.raises(OidcTokenError) as excinfo:
        verify_id_token(_id_token(), jwks=_JWKS, issuer=_ISSUER, audience=_CLIENT_ID, algorithms=[])
    assert excinfo.value.reason == "configuration"


def test_a_forbidden_algorithm_anywhere_in_the_list_is_refused() -> None:
    with pytest.raises(OidcTokenError):
        verify_id_token(_id_token(), jwks=_JWKS, issuer=_ISSUER, audience=_CLIENT_ID, algorithms=["RS256", "HS256"])


def test_extract_identity_pulls_the_subject_and_profile() -> None:
    identity = extract_identity({"sub": "user-123", "email": "a@example.com", "email_verified": True, "name": "Ada"})
    assert identity.subject == "user-123"
    assert identity.email == "a@example.com"
    assert identity.email_verified
    assert identity.display_name == "Ada"
    assert identity.claims["sub"] == "user-123"


def test_extract_identity_requires_a_subject() -> None:
    with pytest.raises(OidcTokenError, match="sub"):
        extract_identity({"email": "a@example.com"})


def test_extract_identity_defaults_email_verified_to_false() -> None:
    assert not extract_identity({"sub": "u", "email": "a@example.com"}).email_verified


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        (" true ", True),
        # The inversion this function exists to prevent: a naive truthiness test reads the
        # STRING "false" as verified.
        ("false", False),
        ("FALSE", False),
        ("", False),
        (None, False),
        (1, False),
        ("yes", False),
    ],
)
def test_coerce_email_verified(raw: object, expected: bool) -> None:
    assert coerce_email_verified(raw) is expected
