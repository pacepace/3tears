"""offline contract tests for the NATS auth-callout request/response codecs (platform-auth A).

decode_auth_request reads the server's AuthorizationRequest (signature-unverified -- the $SYS
transport is the trust boundary); mint_auth_response signs the reply with the auth account key.
The response signature is verified INDEPENDENTLY via cryptography Ed25519 over header.payload.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from threetears.nats.auth_callout import decode_auth_request, mint_auth_response
from threetears.nats.user_jwt import account_public_key, generate_account_seed


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def _request_jwt(
    *,
    server_id: str = "NSERVER1",
    user_nkey: str = "UUSER1",
    auth_token: str = "boot-tok",
    client_name: str = "agent-x",
) -> str:
    # the NATS server signs this; decode ignores the signature, so a dummy sig segment is fine.
    payload = {
        "iss": server_id,
        "aud": "nats-authorization-request",
        "nats": {
            "server_id": {"id": server_id, "name": "n1"},
            "user_nkey": user_nkey,
            "connect_opts": {"auth_token": auth_token, "name": client_name},
            "client_info": {"host": "10.0.0.1", "name": client_name},
        },
    }
    header = _b64url(json.dumps({"typ": "JWT", "alg": "ed25519-nkey"}).encode("ascii"))
    body = _b64url(json.dumps(payload).encode("ascii"))
    return f"{header}.{body}.{_b64url(b'serversig')}"


def _account_pub_raw(account_pub: str) -> bytes:
    decoded = base64.b32decode(account_pub + "=" * (-len(account_pub) % 8))
    return decoded[1:33]  # strip 1 prefix byte; 32 key bytes (last 2 = crc16)


def _verify_response(token: str, account_pub: str) -> dict[str, Any]:
    header, payload, sig = token.split(".")
    Ed25519PublicKey.from_public_bytes(_account_pub_raw(account_pub)).verify(
        _b64url_decode(sig), f"{header}.{payload}".encode("ascii")
    )
    return json.loads(_b64url_decode(payload))


class TestDecodeAuthRequest:
    def test_extracts_server_user_and_bootstrap_token(self) -> None:
        req = decode_auth_request(_request_jwt(server_id="NABC", user_nkey="UXYZ", auth_token="tok-1"))
        assert req.server_id_value == "NABC"
        assert req.user_nkey == "UXYZ"
        assert req.bootstrap_token == "tok-1"

    def test_absent_token_is_none(self) -> None:
        assert decode_auth_request(_request_jwt(auth_token="")).bootstrap_token is None

    def test_malformed_jwt_raises(self) -> None:
        with pytest.raises(ValueError):
            decode_auth_request("not-a-jwt")

    def test_missing_nats_claim_raises(self) -> None:
        header = _b64url(json.dumps({"typ": "JWT"}).encode("ascii"))
        body = _b64url(json.dumps({"iss": "x"}).encode("ascii"))
        with pytest.raises(ValueError):
            decode_auth_request(f"{header}.{body}.{_b64url(b'sig')}")


class TestMintAuthResponse:
    def test_admit_carries_user_jwt_and_verifies(self) -> None:
        seed = generate_account_seed()
        token = mint_auth_response(account_seed=seed, server_id="NSRV", user_nkey="UUSR", user_jwt="THE.USER.JWT")
        payload = _verify_response(token, account_public_key(seed))
        assert payload["aud"] == "NSRV"  # must equal the requesting server id
        assert payload["sub"] == "UUSR"  # must equal the server-supplied user nkey
        assert payload["iss"] == account_public_key(seed)
        assert payload["nats"]["jwt"] == "THE.USER.JWT"
        assert payload["nats"]["type"] == "authorization_response"
        assert "error" not in payload["nats"]

    def test_deny_carries_error_and_no_jwt(self) -> None:
        seed = generate_account_seed()
        token = mint_auth_response(
            account_seed=seed, server_id="NSRV", user_nkey="UUSR", error="invalid bootstrap token"
        )
        payload = _verify_response(token, account_public_key(seed))
        assert payload["nats"]["error"] == "invalid bootstrap token"
        assert "jwt" not in payload["nats"]

    def test_requires_exactly_one_of_jwt_or_error(self) -> None:
        seed = generate_account_seed()
        with pytest.raises(ValueError):
            mint_auth_response(account_seed=seed, server_id="N", user_nkey="U")  # neither
        with pytest.raises(ValueError):
            mint_auth_response(account_seed=seed, server_id="N", user_nkey="U", user_jwt="j", error="e")  # both

    def test_signature_fails_under_a_different_account(self) -> None:
        token = mint_auth_response(account_seed=generate_account_seed(), server_id="N", user_nkey="U", user_jwt="j")
        with pytest.raises(InvalidSignature):
            _verify_response(token, account_public_key(generate_account_seed()))
