"""offline contract tests for the NATS v2 user-JWT minter (platform-auth A).

These pin the encodings/fields the NATS server rejects but an offline JSON decode would accept:
``alg``, base64url-no-pad on all three segments, the signature over ``header.payload`` (v2, not the
v1 payload-only), the ``resp`` shape (ttl in nanoseconds), and the ``issuer_account`` rule. The
signature is verified INDEPENDENTLY via ``cryptography`` Ed25519 over the ``header.payload`` bytes,
decoding the account public key from its nkey -- the strongest offline oracle, catching the classic
"decodes fine, server rejects" v1-signing-input bug.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import nkeys
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from threetears.nats.subject_permissions import PrincipalPermissions
from threetears.nats.user_jwt import account_public_key, generate_account_seed, mint_user_jwt


def _b64url_decode(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def _split(token: str) -> tuple[str, str, str]:
    header, payload, sig = token.split(".")
    return header, payload, sig


def _payload(token: str) -> dict[str, Any]:
    return json.loads(_b64url_decode(_split(token)[1]))


def _header(token: str) -> dict[str, Any]:
    return json.loads(_b64url_decode(_split(token)[0]))


def _account_pub_raw(account_pub: str) -> bytes:
    decoded = base64.b32decode(account_pub + "=" * (-len(account_pub) % 8))
    return decoded[1:33]  # strip 1 prefix byte; 32 key bytes (last 2 = crc16)


def _verify(token: str, account_pub: str) -> None:
    """raise InvalidSignature unless the token signs ``header.payload`` under ``account_pub``."""
    header, payload, sig = _split(token)
    Ed25519PublicKey.from_public_bytes(_account_pub_raw(account_pub)).verify(
        _b64url_decode(sig), f"{header}.{payload}".encode("ascii")
    )


def _user_pub() -> str:
    kp = nkeys.from_seed(nkeys.encode_seed(os.urandom(32), nkeys.PREFIX_BYTE_USER))
    return str(bytes(kp.public_key), "ascii")


def _perms(*, allow_responses: bool = True) -> PrincipalPermissions:
    return PrincipalPermissions(
        publish=("aibots.tools.call", "aibots.hub.handshake"),
        subscribe=("_INBOX_agent_pod_p1.>", "aibots.agents.internal.a1.p1"),
        allow_responses=allow_responses,
        inbox_prefix="_INBOX_agent_pod_p1",
    )


def _mint(**overrides: Any) -> str:
    kwargs: dict[str, Any] = {
        "account_seed": generate_account_seed(),
        "user_public_key": _user_pub(),
        "permissions": _perms(),
        "name": "agent-x",
        "expires_in_seconds": 600,
    }
    kwargs.update(overrides)
    return mint_user_jwt(**kwargs)


class TestUserJwtEncoding:
    def test_header_is_nats_jwt_v2(self) -> None:
        # alg MUST be ed25519-nkey (v2) -- not v1 'ed25519' nor JOSE 'EdDSA'.
        assert _header(_mint()) == {"typ": "JWT", "alg": "ed25519-nkey"}

    def test_all_segments_base64url_without_padding(self) -> None:
        for seg in _split(_mint()):
            assert "=" not in seg  # no padding
            assert "+" not in seg and "/" not in seg  # url-safe alphabet

    def test_signature_verifies_over_header_dot_payload(self) -> None:
        seed = generate_account_seed()
        token = _mint(account_seed=seed)
        _verify(token, account_public_key(seed))  # raises on the v1 payload-only signing bug

    def test_signature_fails_under_a_different_account(self) -> None:
        token = _mint(account_seed=generate_account_seed())
        with pytest.raises(InvalidSignature):
            _verify(token, account_public_key(generate_account_seed()))

    def test_iss_matches_the_signing_account(self) -> None:
        seed = generate_account_seed()
        assert _payload(_mint(account_seed=seed))["iss"] == account_public_key(seed)


class TestUserJwtClaims:
    def test_carries_identity_and_lifetime(self) -> None:
        upub = _user_pub()
        payload = _payload(
            _mint(user_public_key=upub, name="agent-7", expires_in_seconds=300, now=1000)
        )
        assert payload["sub"] == upub
        assert payload["iss"].startswith("A")
        assert payload["iat"] == 1000
        assert payload["exp"] == 1300
        assert payload["name"] == "agent-7"

    def test_nats_claim_carries_the_allow_lists(self) -> None:
        perms = _perms()
        nats = _payload(_mint(permissions=perms))["nats"]
        assert nats["type"] == "user"
        assert nats["version"] == 2
        assert nats["pub"]["allow"] == list(perms.publish)
        assert nats["sub"]["allow"] == list(perms.subscribe)

    def test_resp_present_for_responders_with_nanosecond_ttl(self) -> None:
        nats = _payload(_mint(permissions=_perms(allow_responses=True)))["nats"]
        assert nats["resp"] == {"max": 1, "ttl": 0}
        assert isinstance(nats["resp"]["ttl"], int)  # nanoseconds integer, never seconds/string

    def test_no_resp_for_non_responders(self) -> None:
        assert "resp" not in _payload(_mint(permissions=_perms(allow_responses=False)))["nats"]

    def test_issuer_account_absent_by_default(self) -> None:
        assert "issuer_account" not in _payload(_mint())["nats"]

    def test_issuer_account_set_when_a_signing_key_signs(self) -> None:
        nats = _payload(_mint(issuer_account="AIDENTITYKEY123"))["nats"]
        assert nats["issuer_account"] == "AIDENTITYKEY123"

    def test_audience_sets_account_placement(self) -> None:
        assert _payload(_mint(audience="AIBOTS"))["aud"] == "AIBOTS"

    def test_no_audience_by_default(self) -> None:
        assert "aud" not in _payload(_mint())

    def test_jti_present(self) -> None:
        assert _payload(_mint())["jti"]  # a non-empty claims id
