"""NATS auth-callout request/response codecs (platform-auth A).

The NATS server delegates each connection's authentication: it publishes an AuthorizationRequest
JWT on ``$SYS.REQ.USER.AUTH`` and applies the AuthorizationResponse the Hub replies with. These are
the generic NATS ``jwt/v2`` codecs for that exchange; the responder loop + the principal-resolution
policy (validate the bootstrap token -> which principal + ids) live in the Hub.

The request decode is intentionally signature-UNVERIFIED: the request arrives on the system account,
reachable only by the NATS server itself, so the transport is the trust boundary. The response is
SIGNED with the auth account key, and the server validates it against the configured ``issuer``
(``aud`` must equal the requesting server's id, ``sub`` the server-supplied user nkey).
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any

from threetears.nats.user_jwt import account_public_key, encode_and_sign

__all__ = ["AuthCalloutRequest", "decode_auth_request", "mint_auth_response"]


def _b64url_decode(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


@dataclass(frozen=True, slots=True)
class AuthCalloutRequest:
    """the decoded inbound AuthorizationRequest -- the fields the responder needs.

    :param server_id: the requesting server's id object; its ``id`` is the response ``aud``
    :ptype server_id: dict[str, Any]
    :param user_nkey: the user nkey the SERVER pre-generated; the minted user JWT's ``sub`` MUST equal it
    :ptype user_nkey: str
    :param connect_opts: the client's connect options; ``auth_token`` carries the bootstrap token
    :ptype connect_opts: dict[str, Any]
    :param client_info: the client's connection info (host, name, ...)
    :ptype client_info: dict[str, Any]
    """

    server_id: dict[str, Any]
    user_nkey: str
    connect_opts: dict[str, Any]
    client_info: dict[str, Any]

    @property
    def server_id_value(self) -> str:
        """the server instance id -- the value the response ``aud`` must equal."""
        sid = self.server_id.get("id")
        if not isinstance(sid, str) or not sid:
            raise ValueError("auth request server_id.id missing")
        return sid

    @property
    def bootstrap_token(self) -> str | None:
        """the bootstrap token the client presented (``connect_opts.auth_token``), or ``None``."""
        token = self.connect_opts.get("auth_token")
        return token if isinstance(token, str) and token else None


def decode_auth_request(request_jwt: str) -> AuthCalloutRequest:
    """decode the AuthorizationRequest JWT received on ``$SYS.REQ.USER.AUTH``.

    Signature is NOT verified (the request is reachable only by the NATS server over the system
    account -- the transport is the trust boundary).

    :param request_jwt: the compact AuthorizationRequest JWT
    :ptype request_jwt: str
    :return: the decoded request
    :rtype: AuthCalloutRequest
    :raises ValueError: when the JWT is malformed or missing required fields
    """
    try:
        _, payload_seg, _ = request_jwt.split(".")
        claims = json.loads(_b64url_decode(payload_seg))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed auth request ({type(exc).__name__})") from None
    nats = claims.get("nats") if isinstance(claims, dict) else None
    if not isinstance(nats, dict):
        raise ValueError("auth request missing nats claim")
    server_id = nats.get("server_id")
    user_nkey = nats.get("user_nkey")
    if not isinstance(server_id, dict) or not isinstance(user_nkey, str) or not user_nkey:
        raise ValueError("auth request missing server_id / user_nkey")
    raw_connect = nats.get("connect_opts")
    raw_client = nats.get("client_info")
    return AuthCalloutRequest(
        server_id=server_id,
        user_nkey=user_nkey,
        connect_opts=raw_connect if isinstance(raw_connect, dict) else {},
        client_info=raw_client if isinstance(raw_client, dict) else {},
    )


def mint_auth_response(
    *,
    account_seed: bytes,
    server_id: str,
    user_nkey: str,
    user_jwt: str | None = None,
    error: str | None = None,
    issuer_account: str | None = None,
    now: int | None = None,
) -> str:
    """mint the AuthorizationResponse JWT the responder replies with.

    Exactly one of ``user_jwt`` (admit) or ``error`` (deny) is set. ``aud`` MUST be the requesting
    server's id and ``sub`` MUST be the request's ``user_nkey`` -- both server-enforced.

    :param account_seed: the auth account signing seed (signs the response)
    :ptype account_seed: bytes
    :param server_id: the requesting server's instance id (the response ``aud``)
    :ptype server_id: str
    :param user_nkey: the user nkey from the request (the response ``sub``)
    :ptype user_nkey: str
    :param user_jwt: the minted user JWT to admit the connection with, or ``None`` to deny
    :ptype user_jwt: str | None
    :param error: a denial reason, or ``None`` to admit
    :ptype error: str | None
    :param issuer_account: the account identity key when a signing key (not the identity key) signs
    :ptype issuer_account: str | None
    :param now: unix-seconds issue time; defaults to the current time
    :ptype now: int | None
    :return: the compact signed AuthorizationResponse JWT
    :rtype: str
    :raises ValueError: when neither or both of ``user_jwt``/``error`` are provided
    """
    if (user_jwt is None) == (error is None):
        raise ValueError("exactly one of user_jwt or error must be set")
    issued_at = now if now is not None else int(time.time())
    nats_claim: dict[str, Any] = {"type": "authorization_response", "version": 2}
    if user_jwt is not None:
        nats_claim["jwt"] = user_jwt
    else:
        nats_claim["error"] = error
    if issuer_account is not None:
        nats_claim["issuer_account"] = issuer_account
    payload: dict[str, Any] = {
        "iss": account_public_key(account_seed),
        "aud": server_id,
        "sub": user_nkey,
        "iat": issued_at,
        "nats": nats_claim,
    }
    return encode_and_sign(account_seed=account_seed, payload=payload)
