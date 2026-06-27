"""Mint NATS v2 user JWTs from a per-principal subject-permission allow-list (platform-auth A).

The Hub's auth-callout responder calls :func:`mint_user_jwt` to issue a connecting principal's NATS
user JWT -- the credential the NATS server applies to scope the connection's pub/sub. The format is
the NATS ``jwt/v2`` wire spec (NOT JOSE): header ``{"typ":"JWT","alg":"ed25519-nkey"}``, claims
signed with an ACCOUNT nkey over ``base64url(header).base64url(payload)``, every segment base64url
WITHOUT padding.

There is no official Python NATS-JWT library (upstream minting is Go-only), so this is a small,
audited hand-roll on the official ``nkeys`` Ed25519 primitive. The fields + encodings the NATS server
rejects but an offline JSON decode would accept are pinned by tests:

- ``alg`` is ``ed25519-nkey`` (v2), never ``ed25519`` (v1, rejected) nor JOSE ``EdDSA``;
- all three segments are base64url with NO padding;
- the signature is over the ASCII ``header.payload`` (v2), not payload-only (v1);
- ``resp`` is ``{"max":int,"ttl":int}`` with ttl in NANOSECONDS (a Go ``time.Duration``);
- ``nats.issuer_account`` is set IFF an account SIGNING key (not the identity key) signs.

Whether a LIVE NATS server accepts a minted JWT additionally depends on deployment-time facts the
auth-callout responder supplies (``sub`` == the server-provided user nkey, the response wrapper
``aud`` == the server id) and the account placement (``aud`` == account NAME in config mode) -- those
are verified by an integration test against a NATS server configured with ``auth_callout``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from typing import Any

import nkeys

from threetears.nats.subject_permissions import PrincipalPermissions

__all__ = ["account_public_key", "generate_account_seed", "mint_user_jwt"]

_ALG = "ed25519-nkey"
_HEADER: dict[str, str] = {"typ": "JWT", "alg": _ALG}


def _b64url(raw: bytes) -> str:
    """base64url WITHOUT padding -- the NATS jwt/v2 segment encoding."""
    return str(base64.urlsafe_b64encode(raw).rstrip(b"="), "ascii")


def _nkey_text(value: bytes | bytearray | str) -> str:
    """nkeys returns public keys as bytes; render the ``A...``/``U...`` text form."""
    if isinstance(value, str):
        return value
    return str(bytes(value), "ascii")


def generate_account_seed() -> bytes:
    """generate a fresh ACCOUNT nkey seed (``S...A...``) for one-time provisioning.

    Key creation is a provisioning step: the account signing key is generated once, stored via
    ``secret_refs``, and thereafter only loaded to sign. Uses the OS CSPRNG for the Ed25519 seed.

    :return: the encoded account nkey seed
    :rtype: bytes
    """
    return bytes(nkeys.encode_seed(os.urandom(32), nkeys.PREFIX_BYTE_ACCOUNT))


def account_public_key(account_seed: bytes) -> str:
    """the public account key (``A...``) for a signing seed -- e.g. for the server ``issuer`` config.

    :param account_seed: the account nkey seed
    :ptype account_seed: bytes
    :return: the public account key text
    :rtype: str
    """
    return _nkey_text(nkeys.from_seed(account_seed).public_key)


def mint_user_jwt(
    *,
    account_seed: bytes,
    user_public_key: str,
    permissions: PrincipalPermissions,
    name: str,
    expires_in_seconds: int,
    audience: str | None = None,
    issuer_account: str | None = None,
    now: int | None = None,
) -> str:
    """mint + sign a NATS v2 user JWT scoping a connection to ``permissions``.

    :param account_seed: the signing ACCOUNT nkey seed (the signer; loaded from ``secret_refs``)
    :ptype account_seed: bytes
    :param user_public_key: the user nkey (``U...``) the JWT is issued for. under auth-callout this
        is the user nkey the NATS server pre-generated and supplied in the AuthorizationRequest;
        the server rejects a JWT whose ``sub`` is anything else.
    :ptype user_public_key: str
    :param permissions: the principal's resolved pub/sub allow-list + ``allow_responses``
    :ptype permissions: PrincipalPermissions
    :param name: a human-readable name for the user claim
    :ptype name: str
    :param expires_in_seconds: TTL; ``exp`` is ``iat + expires_in_seconds``
    :ptype expires_in_seconds: int
    :param audience: the JWT ``aud`` -- the account NAME the user is placed in (config-mode
        auth-callout). omit in operator mode (placement follows the issuer).
    :ptype audience: str | None
    :param issuer_account: the account's public IDENTITY key (``A...``) when ``account_seed`` is a
        SIGNING key distinct from the identity key; omit when the identity key itself signs.
    :ptype issuer_account: str | None
    :param now: unix-seconds issue time; defaults to the current time
    :ptype now: int | None
    :return: the compact NATS v2 user JWT
    :rtype: str
    """
    signer = nkeys.from_seed(account_seed)
    issued_at = now if now is not None else int(time.time())

    nats_claim: dict[str, Any] = {
        "pub": {"allow": list(permissions.publish)},
        "sub": {"allow": list(permissions.subscribe)},
        "subs": -1,
        "data": -1,
        "payload": -1,
        "type": "user",
        "version": 2,
    }
    if permissions.allow_responses:
        # resp is a Go struct with no omitempty: max + ttl always present. ttl is a time.Duration
        # serialized as integer NANOSECONDS; 0 = the response permission never expires.
        nats_claim["resp"] = {"max": 1, "ttl": 0}
    if issuer_account is not None:
        nats_claim["issuer_account"] = issuer_account

    payload: dict[str, Any] = {
        "iss": _nkey_text(signer.public_key),
        "sub": user_public_key,
        "iat": issued_at,
        "exp": issued_at + expires_in_seconds,
        "name": name,
        "nats": nats_claim,
    }
    if audience is not None:
        payload["aud"] = audience
    payload["jti"] = _claims_id(payload)

    header_seg = _b64url(json.dumps(_HEADER, separators=(",", ":")).encode("ascii"))
    payload_seg = _b64url(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("ascii")
    )
    signing_input = f"{header_seg}.{payload_seg}".encode("ascii")
    signature = bytes(signer.sign(signing_input))
    return f"{header_seg}.{payload_seg}.{_b64url(signature)}"


def _claims_id(payload: dict[str, Any]) -> str:
    """canonical NATS claims id: base32(sha256(serialized claims)) with padding stripped.

    informational for user claims (the server does not re-derive it), but set canonically so the
    minted JWT matches the shape ``nsc``/Go produce.
    """
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("ascii")
    digest = hashlib.sha256(serialized).digest()
    return str(base64.b32encode(digest).rstrip(b"="), "ascii")
