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

__all__ = ["account_public_key", "encode_and_sign", "generate_account_seed", "mint_user_jwt"]

_ALG = "ed25519-nkey"
_HEADER: dict[str, str] = {"typ": "JWT", "alg": _ALG}

#: the ONE account-level (stream-less) JetStream API subject a JS-using principal is granted.
#: ``$JS.API.INFO`` returns only the connection's OWN account JetStream limits/usage -- it cannot
#: read another principal's stream/bucket data. it is unavoidable here: ``NatsClient.connect`` runs
#: ``account_info()`` as its post-connect JetStream reachability probe (``_verify_jetstream``, fatal
#: on failure) and the core KV cache's ``ping()`` health check calls it too, so omitting it would
#: brick every JS principal at connect under enforce. it carries no stream token, so it cannot be
#: pinned per-stream; it is granted (pub-only) only to principals that declare a bucket/stream.
_JS_API_ACCOUNT_INFO = "$JS.API.INFO"


def _js_api_grants_for_stream(stream: str) -> list[str]:
    """the JetStream control-plane subjects scoped to ONE stream ``stream``, pinned by literal name.

    Every entry carries ``stream`` as a LITERAL subject token, so the grant permits exactly the JS
    API operations nats-py issues against THIS principal's own stream and matches no other stream's
    control subjects (the cross-tenant ``$JS.API.STREAM.MSG.GET.KV_<other>`` direct-read and
    ``STREAM.DELETE``/``PURGE`` destroy that a bare ``$JS.API.>`` would have allowed are denied). a
    KV bucket ``<b>`` is backed by the stream ``KV_<b>``; a declared stream is its own name.

    The stream-name token position differs per op family (verified against the installed nats-py
    2.x: ``nats/js/manager.py`` STREAM/CONSUMER/DIRECT builders + ``nats/js/client.py`` pull-consumer
    ``CONSUMER.MSG.NEXT``), so the set pins the name at each position it can occupy:

    - ``$JS.API.STREAM.*.{stream}`` -- STREAM INFO/CREATE/UPDATE/DELETE/PURGE (name at token 5);
      ``manager.stream_info``/``add_stream``/``update_stream``/``delete_stream``/``purge_stream``.
    - ``$JS.API.STREAM.MSG.*.{stream}`` -- STREAM.MSG.GET / STREAM.MSG.DELETE (name at token 6);
      ``manager.get_msg`` (non-direct) / ``manager.delete_msg``.
    - ``$JS.API.DIRECT.GET.{stream}`` -- direct get by sequence (``manager.get_msg`` direct path).
    - ``$JS.API.DIRECT.GET.{stream}.>`` -- direct get by subject; the ``$KV.<b>.<key>`` suffix the
      KV ``get`` appends has its own dots, so it rides the ``>`` tail.
    - ``$JS.API.CONSUMER.*.{stream}`` -- CONSUMER CREATE (ephemeral, no name) / LIST (name at token 5).
    - ``$JS.API.CONSUMER.*.{stream}.>`` -- CONSUMER CREATE.<name>[.<filter>] / INFO / DELETE / PAUSE
      (name at token 5, with a trailing consumer/name/filter tail).
    - ``$JS.API.CONSUMER.*.*.{stream}.>`` -- CONSUMER DURABLE.CREATE.<durable> and MSG.NEXT.<consumer>
      (name at token 6; both are the only 7-token consumer ops and both put the stream at token 6,
      so a literal ``{stream}`` there can only ever target this stream).

    JetStream consumer ACK/NAK is NOT listed: it publishes to the delivered message's ``$JS.ACK.*``
    reply subject and rides the principal's ``allow_responses`` grant (the same way it did under the
    old ``$JS.API.>``, which never covered ``$JS.ACK``), so it needs no standing control grant here.

    :param stream: the JetStream stream name to pin every entry to
    :ptype stream: str
    :return: the per-stream JS-API control-plane allow-list (publish subjects)
    :rtype: list[str]
    """
    return [
        f"$JS.API.STREAM.*.{stream}",
        f"$JS.API.STREAM.MSG.*.{stream}",
        f"$JS.API.DIRECT.GET.{stream}",
        f"$JS.API.DIRECT.GET.{stream}.>",
        f"$JS.API.CONSUMER.*.{stream}",
        f"$JS.API.CONSUMER.*.{stream}.>",
        f"$JS.API.CONSUMER.*.*.{stream}.>",
    ]


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


def encode_and_sign(*, account_seed: bytes, payload: dict[str, Any]) -> str:
    """encode + sign a NATS jwt/v2 claims payload with an account nkey -- the shared signing core.

    header ``{"typ":"JWT","alg":"ed25519-nkey"}``; signature over the ASCII ``header.payload``; all
    three segments base64url WITHOUT padding. The single implementation used for both user JWTs and
    auth-callout response JWTs, so the security-critical encoding lives in exactly one place.

    :param account_seed: the signing account nkey seed
    :ptype account_seed: bytes
    :param payload: the claims payload (caller sets ``iss``/``sub``/``nats``/etc.)
    :ptype payload: dict[str, Any]
    :return: the compact signed NATS v2 JWT
    :rtype: str
    """
    signer = nkeys.from_seed(account_seed)
    header_seg = _b64url(json.dumps(_HEADER, separators=(",", ":")).encode("ascii"))
    payload_seg = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("ascii"))
    signing_input = f"{header_seg}.{payload_seg}".encode("ascii")
    return f"{header_seg}.{payload_seg}.{_b64url(bytes(signer.sign(signing_input)))}"


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
    issued_at = now if now is not None else int(time.time())

    # JetStream grants for a callout-minted principal: a scoped user JWT carries its OWN pub/sub
    # allow-list (there is no account-wide JS grant behind it in config mode), so a principal that
    # touches KV/streams must be granted the JetStream subjects HERE or its JS operations time out.
    #   - per declared KV bucket: the bucket's data subtree ``$KV.{bucket}.>`` (pub + sub).
    #   - the JetStream control plane scoped to ONLY the streams this principal declares (pub; the
    #     API is request/reply and the reply rides the principal's already-scoped inbox). A KV bucket
    #     ``<b>`` is backed by the stream ``KV_<b>``; a declared stream is its own name. Each control
    #     subject is PINNED to its stream's literal name (see ``_js_api_grants_for_stream``), so the
    #     principal can drive every JS op against its OWN streams but is DENIED the cross-tenant
    #     direct-read (``$JS.API.STREAM.MSG.GET.KV_<other>``) / destroy (``STREAM.DELETE``/``PURGE``)
    #     that a bare ``$JS.API.>`` exposed on a shared account. Plus one account-level (stream-less)
    #     subject, ``$JS.API.INFO`` -- see ``_JS_API_ACCOUNT_INFO`` for why it cannot be scoped away.
    #
    # RESIDUAL (deliberate, not a regression): the ``$KV.{bucket}.>`` data grant is bucket-scoped but
    # NOT key-scoped. Buckets shared across principals (``{ns}-collections`` keyed by entity, the
    # ``checkpoints`` bucket keyed by thread/conversation, ``{ns}_agent_config``) intentionally hold
    # no per-agent key prefix -- the collections L2 key is ``{table}.{pk}`` and the checkpoint key is
    # ``{thread_id}[.{ns}]`` (see ``collections/base.py:l2_key`` / ``langgraph/checkpoint.py:l2_key``),
    # so a peer that legitimately holds the same bucket can read peers' keys within it. Tightening to
    # ``$KV.{bucket}.{prefix}.>`` is impossible without a key-prefix the data layer does not write; it
    # would break every read. Per-bucket isolation (the control-plane fix above) is what closes the
    # cross-BUCKET hole; intra-bucket key isolation is tracked separately.
    kv_data = [f"$KV.{bucket}.>" for bucket in permissions.kv_buckets]
    js_streams = [f"KV_{bucket}" for bucket in permissions.kv_buckets] + list(permissions.streams)
    js_control: list[str] = []
    if js_streams:
        js_control.append(_JS_API_ACCOUNT_INFO)
        for stream in js_streams:
            js_control.extend(_js_api_grants_for_stream(stream))

    nats_claim: dict[str, Any] = {
        "pub": {"allow": [*permissions.publish, *kv_data, *js_control]},
        "sub": {"allow": [*permissions.subscribe, *kv_data]},
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
        "iss": account_public_key(account_seed),
        "sub": user_public_key,
        "iat": issued_at,
        "exp": issued_at + expires_in_seconds,
        "name": name,
        "nats": nats_claim,
    }
    if audience is not None:
        payload["aud"] = audience
    payload["jti"] = _claims_id(payload)
    return encode_and_sign(account_seed=account_seed, payload=payload)


def _claims_id(payload: dict[str, Any]) -> str:
    """canonical NATS claims id: base32(sha256(serialized claims)) with padding stripped.

    informational for user claims (the server does not re-derive it), but set canonically so the
    minted JWT matches the shape ``nsc``/Go produce.
    """
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("ascii")
    digest = hashlib.sha256(serialized).digest()
    return str(base64.b32encode(digest).rstrip(b"="), "ascii")
