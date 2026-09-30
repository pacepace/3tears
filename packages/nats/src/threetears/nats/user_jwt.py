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

from threetears.nats.subject_permissions import (
    JsCapability,
    JsResourceKind,
    PrincipalPermissions,
    capability_declares,
    capability_is_scoped,
)

__all__ = [
    "account_public_key",
    "encode_and_sign",
    "generate_account_seed",
    "js_api_grants_for_stream",
    "mint_user_jwt",
]

_ALG = "ed25519-nkey"
_HEADER: dict[str, str] = {"typ": "JWT", "alg": _ALG}

#: account-level (stream-LESS) JetStream API subjects a JS-using principal is granted. Neither
#: carries a stream token, so neither can be pinned per-stream; both only inspect/enumerate the
#: connection's OWN account (per-customer-account prod) or the shared account's platform-constant
#: stream names (shared-account dev) -- never another principal's stream/bucket DATA. Granted
#: (pub-only) only to principals that declare a bucket/stream.
#:   - ``$JS.API.INFO``: ``NatsClient.connect`` runs ``account_info()`` as its post-connect JetStream
#:     reachability probe (``_verify_jetstream``, fatal on failure) and the core KV cache's ``ping()``
#:     calls it; omitting it bricks every JS principal at connect under enforce.
#:   - ``$JS.API.STREAM.NAMES``: nats-py resolves a KV bucket's backing stream by subject when setting
#:     up a ``kv.watch()`` (the agent-config hot-reload watcher, ``runtime/hot_reload.py``). Without
#:     it the WATCH fails (KV get/put/bind still work via the per-stream grants), silently disabling
#:     agent.yaml hot-reload under enforce.
_JS_API_ACCOUNT = ("$JS.API.INFO", "$JS.API.STREAM.NAMES")


def js_api_grants_for_stream(
    stream: str,
    *,
    capability: JsCapability = JsCapability.FULL,
    bucket: str | None = None,
    scope: str | None = None,
    filter_subject: str | None = None,
) -> list[str]:
    """the JetStream control-plane subjects scoped to ONE stream ``stream``, pinned by literal name.

    Every entry carries ``stream`` as a LITERAL subject token, so the grant permits only the JS API
    operations nats-py issues against THIS stream and matches no other stream's control subjects. A
    KV bucket ``<b>`` is backed by the stream ``KV_<b>``; a declared stream is its own name.

    **Pinning the stream name pins nothing on a SHARED stream.** ``{ns}-collections`` is one bucket
    held by many principals, so :attr:`JsCapability.FULL` on it admits every principal's keys
    through four separate routes. The scoped capabilities exist to close them:

    - ``$JS.API.STREAM.MSG.*.{stream}`` covers ``STREAM.MSG.GET``, whose key rides in the request
      BODY where no subject permission can see it;
    - ``$JS.API.DIRECT.GET.{stream}`` without a tail is get-by-SEQUENCE, also body-carried;
    - ``$JS.API.CONSUMER.*`` lets the holder create a consumer whose ``filter_subject`` (again, in
      the body) is the whole bucket, delivering to an inbox it names;
    - ``$JS.API.STREAM.*`` puts the VERB at token 4, so the wildcard silently covers ``SNAPSHOT``
      (the whole bucket, streamed to a caller-named ``deliver_subject``), ``RESTORE``, ``PURGE`` and
      ``UPDATE`` -- and ``UPDATE`` is itself a read-all primitive, since ``republish`` / ``sources``
      mirror every key onto a subject the caller controls.

    Public because ``coll-task-05b`` must generate each STATIC NATS user's deny set from the
    function that emits the grants rather than by hand: hand-deriving these shapes has failed twice,
    once on whole-token wildcards (``KV_x*`` is a literal token containing an asterisk and matches
    nothing) and once by missing ``$JS.API.STREAM.MSG.*.{stream}`` -- six tokens, terminal, which is
    the body-carried read this whole sequence exists to close.

    The stream-name token position differs per op family (verified against the installed nats-py
    2.x: ``nats/js/manager.py`` STREAM/CONSUMER/DIRECT builders + ``nats/js/client.py`` pull-consumer
    ``CONSUMER.MSG.NEXT``), so :attr:`JsCapability.FULL` -- the infra identities' management grant,
    never a pod's -- pins the name at each position it can occupy:

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

    The scoped capabilities emit LITERAL verb tokens instead, allow-listing rather than enumerating
    destructive verbs to deny -- an enumeration of what to deny is wrong the moment nats-server adds
    a verb:

    - ``$JS.API.STREAM.INFO.{stream}`` -- ``js.key_value()`` binds through ``stream_info``, so
      without this the bucket cannot even be opened.
    - ``$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{scope}.>`` -- **``$KV`` and ``{bucket}`` are
      SEPARATE TOKENS** (``manager.get_msg``'s direct branch appends the whole ``$KV.<b>.<key>``
      subject after the stream name). Writing ``$JS.API.DIRECT.GET.{stream}.{scope}.>`` matches
      nothing, and a grant that matches nothing does not raise -- the request is dropped unanswered
      and the caller sees a ten-second deadline, indistinguishable from an unreachable broker.
      This read subject is only reachable at all because the bucket runs ``allow_direct: true``
      (``coll-task-04a``); with it false nats-py falls back to the body-carried form.
    - plus ``$JS.API.STREAM.CREATE.{stream}`` and ``$JS.API.STREAM.UPDATE.{stream}`` for
      :attr:`JsCapability.KV_SCOPED_DECLARE` alone.

    :attr:`JsCapability.KV_KEY_READ` narrows further, to ONE whole key carried in ``scope``:

    - ``$JS.API.STREAM.INFO.{stream}`` -- the bind.
    - ``$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{key}`` -- the read, with the key as the LITERAL
      final token. A ``{key}.>`` tail needs at least one more token and matches nothing a
      single-token key produces. Reachable only on a bucket created with ``allow_direct: true``.
    - ``$JS.API.CONSUMER.CREATE.{stream}.*.$KV.{bucket}.{key}`` -- the watch, as a NAMED consumer
      whose filter rides in the SUBJECT, where nats-server checks it against the body's
      ``filter_subject``. The unnamed ``CONSUMER.CREATE.{stream}`` is NOT granted: its filter rides
      only in the body and could name the whole bucket. nats-py's ``KeyValue.watch`` creates an
      unnamed consumer, so a watcher on this grant subscribes with an explicit consumer name. The
      delivered messages land on the holder's own inbox, already covered by its ``{inbox}.>``
      subscribe grant; flow-control replies ride ``allow_responses``.

    :attr:`JsCapability.KV_TABLE_SCOPED` emits the :attr:`JsCapability.KV_SCOPED` pair with
    ``scope`` carrying the resource's whole ``{owner_scope}.{table}`` key prefix
    (:attr:`~threetears.nats.subject_permissions.JsResource.key_prefix`), so the direct read is
    ``$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{owner_scope}.{table}.>`` -- one table of another
    principal's keys, and no consumer or watch route at all.

    :attr:`JsCapability.KV_BUCKET_KEYS` is every POD's grant on an unscoped bucket -- its own
    coordination buckets, the platform's shared per-agent buckets, and another agent's coordination
    bucket an operator granted it. It covers the WHOLE of one bucket, through key-addressed calls
    and named key consumers, and nothing about the stream itself:

    - ``$JS.API.STREAM.INFO.{stream}`` -- the bind.
    - ``$JS.API.STREAM.MSG.GET.{stream}`` -- the read nats-py issues on a bucket bound WITHOUT
      ``allow_direct``, with the key in the body. Every key it can name is a key the grant already
      covers, since the bucket is the grant's boundary. ``STREAM.MSG.DELETE`` is NOT granted.
    - ``$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.>`` -- the read on a bucket WITH ``allow_direct``.
    - ``$JS.API.CONSUMER.CREATE.{stream}.*.$KV.{bucket}.>`` -- a key watch or a key listing, as a
      NAMED consumer whose filter rides in the SUBJECT (:meth:`threetears.nats.kv.NatsKvBucket.watch_key`,
      :meth:`threetears.nats.kv.NatsKvBucket.keys`). The unnamed ``CONSUMER.CREATE.{stream}``, the
      durable create and ``MSG.NEXT`` are NOT granted; nats-py's ``KeyValue.watch``/``keys`` use the
      unnamed form and are refused.

    :attr:`JsCapability.KV_OWNER_KEYS` is a pod's grant on a SHARED pod bucket, where the key
    prefix rather than the bucket is the boundary: the same three routes narrowed to the holder's
    own ``{scope}.>`` -- ``STREAM.INFO`` (the bind), ``DIRECT.GET.{stream}.$KV.{bucket}.{scope}.>``
    and ``CONSUMER.CREATE.{stream}.*.$KV.{bucket}.{scope}.>`` -- and NOT ``STREAM.MSG.GET``, whose
    key rides in the body and so would read every owner's keys. The bucket must run
    ``allow_direct``, which the hub declares, or nats-py falls back to that body-carried read.

    No stream-admin verb, ever: ``CREATE``/``UPDATE`` accept ``sources`` and ``republish``, which
    copy ANY stream's messages into one the holder can read; ``DELETE``/``PURGE`` destroy state;
    ``SNAPSHOT``/``RESTORE`` export or replace it. The hub declares every bucket a pod binds. The
    ``$KV.`` publish for a writable grant is minted by :func:`mint_user_jwt` from the resource, not
    here.

    :attr:`JsCapability.STREAM_CONSUMER` is a pod's grant on a plain stream it collects its OWN
    messages from, and it is exactly ONE subject:

    - ``$JS.API.CONSUMER.CREATE.{stream}.*.{filter}`` -- a NAMED consumer whose filter rides in the
      subject, where nats-server checks it against the body's ``filter_subject``; ``filter`` is the
      resource's ``filter_subject``, a pattern inside the pod's own subjects. The consumer PUSHES to
      a deliver subject the pod names, and the pod can only subscribe its own inbox; acknowledgements
      and flow control ride ``allow_responses``.

    Nothing else, and each omission is a cross-principal read: the unnamed
    ``CONSUMER.CREATE.{stream}`` and ``DURABLE.CREATE`` carry their filter only in the body;
    ``MSG.NEXT``, ``INFO`` and ``DELETE`` reach an existing consumer BY NAME whoever created it, so a
    pod holding them could pull, inspect or destroy the collector's consumer over every principal's
    messages; ``STREAM.INFO`` and every read verb describe or return the whole stream. Publishing
    into a stream is an ordinary publish grant and needs nothing here.

    JetStream consumer ACK/NAK is NOT listed: it publishes to the delivered message's ``$JS.ACK.*``
    reply subject and rides the principal's ``allow_responses`` grant (the same way it did under the
    old ``$JS.API.>``, which never covered ``$JS.ACK``), so it needs no standing control grant here.

    :param stream: the JetStream stream name to pin every entry to
    :ptype stream: str
    :param capability: what the holder may do against ``stream``; defaults to the historical full
        set, which is also what a deny-list generator wants
    :ptype capability: JsCapability
    :param bucket: the KV bucket ``stream`` backs; required for a scoped capability
    :ptype bucket: str | None
    :param scope: the key prefix the grant narrows to -- the holder's L2 key scope, the one key
        for :attr:`JsCapability.KV_KEY_READ`, or ``{owner_scope}.{table}`` for
        :attr:`JsCapability.KV_TABLE_SCOPED`; required for a scoped capability
    :ptype scope: str | None
    :param filter_subject: the pattern a :attr:`JsCapability.STREAM_CONSUMER` holder's consumers
        filter on; required for that capability and ignored by every other
    :ptype filter_subject: str | None
    :return: the per-stream JS-API control-plane allow-list (publish subjects)
    :rtype: list[str]
    :raises ValueError: if a scoped capability is requested without both ``bucket`` and ``scope``,
        or a consumer capability without ``filter_subject``
    """
    grants: list[str]
    if capability is JsCapability.KV_BUCKET_KEYS:
        if bucket is None:
            raise ValueError(
                f"{capability.value} grants for stream {stream!r} need the bucket name: the direct "
                f"read subject is $JS.API.DIRECT.GET.{stream}.$KV.<bucket>.>, in which $KV and the "
                f"bucket are separate tokens"
            )
        grants = [
            f"$JS.API.STREAM.INFO.{stream}",
            f"$JS.API.STREAM.MSG.GET.{stream}",
            f"$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.>",
            f"$JS.API.CONSUMER.CREATE.{stream}.*.$KV.{bucket}.>",
        ]
    elif capability is JsCapability.STREAM_CONSUMER:
        if not filter_subject:
            raise ValueError(
                f"{capability.value} grants for stream {stream!r} need the consumer filter: the grant is "
                f"$JS.API.CONSUMER.CREATE.{stream}.*.<filter>, and with no filter a consumer could read "
                f"every principal's messages"
            )
        grants = [f"$JS.API.CONSUMER.CREATE.{stream}.*.{filter_subject}"]
    elif not capability_is_scoped(capability):
        grants = [
            f"$JS.API.STREAM.*.{stream}",
            f"$JS.API.STREAM.MSG.*.{stream}",
            f"$JS.API.DIRECT.GET.{stream}",
            f"$JS.API.DIRECT.GET.{stream}.>",
            f"$JS.API.CONSUMER.*.{stream}",
            f"$JS.API.CONSUMER.*.{stream}.>",
            f"$JS.API.CONSUMER.*.*.{stream}.>",
        ]
    else:
        if bucket is None or scope is None:
            raise ValueError(
                f"{capability.value} grants for stream {stream!r} need both the bucket name and the "
                f"key scope: the read subject is $JS.API.DIRECT.GET.{stream}.$KV.<bucket>.<scope>.>, "
                f"in which $KV and the bucket are separate tokens. "
                f"got bucket={bucket!r} scope={scope!r}"
            )
        if capability is JsCapability.KV_KEY_READ:
            grants = [
                f"$JS.API.STREAM.INFO.{stream}",
                f"$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{scope}",
                f"$JS.API.CONSUMER.CREATE.{stream}.*.$KV.{bucket}.{scope}",
            ]
        elif capability is JsCapability.KV_OWNER_KEYS:
            grants = [
                f"$JS.API.STREAM.INFO.{stream}",
                f"$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{scope}.>",
                f"$JS.API.CONSUMER.CREATE.{stream}.*.$KV.{bucket}.{scope}.>",
            ]
        else:
            grants = [
                f"$JS.API.STREAM.INFO.{stream}",
                f"$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{scope}.>",
            ]
        if capability_declares(capability):
            grants.extend([f"$JS.API.STREAM.CREATE.{stream}", f"$JS.API.STREAM.UPDATE.{stream}"])
    return grants


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
    #   - per declared KV bucket the principal WRITES: a ``$KV.`` publish grant, narrowed to the
    #     principal's own key scope where the bucket's keys carry one and left at the whole
    #     ``$KV.{bucket}.>`` subtree where they do not.
    #   - the JetStream control plane scoped to ONLY the streams this principal declares (pub; the
    #     API is request/reply and the reply rides the principal's already-scoped inbox), at the
    #     capability that resource declares -- see ``js_api_grants_for_stream``. Plus the two
    #     account-level (stream-less) subjects (``$JS.API.INFO`` + ``$JS.API.STREAM.NAMES``) -- see
    #     ``_JS_API_ACCOUNT`` for why they cannot be scoped per-stream.
    #
    # ``$KV.`` GOES ON PUBLISH ONLY, FOR EVERY BUCKET. It used to go into both allow lists, and the
    # subscribe half conferred NO read capability at all: nothing in nats-py ever subscribes a
    # ``$KV.`` subject -- ``put`` is a publish, ``get`` is a request to ``$JS.API``, and ``watch``
    # creates a push consumer that delivers to ``nc.new_inbox()``, which the principal's own
    # ``{inbox}.>`` grant already covers. What it DID confer was a firehose of every write's full
    # value on every bucket the principal held, including the two cross-agent, cross-customer
    # buckets (``checkpoints`` and ``{ns}_agent_config``) that are out of scope for KEY isolation.
    # Dropping it costs nothing and closes that firehose everywhere, not only on the scoped bucket.
    #
    # PER-RESOURCE OPT-IN, and it is not optional. A resource carries a scope only where its keys
    # lead with one: ``{ns}-collections`` and the shared pod buckets (``KV_OWNER_KEYS``). Emitting
    # ``{scope}.>`` for ``{ns}_agent_config``, ``{ns}-epochs`` or a pod's own coordination bucket
    # would deny every read on it -- and a refused JetStream request is never answered, so the
    # failure arrives as a ten-second deadline that reads as an unreachable broker.
    #
    # ``key_prefix`` rather than ``scope`` for both tails: a table-scoped resource narrows one
    # token past its scope (``{scope}.{table}``), and reading the prefix from one property is what
    # keeps the publish tail and the read tail from ever naming different prefixes. Several
    # resources on one stream each emit its ``STREAM.INFO`` bind; the repeats are left in, because
    # the hub renders its static-user confs from this same composition.
    kv_data: list[str] = []
    js_control: list[str] = []
    for resource in permissions.js_resources:
        if resource.kind is JsResourceKind.KV_BUCKET and resource.writable:
            tail = ">" if resource.key_prefix is None else f"{resource.key_prefix}.>"
            kv_data.append(f"$KV.{resource.name}.{tail}")
        bucket = resource.name if resource.kind is JsResourceKind.KV_BUCKET else None
        js_control.extend(
            js_api_grants_for_stream(
                resource.stream_name,
                capability=resource.capability,
                bucket=bucket,
                scope=resource.key_prefix,
                filter_subject=resource.filter_subject,
            )
        )
    if permissions.js_resources:
        js_control = [*_JS_API_ACCOUNT, *js_control]

    nats_claim: dict[str, Any] = {
        "pub": {"allow": [*permissions.publish, *kv_data, *js_control]},
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
