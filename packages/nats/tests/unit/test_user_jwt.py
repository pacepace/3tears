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

from threetears.nats.subject_permissions import JsCapability, JsResource, PrincipalPermissions
from threetears.nats.user_jwt import (
    account_public_key,
    generate_account_seed,
    js_api_grants_for_stream,
    mint_user_jwt,
)


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
        publish=("3tears.tools.call", "3tears.hub.handshake"),
        subscribe=("_INBOX_agent_pod_p1.>", "3tears.agents.internal.a1.p1"),
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


def _nats_subject_match(pattern: str, subject: str) -> bool:
    """does a NATS publish-permission ``pattern`` admit ``subject``? (``*`` = one token, ``>`` = tail).

    a faithful tiny re-implementation of nats-server subject matching so the tests can prove, by the
    server's OWN rule, that a granted pattern admits a real op AND denies a cross-stream op. ``*``
    matches exactly one token; ``>`` is terminal and matches one-or-more remaining tokens; every
    other token matches literally.
    """
    p = pattern.split(".")
    s = subject.split(".")
    for i, tok in enumerate(p):
        if tok == ">":
            return len(s) >= i + 1  # one-or-more remaining tokens
        if i >= len(s):
            return False
        if tok != "*" and tok != s[i]:
            return False
    return len(p) == len(s)


class TestJetStreamGrants:
    """a callout-minted JWT must carry the KV/JS grants for its declared buckets+streams, scoped.

    config-mode callout JWTs carry their OWN allow-list (no account-wide JS grant behind them), so
    omitting these bricks every JetStream op (agent KV config/collections/checkpoints + streams).
    The control-plane grant is PINNED per declared stream: it must admit every real JS op against
    the principal's OWN streams yet DENY the cross-tenant direct-read / destroy a bare ``$JS.API.>``
    once exposed on a shared account.
    """

    def _kv_perms(self) -> PrincipalPermissions:
        return PrincipalPermissions(
            publish=("3tears.tools.call",),
            subscribe=("_INBOX_agent_pod_p1.>",),
            allow_responses=True,
            inbox_prefix="_INBOX_agent_pod_p1",
            js_resources=(
                JsResource.kv("3tears_agent_config", scope=None, writable=True),
                JsResource.kv("checkpoints", scope=None, writable=True),
                JsResource.stream("3tears_channels_deliver"),
            ),
        )

    def _js_pub(self) -> list[str]:
        nats = _payload(_mint(permissions=self._kv_perms()))["nats"]
        return [s for s in nats["pub"]["allow"] if s.startswith("$JS")]

    def test_kv_data_subtree_is_publish_only(self) -> None:
        nats = _payload(_mint(permissions=self._kv_perms()))["nats"]
        pub, sub = nats["pub"]["allow"], nats["sub"]["allow"]
        # per-bucket KV DATA subtree on PUBLISH for an unscoped bucket...
        for bucket in ("3tears_agent_config", "checkpoints"):
            assert f"$KV.{bucket}.>" in pub
        # ...and on SUBSCRIBE for none of them. nothing in nats-py ever subscribes a $KV subject:
        # put is a publish, get is a $JS.API request, and watch delivers to nc.new_inbox(). the
        # grant conferred no read and handed the holder every write's full value on every bucket.
        assert not [s for s in sub if s.startswith("$KV")], sub
        # the app allow-list is preserved, not replaced
        assert "3tears.tools.call" in pub

    def test_no_bare_js_api_wildcard(self) -> None:
        # the fail-closed-isolation fix: never the whole JetStream control plane.
        pub = _payload(_mint(permissions=self._kv_perms()))["nats"]["pub"]["allow"]
        assert "$JS.API.>" not in pub
        assert not any(g == "$JS.API.>" or g.startswith("$JS.API.*") for g in pub)

    def test_every_js_grant_pins_a_declared_stream_or_is_account_info(self) -> None:
        # every $JS grant must carry a declared stream name as a literal token -- or be one of the two
        # documented stream-LESS account subjects (the INFO connect probe + STREAM.NAMES, which
        # nats-py needs to resolve a KV bucket's stream for a kv.watch()/hot-reload). nothing else is
        # account-wide.
        declared = {"KV_3tears_agent_config", "KV_checkpoints", "3tears_channels_deliver"}
        account_level = {"$JS.API.INFO", "$JS.API.STREAM.NAMES"}
        for grant in self._js_pub():
            if grant in account_level:
                continue
            tokens = grant.split(".")
            assert any(tok in declared for tok in tokens), grant

    def test_account_level_subjects_granted_once_for_js_principal(self) -> None:
        # both stream-less account subjects granted exactly once: INFO (connect probe) +
        # STREAM.NAMES (KV-watch stream resolution -- without it agent.yaml hot-reload silently dies).
        js = self._js_pub()
        assert js.count("$JS.API.INFO") == 1
        assert js.count("$JS.API.STREAM.NAMES") == 1

    def test_real_js_ops_on_declared_streams_are_admitted(self) -> None:
        # the EXACT $JS.API subjects nats-py constructs for the KV + stream paths must each be
        # admitted by some granted pattern -- otherwise the op silently times out under enforce.
        js = self._js_pub()
        for stream in ("KV_3tears_agent_config", "KV_checkpoints", "3tears_channels_deliver"):
            real_ops = [
                f"$JS.API.STREAM.INFO.{stream}",
                f"$JS.API.STREAM.CREATE.{stream}",
                f"$JS.API.STREAM.UPDATE.{stream}",
                f"$JS.API.STREAM.DELETE.{stream}",
                f"$JS.API.STREAM.PURGE.{stream}",
                f"$JS.API.STREAM.MSG.GET.{stream}",
                f"$JS.API.STREAM.MSG.DELETE.{stream}",
                f"$JS.API.DIRECT.GET.{stream}",
                f"$JS.API.DIRECT.GET.{stream}.$KV.3tears_agent_config.somekey",
                f"$JS.API.CONSUMER.CREATE.{stream}",
                f"$JS.API.CONSUMER.LIST.{stream}",
                f"$JS.API.CONSUMER.CREATE.{stream}.eph-consumer",
                f"$JS.API.CONSUMER.CREATE.{stream}.eph-consumer.3tears.channels.deliver.x",
                f"$JS.API.CONSUMER.INFO.{stream}.dur1",
                f"$JS.API.CONSUMER.DELETE.{stream}.dur1",
                f"$JS.API.CONSUMER.DURABLE.CREATE.{stream}.dur1",
                f"$JS.API.CONSUMER.MSG.NEXT.{stream}.dur1",
            ]
            for op in real_ops:
                assert any(_nats_subject_match(g, op) for g in js), op

    def test_cross_stream_js_ops_are_denied(self) -> None:
        # a NON-declared bucket/stream's control subjects must match NO granted pattern: this is the
        # exact direct-read / destroy / info-leak a bare $JS.API.> exposed on a shared account.
        js = self._js_pub()
        forbidden = [
            "$JS.API.STREAM.MSG.GET.KV_other",  # direct-read another bucket's backing stream
            "$JS.API.STREAM.INFO.KV_other",
            "$JS.API.STREAM.DELETE.KV_other",  # destroy another bucket
            "$JS.API.STREAM.PURGE.KV_other",
            "$JS.API.DIRECT.GET.KV_other",
            "$JS.API.DIRECT.GET.KV_other.$KV.other.secret",
            "$JS.API.CONSUMER.CREATE.KV_other",
            "$JS.API.CONSUMER.DURABLE.CREATE.KV_other.spy",
            "$JS.API.CONSUMER.MSG.NEXT.KV_other.spy",
            "$JS.API.STREAM.MSG.GET.3tears_other_stream",
            # NB: $JS.API.STREAM.NAMES IS granted (account-level) -- nats-py needs it to resolve a KV
            # bucket's stream for kv.watch(); it enumerates only platform-constant stream names, no
            # stream DATA. STREAM.LIST (full per-stream config) is NOT needed by any client path.
            "$JS.API.STREAM.LIST",
        ]
        for op in forbidden:
            assert not any(_nats_subject_match(g, op) for g in js), op

    def test_no_js_grants_when_no_buckets_or_streams(self) -> None:
        nats = _payload(_mint(permissions=_perms()))["nats"]  # _perms declares neither
        assert not any(s.startswith(("$JS", "$KV")) for s in nats["pub"]["allow"])
        assert not any(s.startswith("$KV") for s in nats["sub"]["allow"])


_SCOPE = "agent_pod-019470a8b5c37def8123456789abcdef"
_OTHER_SCOPE = "agent_pod-ffffffffffffffffffffffffffffffff"
_COLL = "3tears-collections"
_COLL_STREAM = f"KV_{_COLL}"


class TestScopedKvGrants:
    """One bucket is SHARED by every principal, and pinning its stream name pins nothing.

    ``{ns}-collections`` is held by the agent pod, the hub, the gateway, the registry, the router
    and the adapter at once, so the per-stream grant that isolates buckets from each other does
    nothing between the principals inside this one. ``coll-task-03`` put a principal scope in the
    key and ``coll-task-04a`` made reads subject-addressable (``allow_direct: true``); these pin
    that the mint actually narrows to it, and -- the half that is easy to forget -- that it closes
    the four JetStream routes that carry a key in a request BODY or export the whole stream, none
    of which a ``$KV.`` grant can see.
    """

    def _scoped_perms(self, *, declare: bool = False) -> PrincipalPermissions:
        return PrincipalPermissions(
            publish=(),
            subscribe=("_INBOX_agent_pod_p1.>",),
            allow_responses=True,
            inbox_prefix="_INBOX_agent_pod_p1",
            js_resources=(
                JsResource.kv(_COLL, scope=_SCOPE, writable=True, declare=declare),
                # an UNSCOPED neighbour in the same principal, so every assertion below also
                # proves the narrowing did not leak sideways onto a bucket that writes no prefix.
                JsResource.kv("checkpoints", scope=None, writable=True),
            ),
        )

    def _pub(self, *, declare: bool = False) -> list[str]:
        return list(_payload(_mint(permissions=self._scoped_perms(declare=declare)))["nats"]["pub"]["allow"])

    def test_publish_grant_is_the_principals_own_scope_not_the_bucket(self) -> None:
        pub = self._pub()
        assert f"$KV.{_COLL}.{_SCOPE}.>" in pub
        assert f"$KV.{_COLL}.>" not in pub
        # and a peer's scope is admitted by no pattern at all
        assert not any(_nats_subject_match(g, f"$KV.{_COLL}.{_OTHER_SCOPE}.widgets.e1") for g in pub)
        assert any(_nats_subject_match(g, f"$KV.{_COLL}.{_SCOPE}.widgets.e1") for g in pub)

    def test_the_unscoped_neighbour_keeps_the_whole_subtree(self) -> None:
        # per-resource opt-in. ``checkpoints`` has its OWN l2_key keyed by thread id and writes no
        # scope prefix, so a uniform narrowing would deny every read on it -- as a ten-second
        # deadline, not as an error.
        assert "$KV.checkpoints.>" in self._pub()

    def test_the_read_subject_puts_kv_and_the_bucket_in_separate_tokens(self) -> None:
        # THE most expensive thing to get wrong in this shard. nats-py's direct read is
        # ``$JS.API.DIRECT.GET.{stream}.{subject}`` where ``{subject}`` is the whole
        # ``$KV.{bucket}.{key}`` -- so ``$JS.API.DIRECT.GET.{stream}.{scope}.>`` matches NOTHING,
        # and a grant that matches nothing is never refused out loud: the request is dropped and
        # the caller blames the broker ten seconds later.
        pub = self._pub()
        assert f"$JS.API.DIRECT.GET.{_COLL_STREAM}.$KV.{_COLL}.{_SCOPE}.>" in pub
        assert f"$JS.API.DIRECT.GET.{_COLL_STREAM}.{_SCOPE}.>" not in pub
        real_read = f"$JS.API.DIRECT.GET.{_COLL_STREAM}.$KV.{_COLL}.{_SCOPE}.widgets.e1"
        assert any(_nats_subject_match(g, real_read) for g in pub), pub

    def test_a_peers_scope_is_denied_on_every_route(self) -> None:
        pub = self._pub()
        peer_key = f"$KV.{_COLL}.{_OTHER_SCOPE}.widgets.e1"
        forbidden = [
            # BYPASS: the direct read of a peer's key
            f"$JS.API.DIRECT.GET.{_COLL_STREAM}.{peer_key}",
            # BYPASS 2: get-by-SEQUENCE. no tail, sequence in the body, so no subject names a key
            f"$JS.API.DIRECT.GET.{_COLL_STREAM}",
            # BYPASS 1: the body-carried read. the key rides in {"last_by_subj": ...}
            f"$JS.API.STREAM.MSG.GET.{_COLL_STREAM}",
            f"$JS.API.STREAM.MSG.DELETE.{_COLL_STREAM}",
            # BYPASS 3: a consumer whose filter_subject (in the body) is the whole bucket,
            # delivering to an inbox the holder names. both subject branches.
            f"$JS.API.CONSUMER.CREATE.{_COLL_STREAM}",
            f"$JS.API.CONSUMER.CREATE.{_COLL_STREAM}.spy",
            f"$JS.API.CONSUMER.CREATE.{_COLL_STREAM}.spy.$KV.{_COLL}.>",
            f"$JS.API.CONSUMER.DURABLE.CREATE.{_COLL_STREAM}.spy",
            f"$JS.API.CONSUMER.MSG.NEXT.{_COLL_STREAM}.spy",
            f"$JS.API.CONSUMER.LIST.{_COLL_STREAM}",
            # BYPASS 4: the verb sits at token 4, so ``STREAM.*`` covered all of these
            f"$JS.API.STREAM.SNAPSHOT.{_COLL_STREAM}",
            f"$JS.API.STREAM.RESTORE.{_COLL_STREAM}",
            f"$JS.API.STREAM.PURGE.{_COLL_STREAM}",
            f"$JS.API.STREAM.DELETE.{_COLL_STREAM}",
            # UPDATE is a READ primitive on a shared stream: republish/sources mirror every key
            # to a subject the caller controls.
            f"$JS.API.STREAM.UPDATE.{_COLL_STREAM}",
            f"$JS.API.STREAM.CREATE.{_COLL_STREAM}",
        ]
        for op in forbidden:
            assert not any(_nats_subject_match(g, op) for g in pub), op

    def test_the_bucket_can_still_be_bound(self) -> None:
        # ``js.key_value()`` binds through ``stream_info``. without INFO the bucket cannot be
        # opened at all, so the narrowing would brick every principal rather than isolate them.
        pub = self._pub()
        assert any(_nats_subject_match(g, f"$JS.API.STREAM.INFO.{_COLL_STREAM}") for g in pub)

    def test_declare_adds_create_and_update_and_nothing_else(self) -> None:
        plain = set(self._pub())
        declaring = set(self._pub(declare=True))
        assert declaring - plain == {
            f"$JS.API.STREAM.CREATE.{_COLL_STREAM}",
            f"$JS.API.STREAM.UPDATE.{_COLL_STREAM}",
        }
        assert not plain - declaring

    def test_no_verb_wildcard_survives_on_a_scoped_stream(self) -> None:
        # allow-list literal verbs; never enumerate destructive ones to deny. an enumeration is
        # wrong the moment nats-server adds a verb, and ``STREAM.*`` already covered four.
        scoped = [g for g in self._pub(declare=True) if _COLL_STREAM in g]
        for grant in scoped:
            head = grant.split(f".{_COLL_STREAM}")[0]
            assert "*" not in head, grant

    def test_a_scoped_bucket_without_a_scope_is_refused_at_construction(self) -> None:
        # GRANT-10, and it fails where a resolver builds the record rather than at the wire, so a
        # principal that cannot produce a scope can never be minted a grant that matches nothing.
        with pytest.raises(ValueError, match="no key scope"):
            JsResource(
                name=_COLL,
                kind=JsResource.kv(_COLL, scope=_SCOPE, writable=True).kind,
                capability=JsCapability.KV_SCOPED,
                scope=None,
                writable=True,
            )

    def test_a_scope_with_a_dot_is_refused(self) -> None:
        # a scope is ONE subject token. a dot silently produces two, and the grant then stops
        # matching the keys the principal writes -- a check that accepts a dot validates nothing.
        with pytest.raises(ValueError, match="does not match"):
            JsResource.kv(_COLL, scope="agent.pod", writable=True)

    def test_a_read_only_resource_gets_no_kv_publish(self) -> None:
        perms = PrincipalPermissions(
            publish=(),
            subscribe=("_INBOX_x.>",),
            allow_responses=False,
            inbox_prefix="_INBOX_x",
            js_resources=(JsResource.kv(_COLL, scope=_SCOPE, writable=False),),
        )
        pub = _payload(_mint(permissions=perms))["nats"]["pub"]["allow"]
        assert not [s for s in pub if s.startswith("$KV")], pub
        # ...and it can still READ, because a KV read is a $JS.API request and never a $KV publish
        assert f"$JS.API.DIRECT.GET.{_COLL_STREAM}.$KV.{_COLL}.{_SCOPE}.>" in pub


class TestJsApiGrantsForStreamIsPublic:
    """GRANT-12: ``coll-task-05b`` generates each static user's deny set FROM this function.

    Hand-deriving those subjects has failed twice -- once on whole-token wildcards, once by missing
    the six-token terminal ``$JS.API.STREAM.MSG.*.{stream}``, which is the body-carried read the
    whole sequence exists to close. Both drafts read green. So the function is public, reachable
    across the repo boundary without a Shape-A underscore violation, and defaults to the full set.
    """

    def test_reachable_from_the_package_root(self) -> None:
        import threetears.nats as nats_pkg  # noqa: PLC0415 - the lazy re-export is the thing under test

        assert nats_pkg.js_api_grants_for_stream is js_api_grants_for_stream
        assert "js_api_grants_for_stream" in nats_pkg.__all__

    def test_defaults_to_the_full_set_a_deny_list_needs(self) -> None:
        assert js_api_grants_for_stream(_COLL_STREAM) == [
            f"$JS.API.STREAM.*.{_COLL_STREAM}",
            f"$JS.API.STREAM.MSG.*.{_COLL_STREAM}",
            f"$JS.API.DIRECT.GET.{_COLL_STREAM}",
            f"$JS.API.DIRECT.GET.{_COLL_STREAM}.>",
            f"$JS.API.CONSUMER.*.{_COLL_STREAM}",
            f"$JS.API.CONSUMER.*.{_COLL_STREAM}.>",
            f"$JS.API.CONSUMER.*.*.{_COLL_STREAM}.>",
        ]

    def test_a_scoped_capability_without_its_inputs_raises_rather_than_emitting_a_dead_grant(self) -> None:
        with pytest.raises(ValueError, match="separate tokens"):
            js_api_grants_for_stream(_COLL_STREAM, capability=JsCapability.KV_SCOPED, bucket=_COLL)


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
        payload = _payload(_mint(user_public_key=upub, name="agent-7", expires_in_seconds=300, now=1000))
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
