"""offline contract tests for the generalized NATS auth-callout responder (platform-auth A).

The responder owns the generic loop: subscribe the callout subject, decode the request, delegate
"who is this?" to a :class:`PrincipalResolver` and "what may they do?" to a :class:`GrantPolicy`,
then mint the admit (a scoped user JWT) or the deny response. These drive it with fake resolver /
policy / NATS client and assert the fail-closed contract; the codecs + user-JWT minting are proven
in ``test_auth_callout.py`` / ``test_user_jwt.py``.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from threetears.nats.auth_callout_responder import (
    AUTH_CALLOUT_SUBJECT,
    AuthAccountKeyError,
    AuthCalloutResponder,
    ResolvedPrincipal,
)
from threetears.nats.subject_permissions import PrincipalPermissions
from threetears.nats.subjects import Subject
from threetears.nats.user_jwt import generate_account_seed


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def _request_jwt(*, server_id: str = "NSERVER1", user_nkey: str = "UUSER1", auth_token: str = "tok-1") -> str:
    """a decode-shaped (signature-unverified) AuthorizationRequest JWT."""
    header = _b64url(json.dumps({"typ": "JWT", "alg": "ed25519-nkey"}).encode())
    payload = _b64url(
        json.dumps(
            {
                "nats": {
                    "server_id": {"id": server_id},
                    "user_nkey": user_nkey,
                    "connect_opts": {"auth_token": auth_token, "name": "pod-7"},
                    "client_info": {},
                }
            }
        ).encode()
    )
    return f"{header}.{payload}.{_b64url(b'sig')}"


def _decode_payload(jwt: str) -> dict[str, Any]:
    """the JSON payload of a compact JWT (middle segment); signature not checked."""
    _, payload_seg, _ = jwt.split(".")
    result: dict[str, Any] = json.loads(_b64url_decode(payload_seg))
    return result


def _perms() -> PrincipalPermissions:
    return PrincipalPermissions(
        publish=("scriob.>",),
        subscribe=("scriob.>", "_INBOX_scriob-pod_c1.>"),
        allow_responses=True,
        inbox_prefix="_INBOX_scriob-pod_c1",
    )


# parity-with: threetears.nats.auth_callout_responder.PrincipalResolver
class _FakeResolver:
    def __init__(self, result: ResolvedPrincipal | None) -> None:
        self._result = result
        self.seen: list[Any] = []

    async def resolve(self, request: Any) -> ResolvedPrincipal | None:
        self.seen.append(request)
        return self._result


# parity-with: threetears.nats.auth_callout_responder.GrantPolicy
class _FakePolicy:
    def __init__(self, perms: PrincipalPermissions) -> None:
        self._perms = perms
        self.seen: list[ResolvedPrincipal] = []

    def permissions(self, principal: ResolvedPrincipal) -> PrincipalPermissions:
        self.seen.append(principal)
        return self._perms


# parity-exempt: narrow offline double for the NATS subscription handle; the responder holds it opaquely (unsubscribe target only)
class _FakeSub:
    pass


# parity-exempt: narrow offline double for the NATS wire client, which AuthCalloutResponder types as `nc: Any` (no production protocol to bind)
class _FakeNats:
    def __init__(self) -> None:
        self.subscribed: dict[str, Any] | None = None
        self.unsubscribed: list[Any] = []
        self.replies: list[tuple[str, bytes]] = []
        self._sub = _FakeSub()

    async def subscribe(self, *, subject: Any, queue: str, cb: Any) -> _FakeSub:
        self.subscribed = {"subject": subject, "queue": queue, "cb": cb}
        return self._sub

    async def unsubscribe(self, sub: Any) -> None:
        self.unsubscribed.append(sub)

    async def publish_raw_reply(self, *, reply_subject: str, payload: bytes) -> None:
        self.replies.append((reply_subject, payload))


# parity-exempt: narrow offline double for the raw NATS callout message (data + reply_subject only)
class _FakeMsg:
    def __init__(self, data: bytes, reply_subject: str) -> None:
        self.data = data
        self.reply_subject = reply_subject


def _responder(nc: _FakeNats, *, resolver: Any, policy: Any) -> AuthCalloutResponder:
    return AuthCalloutResponder(
        nc,
        account_seed=generate_account_seed(),
        resolver=resolver,
        policy=policy,
        account_name="SCRIOB",
    )


def _principal() -> ResolvedPrincipal:
    return ResolvedPrincipal(conn_id="c1", name="scriob-pod:c1", claims={"role": "scriob-server-pod"})


async def test_admit_mints_a_scoped_user_jwt_for_the_resolved_principal() -> None:
    """A resolved principal → the response ADMITS with a user JWT scoped by the grant policy."""
    resolver = _FakeResolver(_principal())
    policy = _FakePolicy(_perms())
    responder = _responder(_FakeNats(), resolver=resolver, policy=policy)

    from threetears.nats.auth_callout import decode_auth_request

    request = decode_auth_request(_request_jwt(server_id="NSRV", user_nkey="UME", auth_token="tok"))
    response = await responder.build_response(request)

    payload = _decode_payload(response)
    assert payload["aud"] == "NSRV" and payload["sub"] == "UME", "response is bound to the server + user nkey"
    nats_claim = payload["nats"]
    assert "error" not in nats_claim, "an admit carries no error"
    assert "jwt" in nats_claim, "an admit carries the minted user JWT"
    # the policy scoped the ACTUAL resolved principal; the minted JWT carries its name.
    assert policy.seen == [_principal()], "the grant policy scoped the resolved principal"
    user_payload = _decode_payload(nats_claim["jwt"])
    assert user_payload["sub"] == "UME", "the user JWT is issued for the server-supplied user nkey"
    assert user_payload["name"] == "scriob-pod:c1", "the user JWT carries the principal's name"
    assert user_payload["aud"] == "SCRIOB", "account_name routes to the user JWT aud (config-mode placement)"


async def test_resolver_that_raises_denies_fail_closed() -> None:
    """A resolver that RAISES (a third-party fault) denies with a signed error — never propagates."""

    class _RaisingResolver:
        async def resolve(self, request: Any) -> ResolvedPrincipal | None:
            raise RuntimeError("resolver blew up")

    responder = _responder(_FakeNats(), resolver=_RaisingResolver(), policy=_FakePolicy(_perms()))
    from threetears.nats.auth_callout import decode_auth_request

    request = decode_auth_request(_request_jwt(server_id="NSRV", user_nkey="UME"))
    nats_claim = _decode_payload(await responder.build_response(request))["nats"]
    assert nats_claim.get("error") == "authentication failed", "a resolver fault fails closed to a signed deny"
    assert "jwt" not in nats_claim


async def test_grant_policy_that_raises_denies_fail_closed() -> None:
    """A grant policy that RAISES after a successful resolve still denies (fail closed)."""

    class _RaisingPolicy:
        def permissions(self, principal: ResolvedPrincipal) -> PrincipalPermissions:
            raise RuntimeError("policy blew up")

    responder = _responder(_FakeNats(), resolver=_FakeResolver(_principal()), policy=_RaisingPolicy())
    from threetears.nats.auth_callout import decode_auth_request

    request = decode_auth_request(_request_jwt(server_id="NSRV", user_nkey="UME"))
    nats_claim = _decode_payload(await responder.build_response(request))["nats"]
    assert nats_claim.get("error") == "authentication failed", "a grant/mint fault fails closed to a signed deny"
    assert "jwt" not in nats_claim


async def test_a_request_without_a_server_id_cannot_be_signed_and_is_left_unanswered() -> None:
    """No ``server_id.id`` → no ``aud`` → a deny cannot be signed either, so build_response raises
    (the caller leaves it unanswered; the server times out and denies — the only fail-closed path)."""
    from threetears.nats.auth_callout import AuthCalloutRequest

    request = AuthCalloutRequest(server_id={}, user_nkey="UME", connect_opts={}, client_info={})
    responder = _responder(_FakeNats(), resolver=_FakeResolver(_principal()), policy=_FakePolicy(_perms()))
    with pytest.raises(ValueError):
        await responder.build_response(request)


def test_constructing_with_a_bad_seed_fails_closed_on_every_path() -> None:
    """A garbage seed passed to __init__ directly (not just from_secret) also fails closed."""
    with pytest.raises(AuthAccountKeyError):
        AuthCalloutResponder(
            _FakeNats(),
            account_seed=b"not-a-real-nkey-seed",
            resolver=_FakeResolver(_principal()),
            policy=_FakePolicy(_perms()),
        )


async def test_deny_when_no_principal_resolves_and_policy_is_not_consulted() -> None:
    """An unresolved principal → DENY (fail closed); the grant policy is never asked."""
    resolver = _FakeResolver(None)
    policy = _FakePolicy(_perms())
    responder = _responder(_FakeNats(), resolver=resolver, policy=policy)

    from threetears.nats.auth_callout import decode_auth_request

    request = decode_auth_request(_request_jwt(server_id="NSRV", user_nkey="UME"))
    response = await responder.build_response(request)

    nats_claim = _decode_payload(response)["nats"]
    assert nats_claim.get("error") == "authentication failed", "an unresolved principal is denied"
    assert "jwt" not in nats_claim, "a deny carries no user JWT"
    assert policy.seen == [], "the grant policy is not consulted for a denied connection"


async def test_handle_request_decodes_then_replies_on_the_inbox() -> None:
    """A well-formed request over a reply-subject → the signed response is published on that inbox."""
    nc = _FakeNats()
    responder = _responder(nc, resolver=_FakeResolver(_principal()), policy=_FakePolicy(_perms()))

    await responder.handle_request(_FakeMsg(_request_jwt().encode("utf-8"), reply_subject="_INBOX.reply.1"))

    assert len(nc.replies) == 1
    reply_subject, payload = nc.replies[0]
    assert reply_subject == "_INBOX.reply.1"
    assert "jwt" in _decode_payload(payload.decode("utf-8"))["nats"], "the reply admits the resolved principal"


async def test_handle_request_undecodable_request_is_left_unanswered() -> None:
    """A request we cannot decode is NOT answered — the server times out and denies (fail closed)."""
    nc = _FakeNats()
    resolver = _FakeResolver(_principal())
    responder = _responder(nc, resolver=resolver, policy=_FakePolicy(_perms()))

    await responder.handle_request(_FakeMsg(b"this-is-not-a-jwt", reply_subject="_INBOX.reply.2"))

    assert nc.replies == [], "an undecodable request produces no reply"
    assert resolver.seen == [], "resolution is never attempted on an undecodable request"


async def test_handle_request_without_a_reply_subject_is_ignored() -> None:
    """A request with no reply subject is a no-op (nowhere to answer)."""
    nc = _FakeNats()
    resolver = _FakeResolver(_principal())
    responder = _responder(nc, resolver=resolver, policy=_FakePolicy(_perms()))

    await responder.handle_request(_FakeMsg(_request_jwt().encode("utf-8"), reply_subject=""))

    assert nc.replies == [] and resolver.seen == [], "no reply subject → nothing is done"


async def test_start_subscribes_the_callout_subject_on_the_queue_group_then_stop_unsubscribes() -> None:
    """start() joins the callout subject in the queue group; stop() unsubscribes it."""
    nc = _FakeNats()
    responder = AuthCalloutResponder(
        nc,
        account_seed=generate_account_seed(),
        resolver=_FakeResolver(_principal()),
        policy=_FakePolicy(_perms()),
        queue_group="scriob-auth-callout",
    )

    await responder.start()
    assert nc.subscribed is not None
    assert nc.subscribed["subject"] == Subject.raw(AUTH_CALLOUT_SUBJECT), "subscribes the auth-callout subject"
    assert nc.subscribed["queue"] == "scriob-auth-callout", "joins the configured queue group"
    assert nc.subscribed["cb"] == responder.handle_request, "the callback is the responder's handler"

    await responder.stop()
    assert len(nc.unsubscribed) == 1, "stop() unsubscribes the callout subscription"


def test_from_secret_rejects_a_seed_that_is_not_a_usable_nkey() -> None:
    """A bad account signing key FAILS CLOSED at construction (never mints unverifiable responses)."""
    with pytest.raises(AuthAccountKeyError):
        AuthCalloutResponder.from_secret(
            "not-a-real-nkey-seed",
            nc=_FakeNats(),
            resolver=_FakeResolver(_principal()),
            policy=_FakePolicy(_perms()),
        )


def test_from_secret_accepts_a_valid_account_seed() -> None:
    """A valid account nkey seed builds a ready responder (str or bytes)."""
    responder = AuthCalloutResponder.from_secret(
        generate_account_seed(),
        nc=_FakeNats(),
        resolver=_FakeResolver(_principal()),
        policy=_FakePolicy(_perms()),
        account_name="SCRIOB",
    )
    assert isinstance(responder, AuthCalloutResponder)
