"""v0.13.9 auth C3 (enforce-only): the proxy verifies the Hub identity token + the per-call pop on
EVERY dispatch and re-stamps the verified identity.

The contract this pins (exercised end-to-end through the public dispatch surface):

- verification is UNCONDITIONAL and fail-closed -- there is no off/warn ladder. a call the proxy
  cannot authenticate (absent/invalid/expired/wrong-issuer token, unverifiable pop, replayed nonce)
  is rejected with ``IDENTITY_REFUSED`` (the token or the user assertion did not verify) or
  ``TOOL_POP_UNVERIFIED`` (the caller did not prove it holds the token's key) and never forwarded;
- on a VALID token the verified ``agent_id``/``user_id``/``customer_id`` OVERWRITE whatever the
  envelope claimed -- so a lying envelope cannot impersonate another agent or inject a user_id; the
  verified identity is what reaches BOTH the authorizer and the tool pod;
- the per-call pop proves the caller holds the token's bound key for THIS token + THIS body, once
  (single-use via a mandatory replay guard) -- so a leaked token alone is unusable.
"""

from __future__ import annotations

from threetears.core.testing.replay_guard import FakeReplayGuard

from datetime import UTC, datetime

import jwt

import asyncio
import json
import logging
import time
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid7

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from threetears.agent.tools.context_envelope import CallContext

from threetears.core.security.identity_token import (
    IDENTITY_REFUSED,
    IDENTITY_REFUSED_MESSAGE,
    PLATFORM_CUSTOMER_SENTINEL,
    IdentityClaims,
    build_jwks,
    canonical_call_hash,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
)
from threetears.core.security import TOOL_POP_LEDGER_UNAVAILABLE, TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE
from threetears.core.security.pop import access_token_hash, make_pop_proof
from threetears.core.exceptions import DataLayerUnavailableError
from threetears.nats import IncomingMessage, KvError, set_default_namespace
from threetears.registry.auth import AllowAllAuthorizer, AllowAllLimitGuard
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.proxy import CallProxy, ProxyCallRequest, ProxyCallResponse

from ._copies import uniform_entry

_ISS = "hub"
_KID = "kid-1"
_TOOL = "threetears.calculator"

# one ephemeral holder key whose thumbprint binds every token's ``cnf``; the per-call pop proves
# possession of it. shared across this module so the proxy's pop gate accepts the authenticated
# requests the identity-forwarding tests build.
_HOLDER_KEY = Ed25519PrivateKey.generate()
_HOLDER_CNF = jwk_thumbprint(_HOLDER_KEY.public_key())


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace("test")


@pytest.fixture
def hub() -> tuple[Any, dict[str, Any]]:
    """a Hub signing key + the matching JWKS the proxy verifies against."""
    priv, pub = generate_signing_keypair()
    return priv, build_jwks({_KID: pub})


def _token(
    priv: Any,
    *,
    sub: UUID,
    customer_id: UUID | str,
    user_id: UUID | None,
    exp_delta: int = 600,
    iss: str = _ISS,
    cnf: str | None = None,
    conversation_id: UUID | None = None,
) -> str:
    now = int(time.time())
    claims = IdentityClaims(
        sub=str(sub),
        customer_id=str(customer_id),
        user_id=str(user_id) if user_id is not None else None,
        sid="sid-1",
        pod_id="pod-1",
        iss=iss,
        iat=now,
        exp=now + exp_delta,
        cnf=cnf,
        conversation_id=str(conversation_id) if conversation_id is not None else None,
    )
    return sign_identity_token(claims, signing_key=priv, kid=_KID)


def _request(
    *,
    agent_id: UUID,
    user_id: UUID | None = None,
    customer_id: UUID | None = None,
    token: str | None = None,
) -> ProxyCallRequest:
    """a request carrying NO pop -- used by the identity-gate REJECTION tests, which short-circuit
    at the identity gate (the first gate) before pop verification is reached."""
    return ProxyCallRequest(
        tool_name=_TOOL,
        tool_version="1.0.0",
        arguments={"expression": "2+2"},
        context=CallContext(
            agent_id=agent_id,
            user_id=user_id,
            customer_id=customer_id,
            correlation_id=uuid7(),
            identity_token=token,
        ),
    )


def _authed_request(
    priv: Any,
    *,
    token_sub: UUID,
    token_customer: UUID | str,
    token_user: UUID | None,
    envelope_agent: UUID,
    envelope_user: UUID | None = None,
    exp_delta: int = 600,
    arguments: dict[str, Any] | None = None,
    user_assertion: str | None = None,
    conversation_id: UUID | None = None,
) -> ProxyCallRequest:
    """a FULLY-authenticated request: a cnf-bound token (sub=token_sub) + a matching per-call pop.

    the ENVELOPE is free to claim a different (or null) agent/user than the token so the verified
    re-stamp (which overwrites the envelope's claim with the token's identity) is observable through
    the forwarded payload / the authorizer. the pop binds to THIS body so the unconditional pop gate
    accepts it. ``user_assertion`` is the optional Hub-minted, cnf-LESS user-identity token the proxy
    verifies, binds to the handshake token, and re-stamps ``user_id`` from. ``conversation_id`` rides
    on the CallContext -- the proxy re-checks the user-assertion's bound conversation_id against it.
    """
    args = arguments if arguments is not None else {"expression": "2+2"}
    correlation_id = uuid7()
    token = _token(
        priv,
        sub=token_sub,
        customer_id=token_customer,
        user_id=token_user,
        exp_delta=exp_delta,
        cnf=_HOLDER_CNF,
    )
    body_hash = canonical_call_hash(_TOOL, args, str(correlation_id))
    pop = make_pop_proof(
        holder_key=_HOLDER_KEY,
        access_token_hash=access_token_hash(token),
        body_hash=body_hash,
        nonce=str(uuid7()),
        iat=int(time.time()),
    )
    return ProxyCallRequest(
        tool_name=_TOOL,
        tool_version="1.0.0",
        arguments=args,
        context=CallContext(
            agent_id=envelope_agent,
            user_id=envelope_user,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            identity_token=token,
            user_identity_token=user_assertion,
        ),
        pop=pop,
    )


def _user_assertion(
    priv: Any,
    *,
    sub: UUID,
    customer_id: UUID | str,
    user_id: UUID,
    exp_delta: int = 3600,
    conversation_id: UUID | None = None,
) -> str:
    """mint a Hub-style, cnf-LESS user-assertion (carries the per-turn verified user_id).

    signed by the same Hub key the handshake token uses, so the proxy verifies it against the SAME
    JWKS/issuer. carries NO ``cnf`` (the Hub cannot know the target pod's holder key at mint) -- the
    proxy binds it to the handshake token by ``sub`` + ``customer_id`` (and to ``conversation_id``)
    instead of by pop.
    """
    return _token(
        priv, sub=sub, customer_id=customer_id, user_id=user_id, exp_delta=exp_delta, conversation_id=conversation_id
    )


def _entry() -> CatalogEntry:
    return uniform_entry(
        tool_name=_TOOL,
        tool_version="1.0.0",
        full_name=f"{_TOOL}@1.0.0",
        description="test tool",
        input_schema={"type": "object", "properties": {}},
        endpoints=[ToolEndpoint(pod_id="pod-001", status="available")],
    )


def _tool_reply() -> bytes:
    return (
        ProxyCallResponse(success=True, content="ok", context=CallContext(correlation_id=uuid7()))
        .model_dump_json()
        .encode("utf-8")
    )


async def _catalog() -> ToolCatalog:
    catalog = ToolCatalog()
    await catalog.register(_entry())
    return catalog


def _raising_provider() -> dict[str, Any]:
    """a JWKS provider that fails the way a Hub-backed network fetch could."""
    raise ConnectionError("hub jwks endpoint unavailable")


class TestProxyConstruction:
    """the enforce-only proxy structurally REQUIRES a pop replay guard."""

    def test_requires_a_replay_guard(self) -> None:
        # without a replay guard a captured pop could be replayed verbatim in-window; the guard is a
        # required constructor argument so a guardless (replay-vulnerable) proxy cannot be built.
        with pytest.raises(TypeError):
            CallProxy(ToolCatalog(), AllowAllAuthorizer())  # type: ignore[call-arg]


class TestDispatchIdentityEnforcement:
    """end-to-end through ``handle_call``: rejection without forwarding, and verified-identity flow."""

    async def _drive(
        self,
        jwks_provider: Any,
        req: ProxyCallRequest,
        *,
        authorizer: Any = None,
    ) -> AsyncMock:
        proxy = CallProxy(
            await _catalog(),
            authorizer if authorizer is not None else AllowAllAuthorizer(),
            FakeReplayGuard(fresh=True),
            limit_guard=AllowAllLimitGuard(),
            namespace="test",
            jwks_provider=jwks_provider,
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_tool_reply())
        await proxy.start(nc)
        msg = IncomingMessage(
            data=req.model_dump_json().encode("utf-8"),
            reply_subject="reply.subject",
            subject="test.tools.call",
        )
        await proxy.handle_call(msg)
        await asyncio.sleep(0)
        return nc

    @staticmethod
    def _forwarded_context(nc: AsyncMock) -> dict[str, Any]:
        payload = json.loads(nc.request_raw.call_args.kwargs["payload"])
        context: dict[str, Any] = payload["context"]
        return context

    @staticmethod
    def _reply(nc: AsyncMock) -> ProxyCallResponse:
        message: ProxyCallResponse = nc.publish_reply.call_args.kwargs["message"]
        return message

    # -- fail closed, verified identity flows on success --

    @pytest.mark.asyncio
    async def test_rejects_unverified_without_forwarding(self, hub: tuple[Any, dict[str, Any]]) -> None:
        _priv, jwks = hub
        nc = await self._drive(lambda: jwks, _request(agent_id=uuid7(), token=None))
        nc.request_raw.assert_not_called()
        reply = self._reply(nc)
        assert reply.success is False
        assert reply.error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_forwards_verified_identity_to_pod_over_a_lie(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        real_agent, real_cust = uuid7(), uuid7()
        # the envelope claims a DIFFERENT agent than the token; the pod must get the verified one
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=real_agent,
                token_customer=real_cust,
                token_user=None,
                envelope_agent=uuid7(),
            ),
        )
        nc.request_raw.assert_called_once()
        forwarded = self._forwarded_context(nc)
        assert forwarded["agent_id"] == str(real_agent)
        assert forwarded["customer_id"] == str(real_cust)

    @pytest.mark.asyncio
    async def test_verified_user_id_none_overrides_claimed_user(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # privilege guard: a token with NO user_id blanks the envelope's claimed user_id.
        priv, jwks = hub
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=uuid7(),
                token_customer=uuid7(),
                token_user=None,
                envelope_agent=uuid7(),
                envelope_user=uuid7(),
            ),
        )
        nc.request_raw.assert_called_once()
        assert self._forwarded_context(nc)["user_id"] is None

    @pytest.mark.asyncio
    async def test_restamped_identity_reaches_authorizer(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        real_agent, real_user = uuid7(), uuid7()
        captured: dict[str, str | None] = {}

        class _RecordingAuthorizer:
            async def is_authorized(
                self,
                agent_id: str,
                user_id: str | None,
                tool_name: str,
                tool_version: str,
                *,
                principal_is_tool_pod: bool,
            ) -> bool:
                captured["agent_id"] = agent_id
                captured["user_id"] = user_id
                return True

        # envelope claims a different agent AND a different user than the token
        await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=real_agent,
                token_customer=uuid7(),
                token_user=real_user,
                envelope_agent=uuid7(),
                envelope_user=uuid7(),
            ),
            authorizer=_RecordingAuthorizer(),
        )
        assert captured["agent_id"] == str(real_agent)  # authorizer saw the VERIFIED agent
        assert captured["user_id"] == str(real_user)  # ...and the VERIFIED user

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flavor", ["invalid", "absent", "no_provider", "expired", "wrong_issuer", "provider_raises"]
    )
    async def test_rejects_each_failure_mode(self, hub: tuple[Any, dict[str, Any]], flavor: str) -> None:
        priv, jwks = hub
        provider: Any = lambda: jwks
        if flavor == "invalid":
            req = _request(agent_id=uuid7(), token="not.a.valid.jws")
        elif flavor == "absent":
            req = _request(agent_id=uuid7(), token=None)
        elif flavor == "no_provider":
            provider = None
            req = _request(agent_id=uuid7(), token=_token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None))
        elif flavor == "expired":
            req = _request(
                agent_id=uuid7(), token=_token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None, exp_delta=-120)
            )
        elif flavor == "wrong_issuer":
            req = _request(
                agent_id=uuid7(), token=_token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None, iss="evil")
            )
        else:  # provider_raises -- a flaky/Hub-down JWKS fetch must reject cleanly, never hang
            provider = _raising_provider
            req = _request(agent_id=uuid7(), token=_token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None))
        nc = await self._drive(provider, req)
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_forwards_within_leeway(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a token expired by less than the proxy's clock-skew leeway is still accepted + forwarded.
        priv, jwks = hub
        real_agent = uuid7()
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=real_agent,
                token_customer=uuid7(),
                token_user=None,
                envelope_agent=uuid7(),
                exp_delta=-30,
            ),
        )
        nc.request_raw.assert_called_once()
        assert self._forwarded_context(nc)["agent_id"] == str(real_agent)

    @pytest.mark.asyncio
    async def test_malformed_non_uuid_handshake_claim_denies(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a SIGNED-but-malformed (non-UUID) handshake claim must fail closed at the identity gate, not
        # escape as an uncaught ValueError that would hang the dispatch with no reply. locks the NIT
        # fix that moved the handshake UUID conversions INSIDE the verify try (symmetric with the
        # user-assertion's user_id conversion).
        priv, jwks = hub
        now = int(time.time())
        claims = IdentityClaims(
            sub="not-a-uuid",  # signed, but not coercible to UUID
            customer_id=str(uuid7()),
            user_id=None,
            sid="sid-1",
            pod_id="pod-1",
            iss=_ISS,
            iat=now,
            exp=now + 600,
        )
        token = sign_identity_token(claims, signing_key=priv, kid=_KID)
        nc = await self._drive(lambda: jwks, _request(agent_id=uuid7(), token=token))
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED


class TestDispatchToolPodPrincipal:
    """a TOOL POD's token carries the platform customer sentinel where a customer UUID would be.

    the door reads that claim, off the SIGNED token, as ``customer_id=None`` plus a tool-pod
    mark the authorizer receives beside the ids. a UUID claim is an agent and carries no mark; any
    other non-UUID claim is a malformed token and fails closed exactly as before.
    """

    async def _drive(
        self,
        jwks_provider: Any,
        req: ProxyCallRequest,
        *,
        authorizer: Any = None,
    ) -> AsyncMock:
        proxy = CallProxy(
            await _catalog(),
            authorizer if authorizer is not None else AllowAllAuthorizer(),
            FakeReplayGuard(fresh=True),
            limit_guard=AllowAllLimitGuard(),
            namespace="test",
            jwks_provider=jwks_provider,
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_tool_reply())
        await proxy.start(nc)
        msg = IncomingMessage(
            data=req.model_dump_json().encode("utf-8"),
            reply_subject="reply.subject",
            subject="test.tools.call",
        )
        await proxy.handle_call(msg)
        await asyncio.sleep(0)
        return nc

    @staticmethod
    def _forwarded_context(nc: AsyncMock) -> dict[str, Any]:
        payload = json.loads(nc.request_raw.call_args.kwargs["payload"])
        context: dict[str, Any] = payload["context"]
        return context

    @staticmethod
    def _reply(nc: AsyncMock) -> ProxyCallResponse:
        message: ProxyCallResponse = nc.publish_reply.call_args.kwargs["message"]
        return message

    @staticmethod
    def _recording_authorizer(captured: dict[str, Any]) -> Any:
        class _RecordingAuthorizer:
            async def is_authorized(
                self,
                agent_id: str,
                user_id: str | None,
                tool_name: str,
                tool_version: str,
                *,
                principal_is_tool_pod: bool,
            ) -> bool:
                captured["agent_id"] = agent_id
                captured["user_id"] = user_id
                captured["principal_is_tool_pod"] = principal_is_tool_pod
                return True

        return _RecordingAuthorizer()

    @pytest.mark.asyncio
    async def test_sentinel_customer_forwards_no_customer_and_marks_the_pod(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        priv, jwks = hub
        pod_id = uuid7()
        captured: dict[str, Any] = {}
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=pod_id,
                token_customer=PLATFORM_CUSTOMER_SENTINEL,
                token_user=None,
                envelope_agent=uuid7(),
            ),
            authorizer=self._recording_authorizer(captured),
        )
        nc.request_raw.assert_called_once()
        forwarded = self._forwarded_context(nc)
        assert forwarded["agent_id"] == str(pod_id)
        assert forwarded["customer_id"] is None  # no customer was stamped, not a UUID and not the literal
        assert forwarded["user_id"] is None
        assert captured["agent_id"] == str(pod_id)
        assert captured["user_id"] is None
        assert captured["principal_is_tool_pod"] is True

    @pytest.mark.asyncio
    async def test_uuid_customer_is_an_agent_and_carries_no_mark(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        captured: dict[str, Any] = {}
        await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=uuid7(),
                token_customer=uuid7(),
                token_user=None,
                envelope_agent=uuid7(),
            ),
            authorizer=self._recording_authorizer(captured),
        )
        assert captured["principal_is_tool_pod"] is False

    @pytest.mark.asyncio
    async def test_a_non_uuid_customer_that_is_not_the_sentinel_still_fails_closed(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        # the sentinel is the ONE non-UUID value the hub mints on purpose; a signed token carrying
        # any other non-UUID customer is malformed and must never read as a platform principal.
        priv, jwks = hub
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=uuid7(),
                token_customer="aibots-platform-but-not-quite",
                token_user=None,
                envelope_agent=uuid7(),
            ),
        )
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_a_pod_presenting_a_user_assertion_is_refused(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # the assertion is minted with the pod's own sub AND the sentinel customer, so the customer
        # binding check -- which compares claim strings -- would admit it. only the explicit refusal of
        # any assertion on a pod token can turn this call away, which is the rule under test: a pod
        # acts on nobody's behalf and cannot borrow a user by attaching one.
        priv, jwks = hub
        pod_id, conv = uuid7(), uuid7()
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=pod_id,
                token_customer=PLATFORM_CUSTOMER_SENTINEL,
                token_user=None,
                envelope_agent=uuid7(),
                conversation_id=conv,
                user_assertion=_user_assertion(
                    priv,
                    sub=pod_id,
                    customer_id=PLATFORM_CUSTOMER_SENTINEL,
                    user_id=uuid7(),
                    conversation_id=conv,
                ),
            ),
        )
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED


# ---------------------------------------------------------------------------
# v0.13.9 enforce-flip final piece: the Hub-minted, cnf-less user-assertion (two-token design).
#
# the handshake token is one-per-pod and carries NO user_id (the user varies per turn), so the
# proxy re-stamp nulls user_id on a user-driven tool call -> RBAC two-sided-denies. a SECOND token
# -- a Hub-minted, cnf-less user-assertion -- carries the per-turn verified user_id; the proxy
# verifies it, BINDS it to the handshake token (sub + customer must match), and re-stamps user_id.
# ---------------------------------------------------------------------------


class TestDispatchUserAssertion:
    """end-to-end through ``handle_call``: the proxy binds + re-stamps the user-assertion's user_id.

    these run with a valid (identity + pop passing) handshake token so the case under test is the
    USER-ASSERTION gate specifically. the handshake token carries ``token_user=None`` (an agent
    handshake token never carries a user) so any non-null forwarded user_id can ONLY have come from
    the bound user-assertion.
    """

    async def _drive(
        self,
        jwks_provider: Any,
        req: ProxyCallRequest,
        *,
        authorizer: Any = None,
    ) -> AsyncMock:
        proxy = CallProxy(
            await _catalog(),
            authorizer if authorizer is not None else AllowAllAuthorizer(),
            FakeReplayGuard(fresh=True),
            limit_guard=AllowAllLimitGuard(),
            namespace="test",
            jwks_provider=jwks_provider,
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_tool_reply())
        await proxy.start(nc)
        msg = IncomingMessage(
            data=req.model_dump_json().encode("utf-8"),
            reply_subject="reply.subject",
            subject="test.tools.call",
        )
        await proxy.handle_call(msg)
        await asyncio.sleep(0)
        return nc

    @staticmethod
    def _forwarded_context(nc: AsyncMock) -> dict[str, Any]:
        payload = json.loads(nc.request_raw.call_args.kwargs["payload"])
        context: dict[str, Any] = payload["context"]
        return context

    @staticmethod
    def _reply(nc: AsyncMock) -> ProxyCallResponse:
        message: ProxyCallResponse = nc.publish_reply.call_args.kwargs["message"]
        return message

    @pytest.mark.asyncio
    async def test_bound_user_assertion_restamps_user_id_to_pod(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # the handshake token carries NO user; a bound user-assertion supplies the verified user_id
        # that reaches the pod.
        priv, jwks = hub
        agent, cust, real_user, conv = uuid7(), uuid7(), uuid7(), uuid7()
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=uuid7(),
                envelope_user=uuid7(),  # the envelope's claimed user is discarded
                conversation_id=conv,
                user_assertion=_user_assertion(
                    priv, sub=agent, customer_id=cust, user_id=real_user, conversation_id=conv
                ),
            ),
        )
        nc.request_raw.assert_called_once()
        assert self._forwarded_context(nc)["user_id"] == str(real_user)

    @pytest.mark.asyncio
    async def test_bound_user_assertion_reaches_authorizer(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        agent, cust, real_user = uuid7(), uuid7(), uuid7()
        captured: dict[str, str | None] = {}

        class _RecordingAuthorizer:
            async def is_authorized(
                self,
                agent_id: str,
                user_id: str | None,
                tool_name: str,
                tool_version: str,
                *,
                principal_is_tool_pod: bool,
            ) -> bool:
                captured["agent_id"] = agent_id
                captured["user_id"] = user_id
                return True

        conv = uuid7()
        await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=uuid7(),
                conversation_id=conv,
                user_assertion=_user_assertion(
                    priv, sub=agent, customer_id=cust, user_id=real_user, conversation_id=conv
                ),
            ),
            authorizer=_RecordingAuthorizer(),
        )
        assert captured["agent_id"] == str(agent)
        assert captured["user_id"] == str(real_user)  # RBAC saw the VERIFIED, bound user

    @pytest.mark.asyncio
    async def test_absent_user_assertion_leaves_user_id_none(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # back-compat: no user-assertion -> user_id stays as the handshake token's (None here).
        priv, jwks = hub
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=uuid7(),
                token_customer=uuid7(),
                token_user=None,
                envelope_agent=uuid7(),
                user_assertion=None,
            ),
        )
        nc.request_raw.assert_called_once()
        assert self._forwarded_context(nc)["user_id"] is None

    @pytest.mark.asyncio
    async def test_empty_user_assertion_treated_as_absent(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # an empty-string user-assertion (a caller that built the envelope without one) is ABSENT,
        # not a verification failure: the call forwards with the handshake token's user_id (None).
        priv, jwks = hub
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=uuid7(),
                token_customer=uuid7(),
                token_user=None,
                envelope_agent=uuid7(),
                user_assertion="",
            ),
        )
        nc.request_raw.assert_called_once()
        assert self._forwarded_context(nc)["user_id"] is None

    @pytest.mark.asyncio
    async def test_sub_mismatch_denies(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a user-assertion minted for a DIFFERENT agent must not be usable here (cross-agent replay).
        priv, jwks = hub
        agent, cust = uuid7(), uuid7()
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=uuid7(),
                user_assertion=_user_assertion(priv, sub=uuid7(), customer_id=cust, user_id=uuid7()),
            ),
        )
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_customer_mismatch_denies(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a user-assertion for a DIFFERENT customer must not bind (cross-customer replay).
        priv, jwks = hub
        agent, cust = uuid7(), uuid7()
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=uuid7(),
                user_assertion=_user_assertion(priv, sub=agent, customer_id=uuid7(), user_id=uuid7()),
            ),
        )
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_expired_user_assertion_denies(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        agent, cust = uuid7(), uuid7()
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=uuid7(),
                user_assertion=_user_assertion(priv, sub=agent, customer_id=cust, user_id=uuid7(), exp_delta=-120),
            ),
        )
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_invalid_user_assertion_denies(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a non-JWS / unverifiable user-assertion is rejected fail-closed, never forwarded.
        priv, jwks = hub
        agent, cust = uuid7(), uuid7()
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=uuid7(),
                user_assertion="not.a.valid.jws",
            ),
        )
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_null_user_id_user_assertion_denies(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a present, signed, BOUND user-assertion whose user_id claim is NULL is not a valid per-turn
        # user -> deny fail-closed. locks proxy.py's "user-assertion carries no user_id" contract so a
        # user-less assertion can never re-stamp a None user_id past the gate as if it were verified.
        priv, jwks = hub
        agent, cust = uuid7(), uuid7()
        null_user_assertion = _token(priv, sub=agent, customer_id=cust, user_id=None)
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=uuid7(),
                user_assertion=null_user_assertion,
            ),
        )
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_user_assertion_for_same_conversation_is_accepted(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # CONVERSATION-BINDING accept: an assertion minted for conversation C, on a call for C, is
        # accepted and re-stamps the verified user_id.
        priv, jwks = hub
        agent, cust, real_user, conv = uuid7(), uuid7(), uuid7(), uuid7()
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=uuid7(),
                conversation_id=conv,
                user_assertion=_user_assertion(
                    priv, sub=agent, customer_id=cust, user_id=real_user, conversation_id=conv
                ),
            ),
        )
        nc.request_raw.assert_called_once()
        assert self._forwarded_context(nc)["user_id"] == str(real_user)

    @pytest.mark.asyncio
    async def test_user_assertion_replayed_into_different_conversation_denies(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        # CONVERSATION-BINDING deny: an assertion minted for conversation C, REPLAYED on a call for a
        # DIFFERENT conversation D, is rejected fail-closed -- the cross-conversation impersonation a
        # compromised pod could attempt (inject a captured user-U assertion into another conversation)
        # is denied, so U is never acted-as outside conversation C.
        priv, jwks = hub
        agent, cust = uuid7(), uuid7()
        conv_minted_for, conv_of_call = uuid7(), uuid7()  # C != D
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=uuid7(),
                conversation_id=conv_of_call,  # the call is for conversation D...
                # ...but the captured assertion was minted for conversation C
                user_assertion=_user_assertion(
                    priv, sub=agent, customer_id=cust, user_id=uuid7(), conversation_id=conv_minted_for
                ),
            ),
        )
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_user_assertion_with_no_conversation_id_denies(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a user-driven turn ALWAYS mints with a conversation_id; a present assertion lacking one is a
        # denial -- "no conversation_id" must NOT skip the conversation-binding check.
        priv, jwks = hub
        agent, cust = uuid7(), uuid7()
        nc = await self._drive(
            lambda: jwks,
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=uuid7(),
                conversation_id=uuid7(),  # the call has a conversation; the assertion has none
                user_assertion=_user_assertion(priv, sub=agent, customer_id=cust, user_id=uuid7()),
            ),
        )
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_pop_still_verifies_against_handshake_token_with_user_assertion(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        # the pop path is UNCHANGED: it binds the HANDSHAKE token's cnf. adding a (cnf-less)
        # user-assertion must not disturb it -- a fully-authed request with a bound assertion still
        # passes the pop gate and forwards.
        priv, jwks = hub
        agent, cust, real_user, conv = uuid7(), uuid7(), uuid7(), uuid7()
        guard = FakeReplayGuard(fresh=True)
        proxy = CallProxy(
            await _catalog(),
            AllowAllAuthorizer(),
            guard,
            limit_guard=AllowAllLimitGuard(),
            namespace="test",
            jwks_provider=lambda: jwks,
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_tool_reply())
        await proxy.start(nc)
        req = _authed_request(
            priv,
            token_sub=agent,
            token_customer=cust,
            token_user=None,
            envelope_agent=uuid7(),
            conversation_id=conv,
            user_assertion=_user_assertion(priv, sub=agent, customer_id=cust, user_id=real_user, conversation_id=conv),
        )
        msg = IncomingMessage(
            data=req.model_dump_json().encode("utf-8"),
            reply_subject="reply.subject",
            subject="test.tools.call",
        )
        await proxy.handle_call(msg)
        await asyncio.sleep(0)
        nc.request_raw.assert_called_once()
        assert len(guard.seen) == 1  # the pop gate DID run (consulted the replay guard) and passed


# ---------------------------------------------------------------------------
# v0.13.9 auth W4c: per-call proof-of-possession (the agent->proxy holder binding)
# ---------------------------------------------------------------------------


def _pop_request(
    priv: Any,
    holder_key: Ed25519PrivateKey,
    *,
    correlation_id: UUID,
    pop_body_args: dict[str, Any] | None = None,
    include_pop: bool = True,
    bind_cnf: bool = True,
) -> ProxyCallRequest:
    """a request carrying a cnf-bound token + a matching per-call pop proof.

    ``bind_cnf=False`` mints a token with NO holder binding; ``include_pop=False`` omits the proof;
    ``pop_body_args`` computes the proof's body hash from DIFFERENT arguments than the request
    actually carries (a spliced proof).
    """
    args = {"expression": "2+2"}
    cnf = jwk_thumbprint(holder_key.public_key()) if bind_cnf else None
    token = _token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None, cnf=cnf)
    pop: str | None = None
    if include_pop:
        body_hash = canonical_call_hash(_TOOL, pop_body_args or args, str(correlation_id))
        pop = make_pop_proof(
            holder_key=holder_key,
            access_token_hash=access_token_hash(token),
            body_hash=body_hash,
            nonce=str(uuid7()),
            iat=int(time.time()),
        )
    return ProxyCallRequest(
        tool_name=_TOOL,
        tool_version="1.0.0",
        arguments=args,
        context=CallContext(
            agent_id=uuid7(),
            correlation_id=correlation_id,
            identity_token=token,
        ),
        pop=pop,
    )


class _RekeyingProvider:
    """models a verifier JWKS cache that is STALE for the token's signing key until ONE reactive
    refresh brings it current -- the Hub re-key / pod-move scenario B5 self-heals. ``refresh_calls``
    counts reactive refreshes so a test can assert "exactly one, never a stampede"."""

    def __init__(self, *, stale: dict[str, Any], fresh: dict[str, Any]) -> None:
        self._jwks = stale
        self._fresh = fresh
        self.refresh_calls = 0

    def __call__(self) -> dict[str, Any]:
        return self._jwks

    async def refresh_now(self) -> bool:
        self.refresh_calls += 1
        self._jwks = self._fresh
        return True


class _CountingProvider:
    """a fixed-JWKS provider that counts reactive refreshes -- which must NOT happen for a token
    whose verification fails for a reason a refresh cannot fix (expired / bad signature)."""

    def __init__(self, jwks: dict[str, Any]) -> None:
        self._jwks = jwks
        self.refresh_calls = 0

    def __call__(self) -> dict[str, Any]:
        return self._jwks

    async def refresh_now(self) -> bool:
        self.refresh_calls += 1
        return True


class TestDispatchReactiveJwksRefresh:
    """B5: a kid-not-in-cache miss triggers exactly ONE reactive JWKS refresh + re-verify (Hub re-key
    self-heal); an expired / bad-signature token does NOT, so a flood of bad tokens cannot stampede
    the Hub."""

    async def _drive(self, provider: Any, req: ProxyCallRequest) -> AsyncMock:
        proxy = CallProxy(
            await _catalog(),
            AllowAllAuthorizer(),
            FakeReplayGuard(fresh=True),
            limit_guard=AllowAllLimitGuard(),
            namespace="test",
            jwks_provider=provider,
            jwks_refresh=provider.refresh_now,
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_tool_reply())
        await proxy.start(nc)
        msg = IncomingMessage(
            data=req.model_dump_json().encode("utf-8"),
            reply_subject="reply.subject",
            subject="test.tools.call",
        )
        await proxy.handle_call(msg)
        await asyncio.sleep(0)
        return nc

    @staticmethod
    def _reply(nc: AsyncMock) -> ProxyCallResponse:
        message: ProxyCallResponse = nc.publish_reply.call_args.kwargs["message"]
        return message

    @pytest.mark.asyncio
    async def test_kid_miss_triggers_one_reactive_refresh_then_forwards(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # the cache is stale (empty) for the token's key after a Hub re-key; the key arrives on the
        # FIRST reactive refresh, so a VALID token self-heals and forwards rather than being rejected
        # for up to a full steady refresh interval.
        priv, jwks = hub
        real_agent = uuid7()
        provider = _RekeyingProvider(stale={"keys": []}, fresh=jwks)
        nc = await self._drive(
            provider,
            _authed_request(
                priv, token_sub=real_agent, token_customer=uuid7(), token_user=None, envelope_agent=uuid7()
            ),
        )
        assert provider.refresh_calls == 1  # EXACTLY one reactive refresh
        nc.request_raw.assert_called_once()  # re-verify succeeded -> the call forwarded
        forwarded = json.loads(nc.request_raw.call_args.kwargs["payload"])["context"]
        assert forwarded["agent_id"] == str(real_agent)  # the verified, re-stamped identity

    @pytest.mark.asyncio
    async def test_expired_token_does_not_trigger_refresh(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # an expired token is signed under a key the cache HOLDS -> the failure is expiry, not a
        # kid-miss, so it must NOT provoke a Hub refresh (else every bad token becomes a Hub request).
        priv, jwks = hub
        provider = _CountingProvider(jwks)
        nc = await self._drive(
            provider,
            _request(
                agent_id=uuid7(),
                token=_token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None, exp_delta=-3600),
            ),
        )
        assert provider.refresh_calls == 0  # NO reactive refresh on an expired token
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_invalid_signature_does_not_trigger_refresh(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a token signed by a DIFFERENT key but carrying the cache's kid fails the signature check
        # (the key IS present, so it is not a kid-miss) -> must NOT provoke a refresh.
        _priv, jwks = hub
        other_priv = Ed25519PrivateKey.generate()
        forged = _token(other_priv, sub=uuid7(), customer_id=uuid7(), user_id=None)  # signed by other_priv, kid=_KID
        provider = _CountingProvider(jwks)
        nc = await self._drive(provider, _request(agent_id=uuid7(), token=forged))
        assert provider.refresh_calls == 0  # bad signature against a PRESENT key is not refreshable
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == IDENTITY_REFUSED

    @pytest.mark.asyncio
    async def test_kid_miss_unresolved_refreshes_once_then_rejects(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a kid the Hub genuinely does not have: the reactive refresh runs ONCE, the key still is not
        # there, and the call is rejected. proves the "only once" bound -- a flood of such tokens
        # cannot turn into a Hub stampede (and refresh_now is itself rate-limited on top).
        priv, _jwks = hub
        provider = _RekeyingProvider(stale={"keys": []}, fresh={"keys": []})  # refresh never brings the key
        nc = await self._drive(
            provider,
            _request(agent_id=uuid7(), token=_token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None)),
        )
        assert provider.refresh_calls == 1  # tried exactly once
        nc.request_raw.assert_not_called()  # still rejected (the key truly is not at the Hub)
        assert self._reply(nc).error_code == IDENTITY_REFUSED


class TestVerificationObservability:
    """B8: the verify-failure log carries the exception MESSAGE so a stale-JWKS (kid-miss) failure is
    distinguishable from an expired-token failure in production (the gap that masked the datasource
    failure). The message is the STRUCTURAL reason -- never token or key material."""

    async def _drive_capture(self, provider: Any, req: ProxyCallRequest, caplog: pytest.LogCaptureFixture) -> None:
        proxy = CallProxy(
            await _catalog(),
            AllowAllAuthorizer(),
            FakeReplayGuard(fresh=True),
            limit_guard=AllowAllLimitGuard(),
            namespace="test",
            jwks_provider=provider,
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_tool_reply())
        await proxy.start(nc)
        msg = IncomingMessage(
            data=req.model_dump_json().encode("utf-8"),
            reply_subject="reply.subject",
            subject="test.tools.call",
        )
        with caplog.at_level(logging.WARNING, logger="threetears.registry.proxy"):
            await proxy.handle_call(msg)
            await asyncio.sleep(0)

    @staticmethod
    def _identity_detail(caplog: pytest.LogCaptureFixture) -> str:
        rec = next(r for r in caplog.records if r.getMessage() == "identity verification failed; rejecting call")
        extra = getattr(rec, "extra_data", None)
        assert extra is not None, "the identity-verification-failed log must carry structured extra_data"
        detail: str = extra["detail"]
        return detail

    @pytest.mark.asyncio
    async def test_kid_miss_vs_expired_logs_are_distinguishable(
        self, hub: tuple[Any, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ) -> None:
        priv, jwks = hub
        # (1) an EXPIRED token, with the signing key present in the cache.
        caplog.clear()
        await self._drive_capture(
            lambda: jwks,
            _request(
                agent_id=uuid7(),
                token=_token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None, exp_delta=-3600),
            ),
            caplog,
        )
        expired_detail = self._identity_detail(caplog)
        # (2) a kid-MISS: the cache holds a DIFFERENT kid, so the token's key is absent (stale cache).
        caplog.clear()
        other_jwks = build_jwks({"kid-other": generate_signing_keypair()[1]})
        token = _token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None)
        await self._drive_capture(lambda: other_jwks, _request(agent_id=uuid7(), token=token), caplog)
        kid_miss_detail = self._identity_detail(caplog)

        # the two failure modes log DIFFERENT, recognizable reasons -- the gap B8 closes.
        assert "no JWKS key matches the token kid" in kid_miss_detail
        assert "ExpiredSignature" in expired_detail
        assert expired_detail != kid_miss_detail
        # security: the structured detail is the structural reason, never the token string itself.
        assert token not in kid_miss_detail
        assert token not in expired_detail

    @pytest.mark.asyncio
    async def test_absent_token_logs_its_own_reason(
        self, hub: tuple[Any, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ) -> None:
        # a third, distinct failure mode: no token at all -> "token absent", not a JWKS/expiry reason.
        _priv, jwks = hub
        caplog.clear()
        await self._drive_capture(lambda: jwks, _request(agent_id=uuid7(), token=None), caplog)
        assert "identity token absent" in self._identity_detail(caplog)


class TestDispatchPopEnforcement:
    """end-to-end through ``handle_call``: the proxy requires a valid per-call pop on every dispatch.

    pop verification is self-contained (it re-verifies the token for a trusted cnf); these run with
    a valid (identity-passing) token so the case under test is the POP gate specifically.
    """

    async def _drive(
        self,
        jwks_provider: Any,
        req: ProxyCallRequest,
        *,
        pop_replay_guard: Any = None,
    ) -> AsyncMock:
        proxy = CallProxy(
            await _catalog(),
            AllowAllAuthorizer(),
            pop_replay_guard if pop_replay_guard is not None else FakeReplayGuard(fresh=True),
            limit_guard=AllowAllLimitGuard(),
            namespace="test",
            jwks_provider=jwks_provider,
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_tool_reply())
        await proxy.start(nc)
        msg = IncomingMessage(
            data=req.model_dump_json().encode("utf-8"),
            reply_subject="reply.subject",
            subject="test.tools.call",
        )
        await proxy.handle_call(msg)
        await asyncio.sleep(0)
        return nc

    @staticmethod
    def _reply(nc: AsyncMock) -> ProxyCallResponse:
        message: ProxyCallResponse = nc.publish_reply.call_args.kwargs["message"]
        return message

    @pytest.mark.asyncio
    async def test_forwards_a_valid_pop(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7())
        nc = await self._drive(lambda: jwks, req, pop_replay_guard=FakeReplayGuard(fresh=True))
        nc.request_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejects_a_missing_pop(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7(), include_pop=False)
        nc = await self._drive(lambda: jwks, req)
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == "TOOL_POP_UNVERIFIED"

    @pytest.mark.asyncio
    async def test_rejects_a_pop_for_a_different_body(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a proof minted for different arguments cannot be spliced onto this call.
        priv, jwks = hub
        req = _pop_request(
            priv, Ed25519PrivateKey.generate(), correlation_id=uuid7(), pop_body_args={"expression": "9+9"}
        )
        nc = await self._drive(lambda: jwks, req)
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == "TOOL_POP_UNVERIFIED"

    @pytest.mark.asyncio
    async def test_rejects_a_token_without_cnf(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a pop-enforced fleet requires holder-bound tokens; an unbound token cannot satisfy pop.
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7(), bind_cnf=False)
        nc = await self._drive(lambda: jwks, req)
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == "TOOL_POP_UNVERIFIED"

    @pytest.mark.asyncio
    async def test_rejects_a_replayed_nonce(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7())
        guard = FakeReplayGuard(fresh=False)  # the nonce was already consumed
        nc = await self._drive(lambda: jwks, req, pop_replay_guard=guard)
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == "TOOL_POP_UNVERIFIED"
        assert len(guard.seen) == 1  # the proxy DID consult the guard before rejecting

    @pytest.mark.asyncio
    async def test_forwards_when_the_nonce_is_fresh(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7())
        guard = FakeReplayGuard(fresh=True)
        nc = await self._drive(lambda: jwks, req, pop_replay_guard=guard)
        nc.request_raw.assert_called_once()
        assert len(guard.seen) == 1

    @pytest.mark.asyncio
    async def test_the_guard_receives_the_proofs_signed_issue_time(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # the guard refuses a proof issued before a wipe of its bucket only if it is handed the
        # proof's own signed iat -- not the proxy's clock, which would make every replay look new.
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7())
        guard = FakeReplayGuard(fresh=True)
        await self._drive(lambda: jwks, req, pop_replay_guard=guard)
        assert req.pop is not None
        signed_iat = jwt.decode(req.pop, options={"verify_signature": False})["iat"]
        assert guard.issued_at == [datetime.fromtimestamp(signed_iat, UTC)]


class _RaisingAuthorizer:
    """an authorizer whose store cannot be read, as the rbac authorizer's is when the broker refuses."""

    async def is_authorized(
        self,
        agent_id: str,
        user_id: str | None,
        tool_name: str,
        tool_version: str,
        *,
        principal_is_tool_pod: bool,
    ) -> bool:
        """raise the data-layer failure a proxy-backed collection raises.

        :param agent_id: the verified principal
        :ptype agent_id: str
        :param user_id: the verified user
        :ptype user_id: str | None
        :param tool_name: the tool
        :ptype tool_name: str
        :param tool_version: its version
        :ptype tool_version: str
        :param principal_is_tool_pod: whether the principal is a tool pod
        :ptype principal_is_tool_pod: bool
        :return: never returns
        :rtype: bool
        :raises DataLayerUnavailableError: always
        """
        del agent_id, user_id, tool_name, tool_version, principal_is_tool_pod
        raise DataLayerUnavailableError("L3 query failed: NAMESPACE_ACCESS_DENIED: carve-out read refused")


class TestAHostFailureIsAnsweredNotTimedOut:
    """a gate whose store fails answers the caller with a typed code -- never silence.

    Left uncaught, each of these killed the dispatch task with the reply subject unanswered, so the
    caller waited out its own deadline and read an outage in the registry's store as a dead tool.
    """

    @staticmethod
    async def _drive(
        req: ProxyCallRequest,
        jwks: dict[str, Any],
        *,
        authorizer: Any = None,
        guard: Any = None,
        pod_reply: bytes | None = None,
    ) -> AsyncMock:
        """dispatch one call through a proxy wired with the given seams.

        :param req: the call
        :ptype req: ProxyCallRequest
        :param jwks: the hub key set
        :ptype jwks: dict[str, Any]
        :param authorizer: the authorizer, allow-all when ``None``
        :ptype authorizer: Any
        :param guard: the replay guard, always-fresh when ``None``
        :ptype guard: Any
        :param pod_reply: what the pod answers, a successful reply when ``None``
        :ptype pod_reply: bytes | None
        :return: the NATS double
        :rtype: AsyncMock
        """
        proxy = CallProxy(
            await _catalog(),
            authorizer if authorizer is not None else AllowAllAuthorizer(),
            guard if guard is not None else FakeReplayGuard(fresh=True),
            limit_guard=AllowAllLimitGuard(),
            namespace="test",
            jwks_provider=lambda: jwks,
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=pod_reply if pod_reply is not None else _tool_reply())
        await proxy.start(nc)
        await proxy.handle_call(
            IncomingMessage(data=req.model_dump_json().encode("utf-8"), reply_subject="r", subject="test.tools.call")
        )
        for _ in range(5):
            await asyncio.sleep(0)
        return nc

    @staticmethod
    def _reply(nc: AsyncMock) -> ProxyCallResponse:
        """the one reply the proxy published.

        :param nc: the NATS double
        :ptype nc: AsyncMock
        :return: the reply
        :rtype: ProxyCallResponse
        """
        nc.publish_reply.assert_awaited_once()
        message: ProxyCallResponse = nc.publish_reply.call_args.kwargs["message"]
        return message

    @pytest.mark.asyncio
    async def test_an_authorizer_that_cannot_read_its_store_refuses_with_a_reply(
        self, hub: tuple[Any, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ) -> None:
        """fail closed, and say so: the call is not forwarded and the caller learns why.

        :param hub: the hub key and key set
        :ptype hub: tuple[Any, dict[str, Any]]
        :param caplog: the log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        priv, jwks = hub
        req = _authed_request(
            priv, token_sub=uuid7(), token_customer=uuid7(), token_user=uuid7(), envelope_agent=uuid7()
        )
        with caplog.at_level(logging.ERROR, logger="threetears.registry.proxy"):
            nc = await self._drive(req, jwks, authorizer=_RaisingAuthorizer())

        nc.request_raw.assert_not_called()
        reply = self._reply(nc)
        assert reply.success is False
        assert reply.error_code == "TOOL_AUTHORIZATION_UNAVAILABLE"
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "NAMESPACE_ACCESS_DENIED" in json.dumps(errors[0].__dict__.get("extra_data", {}))

    @pytest.mark.asyncio
    async def test_an_unreachable_replay_ledger_refuses_with_a_reply(self, hub: tuple[Any, dict[str, Any]]) -> None:
        """the ledger failing is a failed check (fail closed), but not an unverified proof.

        :param hub: the hub key and key set
        :ptype hub: tuple[Any, dict[str, Any]]
        :return: nothing
        :rtype: None
        """
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7())
        guard = FakeReplayGuard(record_error=KvError("nonce bucket unreachable"))

        nc = await self._drive(req, jwks, guard=guard)

        nc.request_raw.assert_not_called()
        assert len(guard.seen) == 1
        reply = self._reply(nc)
        assert reply.error_code == TOOL_POP_LEDGER_UNAVAILABLE == "TOOL_POP_LEDGER_UNAVAILABLE"
        # one client message, the same one the tool pod answers its own ledger outage with; the
        # exception type and text stay in the registry's ERROR log.
        assert reply.error == TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE

    @pytest.mark.asyncio
    async def test_a_pod_answer_that_does_not_parse_is_answered(self, hub: tuple[Any, dict[str, Any]]) -> None:
        """the pod may have run the tool, so this is not retried -- but it is answered.

        :param hub: the hub key and key set
        :ptype hub: tuple[Any, dict[str, Any]]
        :return: nothing
        :rtype: None
        """
        priv, jwks = hub
        req = _authed_request(
            priv, token_sub=uuid7(), token_customer=uuid7(), token_user=uuid7(), envelope_agent=uuid7()
        )

        nc = await self._drive(req, jwks, pod_reply=b"{not json")

        nc.request_raw.assert_called_once()
        assert self._reply(nc).error_code == "TOOL_RESPONSE_MALFORMED"


class TestOneAnswerForEveryUnverifiedIdentity:
    """a forwarded identity that does not verify answers what every hub door answers.

    The proxy used to answer ``TOOL_IDENTITY_UNVERIFIED`` for the handshake token and
    ``TOOL_USER_IDENTITY_UNVERIFIED`` for the user assertion, with a message naming the exception
    type that failed, while every hub door answered the same condition ``IDENTITY_REFUSED`` with
    one undiscriminating message. One condition answers one code: whichever check refused, the
    caller reads the same code and the same message, and the reason stays in the registry's log.
    """

    async def _reply_for(self, jwks: dict[str, Any], req: ProxyCallRequest) -> tuple[AsyncMock, ProxyCallResponse]:
        proxy = CallProxy(
            await _catalog(),
            AllowAllAuthorizer(),
            FakeReplayGuard(fresh=True),
            limit_guard=AllowAllLimitGuard(),
            namespace="test",
            jwks_provider=lambda: jwks,
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_tool_reply())
        await proxy.start(nc)
        await proxy.handle_call(
            IncomingMessage(
                data=req.model_dump_json().encode("utf-8"),
                reply_subject="reply.subject",
                subject="test.tools.call",
            )
        )
        await asyncio.sleep(0)
        reply: ProxyCallResponse = nc.publish_reply.call_args.kwargs["message"]
        return nc, reply

    def test_the_code_is_the_platform_wide_spelling(self) -> None:
        """the wire value every hub door answers; the hub imports this constant rather than re-spelling it."""
        assert IDENTITY_REFUSED == "IDENTITY_REFUSED"
        assert IDENTITY_REFUSED_MESSAGE == "forwarded identity could not be verified"

    @pytest.mark.asyncio
    async def test_every_failed_identity_check_answers_the_same_code_and_message(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        """an absent token, an expired token, an unbound assertion and an assertion on a tool pod.

        Four different checks, one answer. A caller that could tell them apart would learn which
        check its credential failed, which is exactly what the message must not disclose.
        """
        priv, jwks = hub
        agent, cust, conv, pod_id = uuid7(), uuid7(), uuid7(), uuid7()
        requests = [
            _request(agent_id=uuid7(), token=None),
            _authed_request(
                priv, token_sub=agent, token_customer=cust, token_user=None, envelope_agent=agent, exp_delta=-600
            ),
            _authed_request(
                priv,
                token_sub=agent,
                token_customer=cust,
                token_user=None,
                envelope_agent=agent,
                user_assertion=_user_assertion(priv, sub=uuid7(), customer_id=cust, user_id=uuid7()),
            ),
            _authed_request(
                priv,
                token_sub=pod_id,
                token_customer=PLATFORM_CUSTOMER_SENTINEL,
                token_user=None,
                envelope_agent=pod_id,
                conversation_id=conv,
                user_assertion=_user_assertion(
                    priv, sub=pod_id, customer_id=PLATFORM_CUSTOMER_SENTINEL, user_id=uuid7(), conversation_id=conv
                ),
            ),
        ]
        answers = []
        for req in requests:
            nc, reply = await self._reply_for(jwks, req)
            nc.request_raw.assert_not_called()
            assert reply.success is False
            answers.append((reply.error_code, reply.error))
        assert answers == [(IDENTITY_REFUSED, IDENTITY_REFUSED_MESSAGE)] * len(requests)

    @pytest.mark.asyncio
    async def test_the_reason_stays_in_the_registry_log(
        self, hub: tuple[Any, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ) -> None:
        """the caller learns nothing about which check refused; the operator reading the log does."""
        priv, jwks = hub
        agent, cust = uuid7(), uuid7()
        req = _authed_request(
            priv,
            token_sub=agent,
            token_customer=cust,
            token_user=None,
            envelope_agent=agent,
            user_assertion=_user_assertion(priv, sub=uuid7(), customer_id=cust, user_id=uuid7()),
        )
        with caplog.at_level(logging.WARNING, logger="threetears.registry.proxy"):
            _nc, reply = await self._reply_for(jwks, req)
        assert "sub/customer mismatch" not in (reply.error or "")
        details = [getattr(record, "extra_data", {}).get("detail", "") for record in caplog.records]
        assert any("sub/customer mismatch" in detail for detail in details)
