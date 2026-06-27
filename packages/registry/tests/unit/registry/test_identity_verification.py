"""v0.13.9 auth C3: the proxy verifies the Hub identity token + re-stamps verified identity.

The contract this pins (exercised end-to-end through the public dispatch surface):

- the enforcement flag ladders off -> warn -> enforce (default off; a typo'd value fails loud);
- ``off`` is fully inert: the token is ignored and the self-asserted envelope is used as before;
- on a VALID token the verified ``agent_id``/``user_id``/``customer_id`` OVERWRITE whatever the
  envelope claimed -- so a lying envelope cannot impersonate another agent or inject a user_id;
  the verified identity is what reaches BOTH the authorizer and the tool pod;
- ``warn`` fails OPEN (verify, log on failure, still forward) so an incomplete fleet isn't broken;
- ``enforce`` fails CLOSED (reject with TOOL_IDENTITY_UNVERIFIED, never forwarding).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid7

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from threetears.agent.tools.context_envelope import CallContext

from threetears.core.security.identity_token import (
    IdentityClaims,
    build_jwks,
    canonical_call_hash,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
)
from threetears.core.security.pop import access_token_hash, make_pop_proof
from threetears.nats import IncomingMessage, set_default_namespace
from threetears.registry.auth import AllowAllAuthorizer
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.config import (
    IdentityEnforcement,
    get_identity_enforcement,
    get_pop_enforcement,
)
from threetears.registry.proxy import CallProxy, ProxyCallRequest, ProxyCallResponse

_ENV = "THREETEARS_REGISTRY_IDENTITY_ENFORCEMENT"
_ISS = "hub"
_KID = "kid-1"
_TOOL = "threetears.calculator"


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
    customer_id: UUID,
    user_id: UUID | None,
    exp_delta: int = 600,
    iss: str = _ISS,
    cnf: str | None = None,
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
    )
    return sign_identity_token(claims, signing_key=priv, kid=_KID)


def _request(
    *,
    agent_id: UUID,
    user_id: UUID | None = None,
    customer_id: UUID | None = None,
    token: str | None = None,
) -> ProxyCallRequest:
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


def _entry() -> CatalogEntry:
    return CatalogEntry(
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


class TestIdentityEnforcementConfig:
    """``get_identity_enforcement`` defaults off, parses the ladder, and fails loud on a typo."""

    def test_default_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_ENV, raising=False)
        assert get_identity_enforcement() is IdentityEnforcement.OFF

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("off", IdentityEnforcement.OFF),
            ("warn", IdentityEnforcement.WARN),
            ("enforce", IdentityEnforcement.ENFORCE),
            ("ENFORCE", IdentityEnforcement.ENFORCE),
            ("  Warn  ", IdentityEnforcement.WARN),
        ],
    )
    def test_valid_values(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: IdentityEnforcement
    ) -> None:
        monkeypatch.setenv(_ENV, raw)
        assert get_identity_enforcement() is expected

    def test_invalid_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV, "audit")
        with pytest.raises(ValueError):
            get_identity_enforcement()


class TestDispatchIdentityEnforcement:
    """end-to-end through ``handle_call``: rejection without forwarding, and verified-identity flow."""

    async def _drive(
        self,
        mode: IdentityEnforcement,
        jwks_provider: Any,
        req: ProxyCallRequest,
        *,
        authorizer: Any = None,
    ) -> AsyncMock:
        proxy = CallProxy(
            await _catalog(),
            authorizer if authorizer is not None else AllowAllAuthorizer(),
            namespace="test",
            identity_enforcement=mode,
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

    # -- off: fully inert --

    @pytest.mark.asyncio
    async def test_off_forwards_claimed_identity_unverified(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        _priv, jwks = hub
        claimed = uuid7()
        nc = await self._drive(IdentityEnforcement.OFF, lambda: jwks, _request(agent_id=claimed, token=None))
        nc.request_raw.assert_called_once()
        assert self._forwarded_context(nc)["agent_id"] == str(claimed)

    # -- enforce: fail closed, verified identity flows on success --

    @pytest.mark.asyncio
    async def test_enforce_rejects_unverified_without_forwarding(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        _priv, jwks = hub
        nc = await self._drive(IdentityEnforcement.ENFORCE, lambda: jwks, _request(agent_id=uuid7(), token=None))
        nc.request_raw.assert_not_called()
        reply = self._reply(nc)
        assert reply.success is False
        assert reply.error_code == "TOOL_IDENTITY_UNVERIFIED"

    @pytest.mark.asyncio
    async def test_enforce_forwards_verified_identity_to_pod_over_a_lie(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        priv, jwks = hub
        real_agent, real_cust = uuid7(), uuid7()
        token = _token(priv, sub=real_agent, customer_id=real_cust, user_id=None)
        # the envelope claims a DIFFERENT agent than the token; the pod must get the verified one
        nc = await self._drive(IdentityEnforcement.ENFORCE, lambda: jwks, _request(agent_id=uuid7(), token=token))
        nc.request_raw.assert_called_once()
        forwarded = self._forwarded_context(nc)
        assert forwarded["agent_id"] == str(real_agent)
        assert forwarded["customer_id"] == str(real_cust)

    @pytest.mark.asyncio
    async def test_enforce_verified_user_id_none_overrides_claimed_user(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        # privilege guard: a token with NO user_id blanks the envelope's claimed user_id.
        priv, jwks = hub
        token = _token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None)
        nc = await self._drive(
            IdentityEnforcement.ENFORCE, lambda: jwks, _request(agent_id=uuid7(), user_id=uuid7(), token=token)
        )
        nc.request_raw.assert_called_once()
        assert self._forwarded_context(nc)["user_id"] is None

    @pytest.mark.asyncio
    async def test_enforce_restamped_identity_reaches_authorizer(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        priv, jwks = hub
        real_agent, real_user = uuid7(), uuid7()
        captured: dict[str, str | None] = {}

        class _RecordingAuthorizer:
            async def is_authorized(
                self, agent_id: str, user_id: str | None, tool_name: str, tool_version: str
            ) -> bool:
                captured["agent_id"] = agent_id
                captured["user_id"] = user_id
                return True

        # envelope claims a different agent AND a different user than the token
        token = _token(priv, sub=real_agent, customer_id=uuid7(), user_id=real_user)
        await self._drive(
            IdentityEnforcement.ENFORCE,
            lambda: jwks,
            _request(agent_id=uuid7(), user_id=uuid7(), token=token),
            authorizer=_RecordingAuthorizer(),
        )
        assert captured["agent_id"] == str(real_agent)  # authorizer saw the VERIFIED agent
        assert captured["user_id"] == str(real_user)  # ...and the VERIFIED user

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flavor", ["invalid", "absent", "no_provider", "expired", "wrong_issuer", "provider_raises"]
    )
    async def test_enforce_rejects_each_failure_mode(
        self, hub: tuple[Any, dict[str, Any]], flavor: str
    ) -> None:
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
            req = _request(agent_id=uuid7(), token=_token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None, exp_delta=-120))
        elif flavor == "wrong_issuer":
            req = _request(agent_id=uuid7(), token=_token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None, iss="evil"))
        else:  # provider_raises -- a flaky/Hub-down JWKS fetch must reject cleanly, never hang
            provider = _raising_provider
            req = _request(agent_id=uuid7(), token=_token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None))
        nc = await self._drive(IdentityEnforcement.ENFORCE, provider, req)
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == "TOOL_IDENTITY_UNVERIFIED"

    @pytest.mark.asyncio
    async def test_enforce_forwards_within_leeway(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # a token expired by less than the proxy's clock-skew leeway is still accepted + forwarded.
        priv, jwks = hub
        real_agent = uuid7()
        token = _token(priv, sub=real_agent, customer_id=uuid7(), user_id=None, exp_delta=-30)
        nc = await self._drive(IdentityEnforcement.ENFORCE, lambda: jwks, _request(agent_id=uuid7(), token=token))
        nc.request_raw.assert_called_once()
        assert self._forwarded_context(nc)["agent_id"] == str(real_agent)

    @pytest.mark.asyncio
    async def test_warn_provider_failure_fails_open_not_hang(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        # a throwing provider in warn mode must allow the call (fail-open), never hang it.
        priv, _jwks = hub
        token = _token(priv, sub=uuid7(), customer_id=uuid7(), user_id=None)
        nc = await self._drive(IdentityEnforcement.WARN, _raising_provider, _request(agent_id=uuid7(), token=token))
        nc.request_raw.assert_called_once()

    # -- warn: fail open --

    @pytest.mark.asyncio
    async def test_warn_forwards_verified_identity_on_valid_token(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        priv, jwks = hub
        real_agent = uuid7()
        token = _token(priv, sub=real_agent, customer_id=uuid7(), user_id=None)
        nc = await self._drive(IdentityEnforcement.WARN, lambda: jwks, _request(agent_id=uuid7(), token=token))
        nc.request_raw.assert_called_once()
        assert self._forwarded_context(nc)["agent_id"] == str(real_agent)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("token", ["bad.token", None])
    async def test_warn_forwards_despite_verification_failure(
        self, hub: tuple[Any, dict[str, Any]], token: str | None
    ) -> None:
        _priv, jwks = hub
        claimed = uuid7()
        nc = await self._drive(IdentityEnforcement.WARN, lambda: jwks, _request(agent_id=claimed, token=token))
        nc.request_raw.assert_called_once()  # fail-open: still forwarded
        # claimed identity is kept (unverified) since verification failed
        assert self._forwarded_context(nc)["agent_id"] == str(claimed)


# ---------------------------------------------------------------------------
# v0.13.9 auth W4c: per-call proof-of-possession (the agent->proxy holder binding)
# ---------------------------------------------------------------------------

_POP_ENV = "THREETEARS_REGISTRY_POP_ENFORCEMENT"


class _StubReplayGuard:
    """returns a fixed freshness verdict so the proxy's replay wiring can be tested without a live
    NATS-KV (the real guard's compare-and-set is covered by its own coordination tests)."""

    def __init__(self, *, fresh: bool) -> None:
        self._fresh = fresh
        self.seen: list[str] = []

    async def record_unique(self, nonce: str) -> bool:
        self.seen.append(nonce)
        return self._fresh


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


class TestPopEnforcementConfig:
    """``get_pop_enforcement`` defaults off, parses the ladder, and fails loud on a typo."""

    def test_default_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_POP_ENV, raising=False)
        assert get_pop_enforcement() is IdentityEnforcement.OFF

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("warn", IdentityEnforcement.WARN), ("enforce", IdentityEnforcement.ENFORCE)],
    )
    def test_valid_values(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: IdentityEnforcement
    ) -> None:
        monkeypatch.setenv(_POP_ENV, raw)
        assert get_pop_enforcement() is expected

    def test_invalid_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_POP_ENV, "audit")
        with pytest.raises(ValueError):
            get_pop_enforcement()


class TestDispatchPopEnforcement:
    """end-to-end through ``handle_call``: under enforce the proxy requires a valid per-call pop.

    pop verification is self-contained (it re-verifies the token for a trusted cnf), so these run
    with identity enforcement at its default OFF -- proving the two gates ladder independently.
    """

    async def _drive(
        self,
        mode: IdentityEnforcement,
        jwks_provider: Any,
        req: ProxyCallRequest,
        *,
        pop_replay_guard: Any = None,
    ) -> AsyncMock:
        proxy = CallProxy(
            await _catalog(),
            AllowAllAuthorizer(),
            namespace="test",
            jwks_provider=jwks_provider,
            pop_enforcement=mode,
            pop_replay_guard=pop_replay_guard,
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
    async def test_enforce_forwards_a_valid_pop(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7())
        nc = await self._drive(IdentityEnforcement.ENFORCE, lambda: jwks, req)
        nc.request_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_enforce_rejects_a_missing_pop(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7(), include_pop=False)
        nc = await self._drive(IdentityEnforcement.ENFORCE, lambda: jwks, req)
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == "TOOL_POP_UNVERIFIED"

    @pytest.mark.asyncio
    async def test_enforce_rejects_a_pop_for_a_different_body(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        # a proof minted for different arguments cannot be spliced onto this call.
        priv, jwks = hub
        req = _pop_request(
            priv, Ed25519PrivateKey.generate(), correlation_id=uuid7(), pop_body_args={"expression": "9+9"}
        )
        nc = await self._drive(IdentityEnforcement.ENFORCE, lambda: jwks, req)
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == "TOOL_POP_UNVERIFIED"

    @pytest.mark.asyncio
    async def test_enforce_rejects_a_token_without_cnf(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        # a pop-enforced fleet requires holder-bound tokens; an unbound token cannot satisfy pop.
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7(), bind_cnf=False)
        nc = await self._drive(IdentityEnforcement.ENFORCE, lambda: jwks, req)
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == "TOOL_POP_UNVERIFIED"

    @pytest.mark.asyncio
    async def test_enforce_rejects_a_replayed_nonce(self, hub: tuple[Any, dict[str, Any]]) -> None:
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7())
        guard = _StubReplayGuard(fresh=False)  # the nonce was already consumed
        nc = await self._drive(
            IdentityEnforcement.ENFORCE, lambda: jwks, req, pop_replay_guard=guard
        )
        nc.request_raw.assert_not_called()
        assert self._reply(nc).error_code == "TOOL_POP_UNVERIFIED"
        assert len(guard.seen) == 1  # the proxy DID consult the guard before rejecting

    @pytest.mark.asyncio
    async def test_enforce_forwards_when_the_nonce_is_fresh(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7())
        guard = _StubReplayGuard(fresh=True)
        nc = await self._drive(
            IdentityEnforcement.ENFORCE, lambda: jwks, req, pop_replay_guard=guard
        )
        nc.request_raw.assert_called_once()
        assert len(guard.seen) == 1

    @pytest.mark.asyncio
    async def test_off_forwards_without_a_pop(self, hub: tuple[Any, dict[str, Any]]) -> None:
        # pop OFF: a request with no pop is forwarded unchanged (the gate is inert).
        _priv, jwks = hub
        nc = await self._drive(IdentityEnforcement.OFF, lambda: jwks, _request(agent_id=uuid7(), token=None))
        nc.request_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_warn_fails_open_on_a_missing_pop(
        self, hub: tuple[Any, dict[str, Any]]
    ) -> None:
        priv, jwks = hub
        req = _pop_request(priv, Ed25519PrivateKey.generate(), correlation_id=uuid7(), include_pop=False)
        nc = await self._drive(IdentityEnforcement.WARN, lambda: jwks, req)
        nc.request_raw.assert_called_once()  # fail-open: still forwarded
