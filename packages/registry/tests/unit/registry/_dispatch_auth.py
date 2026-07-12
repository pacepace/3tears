"""shared enforce-only auth scaffolding for the registry proxy dispatch tests.

v0.13.9 made :class:`~threetears.registry.proxy.CallProxy` verify the Hub identity token + the
per-call proof-of-possession UNCONDITIONALLY and fail-closed, then re-stamp the verified identity
before forwarding. so every request a dispatch test drives must carry a valid cnf-bound token + a
matching pop, and every proxy must be wired with the Hub JWKS + a pop replay guard. these helpers
centralise that wiring so the dispatch-mechanics tests stay focused on routing / timeout / catalog
behaviour rather than re-deriving the crypto in every module.

not a ``test_*`` module, so pytest does not collect it; the registry test package
(``tests/unit/registry/__init__.py``) makes it importable as a sibling.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from threetears.agent.tools.context_envelope import CallContext

from threetears.core.security import canonical_call_hash
from threetears.core.security.identity_token import (
    IdentityClaims,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
)
from threetears.core.security.pop import access_token_hash, make_pop_proof
from threetears.registry.auth import AllowAllAuthorizer, AllowAllLimitGuard, LimitGuard
from threetears.registry.catalog import ToolCatalog
from threetears.registry.proxy import CallProxy, ProxyCallRequest

__all__ = [
    "DEFAULT_AGENT_ID",
    "DEFAULT_CORRELATION_ID",
    "DEFAULT_CUSTOMER_ID",
    "HUB_JWKS",
    "StubReplayGuard",
    "hub_jwks_provider",
    "make_authed_request",
    "make_proxy",
]


# a single Hub signing keypair + the JWKS the proxy verifies against, and one ephemeral holder key
# whose thumbprint binds every token's ``cnf`` (so the per-call pop proves possession of it).
_HUB_PRIV, _HUB_PUB = generate_signing_keypair()
HUB_JWKS: dict[str, Any] = build_jwks({"kid-1": _HUB_PUB})
_HOLDER_KEY = Ed25519PrivateKey.generate()
_HOLDER_CNF = jwk_thumbprint(_HOLDER_KEY.public_key())

# stable identity defaults so re-stamp is observable + deterministic across the dispatch tests.
DEFAULT_AGENT_ID = UUID("01948a00-aaaa-7000-8000-000000a9e777")
DEFAULT_CUSTOMER_ID = UUID("01948a00-cccc-7000-8000-00000000c001")
DEFAULT_CORRELATION_ID = UUID("01948a00-0000-7000-8000-0000000abc12")


def hub_jwks_provider() -> dict[str, Any]:
    """the Hub JWKS the proxy verifies identity tokens (and re-derives the trusted cnf) against."""
    return HUB_JWKS


class StubReplayGuard:
    """records each pop nonce + returns a fixed freshness verdict so the proxy's replay wiring runs
    without a live NATS-KV (the real guard's compare-and-set is covered by its own coordination
    tests). default ``fresh=True`` -> every first-seen pop is accepted."""

    def __init__(self, *, fresh: bool = True) -> None:
        self._fresh = fresh
        self.seen: list[str] = []

    async def record_unique(self, nonce: str) -> bool:
        self.seen.append(nonce)
        return self._fresh


def make_proxy(
    catalog: ToolCatalog,
    authorizer: Any = None,
    *,
    pop_replay_guard: Any = None,
    limit_guard: "LimitGuard | None" = None,
    jwks_provider: Any = hub_jwks_provider,
    **kwargs: Any,
) -> CallProxy:
    """build a :class:`CallProxy` wired for enforce-only verification.

    injects the Hub-JWKS provider + a fresh pop replay guard so the unconditional identity + pop
    gates accept the authenticated requests :func:`make_authed_request` builds. ``authorizer``
    defaults to :class:`AllowAllAuthorizer` and ``limit_guard`` to :class:`AllowAllLimitGuard`
    (the spend gate is a required constructor arg, so the dispatch-mechanics tests wire an
    allow-all double unless a test overrides it); extra kwargs (namespace, timeout, proxy_signer,
    routing_strategy, usage_emitter) pass straight through.
    """
    return CallProxy(
        catalog,
        authorizer if authorizer is not None else AllowAllAuthorizer(),
        pop_replay_guard if pop_replay_guard is not None else StubReplayGuard(),
        limit_guard if limit_guard is not None else AllowAllLimitGuard(),
        jwks_provider=jwks_provider,
        **kwargs,
    )


def make_authed_request(
    *,
    agent_id: UUID | None = None,
    tool_name: str = "threetears.calculator",
    tool_version: str = "1.0.0",
    arguments: dict[str, Any] | None = None,
    correlation_id: UUID | None = None,
    customer_id: UUID | None = None,
) -> ProxyCallRequest:
    """create an AUTHENTICATED :class:`ProxyCallRequest`.

    carries a valid Hub identity token (``sub`` = ``agent_id``, ``customer_id``, ``cnf`` bound to
    the shared holder key) + a per-call pop bound to THIS body, so the enforce-only proxy verifies
    and re-stamps it rather than rejecting it. the re-stamp is identity-preserving here (the token's
    ``sub`` == the request's ``agent_id``) so routing / forwarding assertions still see the same
    agent; ``customer_id`` rides on the token so the re-stamped customer is observable too.
    """
    if arguments is None:
        arguments = {"expression": "2+2"}
    effective_correlation_id = correlation_id if correlation_id is not None else DEFAULT_CORRELATION_ID
    effective_agent_id = agent_id if agent_id is not None else DEFAULT_AGENT_ID
    effective_customer_id = customer_id if customer_id is not None else DEFAULT_CUSTOMER_ID
    now = int(time.time())
    token = sign_identity_token(
        IdentityClaims(
            sub=str(effective_agent_id),
            customer_id=str(effective_customer_id),
            user_id=None,
            sid="sid-1",
            pod_id="pod-1",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=_HOLDER_CNF,
        ),
        signing_key=_HUB_PRIV,
        kid="kid-1",
    )
    body_hash = canonical_call_hash(tool_name, arguments, str(effective_correlation_id))
    pop = make_pop_proof(
        holder_key=_HOLDER_KEY,
        access_token_hash=access_token_hash(token),
        body_hash=body_hash,
        nonce=str(uuid4()),
        iat=now,
    )
    return ProxyCallRequest(
        tool_name=tool_name,
        tool_version=tool_version,
        arguments=arguments,
        context=CallContext(
            correlation_id=effective_correlation_id,
            agent_id=effective_agent_id,
            identity_token=token,
        ),
        pop=pop,
    )
