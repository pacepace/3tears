"""shared enforce-only auth scaffolding for the pod-side (ToolServer) dispatch tests.

v0.13.9 made :class:`~threetears.agent.tools.server.ToolServer` re-verify the Hub identity token AND
the registry proxy's body-bound assertion on EVERY inbound call, UNCONDITIONALLY and fail-closed,
re-stamping the verified identity before dispatch. so every message a dispatch test drives must
carry a valid identity token + a valid proxy assertion bound to THIS pod + body, and the server must
be wired with a JWKS provider that verifies both. these helpers centralise that wiring.

a USER-driven turn additionally carries a Hub-minted, cnf-LESS user-assertion at
``context.user_identity_token`` (the handshake token is one-per-pod and user-LESS); the pod verifies
it, binds it to the handshake token by ``sub`` + ``customer_id``, and re-stamps the per-turn user_id
from it. :func:`mint_user_assertion` builds one (bound or deliberately mis-bound for the deny tests).

not a ``test_*`` module, so pytest does not collect it; the tools test package
(``tests/unit/tools/__init__.py``) makes it importable as a sibling.
"""

from __future__ import annotations

import base64
import time
from typing import Any
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr

from threetears.core.security import ProxyAssertionSigner, canonical_call_hash
from threetears.core.security.identity_token import (
    IdentityClaims,
    build_jwks,
    generate_signing_keypair,
    sign_identity_token,
)

__all__ = ["StubReplayGuard", "jwks_provider", "mint_user_assertion", "signed_call_payload"]


class StubReplayGuard:
    """records each proxy-assertion nonce + returns a fixed freshness verdict so the pod's MANDATORY
    replay-guard wiring runs without a live NATS-KV (the real guard's compare-and-set is covered by
    its own coordination tests). default ``fresh=True`` -> every first-seen assertion is accepted."""

    def __init__(self, *, fresh: bool = True) -> None:
        self._fresh = fresh
        self.seen: list[str] = []

    async def record_unique(self, nonce: str) -> bool:
        self.seen.append(nonce)
        return self._fresh


# one Hub identity keypair + one proxy-assertion signer, merged into a single JWKS under distinct
# kids: the pod's single ``jwks_provider`` must verify BOTH the identity token (Hub key) and the
# proxy assertion (proxy key).
_HUB_PRIV, _HUB_PUB = generate_signing_keypair()
_SEED = base64.urlsafe_b64encode(Ed25519PrivateKey.generate().private_bytes_raw()).decode("ascii")
_SIGNER = ProxyAssertionSigner.from_secret(SecretStr(_SEED))
_JWKS: dict[str, Any] = {"keys": [*build_jwks({"kid-1": _HUB_PUB})["keys"], *_SIGNER.public_jwks()["keys"]]}


def jwks_provider() -> dict[str, Any]:
    """the combined Hub-identity + proxy-assertion JWKS the pod verifies both gates against."""
    return _JWKS


def _hub_token(*, sub: UUID, customer_id: UUID, user_id: UUID | None, exp_delta: int = 600) -> str:
    """sign a Hub-style, cnf-LESS identity token (shared by the handshake token + the user-assertion).

    both are Hub-signed and verify against the same ``jwks_provider``/issuer; the user-assertion is
    just a second such token carrying the per-turn ``user_id`` and bound to the handshake token by
    ``sub`` + ``customer_id`` rather than by pop.
    """
    now = int(time.time())
    claims = IdentityClaims(
        sub=str(sub),
        customer_id=str(customer_id),
        user_id=str(user_id) if user_id is not None else None,
        sid="sid-1",
        pod_id="pod-1",
        iss="hub",
        iat=now,
        exp=now + exp_delta,
    )
    return sign_identity_token(claims, signing_key=_HUB_PRIV, kid="kid-1")


def mint_user_assertion(*, sub: UUID, customer_id: UUID, user_id: UUID | None, exp_delta: int = 3600) -> str:
    """mint a Hub-style, cnf-LESS user-assertion (the per-turn verified ``user_id`` token).

    signed by the same Hub key the handshake token uses, so the pod verifies it against the SAME
    JWKS/issuer. carries NO ``cnf`` (the Hub cannot know the target pod's holder key at mint) -- the
    pod binds it to the handshake token by ``sub`` + ``customer_id``. callers drive the pod's
    user-assertion deny paths by passing a mismatched ``sub``/``customer_id``, a ``None`` ``user_id``,
    or a negative ``exp_delta`` (expired).
    """
    return _hub_token(sub=sub, customer_id=customer_id, user_id=user_id, exp_delta=exp_delta)


def signed_call_payload(
    *,
    pod_id: str,
    tool_name: str = "test.stub",
    tool_version: str = "1.0",
    arguments: dict[str, Any] | None = None,
    correlation_id: str | None = None,
    agent_id: UUID | None = None,
    customer_id: UUID | None = None,
    conversation_id: UUID | None = None,
    user_id: UUID | None = None,
    user_assertion: str | None = None,
) -> dict[str, Any]:
    """build a fully-authenticated CallRequest payload dict the enforce-only pod will dispatch.

    PRODUCTION SHAPE: the handshake identity token (``sub`` = ``agent_id``, ``customer_id``) is
    user-LESS -- production NEVER puts the per-turn user on the handshake token, because that token
    is one-per-pod and the user varies each inbound message. the per-turn user instead rides as a
    SEPARATE Hub-minted, cnf-LESS user-assertion attached at ``context.user_identity_token``, which
    the pod verifies + binds to the handshake token + re-stamps ``user_id`` from (mirroring the
    proxy).

    ``user_id`` (when set) mints a BOUND user-assertion (``sub`` = the handshake ``agent_id``,
    ``customer_id`` = the handshake ``customer_id``, that ``user_id``) so the pod re-stamps the
    dispatch user from it. callers exercising the user-assertion DENY paths pass an explicit
    ``user_assertion=`` (a mis-bound :func:`mint_user_assertion`, an expired one, or a raw
    non-JWS string) which overrides and is attached verbatim. ``conversation_id`` rides on the
    envelope unchanged (the pod re-stamps only agent/user/customer) so the per-user context manager
    gate can be exercised.

    the server under test MUST be constructed with ``pod_id=<pod_id>`` and ``jwks_provider`` (so the
    proxy assertion's ``aud`` matches and both gates verify). the handshake agent/customer are NOT
    put on the envelope context; the pod overwrites them from the verified token regardless.
    """
    args = arguments if arguments is not None else {}
    corr = correlation_id if correlation_id is not None else str(uuid4())
    effective_agent_id = agent_id if agent_id is not None else uuid4()
    effective_customer_id = customer_id if customer_id is not None else uuid4()
    token = _hub_token(sub=effective_agent_id, customer_id=effective_customer_id, user_id=None)
    # a bound user-assertion is minted when a user_id is asked for; an explicit ``user_assertion``
    # (mis-bound / expired / invalid) overrides for the deny-path tests.
    if user_assertion is None and user_id is not None:
        user_assertion = mint_user_assertion(sub=effective_agent_id, customer_id=effective_customer_id, user_id=user_id)
    body_hash = canonical_call_hash(tool_name, args, corr)
    proxy_assertion = _SIGNER.mint(
        pod_id=pod_id,
        agent_id=str(uuid4()),
        customer_id=str(uuid4()),
        body_hash=body_hash,
        nonce=str(uuid4()),
        now=int(time.time()),
    )
    context: dict[str, Any] = {"correlation_id": corr, "identity_token": token}
    if conversation_id is not None:
        context["conversation_id"] = str(conversation_id)
    if user_assertion is not None:
        context["user_identity_token"] = user_assertion
    return {
        "tool_name": tool_name,
        "tool_version": tool_version,
        "arguments": args,
        "context": context,
        "proxy_assertion": proxy_assertion,
    }
