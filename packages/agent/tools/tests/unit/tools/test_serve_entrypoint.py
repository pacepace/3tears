"""tests for the built-in tool-pod entrypoint's per-key identity connect path.

covers ``_BuiltinToolBootstrap.build_server`` (v0.14.1 per-key ONLY):

* per-key identity: with an identity signing key + pod id + issuer, the pod self-mints a verifiable
  identity JWT and hands ``ToolServer`` an ``auth_token`` provider (no static creds).
* fail loud: with NO signing key the pod crashes startup rather than connect unauthenticated -- the
  static-credential fallback was deleted in the per-key cutover.

plus the fail-loud guards on a partial identity config.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from threetears.agent.tools.serve import _BuiltinToolBootstrap
from threetears.core.security import IdentityMinter, verify_identity_token

_ISSUER = "aibots-tool-pod"
_POD_ID = "11111111-1111-1111-1111-111111111111"


def _signing_key_pem() -> str:
    """a fresh PKCS#8 PEM Ed25519 private key, as the operator injects via env."""
    key = Ed25519PrivateKey.generate()
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


@pytest.mark.asyncio
async def test_identity_path_hands_toolserver_a_verifiable_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """with the signing key + pod id + issuer set, ToolServer gets an auth_token that mints a
    JWT verifiable against the pod's public JWKS (issuer + kid = pod id), and NO static creds."""
    pem = _signing_key_pem()
    monkeypatch.setenv("THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY", pem)
    monkeypatch.setenv("THREETEARS_TOOL_POD_ID", _POD_ID)
    monkeypatch.setenv("THREETEARS_TOOL_POD_CONNECT_ISSUER", _ISSUER)
    monkeypatch.delenv("THREETEARS_TOOL_POD_CUSTOMER_ID", raising=False)

    with patch("threetears.agent.tools.serve.ToolServer") as tool_server_cls:
        tool_server_cls.return_value = MagicMock()
        await _BuiltinToolBootstrap("builtin").build_server()

    kwargs = tool_server_cls.call_args.kwargs
    assert kwargs["pod_id"] == _POD_ID
    assert kwargs.get("nats_user") is None
    assert kwargs.get("nats_password") is None
    provider = kwargs["auth_token"]
    assert callable(provider)

    # the provider mints a JWT verifiable against the pod's own public key (kid = pod id, issuer
    # pinned) -- exactly what the Hub auth-callout + registry verifiers do.
    verifying_jwks = IdentityMinter.from_pem(pem, kid=_POD_ID, issuer=_ISSUER).jwks()
    claims = verify_identity_token(provider(), jwks=verifying_jwks, issuer=_ISSUER)
    assert claims.sub == _POD_ID
    assert claims.pod_id == _POD_ID


@pytest.mark.asyncio
async def test_missing_signing_key_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """no signing key -> crash startup (the static-credential fallback was deleted, v0.14.1)."""
    monkeypatch.delenv("THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY", raising=False)
    # even with a static user/password present, there is NO fallback -- per-key is mandatory.
    monkeypatch.setenv("THREETEARS_NATS_USER", "tooling")
    monkeypatch.setenv("THREETEARS_NATS_PASSWORD", "secret")

    with pytest.raises(ValueError, match="THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY is required"):
        await _BuiltinToolBootstrap("builtin").build_server()


@pytest.mark.asyncio
async def test_identity_key_without_pod_id_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """a signing key with no pod id is a partial config -> crash (never silent fallback)."""
    monkeypatch.setenv("THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY", _signing_key_pem())
    monkeypatch.delenv("THREETEARS_TOOL_POD_ID", raising=False)
    monkeypatch.setenv("THREETEARS_TOOL_POD_CONNECT_ISSUER", _ISSUER)

    with pytest.raises(ValueError, match="THREETEARS_TOOL_POD_ID"):
        await _BuiltinToolBootstrap("builtin").build_server()


@pytest.mark.asyncio
async def test_identity_key_without_issuer_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """a signing key with no connect issuer is a partial config -> crash."""
    monkeypatch.setenv("THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY", _signing_key_pem())
    monkeypatch.setenv("THREETEARS_TOOL_POD_ID", _POD_ID)
    monkeypatch.delenv("THREETEARS_TOOL_POD_CONNECT_ISSUER", raising=False)

    with pytest.raises(ValueError, match="THREETEARS_TOOL_POD_CONNECT_ISSUER"):
        await _BuiltinToolBootstrap("builtin").build_server()
