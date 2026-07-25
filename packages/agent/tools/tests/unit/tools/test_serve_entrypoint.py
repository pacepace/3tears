"""tests for the built-in tool-pod entrypoint's per-key identity connect path.

covers ``_BuiltinToolBootstrap.build_server`` (v0.14.1 per-key ONLY):

* per-key identity: with an identity signing key + pod id + issuer, the pod self-mints a verifiable
  identity JWT and hands ``ToolServer`` an ``auth_token`` provider (no static creds).
* fail loud: with NO signing key the pod crashes startup rather than connect unauthenticated -- the
  static-credential fallback was deleted in the per-key cutover.

plus the fail-loud guards on a partial identity config.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from threetears.agent.tools.serve import _BuiltinToolBootstrap, _register_builtin_tools
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


def _set_signing_key_ref(monkeypatch: pytest.MonkeyPatch, pem: str) -> None:
    """provision THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF to resolve to ``pem``.

    the entrypoint reads a secret REFERENCE (``threetears.core.security.secret_refs``), never the
    raw key: publish ``pem`` under a private carrier var and point a plain ``env://`` ref at it.
    """
    monkeypatch.setenv("THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_TEST_RAW", pem)
    monkeypatch.setenv(
        "THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF",
        "env://THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_TEST_RAW",
    )


@pytest.mark.asyncio
async def test_identity_path_hands_toolserver_a_verifiable_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """with the signing key + pod id + issuer set, ToolServer gets an auth_token that mints a
    JWT verifiable against the pod's public JWKS (issuer + kid = pod id), and NO static creds."""
    pem = _signing_key_pem()
    _set_signing_key_ref(monkeypatch, pem)
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
async def test_identity_path_resolves_a_k8s_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF also resolves a k8s:// mounted-Secret ref.

    mirrors the ``env://`` / ``k8s://`` coverage pattern in
    ``3tears/packages/datasources/tests/unit/test_secrets.py``.
    """
    pem = _signing_key_pem()
    secret_dir = tmp_path / "secrets"
    (secret_dir / "builtin-tool-pod").mkdir(parents=True)
    (secret_dir / "builtin-tool-pod" / "identity-signing-key").write_text(pem, encoding="utf-8")
    monkeypatch.setenv("THREETEARS_DATASOURCE_SECRETS_DIR", str(secret_dir))
    monkeypatch.setenv(
        "THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF",
        "k8s://builtin-tool-pod/identity-signing-key",
    )
    monkeypatch.setenv("THREETEARS_TOOL_POD_ID", _POD_ID)
    monkeypatch.setenv("THREETEARS_TOOL_POD_CONNECT_ISSUER", _ISSUER)
    monkeypatch.delenv("THREETEARS_TOOL_POD_CUSTOMER_ID", raising=False)

    with patch("threetears.agent.tools.serve.ToolServer") as tool_server_cls:
        tool_server_cls.return_value = MagicMock()
        await _BuiltinToolBootstrap("builtin").build_server()

    provider = tool_server_cls.call_args.kwargs["auth_token"]
    verifying_jwks = IdentityMinter.from_pem(pem, kid=_POD_ID, issuer=_ISSUER).jwks()
    claims = verify_identity_token(provider(), jwks=verifying_jwks, issuer=_ISSUER)
    assert claims.sub == _POD_ID


@pytest.mark.asyncio
async def test_missing_signing_key_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """no signing key -> crash startup (the static-credential fallback was deleted, v0.14.1)."""
    monkeypatch.delenv("THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF", raising=False)
    # even with a static user/password present, there is NO fallback -- per-key is mandatory.
    monkeypatch.setenv("THREETEARS_NATS_USER", "tooling")
    monkeypatch.setenv("THREETEARS_NATS_PASSWORD", "secret")

    with pytest.raises(ValueError, match="THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF is required"):
        await _BuiltinToolBootstrap("builtin").build_server()


@pytest.mark.asyncio
async def test_identity_key_without_pod_id_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """a signing key with no pod id is a partial config -> crash (never silent fallback)."""
    _set_signing_key_ref(monkeypatch, _signing_key_pem())
    monkeypatch.delenv("THREETEARS_TOOL_POD_ID", raising=False)
    monkeypatch.setenv("THREETEARS_TOOL_POD_CONNECT_ISSUER", _ISSUER)

    with pytest.raises(ValueError, match="THREETEARS_TOOL_POD_ID"):
        await _BuiltinToolBootstrap("builtin").build_server()


@pytest.mark.asyncio
async def test_identity_key_without_issuer_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """a signing key with no connect issuer is a partial config -> crash."""
    _set_signing_key_ref(monkeypatch, _signing_key_pem())
    monkeypatch.setenv("THREETEARS_TOOL_POD_ID", _POD_ID)
    monkeypatch.delenv("THREETEARS_TOOL_POD_CONNECT_ISSUER", raising=False)

    with pytest.raises(ValueError, match="THREETEARS_TOOL_POD_CONNECT_ISSUER"):
        await _BuiltinToolBootstrap("builtin").build_server()


class TestEveryBuiltinToolIsAccountedFor:
    """a tool this pod does not serve must say so; it may never just be absent.

    ``_register_builtin_tools`` reports a ``registered`` / ``skipped`` summary that an
    operator reads as the complete picture. Any tool that is neither registered NOR in
    ``skipped_reasons`` vanishes from that accounting: the pod logs a healthy total,
    the registry never receives the tool, an agent's readiness gate never waits for it
    (the gate intersects the catalog with the access patterns, so a tool that never
    registered is never expected), and the only symptom is an agent that quietly
    cannot do the thing.

    ``web_search`` shipped that way. Its registration was gated on
    ``THREETEARS_SEARXNG_URL`` with no ``else``, so an unset variable dropped it with
    no counter, no reason and no log line -- observed live on cobalt-dev reporting
    "registered: 8, skipped: 3" with web_search mentioned nowhere.
    """

    def test_web_search_is_reported_as_skipped_when_searxng_is_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """an unset SEARXNG url must produce a skip REASON, not silence."""
        monkeypatch.delenv("THREETEARS_SEARXNG_URL", raising=False)
        server = MagicMock()

        with caplog.at_level("WARNING"):
            _register_builtin_tools(server)

        registered = {call.args[0].__class__.__name__ for call in server.register.call_args_list}
        assert "WebSearchTool" not in registered, "web_search cannot register without a SearXNG url"
        assert any("web_search" in record.message for record in caplog.records), (
            "web_search was dropped with NO log line. it must report itself as skipped, "
            "naming THREETEARS_SEARXNG_URL, or its absence is invisible to the operator."
        )
        assert any("THREETEARS_SEARXNG_URL" in str(record.__dict__) for record in caplog.records), (
            "the skip must name the env var that turns the tool back on"
        )

    def test_web_search_registers_when_searxng_is_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """the complement: with the url set, the tool is actually served.

        Without this half the test above would pass on a pod that never serves
        web_search under any configuration.
        """
        monkeypatch.setenv("THREETEARS_SEARXNG_URL", "http://searxng:8080")
        server = MagicMock()

        _register_builtin_tools(server)

        registered = {call.args[0].__class__.__name__ for call in server.register.call_args_list}
        assert "WebSearchTool" in registered, "web_search did not register even with THREETEARS_SEARXNG_URL set"
