"""unit tests for :class:`threetears.mcp.auth.BearerTokenIdentityProvider`.

the v2 HTTP identity provider resolves the calling :class:`Identity`
per request from a bearer token via an injected token->identity
resolver. these tests exercise the four contract points:

- a resolvable token returns the expected identity;
- an absent token raises ``RuntimeError``;
- an unresolvable token (resolver raises) raises ``RuntimeError``;
- two requests with two different tokens resolve to two different
  identities (per-request, never a cached single identity).
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from threetears.mcp.auth import (
    BearerTokenIdentityProvider,
    Identity,
    IdentityProvider,
)


class _TokenHolder:
    """mutable per-request token source used to prove re-resolution.

    stands in for the request-scoped contextvar the transport layer
    populates in production: mutating :attr:`token` between
    :meth:`identify` calls models two distinct requests.
    """

    def __init__(self, token: str | None = None) -> None:
        """capture the initial token value.

        :param token: initial bearer token (or ``None`` for absent)
        :ptype token: str | None
        :return: nothing
        :rtype: None
        """
        self.token = token

    def __call__(self) -> str | None:
        """return the current token value.

        :return: current bearer token or ``None``
        :rtype: str | None
        """
        return self.token


def _resolver_over(mapping: dict[str, Identity]):  # type: ignore[no-untyped-def]
    """build a token->identity resolver backed by ``mapping``.

    :param mapping: token string to :class:`Identity`
    :ptype mapping: dict[str, Identity]
    :return: async resolver raising ``RuntimeError`` on unknown token
    :rtype: Callable
    """

    async def _resolve(token: str) -> Identity:
        if token not in mapping:
            raise RuntimeError(f"no identity for token {token!r}")
        return mapping[token]

    return _resolve


class TestBearerTokenIdentityProvider:
    """bearer-token identity resolution contract."""

    def test_satisfies_identity_provider_protocol(self) -> None:
        """the provider passes the ``@runtime_checkable`` Protocol check."""
        provider = BearerTokenIdentityProvider(
            resolver=_resolver_over({}),
            token_source=_TokenHolder(),
        )
        assert isinstance(provider, IdentityProvider)

    @pytest.mark.asyncio
    async def test_resolvable_token_returns_expected_identity(self) -> None:
        """a token the resolver knows returns the mapped identity."""
        expected = Identity(principal_type="user", principal_id=uuid4())
        provider = BearerTokenIdentityProvider(
            resolver=_resolver_over({"tok-a": expected}),
            token_source=_TokenHolder("tok-a"),
        )
        identity = await provider.identify()
        assert identity == expected

    @pytest.mark.asyncio
    async def test_absent_token_raises_runtime_error(self) -> None:
        """no bearer token in the request context raises RuntimeError."""
        provider = BearerTokenIdentityProvider(
            resolver=_resolver_over({}),
            token_source=_TokenHolder(None),
        )
        with pytest.raises(RuntimeError):
            await provider.identify()

    @pytest.mark.asyncio
    async def test_empty_token_raises_runtime_error(self) -> None:
        """an empty-string bearer token is treated as absent."""
        provider = BearerTokenIdentityProvider(
            resolver=_resolver_over({}),
            token_source=_TokenHolder(""),
        )
        with pytest.raises(RuntimeError):
            await provider.identify()

    @pytest.mark.asyncio
    async def test_unresolvable_token_raises_runtime_error(self) -> None:
        """a token the resolver rejects (raises RuntimeError) propagates as RuntimeError."""
        provider = BearerTokenIdentityProvider(
            resolver=_resolver_over({}),
            token_source=_TokenHolder("nope"),
        )
        with pytest.raises(RuntimeError):
            await provider.identify()

    @pytest.mark.asyncio
    async def test_resolver_non_runtime_exception_wrapped_as_runtime_error(self) -> None:
        """a resolver raising a non-RuntimeError still surfaces as RuntimeError (contract)."""

        async def _boom(token: str) -> Identity:  # noqa: ARG001
            raise ValueError("decode failed")

        provider = BearerTokenIdentityProvider(
            resolver=_boom,
            token_source=_TokenHolder("bad"),
        )
        with pytest.raises(RuntimeError):
            await provider.identify()

    @pytest.mark.asyncio
    async def test_per_request_reresolution_two_tokens_two_identities(self) -> None:
        """two requests with two tokens resolve to two identities (never cached)."""
        id_a = Identity(principal_type="user", principal_id=uuid4())
        id_b = Identity(principal_type="user", principal_id=uuid4())
        holder = _TokenHolder("tok-a")
        provider = BearerTokenIdentityProvider(
            resolver=_resolver_over({"tok-a": id_a, "tok-b": id_b}),
            token_source=holder,
        )
        first = await provider.identify()
        holder.token = "tok-b"
        second = await provider.identify()
        assert first == id_a
        assert second == id_b
        assert first != second
