"""GitHub OAuth 2.0 sign-in."""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from threetears.iam.github import (
    AUTHORIZE_URL,
    DEFAULT_SCOPES,
    GithubOAuth2Error,
    HttpxGithubOAuth2Client,
    authorize_url,
    primary_verified_email,
)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> HttpxGithubOAuth2Client:
    return HttpxGithubOAuth2Client(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        client_id="cid",
        client_secret="csecret",
        redirect_uri="https://app.example/callback",
    )


def test_authorize_url_carries_the_expected_parameters() -> None:
    url = authorize_url(client_id="cid", redirect_uri="https://app.example/cb", state="st4te")
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    assert url.startswith(AUTHORIZE_URL)
    assert params["client_id"] == ["cid"]
    assert params["redirect_uri"] == ["https://app.example/cb"]
    assert params["state"] == ["st4te"]
    assert params["scope"] == [" ".join(DEFAULT_SCOPES)]


def test_authorize_url_requests_minimal_scopes() -> None:
    # Nothing here needs repository access; asking for it is a worse consent screen and a
    # far worse breach.
    assert DEFAULT_SCOPES == ("read:user", "user:email")
    assert "repo" not in authorize_url(client_id="c", redirect_uri="https://a/b", state="s")


def test_authorize_url_refuses_an_empty_state() -> None:
    # A flow without state accepts a callback the user never initiated.
    with pytest.raises(GithubOAuth2Error, match="state"):
        authorize_url(client_id="cid", redirect_uri="https://app.example/cb", state="")


def test_scopes_are_overridable() -> None:
    url = authorize_url(client_id="c", redirect_uri="https://a/b", state="s", scopes=["read:org"])
    assert parse_qs(urlparse(url).query)["scope"] == ["read:org"]


async def test_exchange_code_returns_the_access_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/access_token")
        body = request.content.decode()
        assert "code=the-code" in body
        assert "client_secret=csecret" in body
        return httpx.Response(200, json={"access_token": "gho_token", "token_type": "bearer"})

    assert await _client(handler).exchange_code("the-code") == "gho_token"


async def test_exchange_code_rejects_githubs_200_error_shape() -> None:
    # GitHub answers a bad or expired code with HTTP 200 and an `error` field, so a status
    # check alone would treat a failed exchange as a success.
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "bad_verification_code"})

    with pytest.raises(GithubOAuth2Error, match="no access_token"):
        await _client(handler).exchange_code("stale")


@pytest.mark.parametrize("status", [400, 401, 500, 503])
async def test_exchange_code_rejects_error_statuses(status: int) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    with pytest.raises(GithubOAuth2Error, match="token exchange failed"):
        await _client(handler).exchange_code("code")


async def test_exchange_code_surfaces_transport_failure_as_auth_failure() -> None:
    # A GitHub outage is "sign-in unavailable", not a 500 from this service.
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(GithubOAuth2Error, match="request failed"):
        await _client(handler).exchange_code("code")


async def test_exchange_code_rejects_a_non_json_body() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    with pytest.raises(GithubOAuth2Error, match="non-JSON"):
        await _client(handler).exchange_code("code")


async def test_fetch_profile_uses_the_public_email_when_present() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer gho_token"
        return httpx.Response(200, json={"id": 42, "login": "octocat", "email": "o@example.com", "name": "Octo"})

    profile = await _client(handler).fetch_profile("gho_token")
    assert (profile.id, profile.login, profile.email, profile.name) == (42, "octocat", "o@example.com", "Octo")


async def test_fetch_profile_falls_back_to_the_verified_primary_email() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/user/emails"):
            return httpx.Response(
                200,
                json=[
                    {"email": "secondary@example.com", "primary": False, "verified": True},
                    {"email": "primary@example.com", "primary": True, "verified": True},
                ],
            )
        return httpx.Response(200, json={"id": 42, "login": "octocat", "email": None})

    assert (await _client(handler).fetch_profile("t")).email == "primary@example.com"


async def test_fetch_profile_tolerates_an_unavailable_email_endpoint() -> None:
    # An account with no resolvable address still has a valid identity.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/user/emails"):
            return httpx.Response(403)
        return httpx.Response(200, json={"id": 42, "login": "octocat", "email": None})

    profile = await _client(handler).fetch_profile("t")
    assert profile.email is None
    assert profile.id == 42


async def test_fetch_profile_rejects_a_response_missing_the_identity_key() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"login": "octocat"})

    with pytest.raises(GithubOAuth2Error, match="missing an id or login"):
        await _client(handler).fetch_profile("t")


@pytest.mark.parametrize("status", [401, 404, 500])
async def test_fetch_profile_rejects_error_statuses(status: int) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    with pytest.raises(GithubOAuth2Error, match="profile fetch failed"):
        await _client(handler).fetch_profile("t")


def test_primary_verified_email_requires_both_flags() -> None:
    # An unverified address proves only that someone typed it. Trusting one lets an attacker
    # claim an account by pre-registering against a victim's address.
    assert primary_verified_email([{"email": "a@example.com", "primary": True, "verified": False}]) is None
    assert primary_verified_email([{"email": "a@example.com", "primary": False, "verified": True}]) is None
    assert primary_verified_email([{"email": "a@example.com", "primary": True, "verified": True}]) == "a@example.com"


@pytest.mark.parametrize("payload", [None, {}, "not-a-list", [], [None], ["nope"], [{"primary": True}]])
def test_primary_verified_email_tolerates_malformed_payloads(payload: object) -> None:
    assert primary_verified_email(payload) is None
