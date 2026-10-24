"""GitHub OAuth 2.0 sign-in."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from threetears.iam.github import (
    AUTHORIZE_URL,
    DEFAULT_SCOPES,
    TOKEN_URL,
    DevicePending,
    DevicePendingReason,
    DeviceToken,
    GithubAccessDeniedError,
    GithubDeviceCode,
    GithubDeviceCodeExpiredError,
    GithubDeviceFlowDisabledError,
    GithubDeviceFlowError,
    GithubOAuth2Error,
    GithubProfile,
    HttpxGithubOAuth2Client,
    authorize_url,
    primary_verified_email,
    wait_for_device_token,
)
from threetears.iam.pkce import PkceError, compute_s256_challenge, create_challenge

#: GitHub's device-code endpoint and RFC 8628's device grant type: wire values, pinned here
DEVICE_CODE_URL = "https://github.com/login/device/code"
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


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


def test_authorize_url_suppresses_signup_by_default() -> None:
    # The opposite of GitHub's own default, deliberately: an app that allow-lists or provisions
    # against known accounts does not want a brand-new account created mid-sign-in, because
    # that account satisfies the flow while matching nothing on the other side.
    url = authorize_url(client_id="c", redirect_uri="https://a/b", state="s")
    assert parse_qs(urlparse(url).query)["allow_signup"] == ["false"]


def test_authorize_url_allows_signup_when_asked() -> None:
    url = authorize_url(client_id="c", redirect_uri="https://a/b", state="s", allow_signup=True)
    # Lower-case: GitHub reads this as a string, and "True" is not "true".
    assert parse_qs(urlparse(url).query)["allow_signup"] == ["true"]


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


# --- PKCE ---


def test_authorize_url_carries_an_s256_challenge_when_given_one() -> None:
    pair = create_challenge()
    params = parse_qs(
        urlparse(
            authorize_url(client_id="c", redirect_uri="https://a/b", state="s", code_challenge=pair.code_challenge)
        ).query
    )
    assert params["code_challenge"] == [pair.code_challenge]
    assert params["code_challenge_method"] == ["S256"]
    assert pair.code_verifier not in authorize_url(
        client_id="c", redirect_uri="https://a/b", state="s", code_challenge=pair.code_challenge
    )


def test_authorize_url_sends_no_pkce_parameters_without_a_challenge() -> None:
    """GitHub Enterprise Server 3.21 does not accept them, so absent means absent."""
    params = parse_qs(urlparse(authorize_url(client_id="c", redirect_uri="https://a/b", state="s")).query)
    assert "code_challenge" not in params
    assert "code_challenge_method" not in params


def test_authorize_url_refuses_an_empty_challenge() -> None:
    with pytest.raises(GithubOAuth2Error, match="code_challenge"):
        authorize_url(client_id="c", redirect_uri="https://a/b", state="s", code_challenge="")


async def test_exchange_code_sends_the_verifier_matching_the_challenge() -> None:
    pair = create_challenge()
    seen: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(parse_qs(request.content.decode()))
        return httpx.Response(200, json={"access_token": "gho_token"})

    assert await _client(handler).exchange_code("the-code", code_verifier=pair.code_verifier) == "gho_token"
    assert seen["code_verifier"] == [pair.code_verifier]
    assert compute_s256_challenge(seen["code_verifier"][0]) == pair.code_challenge
    assert seen["code"] == ["the-code"]


async def test_exchange_code_sends_no_verifier_without_one() -> None:
    seen: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(parse_qs(request.content.decode()))
        return httpx.Response(200, json={"access_token": "gho_token"})

    await _client(handler).exchange_code("the-code")
    assert "code_verifier" not in seen


async def test_exchange_code_refuses_a_malformed_verifier_before_sending() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a malformed verifier reached the wire")

    with pytest.raises(PkceError):
        await _client(handler).exchange_code("the-code", code_verifier="too-short")


async def test_exchange_code_on_a_client_built_for_the_device_flow_names_what_it_lacks() -> None:
    client = HttpxGithubOAuth2Client(
        httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500))), client_id="cid"
    )
    with pytest.raises(GithubOAuth2Error, match="client_secret and redirect_uri"):
        await client.exchange_code("the-code")


# --- the device flow (RFC 8628) ---


def _device_client(handler: Callable[[httpx.Request], httpx.Response]) -> HttpxGithubOAuth2Client:
    """A client as a headless CLI builds one: a client id and no secret."""
    return HttpxGithubOAuth2Client(httpx.AsyncClient(transport=httpx.MockTransport(handler)), client_id="cid")


_DEVICE_ANSWER = {
    "device_code": "dev-code",
    "user_code": "WDJB-MJHT",
    "verification_uri": "https://github.com/login/device",
    "expires_in": 900,
    "interval": 5,
}


async def test_request_device_code_returns_what_to_show_and_what_to_poll_with() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["accept"] = request.headers["accept"]
        seen["form"] = parse_qs(request.content.decode())
        return httpx.Response(200, json=_DEVICE_ANSWER)

    device = await _device_client(handler).request_device_code()

    assert device == GithubDeviceCode(
        device_code="dev-code",
        user_code="WDJB-MJHT",
        verification_uri="https://github.com/login/device",
        expires_in=900,
        interval=5,
    )
    assert seen["url"] == DEVICE_CODE_URL
    assert seen["accept"] == "application/json"
    assert seen["form"] == {"client_id": ["cid"], "scope": [" ".join(DEFAULT_SCOPES)]}


async def test_request_device_code_defaults_a_missing_interval_to_five_seconds() -> None:
    answer = {key: value for key, value in _DEVICE_ANSWER.items() if key != "interval"}
    device = await _device_client(lambda request: httpx.Response(200, json=answer)).request_device_code()
    assert device.interval == 5


async def test_request_device_code_on_an_app_without_the_device_flow_raises_disabled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": "device_flow_disabled", "error_description": "Device Flow must be explicitly enabled"}
        )

    with pytest.raises(GithubDeviceFlowDisabledError) as raised:
        await _device_client(handler).request_device_code()
    assert raised.value.error == "device_flow_disabled"


@pytest.mark.parametrize(
    "answer",
    [
        {**_DEVICE_ANSWER, "device_code": ""},
        {**_DEVICE_ANSWER, "expires_in": "900"},
        {**_DEVICE_ANSWER, "interval": True},
        {key: value for key, value in _DEVICE_ANSWER.items() if key != "user_code"},
    ],
)
async def test_request_device_code_rejects_a_malformed_answer(answer: dict[str, object]) -> None:
    with pytest.raises(GithubOAuth2Error, match="device code response"):
        await _device_client(lambda request: httpx.Response(200, json=answer)).request_device_code()


async def test_request_device_code_surfaces_a_server_error_by_status() -> None:
    with pytest.raises(GithubOAuth2Error, match="HTTP 503"):
        await _device_client(lambda request: httpx.Response(503, text="down")).request_device_code()


async def test_poll_device_token_sends_the_device_grant_and_returns_the_token() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["form"] = parse_qs(request.content.decode())
        return httpx.Response(200, json={"access_token": "gho_device", "token_type": "bearer"})

    assert await _device_client(handler).poll_device_token("dev-code") == DeviceToken(access_token="gho_device")
    assert seen["url"] == TOKEN_URL
    assert seen["form"] == {"client_id": ["cid"], "device_code": ["dev-code"], "grant_type": [DEVICE_GRANT_TYPE]}


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"error": "authorization_pending"}, DevicePending(reason=DevicePendingReason.AUTHORIZATION_PENDING)),
        ({"error": "slow_down", "interval": 10}, DevicePending(reason=DevicePendingReason.SLOW_DOWN, interval=10)),
        ({"error": "slow_down"}, DevicePending(reason=DevicePendingReason.SLOW_DOWN)),
    ],
)
async def test_poll_device_token_reports_a_pending_answer(answer: dict[str, object], expected: DevicePending) -> None:
    assert await _device_client(lambda request: httpx.Response(200, json=answer)).poll_device_token("d") == expected


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        ("expired_token", GithubDeviceCodeExpiredError),
        ("access_denied", GithubAccessDeniedError),
        ("device_flow_disabled", GithubDeviceFlowDisabledError),
        ("incorrect_device_code", GithubDeviceFlowError),
    ],
)
async def test_poll_device_token_raises_a_terminal_answer_as_its_type(
    code: str, error_type: type[GithubDeviceFlowError]
) -> None:
    with pytest.raises(error_type) as raised:
        await _device_client(lambda request: httpx.Response(200, json={"error": code})).poll_device_token("d")
    assert type(raised.value) is error_type
    assert raised.value.error == code


async def test_poll_device_token_rejects_an_answer_with_neither_token_nor_error() -> None:
    with pytest.raises(GithubOAuth2Error, match="neither a token nor an error"):
        await _device_client(lambda request: httpx.Response(200, json={"token_type": "bearer"})).poll_device_token("d")


async def test_poll_device_token_surfaces_transport_failure_as_auth_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(GithubOAuth2Error, match="ConnectError"):
        await _device_client(handler).poll_device_token("d")


class _ScriptedDevice:
    """A :class:`~threetears.iam.github.GithubOAuth2Client` whose polls answer from a script, with a clock it advances."""

    def __init__(self, answers: list[DeviceToken | DevicePending | GithubOAuth2Error]) -> None:
        self.answers = answers
        self.now = 0.0
        self.sleeps: list[float] = []
        self.polled_at: list[float] = []

    async def exchange_code(self, code: str, *, code_verifier: str | None = None) -> str:
        raise AssertionError("the device flow never exchanges a code")

    async def request_device_code(self, *, scopes: Sequence[str] = DEFAULT_SCOPES) -> GithubDeviceCode:
        raise AssertionError("the test hands the device code in")

    async def poll_device_token(self, device_code: str) -> DeviceToken | DevicePending:
        self.polled_at.append(self.now)
        answer = self.answers.pop(0)
        if isinstance(answer, GithubOAuth2Error):
            raise answer
        return answer

    async def fetch_profile(self, access_token: str) -> GithubProfile:
        raise AssertionError("not reached")

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


def _device(*, expires_in: int = 900, interval: int = 5) -> GithubDeviceCode:
    return GithubDeviceCode(
        device_code="dev-code",
        user_code="WDJB-MJHT",
        verification_uri="https://github.com/login/device",
        expires_in=expires_in,
        interval=interval,
    )


async def test_waiting_polls_at_the_interval_until_the_person_approves() -> None:
    pending = DevicePending(reason=DevicePendingReason.AUTHORIZATION_PENDING)
    script = _ScriptedDevice([pending, pending, DeviceToken(access_token="gho_device")])

    token = await wait_for_device_token(script, _device(interval=5), sleep=script.sleep, clock=script.clock)

    assert token == "gho_device"
    assert script.sleeps == [5, 5, 5]
    assert script.polled_at == [5, 10, 15]


async def test_slow_down_adds_five_seconds_to_every_later_poll() -> None:
    pending = DevicePending(reason=DevicePendingReason.AUTHORIZATION_PENDING)
    slow = DevicePending(reason=DevicePendingReason.SLOW_DOWN)
    script = _ScriptedDevice([pending, slow, pending, slow, DeviceToken(access_token="t")])

    await wait_for_device_token(script, _device(interval=5), sleep=script.sleep, clock=script.clock)

    assert script.sleeps == [5, 5, 10, 10, 15]


async def test_slow_down_takes_the_interval_github_names_when_it_is_longer() -> None:
    slow = DevicePending(reason=DevicePendingReason.SLOW_DOWN, interval=30)
    script = _ScriptedDevice([slow, DeviceToken(access_token="t")])

    await wait_for_device_token(script, _device(interval=5), sleep=script.sleep, clock=script.clock)

    assert script.sleeps == [5, 30]


async def test_waiting_stops_at_expires_in_without_polling_past_it() -> None:
    pending = DevicePending(reason=DevicePendingReason.AUTHORIZATION_PENDING)
    script = _ScriptedDevice([pending] * 10)

    with pytest.raises(GithubDeviceCodeExpiredError) as raised:
        await wait_for_device_token(script, _device(expires_in=17, interval=5), sleep=script.sleep, clock=script.clock)

    assert script.polled_at == [5, 10, 15]
    assert raised.value.error == "expired_token"


@pytest.mark.parametrize(
    "error",
    [
        GithubAccessDeniedError("declined", error="access_denied"),
        GithubDeviceCodeExpiredError("expired", error="expired_token"),
        GithubDeviceFlowDisabledError("disabled", error="device_flow_disabled"),
    ],
)
async def test_waiting_raises_a_terminal_answer_as_it_came(error: GithubDeviceFlowError) -> None:
    pending = DevicePending(reason=DevicePendingReason.AUTHORIZATION_PENDING)
    script = _ScriptedDevice([pending, error])

    with pytest.raises(type(error)) as raised:
        await wait_for_device_token(script, _device(), sleep=script.sleep, clock=script.clock)

    assert raised.value is error


async def test_the_device_token_reaches_the_same_profile_fetch_as_the_redirect_flow() -> None:
    """Request, poll, approve, then fetch_profile: the whole flow over one client."""
    polls = iter([{"error": "authorization_pending"}, {"access_token": "gho_device"}])

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == DEVICE_CODE_URL:
            return httpx.Response(200, json={**_DEVICE_ANSWER, "interval": 1})
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json=next(polls))
        assert request.headers["authorization"] == "Bearer gho_device"
        return httpx.Response(200, json={"id": 42, "login": "octocat", "email": "o@example.com"})

    client = _device_client(handler)
    slept: list[float] = []

    async def no_wait(seconds: float) -> None:
        slept.append(seconds)

    device = await client.request_device_code()
    token = await wait_for_device_token(client, device, sleep=no_wait)
    profile = await client.fetch_profile(token)

    assert profile.id == 42
    assert slept == [1, 1]
