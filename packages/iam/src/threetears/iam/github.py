"""GitHub sign-in: the OAuth 2.0 authorization-code flow against GitHub.

GitHub is not an OIDC provider -- there is no discovery document and no
``id_token`` -- so it needs its own small client rather than going through
:mod:`threetears.iam.oidc`. That client is about a hundred lines, and it had
been written twice before this module existed.

**Scopes are minimal and fixed.** ``read:user`` for the login and the immutable
numeric id; ``user:email`` so an account with a private email still resolves a
verified address. Nothing here needs repository access, and asking for it on
the consent screen is both a worse experience and a much worse breach.

**Identity is keyed on the numeric id, never the login.** A GitHub login can be
changed by its owner and, once released, claimed by someone else. Keying a
local account on the login means a renamed user loses their account and a
stranger can inherit it. The numeric id is immutable.

**The state parameter is not this module's job.** CSRF protection for the
round trip needs storage, and the two services this was factored out of park it
in different places -- so it lives behind
:class:`~threetears.iam.stores.base.StateStore` and
:class:`~threetears.iam.stores.base.SingleUseTicketStore`. What this module
guarantees is that :func:`authorize_url` will not build a URL without a state
value, so forgetting it is a type error rather than a silent CSRF hole.

**PKCE is recommended, not forced.** GitHub accepts an S256 ``code_challenge`` for OAuth
apps; ``plain`` is refused. Pass :func:`authorize_url` a challenge built with
:func:`threetears.iam.pkce.create_challenge` and the matching verifier to
:meth:`HttpxGithubOAuth2Client.exchange_code`, and an intercepted code is useless without
the verifier. Both are optional only because GitHub Enterprise Server 3.21 does not accept
the parameters yet.

**The device flow (RFC 8628) is for machines without a browser.** A CLI reached over SSH
cannot receive a redirect, so it asks for a device code
(:meth:`HttpxGithubOAuth2Client.request_device_code`), shows the person the user code and
verification URL to open on any other device, and polls until they approve
(:func:`wait_for_device_token`). The token it yields goes through the same
:meth:`~HttpxGithubOAuth2Client.fetch_profile` as the redirect flow. The OAuth app must have
"Enable Device Flow" switched on in its settings, and GitHub advises it only for headless
use; without it GitHub answers ``device_flow_disabled``, raised as
:class:`GithubDeviceFlowDisabledError`. The device flow does not use PKCE.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Protocol, TypeGuard
from urllib.parse import urlencode

import httpx

from threetears.iam.pkce import S256_METHOD, validate_code_verifier
from threetears.observe import get_logger

__all__ = [
    "USER_URL",
    "TOKEN_URL",
    "EMAILS_URL",
    "AUTHORIZE_URL",
    "DEFAULT_SCOPES",
    "DevicePending",
    "DevicePendingReason",
    "DeviceToken",
    "GithubAccessDeniedError",
    "GithubDeviceCode",
    "GithubDeviceCodeExpiredError",
    "GithubDeviceFlowDisabledError",
    "GithubDeviceFlowError",
    "GithubOAuth2Client",
    "GithubOAuth2Error",
    "GithubProfile",
    "HttpxGithubOAuth2Client",
    "authorize_url",
    "primary_verified_email",
    "wait_for_device_token",
]

log = get_logger(__name__)

AUTHORIZE_URL: Final[str] = "https://github.com/login/oauth/authorize"
TOKEN_URL: Final[str] = "https://github.com/login/oauth/access_token"
USER_URL: Final[str] = "https://api.github.com/user"
EMAILS_URL: Final[str] = "https://api.github.com/user/emails"
_DEVICE_CODE_URL: Final[str] = "https://github.com/login/device/code"

#: RFC 8628 SS3.4's grant type for redeeming a device code.
_DEVICE_GRANT_TYPE: Final[str] = "urn:ietf:params:oauth:grant-type:device_code"

#: RFC 8628 SS3.5: on ``slow_down`` the interval grows by five seconds, for this and every
#: later poll.
_SLOW_DOWN_STEP: Final[int] = 5

#: RFC 8628 SS3.2: the interval a client uses when the server names none.
_DEFAULT_DEVICE_INTERVAL: Final[int] = 5

#: The login and id, plus the verified primary email. Nothing more.
DEFAULT_SCOPES: Final[tuple[str, ...]] = ("read:user", "user:email")


class GithubOAuth2Error(Exception):
    """A GitHub sign-in step failed.

    Covers a transport failure, a non-200 response, and GitHub's habit of returning HTTP 200
    with an ``error`` field for a bad or expired code. All three are authentication failures
    rather than server errors: a GitHub outage must surface as "sign-in unavailable", not a
    500 from this service.
    """


class GithubDeviceFlowError(GithubOAuth2Error):
    """The device flow ended without a token.

    Raised for the terminal answers below, by their subclasses, and for any other ``error``
    GitHub returns (an unknown client id, a malformed device code), which this class carries
    as :attr:`error`.

    :ivar error: GitHub's ``error`` code, or ``None`` when the flow ended for another reason
    """

    def __init__(self, message: str, *, error: str | None = None) -> None:
        super().__init__(message)
        self.error = error


class GithubDeviceCodeExpiredError(GithubDeviceFlowError):
    """The device code expired before the person approved it (``expired_token``, or
    ``expires_in`` passed while polling). Start again with a new device code."""


class GithubAccessDeniedError(GithubDeviceFlowError):
    """The person declined the authorization (``access_denied``)."""


class GithubDeviceFlowDisabledError(GithubDeviceFlowError):
    """The OAuth app does not have "Enable Device Flow" switched on (``device_flow_disabled``).

    A configuration fault, not a user action: the app's owner turns it on in the app's settings.
    """


#: GitHub's terminal device-flow error codes, and the error each is raised as.
_DEVICE_FLOW_ERRORS: Final[dict[str, type[GithubDeviceFlowError]]] = {
    "expired_token": GithubDeviceCodeExpiredError,
    "access_denied": GithubAccessDeniedError,
    "device_flow_disabled": GithubDeviceFlowDisabledError,
}


class DevicePendingReason(StrEnum):
    """Why a device-flow poll has no token yet. Both mean "poll again"."""

    #: the person has not approved yet
    AUTHORIZATION_PENDING = "authorization_pending"
    #: polling too fast: the interval grows by five seconds
    SLOW_DOWN = "slow_down"


@dataclass(frozen=True, slots=True)
class GithubDeviceCode:
    """What a device-code request returns: what to show the person, and what to poll with.

    :ivar device_code: the code the client polls with. Not shown to the person.
    :ivar user_code: the code the person types at :attr:`verification_uri`.
    :ivar verification_uri: where the person approves the request, on any device.
    :ivar expires_in: seconds until the device code expires.
    :ivar interval: the minimum seconds between polls.
    """

    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


@dataclass(frozen=True, slots=True)
class DeviceToken:
    """A device-flow poll that yielded the access token.

    :ivar access_token: the GitHub access token, for :meth:`GithubOAuth2Client.fetch_profile`
    """

    access_token: str


@dataclass(frozen=True, slots=True)
class DevicePending:
    """A device-flow poll with no token yet.

    :ivar reason: why there is none yet
    :ivar interval: the interval GitHub named with a ``slow_down``, if it named one
    """

    reason: DevicePendingReason
    interval: int | None = None


@dataclass(frozen=True, slots=True)
class GithubProfile:
    """A GitHub account, as far as authentication cares.

    :ivar id: the immutable numeric account id. THIS is the identity key -- see the module
        docstring on why the login is not.
    :ivar login: the current username. Display and allow-listing only.
    :ivar email: the primary verified email, or ``None`` when the account has none public and
        the ``user:email`` scope was refused. Never treat ``None`` as "no such user".
    :ivar name: the display name, if set.
    """

    id: int
    login: str
    email: str | None = None
    name: str | None = None


def authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    scopes: Sequence[str] = DEFAULT_SCOPES,
    allow_signup: bool = False,
    code_challenge: str | None = None,
) -> str:
    """Build the URL to send a signing-in user to.

    :param client_id: the OAuth app's client id.
    :ptype client_id: str
    :param redirect_uri: where GitHub returns the user. Must match the app's registered
        callback exactly.
    :ptype redirect_uri: str
    :param state: the CSRF value to correlate this redirect with its callback. Required, and
        rejected if empty: a flow without state accepts a callback the user never initiated.
    :ptype state: str
    :param scopes: the scopes to request.
    :ptype scopes: Sequence[str]
    :param allow_signup: whether GitHub may offer account creation during the flow. Defaults
        to ``False``, which is the opposite of GitHub's own default and deliberate: an
        application that allow-lists or provisions against known accounts does not want a
        brand-new account created mid-sign-in, because that account satisfies the flow while
        matching nothing on the other side. Pass ``True`` only for genuinely open sign-up.
    :ptype allow_signup: bool
    :param code_challenge: the PKCE S256 challenge, from
        :func:`threetears.iam.pkce.create_challenge`; sent with ``code_challenge_method=S256``,
        the only method this package sends. Recommended: keep the challenge's verifier and pass
        it to :meth:`GithubOAuth2Client.exchange_code`. Optional only because GitHub Enterprise
        Server 3.21 does not accept it yet.
    :ptype code_challenge: str | None
    :return: the authorization URL.
    :rtype: str
    :raises GithubOAuth2Error: ``state`` is empty, or ``code_challenge`` is given but empty.
    """
    if not state:
        raise GithubOAuth2Error("an authorization URL requires a non-empty state value.")
    if code_challenge is not None and not code_challenge:
        raise GithubOAuth2Error("a PKCE code_challenge, when given, must be non-empty.")
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        # Lower-case literals: GitHub reads this as a string, and "False" is not "false".
        "allow_signup": "true" if allow_signup else "false",
    }
    if code_challenge is not None:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = S256_METHOD
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


class GithubOAuth2Client(Protocol):
    """The GitHub HTTP seam: code to token, device code to token, token to profile."""

    async def exchange_code(self, code: str, *, code_verifier: str | None = None) -> str:
        """Exchange an authorization ``code`` (and its PKCE verifier, if any) for an access token."""
        ...

    async def request_device_code(self, *, scopes: Sequence[str] = DEFAULT_SCOPES) -> GithubDeviceCode:
        """Start the device flow: a device code to poll with, and a user code to show."""
        ...

    async def poll_device_token(self, device_code: str) -> DeviceToken | DevicePending:
        """Poll once for the device flow's token."""
        ...

    async def fetch_profile(self, access_token: str) -> GithubProfile:
        """Fetch the authenticated account's profile."""
        ...


class HttpxGithubOAuth2Client:
    """The real :class:`GithubOAuth2Client`, over an injected ``httpx.AsyncClient``.

    The client is supplied rather than constructed here so its timeouts, connection limits,
    and lifecycle stay with the application that owns them -- and so a test can pass an
    ``httpx.MockTransport`` and never touch the network.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        client_id: str,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
    ) -> None:
        """
        :param client: the HTTP client to issue requests through.
        :ptype client: httpx.AsyncClient
        :param client_id: the OAuth app's client id.
        :ptype client_id: str
        :param client_secret: the OAuth app's client secret. Held as a plain string because it
            materializes in a form body on every exchange; keep it in a
            ``pydantic.SecretStr`` (or resolve it from a secret reference) at the CONFIG
            layer and unwrap it here, so it is masked everywhere it is merely carried. Needed
            by :meth:`exchange_code` only: the device flow sends none, so a headless client
            that holds no secret builds this without one.
        :ptype client_secret: str | None
        :param redirect_uri: the registered callback, echoed back at exchange time. Needed by
            :meth:`exchange_code` only.
        :ptype redirect_uri: str | None
        """
        self._client = client
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri

    async def exchange_code(self, code: str, *, code_verifier: str | None = None) -> str:
        """Exchange an authorization ``code`` for a GitHub access token.

        :param code: the code from the callback
        :ptype code: str
        :param code_verifier: the verifier of the PKCE challenge given to :func:`authorize_url`,
            when one was. Recommended; see that function.
        :ptype code_verifier: str | None
        :return: the access token
        :rtype: str
        :raises GithubOAuth2Error: the exchange failed, or this client was built without the
            client secret and redirect URI an exchange needs
        :raises threetears.iam.pkce.PkceError: ``code_verifier`` is not a well-formed verifier
        """
        if self._client_secret is None or self._redirect_uri is None:
            raise GithubOAuth2Error(
                "exchange_code needs the client_secret and redirect_uri this client was built without."
            )
        data = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "code": code,
            "redirect_uri": self._redirect_uri,
        }
        if code_verifier is not None:
            validate_code_verifier(code_verifier)
            data["code_verifier"] = code_verifier
        try:
            response = await self._client.post(TOKEN_URL, headers={"Accept": "application/json"}, data=data)
        except httpx.HTTPError as exc:
            raise GithubOAuth2Error(f"github token exchange request failed ({type(exc).__name__}).") from exc
        if response.status_code != httpx.codes.OK:
            raise GithubOAuth2Error(f"github token exchange failed (HTTP {response.status_code}).")
        token = _json_object(response).get("access_token")
        if not isinstance(token, str) or not token:
            # GitHub answers a bad or expired code with HTTP 200 and an `error` field, so the
            # status code alone does not tell you the exchange worked.
            raise GithubOAuth2Error("github token exchange returned no access_token.")
        return token

    async def request_device_code(self, *, scopes: Sequence[str] = DEFAULT_SCOPES) -> GithubDeviceCode:
        """Start the device flow (RFC 8628 SS3.1).

        :param scopes: the scopes to request
        :ptype scopes: Sequence[str]
        :return: the device code, and what to show the person
        :rtype: GithubDeviceCode
        :raises GithubDeviceFlowDisabledError: the app does not have the device flow enabled
        :raises GithubDeviceFlowError: GitHub answered with another ``error``
        :raises GithubOAuth2Error: the request failed, or the answer is malformed
        """
        body = await self._post_form(
            _DEVICE_CODE_URL, {"client_id": self._client_id, "scope": " ".join(scopes)}, what="device code request"
        )
        _raise_device_error(body)
        device_code = body.get("device_code")
        user_code = body.get("user_code")
        verification_uri = body.get("verification_uri")
        expires_in = body.get("expires_in")
        interval = body.get("interval", _DEFAULT_DEVICE_INTERVAL)
        if not (
            isinstance(device_code, str)
            and device_code
            and isinstance(user_code, str)
            and user_code
            and isinstance(verification_uri, str)
            and verification_uri
            and _is_positive_int(expires_in)
            and _is_positive_int(interval)
        ):
            raise GithubOAuth2Error("github device code response is missing or mistypes a required field.")
        return GithubDeviceCode(
            device_code=device_code,
            user_code=user_code,
            verification_uri=verification_uri,
            expires_in=expires_in,
            interval=interval,
        )

    async def poll_device_token(self, device_code: str) -> DeviceToken | DevicePending:
        """Poll once for the device flow's token (RFC 8628 SS3.4).

        :func:`wait_for_device_token` drives this at the right pace; call it directly only to
        run your own loop, and then honour the interval and ``slow_down`` yourself.

        :param device_code: :attr:`GithubDeviceCode.device_code`
        :ptype device_code: str
        :return: the token, or why there is none yet
        :rtype: DeviceToken | DevicePending
        :raises GithubDeviceCodeExpiredError: the device code expired
        :raises GithubAccessDeniedError: the person declined
        :raises GithubDeviceFlowDisabledError: the app does not have the device flow enabled
        :raises GithubDeviceFlowError: GitHub answered with another ``error``
        :raises GithubOAuth2Error: the request failed, or the answer is malformed
        """
        body = await self._post_form(
            TOKEN_URL,
            {"client_id": self._client_id, "device_code": device_code, "grant_type": _DEVICE_GRANT_TYPE},
            what="device token poll",
        )
        error = body.get("error")
        if error in (DevicePendingReason.AUTHORIZATION_PENDING, DevicePendingReason.SLOW_DOWN):
            interval = body.get("interval")
            return DevicePending(
                reason=DevicePendingReason(error),
                interval=interval if _is_positive_int(interval) else None,
            )
        _raise_device_error(body)
        token = body.get("access_token")
        if not isinstance(token, str) or not token:
            raise GithubOAuth2Error("github device token response carries neither a token nor an error.")
        return DeviceToken(access_token=token)

    async def _post_form(self, url: str, data: Mapping[str, str], *, what: str) -> Mapping[str, Any]:
        """POST a form asking for JSON, and decode the JSON object GitHub answers with.

        GitHub answers device-flow errors with HTTP 200 and an ``error`` field, and some with
        a 4xx carrying the same field, so a 4xx whose body names an ``error`` is returned for
        the caller to map rather than collapsed into a status code.
        """
        try:
            response = await self._client.post(url, headers={"Accept": "application/json"}, data=dict(data))
        except httpx.HTTPError as exc:
            raise GithubOAuth2Error(f"github {what} failed ({type(exc).__name__}).") from exc
        if response.status_code == httpx.codes.OK:
            return _json_object(response)
        if httpx.codes.BAD_REQUEST <= response.status_code < httpx.codes.INTERNAL_SERVER_ERROR:
            try:
                body = _json_object(response)
            except GithubOAuth2Error:
                body = {}
            if isinstance(body.get("error"), str):
                return body
        raise GithubOAuth2Error(f"github {what} failed (HTTP {response.status_code}).")

    async def fetch_profile(self, access_token: str) -> GithubProfile:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }
        try:
            response = await self._client.get(USER_URL, headers=headers)
            if response.status_code != httpx.codes.OK:
                raise GithubOAuth2Error(f"github profile fetch failed (HTTP {response.status_code}).")
            profile = _json_object(response)
            email = profile.get("email")
            if not email:
                # A private-email account returns null here. The verified primary address is
                # a separate call, and a failure of it is not fatal -- an account with no
                # resolvable email still has a valid identity.
                emails = await self._client.get(EMAILS_URL, headers=headers)
                if emails.status_code == httpx.codes.OK:
                    email = primary_verified_email(emails.json())
                else:
                    log.info(
                        "github email lookup unavailable; continuing without an address",
                        extra={"extra_data": {"status": emails.status_code}},
                    )
        except httpx.HTTPError as exc:
            raise GithubOAuth2Error(f"github profile fetch request failed ({type(exc).__name__}).") from exc

        account_id = profile.get("id")
        login = profile.get("login")
        if not isinstance(account_id, int) or not isinstance(login, str) or not login:
            raise GithubOAuth2Error("github profile response is missing an id or login.")
        name = profile.get("name")
        return GithubProfile(
            id=account_id,
            login=login,
            email=str(email) if email else None,
            name=str(name) if name else None,
        )


async def wait_for_device_token(
    client: GithubOAuth2Client,
    device: GithubDeviceCode,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Poll the device flow until the person approves, declines, or the code expires.

    Waits :attr:`GithubDeviceCode.interval` before each poll. On ``slow_down`` the interval
    grows by five seconds for every later poll (RFC 8628 SS3.5), or to the interval GitHub
    names if that is longer. Stops once the next poll would land past
    :attr:`GithubDeviceCode.expires_in`, counted from the call.

    :param client: the GitHub client
    :ptype client: GithubOAuth2Client
    :param device: what :meth:`GithubOAuth2Client.request_device_code` returned
    :ptype device: GithubDeviceCode
    :param sleep: the wait between polls; injectable so a test does not wait
    :ptype sleep: Callable[[float], Awaitable[None]]
    :param clock: seconds, monotonic; injectable with ``sleep``
    :ptype clock: Callable[[], float]
    :return: the access token, for :meth:`GithubOAuth2Client.fetch_profile`
    :rtype: str
    :raises GithubDeviceCodeExpiredError: the code expired, by GitHub's answer or by the clock
    :raises GithubAccessDeniedError: the person declined
    :raises GithubDeviceFlowDisabledError: the app does not have the device flow enabled
    :raises GithubDeviceFlowError: GitHub answered with another ``error``
    :raises GithubOAuth2Error: a poll failed
    """
    deadline = clock() + device.expires_in
    interval = device.interval
    while True:
        if clock() + interval > deadline:
            raise GithubDeviceCodeExpiredError(
                f"the device code expired after {device.expires_in}s without approval.", error="expired_token"
            )
        await sleep(interval)
        outcome = await client.poll_device_token(device.device_code)
        if isinstance(outcome, DeviceToken):
            return outcome.access_token
        if outcome.reason is DevicePendingReason.SLOW_DOWN:
            interval = max(interval + _SLOW_DOWN_STEP, outcome.interval or 0)


def _raise_device_error(body: Mapping[str, Any]) -> None:
    """Raise the typed error for a terminal device-flow ``error``, if the body carries one.

    :param body: GitHub's decoded answer
    :ptype body: Mapping[str, Any]
    :return: nothing, when the body carries no ``error``
    :rtype: None
    :raises GithubDeviceFlowError: the error the body names, as its subclass where it has one
    """
    error = body.get("error")
    if error is None:
        return
    code = str(error)
    raise _DEVICE_FLOW_ERRORS.get(code, GithubDeviceFlowError)(f"github device flow refused ({code}).", error=code)


def _is_positive_int(value: object) -> TypeGuard[int]:
    """Whether ``value`` is an ``int`` above zero, and not a ``bool``.

    :param value: an untrusted JSON value
    :ptype value: object
    :return: whether it is a usable count of seconds
    :rtype: TypeGuard[int]
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def primary_verified_email(entries: object) -> str | None:
    """Pick the primary VERIFIED address from GitHub's ``/user/emails`` payload.

    Both conditions are required. An unverified address proves only that someone typed it,
    so trusting one lets an attacker claim an account by pre-registering a GitHub account
    against a victim's address -- the classic account-linking takeover.

    :param entries: the decoded ``/user/emails`` response. Typed loosely because it is
        untrusted input, and this function is where it stops being untrusted.
    :ptype entries: object
    :return: the address, or ``None`` if there is no primary verified one.
    :rtype: str | None
    """
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("primary") and entry.get("verified"):
            value = entry.get("email")
            return str(value) if value else None
    return None


def _json_object(response: httpx.Response) -> Mapping[str, Any]:
    """Decode a response body that must be a JSON object.

    A non-object body is a protocol violation, and treating it as an empty mapping would
    turn "GitHub returned something unexpected" into "the field was missing" -- the same
    outcome for very different reasons.
    """
    try:
        body = response.json()
    except ValueError as exc:
        raise GithubOAuth2Error("github returned a non-JSON response.") from exc
    if not isinstance(body, Mapping):
        raise GithubOAuth2Error("github returned a JSON value that is not an object.")
    return body
