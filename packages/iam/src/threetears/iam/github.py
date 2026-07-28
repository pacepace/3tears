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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol
from urllib.parse import urlencode

import httpx

from threetears.observe import get_logger

__all__ = [
    "USER_URL",
    "TOKEN_URL",
    "EMAILS_URL",
    "AUTHORIZE_URL",
    "DEFAULT_SCOPES",
    "GithubOAuth2Client",
    "GithubOAuth2Error",
    "GithubProfile",
    "HttpxGithubOAuth2Client",
    "authorize_url",
    "primary_verified_email",
]

log = get_logger(__name__)

AUTHORIZE_URL: Final[str] = "https://github.com/login/oauth/authorize"
TOKEN_URL: Final[str] = "https://github.com/login/oauth/access_token"
USER_URL: Final[str] = "https://api.github.com/user"
EMAILS_URL: Final[str] = "https://api.github.com/user/emails"

#: The login and id, plus the verified primary email. Nothing more.
DEFAULT_SCOPES: Final[tuple[str, ...]] = ("read:user", "user:email")


class GithubOAuth2Error(Exception):
    """A GitHub sign-in step failed.

    Covers a transport failure, a non-200 response, and GitHub's habit of returning HTTP 200
    with an ``error`` field for a bad or expired code. All three are authentication failures
    rather than server errors: a GitHub outage must surface as "sign-in unavailable", not a
    500 from this service.
    """


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
    :return: the authorization URL.
    :rtype: str
    :raises GithubOAuth2Error: ``state`` is empty.
    """
    if not state:
        raise GithubOAuth2Error("an authorization URL requires a non-empty state value.")
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        # Lower-case literals: GitHub reads this as a string, and "False" is not "false".
        "allow_signup": "true" if allow_signup else "false",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


class GithubOAuth2Client(Protocol):
    """The GitHub HTTP seam: code to token, token to profile."""

    async def exchange_code(self, code: str) -> str:
        """Exchange an authorization ``code`` for a GitHub access token."""
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
        client_secret: str,
        redirect_uri: str,
    ) -> None:
        """
        :param client: the HTTP client to issue requests through.
        :ptype client: httpx.AsyncClient
        :param client_id: the OAuth app's client id.
        :ptype client_id: str
        :param client_secret: the OAuth app's client secret. Held as a plain string because it
            materializes in a form body on every exchange; keep it in a
            ``pydantic.SecretStr`` (or resolve it from a secret reference) at the CONFIG
            layer and unwrap it here, so it is masked everywhere it is merely carried.
        :ptype client_secret: str
        :param redirect_uri: the registered callback, echoed back at exchange time.
        :ptype redirect_uri: str
        """
        self._client = client
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri

    async def exchange_code(self, code: str) -> str:
        try:
            response = await self._client.post(
                TOKEN_URL,
                headers={"Accept": "application/json"},
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": self._redirect_uri,
                },
            )
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
