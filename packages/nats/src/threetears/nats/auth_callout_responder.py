"""Generalized NATS auth-callout responder (platform-auth A).

The NATS server delegates each connection's authentication to a callout responder: it publishes an
AuthorizationRequest on ``$SYS.REQ.USER.AUTH`` and applies the AuthorizationResponse the responder
replies with. :mod:`threetears.nats.auth_callout` holds the generic ``jwt/v2`` request/response
codecs; THIS module holds the generic responder LOOP, parameterized by two consumer seams so every
3tears consumer (aibots, scriob, ...) shares one responder and supplies only its own policy:

* :class:`PrincipalResolver` -- "who is this connection?": map an :class:`AuthCalloutRequest` to a
  :class:`ResolvedPrincipal`, or ``None`` to DENY. Each consumer implements its own (a
  bootstrap-token-hash lookup; an EdDSA identity-JWT verification; ...). Fail closed: a resolver
  returns ``None`` for anything it does not positively authenticate.
* :class:`GrantPolicy` -- "what may this principal do?": the least-privilege
  :class:`PrincipalPermissions` allow-list for a resolved principal.

The responder owns only the generic mechanics: subscribe the callout subject on the system account
(a queue group, so one responder in the group answers each request), decode the request, delegate
resolve + grant, mint the admit (a scoped user JWT) or the deny response, and reply. It fails closed
everywhere -- an unresolved principal denies, an undecodable request is left unanswered (the server
times out and denies), and a bad account signing key refuses to start.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nkeys import PREFIX_BYTE_ACCOUNT, NkeysError

from threetears.nats.auth_callout import AuthCalloutRequest, decode_auth_request, mint_auth_response
from threetears.nats.subject_permissions import PrincipalPermissions
from threetears.nats.subjects import Subject
from threetears.nats.user_jwt import account_public_key, mint_user_jwt
from threetears.observe import get_logger

__all__ = [
    "AUTH_CALLOUT_SUBJECT",
    "DEFAULT_AUTH_CALLOUT_QUEUE_GROUP",
    "DEFAULT_NATS_USER_JWT_TTL_SECONDS",
    "AuthAccountKeyError",
    "AuthCalloutResponder",
    "GrantPolicy",
    "PrincipalResolver",
    "ResolvedPrincipal",
]

log = get_logger(__name__)

#: the system subject the NATS server publishes each AuthorizationRequest on.
AUTH_CALLOUT_SUBJECT = "$SYS.REQ.USER.AUTH"

#: the default queue group the responder joins, so only ONE responder in the group answers each
#: request when a consumer runs several (e.g. one per pod). Override per deployment.
DEFAULT_AUTH_CALLOUT_QUEUE_GROUP = "auth-callout"

#: canonical default TTL (seconds) for each minted NATS user JWT -- the connection-auth credential
#: lifetime. A SHORT TTL bounds how long a since-revoked principal keeps access; because a client
#: re-auths a margin before expiry (a proactive reconnect), a short TTL just means more frequent
#: transparent reconnects, not downtime. The consumer that reports a re-auth margin to its clients
#: should pass the SAME value here so the minted TTL and the reported TTL can never drift.
DEFAULT_NATS_USER_JWT_TTL_SECONDS = 3600


class AuthAccountKeyError(ValueError):
    """a configured NATS auth-account key is not usable (fail closed).

    Raised at :class:`AuthCalloutResponder` construction (every path) for either bad account-key
    input -- the signing ``account_seed`` (not a usable nkey seed) or the ``issuer_account`` identity
    key (not an account public key) -- so a misconfigured key aborts startup rather than yielding a
    responder that mints unverifiable (server-rejected) responses.
    """


def _validated_seed(account_seed: bytes) -> bytes:
    """the stripped account seed, or raise :class:`AuthAccountKeyError` (validates WITHOUT leaking it)."""
    seed = account_seed.strip()
    try:
        account_public_key(seed)
    except (ValueError, NkeysError) as exc:
        raise AuthAccountKeyError(
            f"NATS auth account signing key is not a usable nkey seed ({type(exc).__name__})"
        ) from None
    return seed


def _validated_issuer_account(issuer_account: str) -> str:
    """the account IDENTITY public key (``A...``), or raise :class:`AuthAccountKeyError`.

    When ``account_seed`` is a subordinate SIGNING key rather than the account's identity key, every
    minted JWT must name the account it belongs to via ``issuer_account`` -- the account's public
    IDENTITY key -- or the NATS server cannot bind the connection to the account and denies it. A
    misconfigured value would therefore fail EVERY connect at runtime; validating it here fails closed
    at construction instead (mirroring :func:`_validated_seed`). Validates the nkey shape the same way
    ``nkeys`` itself does -- base32-decodable with the ACCOUNT prefix byte -- without trusting the input.
    """
    candidate = issuer_account.strip()
    try:
        raw = base64.b32decode(candidate + "=" * (-len(candidate) % 8))
    except binascii.Error, ValueError:
        raise AuthAccountKeyError("NATS issuer_account is not a decodable nkey public key") from None
    # a 32-byte ed25519 account public key encodes to prefix(1) + key(32) + crc(2) = 35 bytes, prefix first.
    if len(raw) != 35 or raw[0] != PREFIX_BYTE_ACCOUNT:
        raise AuthAccountKeyError("NATS issuer_account is not an ACCOUNT public key (expects 'A...')") from None
    return candidate


@dataclass(frozen=True, slots=True)
class ResolvedPrincipal:
    """an authenticated principal the responder mints a user JWT for + a :class:`GrantPolicy` scopes.

    :param conn_id: the AUTHENTICATED id a scoped reply inbox keys on -- from the verified identity,
        NEVER the spoofable connect ``name`` -- so one principal cannot key its inbox onto another's
        subtree and intercept its replies. A :class:`GrantPolicy` typically feeds it to
        :func:`~threetears.nats.inbox_prefix_for`.
    :ptype conn_id: str
    :param name: the human-readable name the minted user JWT carries (for operator visibility).
    :ptype name: str
    :param claims: the resolver->policy payload -- the identity + scoping fields a
        :class:`GrantPolicy` reads to build the allow-list (e.g. a principal kind, a tenant id, an
        agent id). Opaque to the responder; each consumer defines its own shape.
    :ptype claims: Mapping[str, Any]
    """

    conn_id: str
    name: str
    claims: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PrincipalResolver(Protocol):
    """maps an inbound auth-callout request to an authenticated principal, or denies (fail closed)."""

    async def resolve(self, request: AuthCalloutRequest) -> ResolvedPrincipal | None:
        """resolve ``request`` to a :class:`ResolvedPrincipal`, or ``None`` to DENY the connection.

        :param request: the decoded AuthorizationRequest (the presented credential + connect opts).
        :ptype request: AuthCalloutRequest
        :return: the authenticated principal, or ``None`` when nothing is positively authenticated.
        :rtype: ResolvedPrincipal | None
        """
        ...


@runtime_checkable
class GrantPolicy(Protocol):
    """the least-privilege NATS subject allow-list for one resolved principal."""

    def permissions(self, principal: ResolvedPrincipal) -> PrincipalPermissions:
        """the pub/sub allow-list + JetStream resources ``principal`` is scoped to.

        :param principal: the authenticated principal to scope.
        :ptype principal: ResolvedPrincipal
        :return: the resolved allow-list the minted user JWT carries.
        :rtype: PrincipalPermissions
        """
        ...


class AuthCalloutResponder:
    """answers NATS auth-callout requests by minting per-connection user JWTs from a grant policy.

    :param nc: a connected NATS client that is a member of the system account (so it can subscribe
        ``$SYS.REQ.USER.AUTH``). Only ``subscribe``/``unsubscribe``/``publish_raw_reply`` are used.
    :ptype nc: Any
    :param account_seed: the NATS auth-account nkey seed that SIGNS the response + user JWTs. Either
        the account's identity seed, or a subordinate SIGNING seed registered in the account's
        ``signing_keys`` -- in which case pass ``issuer_account`` so the identity seed can stay offline.
    :ptype account_seed: bytes
    :param resolver: the consumer's :class:`PrincipalResolver` ("who is this?").
    :ptype resolver: PrincipalResolver
    :param policy: the consumer's :class:`GrantPolicy` ("what may they do?").
    :ptype policy: GrantPolicy
    :param account_name: the NATS account name minted users are placed in (the user JWT ``aud`` in
        config-mode callout); ``None`` in operator mode (placement follows the issuer).
    :ptype account_name: str | None
    :param issuer_account: the account's public IDENTITY key (``A...``) to stamp on every minted JWT
        when ``account_seed`` is a subordinate SIGNING key distinct from the identity key -- the NATS
        server reads it to bind the connection to the account. OMIT (``None``) when the identity seed
        itself signs. Supplying it lets the root identity seed stay OFFLINE (only the rotatable
        signing seed is deployed), which shrinks the blast radius of a compromised responder host.
    :ptype issuer_account: str | None
    :param queue_group: the callout subscription's queue group (one responder per group answers).
    :ptype queue_group: str
    :param user_jwt_ttl_seconds: TTL on each minted user JWT; a reconnect re-mints, so a short TTL
        bounds how long a since-revoked principal keeps access.
    :ptype user_jwt_ttl_seconds: int
    """

    def __init__(
        self,
        nc: Any,
        *,
        account_seed: bytes,
        resolver: PrincipalResolver,
        policy: GrantPolicy,
        account_name: str | None = None,
        issuer_account: str | None = None,
        queue_group: str = DEFAULT_AUTH_CALLOUT_QUEUE_GROUP,
        user_jwt_ttl_seconds: int = DEFAULT_NATS_USER_JWT_TTL_SECONDS,
    ) -> None:
        self._nc = nc
        self._account_seed = _validated_seed(account_seed)
        self._resolver = resolver
        self._policy = policy
        self._account_name = account_name
        self._issuer_account = _validated_issuer_account(issuer_account) if issuer_account is not None else None
        self._queue_group = queue_group
        self._ttl_seconds = user_jwt_ttl_seconds
        self._subscription: Any = None

    @classmethod
    def from_secret(
        cls,
        account_seed: str | bytes,
        *,
        nc: Any,
        resolver: PrincipalResolver,
        policy: GrantPolicy,
        account_name: str | None = None,
        issuer_account: str | None = None,
        queue_group: str = DEFAULT_AUTH_CALLOUT_QUEUE_GROUP,
        user_jwt_ttl_seconds: int = DEFAULT_NATS_USER_JWT_TTL_SECONDS,
    ) -> AuthCalloutResponder:
        """build a responder from the account signing seed, FAILING CLOSED on a bad key.

        :param account_seed: the account nkey seed (``SA...``), as ``str`` or ``bytes``; never logged.
        :ptype account_seed: str | bytes
        :param issuer_account: the account's public IDENTITY key (``A...``) when ``account_seed`` is a
            subordinate signing seed; ``None`` when the identity seed signs. See :class:`AuthCalloutResponder`.
        :ptype issuer_account: str | None
        :return: a ready responder.
        :rtype: AuthCalloutResponder
        :raises AuthAccountKeyError: when the seed is not a usable NATS account nkey seed, or
            ``issuer_account`` is not an account public key.
        """
        seed = account_seed.encode("ascii") if isinstance(account_seed, str) else bytes(account_seed)
        # __init__ validates the seed + issuer_account (fail closed) — a str convenience over the same guard.
        return cls(
            nc,
            account_seed=seed,
            resolver=resolver,
            policy=policy,
            account_name=account_name,
            issuer_account=issuer_account,
            queue_group=queue_group,
            user_jwt_ttl_seconds=user_jwt_ttl_seconds,
        )

    async def start(self) -> None:
        """subscribe the auth-callout subject on the system account (in the queue group)."""
        self._subscription = await self._nc.subscribe(
            subject=Subject.raw(AUTH_CALLOUT_SUBJECT),
            queue=self._queue_group,
            cb=self.handle_request,
        )
        log.info("auth-callout responder started: subject=%s queue=%s", AUTH_CALLOUT_SUBJECT, self._queue_group)

    async def stop(self) -> None:
        """unsubscribe the auth-callout subject."""
        if self._subscription is not None:
            await self._nc.unsubscribe(self._subscription)
            self._subscription = None

    async def handle_request(self, msg: Any) -> None:
        """decode an AuthorizationRequest, mint the response, and reply on the request's inbox.

        A request without a reply subject, or one we cannot even decode, is left UNANSWERED -- the
        NATS server then times out and denies. (An undecodable request is not ours to answer; a
        malformed one cannot be safely admitted.)

        :param msg: the inbound message; ``data`` is the request JWT, ``reply_subject`` the inbox.
        :ptype msg: Any
        :return: nothing.
        :rtype: None
        """
        if not msg.reply_subject:
            return
        try:
            request = decode_auth_request(bytes(msg.data).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, TypeError) as exc:
            # malformed / non-UTF-8 / absent (``data is None``) payload — not ours to answer.
            log.warning("auth-callout: undecodable request (%s)", type(exc).__name__)
            return
        response = await self.build_response(request)
        await self._nc.publish_raw_reply(reply_subject=msg.reply_subject, payload=response.encode("utf-8"))

    async def build_response(self, request: AuthCalloutRequest) -> str:
        """resolve + authorize the principal, then mint the admit/deny AuthorizationResponse.

        Delegates "who is this?" to the :class:`PrincipalResolver` and "what may they do?" to the
        :class:`GrantPolicy` (via :meth:`_authorize`). ANYTHING short of a fully resolved, granted,
        minted user JWT DENIES (fail closed) with a signed error the server applies.

        ``server_id_value`` is read FIRST and is allowed to propagate: a request with no server id
        cannot be signed a deny (there is no ``aud`` to bind it to), so it is left unanswered and the
        server times out — the only fail-closed option when a deny itself cannot be minted.

        :param request: the decoded AuthorizationRequest.
        :ptype request: AuthCalloutRequest
        :return: the signed AuthorizationResponse JWT (admit with a scoped user JWT, or deny).
        :rtype: str
        """
        server_id = request.server_id_value
        user_nkey = request.user_nkey
        user_jwt = await self._authorize(request)
        if user_jwt is None:
            return mint_auth_response(
                account_seed=self._account_seed,
                server_id=server_id,
                user_nkey=user_nkey,
                issuer_account=self._issuer_account,
                error="authentication failed",
            )
        return mint_auth_response(
            account_seed=self._account_seed,
            server_id=server_id,
            user_nkey=user_nkey,
            issuer_account=self._issuer_account,
            user_jwt=user_jwt,
        )

    async def _authorize(self, request: AuthCalloutRequest) -> str | None:
        """resolve + grant + mint the scoped user JWT, or ``None`` to DENY.

        The security boundary. The resolver + policy are consumer-supplied code, so ANY exception
        they (or the mint) raise must DENY — never propagate to admit-by-accident or wedge the
        callback. A resolver that returns ``None``, or that raises, both deny; only a fully resolved,
        granted, minted JWT admits.

        :param request: the decoded AuthorizationRequest.
        :ptype request: AuthCalloutRequest
        :return: the minted user JWT to admit with, or ``None`` to deny.
        :rtype: str | None
        """
        try:
            resolved = await self._resolver.resolve(request)
            if resolved is None:
                log.warning("auth-callout: denied -- no principal resolved for the presented credential")
                return None
            permissions = self._policy.permissions(resolved)
            user_jwt = mint_user_jwt(
                account_seed=self._account_seed,
                user_public_key=request.user_nkey,
                permissions=permissions,
                name=resolved.name,
                expires_in_seconds=self._ttl_seconds,
                audience=self._account_name,
                issuer_account=self._issuer_account,
            )
            log.info("auth-callout: admitted name=%s conn_id=%s", resolved.name, resolved.conn_id)
            return user_jwt
        except Exception as exc:  # noqa: BLE001 - auth boundary: any resolver/policy/mint fault denies (fail closed)
            log.warning("auth-callout: denied -- resolve/grant/mint raised (%s)", type(exc).__name__)
            return None
