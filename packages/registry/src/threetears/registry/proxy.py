"""call proxy for routing tool calls to tool pods.

subscribes to NATS call subject, validates tool availability
in catalog, selects endpoint via pluggable routing strategy,
tracks in-flight calls, and forwards to tool pod via
NATS request-reply.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, model_validator

from threetears.agent.tools.context_envelope import CallContext, bind_log_context
from threetears.core.security.identity_token import (
    IdentityClaims,
    IdentityKeyNotFoundError,
    IdentityTokenError,
    canonical_call_hash,
    verify_identity_token,
)
from threetears.core.security.pop import access_token_hash, verify_pop_proof
from threetears.nats import IncomingMessage, RequestError, Subject, Subjects
from threetears.observe import InflightRequestsGauge, clear_context, get_logger
from threetears.registry.auth import AgentToolAuthorizer, EndpointUsageEmitter, LimitGuard
from threetears.registry.catalog import ToolCatalog
from threetears.registry.routing import LeastConnectionsStrategy, RoutingStrategy

# the issuer the Hub stamps on identity tokens, and the clock-skew tolerance the proxy allows
# on exp/iat + the pop iat freshness window. constants for now; promote to config if operations
# need to tune them.
_IDENTITY_ISSUER = "hub"
_IDENTITY_LEEWAY_SECONDS = 60
_POP_LEEWAY_SECONDS = 60

if TYPE_CHECKING:
    from threetears.core.coordination.replay_guard import ReplayGuard
    from threetears.core.security import ProxyAssertionSigner
    from threetears.nats import NatsClient, Subscription

__all__ = [
    "CallProxy",
    "ProxyCallRequest",
    "ProxyCallResponse",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Wire-format Pydantic models
# ---------------------------------------------------------------------------


_LEGACY_FLAT_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {"conversation_id", "user_id", "customer_id", "correlation_id", "agent_id"}
)


class ProxyCallRequest(BaseModel):
    """incoming tool call request from agent.

    all per-call identity dimensions (conversation_id, user_id,
    customer_id, correlation_id, agent_id) ride as a single nested
    :class:`CallContext` under ``context`` and are forwarded to the
    target tool pod untouched. :class:`CallContext.agent_id` is the
    single source of truth for the originating agent identity -- the
    proxy reads it for authorization + routing decisions after
    deserialization, which happens at the same moment the context
    becomes available, so a separate top-level ``agent_id`` field was a
    duplicate representation and has been removed.

    :param tool_name: namespaced name of tool to invoke
    :ptype tool_name: str
    :param tool_version: version of tool to invoke
    :ptype tool_version: str
    :param arguments: tool input parameters
    :ptype arguments: dict[str, Any]
    :param context: unified identity + trace envelope forwarded
        verbatim to the tool pod. must be present and carry
        ``agent_id`` for the proxy to route the call; stateless
        utility calls still populate :class:`CallContext` even if only
        with ``agent_id`` + ``correlation_id``
    :ptype context: CallContext | None
    :param pop: proof-of-possession for THIS request on the agent→proxy
        hop (the caller signs over the request so a leaked identity
        token alone is unusable). the proxy verifies it on every call
        (enforce-only); a request without a valid pop is rejected
    :ptype pop: str | None
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_version: str
    arguments: dict[str, Any]
    context: CallContext | None = None
    pop: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_flat_identity_fields(cls, data: Any) -> Any:
        """reject removed flat identity fields with a migration pointer.

        all legacy flat identity fields -- ``conversation_id``,
        ``user_id``, ``customer_id``, ``correlation_id``, and
        ``agent_id`` -- have moved onto :class:`CallContext`. callers
        sending any of them as top-level wire fields hit this rejector
        with a message naming the offender so the migration point is
        obvious.

        :param data: raw input dict (mode='before' runs pre-coercion)
        :ptype data: Any
        :return: unchanged input when no legacy fields are present
        :rtype: Any
        :raises ValueError: when any legacy flat identity field is
            present on the wire
        """
        if isinstance(data, dict):
            offending = sorted(_LEGACY_FLAT_IDENTITY_FIELDS & data.keys())
            if offending:
                fields_list = ", ".join(offending)
                raise ValueError(
                    f"legacy flat identity field(s) {fields_list} rejected on "
                    f"ProxyCallRequest; migrated to CallContext, see "
                    f"threetears.agent.tools.context_envelope.CallContext"
                )
        return data


class ProxyCallResponse(BaseModel):
    """outgoing tool call response to agent.

    the response echoes the inbound :class:`CallContext` verbatim so
    identity has one shape on both sides of the proxy hop. there is no
    top-level ``correlation_id`` string; log-border stringification
    reads ``str(response.context.correlation_id)`` when needed. the
    field is ``None`` only when the inbound request carried no context
    at all (which is also rejected upstream because routing requires
    ``context.agent_id``; ``None`` survives only in error responses
    built from a malformed inbound request).

    :param success: whether tool execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param metadata: optional additional metadata
    :ptype metadata: dict[str, Any] | None
    :param error: error message if execution failed
    :ptype error: str | None
    :param error_code: machine-readable error code
    :ptype error_code: str | None
    :param context: unified identity + trace envelope echoed from the
        inbound :class:`ProxyCallRequest`; ``None`` only on
        malformed-request error responses where no context could be
        parsed
    :ptype context: CallContext | None
    """

    success: bool
    content: str
    metadata: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None
    context: CallContext | None = None


# ---------------------------------------------------------------------------
# CallProxy
# ---------------------------------------------------------------------------


class CallProxy:
    """proxies tool calls from agents to tool pods.

    subscribes to call subject with queue group for HA,
    validates tool availability in catalog, selects endpoint
    via configurable routing strategy, tracks in-flight call
    counts, and forwards calls via NATS request-reply.
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        authorizer: AgentToolAuthorizer,
        pop_replay_guard: "ReplayGuard",
        limit_guard: LimitGuard,
        namespace: str = "3tears",
        timeout: float | None = None,
        routing_strategy: RoutingStrategy | None = None,
        jwks_provider: Callable[[], dict[str, Any]] | None = None,
        jwks_refresh: Callable[[], Awaitable[bool]] | None = None,
        proxy_signer: "ProxyAssertionSigner | None" = None,
        usage_emitter: EndpointUsageEmitter | None = None,
        inflight_gauge: InflightRequestsGauge | None = None,
    ) -> None:
        """initialize call proxy.

        platform-auth is now ENFORCE-ONLY: every dispatch verifies the Hub-issued identity
        token and the per-call proof-of-possession unconditionally and re-stamps the verified
        identity onto the request; a call that fails either gate is rejected (fail-closed). There
        is no off/warn ladder and no inert path.

        :param catalog: tool catalog for tool lookup
        :ptype catalog: ToolCatalog
        :param authorizer: agent tool authorizer for access control;
            REQUIRED. every tool dispatch is gated through the
            authorizer — no silent-bypass path. dev/test callers pass
            :class:`AllowAllAuthorizer` /
            :class:`DenyAllAuthorizer`; production wires
            :class:`RbacEvaluatorAuthorizer`
        :ptype authorizer: AgentToolAuthorizer
        :param pop_replay_guard: records each pop nonce for single-use enforcement; REQUIRED.
            without it a captured pop could be replayed verbatim for the same call body within the
            iat freshness window, so the enforce-only proxy must always carry one
        :ptype pop_replay_guard: ReplayGuard
        :param limit_guard: pre-call spend gate; REQUIRED. every tool dispatch is
            gated through the limit guard after the pop check and before catalog
            routing -- no silent-bypass path, same discipline as ``authorizer``. the
            money path FAILS OPEN (Fork-2): the proxy denies only on a returned
            :class:`LimitDecision(allowed=False)`; a guard that RAISES or is
            unreachable makes the proxy SERVE the call (and log loudly) so a
            billing-infra outage never bricks tool traffic. dev/test callers pass
            :class:`AllowAllLimitGuard` / :class:`DenyAllLimitGuard`; production wires
            the counter-backed ``KvCallLimitGuard`` (gu-task-15a)
        :ptype limit_guard: LimitGuard
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param timeout: default timeout in seconds for forwarded NATS requests.
            sourced from THREETEARS_REGISTRY_CALL_TIMEOUT env var if not provided.
        :ptype timeout: float | None
        :param routing_strategy: endpoint selection strategy (defaults to least-connections)
        :ptype routing_strategy: RoutingStrategy | None
        :param jwks_provider: zero-arg callable returning the current Hub
            JWKS (the public keys the identity token is verified against).
            ``None`` makes every verification fail-closed (the call is rejected). The
            provider's contract is to return a JWKS dict
        :ptype jwks_provider: Callable[[], dict[str, Any]] | None
        :param jwks_refresh: optional zero-arg coroutine that triggers ONE
            immediate, debounced + rate-limited Hub JWKS refresh and returns
            whether it ran (typically :meth:`CachedHubJwksProvider.refresh_now`).
            When a token verification fails because the cached JWKS holds no key
            for the token's ``kid`` (a Hub re-key the cache has not caught up to),
            ``_verify_identity`` calls this ONCE and re-verifies, so a valid token
            signed under a freshly-rotated key self-heals on the first such failure
            rather than after a full steady refresh interval. ``None`` (the only
            shape dev/test callers wire, with a static JWKS) disables the reactive
            path -- verification stays fail-closed against the supplied JWKS
        :ptype jwks_refresh: Callable[[], Awaitable[bool]] | None
        :param proxy_signer: signs a proxy->pod assertion onto each forwarded call so the pod can
            verify the call came from the proxy, for this body, once; ``None`` leaves the call
            unsigned (the binding is inert until the proxy key is provisioned)
        :ptype proxy_signer: ProxyAssertionSigner | None
        :param usage_emitter: post-call usage-emit seam; ``None`` (the safe default)
            emits nothing. when present, the proxy calls
            :meth:`EndpointUsageEmitter.emit` fire-and-forget after the tool pod
            replies (both request args + response content in hand) -- an emit failure
            is caught and logged and NEVER affects the reply. this is the slot the hub
            fills with its concrete SDK-typed emitter (gu-task-16); 3tears holds only
            the protocol + the slot
        :ptype usage_emitter: EndpointUsageEmitter | None
        :param inflight_gauge: leak-safe prometheus in-flight-requests gauge
            bracketed around every :meth:`_process_call` so KEDA's prometheus
            scaler can autoscale registry replicas on aggregate in-flight tool-
            call load. the registry server owns the one gauge (on the registry
            it serves through the HealthServer's ``/metrics`` route) and passes
            it here; ``None`` (tests / standalone) self-provisions a private
            gauge so the bracket is always live
        :ptype inflight_gauge: InflightRequestsGauge | None
        """
        from threetears.registry.config import get_call_timeout

        self._catalog = catalog
        self._namespace = namespace
        self.timeout = timeout if timeout is not None else get_call_timeout()
        self._authorizer = authorizer
        self._limit_guard = limit_guard
        self._usage_emitter = usage_emitter
        self._routing_strategy: RoutingStrategy = routing_strategy or LeastConnectionsStrategy()
        self._jwks_provider = jwks_provider
        self._jwks_refresh = jwks_refresh
        self._proxy_signer = proxy_signer
        self._pop_replay_guard = pop_replay_guard
        self._inflight_gauge = inflight_gauge or InflightRequestsGauge("threetears_registry_inflight_requests")
        self._nc: "NatsClient | None" = None
        self._sub: "Subscription | None" = None
        self._active_tasks: set[asyncio.Task[None]] = set()

    async def start(self, nc: "NatsClient") -> None:
        """start listening for tool call requests.

        DQ-B7 queue-group note: ``queue="registry"`` makes
        ``{ns}.tools.call`` load-balance across registry replicas so
        each agent call is handled by exactly one replica. each
        replica's local routing strategy then selects an endpoint
        from the shared catalog.

        :param nc: connected canonical NATS wrapper client
        :ptype nc: NatsClient
        :return: nothing
        :rtype: None
        """
        self._nc = nc
        subject = Subjects.tools_call()
        self._sub = await nc.subscribe(
            subject=subject,
            queue="registry",
            cb=self.handle_call,
        )
        log.info(
            "call proxy started",
            extra={"extra_data": {"subject": subject.path, "timeout": self.timeout}},
        )

    async def stop(self) -> None:
        """stop listening and drain in-flight tool call tasks."""
        if self._sub is not None and self._nc is not None:
            await self._nc.unsubscribe(self._sub)
            self._sub = None
        if self._active_tasks:
            log.info(
                "draining in-flight tool call tasks",
                extra={"extra_data": {"count": len(self._active_tasks)}},
            )
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()
        log.info("call proxy stopped")

    async def handle_call(self, msg: IncomingMessage) -> None:
        """public NATS-subject handler that dispatches a tool call.

        bound by :meth:`start` as the ``cb`` callback on
        ``{namespace}.tools.call``. tests exercise this surface
        directly; the name + single-``msg`` shape are part of the
        stability contract.

        spawns _process_call as concurrent task so the NATS
        subscription callback returns immediately, allowing
        parallel processing of multiple tool call requests.

        :param msg: incoming wrapper envelope containing call request
        :ptype msg: IncomingMessage
        """
        task = asyncio.create_task(self._process_call(msg))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def _process_call(self, msg: IncomingMessage) -> None:
        """process tool call request concurrently.

        validates tool exists, selects endpoint via routing strategy,
        tracks in-flight count, forwards call to tool pod, and
        returns result transparently. :attr:`CallContext.agent_id` is
        the single source of truth for routing / authorization; a
        missing context or missing ``context.agent_id`` surfaces as a
        ``MALFORMED_REQUEST`` response with a pointer to the rename.
        binds the canonical logging context tags (``cid``/``conv``/
        ``user``/``agent``/``customer``) from the inbound
        :class:`CallContext` for the duration of the dispatch so every
        log line in this handler and its callees renders with those
        tags; cleared in ``finally`` to avoid bleeding identifiers
        across concurrently-handled calls on the same asyncio task.

        :param msg: incoming NATS message containing call request
        :ptype msg: Any
        :raises RuntimeError: when invoked before ``start`` connects NATS
        """
        if self._nc is None:
            raise RuntimeError("_process_call invoked before NATS connected")
        # bracket the whole dispatch in the in-flight gauge: increment on entry,
        # decrement on exit even when dispatch raises (try/finally inside
        # ``track``), so KEDA's prometheus scaler reads the true concurrent-call
        # count and a failed call never strands the counter above baseline.
        with self._inflight_gauge.track():
            try:
                request = ProxyCallRequest.model_validate_json(msg.data)
            except Exception as exc:
                response = ProxyCallResponse(
                    success=False,
                    content="",
                    error=f"malformed call request: {exc}",
                    error_code="MALFORMED_REQUEST",
                )
                if msg.reply_subject is not None:
                    await self._nc.publish_reply(
                        reply_subject=msg.reply_subject,
                        message=response,
                    )
                return

            bind_log_context(request.context)
            try:
                await self._dispatch_call(request, msg)
            finally:
                clear_context()

    def _load_jwks(self) -> dict[str, Any]:
        """fetch the current Hub JWKS via the injected provider, converting ANY provider

        failure into an :class:`IdentityTokenError`. Once the provider is Hub-backed it may be a
        network fetch and can raise far beyond the verification exceptions (ConnectionError,
        TimeoutError, ...). Converting here keeps a flaky provider from ESCAPING verification
        and hanging the call: it becomes a well-typed verification failure that fails the call
        closed -- always a response, never a silent hang. The failure is logged (not
        swallowed) before being re-raised.
        """
        assert self._jwks_provider is not None  # guarded by the caller
        try:
            return self._jwks_provider()
        except Exception as exc:
            # the provider is external (a network fetch once Hub-wired); we cannot enumerate its
            # failure modes, and any of them means "cannot verify" -> fail to a response. log the
            # exception MESSAGE (str(exc)) alongside its type so a provider-unavailable failure is
            # distinguishable in the log from a token/JWKS-shape failure (the message is a structural
            # reason, never token or key material).
            log.warning(
                "JWKS provider failed during identity verification",
                extra={"extra_data": {"reason": type(exc).__name__, "detail": str(exc)}},
            )
            raise IdentityTokenError(f"JWKS provider unavailable ({type(exc).__name__})") from exc

    async def _verify_token_reactively(self, token: str, *, refreshed: list[bool]) -> "IdentityClaims":
        """verify a Hub token against the cached JWKS; on a kid-not-in-cache miss, refresh once + retry.

        The reactive self-heal for a Hub re-key: :func:`verify_identity_token` raises the distinct
        :class:`IdentityKeyNotFoundError` when the cached JWKS holds no key for the token's ``kid``
        (the Hub rotated, or the cache is stale after a Hub pod move). That -- and ONLY that -- is
        recoverable, so this triggers one immediate :attr:`_jwks_refresh` and re-verifies. An expired
        / bad-signature / malformed token raises the BASE :class:`IdentityTokenError`, which is NOT
        caught here, so it never provokes a Hub fetch -- a flood of bad tokens cannot be turned into a
        Hub stampede. The refresh is fired at most ONCE per verify-path call (``refreshed`` is shared
        across the handshake + user-assertion verifications), and :meth:`refresh_now` is itself
        debounced + rate-limited, so the two layers together bound Hub load.

        :param token: the compact-JWS identity token to verify
        :ptype token: str
        :param refreshed: a single-element mutable flag, shared across this call's verifications, so
            the reactive refresh fires at most once even if both tokens miss the cache
        :ptype refreshed: list[bool]
        :return: the verified identity claims
        :rtype: IdentityClaims
        :raises IdentityTokenError: when the token cannot be verified (after the at-most-one refresh)
        """
        try:
            return verify_identity_token(
                token, jwks=self._load_jwks(), issuer=_IDENTITY_ISSUER, leeway_seconds=_IDENTITY_LEEWAY_SECONDS
            )
        except IdentityKeyNotFoundError:
            if self._jwks_refresh is None or refreshed[0]:
                raise  # no reactive trigger wired, or we already refreshed once this call -> reject
            refreshed[0] = True
            await self._jwks_refresh()
            return verify_identity_token(
                token, jwks=self._load_jwks(), issuer=_IDENTITY_ISSUER, leeway_seconds=_IDENTITY_LEEWAY_SECONDS
            )

    async def _verify_identity(
        self, request: "ProxyCallRequest"
    ) -> tuple["ProxyCallRequest", "ProxyCallResponse | None"]:
        """verify the Hub-issued identity token and re-stamp the VERIFIED identity.

        the heart of the platform-auth fix: authorization + forwarding must act on an
        authenticated identity, not the self-asserted envelope. on success the verified
        ``agent_id`` (``= token.sub``), ``user_id``, and ``customer_id`` overwrite whatever the
        envelope claimed; the envelope's claimed identity is discarded.

        verification is UNCONDITIONAL and fail-closed (caller guarantees ``request.context`` and
        ``context.agent_id`` present): verify; on success return the re-stamped request; on ANY
        failure return ``(request, <TOOL_IDENTITY_UNVERIFIED response>)`` so the dispatcher rejects
        the call without forwarding. there is no off/warn passthrough -- a call the proxy cannot
        authenticate never reaches the tool pod on the self-asserted envelope.

        :param request: the parsed call request (its context carries the identity token)
        :ptype request: ProxyCallRequest
        :return: ``(possibly re-stamped request, error response or None)``. a non-None response
            means the caller must reject the call without dispatching
        :rtype: tuple[ProxyCallRequest, ProxyCallResponse | None]
        """
        context = request.context
        assert context is not None  # guaranteed by the caller's agent_id presence check
        # shared across the handshake + user-assertion verifications so the reactive Hub refresh
        # (on a kid-not-in-cache miss) fires at most ONCE per dispatch, not once per token.
        refreshed = [False]
        try:
            token = context.identity_token
            if token is None:
                raise IdentityTokenError("identity token absent from call context")
            if self._jwks_provider is None:
                raise IdentityTokenError("no JWKS provider configured for identity verification")
            claims = await self._verify_token_reactively(token, refreshed=refreshed)
            # the VERIFIED handshake identity. these UUID conversions live INSIDE the try so a
            # malformed-but-signed non-UUID claim fails closed (TOOL_IDENTITY_UNVERIFIED) rather
            # than escaping as an uncaught ValueError. user_id DEFAULTS to the handshake token's:
            # ``None`` for an agent handshake token (one per pod; it CANNOT carry the per-turn
            # user), the system principal for a hub-originated call. the bound user-assertion below
            # may override it with the per-turn verified user.
            agent_id_value = UUID(claims.sub)
            customer_id_value = UUID(claims.customer_id)
            user_id_value: UUID | None = UUID(claims.user_id) if claims.user_id is not None else None
        except (IdentityTokenError, ValueError, KeyError, TypeError) as exc:
            reason = type(exc).__name__
            extra = {
                "extra_data": {
                    "tool_name": request.tool_name,
                    "reason": reason,
                    # log the exception MESSAGE too (the structural failure reason -- "no JWKS key
                    # matches the token kid" vs "token expired" vs "token absent"), so a stale-JWKS
                    # failure is distinguishable from an expired-token failure in production (the gap
                    # that masked the datasource failure). str(exc) is never token or key material.
                    "detail": str(exc),
                    "correlation_id": _correlation_id_str(request),
                }
            }
            log.warning("identity verification failed; rejecting call", extra=extra)
            response = ProxyCallResponse(
                success=False,
                content="",
                error=f"identity verification failed ({reason})",
                error_code="TOOL_IDENTITY_UNVERIFIED",
                context=context,
            )
            return request, response

        # the verified user identity DEFAULTS to the handshake token's user_id: ``None`` for an
        # agent handshake token (one per pod; it CANNOT carry the per-turn user), the system
        # principal for a hub-originated call. a user-driven turn's tool call ALSO carries a
        # Hub-minted, cnf-LESS user-assertion (``context.user_identity_token``) holding the
        # per-turn VERIFIED user_id. verify it against the SAME issuer/JWKS and BIND it to the
        # handshake token -- the assertion's ``sub`` and ``customer_id`` MUST match the handshake
        # token's -- so a user-assertion minted for agent A (customer X) cannot be replayed under
        # agent B (or customer Y); AND bind it to the conversation -- the assertion's
        # ``conversation_id`` MUST equal the inbound call's -- so a captured assertion cannot be
        # replayed into a DIFFERENT conversation. on ANY failure the call is rejected fail-closed;
        # the verified user_id then re-stamps ``context.user_id`` below, so RBAC evaluates an
        # AUTHENTICATED user.
        #
        # SECURITY (the user-assertion is cnf-LESS, because the Hub cannot know the target pod's
        # holder key at mint time -- a single per-turn token, bound to no pod -- so unlike the
        # handshake token it is NOT proof-of-possession bound). a user-assertion captured off the bus
        # is contained by three bindings: (1) connection auth (only an authenticated pod can reach
        # the tools.call subject at all); (2) the sub+customer binding below (a captured assertion is
        # usable only under its own agent+customer, never to impersonate a user to a DIFFERENT
        # agent); and (3) CONVERSATION-BINDING below (the assertion carries the conversation_id it was
        # minted for, and the call is rejected unless the inbound CallContext.conversation_id matches)
        # -- so a captured assertion cannot be replayed into a DIFFERENT conversation (acting as the
        # user where they are not, or after they have left), only into the SAME conversation it was
        # minted for, where that user legitimately is and this agent legitimately serves. a
        # generous-but-bounded TTL bounds the in-conversation window to roughly one turn.
        #
        # ``user_id_value`` was seeded from the handshake token inside the try above (so a malformed
        # claim fails closed); the bound user-assertion below may override it.
        # a present, NON-EMPTY user-assertion triggers verify + bind. an empty string is treated as
        # ABSENT (the user_id stays the handshake token's) -- a caller that builds the envelope
        # without a user-assertion must never trip a fail-closed deny on the empty value.
        user_assertion = context.user_identity_token
        if user_assertion:
            try:
                user_claims = await self._verify_token_reactively(user_assertion, refreshed=refreshed)
                if user_claims.sub != claims.sub or user_claims.customer_id != claims.customer_id:
                    raise IdentityTokenError(
                        "user-assertion not bound to the handshake identity (sub/customer mismatch)"
                    )
                if user_claims.user_id is None:
                    raise IdentityTokenError("user-assertion carries no user_id")
                # CONVERSATION-BINDING: the assertion must carry the conversation_id it was minted
                # for, and it must equal this call's. a user-driven turn ALWAYS mints with a
                # conversation_id, so an assertion lacking one is a denial -- never a check the
                # caller can skip by omitting it. a mismatch (or a call with no conversation_id at
                # all, when the assertion carries one) is the cross-conversation replay this gate
                # closes. ``context.conversation_id`` is a UUID; stringify to compare against the
                # wire-string claim.
                if user_claims.conversation_id is None:
                    raise IdentityTokenError("user-assertion carries no conversation_id")
                if context.conversation_id is None or str(context.conversation_id) != user_claims.conversation_id:
                    raise IdentityTokenError(
                        "user-assertion conversation_id does not match the call (cross-conversation replay)"
                    )
                user_id_value = UUID(user_claims.user_id)
            except (IdentityTokenError, ValueError, KeyError, TypeError) as exc:
                reason = type(exc).__name__
                extra = {
                    "extra_data": {
                        "tool_name": request.tool_name,
                        "reason": reason,
                        # the structural failure reason (binding mismatch vs cross-conversation
                        # replay vs expired/absent assertion), never token or key material.
                        "detail": str(exc),
                        "correlation_id": _correlation_id_str(request),
                    }
                }
                log.warning("user-assertion verification failed; rejecting call", extra=extra)
                response = ProxyCallResponse(
                    success=False,
                    content="",
                    error=f"user-assertion verification failed ({reason})",
                    error_code="TOOL_USER_IDENTITY_UNVERIFIED",
                    context=context,
                )
                return request, response

        verified_context = context.model_copy(
            update={
                "agent_id": agent_id_value,
                "user_id": user_id_value,
                "customer_id": customer_id_value,
            }
        )
        return request.model_copy(update={"context": verified_context}), None

    async def _verify_pop(self, request: "ProxyCallRequest") -> "ProxyCallResponse | None":
        """verify the per-call proof-of-possession against the token's holder-key binding.

        Self-contained: re-verifies the identity token to obtain a TRUSTED ``cnf`` thumbprint, then
        checks the request's pop proves possession of that key for THIS token (``ath``) + THIS call
        body (``bh``) + is fresh + single-use (the ``jti`` is recorded in the replay guard). So a
        leaked token alone -- without the holder private key -- cannot be replayed.

        verification is UNCONDITIONAL and fail-closed: on ANY failure (absent/invalid token, no
        ``cnf`` holder binding, absent/invalid pop, spliced body, or a nonce the replay guard has
        already seen) the call is rejected with a TOOL_POP_UNVERIFIED response. the replay guard is
        always present (required at construction), so a captured pop can never be replayed verbatim.

        :param request: the identity-verified call request (its context carries the token + pop)
        :ptype request: ProxyCallRequest
        :return: an error response when the call must be rejected, else ``None``
        :rtype: ProxyCallResponse | None
        """
        context = request.context
        assert context is not None  # guaranteed by the caller's agent_id presence check
        try:
            token = context.identity_token
            if token is None:
                raise IdentityTokenError("identity token absent; cannot verify pop")
            if self._jwks_provider is None:
                raise IdentityTokenError("no JWKS provider configured for pop verification")
            claims = verify_identity_token(
                token,
                jwks=self._load_jwks(),
                issuer=_IDENTITY_ISSUER,
                leeway_seconds=_IDENTITY_LEEWAY_SECONDS,
            )
            if claims.cnf is None:
                raise IdentityTokenError("identity token carries no cnf holder binding")
            if request.pop is None:
                raise IdentityTokenError("pop proof absent from request")
            body_hash = canonical_call_hash(
                request.tool_name,
                request.arguments,
                str(context.correlation_id) if context.correlation_id is not None else None,
            )
            jti = verify_pop_proof(
                request.pop,
                expected_jkt=claims.cnf,
                access_token_hash=access_token_hash(token),
                body_hash=body_hash,
                leeway_seconds=_POP_LEEWAY_SECONDS,
            )
            if not await self._pop_replay_guard.record_unique(jti):
                raise IdentityTokenError("pop nonce replay")
            return None
        except (IdentityTokenError, ValueError, KeyError, TypeError) as exc:
            reason = type(exc).__name__
            extra = {
                "extra_data": {
                    "tool_name": request.tool_name,
                    "reason": reason,
                    # the structural pop-failure reason (absent token/pop, no cnf binding, spliced
                    # body, replayed nonce), never token or key material.
                    "detail": str(exc),
                    "correlation_id": _correlation_id_str(request),
                }
            }
            log.warning("pop verification failed; rejecting call", extra=extra)
            return ProxyCallResponse(
                success=False,
                content="",
                error=f"pop verification failed ({reason})",
                error_code="TOOL_POP_UNVERIFIED",
                context=context,
            )

    async def _dispatch_call(
        self,
        request: "ProxyCallRequest",
        msg: IncomingMessage,
    ) -> None:
        """body of :meth:`_process_call` after the logging-context bind.

        kept separate so the ``try``/``finally`` wrapping the
        :func:`bind_log_context` / :func:`clear_context` pair stays
        shallow; the operational flow lives here untouched. the gate
        order is verify-identity -> verify-pop -> **limit-guard** ->
        authorizer -> catalog -> route: the spend gate sits immediately
        after pop and immediately before the authorizer so a
        spend-denied call never consumes a catalog lookup, while an
        unauthorized-for-the-tool call still gets ``TOOL_NOT_AUTHORIZED``
        rather than a spend error. the limit gate is the ONE fail-OPEN
        gate (a guard that raises serves the call); every other gate is
        fail-CLOSED. after a successful forward the post-call
        usage-emit seam fires fire-and-forget.

        :param request: parsed + identity-bound call request
        :ptype request: ProxyCallRequest
        :param msg: incoming wrapper envelope (for reply subject)
        :ptype msg: IncomingMessage
        :return: nothing; response is published to ``msg.reply_subject``
            by each branch below
        :rtype: None
        """
        assert self._nc is not None
        if request.context is None or request.context.agent_id is None:
            response = ProxyCallResponse(
                success=False,
                content="",
                error=("ProxyCallRequest received without context.agent_id; cannot route"),
                error_code="MALFORMED_REQUEST",
                context=request.context,
            )
            if msg.reply_subject is not None:
                await self._nc.publish_reply(
                    reply_subject=msg.reply_subject,
                    message=response,
                )
            log.warning(
                "proxy call missing agent_id in context",
                extra={
                    "extra_data": {
                        "tool_name": request.tool_name,
                        "correlation_id": _correlation_id_str(request),
                    }
                },
            )
            return

        # verify the Hub-issued identity token and re-stamp the VERIFIED identity onto the
        # request BEFORE authorization + forwarding, so RBAC and the tool pod act on an
        # authenticated identity rather than the self-asserted envelope. unconditional + fail-closed.
        verified_request, identity_error = await self._verify_identity(request)
        if identity_error is not None:
            if msg.reply_subject is not None:
                await self._nc.publish_reply(
                    reply_subject=msg.reply_subject,
                    message=identity_error,
                )
            return
        if verified_request is not request:
            request = verified_request
            bind_log_context(request.context)  # refresh log tags with the verified identity
        assert request.context is not None  # held by the agent_id check; re-narrow after re-stamp

        # verify the per-call proof-of-possession: the caller must prove it holds the key the token
        # is bound to (cnf), for THIS token + THIS body, once. self-contained (re-verifies the token
        # for a trusted cnf). unconditional + fail-closed, same as identity verification above.
        pop_error = await self._verify_pop(request)
        if pop_error is not None:
            if msg.reply_subject is not None:
                await self._nc.publish_reply(
                    reply_subject=msg.reply_subject,
                    message=pop_error,
                )
            return

        # log-border stringification of identity dimensions; the
        # ProxyCallResponse echoes the whole context so these string
        # forms are for log records only. user_id rides on the same
        # CallContext envelope (context-task-01) and is plumbed to
        # the authorizer so rbac-evaluator implementations can
        # resolve user-side grants; ``None`` when the dispatch
        # carries no user identity (authorizer will deny).
        correlation_id_log = _correlation_id_str(request)
        agent_id_log = str(request.context.agent_id)
        user_id_log: str | None = str(request.context.user_id) if request.context.user_id is not None else None
        customer_id_log: str | None = (
            str(request.context.customer_id) if request.context.customer_id is not None else None
        )

        # pre-call spend gate (gu-task-06): AFTER pop / BEFORE the authorizer + catalog routing, so a
        # spend-denied call never consumes a catalog lookup. FAIL-OPEN (Fork-2): a guard that RAISES
        # or is unreachable SERVES the call (loud WARNING) -- a billing-infra outage must not brick
        # tool traffic. this inverts the fail-CLOSED identity/pop/authorizer gates ON PURPOSE. only a
        # returned LimitDecision(allowed=False) hard-denies.
        try:
            limit_decision = await self._limit_guard.check(
                agent_id_log,
                user_id_log,
                customer_id_log,
                request.tool_name,
                request.tool_version,
            )
        except Exception:  # noqa: BLE001 -- fail-open per Fork-2: a guard outage must never deny
            log.warning(
                "limit guard unreachable; serving fail-open",
                extra={
                    "extra_data": {
                        "agent_id": agent_id_log,
                        "customer_id": customer_id_log,
                        "tool_name": request.tool_name,
                        "correlation_id": correlation_id_log,
                    }
                },
            )
        else:
            if not limit_decision.allowed:
                response = ProxyCallResponse(
                    success=False,
                    content="",
                    error=f"tool call denied by spend limit ({limit_decision.error_code})",
                    error_code=limit_decision.error_code,
                    context=request.context,
                )
                if msg.reply_subject is not None:
                    await self._nc.publish_reply(
                        reply_subject=msg.reply_subject,
                        message=response,
                    )
                log.warning(
                    "tool call denied by limit guard",
                    extra={
                        "extra_data": {
                            "agent_id": agent_id_log,
                            "customer_id": customer_id_log,
                            "tool_name": request.tool_name,
                            "error_code": limit_decision.error_code,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )
                return

        if self._authorizer is not None:
            authorized = await self._authorizer.is_authorized(
                agent_id_log,
                user_id_log,
                request.tool_name,
                request.tool_version,
            )
            if not authorized:
                response = ProxyCallResponse(
                    success=False,
                    content="",
                    error=f"agent not authorized for tool {request.tool_name}",
                    error_code="TOOL_NOT_AUTHORIZED",
                    context=request.context,
                )
                if msg.reply_subject is not None:
                    await self._nc.publish_reply(
                        reply_subject=msg.reply_subject,
                        message=response,
                    )
                log.warning(
                    "agent tool call denied",
                    extra={
                        "extra_data": {
                            "agent_id": agent_id_log,
                            "user_id": user_id_log,
                            "tool_name": request.tool_name,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )
                return

        full_name = f"{request.tool_name}@{request.tool_version}"
        entry = self._catalog.get(full_name)

        if entry is None:
            response = ProxyCallResponse(
                success=False,
                content="",
                error=f"tool {full_name} is not available",
                error_code="TOOL_UNAVAILABLE",
                context=request.context,
            )
            if msg.reply_subject is not None:
                await self._nc.publish_reply(
                    reply_subject=msg.reply_subject,
                    message=response,
                )
            log.warning(
                "tool not found for call",
                extra={
                    "extra_data": {
                        "full_name": full_name,
                        "agent_id": agent_id_log,
                        "correlation_id": correlation_id_log,
                    }
                },
            )
            return

        endpoint = self._routing_strategy.select(entry.endpoints)

        if endpoint is None:
            # TOOL_NOT_READY takes priority over TOOL_UNAVAILABLE: if ANY
            # endpoint is still pending its probe confirmation, the caller
            # should retry shortly rather than give up. TOOL_UNAVAILABLE is
            # only reported when no pending endpoints exist either.
            has_pending = any(ep.status == "pending" for ep in entry.endpoints)
            if has_pending:
                response = ProxyCallResponse(
                    success=False,
                    content="",
                    error=(f"tool {full_name} endpoints have not yet confirmed reachability"),
                    error_code="TOOL_NOT_READY",
                    context=request.context,
                )
                if msg.reply_subject is not None:
                    await self._nc.publish_reply(
                        reply_subject=msg.reply_subject,
                        message=response,
                    )
                log.warning(
                    "tool endpoints still pending probe confirmation",
                    extra={
                        "extra_data": {
                            "full_name": full_name,
                            "endpoint_count": len(entry.endpoints),
                            "agent_id": agent_id_log,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )
                return
            response = ProxyCallResponse(
                success=False,
                content="",
                error=f"tool {full_name} has no available endpoints",
                error_code="TOOL_UNAVAILABLE",
                context=request.context,
            )
            if msg.reply_subject is not None:
                await self._nc.publish_reply(
                    reply_subject=msg.reply_subject,
                    message=response,
                )
            log.warning(
                "no available endpoints for call",
                extra={
                    "extra_data": {
                        "full_name": full_name,
                        "endpoint_count": len(entry.endpoints),
                        "agent_id": agent_id_log,
                        "correlation_id": correlation_id_log,
                    }
                },
            )
            return

        # in_flight is read by routing strategies during endpoint selection
        # and incremented/decremented here. the +=/-= pair is safe under
        # asyncio (no preemption between the read and the store within a
        # single bytecode op) but would race under threaded execution. if
        # this proxy is ever moved off a single event loop, wrap these
        # ops in an asyncio.Lock or swap to a threadsafe counter.
        endpoint.in_flight += 1
        try:
            response = await self._forward_call(request, endpoint.pod_id)
        finally:
            endpoint.in_flight -= 1
        if msg.reply_subject is not None:
            await self._nc.publish_reply(
                reply_subject=msg.reply_subject,
                message=response,
            )

        # post-call usage-emit seam (gu-task-16): this is the one place both the inbound request
        # arguments and the outbound response content are local. the reply is already published, so
        # a fire-and-forget emit can never delay or break it; an emit failure is caught + logged and
        # NEVER affects the reply. the hub injects its concrete SDK-typed emitter into this slot.
        if self._usage_emitter is not None:
            try:
                await self._usage_emitter.emit(request, response)
            except Exception:  # noqa: BLE001 -- fire-and-forget: a usage-emit failure never affects the reply
                log.warning(
                    "endpoint usage emit failed",
                    extra={
                        "extra_data": {
                            "agent_id": agent_id_log,
                            "tool_name": request.tool_name,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )

    def _resolve_timeout(self, tool_name: str, tool_version: str) -> float:
        """resolve effective timeout for a tool call.

        checks catalog entry for per-tool declared timeout, falls back
        to proxy default (from env var or platform default).

        :param tool_name: namespaced tool name
        :ptype tool_name: str
        :param tool_version: tool version string
        :ptype tool_version: str
        :return: effective timeout in seconds
        :rtype: float
        """
        full_name = f"{tool_name}@{tool_version}"
        entry = self._catalog.get(full_name)
        if entry is not None and entry.timeout_seconds is not None:
            result: float = entry.timeout_seconds
            return result
        return self.timeout

    async def _forward_call(
        self,
        request: ProxyCallRequest,
        pod_id: str,
    ) -> ProxyCallResponse:
        """forward tool call to target tool pod via NATS request-reply.

        uses per-tool timeout from catalog if declared, otherwise
        falls back to proxy default.

        :param request: original call request from agent
        :ptype request: ProxyCallRequest
        :param pod_id: identifier of target tool pod
        :ptype pod_id: str
        :return: response from tool pod or error response on timeout
        :rtype: ProxyCallResponse
        :raises RuntimeError: when invoked before ``start`` connects NATS
        """
        if self._nc is None:
            raise RuntimeError("_forward_call invoked before NATS connected")
        internal_subject = Subjects.tools_internal(pod_id)
        internal_payload = _build_internal_payload(request, self._mint_proxy_assertion(request, pod_id))
        effective_timeout = self._resolve_timeout(request.tool_name, request.tool_version)
        correlation_id_log = _correlation_id_str(request)

        try:
            reply_bytes = await self._nc.request_raw(
                subject=internal_subject,
                payload=internal_payload,
                timeout=timedelta(seconds=effective_timeout),
            )
            response = ProxyCallResponse.model_validate_json(reply_bytes)
        except (TimeoutError, RequestError) as exc:
            # the wrapper raises RequestError ("timed out" / "no responders" /
            # "connection closed") for transport-level failures; we coalesce
            # the timeout case (which the catalog mapping cares about) and
            # surface anything else as TOOL_UNAVAILABLE so the agent gets a
            # well-typed response rather than a bare TOOL_TIMEOUT for a
            # connectivity blip.
            if isinstance(exc, RequestError) and "timed out" not in str(exc):
                error_code = "TOOL_UNAVAILABLE"
                error_msg = f"tool call transport failure after {effective_timeout}s: {exc}"
            else:
                error_code = "TOOL_TIMEOUT"
                error_msg = f"tool call timed out after {effective_timeout}s"
            log.warning(
                "tool call failed in transport",
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "tool_name": request.tool_name,
                        "correlation_id": correlation_id_log,
                        "timeout": effective_timeout,
                        "error_code": error_code,
                    }
                },
            )
            response = ProxyCallResponse(
                success=False,
                content="",
                error=error_msg,
                error_code=error_code,
                context=request.context,
            )
        return response

    def _mint_proxy_assertion(self, request: ProxyCallRequest, pod_id: str) -> str | None:
        """sign a proxy->pod assertion for a forwarded call, or ``None`` when unsignable.

        Binds the VERIFIED caller identity (already re-stamped onto the context by
        :meth:`_verify_identity`) + the call body + a single-use nonce + the target pod, so the pod
        can verify the call came from THIS proxy, for THIS body, once. Returns ``None`` when no
        signer is configured (the binding is inert) or the verified identity is incomplete.

        :param request: the forwarded call request (its context carries the verified identity)
        :ptype request: ProxyCallRequest
        :param pod_id: the target pod id (the assertion ``aud``)
        :ptype pod_id: str
        :return: a compact JWS assertion, or ``None``
        :rtype: str | None
        """
        context = request.context
        result: str | None = None
        if (
            self._proxy_signer is not None
            and context is not None
            and context.agent_id is not None
            and context.customer_id is not None
        ):
            body_hash = canonical_call_hash(
                request.tool_name,
                request.arguments,
                str(context.correlation_id) if context.correlation_id is not None else None,
            )
            result = self._proxy_signer.mint(
                pod_id=pod_id,
                agent_id=str(context.agent_id),
                customer_id=str(context.customer_id),
                body_hash=body_hash,
                nonce=str(uuid7()),
                now=int(datetime.now(UTC).timestamp()),
                user_id=str(context.user_id) if context.user_id is not None else None,
            )
        return result


def _correlation_id_str(request: ProxyCallRequest) -> str:
    """stringify the correlation id riding on ``request.context``.

    the wire-level correlation id lives on
    :attr:`CallContext.correlation_id`. log records carry it as a
    string for human consumption; :class:`ProxyCallResponse` itself
    echoes the whole :class:`CallContext` so the response shape stays
    identical to the request. this helper centralizes the log-border
    conversion: returns ``str(request.context.correlation_id)`` when
    present, else the empty string.

    :param request: parsed proxy call request
    :ptype request: ProxyCallRequest
    :return: stringified correlation id or ``""`` when absent
    :rtype: str
    """
    result = ""
    if request.context is not None and request.context.correlation_id is not None:
        result = str(request.context.correlation_id)
    return result


def _build_internal_payload(request: ProxyCallRequest, proxy_assertion: str | None = None) -> bytes:
    """build internal NATS payload for forwarding to tool pod.

    constructs :class:`CallRequest` from the proxy request, copying
    ``context`` through verbatim so identity dimensions (including
    ``correlation_id`` which now lives exclusively on
    :class:`CallContext`) survive the hop from registry to tool pod.
    ``proxy_assertion`` is the proxy's body-bound signature for the pod
    to verify, or ``None`` when the binding is inert.

    :param request: original proxy call request
    :ptype request: ProxyCallRequest
    :param proxy_assertion: the proxy->pod assertion JWS, or ``None``
    :ptype proxy_assertion: str | None
    :return: serialized internal call request bytes
    :rtype: bytes
    """
    from threetears.agent.tools.server import CallRequest

    internal_request = CallRequest(
        tool_name=request.tool_name,
        tool_version=request.tool_version,
        arguments=request.arguments,
        context=request.context,
        proxy_assertion=proxy_assertion,
    )
    result = internal_request.model_dump_json().encode("utf-8")
    return result
