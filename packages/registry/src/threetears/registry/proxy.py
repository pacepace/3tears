"""call proxy for routing tool calls to tool pods.

subscribes to NATS call subject, validates tool availability
in catalog, selects endpoint via pluggable routing strategy,
tracks in-flight calls, and forwards to tool pod via
NATS request-reply.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from threetears.agent.tools.context_envelope import CallContext, bind_log_context
from threetears.core.security.identity_token import (
    IdentityClaims,
    IdentityKeyNotFoundError,
    IdentityTokenError,
    canonical_call_hash,
    verify_identity_token,
)
from threetears.core.security.pop import access_token_hash, verify_pop_proof
from threetears.nats import (
    RESULT_ACK_TIMEOUT_SECONDS,
    IncomingMessage,
    RequestError,
    Subject,
    Subjects,
    reply_subject_is_owned_by_agent,
    reply_subject_prefix_for_agent,
    requires_async_result,
    result_stream_name,
)
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

# how many times a durable result publish to the caller is retried before the answer is declared
# lost. by that point the tool has already run, so a transport blip must not cost the work; the
# caller has a deadline, so the retrying cannot be unbounded either. mirrors the pod-side pair.
_RESULT_DELIVERY_ATTEMPTS = 3
_RESULT_DELIVERY_RETRY_SECONDS = 2.0

if TYPE_CHECKING:
    from threetears.agent.tools.server import CallAccepted
    from threetears.core.coordination.replay_guard import ReplayGuard
    from threetears.core.security import ProxyAssertionSigner
    from threetears.nats import NatsClient, Subscription

__all__ = [
    "CallProxy",
    "ProxyCallAccepted",
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
    :param result_subject: where to DELIVER the answer, for a call the
        agent has decided is too long to answer on the reply inbox.
        ``allow_responses`` belongs to the connection that received the
        request, and the credential refresh that keeps the registry
        authenticated is a reconnect, so a call spanning one connection
        lifetime cannot be answered on the inbox at all. when set, the
        proxy acknowledges immediately and publishes the
        :class:`ProxyCallResponse` here instead. it must name the
        caller's VERIFIED agent id -- the proxy refuses any other, which
        is what keeps its two-token wildcard publish grant from being a
        way to redirect one agent's result onto another's in-flight call
    :ptype result_subject: str | None
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_version: str
    arguments: dict[str, Any]
    context: CallContext | None = None
    pop: str | None = None
    result_subject: str | None = None

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


class ProxyCallAccepted(BaseModel):
    """the immediate answer to a call the proxy will DELIVER rather than reply to.

    Mirrors the pod-side :class:`threetears.agent.tools.server.CallAccepted` one hop up. Published on
    the agent's reply inbox before routing begins, while that publish is still guaranteed to work,
    so the agent can distinguish "the registry has your long call" from "no registry answered" -- a
    distinction its own retry and error handling depends on and that a silent long wait destroys.

    :param accepted: whether the proxy took the call and will publish to the delivery subject
    :ptype accepted: bool
    :param result_subject: the subject the answer will be published to; echoed so the caller can
        assert it matches the one it is waiting on
    :ptype result_subject: str | None
    :param error: why the call was refused, when ``accepted`` is false
    :ptype error: str | None
    """

    accepted: bool
    result_subject: str | None = None
    error: str | None = None


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

    @property
    def subscription_active(self) -> bool:
        """whether the tool-call subscription is currently bound.

        readiness signal: a registry replica not subscribed to
        ``{ns}.tools.call`` routes nothing, so it must leave rotation. a
        restart would not fix an underlying NATS outage, so this is readiness
        rather than liveness.

        ``True`` between a successful :meth:`start` and :meth:`stop`.

        :return: true while the tool-call subject is subscribed
        :rtype: bool
        """
        return self._sub is not None

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

        # Where this call's answer goes. Resolved HERE -- after identity verification, before any
        # routing -- because the check that makes it safe is against the VERIFIED agent id that
        # ``_verify_identity`` just re-stamped, never the one the envelope claimed. The registry holds
        # a two-token wildcard publish grant on the reply family (one connection fronts every agent,
        # so there is no per-connection list of literals to mint from); this refusal is what stops
        # that wildcard from being a way to redirect one agent's result onto a peer's in-flight call.
        delivery_subject: Subject | None = None
        if request.result_subject is not None:
            if not reply_subject_is_owned_by_agent(request.result_subject, agent_id=agent_id_log):
                expected = reply_subject_prefix_for_agent(agent_id_log)
                log.warning(
                    "rejecting a delivery subject that does not name the verified caller",
                    extra={
                        "extra_data": {
                            "requested_subject": request.result_subject,
                            "expected_prefix": expected,
                            "agent_id": agent_id_log,
                            "tool_name": request.tool_name,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )
                if msg.reply_subject is not None:
                    await self._nc.publish_reply(
                        reply_subject=msg.reply_subject,
                        message=ProxyCallAccepted(
                            accepted=False,
                            result_subject=request.result_subject,
                            error=(
                                f"result subject {request.result_subject!r} does not belong to the "
                                f"verified caller; expected one token under {expected!r}"
                            ),
                        ),
                    )
                return
            delivery_subject = Subject.raw(request.result_subject)
            if msg.reply_subject is not None:
                await self._nc.publish_reply(
                    reply_subject=msg.reply_subject,
                    message=ProxyCallAccepted(accepted=True, result_subject=delivery_subject.path),
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
                await self._answer(msg, response, delivery_subject)
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
            await self._answer(msg, response, delivery_subject)
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
                await self._answer(msg, response, delivery_subject)
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
            await self._answer(msg, response, delivery_subject)
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

        # failover loop: forward to the selected endpoint, and on a
        # DEAD-POD transport failure (TOOL_UNAVAILABLE -- the wrapper saw
        # "no responders" / "connection closed", so the call never reached
        # a pod) retry against another available endpoint the routing
        # strategy has not yet handed us. this survives the window between a
        # pod dying and the heartbeat sweep evicting its catalog endpoints:
        # a single call to a not-yet-evicted dead pod no longer fails the
        # whole request when a healthy sibling pod serves the same tool.
        #
        # only TOOL_UNAVAILABLE is retried. a TOOL_TIMEOUT may have reached
        # the pod and be executing, so retrying it would risk double-execution
        # of a non-idempotent tool; a tool-level error (success=False from the
        # pod) or a success is the pod's authoritative answer. all three
        # short-circuit the loop.
        #
        # in_flight is read by routing strategies during endpoint selection
        # and incremented/decremented here. the +=/-= pair is safe under
        # asyncio (no preemption between the read and the store within a
        # single bytecode op) but would race under threaded execution. if
        # this proxy is ever moved off a single event loop, wrap these
        # ops in an asyncio.Lock or swap to a threadsafe counter.
        attempted_pod_ids: set[str] = set()
        while True:
            attempted_pod_ids.add(endpoint.pod_id)
            endpoint.in_flight += 1
            try:
                response = await self._forward_call(request, endpoint.pod_id)
            finally:
                endpoint.in_flight -= 1
            if response.error_code != "TOOL_UNAVAILABLE":
                break
            remaining = [ep for ep in entry.endpoints if ep.pod_id not in attempted_pod_ids]
            next_endpoint = self._routing_strategy.select(remaining)
            if next_endpoint is None:
                # NOSILENT: the failover warning below fires only when a SIBLING pod exists, so
                # the single-endpoint case -- the common one, and the total outage -- was the one
                # shape that exited here saying nothing. The call never reached a pod, yet the
                # only account of it was a TOOL_UNAVAILABLE the caller then had to explain. Say
                # which pod did not answer, so "the call never arrived" is distinguishable from
                # "the pod ran it and refused" without correlating two services by timestamp.
                log.warning(
                    "tool endpoint unreachable and no other endpoint to fail over to; the call never reached a pod",
                    extra={
                        "extra_data": {
                            "full_name": full_name,
                            "failed_pod_id": endpoint.pod_id,
                            "endpoint_count": len(entry.endpoints),
                            "agent_id": agent_id_log,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )
                break
            log.warning(
                "tool endpoint unreachable; failing over to another pod",
                extra={
                    "extra_data": {
                        "full_name": full_name,
                        "failed_pod_id": endpoint.pod_id,
                        "failover_pod_id": next_endpoint.pod_id,
                        "agent_id": agent_id_log,
                        "correlation_id": correlation_id_log,
                    }
                },
            )
            endpoint = next_endpoint
        await self._answer(msg, response, delivery_subject)

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

    async def _answer(
        self,
        msg: IncomingMessage,
        response: ProxyCallResponse,
        delivery_subject: "Subject | None",
    ) -> None:
        """route one dispatch's answer to wherever this call agreed it would go.

        one function for every branch -- limit denial, authorization denial, unknown tool, transport
        failure, success -- so a call acknowledged for delivery can never have its answer published
        on the inbox instead. that mistake does not fail loudly: the caller simply waits out its
        whole timeout on a subject nothing ever publishes to.

        :param msg: inbound wrapper envelope (carries the reply inbox)
        :ptype msg: IncomingMessage
        :param response: the answer to send
        :ptype response: ProxyCallResponse
        :param delivery_subject: the durable subject this call was accepted for, or ``None`` for the
            synchronous reply-inbox path
        :ptype delivery_subject: Subject | None
        :return: nothing
        :rtype: None
        """
        assert self._nc is not None
        if delivery_subject is None:
            if msg.reply_subject is not None:
                await self._nc.publish_reply(reply_subject=msg.reply_subject, message=response)
            return
        payload = response.model_dump_json().encode("utf-8")
        for attempt in range(1, _RESULT_DELIVERY_ATTEMPTS + 1):
            try:
                await self._nc.jetstream_publish(subject=delivery_subject, payload=payload)
                return
            except Exception as exc:  # noqa: BLE001 — the tool already ran; retry beats discarding it
                if attempt == _RESULT_DELIVERY_ATTEMPTS:
                    # NOSILENT: the answer is now lost. name it, rather than letting it surface as an
                    # unexplained timeout at the agent.
                    log.error(
                        "tool result could not be delivered to the caller after %d attempts; the "
                        "answer is lost (subject=%s bytes=%d): %s",
                        _RESULT_DELIVERY_ATTEMPTS,
                        delivery_subject.path,
                        len(payload),
                        exc,
                    )
                    return
                log.warning(
                    "tool result delivery to caller failed (attempt %d/%d, subject=%s); retrying: %s",
                    attempt,
                    _RESULT_DELIVERY_ATTEMPTS,
                    delivery_subject.path,
                    exc,
                )
                await asyncio.sleep(_RESULT_DELIVERY_RETRY_SECONDS)

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
        """forward tool call to target tool pod, on whichever path its declared timeout allows.

        uses per-tool timeout from catalog if declared, otherwise
        falls back to proxy default.

        A tool whose timeout exceeds :data:`threetears.nats.SYNC_REPLY_BUDGET_SECONDS` cannot be
        answered on the reply inbox: the pod's right to publish there dies with the connection that
        received the call, and that connection is rebuilt every time its credential is refreshed. Such
        calls go through :meth:`_forward_call_durable` instead. Everything shorter keeps the fast
        request/reply path unchanged.

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
        effective_timeout = self._resolve_timeout(request.tool_name, request.tool_version)
        if requires_async_result(effective_timeout):
            return await self._forward_call_durable(request, pod_id, effective_timeout)
        internal_subject = Subjects.tools_internal(pod_id)
        internal_payload = _build_internal_payload(request, self._mint_proxy_assertion(request, pod_id))
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

    async def _forward_call_durable(
        self,
        request: ProxyCallRequest,
        pod_id: str,
        effective_timeout: float,
    ) -> ProxyCallResponse:
        """forward a LONG call: subscribe, dispatch, and collect the answer off a durable subject.

        The order is the contract. The waiter is opened BEFORE the call is dispatched, so there is no
        window in which the pod could answer with nothing listening. The dispatch itself is still
        request/reply, but what comes back is only an ACCEPT -- published before the tool starts, so
        it is always inside the window the pod's connection is guaranteed to survive. That accept is
        what preserves the dead-pod signal the failover loop above depends on: without it, an endpoint
        that had vanished would be indistinguishable from one running a 20-minute scan, and the
        failover could not fire until the entire tool budget had elapsed.

        The answer then arrives on ``{ns}.tools.result.{pod_id}.{call_id}``. ``call_id`` is minted
        per CALL, never the per-turn ``correlation_id``: a turn can dispatch several tool calls, and a
        correlation-keyed subject would hand this waiter another call's result.

        :param request: original call request from agent
        :ptype request: ProxyCallRequest
        :param pod_id: identifier of target tool pod
        :ptype pod_id: str
        :param effective_timeout: how long the tool is allowed to take
        :ptype effective_timeout: float
        :return: the pod's answer, or a well-typed error response
        :rtype: ProxyCallResponse
        """
        assert self._nc is not None  # guarded by the caller
        correlation_id_log = _correlation_id_str(request)
        result_subject = Subjects.tools_result(pod_id, uuid7())
        internal_payload = _build_internal_payload(
            request,
            self._mint_proxy_assertion(request, pod_id),
            result_subject=result_subject.path,
        )
        waiter = await self._nc.jetstream_result_waiter(
            subject=result_subject,
            stream=result_stream_name(),
            wait_budget=timedelta(seconds=effective_timeout),
        )
        try:
            accept_error = await self._await_pod_accept(
                pod_id=pod_id,
                internal_payload=internal_payload,
                result_subject=result_subject,
                request=request,
                correlation_id_log=correlation_id_log,
            )
            if accept_error is not None:
                return accept_error
            try:
                delivered = await waiter.wait(timeout=timedelta(seconds=effective_timeout))
            except Exception as exc:  # noqa: BLE001 — every failure becomes a typed response, never a hang
                log.warning(
                    "tool result never arrived on its durable subject",
                    extra={
                        "extra_data": {
                            "pod_id": pod_id,
                            "tool_name": request.tool_name,
                            "correlation_id": correlation_id_log,
                            "subject": result_subject.path,
                            "timeout": effective_timeout,
                            "detail": str(exc),
                        }
                    },
                )
                # TOOL_TIMEOUT, not TOOL_UNAVAILABLE: the pod ACCEPTED this call, so it may well be
                # running it. Retrying elsewhere would risk executing a non-idempotent tool twice,
                # which is the same reason the synchronous path never retries a timeout either.
                return ProxyCallResponse(
                    success=False,
                    content="",
                    error=f"tool call timed out after {effective_timeout}s awaiting durable result",
                    error_code="TOOL_TIMEOUT",
                    context=request.context,
                )
        finally:
            await waiter.close()
        return ProxyCallResponse.model_validate_json(delivered)

    async def _await_pod_accept(
        self,
        *,
        pod_id: str,
        internal_payload: bytes,
        result_subject: "Subject",
        request: ProxyCallRequest,
        correlation_id_log: str,
    ) -> ProxyCallResponse | None:
        """dispatch the call and confirm the pod took it; ``None`` means it did.

        A pod may answer this request/reply with something other than an accept, and both cases are
        real. A malformed-request rejection comes back as a full :class:`ProxyCallResponse`-shaped
        body, because the pod could not parse far enough to learn where to deliver. A refusal comes
        back as ``accepted=False`` when the delivery subject was not the pod's own to publish. Either
        way the caller must be told now rather than waiting out the tool's whole budget for an answer
        that will never be published.

        :param pod_id: identifier of target tool pod
        :ptype pod_id: str
        :param internal_payload: serialized :class:`CallRequest` bytes
        :ptype internal_payload: bytes
        :param result_subject: the subject the pod was asked to deliver on
        :ptype result_subject: Subject
        :param request: original call request from agent (for context echoing)
        :ptype request: ProxyCallRequest
        :param correlation_id_log: stringified correlation id for log records
        :ptype correlation_id_log: str
        :return: an error response to return to the caller, or ``None`` when the pod accepted
        :rtype: ProxyCallResponse | None
        """
        assert self._nc is not None  # guarded by the caller
        try:
            accept_bytes = await self._nc.request_raw(
                subject=Subjects.tools_internal(pod_id),
                payload=internal_payload,
                timeout=timedelta(seconds=RESULT_ACK_TIMEOUT_SECONDS),
            )
        except (TimeoutError, RequestError) as exc:
            # TOOL_UNAVAILABLE for BOTH shapes here, unlike the synchronous path. There the timeout
            # could mean "the pod is running your tool", so retrying elsewhere risked a double
            # execution. Here the pod had only to acknowledge before starting any work, so a silent
            # accept window means it is not there -- and failing over to a sibling endpoint is right.
            log.warning(
                "tool pod did not acknowledge a durably-delivered call",
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "tool_name": request.tool_name,
                        "correlation_id": correlation_id_log,
                        "accept_timeout": RESULT_ACK_TIMEOUT_SECONDS,
                        "detail": str(exc),
                    }
                },
            )
            return ProxyCallResponse(
                success=False,
                content="",
                error=f"tool pod did not accept the call within {RESULT_ACK_TIMEOUT_SECONDS}s: {exc}",
                error_code="TOOL_UNAVAILABLE",
                context=request.context,
            )
        accept = _parse_pod_accept(accept_bytes)
        if accept is None:
            # not an accept envelope: the pod answered the call outright (it could not parse the
            # request far enough to learn where to deliver). that IS the answer; pass it through.
            return ProxyCallResponse.model_validate_json(accept_bytes)
        if not accept.accepted:
            log.error(
                "tool pod refused the delivery subject it was given",
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "tool_name": request.tool_name,
                        "correlation_id": correlation_id_log,
                        "subject": result_subject.path,
                        "reason": accept.error,
                    }
                },
            )
            # NOT retried against a sibling endpoint. A refusal means this proxy and that pod
            # disagree about the pod's own identity, which is a wiring fault rather than a dead
            # endpoint; failing over would paper over it until it applied to every pod at once.
            return ProxyCallResponse(
                success=False,
                content="",
                error=f"tool pod refused the result subject: {accept.error}",
                error_code="TOOL_RESULT_SUBJECT_REFUSED",
                context=request.context,
            )
        return None

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


def _build_internal_payload(
    request: ProxyCallRequest,
    proxy_assertion: str | None = None,
    *,
    result_subject: str | None = None,
) -> bytes:
    """build internal NATS payload for forwarding to tool pod.

    constructs :class:`CallRequest` from the proxy request, copying
    ``context`` through verbatim so identity dimensions (including
    ``correlation_id`` which now lives exclusively on
    :class:`CallContext`) survive the hop from registry to tool pod.
    ``proxy_assertion`` is the proxy's body-bound signature for the pod
    to verify, or ``None`` when the binding is inert.

    The agent's OWN ``result_subject`` is deliberately not forwarded: the two hops carry independent
    delivery subjects, each naming its own responder, and passing the agent's through would ask the
    pod to publish somewhere it holds no grant.

    :param request: original proxy call request
    :ptype request: ProxyCallRequest
    :param proxy_assertion: the proxy->pod assertion JWS, or ``None``
    :ptype proxy_assertion: str | None
    :param result_subject: the pod-owned subject to deliver the answer on, or ``None`` to keep the
        synchronous reply-inbox path
    :ptype result_subject: str | None
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
        result_subject=result_subject,
    )
    # Dropping unset TOP-LEVEL optionals is what makes a newer registry safe against an older
    # pod, and it is load-bearing rather than tidiness. :class:`CallRequest` is
    # ``extra="forbid"``, so a pod predating a field refuses the WHOLE call rather than ignoring
    # the field -- and since every declared field is serialized, an optional nobody set still
    # reaches the wire as an explicit null. A field can therefore break every lagging pod in the
    # fleet without a single caller populating it, which is exactly the trap the "ship the
    # accepting server first" rollout note is written to avoid: that note assumes absence follows
    # from not setting it, and only this makes it true. An absent key and a null key parse
    # identically on a pod new enough to declare the field, so nothing is lost.
    #
    # Scoped to the top level ON PURPOSE, rather than a recursive ``exclude_none``. The nesting
    # that matters here is ``context``, and :class:`CallContext` does NOT forbid extras -- an
    # unknown dimension there is ignored, never fatal -- so it needs no such protection. Pruning
    # it anyway would strip identity dimensions the proxy deliberately stamps as null, including
    # the verified-``user_id``-is-None case that must override a claimed user; those travel as
    # explicit nulls and the tests around them read the wire.
    payload = internal_request.model_dump(mode="json")
    result = json.dumps({key: value for key, value in payload.items() if value is not None}).encode("utf-8")
    return result


def _parse_pod_accept(payload: bytes) -> "CallAccepted | None":
    """read a pod's acknowledgement, or ``None`` when the body is not one.

    Discriminated on the presence of ``accepted`` rather than on a successful parse: both envelopes
    the pod can send here decode without error under a permissive model, so "did it validate" would
    silently classify a real answer as an acknowledgement and then wait for a result that was already
    in hand.

    :param payload: the raw reply bytes from the pod
    :ptype payload: bytes
    :return: the parsed acknowledgement, or ``None`` when the body is a full answer instead
    :rtype: CallAccepted | None
    """
    from threetears.agent.tools.server import CallAccepted

    result: CallAccepted | None = None
    try:
        decoded = json.loads(payload)
    except ValueError, TypeError:
        # NOSILENT: "this is not an acknowledgement" is the answer this function exists to give.
        # the caller then parses the same bytes as a full response, and reports the failure there
        # with the detail, so raising or logging here would double-report one bad payload.
        return None
    if isinstance(decoded, dict) and "accepted" in decoded:
        try:
            result = CallAccepted.model_validate(decoded)
        except ValidationError:
            # NOSILENT: same contract as above -- a body carrying ``accepted`` but not matching the
            # model is handed on as a response and reported there rather than twice.
            result = None
    return result
