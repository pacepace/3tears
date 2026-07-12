"""authentication and authorization protocols for tool registry.

defines protocols that host applications implement to provide
tool pod verification and agent tool access control. the registry
uses these to enforce security without depending on specific
persistence implementations.

namespace-task-01 phase 2: the legacy :class:`KvAgentToolAuthorizer`
(fnmatch patterns read from NATS KV) has been retired. the
production authorizer is :class:`~threetears.registry.rbac_authorizer.RbacEvaluatorAuthorizer`
which delegates to the unified rbac evaluator. the protocol
signature widened to carry ``user_id`` so the evaluator can resolve
user-side grants alongside agent-side ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.registry.proxy import ProxyCallRequest, ProxyCallResponse

__all__ = [
    "INSUFFICIENT_CREDITS",
    "LIMIT_EXCEEDED",
    "AgentToolAuthorizer",
    "AllowAllAuthorizer",
    "AllowAllLimitGuard",
    "DenyAllAuthorizer",
    "DenyAllLimitGuard",
    "EndpointUsageEmitter",
    "LimitDecision",
    "LimitGuard",
    "ToolPodAuth",
    "ToolPodAuthenticator",
]

# the two canonical spend-deny codes a LimitGuard returns. exported as module constants
# so the dispatcher, the tests, and the hub KvCallLimitGuard impl (gu-task-15a) reference
# ONE literal each -- no re-typed strings drifting across the seam.
INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
LIMIT_EXCEEDED = "LIMIT_EXCEEDED"

_logger = get_logger(__name__)


@dataclass
class ToolPodAuth:
    """authentication context for a verified tool pod.

    :param pod_entity_id: tool pod entity identifier
    :ptype pod_entity_id: str
    :param name: tool pod display name
    :ptype name: str
    :param allowed_namespaces: tool name prefixes this pod may register
    :ptype allowed_namespaces: list[str]
    """

    pod_entity_id: str
    name: str
    allowed_namespaces: list[str]


@runtime_checkable
class ToolPodAuthenticator(Protocol):
    """protocol for verifying tool pod identity during registration.

    host applications implement this to VERIFY the pod's self-minted identity JWT (per-key
    identity) against the pod's stored public key in their persistence layer (e.g. the tool_pods
    table). the registry passes the RAW token straight through -- verification (signature, issuer,
    expiry, kid==pod) is the implementer's responsibility, so a bearer-hash comparison is no longer
    the model (a hashed opaque token could not be cryptographically verified).
    """

    async def verify_pod(self, token: str) -> ToolPodAuth | None:
        """verify a tool pod by its presented registration token.

        :param token: the RAW token the pod carried on its registration manifest
            (``RegistrationManifest.bootstrap_token``). under per-key identity this is the pod's
            self-minted identity JWT; the implementer verifies it against the pod's stored key.
        :ptype token: str
        :return: auth context with allowed namespaces, or None if verification fails
        :rtype: ToolPodAuth | None
        """
        ...


@runtime_checkable
class AgentToolAuthorizer(Protocol):
    """protocol for checking agent authorization to call specific tools.

    namespace-task-01 phase 2 widened the protocol from the
    two-argument ``(agent_id, tool_name)`` shape to include the
    calling user identity. the unified rbac evaluator resolves a
    two-sided decision (user grants intersected with agent grants)
    so the user dimension is mandatory on the protocol. callers
    without a user identity (fully-stateless tool dispatch) pass
    ``user_id=None``; implementations return ``False`` because tool
    grants are always two-sided.

    Phase 26 widened the protocol again to carry ``tool_version`` so
    rbac implementations can construct the canonical
    ``platform.namespaces.name`` shape
    (``tools.<sanitized-mcp>.<sanitized-version>``) from the
    dispatch tuple. without the version on the protocol, the
    canonical name is undefined and the namespace lookup is
    inherently ambiguous between concurrent versions of the same
    tool.
    """

    async def is_authorized(
        self,
        agent_id: str,
        user_id: str | None,
        tool_name: str,
        tool_version: str,
    ) -> bool:
        """check if agent + user pair is authorized to call named tool.

        :param agent_id: calling agent UUID in string form
        :ptype agent_id: str
        :param user_id: invoking user UUID in string form, or
            ``None`` when the dispatch carries no user identity
        :ptype user_id: str | None
        :param tool_name: fully qualified ``mcp_name`` to check
        :ptype tool_name: str
        :param tool_version: ``mcp_version`` of the tool dispatch;
            paired with ``tool_name`` to build the canonical
            namespace lookup key
        :ptype tool_version: str
        :return: True if authorized, False if denied
        :rtype: bool
        """
        ...


class AllowAllAuthorizer:
    """authorizer that permits all tool calls unconditionally.

    intended for development and testing environments where tool
    access control is not needed. enabled via the
    THREETEARS_REGISTRY_ALLOW_ALL_TOOLS=true environment variable.
    """

    async def is_authorized(
        self,
        agent_id: str,
        user_id: str | None,
        tool_name: str,
        tool_version: str,
    ) -> bool:
        """return True for any agent and tool combination.

        :param agent_id: calling agent UUID (ignored)
        :ptype agent_id: str
        :param user_id: invoking user UUID (ignored)
        :ptype user_id: str | None
        :param tool_name: fully qualified tool name (ignored)
        :ptype tool_name: str
        :param tool_version: tool version (ignored)
        :ptype tool_version: str
        :return: always True
        :rtype: bool
        """
        return True


class DenyAllAuthorizer:
    """authorizer that denies all tool calls unconditionally.

    serves as default-deny fallback when no custom authorizer is
    provided and allow-all mode is not enabled. production
    deployments should provide a proper AgentToolAuthorizer
    implementation such as
    :class:`~threetears.registry.rbac_authorizer.RbacEvaluatorAuthorizer`.
    """

    async def is_authorized(
        self,
        agent_id: str,
        user_id: str | None,
        tool_name: str,
        tool_version: str,
    ) -> bool:
        """return False for any agent and tool combination.

        :param agent_id: calling agent UUID (ignored)
        :ptype agent_id: str
        :param user_id: invoking user UUID (ignored)
        :ptype user_id: str | None
        :param tool_name: fully qualified tool name (ignored)
        :ptype tool_name: str
        :param tool_version: tool version (ignored)
        :ptype tool_version: str
        :return: always False
        :rtype: bool
        """
        return False


@dataclass(frozen=True)
class LimitDecision:
    """verdict a :class:`LimitGuard` returns for one pre-call spend check.

    a bare bool cannot carry the ``INSUFFICIENT_CREDITS`` vs ``LIMIT_EXCEEDED``
    distinction the dispatcher needs to set the right ``error_code`` on the deny
    response, so the guard returns this two-field frozen carrier instead. an
    allow verdict leaves ``error_code`` ``None``.

    :param allowed: whether the call may proceed to routing
    :ptype allowed: bool
    :param error_code: canonical deny code (:data:`INSUFFICIENT_CREDITS` or
        :data:`LIMIT_EXCEEDED`) when ``allowed`` is ``False``; ``None`` on allow
    :ptype error_code: str | None
    """

    allowed: bool
    error_code: str | None = None


@runtime_checkable
class LimitGuard(Protocol):
    """protocol for the pre-call spend gate, mirroring :class:`AgentToolAuthorizer`.

    every tool dispatch is gated through a limit guard right after the pop check
    and before catalog routing. the guard receives the same dispatch identity the
    authorizer sees plus ``customer_id`` (the spend limit is per-customer) and
    returns a typed :class:`LimitDecision` rather than a bool so the dispatcher can
    map a deny to the right ``error_code``.

    the money path FAILS OPEN (Fork-2): the proxy denies only on a returned
    ``LimitDecision(allowed=False)``. a guard that RAISES or is unreachable makes
    the proxy SERVE the call (and log loudly) -- a billing-infra outage must never
    brick tool traffic. this inverts the fail-CLOSED identity/pop/authorizer gates
    on purpose. the concrete counter-backed implementation is hub code
    (``KvCallLimitGuard``, gu-task-15a); dev/test callers wire
    :class:`AllowAllLimitGuard` / :class:`DenyAllLimitGuard`.
    """

    async def check(
        self,
        agent_id: str,
        user_id: str | None,
        customer_id: str | None,
        tool_name: str,
        tool_version: str,
    ) -> LimitDecision:
        """check whether the customer may spend on this tool call.

        :param agent_id: calling agent UUID in string form
        :ptype agent_id: str
        :param user_id: invoking user UUID in string form, or ``None`` when the
            dispatch carries no user identity
        :ptype user_id: str | None
        :param customer_id: owning customer UUID in string form, or ``None`` when
            the dispatch carries no customer identity; the spend limit is scoped
            to this customer
        :ptype customer_id: str | None
        :param tool_name: fully qualified ``mcp_name`` being called
        :ptype tool_name: str
        :param tool_version: ``mcp_version`` of the tool dispatch
        :ptype tool_version: str
        :return: allow-or-deny verdict carrying the deny ``error_code``
        :rtype: LimitDecision
        """
        ...


class AllowAllLimitGuard:
    """limit guard that permits every call unconditionally.

    intended for development and testing environments where spend limits are not
    enforced, mirroring :class:`AllowAllAuthorizer`. production wires the concrete
    counter-backed ``KvCallLimitGuard`` (gu-task-15a).
    """

    async def check(
        self,
        agent_id: str,
        user_id: str | None,
        customer_id: str | None,
        tool_name: str,
        tool_version: str,
    ) -> LimitDecision:
        """return an allow verdict for any call.

        :param agent_id: calling agent UUID (ignored)
        :ptype agent_id: str
        :param user_id: invoking user UUID (ignored)
        :ptype user_id: str | None
        :param customer_id: owning customer UUID (ignored)
        :ptype customer_id: str | None
        :param tool_name: fully qualified tool name (ignored)
        :ptype tool_name: str
        :param tool_version: tool version (ignored)
        :ptype tool_version: str
        :return: always ``LimitDecision(allowed=True)``
        :rtype: LimitDecision
        """
        return LimitDecision(allowed=True)


class DenyAllLimitGuard:
    """limit guard that denies every call with :data:`INSUFFICIENT_CREDITS`.

    serves as a deterministic deny stub for tests + kill-switch wiring, mirroring
    :class:`DenyAllAuthorizer`.
    """

    async def check(
        self,
        agent_id: str,
        user_id: str | None,
        customer_id: str | None,
        tool_name: str,
        tool_version: str,
    ) -> LimitDecision:
        """return a deny verdict carrying :data:`INSUFFICIENT_CREDITS` for any call.

        :param agent_id: calling agent UUID (ignored)
        :ptype agent_id: str
        :param user_id: invoking user UUID (ignored)
        :ptype user_id: str | None
        :param customer_id: owning customer UUID (ignored)
        :ptype customer_id: str | None
        :param tool_name: fully qualified tool name (ignored)
        :ptype tool_name: str
        :param tool_version: tool version (ignored)
        :ptype tool_version: str
        :return: always ``LimitDecision(allowed=False, error_code=INSUFFICIENT_CREDITS)``
        :rtype: LimitDecision
        """
        return LimitDecision(allowed=False, error_code=INSUFFICIENT_CREDITS)


@runtime_checkable
class EndpointUsageEmitter(Protocol):
    """protocol for the post-call usage-emit seam, mirroring the guard injection.

    invoked at the one dispatch point where both the inbound ``request`` arguments
    and the outbound ``response`` content are in hand (after the tool pod replies),
    fire-and-forget: an emit failure is caught and logged by the proxy and NEVER
    affects the reply. 3tears holds ONLY this protocol + the injection slot; the
    concrete emitter that builds the SDK-typed usage event and publishes it on
    :meth:`~threetears.nats.Subjects.hub_endpoint_usage_track` is hub code
    (gu-task-16) -- a 3tears→SDK type import would be a layering violation, so the
    emit is injected exactly as the limit guard is.
    """

    async def emit(self, request: ProxyCallRequest, response: ProxyCallResponse) -> None:
        """emit one endpoint-usage record for a completed tool call.

        :param request: the verified inbound call request (arguments + identity)
        :ptype request: ProxyCallRequest
        :param response: the outbound tool response (content + success)
        :ptype response: ProxyCallResponse
        :return: nothing
        :rtype: None
        """
        ...
