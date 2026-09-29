"""authentication and authorization protocols for tool registry.

defines protocols that host applications implement to provide
tool pod verification and tool access control for every caller. the registry
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
from uuid import UUID

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
    :param owned_namespaces: the provider nodes this pod OWNS, as
        ``namespaces.name`` values (``tools.pentest``). A bare stem
        (``pentest``) is accepted and rooted by the registry through
        :func:`threetears.core.namespaces.build_tool_provider_node_name`,
        so an implementer reading either spelling out of its store
        produces one family of names rather than two.

        **This is ownership, not permission.** It used to be
        ``allowed_namespaces`` -- a list of what a pod was allowed to
        register at, held in a text column beside the pod's row and
        answerable only there. That column survives as a DECLARATION
        under a name that says so (``tool_pods.declared_namespaces`` on
        the aibots hub), and no longer decides anything. It is now resolved from the namespace
        GRAPH, the same rows a grant names and a schema is derived
        from, so registration and authorization can no longer disagree
        about who owns a provider.

        Written WITHOUT a trailing separator and WITHOUT a glob: these
        are names compared on a segment boundary, so ``pentest.`` and
        ``pentest.*`` match nothing at all.
    :ptype owned_namespaces: list[str]
    :param platform_shared: whether this verified pod is the PLATFORM -- the
        shared built-in tool pod, or a pod the host itself runs -- and so may
        serve every caller a tool name no provider node contains. Any other
        pod offering such a name is refused ``NOT_PLATFORM_SHARED``.

        **The host sets this from VERIFIED identity, and nothing else may.**
        3tears defines the flag and honours it; it never infers it. The aibots
        hub sets it for the ``tool_pods`` row named ``builtin-tool-server``
        (unique by index) and for the identity it issues its own in-process
        pods. A manifest has no field that could set it.
    :ptype platform_shared: bool
    """

    pod_entity_id: str
    name: str
    owned_namespaces: list[str]
    platform_shared: bool = False


@runtime_checkable
class ToolPodAuthenticator(Protocol):
    """protocol for verifying WHO published a registration manifest.

    Every manifest carries its publisher's credential in ``RegistrationManifest.bootstrap_token``,
    re-minted per manifest by the publishing ``ToolServer``'s ``auth_token`` provider. The registry
    reads the pod id to decide which kind of publisher is claimed and asks the matching method --
    never both, and never chosen from anything the token says about itself:

    * a SINGLE-TOKEN pod id is a Tool Pod's, and :meth:`verify_pod` is asked. The verified
      :attr:`ToolPodAuth.pod_entity_id` must equal the manifest's pod id or the whole manifest is
      refused ``POD_ID_MISMATCH``. This covers every pod that serves every caller: a
      ``tool_pods`` row, and the pods the host runs in its own process, for which the host issues
      an identity of its own and answers with ``platform_shared=True``.
    * a DOTTED pod id is an agent's in-process server (``{agent_id}.{instance}``), and
      :meth:`verify_agent` is asked. The verified agent must be the agent the pod id names, or the
      whole manifest is refused ``POD_ID_MISMATCH`` -- which is what stops agent A publishing
      under agent B's pod id.

    The registry passes the RAW token straight through; verification (signature, issuer, expiry,
    key id) is the implementer's responsibility, and any failure answers ``None``. A token that
    fails is REFUSED, never treated as if the manifest had carried none.
    """

    async def verify_pod(self, token: str) -> ToolPodAuth | None:
        """verify a TOOL POD by the token on its registration manifest.

        :param token: the RAW token the pod carried on its registration manifest
            (``RegistrationManifest.bootstrap_token``). under per-key identity this is the pod's
            self-minted identity JWT; the implementer verifies it against the pod's stored key.
            a host that runs tool pods in its own process verifies the identity it issued them
            here too, and answers ``platform_shared=True`` for them
        :ptype token: str
        :return: auth context with the namespaces the pod owns and whether it is the platform, or
            ``None`` if verification fails
        :rtype: ToolPodAuth | None
        """
        ...

    async def verify_agent(self, token: str) -> UUID | None:
        """verify an AGENT by the token on its in-process server's registration manifest.

        The token is the agent's OWN identity -- on the aibots platform the connect JWT the agent
        self-mints with its per-agent Ed25519 key (or the one its hosting runtime mints on its
        behalf), which the host already verifies at NATS connect and on the hub's manifest
        reconciliation. The registry compares the returned id with the agent its pod id names.

        :param token: the RAW token the agent's in-process server carried on its manifest
        :ptype token: str
        :return: the verified agent's id, or ``None`` if verification fails
        :rtype: UUID | None
        """
        ...

    async def provider_nodes(self) -> tuple[str, ...]:
        """every tool PROVIDER node the host's namespace graph holds.

        The whole inventory, not one pod's slice, because the rule needs both
        halves: a pod may claim a name its own node contains, and may not claim
        one inside a node somebody else owns -- and an agent's in-process copy,
        which owns no node, is filtered against it too.

        **A host that returns an empty tuple enforces no ownership**, which is the
        honest answer for a deployment whose graph has no provider nodes. It must
        not be used to signal a read FAILURE: an empty inventory silently widens
        every pod, so an implementer that cannot read the graph should raise and
        let the registration be refused rather than answer with nothing.

        :return: the canonical ``namespaces.name`` of every provider node
        :rtype: tuple[str, ...]
        """
        ...


@runtime_checkable
class AgentToolAuthorizer(Protocol):
    """protocol for checking agent authorization to call specific tools.

    namespace-task-01 phase 2 widened the protocol from the
    two-argument ``(agent_id, tool_name)`` shape to include the
    calling user identity. the unified rbac evaluator resolves a
    two-sided decision (user grants intersected with agent grants)
    so the user dimension is mandatory on the protocol. an AGENT
    dispatch without a user identity passes ``user_id=None`` and
    implementations return ``False``, because an agent's tool grants
    are always two-sided.

    Phase 26 widened the protocol again to carry ``tool_version`` so
    rbac implementations can construct the canonical
    ``namespaces.name`` shape
    (``tools.<sanitized-mcp>.<sanitized-version>``) from the
    dispatch tuple. without the version on the protocol, the
    canonical name is undefined and the namespace lookup is
    inherently ambiguous between concurrent versions of the same
    tool.

    ``principal_is_tool_pod`` carries the one fact the proxy knows
    and the authorizer cannot recover from the ids alone: whether the
    verified principal is a TOOL POD rather than an agent. a tool pod
    acts on nobody's behalf, so it never carries a user, and an
    implementation evaluates it on its own grant alone -- exactly as
    the L3 broker and the hub's datasource authorizer evaluate the
    same principal. the keyword is required rather than defaulted so
    an implementer cannot forget it and silently keep refusing every
    pod, and so a fake that predates it fails loudly under the parity
    walker rather than passing on the wrong protocol.
    """

    async def is_authorized(
        self,
        agent_id: str,
        user_id: str | None,
        tool_name: str,
        tool_version: str,
        *,
        principal_is_tool_pod: bool,
    ) -> bool:
        """check if the verified principal may call the named tool.

        :param agent_id: calling principal UUID in string form: an
            agent's id, or a tool pod's ``tool_pods.id``
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
        :param principal_is_tool_pod: whether the proxy verified the
            principal as a tool pod; with no user, a tool pod is
            evaluated on its own grant and an agent is refused
        :ptype principal_is_tool_pod: bool
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
        *,
        principal_is_tool_pod: bool,
    ) -> bool:
        """return True for any principal and tool combination.

        :param agent_id: calling principal UUID (ignored)
        :ptype agent_id: str
        :param user_id: invoking user UUID (ignored)
        :ptype user_id: str | None
        :param tool_name: fully qualified tool name (ignored)
        :ptype tool_name: str
        :param tool_version: tool version (ignored)
        :ptype tool_version: str
        :param principal_is_tool_pod: whether the principal is a tool pod (ignored)
        :ptype principal_is_tool_pod: bool
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
        *,
        principal_is_tool_pod: bool,
    ) -> bool:
        """return False for any principal and tool combination.

        :param agent_id: calling principal UUID (ignored)
        :ptype agent_id: str
        :param user_id: invoking user UUID (ignored)
        :ptype user_id: str | None
        :param tool_name: fully qualified tool name (ignored)
        :ptype tool_name: str
        :param tool_version: tool version (ignored)
        :ptype tool_version: str
        :param principal_is_tool_pod: whether the principal is a tool pod (ignored)
        :ptype principal_is_tool_pod: bool
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
