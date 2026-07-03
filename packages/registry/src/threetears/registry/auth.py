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
from typing import Protocol, runtime_checkable

from threetears.observe import get_logger

__all__ = [
    "AgentToolAuthorizer",
    "AllowAllAuthorizer",
    "DenyAllAuthorizer",
    "ToolPodAuth",
    "ToolPodAuthenticator",
]

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

    host applications implement this to check bootstrap tokens
    against their persistence layer (e.g., tool_pods table).
    """

    async def verify_pod(self, token_hash: str) -> ToolPodAuth | None:
        """verify tool pod by bootstrap token hash.

        :param token_hash: SHA256 hex digest of bootstrap token
        :ptype token_hash: str
        :return: auth context with allowed namespaces, or None if invalid
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
