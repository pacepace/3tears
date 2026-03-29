"""authentication and authorization protocols for tool registry.

defines protocols that host applications implement to provide
tool pod verification and agent tool access control. the registry
uses these to enforce security without depending on specific
persistence implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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

    host applications implement this to enforce tool access control
    based on agent configuration (e.g., tool_access patterns in KV).
    """

    async def is_authorized(self, agent_id: str, tool_name: str) -> bool:
        """check if agent is authorized to call named tool.

        :param agent_id: agent identifier
        :ptype agent_id: str
        :param tool_name: fully qualified tool name to check
        :ptype tool_name: str
        :return: True if authorized, False if denied
        :rtype: bool
        """
        ...
