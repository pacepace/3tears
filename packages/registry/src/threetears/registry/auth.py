"""authentication and authorization protocols for tool registry.

defines protocols that host applications implement to provide
tool pod verification and agent tool access control. the registry
uses these to enforce security without depending on specific
persistence implementations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import Any, Protocol, runtime_checkable

from threetears.observe import get_logger

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


class AllowAllAuthorizer:
    """authorizer that permits all tool calls unconditionally.

    intended for development and testing environments where tool
    access control is not needed. enabled via the
    FOURTEENAIBOTS_REGISTRY_ALLOW_ALL_TOOLS=true environment variable.
    """

    async def is_authorized(self, agent_id: str, tool_name: str) -> bool:
        """return True for any agent and tool combination.

        :param agent_id: agent identifier (ignored)
        :ptype agent_id: str
        :param tool_name: fully qualified tool name (ignored)
        :ptype tool_name: str
        :return: always True
        :rtype: bool
        """
        return True


class DenyAllAuthorizer:
    """authorizer that denies all tool calls unconditionally.

    serves as default-deny fallback when no custom authorizer is
    provided and allow-all mode is not enabled. production
    deployments should provide a proper AgentToolAuthorizer
    implementation (e.g., ToolAccessAuthorizer wrapping
    UnifiedAccessCache).
    """

    async def is_authorized(self, agent_id: str, tool_name: str) -> bool:
        """return False for any agent and tool combination.

        :param agent_id: agent identifier (ignored)
        :ptype agent_id: str
        :param tool_name: fully qualified tool name (ignored)
        :ptype tool_name: str
        :return: always False
        :rtype: bool
        """
        return False


@dataclass
class _CachedToolPatterns:
    """TTL-cached tool access patterns for single agent.

    :param patterns: list of tool name glob patterns
    :ptype patterns: list[str]
    :param date_cached: when entry was cached
    :ptype date_cached: datetime
    """

    patterns: list[str]
    date_cached: datetime


class KvAgentToolAuthorizer:
    """authorizer that reads agent tool access patterns from NATS KV.

    reads agent config from ``{namespace}_agent_config`` KV bucket,
    extracts ``access.tools`` patterns, and checks tool names against
    them using fnmatch. caches parsed patterns per agent with
    configurable TTL. fail-closed: denies access if KV is unavailable
    or agent config is missing.

    :param ttl_seconds: cache entry TTL in seconds
    :ptype ttl_seconds: float
    """

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        """initialize KV agent tool authorizer.

        :param ttl_seconds: cache entry TTL in seconds
        :ptype ttl_seconds: float
        """
        self._ttl_seconds = ttl_seconds
        self._kv: Any = None
        self._cache: dict[str, _CachedToolPatterns] = {}

    async def initialize(self, js: Any, namespace: str) -> None:
        """open NATS KV bucket for agent config.

        attempts to open existing bucket first, then creates it if
        not found. logs warning and continues if both fail (authorizer
        will fail-closed on every check until bucket becomes available).

        :param js: NATS JetStream context
        :ptype js: Any
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :return: nothing
        :rtype: None
        """
        bucket_name = f"{namespace}_agent_config"
        try:
            self._kv = await js.key_value(bucket=bucket_name)
            _logger.info(
                "KV agent tool authorizer initialized",
                extra={"extra_data": {"bucket": bucket_name}},
            )
        except Exception:
            try:
                self._kv = await js.create_key_value(bucket=bucket_name)
                _logger.info(
                    "KV agent tool authorizer created bucket",
                    extra={"extra_data": {"bucket": bucket_name}},
                )
            except Exception:
                _logger.warning(
                    "KV agent tool authorizer failed to open bucket, "
                    "will fail-closed until bucket is available",
                    extra={"extra_data": {"bucket": bucket_name}},
                )

    async def is_authorized(self, agent_id: str, tool_name: str) -> bool:
        """check if agent is authorized to call named tool.

        reads agent config from NATS KV, extracts access.tools
        patterns, and matches tool_name against each pattern using
        fnmatch. uses TTL-based cache to avoid repeated KV reads.
        fail-closed: returns False if KV bucket not available,
        agent config missing, or any error occurs.

        :param agent_id: agent identifier
        :ptype agent_id: str
        :param tool_name: fully qualified tool name to check
        :ptype tool_name: str
        :return: True if authorized, False if denied
        :rtype: bool
        """
        patterns = await self._get_patterns(agent_id)
        result = False
        for pattern in patterns:
            if fnmatch(tool_name, pattern):
                result = True
                break
        return result

    async def _get_patterns(self, agent_id: str) -> list[str]:
        """get tool access patterns from cache or load from KV.

        :param agent_id: agent identifier
        :ptype agent_id: str
        :return: list of tool name glob patterns (empty if unavailable)
        :rtype: list[str]
        """
        cached = self._cache.get(agent_id)
        if cached is not None:
            age = (datetime.now(UTC) - cached.date_cached).total_seconds()
            if age < self._ttl_seconds:
                return cached.patterns
            del self._cache[agent_id]

        patterns = await self._load_patterns(agent_id)
        self._cache[agent_id] = _CachedToolPatterns(
            patterns=patterns,
            date_cached=datetime.now(UTC),
        )
        result = patterns
        return result

    async def _load_patterns(self, agent_id: str) -> list[str]:
        """load tool access patterns from agent config in KV.

        fail-closed: returns empty list on any error.

        :param agent_id: agent identifier
        :ptype agent_id: str
        :return: list of tool name glob patterns
        :rtype: list[str]
        """
        result: list[str] = []
        if self._kv is None:
            _logger.warning(
                "KV bucket not available, denying tool access for agent %s",
                agent_id,
            )
            return result
        try:
            entry = await self._kv.get(agent_id)
            raw = entry.value
            if raw is not None:
                config = json.loads(raw)
                access_block = config.get("access", {})
                if isinstance(access_block, dict):
                    tools = access_block.get("tools", [])
                    if isinstance(tools, list):
                        result = tools
        except Exception:
            _logger.warning(
                "failed to load tool access config from KV for agent %s, denying access",
                agent_id,
            )
        return result
