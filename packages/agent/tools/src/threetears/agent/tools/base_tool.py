"""TearsTool abstract base class and supporting dataclasses.

defines interface that all tools must implement to participate
in both direct mode and MCP server mode. no platform dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from threetears.agent.tools._coercion import normalize_kwargs
from threetears.observe import get_logger

__all__ = [
    "MCPToolDefinition",
    "TearsTool",
    "ToolResult",
]

_log = get_logger(__name__)


@dataclass
class ToolResult:
    """result returned from tool execution.

    :param success: whether execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param metadata: optional additional metadata
    :ptype metadata: dict[str, Any] | None
    :param error: error message if execution failed
    :ptype error: str | None
    """

    success: bool
    content: str
    metadata: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class MCPToolDefinition:
    """MCP-compatible tool definition schema.

    :param name: tool name in namespace convention ({package}.{name})
    :ptype name: str
    :param version: semver-compatible version string
    :ptype version: str
    :param description: human-readable tool description
    :ptype description: str
    :param input_schema: JSON Schema for tool input parameters
    :ptype input_schema: dict[str, Any]
    :param output_schema: optional JSON Schema for tool output
    :ptype output_schema: dict[str, Any] | None
    :param timeout_seconds: expected maximum execution time, None uses caller default
    :ptype timeout_seconds: float | None
    """

    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    timeout_seconds: float | None = None


class TearsTool(ABC):
    """abstract base class for all platform tools.

    tools implement this interface to work in both direct mode
    (called from agent code) and MCP server mode (served via
    ToolServer, called through Registry proxy). no platform
    dependencies required -- standalone by design.

    ``run`` is the callable surface: a template method that normalizes
    loose LLM-supplied inputs (empty strings or JSON-encoded strings
    for object/array fields) against this tool's declared
    ``mcp_schema()`` input schema, then dispatches to subclass
    ``execute``. subclasses override ``execute`` and never ``run``;
    the ``__init_subclass__`` guard below enforces that contract at
    class-creation time so the coercion step can never be bypassed.

    Visibility flags decouple "is this tool in the agent's default tool
    surface?" from "is this tool discoverable via the skills catalog?".
    They govern default visibility only -- ACL still governs
    authorization. The four-state matrix is:

    - ``tool_eligible=True, skill_eligible=False`` (default): regular
      tool; appears in the agent's default tool surface, NOT in the
      skills catalog.
    - ``tool_eligible=True, skill_eligible=True``: tool-shaped skill;
      in the default surface AND surfaced via the skills catalog.
    - ``tool_eligible=False, skill_eligible=True``: skill-only tool;
      not in the default surface, available only when a skill lists
      it in ``tool_additions``. The "code-skill without sandbox"
      pattern.
    - ``tool_eligible=False, skill_eligible=False``: registers but
      never visible. Almost certainly a configuration error;
      :class:`~threetears.agent.tools.server.ToolServer` emits a
      WARNING at registration time.

    Subclasses set these as class attributes, not instance attributes;
    they are part of the tool's authored declaration and do not change
    at runtime. Per-customer overrides live in the ACL grant layer, not
    in these flags.

    Face flags decouple a capability's *reach* -- the surfaces through
    which it can be called -- from its declaration. A tool may expose
    any combination of the three reach faces:

    - ``face_platform_tool`` (default ``True``): reachable over the
      internal NATS mesh as a native platform tool (the historical
      one-and-only reach). ON by default so a tool authored before this
      distinction is a platform mesh tool unless told otherwise.
    - ``face_api`` (default ``False``): reachable as an external HTTP
      API operation.
    - ``face_mcp`` (default ``False``): reachable as an external MCP
      tool.

    The face flags govern *reach* only -- ACL still governs
    *authorization*. Like the eligibility flags, subclasses set them as
    class attributes, not instance attributes.

    :cvar tool_eligible: whether this tool appears in the agent's
        default tool surface. Defaults to ``True`` so existing tools
        keep their pre-shard behaviour without opting in.
    :cvar skill_eligible: whether this tool is discoverable via the
        skills catalog (e.g. via ``skill_list`` queries or
        ``list_skill_eligible_tools``). Defaults to ``False`` so no
        existing tool leaks into the skills catalog without explicit
        opt-in.
    :cvar face_platform_tool: whether this tool is reachable over the
        internal NATS mesh as a native platform tool. Defaults to
        ``True`` so existing tools keep their historical reach without
        opting in.
    :cvar face_api: whether this tool is reachable as an external HTTP
        API operation. Defaults to ``False``.
    :cvar face_mcp: whether this tool is reachable as an external MCP
        tool. Defaults to ``False``.
    """

    tool_eligible: bool = True
    skill_eligible: bool = False
    face_platform_tool: bool = True
    face_api: bool = False
    face_mcp: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """enforce the run/execute contract at subclass-creation time.

        subclasses MUST NOT override ``run`` -- coercion is the
        platform's job, not the subclass's. the "must define execute"
        half of the contract is already handled by Python's abstract
        method machinery: an intermediate base (SchemaTool, AdminTool)
        can satisfy ``execute`` on behalf of its concrete subclasses,
        and Python raises cleanly on instantiation if the abstract
        set is nonempty. adding an explicit "execute in __dict__"
        check here would wrongly reject those intermediate-base
        dispatch patterns.

        :param kwargs: forwarded to ``object.__init_subclass__``
        :ptype kwargs: Any
        :raises TypeError: when ``run`` is overridden by a subclass
        """
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            msg = (
                f"{cls.__name__} overrides TearsTool.run; run is the "
                "platform-owned template method that wraps input "
                "coercion. override ``execute`` instead."
            )
            raise TypeError(msg)

    async def run(self, **kwargs: Any) -> ToolResult:
        """normalize loose LLM-supplied kwargs then dispatch to subclass.

        coerces wrong-shape object/array values (empty strings or
        JSON-encoded strings) toward their declared JSON schema
        types, then calls subclass ``execute`` with the normalized
        kwargs. if ``mcp_schema`` itself raises, falls back to the
        original kwargs so coercion never breaks a tool.

        :param kwargs: raw tool input parameters
        :ptype kwargs: Any
        :return: execution result from subclass ``execute``
        :rtype: ToolResult
        """
        try:
            schema = self.mcp_schema()
            normalized = normalize_kwargs(kwargs, schema.input_schema)
        except Exception:
            # coercion never breaks a tool: fall back to raw kwargs if
            # mcp_schema() or normalize_kwargs raises. log so the fallback
            # is visible in diagnostics instead of silently masking a bug
            # in the tool's schema.
            _log.warning(
                "tool input coercion fell back to raw kwargs",
                extra={"extra_data": {"tool_class": type(self).__name__}},
                exc_info=True,
            )
            normalized = kwargs
        return await self.execute(**normalized)

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """execute tool with normalized parameters.

        subclasses implement this. ``run`` runs coercion against
        ``mcp_schema().input_schema`` before calling ``execute``, so
        object/array kwargs arrive as native containers even when the
        LLM caller supplied empty strings or JSON-encoded strings.

        :param kwargs: tool-specific input parameters, post-coercion
        :ptype kwargs: Any
        :return: execution result
        :rtype: ToolResult
        """
        ...

    @abstractmethod
    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition.

        :return: tool definition with name, version, description, schemas
        :rtype: MCPToolDefinition
        """
        ...

    @abstractmethod
    def mcp_name(self) -> str:
        """return namespaced tool name.

        format: {package}.{name} (e.g., 'threetears.calculator')

        :return: namespaced tool name
        :rtype: str
        """
        ...

    @abstractmethod
    def mcp_version(self) -> str:
        """return semver-compatible tool version.

        :return: version string (e.g., '1.0.0')
        :rtype: str
        """
        ...
