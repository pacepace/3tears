"""tool primitives -- :class:`McpTool` dataclass + ``register_tool`` decorator.

every MCP tool exposed via :class:`~threetears.mcp.server.McpServer` is
described by an :class:`McpTool` instance. tools are typically declared
via the :func:`register_tool` decorator which wraps a handler and
records the tool into a per-process registry the server consumes at
:meth:`McpServer.serve` time.

each tool MUST declare a ``required_permission`` string (e.g.
``"myapp.conversations.read"``). a tool with ``required_permission``
set runs only when :meth:`Authorizer.allows(identity, permission)`
returns True; the framework defaults to default-deny. an AST
enforcement test forbids ``required_permission=None`` in production
``register_tool`` call sites.
"""

from __future__ import annotations

import annotationlib
import inspect
import re
import types
import typing
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from threetears.observe import get_logger

from threetears.mcp.context import CallContext

__all__ = [
    "McpTool",
    "McpToolAnnotations",
    "ToolHandler",
    "ToolProvider",
    "ToolRegistry",
    "register_tool",
]


_log = get_logger(__name__)

#: parameter kinds that collect what is left over; they are never passed the context
_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)

ToolHandler = Callable[..., Awaitable[Any]]
"""signature for a tool's handler.

invoked with the validated input kwargs; returns whatever the tool
produces. the framework formats the return value into the MCP
response envelope (typically ``TextContent``). a handler that declares
a parameter annotated :class:`~threetears.mcp.context.CallContext` is
also passed the call's context in it (see :attr:`McpTool.context_parameter`).
"""


@dataclass(frozen=True, slots=True)
class McpToolAnnotations:
    """behaviour hints a client reads from ``tools/list`` (the spec's ``ToolAnnotations``).

    every hint is optional and ``None`` leaves it out of the listing, so the client applies the
    spec's default for it. hints are advice to the client, not a guarantee the server enforces.
    a 3tears type rather than the SDK's, so a tool declares its hints without importing the SDK.

    :ivar read_only_hint: the tool does not modify its environment (spec default ``False``)
    :ivar destructive_hint: a modifying tool may destroy or overwrite something, rather than only
        add to it (spec default ``True``; meaningful only when ``read_only_hint`` is not ``True``)
    :ivar idempotent_hint: repeating a call with the same arguments has no further effect (spec
        default ``False``; meaningful only when ``read_only_hint`` is not ``True``)
    :ivar open_world_hint: the tool reaches entities outside the server's own closed domain, such
        as the web (spec default ``True``)
    """

    read_only_hint: bool | None = None
    destructive_hint: bool | None = None
    idempotent_hint: bool | None = None
    open_world_hint: bool | None = None

    def to_wire(self) -> dict[str, bool]:
        """the hints as the protocol names them, the unset ones left out.

        :return: e.g. ``{"readOnlyHint": True}``
        :rtype: dict[str, bool]
        """
        hints = {
            "readOnlyHint": self.read_only_hint,
            "destructiveHint": self.destructive_hint,
            "idempotentHint": self.idempotent_hint,
            "openWorldHint": self.open_world_hint,
        }
        return {name: value for name, value in hints.items() if value is not None}


@dataclass(frozen=True, slots=True)
class McpTool:
    """one MCP tool's full descriptor.

    :ivar name: MCP tool name as the client sees it (must be unique
        per server); must be non-empty
    :ivar description: human-readable description; the LLM uses this
        to decide whether to invoke the tool
    :ivar input_schema: JSON Schema (draft 2020-12 compatible) for
        the tool's input arguments; the MCP SDK validates against
        this before dispatch
    :ivar required_permission: permission string the caller must
        hold; default-deny when no grant matches. ``""`` (empty
        string) is rejected at construction because an empty
        permission would silently default-deny every call without
        the explicit "no permission gate" intent the framework
        forbids. tools that genuinely want public access should
        declare a sentinel like ``"public"`` and the consuming
        :class:`Authorizer` impl decides
    :ivar handler: async callable invoked with validated input
        kwargs; return value is wrapped into MCP response envelope
    :ivar output_schema: JSON Schema the tool's structured result
        conforms to, listed as ``outputSchema``. its ``type`` is
        ``"object"`` (the spec's rule for ``structuredContent``). when
        set, a successful call must carry structured content that
        validates against it, or the server answers
        ``OUTPUT_VALIDATION_ERROR`` in place of the result
    :ivar title: human-readable display name, listed as ``title``;
        ``name`` stays the identifier the client calls
    :ivar annotations: behaviour hints, listed as ``annotations``
    :ivar context_parameter: the name of the handler's parameter annotated
        :class:`~threetears.mcp.context.CallContext`, found from the handler's
        signature at construction; the server passes the call's context in it.
        ``None`` when the handler declares none, and it is called with its
        arguments only
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    required_permission: str
    handler: ToolHandler
    output_schema: dict[str, Any] | None = None
    title: str | None = None
    annotations: McpToolAnnotations | None = None
    context_parameter: str | None = field(init=False, default=None, compare=False)

    def __post_init__(self) -> None:
        """validate non-empty fields and the output schema, and find the context parameter, at construction.

        :raises ValueError: ``name``, ``required_permission`` or
            ``title`` is an empty string, ``output_schema`` is not
            a valid JSON Schema whose ``type`` is ``"object"``, or the
            handler's context parameter cannot be passed (see
            :func:`_context_parameter`). silent
            empty-permission default-deny is the kind of bug that
            doesn't surface until a real user is denied for an
            unexplained reason; a bad output schema would otherwise
            fail every call at dispatch instead of once at startup.
        """
        if not self.name:
            raise ValueError("McpTool.name must be non-empty")
        if not self.required_permission:
            raise ValueError(
                "McpTool.required_permission must be non-empty; "
                "use a sentinel like 'public' if the tool truly has "
                "no permission gate (the Authorizer impl decides)",
            )
        if self.title == "":
            raise ValueError(f"McpTool {self.name!r}: title must be non-empty when given; leave it None to omit it")
        if self.output_schema is not None:
            if self.output_schema.get("type") != "object":
                raise ValueError(
                    f"McpTool {self.name!r}: output_schema's type must be 'object', since structuredContent "
                    f"is a JSON object; got {self.output_schema.get('type')!r}",
                )
            try:
                jsonschema.validators.validator_for(self.output_schema).check_schema(self.output_schema)
            except jsonschema.SchemaError as exc:
                raise ValueError(
                    f"McpTool {self.name!r}: output_schema is not a valid JSON Schema: {exc.message}"
                ) from exc
        # frozen: the derived field is set once, here, through object's own setter
        object.__setattr__(self, "context_parameter", _context_parameter(self.name, self.handler, self.input_schema))


#: a call context named as written: ``CallContext`` or a dotted name ending in it
_CONTEXT_NAME = r"(?:[A-Za-z_][A-Za-z0-9_]*\.)*CallContext"

#: the written forms that ARE the call context: the class, or the class or ``None``
_CONTEXT_TEXT = re.compile(
    rf"^(?:{_CONTEXT_NAME}"
    rf"|{_CONTEXT_NAME}\s*\|\s*None|None\s*\|\s*{_CONTEXT_NAME}"
    rf"|(?:typing\.)?Optional\[\s*{_CONTEXT_NAME}\s*\]"
    rf"|(?:typing\.)?Union\[\s*{_CONTEXT_NAME}\s*,\s*None\s*\]"
    rf"|(?:typing\.)?Union\[\s*None\s*,\s*{_CONTEXT_NAME}\s*\])$",
)

#: any written mention of the call context, as a whole name
_CONTEXT_MENTION = re.compile(rf"(?<![A-Za-z0-9_.]){_CONTEXT_NAME}(?![A-Za-z0-9_])")


def _context_annotation(annotation: Any) -> bool | None:
    """whether ``annotation`` is the call context, mentions it in another form, or does neither.

    the call context is :class:`CallContext` itself or ``CallContext | None`` (``Optional``), as an
    evaluated type or, when it could not be evaluated (a module that imports the type only under
    ``TYPE_CHECKING``), as written, by the class's bare or dotted name.

    :param annotation: a parameter's annotation, evaluated where it could be, else its text
    :ptype annotation: Any
    :return: ``True`` when it is the call context, ``None`` when it mentions :class:`CallContext`
        in any other form (a list of them, a union with another type), ``False`` otherwise
    :rtype: bool | None
    """
    if isinstance(annotation, str):
        text = annotation.strip()
        if _CONTEXT_TEXT.match(text):
            return True
        return None if _CONTEXT_MENTION.search(text) else False
    if typing.get_origin(annotation) is typing.Annotated:
        return _context_annotation(typing.get_args(annotation)[0])
    if annotation is CallContext:
        return True
    arguments = typing.get_args(annotation)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        if set(arguments) == {CallContext, type(None)}:
            return True
    if any(_context_annotation(argument) is not False for argument in arguments):
        return None
    return False


def _evaluated(handler: ToolHandler, annotation: Any) -> tuple[Any, bool]:
    """``annotation`` evaluated in the handler's module, or as written when it cannot be.

    each parameter is evaluated on its own, so an annotation that fails (a name imported only under
    ``TYPE_CHECKING``, or one whose evaluation raises) leaves every other parameter's evaluated.

    :param handler: the tool's handler, whose module the annotation is evaluated in
    :ptype handler: ToolHandler
    :param annotation: the annotation as the signature holds it: a type, its text (a module with
        ``from __future__ import annotations``) or a forward reference to a name it could not find
    :ptype annotation: Any
    :return: the evaluated annotation and ``True``, or its text and ``False`` when it could not be
    :rtype: tuple[Any, bool]
    """
    if isinstance(annotation, annotationlib.ForwardRef):
        text = annotation.__forward_arg__
    elif isinstance(annotation, str):
        text = annotation
    else:
        return annotation, True
    module_globals = getattr(inspect.unwrap(handler), "__globals__", {})
    try:
        return eval(text, module_globals), True  # noqa: S307 -- the handler's own annotation, in its own module
    except NameError, AttributeError, TypeError, SyntaxError:
        # NOSILENT: an annotation that cannot be evaluated here is read as written; the caller
        # warns when such a parameter could be an unrecognised context
        return text, False


def _context_parameter(tool_name: str, handler: ToolHandler, input_schema: dict[str, Any]) -> str | None:
    """the name of ``handler``'s parameter annotated :class:`CallContext`, or ``None``.

    :param tool_name: the tool's name, for the error message
    :ptype tool_name: str
    :param handler: the tool's handler
    :ptype handler: ToolHandler
    :param input_schema: the tool's input schema
    :ptype input_schema: dict[str, Any]
    :return: the parameter's name, or ``None`` when the handler declares none
    :rtype: str | None
    :raises ValueError: the handler declares more than one, declares it positional-only (it is
        passed by keyword), names it as one of the tool's input properties (an argument would
        then collide with the context), or annotates a parameter with :class:`CallContext` in a
        form other than the class or the class or ``None`` (it would otherwise run without one)
    """
    try:
        signature = inspect.signature(handler, annotation_format=annotationlib.Format.FORWARDREF)
    except TypeError, ValueError:
        # NOSILENT: a callable with no inspectable signature (some builtins) declares no context
        return None
    properties = input_schema.get("properties")
    input_names = set(properties) if isinstance(properties, dict) else set()
    found: list[inspect.Parameter] = []
    for parameter in signature.parameters.values():
        if parameter.annotation is inspect.Parameter.empty:
            continue
        annotation, evaluated = _evaluated(handler, parameter.annotation)
        verdict = _context_annotation(annotation)
        if verdict is None:
            raise ValueError(
                f"McpTool {tool_name!r}: the handler's parameter {parameter.name!r} is annotated "
                f"{annotation!r}, which names CallContext in a form the server cannot pass; annotate it "
                "CallContext (or CallContext | None)",
            )
        if verdict:
            found.append(parameter)
        elif not evaluated and parameter.name not in input_names and parameter.kind not in _VARIADIC:
            _log.warning(
                "McpTool %r: the annotation %r of the handler's parameter %r cannot be evaluated, so it is "
                "not passed the call context; if it names CallContext under another name, import "
                "CallContext under its own name",
                tool_name,
                annotation,
                parameter.name,
            )
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            f"McpTool {tool_name!r}: the handler declares more than one CallContext parameter "
            f"({', '.join(parameter.name for parameter in found)}); declare one",
        )
    parameter = found[0]
    if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
        raise ValueError(
            f"McpTool {tool_name!r}: the handler's CallContext parameter {parameter.name!r} must be passable "
            "by keyword",
        )
    if parameter.name in input_names:
        raise ValueError(
            f"McpTool {tool_name!r}: the handler's CallContext parameter {parameter.name!r} is also an input "
            "property; rename one",
        )
    return parameter.name


ToolProvider = Callable[[], Iterable[McpTool]]
"""a callable answering the tools a server exposes right now.

:class:`~threetears.mcp.server.McpServer` calls it on every
``tools/list`` and every ``tools/call``, so a tool set that follows a
live registry (one that grows, shrinks or changes a schema at runtime)
is served as it stands at each request. tool names must be unique in
each answer. :meth:`ToolRegistry.all` is one.
"""


class ToolRegistry:
    """per-process tool registry consumed by :class:`McpServer`.

    populated either via direct :meth:`register` calls or via the
    :func:`register_tool` decorator (which calls into a module-level
    default registry). servers that want isolation (e.g. tests
    constructing multiple servers in one process) can pass their own
    :class:`ToolRegistry` instance to :class:`McpServer`.
    """

    def __init__(self) -> None:
        """initialize an empty registry."""
        self._tools: dict[str, McpTool] = {}

    def register(self, tool: McpTool) -> None:
        """add a tool to the registry; raises on duplicate name.

        :param tool: tool descriptor
        :ptype tool: McpTool
        :return: nothing
        :rtype: None
        :raises ValueError: when a tool with the same name is
            already registered
        """
        if tool.name in self._tools:
            raise ValueError(
                f"tool name conflict: {tool.name!r} already registered",
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> McpTool | None:
        """look up a tool by name.

        :param name: MCP tool name
        :ptype name: str
        :return: tool descriptor or ``None`` if not registered
        :rtype: McpTool | None
        """
        return self._tools.get(name)

    def all(self) -> list[McpTool]:
        """return every registered tool in registration order.

        :return: list of tool descriptors
        :rtype: list[McpTool]
        """
        return list(self._tools.values())

    def names(self) -> list[str]:
        """return the names of every registered tool.

        :return: list of tool names
        :rtype: list[str]
        """
        return list(self._tools.keys())


def register_tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    required_permission: str,
    registry: ToolRegistry,
    output_schema: dict[str, Any] | None = None,
    title: str | None = None,
    annotations: McpToolAnnotations | None = None,
) -> Callable[[ToolHandler], ToolHandler]:
    """decorator that registers ``handler`` as an MCP tool against ``registry``.

    canonical usage::

        registry = ToolRegistry()

        @register_tool(
            name="list_conversations",
            description="list recent conversations",
            input_schema={"type": "object", "properties": {...}},
            required_permission="myapp.conversations.read",
            registry=registry,
        )
        async def handle_list_conversations(**kwargs) -> str:
            ...

    ``registry`` is required: the framework intentionally does not
    keep a module-level default registry. Per-product MCP servers
    construct their own registry inside a ``build_registry()``
    factory so the registry's lifetime is tied to one server
    process and tests get isolation by construction. The
    earlier-shipped module-level default was unused by every
    production caller (all three per-product servers passed
    ``registry=`` explicitly) and was removed 2026-05-06 to keep
    the framework's surface honest.

    :param name: MCP tool name (must be unique within the registry)
    :ptype name: str
    :param description: human-readable description
    :ptype description: str
    :param input_schema: JSON Schema for input arguments
    :ptype input_schema: dict[str, Any]
    :param required_permission: permission string the caller must
        hold; default-deny when no grant matches
    :ptype required_permission: str
    :param registry: target registry; required (no module-level default)
    :ptype registry: ToolRegistry
    :param output_schema: JSON Schema of the structured result (see :class:`McpTool`)
    :ptype output_schema: dict[str, Any] | None
    :param title: human-readable display name
    :ptype title: str | None
    :param annotations: behaviour hints
    :ptype annotations: McpToolAnnotations | None
    :return: decorator returning the handler unchanged
    :rtype: Callable[[ToolHandler], ToolHandler]
    """

    def _decorator(handler: ToolHandler) -> ToolHandler:
        """register ``handler`` as :class:`McpTool` and return it unchanged."""
        registry.register(
            McpTool(
                name=name,
                description=description,
                input_schema=input_schema,
                required_permission=required_permission,
                handler=handler,
                output_schema=output_schema,
                title=title,
                annotations=annotations,
            ),
        )
        return handler

    return _decorator
