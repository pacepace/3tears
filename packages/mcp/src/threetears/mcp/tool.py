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
import ast
import builtins
import inspect
import re
import types
import typing
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
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


#: the modules :class:`CallContext` is importable from
_CONTEXT_MODULES = ("threetears.mcp", "threetears.mcp.context")

#: names the call context goes by in any module, imports aside: its fully qualified ones
_QUALIFIED_CONTEXT_NAMES = frozenset(f"{module}.CallContext" for module in _CONTEXT_MODULES)

#: a dotted name, as an annotation's text writes one
_NAME = re.compile(r"(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

#: a ``Literal[...]`` inside an annotation's text: its arguments are values, which name no type
_LITERAL = re.compile(r"(?<![A-Za-z0-9_])(?:typing\.)?Literal\[[^\]]*\]")

#: a quoted forward reference inside an annotation's text, read as the name it quotes
_QUOTED = re.compile(r"(['\"])(.*?)\1")


#: where an ``Annotated[`` opens inside an annotation's text
_ANNOTATED_OPEN = re.compile(r"(?<![A-Za-z0-9_])(?:typing\.)?Annotated\[")


def _without_metadata(text: str) -> str:
    """``text`` with every ``Annotated[X, ...]`` in it replaced by ``X``: metadata names no type.

    :param text: an annotation as written
    :ptype text: str
    :return: the text, each ``Annotated`` reduced to the type it wraps
    :rtype: str
    """
    while (match := _ANNOTATED_OPEN.search(text)) is not None:
        depth = 0
        first_end: int | None = None
        quote: str | None = None
        for index in range(match.end(), len(text)):
            char = text[index]
            if quote is not None:
                # inside quoted metadata: only its closing quote matters
                quote = None if char == quote else quote
            elif char in "'\"":
                quote = char
            elif char in "[(":
                depth += 1
            elif char in "])":
                if depth == 0:
                    inner = text[match.end() : first_end if first_end is not None else index]
                    text = text[: match.start()] + inner.strip() + text[index + 1 :]
                    break
                depth -= 1
            elif char == "," and depth == 0 and first_end is None:
                first_end = index
        else:
            return text  # unbalanced: leave it as written
    return text


def _type_text(text: str) -> str:
    """``text`` holding only names of types: ``Annotated`` metadata and ``Literal`` values dropped,
    quoted forward references unquoted.

    :param text: an annotation as written
    :ptype text: str
    :return: the text, holding only the names of types
    :rtype: str
    """
    return _QUOTED.sub(lambda match: match.group(2), _LITERAL.sub("", _without_metadata(text)))


#: names an annotation's text may use beside the type itself, which no import binds
_BUILTIN_NAMES = frozenset({"None", "Optional", "Union", "typing.Optional", "typing.Union"})

#: warnings already given, so a tool rebuilt on every listing warns once: (tool, parameter, annotation)
_WARNED: set[tuple[str, str, str]] = set()


@dataclass(frozen=True)
class _ModuleNames:
    """what the handler's module binds by import, read from its source (``TYPE_CHECKING`` imports too).

    :param context: the names that mean :class:`CallContext` there, its qualified ones included
    :ptype context: frozenset[str]
    :param imported: every name an import binds there, whatever it names
    :ptype imported: frozenset[str]
    """

    context: frozenset[str]
    imported: frozenset[str]


def _module_names(handler: ToolHandler) -> _ModuleNames:
    """read the handler's module's imports, so an annotation it could not evaluate is read by them.

    an annotation is text when its module imports the type only under ``TYPE_CHECKING``. the text
    alone cannot say which ``CallContext`` it means (``threetears.core`` exports one too) or what an
    alias stands for; the module's own import statements can.

    :param handler: the tool's handler
    :ptype handler: ToolHandler
    :return: the names that mean the call context there, and every name an import binds
    :rtype: _ModuleNames
    """
    context = set(_QUALIFIED_CONTEXT_NAMES)
    imported: set[str] = set()
    module = inspect.getmodule(inspect.unwrap(handler))
    try:
        tree = ast.parse(inspect.getsource(module)) if module is not None else None
    except OSError, TypeError, SyntaxError:
        # NOSILENT: no readable source (a built or generated module); only qualified names are known
        tree = None
    for node in ast.walk(tree) if tree is not None else ():
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local = alias.asname or alias.name
                imported.add(local)
                if node.level != 0:
                    continue  # a relative import binds a name, never threetears.mcp's
                if node.module in _CONTEXT_MODULES and alias.name == "CallContext":
                    context.add(local)
                elif node.module == "threetears" and alias.name == "mcp":
                    context.add(f"{local}.CallContext")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
                if alias.name in _CONTEXT_MODULES and alias.asname:
                    context.add(f"{alias.asname}.CallContext")
    return _ModuleNames(context=frozenset(context), imported=frozenset(imported))


class _LazyNames:
    """the handler's module's imports, read the first time an annotation's text needs them.

    every text the verdict meets (a whole annotation that would not evaluate, or a quoted type
    argument inside one that did) is read by the same imports, so a parameter's verdict does not
    depend on whether another parameter happened to need them first.

    :param handler: the tool's handler
    :ptype handler: ToolHandler
    """

    def __init__(self, handler: ToolHandler) -> None:
        """remember the handler; nothing is read yet.

        :param handler: the tool's handler
        :ptype handler: ToolHandler
        :return: nothing
        :rtype: None
        """
        self._handler = handler
        self._names: _ModuleNames | None = None

    def get(self) -> _ModuleNames:
        """the module's names, read once.

        :return: what the handler's module binds by import
        :rtype: _ModuleNames
        """
        if self._names is None:
            self._names = _module_names(self._handler)
        return self._names


def _known(text: str, names: _ModuleNames) -> bool:
    """whether every name ``text`` writes is bound by an import of its module, or is a builtin.

    :param text: an annotation as written
    :ptype text: str
    :param names: what the handler's module binds by import
    :ptype names: _ModuleNames
    :return: ``True`` when the text names nothing the module leaves unaccounted for
    :rtype: bool
    """
    for match in _NAME.finditer(_type_text(text)):
        name = match.group(0)
        root = name.split(".")[0]
        if name not in _BUILTIN_NAMES and root not in names.imported and not hasattr(builtins, root):
            return False
    return True


def _text_verdict(text: str, names: _ModuleNames) -> bool | None:
    """read an annotation's text the way :func:`_context_annotation` reads a type.

    :param text: the annotation as written
    :ptype text: str
    :param names: what the handler's module binds by import
    :ptype names: _ModuleNames
    :return: ``True`` when the text is the call context or the call context or ``None``, ``None``
        when it names the call context in any other form, ``False`` otherwise
    :rtype: bool | None
    """
    written = [match.group(0) for match in _NAME.finditer(_type_text(text))]
    mentions = [name for name in written if name in names.context]
    if not mentions:
        return False
    if len(mentions) == 1:
        alone = re.sub(rf"(?<![A-Za-z0-9_.]){re.escape(mentions[0])}(?![A-Za-z0-9_])", "T", _type_text(text), count=1)
        rest = re.sub(r"\s+", "", alone)
        if rest in {
            "T",
            "T|None",
            "None|T",
            "Optional[T]",
            "typing.Optional[T]",
            "Union[T,None]",
            "Union[None,T]",
            "typing.Union[T,None]",
            "typing.Union[None,T]",
        }:
            return True
    return None


def _context_annotation(annotation: Any, names: _LazyNames) -> bool | None:
    """whether ``annotation`` is the call context, mentions it in another form, or does neither.

    the call context is :class:`CallContext` itself or ``CallContext | None`` (``Optional``), possibly
    wrapped in ``Annotated``, as an evaluated type or, when it could not be evaluated, as written,
    under a name the handler's module imports it by (an alias included) or a qualified name.

    :param annotation: a parameter's annotation, evaluated where it could be, else its text
    :ptype annotation: Any
    :param names: what the handler's module binds by import, read when a text needs it
    :ptype names: _LazyNames
    :return: ``True`` when it is the call context, ``None`` when it mentions :class:`CallContext`
        in any other form (a list of them, a union with another type), ``False`` otherwise
    :rtype: bool | None
    """
    if isinstance(annotation, annotationlib.ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        return _text_verdict(annotation.strip(), names.get())
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _context_annotation(typing.get_args(annotation)[0], names)
    if annotation is CallContext:
        return True
    if origin is typing.Literal:
        # a Literal's arguments are values, not types, whatever text they hold
        return False
    arguments = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType) and type(None) in arguments:
        # `X | None`: X is CallContext itself or a forward reference (`Optional["CallContext"]`) to it
        others = [argument for argument in arguments if argument is not type(None)]
        if len(others) == 1 and _context_annotation(others[0], names) is True:
            return True
    if any(_context_annotation(argument, names) is not False for argument in arguments):
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
        # NOSILENT: an annotation that cannot be evaluated here is read as written, by the module's
        # imports; the caller warns when that still cannot say what it names
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
    names = _LazyNames(handler)
    found: list[inspect.Parameter] = []
    for parameter in signature.parameters.values():
        if parameter.annotation is inspect.Parameter.empty:
            continue
        annotation, evaluated = _evaluated(handler, parameter.annotation)
        verdict = _context_annotation(annotation, names)
        if verdict is None:
            raise ValueError(
                f"McpTool {tool_name!r}: the handler's parameter {parameter.name!r} is annotated "
                f"{annotation!r}, which names CallContext in a form the server cannot pass; annotate it "
                "CallContext (or CallContext | None)",
            )
        if verdict:
            found.append(parameter)
        elif (
            not evaluated
            and parameter.name not in input_names
            and parameter.kind not in _VARIADIC
            and not _known(annotation, names.get())
        ):
            key = (tool_name, parameter.name, annotation)
            if key not in _WARNED:
                _WARNED.add(key)
                _log.warning(
                    "McpTool %r: the annotation %r of the handler's parameter %r cannot be evaluated and "
                    "names nothing its module imports, so it is not passed the call context; if it means "
                    "CallContext, import CallContext from threetears.mcp in that module",
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
        # a read-only snapshot replaced whole on a change (decision 41); one entry per tool
        self._tools: Mapping[str, McpTool] = MappingProxyType({})

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
        self._tools = MappingProxyType({**self._tools, tool.name: tool})

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
