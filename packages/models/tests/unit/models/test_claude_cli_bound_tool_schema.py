"""What a subscription model's bound tool looks like to the CLI, and that its handler runs nothing.

The model hands every tool call back to the caller (see ``_claude_cli``'s module docstring), so the
handler the CLI calls between the model's tool use and its one-turn limit must not run the tool.
What still matters on the CLI side is what the model is SHOWN: the tool's wire name and schema.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.providers import _claude_cli as claude_cli
from threetears.models.providers._claude_cli import create_subscription_chat


class _EchoTool(BaseTool):
    """A trivial :class:`BaseTool` that returns its args, for tool-event tests."""

    name: str = "echo"
    description: str = "echoes its input"

    def _run(self, **kwargs: Any) -> str:
        return f"echo:{kwargs}"

    async def _arun(self, **kwargs: Any) -> str:
        return f"echo:{kwargs}"


class _DottedNameTool(BaseTool):
    """A :class:`BaseTool` with a canonical dotted 3tears name, like every real builtin
    (``threetears.web_search``, ``threetears.calculator``, ...) -- see
    :meth:`threetears.agent.tools.base_tool.BaseAgentTool.mcp_name`."""

    name: str = "threetears.web_search"
    description: str = "search the web"

    def _run(self, **kwargs: Any) -> str:
        return f"result:{kwargs}"

    async def _arun(self, **kwargs: Any) -> str:
        return f"result:{kwargs}"


def _wrap(tool: BaseTool):
    """Build the subscription model and wrap ``tool``, returning the raw ``SdkMcpTool``."""
    model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
    schema = {"properties": {"x": {"type": "string"}}, "required": []}
    return model._wrap_langchain_tool(tool, schema)  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]


class _MustNotRunTool(BaseTool):
    """A tool whose running is the failure."""

    name: str = "threetears.send_email"
    description: str = "sends an email"

    def _run(self, **kwargs: Any) -> str:
        raise AssertionError("the CLI's handler ran the tool; tool calls belong to the caller")

    async def _arun(self, **kwargs: Any) -> str:
        raise AssertionError("the CLI's handler ran the tool; tool calls belong to the caller")


def _wrapped_handler(tool: BaseTool):
    """Build the subscription model, wrap ``tool``, and return the SDK tool's raw async handler."""
    return _wrap(tool).handler


class TestTheHandlerRunsNothing:
    async def test_the_handler_answers_without_running_the_tool(self) -> None:
        result = await _wrapped_handler(_MustNotRunTool())({"to": "someone"})
        assert result == {"content": [{"type": "text", "text": claude_cli._HANDED_BACK}]}  # noqa: SLF001

    async def test_a_dotted_tool_is_registered_under_its_wire_name_and_still_not_run(self) -> None:
        from threetears.models.tool_name_translation import build_name_translation

        [wire_tool], _reverse_map = build_name_translation([_MustNotRunTool()])
        sdk_tool = _wrap(wire_tool)
        assert sdk_tool.name == "threetears_send_email"
        assert "is_error" not in await sdk_tool.handler({})


class TestDottedToolNamePermissionFix:
    """Every 3tears builtin's canonical name is dotted (``threetears.web_search``, per
    ``BaseAgentTool.mcp_name()``). Live-observed bug: every real call to a bound dotted tool was
    silently DENIED under a subscription turn ("Claude requested permissions to use ... but you
    haven't granted it yet"), with nothing logged anywhere, while the same tool worked fine on
    every other backend. Root cause: the base class's own ``bind_tools`` already tries to
    auto-approve bound tools by deriving ``allowed_tools`` from each tool's raw (dotted) ``.name``,
    but the SDK/CLI normalizes dots out of tool identities on the wire, so that entry never matched
    -- the auto-approval attempt silently failed to match its own target. ``bind_tools`` here
    substitutes each dotted tool for a ``NameMangledToolProxy`` (the same translation
    ``anthropic.py``/``openrouter.py`` already apply for the identical Anthropic tool-name
    constraint) before the base class ever derives ``allowed_tools``, so the entry matches.
    """

    def test_allowed_tools_matches_the_underscored_wire_identity(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_DottedNameTool()])
        assert bound.bound.allowed_tools == ["mcp__langchain-tools__threetears_web_search"]  # type: ignore[attr-defined]

    def test_dotless_tool_name_is_unaffected(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_EchoTool()])
        assert bound.bound.allowed_tools == ["mcp__langchain-tools__echo"]  # type: ignore[attr-defined]

    def test_permission_mode_is_left_at_the_default(self) -> None:
        """The fix is a precise allowed_tools match, not a blanket permission bypass."""
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_DottedNameTool()])
        assert bound.bound.permission_mode == "default"  # type: ignore[attr-defined]


class TestOptionalParameterSchemaFix:
    """Live-observed bug, two compounding root causes on the same bound tool
    (``memory_search``'s real schema, lifted from
    ``MemorySearchInput.model_json_schema()`` in ``threetears.agent.memory.tools``):

    1. A ``list``-typed Optional parameter (``ids: list[str] | None``) arrived at the handler as
       the literal string ``"[]"``, failing pydantic validation ("Input should be a valid list")
       every time the model tried to pass it. Pydantic renders an ``X | None`` field as
       ``anyOf: [{type: X}, {type: null}]`` with no top-level ``type`` key, and the naive
       ``prop.get("type", "string")`` conversion used to default every such field to ``string``.
    2. Every OTHER optional filter (``alias``, ...) arrived populated with an empty string on
       every call instead of being omitted. ``claude_agent_sdk.create_sdk_mcp_server``'s own
       schema builder marks every key in a bare ``{name: type}`` map as ``required`` -- forcing
       the model to invent a value for filters it had nothing to fill in. The fix now hands the
       SDK a full ``{"type": "object", "properties": ..., "required": [...]}`` schema so its
       required-everything fallback never triggers.
    """

    _MEMORY_SEARCH_LIKE_SCHEMA = {
        "properties": {
            "query": {"type": "string"},
            "ids": {
                "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
                "default": None,
            },
            "limit": {"type": "integer", "default": 10},
            "alias": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "required": ["query"],
    }

    def test_optional_list_parameter_resolves_to_array_not_string(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        sdk_tool = model._wrap_langchain_tool(  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            _EchoTool(), self._MEMORY_SEARCH_LIKE_SCHEMA
        )
        assert sdk_tool.input_schema["properties"]["ids"]["type"] == "array"

    def test_optional_string_parameter_still_resolves_to_string(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        sdk_tool = model._wrap_langchain_tool(  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            _EchoTool(), self._MEMORY_SEARCH_LIKE_SCHEMA
        )
        assert sdk_tool.input_schema["properties"]["alias"]["type"] == "string"

    def test_plain_typed_parameters_are_unaffected(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        sdk_tool = model._wrap_langchain_tool(  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            _EchoTool(), self._MEMORY_SEARCH_LIKE_SCHEMA
        )
        assert sdk_tool.input_schema["properties"]["query"]["type"] == "string"
        assert sdk_tool.input_schema["properties"]["limit"]["type"] == "integer"

    def test_only_the_genuinely_required_field_is_marked_required(self) -> None:
        """The SDK's own required-everything fallback must never trigger: ``ids``/``limit``/
        ``alias`` all have defaults in the source schema and must NOT appear in ``required``,
        even though they're present in ``properties``."""
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        sdk_tool = model._wrap_langchain_tool(  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            _EchoTool(), self._MEMORY_SEARCH_LIKE_SCHEMA
        )
        assert sdk_tool.input_schema["required"] == ["query"]

    async def test_final_advertised_schema_survives_create_sdk_mcp_server_unmodified(self) -> None:
        """End-to-end: create_sdk_mcp_server's own schema builder must pass our full schema
        through VERBATIM (its required-everything fallback only fires for a bare {name: type}
        map) -- this is what the model actually sees when a turn starts."""
        from claude_agent_sdk import create_sdk_mcp_server
        from mcp.types import ListToolsRequest

        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        sdk_tool = model._wrap_langchain_tool(  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            _EchoTool(), self._MEMORY_SEARCH_LIKE_SCHEMA
        )
        server = create_sdk_mcp_server("test", tools=[sdk_tool])
        handler = server["instance"].request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list"))

        advertised = result.root.tools[0].inputSchema
        assert advertised["required"] == ["query"]
        assert advertised["properties"]["ids"]["type"] == "array"


class Shot(BaseModel):
    """One camera shot."""

    prompt: str = Field(description="what the shot shows")
    seconds: int | None = Field(default=None, description="how long it runs")


class Lighting(BaseModel):
    """How a scene is lit."""

    key: str = Field(description="the key light")


class Scene(BaseModel):
    """Where a scene happens."""

    location: str = Field(description="the place")
    lighting: Lighting | None = Field(default=None, description="the lighting, when it matters")


class Close(BaseModel):
    """A close-up."""

    subject: str


class Wide(BaseModel):
    """A wide shot."""

    landscape: str


class Node(BaseModel):
    """An outline node."""

    label: str = Field(description="the node's text")
    children: list[Node] = Field(default_factory=list, description="the nodes under this one")


class StoryboardInput(BaseModel):
    shots: list[Shot] = Field(description="the shots, in order")
    scene: Scene = Field(description="the scene they belong to")
    framing: Close | Wide = Field(description="the framing")
    anything: Any = Field(default=None, description="any value at all")
    outline: Node | None = Field(default=None, description="the outline, if there is one")


def _storyboard(**_: Any) -> str:
    return "ok"


def _advertised(tool: BaseTool) -> dict[str, Any]:
    """The input schema the CLI is shown for ``tool``, through the path ``bind_tools`` takes.

    ``_get_tool_schema`` then ``_wrap_langchain_tool`` -- what the base class's ``bind_tools`` calls
    for every tool -- then the SDK's own MCP server, whose tool listing is what the model reads.
    """
    import asyncio

    from claude_agent_sdk import create_sdk_mcp_server
    from mcp.types import ListToolsRequest

    model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
    schema = model._get_tool_schema(tool)  # noqa: SLF001 -- the base class's seam under test  # type: ignore[attr-defined]
    sdk_tool = model._wrap_langchain_tool(tool, schema)  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
    server = create_sdk_mcp_server("test", tools=[sdk_tool])
    handler = server["instance"].request_handlers[ListToolsRequest]
    result = asyncio.run(handler(ListToolsRequest(method="tools/list")))
    advertised: dict[str, Any] = result.root.tools[0].inputSchema
    return advertised


_SHOT = {
    "type": "object",
    "description": "One camera shot.",
    "properties": {
        "prompt": {"type": "string", "description": "what the shot shows"},
        "seconds": {"type": "integer", "description": "how long it runs"},
    },
    "required": ["prompt"],
}


class TestNestedModelSchemas:
    """A tool whose args model nests another model reached the model as a plain string field.

    Pydantic renders a nested model as ``{"$ref": "#/$defs/Shot"}`` with the definition under the
    schema's ``$defs``. The wrapper dropped ``$defs`` and turned every ``$ref`` property into
    ``{"type": "string"}``, and an array whose ``items`` was a ``$ref`` kept a ref to nothing -- so
    ``list[Shot]`` and a sub-object both arrived as strings. The API-key route dereferences them.
    """

    def _tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=_storyboard, name="storyboard", description="plans shots", args_schema=StoryboardInput
        )

    def test_a_list_of_models_is_an_array_of_objects(self) -> None:
        assert _advertised(self._tool())["properties"]["shots"] == {
            "type": "array",
            "description": "the shots, in order",
            "items": _SHOT,
        }

    def test_a_nested_sub_object_keeps_its_properties_required_list_and_field_description(self) -> None:
        scene = _advertised(self._tool())["properties"]["scene"]
        assert scene["type"] == "object"
        assert scene["description"] == "the scene they belong to"
        assert scene["required"] == ["location"]
        assert scene["properties"]["location"] == {"type": "string", "description": "the place"}

    def test_an_optional_nested_model_is_unwrapped_to_the_object(self) -> None:
        lighting = _advertised(self._tool())["properties"]["scene"]["properties"]["lighting"]
        assert lighting == {
            "type": "object",
            "description": "the lighting, when it matters",
            "properties": {"key": {"type": "string", "description": "the key light"}},
            "required": ["key"],
        }

    def test_a_union_of_models_keeps_every_member(self) -> None:
        framing = _advertised(self._tool())["properties"]["framing"]
        assert framing["description"] == "the framing"
        members = framing["anyOf"]
        assert [sorted(member["properties"]) for member in members] == [["subject"], ["landscape"]]

    def test_an_untyped_field_is_not_forced_to_a_string(self) -> None:
        assert _advertised(self._tool())["properties"]["anything"] == {
            "default": None,
            "description": "any value at all",
        }

    def test_no_reference_is_left_dangling(self) -> None:
        advertised = _advertised(self._tool())
        assert "$ref" not in repr(advertised)
        assert "$defs" not in advertised

    def test_the_top_level_required_list_is_the_models(self) -> None:
        assert _advertised(self._tool())["required"] == ["shots", "scene", "framing"]

    def test_a_recursive_model_is_expanded_once_and_its_recursion_named(self) -> None:
        """A recursive model cannot be inlined completely. It is expanded until it recurs, and
        the point of recursion is described in words rather than cut to an empty schema."""
        outline = _advertised(self._tool())["properties"]["outline"]
        assert outline["type"] == "object"
        assert outline["description"] == "the outline, if there is one"
        children = outline["properties"]["children"]
        assert children["type"] == "array"
        assert children["description"] == "the nodes under this one"
        assert children["items"] == {
            "type": "object",
            "description": "A Node: the same shape as the Node that contains it.",
        }

    def test_a_json_schema_args_schema_is_read_not_replaced_by_an_empty_one(self) -> None:
        """A tool may carry its schema as a JSON Schema dict rather than a pydantic model -- every
        TearsTool wrapped for LangChain does. The base class called ``.model_json_schema()`` on
        it, failed, and advertised a tool with no parameters at all."""
        tool = StructuredTool.from_function(
            func=_storyboard,
            name="storyboard",
            description="plans shots",
            args_schema=StoryboardInput.model_json_schema(),
        )
        assert _advertised(tool)["properties"]["shots"]["items"] == _SHOT

    def test_a_reference_outside_the_schema_is_refused_by_name(self) -> None:
        tool = StructuredTool.from_function(
            func=_storyboard,
            name="storyboard",
            description="plans shots",
            args_schema={
                "type": "object",
                "properties": {"shot": {"$ref": "https://example.com/shot.json"}},
                "required": ["shot"],
            },
        )
        with pytest.raises(ValueError, match=r"storyboard.*https://example.com/shot.json"):
            _advertised(tool)
