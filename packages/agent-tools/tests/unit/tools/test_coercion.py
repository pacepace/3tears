"""unit tests for LLM-flavored input coercion in TearsTool base.

locks in behavior of ``coerce_value`` and ``normalize_kwargs`` so that
loose LLM-supplied argument shapes (empty strings, JSON-serialized
containers) are rewritten to match each parameter's declared JSON
schema type before being dispatched to subclass ``_execute``.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.agent.tools._coercion import coerce_value, normalize_kwargs
from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)


class _CoercionFixtureTool(TearsTool):
    """concrete TearsTool with an input schema exercising object + array types.

    used by ``normalize_kwargs`` and end-to-end tests. ``_execute`` simply
    records the kwargs it receives so tests can assert on post-coercion
    shapes. optional ``schema_error`` attribute lets tests simulate a
    broken ``mcp_schema``.
    """

    def __init__(self, schema_error: bool = False) -> None:
        """initialize fixture tool.

        :param schema_error: when True, mcp_schema raises to test fallback
        :ptype schema_error: bool
        """
        self._schema_error = schema_error
        self.received: dict[str, Any] = {}

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """record received kwargs and return success.

        :param kwargs: post-coercion tool input parameters
        :ptype kwargs: Any
        :return: success result
        :rtype: ToolResult
        """
        self.received = kwargs
        return ToolResult(success=True, content="ok")

    def mcp_schema(self) -> MCPToolDefinition:
        """return fixture schema or raise when configured.

        :return: tool definition
        :rtype: MCPToolDefinition
        :raises RuntimeError: when schema_error flag set
        """
        if self._schema_error:
            raise RuntimeError("schema boom")
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="coercion fixture",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "connection_config": {"type": "object"},
                    "allowed_schemas": {"type": "array"},
                    "count": {"type": "integer"},
                    "label": {"type": "string"},
                },
                "required": ["action"],
            },
        )

    def mcp_name(self) -> str:
        """return fixture tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.coercion_fixture"

    def mcp_version(self) -> str:
        """return fixture tool version.

        :return: version string
        :rtype: str
        """
        return "1.0.0"


class TestCoerceValueObject:
    """``coerce_value`` rewrites loose inputs toward ``type: "object"``."""

    def test_empty_string_becomes_empty_dict(self) -> None:
        """verify empty string coerces to empty dict for object-typed field."""
        assert coerce_value("", "object") == {}

    def test_empty_json_object_string_parses(self) -> None:
        """verify ``"{}"`` parses to empty dict for object-typed field."""
        assert coerce_value("{}", "object") == {}

    def test_json_object_string_parses_to_dict(self) -> None:
        """verify JSON-encoded object string decodes to native dict."""
        assert coerce_value('{"host":"x"}', "object") == {"host": "x"}

    def test_dict_passthrough(self) -> None:
        """verify already-native dict passes through unchanged."""
        value = {"host": "x"}
        assert coerce_value(value, "object") == {"host": "x"}

    def test_invalid_json_passthrough(self) -> None:
        """verify invalid JSON string is returned unchanged.

        surfacing the original value lets downstream validation produce
        a clear 422 rather than silently swallowing malformed input.
        """
        assert coerce_value("not-json{", "object") == "not-json{"

    def test_json_string_scalar_stays_unchanged(self) -> None:
        """verify JSON string that parses to non-dict is not coerced.

        ``'"not-a-dict"'`` decodes to the string ``"not-a-dict"`` which
        is not safely usable as an object, so the original value is
        preserved for downstream validation to reject explicitly.
        """
        assert coerce_value('"not-a-dict"', "object") == '"not-a-dict"'


class TestCoerceValueArray:
    """``coerce_value`` rewrites loose inputs toward ``type: "array"``."""

    def test_empty_string_becomes_empty_list(self) -> None:
        """verify empty string coerces to empty list for array-typed field."""
        assert coerce_value("", "array") == []

    def test_empty_json_array_string_parses(self) -> None:
        """verify ``"[]"`` parses to empty list for array-typed field."""
        assert coerce_value("[]", "array") == []

    def test_json_array_string_parses_to_list(self) -> None:
        """verify JSON-encoded array string decodes to native list."""
        assert coerce_value('["a","b"]', "array") == ["a", "b"]

    def test_list_passthrough(self) -> None:
        """verify already-native list passes through unchanged."""
        value = ["a"]
        assert coerce_value(value, "array") == ["a"]

    def test_invalid_json_passthrough(self) -> None:
        """verify invalid JSON string is returned unchanged."""
        assert coerce_value("[incomplete", "array") == "[incomplete"


class TestCoerceValueOtherTypes:
    """``coerce_value`` leaves non-object/array declared types alone."""

    def test_string_type_passthrough(self) -> None:
        """verify string-typed values are not coerced."""
        assert coerce_value("hello", "string") == "hello"

    def test_integer_type_passthrough(self) -> None:
        """verify integer-typed values are not coerced."""
        assert coerce_value(42, "integer") == 42

    def test_none_declared_type_passthrough(self) -> None:
        """verify values with no declared type are left alone."""
        assert coerce_value("anything", None) == "anything"


class TestNormalizeKwargsNoneHandling:
    """``normalize_kwargs`` preserves explicit ``None`` values.

    preserving ``None`` is critical so per-action required-field checks
    like ``if not connection_config`` continue to fire and raise
    ``ValueError`` rather than being silently replaced by empty
    containers.
    """

    def test_none_preserved_for_object_field(self) -> None:
        """verify ``None`` for object-typed field stays ``None``."""
        tool = _CoercionFixtureTool()
        schema = tool.mcp_schema()
        normalized = normalize_kwargs(
            {"action": "create", "connection_config": None},
            schema.input_schema,
        )
        assert normalized["connection_config"] is None

    def test_none_preserved_for_array_field(self) -> None:
        """verify ``None`` for array-typed field stays ``None``."""
        tool = _CoercionFixtureTool()
        schema = tool.mcp_schema()
        normalized = normalize_kwargs(
            {"action": "create", "allowed_schemas": None},
            schema.input_schema,
        )
        assert normalized["allowed_schemas"] is None


class TestNormalizeKwargsUnknownKey:
    """``normalize_kwargs`` does not invent coercion for unknown keys."""

    def test_unknown_key_passthrough(self) -> None:
        """verify key absent from schema is left alone."""
        tool = _CoercionFixtureTool()
        schema = tool.mcp_schema()
        normalized = normalize_kwargs(
            {"action": "create", "wholly_unknown": "value"},
            schema.input_schema,
        )
        assert normalized["wholly_unknown"] == "value"


class TestEndToEndCoercionThroughTearsTool:
    """LLM-flavored inputs reach subclass ``_execute`` as native containers."""

    async def test_json_string_object_and_array_are_decoded(self) -> None:
        """verify JSON-string inputs for object/array fields are decoded.

        exercises the full ``execute`` template method: normalization
        engages before dispatch, so ``_execute`` receives native
        ``dict`` and ``list`` rather than their JSON-encoded forms.
        """
        tool = _CoercionFixtureTool()
        result = await tool.execute(
            action="create",
            connection_config='{"host":"x","port":5432}',
            allowed_schemas='["public","analytics"]',
        )
        assert result.success is True
        assert tool.received["connection_config"] == {"host": "x", "port": 5432}
        assert isinstance(tool.received["connection_config"], dict)
        assert tool.received["allowed_schemas"] == ["public", "analytics"]
        assert isinstance(tool.received["allowed_schemas"], list)

    async def test_empty_string_array_becomes_empty_list(self) -> None:
        """verify empty string for array-typed field reaches ``_execute`` as ``[]``."""
        tool = _CoercionFixtureTool()
        result = await tool.execute(
            action="create",
            connection_config={"host": "x"},
            allowed_schemas="",
        )
        assert result.success is True
        assert tool.received["allowed_schemas"] == []
        assert isinstance(tool.received["allowed_schemas"], list)

    async def test_native_dict_and_list_pass_through_unchanged(self) -> None:
        """verify native container inputs are not perturbed by normalization."""
        tool = _CoercionFixtureTool()
        cfg = {"host": "x", "port": 5432}
        schemas = ["public"]
        result = await tool.execute(
            action="create",
            connection_config=cfg,
            allowed_schemas=schemas,
        )
        assert result.success is True
        assert tool.received["connection_config"] == cfg
        assert tool.received["allowed_schemas"] == schemas


class TestSchemaFailureFallback:
    """``TearsTool.execute`` falls back to raw kwargs if ``mcp_schema`` raises.

    defensive behavior: a subclass whose ``mcp_schema`` is broken must
    not prevent ``execute`` from running. the template method silently
    falls back to the original kwargs so subclass ``_execute`` still
    runs.
    """

    async def test_schema_exception_falls_back_to_raw_kwargs(self) -> None:
        """verify kwargs pass through untouched when schema lookup raises."""
        tool = _CoercionFixtureTool(schema_error=True)
        raw = {"action": "x", "payload": '{"y":1}'}
        result = await tool.execute(**raw)
        assert result.success is True
        assert tool.received == raw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
