"""unit tests for :mod:`threetears.mcp.tool` -- registry + decorator."""

from __future__ import annotations

import pytest
from threetears.mcp.tool import (
    McpTool,
    ToolRegistry,
    register_tool,
)


class TestToolRegistry:
    """direct registry API: register, get, all, names, dup-detection."""

    def test_register_then_get_returns_same_descriptor(self) -> None:
        """get(name) returns the registered tool."""
        registry = ToolRegistry()

        async def handler(**_kwargs: object) -> str:
            return "ok"

        tool = McpTool(
            name="probe",
            description="probe",
            input_schema={"type": "object"},
            required_permission="t.probe.read",
            handler=handler,
        )
        registry.register(tool)
        assert registry.get("probe") is tool

    def test_get_returns_none_for_unknown(self) -> None:
        """missing tool yields None, not KeyError."""
        registry = ToolRegistry()
        assert registry.get("nonesuch") is None

    def test_register_duplicate_raises(self) -> None:
        """registering the same name twice raises ValueError."""
        registry = ToolRegistry()

        async def handler(**_kwargs: object) -> str:
            return "ok"

        tool = McpTool(
            name="dup",
            description="d",
            input_schema={},
            required_permission="t.dup.read",
            handler=handler,
        )
        registry.register(tool)
        with pytest.raises(ValueError, match="conflict"):
            registry.register(tool)

    def test_all_preserves_registration_order(self) -> None:
        """all() returns tools in insertion order (dict-preserving)."""
        registry = ToolRegistry()

        async def handler(**_kwargs: object) -> str:
            return "ok"

        for name in ("alpha", "beta", "gamma"):
            registry.register(
                McpTool(
                    name=name,
                    description=name,
                    input_schema={},
                    required_permission=f"t.{name}.read",
                    handler=handler,
                ),
            )
        assert [t.name for t in registry.all()] == ["alpha", "beta", "gamma"]
        assert registry.names() == ["alpha", "beta", "gamma"]


class TestRegisterToolDecorator:
    """``@register_tool(...)`` happy path + isolation behaviour."""

    def test_decorator_registers_against_supplied_registry(self) -> None:
        """decorator calls ``registry.register`` with full McpTool."""
        registry = ToolRegistry()

        @register_tool(
            name="probe",
            description="probe tool",
            input_schema={"type": "object", "properties": {}},
            required_permission="t.probe.read",
            registry=registry,
        )
        async def handler(**_kwargs: object) -> str:
            return "from-handler"

        recovered = registry.get("probe")
        assert recovered is not None
        assert recovered.name == "probe"
        assert recovered.description == "probe tool"
        assert recovered.required_permission == "t.probe.read"
        assert recovered.handler is handler

    def test_decorator_returns_handler_unchanged(self) -> None:
        """the wrapped handler is returned as-is for direct call sites."""
        registry = ToolRegistry()

        @register_tool(
            name="probe",
            description="d",
            input_schema={},
            required_permission="t.probe.read",
            registry=registry,
        )
        async def handler(**_kwargs: object) -> str:
            return "raw-handler"

        # decorator did not wrap; the handler is the same callable.
        assert handler.__name__ == "handler"

    def test_decorator_registries_are_isolated(self) -> None:
        """tools registered against registry_a do not leak into registry_b.

        Tests construct their own registries; with no module-level
        default to fall back on, isolation is guaranteed by
        construction. This is the canonical shape post-cleanup.
        """
        registry_a = ToolRegistry()
        registry_b = ToolRegistry()

        @register_tool(
            name="a-only",
            description="d",
            input_schema={},
            required_permission="t.a.read",
            registry=registry_a,
        )
        async def handler_a(**_kwargs: object) -> str:
            return "a"

        assert registry_a.get("a-only") is not None
        assert registry_b.get("a-only") is None
