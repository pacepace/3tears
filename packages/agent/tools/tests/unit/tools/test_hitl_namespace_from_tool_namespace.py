"""pins the pair: what ``tool_namespace_name`` emits, the hitl builder consumes.

:func:`threetears.core.namespaces.build_hitl_namespace_name` lifts the
tool components out of a tool namespace name by stripping a fixed
``tools.`` prefix, so it is coupled to the shape
:func:`tool_namespace_name` renders and would raise at attach time,
in a deployment, if that shape ever moved. this file asserts no rule
about the spelling of either one -- it only knows that two things
which must agree still do.

the hostile input is not hypothetical: ``ToolManifestEntry.name`` is a
bare ``str`` and nothing validates it, so a name carrying a space, a
``*`` or a ``>`` reaches the namespace builders as written.
"""

from __future__ import annotations

from uuid import UUID

from threetears.agent.tools.server import tool_namespace_name
from threetears.core.namespaces import build_hitl_namespace_name

CUSTOMER = UUID("7f3c9a1d-1111-4111-8111-000000000001")


class TestHitlNamespaceConsumesAToolNamespaceName:
    """the composed pair, exercised rather than reasoned about."""

    def test_a_real_tool_namespace_name_keeps_its_components(self) -> None:
        tool_ns = tool_namespace_name("scrape.zone_alpha", "1.0.0")
        hitl_ns = build_hitl_namespace_name(tool_ns, CUSTOMER)
        assert hitl_ns.split(".") == ["hitl", *tool_ns.split(".")[1:], CUSTOMER.hex]

    def test_a_hostile_mcp_name_cannot_add_a_component(self) -> None:
        tool_ns = tool_namespace_name("scrape zone_alpha.*.>", "1.0.0")
        hitl_ns = build_hitl_namespace_name(tool_ns, CUSTOMER)
        assert len(hitl_ns.split(".")) == len(tool_ns.split(".")) + 1
        assert hitl_ns.split(".")[-1] == CUSTOMER.hex
