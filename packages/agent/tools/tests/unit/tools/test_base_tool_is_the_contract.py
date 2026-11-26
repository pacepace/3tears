"""``threetears.agent.tools.base_tool`` and ``.coercion`` are the contract package's own objects, re-exported.

a tool written against either import path is the same class, so the mesh's ToolServer and the MCP face serve one
definition.
"""

from __future__ import annotations

import threetears.tool_contract as contract
from threetears.agent.tools import base_tool, coercion


def test_every_contract_name_is_the_same_object_under_its_agent_tools_name() -> None:
    for name in base_tool.__all__:
        assert getattr(base_tool, name) is getattr(contract, name), name
    for name in coercion.__all__:
        assert getattr(coercion, name) is getattr(contract, name), name
    assert set(base_tool.__all__) | set(coercion.__all__) == set(contract.__all__)
