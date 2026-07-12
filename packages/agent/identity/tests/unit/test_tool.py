"""Unit coverage for the identity_propose tool (lifecycle patched)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from threetears.agent.identity.tools import load_identity_propose_tool

pytestmark = pytest.mark.asyncio

_AGENT = UUID("00000000-0000-0000-0000-000000000001")
_USER = UUID("00000000-0000-0000-0000-00000000000b")
_CUST = UUID("00000000-0000-0000-0000-00000000000c")

_PROPOSE = "threetears.agent.identity.tools.propose"


async def _tool():
    tools = await load_identity_propose_tool(_USER, _AGENT, _CUST, MagicMock(), MagicMock())
    return tools[0]


async def test_tier1_return_mentions_awaiting_consent() -> None:
    version = SimpleNamespace(version_id=uuid4())
    with patch(_PROPOSE, new_callable=AsyncMock, return_value=version) as prop:
        tool = await _tool()
        out = await tool.ainvoke(
            {"block_key": "personality", "content": "v2", "rationale": "sharper"}
        )
    assert "awaiting the user's consent" in out
    # user_id + agent bound; proposer + caller are the agent
    kw = prop.await_args.kwargs
    assert kw["user_id"] == _USER and kw["agent_id"] == _AGENT
    assert kw["proposer_agent_id"] == _AGENT and kw["caller_agent_id"] == _AGENT


async def test_tier2_return_mentions_live_now() -> None:
    version = SimpleNamespace(version_id=uuid4())
    with patch(_PROPOSE, new_callable=AsyncMock, return_value=version):
        tool = await _tool()
        out = await tool.ainvoke(
            {"block_key": "self_improvement", "content": "note", "rationale": "r"}
        )
    assert "live now" in out


async def test_unknown_block_key_is_rejected_without_calling_lifecycle() -> None:
    with patch(_PROPOSE, new_callable=AsyncMock) as prop:
        tool = await _tool()
        out = await tool.ainvoke({"block_key": "nope", "content": "x", "rationale": "r"})
    assert "Unknown block_key" in out
    prop.assert_not_awaited()
