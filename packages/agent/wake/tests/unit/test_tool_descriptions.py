"""Drift guard: each wake tool's description must be ≤6 short lines.

Per Requirement TOOL-07 + the project-wide "terse tool descriptions"
learning. The assertion lands here so future edits to the description
strings can't quietly grow them past the human-readability budget.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from threetears.agent.wake.tools.schedule_tools import (
    load_wake_schedule_create_tool,
    load_wake_schedule_delete_tool,
    load_wake_schedule_list_tool,
    load_wake_schedule_pause_tool,
    load_wake_schedule_resume_tool,
    load_wake_schedule_update_tool,
    load_wake_yield_tool,
)
from threetears.agent.wake.tools.webhook_tools import (
    load_webhook_subscription_create_tool,
    load_webhook_subscription_delete_tool,
    load_webhook_subscription_list_tool,
    load_webhook_subscription_pause_tool,
    load_webhook_subscription_resume_tool,
    load_webhook_subscription_rotate_secret_tool,
    load_webhook_subscription_update_tool,
)


MAX_DESC_LINES = 6


# parity-with: threetears.agent.wake.tools.schedule_tools.WakeRegistryClient
class _StubRegistry:
    """No-op registry stub for the description-only smoke loads."""

    async def acl_permits_skill(self, *, user_id, agent_id, skill_id) -> bool:  # type: ignore[no-untyped-def]
        del user_id, agent_id, skill_id
        return True

    async def skill_name_for_id(self, *, user_id, agent_id, skill_id) -> str | None:  # type: ignore[no-untyped-def]
        del user_id, agent_id, skill_id
        return None


# parity-with: threetears.agent.wake.entities.EncryptionService
class _StubEncryption:
    """No-op encryption stub for description loads (never actually called here)."""

    def encrypt(self, plaintext: bytes) -> bytes:
        return bytes(plaintext)

    def decrypt(self, ciphertext: bytes) -> str:
        del ciphertext
        return ""


def _description_lines(tool_obj) -> list[str]:  # type: ignore[no-untyped-def]
    raw = tool_obj.description or ""
    # ignore trailing blank line(s)
    return [line for line in raw.rstrip().splitlines()]


_conv = uuid4()
_user = uuid4()
_agent = uuid4()


def _all_schedule_tools():  # type: ignore[no-untyped-def]
    registry = _StubRegistry()
    tools = []
    # schedules collection is only touched at call-time, not load-time;
    # ``None`` is fine for the description smoke test.
    tools += load_wake_schedule_create_tool(
        conversation_id=_conv,
        user_id=_user,
        agent_id=_agent,
        schedules_collection=None,  # type: ignore[arg-type]
        registry=registry,
    )
    tools += load_wake_schedule_update_tool(
        conversation_id=_conv,
        user_id=_user,
        agent_id=_agent,
        schedules_collection=None,  # type: ignore[arg-type]
        registry=registry,
    )
    tools += load_wake_schedule_list_tool(
        conversation_id=_conv,
        user_id=_user,
        agent_id=_agent,
        schedules_collection=None,  # type: ignore[arg-type]
        registry=registry,
    )
    tools += load_wake_schedule_pause_tool(
        conversation_id=_conv,
        user_id=_user,
        schedules_collection=None,  # type: ignore[arg-type]
    )
    tools += load_wake_schedule_resume_tool(
        conversation_id=_conv,
        user_id=_user,
        schedules_collection=None,  # type: ignore[arg-type]
    )
    tools += load_wake_schedule_delete_tool(
        conversation_id=_conv,
        user_id=_user,
        schedules_collection=None,  # type: ignore[arg-type]
    )
    return tools


def _all_webhook_tools():  # type: ignore[no-untyped-def]
    registry = _StubRegistry()
    enc = _StubEncryption()
    tools = []
    tools += load_webhook_subscription_create_tool(
        conversation_id=_conv,
        user_id=_user,
        agent_id=_agent,
        subscriptions_collection=None,  # type: ignore[arg-type]
        encryption_service=enc,
        registry=registry,
    )
    tools += load_webhook_subscription_update_tool(
        conversation_id=_conv,
        user_id=_user,
        agent_id=_agent,
        subscriptions_collection=None,  # type: ignore[arg-type]
        registry=registry,
    )
    tools += load_webhook_subscription_list_tool(
        conversation_id=_conv,
        user_id=_user,
        agent_id=_agent,
        subscriptions_collection=None,  # type: ignore[arg-type]
        registry=registry,
    )
    tools += load_webhook_subscription_pause_tool(
        conversation_id=_conv,
        user_id=_user,
        subscriptions_collection=None,  # type: ignore[arg-type]
    )
    tools += load_webhook_subscription_resume_tool(
        conversation_id=_conv,
        user_id=_user,
        subscriptions_collection=None,  # type: ignore[arg-type]
    )
    tools += load_webhook_subscription_delete_tool(
        conversation_id=_conv,
        user_id=_user,
        subscriptions_collection=None,  # type: ignore[arg-type]
    )
    tools += load_webhook_subscription_rotate_secret_tool(
        conversation_id=_conv,
        user_id=_user,
        subscriptions_collection=None,  # type: ignore[arg-type]
        encryption_service=enc,
    )
    return tools


@pytest.mark.parametrize("tool_obj", _all_schedule_tools() + _all_webhook_tools())
def test_tool_description_is_at_most_six_lines(tool_obj) -> None:  # type: ignore[no-untyped-def]
    lines = _description_lines(tool_obj)
    assert lines, f"tool {tool_obj.name!r} has empty description"
    assert len(lines) <= MAX_DESC_LINES, (
        f"tool {tool_obj.name!r} description has {len(lines)} lines (cap={MAX_DESC_LINES})"
    )


def test_wake_yield_description_loads_on_wake_turn() -> None:
    tools = load_wake_yield_tool(
        is_wake_turn=lambda: True,
        set_yield_requested=lambda: None,
    )
    assert len(tools) == 1
    lines = _description_lines(tools[0])
    assert lines, "wake_yield description is empty"
    assert len(lines) <= MAX_DESC_LINES


def test_wake_yield_refuses_on_non_wake_turn() -> None:
    with pytest.raises(RuntimeError, match="non-wake turn"):
        load_wake_yield_tool(
            is_wake_turn=lambda: False,
            set_yield_requested=lambda: None,
        )
