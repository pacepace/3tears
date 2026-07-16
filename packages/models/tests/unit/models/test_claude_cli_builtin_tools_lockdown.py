"""Tests for the subscription-backend built-in-tools default-deny (claude-max-convergence Chunk 8).

``ClaudeAgentOptions.tools`` gates whether ANY Claude Code built-in tool (Bash, Read, Write, Edit,
WebFetch, WebSearch, ...) is available at all -- independent of ``allowed_tools``/``disallowed_tools``,
which only gate auto-approval / removal of tools ``tools`` already made available.
``ClaudeCodeChatModel`` declares no ``tools`` field and has ``extra="ignore"``, so a bare
``tools=...`` constructor kwarg is silently dropped -- ``_SubscriptionChatModel`` declares its own
``tools`` field (default ``[]``, default-deny) and overrides ``_build_options`` to forward it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.providers._claude_cli import create_subscription_chat

_FAKE_TOKEN = "sk-ant-oat01-faketokenfortest"


class TestBuiltinToolsDefaultDeny:
    def test_default_disables_every_builtin_tool(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _FAKE_TOKEN)
        options = model._build_options()  # noqa: SLF001 -- the method under test
        assert options.tools == []

    def test_caller_can_explicitly_opt_a_builtin_back_in(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _FAKE_TOKEN, tools=["WebSearch"])
        options = model._build_options()  # noqa: SLF001
        assert options.tools == ["WebSearch"]

    def test_an_unrelated_kwarg_does_not_reopen_builtin_tools(self) -> None:
        """Passing some other forwarded kwarg must not accidentally clear the default-deny."""
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _FAKE_TOKEN, permission_mode="default")
        options = model._build_options()  # noqa: SLF001
        assert options.tools == []

    def test_bare_constructor_kwarg_on_the_base_class_would_have_been_silently_dropped(self) -> None:
        """Regression guard for the bug caught before this landed: ClaudeCodeChatModel's own
        `extra="ignore"` config silently swallows an undeclared `tools` kwarg -- this is why
        `_SubscriptionChatModel` must declare `tools` as a real field rather than relying on
        `_FORWARDED_KWARGS` membership alone."""
        from langchain_claude_code import ClaudeCodeChatModel

        base = ClaudeCodeChatModel(model=DEFAULT_CHAT_MODEL, oauth_token=_FAKE_TOKEN, tools=[])
        assert not hasattr(base, "tools")
        assert base._build_options().tools is None  # noqa: SLF001
