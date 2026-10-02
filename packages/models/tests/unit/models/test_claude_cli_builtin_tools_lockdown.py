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

from langchain_core.messages import HumanMessage

from threetears.models import DEFAULT_CHAT_MODEL

from .claude_cli_recorder import TOKEN, sent_to_cli, subscription_model


def _launched_with(**model_kwargs: object) -> object:
    """the options the CLI is launched with for one call of a model built with ``model_kwargs``."""
    return sent_to_cli([HumanMessage(content="hi")], build=lambda: subscription_model(**model_kwargs)).options


class TestBuiltinToolsDefaultDeny:
    def test_default_disables_every_builtin_tool(self) -> None:
        assert _launched_with().tools == []  # type: ignore[attr-defined]

    def test_caller_can_explicitly_opt_a_builtin_back_in(self) -> None:
        assert _launched_with(tools=["WebSearch"]).tools == ["WebSearch"]  # type: ignore[attr-defined]

    def test_an_unrelated_kwarg_does_not_reopen_builtin_tools(self) -> None:
        """Passing some other forwarded kwarg must not accidentally clear the default-deny."""
        assert _launched_with(permission_mode="default").tools == []  # type: ignore[attr-defined]

    def test_bare_constructor_kwarg_on_the_base_class_would_have_been_silently_dropped(self) -> None:
        """Regression guard for the bug caught before this landed: ClaudeCodeChatModel's own
        `extra="ignore"` config silently swallows an undeclared `tools` kwarg -- this is why
        `_SubscriptionChatModel` must declare `tools` as a real field rather than relying on
        `_FORWARDED_KWARGS` membership alone."""
        from langchain_claude_code import ClaudeCodeChatModel

        def base() -> ClaudeCodeChatModel:
            return ClaudeCodeChatModel(model=DEFAULT_CHAT_MODEL, oauth_token=TOKEN, tools=[])

        assert not hasattr(base(), "tools")
        assert sent_to_cli([HumanMessage(content="hi")], build=base).options.tools is None
