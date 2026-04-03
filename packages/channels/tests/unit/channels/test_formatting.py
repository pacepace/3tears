"""unit tests for channel response formatting module."""

from __future__ import annotations

import pytest

from threetears.channels.formatting import (
    _plain_text_fallback,
    build_discord_embed,
    build_discord_payload,
    build_slack_blocks,
    build_slack_payload,
    should_use_rich_formatting,
)


class TestShouldUseRichFormatting:
    """tests for should_use_rich_formatting detection."""

    def test_returns_true_for_rich_format_hint(self) -> None:
        result = should_use_rich_formatting({"format": "rich"})
        assert result is True

    def test_returns_true_for_title_hint(self) -> None:
        result = should_use_rich_formatting({"title": "Results"})
        assert result is True

    def test_returns_true_for_color_hint(self) -> None:
        result = should_use_rich_formatting({"color": "#ff0000"})
        assert result is True

    def test_returns_false_for_empty_hints(self) -> None:
        result = should_use_rich_formatting({})
        assert result is False

    def test_returns_false_for_no_rich_keys(self) -> None:
        result = should_use_rich_formatting({"some_other": "value"})
        assert result is False


class TestBuildSlackBlocks:
    """tests for Slack Block Kit block generation."""

    def test_simple_text_becomes_section_block(self) -> None:
        blocks = build_slack_blocks("hello world", {})
        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert blocks[0]["text"]["type"] == "mrkdwn"
        assert blocks[0]["text"]["text"] == "hello world"

    def test_code_block_preserved(self) -> None:
        content = "Here is code:\n\n```python\nprint('hi')\n```"
        blocks = build_slack_blocks(content, {})
        code_found = False
        for block in blocks:
            if "```" in block.get("text", {}).get("text", ""):
                code_found = True
                break
        assert code_found is True

    def test_title_adds_header_block(self) -> None:
        blocks = build_slack_blocks("content", {"title": "Foo"})
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["text"] == "Foo"

    def test_long_text_split_into_multiple_blocks(self) -> None:
        # single paragraph exceeding 3000 chars
        content = "x" * 6500
        blocks = build_slack_blocks(content, {})
        assert len(blocks) >= 2
        for block in blocks:
            assert len(block["text"]["text"]) <= 3000

    def test_multiple_paragraphs_become_multiple_blocks(self) -> None:
        content = "para1\n\npara2"
        blocks = build_slack_blocks(content, {})
        assert len(blocks) == 2
        assert blocks[0]["text"]["text"] == "para1"
        assert blocks[1]["text"]["text"] == "para2"


class TestBuildDiscordEmbed:
    """tests for Discord embed generation."""

    def test_content_becomes_description(self) -> None:
        embed = build_discord_embed("hello **bold**", {})
        assert embed["description"] == "hello **bold**"

    def test_title_from_format_hints(self) -> None:
        embed = build_discord_embed("content", {"title": "My Title"})
        assert embed["title"] == "My Title"

    def test_color_parsed_from_hex(self) -> None:
        embed = build_discord_embed("content", {"color": "#36a64f"})
        assert embed["color"] == int("36a64f", 16)

    def test_long_content_truncated(self) -> None:
        content = "y" * 5000
        embed = build_discord_embed(content, {})
        assert len(embed["description"]) <= 4096
        assert embed["description"].endswith("[truncated]")

    def test_no_title_or_color_minimal_embed(self) -> None:
        embed = build_discord_embed("just text", {})
        assert "description" in embed
        assert "title" not in embed
        assert "color" not in embed


class TestBuildSlackPayload:
    """tests for full Slack payload construction."""

    def test_includes_text_fallback(self) -> None:
        payload = build_slack_payload("hello", {"format": "rich"}, "C123", None)
        assert "text" in payload
        assert isinstance(payload["text"], str)

    def test_includes_blocks(self) -> None:
        payload = build_slack_payload("hello", {"format": "rich"}, "C123", None)
        assert "blocks" in payload
        assert isinstance(payload["blocks"], list)

    def test_thread_ts_included_when_present(self) -> None:
        payload = build_slack_payload("hello", {}, "C123", "1234.5678")
        assert payload["thread_ts"] == "1234.5678"

    def test_thread_ts_absent_when_none(self) -> None:
        payload = build_slack_payload("hello", {}, "C123", None)
        assert "thread_ts" not in payload

    def test_channel_set(self) -> None:
        payload = build_slack_payload("hello", {}, "C999", None)
        assert payload["channel"] == "C999"


class TestBuildDiscordPayload:
    """tests for full Discord payload construction."""

    def test_has_embeds_array(self) -> None:
        payload = build_discord_payload("content", {"format": "rich"})
        assert "embeds" in payload
        assert isinstance(payload["embeds"], list)
        assert len(payload["embeds"]) == 1

    def test_content_empty_with_embed(self) -> None:
        payload = build_discord_payload("content", {"format": "rich"})
        assert payload["content"] == ""


class TestPlainTextFallback:
    """tests for markdown stripping fallback."""

    def test_strips_markdown(self) -> None:
        result = _plain_text_fallback("**bold** and _italic_")
        assert "**" not in result
        assert result == "bold and italic"

    def test_truncates_long_text(self) -> None:
        long_text = "a" * 500
        result = _plain_text_fallback(long_text, max_length=100)
        assert len(result) <= 100
        assert result.endswith("...")

    def test_strips_heading_prefix(self) -> None:
        result = _plain_text_fallback("## Heading")
        assert result == "Heading"

    def test_strips_code_block_markers(self) -> None:
        result = _plain_text_fallback("```python\ncode\n```")
        assert "```" not in result
