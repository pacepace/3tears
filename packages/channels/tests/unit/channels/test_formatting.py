"""unit tests for channel response formatting module."""

from __future__ import annotations


from threetears.channels.formatting import (
    build_discord_embed,
    build_discord_payload,
    build_slack_blocks,
    build_slack_payload,
    markdown_to_slack_blocks,
    plain_text_fallback,
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
        result = plain_text_fallback("**bold** and _italic_")
        assert "**" not in result
        assert result == "bold and italic"

    def test_truncates_long_text(self) -> None:
        long_text = "a" * 500
        result = plain_text_fallback(long_text, max_length=100)
        assert len(result) <= 100
        assert result.endswith("...")

    def test_strips_heading_prefix(self) -> None:
        result = plain_text_fallback("## Heading")
        assert result == "Heading"

    def test_strips_code_block_markers(self) -> None:
        result = plain_text_fallback("```python\ncode\n```")
        assert "```" not in result


class TestMarkdownToSlackBlocks:
    """tests for markdown_to_slack_blocks (GitHub markdown -> native Slack)."""

    def test_plain_prose_becomes_one_mrkdwn_section(self) -> None:
        blocks = markdown_to_slack_blocks("hello world")
        assert blocks == [
            {"type": "section", "text": {"type": "mrkdwn", "text": "hello world"}},
        ]

    def test_bold_converts_double_star_to_single_star(self) -> None:
        blocks = markdown_to_slack_blocks("this is **important** text")
        assert blocks[0]["text"]["text"] == "this is *important* text"

    def test_bold_underscores_convert_to_single_star(self) -> None:
        blocks = markdown_to_slack_blocks("this is __important__ text")
        assert blocks[0]["text"]["text"] == "this is *important* text"

    def test_italic_single_star_converts_to_underscore(self) -> None:
        blocks = markdown_to_slack_blocks("this is *subtle* text")
        assert blocks[0]["text"]["text"] == "this is _subtle_ text"

    def test_link_converts_to_slack_angle_form(self) -> None:
        blocks = markdown_to_slack_blocks("see [the docs](https://example.com/x)")
        assert blocks[0]["text"]["text"] == "see <https://example.com/x|the docs>"

    def test_strikethrough_converts_to_single_tilde(self) -> None:
        blocks = markdown_to_slack_blocks("this is ~~gone~~ now")
        assert blocks[0]["text"]["text"] == "this is ~gone~ now"

    def test_dash_bullets_become_slack_bullets(self) -> None:
        blocks = markdown_to_slack_blocks("- one\n- two\n- three")
        text = blocks[0]["text"]["text"]
        assert text == "• one\n• two\n• three"

    def test_star_bullets_become_slack_bullets(self) -> None:
        blocks = markdown_to_slack_blocks("* alpha\n* beta")
        assert blocks[0]["text"]["text"] == "• alpha\n• beta"

    def test_ordered_list_is_preserved(self) -> None:
        blocks = markdown_to_slack_blocks("1. first\n2. second")
        assert blocks[0]["text"]["text"] == "1. first\n2. second"

    def test_header_becomes_header_block_plain_text(self) -> None:
        blocks = markdown_to_slack_blocks("## Results")
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["type"] == "plain_text"
        assert blocks[0]["text"]["text"] == "Results"

    def test_header_strips_inline_markdown(self) -> None:
        blocks = markdown_to_slack_blocks("# The **big** result")
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["text"] == "The big result"

    def test_horizontal_rule_becomes_divider(self) -> None:
        blocks = markdown_to_slack_blocks("above\n\n---\n\nbelow")
        types = [b["type"] for b in blocks]
        assert "divider" in types

    def test_fenced_code_block_becomes_mrkdwn_section(self) -> None:
        content = "```python\nprint('hi')\n```"
        blocks = markdown_to_slack_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert blocks[0]["text"]["type"] == "mrkdwn"
        assert blocks[0]["text"]["text"] == "```\nprint('hi')\n```"

    def test_blockquote_is_preserved(self) -> None:
        blocks = markdown_to_slack_blocks("> a quoted line")
        assert blocks[0]["text"]["text"] == "> a quoted line"

    def test_table_becomes_native_table_block(self) -> None:
        content = "| Name | Revenue |\n| --- | --- |\n| Acme | 1200 |\n| Globex | 980 |"
        blocks = markdown_to_slack_blocks(content)
        assert len(blocks) == 1
        table = blocks[0]
        assert table["type"] == "table"
        # header + 2 data rows.
        assert len(table["rows"]) == 3
        assert table["rows"][0] == [
            {"type": "raw_text", "text": "Name"},
            {"type": "raw_text", "text": "Revenue"},
        ]
        assert table["rows"][1] == [
            {"type": "raw_text", "text": "Acme"},
            {"type": "raw_text", "text": "1200"},
        ]

    def test_table_numeric_column_is_right_aligned(self) -> None:
        content = "| Name | Revenue |\n| --- | --- |\n| Acme | 1200 |\n| Globex | 980 |"
        table = markdown_to_slack_blocks(content)[0]
        settings = table["column_settings"]
        assert settings[0] == {"align": "left"}  # names
        assert settings[1] == {"align": "right"}  # revenue

    def test_table_cell_inline_markdown_is_stripped(self) -> None:
        content = "| Name | Note |\n| --- | --- |\n| **Acme** | [link](http://x) |"
        table = markdown_to_slack_blocks(content)[0]
        assert table["rows"][1][0] == {"type": "raw_text", "text": "Acme"}
        assert table["rows"][1][1] == {"type": "raw_text", "text": "link"}

    def test_mixed_document_preserves_order(self) -> None:
        content = (
            "# Summary\n"
            "\n"
            "Here are the **top** accounts:\n"
            "\n"
            "| Account | Spend |\n"
            "| --- | --- |\n"
            "| Acme | 1200 |\n"
            "\n"
            "- follow up with Acme\n"
        )
        blocks = markdown_to_slack_blocks(content)
        types = [b["type"] for b in blocks]
        assert types == ["header", "section", "table", "section"]
        assert blocks[1]["text"]["text"] == "Here are the *top* accounts:"
        assert blocks[3]["text"]["text"] == "• follow up with Acme"

    def test_blocks_are_capped_at_slack_limit(self) -> None:
        # 200 header lines -> 200 candidate header blocks, capped to 50.
        content = "\n\n".join(f"# H{i}" for i in range(200))
        blocks = markdown_to_slack_blocks(content)
        assert len(blocks) == 50

    def test_empty_content_yields_no_blocks(self) -> None:
        assert markdown_to_slack_blocks("") == []
        assert markdown_to_slack_blocks("   \n  \n") == []
