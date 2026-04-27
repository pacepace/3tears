"""response formatting for channel adapters.

converts agent markdown responses into platform-specific rich
payloads: Slack Block Kit blocks and Discord embeds. falls back
to plain text when no format hints are present.
"""

from __future__ import annotations

import re
from typing import Any

from threetears.observe import get_logger

__all__ = [
    "build_discord_embed",
    "build_discord_payload",
    "build_slack_blocks",
    "build_slack_payload",
    "should_use_rich_formatting",
]

log = get_logger(__name__)

_SLACK_BLOCK_TEXT_LIMIT = 3000
_DISCORD_EMBED_DESCRIPTION_LIMIT = 4096
_RICH_FORMAT_KEYS = {"format", "title", "color"}


def should_use_rich_formatting(format_hints: dict[str, Any]) -> bool:
    """determine whether rich formatting should be used for response.

    returns True if format_hints contain recognized rich formatting
    keys (format=rich, title, or color).

    :param format_hints: agent-provided formatting preferences
    :ptype format_hints: dict[str, Any]
    :return: True if rich formatting should be applied
    :rtype: bool
    """
    if format_hints.get("format") == "rich":
        return True

    result = bool(_RICH_FORMAT_KEYS & set(format_hints.keys()))
    return result


def build_slack_blocks(content: str, format_hints: dict[str, Any]) -> list[dict[str, Any]]:
    """build Slack Block Kit blocks from markdown content.

    splits content into paragraphs and converts each into section
    blocks with mrkdwn type. code blocks are preserved as-is since
    Slack renders triple-backtick blocks natively. long text sections
    are split at 3000-char Slack limit.

    :param content: markdown text from agent response
    :ptype content: str
    :param format_hints: agent-provided formatting preferences
    :ptype format_hints: dict[str, Any]
    :return: list of Slack Block Kit block dicts
    :rtype: list[dict[str, Any]]
    """
    blocks: list[dict[str, Any]] = []

    title = format_hints.get("title")
    if title:
        blocks.append(
            {
                "type": "header",
                "text": {"type": "plain_text", "text": str(title)},
            }
        )

    paragraphs = _split_into_paragraphs(content)
    for paragraph in paragraphs:
        text_chunks = _split_long_text(paragraph, _SLACK_BLOCK_TEXT_LIMIT)
        for chunk in text_chunks:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": chunk},
                }
            )

    result = blocks
    return result


def build_discord_embed(content: str, format_hints: dict[str, Any]) -> dict[str, Any]:
    """build Discord embed from markdown content.

    Discord renders markdown natively in embed descriptions.
    truncates content exceeding 4096-char embed description limit.

    :param content: markdown text from agent response
    :ptype content: str
    :param format_hints: agent-provided formatting preferences
    :ptype format_hints: dict[str, Any]
    :return: Discord embed object dict
    :rtype: dict[str, Any]
    """
    description = content
    if len(description) > _DISCORD_EMBED_DESCRIPTION_LIMIT:
        truncation_marker = " [truncated]"
        cut_point = _DISCORD_EMBED_DESCRIPTION_LIMIT - len(truncation_marker)
        description = description[:cut_point] + truncation_marker

    embed: dict[str, Any] = {"description": description}

    title = format_hints.get("title")
    if title:
        embed["title"] = str(title)

    color_hex = format_hints.get("color")
    if color_hex and isinstance(color_hex, str):
        parsed_color = _parse_hex_color(color_hex)
        if parsed_color is not None:
            embed["color"] = parsed_color

    result = embed
    return result


def build_slack_payload(
    content: str,
    format_hints: dict[str, Any],
    channel: str,
    thread_ts: str | None,
) -> dict[str, Any]:
    """build full Slack chat.postMessage payload with Block Kit blocks.

    always includes plain text fallback in text field for push
    notifications and screen readers.

    :param content: markdown text from agent response
    :ptype content: str
    :param format_hints: agent-provided formatting preferences
    :ptype format_hints: dict[str, Any]
    :param channel: Slack channel ID
    :ptype channel: str
    :param thread_ts: thread timestamp for reply threading, None for top-level
    :ptype thread_ts: str | None
    :return: Slack API chat.postMessage payload dict
    :rtype: dict[str, Any]
    """
    payload: dict[str, Any] = {
        "channel": channel,
        "text": _plain_text_fallback(content),
        "blocks": build_slack_blocks(content, format_hints),
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts

    result = payload
    return result


def build_discord_payload(content: str, format_hints: dict[str, Any]) -> dict[str, Any]:
    """build full Discord message payload with embed.

    sets content to empty string when using embeds so Discord
    displays only the rich embed.

    :param content: markdown text from agent response
    :ptype content: str
    :param format_hints: agent-provided formatting preferences
    :ptype format_hints: dict[str, Any]
    :return: Discord API message payload dict
    :rtype: dict[str, Any]
    """
    result: dict[str, Any] = {
        "content": "",
        "embeds": [build_discord_embed(content, format_hints)],
    }
    return result


def _plain_text_fallback(content: str, max_length: int = 300) -> str:
    """strip markdown formatting for Slack notification fallback text.

    removes common markdown syntax characters and truncates to
    max_length for use in push notifications and screen readers.

    :param content: markdown text to strip
    :ptype content: str
    :param max_length: maximum length of fallback text
    :ptype max_length: int
    :return: plain text fallback string
    :rtype: str
    """
    stripped = content
    # remove code block markers
    stripped = stripped.replace("```", "")
    # remove bold markers
    stripped = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
    # remove single bold/italic markers
    stripped = re.sub(r"\*(.+?)\*", r"\1", stripped)
    # remove italic underscores
    stripped = re.sub(r"_(.+?)_", r"\1", stripped)
    # remove inline code
    stripped = re.sub(r"`(.+?)`", r"\1", stripped)
    # remove heading prefixes
    stripped = re.sub(r"^#{1,6}\s+", "", stripped, flags=re.MULTILINE)

    stripped = stripped.strip()
    if len(stripped) > max_length:
        stripped = stripped[: max_length - 3] + "..."

    result = stripped
    return result


def _split_long_text(text: str, max_length: int = 3000) -> list[str]:
    """split text exceeding max_length into chunks.

    splits on paragraph boundaries first, then newlines, then
    performs hard split if needed.

    :param text: text to split
    :ptype text: str
    :param max_length: maximum length per chunk
    :ptype max_length: int
    :return: list of text chunks each within max_length
    :rtype: list[str]
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    # try splitting on paragraph boundaries
    paragraphs = text.split("\n\n")
    current = ""

    for paragraph in paragraphs:
        candidate = current + "\n\n" + paragraph if current else paragraph
        if len(candidate) <= max_length:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # paragraph itself may exceed limit
            if len(paragraph) <= max_length:
                current = paragraph
            else:
                # split on newlines within paragraph
                sub_chunks = _split_on_newlines(paragraph, max_length)
                chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1] if sub_chunks else ""

    if current:
        chunks.append(current)

    result = chunks
    return result


def _split_on_newlines(text: str, max_length: int) -> list[str]:
    """split text on newline boundaries within max_length.

    falls back to hard splitting when individual lines exceed limit.

    :param text: text to split on newlines
    :ptype text: str
    :param max_length: maximum length per chunk
    :ptype max_length: int
    :return: list of text chunks
    :rtype: list[str]
    """
    lines = text.split("\n")
    chunks: list[str] = []
    current = ""

    for line in lines:
        candidate = current + "\n" + line if current else line
        if len(candidate) <= max_length:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(line) <= max_length:
                current = line
            else:
                # hard split
                while len(line) > max_length:
                    chunks.append(line[:max_length])
                    line = line[max_length:]
                current = line

    if current:
        chunks.append(current)

    result = chunks
    return result


def _split_into_paragraphs(content: str) -> list[str]:
    """split content into non-empty paragraphs on double newlines.

    :param content: text content to split
    :ptype content: str
    :return: list of non-empty paragraph strings
    :rtype: list[str]
    """
    parts = content.split("\n\n")
    result = [p for p in parts if p.strip()]
    return result


def _parse_hex_color(color_str: str) -> int | None:
    """parse hex color string to integer.

    accepts formats: "#36a64f", "36a64f", "#FFF", "FFF".

    :param color_str: hex color string
    :ptype color_str: str
    :return: integer color value or None on parse failure
    :rtype: int | None
    """
    cleaned = color_str.lstrip("#")
    try:
        result: int | None = int(cleaned, 16)
    except ValueError:
        result = None
    return result
