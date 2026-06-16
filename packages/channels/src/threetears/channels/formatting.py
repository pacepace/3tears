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
    "markdown_to_slack_blocks",
    "plain_text_fallback",
    "should_use_rich_formatting",
]

log = get_logger(__name__)

_SLACK_BLOCK_TEXT_LIMIT = 3000
_DISCORD_EMBED_DESCRIPTION_LIMIT = 4096
_RICH_FORMAT_KEYS = {"format", "title", "color"}

# Slack Block Kit hard limits (chat.postMessage).
_SLACK_MAX_BLOCKS = 50
_SLACK_MAX_TABLE_ROWS = 100
_SLACK_MAX_TABLE_COLS = 20
_SLACK_HEADER_TEXT_LIMIT = 150
# sentinel used to protect bold markers across the italic conversion pass.
_BOLD_SENTINEL = "\x00"


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


def markdown_to_slack_blocks(content: str) -> list[dict[str, Any]]:
    """convert GitHub-flavored markdown to native Slack Block Kit blocks.

    Slack does not render GitHub markdown -- it uses ``mrkdwn`` (``*bold*``,
    ``_italic_``, ``<url|text>``, ``• `` bullets) and Block Kit blocks, with no
    ``**``/``##``/``| table |`` syntax. this renders the agent's answer into the
    nicest NATIVE representation Slack supports:

    - markdown tables -> native ``table`` blocks (real columns; numeric columns
      right-aligned)
    - ``#`` headers -> ``header`` blocks
    - fenced code blocks + ``>`` quotes -> ``mrkdwn`` sections (Slack renders both)
    - ``**bold**`` -> ``*bold*``, ``[t](u)`` -> ``<u|t>``, ``- ``/``* `` -> ``• ``
    - ``---`` -> dividers; everything else -> ``mrkdwn`` sections

    bounded to Slack's limits (50 blocks / message, 3000 chars / section,
    100 rows x 20 cols / table).

    :param content: agent response in markdown
    :ptype content: str
    :return: Slack Block Kit blocks
    :rtype: list[dict[str, Any]]
    """
    blocks: list[dict[str, Any]] = []
    para: list[str] = []
    lines = content.split("\n")
    total = len(lines)

    def flush_para() -> None:
        if para:
            joined = "\n".join(para).strip()
            if joined:
                for chunk in _split_long_text(joined, _SLACK_BLOCK_TEXT_LIMIT):
                    blocks.append(_mrkdwn_section(chunk))
            para.clear()

    i = 0
    while i < total:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_para()
            code: list[str] = []
            i += 1
            while i < total and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # consume the closing fence (if present)
            for chunk in _split_long_text("\n".join(code), _SLACK_BLOCK_TEXT_LIMIT - 8):
                blocks.append(_mrkdwn_section(f"```\n{chunk}\n```"))
            continue

        if _is_table_row(line) and i + 1 < total and _is_table_separator(lines[i + 1]):
            flush_para()
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < total and _is_table_row(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append(_markdown_table_to_block(table_lines))
            continue

        header = re.match(r"^(#{1,6})\s+(.*\S)\s*$", stripped)
        if header:
            flush_para()
            blocks.append(
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": _strip_inline(header.group(2))[:_SLACK_HEADER_TEXT_LIMIT],
                        "emoji": True,
                    },
                }
            )
            i += 1
            continue

        if re.fullmatch(r"\*{3,}|-{3,}|_{3,}", stripped):
            flush_para()
            blocks.append({"type": "divider"})
            i += 1
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        para.append(_convert_line(line))
        i += 1

    flush_para()
    return blocks[:_SLACK_MAX_BLOCKS]


def _mrkdwn_section(text: str) -> dict[str, Any]:
    """wrap mrkdwn text in a Slack section block.

    :param text: mrkdwn-formatted text
    :ptype text: str
    :return: section block
    :rtype: dict[str, Any]
    """
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _is_table_row(line: str) -> bool:
    """return whether a line is a pipe-delimited markdown table row.

    :param line: source line
    :ptype line: str
    :return: True if it looks like ``| a | b |``
    :rtype: bool
    """
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    """return whether a line is a markdown table header separator.

    :param line: source line
    :ptype line: str
    :return: True if it looks like ``| --- | :--: |``
    :rtype: bool
    """
    s = line.strip()
    if not s.startswith("|"):
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{1,}:?", c) is not None for c in cells if c)


def _split_table_row(line: str) -> list[str]:
    """split a pipe-delimited row into trimmed cell strings.

    :param line: table row line
    :ptype line: str
    :return: cell values
    :rtype: list[str]
    """
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _looks_numeric(text: str) -> bool:
    """return whether a cell value is a number (for right-alignment).

    :param text: cell text
    :ptype text: str
    :return: True if numeric-looking
    :rtype: bool
    """
    cleaned = text.strip()
    return (
        bool(cleaned) and re.fullmatch(r"[\s\d,.\-+%$()]+", cleaned) is not None and any(ch.isdigit() for ch in cleaned)
    )


def _markdown_table_to_block(table_lines: list[str]) -> dict[str, Any]:
    """convert markdown table lines into a native Slack ``table`` block.

    cells are ``raw_text`` (Slack table cells are plain -- inline markdown is
    stripped); columns whose data cells are all numeric are right-aligned.

    :param table_lines: header row, separator row, then data rows
    :ptype table_lines: list[str]
    :return: Slack ``table`` block
    :rtype: dict[str, Any]
    """
    header = _split_table_row(table_lines[0])
    ncols = min(len(header), _SLACK_MAX_TABLE_COLS)
    data = [_split_table_row(r) for r in table_lines[2:]]

    rows: list[list[dict[str, Any]]] = [
        [{"type": "raw_text", "text": _strip_inline(c)} for c in header[:ncols]],
    ]
    for dr in data[: _SLACK_MAX_TABLE_ROWS - 1]:
        padded = (dr + [""] * ncols)[:ncols]
        rows.append(
            [{"type": "raw_text", "text": _strip_inline(c)} for c in padded],
        )

    col_settings: list[dict[str, Any]] = []
    for ci in range(ncols):
        vals = [dr[ci] for dr in data if ci < len(dr) and dr[ci].strip()]
        align = "right" if vals and all(_looks_numeric(v) for v in vals) else "left"
        col_settings.append({"align": align})

    return {"type": "table", "rows": rows, "column_settings": col_settings}


def _convert_line(line: str) -> str:
    """convert one non-block markdown line to mrkdwn (lists, quotes, inline).

    :param line: source line
    :ptype line: str
    :return: mrkdwn line
    :rtype: str
    """
    bullet = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
    if bullet:
        return f"{bullet.group(1)}• {_convert_inline(bullet.group(2))}"
    ordered = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
    if ordered:
        return f"{ordered.group(1)}{ordered.group(2)}. {_convert_inline(ordered.group(3))}"
    if line.lstrip().startswith(">"):
        return line  # blockquote: Slack renders ``>`` natively
    return _convert_inline(line)


def _convert_inline(text: str) -> str:
    """convert inline markdown emphasis / links to Slack mrkdwn.

    ``[t](u)`` -> ``<u|t>``; ``**b**`` / ``__b__`` -> ``*b*``; ``*i*`` -> ``_i_``;
    ``~~s~~`` -> ``~s~``. (inline code spans are left as-is; rare emphasis inside
    them is accepted.)

    :param text: source text
    :ptype text: str
    :return: mrkdwn text
    :rtype: str
    """
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r"<\2|\1>", text)
    text = re.sub(r"\*\*(.+?)\*\*", rf"{_BOLD_SENTINEL}\1{_BOLD_SENTINEL}", text)
    text = re.sub(r"__(.+?)__", rf"{_BOLD_SENTINEL}\1{_BOLD_SENTINEL}", text)
    text = re.sub(r"\*(\S.*?\S|\S)\*", r"_\1_", text)
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)
    return text.replace(_BOLD_SENTINEL, "*")


def _strip_inline(text: str) -> str:
    """strip markdown to plain text (for header blocks + table cells).

    :param text: source text
    :ptype text: str
    :return: plain text
    :rtype: str
    """
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"^#{1,6}\s+", "", text)
    text = re.sub(r"[*_]", "", text)
    return text.strip()


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
        "text": plain_text_fallback(content),
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


def plain_text_fallback(content: str, max_length: int = 300) -> str:
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
