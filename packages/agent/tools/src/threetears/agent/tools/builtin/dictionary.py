"""Dictionary lookup: the Free Dictionary API, and Wiktionary when it does not answer.

The Free Dictionary API (api.dictionaryapi.dev) is a free community service with
one host. On 2026-10-01 it took connections and never answered, so every lookup
waited out its timeout -- and the lookup was a blocking call inside an async
tool, so each wait stalled every turn on the worker. The lookup is async now,
waits less, and asks Wiktionary's REST API when the first source times out,
cannot be reached or fails on its side. A "no such word" from the first source
is an answer, not a failure: it is not asked again.
"""

from __future__ import annotations

import html
import re
from typing import Any

import httpx
from langchain_core.tools import StructuredTool

from threetears.agent.tools.text_window import window_text

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.utils import tool_error

__all__ = [
    "DictionaryTool",
    "create_dictionary_tool",
]

_MAX_CHARS = 3000

#: How long the first source gets before Wiktionary is asked.
_PRIMARY_TIMEOUT_S = 5.0
_FALLBACK_TIMEOUT_S = 10.0
#: Wikimedia asks every client to say who it is.
_USER_AGENT = "threetears-dictionary/1.1 (https://github.com/pacepace/3tears)"
_TAGS = re.compile(r"<[^>]+>")


def _plain(fragment: str) -> str:
    """Wiktionary's HTML fragment as plain text."""
    return " ".join(html.unescape(_TAGS.sub("", fragment)).split())


def _format_wiktionary(word: str, language: str, data: dict[str, Any]) -> str | None:
    """Wiktionary's definitions in the same shape as the first source's; None when it has none in ``language``."""
    entries = [e for e in data.get(language, []) if e.get("language") != "Translingual"]
    if not entries:
        return None
    parts: list[str] = [word, ""]
    for entry in entries:
        definitions = [d for d in entry.get("definitions", []) if _plain(d.get("definition", ""))][:3]
        if not definitions:
            continue
        parts.append(f"[{str(entry.get('partOfSpeech', '')).lower()}]")
        for i, defn in enumerate(definitions, 1):
            parts.append(f"  {i}. {_plain(defn.get('definition', ''))}")
            examples = [_plain(x) for x in defn.get("examples", []) if _plain(x)]
            if examples:
                parts.append(f"     Example: {examples[0]}")
        parts.append("")
    if len(parts) == 2:
        return None
    parts.append("(from Wiktionary)")
    return window_text("\n".join(parts).strip(), max_chars=_MAX_CHARS).rendered(tool="dictionary")


def _format_entry(data: list[dict[str, Any]]) -> str:
    """Format dictionary API response into readable text."""
    entry = data[0]
    parts: list[str] = []

    # Word + phonetic
    word = entry.get("word", "")
    phonetic = entry.get("phonetic", "")
    header = word
    if phonetic:
        header += f" {phonetic}"
    parts.append(header)
    parts.append("")

    # Meanings
    for meaning in entry.get("meanings", []):
        pos = meaning.get("partOfSpeech", "")
        parts.append(f"[{pos}]")

        definitions = meaning.get("definitions", [])[:3]
        for i, defn in enumerate(definitions, 1):
            parts.append(f"  {i}. {defn.get('definition', '')}")
            example = defn.get("example")
            if example:
                parts.append(f"     Example: {example}")

        synonyms = meaning.get("synonyms", [])[:5]
        if synonyms:
            parts.append(f"  Synonyms: {', '.join(synonyms)}")

        antonyms = meaning.get("antonyms", [])[:5]
        if antonyms:
            parts.append(f"  Antonyms: {', '.join(antonyms)}")

        parts.append("")

    # One entry rarely reaches the bound, but when it does it is windowed like
    # every other long result rather than cut with a phrase of its own.
    return window_text("\n".join(parts).strip(), max_chars=_MAX_CHARS).rendered(tool="dictionary")


async def _from_wiktionary(client: httpx.AsyncClient, word: str, language: str, why: str) -> str:
    """Ask Wiktionary, after the first source failed for ``why``."""
    try:
        resp = await client.get(
            f"https://en.wiktionary.org/api/rest_v1/page/definition/{word}",
            timeout=_FALLBACK_TIMEOUT_S,
            headers={"User-Agent": _USER_AGENT},
        )
        if resp.status_code == 404:
            return f"No definition found for '{word}'"
        resp.raise_for_status()
        found = _format_wiktionary(word, language, resp.json())
        return found if found is not None else f"No definition found for '{word}'"
    except httpx.HTTPStatusError as exc:
        return tool_error("dictionary", "lookup", f"{why}; Wiktionary answered HTTP {exc.response.status_code}")
    except httpx.HTTPError as exc:
        return tool_error("dictionary", "lookup", f"{why}; Wiktionary failed too: {type(exc).__name__}")


async def _lookup(word: str, language: str = "en", *, transport: httpx.AsyncBaseTransport | None = None) -> str:
    """A word's definitions, from the Free Dictionary API or, when it does not answer, Wiktionary.

    :param word: the word
    :ptype word: str
    :param language: ISO 639-1 language code
    :ptype language: str
    :param transport: the HTTP transport, for tests; None uses the network
    :ptype transport: httpx.AsyncBaseTransport | None
    :return: the definitions as text, "No definition found ..." or a tool error
    :rtype: str
    """
    async with httpx.AsyncClient(transport=transport) as client:
        try:
            resp = await client.get(
                f"https://api.dictionaryapi.dev/api/v2/entries/{language}/{word}", timeout=_PRIMARY_TIMEOUT_S
            )
        except httpx.TimeoutException:
            return await _from_wiktionary(client, word, language, "the Free Dictionary API did not answer")
        except httpx.HTTPError as exc:
            return await _from_wiktionary(
                client, word, language, f"the Free Dictionary API failed: {type(exc).__name__}"
            )
        if resp.status_code == 404:
            return f"No definition found for '{word}'"
        if resp.status_code >= 500:
            return await _from_wiktionary(
                client, word, language, f"the Free Dictionary API answered HTTP {resp.status_code}"
            )
        if resp.status_code >= 400:
            return tool_error("dictionary", "lookup", f"HTTP {resp.status_code}")
        try:
            return _format_entry(resp.json())
        except ValueError, LookupError, AttributeError:
            return await _from_wiktionary(
                client, word, language, "the Free Dictionary API answered something unreadable"
            )


def create_dictionary_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a dictionary lookup tool.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the StructuredTool path and the NATS-dispatched ToolServer
    path share :meth:`DictionaryTool.execute` as their single
    execution body. ``config["language"]`` (default ``"en"``) feeds
    :class:`DictionaryTool`'s ``language`` ``__init__`` arg.
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    language = config.get("language", "en")
    return to_langchain_tool(
        DictionaryTool(language=language),
        description=description,
    )


class DictionaryTool(TearsTool):
    """TearsTool wrapper for dictionary lookups: the Free Dictionary API, then Wiktionary.

    looks up word definitions, phonetics, synonyms, and antonyms
    using free dictionary API. configurable language at construction.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "word": {
                "type": "string",
                "description": "word to look up",
            },
        },
        "required": ["word"],
    }

    def __init__(self, language: str = "en", *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """initialize dictionary tool with language.

        :param language: ISO 639-1 language code for lookups
        :ptype language: str
        :param transport: the HTTP transport, for tests; None uses the network
        :ptype transport: httpx.AsyncBaseTransport | None
        """
        self._language = language
        self._transport = transport

    async def execute(self, **kwargs: Any) -> ToolResult:
        """look up word definition.

        :param kwargs: must include 'word' key with word to look up
        :ptype kwargs: Any
        :return: result containing definition or error
        :rtype: ToolResult
        """
        word = kwargs.get("word", "")
        content = await _lookup(word, self._language, transport=self._transport)
        success = not content.startswith("[TOOL ERROR]")
        result = ToolResult(
            success=success,
            content=content,
            error=content if not success else None,
        )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for dictionary.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="look up word definitions, phonetics, synonyms, and antonyms",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.dictionary"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
