"""Dictionary lookup tool using the free dictionaryapi.dev API."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.utils import tool_error

_MAX_CHARS = 3000


class DictionaryInput(BaseModel):
    """Input for the dictionary tool."""

    word: str = Field(description="Word to look up")


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

    result = "\n".join(parts).strip()
    if len(result) > _MAX_CHARS:
        result = result[: _MAX_CHARS - 12] + "\n[Truncated]"
    return result


def _create_lookup_fn(language: str) -> Any:
    """Create a lookup function bound to a language."""

    def _lookup(word: str) -> str:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/{language}/{word}"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url)
            if resp.status_code == 404:
                return f"No definition found for '{word}'"
            resp.raise_for_status()
            return _format_entry(resp.json())
        except httpx.HTTPStatusError as exc:
            return tool_error("dictionary", "lookup", f"HTTP {exc.response.status_code}")
        except Exception as exc:
            return tool_error("dictionary", "lookup", str(exc))

    return _lookup


def create_dictionary_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a dictionary lookup tool."""
    language = config.get("language", "en")
    return StructuredTool.from_function(
        func=_create_lookup_fn(language),
        name="dictionary",
        description=description,
        args_schema=DictionaryInput,
    )
