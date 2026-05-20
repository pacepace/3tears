"""extract text + reasoning per ``AIMessageChunk`` across all observed shapes.

Streaming chat-model chunks arrive in one of three shapes depending on
the provider and whether the model is in reasoning mode:

- **OpenAI / OpenRouter non-reasoning**: ``chunk.content`` is a string
  (possibly empty).
- **OpenAI / OpenRouter reasoning** (deepseek-r1, o1-style models):
  ``chunk.content`` is empty during reasoning;
  ``additional_kwargs["reasoning_content"]`` carries the reasoning text
  per chunk.
- **Anthropic-direct v1**: ``chunk.content`` is ``list[dict]`` with
  block entries ``{"type": "text", "text": "..."}`` and
  ``{"type": "thinking", "thinking": "..."}``.

Every chat-model consumer that drives the model via ``astream`` needs
this introspection once per chunk to route the streamed tokens to the
correct sink (UI text channel vs. reasoning-trace channel). Until v0.9.0
the logic lived inline in metallm's personality node and was duplicated
across that node's main round loop and tool-limit fallback pass; this
module is the canonical 3tears extraction so every consumer
(14-eng-ai-bot, 14-eng-ai-bot-agents, metallm) shares one
implementation.

The function is pure (no I/O, no logging) and lives on the hot
streaming path -- it is invoked per chunk, per stream, so the
implementation deliberately uses simple control flow over a more
elegant pattern-matching shape.
"""

from __future__ import annotations

from typing import NamedTuple

from langchain_core.messages import AIMessageChunk

__all__ = ["ChunkParsed", "parse_chunk"]


class ChunkParsed(NamedTuple):
    """text + reasoning tokens extracted from a single ``AIMessageChunk``.

    :ivar text: accumulated visible-to-user text from ``chunk.content``
        (string form) or from ``type=="text"`` blocks (list form)
    :ptype text: str
    :ivar reasoning: accumulated reasoning trace from
        ``additional_kwargs["reasoning_content"]`` (OpenAI / OpenRouter
        reasoning models) and from ``type=="thinking"`` blocks
        (Anthropic-direct v1)
    :ptype reasoning: str
    """

    text: str
    reasoning: str


def parse_chunk(chunk: AIMessageChunk) -> ChunkParsed:
    """extract text + reasoning tokens from a streaming chat-model chunk.

    Handles the three observed input shapes (string content, list of
    content blocks, ``additional_kwargs["reasoning_content"]``) and
    returns both extracted token strings on every call. Both fields
    are always present and always strings; an empty string means the
    chunk carried no content of that kind. The function is pure: it
    does not mutate ``chunk`` and produces no side effects.

    The introspection covers all production-observed chunk shapes:

    - **string content**: ``chunk.content`` is a ``str`` -- the entire
      string becomes ``text``.
    - **list-of-blocks content** (Anthropic-direct v1): each block is
      a dict with a ``type`` field. ``type=="text"`` blocks contribute
      to ``text`` via ``block["text"]``; ``type=="thinking"`` blocks
      contribute to ``reasoning`` via ``block["thinking"]``. Other
      block types are silently ignored (consistent with current
      production behavior). Multiple blocks of the same type
      concatenate in encounter order.
    - **OpenRouter reasoning content**: ``additional_kwargs`` may
      carry a ``reasoning_content`` string regardless of which content
      shape the provider used; when present it appends to
      ``reasoning``.

    A single chunk can simultaneously carry visible text AND
    reasoning text (e.g. an Anthropic-direct chunk with both a
    ``text`` and a ``thinking`` block, or an OpenRouter chunk with a
    string ``content`` and an ``additional_kwargs["reasoning_content"]``
    populated by the provider's reasoning trailer). Both fields are
    extracted independently.

    :param chunk: streaming chat-model chunk (any provider shape)
    :ptype chunk: AIMessageChunk
    :return: ``ChunkParsed`` with ``text`` and ``reasoning`` strings
        (each possibly empty)
    :rtype: ChunkParsed
    """
    content = getattr(chunk, "content", "")
    text = ""
    reasoning = ""

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                block_text = block.get("text", "")
                if isinstance(block_text, str):
                    text += block_text
            elif btype == "thinking":
                block_thinking = block.get("thinking", "")
                if isinstance(block_thinking, str):
                    reasoning += block_thinking

    extras = getattr(chunk, "additional_kwargs", None) or {}
    extra_reasoning = extras.get("reasoning_content")
    if isinstance(extra_reasoning, str):
        reasoning += extra_reasoning

    return ChunkParsed(text=text, reasoning=reasoning)
