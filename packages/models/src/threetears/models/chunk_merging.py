"""merge streamed ``AIMessageChunk`` lists into a single ``AIMessage``.

Every chat-model consumer that drives the model via ``astream`` /
``astream_events`` ends each round with a list of accumulated
``AIMessageChunk`` objects. Producing the final ``AIMessage`` from that
list -- merging content, accumulating ``tool_call_chunks`` into
``tool_calls``, summing ``usage_metadata``, and preserving
``additional_kwargs`` / ``response_metadata`` -- is identical work for
every consumer (metallm's personality node, 14-eng-ai-bot's router,
14-eng-ai-bot-agents' tool loop, 14-eng-ai-bot-agent-admin).

:func:`merge_chunks` is the canonical 3tears version of that operation.
LangChain's ``AIMessageChunk.__add__`` handles the content and
tool-call-chunk merge correctly on its own; this module wraps it,
finalizes to a concrete ``AIMessage`` (with ``invalid_tool_calls``
preserved for downstream recovery), and rejects the empty-list case
explicitly so accidental no-op calls surface as a typed error rather
than silently returning an empty message.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk

__all__ = ["merge_chunks"]


def merge_chunks(chunks: list[AIMessageChunk]) -> AIMessage:
    """merge a list of streamed ``AIMessageChunk`` objects into one ``AIMessage``.

    Uses LangChain's built-in ``AIMessageChunk.__add__`` (which is the
    canonical merge implementation: content concatenates, tool-call
    chunks accumulate by index, ``additional_kwargs`` and
    ``response_metadata`` deep-merge, ``usage_metadata`` sums per field)
    and then finalizes the accumulator into a concrete ``AIMessage`` so
    downstream consumers do not have to handle the chunk subclass.

    ``invalid_tool_calls`` survives the conversion -- consumers
    (metallm's tool router, aibots-agents' dispatch) inspect that field
    when ``tool_calls`` is empty to attempt JSON-recovery on malformed
    streamed tool calls.

    :param chunks: streamed chunks to merge (non-empty)
    :ptype chunks: list[AIMessageChunk]
    :return: merged ``AIMessage`` with concatenated content,
        accumulated tool calls, summed usage, and preserved
        ``additional_kwargs`` / ``response_metadata``
    :rtype: AIMessage
    :raises ValueError: when ``chunks`` is empty (an empty stream is
        a programming error at every observed call site -- the caller
        is expected to short-circuit before reaching here)
    """
    if not chunks:
        raise ValueError(
            "merge_chunks() received an empty chunk list; callers must"
            " short-circuit empty streams before invoking the merge."
        )

    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk  # type: ignore[assignment]

    return AIMessage(
        content=merged.content if merged.content is not None else "",
        tool_calls=list(getattr(merged, "tool_calls", []) or []),
        invalid_tool_calls=list(getattr(merged, "invalid_tool_calls", []) or []),
        additional_kwargs=dict(getattr(merged, "additional_kwargs", {}) or {}),
        response_metadata=dict(getattr(merged, "response_metadata", {}) or {}),
        usage_metadata=getattr(merged, "usage_metadata", None),
    )
