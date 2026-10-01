"""Keeps a tool call with a junk name away from everything downstream of a chat model.

A junk name is a concrete name claim that fails the canonical 3tears tool-name regex
(:func:`threetears.models.tool_name_validation.is_valid_tool_name`), such as the XML-attribute
leak ``memory_recall" name="memory_recall`` that was dispatched and persisted in prod on
2026-05-19. A missing or empty name is not junk: it is how every streamed fragment after a call's
first one looks.

Two places, because a call reaches a consumer in two shapes.

**A finished message** (:func:`drop_junk_tool_calls`): the call is removed from ``tool_calls``,
``invalid_tool_calls`` and ``tool_call_chunks``, and any content block carrying its id goes with
it. A junk name with well-formed arguments parses into ``tool_calls``, not
``invalid_tool_calls``, and Anthropic also carries every call as a ``tool_use`` content block,
which LangChain sends back to the provider on the next turn.

**A stream** (:class:`JunkToolCallStreamFilter`): a chunk carries fragments of calls, keyed by
``index``, and only a call's first fragment usually carries its name. Clearing a chunk's
``invalid_tool_calls`` does nothing here: the name rides ``tool_call_chunks`` and comes back
when the chunks are added up, and a fragment whose arguments are empty parses as a valid
``tool_calls`` entry that no ``invalid_tool_calls`` filter ever sees. So the filter decides per
call: a fragment is released, held, or dropped by the verdict on its call's name.

- A call whose name has arrived and is valid is released: its fragments pass the moment they
  arrive. Every provider wired here (chat completions for OpenAI and OpenRouter, Anthropic's
  Messages API) names a call on its first fragment, so a valid call is never held at all.
- A call whose name has arrived and is junk is dropped: the fragments held for it, the rest of
  its fragments, and the content blocks sharing its index (Anthropic's ``tool_use`` and
  ``input_json_delta``, OpenAI Responses' ``function_call``: each provider numbers a call's
  content blocks and its tool-call fragments from the same per-block index).
- A fragment that arrives before its call's name is held until the name does. Text, usage and
  every other part of the chunk carrying it go out at once, ahead of the held fragment. That
  reorders a held fragment behind later text, which adding chunks up does not notice: content
  concatenates, and fragments merge by index in the order they were held.

The trade-off: holding is the only added latency, and it applies only to fragments that come
before their name, which no wired provider sends. The alternative -- holding each call until it
is complete -- would delay every tool call's arguments to the end of the call for a case that
does not occur. What this design cannot do is take back a fragment already released: a
provider that split one name across fragments, valid at first and junk once completed, would
have released a call under the valid prefix. That call is dropped from the moment its name turns
junk, so what was released is a valid-shaped name a dispatcher refuses as an unknown tool, never
the junk name itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from threetears.models.tool_name_validation import is_valid_tool_name
from threetears.observe import get_logger

if TYPE_CHECKING:
    from langchain_core.messages import ToolCallChunk
    from langchain_core.outputs import ChatGenerationChunk

__all__ = ["JunkToolCallStreamFilter", "drop_junk_tool_calls"]

_logger = get_logger(__name__)

#: the message fields that carry tool calls, each a list of dicts with a ``name``
_CALL_FIELDS = ("tool_calls", "invalid_tool_calls", "tool_call_chunks")

#: the content-block keys that carry a tool call's id
_BLOCK_ID_KEYS = ("id", "call_id")


def _is_junk(name: object) -> bool:
    """whether ``name`` is a concrete name claim that cannot be a tool's.

    :param name: a tool call's name, as received
    :ptype name: object
    :return: ``True`` for a non-empty name failing the canonical regex, or a non-string one
    :rtype: bool
    """
    if name is None or name == "":
        return False
    return not (isinstance(name, str) and is_valid_tool_name(name))


def _warn_dropped(name: object) -> None:
    """log one dropped call, its name truncated so a long or hostile one cannot flood the log.

    :param name: the junk name
    :ptype name: object
    """
    truncated = name[:80] if isinstance(name, str) else repr(name)[:80]
    _logger.warning(
        "dropped a tool call whose name is not a tool name (the model wrote junk where a name goes): %s",
        truncated,
    )


def drop_junk_tool_calls(message: Any) -> None:
    """remove every tool call with a junk name from a finished message, in place.

    Each dropped call is logged once at WARNING. Content blocks carrying a dropped call's id go
    with it.

    :param message: a chat model's answer (``AIMessage`` or ``AIMessageChunk``); duck-typed
    :ptype message: Any
    """
    dropped_ids: set[Any] = set()
    logged: set[tuple[Any, Any]] = set()
    for field_name in _CALL_FIELDS:
        entries = getattr(message, field_name, None)
        if not entries:
            continue
        kept: list[Any] = []
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else entry
            if not isinstance(entry, dict) or _is_junk(name):
                call_id = entry.get("id") if isinstance(entry, dict) else None
                if call_id is not None:
                    dropped_ids.add(call_id)
                if (name, call_id) not in logged:
                    logged.add((name, call_id))
                    _warn_dropped(name)
            else:
                kept.append(entry)
        if len(kept) != len(entries):
            entries[:] = kept
    content = getattr(message, "content", None)
    if dropped_ids and isinstance(content, list):
        content[:] = [
            block
            for block in content
            if not (isinstance(block, dict) and any(block.get(key) in dropped_ids for key in _BLOCK_ID_KEYS))
        ]


@dataclass
class _Call:
    """one streamed tool call, as far as it has arrived.

    :ivar name: the name fragments so far, joined
    :ivar verdict: ``pending`` until a name arrives, then ``released`` or ``dropped``
    :ivar held: the chunks built from this call's fragments while it was pending, in order
    """

    name: str = ""
    verdict: Literal["pending", "released", "dropped"] = "pending"
    held: list[ChatGenerationChunk] = field(default_factory=list)


class JunkToolCallStreamFilter:
    """per-stream filter that keeps every fragment of a junk-named tool call out of the stream.

    Feed it each chunk in order with :meth:`feed` and emit what it returns; call :meth:`finish`
    once the source ends and emit that too. One instance per stream: it carries each call's
    state from chunk to chunk. The module docstring has the release rules and their trade-off.
    """

    def __init__(self) -> None:
        """start with no calls seen."""
        self._calls: dict[Any, _Call] = {}

    def feed(self, chunk: ChatGenerationChunk) -> list[ChatGenerationChunk]:
        """take one chunk from the source; return the chunks to emit now, in order.

        A chunk with nothing held or dropped is returned as the same object. Otherwise the parts
        that may go out are rebuilt into a new chunk alongside the original's metadata, and the
        held parts wait in their call.

        :param chunk: the next chunk from the provider's stream
        :ptype chunk: ChatGenerationChunk
        :return: the chunks to emit, possibly none
        :rtype: list[ChatGenerationChunk]
        """
        message = chunk.message
        fragments: list[ToolCallChunk] = list(getattr(message, "tool_call_chunks", None) or [])
        out: list[ChatGenerationChunk] = []
        for fragment in fragments:
            out.extend(self._take_name(fragment))

        kept: list[ToolCallChunk] = []
        held: dict[Any, list[ToolCallChunk]] = {}
        dropped = False
        for fragment in fragments:
            index = fragment.get("index")
            if index is None:
                if _is_junk(fragment.get("name")):
                    _warn_dropped(fragment.get("name"))
                    dropped = True
                else:
                    kept.append(fragment)
                continue
            verdict = self._calls[index].verdict
            if verdict == "released":
                kept.append(fragment)
            elif verdict == "pending":
                held.setdefault(index, []).append(fragment)
            else:
                dropped = True

        content = message.content
        kept_blocks: list[Any] = []
        held_blocks: dict[Any, list[Any]] = {}
        if isinstance(content, list):
            for block in content:
                block_index = block.get("index") if isinstance(block, dict) else None
                call = self._calls.get(block_index) if block_index is not None else None
                if call is None or call.verdict == "released":
                    kept_blocks.append(block)
                elif call.verdict == "pending":
                    held_blocks.setdefault(block_index, []).append(block)
                else:
                    dropped = True

        if not held and not held_blocks and not dropped:
            out.append(chunk)
        else:
            for index in dict.fromkeys([*held, *held_blocks]):
                self._calls[index].held.append(
                    _bare_chunk(
                        chunk,
                        content=held_blocks.get(index, []) if isinstance(content, list) else "",
                        tool_call_chunks=held.get(index, []),
                    )
                )
            remainder = _rebuilt_chunk(chunk, content=kept_blocks if isinstance(content, list) else content, kept=kept)
            if remainder is not None:
                out.append(remainder)
        return out

    def finish(self) -> list[ChatGenerationChunk]:
        """release what is still held once the source has ended.

        A call that never received a name is not junk -- a nameless fragment is kept everywhere
        else too -- so its held fragments go out as they are.

        :return: the held chunks, in the order their calls first appeared
        :rtype: list[ChatGenerationChunk]
        """
        out: list[ChatGenerationChunk] = []
        for call in self._calls.values():
            if call.verdict == "pending":
                out.extend(call.held)
                call.held = []
        return out

    def _take_name(self, fragment: ToolCallChunk) -> list[ChatGenerationChunk]:
        """record a fragment's name, if it has one, and settle its call's verdict.

        :param fragment: one ``tool_call_chunks`` entry
        :ptype fragment: ToolCallChunk
        :return: the call's held chunks, when this name released it
        :rtype: list[ChatGenerationChunk]
        """
        index = fragment.get("index")
        name = fragment.get("name")
        released: list[ChatGenerationChunk] = []
        call = self._calls.setdefault(index, _Call()) if index is not None else None
        if call is not None and call.verdict != "dropped" and name is not None and name != "":
            released_as = call.name if call.verdict == "released" else None
            call.name = call.name + name if isinstance(name, str) else repr(name)
            if _is_junk(call.name) or not isinstance(name, str):
                if released_as is not None:
                    _logger.warning(
                        "a tool call released as %s had more name fragments that made it junk; what"
                        " was released stays out, the rest of the call is dropped",
                        released_as[:80],
                    )
                _warn_dropped(call.name)
                call.verdict = "dropped"
                call.held = []
            elif call.verdict == "pending":
                call.verdict = "released"
                released, call.held = call.held, []
        return released


def _bare_chunk(source: ChatGenerationChunk, *, content: Any, tool_call_chunks: list[ToolCallChunk]) -> Any:
    """a chunk holding only the given parts of ``source``, under its message id.

    :param source: the chunk the parts came from
    :ptype source: ChatGenerationChunk
    :param content: the content to carry
    :ptype content: Any
    :param tool_call_chunks: the fragments to carry
    :ptype tool_call_chunks: list[ToolCallChunk]
    :return: the new chunk
    :rtype: ChatGenerationChunk
    """
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    message = AIMessageChunk(content=content, tool_call_chunks=tool_call_chunks, id=source.message.id)
    return ChatGenerationChunk(message=message)


def _rebuilt_chunk(source: ChatGenerationChunk, *, content: Any, kept: list[ToolCallChunk]) -> Any:
    """``source`` with only the content and fragments that may go out, or ``None`` if it carries nothing.

    Rebuilt rather than copied, so the message derives ``tool_calls`` and ``invalid_tool_calls``
    afresh from the fragments it keeps. Everything else on the message and the generation is kept.

    :param source: the chunk as the provider produced it
    :ptype source: ChatGenerationChunk
    :param content: the content to keep
    :ptype content: Any
    :param kept: the fragments to keep
    :ptype kept: list[ToolCallChunk]
    :return: the rebuilt chunk, or ``None`` when nothing is left worth emitting
    :rtype: ChatGenerationChunk | None
    """
    from langchain_core.outputs import ChatGenerationChunk

    message = source.message
    fields = {
        name: getattr(message, name)
        for name in type(message).model_fields
        if name not in ("tool_calls", "invalid_tool_calls")
    }
    fields.update(content=content, tool_call_chunks=kept)
    rebuilt = type(message)(**fields)
    carries = (
        bool(content)
        or bool(kept)
        or bool(rebuilt.additional_kwargs)
        or bool(rebuilt.response_metadata)
        or getattr(rebuilt, "usage_metadata", None) is not None
        or getattr(rebuilt, "chunk_position", None) is not None
        or bool(source.generation_info)
    )
    return ChatGenerationChunk(message=rebuilt, generation_info=source.generation_info) if carries else None
