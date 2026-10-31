"""A tool call's provider-specific fields, kept on the answer and sent back on the next request.

Some providers attach fields to a tool call that the chat-completions protocol does not define and
that must come back unchanged on the same call in the next request. Gemini 3 is the case that forced
this: on Google's OpenAI-compatible endpoint each signed call carries
``extra_content.google.thought_signature``, and a request whose assistant tool call lacks it is
refused with HTTP 400 ("Function call is missing a thought_signature"), so a tool-using agent dies on
its second turn (https://ai.google.dev/gemini-api/docs/thought-signatures). LangChain's OpenAI
integration keeps only ``id``, ``type`` and ``function`` in both directions, so nothing above the
provider client could ever see the field.

The fields are kept on the answer as :data:`TOOL_CALL_PROVIDER_EXTRA_KEY` in ``additional_kwargs``:
tool-call id -> every top-level field of that raw call other than the standard ones, verbatim. The key
is absent when no call carried any. Code above the provider (a gateway relaying the answer, an agent
replaying history) carries that entry as an opaque value and puts it back on the ``AIMessage`` it
replays; this module writes it back onto the matching tool calls of the outgoing request.

**Streams need per-stream state.** A chunk's conversion sees only the fragment it is converting, and
after a call's first fragment the provider identifies the call by ``index`` alone. LangChain's
``merge_dicts`` also concatenates two equal strings when chunks are summed, so the same signature on
two fragments would come out doubled and invalid. So a chunk's conversion records the raw fields by
index under a transient key (:func:`stash_stream_extras`), and :class:`StreamExtraResolver`, one per
stream, rewrites them under the public key by call id, each id's fields once, and always removes the
transient key, before anything downstream sees the chunk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage

from threetears.observe import get_logger

if TYPE_CHECKING:
    from langchain_core.outputs import ChatGenerationChunk, ChatResult

__all__ = [
    "TOOL_CALL_PROVIDER_EXTRA_KEY",
    "StreamExtraResolver",
    "capture_result_extras",
    "reinsert_payload_extras",
    "stash_stream_extras",
]

_logger = get_logger(__name__)

#: ``AIMessage.additional_kwargs`` key holding each tool call's provider-specific fields, by call id.
TOOL_CALL_PROVIDER_EXTRA_KEY = "tool_call_provider_extra"

#: where a streamed chunk carries its fragments' raw fields by index until the stream's resolver
#: rewrites them by id. Never seen outside this module: the resolver removes it from every chunk.
_STREAM_STASH_KEY = "__threetears_tool_call_provider_extra_by_index__"

#: the tool-call fields the chat-completions protocol defines; everything else is the provider's own.
_STANDARD_FIELDS = frozenset({"id", "type", "function", "index"})


def _extras_of(raw_call: Mapping[str, Any]) -> dict[str, Any]:
    """the provider-specific fields of one raw tool call, verbatim.

    :param raw_call: a tool call (or a streamed fragment of one) as the provider sent it
    :ptype raw_call: Mapping[str, Any]
    :return: every top-level field outside the protocol's own, possibly none
    :rtype: dict[str, Any]
    """
    return {key: value for key, value in raw_call.items() if key not in _STANDARD_FIELDS}


def capture_result_extras(result: ChatResult, response_dict: Mapping[str, Any]) -> None:
    """put each call's provider fields from a non-streamed answer onto its generation's message.

    :param result: the result LangChain built from ``response_dict``, one generation per choice
    :ptype result: ChatResult
    :param response_dict: the raw answer as a dict (the SDK object's ``model_dump()``)
    :ptype response_dict: Mapping[str, Any]
    """
    choices = response_dict.get("choices") or []
    for generation, choice in zip(result.generations, choices, strict=False):
        message = generation.message
        raw_calls = ((choice or {}).get("message") or {}).get("tool_calls") or []
        by_id: dict[str, dict[str, Any]] = {}
        for raw in raw_calls:
            if not isinstance(raw, Mapping) or not (extras := _extras_of(raw)):
                continue
            if raw.get("id"):
                by_id[raw["id"]] = extras
            else:
                _logger.warning(
                    "a tool call carried provider fields %s but no id; they cannot be sent back",
                    sorted(extras),
                )
        if by_id and isinstance(message, AIMessage):
            message.additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] = by_id


def stash_stream_extras(generation_chunk: ChatGenerationChunk | None, chunk: Mapping[str, Any]) -> None:
    """record the provider fields of a streamed chunk's tool-call fragments, by fragment index.

    :param generation_chunk: what LangChain converted ``chunk`` into, or ``None`` when it skipped it
    :ptype generation_chunk: ChatGenerationChunk | None
    :param chunk: the raw streamed chunk as a dict
    :ptype chunk: Mapping[str, Any]
    """
    if generation_chunk is None:
        return
    choices = chunk.get("choices") or (chunk.get("chunk") or {}).get("choices") or []
    if not choices:
        return
    delta = (choices[0] or {}).get("delta") or {}
    by_index: dict[int, dict[str, Any]] = {}
    for raw in delta.get("tool_calls") or []:
        if isinstance(raw, Mapping) and isinstance(raw.get("index"), int) and (extras := _extras_of(raw)):
            by_index[raw["index"]] = extras
    if by_index:
        generation_chunk.message.additional_kwargs[_STREAM_STASH_KEY] = by_index


class StreamExtraResolver:
    """per-stream rewrite of stashed fragment fields into :data:`TOOL_CALL_PROVIDER_EXTRA_KEY` by call id.

    Feed it every chunk in stream order with :meth:`resolve` before anything else sees the chunk, then
    call :meth:`finish` when the stream ends. One instance per stream. A chunk with nothing stashed
    passes through untouched, so a provider that never stashes (Anthropic, OpenRouter) pays nothing.
    """

    def __init__(self) -> None:
        """start with no calls seen."""
        self._ids: dict[int, str] = {}
        self._emitted: dict[str, dict[str, Any]] = {}
        self._pending: dict[int, dict[str, Any]] = {}

    def resolve(self, chunk: ChatGenerationChunk) -> ChatGenerationChunk:
        """rewrite one chunk's stashed fields under the public key, in place.

        :param chunk: the next chunk of the stream
        :ptype chunk: ChatGenerationChunk
        :return: the same chunk
        :rtype: ChatGenerationChunk
        """
        message = chunk.message
        for fragment in getattr(message, "tool_call_chunks", None) or []:
            index, call_id = fragment.get("index"), fragment.get("id")
            if isinstance(index, int) and call_id:
                self._ids.setdefault(index, call_id)
        stashed = message.additional_kwargs.pop(_STREAM_STASH_KEY, None) or {}
        for index, extras in stashed.items():
            self._pending.setdefault(index, {}).update(extras)
        fresh: dict[str, dict[str, Any]] = {}
        # a chunk can arrive already keyed by id: LangChain builds a structured-output stream's final
        # chunk from the whole completion (``_create_chat_result``), repeating what the deltas carried.
        # It goes through the same emit-once check, or summing the stream would double the signature.
        keyed = message.additional_kwargs.pop(TOOL_CALL_PROVIDER_EXTRA_KEY, None) or {}
        for call_id, extras in keyed.items():
            new = self._first_seen(call_id, dict(extras))
            if new:
                fresh.setdefault(call_id, {}).update(new)
        for index in list(self._pending):
            call_id = self._ids.get(index)
            if call_id is None:
                continue
            new = self._first_seen(call_id, self._pending.pop(index))
            if new:
                fresh.setdefault(call_id, {}).update(new)
        if fresh:
            message.additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] = fresh
        return chunk

    def finish(self) -> None:
        """report fields whose call never received an id; they cannot be sent back."""
        for index in self._pending:
            _logger.warning(
                "a streamed tool call at index %s carried provider fields but never an id; they are dropped",
                index,
            )
        self._pending.clear()

    def _first_seen(self, call_id: str, extras: dict[str, Any]) -> dict[str, Any]:
        """the fields of ``extras`` not yet emitted for ``call_id``; a repeat is dropped, first value wins.

        :param call_id: the call the fields belong to
        :ptype call_id: str
        :param extras: the fields arriving now
        :ptype extras: dict[str, Any]
        :return: the fields to emit now
        :rtype: dict[str, Any]
        """
        seen = self._emitted.setdefault(call_id, {})
        new: dict[str, Any] = {}
        for key, value in extras.items():
            if key not in seen:
                seen[key] = value
                new[key] = value
            elif seen[key] != value:
                _logger.warning(
                    "tool call %s sent provider field %r twice with different values; the first is kept",
                    call_id,
                    key,
                )
        return new


def reinsert_payload_extras(payload: dict[str, Any], messages: Sequence[Any]) -> None:
    """write each call's kept provider fields back onto its tool call in an outgoing request, in place.

    Untouched unless some input ``AIMessage`` carries the key and the request has an assistant tool
    call with a matching id, so a request with nothing to send back is exactly what it was.

    :param payload: the chat-completions request body LangChain built
    :ptype payload: dict[str, Any]
    :param messages: the input messages the body was built from
    :ptype messages: Sequence[Any]
    """
    kept: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            extras = message.additional_kwargs.get(TOOL_CALL_PROVIDER_EXTRA_KEY)
            if isinstance(extras, Mapping):
                kept.update({call_id: dict(fields) for call_id, fields in extras.items() if isinstance(call_id, str)})
    if not kept:
        return
    placed: set[str] = set()
    for sent in payload.get("messages") or []:
        if not isinstance(sent, dict) or sent.get("role") != "assistant":
            continue
        for call in sent.get("tool_calls") or []:
            call_id = call.get("id") if isinstance(call, dict) else None
            if isinstance(call_id, str) and (fields := kept.get(call_id)):
                placed.add(call_id)
                for key, value in fields.items():
                    if key not in _STANDARD_FIELDS:
                        call[key] = value
    if unplaced := sorted(set(kept) - placed):
        _logger.warning(
            "kept provider fields for tool call(s) %s match no tool call in the request; they are not sent",
            unplaced,
        )
