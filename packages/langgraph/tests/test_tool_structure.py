"""unit tests for :mod:`threetears.langgraph.tool_structure` and its two faces.

three things are pinned here, and only the first is ordinary unit coverage.

**The projection decision** -- inline under the bound, an omission record over
it, nothing at all for a tool that produced no structure -- including that an
inline payload is the artifact *verbatim*. A test that accepted a re-keyed or
re-shaped payload would be accepting the birth of a second result shape, which
is the thing success check 14 exists to refuse.

**That the two faces cannot drift.** ``ToolCompletedEvent`` and
``ToolCallEndEvent`` are the same channel rendered for two consumers, and a
field added to one and not the other is the defect this design is fixing,
recreated. The pin compares the faces to *each other*, not each to a constant
-- the Gate B sweep's lesson, where egress independence was "pinned" by two
tests that each compared a side to the value it was configured from and so
could not have failed.

**That a reader predating the change still parses the bytes.** Both events
grew a declared optional, and a declared optional is *serialized*, so it
crosses the wire as an explicit ``null`` from the first emit -- the exact shape
that refused every call from a lagging pod for three days on 2026-08-13. The
argument that these two models tolerate it (neither sets ``extra="forbid"``;
``FrameworkEvent``'s docstring promises additive safety) is an argument, and
what that outage cost was the difference between an argument and a test. So
the pins below parse real emitted bytes with hand-written models that have
never heard of ``structured``.
"""

from __future__ import annotations

import json
from typing import Any, Literal
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from threetears.langgraph.events import ToolCompletedEvent, default_registry
from threetears.langgraph.streaming import StreamingResponse, ToolCallEndEvent, parse_stream_event
from threetears.langgraph.tool_structure import (
    DEFAULT_STRUCTURED_INLINE_MAX_CHARS,
    NO_STREAM_STRUCTURE,
    OMISSION_KEY,
    OMISSION_REASON_OVER_BOUND,
    OMISSION_REASON_UNSERIALIZABLE,
    STRUCTURED_KIND_INLINE,
    STRUCTURED_KIND_OMITTED,
    StructuredToolResultFields,
    structure_for_stream,
)

#: a search-shaped artifact: the named-key projection a bound search tool puts
#: on ``ToolMessage.artifact``, plus the tool-authored summary the offload seam
#: already reads off the same dict.
_SEARCH_ARTIFACT: dict[str, Any] = {
    "search_results": {
        "schema_version": 1,
        "query": "otter population survey",
        "candidates": [{"url": "https://example.test/a", "title": "A"}],
    },
    "summary": "1 result",
}


class _RecordingTransport:
    """``StreamTransport`` test double recording every published payload.

    the same shape as the one in ``test_streaming.py``; kept local so this
    module's pins read as one file and the wire bytes stay inspectable as
    bytes (the predates-the-field pins below need the raw encoding, not a
    parsed object).
    """

    def __init__(self) -> None:
        """initialize the empty recording list.

        :return: nothing
        :rtype: None
        """
        self.payloads: list[bytes] = []

    async def publish(self, payload: bytes) -> None:
        """record one publish call.

        :param payload: serialized envelope bytes
        :ptype payload: bytes
        :return: nothing
        :rtype: None
        """
        self.payloads.append(payload)


class _PreStructureToolCallEndEvent(BaseModel):
    """``ToolCallEndEvent`` exactly as it was before this channel existed.

    hand-written rather than imported, because importing the current model
    would pin the change against itself. this is the reader a lagging scriob
    (or any consumer pinned to an earlier family version) runs.
    """

    type: Literal["tool_call_end"] = "tool_call_end"
    correlation_id: UUID
    tool_name: str
    success: bool
    elapsed_ms: int


class _PreStructureToolCompletedEvent(BaseModel):
    """``ToolCompletedEvent`` exactly as it was before this channel existed.

    the metallm-side reader: its ws handler validates the typed event and
    renders name + status + duration.
    """

    type: Literal["tool_completed"] = "tool_completed"
    tool_name: str
    tool_status: str = "completed"
    tool_duration_ms: int = 0


class TestStructureForStream:
    """the inline / omitted / absent decision."""

    def test_a_search_artifact_rides_inline_and_verbatim(self) -> None:
        """a small artifact is carried whole, unchanged, kinded inline.

        :return: nothing
        :rtype: None
        """
        structure = structure_for_stream(_SEARCH_ARTIFACT)

        assert structure.structured_kind == STRUCTURED_KIND_INLINE
        assert structure.structured == _SEARCH_ARTIFACT

    def test_the_inline_payload_is_a_copy_not_the_caller_s_dict(self) -> None:
        """mutating the artifact after the decision cannot rewrite the wire.

        :return: nothing
        :rtype: None
        """
        artifact = dict(_SEARCH_ARTIFACT)
        structure = structure_for_stream(artifact)
        artifact["summary"] = "mutated after the fact"

        assert structure.structured is not None
        assert structure.structured["summary"] == "1 result"

    def test_a_tool_with_no_artifact_carries_nothing(self) -> None:
        """the majority case costs two nulls and no omission record.

        :return: nothing
        :rtype: None
        """
        assert structure_for_stream(None) is NO_STREAM_STRUCTURE

    @pytest.mark.parametrize("artifact", ["a string artifact", 42, ["a", "list"], {}])
    def test_a_non_mapping_or_empty_artifact_carries_nothing(self, artifact: Any) -> None:
        """only a non-empty mapping is structure; the rest is simply absent.

        :param artifact: an artifact shape that is not carryable structure
        :ptype artifact: Any
        :return: nothing
        :rtype: None
        """
        assert structure_for_stream(artifact) is NO_STREAM_STRUCTURE

    def test_an_artifact_at_the_bound_still_rides(self) -> None:
        """the bound is inclusive -- ``>`` refuses, ``==`` carries.

        :return: nothing
        :rtype: None
        """
        artifact = {"blob": "x"}
        exact = len(json.dumps(artifact, ensure_ascii=False))

        structure = structure_for_stream(artifact, max_chars=exact)

        assert structure.structured_kind == STRUCTURED_KIND_INLINE

    def test_an_oversized_artifact_says_so_with_both_numbers(self) -> None:
        """an omission carries its reason, its size and the bound it missed.

        a truncated payload that does not admit it is the silent-partial-
        answer defect; the client needs the size to decide what to do next.

        :return: nothing
        :rtype: None
        """
        artifact = {"blob": "x" * 200}

        structure = structure_for_stream(artifact, max_chars=64)

        assert structure.structured_kind == STRUCTURED_KIND_OMITTED
        assert structure.structured is not None
        record = structure.structured[OMISSION_KEY]
        assert record["reason"] == OMISSION_REASON_OVER_BOUND
        assert record["limit_chars"] == 64
        assert record["size_chars"] > 64

    def test_an_omission_is_not_mistakable_for_a_projection(self) -> None:
        """D-S4: the omission record does not wear the projection's key.

        a narrowed payload under the full projection's key parses while
        under-reporting -- the latent defect found in the context-save node
        while closing check 14, which this channel must not make real on a
        surface with more readers.

        :return: nothing
        :rtype: None
        """
        structure = structure_for_stream(_SEARCH_ARTIFACT, max_chars=8)

        assert structure.structured is not None
        assert set(structure.structured) == {OMISSION_KEY}
        assert "search_results" not in structure.structured

    def test_an_unserializable_artifact_is_an_omission_not_a_crash(self) -> None:
        """a tool whose artifact holds a live object still completes.

        :return: nothing
        :rtype: None
        """

        class _NotJson:
            pass

        structure = structure_for_stream({"handle": _NotJson()})

        assert structure.structured_kind == STRUCTURED_KIND_OMITTED
        assert structure.structured is not None
        assert structure.structured[OMISSION_KEY]["reason"] == OMISSION_REASON_UNSERIALIZABLE
        assert "size_chars" not in structure.structured[OMISSION_KEY]

    def test_a_non_positive_bound_carries_nothing_inline(self) -> None:
        """ "nothing fits" is a legal configuration, read honestly.

        :return: nothing
        :rtype: None
        """
        structure = structure_for_stream(_SEARCH_ARTIFACT, max_chars=0)

        assert structure.structured_kind == STRUCTURED_KIND_OMITTED

    def test_the_default_bound_is_the_shared_constant(self) -> None:
        """every face reads one number, so answering OQ1 moves one line.

        :return: nothing
        :rtype: None
        """
        artifact = {"blob": "x" * (DEFAULT_STRUCTURED_INLINE_MAX_CHARS + 100)}

        assert structure_for_stream(artifact).structured_kind == STRUCTURED_KIND_OMITTED


class TestTheTwoFacesAgree:
    """one contract, two renderings -- compared to each other, not to a constant."""

    def test_both_faces_carry_the_same_structure_fields(self) -> None:
        """a field added to one face and not the other fails here.

        :return: nothing
        :rtype: None
        """
        channel_fields = set(StructuredToolResultFields.model_fields)

        completed = set(ToolCompletedEvent.model_fields)
        call_end = set(ToolCallEndEvent.model_fields)

        assert channel_fields <= completed
        assert channel_fields <= call_end
        assert completed & channel_fields == call_end & channel_fields

    def test_both_faces_render_one_artifact_identically(self) -> None:
        """the same artifact produces the same payload on both faces.

        :return: nothing
        :rtype: None
        """
        structure = structure_for_stream(_SEARCH_ARTIFACT)

        completed = ToolCompletedEvent(tool_name="threetears.web_search", **structure.as_fields())
        call_end = ToolCallEndEvent(
            correlation_id=uuid4(),
            tool_name="threetears.web_search",
            success=True,
            elapsed_ms=12,
            **structure.as_fields(),
        )

        assert completed.structured == call_end.structured
        assert completed.structured_kind == call_end.structured_kind

    def test_a_tool_with_no_structure_leaves_both_faces_as_they_were(self) -> None:
        """the default on both faces is a pair of nulls, not an empty dict.

        :return: nothing
        :rtype: None
        """
        completed = ToolCompletedEvent(tool_name="threetears.todo_write")
        call_end = ToolCallEndEvent(
            correlation_id=uuid4(),
            tool_name="threetears.todo_write",
            success=True,
            elapsed_ms=3,
        )

        assert (completed.structured, completed.structured_kind) == (None, None)
        assert (call_end.structured, call_end.structured_kind) == (None, None)


class TestTheStreamingFaceOnTheWire:
    """``emit_tool_call_end`` is the one place the decision is made."""

    @pytest.mark.asyncio
    async def test_an_artifact_reaches_the_wire_through_the_emitter(self) -> None:
        """the caller hands over the artifact; the emitter does the rest.

        :return: nothing
        :rtype: None
        """
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=uuid4(),
            conversation_id=uuid4(),
        )

        await stream.emit_tool_call_end(
            tool_name="threetears.web_search",
            success=True,
            elapsed_ms=42,
            artifact=_SEARCH_ARTIFACT,
        )

        event = parse_stream_event(transport.payloads[-1])
        assert isinstance(event, ToolCallEndEvent)
        assert event.structured_kind == STRUCTURED_KIND_INLINE
        assert event.structured == _SEARCH_ARTIFACT

    @pytest.mark.asyncio
    async def test_a_caller_that_passes_no_artifact_emits_what_it_always_did(self) -> None:
        """the existing call signature keeps its existing behaviour.

        :return: nothing
        :rtype: None
        """
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=uuid4(),
            conversation_id=uuid4(),
        )

        await stream.emit_tool_call_end(tool_name="threetears.todo_write", success=True, elapsed_ms=3)

        event = parse_stream_event(transport.payloads[-1])
        assert isinstance(event, ToolCallEndEvent)
        assert event.structured is None
        assert event.structured_kind is None

    @pytest.mark.asyncio
    async def test_the_emitter_honours_a_caller_s_own_bound(self) -> None:
        """a host with a real per-frame budget passes it per emit.

        :return: nothing
        :rtype: None
        """
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=uuid4(),
            conversation_id=uuid4(),
        )

        await stream.emit_tool_call_end(
            tool_name="threetears.web_search",
            success=True,
            elapsed_ms=42,
            artifact=_SEARCH_ARTIFACT,
            structured_max_chars=16,
        )

        event = parse_stream_event(transport.payloads[-1])
        assert isinstance(event, ToolCallEndEvent)
        assert event.structured_kind == STRUCTURED_KIND_OMITTED


class TestAReaderPredatingTheField:
    """real emitted bytes, parsed by models that never heard of ``structured``."""

    @pytest.mark.asyncio
    async def test_a_lagging_stream_reader_accepts_a_populated_event(self) -> None:
        """the bytes a structure-carrying emit produces still parse.

        :return: nothing
        :rtype: None
        """
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=uuid4(),
            conversation_id=uuid4(),
        )
        await stream.emit_tool_call_end(
            tool_name="threetears.web_search",
            success=True,
            elapsed_ms=42,
            artifact=_SEARCH_ARTIFACT,
        )

        old = _PreStructureToolCallEndEvent.model_validate_json(transport.payloads[-1])

        assert old.tool_name == "threetears.web_search"
        assert old.elapsed_ms == 42

    @pytest.mark.asyncio
    async def test_a_lagging_stream_reader_accepts_the_explicit_nulls(self) -> None:
        """the unset case crosses as ``"structured":null`` -- parse that too.

        this is the precise shape that took cobalt-dev down: not a value
        anybody sent, a declared optional nobody populated.

        :return: nothing
        :rtype: None
        """
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=uuid4(),
            conversation_id=uuid4(),
        )
        await stream.emit_tool_call_end(tool_name="threetears.todo_write", success=True, elapsed_ms=3)
        raw = transport.payloads[-1]

        assert b'"structured":null' in raw
        assert _PreStructureToolCallEndEvent.model_validate_json(raw).tool_name == "threetears.todo_write"

    def test_a_lagging_custom_event_reader_accepts_the_dispatched_payload(self) -> None:
        """``dispatch_event`` dumps to a dict; an older typed reader takes it.

        :return: nothing
        :rtype: None
        """
        event = ToolCompletedEvent(
            tool_name="threetears.web_search",
            tool_duration_ms=42,
            **structure_for_stream(_SEARCH_ARTIFACT).as_fields(),
        )
        payload = event.model_dump(mode="json")

        old = _PreStructureToolCompletedEvent.model_validate(payload)

        assert old.tool_name == "threetears.web_search"
        assert old.tool_duration_ms == 42

    def test_the_registry_round_trip_keeps_the_structure(self) -> None:
        """``dispatch_event`` -> ``registry.parse`` returns the payload intact.

        :return: nothing
        :rtype: None
        """
        event = ToolCompletedEvent(
            tool_name="threetears.web_search",
            **structure_for_stream(_SEARCH_ARTIFACT).as_fields(),
        )
        payload = event.model_dump(mode="json")
        name = payload.pop("type")

        parsed = default_registry.parse(name, payload)

        assert isinstance(parsed, ToolCompletedEvent)
        assert parsed.structured == _SEARCH_ARTIFACT
        assert parsed.structured_kind == STRUCTURED_KIND_INLINE
