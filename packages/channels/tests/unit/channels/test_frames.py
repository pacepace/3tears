"""tests for the typed Frame envelope + injected-seam protocols (channels-task-03).

``Frame`` is the inbound/outbound typed envelope the typed-frame router
parses every raw websocket message into (design T3-D2): a ``type``
discriminator plus optional ``room`` / ``payload`` / ``seq``. The
injected-seam protocols (``NsResolver`` / ``OpHandler`` / ``ReplaySource``)
and the structural ``NsEntity`` protocol are the scriob→channels boundary
seams (T3-D1/D3/D4) — channels depends only on these shapes, never on a
scriob type.

The "no in-process seq" enforcement (T3-D4) lives here as a field-set
freeze on ``Frame``: the only sequence ``Frame`` carries is the durable
op-log ``seq`` echoed in/out, never a counter the envelope mints itself.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest


class TestFrameParsing:
    """Frame.from_raw parses raw JSON text into the typed envelope."""

    def test_parses_message_frame(self) -> None:
        """a minimal ``message`` frame round-trips type + payload."""
        from threetears.channels.frames import Frame

        raw = json.dumps({"type": "message", "content": "hi"})
        frame = Frame.from_raw(raw)
        assert frame.type == "message"

    def test_parses_room_and_seq(self) -> None:
        """``room`` and ``seq`` parse when present."""
        from threetears.channels.frames import Frame

        raw = json.dumps({"type": "editor.op", "room": "c:s:main:f.md", "seq": 7})
        frame = Frame.from_raw(raw)
        assert frame.type == "editor.op"
        assert frame.room == "c:s:main:f.md"
        assert frame.seq == 7

    def test_missing_optional_fields_default_to_none(self) -> None:
        """absent ``room`` / ``payload`` / ``seq`` default to ``None``."""
        from threetears.channels.frames import Frame

        frame = Frame.from_raw(json.dumps({"type": "join"}))
        assert frame.room is None
        assert frame.payload is None
        assert frame.seq is None

    def test_unknown_extra_fields_are_ignored(self) -> None:
        """unknown keys are dropped (``extra='ignore'``) for forward-compat."""
        from threetears.channels.frames import Frame

        raw = json.dumps({"type": "cursor", "room": "r", "unknown_future_field": 1})
        frame = Frame.from_raw(raw)
        assert frame.type == "cursor"
        assert not hasattr(frame, "unknown_future_field")

    def test_invalid_json_raises(self) -> None:
        """non-JSON text raises so the caller can emit an error frame."""
        from threetears.channels.frames import Frame

        with pytest.raises(json.JSONDecodeError):
            Frame.from_raw("not json{")


class TestFrameErrorConstructor:
    """Frame.error builds an outbound error frame serialized for the wire."""

    def test_error_frame_is_json_with_type_error(self) -> None:
        """``Frame.error`` yields JSON text carrying ``type='error'`` + message."""
        from threetears.channels.frames import Frame

        text = Frame.error("nope")
        parsed = json.loads(text)
        assert parsed["type"] == "error"
        assert parsed["message"] == "nope"


class TestFrameHasNoInProcessSeqCounter:
    """enforcement (T3-D4): Frame carries only the durable op-log seq, never mints one."""

    def test_frame_seq_is_an_echoed_value_not_a_counter(self) -> None:
        """``seq`` is a plain optional field, defaulting to None (no auto-increment)."""
        from threetears.channels.frames import Frame

        a = Frame.from_raw(json.dumps({"type": "editor.op"}))
        b = Frame.from_raw(json.dumps({"type": "editor.op"}))
        # two freshly-parsed frames both default to None: nothing auto-increments.
        assert a.seq is None
        assert b.seq is None

    def test_frame_has_no_counter_attributes(self) -> None:
        """no ``next_seq`` / ``_counter`` / ``increment`` surface on Frame."""
        from threetears.channels.frames import Frame

        banned = {"next_seq", "_seq", "_counter", "counter", "increment", "advance"}
        attrs = set(dir(Frame))
        assert banned.isdisjoint(attrs), f"Frame exposes a seq-counter surface: {banned & attrs}"


class TestSeamProtocols:
    """the injected-seam protocols are runtime-checkable structural contracts."""

    def test_op_result_carries_seq(self) -> None:
        """``OpResult`` exposes an int ``seq`` (the op-log's authoritative sequence)."""
        from threetears.channels.frames import OpResult

        result = OpResult(seq=42)
        assert result.seq == 42

    def test_ns_resolver_protocol_is_runtime_checkable(self) -> None:
        """a conforming async ``(room_id) -> NsEntity`` satisfies ``NsResolver``."""
        from threetears.channels.frames import NsEntity, NsResolver

        class _Ns:
            id = uuid4()
            customer_id = uuid4()
            namespace_type = "story"
            owner_agent_id = uuid4()

        class _Resolver:
            async def __call__(self, room_id: str) -> NsEntity:
                return _Ns()  # type: ignore[return-value]

        assert isinstance(_Resolver(), NsResolver)

    def test_op_handler_protocol_is_runtime_checkable(self) -> None:
        """a conforming async op handler satisfies ``OpHandler``."""
        from threetears.channels.frames import Frame, OpHandler, OpResult

        class _Handler:
            async def __call__(self, room_id: str, user_id: str, frame: Frame) -> OpResult:
                return OpResult(seq=1)

        assert isinstance(_Handler(), OpHandler)

    def test_replay_source_protocol_is_runtime_checkable(self) -> None:
        """a conforming async-iterator replay source satisfies ``ReplaySource``."""
        from threetears.channels.frames import ReplaySource

        class _Replay:
            async def __call__(self, room_id: str, from_seq: int) -> AsyncIterator[str]:
                for i in range(from_seq, from_seq + 1):
                    yield str(i)

        assert isinstance(_Replay(), ReplaySource)

    def test_ns_entity_protocol_matches_acl_namespace_fields(self) -> None:
        """``NsEntity`` is structural over the four fields ``authorize_on_entity`` reads."""
        from threetears.channels.frames import NsEntity

        class _Ns:
            id = uuid4()
            customer_id = uuid4()
            namespace_type = "story"
            owner_agent_id = uuid4()

        assert isinstance(_Ns(), NsEntity)

        class _Partial:
            id = uuid4()
            # missing customer_id / namespace_type / owner_agent_id

        assert not isinstance(_Partial(), NsEntity)
