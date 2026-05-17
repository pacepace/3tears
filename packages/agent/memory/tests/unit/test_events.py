"""Tests for memory-package event classes + shared-registry registration.

The shared default registry lives in :mod:`threetears.langgraph.events`;
memory events register themselves into it on import. These tests pin
that registration so a future refactor that quietly drops the import
side effect (and breaks every product that parses memory events through
the shared registry) fails the build instead of silently regressing.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
from threetears.agent.memory.events import (
    MemoryCreatedEvent,
    MemoryRetrievedEvent,
    default_memory_created_dispatcher,
)
from threetears.langgraph.events import default_registry


class TestSharedRegistryHasMemoryEvents:
    def test_memory_retrieved_registered(self) -> None:
        assert "memory_retrieved" in default_registry.names()
        parsed = default_registry.parse(
            "memory_retrieved",
            {"memories_count": 2, "memories": [{"memory_id": "abc"}]},
        )
        assert isinstance(parsed, MemoryRetrievedEvent)
        assert parsed.memories_count == 2

    def test_memory_created_registered(self) -> None:
        assert "memory_created" in default_registry.names()
        parsed = default_registry.parse(
            "memory_created",
            {
                "user_id": "u-1",
                "memory_id": "m-1",
                "conversation_id": "c-1",
                "type_memory": "topical_context",
                "content_preview": "hello",
            },
        )
        assert isinstance(parsed, MemoryCreatedEvent)
        assert parsed.user_id == "u-1"
        assert parsed.content_preview == "hello"


class TestMemoryEventDefaults:
    def test_memory_retrieved_defaults_empty(self) -> None:
        ev = MemoryRetrievedEvent()
        assert ev.memories_count == 0
        assert ev.memories == []

    def test_memory_created_requires_identifying_fields(self) -> None:
        # user_id, memory_id, conversation_id, type_memory are required.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MemoryCreatedEvent()  # type: ignore[call-arg]


class _StubMemoryEntity:
    """duck-typed stand-in for :class:`MemoryEntity` carrying the
    attribute surface the dispatcher reads.

    avoids constructing a full :class:`MemoryEntity` (which requires a
    composite-pk ``_id`` and BaseEntity wiring) since the dispatcher
    only reads ``user_id`` / ``memory_id`` / ``conversation_id`` /
    ``type_memory`` / ``content`` by attribute.
    """

    def __init__(
        self,
        *,
        user_id: Any,
        memory_id: Any,
        conversation_id: Any,
        type_memory: str,
        content: str | None = None,
    ) -> None:
        self.user_id = user_id
        self.memory_id = memory_id
        self.conversation_id = conversation_id
        self.type_memory = type_memory
        self.content = content


class TestDefaultMemoryCreatedDispatcher:
    """Pins the contract for :func:`default_memory_created_dispatcher`.

    the helper is the canonical "opt-in default" wired by consumers
    (the aibots-agents SDK, metallm, etc.) as the
    :attr:`MemoryExtractor.on_memory_created` callback. these tests pin
    the build-and-dispatch path, the content_preview truncation, the
    str-cast of identifying fields, and the graceful no-op when no run
    manager is in scope.
    """

    @pytest.mark.asyncio
    async def test_dispatches_memory_created_event_inside_graph(self) -> None:
        """when ``dispatch_event`` succeeds, the helper builds the
        right :class:`MemoryCreatedEvent` and forwards it.

        verifies the field mapping from :class:`MemoryEntity` attributes
        to the event payload: ``user_id`` / ``memory_id`` /
        ``conversation_id`` cast to ``str``, ``type_memory`` passes
        through verbatim, and ``content_preview`` is the leading slice
        of ``content``.
        """
        from uuid import uuid4

        entity = _StubMemoryEntity(
            user_id=uuid4(),
            memory_id=uuid4(),
            conversation_id=uuid4(),
            type_memory="topical_context",
            content="The user prefers Python over Rust for prototypes.",
        )

        captured: list[MemoryCreatedEvent] = []

        async def _fake_dispatch(event: Any, *, config: Any = None) -> None:
            captured.append(event)

        with patch(
            "threetears.agent.memory.events.dispatch_event",
            new=_fake_dispatch,
        ):
            await default_memory_created_dispatcher(entity)

        assert len(captured) == 1
        emitted = captured[0]
        assert isinstance(emitted, MemoryCreatedEvent)
        assert emitted.user_id == str(entity.user_id)
        assert emitted.memory_id == str(entity.memory_id)
        assert emitted.conversation_id == str(entity.conversation_id)
        assert emitted.type_memory == "topical_context"
        assert emitted.content_preview == "The user prefers Python over Rust for prototypes."

    @pytest.mark.asyncio
    async def test_content_preview_truncated_to_120_chars(self) -> None:
        """``content_preview`` is sliced to the documented ~120 chars.

        the event docstring says the preview is "first ~120 chars";
        the helper's ``_CONTENT_PREVIEW_LEN`` is the source of truth.
        long content gets sliced; shorter content passes through.
        """
        long_content = "x" * 500

        captured: list[MemoryCreatedEvent] = []

        async def _fake_dispatch(event: Any, *, config: Any = None) -> None:
            captured.append(event)

        with patch(
            "threetears.agent.memory.events.dispatch_event",
            new=_fake_dispatch,
        ):
            await default_memory_created_dispatcher(
                _StubMemoryEntity(
                    user_id="u-1",
                    memory_id="m-1",
                    conversation_id="c-1",
                    type_memory="fact",
                    content=long_content,
                ),
            )

        assert len(captured[0].content_preview) == 120
        assert captured[0].content_preview == "x" * 120

    @pytest.mark.asyncio
    async def test_empty_content_yields_empty_preview(self) -> None:
        """``content=None`` or empty string yields empty ``content_preview``.

        defensive: the extractor pipeline always populates ``content``
        on commit, but a caller wiring the dispatcher outside that path
        might forget. helper falls through to ``""`` rather than tripping
        a ``TypeError`` on ``None[:N]``.
        """
        captured: list[MemoryCreatedEvent] = []

        async def _fake_dispatch(event: Any, *, config: Any = None) -> None:
            captured.append(event)

        with patch(
            "threetears.agent.memory.events.dispatch_event",
            new=_fake_dispatch,
        ):
            await default_memory_created_dispatcher(
                _StubMemoryEntity(
                    user_id="u-1",
                    memory_id="m-1",
                    conversation_id="c-1",
                    type_memory="preference",
                    content=None,
                ),
            )

        assert captured[0].content_preview == ""

    @pytest.mark.asyncio
    async def test_no_run_manager_swallows_runtime_error_and_logs_debug(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """outside a langgraph run, ``RuntimeError`` from no-manager is
        logged at debug and swallowed.

        when a caller (cli, background job, test harness) invokes
        :meth:`MemoryExtractor.run_extraction` without a langgraph
        context, ``adispatch_custom_event`` raises ``RuntimeError`` from
        deep inside. the dispatcher must not propagate -- the memory
        row is already committed by the time the callback fires, so a
        crash here would break the extraction pipeline for what is a
        best-effort surface.

        the log line names the memory_id so an operator grepping
        ``memory_created event dropped`` can correlate.
        """

        async def _raise_no_manager(event: Any, *, config: Any = None) -> None:
            raise RuntimeError("Unable to find run manager")

        with (
            patch(
                "threetears.agent.memory.events.dispatch_event",
                new=_raise_no_manager,
            ),
            caplog.at_level(logging.DEBUG, logger="threetears.agent.memory.events"),
        ):
            # Should NOT raise.
            await default_memory_created_dispatcher(
                _StubMemoryEntity(
                    user_id="u-99",
                    memory_id="m-99",
                    conversation_id="c-99",
                    type_memory="fact",
                    content="probe",
                ),
            )

        assert any(
            "memory_created event dropped" in record.message and "memory_id=m-99" in record.message
            for record in caplog.records
        ), f"expected a DEBUG log line naming the dropped memory_id; got {[r.message for r in caplog.records]!r}"

    @pytest.mark.asyncio
    async def test_propagates_non_runtime_errors(self) -> None:
        """exceptions other than ``RuntimeError`` propagate.

        narrowing the swallow to ``RuntimeError`` is deliberate: a
        :class:`pydantic.ValidationError` from a future schema regression,
        a :class:`TypeError` from a malformed entity, or any other
        bug-class failure inside ``dispatch_event`` MUST surface so
        ops + tests notice. the no-manager case is the only documented
        "expected" failure mode that gets the log+swallow treatment.
        """

        async def _raise_value_error(event: Any, *, config: Any = None) -> None:
            raise ValueError("schema regression simulation")

        with patch(
            "threetears.agent.memory.events.dispatch_event",
            new=_raise_value_error,
        ):
            with pytest.raises(ValueError, match="schema regression"):
                await default_memory_created_dispatcher(
                    _StubMemoryEntity(
                        user_id="u-1",
                        memory_id="m-1",
                        conversation_id="c-1",
                        type_memory="fact",
                        content="probe",
                    ),
                )
