"""Tests for the typed framework event registry + dispatch helper.

Pins the round-trip contract: ``dispatch_event`` writes the same shape
that ``FrameworkEventRegistry.parse`` reads. Catches drift between
producer-side serialization and consumer-side parsing -- the regression
class the registry was built to prevent.

Also pins noun_verb naming on every event registered into the default
registry. The convention is documented in ``langgraph/events.py``; this
test makes it mechanically enforceable so a future event added without
following the convention fails the build instead of slipping in.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from threetears.langgraph.events import (
    FrameworkEvent,
    FrameworkEventRegistry,
    ImageGeneratedEvent,
    PromptBuiltEvent,
    ReasoningStreamedEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ToolCompletedEvent,
    ToolDispatchedEvent,
    ToolStartedEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
    WorkflowStepCompletedEvent,
    default_registry,
    dispatch_event,
)


# ---------------------------------------------------------------------------
# Registry shape + convention
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    """The shared registry must contain every framework event class
    declared in this package, and every name must follow noun_verb form.
    """

    def test_all_langgraph_events_registered(self) -> None:
        expected = {
            "image_generated",
            "prompt_built",
            "reasoning_streamed",
            "response_completed",
            "response_failed",
            "tool_completed",
            "tool_dispatched",
            "tool_started",
            "workflow_completed",
            "workflow_started",
            "workflow_step_completed",
        }
        assert expected.issubset(set(default_registry.names()))

    def test_every_registered_name_is_noun_verb(self) -> None:
        # noun_verb means: at least one underscore, all lowercase, no
        # digits, and the final segment is a past-tense verb (ending in
        # -ed) OR a single past-tense word with an irregular ending.
        # Allowed irregular endings: -t (built), -en (broken), -ne (done).
        # The full list of registered events ships with verbs ending in
        # one of these; the test pins the convention going forward.
        pattern = re.compile(r"^[a-z]+(_[a-z]+)+$")
        allowed_suffixes = (
            "ed",
            "built",
            "streamed",
            "completed",
            "started",
            "dispatched",
            "failed",
            "generated",
            "retrieved",
            "summarized",
            "created",
            "changed",
        )
        for name in default_registry.names():
            assert pattern.match(name), f"event name {name!r} is not lowercase_underscore form"
            assert any(name.endswith(s) for s in allowed_suffixes), (
                f"event name {name!r} does not end in a recognized "
                f"past-tense suffix; framework events use noun_verb with "
                f"past-tense verbs"
            )


# ---------------------------------------------------------------------------
# Registry parse contract
# ---------------------------------------------------------------------------


class TestRegistryParse:
    def test_parse_known_event_returns_typed_instance(self) -> None:
        typed = default_registry.parse(
            "prompt_built",
            {"system_prompt": "hi", "jailbreak_text": None},
        )
        assert isinstance(typed, PromptBuiltEvent)
        assert typed.system_prompt == "hi"
        assert typed.jailbreak_text is None

    def test_parse_unknown_event_returns_none(self) -> None:
        result = default_registry.parse("never_registered", {"x": 1})
        assert result is None

    def test_parse_bad_payload_raises_validation_error(self) -> None:
        # PromptBuiltEvent.system_prompt is required and is str-typed.
        with pytest.raises(ValidationError):
            default_registry.parse("prompt_built", {"system_prompt": None})


# ---------------------------------------------------------------------------
# Registry register contract
# ---------------------------------------------------------------------------


class TestRegistryRegister:
    def test_register_class_without_type_field_raises(self) -> None:
        class NoType(FrameworkEvent):
            pass

        reg = FrameworkEventRegistry()
        with pytest.raises(ValueError, match="no 'type' field"):
            reg.register(NoType)

    def test_register_duplicate_name_for_different_class_raises(self) -> None:
        reg = FrameworkEventRegistry()
        reg.register(PromptBuiltEvent)
        # Register a different class under a colliding type literal.
        # Pydantic does not let us subclass with the same literal default
        # cleanly, so we construct a fresh class with the same discriminator.
        from typing import Literal

        class Imposter(FrameworkEvent):
            type: Literal["prompt_built"] = "prompt_built"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="already registered"):
            reg.register(Imposter)

    def test_register_same_class_twice_is_idempotent(self) -> None:
        reg = FrameworkEventRegistry()
        reg.register(PromptBuiltEvent)
        reg.register(PromptBuiltEvent)  # must not raise
        assert reg.names() == ["prompt_built"]


# ---------------------------------------------------------------------------
# Round-trip: dispatch_event -> registry.parse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_event_roundtrip_via_registry() -> None:
    """``dispatch_event(event)`` must publish the same shape that
    ``registry.parse(name, data)`` reads back as the typed event.

    The integration value: a producer that switches to dispatch_event
    sees no consumer-side breakage as long as the consumer parses via
    the registry.
    """
    captured: dict[str, Any] = {}

    async def fake_dispatch(name: str, data: dict[str, Any], *, config: Any = None) -> None:
        captured["name"] = name
        captured["data"] = data

    event = ToolCompletedEvent(
        tool_name="memory_recall",
        tool_status="completed",
        tool_duration_ms=42,
    )
    with patch(
        "threetears.langgraph.events.adispatch_custom_event",
        side_effect=fake_dispatch,
    ):
        await dispatch_event(event)

    assert captured["name"] == "tool_completed"
    # The data dict must NOT contain the 'type' field (it became the name).
    assert "type" not in captured["data"]

    parsed = default_registry.parse(captured["name"], captured["data"])
    assert isinstance(parsed, ToolCompletedEvent)
    assert parsed.tool_name == "memory_recall"
    assert parsed.tool_status == "completed"
    assert parsed.tool_duration_ms == 42


# ---------------------------------------------------------------------------
# Event-class field defaults (catches regressions in payload shape)
# ---------------------------------------------------------------------------


class TestEventFieldDefaults:
    """Spot checks: every event must be constructible with the minimum
    required fields the producer site supplies. Catches accidentally-
    required fields that would break older producers.
    """

    def test_prompt_built_minimum_fields(self) -> None:
        ev = PromptBuiltEvent(system_prompt="hi")
        assert ev.jailbreak_text is None
        assert ev.jailbreak_role is None

    def test_tool_started_minimum_fields(self) -> None:
        ev = ToolStartedEvent(tool_name="x")
        assert ev.tool_args == {}

    def test_response_completed_minimum_fields(self) -> None:
        ev = ResponseCompletedEvent(response_content="ok")
        assert ev.token_count_prompt == 0
        assert ev.tools_used == {}
        assert ev.tool_llm_id is None

    def test_response_failed_minimum_fields(self) -> None:
        ev = ResponseFailedEvent(content="provider went sideways")
        assert ev.type == "response_failed"

    def test_workflow_step_completed_minimum_fields(self) -> None:
        ev = WorkflowStepCompletedEvent()
        assert ev.step_index == 0
        assert ev.step_description == ""
        assert ev.step_status == "completed"
        assert ev.tool_name is None

    def test_image_generated_requires_markdown(self) -> None:
        with pytest.raises(ValidationError):
            ImageGeneratedEvent()  # type: ignore[call-arg]

    def test_reasoning_streamed_requires_token(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningStreamedEvent()  # type: ignore[call-arg]

    # Smoke: each event's model_dump round-trips through registry.
    def test_every_event_round_trips(self) -> None:
        events: list[FrameworkEvent] = [
            PromptBuiltEvent(system_prompt="x"),
            ToolDispatchedEvent(tool_name="t"),
            ToolStartedEvent(tool_name="t"),
            ToolCompletedEvent(tool_name="t"),
            ReasoningStreamedEvent(token="r"),
            ResponseCompletedEvent(response_content="r"),
            ResponseFailedEvent(content="boom"),
            WorkflowStartedEvent(),
            WorkflowStepCompletedEvent(),
            WorkflowCompletedEvent(),
            ImageGeneratedEvent(markdown="![](x)"),
        ]
        for ev in events:
            payload = ev.model_dump(mode="json")
            name = payload.pop("type")
            parsed = default_registry.parse(name, payload)
            assert type(parsed) is type(ev), f"round-trip lost type for {ev}: got {type(parsed)}"
