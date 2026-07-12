"""unit tests for intention tool input schemas + local helpers.

Pins the Pydantic tool schemas (what the LLM may pass) and the two
package-private helpers (``_tool_error`` formatting, ``_safe_aembed_query``
soft-fail) without standing up a database.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from threetears.agent.intention.events import IntentionSurfacedEvent
from threetears.agent.intention.tools import (
    IntentionListInput,
    IntentionLogInput,
    IntentionMarkSurfacedInput,
    _emit_intention_event,
    _safe_aembed_query,
    _tool_error,
)


class TestIntentionLogInput:
    def test_content_only(self) -> None:
        inp = IntentionLogInput(content="learn more about the user's work")
        assert inp.content == "learn more about the user's work"
        assert inp.source_memory_id is None

    def test_content_required(self) -> None:
        with pytest.raises(ValidationError):
            IntentionLogInput()  # type: ignore[call-arg]

    def test_optional_source_memory(self) -> None:
        inp = IntentionLogInput(content="x", source_memory_id="abc-123")
        assert inp.source_memory_id == "abc-123"


class TestIntentionListInput:
    def test_default_limit(self) -> None:
        assert IntentionListInput().limit == 20

    def test_limit_bounds(self) -> None:
        with pytest.raises(ValidationError):
            IntentionListInput(limit=0)
        with pytest.raises(ValidationError):
            IntentionListInput(limit=101)


class TestIntentionMarkSurfacedInput:
    def test_fields_required(self) -> None:
        with pytest.raises(ValidationError):
            IntentionMarkSurfacedInput(intention_id="i-1")  # type: ignore[call-arg]

    def test_valid(self) -> None:
        inp = IntentionMarkSurfacedInput(intention_id="i-1", new_status="asked")
        assert inp.intention_id == "i-1"
        assert inp.new_status == "asked"


class TestToolError:
    def test_format(self) -> None:
        msg = _tool_error("intention_log", "embed", "provider down")
        assert msg.startswith("[TOOL ERROR] intention_log: embed failed")
        assert "provider down" in msg


class _RaisingEmbedder:
    async def aembed_query(self, text: str) -> list[float]:
        raise RuntimeError("upstream embedding outage")


class _OkEmbedder:
    async def aembed_query(self, text: str) -> list[float]:
        _ = text
        return [0.1, 0.2, 0.3]


class TestSafeAembedQuery:
    async def test_empty_text_returns_none(self) -> None:
        assert await _safe_aembed_query(_OkEmbedder(), "") is None  # type: ignore[arg-type]

    async def test_failure_soft_fails_to_none(self) -> None:
        assert await _safe_aembed_query(_RaisingEmbedder(), "want") is None  # type: ignore[arg-type]

    async def test_success_returns_vector(self) -> None:
        assert await _safe_aembed_query(_OkEmbedder(), "want") == [0.1, 0.2, 0.3]  # type: ignore[arg-type]


class TestEmitIntentionEvent:
    async def test_dispatches_event(self) -> None:
        captured: list[Any] = []

        async def _capture(event: Any, *, config: Any = None) -> None:
            captured.append(event)

        with patch("threetears.agent.intention.tools.dispatch_event", new=_capture):
            await _emit_intention_event(IntentionSurfacedEvent(agent_id="a", intention_id="i"))
        assert len(captured) == 1

    async def test_no_run_manager_runtime_error_is_swallowed(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """outside a langgraph run, the no-manager ``RuntimeError`` is logged + swallowed.

        the row is already committed by the time the event fires, so a best-effort
        stream surface must not crash the tool.
        """

        async def _raise_no_manager(event: Any, *, config: Any = None) -> None:
            raise RuntimeError("Unable to find run manager")

        with (
            patch("threetears.agent.intention.tools.dispatch_event", new=_raise_no_manager),
            caplog.at_level(logging.DEBUG, logger="threetears.agent.intention.tools"),
        ):
            # must NOT raise
            await _emit_intention_event(IntentionSurfacedEvent(agent_id="a", intention_id="i"))
        assert any("intention event dropped" in r.message for r in caplog.records)

    async def test_non_runtime_error_propagates(self) -> None:
        """a non-``RuntimeError`` (e.g. a schema regression) still surfaces."""

        async def _raise_value_error(event: Any, *, config: Any = None) -> None:
            raise ValueError("schema regression")

        with patch("threetears.agent.intention.tools.dispatch_event", new=_raise_value_error):
            with pytest.raises(ValueError, match="schema regression"):
                await _emit_intention_event(IntentionSurfacedEvent(agent_id="a", intention_id="i"))
