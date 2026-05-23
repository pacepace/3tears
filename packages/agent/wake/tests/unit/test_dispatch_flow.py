"""Unit tests for :func:`threetears.agent.wake.dispatch.dispatch_wake`.

Pool-free unit cases: with ``pool=None`` the context_from + skill
resolvers short-circuit to ``()`` / ``None`` so the handler-invocation
+ ``[SILENT]`` detection + delivery routing paths can be exercised
without touching the database. The DB-touching paths
(``context_from`` chain resolution, skill resolution) live in the
integration tests.

Covered branches:

- handler returns ``status='fired'`` with normal content -> result
  carries ``status='fired'``, ``display_suppressed=False``, output
  text preserved
- handler returns ``status='fired'`` with ``[SILENT]`` content ->
  result promoted to ``status='fired_silent'``,
  ``display_suppressed=True``, no delivery
- handler returns explicit ``status='fired_silent'`` -> result keeps
  that status, ``display_suppressed=True``
- handler returns ``status='yielded'`` -> result carries that status
  through unchanged
- handler returns ``status='failed'`` with error -> dispatch result
  preserves both fields
- handler raises -> ``dispatch_wake`` propagates (the tick caller
  handles the exception per shard-02 contract)
- delivery_target='email' + adapter registered -> adapter invoked
  with the handler result, returned status recorded
- delivery_target='email' + NO adapter registered -> result's
  ``delivery_status['email'] == 'no_adapter'``, NO raise
- delivery_target='email' + adapter raises -> result's
  ``delivery_status['email'] == 'failed'``, NO raise out of dispatch
- delivery_target='email' + [SILENT] -> adapter NOT invoked + result's
  ``delivery_status['email'] == 'skipped_silent'``
- handler explicitly returns ``status='fired_silent'`` WITHOUT the
  ``[SILENT]`` marker -> the self-reported silent is authoritative:
  ``display_suppressed=True`` AND delivery is skipped (matches the
  fired_silent + marker case for coherence)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from uuid_utils import uuid7

from threetears.agent.wake.dispatch import dispatch_wake
from threetears.agent.wake.types import (
    DeliveryAdapter,
    HandlerCallback,
    HandlerCallbackResult,
    PreparedWakeContext,
    WakeTrigger,
)


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


def _make_trigger(
    *,
    delivery_target: str = "conversation",
    skill_id: UUID | None = None,
    context_from_schedule_id: UUID | None = None,
) -> WakeTrigger:
    return WakeTrigger(
        schedule_id=_new_uuid(),
        user_id=_new_uuid(),
        agent_id=_new_uuid(),
        conversation_id=_new_uuid(),
        fire_source="scheduled_tick",
        execution_mode="inline",
        schedule_type="interval",
        fired_at=datetime.now(UTC),
        schedule_name="unit-test",
        delivery_target=delivery_target,
        skill_id=skill_id,
        context_from_schedule_id=context_from_schedule_id,
    )


# parity-with: threetears.agent.wake.types.HandlerCallback
class _StubHandler(HandlerCallback):
    """Records the prepared context handed to it; returns a configured result."""

    def __init__(self, result: HandlerCallbackResult) -> None:
        self._result = result
        self.invocations: list[tuple[WakeTrigger, PreparedWakeContext]] = []

    async def __call__(
        self,
        trigger: WakeTrigger,
        prepared_context: PreparedWakeContext,
        pool: Any,
    ) -> HandlerCallbackResult:
        del pool
        self.invocations.append((trigger, prepared_context))
        return self._result


# parity-with: threetears.agent.wake.types.HandlerCallback
class _RaisingHandler(HandlerCallback):
    """Raises a configured exception so the propagation path can be asserted."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def __call__(
        self,
        trigger: WakeTrigger,
        prepared_context: PreparedWakeContext,
        pool: Any,
    ) -> HandlerCallbackResult:
        del trigger, prepared_context, pool
        raise self._exc


# parity-with: threetears.agent.wake.types.DeliveryAdapter
class _RecordingAdapter(DeliveryAdapter):
    """Records every invocation; returns a configured status."""

    def __init__(self, status: str = "delivered") -> None:
        self._status = status
        self.calls: list[tuple[WakeTrigger, HandlerCallbackResult]] = []

    async def deliver(
        self,
        trigger: WakeTrigger,
        prepared_context: PreparedWakeContext,
        handler_result: HandlerCallbackResult,
        pool: Any,
    ) -> str:
        del prepared_context, pool
        self.calls.append((trigger, handler_result))
        return self._status


# parity-with: threetears.agent.wake.types.DeliveryAdapter
class _RaisingAdapter(DeliveryAdapter):
    """Raises on deliver -- pins the "delivery failure must not escape" branch."""

    async def deliver(
        self,
        trigger: WakeTrigger,
        prepared_context: PreparedWakeContext,
        handler_result: HandlerCallbackResult,
        pool: Any,
    ) -> str:
        del trigger, prepared_context, handler_result, pool
        raise RuntimeError("smtp connection refused")


class TestHappyPathFiredStatus:
    """``status='fired'`` with normal content -> result preserves output."""

    async def test_fired_status_preserved(self) -> None:
        trigger = _make_trigger()
        fire_id = _new_uuid()
        handler = _StubHandler(
            HandlerCallbackResult(
                status="fired",
                assistant_message_content="all clear — no anomalies detected",
                target_conversation_id=trigger.conversation_id,
                assistant_message_id=_new_uuid(),
                latency_ms=42,
            ),
        )
        result = await dispatch_wake(
            trigger,
            fire_id,
            pool=None,
            handler=handler,
        )
        assert result.status == "fired"
        assert result.display_suppressed is False
        assert result.output_text == "all clear — no anomalies detected"
        assert result.latency_ms == 42
        assert result.error is None
        assert len(handler.invocations) == 1
        # prepared context defaults: no skill, no blocks
        prepared = handler.invocations[0][1]
        assert prepared.attached_skill is None
        assert prepared.context_blocks == ()
        assert prepared.trigger is trigger


class TestSilentPromotion:
    """``[SILENT]`` content auto-promotes ``status='fired'`` to ``'fired_silent'``."""

    async def test_silent_prefix_flips_status(self) -> None:
        trigger = _make_trigger()
        handler = _StubHandler(
            HandlerCallbackResult(
                status="fired",
                assistant_message_content="[SILENT] watchdog observed nothing new",
                target_conversation_id=trigger.conversation_id,
            ),
        )
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
        )
        assert result.status == "fired_silent"
        assert result.display_suppressed is True
        # The marker is preserved in output_text so audit history can
        # reproduce what the agent actually generated; the handler is
        # responsible for stripping it from its visible messages table.
        assert result.output_text is not None
        assert result.output_text.startswith("[SILENT]")

    async def test_explicit_fired_silent_kept(self) -> None:
        trigger = _make_trigger()
        handler = _StubHandler(
            HandlerCallbackResult(
                status="fired_silent",
                assistant_message_content="[SILENT] explicit silent",
                target_conversation_id=trigger.conversation_id,
            ),
        )
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
        )
        assert result.status == "fired_silent"
        assert result.display_suppressed is True

    async def test_explicit_fired_silent_without_marker_is_authoritative(self) -> None:
        """Handler returns ``status='fired_silent'`` with NO marker -- still silent.

        Pins the coherence-asymmetry resolution (Critic finding #3): a
        handler's explicit ``status='fired_silent'`` is authoritative
        even when the assistant text lacks the ``[SILENT]`` prefix.
        Without this, the marker would be the sole signal and the
        result would carry ``display_suppressed=False`` alongside
        ``status='fired_silent'`` -- an internally inconsistent state.
        """
        trigger = _make_trigger(delivery_target="email")
        adapter = _RecordingAdapter(status="delivered")
        handler = _StubHandler(
            HandlerCallbackResult(
                status="fired_silent",
                assistant_message_content="hello world",  # NO [SILENT] marker
                target_conversation_id=trigger.conversation_id,
            ),
        )
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
            delivery_adapters={"email": adapter},
        )
        # status stays fired_silent (the handler's self-report wins)
        assert result.status == "fired_silent"
        # display_suppressed is True even without the marker -- the
        # explicit fired_silent self-report is enough on its own
        assert result.display_suppressed is True
        # delivery routing is skipped on silent fires regardless of
        # marker presence -- shard-05 distinguishes "no adapter" from
        # "agent chose silence" via this string
        assert adapter.calls == []
        assert result.delivery_status == {"email": "skipped_silent"}


class TestYieldedStatusPropagates:
    """``status='yielded'`` survives ``dispatch_wake`` unchanged (wake-yield)."""

    async def test_yielded_passthrough(self) -> None:
        trigger = _make_trigger()
        handler = _StubHandler(
            HandlerCallbackResult(
                status="yielded",
                assistant_message_content="wrapping up; user message coming",
                target_conversation_id=trigger.conversation_id,
            ),
        )
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
        )
        assert result.status == "yielded"
        assert result.display_suppressed is False


class TestFailedStatusPreservesError:
    """``status='failed'`` with error string is forwarded onto the dispatch result.

    Pins the Critic-flagged asymmetry from shard-02: a non-raising
    failure carrying ``error`` must reach the tick's
    ``finalize_failed`` path via the typed return value.
    """

    async def test_failed_status_and_error_preserved(self) -> None:
        trigger = _make_trigger()
        handler = _StubHandler(
            HandlerCallbackResult(
                status="failed",
                assistant_message_content="",
                target_conversation_id=trigger.conversation_id,
                error="downstream rate-limited",
                latency_ms=17,
            ),
        )
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
        )
        assert result.status == "failed"
        assert result.error == "downstream rate-limited"
        assert result.latency_ms == 17


class TestHandlerExceptionPropagates:
    """Raising handlers escape ``dispatch_wake`` — tick caller catches them."""

    async def test_runtime_error_propagates(self) -> None:
        trigger = _make_trigger()
        handler = _RaisingHandler(RuntimeError("LLM service unavailable"))
        with pytest.raises(RuntimeError, match="LLM service unavailable"):
            await dispatch_wake(
                trigger,
                _new_uuid(),
                pool=None,
                handler=handler,
            )


class TestDeliveryRoutingEmail:
    """Non-conversation delivery routes through the registered adapter."""

    async def test_email_adapter_invoked_on_fired(self) -> None:
        trigger = _make_trigger(delivery_target="email")
        adapter = _RecordingAdapter(status="delivered")
        handler = _StubHandler(
            HandlerCallbackResult(
                status="fired",
                assistant_message_content="daily digest",
                target_conversation_id=trigger.conversation_id,
            ),
        )
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
            delivery_adapters={"email": adapter},
        )
        assert result.status == "fired"
        assert len(adapter.calls) == 1
        # caller passed the trigger + handler result through
        recv_trigger, recv_result = adapter.calls[0]
        assert recv_trigger.delivery_target == "email"
        assert recv_result.assistant_message_content == "daily digest"
        # adapter's returned status string lands on delivery_status
        # under the target key for shard-05 metrics
        assert result.delivery_status == {"email": "delivered"}

    async def test_email_adapter_not_invoked_on_silent(self) -> None:
        trigger = _make_trigger(delivery_target="email")
        adapter = _RecordingAdapter(status="delivered")
        handler = _StubHandler(
            HandlerCallbackResult(
                status="fired",
                assistant_message_content="[SILENT] no digest today",
                target_conversation_id=trigger.conversation_id,
            ),
        )
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
            delivery_adapters={"email": adapter},
        )
        assert result.status == "fired_silent"
        assert adapter.calls == []
        # silent fires record 'skipped_silent' so shard-05 metrics can
        # distinguish "no adapter" from "agent chose silence"
        assert result.delivery_status == {"email": "skipped_silent"}

    async def test_no_adapter_registered_records_audit_status(self) -> None:
        """Missing adapter records ``delivery_status['email']='no_adapter'``."""
        trigger = _make_trigger(delivery_target="email")
        handler = _StubHandler(
            HandlerCallbackResult(
                status="fired",
                assistant_message_content="digest",
                target_conversation_id=trigger.conversation_id,
            ),
        )
        # passes no delivery_adapters at all -- the registry default is empty
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
        )
        # delivery failed to route, but the fire still succeeded
        assert result.status == "fired"
        # The audit string -- promised by the test name, delivered by
        # delivery_status (Critic finding #6 + #1: previously the
        # status string lived only in log output and the test was
        # asserting only the fire status; now WakeDispatchResult
        # surfaces it so the audit claim has teeth).
        assert result.delivery_status == {"email": "no_adapter"}

    async def test_adapter_raise_does_not_escape(self) -> None:
        trigger = _make_trigger(delivery_target="email")
        adapter = _RaisingAdapter()
        handler = _StubHandler(
            HandlerCallbackResult(
                status="fired",
                assistant_message_content="digest",
                target_conversation_id=trigger.conversation_id,
            ),
        )
        # raising adapter is logged + recorded, but the dispatch
        # completes -- the assistant message already landed in the
        # conversation, so a failed side-channel must not invalidate
        # the fire.
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
            delivery_adapters={"email": adapter},
        )
        assert result.status == "fired"
        assert result.delivery_status == {"email": "failed"}

    async def test_conversation_target_records_empty_delivery_status(self) -> None:
        """``delivery_target='conversation'`` is a no-op -- delivery_status stays empty.

        The handler placed the message in the conversation; there is
        no side-channel to route. shard-05's emit treats an empty dict
        as "no delivery routing attempted" and does NOT increment any
        delivery counter.
        """
        trigger = _make_trigger(delivery_target="conversation")
        handler = _StubHandler(
            HandlerCallbackResult(
                status="fired",
                assistant_message_content="visible reply",
                target_conversation_id=trigger.conversation_id,
            ),
        )
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
        )
        assert result.status == "fired"
        assert result.delivery_status == {}


class TestLatencyComputedWhenAbsent:
    """``dispatch_wake`` synthesises latency_ms when the handler omits it."""

    async def test_latency_synthesised(self) -> None:
        trigger = _make_trigger()
        handler = _StubHandler(
            HandlerCallbackResult(
                status="fired",
                assistant_message_content="ok",
                target_conversation_id=trigger.conversation_id,
                latency_ms=None,
            ),
        )
        result = await dispatch_wake(
            trigger,
            _new_uuid(),
            pool=None,
            handler=handler,
        )
        # synthesised value is monotonic-clock measured; non-negative
        # integer is the contract floor.
        assert result.latency_ms is not None
        assert result.latency_ms >= 0
