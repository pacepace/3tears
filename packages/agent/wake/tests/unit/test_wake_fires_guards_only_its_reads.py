"""``wake_fires`` answers a history it could not read, and only that.

A read that fails is the tool's reply, "the history could not be read". What
the tool does with rows it did read is its own code: a fault there is a bug
and propagates, rather than reading to the agent as an unreadable history.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from uuid_utils import uuid7

from threetears.core.exceptions import DataLayerUnavailableError
from threetears.agent.wake.tools.schedule_tools import load_wake_fires_tool


def _id() -> UUID:
    return UUID(str(uuid7()))


AGENT, CONV, SCHEDULE, SUB = _id(), _id(), _id(), _id()
AT = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


# parity-exempt: answers only find_for_agent, the one read a schedule's wake_fires makes of schedules; the full WakeScheduleCollection is L3-backed and covered by the integration suite
class _FakeSchedules:
    """Hands back ``entity`` for the one schedule, or raises ``failure``."""

    def __init__(self, entity: Any = None, failure: Exception | None = None) -> None:
        self.entity = entity
        self.failure = failure

    async def find_for_agent(self, agent_id: UUID, schedule_id: UUID) -> Any:
        _ = agent_id, schedule_id
        if self.failure is not None:
            raise self.failure
        return self.entity


# parity-exempt: answers only list_for_schedule and list_for_subscription, the reads wake_fires makes of fires; the full WakeFireCollection is L3-backed and covered by the integration suite
class _FakeFires:
    """Holds no fires."""

    async def list_for_schedule(self, conversation_id: UUID, schedule_id: UUID, *, limit: int) -> list[Any]:
        _ = conversation_id, schedule_id, limit
        return []

    async def list_for_subscription(self, conversation_id: UUID, subscription_id: UUID, *, limit: int) -> list[Any]:
        _ = conversation_id, subscription_id, limit
        return []


# parity-exempt: answers only find_for_agent and list_refusals, the two reads a subscription's wake_fires makes; the full WebhookSubscriptionCollection is L3-backed and covered by the integration suite
class _FakeSubscriptions:
    """Holds one subscription; its refusals are ``refusals``, or the read raises ``failure``."""

    def __init__(self, refusals: list[dict[str, Any]], failure: Exception | None = None) -> None:
        self.refusals = refusals
        self.failure = failure

    async def find_for_agent(self, agent_id: UUID, subscription_id: UUID) -> Any:
        _ = agent_id, subscription_id
        return SimpleNamespace(conversation_id=CONV, name="deploys", status="active", last_fired_at=None)

    async def list_refusals(self, subscription_id: UUID, *, limit: int) -> list[dict[str, Any]]:
        _ = subscription_id, limit
        if self.failure is not None:
            raise self.failure
        return self.refusals


async def _of_schedule(schedules: _FakeSchedules) -> str:
    [tool] = load_wake_fires_tool(
        agent_id=AGENT,
        schedules_collection=schedules,  # type: ignore[arg-type]
        fires_collection=_FakeFires(),  # type: ignore[arg-type]
    )
    out = await tool.ainvoke({"schedule_id": str(SCHEDULE)})
    assert isinstance(out, str)
    return out


async def _of_subscription(subscriptions: _FakeSubscriptions) -> str:
    [tool] = load_wake_fires_tool(
        agent_id=AGENT,
        schedules_collection=None,  # type: ignore[arg-type]
        fires_collection=_FakeFires(),  # type: ignore[arg-type]
        subscriptions_collection=subscriptions,  # type: ignore[arg-type]
    )
    out = await tool.ainvoke({"subscription_id": str(SUB)})
    assert isinstance(out, str)
    return out


class TestAReadThatFailedSaysSo:
    async def test_a_schedule_that_could_not_be_read(self) -> None:
        out = await _of_schedule(_FakeSchedules(failure=ConnectionError("l3 timeout")))
        assert out == "[TOOL ERROR] wake_fires: the history could not be read (ConnectionError)"

    async def test_refusals_that_could_not_be_read(self) -> None:
        out = await _of_subscription(_FakeSubscriptions([], failure=DataLayerUnavailableError("broker down")))
        assert out == "[TOOL ERROR] wake_fires: the history could not be read (DataLayerUnavailableError)"


class TestAFaultInTheWritingOutPropagates:
    async def test_a_schedule_the_line_cannot_be_written_from(self) -> None:
        # read fine; it carries no schedule_type, which the head line reads
        entity = SimpleNamespace(conversation_id=CONV, name="nightly")
        with pytest.raises(AttributeError, match="schedule_type"):
            await _of_schedule(_FakeSchedules(entity=entity))

    async def test_a_refusal_row_the_line_cannot_be_written_from(self) -> None:
        # read fine; it carries no last_refused_at, which the refusal line reads
        row = {"reason": "bad_payload", "detail": None, "refusals": 3, "status_code": 400}
        with pytest.raises(KeyError, match="last_refused_at"):
            await _of_subscription(_FakeSubscriptions([row]))

    async def test_a_readable_history_still_reads(self) -> None:
        row = {"reason": "bad_payload", "detail": None, "refusals": 3, "status_code": 400, "last_refused_at": AT}
        out = await _of_subscription(_FakeSubscriptions([row]))
        assert "3 refused · the body could not be read · answered 400" in out
