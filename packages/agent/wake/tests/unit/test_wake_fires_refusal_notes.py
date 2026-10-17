"""``wake_fires`` shows a refused delivery's detail on the refusal's own line.

The detail is free text from the sender's side (a parser's message, say) and
may run to many lines; it is read as one line of the refusal list, so the next
``- `` line is the next refusal.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from uuid_utils import uuid7

from threetears.agent.wake.entities import WebhookSubscriptionEntity
from threetears.agent.wake.tools.schedule_tools import load_wake_fires_tool


def _id() -> UUID:
    return UUID(str(uuid7()))


AGENT, CONV, SUB = _id(), _id(), _id()
REFUSED_AT = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
WHEN = "Oct 9, 2026 at 5:00 AM PDT"


# parity-exempt: answers only find_for_agent and list_refusals, the two reads a subscription's wake_fires makes; the full WebhookSubscriptionCollection is L3-backed and covered by the integration suite
class _FakeSubscriptions:
    """In-memory stand-in for the slice of the subscriptions collection ``wake_fires`` reads."""

    def __init__(self, refusals: list[dict[str, Any]]) -> None:
        self.refusals = refusals

    async def find_for_agent(self, agent_id: UUID, subscription_id: UUID) -> WebhookSubscriptionEntity | None:
        if agent_id != AGENT or subscription_id != SUB:
            return None
        return WebhookSubscriptionEntity(
            {
                "conversation_id": CONV,
                "subscription_id": SUB,
                "user_id": _id(),
                "agent_id": AGENT,
                "default_skill_id": None,
                "name": "deploys",
                "secret_ciphertext": b"sec",
                "allowed_source_pattern": None,
                "execution_mode": "inline",
                "task_prompt_template": "x",
                "verification_scheme": "generic_hmac_sha256",
                "status": "active",
                "rate_limit_per_minute": None,
                "last_fired_at": None,
                "date_created": REFUSED_AT,
                "date_updated": REFUSED_AT,
            },
            is_new=False,
            collection=None,
        )

    async def list_refusals(self, subscription_id: UUID, *, limit: int) -> list[dict[str, Any]]:
        _ = subscription_id, limit
        return self.refusals


# parity-exempt: answers only list_for_subscription, the one read a subscription's wake_fires makes of fires; the full WakeFireCollection is L3-backed and covered by the integration suite
class _FakeNoFires:
    """In-memory stand-in for a fires collection holding no fires."""

    async def list_for_subscription(self, conversation_id: UUID, subscription_id: UUID, *, limit: int) -> list[Any]:
        _ = conversation_id, subscription_id, limit
        return []


def _refusal(detail: str | None, reason: str = "bad_payload") -> dict[str, Any]:
    return {
        "reason": reason,
        "detail": detail,
        "refusals": 3,
        "last_refused_at": REFUSED_AT,
        "status_code": 400,
    }


async def _refusal_lines(refusals: list[dict[str, Any]]) -> list[str]:
    [tool] = load_wake_fires_tool(
        agent_id=AGENT,
        schedules_collection=None,  # type: ignore[arg-type]
        fires_collection=_FakeNoFires(),  # type: ignore[arg-type]
        subscriptions_collection=_FakeSubscriptions(refusals),  # type: ignore[arg-type]
        tz="America/Los_Angeles",
    )
    out = await tool.ainvoke({"subscription_id": str(SUB)})
    lines = out.splitlines()
    return lines[lines.index("Refused deliveries, newest first, counted per reason and minute:") + 1 :]


class TestRefusalNotes:
    async def test_a_many_line_detail_is_one_line_of_its_refusal(self) -> None:
        lines = await _refusal_lines(
            [_refusal("expected a JSON object\n\n  got a list\nat line 1"), _refusal(None, "paused")]
        )
        assert lines == [
            f"- {WHEN} · 3 refused · the body could not be read (expected a JSON object got a list at line 1)"
            " · answered 400",
            f"- {WHEN} · 3 refused · the subscription was paused · answered 400",
        ]

    async def test_a_detail_with_nothing_in_it_still_shows_the_refusal_line(self) -> None:
        lines = await _refusal_lines([_refusal("  \n \n")])
        assert lines == [f"- {WHEN} · 3 refused · the body could not be read · answered 400"]
