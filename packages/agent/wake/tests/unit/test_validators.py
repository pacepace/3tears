"""Unit tests for :mod:`threetears.agent.wake.tools.validators`.

Covers every schedule_type's config validator + the context_from
chain walker. No DB; the chain walker uses an in-memory resolver fake.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from threetears.agent.wake.tools.validators import (
    CONTEXT_FROM_MAX_DEPTH,
    SUPPORTED_SCHEDULE_TYPES,
    _ChainNode,
    validate_context_from_chain,
    validate_schedule_config,
)


# ---------------------------------------------------------------------------
# validate_schedule_config
# ---------------------------------------------------------------------------


def test_supported_schedule_types_matches_documented_set() -> None:
    assert SUPPORTED_SCHEDULE_TYPES == {
        "daily_at",
        "every_n_hours",
        "random_within_window",
        "one_shot_at",
        "cron",
        "relative_delay",
        "interval",
    }


def test_validate_schedule_config_rejects_unknown_type() -> None:
    err = validate_schedule_config("bogus", {})
    assert err is not None
    assert "unknown schedule_type" in err


def test_validate_schedule_config_rejects_non_dict_config() -> None:
    err = validate_schedule_config("daily_at", "not-a-dict")  # type: ignore[arg-type]
    assert err == "schedule_config must be a JSON object"


# daily_at ------------------------------------------------------------------


def test_daily_at_valid() -> None:
    assert (
        validate_schedule_config(
            "daily_at",
            {"hour": 9, "minute": 30, "tz": "UTC"},
        )
        is None
    )


def test_daily_at_missing_hour() -> None:
    err = validate_schedule_config("daily_at", {"minute": 0})
    assert err is not None
    assert "'hour'" in err


def test_daily_at_hour_out_of_range() -> None:
    err = validate_schedule_config("daily_at", {"hour": 24})
    assert err is not None and "[0, 23]" in err


def test_daily_at_minute_out_of_range() -> None:
    err = validate_schedule_config("daily_at", {"hour": 9, "minute": 60})
    assert err is not None and "[0, 59]" in err


# every_n_hours -------------------------------------------------------------


def test_every_n_hours_valid() -> None:
    assert validate_schedule_config("every_n_hours", {"n": 3}) is None


def test_every_n_hours_zero_rejected() -> None:
    err = validate_schedule_config("every_n_hours", {"n": 0})
    assert err is not None and "positive" in err


# random_within_window ------------------------------------------------------


def test_random_within_window_valid() -> None:
    assert (
        validate_schedule_config(
            "random_within_window",
            {"start_hour": 9, "end_hour": 21, "tz": "UTC"},
        )
        is None
    )


def test_random_within_window_equal_hours_rejected() -> None:
    err = validate_schedule_config(
        "random_within_window",
        {"start_hour": 9, "end_hour": 9},
    )
    assert err is not None and "start_hour != end_hour" in err


# one_shot_at ---------------------------------------------------------------


def test_one_shot_at_future() -> None:
    now = datetime.now(UTC)
    future = (now + timedelta(hours=1)).isoformat()
    assert validate_schedule_config("one_shot_at", {"fire_at_iso": future}, now=now) is None


def test_one_shot_at_past_rejected() -> None:
    now = datetime.now(UTC)
    past = (now - timedelta(minutes=1)).isoformat()
    err = validate_schedule_config("one_shot_at", {"fire_at_iso": past}, now=now)
    assert err is not None and "future" in err


def test_one_shot_at_malformed_iso() -> None:
    err = validate_schedule_config("one_shot_at", {"fire_at_iso": "not-iso"})
    assert err is not None and "ISO 8601" in err


# cron ----------------------------------------------------------------------


def test_cron_valid() -> None:
    assert validate_schedule_config("cron", {"expr": "0 */3 * * *"}) is None


def test_cron_invalid_expr_rejected() -> None:
    err = validate_schedule_config("cron", {"expr": "this is not cron"})
    assert err is not None and "cron" in err


# relative_delay ------------------------------------------------------------


@pytest.mark.parametrize("delay", ["30s", "30m", "2h", "1d"])
def test_relative_delay_valid_units(delay: str) -> None:
    assert validate_schedule_config("relative_delay", {"delay": delay}) is None


def test_relative_delay_invalid_unit_rejected() -> None:
    err = validate_schedule_config("relative_delay", {"delay": "30x"})
    assert err is not None


def test_relative_delay_zero_rejected() -> None:
    err = validate_schedule_config("relative_delay", {"delay": "0m"})
    assert err is not None and "positive" in err


# interval ------------------------------------------------------------------


def test_interval_valid() -> None:
    assert validate_schedule_config("interval", {"seconds": 60}) is None


def test_interval_zero_rejected() -> None:
    err = validate_schedule_config("interval", {"seconds": 0})
    assert err is not None and "positive" in err


# ---------------------------------------------------------------------------
# validate_context_from_chain
# ---------------------------------------------------------------------------


class _InMemoryResolver:
    """Tiny resolver feeding the chain walker chain nodes."""

    def __init__(self, conv_id: UUID, edges: dict[UUID, UUID | None]) -> None:
        self._conv = conv_id
        self._edges = edges

    async def __call__(self, schedule_id: UUID) -> _ChainNode | None:
        if schedule_id not in self._edges:
            return None
        return _ChainNode(
            conversation_id=self._conv,
            context_from_schedule_id=self._edges[schedule_id],
        )


@pytest.mark.asyncio
async def test_context_from_clean_chain_accepted() -> None:
    conv = uuid4()
    target = uuid4()
    upstream = uuid4()
    new = uuid4()
    resolver = _InMemoryResolver(conv, {target: upstream, upstream: None})
    err = await validate_context_from_chain(
        new_schedule_id=new,
        proposed_context_from=target,
        conversation_id=conv,
        resolver=resolver,
    )
    assert err is None


@pytest.mark.asyncio
async def test_context_from_self_reference_rejected() -> None:
    conv = uuid4()
    new = uuid4()
    resolver = _InMemoryResolver(conv, {})
    err = await validate_context_from_chain(
        new_schedule_id=new,
        proposed_context_from=new,
        conversation_id=conv,
        resolver=resolver,
    )
    assert err is not None and "cycle" in err


@pytest.mark.asyncio
async def test_context_from_cycle_rejected() -> None:
    conv = uuid4()
    a = uuid4()
    b = uuid4()
    # A -> B -> A
    resolver = _InMemoryResolver(conv, {a: b, b: a})
    new = uuid4()
    err = await validate_context_from_chain(
        new_schedule_id=new,
        proposed_context_from=a,
        conversation_id=conv,
        resolver=resolver,
    )
    assert err is not None and "cycle" in err


@pytest.mark.asyncio
async def test_context_from_missing_target_rejected() -> None:
    conv = uuid4()
    resolver = _InMemoryResolver(conv, {})
    err = await validate_context_from_chain(
        new_schedule_id=uuid4(),
        proposed_context_from=uuid4(),
        conversation_id=conv,
        resolver=resolver,
    )
    assert err is not None and "not found" in err


@pytest.mark.asyncio
async def test_context_from_cross_conversation_rejected() -> None:
    conv_a = uuid4()
    conv_b = uuid4()
    target = uuid4()

    class _CrossConvResolver:
        async def __call__(self, schedule_id: UUID) -> _ChainNode | None:
            del schedule_id
            return _ChainNode(
                conversation_id=conv_b,
                context_from_schedule_id=None,
            )

    err = await validate_context_from_chain(
        new_schedule_id=uuid4(),
        proposed_context_from=target,
        conversation_id=conv_a,
        resolver=_CrossConvResolver(),
    )
    assert err is not None and "different conversation" in err


@pytest.mark.asyncio
async def test_context_from_max_depth_exceeded() -> None:
    conv = uuid4()
    # build a chain that exceeds the max depth
    nodes = [uuid4() for _ in range(CONTEXT_FROM_MAX_DEPTH + 2)]
    edges: dict[UUID, UUID | None] = {}
    for i, nid in enumerate(nodes):
        edges[nid] = nodes[i + 1] if i + 1 < len(nodes) else None
    resolver = _InMemoryResolver(conv, edges)
    err = await validate_context_from_chain(
        new_schedule_id=uuid4(),
        proposed_context_from=nodes[0],
        conversation_id=conv,
        resolver=resolver,
    )
    assert err is not None and "max depth" in err
