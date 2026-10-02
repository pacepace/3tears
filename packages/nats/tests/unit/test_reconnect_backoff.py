"""unit tests for the jittered reconnect backoff (resilience-task-06, thundering-herd defense).

these tests exercise the ``reconnect_to_server_handler`` :meth:`NatsClient.connect` hands nats-py,
read straight out of the connect options, WITHOUT a live broker. they assert what the shard requires:

* the reconnect delay is FULL jitter within ``[0, min(cap, base * 2**attempt)]`` and its ceiling
  GROWS with the per-attempt count (``Server.reconnects``) until ``cap`` clamps it;
* the connect options carry the jittered handler -- a jitter FUNCTION, not a fixed reconnect interval;
* two independent draws de-synchronize (a mass reconnect spreads instead of thundering-herd-ing).

The draw comes from the stdlib ``random.uniform``; a test that needs one end of the jitter window
pins it there with :func:`_pin_draw` instead of sampling and hoping.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlparse

import pytest
from nats.aio.client import Server

from threetears.nats import NatsClient
from threetears.nats.client import (
    DEFAULT_RECONNECT_BACKOFF_BASE_SECONDS,
    DEFAULT_RECONNECT_BACKOFF_CAP_SECONDS,
)

#: the floor the handler never returns below, so a near-zero jitter draw cannot hot-spin nats-py's
#: reconnect path. The observable value the client promises, restated so a change to it is a decision.
_FLOOR_SECONDS = 0.05

#: the whole-second pace the handler returns when computing a delay raises, so nats-py always has a
#: delay to sleep instead of busy-spinning.
_FALLBACK_SECONDS = 1.0

_Handler = Callable[[list[Server], dict[str, Any]], tuple[Server | None, float]]


def _handler(*, base: float, cap: float) -> _Handler:
    """the reconnect handler a client connected with ``base``/``cap`` hands nats-py.

    :param base: the ``reconnect_backoff_base`` passed to :meth:`NatsClient.connect`
    :ptype base: float
    :param cap: the ``reconnect_backoff_cap`` passed to :meth:`NatsClient.connect`
    :ptype cap: float
    :return: the ``reconnect_to_server_handler`` that reached the connect options
    :rtype: _Handler
    """
    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    async def _connect() -> None:
        await NatsClient.connect(
            establish_connection=_fake_establish,
            nats_url="nats://localhost:4222",
            nats_subject_namespace="3tears",
            client_name="agent-x",
            verify_jetstream=False,
            reconnect_backoff_base=base,
            reconnect_backoff_cap=cap,
        )

    asyncio.run(_connect())
    handler: _Handler = captured["options"]["reconnect_to_server_handler"]
    return handler


def _pin_draw(monkeypatch: pytest.MonkeyPatch, fraction: float) -> None:
    """make every jitter draw land at ``fraction`` of its window: 0.0 is the low end, 1.0 the high.

    :param monkeypatch: pytest monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :param fraction: where in ``[a, b]`` each ``random.uniform(a, b)`` lands
    :ptype fraction: float
    :return: nothing
    :rtype: None
    """
    monkeypatch.setattr(random, "uniform", lambda a, b: a + (b - a) * fraction)


def _server(reconnects: int) -> Server:
    """build a nats-py ``Server`` snapshot entry with a given reconnect count.

    :param reconnects: the per-server attempt count the handler keys its backoff on
    :ptype reconnects: int
    :return: a ``Server`` the handler can select and read ``reconnects`` from
    :rtype: Server
    """
    return Server(uri=urlparse("nats://localhost:4222"), reconnects=reconnects)


def _delay(handler: _Handler, attempt: int) -> float:
    """the delay ``handler`` returns for a server on its ``attempt``-th reconnect.

    :param handler: the reconnect handler
    :ptype handler: _Handler
    :param attempt: the server's reconnect count
    :ptype attempt: int
    :return: the delay nats-py would sleep
    :rtype: float
    """
    return handler([_server(attempt)], {})[1]


# ---------------------------------------------------------------------------
# full-jitter ceiling + growth
# ---------------------------------------------------------------------------


def test_full_jitter_delay_stays_within_ceiling() -> None:
    """every draw lands within ``[floor, min(cap, base * 2**attempt)]`` across a range of attempts."""
    base, cap = 1.0, 30.0
    handler = _handler(base=base, cap=cap)
    for attempt in range(0, 12):
        ceiling = min(cap, base * (2**attempt))
        for _ in range(50):
            delay = _delay(handler, attempt)
            assert _FLOOR_SECONDS <= delay <= ceiling


def test_full_jitter_ceiling_grows_with_attempt_until_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """pinned to the HIGH end, the delay equals the ceiling and grows with attempt until the cap clamps."""
    base, cap = 1.0, 30.0
    handler = _handler(base=base, cap=cap)
    _pin_draw(monkeypatch, 1.0)
    delays = [_delay(handler, attempt) for attempt in range(0, 8)]
    # strictly increasing while below the cap: 1, 2, 4, 8, 16, then clamped at 30, 30, 30.
    assert delays[0] == pytest.approx(1.0)
    assert delays[1] == pytest.approx(2.0)
    assert delays[4] == pytest.approx(16.0)
    # once base * 2**attempt exceeds cap, the ceiling is clamped -- never unbounded.
    assert delays[5] == pytest.approx(cap)
    assert delays[7] == pytest.approx(cap)
    # monotonic non-decreasing throughout.
    assert all(delays[i] <= delays[i + 1] for i in range(len(delays) - 1))


def test_full_jitter_spans_full_range_low_and_high(monkeypatch: pytest.MonkeyPatch) -> None:
    """FULL jitter: the low end is 0 (floored) and the high end is the ceiling, not an equal-jitter floor.

    Equal jitter would keep the low end at half the ceiling (4s at attempt 3); full jitter lets it
    reach zero, which the handler's hot-spin floor then lifts to the floor and no further.
    """
    handler = _handler(base=1.0, cap=30.0)
    _pin_draw(monkeypatch, 0.0)
    assert _delay(handler, 3) == pytest.approx(_FLOOR_SECONDS)
    _pin_draw(monkeypatch, 1.0)
    assert _delay(handler, 3) == pytest.approx(8.0)


def test_full_jitter_caps_high_attempt() -> None:
    """a large attempt count never exceeds the cap -- single-agent recovery stays prompt."""
    delay = _delay(_handler(base=1.0, cap=30.0), 100)
    assert 0.0 <= delay <= 30.0


def test_full_jitter_two_instances_desynchronize() -> None:
    """independent draws produce spread (no synchronization) -- the whole point of jitter."""
    handler = _handler(base=1.0, cap=30.0)
    draws = {_delay(handler, 5) for _ in range(200)}
    # a constant delay would collapse to a single value; jitter yields many distinct draws.
    assert len(draws) > 100


def test_full_jitter_does_not_overflow_on_huge_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """REGRESSION: a long-running reconnect (attempt past ~1024) must NOT raise OverflowError.

    with an unclamped exponent, ``base * 2**attempt`` for a large attempt is a >308-digit int, so the
    float multiply raised ``OverflowError`` ("int too large to convert to float"), which escaped the
    reconnect handler, broke every reconnect attempt, and busy-spun the client at 100% CPU while its
    tools silently deregistered. the exponent clamp keeps the delay bounded by the cap, no exception
    -- and pinned to the high end, the delay is the cap itself rather than the handler's fallback.
    """
    handler = _handler(base=1.0, cap=30.0)
    _pin_draw(monkeypatch, 1.0)
    for attempt in (1024, 5000, 100_000):
        assert _delay(handler, attempt) == pytest.approx(30.0)  # clamped to cap, never overflow


def test_handler_survives_huge_reconnect_count() -> None:
    """the handler keyed on a huge ``Server.reconnects`` returns a bounded, floored delay -- no raise."""
    handler = _handler(base=1.0, cap=30.0)
    selected, delay = handler([_server(50_000)], {})
    assert selected is not None
    assert _FLOOR_SECONDS <= delay <= 30.0


def test_handler_floors_delay_to_prevent_hot_spin() -> None:
    """every handler delay is >= the floor, so a near-zero jitter draw can never let nats-py hot-spin."""
    handler = _handler(base=1.0, cap=30.0)
    # attempt 0 has ceiling == base == 1.0, so raw draws land in [0, 1] and many fall below the floor;
    # every returned delay must still be floored, never 0.
    assert all(_delay(handler, 0) >= _FLOOR_SECONDS for _ in range(500))


def test_handler_never_raises_fails_safe_to_fallback() -> None:
    """if delay computation raises, the handler returns the whole-second fallback rather than escaping.

    a raising sync handler gives nats-py no delay to sleep, so it busy-spins the reconnect path -- the
    handler must fail safe instead.
    """

    class _BadServer:
        """a server snapshot whose ``reconnects`` access raises, forcing the handler's except path."""

        @property
        def reconnects(self) -> int:
            raise ValueError("boom")

    handler = _handler(base=1.0, cap=30.0)
    selected, delay = handler([_BadServer()], {})  # type: ignore[list-item]
    assert delay == pytest.approx(_FALLBACK_SECONDS)


# ---------------------------------------------------------------------------
# reconnect_to_server_handler
# ---------------------------------------------------------------------------


def test_handler_selects_first_server_and_jitters_on_its_reconnects() -> None:
    """the handler picks the first eligible server and returns a jittered delay keyed on its reconnects."""
    handler = _handler(base=1.0, cap=30.0)
    servers = [_server(4), _server(0)]
    selected, delay = handler(servers, {})
    assert selected is servers[0]
    # ceiling for reconnects=4 is min(30, 1 * 2**4) = 16.
    assert 0.0 <= delay <= 16.0


def test_handler_delay_grows_with_reconnect_count() -> None:
    """higher ``Server.reconnects`` widens the jitter window (backoff grows per attempt)."""
    handler = _handler(base=1.0, cap=30.0)
    # sample the max draw at each reconnect level; the achievable ceiling must grow.
    low_ceiling = max(_delay(handler, 1) for _ in range(500))
    high_ceiling = max(_delay(handler, 5) for _ in range(500))
    assert high_ceiling > low_ceiling


def test_handler_empty_snapshot_returns_none_without_raising() -> None:
    """a defensive empty snapshot yields ``(None, delay)`` rather than an IndexError."""
    handler = _handler(base=1.0, cap=30.0)
    selected, delay = handler([], {})
    assert selected is None
    assert 0.0 <= delay <= 1.0


def test_handler_delays_are_not_constant() -> None:
    """repeated handler invocations for the SAME server produce varied delays (a jitter fn, not a constant)."""
    handler = _handler(base=1.0, cap=30.0)
    draws = {_delay(handler, 5) for _ in range(200)}
    assert len(draws) > 100


# ---------------------------------------------------------------------------
# connect-options wiring (the enforcement suggestion: a jitter fn, not a constant)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_options_carry_jittered_reconnect_handler() -> None:
    """the options reaching nats-py carry a callable ``reconnect_to_server_handler`` (jittered, not fixed)."""

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    await NatsClient.connect(
        establish_connection=_fake_establish,
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="agent-x",
        verify_jetstream=False,
    )
    options = captured["options"]
    handler = options["reconnect_to_server_handler"]
    assert callable(handler)
    # the reconnect delay SOURCE is a jitter function of the attempt, never a fixed constant: sampling
    # the same server many times yields spread, all within the default cap.
    delays = {handler([_server(6)], {})[1] for _ in range(200)}
    assert len(delays) > 100
    assert all(0.0 <= d <= DEFAULT_RECONNECT_BACKOFF_CAP_SECONDS for d in delays)


@pytest.mark.asyncio
async def test_connect_accepts_custom_backoff_params() -> None:
    """caller-supplied backoff base/cap (SDK pass-through) reach the handler's jitter window."""

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    await NatsClient.connect(
        establish_connection=_fake_establish,
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="agent-x",
        verify_jetstream=False,
        reconnect_backoff_base=0.5,
        reconnect_backoff_cap=4.0,
    )
    handler = captured["options"]["reconnect_to_server_handler"]
    # attempt 0 ceiling is base (0.5); the custom cap (4.0) clamps large attempts.
    assert all(0.0 <= handler([_server(0)], {})[1] <= 0.5 for _ in range(50))
    assert all(0.0 <= handler([_server(20)], {})[1] <= 4.0 for _ in range(50))


def test_default_backoff_params_are_deliberate_values() -> None:
    """the defaults are explicit, documented tunables (base 1s, cap 30s)."""
    assert DEFAULT_RECONNECT_BACKOFF_BASE_SECONDS == 1.0
    assert DEFAULT_RECONNECT_BACKOFF_CAP_SECONDS == 30.0
