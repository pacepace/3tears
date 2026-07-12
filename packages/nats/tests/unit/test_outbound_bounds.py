"""unit tests for the bounded NATS outbound/pending buffer (resilience-task-03).

these tests substitute fakes for the underlying nats-py client so the wrapper's
outbound-buffer bound + overflow-into-health-signal logic can be exercised without
a live broker. they assert THREE things the shard requires:

* the connect options carry an EXPLICIT, non-library-default pending/flusher bound;
* an :class:`nats.errors.OutboundBufferLimitError` raised synchronously at the
  publish boundary is caught, counted, and flips :attr:`NatsClient.is_healthy`
  (rather than propagating as an unhandled raise that crashes the caller);
* a successful publish or a reconnect resets the overflow counter (no flapping).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nats.aio.client import (
    DEFAULT_MAX_FLUSHER_QUEUE_SIZE as _NATS_DEFAULT_FLUSHER_QUEUE_SIZE,
    DEFAULT_PENDING_SIZE as _NATS_DEFAULT_PENDING_SIZE,
)
from nats.errors import OutboundBufferLimitError as _NatsOutboundBufferLimitError

from threetears.nats import NatsClient, PublishError, Subject
from threetears.nats.client import (
    DEFAULT_FLUSHER_QUEUE_SIZE,
    DEFAULT_PENDING_SIZE_BYTES,
    _is_outbound_overflow,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _OverflowRaw:
    """fake nats-py client whose publish always overflows the outbound buffer.

    models the wedge state: the client is connected-per-cache but nats-py raises
    ``OutboundBufferLimitError`` synchronously (the disconnected/reconnecting
    guard in ``aio/client.py`` firing on a full pending buffer).
    """

    def __init__(self) -> None:
        self.is_connected = True
        self.is_closed = False

    async def publish(self, *args: Any, **kwargs: Any) -> None:
        raise _NatsOutboundBufferLimitError


def _client(raw: Any) -> NatsClient:
    """build a bare wrapper over a fake raw client (no connect())."""
    return NatsClient(raw=raw, namespace="3tears", client_name="test")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# option-bound assertions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_options_carry_explicit_pending_and_flusher_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the connect options reaching nats-py carry the explicit pending/flusher bounds."""
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    await NatsClient.connect(
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="agent-x",
        verify_jetstream=False,
    )
    options = captured["options"]
    assert options["pending_size"] == DEFAULT_PENDING_SIZE_BYTES
    assert options["flusher_queue_size"] == DEFAULT_FLUSHER_QUEUE_SIZE


def test_pending_bound_is_explicit_not_library_default() -> None:
    """the bound is a DELIBERATE value, not silently inherited from nats-py's default."""
    assert DEFAULT_PENDING_SIZE_BYTES != _NATS_DEFAULT_PENDING_SIZE
    assert DEFAULT_FLUSHER_QUEUE_SIZE != _NATS_DEFAULT_FLUSHER_QUEUE_SIZE
    # sized ABOVE the library default so healthy bursty traffic never trips it.
    assert DEFAULT_PENDING_SIZE_BYTES > _NATS_DEFAULT_PENDING_SIZE


@pytest.mark.asyncio
async def test_connect_accepts_custom_buffer_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """caller-supplied buffer bounds (SDK pass-through) reach the connect options."""
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    await NatsClient.connect(
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="agent-x",
        verify_jetstream=False,
        pending_size=1_234_567,
        flusher_queue_size=99,
    )
    options = captured["options"]
    assert options["pending_size"] == 1_234_567
    assert options["flusher_queue_size"] == 99


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------


def test_is_outbound_overflow_detects_typed_and_text() -> None:
    """the classifier matches the typed error AND its -ERR text, robust across nats-py versions."""
    assert _is_outbound_overflow(_NatsOutboundBufferLimitError())
    assert _is_outbound_overflow(Exception("nats: outbound buffer limit exceeded"))
    assert not _is_outbound_overflow(OSError("connection reset by peer"))


# ---------------------------------------------------------------------------
# overflow -> health signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_overflow_increments_counter_and_flips_health() -> None:
    """sustained overflow at the publish boundary flips is_healthy; each raise maps to PublishError."""
    client = _client(_OverflowRaw())
    subject = Subject.raw("agents.x.heartbeat")

    assert client.is_healthy  # fresh wrapper is healthy

    # below the threshold (3): still healthy, but each overflow raises a TYPED PublishError
    # (not an unhandled OutboundBufferLimitError crashing the caller).
    for _ in range(2):
        with pytest.raises(PublishError):
            await client.publish_raw(subject=subject, payload=b"beat")
    assert client.is_healthy

    # crossing the threshold: unhealthy -- the connection is wedged with a full outbound buffer.
    with pytest.raises(PublishError):
        await client.publish_raw(subject=subject, payload=b"beat")
    assert not client.is_healthy


@pytest.mark.asyncio
async def test_successful_publish_resets_overflow_counter() -> None:
    """a successful publish clears the overflow streak, so it takes a FRESH run to trip health."""
    raw = MagicMock()
    raw.is_connected = True
    raw.is_closed = False
    overflow = {"on": True}

    async def _publish(*args: Any, **kwargs: Any) -> None:
        if overflow["on"]:
            raise _NatsOutboundBufferLimitError

    raw.publish = _publish
    client = _client(raw)
    subject = Subject.raw("agents.x.heartbeat")

    # two overflows -- below threshold, still healthy
    for _ in range(2):
        with pytest.raises(PublishError):
            await client.publish_raw(subject=subject, payload=b"beat")
    assert client.is_healthy

    # a successful publish resets the streak
    overflow["on"] = False
    await client.publish_raw(subject=subject, payload=b"beat")
    assert client.is_healthy

    # two more overflows -- WITHOUT the reset this would be 4 total (unhealthy); with it, only 2
    overflow["on"] = True
    for _ in range(2):
        with pytest.raises(PublishError):
            await client.publish_raw(subject=subject, payload=b"beat")
    assert client.is_healthy


@pytest.mark.asyncio
async def test_reconnect_resets_overflow_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a successful reconnect clears the wedged-buffer signal (mirrors the auth-violation reset)."""
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        raw = MagicMock()
        raw.is_connected = True
        raw.is_closed = False
        raw.publish = AsyncMock(side_effect=_NatsOutboundBufferLimitError)
        return raw

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    client = await NatsClient.connect(
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="agent-x",
        verify_jetstream=False,
    )
    reconnected_cb = captured["options"]["reconnected_cb"]
    subject = Subject.raw("agents.x.heartbeat")

    for _ in range(3):
        with pytest.raises(PublishError):
            await client.publish_raw(subject=subject, payload=b"beat")
    assert not client.is_healthy

    await reconnected_cb()
    assert client.is_healthy


@pytest.mark.asyncio
async def test_healthy_publishes_below_bound_do_not_flip_health() -> None:
    """a healthy burst (publishes that succeed, no overflow) never flips is_healthy -- no flapping."""
    raw = MagicMock()
    raw.is_connected = True
    raw.is_closed = False
    raw.publish = AsyncMock(return_value=None)
    client = _client(raw)
    subject = Subject.raw("agents.x.heartbeat")

    for _ in range(50):
        await client.publish_raw(subject=subject, payload=b"beat")
    assert client.is_healthy
