"""Sending to many recipients, where one of them failing is ordinary.

The single-message transport is sized for transactional mail: one exception aborts one
send, and that is right when the send IS the operation. A reminder blast inverts it --
one refused address out of two hundred thousand must cost that address and nothing
else, and the caller needs to know WHICH address so it can record the outcome.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from threetears.channels.mail.batch import (
    BatchSendResult,
    RecipientFailure,
    send_batch,
)
from threetears.channels.mail.message import EmailMessage, EmailSendError


def _messages(count: int) -> list[EmailMessage]:
    return [EmailMessage(to=f"r{index}@acme.example", subject="s", body_text="b") for index in range(count)]


# parity-with: threetears.channels.mail.message.EmailTransport
class _CountingTransport:
    """Accepts every message, and reports the highest concurrency it ever saw."""

    def __init__(self, *, refuse: frozenset[str] = frozenset(), delay: float = 0.0) -> None:
        self.accepted: list[str] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self._refuse = refuse
        self._delay = delay

    async def send(self, message: EmailMessage) -> None:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self._delay)
            if message.to in self._refuse:
                raise EmailSendError(f"relay refused {message.to}")
            self.accepted.append(message.to)
        finally:
            self.in_flight -= 1


# parity-with: threetears.channels.mail.batch.SendPacer
class _CountingPacer:
    def __init__(self, *, raise_for: frozenset[int] = frozenset()) -> None:
        self.acquisitions = 0
        self._raise_for = raise_for

    async def acquire(self) -> None:
        self.acquisitions += 1
        if self.acquisitions in self._raise_for:
            raise RuntimeError("no tokens available within the wait budget")


class TestPerRecipientIsolation:
    async def test_one_refused_recipient_does_not_abort_the_batch(self) -> None:
        transport = _CountingTransport(refuse=frozenset({"r2@acme.example"}))

        result = await send_batch(transport, _messages(5))

        assert result.sent == 4
        assert result.failed == 1
        assert sorted(transport.accepted) == [
            "r0@acme.example",
            "r1@acme.example",
            "r3@acme.example",
            "r4@acme.example",
        ]

    async def test_the_failure_names_the_recipient_and_the_reason(self) -> None:
        """The caller has to be able to mark that address, and a count alone cannot."""
        transport = _CountingTransport(refuse=frozenset({"r1@acme.example"}))

        result = await send_batch(transport, _messages(3))

        assert result.failures == (
            RecipientFailure(
                recipient="r1@acme.example",
                error_type="EmailSendError",
                reason="relay refused r1@acme.example",
            ),
        )

    async def test_an_unforeseen_transport_error_is_isolated_too(self) -> None:
        """A transport is free to raise anything. Isolating only `EmailSendError` would
        abort the run on the one failure nobody predicted."""

        # parity-with: threetears.channels.mail.message.EmailTransport
        class _Surprising:
            async def send(self, message: EmailMessage) -> None:
                if message.to == "r0@acme.example":
                    raise RuntimeError("something nobody predicted")

        result = await send_batch(_Surprising(), _messages(2))

        assert result.sent == 1
        assert result.failures[0].error_type == "RuntimeError"

    async def test_a_cancellation_is_not_swallowed_as_a_recipient_failure(self) -> None:
        """Cancelling the batch must stop it, not record two hundred thousand
        `CancelledError` failures and report a completed run."""

        # parity-with: threetears.channels.mail.message.EmailTransport
        class _Cancelling:
            async def send(self, message: EmailMessage) -> None:
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await send_batch(_Cancelling(), _messages(3))


class TestTheCallbackFiresAsItGoes:
    async def test_each_failure_is_reported_before_the_batch_ends(self) -> None:
        """A run over a large sample must not hold every outcome until the end: the
        caller records a disposition per recipient as it happens."""
        seen: list[str] = []

        async def _record(failure: RecipientFailure) -> None:
            seen.append(failure.recipient)

        transport = _CountingTransport(refuse=frozenset({"r0@acme.example", "r2@acme.example"}))

        await send_batch(transport, _messages(3), on_failure=_record)

        assert sorted(seen) == ["r0@acme.example", "r2@acme.example"]

    async def test_a_failing_callback_does_not_abort_the_batch(self) -> None:
        """The callback writes to a database. That database being briefly unavailable
        must not cost the remaining recipients their send."""

        async def _broken(failure: RecipientFailure) -> None:
            raise RuntimeError("disposition write failed")

        transport = _CountingTransport(refuse=frozenset({"r0@acme.example"}))

        result = await send_batch(transport, _messages(3), on_failure=_broken)

        assert result.sent == 2


class TestBoundedMemory:
    async def test_the_recorded_failures_are_capped_but_the_count_is_not(self) -> None:
        """A blast where the relay is down fails for every recipient. Retaining two
        hundred thousand failure records to report that is its own outage."""
        transport = _CountingTransport(refuse=frozenset(f"r{index}@acme.example" for index in range(10)))

        result = await send_batch(transport, _messages(10), max_recorded_failures=3)

        assert result.failed == 10
        assert len(result.failures) == 3

    async def test_the_result_reports_what_it_attempted(self) -> None:
        transport = _CountingTransport(refuse=frozenset({"r0@acme.example"}))

        result = await send_batch(transport, _messages(4))

        assert isinstance(result, BatchSendResult)
        assert result.attempted == 4


class TestThroughput:
    async def test_concurrency_is_bounded(self) -> None:
        """Unbounded, a two-hundred-thousand-recipient run opens two hundred thousand
        SMTP conversations at once and the relay closes the connection on all of them."""
        transport = _CountingTransport(delay=0.01)

        await send_batch(transport, _messages(12), concurrency=3)

        assert transport.peak_in_flight <= 3
        assert len(transport.accepted) == 12

    async def test_the_pacer_is_consulted_once_per_recipient(self) -> None:
        """Concurrency caps how many are in flight; it does not cap the RATE. A relay
        with a per-second quota needs the second one."""
        pacer = _CountingPacer()
        transport = _CountingTransport()

        await send_batch(transport, _messages(5), pacer=pacer)

        assert pacer.acquisitions == 5

    async def test_a_recipient_that_cannot_get_a_token_fails_alone(self) -> None:
        """Being rate-limited out is that recipient's outcome, not the batch's."""
        pacer = _CountingPacer(raise_for=frozenset({2}))
        transport = _CountingTransport()

        result = await send_batch(transport, _messages(3), concurrency=1, pacer=pacer)

        assert result.sent == 2
        assert result.failed == 1

    async def test_a_concurrency_of_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            await send_batch(_CountingTransport(), _messages(1), concurrency=0)


class TestItDoesNotMaterialiseTheWholeSample:
    async def test_it_accepts_a_generator_and_pulls_lazily(self) -> None:
        """Two hundred thousand `EmailMessage` objects held at once is avoidable, and
        the caller almost always has them behind a cursor already."""
        pulled = 0

        def _lazy() -> Iterator[EmailMessage]:
            nonlocal pulled
            for message in _messages(6):
                pulled += 1
                yield message

        transport = _CountingTransport()

        result = await send_batch(transport, _lazy(), concurrency=2)

        assert result.sent == 6
        assert pulled == 6
