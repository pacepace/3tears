"""``retry_with_backoff``: every line it writes names the failure, and exhaustion is logged as the caller needs.

A message-less exception once rendered every retry line and the exhaustion line with an empty reason
(``TracedHttpClient``'s private 5xx signal), so the log said a call failed and never how. The type name
stands in when the message is empty. Exhaustion is contained at ERROR with the traceback by default,
because a startup step that degrades on ``False`` leaves no other record; a caller that raises its own
error on ``False`` gets a WARNING without a traceback, and its error carries the rest.
"""

from __future__ import annotations

import logging

import pytest

from threetears.observe.resilience import retry_with_backoff

_LOGGER = "threetears.observe.resilience"


class _Silent(Exception):
    """an exception with no message, like a private retry signal."""


async def _always_fails() -> None:
    raise _Silent


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == _LOGGER]


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_raises", [False, True])
async def test_a_message_less_failure_is_named_by_its_type(
    caplog: pytest.LogCaptureFixture, caller_raises: bool
) -> None:
    caplog.set_level(logging.INFO)
    succeeded = await retry_with_backoff(
        _always_fails, "step", max_attempts=3, initial_backoff=0.0, max_backoff=0.0, caller_raises=caller_raises
    )
    assert succeeded is False
    lines = _records(caplog)
    assert len(lines) == 3
    for record in lines:
        assert record.getMessage().endswith(": _Silent")


@pytest.mark.asyncio
async def test_by_default_exhaustion_is_contained_at_error_with_the_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """the registry's startup steps discard the result: this line is their only record."""
    caplog.set_level(logging.INFO)
    await retry_with_backoff(_always_fails, "step", max_attempts=2, initial_backoff=0.0, max_backoff=0.0)
    exhausted = [record for record in _records(caplog) if "failed after" in record.getMessage()]
    assert len(exhausted) == 1
    assert exhausted[0].levelno == logging.ERROR
    assert exhausted[0].exc_info and isinstance(exhausted[0].exc_info[1], _Silent)


@pytest.mark.asyncio
async def test_a_caller_that_raises_gets_a_warning_without_the_traceback(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    await retry_with_backoff(
        _always_fails, "step", max_attempts=2, initial_backoff=0.0, max_backoff=0.0, caller_raises=True
    )
    exhausted = [record for record in _records(caplog) if "failed after" in record.getMessage()]
    assert len(exhausted) == 1
    assert exhausted[0].levelno == logging.WARNING
    assert not exhausted[0].exc_info
    assert exhausted[0].getMessage() == "step failed after 2 attempts: _Silent"


@pytest.mark.asyncio
async def test_the_startup_critical_retry_line_names_the_failure_too(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    calls = 0

    async def fails_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _Silent

    assert await retry_with_backoff(fails_once, "step", max_attempts=None, initial_backoff=0.0, max_backoff=0.0)
    retry = [record for record in _records(caplog) if "retrying" in record.getMessage()]
    assert len(retry) == 1
    assert retry[0].getMessage().endswith(": _Silent")
