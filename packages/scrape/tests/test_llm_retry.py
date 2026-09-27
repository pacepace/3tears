"""Unit tests for threetears.scrape.llm_retry -- the two faces of one bounded retry.

``bounded_retry_structured_call_or_raise`` answers total failure with a typed error that
carries the last cause, for a caller that must RECORD the failure.
``bounded_retry_structured_call`` answers it with ``None``, for a caller whose honest reading
of "no answer" is "nothing here". Both share one retry loop, so these tests pin that the two
agree on everything except how exhaustion is reported.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel
from threetears.models import LlmPurpose

from threetears.scrape.llm_retry import (
    StructuredCallExhaustedError,
    bounded_retry_structured_call,
    bounded_retry_structured_call_or_raise,
)


class _Answer(BaseModel):
    value: str


def _fake_structured_model(result=None, *, side_effect=None):
    ainvoke_mock = AsyncMock(return_value=result, side_effect=side_effect)
    structured = SimpleNamespace(ainvoke=ainvoke_mock)
    return SimpleNamespace(with_structured_output=lambda schema, **kwargs: structured), ainvoke_mock


_CALL = {
    "model_id": "m",
    "api_key": "k",
    "purpose": LlmPurpose.SUMMARIZATION,
    "temperature": 0.0,
    "timeout": 1.0,
    "backoff_seconds": 0.0,
    "log_label": "unit call",
}


def _errors(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "threetears.scrape.llm_retry" and r.levelno >= logging.ERROR]


class TestOrRaise:
    async def test_returns_the_first_good_answer(self):
        fake_model, ainvoke_mock = _fake_structured_model(side_effect=[RuntimeError("once"), _Answer(value="ok")])
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            answer = await bounded_retry_structured_call_or_raise("p", _Answer, attempts=3, **_CALL)
        assert answer == _Answer(value="ok")
        assert ainvoke_mock.await_count == 2

    async def test_exhaustion_raises_carrying_the_last_cause(self, caplog: pytest.LogCaptureFixture):
        last = ValueError("second")
        fake_model, _ = _fake_structured_model(side_effect=[RuntimeError("first"), last])
        with (
            caplog.at_level(logging.WARNING),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
            pytest.raises(StructuredCallExhaustedError) as exc_info,
        ):
            await bounded_retry_structured_call_or_raise("p", _Answer, attempts=2, **_CALL)
        assert exc_info.value.attempts == 2
        assert exc_info.value.last_error is last
        assert exc_info.value.__cause__ is last
        assert exc_info.value.log_label == "unit call"
        # Logged once, where it happened, with the last cause; a caller that records the
        # failure does not log it again.
        errors = _errors(caplog)
        assert len(errors) == 1
        assert "unit call" in errors[0].getMessage()
        assert "ValueError: second" in errors[0].getMessage()

    async def test_cancellation_is_not_an_attempt(self):
        fake_model, ainvoke_mock = _fake_structured_model(side_effect=asyncio.CancelledError())
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            pytest.raises(asyncio.CancelledError),
        ):
            await bounded_retry_structured_call_or_raise("p", _Answer, attempts=3, **_CALL)
        assert ainvoke_mock.await_count == 1

    async def test_zero_attempts_is_refused(self):
        with pytest.raises(ValueError, match="attempts"):
            await bounded_retry_structured_call_or_raise("p", _Answer, attempts=0, **_CALL)


class TestDegradeToNone:
    async def test_exhaustion_degrades_to_none_and_logs_one_error(self, caplog: pytest.LogCaptureFixture):
        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("down"))
        with (
            caplog.at_level(logging.WARNING),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            answer = await bounded_retry_structured_call("p", _Answer, attempts=2, degraded_to="nothing", **_CALL)
        assert answer is None
        errors = _errors(caplog)
        assert len(errors) == 1
        assert "down" in errors[0].getMessage()
