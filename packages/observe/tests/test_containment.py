"""unit tests for ``threetears.observe.contain`` and ``threetears.observe.contained``.

a contained failure is recorded the same way wherever it is absorbed: a WARNING with the
traceback, and one count on ``contained_errors_total`` against the function that caught it.
``contained`` absorbs :class:`Exception` and nothing wider, so a cancellation still reaches
the task that was asked to stop.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from prometheus_client import REGISTRY

from threetears.observe import contain, contained, get_logger

_log = get_logger(__name__)
_MODULE = __name__


def _count(site: str, exc_type: str) -> float:
    value = REGISTRY.get_sample_value("contained_errors_total", {"site": site, "exc_type": exc_type})
    return value or 0.0


def _catches_and_contains() -> None:
    try:
        raise ValueError("the upstream said no")
    except ValueError as exc:
        contain(exc, log=_log, what="upstream refused", extra={"upstream": "pricing"})


def _raises_from_a_callee() -> None:
    raise LookupError("deep down")


def _catches_a_callees_failure() -> None:
    try:
        _raises_from_a_callee()
    except LookupError as exc:
        contain(exc, log=_log, what="lookup failed")


def _contains_a_block() -> None:
    with contained("scratch file not removed", log=_log):
        raise OSError("read-only file system")


class TestContain:
    def test_logs_a_warning_that_carries_the_traceback(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _catches_and_contains()
        (record,) = [r for r in caplog.records if r.getMessage() == "upstream refused"]
        assert record.levelno == logging.WARNING
        assert record.exc_info is not None and isinstance(record.exc_info[1], ValueError)

    def test_keeps_the_callers_fields_and_names_the_site(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _catches_and_contains()
        (record,) = [r for r in caplog.records if r.getMessage() == "upstream refused"]
        assert getattr(record, "extra_data") == {
            "upstream": "pricing",
            "contained_site": f"{_MODULE}:_catches_and_contains",
            "exc_type": "ValueError",
        }

    def test_counts_against_the_function_that_caught_it(self) -> None:
        site = f"{_MODULE}:_catches_and_contains"
        before = _count(site, "ValueError")
        _catches_and_contains()
        assert _count(site, "ValueError") == before + 1

    def test_the_site_is_the_catcher_not_the_raiser(self) -> None:
        catcher = f"{_MODULE}:_catches_a_callees_failure"
        raiser = f"{_MODULE}:_raises_from_a_callee"
        before = _count(catcher, "LookupError")
        _catches_a_callees_failure()
        assert _count(catcher, "LookupError") == before + 1
        assert _count(raiser, "LookupError") == 0.0

    def test_an_exception_never_raised_is_named_after_the_caller(self) -> None:
        site = f"{_MODULE}:TestContain.test_an_exception_never_raised_is_named_after_the_caller"
        before = _count(site, "RuntimeError")
        contain(RuntimeError("handed over, never raised"), log=_log, what="handed-over failure")
        assert _count(site, "RuntimeError") == before + 1

    def test_logs_at_error_when_asked(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            try:
                raise TimeoutError("gave up")
            except TimeoutError as exc:
                contain(exc, log=_log, what="retries exhausted", level="error")
        (record,) = [r for r in caplog.records if r.getMessage() == "retries exhausted"]
        assert record.levelno == logging.ERROR

    def test_refuses_a_level_it_does_not_name(self) -> None:
        with pytest.raises(ValueError, match="warning, error or critical"):
            contain(RuntimeError("quiet"), log=_log, what="whispered", level="debug")  # type: ignore[arg-type]


class TestContained:
    def test_absorbs_an_exception_and_counts_it_against_the_with_block(self) -> None:
        site = f"{_MODULE}:_contains_a_block"
        before = _count(site, "OSError")
        _contains_a_block()
        assert _count(site, "OSError") == before + 1

    def test_a_clean_block_records_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING), contained("doing nothing", log=_log):
            pass
        assert not [r for r in caplog.records if r.getMessage() == "doing nothing"]

    async def test_a_cancellation_is_not_absorbed(self) -> None:
        async def _stops_when_told() -> None:
            with contained("waiting forever", log=_log):
                await asyncio.Event().wait()

        task = asyncio.create_task(_stops_when_told())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    def test_a_keyboard_interrupt_is_not_absorbed(self) -> None:
        with pytest.raises(KeyboardInterrupt), contained("an interactive step", log=_log):
            raise KeyboardInterrupt
