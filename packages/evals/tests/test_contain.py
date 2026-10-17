"""``threetears.evals.schema.logs.contain`` keeps :func:`threetears.observe.contain`'s contract, less the metric.

The message verbatim at WARNING or above with the traceback, ``contained_site`` and ``exc_type`` added to
``extra_data``, and ``contained`` absorbing :class:`Exception` and nothing wider.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from threetears.evals.schema.logs import contain, contained, get_logger

_log = get_logger(__name__)


def _catches(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
    with caplog.at_level(logging.WARNING, logger="threetears"):
        try:
            raise ValueError("the provider said no")
        except ValueError as exc:
            contain(exc, log=_log, what="provider refused", extra={"provider": "p"})
    (record,) = [r for r in caplog.records if r.getMessage() == "provider refused"]
    return record


def test_contain_logs_the_message_with_its_traceback_and_names_the_catcher(caplog: pytest.LogCaptureFixture) -> None:
    record = _catches(caplog)
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None and isinstance(record.exc_info[1], ValueError)
    assert getattr(record, "extra_data") == {
        "provider": "p",
        "contained_site": f"{__name__}:_catches",
        "exc_type": "ValueError",
    }


def test_contain_refuses_a_level_it_does_not_name() -> None:
    with pytest.raises(ValueError, match="warning, error or critical"):
        contain(RuntimeError("quiet"), log=_log, what="whispered", level="info")  # type: ignore[arg-type]


def test_contained_absorbs_an_exception_and_names_the_with_block(caplog: pytest.LogCaptureFixture) -> None:
    def _tidies() -> None:
        with contained("scratch not removed", log=_log):
            raise OSError("read-only")

    with caplog.at_level(logging.WARNING, logger="threetears"):
        _tidies()
    (record,) = [r for r in caplog.records if r.getMessage() == "scratch not removed"]
    assert getattr(record, "extra_data")["contained_site"].endswith("._tidies")


async def test_contained_lets_a_cancellation_through() -> None:
    async def _waits() -> None:
        with contained("waiting", log=_log):
            await asyncio.Event().wait()

    task = asyncio.create_task(_waits())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
