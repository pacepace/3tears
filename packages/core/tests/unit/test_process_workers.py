"""CPU work in worker processes, plain data in and out (:mod:`threetears.core.process_workers`)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import pytest

from threetears.core.process_workers import ProcessWorkers, require_plain


def _pid_and(value: bytes) -> tuple[int, bytes]:
    """
    :param value: anything plain
    :ptype value: bytes
    :return: the process that ran it, and the value reversed
    :rtype: tuple[int, bytes]
    """
    return os.getpid(), value[::-1]


def _an_object() -> object:
    """
    :return: an object, which no worker may hand back
    :rtype: object
    """
    return object()


def _dies() -> None:
    """end the worker's process, as a worker killed mid-job would.

    :return: never
    :rtype: None
    """
    os._exit(1)


def _raises(message: str) -> None:
    """
    :param message: the error's text
    :ptype message: str
    :return: never
    :rtype: None
    :raises ValueError: always
    """
    raise ValueError(message)


@pytest.fixture
async def workers() -> AsyncIterator[ProcessWorkers]:
    """
    :return: two workers, shut down after the test
    :rtype: AsyncIterator[ProcessWorkers]
    """
    made = ProcessWorkers(workers=2, name="test")
    try:
        yield made
    finally:
        await made.close()


class TestPlainData:
    @pytest.mark.parametrize("value", [b"x", "x", 1, 1.5, True, None, (b"a", ["b", frozenset({"c"})]), bytes])
    def test_plain_data_passes(self, value: Any) -> None:
        require_plain(value, what="a value")

    @pytest.mark.parametrize("value", [object(), {"a": 1}, (b"a", [object()]), {1, 2}])
    def test_an_object_is_refused(self, value: Any) -> None:
        with pytest.raises(TypeError, match="plain data"):
            require_plain(value, what="a value")

    async def test_an_argument_that_is_an_object_is_refused_before_any_worker_starts(
        self, workers: ProcessWorkers
    ) -> None:
        with pytest.raises(TypeError, match="arguments of _pid_and"):
            await workers.run(_pid_and, object())

    async def test_an_answer_that_is_an_object_is_refused(self, workers: ProcessWorkers) -> None:
        with pytest.raises(TypeError, match="answer of _an_object"):
            await workers.run(_an_object)


class TestWhereItRuns:
    async def test_a_job_runs_in_another_process(self, workers: ProcessWorkers) -> None:
        pid, value = await workers.run(_pid_and, b"abc")

        assert pid != os.getpid() and value == b"cba"

    async def test_an_inline_job_runs_here_by_the_same_function(self, workers: ProcessWorkers) -> None:
        assert await workers.run(_pid_and, b"abc", inline=True) == (os.getpid(), b"cba")

    async def test_each_run_answers_in_its_order(self, workers: ProcessWorkers) -> None:
        answers = await workers.run_each(_pid_and, [(bytes([n]),) for n in range(8)])

        assert [value for _, value in answers] == [bytes([n]) for n in range(8)]
        assert len({pid for pid, _ in answers}) <= workers.workers

    async def test_a_jobs_error_reaches_the_caller(self, workers: ProcessWorkers) -> None:
        with pytest.raises(ValueError, match="no good"):
            await workers.run(_raises, "no good")


class TestTheWorkersLifetime:
    async def test_a_dead_worker_fails_its_job_and_the_next_job_gets_a_fresh_pool(
        self, workers: ProcessWorkers
    ) -> None:
        with pytest.raises(BrokenProcessPool):
            await workers.run(_dies)

        pid, _ = await workers.run(_pid_and, b"a")
        assert pid != os.getpid()

    async def test_closed_workers_are_started_again_by_the_next_job(self, workers: ProcessWorkers) -> None:
        first, _ = await workers.run(_pid_and, b"a")
        await workers.close()

        second, _ = await workers.run(_pid_and, b"a")

        assert second != first

    def test_a_pool_needs_a_worker(self) -> None:
        with pytest.raises(ValueError, match="needs a worker"):
            ProcessWorkers(workers=0, name="none")
