"""the per-registry wake emitter is built once when several threads ask for it first.

the cache is checked and then filled; without a lock every thread that checked before the
first one filled it built its own emitter, and the second one's registration raised
``Duplicated timeseries in CollectorRegistry``.
"""

from __future__ import annotations

import threading
import time

import pytest

from threetears.agent.wake.metrics import get_wake_emitter, reset_wake_emitter_for_testing


def test_emitters_first_asked_for_on_several_threads_register_once() -> None:
    prometheus_client = pytest.importorskip("prometheus_client")

    class _SlowRegistry(prometheus_client.CollectorRegistry):  # type: ignore[misc]
        """a registry that holds each registration open long enough for every thread to arrive."""

        def __init__(self) -> None:
            super().__init__()
            self.registrations = 0

        def register(self, collector: object) -> None:
            """count, wait, register.

            :param collector: the collector being registered
            :ptype collector: object
            :return: None
            :rtype: None
            """
            self.registrations += 1
            time.sleep(0.05)
            super().register(collector)

    registry = _SlowRegistry()
    threads_count = 8
    start = threading.Barrier(threads_count)
    emitters: list[object] = []
    errors: list[BaseException] = []
    record = threading.Lock()

    def first_use() -> None:
        start.wait()
        try:
            emitter = get_wake_emitter(registry)
        except BaseException as exc:  # noqa: BLE001 -- the assertion below reports every one
            with record:
                errors.append(exc)
            return
        with record:
            emitters.append(emitter)

    try:
        workers = [threading.Thread(target=first_use) for _ in range(threads_count)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)

        assert errors == [], f"concurrent first callers raised: {errors!r}"
        assert len({id(emitter) for emitter in emitters}) == 1
        once = registry.registrations
        get_wake_emitter(registry)
        assert registry.registrations == once, "a later call registered again"
    finally:
        reset_wake_emitter_for_testing()
