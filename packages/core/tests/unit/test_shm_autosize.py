"""Shared-memory autosize, the 3tears mechanism: measure, export, decide "need N bytes", hand it to an actuator.

The contracts. Off by default, it does nothing at all. Enabled with no actuator, it refuses at startup. Enabled, it
samples the pod's shared-memory volume (used and size), the container's memory (in use and limit) and its stores'
in-use bytes against their budgets on a thread, never the event loop; exports each as a gauge of the shared-memory
family; and when a use crosses the settable threshold it asks for the volume doubled per step until the measured
peak sits under the threshold, as one platform record per need, handed to the actuator once however often the need
is measured.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

from threetears.core.shm_autosize import (
    DEFAULT_LIMIT_BYTES,
    DEFAULT_THRESHOLD,
    AutosizeSettings,
    Sample,
    ShmAutosize,
    SizeRequest,
    measure,
)

GIB = 1 << 30


class _Actuator:
    """a fake actuator: records every request it is handed, and may fail the first."""

    def __init__(self, *, fail_first: bool = False) -> None:
        self.requests: list[SizeRequest] = []
        self._fail = fail_first

    def request(self, need: SizeRequest) -> None:
        if self._fail:
            self._fail = False
            raise RuntimeError("the deploy path is down")
        self.requests.append(need)


def _sample(
    used: int, size: int = GIB, *, memory: int | None = None, limit: int | None = None, store: tuple[int, int] = (0, 0)
) -> Sample:
    return Sample(
        volume_used_bytes=used,
        volume_size_bytes=size,
        memory_used_bytes=memory if memory is not None else used,
        memory_limit_bytes=limit,
        store_in_use_bytes=store[0],
        store_budget_bytes=store[1],
    )


def _on(product: str, **changes: object) -> AutosizeSettings:
    return replace(AutosizeSettings(enabled=True, product=product), **changes)  # type: ignore[arg-type]


def _gauge(name: str, product: str) -> float | None:
    return REGISTRY.get_sample_value(name, {"tier": "shared", "name": product})


@pytest.fixture
def product() -> str:
    return f"p-{uuid.uuid4().hex[:8]}"


class TestOffByDefault:
    def test_the_defaults_are_off_75_percent_and_16_gib(self) -> None:
        settings = AutosizeSettings()
        assert settings.enabled is False
        assert settings.threshold == DEFAULT_THRESHOLD == 0.75
        assert settings.limit_bytes == DEFAULT_LIMIT_BYTES == 16 * GIB

    def test_off_it_measures_nothing_asks_nothing_and_starts_no_thread(self, product: str) -> None:
        measured: list[int] = []

        def counted() -> Sample:
            measured.append(1)
            return _sample(GIB)

        actuator = _Actuator()
        autosize = ShmAutosize(AutosizeSettings(product=product), actuator, measure=counted)
        before = threading.active_count()
        autosize.start()
        assert threading.active_count() == before
        assert autosize.sample_once() is None
        autosize.stop()
        assert measured == [] and actuator.requests == []
        assert _gauge("threetears_cache_volume_used_bytes", product) is None


class TestEnablingNeedsAnActuator:
    def test_enabled_with_no_actuator_refuses_at_startup(self, product: str) -> None:
        with pytest.raises(ValueError, match="actuator"):
            ShmAutosize(_on(product), None)

    def test_enabled_with_no_product_refuses(self) -> None:
        with pytest.raises(ValueError, match="product"):
            ShmAutosize(AutosizeSettings(enabled=True), _Actuator())

    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.5, 1.5])
    def test_a_threshold_outside_zero_to_one_refuses(self, product: str, threshold: float) -> None:
        with pytest.raises(ValueError, match="threshold"):
            ShmAutosize(_on(product, threshold=threshold), _Actuator())


def decide(sample: Sample, peak: int, settings: AutosizeSettings) -> SizeRequest | None:
    """the need autosize decides from ``sample`` after it has measured ``peak``: through the front door, a monitor fed
    the peak first and the sample after.

    :return: the need, or None
    :rtype: SizeRequest | None
    """
    samples = iter([replace(sample, volume_used_bytes=peak), sample])
    autosize = ShmAutosize(settings, _Actuator(), measure=lambda: next(samples))
    autosize.sample_once()
    return autosize.sample_once()


class TestTheDecision:
    def test_under_the_threshold_asks_nothing(self) -> None:
        assert decide(_sample(int(0.74 * GIB)), int(0.74 * GIB), _on("p")) is None

    def test_at_the_threshold_asks_the_volume_doubled(self) -> None:
        need = decide(_sample(int(0.75 * GIB)), int(0.75 * GIB), _on("p"))
        assert need is not None
        assert (need.size_bytes, need.volume_bytes, need.reason) == (2 * GIB, GIB, "volume")
        assert need.request_id == f"p:{2 * GIB}"

    def test_it_doubles_per_step_until_the_peak_is_under_the_threshold(self) -> None:
        need = decide(_sample(int(0.9 * GIB)), int(3.5 * GIB), _on("p"))
        assert need is not None and need.size_bytes == 8 * GIB, "2 GiB and 4 GiB both leave 3.5 GiB past 75%"
        assert need.peak_bytes == int(3.5 * GIB)

    def test_the_threshold_is_settable(self) -> None:
        assert decide(_sample(int(0.6 * GIB)), int(0.6 * GIB), _on("p")) is None
        assert decide(_sample(int(0.6 * GIB)), int(0.6 * GIB), _on("p", threshold=0.5)) is not None

    def test_memory_near_its_limit_asks_and_raises_the_limit_by_the_volumes_growth(self) -> None:
        need = decide(_sample(int(0.1 * GIB), memory=7 * GIB, limit=8 * GIB), int(0.1 * GIB), _on("p"))
        assert need is not None and need.reason == "memory"
        assert (need.size_bytes, need.memory_limit_bytes) == (2 * GIB, 9 * GIB)

    def test_a_stores_in_use_set_near_its_budget_asks(self) -> None:
        near = decide(_sample(int(0.5 * GIB), store=(int(0.8 * GIB), GIB)), int(0.5 * GIB), _on("p"))
        assert near is not None and near.reason == "store"
        assert decide(_sample(int(0.5 * GIB), store=(int(0.5 * GIB), GIB)), int(0.5 * GIB), _on("p")) is None

    def test_past_the_product_limit_it_still_asks_and_says_so(self) -> None:
        need = decide(_sample(int(0.9 * 16 * GIB), 16 * GIB), int(0.9 * 16 * GIB), _on("p"))
        assert need is not None and need.size_bytes == 32 * GIB
        assert need.past_limit is True and need.limit_bytes == 16 * GIB


class TestTheMonitor:
    def test_one_need_is_handed_once_however_often_it_is_measured(self, product: str) -> None:
        actuator = _Actuator()
        autosize = ShmAutosize(_on(product), actuator, measure=lambda: _sample(int(0.8 * GIB)))
        for _ in range(3):
            autosize.sample_once()
        assert [each.request_id for each in actuator.requests] == [f"{product}:{2 * GIB}"]

    def test_a_failed_hand_over_is_handed_again_at_the_next_sample(self, product: str) -> None:
        actuator = _Actuator(fail_first=True)
        autosize = ShmAutosize(_on(product), actuator, measure=lambda: _sample(int(0.8 * GIB)))
        autosize.sample_once()
        autosize.sample_once()
        assert len(actuator.requests) == 1

    def test_the_peak_is_what_it_measured_most(self, product: str) -> None:
        uses = iter([int(0.8 * GIB), int(0.2 * GIB)])
        actuator = _Actuator()
        autosize = ShmAutosize(_on(product), actuator, measure=lambda: _sample(next(uses)))
        autosize.sample_once()
        need = autosize.sample_once()
        assert need is not None and need.peak_bytes == int(0.8 * GIB), "the peak, not the use now, sizes the need"
        assert [each.request_id for each in actuator.requests] == [need.request_id]

    def test_its_gauges_are_exported(self, product: str) -> None:
        autosize = ShmAutosize(
            _on(product), _Actuator(), measure=lambda: _sample(int(0.8 * GIB), memory=3 * GIB, limit=8 * GIB)
        )
        autosize.sample_once()
        assert _gauge("threetears_cache_volume_used_bytes", product) == int(0.8 * GIB)
        assert _gauge("threetears_cache_volume_size_bytes", product) == GIB
        assert _gauge("threetears_cache_rss_bytes", product) == 3 * GIB
        assert _gauge("threetears_cache_memory_limit_bytes", product) == 8 * GIB
        assert _gauge("threetears_cache_requested_bytes", product) == 2 * GIB
        assert _gauge("threetears_cache_autosize_limit_bytes", product) == 16 * GIB

    def test_it_samples_on_a_thread_of_its_own(self, product: str) -> None:
        seen: list[threading.Thread] = []
        sampled = threading.Event()

        def measured() -> Sample:
            seen.append(threading.current_thread())
            sampled.set()
            return _sample(0)

        autosize = ShmAutosize(_on(product, interval_seconds=0.01), _Actuator(), measure=measured)
        autosize.start()
        try:
            assert sampled.wait(10)
        finally:
            autosize.stop()
        assert seen and all(each is not threading.main_thread() for each in seen)


class TestMeasuringThePod:
    def test_it_reads_the_volume_and_a_cgroup_v2_memory(self, tmp_path: Path) -> None:
        cgroup = tmp_path / "cgroup"
        cgroup.mkdir()
        (cgroup / "memory.current").write_text("3221225472\n")
        (cgroup / "memory.max").write_text("8589934592\n")
        sample = measure(tmp_path, cgroup_root=cgroup)
        assert sample.volume_size_bytes > 0 and 0 <= sample.volume_used_bytes <= sample.volume_size_bytes
        assert (sample.memory_used_bytes, sample.memory_limit_bytes) == (3 * GIB, 8 * GIB)

    def test_no_memory_limit_reads_as_none(self, tmp_path: Path) -> None:
        cgroup = tmp_path / "cgroup"
        cgroup.mkdir()
        (cgroup / "memory.current").write_text("1024\n")
        (cgroup / "memory.max").write_text("max\n")
        assert measure(tmp_path, cgroup_root=cgroup).memory_limit_bytes is None

    def test_with_no_cgroup_it_reads_the_process(self, tmp_path: Path) -> None:
        sample = measure(tmp_path, cgroup_root=tmp_path / "none")
        assert sample.memory_used_bytes > 0 and sample.memory_limit_bytes is None
