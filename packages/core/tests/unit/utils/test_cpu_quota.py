"""the CPUs a process may use: its cgroup's quota (v2, then v1), rounded up, else the machine's cores."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.core.utils.cpu_quota import bound_native_threads, default_executor_workers, effective_cpu_count

pyarrow = pytest.importorskip("pyarrow")


def _cores(n: int | None) -> object:
    return lambda: n


def _v2(tmp_path: Path, content: str) -> Path:
    (tmp_path / "cpu.max").write_text(content)
    return tmp_path


def _v1(tmp_path: Path, quota: str, period: str, directory: str = "cpu") -> Path:
    controller = tmp_path / directory
    controller.mkdir()
    (controller / "cpu.cfs_quota_us").write_text(quota)
    (controller / "cpu.cfs_period_us").write_text(period)
    return tmp_path


class TestCgroupV2:
    def test_a_whole_quota_is_its_cpus(self, tmp_path: Path) -> None:
        assert effective_cpu_count(_v2(tmp_path, "400000 100000\n"), cpu_count=_cores(32)) == 4

    def test_no_quota_is_the_machines_cores(self, tmp_path: Path) -> None:
        assert effective_cpu_count(_v2(tmp_path, "max 100000\n"), cpu_count=_cores(32)) == 32

    @pytest.mark.parametrize(("content", "cpus"), [("150000 100000", 2), ("50000 100000", 1), ("1000 100000", 1)])
    def test_a_fractional_quota_rounds_up_to_a_whole_cpu(self, tmp_path: Path, content: str, cpus: int) -> None:
        assert effective_cpu_count(_v2(tmp_path, content), cpu_count=_cores(32)) == cpus

    def test_a_quota_above_the_cores_is_the_cores(self, tmp_path: Path) -> None:
        assert effective_cpu_count(_v2(tmp_path, "6400000 100000"), cpu_count=_cores(8)) == 8

    @pytest.mark.parametrize("content", ["", "400000", "lots 100000", "0 100000", "400000 0"])
    def test_an_unreadable_quota_is_the_machines_cores(self, tmp_path: Path, content: str) -> None:
        assert effective_cpu_count(_v2(tmp_path, content), cpu_count=_cores(6)) == 6

    def test_v2_wins_over_v1(self, tmp_path: Path) -> None:
        _v1(tmp_path, "800000", "100000")
        assert effective_cpu_count(_v2(tmp_path, "200000 100000"), cpu_count=_cores(32)) == 2


class TestCgroupV1:
    def test_a_quota_is_its_cpus(self, tmp_path: Path) -> None:
        assert effective_cpu_count(_v1(tmp_path, "300000\n", "100000\n"), cpu_count=_cores(32)) == 3

    def test_a_fractional_quota_rounds_up(self, tmp_path: Path) -> None:
        assert effective_cpu_count(_v1(tmp_path, "250000", "100000"), cpu_count=_cores(32)) == 3

    def test_minus_one_is_no_quota(self, tmp_path: Path) -> None:
        assert effective_cpu_count(_v1(tmp_path, "-1", "100000"), cpu_count=_cores(12)) == 12

    def test_the_combined_controller_directory_is_read(self, tmp_path: Path) -> None:
        assert effective_cpu_count(_v1(tmp_path, "200000", "100000", "cpu,cpuacct"), cpu_count=_cores(32)) == 2


class TestNoCgroup:
    def test_no_cgroup_is_the_machines_cores(self, tmp_path: Path) -> None:
        assert effective_cpu_count(tmp_path, cpu_count=_cores(16)) == 16

    def test_unknown_cores_is_one(self, tmp_path: Path) -> None:
        assert effective_cpu_count(tmp_path, cpu_count=_cores(None)) == 1

    def test_the_default_root_answers_at_least_one(self) -> None:
        assert effective_cpu_count() >= 1


class TestPoolSizes:
    @pytest.mark.parametrize(("cpus", "workers"), [(1, 5), (4, 8), (28, 32), (64, 32)])
    def test_the_default_executor_follows_the_standard_rule_on_the_quota(self, cpus: int, workers: int) -> None:
        assert default_executor_workers(cpus) == workers

    @pytest.mark.parametrize(("cpus", "io"), [(1, 2), (2, 4), (4, 8), (32, 8)])
    def test_pyarrow_is_sized_to_the_quota(self, cpus: int, io: int) -> None:
        before = (pyarrow.cpu_count(), pyarrow.io_thread_count())
        try:
            assert bound_native_threads(cpus) is True
            assert (pyarrow.cpu_count(), pyarrow.io_thread_count()) == (cpus, io)
        finally:
            pyarrow.set_cpu_count(before[0])
            pyarrow.set_io_thread_count(before[1])
