"""Container starts are staggered across xdist workers.

Several xdist workers each starting their first container in the same instant is a burst the
Docker daemon does not always survive -- on ZFS-backed storage it leaves half-created containers
(``zfs destroy ... dataset does not exist``) and the session fixture errors. Consumers were each
carrying the same ``pytest_fixture_setup`` hook to spread the starts out; the shared fixtures now do
it themselves, and so can any fixture that starts its own container.
"""

from __future__ import annotations

import pytest

from threetears.core.testing import containers
from threetears.core.testing.containers import stagger_container_start


class _Clock:
    def __init__(self) -> None:
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


@pytest.fixture(autouse=True)
def _fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test is a fresh worker process as far as the once-per-process flag is concerned."""
    monkeypatch.setattr(containers, "_staggered", False)
    monkeypatch.delenv("THREETEARS_TEST_CONTAINER_STAGGER_SECONDS", raising=False)


def test_worker_n_waits_n_times_the_stagger_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    clock = _Clock()
    stagger_container_start(sleep=clock)
    stagger_container_start(sleep=clock)
    assert clock.slept == [6.0], "gw3 waits 3 x 2s before its FIRST container, and never again"


def test_the_first_worker_never_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    clock = _Clock()
    stagger_container_start(sleep=clock)
    assert clock.slept == []


def test_without_xdist_nothing_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    clock = _Clock()
    stagger_container_start(sleep=clock)
    assert clock.slept == []


def test_the_stagger_is_configurable_and_zero_disables_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw2")
    monkeypatch.setenv("THREETEARS_TEST_CONTAINER_STAGGER_SECONDS", "0.5")
    clock = _Clock()
    stagger_container_start(sleep=clock)
    assert clock.slept == [1.0]

    monkeypatch.setattr(containers, "_staggered", False)
    monkeypatch.setenv("THREETEARS_TEST_CONTAINER_STAGGER_SECONDS", "0")
    clock = _Clock()
    stagger_container_start(sleep=clock)
    assert clock.slept == []


@pytest.mark.parametrize("value", ["-1", "soon"])
def test_an_invalid_stagger_fails_loudly(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    monkeypatch.setenv("THREETEARS_TEST_CONTAINER_STAGGER_SECONDS", value)
    with pytest.raises(ValueError, match="THREETEARS_TEST_CONTAINER_STAGGER_SECONDS"):
        stagger_container_start(sleep=_Clock())


def test_an_unrecognised_worker_name_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "master")
    clock = _Clock()
    stagger_container_start(sleep=clock)
    assert clock.slept == []


def test_the_shared_fixtures_stagger_before_starting_a_container() -> None:
    import inspect

    from threetears.core.testing import fixtures

    for name in ("db_container", "nats_container", "s3_container", "searxng_container"):
        source = inspect.getsource(getattr(fixtures, name).__wrapped__)
        assert "stagger_container_start()" in source, f"{name} starts a container without staggering"
