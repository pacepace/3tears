"""Container starts are staggered across xdist workers.

Several xdist workers each starting their first container in the same instant is a burst the
Docker daemon does not always survive -- on ZFS-backed storage it leaves half-created containers
(``zfs destroy ... dataset does not exist``) and the session fixture errors. Consumers were each
carrying the same ``pytest_fixture_setup`` hook to spread the starts out; the shared fixtures now do
it themselves, and so can any fixture that starts its own container.
"""

from __future__ import annotations

import pytest

from threetears.core.testing.containers import ContainerStartStagger


class _Clock:
    def __init__(self) -> None:
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


@pytest.fixture(autouse=True)
def _default_stagger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from the default stagger, whatever the environment running it sets."""
    monkeypatch.delenv("THREETEARS_TEST_CONTAINER_STAGGER_SECONDS", raising=False)


@pytest.fixture
def stagger() -> ContainerStartStagger:
    """A fresh worker process's stagger: no first start taken yet."""
    return ContainerStartStagger()


def test_worker_n_waits_n_times_the_stagger_once(
    stagger: ContainerStartStagger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    clock = _Clock()
    stagger.wait_before_first_start(sleep=clock)
    stagger.wait_before_first_start(sleep=clock)
    assert clock.slept == [6.0], "gw3 waits 3 x 2s before its FIRST container, and never again"


def test_the_first_worker_never_waits(stagger: ContainerStartStagger, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    clock = _Clock()
    stagger.wait_before_first_start(sleep=clock)
    assert clock.slept == []


def test_without_xdist_nothing_waits(stagger: ContainerStartStagger, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    clock = _Clock()
    stagger.wait_before_first_start(sleep=clock)
    assert clock.slept == []


def test_the_stagger_is_configurable_and_zero_disables_it(
    stagger: ContainerStartStagger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw2")
    monkeypatch.setenv("THREETEARS_TEST_CONTAINER_STAGGER_SECONDS", "0.5")
    clock = _Clock()
    stagger.wait_before_first_start(sleep=clock)
    assert clock.slept == [1.0]

    monkeypatch.setenv("THREETEARS_TEST_CONTAINER_STAGGER_SECONDS", "0")
    clock = _Clock()
    ContainerStartStagger().wait_before_first_start(sleep=clock)
    assert clock.slept == []


@pytest.mark.parametrize("value", ["-1", "soon", "nan", "inf"])
def test_an_invalid_stagger_fails_loudly(
    stagger: ContainerStartStagger, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    monkeypatch.setenv("THREETEARS_TEST_CONTAINER_STAGGER_SECONDS", value)
    with pytest.raises(ValueError, match="THREETEARS_TEST_CONTAINER_STAGGER_SECONDS"):
        stagger.wait_before_first_start(sleep=_Clock())


def test_an_unrecognised_worker_name_does_not_wait(
    stagger: ContainerStartStagger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "master")
    clock = _Clock()
    stagger.wait_before_first_start(sleep=clock)
    assert clock.slept == []


#: what starts a container in the shared fixtures and the services they start through
_CONTAINER_STARTERS = frozenset({"PostgresContainer", "DockerContainer", "nats_server"})


def test_the_shared_fixtures_stagger_before_starting_a_container() -> None:
    """every function of the shared fixtures, and of the services module they start containers through
    (db, NATS and S3 since the services gained a process mode), that starts a container staggers first."""
    import ast
    import inspect

    from threetears.core.testing import fixtures, services

    for name in ("db_container", "nats_container", "s3_container"):
        tree = ast.parse(inspect.getsource(getattr(fixtures, name).__wrapped__))
        called = {sub.func.id for sub in ast.walk(tree) if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)}
        assert {"running_postgres", "running_nats", "running_s3"} & called, (
            f"{name} no longer starts its service through threetears.core.testing.services"
        )
    starters: list[str] = []
    for module in (fixtures, services):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name == "nats_server":
                continue
            calls = [sub for sub in ast.walk(node) if isinstance(sub, ast.Call)]
            names = {sub.func.id if isinstance(sub.func, ast.Name) else getattr(sub.func, "attr", "") for sub in calls}
            if names & _CONTAINER_STARTERS:
                starters.append(f"{module.__name__}.{node.name}")
                assert "stagger_container_start" in names, (
                    f"{module.__name__}.{node.name} starts a container without staggering"
                )
    # the control: Postgres, NATS and S3 in services, SearXNG in the fixtures, so an empty walk can't pass
    assert len(starters) >= 4, starters


def test_an_external_database_skips_the_container_and_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CI service container (the external URL) starts no container, so it must not wait for one."""
    from threetears.core.testing import fixtures, services

    def must_not_wait() -> None:
        raise AssertionError("an external URL waited on a container start that never happens")

    # the fixtures stagger SearXNG themselves; Postgres, NATS and S3 stagger in the services they start through
    monkeypatch.setattr(fixtures, "stagger_container_start", must_not_wait)
    monkeypatch.setattr(services, "stagger_container_start", must_not_wait)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw5")
    monkeypatch.setenv("THREETEARS_TEST_POSTGRES_URL", "postgresql://ci/db")
    gen = fixtures.db_container.__wrapped__("unused-image")
    assert next(gen) == "postgresql://ci/db"
    monkeypatch.setenv("THREETEARS_TEST_NATS_URL", "nats://ci:4222")
    assert next(fixtures.nats_container.__wrapped__(True, True)) == "nats://ci:4222"
    monkeypatch.setenv("THREETEARS_TEST_SEARXNG_URL", "http://searxng.ci/")
    assert next(fixtures.searxng_container.__wrapped__()) == "http://searxng.ci"


def test_the_stagger_setting_name_is_exported() -> None:
    from threetears.core.testing import CONTAINER_STAGGER_ENV

    assert CONTAINER_STAGGER_ENV == "THREETEARS_TEST_CONTAINER_STAGGER_SECONDS"
