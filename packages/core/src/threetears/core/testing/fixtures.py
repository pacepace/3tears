"""canonical pytest fixtures for testcontainer-backed integration tests.

downstream test repos pull these via::

    pytest_plugins = ["threetears.core.testing.fixtures"]

every fixture here gates on :func:`check_docker_available` and calls
``pytest.skip`` when the daemon is unreachable. tests inheriting
these fixtures behave correctly on a fresh checkout without docker
installed: they skip cleanly instead of hard-failing on
``ConnectionRefusedError``.

session-scoped containers + their connection URIs are the public
surface here; per-test fixtures (HTTP clients, NATS connections,
db pools) belong in the consuming repo's conftest because their
shape varies (auth headers / connection-pool config / namespace
prefix / etc.).

DO NOT define your own postgres / nats container fixtures in
per-repo conftests. import these and wrap them with the
repo-specific shape if you need it. the docker-skip discipline +
asyncpg URL normalisation + jetstream toggle have been audited
exactly once and we want to keep it that way.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from threetears.core.testing.containers import check_docker_available

__all__ = [
    "db_container",
    "db_image",
    "nats_container",
    "nats_jetstream",
]


@pytest.fixture(scope="session")
def db_image(request: pytest.FixtureRequest) -> str:
    """docker image tag for the session-scoped postgres container.

    defaults to ``postgres:16``. tests that need pgvector parameterize
    via::

        @pytest.mark.parametrize(
            "db_image", ["pgvector/pgvector:pg16"], indirect=True,
        )

    :param request: pytest fixture request exposing indirect params
    :ptype request: pytest.FixtureRequest
    :return: container image reference
    :rtype: str
    """
    return getattr(request, "param", "postgres:16")


@pytest.fixture(scope="session")
def nats_jetstream() -> bool:
    """whether the session-scoped NATS container enables JetStream.

    defaults to True so consumers exercising KV buckets or streams
    work without extra wiring. tests that want a leaner NATS can
    override this fixture in their own conftest.

    :return: JetStream enable flag
    :rtype: bool
    """
    return True


@pytest.fixture(scope="session")
def db_container(db_image: str) -> Iterator[str]:
    """session-scoped postgres testcontainer.

    yields the asyncpg-compatible connection URL (``postgresql://``,
    NOT ``postgresql+psycopg2://`` -- the testcontainers default
    suffix gets normalised here). gated on docker availability:
    fresh checkouts without docker get a clean ``pytest.skip``
    instead of a connection-refused stack trace.

    parameterise the image via ``db_image`` indirect parametrize when
    tests need pgvector.

    :param db_image: docker image tag (defaults to ``postgres:16``)
    :ptype db_image: str
    :yield: asyncpg-compatible PostgreSQL connection URL
    :rtype: Iterator[str]
    """
    if not check_docker_available():
        pytest.skip("Docker not available")

    from testcontainers.postgres import PostgresContainer  # noqa: PLC0415

    with PostgresContainer(db_image) as container:
        raw_url = container.get_connection_url()
        normalised = raw_url
        if normalised.startswith("postgresql+psycopg2://"):
            normalised = normalised.replace(
                "postgresql+psycopg2://",
                "postgresql://",
                1,
            )
        yield normalised


@pytest.fixture(scope="session")
def nats_container(nats_jetstream: bool) -> Iterator[str]:
    """session-scoped NATS testcontainer.

    yields the ``nats://`` connection URI from the container.
    gated on docker availability with the same skip-on-no-docker
    discipline as :func:`db_container`.

    JetStream is enabled by default; override ``nats_jetstream``
    in your conftest to disable it.

    :param nats_jetstream: whether to enable JetStream
    :ptype nats_jetstream: bool
    :yield: NATS connection URI
    :rtype: Iterator[str]
    """
    if not check_docker_available():
        pytest.skip("Docker not available")

    from testcontainers.nats import NatsContainer  # noqa: PLC0415

    with NatsContainer(jetstream=nats_jetstream) as container:
        yield container.nats_uri()
