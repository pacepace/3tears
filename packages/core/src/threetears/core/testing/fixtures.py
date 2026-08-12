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
    "searxng_container",
]


@pytest.fixture(scope="session")
def db_image(request: pytest.FixtureRequest) -> str:
    """docker image tag for the session-scoped postgres container.

    defaults to ``pgvector/pgvector:pg16`` -- a strict superset of the
    official ``postgres:16`` image (same Postgres 16 server, plus the
    ``vector`` extension binaries). the default carries pgvector on
    purpose: :func:`db_container` is *session*-scoped and therefore
    SHARED across every package in the workspace. whichever test first
    materialises the container pins its image for the whole session, so
    a plain ``postgres:16`` default lets a non-vector consumer (run
    earlier in collection order) poison the shared container for the
    vector-bearing suites that run later -- they would fail with
    ``extension "vector" is not available`` even though they declare a
    pgvector override of their own. defaulting to pgvector removes that
    cross-package ordering trap entirely; the extension is a no-op for
    suites that never touch a ``vector`` column.

    tests that genuinely need a leaner image can still parameterize via::

        @pytest.mark.parametrize(
            "db_image", ["postgres:16"], indirect=True,
        )

    :param request: pytest fixture request exposing indirect params
    :ptype request: pytest.FixtureRequest
    :return: container image reference
    :rtype: str
    """
    return getattr(request, "param", "pgvector/pgvector:pg16")


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

    the image defaults to ``pgvector/pgvector:pg16`` (see
    :func:`db_image`) so the shared session container is vector-capable
    for every consumer regardless of collection order. override
    ``db_image`` via indirect parametrize for a leaner image when a
    suite never touches a ``vector`` column.

    :param db_image: docker image tag (defaults to
        ``pgvector/pgvector:pg16``)
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


@pytest.fixture(scope="session")
def searxng_container() -> Iterator[str]:
    """session-scoped SearXNG testcontainer, yielding its base URL.

    Exists so a claim about SearXNG's scoring can be checked against SearXNG
    rather than against a fixture that restates it. The shipped
    ``search/tests`` suite drives the adapter over recorded payloads, which
    proves the adapter reads the field and cannot prove the field means what
    the adapter's docstring says it means -- only the real scorer can do
    that.

    **JSON is off by default in the image**, so the settings file is not
    optional configuration: without ``search.formats`` naming ``json`` the
    instance answers HTML to an API request and every caller sees a parse
    failure rather than a disabled format. The limiter is off for the same
    class of reason -- it is a bot defence, and a test client hammering one
    query looks exactly like the thing it defends against.

    **What this container cannot give you is engines.** SearXNG's own
    scoring is deterministic; the upstream engines it federates are not.
    They rate-limit, they vary between calls, and a run that saw two engines
    agree will not reliably see it again -- which is why a test using this
    fixture must assert an invariant that holds over whatever came back,
    never that a particular fusion occurred. See
    ``packages/search/tests/test_searxng_live_scoring.py``.

    :yield: the container's base URL, e.g. ``http://localhost:32768``
    :rtype: Iterator[str]
    """
    if not check_docker_available():
        pytest.skip("Docker not available")

    import tempfile  # noqa: PLC0415
    import time  # noqa: PLC0415
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from testcontainers.core.container import DockerContainer  # noqa: PLC0415

    settings = (
        "use_default_settings: true\n"
        "server:\n"
        '  secret_key: "testcontainer-only-never-a-deployment"\n'
        "  limiter: false\n"
        "  public_instance: false\n"
        "search:\n"
        "  formats:\n"
        "    - html\n"
        "    - json\n"
    )
    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / "settings.yml"
        path.write_text(settings, encoding="utf-8")
        container = (
            DockerContainer("searxng/searxng:latest")
            .with_exposed_ports(8080)
            .with_volume_mapping(str(path), "/etc/searxng/settings.yml", "ro")
        )
        with container:
            host = container.get_container_host_ip()
            port = container.get_exposed_port(8080)
            base_url = f"http://{host}:{port}"
            # Readiness is "answers a search", not "logged a line". The image
            # prints its listening banner before it will serve, so a log-match
            # wait hands the first request a reset connection -- observed, not
            # feared. Engines failing to register (a 403 from an upstream on
            # init) is normal and must NOT block readiness: an instance with
            # half its engines suspended still scores correctly, which is the
            # only thing a caller of this fixture is asking it.
            deadline = time.monotonic() + 180
            while True:
                try:
                    with urllib.request.urlopen(  # noqa: S310
                        f"{base_url}/search?q=ready&format=json", timeout=10
                    ) as probe:
                        if probe.status == 200:
                            break
                except urllib.error.URLError, OSError:
                    # NOSILENT: a refused or reset connection IS the not-ready
                    # signal this loop exists to poll for, so each one is an
                    # expected outcome rather than a fault worth reporting.
                    # The failure that matters -- never becoming ready -- is
                    # raised at the deadline below, naming the URL.
                    pass
                if time.monotonic() > deadline:
                    pytest.fail(f"SearXNG container at {base_url} never answered a JSON search")
                time.sleep(2)
            yield base_url
