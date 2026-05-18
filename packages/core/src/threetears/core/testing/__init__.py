"""shared test infrastructure for 3tears + downstream consumers.

single source of truth for testcontainer fixtures, docker-availability
gating, and "is the local NATS reachable" probes that integration
tests use across every repo. lives in :mod:`threetears.core` so every
package in the dependency tree (3tears libraries, ``aibots`` hub,
``aibots_agents`` SDK, downstream agents) can import from one place
without duplicating the harness per-repo.

usage shape (downstream pytest):

.. code-block:: python

    pytest_plugins = ["threetears.core.testing.fixtures"]

then any test can declare ``def test_x(nats_container, db_container)``
to get session-scoped testcontainers that auto-skip when docker is
unreachable. for tests that hit a long-running NATS instead of
spinning a container, use :func:`nats_reachable` to gate the test
with a ``pytest.mark.skipif``.

DO NOT define your own ``check_docker_available`` /
``NatsContainer`` / ``PostgresContainer`` fixtures in per-repo
conftests. import from here. the fixtures here are the ONLY ones
that have been audited for the docker-skip pattern, the asyncpg-URL
normalisation, and the lazy-import discipline (testcontainers /
docker imports happen inside fixture bodies so this module stays
cheap to import from non-test code).
"""

from __future__ import annotations

from threetears.core.testing.containers import (
    check_docker_available,
    nats_reachable,
    skip_without_docker_marker,
    skip_without_nats_marker,
)
from threetears.core.testing.sqla_parity import (
    assert_tables_equivalent,
    column_signature,
    fk_constraint_signature,
    index_signature,
    inline_fk_signatures,
)

__all__ = [
    "assert_tables_equivalent",
    "check_docker_available",
    "column_signature",
    "fk_constraint_signature",
    "index_signature",
    "inline_fk_signatures",
    "nats_reachable",
    "skip_without_docker_marker",
    "skip_without_nats_marker",
]
