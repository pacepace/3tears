"""Integration fixtures for the backup round-trip tests.

The round-trip runs the *host's* ``pg_dump``/``pg_restore`` against the container. A newer client
than the server emits settings the older server rejects (e.g. ``transaction_timeout``, pg17+), so
pin the container to a recent major that a current client can safely restore into — matching the
real-deployment rule that the dump tools track the target server version.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def db_image() -> str:
    return "pgvector/pgvector:pg17"
