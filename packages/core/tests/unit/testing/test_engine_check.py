"""a run naming the engine it means (``THREETEARS_TEST_ENGINE``) stops when its database is another.

A YugabyteDB plan test whose fixture quietly reached Postgres passes on Postgres's plan and is reported green on
YugabyteDB (the hub's shared ``conn`` fixture did, for 18 files). Every 3tears integration fixture reaches its server
through ``db_container``; :func:`~threetears.core.testing.fixtures.check_engine` is asked there.
"""

from __future__ import annotations

import pytest

from threetears.core.testing import fixtures


@pytest.mark.parametrize(
    ("expected", "url", "server", "stops"),
    [
        ("yugabyte", "postgresql://yb", "yugabyte", False),
        ("yugabyte", "postgresql://pg", "postgres", True),
        ("postgres", "postgresql://yb", "yugabyte", True),
        ("postgres", "postgresql://pg", "postgres", False),
        ("yugabyte", None, None, True),
        ("postgres", None, None, False),
        ("yb", "postgresql://yb", "yugabyte", True),
        (None, "postgresql://pg", "postgres", False),
    ],
)
def test_a_run_reaching_another_engine_than_it_names_stops(
    monkeypatch: pytest.MonkeyPatch, expected: str | None, url: str | None, server: str | None, stops: bool
) -> None:
    """Flips if a run naming YugabyteDB goes on against a Postgres server (or with no server named, where a local
    Postgres container would stand in), a run naming Postgres goes on against YugabyteDB, an unknown engine name is
    taken, or a run naming none is asked anything."""
    if expected is None:
        monkeypatch.delenv("THREETEARS_TEST_ENGINE", raising=False)
    else:
        monkeypatch.setenv("THREETEARS_TEST_ENGINE", expected)
    asked: list[str] = []

    def engine(at: str) -> str:
        asked.append(at)
        assert server is not None
        return server

    monkeypatch.setattr(fixtures, "server_engine", engine)
    if stops:
        with pytest.raises(pytest.exit.Exception):
            fixtures.check_engine(url)
    else:
        fixtures.check_engine(url)
    if expected is None:
        assert not asked
