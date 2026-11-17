"""the process-mode YugabyteDB the pool's YugabyteDB suites start: its command lines, its address, and its checks.

The integration suites on YugabyteDB (``.testament.toml``'s ``[suites.integration_yugabyte]``) start one single-node
cluster per test process, as child processes, from the binaries the testament image copies from the pinned
``yugabytedb/yugabyte`` image. These tests hold what that start decides without running it (the servers themselves
are proven in ``packages/core/tests/integration/test_process_yugabyte.py``, on the pool):

- the cluster runs as ``yugabyted`` runs the local YugabyteDB every earlier proof ran on: one tablet per table, read
  committed, the cost-based optimizer, packed rows, the clock-skew allowance; each process bound to the cluster's own
  loopback address, so two clusters in one container never share a port;
- the address is a loopback one other than 127.0.0.1, free on every port the cluster listens on;
- a cluster whose tcmalloc runs per-CPU caches is refused (yugabyte-db #33458 corrupts its heap on some kernels);
- ``db_container`` starts it when a process-mode run names YugabyteDB and no server.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from threetears.core.testing import fixtures, services


def _flags(argv: list[str]) -> dict[str, str]:
    """``--name=value`` arguments as a mapping (a bare ``--name`` maps to ``""``)."""
    found: dict[str, str] = {}
    for arg in argv[1:]:
        assert arg.startswith("--"), arg
        name, _, value = arg[2:].partition("=")
        found[name] = value
    return found


#: the settings every earlier YugabyteDB proof ran under (``yugabyted start`` in the local harness's container)
_YUGABYTED = {
    "yb_num_shards_per_tserver": "1",
    "ysql_num_shards_per_tserver": "1",
    "yb_enable_read_committed_isolation": "true",
    "max_clock_skew_usec": "5000000",
    "use_memory_defaults_optimized_for_ysql": "true",
}


def test_the_master_runs_as_yugabyted_runs_it_on_its_own_address(tmp_path: Path) -> None:
    """Flips if the master drops a setting the local proofs ran under, or binds anything off its address."""
    flags = _flags(services.yugabyte_master_argv("127.9.8.7", tmp_path / "master"))
    assert _YUGABYTED.items() <= flags.items()
    assert flags["replication_factor"] == "1"
    assert flags["master_addresses"] == flags["rpc_bind_addresses"] == "127.9.8.7:7100"
    assert flags["webserver_interface"] == "127.9.8.7"
    assert flags["fs_data_dirs"] == str(tmp_path / "master")
    assert flags["callhome_enabled"] == "false"
    assert "stop_on_parent_termination" in flags
    assert int(flags["memory_limit_hard_bytes"]) > 0


def test_the_tserver_runs_as_yugabyted_runs_it_on_its_own_address(tmp_path: Path) -> None:
    """Flips if the tablet server drops a setting the local proofs ran under (the cost-based optimizer, packed rows),
    binds anything off its address, or serves YSQL anywhere but 5433 there."""
    flags = _flags(services.yugabyte_tserver_argv("127.9.8.7", tmp_path / "tserver"))
    assert _YUGABYTED.items() <= flags.items()
    assert flags["ysql_enable_packed_row"] == "true"
    assert flags["ysql_pg_conf_csv"] == "yb_enable_cbo=on"
    # as in production: a client gives the password, and a wrong one is refused
    assert flags["ysql_enable_auth"] == "true"
    assert flags["tserver_master_addrs"] == "127.9.8.7:7100"
    assert flags["rpc_bind_addresses"] == "127.9.8.7:9100"
    assert flags["pgsql_proxy_bind_address"] == "127.9.8.7:5433"
    assert flags["cql_proxy_bind_address"] == "127.9.8.7:9042"
    assert flags["webserver_interface"] == "127.9.8.7"
    assert flags["start_redis_proxy"] == "false"
    assert flags["callhome_enabled"] == "false"
    assert "stop_on_parent_termination" in flags
    assert int(flags["memory_limit_hard_bytes"]) > 0


def test_a_cluster_address_is_a_free_loopback_other_than_127_0_0_1() -> None:
    """Flips if a cluster takes 127.0.0.1 (where every other service and test listens) or an address off loopback."""
    for _ in range(20):
        address = services.yugabyte_address()
        first, *rest = (int(part) for part in address.split("."))
        assert first == 127 and address != "127.0.0.1", address
        assert all(0 <= part <= 255 for part in rest)


def test_an_address_with_a_cluster_port_taken_is_passed_over() -> None:
    """Flips if an address one of whose cluster ports is taken is handed out."""
    picks = iter(["127.3.3.3", "127.3.3.4"])
    chosen = services.yugabyte_address(pick=lambda: next(picks), free=lambda address: address == "127.3.3.4")
    assert chosen == "127.3.3.4"


def test_a_port_held_on_the_address_is_not_free() -> None:
    """Flips if the probe calls an address free while one of the ports asked about is taken there."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        assert not services.ports_free("127.0.0.1", (port,))
    assert services.ports_free("127.0.0.1", (port,))


@pytest.mark.parametrize(
    ("logged", "verdict"),
    [
        pytest.param("I1117 mem_tracker.cc:245] TCMalloc per cpu caches active: 0\n", True, id="off"),
        pytest.param("I1117 mem_tracker.cc:245] TCMalloc per cpu caches active: 1\n", False, id="on"),
        pytest.param("I1117 nothing about it\n", None, id="not-yet"),
    ],
)
def test_the_per_cpu_cache_line_is_read(tmp_path: Path, logged: str, verdict: bool | None) -> None:
    """Flips if a server running tcmalloc's per-CPU caches reads as safe, or one that has not said reads as either."""
    log = tmp_path / "yb-tserver.INFO"
    log.write_text(logged)
    assert services.per_cpu_caches_off(log) is verdict


def test_a_missing_log_has_not_said(tmp_path: Path) -> None:
    assert services.per_cpu_caches_off(tmp_path / "absent.INFO") is None


def test_outside_process_mode_a_yugabyte_start_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a run naming YugabyteDB with no server, off the pool's process mode, goes on (a local Postgres
    container would stand in, and its plans would read as YugabyteDB's)."""
    monkeypatch.delenv("THREETEARS_TEST_SERVICES", raising=False)
    with pytest.raises(pytest.exit.Exception), services.running_yugabyte():
        pass


@pytest.mark.parametrize(
    ("engine", "mode", "url", "started"),
    [
        pytest.param("yugabyte", "process", None, "yugabyte", id="yugabyte-process"),
        pytest.param("postgres", "process", None, "postgres", id="postgres-process"),
        pytest.param(None, "process", None, "postgres", id="unnamed-process"),
        pytest.param("yugabyte", "process", "postgresql://given", None, id="yugabyte-given"),
    ],
)
def test_db_container_starts_the_engine_the_run_names(
    monkeypatch: pytest.MonkeyPatch, engine: str | None, mode: str, url: str | None, started: str | None
) -> None:
    """Flips if a process-mode run naming YugabyteDB and no server gets a Postgres, if one naming Postgres (or none)
    gets a YugabyteDB, if a server the run names is passed over, or if the started server is not checked."""
    if engine is None:
        monkeypatch.delenv("THREETEARS_TEST_ENGINE", raising=False)
    else:
        monkeypatch.setenv("THREETEARS_TEST_ENGINE", engine)
    monkeypatch.setenv("THREETEARS_TEST_SERVICES", mode)
    if url is None:
        monkeypatch.delenv("THREETEARS_TEST_POSTGRES_URL", raising=False)
    else:
        monkeypatch.setenv("THREETEARS_TEST_POSTGRES_URL", url)
    calls: list[str] = []
    checked: list[str | None] = []

    @contextmanager
    def start(kind: str) -> Iterator[str]:
        calls.append(kind)
        yield f"postgresql://{kind}"

    monkeypatch.setattr(fixtures, "running_postgres", lambda image: start("postgres"))
    monkeypatch.setattr(fixtures, "running_yugabyte", lambda: start("yugabyte"))
    monkeypatch.setattr(fixtures, "check_engine", checked.append)
    gen = fixtures.db_container.__wrapped__("pgvector/pgvector:pg16")  # type: ignore[attr-defined]
    yielded = next(gen)
    assert calls == ([] if started is None else [started])
    assert yielded == (url if started is None else f"postgresql://{started}")
    # a server the run names, or the YugabyteDB it started, is asked its engine; before a Postgres start the run's
    # name alone is checked, as before
    assert checked == ([yielded] if (started == "yugabyte" or url) else [None])
