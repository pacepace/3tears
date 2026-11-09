"""The code half of the no-whole-reads rule, driven by synthesised modules shaped like the paths that failed on
election night, and its one way out: the exemptions ledger."""

from __future__ import annotations

import textwrap
from pathlib import Path

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, no_whole_reads_findings
from threetears.enforcement.no_whole_reads.walker import scan_whole_reads

#: the ledger's name, beside each repo's gate
EXEMPTIONS_NAME = "_no_whole_reads_exemptions.txt"


def _config(tmp_path: Path, modules: dict[str, str], *, ledger: str = "", **options: object) -> NoWholeReadsConfig:
    """write synthesised modules under ``tmp_path/src`` and the ledger beside them, and configure the gate.

    :param tmp_path: the repo root
    :ptype tmp_path: Path
    :param modules: module name onto its source
    :ptype modules: dict[str, str]
    :param ledger: the exemptions ledger's body
    :ptype ledger: str
    :param options: other config fields
    :ptype options: object
    :return: the config
    :rtype: NoWholeReadsConfig
    """
    src = tmp_path / "src"
    for name, source in modules.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / EXEMPTIONS_NAME).write_text(ledger, encoding="utf-8")
    return NoWholeReadsConfig(repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, **options)  # type: ignore[arg-type]


def _symbols(config: NoWholeReadsConfig) -> set[tuple[str, str]]:
    scan = scan_whole_reads(config.repo_root, config.src_roots, verify_prefixes=config.verify_prefixes)
    return {(hit.function, hit.symbol) for hit in scan.hits}


_FOLLOWER = """
    from threetears.core.keyset import keyset_arrow_pages

    class CopyFollower:
        async def _stage(self, whole):
            for name in whole:
                self._whole.add(name)
                await self._stage_whole(name)

        async def _stage_whole(self, table):
            pages = keyset_arrow_pages(self._l3, f"SELECT {cols} FROM {table}", ("k",), types={}, page_size=1000)
            async for page in pages:
                pass

        async def _stage_partition(self, table, partition):
            pages = keyset_arrow_pages(self._l3, f"SELECT {cols} FROM {table}", ("k",), types={}, where=partition,
                                       page_size=1000)
            async for page in pages:
                pass

        async def start(self):
            if await self._checkpoints.stage() is None:
                await self.warm()

        async def warm(self):
            self._whole.update(self._tables)
"""

_EPOCHS = """
    from threetears.core.collections.complete_copy import read_l3_rows

    class ScopeEpochs:
        async def snapshot(self):
            return await read_l3_rows(self._pool, self._table, ("scope", "epoch"), ("scope",), page_size=1000)

        async def scope(self, name):
            return await read_l3_rows(self._pool, self._table, ("scope", "epoch"), ("scope",), where={"scope": name})

        async def settled(self):
            snapshot = await self.snapshot()
            return snapshot
"""

_WRITER = """
    SOURCES = ("results", "calls")

    def checks_sql(asks):
        parts = [f"SELECT {i} AS ask, state, count(*) AS n FROM {t} GROUP BY state" for i, t in enumerate(asks)]
        return " UNION ALL ".join(parts)

    class Store:
        async def count(self, table):
            return await self._backend.fetchval(f"SELECT COUNT(*) FROM {table}")

        async def count_held(self, table, state):
            return await self._backend.fetchval(f"SELECT COUNT(*) FROM {table} WHERE state = $1", state)

        async def stamps(self):
            found = {}
            for source in SOURCES:
                found[source] = await read_l3_rows(self._backend, "fingerprints", ("s",), ("s",), where={"source": source})
            return found

    class Warehouse:
        async def checks_by_state(self, asks):
            return await self._client.query(self._datasource, checks_sql(asks))

    class Refresher:
        async def _check_counts(self, moved):
            for source in moved:
                held = await self._store.count(source.name)

        async def _plan(self, state):
            held = await self._store.count_held("results", state)
"""


def test_the_election_night_paths_are_named_where_they_read_and_where_they_are_called(tmp_path: Path) -> None:
    """Flips if any shape that failed on election night passes: the copy follower's whole stage and the set that
    sends tables to it, the warm a failed checkpoint falls back to, ScopeEpochs.settled's whole snapshot, the
    writer's grouped check and its whole count, and fingerprints read every key of a constant set."""
    config = _config(
        tmp_path, {"follower.py": _FOLLOWER, "epochs.py": _EPOCHS, "writer.py": _WRITER}, verify_prefixes=("src/",)
    )
    found = _symbols(config)
    expected = {
        ("CopyFollower._stage_whole", "unkeyed-pager keyset_arrow_pages"),
        ("CopyFollower._stage", "calls-whole-read CopyFollower._stage_whole"),
        ("CopyFollower._stage", "marks-whole _whole"),
        ("CopyFollower.warm", "marks-whole _whole"),
        ("CopyFollower.start", "whole-call warm"),
        ("ScopeEpochs.snapshot", "unkeyed-pager read_l3_rows"),
        ("ScopeEpochs.settled", "calls-whole-read ScopeEpochs.snapshot"),
        ("checks_sql", "whole-aggregate {}"),
        ("Warehouse.checks_by_state", "calls-whole-read checks_sql"),
        ("Store.count", "whole-aggregate {}"),
        ("Refresher._check_counts", "calls-whole-read Store.count"),
        ("Store.stamps", "whole-by-loop read_l3_rows"),
        ("Store.count_held", "verify-aggregate {}"),
    }
    assert expected <= found, sorted(expected - found)
    # the keyed reads beside them are not named
    assert not {f for f, _ in found} & {"CopyFollower._stage_partition", "ScopeEpochs.scope"}


def test_a_keyed_count_is_a_verification_scan_only_in_the_code_named_load_code(tmp_path: Path) -> None:
    """Flips if a per-key count is flagged outside the repo's load code, or passes inside it."""
    source = {"writer.py": _WRITER}
    assert ("Store.count_held", "verify-aggregate {}") not in _symbols(_config(tmp_path, source))
    assert ("Store.count_held", "verify-aggregate {}") in _symbols(_config(tmp_path, source, verify_prefixes=("src/",)))


def test_a_call_resolves_only_when_every_function_it_may_reach_reads_whole(tmp_path: Path) -> None:
    """Flips if a common method name (``count``) on an unrelated receiver, or one a Protocol stub shares, is taken
    for the whole-read one."""
    modules = {
        "store.py": """
            class Store:
                async def count(self, table):
                    return await self._backend.fetchval(f"SELECT COUNT(*) FROM {table}")
            """,
        "proto.py": """
            class StoreProtocol:
                async def count(self, table):
                    \"\"\"a count.\"\"\"
                    ...
            """,
        "tally.py": """
            class Tally:
                def count(self, panel):
                    self._n += 1

            class Reader:
                def read(self):
                    self._tally.count("x")

            class Loader:
                async def load(self, table):
                    return await self._store.count(table)
            """,
    }
    found = _symbols(_config(tmp_path, modules))
    assert ("Loader.load", "calls-whole-read Store.count") in found
    assert not any(function == "Reader.read" for function, _ in found)


def test_a_whole_named_call_counts_only_when_it_reaches_a_whole_read_or_cannot_be_found(tmp_path: Path) -> None:
    """Flips if every ``_rebuild`` in the repo is named (a NATS waiter's, a pydantic model's) or a rebuild that
    reads whole through another function passes."""
    modules = {
        "snap.py": """
            class Snapshot:
                async def _sync(self):
                    await self._rebuild()

                async def _rebuild(self):
                    await self._read_all_scopes()

                async def _read_all_scopes(self):
                    return await self._pool.fetch("SELECT DISTINCT scope FROM chunks")

            class Waiter:
                def wait(self):
                    self._rebuild()

                def _rebuild(self):
                    self._sub = None

            class Model:
                def declare(self):
                    Model.model_rebuild()
                    self._client.resync_everything()
            """,
    }
    found = _symbols(_config(tmp_path, modules))
    assert ("Snapshot._sync", "whole-call _rebuild") in found
    assert ("Model.declare", "whole-call resync_everything") in found
    assert not any(function in {"Waiter.wait", "Waiter._rebuild"} for function, _ in found)
    assert ("Model.declare", "whole-call model_rebuild") not in found


def test_registered_rows_and_migrations_are_not_data_tables(tmp_path: Path) -> None:
    """Flips if writing registered in-memory rows reads as a whole read, or a migration's backfill is listed."""
    modules = {
        "duck.py": """
            def insert(connection, target, arrow):
                connection.register("_bulk_rows", arrow)
                connection.execute(f"INSERT INTO {target} SELECT * FROM _bulk_rows")
            """,
        "migrations/v001.py": """
            SQL = "UPDATE memories SET v = 1 WHERE v IS NULL"
            """,
    }
    assert _symbols(_config(tmp_path, modules)) == set()


_COLD = """
    class Follower:
        async def warm(self):
            return await read_l3_rows(self._pool, "t", ("k",), ("k",))
"""

_PROOF = """
    async def test_warm_runs_only_on_an_empty_copy():
        follower = Follower()
        assert follower.warm
"""


def _ledger(
    kind: str = "cold-start", count: int = 1, test: str = "tests/test_cold.py::test_warm_runs_only_on_an_empty_copy"
) -> str:
    reason = "the first start of an empty copy, with no checkpoint in L2 to load"
    return f"src/follower.py :: Follower.warm :: {count} :: {kind} :: {test} :: {reason}\n"


def _proof(tmp_path: Path, body: str = _PROOF) -> None:
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_cold.py").write_text(textwrap.dedent(body), encoding="utf-8")


def test_without_an_exemption_every_hit_is_a_finding(tmp_path: Path) -> None:
    """Flips if the gate stops failing on a whole read nobody listed."""
    findings = no_whole_reads_findings(_config(tmp_path, {"follower.py": _COLD}))
    assert len(findings) == 1
    assert "src/follower.py:4 in Follower.warm: unkeyed-pager read_l3_rows" in findings[0]


def test_a_cold_start_with_its_proving_test_is_exempt(tmp_path: Path) -> None:
    """Flips if a true cold start, listed with the test proving where it is reached, still fails."""
    _proof(tmp_path)
    assert no_whole_reads_findings(_config(tmp_path, {"follower.py": _COLD}, ledger=_ledger())) == []


def test_an_exemption_needs_a_cold_start_or_schema_change_and_a_test_that_names_the_function(tmp_path: Path) -> None:
    """Flips if a ledger line is accepted for any other reason, without a test, with a test that never names the
    function, or with a hit count the code does not have."""
    _proof(tmp_path, "async def test_warm_runs_only_on_an_empty_copy():\n    assert True\n")
    cases = {
        "small-table": "is not a reason to read whole",
        "cold-start-count": "pins 2 hits but the code has 1",
        "cold-start-missing": "names no test file",
        "cold-start-silent": "never names warm",
    }
    for case, message in cases.items():
        ledger = {
            "small-table": _ledger(kind="small-table"),
            "cold-start-count": _ledger(count=2),
            "cold-start-missing": _ledger(test="tests/nowhere.py::test_x"),
            "cold-start-silent": _ledger(),
        }[case]
        findings = no_whole_reads_findings(_config(tmp_path, {"follower.py": _COLD}, ledger=ledger))
        assert any(message in finding for finding in findings), (case, findings)
