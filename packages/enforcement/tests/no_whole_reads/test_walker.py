"""The code half of the no-whole-reads rule, driven by synthesised modules shaped like the paths that failed on
election night, and its one way out: the exemptions ledger."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import re

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, no_whole_reads_findings
from threetears.enforcement.no_whole_reads.schema import collect_schema, index_findings

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
    # 3tears' own declarations are not read: each test declares the tables it reads
    options.setdefault("dependency_schema", False)
    return NoWholeReadsConfig(repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, **options)  # type: ignore[arg-type]


#: a finding naming one read: ``<path>:<line> in <function>: <kind> <what> -- ...``
_HIT: re.Pattern[str] = re.compile(r"^\S+:\d+ in (\S+): (\S+ \S+) -- ")


def _symbols(config: NoWholeReadsConfig) -> set[tuple[str, str]]:
    """``(function, symbol)`` of every read the gate names."""
    return {
        (match.group(1), match.group(2))
        for match in (_HIT.match(finding) for finding in no_whole_reads_findings(config))
        if match is not None
    }


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
    from threetears.core.collections.schema_backed import TableSchema

    SOURCES = ("results", "calls")
    RESULTS = TableSchema(name="results", primary_key=("state", "geo"), columns=[])

    def checks_sql(asks):
        parts = [f"SELECT {i} AS ask, state, count(*) AS n FROM {t} GROUP BY state" for i, t in enumerate(asks)]
        return " UNION ALL ".join(parts)

    class Store:
        async def count(self, table):
            return await self._backend.fetchval(f"SELECT COUNT(*) FROM {table}")

        async def count_held(self, table, state):
            return await self._backend.fetchval("SELECT COUNT(*) FROM results WHERE state = $1", state)

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
    }
    assert expected <= found, sorted(expected - found)
    # the keyed reads beside them are not named
    assert not {f for f, _ in found} & {"CopyFollower._stage_partition", "ScopeEpochs.scope"}


def test_a_keyed_count_is_no_verification_scan_even_in_load_code(tmp_path: Path) -> None:
    """Flips if a count bound by a key prefix is named, in load code or out of it, or if a whole count is not."""
    source = {"writer.py": _WRITER}
    for config in (_config(tmp_path, source), _config(tmp_path, source, verify_prefixes=("src/",))):
        found = _symbols(config)
        assert not any(function == "Store.count_held" for function, _ in found), found
        assert ("Store.count", "whole-aggregate {}") in found, found


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

    async def test_warm_is_within_its_calibrated_budget():
        baseline = await plain_read_seconds(conn, "SELECT * FROM t")
        took = await best_seconds(Follower().warm)
        assert not budget_problem("warm", took, baseline)
"""


_SPEED = "tests/test_cold.py::test_warm_is_within_its_calibrated_budget"


def _ledger(
    kind: str = "cold-start",
    count: int = 1,
    test: str = "tests/test_cold.py::test_warm_runs_only_on_an_empty_copy",
    speed: str = _SPEED,
) -> str:
    reason = "the first start of an empty copy, with no checkpoint in L2 to load"
    return f"src/follower.py :: Follower.warm :: {count} :: {kind} :: {test} :: {speed} :: {reason}\n"


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
    function, with a hit count the code does not have, or without a speed test held to a calibrated budget."""
    _proof(
        tmp_path,
        "async def test_warm_runs_only_on_an_empty_copy():\n    assert True\n\n"
        "async def test_warm_is_within_its_calibrated_budget():\n    assert Follower().warm\n",
    )
    cases = {
        "small-table": "is not a reason to read whole",
        "cold-start-count": "pins 2 hits but the code has 1",
        "cold-start-missing": "names no test file",
        "cold-start-silent": "never names warm",
        "cold-start-no-speed": "the speed test names no test file",
        "cold-start-no-budget": "never calls budget_problem",
    }
    for case, message in cases.items():
        ledger = {
            "small-table": _ledger(kind="small-table"),
            "cold-start-count": _ledger(count=2),
            "cold-start-missing": _ledger(test="tests/nowhere.py::test_x"),
            "cold-start-silent": _ledger(),
            "cold-start-no-speed": _ledger(speed="tests/nowhere.py::test_x"),
            "cold-start-no-budget": _ledger(),
        }[case]
        findings = no_whole_reads_findings(_config(tmp_path, {"follower.py": _COLD}, ledger=ledger))
        assert any(message in finding for finding in findings), (case, findings)


def test_a_class_constructor_reaches_its_init_not_an_unfound_whole_call(tmp_path: Path) -> None:
    """Flips if constructing a record type whose name says reload (a dataclass of task handles) is a whole call,
    or if constructing a class whose ``__init__`` reads whole passes."""
    modules = {
        "jobs.py": """
            from dataclasses import dataclass

            @dataclass
            class _Reload:
                task: object

            class _ReloadAll:
                def __init__(self, pool):
                    self._rows = pool.fetch("SELECT * FROM bands")

            class Precompute:
                def start(self, task):
                    return _Reload(task=task)

                def start_all(self, pool):
                    return _ReloadAll(pool)
            """,
    }
    found = _symbols(_config(tmp_path, modules))
    assert not any(function == "Precompute.start" for function, _ in found), found
    assert ("Precompute.start_all", "calls-whole-read _ReloadAll.__init__") in found


_RESET = """
    from threetears.core.operator import operator_reset, operator_command

    @operator_reset("an operator empties the tables of one data space it names", proof="tests/test_ops.py::test_reset")
    async def truncate_space(conn, names):
        await conn.execute(f"TRUNCATE TABLE {names}")
        await conn.fetch("SELECT * FROM data")

    @operator_command("a restore drill counts every table of the scratch copy", proof="tests/test_ops.py::test_drill")
    async def drill(conn, table):
        return await conn.fetchval(f"SELECT count(*) FROM {table}")

    @operator_command("a restore drill counts every table of the scratch copy", proof="tests/test_ops.py::test_drill")
    async def drill_entry(conn):
        return await drill(conn, "t")
"""

_OPS_PROOF = """
    def test_reset():
        assert truncate_space

    def test_drill():
        assert drill and drill_entry
"""


def _ops(tmp_path: Path, body: str = _OPS_PROOF) -> None:
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_ops.py").write_text(textwrap.dedent(body), encoding="utf-8")


def test_an_operator_reset_may_truncate_and_an_operator_command_may_read_its_copy(tmp_path: Path) -> None:
    """Flips if an operator's marked reset still fails on its TRUNCATE, if its mark lets a whole read through, or if
    a marked command over a scratch restore (and its marked entry) fails on its counts."""
    _ops(tmp_path)
    findings = no_whole_reads_findings(_config(tmp_path, {"ops.py": _RESET}))
    assert len(findings) == 1, findings
    assert "in truncate_space: whole-select data" in findings[0]


def test_an_operator_mark_needs_its_proof_and_no_unmarked_caller(tmp_path: Path) -> None:
    """Flips if a marked function is accepted when something unmarked in the repo calls it, or when its proof
    names no test, or a test that never names it."""
    _ops(tmp_path, "def test_reset():\n    assert truncate_space\n\ndef test_drill():\n    assert True\n")
    caller = """
        from ops import truncate_space

        async def nightly(conn):
            await truncate_space(conn, "t")
        """
    findings = no_whole_reads_findings(_config(tmp_path, {"ops.py": _RESET, "nightly.py": caller}))
    assert any("nightly.py:5 in nightly" in finding and "@operator_reset" in finding for finding in findings)
    assert any(
        "drill @operator_command: the proof" in finding and "never names drill" in finding for finding in findings
    )


_MARKED_METHOD = """
    from threetears.core.operator import operator_command

    class RereadStateTool:
        @operator_command("an operator's explicit one-off reread of every state row", proof="tests/test_ops.py::test_reread")
        async def execute(self, conn):
            return await conn.fetch("SELECT * FROM state")
"""


def test_an_entry_only_mark_counts_a_method_call_only_on_a_receiver_known_to_be_its_class(tmp_path: Path) -> None:
    """Flips if a call of a marked method's name on a receiver of unknown type (``self._db.execute``, a DuckDB
    connection) is reported as an unmarked caller of it, or if a receiver typed or assigned as the marked class is
    not."""
    _ops(tmp_path, "def test_reread():\n    assert RereadStateTool.execute\n")
    unrelated = """
        class Pod:
            def __init__(self, db):
                self._db = db

            async def step(self, key):
                return await self._db.execute("SELECT 1 FROM state WHERE k = $1", key)

            async def other(self, conn, key):
                return await conn.execute("SELECT 1 FROM state WHERE k = $1", key)
        """
    typed = """
        from tool import RereadStateTool

        async def annotated(tool: RereadStateTool, conn):
            return await tool.execute(conn)

        async def assigned(conn):
            tool = RereadStateTool()
            return await tool.execute(conn)

        class Holder:
            def __init__(self):
                self._tool = RereadStateTool()

            async def run(self, conn):
                return await self._tool.execute(conn)
        """
    findings = no_whole_reads_findings(
        _config(tmp_path, {"tool.py": _MARKED_METHOD, "pod.py": unrelated, "typed.py": typed})
    )
    callers = [finding for finding in findings if "marked @operator_command" in finding]
    assert not [finding for finding in callers if "pod.py" in finding], callers
    for function in ("in annotated:", "in assigned:", "in Holder.run:"):
        assert any("typed.py" in finding and function in finding for finding in callers), (function, callers)


def test_a_ledger_line_without_its_speed_test_field_is_malformed(tmp_path: Path) -> None:
    """Flips if a line of the old six fields, a cold start with no speed test, is still read as an exemption."""
    _proof(tmp_path)
    reason = "the first start of an empty copy, with no checkpoint in L2 to load"
    old = f"src/follower.py :: Follower.warm :: 1 :: cold-start :: tests/test_cold.py::test_warm_runs_only_on_an_empty_copy :: {reason}\n"
    findings = no_whole_reads_findings(_config(tmp_path, {"follower.py": _COLD}, ledger=old))
    assert any("expected 7 fields" in finding for finding in findings), findings


_REGISTRY = """
    from threetears.core.collections.schema_backed import Index, TableSchema

    DATASOURCES = TableSchema(
        name="datasources",
        primary_key="id",
        columns=[],
        indexes=(
            Index(
                "datasources_by_kind",
                "kind",
                purpose="src/registry.py::Digest.refresh_active",
                plan_test="tests/test_plans.py::test_refresh_active_uses_datasources_by_kind",
            ),
        ),
    )

    class Digest:
        async def one(self, conn, datasource_id):
            return await conn.fetchrow("SELECT id, kind FROM datasources WHERE id = $1", datasource_id)

        async def page(self, conn, after, size):
            return await conn.fetch("SELECT id FROM datasources WHERE id > $1 ORDER BY id LIMIT $2", after, size)

        async def refresh_active(self, conn, kind):
            return await conn.fetch("SELECT id, customer_id FROM datasources WHERE status = 'active' AND kind = $1", kind)

        async def elsewhere(self, conn, kind):
            return await conn.fetch("SELECT id, customer_id FROM datasources WHERE status = 'active' AND kind = $1", kind)

        async def missing(self, conn, key):
            return await conn.fetch("SELECT a FROM undeclared WHERE k = $1", key)
"""

_PLANS = """
    from threetears.core.testing.plans import explain_statement, plan_uses_index

    async def test_refresh_active_uses_datasources_by_kind(conn):
        sql = "SELECT id, customer_id FROM datasources WHERE status = 'active' AND kind = $1"
        assert plan_uses_index(await explain_statement(conn, sql, "pg"), "datasources_by_kind")
"""

_PURPOSE = 'purpose="src/registry.py::Digest.refresh_active",'
_PLAN_TEST = 'plan_test="tests/test_plans.py::test_refresh_active_uses_datasources_by_kind",'


def _plans(tmp_path: Path, body: str = _PLANS) -> None:
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_plans.py").write_text(textwrap.dedent(body), encoding="utf-8")


def _index_findings(tmp_path: Path, source: str) -> list[str]:
    findings = no_whole_reads_findings(_config(tmp_path, {"registry.py": source}))
    return [finding for finding in findings if ": index-" in finding]


def test_a_read_is_keyed_by_its_key_a_keyset_page_or_the_index_its_purpose_names(tmp_path: Path) -> None:
    """Flips if a status-and-kind filter outside the purposed function passes, if the purposed function's read, a
    key read or a keyset page fails, or if a read of a table declared nowhere passes."""
    _plans(tmp_path)
    found = _symbols(_config(tmp_path, {"registry.py": _REGISTRY}))
    assert ("Digest.elsewhere", "non-key-where datasources") in found
    assert ("Digest.missing", "unknown-schema undeclared") in found
    assert not {function for function, _ in found} & {"Digest.one", "Digest.page", "Digest.refresh_active"}, found


def test_a_purposed_index_with_its_plan_test_is_accepted(tmp_path: Path) -> None:
    """Flips if an index naming the read that uses it, with a plan test proving it, is still refused."""
    _plans(tmp_path)
    assert _index_findings(tmp_path, _REGISTRY) == []


@pytest.mark.parametrize(
    ("old", "new", "kind"),
    [
        (_PURPOSE, "", "index-no-purpose"),
        (_PURPOSE, 'purpose="src/registry.py::Digest.gone",', "index-purpose-unknown"),
        (_PURPOSE, 'purpose="src/registry.py::Digest.one",', "index-purpose-unused"),
        (_PLAN_TEST, "", "index-no-plan-test"),
    ],
)
def test_an_index_needs_a_purpose_its_read_uses_and_a_plan_test(tmp_path: Path, old: str, new: str, kind: str) -> None:
    """Flips if an index without a purpose, with a purpose naming no function, with a purpose whose function never
    reads the table by it, or without a plan test, is accepted."""
    _plans(tmp_path)
    findings = _index_findings(tmp_path, _REGISTRY.replace(old, new))
    assert any(f"index datasources_by_kind on datasources: {kind}" in finding for finding in findings), findings


def test_a_plan_test_must_ask_the_plan_for_the_index(tmp_path: Path) -> None:
    """Flips if a plan test that never calls plan_uses_index, or never names the index, proves the index is used."""
    _plans(tmp_path, _PLANS.replace("plan_uses_index(await", "(await"))
    assert any("never calls plan_uses_index" in finding for finding in _index_findings(tmp_path, _REGISTRY))
    _plans(tmp_path, _PLANS.replace('"datasources_by_kind")', '"another_index")'))
    assert any("never names datasources_by_kind" in finding for finding in _index_findings(tmp_path, _REGISTRY))


def test_a_wide_index_or_a_crowded_table_needs_a_written_reason(tmp_path: Path) -> None:
    """Flips if an index with INCLUDE columns, or one of more than three on a table, is accepted without a written
    reason, or refused with one."""
    _plans(tmp_path)
    wide = _REGISTRY.replace(_PURPOSE, f'include=("customer_id",),\n                {_PURPOSE}')
    assert any("index-wide-no-reason" in finding for finding in _index_findings(tmp_path, wide))
    reason = 'reason="the digest answers each active datasource\'s customer from the index alone",'
    reasoned = wide.replace(_PURPOSE, f"{_PURPOSE}\n                {reason}")
    assert not any("no-reason" in finding for finding in _index_findings(tmp_path, reasoned))
    extra = "".join(f'            Index("datasources_extra_{n}", "c{n}"),\n' for n in range(3))
    crowded = _REGISTRY.replace(
        "        ),\n    )\n\n    class Digest", f"{extra}        ),\n    )\n\n    class Digest"
    )
    crowding = [finding for finding in _index_findings(tmp_path, crowded) if "index-crowded-no-reason" in finding]
    assert len(crowding) == 4, crowding


_MIGRATION = '''
UP = """
CREATE TABLE grants (grant_id UUID PRIMARY KEY, agent_id UUID, status TEXT);
CREATE INDEX grants_by_status ON grants (status);
-- purpose: src/grants.py::agent_grants
-- plan_test: tests/test_plans.py::test_agent_grants_uses_grants_by_agent
CREATE INDEX grants_by_agent ON grants (agent_id);
"""
'''

_GRANTS = """
    async def agent_grants(conn, agent_id):
        return await conn.fetch("SELECT grant_id FROM grants WHERE agent_id = $1", agent_id)

    async def active(conn):
        return await conn.fetch("SELECT grant_id FROM grants WHERE status = $1", "active")

    async def one(conn, grant_id):
        return await conn.fetchrow("SELECT * FROM grants WHERE grant_id = $1", grant_id)

    async def latest(conn, state):
        return await conn.fetch("SELECT * FROM latest WHERE state = $1", state)
"""

_TABLE_LIST = """
providers:
- tables:
  - name: latest
    columns: [{name: state}, {name: geo}]
    primary_key: [state, geo]
    indexes:
    - name: latest_by_geo
      columns: [geo]
"""


def test_migration_ddl_and_a_table_list_declare_keys_and_indexes(tmp_path: Path) -> None:
    """Flips if a key or index declared only in a migration's DDL or a tables.yaml is not read, if a later
    migration's DROP INDEX is not followed, or if an index declared there without a purpose passes."""
    modules = {
        "migrations/v001.py": _MIGRATION,
        "migrations/v002.py": 'UP = "DROP INDEX IF EXISTS grants_by_status"\n',
        "grants.py": _GRANTS,
    }
    _plans(
        tmp_path,
        "def test_agent_grants_uses_grants_by_agent(plan):\n    assert plan_uses_index(plan, 'grants_by_agent')\n",
    )
    (tmp_path / "tables.yaml").write_text(_TABLE_LIST, encoding="utf-8")
    config = _config(tmp_path, modules)
    found = _symbols(config)
    assert ("active", "non-key-where grants") in found
    assert not {function for function, _ in found} & {"agent_grants", "one", "latest"}, found
    findings = no_whole_reads_findings(config)
    assert any("tables.yaml" in f and "index latest_by_geo on latest: index-no-purpose" in f for f in findings)
    assert not any("grants_by_status" in finding or "grants_by_agent" in finding for finding in findings), findings


_ASSEMBLED = """
    from threetears.core.collections.schema_backed import TableSchema, UniqueConstraint

    GRANTS = TableSchema(
        name="grants",
        primary_key=("customer_id", "id"),
        columns=[],
        unique_constraints=(UniqueConstraint("grants_name_uq", "customer_id", "name"),),
    )

    async def by_id(conn, grant_id):
        where = "WHERE customer_id = $1 AND id = $2"
        return await conn.fetch(f"SELECT * FROM grants {where}", grant_id)

    async def by_name(conn, customer_id, name):
        return await conn.fetch("SELECT * FROM grants WHERE customer_id = $1 AND name = $2", customer_id, name)

    async def by_bare_id(conn, grant_id):
        return await conn.fetch("SELECT * FROM grants WHERE id = $1", grant_id)

    async def maybe_all(conn, grant_id):
        where = ""
        if grant_id is not None:
            where = "WHERE customer_id = $1 AND id = $2"
        return await conn.fetch(f"SELECT * FROM grants {where}")

    async def handed_in(conn, where):
        return await conn.fetch(f"SELECT * FROM grants {where}")
"""


def test_a_predicate_bound_once_is_judged_and_one_the_walk_cannot_see_is_unknown(tmp_path: Path) -> None:
    """Flips if a predicate the function binds once is not judged as written, if a TableSchema's unique constraint
    is no key, or if a predicate bound twice (maybe empty), or handed in, passes unseen."""
    found = _symbols(_config(tmp_path, {"grants.py": _ASSEMBLED}))
    assert ("maybe_all", "unknown-predicate grants") in found, found
    assert ("handed_in", "unknown-predicate grants") in found, found
    assert ("by_bare_id", "non-key-where grants") in found, found
    assert not {function for function, _ in found} & {"by_id", "by_name"}, found


_TABLES = """
    from threetears.core.collections.schema_backed import TableSchema

    GEOS = "geos"
    LSF_STAGE = "__stage_lsf"
    GEOS_SCHEMA = TableSchema(name=GEOS, primary_key="geo_id", columns=[])
"""

_PAGE = """
    from pkg.tables import GEOS, LSF_STAGE

    def quoted(name):
        return name

    async def one(conn, geo_id):
        return await conn.fetch(f"SELECT * FROM {quoted(GEOS)} WHERE geo_id = $1", geo_id)

    async def by_state(conn, state):
        return await conn.fetch(f"SELECT * FROM {GEOS} WHERE state = $1", state)

    def from_copy(snapshot, geo_id):
        return snapshot.backend.execute_query(f"SELECT * FROM {snapshot.relation(GEOS)} WHERE geo_id = {geo_id}")

    def copy_by_state(snapshot, state):
        return snapshot.backend.execute_query(f"SELECT * FROM {snapshot.relation_reading(GEOS, set())} WHERE state = {state}")

    async def staged(conn):
        return await conn.fetch(f"SELECT * FROM {LSF_STAGE}")
"""


def test_a_constant_resolves_through_its_import_and_a_call_on_it(tmp_path: Path) -> None:
    """Flips if a table named by an imported constant, bare or handed to a call, is not judged against its
    declaration, or if a stage named through a constant is not the change set the repo says it is."""
    modules = {"pkg/tables.py": _TABLES, "pkg/page.py": _PAGE}
    found = _symbols(_config(tmp_path, modules, transient_relations=r"^__stage"))
    assert ("by_state", "non-key-where geos") in found, found
    assert not {function for function, _ in found} & {"one", "staged"}, found


def test_a_local_copy_is_keyed_only_by_its_own_declaration(tmp_path: Path) -> None:
    """Flips if a local copy's relation is keyed by the L3 table that shares its name, or not keyed by its own
    declared key, or if declaring the copy's key loosens the L3 table."""
    modules = {"pkg/tables.py": _TABLES, "pkg/page.py": _PAGE}
    found = _symbols(_config(tmp_path, modules))
    assert ("from_copy", "unknown-schema copy.geos") in found, found
    keyed = _symbols(_config(tmp_path, modules, local_relations={"geos": ("geo_id",)}))
    assert not any(function == "from_copy" for function, _ in keyed), keyed
    assert ("copy_by_state", "non-key-where copy.geos") in keyed, keyed
    loosened = _symbols(_config(tmp_path, modules, local_relations={"geos": ("state",)}))
    assert ("by_state", "non-key-where geos") in loosened, loosened


def test_a_local_copy_may_declare_several_keys(tmp_path: Path) -> None:
    """Flips if a read on any of a copy's declared keys fails, or one on none of them passes."""
    modules = {"pkg/tables.py": _TABLES, "pkg/page.py": _PAGE}
    found = _symbols(_config(tmp_path, modules, local_relations={"geos": (("geo_id",), ("state",))}))
    assert not {function for function, _ in found} & {"from_copy", "copy_by_state"}, found
    neither = _symbols(_config(tmp_path, modules, local_relations={"geos": (("office_key",), ("district",))}))
    assert {("from_copy", "non-key-where copy.geos"), ("copy_by_state", "non-key-where copy.geos")} <= neither


def test_an_index_a_dependency_declares_is_the_dependency_s_to_judge(tmp_path: Path) -> None:
    """Flips if the repo's table-list copy of a dependency's index is judged by the repo's gate, or if a same-named
    index on another table escapes the repo's gate."""
    dependency = tmp_path / "dep"
    dependency.mkdir()
    (dependency / "tables.py").write_text(
        textwrap.dedent(
            """
            from threetears.core.collections.schema_backed import Index, TableSchema

            REDEMPTIONS = TableSchema(
                name="coordination_redemptions",
                primary_key="token_id",
                columns=[],
                indexes=(Index("idx_redemptions_expiry", "expires_at"),),
            )
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    listed = """
    providers:
    - tables:
      - name: coordination_redemptions
        columns: [{name: token_id}, {name: expires_at}]
        primary_key: [token_id]
        indexes:
        - name: idx_redemptions_expiry
          columns: [expires_at]
      - name: own_redemptions
        columns: [{name: token_id}, {name: expires_at}]
        primary_key: [token_id]
        indexes:
        - name: idx_redemptions_expiry
          columns: [expires_at]
    """
    (tmp_path / "tables.yaml").write_text(textwrap.dedent(listed), encoding="utf-8")
    catalog = collect_schema(tmp_path, (tmp_path / "src",), dependency_roots=(dependency,))
    findings = index_findings(catalog, tmp_path, set())
    assert any("on own_redemptions: index-no-purpose" in finding for finding in findings), findings
    assert not any("on coordination_redemptions" in finding for finding in findings), findings


_RING = """
    from threetears.core.collections.schema_backed import quote_identifier

    PREFIX = "threetears_change_notes_"

    def note_table(index):
        return f"{PREFIX}{index}"

    def ddl(owner, index):
        return f"CREATE TABLE IF NOT EXISTS {owner}.{quote_identifier(note_table(index))} (id UUID PRIMARY KEY, scope TEXT)"
"""

_RING_READS = """
    from pkg.ring import note_table

    async def by_id(conn, owner, index, note_id):
        return await conn.fetch(f"SELECT * FROM {owner}.{quote_identifier(note_table(index))} WHERE id = $1", note_id)

    async def by_scope(conn, owner, index, scope):
        return await conn.fetch(f"SELECT * FROM {owner}.{quote_identifier(note_table(index))} WHERE scope = $1", scope)
"""


def test_a_table_numbered_at_run_time_is_judged_against_its_template(tmp_path: Path) -> None:
    """Flips if a ring table a function names at run time is unknown when its DDL declares the template, or if a
    non-key read of it passes."""
    found = _symbols(_config(tmp_path, {"pkg/ring.py": _RING, "pkg/reads.py": _RING_READS}))
    assert ("by_scope", "non-key-where {}.threetears_change_notes_{}") in found, found
    assert not any(function == "by_id" for function, _ in found), found


_DEDUP = """
    from sqlalchemy import Column, Index, MetaData, String, Table

    METADATA = MetaData()
    ROWS = "turn_dedup_rows"
    turn_dedup_table = Table("turn_dedup", METADATA, Column("key", String, primary_key=True), Column("date_updated", String))
    Index(
        "turn_dedup_date_updated",
        turn_dedup_table.c.date_updated,
        info={
            "purpose": "src/dedup.py::Cap.evict",
            "plan_test": "tests/test_plans.py::test_evict_uses_turn_dedup_date_updated",
        },
    )
    DDL = (f"CREATE TABLE IF NOT EXISTS {ROWS} (id INTEGER PRIMARY KEY CHECK (id = 0), n INTEGER NOT NULL)",)

    class Cap:
        def evict(self, l1, over):
            rows = l1.execute_query(f"SELECT n FROM {ROWS} WHERE id = 0")
            l1.execute_query(
                "DELETE FROM turn_dedup WHERE key IN (SELECT key FROM turn_dedup ORDER BY date_updated ASC LIMIT ?)",
                (over,),
            )
            return rows
"""


def test_a_sqlalchemy_index_carries_its_purpose_in_info_and_f_string_ddl_declares(tmp_path: Path) -> None:
    """Flips if a SQLAlchemy Index's purpose in ``info`` is not read (or its table not found from ``table.c``), if
    DDL written as an f-string declares nothing, or if the purposed eviction page is called whole."""
    _plans(
        tmp_path,
        "def test_evict_uses_turn_dedup_date_updated(plan):\n"
        "    assert plan_uses_index(plan, 'turn_dedup_date_updated')\n",
    )
    findings = no_whole_reads_findings(_config(tmp_path, {"dedup.py": _DEDUP}))
    assert findings == [], findings
    unpurposed = _DEDUP.replace('"purpose": "src/dedup.py::Cap.evict",', "")
    findings = no_whole_reads_findings(_config(tmp_path, {"dedup.py": unpurposed}))
    assert any("in Cap.evict: whole-select turn_dedup" in finding for finding in findings), findings
    assert any("index turn_dedup_date_updated on turn_dedup: index-no-purpose" in finding for finding in findings)


def test_a_migration_renaming_a_key_column_renames_the_key(tmp_path: Path) -> None:
    """Flips if a read by a key column a later migration renamed fails, or one by its old name still passes."""
    modules = {
        "migrations/v001.py": 'UP = "CREATE TABLE namespaces (row_scope TEXT, id UUID, PRIMARY KEY (row_scope, id))"\n',
        "migrations/v009.py": 'UP = "ALTER TABLE namespaces RENAME COLUMN id TO namespace_id"\n',
        "reads.py": """
            async def by_new(conn, scope, namespace_id):
                return await conn.fetch("SELECT * FROM namespaces WHERE row_scope = $1 AND namespace_id = $2", scope, namespace_id)

            async def by_old(conn, scope, namespace_id):
                return await conn.fetch("SELECT * FROM namespaces WHERE row_scope = $1 AND id = $2", scope, namespace_id)
            """,
    }
    found = _symbols(_config(tmp_path, modules))
    assert ("by_old", "non-key-where namespaces") in found, found
    assert not any(function == "by_new" for function, _ in found), found


_COPY_ANY = """
    def by_state(snapshot, name, state):
        return snapshot.backend.execute_query(f"SELECT * FROM {snapshot.relation(name)} WHERE join_state_code = {state}")

    def by_office(snapshot, name, office):
        return snapshot.backend.execute_query(f"SELECT * FROM {snapshot.relation(name)} WHERE office = {office}")
"""


def test_a_copy_relation_named_at_run_time_takes_the_default_copy_key(tmp_path: Path) -> None:
    """Flips if a copy relation named at run time is not judged against the default copy key, or passes on another
    column, or passes with no default declared."""
    keyed = _symbols(_config(tmp_path, {"copy.py": _COPY_ANY}, local_relations={"*": ("join_state_code",)}))
    assert ("by_office", "non-key-where copy.{}") in keyed, keyed
    assert not any(function == "by_state" for function, _ in keyed), keyed
    # the run-time copy token is a relation, never a predicate interpolated after a table: judged, not unknown
    unkeyed = _symbols(_config(tmp_path, {"copy.py": _COPY_ANY}))
    assert ("by_state", "unknown-schema copy.{}") in unkeyed, unkeyed
    assert not any(symbol.startswith("unknown-predicate") for _, symbol in unkeyed), unkeyed


_STAGED = """
    from threetears.core.collections.schema_backed import TableSchema

    RUNS = TableSchema(name="runs", primary_key="run_id", columns=[])

    def staged(conn, run_id, suffix):
        conn.execute("WITH stage0 AS (SELECT * FROM runs WHERE run_id = $1) SELECT 1", run_id)
        return conn.fetch(f"SELECT * FROM stage0{suffix}")

    def unstaged(conn, suffix):
        return conn.fetch(f"SELECT * FROM stage0{suffix}")
"""


def test_a_fragment_reading_a_cte_its_function_defines_is_no_table(tmp_path: Path) -> None:
    """Flips if a fragment reading a CTE its own function writes is unknown, or one with no such CTE passes."""
    found = _symbols(_config(tmp_path, {"staged.py": _STAGED}))
    assert ("unstaged", "unknown-predicate stage0") in found, found
    assert not any(function == "staged" for function, _ in found), found


_BY_STATE = """
    from threetears.core.collections.schema_backed import Index

    class Source:
        def state_index(self):
            return Index(
                f"{self.name}_by_state",
                "state",
                purpose="src/source.py::Source.state_index",
                plan_test="tests/test_plans.py::test_by_state",
            )
"""


def test_a_pattern_index_s_plan_test_names_a_concrete_index_it_makes(tmp_path: Path) -> None:
    """Flips if a pattern-named index's plan test fails for naming a concrete name the pattern makes, or passes
    naming none."""
    _plans(tmp_path, "def test_by_state(plan):\n    assert plan_uses_index(plan, 'latest_results_by_state')\n")
    assert not [f for f in no_whole_reads_findings(_config(tmp_path, {"source.py": _BY_STATE})) if "index-" in f]
    _plans(tmp_path, "def test_by_state(plan):\n    assert plan_uses_index(plan, 'latest_results_geo')\n")
    findings = no_whole_reads_findings(_config(tmp_path, {"source.py": _BY_STATE}))
    assert any("index-plan-test" in finding and "never names an index" in finding for finding in findings), findings


_LISTING = """
    from typing import Final

    from threetears.core.collections.schema_backed import Index, TableSchema

    DATASOURCES = TableSchema(
        name="datasources",
        primary_key="id",
        columns=[],
        indexes=(
            Index("ds_kind_name", "kind", "name", purpose="src/listing.py::list_rows"),
            Index("ds_kind_created", "kind", "date_created", purpose="src/listing.py::list_rows"),
        ),
    )

    _SORT_COLUMNS: Final = {
        Field.NAME: SortColumn(column="name", index="ds_kind_name"),
        Field.CREATED: SortColumn(column="date_created", index="ds_kind_created"),
    }

    async def list_rows(pool, position, limit, spec):
        params = [limit + 1]
        conditions: list[str] = [f"kind = ${len(params) + 1}"]
        if position is not None:
            conditions.append(keyset_condition(spec, position, params))
        where_clause = where_sql(conditions)
        return await pool.fetch(f"SELECT * FROM datasources {where_clause} ORDER BY {order_by_clause(spec)} LIMIT $1")

    async def endpoint(pool, sort):
        spec = _SORT_COLUMNS[sort]
        return await list_rows(pool, None, 50, spec)
"""


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (', index="ds_kind_created"', ""),
        (
            "    where_clause = where_sql(conditions)",
            "    conditions.append(visibility)\n        where_clause = where_sql(conditions)",
        ),
        ('[f"kind = ${len(params) + 1}"]', '[f"status = ${len(params) + 1}"]'),
    ],
)
def test_a_keyset_listing_pages_by_key_only_when_every_sort_names_its_index(tmp_path: Path, old: str, new: str) -> None:
    """Flips if a listing whose every sort column pages through its purposed index fails, or if one sort with no
    index, a condition that is neither the index prefix nor the page's range, or a prefix no condition binds,
    passes."""
    assert _symbols(_config(tmp_path, {"listing.py": _LISTING})) == set()
    found = _symbols(_config(tmp_path, {"listing.py": _LISTING.replace(old, new)}))
    assert any(function == "list_rows" and symbol.startswith("unkeyed-sort datasources.") for function, symbol in found)


_TWO_READERS = """
    from threetears.core.collections.schema_backed import Index, TableSchema

    GRANTS = TableSchema(
        name="grants",
        primary_key="id",
        columns=[],
        indexes=(Index("grants_by_agent", "agent_id", purpose=PURPOSE, plan_test="tests/test_plans.py::test_both"),),
    )

    async def one(conn, agent_id):
        return await conn.fetch("SELECT * FROM grants WHERE agent_id = $1", agent_id)

    async def two(conn, agent_id):
        return await conn.fetch("SELECT id FROM grants WHERE agent_id = $1 AND state = $2", agent_id, "live")

    async def neither(conn, grant_id):
        return await conn.fetch("SELECT * FROM grants WHERE id = $1", grant_id)
"""


def test_a_purpose_may_name_several_reads_each_of_which_uses_the_index(tmp_path: Path) -> None:
    """Flips if an index naming two reads that both use it is refused, or one naming a read that does not use it is
    accepted."""
    _plans(tmp_path, "def test_both(plan):\n    assert plan_uses_index(plan, 'grants_by_agent')\n")
    both = _TWO_READERS.replace("PURPOSE", '"src/grants.py::one; src/grants.py::two"')
    assert not [f for f in no_whole_reads_findings(_config(tmp_path, {"grants.py": both})) if ": index-" in f]
    stray = _TWO_READERS.replace("PURPOSE", '"src/grants.py::one; src/grants.py::neither"')
    findings = no_whole_reads_findings(_config(tmp_path, {"grants.py": stray}))
    assert any("index-purpose-unused -- src/grants.py::neither" in finding for finding in findings), findings
    assert not any("index-purpose-unused -- src/grants.py::one" in finding for finding in findings), findings


def test_a_purpose_may_be_a_tuple_of_the_reads_it_serves(tmp_path: Path) -> None:
    """Flips if purpose=("<path>::<fn>", ...) is not read as the reads it names: both found and using the index
    pass; neither using it is named, each one."""
    _plans(tmp_path, "def test_both(plan):\n    assert plan_uses_index(plan, 'grants_by_agent')\n")
    both = _TWO_READERS.replace("PURPOSE", '("src/grants.py::one", "src/grants.py::two")')
    assert not [f for f in no_whole_reads_findings(_config(tmp_path, {"grants.py": both})) if ": index-" in f]
    neither = _TWO_READERS.replace("PURPOSE", '("src/grants.py::neither", "src/grants.py::absent")')
    findings = no_whole_reads_findings(_config(tmp_path, {"grants.py": neither}))
    assert any("index-purpose-unused -- src/grants.py::neither" in finding for finding in findings), findings
    assert any("index-purpose-unknown -- purpose 'src/grants.py::absent'" in finding for finding in findings), findings


def test_a_second_declaration_of_an_index_carries_the_first_one_s_notes(tmp_path: Path) -> None:
    """Flips if a migration's CREATE INDEX of an index a TableSchema declares with its purpose is refused for having
    none, or if neither declaration names one and it passes."""
    _plans(tmp_path, "def test_both(plan):\n    assert plan_uses_index(plan, 'grants_by_agent')\n")
    noted = _TWO_READERS.replace("PURPOSE", '"src/grants.py::one; src/grants.py::two"')
    migration = 'UP = "CREATE INDEX grants_by_agent ON grants (agent_id)"\n'
    modules = {"grants.py": noted, "migrations/v002.py": migration}
    assert not [f for f in no_whole_reads_findings(_config(tmp_path, modules)) if ": index-" in f]
    bare = {
        "grants.py": _TWO_READERS.replace('purpose=PURPOSE, plan_test="tests/test_plans.py::test_both"', ""),
        "migrations/v002.py": migration,
    }
    assert any("index-no-purpose" in finding for finding in no_whole_reads_findings(_config(tmp_path, bare)))


_FENCE = """
    from threetears.core.operator import security_backstop

    class Fence:
        @security_backstop(
            "a revocation the fence missed would leave a connection open for up to a day",
            proof="tests/test_fence.py::test_reconcile_closes_a_missed_revocation",
            speed="tests/test_fence.py::test_reconcile_within_budget",
        )
        async def reconcile(self, conn):
            return await conn.fetch("SELECT * FROM connections")

        async def loop(self, conn):
            while True:
                await self.reconcile(conn)
"""

_FENCE_TESTS = """
    async def test_reconcile_closes_a_missed_revocation(fence):
        assert await fence.reconcile(conn)

    async def test_reconcile_within_budget(fence):
        took = await best_seconds(fence.reconcile)
        assert not budget_problem("reconcile", took, baseline)
"""


def _fence_tests(tmp_path: Path, body: str = _FENCE_TESTS) -> None:
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_fence.py").write_text(textwrap.dedent(body), encoding="utf-8")


def test_a_security_backstop_with_its_proof_and_speed_test_is_accepted_and_announced(tmp_path: Path) -> None:
    """Flips if a marked backstop's whole pass, or the loop calling it, is a finding, or if its acceptance is not
    announced."""
    _fence_tests(tmp_path)
    with pytest.warns(UserWarning, match="security_backstop src/fence.py"):
        findings = no_whole_reads_findings(_config(tmp_path, {"fence.py": _FENCE}))
    assert findings == [], findings


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("a revocation the fence missed would leave a connection open for up to a day", "a backstop"), "the risk"),
        (("budget_problem(", "print("), "never calls budget_problem"),
        (("test_reconcile_closes_a_missed_revocation(fence)", "test_other(fence)"), "defines no"),
    ],
)
def test_a_security_backstop_needs_its_reason_proof_and_calibrated_speed_test(
    tmp_path: Path, change: tuple[str, str], message: str
) -> None:
    """Flips if a backstop is accepted with a short reason, a speed test held to no calibrated budget, or no proof
    test."""
    old, new = change
    _fence_tests(tmp_path, _FENCE_TESTS.replace(old, new))
    with pytest.warns(UserWarning):
        findings = no_whole_reads_findings(_config(tmp_path, {"fence.py": _FENCE.replace(old, new)}))
    assert any(message in finding for finding in findings), findings


_SOURCE = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.operator import source_change_read

    LOADS = TableSchema(name="loads", primary_key="load_id", columns=[])

    @source_change_read(stamps=("upload_timestamp", "winnerdatetime"), proof="tests/test_source.py::test_changes")
    async def changes(client, relation, upload_mark, winner_mark):
        return await client.query(
            f"SELECT state, count(*) FROM {relation} WHERE CAST(upload_timestamp AS TIMESTAMP) > CAST($1 AS TIMESTAMP) "
            "OR CAST(winnerdatetime AS TIMESTAMP) > CAST($2 AS TIMESTAMP) GROUP BY state",
            upload_mark,
            winner_mark,
        )

    @source_change_read(ids=("run_id",), proof="tests/test_source.py::test_runs")
    async def runs(client, relation, run_ids):
        return await client.query(f"SELECT * FROM {relation} WHERE run_id = ANY($1)", run_ids)

    @source_change_read(unit="table", proof="tests/test_source.py::test_static")
    async def static(client, relation):
        loads = await client.query("SELECT * FROM loads")
        return await client.query(f"SELECT * FROM {relation}"), loads

    async def unmarked(client, relation, upload_mark):
        return await client.query(f"SELECT * FROM {relation} WHERE CAST(upload_timestamp AS TIMESTAMP) > $1", upload_mark)
"""

_SOURCE_TESTS = """
    async def test_changes(client):
        assert await changes(client, "r", 1, 2)

    async def test_runs(client):
        assert await runs(client, "r", [1])

    async def test_static(client):
        assert await static(client, "r")
"""


def test_a_source_change_read_passes_only_its_stamps_ids_or_static_unit_on_an_external_relation(tmp_path: Path) -> None:
    """Flips if a marked read past the source's stamps, of its runs' ids, or of a static input fails, if an unmarked
    or narrowed read passes, or if the mark loosens a declared L3 table."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_source.py").write_text(textwrap.dedent(_SOURCE_TESTS), encoding="utf-8")
    with pytest.warns(UserWarning):
        found = _symbols(_config(tmp_path, {"source.py": _SOURCE}))
    assert ("unmarked", "unknown-predicate {}") in found or ("unmarked", "unknown-schema {}") in found, found
    assert ("static", "whole-select loads") in found, found
    assert not {function for function, _ in found} & {"changes", "runs"}, found
    narrowed = _SOURCE.replace("GROUP BY state", "AND state = $3 GROUP BY state")
    with pytest.warns(UserWarning):
        found = _symbols(_config(tmp_path, {"source.py": narrowed}))
    assert any(function == "changes" for function, _ in found), found


def test_a_listing_is_judged_against_the_vocabulary_handed_to_it_and_its_page_uses_its_index(tmp_path: Path) -> None:
    """Flips if a listing is judged against another endpoint's sort vocabulary in its module, or if a clean page
    does not record its index as used (so the index reports its purpose unused)."""
    other = """
    MODELS = TableSchema(name="models", primary_key="id", columns=[])
    _MODEL_SORTS: Final = {Field.NAME: SortColumn(column="model_name")}

    async def models(pool, sort):
        spec = _MODEL_SORTS[sort]
        return await other_listing(pool, spec)
"""
    module = _LISTING + textwrap.indent(textwrap.dedent(other), "    ")
    _plans(
        tmp_path,
        "def test_pages(plan):\n    assert plan_uses_index(plan, 'ds_kind_name')\n    assert plan_uses_index(plan, 'ds_kind_created')\n",
    )
    noted = module.replace(
        'purpose="src/listing.py::list_rows")',
        'purpose="src/listing.py::list_rows", plan_test="tests/test_plans.py::test_pages")',
    )
    findings = no_whole_reads_findings(_config(tmp_path, {"listing.py": noted}))
    assert not any("unkeyed-sort" in finding for finding in findings), findings
    assert not any("index-purpose-unused" in finding for finding in findings), findings


_FACTORY = """
    from threetears.core.collections.schema_backed import Index, TableSchema

    def by_column(name, column):
        return Index(name, column)

    def epochs_schema(name: str = "scope_epochs"):
        return TableSchema(name=name, primary_key="scope", columns=[])

    class Epochs:
        async def one(self, conn, scope):
            return await conn.fetch(f"SELECT * FROM {self._table} WHERE scope = $1", scope)

        async def stale(self, conn, epoch):
            return await conn.fetch(f"SELECT * FROM {self._table} WHERE epoch < $1", epoch)

    async def named(conn, scope):
        return await conn.fetch("SELECT * FROM scope_epochs WHERE scope = $1", scope)

    LAYOUT = "CREATE TABLE IF NOT EXISTS {table} (id UUID PRIMARY KEY, kind TEXT)"
    DDL = LAYOUT.format(table="kinds")

    async def kind_by_id(conn, kind_id):
        return await conn.fetch("SELECT * FROM kinds WHERE id = $1", kind_id)

    def record():
        return NotedChange(table="t", whole_table=True)

    def consume(follower):
        return follower.read(whole_table=True)
"""


def test_factories_templates_format_ddl_and_whole_records_are_judged_where_they_mean_something(tmp_path: Path) -> None:
    """Flips if an index factory is a declaration of its own, if a template table's key does not key its module's
    generic reads (or its default-named table), if ``.format`` DDL declares nothing, or if building a record that a
    whole table changed is a whole read while asking for one still is."""
    found = _symbols(_config(tmp_path, {"epochs.py": _FACTORY}))
    findings = no_whole_reads_findings(_config(tmp_path, {"epochs.py": _FACTORY}))
    assert not any("index name" in finding for finding in findings), findings
    assert ("Epochs.stale", "non-key-where {}") in found, found
    assert ("consume", "whole-flag whole_table=") in found, found
    assert not {function for function, _ in found} & {"Epochs.one", "named", "kind_by_id", "record"}, found


_FENCE_TEMPLATE = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import delete_keyed, update_keyed

    def fence_schema(name: str = "fence"):
        return TableSchema(
            name=name,
            primary_key="signal",
            columns=[],
            indexes=(
                Index(
                    f"{name}_by_signal_c",
                    "signal",
                    asc="signal",
                    collate={"signal": "C"},
                    purpose=("src/fence.py::clear_signals", "src/fence.py::restamp"),
                    plan_test="tests/test_fence.py::test_plan",
                ),
            ),
        )

    async def clear_signals(conn, table, prefix):
        return await delete_keyed(conn, table, prefix={"signal": prefix})

    async def restamp(conn, table, prefix, run_id):
        return await update_keyed(conn, table, values={"run_id": run_id}, prefix={"signal": prefix})

    async def elsewhere(conn, table, prefix):
        return await delete_keyed(conn, table, prefix={"owner": prefix})
"""

_NO_TEMPLATE = """
    from threetears.core.keyed import delete_keyed

    async def clear_signals(conn, table, prefix):
        return await delete_keyed(conn, table, prefix={"signal": prefix})
"""


def test_a_prefix_write_on_a_table_its_caller_names_is_judged_by_its_module_s_template(tmp_path: Path) -> None:
    """Flips if a prefix delete or update on a table the caller names passes unjudged: keyed through the template's
    C-ordered index (named from the table) whose purpose names the writer, with that purpose found used; a whole
    read on a column the template keys by nothing, and where the module declares no template to judge it by; and
    flagged unused when the purpose names a writer that never reads through the index."""
    findings = no_whole_reads_findings(_config(tmp_path, {"fence.py": _FENCE_TEMPLATE}))
    found = _symbols(_config(tmp_path, {"fence.py": _FENCE_TEMPLATE}))
    assert not {function for function, _ in found} & {"clear_signals", "restamp"}, found
    assert not any("index-purpose-unused" in finding for finding in findings), findings
    assert any(function == "elsewhere" for function, _ in found), found
    bare = _symbols(_config(tmp_path / "bare", {"fence.py": _NO_TEMPLATE}))
    assert any(function == "clear_signals" for function, _ in bare), bare
    unused = _FENCE_TEMPLATE.replace('prefix={"signal": prefix}, returning', "returning").replace(
        'values={"run_id": run_id}, prefix={"signal": prefix}', 'values={"run_id": run_id}, by={"signal": prefix}'
    )
    stale = no_whole_reads_findings(_config(tmp_path / "stale", {"fence.py": unused}))
    assert any("index-purpose-unused -- src/fence.py::restamp" in finding for finding in stale), stale


_HELPER = """
    from threetears.core.collections.schema_backed import TableSchema

    TIMES = TableSchema(name="times", primary_key="race_id", columns=[])

    def _times_sql(race_id):
        return f"SELECT race_id, at FROM times WHERE race_id = {race_id}"

    def page(conn, race_id):
        return conn.fetch(f"WITH t AS ({_times_sql(race_id)}) SELECT * FROM t")

    def derived(conn, race_id):
        return conn.fetch(f"SELECT * FROM ({_times_sql(race_id)}) AS d")
"""


def test_a_helper_s_returned_sql_is_judged_once_where_it_is_written(tmp_path: Path) -> None:
    """Flips if a statement a helper returns is judged again in its caller (as a whole read in the caller's env),
    or if the helper's own read is not judged."""
    found = _symbols(_config(tmp_path, {"helper.py": _HELPER}))
    assert not {function for function, _ in found} & {"page", "derived", "_times_sql"}, found
    unkeyed = _symbols(
        _config(tmp_path, {"helper.py": _HELPER.replace("WHERE race_id = {race_id}", "WHERE at > {race_id}")})
    )
    assert ("_times_sql", "non-key-where times") in unkeyed, unkeyed


_RUNNERS = """
    def from_copy(backend):
        return backend.execute_query("SELECT * FROM geos")

    def asked(snapshot):
        return snapshot.ask("SELECT * FROM geos")
"""


def test_a_head_run_through_a_query_backend_or_a_repo_runner_is_a_whole_read(tmp_path: Path) -> None:
    """Flips if a bare read run through 3tears' ``execute_query`` passes, or one run through a runner the repo names
    in ``extra_exec_calls`` passes (or is named when the repo names no such runner)."""
    assert _symbols(_config(tmp_path, {"runners.py": _RUNNERS})) == {("from_copy", "whole-select geos")}
    named = _symbols(_config(tmp_path, {"runners.py": _RUNNERS}, extra_exec_calls=frozenset({"ask"})))
    assert ("asked", "whole-select geos") in named, named


_CTE_PARAMETER = """
    from threetears.core.collections.schema_backed import TableSchema

    RUNS = TableSchema(name="runs", primary_key="run_id", columns=[])

    def kept_runs_sql(source):
        return f"SELECT run_id FROM {source} WHERE kept"

    def page(conn, run_id):
        return conn.fetch(f"WITH chosen AS (SELECT * FROM runs WHERE run_id = $1), kept AS ({kept_runs_sql('chosen')}) SELECT 1", run_id)
"""


def test_a_parameter_every_caller_fills_with_its_own_cte_is_that_cte(tmp_path: Path) -> None:
    """Flips if a helper reading a CTE its every caller defines is judged a whole read, or if a caller handing it a
    table leaves the helper's read unjudged."""
    assert _symbols(_config(tmp_path, {"runs.py": _CTE_PARAMETER})) == set()
    tabled = _CTE_PARAMETER + "\n    def whole(conn):\n        return conn.fetch(kept_runs_sql('runs'))\n"
    found = _symbols(_config(tmp_path, {"runs.py": tabled}))
    assert any(function == "kept_runs_sql" for function, _ in found), found


_TYPED = """
    from threetears.core.collections.schema_backed import TableSchema

    RESULTS = TableSchema(name="results", primary_key="state", columns=[])

    def load(db, arrow):
        db.register("raw", arrow)
        db.execute("CREATE TEMP VIEW typed AS SELECT CAST(a AS INT) AS a FROM raw")
        db.execute("CREATE TEMP VIEW widened AS SELECT * FROM results")
        db.execute("INSERT INTO results SELECT * FROM typed")
        db.execute("INSERT INTO results SELECT * FROM widened")
"""


def test_a_temp_view_over_only_written_rows_is_those_rows(tmp_path: Path) -> None:
    """Flips if a temp view over registered rows is read as a whole table, or one over a declared table is not."""
    found = _symbols(_config(tmp_path, {"load.py": _TYPED}))
    assert ("load", "whole-select widened") in found, found
    assert not any("typed" in symbol for _, symbol in found), found


_VIEWS = """
    def make(db):
        db.execute("CREATE OR REPLACE VIEW needle AS SELECT office_key, state, votes FROM lsf_rows")
        db.execute("CREATE VIEW renamed AS SELECT upper(office_key) AS office_key, state FROM lsf_rows")
        db.execute("CREATE VIEW loose AS SELECT * FROM lsf_rows")

    def by_office(db, office):
        return db.execute_query(f"SELECT * FROM needle WHERE office_key = {office}")

    def by_votes(db, votes):
        return db.execute_query(f"SELECT * FROM needle WHERE votes > {votes}")
"""


def test_a_view_is_declared_with_keys_its_body_passes_through_and_read_by_them(tmp_path: Path) -> None:
    """Flips if a view's creation is judged as a read, if an undeclared view passes, if a declared key its body
    computes rather than passes passes, or if a read of a declared view is not judged by its keys."""
    relations = {"needle": ("office_key",), "renamed": ("office_key",)}
    found = _symbols(_config(tmp_path, {"views.py": _VIEWS}, local_relations=relations))
    assert ("make", "undeclared-view loose") in found, found
    assert ("make", "view-key-not-passed renamed.office_key") in found, found
    assert ("by_votes", "non-key-where needle") in found, found
    assert not any(function == "by_office" for function, _ in found), found
    assert not any(symbol.endswith(" needle") and function == "make" for function, symbol in found), found


_CONSTRAINT = """
    from threetears.core.collections.schema_backed import Index, TableSchema

    DEFINITIONS = TableSchema(
        name="definitions",
        primary_key="id",
        columns=[],
        indexes=(
            Index(
                "definitions_hash_key",
                "dataset_id",
                "content_hash",
                unique=UNIQUE,
                purpose="constraint",
                plan_test="tests/test_plans.py::test_a_duplicate_definition_is_refused",
            ),
        ),
    )
"""


@pytest.mark.parametrize(
    ("unique", "body", "message"),
    [
        ("True", "with pytest.raises(UniqueViolation):\n        insert('definitions_hash_key')", None),
        (
            "False",
            "with pytest.raises(UniqueViolation):\n        insert('definitions_hash_key')",
            "index-constraint-not-unique",
        ),
        ("True", "assert insert('definitions_hash_key')", "never calls raises"),
    ],
)
def test_a_unique_index_no_read_uses_may_serve_a_proven_constraint(
    tmp_path: Path, unique: str, body: str, message: str | None
) -> None:
    """Flips if a unique index serving only a constraint, proven by a refused duplicate, is refused, or if a
    non-unique one, or one whose test never sees a duplicate refused, is accepted."""
    _plans(tmp_path, f"def test_a_duplicate_definition_is_refused():\n    {body}\n")
    findings = [
        f
        for f in no_whole_reads_findings(_config(tmp_path, {"defs.py": _CONSTRAINT.replace("UNIQUE", unique)}))
        if ": index-" in f
    ]
    if message is None:
        assert findings == [], findings
    else:
        assert any(message in finding for finding in findings), findings


_QUOTED = """
    from threetears.core.collections.schema_backed import TableSchema

    GEOS = "geos"
    GEOS_SCHEMA = TableSchema(name=GEOS, primary_key="geo_id", columns=[])

    def quote_ident(name):
        return '"' + name + '"'

    def one(conn, geo_id):
        return conn.fetch(f"SELECT * FROM main.{quote_ident(GEOS)} WHERE geo_id = $1", geo_id)

    def by_state(conn, state):
        return conn.fetch(f"SELECT * FROM main.{quote_ident(GEOS)} WHERE state = $1", state)
"""


def test_a_quoting_helper_reads_as_the_name_it_quotes(tmp_path: Path) -> None:
    """Flips if a module's own quoting helper renders as its template (``"{}"``) rather than the constant it quotes,
    so a keyed read of the table is unknown, or an unkeyed one is not named against the table."""
    found = _symbols(_config(tmp_path, {"quoted.py": _QUOTED}))
    assert ("by_state", 'non-key-where main."geos"') in found or ("by_state", "non-key-where main.geos") in found, found
    assert not any(function == "one" for function, _ in found), found


_VISIBLE = """
    ROLE_ASSIGNMENTS = TableSchema(name="role_assignments", primary_key=("customer_id", "user_id"), columns=[])
"""


@pytest.mark.parametrize(
    ("clause", "passes"),
    [
        (
            "EXISTS (SELECT 1 FROM role_assignments ra WHERE ra.customer_id = datasources.customer_id AND ra.user_id = $9)",
            True,
        ),
        ("EXISTS (SELECT 1 FROM role_assignments ra WHERE ra.scope_type = 'all')", False),
    ],
)
def test_a_keyed_correlated_exists_filters_each_page_row(tmp_path: Path, clause: str, passes: bool) -> None:
    """Flips if a listing condition that is a correlated EXISTS keyed by the outer row fails the page, or one whose
    inner read is not keyed passes."""
    module = _LISTING + textwrap.indent(textwrap.dedent(_VISIBLE), "    ")
    module = module.replace(
        "    where_clause = where_sql(conditions)",
        f'    conditions.append("{clause}")\n        where_clause = where_sql(conditions)',
    )
    found = _symbols(_config(tmp_path, {"listing.py": module}))
    unkeyed = [symbol for function, symbol in found if function == "list_rows"]
    assert (not unkeyed) if passes else bool(unkeyed), found


_COPY_NULLS = """
    def null_bucket(snapshot):
        return snapshot.backend.execute_query(f"SELECT * FROM {snapshot.relation('lsf_rows')} WHERE office_key IS NULL")

    def null_other(snapshot):
        return snapshot.backend.execute_query(f"SELECT * FROM {snapshot.relation('lsf_rows')} WHERE votes IS NULL")
"""


def test_a_copy_s_declared_key_is_null_is_its_null_bucket(tmp_path: Path) -> None:
    """Flips if a copy read of a declared key column's NULL bucket is called whole, or a NULL test of another column
    passes."""
    found = _symbols(_config(tmp_path, {"nulls.py": _COPY_NULLS}, local_relations={"lsf_rows": ("office_key",)}))
    assert ("null_other", "unbound-where copy.lsf_rows") in found, found
    assert not any(function == "null_bucket" for function, _ in found), found


def test_a_source_gap_names_its_gap_and_is_announced(tmp_path: Path) -> None:
    """Flips if a whole read kept for an upstream source's missing change record is accepted without its gap id and
    condition, or accepted silently."""
    _proof(tmp_path)
    reason = "G7: the warehouse's LSF rows changed under a held run, and no change record says which"
    line = _ledger(kind="source-gap").replace(
        "the first start of an empty copy, with no checkpoint in L2 to load", reason
    )
    with pytest.warns(UserWarning, match="source-gap src/follower.py :: Follower.warm: G7"):
        assert no_whole_reads_findings(_config(tmp_path, {"follower.py": _COLD}, ledger=line)) == []
    unnamed = _ledger(kind="source-gap")
    findings = no_whole_reads_findings(_config(tmp_path, {"follower.py": _COLD}, ledger=unnamed))
    assert any("names its gap id" in finding for finding in findings), findings


_STATIC = """
    from threetears.core.operator import source_change_read

    class Warehouse:
        @source_change_read(unit="table", proof="tests/test_source.py::test_static")
        async def static_records(self, relation):
            rows = await read_all(self._client, f"SELECT * FROM {relation}")
            heads = await self._client.query(f"SELECT name FROM {relation}")
            return rows, heads

        async def load(self, relation):
            return await self.static_records(relation)

    @source_change_read(ids=("run_id",), proof="tests/test_source.py::test_runs")
    async def grouped_by(client, relation, run_ids, required):
        condition, params = filter_conditions(where_in={"run_id": run_ids, "family": ["race"]}, not_null=required)
        return await client.query(f"SELECT office_key FROM {relation} WHERE {condition} GROUP BY office_key", params)

    @source_change_read(ids=("run_id",), proof="tests/test_source.py::test_runs")
    async def filtered_unknown(client, relation, filters):
        condition, params = filter_conditions(where_in=filters)
        return await client.query(f"SELECT office_key FROM {relation} WHERE {condition}", params)

    def typed(connection, rows, typed_columns):
        connection.register("raw", rows)
        connection.execute(f"CREATE TEMP VIEW typed AS SELECT {typed_columns}, raw.{ORDER} FROM raw")
        return connection.execute("SELECT * FROM typed")
"""

_STATIC_TESTS = """
    async def test_static(warehouse):
        assert await warehouse.static_records("r")

    async def test_runs(client):
        assert await grouped_by(client, "r", [1], ()) and await filtered_unknown(client, "r", {})
"""


def test_a_static_input_s_pager_its_callers_filter_conditions_and_temp_views_are_judged_right(tmp_path: Path) -> None:
    """Flips if a static input read whole (a pager, a bare head) inside its mark, or its caller, is a finding; if a
    read by ids built through filter_conditions is unknown; if one whose columns are not written out passes; or if a
    temp view over registered rows, built in an f-string, is undeclared or its reads are named."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_source.py").write_text(textwrap.dedent(_STATIC_TESTS), encoding="utf-8")
    with pytest.warns(UserWarning):
        found = _symbols(_config(tmp_path, {"warehouse.py": _STATIC}))
    assert not {function for function, _ in found} & {
        "Warehouse.static_records",
        "Warehouse.load",
        "grouped_by",
        "typed",
    }, found
    assert any(function == "filtered_unknown" for function, _ in found), found


_CURSORED = """
    from threetears.core.keyset import keyset_arrow_pages

    async def from_mark(l3, mark):
        return keyset_arrow_pages(l3, "SELECT k FROM t", ("k",), types={}, after=mark)

    async def keyed_from_mark(l3, mark, scope):
        return keyset_arrow_pages(l3, "SELECT k FROM t", ("k",), types={}, after=mark, where={"scope": scope})
"""


def test_a_pager_s_start_alone_does_not_narrow_it(tmp_path: Path) -> None:
    """Flips if a pager given only where to start (after=, since=) passes as narrowed, or one given a where beside it
    is named."""
    found = _symbols(_config(tmp_path, {"cursored.py": _CURSORED}))
    assert ("from_mark", "unkeyed-pager keyset_arrow_pages") in found, found
    assert not any(function == "keyed_from_mark" for function, _ in found), found


_SPREAD = """
    from threetears.core.operator import source_change_read

    @source_change_read(ids=("run_id",), proof="tests/test_source.py::test_runs")
    async def by_runs(client, relation, run_ids, part):
        condition, params = filter_conditions(where_in={"run_id": run_ids, **part.filters})
        return await client.query(f"SELECT office_key FROM {relation} WHERE {condition}", params)

    @source_change_read(ids=("run_id",), proof="tests/test_source.py::test_runs")
    async def by_family(client, relation, part):
        condition, params = filter_conditions(where_in={"family": ["race"], **part.filters})
        return await client.query(f"SELECT office_key FROM {relation} WHERE {condition}", params)
"""


def test_sets_spread_beside_a_bound_id_only_narrow(tmp_path: Path) -> None:
    """Flips if where_in sets spread in beside the bound id make the read unknown, or if spread sets beside no
    bound id or key pass."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_source.py").write_text(
        "async def test_runs(client):\n    assert await by_runs(client, 'r', [1], None) and await by_family(client, 'r', None)\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning):
        found = _symbols(_config(tmp_path, {"spread.py": _SPREAD}))
    assert not any(function == "by_runs" for function, _ in found), found
    assert ("by_family", "unknown-predicate {}") in found, found


_HELPED = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import KeyBatch, Sort, keyset_page, read_keyed

    RESULTS = TableSchema(name="results", primary_key=("state", "geo", "id"), columns=[])

    async def by_key(c, ks, cols):
        return await read_keyed(c, "results", keys=KeyBatch(("state", "geo"), ks), columns=cols)

    async def by_kind(c, cols):
        return await keyset_page(c, "results", columns=cols, by={}, order=[Sort("kind")], limit=5)

    async def unread(c, table, ks, cols):
        return await read_keyed(c, table, keys=KeyBatch(("state",), ks), columns=cols)
"""


def test_a_keyed_helper_is_judged_by_the_statement_it_sends(tmp_path: Path) -> None:
    """Flips if a keyed helper call naming a key prefix is named, one ordering by no key passes, or one whose table
    the walk cannot read passes rather than being unknown."""
    found = _symbols(_config(tmp_path, {"helped.py": _HELPED}))
    assert not any(function == "by_key" for function, _ in found), found
    assert any(function == "by_kind" for function, _ in found), found
    assert any(function == "unread" and symbol.startswith("unknown-predicate read_keyed") for function, symbol in found)


_CONSTANT_TABLE = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import delete_before

    LINKS = "link_redemptions"
    LINKS_SCHEMA = TableSchema(
        name=LINKS,
        primary_key="token_id",
        columns=[],
        indexes=(Index("links_by_expiry", "expires_at", purpose="src/links.py::prune"),),
    )

    async def prune(conn, cutoff):
        return await delete_before(LINKS, collection=None, conn=conn, key=["token_id"], column="expires_at", before=cutoff, page=1000)

    async def prune_any(conn, table, cutoff):
        return await delete_before(table, collection=None, conn=conn, key=["token_id"], column="expires_at", before=cutoff, page=1000)
"""


def test_a_helper_s_constant_argument_reads_as_its_literal(tmp_path: Path) -> None:
    """Flips if a helper whose table is a module constant is unknown rather than judged, or one whose table is a
    parameter the walk cannot read passes."""
    found = _symbols(_config(tmp_path, {"links.py": _CONSTANT_TABLE}))
    assert not any(function == "prune" for function, _ in found), found
    assert any(function == "prune_any" and "unknown-predicate delete_before" in s for function, s in found), found


_DRAIN = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyset import keyset_pages

    RUNS = TableSchema(
        name="dataset_runs",
        primary_key="id",
        columns=[],
        indexes=(
            Index(
                "dataset_runs_live",
                "status",
                "date_last_progress",
                "id",
                purpose="src/runs.py::live_run_ids",
                plan_test="tests/test_plans.py::test_live",
            ),
        ),
    )

    async def live_run_ids(pool, status, cutoff):
        async for page in keyset_pages(
            pool,
            "SELECT id, date_last_progress FROM dataset_runs",
            ["date_last_progress", "id"],
            where={"status": status},
            conditions=[Fragment('"date_last_progress" < $1', (cutoff,))],
            page_size=10,
        ):
            return page
"""


def test_a_keyset_pager_s_purposed_index_is_recorded_as_used(tmp_path: Path) -> None:
    """Flips if a drain through keyset_pages does not record the purposed index its pages read through (so the index
    reports its purpose unused), or records one whose purpose names another function."""
    _plans(tmp_path, "def test_live(plan):\n    assert plan_uses_index(plan, 'dataset_runs_live')\n")
    findings = no_whole_reads_findings(_config(tmp_path, {"runs.py": _DRAIN}))
    assert not any("index-purpose-unused" in finding for finding in findings), findings
    other = _DRAIN.replace('purpose="src/runs.py::live_run_ids"', 'purpose="src/runs.py::other"').replace(
        "async def live_run_ids", "async def other(pool):\n        return pool\n\n    async def live_run_ids"
    )
    findings = no_whole_reads_findings(_config(tmp_path, {"runs.py": other}))
    assert any("index-purpose-unused" in finding for finding in findings), findings


_MARKED_COLUMNS = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import Sort, keyset_page

    CONSOLIDATIONS = TableSchema(
        name="memory_consolidations",
        primary_key=("agent_id", "consolidated_memory_id", "source_memory_id"),
        columns=[],
        indexes=(Index("by_source", "agent_id", "source_memory_id", purpose="src/memory.py::_MARK_COLUMNS"),),
    )
    OPERATIONS = TableSchema(
        name="backup_operations",
        primary_key="operation_id",
        columns=[],
        indexes=(Index("recent", "date_started", purpose="src/memory.py::newest"),),
    )
    _MARK_COLUMNS = (
        "(SELECT c.consolidated_memory_id FROM memory_consolidations c "
        "WHERE c.agent_id = memories.agent_id AND c.source_memory_id = memories.memory_id LIMIT 1) AS corrected_by"
    )
    _NEWEST_FIRST = (Sort("date_started", descending=True), Sort("operation_id", descending=True))

    async def by_ids(pool, agent_id, ids):
        return await pool.fetch(f"SELECT memory_id, {_MARK_COLUMNS} FROM memories WHERE agent_id = $1 AND memory_id = ANY($2)", agent_id, ids)

    async def for_user(pool, agent_id, user_id):
        return await pool.fetch(f"SELECT memory_id, {_MARK_COLUMNS} FROM memories WHERE agent_id = $1 AND user_id = $2", agent_id, user_id)

    async def newest(pool, cols):
        return await keyset_page(pool, "backup_operations", columns=cols, by={}, order=_NEWEST_FIRST, limit=10)
"""


def test_a_read_in_a_shared_constant_and_a_helper_s_constant_order_are_judged_and_credited(tmp_path: Path) -> None:
    """Flips if a subquery written once in a module constant can't key through the index whose purpose names that
    constant, or if a helper's order given as a module constant is unread (its purposed index unused)."""
    found = _symbols(_config(tmp_path, {"memory.py": _MARKED_COLUMNS}))
    assert not any("memory_consolidations" in symbol for _, symbol in found), found
    assert not any(function == "newest" for function, _ in found), found
    findings = no_whole_reads_findings(_config(tmp_path, {"memory.py": _MARKED_COLUMNS}))
    assert not any("index-purpose-unused" in finding for finding in findings), findings


_ORDERS_BY_FIELD = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import Sort, keyset_page

    DATASOURCES = TableSchema(
        name="datasources",
        primary_key="id",
        columns=[],
        indexes=(
            Index("by_name", "name", purpose=("src/listing.py::page_of", "src/listing.py::page_by")),
            Index("by_created", "created_at", purpose=("src/listing.py::page_of", "src/listing.py::page_by")),
        ),
    )
    DATASOURCE_ORDERS = {
        "name": [Sort("name"), Sort("id")],
        "created": (Sort("created_at", descending=True), Sort("id", descending=True)),
    }

    async def page_of(pool, field, cols):
        order = DATASOURCE_ORDERS[field]
        return await keyset_page(pool, "datasources", columns=cols, by={}, order=order, limit=50)

    async def page_by(pool, field, cols):
        return await keyset_page(pool, "datasources", columns=cols, by={}, order=DATASOURCE_ORDERS[field], limit=50)
"""


def test_a_helper_s_order_picked_from_a_module_mapping_is_judged_entry_by_entry(tmp_path: Path) -> None:
    """Flips if a keyset page whose order is one entry of a module mapping of literal sort lists
    (``order=DATASOURCE_ORDERS[field]``, directly or through a local) is unread, if any entry escapes judgement
    against its index, or if the indexes those entries page through are not credited to the calling functions."""
    found = _symbols(_config(tmp_path, {"listing.py": _ORDERS_BY_FIELD}))
    assert not any(function in {"page_of", "page_by"} for function, _ in found), found
    findings = no_whole_reads_findings(_config(tmp_path, {"listing.py": _ORDERS_BY_FIELD}))
    assert not any("index-purpose-unused" in finding for finding in findings), findings
    unindexed = _ORDERS_BY_FIELD.replace(
        '"name": [Sort("name"), Sort("id")],',
        '"name": [Sort("name"), Sort("id")],\n        "kind": [Sort("kind"), Sort("id")],',
    )
    found = _symbols(_config(tmp_path, {"listing.py": unindexed}))
    flagged = {function for function, symbol in found if "datasources" in symbol}
    assert flagged == {"page_of", "page_by"}, found


_ORDERS_AFTER_A_PIN = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import Sort, keyset_page

    DATASOURCES = TableSchema(
        name="datasources",
        primary_key="id",
        columns=[],
        indexes=(
            Index("page_by_name", "kind", "name", "id", purpose="src/listing.py::page_of"),
            Index("page_by_status", "kind", "status", "id", purpose="src/listing.py::page_of"),
        ),
    )
    DATASOURCE_ORDERS = {
        "name": [Sort("name"), Sort("id")],
        "status": [Sort("status"), Sort("id")],
    }

    async def page_of(pool, kind, field, cols):
        return await keyset_page(pool, "datasources", columns=cols, by={"kind": kind}, order=DATASOURCE_ORDERS[field], limit=50)
"""


def test_each_mapped_order_after_a_pinned_prefix_credits_its_own_index(tmp_path: Path) -> None:
    """Flips if the entries of a mapped order after a pinned ``by=`` column all credit the first index the pin leads
    (``page_by_name``) rather than each the index of the pin followed by its own sort (``page_by_status``)."""
    found = _symbols(_config(tmp_path, {"listing.py": _ORDERS_AFTER_A_PIN}))
    assert not any(function == "page_of" for function, _ in found), found
    findings = no_whole_reads_findings(_config(tmp_path, {"listing.py": _ORDERS_AFTER_A_PIN}))
    assert not any("index-purpose-unused" in finding for finding in findings), findings


_PRODUCED = """
    class Stages:
        def __init__(self, db):
            self._db = db

        @staticmethod
        def _stage(stage):
            return f'"__s_{stage}"'

        def rows(self, stage):
            return self._db.execute(f"SELECT * FROM {self._stage(stage)}")


    class _Work:
        def __init__(self, conn):
            self._conn = conn

        def put(self, rows):
            return "w"


    class Other:
        def put(self, rows):
            return "o"


    def settle(conn, rows):
        work = _Work(conn)
        named = work.put(rows)
        return conn.execute(f"SELECT * FROM {named}")


    def widen(work: _Work, conn, rows):
        return conn.execute(f"SELECT * FROM {work.put(rows)}")


    class Pass:
        def __init__(self, work: _Work):
            self.work = work
            self.new = work.put(None)

        def read(self, conn):
            return conn.execute(f"SELECT * FROM {self.new} AS n JOIN {self.work.put(None)} AS k ON n.a = k.a")


    def unrelated(other: Other, conn):
        return conn.execute(f"SELECT * FROM {other.put(1)}")
"""


def test_a_relation_a_declared_transient_producer_returns_is_a_change_set(tmp_path: Path) -> None:
    """Flips if a relation named by a declared transient producer's return (``Stages._stage``, ``_Work.put``) -- the
    call in the placeholder, a name bound to it, or a ``self`` attribute assigned it -- is judged as a data table, if
    an undeclared producer's (``Other.put``, the same method name on another class) is not, or if undeclared the same
    reads are not judged at all."""
    modules = {"pkg/produced.py": _PRODUCED}
    readers = {"Stages.rows", "settle", "widen", "Pass.read"}
    bare = _symbols(_config(tmp_path, modules))
    assert readers | {"unrelated"} <= {function for function, _ in bare}, bare
    found = _symbols(_config(tmp_path, modules, transient_producers=frozenset({"Stages._stage", "_Work.put"})))
    assert not readers & {function for function, _ in found}, found
    assert "unrelated" in {function for function, _ in found}, found


_TABLE_PARAMETER = """
    from threetears.core.keyed import holds_any

    from pkg.tables import GEOS


    async def _held(conn, table, geo):
        return await holds_any(conn, table, order=("geo_id",), by={"geo_id": geo})


    async def _by_state(conn, table, state):
        return await holds_any(conn, table, order=("state",), by={"state": state})


    async def _loose(conn, table, geo):
        return await holds_any(conn, table, order=("geo_id",), by={"geo_id": geo})


    async def callers(conn, named):
        await _held(conn, GEOS, "g1")
        await _held(conn, table="geos", geo="g2")
        await _by_state(conn, GEOS, "s")
        await _loose(conn, GEOS, "g3")
        await _loose(conn, named, "g4")
"""


def test_a_keyed_helpers_table_parameter_resolves_when_every_caller_names_a_declared_table(tmp_path: Path) -> None:
    """Flips if a keyed helper called with a table parameter stays unknown when every caller of its function passes a
    literal or a module constant (``_held``: keyed; ``_by_state``: judged against that table, so non-key), or
    resolves when one caller passes a value the walk cannot read (``_loose``)."""
    found = _symbols(_config(tmp_path, {"pkg/tables.py": _TABLES, "pkg/probe.py": _TABLE_PARAMETER}))
    assert not any(function == "_held" for function, _ in found), found
    assert ("_by_state", 'non-key-where "geos"') in found, found
    assert ("_loose", "unknown-predicate holds_any(table)") in found, found


_RENAMED_VIEWS = """
    def make(db):
        db.execute("CREATE VIEW offices AS SELECT p.race AS office_key, p.state FROM lsf_rows AS p")
        db.execute("CREATE VIEW cased AS SELECT CASE WHEN p.race IS NULL THEN '' ELSE p.race END AS office_key FROM lsf_rows AS p")
        db.execute("CREATE VIEW joined AS SELECT p.race || p.state AS office_key FROM lsf_rows AS p")

    def by_office(db, office):
        return db.execute_query(f"SELECT * FROM offices WHERE office_key = {office}")
"""


def test_a_view_column_that_renames_a_plain_column_passes_its_key_through(tmp_path: Path) -> None:
    """Flips if a key column a view's body renames from a plain column (``p.race AS office_key``) is not passed
    through, or if one it computes (a CASE, a concatenation) is."""
    relations = {"offices": ("office_key",), "cased": ("office_key",), "joined": ("office_key",)}
    found = _symbols(_config(tmp_path, {"views.py": _RENAMED_VIEWS}, local_relations=relations))
    assert ("make", "view-key-not-passed cased.office_key") in found, found
    assert ("make", "view-key-not-passed joined.office_key") in found, found
    assert not any("offices" in symbol for _, symbol in found), found
    assert not any(function == "by_office" for function, _ in found), found


_QUALIFIED_VIEWS = """
    from threetears.core.collections.schema_backed import TableSchema

    SETTINGS = TableSchema(name="report_settings", primary_key="join_state_code", columns=[])

    def make(db):
        db.execute("CREATE OR REPLACE VIEW report.report_settings AS SELECT name, value FROM copy_settings")

    def by_name(db, name):
        return db.execute_query(f"SELECT value FROM report.report_settings WHERE name = {name}")

    def by_state(db, state):
        return db.execute_query(f"SELECT value FROM report.report_settings WHERE join_state_code = {state}")

    async def l3_by_state(conn, state):
        return await conn.fetch("SELECT * FROM report_settings WHERE join_state_code = $1", state)
"""


def test_a_view_in_a_schema_of_its_own_is_judged_by_its_copy_declaration_before_an_l3_table(tmp_path: Path) -> None:
    """Flips if a view named in a schema of its own (``report.<v>``) is judged by the L3 table sharing its bare name,
    rather than by the copy relation declared for it, or if the L3 table loses its own key."""
    found = _symbols(_config(tmp_path, {"views.py": _QUALIFIED_VIEWS}, local_relations={"report_settings": ("name",)}))
    assert not any(function in ("make", "by_name", "l3_by_state") for function, _ in found), found
    assert ("by_state", "non-key-where report.report_settings") in found, found


_VIEW_PRODUCERS = """
    def _view_sql(view):
        return f"CREATE OR REPLACE VIEW {view} AS SELECT p.race AS office_key, p.state FROM lsf_rows AS p"

    def _bad_view_sql(view):
        return f"CREATE VIEW {view} AS SELECT upper(office_key) AS office_key FROM lsf_rows"

    class Parts:
        def copy_view(self, view):
            return f"CREATE VIEW {view} AS SELECT office_key, votes FROM lsf_rows"

    def _stray_view_sql(view):
        return f"CREATE VIEW {view} AS SELECT office_key FROM lsf_rows"

    def newest(db):
        db.execute("CREATE OR REPLACE VIEW report.newest AS SELECT max(model) AS model FROM lsf_rows")
"""


def test_a_view_named_at_run_time_is_judged_by_the_keys_its_producer_declares(tmp_path: Path) -> None:
    """Flips if a view a declared producer builds under a name it is handed is not judged by the producer's declared
    keys (passed through, or computed), or if a producer nobody declares builds an undeclared view unseen."""
    producers = {"_view_sql": ("office_key",), "_bad_view_sql": ("office_key",), "Parts.copy_view": ("office_key",)}
    found = _symbols(
        _config(tmp_path, {"views.py": _VIEW_PRODUCERS}, view_producers=producers, local_relations={"newest": ()})
    )
    # a copy relation declared with no key (one row) is declared all the same
    assert not any(function in ("_view_sql", "Parts.copy_view", "newest") for function, _ in found), found
    assert any(function == "_bad_view_sql" and "view-key-not-passed" in symbol for function, symbol in found), found
    assert any(function == "_stray_view_sql" and "undeclared-view" in symbol for function, symbol in found), found


_FORMAT_VIEWS = """
    def _around_view_sql(view, columns):
        before, after = _around(columns, "l.state")
        return f"CREATE OR REPLACE VIEW {view} AS SELECT {before}l.state{after} FROM main.{view} AS l"

    def _around_two_sql(view, columns):
        before, middle, after = _around(columns, "state", "county")
        return f"CREATE OR REPLACE VIEW {view} AS SELECT {before}state{middle}county{after} FROM main.{view}"

    def _around_short_sql(view, columns):
        before, after = _around(columns, "state")
        return f"CREATE OR REPLACE VIEW {view} AS SELECT {before}state{after} FROM main.{view}"

    def _union_view_sql(view, held, columns):
        before, after = _around(columns, "t.state")
        union = " UNION ALL ".join(f"SELECT {before}t.state{after} FROM main.{name} AS t" for name in held)
        return f"CREATE OR REPLACE VIEW {view} AS {union}"

    def _union_dropped_sql(view, held, columns):
        union = " UNION ALL ".join(f"SELECT upper(t.state) AS state FROM main.{name} AS t" for name in held)
        return f"CREATE OR REPLACE VIEW {view} AS {union}"

    def _branch_view_sql(view, held, raced, columns):
        if raced:
            first, second, rest = _around(columns, "t.state", "t.race")
            select = f"{first}t.state{second}t.race AS race{rest}"
        else:
            first, rest = _around(columns, "t.state")
            select = f"{first}t.state{rest}"
        union = " UNION ALL ".join(f"SELECT {select} FROM main.{name} AS t" for name in held)
        return f"CREATE OR REPLACE VIEW {view} AS {union}"

    def _branch_dropped_sql(view, held, raced, columns):
        if raced:
            select = "t.state, t.race AS race"
        else:
            select = "upper(t.state) AS state"
        union = " UNION ALL ".join(f"SELECT {select} FROM main.{name} AS t" for name in held)
        return f"CREATE OR REPLACE VIEW {view} AS {union}"

    def _keyless_view_sql(view, table):
        return f"CREATE OR REPLACE VIEW {view} AS SELECT * EXCLUDE (state) FROM main.{table}"

    def _keyed_exclude_sql(view, table):
        return f"CREATE OR REPLACE VIEW {view} AS SELECT * EXCLUDE (state) FROM main.{table}"
"""


def test_a_view_written_around_its_key_and_a_keyless_producers_view_are_judged_by_their_declarations(
    tmp_path: Path,
) -> None:
    """Flips if a view whose select list is interpolations around its key written literally (``{before}l.state{after}``,
    a body joined from one statement per table, a select list each branch of an if/else binds) is not read as passing
    that key (any one of a producer's keys), if a key column no literal text names or a branch dropping the key stops
    being a finding, or if a producer declared keyless loses to the default copy key its run-time name matches
    (``SELECT * EXCLUDE (state)``) while a keyed producer's view excluding its key stays a finding."""
    producers = {
        "_around_view_sql": ("state",),
        "_around_two_sql": ("state", "county"),
        "_around_short_sql": ("state", "county"),
        "_union_view_sql": ("state",),
        "_union_dropped_sql": ("state",),
        "_branch_view_sql": (("state", "race"), ("state",)),
        "_branch_dropped_sql": (("state",),),
        "_keyless_view_sql": (),
        "_keyed_exclude_sql": ("state",),
    }
    found = _symbols(
        _config(tmp_path, {"views.py": _FORMAT_VIEWS}, view_producers=producers, local_relations={"*": ("state",)})
    )
    passing = {"_around_view_sql", "_around_two_sql", "_union_view_sql", "_branch_view_sql", "_keyless_view_sql"}
    assert not any(function in passing for function, _ in found), found
    assert any(function == "_around_short_sql" and symbol.endswith(".county") for function, symbol in found), found
    assert any(function == "_union_dropped_sql" and "view-key-not-passed" in symbol for function, symbol in found), (
        found
    )
    assert any(function == "_keyed_exclude_sql" and "view-key-not-passed" in symbol for function, symbol in found), (
        found
    )
    # a name each branch of an if/else binds is judged as each branch writes it: the branch dropping the key fails
    assert any(function == "_branch_dropped_sql" and "view-key-not-passed" in symbol for function, symbol in found), (
        found
    )


_DECLARED_TABLES = """
    from dataclasses import dataclass

    from threetears.core.collections.schema_backed import TableSchema

    COUNTIES = "counties"
    PRECINCTS = "precincts"
    COUNTIES_SCHEMA = TableSchema(name=COUNTIES, primary_key=("state", "county"), columns=[])
    PRECINCTS_SCHEMA = TableSchema(name=PRECINCTS, primary_key=("state", "precinct"), columns=[])
    RUNS_SCHEMA = TableSchema(name="runs", primary_key="run_id", columns=[])


    @dataclass(frozen=True)
    class Declared:
        name: str
        per_run: bool = False


    TABLES = (Declared(name=COUNTIES), Declared(name=PRECINCTS, per_run=True))
    WITH_RUNS = (Declared(name=COUNTIES), Declared(name="runs"))
    NAMES = (COUNTIES, PRECINCTS)
"""

_LOOPED = """
    from threetears.core.keyed import holds_any

    from pkg.tables import NAMES, TABLES, WITH_RUNS


    class Store:
        async def holds_state(self, table, state):
            return await holds_any(self._conn, table, order=("state",), by={"state": state})

        async def holds_run(self, table, state):
            return await holds_any(self._conn, table, order=("state",), by={"state": state})


    class Pull:
        async def _held(self, table, state):
            return await self._store.holds_state(table.name, state)

        async def _write(self, table, state):
            return await self._held(table, state)

        async def run(self, state):
            for table in TABLES:
                await self._write(table, state)

        async def by_name(self, state):
            for name in NAMES:
                await holds_any(self._conn, name, order=("state",), by={"state": state})

        async def with_runs(self, state):
            for table in WITH_RUNS:
                await self._store.holds_run(table.name, state)
"""


def test_a_table_bound_by_a_loop_over_a_declared_tuple_resolves_through_its_callers(tmp_path: Path) -> None:
    """Flips if a keyed helper's table, named by a loop over a module tuple of declared tables (``for t in TABLES``,
    ``t.name``) and handed down through parameters (``run`` -> ``_write`` -> ``_held`` -> ``Store.holds_state``),
    stays unknown, or if one element whose key the read does not bind (``runs``) passes."""
    found = _symbols(_config(tmp_path, {"pkg/tables.py": _DECLARED_TABLES, "pkg/pull.py": _LOOPED}))
    assert not any(function in ("Store.holds_state", "Pull._held", "Pull.by_name") for function, _ in found), found
    assert any(function == "Store.holds_run" and "runs" in symbol for function, symbol in found), found


_SPLICED_TABLES = """
    from dataclasses import dataclass

    from threetears.core.collections.schema_backed import TableSchema

    COUNTIES_SCHEMA = TableSchema(name="counties", primary_key=("state", "county"), columns=[])
    FEED_SCHEMA = TableSchema(name="feed", primary_key=("state", "race"), columns=[])


    @dataclass(frozen=True)
    class Declared:
        name: str


    FEEDS = (Declared(name="feed"),)
    TABLES = (Declared(name="counties"), *FEEDS)
    STATE = "state"
"""

_SPLICED = """
    from typing import Protocol

    from threetears.core.keyed import holds_any

    from pkg.tables import TABLES

    BY_NAME = {table.name: table for table in TABLES}


    class Holding(Protocol):
        async def holds_state(self, table: str, state: str) -> bool:
            'whether the table holds the state.'
            ...


    class Store:
        async def holds_state(self, table, state):
            return await holds_any(self._conn, table, order=("state",), by={"state": state})


    class Pull:
        async def _held(self, table, state, *, pulled=None):
            return await self._store.holds_state(table.name, state)

        async def run(self, state, behind, fresh):
            for table in () if behind else TABLES:
                await self._held(table, state, **({"pulled": {}} if fresh else {}))

        async def one(self, name, state):
            await self._held(BY_NAME[name], state)
"""


def test_a_loop_s_tables_resolve_through_a_spliced_tuple_a_conditional_a_lookup_and_a_spread_call(
    tmp_path: Path,
) -> None:
    """Flips if a keyed helper's table stays unknown when the tuple splices another (``*FEEDS``), the loop iterates a
    conditional (``() if behind else TABLES``), a caller looks one up in a dict built over the tuple
    (``BY_NAME[name]``) or spreads keywords (``**{...}``) beside the table it names, or the helper's protocol declares
    the same method (a stub, not a second definition)."""
    found = _symbols(_config(tmp_path, {"pkg/tables.py": _SPLICED_TABLES, "pkg/pull.py": _SPLICED}))
    assert not any(function in ("Store.holds_state", "Pull._held") for function, _ in found), found


_SAME_NAMED = """
    from threetears.core import keyed

    from pkg.tables import STATE, TABLES


    class Store:
        async def holds_any(self, table, state):
            return await keyed.holds_any(self._conn, table, order=(STATE,), by={STATE: state})


    class Pull:
        async def _held(self, table, state):
            return await self._store.holds_any(table.name, state)

        async def run(self, state):
            for table in TABLES:
                await self._held(table, state)
"""


def test_a_repo_method_named_as_a_keyed_helper_is_the_repo_s_and_its_body_is_judged(tmp_path: Path) -> None:
    """Flips if a call of the repo's own method named as a 3tears keyed helper (``self._store.holds_any``) is judged
    as the helper's call, or if the helper's own call in its body (``keyed.holds_any``) counts as a call of the
    method, so its table never resolves, or a key its ``by`` names by an imported constant stays unread."""
    found = _symbols(_config(tmp_path, {"pkg/tables.py": _SPLICED_TABLES, "pkg/pull.py": _SAME_NAMED}))
    assert not any(function in ("Store.holds_any", "Pull._held") for function, _ in found), found


_PRUNED = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import delete_before, read_after

    LINKS = TableSchema(
        name="links",
        primary_key="token_id",
        columns=[],
        indexes=(Index("links_by_expiry", "expires_at"ASC, purpose="src/links.py::prune"),),
    )
    RUNS = TableSchema(
        name="runs",
        primary_key=("job", "run_id"),
        columns=[],
        indexes=(Index("runs_by_job_start", "job", "started", purpose="src/links.py::after"),),
    )

    async def prune(conn, cutoff):
        await delete_before("links", collection=None, conn=conn, key=["token_id"], column="expires_at", before=cutoff)

    async def after(conn, job, started):
        return await read_after(conn, "runs", columns=("job", "started"), id_column="started", after=started, limit=50, by={"job": job})
"""

_PRUNED_DDL = """
    PRUNED = '''
    -- purpose: src/links.py::prune
    CREATE INDEX links_by_expiry ON links (expires_atASC);
    '''

    LINKS = "CREATE TABLE links (token_id TEXT PRIMARY KEY, expires_at TIMESTAMP)"
"""


@pytest.mark.parametrize(
    ("declared", "declared_ddl"), [("", ""), (', asc="expires_at"', " ASC")], ids=["hashed", "ranged"]
)
def test_an_index_a_read_ranges_or_orders_from_its_first_column_is_declared_range_sharded(
    tmp_path: Path, declared: str, declared_ddl: str
) -> None:
    """Flips if an index whose purposed read ranges or orders by its first column, nothing bound before it
    (``delete_before`` on ``expires_at``), is accepted undeclared ``ASC`` (hashed on YugabyteDB, the read scans the
    table), or refused once declared (``asc=``, or ``ASC`` in its ``CREATE INDEX``); or if one led by a bound column
    (``runs_by_job_start``) is held to it."""
    module = _PRUNED.replace('"expires_at"ASC', f'"expires_at"{declared}')
    found = [f for f in no_whole_reads_findings(_config(tmp_path, {"links.py": module})) if "index-hashed-range" in f]
    assert bool(found) == (not declared), found
    assert not any("runs_by_job_start" in finding for finding in found), found
    raw = _PRUNED.replace('indexes=(Index("links_by_expiry", "expires_at"ASC, purpose="src/links.py::prune"),),', "")
    ddl = _PRUNED_DDL.replace("expires_atASC", f"expires_at{declared_ddl}")
    found = [
        f
        for f in no_whole_reads_findings(_config(tmp_path, {"links.py": raw, "schema.py": ddl}))
        if "index-hashed-range" in f
    ]
    assert bool(found) == (not declared_ddl), found


_STEPS = """
    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import Match, change_relation, delete_matching, read_step, semi, union

    RESULTS = TableSchema(name="results", primary_key=("state", "run_id"), columns=[])
    _CHANGED = change_relation("changed_runs")


    async def over_the_change(conn, held):
        conn.register("held_rows", held)
        step = semi("held_rows", _CHANGED, on=[Match("state"), Match("run_id")])
        return await read_step(conn, keyed.distinct(step, ["state"]))


    async def drop_changed(conn):
        return await delete_matching(conn, "results", _CHANGED, on=[Match("state"), Match("run_id")])


    async def over_a_table(conn):
        return await read_step(conn, keyed.scan("results"))


    async def semi_of_a_table(conn):
        return await read_step(conn, semi("results", _CHANGED, on=[Match("state")]))


    def laundered():
        return change_relation("results")


    def a_set(left, right):
        return left.union(right)
"""


def test_a_step_over_a_change_relation_passes_and_one_over_a_table_is_the_whole_read_it_is(tmp_path: Path) -> None:
    """Flips if a relational step over a registered or declared change relation is flagged, a delete by a change
    relation's keys is judged other than as the keyed delete it is, a step over a data table (whole, or as a semi's
    read side) passes, a change relation declared over a data table passes, or a set's ``union`` is judged as a step."""
    found = _symbols(_config(tmp_path, {"steps.py": _STEPS}))
    functions = {function for function, _ in found}
    assert not functions & {"over_the_change", "drop_changed", "a_set"}, found
    assert {"over_a_table", "semi_of_a_table"} <= functions, found
    assert ("laundered", "change-relation-is-a-table results") in found, found


_STAGE_HELPER = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import change_relation
    from threetears.core.keyed._filters import quoted_table

    RESULTS = TableSchema(name="results", primary_key=("state", "run_id"), columns=[])


    async def apply_stage(conn, target, stage, *, key, columns):
        staged = quoted_table(change_relation(stage))
        return await conn.execute(f"INSERT INTO results (state, run_id) SELECT state, run_id FROM {staged}")
"""

_STAGE_CALLERS = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import apply_stage, change_relation
    from threetears.core.keyed._filters import quoted_table

    RESULTS = TableSchema(name="results", primary_key=("state", "run_id"), columns=[])
    HELD = TableSchema(name="held", primary_key=("state", "run_id"), columns=[])


    async def from_a_chunk(conn, chunk):
        conn.register("chunk", chunk)
        return await apply_stage(conn, "held", "chunk", key=["state", "run_id"], columns=["state", "run_id"])


    async def from_a_table(conn):
        return await apply_stage(conn, "held", "results", key=["state", "run_id"], columns=["state", "run_id"])


    async def by_part_of_the_key(conn, chunk):
        conn.register("chunk", chunk)
        return await apply_stage(conn, "held", "chunk", key=["run_id"], columns=["state", "run_id"])


    async def appended(conn, chunk):
        conn.register("chunk", chunk)
        return await apply_stage(conn, "held", "chunk", key=[], columns=["state", "run_id"], mode="append")


    async def scoped(conn, chunk):
        conn.register("chunk", chunk)
        return await apply_stage(
            conn, "held", "chunk", key=["state", "run_id"], columns=["state", "run_id"], mode="difference", by={"state": "S"}
        )


    async def not_a_helper(conn, rel):
        staged = quoted_table(change_relation(rel))
        return await conn.execute(f"INSERT INTO held (state, run_id) SELECT state, run_id FROM {staged}")
"""


def test_a_keyed_helper_s_stage_parameter_is_a_change_relation_judged_where_it_is_handed(tmp_path: Path) -> None:
    """Flips if apply_stage's statements over its own stage parameter (declared ``change_relation(stage)``) are
    flagged, if a caller handing it a data table as the stage passes, if one handing it a registered chunk is flagged,
    or if ``change_relation`` of a parameter launders a function that is no 3tears keyed helper."""
    found = _symbols(
        _config(
            tmp_path,
            {"threetears/core/keyed/writes.py": _STAGE_HELPER, "app.py": _STAGE_CALLERS},
        )
    )
    functions = {function for function, _ in found}
    assert not functions & {"apply_stage", "from_a_chunk", "appended", "scoped"}, found
    assert {"from_a_table", "not_a_helper", "by_part_of_the_key"} <= functions, found


_HELPER_BODIES = """
    from dataclasses import dataclass

    from threetears.core.keyed._filters import quote_identifier, quoted_table, relation

    async def read_keyed(conn, table, keys):
        return await conn.fetch(f"SELECT * FROM {relation(table, 't')}", keys)

    async def holds_any(conn, table, order):
        target = quoted_table(table)
        return await conn.fetch(f"SELECT 1 AS held FROM {target} LIMIT 1")

    async def read_past_marks(conn, named, mark):
        return await conn.fetch(f"SELECT * FROM {quote_identifier(named)} WHERE stamp > $1", mark)

    async def any_rows(conn, table, limit):
        return await conn.fetch(f"SELECT * FROM events WHERE note = $1 LIMIT {limit}", table)

    def _render(table, where):
        return f"SELECT * FROM {quoted_table(table)} WHERE {where}"

    async def keyset_page(conn, table, where):
        return await conn.fetch(_render(table, where))

    def _shared(table):
        return f"SELECT count(*) AS n FROM {quoted_table(table)}"

    async def counts_by(conn, table):
        return await conn.fetch(_shared(table))

    async def not_a_helper(conn, table):
        return await conn.fetch(_shared(table))

    @dataclass
    class _Statement:
        table: str

        def render(self):
            return f"SELECT * FROM {relation(self.table, 't')}"

    async def read_keyed_arrow(conn, table):
        return await conn.fetch(_Statement(table=table).render())

    async def read_joined(conn, table, keys):
        return await read_keyed(conn, table, keys=keys)

    async def counts_by_relations(conn, tables, column):
        parts = [f"SELECT count(*) AS held FROM {relation(tables[label], 'r')} GROUP BY {column}" for label in tables]
        return await conn.fetch(" UNION ALL ".join(parts))
"""

_HELPER_BODY_CALLERS = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import holds_any, scan, update_from

    EVENTS = TableSchema(name="events", primary_key=("id",), columns=[])

    async def by_id(conn, event_id):
        return await holds_any(conn, "events", order=["id"], by={"id": event_id})

    async def by_note(conn, note):
        return await holds_any(conn, "events", order=["note"], by={"note": note})

    async def from_chunk(conn, chunk):
        conn.register("chunk", chunk)
        return await update_from(conn, "events", key=["id"], source=scan("chunk"), columns=["v"])

    async def from_chunk_by_note(conn, chunk):
        conn.register("chunk", chunk)
        return await update_from(conn, "events", key=["note"], source=scan("chunk"), columns=["v"])

    async def from_the_table(conn):
        return await update_from(conn, "events", key=["id"], source="events", columns=["v"])
"""


def test_a_keyed_helper_s_relation_parameter_is_judged_at_its_call_sites_not_in_its_body(tmp_path: Path) -> None:
    """Flips if a keyed helper's statement over its own relation parameter is flagged in its body (straight, through
    a local binding, a bare name in relation position, a private renderer or a field every in-module caller fills
    from a helper's parameter, a nested helper call on it), if literal-table SQL in a helper body passes, if a private
    renderer some other function reaches passes, or if a call site keying a helper by no key (``holds_any`` by a
    non-key column, ``update_from`` joined on one or reading a data table as its source) passes."""
    found = _symbols(
        _config(
            tmp_path,
            {"threetears/core/keyed/rows.py": _HELPER_BODIES, "app.py": _HELPER_BODY_CALLERS},
        )
    )
    functions = {function for function, _ in found}
    covered = {"read_keyed", "holds_any", "read_past_marks", "_render", "keyset_page"}
    covered |= {"_Statement.render", "read_keyed_arrow", "read_joined", "counts_by_relations", "by_id", "from_chunk"}
    assert not functions & covered, found
    assert {"any_rows", "_shared", "by_note", "from_chunk_by_note", "from_the_table"} <= functions, found


_STAGE_TEMPLATE = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import apply_stage

    def fence_schema(name: str = "fence"):
        return TableSchema(name=name, primary_key=("signal", "seq"), columns=[])

    async def settle(conn, table, chunk):
        conn.register("chunk", chunk)
        return await apply_stage(conn, table, "chunk", key=["signal", "seq"], columns=["signal", "seq"])

    async def by_seq(conn, table, chunk):
        conn.register("chunk", chunk)
        return await apply_stage(conn, table, "chunk", key=["seq"], columns=["signal", "seq"])
"""

_STAGE_NO_TEMPLATE = """
    from threetears.core.keyed import apply_stage

    async def settle(conn, table, chunk):
        conn.register("chunk", chunk)
        return await apply_stage(conn, table, "chunk", key=["signal", "seq"], columns=["signal", "seq"])
"""


def test_a_stage_s_target_its_caller_names_is_judged_by_its_module_s_template(tmp_path: Path) -> None:
    """Flips if apply_stage onto a table its caller names passes unjudged: keyed by the template's key where the
    stage's key leads it, a whole delete where it does not, and where the module declares no template to judge it by."""
    found = _symbols(_config(tmp_path, {"fence.py": _STAGE_TEMPLATE}))
    functions = {function for function, _ in found}
    assert "settle" not in functions and "by_seq" in functions, found
    bare = _symbols(_config(tmp_path / "bare", {"fence.py": _STAGE_NO_TEMPLATE}))
    assert any(function == "settle" for function, _ in bare), bare


_ANY_KIND_LEAD = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import AnyOf, Sort, keyset_page

    DATASOURCES = TableSchema(
        name="datasources",
        primary_key="id",
        columns=[],
        indexes=(
            Index("page_by_name", "kind", "name", "id", purpose="src/listing.py::name_page"),
            Index("page_by_date_created", "kind", "date_created", "id", purpose="src/listing.py::source_page"),
        ),
    )

    async def source_page(pool, kinds, cols):
        return await keyset_page(
            pool, "datasources", columns=cols, by={"kind": AnyOf(kinds)},
            order=[Sort("date_created"), Sort("id")], limit=50,
        )

    async def name_page(pool, kind, cols):
        return await keyset_page(pool, "datasources", columns=cols, by={"kind": kind}, order=[Sort("name"), Sort("id")], limit=50)
"""


def test_a_page_whose_lead_is_a_set_credits_the_index_it_orders_by_after_it(tmp_path: Path) -> None:
    """Flips if a keyset page pinning its index's lead to a set (``by={"kind": AnyOf(...)}``) and ordered by the
    index's next columns is flagged, or leaves that index's purpose unused (credited to the pinned-lead index of
    another order instead)."""
    found = _symbols(_config(tmp_path, {"listing.py": _ANY_KIND_LEAD}))
    assert not any(function in {"source_page", "name_page"} for function, _ in found), found
    findings = no_whole_reads_findings(_config(tmp_path, {"listing.py": _ANY_KIND_LEAD}))
    assert not any("index-purpose-unused" in finding for finding in findings), findings


_COPIED = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import Sort, copy_keyed

    CONNECTIONS = TableSchema(
        name="connections",
        primary_key="connection_id",
        columns=[],
        indexes=(Index("by_principal", "principal_kind", "principal_id", "admitted", purpose="src/fence.py::by_principal"),),
    )
    _ORDER = [Sort("admitted"), Sort("connection_id")]

    async def by_principal(pool, kind, principal):
        return await copy_keyed(pool, "connections", columns=["connection_id"], by={"principal_kind": kind, "principal_id": principal}, order=_ORDER)

    async def every_one(pool):
        return await copy_keyed(pool, "connections", columns=["connection_id"], by={}, order=_ORDER)
"""


def test_a_copy_of_a_prefix_is_judged_as_the_keyed_drain_it_is(tmp_path: Path) -> None:
    """Flips if copy_keyed by a prefix its purposed index leads is flagged or leaves the index unused, or if a copy
    with no prefix (every row) passes."""
    found = _symbols(_config(tmp_path, {"fence.py": _COPIED}))
    assert {function for function, _ in found} == {"every_one"}, found
    findings = no_whole_reads_findings(_config(tmp_path, {"fence.py": _COPIED}))
    assert not any("index-purpose-unused" in finding for finding in findings), findings


_RELATION_PRODUCED = """
    from dataclasses import dataclass
    from collections.abc import Sequence

    from threetears.core import keyed


    @dataclass
    class SourceTable:
        name: str
        relation: str


    def derived(relation):
        return keyed.Derived(relation)


    async def by_state(reader, source: SourceTable, state):
        return await keyed.read_past_marks(reader, source.relation, columns=["a"], marks={"up": 1}, by={"state": state})


    async def by_kind(reader, source: SourceTable | None, kinds):
        return await keyed.counts_by(reader, source.relation, column="kind", values=keyed.AnyOf(kinds))


    async def every_state(reader, sources: Sequence[SourceTable], states):
        for source in sources:
            await keyed.counts_by(reader, source.relation, column="state", values=keyed.AnyOf(states))


    async def every_marked(reader, marks: Sequence[tuple[SourceTable, int | None]], states):
        marked: dict[str, tuple[int, SourceTable]] = {s.name: (m, s) for s, m in marks if m is not None}
        return await keyed.counts_by_relations(
            reader,
            {name: source.relation for name, (_, source) in marked.items()},
            column="state",
            values=keyed.AnyOf(states),
        )


    async def derived_by_run(reader, part, runs):
        return await keyed.counts_by(reader, derived(part.relation), column="run_id", values=keyed.AnyOf(runs))


    async def derived_by_office(reader, part, offices):
        return await keyed.counts_by(reader, derived(part.relation), column="office", values=keyed.AnyOf(offices))


    async def untyped(reader, source, state):
        return await keyed.counts_by(reader, source.relation, column="state", values=keyed.AnyOf([state]))


    class Inputs:
        def __init__(self, inputs):
            self._inputs = inputs

        async def named(self, reader, names):
            return await keyed.read_keyed(reader, self._inputs, keys=keyed.KeyBatch(("name",), names), columns=["n"])

        async def every(self, reader, kinds):
            return await keyed.counts_by(reader, self._inputs, column="kind", values=keyed.AnyOf(kinds))
"""

_PRODUCERS = {"SourceTable.relation": ("state",), "derived": ("run_id",), "Inputs._inputs": ("name",)}


def test_a_relation_a_declared_producer_yields_is_judged_by_its_declared_key(tmp_path: Path) -> None:
    """Flips if a helper handed a run-time relation a declared producer yields (a typed receiver's attribute, a
    producer function's return, a ``self`` attribute in its class) is not judged by the declared key: a read bound by
    the key passes, one bound by another column is the whole read it is, and one the walk cannot type stays unseen."""
    found = _symbols(_config(tmp_path, {"produced.py": _RELATION_PRODUCED}, relation_producers=_PRODUCERS))
    functions = {function for function, _ in found}
    for keyed_read in ("by_state", "every_state", "every_marked", "derived_by_run", "Inputs.named"):
        assert keyed_read not in functions, found
    assert ("by_kind", "whole-aggregate produced.sourcetable__relation") in found, found
    assert ("derived_by_office", "whole-aggregate produced.derived") in found, found
    assert ("Inputs.every", "whole-aggregate produced.inputs___inputs") in found, found
    assert any(function == "untyped" and symbol.startswith("unknown-predicate") for function, symbol in found), found


def test_a_run_time_relation_with_no_producer_declared_stays_unseen(tmp_path: Path) -> None:
    """Flips if a run-time relation passes, or is judged against any key, with no producer declaring it."""
    found = _symbols(_config(tmp_path, {"produced.py": _RELATION_PRODUCED}))
    for function in ("by_state", "every_state", "every_marked", "derived_by_run", "Inputs.named"):
        assert any(f == function and s.startswith("unknown-predicate") for f, s in found), (function, found)
