"""The code half of the no-whole-reads rule, driven by synthesised modules shaped like the paths that failed on
election night, and its one way out: the exemptions ledger."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

import re

from threetears.core.declared import sentinel_templates
from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, no_whole_reads_findings, use_the_helper_findings
from threetears.enforcement.no_whole_reads import walker
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


def test_an_index_a_migration_renames_onto_its_declared_name_keeps_the_declared_purpose(tmp_path: Path) -> None:
    """Flips if a later migration's ALTER INDEX ... RENAME TO the name a collection declares (with its purpose and
    plan test) leaves that index with no purpose: the renamed index is the declared one. The migrations sit after the
    declaring module in the walk, as a package's migrations/ sits after its collections.py."""
    _plans(tmp_path)
    modules = {
        "registry.py": _REGISTRY,
        "versions/v001.py": 'UP = "CREATE INDEX IF NOT EXISTS datasources_kind ON datasources (kind)"\n',
        "versions/v002.py": 'UP = "ALTER INDEX IF EXISTS datasources_kind RENAME TO datasources_by_kind"\n',
    }
    findings = no_whole_reads_findings(_config(tmp_path, modules))
    assert not [finding for finding in findings if ": index-" in finding], findings


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


_DROPPED = '''
UP = """
CREATE TABLE grants (grant_id UUID PRIMARY KEY, agent_id UUID, status TEXT, status_note TEXT);
CREATE INDEX grants_by_status ON grants (status);
CREATE INDEX grants_live ON grants (agent_id) WHERE status = 'live';
CREATE INDEX grants_wide ON grants (agent_id) INCLUDE (status);
CREATE INDEX grants_lower ON grants (lower(status));
CREATE INDEX grants_by_note ON grants (status_note);
-- purpose: src/grants.py::agent_grants
-- plan_test: tests/test_plans.py::test_agent_grants_uses_grants_by_agent
CREATE INDEX grants_by_agent ON grants (agent_id);
CREATE TABLE seats (seat_id UUID PRIMARY KEY, owner TEXT);
CREATE INDEX seats_by_owner ON seats (owner);
"""
'''


@pytest.mark.parametrize(
    ("drop", "gone", "kept"),
    [
        (
            'UP = "ALTER TABLE grants DROP COLUMN IF EXISTS status"\n',
            ("grants_by_status", "grants_live", "grants_wide", "grants_lower"),
            ("grants_by_note", "seats_by_owner"),
        ),
        (
            'UP = "ALTER TABLE ONLY public.grants ADD COLUMN x TEXT, DROP status CASCADE"\n',
            ("grants_by_status", "grants_live", "grants_wide", "grants_lower"),
            ("grants_by_note", "seats_by_owner"),
        ),
        (
            'UP = "ALTER TABLE grants ALTER COLUMN status DROP NOT NULL, DROP CONSTRAINT status"\n',
            (),
            ("grants_by_status", "grants_live", "grants_by_note", "seats_by_owner"),
        ),
        (
            'UP = "DROP TABLE IF EXISTS seats CASCADE"\n',
            ("seats_by_owner",),
            ("grants_by_status", "grants_by_note"),
        ),
        (
            'UP = "DROP TABLE IF EXISTS public.grants, seats CASCADE"\n',
            ("seats_by_owner", "grants_by_status", "grants_live", "grants_by_note"),
            (),
        ),
    ],
    ids=["drop-column", "drop-column-among-actions", "drop-not-null-and-constraint", "drop-table", "drop-two-tables"],
)
def test_a_dropped_column_or_table_takes_its_indexes_with_it(
    tmp_path: Path, drop: str, gone: tuple[str, ...], kept: tuple[str, ...]
) -> None:
    """Flips if an index a later migration's DROP COLUMN takes with it (the column in its key, its INCLUDE, its
    predicate or an expression of it) or a DROP TABLE takes with its table (every table a DROP TABLE lists) is still
    counted, or if a drop takes an index it does not (another column whose name starts the same, another table, an
    ALTER COLUMN ... DROP NOT NULL or a DROP CONSTRAINT)."""
    _plans(
        tmp_path,
        "def test_agent_grants_uses_grants_by_agent(plan):\n    assert plan_uses_index(plan, 'grants_by_agent')\n",
    )
    config = _config(tmp_path, {"migrations/v001.py": _DROPPED, "migrations/v002.py": drop, "grants.py": _GRANTS})
    findings = [finding for finding in no_whole_reads_findings(config) if ": index-" in finding]
    for index in gone:
        assert not any(f"index {index} on" in finding for finding in findings), (index, findings)
    for index in kept:
        assert any(f"index {index} on" in finding for finding in findings), (index, findings)


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


def test_a_table_list_copy_of_a_dependency_s_pattern_named_index_is_the_dependency_s(tmp_path: Path) -> None:
    """Flips if the repo's table-list copy of an index a dependency names at run time (3tears' source_tokens_schema:
    ``f"{name}_by_signal_c"``) is asked a purpose by the repo's gate (reports #38: the pattern matched only a copy
    declared on the same side), or if a copy whose name the pattern does not make escapes the repo's gate."""
    dependency = tmp_path / "dep"
    dependency.mkdir()
    (dependency / "tokens.py").write_text(
        textwrap.dedent(
            """
            from threetears.core.collections.schema_backed import Index, TableSchema

            def source_tokens_schema(name: str):
                return TableSchema(
                    name=name,
                    primary_key="signal",
                    columns=[],
                    indexes=(
                        Index(
                            f"{name}_by_signal_c",
                            "signal",
                            purpose="dep/tokens.py::clear_signals",
                            plan_test="dep/test_tokens.py::test_plan",
                        ),
                    ),
                )
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    listed = """
    providers:
    - tables:
      - name: source_tokens
        columns: [{name: signal}, {name: owner}]
        primary_key: [signal]
        indexes:
        - name: source_tokens_by_signal_c
          columns: [signal]
        - name: source_tokens_by_owner
          columns: [owner]
    """
    (tmp_path / "tables.yaml").write_text(textwrap.dedent(listed), encoding="utf-8")
    catalog = collect_schema(tmp_path, (tmp_path / "src",), dependency_roots=(dependency,))
    findings = index_findings(catalog, tmp_path, set())
    assert not any("source_tokens_by_signal_c" in finding for finding in findings), findings
    assert any("index source_tokens_by_owner on source_tokens: index-no-purpose" in finding for finding in findings), (
        findings
    )


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


_RING_METHOD_READS = """
    from pkg.ring import note_table

    class Keeper:
        def __init__(self, owner):
            self._owner = owner

        def _ring_table(self, index):
            '''the ring's table at ``index``.'''
            return f"{self._owner}.{quote_identifier(note_table(index))}"

        async def by_id(self, conn, index, note_id):
            return await conn.fetch(f"SELECT * FROM {self._ring_table(index)} WHERE id = $1", note_id)

        async def by_scope(self, conn, index, scope):
            table = self._ring_table(index)
            return await conn.fetch(f"SELECT * FROM {table} WHERE scope = $1", scope)

    async def elsewhere(conn, keeper, index):
        return await conn.fetch(f"SELECT * FROM {keeper._ring_table(index)} WHERE scope = $1", 1)
"""


def test_a_method_that_only_returns_a_name_reads_as_its_template(tmp_path: Path) -> None:
    """Flips if a ring table a method only returns (``self._ring_table(index)``) stays unknown when its DDL declares
    the template, if a non-key read of it passes, or if a call on anything but ``self`` reads as some class's
    method."""
    found = _symbols(_config(tmp_path, {"pkg/ring.py": _RING, "pkg/reads.py": _RING_METHOD_READS}))
    assert ("Keeper.by_scope", "non-key-where {}.threetears_change_notes_{}") in found, found
    assert not any(function == "Keeper.by_id" for function, _ in found), found
    assert not any(function == "elsewhere" and "threetears_change_notes" in what for function, what in found), found


_SPARSE = """
    NOTES = (
        "CREATE TABLE IF NOT EXISTS notes (id UUID PRIMARY KEY, scope TEXT, at TIMESTAMPTZ, flag TEXT, hot BOOLEAN)",
        "-- purpose: src/sparse.py::by_scope; src/sparse.py::flagged\\n"
        "-- plan_test: tests/test_plans.py::test_notes_scope_at\\n"
        "CREATE INDEX IF NOT EXISTS notes_scope_at ON notes (scope, at)",
        "-- purpose: src/sparse.py::flagged\\n"
        "-- plan_test: tests/test_plans.py::test_notes_flagged\\n"
        "CREATE INDEX IF NOT EXISTS notes_flagged ON notes (scope, at) WHERE flag IS NULL",
        "-- purpose: src/sparse.py::flagged\\n"
        "-- plan_test: tests/test_plans.py::test_notes_flag_scope\\n"
        "CREATE INDEX IF NOT EXISTS notes_flag_scope ON notes (flag, scope)",
        "-- purpose: src/sparse.py::hot_counts; src/sparse.py::cold_counts\\n"
        "-- plan_test: tests/test_plans.py::test_notes_hot\\n"
        "-- reason: partial over the few hot notes alone, counted at each switch\\n"
        "-- speed: tests/test_speed.py::test_hot_counts_bounded\\n"
        "CREATE INDEX IF NOT EXISTS notes_hot ON notes (scope) WHERE hot",
    )

    async def by_scope(conn, scopes):
        return await conn.fetch("SELECT DISTINCT scope FROM notes WHERE scope = ANY($1::text[])", scopes)

    async def flagged(conn, scopes):
        return await conn.fetch("SELECT DISTINCT scope FROM notes n WHERE n.flag IS NULL AND scope = ANY($1::text[])", scopes)

    async def hot_counts(conn):
        return await conn.fetch("SELECT scope, count(*) AS n FROM notes WHERE hot GROUP BY 1")

    async def cold_counts(conn):
        return await conn.fetch("SELECT scope, count(*) AS n FROM notes WHERE NOT hot GROUP BY 1")
"""

_SPARSE_PLANS = """
    from threetears.core.testing.plans import plan_uses_index

    def test_notes_scope_at(plan):
        assert plan_uses_index(plan, "notes_scope_at")

    def test_notes_flagged(plan):
        assert plan_uses_index(plan, "notes_flagged")

    def test_notes_hot(plan):
        assert plan_uses_index(plan, "notes_hot")

    def test_notes_flag_scope(plan):
        assert plan_uses_index(plan, "notes_flag_scope")
"""

_SPARSE_SPEED = """
    from threetears.core.testing.calibrate import budget_problem

    async def test_hot_counts_bounded(conn, took, baseline):
        await hot_counts(conn)
        assert not budget_problem("hot counts", took, baseline)
"""


def test_a_partial_index_whose_predicate_the_read_states_is_the_one_it_reads(tmp_path: Path) -> None:
    """Flips if a read stating a partial index's predicate is not credited to it (tied on the bound lead with a full
    index, beside a full index its bound columns cover longer, or with nothing of its columns bound), or if a read
    that does not state it is (partial-index credit, 3tears #79)."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_plans.py").write_text(textwrap.dedent(_SPARSE_PLANS), encoding="utf-8")
    (tmp_path / "tests" / "test_speed.py").write_text(textwrap.dedent(_SPARSE_SPEED), encoding="utf-8")
    findings = no_whole_reads_findings(_config(tmp_path, {"sparse.py": _SPARSE}))
    assert not [f for f in findings if "in flagged:" in f or "in hot_counts:" in f or "in by_scope:" in f], findings
    assert not [f for f in findings if "index-speed" in f], findings
    assert not [f for f in findings if "notes_flagged on notes: index-purpose-unused" in f], findings
    assert [f for f in findings if "in cold_counts: whole-aggregate notes" in f], findings
    assert [f for f in findings if "cold_counts holds no read" in f], findings


def test_a_partial_index_keys_a_predicate_only_read_only_with_its_proof(tmp_path: Path) -> None:
    """Flips if a read binding none of a partial index's columns (``WHERE hot``, ``WHERE deleted_at IS NULL``) is
    keyed by the index without its full proof -- purpose, plan test, written reason and a speed test at real size
    calling ``budget_problem`` (coordinator ruling, 3tears #80) -- or if a speed test that never calls
    ``budget_problem`` or never runs the read is accepted."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_plans.py").write_text(textwrap.dedent(_SPARSE_PLANS), encoding="utf-8")
    # no reason, no speed test: the predicate alone keys nothing
    bare = _SPARSE.replace(
        '"-- reason: partial over the few hot notes alone, counted at each switch\\n"\n', ""
    ).replace('"-- speed: tests/test_speed.py::test_hot_counts_bounded\\n"\n', "")
    assert bare != _SPARSE
    findings = no_whole_reads_findings(_config(tmp_path, {"sparse.py": bare}))
    assert [f for f in findings if "in hot_counts:" in f], findings
    # a reason but no speed test: still a finding
    reasoned = _SPARSE.replace('"-- speed: tests/test_speed.py::test_hot_counts_bounded\\n"\n', "")
    findings = no_whole_reads_findings(_config(tmp_path, {"sparse.py": reasoned}))
    assert [f for f in findings if "in hot_counts:" in f], findings
    # a speed test that never calls budget_problem: the index declaration is a finding
    (tmp_path / "tests" / "test_speed.py").write_text(
        textwrap.dedent(_SPARSE_SPEED).replace('assert not budget_problem("hot counts", took, baseline)', "pass"),
        encoding="utf-8",
    )
    findings = no_whole_reads_findings(_config(tmp_path, {"sparse.py": _SPARSE}))
    assert [f for f in findings if "notes_hot on notes: index-speed-test" in f and "budget_problem" in f], findings


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


_TWO_SERVING = """
    from threetears.core.collections.schema_backed import Index, TableSchema

    NOTES = TableSchema(
        name="notes",
        primary_key="note_id",
        columns=[],
        indexes=(
            Index("by_owner", "agent_id", "user_id", purpose="src/notes.py::nearest", plan_test="tests/test_plans.py::test_owner"),
            Index("by_user_date", "user_id", "date_created", purpose="src/notes.py::nearest", plan_test="tests/test_plans.py::test_user_date"),
            Index("by_kind", "kind", purpose="src/notes.py::nearest", plan_test="tests/test_plans.py::test_kind"),
        ),
    )

    async def nearest(conn, agent_id, user_id, older_than):
        return await conn.fetch(
            "SELECT note_id FROM notes WHERE agent_id = $1 AND user_id = $2 AND date_created < $3 ORDER BY score LIMIT 5",
            agent_id,
            user_id,
            older_than,
        )
"""


def test_every_purposed_index_serving_a_read_is_credited_with_it(tmp_path: Path) -> None:
    """Flips if, of two purposed indexes that both serve one read no key prefix keys (the owner pair; the user and a
    date range), only the first declared is credited (the other reports its purpose unused), or if an index naming the
    read whose leading column the read never binds is credited. (A read a key prefix keys already credits every
    purposed index serving it.)"""
    _plans(
        tmp_path,
        "def test_owner(plan):\n    assert plan_uses_index(plan, 'by_owner')\n"
        "def test_user_date(plan):\n    assert plan_uses_index(plan, 'by_user_date')\n"
        "def test_kind(plan):\n    assert plan_uses_index(plan, 'by_kind')\n",
    )
    findings = no_whole_reads_findings(_config(tmp_path, {"notes.py": _TWO_SERVING}))
    unused = [finding for finding in findings if "index-purpose-unused" in finding]
    assert not any("index by_owner" in finding for finding in unused), unused
    assert not any("index by_user_date" in finding for finding in unused), unused
    assert any("index by_kind" in finding for finding in unused), findings


_PARTLY_SERVING = _TWO_SERVING.replace(
    'Index("by_kind", "kind",',
    'Index("by_user_kind", "user_id", "kind", purpose="src/notes.py::nearest", '
    'plan_test="tests/test_plans.py::test_user_kind"),\n            Index("by_kind", "kind",',
)


def test_an_index_whose_next_column_the_read_neither_binds_ranges_nor_orders_is_not_credited(tmp_path: Path) -> None:
    """Flips if a purposed index led by a bound column whose next column the read neither binds, ranges nor orders
    by (``(user_id, kind)`` under the owner pair and a date range) is credited beside the indexes that serve it."""
    _plans(
        tmp_path,
        "def test_owner(plan):\n    assert plan_uses_index(plan, 'by_owner')\n"
        "def test_user_date(plan):\n    assert plan_uses_index(plan, 'by_user_date')\n"
        "def test_user_kind(plan):\n    assert plan_uses_index(plan, 'by_user_kind')\n"
        "def test_kind(plan):\n    assert plan_uses_index(plan, 'by_kind')\n",
    )
    findings = no_whole_reads_findings(_config(tmp_path, {"notes.py": _PARTLY_SERVING}))
    unused = [finding for finding in findings if "index-purpose-unused" in finding]
    assert any("index by_user_kind" in finding for finding in unused), findings
    assert not any("index by_owner" in finding or "index by_user_date" in finding for finding in unused), unused


_NEAREST = """
    from threetears.core.collections.schema_backed import Index, TableSchema

    NOTES = TableSchema(
        name="notes",
        primary_key="note_id",
        columns=[],
        indexes=(
            Index("by_owner", "agent_id", "user_id", purpose="src/notes.py::owned_nearest", plan_test="tests/test_plans.py::test_owner"),
            Index(
                "by_embedding",
                "embedding",
                using="hnsw",
                purpose="src/notes.py::nearest; src/notes.py::owned_nearest",
                plan_test="tests/test_plans.py::test_embedding",
            ),
            Index("by_tags", "tags", using="gin", purpose="src/notes.py::tagged", plan_test="tests/test_plans.py::test_tags"),
        ),
    )

    async def nearest(conn, vector):
        return await conn.fetch("SELECT note_id FROM notes ORDER BY embedding <=> $1 LIMIT 5", vector)

    async def owned_nearest(conn, vector, agent_id, user_id):
        return await conn.fetch(
            "SELECT note_id FROM notes WHERE agent_id = $2 AND user_id = $3 AND embedding IS NOT NULL"
            " ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector LIMIT 10",
            vector,
            agent_id,
            user_id,
        )

    async def tagged(conn, tags):
        return await conn.fetch("SELECT note_id FROM notes WHERE tags @> $1::jsonb", tags)
"""

_NEAREST_PLANS = (
    "def test_owner(plan):\n    assert plan_uses_index(plan, 'by_owner')\n"
    "def test_embedding(plan):\n    assert plan_uses_index(plan, 'by_embedding')\n"
    "def test_tags(plan):\n    assert plan_uses_index(plan, 'by_tags')\n"
)


def test_the_k_nearest_and_a_containment_are_read_through_their_purposed_hnsw_and_gin_indexes(tmp_path: Path) -> None:
    """Flips if the k nearest rows by a distance from a column (``ORDER BY embedding <=> $1 LIMIT k``, bare or as
    ``OPERATOR(public.<=>)``) are not judged index-served through a purposed HNSW index of that column, or a
    containment test (``tags @> $1``) through a purposed GIN index of it; or if either index is reported unused."""
    _plans(tmp_path, _NEAREST_PLANS)
    findings = no_whole_reads_findings(_config(tmp_path, {"notes.py": _NEAREST}))
    assert findings == [], findings


def test_the_k_nearest_or_a_containment_without_its_index_is_a_whole_read(tmp_path: Path) -> None:
    """Flips if the k nearest rows pass with no approximate-nearest-neighbour index of the column (none at all, or a
    plain btree one: it holds no distance order), or a containment test passes with no index of its column."""
    _plans(tmp_path, _NEAREST_PLANS)
    unserved = (
        _NEAREST.replace('using="hnsw",', "")
        .replace(
            'purpose="src/notes.py::nearest; src/notes.py::owned_nearest"', 'purpose="src/notes.py::owned_nearest"'
        )
        .replace(
            'Index("by_tags", "tags", using="gin", purpose="src/notes.py::tagged", plan_test="tests/test_plans.py::test_tags"),',
            "",
        )
    )
    findings = no_whole_reads_findings(_config(tmp_path, {"notes.py": unserved}))
    assert any("in nearest:" in finding for finding in findings), findings
    assert any("in tagged:" in finding for finding in findings), findings
    # owned_nearest is still keyed by the owner pair; the plain index of embedding is not what serves its order
    assert not any("in owned_nearest:" in finding for finding in findings), findings
    assert any("index by_embedding" in finding and "index-purpose-unused" in finding for finding in findings), findings


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


_LOCAL_NAMES = """
    LAYOUT = "CREATE TABLE IF NOT EXISTS kinds (id UUID PRIMARY KEY, kind TEXT)"

    def _number(value, kind, *, whole=False):
        return int(value) if whole else float(value)

    def judge(value):
        return _number(value, "share", whole=True)

    def outside(value):
        return fetch_everything(value, whole=True)

    class Store:
        async def load(self, *, whole=False):
            return await self._pool.fetch("SELECT * FROM kinds")

        async def warm(self):
            return await self.load(whole=True)

        def label(self, value, *, whole=False):
            return str(value)

        def shown(self, value):
            return self.label(value, whole=True)

    def cell(value):
        def whole(v):
            return isinstance(v, int) and v >= 0

        return whole(value)

    async def reread(conn):
        async def whole(c):
            return await c.fetch("SELECT * FROM kinds")

        return await whole(conn)
"""


def test_a_whole_flag_or_whole_named_call_to_a_function_that_reads_nothing_whole_is_no_read(tmp_path: Path) -> None:
    """Flips if a ``whole=`` keyword to a function of the module (or a method of the caller's class) that reaches no
    whole read is a finding (evals' number-format flag), if a call to a local function named ``whole`` that reads
    nothing is (evals' type check), or if either stops being one where the callee does read whole or is not ours."""
    found = _symbols(_config(tmp_path, {"local.py": _LOCAL_NAMES}))
    assert not {function for function, _ in found} & {"judge", "cell", "Store.shown"}, found
    assert ("outside", "whole-flag whole=") in found, found
    assert ("Store.warm", "whole-flag whole=") in found, found
    assert ("reread", "calls-whole-read reread.whole") in found, found


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


_FK_DDL = """
    UP = (
        "CREATE TABLE schedules (conversation_id UUID, schedule_id UUID, context_from_schedule_id UUID,"
        " PRIMARY KEY (conversation_id, schedule_id), UNIQUE (schedule_id),"
        " CONSTRAINT schedules_context_from_fk FOREIGN KEY (context_from_schedule_id)"
        " REFERENCES schedules(schedule_id) ON DELETE SET NULL)"
    )
    INDEX = (
        "-- purpose: PURPOSE\\n"
        "-- plan_test: tests/test_plans.py::test_a_deleted_parent_finds_its_children\\n"
        "CREATE INDEX idx_schedules_context_from ON schedules (COLUMNS) WHERE context_from_schedule_id IS NOT NULL"
    )
"""

_FK_SQLALCHEMY = """
    from sqlalchemy import Column, ForeignKeyConstraint, Index, MetaData, String, Table

    METADATA = MetaData()
    schedules = Table(
        "schedules",
        METADATA,
        Column("conversation_id", String, primary_key=True),
        Column("schedule_id", String, primary_key=True),
        Column("context_from_schedule_id", String),
        ForeignKeyConstraint(
            ["context_from_schedule_id"],
            ["schedules.schedule_id"],
            name="schedules_context_from_fk",
        ),
        Index(
            "idx_schedules_context_from",
            "COLUMNS",
            info={
                "purpose": "PURPOSE",
                "plan_test": "tests/test_plans.py::test_a_deleted_parent_finds_its_children",
            },
        ),
    )
"""

_FK_ALTER = """
    UP = (
        "CREATE TABLE schedules (conversation_id UUID, schedule_id UUID, context_from_schedule_id UUID,"
        " PRIMARY KEY (conversation_id, schedule_id))"
    )
    LATER = (
        "ALTER TABLE schedules ADD CONSTRAINT schedules_context_from_fk FOREIGN KEY (context_from_schedule_id)"
        " REFERENCES schedules(schedule_id)"
    )
    INDEX = (
        "-- purpose: PURPOSE\\n"
        "-- plan_test: tests/test_plans.py::test_a_deleted_parent_finds_its_children\\n"
        "CREATE INDEX idx_schedules_context_from ON schedules (COLUMNS)"
    )
"""


@pytest.mark.parametrize("source", [_FK_DDL, _FK_SQLALCHEMY, _FK_ALTER], ids=["ddl", "sqlalchemy", "alter"])
@pytest.mark.parametrize(
    ("purpose", "columns", "message"),
    [
        ("fk:schedules_context_from_fk", "context_from_schedule_id", None),
        ("fk:schedules_skill_fk", "context_from_schedule_id", "index-fk-unknown"),
        ("fk:schedules_context_from_fk", "conversation_id", "index-fk-mismatch"),
    ],
)
def test_an_index_may_serve_a_declared_foreign_key_led_by_its_referencing_columns(
    tmp_path: Path, source: str, purpose: str, columns: str, message: str | None
) -> None:
    """Flips if an index serving a foreign key (purpose ``fk:<constraint>``, the database's lookup of the referencing
    rows when a parent is deleted) is refused when its table declares that key and the index leads with its
    referencing columns, or accepted when the constraint is not a foreign key of its table or the index leads with
    other columns."""
    _plans(
        tmp_path,
        "def test_a_deleted_parent_finds_its_children(plan):\n"
        "    assert plan_uses_index(plan, 'idx_schedules_context_from')\n",
    )
    module = source.replace("PURPOSE", purpose).replace("COLUMNS", columns)
    findings = [f for f in no_whole_reads_findings(_config(tmp_path, {"schedules.py": module})) if ": index-" in f]
    if message is None:
        assert findings == [], findings
    else:
        assert any(message in finding for finding in findings), findings


def test_a_partial_index_serves_no_foreign_key(tmp_path: Path) -> None:
    """Flips if a partial index is accepted as serving a foreign key (``fk:<constraint>``): it holds only the rows its
    ``WHERE`` admits, and the database's lookup of a deleted parent's referencing rows asks every one of them."""
    _plans(
        tmp_path,
        "def test_a_deleted_parent_finds_its_children(plan):\n"
        "    assert plan_uses_index(plan, 'idx_schedules_context_from')\n",
    )
    module = (
        _FK_ALTER.replace("PURPOSE", "fk:schedules_context_from_fk")
        .replace("COLUMNS", "context_from_schedule_id")
        .replace(
            'ON schedules (context_from_schedule_id)"',
            'ON schedules (context_from_schedule_id) WHERE conversation_id IS NOT NULL"',
        )
    )
    assert "WHERE conversation_id IS NOT NULL" in module
    findings = [f for f in no_whole_reads_findings(_config(tmp_path, {"schedules.py": module})) if ": index-" in f]
    assert any("index-fk-partial" in finding for finding in findings), findings


_TIE_PARENT = (
    "CREATE TABLE parents (conversation_id UUID, parent_id UUID, PRIMARY KEY (conversation_id, parent_id),"
    " UNIQUE (parent_id))"
)
_TIES = {
    "ddl": f"""
        P = {_TIE_PARENT!r}
        C = (
            "CREATE TABLE children (conversation_id UUID, child_id UUID, parent_id UUID,"
            " PRIMARY KEY (conversation_id, child_id),"
            " CONSTRAINT children_parent_fk FOREIGN KEY (conversation_id, parent_id)"
            " REFERENCES parents (conversation_id, parent_id) ON DELETE CASCADE)"
        )
    """,
    "alter_not_valid": f"""
        P = {_TIE_PARENT!r}
        C = "CREATE TABLE children (conversation_id UUID, child_id UUID, parent_id UUID, PRIMARY KEY (conversation_id, child_id))"
        FK = (
            "ALTER TABLE children ADD CONSTRAINT children_parent_fk FOREIGN KEY (conversation_id, parent_id)"
            " REFERENCES parents (conversation_id, parent_id) ON DELETE CASCADE NOT VALID"
        )
    """,
    "sqlalchemy": f"""
        from sqlalchemy import Column, ForeignKeyConstraint, MetaData, Table
        P = {_TIE_PARENT!r}
        CHILDREN = Table(
            "children",
            MetaData(),
            Column("conversation_id", primary_key=True),
            Column("child_id", primary_key=True),
            Column("parent_id"),
            ForeignKeyConstraint(
                ["conversation_id", "parent_id"], ["parents.conversation_id", "parents.parent_id"], name="children_parent_fk"
            ),
        )
    """,
    "foreign_key_def": f"""
        from threetears.core.collections.schema_backed import ForeignKeyDef, TableSchema
        P = {_TIE_PARENT!r}
        CHILDREN = TableSchema(
            name="children",
            primary_key=("conversation_id", "child_id"),
            columns=[],
            foreign_keys=(
                ForeignKeyDef(local_cols=("conversation_id", "parent_id"), ref_table="parents", ref_cols=("conversation_id", "parent_id")),
            ),
        )
    """,
}


@pytest.mark.parametrize("source", sorted(_TIES))
def test_a_foreign_key_records_what_it_references(tmp_path: Path, source: str) -> None:
    """Flips if a foreign key's target (the referenced table and columns) is lost from the catalog for any of the ways
    one is declared, so a column it ties to the partition column (``parent_id``: its referenced column alone is unique,
    so one value fixes the conversation) is not found, or one it does not tie is."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "tables.py").write_text(textwrap.dedent(_TIES[source]), encoding="utf-8")
    catalog = collect_schema(tmp_path, (tmp_path / "src",))
    children = catalog.tables["children"]
    # 3tears' ForeignKeyDef names no constraint: Postgres' default name
    name = "children_conversation_id_parent_id_fkey" if source == "foreign_key_def" else "children_parent_fk"
    assert children.references == {name: ("parents", ("conversation_id", "parent_id"))}
    assert catalog.tied_columns("children", "conversation_id") == frozenset({"parent_id"})
    assert catalog.tied_columns("children", "child_id") == frozenset()
    assert catalog.tied_columns("parents", "conversation_id") == frozenset()


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


_COPY_NULL_PINS = """
    from threetears.core import keyed
    from threetears.core.keyed import run_inline

    HELD = "lsf_held_runs"


    def renew_nameless(conn, table):
        runs = f"main.{HELD}"
        run_inline(keyed.delete_keyed(conn, runs, by={"tbl": table, "state": None}))


    def renew_by_other(conn, table):
        runs = f"main.{HELD}"
        run_inline(keyed.delete_keyed(conn, runs, by={"tbl": None}))
"""


def test_a_none_pin_on_a_local_copy_s_declared_key_is_its_null_bucket(tmp_path: Path) -> None:
    """Flips if a keyed helper's ``by={<declared key column>: None}`` on a local copy written in the local database's
    own schema (``main.<name>``) is called a whole read, as the same ``IS NULL`` on a ``copy.`` relation is not; or a
    None pin on a column no key of the copy names passes (reports #48 item 1)."""
    found = _symbols(_config(tmp_path, {"held.py": _COPY_NULL_PINS}, local_relations={"lsf_held_runs": ("state",)}))
    assert not any(function == "renew_nameless" for function, _ in found), found
    assert any(function == "renew_by_other" for function, _ in found), found


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


_SOURCE_DERIVED = """
    from threetears.core import keyed
    from threetears.core.operator import source_change_read


    class Lsf:
        @source_change_read(ids=("run_id",), proof="tests/test_source.py::test_runs")
        async def race_states(self, relation, run_ids):
            paired = keyed.Derived(f"SELECT *, office_key || state AS race_state FROM ({relation.sql}) AS race_rows")
            return await keyed.counts_by(
                self._reader, paired, column="race_state", by={"run_id": keyed.AnyOf(run_ids, as_list=True)}
            )

        @source_change_read(ids=("run_id",), proof="tests/test_source.py::test_runs")
        async def race_states_by_fragment(self, relation, run_ids):
            paired = keyed.Derived(f"SELECT * FROM ({relation.sql}) AS race_rows")
            return await keyed.counts_by(
                self._reader, paired, column="race_state", where=[keyed.Fragment("run_id = ANY($1)", run_ids)]
            )

        @source_change_read(ids=("run_id",), proof="tests/test_source.py::test_runs")
        async def produced_by_runs(self, relation, run_ids):
            return await keyed.counts_by(
                self._reader, long_derived(relation), column="race_state", by={"run_id": keyed.AnyOf(run_ids)}
            )

        @source_change_read(ids=("run_id",), proof="tests/test_source.py::test_runs")
        async def race_states_by_family(self, relation, families):
            paired = keyed.Derived(f"SELECT * FROM ({relation.sql}) AS race_rows")
            return await keyed.counts_by(
                self._reader, paired, column="race_state", by={"family": keyed.AnyOf(families, as_list=True)}
            )

        @source_change_read(ids=("run_id",), proof="tests/test_source.py::test_runs")
        async def produced_by_family(self, relation, families):
            return await keyed.counts_by(
                self._reader, long_derived(relation), column="race_state", by={"family": keyed.AnyOf(families)}
            )

        async def race_states_unmarked(self, relation, run_ids):
            paired = keyed.Derived(f"SELECT * FROM ({relation.sql}) AS race_rows")
            return await keyed.counts_by(
                self._reader, paired, column="race_state", by={"run_id": keyed.AnyOf(run_ids, as_list=True)}
            )
"""


def test_a_source_read_over_a_run_time_relation_is_bounded_by_its_marked_id_pin(tmp_path: Path) -> None:
    """Flips if, inside ``@source_change_read(ids=...)``, a keyed helper over a run-time ``Derived`` relation or a
    keyless declared producer's relation is a finding when the call pins one of the mark's id columns (``by=`` or a
    literal ``Fragment``); or passes with no such pin, or outside the mark (reports #48 item 2)."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_source.py").write_text("async def test_runs(lsf):\n    assert lsf\n", encoding="utf-8")
    with pytest.warns(UserWarning):
        found = _symbols(_config(tmp_path, {"lsf.py": _SOURCE_DERIVED}, relation_producers={"long_derived": ()}))
    functions = {function for function, _ in found}
    for bounded in ("Lsf.race_states", "Lsf.race_states_by_fragment", "Lsf.produced_by_runs"):
        assert bounded not in functions, found
    for unbounded in ("Lsf.race_states_by_family", "Lsf.produced_by_family", "Lsf.race_states_unmarked"):
        assert unbounded in functions, found


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


_PAGED_ORDERS = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.data.migrations.backfill import backfill_pages
    from threetears.core.keyed import AnyOf, Sort, keyset_page, keyset_rows
    from threetears.core.keyset import keyset_pages

    EVENTS = TableSchema(
        name="events",
        primary_key=("agent_id", "event_id"),
        columns=[],
        indexes=(
            Index("events_by_time", "agent_id", "created_at", "event_id", purpose="src/events.py::*"),
        ),
    )
    _NEWEST = (Sort("created_at", descending=True), Sort("event_id", descending=True))

    async def unique_page(c, agent_id, cols, cursor):
        return await keyset_page(
            c, "events", columns=cols, by={"agent_id": agent_id}, order=_NEWEST, limit=5, cursor=cursor
        )

    async def first_page(c, agent_id, cols):
        return await keyset_page(
            c, "events", columns=cols, by={"agent_id": agent_id}, order=[Sort("created_at")], limit=5
        )

    async def unique_rows(c, cols):
        return await keyset_rows(c, "events", columns=cols, by={}, order=[Sort("agent_id"), Sort("event_id")])

    async def tied_page(c, agent_id, cols, cursor):
        return await keyset_page(
            c, "events", columns=cols, by={"agent_id": agent_id}, order=[Sort("created_at")], limit=5, cursor=cursor
        )

    async def tied_set(c, agents, cols):
        return await keyset_rows(c, "events", columns=cols, by={"agent_id": AnyOf(agents)}, order=[Sort("event_id")])

    async def unique_drain(pool, agent_id):
        async for page in keyset_pages(
            pool, "SELECT event_id FROM events", ["created_at", "event_id"], where={"agent_id": agent_id}, page_size=10
        ):
            return page

    async def tied_drain(pool, agent_id):
        async for page in keyset_pages(
            pool, "SELECT event_id FROM events", ["created_at"], where={"agent_id": agent_id}, page_size=10
        ):
            return page

    GROUPS_DDL = (
        "CREATE TABLE groups (group_id UUID PRIMARY KEY, customer_id UUID, name TEXT);"
        "CREATE UNIQUE INDEX uq_groups_customer_name ON groups (customer_id, name) WHERE customer_id IS NOT NULL;"
    )

    async def unique_by_name(c, customer_id, cols, cursor):
        return await keyset_page(
            c, "groups", columns=cols, by={"customer_id": customer_id}, order=[Sort("name")], limit=5, cursor=cursor
        )

    async def tied_platform_names(c, cols, cursor):
        return await keyset_page(
            c, "groups", columns=cols, by={"customer_id": None}, order=[Sort("name")], limit=5, cursor=cursor
        )

    async def unique_fill(conn):
        return await backfill_pages(conn, "events", ["agent_id", "event_id"], columns=["x"], fill=dict)

    async def tied_fill(conn):
        return await backfill_pages(conn, "events", ["event_id"], columns=["x"], fill=dict)
"""


def test_a_keyset_pager_whose_pins_and_order_hold_no_primary_key_is_named(tmp_path: Path) -> None:
    """Flips if a keyset pager (keyset_page / keyset_rows, a keyset_pages drain, a backfill) whose pinned columns and
    order hold no primary key of its table passes: each page starts past the last row read, so the rows sharing that
    row's order values are skipped. A column pinned to a set (AnyOf) holds many values and pins nothing; pinned and
    ordered columns that hold the key pass, and so does a keyset_page handed no cursor (one page, past nothing). A
    partial unique index is a key where the pins hold its predicate (``customer_id IS NOT NULL``: customer_id pinned
    to a value), never where they pin it NULL."""
    found = _symbols(_config(tmp_path, {"events.py": _PAGED_ORDERS}))
    ordered = {function for function, symbol in found if symbol.startswith("non-unique-order")}
    assert ordered == {"tied_page", "tied_set", "tied_drain", "tied_fill", "tied_platform_names"}, found


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


_IMPORTED_ORDER = {
    "listing_order.py": """
    from typing import Final

    from threetears.core.keyed import Sort

    OLDEST_FIRST: Final = (Sort("date_started", cast="timestamptz"), Sort("operation_id", cast="uuid"))
""",
    "listing.py": """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import keyset_page
    from listing_order import OLDEST_FIRST as OLDEST

    OPERATIONS = TableSchema(
        name="backup_operations",
        primary_key="operation_id",
        columns=[],
        indexes=(Index("recent", "date_started", purpose="src/listing.py::oldest"),),
    )

    async def oldest(pool, cols):
        return await keyset_page(pool, "backup_operations", columns=cols, by={}, order=OLDEST, limit=10)
""",
}


def test_a_helper_s_order_imported_from_another_module_is_read(tmp_path: Path) -> None:
    """Flips if a helper's order given as a constant another module of the repo declares (``from x import NAME``,
    renamed too) is unread: the read unknown and its purposed index unused (hub #55 S2)."""
    found = _symbols(_config(tmp_path, _IMPORTED_ORDER))
    assert not any(function == "oldest" for function, _ in found), found
    findings = no_whole_reads_findings(_config(tmp_path, _IMPORTED_ORDER))
    assert not any("index-purpose-unused" in finding for finding in findings), findings


_VISIBLE_LISTING = {
    "listing_order.py": _IMPORTED_ORDER["listing_order.py"],
    "concepts.py": """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.agent.acl import visible_rows
    from listing_order import OLDEST_FIRST

    CONCEPTS = TableSchema(
        name="concepts",
        primary_key="operation_id",
        columns=[],
        indexes=(Index("by_customer", "customer_id", "date_started", purpose="src/concepts.py::listed"),),
    )

    async def listed(conn, user_id):
        return await visible_rows(
            conn, "concepts", columns=["operation_id"], user_id=user_id, namespace_type="knowledge",
            customer_column="customer_id", user_column="user_id", order=OLDEST_FIRST,
        )
""",
}


def test_a_visible_listing_is_narrowed_to_the_caller_s_customers(tmp_path: Path) -> None:
    """Flips if visible_rows is judged as a page over every row: it narrows each page to the caller's customers by
    its ``customer_column`` (``= ANY($n)``), so an index led by that column keys it (hub #55 S2)."""
    found = _symbols(_config(tmp_path, _VISIBLE_LISTING))
    assert not any(function == "listed" for function, _ in found), found


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


_ORDERS_PICKED_BY_A_CONSTANT = """
    from typing import Final

    from threetears.core.collections.schema_backed import Column, TableSchema
    from threetears.core.keyed import Sort, keyset_page, keyset_rows

    CHANNEL_CONFIGS = TableSchema(
        name="channel_configs",
        primary_key=("agent_id", "id"),
        columns=[Column("id"), Column("agent_id", partition=True), Column("status"), Column("date_created")],
    )
    CHANNEL_ORDERS: Final[dict[tuple[str, bool], tuple[Sort, ...]]] = {
        ("status", False): (Sort("status", "text"), Sort("id", "uuid")),
        ("date_created", False): (Sort("date_created", "timestamptz"), Sort("id", "uuid")),
    }


    class Configs:
        async def page(self, pool, agent_id, sort, cols):
            order = CHANNEL_ORDERS[(sort, False)]
            return await keyset_page(pool, "channel_configs", columns=cols, by={"agent_id": agent_id}, order=order, limit=50)

        async def every(self, pool, agent_id, cols):
            return await keyset_rows(
                pool, "channel_configs", columns=cols, by={"agent_id": agent_id},
                order=CHANNEL_ORDERS[("date_created", False)], page=100,
            )

        async def every_by_status(self, pool, agent_id, cols):
            return await keyset_rows(
                pool, "channel_configs", columns=cols, by={"agent_id": agent_id},
                order=CHANNEL_ORDERS[("status", False)], page=100,
            )
"""

#: the indexes, as hub v162 declares them: each sort's own index, its purpose naming only the reads that page it
_ORDERS_PICKED_DDL = """
    DDL_STATEMENTS: tuple[str, ...] = (
        \"\"\"
    -- purpose: src/channels.py::Configs.page
    -- plan_test: tests/test_plans.py::test_status
    CREATE INDEX IF NOT EXISTS page_by_agent_status ON channel_configs (agent_id, status ASC, id ASC)
    \"\"\",
        \"\"\"
    -- purpose: src/channels.py::Configs.page; src/channels.py::Configs.every
    -- plan_test: tests/test_plans.py::test_date_created
    CREATE INDEX IF NOT EXISTS page_by_agent_date_created ON channel_configs (agent_id, date_created ASC, id ASC)
    \"\"\",
    )
"""


def test_an_order_picked_from_a_module_mapping_by_a_literal_key_is_judged_as_that_entry(tmp_path: Path) -> None:
    """Flips if a keyset read whose order picks one entry of a module mapping by a literal key
    (``CHANNEL_ORDERS[("date_created", False)]``, hub #26) is judged against every entry -- the ``status`` one its
    index's purpose never names -- rather than the entry it picks, while a pick by a run-time key still judges each;
    and if the picked entry is judged as anything but that entry (an entry whose index names another reader is still
    a finding)."""
    modules = {"channels.py": _ORDERS_PICKED_BY_A_CONSTANT, "v162.py": _ORDERS_PICKED_DDL}
    findings = no_whole_reads_findings(_config(tmp_path, modules))
    functions = {match.group(1) for match in (_HIT.match(finding) for finding in findings) if match is not None}
    assert not functions & {"Configs.page", "Configs.every"}, findings
    assert "Configs.every_by_status" in functions, findings
    assert not any("index-purpose-unused" in finding for finding in findings), findings


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
    literal or a module constant (``_held``: keyed; ``_by_state``: judged against that table, so non-key, reported at
    the caller that passes it), or resolves when one caller passes a value the walk cannot read (``_loose``)."""
    found = _symbols(_config(tmp_path, {"pkg/tables.py": _TABLES, "pkg/probe.py": _TABLE_PARAMETER}))
    assert not any(function == "_held" for function, _ in found), found
    assert ("callers", 'non-key-where "geos"') in found, found
    assert ("_loose", "unknown-predicate holds_any(table)") in found, found


_FORWARDED_SHAPES = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import KeyBatch, read_keyed, update_keyed

    SCHEDULES = "schedules"
    SUBSCRIPTIONS = "subscriptions"
    SCHEDULE_PK = ("conversation_id", "schedule_id")
    SUBSCRIPTION_PK = ("conversation_id", "subscription_id")
    OWNER_KEY = ("owner_id", "subscription_id")
    SCHEDULES_SCHEMA = TableSchema(name=SCHEDULES, primary_key=SCHEDULE_PK, columns=[])
    SUBSCRIPTIONS_SCHEMA = TableSchema(name=SUBSCRIPTIONS, primary_key=SUBSCRIPTION_PK, columns=[])


    async def _fenced(target, table, data, pk_cols, stamp):
        return await update_keyed(
            target, table, values={"x": 1}, by={**{column: data[column] for column in pk_cols}, "date_updated": stamp}
        )


    async def _fenced_loose(target, table, data, pk_cols, stamp):
        return await update_keyed(
            target, table, values={"x": 1}, by={**{column: data[column] for column in pk_cols}, "date_updated": stamp}
        )


    async def _fenced_unread(target, table, data, pk_cols, stamp):
        return await update_keyed(
            target, table, values={"x": 1}, by={**{column: data[column] for column in pk_cols}, "date_updated": stamp}
        )


    async def _one(pool, table, key, value, by):
        rows = await read_keyed(pool, table, keys=KeyBatch((key,), [(value,)], ("uuid",)), columns=["x"], by=by)
        return rows[0] if rows else None


    async def _one_loose(pool, table, key, value, by):
        rows = await read_keyed(pool, table, keys=KeyBatch((key,), [(value,)], ("uuid",)), columns=["x"], by=by)
        return rows[0] if rows else None


    async def save_schedule(target, data, stamp):
        return await _fenced(target, SCHEDULES, data, SCHEDULE_PK, stamp)


    async def save_subscription(target, data, stamp):
        return await _fenced(target, SUBSCRIPTIONS, data, SUBSCRIPTION_PK, stamp)


    async def save_crossed(target, data, stamp):
        return await _fenced_loose(target, SCHEDULES, data, OWNER_KEY, stamp)


    async def save_named(target, data, stamp, columns):
        return await _fenced_unread(target, SCHEDULES, data, columns, stamp)


    async def one_schedule(pool, conv, schedule):
        return await _one(pool, SCHEDULES, "schedule_id", schedule, {"conversation_id": conv})


    async def one_subscription(pool, conv, subscription):
        return await _one(pool, SUBSCRIPTIONS, "subscription_id", subscription, {"conversation_id": conv})


    async def one_anywhere(pool, subscription):
        return await _one(pool, SUBSCRIPTIONS, "subscription_id", subscription, {})


    async def one_by_owner(pool, owner, schedule):
        return await _one_loose(pool, SCHEDULES, "schedule_id", schedule, {"owner_id": owner})
"""


def test_a_forwarder_s_key_parameters_are_judged_by_what_each_call_site_passes(tmp_path: Path) -> None:
    """Flips if a forwarding helper's by= built over a key-column tuple its callers pass (a comprehension spliced
    into a mapping), its KeyBatch key column or its by= mapping, stays unknown when every call site passes a module
    tuple, a literal or a mapping with literal keys (``_fenced``, ``_one``: keyed at every site, nothing reported);
    if one site's values are judged crossed with another's (``save_schedule``'s table with ``save_subscription``'s
    key); if a site passing another table's key (``save_crossed``) or a mapping off the key (``one_by_owner``) is
    not reported at that caller, where the read is chosen; or if a site passing what the walk cannot read
    (``save_named``) is anything but unknown, at the forwarder, its caller calling a whole read."""
    found = _symbols(_config(tmp_path, {"pkg/forward.py": _FORWARDED_SHAPES}))
    quiet = {"_fenced", "_one", "save_schedule", "save_subscription", "one_schedule", "one_subscription"}
    assert not any(function in quiet for function, _ in found), found
    assert ("save_crossed", 'non-key-where "schedules"') in found, found
    assert ("one_by_owner", 'non-key-where "schedules"') in found, found
    # an empty mapping is read too: the key column alone, off the key's lead, judged (at the caller) not unknown
    assert any(function == "one_anywhere" and '"subscriptions"' in symbol for function, symbol in found), found
    assert not any(symbol.startswith("unknown-") and "read_keyed" in symbol for _, symbol in found), found
    assert not any(function in {"_fenced_loose", "_one_loose"} for function, _ in found), found
    assert ("_fenced_unread", "unknown-predicate update_keyed(by)") in found, found
    assert ("save_named", "calls-whole-read _fenced_unread") in found, found


_LOCAL_MAPPINGS = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import AnyOf, Sort, counts_by, keyset_page

    FIRES = "fires"
    FIRES_SCHEMA = TableSchema(name=FIRES, primary_key=("conversation_id", "fire_id"), columns=[])


    async def latest(pool, conversation_id, schedule_id, statuses=None):
        by: dict = {"conversation_id": conversation_id, "schedule_id": schedule_id}
        if statuses is not None:
            by["status"] = AnyOf(list(statuses))
        return await keyset_page(pool, FIRES, columns=["fire_id"], by=by, order=[Sort("fire_id", cast="uuid")], limit=1)


    async def counted(pool, conversation_id, webhook=None):
        by: dict = {} if webhook is None else {"webhook_id": webhook}
        return await counts_by(pool, FIRES, column="conversation_id", values=AnyOf([conversation_id]), by=by)


    async def by_schedule(pool, schedule_id, statuses=None):
        by = {"schedule_id": schedule_id}
        if statuses is not None:
            by["conversation_id"] = statuses
        return await keyset_page(pool, FIRES, columns=["fire_id"], by=by, order=[Sort("fire_id", cast="uuid")], limit=1)


    async def popped(pool, conversation_id, schedule_id):
        by = {"conversation_id": conversation_id, "schedule_id": schedule_id}
        by.pop("conversation_id")
        return await keyset_page(pool, FIRES, columns=["fire_id"], by=by, order=[Sort("fire_id", cast="uuid")], limit=1)
"""


def test_a_local_mapping_built_by_literal_keys_is_judged_by_each_shape_it_can_take(tmp_path: Path) -> None:
    """Flips if a by= mapping the function binds once to a literal (or either arm of a conditional) and adds literal
    keys to stays unknown (``latest``, ``counted``: keyed), if a key added only under a condition is read as always
    there (``by_schedule``: without it the read is off the key), or if a mapping changed any other way (``popped``)
    is judged rather than unknown."""
    found = _symbols(_config(tmp_path, {"pkg/fires.py": _LOCAL_MAPPINGS}))
    assert not any(function in {"latest", "counted"} for function, _ in found), found
    assert ("by_schedule", 'non-key-where "fires"') in found, found
    assert ("popped", "unknown-predicate keyset_page(by)") in found, found


_NESTED_MAPPINGS = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import AnyOf, Sort, keyset_page

    FIRES = "fires"
    FIRES_SCHEMA = TableSchema(name=FIRES, primary_key=("conversation_id", "fire_id"), columns=[])


    async def nested(pool, conversation_id=None, fire_id=None):
        by = (
            {"conversation_id": conversation_id}
            if fire_id is None
            else ({"conversation_id": conversation_id, "fire_id": fire_id} if conversation_id is not None else {})
        )
        return await keyset_page(
            pool, FIRES, columns=["fire_id"], by=by, order=[Sort("conversation_id"), Sort("fire_id")], limit=1
        )


    async def nested_off_key(pool, conversation_id=None, status=None):
        by = (
            {"conversation_id": conversation_id}
            if status is None
            else ({"status": status} if conversation_id is None else {"conversation_id": conversation_id})
        )
        return await keyset_page(pool, FIRES, columns=["fire_id"], by=by, order=[Sort("fire_id")], limit=1)


    async def nested_unread(pool, pins, conversation_id=None, fire_id=None):
        by = {"conversation_id": conversation_id} if fire_id is None else (pins if conversation_id else {})
        return await keyset_page(pool, FIRES, columns=["fire_id"], by=by, order=[Sort("fire_id")], limit=1)
"""


def test_a_local_mapping_bound_once_to_a_nested_conditional_is_judged_arm_by_arm(tmp_path: Path) -> None:
    """Flips if a by= the function binds once to a conditional of conditionals of literal mappings stays unknown
    rather than judged by each mapping it can be (``nested``: every arm keyed, ``{}`` a page in the key's order), if an arm off the key is not found (``nested_off_key``), or if a conditional
    with an arm the walk cannot read is judged rather than left unknown (``nested_unread``)."""
    found = _symbols(_config(tmp_path, {"pkg/fires.py": _NESTED_MAPPINGS}))
    assert not any(function == "nested" for function, _ in found), found
    assert ("nested_off_key", 'non-key-where "fires"') in found, found
    assert ("nested_unread", "unknown-predicate keyset_page(by)") in found, found


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


_VIEW_PARTS = """
    def needle_view_sql(schema, projections, moves, extra):
        parts = [f"SELECT p.state AS state, p.office_key AS office_key FROM main.{projections} AS p WHERE p.k = 'x'"]
        if moves is not None:
            parts.append(f"SELECT m.state, m.office_key FROM main.{moves} AS m")
        if extra is not None:
            parts.append(f"SELECT e.office_key FROM main.{extra} AS e")
        return f"CREATE VIEW {schema}.lsf_needle AS {' UNION ALL '.join(parts)}"

    def needle_dropped_sql(schema, projections, moves):
        parts = [f"SELECT p.state AS state FROM main.{projections} AS p"]
        if moves is not None:
            parts.append(f"SELECT upper(m.state) AS state FROM main.{moves} AS m")
        return f"CREATE VIEW {schema}.lsf_needle AS {' UNION ALL '.join(parts)}"

    def not_a_producer(conn, projections):
        return conn.execute(f"SELECT p.state FROM main.{projections} AS p").fetchall()
"""


def test_a_view_producer_s_parts_written_apart_are_judged_as_its_view(tmp_path: Path) -> None:
    """Flips if a declared view producer whose view is joined from SELECTs written as separate strings has those parts
    judged as reads (they read nothing: the view is judged at its readers by the producer's keys), or a part passing
    none of the producer's keys stops being a finding, or the same SELECT outside a producer passes (reports #48
    item 3)."""
    producers = {"needle_view_sql": (("office_key",), ("state",)), "needle_dropped_sql": (("state",),)}
    found = _symbols(
        _config(tmp_path, {"needle.py": _VIEW_PARTS}, view_producers=producers, local_relations={"*": ("state",)})
    )
    assert not any(function == "needle_view_sql" for function, _ in found), found
    assert any(
        function == "needle_dropped_sql" and symbol.startswith("view-key-not-passed") for function, symbol in found
    ), found
    assert any(function == "not_a_producer" for function, _ in found), found


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
    stays unknown, or if one element whose key the read does not bind (``runs``) passes, at the call site whose loop
    hands it on (``Pull.with_runs``)."""
    found = _symbols(_config(tmp_path, {"pkg/tables.py": _DECLARED_TABLES, "pkg/pull.py": _LOOPED}))
    assert not any(function in ("Store.holds_state", "Pull._held", "Pull.by_name") for function, _ in found), found
    assert any(function == "Pull.with_runs" and "runs" in symbol for function, symbol in found), found


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


_READINGS = """
    from threetears.core import keyed


    class FloorReadings:
        def read_by(self, clause):
            return [rate for rate in self.rates if rate.clause == clause]


    def rate_floor(clause, readings):
        return readings.read_by(clause)


    async def undeclared(conn):
        return await keyed.read_by(conn, "undeclared_widgets", by={"id": 1})
"""


def test_a_method_of_an_object_a_name_holds_named_as_a_keyed_helper_is_not_the_helper(tmp_path: Path) -> None:
    """Flips if a call of a method on an object a parameter holds (``readings.read_by(clause)``, evals'
    ``FloorReadings``) is judged as the keyed helper's call (an unknown-predicate ``read_by(table)``), or if the
    helper's own call through the imported module (``keyed.read_by``) stops being judged."""
    found = _symbols(_config(tmp_path, {"pkg/floor.py": _READINGS}))
    assert not any(function == "rate_floor" for function, _ in found), found
    assert any(function == "undeclared" for function, _ in found), found


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


@pytest.mark.parametrize("home", ["threetears.core.keyed", "threetears.interfaces.keyed"])
def test_a_step_over_a_change_relation_passes_and_one_over_a_table_is_the_whole_read_it_is(
    tmp_path: Path, home: str
) -> None:
    """Flips if a relational step over a registered or declared change relation is flagged, a delete by a change
    relation's keys is judged other than as the keyed delete it is, a step over a data table (whole, or as a semi's
    read side) passes, a change relation declared over a data table passes, or a set's ``union`` is judged as a step;
    the steps imported from core's keyed package or the interfaces package's alike."""
    found = _symbols(
        _config(
            tmp_path,
            {
                "steps.py": _STEPS.replace("threetears.core.keyed", home).replace(
                    "from threetears.core import keyed",
                    "from threetears.core import keyed"
                    if home.startswith("threetears.core")
                    else "from threetears.interfaces import keyed",
                )
            },
        )
    )
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


@pytest.mark.parametrize("package", ["threetears/core/keyed", "threetears/interfaces/keyed"])
def test_a_keyed_helper_s_relation_parameter_is_judged_at_its_call_sites_not_in_its_body(
    tmp_path: Path, package: str
) -> None:
    """Flips if a keyed helper's statement over its own relation parameter is flagged in its body (straight, through
    a local binding, a bare name in relation position, a private renderer or a field every in-module caller fills
    from a helper's parameter, a nested helper call on it), if literal-table SQL in a helper body passes, if a private
    renderer some other function reaches passes, or if a call site keying a helper by no key (``holds_any`` by a
    non-key column, ``update_from`` joined on one or reading a data table as its source) passes."""
    found = _symbols(
        _config(
            tmp_path,
            {f"{package}/rows.py": _HELPER_BODIES, "app.py": _HELPER_BODY_CALLERS},
        )
    )
    functions = {function for function, _ in found}
    covered = {"read_keyed", "holds_any", "read_past_marks", "_render", "keyset_page"}
    covered |= {"_Statement.render", "read_keyed_arrow", "read_joined", "counts_by_relations", "by_id", "from_chunk"}
    assert not functions & covered, found
    assert {"any_rows", "_shared", "by_note", "from_chunk_by_note", "from_the_table"} <= functions, found


_VISIBLE_ROWS_BODY = """
    from threetears.core.keyed import Fragment, keyset_rows

    async def visible_rows(conn, table, *, columns, user_id, namespace_type, customer_column, order, where=(), page=500):
        clauses = [Fragment("TRUE", ())]
        return await keyset_rows(conn, table, columns=columns, by={}, order=order, page=page, visible=clauses, where=where)

    async def grants_whole(conn):
        return await conn.fetch("SELECT * FROM role_assignments")
"""

_VISIBLE_ROWS_CALLERS = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import Sort
    from threetears.agent.acl import visible_rows

    EVAL_SETS = TableSchema(
        name="eval_sets",
        primary_key="id",
        columns=[],
        indexes=(Index("eval_sets_by_date", "date_created", "id", purpose="src/app.py::by_date"),),
    )

    async def by_date(conn, user_id):
        return await visible_rows(
            conn, "eval_sets", columns=["id", "date_created"], user_id=user_id, namespace_type="knowledge",
            customer_column="customer_id", order=[Sort("date_created"), Sort("id")],
        )

    async def by_name(conn, user_id):
        return await visible_rows(
            conn, "eval_sets", columns=["id", "name"], user_id=user_id, namespace_type="knowledge",
            customer_column="customer_id", order=[Sort("name"), Sort("id")],
        )
"""


def test_visible_rows_is_judged_at_its_call_sites_by_the_order_each_pages_in(tmp_path: Path) -> None:
    """Flips if acl's visible_rows' keyset read over its own table parameter is flagged in its body, if a call site
    paging in an order no index serves passes, if one paging through its purposed index is flagged, or if
    literal-table SQL in visible_rows' module passes."""
    found = _symbols(
        _config(
            tmp_path,
            {"threetears/agent/acl/visible_customers.py": _VISIBLE_ROWS_BODY, "app.py": _VISIBLE_ROWS_CALLERS},
        )
    )
    functions = {function for function, _ in found}
    assert "visible_rows" not in functions, found
    assert "by_date" not in functions, found
    assert {"by_name", "grants_whole"} <= functions, found


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


_READ_TEMPLATE = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import Sort, keyset_rows, read_row, read_value

    def epochs_schema(name: str = "scope_epochs"):
        return TableSchema(
            name=name,
            primary_key="scope",
            columns=[],
            indexes=(
                Index(
                    f"{name}_by_kind",
                    "kind",
                    "scope",
                    purpose="src/epochs.py::Epochs.kinds",
                    plan_test="tests/test_epochs.py::test_plan",
                ),
            ),
        )

    class Epochs:
        async def one(self, conn, scope):
            return await read_row(conn, self.table_name, by={"scope": scope})

        async def epoch(self, conn, table, scope):
            return await read_value(conn, table, by={"scope": scope}, column="epoch")

        async def kinds(self, conn, kind):
            return await keyset_rows(conn, self.table_name, columns=("scope",), by={"kind": kind}, order=[Sort("scope")])

        async def stale(self, conn, epoch):
            return await read_row(conn, self.table_name, by={"epoch": epoch})
"""

_READ_NO_TEMPLATE = """
    from threetears.core.keyed import read_row

    class Epochs:
        async def one(self, conn, scope):
            return await read_row(conn, self.table_name, by={"scope": scope})
"""


def test_a_read_of_a_table_its_caller_names_is_judged_by_its_module_s_template(tmp_path: Path) -> None:
    """Flips if a keyed read (read_row, read_value, keyset_rows) of a table the caller names at run time stays
    unknown where its module declares the template the table is built from (ruling #13): keyed by the template's key
    or its purposed index, a non-key read where neither leads; and unknown where the module declares none."""
    found = _symbols(_config(tmp_path, {"epochs.py": _READ_TEMPLATE}))
    functions = {function for function, _ in found}
    assert not functions & {"Epochs.one", "Epochs.epoch", "Epochs.kinds"}, found
    assert ("Epochs.stale", 'non-key-where "{}"') in found, found
    assert not any(symbol.startswith("unknown-predicate") for _, symbol in found), found
    bare = _symbols(_config(tmp_path / "bare", {"epochs.py": _READ_NO_TEMPLATE}))
    assert ("Epochs.one", "unknown-predicate read_row(table)") in bare, bare


_HELD_TABLE = """
    from threetears.core.keyed import KeyBatch, delete_keyed, read_keyed, read_row

    from pkg.tables import GEOS


    class Geos:
        def __init__(self, conn, *, table):
            self._conn = conn
            self._name = table

        async def read(self, ids):
            return await read_keyed(self._conn, self._name, keys=KeyBatch(("geo_id",), ids), columns=("geo_id",))

        async def drop(self, geo):
            return await delete_keyed(self._conn, self._name, by={"geo_id": geo})

        async def by_state(self, state):
            return await read_row(self._conn, self._name, by={"state": state})


    class Loose:
        def __init__(self, conn, table):
            self._conn = conn
            self._name = table

        async def read(self, ids):
            return await read_keyed(self._conn, self._name, keys=KeyBatch(("geo_id",), ids), columns=("geo_id",))


    class Moved:
        def __init__(self, conn, table):
            self._conn = conn
            self._name = table

        def move(self, table):
            self._name = table

        async def read(self, ids):
            return await read_keyed(self._conn, self._name, keys=KeyBatch(("geo_id",), ids), columns=("geo_id",))


    class Based:
        def __init__(self, conn, table):
            self._conn = conn
            self._name = table

        async def read(self, ids):
            return await read_keyed(self._conn, self._name, keys=KeyBatch(("geo_id",), ids), columns=("geo_id",))


    class Derived(Based):
        pass


    def build(conn, named):
        Geos(conn, table=GEOS)
        Loose(conn, named)
        Moved(conn, GEOS)
        Based(conn, GEOS)
        Derived(conn, named)
"""


def test_a_table_an_object_holds_from_its_constructor_is_the_table_every_construction_passes(tmp_path: Path) -> None:
    """Flips if a keyed read or write of a table an object keeps from its constructor (``self._name = table``)
    stays unknown when every construction passes a declared table's constant (``Geos``: keyed by its key, non-key by
    another column), or resolves when one construction passes a value the walk cannot read (``Loose``), the object
    rebinds it after (``Moved``) or a subclass may be constructed with another (``Based``)."""
    found = _symbols(_config(tmp_path, {"pkg/tables.py": _TABLES, "pkg/geos.py": _HELD_TABLE}))
    functions = {function for function, _ in found}
    assert not functions & {"Geos.read", "Geos.drop"}, found
    assert ("Geos.by_state", 'non-key-where "geos"') in found, found
    for unread in ("Loose.read", "Moved.read", "Based.read"):
        assert (unread, "unknown-predicate read_keyed(table)") in found, (unread, found)


_LOCAL_ARGUMENTS = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import AnyOf, Sort, counts_by, keyset_page, read_row

    FIRES = TableSchema(
        name="fires",
        primary_key="id",
        columns=[],
        indexes=(
            Index(
                "fires_by_schedule",
                "schedule_id",
                "fired_at",
                "id",
                purpose=("src/fires.py::latest", "src/fires.py::counted"),
                plan_test="tests/test_fires.py::test_plan",
            ),
        ),
    )

    async def latest(conn, schedule):
        by = {"schedule_id": schedule}
        order = [Sort("fired_at"), Sort("id")]
        return await keyset_page(conn, "fires", columns=("id",), by=by, order=order, limit=1)

    async def counted(conn, schedule, kinds):
        by = {"schedule_id": schedule}
        return await counts_by(conn, "fires", column="kind", values=AnyOf(kinds), by=by)

    async def loose(conn, kind):
        by = {"kind": kind}
        return await read_row(conn, "fires", by=by)

    async def widened(conn, schedule):
        by = {"schedule_id": schedule}
        by.pop("schedule_id")
        return await read_row(conn, "fires", by=by)

    async def rebound(conn, schedule, kind):
        by = {"schedule_id": schedule}
        if kind:
            by = {"kind": kind}
        return await read_row(conn, "fires", by=by)
"""


def test_a_helper_s_by_or_order_held_in_a_local_is_read_as_its_value(tmp_path: Path) -> None:
    """Flips if a by= or order= the function binds once to a literal (``by = {...}; keyset_page(..., by=by)``) stays
    unknown rather than judged as that literal, or if one the function changes after binding (``by.pop``) or binds
    twice is read as either value instead of staying unknown."""
    found = _symbols(_config(tmp_path, {"fires.py": _LOCAL_ARGUMENTS}))
    functions = {function for function, _ in found}
    assert not functions & {"latest", "counted"}, found
    assert ("loose", 'non-key-where "fires"') in found, found
    assert ("widened", "unknown-predicate read_row(by)") in found, found
    assert ("rebound", "unknown-predicate read_row(by)") in found, found


_KEY_HELD_IN_A_LOCAL = """
    from threetears.core.keyed import KeyBatch, read_keyed


    class Store:
        def _pk_columns(self, primary_key):
            return (primary_key,) if isinstance(primary_key, str) else tuple(primary_key)

        async def by_key(self, conn, table, primary_key, values):
            columns = self._pk_columns(primary_key)
            first = values[columns.index(primary_key[0])]
            return await read_keyed(conn, table, keys=KeyBatch(tuple(columns), [first]))

        async def by_other(self, conn, table, wanted, values):
            columns = list(wanted)
            return await read_keyed(conn, table, keys=KeyBatch(tuple(columns), values))

        async def rebound(self, conn, table, primary_key, wanted, values):
            columns = self._pk_columns(primary_key)
            if wanted:
                columns = list(wanted)
            return await read_keyed(conn, table, keys=KeyBatch(tuple(columns), values))

        async def changed(self, conn, table, primary_key, extra, values):
            columns = list(self._pk_columns(primary_key))
            columns.append(extra)
            return await read_keyed(conn, table, keys=KeyBatch(tuple(columns), values))
"""


def test_a_batch_s_columns_held_in_a_local_bound_from_the_key_are_the_key(tmp_path: Path) -> None:
    """Flips if a keyed batch whose columns a local holds, bound once from the table's key
    (``columns = self._pk_columns(primary_key)``; ``KeyBatch(tuple(columns), ...)``), stays unknown rather than keyed,
    or if a local bound from anything else, bound twice, or changed after binding (``columns.append``) is read as the
    key. Reading it (``columns.index``) changes nothing."""
    found = _symbols(_config(tmp_path, {"store.py": _KEY_HELD_IN_A_LOCAL}))
    functions = {function for function, _ in found}
    assert "Store.by_key" not in functions, found
    for unkeyed in ("Store.by_other", "Store.rebound", "Store.changed"):
        assert unkeyed in functions, (unkeyed, found)


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


_TIED_ON_A_SET_LEAD = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import AnyOf, Sort, keyset_page

    DATASOURCES = TableSchema(
        name="datasources",
        primary_key="id",
        columns=[],
        indexes=(
            Index("page_by_name", "kind", "name", "id", purpose=("src/listing.py::name_page", "src/listing.py::source_page")),
            Index("page_by_date_created", "kind", "date_created", "id", purpose="src/listing.py::source_page"),
        ),
    )

    async def source_page(pool, kinds, cols, cursor):
        return await keyset_page(
            pool, "datasources", columns=cols, by={"kind": AnyOf(kinds)},
            order=[Sort("kind"), Sort("date_created"), Sort("id")], limit=50, cursor=cursor,
        )

    async def name_page(pool, kind, cols):
        return await keyset_page(pool, "datasources", columns=cols, by={"kind": kind}, order=[Sort("name"), Sort("id")], limit=50)
"""


def test_two_indexes_tied_on_a_set_lead_credit_the_one_the_order_follows(tmp_path: Path) -> None:
    """Flips if a page ordered ``(kind, date_created, id)`` with ``kind`` bound to a set, beside two indexes both led
    by ``kind`` (the hub's datasources: ``(kind, name, id)`` and ``(kind, date_created, id)``), credits the first
    declared rather than the one whose columns after the bound lead follow the order past its bound columns: the
    ``(kind, date_created, id)`` index was left purpose-unused. The ``(kind, name, id)`` index's purpose naming the
    page too is then honestly unused: the page never reads it."""
    findings = no_whole_reads_findings(_config(tmp_path, {"listing.py": _TIED_ON_A_SET_LEAD}))
    unused = [finding for finding in findings if "index-purpose-unused" in finding]
    assert not any("page_by_date_created" in finding for finding in unused), unused
    assert [finding for finding in unused if "page_by_name" in finding and "source_page" in finding], findings
    found = _symbols(_config(tmp_path, {"listing.py": _TIED_ON_A_SET_LEAD}))
    assert not any(function in {"source_page", "name_page"} for function, _ in found), found


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
        key: tuple[str, ...] = ()


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


    async def changed_by_state(reader, source: SourceTable, state, held):
        return await keyed.read_changed(
            reader, source.relation, key=list(source.key), columns=["a"], held=held, hash_columns=["a"],
            by={"state": state},
        )


    async def changed_everywhere(reader, source: SourceTable, held):
        return await keyed.read_changed(
            reader, source.relation, key=list(source.key), columns=["a"], held=held, hash_columns=["a"]
        )


    async def changed_by_kind(reader, source: SourceTable, kind, held):
        return await keyed.read_changed(
            reader, source.relation, key=list(source.key), columns=["a"], held=held, hash_columns=["a"],
            by={"kind": kind},
        )


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
    the key passes, one bound by another column is the whole read it is, and one the walk cannot type stays unseen;
    a changed-rows read whose key is handed at run time too (``key=list(source.key)``) is judged by its pins."""
    found = _symbols(_config(tmp_path, {"produced.py": _RELATION_PRODUCED}, relation_producers=_PRODUCERS))
    functions = {function for function, _ in found}
    for keyed_read in ("by_state", "every_state", "every_marked", "derived_by_run", "Inputs.named", "changed_by_state"):
        assert keyed_read not in functions, found
    # a changed-rows read whose key is handed at run time (R-D, coordinator 2026-11-12): phase 1 judged by its pins
    assert ("changed_everywhere", "whole-select produced.sourcetable__relation") in found, found
    assert ("changed_by_kind", "non-key-where produced.sourcetable__relation") in found, found
    assert ("by_kind", "whole-aggregate produced.sourcetable__relation") in found, found
    assert ("derived_by_office", "whole-aggregate produced.derived") in found, found
    assert ("Inputs.every", "whole-aggregate produced.inputs___inputs") in found, found
    assert any(function == "untyped" and symbol.startswith("unknown-predicate") for function, symbol in found), found


def test_a_run_time_relation_with_no_producer_declared_stays_unseen(tmp_path: Path) -> None:
    """Flips if a run-time relation passes, or is judged against any key, with no producer declaring it."""
    found = _symbols(_config(tmp_path, {"produced.py": _RELATION_PRODUCED}))
    for function in ("by_state", "every_state", "every_marked", "derived_by_run", "Inputs.named"):
        assert any(f == function and s.startswith("unknown-predicate") for f, s in found), (function, found)


#: reports' counted relations (``aibots_reports.pod.lsf``, reports #55 asks 3 and 4): a producer's mapping of
#: relations handed to ``counts_by_relations``, its runs pinned by literal ids or by the ok runs the warehouse selects
_COUNTED = """
    from collections.abc import Sequence

    from threetears.core import keyed

    _OK = "ok"


    class LsfRelations:
        runs: str
        run_role: str


    def counted_relations(relations: LsfRelations) -> dict[str, str]:
        return {name: relations.relation(name) for name in counted_tables(relations)}


    class LsfWarehouse:
        def __init__(self, reader, *, relations: LsfRelations) -> None:
            self._reader = reader
            self._relations = relations

        async def _counted(self, run_ids: Sequence[int], ok: int):
            named = counted_relations(self._relations)
            if len(run_ids) <= 10:
                chosen = f"run_id IN ({', '.join(str(int(run_id)) for run_id in run_ids)})"
                items = len(run_ids) * len(named)
            else:
                role = f" AND run_role = '{self._relations.run_role}'" if self._relations.run_role else ""
                chosen = f"run_id IN (SELECT run_id FROM {self._relations.runs} WHERE status = '{_OK}'{role})"
                items = ok
            return await keyed.counts_by_relations(
                self._reader, named, column="run_id", where=[keyed.Fragment(chosen)], groups=items
            )

        async def named_runs(self, run_ids):
            chosen = f"run_id IN ({', '.join(str(int(run_id)) for run_id in run_ids)})"
            return await keyed.counts_by_relations(
                self._reader, counted_relations(self._relations), column="run_id", where=[keyed.Fragment(chosen)]
            )

        async def runs_of(self, run_ids):
            return await keyed.counts_by_relations(
                self._reader, counted_relations(self._relations), column="run_id", values=keyed.AnyOf(run_ids)
            )

        async def runs_at(self, at):
            chosen = f"run_id IN (SELECT run_id FROM {self._relations.runs} WHERE data_time = '{at}')"
            return await keyed.counts_by_relations(
                self._reader, counted_relations(self._relations), column="run_id", where=[keyed.Fragment(chosen)]
            )

        async def kinds_of(self, kinds):
            return await keyed.counts_by_relations(
                self._reader, counted_relations(self._relations), column="kind", values=keyed.AnyOf(kinds)
            )

        async def kinds_at(self, at):
            chosen = f"kind IN (SELECT kind FROM {self._relations.runs} WHERE data_time = '{at}')"
            return await keyed.counts_by_relations(
                self._reader, counted_relations(self._relations), column="run_id", where=[keyed.Fragment(chosen)]
            )
"""

_COUNTED_PRODUCERS = {"counted_relations": ("run_id",), "LsfRelations.runs": ("data_time",)}


def test_a_producer_s_mapping_of_relations_is_judged_per_relation_by_its_key(tmp_path: Path) -> None:
    """Flips if a mapping of relations a declared producer returns (``counted_relations(rels)``, called in place or
    bound to a local once) is not judged by the producer's key when handed to ``counts_by_relations`` (reports #55
    ask 3): runs named by values or literal ids pass, a count of another column is the whole read it is."""
    found = _symbols(_config(tmp_path, {"lsf.py": _COUNTED}, relation_producers=_COUNTED_PRODUCERS))
    functions = {function for function, _ in found}
    for keyed_read in ("LsfWarehouse.named_runs", "LsfWarehouse.runs_of", "LsfWarehouse.runs_at"):
        assert keyed_read not in functions, found
    assert ("LsfWarehouse.kinds_of", "whole-aggregate produced.counted_relations") in found, found
    assert ("LsfWarehouse.kinds_at", "whole-aggregate produced.counted_relations") in found, found


def test_a_fragment_chosen_by_a_branch_is_judged_in_each_arm(tmp_path: Path) -> None:
    """Flips if a ``Fragment`` over a local bound to one f-string or another (``_counted``'s literal ids, or the ok
    runs the warehouse selects) is dropped or left unseen instead of judged arm by arm (reports #55 ask 4): the
    literal-ids arm is keyed; the ok-runs arm keys the counted relations by the runs it selects, and its read of the
    runs relation by status, a column no key of it leads, is the non-key read it is."""
    found = _symbols(_config(tmp_path, {"lsf.py": _COUNTED}, relation_producers=_COUNTED_PRODUCERS))
    counted = {symbol for function, symbol in found if function == "LsfWarehouse._counted"}
    assert counted == {"non-key-where produced.lsfrelations__runs"}, found


def test_a_producer_s_mapping_with_no_producer_declared_stays_unseen(tmp_path: Path) -> None:
    """Flips if a producer's mapping of relations passes, or is judged against any key, with no producer declared."""
    found = _symbols(_config(tmp_path, {"lsf.py": _COUNTED}))
    for function in ("LsfWarehouse.named_runs", "LsfWarehouse.runs_of", "LsfWarehouse._counted"):
        assert any(f == function and s.startswith("unknown-") for f, s in found), (function, found)


#: a settle's relations, as ``aibots_reports.pod.settle`` writes them: a change's rows registered by a transient
#: producer (``_Work.put``) and read through ``self`` attributes, mappings of them, names bound more than once, lists
#: of steps and forwarders that hand a step on (``_Work.step``, ``_union``, ``_undropped``, ``_Lsf(work, sources)``)
_SETTLE_FLOW = """
    from collections.abc import Mapping, Sequence

    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import InlineDuckDB, run_inline

    RESULTS = TableSchema(name="results", primary_key=("state", "run_id"), columns=[])


    class _Work:
        def __init__(self, conn):
            self.cursor = conn

        def step(self, step):
            return run_inline(keyed.read_step_arrow(InlineDuckDB(self.cursor), step))

        def count(self, step):
            return run_inline(keyed.count_step(InlineDuckDB(self.cursor), step))

        def put(self, rows):
            name = keyed.change_relation("_settle_1")
            self.cursor.register(name, rows)
            return name


    def _step_rows(conn, step):
        return run_inline(keyed.read_step_arrow(InlineDuckDB(conn), step))


    def _union(parts: Sequence[str], *, keep_all: bool = False):
        return parts[0] if len(parts) == 1 else keyed.union(*parts, keep_all=keep_all)


    def _text(rel):
        if isinstance(rel, keyed.Step):
            return rel.sql
        return keyed.scan(rel).sql


    class _Lsf:
        def __init__(self, work: _Work, sources: Mapping[str, str], *, removed: Mapping[str, str] | None = None):
            self.work = work
            self.sources = sources
            self.removed: dict[str, str] = {name: rel for name, rel in (removed or {}).items()}
            self.let_go: dict[str, str] = {}
            self.dropped: str | None = None

        def _undropped(self, rel):
            if self.dropped is None:
                return rel
            return keyed.anti(rel, self.dropped, on=[keyed.Match("state")])

        def entries(self, rows):
            work, settled = self.work, None
            self.new = work.put(rows)
            self.annotated: dict[str, str] = {name: work.put(rows) for name in ("a", "b")}
            for name, gone in sorted(self.removed.items()):
                self.let_go[name] = work.put(gone)
            self.dropped = work.put(rows)
            touched = keyed.distinct(self.new, ["state"])
            if self.removed:
                touched = keyed.union(touched, keyed.distinct(self.removed["a"], ["state"]))
            parts = [keyed.scan(annotated, ["state"]) for name, annotated in sorted(self.annotated.items())]
            parts.append(keyed.semi(self._undropped(work.put(rows)), self.let_go["a"], on=[keyed.Match("state")]))
            parts += [keyed.scan(self.sources[name]) for name in sorted(self.sources)]
            runs = [self.new, work.put(rows)]
            work.step(touched)
            work.step(_union(parts))
            work.step(keyed.union(*runs, keep_all=True))
            _step_rows(work.cursor, keyed.distinct(self.new, ["state"]))
            return self.work.count(keyed.semi(self._undropped(self.dropped), work.put(rows), on=[keyed.Match("state")]))

        def holders(self):
            return self.work.step(keyed.scan(self.work.put(None))), _text(_union([self.new, self.dropped]))


    def _settle(work: _Work, sources: Mapping[str, str], removed: Mapping[str, str]) -> None:
        _Lsf(work, sources, removed=removed).entries(None)


    def settle(conn, written, gone):
        work = _Work(conn)
        sources = {name: work.put(rows) for name, rows in written.items()}
        sources["x"] = work.put(None)
        removed = {name: work.put(rows) for name, rows in gone.items()}
        _settle(work, sources, removed)


    def hands_a_table(conn):
        return _Work(conn).step("results")


    def hands_a_table_to_count(work: _Work):
        return work.count(keyed.distinct(work.put(None), ["state"])), _step_rows(work.cursor, "results")


    def unions_a_table(work: _Work):
        return work.step(_union([work.put(None), "results"]))


    def settles_over_a_table(work: _Work):
        _settle(work, {"a": "results"}, {})


    def steps_a_parameter(work: _Work, rel):
        return work.step(rel)


    def rebinds_to_a_table(work: _Work):
        rel = keyed.distinct(work.put(None), ["state"])
        rel = "results"
        return work.step(keyed.scan(rel))


    def appends_a_table(work: _Work):
        parts = [work.put(None)]
        parts.append("results")
        return work.step(keyed.union(*parts))
"""

#: the functions of :data:`_SETTLE_FLOW` that read only change relations
_FLOWED = {
    "_Work.step",
    "_Work.count",
    "_step_rows",
    "_union",
    "_text",
    "_Lsf._undropped",
    "_Lsf.entries",
    "_Lsf.holders",
    "_settle",
    "settle",
}


def test_a_step_over_a_producers_relation_held_by_a_name_an_attribute_or_a_mapping_reads_a_change_set(
    tmp_path: Path,
) -> None:
    """Flips if a step naming a declared transient producer's relation through a ``self`` attribute
    (``keyed.distinct(self.new, ...)``), a mapping attribute it fills (``self.let_go[name]``), a comprehension over one
    (``for name, annotated in sorted(self.annotated.items())``) or a receiver bound by a tuple assignment
    (``work, settled = self.work, self.settled``) is judged as a data table; or if ``self.removed[name]``, a mapping
    the constructor's caller fills with the producer's relations, is."""
    found = _symbols(_config(tmp_path, {"settle.py": _SETTLE_FLOW}, transient_producers=frozenset({"_Work.put"})))
    assert not {function for function, _ in found} & {"_Lsf.entries", "_Lsf.holders"}, found


def test_a_forwarder_handing_its_parameter_to_a_step_is_judged_at_each_call_by_what_the_call_hands_it(
    tmp_path: Path,
) -> None:
    """Flips if a private function, method or constructor that hands a parameter on to a step, or returns a step over
    it (``_Work.step``, ``_Work.count``, ``_step_rows``, ``_union``, ``_undropped``, ``_text``, ``_Lsf(work,
    sources)``), is flagged in its own body or makes its callers whole readers when each hands it a change relation;
    or if a caller handing one a data table (as the relation, in a list, or in the mapping a constructor keeps), or a
    public function's own parameter, passes."""
    found = _symbols(_config(tmp_path, {"settle.py": _SETTLE_FLOW}, transient_producers=frozenset({"_Work.put"})))
    functions = {function for function, _ in found}
    assert not functions & _FLOWED, found
    for caller in ("hands_a_table", "hands_a_table_to_count", "unions_a_table", "settles_over_a_table"):
        assert any(f == caller and not s.startswith("calls-whole-read") for f, s in found), (caller, found)
    assert any(f == "steps_a_parameter" for f, _ in found), found


def test_a_name_bound_more_than_once_or_a_list_of_steps_is_judged_by_every_value_it_holds(tmp_path: Path) -> None:
    """Flips if a local bound to a step and bound again (``touched = union(touched, ...)``), a list of steps a
    function appends to (``parts.append(semi(...))``, ``parts += [...]``) or a ``union(*runs)`` of change relations
    is flagged, or if any of them passes once one value it holds is a data table."""
    found = _symbols(_config(tmp_path, {"settle.py": _SETTLE_FLOW}, transient_producers=frozenset({"_Work.put"})))
    functions = {function for function, _ in found}
    assert "_Lsf.entries" not in functions, found
    assert {"rebinds_to_a_table", "appends_a_table"} <= functions, found


_MATCH_RULES = """
    from threetears.core import keyed

    STATE = "state"


    def let_go_on(table):
        on = [keyed.Match(STATE), keyed.Match("run_id"), keyed.Match(table.group, null_safe=True)]
        return on, [keyed.Fragment("t.model IS NOT DISTINCT FROM k.model")]
"""

#: a settle's deletes, as ``aibots_reports.pod.settle.apply_settled`` writes them: ``on=`` built by a function of the
#: module or of another, each copy table named ``main.<name>`` in a loop over the change's tables
_MATCH_DELETES = """
    from threetears.core import keyed
    from threetears.core.keyed import InlineDuckDB, run_inline

    from pkg.rules import STATE, let_go_on

    _GONE = "_gone"
    _KEYS = "_keys"
    _RUN = (STATE, "office_key", "run_id")
    SETTLE_KEYS = {"needle_rows": (STATE, "office_key", "run_id"), "runs": (STATE, "run_id")}
    _PANEL_KEY = (STATE,)


    def _alike(columns):
        return [keyed.Match(column, null_safe=True) for column in columns]


    def apply_settled(cursor, tables, settled):
        db = InlineDuckDB(cursor)
        for name, rows in sorted(settled.let_go.items()):
            cursor.register(keyed.change_relation(_GONE), rows)
            on, also = let_go_on(tables[name])
            run_inline(keyed.delete_matching(db, f"main.{name}", _GONE, on=on, also=also))
        for name, keys in sorted(settled.derived_keys.items()):
            table = f"main.{name}"
            cursor.register(keyed.change_relation(_KEYS), keys)
            run_inline(keyed.delete_matching(db, table, _KEYS, on=_alike(SETTLE_KEYS.get(name, _PANEL_KEY))))
            run_inline(keyed.delete_matching(db, table, _KEYS, on=_alike(_RUN)))


    def drops_by_run(cursor, settled):
        db = InlineDuckDB(cursor)
        for name in settled.derived_keys:
            cursor.register(keyed.change_relation(_KEYS), settled.derived_keys[name])
            run_inline(keyed.delete_matching(db, f"main.{name}", _KEYS, on=_alike(("run_id",))))
"""


def test_a_delete_matching_on_built_by_a_function_over_a_copy_table_named_at_run_time_is_judged_by_its_columns(
    tmp_path: Path,
) -> None:
    """Flips if ``delete_matching``'s ``on=`` built by a repo function (``_alike(columns)`` over a module constant or
    a module mapping's pick, ``on, also = let_go_on(table)`` from another module) or its ``table=`` named
    ``f"main.{name}"`` is unseen, so a delete by a copy's key passes; or if one by no key of the copy (``run_id``)
    passes."""
    found = _symbols(
        _config(
            tmp_path,
            {"pkg/rules.py": _MATCH_RULES, "pkg/settle.py": _MATCH_DELETES},
            local_relations={"*": (("state",),)},
            transient_relations=r"^_(?:gone|keys)$",
        )
    )
    functions = {function for function, _ in found}
    assert "apply_settled" not in functions, found
    assert "drops_by_run" in functions, found


#: a follower's stage deletes (reports #56, ``aibots_reports.pod.follower_parts.Stages._add_keys``): ``on=`` built by
#: a repo function over the followed table's declared key, a parameter annotated as one
_DECLARED_KEY_DELETES = """
    from collections.abc import Sequence
    from typing import Annotated

    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import InlineDuckDB, run_inline
    from threetears.interfaces.keyed import DeclaredKey, KeyColumns

    RESULTS = TableSchema(name="results", primary_key=("state", "run_id"), columns=[])
    _TYPED_KEYS = "_typed"


    def _keyed_on(key: Sequence[str]) -> list[keyed.Match]:
        return [keyed.Match(column) for column in key]


    class Stages:
        @staticmethod
        def _stage_name(stage: str) -> str:
            return stage

        def _add_keys(self, db, key: Annotated[Sequence[str], DeclaredKey], rows_stage: str) -> None:
            inline = InlineDuckDB(db)
            run_inline(keyed.delete_matching(inline, self._stage_name(rows_stage), _TYPED_KEYS, on=_keyed_on(key)))

        def drop_results(self, db, key: KeyColumns) -> None:
            run_inline(keyed.delete_matching(InlineDuckDB(db), "results", _TYPED_KEYS, on=_keyed_on(list(key))))

        def drop_by_columns(self, db, columns: Sequence[str]) -> None:
            run_inline(keyed.delete_matching(InlineDuckDB(db), "results", _TYPED_KEYS, on=_keyed_on(columns)))
"""


def test_a_delete_matching_on_built_from_a_declared_key_parameter_is_judged_as_the_key(tmp_path: Path) -> None:
    """Flips if ``delete_matching``'s ``on=`` built by a repo function over a parameter annotated as a declared key
    (``Annotated[Sequence[str], DeclaredKey]``, ``KeyColumns``, or ``list()`` of one) is unseen instead of judged as
    the key, as ``apply_stage(key=list(key))`` is (reports #56); or if one over a column list no declaration names
    passes."""
    found = _symbols(
        _config(
            tmp_path,
            {"follower_parts.py": _DECLARED_KEY_DELETES},
            transient_relations=r"^_typed$",
            transient_producers=frozenset({"Stages._stage_name"}),
        )
    )
    functions = {function for function, _ in found}
    assert "Stages._add_keys" not in functions, found
    assert "Stages.drop_results" not in functions, found
    assert any(f == "Stages.drop_by_columns" and s.startswith("unknown-predicate") for f, s in found), found


#: a tile's source rows (hub #59, ``aibots.hub.geo.loader.build_bbox_query``): a relation and its columns named by an
#: operator's layer declaration at run time, the statement declaring its key where it is written
_DECLARED_STATEMENT_KEY = """
    from dataclasses import dataclass
    from threetears.core.keyed import declared_key

    _MAX = 200000

    def _identifier(name):
        return f'"{name}"'

    @dataclass
    class GeoLayerConfig:
        table: str
        version_column: str
        feature_id: str
        bbox_columns: tuple[str, str, str, str]

    def build_bbox_query(layer: GeoLayerConfig) -> str:
        min_x, min_y, max_x, max_y = layer.bbox_columns
        return declared_key(
            f"SELECT a FROM {_identifier(layer.table)} "
            f"WHERE {_identifier(layer.version_column)} = $1 "
            f"AND {_identifier(min_x)} <= $4 AND {_identifier(max_x)} >= $2 "
            f"LIMIT {_MAX}",
            key=(layer.version_column,),
        )

    def by_box_only(layer: GeoLayerConfig) -> str:
        min_x, min_y, max_x, max_y = layer.bbox_columns
        return declared_key(
            f"SELECT a FROM {_identifier(layer.table)} WHERE {_identifier(min_x)} <= $1 LIMIT {_MAX}",
            key=(layer.version_column,),
        )

    def other_column(layer: GeoLayerConfig) -> str:
        return declared_key(
            f"SELECT a FROM {_identifier(layer.table)} WHERE {_identifier(layer.feature_id)} = $1",
            key=(layer.version_column,),
        )

    def undeclared(layer: GeoLayerConfig) -> str:
        return f"SELECT a FROM {_identifier(layer.table)} WHERE {_identifier(layer.version_column)} = $1"

    class DriverFeatureLoader:
        def __init__(self, *, layers):
            self._queries = {name: build_bbox_query(layer) for name, layer in layers.items()}
"""


def test_a_statement_declaring_its_key_is_judged_by_it(tmp_path: Path) -> None:
    """Flips if a statement over a run-time relation that declares its key at the statement (``declared_key(sql,
    key=(layer.version_column,))``, hub #59) is not judged by that key: a read pinning it passes (and so does the call
    chain into it), while one pinning another column, one pinning nothing, and the same statement undeclared stay
    findings."""
    found = _symbols(_config(tmp_path, {"loader.py": _DECLARED_STATEMENT_KEY}))
    functions = {function for function, _ in found}
    assert "build_bbox_query" not in functions, found
    assert "DriverFeatureLoader.__init__" not in functions, found
    assert {"by_box_only", "other_column", "undeclared"} <= functions, found


#: a run-time ``by=`` pin (reports ``RetainingDuckDB._delete_key``): a key's values handed as a declared key source
_RUN_TIME_PINS = """
    from collections.abc import Mapping
    from typing import Any

    from threetears.core.keyed import InlineDuckDB, delete_keyed, run_inline
    from threetears.interfaces.keyed import KeyValues


    class Retaining:
        def _delete_key(self, table: str, named: KeyValues) -> None:
            run_inline(delete_keyed(InlineDuckDB(self._db), table, by=named))

        def _delete_any(self, table: str, named: Mapping[str, Any]) -> None:
            run_inline(delete_keyed(InlineDuckDB(self._db), table, by=named))
"""


def test_a_by_holding_a_declared_key_source_s_values_is_the_key(tmp_path: Path) -> None:
    """Flips if a ``by=`` holding a declared key source's values (``named: KeyValues``, reports #55's retention
    delete) is not read as the key, or if a mapping no declaration names passes."""
    found = _symbols(_config(tmp_path, {"pins.py": _RUN_TIME_PINS}))
    functions = {function for function, _ in found}
    assert "Retaining._delete_key" not in functions, found
    assert any(f == "Retaining._delete_any" and s.startswith("unknown-predicate") for f, s in found), found


_UNTYPED_ARROW = """
    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import InlineDuckDB, run_inline

    RESULTS = TableSchema(name="results", primary_key=("state", "run_id"), columns=[])
    _CHANGED = keyed.change_relation("changed")


    def changed_rows(conn):
        return run_inline(keyed.read_step_arrow(InlineDuckDB(conn), keyed.distinct(_CHANGED, ["state"])))


    def every_result(conn):
        return run_inline(keyed.read_step_arrow(InlineDuckDB(conn), "results"))
"""


def test_a_step_read_into_arrow_with_duckdbs_own_types_is_judged_by_its_step(tmp_path: Path) -> None:
    """Flips if ``read_step_arrow`` on DuckDB with no ``types=`` is not judged by the relation it reads: a change
    relation's rows pass, a table read whole is the whole read it is."""
    found = _symbols(_config(tmp_path, {"untyped.py": _UNTYPED_ARROW}))
    functions = {function for function, _ in found}
    assert "changed_rows" not in functions, found
    assert "every_result" in functions, found


#: the settle rules' steps, as ``aibots_reports.pod.settle_rules`` / ``run_thinning`` / ``retention`` write them:
#: public functions, a method and a constructor of other modules, each building a step over the relation handed it
_PUBLIC_RULES = """
    from threetears.core import keyed

    _GONE = keyed.change_relation("_gone")


    def annotated_step(table, relation, *, figures=False):
        return keyed.scan(relation, ["state", "run_id"], where=[keyed.Fragment("t.instant IS NOT NULL")])


    def runs_step(table, rows):
        return keyed.grouped(rows, by=["state", "run_id"], aggregates=[keyed.largest("instant")])


    class Thinning:
        def kept_runs_step(self, relation):
            return keyed.windowed(relation, partition=["state"], order=[keyed.Sort("instant")], keep="rank <= 3")


    class Eviction:
        def __init__(self, table, target):
            self.table = table
            self.target = target

        def let_go(self, state):
            return keyed.semi(self.target, _GONE, on=[keyed.Match("state")])
"""

_PUBLIC_CALLERS = """
    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import InlineDuckDB, run_inline

    from pkg.rules import Eviction, Thinning, annotated_step, runs_step

    RESULTS = TableSchema(name="results", primary_key=("state", "run_id"), columns=[])
    _THINNING = Thinning()


    def settled(conn, rows):
        changed = keyed.change_relation("_settle_1")
        conn.register(changed, rows)
        db = InlineDuckDB(conn)
        run_inline(keyed.read_step_arrow(db, keyed.distinct(annotated_step(None, changed), ["state"])))
        run_inline(keyed.read_step_arrow(db, runs_step(None, changed)))
        run_inline(keyed.read_step_arrow(db, _THINNING.kept_runs_step(changed)))
        return Eviction(None, changed).let_go("S")


    def over_the_table(conn):
        return run_inline(keyed.read_step_arrow(InlineDuckDB(conn), runs_step(None, "results")))


    def evicts_the_table():
        return Eviction(None, "results")
"""


def test_a_public_forwarder_of_another_module_is_judged_at_each_call_where_the_repo_says_only_it_calls_them(
    tmp_path: Path,
) -> None:
    """Flips if, with ``public_forwarders``, a public function, method or constructor of one module building a step
    over the relation handed it (``annotated_step``, ``runs_step``, ``Thinning.kept_runs_step``, ``Eviction(target)``
    whose ``let_go`` reads ``self.target``) is flagged in its body, or its callers handing it a change relation are;
    or if a caller handing one a data table passes; or if without it (a library's public functions, called from
    repos the walk never reads) a step over a public function's parameter is not judged in its body."""
    modules = {"pkg/rules.py": _PUBLIC_RULES, "pkg/settle.py": _PUBLIC_CALLERS}
    found = _symbols(_config(tmp_path, modules, public_forwarders=True))
    functions = {function for function, _ in found}
    assert not functions & {"annotated_step", "runs_step", "Thinning.kept_runs_step", "Eviction.let_go", "settled"}, (
        found
    )
    assert {"over_the_table", "evicts_the_table"} <= functions, found
    library = {function for function, _ in _symbols(_config(tmp_path, modules))}
    assert {"annotated_step", "runs_step", "Thinning.kept_runs_step", "Eviction.let_go"} <= library, library


def test_a_walk_judges_the_same_when_the_parse_cache_keeps_no_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the walk's findings change when the shared parse cache hands out a new tree on every call (what it
    does for every file past its memory bound): the walk's passes key a forwarder, a builder or a module's tuples by
    the nodes they found them at, so a pass handed a second tree of the same module stops finding them (reports'
    gate went 55 -> 374 findings, 3tears' 150 -> 193, once the cache was bounded)."""
    modules = {"pkg/rules.py": _PUBLIC_RULES, "pkg/settle.py": _PUBLIC_CALLERS, **_STEP_BUILDERS}
    kept = _symbols(_config(tmp_path, modules, public_forwarders=True))
    monkeypatch.setattr(walker, "parse_python_path", lambda path: ast.parse(path.read_text(encoding="utf-8")))
    assert _symbols(_config(tmp_path, modules, public_forwarders=True)) == kept


#: a local in each module sharing a public forwarder's name: a local, never the function used as a value
_SHADOWING_LOCALS = """

    def views(held):
        runs_step = [("runs", "SELECT 1")] if "runs" in held else []
        for name, view in [*runs_step]:
            held.add(name)
        return [annotated_step for annotated_step in held], lambda runs_step: runs_step
"""

#: the same name read as the module's function, inside a function (handed on as a value)
_HANDED_ON = """

    def handlers():
        return [runs_step]
"""


def test_a_local_sharing_a_public_forwarder_s_name_does_not_stop_its_calls_being_judged(tmp_path: Path) -> None:
    """Flips if a local (an assignment, a comprehension target, a lambda's parameter) in either module that shares a
    public forwarder's name (``runs_step = [...]`` in a function of the callers' module; ``namesakes = [...]`` beside
    reports' ``report_data.namesakes``) is read as that function used as a value, so the forwarder is judged in its
    body as written (unknown) instead of at each call; or if the function itself read as a value inside a function
    (``[runs_step]``) no longer keeps it judged in its body."""
    shadowed = {
        "pkg/rules.py": _PUBLIC_RULES + _SHADOWING_LOCALS,
        "pkg/settle.py": _PUBLIC_CALLERS + _SHADOWING_LOCALS,
    }
    functions = {function for function, _ in _symbols(_config(tmp_path, shadowed, public_forwarders=True))}
    assert not functions & {"annotated_step", "runs_step", "settled"}, functions
    assert {"over_the_table", "evicts_the_table"} <= functions, functions
    handed = {"pkg/rules.py": _PUBLIC_RULES, "pkg/settle.py": _PUBLIC_CALLERS + _HANDED_ON}
    functions = {function for function, _ in _symbols(_config(tmp_path, handed, public_forwarders=True))}
    assert "runs_step" in functions, functions


_NEW_STEPS = """
    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import InlineDuckDB, run_inline

    RESULTS = TableSchema(name="results", primary_key=("state", "run_id"), columns=[])
    _ROWS = keyed.change_relation("_rows")
    _ROUTES = keyed.change_relation("_routes")
    _STAGE = keyed.change_relation("__stage__results")


    def over_the_change(conn):
        db = InlineDuckDB(conn)
        on = [keyed.Match("state")]
        fresh = keyed.joined(_ROWS, _ROUTES, on=on, columns=["state", keyed.Selected("k.part", "part")], left=True)
        run_inline(keyed.read_step_arrow(db, fresh, order=[keyed.Sort("part")], limit=1))
        run_inline(keyed.materialize(db, _STAGE, keyed.anti(_ROWS, _ROUTES, on=on), temp=True))
        found = run_inline(keyed.first_difference(db, _ROWS, _ROUTES, on=on, compare=[("v", "v")], order=["state"],
                                                  where=[keyed.Fragment("t.v IS NULL")]))
        return run_inline(keyed.any_step(db, keyed.semi(_ROWS, _ROUTES, on=on))), found


    def joins_the_table(conn):
        on = [keyed.Match("state")]
        return run_inline(keyed.read_step(InlineDuckDB(conn), keyed.joined(_ROWS, "results", on=on, columns=["state"])))


    def any_result(conn):
        return run_inline(keyed.any_step(InlineDuckDB(conn), "results"))


    def first_of_the_table(conn):
        return run_inline(keyed.read_step(InlineDuckDB(conn), "results", limit=1))


    def materializes_the_table(conn):
        return run_inline(keyed.materialize(InlineDuckDB(conn), "_copy", "results"))


    def overwrites_the_table(conn):
        return run_inline(keyed.materialize(InlineDuckDB(conn), "results", _ROWS))
"""


def test_a_join_an_any_a_limited_read_a_table_made_and_a_bounded_difference_are_judged_by_their_steps(
    tmp_path: Path,
) -> None:
    """Flips if ``joined``, ``any_step``, ``read_step(limit=)``, ``materialize`` or ``first_difference(where=)`` over
    change relations into a stage is flagged; or if one reading a table whole (a join's other side, any row of it, its
    first row, a table made of it) or ``materialize`` replacing a data table passes."""
    found = _symbols(_config(tmp_path, {"steps.py": _NEW_STEPS}, transient_relations=r"^_copy$"))
    functions = {function for function, _ in found}
    assert "over_the_change" not in functions, found
    for whole in ("joins_the_table", "any_result", "first_of_the_table", "materializes_the_table"):
        assert whole in functions, (whole, found)
    assert any(f == "overwrites_the_table" and s.startswith("whole-delete") for f, s in found), found


_RANKED_STEPS = """
    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import InlineDuckDB, run_inline

    RESULTS = TableSchema(name="results", primary_key=("state", "run_id"), columns=[])
    _ROWS = keyed.change_relation("_rows")
    _NATURAL = keyed.Ranked("county", "length(v), v")


    def ranks_the_change(conn):
        db = InlineDuckDB(conn)
        first = [keyed.Sort("votes", descending=True, nulls="last")]
        rows = run_inline(keyed.read_step(db, keyed.ranked_order(_ROWS, [_NATURAL], first=first, limit=5)))
        return rows, run_inline(keyed.read_step(db, keyed.value_ranks(_ROWS, "county", "v")))


    def ranks_the_table(conn):
        return run_inline(keyed.read_step(InlineDuckDB(conn), keyed.ranked_order("results", [_NATURAL])))


    def ranks_the_table_s_values(conn):
        return run_inline(keyed.read_step(InlineDuckDB(conn), keyed.value_ranks("results", "county", "v")))
"""


def test_a_ranked_order_and_its_value_ranks_are_judged_by_the_relation_they_order(tmp_path: Path) -> None:
    """Flips if ``ranked_order`` or ``value_ranks`` over a change relation is flagged, or one over a data table (read
    whole to order it) passes."""
    config = _config(tmp_path, {"ranked.py": _RANKED_STEPS})
    functions = {function for function, _ in _symbols(config)}
    assert "ranks_the_change" not in functions, functions
    assert {"ranks_the_table", "ranks_the_table_s_values"} <= functions, functions
    helped = use_the_helper_findings(config)
    assert not [line for line in helped if "ranks_the_change" in line], helped


#: one statement assembled across a class's methods, as ``aibots_reports.tableau.evaluate._PanelQuery`` and
#: ``aibots_reports.comparison._StateQuery`` write it: ``run`` defines ``stage0`` and embeds ``_fixed``'s ``fixed``
#: CTE, which reads ``stage0``; ``sql`` embeds ``_race`` (defining ``o_<race>``) and ``model_cte`` (reading it)
_ASSEMBLED_CTES = """
    STATE = "join_state_code"


    class _PanelQuery:
        def _fixed(self, where):
            kept = "SELECT * FROM stage0" + (f" WHERE {where}" if where else "")
            return f"fixed AS ({kept})"

        def run(self, snapshot, states, where, inlined):
            stage0 = f"SELECT * FROM {snapshot.relation(self.table)} WHERE {STATE} IN ({states})"
            base = "SELECT * FROM fixed" + (f" WHERE {where}" if where else "")
            sql = f"WITH stage0 AS {inlined}({stage0}), {self._fixed(where)}, base AS ({base}) SELECT * FROM base"
            return snapshot.backend.execute_query(sql)


    class _StateQuery:
        def _race(self, snapshot, race, states):
            return f"o_{race} AS (SELECT {STATE}, office_key FROM {snapshot.relation('races')} WHERE {STATE} IN ({states}))"

        def model_cte(self, race):
            return f"l_{race} AS (SELECT office_key FROM o_{race})"

        def sql(self, snapshot, race, states):
            ctes = [self._race(snapshot, race, states), self.model_cte(race)]
            return snapshot.backend.execute_query(f"WITH {', '.join(ctes)} SELECT * FROM l_{race}")


    def reads_stage0_alone(snapshot, suffix):
        return snapshot.backend.execute_query(f"SELECT * FROM stage0{suffix}")
"""


def test_a_name_a_cte_of_the_statement_its_code_assembles_is_that_cte(tmp_path: Path) -> None:
    """Flips if a fragment reading a CTE that the statement it is assembled into defines -- in a method its own
    calls or is called by (``_fixed`` reading ``run``'s ``stage0``; ``run`` reading ``_fixed``'s ``fixed``; a
    templated ``o_{race}`` another method writes) -- is judged as a table, by either rule; or if a function no call
    joins to the definition reads it and passes."""
    config = _config(tmp_path, {"assembled.py": _ASSEMBLED_CTES}, local_relations={"*": ("join_state_code",)})
    found = _symbols(config)
    functions = {function for function, _ in found}
    assert not functions & {"_PanelQuery._fixed", "_PanelQuery.run", "_StateQuery.model_cte", "_StateQuery.sql"}, found
    assert ("reads_stage0_alone", "unknown-predicate stage0") in found, found
    helped = use_the_helper_findings(config)
    assert not [line for line in helped if "_PanelQuery" in line or "_StateQuery" in line], helped


#: a reader's steps over a copy's run-time relation, already quoted (``Quoted(snapshot.relation(NAME))``)
_QUOTED_READS = """
    from threetears.core import keyed
    from threetears.core.keyed import InlineDuckDB, run_inline

    LONG_ROWS = "lsf_rows"


    class _Work:
        def put(self, rows):
            return keyed.change_relation("_settle_1")


    def one_state(snapshot, state):
        rows = keyed.Quoted(snapshot.relation(LONG_ROWS))
        step = keyed.scan(rows, where=[keyed.Fragment("t.join_state_code = $1", (state,))])
        return run_inline(keyed.read_step(InlineDuckDB(snapshot.cursor), step))


    def every_state(snapshot):
        return run_inline(keyed.read_step(InlineDuckDB(snapshot.cursor), keyed.scan(keyed.Quoted(snapshot.relation(LONG_ROWS)))))


    def every_office(snapshot, office):
        where = [keyed.Fragment("t.office = $1", (office,))]
        step = keyed.distinct(keyed.Quoted(snapshot.relation(LONG_ROWS)), ["office"], where=where)
        return run_inline(keyed.read_step(InlineDuckDB(snapshot.cursor), step))


    def staged(work: _Work, conn, rows):
        return run_inline(keyed.read_step(InlineDuckDB(conn), keyed.scan(keyed.Quoted(work.put(rows)))))
"""


def test_a_relation_already_quoted_is_judged_as_the_relation_it_names(tmp_path: Path) -> None:
    """Flips if a step over ``Quoted(...)`` is trusted for being quoted: a copy relation read whole, or narrowed by a
    column no copy key leads with, passes; or if one read by the copy's key (a step's ``where=`` on ``t``), or a
    quoted change set, is flagged."""
    found = _symbols(
        _config(
            tmp_path,
            {"quoted.py": _QUOTED_READS},
            local_relations={"*": ("join_state_code",)},
            transient_producers=frozenset({"_Work.put"}),
        )
    )
    functions = {function for function, _ in found}
    assert not functions & {"one_state", "staged"}, found
    assert {"every_state", "every_office"} <= functions, found


_SCRATCH_STEPS = """
    from threetears.core import keyed
    from threetears.core.keyed import InlineDuckDB, run_inline

    _SCRATCH = "_export"
    STATE_COLUMN = "state"


    def quote_identifier(name):
        return '"' + name + '"'


    def runs_step(relation, state):
        return keyed.grouped(
            relation,
            by=["run_id"],
            aggregates=[keyed.Selected("max(t.at)", "instant")],
            where=[keyed.Fragment(f"t.{quote_identifier(STATE_COLUMN)} = $1", (state,))],
        )


    class Retaining:
        def _scratch(self, table):
            return f"{_SCRATCH}.{quote_identifier(table)}"

        def block_runs(self, table, state):
            runs = runs_step(f"main.{table}", state)
            return run_inline(keyed.read_step(InlineDuckDB(self._db), runs))

        def exported(self, table, state):
            scoped = keyed.scan(f"{_SCRATCH}.{table}", where=[keyed.Fragment(f"t.{STATE_COLUMN} = $1", (state,))])
            return run_inline(keyed.read_step(InlineDuckDB(self._db), scoped))

        def exported_whole(self, table):
            return run_inline(keyed.read_step(InlineDuckDB(self._db), keyed.scan(f"{_SCRATCH}.{table}")))

        def options(self, table, states):
            pinned = keyed.Fragment(f"t.{STATE_COLUMN} IN ({', '.join('?' for _ in states)})", tuple(states))
            return run_inline(keyed.read_step(InlineDuckDB(self._db), keyed.scan(f"main.{table}", where=[pinned])))

        def whole(self, table):
            return run_inline(keyed.read_step(InlineDuckDB(self._db), keyed.scan(f"main.{table}")))

        def unpinned(self, table, kind):
            narrowed = keyed.Fragment(f"t.{quote_identifier('kind')} = $1", (kind,))
            return run_inline(keyed.read_step(InlineDuckDB(self._db), keyed.scan(f"main.{table}", where=[narrowed])))
"""


def test_an_f_string_relation_and_an_f_string_fragment_are_read_as_their_text(tmp_path: Path) -> None:
    """Flips if a step's relation written as an f-string (``f"main.{table}"``, ``f"{_SCRATCH}.{table}"``) stays
    unknown rather than read as the relation it names (a local copy's, judged against the copy's default key; a
    declared transient producer's spelling, a change set), or if a ``where=`` fragment written as an f-string pins
    nothing rather than what its text pins (R6, R7); a read of the copy whole, or pinned by a column no key leads
    with, is still a finding."""
    found = _symbols(
        _config(
            tmp_path,
            {"retention.py": _SCRATCH_STEPS},
            transient_producers=frozenset({"Retaining._scratch"}),
            local_relations={"*": ("state", "run_id")},
            public_forwarders=True,
        )
    )
    functions = {function for function, _ in found}
    assert not functions & {"Retaining.block_runs", "Retaining.exported", "Retaining.exported_whole"}, found
    assert not functions & {"Retaining.options"}, found
    assert "Retaining.whole" in functions, found
    assert "Retaining.unpinned" in functions, found
    assert not any(symbol.startswith("unknown-predicate") for _, symbol in found), found


_FORWARDED_WHERE = """
    from threetears.core import keyed
    from threetears.core.keyed import InlineDuckDB, run_inline

    STATE_COLUMN = "state"


    def annotated_step(relation, where=()):
        return keyed.scan(relation, where=list(where))


    def runs_step(relation, where=()):
        rows = annotated_step(relation, where)
        return keyed.grouped(rows, by=["run_id"], aggregates=[keyed.row_count("n")])


    def one_state(db, table, state):
        where = [keyed.Fragment(f"t.{STATE_COLUMN} = $1", (state,))]
        return run_inline(keyed.read_step(InlineDuckDB(db), runs_step(f"main.{table}", where)))


    def one_state_inline(db, table, state):
        step = runs_step(f"main.{table}", [keyed.Fragment("t.state = $1", (state,))])
        return run_inline(keyed.read_step(InlineDuckDB(db), step))


    def every_state(db, table):
        return run_inline(keyed.read_step(InlineDuckDB(db), runs_step(f"main.{table}")))


    def by_kind(db, table, kind):
        where = [keyed.Fragment("t.kind = $1", (kind,))]
        return run_inline(keyed.read_step(InlineDuckDB(db), runs_step(f"main.{table}", where)))


    def _made(db, table, step):
        return run_inline(keyed.materialize(InlineDuckDB(db), table, step, temp=True))


    def made_keys(db, flag):
        one = keyed.scan(keyed.change_relation("changed"), ["k"])
        return _made(db, "_named_keys", one if flag else keyed.union(one, one, keep_all=True))


    def made_whole(db, table):
        return _made(db, "_named_keys", f"main.{table}")
"""


def test_a_forwarder_s_steps_are_judged_at_its_call_with_what_the_call_passes(tmp_path: Path) -> None:
    """Flips if a public forwarder whose step reads the relation it is handed under the where= it is handed
    (``runs_step`` -> ``annotated_step``, R8) is judged at its call as a whole read of the relation, dropping the
    call's pins (a list literal, a local bound once to one), or if a call passing no pins, or pins on a column no key
    leads with, passes."""
    found = _symbols(
        _config(
            tmp_path,
            {"runs.py": _FORWARDED_WHERE},
            local_relations={"*": ("state", "run_id")},
            public_forwarders=True,
            transient_relations=r"^_named_keys$",
        )
    )
    functions = {function for function, _ in found}
    assert not functions & {"one_state", "one_state_inline", "annotated_step", "runs_step", "made_keys", "_made"}, found
    assert {"every_state", "by_kind", "made_whole"} <= functions, found
    assert not any(symbol.startswith("unknown-predicate") for _, symbol in found), found


_WRITES_FROM = """
    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import InlineDuckDB, run_inline

    GEOS = TableSchema(name="geos", primary_key="geo_id", columns=[])


    def _computed_source(relation, state):
        return keyed.scan(relation, where=[keyed.Fragment("t.state = $1", (state,))])


    class Writer:
        def compute(self, state):
            source = _computed_source("_partition_chunk", state)
            return run_inline(
                keyed.update_from(InlineDuckDB(self._db), "__stage__x", key=["geo_id"], source=source, columns=["n"])
            )

        def compute_inline(self, state):
            return run_inline(
                keyed.update_from(
                    InlineDuckDB(self._db),
                    "__stage__x",
                    key=["geo_id"],
                    source=_computed_source("_partition_chunk", state),
                    columns=["n"],
                )
            )

        def add_rows(self, stage, key):
            return run_inline(
                keyed.apply_stage(InlineDuckDB(self._db), "__stage__x", "_partition_chunk", key=key, columns=["a"])
            )

        def replace(self, table):
            return run_inline(
                keyed.apply_stage(
                    InlineDuckDB(self._db), "geos", "_partition_chunk", key=self._keys[table], columns=["a"]
                )
            )
"""


def test_a_write_from_a_forwarder_s_step_or_by_a_run_time_key_is_judged(tmp_path: Path) -> None:
    """Flips if update_from's source passed straight as a forwarder's call stays unknown where the same call bound to
    a local is judged (R8), or if apply_stage / update_from by a key named at run time (``key=key``,
    ``self._keys[table]``) stays unknown rather than judged as a key the walk cannot see (R9): onto a change set's
    stage it writes nothing whole; onto a data table it is still a finding."""
    found = _symbols(
        _config(
            tmp_path,
            {"writer.py": _WRITES_FROM},
            transient_relations=r"^(?:__stage__\w+|_partition_chunk)$",
            public_forwarders=True,
        )
    )
    functions = {function for function, _ in found}
    assert not functions & {"Writer.compute", "Writer.compute_inline", "Writer.add_rows", "_computed_source"}, found
    assert "Writer.replace" in functions, found
    assert not any(symbol.startswith("unknown-predicate") for _, symbol in found), found


_HANDED_TARGET = """
    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.keyed import InlineDuckDB, run_inline

    GEOS = TableSchema(name="geos", primary_key="geo_id", columns=[])


    class Stages:
        def _add_rows(self, stage, key):
            run_inline(keyed.apply_stage(InlineDuckDB(self._db), stage, "_page_rows", key=key, columns=["a"]))

        def _set_from(self, table, key):
            source = keyed.scan("_page_rows")
            run_inline(keyed.update_from(InlineDuckDB(self._db), table, key=key, source=source, columns=["n"]))

        def into_stage(self):
            self._add_rows("__stage__geos", ["geo_id"])

        def into_geos(self):
            self._add_rows("geos", ["geo_id"])

        def into_geos_by_name(self):
            self._add_rows("geos", ["name"])

        def set_geos(self):
            self._set_from("geos", ["geo_id"])

        def set_geos_by_name(self):
            self._set_from("geos", ["name"])

        def _replace(self, table):
            run_inline(
                keyed.apply_stage(InlineDuckDB(self._db), table, "_page_rows", key=self._keys[table], columns=["a"])
            )

        def replace_stage(self):
            self._replace("__stage__geos")

        def replace_geos(self):
            self._replace("geos")
"""


def test_a_write_target_a_forwarder_is_handed_is_judged_at_its_call(tmp_path: Path) -> None:
    """Flips if apply_stage / update_from onto a target named by the forwarder's parameter (``Stages._add_rows``'
    ``stage``) stays an unknown "{}" in the forwarder rather than judged at each call with the target and key the call
    passes (R9): onto a change set's stage it writes nothing whole, onto a data table by its key it is keyed, and onto
    a data table by a column no key leads with, or by a key the forwarder reads off its ``self``, it is still a
    finding, at the call, as the write it is."""
    found = _symbols(
        _config(
            tmp_path,
            {"stages.py": _HANDED_TARGET},
            transient_relations=r"^(?:__stage__\w+|_page_rows)$",
            public_forwarders=True,
        )
    )
    functions = {function for function, _ in found}
    assert not functions & {"Stages._add_rows", "Stages._set_from", "Stages.into_stage", "Stages.into_geos"}, found
    assert not functions & {"Stages.set_geos", "Stages._replace", "Stages.replace_stage"}, found
    assert {"Stages.into_geos_by_name", "Stages.set_geos_by_name", "Stages.replace_geos"} <= functions, found
    assert not any("unknown" in symbol for _, symbol in found), found
    # a target write is judged as the write it is, never as a whole read of the table it writes
    assert not any(symbol.startswith("whole-select") for _, symbol in found), found


_HANDED_DERIVED = """
    from threetears.core import keyed
    from threetears.core.keyed import InlineDuckDB, run_inline

    GEOS = "geos"


    def _relation(snapshot, name):
        return keyed.Quoted(snapshot.relation(name)) if snapshot.states is None else keyed.Derived(snapshot.relation(name))


    def by_geo(db, snapshot, geo):
        rel = keyed.Derived(snapshot.relation(GEOS))
        step = keyed.scan(rel, where=[keyed.Fragment("t.geo_id = $1", (geo,))])
        return run_inline(keyed.read_step(InlineDuckDB(db), step))


    def every_geo(db, snapshot):
        return run_inline(keyed.read_step(InlineDuckDB(db), keyed.scan(keyed.Derived(snapshot.relation(GEOS)))))


    def handed(db, sql):
        return run_inline(keyed.read_step(InlineDuckDB(db), keyed.scan(keyed.Derived(sql))))
"""


def test_a_derived_select_handed_as_a_copy_s_relation_is_read_as_that_relation(tmp_path: Path) -> None:
    """Flips if ``Derived(snapshot.relation(NAME))`` (a limited copy's relation, a parenthesized select the code does
    not write) is passed as a derived select judged elsewhere, or stays unknown, rather than read as the copy relation
    it names, keyed by its key and whole when it pins nothing (R10); a derived select of text the walk cannot read is
    no relation it can name."""
    found = _symbols(_config(tmp_path, {"read.py": _HANDED_DERIVED}, local_relations={"geos": ("geo_id",)}))
    functions = {function for function, _ in found}
    assert "by_geo" not in functions, found
    assert "every_geo" in functions, found


_PUBLISHED_DDL = '''
    from threetears.interfaces.keyed import delete_before_statement, read_row_statement

    TOKENS_DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        token TEXT PRIMARY KEY,
        expires_at TIMESTAMPTZ NOT NULL
    )
    """

    EXPIRES_DDL = (
        "-- purpose: src/tokens.py::Tokens.purge\\n"
        "-- plan_test: tests/test_tokens.py::test_purge_plan\\n"
        "CREATE INDEX IF NOT EXISTS {index} ON {table} (expires_at)"
    )

    def _spelled(table):
        return table.lower()

    class Tokens:
        def __init__(self, table):
            self._table = table

        def get(self, token):
            return read_row_statement(_spelled(self._table), by={"token": token})

        def purge(self, before):
            return delete_before_statement(_spelled(self._table), key=("token",), column="expires_at", before=before)

        def scan(self, before):
            return read_row_statement(_spelled(self._table), by={"expires_at": before})
'''


def test_a_table_its_caller_names_is_judged_by_the_module_s_published_ddl(tmp_path: Path) -> None:
    """Flips if a builder's statement over a table the caller names (through one spelling call) is not judged
    against the module's published DDL (``CREATE TABLE IF NOT EXISTS {table}``, its ``{table}`` index): the key's
    point read and the purposed purge keyed, a read by an unindexed column flagged."""
    found = _symbols(_config(tmp_path, {"tokens.py": _PUBLISHED_DDL}))
    functions = {function for function, _ in found}
    assert not functions & {"Tokens.get", "Tokens.purge"}, found
    assert "Tokens.scan" in functions, found


# --- threetears_tables_only: the gate covers 3tears' data access only, a side database unchecked -------------------

_SIDE_DATABASE = {
    "side/models.py": """
        import sqlalchemy as sa

        metadata = sa.MetaData()
        audit = sa.Table("side_audit", metadata, sa.Column("id", sa.Integer, primary_key=True), sa.Column("kind", sa.Text))
    """,
    "side/migrations/v001.py": """
        DDL = "CREATE TABLE side_jobs (id BIGINT PRIMARY KEY, state TEXT)"
    """,
    "side/reads.py": """
        async def every_audit(conn):
            return await conn.fetch("SELECT id, kind FROM side_audit")

        async def stuck_jobs(conn):
            return await conn.fetch("SELECT id FROM side_jobs WHERE state = $1", "stuck")

        async def clear_jobs(conn):
            await conn.execute("DELETE FROM side_jobs")
    """,
    "data/schema.py": """
        from threetears.core.collections import TableSchema

        RESULTS = TableSchema(name="results", columns=(), primary_key=("state", "geo"))

        def tenant_table(name):
            return TableSchema(name=name, columns=(), primary_key=("customer_id", "id"))
    """,
    "data/reads.py": """
        async def every_result(conn):
            return await conn.fetch("SELECT state, geo FROM results")
    """,
    "data/schema_reads.py": """
        from threetears.core.collections import TableSchema

        def tenant_table(name):
            return TableSchema(name=name, columns=(), primary_key=("customer_id", "id"))

        async def every_row(conn, table):
            return await conn.fetch(f"SELECT id FROM {table}")
    """,
    "side/generic.py": """
        def side_table(name):
            return f"CREATE TABLE {name} (id BIGINT PRIMARY KEY, body TEXT)"

        async def every_side_row(conn, table):
            return await conn.fetch(f"SELECT id FROM {table}")
    """,
}


def test_with_only_3tears_tables_a_side_database_s_tables_are_no_data_tables(tmp_path: Path) -> None:
    """Flips if, with ``threetears_tables_only``, a table only a SQLAlchemy ``Table`` or the repo's own migration DDL
    declares (or a run-time table whose module template is DDL, not a ``TableSchema``) is still judged, or a table 3tears
    declares (literal, or a run-time one its ``TableSchema`` template builds) stops being judged; and if the default
    (off) stops judging the side database's reads. use-the-helper follows the same switch."""
    side = {
        ("every_audit", "whole-select side_audit"),
        ("stuck_jobs", "non-key-where side_jobs"),
        ("clear_jobs", "whole-delete side_jobs"),
    }
    ours = {("every_result", "whole-select results")}
    default = _symbols(_config(tmp_path / "off", _SIDE_DATABASE))
    assert side <= default and ours <= default
    assert any(function == "every_side_row" for function, _ in default)
    assert any(function == "every_row" for function, _ in default)
    switched = _symbols(_config(tmp_path / "on", _SIDE_DATABASE, threetears_tables_only=True))
    assert not side & switched and ours <= switched
    assert not any(function == "every_side_row" for function, _ in switched)
    assert any(function == "every_row" for function, _ in switched)
    helper_off = " ".join(use_the_helper_findings(_config(tmp_path / "uoff", _SIDE_DATABASE)))
    helper_on = " ".join(
        use_the_helper_findings(_config(tmp_path / "uon", _SIDE_DATABASE, threetears_tables_only=True))
    )
    assert "every_audit" in helper_off and "every_audit" not in helper_on
    assert "every_result" in helper_off and "every_result" in helper_on


_DECLARED_SCHEMA_NAME = {
    "audit.py": """
        import dataclasses

        from threetears.core.collections.schema_backed import Index, SchemaBackedCollection, TableSchema
        from threetears.core.keyed import Sort, keyset_page, read_row

        class AuditEventCollection(SchemaBackedCollection):
            schema = TableSchema(
                name="platform_audit.audit_events",
                primary_key=("id",),
                columns=[],
                indexes=(
                    Index(
                        "idx_audit_actor",
                        ("actor_user_id", "timestamp"),
                        purpose=(
                            "src/audit.py::AuditEventCollection.by_actor; "
                            "src/audit.py::AuditEventCollection.by_actor_local; "
                            "src/audit.py::AuditEventCollection.by_actor_replaced; "
                            "src/audit.py::AuditEventCollection.page_actor"
                        ),
                        plan_test="tests/test_plans.py::test_by_actor_uses_idx_audit_actor",
                    ),
                ),
            )

            def __init__(self, audit_schema):
                object.__setattr__(
                    self, "schema", dataclasses.replace(type(self).schema, name=f"{audit_schema}.audit_events")
                )

            async def by_actor(self, conn, actor):
                return await conn.fetch(
                    f"SELECT id FROM {self.schema.name} WHERE actor_user_id = $1 ORDER BY timestamp LIMIT 10", actor
                )

            async def by_actor_local(self, conn, actor):
                table = self.schema.name
                return await conn.fetch(
                    f"SELECT id FROM {table} WHERE actor_user_id = $1 ORDER BY timestamp LIMIT 10", actor
                )

            async def by_actor_replaced(self, conn, audit, actor):
                schema = dataclasses.replace(type(self).schema, name=f"{audit}.audit_events")
                return await conn.fetch(
                    f"SELECT id FROM {schema.name} WHERE actor_user_id = $1 ORDER BY timestamp LIMIT 10", actor
                )

            async def one(self, conn, key):
                return await read_row(conn, self.schema.name, by={"id": key})

            async def page_actor(self, conn, actor):
                table = self.schema.name
                return await keyset_page(
                    conn, table, columns=("id",), by={"actor_user_id": actor}, order=(Sort("timestamp"),), limit=10
                )

            async def by_details(self, conn, details):
                return await conn.fetch(f"SELECT id FROM {self.schema.name} WHERE details = $1", details)
    """,
    "conversations/schema.py": """
        from threetears.core.collections.schema_backed import Index, TableSchema

        CONVERSATIONS = TableSchema(
            name="conversations",
            primary_key=("conversation_id",),
            columns=[],
            indexes=(
                Index(
                    "idx_conv_agent_status",
                    ("agent_id", "status", "conversation_id"),
                    purpose="src/admin.py::by_status; src/admin.py::page_status",
                    plan_test="tests/test_plans.py::test_by_status_uses_idx_conv_agent_status",
                ),
            ),
        )
    """,
    "admin.py": """
        from threetears.core.keyed import Sort, keyset_page

        def agent_schema_name(agent_id):
            return f"agent_{agent_id}"

        async def by_status(pool, agent_id, status):
            schema = agent_schema_name(agent_id)
            return await pool.fetch(
                f"SELECT * FROM {schema}.conversations WHERE agent_id = $1 AND status = $2 "
                "ORDER BY conversation_id LIMIT 20",
                agent_id,
                status,
            )

        async def page_status(pool, agent_id, status):
            return await keyset_page(
                pool,
                f"{agent_schema_name(agent_id)}.conversations",
                columns=("conversation_id",),
                by={"agent_id": agent_id, "status": status},
                order=(Sort("conversation_id"),),
                limit=20,
            )

        async def by_title(pool, agent_id, title):
            return await pool.fetch(f"SELECT * FROM {agent_schema_name(agent_id)}.conversations WHERE title = $1", title)
    """,
}

_DECLARED_SCHEMA_PLANS = """
    from threetears.core.testing.plans import explain_statement, plan_uses_index

    async def test_by_actor_uses_idx_audit_actor(conn):
        sql = "SELECT id FROM platform_audit.audit_events WHERE actor_user_id = $1 ORDER BY timestamp LIMIT 10"
        assert plan_uses_index(await explain_statement(conn, sql, "pg"), "idx_audit_actor")

    async def test_by_status_uses_idx_conv_agent_status(conn):
        sql = "SELECT * FROM conversations WHERE agent_id = $1 AND status = $2 ORDER BY conversation_id LIMIT 20"
        assert plan_uses_index(await explain_statement(conn, sql, "pg"), "idx_conv_agent_status")
"""


def test_a_collection_s_own_schema_name_and_a_schema_qualified_table_read_as_the_declared_table(
    tmp_path: Path,
) -> None:
    """Flips if a SchemaBackedCollection method's ``self.schema.name`` (directly, through a local bound from it, or
    through ``dataclasses.replace`` of the class's TableSchema) is not the class's declared table (by its last segment:
    ``audit_events`` of ``platform_audit.audit_events``), or a per-agent schema's table (``f"{schema}.conversations"``,
    ``f"{agent_schema_name(...)}.conversations"``) is not the declared ``conversations``; each judged with that table's
    declared indexes: the purposed reads pass and keep their index's purpose used, a read by an unindexed column is a
    non-key read of the declared table, never an unknown one."""
    _plans(tmp_path, _DECLARED_SCHEMA_PLANS)
    config = _config(tmp_path, _DECLARED_SCHEMA_NAME)
    findings = no_whole_reads_findings(config)
    found = _symbols(config)
    assert not [finding for finding in findings if "unknown-" in finding], findings
    assert not [finding for finding in findings if ": index-" in finding], findings
    assert ("AuditEventCollection.by_details", "non-key-where audit_events") in {
        (function, re.sub(r"^(\S+) .*?(\w+)$", r"\1 \2", symbol)) for function, symbol in found
    }, found
    assert any(function == "by_title" and symbol.startswith("non-key-where") for function, symbol in found), found
    judged = {function for function, _ in found}
    assert not judged & {
        "AuditEventCollection.by_actor",
        "AuditEventCollection.by_actor_local",
        "AuditEventCollection.by_actor_replaced",
        "AuditEventCollection.one",
        "AuditEventCollection.page_actor",
        "by_status",
        "page_status",
    }, found


_CONSUMER_PURPOSE = """
    from threetears.core.collections.schema_backed import Index, TableSchema

    CONVERSATIONS = TableSchema(
        name="conversations",
        primary_key=("agent_id", "conversation_id"),
        columns=[],
        indexes=(
            Index(
                "idx_conv_agent_status",
                ("agent_id", "status", "conversation_id"),
                purpose="src/aibots/hub/admin/conversations.py::_fetch_paginated_rows",
                plan_test="tests/test_plans.py::test_page_uses_idx_conv_agent_status",
            ),
            Index(
                "idx_conv_agent_title",
                ("agent_id", "title"),
                purpose="src/aibots/hub/admin/gone.py::nothing; src/elsewhere.py::nothing",
                plan_test="tests/test_plans.py::test_page_uses_idx_conv_agent_status",
            ),
        ),
    )
"""

_CONSUMER_PLANS = """
    from threetears.core.testing.plans import explain_statement, plan_uses_index

    async def test_page_uses_idx_conv_agent_status(conn):
        sql = "SELECT * FROM conversations WHERE agent_id = $1 AND status = $2 ORDER BY conversation_id LIMIT 20"
        assert plan_uses_index(await explain_statement(conn, sql, "pg"), "idx_conv_agent_status")
        assert plan_uses_index(await explain_statement(conn, sql, "pg"), "idx_conv_agent_title")
"""


def test_a_purpose_may_name_a_consumer_repo_s_read_under_a_declared_consumer_path(tmp_path: Path) -> None:
    """Flips if, with ``consumer_purposes``, an index purpose naming a read under a consumer's path (the hub's admin
    listing) is still unknown here, or one outside every consumer path stops being unknown; and if without the
    option the consumer entry passes."""
    for side in ("on", "off"):
        (tmp_path / side).mkdir()
        _plans(tmp_path / side, _CONSUMER_PLANS)
    on = no_whole_reads_findings(
        _config(tmp_path / "on", {"schema.py": _CONSUMER_PURPOSE}, consumer_purposes=("src/aibots/hub/",))
    )
    off = no_whole_reads_findings(_config(tmp_path / "off", {"schema.py": _CONSUMER_PURPOSE}))
    consumer = "purpose 'src/aibots/hub/admin/conversations.py::_fetch_paginated_rows'"
    assert not any(consumer in finding for finding in on), on
    assert not any("idx_conv_agent_status" in finding for finding in on), on
    assert any("purpose 'src/elsewhere.py::nothing'" in finding for finding in on), on
    assert any(consumer in finding and "index-purpose-unknown" in finding for finding in off), off


_STEP_BUILDERS = {
    "tables.py": """
        from threetears.core.collections.schema_backed import TableSchema

        PROJECTIONS = TableSchema(name="projections", primary_key=("state", "race"), columns=[])
    """,
    "builders.py": """
        from threetears.core import keyed

        def newest(states) -> keyed.Step:
            return keyed.scan("projections", where=[keyed.Fragment('"state" = ANY($1)', (list(states),))])

        def everything(states) -> keyed.Step:
            return keyed.scan("projections")

        def unannotated(states):
            return keyed.scan("projections", where=[keyed.Fragment('"state" = ANY($1)', (list(states),))])
    """,
    "derived.py": """
        from collections.abc import Callable, Sequence
        from dataclasses import dataclass

        from threetears.core import keyed
        from threetears.core.keyed import InlineDuckDB, run_inline

        from builders import everything, newest, unannotated

        @dataclass(frozen=True)
        class Derived:
            sources: tuple[str, ...]
            select: str | Callable[[Sequence[str]], keyed.Step]

            def step(self, states) -> keyed.Step | None:
                return None if isinstance(self.select, str) else self.select(sorted(states))

        class Copy:
            def created(self, name, made):
                step = made.step(())
                if step is not None:
                    run_inline(keyed.materialize(InlineDuckDB(self._db), f"main.{name}", step, replace=False))

            def built(self, name):
                step = newest(())
                run_inline(keyed.materialize(InlineDuckDB(self._db), f"main.{name}", step, replace=False))

            def built_whole(self, name):
                run_inline(keyed.materialize(InlineDuckDB(self._db), f"main.{name}", everything(()), replace=False))

            def unknown(self, name):
                step = unannotated(())
                run_inline(keyed.materialize(InlineDuckDB(self._db), f"main.{name}", step, replace=False))

            def replaced(self, name):
                run_inline(keyed.materialize(InlineDuckDB(self._db), f"main.{name}", newest(())))
    """,
}


def test_a_step_a_builder_returns_is_judged_where_the_builder_writes_it(tmp_path: Path) -> None:
    """Flips if a step got from a step builder (a function annotated to return a keyed step, a Derived's ``step()``
    calling its field annotated a callable returning one) reads as a whole select of an unknown relation where it is
    materialized (R11), if a builder's own whole scan stops being named in it, if a step from a function not annotated
    as a builder stops being unknown, or if ``materialize(replace=False)`` (a table made new) is read as a whole
    delete of its target while ``replace=True`` (dropped first) is not."""
    found = _symbols(_config(tmp_path, _STEP_BUILDERS))
    judged = {function for function, _ in found}
    assert not judged & {"Copy.created", "Copy.built"}, found
    assert ("everything", 'whole-select "projections"') in found, found
    assert ("Copy.built_whole", "calls-whole-read everything") in found, found
    assert ("Copy.unknown", "whole-select {}") in found, found
    assert ("Copy.replaced", "whole-delete copy.{}") in found, found


_BRANCHED_BY = {
    "merge.py": """
        from threetears.core.collections.schema_backed import Index, TableSchema
        from threetears.core.keyed import update_keyed

        FILES = TableSchema(
            name="files",
            primary_key=("agent_id", "file_id"),
            columns=[],
            indexes=(
                Index(
                    "idx_files_agent_user",
                    ("agent_id", "user_id"),
                    purpose="src/merge.py::repoint",
                    plan_test="tests/test_plans.py::test_repoint_uses_idx_files_agent_user",
                ),
                Index(
                    "idx_files_user",
                    ("user_id",),
                    purpose="src/merge.py::repoint",
                    plan_test="tests/test_plans.py::test_repoint_uses_idx_files_user",
                ),
            ),
        )

        async def repoint(conn, from_user, to_user, agent_id=None):
            by = {"user_id": from_user} if agent_id is None else {"agent_id": agent_id, "user_id": from_user}
            return await update_keyed(conn, "files", values={"user_id": to_user}, by=by, returning=("file_id",))

        async def repoint_inline(conn, from_user, to_user, agent_id=None):
            return await update_keyed(
                conn,
                "files",
                values={"user_id": to_user},
                by={"user_id": from_user} if agent_id is None else {"title": from_user},
                returning=("file_id",),
            )
    """,
}

_BRANCHED_PLANS = """
    from threetears.core.testing.plans import explain_write, plan_uses_index

    async def test_repoint_uses_idx_files_agent_user(conn):
        sql = "UPDATE files SET user_id = $1 WHERE agent_id = $2 AND user_id = $3"
        assert plan_uses_index(await explain_write(conn, sql), "idx_files_agent_user")

    async def test_repoint_uses_idx_files_user(conn):
        sql = "UPDATE files SET user_id = $1 WHERE user_id = $2"
        assert plan_uses_index(await explain_write(conn, sql), "idx_files_user")
"""


def test_a_by_that_is_one_literal_or_another_is_judged_per_branch(tmp_path: Path) -> None:
    """Flips if a keyed write whose ``by=`` is one of two literal mappings by a condition (a local bound once to it, or
    written inline) is unknown rather than judged branch by branch: each branch's index keeps its purpose's read, and a
    branch by an unindexed column is a non-key write."""
    _plans(tmp_path, _BRANCHED_PLANS)
    config = _config(tmp_path, _BRANCHED_BY)
    findings = no_whole_reads_findings(config)
    assert not [finding for finding in findings if "unknown-" in finding], findings
    assert not [finding for finding in findings if ": index-" in finding], findings
    found = _symbols(config)
    assert not any(function == "repoint" for function, _ in found), found
    assert ("repoint_inline", 'non-key-where "files"') in found, found


_BASE_CLASS_SWEEP = {
    "coordination.py": """
        from threetears.core.collections.schema_backed import Index, SchemaBackedCollection, TableSchema
        from threetears.core.keyed import delete_before

        class Shared(SchemaBackedCollection):
            async def sweep(self, cutoff):
                return await delete_before(
                    self.schema.name, collection=self, key=("purpose", "key"), column="expires_at", before=cutoff, page=5
                )

            async def by_owner(self, owner):
                return await delete_before(
                    self.schema.name, collection=self, key=("purpose", "key"), column="owner", before=owner, page=5
                )

        class Counters(Shared):
            schema = TableSchema(
                name="counters",
                primary_key=("purpose", "key"),
                columns=[],
                indexes=(
                    Index(
                        "counters_expiry",
                        "expires_at",
                        asc=("expires_at",),
                        purpose="src/coordination.py::Shared.sweep",
                        plan_test="tests/test_plans.py::test_sweeps",
                    ),
                ),
            )

        class Claims(Shared):
            schema = TableSchema(
                name="claims",
                primary_key=("purpose", "key"),
                columns=[],
                indexes=(
                    Index(
                        "claims_expiry",
                        "expires_at",
                        asc=("expires_at",),
                        purpose="src/coordination.py::Shared.sweep",
                        plan_test="tests/test_plans.py::test_sweeps",
                    ),
                ),
            )
    """,
}

_BASE_CLASS_SWEEP_PLANS = """
    from threetears.core.testing.plans import explain_write, plan_uses_index

    async def test_sweeps(conn):
        plan = await explain_write(conn, "DELETE FROM counters WHERE expires_at < $1", 0)
        assert plan_uses_index(plan, "counters_expiry")
        plan = await explain_write(conn, "DELETE FROM claims WHERE expires_at < $1", 0)
        assert plan_uses_index(plan, "claims_expiry")
"""


def test_a_base_class_method_s_schema_name_is_each_subclass_s_table(tmp_path: Path) -> None:
    """Flips if a shared method of a base collection declaring no TableSchema (``self.schema.name``) is not judged on
    each same-module subclass's declared table: the sweep through each subclass's purposed expiry index passes and
    keeps both purposes used; a sweep by a column no subclass indexes is a non-key read, never an unknown one."""
    _plans(tmp_path, _BASE_CLASS_SWEEP_PLANS)
    config = _config(tmp_path, _BASE_CLASS_SWEEP)
    findings = no_whole_reads_findings(config)
    found = _symbols(config)
    assert not [finding for finding in findings if "unknown-" in finding], findings
    assert not [finding for finding in findings if ": index-" in finding], findings
    assert not {function for function, _ in found} & {"Shared.sweep"}, found
    assert {symbol.split(" ")[0] for function, symbol in found if function == "Shared.by_owner"} == {"non-key-where"}, (
        found
    )


_FORWARDED_REPOINT = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import update_keyed

    FILES = TableSchema(
        name="files",
        primary_key=("agent_id", "file_id"),
        columns=[],
        indexes=(
            Index(
                "idx_files_agent_user",
                ("agent_id", "user_id"),
                purpose="src/helpers.py::repoint",
                plan_test="tests/test_plans.py::test_repoint_uses_idx_files_agent_user",
            ),
        ),
    )

    async def repoint(conn, *, table, user_column, from_user, to_user, agent_id=None):
        by = {user_column: from_user} if agent_id is None else {"agent_id": agent_id, user_column: from_user}
        return await update_keyed(conn, table, values={user_column: to_user}, by=by, returning=("file_id",))
"""

_REPOINT_CALLERS = """
    from uuid import UUID

    from helpers import repoint

    async def move_files(conn, *, from_user: UUID, to_user: UUID, agent_id: UUID):
        return await repoint(
            conn, table="files", user_column="user_id", from_user=from_user, to_user=to_user, agent_id=agent_id
        )
"""

_REPOINT_PLANS = """
    from threetears.core.testing.plans import explain_write, plan_uses_index

    async def test_repoint_uses_idx_files_agent_user(conn):
        sql = "UPDATE files SET user_id = $1 WHERE agent_id = $2 AND user_id = $3"
        assert plan_uses_index(await explain_write(conn, sql), "idx_files_agent_user")
"""


def test_a_forwarded_column_and_a_never_none_agent_judge_the_one_statement_sent(tmp_path: Path) -> None:
    """Flips if a helper judged at its callers reads a mapping key that is its parameter (``{user_column: ...}``) as
    unknown rather than each string its callers pass, or judges the ``agent_id is None`` arm where every caller passes
    an agent that is never None (a required ``UUID`` parameter): the agent's write by its index is all it sends."""
    _plans(tmp_path, _REPOINT_PLANS)
    config = _config(tmp_path, {"helpers.py": _FORWARDED_REPOINT, "merge.py": _REPOINT_CALLERS})
    findings = no_whole_reads_findings(config)
    assert not [finding for finding in findings if "unknown-" in finding], findings
    assert not [finding for finding in findings if ": index-" in finding], findings
    assert not any(function == "repoint" for function, _ in _symbols(config)), findings


def test_an_agent_a_caller_may_leave_none_keeps_the_owner_only_arm_judged(tmp_path: Path) -> None:
    """Flips if the ``agent_id is None`` arm is folded away while a caller may pass None (``UUID | None``) or leaves
    the default: its write by the owner alone has no index and stays a non-key write, reported at the caller passing
    that table and column (``move_files``)."""
    _plans(tmp_path, _REPOINT_PLANS)
    optional = _REPOINT_CALLERS.replace("agent_id: UUID)", "agent_id: UUID | None)")
    config = _config(tmp_path, {"helpers.py": _FORWARDED_REPOINT, "merge.py": optional})
    findings = no_whole_reads_findings(config)
    assert ("move_files", 'non-key-where "files"') in _symbols(config), findings
    assert not [finding for finding in findings if "unknown-" in finding], findings


def test_a_dependency_s_later_drop_of_an_index_on_its_own_table_drops_the_repo_s_copy(tmp_path: Path) -> None:
    """Flips if an index the repo's squashed migration creates on a dependency's table (the hub's v001 copy of
    idx_conv_mem_refs_conversation_date_created) outlives the dependency's own migration dropping it (agent-memory
    v034), or if a dependency's drop of a same-named index on ITS table drops the repo's index on an unrelated table,
    or a drop the dependency took back (it created the index again after) still drops the repo's copy."""
    dependency = tmp_path / "dep"
    dependency.mkdir()
    (dependency / "v001.sql").write_text(
        "CREATE TABLE refs (conversation_id UUID, id UUID, date_created TIMESTAMPTZ, PRIMARY KEY (conversation_id, id));\n"
        "CREATE INDEX idx_refs_conv_created ON refs (conversation_id, date_created);\n"
        "CREATE TABLE notes (id UUID PRIMARY KEY, at TIMESTAMPTZ);\n"
        "CREATE INDEX idx_by_at ON notes (at);\n"
        "CREATE INDEX idx_kept ON notes (at);\n",
        encoding="utf-8",
    )
    (dependency / "v002.sql").write_text(
        "DROP INDEX IF EXISTS idx_refs_conv_created;\nDROP INDEX IF EXISTS idx_by_at;\nDROP INDEX IF EXISTS idx_kept;\n",
        encoding="utf-8",
    )
    (dependency / "v003.sql").write_text("CREATE INDEX idx_kept ON notes (at);\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "v001_squash.sql").write_text(
        "CREATE INDEX IF NOT EXISTS idx_refs_conv_created ON refs (conversation_id, date_created);\n"
        "CREATE TABLE own (id UUID PRIMARY KEY, at TIMESTAMPTZ);\n"
        "CREATE INDEX idx_by_at ON own (at);\n"
        "CREATE INDEX IF NOT EXISTS idx_kept ON notes (at);\n",
        encoding="utf-8",
    )
    catalog = collect_schema(tmp_path, (tmp_path / "src",), dependency_roots=(dependency,))
    assert "idx_refs_conv_created" not in catalog.tables["refs"].indexes
    assert "idx_by_at" in catalog.tables["own"].indexes
    assert "idx_kept" in catalog.tables["notes"].indexes
    findings = index_findings(catalog, tmp_path, set())
    assert not any("idx_refs_conv_created" in finding for finding in findings), findings
    assert any("idx_by_at on own: index-no-purpose" in finding for finding in findings), findings


_RELATION_WRAPPER = """
    from threetears.core import keyed

    STATE_COLUMN = "join_state_code"


    def _step_relation(relation):
        if relation.startswith("(") and relation.endswith(")"):
            return keyed.Derived(relation[1:-1])
        return keyed.Quoted(relation)


    def _not_a_wrapper(relation, other):
        return keyed.Quoted(other)


    def by_state(snapshot, table, states):
        return keyed.scan(_step_relation(snapshot.relation(table)), ["a"], where=[keyed.Fragment(f"t.{STATE_COLUMN} IN ({states})")])


    def by_kind(snapshot, table, kinds):
        return keyed.scan(_step_relation(snapshot.relation(table)), ["a"], where=[keyed.Fragment(f"t.kind IN ({kinds})")])


    def static_by_state(snapshot, states):
        return keyed.scan(_step_relation(snapshot.relation("benchmarks")), ["a"], where=[keyed.Fragment(f"t.{STATE_COLUMN} IN ({states})")])


    def unwrapped(snapshot, table, states):
        return keyed.scan(_not_a_wrapper(table, snapshot.relation(table)), ["a"], where=[keyed.Fragment(f"t.{STATE_COLUMN} IN ({states})")])
"""


def test_a_module_function_wrapping_its_parameter_as_a_step_relation_reads_as_the_relation_quoted(
    tmp_path: Path,
) -> None:
    """Flips if a step's relation built by a module function returning ``Quoted(<its parameter>)`` or a ``Derived``
    of it stays unknown (``by_state``: the copy's run-time relation by its state key), if the wrapper's relation is
    trusted rather than judged (``by_kind`` off the key, ``static_by_state`` on a copy keyed otherwise), or if a
    function returning ``Quoted`` of something else is read as a wrapper (``unwrapped``)."""
    keys = {"*": (("join_state_code",),), "benchmarks": (("benchmark",),)}
    found = _symbols(_config(tmp_path, {"copy/evaluate.py": _RELATION_WRAPPER}, local_relations=keys))
    assert not any(function == "by_state" for function, _ in found), found
    assert ("by_kind", "non-key-where copy.{}") in found, found
    assert ("static_by_state", "non-key-where copy.benchmarks") in found, found
    assert ("unwrapped", "unknown-predicate scan(rel)") in found, found


_COPY_FAMILY = """
    from typing import Any, Final, Mapping

    from threetears.core import keyed
    from threetears.core.keyed import InlineDuckDB, quote_identifier

    STATE_COLUMN: Final = "join_state_code"
    HELD_RUNS: Final = "held_runs"
    _STATE_COLUMNS: Final = {HELD_RUNS: "state"}


    def _state_column(table: str) -> str:
        \"\"\"the column naming a row's state.\"\"\"
        return _STATE_COLUMNS.get(table, STATE_COLUMN)


    def _held_rows(connection, table: str, states):
        return keyed.read_keyed_arrow(InlineDuckDB(connection), table, keys=keyed.KeyBatch((_state_column(table),), [(s,) for s in states]))


    def apply_slice(scratch, connection, tables: Mapping[str, Any], states):
        for table in tables:
            _held_rows(connection, table, states)
        for table in tables:
            column = quote_identifier(_state_column(table))
            held_in = f"main.{table}"
            scratch.step_arrow(keyed.scan(held_in, where=[keyed.Fragment(f"t.{column} IN (SELECT unnest($1))", (list(states),))]))


    def small_parts(connection, small, states):
        for table, pieces in small.items():
            _held_rows(connection, table, states)


    def by_kind(scratch, tables: Mapping[str, Any], kind):
        for table in sorted(tables):
            scratch.step_arrow(keyed.scan(f"main.{table}", where=[keyed.Fragment("t.kind = $1", (kind,))]))


    def undeclared(scratch, tables: Mapping[str, Any], states):
        for table in tables:
            column = quote_identifier(_state_column(table))
            scratch.step_arrow(keyed.scan(f"main.{table}", where=[keyed.Fragment(f"t.{column} = $1", (states,))]))


    class Backend:
        def export_partition(self, table: str, column: str, value):
            return _partition(None, table, column, value)


    def _partition(cursor, table: str, column: str, value):
        return keyed.read_keyed_arrow(InlineDuckDB(cursor), table, keys=keyed.KeyBatch((column,), [(value,)]))
"""

#: the copy's keys: by default its state, the race; the held runs table by its own state column too
_FAMILY_KEYS = {"*": (("join_state_code",), ("office_key",)), "held_runs": (("join_state_code", "run_id"), ("state",))}


def test_a_declared_parameter_is_judged_once_per_value_it_declares(tmp_path: Path) -> None:
    """Flips if a read over a declared parameter's values (``declared_parameters``: a collection a loop binds each
    of, a string holding one) stays unknown when each value's own key is bound (``apply_slice``, its forwarder
    ``_held_rows`` at its call, ``Backend.export_partition`` through ``_partition``: a lookup function's column per
    table, through names bound once from the loop; ``small_parts``, a declared loop target), if a member keyed otherwise is not reported by name
    (``benchmarks`` read by state: its key is ``benchmark``), if a read off every member's key passes or is reported
    once per member rather than once for the family (``by_kind``),
    or if an undeclared parameter is judged rather than left as the run-time relation it is (``undeclared``)."""
    members = ("rows", "held_runs", "benchmarks")
    declared = {
        "apply_slice.tables": members,
        "by_kind.tables": members,
        "small_parts.table": ("rows", "held_runs"),
        "Backend.export_partition.table": ("rows", "held_runs"),
        "Backend.export_partition.column": ("join_state_code",),
    }
    keys = {**_FAMILY_KEYS, "benchmarks": (("benchmark",),)}
    config = _config(tmp_path, {"copy/apply.py": _COPY_FAMILY}, local_relations=keys, declared_parameters=declared)
    found = _symbols(config)
    assert ("apply_slice", 'non-key-where "main"."benchmarks"') in found, found
    assert ("apply_slice", 'whole-select "main"."benchmarks"') in found, found
    assert not any(function == "apply_slice" and "held_runs" in symbol for function, symbol in found), found
    # off the key of every member: one finding naming the family, not one per member
    assert ("by_kind", "non-key-where <each-declared-value>") in found, found
    assert not any(function == "by_kind" and '"main"' in symbol for function, symbol in found), found
    assert ("undeclared", "non-key-where copy.{}") in found, found
    assert not any(function in {"Backend.export_partition", "_partition"} for function, _ in found), found
    assert not any(function == "_held_rows" for function, _ in found), found


def test_a_declared_parameter_naming_no_parameter_is_a_finding(tmp_path: Path) -> None:
    """Flips if a declaration (``declared_parameters``) naming no parameter of a function the walk reads (a renamed
    function or parameter) is silently ignored, or if one naming a parameter is reported."""
    declared = {"apply_slice.tables": ("rows",), "apply_slice.gone": ("rows",), "renamed.tables": ("rows",)}
    config = _config(
        tmp_path, {"copy/apply.py": _COPY_FAMILY}, local_relations=_FAMILY_KEYS, declared_parameters=declared
    )
    stale = [finding for finding in no_whole_reads_findings(config) if finding.startswith("declared parameter")]
    assert len(stale) == 2 and all("apply_slice.tables" not in line for line in stale), stale


_DEFAULTED_FORWARDER = """
    from threetears.core import keyed

    FEED_RACE = "feed_race"
    FEED_COUNTY = "feed_county"


    def moves(states, feed_race: str = f"main.{FEED_RACE}", feed_county: str = f"main.{FEED_COUNTY}"):
        within = keyed.Fragment("t.join_state_code IN (SELECT unnest($1))", (sorted(states),))
        race = keyed.scan(feed_race, ["office_key"], where=[within])
        county = keyed.scan(feed_county, ["office_key"], where=[within])
        return keyed.union(race, county)


    def by_kind(kinds, feed: str = f"main.{FEED_RACE}"):
        return keyed.scan(feed, ["office_key"], where=[keyed.Fragment("t.kind IN (SELECT unnest($1))", (kinds,))])


    class _Work:
        def put(self, rows):
            return keyed.change_relation("rows")

        def step(self, step):
            return step


    class Settle:
        def moved(self, of):
            work: _Work = self.work

            def made(states):
                race, county = (work.put(of[name]) for name in (FEED_RACE, FEED_COUNTY))
                return work.step(moves(states, race, county))

            return made(["VA"])

        def unpacked_whole(self, of):
            work: _Work = self.work

            def made(states):
                race, county = (of[name] for name in (FEED_RACE, FEED_COUNTY))
                return work.step(moves(states, race, county))

            return made(["VA"])


    def copy_moves(states):
        return moves(states)


    def copy_kinds(kinds):
        return by_kind(kinds)
"""


def test_a_forwarder_s_defaulted_relation_is_judged_at_the_call_that_leaves_it(tmp_path: Path) -> None:
    """Flips if a forwarder's relation parameter a call leaves to its default is not judged there with the steps the
    forwarder makes of it (``copy_moves``: the copy's feeds by state, keyed; ``copy_kinds``: a default read off the
    key, reported at the call), if a change set a closure's receiver makes (``work`` annotated in the function around
    it) or names unpacked from a comprehension of such change sets are not read as change sets (``Settle.moved``),
    or if names unpacked from a comprehension of anything else are (``Settle.unpacked_whole``)."""
    found = _symbols(
        _config(
            tmp_path,
            {"copy/moves.py": _DEFAULTED_FORWARDER},
            local_relations=_FAMILY_KEYS,
            transient_producers=frozenset({"_Work.put"}),
            public_forwarders=True,
        )
    )
    assert ("copy_kinds", 'non-key-where "main"."feed_race"') in found, found
    assert not any(function in {"copy_moves", "moves", "Settle.moved.made"} for function, _ in found), found
    assert any(function == "Settle.unpacked_whole.made" for function, _ in found), found


_CONSTANT_READS = """
    from threetears.core.collections.schema_backed import TableSchema

    ITEMS = TableSchema(name="items", primary_key="id", columns=[])
    _COUNT = "SELECT count(*) FROM items"
    _ONE = "SELECT body FROM items WHERE id = $1"
    _ALIAS = _COUNT

    async def count(conn):
        return await conn.fetchval(_COUNT)

    async def one(conn, item):
        return await conn.fetchrow(_ONE, item)

    async def nightly(conn):
        return await count(conn)

    async def lookup(conn, item):
        return await one(conn, item)
"""


def test_a_whole_read_written_as_a_module_constant_names_its_callers(tmp_path: Path) -> None:
    """Flips if a function running a whole read written as a module constant hides it from its callers (no
    calls-whole-read at nightly), if a keyed constant's caller is named, or if the read is counted twice."""
    findings = no_whole_reads_findings(_config(tmp_path, {"store.py": _CONSTANT_READS}))
    named = [finding for finding in findings if " in " in finding]
    assert any("in nightly: calls-whole-read count" in finding for finding in named), named
    assert not any("in lookup:" in finding for finding in named), named
    assert sum(": whole-aggregate items" in finding for finding in named) == 1, named


_TRIMMED = """
    from threetears.core.collections.schema_backed import Index, TableSchema
    from threetears.core.keyed import delete_oldest

    TURNS = TableSchema(
        name="turns",
        primary_key="key",
        columns=[],
        indexes=(
            Index("turns_by_updated", "date_updated", "key", asc="date_updated", purpose="src/turns.py::enforce_cap", plan_test="tests/test_plans.py::test_updated"),
        ),
    )

    async def enforce_cap(conn, over):
        return await delete_oldest(conn, "turns", count=over, key="key", order=ORDER)
"""


def test_a_trim_by_an_age_column_uses_the_index_that_age_leads(tmp_path: Path) -> None:
    """Flips if a cache's trim by an age column (``delete_oldest(order="date_updated")``, the SDK's turn_dedup cap) is
    not credited to the index leading with that column then the key, or if a trim by a column no index leads is
    credited (its purpose unused) or passes as keyed."""
    _plans(tmp_path, "def test_updated(plan):\n    assert plan_uses_index(plan, 'turns_by_updated')\n")
    aged = no_whole_reads_findings(_config(tmp_path, {"turns.py": _TRIMMED.replace("ORDER", '"date_updated"')}))
    assert not [finding for finding in aged if "turns" in finding], aged
    other = no_whole_reads_findings(_config(tmp_path, {"turns.py": _TRIMMED.replace("ORDER", '"kind"')}))
    assert any("index-purpose-unused -- src/turns.py::enforce_cap" in finding for finding in other), other
    assert any("enforce_cap" in finding and "index-" not in finding for finding in other), other


@pytest.mark.parametrize(("conn", "hashed"), [("InlineSQLite(l1)", False), ("conn", True)], ids=["sqlite", "l3"])
def test_a_trim_on_a_sqlite_connection_is_asked_no_range_sharding(tmp_path: Path, conn: str, hashed: bool) -> None:
    """Flips if a trim handed a connection over SQLite in this process (``delete_oldest(InlineSQLite(l1), ...)``, the
    SDK's L1 turn_dedup cap) is asked to declare its age index range-sharded for YugabyteDB, which never holds it, or
    if the same trim on an L3 connection is not."""
    _plans(tmp_path, "def test_updated(plan):\n    assert plan_uses_index(plan, 'turns_by_updated')\n")
    module = (
        _TRIMMED.replace(', asc="date_updated"', "")
        .replace("ORDER", '"date_updated"')
        .replace("delete_oldest(conn,", f"delete_oldest({conn},")
        .replace("async def enforce_cap(conn, over):", "async def enforce_cap(conn, l1, over):")
        .replace(
            "from threetears.core.keyed import delete_oldest",
            "from threetears.core.keyed import InlineSQLite, delete_oldest",
        )
    )
    found = no_whole_reads_findings(_config(tmp_path, {"turns.py": module}))
    assert any("index-hashed-range" in finding for finding in found) is hashed, found
    assert not [finding for finding in found if "turns" in finding and "index-hashed-range" not in finding], found


@pytest.mark.parametrize(
    ("binding", "hashed"),
    [
        ("db = InlineSQLite(l1)", False),
        ("db = InlineDuckDB(l1)", False),
        ("db = conn", True),
        ("db = InlineSQLite(l1)\n        if over > 9:\n            db = conn", True),
    ],
    ids=["sqlite", "duckdb", "l3", "rebound-l3"],
)
def test_a_trim_on_a_local_connection_bound_to_a_name_is_asked_no_range_sharding(
    tmp_path: Path, binding: str, hashed: bool
) -> None:
    """Flips if a trim handed a name the function binds to a connection over a database in this process (the SDK's
    ``db = InlineSQLite(l1)`` then ``delete_oldest(db, ...)``) is asked to declare its age index range-sharded for
    YugabyteDB, or if the name bound to an L3 connection (anywhere in the function) is not."""
    _plans(tmp_path, "def test_updated(plan):\n    assert plan_uses_index(plan, 'turns_by_updated')\n")
    module = (
        _TRIMMED.replace(', asc="date_updated"', "")
        .replace("ORDER", '"date_updated"')
        .replace(
            "    async def enforce_cap(conn, over):\n        return await delete_oldest(conn,",
            f"    async def enforce_cap(conn, l1, over):\n        {binding}\n        return await delete_oldest(db,",
        )
        .replace(
            "from threetears.core.keyed import delete_oldest",
            "from threetears.core.keyed import InlineDuckDB, InlineSQLite, delete_oldest",
        )
    )
    assert "delete_oldest(db," in module
    found = no_whole_reads_findings(_config(tmp_path, {"turns.py": module}))
    assert any("index-hashed-range" in finding for finding in found) is hashed, found
    assert not [finding for finding in found if "turns" in finding and "index-hashed-range" not in finding], found


_SCOPED_TABLES = """
    from threetears.core.collections.schema_backed import TableSchema

    SOURCES = TableSchema(name="sources", primary_key=("owner_id", "id"), columns=[])
    DOCS = TableSchema(name="docs", primary_key=("source_id", "id"), columns=[])
    NOTES = TableSchema(name="notes", primary_key=("owner_id", "id"), columns=[])
    BARE = TableSchema(name="bare", primary_key="id", columns=[])
    BOTH = TableSchema(name="both", primary_key=("source_id", "id"), columns=[])
"""

_SCOPED_MAP = """
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Scoped:
        owner_column: str | None
        anchor_column: str | None

    SCOPED = {
        "docs": Scoped(owner_column=None, anchor_column="source_id"),
        "notes": Scoped(owner_column="owner_id", anchor_column=None),
        "bare": Scoped(owner_column="owner_id", anchor_column=None),
        "both": Scoped(owner_column="owner_id", anchor_column="source_id"),
    }
"""

_SCOPE_INJECTION = """
    from threetears.core.declared import sentinel_templates

    _SCHEMA = "__schema__"
    _TABLE = "__table__"
    _OWNER = "__owner_column__"
    _ANCHOR = "__anchor_column__"
    _CLAMP = f"({_OWNER} IS NULL OR {_OWNER} = $9001)"
    _THROUGH = f"{_ANCHOR} IN (SELECT id FROM {_SCHEMA}.sources WHERE owner_id IS NULL OR owner_id = $9001)"

    def parse(sql):
        return sql

    OWNER_ONLY = parse(f"SELECT * FROM {_SCHEMA}.{_TABLE} WHERE {_CLAMP}")
    OWNER_AND_ANCHOR = parse(f"SELECT * FROM {_SCHEMA}.{_TABLE} WHERE {_CLAMP} AND {_ANCHOR} = ANY($9002)")
    ANCHOR_ONLY = parse(f"SELECT * FROM {_SCHEMA}.{_TABLE} WHERE {_THROUGH} AND {_ANCHOR} = ANY($9002)")
    UNDECLARED = parse(f"SELECT * FROM {_SCHEMA}.{_TABLE} WHERE {_ANCHOR} = ANY($9002)")

    sentinel_templates(
        templates=("OWNER_ONLY", "OWNER_AND_ANCHOR", "ANCHOR_ONLY"),
        table=_TABLE,
        tables="scoping.maps.SCOPED",
        columns={_OWNER: "owner_column", _ANCHOR: "anchor_column"},
    )
"""


def test_a_template_over_a_declared_closed_map_is_judged_once_per_entry_it_fills(tmp_path: Path) -> None:
    """Flips if a template whose table is a sentinel filled from a declared closed map (hub #44 spec 6) is not judged
    once per entry with the entry's table and columns: a keyed instance found unknown, an unkeyed one (``bare`` and
    ``both`` have no key the owner clamp leads) not named against its table, an instance made for an entry that
    leaves one of the template's sentinels empty, or a template the declaration does not name judged as one."""
    found = _symbols(
        _config(
            tmp_path,
            {
                "scoping/__init__.py": "",
                "scoping/tables.py": _SCOPED_TABLES,
                "scoping/maps.py": _SCOPED_MAP,
                "scoping/injection.py": _SCOPE_INJECTION,
            },
        )
    )
    named = {symbol for _, symbol in found}
    assert named == {
        "non-key-where __schema__.bare",
        "non-key-where __schema__.both",
        "unknown-schema __schema__.__table__",
    }, found


def test_a_template_over_a_map_the_walk_cannot_read_is_judged_as_written(tmp_path: Path) -> None:
    """Flips if a declared template whose map the walk cannot read (a wrong path, or entries that are not literal
    calls) passes unjudged instead of standing as ``unknown-schema``."""
    unread = _SCOPED_MAP.replace('Scoped(owner_column=None, anchor_column="source_id")', "Scoped(**DOCS_COLUMNS)")
    for maps, injection in (
        (_SCOPED_MAP, _SCOPE_INJECTION.replace("scoping.maps.SCOPED", "scoping.elsewhere.SCOPED")),
        (unread, _SCOPE_INJECTION),
    ):
        found = _symbols(
            _config(
                tmp_path,
                {
                    "scoping/__init__.py": "",
                    "scoping/tables.py": _SCOPED_TABLES,
                    "scoping/maps.py": maps,
                    "scoping/injection.py": injection,
                },
            )
        )
        assert {symbol for _, symbol in found} == {"unknown-schema __schema__.__table__"}, found


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"templates": ()}, "at least one template"),
        ({"table": "table"}, "spelled __name__"),
        ({"columns": {"owner": "owner_column"}}, "spelled __name__"),
        ({"tables": "SCOPED"}, "dotted path"),
    ],
)
def test_a_sentinel_templates_declaration_is_checked_where_it_runs(change: dict[str, object], message: str) -> None:
    """Flips if the run-time declaration accepts no template, a sentinel not spelled ``__name__``, or a map that is
    not a dotted path the walk can find."""
    given: dict[str, object] = {
        "templates": ("OWNER_ONLY",),
        "table": "__table__",
        "tables": "scoping.maps.SCOPED",
        "columns": {"__owner_column__": "owner_column"},
    }
    sentinel_templates(**given)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        sentinel_templates(**{**given, **change})  # type: ignore[arg-type]


_KEY_BY_PROVENANCE = """
    from typing import Annotated
    from collections.abc import Sequence

    from threetears.core.keyed import DeclaredKey, KeyBatch, KeyColumn, KeyColumns, read_keyed


    class Store:
        primary_key_column: KeyColumn = "id"

        def _key(self, table) -> KeyColumns:
            return self._declared[table]

        async def by_misnamed_columns(self, conn, table, values):
            pk_cols = ("name", "kind")
            where = " AND ".join(f"{c} = ${i + 1}" for i, c in enumerate(pk_cols))
            return await conn.fetch(f"SELECT * FROM {table} AS t WHERE {where}", *values)

        async def by_a_key_named_otherwise(self, conn, table, values):
            names = self._key(table)
            where = " AND ".join(f"{c} = ${i + 1}" for i, c in enumerate(names))
            return await conn.fetch(f"SELECT * FROM {table} AS t WHERE {where}", *values)

        async def by_a_declared_parameter(self, conn, table, columns: Annotated[Sequence[str], DeclaredKey], value):
            return await conn.fetch(f"SELECT * FROM {table} AS t WHERE t.{columns[0]} = $1", value)

        async def by_a_parameter_named_key(self, conn, table, key, value):
            return await conn.fetch(f"SELECT * FROM {table} AS t WHERE t.{key} = $1", value)

        async def by_the_declared_attribute(self, conn, table, value):
            return await conn.fetch(f"SELECT * FROM {table} AS t WHERE t.{self.primary_key_column} = $1", value)

        async def batch_by_misnamed(self, conn, table, wanted, values):
            pk_cols = list(wanted)
            return await read_keyed(conn, table, keys=KeyBatch(tuple(pk_cols), values))

        async def batch_by_declared(self, conn, table, values):
            columns = self._key(table)
            return await read_keyed(conn, table, keys=KeyBatch(tuple(columns), values))

        async def template_spelling_pk(self, conn, table, other, value):
            return await conn.fetch("SELECT * FROM {table} AS t WHERE t.{pk} = $1".format(table=table, pk=other), value)

"""


def test_a_column_is_the_key_by_where_it_comes_from_never_by_its_name(tmp_path: Path) -> None:
    """Flips if a column list's NAME decides whether it is the key: ``pk_cols`` holding the table's other columns, a
    parameter named ``key``, a ``KeyBatch`` of a list named ``pk_cols``, or a template field spelled ``{pk}`` filled
    by a non-key value must each be flagged; a list named ``names`` written from a function declared to return the
    key (``-> KeyColumns``), a parameter annotated as a declared key, a declared attribute, and a batch of a list
    bound from the declared function, must each pass."""
    found = _symbols(_config(tmp_path, {"store.py": _KEY_BY_PROVENANCE}))
    functions = {function for function, _ in found}
    for flagged in (
        "Store.by_misnamed_columns",
        "Store.by_a_parameter_named_key",
        "Store.batch_by_misnamed",
        "Store.template_spelling_pk",
    ):
        assert flagged in functions, (flagged, found)
    for keyed in (
        "Store.by_a_key_named_otherwise",
        "Store.by_a_declared_parameter",
        "Store.by_the_declared_attribute",
        "Store.batch_by_declared",
    ):
        assert keyed not in functions, (keyed, found)


def test_renaming_a_column_list_never_changes_a_verdict(tmp_path: Path) -> None:
    """Flips if renaming the locals and parameters that hold column lists (``pk_cols`` <-> ``names``, ``key`` ->
    ``other``) changes any function's findings: provenance, not spelling, decides."""
    renamed = (
        _KEY_BY_PROVENANCE.replace("pk_cols", "__a__")
        .replace("names", "pk_cols")
        .replace("__a__", "names")
        .replace(", key, value", ", other_col, value")
        .replace("t.{key}", "t.{other_col}")
    )
    assert renamed != _KEY_BY_PROVENANCE
    before = _symbols(_config(tmp_path / "before", {"store.py": _KEY_BY_PROVENANCE}))
    after = _symbols(_config(tmp_path / "after", {"store.py": renamed}))
    assert {function for function, _ in before} == {function for function, _ in after}, (before, after)


_TABLE_PARAMETERS = """
    from typing import Any

    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema

    ROWS_SCHEMA = TableSchema(name="rows", primary_key=("join_state_code", "office_key"), columns=[])


    async def prefix_rows(backend, schema: TableSchema, prefix: tuple[Any, ...]) -> int:
        columns = schema.pk_columns
        counted = await keyed.counts_by(
            backend,
            schema.name,
            column=columns[len(prefix)],
            values=keyed.AnyOf([None]),
            by=dict(zip(columns[: len(prefix)], prefix, strict=True)) or None,
            batch=1,
        )
        return sum(counted.values())


    async def first_column(backend, schema: TableSchema, value: Any) -> int:
        counted = await keyed.counts_by(backend, schema.name, column=schema.pk_columns[0], values=keyed.AnyOf([value]))
        return sum(counted.values())


    async def off_key(backend, schema: TableSchema, kind: str) -> int:
        counted = await keyed.counts_by(backend, schema.name, column="kind", values=keyed.AnyOf([kind]))
        return sum(counted.values())


    async def maybe_short(backend, schema: TableSchema, prefix: tuple[Any, ...]) -> int:
        counted = await keyed.counts_by(
            backend,
            schema.name,
            column=schema.pk_columns[len(prefix)],
            values=keyed.AnyOf([None]),
            by=dict(zip(schema.pk_columns[: len(prefix)], prefix)),
        )
        return sum(counted.values())


    async def skips_one(backend, schema: TableSchema, prefix: tuple[Any, ...]) -> int:
        counted = await keyed.counts_by(
            backend,
            schema.name,
            column=schema.pk_columns[len(prefix) + 1],
            values=keyed.AnyOf([None]),
            by=dict(zip(schema.pk_columns[: len(prefix)], prefix, strict=True)),
        )
        return sum(counted.values())


    async def not_a_schema(backend, schema, prefix: tuple[Any, ...]) -> int:
        counted = await keyed.counts_by(
            backend,
            schema.name,
            column=schema.pk_columns[len(prefix)],
            values=keyed.AnyOf([None]),
            by=dict(zip(schema.pk_columns[: len(prefix)], prefix, strict=True)),
        )
        return sum(counted.values())


    class CollectionTableStore:
        def __init__(self, backend):
            self._backend = backend

        async def _race_counts(self, table: str, state: str) -> dict[Any, int]:
            return await keyed.counts_by(
                self._backend, table, column="office_key", by={"join_state_code": state}, batch=100, nullable=False
            )
"""


def test_a_keyed_helper_over_a_table_schema_parameter_s_key_prefix_is_keyed(tmp_path: Path) -> None:
    """Flips if a keyed helper reading a ``TableSchema`` parameter's table (``schema.name``) by that schema's own key
    prefix (``column=pk_columns[n]``, ``by=dict(zip(pk_columns[:n], values, strict=True))``, or ``n`` 0 with no
    ``by=``) stays unknown; or if one off the key (a literal column), one whose pins may stop short (no ``strict``),
    one counting a column past the next (``n + 1``), or one over an unannotated parameter is taken as keyed. A ``str``
    table parameter declared to its family (``declared_parameters``) is judged member by member (reports'
    ``_race_counts``): keyed by each member's key, and unknown undeclared."""
    found = _symbols(_config(tmp_path, {"pod/m.py": _TABLE_PARAMETERS}))
    flagged = {function for function, _ in found}
    assert not flagged & {"prefix_rows", "first_column"}, found
    assert {"off_key", "maybe_short", "skips_one", "not_a_schema"} <= flagged, found
    assert ("CollectionTableStore._race_counts", "unknown-predicate counts_by(table)") in found, found
    declared = {"CollectionTableStore._race_counts.table": ("rows",)}
    found = _symbols(_config(tmp_path, {"pod/m.py": _TABLE_PARAMETERS}, declared_parameters=declared))
    assert not any(function == "CollectionTableStore._race_counts" for function, _ in found), found


def _unused(findings: list[str]) -> list[str]:
    return [
        finding for finding in findings if "matches nothing the walk read" in finding or "matches only what" in finding
    ]


def test_a_loosening_declaration_nothing_used_is_a_finding(tmp_path: Path) -> None:
    """Flips if a loosening declaration no judgement of the walk went through -- a transient producer, a relation or
    view producer, a local relation, a transient pattern -- is not reported (a dead allowance would let a new read
    through unjudged: reports #59), if one in use is, or if a transient pattern matching only what another matches is
    not reported as covered by it."""
    tables = {"pkg/tables.py": _TABLES, "pkg/page.py": _PAGE}
    used = _unused(
        no_whole_reads_findings(
            _config(tmp_path, tables, transient_relations=r"^__stage", local_relations={"geos": ("geo_id",)})
        )
    )
    assert used == [], used
    dead = _unused(
        no_whole_reads_findings(
            _config(
                tmp_path,
                tables,
                transient_relations=(r"^__stage", r"^__nothing", r"^__stag"),
                local_relations={"geos": ("geo_id",), "towns": ("town_id",)},
            )
        )
    )
    assert "transient_relations ^__nothing matches nothing the walk read; delete it" in dead, dead
    assert "local_relations towns matches nothing the walk read; delete it" in dead, dead
    assert any(finding.startswith("transient_relations ^__stage matches only what ^__stag") for finding in dead), dead
    assert not any("local_relations geos" in finding for finding in dead), dead
    unchecked = _unused(
        no_whole_reads_findings(
            _config(tmp_path, tables, local_relations={"towns": ("town_id",)}, local_relations_checked=False)
        )
    )
    assert unchecked == [], unchecked

    produced = _unused(
        no_whole_reads_findings(
            _config(
                tmp_path,
                {"pkg/produced.py": _PRODUCED},
                transient_producers=frozenset({"Stages._stage", "_Work.put", "Gone.stage"}),
            )
        )
    )
    assert produced == ["transient_producers Gone.stage matches nothing the walk read; delete it"], produced

    views = _unused(
        no_whole_reads_findings(
            _config(
                tmp_path,
                {"views.py": _VIEW_PRODUCERS},
                view_producers={"_view_sql": ("office_key",), "Parts.copy_view": ("office_key",), "_gone": ("x",)},
                local_relations={"newest": ()},
            )
        )
    )
    assert "view_producers _gone matches nothing the walk read; delete it" in views, views
    assert not any("_view_sql" in finding or "copy_view" in finding for finding in views), views

    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_source.py").write_text("async def test_runs(lsf):\n    assert lsf\n", encoding="utf-8")
    with pytest.warns(UserWarning):
        relations = _unused(
            no_whole_reads_findings(
                _config(tmp_path, {"lsf.py": _SOURCE_DERIVED}, relation_producers={"long_derived": (), "gone": ()})
            )
        )
    assert relations == ["relation_producers gone matches nothing the walk read; delete it"], relations
