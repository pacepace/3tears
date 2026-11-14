"""The index declarations ledger: an index whose DDL sits in a frozen migration declares its purpose, plan test and
reason in ``_index_declarations.txt``, judged by the same checks as comments beside its DDL, plus the ledger's own."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, no_whole_reads_findings

_MIGRATION = """
    UP = "CREATE TABLE IF NOT EXISTS datasources (id UUID PRIMARY KEY, kind TEXT, status TEXT, customer_id UUID)"
    BY_KIND = "CREATE INDEX IF NOT EXISTS datasources_by_kind ON datasources (kind)"
"""

_INLINE = """
    UP = "CREATE TABLE IF NOT EXISTS datasources (id UUID PRIMARY KEY, kind TEXT, status TEXT, customer_id UUID)"
    BY_KIND = (
        "-- purpose: src/registry.py::Digest.refresh_active\\n"
        "-- plan_test: tests/test_plans.py::test_refresh_active_uses_datasources_by_kind\\n"
        "CREATE INDEX IF NOT EXISTS datasources_by_kind ON datasources (kind)"
    )
"""

_REGISTRY = """
    class Digest:
        async def one(self, conn, datasource_id):
            return await conn.fetchrow("SELECT id, kind FROM datasources WHERE id = $1", datasource_id)

        async def refresh_active(self, conn, kind):
            return await conn.fetch("SELECT id, customer_id FROM datasources WHERE status = 'active' AND kind = $1", kind)
"""

_PLANS = """
    from threetears.core.testing.plans import explain_statement, plan_uses_index

    async def test_refresh_active_uses_datasources_by_kind(conn):
        sql = "SELECT id, customer_id FROM datasources WHERE status = 'active' AND kind = $1"
        assert plan_uses_index(await explain_statement(conn, sql, "pg"), "datasources_by_kind")
"""

_LINE = (
    "datasources_by_kind :: on datasources :: purpose=src/registry.py::Digest.refresh_active :: "
    "plan_test=tests/test_plans.py::test_refresh_active_uses_datasources_by_kind"
)

_FROZEN = ("src/versions/v001.py",)


def _findings(
    tmp_path: Path, ledger: str | None, *, migration: str = _MIGRATION, frozen: tuple[str, ...] = _FROZEN
) -> list[str]:
    """the gate's findings over a migration, a reader and its plan test, with ``ledger`` as the declarations."""
    src = tmp_path / "src"
    for name, source in {"versions/v001.py": migration, "registry.py": _REGISTRY}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_plans.py").write_text(textwrap.dedent(_PLANS), encoding="utf-8")
    (tmp_path / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    if ledger is not None:
        (tmp_path / "_index_declarations.txt").write_text(f"# frozen v001\n{ledger}\n", encoding="utf-8")
    config = NoWholeReadsConfig(
        repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False, frozen_migrations=frozen
    )
    return no_whole_reads_findings(config)


def _index(findings: list[str]) -> list[str]:
    return [finding for finding in findings if ": index-" in finding]


def test_without_a_declaration_the_frozen_index_has_no_purpose_and_its_read_is_whole(tmp_path: Path) -> None:
    """Flips if an index in a frozen migration with no declaration anywhere passes, or its read is keyed by it."""
    findings = _findings(tmp_path, None)
    assert any("index datasources_by_kind on datasources: index-no-purpose" in f for f in findings), findings
    assert any("in Digest.refresh_active: non-key-where datasources" in f for f in findings), findings


def test_a_correct_frozen_entry_declares_the_index_and_keys_its_read(tmp_path: Path) -> None:
    """Flips if a ledger line for an index in a frozen migration, naming the read that uses it and its plan test, does
    not clear index-no-purpose, or leaves that read a whole read."""
    findings = _findings(tmp_path, _LINE)
    assert _index(findings) == [], findings
    assert not [f for f in findings if "Digest.refresh_active" in f], findings


def test_an_entry_for_an_index_in_no_frozen_migration_is_refused(tmp_path: Path) -> None:
    """Flips if the ledger may declare an index whose DDL can still be edited (no frozen migration holds it)."""
    findings = _findings(tmp_path, _LINE, frozen=())
    assert any("index-ledger-not-frozen" in f for f in findings), findings
    assert any("index-no-purpose" in f for f in findings), "a refused line declared the index anyway"


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        (_LINE.replace("datasources_by_kind ::", "datasources_by_gone ::"), "index-ledger-stale"),
        (_LINE.replace("on datasources", "on other_table"), "index-ledger-stale"),
        (_LINE.replace("Digest.refresh_active", "Digest.one"), "index-purpose-unused"),
        (_LINE.replace("Digest.refresh_active", "Digest.gone"), "index-purpose-unknown"),
        (_LINE.replace("test_refresh_active_uses_datasources_by_kind", "test_not_there"), "index-plan-test"),
        (_LINE.replace(" :: purpose=", " purpose="), "index-ledger-malformed"),
    ],
)
def test_a_wrong_entry_is_reported(tmp_path: Path, line: str, kind: str) -> None:
    """Flips if a stale entry (an index no migration creates, or on another table), a purpose whose function holds no
    keyed read by the index or names no function, a plan test that is not there, or a malformed line passes."""
    findings = _findings(tmp_path, line)
    assert any(kind in f for f in findings), findings


_TABLE = '    UP = "CREATE TABLE IF NOT EXISTS datasources (id UUID PRIMARY KEY, kind TEXT, status TEXT, customer_id UUID)"\n'

#: an index named by an f-string of a builder's parameter, each table the module's calls pass (hub v019's shape)
_BY_CALLS = (
    _TABLE
    + """
    def _by_kind_sql(table: str) -> str:
        return f"CREATE INDEX IF NOT EXISTS {table}_by_kind ON {table} (kind)"

    STATEMENTS = (_by_kind_sql("datasources"),)
"""
)

#: the same f-string in a loop over a literal tuple
_BY_LOOP = (
    _TABLE
    + """
    TABLES = ("datasources",)

    def statements() -> list[str]:
        out = []
        for table in TABLES:
            out.append(f"CREATE INDEX IF NOT EXISTS {table}_by_kind ON {table} (kind)")
        return out
"""
)

#: a builder called from a loop over literals: the call's argument is the loop's name
_BY_LOOPED_CALLS = (
    _TABLE
    + """
    def _by_kind_sql(table: str) -> str:
        return f"CREATE INDEX IF NOT EXISTS {table}_by_kind ON {table} (kind)"

    STATEMENTS = [_by_kind_sql(name) for name in ("datasources",)]
"""
)

#: the loop iterates what only run time knows
_BY_RUN_TIME = (
    _TABLE
    + """
    def statements(tables: list[str]) -> list[str]:
        return [f"CREATE INDEX IF NOT EXISTS {table}_by_kind ON {table} (kind)" for table in tables]

    STATEMENTS = statements(load_tables())
"""
)


@pytest.mark.parametrize("migration", [_BY_CALLS, _BY_LOOP, _BY_LOOPED_CALLS], ids=["calls", "loop", "looped-calls"])
def test_an_f_string_index_name_bound_to_literals_is_declared_by_its_line(tmp_path: Path, migration: str) -> None:
    """Flips if an index whose name an f-string builds from a name bound only to literals (a builder's calls, a loop
    over a literal tuple) is not seen, so its ledger line is refused as stale and its read stays whole."""
    findings = _findings(tmp_path, _LINE, migration=migration)
    assert _index(findings) == [], findings
    assert not [f for f in findings if "Digest.refresh_active" in f], findings


def test_an_f_string_index_name_bound_at_run_time_stays_unknown(tmp_path: Path) -> None:
    """Flips if an f-string index name over a loop of run-time values is read as a concrete index (its line accepted,
    its read keyed)."""
    findings = _findings(tmp_path, _LINE, migration=_BY_RUN_TIME)
    assert any("index-ledger-stale" in f for f in findings), findings
    assert any("in Digest.refresh_active: non-key-where datasources" in f for f in findings), findings


def test_an_index_declared_inline_and_in_the_ledger_is_declared_twice(tmp_path: Path) -> None:
    """Flips if an index carrying its comments beside its DDL may be declared in the ledger too."""
    findings = _findings(tmp_path, _LINE, migration=_INLINE)
    assert any("index-declared-twice" in f for f in findings), findings


def test_a_crowded_or_wide_index_s_reason_comes_from_its_line(tmp_path: Path) -> None:
    """Flips if a ledger line's reason is not the index's reason (a short one passes a wide index)."""
    wide = _MIGRATION.replace("(kind)", "(kind) INCLUDE (customer_id)")
    assert any("index-wide-no-reason" in f for f in _findings(tmp_path, _LINE, migration=wide))
    reasoned = f"{_LINE} :: reason=the digest answers each active datasource's customer from the index alone"
    assert _index(_findings(tmp_path, reasoned, migration=wide)) == []


def test_an_index_declared_on_two_lines_is_declared_twice(tmp_path: Path) -> None:
    """Flips if a ledger may declare one index on two lines."""
    findings = _findings(tmp_path, f"{_LINE}\n{_LINE}")
    assert any("index-declared-twice -- datasources_by_kind is declared on line 2 too" in f for f in findings), findings


_WIDE_INLINE = _INLINE.replace("(kind)", "(kind) INCLUDE (customer_id)")

_REASON = "the digest answers each active datasource's customer from the index alone"

_SUPPLEMENT = f"datasources_by_kind :: on datasources :: reason={_REASON}"


def test_a_reason_line_supplements_a_frozen_index_declared_inline(tmp_path: Path) -> None:
    """Flips if a frozen index whose purpose and plan test sit beside its DDL, but which needs a reason its migration
    cannot be edited to give, cannot take that reason from a ledger line naming only the reason."""
    assert any("index-wide-no-reason" in f for f in _findings(tmp_path, None, migration=_WIDE_INLINE))
    findings = _findings(tmp_path, _SUPPLEMENT, migration=_WIDE_INLINE)
    assert _index(findings) == [], findings


@pytest.mark.parametrize(
    ("line", "migration", "frozen", "kind"),
    [
        # too short to be a reason
        ("datasources_by_kind :: on datasources :: reason=wide on purpose", _WIDE_INLINE, _FROZEN, "index-supplement"),
        # the index's DDL is not frozen: the reason goes beside it
        (_SUPPLEMENT, _WIDE_INLINE, (), "index-ledger-not-frozen"),
        # a supplement names no purpose: purpose= on it is refused
        (
            f"datasources_by_kind :: on datasources :: purpose=src/registry.py::Digest.refresh_active :: reason={_REASON}",
            _WIDE_INLINE,
            _FROZEN,
            "index-ledger-malformed",
        ),
        # nor a plan test alone
        (
            "datasources_by_kind :: on datasources :: plan_test=tests/test_plans.py::"
            f"test_refresh_active_uses_datasources_by_kind :: reason={_REASON}",
            _WIDE_INLINE,
            _FROZEN,
            "index-ledger-malformed",
        ),
        # a whole line on an index declared inline stays declared twice
        (f"{_LINE} :: reason={_REASON}", _WIDE_INLINE, _FROZEN, "index-declared-twice"),
        # nothing inline to supplement: the index is declared whole on its line, or not at all
        (_SUPPLEMENT, _MIGRATION.replace("(kind)", "(kind) INCLUDE (customer_id)"), _FROZEN, "index-supplement"),
        # the inline declaration gives its own reason: one each
        (
            _SUPPLEMENT,
            _WIDE_INLINE.replace('"-- purpose:', f'"-- reason: {_REASON}\\n"\n        "-- purpose:'),
            _FROZEN,
            "index-declared-twice",
        ),
        # stale
        (_SUPPLEMENT.replace("datasources_by_kind ::", "datasources_by_gone ::"), _WIDE_INLINE, _FROZEN, "stale"),
    ],
    ids=["short", "not-frozen", "purpose", "plan-test", "whole-line", "nothing-inline", "inline-reason", "stale"],
)
def test_a_wrong_reason_supplement_is_refused(
    tmp_path: Path, line: str, migration: str, frozen: tuple[str, ...], kind: str
) -> None:
    """Flips if a reason-only line passes when its reason is under 30 characters, its index is not frozen, it carries
    purpose= or plan_test= (a supplement gives the reason only), the index has no inline declaration to supplement,
    the inline declaration already gives a reason, or no migration creates the index; and if a refused line's reason
    is applied anyway."""
    findings = _findings(tmp_path, line, migration=migration, frozen=frozen)
    assert any(kind in f for f in findings), findings
    if "-- reason:" not in migration:
        assert any("index-wide-no-reason" in f for f in findings), f"a refused line's reason was applied: {findings}"


_SECOND_READER = """
    class Other:
        async def refresh_other(self, conn, kind):
            return await conn.fetch("SELECT id FROM datasources WHERE status = 'active' AND kind = $1", kind)
"""

_AGAIN = """
    BY_KIND = (
        "-- purpose: src/other.py::Other.refresh_other\\n"
        "-- plan_test: tests/test_plans.py::test_refresh_active_uses_datasources_by_kind\\n"
        "CREATE INDEX IF NOT EXISTS datasources_by_kind ON datasources (kind)"
    )
"""


def test_a_second_declaration_of_an_index_adds_its_purposes(tmp_path: Path) -> None:
    """Flips if a second declaration of the same index (a consumer's own DDL of an index its dependency declares) loses
    its purposes to the first: each declaration's reads are the index's (hub #55 S1)."""
    src = tmp_path / "src"
    for name, source in {"versions/v002.py": _AGAIN, "other.py": _SECOND_READER}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    findings = _findings(tmp_path, None, migration=_INLINE)
    assert not [finding for finding in findings if "refresh_other" in finding], findings
    assert not _index(findings), findings


_CONSUMER = """
    class Listing:
        async def open_of_agent(self, conn, agent_id):
            return await conn.fetch(
                "SELECT conversation_id FROM conversations WHERE agent_id = $1 AND status = $2", agent_id, "open"
            )

        async def by_title(self, conn, title):
            return await conn.fetch("SELECT conversation_id FROM conversations WHERE title = $1", title)
"""


def _consumer_findings(tmp_path: Path, ledger: str) -> list[str]:
    """the gate's findings over a consumer that reads 3tears' conversations table through its declarations."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "listing.py").write_text(textwrap.dedent(_CONSUMER), encoding="utf-8")
    (tmp_path / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    (tmp_path / "_index_declarations.txt").write_text(f"{ledger}\n", encoding="utf-8")
    config = NoWholeReadsConfig(repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path)
    return [finding for finding in no_whole_reads_findings(config) if "conversations" in finding or "index-" in finding]


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("idx_conv_agent_status :: on conversations :: purpose=src/listing.py::Listing.open_of_agent", None),
        (
            "idx_conv_agent_status :: on conversations :: purpose=src/listing.py::Listing.by_title",
            "index-purpose-unused",
        ),
        ("idx_conv_agent_status :: on conversations :: purpose=src/listing.py::Listing.gone", "index-purpose-unknown"),
        ("idx_conv_gone :: on conversations :: purpose=src/listing.py::Listing.open_of_agent", "index-ledger-stale"),
    ],
)
def test_a_consumer_adds_its_read_to_a_dependency_s_index_purpose(tmp_path: Path, line: str, kind: str | None) -> None:
    """Flips if a purpose-only line naming the consumer's own read is refused as a re-declaration of the dependency's
    index, does not key that read, or adds a read that does not use the index (or names none, or an index the
    dependency does not declare) without a finding. Coordinator ruling 2026-11-13, for the hub."""
    findings = _consumer_findings(tmp_path, line)
    assert not [f for f in findings if "index-declared-twice" in f or "index-ledger-malformed" in f], findings
    keyed = not any("in Listing.open_of_agent:" in f for f in findings)
    assert keyed is (kind is None), findings
    if kind is not None:
        assert any(kind in f for f in findings), findings


_FROZEN_COPY = """
    CONVERSATIONS_INDEX = "CREATE INDEX IF NOT EXISTS idx_conv_agent_status ON conversations (agent_id, status, conversation_id)"
"""


@pytest.mark.parametrize(("frozen", "refused"), [(("src/versions/v019.py",), False), ((), True)])
def test_a_consumer_s_frozen_copy_of_a_dependency_s_index_takes_a_purpose_only_line(
    tmp_path: Path, frozen: tuple[str, ...], refused: bool
) -> None:
    """Flips if a purpose-only line is refused for an index the consumer's frozen migration also creates while the
    dependency declares it inline (hub #62 spec 1, form A), or if it is accepted where the consumer's own copy is not
    frozen (that one is declared where its DDL is)."""
    src = tmp_path / "src"
    (src / "versions").mkdir(parents=True)
    (src / "versions" / "v019.py").write_text(textwrap.dedent(_FROZEN_COPY), encoding="utf-8")
    (src / "listing.py").write_text(textwrap.dedent(_CONSUMER), encoding="utf-8")
    (tmp_path / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    (tmp_path / "_index_declarations.txt").write_text(
        "idx_conv_agent_status :: on conversations :: purpose=src/listing.py::Listing.open_of_agent\n", encoding="utf-8"
    )
    config = NoWholeReadsConfig(repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, frozen_migrations=frozen)
    findings = no_whole_reads_findings(config)
    assert any("index-addition-refused" in f for f in findings) is refused, findings
    if not refused:
        assert not any("in Listing.open_of_agent:" in f for f in findings), findings


_CUSTOMER_PAGES = """
    UP = "CREATE TABLE IF NOT EXISTS datasources (id UUID PRIMARY KEY, kind TEXT, status TEXT, customer_id UUID)"
    BY_CUSTOMER = (
        "-- purpose: src/registry.py::Digest.of_customer; src/registry.py::Digest.page_of_customer\\n"
        "-- plan_test: tests/test_plans.py::test_refresh_active_uses_datasources_by_kind\\n"
        "-- reason: one customer's datasources, read whole by the customer and paged by kind under it\\n"
        "CREATE INDEX IF NOT EXISTS datasources_by_customer_kind ON datasources (customer_id, kind, id)"
    )
"""

_CUSTOMER_READ = """
    class Digest:
        async def of_customer(self, conn, customer):
            return await conn.fetch("SELECT id, kind FROM datasources WHERE customer_id = $1", customer)

        async def page_of_customer(self, conn, customer):
            return await conn.fetch(
                "SELECT id, kind FROM datasources WHERE customer_id = $1 ORDER BY status LIMIT 5", customer
            )

        async def of_any_customer(self, conn, customer):
            return await conn.fetch("SELECT id, kind FROM datasources WHERE customer_id = $1 AND status = 'x'", customer)
"""


def test_a_purposed_index_whose_lead_the_read_binds_keys_it(tmp_path: Path) -> None:
    """Flips if a read binding by equality the whole first column of an index whose purpose names it (every row of
    that lead value, as an index on the column alone reads) is not keyed by it, or its purpose is called unused; or if
    the same read is keyed by such an index whose purpose does not name it, or a page under a LIMIT ordered by a column
    the index does not hold is keyed by it."""
    src = tmp_path / "src"
    for name, source in {"versions/v001.py": _CUSTOMER_PAGES, "registry.py": _CUSTOMER_READ}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_plans.py").write_text(textwrap.dedent(_PLANS), encoding="utf-8")
    (tmp_path / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    config = NoWholeReadsConfig(repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False)
    findings = no_whole_reads_findings(config)
    assert not [f for f in findings if "Digest.of_customer:" in f], findings
    unused = [f for f in findings if "index-purpose-unused" in f]
    assert unused and all("Digest.page_of_customer holds no read" in f for f in unused), unused
    assert any("in Digest.of_any_customer: non-key-where datasources" in f for f in findings), findings
    # under a LIMIT the page's order is not the index's: it sorts the customer's every row
    assert any("in Digest.page_of_customer:" in f for f in findings), findings


@pytest.mark.parametrize(
    ("dropped", "left"),
    [
        ('ALTER TABLE datasources DROP CONSTRAINT IF EXISTS "datasources_kind_key"', False),
        ("ALTER TABLE datasources DROP CONSTRAINT datasources_kind_key, DROP COLUMN status", False),
        ("ALTER TABLE datasources DROP CONSTRAINT IF EXISTS datasources_kind_check", True),
    ],
)
def test_a_dropped_constraint_takes_the_index_named_as_it(tmp_path: Path, dropped: str, left: bool) -> None:
    """Flips if a later migration's ``ALTER TABLE ... DROP CONSTRAINT <name>`` leaves the index that constraint owns
    (a ``UNIQUE`` constraint's, named as it) to be judged, or drops an index named otherwise."""
    src = tmp_path / "src"
    migrations = {
        "versions/v001.py": """
            UP = "CREATE TABLE IF NOT EXISTS datasources (id UUID PRIMARY KEY, kind TEXT, status TEXT)"
            KIND = "CREATE UNIQUE INDEX IF NOT EXISTS datasources_kind_key ON datasources (kind)"
        """,
        "versions/v002.py": f"""
            DOWN = '{dropped}'
        """,
        "registry.py": "",
    }
    for name, source in migrations.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    config = NoWholeReadsConfig(repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False)
    named = [f for f in _index(no_whole_reads_findings(config)) if "datasources_kind_key" in f]
    assert bool(named) == left, named
