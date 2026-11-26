"""Two reads 3tears makes at run time, judged as it makes them.

- A shared keyed read (``read_row``, ``read_by``, ``read_keyed``, ``keyset_rows``, ``counts_by``, ...) of a table a
  ``TableSchema(row_epoch=ROW_STAMPED)`` declares adds the table's live guard (``NOT gone``) beside its filters
  (``threetears.core.keyed.live_read``): the walk judges it with that guard, so a live-only partial index (``WHERE NOT
  gone``) holds every row it asks. A read with ``with_gone=True``, a pure builder, or a read of a table not declared
  stamped gets no guard.
- A collection's declared ``DeleteCascade(child, column, ...)`` / ``DeleteRestrict(...)``, imported from 3tears, is
  the keyed read of the child 3tears runs for it (``delete_cascade._children`` / ``_live_children``), judged where it
  is constructed, under the class attribute holding it (``<path>::<Class>.delete_cascades``), which an index's purpose
  names.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, no_whole_reads_findings
from threetears.enforcement.no_whole_reads.helpers import helper_statements, live_guarded

#: a finding naming one read, or one index's purpose
_READ: re.Pattern[str] = re.compile(r"^\S+:\d+ in (\S+): (\S+ \S+) -- ")
_PURPOSE: re.Pattern[str] = re.compile(r" index (\S+) on \S+: (index-purpose-\S+) -- ")


def _findings(tmp_path: Path, modules: dict[str, str]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """the reads the gate names, ``(function, symbol)``, and the purpose findings, ``(index, kind)``, over ``modules``
    written under ``tmp_path/src`` (3tears' own declarations not read: each test declares its tables)."""
    src = tmp_path / "src"
    for name, source in modules.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    found = no_whole_reads_findings(
        NoWholeReadsConfig(
            repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, minimum_files_scanned=1, dependency_schema=False
        )
    )
    reads = {(m.group(1), m.group(2)) for m in map(_READ.match, found) if m is not None}
    purposes = {(m.group(1), m.group(2)) for m in (_PURPOSE.search(f) for f in found) if m is not None}
    return reads, purposes


#: a stamped table (``models``) and its unstamped twin (``plain_models``), each with live-only partial indexes, read
#: through 3tears' keyed helpers
_MODELS = """
    from threetears.core.collections.schema_backed import Column, IndexDef, TableSchema
    from threetears.core.ddl.row_epoch import ROW_STAMPED
    from threetears.core.keyed import AnyOf, KeyBatch, Sort, keyset_rows, read_by_statement, read_keyed, read_row


    MODELS = TableSchema(
        name="models",
        row_epoch=ROW_STAMPED,
        primary_key="id",
        columns=[Column("id", "uuid"), Column("provider_id", "uuid"), Column("name", "string")],
        indexes=(
            IndexDef(
                name="models_provider_name",
                columns=("provider_id", "name"),
                unique=True,
                where="NOT gone",
                purpose="src/models.py::by_name; src/models.py::by_providers; src/models.py::page",
            ),
            IndexDef(
                name="models_utility",
                columns=("utility",),
                unique=True,
                where="utility = true AND NOT gone",
                purpose="src/models.py::utility",
            ),
        ),
    )
    PLAIN = TableSchema(
        name="plain_models",
        primary_key="id",
        columns=[Column("id", "uuid"), Column("provider_id", "uuid"), Column("name", "string")],
        indexes=(
            IndexDef(
                name="plain_models_provider_name",
                columns=("provider_id", "name"),
                unique=True,
                where="NOT gone",
                purpose="src/models.py::plain_by_name",
            ),
        ),
    )


    async def by_name(pool, provider_id, name):
        return await read_row(pool, "models", by={"provider_id": provider_id, "name": name})


    async def utility(pool):
        return await read_row(pool, "models", by={"utility": True})


    async def by_providers(pool, providers):
        return await read_keyed(pool, "models", keys=KeyBatch(("provider_id",), [providers], ("uuid",)))


    async def page(pool, provider_id):
        return await keyset_rows(pool, "models", by={"provider_id": provider_id}, order=[Sort("name"), Sort("id")])


    async def page_as_stored(pool, provider_id):
        return await keyset_rows(
            pool, "models", by={"provider_id": provider_id}, order=[Sort("name"), Sort("id")], with_gone=True
        )


    async def built(provider_id):
        return read_by_statement("models", by={"provider_id": provider_id})


    async def plain_by_name(pool, provider_id, name):
        return await read_row(pool, "plain_models", by={"provider_id": provider_id, "name": name})
"""


def test_a_keyed_helper_read_of_a_stamped_table_is_judged_with_its_live_guard(tmp_path: Path) -> None:
    """Flips if a shared keyed read of a ``ROW_STAMPED`` table is judged without the ``NOT gone`` 3tears adds at run
    time: its live-only partial index then never fits, and the read is named whole (hub v191's indexes)."""
    reads, purposes = _findings(tmp_path, {"models.py": _MODELS})
    for function in ("by_name", "utility", "by_providers", "page"):
        assert not {read for read in reads if read[0] == function}, (function, reads)
    assert ("models_provider_name", "index-purpose-unused") not in purposes, purposes
    assert ("models_utility", "index-purpose-unused") not in purposes, purposes


def test_a_read_3tears_sends_unguarded_gets_no_live_guard(tmp_path: Path) -> None:
    """Flips if the live guard is added where 3tears adds none: a read ``with_gone=True`` (the table as stored), a
    pure builder's statement (sent as built), or a read of a table no ``ROW_STAMPED`` declares. Each still asks rows a
    live-only partial index does not hold, so each is named."""
    reads, _ = _findings(tmp_path, {"models.py": _MODELS})
    assert any(function == "page_as_stored" for function, _ in reads), reads
    assert any(function == "built" for function, _ in reads), reads
    assert any(function == "plain_by_name" for function, _ in reads), reads


def test_a_row_stamped_of_the_module_s_own_is_no_stamped_declaration(tmp_path: Path) -> None:
    """Flips if any name spelled ``ROW_STAMPED`` marks a table stamped: only 3tears' own, imported from a 3tears
    module, does."""
    source = _MODELS.replace("from threetears.core.ddl.row_epoch import ROW_STAMPED\n", "ROW_STAMPED = object()\n")
    reads, _ = _findings(tmp_path, {"models.py": source})
    assert any(function == "by_name" for function, _ in reads), reads


def test_live_guarded_follows_with_gone_and_the_table_named() -> None:
    """Flips if :func:`live_guarded` guards a read whose ``with_gone`` is not a literal False, a table named at run
    time, a non-stamped table or a pure builder, or leaves a stamped read unguarded."""

    def sent(code: str) -> list[str | None]:
        call = ast.parse(code).body[0]
        assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
        guarded = live_guarded(call.value, lambda name: name == "models", "gone")
        return [statement.sql for statement in helper_statements(guarded)]

    def guarded(code: str) -> bool:
        return any('(NOT "gone")' in (sql or "") for sql in sent(code))

    assert guarded('read_row(c, "models", by={"id": i})')
    assert guarded('keyset_rows(c, "models", by={"p": p}, order=[Sort("id")], with_gone=False)')
    assert guarded('counts_by(c, "models", column="p", values=AnyOf(ps))')
    assert not guarded('keyset_rows(c, "models", by={"p": p}, order=[Sort("id")], with_gone=True)')
    assert not guarded('keyset_rows(c, "models", by={"p": p}, order=[Sort("id")], with_gone=flag)')
    assert not guarded('read_row(c, "plain_models", by={"id": i})')
    assert not guarded('read_row(c, table, by={"id": i})')
    assert not guarded('read_by_statement("models", by={"id": i})')


#: a stamped child (``kids``) indexed live-only on ``parent_id`` for its parent's declared actions, and an unstamped
#: child (``notes``) with no index on ``page_id``
_CHILDREN = """
    from threetears.core.collections.schema_backed import Column, IndexDef, TableSchema
    from threetears.core.ddl.row_epoch import ROW_STAMPED

    KIDS = TableSchema(
        name="kids",
        row_epoch=ROW_STAMPED,
        primary_key="id",
        columns=[Column("id", "uuid"), Column("parent_id", "uuid"), Column("other_id", "uuid")],
        indexes=(
            IndexDef(
                name="kids_parent",
                columns=("parent_id",),
                where="NOT gone",
                purpose="src/parents.py::Parents.delete_cascades; src/parents.py::Parents.delete_restricts",
            ),
        ),
    )
    NOTES = TableSchema(name="notes", primary_key="id", columns=[Column("id", "uuid"), Column("page_id", "uuid")])
"""

_PARENTS = """
    from typing import ClassVar

    from threetears.core.collections import DeleteCascade, DeleteRestrict


    class Parents:
        delete_cascades: ClassVar[tuple[DeleteCascade, ...]] = (DeleteCascade("kids", "parent_id", "CASCADE"),)
        delete_restricts: ClassVar[tuple[DeleteRestrict, ...]] = (DeleteRestrict("kids", "parent_id", "NO ACTION"),)
"""


def test_a_declared_cascade_is_the_keyed_read_of_its_child_named_by_its_attribute(tmp_path: Path) -> None:
    """Flips if a declared ``DeleteCascade`` / ``DeleteRestrict`` is not judged as the keyed read 3tears runs for it
    (a stamped child's with its live guard), or if an index's purpose naming the class attribute that holds it is
    unknown or unused."""
    reads, purposes = _findings(tmp_path, {"children.py": _CHILDREN, "parents.py": _PARENTS})
    assert not reads, reads
    assert not {purpose for purpose in purposes if purpose[0] == "kids_parent"}, purposes


def test_a_declared_cascade_no_index_serves_is_named_at_its_construction(tmp_path: Path) -> None:
    """Flips if a declared action whose child column no purposed index leads with passes: its read pages (or counts)
    every row of the child to find the parent's, so it is named where it is declared."""
    parents = """
        from threetears.core.collections import DeleteCascade, DeleteRestrict


        class Parents:
            delete_cascades = (DeleteCascade("kids", "other_id", "SET NULL"), DeleteCascade("notes", "page_id", "CASCADE"))
            delete_restricts = (DeleteRestrict("notes", "page_id", "NO ACTION"),)
    """
    reads, purposes = _findings(tmp_path, {"children.py": _CHILDREN, "parents.py": parents})
    assert ("Parents", 'non-key-where "kids"') in reads, reads
    assert ("Parents", 'non-key-where "notes"') in reads, reads
    assert ("Parents", 'whole-aggregate "notes"') in reads, reads
    # the attributes its purpose names hold no read the index serves
    assert ("kids_parent", "index-purpose-unused") in purposes, purposes


def test_a_declared_cascade_named_at_run_time_is_unknown(tmp_path: Path) -> None:
    """Flips if a declared action whose child table the walk cannot read as a string passes."""
    parents = """
        from threetears.core.collections import DeleteCascade


        def cascades(table):
            return (DeleteCascade(table, "parent_id", "CASCADE"),)
    """
    reads, _ = _findings(tmp_path, {"children.py": _CHILDREN, "parents.py": parents})
    assert any(
        function == "cascades" and symbol.startswith("unknown-predicate DeleteCascade") for function, symbol in reads
    ), reads


def test_a_repo_holding_3tears_delete_cascade_judges_the_reads_there_not_at_each_construction(tmp_path: Path) -> None:
    """Flips if a walk holding 3tears' own ``delete_cascade`` module (whose reads it judges once per construction)
    judges each construction a second time."""
    parents = """
        from threetears.core.collections import DeleteCascade


        class Parents:
            delete_cascades = (DeleteCascade("notes", "page_id", "CASCADE"),)
    """
    modules = {
        "children.py": _CHILDREN,
        "parents.py": parents,
        "threetears/core/collections/delete_cascade.py": '"""stand-in for 3tears\' own module."""\n',
    }
    reads, _ = _findings(tmp_path, modules)
    assert not {read for read in reads if read[0] == "Parents"}, reads
