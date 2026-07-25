"""Tests guarding against ScrapeTarget/ScrapeRecipe/ScrapeExtraction <-> DDL drift.

The bug class: an entity exposes a persisted field with no matching DDL
column. Nothing fails until a real L3 store is involved, at which point the
first upsert raises ``asyncpg.UndefinedColumnError``. Every in-memory test in
this package is blind to it, because ``ScrapeCollection``'s fallback L3 is a
plain dict that ignores schema entirely.

So these tests do not use a collection at all. They run every registered
migration against a recording fake store, scrape the column names back out of
the captured SQL, and compare that against the fields the entity classes
actually expose -- derived by introspection, never restated by hand. The
hand-restated version of this file is why the guard was useless: its literal
field sets omitted ``link_selector`` exactly as the DDL did, so it agreed with
the bug instead of catching it (fixed alongside
``migrations.v009_target_link_selector``).
"""

from __future__ import annotations

import re

import pytest
from threetears.core.data.migrations import MigrationRunner
from threetears.core.entities.base import BaseEntity

from threetears.scrape.collections import ScrapeExtraction, ScrapeRecipe, ScrapeTarget
from threetears.scrape.migrations import register

_CREATE_COLUMN_RE = re.compile(
    r"^\s*(\w+)\s+(?:TEXT|INTEGER|FLOAT8|TIMESTAMPTZ|JSONB|BOOLEAN|BIGINT)\b",
    re.IGNORECASE | re.MULTILINE,
)
_ADD_COLUMN_RE = re.compile(r"ADD COLUMN IF NOT EXISTS\s+(\w+)", re.IGNORECASE)

#: Entity properties that are genuinely NOT backed by a column -- computed or
#: derived values with nothing to persist. Keyed by ``"<EntityClass>.<property>"``
#: so an exemption can never accidentally silence the same-named property on a
#: different entity.
#:
#: Deliberately empty today: every property on all three entities reads a raw
#: field via ``_get_raw()`` and therefore needs a column. The set exists as the
#: declared escape hatch -- the ONLY sanctioned way to exclude a property from
#: the coverage check below -- so that a future computed property is excluded
#: visibly, with a reviewable reason on the line, rather than by quietly editing
#: a hand-maintained list of field names (which is exactly how ``link_selector``
#: shipped with no DDL column while these tests sat green).
#:
#: Every entry MUST carry a trailing comment stating why the property has no
#: column, e.g.::
#:
#:     "ScrapeTarget.some_derived_flag",  # computed from multi_row + driver_backend, never stored
_NON_PERSISTED_PROPERTIES: frozenset[str] = frozenset()


def _persisted_fields(entity_cls: type[BaseEntity]) -> set[str]:
    """Return every persisted field name *entity_cls* exposes, by introspection.

    Walks the MRO and collects the name of every ``property`` descriptor
    declared on the entity's own classes, stopping short of
    :class:`~threetears.core.entities.base.BaseEntity` -- ``BaseEntity``'s own
    ``id``/``is_dirty``/``is_new`` are cache-proxy machinery describing the
    entity's state, not per-table columns, and would otherwise leak
    ``is_dirty``/``is_new`` into every table's expected column set.

    Filtering by DECLARING class (rather than by name) is deliberate: it keeps
    a property a subclass genuinely redefines. ``ScrapeExtraction.id`` shadows
    ``BaseEntity.id`` and IS a real ``scrape_extractions`` column, so a
    name-based exclusion would silently stop checking the primary key of the
    one table whose key isn't also a plain foreign/natural key elsewhere.

    Derived rather than hand-listed on purpose. The previous version of these
    tests restated each entity's fields as string literals, which meant a new
    persisted property was only covered once someone remembered to add it here
    too -- and when ``link_selector`` was added to ``ScrapeTarget`` nobody did,
    so the guard whose entire job is catching "field with no column" reported
    green while exactly that bug shipped.

    :param entity_cls: entity class to introspect
    :ptype entity_cls: type[BaseEntity]
    :return: persisted field names, minus any declared in :data:`_NON_PERSISTED_PROPERTIES`
    :rtype: set[str]
    """
    fields: set[str] = set()
    for klass in entity_cls.__mro__:
        if klass is BaseEntity:
            break
        for name, attr in vars(klass).items():
            if isinstance(attr, property) and f"{entity_cls.__name__}.{name}" not in _NON_PERSISTED_PROPERTIES:
                fields.add(name)
    return fields


# parity-exempt: hand-rolled subset stub of 3tears' DataStore (execute/query only) -- a real DataStore needs a live registry/pool, defeating the point of a fast, network-free unit test
class _FakeStore:
    """Records every SQL string passed to ``execute()`` -- never touches a database."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, sql: str, *params: object) -> str:
        self.executed.append(sql)
        return "OK"

    async def query(self, sql: str, *params: object) -> list[dict[str, object]]:
        self.executed.append(sql)
        return []


@pytest.fixture(scope="module")
def captured_ddl() -> list[str]:
    """Run every registered 3tears-scrape migration version against a fake
    store and return the full list of executed SQL strings, in registration
    order."""
    import asyncio

    async def _capture() -> list[str]:
        runner = MigrationRunner()
        pkg = register(runner)
        store = _FakeStore()
        for version_num in sorted(pkg.versions.keys()):
            await pkg.versions[version_num](store)
        return store.executed

    return asyncio.run(_capture())


def _ddl_columns(table_name: str, statements: list[str]) -> set[str]:
    """Return every column name defined for *table_name* across every captured statement."""
    columns: set[str] = set()
    for stmt in statements:
        if table_name not in stmt:
            continue
        columns.update(_CREATE_COLUMN_RE.findall(stmt))
        columns.update(_ADD_COLUMN_RE.findall(stmt))
    return columns


def test_registered_versions_are_sequential_starting_at_one():
    runner = MigrationRunner()
    pkg = register(runner)
    versions = sorted(pkg.versions.keys())
    assert versions == list(range(1, len(versions) + 1)), f"non-sequential versions: {versions}"


def test_target_fields_covered_by_ddl(captured_ddl: list[str]):
    """Every field ScrapeTarget exposes must have a matching scrape_targets column."""
    persisted_fields = _persisted_fields(ScrapeTarget)
    columns = _ddl_columns("scrape_targets", captured_ddl)
    missing = persisted_fields - columns
    assert not missing, f"ScrapeTarget fields with no matching scrape_targets DDL column: {missing}"


def test_recipe_fields_covered_by_ddl(captured_ddl: list[str]):
    """Every field ScrapeRecipe exposes must have a matching scrape_recipes column."""
    persisted_fields = _persisted_fields(ScrapeRecipe)
    columns = _ddl_columns("scrape_recipes", captured_ddl)
    missing = persisted_fields - columns
    assert not missing, f"ScrapeRecipe fields with no matching scrape_recipes DDL column: {missing}"


def test_extraction_fields_covered_by_ddl(captured_ddl: list[str]):
    """Every field ScrapeExtraction exposes must have a matching scrape_extractions column."""
    persisted_fields = _persisted_fields(ScrapeExtraction)
    columns = _ddl_columns("scrape_extractions", captured_ddl)
    missing = persisted_fields - columns
    assert not missing, f"ScrapeExtraction fields with no matching scrape_extractions DDL column: {missing}"


def test_introspection_actually_finds_each_entitys_fields():
    """The coverage tests above are only as good as :func:`_persisted_fields`.

    A silently-empty (or BaseEntity-polluted) derivation would make all three
    pass vacuously -- the same false-green failure mode the hand-maintained
    literal sets had, just arrived at differently. Assert the shape of what
    introspection returns directly: a known-real field is present, the entity's
    own primary key is present, and none of ``BaseEntity``'s cache-proxy
    machinery properties leak through.
    """
    target_fields = _persisted_fields(ScrapeTarget)
    assert "link_selector" in target_fields
    assert "target_id" in target_fields
    assert _persisted_fields(ScrapeRecipe) >= {"target_id", "extraction_strategy"}
    # ScrapeExtraction redefines ``id``; it is a real column and must survive the MRO walk.
    assert "id" in _persisted_fields(ScrapeExtraction)
    for entity_cls in (ScrapeTarget, ScrapeRecipe, ScrapeExtraction):
        leaked = _persisted_fields(entity_cls) & {"is_dirty", "is_new"}
        assert not leaked, f"{entity_cls.__name__}: BaseEntity machinery leaked into persisted fields: {leaked}"


def test_every_scrape_table_has_date_created_and_date_updated(captured_ddl: list[str]):
    """BaseCollection.save_entity() unconditionally stamps date_created/date_updated
    on every upsert regardless of what a collection's entity class exposes, so
    every scrape table must declare both from the start. Introspection cannot
    catch this one: neither column is a property on any entity class, which is
    precisely why it needs its own assertion rather than falling out of
    _persisted_fields()."""
    for table in ("scrape_targets", "scrape_recipes", "scrape_extractions"):
        columns = _ddl_columns(table, captured_ddl)
        assert "date_created" in columns, f"{table} is missing date_created"
        assert "date_updated" in columns, f"{table} is missing date_updated"
