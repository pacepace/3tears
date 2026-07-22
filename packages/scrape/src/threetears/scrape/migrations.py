"""3tears-scrape's own schema migrations, registered with 3tears' canonical ``MigrationRunner``.

Lives inside ``src/faidh/scrape/`` (not ``src/faidh/db/migrations.py``) so this
package's DDL travels with it when scrape is lifted into a real 3tears
package -- a directory move, not a disentangling exercise. Zero faidh
imports, mirroring the rest of this package's lift-readiness discipline
(enforced by ``tests/enforcement/test_scrape_no_faidh_imports.py``).

Registered under its own ``PACKAGE_NAME`` ("3tears_scrape") so its
``_schema_migrations`` history is distinct from faidh's own ("faidh")
package rows, even though both apply against the same PLATFORM schema.
"""

from __future__ import annotations

import uuid
from typing import Any

import uuid_utils
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner, MigrationScope, PackageMigrations
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["PACKAGE_NAME", "apply_migrations", "register"]

log = get_logger(__name__)

PACKAGE_NAME = "3tears_scrape"


async def v001_create_scrape_tables(store: DataStore) -> None:
    """Create ``scrape_targets`` / ``scrape_recipes`` / ``scrape_extractions``.

    Column shape matches ``ScrapeTarget`` / ``ScrapeRecipe`` / ``ScrapeExtraction``
    (``src/faidh/scrape/collections.py``) exactly. ``date_created``/``date_updated``
    included on every table from the start -- ``BaseCollection.save_entity()``
    unconditionally stamps both on every upsert regardless of what a collection's
    entity class exposes, so omitting them would raise
    ``asyncpg.UndefinedColumnError`` on the first real write (the exact failure
    mode faidh's own v018/v019/v022/v023 migrations document and fixed).
    """
    await store.execute("""
        CREATE TABLE IF NOT EXISTS scrape_targets (
            target_id      TEXT        NOT NULL,
            url            TEXT        NOT NULL,
            driver_backend TEXT        NOT NULL DEFAULT 'nodriver',
            rate_limit_key TEXT        NOT NULL DEFAULT '',
            cadence        TEXT        NOT NULL DEFAULT '',
            date_created   TIMESTAMPTZ,
            date_updated   TIMESTAMPTZ,
            PRIMARY KEY (target_id)
        )
    """)
    await store.execute("""
        CREATE TABLE IF NOT EXISTS scrape_recipes (
            target_id                        TEXT        NOT NULL,
            extraction_strategy               JSONB       NOT NULL DEFAULT '{}'::jsonb,
            won_at                            TIMESTAMPTZ,
            last_validated_at                 TIMESTAMPTZ,
            consecutive_validation_failures    INTEGER     NOT NULL DEFAULT 0,
            date_created                      TIMESTAMPTZ,
            date_updated                      TIMESTAMPTZ,
            PRIMARY KEY (target_id)
        )
    """)
    await store.execute("""
        CREATE TABLE IF NOT EXISTS scrape_extractions (
            id                    TEXT        NOT NULL,
            target_id             TEXT        NOT NULL,
            extraction_recipe_id  TEXT,
            source_url            TEXT        NOT NULL DEFAULT '',
            retrieved_at          TIMESTAMPTZ,
            structured_fields     JSONB       NOT NULL DEFAULT '{}'::jsonb,
            field_confidences     JSONB,
            enrichment_notes      JSONB,
            validation_status     TEXT        NOT NULL DEFAULT 'needs_review',
            date_created          TIMESTAMPTZ,
            date_updated          TIMESTAMPTZ,
            PRIMARY KEY (id)
        )
    """)
    await store.execute("CREATE INDEX IF NOT EXISTS scrape_extractions_target_id ON scrape_extractions (target_id)")
    await store.execute(
        "CREATE INDEX IF NOT EXISTS scrape_extractions_retrieved_at ON scrape_extractions (retrieved_at DESC)"
    )


async def v002_target_multi_row_flag(store: DataStore) -> None:
    """SCR-6P2X -- ``ScrapeTarget.multi_row`` selects which eval loop
    ``poll_scrape_targets`` runs (``run_eval_loop_multi_row`` vs. the
    original single-record ``run_eval_loop``). Defaults ``false`` so every
    pre-existing target keeps its current (single-record) behavior.
    """
    await store.execute("ALTER TABLE scrape_targets ADD COLUMN IF NOT EXISTS multi_row BOOLEAN NOT NULL DEFAULT false")


async def v003_target_wait_for(store: DataStore) -> None:
    """SCR-2N8W follow-up -- ``ScrapeTarget.wait_for`` is a CSS selector the
    driver waits for before considering the page settled, passed straight
    through to ``ScrapeDriver.render(..., wait_for=...)``. Nullable; ``None``
    keeps every pre-existing target's current behavior (a plain settle
    sleep). Live-verified need: Nebraska's WARN listing returns a near-empty
    page without a longer, selector-gated wait.
    """
    await store.execute("ALTER TABLE scrape_targets ADD COLUMN IF NOT EXISTS wait_for TEXT")


async def v004_target_field_schema(store: DataStore) -> None:
    """``ScrapeTarget.field_schema`` -- field_name -> type-name string (e.g.
    ``{"employer": "str"}``), the eval loop's per-target extraction schema.

    Consolidates what used to be a caller-supplied-only parameter onto the
    target itself, on direct instruction: once target config needed to
    round-trip through YAML and a database (not just live in a Python
    dict), a target's config and its schema had to be one unit, not two
    dicts a test had to keep in sync by hand. ``type`` objects aren't
    JSON-safe -- see ``collections.encode_field_schema``/``decode_field_schema``.
    """
    await store.execute(
        "ALTER TABLE scrape_targets ADD COLUMN IF NOT EXISTS field_schema JSONB NOT NULL DEFAULT '{}'::jsonb"
    )


async def v005_target_nav_steps(store: DataStore) -> None:
    """``ScrapeTarget.nav_steps`` -- ordered browser actions (click/fill/
    wait_for/wait_ms) the driver performs before the page is considered
    ready, passed straight through to ``ScrapeDriver.render(...,
    nav_steps=...)``. Nullable; ``None`` keeps every pre-existing target's
    current behavior (plain navigation, no interaction). Multi-step
    navigation capability (2026-07-14) -- see ``driver.NavStep``/
    ``collections.encode_nav_steps``/``decode_nav_steps``.
    """
    await store.execute("ALTER TABLE scrape_targets ADD COLUMN IF NOT EXISTS nav_steps JSONB")


async def v006_target_extraction_strategy_type(store: DataStore) -> None:
    """``ScrapeTarget.extraction_strategy_type`` -- ``"css"`` or ``"regex"``,
    which extraction-strategy shape the eval loop proposes (CSS selectors
    against an HTML table, or regex patterns against the page's plain text
    for a text-block/prose listing with no table structure). Defaults
    ``'css'`` so every pre-existing target keeps its current behavior.
    Regex/text-block extraction capability (2026-07-14) -- see
    ``eval_loop.StrategyType``.
    """
    await store.execute(
        "ALTER TABLE scrape_targets ADD COLUMN IF NOT EXISTS extraction_strategy_type TEXT NOT NULL DEFAULT 'css'"
    )


async def v007_target_api_config(store: DataStore) -> None:
    """``ScrapeTarget.api_results_path``/``api_fragment_field`` -- required
    when ``driver_backend == "api"``: the dotted JSON path to the list of
    per-record objects, and which field within each holds the HTML/text
    fragment to concatenate into a synthetic page. Both nullable; ``None``
    is fine for every non-``"api"`` target. Network/API-query capability
    (2026-07-14) -- see ``drivers.api.ApiDriver``.
    """
    await store.execute("ALTER TABLE scrape_targets ADD COLUMN IF NOT EXISTS api_results_path TEXT")
    await store.execute("ALTER TABLE scrape_targets ADD COLUMN IF NOT EXISTS api_fragment_field TEXT")


async def v008_target_timeout_seconds(store: DataStore) -> None:
    """``ScrapeTarget.timeout_seconds`` -- seconds to wait for this target's
    render before failing. Defaults to 30.0, the value every pre-existing
    target already got hardcoded at the call site, so every existing row
    keeps its current behavior. A target whose own ``nav_steps`` include a
    long ``wait_ms`` (Oklahoma's Salesforce Aura page, needing 15s alone
    just for its real data call to fire) can need more (network_capture
    capability, 2026-07-15) -- see ``drivers.network_capture.NetworkCaptureDriver``.
    """
    await store.execute(
        "ALTER TABLE scrape_targets ADD COLUMN IF NOT EXISTS timeout_seconds FLOAT8 NOT NULL DEFAULT 30.0"
    )


def register(runner: MigrationRunner) -> PackageMigrations:
    """Register every 3tears-scrape migration version with the given runner.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(name=PACKAGE_NAME, scope=MigrationScope.PLATFORM)
    pkg.version(1)(v001_create_scrape_tables)
    pkg.version(2)(v002_target_multi_row_flag)
    pkg.version(3)(v003_target_wait_for)
    pkg.version(4)(v004_target_field_schema)
    pkg.version(5)(v005_target_nav_steps)
    pkg.version(6)(v006_target_extraction_strategy_type)
    pkg.version(7)(v007_target_api_config)
    pkg.version(8)(v008_target_timeout_seconds)
    runner.register(pkg)
    return pkg


async def apply_migrations(pool: Any) -> None:
    """Apply every pending 3tears-scrape migration against ``pool`` via MigrationRunner.

    Mirrors ``faidh.db.migrations.apply_migrations`` exactly: a throwaway
    registry/config bound to ``pool`` and a ``DataStore`` wrapping it.
    ``DataStore`` requires an ``agent_id`` (3tears' per-agent-schema concept),
    inert here for the same reason documented on the faidh-side twin -- scrape
    is a single fixed-schema application, not a multi-tenant one.

    :param pool: asyncpg-compatible pool
    :ptype pool: Any
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig()
    store = DataStore(agent_id=uuid.UUID(str(uuid_utils.uuid7())), registry=registry, config=config)

    runner = MigrationRunner()
    register(runner)
    applied = await runner.apply_for_platform_schema(store)
    log.info("migrations: %d applied via MigrationRunner (package=%s)", applied, PACKAGE_NAME)
