"""3tears-scrape's own schema migrations, registered with 3tears' canonical ``MigrationRunner``.

Lives inside this package rather than in a consuming application's own
migrations module, so the DDL travels with the code that depends on it --
which is what made the original lift out of the application repo a directory
move rather than a disentangling exercise.

Registered under its own ``PACKAGE_NAME`` ("3tears_scrape") so its
``_schema_migrations`` history is distinct from every other package's, even
though several packages can apply against the same PLATFORM schema.
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
    (``collections.py``) exactly. ``date_created``/``date_updated`` included on
    every table from the start -- ``BaseCollection.save_entity()``
    unconditionally stamps both on every upsert regardless of what a collection's
    entity class exposes, so omitting them would raise
    ``asyncpg.UndefinedColumnError`` on the first real write, a failure mode
    already paid for once in the application this package was lifted out of.
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


async def v009_target_link_selector(store: DataStore) -> None:
    """``ScrapeTarget.link_selector`` -- CSS selector matching the document
    links on a listing page, required by ``MultiDocumentDriver``'s HTML
    discovery mode (``driver_backend: "multi_document"`` without
    :attr:`~threetears.scrape.collections.ScrapeTarget.api_results_path`).
    Nullable; ``None`` is fine for every non-``"multi_document"`` target,
    which ignores it, so this is a no-op for every pre-existing row.

    Shipped a release late: the entity property landed with the
    multi-document capability (2026-07-15) but no migration followed it, so
    a ``multi_document`` target seeded from YAML by ``bootstrap_targets()``
    raised ``asyncpg.UndefinedColumnError`` on its first real L3 upsert --
    the exact failure mode v001's docstring already describes, invisible to
    every unit test because ``ScrapeCollection``'s in-memory L3 fallback
    ignores schema entirely. ``tests/test_migrations_drift.py`` now derives
    its field set by introspecting the entity classes rather than restating
    them by hand, which is what lets it catch the next one.
    """
    await store.execute("ALTER TABLE scrape_targets ADD COLUMN IF NOT EXISTS link_selector TEXT")


async def v010_create_scrape_target_health(store: DataStore) -> None:
    """Create ``scrape_target_health`` -- per-target fetch health, one row per target.

    Column shape matches ``ScrapeTargetHealth`` (``health.py``) exactly. A separate table
    rather than more columns on ``scrape_recipes`` because health exists for targets that
    have never had a recipe: one blocked before it ever extracted successfully has real
    health and no strategy, and giving it a strategy-less recipe row would mean adding a
    guard so the reuse path never mistakes that empty strategy for a real one.

    Every column is nullable or defaulted, so a row can be created knowing only the
    ``target_id``. ``date_created``/``date_updated`` are present because
    ``BaseCollection.save_entity()`` stamps both on every upsert regardless of what the
    entity class exposes; omitting them raises ``asyncpg.UndefinedColumnError`` on the
    first real write.

    The block/circuit/session columns are created here rather than added later: the shape
    is already settled, and one CREATE beats several ALTERs against a table this young. The
    code that writes them lands with the backoff and human-in-the-loop work.

    The three ``classified_*`` columns joined this CREATE after the table was first written,
    on that same reasoning, while the branch introducing it was still unmerged and unpushed.
    That was checked rather than assumed: no ``scrape_*`` table and no ``3tears_scrape`` row
    existed in any local database, because the only thing that has ever run this migration
    is the integration suite's throwaway container. A database that HAD applied an earlier
    form of v010 would not pick the new columns up, since the version is already recorded as
    applied, and would need dropping. Once this ships, the same change has to be an ALTER in
    a later version instead.
    """
    await store.execute("""
        CREATE TABLE IF NOT EXISTS scrape_target_health (
            target_id                   TEXT        NOT NULL,
            content_fingerprint         TEXT,
            fingerprint_updated_at      TIMESTAMPTZ,
            consecutive_fetch_failures  INTEGER     NOT NULL DEFAULT 0,
            circuit_state               TEXT        NOT NULL DEFAULT 'closed',
            blocked_until               TIMESTAMPTZ,
            last_blocked_at             TIMESTAMPTZ,
            last_block_kind             TEXT,
            classified_fingerprint      TEXT,
            classified_verdict          TEXT,
            classified_evidence         TEXT,
            session_state_sealed        TEXT,
            session_state_expires_at    TIMESTAMPTZ,
            date_created                TIMESTAMPTZ,
            date_updated                TIMESTAMPTZ,
            PRIMARY KEY (target_id)
        )
    """)
    # Will answer "which targets are currently walled off", the one query an operator runs
    # against this table that is not a primary-key lookup. It returns the empty set today
    # and will keep doing so until something writes ``circuit_state`` -- said here plainly
    # because an index whose comment describes a working operator query is exactly the kind
    # of thing that gets trusted and then quietly reports "no targets are blocked" while
    # several are.
    await store.execute(
        "CREATE INDEX IF NOT EXISTS scrape_target_health_circuit_state "
        "ON scrape_target_health (circuit_state) WHERE circuit_state <> 'closed'"
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
    pkg.version(9)(v009_target_link_selector)
    pkg.version(10)(v010_create_scrape_target_health)
    runner.register(pkg)
    return pkg


async def apply_migrations(pool: Any) -> None:
    """Apply every pending 3tears-scrape migration against ``pool`` via MigrationRunner.

    A throwaway registry/config bound to ``pool`` and a ``DataStore`` wrapping
    it. ``DataStore`` requires an ``agent_id`` (3tears' per-agent-schema
    concept), inert here: scrape's tables are a single fixed PLATFORM-scope
    schema shared by every caller, not per-agent state, so the value only has
    to exist, not to mean anything.

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
