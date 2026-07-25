"""Tests for per-target fetch health: the fingerprint and where it gets stamped.

Two concerns, deliberately separated. :func:`content_fingerprint` answers "is this the
same page as last time", so its tests are about what should and should not change the
answer. The eval-loop tests answer "when is a fingerprint allowed to be written", which
matters more than it looks: a fingerprint stamped from a page we did not successfully
read is worse than no fingerprint at all, because the next failure would compare against
it and conclude the page had changed.

No database anywhere here. These collections fall back to an in-memory L3, which is fine
for behaviour but structurally cannot catch a missing DDL column -- that check lives in
``test_migrations_drift.py`` (offline) and ``tests/integration/test_health_l3_roundtrip.py``
(real Postgres).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scrape.collections import ScrapeExtraction, ScrapeExtractionCollection, ScrapeRecipeCollection
from threetears.scrape.eval_loop import _stamp_fingerprint_if_validated, run_eval_loop, run_eval_loop_multi_row
from threetears.scrape.health import (
    ScrapeTargetHealthCollection,
    content_fingerprint,
    record_validated_fetch,
)

_PAGE = """
<html><body>
  <table>
    <tr><th>Employer</th><th>County</th></tr>
    <tr><td>Acme Corp</td><td>Franklin</td></tr>
  </table>
</body></html>
"""


def extractions_row(*, validation_status: str) -> ScrapeExtraction:
    """A transient extraction carrying just the status the stamp helper branches on."""
    return ScrapeExtraction({"target_id": "warn_oh", "validation_status": validation_status})


@pytest.fixture()
def config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS")


@pytest.fixture()
def registry() -> CollectionRegistry:
    """No L3 pool: the in-memory fallback, which every unit test in this package uses."""
    return CollectionRegistry()


@pytest.fixture()
def health(registry: CollectionRegistry, config: DefaultCoreConfig) -> ScrapeTargetHealthCollection:
    return ScrapeTargetHealthCollection(registry, config, nats_client=None)


# ---------------------------------------------------------------------------
# content_fingerprint
# ---------------------------------------------------------------------------


def test_cosmetic_markup_changes_do_not_change_the_fingerprint() -> None:
    """Re-indentation and attribute churn are not content changes.

    A site that reformats its template, adds a wrapper class, or re-orders attributes
    has not changed what it says. A fingerprint that flipped on those would report "the
    site changed" on every deploy the site makes, which would then trigger a needless
    LLM regeneration round every time.
    """
    reformatted = (
        '<html><body><table class="table table-striped" data-build="7">'
        "<tr><th>Employer</th>   <th>County</th></tr>"
        "<tr><td>Acme Corp</td>\n\n<td>Franklin</td></tr>"
        "</table></body></html>"
    )
    assert content_fingerprint(_PAGE) == content_fingerprint(reformatted)


def test_a_real_content_change_changes_the_fingerprint() -> None:
    """A different employer in the table is exactly the change this must detect."""
    changed = _PAGE.replace("Acme Corp", "Beta Industries")
    assert content_fingerprint(_PAGE) != content_fingerprint(changed)


def test_fingerprint_is_stable_across_calls() -> None:
    """Same input, same digest. Guards against accidentally folding in time or randomness."""
    assert content_fingerprint(_PAGE) == content_fingerprint(_PAGE)


# ---------------------------------------------------------------------------
# record_validated_fetch
# ---------------------------------------------------------------------------


async def test_record_validated_fetch_writes_a_readable_row(health: ScrapeTargetHealthCollection) -> None:
    await record_validated_fetch(health, target_id="warn_oh", html=_PAGE)

    stored = await health.get("warn_oh")
    assert stored is not None
    assert stored.content_fingerprint == content_fingerprint(_PAGE)
    assert stored.fingerprint_updated_at is not None
    # Never-observed defaults, so a healthy target reads as healthy without anything
    # having written a row's worth of zeroes on its behalf.
    assert stored.consecutive_fetch_failures == 0
    assert stored.circuit_state == "closed"
    assert stored.blocked_until is None


async def test_a_later_fetch_updates_the_fingerprint(health: ScrapeTargetHealthCollection) -> None:
    await record_validated_fetch(health, target_id="warn_oh", html=_PAGE)
    changed = _PAGE.replace("Acme Corp", "Beta Industries")
    await record_validated_fetch(health, target_id="warn_oh", html=changed)

    stored = await health.get("warn_oh")
    assert stored is not None
    assert stored.content_fingerprint == content_fingerprint(changed)


async def test_recording_a_fingerprint_preserves_unrelated_health(health: ScrapeTargetHealthCollection) -> None:
    """A success must not silently wipe the failure and circuit columns.

    Those describe a different concern and are written by a different code path. If a
    fingerprint stamp replaced the row wholesale, a target recovering from a block would
    have its block history erased by the very fetch that proves it recovered, which is
    the moment that history is most worth keeping.
    """
    seeded = health.create(
        {
            "target_id": "warn_ok",
            "consecutive_fetch_failures": 4,
            "circuit_state": "half_open",
            "last_block_kind": "interstitial",
        }
    )
    await health.save_entity(seeded)

    await record_validated_fetch(health, target_id="warn_ok", html=_PAGE)

    stored = await health.get("warn_ok")
    assert stored is not None
    assert stored.content_fingerprint == content_fingerprint(_PAGE)
    assert stored.consecutive_fetch_failures == 4
    assert stored.circuit_state == "half_open"
    assert stored.last_block_kind == "interstitial"


# ---------------------------------------------------------------------------
# Where the eval loop stamps
# ---------------------------------------------------------------------------


@pytest.fixture()
def recipes(registry: CollectionRegistry, config: DefaultCoreConfig) -> ScrapeRecipeCollection:
    return ScrapeRecipeCollection(registry, config, nats_client=None)


@pytest.fixture()
def extractions(registry: CollectionRegistry, config: DefaultCoreConfig) -> ScrapeExtractionCollection:
    return ScrapeExtractionCollection(registry, config, nats_client=None)


async def _seed_working_recipe(recipes: ScrapeRecipeCollection, target_id: str, strategy: dict[str, Any]) -> None:
    entity = recipes.create(
        {
            "target_id": target_id,
            "extraction_strategy": strategy,
            "consecutive_validation_failures": 0,
        }
    )
    await recipes.save_entity(entity)


async def test_a_validated_reuse_stamps_the_fingerprint(
    recipes: ScrapeRecipeCollection,
    extractions: ScrapeExtractionCollection,
    health: ScrapeTargetHealthCollection,
) -> None:
    """The success path writes a fingerprint, with no LLM call anywhere in it."""
    await _seed_working_recipe(recipes, "warn_oh", {"row_selector": "table tr", "field_selectors": {"employer": "td"}})

    result = await run_eval_loop_multi_row(
        "warn_oh",
        _PAGE,
        "https://example.gov/warn",
        {"employer": str},
        recipe_collection=recipes,
        extraction_collection=extractions,
        api_key="unused-no-llm-call-on-the-reuse-path",
        health_collection=health,
    )

    assert result.validation_status == "validated"
    stored = await health.get("warn_oh")
    assert stored is not None
    assert stored.content_fingerprint == content_fingerprint(_PAGE)


async def test_a_failed_reuse_stamps_nothing(
    recipes: ScrapeRecipeCollection,
    extractions: ScrapeExtractionCollection,
    health: ScrapeTargetHealthCollection,
) -> None:
    """A page we could not read is not a reference for what the page looks like working.

    This is the assertion that matters most in the file. Stamping here would poison the
    comparison the fingerprint exists for: the next failure would diff against a
    fingerprint taken from a broken read and conclude the page had changed.
    """
    await _seed_working_recipe(
        recipes, "warn_oh", {"row_selector": "table tr.nonexistent", "field_selectors": {"employer": "td.nope"}}
    )

    result = await run_eval_loop_multi_row(
        "warn_oh",
        _PAGE,
        "https://example.gov/warn",
        {"employer": str},
        recipe_collection=recipes,
        extraction_collection=extractions,
        api_key="unused-no-llm-call-below-the-failure-threshold",
        health_collection=health,
    )

    assert result.validation_status == "failed"
    assert await health.get("warn_oh") is None


async def test_omitting_the_health_collection_changes_nothing(
    recipes: ScrapeRecipeCollection,
    extractions: ScrapeExtractionCollection,
) -> None:
    """Every existing caller passes no health collection and must be unaffected."""
    await _seed_working_recipe(recipes, "warn_oh", {"row_selector": "table tr", "field_selectors": {"employer": "td"}})

    result = await run_eval_loop_multi_row(
        "warn_oh",
        _PAGE,
        "https://example.gov/warn",
        {"employer": str},
        recipe_collection=recipes,
        extraction_collection=extractions,
        api_key="unused",
    )

    assert result.validation_status == "validated"


# ---------------------------------------------------------------------------
# The single-row entry point, and the statuses that must NOT stamp
# ---------------------------------------------------------------------------


async def test_a_validated_single_row_reuse_stamps_the_fingerprint(
    recipes: ScrapeRecipeCollection,
    extractions: ScrapeExtractionCollection,
    health: ScrapeTargetHealthCollection,
) -> None:
    """``run_eval_loop`` is a separate entry point and needs its own proof.

    Written because the multi-row version of this test caught the parameter being accepted
    and ignored on one of the two entry points. Asserting it on only one of them is how
    that class of mistake survives.
    """
    entity = recipes.create(
        {
            "target_id": "warn_single",
            "extraction_strategy": {"selectors": {"employer": "td"}},
            "consecutive_validation_failures": 0,
        }
    )
    await recipes.save_entity(entity)

    result = await run_eval_loop(
        "warn_single",
        _PAGE,
        "https://example.gov/warn",
        {"employer": str},
        recipe_collection=recipes,
        extraction_collection=extractions,
        api_key="unused-no-llm-call-on-the-reuse-path",
        health_collection=health,
    )

    assert result.validation_status == "validated"
    stored = await health.get("warn_single")
    assert stored is not None
    assert stored.content_fingerprint == content_fingerprint(_PAGE)


async def test_needs_review_stamps_nothing(health: ScrapeTargetHealthCollection) -> None:
    """``needs_review`` means nothing confirmed the extraction was right.

    That page is not a trustworthy "this is what the target looks like when it works"
    reference, so it must not become the comparison value. Asserted directly against the
    helper rather than through a full judge round: the rule is about the status, and
    driving an LLM judge to produce it would test the judge instead.
    """
    unconfirmed = extractions_row(validation_status="needs_review")
    await _stamp_fingerprint_if_validated(health, unconfirmed, target_id="warn_oh", html=_PAGE)
    assert await health.get("warn_oh") is None

    blocked = extractions_row(validation_status="blocked")
    await _stamp_fingerprint_if_validated(health, blocked, target_id="warn_oh", html=_PAGE)
    assert await health.get("warn_oh") is None


async def test_a_health_write_failure_never_fails_the_scrape(
    health: ScrapeTargetHealthCollection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Health is a diagnostic aid, and the extraction is already durable by this point.

    Losing real extracted data because a bookkeeping row could not be written would be a
    strictly worse outcome than having no fingerprint. The failure is logged with its
    traceback, never silenced.
    """
    validated = extractions_row(validation_status="validated")

    async def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("l3 is having a day")

    with patch("threetears.scrape.eval_loop.record_validated_fetch", side_effect=_boom):
        await _stamp_fingerprint_if_validated(health, validated, target_id="warn_oh", html=_PAGE)

    assert "fingerprint stamp failed" in caplog.text


async def test_the_vision_strategies_also_stamp(
    recipes: ScrapeRecipeCollection,
    extractions: ScrapeExtractionCollection,
    health: ScrapeTargetHealthCollection,
) -> None:
    """``per_document`` and ``multi_row_vision`` return early and must still stamp.

    The first version of this feature put the stamp only at the multi-row entry point's
    common exit, which both of these strategies return before reaching, so two whole
    classes of target silently never got a fingerprint while the helper's docstring
    claimed full coverage. Their inner extraction functions are patched out here because
    the question is purely whether the surrounding entry point stamps, not how a vision
    read behaves.
    """
    for strategy_type, inner in (
        ("per_document", "_run_per_document_extraction"),
        ("multi_row_vision", "_run_multi_row_vision_extraction"),
    ):
        target_id = f"warn_{strategy_type}"

        async def _validated(*_args: Any, **_kwargs: Any) -> ScrapeExtraction:
            return ScrapeExtraction({"target_id": target_id, "validation_status": "validated"})

        with patch(f"threetears.scrape.eval_loop.{inner}", side_effect=_validated):
            result = await run_eval_loop_multi_row(
                target_id,
                _PAGE,
                "https://example.gov/warn",
                {"employer": str},
                recipe_collection=recipes,
                extraction_collection=extractions,
                api_key="unused",
                strategy_type=strategy_type,  # type: ignore[arg-type]
                health_collection=health,
            )

        assert result.validation_status == "validated"
        stored = await health.get(target_id)
        assert stored is not None, f"{strategy_type} returned validated but stamped no fingerprint"
        assert stored.content_fingerprint == content_fingerprint(_PAGE)
