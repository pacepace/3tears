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

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scrape.challenge import PageVerdict
from threetears.scrape.collections import ScrapeExtraction, ScrapeExtractionCollection, ScrapeRecipeCollection
from threetears.scrape.eval_loop import _stamp_fingerprint_if_validated, run_eval_loop, run_eval_loop_multi_row
from threetears.scrape.health import (
    ScrapeTargetHealth,
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

    The classifier is faked out because a reuse failure now asks it what the page was, and
    an unfaked call here would reach for the network and then retry itself through half a
    minute of backoff. It answers ``content`` -- our selectors are simply wrong -- which is
    the route that leaves today's behaviour untouched, so the assertion below is about the
    fingerprint and nothing else. The row it DOES write is the verdict cache, which is why
    this checks ``content_fingerprint`` rather than the row's existence.
    """
    await _seed_working_recipe(
        recipes, "warn_oh", {"row_selector": "table tr.nonexistent", "field_selectors": {"employer": "td.nope"}}
    )
    verdict = PageVerdict(kind="content", evidence="the listing is present", confidence="high")
    fake_model = SimpleNamespace(
        with_structured_output=lambda _schema, **_kw: SimpleNamespace(ainvoke=AsyncMock(return_value=verdict))
    )

    with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
        result = await run_eval_loop_multi_row(
            "warn_oh",
            _PAGE,
            "https://example.gov/warn",
            {"employer": str},
            recipe_collection=recipes,
            extraction_collection=extractions,
            api_key="k",
            health_collection=health,
        )

    assert result.validation_status == "failed"
    stored = await health.get("warn_oh")
    assert stored is not None, "the verdict cache should have been written"
    assert stored.content_fingerprint is None, "a page we could not read must never become the reference page"
    assert stored.fingerprint_updated_at is None


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


async def test_a_row_round_tripped_through_l2_comes_back_with_real_datetimes(
    health: ScrapeTargetHealthCollection,
) -> None:
    """L2 serialization is lossy in one direction, and ``deserialize`` is where that is repaired.

    ``serialize`` writes JSON with ``default=str``, so every timestamp leaves as an ISO
    string. Until ``deserialize`` turned them back, a row read through L2 differed in TYPE
    from the identical row read through L1 or L3. Harmless while it is only read, because
    the entity accessors parse on the way out. Not harmless when it is written BACK: an
    update fences on the row's own ``date_updated`` as an optimistic lock against a
    ``TIMESTAMPTZ`` column, and a string bound there fails at the asyncpg border.

    Asserted across the serialize/rehydrate boundary rather than by mocking a hydrated row
    into the merge; a test that injected strings past this layer would be testing its own
    setup. The rehydration itself now lives on `BaseCollection` rather than in this
    collection's `deserialize`, so the composition below is what the L2 read path performs.
    """
    written = {
        "target_id": "warn_l2",
        "consecutive_fetch_failures": 1,
        "date_created": datetime(2026, 7, 25, 3, 0, tzinfo=UTC),
        "date_updated": datetime(2026, 7, 25, 3, 10, tzinfo=UTC),
        "last_blocked_at": datetime(2026, 7, 25, 3, 30, tzinfo=UTC),
        "fingerprint_updated_at": datetime(2026, 7, 25, 3, 30, tzinfo=UTC),
    }

    round_tripped = health._rehydrate_datetimes(health.deserialize(health.serialize(written)))

    for column in ("date_created", "date_updated", "last_blocked_at", "fingerprint_updated_at"):
        assert isinstance(round_tripped[column], datetime), (
            f"{column} came back from L2 as {type(round_tripped[column]).__name__}; "
            "written back, a string cannot satisfy a TIMESTAMPTZ optimistic lock"
        )
        assert round_tripped[column] == written[column]
    # Non-timestamp columns are untouched by the rehydration.
    assert round_tripped["consecutive_fetch_failures"] == 1
    assert round_tripped["target_id"] == "warn_l2"


def test_an_unparseable_timestamp_is_a_corrupt_cache_entry(
    health: ScrapeTargetHealthCollection,
) -> None:
    """A value that will not decode is refused, not carried onward.

    CONTRACT CHANGE, decided rather than drifted into. This collection used to preserve such a
    value verbatim, reasoning that nulling it would discard real data and, for ``date_updated``,
    would silently disable the next write's optimistic lock by making the fence ``None``. That
    reasoning was sound about nulling and wrong about the alternative it chose: preserving the
    string hands the caller a row with a ``str`` in a column typed ``datetime``, which is the
    precise fault the round-trip test above says this rehydration exists to prevent.

    The third option is the one taken. L2 is a cache, so a value that will not decode is a
    corrupt cache entry: `BaseCollection` raises here, the L2 read path treats it as a miss and
    falls through to L3, and the CAS path replaces the entry at the revision that held it.
    Nothing is discarded, because L3 is authoritative and still holds the row.
    """
    from threetears.core.exceptions import CorruptCacheEntry

    payload = b'{"target_id": "warn_bad", "date_updated": "not-a-timestamp"}'

    with pytest.raises(CorruptCacheEntry) as caught:
        health._rehydrate_datetimes(health.deserialize(payload))

    # Names the column, so the log line the read path emits can point at the bad data.
    assert caught.value.column == "date_updated"
    assert caught.value.value == "not-a-timestamp"


async def test_the_fingerprint_merge_carries_the_lock_forward(health: ScrapeTargetHealthCollection) -> None:
    """The read-modify-write path fences on the row it read.

    Two pods can both read a health row and both write it back, so the update relies on a
    compare-and-swap rather than on ordering. This pins that the fence reaching the write
    is a real ``datetime`` taken from the row that was read.

    The first write after creation is deliberately exercised too, because it behaves
    differently and that difference is easy to mistake for this bug: ``save_entity`` only
    stamps ``date_updated`` once an entity is no longer new, so a freshly created row has
    none and its first update is genuinely unfenced -- there is no prior value to compare
    against. Every update after that is fenced, which is the state this asserts.

    Asserting the positive type matters and not merely "it is not a string": ``None``
    passes a not-a-string check and makes ``save_entity`` skip the compare-and-swap
    entirely, silently turning the optimistic lock into last-write-wins. An earlier
    version of this test checked only the negative and passed in exactly that state.
    """
    seeded = health.create({"target_id": "warn_cas", "consecutive_fetch_failures": 1})
    await health.save_entity(seeded)

    fences: list[Any] = []
    original_save = health.save_to_store

    async def _capture(data: dict[str, Any], original_timestamp: Any = None, **kwargs: Any) -> int:
        fences.append(original_timestamp)
        return await original_save(data, original_timestamp, **kwargs)

    with patch.object(health, "save_to_store", side_effect=_capture):
        await record_validated_fetch(health, target_id="warn_cas", html=_PAGE)
        # Captured BETWEEN the two merges: read afterwards, this is the timestamp the
        # second merge WROTE, not the one it fenced on, and the comparison passes for the
        # wrong reason.
        between = await health.get("warn_cas")
        assert between is not None
        expected_fence = getattr(between, "original_date_updated", None)
        entity = await record_validated_fetch(health, target_id="warn_cas", html=_PAGE)

    assert len(fences) == 2, "save_to_store was not reached twice"
    assert fences[0] is None, (
        "a row that has never been updated has no date_updated to fence on; if this ever "
        "becomes non-None the framework's stamping rules changed and the comment above is stale"
    )
    fence = fences[1]
    assert not isinstance(fence, str), (
        f"the optimistic-lock fence was bound as a string ({fence!r}); "
        "asyncpg cannot compare that against a TIMESTAMPTZ column"
    )
    assert isinstance(fence, datetime), f"the fence was {fence!r}, so the update would not be fenced at all"
    # The VALUE matters, not just the type: any fresh datetime satisfies the checks above
    # while fencing on the wrong row entirely, and the in-memory store ignores
    # original_timestamp, so nothing else on this path would notice. It must be the
    # date_updated of the row the second merge actually read.
    assert fence == expected_fence, (
        "the fence is not the timestamp of the row that was read, so the compare-and-swap "
        "is guarding against a different version than the one this write was based on"
    )
    assert entity.content_fingerprint == content_fingerprint(_PAGE)
