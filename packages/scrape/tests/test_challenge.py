"""Tests for classifying a failed page, and for what each verdict does to a target.

The bug being fixed is live behaviour, not a hypothetical: a bot wall returns HTML, the
stored selectors miss it, the failure counter climbs exactly as if the site had been
redesigned, and three polls later a working recipe is thrown away and several LLM calls are
spent learning to extract data from a challenge page. So the assertions that matter most
here are about what does NOT happen -- the recipe not moving, the classifier not being
called, records not being written.

Every model call is faked at ``llm_retry.create_chat_model``, dispatched by the response
model each call asks for, which is what lets one test hold a classifier answer and a judge
answer at once without guessing at prompt text. Counting entries in ``requested`` is how the
cost claims in the design are actually checked rather than asserted in prose.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scrape.challenge import PageVerdict, build_classification_prompt, classify_failed_page
from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
from threetears.scrape.eval_loop import _JudgeVerdict, run_eval_loop, run_eval_loop_multi_row
from threetears.scrape.extraction import (
    _CandidateStrategy,
    _CandidateStrategyList,
    _RegexCandidateStrategy,
    _RegexCandidateStrategyList,
    _RowCandidateStrategy,
    _RowCandidateStrategyList,
)
from threetears.scrape.health import ScrapeTargetHealthCollection, content_fingerprint

# A wall: real HTML, HTTP 200, and nothing any stored selector will ever match.
_WALL = """
<html><body>
  <h1>Checking your browser before you continue</h1>
  <p>This process is automatic. Your browser will redirect shortly.</p>
</body></html>
"""

# The content page, and the same CONTENT under different markup. Identical readable text
# means an identical fingerprint, so the pair is how "the page did not change, our selectors
# did" gets exercised -- the case that must never cost a model call.
_TABLE_PAGE = '<html><body><table><tr><td class="employer">Acme Corp</td></tr></table></body></html>'
_RESTYLED_PAGE = '<html><body><table><tr><td class="org">Acme Corp</td></tr></table></body></html>'

_SCHEMA = {"employer": str}
_ROW_STRATEGY = {"row_selector": "tr", "field_selectors": {"employer": "td.employer"}}
_SINGLE_STRATEGY = {"selectors": {"employer": "td.employer"}}

_DEAD_REGEX_PATTERN = r"(?P<employer>Nothing Matches This)"

_BLOCKED = PageVerdict(kind="blocked", evidence="the page asks the visitor to verify a browser", confidence="high")
_CHANGED = PageVerdict(
    kind="changed", evidence="the employer column moved to a differently named cell", confidence="high"
)
_CONTENT = PageVerdict(kind="content", evidence="the listing is present and looks normal", confidence="medium")


@pytest.fixture(autouse=True)
def no_retry_sleeps() -> Iterator[None]:
    """Retry backoff is real time, and a misdirected fake would otherwise burn 30 seconds.

    Autouse rather than per-test: several tests here deliberately drive the classifier to
    total failure, and a test that accidentally requests a response model this file has no
    answer for should fail on its ``requested`` assertion in milliseconds rather than look
    like a hang.
    """
    with patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()):
        yield


def fake_models(responses: dict[type, Any], requested: list[type] | None = None) -> Any:
    """A ``create_chat_model`` replacement that answers by the response model asked for.

    Dispatching on the requested model rather than on ``purpose`` is what this file needs
    and the older tests did not: the page classifier and the candidate judge both run as
    ``LlmPurpose.UTILITY``, so a purpose-keyed fake cannot tell them apart, and a test that
    needs a classifier verdict AND a judge verdict in one run would get whichever was
    listed first for both.

    :param responses: response model -> the value its call returns, or an exception to raise
    :ptype responses: dict[type, Any]
    :param requested: accumulates every response model requested, in order; the record a
        test asserts against to prove a call was or was not made
    :ptype requested: list[type] | None
    :return: a side_effect suitable for patching ``llm_retry.create_chat_model``
    :rtype: Any
    """

    def _create(*_args: Any, **_kwargs: Any) -> Any:
        def _with_structured_output(schema: type, **_kw: Any) -> Any:
            if requested is not None:
                requested.append(schema)
            answer = responses.get(schema)
            if isinstance(answer, Exception):
                return SimpleNamespace(ainvoke=AsyncMock(side_effect=answer))
            return SimpleNamespace(ainvoke=AsyncMock(return_value=answer))

        return SimpleNamespace(with_structured_output=_with_structured_output)

    return _create


@pytest.fixture()
def collections() -> tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection]:
    """Fresh in-memory collections per test, so no target's state leaks into the next."""
    registry = CollectionRegistry()
    config = DefaultCoreConfig(collection_flush="ALWAYS")
    return (
        ScrapeRecipeCollection(registry, config, nats_client=None),
        ScrapeExtractionCollection(registry, config, nats_client=None),
        ScrapeTargetHealthCollection(registry, config, nats_client=None),
    )


async def seed_recipe(
    recipes: ScrapeRecipeCollection,
    target_id: str,
    strategy: dict[str, Any],
    *,
    failures: int = 0,
) -> None:
    entity = recipes.create(
        {"target_id": target_id, "extraction_strategy": strategy, "consecutive_validation_failures": failures}
    )
    await recipes.save_entity(entity)


# ---------------------------------------------------------------------------
# challenge.py on its own
# ---------------------------------------------------------------------------


def test_the_prompt_carries_the_page_and_the_fields_we_wanted() -> None:
    """Both halves are load-bearing.

    "Is this the content we wanted" cannot be answered without saying what was wanted, and
    a classifier given only the schema would be guessing about a page it never saw.
    """
    prompt = build_classification_prompt(_WALL, {"employer": str, "affected_count": int})

    assert "employer" in prompt
    assert "affected_count" in prompt
    assert "Checking your browser" in prompt


def test_the_status_line_appears_only_when_the_caller_knows_it() -> None:
    """An unknown status must not be presented as evidence.

    Most walls return 200, so the status is a weak signal at best; inventing one would make
    it a misleading signal, which is worse than an absent one.
    """
    assert "HTTP status 403" in build_classification_prompt(_WALL, _SCHEMA, page_status=403)
    assert "HTTP status" not in build_classification_prompt(_WALL, _SCHEMA)


async def test_the_classifier_returns_the_models_verdict() -> None:
    with patch("threetears.scrape.llm_retry.create_chat_model", side_effect=fake_models({PageVerdict: _BLOCKED})):
        verdict = await classify_failed_page(_WALL, _SCHEMA, api_key="k")

    assert verdict is not None
    assert verdict.kind == "blocked"
    assert verdict.evidence == _BLOCKED.evidence


async def test_a_classifier_that_never_answers_degrades_to_none() -> None:
    """``None`` is the honest "we could not tell", and callers must read it as such."""
    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({PageVerdict: RuntimeError("upstream is down")}),
    ):
        assert await classify_failed_page(_WALL, _SCHEMA, api_key="k") is None


# ---------------------------------------------------------------------------
# The bug this chunk exists to fix
# ---------------------------------------------------------------------------


async def test_a_wall_leaves_the_recipe_byte_identical(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """The regression test for the whole feature.

    Today this exact sequence increments ``consecutive_validation_failures``, and three
    polls of it discard the recipe and spend a candidate-generation-plus-judge round
    learning to extract data from a challenge page. Every field of the recipe row is
    asserted, not just the counter: "untouched" has to mean untouched, and a `last_validated_at`
    quietly moved forward would misreport when the recipe last actually worked.
    """
    recipes, extractions, health = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)
    before = await recipes.get("warn_oh")
    assert before is not None
    strategy_before, failures_before = before.extraction_strategy, before.consecutive_validation_failures
    validated_before, won_before = before.last_validated_at, before.won_at

    with patch("threetears.scrape.llm_retry.create_chat_model", side_effect=fake_models({PageVerdict: _BLOCKED})):
        result = await run_eval_loop_multi_row(
            "warn_oh",
            _WALL,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert result.validation_status == "blocked"
    assert result.structured_fields == {"records": []}, "a wall has no data on it; nothing may be written as if it did"
    assert result.extraction_recipe_id is None
    assert result.field_confidences == {"page_verdict": "blocked", "page_verdict_evidence": _BLOCKED.evidence}

    after = await recipes.get("warn_oh")
    assert after is not None
    assert after.consecutive_validation_failures == failures_before
    assert after.extraction_strategy == strategy_before
    assert after.last_validated_at == validated_before
    assert after.won_at == won_before


async def test_a_wall_never_invents_a_recipe_for_a_target_that_had_none(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """A target blocked before it ever succeeded stays a health row with no recipe row.

    The alternative -- writing a strategy-less recipe to hold the blocked state -- is what
    the separate health entity exists to avoid, because the reuse branch would then happily
    try to reuse an empty strategy.
    """
    recipes, extractions, health = collections
    candidates = _RowCandidateStrategyList(
        candidates=[_RowCandidateStrategy(row_selector="tr.nope", field_selectors={})]
    )

    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({_RowCandidateStrategyList: candidates, PageVerdict: _BLOCKED}),
    ):
        result = await run_eval_loop_multi_row(
            "warn_new",
            _WALL,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert result.validation_status == "blocked"
    assert await recipes.get("warn_new") is None
    stored = await health.get("warn_new")
    assert stored is not None
    assert stored.classified_verdict == "blocked"
    assert stored.last_blocked_at is not None


async def test_a_changed_page_regenerates_on_the_first_failure_not_the_third(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """Waiting two more polls is pure latency once we have positive evidence.

    The counter starts at 0 and the threshold is 3, so today this poll would only have
    incremented it. Asserting the NEW strategy landed is what distinguishes "regenerated"
    from "happened to fail differently".
    """
    recipes, extractions, health = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)
    candidates = _RowCandidateStrategyList(
        candidates=[_RowCandidateStrategy(row_selector="tr", field_selectors={"employer": "td.org"})]
    )
    judged = _JudgeVerdict(winning_candidate_index=0, reasoning="the new cell holds the employer")

    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({PageVerdict: _CHANGED, _RowCandidateStrategyList: candidates, _JudgeVerdict: judged}),
    ):
        result = await run_eval_loop_multi_row(
            "warn_oh",
            _RESTYLED_PAGE,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert result.validation_status == "validated"
    assert result.structured_fields == {"records": [{"employer": "Acme Corp"}]}
    recipe = await recipes.get("warn_oh")
    assert recipe is not None
    assert recipe.extraction_strategy == {"row_selector": "tr", "field_selectors": {"employer": "td.org"}}
    assert recipe.consecutive_validation_failures == 0


async def test_an_ordinary_failure_still_just_counts(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """ "Our selectors are wrong" is the case the existing threshold already handles well."""
    recipes, extractions, health = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)

    with patch("threetears.scrape.llm_retry.create_chat_model", side_effect=fake_models({PageVerdict: _CONTENT})):
        result = await run_eval_loop_multi_row(
            "warn_oh",
            _RESTYLED_PAGE,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert result.validation_status == "failed"
    recipe = await recipes.get("warn_oh")
    assert recipe is not None
    assert recipe.consecutive_validation_failures == 1
    assert recipe.extraction_strategy == _ROW_STRATEGY


async def test_a_classifier_that_cannot_answer_behaves_exactly_as_today(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """An unanswerable question must never be more destructive than not having asked one.

    This is the safety property behind the whole design: the worst case of adding
    classification is the behaviour that existed before it.
    """
    recipes, extractions, health = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)

    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({PageVerdict: RuntimeError("upstream is down")}),
    ):
        result = await run_eval_loop_multi_row(
            "warn_oh",
            _WALL,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert result.validation_status == "failed"
    recipe = await recipes.get("warn_oh")
    assert recipe is not None
    assert recipe.consecutive_validation_failures == 1
    assert await health.get("warn_oh") is None, "a verdict that was never reached must not be cached"


# ---------------------------------------------------------------------------
# What it costs -- the claims the design makes, checked rather than asserted in prose
# ---------------------------------------------------------------------------


async def test_an_unchanged_page_never_reaches_the_classifier(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """The free check settles it, and it is the common case.

    Identical readable text to the page the recipe last validated against proves two things
    at once: the site did not change, and this is not a new wall, because a wall would not
    digest to the same content. Without this branch every transient miss would cost a model
    call that today costs nothing, which is a real regression dressed up as a feature.
    """
    recipes, extractions, health = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)
    seeded = health.create({"target_id": "warn_oh", "content_fingerprint": content_fingerprint(_TABLE_PAGE)})
    await health.save_entity(seeded)
    requested: list[type] = []

    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({PageVerdict: _BLOCKED}, requested),
    ):
        result = await run_eval_loop_multi_row(
            "warn_oh",
            _RESTYLED_PAGE,  # same readable text as _TABLE_PAGE, selectors no longer match
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert requested == [], "an unchanged page must not cost a model call"
    assert result.validation_status == "failed"
    recipe = await recipes.get("warn_oh")
    assert recipe is not None
    assert recipe.consecutive_validation_failures == 1


async def test_the_same_wall_next_poll_costs_nothing(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """A target walled for a week costs one classification, not seven.

    Runs the loop twice against the identical page and asserts the classifier was consulted
    once. The second poll must still route to ``blocked``, so this is not testing that the
    cache silently stops working.
    """
    recipes, extractions, health = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)
    requested: list[type] = []

    async def _poll() -> Any:
        return await run_eval_loop_multi_row(
            "warn_oh",
            _WALL,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({PageVerdict: _BLOCKED}, requested),
    ):
        first = await _poll()
        second = await _poll()

    assert requested == [PageVerdict], "the second poll re-asked about a page it had already judged"
    assert first.validation_status == "blocked"
    assert second.validation_status == "blocked"
    assert second.field_confidences == {"page_verdict": "blocked", "page_verdict_evidence": _BLOCKED.evidence}, (
        "the cached verdict must carry its original evidence forward, not a placeholder"
    )


async def test_a_changed_verdict_stops_regenerating_once_it_has_been_acted_on(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """The loop this design would otherwise create, and the reason the cache stores every verdict.

    A page classified ``changed`` regenerates immediately. If regeneration cannot learn it,
    nothing about the page changes, so a cache that only remembered ``blocked`` would
    re-classify, get ``changed`` again, and regenerate on EVERY subsequent poll -- strictly
    worse than the three-poll cadence it replaced. The second poll must fall back to
    counting the failure.
    """
    recipes, extractions, health = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)
    # Regeneration proposes something that matches nothing, so it cannot learn this page.
    hopeless = _RowCandidateStrategyList(
        candidates=[_RowCandidateStrategy(row_selector="tr.nope", field_selectors={"employer": "td.nope"})]
    )
    requested: list[type] = []

    async def _poll() -> Any:
        return await run_eval_loop_multi_row(
            "warn_oh",
            _RESTYLED_PAGE,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({PageVerdict: _CHANGED, _RowCandidateStrategyList: hopeless}, requested),
    ):
        await _poll()
        await _poll()

    assert requested.count(PageVerdict) == 1, "the second poll re-asked about a page it had already judged"
    assert requested.count(_RowCandidateStrategyList) == 1, (
        "the second poll regenerated again against a page the first poll already failed to learn"
    )
    recipe = await recipes.get("warn_oh")
    assert recipe is not None
    assert recipe.consecutive_validation_failures == 1, "the second poll must fall back to counting the failure"


async def test_omitting_the_health_collection_spends_nothing_and_changes_nothing(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """Every pre-existing caller passes no health collection and must be untouched by all of this."""
    recipes, extractions, _ = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)
    requested: list[type] = []

    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({PageVerdict: _BLOCKED}, requested),
    ):
        result = await run_eval_loop_multi_row(
            "warn_oh",
            _WALL,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            api_key="k",
        )

    assert requested == []
    assert result.validation_status == "failed"
    recipe = await recipes.get("warn_oh")
    assert recipe is not None
    assert recipe.consecutive_validation_failures == 1


# ---------------------------------------------------------------------------
# Every strategy shape, not a representative one
# ---------------------------------------------------------------------------


async def test_every_reuse_shape_routes_a_wall_the_same_way(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """All four reuse paths, because the chunk spec named two of them.

    The regex variants are not a footnote: a regex target behind a wall loses its recipe
    exactly as a CSS one does. Covering only the shapes that were named is how the same
    drift keeps happening -- ``link_selector`` shipped with no DDL column for want of a test
    that checked the whole family rather than a sample of it.
    """
    recipes, extractions, health = collections
    shapes = [
        ("css_single", run_eval_loop, "css", _SINGLE_STRATEGY),
        ("regex_single", run_eval_loop, "regex", {"pattern": _DEAD_REGEX_PATTERN}),
        ("css_rows", run_eval_loop_multi_row, "css", _ROW_STRATEGY),
        ("regex_rows", run_eval_loop_multi_row, "regex", {"pattern": _DEAD_REGEX_PATTERN}),
    ]

    for name, entry_point, strategy_type, strategy in shapes:
        await seed_recipe(recipes, name, strategy)

        with patch("threetears.scrape.llm_retry.create_chat_model", side_effect=fake_models({PageVerdict: _BLOCKED})):
            result = await entry_point(
                name,
                _WALL,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipes,
                extraction_collection=extractions,
                health_collection=health,
                api_key="k",
                strategy_type=strategy_type,  # type: ignore[arg-type]
            )

        assert result.validation_status == "blocked", f"{name} did not route a wall to blocked"
        recipe = await recipes.get(name)
        assert recipe is not None
        assert recipe.consecutive_validation_failures == 0, f"{name} counted a wall against its recipe"
        assert recipe.extraction_strategy == strategy, f"{name} altered its recipe on a wall"


async def test_every_regeneration_shape_routes_a_wall_the_same_way(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """All four no-survivor paths, which is the only route a never-succeeded target has.

    A target walled from its very first fetch never reaches a reuse branch, so without this
    hook nothing would ever mark it as needing a human -- it would simply fail forever.
    """
    recipes, extractions, health = collections
    dead_css = _CandidateStrategyList(candidates=[_CandidateStrategy(selectors={"employer": ".nope"})])
    dead_regex = _RegexCandidateStrategyList(candidates=[_RegexCandidateStrategy(pattern=_DEAD_REGEX_PATTERN)])
    dead_rows = _RowCandidateStrategyList(
        candidates=[_RowCandidateStrategy(row_selector="tr.nope", field_selectors={"employer": "td.nope"})]
    )
    shapes = [
        ("css_single", run_eval_loop, "css", _CandidateStrategyList, dead_css),
        ("regex_single", run_eval_loop, "regex", _RegexCandidateStrategyList, dead_regex),
        ("css_rows", run_eval_loop_multi_row, "css", _RowCandidateStrategyList, dead_rows),
        ("regex_rows", run_eval_loop_multi_row, "regex", _RegexCandidateStrategyList, dead_regex),
    ]

    for name, entry_point, strategy_type, candidate_model, candidates in shapes:
        target_id = f"regen_{name}"

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=fake_models({candidate_model: candidates, PageVerdict: _BLOCKED}),
        ):
            result = await entry_point(
                target_id,
                _WALL,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipes,
                extraction_collection=extractions,
                health_collection=health,
                api_key="k",
                strategy_type=strategy_type,  # type: ignore[arg-type]
            )

        assert result.validation_status == "blocked", f"{name} regeneration did not route a wall to blocked"
        assert await recipes.get(target_id) is None, f"{name} invented a recipe for a walled target"


# ---------------------------------------------------------------------------
# Health is a diagnostic aid, and must never cost a real result
# ---------------------------------------------------------------------------


async def test_an_unreadable_health_store_degrades_rather_than_failing_the_poll(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A health outage must return the target to its pre-health behaviour, not break it."""
    recipes, extractions, health = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)

    async def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("l3 is having a day")

    with (
        patch.object(health, "get", side_effect=_boom),
        patch("threetears.scrape.llm_retry.create_chat_model", side_effect=fake_models({PageVerdict: _BLOCKED})),
    ):
        result = await run_eval_loop_multi_row(
            "warn_oh",
            _WALL,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert result.validation_status == "failed"
    assert "could not read health" in caplog.text


async def test_a_verdict_that_cannot_be_cached_is_still_acted_on(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Losing the cache costs one repeated call next poll. Losing the verdict costs the recipe."""
    recipes, extractions, health = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)

    async def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("l3 is having a day")

    with (
        patch("threetears.scrape.eval_loop.record_classification", side_effect=_boom),
        patch("threetears.scrape.llm_retry.create_chat_model", side_effect=fake_models({PageVerdict: _BLOCKED})),
    ):
        result = await run_eval_loop_multi_row(
            "warn_oh",
            _WALL,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert result.validation_status == "blocked"
    assert "could not cache the page verdict" in caplog.text
    recipe = await recipes.get("warn_oh")
    assert recipe is not None
    assert recipe.consecutive_validation_failures == 0


async def test_a_stored_verdict_nobody_recognises_is_re_asked_not_acted_on(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """A cached value outside the known set is treated as no cache at all.

    It is either a row written by a version that meant something different by it, or
    corruption. Re-asking costs one call; acting on a value we cannot interpret costs
    whatever that value happens to collide with.
    """
    recipes, extractions, health = collections
    await seed_recipe(recipes, "warn_oh", _ROW_STRATEGY)
    seeded = health.create(
        {
            "target_id": "warn_oh",
            "classified_fingerprint": content_fingerprint(_WALL),
            "classified_verdict": "quarantined",
            "classified_evidence": "from some future version",
        }
    )
    await health.save_entity(seeded)
    requested: list[type] = []

    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({PageVerdict: _BLOCKED}, requested),
    ):
        result = await run_eval_loop_multi_row(
            "warn_oh",
            _WALL,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert requested == [PageVerdict]
    assert result.validation_status == "blocked"
