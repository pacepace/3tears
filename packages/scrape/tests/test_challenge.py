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
    RowValidationResult,
    ValidationResult,
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


async def test_the_page_status_reaches_the_prompt_from_the_entry_point(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """A parameter accepted and quietly dropped is worse than one that was never added.

    ``page_status`` crosses eight hops between the entry point and the prompt, and asserting
    only that :func:`build_classification_prompt` renders it proves nothing about whether
    anything reaches it. This captures the prompt the model was actually handed. Chunk 01
    shipped a parameter wired into one of two entry points on exactly this basis, so the
    single-row entry point is checked too rather than assumed to match.
    """
    recipes, extractions, health = collections
    for name, entry_point, strategy in (
        ("rows", run_eval_loop_multi_row, _ROW_STRATEGY),
        ("single", run_eval_loop, _SINGLE_STRATEGY),
    ):
        await seed_recipe(recipes, name, strategy)
        seen: list[Any] = []

        def _capture(*_args: Any, **_kwargs: Any) -> Any:
            def _with_structured_output(_schema: type, **_kw: Any) -> Any:
                async def _ainvoke(prompt: Any) -> PageVerdict:
                    seen.append(prompt)
                    return _BLOCKED

                return SimpleNamespace(ainvoke=_ainvoke)

            return SimpleNamespace(with_structured_output=_with_structured_output)

        with patch("threetears.scrape.llm_retry.create_chat_model", side_effect=_capture):
            await entry_point(
                name,
                _WALL,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipes,
                extraction_collection=extractions,
                health_collection=health,
                api_key="k",
                page_status=503,
            )

        assert seen, f"{name} never reached the classifier"
        assert "HTTP status 503" in seen[0], (
            f"{name} dropped page_status somewhere between the entry point and the prompt"
        )


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


# ---------------------------------------------------------------------------
# The one behaviour the four-into-one strategy collapse could have changed quietly
# ---------------------------------------------------------------------------


async def test_an_unconfirmed_single_record_surfaces_the_first_survivor(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """A single-record shape must still surface survivors[0] when the judge confirms nothing.

    The four regeneration bodies were collapsed into one, and the two shapes disagreed about
    "best survivor": the row shapes picked the candidate capturing the most records, the
    single-record shapes took the first proposed. The shared body uses max-by-record-count
    for both, which is only equivalent because every single-record survivor holds exactly one
    record and ``max`` returns the FIRST maximal element.

    That equivalence is an argument, and arguments rot. This pins it with two survivors whose
    order is the only thing distinguishing them, so a later switch to (say) ``sorted(...)[-1]``
    or a max that returned the last maximal element fails here rather than silently changing
    which extraction a human reviews.
    """
    recipes, extractions, health = collections
    # The two candidates must extract DIFFERENT values or this test cannot tell them apart.
    # The first version of it used a page with a single cell, so both candidates returned
    # "Acme Corp" and it passed against a deliberately broken tie-break.
    two_cells = (
        "<html><body><table><tr>"
        '<td class="employer">Acme Corp</td><td class="alt">Beta LLC</td>'
        "</tr></table></body></html>"
    )
    both_valid = _CandidateStrategyList(
        candidates=[
            _CandidateStrategy(selectors={"employer": "td.employer"}),
            _CandidateStrategy(selectors={"employer": "td.alt"}),
        ]
    )
    no_winner = _JudgeVerdict(winning_candidate_index=None, reasoning="cannot confirm either")

    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({_CandidateStrategyList: both_valid, _JudgeVerdict: no_winner}),
    ):
        result = await run_eval_loop(
            "warn_tie",
            two_cells,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert result.validation_status == "needs_review"
    assert await recipes.get("warn_tie") is None, "an unconfirmed candidate must never be crowned"
    assert result.structured_fields == {"records": [{"employer": "Acme Corp"}]}


async def test_an_unconfirmed_row_set_still_surfaces_the_richest_survivor(
    collections: tuple[ScrapeRecipeCollection, ScrapeExtractionCollection, ScrapeTargetHealthCollection],
) -> None:
    """And the row shape must still prefer record count over proposal order.

    The other half of the same collapse: here "best" genuinely means most rows captured, and
    the richer candidate is deliberately proposed SECOND so that falling back to "first
    proposed" fails this test.
    """
    recipes, extractions, health = collections
    page = (
        "<html><body><table>"
        '<tr class="r"><td class="employer">Acme Corp</td></tr>'
        '<tr class="r"><td class="employer">Beta LLC</td></tr>'
        "</table></body></html>"
    )
    thin_then_rich = _RowCandidateStrategyList(
        candidates=[
            _RowCandidateStrategy(row_selector="tr:first-child", field_selectors={"employer": "td.employer"}),
            _RowCandidateStrategy(row_selector="tr.r", field_selectors={"employer": "td.employer"}),
        ]
    )
    no_winner = _JudgeVerdict(winning_candidate_index=None, reasoning="cannot confirm either")

    with patch(
        "threetears.scrape.llm_retry.create_chat_model",
        side_effect=fake_models({_RowCandidateStrategyList: thin_then_rich, _JudgeVerdict: no_winner}),
    ):
        result = await run_eval_loop_multi_row(
            "warn_rows_tie",
            page,
            "https://example.gov/warn",
            _SCHEMA,
            recipe_collection=recipes,
            extraction_collection=extractions,
            health_collection=health,
            api_key="k",
        )

    assert result.validation_status == "needs_review"
    assert result.structured_fields == {"records": [{"employer": "Acme Corp"}, {"employer": "Beta LLC"}]}


# ---------------------------------------------------------------------------
# The strategy shapes themselves
# ---------------------------------------------------------------------------


def test_every_shipped_strategy_shape_is_internally_consistent() -> None:
    """A `_StrategyShape` has ten fields over two real axes, so wrong pairings are constructible.

    Nothing stops someone declaring a single-record `records` alongside the row judge, or a
    row `judge_payload` with the single-record one. Both type-check, and both would hand the
    judge a shape it cannot read -- at which point every candidate is rejected and the target
    quietly stops learning, a failure that looks like a bad page rather than bad wiring.

    Rather than collapse the descriptor (the ten fields are ten genuine differences), this
    pins the invariant that actually matters: the record extractor, the judge and the payload
    adapter must all agree about whether the shape is single-record or multi-row. Asserted
    behaviourally -- what each field DOES -- rather than by identity against private
    functions, which also keeps it honest if one is ever reimplemented.
    """
    from threetears.scrape import eval_loop as el

    shapes = {
        "_CSS_SHAPE": False,
        "_REGEX_SHAPE": False,
        "_CSS_ROW_SHAPE": True,
        "_REGEX_ROW_SHAPE": True,
    }
    one = {"employer": "Acme Corp"}
    two = {"employer": "Beta LLC"}
    # Real result objects, so `records` is exercised against the types the seam declares
    # rather than a duck-typed stand-in that would accept a wrong pairing silently.
    single = ValidationResult(valid=True, extracted=one)
    rows = RowValidationResult(valid=True, records=[one, two], total_rows_matched=2)

    for name, is_row in shapes.items():
        shape = getattr(el, name)

        got = shape.records(rows if is_row else single)
        assert got == ([one, two] if is_row else [one]), f"{name}'s record extractor has the wrong arity"

        judge_name = shape.judge.__name__
        assert ("row" in judge_name) is is_row, f"{name} pairs {judge_name}, the wrong judge for its arity"

        # The payload adapter is the piece that must match the judge: the single-record
        # judge wants one dict per candidate, the row judge wants the whole list.
        payload = shape.judge_payload([one, two])
        assert payload == ([one, two] if is_row else one), f"{name} hands its judge the wrong payload shape"

        assert shape.log_label.startswith("scrape "), f"{name} has an off-pattern log label"


def test_a_strategy_shape_round_trips_its_own_stored_strategy() -> None:
    """`as_strategy` and `from_strategy` must be inverses, or reuse reads back nonsense.

    They are declared as two independent lambdas per shape, so nothing structurally forces
    them to agree. If they disagree, a recipe written by regeneration cannot be read back by
    reuse: every poll re-validates against a garbage strategy, fails, and the target burns its
    way to a regeneration it did not need.
    """
    from threetears.scrape import eval_loop as el

    candidates = {
        "_CSS_SHAPE": {"employer": "td.employer"},
        "_REGEX_SHAPE": r"(?P<employer>Acme Corp)",
        "_CSS_ROW_SHAPE": {"row_selector": "tr", "field_selectors": {"employer": "td"}},
        "_REGEX_ROW_SHAPE": r"(?P<employer>Acme Corp)",
    }
    for name, candidate in candidates.items():
        shape = getattr(el, name)
        assert shape.from_strategy(shape.as_strategy(candidate)) == candidate, (
            f"{name}'s as_strategy/from_strategy are not inverses, so a stored recipe "
            "cannot be read back as the candidate that produced it"
        )
