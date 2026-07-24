"""Unit tests for threetears.scrape.eval_loop -- recipe reuse vs. re-trigger
threshold logic and LLM-judge candidate comparison (mocking approach mirrors
tests/unit/test_query_agent_matching.py's create_chat_model pattern; real
sidecar + real LLM proof lives in tests/e2e/test_scrape_eval_loop_live.py).

Both the candidate-generation call (extraction.py) and the judge call
(eval_loop.py) now funnel through the single shared
``threetears.scrape.llm_retry.create_chat_model`` (backlog SCR-K7M3) -- tests that
need to return different fakes for the two calls dispatch on the ``purpose``
kwarg (``LlmPurpose.EXTRACTION`` vs ``LlmPurpose.UTILITY``) via
:func:`_dispatch_by_purpose`, rather than patching two separate module
namespaces.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from threetears.models import LlmPurpose

from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
from threetears.scrape.eval_loop import (
    _JudgeVerdict,
    _MultiRowJudgeVerdict,
    _judge_multi_row_extraction,
    _judge_one_document_extraction,
    run_eval_loop,
    run_eval_loop_multi_row,
)
from threetears.scrape.extraction import (
    NoticeDocument,
    _CandidateStrategy,
    _CandidateStrategyList,
    _RegexCandidateStrategy,
    _RegexCandidateStrategyList,
    _RowCandidateStrategy,
    _RowCandidateStrategyList,
)
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

_test_registry = CollectionRegistry()
_test_config = DefaultCoreConfig()


def get_registry() -> CollectionRegistry:
    return _test_registry


def get_config() -> DefaultCoreConfig:
    return _test_config


_PAGE_HTML = """
<html><body>
    <table class="warn-notices">
        <tr><td class="employer">Acme Corp</td><td class="count">42</td></tr>
    </table>
</body></html>
"""
_SCHEMA = {"employer": str, "affected_count": int}
_WINNING_STRATEGY = {"employer": "td.employer", "affected_count": "td.count"}

_ROWS_PAGE_HTML = """
<html><body>
<table><tbody>
    <tr><td class="employer">Acme Corp</td><td class="count">42</td></tr>
    <tr><td class="employer">Beta LLC</td><td class="count">7</td></tr>
</tbody></table>
</body></html>
"""
_WINNING_ROW_STRATEGY = {
    "row_selector": "tbody tr",
    "field_selectors": {"employer": "td.employer", "affected_count": "td.count"},
}

# Text-block pages (regex strategy) -- no <table> at all, mirroring
# Pennsylvania's real WARN page shape (Chunk 20's own live proof case):
# labeled fields, one per line, per record.
_TEXT_PAGE_HTML = """
<html><body><p>Acme Corp</p><p>AFFECTED: 42</p></body></html>
"""
_WINNING_REGEX_PATTERN = r"(?P<employer>[^\n]+)\nAFFECTED: (?P<affected_count>\d+)"

_TEXT_ROWS_PAGE_HTML = """
<html><body>
<p>Acme Corp</p><p>AFFECTED: 42</p>
<p>Beta LLC</p><p>AFFECTED: 7</p>
</body></html>
"""
_WINNING_REGEX_ROW_PATTERN = r"(?P<employer>[^\n]+)\nAFFECTED: (?P<affected_count>\d+)"


def _fake_structured_model(result=None, *, side_effect=None):
    ainvoke_mock = AsyncMock(return_value=result, side_effect=side_effect)
    structured = SimpleNamespace(ainvoke=ainvoke_mock)
    return SimpleNamespace(with_structured_output=lambda schema, **kwargs: structured), ainvoke_mock


def _dispatch_by_purpose(extraction_model, judge_model):
    """``create_chat_model`` side_effect: pick the extraction or judge fake by ``purpose``."""

    def _dispatch(*args, purpose=None, **kwargs):
        return extraction_model if purpose == LlmPurpose.EXTRACTION else judge_model

    return _dispatch


def _collections():
    recipe_collection = ScrapeRecipeCollection(get_registry(), get_config(), nats_client=None)
    extraction_collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
    return recipe_collection, extraction_collection


class TestRunEvalLoopFirstRun:
    async def test_no_existing_recipe_generates_and_persists_winner(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _CandidateStrategyList(candidates=[_CandidateStrategy(selectors=_WINNING_STRATEGY)])
        judge_verdict = _JudgeVerdict(
            winning_candidate_index=0, reasoning="matches page content", field_confidences={"employer": "confident"}
        )

        fake_extraction_model, _ = _fake_structured_model(candidates)
        fake_judge_model, _ = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            extraction = await run_eval_loop(
                "warn_act_ca",
                _PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {"records": [{"employer": "Acme Corp", "affected_count": 42}]}
        assert extraction.extraction_recipe_id == "warn_act_ca"
        assert extraction.field_confidences == {"employer": "confident"}

        recipe = await recipe_collection.get("warn_act_ca")
        assert recipe is not None
        assert recipe.extraction_strategy == {"selectors": _WINNING_STRATEGY}
        assert recipe.consecutive_validation_failures == 0

    async def test_no_structurally_valid_candidates_persists_failed_no_recipe(self):
        recipe_collection, extraction_collection = _collections()
        # Every proposed selector matches nothing in the page.
        candidates = _CandidateStrategyList(
            candidates=[_CandidateStrategy(selectors={"employer": ".nope", "affected_count": ".also-nope"})]
        )
        fake_extraction_model, _ = _fake_structured_model(candidates)

        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_extraction_model):
            extraction = await run_eval_loop(
                "warn_act_ca",
                _PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        assert extraction.validation_status == "failed"
        assert extraction.extraction_recipe_id is None
        assert await recipe_collection.get("warn_act_ca") is None

    async def test_judge_picks_no_winner_persists_needs_review_no_recipe(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _CandidateStrategyList(candidates=[_CandidateStrategy(selectors=_WINNING_STRATEGY)])
        judge_verdict = _JudgeVerdict(winning_candidate_index=None, reasoning="none of these look right")

        fake_extraction_model, _ = _fake_structured_model(candidates)
        fake_judge_model, _ = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            extraction = await run_eval_loop(
                "warn_act_ca",
                _PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        assert extraction.validation_status == "needs_review"
        assert extraction.extraction_recipe_id is None
        assert await recipe_collection.get("warn_act_ca") is None

    async def test_judge_failure_degrades_to_needs_review_not_a_crash(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _CandidateStrategyList(candidates=[_CandidateStrategy(selectors=_WINNING_STRATEGY)])
        fake_extraction_model, _ = _fake_structured_model(candidates)
        fake_judge_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))

        with (
            patch(
                "threetears.scrape.llm_retry.create_chat_model",
                side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
            ),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            extraction = await run_eval_loop(
                "warn_act_ca",
                _PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        assert extraction.validation_status == "needs_review"


class TestRunEvalLoopRecipeReuse:
    async def test_healthy_recipe_reused_without_any_llm_call(self):
        recipe_collection, extraction_collection = _collections()
        recipe_entity = recipe_collection.create(
            {
                "target_id": "warn_act_ca",
                "extraction_strategy": {"selectors": _WINNING_STRATEGY},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
        await recipe_collection.save_entity(recipe_entity)

        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            extraction = await run_eval_loop(
                "warn_act_ca",
                _PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        create_model.assert_not_called()
        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {"records": [{"employer": "Acme Corp", "affected_count": 42}]}
        assert extraction.extraction_recipe_id == "warn_act_ca"

        recipe = await recipe_collection.get("warn_act_ca")
        assert recipe.consecutive_validation_failures == 0

    async def test_below_threshold_failure_keeps_recipe_and_increments_counter(self):
        recipe_collection, extraction_collection = _collections()
        recipe_entity = recipe_collection.create(
            {
                "target_id": "warn_act_ca",
                # A selector that no longer matches the (changed) page -- simulates
                # a site markup change without yet crossing the failure threshold.
                "extraction_strategy": {"selectors": {"employer": ".gone", "affected_count": ".also-gone"}},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
        await recipe_collection.save_entity(recipe_entity)

        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            extraction = await run_eval_loop(
                "warn_act_ca",
                _PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                failure_threshold=3,
            )

        create_model.assert_not_called()  # still below threshold -- no regeneration yet
        assert extraction.validation_status == "failed"
        assert extraction.extraction_recipe_id == "warn_act_ca"

        recipe = await recipe_collection.get("warn_act_ca")
        assert recipe.consecutive_validation_failures == 1
        assert recipe.extraction_strategy == {"selectors": {"employer": ".gone", "affected_count": ".also-gone"}}

    async def test_threshold_crossed_triggers_regeneration(self):
        recipe_collection, extraction_collection = _collections()
        recipe_entity = recipe_collection.create(
            {
                "target_id": "warn_act_ca",
                "extraction_strategy": {"selectors": {"employer": ".gone"}},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 3,  # already at the default threshold
            }
        )
        await recipe_collection.save_entity(recipe_entity)

        candidates = _CandidateStrategyList(candidates=[_CandidateStrategy(selectors=_WINNING_STRATEGY)])
        judge_verdict = _JudgeVerdict(winning_candidate_index=0, reasoning="matches page content")
        fake_extraction_model, extraction_ainvoke = _fake_structured_model(candidates)
        fake_judge_model, judge_ainvoke = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            extraction = await run_eval_loop(
                "warn_act_ca",
                _PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                failure_threshold=3,
            )

        assert extraction_ainvoke.await_count == 1
        assert judge_ainvoke.await_count == 1
        assert extraction.validation_status == "validated"

        recipe = await recipe_collection.get("warn_act_ca")
        assert recipe.extraction_strategy == {"selectors": _WINNING_STRATEGY}
        assert recipe.consecutive_validation_failures == 0

    async def test_second_run_against_same_healthy_recipe_reuses_it_again(self):
        """The build plan's own acceptance criteria: a second run against the
        same page reuses the recipe rather than re-invoking candidate generation."""
        recipe_collection, extraction_collection = _collections()
        candidates = _CandidateStrategyList(candidates=[_CandidateStrategy(selectors=_WINNING_STRATEGY)])
        judge_verdict = _JudgeVerdict(winning_candidate_index=0, reasoning="matches")
        fake_extraction_model, extraction_ainvoke = _fake_structured_model(candidates)
        fake_judge_model, judge_ainvoke = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            first = await run_eval_loop(
                "warn_act_ca",
                _PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )
            second = await run_eval_loop(
                "warn_act_ca",
                _PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        assert extraction_ainvoke.await_count == 1  # candidate generation ran exactly once
        assert judge_ainvoke.await_count == 1  # judge ran exactly once
        assert first.id != second.id  # two distinct fetch rows
        assert first.structured_fields == second.structured_fields
        assert second.extraction_recipe_id == "warn_act_ca"


# ===========================================================================
# run_eval_loop_multi_row -- Chunk 07: many records per page, not one
# ===========================================================================


class TestRunEvalLoopMultiRowFirstRun:
    async def test_no_existing_recipe_generates_and_persists_every_row(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _RowCandidateStrategyList(candidates=[_RowCandidateStrategy(**_WINNING_ROW_STRATEGY)])
        judge_verdict = _JudgeVerdict(
            winning_candidate_index=0, reasoning="matches page content", field_confidences={"employer": "confident"}
        )
        fake_extraction_model, _ = _fake_structured_model(candidates)
        fake_judge_model, _ = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_md",
                _ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {
            "records": [
                {"employer": "Acme Corp", "affected_count": 42},
                {"employer": "Beta LLC", "affected_count": 7},
            ]
        }
        assert extraction.extraction_recipe_id == "warn_act_md"
        assert extraction.field_confidences == {"employer": "confident"}

        recipe = await recipe_collection.get("warn_act_md")
        assert recipe is not None
        assert recipe.extraction_strategy == _WINNING_ROW_STRATEGY
        assert recipe.consecutive_validation_failures == 0

    async def test_no_structurally_valid_candidates_persists_failed_no_recipe(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _RowCandidateStrategyList(
            candidates=[_RowCandidateStrategy(row_selector=".nope", field_selectors={"employer": ".also-nope"})]
        )
        fake_extraction_model, _ = _fake_structured_model(candidates)

        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_extraction_model):
            extraction = await run_eval_loop_multi_row(
                "warn_act_md",
                _ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        assert extraction.validation_status == "failed"
        assert extraction.structured_fields == {"records": []}
        assert extraction.extraction_recipe_id is None
        assert await recipe_collection.get("warn_act_md") is None

    async def test_judge_picks_no_winner_surfaces_best_row_count_candidate(self):
        """Unlike the single-record path, "best" has a real comparable signal
        here (row count captured), not just "first proposed"."""
        recipe_collection, extraction_collection = _collections()
        candidates = _RowCandidateStrategyList(
            candidates=[
                # This one only captures 1 of the 2 real rows (employer selector too narrow).
                _RowCandidateStrategy(
                    row_selector="tbody tr", field_selectors={"employer": "td.employer:-soup-contains('Acme')"}
                ),
                _RowCandidateStrategy(**_WINNING_ROW_STRATEGY),
            ]
        )
        judge_verdict = _JudgeVerdict(winning_candidate_index=None, reasoning="none of these look right")
        fake_extraction_model, _ = _fake_structured_model(candidates)
        fake_judge_model, _ = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_md",
                _ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        assert extraction.validation_status == "needs_review"
        assert extraction.extraction_recipe_id is None
        # The 2-row candidate won on row count, not the 1-row candidate.
        assert len(extraction.structured_fields["records"]) == 2
        assert await recipe_collection.get("warn_act_md") is None


class TestRunEvalLoopMultiRowRecipeReuse:
    async def test_healthy_recipe_reused_without_any_llm_call(self):
        recipe_collection, extraction_collection = _collections()
        recipe_entity = recipe_collection.create(
            {
                "target_id": "warn_act_md",
                "extraction_strategy": _WINNING_ROW_STRATEGY,
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
        await recipe_collection.save_entity(recipe_entity)

        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            extraction = await run_eval_loop_multi_row(
                "warn_act_md",
                _ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        create_model.assert_not_called()
        assert extraction.validation_status == "validated"
        assert len(extraction.structured_fields["records"]) == 2

        recipe = await recipe_collection.get("warn_act_md")
        assert recipe.consecutive_validation_failures == 0

    async def test_below_threshold_failure_keeps_recipe_and_increments_counter(self):
        recipe_collection, extraction_collection = _collections()
        recipe_entity = recipe_collection.create(
            {
                "target_id": "warn_act_md",
                # Selectors that no longer match the (changed) page.
                "extraction_strategy": {"row_selector": "tbody tr", "field_selectors": {"employer": ".gone"}},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
        await recipe_collection.save_entity(recipe_entity)

        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            extraction = await run_eval_loop_multi_row(
                "warn_act_md",
                _ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                failure_threshold=3,
            )

        create_model.assert_not_called()
        assert extraction.validation_status == "failed"
        assert extraction.structured_fields == {"records": []}

        recipe = await recipe_collection.get("warn_act_md")
        assert recipe.consecutive_validation_failures == 1

    async def test_threshold_crossed_triggers_regeneration(self):
        recipe_collection, extraction_collection = _collections()
        recipe_entity = recipe_collection.create(
            {
                "target_id": "warn_act_md",
                "extraction_strategy": {"row_selector": "tbody tr", "field_selectors": {"employer": ".gone"}},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 3,
            }
        )
        await recipe_collection.save_entity(recipe_entity)

        candidates = _RowCandidateStrategyList(candidates=[_RowCandidateStrategy(**_WINNING_ROW_STRATEGY)])
        judge_verdict = _JudgeVerdict(winning_candidate_index=0, reasoning="matches page content")
        fake_extraction_model, extraction_ainvoke = _fake_structured_model(candidates)
        fake_judge_model, judge_ainvoke = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_md",
                _ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                failure_threshold=3,
            )

        assert extraction_ainvoke.await_count == 1
        assert judge_ainvoke.await_count == 1
        assert extraction.validation_status == "validated"

        recipe = await recipe_collection.get("warn_act_md")
        assert recipe.extraction_strategy == _WINNING_ROW_STRATEGY
        assert recipe.consecutive_validation_failures == 0

    async def test_second_run_against_same_healthy_recipe_reuses_it_again(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _RowCandidateStrategyList(candidates=[_RowCandidateStrategy(**_WINNING_ROW_STRATEGY)])
        judge_verdict = _JudgeVerdict(winning_candidate_index=0, reasoning="matches")
        fake_extraction_model, extraction_ainvoke = _fake_structured_model(candidates)
        fake_judge_model, judge_ainvoke = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            first = await run_eval_loop_multi_row(
                "warn_act_md",
                _ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )
            second = await run_eval_loop_multi_row(
                "warn_act_md",
                _ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        assert extraction_ainvoke.await_count == 1
        assert judge_ainvoke.await_count == 1
        assert first.id != second.id
        assert first.structured_fields == second.structured_fields


# ===========================================================================
# Regex/text-block strategy (2026-07-14) -- strategy_type="regex". Same
# propose -> structurally-validate -> judge -> persist cycle as the CSS
# tests above, mirrored exactly, just against a text-block page shape with
# no <table> at all (Pennsylvania's real WARN page is the concrete driver
# -- see build-plan.md Chunk 20). The judge step itself is unmodified/
# shared code, already covered by the CSS tests above.
# ===========================================================================


class TestRunEvalLoopRegexStrategyFirstRun:
    async def test_no_existing_recipe_generates_and_persists_winner(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _RegexCandidateStrategyList(candidates=[_RegexCandidateStrategy(pattern=_WINNING_REGEX_PATTERN)])
        judge_verdict = _JudgeVerdict(
            winning_candidate_index=0, reasoning="matches page content", field_confidences={"employer": "confident"}
        )
        fake_extraction_model, _ = _fake_structured_model(candidates)
        fake_judge_model, _ = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            extraction = await run_eval_loop(
                "warn_act_pa",
                _TEXT_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="regex",
            )

        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {"records": [{"employer": "Acme Corp", "affected_count": 42}]}
        assert extraction.extraction_recipe_id == "warn_act_pa"
        assert extraction.field_confidences == {"employer": "confident"}

        recipe = await recipe_collection.get("warn_act_pa")
        assert recipe is not None
        assert recipe.extraction_strategy == {"pattern": _WINNING_REGEX_PATTERN}
        assert recipe.consecutive_validation_failures == 0

    async def test_no_structurally_valid_candidates_persists_failed_no_recipe(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _RegexCandidateStrategyList(
            candidates=[_RegexCandidateStrategy(pattern=r"NOPE: (?P<employer>x)(?P<affected_count>x)")]
        )
        fake_extraction_model, _ = _fake_structured_model(candidates)

        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_extraction_model):
            extraction = await run_eval_loop(
                "warn_act_pa",
                _TEXT_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="regex",
            )

        assert extraction.validation_status == "failed"
        assert extraction.extraction_recipe_id is None
        assert await recipe_collection.get("warn_act_pa") is None

    async def test_invalid_regex_pattern_is_rejected_not_a_crash(self):
        """A malformed pattern (unbalanced parens) must fail structural
        validation cleanly, not raise out of the eval loop."""
        recipe_collection, extraction_collection = _collections()
        candidates = _RegexCandidateStrategyList(candidates=[_RegexCandidateStrategy(pattern=r"(?P<employer>[")])
        fake_extraction_model, _ = _fake_structured_model(candidates)

        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_extraction_model):
            extraction = await run_eval_loop(
                "warn_act_pa",
                _TEXT_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="regex",
            )

        assert extraction.validation_status == "failed"


class TestRunEvalLoopRegexStrategyRecipeReuse:
    async def test_healthy_recipe_reused_without_any_llm_call(self):
        recipe_collection, extraction_collection = _collections()
        recipe_entity = recipe_collection.create(
            {
                "target_id": "warn_act_pa",
                "extraction_strategy": {"pattern": _WINNING_REGEX_PATTERN},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
        await recipe_collection.save_entity(recipe_entity)

        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            extraction = await run_eval_loop(
                "warn_act_pa",
                _TEXT_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="regex",
            )

        create_model.assert_not_called()
        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {"records": [{"employer": "Acme Corp", "affected_count": 42}]}

    async def test_below_threshold_failure_keeps_recipe_and_increments_counter(self):
        recipe_collection, extraction_collection = _collections()
        recipe_entity = recipe_collection.create(
            {
                "target_id": "warn_act_pa",
                "extraction_strategy": {"pattern": r"GONE: (?P<employer>x)(?P<affected_count>x)"},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
        await recipe_collection.save_entity(recipe_entity)

        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            extraction = await run_eval_loop(
                "warn_act_pa",
                _TEXT_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="regex",
                failure_threshold=3,
            )

        create_model.assert_not_called()
        assert extraction.validation_status == "failed"

        recipe = await recipe_collection.get("warn_act_pa")
        assert recipe.consecutive_validation_failures == 1


class TestRunEvalLoopMultiRowRegexStrategyFirstRun:
    async def test_no_existing_recipe_generates_and_persists_every_row(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _RegexCandidateStrategyList(
            candidates=[_RegexCandidateStrategy(pattern=_WINNING_REGEX_ROW_PATTERN)]
        )
        judge_verdict = _JudgeVerdict(
            winning_candidate_index=0, reasoning="matches page content", field_confidences={"employer": "confident"}
        )
        fake_extraction_model, _ = _fake_structured_model(candidates)
        fake_judge_model, _ = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_pa",
                _TEXT_ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="regex",
            )

        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {
            "records": [
                {"employer": "Acme Corp", "affected_count": 42},
                {"employer": "Beta LLC", "affected_count": 7},
            ]
        }
        assert extraction.extraction_recipe_id == "warn_act_pa"

        recipe = await recipe_collection.get("warn_act_pa")
        assert recipe is not None
        assert recipe.extraction_strategy == {"pattern": _WINNING_REGEX_ROW_PATTERN}

    async def test_no_structurally_valid_candidates_persists_failed_no_recipe(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _RegexCandidateStrategyList(
            candidates=[_RegexCandidateStrategy(pattern=r"NOPE: (?P<employer>x)(?P<affected_count>x)")]
        )
        fake_extraction_model, _ = _fake_structured_model(candidates)

        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_extraction_model):
            extraction = await run_eval_loop_multi_row(
                "warn_act_pa",
                _TEXT_ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="regex",
            )

        assert extraction.validation_status == "failed"
        assert extraction.structured_fields == {"records": []}
        assert extraction.extraction_recipe_id is None


class TestRunEvalLoopMultiRowRegexStrategyRecipeReuse:
    async def test_healthy_recipe_reused_without_any_llm_call(self):
        recipe_collection, extraction_collection = _collections()
        recipe_entity = recipe_collection.create(
            {
                "target_id": "warn_act_pa",
                "extraction_strategy": {"pattern": _WINNING_REGEX_ROW_PATTERN},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
        await recipe_collection.save_entity(recipe_entity)

        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            extraction = await run_eval_loop_multi_row(
                "warn_act_pa",
                _TEXT_ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="regex",
            )

        create_model.assert_not_called()
        assert extraction.validation_status == "validated"
        assert len(extraction.structured_fields["records"]) == 2

    async def test_second_run_against_same_healthy_recipe_reuses_it_again(self):
        recipe_collection, extraction_collection = _collections()
        candidates = _RegexCandidateStrategyList(
            candidates=[_RegexCandidateStrategy(pattern=_WINNING_REGEX_ROW_PATTERN)]
        )
        judge_verdict = _JudgeVerdict(winning_candidate_index=0, reasoning="matches")
        fake_extraction_model, extraction_ainvoke = _fake_structured_model(candidates)
        fake_judge_model, judge_ainvoke = _fake_structured_model(judge_verdict)

        with patch(
            "threetears.scrape.llm_retry.create_chat_model",
            side_effect=_dispatch_by_purpose(fake_extraction_model, fake_judge_model),
        ):
            first = await run_eval_loop_multi_row(
                "warn_act_pa",
                _TEXT_ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="regex",
            )
            second = await run_eval_loop_multi_row(
                "warn_act_pa",
                _TEXT_ROWS_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="regex",
            )

        assert extraction_ainvoke.await_count == 1
        assert judge_ainvoke.await_count == 1
        assert first.id != second.id
        assert first.structured_fields == second.structured_fields
        assert second.extraction_recipe_id == "warn_act_pa"


# ===========================================================================
# run_eval_loop_multi_row -- strategy_type="per_document" (scrape-task-05)
# ===========================================================================

_NOTICES_PAGE_HTML = """
<html><body>
<div class="notice"><p>Acme Corp</p><p>We will terminate 42 employees on July 1.</p></div>
<div class="notice"><p>Beta LLC</p><p>7 positions eliminated effective August 15.</p></div>
</body></html>
"""


class TestRunEvalLoopMultiRowPerDocumentStrategy:
    """Mocks at the extract_fields_directly_chunked / _judge_one_document_extraction
    function boundary (patched directly on the eval_loop module), not the deep
    create_chat_model level -- these tests exercise per-document ROUTING/aggregation
    logic; extraction's and the judge's own internals get their own dedicated unit
    tests in test_extraction.py and TestJudgeOneDocumentExtraction below."""

    async def test_extracts_one_confirmed_record_per_document(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()

        async def fake_extract(text, schema, *, model_id, api_key):
            return (
                {"employer": "Acme Corp", "affected_count": 42}
                if "Acme" in text
                else {"employer": "Beta LLC", "affected_count": 7}
            )

        judge_mock = AsyncMock(return_value=True)

        with (
            patch.object(eval_loop_module, "extract_fields_directly_chunked", fake_extract),
            patch.object(eval_loop_module, "_judge_one_document_extraction", judge_mock),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_wv",
                _NOTICES_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {
            "records": [
                {"employer": "Acme Corp", "affected_count": 42},
                {"employer": "Beta LLC", "affected_count": 7},
            ]
        }
        assert extraction.extraction_recipe_id == "warn_act_wv"
        assert judge_mock.await_count == 2

    async def test_a_judge_rejected_record_is_dropped_even_though_structurally_complete(self):
        """The whole point of scrape-task-06's own grounding check: a well-typed,
        complete-looking extraction that the judge says is wrong/hallucinated must
        never count as a real record."""
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        single_notice_html = '<html><body><div class="notice"><p>Acme Corp</p></div></body></html>'

        async def fake_extract(text, schema, *, model_id, api_key):
            return {"employer": "Acme Corp", "affected_count": 42}

        with (
            patch.object(eval_loop_module, "extract_fields_directly_chunked", fake_extract),
            patch.object(eval_loop_module, "_judge_one_document_extraction", AsyncMock(return_value=False)),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_wv",
                single_notice_html,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        assert extraction.validation_status == "failed"
        assert extraction.structured_fields == {"records": []}

    async def test_second_run_always_calls_the_llm_again_no_caching(self):
        """Unlike css/regex, a healthy "recipe" (there isn't a reusable one) never
        skips the LLM call -- every document, every poll, gets its own fresh call."""
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        single_notice_html = '<html><body><div class="notice"><p>Acme Corp</p></div></body></html>'
        extract_mock = AsyncMock(return_value={"employer": "Acme Corp", "affected_count": 42})

        with (
            patch.object(eval_loop_module, "extract_fields_directly_chunked", extract_mock),
            patch.object(eval_loop_module, "_judge_one_document_extraction", AsyncMock(return_value=True)),
        ):
            await run_eval_loop_multi_row(
                "warn_act_wv",
                single_notice_html,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )
            await run_eval_loop_multi_row(
                "warn_act_wv",
                single_notice_html,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        assert extract_mock.await_count == 2

    async def test_a_document_missing_a_required_field_is_dropped_without_spending_a_judge_call(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        # affected_count missing (None) -- all-or-nothing, this record must not appear.
        extract_mock = AsyncMock(return_value={"employer": "Acme Corp"})
        judge_mock = AsyncMock(return_value=True)
        single_notice_html = '<html><body><div class="notice"><p>Acme Corp</p></div></body></html>'

        with (
            patch.object(eval_loop_module, "extract_fields_directly_chunked", extract_mock),
            patch.object(eval_loop_module, "_judge_one_document_extraction", judge_mock),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_wv",
                single_notice_html,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        assert extraction.validation_status == "failed"
        assert extraction.structured_fields == {"records": []}
        assert extraction.extraction_recipe_id is None
        judge_mock.assert_not_called()

    async def test_no_notice_divs_found_persists_failed_not_a_crash(self):
        recipe_collection, extraction_collection = _collections()

        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            extraction = await run_eval_loop_multi_row(
                "warn_act_wv",
                "<html><body>no notices here</body></html>",
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        create_model.assert_not_called()
        assert extraction.validation_status == "failed"
        assert extraction.structured_fields == {"records": []}

    async def test_one_document_llm_total_failure_does_not_sink_the_others(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()

        async def fake_extract(text, schema, *, model_id, api_key):
            if "Acme" in text:
                return None
            return {"employer": "Beta LLC", "affected_count": 7}

        with (
            patch.object(eval_loop_module, "extract_fields_directly_chunked", fake_extract),
            patch.object(eval_loop_module, "_judge_one_document_extraction", AsyncMock(return_value=True)),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_wv",
                _NOTICES_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {"records": [{"employer": "Beta LLC", "affected_count": 7}]}

    async def test_one_document_hanging_past_the_deadline_does_not_block_the_others(self):
        """Live-reproduced (scrape-task-05): the underlying chat client can hang well
        past its own per-attempt timeout with zero further retry activity -- a real
        West Virginia document reproduced this directly. asyncio.wait_for's outer
        deadline (_PER_DOCUMENT_TIMEOUT_SECONDS) is what actually bounds it, not
        extract_fields_directly_chunked's own timeout/attempts alone. Documents run
        concurrently (asyncio.gather) -- call_count still orders deterministically
        since it increments synchronously before either fake awaits anything."""
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        call_count = 0

        async def fake_extract_chunked(text, schema, *, model_id, api_key):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulates the live-reproduced hang -- asyncio.wait_for's outer
                # deadline cancels this mid-sleep; it never returns on its own.
                await asyncio.sleep(1)
            return {
                "employer": "Beta LLC",
                "affected_count": 7,
            }  # already-coerced, extract_fields_directly's own contract

        with (
            patch.object(eval_loop_module, "extract_fields_directly_chunked", fake_extract_chunked),
            patch.object(eval_loop_module, "_judge_one_document_extraction", AsyncMock(return_value=True)),
            patch.object(eval_loop_module, "_PER_DOCUMENT_TIMEOUT_SECONDS", 0.05),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_wv",
                _NOTICES_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {"records": [{"employer": "Beta LLC", "affected_count": 7}]}

    async def test_a_hanging_judge_call_is_bounded_the_same_as_a_hanging_extraction(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        single_notice_html = '<html><body><div class="notice"><p>Acme Corp</p></div></body></html>'

        async def hanging_judge(document, extracted, schema, *, api_key, judge_model_id):
            await asyncio.sleep(1)
            raise AssertionError("should have been cancelled by the outer deadline")

        with (
            patch.object(
                eval_loop_module,
                "extract_fields_directly_chunked",
                AsyncMock(return_value={"employer": "Acme Corp", "affected_count": 42}),
            ),
            patch.object(eval_loop_module, "_judge_one_document_extraction", hanging_judge),
            patch.object(eval_loop_module, "_PER_DOCUMENT_TIMEOUT_SECONDS", 0.05),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_wv",
                single_notice_html,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        assert extraction.validation_status == "failed"
        assert extraction.structured_fields == {"records": []}

    async def test_consecutive_failures_tracked_on_the_marker_recipe(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        single_notice_html = '<html><body><div class="notice"><p>Acme Corp</p></div></body></html>'

        with (
            patch.object(
                eval_loop_module, "extract_fields_directly_chunked", AsyncMock(return_value={"employer": "Acme Corp"})
            ),
            patch.object(eval_loop_module, "_judge_one_document_extraction", AsyncMock(return_value=True)),
        ):
            await run_eval_loop_multi_row(
                "warn_act_wv",
                single_notice_html,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )
            await run_eval_loop_multi_row(
                "warn_act_wv",
                single_notice_html,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        recipe = await recipe_collection.get("warn_act_wv")
        assert recipe is not None
        assert recipe.consecutive_validation_failures == 2
        assert recipe.extraction_strategy == {"strategy": "per_document"}


# ===========================================================================
# run_eval_loop_multi_row -- per_document routing by document shape (scrape-task-06)
# ===========================================================================

_MIXED_SHAPE_PAGE_HTML = """
<html><body>
<div class="notice" data-was-ocr="false"><p>Acme Corp born-digital</p></div>
<div class="notice" data-was-ocr="true"><p>ignored ocr text</p><img class="ocr-page-image" data-page="0" src="data:image/png;base64,ZmFrZS1wbmc="></div>
</body></html>
"""


class TestRunEvalLoopMultiRowVisionRouting:
    async def test_scanned_document_routes_to_vision_born_digital_routes_to_text(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        text_calls: list[str] = []
        vision_calls: list[list[bytes]] = []

        async def fake_text(text, schema, *, model_id, api_key):
            text_calls.append(text)
            return {"employer": "Acme Corp", "affected_count": 1}

        async def fake_vision(images, schema, *, api_key):
            vision_calls.append(images)
            return {"employer": "Scanned Co", "affected_count": 2}

        judge_calls: list[bool] = []

        async def fake_judge(document, extracted, schema, *, api_key, judge_model_id):
            judge_calls.append(document.was_ocr)
            return True

        with (
            patch.object(eval_loop_module, "extract_fields_directly_chunked", fake_text),
            patch.object(eval_loop_module, "extract_fields_from_images", fake_vision),
            patch.object(eval_loop_module, "_judge_one_document_extraction", fake_judge),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_hi",
                _MIXED_SHAPE_PAGE_HTML,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        assert len(text_calls) == 1
        assert len(vision_calls) == 1
        assert vision_calls[0] == [b"fake-png"]
        assert sorted(judge_calls) == [False, True]  # judged once per document, matching its own was_ocr
        assert extraction.validation_status == "validated"
        records = extraction.structured_fields["records"]
        assert {"employer": "Acme Corp", "affected_count": 1} in records
        assert {"employer": "Scanned Co", "affected_count": 2} in records

    async def test_a_hanging_vision_call_is_bounded_the_same_as_a_hanging_text_call(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        single_scanned_html = (
            '<html><body><div class="notice" data-was-ocr="true">'
            '<img class="ocr-page-image" data-page="0" src="data:image/png;base64,ZmFrZS1wbmc=">'
            "</div></body></html>"
        )

        async def hanging_vision(images, schema, *, api_key):
            await asyncio.sleep(1)
            raise AssertionError("should have been cancelled by the outer deadline")

        with (
            patch.object(eval_loop_module, "extract_fields_from_images", hanging_vision),
            patch.object(eval_loop_module, "_judge_one_document_extraction", AsyncMock(return_value=True)),
            patch.object(eval_loop_module, "_PER_DOCUMENT_TIMEOUT_SECONDS", 0.05),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_hi",
                single_scanned_html,
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="per_document",
            )

        assert extraction.validation_status == "failed"
        assert extraction.structured_fields == {"records": []}


class TestRunEvalLoopMultiRowVisionStrategy:
    """Mocks at the extract_page_images / extract_multi_row_fields_from_images /
    _judge_multi_row_extraction function boundary, same mocking-pattern-evolution
    lesson as TestRunEvalLoopMultiRowPerDocumentStrategy -- these tests exercise
    multi_row_vision's own ROUTING/aggregation/validation_status logic; extraction's
    and the judge's own internals get their own dedicated unit tests below and in
    test_extraction.py."""

    async def test_all_records_confirmed_is_validated(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        records = [
            {"employer": "Acme Corp", "affected_count": 42},
            {"employer": "Beta LLC", "affected_count": 7},
        ]

        with (
            patch.object(eval_loop_module, "extract_page_images", lambda html: [b"page-0"]),
            patch.object(eval_loop_module, "extract_multi_row_fields_from_images", AsyncMock(return_value=records)),
            patch.object(eval_loop_module, "_judge_multi_row_extraction", AsyncMock(return_value={0, 1})),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_nv",
                "<html><body>irrelevant, extract_page_images is mocked</body></html>",
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="multi_row_vision",
            )

        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {"records": records}
        assert extraction.extraction_recipe_id == "warn_act_nv"

    async def test_some_records_rejected_is_needs_review_not_silently_validated(self):
        """The real teeth of the multi-row grounding requirement: a 17-row table
        that's 15/17 right must never look identical to a 17/17 clean pass."""
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        records = [
            {"employer": "Acme Corp", "affected_count": 42},
            {"employer": "Wrong Row", "affected_count": 999},
            {"employer": "Beta LLC", "affected_count": 7},
        ]

        with (
            patch.object(eval_loop_module, "extract_page_images", lambda html: [b"page-0"]),
            patch.object(eval_loop_module, "extract_multi_row_fields_from_images", AsyncMock(return_value=records)),
            patch.object(eval_loop_module, "_judge_multi_row_extraction", AsyncMock(return_value={0, 2})),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_nv",
                "<html><body>irrelevant</body></html>",
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="multi_row_vision",
            )

        assert extraction.validation_status == "needs_review"
        assert extraction.structured_fields == {
            "records": [
                {"employer": "Acme Corp", "affected_count": 42},
                {"employer": "Beta LLC", "affected_count": 7},
            ]
        }
        # the rejected row is dropped, never persisted as if it were correct
        assert {"employer": "Wrong Row", "affected_count": 999} not in extraction.structured_fields["records"]

    async def test_zero_records_confirmed_is_failed(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        records = [{"employer": "Acme Corp", "affected_count": 42}]

        with (
            patch.object(eval_loop_module, "extract_page_images", lambda html: [b"page-0"]),
            patch.object(eval_loop_module, "extract_multi_row_fields_from_images", AsyncMock(return_value=records)),
            patch.object(eval_loop_module, "_judge_multi_row_extraction", AsyncMock(return_value=set())),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_nv",
                "<html><body>irrelevant</body></html>",
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="multi_row_vision",
            )

        assert extraction.validation_status == "failed"
        assert extraction.structured_fields == {"records": []}

    async def test_no_page_images_fails_without_calling_extraction_or_judge(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        extract_mock = AsyncMock()
        judge_mock = AsyncMock()

        with (
            patch.object(eval_loop_module, "extract_page_images", lambda html: []),
            patch.object(eval_loop_module, "extract_multi_row_fields_from_images", extract_mock),
            patch.object(eval_loop_module, "_judge_multi_row_extraction", judge_mock),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_nv",
                "<html><body>no images here</body></html>",
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="multi_row_vision",
            )

        assert extraction.validation_status == "failed"
        extract_mock.assert_not_called()
        judge_mock.assert_not_called()

    async def test_no_page_images_still_persists_a_recipe_bumping_the_failure_counter(self):
        """Independent-review finding: the zero-images early return skipped
        _save_recipe entirely, contradicting this strategy's own claim to persist
        a marker recipe for the same operational-observability reason per_document
        does (consecutive_validation_failures must advance on every real failure,
        not just extraction/judge failures downstream of a successful image find)."""
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()

        with patch.object(eval_loop_module, "extract_page_images", lambda html: []):
            await run_eval_loop_multi_row(
                "warn_act_nv",
                "<html><body>no images here</body></html>",
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="multi_row_vision",
            )

        recipe = await recipe_collection.get("warn_act_nv")
        assert recipe is not None
        assert recipe.consecutive_validation_failures == 1

    async def test_a_record_missing_a_required_field_is_dropped_before_spending_a_judge_call(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        records = [
            {"employer": "Acme Corp", "affected_count": 42},
            {"employer": "Incomplete Row"},  # missing affected_count
        ]
        judge_mock = AsyncMock(return_value={0})

        with (
            patch.object(eval_loop_module, "extract_page_images", lambda html: [b"page-0"]),
            patch.object(eval_loop_module, "extract_multi_row_fields_from_images", AsyncMock(return_value=records)),
            patch.object(eval_loop_module, "_judge_multi_row_extraction", judge_mock),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_nv",
                "<html><body>irrelevant</body></html>",
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="multi_row_vision",
            )

        assert extraction.validation_status == "validated"
        assert extraction.structured_fields == {"records": [{"employer": "Acme Corp", "affected_count": 42}]}
        # only the one complete record was ever handed to the judge
        judge_mock.assert_awaited_once()
        assert judge_mock.await_args.args[1] == [{"employer": "Acme Corp", "affected_count": 42}]

    async def test_extraction_returning_none_is_failed_not_a_crash(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        judge_mock = AsyncMock()

        with (
            patch.object(eval_loop_module, "extract_page_images", lambda html: [b"page-0"]),
            patch.object(eval_loop_module, "extract_multi_row_fields_from_images", AsyncMock(return_value=None)),
            patch.object(eval_loop_module, "_judge_multi_row_extraction", judge_mock),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_nv",
                "<html><body>irrelevant</body></html>",
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="multi_row_vision",
            )

        assert extraction.validation_status == "failed"
        judge_mock.assert_not_called()

    async def test_a_hanging_extraction_call_is_bounded(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()

        async def hanging_extraction(images, schema, *, api_key):
            await asyncio.sleep(1)
            raise AssertionError("should have been cancelled by the outer deadline")

        with (
            patch.object(eval_loop_module, "extract_page_images", lambda html: [b"page-0"]),
            patch.object(eval_loop_module, "extract_multi_row_fields_from_images", hanging_extraction),
            patch.object(eval_loop_module, "_MULTI_ROW_EXTRACTION_TIMEOUT_SECONDS", 0.05),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_nv",
                "<html><body>irrelevant</body></html>",
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="multi_row_vision",
            )

        assert extraction.validation_status == "failed"

    async def test_a_hanging_judge_call_is_bounded_and_treats_all_records_as_unconfirmed(self):
        import threetears.scrape.eval_loop as eval_loop_module

        recipe_collection, extraction_collection = _collections()
        records = [{"employer": "Acme Corp", "affected_count": 42}]

        async def hanging_judge(images, records, schema, *, api_key):
            await asyncio.sleep(1)
            raise AssertionError("should have been cancelled by the outer deadline")

        with (
            patch.object(eval_loop_module, "extract_page_images", lambda html: [b"page-0"]),
            patch.object(eval_loop_module, "extract_multi_row_fields_from_images", AsyncMock(return_value=records)),
            patch.object(eval_loop_module, "_judge_multi_row_extraction", hanging_judge),
            patch.object(eval_loop_module, "_MULTI_ROW_EXTRACTION_TIMEOUT_SECONDS", 0.05),
        ):
            extraction = await run_eval_loop_multi_row(
                "warn_act_nv",
                "<html><body>irrelevant</body></html>",
                "https://example.gov/warn",
                _SCHEMA,
                recipe_collection=recipe_collection,
                extraction_collection=extraction_collection,
                api_key="k",
                strategy_type="multi_row_vision",
            )

        assert extraction.validation_status == "failed"
        assert extraction.structured_fields == {"records": []}


# ===========================================================================
# _judge_one_document_extraction -- per_document's own grounding check (scrape-task-06)
# ===========================================================================

_TEXT_DOCUMENT = NoticeDocument(text="Acme Corp letter text", was_ocr=False, images=[])
_VISION_DOCUMENT = NoticeDocument(text="", was_ocr=True, images=[b"fake-png-page-0"])
_EXTRACTED = {"employer": "Acme Corp", "affected_count": 42}


class TestJudgeOneDocumentExtraction:
    async def test_confirmed_verdict_returns_true(self):
        verdict = _JudgeVerdict(winning_candidate_index=0, reasoning="matches the document")
        fake_model, ainvoke_mock = _fake_structured_model(verdict)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await _judge_one_document_extraction(
                _TEXT_DOCUMENT, _EXTRACTED, _SCHEMA, api_key="k", judge_model_id="deepseek/deepseek-chat-v3-0324"
            )
        assert result is True
        assert ainvoke_mock.await_count == 1

    async def test_rejected_verdict_returns_false(self):
        verdict = _JudgeVerdict(winning_candidate_index=None, reasoning="affected_count is not stated anywhere")
        fake_model, _ = _fake_structured_model(verdict)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await _judge_one_document_extraction(
                _TEXT_DOCUMENT, _EXTRACTED, _SCHEMA, api_key="k", judge_model_id="deepseek/deepseek-chat-v3-0324"
            )
        assert result is False

    async def test_total_judge_failure_returns_false_not_a_crash(self):
        """An unconfirmable record is treated the same as a rejected one -- never
        silently kept just because the judge itself couldn't be reached."""
        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            result = await _judge_one_document_extraction(
                _TEXT_DOCUMENT, _EXTRACTED, _SCHEMA, api_key="k", judge_model_id="deepseek/deepseek-chat-v3-0324"
            )
        assert result is False

    async def test_text_document_uses_the_given_judge_model_no_provider_override(self):
        verdict = _JudgeVerdict(winning_candidate_index=0, reasoning="ok")
        fake_model, _ = _fake_structured_model(verdict)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model) as create_model:
            await _judge_one_document_extraction(
                _TEXT_DOCUMENT, _EXTRACTED, _SCHEMA, api_key="k", judge_model_id="deepseek/deepseek-chat-v3-0324"
            )
        assert create_model.call_args.args[0] == "deepseek/deepseek-chat-v3-0324"
        assert create_model.call_args.kwargs["provider"] is None

    async def test_scanned_document_uses_the_vision_model_and_provider_ignoring_judge_model_id(self):
        verdict = _JudgeVerdict(winning_candidate_index=0, reasoning="matches the image")
        fake_model, _ = _fake_structured_model(verdict)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model) as create_model:
            result = await _judge_one_document_extraction(
                _VISION_DOCUMENT, _EXTRACTED, _SCHEMA, api_key="k", judge_model_id="this-should-be-ignored"
            )
        assert result is True
        assert create_model.call_args.args[0] == "anthropic/claude-sonnet-5"
        assert create_model.call_args.kwargs["provider"] == "openrouter"

    async def test_scanned_document_judge_prompt_includes_the_images(self):
        verdict = _JudgeVerdict(winning_candidate_index=0, reasoning="ok")
        fake_model, ainvoke_mock = _fake_structured_model(verdict)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            await _judge_one_document_extraction(
                _VISION_DOCUMENT, _EXTRACTED, _SCHEMA, api_key="k", judge_model_id="deepseek/deepseek-chat-v3-0324"
            )
        [call] = ainvoke_mock.await_args_list
        [message] = call.args[0]
        image_blocks = [block for block in message.content if block.get("type") == "image_url"]
        assert len(image_blocks) == 1


# ===========================================================================
# _judge_multi_row_extraction -- multi_row_vision's own grounding check (scrape-task-07)
# ===========================================================================

_MULTI_ROW_RECORDS = [
    {"employer": "Acme Corp", "affected_count": 42},
    {"employer": "Beta LLC", "affected_count": 7},
]


class TestJudgeMultiRowExtraction:
    async def test_confirmed_indices_are_returned_as_a_set(self):
        verdict = _MultiRowJudgeVerdict(confirmed_record_indices=[0, 1], reasoning="both rows match")
        fake_model, ainvoke_mock = _fake_structured_model(verdict)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await _judge_multi_row_extraction([b"fake-png-page-0"], _MULTI_ROW_RECORDS, _SCHEMA, api_key="k")
        assert result == {0, 1}
        assert ainvoke_mock.await_count == 1

    async def test_a_subset_of_records_confirmed(self):
        verdict = _MultiRowJudgeVerdict(confirmed_record_indices=[0], reasoning="row 1 bled into row 0's count")
        fake_model, _ = _fake_structured_model(verdict)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await _judge_multi_row_extraction([b"fake-png-page-0"], _MULTI_ROW_RECORDS, _SCHEMA, api_key="k")
        assert result == {0}

    async def test_out_of_range_indices_are_filtered_out_fail_closed(self):
        """A hallucinated index (the judge names a record that doesn't exist) must
        never crash the caller or silently pass through as a confirmed record."""
        verdict = _MultiRowJudgeVerdict(confirmed_record_indices=[0, 99], reasoning="oops")
        fake_model, _ = _fake_structured_model(verdict)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await _judge_multi_row_extraction([b"fake-png-page-0"], _MULTI_ROW_RECORDS, _SCHEMA, api_key="k")
        assert result == {0}

    async def test_total_judge_failure_returns_empty_set_not_a_crash(self):
        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            result = await _judge_multi_row_extraction([b"fake-png-page-0"], _MULTI_ROW_RECORDS, _SCHEMA, api_key="k")
        assert result == set()

    async def test_empty_records_returns_empty_set_without_calling_the_model(self):
        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            result = await _judge_multi_row_extraction([b"fake-png"], [], _SCHEMA, api_key="k")
        assert result == set()
        create_model.assert_not_called()

    async def test_always_uses_the_vision_model_and_provider(self):
        verdict = _MultiRowJudgeVerdict(confirmed_record_indices=[0], reasoning="ok")
        fake_model, _ = _fake_structured_model(verdict)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model) as create_model:
            await _judge_multi_row_extraction([b"fake-png-page-0"], _MULTI_ROW_RECORDS, _SCHEMA, api_key="k")
        assert create_model.call_args.args[0] == "anthropic/claude-sonnet-5"
        assert create_model.call_args.kwargs["provider"] == "openrouter"

    async def test_judge_prompt_includes_every_image_in_one_call(self):
        verdict = _MultiRowJudgeVerdict(confirmed_record_indices=[0, 1], reasoning="ok")
        fake_model, ainvoke_mock = _fake_structured_model(verdict)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            await _judge_multi_row_extraction([b"page-0", b"page-1"], _MULTI_ROW_RECORDS, _SCHEMA, api_key="k")
        assert ainvoke_mock.await_count == 1
        [call] = ainvoke_mock.await_args_list
        [message] = call.args[0]
        image_blocks = [block for block in message.content if block.get("type") == "image_url"]
        assert len(image_blocks) == 2
