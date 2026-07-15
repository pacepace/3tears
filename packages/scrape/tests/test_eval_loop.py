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

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from threetears.models import LlmPurpose

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
