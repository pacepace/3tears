"""Eval loop: LLM-judge candidate comparison + recipe persistence/reuse.

Orchestrates ``extraction.py``'s candidate generation and structural
validation into the self-healing cycle the product brief describes: a
healthy target's existing ``ScrapeRecipe`` is reused fetch after fetch with
no LLM call at all (just re-executing its stored selectors); only when
``consecutive_validation_failures`` crosses a threshold does candidate
generation re-run, survivors get compared by an LLM judge against the real
page content, and the winner is persisted as the new recipe.

Zero faidh imports (see ``scrape/__init__.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel
from pydantic import Field as PydanticField
from threetears.models import LlmPurpose
from threetears.observe import get_logger

from .collections import ScrapeExtraction, ScrapeExtractionCollection, ScrapeRecipe, ScrapeRecipeCollection
from .extraction import (
    DEFAULT_EXTRACTION_MODEL_ID,
    MAX_HTML_CHARS_IN_PROMPT,
    FieldSchema,
    generate_candidates,
    generate_regex_candidates,
    generate_regex_row_candidates,
    generate_row_candidates,
    html_to_text,
    strip_boilerplate,
    validate_candidate,
    validate_regex_candidate,
    validate_regex_row_candidate,
    validate_row_candidate,
)
from .llm_retry import bounded_retry_structured_call

__all__ = ["DEFAULT_JUDGE_MODEL_ID", "StrategyType", "run_eval_loop", "run_eval_loop_multi_row"]

#: Which extraction-strategy shape a target's page needs -- "css" (an HTML
#: table, the original v1 shape) or "regex" (a text-block/prose listing with
#: no table structure at all, added 2026-07-14). A per-call flag mirroring
#: how ``multi_row`` already works, not read from the stored recipe -- a
#: target's own page shape doesn't change between calls, so the caller
#: (``ScrapeTarget.extraction_strategy_type``) is the source of truth.
StrategyType = Literal["css", "regex"]

log = get_logger(__name__)

# Same reliability posture as extraction.py / query_agent/matching.py.
DEFAULT_JUDGE_MODEL_ID = "deepseek/deepseek-chat-v3-0324"

_JUDGE_TIMEOUT_SECONDS = 30
_JUDGE_ATTEMPTS = 6
_JUDGE_BACKOFF_SECONDS = 2.0

#: Default consecutive-failure threshold before a target's recipe is
#: abandoned and candidate generation re-runs. No artifact specifies a
#: concrete number (Requirements Confidence flagged this as
#: build-time-discovered); 3 mirrors the tolerance-for-transient-failure
#: shape the product brief describes ("self-healing... not AI on every
#: page") without letting a target silently stay broken for long.
DEFAULT_FAILURE_THRESHOLD = 3

#: Default candidate count per (re)generation round.
DEFAULT_CANDIDATE_COUNT = 3


class _JudgeVerdict(BaseModel):
    """Forced response shape for the candidate-comparison LLM call."""

    winning_candidate_index: int | None = PydanticField(
        default=None,
        description="0-based index into the candidate list of the best extraction, or null if none look correct",
    )
    reasoning: str = PydanticField(
        description="one-sentence justification citing what in the page content confirms or refutes each candidate"
    )
    field_confidences: dict[str, str] = PydanticField(
        default_factory=dict,
        description="per-field confidence note on the WINNING candidate only ('confident' | 'uncertain'), keyed by field name; empty if no winner",
    )


def _build_judge_prompt(html: str, survivors: list[dict[str, Any]], schema: FieldSchema) -> str:
    truncated = strip_boilerplate(html)[:MAX_HTML_CHARS_IN_PROMPT]
    candidate_lines = "\n".join(f"[{i}] {values}" for i, values in enumerate(survivors))
    field_lines = ", ".join(schema.keys())
    return (
        f"You are judging which of several structurally-valid extraction candidates actually matches "
        f"the real content of a web page. Fields being extracted: {field_lines}.\n\n"
        f"Page HTML (may be truncated):\n{truncated}\n\n"
        f"Candidate extracted values (index: field->value):\n{candidate_lines}\n\n"
        f"Compare each candidate's values against what the page content actually says. Pick the single "
        f"candidate whose values are correct, or null if none of them are. Structural validity (the "
        f"selectors matched something and the types parsed) has already been checked -- your job is "
        f"semantic correctness against the real page content."
    )


async def _judge_candidates(
    html: str,
    survivors: list[dict[str, Any]],
    schema: FieldSchema,
    *,
    model_id: str,
    api_key: str,
    attempts: int = _JUDGE_ATTEMPTS,
    backoff_seconds: float = _JUDGE_BACKOFF_SECONDS,
) -> _JudgeVerdict | None:
    """Structured-output judge call, retried on transient failure.

    Same bounded-retry shape as ``extraction.generate_candidates`` /
    ``query_agent/matching.py``'s ``_invoke_match_disambiguation``. Never
    raises; returns ``None`` only after every attempt fails.
    """
    prompt = _build_judge_prompt(html, survivors, schema)
    return await bounded_retry_structured_call(
        prompt,
        _JudgeVerdict,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.UTILITY,
        temperature=0.0,
        timeout=_JUDGE_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape judge",
        degraded_to="no winner",
    )


async def _persist_extraction(
    extraction_collection: ScrapeExtractionCollection,
    *,
    target_id: str,
    source_url: str,
    structured_fields: dict[str, Any],
    validation_status: str,
    extraction_recipe_id: str | None,
    field_confidences: dict[str, Any] | None = None,
) -> ScrapeExtraction:
    entity = extraction_collection.create(
        {
            "target_id": target_id,
            "source_url": source_url,
            "retrieved_at": datetime.now(UTC),
            "structured_fields": structured_fields,
            "field_confidences": field_confidences,
            "extraction_recipe_id": extraction_recipe_id,
            "validation_status": validation_status,
        }
    )
    await extraction_collection.save_entity(entity)
    return entity


async def _save_recipe(
    recipe_collection: ScrapeRecipeCollection,
    *,
    target_id: str,
    extraction_strategy: dict[str, Any],
    won_at: datetime,
    last_validated_at: datetime,
    consecutive_validation_failures: int,
) -> None:
    recipe_entity = recipe_collection.create(
        {
            "target_id": target_id,
            "extraction_strategy": extraction_strategy,
            "won_at": won_at,
            "last_validated_at": last_validated_at,
            "consecutive_validation_failures": consecutive_validation_failures,
        }
    )
    await recipe_collection.save_entity(recipe_entity)


async def _reuse_recipe(
    existing_recipe: ScrapeRecipe,
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
) -> ScrapeExtraction:
    """Validate *existing_recipe* against a freshly fetched page; keep it either way.

    Below the failure threshold the recipe is never abandoned on a single
    miss (transient-failure tolerance) -- only the failure counter moves.
    """
    strategy = existing_recipe.extraction_strategy.get("selectors", {})
    validation = validate_candidate(html, strategy, schema)
    now = datetime.now(UTC)
    if validation.valid:
        await _save_recipe(
            recipe_collection,
            target_id=target_id,
            extraction_strategy=existing_recipe.extraction_strategy,
            won_at=existing_recipe.won_at or now,
            last_validated_at=now,
            consecutive_validation_failures=0,
        )
        validation_status = "validated"
    else:
        await _save_recipe(
            recipe_collection,
            target_id=target_id,
            extraction_strategy=existing_recipe.extraction_strategy,
            won_at=existing_recipe.won_at or now,
            last_validated_at=existing_recipe.last_validated_at or now,
            consecutive_validation_failures=existing_recipe.consecutive_validation_failures + 1,
        )
        validation_status = "failed"
    return await _persist_extraction(
        extraction_collection,
        target_id=target_id,
        source_url=source_url,
        structured_fields={"records": [validation.extracted]},
        validation_status=validation_status,
        extraction_recipe_id=target_id,
    )


async def _regenerate_recipe(
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    api_key: str,
    candidate_count: int,
    extraction_model_id: str,
    judge_model_id: str,
) -> ScrapeExtraction:
    """No healthy recipe exists: generate fresh candidates and consult the LLM judge."""
    candidates = await generate_candidates(
        html, schema, n=candidate_count, model_id=extraction_model_id, api_key=api_key
    )
    validations = [validate_candidate(html, candidate, schema) for candidate in candidates]
    survivors = [
        (candidate, validation)
        for candidate, validation in zip(candidates, validations, strict=True)
        if validation.valid
    ]

    if not survivors:
        log.warning(
            "scrape eval loop: no structurally-valid candidates for target %s (%d proposed)",
            target_id,
            len(candidates),
            extra={"extra_data": {"target_id": target_id}},
        )
        result = await _persist_extraction(
            extraction_collection,
            target_id=target_id,
            source_url=source_url,
            structured_fields={"records": []},
            validation_status="failed",
            extraction_recipe_id=None,
        )
    else:
        verdict = await _judge_candidates(
            html,
            [validation.extracted for _, validation in survivors],
            schema,
            model_id=judge_model_id,
            api_key=api_key,
        )
        if (
            verdict is None
            or verdict.winning_candidate_index is None
            or not (0 <= verdict.winning_candidate_index < len(survivors))
        ):
            # Structurally sound candidates exist, but the judge couldn't confirm any of
            # them (or failed outright) -- an honest needs_review, not a crash, and not a
            # silently-crowned recipe. Surface the best-scoring survivor's data for human
            # review rather than nothing at all.
            _, best_validation = survivors[0]
            result = await _persist_extraction(
                extraction_collection,
                target_id=target_id,
                source_url=source_url,
                structured_fields={"records": [best_validation.extracted]},
                validation_status="needs_review",
                extraction_recipe_id=None,
            )
        else:
            winning_strategy, winning_validation = survivors[verdict.winning_candidate_index]
            now = datetime.now(UTC)
            await _save_recipe(
                recipe_collection,
                target_id=target_id,
                extraction_strategy={"selectors": winning_strategy},
                won_at=now,
                last_validated_at=now,
                consecutive_validation_failures=0,
            )
            result = await _persist_extraction(
                extraction_collection,
                target_id=target_id,
                source_url=source_url,
                structured_fields={"records": [winning_validation.extracted]},
                validation_status="validated",
                extraction_recipe_id=target_id,
                field_confidences=verdict.field_confidences,
            )
    return result


async def _reuse_regex_recipe(
    existing_recipe: ScrapeRecipe,
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
) -> ScrapeExtraction:
    """Regex counterpart to :func:`_reuse_recipe` -- text-block strategy shape.

    Same transient-failure tolerance: below the failure threshold the
    recipe is never abandoned on a single miss.
    """
    pattern = existing_recipe.extraction_strategy.get("pattern", "")
    text = html_to_text(html)
    validation = validate_regex_candidate(text, pattern, schema)
    now = datetime.now(UTC)
    if validation.valid:
        await _save_recipe(
            recipe_collection,
            target_id=target_id,
            extraction_strategy=existing_recipe.extraction_strategy,
            won_at=existing_recipe.won_at or now,
            last_validated_at=now,
            consecutive_validation_failures=0,
        )
        validation_status = "validated"
    else:
        await _save_recipe(
            recipe_collection,
            target_id=target_id,
            extraction_strategy=existing_recipe.extraction_strategy,
            won_at=existing_recipe.won_at or now,
            last_validated_at=existing_recipe.last_validated_at or now,
            consecutive_validation_failures=existing_recipe.consecutive_validation_failures + 1,
        )
        validation_status = "failed"
    return await _persist_extraction(
        extraction_collection,
        target_id=target_id,
        source_url=source_url,
        structured_fields={"records": [validation.extracted]},
        validation_status=validation_status,
        extraction_recipe_id=target_id,
    )


async def _regenerate_regex_recipe(
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    api_key: str,
    candidate_count: int,
    extraction_model_id: str,
    judge_model_id: str,
) -> ScrapeExtraction:
    """Regex counterpart to :func:`_regenerate_recipe`.

    The judge step is shared, unmodified -- :func:`_judge_candidates` only
    ever sees each candidate's *extracted values* and the real page HTML,
    never the mechanism (CSS selector or regex pattern) that produced them,
    so semantic comparison works identically regardless of strategy type.
    """
    text = html_to_text(html)
    candidates = await generate_regex_candidates(
        text, schema, n=candidate_count, model_id=extraction_model_id, api_key=api_key
    )
    validations = [validate_regex_candidate(text, candidate, schema) for candidate in candidates]
    survivors = [
        (candidate, validation)
        for candidate, validation in zip(candidates, validations, strict=True)
        if validation.valid
    ]

    if not survivors:
        log.warning(
            "scrape regex eval loop: no structurally-valid candidates for target %s (%d proposed)",
            target_id,
            len(candidates),
            extra={"extra_data": {"target_id": target_id}},
        )
        result = await _persist_extraction(
            extraction_collection,
            target_id=target_id,
            source_url=source_url,
            structured_fields={"records": []},
            validation_status="failed",
            extraction_recipe_id=None,
        )
    else:
        verdict = await _judge_candidates(
            html,
            [validation.extracted for _, validation in survivors],
            schema,
            model_id=judge_model_id,
            api_key=api_key,
        )
        if (
            verdict is None
            or verdict.winning_candidate_index is None
            or not (0 <= verdict.winning_candidate_index < len(survivors))
        ):
            _, best_validation = survivors[0]
            result = await _persist_extraction(
                extraction_collection,
                target_id=target_id,
                source_url=source_url,
                structured_fields={"records": [best_validation.extracted]},
                validation_status="needs_review",
                extraction_recipe_id=None,
            )
        else:
            winning_pattern, winning_validation = survivors[verdict.winning_candidate_index]
            now = datetime.now(UTC)
            await _save_recipe(
                recipe_collection,
                target_id=target_id,
                extraction_strategy={"pattern": winning_pattern},
                won_at=now,
                last_validated_at=now,
                consecutive_validation_failures=0,
            )
            result = await _persist_extraction(
                extraction_collection,
                target_id=target_id,
                source_url=source_url,
                structured_fields={"records": [winning_validation.extracted]},
                validation_status="validated",
                extraction_recipe_id=target_id,
                field_confidences=verdict.field_confidences,
            )
    return result


async def run_eval_loop(
    target_id: str,
    html: str,
    source_url: str,
    schema: FieldSchema,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    api_key: str,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    extraction_model_id: str = DEFAULT_EXTRACTION_MODEL_ID,
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
    strategy_type: StrategyType = "css",
) -> ScrapeExtraction:
    """Run one fetch through the eval loop and persist a ``ScrapeExtraction`` row.

    Reuses *target_id*'s existing recipe (no LLM call) while it's healthy;
    once ``consecutive_validation_failures`` crosses *failure_threshold*,
    regenerates candidates and consults the LLM judge for a new winner.

    :param target_id: the target this fetch belongs to
    :ptype target_id: str
    :param html: the freshly rendered page's full HTML
    :ptype html: str
    :param source_url: the final URL actually fetched (post-redirect)
    :ptype source_url: str
    :param schema: field_name -> expected Python type (caller-supplied; the
        core never hardcodes domain field meanings)
    :ptype schema: FieldSchema
    :param recipe_collection: this target's recipe store
    :ptype recipe_collection: ScrapeRecipeCollection
    :param extraction_collection: where the resulting row is persisted
    :ptype extraction_collection: ScrapeExtractionCollection
    :param api_key: OpenRouter API key for both the candidate-generation and judge calls
    :ptype api_key: str
    :param candidate_count: how many candidates to request on a (re)generation round
    :ptype candidate_count: int
    :param failure_threshold: consecutive structural-validation failures before regenerating
    :ptype failure_threshold: int
    :param extraction_model_id: model for candidate generation
    :ptype extraction_model_id: str
    :param judge_model_id: model for the candidate-comparison judge
    :ptype judge_model_id: str
    :param strategy_type: ``"css"`` (an HTML table, CSS-selector candidates)
        or ``"regex"`` (a text-block/prose listing, regex-pattern
        candidates against the page's plain text) -- a per-target config
        value, the page's own shape, not something the eval loop infers
    :ptype strategy_type: StrategyType
    :return: the persisted ``ScrapeExtraction`` row (``structured_fields["records"]``
        holds a single-element list -- the same shape :func:`run_eval_loop_multi_row`
        uses, just always exactly one record)
    :rtype: ScrapeExtraction
    """
    reuse_fn = _reuse_regex_recipe if strategy_type == "regex" else _reuse_recipe
    regenerate_fn = _regenerate_regex_recipe if strategy_type == "regex" else _regenerate_recipe
    existing_recipe = await recipe_collection.get(target_id)
    if existing_recipe is not None and existing_recipe.consecutive_validation_failures < failure_threshold:
        result = await reuse_fn(
            existing_recipe,
            html,
            schema,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
        )
    else:
        result = await regenerate_fn(
            html,
            schema,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            api_key=api_key,
            candidate_count=candidate_count,
            extraction_model_id=extraction_model_id,
            judge_model_id=judge_model_id,
        )
    return result


# Row-count-sampled per candidate to keep the judge prompt bounded -- passing all
# rows for every candidate (e.g. 80 rows x 3 candidates) would blow the token budget
# for no real benefit; a handful of sample rows is enough to judge selector quality.
_MAX_SAMPLE_ROWS_IN_JUDGE_PROMPT = 5


def _build_row_judge_prompt(html: str, survivors: list[list[dict[str, Any]]], schema: FieldSchema) -> str:
    truncated = strip_boilerplate(html)[:MAX_HTML_CHARS_IN_PROMPT]
    field_lines = ", ".join(schema.keys())
    candidate_lines = "\n".join(
        f"[{i}] {len(rows)} rows total, first {min(len(rows), _MAX_SAMPLE_ROWS_IN_JUDGE_PROMPT)} shown: "
        f"{rows[:_MAX_SAMPLE_ROWS_IN_JUDGE_PROMPT]}"
        for i, rows in enumerate(survivors)
    )
    return (
        f"You are judging which of several structurally-valid row-extraction candidates actually "
        f"matches the real content of a web page that lists MANY repeating records. Fields being "
        f"extracted per record: {field_lines}.\n\n"
        f"Page HTML (may be truncated):\n{truncated}\n\n"
        f"Candidate extracted rows (index: row count and a sample):\n{candidate_lines}\n\n"
        f"Compare each candidate's sampled rows against what the page content actually says. Pick the "
        f"single candidate whose values are correct, or null if none of them are. Structural validity "
        f"(the selectors matched something and the types parsed) has already been checked -- your job "
        f"is semantic correctness against the real page content, and picking the candidate that captures "
        f"the MOST real records correctly, not just the one with the most plausible-looking sample."
    )


async def _judge_row_candidates(
    html: str,
    survivors: list[list[dict[str, Any]]],
    schema: FieldSchema,
    *,
    model_id: str,
    api_key: str,
    attempts: int = _JUDGE_ATTEMPTS,
    backoff_seconds: float = _JUDGE_BACKOFF_SECONDS,
) -> _JudgeVerdict | None:
    """Structured-output judge call for row-set candidates, retried on transient failure.

    Shares :func:`_judge_candidates`'s retry/logging shape via
    :func:`bounded_retry_structured_call` (backlog SCR-K7M3, closed
    2026-07-14 -- see build-plan.md's Chunk 07 design decision for the
    original "duplicated, not shared" call and this chunk for why it
    changed). Never raises; returns ``None`` only after every attempt fails.
    """
    prompt = _build_row_judge_prompt(html, survivors, schema)
    return await bounded_retry_structured_call(
        prompt,
        _JudgeVerdict,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.UTILITY,
        temperature=0.0,
        timeout=_JUDGE_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape row judge",
        degraded_to="no winner",
    )


async def _reuse_row_recipe(
    existing_recipe: ScrapeRecipe,
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
) -> ScrapeExtraction:
    """Validate *existing_recipe* against a freshly fetched page; keep it either way.

    Same transient-failure tolerance as :func:`_reuse_recipe`: below the
    failure threshold the recipe is never abandoned on a single miss.
    """
    validation = validate_row_candidate(html, existing_recipe.extraction_strategy, schema)
    now = datetime.now(UTC)
    if validation.valid:
        await _save_recipe(
            recipe_collection,
            target_id=target_id,
            extraction_strategy=existing_recipe.extraction_strategy,
            won_at=existing_recipe.won_at or now,
            last_validated_at=now,
            consecutive_validation_failures=0,
        )
        validation_status = "validated"
    else:
        await _save_recipe(
            recipe_collection,
            target_id=target_id,
            extraction_strategy=existing_recipe.extraction_strategy,
            won_at=existing_recipe.won_at or now,
            last_validated_at=existing_recipe.last_validated_at or now,
            consecutive_validation_failures=existing_recipe.consecutive_validation_failures + 1,
        )
        validation_status = "failed"
    log.info(
        "scrape row recipe reuse: target=%s records_captured=%d rows_matched=%d",
        target_id,
        len(validation.records),
        validation.total_rows_matched,
        extra={"extra_data": {"target_id": target_id}},
    )
    return await _persist_extraction(
        extraction_collection,
        target_id=target_id,
        source_url=source_url,
        structured_fields={"records": validation.records},
        validation_status=validation_status,
        extraction_recipe_id=target_id,
    )


async def _regenerate_row_recipe(
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    api_key: str,
    candidate_count: int,
    extraction_model_id: str,
    judge_model_id: str,
) -> ScrapeExtraction:
    """No healthy recipe exists: generate fresh row candidates and consult the LLM judge."""
    candidates = await generate_row_candidates(
        html, schema, n=candidate_count, model_id=extraction_model_id, api_key=api_key
    )
    validations = [validate_row_candidate(html, candidate, schema) for candidate in candidates]
    survivors = [
        (candidate, validation)
        for candidate, validation in zip(candidates, validations, strict=True)
        if validation.valid
    ]

    if not survivors:
        log.warning(
            "scrape row eval loop: no structurally-valid row candidates for target %s (%d proposed)",
            target_id,
            len(candidates),
            extra={"extra_data": {"target_id": target_id}},
        )
        result = await _persist_extraction(
            extraction_collection,
            target_id=target_id,
            source_url=source_url,
            structured_fields={"records": []},
            validation_status="failed",
            extraction_recipe_id=None,
        )
    else:
        verdict = await _judge_row_candidates(
            html,
            [validation.records for _, validation in survivors],
            schema,
            model_id=judge_model_id,
            api_key=api_key,
        )
        if (
            verdict is None
            or verdict.winning_candidate_index is None
            or not (0 <= verdict.winning_candidate_index < len(survivors))
        ):
            # Structurally sound candidates exist, but the judge couldn't confirm any of them
            # (or failed outright) -- an honest needs_review, not a crash, and not a silently-
            # crowned recipe. Surface the candidate that captured the most rows for human review
            # rather than nothing at all -- unlike the single-record path, "best" has a real,
            # comparable signal here (row count), not just "first proposed."
            _, best_validation = max(survivors, key=lambda pair: len(pair[1].records))
            result = await _persist_extraction(
                extraction_collection,
                target_id=target_id,
                source_url=source_url,
                structured_fields={"records": best_validation.records},
                validation_status="needs_review",
                extraction_recipe_id=None,
            )
        else:
            winning_strategy, winning_validation = survivors[verdict.winning_candidate_index]
            now = datetime.now(UTC)
            await _save_recipe(
                recipe_collection,
                target_id=target_id,
                extraction_strategy=winning_strategy,
                won_at=now,
                last_validated_at=now,
                consecutive_validation_failures=0,
            )
            result = await _persist_extraction(
                extraction_collection,
                target_id=target_id,
                source_url=source_url,
                structured_fields={"records": winning_validation.records},
                validation_status="validated",
                extraction_recipe_id=target_id,
                field_confidences=verdict.field_confidences,
            )
    return result


async def _reuse_regex_row_recipe(
    existing_recipe: ScrapeRecipe,
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
) -> ScrapeExtraction:
    """Regex counterpart to :func:`_reuse_row_recipe` -- text-block strategy shape.

    Same transient-failure tolerance: below the failure threshold the
    recipe is never abandoned on a single miss.
    """
    pattern = existing_recipe.extraction_strategy.get("pattern", "")
    text = html_to_text(html)
    validation = validate_regex_row_candidate(text, pattern, schema)
    now = datetime.now(UTC)
    if validation.valid:
        await _save_recipe(
            recipe_collection,
            target_id=target_id,
            extraction_strategy=existing_recipe.extraction_strategy,
            won_at=existing_recipe.won_at or now,
            last_validated_at=now,
            consecutive_validation_failures=0,
        )
        validation_status = "validated"
    else:
        await _save_recipe(
            recipe_collection,
            target_id=target_id,
            extraction_strategy=existing_recipe.extraction_strategy,
            won_at=existing_recipe.won_at or now,
            last_validated_at=existing_recipe.last_validated_at or now,
            consecutive_validation_failures=existing_recipe.consecutive_validation_failures + 1,
        )
        validation_status = "failed"
    log.info(
        "scrape regex row recipe reuse: target=%s records_captured=%d matches=%d",
        target_id,
        len(validation.records),
        validation.total_rows_matched,
        extra={"extra_data": {"target_id": target_id}},
    )
    return await _persist_extraction(
        extraction_collection,
        target_id=target_id,
        source_url=source_url,
        structured_fields={"records": validation.records},
        validation_status=validation_status,
        extraction_recipe_id=target_id,
    )


async def _regenerate_regex_row_recipe(
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    api_key: str,
    candidate_count: int,
    extraction_model_id: str,
    judge_model_id: str,
) -> ScrapeExtraction:
    """Regex counterpart to :func:`_regenerate_row_recipe`.

    The judge step is shared, unmodified -- see :func:`_regenerate_regex_recipe`'s docstring.
    """
    text = html_to_text(html)
    candidates = await generate_regex_row_candidates(
        text, schema, n=candidate_count, model_id=extraction_model_id, api_key=api_key
    )
    validations = [validate_regex_row_candidate(text, candidate, schema) for candidate in candidates]
    survivors = [
        (candidate, validation)
        for candidate, validation in zip(candidates, validations, strict=True)
        if validation.valid
    ]

    if not survivors:
        log.warning(
            "scrape regex row eval loop: no structurally-valid row candidates for target %s (%d proposed)",
            target_id,
            len(candidates),
            extra={"extra_data": {"target_id": target_id}},
        )
        result = await _persist_extraction(
            extraction_collection,
            target_id=target_id,
            source_url=source_url,
            structured_fields={"records": []},
            validation_status="failed",
            extraction_recipe_id=None,
        )
    else:
        verdict = await _judge_row_candidates(
            html,
            [validation.records for _, validation in survivors],
            schema,
            model_id=judge_model_id,
            api_key=api_key,
        )
        if (
            verdict is None
            or verdict.winning_candidate_index is None
            or not (0 <= verdict.winning_candidate_index < len(survivors))
        ):
            _, best_validation = max(survivors, key=lambda pair: len(pair[1].records))
            result = await _persist_extraction(
                extraction_collection,
                target_id=target_id,
                source_url=source_url,
                structured_fields={"records": best_validation.records},
                validation_status="needs_review",
                extraction_recipe_id=None,
            )
        else:
            winning_pattern, winning_validation = survivors[verdict.winning_candidate_index]
            now = datetime.now(UTC)
            await _save_recipe(
                recipe_collection,
                target_id=target_id,
                extraction_strategy={"pattern": winning_pattern},
                won_at=now,
                last_validated_at=now,
                consecutive_validation_failures=0,
            )
            result = await _persist_extraction(
                extraction_collection,
                target_id=target_id,
                source_url=source_url,
                structured_fields={"records": winning_validation.records},
                validation_status="validated",
                extraction_recipe_id=target_id,
                field_confidences=verdict.field_confidences,
            )
    return result


async def run_eval_loop_multi_row(
    target_id: str,
    html: str,
    source_url: str,
    schema: FieldSchema,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    api_key: str,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    extraction_model_id: str = DEFAULT_EXTRACTION_MODEL_ID,
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
    strategy_type: StrategyType = "css",
) -> ScrapeExtraction:
    """Run one fetch through the multi-row eval loop and persist a ``ScrapeExtraction`` row.

    The multi-row counterpart to :func:`run_eval_loop` -- extracts every matching
    record on the page (``structured_fields={"records": [...]}``), not a single set
    of values. Reuses *target_id*'s existing recipe (no LLM call) while it's
    healthy; once ``consecutive_validation_failures`` crosses *failure_threshold*,
    regenerates candidates and consults the LLM judge for a new winner -- same
    cadence as :func:`run_eval_loop`, just row-shaped throughout.

    :param target_id: the target this fetch belongs to
    :ptype target_id: str
    :param html: the freshly rendered page's full HTML
    :ptype html: str
    :param source_url: the final URL actually fetched (post-redirect)
    :ptype source_url: str
    :param schema: field_name -> expected Python type, applied to every row
    :ptype schema: FieldSchema
    :param recipe_collection: this target's recipe store
    :ptype recipe_collection: ScrapeRecipeCollection
    :param extraction_collection: where the resulting row is persisted
    :ptype extraction_collection: ScrapeExtractionCollection
    :param api_key: OpenRouter API key for both the candidate-generation and judge calls
    :ptype api_key: str
    :param candidate_count: how many candidates to request on a (re)generation round
    :ptype candidate_count: int
    :param failure_threshold: consecutive structural-validation failures before regenerating
    :ptype failure_threshold: int
    :param extraction_model_id: model for candidate generation
    :ptype extraction_model_id: str
    :param judge_model_id: model for the candidate-comparison judge
    :ptype judge_model_id: str
    :param strategy_type: ``"css"`` (row/field CSS selectors) or ``"regex"``
        (a single pattern matched repeatedly via ``re.finditer`` against the
        page's plain text, one match per record) -- see :func:`run_eval_loop`
    :ptype strategy_type: StrategyType
    :return: the persisted ``ScrapeExtraction`` row (``structured_fields["records"]`` holds every record)
    :rtype: ScrapeExtraction
    """
    reuse_fn = _reuse_regex_row_recipe if strategy_type == "regex" else _reuse_row_recipe
    regenerate_fn = _regenerate_regex_row_recipe if strategy_type == "regex" else _regenerate_row_recipe
    existing_recipe = await recipe_collection.get(target_id)
    if existing_recipe is not None and existing_recipe.consecutive_validation_failures < failure_threshold:
        result = await reuse_fn(
            existing_recipe,
            html,
            schema,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
        )
    else:
        result = await regenerate_fn(
            html,
            schema,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            api_key=api_key,
            candidate_count=candidate_count,
            extraction_model_id=extraction_model_id,
            judge_model_id=judge_model_id,
        )
    return result
