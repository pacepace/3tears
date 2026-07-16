"""Eval loop: LLM-judge candidate comparison + recipe persistence/reuse.

Orchestrates ``extraction.py``'s candidate generation and structural
validation into the self-healing cycle the product brief describes: a
healthy target's existing ``ScrapeRecipe`` is reused fetch after fetch with
no LLM call at all (just re-executing its stored selectors); only when
``consecutive_validation_failures`` crosses a threshold does candidate
generation re-run, survivors get compared by an LLM judge against the real
page content, and the winner is persisted as the new recipe.

**Exception: ``StrategyType`` ``"per_document"``** (scrape-task-05,
2026-07-15) has no cached-recipe cycle at all -- some real multi-document
targets (independently-worded documents sharing no template, see
``drivers/multi_document.py``) genuinely cannot be served by a pattern
learned once and reused; every document gets its own fresh LLM extraction
call on every poll instead (:func:`_run_per_document_extraction`).

Zero faidh imports (see ``scrape/__init__.py``).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar

from pydantic import BaseModel
from pydantic import Field as PydanticField
from threetears.models import LlmPurpose
from threetears.observe import get_logger

from .collections import ScrapeExtraction, ScrapeExtractionCollection, ScrapeRecipe, ScrapeRecipeCollection
from .extraction import (
    DEFAULT_EXTRACTION_MODEL_ID,
    DEFAULT_VISION_MODEL_ID,
    MAX_HTML_CHARS_IN_PROMPT,
    FieldSchema,
    NoticeDocument,
    _VISION_PROVIDER,
    extract_fields_directly_chunked,
    extract_fields_from_images,
    extract_multi_row_fields_from_images,
    extract_page_images,
    generate_candidates,
    generate_regex_candidates,
    generate_regex_row_candidates,
    generate_row_candidates,
    html_to_text,
    split_notice_documents,
    strip_boilerplate,
    validate_candidate,
    validate_regex_candidate,
    validate_regex_row_candidate,
    validate_row_candidate,
)
from .llm_retry import bounded_retry_structured_call

__all__ = ["DEFAULT_JUDGE_MODEL_ID", "StrategyType", "run_eval_loop", "run_eval_loop_multi_row"]

#: Which extraction-strategy shape a target's page needs -- "css" (an HTML
#: table, the original v1 shape), "regex" (a text-block/prose listing with
#: no table structure at all, added 2026-07-14), "per_document" (a
#: MultiDocumentDriver-combined page whose documents are each independently
#: worded -- e.g. one employer's own freeform letter per notice -- sharing
#: no boilerplate any single cached pattern could generalize across; added
#: scrape-task-05, 2026-07-15, live-verified against West Virginia's real
#: WARN letters after a regex-strategy attempt matched only 1 of 10), or
#: "multi_row_vision" (a single born-digital PDF whose own table structure
#: defeats text-based table extraction -- e.g. Nevada's real master WARN
#: PDF, where ``find_tables()`` finds only the header one way and mis-splits
#: columns the other -- needs a vision read of the whole table at once;
#: added scrape-task-07, 2026-07-15, explicitly chosen per-target in config,
#: never auto-detected by shape: Mississippi's superficially similar
#: "multi-row PDF" needs the OPPOSITE fix, a plain row-merge on its own
#: already-working text-based extraction, proof shape alone doesn't decide
#: this). A per-call flag mirroring how ``multi_row`` already works, not
#: read from the stored recipe -- a target's own page shape doesn't change
#: between calls, so the caller (``ScrapeTarget.extraction_strategy_type``)
#: is the source of truth.
StrategyType = Literal["css", "regex", "per_document", "multi_row_vision"]

T = TypeVar("T", bound=BaseModel)

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

#: Hard outer deadline for ONE document's ``extract_fields_directly`` call inside
#: ``"per_document"`` StrategyType (:func:`_run_per_document_extraction`) -- live-
#: reproduced (scrape-task-05, a real West Virginia document): the underlying chat
#: client can hang well past its own configured per-attempt timeout with zero
#: further retry activity, so ``extract_fields_directly``'s own *timeout*/*attempts*
#: parameters alone are not a reliable bound. 90s covers every well-behaved case
#: seen live (a successful call takes single-digit seconds; a well-behaved retry
#: cycle through several failed attempts still lands well under a minute) with
#: margin, while still keeping one truly-hung document from blocking an entire
#: poll of N documents indefinitely.
_PER_DOCUMENT_TIMEOUT_SECONDS = 90

#: Same hang-mitigation posture as :data:`_PER_DOCUMENT_TIMEOUT_SECONDS`, but wider --
#: a multi-row vision extraction/judge call reasons over a whole table's worth of
#: records in one call (Nevada's real master WARN PDF: 17 records), not one document's,
#: so a well-behaved call legitimately takes longer.
_MULTI_ROW_EXTRACTION_TIMEOUT_SECONDS = 150


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


async def _judge(
    prompt_or_messages: str | list[Any],
    *,
    response_model: type[T],
    model_id: str,
    api_key: str,
    provider: str | None = None,
    attempts: int = _JUDGE_ATTEMPTS,
    backoff_seconds: float = _JUDGE_BACKOFF_SECONDS,
    log_label: str,
) -> T | None:
    """The one shared judge call every judge use in this module funnels through --
    structured-output response, retried on transient failure, never raises. Callers
    vary in how *prompt_or_messages* was built (a plain text prompt for css/regex
    candidate comparison against real page HTML; a multimodal message list for a
    vision-grounded per_document/multi_row confirmation against real page images --
    ``bounded_retry_structured_call``'s own ``prompt`` parameter already accepts
    either shape), *log_label*, and -- since scrape-task-07 -- *response_model*
    (:class:`_JudgeVerdict`'s single-winner shape for css/regex/per_document; the
    multi-row judge passes :class:`_MultiRowJudgeVerdict` instead, since "which ONE
    candidate wins" doesn't fit "which of these N independent records are each
    individually correct" -- required explicitly, not defaulted, so mypy can infer
    *T* precisely at each call site). Previously ``_judge_candidates`` and
    ``_judge_row_candidates`` each called ``bounded_retry_structured_call`` directly
    with near-identical arguments (scrape-task-06's own per-document grounding
    check made that duplication worth closing, not adding a third copy of it).
    """
    return await bounded_retry_structured_call(
        prompt_or_messages,
        response_model,
        model_id=model_id,
        api_key=api_key,
        provider=provider,
        purpose=LlmPurpose.UTILITY,
        temperature=0.0,
        timeout=_JUDGE_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label=log_label,
        degraded_to="no winner",
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
    """Structured-output judge call comparing several candidates, retried on transient failure.

    Same bounded-retry shape as ``extraction.generate_candidates`` /
    ``query_agent/matching.py``'s ``_invoke_match_disambiguation`` -- via the
    shared :func:`_judge`. Never raises; returns ``None`` only after every
    attempt fails.
    """
    prompt = _build_judge_prompt(html, survivors, schema)
    return await _judge(
        prompt,
        response_model=_JudgeVerdict,
        model_id=model_id,
        api_key=api_key,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape judge",
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

    Shares :func:`_judge_candidates`'s retry/logging shape via the shared
    :func:`_judge` (backlog SCR-K7M3, closed 2026-07-14 -- see build-plan.md's
    Chunk 07 design decision for the original "duplicated, not shared" call
    and this chunk for why it changed; scrape-task-06 closed the SAME
    duplication again between this function and ``_judge_candidates`` once a
    third judge use -- per-document grounding -- made it worth a shared
    primitive instead of a third copy). Never raises; returns ``None`` only
    after every attempt fails.
    """
    prompt = _build_row_judge_prompt(html, survivors, schema)
    return await _judge(
        prompt,
        response_model=_JudgeVerdict,
        model_id=model_id,
        api_key=api_key,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape row judge",
    )


def _build_per_document_judge_prompt(text: str, extracted: dict[str, Any], schema: FieldSchema) -> str:
    field_lines = ", ".join(schema.keys())
    truncated = text[:MAX_HTML_CHARS_IN_PROMPT]
    return (
        f"You are judging whether an already-extracted record actually matches the real content of "
        f"ONE independent document -- there is exactly one candidate (no page-wide pattern to compare "
        f"against others), confirm it or reject it. Fields: {field_lines}.\n\n"
        f"Document text (may be truncated):\n{truncated}\n\n"
        f"Extracted record: {extracted}\n\n"
        f"Compare the extracted record against what the document's own text actually says. If every "
        f"field's value is genuinely grounded in and correct per the document's own content, return "
        f"winning_candidate_index=0. If any field is wrong, hallucinated, or not actually supported by "
        f"the document's own text, return winning_candidate_index=null."
    )


def _build_per_document_vision_judge_content(
    images: list[bytes], extracted: dict[str, Any], schema: FieldSchema
) -> list[Any]:
    from threetears.models import format_vision_content

    field_lines = ", ".join(schema.keys())
    content: list[Any] = []
    for image_bytes in images:
        content.extend(format_vision_content(image_bytes, "image/png", "")[:-1])
    content.append(
        {
            "type": "text",
            "text": (
                "You are judging whether an already-extracted record actually matches what you see in "
                "this document's own page image(s) -- there is exactly one candidate (no page-wide "
                f"pattern to compare against others), confirm it or reject it. Fields: {field_lines}.\n\n"
                f"Extracted record: {extracted}\n\n"
                "Compare the extracted record against what the image(s) actually show. If every field's "
                "value is genuinely grounded in and correct per the document's own content, return "
                "winning_candidate_index=0. If any field is wrong, hallucinated, or not actually "
                "supported by what you see, return winning_candidate_index=null."
            ),
        }
    )
    return content


async def _judge_one_document_extraction(
    document: NoticeDocument,
    extracted: dict[str, Any],
    schema: FieldSchema,
    *,
    api_key: str,
    judge_model_id: str,
) -> bool:
    """Confirms (or rejects) ONE document's already-extracted record against its own real
    source content -- the ``"per_document"`` StrategyType's counterpart to
    :func:`_judge_candidates`/:func:`_judge_row_candidates`'s own "semantic correctness
    against the real page content" check (css/regex strategies get this once, on cold
    start, when candidates are first generated; per_document has no cached recipe to
    ever skip it, so every document, every poll, gets grounded the same way).

    Routes through the SAME shared :func:`_judge` used by every other judge call in this
    module -- for a scanned document (``document.was_ocr``), the grounding source is its
    own page images (a vision-capable model, mirroring
    :func:`~threetears.scrape.extraction.extract_fields_from_images`'s own model/provider
    choice, since the text-only judge model can't read images); for a born-digital
    document, the grounding source is its own plain text, judged by the regular
    *judge_model_id*.

    :param document: the document *extracted* came from (its own text/images/was_ocr)
    :ptype document: NoticeDocument
    :param extracted: the already-coerced, already-complete field values to confirm
    :ptype extracted: dict[str, Any]
    :param schema: field_name -> expected Python type
    :ptype schema: FieldSchema
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param judge_model_id: the text judge model (ignored for a scanned document, which
        always uses the vision model instead -- see above)
    :ptype judge_model_id: str
    :return: ``True`` only if the judge explicitly confirmed the record (``winning_
        candidate_index == 0``); ``False`` on rejection OR total judge failure -- an
        unconfirmable record is treated the same as a rejected one, never silently kept
    :rtype: bool
    """
    prompt_or_messages: str | list[Any]
    if document.was_ocr:
        from langchain_core.messages import HumanMessage

        content = _build_per_document_vision_judge_content(document.images, extracted, schema)
        prompt_or_messages = [HumanMessage(content=content)]
        model_id = DEFAULT_VISION_MODEL_ID
        provider = _VISION_PROVIDER
    else:
        prompt_or_messages = _build_per_document_judge_prompt(document.text, extracted, schema)
        model_id = judge_model_id
        provider = None
    verdict = await _judge(
        prompt_or_messages,
        response_model=_JudgeVerdict,
        model_id=model_id,
        api_key=api_key,
        provider=provider,
        log_label="scrape per-document judge",
    )
    return verdict is not None and verdict.winning_candidate_index == 0


class _MultiRowJudgeVerdict(BaseModel):
    """Forced response shape for the multi-row grounding-judge call.

    Deliberately NOT :class:`_JudgeVerdict` -- a multi-row table read isn't "pick the
    one best candidate among several," it's "independently confirm or reject EACH of
    these N already-extracted records against the same source image(s)." A single
    ``winning_candidate_index`` can't express "records 0, 2, and 5 are right but 1, 3,
    4 are wrong" -- this shape can.
    """

    confirmed_record_indices: list[int] = PydanticField(
        description=(
            "0-based indices, into the given records list, of every record that is fully "
            "and correctly grounded in the source image(s) -- every field matches what the "
            "image(s) actually show, no row bled into its neighbor, no column misaligned. "
            "Omit the index of any record with even one wrong, hallucinated, or misaligned "
            "field. Empty list if none of the records are fully correct."
        )
    )
    reasoning: str = PydanticField(
        description="one-sentence justification per rejected record citing what's wrong, or confirming all are correct"
    )


def _build_multi_row_judge_content(images: list[bytes], records: list[dict[str, Any]], schema: FieldSchema) -> list[Any]:
    from threetears.models import format_vision_content

    field_lines = ", ".join(schema.keys())
    record_lines = "\n".join(f"[{i}] {record}" for i, record in enumerate(records))
    content: list[Any] = []
    for image_bytes in images:
        content.extend(format_vision_content(image_bytes, "image/png", "")[:-1])
    content.append(
        {
            "type": "text",
            "text": (
                "You are judging a set of already-extracted records against a table shown in "
                "these page image(s). Each record was extracted from one row of the table. "
                f"Fields: {field_lines}.\n\n"
                f"Extracted records (index: field->value):\n{record_lines}\n\n"
                "For EACH record, compare it against the actual row it should correspond to in "
                "the image(s). A record is correct only if every one of its field values "
                "genuinely matches that row -- watch specifically for a row bleeding into its "
                "neighbor (a value from the row above or below), a count or date shifted by one "
                "column, or a value invented that isn't in the table at all. Return the indices "
                "of every record that is fully and correctly grounded; omit any record with even "
                "one wrong field."
            ),
        }
    )
    return content


async def _judge_multi_row_extraction(
    images: list[bytes],
    records: list[dict[str, Any]],
    schema: FieldSchema,
    *,
    api_key: str,
) -> set[int]:
    """Confirms (or rejects) EACH of *records* against the same source page image(s), in
    ONE judge call -- the ``"multi_row_vision"`` StrategyType's own grounding check.

    Batched into a single call deliberately (not N per-record calls): a table read
    returning many records has many independent ways to be confidently wrong (a row
    bled into its neighbor, a count misaligned by one column), and completeness is not
    correctness -- but N separate judge calls would multiply the LLM cost by the row
    count on every poll. Always uses the vision model/provider (unlike
    :func:`_judge_one_document_extraction`, there is no born-digital-text branch here --
    a ``multi_row_vision`` target is explicitly chosen because its table structure
    defeats text-based extraction in the first place, so there is no reliable text
    source to judge against either).

    :param images: the same page image(s) the extraction itself was read from
    :ptype images: list[bytes]
    :param records: every already-coerced, already-complete record extracted from *images*
    :ptype records: list[dict[str, Any]]
    :param schema: field_name -> expected Python type
    :ptype schema: FieldSchema
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :return: the 0-based indices of *records* the judge explicitly confirmed -- empty on
        total judge failure (fail-closed: an unconfirmable record is dropped, never kept
        just because judging itself failed)
    :rtype: set[int]
    """
    if not records:
        return set()
    content = _build_multi_row_judge_content(images, records, schema)
    from langchain_core.messages import HumanMessage

    verdict = await _judge(
        [HumanMessage(content=content)],
        response_model=_MultiRowJudgeVerdict,
        model_id=DEFAULT_VISION_MODEL_ID,
        api_key=api_key,
        provider=_VISION_PROVIDER,
        log_label="scrape multi-row judge",
    )
    if verdict is None:
        return set()
    return {i for i in verdict.confirmed_record_indices if 0 <= i < len(records)}


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


async def _run_per_document_extraction(
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    api_key: str,
    extraction_model_id: str,
    judge_model_id: str,
) -> ScrapeExtraction:
    """``"per_document"`` StrategyType: no cached pattern is possible (see that
    Literal's own comment) -- every document gets a fresh, independent LLM
    extraction call, every single poll, never a reuse-without-an-LLM-call path.

    Still persists a marker ``ScrapeRecipe`` (``extraction_strategy={"strategy":
    "per_document"}``, no reusable pattern inside it) so ``consecutive_validation_
    failures`` keeps tracking a real operational signal -- "how many recent polls
    found zero extractable records," e.g. the listing's own JSON shape changed --
    the same way css/regex targets are already observable, even though nothing
    here is ever reused to skip an LLM call.

    Each document's call is bounded by an explicit outer deadline
    (:data:`_PER_DOCUMENT_TIMEOUT_SECONDS`), not just ``extract_fields_directly``'s
    own per-attempt *timeout* -- live-reproduced (scrape-task-05, real West Virginia
    document): the underlying chat client occasionally hangs well past its
    configured per-attempt timeout with zero further retry activity (the 200 OK
    response headers land, the body read then never completes), a pre-existing
    reliability gap in :func:`~threetears.scrape.llm_retry.bounded_retry_structured_call`
    shared by every caller in this module, filed separately (not fixed here -- out
    of scope for this feature, bigger blast radius). What per_document uniquely
    needs, and gets: one stuck document must never hang an entire poll of N
    documents forever, the same "isolate one bad unit's failure" philosophy
    :class:`~threetears.scrape.drivers.multi_document.MultiDocumentDriver` already
    applies to one document's FETCH failing.

    Documents run concurrently (``asyncio.gather``), not one at a time -- each is a
    fully independent extraction (no shared cache/state), and
    :func:`~threetears.scrape.extraction.extract_fields_directly_chunked` already
    roughly doubles the LLM calls a single document needs (scrape-task-05's own
    reliability fix), so serializing across documents on top of that would make an
    N-document poll's wall-clock cost grow far faster than the accuracy gain
    justifies.

    **Routing by document shape (scrape-task-06):** a scanned document
    (``NoticeDocument.was_ocr``) routes to :func:`~threetears.scrape.extraction.
    extract_fields_from_images` (a vision-capable model reading the original page
    images) rather than the OCR'd-text path -- full-set live comparison against a
    real target's own documents found OCR'd text measurably less reliable (2/10
    complete records vs. vision's 10/10, same documents). A born-digital document
    (``was_ocr=False``, no embedded images to read anyway) stays on the fast/cheap
    text path unchanged -- vision's own real cost/latency is only paid where OCR
    was needed in the first place, not globally.

    **Grounding check (scrape-task-06):** css/regex strategies get a real semantic-
    correctness check -- the judge (:func:`_judge_candidates`/:func:`_judge_row_candidates`)
    compares candidate values against real page content -- but only once, on cold
    start, when candidates are first generated; a healthy cached recipe skips it on
    every later poll. per_document has no cached recipe to ever skip it: every
    document's own extraction, every poll, is confirmed against its own real source
    content (text or images, matching the extraction path's own choice) via
    :func:`_judge_one_document_extraction` before counting as a real record --
    otherwise structural type-validity alone (the only check
    :func:`~threetears.scrape.extraction.extract_fields_directly`'s own
    ``is_acceptable`` plausibility guard provides) can't catch a confident,
    well-typed, but wrong or hallucinated value.
    """
    documents = split_notice_documents(html)

    async def _extract_one(document: NoticeDocument) -> dict[str, Any] | None:
        extraction_call = (
            extract_fields_from_images(document.images, schema, api_key=api_key)
            if document.was_ocr
            else extract_fields_directly_chunked(document.text, schema, model_id=extraction_model_id, api_key=api_key)
        )
        try:
            extracted = await asyncio.wait_for(extraction_call, timeout=_PER_DOCUMENT_TIMEOUT_SECONDS)
        except TimeoutError:
            log.warning(
                "scrape per-document extraction: one document hung past %ss, skipping",
                _PER_DOCUMENT_TIMEOUT_SECONDS,
                extra={"extra_data": {"target_id": target_id}},
            )
            return None
        # All-or-nothing-per-record, matching every other strategy's own philosophy
        # (validate_row_candidate / validate_regex_row_candidate): a record only
        # counts if EVERY schema field was found and coerced, never a partial one --
        # checked before spending a judge call on something already going to be dropped.
        if extracted is None or set(extracted) != set(schema):
            return None
        try:
            confirmed = await asyncio.wait_for(
                _judge_one_document_extraction(document, extracted, schema, api_key=api_key, judge_model_id=judge_model_id),
                timeout=_PER_DOCUMENT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            log.warning(
                "scrape per-document extraction: judge hung past %ss, treating as unconfirmed",
                _PER_DOCUMENT_TIMEOUT_SECONDS,
                extra={"extra_data": {"target_id": target_id}},
            )
            return None
        return extracted if confirmed else None

    records = [
        extracted for extracted in await asyncio.gather(*(_extract_one(document) for document in documents)) if extracted is not None
    ]

    now = datetime.now(UTC)
    existing_recipe = await recipe_collection.get(target_id)
    await _save_recipe(
        recipe_collection,
        target_id=target_id,
        extraction_strategy={"strategy": "per_document"},
        won_at=existing_recipe.won_at if existing_recipe is not None and existing_recipe.won_at else now,
        last_validated_at=now,
        consecutive_validation_failures=(
            0 if records else (existing_recipe.consecutive_validation_failures + 1 if existing_recipe else 1)
        ),
    )
    log.info(
        "scrape per-document extraction: target=%s documents=%d records_captured=%d",
        target_id,
        len(documents),
        len(records),
        extra={"extra_data": {"target_id": target_id}},
    )
    return await _persist_extraction(
        extraction_collection,
        target_id=target_id,
        source_url=source_url,
        structured_fields={"records": records},
        validation_status="validated" if records else "failed",
        extraction_recipe_id=target_id if records else None,
    )


async def _run_multi_row_vision_extraction(
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    api_key: str,
) -> ScrapeExtraction:
    """``"multi_row_vision"`` StrategyType: one page, one table, read once via vision,
    every record grounded against the same source image(s) before counting as real.

    Like :func:`_run_per_document_extraction`, there is no cached selector pattern to
    reuse -- ``find_tables()`` (the text-based table extraction every other multi-row
    strategy could fall back on) is exactly what this StrategyType exists because it
    fails for this target's own real table (see :func:`~threetears.scrape.extraction.
    extract_multi_row_fields_from_images`'s own docstring for the live evidence). Still
    persists a marker ``ScrapeRecipe`` for the same operational-observability reason
    per_document does.

    **Partial-confidence ``validation_status``, unlike per_document's binary validated/
    failed:** per_document extracts one record per document, so there's no "some right,
    some wrong" middle state to represent -- it's validated if any document's record
    survived judging, failed otherwise. A single multi-row table read can PARTIALLY
    succeed (say, 15 of 17 rows judge-confirmed, 2 rejected) -- persisting only the
    confirmed rows (never a rejected one, matching every other strategy's fail-closed
    contract) but marking the whole extraction ``"needs_review"`` rather than silently
    ``"validated"`` when it isn't complete, a real, human-checkable signal a 17-row
    table can genuinely produce that a 1-record document can't.
    """
    images = extract_page_images(html)
    if not images:
        log.warning(
            "scrape multi-row vision extraction: no page images found for target %s",
            target_id,
            extra={"extra_data": {"target_id": target_id}},
        )
        now = datetime.now(UTC)
        existing_recipe = await recipe_collection.get(target_id)
        await _save_recipe(
            recipe_collection,
            target_id=target_id,
            extraction_strategy={"strategy": "multi_row_vision"},
            won_at=existing_recipe.won_at if existing_recipe is not None and existing_recipe.won_at else now,
            last_validated_at=now,
            consecutive_validation_failures=(existing_recipe.consecutive_validation_failures + 1 if existing_recipe else 1),
        )
        return await _persist_extraction(
            extraction_collection,
            target_id=target_id,
            source_url=source_url,
            structured_fields={"records": []},
            validation_status="failed",
            extraction_recipe_id=None,
        )

    try:
        extracted_records = await asyncio.wait_for(
            extract_multi_row_fields_from_images(images, schema, api_key=api_key),
            timeout=_MULTI_ROW_EXTRACTION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        log.warning(
            "scrape multi-row vision extraction: extraction hung past %ss for target %s",
            _MULTI_ROW_EXTRACTION_TIMEOUT_SECONDS,
            target_id,
            extra={"extra_data": {"target_id": target_id}},
        )
        extracted_records = None

    # All-or-nothing-per-record, same philosophy as every other strategy: a record
    # only counts if EVERY schema field was found and coerced.
    complete_records = [record for record in (extracted_records or []) if set(record) == set(schema)]

    if not complete_records:
        now = datetime.now(UTC)
        existing_recipe = await recipe_collection.get(target_id)
        await _save_recipe(
            recipe_collection,
            target_id=target_id,
            extraction_strategy={"strategy": "multi_row_vision"},
            won_at=existing_recipe.won_at if existing_recipe is not None and existing_recipe.won_at else now,
            last_validated_at=now,
            consecutive_validation_failures=(existing_recipe.consecutive_validation_failures + 1 if existing_recipe else 1),
        )
        return await _persist_extraction(
            extraction_collection,
            target_id=target_id,
            source_url=source_url,
            structured_fields={"records": []},
            validation_status="failed",
            extraction_recipe_id=None,
        )

    try:
        confirmed_indices = await asyncio.wait_for(
            _judge_multi_row_extraction(images, complete_records, schema, api_key=api_key),
            timeout=_MULTI_ROW_EXTRACTION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        log.warning(
            "scrape multi-row vision extraction: judge hung past %ss for target %s, treating all as unconfirmed",
            _MULTI_ROW_EXTRACTION_TIMEOUT_SECONDS,
            target_id,
            extra={"extra_data": {"target_id": target_id}},
        )
        confirmed_indices = set()

    confirmed_records = [record for i, record in enumerate(complete_records) if i in confirmed_indices]

    now = datetime.now(UTC)
    existing_recipe = await recipe_collection.get(target_id)
    await _save_recipe(
        recipe_collection,
        target_id=target_id,
        extraction_strategy={"strategy": "multi_row_vision"},
        won_at=existing_recipe.won_at if existing_recipe is not None and existing_recipe.won_at else now,
        last_validated_at=now,
        consecutive_validation_failures=(
            0 if confirmed_records else (existing_recipe.consecutive_validation_failures + 1 if existing_recipe else 1)
        ),
    )
    log.info(
        "scrape multi-row vision extraction: target=%s extracted=%d confirmed=%d",
        target_id,
        len(complete_records),
        len(confirmed_records),
        extra={"extra_data": {"target_id": target_id}},
    )
    if not confirmed_records:
        validation_status = "failed"
    elif len(confirmed_records) == len(complete_records):
        validation_status = "validated"
    else:
        validation_status = "needs_review"
    return await _persist_extraction(
        extraction_collection,
        target_id=target_id,
        source_url=source_url,
        structured_fields={"records": confirmed_records},
        validation_status=validation_status,
        extraction_recipe_id=target_id if confirmed_records else None,
    )


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
    :param strategy_type: ``"css"`` (row/field CSS selectors), ``"regex"``
        (a single pattern matched repeatedly via ``re.finditer`` against the
        page's plain text, one match per record), ``"per_document"`` (no
        cached pattern at all -- a fresh LLM extraction call per document,
        every poll), or ``"multi_row_vision"`` (a single PDF whose own table
        structure defeats text-based extraction -- a vision read of the
        whole table, every record grounded before counting; see
        :data:`StrategyType`'s own comment for why)
    :ptype strategy_type: StrategyType
    :return: the persisted ``ScrapeExtraction`` row (``structured_fields["records"]`` holds every record)
    :rtype: ScrapeExtraction
    """
    if strategy_type == "per_document":
        return await _run_per_document_extraction(
            html,
            schema,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            api_key=api_key,
            extraction_model_id=extraction_model_id,
            judge_model_id=judge_model_id,
        )
    if strategy_type == "multi_row_vision":
        return await _run_multi_row_vision_extraction(
            html,
            schema,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            api_key=api_key,
        )
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
