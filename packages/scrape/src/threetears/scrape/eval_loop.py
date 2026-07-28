"""Eval loop: LLM-judge candidate comparison + recipe persistence/reuse.

Orchestrates ``extraction.py``'s candidate generation and structural
validation into the self-healing cycle the product brief describes: a
healthy target's existing ``ScrapeRecipe`` is reused fetch after fetch with
no LLM call at all (just re-executing its stored selectors); only when
``consecutive_validation_failures`` crosses a threshold does candidate
generation re-run, survivors get compared by an LLM judge against the real
page content, and the winner is persisted as the new recipe.

**Two exceptions, and they are exceptions to the whole cycle rather than variations
of it: ``StrategyType`` ``"per_document"`` and ``"multi_row_vision"``.** Neither has
a cached pattern to reuse, so neither can make a poll free, and each persists a
marker ``ScrapeRecipe`` for operational visibility rather than for reuse -- a recipe
row existing is not a recipe being reused. ``multi_row_vision``'s reason is its own
(``find_tables()``, the text substrate a pattern would be cached against, fails on
its table); see :func:`_run_multi_row_vision_extraction`, which costs an extraction
call AND an unconditional grounding judge on every poll.

``"per_document"`` (2026-07-15) has no cached-recipe cycle at all -- some real
multi-document targets (independently-worded documents sharing no template, see
``drivers/multi_document.py``) genuinely cannot be served by a pattern learned once
and reused; every document gets its own fresh extraction on every poll instead, plus
an unconditional grounding judge. The extraction's cost depends on the document's
shape: a born-digital one is chunked by field count, so it can be more than one call
past the chunk size, while an OCR'd one is a single vision call over its images
whatever the schema. So the floor is two calls per document -- one extraction, one
judge -- and the ceiling scales with your schema
(:func:`_run_per_document_extraction`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar, get_args

from pydantic import BaseModel
from pydantic import Field as PydanticField
from threetears.models import LlmPurpose
from threetears.observe import get_logger

from .challenge import DEFAULT_CLASSIFIER_MODEL_ID, PageVerdictKind, classify_failed_page
from .collections import (
    ScrapeExtraction,
    ScrapeExtractionCollection,
    ScrapeRecipe,
    ScrapeRecipeCollection,
    ValidationStatus,
)
from .health import (
    ScrapeTargetHealthCollection,
    content_fingerprint,
    record_classification,
    record_validated_fetch,
)
from .extraction import (
    DEFAULT_EXTRACTION_MODEL_ID,
    DEFAULT_VISION_MODEL_ID,
    MAX_HTML_CHARS_IN_PROMPT,
    FieldSchema,
    NoticeDocument,
    RowValidationResult,
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
    ValidationResult,
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
#: 2026-07-15, live-verified against West Virginia's real
#: WARN letters after a regex-strategy attempt matched only 1 of 10), or
#: "multi_row_vision" (a single born-digital PDF whose own table structure
#: defeats text-based table extraction -- e.g. Nevada's real master WARN
#: PDF, where ``find_tables()`` finds only the header one way and mis-splits
#: columns the other -- needs a vision read of the whole table at once;
#: added 2026-07-15, explicitly chosen per-target in config,
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

#: Hard outer deadline for ONE document's extraction inside
#: ``"per_document"`` StrategyType (:func:`_run_per_document_extraction`) -- live-
#: reproduced (a real West Virginia document): the underlying chat
#: client can hang well past its own configured per-attempt timeout with zero
#: further retry activity, so the extractor's own *timeout*/*attempts*
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
    either shape), *log_label*, and -- since the multi-row judge arrived --
    *response_model*
    (:class:`_JudgeVerdict`'s single-winner shape for css/regex/per_document; the
    multi-row judge passes :class:`_MultiRowJudgeVerdict` instead, since "which ONE
    candidate wins" doesn't fit "which of these N independent records are each
    individually correct" -- required explicitly, not defaulted, so mypy can infer
    *T* precisely at each call site). Previously ``_judge_candidates`` and
    ``_judge_row_candidates`` each called ``bounded_retry_structured_call`` directly
    with near-identical arguments; the per-document grounding check made that
    duplication worth closing rather than adding a third copy of it.
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


async def _stamp_fingerprint_if_validated(
    health_collection: ScrapeTargetHealthCollection | None,
    result: ScrapeExtraction,
    *,
    target_id: str,
    html: str,
) -> None:
    """Record what the page looked like, but only when we actually read it successfully.

    Called on every path that can persist an extraction, rather than inside each of the
    reuse and regeneration branches that produce one: a fingerprint only some paths wrote
    would go stale silently after whichever path was missed, and the first version of this
    change did exactly that, leaving ``per_document`` and ``multi_row_vision`` unstamped
    because both return before the multi-row entry point's common exit. Keying off the
    persisted row's own ``validation_status`` means each call cannot disagree with the
    outcome it describes; keeping the number of call sites small is what makes "every path"
    checkable by reading one function.

    ``needs_review`` deliberately does not stamp. That status means structurally valid
    candidates existed but nothing confirmed they were right, so the page is not a
    trustworthy "this is what the target looks like when it works" reference.

    Silent no-op when *health_collection* is ``None``, which is every caller that has not
    opted in yet.
    """
    if health_collection is None or result.validation_status != "validated":
        return
    try:
        await record_validated_fetch(health_collection, target_id=target_id, html=html)
    except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- health is a diagnostic aid and the extraction is already durable; logged with its traceback below, never silenced
        # The extraction is already persisted and correct by the time this runs. Health is
        # a diagnostic aid, so letting its write failure propagate would turn a successful
        # scrape into a failed one for the caller and lose real extracted data over a
        # bookkeeping row. Logged at exception level with the traceback, never silenced.
        log.exception(
            "scrape health: fingerprint stamp failed for target %s; extraction is unaffected",
            target_id,
            extra={"extra_data": {"target_id": target_id}},
        )


async def _persist_extraction(
    extraction_collection: ScrapeExtractionCollection,
    *,
    target_id: str,
    source_url: str,
    structured_fields: dict[str, Any],
    validation_status: ValidationStatus,
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


@dataclass(frozen=True)
class _ReuseCheck:
    """What re-running a stored strategy against a freshly fetched page produced.

    Deliberately carries no decision. Each strategy shape knows how to run its own stored
    strategy and nothing else; what a miss MEANS -- a transient blip, a redesign, a wall --
    is one question with one answer, and it is answered in one place
    (:func:`_run_reuse_cycle`) rather than four times over.
    """

    valid: bool
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class _FailureVerdict:
    """What a page that failed extraction turned out to be.

    Distinct from :class:`~threetears.scrape.challenge.PageVerdict`, which is what a model
    returns. This is what the eval loop knows, and it may have come from a cached row rather
    than from a call just made.

    *from_cache* is not bookkeeping -- it changes what a verdict MEANS. A fresh ``"changed"``
    is new evidence and calls for regeneration. A cached ``"changed"`` says we already
    regenerated against this exact page and it did not stick, so regenerating again would
    burn a candidate round on every poll, for a page we have demonstrably failed to learn.
    A ``"blocked"`` verdict is unaffected either way, because blocked describes a state
    rather than an action outstanding.
    """

    kind: PageVerdictKind
    evidence: str
    from_cache: bool


#: Values :attr:`~threetears.scrape.health.ScrapeTargetHealth.classified_verdict` is allowed
#: to hold, derived from the Literal rather than restated, so a new verdict kind cannot be
#: added without this accepting it. A stored value outside the set is treated as no cached
#: verdict at all: it is either data from a version that meant something different, or
#: corruption, and re-asking is cheap while acting on a value we cannot interpret is not.
_KNOWN_VERDICT_KINDS: frozenset[str] = frozenset(get_args(PageVerdictKind))


@dataclass(frozen=True)
class _StrategyShape:
    """Everything that differs between the four cached-recipe extraction strategies.

    There are four: CSS selectors or a regex pattern, each over a single record or many
    rows. They used to be eight near-identical functions -- a reuse checker and a ~90-line
    regeneration body apiece -- differing only in which generator and validator they called
    and how they wrapped a winning candidate. Every one of those bodies also had to grow the
    same failure-classification hook, which is what made the duplication expensive rather
    than merely untidy: a change belongs in one of these fields or in the one shared body,
    never in four places that can drift apart.

    Each field is a real difference. If a fifth strategy shape arrives it declares these and
    inherits the reuse cycle, the judging, the classification routing and the persistence by
    construction.
    """

    #: Prefix for this shape's log lines, e.g. ``"scrape regex row eval loop"``.
    log_label: str
    #: What the generator and validator read: the raw HTML, or its extracted plain text.
    source: Callable[[str], str]
    #: Ask the model for *n* candidate strategies of this shape.
    generate: Callable[..., Awaitable[list[Any]]]
    #: Run one candidate against the source and report what it structurally yielded.
    #: The two result types are genuinely different -- ``ValidationResult`` carries one
    #: ``extracted`` dict, ``RowValidationResult`` carries a ``records`` list and a row
    #: count -- so the union is named rather than erased to ``Any``. Erasing it here
    #: would have removed the one place a mismatched shape could be caught statically,
    #: at exactly the seam every strategy now funnels through.
    validate: Callable[[str, Any, FieldSchema], ValidationResult | RowValidationResult]
    #: Pull the records out of a validation. One element for a single-record shape.
    records: Callable[[ValidationResult | RowValidationResult], list[dict[str, Any]]]
    #: Wrap a winning candidate into the ``extraction_strategy`` dict stored on the recipe.
    as_strategy: Callable[[Any], dict[str, Any]]
    #: Read a stored ``extraction_strategy`` back into a candidate. Inverse of *as_strategy*.
    from_strategy: Callable[[dict[str, Any]], Any]
    #: Compare candidates against the real page. Single-record and row shapes ask differently.
    judge: Callable[..., Awaitable[_JudgeVerdict | None]]
    #: Shape each survivor's records into what *judge* expects for this strategy.
    judge_payload: Callable[[list[dict[str, Any]]], Any]


def _check_reuse(
    shape: _StrategyShape,
    existing_recipe: ScrapeRecipe,
    html: str,
    schema: FieldSchema,
    target_id: str,
) -> _ReuseCheck:
    """Re-run *existing_recipe*'s stored strategy against a freshly fetched page. Pure."""
    validation = shape.validate(shape.source(html), shape.from_strategy(existing_recipe.extraction_strategy), schema)
    # Narrowed on the result TYPE, not on a flag saying what the type should be. The row
    # counts only exist on the row result, so asking the value itself is both the correct
    # guard and one fewer field that a fifth strategy shape could pair wrongly.
    if isinstance(validation, RowValidationResult):
        log.info(
            "%s recipe reuse: target=%s records_captured=%d rows_matched=%d",
            shape.log_label,
            target_id,
            len(validation.records),
            validation.total_rows_matched,
            extra={"extra_data": {"target_id": target_id}},
        )
    return _ReuseCheck(valid=validation.valid, records=shape.records(validation))


async def _commit_reuse(
    existing_recipe: ScrapeRecipe,
    check: _ReuseCheck,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
) -> ScrapeExtraction:
    """Persist the outcome of re-running a stored strategy; keep the recipe either way.

    Below the failure threshold the recipe is never abandoned on a single miss
    (transient-failure tolerance) -- only the failure counter moves. Shared by all four
    strategy shapes because all four did exactly this, character for character apart from
    the record list, and a fifth copy is how one of them ends up counting differently from
    the others.
    """
    now = datetime.now(UTC)
    await _save_recipe(
        recipe_collection,
        target_id=target_id,
        extraction_strategy=existing_recipe.extraction_strategy,
        won_at=existing_recipe.won_at or now,
        last_validated_at=now if check.valid else (existing_recipe.last_validated_at or now),
        consecutive_validation_failures=0 if check.valid else existing_recipe.consecutive_validation_failures + 1,
    )
    return await _persist_extraction(
        extraction_collection,
        target_id=target_id,
        source_url=source_url,
        structured_fields={"records": check.records},
        validation_status="validated" if check.valid else "failed",
        extraction_recipe_id=target_id,
    )


async def _commit_blocked(
    verdict: _FailureVerdict,
    target_id: str,
    source_url: str,
    *,
    extraction_collection: ScrapeExtractionCollection,
) -> ScrapeExtraction:
    """Record that we never received the content, and touch nothing else.

    The absence of a ``_save_recipe`` call here is the entire fix, not an omission. A page
    we were walled off from says nothing about whether the stored strategy still works, so
    incrementing its failure counter -- which is what happens today -- marches a perfectly
    good recipe toward being discarded and replaced by whatever selectors best fit a
    challenge page. The recipe row is left byte-identical, which also means a target blocked
    before it ever won a recipe stays a target with no recipe rather than acquiring an empty
    one.

    No records are persisted either. There were none; a wall has no data on it, and writing
    an empty extraction with a status that says so is the honest description.

    The verdict rides in ``field_confidences``, which is a stretch of that column's name and
    is deliberate rather than accidental. It is the row's existing free-form JSONB slot for
    "what the eval loop thought about this extraction", it is already the only such slot, and
    an operator looking at a blocked row needs the page's own words in front of them. The
    alternative was a column that only one status ever populates.
    """
    log.warning(
        "scrape eval loop: target %s appears to be behind a wall (%s); recipe left untouched",
        target_id,
        verdict.evidence,
        extra={"extra_data": {"target_id": target_id, "page_verdict": verdict.kind}},
    )
    return await _persist_extraction(
        extraction_collection,
        target_id=target_id,
        source_url=source_url,
        structured_fields={"records": []},
        validation_status="blocked",
        extraction_recipe_id=None,
        field_confidences={"page_verdict": verdict.kind, "page_verdict_evidence": verdict.evidence},
    )


async def _resolve_failure_verdict(
    html: str,
    schema: FieldSchema,
    target_id: str,
    *,
    health_collection: ScrapeTargetHealthCollection | None,
    api_key: str,
    classifier_model_id: str,
    page_status: int | None,
) -> _FailureVerdict | None:
    """Work out what a page that just failed extraction actually is. Cheapest question first.

    Three checks, and only the last one costs a model call:

    1. **Is this the page the recipe last validated against?** If the fingerprint matches,
       the page provably has not changed, and provably is not a new wall either, because a
       wall would not digest to the same content. Nothing to ask. This is the common case on
       a transient miss and it stays exactly as cheap as it is today.
    2. **Have we already asked about this exact page?** A cached verdict answers for free. It
       also records that we have already ACTED on this page, which is what stops a
       ``"changed"`` verdict regenerating on every poll after a regeneration that did not
       stick.
    3. **Otherwise ask**, once, and cache the answer against the page it was about.

    Known limit, because the cost of a wall is easy to overstate: check 2 only hits for a
    wall that renders identically each time. The fingerprint digests visible text, and a real
    Cloudflare interstitial puts a per-request Ray ID in exactly that, so such a target
    re-asks every poll. That is deliberately not solved by normalising ids out of the
    fingerprint, which would put vendor-shaped pattern matching back into the one place this
    design removed it and would suppress real content changes that happen to look like ids.
    What bounds a walled target is not fetching it every poll, which is
    :class:`threetears.scrape.circuit.TargetCircuit`'s job at the fetch boundary rather than
    this function's -- by the time a page reaches here it has already been paid for.

    ``None`` means "no opinion", and every caller must treat it as exactly today's
    behaviour. It is returned when there is no health store to consult, when the page is
    unchanged, and when the classifier could not answer -- an unanswerable question must
    never be more destructive than not having asked one.

    :param html: the page extraction just failed against
    :ptype html: str
    :param schema: field_name -> expected Python type, the fields we were looking for
    :ptype schema: FieldSchema
    :param target_id: the target this fetch belongs to
    :ptype target_id: str
    :param health_collection: the health store, or ``None`` for a caller that has not opted in
    :ptype health_collection: ScrapeTargetHealthCollection | None
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param classifier_model_id: model for the classification call
    :ptype classifier_model_id: str
    :param page_status: the HTTP status the page came back with, when the caller knows it
    :ptype page_status: int | None
    :return: the verdict, or ``None`` for no opinion
    :rtype: _FailureVerdict | None
    """
    if health_collection is None:
        return None

    fingerprint = content_fingerprint(html)
    try:
        health = await health_collection.get(target_id)
    except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- an unreadable health store must degrade this target to its pre-health behaviour, not fail every poll; logged with its traceback below
        # Health is a diagnostic aid. A store that cannot be read must degrade this target to
        # the behaviour it had before health existed, not turn every poll into a hard
        # failure. Logged with its traceback, never silenced.
        log.exception(
            "scrape health: could not read health for target %s; classifying nothing",
            target_id,
            extra={"extra_data": {"target_id": target_id}},
        )
        return None

    if health is not None:
        if health.content_fingerprint == fingerprint:
            return None
        cached = health.classified_verdict
        if health.classified_fingerprint == fingerprint and cached in _KNOWN_VERDICT_KINDS:
            log.info(
                "scrape page classifier: reusing the cached %s verdict for target %s, no model call",
                cached,
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )
            # Narrowed by the membership test above, which is why it is a membership test
            # against the Literal's own values rather than a cast.
            return _FailureVerdict(
                kind=cached,  # type: ignore[arg-type]
                evidence=health.classified_evidence or "",
                from_cache=True,
            )

    # The free paths above announce themselves; this is the one that costs a model
    # call, so it says so before spending it. Without this the cheap branch was the
    # only one visible in a log, which is exactly backwards for anything anyone
    # would want to count.
    log.info(
        "scrape page classifier: asking about target %s (no cached verdict for this page)",
        target_id,
        extra={"extra_data": {"target_id": target_id}},
    )
    verdict = await classify_failed_page(
        html,
        schema,
        api_key=api_key,
        model_id=classifier_model_id,
        page_status=page_status,
    )
    if verdict is None:
        # bounded_retry_structured_call already logged the failure itself; this
        # records what the eval loop does about it, which is nothing.
        log.info(
            "scrape page classifier: no verdict for target %s; treating the failure as today would",
            target_id,
            extra={"extra_data": {"target_id": target_id}},
        )
        return None

    # Every verdict leaves a trace, not just the two that change behaviour: a fresh
    # content/empty/other verdict used to buy a model call and produce no log line
    # anywhere, so the spend was invisible.
    log.info(
        "scrape page classifier: target %s judged %s (%s confidence) -- %s",
        target_id,
        verdict.kind,
        verdict.confidence,
        verdict.evidence,
        extra={"extra_data": {"target_id": target_id, "page_verdict": verdict.kind}},
    )

    try:
        await record_classification(
            health_collection,
            target_id=target_id,
            fingerprint=fingerprint,
            kind=verdict.kind,
            evidence=verdict.evidence,
        )
    except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the verdict is already in hand and routing on it beats discarding it; only the cache is lost, and it is logged with its traceback below
        # The verdict is already in hand and routing on it is strictly better than not. All
        # that is lost is the cache, so the next poll asks again. Same reasoning as the
        # fingerprint stamp: bookkeeping must not cost a real result.
        log.exception(
            "scrape health: could not cache the page verdict for target %s; routing on it anyway",
            target_id,
            extra={"extra_data": {"target_id": target_id}},
        )
    return _FailureVerdict(kind=verdict.kind, evidence=verdict.evidence, from_cache=False)


async def _regenerate(
    shape: _StrategyShape,
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    health_collection: ScrapeTargetHealthCollection | None,
    api_key: str,
    candidate_count: int,
    extraction_model_id: str,
    judge_model_id: str,
    classifier_model_id: str,
    page_status: int | None,
) -> ScrapeExtraction:
    """No healthy recipe exists: generate fresh candidates and consult the LLM judge.

    One body for all four strategy shapes. What varies is declared on *shape*.
    """
    source = shape.source(html)
    candidates = await shape.generate(source, schema, n=candidate_count, model_id=extraction_model_id, api_key=api_key)
    survivors = [
        (candidate, validation)
        for candidate, validation in ((c, shape.validate(source, c, schema)) for c in candidates)
        if validation.valid
    ]

    if not survivors:
        return await _persist_no_survivors(
            html,
            schema,
            target_id,
            source_url,
            log_label=shape.log_label,
            proposed=len(candidates),
            extraction_collection=extraction_collection,
            health_collection=health_collection,
            api_key=api_key,
            classifier_model_id=classifier_model_id,
            page_status=page_status,
        )

    verdict = await shape.judge(
        html,
        [shape.judge_payload(shape.records(validation)) for _, validation in survivors],
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
        # (or failed outright) -- an honest needs_review, not a crash, and not a
        # silently-crowned recipe. Surface the best survivor's data for human review rather
        # than nothing at all.
        #
        # "Best" is the one that captured the most records. For a row shape that is a real,
        # comparable signal rather than "first proposed". For a single-record shape every
        # survivor holds exactly one record, so max() returns the first maximal element and
        # this reduces to survivors[0] -- which is what the single-record paths did when
        # they were separate functions. Pinned by test, because it is the one place this
        # collapse could quietly change behaviour.
        _, best_validation = max(survivors, key=lambda pair: len(shape.records(pair[1])))
        return await _persist_extraction(
            extraction_collection,
            target_id=target_id,
            source_url=source_url,
            structured_fields={"records": shape.records(best_validation)},
            validation_status="needs_review",
            extraction_recipe_id=None,
        )

    winning_candidate, winning_validation = survivors[verdict.winning_candidate_index]
    now = datetime.now(UTC)
    await _save_recipe(
        recipe_collection,
        target_id=target_id,
        extraction_strategy=shape.as_strategy(winning_candidate),
        won_at=now,
        last_validated_at=now,
        consecutive_validation_failures=0,
    )
    return await _persist_extraction(
        extraction_collection,
        target_id=target_id,
        source_url=source_url,
        structured_fields={"records": shape.records(winning_validation)},
        validation_status="validated",
        extraction_recipe_id=target_id,
        field_confidences=verdict.field_confidences,
    )


async def _run_reuse_cycle(
    existing_recipe: ScrapeRecipe,
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    shape: _StrategyShape,
    regenerate: Callable[[], Awaitable[ScrapeExtraction]],
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    health_collection: ScrapeTargetHealthCollection | None,
    api_key: str,
    classifier_model_id: str,
    page_status: int | None,
) -> ScrapeExtraction:
    """Re-run a stored strategy, and decide what its failure meant before acting on it.

    The one place a reuse miss is interpreted, for every strategy shape. Three genuinely
    different situations used to share one response -- count the failure and, three polls
    later, throw the recipe away -- which is what let a bot wall destroy a working recipe.
    They are separated here:

    - **blocked**: we never received the content. The recipe is untouched.
    - **changed, newly**: the page really is different, so waiting two more polls to act on
      evidence we already have is pure latency. Regenerate now.
    - **changed, but read from the cache**: we already regenerated against this exact page
      and it did not stick. Regenerating again would spend a candidate round on every poll
      for a page we have demonstrably failed to learn -- strictly worse than the three-poll
      cadence this replaced -- so it falls through and counts the failure instead.
    - **anything else, or no opinion at all**: today's behaviour, unchanged. Count the
      failure and let the threshold decide, which is the right response to "our selectors
      are wrong" and the safe response to "we could not tell".
    """
    check = _check_reuse(shape, existing_recipe, html, schema, target_id)
    if check.valid:
        return await _commit_reuse(
            existing_recipe,
            check,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
        )

    verdict = await _resolve_failure_verdict(
        html,
        schema,
        target_id,
        health_collection=health_collection,
        api_key=api_key,
        classifier_model_id=classifier_model_id,
        page_status=page_status,
    )
    if verdict is not None:
        if verdict.kind == "blocked":
            return await _commit_blocked(verdict, target_id, source_url, extraction_collection=extraction_collection)
        if verdict.kind == "changed" and not verdict.from_cache:
            log.info(
                "scrape eval loop: target %s's page changed (%s); regenerating now rather than counting to the threshold",
                target_id,
                verdict.evidence,
                extra={"extra_data": {"target_id": target_id}},
            )
            return await regenerate()

    return await _commit_reuse(
        existing_recipe,
        check,
        target_id,
        source_url,
        recipe_collection=recipe_collection,
        extraction_collection=extraction_collection,
    )


async def _persist_no_survivors(
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    log_label: str,
    proposed: int,
    extraction_collection: ScrapeExtractionCollection,
    health_collection: ScrapeTargetHealthCollection | None,
    api_key: str,
    classifier_model_id: str,
    page_status: int | None,
) -> ScrapeExtraction:
    """Regeneration proposed candidates and none of them structurally matched the page.

    Shared by all four regeneration shapes. The classification matters here for a case the
    reuse path cannot reach: a target that has never had a working recipe, walled from the
    first fetch. Without this it would fail identically forever and nothing would ever mark
    it as needing a human. Only ``blocked`` changes anything -- a ``changed`` verdict has
    nowhere left to route, since regenerating is exactly what just failed.
    """
    log.warning(
        "%s: no structurally-valid candidates for target %s (%d proposed)",
        log_label,
        target_id,
        proposed,
        extra={"extra_data": {"target_id": target_id}},
    )
    verdict = await _resolve_failure_verdict(
        html,
        schema,
        target_id,
        health_collection=health_collection,
        api_key=api_key,
        classifier_model_id=classifier_model_id,
        page_status=page_status,
    )
    if verdict is not None and verdict.kind == "blocked":
        return await _commit_blocked(verdict, target_id, source_url, extraction_collection=extraction_collection)
    return await _persist_extraction(
        extraction_collection,
        target_id=target_id,
        source_url=source_url,
        structured_fields={"records": []},
        validation_status="failed",
        extraction_recipe_id=None,
    )


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
    classifier_model_id: str = DEFAULT_CLASSIFIER_MODEL_ID,
    strategy_type: StrategyType = "css",
    health_collection: ScrapeTargetHealthCollection | None = None,
    page_status: int | None = None,
) -> ScrapeExtraction:
    """Run one fetch through the eval loop and persist a ``ScrapeExtraction`` row.

    Reuses *target_id*'s existing recipe (no LLM call) while it's healthy;
    once ``consecutive_validation_failures`` crosses *failure_threshold*,
    regenerates candidates and consults the LLM judge for a new winner.

    When *health_collection* is supplied, a failure is interpreted before it is acted on:
    a page that turns out to be a bot wall leaves the recipe byte-identical instead of
    marching it toward the threshold, and a page that really did change regenerates on the
    first failure rather than the third. Omitting it keeps the pre-existing behaviour
    exactly, including spending no model call on any reuse failure.

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
    :param classifier_model_id: model for the "what is this page" call made when
        extraction fails and *health_collection* is supplied
    :ptype classifier_model_id: str
    :param strategy_type: ``"css"`` (an HTML table, CSS-selector candidates)
        or ``"regex"`` (a text-block/prose listing, regex-pattern
        candidates against the page's plain text) -- a per-target config
        value, the page's own shape, not something the eval loop infers
    :ptype strategy_type: StrategyType
    :param health_collection: this target's fetch-health store; ``None`` opts out of
        fingerprinting and failure classification entirely
    :ptype health_collection: ScrapeTargetHealthCollection | None
    :param page_status: the HTTP status the page came back with, when the caller knows
        it -- real evidence for the classifier, though rarely decisive on its own since
        most walls return 200
    :ptype page_status: int | None
    :return: the persisted ``ScrapeExtraction`` row (``structured_fields["records"]``
        holds a single-element list -- the same shape :func:`run_eval_loop_multi_row`
        uses, just always exactly one record)
    :rtype: ScrapeExtraction
    """
    shape = _REGEX_SHAPE if strategy_type == "regex" else _CSS_SHAPE

    async def _regenerate_now() -> ScrapeExtraction:
        return await _regenerate(
            shape,
            html,
            schema,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            health_collection=health_collection,
            api_key=api_key,
            candidate_count=candidate_count,
            extraction_model_id=extraction_model_id,
            judge_model_id=judge_model_id,
            classifier_model_id=classifier_model_id,
            page_status=page_status,
        )

    existing_recipe = await recipe_collection.get(target_id)
    if existing_recipe is not None and existing_recipe.consecutive_validation_failures < failure_threshold:
        result = await _run_reuse_cycle(
            existing_recipe,
            html,
            schema,
            target_id,
            source_url,
            shape=shape,
            regenerate=_regenerate_now,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            health_collection=health_collection,
            api_key=api_key,
            classifier_model_id=classifier_model_id,
            page_status=page_status,
        )
    else:
        result = await _regenerate_now()
    await _stamp_fingerprint_if_validated(health_collection, result, target_id=target_id, html=html)
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
    :func:`_judge`. The multi-row judge
    was first written as a deliberate copy of the single-record one rather
    than a shared abstraction -- two callers did not justify the indirection.
    A third judge use (per-document grounding) is what tipped it: at that
    point the same retry/backoff/degrade-to-``None`` policy was being
    maintained in three places. Never raises; returns ``None`` only after
    every attempt fails.
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


# ---------------------------------------------------------------------------
# The four cached-recipe strategy shapes.
#
# Declared here rather than beside :class:`_StrategyShape` because they name both judges,
# and the row judge is defined just above. Each is the complete description of one strategy:
# the reuse cycle, the regeneration body, the failure classification and the persistence are
# all shared, and nothing below this block knows which shape it is holding.
# ---------------------------------------------------------------------------


#: A single record's worth of extracted values, as the judge and the record list want it.
def _single_records(validation: ValidationResult | RowValidationResult) -> list[dict[str, Any]]:
    """One record, from the single-record result shape."""
    assert isinstance(validation, ValidationResult), "single-record shape got a row validation"
    return [validation.extracted]


def _row_records(validation: ValidationResult | RowValidationResult) -> list[dict[str, Any]]:
    """Every record, from the row result shape."""
    assert isinstance(validation, RowValidationResult), "row shape got a single-record validation"
    return validation.records


_CSS_SHAPE = _StrategyShape(
    log_label="scrape eval loop",
    source=lambda html: html,
    generate=generate_candidates,
    validate=validate_candidate,
    records=_single_records,
    as_strategy=lambda candidate: {"selectors": candidate},
    from_strategy=lambda strategy: strategy.get("selectors", {}),
    judge=_judge_candidates,
    # The single-record judge compares one set of values per candidate, not a row set.
    judge_payload=lambda records: records[0],
)

_REGEX_SHAPE = _StrategyShape(
    log_label="scrape regex eval loop",
    source=html_to_text,
    generate=generate_regex_candidates,
    validate=validate_regex_candidate,
    records=_single_records,
    as_strategy=lambda candidate: {"pattern": candidate},
    from_strategy=lambda strategy: strategy.get("pattern", ""),
    judge=_judge_candidates,
    judge_payload=lambda records: records[0],
)

_CSS_ROW_SHAPE = _StrategyShape(
    log_label="scrape row eval loop",
    source=lambda html: html,
    generate=generate_row_candidates,
    validate=validate_row_candidate,
    records=_row_records,
    # The row strategy is stored bare, not under a key: it is already a dict of a row
    # selector plus per-field selectors, so there is nothing to wrap it in.
    as_strategy=lambda candidate: dict(candidate),
    from_strategy=lambda strategy: strategy,
    judge=_judge_row_candidates,
    judge_payload=lambda records: records,
)

_REGEX_ROW_SHAPE = _StrategyShape(
    log_label="scrape regex row eval loop",
    source=html_to_text,
    generate=generate_regex_row_candidates,
    validate=validate_regex_row_candidate,
    records=_row_records,
    as_strategy=lambda candidate: {"pattern": candidate},
    from_strategy=lambda strategy: strategy.get("pattern", ""),
    judge=_judge_row_candidates,
    judge_payload=lambda records: records,
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


def _build_multi_row_judge_content(
    images: list[bytes], records: list[dict[str, Any]], schema: FieldSchema
) -> list[Any]:
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
    Literal's own comment) -- every document gets a fresh, independent extraction
    AND an unconditional grounding judge, every single poll, never a
    reuse-without-an-LLM-call path. Two calls per document is the floor. A
    born-digital document's extraction is chunked by field count, so it CAN cost more
    -- past the chunk size, not below it: a two-field schema is one chunk and hits
    the floor exactly. An OCR'd document is a single vision call whatever the schema.

    Still persists a marker ``ScrapeRecipe`` (``extraction_strategy={"strategy":
    "per_document"}``, no reusable pattern inside it) so ``consecutive_validation_
    failures`` keeps tracking a real operational signal -- "how many recent polls
    found zero extractable records," e.g. the listing's own JSON shape changed --
    the same way css/regex targets are already observable, even though nothing
    here is ever reused to skip an LLM call.

    Each document's call is bounded by an explicit outer deadline
    (:data:`_PER_DOCUMENT_TIMEOUT_SECONDS`), not just ``extract_fields_directly``'s
    own per-attempt *timeout* -- live-reproduced against a real West Virginia
    document: the underlying chat client occasionally hangs well past its
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
    roughly doubles the LLM calls a single document needs (its own
    reliability fix), so serializing across documents on top of that would make an
    N-document poll's wall-clock cost grow far faster than the accuracy gain
    justifies.

    **Routing by document shape:** a scanned document
    (``NoticeDocument.was_ocr``) routes to :func:`~threetears.scrape.extraction.
    extract_fields_from_images` (a vision-capable model reading the original page
    images) rather than the OCR'd-text path -- full-set live comparison against a
    real target's own documents found OCR'd text measurably less reliable (2/10
    complete records vs. vision's 10/10, same documents). A born-digital document
    (``was_ocr=False``, no embedded images to read anyway) stays on the fast/cheap
    text path unchanged -- vision's own real cost/latency is only paid where OCR
    was needed in the first place, not globally.

    **Grounding check:** css/regex strategies get a real semantic-
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
                _judge_one_document_extraction(
                    document, extracted, schema, api_key=api_key, judge_model_id=judge_model_id
                ),
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
        extracted
        for extracted in await asyncio.gather(*(_extract_one(document) for document in documents))
        if extracted is not None
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
            consecutive_validation_failures=(
                existing_recipe.consecutive_validation_failures + 1 if existing_recipe else 1
            ),
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
            consecutive_validation_failures=(
                existing_recipe.consecutive_validation_failures + 1 if existing_recipe else 1
            ),
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
    validation_status: ValidationStatus
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
    classifier_model_id: str = DEFAULT_CLASSIFIER_MODEL_ID,
    strategy_type: StrategyType = "css",
    health_collection: ScrapeTargetHealthCollection | None = None,
    page_status: int | None = None,
) -> ScrapeExtraction:
    """Run one fetch through the multi-row eval loop and persist a ``ScrapeExtraction`` row.

    The multi-row counterpart to :func:`run_eval_loop` -- extracts every matching
    record on the page (``structured_fields={"records": [...]}``), not a single set
    of values.

    **The reuse below describes the ``css`` and ``regex`` strategies only.** This
    function also dispatches ``per_document`` and ``multi_row_vision``, and neither
    has a cached pattern to reuse: each pays LLM calls on every poll regardless of
    how healthy its recipe row looks. Read *strategy_type* below before costing a
    target from this paragraph.

    For ``css``/``regex``: reuses *target_id*'s existing recipe (no LLM call) while
    it's healthy; once ``consecutive_validation_failures`` crosses
    *failure_threshold*, regenerates candidates and consults the LLM judge for a new
    winner -- same cadence as :func:`run_eval_loop`, just row-shaped throughout.

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
        cached pattern at all -- a fresh extraction plus a grounding judge per document,
        every poll), or ``"multi_row_vision"`` (a single PDF whose own table
        structure defeats text-based extraction -- a vision read of the
        whole table, every record grounded before counting; see
        :data:`StrategyType`'s own comment for why)
    :ptype strategy_type: StrategyType
    :param classifier_model_id: model for the "what is this page" call made when
        extraction fails and *health_collection* is supplied
    :ptype classifier_model_id: str
    :param health_collection: this target's fetch-health store; ``None`` opts out of
        fingerprinting and failure classification entirely
    :ptype health_collection: ScrapeTargetHealthCollection | None
    :param page_status: the HTTP status the page came back with, when the caller knows it
    :ptype page_status: int | None
    :return: the persisted ``ScrapeExtraction`` row (``structured_fields["records"]`` holds every record)
    :rtype: ScrapeExtraction
    """
    # One exit, deliberately. These two strategies used to `return` here, which put them
    # past the fingerprint stamp below even though both can persist a validated
    # extraction. Assigning and falling through to a single stamp-then-return means a
    # strategy added later is covered by construction rather than by remembering, which
    # is the property this needs: the first version of this feature claimed that coverage
    # while two strategies quietly lacked it.
    if strategy_type == "per_document":
        result = await _run_per_document_extraction(
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
    elif strategy_type == "multi_row_vision":
        result = await _run_multi_row_vision_extraction(
            html,
            schema,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            api_key=api_key,
        )
    else:
        result = await _run_row_strategy(
            html,
            schema,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            health_collection=health_collection,
            api_key=api_key,
            candidate_count=candidate_count,
            failure_threshold=failure_threshold,
            extraction_model_id=extraction_model_id,
            judge_model_id=judge_model_id,
            classifier_model_id=classifier_model_id,
            strategy_type=strategy_type,
            page_status=page_status,
        )
    await _stamp_fingerprint_if_validated(health_collection, result, target_id=target_id, html=html)
    return result


async def _run_row_strategy(
    html: str,
    schema: FieldSchema,
    target_id: str,
    source_url: str,
    *,
    recipe_collection: ScrapeRecipeCollection,
    extraction_collection: ScrapeExtractionCollection,
    health_collection: ScrapeTargetHealthCollection | None,
    api_key: str,
    candidate_count: int,
    failure_threshold: int,
    extraction_model_id: str,
    judge_model_id: str,
    classifier_model_id: str,
    strategy_type: StrategyType,
    page_status: int | None,
) -> ScrapeExtraction:
    """The cached-recipe cycle for the ``css`` and ``regex`` row strategies.

    Split out of :func:`run_eval_loop_multi_row` so that function is a flat choice between
    strategies with one exit, rather than a mix of early returns and inline logic that a
    later strategy could be appended to without picking up the shared post-processing.
    """
    shape = _REGEX_ROW_SHAPE if strategy_type == "regex" else _CSS_ROW_SHAPE

    async def _regenerate_now() -> ScrapeExtraction:
        return await _regenerate(
            shape,
            html,
            schema,
            target_id,
            source_url,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            health_collection=health_collection,
            api_key=api_key,
            candidate_count=candidate_count,
            extraction_model_id=extraction_model_id,
            judge_model_id=judge_model_id,
            classifier_model_id=classifier_model_id,
            page_status=page_status,
        )

    existing_recipe = await recipe_collection.get(target_id)
    if existing_recipe is not None and existing_recipe.consecutive_validation_failures < failure_threshold:
        result = await _run_reuse_cycle(
            existing_recipe,
            html,
            schema,
            target_id,
            source_url,
            shape=shape,
            regenerate=_regenerate_now,
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            health_collection=health_collection,
            api_key=api_key,
            classifier_model_id=classifier_model_id,
            page_status=page_status,
        )
    else:
        result = await _regenerate_now()
    return result
