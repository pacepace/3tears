"""Secondary enrichment pass -- free-form LLM notes, kept separate from structured data.

A second, separate LLM call over the rendered page, capturing free-form
metadata/context the structured extraction schema has no field for (e.g.
ambiguous wording, unusual formatting, nearby related information). This is
deliberately NOT validated structured data -- it's LLM commentary, stored
in ``ScrapeExtraction.enrichment_notes``, kept distinct from
``structured_fields`` so consumers can always tell the two apart: a
consumer that trusts ``structured_fields`` is trusting something a candidate
strategy structurally validated, while anything read out of
``enrichment_notes`` is unvalidated model commentary and has to be treated
as such.

A pass whose every attempt fails is recorded as failed, with its reason, in
``ScrapeExtraction.enrichment_status`` / ``enrichment_failure``, and never as
``{}``: ``{}`` means the model answered and had nothing to add, and a reader
must never have to infer which of the two it is looking at.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field as PydanticField
from threetears.models import LlmPurpose
from threetears.observe import get_logger

from .collections import ScrapeExtraction, ScrapeExtractionCollection
from .llm_retry import StructuredCallExhaustedError, bounded_retry_structured_call_or_raise

__all__ = ["DEFAULT_ENRICHMENT_MODEL_ID", "EnrichmentFailedError", "enrich_extraction", "run_enrichment"]

log = get_logger(__name__)

# Same reliability posture as extraction.py / eval_loop.py / query_agent/matching.py.
DEFAULT_ENRICHMENT_MODEL_ID = "deepseek/deepseek-chat-v3-0324"

_ENRICHMENT_TIMEOUT_SECONDS = 30
_ENRICHMENT_ATTEMPTS = 6
_ENRICHMENT_BACKOFF_SECONDS = 2.0
_MAX_HTML_CHARS_IN_PROMPT = 12000


class EnrichmentFailedError(RuntimeError):
    """Every attempt of the enrichment pass failed.

    Raised by :func:`run_enrichment`. :func:`enrich_extraction` catches it and stores
    the row as ``enrichment_status="failed"`` with :attr:`reason` as its
    ``enrichment_failure``. The last attempt's own exception is chained as
    ``__cause__``.

    :param reason: the last attempt's exception as ``"<ExceptionType>: <message>"``
    :ptype reason: str
    :param attempts: how many attempts were made, every one of which failed
    :ptype attempts: int
    """

    def __init__(self, reason: str, *, attempts: int) -> None:
        """Record why the pass failed and after how many attempts.

        :param reason: the last attempt's exception as ``"<ExceptionType>: <message>"``
        :ptype reason: str
        :param attempts: how many attempts were made, every one of which failed
        :ptype attempts: int
        :return: nothing
        :rtype: None
        """
        self.reason = reason
        self.attempts = attempts
        super().__init__(f"scrape enrichment failed after {attempts} attempts: {reason}")


class _EnrichmentResult(BaseModel):
    """Forced response shape for the enrichment LLM call."""

    notes: dict[str, str] = PydanticField(
        default_factory=dict,
        description=(
            "free-form key -> observation about the page's content or context that the "
            "structured fields don't capture; empty if there's genuinely nothing noteworthy"
        ),
    )


def _build_enrichment_prompt(html: str, structured_fields: dict[str, Any]) -> str:
    truncated = html[:_MAX_HTML_CHARS_IN_PROMPT]
    return (
        "You are reviewing a rendered web page that has already been parsed into structured "
        "fields. Note any additional context, caveats, or noteworthy observations about the "
        "page that the structured fields below do NOT capture -- e.g. ambiguous wording, "
        "unusual formatting, or additional related information nearby that a human reviewer "
        "should know about. Return free-form key->note pairs; return an empty object if "
        "there's genuinely nothing noteworthy beyond the structured fields.\n\n"
        f"Structured fields already extracted:\n{structured_fields}\n\n"
        f"Page HTML (may be truncated):\n{truncated}"
    )


async def run_enrichment(
    html: str,
    structured_fields: dict[str, Any],
    *,
    model_id: str = DEFAULT_ENRICHMENT_MODEL_ID,
    api_key: str,
    attempts: int = _ENRICHMENT_ATTEMPTS,
    backoff_seconds: float = _ENRICHMENT_BACKOFF_SECONDS,
) -> dict[str, str]:
    """Run the secondary enrichment LLM pass and return free-form notes.

    Same bounded retry as ``extraction.generate_candidates`` / ``eval_loop``'s judge
    call, but exhaustion raises rather than degrading: an empty dict is only ever the
    model's own answer that there is nothing to add. The failure is logged once, here,
    with its cause. ``asyncio.CancelledError`` is not a failure and propagates untouched.

    :param html: the rendered page's full HTML
    :ptype html: str
    :param structured_fields: the extraction's already-validated fields, given as
        context so the enrichment pass adds to them rather than repeating them
    :ptype structured_fields: dict[str, Any]
    :param model_id: the enrichment model
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :return: free-form key -> note pairs; empty only when the model had nothing to add
    :rtype: dict[str, str]
    :raises EnrichmentFailedError: if every attempt failed
    """
    prompt = _build_enrichment_prompt(html, structured_fields)
    try:
        result = await bounded_retry_structured_call_or_raise(
            prompt,
            _EnrichmentResult,
            model_id=model_id,
            api_key=api_key,
            purpose=LlmPurpose.SUMMARIZATION,
            temperature=0.3,
            timeout=_ENRICHMENT_TIMEOUT_SECONDS,
            attempts=attempts,
            backoff_seconds=backoff_seconds,
            log_label="scrape enrichment",
        )
    except StructuredCallExhaustedError as exc:
        cause = exc.last_error
        reason = f"{type(cause).__name__}: {cause}"
        log.error(
            "scrape enrichment failed after %d attempts: %s",
            exc.attempts,
            reason,
            extra={"extra_data": {"model_id": model_id}},
        )
        raise EnrichmentFailedError(reason, attempts=exc.attempts) from cause
    return result.notes


async def enrich_extraction(
    extraction: ScrapeExtraction,
    html: str,
    *,
    extraction_collection: ScrapeExtractionCollection,
    model_id: str = DEFAULT_ENRICHMENT_MODEL_ID,
    api_key: str,
) -> ScrapeExtraction:
    """Run the enrichment pass over *html* and persist its outcome onto *extraction*'s row.

    Only the three enrichment fields change -- ``structured_fields`` (and every
    other field) is carried through unmodified, so the eval loop's already-
    validated data is never touched by this second, separate LLM pass.

    The row records what happened, so no reader has to infer it:

    * the model answered -- ``enrichment_status="enriched"``, its notes (``{}`` when it
      had nothing to add), no ``enrichment_failure``;
    * every attempt failed -- ``enrichment_status="failed"``, ``enrichment_notes=None``,
      and the reason in ``enrichment_failure``. Not raised: the stored row is the
      result, and the failure was already logged once by :func:`run_enrichment`.

    The row describes the latest run. Passing a ``"failed"`` row back in is how it is
    retried; passing an ``"enriched"`` one back in replaces its notes, including with a
    failure if that run fails. ``asyncio.CancelledError`` propagates and persists nothing.

    :param extraction: the already-persisted row to enrich
    :ptype extraction: ScrapeExtraction
    :param html: the same rendered page's full HTML the extraction came from
    :ptype html: str
    :param extraction_collection: where the updated row is persisted
    :ptype extraction_collection: ScrapeExtractionCollection
    :param model_id: the enrichment model
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :return: the updated, re-persisted row
    :rtype: ScrapeExtraction
    """
    row = extraction.to_dict()
    try:
        notes = await run_enrichment(html, extraction.structured_fields, model_id=model_id, api_key=api_key)
    except EnrichmentFailedError as exc:
        row["enrichment_notes"] = None
        row["enrichment_status"] = "failed"
        row["enrichment_failure"] = exc.reason
    else:
        row["enrichment_notes"] = notes
        row["enrichment_status"] = "enriched"
        row["enrichment_failure"] = None
    updated = extraction_collection.create(row)
    await extraction_collection.save_entity(updated)
    return updated
