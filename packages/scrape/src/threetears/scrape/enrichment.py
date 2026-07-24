"""Secondary enrichment pass -- free-form LLM notes, kept separate from structured data.

A second, separate LLM call over the rendered page, capturing free-form
metadata/context the structured extraction schema has no field for (e.g.
ambiguous wording, unusual formatting, nearby related information). This is
deliberately NOT validated structured data -- it's LLM commentary, stored
in ``ScrapeExtraction.enrichment_notes``, kept distinct from
``structured_fields`` so consumers can always tell the two apart (see
``scrape-data-model.md``'s ``enrichment_notes`` field).

Zero faidh imports (see ``scrape/__init__.py``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field as PydanticField
from threetears.models import LlmPurpose
from threetears.observe import get_logger

from .collections import ScrapeExtraction, ScrapeExtractionCollection
from .llm_retry import bounded_retry_structured_call

__all__ = ["DEFAULT_ENRICHMENT_MODEL_ID", "enrich_extraction", "run_enrichment"]

log = get_logger(__name__)

# Same reliability posture as extraction.py / eval_loop.py / query_agent/matching.py.
DEFAULT_ENRICHMENT_MODEL_ID = "deepseek/deepseek-chat-v3-0324"

_ENRICHMENT_TIMEOUT_SECONDS = 30
_ENRICHMENT_ATTEMPTS = 6
_ENRICHMENT_BACKOFF_SECONDS = 2.0
_MAX_HTML_CHARS_IN_PROMPT = 12000


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

    Same bounded-retry shape as ``extraction.generate_candidates`` /
    ``eval_loop``'s judge call: never raises, returns an empty dict only
    after every attempt fails (an honest "nothing to add" degrade, not a
    crash).

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
    :return: free-form key -> note pairs; empty on total failure or genuinely nothing to add
    :rtype: dict[str, str]
    """
    prompt = _build_enrichment_prompt(html, structured_fields)
    result = await bounded_retry_structured_call(
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
        degraded_to="no notes",
    )
    return {} if result is None else result.notes


async def enrich_extraction(
    extraction: ScrapeExtraction,
    html: str,
    *,
    extraction_collection: ScrapeExtractionCollection,
    model_id: str = DEFAULT_ENRICHMENT_MODEL_ID,
    api_key: str,
) -> ScrapeExtraction:
    """Run the enrichment pass over *html* and persist the result onto *extraction*'s row.

    Only ``enrichment_notes`` changes -- ``structured_fields`` (and every
    other field) is carried through unmodified, so the eval loop's already-
    validated data is never touched by this second, separate LLM pass.

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
    notes = await run_enrichment(html, extraction.structured_fields, model_id=model_id, api_key=api_key)
    row = extraction.to_dict()
    row["enrichment_notes"] = notes
    updated = extraction_collection.create(row)
    await extraction_collection.save_entity(updated)
    return updated
