"""Ask what a page actually is, at the moment extraction failed on it.

A bot wall returns HTML. It returns 200, it renders, and it contains nothing the stored
selectors match. Every signal available to the eval loop looks identical to a site
redesign, so today a blocked target burns through its failure threshold and spends a full
candidate-generation-plus-judge round learning to extract data from a challenge page,
discarding a recipe that was never broken.

**Why this asks rather than pattern-matches.** The obvious implementation is a marker list:
match the strings Cloudflare's current interstitial contains. That is a hand-written parser
for one vendor's page as it looks this week. Vendors reword and restyle these pages, so a
fixture set captured today specifies nothing about tomorrow, and the failure is silent in
the worst direction: a rotted marker means a blocked page is read as "the site changed" and
the recipe is burned, which is the exact bug being fixed. Asking a model what the page *is*
generalises to a wall it has never seen, because it classifies on meaning rather than
markup.

The trigger is deterministic and free: extraction already failed. This module is never
consulted about a page that worked, and never consulted to rule a wall *out* on a healthy
page.

This module holds the question and the answer shape, nothing else. It reads no database,
fetches no page, and writes nothing. Deciding what to DO with a verdict -- which counter
moves, which row is written, whether regeneration follows -- lives with the code that owns
persistence, in ``eval_loop.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field as PydanticField
from threetears.models import LlmPurpose

from .extraction import MAX_HTML_CHARS_IN_PROMPT, FieldSchema, strip_boilerplate
from .llm_retry import bounded_retry_structured_call

__all__ = [
    "DEFAULT_CLASSIFIER_MODEL_ID",
    "PageVerdict",
    "PageVerdictKind",
    "classify_failed_page",
]

#: What a page that failed extraction turned out to be.
#:
#: ``"blocked"`` is the one that changes behaviour: it means a bot wall, captcha, or
#: human-verification interstitial stands where the content should be, so the stored recipe
#: is not at fault and must not be touched. The rest describe pages we genuinely received:
#: ``"changed"`` is the real content in a shape we no longer recognise, ``"content"`` is the
#: content we expected (so the recipe itself is what is wrong), ``"empty"`` is a page with
#: no records on it at all -- a listing with nothing to list is a normal state, not a
#: failure -- and ``"other"`` is the honest escape hatch (an error page, a redirect
#: landing, a maintenance notice) that deliberately routes the same way ``"content"`` does
#: rather than inventing a response for a case nobody has characterised.
PageVerdictKind = Literal["content", "changed", "blocked", "empty", "other"]

#: Same model and reliability posture as the candidate-comparison judge. This is a reading
#: task over one page, not a generation task, so it needs no more capability than judging
#: already does.
DEFAULT_CLASSIFIER_MODEL_ID = "deepseek/deepseek-chat-v3-0324"

_CLASSIFIER_TIMEOUT_SECONDS = 30
_CLASSIFIER_ATTEMPTS = 6
_CLASSIFIER_BACKOFF_SECONDS = 2.0


class PageVerdict(BaseModel):
    """Forced response shape for the "what is this page" call."""

    kind: PageVerdictKind = PydanticField(
        description=(
            "what this page actually is: 'blocked' for a bot wall, captcha, or "
            "human-verification interstitial standing where content should be; 'changed' for "
            "the content we wanted in a restructured or redesigned form; 'content' for the "
            "content we wanted, substantially as described; 'empty' for the right page with "
            "no records on it at all; 'other' for anything else, such as an error page, a "
            "maintenance notice, or a login screen that is not a bot challenge"
        )
    )
    evidence: str = PydanticField(
        description=(
            "one sentence quoting or describing what on the page itself justifies that "
            "classification -- what a human reading it would point at"
        )
    )
    confidence: Literal["high", "medium", "low"] = PydanticField(
        description="how certain the classification is, given what the page actually shows"
    )


def build_classification_prompt(html: str, schema: FieldSchema, *, page_status: int | None = None) -> str:
    """Build the "what is this page" question for *html*.

    Gives the model the same view of the page the extraction attempt had (boilerplate
    stripped, truncated to the shared prompt budget) plus the fields we expected to find,
    because "is this the content we wanted" is unanswerable without knowing what was wanted.
    The HTTP status is included when the caller knows it, since a 403 or 503 behind a
    rendered page is real evidence, but it is never the deciding input on its own -- most
    walls return 200.

    :param html: the page extraction just failed against
    :ptype html: str
    :param schema: field_name -> expected Python type, the fields we were looking for
    :ptype schema: FieldSchema
    :param page_status: the HTTP status the page came back with, when the caller knows it
    :ptype page_status: int | None
    :return: the prompt text
    :rtype: str
    """
    truncated = strip_boilerplate(html)[:MAX_HTML_CHARS_IN_PROMPT]
    field_lines = ", ".join(schema.keys())
    status_line = f"The page was served with HTTP status {page_status}.\n\n" if page_status is not None else ""
    return (
        f"We fetched a page expecting to find records with these fields: {field_lines}. "
        f"Extracting those fields from it just failed. Your job is to say what the page "
        f"actually is, so we know whether our extraction rules are wrong or whether we never "
        f"received the content at all.\n\n"
        f"{status_line}"
        f"Page content (may be truncated):\n{truncated}\n\n"
        f"Classify it. A bot wall, captcha, 'verify you are human', 'checking your browser', "
        f"or similar interstitial standing in place of the content is 'blocked' -- judge that "
        f"on what the page is doing, not on which vendor produced it. The content we wanted, "
        f"reorganised or restyled so our old rules miss it, is 'changed'. The content we "
        f"wanted, substantially as described, is 'content'. The right page carrying no records "
        f"at all is 'empty'. Anything else is 'other'. Cite what on the page itself justifies "
        f"your answer."
    )


async def classify_failed_page(
    html: str,
    schema: FieldSchema,
    *,
    api_key: str,
    model_id: str = DEFAULT_CLASSIFIER_MODEL_ID,
    page_status: int | None = None,
    attempts: int = _CLASSIFIER_ATTEMPTS,
    backoff_seconds: float = _CLASSIFIER_BACKOFF_SECONDS,
) -> PageVerdict | None:
    """Ask what *html* is. Never raises; returns ``None`` when the call could not be made.

    Same bounded-retry posture as every other structured call in this package. ``None`` is
    an honest "we could not tell", and callers are expected to treat it as exactly today's
    behaviour rather than as a wall or as a change -- an unanswerable classification must
    never be more destructive than not having asked.

    :param html: the page extraction just failed against
    :ptype html: str
    :param schema: field_name -> expected Python type, the fields we were looking for
    :ptype schema: FieldSchema
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param model_id: the model to classify with
    :ptype model_id: str
    :param page_status: the HTTP status the page came back with, when the caller knows it
    :ptype page_status: int | None
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries
    :ptype backoff_seconds: float
    :return: the verdict, or ``None`` after every attempt failed
    :rtype: PageVerdict | None
    """
    return await bounded_retry_structured_call(
        build_classification_prompt(html, schema, page_status=page_status),
        PageVerdict,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.UTILITY,
        temperature=0.0,
        timeout=_CLASSIFIER_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape page classifier",
        degraded_to="no verdict",
    )
