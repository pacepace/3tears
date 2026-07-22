"""AI-driven extraction-strategy candidate generation + structural validation.

Given a rendered page and a caller-supplied field schema, an LLM proposes N
candidate CSS-selector extraction strategies (``{field_name: css_selector}``
— re-executable against a fresh page without another LLM call, unlike a
generated-code strategy). Each candidate is validated structurally: does its
selector match something, and does the matched text parse as the field's
declared type. Domain-agnostic: this module never hardcodes what a field
means (no WARN-Act-shaped assumptions) — the field schema is supplied by the
caller (e.g. Chunk 5's WARN Act plugin), never stored in the core's own
data model. See ``scrape-data-model.md`` / ``scrape-product-brief.md`` and
``build-plan.md``'s Chunk 02 "Design decisions made during build" note.

Zero faidh imports (see ``scrape/__init__.py``).
"""

from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

from bs4 import BeautifulSoup, Comment
from bs4.element import Tag
from pydantic import BaseModel, create_model
from pydantic import Field as PydanticField
from soupsieve.util import SelectorSyntaxError
from threetears.models import LlmPurpose
from threetears.observe import get_logger

from .llm_retry import bounded_retry_structured_call

__all__ = [
    "DEFAULT_EXTRACTION_MODEL_ID",
    "MAX_HTML_CHARS_IN_PROMPT",
    "DiscoveredField",
    "DiscoverySchemaResult",
    "FieldSchema",
    "RowValidationResult",
    "ValidationResult",
    "apply_row_recipe",
    "discover_candidates",
    "discover_row_candidates",
    "DEFAULT_VISION_MODEL_ID",
    "extract_fields_directly",
    "extract_fields_directly_chunked",
    "extract_fields_from_images",
    "extract_multi_row_fields_from_images",
    "extract_page_images",
    "generate_candidates",
    "generate_regex_candidates",
    "generate_regex_row_candidates",
    "generate_row_candidates",
    "html_to_text",
    "NOTICE_DOCUMENT_CLASS",
    "NoticeDocument",
    "OCR_PAGE_IMAGE_CLASS",
    "split_notice_documents",
    "strip_boilerplate",
    "validate_candidate",
    "validate_regex_candidate",
    "validate_regex_row_candidate",
    "validate_row_candidate",
]

log = get_logger(__name__)

# Same default and reliability posture as query_agent/matching.py's
# DEFAULT_MATCH_MODEL_ID -- one shared, live-measured-reliable choice
# (~50% single-call structured-output failure rate via OpenRouter) rather
# than a second, independently drifting default.
DEFAULT_EXTRACTION_MODEL_ID = "deepseek/deepseek-chat-v3-0324"

_EXTRACTION_TIMEOUT_SECONDS = 30
_EXTRACTION_ATTEMPTS = 6
_EXTRACTION_BACKOFF_SECONDS = 2.0

# Real pages can be far larger than a model's useful context budget; a
# candidate-generation prompt only needs enough of the page to identify
# selectors, not the whole document. Cost/token bound, not a correctness
# requirement -- structural validation (validate_candidate) always runs
# against the FULL html, never the truncated prompt copy.
#
# 30000 (raised from 12000, live finding 2026-07-14): a naive html[:12000]
# never reached real content on real government pages -- Maryland's WARN
# table starts at character 45230 (nav/header/notification-banner chrome
# ahead of it), so the model was proposing plausible-looking but entirely
# hallucinated selectors (table.warn-table, div.warn-records) with zero
# grounding in the actual page, structurally invalid every time. Paired
# with strip_boilerplate below (nav/header/footer/script/style/comments
# removed before truncating) rather than raising the budget alone, so the
# model doesn't have to page through chrome to reach content -- covers
# Maryland's real preamble (~19.6k chars post-strip) plus several real rows
# with margin, without sending all ~80 rows (the model only needs to see
# enough real rows to infer the repeating pattern; validate_row_candidate
# applies it to every row deterministically afterward).
MAX_HTML_CHARS_IN_PROMPT = 30000

_BOILERPLATE_TAGS = ("script", "style", "nav", "header", "footer", "noscript", "svg")

#: field_name -> expected Python type for that field's extracted text.
#: Every entry is required (a domain-agnostic core has no concept of
#: "optional" beyond "the caller didn't ask for this field at all").
FieldSchema = dict[str, type]

_NUMERIC_SUFFIX_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
# Matches a leading number (optional K/M/B suffix), NOT anchored to the end of
# the string -- live finding (Maryland's real Total Employees column, 2026-07-14):
# genuine entries like "5 (Remote workers in MD)", a real count with a trailing
# annotation the source site itself put there. Trailing text after the number is
# discarded, not validated -- this function's job is normalizing a number's
# shorthand, not deciding whether trailing prose is meaningful.
_NUMERIC_TOKEN_PATTERN = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*([kKmMbB])?")


def _normalize_numeric_text(text: str) -> str:
    """Normalize a numeric field's raw text before ``int``/``float`` coercion.

    Different sites report the same kind of number in different shorthand --
    "1,234" vs "1234", "$1,234" vs "1234", "1.2M" vs "1200000", "5 (Remote
    workers in MD)" vs "5" -- so no single site's format has to be guessed
    at selector-discovery time; every numeric field goes through the same
    normalization regardless of source. Strips thousands-separator commas
    and currency symbols, expands a K/M/B magnitude suffix, and takes the
    leading number when trailing annotation follows it. Deterministic, not
    an LLM judgment call -- reliable and free, unlike asking a model to do
    arithmetic.

    :param text: raw matched text for a field declared ``int``/``float`` in the schema
    :ptype text: str
    :return: text ready for ``int()``/``float()`` -- may still fail to parse
        if the source text wasn't actually numeric
    :rtype: str
    """
    cleaned = text.strip().replace(",", "").replace("$", "")
    match = _NUMERIC_TOKEN_PATTERN.match(cleaned)
    if match:
        number, suffix = match.groups()
        cleaned = str(int(float(number) * _NUMERIC_SUFFIX_MULTIPLIERS[suffix.lower()])) if suffix else number
    return cleaned


def _normalize_whitespace_text(text: str) -> str:
    """Collapse runs of whitespace to a single space.

    A real site's own HTML formatting quirk, not a BeautifulSoup artifact --
    live against Maryland's WARN page (Chunk 5): "Dejana    Truck and
    Utility Equipment", multiple literal spaces in the source markup itself.
    """
    return " ".join(text.split())


def _coerce_field_value(text: str, expected_type: type) -> Any:
    """Coerce *text* to *expected_type*, normalizing numeric shorthand first.

    Shared by both the single-record (:func:`validate_candidate`) and
    multi-row (:func:`validate_row_candidate`) structural validators, so
    normalization behaves identically regardless of which path a target uses.

    :param text: the matched element's text content
    :ptype text: str
    :param expected_type: the schema's declared type for this field
    :ptype expected_type: type
    :return: the coerced value
    :rtype: Any
    :raises ValueError: text doesn't parse as *expected_type*
    :raises TypeError: text doesn't parse as *expected_type*
    """
    result: Any
    if expected_type is str:
        result = text
    elif expected_type in (int, float):
        result = expected_type(_normalize_numeric_text(text))
    else:
        result = expected_type(text)
    return result


def _select_one_safe(node: BeautifulSoup | Tag, selector: str) -> Tag | None:
    """``node.select_one(selector)``, treating an invalid CSS selector as "matched nothing."

    Live-discovered (schema-discovery mode, real Maryland WARN page,
    2026-07-15): an LLM-proposed selector is not guaranteed to be valid CSS
    -- a jQuery-ism like ``td:eq(0)`` raises ``SelectorSyntaxError``
    (soupsieve, bs4's own selector engine), an UNCAUGHT exception that
    crashes the entire caller. This was a real, pre-existing gap in this
    function (not new code) -- every candidate-generation path in
    ``eval_loop.py`` calls this with LLM-proposed selectors and was equally
    exposed. Fixed at the source rather than worked around only in the new
    discovery code that surfaced it.

    :param node: the ``BeautifulSoup`` document or a ``Tag`` to search within
    :ptype node: BeautifulSoup | Tag
    :param selector: a CSS selector, not guaranteed to be syntactically valid
    :ptype selector: str
    :return: the first matching element, or ``None`` if nothing matched or the selector was invalid
    :rtype: Tag | None
    """
    try:
        return node.select_one(selector)
    except SelectorSyntaxError:
        return None


def _select_safe(node: BeautifulSoup | Tag, selector: str) -> list[Tag]:
    """``node.select(selector)``, treating an invalid CSS selector as "matched nothing."

    See :func:`_select_one_safe`'s docstring for why this exists.

    :param node: the ``BeautifulSoup`` document or a ``Tag`` to search within
    :ptype node: BeautifulSoup | Tag
    :param selector: a CSS selector, not guaranteed to be syntactically valid
    :ptype selector: str
    :return: every matching element, or ``[]`` if nothing matched or the selector was invalid
    :rtype: list[Tag]
    """
    try:
        return node.select(selector)
    except SelectorSyntaxError:
        return []


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of applying one candidate strategy's selectors to real HTML."""

    valid: bool
    extracted: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def validate_candidate(html: str, strategy: dict[str, str], schema: FieldSchema) -> ValidationResult:
    """Apply *strategy*'s CSS selectors to *html* and structurally validate the result.

    Valid iff every field in *schema* has a selector that matches something
    in *html* and whose matched text parses as that field's declared type.

    :param html: the rendered page's full HTML (never the prompt-truncated copy)
    :ptype html: str
    :param strategy: field_name -> CSS selector string
    :ptype strategy: dict[str, str]
    :param schema: field_name -> expected Python type
    :ptype schema: FieldSchema
    :return: extracted values (only for fields that passed) plus every failure reason
    :rtype: ValidationResult
    """
    soup = BeautifulSoup(html, "html.parser")
    extracted: dict[str, Any] = {}
    errors: list[str] = []
    for field_name, expected_type in schema.items():
        selector = strategy.get(field_name)
        if not selector:
            errors.append(f"{field_name}: no selector proposed")
            continue
        element = _select_one_safe(soup, selector)
        if element is None:
            errors.append(f"{field_name}: selector {selector!r} matched nothing")
            continue
        # separator=" " matters: get_text(strip=True) strips each text node individually
        # but inserts nothing between them, so a <br>-split cell concatenates with no
        # space at all (Chunk 5's live finding, WARN Act's own effective-date column).
        text = _normalize_whitespace_text(element.get_text(" ", strip=True))
        if not text:
            errors.append(f"{field_name}: selector {selector!r} matched an empty element")
            continue
        try:
            coerced = _coerce_field_value(text, expected_type)
        # Unparenthesized multi-exception except is valid Python 3.14+ (PEP 758), not the
        # removed Python 2 `except Type, name:` bind-syntax -- ruff's formatter (this repo
        # targets py314) enforces this exact form and undoes manual parenthesization.
        except ValueError, TypeError:
            errors.append(f"{field_name}: {text!r} does not parse as {expected_type.__name__}")
            continue
        extracted[field_name] = coerced
    return ValidationResult(valid=not errors, extracted=extracted, errors=errors)


@dataclass(frozen=True)
class RowValidationResult:
    """Outcome of applying one candidate row-strategy to real HTML.

    Unlike :class:`ValidationResult`, "valid" is per-row: a row counts only
    if every schema field parsed for that row (the same all-or-nothing-per-
    unit philosophy the single-record path applies per-page, generalized to
    per-row). The candidate as a whole is valid iff the row selector matched
    at least one element AND at least one of those rows fully parsed -- a
    header/spacer row failing doesn't sink an otherwise-good candidate.

    Named ``records``, not ``rows``: the *strategy* is row-based (a CSS
    selector over repeating DOM elements, "row" is an accurate description
    of that), but what comes out is domain-neutral extracted data, whether
    the source was an HTML table, a list of ``<div>`` cards, or anything
    else a ``row_selector`` can match -- "record" is the correct term for
    that, and matches the persisted ``ScrapeExtraction.structured_fields``
    key (``{"records": [...]}``, shared with the single-record path).
    """

    valid: bool
    records: list[dict[str, Any]] = field(default_factory=list)
    total_rows_matched: int = 0
    errors: list[str] = field(default_factory=list)


#: Row-level errors are capped in the returned list -- a badly wrong row
#: selector can match hundreds of elements, and a full per-row error list
#: for all of them is noise, not signal (the judge prompt / log line only
#: needs enough detail to see the shape of the failure).
_MAX_ROW_ERRORS = 20


def validate_row_candidate(html: str, strategy: dict[str, Any], schema: FieldSchema) -> RowValidationResult:
    """Apply *strategy*'s row selector + per-field selectors to *html*.

    *strategy* is ``{"row_selector": css_selector, "field_selectors": {field_name: css_selector}}``
    where each field selector is applied *relative to* each row element
    matched by ``row_selector`` (``row.select_one(...)``, not the whole
    document) -- the multi-row counterpart to :func:`validate_candidate`'s
    single, whole-page selectors.

    :param html: the rendered page's full HTML (never the prompt-truncated copy)
    :ptype html: str
    :param strategy: ``{"row_selector": str, "field_selectors": dict[str, str]}``
    :ptype strategy: dict[str, Any]
    :param schema: field_name -> expected Python type, applied to every row
    :ptype schema: FieldSchema
    :return: every record that fully parsed, plus how many the row selector
        matched and every failure reason (capped)
    :rtype: RowValidationResult
    """
    soup = BeautifulSoup(html, "html.parser")
    row_selector = strategy.get("row_selector")
    field_selectors = strategy.get("field_selectors", {})
    errors: list[str] = [] if row_selector else ["no row_selector proposed"]
    records_out: list[dict[str, Any]] = []
    row_elements = _select_safe(soup, row_selector) if row_selector else []
    for row_index, row_element in enumerate(row_elements):
        row_extracted: dict[str, Any] = {}
        row_errors: list[str] = []
        for field_name, expected_type in schema.items():
            selector = field_selectors.get(field_name)
            if not selector:
                row_errors.append(f"row {row_index} {field_name}: no selector proposed")
                continue
            element = _select_one_safe(row_element, selector)
            if element is None:
                row_errors.append(f"row {row_index} {field_name}: selector {selector!r} matched nothing")
                continue
            text = _normalize_whitespace_text(element.get_text(" ", strip=True))
            if not text:
                row_errors.append(f"row {row_index} {field_name}: selector {selector!r} matched an empty element")
                continue
            try:
                row_extracted[field_name] = _coerce_field_value(text, expected_type)
            except ValueError, TypeError:
                row_errors.append(f"row {row_index} {field_name}: {text!r} does not parse as {expected_type.__name__}")
        if row_errors:
            errors.extend(row_errors[: max(0, _MAX_ROW_ERRORS - len(errors))])
        else:
            records_out.append(row_extracted)
    return RowValidationResult(
        valid=len(records_out) > 0, records=records_out, total_rows_matched=len(row_elements), errors=errors
    )


def apply_row_recipe(html: str, strategy: dict[str, Any]) -> list[dict[str, str | None]]:
    """Replay an already-proven row recipe over *html*, returning raw text per field.

    The non-scoring counterpart to :func:`validate_row_candidate`, for callers
    replaying a strategy the eval loop has ALREADY validated against a fresh
    copy of the page. Identical structural walk -- parse under ``html.parser``
    (the parser the recipe was authored and validated against; ``:nth-child``
    can resolve differently across parsers on malformed tables, and lxml
    synthesizes ``<tbody>`` where the source has none), match each record with
    ``row_selector``, then apply each field's selector RELATIVE to its row.

    Three deliberate divergences from :func:`validate_row_candidate`, all
    following from "the recipe is already proven, and the caller wants the
    text, not a verdict":

    - **No schema, no coercion.** Every value comes back as source text or
      ``None``. A bronze load keeps source text verbatim and defers type
      assertions to a later contract, so coercing here would be a lossy
      transform the caller has to undo -- and a leading-zero identifier or a
      currency string would not survive the round trip.
    - **Partial rows survive.** :func:`validate_row_candidate` drops a row
      when any field fails, because it is SCORING selectors and a
      half-matching row is evidence against the candidate. Here a real record
      with one blank optional cell must not vanish, so a missing or empty
      cell is ``None`` and the row stays.
    - **A bad selector raises.** :func:`validate_row_candidate` treats
      invalid CSS as "matched nothing" (see :func:`_select_one_safe`) because
      it scores LLM-PROPOSED selectors never guaranteed to be valid. A recipe
      reaching this function already cleared that gate, so invalid CSS here
      means a corrupt or hand-edited recipe -- failing loud with the
      offending field beats silently nulling an entire column.

    Only an all-empty row is skipped: a spacer or header element the
    ``row_selector`` still matched carries no addressable data, and emitting
    it as an all-``None`` record would invent a row the page does not have.

    An empty return is data, not an error -- the row selector matching
    nothing (a restructured page, a stale recipe) is a condition only the
    caller can weigh, so it is reported as ``[]`` rather than raised.

    :param html: the rendered page's full HTML
    :ptype html: str
    :param strategy: ``{"row_selector": str, "field_selectors": dict[str, str]}``
    :ptype strategy: dict[str, Any]
    :return: one dict per non-empty matched row, keyed by ``field_selectors``
        in declaration order, values being collapsed source text or ``None``
    :rtype: list[dict[str, str | None]]
    :raises ValueError: *strategy* lacks ``row_selector`` or ``field_selectors``,
        or either contains syntactically invalid CSS
    """
    row_selector = strategy.get("row_selector")
    field_selectors = strategy.get("field_selectors") or {}
    if not row_selector:
        raise ValueError("apply_row_recipe: strategy has no row_selector")
    if not field_selectors:
        raise ValueError("apply_row_recipe: strategy has no field_selectors")

    soup = BeautifulSoup(html, "html.parser")
    try:
        row_elements = soup.select(row_selector)
    except SelectorSyntaxError as exc:
        raise ValueError(f"apply_row_recipe: invalid row_selector {row_selector!r}: {exc}") from exc

    records: list[dict[str, str | None]] = []
    for row_element in row_elements:
        record: dict[str, str | None] = {}
        row_has_value = False
        for field_name, selector in field_selectors.items():
            try:
                element = row_element.select_one(selector)
            except SelectorSyntaxError as exc:
                raise ValueError(
                    f"apply_row_recipe: invalid field_selector {selector!r} for field {field_name!r}: {exc}"
                ) from exc
            text = _normalize_whitespace_text(element.get_text(" ", strip=True)) if element is not None else ""
            record[field_name] = text or None
            row_has_value = row_has_value or bool(text)
        if row_has_value:
            records.append(record)
    return records


class _CandidateStrategy(BaseModel):
    """One proposed extraction strategy within a candidate-generation response."""

    selectors: dict[str, str] = PydanticField(
        description="field_name -> CSS selector string that extracts that field's text from the page"
    )


class _CandidateStrategyList(BaseModel):
    """Forced response shape for the candidate-generation LLM call."""

    candidates: list[_CandidateStrategy] = PydanticField(default_factory=list)


def strip_boilerplate(html: str) -> str:
    """Remove non-content chrome before a prompt truncates *html*.

    Real pages routinely carry tens of KB of nav/header/footer/script/style
    markup ahead of the actual content (live finding, Maryland's WARN page:
    45KB of chrome before its own ``<table>`` even starts) -- truncating the
    raw HTML burns the whole prompt budget on boilerplate the model can't
    use, and it hallucinates plausible-looking selectors instead of reading
    the real page. Never used for structural validation, which always runs
    against the untouched, full ``html`` -- this exists purely to make
    better use of the prompt's own truncation budget.

    :param html: the rendered page's full HTML
    :ptype html: str
    :return: the same document with non-content tags and comments removed
    :rtype: str
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_BOILERPLATE_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    return str(soup)


def _build_candidate_prompt(html: str, schema: FieldSchema, n: int) -> str:
    truncated = strip_boilerplate(html)[:MAX_HTML_CHARS_IN_PROMPT]
    field_lines = "\n".join(f"- {name} ({expected.__name__})" for name, expected in schema.items())
    return (
        f"You are proposing CSS selector strategies to extract structured data from a rendered web page.\n\n"
        f"Fields to extract:\n{field_lines}\n\n"
        f"Page HTML (may be truncated):\n{truncated}\n\n"
        f"Propose {n} DIFFERENT candidate strategies, each a complete set of CSS selectors (one per field "
        f"above) that would extract that field's value as the text content of the matched element. "
        f"Prefer selectors specific enough to match exactly one element. If a field cannot be found at all, "
        f"still propose your best-guess selector for it (structural validation will reject it if it's wrong)."
    )


async def generate_candidates(
    html: str,
    schema: FieldSchema,
    *,
    n: int = 3,
    model_id: str = DEFAULT_EXTRACTION_MODEL_ID,
    api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS,
    backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> list[dict[str, str]]:
    """Ask an LLM for *n* candidate CSS-selector extraction strategies.

    Same bounded-retry shape as ``query_agent/matching.py``'s
    ``_invoke_match_disambiguation``: never raises, returns an empty list
    only after every attempt fails (an honest "no candidates" degrade, not
    a crash -- the eval loop treats this the same as every candidate
    failing structural validation).

    :param html: the rendered page's full HTML
    :ptype html: str
    :param schema: field_name -> expected Python type
    :ptype schema: FieldSchema
    :param n: how many candidate strategies to request
    :ptype n: int
    :param model_id: the candidate-generation model
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :return: proposed strategies, each a field_name -> CSS selector dict; empty on total failure
    :rtype: list[dict[str, str]]
    """
    prompt = _build_candidate_prompt(html, schema, n)
    result = await bounded_retry_structured_call(
        prompt,
        _CandidateStrategyList,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.EXTRACTION,
        temperature=0.2,
        timeout=_EXTRACTION_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape candidate generation",
        degraded_to="no candidates",
    )
    return [] if result is None else [c.selectors for c in result.candidates]


class _RowCandidateStrategy(BaseModel):
    """One proposed row-extraction strategy within a candidate-generation response."""

    row_selector: str = PydanticField(
        description="CSS selector matching every repeating record's container element (e.g. a table row)"
    )
    field_selectors: dict[str, str] = PydanticField(
        description="field_name -> CSS selector, applied RELATIVE TO each row_selector match, "
        "that extracts that field's text from within one row"
    )


class _RowCandidateStrategyList(BaseModel):
    """Forced response shape for the row candidate-generation LLM call."""

    candidates: list[_RowCandidateStrategy] = PydanticField(default_factory=list)


def _build_row_candidate_prompt(html: str, schema: FieldSchema, n: int) -> str:
    truncated = strip_boilerplate(html)[:MAX_HTML_CHARS_IN_PROMPT]
    field_lines = "\n".join(f"- {name} ({expected.__name__})" for name, expected in schema.items())
    return (
        f"You are proposing extraction strategies for a page that lists MANY repeating records "
        f"(e.g. a table with one row per record), not a single record.\n\n"
        f"Fields to extract from EACH record:\n{field_lines}\n\n"
        f"Page HTML (may be truncated):\n{truncated}\n\n"
        f"Propose {n} DIFFERENT candidate strategies. Each strategy is a single CSS selector "
        f"(row_selector) that matches every repeating record's container element, plus one CSS "
        f"selector per field (field_selectors) that will be applied RELATIVE TO each row_selector "
        f"match -- not to the whole page -- to extract that field's text from within one record. "
        f"Numbers may include commas, currency symbols, or K/M/B magnitude suffixes; propose the "
        f"selector for the field regardless of its exact formatting, that normalization is handled "
        f"separately. If a field cannot be found at all, still propose your best-guess selector for "
        f"it (structural validation will reject it if it's wrong)."
    )


async def generate_row_candidates(
    html: str,
    schema: FieldSchema,
    *,
    n: int = 3,
    model_id: str = DEFAULT_EXTRACTION_MODEL_ID,
    api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS,
    backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> list[dict[str, Any]]:
    """Ask an LLM for *n* candidate row-extraction strategies (many records per page).

    Same bounded-retry shape as :func:`generate_candidates` -- never raises,
    returns an empty list only after every attempt fails.

    :param html: the rendered page's full HTML
    :ptype html: str
    :param schema: field_name -> expected Python type, applied to every row
    :ptype schema: FieldSchema
    :param n: how many candidate strategies to request
    :ptype n: int
    :param model_id: the candidate-generation model
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :return: proposed strategies, each ``{"row_selector": str, "field_selectors": dict[str, str]}``;
        empty on total failure
    :rtype: list[dict[str, Any]]
    """
    prompt = _build_row_candidate_prompt(html, schema, n)
    result = await bounded_retry_structured_call(
        prompt,
        _RowCandidateStrategyList,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.EXTRACTION,
        temperature=0.2,
        timeout=_EXTRACTION_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape row candidate generation",
        degraded_to="no candidates",
    )
    if result is None:
        candidates_out = []
    else:
        candidates_out = [
            {"row_selector": c.row_selector, "field_selectors": c.field_selectors} for c in result.candidates
        ]
    return candidates_out


# ===========================================================================
# Schema discovery (scrape-task-03) -- the inverse of generate_candidates/
# generate_row_candidates: no caller-supplied schema in, a discovered field
# list (name, inferred type, selector, sample value) out, validated against
# the real page by the exact same validate_candidate/validate_row_candidate
# this module already uses for the schema-known path. Deliberately NOT
# wired into eval_loop.py's recipe-persistence functions -- discovery is a
# pre-onboarding operation (no ScrapeTarget/recipe exists yet), not a mode
# of the persisted-recipe lifecycle. See
# docs/scrape-task-03-schema-discovery-mode.md's placement-deviation note.
# ===========================================================================

#: Mirrors collections.py's own _FIELD_SCHEMA_TYPE_NAMES (4 entries, same set) --
#: kept independent rather than imported, since extraction.py has zero dependency
#: on collections.py today and this is a 4-entry dict, not worth a new coupling.
#: If a type is ever added to one, add it to the other.
_DISCOVERY_TYPE_NAMES: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}


@dataclass(frozen=True)
class DiscoveredField:
    """One field discovery validated against the real page -- never a hallucinated, unvalidated guess."""

    name: str
    python_type: type
    selector: str
    sample_value: Any


@dataclass(frozen=True)
class DiscoverySchemaResult:
    """Outcome of a discovery run -- plain data, never persisted by this module.

    ``field_schema``/``strategy`` are already shaped exactly as
    ``ScrapeTarget.field_schema``/``ScrapeRecipe.extraction_strategy`` expect --
    a caller who likes the result hands them straight through, no re-derivation.
    """

    validated: bool
    fields: list[DiscoveredField] = field(default_factory=list)
    field_schema: FieldSchema = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    sample_records: list[dict[str, Any]] = field(default_factory=list)


class _DiscoveredFieldProposal(BaseModel):
    """One field an LLM proposes exists on the page, within a discovery candidate."""

    name: str = PydanticField(description="a short, descriptive field name (snake_case)")
    type_name: Literal["str", "int", "float", "bool"] = PydanticField(description="the field's inferred Python type")
    selector: str = PydanticField(description="a CSS selector that extracts this field's text from the page")
    sample_value_hint: str = PydanticField(description="what you observed this field's value looks like on the page")


class _DiscoveredCandidate(BaseModel):
    """One full discovery proposal -- every field this attempt found."""

    fields: list[_DiscoveredFieldProposal] = PydanticField(default_factory=list)


class _DiscoveredCandidateList(BaseModel):
    """Forced response shape for the single-record discovery LLM call."""

    candidates: list[_DiscoveredCandidate] = PydanticField(default_factory=list)


class _DiscoveredRowCandidate(BaseModel):
    """One full row-discovery proposal -- a row selector plus every field found within a row."""

    row_selector: str = PydanticField(
        description="CSS selector matching every repeating record's container element (e.g. a table row)"
    )
    fields: list[_DiscoveredFieldProposal] = PydanticField(default_factory=list)


class _DiscoveredRowCandidateList(BaseModel):
    """Forced response shape for the multi-row discovery LLM call."""

    candidates: list[_DiscoveredRowCandidate] = PydanticField(default_factory=list)


def _build_discovery_prompt(html: str, n: int) -> str:
    truncated = strip_boilerplate(html)[:MAX_HTML_CHARS_IN_PROMPT]
    return (
        f"You are examining a rendered web page to identify every distinct, genuinely useful field of "
        f"structured data on it -- e.g. the columns of a table describing ONE record, or the labeled "
        f"values in a single record's own detail view. Do not propose navigation, menu, footer, or "
        f"unrelated boilerplate as fields.\n\n"
        f"Page HTML (may be truncated):\n{truncated}\n\n"
        f"Propose {n} DIFFERENT candidate field lists (different attempts at identifying the same "
        f"underlying data, not different data). For each field, give a short descriptive name, its "
        f"inferred type (str/int/float/bool only), a CSS selector that extracts its text from the page, "
        f"and a hint at what value you observed. Prefer selectors specific enough to match exactly one "
        f"element."
    )


def _build_row_discovery_prompt(html: str, n: int) -> str:
    truncated = strip_boilerplate(html)[:MAX_HTML_CHARS_IN_PROMPT]
    return (
        f"You are examining a rendered web page that lists MANY repeating records (e.g. a table with "
        f"one row per record) to identify every distinct, genuinely useful field each record has. Do "
        f"not propose navigation, menu, footer, or unrelated boilerplate as fields.\n\n"
        f"Page HTML (may be truncated):\n{truncated}\n\n"
        f"Propose {n} DIFFERENT candidate field lists (different attempts at identifying the same "
        f"underlying data, not different data). For each candidate, give a single CSS selector "
        f"(row_selector) that matches every repeating record's container element, plus for each field a "
        f"short descriptive name, its inferred type (str/int/float/bool only), a CSS selector applied "
        f"RELATIVE TO each row_selector match that extracts its text from within one record, and a hint "
        f"at what value you observed."
    )


def _fields_matching_any_row(
    html: str, row_selector: str, fields: list[_DiscoveredFieldProposal]
) -> list[_DiscoveredFieldProposal]:
    """Which of *fields* extract a real, type-parsing value from at least one row.

    ``validate_row_candidate`` is all-or-nothing PER RECORD (a row counts
    only if every schema field matched) -- feeding it a schema with even one
    bad field would zero out every row's record and silently discard every
    OTHER genuinely good field along with it. This pre-filters independently
    per field first, so a single bad discovery proposal can't sink the good
    ones; the real, unmodified ``validate_row_candidate`` still does the
    actual final validation afterward, just against the survivors only.

    :param html: the rendered page's full HTML
    :ptype html: str
    :param row_selector: candidate CSS selector for each record's container
    :ptype row_selector: str
    :param fields: proposed fields to check independently
    :ptype fields: list[_DiscoveredFieldProposal]
    :return: the subset of *fields* that matched real, parsing text in at least one row
    :rtype: list[_DiscoveredFieldProposal]
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = _select_safe(soup, row_selector)
    survivors = []
    for proposal in fields:
        expected_type = _DISCOVERY_TYPE_NAMES[proposal.type_name]
        for row in rows:
            element = _select_one_safe(row, proposal.selector)
            if element is None:
                continue
            text = _normalize_whitespace_text(element.get_text(" ", strip=True))
            if not text:
                continue
            try:
                _coerce_field_value(text, expected_type)
            except ValueError, TypeError:
                continue
            survivors.append(proposal)
            break
    return survivors


def _best_discovery_result(
    proposals_and_validations: list[tuple[list[_DiscoveredFieldProposal], ValidationResult]],
) -> DiscoverySchemaResult:
    """Pick the candidate whose validation kept the most fields -- no judge step.

    Unlike the schema-known path, there's no external ground truth to judge
    semantic correctness against (the LLM invented the fields) -- "most
    fields validated" is the honest, objective, comparable signal, the same
    kind of tiebreak ``_regenerate_row_recipe``'s own ``needs_review``
    fallback already uses.
    """
    best: DiscoverySchemaResult = DiscoverySchemaResult(validated=False)
    for proposals, validation in proposals_and_validations:
        if len(validation.extracted) <= len(best.fields):
            continue
        kept = [p for p in proposals if p.name in validation.extracted]
        best = DiscoverySchemaResult(
            validated=True,
            fields=[
                DiscoveredField(
                    name=p.name,
                    python_type=_DISCOVERY_TYPE_NAMES[p.type_name],
                    selector=p.selector,
                    sample_value=validation.extracted[p.name],
                )
                for p in kept
            ],
            field_schema={p.name: _DISCOVERY_TYPE_NAMES[p.type_name] for p in kept},
            strategy={p.name: p.selector for p in kept},
            sample_records=[validation.extracted],
        )
    return best


async def discover_candidates(
    html: str,
    *,
    n: int = 3,
    model_id: str = DEFAULT_EXTRACTION_MODEL_ID,
    api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS,
    backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> DiscoverySchemaResult:
    """Discover a single-record field schema from *html* -- no caller-supplied schema.

    The inverse of :func:`generate_candidates`: instead of proposing
    selectors for known fields, an LLM proposes field names/types/selectors
    it finds on the page, each validated the same way
    :func:`validate_candidate` validates every other candidate. Never
    raises; returns ``validated=False`` with no fields if every proposal
    validates zero fields.

    :param html: the rendered page's full HTML
    :ptype html: str
    :param n: how many discovery attempts to request
    :ptype n: int
    :param model_id: the discovery model
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :return: the best-validating discovery, or an honest empty result
    :rtype: DiscoverySchemaResult
    """
    prompt = _build_discovery_prompt(html, n)
    result = await bounded_retry_structured_call(
        prompt,
        _DiscoveredCandidateList,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.EXTRACTION,
        temperature=0.2,
        timeout=_EXTRACTION_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape schema discovery",
        degraded_to="no discovered fields",
    )
    if result is None:
        return DiscoverySchemaResult(validated=False)
    proposals_and_validations = []
    for candidate in result.candidates:
        schema = {f.name: _DISCOVERY_TYPE_NAMES[f.type_name] for f in candidate.fields}
        strategy = {f.name: f.selector for f in candidate.fields}
        proposals_and_validations.append((candidate.fields, validate_candidate(html, strategy, schema)))
    return _best_discovery_result(proposals_and_validations)


async def discover_row_candidates(
    html: str,
    *,
    n: int = 3,
    model_id: str = DEFAULT_EXTRACTION_MODEL_ID,
    api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS,
    backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> DiscoverySchemaResult:
    """Discover a multi-row field schema from *html* -- no caller-supplied schema.

    The row counterpart to :func:`discover_candidates` -- the inverse of
    :func:`generate_row_candidates`. ``sample_records`` holds up to every
    validated record from the winning candidate (not capped here --
    ``RowValidationResult`` already caps error detail, not record count).

    :param html: the rendered page's full HTML
    :ptype html: str
    :param n: how many discovery attempts to request
    :ptype n: int
    :param model_id: the discovery model
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :return: the best-validating discovery, or an honest empty result
    :rtype: DiscoverySchemaResult
    """
    prompt = _build_row_discovery_prompt(html, n)
    result = await bounded_retry_structured_call(
        prompt,
        _DiscoveredRowCandidateList,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.EXTRACTION,
        temperature=0.2,
        timeout=_EXTRACTION_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape row schema discovery",
        degraded_to="no discovered fields",
    )
    if result is None:
        return DiscoverySchemaResult(validated=False)
    best: DiscoverySchemaResult = DiscoverySchemaResult(validated=False)
    for candidate in result.candidates:
        survivors = _fields_matching_any_row(html, candidate.row_selector, candidate.fields)
        if not survivors:
            continue
        schema = {f.name: _DISCOVERY_TYPE_NAMES[f.type_name] for f in survivors}
        strategy = {"row_selector": candidate.row_selector, "field_selectors": {f.name: f.selector for f in survivors}}
        validation = validate_row_candidate(html, strategy, schema)
        if not validation.records:
            # _fields_matching_any_row only confirms each field matches SOME row
            # independently -- two sparse/disjoint fields can each survive that check
            # yet never co-occur in any single row (Critic-caught, chunk review): under
            # validate_row_candidate's all-or-nothing-per-record semantics that means
            # zero real records despite nonzero survivors. A candidate that produces no
            # coherent full record is genuinely unusable -- skip it rather than accept a
            # "validated" result no real record backs.
            continue
        if len(survivors) <= len(best.fields):
            continue
        best = DiscoverySchemaResult(
            validated=True,
            fields=[
                DiscoveredField(
                    name=f.name,
                    python_type=_DISCOVERY_TYPE_NAMES[f.type_name],
                    selector=f.selector,
                    sample_value=validation.records[0].get(f.name) if validation.records else None,
                )
                for f in survivors
            ],
            field_schema=schema,
            strategy=strategy,
            sample_records=validation.records,
        )
    return best


# ===========================================================================
# Regex-based extraction (2026-07-14, text-block strategy) -- a second
# extraction-strategy SHAPE, not a bypass of the eval loop: the same
# propose -> structurally-validate -> judge -> persist cycle above, for
# pages whose real content is prose/list text with no <table> (or pipe-
# table-shaped) structure a CSS selector could ever match against.
# Pennsylvania's real WARN page (rejected in Chunk 12: "fields present but
# as a text-block list, not a literal <table>") is the concrete driver --
# the CSS-selector candidate generator had no strategy shape to propose a
# candidate in AT ALL for that page, not a page it tried and failed on.
# ===========================================================================


def html_to_text(html: str) -> str:
    """Convert HTML to plain, line-structured text for regex-based extraction.

    Regex extraction operates against rendered TEXT, not raw markup --
    markup tags interspersed within a field's own value (a ``<br>``, a
    ``<span>``) would otherwise fragment a pattern written assuming
    continuous prose. Boilerplate-stripped first (matching
    :func:`generate_candidates`'s own prompt-truncation precedent), so nav/
    header/footer/script noise never becomes part of a proposed pattern's
    surrounding context. Line breaks are preserved (not collapsed to a
    single space like :func:`_normalize_whitespace_text`'s per-field
    normalization) -- a text-block page's own paragraph/record boundaries
    are real structure a regex pattern needs to anchor on, not noise to
    discard; only now-empty lines (whitespace-only text nodes BeautifulSoup's
    own ``get_text`` can produce between block-level tags) are dropped.

    :param html: the rendered page's full HTML
    :ptype html: str
    :return: plain text, one rendered line per output line
    :rtype: str
    """
    soup = BeautifulSoup(strip_boilerplate(html), "html.parser")
    lines = [line.strip() for line in soup.get_text("\n", strip=True).split("\n")]
    return "\n".join(line for line in lines if line)


#: The ``"per_document"`` StrategyType's (see ``eval_loop.py``) half of a shared
#: contract with :class:`~threetears.scrape.drivers.multi_document.MultiDocumentDriver`:
#: that driver wraps each successfully fetched document in ``<div class="notice">...
#: </div>`` when combining N documents into one page; :func:`split_notice_documents`
#: is the other half, splitting that same combined page back into one plain-text block
#: per document. A plain string constant (not an import of the driver module) --
#: ``extraction.py``/``eval_loop.py`` stay driver-agnostic, the driver depends on this
#: constant, never the reverse (drivers already depend on core utilities, e.g.
#: ``multi_document.py`` importing ``api.py``'s ``_resolve_path``; the reverse would be new).
NOTICE_DOCUMENT_CLASS = "notice"

#: Same driver-agnostic-core convention as :data:`NOTICE_DOCUMENT_CLASS` -- the other
#: half of :class:`~threetears.scrape.drivers.document`'s own embedded-page-image
#: contract (scrape-task-06): when a document needed OCR, its combined-page ``<div>``
#: contains one ``<img class="ocr-page-image">`` per rendered page, read back out by
#: :func:`split_notice_documents` for the vision-extraction path.
OCR_PAGE_IMAGE_CLASS = "ocr-page-image"


class NoticeDocument(NamedTuple):
    """One document recovered from a per_document-strategy combined page.

    :func:`split_notice_documents`'s own return shape -- carries both the plain
    text (the fast/cheap path every document already had) and, when the document
    needed OCR, the embedded page images (the vision path scrape-task-06 added).
    """

    text: str
    was_ocr: bool
    images: list[bytes]


def split_notice_documents(html: str) -> list[NoticeDocument]:
    """Split a per_document-strategy combined page back into one document per notice.

    Used by the ``"per_document"`` StrategyType (see :mod:`threetears.scrape.eval_loop`)
    instead of a page-wide regex/CSS pattern -- some real multi-document targets
    (e.g. Hawaii/West Virginia's WARN Act letters, one independently-worded letter
    per employer) share no boilerplate a single pattern could ever generalize
    across, so each document needs its own fresh extraction call, not a cached recipe.

    :param html: the combined page's full HTML (see :data:`NOTICE_DOCUMENT_CLASS`)
    :ptype html: str
    :return: one :class:`NoticeDocument` per discovered document, in page order
    :rtype: list[NoticeDocument]
    """
    soup = BeautifulSoup(html, "html.parser")
    documents: list[NoticeDocument] = []
    for tag in soup.select(f"div.{NOTICE_DOCUMENT_CLASS}"):
        was_ocr = tag.get("data-was-ocr") == "true"
        images: list[bytes] = []
        for img_tag in tag.select(f"img.{OCR_PAGE_IMAGE_CLASS}"):
            src = str(img_tag.get("src", ""))
            _prefix, _sep, encoded = src.partition("base64,")
            if _sep:
                images.append(base64.b64decode(encoded))
        documents.append(NoticeDocument(text=html_to_text(str(tag)), was_ocr=was_ocr, images=images))
    return documents


def extract_page_images(html: str) -> list[bytes]:
    """Decode every embedded page image directly out of *html*, in page order.

    The ``"multi_row_vision"`` StrategyType's own counterpart to
    :func:`split_notice_documents` -- a ``multi_row_vision`` target's page is ONE
    document (a single PDF with many records inside it, e.g. Nevada's master WARN
    table), rendered through the plain :class:`~threetears.scrape.drivers.
    document.DocumentDriver` (``force_images=True``), not
    :class:`~threetears.scrape.drivers.multi_document.MultiDocumentDriver`'s
    per-notice ``<div class="notice">`` wrapping -- so there's no per-document
    split to do, just every ``<img class="ocr-page-image">`` tag in the whole page.

    :param html: the rendered page's full HTML (see :data:`OCR_PAGE_IMAGE_CLASS`)
    :ptype html: str
    :return: one page image per embedded ``<img>`` tag, in document order
    :rtype: list[bytes]
    """
    soup = BeautifulSoup(html, "html.parser")
    images: list[bytes] = []
    for img_tag in soup.select(f"img.{OCR_PAGE_IMAGE_CLASS}"):
        src = str(img_tag.get("src", ""))
        _prefix, _sep, encoded = src.partition("base64,")
        if _sep:
            images.append(base64.b64decode(encoded))
    return images


#: Every regex candidate is compiled with these flags -- MULTILINE so ``^``/
#: ``$`` anchor to line boundaries (a text-block record's own natural unit),
#: DOTALL so ``.`` can span a record's own internal line breaks (a company
#: name and its date on separate lines within one record) without the LLM
#: needing to know to embed inline flags itself.
_REGEX_FLAGS = re.MULTILINE | re.DOTALL


def validate_regex_candidate(text: str, pattern: str, schema: FieldSchema) -> ValidationResult:
    """Apply *pattern*'s named groups to *text* and structurally validate the result.

    The regex counterpart to :func:`validate_candidate` -- single match
    against the whole page's text (:func:`html_to_text`), one named group
    per schema field, applied once. Valid iff the pattern compiles, matches
    at least once, and every schema field's named group matched non-empty
    text that parses as that field's declared type.

    :param text: the page's plain text (see :func:`html_to_text`), never HTML
    :ptype text: str
    :param pattern: a Python regex with one ``(?P<field_name>...)`` named
        group per *schema* key
    :ptype pattern: str
    :param schema: field_name -> expected Python type
    :ptype schema: FieldSchema
    :return: extracted values (only for fields that passed) plus every failure reason
    :rtype: ValidationResult
    """
    try:
        compiled = re.compile(pattern, _REGEX_FLAGS)
    except re.error as exc:
        result = ValidationResult(valid=False, errors=[f"invalid regex: {exc}"])
    else:
        match = compiled.search(text)
        if match is None:
            result = ValidationResult(valid=False, errors=["pattern matched nothing"])
        else:
            extracted: dict[str, Any] = {}
            errors: list[str] = []
            group_dict = match.groupdict()
            for field_name, expected_type in schema.items():
                raw = group_dict.get(field_name)
                if raw is None:
                    errors.append(f"{field_name}: no named group {field_name!r} in pattern, or it didn't match")
                    continue
                normalized = _normalize_whitespace_text(raw)
                if not normalized:
                    errors.append(f"{field_name}: named group {field_name!r} matched empty text")
                    continue
                try:
                    extracted[field_name] = _coerce_field_value(normalized, expected_type)
                except ValueError, TypeError:
                    errors.append(f"{field_name}: {normalized!r} does not parse as {expected_type.__name__}")
            result = ValidationResult(valid=not errors, extracted=extracted, errors=errors)
    return result


def validate_regex_row_candidate(text: str, pattern: str, schema: FieldSchema) -> RowValidationResult:
    """Apply *pattern*'s named groups to every match in *text*.

    The regex counterpart to :func:`validate_row_candidate` -- *pattern* is
    matched repeatedly (``re.finditer``, one match per record) against the
    page's plain text (:func:`html_to_text`), with one named group per
    schema field. Same all-or-nothing-per-record philosophy: a record counts
    only if every schema field's named group matched and parsed.

    :param text: the page's plain text (see :func:`html_to_text`), never HTML
    :ptype text: str
    :param pattern: a Python regex with one ``(?P<field_name>...)`` named
        group per *schema* key, matching one record per occurrence
    :ptype pattern: str
    :param schema: field_name -> expected Python type, applied to every record
    :ptype schema: FieldSchema
    :return: every record that fully parsed, plus how many the pattern
        matched and every failure reason (capped)
    :rtype: RowValidationResult
    """
    try:
        compiled = re.compile(pattern, _REGEX_FLAGS)
    except re.error as exc:
        result = RowValidationResult(valid=False, errors=[f"invalid regex: {exc}"])
    else:
        matches = list(compiled.finditer(text))
        errors: list[str] = []
        records_out: list[dict[str, Any]] = []
        for match_index, match in enumerate(matches):
            row_extracted: dict[str, Any] = {}
            row_errors: list[str] = []
            group_dict = match.groupdict()
            for field_name, expected_type in schema.items():
                raw = group_dict.get(field_name)
                if raw is None:
                    row_errors.append(f"record {match_index} {field_name}: no named group match")
                    continue
                normalized = _normalize_whitespace_text(raw)
                if not normalized:
                    row_errors.append(f"record {match_index} {field_name}: named group matched empty text")
                    continue
                try:
                    row_extracted[field_name] = _coerce_field_value(normalized, expected_type)
                except ValueError, TypeError:
                    row_errors.append(
                        f"record {match_index} {field_name}: {normalized!r} does not parse as {expected_type.__name__}"
                    )
            if row_errors:
                errors.extend(row_errors[: max(0, _MAX_ROW_ERRORS - len(errors))])
            else:
                records_out.append(row_extracted)
        result = RowValidationResult(
            valid=len(records_out) > 0, records=records_out, total_rows_matched=len(matches), errors=errors
        )
    return result


class _RegexCandidateStrategy(BaseModel):
    """One proposed regex extraction strategy within a candidate-generation response."""

    pattern: str = PydanticField(
        description="a Python regex (re module syntax) with one (?P<field_name>...) named group per field, "
        "matching the ENTIRE record once"
    )


class _RegexCandidateStrategyList(BaseModel):
    """Forced response shape for the single-record regex candidate-generation LLM call."""

    candidates: list[_RegexCandidateStrategy] = PydanticField(default_factory=list)


#: Live-found (Pennsylvania's real WARN page, 2026-07-14): a pattern like
#: ``(?P<employer>[^\n]+)\n(?:[^\n]+\n)*COUNTY:...`` -- a GREEDY repeated
#: group standing in for "any number of lines until the next label" --
#: doesn't stop at the NEAREST following "COUNTY:", it backtracks from the
#: end of the whole text to the LAST "COUNTY:" that still lets the rest of
#: the pattern match, silently jumping hundreds of records ahead (or, with
#: only one overall match found, swallowing the entire rest of the
#: document). The fix is a non-greedy quantifier (``*?``, not ``*``) on any
#: "skip lines until X" group -- non-greedy tries the FEWEST repetitions
#: first, stopping at the nearest match. Included directly in both regex
#: prompts below since this is the single most impactful correctness
#: instruction for a text-block pattern, not an edge case worth omitting.
_REGEX_GREEDY_WARNING = (
    "Prefer '[^\\n]+' over '.+' for a field whose value is on a single line. CRITICAL: if your pattern "
    "needs to skip an unknown number of lines before reaching a label (e.g. '(?:[^\\n]+\\n)*' before "
    "'COUNTY:'), that repetition MUST be non-greedy ('(?:[^\\n]+\\n)*?', with a '?' after the '*') -- a "
    "greedy version does not stop at the nearest occurrence of what follows it, it backtracks from the END "
    "of the whole text to the LAST occurrence that still lets the rest of the pattern match, silently "
    "jumping over many records (or merging them into one enormous match)."
)


def _build_regex_candidate_prompt(text: str, schema: FieldSchema, n: int) -> str:
    truncated = text[:MAX_HTML_CHARS_IN_PROMPT]
    field_lines = "\n".join(f"- {name} ({expected.__name__})" for name, expected in schema.items())
    return (
        f"You are proposing Python regex (re module) strategies to extract structured data from a plain-text "
        f"rendering of a web page that has NO table structure -- the data is in prose or list text.\n\n"
        f"Fields to extract:\n{field_lines}\n\n"
        f"Page text (may be truncated):\n{truncated}\n\n"
        f"Propose {n} DIFFERENT candidate regex patterns. Each pattern must contain exactly one "
        f"(?P<field_name>...) named capture group per field listed above (using the exact field names given), "
        f"and must match the text ONCE, capturing that single record's values. The pattern is compiled with "
        f"re.MULTILINE | re.DOTALL, so '.' matches newlines and '^'/'$' anchor to line boundaries -- you do "
        f"not need to add these flags yourself. {_REGEX_GREEDY_WARNING} If a field cannot be found at all, "
        f"still propose your best-guess named group for it (structural validation will reject it if it's wrong)."
    )


async def generate_regex_candidates(
    text: str,
    schema: FieldSchema,
    *,
    n: int = 3,
    model_id: str = DEFAULT_EXTRACTION_MODEL_ID,
    api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS,
    backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> list[str]:
    """Ask an LLM for *n* candidate single-record regex extraction patterns.

    Same bounded-retry shape as :func:`generate_candidates` -- never raises,
    returns an empty list only after every attempt fails.

    :param text: the page's plain text (see :func:`html_to_text`), never HTML
    :ptype text: str
    :param schema: field_name -> expected Python type
    :ptype schema: FieldSchema
    :param n: how many candidate patterns to request
    :ptype n: int
    :param model_id: the candidate-generation model
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :return: proposed regex pattern strings; empty on total failure
    :rtype: list[str]
    """
    prompt = _build_regex_candidate_prompt(text, schema, n)
    result = await bounded_retry_structured_call(
        prompt,
        _RegexCandidateStrategyList,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.EXTRACTION,
        temperature=0.2,
        timeout=_EXTRACTION_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape regex candidate generation",
        degraded_to="no candidates",
    )
    return [] if result is None else [c.pattern for c in result.candidates]


def _build_regex_row_candidate_prompt(text: str, schema: FieldSchema, n: int) -> str:
    truncated = text[:MAX_HTML_CHARS_IN_PROMPT]
    field_lines = "\n".join(f"- {name} ({expected.__name__})" for name, expected in schema.items())
    return (
        f"You are proposing Python regex (re module) strategies to extract MANY repeating records from a "
        f"plain-text rendering of a web page that has NO table structure -- the data is in prose or list "
        f"text, one block of text per record.\n\n"
        f"Fields to extract from EACH record:\n{field_lines}\n\n"
        f"Page text (may be truncated):\n{truncated}\n\n"
        f"Propose {n} DIFFERENT candidate regex patterns. Each pattern must contain exactly one "
        f"(?P<field_name>...) named capture group per field listed above (using the exact field names given). "
        f"The pattern will be applied repeatedly (re.finditer) against the full text -- EACH match must "
        f"correspond to exactly ONE record. The pattern is compiled with re.MULTILINE | re.DOTALL, so '.' "
        f"matches newlines and '^'/'$' anchor to line boundaries -- you do not need to add these flags "
        f"yourself. {_REGEX_GREEDY_WARNING} CRITICAL: re.finditer's matches are non-overlapping and "
        f"sequential -- if a record has text AFTER your last needed field (a line/paragraph you don't need "
        f"to capture, but which still belongs to this record) and your pattern stops before consuming it, "
        f"the NEXT match will start scanning from partway into that leftover text -- often landing on that "
        f"trailing line and mistaking it for the NEXT record's first field. Your pattern must consume the "
        f"ENTIRE record's text, not just up through your last needed field -- if there's a known trailing "
        f"line/label after your last field, match past it (e.g. with a non-greedy '.*?' or a literal match "
        f"for what that trailing content looks like) even though you don't capture it into a named group. "
        f"Do NOT assume this trailing content is the same fixed number of lines for every record -- a field "
        f"with a variable number of values (e.g. a record affecting multiple counties) can make one record's "
        f"trailing content longer than another's, so prefer an open-ended non-greedy consumer (e.g. "
        f"'(?:[^\\n]+\\n)*?') over a pattern that hardcodes an exact count of trailing lines. CRITICAL: a "
        f"'(?:[^\\n]+\\n)*?' group requires each repetition to START with a non-newline character -- if your "
        f"last named group captured a value that stops mid-line (immediately before that line's own trailing "
        f"newline, not after it), put a literal '\\n' BEFORE the '(?:[^\\n]+\\n)*?' group so it first crosses "
        f"that line's own newline; without it, the group cannot take even its first repetition (it would have "
        f"to start matching at a newline character, which '[^\\n]+' forbids), silently matching zero times no "
        f"matter how much trailing text follows. If you stop your trailing consumer with a lookahead (e.g. "
        f"'(?=...)'), that lookahead MUST check for something that only appears at a real record boundary -- "
        f"a generic condition like 'next line is non-empty' or 'next line starts with a non-whitespace "
        f"character' matches at nearly EVERY line in ordinary text and will stop your consumer far too early, "
        f"one line into the next record; check for an actual label or line shape that only occurs where a "
        f"new record starts. Also prefer "
        f"anchoring the START of your pattern on a stable label that reliably follows your FIRST field in "
        f"every record (e.g. '(?P<employer>[^\\n]+)\\nSome Label:' rather than just '(?P<employer>[^\\n]+)\\n') "
        f"when the text has one -- but check the actual text for whether that label's OWN value sits on the "
        f"same line (label and value separated by a space) or the next line (label alone, value on the line "
        f"after) before writing what comes after the label in your pattern; don't assume one shape without "
        f"looking, and prefer '\\s*' (which crosses a newline) over '[^\\n]+' there if you're not sure. This "
        f"makes matches self-correcting: even if "
        f"a previous match's trailing consumption undershoots and leaves stray text behind, re.finditer's "
        f"forward scan will skip that leftover text (it won't satisfy your start anchor) instead of "
        f"mistaking it for the next record's first field. CRITICAL: a real page's labels are not always used "
        f"100% consistently across every record -- e.g. a record with only one value might still use a plural "
        f"label ('Counties:' instead of 'County:') that a literal 'County:' match won't find at all, causing "
        f"your skip-group to search PAST that entire record into the next one and silently merge two records "
        f"into one match. Check the actual text for label variants (singular/plural, or similar) before "
        f"picking a literal string to anchor on, and match the variants (e.g. 'Counties?:') if you see more "
        f"than one form used for what is semantically the same field. "
        f"If a field cannot be found at all, still propose your best-guess named group for it (structural "
        f"validation will reject it if it's wrong)."
    )


async def generate_regex_row_candidates(
    text: str,
    schema: FieldSchema,
    *,
    n: int = 3,
    model_id: str = DEFAULT_EXTRACTION_MODEL_ID,
    api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS,
    backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> list[str]:
    """Ask an LLM for *n* candidate multi-record regex extraction patterns.

    Same bounded-retry shape as :func:`generate_row_candidates` -- never
    raises, returns an empty list only after every attempt fails.

    :param text: the page's plain text (see :func:`html_to_text`), never HTML
    :ptype text: str
    :param schema: field_name -> expected Python type, applied to every record
    :ptype schema: FieldSchema
    :param n: how many candidate patterns to request
    :ptype n: int
    :param model_id: the candidate-generation model
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :return: proposed regex pattern strings; empty on total failure
    :rtype: list[str]
    """
    prompt = _build_regex_row_candidate_prompt(text, schema, n)
    result = await bounded_retry_structured_call(
        prompt,
        _RegexCandidateStrategyList,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.EXTRACTION,
        temperature=0.2,
        timeout=_EXTRACTION_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape regex row candidate generation",
        degraded_to="no candidates",
    )
    return [] if result is None else [c.pattern for c in result.candidates]


# ===========================================================================
# extract_fields_directly -- no cached pattern, one LLM call per document
# ===========================================================================


def _build_direct_extraction_model(schema: FieldSchema) -> type[BaseModel]:
    """Build a one-off Pydantic model with one optional string field per *schema* key.

    Every field is ``str | None`` (never the schema's own declared type)
    regardless of what the schema asks for -- the LLM returns the RAW text
    it found (or null if genuinely absent), and that raw text goes through
    the exact same :func:`_coerce_field_value` normalization every other
    extraction path already uses (numeric shorthand, whitespace), rather
    than trusting the model's own, less consistent native type coercion.
    """
    fields: dict[str, Any] = {
        name: (
            str | None,
            PydanticField(default=None, description=f"this document's own {name}, or null if not present"),
        )
        for name in schema
    }
    return create_model("DirectExtractionFields", **fields)


def _build_direct_extraction_prompt(text: str, schema: FieldSchema) -> str:
    field_lines = "\n".join(f"- {name} ({expected.__name__})" for name, expected in schema.items())
    truncated = text[:MAX_HTML_CHARS_IN_PROMPT]
    return (
        f"Extract the following fields from this ONE document's own text. This document is "
        f"independently written (e.g. one company's own business letter) -- it shares no "
        f"template with any other document, so extract only what THIS document's own text "
        f"actually says.\n\n"
        f"Fields:\n{field_lines}\n\n"
        f"Document text (may be truncated):\n{truncated}\n\n"
        f"Return each field's raw value exactly as it appears in the text (do not reformat "
        f"dates or numbers). CRITICAL: each field's value must be ONLY the raw text you found "
        f"-- never append your own notes, comments, caveats, or explanations to a value (e.g. "
        f"do NOT return something like 'June 26, 2026\" // Note: the quote is curly in the "
        f"original' -- return exactly 'June 26, 2026' and nothing else). If you have something "
        f"to say about a value (an ambiguity, a formatting quirk), say nothing -- return your "
        f"best raw value or null, never a value with commentary mixed in. Read the ENTIRE "
        f"document text before deciding a field is absent -- e.g. an effective/termination date "
        f"or an affected employee count is often stated in a middle paragraph, a summary table, "
        f"or a schedule near the end, not only in the opening lines. If a field is genuinely not "
        f"present anywhere in this document's text after reading it fully, return null for it "
        f"rather than guessing a plausible-looking value. Two common real-letter shapes to "
        f"watch for: (1) prose sometimes states an explicit total directly (e.g. 'the total "
        f"number of affected employees ... is one') -- ALWAYS prefer that explicit statement "
        f"over adding up any attached position/department breakdown table yourself, especially "
        f"when the table's own scope looks broader than this document's own subject (e.g. a "
        f"nationwide list of job titles attached to a notice about a single state or site) -- "
        f"only sum a breakdown table when NO explicit total is stated anywhere and the table's "
        f"own rows are clearly scoped to this exact document's subject, not a wider company-wide "
        f"list; (2) a document can mention more than one date in connection with the layoff/"
        f"closure (e.g. different phase-out dates for different locations, or a notice date vs. "
        f"a termination date) -- for a field asking for the effective/termination date, prefer "
        f"whichever date is most clearly described as when employment actually ends, and if "
        f"truly no single date applies to everyone, use the LATEST such date mentioned."
    )


#: Live-found (scrape-task-05, a real West Virginia document): every schema field
#: here is ``str | None`` (see :func:`_build_direct_extraction_model`'s own docstring
#: for why), which means a garbage response -- observed live: the model echoed its
#: ENTIRE prompt back into one field's value, with the real answer buried in a
#: trailing ```json code block instead of the forced structured-output shape --
#: still VALIDATES successfully (any string satisfies ``str``), so
#: ``bounded_retry_structured_call``'s own retry-on-failure never triggers; nothing
#: about this is a coercion/type problem :func:`_coerce_field_value` could catch
#: downstream. No genuine field value in this schema's domain (a date, a name, a
#: count) is ever anywhere close to this long -- a generous, cheap sanity ceiling
#: that only rejects obviously-malformed output, not unusually verbose real values.
_MAX_PLAUSIBLE_FIELD_LENGTH = 300


def _is_plausible_direct_extraction(candidate: BaseModel) -> bool:
    """``is_acceptable`` predicate for :func:`extract_fields_directly`'s retry call.

    Rejects (triggers a retry on) a response whose own string values are wildly too
    long to be a genuine field value -- see :data:`_MAX_PLAUSIBLE_FIELD_LENGTH`'s own
    comment for the live-observed failure mode this catches.
    """
    return all(
        not isinstance(value, str) or len(value) <= _MAX_PLAUSIBLE_FIELD_LENGTH
        for value in candidate.model_dump().values()
    )


async def extract_fields_directly(
    text: str,
    schema: FieldSchema,
    *,
    model_id: str = DEFAULT_EXTRACTION_MODEL_ID,
    api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS,
    backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> dict[str, Any] | None:
    """Ask an LLM to extract *schema*'s field values directly from ONE independent document's text.

    No cached pattern, no candidate/judge comparison -- unlike every other
    ``generate_*_candidates`` function in this module, there is nothing to
    learn once and reuse: a :class:`~threetears.scrape.drivers.
    multi_document.MultiDocumentDriver` target whose documents are
    genuinely independently-worded (e.g. Hawaii/West Virginia's real WARN
    Act letters, one freeform letter per employer, live-verified to share
    no boilerplate a single regex/CSS pattern could ever generalize across)
    needs a fresh extraction call on every single document, every poll --
    the eval loop's ``"per_document"`` :data:`~threetears.scrape.eval_loop.
    StrategyType` (see that module) calls this once per document rather
    than once per page.

    :param text: one document's own plain text (see :func:`html_to_text`), never HTML
    :ptype text: str
    :param schema: field_name -> expected Python type
    :ptype schema: FieldSchema
    :param model_id: the extraction model
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :return: field_name -> coerced value for every field the model found AND that
        coerced successfully as *schema* declares (a field it couldn't find, or
        whose text doesn't parse as the declared type, is simply absent from the
        dict -- callers decide whether a partial record counts); ``None`` only on
        total LLM failure (never raises)
    :rtype: dict[str, Any] | None
    """
    model_cls = _build_direct_extraction_model(schema)
    prompt = _build_direct_extraction_prompt(text, schema)
    result = await bounded_retry_structured_call(
        prompt,
        model_cls,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.EXTRACTION,
        temperature=0.1,
        timeout=_EXTRACTION_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape direct per-document field extraction",
        degraded_to="no extraction",
        is_acceptable=_is_plausible_direct_extraction,
    )
    if result is None:
        return None
    return _coerce_direct_extraction_result(result, schema)


def _coerce_direct_extraction_result(result: BaseModel, schema: FieldSchema) -> dict[str, Any]:
    """Shared by :func:`extract_fields_directly` and :func:`extract_fields_from_images` --
    both force the same ``_build_direct_extraction_model``-shaped response (every
    field a plain ``str | None``) and coerce it the same way."""
    extracted: dict[str, Any] = {}
    for name, expected_type in schema.items():
        raw = getattr(result, name)
        if raw is None:
            continue
        normalized = _normalize_whitespace_text(raw)
        if not normalized:
            continue
        try:
            extracted[name] = _coerce_field_value(normalized, expected_type)
        except ValueError, TypeError:
            continue
    return extracted


#: Live-verified (scrape-task-05, real West Virginia and Hawaii documents): a single
#: call asking for every schema field at once is measurably LESS reliable than several
#: smaller calls each asking for fewer fields -- isolated proof: a 2-field-only call
#: succeeded on a real document where that same document's 4-field call returned null
#: for exactly those 2 fields. :func:`extract_fields_directly_chunked` is the fix --
#: split *schema* into chunks of this size, one independent LLM call per chunk. Not a
#: magic number verified across many schema sizes, just the smallest chunk size that
#: showed a real, reproduced improvement -- revisit if a real schema needs otherwise.
_DEFAULT_FIELDS_PER_CALL = 2


async def extract_fields_directly_chunked(
    text: str,
    schema: FieldSchema,
    *,
    model_id: str = DEFAULT_EXTRACTION_MODEL_ID,
    api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS,
    backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
    fields_per_call: int = _DEFAULT_FIELDS_PER_CALL,
) -> dict[str, Any]:
    """Split *schema* into smaller chunks, extract each with its own independent
    :func:`extract_fields_directly` call (run concurrently), merge into one dict.

    The reliability fix :func:`extract_fields_directly` alone couldn't reach -- see
    :data:`_DEFAULT_FIELDS_PER_CALL`'s own comment for the live-reproduced evidence.
    This is what the ``"per_document"`` StrategyType (see
    :mod:`threetears.scrape.eval_loop`) actually calls, not the single-call version
    directly, for exactly this reason.

    :param text: one document's own plain text (see :func:`html_to_text`), never HTML
    :ptype text: str
    :param schema: field_name -> expected Python type
    :ptype schema: FieldSchema
    :param model_id: the extraction model
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures, per chunk
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number), per chunk
    :ptype backoff_seconds: float
    :param fields_per_call: how many schema fields each chunk's own call requests
    :ptype fields_per_call: int
    :return: field_name -> coerced value, the union of every chunk's own result (one
        chunk's total failure only costs that chunk's own fields, never the others --
        the caller decides whether the merged, possibly-partial dict counts as a
        complete record, same contract :func:`extract_fields_directly` itself has)
    :rtype: dict[str, Any]
    """
    items = list(schema.items())
    chunks = [dict(items[i : i + fields_per_call]) for i in range(0, len(items), fields_per_call)]
    chunk_results = await asyncio.gather(
        *(
            extract_fields_directly(
                text,
                chunk_schema,
                model_id=model_id,
                api_key=api_key,
                attempts=attempts,
                backoff_seconds=backoff_seconds,
            )
            for chunk_schema in chunks
        )
    )
    merged: dict[str, Any] = {}
    for chunk_result in chunk_results:
        if chunk_result:
            merged.update(chunk_result)
    return merged


# ===========================================================================
# extract_fields_from_images -- vision extraction path (scrape-task-06)
# ===========================================================================

#: OpenRouter model id for a vision-capable Claude model, reached through the
#: SAME OpenRouter API key every other extraction call in this module already
#: uses -- no new secret needed (live-verified, scrape-task-06). Not pre-
#: registered in threetears-models' own capability registry under the
#: "openrouter" provider (only "anthropic" has it, for a direct-Anthropic-key
#: deployment), so every call here passes ``provider="openrouter"`` explicitly
#: rather than relying on registry auto-resolution.
DEFAULT_VISION_MODEL_ID = "anthropic/claude-sonnet-5"
_VISION_PROVIDER = "openrouter"

#: A vision call reasons over one or more full-page images, not a short text
#: block -- live-measured slower per attempt than a text-only call but still
#: single-digit-to-tens-of-seconds normally; wider than _EXTRACTION_TIMEOUT_SECONDS
#: to give a multi-page document real room before a retry fires.
_VISION_TIMEOUT_SECONDS = 60


def _build_vision_extraction_prompt(schema: FieldSchema) -> str:
    field_lines = "\n".join(f"- {name} ({expected.__name__})" for name, expected in schema.items())
    return (
        "This is one scanned document (one or more page images of the same document). "
        "Extract the following fields directly from what you see in the image(s):\n\n"
        f"{field_lines}\n\n"
        "Return each field's raw value exactly as it appears (do not reformat dates or "
        "numbers). If a field is genuinely not present, illegible, or redacted anywhere "
        "in the image(s), return null for it rather than guessing a plausible-looking value."
    )


async def extract_fields_from_images(
    images: list[bytes],
    schema: FieldSchema,
    *,
    model_id: str = DEFAULT_VISION_MODEL_ID,
    provider: str = _VISION_PROVIDER,
    api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS,
    backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> dict[str, Any] | None:
    """Ask a vision-capable LLM to extract *schema*'s field values directly from page images.

    The scanned-document counterpart to :func:`extract_fields_directly` -- for a
    document that needed OCR (see :class:`~threetears.scrape.drivers.multi_document.
    MultiDocumentDriver`'s own ``data-was-ocr`` convention and :func:`split_notice_documents`),
    reading the ORIGINAL page images directly full-set live-verified dramatically more
    reliable than the OCR'd-text path (scrape-task-06: 10/10 complete records via vision
    vs. 2/10 via OCR'd text across all of a real target's documents) -- OCR can drop or
    garble a narrow numeric table column a vision model reads correctly by seeing the
    actual page layout, and can recover none of a genuinely-redacted value either (a
    real, confirmed finding, not a gap this function claims to close).

    :param images: one or more PNG-encoded page images of the SAME document, in page order
    :ptype images: list[bytes]
    :param schema: field_name -> expected Python type
    :ptype schema: FieldSchema
    :param model_id: the vision-capable model to invoke
    :ptype model_id: str
    :param provider: explicit provider override forwarded to ``create_chat_model``
        (see :data:`_VISION_PROVIDER`'s own comment for why this is needed)
    :ptype provider: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :return: field_name -> coerced value, same partial-result contract as
        :func:`extract_fields_directly`; ``None`` when *images* is empty or every
        attempt failed (never raises)
    :rtype: dict[str, Any] | None
    """
    if not images:
        return None

    from langchain_core.messages import HumanMessage
    from threetears.models import format_vision_content

    content: list[Any] = []
    for image_bytes in images:
        content.extend(format_vision_content(image_bytes, "image/png", "")[:-1])
    content.append({"type": "text", "text": _build_vision_extraction_prompt(schema)})

    model_cls = _build_direct_extraction_model(schema)
    result = await bounded_retry_structured_call(
        [HumanMessage(content=content)],
        model_cls,
        model_id=model_id,
        api_key=api_key,
        provider=provider,
        purpose=LlmPurpose.EXTRACTION,
        temperature=0.1,
        timeout=_VISION_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape vision per-document field extraction",
        degraded_to="no extraction",
        is_acceptable=_is_plausible_direct_extraction,
    )
    if result is None:
        return None
    return _coerce_direct_extraction_result(result, schema)


# ===========================================================================
# extract_multi_row_fields_from_images -- multi-row vision extraction (scrape-task-07)
# ===========================================================================

#: A multi-row vision call reads the same page image(s) as :func:`extract_fields_from_images`
#: but must produce output proportional to the row count (Nevada's real master WARN PDF:
#: 17 records, 8 fields each) rather than one record's worth -- wider than
#: :data:`_VISION_TIMEOUT_SECONDS` to give that larger response real room before a retry fires.
_MULTI_ROW_VISION_TIMEOUT_SECONDS = 120


def _build_multi_row_vision_model(schema: FieldSchema) -> type[BaseModel]:
    """Wrap :func:`_build_direct_extraction_model`'s one-record model in a ``records: list[...]``
    envelope -- the multi-row counterpart to a single ``DirectExtractionFields`` response."""
    record_cls = _build_direct_extraction_model(schema)
    return create_model(
        "MultiRowExtractionFields",
        records=(
            list[record_cls],  # type: ignore[valid-type]
            PydanticField(description="every record found in the table, top to bottom, in the order they appear"),
        ),
    )


def _build_multi_row_vision_extraction_prompt(schema: FieldSchema) -> str:
    field_lines = "\n".join(f"- {name} ({expected.__name__})" for name, expected in schema.items())
    return (
        "This is one or more page images of a SINGLE table containing MANY records (e.g. a "
        "state's master list of WARN Act notices) -- not one record, find EVERY row in the "
        "table. Extract the following fields for EACH row directly from what you see in the "
        "image(s):\n\n"
        f"{field_lines}\n\n"
        "Return one record per row, in the same top-to-bottom order the rows appear in the "
        "table. Do not skip a row, do not merge two rows into one, and do not invent a row "
        "that isn't there. Return each field's raw value exactly as it appears (do not "
        "reformat dates or numbers). If a field is genuinely not present, illegible, or "
        "redacted for a given row, return null for that row's value rather than guessing a "
        "plausible-looking one."
    )


def _is_plausible_multi_row_extraction(candidate: BaseModel) -> bool:
    """``is_acceptable`` predicate for :func:`extract_multi_row_fields_from_images`'s retry
    call -- every record must individually pass :func:`_is_plausible_direct_extraction`'s
    same sanity ceiling (see that function's own docstring for the live-observed failure
    mode this catches), and there must be at least one record at all."""
    records = candidate.model_dump().get("records", [])
    if not records:
        return False
    return all(
        not isinstance(value, str) or len(value) <= _MAX_PLAUSIBLE_FIELD_LENGTH
        for record in records
        for value in record.values()
    )


async def extract_multi_row_fields_from_images(
    images: list[bytes],
    schema: FieldSchema,
    *,
    model_id: str = DEFAULT_VISION_MODEL_ID,
    provider: str = _VISION_PROVIDER,
    api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS,
    backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> list[dict[str, Any]] | None:
    """Ask a vision-capable LLM to extract EVERY record from a table shown in page images.

    The multi-row counterpart to :func:`extract_fields_from_images` -- for a target whose
    single PDF holds many records in one table (not one document = one record, the
    ``"per_document"`` StrategyType's own assumption), used when the table's own structure
    genuinely defeats text-based table extraction. Live-verified against Nevada's real
    master WARN PDF (scrape-task-07): ``find_tables()``'s default ``"lines"`` strategy finds
    only the header (the entire 17-row dataset silently dropped), its ``"text"`` strategy
    mis-splits words/columns (a URL broken mid-string) -- a genuine structural defeat, not a
    scan-quality one (the source PDF is born-digital, has a real text layer). Mississippi's
    superficially similar "multi-row PDF" does NOT need this: its ``find_tables()`` already
    gets correct column boundaries, its own problem (wrapped continuation rows) is a plain
    text-based row-merge fix, filed separately -- proof the two states needed opposite fixes
    despite looking alike, not that "multi-row table" implies vision.

    :param images: one or more PNG-encoded page images of the SAME table, in page order
    :ptype images: list[bytes]
    :param schema: field_name -> expected Python type, applied to every row
    :ptype schema: FieldSchema
    :param model_id: the vision-capable model to invoke
    :ptype model_id: str
    :param provider: explicit provider override forwarded to ``create_chat_model``
    :ptype provider: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :return: one field_name -> coerced value dict per record found, in table order (same
        partial-per-record contract as :func:`extract_fields_directly`); ``None`` when
        *images* is empty or every attempt failed (never raises)
    :rtype: list[dict[str, Any]] | None
    """
    if not images:
        return None

    from langchain_core.messages import HumanMessage
    from threetears.models import format_vision_content

    content: list[Any] = []
    for image_bytes in images:
        content.extend(format_vision_content(image_bytes, "image/png", "")[:-1])
    content.append({"type": "text", "text": _build_multi_row_vision_extraction_prompt(schema)})

    model_cls = _build_multi_row_vision_model(schema)
    result = await bounded_retry_structured_call(
        [HumanMessage(content=content)],
        model_cls,
        model_id=model_id,
        api_key=api_key,
        provider=provider,
        purpose=LlmPurpose.EXTRACTION,
        temperature=0.1,
        timeout=_MULTI_ROW_VISION_TIMEOUT_SECONDS,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        log_label="scrape multi-row vision extraction",
        degraded_to="no extraction",
        is_acceptable=_is_plausible_multi_row_extraction,
    )
    if result is None:
        return None
    return [_coerce_direct_extraction_result(record, schema) for record in getattr(result, "records", [])]
