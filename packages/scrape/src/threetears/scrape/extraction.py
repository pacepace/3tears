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

import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, Comment
from pydantic import BaseModel
from pydantic import Field as PydanticField
from threetears.models import LlmPurpose
from threetears.observe import get_logger

from .llm_retry import bounded_retry_structured_call

__all__ = [
    "DEFAULT_EXTRACTION_MODEL_ID",
    "MAX_HTML_CHARS_IN_PROMPT",
    "FieldSchema",
    "RowValidationResult",
    "ValidationResult",
    "generate_candidates",
    "generate_regex_candidates",
    "generate_regex_row_candidates",
    "generate_row_candidates",
    "html_to_text",
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
        element = soup.select_one(selector)
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
    row_elements = soup.select(row_selector) if row_selector else []
    for row_index, row_element in enumerate(row_elements):
        row_extracted: dict[str, Any] = {}
        row_errors: list[str] = []
        for field_name, expected_type in schema.items():
            selector = field_selectors.get(field_name)
            if not selector:
                row_errors.append(f"row {row_index} {field_name}: no selector proposed")
                continue
            element = row_element.select_one(selector)
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
