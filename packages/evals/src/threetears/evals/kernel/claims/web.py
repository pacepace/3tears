"""Checking a nominated claim on the web: the web check of claims grading, part of its truth step.

The truth step decides a world claim against the case's evidence alone (:mod:`~threetears.evals.kernel.claims.verification`).
A claim no case evidence bears on that the judge doubts is *nominated*, by the judge's suspicion and nothing else: the
documents the candidate was given or fetched never nominate a claim and are never shown here. A criterion with a web
budget (:class:`~threetears.evals.schema.models.WebBudget`) checks its nominated claims on the web, and only those, so
the web never overrules case evidence: a claim case evidence supports or contradicts is never searched.

**One check per nominated claim, in claim order, up to the cell's budget.** For each, one call writes at most
``max_queries_per_claim`` searches from the claim and its excerpt alone; each search goes to the host's port
(:class:`~threetears.evals.schema.WebSearch`), and the first ``results_per_query`` results of each are read, a page
read once however many searches return it. Code cuts windows from each page, as the verify step cuts passages from a
document (:func:`~threetears.evals.kernel.claims.verification.select_spans`): tiles of ``window_chars``, ranked by the
claim's anchors, at most ``page_chars`` of each page. One more call reads the numbered windows and answers ``true``,
``false`` or ``inconclusive``. A claim past the cell's budget is not checked (``skipped_cap``), and neither is one the
run's cost cap or the cell's deadline leaves unchecked: what was not checked reads ``unverifiable``, never false.

**A decisive answer quotes a window it was shown**, under the rule every claim's quote follows
(:func:`~threetears.evals.kernel.claims.verification.check_quote`): one sentence or row of that window, verbatim, at
most :data:`~threetears.evals.schema.models.CLAIM_QUOTE_CHARS` characters, with the words naming the claim's subject
and those stating the value inside it and in one sentence. What is stored is the window's own slice. An answer that
fails is asked again alone, naming why, within the step's own repair budget, as each verify step's is. The prompt asks
for the windows and nothing else: what the model remembers never decides a claim.

**What is read is stored, and never searched again.** Each check's searches, every result's address, the windows read
and the deciding quote are one :class:`~threetears.evals.schema.models.WebCheckRecord` on the grade. Every call is
recorded as a :class:`~threetears.evals.schema.models.ClaimCall` (``web`` for a search's writing and its reading,
``repair`` for an answer asked again) and folded into the cell's ``judge`` ledger; the searches are billed by the
caller's port.

**Failures.** A reply cut short, unreadable after its attempts, or past its repairs fails as the truth step fails
(:class:`~threetears.evals.kernel.claims.failures.ClaimGradingFailed`), costing the cell the criterion. What the port
raises is the caller's to classify (``port_error``). A cost-cap refusal of a call or a search stops the web step: the
claim it struck and every later one read ``skipped_cap``, and the truth already found stands.

Engine-internal: the claims grading of the judge phase builds and calls it, and nothing here is exported.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal

from pydantic import Field, ValidationError

from threetears.evals.kernel.authored import strict_response_format
from threetears.evals.kernel.capped_completion import call_ceiling
from threetears.evals.kernel.claims.extraction import ExtractedClaim, claim_call, echo_chars, filler
from threetears.evals.kernel.claims.failures import ClaimGradingFailed
from threetears.evals.kernel.claims.verification import (
    CheckedQuote,
    WebGraded,
    anchors_of,
    check_quote,
    quote_check_definition,
    select_spans,
)
from threetears.evals.kernel.provider import describe_incomplete_completion, extract_json
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.claims import Segment
from threetears.evals.schema.completion import CompletionClient
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    CLAIM_QUOTE_CHARS,
    ClaimCall,
    PageBudget,
    RepairBudget,
    WebBudget,
    WebCheckRecord,
    WebWindow,
)
from threetears.evals.schema.web_search import WebPage

#: How many times one call is asked in full: the first and two more when a finished reply will not parse.
_ATTEMPTS: Final = 3

#: The ledger every web call is billed to: it grades the candidate's output, as a judge does.
_ROLE: Final = "judge"

#: The most characters of a page's title a reading call is shown and the grade stores: what bounds a call's price
#: before the page exists.
_TITLE_CHARS: Final = 200

#: A ref as wide as any a reading call numbers: what a priced prompt prints where a window's ref will stand.
_WIDEST_REF: Final = 999_999_999

_QUERY_SYSTEM: Final = (
    "You write web searches that would find a page stating whether a factual claim is true. You are given the claim "
    "and the words it is said in.\n"
    "- Write at most {most} searches, each a short query a search engine reads, naming the claim's subject and what "
    "it says of it.\n"
    "- Write each search once.\n\n"
    "Answer with JSON only, in the shape the response format states."
)

_VERDICT_RULES: Final = (
    # one line: the only placeholder is CLAIM_QUOTE_CHARS, an int; every rule's line is written here in full
    "Read the claim against the numbered windows of web pages given, and nothing else: not what you remember.\n"
    "- outcome true: a window states this same fact about this same subject.\n"
    "- outcome false: a window states a different value for this same subject and relation.\n"
    "- outcome inconclusive: no window states it, or the windows disagree. Give ref, quote, subject and value as null.\n"
    "- Bind the value to its subject. A window that states a value for a neighbouring subject (another item, edition, "
    "date, person or place) bears on nothing here, even when its value matches the claim's.\n"
    "- An omission contradicts nothing: a window that does not state the claim's fact is no answer.\n"
    "- For true or false, give ref: the window's ref (W1, W2, ...); quote: the one sentence or table row of that "
    f"window that states it, copied exactly as it is written, at most {CLAIM_QUOTE_CHARS} characters, never "
    "shortened with an ellipsis; subject: the words of that quote that name the claim's subject; and value: the words "
    "of that quote that state the value at issue."
)

_VERDICT_SYSTEM: Final = (
    "You check one factual claim against numbered windows cut from web pages.\n\n"
    + _VERDICT_RULES
    + "\n\nAnswer with JSON only, in the shape the response format states."
)

_REPAIR_SYSTEM: Final = (
    "You correct your answer for one claim checked against numbered windows of web pages. You are given the windows, "
    "the claim, your answer and what is wrong with it. Answer again with that fixed.\n\n"
    + _VERDICT_RULES
    + "\n\nAnswer with JSON only, in the shape the response format states."
)

_GUIDANCE_HEADING: Final = "\n\n# Notes on reading claims here\n\n"

_CLAIM_TEMPLATE: Final = 'The claim: {claim}\nIt is said as: "{excerpt}"'

_QUERY_USER_TEMPLATE: Final = "# The claim\n\n{claim}"

_WINDOW_TEMPLATE: Final = "{ref}. From {title}{snippet}:\n{text}"

_SNIPPET_NOTE: Final = " (the search result's snippet only)"

_VERDICT_USER_TEMPLATE: Final = "# Windows\n\n{windows}\n\n# The claim\n\n{claim}"

_AGAIN_TEMPLATE: Final = "{user}\n\n# Your last reply\n\nIt could not be used: {failure}. Answer again, in full."

_REPAIR_TEMPLATE: Final = (
    "# Windows\n\n{windows}\n\n# The claim\n\n{claim}\n\n# Your answer\n\n{answer}\n\n# What is wrong\n\n{failure}"
)

#: Every failure a web call is told, as the templates it is written from: what the judge is shown, so part of the
#: identity. A quote's own failures are :func:`~threetears.evals.kernel.claims.verification.check_quote`'s.
_FAULTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "shape": "it is not the shape the response format states ({problems})",
        "no_queries": "it writes no search; write at least one",
        "too_many": "it writes {count} searches; write at most {most}",
        "blank": "search {place} is blank; each search names what it looks for",
        "twice": "it writes {query!r} more than once; write each search once",
        "no_ref": "its answer {outcome} names no window; a true or false answer quotes the window that states it",
        "unknown_ref": "it cites {ref}, which is no window shown here; cite the refs listed under Windows",
        "correction": "{failure}; and the last correction could not be used: {why}",
    }
)

_NULLABLE_STRING: Final[dict[str, Any]] = {"anyOf": [{"type": "string"}, {"type": "null"}]}

_VERDICT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["true", "false", "inconclusive"]},
        "ref": _NULLABLE_STRING,
        "quote": _NULLABLE_STRING,
        "subject": _NULLABLE_STRING,
        "value": _NULLABLE_STRING,
    },
}

#: A host's search, as the web step calls it: one query in, its results out, billed and bounded by the caller.
type Search = Callable[[str], Awaitable[Sequence[WebPage]]]

#: The unit a search is billed in, as a ``judge`` row of a cell's spend names it (``provider_unit``).
QUERY_UNIT: Final = "query"


@dataclass(frozen=True)
class SearchBill:
    """One search a claims grade sent to a host's port, as the cell's ``judge`` spend counts it: a call of the port,
    one ``query`` unit, and no tokens.

    :ivar provider: The port's identity.
    :ivar cost_usd: What it cost; None when nothing priced it.
    :ivar price_source: Where ``cost_usd`` came from; None with it.
    """

    provider: str
    cost_usd: float | None
    price_source: str | None


class _Queries(EvalBaseModel):
    queries: list[str]


class _Verdict(EvalBaseModel):
    outcome: Literal["true", "false", "inconclusive"]
    ref: str | None = Field(default=None)
    quote: str | None = Field(default=None)
    subject: str | None = Field(default=None)
    value: str | None = Field(default=None)


@dataclass(frozen=True)
class WebCheck:
    """What the web step left for one cell.

    :ivar calls: Every call it made, in order, after the calls before it in the grade.
    :ivar records: One record per claim it checked, in claim order.
    :ivar outcomes: What each nominated claim's check left, by its position, with each deciding call's place in the
        grade's calls: what :func:`~threetears.evals.kernel.claims.verification.graded_claims` reads.
    """

    calls: tuple[ClaimCall, ...]
    records: tuple[WebCheckRecord, ...]
    outcomes: Mapping[int, WebGraded]


def web_definition(budget: WebBudget, repairs: RepairBudget, guidance: str) -> str:
    """A digest of everything that shapes a web check's verdict: the prompts with the host's guidance, the reply
    schemas, the templates, the failures it names, the quote rule, the budget, the repair budget, the attempts and
    the title cap. No client is read, so a criterion's fingerprint states it from the criterion alone.

    :param budget: The criterion's web budget.
    :param repairs: Its repair budget, which the web step holds on its own.
    :param guidance: The host's notes, appended to every prompt.
    :return: A sha256 hex digest.
    """
    checker = _Prompts(budget=budget, guidance=guidance)
    return canonical_digest(
        {
            "query_system": checker.query_system,
            "verdict_system": checker.verdict_system,
            "repair_system": checker.repair_system,
            "templates": [
                _CLAIM_TEMPLATE,
                _QUERY_USER_TEMPLATE,
                _WINDOW_TEMPLATE,
                _SNIPPET_NOTE,
                _VERDICT_USER_TEMPLATE,
                _AGAIN_TEMPLATE,
                _REPAIR_TEMPLATE,
            ],
            "schemas": [_query_format(), _verdict_format(), _repair_format()],
            "faults": dict(_FAULTS),
            "quotes": quote_check_definition(),
            "budget": budget.model_dump(mode="json"),
            "repairs": repairs.model_dump(mode="json"),
            "attempts": _ATTEMPTS,
            "title_chars": _TITLE_CHARS,
        }
    )


@dataclass(frozen=True)
class _Prompts:
    """The system prompts one criterion's web step sends, its guidance appended."""

    budget: WebBudget
    guidance: str

    def _with_guidance(self, prompt: str) -> str:
        return prompt + _GUIDANCE_HEADING + self.guidance.strip() if self.guidance.strip() else prompt

    @property
    def query_system(self) -> str:
        return self._with_guidance(_QUERY_SYSTEM.format(most=self.budget.max_queries_per_claim))

    @property
    def verdict_system(self) -> str:
        return self._with_guidance(_VERDICT_SYSTEM)

    @property
    def repair_system(self) -> str:
        return self._with_guidance(_REPAIR_SYSTEM)


def _query_format() -> dict[str, Any]:
    schema = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}}}
    return strict_response_format("claim_web_queries", schema)


def _verdict_format() -> dict[str, Any]:
    return strict_response_format("claim_web_verdict", _VERDICT_SCHEMA)


def _repair_format() -> dict[str, Any]:
    return strict_response_format("claim_web_repair", _VERDICT_SCHEMA)


@dataclass
class _Budget:
    """The web step's repair budget as one cell spends it."""

    left: int


@dataclass(frozen=True)
class WebChecker:
    """The web step of one criterion: it checks the claims the judge nominated, on the web, within the cell's budget.

    :ivar client: The judge's completion client, which names its model.
    :ivar budget: The criterion's web budget.
    :ivar repairs: Its repair budget, held by this step on its own.
    :ivar guidance: The host's notes on reading claims in its domain, appended to every prompt.
    """

    client: CompletionClient
    budget: WebBudget
    repairs: RepairBudget
    guidance: str = ""
    _prompts: _Prompts = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_prompts", _Prompts(budget=self.budget, guidance=self.guidance))

    @property
    def definition(self) -> str:
        """:func:`web_definition` of this step's budget, repairs and guidance."""
        return web_definition(self.budget, self.repairs, self.guidance)

    def price_ceiling(self, segments: Sequence[Segment], *, query_usd: float | None) -> float | None:
        """The most the web step can cost a cell whose text is ``segments``, before any claim is listed.

        Per claim checked, up to ``max_checked_per_cell``: the call writing its searches, each asked up to its
        attempts; ``max_queries_per_claim`` searches at ``query_usd``; and the reading call, up to its attempts, over
        the most windows the budget lets it read with their headings. Then the most answers the repair budget lets
        the step ask again, each over the same windows. A claim's text is bounded by the echo allowance of the
        longest segment (:func:`~threetears.evals.kernel.claims.extraction.echo_chars`).

        :param segments: The cell's segments.
        :param query_usd: What one search costs, at most; None when it cannot be said.
        :return: The ceiling in dollars, or None when a search or a call cannot be priced.
        """
        if query_usd is None:
            return None
        budget = self.budget
        longest = max((segment.text for segment in segments), key=len, default="")
        claim = filler(echo_chars(longest))
        echo = filler(echo_chars())
        pages = budget.max_queries_per_claim * budget.results_per_query
        per_page = budget.page_chars // budget.window_chars + 1
        heading = _WINDOW_TEMPLATE.format(
            ref=f"W{_WIDEST_REF}", title=filler(_TITLE_CHARS), snippet=_SNIPPET_NOTE, text=""
        )
        windows = "x" * (pages * (4 * budget.page_chars + per_page * (len(heading.encode("utf-8")) + 2)))
        query = call_ceiling(
            self.client,
            system=self._prompts.query_system,
            user=_AGAIN_TEMPLATE.format(user=_QUERY_USER_TEMPLATE.format(claim=claim), failure=echo),
            response_format=_query_format(),
            attempts=_ATTEMPTS,
        )
        verdict = call_ceiling(
            self.client,
            system=self._prompts.verdict_system,
            user=_AGAIN_TEMPLATE.format(user=_VERDICT_USER_TEMPLATE.format(windows=windows, claim=claim), failure=echo),
            response_format=_verdict_format(),
            attempts=_ATTEMPTS,
        )
        repair = call_ceiling(
            self.client,
            system=self._prompts.repair_system,
            user=_REPAIR_TEMPLATE.format(windows=windows, claim=claim, answer=echo, failure=echo),
            response_format=_repair_format(),
        )
        if query is None or verdict is None or repair is None:
            return None
        per_claim = query + budget.max_queries_per_claim * query_usd + verdict
        return budget.max_checked_per_cell * per_claim + self.repairs.max_items_per_cell * repair

    async def check(
        self,
        claims: Sequence[ExtractedClaim],
        nominated: Sequence[int],
        *,
        search: Search,
        ledger: RoleUsageLedger,
        calls_before: int,
    ) -> WebCheck:
        """Check each nominated claim on the web, in order, within the cell's budget.

        :param claims: The cell's claims, as listed.
        :param nominated: The positions of the claims the judge nominated, each a world claim that restates nothing,
            in claim order.
        :param search: The host's search, billed and bounded by the caller; what it raises, but a cost-cap refusal,
            is not caught.
        :param ledger: The cell's ``judge`` ledger.
        :param calls_before: How many of the grade's calls precede this step's, so each deciding call is a place in
            the grade's calls.
        :return: The calls, the records and each nominated claim's outcome.
        :raises ValueError: ``ledger`` is not the ``judge`` role's, or a nominated position is no world claim that
            restates nothing.
        :raises ClaimGradingFailed: A reply was cut short (``cut_short``), a call's reply did not parse in its
            attempts (``unreadable``), or an answer still failed within the repair budget (``repair_budget_spent``).
            A cost-cap refusal (``cost_cap``) is not raised: the claim it struck and every later one read
            ``skipped_cap``.
        """
        if ledger.role != _ROLE:
            raise ValueError(
                f"a web check's calls are billed to the {_ROLE!r} role, and the ledger is {ledger.role!r}'s"
            )
        cell = tuple(claims)
        # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
        stray = [p for p in nominated if p >= len(cell) or cell[p].scope != "world" or cell[p].restates is not None]
        if stray:
            raise ValueError(
                f"claims {stray} are nominated for the web check, and are no world claims that restate nothing"
            )
        calls: list[ClaimCall] = []
        records: list[WebCheckRecord] = []
        outcomes: dict[int, WebGraded] = {}
        repairs = _Budget(left=self.repairs.max_items_per_cell)
        order = list(nominated)
        for at, position in enumerate(order):
            if at >= self.budget.max_checked_per_cell:
                outcomes[position] = WebGraded("skipped_cap")
                continue
            try:
                record, call = await self._check_one(position, cell[position], search, calls, ledger, repairs)
            except ClaimGradingFailed as failed:
                if failed.loss != "cost_cap":
                    raise
                # Recorded, never swallowed: the cap left this claim and every later one unchecked, which each reads.
                outcomes.update((later, WebGraded("skipped_cap")) for later in order[at:])
                break
            records.append(record)
            outcomes[position] = WebGraded(record.outcome, None if call is None else calls_before + call)
        return WebCheck(calls=tuple(calls), records=tuple(records), outcomes=outcomes)

    async def _check_one(
        self,
        position: int,
        claim: ExtractedClaim,
        search: Search,
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
        repairs: _Budget,
    ) -> tuple[WebCheckRecord, int | None]:
        """One claim's check: its searches, the windows read, and the answer that decided it with its call (None
        where no window was read, and nothing was asked)."""
        stated = _CLAIM_TEMPLATE.format(claim=claim.claim, excerpt=claim.excerpt)
        queries = await self._queries(stated, calls, ledger)
        urls: list[str] = []
        pages: list[WebPage] = []
        read: set[str] = set()
        for query in queries:
            for page in tuple(await search(query))[: self.budget.results_per_query]:
                urls.append(page.url)
                if page.url not in read and page.text.strip():
                    read.add(page.url)
                    pages.append(page)
        windows = self._windows(pages, anchors_of([claim.claim, claim.excerpt]))
        if not windows:
            return WebCheckRecord(claim=position, queries=queries, urls=tuple(urls), outcome="inconclusive"), None
        shown = {f"W{at}": window for at, window in enumerate(windows, start=1)}
        numbered = "\n\n".join(
            _WINDOW_TEMPLATE.format(
                ref=ref,
                title=json.dumps(window.title or window.url, ensure_ascii=False),
                snippet=_SNIPPET_NOTE if window.snippet_only else "",
                text=window.text,
            )
            for ref, window in shown.items()
        )
        answer, call = await self._verdict(numbered, stated, calls, ledger)
        decided, fault = _checked(answer, shown)
        for _round in range(self.repairs.rounds):
            if fault is None:
                break
            if repairs.left < 1:
                break
            repairs.left -= 1
            content, repaired = await self._complete(
                role="repair",
                system=self._prompts.repair_system,
                user=_REPAIR_TEMPLATE.format(
                    windows=numbered,
                    claim=stated,
                    answer=json.dumps(answer.model_dump(mode="json"), ensure_ascii=False),
                    failure=fault,
                ),
                response_format=_repair_format(),
                calls=calls,
                ledger=ledger,
            )
            try:
                again = _Verdict.model_validate(extract_json(content))
            except ValueError as refused:
                fault = _FAULTS["correction"].format(failure=fault, why=_unreadable(refused))
                continue
            answer, call = again, repaired
            decided, fault = _checked(answer, shown)
        if fault is not None:
            raise ClaimGradingFailed(
                f"the web check of claim {position} could not be repaired within the step's repair budget: {fault}",
                loss="repair_budget_spent",
            )
        assert decided is not None
        ref, quote = decided
        window = shown[ref] if ref is not None else None
        record = WebCheckRecord(
            claim=position,
            queries=queries,
            urls=tuple(urls),
            windows=windows,
            outcome=answer.outcome,
            deciding_url=None if window is None else window.url,
            quote=None if quote is None else quote.quote,
            subject_span=None if quote is None else quote.subject,
            value_span=None if quote is None else quote.value,
        )
        return record, call

    def _windows(self, pages: Sequence[WebPage], anchors: frozenset[str]) -> tuple[WebWindow, ...]:
        """The windows read of ``pages``: at most ``page_chars`` of each, in tiles of ``window_chars`` ranked by the
        claim's anchors, adjacent tiles joined, in page order and, within a page, by offset."""
        budget = self.budget
        reading = PageBudget(
            per_document_chars=budget.page_chars, total_chars=budget.page_chars, window_chars=budget.window_chars
        )
        windows: list[WebWindow] = []
        for page in pages:
            (spans,) = select_spans([page.text], anchors, reading)
            windows.extend(
                WebWindow(
                    url=page.url,
                    title=page.title[:_TITLE_CHARS],
                    text=page.text[start:end],
                    snippet_only=page.snippet_only,
                )
                for start, end in spans
                if page.text[start:end].strip()
            )
        return tuple(windows)

    async def _complete(
        self,
        *,
        role: Literal["web", "repair"],
        system: str,
        user: str,
        response_format: dict[str, Any],
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
    ) -> tuple[str, int]:
        """One billed and recorded call's finished reply, and its place among the step's calls.

        :raises ClaimGradingFailed: The reply was cut short.
        """
        completion = await self.client.generate(system=system, user=user, response_format=response_format)
        ledger.add_llm_result(completion)
        calls.append(claim_call(role, completion))
        cut_short = describe_incomplete_completion(completion)
        if cut_short is not None:
            raise ClaimGradingFailed(f"a web check's {role} reply was cut short: {cut_short}", loss="cut_short")
        return completion.content, len(calls) - 1

    async def _queries(self, stated: str, calls: list[ClaimCall], ledger: RoleUsageLedger) -> tuple[str, ...]:
        """The searches written for one claim, asked again in full, naming why, while a reply is not usable.

        :raises ClaimGradingFailed: A reply was cut short, or none of the attempts' replies could be used.
        """
        first = _QUERY_USER_TEMPLATE.format(claim=stated)
        user = first
        refused: ValueError | None = None
        for _attempt in range(_ATTEMPTS):
            content, _ = await self._complete(
                role="web",
                system=self._prompts.query_system,
                user=user,
                response_format=_query_format(),
                calls=calls,
                ledger=ledger,
            )
            try:
                return self._read_queries(content)
            except ValueError as unusable:
                refused = unusable
                user = _AGAIN_TEMPLATE.format(user=first, failure=_unreadable(unusable))
        raise ClaimGradingFailed(
            f"a web check's searches could not be read in {_ATTEMPTS} attempts", loss="unreadable"
        ) from refused

    def _read_queries(self, content: str) -> tuple[str, ...]:
        """:raises ValueError: The reply is not the shape, or its searches are none, too many, blank or repeated."""
        queries = [query.strip() for query in _Queries.model_validate(extract_json(content)).queries]
        most = self.budget.max_queries_per_claim
        if not queries:
            raise ValueError(_FAULTS["no_queries"])
        if len(queries) > most:
            raise ValueError(_FAULTS["too_many"].format(count=len(queries), most=most))
        if blank := [place for place, query in enumerate(queries) if not query]:
            raise ValueError(_FAULTS["blank"].format(place=blank[0]))
        if twice := [query for query in queries if queries.count(query) > 1]:
            raise ValueError(_FAULTS["twice"].format(query=twice[0]))
        return tuple(queries)

    async def _verdict(
        self, numbered: str, stated: str, calls: list[ClaimCall], ledger: RoleUsageLedger
    ) -> tuple[_Verdict, int]:
        """The reading call's answer and its place, asked again in full while a reply will not parse.

        :raises ClaimGradingFailed: A reply was cut short, or none of the attempts' replies parsed.
        """
        first = _VERDICT_USER_TEMPLATE.format(windows=numbered, claim=stated)
        user = first
        refused: ValueError | None = None
        for _attempt in range(_ATTEMPTS):
            content, call = await self._complete(
                role="web",
                system=self._prompts.verdict_system,
                user=user,
                response_format=_verdict_format(),
                calls=calls,
                ledger=ledger,
            )
            try:
                return _Verdict.model_validate(extract_json(content)), call
            except ValueError as unusable:
                refused = unusable
                user = _AGAIN_TEMPLATE.format(user=first, failure=_unreadable(unusable))
        raise ClaimGradingFailed(
            f"a web check's reading could not be read in {_ATTEMPTS} attempts", loss="unreadable"
        ) from refused


def _checked(
    answer: _Verdict, shown: Mapping[str, WebWindow]
) -> tuple[tuple[str | None, CheckedQuote | None] | None, str | None]:
    """An answer checked: an inconclusive one as it stands, a decisive one by its quote of a window it was shown.

    :return: The deciding window's ref and the checked quote (both None for inconclusive), or the failure named.
    """
    if answer.outcome == "inconclusive":
        return (None, None), None
    ref = (answer.ref or "").strip()
    if not ref:
        return None, _FAULTS["no_ref"].format(outcome=answer.outcome)
    if ref not in shown:
        return None, _FAULTS["unknown_ref"].format(ref=ref)
    quote, fault = check_quote(
        ref, quote=answer.quote, subject=answer.subject, value=answer.value, shown=shown[ref].text
    )
    if quote is None:
        return None, fault
    return (ref, quote), None


def _unreadable(refused: ValueError) -> str:
    """Why a reply could not be read, in a line a model can act on."""
    if isinstance(refused, ValidationError):
        problems = "; ".join(
            f"{'/'.join(str(part) for part in error['loc']) or 'the reply'}: {error['msg']}"
            for error in refused.errors()[:5]
        )
        return _FAULTS["shape"].format(problems=problems)
    return str(refused)


__all__: list[str] = []
