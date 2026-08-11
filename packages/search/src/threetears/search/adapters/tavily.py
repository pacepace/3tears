"""Tavily, through the injected transport and nothing else.

One provider's API, mapped onto the contract: Tavily's ``/search`` endpoint
in, typed candidates with named scores, published dates, the page content
Tavily already sold us, and per-criterion dispositions out. What this module
deliberately does NOT do is open a client, read an environment variable, or
keep a provider payload as a disclaimed ``raw`` blob.

**Ported, not invented.** The semantics here are discodon's, carried across
rather than redesigned (search-spec.md §3.2): the depth/credit coupling, the
domain scoping, the score coercion that reads a missing score as *unknown*
rather than zero, and the absolute-dates-beat-``time_range`` precedence that
RES-T4M9 established after Tavily 400'd on the combination. What changed is
where each one lives, because this package has typed places for them that
discodon's tool wrapper did not: spend is a value rather than a counter, a
refusal is a class rather than a sentence, and a criterion the provider
cannot express is a reported disposition rather than a dropped parameter.

**What Tavily can and cannot express** (SR-B4, verified against its search
API): it takes ``search_depth`` (``basic``/``advanced``), ``topic``
(``general``/``news``), ``include_domains``/``exclude_domains``,
``max_results``, ``include_raw_content``, and publication-date scoping in
*both* forms -- a relative ``time_range`` and absolute ``start_date`` /
``end_date``. It has no language filter, no engine list, no safesearch
level, no carrier scoping, no resolution filter and no rights filter. So the
adapter pushes down what it can and names the rest ``unsatisfied`` rather
than dropping it (SR-B3).

The two date forms are the interesting case, and the ported one. Tavily
treats relative and absolute publication scoping as mutually exclusive and
answers HTTP 400 to the combination; RES-T4M9 fixed that by *stating*
precedence -- a valid absolute range wins and the relative window is
reported suppressed -- rather than by silently dropping either. Gating on
validity rather than presence is the other half of it: a malformed absolute
date degrades to the relative window instead of quietly suppressing a
perfectly good one.

**Spend** (SR-E1, SR-E4, D6): Tavily bills in credits, and the weight is a
function of the request, not of the response -- ``basic`` costs one credit
and ``advanced`` costs two. That multiplier is the provider's published API
semantics, not configuration, so it lives in :data:`TAVILY_CREDITS_BY_DEPTH`
and the only way to change the depth on a request is
:meth:`_Plan.set_depth`, which moves both the wire parameter and the billed
weight together. This is the SR-E4 live defect made unreproducible: discodon
counted every search as one unit against a budget whose stated purpose was
"to manage shared API credits", so an operator who set ``advanced``
under-billed by 2x with nothing to notice.

Money is reported only when the host supplied ``usd_per_credit``. Unpriced
is not free: Tavily's per-credit rate is plan-dependent, and a shipped
constant would put an estimate nobody made behind a cost figure. With no
rate configured the credits are still counted in ``provider_units``, which
is the dimension a credit budget reads.

Credits are charged for a *served* search. A call the provider refused --
429, 432, quota, auth -- consumed an exchange (``calls`` is 1, because that
is the number a rate cap enforces and the number a bill would price, SR-E2)
but bought no credits, so ``provider_units`` stays zero on the failure path.

**Failures** (SR-J1, SR-D3): the transport raises whatever it likes, and
mapping onto the typed taxonomy is this module's job. Tavily's account-level
quota exhaustion has its own statuses -- 432 (plan usage limit) and 433
(pay-as-you-go limit) -- and both become :class:`QuotaExhausted`, which is a
different class from the local refusal in
:class:`~threetears.search.contracts.errors.LocalCapExceeded`: one bounds
money and the other bounds a run's shape, and merging them would hide which
authority said no. A dead search backend is an outage rather than a
per-query warning, so it is logged at ERROR as discodon's breaker did.

**What is deliberately not ported.** The per-invocation quota breaker
(circuit-breaking is the transport's, §3.1), the daily/per-invocation call
counters (a budget port's, D4/D5), the per-URL grounding corpus
(Aggregate's, SR-A5), and the LLM-facing prose (Bind's). Each was a
responsibility discodon's wrapper had to carry because there was nowhere
else to put it; here there is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
from typing import Final

from pydantic import JsonValue

from threetears.observe import get_logger
from threetears.search.adapters._common import (
    _as_float,
    _as_str,
    _string_list,
    _DispositionPlan,
    attributed_failure,
    decode_results_payload,
    parsed_base_url,
)
from threetears.search.contracts import (
    CRITERION_CARRIER,
    CRITERION_DOMAINS_EXCLUDE,
    CRITERION_DOMAINS_INCLUDE,
    CRITERION_LANGUAGE,
    CRITERION_MAX_RESULTS,
    CRITERION_MIN_RESOLUTION,
    CRITERION_RIGHTS_CLASS,
    CRITERION_TIME_RANGE,
    FACET_HAS_DOWNLOADABLE_DATA,
    FACET_LOCATOR_KIND,
    FIDELITY_CONTENT,
    FIDELITY_SNIPPET,
    PRICING_PER_WEIGHTED_UNIT,
    PRODUCER_API_PROVIDER,
    SCALE_UNIT_INTERVAL,
    AuthFailed,
    Candidate,
    CandidateSet,
    ContentSlot,
    Criterion,
    LocalCapExceeded,
    Locator,
    MalformedResponse,
    Provenance,
    ProviderCapabilities,
    QuotaExhausted,
    RateLimited,
    ScoreEntry,
    SearchFailure,
    SearchRequest,
    SearchTransport,
    Spend,
    TimedOut,
    TransportFailed,
    TransportResponse,
    register_capabilities,
)

__all__ = [
    "TAVILY_400_REMEDIATION",
    "TAVILY_API_BASE_URL",
    "TAVILY_CAPABILITIES",
    "TAVILY_CREDITS_BY_DEPTH",
    "TAVILY_MAX_QUERY_CHARACTERS",
    "TAVILY_MAX_RESULTS_CEILING",
    "TAVILY_PARAM_RAW_CONTENT",
    "TAVILY_PARAM_SEARCH_DEPTH",
    "TAVILY_PARAM_TIME_RANGE",
    "TAVILY_PARAM_TOPIC",
    "TAVILY_PROVIDER",
    "TAVILY_QUOTA_REMEDIATION",
    "TAVILY_RAW_CONTENT_FORMATS",
    "TAVILY_RELATIVE_TIME_RANGES",
    "TAVILY_SEARCH_DEPTHS",
    "TAVILY_TOPICS",
    "TavilyAdapter",
]

_logger = get_logger(__name__)

#: product name; the key Tavily's capabilities are registered under.
TAVILY_PROVIDER: Final[str] = "tavily"

#: namespace for Tavily-specific criteria. Provider parameters the well-known
#: vocabulary does not own ride ``tavily:<name>`` rather than widening it (the
#: criteria module's rule).
TAVILY_NAMESPACE: Final[str] = "tavily"

#: Tavily's published API root. A compiled-in product constant, not an
#: environment default: D21/SR-K1 forbid deriving an upstream from the
#: environment because such a value is unauditable, and a host that fronts
#: Tavily with a gateway states that gateway explicitly at construction.
TAVILY_API_BASE_URL: Final[str] = "https://api.tavily.com"

#: retrieval depth; value is one of :data:`TAVILY_SEARCH_DEPTHS`. This is the
#: billed parameter (SR-E4).
TAVILY_PARAM_SEARCH_DEPTH: Final[str] = f"{TAVILY_NAMESPACE}:search-depth"
#: result topic; value is one of :data:`TAVILY_TOPICS`.
TAVILY_PARAM_TOPIC: Final[str] = f"{TAVILY_NAMESPACE}:topic"
#: Tavily's *relative* publication window; value is one of
#: :data:`TAVILY_RELATIVE_TIME_RANGES`. The well-known ``time-range``
#: criterion is absolute and is a different quantity -- and when both are
#: asked for, the absolute one wins (RES-T4M9).
TAVILY_PARAM_TIME_RANGE: Final[str] = f"{TAVILY_NAMESPACE}:time-range"
#: page-text format Tavily returns alongside each result; value is one of
#: :data:`TAVILY_RAW_CONTENT_FORMATS`. ``none`` asks for snippets only.
TAVILY_PARAM_RAW_CONTENT: Final[str] = f"{TAVILY_NAMESPACE}:include-raw-content"

#: the retrieval depths Tavily sells. Their credit weights differ, so this
#: vocabulary and :data:`TAVILY_CREDITS_BY_DEPTH` move together.
TAVILY_SEARCH_DEPTHS: Final[tuple[str, ...]] = ("basic", "advanced")

#: the topics Tavily's search endpoint accepts.
TAVILY_TOPICS: Final[tuple[str, ...]] = ("general", "news")

#: the relative publication windows Tavily's ``time_range`` accepts.
TAVILY_RELATIVE_TIME_RANGES: Final[tuple[str, ...]] = ("day", "week", "month", "year")

#: the page-text formats ``include_raw_content`` accepts, plus the ``none``
#: that omits the parameter entirely (snippets only).
TAVILY_RAW_CONTENT_FORMATS: Final[tuple[str, ...]] = ("text", "markdown", "none")

#: Tavily's per-request result ceiling. Asking for more is not an error --
#: ``max-results`` is an upper bound, and a ceiling below it still honours it.
TAVILY_MAX_RESULTS_CEILING: Final[int] = 20

#: Tavily's documented query-length limit. Refused here rather than on the
#: wire: a 400 costs an exchange and teaches the caller nothing.
TAVILY_MAX_QUERY_CHARACTERS: Final[int] = 400

#: **The SR-E4 table.** Credits one search costs, by depth. Provider-published
#: API semantics, not deployment configuration -- which is why it is a module
#: constant rather than a constructor argument, and why :meth:`_Plan.set_depth`
#: is the only way to move the depth: the wire parameter and the billed weight
#: cannot drift apart if they are set together.
TAVILY_CREDITS_BY_DEPTH: Final[Mapping[str, Decimal]] = {
    "basic": Decimal("1"),
    "advanced": Decimal("2"),
}

#: The teaching error. Tavily answers 400 to a request whose parameters
#: disagree -- classically a ``time_range`` sent alongside ``start_date`` /
#: ``end_date``, which is the RES-T4M9 defect. This adapter resolves that
#: combination before sending, so a 400 reaching a caller means something
#: else disagreed, and the remediation says which things to look at.
TAVILY_400_REMEDIATION: Final[str] = (
    "Tavily refuses a request whose publication-date scoping is stated twice: a relative 'time_range' "
    "alongside absolute 'start_date'/'end_date' is the classic 400 (RES-T4M9). This adapter applies "
    f"absolute-wins precedence before sending, so a 400 here points at another parameter -- check the "
    f"'{TAVILY_PARAM_SEARCH_DEPTH}', '{TAVILY_PARAM_TOPIC}' and domain-scoping values this deployment "
    "sent, and that the query is within Tavily's length limit."
)

#: Quota exhaustion is account-level and unfixable by rephrasing, so the
#: remediation names the only two things that clear it (SR-D3).
TAVILY_QUOTA_REMEDIATION: Final[str] = (
    "Tavily's account-level usage quota is spent: HTTP 432 is the plan limit and 433 the "
    "pay-as-you-go limit. No rephrasing or retry clears it -- top up billing or raise the plan "
    "limit. This is the provider's refusal, not a locally-configured cap (SR-D3): stop searching "
    "on this key rather than pacing against it."
)

#: Tavily's capability declaration (SR-B4). Registered at import so a
#: consumer can branch before constructing an adapter.
TAVILY_CAPABILITIES: Final[ProviderCapabilities] = ProviderCapabilities(
    provider=TAVILY_PROVIDER,
    pushdown_criteria=(
        CRITERION_DOMAINS_INCLUDE,
        CRITERION_DOMAINS_EXCLUDE,
        CRITERION_MAX_RESULTS,
        CRITERION_TIME_RANGE,
    ),
    local_criteria=(),
    unsatisfiable_criteria=(
        CRITERION_LANGUAGE,
        CRITERION_CARRIER,
        CRITERION_MIN_RESOLUTION,
        CRITERION_RIGHTS_CLASS,
    ),
    namespaced_parameters=(
        TAVILY_PARAM_RAW_CONTENT,
        TAVILY_PARAM_SEARCH_DEPTH,
        TAVILY_PARAM_TIME_RANGE,
        TAVILY_PARAM_TOPIC,
    ),
    supports_paging=False,
    max_results_per_page=TAVILY_MAX_RESULTS_CEILING,
    categories=None,
    engines=None,
    safesearch_levels=None,
    relative_time_ranges=TAVILY_RELATIVE_TIME_RANGES,
    search_depths=TAVILY_SEARCH_DEPTHS,
    topics=TAVILY_TOPICS,
    pricing_model=PRICING_PER_WEIGHTED_UNIT,
)

register_capabilities(TAVILY_CAPABILITIES)


def _coerce_score(value: object) -> float | None:
    """Read Tavily's per-result ``score`` as a float, or ``None`` if unusable.

    Ported in intent from discodon: Tavily reports a relevance score in
    [0, 1] per result, and a missing or non-numeric one is *unknown*, not
    zero. The distinction is load-bearing downstream -- a relevance cull that
    reads unknown as zero drops the result instead of keeping it -- and the
    contract spells unknown as an absent :class:`ScoreEntry`, so this returns
    ``None`` and the caller publishes nothing.

    :param value: the provider's reported score
    :ptype value: object
    :return: the score, or None when the provider reported none this reader
        can use
    :rtype: float | None
    """
    return _as_float(value)


def _as_published_at(value: object) -> datetime | None:
    """Read a Tavily ``published_date`` as a timezone-aware datetime.

    Tavily reports ISO 8601 for ``general`` results and RFC 2822 (``Wed, 21
    Aug 2024 07:00:00 GMT``) for ``news`` ones, so both are read rather than
    one being assumed and the other silently discarding a date the provider
    did report. A naive value is read as UTC -- which is what the provider's
    sources produce -- and the raw string stays on provenance so the
    assumption is inspectable rather than buried here.

    :param value: the provider's reported publication date
    :ptype value: object
    :return: an aware datetime, or None when the value is absent or
        unparseable (in which case nothing is invented)
    :rtype: datetime | None
    """
    if not isinstance(value, str) or not value:
        return None
    parsed = _parse_iso(value) or _parse_rfc2822(value)
    if parsed is None:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _parse_iso(value: str) -> datetime | None:
    """Read an ISO 8601 timestamp, or ``None`` when it is not one.

    :param value: the provider's date string
    :ptype value: str
    :return: the parsed datetime, or None
    :rtype: datetime | None
    """
    # Probing a provider date for one of the two shapes it uses; the other
    # shape is tried next, and an unparseable value leaves published_at unset
    # with the raw string kept on provenance.
    try:
        return datetime.fromisoformat(value)
    # NOSILENT: not ISO -- the RFC 2822 reader gets the next attempt
    except ValueError:
        return None


def _parse_rfc2822(value: str) -> datetime | None:
    """Read an RFC 2822 timestamp, or ``None`` when it is not one.

    :param value: the provider's date string
    :ptype value: str
    :return: the parsed datetime, or None
    :rtype: datetime | None
    """
    # The second of Tavily's two date shapes. An unparseable value leaves
    # published_at unset and keeps the raw string on provenance, so nothing is
    # lost and nothing is invented -- inventing a date would be the defect.
    try:
        return parsedate_to_datetime(value)
    # NOSILENT: an unparseable provider date leaves published_at unset, raw kept
    except TypeError, ValueError:
        return None


def _as_day(value: object) -> str | None:
    """Read a criterion time bound as Tavily's ``YYYY-MM-DD``.

    Ported guard: discodon validated absolute dates against a ``YYYY-MM-DD``
    pattern before sending, so a malformed value degraded to a less-scoped
    search rather than burning a billed call on a Tavily 400. The same rule
    holds here, widened to the contract's aware ISO instants -- which are
    what :meth:`Criterion.time_range` produces -- normalised to their UTC day.

    :param value: one bound of a ``time-range`` criterion value
    :ptype value: object
    :return: the day in Tavily's format, or None when the bound is not a
        date this reader can send
    :rtype: str | None
    """
    text = _as_str(value)
    if text is None:
        return None
    parsed = _parse_iso(text.strip())
    if parsed is None:
        return None
    aware = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return aware.astimezone(UTC).date().isoformat()


def _absolute_days(criterion: Criterion) -> dict[str, str]:
    """Read an absolute ``time-range`` criterion as Tavily's date parameters.

    :param criterion: a well-known ``time-range`` criterion
    :ptype criterion: Criterion
    :return: the ``start_date`` / ``end_date`` parameters the criterion
        yields; empty when neither bound is readable
    :rtype: dict[str, str]
    """
    value = criterion.value
    if not isinstance(value, Mapping):
        return {}
    days: dict[str, str] = {}
    for bound, parameter in (("start", "start_date"), ("end", "end_date")):
        day = _as_day(value.get(bound))
        if day is not None:
            days[parameter] = day
    return days


class _Plan(_DispositionPlan):
    """The request as Tavily will receive it, with its honest dispositions.

    Not a contract type -- it never leaves this module. It exists so the
    criteria mapping is one readable pass that produces the wire body, the
    dispositions and the billed weight together, instead of three passes that
    can disagree. The billed weight is the point: :meth:`set_depth` is the
    only way to change the depth, and it moves the wire parameter and the
    credits as one operation (SR-E4). The dispositions list and
    :meth:`answer` itself come from :class:`_DispositionPlan`, shared with
    SearXNG's own ``_Plan`` (SR-B2/SR-B3).
    """

    def __init__(self, body: dict[str, JsonValue], *, search_depth: str) -> None:
        """Start a plan from the body every query carries.

        :param body: the base wire body (query, defaults from the deployment)
        :ptype body: dict[str, JsonValue]
        :param search_depth: the depth this deployment defaults to
        :ptype search_depth: str
        """
        super().__init__()
        self.body = body
        self.search_depth = ""
        self.set_depth(search_depth)

    def set_depth(self, depth: str) -> None:
        """Set the retrieval depth, and with it what the call will cost.

        :param depth: one of :data:`TAVILY_SEARCH_DEPTHS`
        :ptype depth: str
        """
        self.search_depth = depth
        self.body["search_depth"] = depth

    @property
    def credits(self) -> Decimal:
        """Credits this request will cost if the provider serves it (SR-E4).

        A depth outside the published table bills the *highest* known weight:
        the defect this table exists to prevent is under-billing, so an
        unknown depth must not be cheaper than a known one. Construction and
        criterion validation both refuse unknown depths, so this is a floor
        under a case that should not arise rather than a live path.

        :return: the weighted units this depth costs
        :rtype: Decimal
        """
        return TAVILY_CREDITS_BY_DEPTH.get(self.search_depth, max(TAVILY_CREDITS_BY_DEPTH.values()))


class TavilyAdapter:
    """Tavily's search API, behind the provider seam.

    Satisfies :class:`~threetears.search.contracts.provider.SearchProvider`
    structurally. Construct one per configured *key*, not per product: two
    Tavily keys are two instances, because they draw on separate provider
    quotas and are rate-limited and exhausted separately (D8, SR-N4). That is
    discodon's EVL-TQ7K ruling carried across -- an eval-scoped key exists
    precisely so an eval burst cannot exhaust the quota protecting live
    research, and provenance records which of them answered.
    """

    def __init__(
        self,
        *,
        api_key: str,
        transport: SearchTransport,
        base_url: str = TAVILY_API_BASE_URL,
        provider_instance: str | None = None,
        default_max_results: int = 5,
        default_search_depth: str = "basic",
        default_topic: str = "general",
        default_include_raw_content: str = "none",
        default_include_domains: Sequence[str] = (),
        usd_per_credit: Decimal | None = None,
    ) -> None:
        """Bind to one configured Tavily key.

        :param api_key: the Tavily API key, resolved by the host. Required
            and never defaulted: this package reads no environment variable
            and no secret store, so a ``scheme://locator`` reference is
            dereferenced by the host before construction (D21, SR-K1)
        :ptype api_key: str
        :param transport: the injected transport every request goes through
        :ptype transport: SearchTransport
        :param base_url: Tavily's API root, or the gateway a deployment
            fronts it with. Defaults to the published product endpoint --
            a compiled-in constant an auditor can read, which an environment
            default would not be; a non-HTTP scheme is refused here rather
            than at the socket
        :ptype base_url: str
        :param provider_instance: name for this configured key, used in
            provenance and as the pacing key. Defaults to the base URL's
            host; a deployment running more than one key (live and eval --
            EVL-TQ7K) MUST name them, because that separation is the whole
            point of holding two
        :ptype provider_instance: str | None
        :param default_max_results: results per search when the request asks
            for no cap (1..:data:`TAVILY_MAX_RESULTS_CEILING`)
        :ptype default_max_results: int
        :param default_search_depth: retrieval depth when the request names
            none. ``basic`` costs one credit, ``advanced`` two (SR-E4)
        :ptype default_search_depth: str
        :param default_topic: result topic when the request names none
        :ptype default_topic: str
        :param default_include_raw_content: page-text format to request with
            every search: ``text``, ``markdown``, or ``none`` for snippets
            only. Raw content is what makes a Tavily result carry its own
            information (SR-A2) and costs no extra credits, but it is off by
            default because a consumer that will not read it pays for it in
            tokens and latency
        :ptype default_include_raw_content: str
        :param default_include_domains: domains every search is scoped to
            unless the request scopes it otherwise. Empty by default and
            deliberately: a deployment-wide allow-list biases every query
            that did not ask for one
        :ptype default_include_domains: Sequence[str]
        :param usd_per_credit: what this account pays per Tavily credit, if
            the host knows. Unset means *unpriced*, not free: the rate is
            plan-dependent, so shipping a constant would put an estimate
            nobody made behind a cost figure. Credits are counted either way
            in :attr:`Spend.provider_units`
        :ptype usd_per_credit: Decimal | None
        :raises ValueError: when ``api_key`` is empty, ``base_url`` is not an
            absolute http(s) URL, a default is outside the vocabulary Tavily
            accepts, or ``usd_per_credit`` is negative
        """
        if not api_key:
            raise ValueError(
                "TavilyAdapter requires a host-supplied api_key: this package reads no environment "
                "variable and no secret store (D21, SR-K1)"
            )
        parsed = parsed_base_url(base_url)
        if default_search_depth not in TAVILY_SEARCH_DEPTHS:
            raise ValueError(
                f"default_search_depth must be one of {TAVILY_SEARCH_DEPTHS}, got {default_search_depth!r}"
            )
        if default_topic not in TAVILY_TOPICS:
            raise ValueError(f"default_topic must be one of {TAVILY_TOPICS}, got {default_topic!r}")
        if default_include_raw_content not in TAVILY_RAW_CONTENT_FORMATS:
            raise ValueError(
                f"default_include_raw_content must be one of {TAVILY_RAW_CONTENT_FORMATS}, "
                f"got {default_include_raw_content!r}"
            )
        if not 1 <= default_max_results <= TAVILY_MAX_RESULTS_CEILING:
            raise ValueError(
                f"default_max_results must be between 1 and {TAVILY_MAX_RESULTS_CEILING}, got {default_max_results!r}"
            )
        if usd_per_credit is not None and usd_per_credit < 0:
            raise ValueError(f"usd_per_credit cannot be negative, got {usd_per_credit!r}")
        self._api_key = api_key
        self._transport = transport
        self._base_url = base_url.rstrip("/")
        self._provider_instance = provider_instance or (parsed.hostname or self._base_url)
        self._default_max_results = default_max_results
        self._default_search_depth = default_search_depth
        self._default_topic = default_topic
        self._default_raw_content = default_include_raw_content
        self._default_include_domains = tuple(default_include_domains)
        self._usd_per_credit = usd_per_credit

    @property
    def provider(self) -> str:
        """Product name.

        :return: :data:`TAVILY_PROVIDER`
        :rtype: str
        """
        return TAVILY_PROVIDER

    @property
    def provider_instance(self) -> str:
        """Name of the configured key this adapter searches with.

        :return: the instance name
        :rtype: str
        """
        return self._provider_instance

    @property
    def capabilities(self) -> ProviderCapabilities:
        """What Tavily can express (SR-B4).

        :return: :data:`TAVILY_CAPABILITIES`
        :rtype: ProviderCapabilities
        """
        return TAVILY_CAPABILITIES

    async def search(self, request: SearchRequest, *, timeout_seconds: float | None = None) -> CandidateSet:
        """Run one query and return typed candidates.

        Tavily fans nothing in and reports no partial-answer signal, so there
        are no notices to raise. Its ``answer`` and ``images`` fields are
        deliberately unmapped: they are a different shape from a candidate,
        and giving them a candidate's fields would be an invention.

        :param request: what the caller asked for
        :ptype request: SearchRequest
        :param timeout_seconds: bound for this call (SR-G2); None leaves the
            transport's configured timeout in force (SR-G1)
        :ptype timeout_seconds: float | None
        :return: candidates, one disposition per criterion, and spend. An
            empty candidate tuple is a success (SR-J2)
        :rtype: CandidateSet
        :raises SearchFailure: one of the typed classes, carrying spend
        """
        try:
            plan = self._plan(request)
            response = await self._exchange(plan, timeout_seconds=timeout_seconds)
            spend = self._spend_for(plan, response, billed=True)
            payload = self._decode(response, spend)
            candidates = self._candidates(request, plan, payload, response, spend)
        except SearchFailure as failure:
            # every failure leaves this adapter fully attributed: which key,
            # which egress, when (D8/D20, SR-A3) -- the record riding
            # ToolResult.metadata is the only fact that survives the wire, so
            # a consumer-side pacing tracker rebuilds its key from it.
            attributed = self._attributed(failure)
            if attributed is failure:
                raise
            raise attributed from failure
        return CandidateSet(candidates=tuple(candidates), dispositions=tuple(plan.dispositions), spend=spend)

    # -- request planning ---------------------------------------------------

    def _plan(self, request: SearchRequest) -> _Plan:
        """Map the request's criteria onto the wire body and dispositions.

        Every criterion the request carried gets exactly one answer, and an
        answer is never "nothing" (SR-B2, SR-B3).

        The absolute-date scan happens before the pass rather than inside it,
        because the RES-T4M9 precedence cannot be decided one criterion at a
        time: whether a relative window is suppressed depends on whether a
        *valid* absolute bound was asked for somewhere else in the same
        request, and criteria have no order the caller must respect.

        :param request: the caller's request
        :ptype request: SearchRequest
        :return: the plan the exchange reads
        :rtype: _Plan
        :raises LocalCapExceeded: when the query is empty or longer than
            Tavily accepts -- refused here, before an exchange is spent
        """
        query = self._checked_query(request)
        body: dict[str, JsonValue] = {
            "query": query,
            "max_results": self._default_max_results,
            "topic": self._default_topic,
        }
        if self._default_raw_content != "none":
            body["include_raw_content"] = self._default_raw_content
        if self._default_include_domains:
            domains: list[JsonValue] = list(self._default_include_domains)
            body["include_domains"] = domains
        plan = _Plan(body, search_depth=self._default_search_depth)
        self._plan_fidelity(plan, request)
        has_absolute = any(
            _absolute_days(criterion) for criterion in request.criteria if criterion.key == CRITERION_TIME_RANGE
        )
        for criterion in request.criteria:
            self._plan_criterion(plan, criterion, has_absolute=has_absolute)
        return plan

    def _checked_query(self, request: SearchRequest) -> str:
        """Refuse a query Tavily will refuse, before an exchange is spent.

        Ported from discodon, which checked both bounds before the HTTP call
        for the same reason: a query the provider will 400 costs a round trip
        and, on a metered key, a place in the caller's call budget. The
        refusal is :class:`LocalCapExceeded` because the authority saying no
        is local -- which SR-D3 requires to stay distinguishable from the
        provider's own :class:`QuotaExhausted`.

        :param request: the caller's request
        :ptype request: SearchRequest
        :return: the query as it will be sent
        :rtype: str
        :raises LocalCapExceeded: when the query is empty or over-long
        """
        query = request.query.strip()
        if not query:
            raise LocalCapExceeded(
                "tavily requires a non-empty query",
                spend=Spend(),
                provider_instance=self._provider_instance,
                remediation="send a query with at least one non-whitespace character",
                scope="query-empty",
            )
        if len(query) > TAVILY_MAX_QUERY_CHARACTERS:
            raise LocalCapExceeded(
                f"query is {len(query)} characters, past tavily's limit of {TAVILY_MAX_QUERY_CHARACTERS}",
                spend=Spend(),
                provider_instance=self._provider_instance,
                remediation=(
                    f"shorten the query to {TAVILY_MAX_QUERY_CHARACTERS} characters or fewer; Tavily answers "
                    "400 to a longer one, which costs an exchange and teaches nothing"
                ),
                scope="query-length",
            )
        return query

    def _plan_fidelity(self, plan: _Plan, request: SearchRequest) -> None:
        """Ask for page content when the consumer said it needs content.

        SR-B6 states fidelity on the request, and Tavily is the SR-A2 case:
        it can return the page text with the search response, so a consumer
        asking for content-grade fidelity should not have to know a
        provider-specific parameter to get it -- and Extract can then no-op
        rather than re-fetching what was already bought. Raw content costs no
        extra credits; the deployment's configured format is used, and
        ``text`` when the deployment configured none.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param request: the caller's request
        :ptype request: SearchRequest
        """
        if request.fidelity != FIDELITY_CONTENT or "include_raw_content" in plan.body:
            return
        plan.body["include_raw_content"] = "text" if self._default_raw_content == "none" else self._default_raw_content

    def _plan_criterion(self, plan: _Plan, criterion: Criterion, *, has_absolute: bool) -> None:
        """Fold one criterion into ``plan``.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the criterion to handle
        :ptype criterion: Criterion
        :param has_absolute: whether the request carries a readable absolute
            publication window, which wins over a relative one (RES-T4M9)
        :ptype has_absolute: bool
        """
        key = criterion.key
        if key == CRITERION_DOMAINS_INCLUDE:
            self._plan_domains(plan, criterion, parameter="include_domains")
        elif key == CRITERION_DOMAINS_EXCLUDE:
            self._plan_domains(plan, criterion, parameter="exclude_domains")
        elif key == CRITERION_MAX_RESULTS:
            self._plan_max_results(plan, criterion)
        elif key == CRITERION_TIME_RANGE:
            self._plan_time_range(plan, criterion)
        elif key == CRITERION_LANGUAGE:
            plan.answer(
                key,
                "unsatisfied",
                "Tavily exposes no language filter and reports no language to filter on locally; scope the "
                "language above this layer, or in the query text",
            )
        elif key == CRITERION_CARRIER:
            plan.answer(
                key,
                "unsatisfied",
                "Tavily searches web pages and scopes results by no carrier: its image results arrive in a "
                "separate list rather than as scoped results, so a carrier criterion cannot be honoured here",
            )
        elif key == CRITERION_MIN_RESOLUTION:
            plan.answer(
                key,
                "unsatisfied",
                "Tavily has no resolution filter and reports no pixel dimensions, so there is nothing to "
                "filter on locally either",
            )
        elif key == CRITERION_RIGHTS_CLASS:
            plan.answer(
                key,
                "unsatisfied",
                "Tavily exposes no rights or licence filter, and reports no rights metadata to filter on locally",
            )
        elif key in TAVILY_CAPABILITIES.namespaced_parameters:
            self._plan_namespaced(plan, criterion, has_absolute=has_absolute)
        else:
            plan.answer(key, "ignored-unknown", f"'{key}' is not a criterion this adapter recognises")

    def _plan_domains(self, plan: _Plan, criterion: Criterion, *, parameter: str) -> None:
        """Push domain scoping down to Tavily, which expresses both halves.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the domain criterion
        :ptype criterion: Criterion
        :param parameter: Tavily's parameter name for this half
        :ptype parameter: str
        """
        names = [domain.lower() for domain in _string_list(criterion.value) if domain]
        if not names:
            plan.answer(criterion.key, "unsatisfied", "the criterion named no domains, so nothing was scoped")
            return
        domains: list[JsonValue] = list(names)
        plan.body[parameter] = domains
        plan.answer(criterion.key, "pushdown", f"sent as Tavily's '{parameter}'")

    def _plan_max_results(self, plan: _Plan, criterion: Criterion) -> None:
        """Push a result cap down, within Tavily's own ceiling.

        A cap above the ceiling is still honoured by the ceiling -- twenty
        results do not exceed a cap of fifty -- so this is pushdown with the
        clamp stated, not an unsatisfied criterion. SR-E5 rides in the
        detail: for a per-request-priced provider a lower cap changes what
        you see, not what you pay.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the ``max-results`` criterion
        :ptype criterion: Criterion
        """
        value = criterion.value
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            plan.answer(
                criterion.key,
                "unsatisfied",
                f"'max-results' takes a positive integer; {value!r} is not one, so no cap was sent",
            )
            return
        capped = min(value, TAVILY_MAX_RESULTS_CEILING)
        plan.body["max_results"] = capped
        detail = f"sent as Tavily's 'max_results'={capped}"
        if capped != value:
            detail += (
                f" -- Tavily returns at most {TAVILY_MAX_RESULTS_CEILING} per search, which is inside the "
                f"requested cap of {value}"
            )
        detail += ". Tavily charges per request rather than per result, so a lower cap costs the same (SR-E5)"
        plan.answer(criterion.key, "pushdown", detail)

    def _plan_time_range(self, plan: _Plan, criterion: Criterion) -> None:
        """Push an absolute publication window down as Tavily's dates.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the well-known ``time-range`` criterion
        :ptype criterion: Criterion
        """
        days = _absolute_days(criterion)
        if not days:
            plan.answer(
                criterion.key,
                "unsatisfied",
                "no bound of this 'time-range' is a date this adapter can send; a malformed date would earn a "
                "Tavily 400 and spend an exchange, so the search goes out unscoped rather than refused",
            )
            return
        for parameter, day in days.items():
            plan.body[parameter] = day
        plan.answer(
            criterion.key,
            "pushdown",
            f"sent as Tavily's {'/'.join(sorted(days))} (UTC days). Absolute scoping wins over Tavily's "
            f"relative '{TAVILY_PARAM_TIME_RANGE}', which the provider refuses alongside it (RES-T4M9)",
        )

    def _plan_namespaced(self, plan: _Plan, criterion: Criterion, *, has_absolute: bool) -> None:
        """Push a ``tavily:``-namespaced parameter down to the provider.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the namespaced criterion
        :ptype criterion: Criterion
        :param has_absolute: whether a readable absolute window was also
            asked for, which suppresses the relative one (RES-T4M9)
        :ptype has_absolute: bool
        """
        key = criterion.key
        value = str(criterion.value)
        if key == TAVILY_PARAM_SEARCH_DEPTH:
            if not self._accepts(plan, key, value, TAVILY_SEARCH_DEPTHS, "search depth"):
                return
            plan.set_depth(value)
            plan.answer(
                key,
                "pushdown",
                f"sent as Tavily's 'search_depth'; this call therefore costs "
                f"{TAVILY_CREDITS_BY_DEPTH[value]} credit(s) (SR-E4)",
            )
            return
        if key == TAVILY_PARAM_TOPIC:
            if not self._accepts(plan, key, value, TAVILY_TOPICS, "topic"):
                return
            plan.body["topic"] = value
            plan.answer(key, "pushdown", "sent as Tavily's 'topic' parameter")
            return
        if key == TAVILY_PARAM_RAW_CONTENT:
            if not self._accepts(plan, key, value, TAVILY_RAW_CONTENT_FORMATS, "raw-content format"):
                return
            if value == "none":
                plan.body.pop("include_raw_content", None)
                plan.answer(key, "pushdown", "'include_raw_content' omitted, so Tavily returns snippets only")
                return
            plan.body["include_raw_content"] = value
            plan.answer(key, "pushdown", "sent as Tavily's 'include_raw_content' parameter")
            return
        self._plan_relative_window(plan, key, value, has_absolute=has_absolute)

    def _plan_relative_window(self, plan: _Plan, key: str, window: str, *, has_absolute: bool) -> None:
        """Push Tavily's relative window down unless an absolute one won.

        The RES-T4M9 rule, stated where it applies: Tavily answers 400 when
        relative and absolute publication scoping arrive together, and the
        fix is precedence a caller can read, not a parameter that vanishes.
        Absolute is strictly more specific, and a relative window is
        reconstructible from "now" if it is ever wanted, so nothing
        recoverable is lost by suppressing it.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param key: the namespaced criterion key
        :ptype key: str
        :param window: the requested relative window
        :ptype window: str
        :param has_absolute: whether a readable absolute window was also asked for
        :ptype has_absolute: bool
        """
        if not self._accepts(plan, key, window, TAVILY_RELATIVE_TIME_RANGES, "relative window"):
            return
        if has_absolute:
            plan.answer(
                key,
                "unsatisfied",
                f"suppressed by the absolute 'time-range' this request also carried: Tavily refuses the two "
                f"together with HTTP 400, and the absolute range is the more specific of them (RES-T4M9). "
                f"Send one form or the other to have '{window}' applied",
            )
            return
        plan.body["time_range"] = window
        plan.answer(key, "pushdown", "sent as Tavily's 'time_range' parameter")

    def _accepts(self, plan: _Plan, key: str, value: str, accepted: tuple[str, ...], what: str) -> bool:
        """Answer whether ``value`` is in ``accepted``, reporting it if not.

        A typo gets the vocabulary back from here rather than a 400 from the
        provider, which would cost an exchange to say less.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param key: the criterion key being answered for
        :ptype key: str
        :param value: the requested value
        :ptype value: str
        :param accepted: the values Tavily accepts
        :ptype accepted: tuple[str, ...]
        :param what: what the value names, for the message
        :ptype what: str
        :return: whether the value is acceptable
        :rtype: bool
        """
        if value in accepted:
            return True
        plan.answer(key, "unsatisfied", f"'{value}' is not a Tavily {what}; accepted: {', '.join(accepted)}")
        return False

    # -- the exchange -------------------------------------------------------

    async def _exchange(self, plan: _Plan, *, timeout_seconds: float | None) -> TransportResponse:
        """Perform the request, mapping every failure onto the taxonomy.

        :param plan: the planned request
        :ptype plan: _Plan
        :param timeout_seconds: per-call bound, or None for the transport's
            configured timeout
        :ptype timeout_seconds: float | None
        :return: the completed exchange
        :rtype: TransportResponse
        :raises SearchFailure: mapped from the provider's status or from
            whatever the transport raised
        """
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self._api_key}"}
        try:
            response = await self._transport.request(
                "POST",
                f"{self._base_url}/search",
                headers=headers,
                json_body=plan.body,
                timeout_seconds=timeout_seconds,
            )
        except SearchFailure as failure:
            # a transport that already speaks the taxonomy (this package's
            # ``standalone`` module) knows things the adapter cannot -- how
            # many attempts, how long, how many bytes. Keep its record and add
            # what only the adapter knows: which configured key.
            raise self._attributed(failure) from failure
        except TimeoutError as exc:
            raise TimedOut(
                f"tavily instance {self._provider_instance} did not answer within the call's bound",
                spend=Spend(calls=0),
                provider_instance=self._provider_instance,
            ) from exc
        except Exception as exc:
            # the protocol lets an implementation raise whatever it likes, and
            # an adapter must not import an HTTP library to name those
            # exceptions. Unrecognised means "gave up at the transport", which
            # is what TransportFailed says (SR-J1).
            raise TransportFailed(
                f"transport failed reaching tavily instance {self._provider_instance}: {type(exc).__name__}: {exc}",
                spend=Spend(calls=0),
                provider_instance=self._provider_instance,
            ) from exc
        self._raise_for_status(plan, response)
        return response

    def _attributed(self, failure: SearchFailure) -> SearchFailure:
        """Re-stamp a failure with what only this adapter can attribute (D8/D20, SR-A3).

        Shared with SearXNG's identical method as
        :func:`_common.attributed_failure`: which key, which egress, when,
        is the same three facts for every provider.

        :param failure: the failure about to leave this adapter
        :ptype failure: SearchFailure
        :return: the same failure class, fully attributed
        :rtype: SearchFailure
        """
        return attributed_failure(
            failure, provider_instance=self._provider_instance, egress_name=self._transport.egress_name
        )

    def _raise_for_status(self, plan: _Plan, response: TransportResponse) -> None:
        """Map a non-2xx status onto the typed taxonomy.

        :param plan: the planned request, for the spend the refusal carries
        :ptype plan: _Plan
        :param response: the completed exchange
        :ptype response: TransportResponse
        :raises SearchFailure: for every status outside 2xx
        """
        status = response.status_code
        if 200 <= status < 300:
            return
        spend = self._spend_for(plan, response, billed=False)
        message = f"tavily instance {self._provider_instance} answered HTTP {status}"
        if status == 429:
            raise RateLimited(
                message,
                spend=spend,
                provider_instance=self._provider_instance,
                remediation="Tavily is pacing this key; lower the configured rate for this instance",
                retry_after_seconds=_as_float(response.headers.get("retry-after")),
            )
        if status in {402, 432, 433}:
            # a dead search backend is an outage, not a per-query warning, and
            # should be alertable -- discodon's breaker logged at ERROR for
            # exactly this reason, and the reason survives the port even though
            # the breaker itself belongs to the transport now.
            _logger.error(
                "tavily instance %s reports its usage quota exhausted (HTTP %d); searches on this key will "
                "keep being refused until billing is topped up",
                self._provider_instance,
                status,
            )
            raise QuotaExhausted(
                message,
                spend=spend,
                provider_instance=self._provider_instance,
                remediation=TAVILY_QUOTA_REMEDIATION,
            )
        if status in {401, 403, 407}:
            raise AuthFailed(
                message,
                spend=spend,
                provider_instance=self._provider_instance,
                remediation="Tavily rejected the API key this deployment supplied",
            )
        if status in {408, 504}:
            raise TimedOut(message, spend=spend, provider_instance=self._provider_instance)
        if status == 400:
            raise TransportFailed(
                message,
                spend=spend,
                provider_instance=self._provider_instance,
                remediation=TAVILY_400_REMEDIATION,
            )
        raise TransportFailed(message, spend=spend, provider_instance=self._provider_instance)

    def _spend_for(self, plan: _Plan, response: TransportResponse, *, billed: bool) -> Spend:
        """What one exchange consumed (SR-E1, SR-E4, D6).

        ``calls`` is 1 because the provider served this exchange -- the count
        a rate cap enforces is the count a bill would price (SR-E2), and
        attempts that never reached it are not in it (D4).

        Credits follow the depth the plan actually sent, which is what makes
        the SR-E4 defect unreproducible: there is no path that sends
        ``advanced`` and bills one unit. They are charged only for a served
        search -- a refused request (429, quota, auth) bought no credits,
        even though it did consume an exchange.

        Money is credits times the host's configured rate. With no rate
        configured it stays zero and the credits carry the whole truth:
        unpriced is not free, and the alternative is inventing a
        plan-dependent number nobody can reconcile (D6).

        :param plan: the planned request, which knows what depth was sent
        :ptype plan: _Plan
        :param response: the completed exchange
        :ptype response: TransportResponse
        :param billed: whether the provider served the search, and therefore
            charged for it
        :ptype billed: bool
        :return: the spend for this call
        :rtype: Spend
        """
        credits_spent = plan.credits if billed else Decimal("0")
        money = credits_spent * self._usd_per_credit if self._usd_per_credit is not None else Decimal("0")
        return Spend(
            money=money,
            wall_clock_seconds=response.elapsed_seconds,
            calls=1,
            provider_units=credits_spent,
            bytes_transferred=len(response.body),
        )

    def _decode(self, response: TransportResponse, spend: Spend) -> Mapping[str, object]:
        """Parse the JSON body Tavily returned.

        Shared shape with SearXNG's identical method as
        :func:`_common.decode_results_payload`; Tavily has no known common
        cause for a non-JSON body, so it names none.

        :param response: the successful exchange
        :ptype response: TransportResponse
        :param spend: what the call consumed, carried onto any failure
        :ptype spend: Spend
        :return: the decoded payload
        :rtype: Mapping[str, object]
        :raises MalformedResponse: when the body is not a JSON object, or
            carries no ``results`` list
        """
        return decode_results_payload(
            response, spend, provider_name="tavily", provider_instance=self._provider_instance
        )

    # -- candidates ---------------------------------------------------------

    def _candidates(
        self,
        request: SearchRequest,
        plan: _Plan,
        payload: Mapping[str, object],
        response: TransportResponse,
        spend: Spend,
    ) -> list[Candidate]:
        """Turn the provider's result list into typed candidates.

        :param request: the request being answered, for provenance
        :ptype request: SearchRequest
        :param plan: the planned request, for the depth provenance records
        :ptype plan: _Plan
        :param payload: the decoded provider payload
        :ptype payload: Mapping[str, object]
        :param response: the exchange, for egress attribution
        :ptype response: TransportResponse
        :param spend: what the call consumed, carried onto any failure
        :ptype spend: Spend
        :return: the candidates, in provider order
        :rtype: list[Candidate]
        :raises MalformedResponse: when a result carries no usable URL -- a
            result with no locator is not a result, and skipping it silently
            would report a narrower set as complete
        """
        raw_results = payload.get("results")
        results = raw_results if isinstance(raw_results, list) else []
        retrieved_at = datetime.now(UTC)
        request_id = _as_str(payload.get("request_id"))
        candidates: list[Candidate] = []
        for index, raw in enumerate(results):
            if not isinstance(raw, dict):
                raise MalformedResponse(
                    f"tavily instance {self._provider_instance} returned a non-object at results[{index}]",
                    spend=spend,
                    provider_instance=self._provider_instance,
                )
            candidates.append(
                self._candidate(request, plan, raw, response, retrieved_at, spend, index, request_id=request_id)
            )
        return candidates

    def _candidate(
        self,
        request: SearchRequest,
        plan: _Plan,
        raw: Mapping[str, object],
        response: TransportResponse,
        retrieved_at: datetime,
        spend: Spend,
        index: int,
        *,
        request_id: str | None,
    ) -> Candidate:
        """Build one candidate from one Tavily result object.

        :param request: the request being answered
        :ptype request: SearchRequest
        :param plan: the planned request, for the depth provenance records
        :ptype plan: _Plan
        :param raw: one entry of the provider's ``results`` list
        :ptype raw: Mapping[str, object]
        :param response: the exchange, for egress attribution
        :ptype response: TransportResponse
        :param retrieved_at: retrieval time shared by the whole set
        :ptype retrieved_at: datetime
        :param spend: what the call consumed, carried onto any failure
        :ptype spend: Spend
        :param index: position in the provider's list, for the error message
        :ptype index: int
        :param request_id: Tavily's id for the whole exchange, when it gave one
        :ptype request_id: str | None
        :return: the typed candidate
        :rtype: Candidate
        :raises MalformedResponse: when the result carries no usable URL
        """
        url = _as_str(raw.get("url"))
        if url is None:
            raise MalformedResponse(
                f"tavily instance {self._provider_instance} returned results[{index}] with no url",
                spend=spend,
                provider_instance=self._provider_instance,
            )
        content = self._content(plan, raw)
        return Candidate(
            identity=url,
            locators=(Locator(url=url, rel="canonical"),),
            provenance=self._provenance(request, plan, raw, response, retrieved_at, request_id=request_id),
            title=_as_str(raw.get("title")),
            snippet=_as_str(raw.get("content")),
            published_at=_as_published_at(raw.get("published_date")),
            scores=self._scores(raw),
            # Tavily can return the page text with any search, so content grade
            # is always *available* from it; what was achieved depends on
            # whether this request asked for it (SR-B6, SR-A2).
            fidelity_available=FIDELITY_CONTENT,
            fidelity_achieved=FIDELITY_CONTENT if content is not None else FIDELITY_SNIPPET,
            content=content,
            facets={FACET_HAS_DOWNLOADABLE_DATA: False, FACET_LOCATOR_KIND: "containing-page"},
        )

    def _content(self, plan: _Plan, raw: Mapping[str, object]) -> ContentSlot | None:
        """The page text Tavily already sold us, when it sent any (SR-A2).

        The origin is what Extract reads to know it has nothing to do: this
        content arrived with the search response, so re-fetching it would be
        paying twice for one page.

        :param plan: the planned request, which knows the format asked for
        :ptype plan: _Plan
        :param raw: one provider result object
        :ptype raw: Mapping[str, object]
        :return: the content slot, or None when the provider sent no page text
        :rtype: ContentSlot | None
        """
        text = _as_str(raw.get("raw_content"))
        if text is None:
            return None
        requested = plan.body.get("include_raw_content")
        mime = "text/markdown" if requested == "markdown" else "text/plain"
        # size_bytes stays unset on purpose: it records what a *fetch* moved,
        # and no fetch happened here -- the bytes are already counted once, in
        # the call's own bytes_transferred.
        return ContentSlot(text=text, origin="provider-response", mime_type=mime)

    def _provenance(
        self,
        request: SearchRequest,
        plan: _Plan,
        raw: Mapping[str, object],
        response: TransportResponse,
        retrieved_at: datetime,
        *,
        request_id: str | None,
    ) -> Provenance:
        """Record where this candidate came from (SR-A3, D20).

        The retrieval depth is provenance rather than an implementation
        detail: it is what the result cost and what quality it was retrieved
        at, and a later reader comparing two candidate sets needs to know
        which was bought at which depth.

        :param request: the request being answered
        :ptype request: SearchRequest
        :param plan: the planned request, for the depth that produced this
        :ptype plan: _Plan
        :param raw: one provider result object
        :ptype raw: Mapping[str, object]
        :param response: the exchange, for the egress it left by
        :ptype response: TransportResponse
        :param retrieved_at: retrieval time
        :ptype retrieved_at: datetime
        :param request_id: Tavily's id for the whole exchange, when it gave one
        :ptype request_id: str | None
        :return: the candidate's provenance
        :rtype: Provenance
        """
        provider_ids: dict[str, str] = {"search_depth": plan.search_depth}
        if request_id is not None:
            provider_ids["request_id"] = request_id
        published_raw = _as_str(raw.get("published_date"))
        if published_raw is not None:
            provider_ids["published_date_raw"] = published_raw
        return Provenance(
            query=request.query,
            provider_instance=self._provider_instance,
            provider_ids=provider_ids,
            retrieved_at=retrieved_at,
            egress=response.egress,
            producer=PRODUCER_API_PROVIDER,
        )

    def _scores(self, raw: Mapping[str, object]) -> tuple[ScoreEntry, ...]:
        """Every judgment Tavily reported, named and scaled (D1, SR-A4).

        Tavily reports one: a relevance score in [0, 1]. It is published as a
        named entry on the unit interval and non-comparable across providers
        -- which :meth:`ScoreEntry.provider_native` enforces rather than
        trusting a caller to remember. A score the provider did not report,
        or reported unusably, yields no entry at all: absent means unknown,
        and a zero would mean "judged irrelevant", which is a different claim
        and the one that gets a result wrongly culled.

        :param raw: one provider result object
        :ptype raw: Mapping[str, object]
        :return: the score entries, empty when the provider reported none
        :rtype: tuple[ScoreEntry, ...]
        """
        relevance = _coerce_score(raw.get("score"))
        if relevance is None:
            return ()
        return (
            ScoreEntry.provider_native(
                name="relevance",
                value=relevance,
                scale=SCALE_UNIT_INTERVAL,
                provider_instance=self._provider_instance,
            ),
        )
