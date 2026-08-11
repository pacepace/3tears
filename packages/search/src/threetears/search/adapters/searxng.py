"""SearXNG, through the injected transport and nothing else.

One provider's API, mapped onto the contract: SearXNG's JSON search
endpoint in, typed candidates with named scores, engine attribution,
published dates and per-criterion dispositions out. What this module
deliberately does NOT do is open a client, read an environment variable, or
keep a provider payload as a disclaimed ``raw`` blob.

**What SearXNG can and cannot express** (SR-B4, verified against its
settings vocabulary): it takes ``categories``, ``engines``, ``language``,
``safesearch`` and ``pageno``, plus a ``time_range`` that is *relative*
(``day``/``week``/``month``/``year``). It has no domain allow-list, no
result-count parameter, no resolution filter and no rights filter. So the
adapter pushes down what it can, filters locally what it honestly can, and
names the rest ``unsatisfied`` rather than dropping it (SR-B3). The
absolute ``time-range`` criterion is the interesting case: a relative
window is a different quantity, and quietly widening an absolute request
into the nearest relative one would return results outside what the caller
asked for while reporting success. It is named unsatisfied, with the
namespaced parameter that *does* work stated in the detail.

**A value the adapter cannot use is unsatisfied, never ignored** (SR-B3).
Recognition is decided by *key* alone -- every well-known criterion and
every parameter :data:`SEARXNG_CAPABILITIES` declares has its own branch --
so ``ignored-unknown`` can only ever be reached by a key this adapter
genuinely does not know. A recognised key carrying a value the adapter
cannot send (a ``language`` that is a list, a ``max-results`` of ``-1``, a
``searxng:safesearch`` of ``"high"``) is answered ``unsatisfied`` with one
shared detail shape -- *what was expected; what arrived, so what happened
instead* -- produced by the checks around :class:`_Refused`. The rule those
checks exist for: nothing malformed reaches the wire, and nothing malformed
is silently applied as something else. ``max-results`` is the sharp case,
because the slice it feeds turns ``-1`` into "drop the last result" and
``0`` into "return nothing", both of which would otherwise be reported as
the cap being honoured.

**One parameter, one owner.** SearXNG scopes by a single ``categories``
parameter, and two criteria can reach for it: the well-known ``carrier``
(mapped onto a category here) and the namespaced
:data:`SEARXNG_PARAM_CATEGORIES` (SearXNG's own vocabulary, stated
directly). They are *not* merged. SearXNG unions the categories it is
given -- ``images,news`` searches both -- so merging two scoping criteria
would search more than either asked for, which is the opposite of the
intersection a caller stating both means. Instead the namespaced criterion
wins, on the RES-T4M9 precedent: the more specific, more direct expression
takes precedence and the loser is reported ``unsatisfied`` naming the
winner, rather than either one vanishing. Precedence is decided by a
pre-pass over the whole request, so it does not depend on criteria order,
and it is gated on the winner's value being *usable* rather than merely
present -- a malformed ``searxng:categories`` suppresses nothing.

**Spend** (SR-E1, D6): a self-hosted SearXNG bills nothing, so money and
weighted units are zero and the real constraints live in the other
dimensions -- wall-clock, bytes, and the ``calls`` count a rate budget
enforces. ``calls`` counts exchanges the provider actually served, so a
connect failure that never reached it counts zero: the number a cap
enforces and the number a bill would price are the same number (SR-E2), and
the retry that never reached the provider is not one of them (D4).

**Failures** (SR-J1): the transport raises whatever it likes, and mapping
onto the typed taxonomy is this module's job. A transport that already
speaks the taxonomy (this package's ``standalone`` module does) has its
failure re-raised with the adapter's spend merged in; anything else is
classified by shape, because an adapter must not import an HTTP library to
recognise its exceptions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final
from urllib.parse import urlparse

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
    FACET_HEIGHT,
    FACET_LOCATOR_KIND,
    FACET_MEDIA_CATEGORY,
    FACET_RIGHTS_STATUS,
    FACET_WIDTH,
    FIDELITY_BYTES,
    FIDELITY_SNIPPET,
    PRICING_FREE_SELF_HOSTED,
    PRODUCER_API_PROVIDER,
    SCALE_RANK,
    SCALE_UNBOUNDED,
    AuthFailed,
    Candidate,
    CandidateSet,
    Criterion,
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
    "SEARXNG_403_REMEDIATION",
    "SEARXNG_CAPABILITIES",
    "SEARXNG_CATEGORIES",
    "SEARXNG_PARAM_CATEGORIES",
    "SEARXNG_PARAM_ENGINES",
    "SEARXNG_PARAM_PAGE",
    "SEARXNG_PARAM_SAFESEARCH",
    "SEARXNG_PARAM_TIME_RANGE",
    "SEARXNG_PROVIDER",
    "SEARXNG_RELATIVE_TIME_RANGES",
    "SearxngAdapter",
]

_logger = get_logger(__name__)

#: product name; the key SearXNG's capabilities are registered under.
SEARXNG_PROVIDER: Final[str] = "searxng"

#: namespace for SearXNG-specific criteria. Provider parameters this
#: vocabulary does not own ride ``searxng:<name>`` rather than widening the
#: well-known set (the criteria module's rule).
SEARXNG_NAMESPACE: Final[str] = "searxng"

#: restrict the search to named SearXNG categories; value is a string or a
#: list of strings.
SEARXNG_PARAM_CATEGORIES: Final[str] = f"{SEARXNG_NAMESPACE}:categories"
#: restrict the search to named engines; value is a string or list of strings.
SEARXNG_PARAM_ENGINES: Final[str] = f"{SEARXNG_NAMESPACE}:engines"
#: safesearch level; value is 0, 1 or 2.
SEARXNG_PARAM_SAFESEARCH: Final[str] = f"{SEARXNG_NAMESPACE}:safesearch"
#: 1-based result page; value is a positive int.
SEARXNG_PARAM_PAGE: Final[str] = f"{SEARXNG_NAMESPACE}:page"
#: SearXNG's *relative* publication window; value is one of
#: :data:`SEARXNG_RELATIVE_TIME_RANGES`. The well-known ``time-range``
#: criterion is absolute and is a different quantity.
SEARXNG_PARAM_TIME_RANGE: Final[str] = f"{SEARXNG_NAMESPACE}:time-range"

#: the relative windows SearXNG's ``time_range`` accepts.
SEARXNG_RELATIVE_TIME_RANGES: Final[tuple[str, ...]] = ("day", "week", "month", "year")

#: SearXNG's standard category names. Open in principle (an operator can
#: define more in ``settings.yml``); these are the ones a stock instance
#: ships, and an unrecognised value is passed through rather than refused.
SEARXNG_CATEGORIES: Final[tuple[str, ...]] = (
    "general",
    "images",
    "videos",
    "news",
    "music",
    "it",
    "science",
    "files",
    "social media",
    "map",
)

#: the ``media-contracts`` carrier taxonomy, mapped onto SearXNG categories.
#: Only these three carriers have a category that means the same thing;
#: anything else is honestly unsatisfiable rather than approximated.
_CARRIER_CATEGORIES: Final[Mapping[str, str]] = {
    "image": "images",
    "video": "videos",
    "audio": "music",
}

#: SearXNG's own category, mapped back onto the ``media-contracts`` carrier
#: taxonomy for the ``media_category`` facet. Categories with no carrier
#: meaning (``general``, ``it``, ``map``, ...) contribute no facet -- absence
#: means "not reported", which is the truth.
_CATEGORY_CARRIERS: Final[Mapping[str, str]] = {
    "images": "image",
    "videos": "video",
    "music": "audio",
    "files": "document",
}

#: The teaching error. A 403 from a SearXNG instance is almost never about
#: us: ``search.formats`` defaults to ``['html']``, so the JSON API answers
#: 403 until an operator adds ``json``. It is the single most common setup
#: failure this adapter will meet, so the remediation ships with the error
#: rather than living in a runbook (SR-J1).
SEARXNG_403_REMEDIATION: Final[str] = (
    "SearXNG refuses its JSON API with 403 when 'json' is absent from "
    "'search.formats' in settings.yml, which defaults to ['html'] only: add "
    "'json' to search.formats and restart the instance. A 403 can also come "
    "from a reverse proxy or auth layer in front of the instance rejecting "
    "the credentials this deployment supplied -- the two are distinguishable "
    "by whether the same instance answers an ordinary HTML query."
)

#: SearXNG's capability declaration (SR-B4). Registered at import so a
#: consumer can branch before constructing an adapter.
SEARXNG_CAPABILITIES: Final[ProviderCapabilities] = ProviderCapabilities(
    provider=SEARXNG_PROVIDER,
    pushdown_criteria=(CRITERION_LANGUAGE, CRITERION_CARRIER),
    local_criteria=(CRITERION_DOMAINS_INCLUDE, CRITERION_DOMAINS_EXCLUDE, CRITERION_MAX_RESULTS),
    unsatisfiable_criteria=(CRITERION_TIME_RANGE, CRITERION_MIN_RESOLUTION, CRITERION_RIGHTS_CLASS),
    namespaced_parameters=(
        SEARXNG_PARAM_CATEGORIES,
        SEARXNG_PARAM_ENGINES,
        SEARXNG_PARAM_PAGE,
        SEARXNG_PARAM_SAFESEARCH,
        SEARXNG_PARAM_TIME_RANGE,
    ),
    supports_paging=True,
    max_results_per_page=None,
    categories=SEARXNG_CATEGORIES,
    engines=None,
    safesearch_levels=(0, 1, 2),
    relative_time_ranges=SEARXNG_RELATIVE_TIME_RANGES,
    pricing_model=PRICING_FREE_SELF_HOSTED,
)

register_capabilities(SEARXNG_CAPABILITIES)


def _domain_matches(host: str, domain: str) -> bool:
    """Whether ``host`` is ``domain`` or a subdomain of it.

    :param host: the candidate's hostname, lower-cased
    :ptype host: str
    :param domain: the domain from a scoping criterion, lower-cased
    :ptype domain: str
    :return: whether the host falls inside the domain
    :rtype: bool
    """
    return host == domain or host.endswith(f".{domain}")


def _hostname(url: str) -> str:
    """Lower-cased hostname of ``url``, or the empty string when it has none.

    :param url: an absolute URL
    :ptype url: str
    :return: the hostname, lower-cased; empty when unparseable
    :rtype: str
    """
    return (urlparse(url).hostname or "").lower()


def _as_published_at(value: object) -> datetime | None:
    """Read a SearXNG ``publishedDate`` as a timezone-aware datetime.

    SearXNG's engines frequently report a naive ISO timestamp. This reads a
    naive value as UTC -- which is what those engines produce -- rather than
    discarding the date; the raw string stays on provenance so the
    assumption is inspectable rather than buried here.

    Stays local rather than sharing Tavily's date reader: SearXNG only ever
    reports ISO 8601, never Tavily's RFC 2822 ``news`` shape, so the second
    parse attempt would be dead code here.

    :param value: the provider's reported publication date
    :ptype value: object
    :return: an aware datetime, or None when the value is absent or
        unparseable (in which case nothing is invented)
    :rtype: datetime | None
    """
    if not isinstance(value, str) or not value:
        return None
    # Probing a provider date for a parseable timestamp. An unparseable one
    # leaves published_at unset and keeps the raw string on provenance, so
    # nothing is lost and nothing is invented -- inventing a date would be the
    # defect here, and logging once per result would be noise.
    try:
        parsed = datetime.fromisoformat(value)
    # NOSILENT: an unparseable provider date leaves published_at unset, raw kept
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _numbers(value: object) -> list[float]:
    """Read a provider value as a list of numbers, dropping what is not one.

    :param value: a sequence of provider values
    :ptype value: object
    :return: the numeric values, in order
    :rtype: list[float]
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return [number for number in (_as_float(item) for item in value) if number is not None]


def _engine_name(entry: object) -> str:
    """Read an engine name from one ``unresponsive_engines`` entry.

    SearXNG has reported this list as ``[engine, error]`` pairs and as
    objects across versions, so both shapes are read rather than one being
    assumed and the other silently producing an empty notice.

    :param entry: one entry of the provider's ``unresponsive_engines``
    :ptype entry: object
    :return: the engine name, or the empty string when there is none
    :rtype: str
    """
    if isinstance(entry, Mapping):
        name = entry.get("engine") or entry.get("name")
        return str(name) if name else ""
    values = _string_list(entry)
    return values[0] if values else ""


@dataclass(frozen=True, slots=True)
class _Refused:
    """A recognised criterion whose value this adapter cannot use.

    The single failure shape every value check below returns, so that a bad
    value lands on the same honest answer wherever it arrives rather than on
    four hand-written variations of one idea. :meth:`_Plan.accept` is what
    turns it into the disposition; nothing else constructs one.
    """

    #: what was expected and what arrived, as a sentence fragment the plan
    #: completes with what it did instead.
    detail: str


def _shape(value: object) -> str:
    """Name a criterion value for a teaching detail, type included.

    The type is half the lesson: ``'10'`` and ``10`` read identically in a
    message that prints only the value, and it is exactly that pair a caller
    building criteria from JSON gets wrong.

    :param value: the value the criterion carried
    :ptype value: object
    :return: the value and its type name
    :rtype: str
    """
    return f"{value!r} ({type(value).__name__})"


def _text(value: object, key: str, expected: str) -> str | _Refused:
    """Read a criterion value as a non-empty string.

    :param value: the value the criterion carried
    :ptype value: object
    :param key: the criterion key, for the detail
    :ptype key: str
    :param expected: what the criterion takes, for the detail
    :ptype expected: str
    :return: the string, or the refusal naming why it is not one
    :rtype: str | _Refused
    """
    if isinstance(value, str) and value:
        return value
    return _Refused(f"'{key}' takes {expected}; got {_shape(value)}")


def _counting_number(value: object, key: str, expected: str) -> int | _Refused:
    """Read a criterion value as an integer of at least one.

    A bool is refused rather than read as its integer value: ``True`` is a
    caller saying something other than "one", and honouring it as a count of
    one would invent a bound nobody asked for. Zero and negatives are
    refused because the things this feeds -- a result cap and a page number
    -- have no meaning at or below zero, and the cap's slice would quietly
    turn them into "return nothing" and "drop the last result".

    :param value: the value the criterion carried
    :ptype value: object
    :param key: the criterion key, for the detail
    :ptype key: str
    :param expected: what the criterion takes, for the detail
    :ptype expected: str
    :return: the count, or the refusal naming why it is not one
    :rtype: int | _Refused
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return _Refused(f"'{key}' takes {expected}; got {_shape(value)}")
    return value


def _level(value: object, accepted: tuple[int, ...], key: str, expected: str) -> int | _Refused:
    """Read a criterion value as one of a closed set of integer levels.

    :param value: the value the criterion carried
    :ptype value: object
    :param accepted: the levels the provider actually accepts
    :ptype accepted: tuple[int, ...]
    :param key: the criterion key, for the detail
    :ptype key: str
    :param expected: what the criterion takes, for the detail
    :ptype expected: str
    :return: the level, or the refusal naming why it is not one
    :rtype: int | _Refused
    """
    if isinstance(value, bool) or not isinstance(value, int) or value not in accepted:
        return _Refused(f"'{key}' takes {expected}; got {_shape(value)}")
    return value


def _names(value: object, key: str, expected: str) -> tuple[str, ...] | _Refused:
    """Read a criterion value as a non-empty tuple of names.

    A single string is one name; a list must be all strings. A list carrying
    a non-string is refused whole rather than having the offender dropped:
    silently scoping by three of the four domains a caller named is the
    quiet-wrong-answer this module exists to avoid.

    :param value: the value the criterion carried
    :ptype value: object
    :param key: the criterion key, for the detail
    :ptype key: str
    :param expected: what the criterion takes, for the detail
    :ptype expected: str
    :return: the names, or the refusal naming why there are none
    :rtype: tuple[str, ...] | _Refused
    """
    if isinstance(value, str):
        raw: list[object] = [value]
    elif isinstance(value, list):
        raw = list(value)
    else:
        return _Refused(f"'{key}' takes {expected}; got {_shape(value)}")
    if not all(isinstance(item, str) for item in raw):
        return _Refused(f"'{key}' takes {expected}; got {_shape(value)}")
    names = tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())
    if not names:
        return _Refused(f"'{key}' takes {expected} and this one named none; got {_shape(value)}")
    return names


class _Plan(_DispositionPlan):
    """The request as SearXNG will receive it, with its honest dispositions.

    Not a contract type -- it never leaves this module. It exists so the
    criteria mapping is one readable pass that produces the wire parameters,
    the dispositions, and the local filters together, instead of three
    passes that can disagree. The dispositions list and :meth:`answer`
    itself come from :class:`_DispositionPlan`, shared with Tavily's own
    ``_Plan`` (SR-B2/SR-B3): what varies below is everything specific to
    what SearXNG can express.
    """

    def __init__(self, params: dict[str, str], *, categories_owner: str | None = None) -> None:
        """Start a plan from the parameters every query carries.

        :param params: the base wire parameters (query, format, defaults)
        :ptype params: dict[str, str]
        :param categories_owner: the criterion key that owns SearXNG's single
            ``categories`` parameter for this request, decided before the
            pass so precedence does not depend on criteria order. ``None``
            means nothing has claimed it
        :ptype categories_owner: str | None
        """
        super().__init__()
        self.params = params
        self.max_results: int | None = None
        self.domains_include: tuple[str, ...] = ()
        self.domains_exclude: tuple[str, ...] = ()
        self.categories_owner = categories_owner

    def accept[T](self, key: str, checked: T | _Refused, *, consequence: str) -> T | None:
        """Unwrap a checked criterion value, answering for it when unusable.

        The one place a bad value becomes a disposition, which is what keeps
        the answer shape identical across every criterion: *what was
        expected; what arrived, so what the adapter did instead*. A caller
        that gets ``None`` back has already been answered for and must
        return without touching the wire parameters.

        :param key: the criterion key being answered for
        :ptype key: str
        :param checked: the value, or the refusal one of the checks produced
        :ptype checked: T | _Refused
        :param consequence: what the adapter did instead, for the detail
        :ptype consequence: str
        :return: the usable value, or None when it was refused
        :rtype: T | None
        """
        if isinstance(checked, _Refused):
            self.answer(key, "unsatisfied", f"{checked.detail}, so {consequence}")
            return None
        return checked


class SearxngAdapter:
    """SearXNG's JSON search API, behind the provider seam.

    Satisfies :class:`~threetears.search.contracts.provider.SearchProvider`
    structurally. Construct one per configured instance: the instance name
    is what rate and ban budgets key on together with the transport's egress
    (D8, SR-N4), and what provenance records as having answered.
    """

    def __init__(
        self,
        *,
        base_url: str,
        transport: SearchTransport,
        provider_instance: str | None = None,
        credentials: Mapping[str, str] | None = None,
        default_categories: Sequence[str] = (),
        default_engines: Sequence[str] = (),
        default_safesearch: int | None = None,
    ) -> None:
        """Bind to one configured SearXNG deployment.

        :param base_url: the instance's base URL, from deployment config.
            MUST NOT come from a caller or the environment (D21, SR-K1); a
            non-HTTP scheme is refused here rather than at the socket
        :ptype base_url: str
        :param transport: the injected transport every request goes through
        :ptype transport: SearchTransport
        :param provider_instance: name for this deployment, used in
            provenance and as the pacing key. Defaults to the base URL's
            host, which is unique per deployment and needs no configuration
        :ptype provider_instance: str | None
        :param credentials: resolved credential headers the host supplies
            for instances behind an auth layer. Already-resolved values:
            this package never reads an environment variable or a secret
            store, so a ``scheme://locator`` reference is dereferenced by
            the host before construction (SR-K1)
        :ptype credentials: Mapping[str, str] | None
        :param default_categories: categories to search when the request
            names none
        :ptype default_categories: Sequence[str]
        :param default_engines: engines to restrict to when the request
            names none
        :ptype default_engines: Sequence[str]
        :param default_safesearch: safesearch level for this deployment
            (0, 1 or 2); None leaves the instance's own default in force
        :ptype default_safesearch: int | None
        :raises ValueError: when ``base_url`` is not an absolute http(s) URL
            or ``default_safesearch`` is outside the declared levels
        """
        parsed = parsed_base_url(base_url)
        levels = SEARXNG_CAPABILITIES.safesearch_levels or ()
        if default_safesearch is not None and default_safesearch not in levels:
            raise ValueError(f"default_safesearch must be one of {levels}, got {default_safesearch!r}")
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._provider_instance = provider_instance or (parsed.hostname or self._base_url)
        self._credentials = dict(credentials or {})
        self._default_categories = tuple(default_categories)
        self._default_engines = tuple(default_engines)
        self._default_safesearch = default_safesearch

    @property
    def provider(self) -> str:
        """Product name.

        :return: :data:`SEARXNG_PROVIDER`
        :rtype: str
        """
        return SEARXNG_PROVIDER

    @property
    def provider_instance(self) -> str:
        """Name of the configured deployment this adapter reaches.

        :return: the instance name
        :rtype: str
        """
        return self._provider_instance

    @property
    def capabilities(self) -> ProviderCapabilities:
        """What SearXNG can express (SR-B4).

        :return: :data:`SEARXNG_CAPABILITIES`
        :rtype: ProviderCapabilities
        """
        return SEARXNG_CAPABILITIES

    async def search(self, request: SearchRequest, *, timeout_seconds: float | None = None) -> CandidateSet:
        """Run one query and return typed candidates.

        :param request: what the caller asked for
        :ptype request: SearchRequest
        :param timeout_seconds: bound for this call (SR-G2); None leaves
            the transport's configured timeout in force (SR-G1)
        :ptype timeout_seconds: float | None
        :return: candidates, one disposition per criterion, and spend. An
            empty candidate tuple is a success (SR-J2)
        :rtype: CandidateSet
        :raises SearchFailure: one of the typed classes, carrying spend
        """
        try:
            plan = self._plan(request)
            response = await self._exchange(plan, timeout_seconds=timeout_seconds)
            spend = self._spend_for(response)
            payload = self._decode(response, spend)
            candidates = self._candidates(request, payload, response, spend)
        except SearchFailure as failure:
            # every failure leaves this adapter fully attributed: which
            # instance, which egress, when (D8/D20, SR-A3) -- the record
            # riding ToolResult.metadata is the only fact that survives the
            # wire, so a consumer-side ban tracker rebuilds its key from it.
            attributed = self._attributed(failure)
            if attributed is failure:
                raise
            raise attributed from failure
        candidates = self._apply_local_criteria(candidates, plan)
        return CandidateSet(
            candidates=tuple(candidates),
            dispositions=tuple(plan.dispositions),
            spend=spend,
            notices=self._notices(payload),
        )

    # -- request planning ---------------------------------------------------

    def _plan(self, request: SearchRequest) -> _Plan:
        """Map the request's criteria onto wire parameters and dispositions.

        Every criterion the request carried gets exactly one answer, and an
        answer is never "nothing" (SR-B2, SR-B3).

        The ``categories`` ownership scan happens before the pass rather
        than inside it, because that precedence cannot be decided one
        criterion at a time: whether a ``carrier`` criterion may claim
        SearXNG's single ``categories`` parameter depends on whether a
        *usable* :data:`SEARXNG_PARAM_CATEGORIES` was asked for somewhere
        else in the same request, and criteria have no order the caller must
        respect. This is the shape Tavily's absolute-window pre-scan already
        has, for the same reason (RES-T4M9).

        :param request: the caller's request
        :ptype request: SearchRequest
        :return: the plan the exchange and the local filters both read
        :rtype: _Plan
        """
        params: dict[str, str] = {"q": request.query, "format": "json"}
        if self._default_categories:
            params["categories"] = ",".join(self._default_categories)
        if self._default_engines:
            params["engines"] = ",".join(self._default_engines)
        if self._default_safesearch is not None:
            params["safesearch"] = str(self._default_safesearch)
        plan = _Plan(params, categories_owner=self._categories_owner(request))
        for criterion in request.criteria:
            self._plan_criterion(plan, criterion)
        return plan

    def _categories_owner(self, request: SearchRequest) -> str | None:
        """Which criterion owns SearXNG's single ``categories`` parameter.

        Gated on the value being usable rather than merely present, the same
        way Tavily gates its absolute window: a malformed
        :data:`SEARXNG_PARAM_CATEGORIES` is going to be refused anyway, and
        letting it suppress a perfectly good ``carrier`` would lose both
        scopings to one typo.

        :param request: the caller's request
        :ptype request: SearchRequest
        :return: :data:`SEARXNG_PARAM_CATEGORIES` when the request carries a
            usable one, otherwise None -- leaving ``carrier`` free to claim it
        :rtype: str | None
        """
        for criterion in request.criteria:
            if criterion.key != SEARXNG_PARAM_CATEGORIES:
                continue
            if not isinstance(_names(criterion.value, criterion.key, "category names"), _Refused):
                return SEARXNG_PARAM_CATEGORIES
        return None

    def _plan_criterion(self, plan: _Plan, criterion: Criterion) -> None:
        """Fold one criterion into ``plan``.

        Dispatch is on the *key* alone. Value validation happens inside each
        branch, so a recognised key carrying an unusable value is answered
        ``unsatisfied`` by that branch and can never fall through to the
        ``ignored-unknown`` arm -- which would tell the caller this adapter
        does not know a criterion its own capabilities declare.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the criterion to handle
        :ptype criterion: Criterion
        """
        key = criterion.key
        if key == CRITERION_LANGUAGE:
            self._plan_language(plan, criterion)
        elif key == CRITERION_CARRIER:
            self._plan_carrier(plan, criterion)
        elif key == CRITERION_DOMAINS_INCLUDE:
            self._plan_domains(plan, criterion, include=True)
        elif key == CRITERION_DOMAINS_EXCLUDE:
            self._plan_domains(plan, criterion, include=False)
        elif key == CRITERION_MAX_RESULTS:
            self._plan_max_results(plan, criterion)
        elif key == CRITERION_TIME_RANGE:
            plan.answer(
                key,
                "unsatisfied",
                "SearXNG expresses only a relative window "
                f"({', '.join(SEARXNG_RELATIVE_TIME_RANGES)}), which is a different quantity from an "
                f"absolute range; widening the request to the nearest relative window would return "
                f"results outside it. Send '{SEARXNG_PARAM_TIME_RANGE}' for the relative window, or "
                f"apply the absolute range above this layer",
            )
        elif key == CRITERION_MIN_RESOLUTION:
            plan.answer(
                key,
                "unsatisfied",
                "SearXNG has no resolution filter. Image results carry a 'resolution' string, "
                f"published here as the '{FACET_WIDTH}'/'{FACET_HEIGHT}' facets, so the cull is "
                "applicable above this layer",
            )
        elif key == CRITERION_RIGHTS_CLASS:
            plan.answer(
                key,
                "unsatisfied",
                "SearXNG exposes no rights or licence filter, and reports no rights metadata to filter on locally",
            )
        elif key in SEARXNG_CAPABILITIES.namespaced_parameters:
            self._plan_namespaced(plan, criterion)
        else:
            # Only reachable by a key with no branch above -- which, since
            # every branch above is keyed off this module's own constants and
            # SEARXNG_CAPABILITIES' declaration, means a key this adapter
            # genuinely does not recognise. A well-known key with a value the
            # adapter cannot use was answered 'unsatisfied' by its own branch.
            plan.answer(key, "ignored-unknown", f"'{key}' is not a criterion this adapter recognises")

    def _plan_language(self, plan: _Plan, criterion: Criterion) -> None:
        """Push a language tag down as SearXNG's ``language`` parameter.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the language criterion
        :ptype criterion: Criterion
        """
        tag = plan.accept(
            criterion.key,
            _text(criterion.value, criterion.key, "a BCP 47 language tag as a non-empty string"),
            consequence="no language was sent and the instance's own default stayed in force",
        )
        if tag is None:
            return
        plan.params["language"] = tag
        plan.answer(criterion.key, "pushdown", "sent as SearXNG's 'language' parameter")

    def _plan_domains(self, plan: _Plan, criterion: Criterion, *, include: bool) -> None:
        """Record domain scoping for the local filter to apply.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the domain criterion
        :ptype criterion: Criterion
        :param include: whether this is the allow half or the deny half
        :ptype include: bool
        """
        half = "allow-list" if include else "deny-list"
        domains = plan.accept(
            criterion.key,
            _names(criterion.value, criterion.key, "a domain string or a list of domain strings"),
            consequence="no domain filter was applied and the full result set is returned",
        )
        if domains is None:
            return
        lowered = tuple(domain.lower() for domain in domains)
        if include:
            plan.domains_include = lowered
        else:
            plan.domains_exclude = lowered
        plan.answer(criterion.key, "local", f"SearXNG has no domain {half}; results are filtered here by hostname")

    def _plan_max_results(self, plan: _Plan, criterion: Criterion) -> None:
        """Record the result cap for the local filter to apply.

        The validation is not decoration: the cap feeds a slice, and a slice
        reads ``-1`` as "drop the last candidate" and ``0`` as "return
        nothing" -- one the opposite of a cap, the other an empty answer
        reported as a clean success. Both would carry a disposition saying
        the cap was honoured, which is the failure mode SR-B3 forbids.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the ``max-results`` criterion
        :ptype criterion: Criterion
        """
        cap = plan.accept(
            criterion.key,
            _counting_number(criterion.value, criterion.key, "a positive integer count"),
            consequence="no cap was applied and every candidate the page returned is kept",
        )
        if cap is None:
            return
        plan.max_results = cap
        plan.answer(criterion.key, "local", "SearXNG returns a full page; the cap is applied here after parsing")

    def _plan_carrier(self, plan: _Plan, criterion: Criterion) -> None:
        """Map a carrier criterion onto a SearXNG category.

        Loses the ``categories`` parameter to an explicit
        :data:`SEARXNG_PARAM_CATEGORIES` in the same request, and says so
        rather than being overwritten while still claiming pushdown.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the carrier criterion
        :ptype criterion: Criterion
        """
        key = criterion.key
        carrier = plan.accept(
            key,
            _text(criterion.value, key, "a media-contracts carrier name as a non-empty string"),
            consequence="the search was not scoped to a category",
        )
        if carrier is None:
            return
        category = _CARRIER_CATEGORIES.get(carrier)
        if category is None:
            plan.answer(
                key,
                "unsatisfied",
                f"no SearXNG category means '{carrier}'; the carriers it can scope are "
                f"{', '.join(sorted(_CARRIER_CATEGORIES))}",
            )
            return
        if plan.categories_owner is not None and plan.categories_owner != key:
            plan.answer(
                key,
                "unsatisfied",
                f"SearXNG scopes by exactly one 'categories' parameter, and this request also carries "
                f"'{plan.categories_owner}', which takes it: that criterion names SearXNG's own category "
                f"vocabulary directly, while a carrier is mapped onto it here. The two are not merged, "
                f"because SearXNG *unions* the categories it is given -- 'images,news' searches both -- so "
                f"merging would widen the search past what either criterion asked for rather than "
                f"intersecting them. To scope by '{carrier}', drop '{plan.categories_owner}' or include "
                f"'{category}' in its value",
            )
            return
        plan.params["categories"] = category
        plan.answer(key, "pushdown", f"sent as SearXNG's categories='{category}'")

    def _plan_namespaced(self, plan: _Plan, criterion: Criterion) -> None:
        """Push a ``searxng:``-namespaced parameter down to the provider.

        Every value is checked against what SearXNG actually accepts before
        it reaches the wire. A bad one refused here costs nothing; sent, it
        buys either an error the caller has to decode or -- worse for
        ``safesearch`` -- a silently-applied instance default under a
        disposition claiming the requested level was in force.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the namespaced criterion
        :ptype criterion: Criterion
        """
        key = criterion.key
        if key == SEARXNG_PARAM_TIME_RANGE:
            self._plan_relative_window(plan, criterion)
        elif key == SEARXNG_PARAM_CATEGORIES:
            self._plan_categories(plan, criterion)
        elif key == SEARXNG_PARAM_ENGINES:
            self._plan_engines(plan, criterion)
        elif key == SEARXNG_PARAM_SAFESEARCH:
            self._plan_safesearch(plan, criterion)
        elif key == SEARXNG_PARAM_PAGE:
            self._plan_page(plan, criterion)
        else:
            # A parameter declared in SEARXNG_CAPABILITIES with no branch
            # here. An answer is still owed (SR-B2), and the honest one is
            # that the declaration outran the mapping.
            plan.answer(
                key,
                "unsatisfied",
                f"'{key}' is declared in this adapter's capabilities but is not mapped onto a SearXNG "
                f"parameter, so nothing was sent for it",
            )

    def _plan_relative_window(self, plan: _Plan, criterion: Criterion) -> None:
        """Push SearXNG's relative publication window down.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the ``searxng:time-range`` criterion
        :ptype criterion: Criterion
        """
        key = criterion.key
        accepted = ", ".join(SEARXNG_RELATIVE_TIME_RANGES)
        window = plan.accept(
            key,
            _text(criterion.value, key, f"one of SearXNG's relative windows ({accepted}) as a string"),
            consequence="no publication window was sent",
        )
        if window is None:
            return
        if window not in SEARXNG_RELATIVE_TIME_RANGES:
            plan.answer(key, "unsatisfied", f"'{window}' is not a SearXNG relative window; accepted: {accepted}")
            return
        plan.params["time_range"] = window
        plan.answer(key, "pushdown", "sent as SearXNG's 'time_range' parameter")

    def _plan_categories(self, plan: _Plan, criterion: Criterion) -> None:
        """Push explicit SearXNG categories down, winning the slot.

        An unrecognised category name is passed through rather than refused:
        :data:`SEARXNG_CATEGORIES` is what a stock instance ships, and an
        operator can define more in ``settings.yml``, so refusing an unknown
        name here would refuse a category that works on this deployment.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the ``searxng:categories`` criterion
        :ptype criterion: Criterion
        """
        key = criterion.key
        names = plan.accept(
            key,
            _names(criterion.value, key, "a category name or a list of category names"),
            consequence="the search was not scoped to a category",
        )
        if names is None:
            return
        plan.params["categories"] = ",".join(names)
        plan.answer(key, "pushdown", "sent as SearXNG's 'categories' parameter")

    def _plan_engines(self, plan: _Plan, criterion: Criterion) -> None:
        """Restrict the search to named engines.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the ``searxng:engines`` criterion
        :ptype criterion: Criterion
        """
        key = criterion.key
        names = plan.accept(
            key,
            _names(criterion.value, key, "an engine name or a list of engine names"),
            consequence="the search was not restricted to any engine",
        )
        if names is None:
            return
        plan.params["engines"] = ",".join(names)
        plan.answer(key, "pushdown", "sent as SearXNG's 'engines' parameter")

    def _plan_safesearch(self, plan: _Plan, criterion: Criterion) -> None:
        """Push a safesearch level down, within SearXNG's own vocabulary.

        The levels are integers in SearXNG's settings vocabulary -- 0 off,
        1 moderate, 2 strict -- and this is the criterion where an
        unvalidated value is worst: SearXNG applies its configured default
        for anything it cannot read, so ``"high"`` sent blind returns
        whatever the instance felt like under a disposition claiming strict
        filtering was applied.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the ``searxng:safesearch`` criterion
        :ptype criterion: Criterion
        """
        key = criterion.key
        levels = SEARXNG_CAPABILITIES.safesearch_levels or ()
        expected = f"one of SearXNG's safesearch levels as an integer ({', '.join(str(level) for level in levels)})"
        level = plan.accept(
            key,
            _level(criterion.value, levels, key, expected),
            consequence="the instance's own configured level stayed in force",
        )
        if level is None:
            return
        plan.params["safesearch"] = str(level)
        plan.answer(key, "pushdown", "sent as SearXNG's 'safesearch' parameter")

    def _plan_page(self, plan: _Plan, criterion: Criterion) -> None:
        """Push a 1-based page number down as SearXNG's ``pageno``.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the ``searxng:page`` criterion
        :ptype criterion: Criterion
        """
        key = criterion.key
        page = plan.accept(
            key,
            _counting_number(criterion.value, key, "a 1-based page number as a positive integer"),
            consequence="the first page was requested",
        )
        if page is None:
            return
        plan.params["pageno"] = str(page)
        plan.answer(key, "pushdown", "sent as SearXNG's 'pageno' parameter")

    # -- the exchange -------------------------------------------------------

    async def _exchange(self, plan: _Plan, *, timeout_seconds: float | None) -> TransportResponse:
        """Perform the request, mapping every failure onto the taxonomy.

        :param plan: the planned request
        :ptype plan: _Plan
        :param timeout_seconds: per-call bound, or None for the
            transport's configured timeout
        :ptype timeout_seconds: float | None
        :return: the completed exchange
        :rtype: TransportResponse
        :raises SearchFailure: mapped from the provider's status or from
            whatever the transport raised
        """
        headers = {"Accept": "application/json", **self._credentials}
        try:
            response = await self._transport.request(
                "GET",
                f"{self._base_url}/search",
                headers=headers,
                params=plan.params,
                timeout_seconds=timeout_seconds,
            )
        except SearchFailure as failure:
            # a transport that already speaks the taxonomy (this package's
            # ``standalone`` module) knows things the adapter cannot -- how
            # many attempts, how long, how many bytes. Keep its record and
            # add what only the adapter knows: which provider instance.
            raise self._attributed(failure) from failure
        except TimeoutError as exc:
            raise TimedOut(
                f"searxng instance {self._provider_instance} did not answer within the call's bound",
                spend=Spend(calls=0),
                provider_instance=self._provider_instance,
            ) from exc
        except Exception as exc:
            # the protocol lets an implementation raise whatever it likes,
            # and an adapter must not import an HTTP library to name those
            # exceptions. Unrecognised means "gave up at the transport",
            # which is what TransportFailed says (SR-J1).
            raise TransportFailed(
                f"transport failed reaching searxng instance {self._provider_instance}: {type(exc).__name__}: {exc}",
                spend=Spend(calls=0),
                provider_instance=self._provider_instance,
            ) from exc
        self._raise_for_status(response)
        return response

    def _attributed(self, failure: SearchFailure) -> SearchFailure:
        """Re-stamp a failure with what only this adapter can attribute (D8/D20, SR-A3).

        Shared with Tavily's identical method as :func:`_common.attributed_failure`:
        which instance, which egress, when, is the same three facts for every provider.

        :param failure: the failure about to leave this adapter
        :ptype failure: SearchFailure
        :return: the same failure class, fully attributed
        :rtype: SearchFailure
        """
        return attributed_failure(
            failure, provider_instance=self._provider_instance, egress_name=self._transport.egress_name
        )

    def _raise_for_status(self, response: TransportResponse) -> None:
        """Map a non-2xx status onto the typed taxonomy.

        :param response: the completed exchange
        :ptype response: TransportResponse
        :raises SearchFailure: for every status outside 2xx
        """
        status = response.status_code
        if 200 <= status < 300:
            return
        spend = self._spend_for(response)
        message = f"searxng instance {self._provider_instance} answered HTTP {status}"
        if status == 429:
            raise RateLimited(
                message,
                spend=spend,
                provider_instance=self._provider_instance,
                remediation="the instance is pacing us; lower the configured rate for this instance",
                retry_after_seconds=_as_float(response.headers.get("retry-after")),
            )
        if status == 403:
            raise MalformedResponse(
                message, spend=spend, provider_instance=self._provider_instance, remediation=SEARXNG_403_REMEDIATION
            )
        if status in {401, 407}:
            raise AuthFailed(
                message,
                spend=spend,
                provider_instance=self._provider_instance,
                remediation="the instance rejected the credentials this deployment supplied",
            )
        if status == 402:
            raise QuotaExhausted(message, spend=spend, provider_instance=self._provider_instance)
        if status in {408, 504}:
            raise TimedOut(message, spend=spend, provider_instance=self._provider_instance)
        raise TransportFailed(message, spend=spend, provider_instance=self._provider_instance)

    def _spend_for(self, response: TransportResponse) -> Spend:
        """What one completed exchange consumed (SR-E1, D6).

        Money and weighted units are zero: a self-hosted instance bills
        nothing, and inventing a synthetic infrastructure price would put a
        number nobody can reconcile into a budget (D6). ``calls`` is 1
        because the provider served this exchange -- the count a rate cap
        enforces is the count a bill would price (SR-E2), and attempts that
        never reached it are not in it (D4).

        :param response: the completed exchange
        :ptype response: TransportResponse
        :return: the spend for this call
        :rtype: Spend
        """
        return Spend(
            money=Decimal("0"),
            wall_clock_seconds=response.elapsed_seconds,
            calls=1,
            provider_units=Decimal("0"),
            bytes_transferred=len(response.body),
        )

    def _decode(self, response: TransportResponse, spend: Spend) -> Mapping[str, object]:
        """Parse the JSON body SearXNG returned.

        Shared shape with Tavily's identical method as
        :func:`_common.decode_results_payload`; SearXNG's own contribution is
        the 403 remediation carried on a non-JSON body, since a body that is
        not JSON at all is the SR-J1 teaching case (``SEARXNG_403_REMEDIATION``).

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
            response,
            spend,
            provider_name="searxng",
            provider_instance=self._provider_instance,
            not_json_remediation=SEARXNG_403_REMEDIATION,
        )

    def _notices(self, payload: Mapping[str, object]) -> tuple[str, ...]:
        """Degradations SearXNG reported about its own fan-in.

        An engine that did not answer means the result set is narrower than
        the query asked for, and a caller that reads it as complete draws a
        conclusion from an absence that was an outage.

        ``answers``, ``infoboxes``, ``suggestions`` and ``corrections`` are
        deliberately not mapped: they are a different shape from a candidate
        and giving them a candidate's fields would be an invention. They are
        left for a layer that has somewhere honest to put them.

        :param payload: the decoded provider payload
        :ptype payload: Mapping[str, object]
        :return: the notices, empty when nothing was reported wrong
        :rtype: tuple[str, ...]
        """
        unresponsive = payload.get("unresponsive_engines")
        if not isinstance(unresponsive, list) or not unresponsive:
            return ()
        names = sorted({name for name in (_engine_name(entry) for entry in unresponsive) if name})
        if not names:
            return ()
        _logger.warning(
            "searxng instance %s reported unresponsive engines: %s", self._provider_instance, ", ".join(names)
        )
        return (f"searxng engines did not answer, so this result set is narrower than the query: {', '.join(names)}",)

    # -- candidates ---------------------------------------------------------

    def _candidates(
        self,
        request: SearchRequest,
        payload: Mapping[str, object],
        response: TransportResponse,
        spend: Spend,
    ) -> list[Candidate]:
        """Turn the provider's result list into typed candidates.

        :param request: the request being answered, for provenance
        :ptype request: SearchRequest
        :param payload: the decoded provider payload
        :ptype payload: Mapping[str, object]
        :param response: the exchange, for egress attribution
        :ptype response: TransportResponse
        :param spend: what the call consumed, carried onto any failure
        :ptype spend: Spend
        :return: the candidates, in provider order
        :rtype: list[Candidate]
        :raises MalformedResponse: when a result carries no usable URL --
            a result with no locator is not a result, and skipping it
            silently would report a narrower set as complete
        """
        raw_results = payload.get("results")
        results = raw_results if isinstance(raw_results, list) else []
        retrieved_at = datetime.now(UTC)
        candidates: list[Candidate] = []
        for index, raw in enumerate(results):
            if not isinstance(raw, dict):
                raise MalformedResponse(
                    f"searxng instance {self._provider_instance} returned a non-object at results[{index}]",
                    spend=spend,
                    provider_instance=self._provider_instance,
                )
            candidates.append(self._candidate(request, raw, response, retrieved_at, spend, index))
        return candidates

    def _candidate(
        self,
        request: SearchRequest,
        raw: Mapping[str, object],
        response: TransportResponse,
        retrieved_at: datetime,
        spend: Spend,
        index: int,
    ) -> Candidate:
        """Build one candidate from one SearXNG result object.

        :param request: the request being answered
        :ptype request: SearchRequest
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
        :return: the typed candidate
        :rtype: Candidate
        :raises MalformedResponse: when the result carries no usable URL
        """
        url = raw.get("url")
        if not isinstance(url, str) or not url:
            raise MalformedResponse(
                f"searxng instance {self._provider_instance} returned results[{index}] with no url",
                spend=spend,
                provider_instance=self._provider_instance,
            )
        image_source = raw.get("img_src")
        thumbnail = raw.get("thumbnail_src") or raw.get("thumbnail")
        locators = [Locator(url=url, rel="canonical")]
        if isinstance(image_source, str) and image_source:
            locators.append(Locator(url=image_source, rel="direct-file"))
        if isinstance(thumbnail, str) and thumbnail:
            locators.append(Locator(url=thumbnail, rel="thumbnail"))
        has_bytes = any(locator.rel == "direct-file" for locator in locators)
        return Candidate(
            identity=url,
            locators=tuple(locators),
            provenance=self._provenance(request, raw, response, retrieved_at),
            title=_as_str(raw.get("title")),
            snippet=_as_str(raw.get("content")),
            published_at=_as_published_at(raw.get("publishedDate")),
            scores=self._scores(raw),
            fidelity_available=FIDELITY_BYTES if has_bytes else FIDELITY_SNIPPET,
            fidelity_achieved=FIDELITY_SNIPPET,
            content=None,
            facets=self._facets(raw, has_bytes=has_bytes),
        )

    def _provenance(
        self,
        request: SearchRequest,
        raw: Mapping[str, object],
        response: TransportResponse,
        retrieved_at: datetime,
    ) -> Provenance:
        """Record where this candidate came from (SR-A3, D20).

        Engine attribution is provenance rather than a score: which engine
        surfaced a result is a fact about its origin, and P2 protects it.

        :param request: the request being answered
        :ptype request: SearchRequest
        :param raw: one provider result object
        :ptype raw: Mapping[str, object]
        :param response: the exchange, for the egress it left by
        :ptype response: TransportResponse
        :param retrieved_at: retrieval time
        :ptype retrieved_at: datetime
        :return: the candidate's provenance
        :rtype: Provenance
        """
        provider_ids: dict[str, str] = {}
        for field, value in (
            ("engine", raw.get("engine")),
            ("category", raw.get("category")),
            ("template", raw.get("template")),
            ("published_date_raw", raw.get("publishedDate")),
            ("img_format", raw.get("img_format")),
            ("resolution", raw.get("resolution")),
        ):
            text = _as_str(value)
            if text is not None:
                provider_ids[field] = text
        engines = _string_list(raw.get("engines"))
        if engines:
            provider_ids["engines"] = ",".join(engines)
        positions = _numbers(raw.get("positions"))
        if positions:
            provider_ids["positions"] = ",".join(f"{position:g}" for position in positions)
        return Provenance(
            query=request.query,
            provider_instance=self._provider_instance,
            provider_ids=provider_ids,
            retrieved_at=retrieved_at,
            egress=response.egress,
            producer=PRODUCER_API_PROVIDER,
        )

    def _scores(self, raw: Mapping[str, object]) -> tuple[ScoreEntry, ...]:
        """Every judgment SearXNG reported, named and scaled (D1, SR-A4).

        SearXNG's ``score`` is an engine-fusion weight on no bounded scale,
        so it is published as unbounded and non-comparable across providers
        -- which :meth:`ScoreEntry.provider_native` enforces rather than
        trusting a caller to remember. Its ``positions`` list is the ranks
        the contributing engines gave, which is a second, differently-scaled
        judgment rather than a restatement of the first.

        :param raw: one provider result object
        :ptype raw: Mapping[str, object]
        :return: the score entries, empty when the provider reported none
        :rtype: tuple[ScoreEntry, ...]
        """
        scores: list[ScoreEntry] = []
        weight = _as_float(raw.get("score"))
        if weight is not None:
            scores.append(
                ScoreEntry.provider_native(
                    name="engine-fusion-weight",
                    value=weight,
                    scale=SCALE_UNBOUNDED,
                    provider_instance=self._provider_instance,
                )
            )
        ranks = _numbers(raw.get("positions"))
        if ranks:
            scores.append(
                ScoreEntry.provider_native(
                    name="best-engine-position",
                    value=min(ranks),
                    scale=SCALE_RANK,
                    provider_instance=self._provider_instance,
                )
            )
        return tuple(scores)

    def _facets(self, raw: Mapping[str, object], *, has_bytes: bool) -> dict[str, JsonValue]:
        """Carrier facets, in the ``media-contracts`` vocabulary (SR-C2/C3).

        :param raw: one provider result object
        :ptype raw: Mapping[str, object]
        :param has_bytes: whether a direct-file locator was found
        :ptype has_bytes: bool
        :return: the facets this result asserts; unset facets are absent
            rather than null, so absence means "not reported"
        :rtype: dict[str, JsonValue]
        """
        facets: dict[str, JsonValue] = {FACET_HAS_DOWNLOADABLE_DATA: has_bytes}
        category = raw.get("category")
        if isinstance(category, str) and category in _CATEGORY_CARRIERS:
            facets[FACET_MEDIA_CATEGORY] = _CATEGORY_CARRIERS[category]
        facets[FACET_LOCATOR_KIND] = "direct-file" if has_bytes else "containing-page"
        resolution = raw.get("resolution")
        if isinstance(resolution, str) and "x" in resolution:
            width, _, height = resolution.partition("x")
            if width.strip().isdigit() and height.strip().isdigit():
                facets[FACET_WIDTH] = int(width.strip())
                facets[FACET_HEIGHT] = int(height.strip())
        rights = raw.get("license") or raw.get("license_name")
        if isinstance(rights, str) and rights:
            facets[FACET_RIGHTS_STATUS] = rights
        return facets

    def _apply_local_criteria(self, candidates: list[Candidate], plan: _Plan) -> list[Candidate]:
        """Apply the criteria this adapter declared it applies locally.

        Declaring ``local`` is a promise, so this is where it is kept:
        domain scoping SearXNG cannot express, and the result cap it has no
        parameter for. Filtering happens before the cap so a cap of 5 with
        a domain filter yields five in-scope candidates rather than five
        pre-filter ones.

        :param candidates: the parsed candidates, in provider order
        :ptype candidates: list[Candidate]
        :param plan: the plan carrying the local filters
        :ptype plan: _Plan
        :return: the candidates that survive
        :rtype: list[Candidate]
        """
        kept = candidates
        if plan.domains_include:
            kept = [
                candidate
                for candidate in kept
                if any(_domain_matches(_hostname(candidate.identity), domain) for domain in plan.domains_include)
            ]
        if plan.domains_exclude:
            kept = [
                candidate
                for candidate in kept
                if not any(_domain_matches(_hostname(candidate.identity), domain) for domain in plan.domains_exclude)
            ]
        if plan.max_results is not None:
            kept = kept[: plan.max_results]
        return kept
