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

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final
from urllib.parse import urlparse

from pydantic import JsonValue

from threetears.observe import get_logger
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
    CriterionDisposition,
    Disposition,
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


def _as_float(value: object) -> float | None:
    """Read ``value`` as a float, or ``None`` when it is not numeric.

    :param value: a JSON value from a provider payload
    :ptype value: object
    :return: the float, or None when the value cannot be one
    :rtype: float | None
    """
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    # Probing a provider value for a number: a non-numeric score is the provider
    # not reporting one, and the caller reads that as an absent score entry.
    # Absence is the honest answer, and there is nothing to log per result.
    try:
        return float(value)
    # NOSILENT: a non-numeric provider value means no score was reported
    except ValueError:
        return None


def _as_published_at(value: object) -> datetime | None:
    """Read a SearXNG ``publishedDate`` as a timezone-aware datetime.

    SearXNG's engines frequently report a naive ISO timestamp. This reads a
    naive value as UTC -- which is what those engines produce -- rather than
    discarding the date; the raw string stays on provenance so the
    assumption is inspectable rather than buried here.

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


def _as_str(value: object) -> str | None:
    """Read ``value`` as a non-empty string, or ``None``.

    :param value: a JSON value from a provider payload
    :ptype value: object
    :return: the string, or None when it is absent or not a string
    :rtype: str | None
    """
    return value if isinstance(value, str) and value else None


def _string_list(value: object) -> list[str]:
    """Read a provider value as a list of strings.

    :param value: a string, or a sequence of values to stringify
    :ptype value: object
    :return: the values as strings; empty when there are none
    :rtype: list[str]
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


class _Plan:
    """The request as SearXNG will receive it, with its honest dispositions.

    Not a contract type -- it never leaves this module. It exists so the
    criteria mapping is one readable pass that produces the wire parameters,
    the dispositions, and the local filters together, instead of three
    passes that can disagree.
    """

    def __init__(self, params: dict[str, str]) -> None:
        """Start a plan from the parameters every query carries.

        :param params: the base wire parameters (query, format, defaults)
        :ptype params: dict[str, str]
        """
        self.params = params
        self.dispositions: list[CriterionDisposition] = []
        self.max_results: int | None = None
        self.domains_include: tuple[str, ...] = ()
        self.domains_exclude: tuple[str, ...] = ()

    def answer(self, key: str, disposition: Disposition, detail: str | None = None) -> None:
        """Record how one criterion was handled.

        :param key: the criterion key being answered for
        :ptype key: str
        :param disposition: how it was handled
        :ptype disposition: Disposition
        :param detail: specifics -- why unsatisfiable, which rule applied
        :ptype detail: str | None
        """
        self.dispositions.append(CriterionDisposition(criterion_key=key, disposition=disposition, detail=detail))


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
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"base_url must be an absolute http(s) URL from deployment config, got {base_url!r}")
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
        plan = self._plan(request)
        response = await self._exchange(plan, timeout_seconds=timeout_seconds)
        spend = self._spend_for(response)
        payload = self._decode(response, spend)
        candidates = self._candidates(request, payload, response, spend)
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
        plan = _Plan(params)
        for criterion in request.criteria:
            self._plan_criterion(plan, criterion)
        return plan

    def _plan_criterion(self, plan: _Plan, criterion: Criterion) -> None:
        """Fold one criterion into ``plan``.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the criterion to handle
        :ptype criterion: Criterion
        """
        key = criterion.key
        if key == CRITERION_LANGUAGE and isinstance(criterion.value, str):
            plan.params["language"] = criterion.value
            plan.answer(key, "pushdown", "sent as SearXNG's 'language' parameter")
        elif key == CRITERION_CARRIER:
            self._plan_carrier(plan, criterion)
        elif key == CRITERION_DOMAINS_INCLUDE:
            plan.domains_include = tuple(domain.lower() for domain in _string_list(criterion.value))
            plan.answer(key, "local", "SearXNG has no domain allow-list; results are filtered here by hostname")
        elif key == CRITERION_DOMAINS_EXCLUDE:
            plan.domains_exclude = tuple(domain.lower() for domain in _string_list(criterion.value))
            plan.answer(key, "local", "SearXNG has no domain deny-list; results are filtered here by hostname")
        elif key == CRITERION_MAX_RESULTS and isinstance(criterion.value, int):
            plan.max_results = criterion.value
            plan.answer(key, "local", "SearXNG returns a full page; the cap is applied here after parsing")
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
            plan.answer(key, "ignored-unknown", f"'{key}' is not a criterion this adapter recognises")

    def _plan_carrier(self, plan: _Plan, criterion: Criterion) -> None:
        """Map a carrier criterion onto a SearXNG category.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the carrier criterion
        :ptype criterion: Criterion
        """
        carrier = criterion.value if isinstance(criterion.value, str) else ""
        category = _CARRIER_CATEGORIES.get(carrier)
        if category is None:
            plan.answer(
                criterion.key,
                "unsatisfied",
                f"no SearXNG category means '{carrier}'; the carriers it can scope are "
                f"{', '.join(sorted(_CARRIER_CATEGORIES))}",
            )
            return
        plan.params["categories"] = category
        plan.answer(criterion.key, "pushdown", f"sent as SearXNG's categories='{category}'")

    def _plan_namespaced(self, plan: _Plan, criterion: Criterion) -> None:
        """Push a ``searxng:``-namespaced parameter down to the provider.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the namespaced criterion
        :ptype criterion: Criterion
        """
        key = criterion.key
        if key == SEARXNG_PARAM_TIME_RANGE:
            window = str(criterion.value)
            if window not in SEARXNG_RELATIVE_TIME_RANGES:
                plan.answer(
                    key,
                    "unsatisfied",
                    f"'{window}' is not a SearXNG relative window; accepted: {', '.join(SEARXNG_RELATIVE_TIME_RANGES)}",
                )
                return
            plan.params["time_range"] = window
        elif key == SEARXNG_PARAM_CATEGORIES:
            plan.params["categories"] = ",".join(_string_list(criterion.value))
        elif key == SEARXNG_PARAM_ENGINES:
            plan.params["engines"] = ",".join(_string_list(criterion.value))
        elif key == SEARXNG_PARAM_SAFESEARCH:
            plan.params["safesearch"] = str(criterion.value)
        elif key == SEARXNG_PARAM_PAGE:
            plan.params["pageno"] = str(criterion.value)
        plan.answer(key, "pushdown", f"sent as SearXNG's '{key.split(':', 1)[1]}' parameter")

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
        """Re-stamp a transport-raised failure with this provider instance.

        :param failure: the failure the transport raised
        :ptype failure: SearchFailure
        :return: the same failure class, naming the provider instance
        :rtype: SearchFailure
        """
        if failure.provider_instance == self._provider_instance:
            return failure
        return failure.to_record().model_copy(update={"provider_instance": self._provider_instance}).to_failure()

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

        :param response: the successful exchange
        :ptype response: TransportResponse
        :param spend: what the call consumed, carried onto any failure
        :ptype spend: Spend
        :return: the decoded payload
        :rtype: Mapping[str, object]
        :raises MalformedResponse: when the body is not a JSON object, or
            carries no ``results`` list
        """
        try:
            payload = json.loads(response.body)
        except ValueError as exc:
            raise MalformedResponse(
                f"searxng instance {self._provider_instance} answered {response.status_code} with a "
                f"body that is not JSON",
                spend=spend,
                provider_instance=self._provider_instance,
                remediation=SEARXNG_403_REMEDIATION,
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise MalformedResponse(
                f"searxng instance {self._provider_instance} answered JSON without a 'results' list",
                spend=spend,
                provider_instance=self._provider_instance,
            )
        return payload

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
