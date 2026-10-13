"""Anthropic's Models API as a source of request-settings facts.

Anthropic publishes, per model, which request settings the model accepts:
``GET https://api.anthropic.com/v1/models/{id}`` (and the paginated
``GET /v1/models``) returns a ``capabilities`` tree with a ``supported``
boolean at every leaf. This module reads the parts of that tree that answer
"what may a request for this model carry" and records them in the
provider-published layer of the capability registry
(:func:`~threetears.models.capabilities.register_published_capabilities`),
which sits above the static entries the provider modules register and below
a deployment's explicit override.

The mapping, published leaf to :class:`~threetears.models.capabilities.ModelCapabilities` field:

========================================================  =======================
``capabilities.effort.supported`` + ``effort.<word>``      ``reasoning_efforts``
``capabilities.thinking.types.enabled.supported``          ``reasoning_budget``
``capabilities.thinking.types.adaptive.supported``         ``adaptive_reasoning``
``capabilities.thinking.types.disabled.supported``         ``reasoning_disable``
``capabilities.structured_outputs.supported``              ``structured_output``
========================================================  =======================

Temperature is not published, so ``accepts_temperature`` is never set here.

Three failure classes, handled differently on purpose:

* **The fetch failed** (transport error, an HTTP error status, a body that is
  not JSON). The load returns a :class:`PublishedCapabilityLoad` whose
  ``failure`` names the error, logs a WARNING, and changes nothing in the
  registry: the static facts (and any earlier published facts) stand. A load
  never raises for this, so a host may run it anywhere, including in front of
  a model call.
* **A key this module does not know** (a new capability, a new effort word).
  Ignored, so a provider adding to the tree never breaks a running build.
* **A key this module does know, in a shape it cannot read** (``supported``
  that is not a boolean, ``effort`` that is not an object). Raises
  :class:`PublishedCapabilityShapeError` before anything is recorded. That is a
  change to the provider's contract, not a transient fault, and reading it as
  "not supported" would hand every caller a wrong fact.

Authentication: the functions take a caller-supplied client of the
:class:`~threetears.models.price_lookup.HttpClient` shape (one ``get(url)``),
so the caller owns the key, the timeout and the client lifecycle. Build it
with :func:`anthropic_models_api_headers`::

    async with httpx.AsyncClient(headers=anthropic_models_api_headers(key), timeout=10.0) as client:
        load = await load_anthropic_capabilities(client, model_id)
        if not load.ok:
            report(load.failure)

Results are cached per process: a model loaded successfully within
``max_age_seconds`` is answered from the registry without a fetch; pass
``max_age_seconds=0`` to read afresh.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode

from threetears.models.capabilities import (
    CapabilityOverride,
    get_published_capabilities,
    register_published_capabilities,
)
from threetears.models.price_lookup import HttpClient
from threetears.observe import get_logger

__all__ = [
    "ANTHROPIC_MODELS_API_SOURCE",
    "ANTHROPIC_MODELS_API_VERSION",
    "ANTHROPIC_MODELS_URL",
    "DEFAULT_PUBLISHED_CAPABILITY_MAX_AGE_SECONDS",
    "PublishedCapabilityLoad",
    "PublishedCapabilityShapeError",
    "anthropic_models_api_headers",
    "anthropic_published_facts",
    "load_all_anthropic_capabilities",
    "load_anthropic_capabilities",
]

log = get_logger(__name__)

#: the Models API collection URL; one model is ``{ANTHROPIC_MODELS_URL}/{id}``.
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
#: the ``anthropic-version`` header value the Models API is read with.
ANTHROPIC_MODELS_API_VERSION = "2023-06-01"
#: the ``source`` a :class:`PublishedCapabilityLoad` from this module carries.
ANTHROPIC_MODELS_API_SOURCE = "anthropic-models-api"
#: how long a successful load answers from the per-process cache. the published
#: tree changes when Anthropic changes a model, which is rare; a host that wants
#: a fresh read sooner passes ``max_age_seconds=0``.
DEFAULT_PUBLISHED_CAPABILITY_MAX_AGE_SECONDS = 24 * 60 * 60.0

#: the reasoning effort ladder, lowest first. only these words are read out of
#: ``capabilities.effort``: any other key there is unknown and ignored (a new
#: word is not claimed until this build knows where it sits on the ladder).
_EFFORT_LADDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
#: the ``thinking.types`` leaves read, and the field each fills.
_THINKING_TYPE_FIELDS = (
    ("enabled", "reasoning_budget"),
    ("adaptive", "adaptive_reasoning"),
    ("disabled", "reasoning_disable"),
)
#: page size for the collection listing (the API's maximum).
_LIST_PAGE_LIMIT = 1000

_CACHE_LOCK = threading.Lock()
#: model id -> monotonic time of its last successful load.
_LOADED_AT: dict[str, float] = {}
#: the last successful whole-collection load as (monotonic time, model ids);
#: empty until one succeeds.
_CATALOGUE_LOADED: list[tuple[float, tuple[str, ...]]] = []


class PublishedCapabilityShapeError(ValueError):
    """a capability key this build reads came back in a shape it cannot read.

    raised before anything is recorded, so the registry is unchanged. the
    provider changed its published contract: update
    :mod:`threetears.models.anthropic_capabilities` to read the new shape.

    :param path: dotted path of the offending node in the model object
    :ptype path: str
    :param value: the value found there
    :ptype value: Any
    """

    def __init__(self, path: str, value: Any) -> None:
        self.path = path
        self.value = value
        super().__init__(
            f"the Anthropic Models API returned {path} = {value!r}, which this build cannot read "
            "(expected an object with a boolean 'supported'). the published contract changed: "
            "update threetears.models.anthropic_capabilities to read it; nothing was recorded",
        )


@dataclass(frozen=True)
class PublishedCapabilityLoad:
    """the outcome of one load: what was recorded, or why nothing was.

    :param source: where the facts came from (``"anthropic-models-api"``)
    :ptype source: str
    :param facts: model id -> the facts recorded for it by this load (or, when
        ``from_cache``, the facts already recorded); empty on failure
    :ptype facts: Mapping[str, CapabilityOverride]
    :param failure: ``"<ExceptionType>: <message>"`` when the fetch failed,
        else ``None``. on failure the registry is unchanged
    :ptype failure: str | None
    :param from_cache: ``True`` when answered from the per-process cache with no fetch
    :ptype from_cache: bool
    """

    source: str
    facts: Mapping[str, CapabilityOverride] = field(default_factory=dict)
    failure: str | None = None
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        """``True`` unless the fetch failed.

        :return: whether the load succeeded
        :rtype: bool
        """
        return self.failure is None


def anthropic_models_api_headers(api_key: str) -> dict[str, str]:
    """the headers a client reading the Models API carries.

    :param api_key: an Anthropic API key
    :ptype api_key: str
    :return: ``x-api-key`` and ``anthropic-version`` headers
    :rtype: dict[str, str]
    """
    return {"x-api-key": api_key, "anthropic-version": ANTHROPIC_MODELS_API_VERSION}


def _node(value: Any, path: str) -> Mapping[str, Any]:
    """the object at ``path``, or a shape error.

    :param value: the value found at ``path``
    :ptype value: Any
    :param path: dotted path, for the error
    :ptype path: str
    :return: ``value`` as a mapping
    :rtype: Mapping[str, Any]
    :raises PublishedCapabilityShapeError: when ``value`` is not an object
    """
    if not isinstance(value, Mapping):
        raise PublishedCapabilityShapeError(path, value)
    return value


def _supported(value: Any, path: str) -> bool:
    """the ``supported`` boolean of the object at ``path``, or a shape error.

    :param value: the value found at ``path``
    :ptype value: Any
    :param path: dotted path, for the error
    :ptype path: str
    :return: the leaf's ``supported`` value
    :rtype: bool
    :raises PublishedCapabilityShapeError: when ``value`` is not an object or
        its ``supported`` is not a boolean
    """
    supported = _node(value, path).get("supported")
    if not isinstance(supported, bool):
        raise PublishedCapabilityShapeError(f"{path}.supported", supported)
    return supported


def anthropic_published_facts(model: Mapping[str, Any]) -> CapabilityOverride:
    """map one Models API model object onto the request-settings fields.

    pure: no fetch, no registry write. only the fields the object publishes
    are set on the result, so recording it leaves every other field to the
    static entry. a model object with no ``capabilities`` key publishes
    nothing and yields an empty result.

    :param model: one model object (``GET /v1/models/{id}`` or one ``data`` entry of ``GET /v1/models``)
    :ptype model: Mapping[str, Any]
    :return: the published facts, explicitly-set fields only
    :rtype: CapabilityOverride
    :raises PublishedCapabilityShapeError: when a key this function reads has a shape it cannot read
    """
    facts: dict[str, Any] = {}
    raw = model.get("capabilities")
    if raw is None:
        return CapabilityOverride()
    caps = _node(raw, "capabilities")

    if "effort" in caps:
        effort = _node(caps["effort"], "capabilities.effort")
        effort_on = _supported(effort, "capabilities.effort")
        words = [
            word
            for word in _EFFORT_LADDER
            if word in effort and _supported(effort[word], f"capabilities.effort.{word}")
        ]
        # effort.supported False means the request may carry no effort
        # parameter at all, whatever the per-word leaves say.
        facts["reasoning_efforts"] = words if effort_on else []

    if "thinking" in caps:
        thinking = _node(caps["thinking"], "capabilities.thinking")
        thinking_on = _supported(thinking, "capabilities.thinking")
        types = _node(thinking.get("types", {}), "capabilities.thinking.types")
        for type_name, field_name in _THINKING_TYPE_FIELDS:
            if type_name in types:
                accepted = _supported(types[type_name], f"capabilities.thinking.types.{type_name}")
                # an "on" mode is accepted only when thinking itself is.
                facts[field_name] = accepted and (thinking_on or type_name == "disabled")
            elif not thinking_on and type_name != "disabled":
                facts[field_name] = False

    if "structured_outputs" in caps:
        facts["structured_output"] = _supported(caps["structured_outputs"], "capabilities.structured_outputs")

    return CapabilityOverride(**facts)


async def _get_json(client: HttpClient, url: str) -> Any:
    """GET ``url`` and decode its JSON body; any transport, status or decode error propagates.

    :param client: caller-supplied client carrying the auth headers
    :ptype client: HttpClient
    :param url: the URL to read
    :ptype url: str
    :return: the decoded body
    :rtype: Any
    """
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


def _failed(url: str, exc: Exception, **context: str) -> PublishedCapabilityLoad:
    """log a fetch failure and build the load that reports it.

    :param url: the URL that failed
    :ptype url: str
    :param exc: the error
    :ptype exc: Exception
    :param context: extra fields for the log record
    :ptype context: str
    :return: a failed load; the registry is unchanged
    :rtype: PublishedCapabilityLoad
    """
    failure = f"{type(exc).__name__}: {exc}"
    log.warning(
        "Anthropic Models API read failed; the static capability facts stand",
        extra={"extra_data": {"url": url, "failure": failure[:512], **context}},
    )
    return PublishedCapabilityLoad(source=ANTHROPIC_MODELS_API_SOURCE, failure=failure)


def _fresh(loaded_at: float | None, max_age_seconds: float) -> bool:
    """whether a load at ``loaded_at`` is younger than ``max_age_seconds``.

    :param loaded_at: monotonic load time, or ``None`` when never loaded
    :ptype loaded_at: float | None
    :param max_age_seconds: the age limit
    :ptype max_age_seconds: float
    :return: ``True`` when the cached load may answer
    :rtype: bool
    """
    return loaded_at is not None and time.monotonic() - loaded_at < max_age_seconds


def _record(facts: Mapping[str, CapabilityOverride]) -> None:
    """record ``facts`` in the published layer and stamp their load time.

    :param facts: model id -> facts
    :ptype facts: Mapping[str, CapabilityOverride]
    """
    now = time.monotonic()
    for model_id, model_facts in facts.items():
        register_published_capabilities(model_id, model_facts)
    with _CACHE_LOCK:
        for model_id in facts:
            _LOADED_AT[model_id] = now


async def load_anthropic_capabilities(
    client: HttpClient,
    model_id: str,
    *,
    max_age_seconds: float = DEFAULT_PUBLISHED_CAPABILITY_MAX_AGE_SECONDS,
    models_url: str = ANTHROPIC_MODELS_URL,
) -> PublishedCapabilityLoad:
    """read one model's published capabilities and record them.

    the facts are recorded under ``model_id``, and also under the id the API
    answers with when that differs (an alias resolving to a dated id), since a
    caller may hold either.

    :param client: caller-supplied client carrying :func:`anthropic_models_api_headers`
    :ptype client: HttpClient
    :param model_id: the model id to read
    :ptype model_id: str
    :param max_age_seconds: a successful load younger than this answers from the cache
    :ptype max_age_seconds: float
    :param models_url: the Models API collection URL (a proxy or test endpoint)
    :ptype models_url: str
    :return: what was recorded, or the fetch failure (logged; registry unchanged)
    :rtype: PublishedCapabilityLoad
    :raises PublishedCapabilityShapeError: when a key this build reads has a shape it cannot read
    """
    with _CACHE_LOCK:
        loaded_at = _LOADED_AT.get(model_id)
    cached = get_published_capabilities(model_id)
    if cached is not None and _fresh(loaded_at, max_age_seconds):
        return PublishedCapabilityLoad(source=ANTHROPIC_MODELS_API_SOURCE, facts={model_id: cached}, from_cache=True)

    url = f"{models_url.rstrip('/')}/{quote(model_id, safe='')}"
    try:
        payload = await _get_json(client, url)
    except Exception as exc:  # the client is caller-supplied, so its error types are not ours to name
        return _failed(url, exc, model_id=model_id)

    model = _node(payload, "<model object>")
    model_facts = anthropic_published_facts(model)
    facts = {model_id: model_facts}
    answered_id = model.get("id")
    if isinstance(answered_id, str) and answered_id and answered_id != model_id:
        facts[answered_id] = model_facts
    _record(facts)
    return PublishedCapabilityLoad(source=ANTHROPIC_MODELS_API_SOURCE, facts=facts)


async def load_all_anthropic_capabilities(
    client: HttpClient,
    *,
    max_age_seconds: float = DEFAULT_PUBLISHED_CAPABILITY_MAX_AGE_SECONDS,
    models_url: str = ANTHROPIC_MODELS_URL,
) -> PublishedCapabilityLoad:
    """read every model the Models API lists and record each one's capabilities.

    follows the listing's pagination (``has_more`` / ``last_id``). every page
    is read and every model mapped before anything is recorded, so a failure
    or a shape error part-way leaves the registry exactly as it was.

    :param client: caller-supplied client carrying :func:`anthropic_models_api_headers`
    :ptype client: HttpClient
    :param max_age_seconds: a successful listing younger than this answers from the cache
    :ptype max_age_seconds: float
    :param models_url: the Models API collection URL (a proxy or test endpoint)
    :ptype models_url: str
    :return: what was recorded, or the fetch failure (logged; registry unchanged)
    :rtype: PublishedCapabilityLoad
    :raises PublishedCapabilityShapeError: when a listing or model object has a shape this build cannot read
    """
    with _CACHE_LOCK:
        catalogue_loaded_at, cached_ids = _CATALOGUE_LOADED[-1] if _CATALOGUE_LOADED else (None, ())
    if _fresh(catalogue_loaded_at, max_age_seconds):
        cached: dict[str, CapabilityOverride] = {}
        for cached_id in cached_ids:
            model_facts = get_published_capabilities(cached_id)
            if model_facts is not None:
                cached[cached_id] = model_facts
        return PublishedCapabilityLoad(source=ANTHROPIC_MODELS_API_SOURCE, facts=cached, from_cache=True)

    facts: dict[str, CapabilityOverride] = {}
    after_id: str | None = None
    while True:
        query: dict[str, str | int] = {"limit": _LIST_PAGE_LIMIT}
        if after_id is not None:
            query["after_id"] = after_id
        url = f"{models_url.rstrip('/')}?{urlencode(query)}"
        try:
            payload = await _get_json(client, url)
        except Exception as exc:  # the client is caller-supplied, so its error types are not ours to name
            return _failed(url, exc)
        page = _node(payload, "<model list>")
        data = page.get("data")
        if not isinstance(data, list):
            raise PublishedCapabilityShapeError("<model list>.data", data)
        for index, entry in enumerate(data):
            model = _node(entry, f"<model list>.data[{index}]")
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id:
                raise PublishedCapabilityShapeError(f"<model list>.data[{index}].id", model_id)
            facts[model_id] = anthropic_published_facts(model)
        last_id = page.get("last_id")
        if page.get("has_more") is not True or not isinstance(last_id, str) or not last_id:
            break
        after_id = last_id

    _record(facts)
    with _CACHE_LOCK:
        _CATALOGUE_LOADED[:] = [(time.monotonic(), tuple(facts))]
    return PublishedCapabilityLoad(source=ANTHROPIC_MODELS_API_SOURCE, facts=facts)
