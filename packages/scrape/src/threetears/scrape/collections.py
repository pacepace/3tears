"""ScrapeTarget / ScrapeRecipe / ScrapeExtraction -- domain-agnostic 3tears-scrape core.

Subclasses ``threetears.core.collections.base.BaseCollection`` directly,
rather than any consuming application's own collection base class -- the
discipline that let this module move out of the application it was written
in as a plain directory move rather than a disentangling exercise.

``BaseCollection`` provides the full three-tier (L1/L2/L3) cache machinery,
subscript access, CAS-mutate, and invalidation-publish for free -- a
subclass only has to implement ``table_name``, ``entity_class``, and the
five storage-tier primitives (``fetch_from_store``, ``save_to_store``,
``delete_from_store``, ``serialize``, ``deserialize``). It does **not**
provide an in-memory L3 fallback of its own; that convenience was an
application-side addition (a ``self._rows`` dict, plus registry/config
resolution from process-wide application state), which this package cannot
import. ``ScrapeCollection`` below re-implements the same shape locally,
minus the process-wide default resolution -- callers must pass
``registry``/``config`` explicitly. L3 is a real asyncpg pool
(``threetears.core.backends.protocol.DurableStore``-conforming) once
``threetears.scrape.migrations.apply_migrations()`` has run and the registry
carries an ``l3_pool``; otherwise CRUD falls back to the in-memory
``self._rows`` dict for the process lifetime -- this fallback is why unit
tests never need a real database, but it is NOT multi-pod-safe (each pod
gets its own dict), which is the whole reason the L3 branch below exists.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, ClassVar, Literal, cast, get_args

from threetears.core.backends.protocol import DurableStore
from threetears.core.collections.base import (
    NATS_CLIENT_FROM_REGISTRY,
    BaseCollection,
    EntityT,
)
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.observe import get_logger
from uuid_utils import uuid7

from .driver import NavStep
from .extraction import FieldSchema

__all__ = [
    "VALIDATION_STATUSES",
    "ScrapeExtraction",
    "ScrapeExtractionCollection",
    "ScrapeRecipe",
    "ScrapeRecipeCollection",
    "ScrapeTarget",
    "ScrapeTargetCollection",
    "decode_field_schema",
    "decode_nav_steps",
    "encode_field_schema",
    "encode_nav_steps",
    "ValidationStatus",
]

log = get_logger(__name__)


def _parse_dt(raw: Any) -> datetime | None:
    """Parse a possibly-ISO-string timestamp back into a ``datetime``.

    L2 (NATS KV) round-trips every value through JSON, which stringifies
    ``datetime`` on write (see :meth:`ScrapeCollection.serialize`) but does
    not parse it back on read -- so a value read through L2 arrives as a
    string even though the in-memory L3 fallback keeps it as a native
    ``datetime``.
    """
    if isinstance(raw, str) and raw:
        try:
            raw = datetime.fromisoformat(raw)
        except ValueError:  # NOSILENT: malformed timestamp, caller treats as absent
            raw = None
    if isinstance(raw, datetime):
        result = raw
    else:
        result = None
    return result


def _decode_json_field(raw: Any, default: Any) -> Any:
    """Decode a JSONB-shaped field, handling the same string/native split as :func:`_parse_dt`."""
    if raw is None:
        return default
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


#: A closed, explicit map -- never resolved via eval()/getattr() on a
#: caller-supplied string. Extend here (not by widening the resolution
#: mechanism) if a target's schema ever genuinely needs another primitive.
_FIELD_SCHEMA_TYPE_NAMES: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}


def encode_field_schema(schema: FieldSchema) -> dict[str, str]:
    """``{"employer": str}`` -> ``{"employer": "str"}``, JSON/YAML-safe (a live ``type`` object isn't).

    Callers constructing a :class:`ScrapeTarget` for persistence (a YAML
    loader, a database writer) must pre-encode ``field_schema`` with this
    before building the raw entity dict -- :class:`ScrapeTarget` stores
    whatever it's given verbatim (matching every other JSON-shaped field on
    this entity), it does not auto-encode on construction.
    """
    return {name: python_type.__name__ for name, python_type in schema.items()}


def decode_field_schema(raw: Any) -> FieldSchema:
    """Inverse of :func:`encode_field_schema`.

    :raises ValueError: if *raw* names a type outside :data:`_FIELD_SCHEMA_TYPE_NAMES`
        -- a typo'd/unsupported type name in a target's config must fail loudly at
        load time, not silently resolve to the wrong field type.
    """
    decoded = _decode_json_field(raw, {})
    result: FieldSchema = {}
    for name, type_name in decoded.items():
        python_type = _FIELD_SCHEMA_TYPE_NAMES.get(type_name)
        if python_type is None:
            raise ValueError(
                f"field_schema entry {name!r} names unsupported type {type_name!r}; "
                f"supported: {sorted(_FIELD_SCHEMA_TYPE_NAMES)}"
            )
        result[name] = python_type
    return result


def encode_nav_steps(steps: list[NavStep]) -> list[dict[str, Any]]:
    """``[NavStep(action="click", selector="#x")]`` -> ``[{"action": "click", "selector": "#x", ...}]``.

    Unlike :func:`encode_field_schema`, no type-name resolution is needed --
    every ``NavStep`` field is already a JSON-safe primitive, so this is a
    plain dataclass-to-dict conversion. Callers constructing a
    :class:`ScrapeTarget` for persistence must pre-encode ``nav_steps`` with
    this before building the raw entity dict, matching every other JSON-
    shaped field on this entity.
    """
    return [asdict(step) for step in steps]


def decode_nav_steps(raw: Any) -> list[NavStep] | None:
    """Inverse of :func:`encode_nav_steps`.

    :raises TypeError: if an entry names a field ``NavStep`` doesn't have --
        a typo'd nav step in a target's config must fail loudly at load
        time, not silently drop or misinterpret the step.
    """
    decoded = _decode_json_field(raw, None)
    if decoded is None:
        result: list[NavStep] | None = None
    else:
        result = [NavStep(**step) for step in decoded]
    return result


class ScrapeTarget(BaseEntity):
    """The config an operator adds to onboard a new scrape site.

    "Onboarding a state = a config addition, not a scraper" concretely
    means: adding one of these.
    """

    primary_key_field: str = "target_id"

    @property
    def target_id(self) -> str:
        """Stable key for this target (e.g. ``"warn_act_ca"``)."""
        return str(self._get_raw("target_id", ""))

    @property
    def url(self) -> str:
        """The page to fetch."""
        return str(self._get_raw("url", ""))

    @property
    def driver_backend(self) -> str:
        """Which ``ScrapeDriver`` backend renders this target.

        One of eight, resolved by the caller (this entity stores the string
        and never constructs a driver itself): ``"nodriver"`` (headless
        Chromium via the HTTP sidecar), ``"camoufox"`` (in-process stealth
        Firefox), ``"document"`` (PDF/DOCX/XLSX/CSV/TXT/MD/LaTeX parsed into
        synthetic HTML), ``"api"`` (a stateless JSON GET, needing
        :attr:`api_results_path`/:attr:`api_fragment_field`),
        ``"network_capture"`` (an authenticated in-session XHR captured
        during a real browser render), ``"multi_document"`` (a listing whose
        links each point at one whole document, needing
        :attr:`link_selector` or the ``api_*`` pair), ``"listing_detail"``
        (a listing table whose rows link to per-record detail pages that
        must be merged back into the row), and ``"nodriver_download"`` (a
        document behind a bot challenge a plain HTTP client can't pass).

        Deliberately a plain string rather than an enum: the set of backends
        a given deployment has wired up is the caller's business -- a
        consumer that never installs the sidecar shouldn't be forced to
        carry a symbol for it.
        """
        return str(self._get_raw("driver_backend", "nodriver"))

    @property
    def rate_limit_key(self) -> str:
        """Opaque string key; the core stores and passes it through but never resolves it.

        Resolution against a real rate-limit strategy happens in the
        consuming application's own scheduling code, which is exactly why
        this is a plain string rather than a strategy instance -- typing it
        as one would force this package to import the scheduler's own
        vocabulary, and the core has no opinion about how a key maps to a
        policy.
        """
        return str(self._get_raw("rate_limit_key", ""))

    @property
    def cadence(self) -> str:
        """How often this target is re-fetched; interpreted by the caller's scheduling layer, never here."""
        return str(self._get_raw("cadence", ""))

    @property
    def multi_row(self) -> bool:
        """Whether this target's page holds many records (a table/listing) rather than one.

        Selects which eval loop the caller's polling code runs:
        ``run_eval_loop_multi_row`` when ``True``, ``run_eval_loop`` (the
        original single-record path) when ``False`` (the default -- preserves
        every pre-existing target's behavior). Domain-agnostic: this is a
        statement about page shape, not about what the records mean, which
        is why it belongs on the core entity rather than in the consumer's
        scheduling code.
        """
        return bool(self._get_raw("multi_row", False))

    @property
    def wait_for(self) -> str | None:
        """CSS selector the driver waits for before considering the page settled.

        Passed straight through to ``ScrapeDriver.render(..., wait_for=...)``
        -- ``None`` (the default) keeps every pre-existing target's current
        behavior (a plain settle sleep). Some real pages need this: e.g. a
        target whose real content loads asynchronously well past the
        driver's default settle wait returns a near-empty page without it
        (live-verified, Nebraska's WARN listing, SCR-2N8W follow-up). A
        genuine input variable like ``url``/``cadence``, not a per-target
        extraction hack -- the eval loop still discovers its own selectors
        from whatever HTML this produces.
        """
        result: str | None = self._get_raw("wait_for", None)
        return result

    @property
    def field_schema(self) -> FieldSchema:
        """field_name -> expected Python type, for the eval loop's candidate
        generation and structural validation.

        Carried on the target itself rather than passed alongside it as a
        separate caller-supplied dict, so one config entry -- a YAML row, a
        database row -- fully describes both how to fetch a target and what
        to extract from it. The two-dict shape came first and did not
        survive contact with persistence: once target config had to
        round-trip through YAML and a database, the schema dict and the
        target dict could drift out of sync, and keeping them aligned needed
        its own dedicated "same key set in both" test. Still domain-
        agnostic: the core never interprets what a field NAME means, only
        its declared type.
        """
        return decode_field_schema(self._get_raw("field_schema"))

    @property
    def nav_steps(self) -> list[NavStep] | None:
        """Ordered browser actions the driver performs before the page is ready.

        Passed straight through to ``ScrapeDriver.render(..., nav_steps=...)``
        -- ``None`` (the default) keeps every pre-existing target's current
        behavior (plain navigation, no interaction). A genuine per-target
        input variable, the same category as ``wait_for``/``multi_row``: some
        real pages are only reachable by driving the browser through a search
        form or into a second page, and the driver needs to be told how,
        deterministically, since that's an orchestration concern -- the eval
        loop's own AI-driven extraction still runs unmodified on whatever
        HTML the driven-to page produces (multi-step navigation, 2026-07-14).
        """
        return decode_nav_steps(self._get_raw("nav_steps"))

    @property
    def extraction_strategy_type(self) -> str:
        """Which extraction-strategy shape the eval loop should propose.

        One of ``eval_loop.StrategyType``'s four values, passed straight
        through to ``run_eval_loop``/``run_eval_loop_multi_row(...,
        strategy_type=...)``: ``"css"`` (the default, preserving every
        pre-existing target's behavior -- CSS-selector candidates against an
        HTML table), ``"regex"`` (a page whose real content is prose/list
        text with no ``<table>`` structure at all -- Pennsylvania's real WARN
        page was rejected outright by the CSS candidate generator, which had
        no strategy shape it could even attempt a candidate in),
        ``"per_document"`` (independently-worded documents sharing no
        template, where no single cached pattern can generalize), and
        ``"multi_row_vision"`` (a table whose own structure defeats
        text-based table extraction, needing a vision read of the whole
        table at once).

        A statement about page shape, the same category as
        ``multi_row``/``wait_for`` -- the eval loop's own AI-driven
        extraction still runs unmodified, just proposing a different kind of
        candidate. Explicitly chosen per target in config and never
        auto-detected: see ``eval_loop.StrategyType``'s own docstring for the
        live case where two superficially identical page shapes needed
        opposite strategies.
        """
        return str(self._get_raw("extraction_strategy_type", "css"))

    @property
    def api_results_path(self) -> str | None:
        """Dotted JSON path to the list of per-record objects, for ``driver_backend: "api"``.

        Passed straight through to ``ScrapeDriver.render(..., results_path=...)``
        -- ``None`` (the default) is fine for every non-``"api"`` target,
        which ignores it. Required when ``driver_backend == "api"``
        (network/API-query capability, 2026-07-14) -- see ``drivers.api.ApiDriver``.
        """
        result: str | None = self._get_raw("api_results_path", None)
        return result

    @property
    def api_fragment_field(self) -> str | None:
        """Which field within each per-record JSON object holds the fragment to concatenate.

        Passed straight through to ``ScrapeDriver.render(..., fragment_field=...)``
        -- see :attr:`api_results_path`. Also doubles as ``MultiDocumentDriver``'s
        JSON discovery mode's document-URL field name (``driver_backend:
        "multi_document"`` with :attr:`api_results_path` set) -- both drivers
        read this exact YAML key, no separate field needed (multi-document
        capability, 2026-07-15).
        """
        result: str | None = self._get_raw("api_fragment_field", None)
        return result

    @property
    def link_selector(self) -> str | None:
        """CSS selector matching document links on a listing page.

        Passed straight through to ``ScrapeDriver.render(..., link_selector=...)``
        -- ``None`` (the default) is fine for every non-``"multi_document"``
        target, which ignores it. Required for ``MultiDocumentDriver``'s HTML
        discovery mode (multi-document capability, 2026-07-15) -- see
        ``drivers.multi_document.MultiDocumentDriver``. Its JSON discovery
        mode uses :attr:`api_results_path`/:attr:`api_fragment_field` instead
        (the same fields ``driver_backend: "api"`` targets already use).
        """
        result: str | None = self._get_raw("link_selector", None)
        return result

    @property
    def timeout_seconds(self) -> float:
        """Seconds to wait for this target's render before failing.

        Passed straight through to ``ScrapeDriver.render(..., timeout=...)``
        -- defaults to 30.0, the value every pre-existing target already got
        hardcoded at the call site, so this preserves current behavior for
        every target that doesn't set it explicitly. A genuine input
        variable, the same category as ``wait_for``/``nav_steps``: a target
        whose own ``nav_steps`` include a long ``wait_ms`` (a slow-hydrating
        JS framework, e.g. Oklahoma's Salesforce Aura page needing 15s alone
        just for its real data call to fire) can exceed the 30s default
        before its own settle logic even finishes (network_capture
        capability, 2026-07-15).
        """
        return float(self._get_raw("timeout_seconds", 30.0))


class ScrapeRecipe(BaseEntity):
    """The eval loop's memory: one row per target, holding its winning extraction strategy."""

    primary_key_field: str = "target_id"

    @property
    def target_id(self) -> str:
        """The target this recipe belongs to."""
        return str(self._get_raw("target_id", ""))

    @property
    def extraction_strategy(self) -> dict[str, Any]:
        """The winning candidate's strategy; its shape is the eval loop's business, not this schema's."""
        result: dict[str, Any] = _decode_json_field(self._get_raw("extraction_strategy"), {})
        return result

    @property
    def won_at(self) -> datetime | None:
        """When this recipe was chosen by the eval loop."""
        return _parse_dt(self._get_raw("won_at"))

    @property
    def last_validated_at(self) -> datetime | None:
        """When this recipe last passed validation on a real fetch."""
        return _parse_dt(self._get_raw("last_validated_at"))

    @property
    def consecutive_validation_failures(self) -> int:
        """Crossing ``eval_loop.DEFAULT_FAILURE_THRESHOLD`` re-triggers candidate generation."""
        return int(self._get_raw("consecutive_validation_failures", 0))


#: The outcome of one fetch, as stored on :attr:`ScrapeExtraction.validation_status`.
#:
#: ``"validated"`` -- records were extracted and confirmed.
#: ``"needs_review"`` -- structurally valid candidates existed but nothing confirmed them.
#: ``"failed"`` -- we received the page and could not extract from it.
#: ``"blocked"`` -- we never received the page: a bot wall or human-verification
#: interstitial stood where the content should be, so no records exist to have got right
#: or wrong and the target's extraction strategy is not implicated.
#:
#: A Literal rather than a bare ``str`` because the fourth value was added long after the
#: first three, across a dozen write sites, with the difference between "extraction failed"
#: and "we never got the page" carrying real consequences for anything that counts failures
#: or retries. Its sibling vocabularies (``challenge.PageVerdictKind``,
#: ``eval_loop.StrategyType``) are both Literals for the same reason.
#:
#: One value a consumer can see under this key is deliberately NOT here: ``ScrapeTool``'s
#: JSON payload reports ``"backoff"`` for a poll its fetch circuit suppressed. That is not a
#: validation outcome and is never stored -- a suppressed poll persists nothing at all,
#: because it observed nothing -- so admitting it to this Literal would declare a storable
#: value that can never be stored. Every one of the four above describes a page we did or
#: did not receive; ``"backoff"`` describes a fetch we declined to attempt.
ValidationStatus = Literal["validated", "needs_review", "failed", "blocked"]

#: Every value :data:`ValidationStatus` permits, derived from the Literal rather than
#: restated, so a new status cannot be added without this following it.
VALIDATION_STATUSES: frozenset[str] = frozenset(get_args(ValidationStatus))


class ScrapeExtraction(BaseEntity):
    """One row per fetch -- the actual output.

    Every field the eval loop or the enrichment pass populates defaults to
    an explicit "not yet" value rather than being assumed present. That is
    not just build-order residue: a row can still be written by a caller
    that runs neither pass (the fetch/persist path works on its own), and a
    consumer must be able to tell "no recipe produced this" from "a recipe
    produced this and found nothing."
    """

    primary_key_field: str = "id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """Inject the uuid7 primary key and pre-eval-loop field defaults.

        :param data: raw extraction fields; ``id`` is generated when absent
        :ptype data: dict[str, Any]
        :param is_new: whether this is a freshly created (unsaved) entity
        :ptype is_new: bool
        :param collection: owning collection, or ``None`` for transient use
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        normalized = dict(data)
        if not normalized.get("id"):
            normalized["id"] = str(uuid7())
        normalized.setdefault("extraction_recipe_id", None)
        normalized.setdefault("field_confidences", None)
        normalized.setdefault("enrichment_notes", None)
        normalized.setdefault("validation_status", "needs_review")
        super().__init__(normalized, is_new=is_new, collection=collection)

    @property
    def id(self) -> str:
        """This row's uuid7 primary key."""
        id_val: str = self._id
        return id_val

    @property
    def target_id(self) -> str:
        """The target this extraction was fetched from."""
        return str(self._get_raw("target_id", ""))

    @property
    def extraction_recipe_id(self) -> str | None:
        """Which recipe produced this row; ``None`` when no eval-loop recipe was involved."""
        result: str | None = self._get_raw("extraction_recipe_id")
        return result

    @property
    def source_url(self) -> str:
        """The URL actually fetched (post-redirect final URL)."""
        return str(self._get_raw("source_url", ""))

    @property
    def retrieved_at(self) -> datetime | None:
        """When this fetch happened."""
        return _parse_dt(self._get_raw("retrieved_at"))

    @property
    def structured_fields(self) -> dict[str, Any]:
        """The winning candidate's extracted values -- the row's actual payload."""
        result: dict[str, Any] = _decode_json_field(self._get_raw("structured_fields"), {})
        return result

    @property
    def field_confidences(self) -> dict[str, Any] | None:
        """Free-form eval-loop notes about this extraction; ``None`` when it never ran.

        Usually per-field confidence keyed by field name, which is what the column was
        named for. On a ``blocked`` row it instead carries ``page_verdict`` and
        ``page_verdict_evidence`` -- why the page was judged a wall, in the classifier's
        own words -- because there are no fields to have confidence about and this is the
        row's only free-form JSONB slot. Read it as "what the eval loop wanted to say
        about this extraction", not strictly as a per-field map.
        """
        result: dict[str, Any] | None = _decode_json_field(self._get_raw("field_confidences"), None)
        return result

    @property
    def enrichment_notes(self) -> dict[str, Any] | None:
        """The secondary LLM pass's free-form findings; ``None`` when enrichment never ran."""
        result: dict[str, Any] | None = _decode_json_field(self._get_raw("enrichment_notes"), None)
        return result

    @property
    def validation_status(self) -> str:
        """One of :data:`ValidationStatus`; defaults to ``"needs_review"``.

        ``"blocked"`` is the one that does not mean "extraction went wrong". It means the
        page never arrived: a bot wall or human-verification interstitial stood where the
        content should be, so no records exist to have got right or wrong, and the target's
        extraction strategy is not implicated. A consumer counting it as a failed extraction
        will over-count failures for a target that is merely walled.
        """
        return str(self._get_raw("validation_status", "needs_review"))


class ScrapeCollection(BaseCollection[EntityT]):
    """Three-tier collection base with an in-memory L3 fallback, depending on nothing outside 3tears.

    See this module's docstring for why the fallback is re-implemented here
    rather than inherited from the application-side base class it mirrors.
    """

    #: Tables already warned about in :meth:`_warn_in_memory_l3`, so a
    #: consumer that rebuilds collections per poll cycle still gets one
    #: warning rather than one per cycle. Each subclass gets its own set via
    #: the ``cls._in_memory_l3_warned_tables = ...`` assignment in that
    #: method; declared here so the attribute is typed and present on the
    #: base. Mirrors ``BaseCollection._missing_nats_warned_tables``.
    _in_memory_l3_warned_tables: ClassVar[frozenset[str]] = frozenset()

    #: Columns this collection's table holds as ``TIMESTAMPTZ``, rehydrated from their
    #: ISO-string form by :meth:`deserialize`. A subclass declares only its OWN columns;
    #: the two framework-stamped ones are added here because ``BaseCollection.save_entity``
    #: writes them to every table regardless of what the entity class exposes.
    #:
    #: Hand-declared but not hand-trusted: ``tests/test_migrations_drift.py`` asserts each
    #: collection's set matches the TIMESTAMPTZ columns its migrations actually create, so
    #: a column added without a declaration fails there rather than silently reopening the
    #: string-into-a-TIMESTAMPTZ-fence bug this exists to prevent.
    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: Any = NATS_CLIENT_FROM_REGISTRY,
    ) -> None:
        """
        :param registry: the process-wide ``CollectionRegistry``; this class
            has no application-wide default to resolve one from, so callers
            must supply it explicitly.
        :ptype registry: CollectionRegistry
        :param config: the process-wide ``CoreConfig``; same no-default
            reasoning as ``registry``.
        :ptype config: CoreConfig
        :param nats_client: L2 NATS client, or the registry sentinel to
            resolve it from the registry.
        :ptype nats_client: Any
        :return: nothing
        :rtype: None
        """
        self._rows: dict[Any, dict[str, Any]] = {}
        super().__init__(registry, config, nats_client)

    @property
    def _durable_store(self) -> DurableStore | None:
        """Narrow ``self.l3_pool`` to the structured ``DurableStore`` surface this class actually uses.

        The cast is safe because every real backend the registry wires
        conforms to ``DurableStore`` by the registry's own ``_as_l3_backend``
        design; ``l3_pool`` is typed loosely only because the registry has no
        way to prove that statically.

        Returning ``None`` is what selects the in-memory fallback in all four
        store primitives, so this is also the one place that can observe the
        fallback being taken at all -- hence the warning below.
        """
        if self.l3_pool is None:
            self._warn_in_memory_l3()
            return None
        return cast(DurableStore, self.l3_pool)

    def _warn_in_memory_l3(self) -> None:
        """Warn once per table that this collection's L3 is a process-local dict, not a database.

        The in-memory fallback is deliberate and every unit test in this
        package legitimately depends on it, so this is a warning and never an
        exception. But silence here has a real cost that has already been
        paid once: the fallback ignores schema entirely, so an entity field
        with no matching DDL column round-trips through ``self._rows``
        perfectly and only fails against a real pool. ``link_selector``
        shipped that way (see ``migrations.v009_target_link_selector``). A
        process that believes it is durably persisting scrape state, but is
        not, should say so out loud rather than look identical to one that
        is.

        Mirrors ``BaseCollection._warn_missing_nats_client_once`` exactly,
        including its class-level-set-keyed-on-``table_name`` mechanism: same
        problem (a process-wide wiring gap worth surfacing once, on a path hot
        enough that per-call logging would be filtered out as noise), so the
        same shape rather than a second one. Keying on the class and table
        rather than on the instance is the part that matters here: nothing in
        this package constructs these collections, so a consumer is free to
        build a fresh one per poll cycle, and a per-instance flag would
        quietly degrade "warn once" into "warn every cycle" -- the exact
        outcome this guard exists to avoid.
        """
        cls = type(self)
        warned: frozenset[str] = getattr(cls, "_in_memory_l3_warned_tables", frozenset())
        if self.table_name in warned:
            return
        # Rebuild-and-replace rather than mutate in place. No await sits between the read
        # and the write below, so nothing can interleave here on a single-threaded asyncio
        # loop and this is not fixing a live race. It is about the shape of the state: this
        # is class-level and shared by every instance and task, and immutable-plus-atomic-
        # rebind cannot degrade if it is ever touched from a thread or a second loop, where
        # an in-place ``add`` on a shared set would be a genuine hazard.
        cls._in_memory_l3_warned_tables = frozenset(warned | {self.table_name})
        log.warning(
            "%s: no L3 pool wired; falling back to a process-local in-memory dict for table %r. "
            "Rows will NOT survive process restart and are NOT shared across pods, and this path "
            "ignores the table schema entirely, so a missing DDL column cannot fail here. "
            "Wire an l3_pool on the CollectionRegistry (and run threetears.scrape.migrations."
            "apply_migrations) for real durability.",
            type(self).__name__,
            self.table_name,
        )

    def _single_pk_column(self) -> str:
        """Return ``primary_key_column`` as a plain string.

        No scrape collection declares a composite (tuple) primary key.

        :raises ValueError: if ``primary_key_column`` is a composite (tuple) key.
        """
        pk_column = self.primary_key_column
        if not isinstance(pk_column, str):
            raise ValueError(
                f"{type(self).__name__}: this operation requires a single-string "
                f"primary_key_column; got composite key {pk_column!r}"
            )
        return pk_column

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """Fetch one row by primary key from L3 (asyncpg pool or in-memory dict)."""
        store = self._durable_store
        if store is not None:
            result = await store.fetch_one(self.table_name, {self._single_pk_column(): entity_id})
        else:
            result = self._rows.get(entity_id)
        return result

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """Upsert a row into L3 (asyncpg pool or in-memory dict); return rows written."""
        store = self._durable_store
        if store is not None:
            result = await store.upsert(
                self.table_name,
                data,
                pk=list(self.primary_key_columns),
                on_conflict="update",
                cas=original_timestamp,
                conn=conn,
            )
        else:
            pk_column = self._single_pk_column()
            pk_value = data.get(pk_column)
            if pk_value is None:
                raise ValueError(f"{type(self).__name__}: save_to_store() row is missing its primary key {pk_column!r}")
            self._rows[pk_value] = dict(data)
            result = 1
        return result

    async def delete_from_store(self, entity_id: Any) -> None:
        """Delete a row by primary key from L3 (asyncpg pool or in-memory dict)."""
        store = self._durable_store
        if store is not None:
            await store.delete(self.table_name, {self._single_pk_column(): entity_id})
            return
        self._rows.pop(entity_id, None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize a row dict to JSON bytes for the L2 (NATS KV) cache tier."""
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize JSON bytes from the L2 cache tier back into a row dict.

        Rehydrates :attr:`datetime_columns` from the ISO strings :meth:`serialize`'s
        ``default=str`` produced. ``BaseCollection.deserialize``'s contract names this as
        the place subclasses restore typed fields, and until this did so, a row that
        happened to be read through L2 differed in TYPE from the identical row read
        through L1 or L3 -- strings where the others hold ``datetime``.

        That asymmetry is only cosmetic while such a row is read. It becomes a real fault
        when one is written BACK: an update fences on the row's own ``date_updated`` as an
        optimistic lock, rendered as ``WHERE date_updated = $n`` against ``TIMESTAMPTZ``,
        and a string bound there fails at the asyncpg border.

        Every read-modify-write path in this package is exposed, though not identically.
        :func:`~threetears.scrape.enrichment.enrich_extraction` rebuilds its row through
        ``create()``, so it binds no fence at all and would instead have failed on
        ``retrieved_at`` going into the upsert's VALUES as a string. The health fingerprint
        merge rebuilds as an existing entity and so fails on the fence itself, where its own
        non-fatal handling would have swallowed it. Both are repaired by fixing the read,
        which is why this belongs here rather than in each writer.

        A value that does not parse is left exactly as found rather than nulled: losing a
        timestamp silently is worse than carrying a malformed one to a border that will
        reject it loudly.
        """
        result: dict[str, Any] = json.loads(data)
        for column in self.datetime_columns:
            raw = result.get(column)
            if isinstance(raw, str) and raw:
                try:
                    result[column] = datetime.fromisoformat(raw)
                except ValueError:  # NOSILENT: unparseable timestamp is preserved verbatim
                    log.warning(
                        "%s: %r in column %r did not parse as a timestamp; left as-is",
                        type(self).__name__,
                        raw,
                        column,
                    )
        return result

    async def list_all(self) -> list[EntityT]:
        """Return every entity in the store (L3 scan or in-memory dict values)."""
        entity_cls = self.entity_class
        store = self._durable_store
        # ``list(...)`` snapshots the fallback dict rather than iterating a live view. The
        # comprehension below contains no await, so a concurrent save cannot interleave on
        # a single-threaded asyncio loop and this is not fixing a live race either. It
        # removes the failure mode entirely ("dictionary changed size during iteration")
        # for one word, so the invariant does not depend on there never being an await
        # added inside that comprehension later. The L3 branch needs no equivalent, since
        # ``scan`` has already materialized its rows.
        rows: Any = await store.scan(self.table_name) if store is not None else list(self._rows.values())
        return [entity_cls(row, is_new=False, collection=self) for row in rows]


class ScrapeTargetCollection(ScrapeCollection[ScrapeTarget]):
    """Collection of onboarded scrape targets, keyed by ``target_id``."""

    primary_key_column = "target_id"

    @property
    def table_name(self) -> str:
        """Return the L3 table name for this collection."""
        return "scrape_targets"

    @property
    def entity_class(self) -> type[ScrapeTarget]:
        """Return the entity type this collection manages."""
        return ScrapeTarget


class ScrapeRecipeCollection(ScrapeCollection[ScrapeRecipe]):
    """Collection of extraction recipes, one per target, keyed by ``target_id``."""

    primary_key_column = "target_id"
    datetime_columns: ClassVar[frozenset[str]] = ScrapeCollection.datetime_columns | {
        "won_at",
        "last_validated_at",
    }

    @property
    def table_name(self) -> str:
        """Return the L3 table name for this collection."""
        return "scrape_recipes"

    @property
    def entity_class(self) -> type[ScrapeRecipe]:
        """Return the entity type this collection manages."""
        return ScrapeRecipe


class ScrapeExtractionCollection(ScrapeCollection[ScrapeExtraction]):
    """Collection of per-fetch extraction rows, keyed by uuid7 ``id``."""

    datetime_columns: ClassVar[frozenset[str]] = ScrapeCollection.datetime_columns | {"retrieved_at"}

    @property
    def table_name(self) -> str:
        """Return the L3 table name for this collection."""
        return "scrape_extractions"

    @property
    def entity_class(self) -> type[ScrapeExtraction]:
        """Return the entity type this collection manages."""
        return ScrapeExtraction
