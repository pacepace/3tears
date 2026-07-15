"""ScrapeTarget / ScrapeRecipe / ScrapeExtraction — domain-agnostic 3tears-scrape core.

Subclasses ``threetears.core.collections.base.BaseCollection`` directly —
**not** ``faidh.db.collection.FaidhCollection`` — to preserve the
zero-faidh-imports discipline that makes lifting this into 3tears later a
directory move, not a disentangling exercise (mirrors
``src/faidh/intake/rate_limit/strategy.py``'s existing precedent).

verify-api finding (Chunk 1 Done-when step 0, reading
``threetears.core.collections.base.BaseCollection`` source directly):
``BaseCollection`` provides the full three-tier (L1/L2/L3) cache machinery,
subscript access, CAS-mutate, and invalidation-publish for free — a
subclass only has to implement ``table_name``, ``entity_class``, and the
five storage-tier primitives (``fetch_from_store``, ``save_to_store``,
``delete_from_store``, ``serialize``, ``deserialize``). It does **not**
provide an in-memory L3 fallback of its own; that convenience is
``FaidhCollection``'s addition (its ``self._rows`` dict, wired through
``faidh.store.get_registry()``/``get_config()`` defaults), which this
module cannot import. ``ScrapeCollection`` below re-implements the same
shape locally, without the faidh.store default resolution — callers must
pass ``registry``/``config`` explicitly. L3 is a real asyncpg pool
(``threetears.core.backends.protocol.DurableStore``-conforming) once
``threetears.scrape.migrations.apply_migrations()`` has run and the registry
carries an ``l3_pool``; otherwise CRUD falls back to the in-memory
``self._rows`` dict for the process lifetime — this fallback is why unit
tests never need a real database, but it is NOT multi-pod-safe (each pod
gets its own dict), which is the whole reason the L3 branch below exists.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, cast

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
]

log = get_logger(__name__)


def _parse_dt(raw: Any) -> datetime | None:
    """Parse a possibly-ISO-string timestamp back into a ``datetime``.

    L2 (NATS KV) round-trips every value through JSON, which stringifies
    ``datetime`` on write (see :meth:`ScrapeCollection.serialize`) but does
    not parse it back on read — so a value read through L2 arrives as a
    string even though the in-memory L3 fallback keeps it as a native
    ``datetime``. Mirrors ``faidh.intake.signals.collections._observation_ts``.
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
        """Which ``ScrapeDriver`` backend renders this target: ``"nodriver"`` | ``"camoufox"``."""
        return str(self._get_raw("driver_backend", "nodriver"))

    @property
    def rate_limit_key(self) -> str:
        """Opaque string key; the core stores and passes it through but never resolves it.

        Resolution against ``FAIDH_RATE_LIMIT_STRATEGIES`` happens in
        faidh-side scheduling code (Chunk 4), which is exactly why this is
        a plain string rather than a ``RateLimitStrategy`` instance — that
        would force a faidh import here.
        """
        return str(self._get_raw("rate_limit_key", ""))

    @property
    def cadence(self) -> str:
        """How often this target is re-fetched; interpreted by the scheduling layer (Chunk 4)."""
        return str(self._get_raw("cadence", ""))

    @property
    def multi_row(self) -> bool:
        """Whether this target's page holds many records (a table/listing) rather than one.

        Selects which eval loop ``poll_scrape_targets`` (Chunk 4) runs:
        ``run_eval_loop_multi_row`` when ``True``, ``run_eval_loop`` (the
        original single-record path) when ``False`` (the default — preserves
        every pre-existing target's behavior). Domain-agnostic: this is a
        statement about page shape, not about what the records mean, so it
        belongs on the core entity rather than in faidh-side scheduling code.
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
        generation and structural validation (Chunk 02's design).

        Carried on the target itself (not a separate caller-supplied dict,
        as originally decided in Chunk 02) so one config entry -- a YAML
        row, a database row -- fully describes both how to fetch a target
        and what to extract from it; consolidated on direct instruction once
        target/schema config needed to round-trip through YAML and a
        database (a two-dict shape can drift, and did require its own
        ``set(WARN_ACT_SCHEMAS) == set(WARN_ACT_TARGETS)`` test to guard
        against it). Still domain-agnostic: the core never interprets what
        a field NAME means, only its declared type -- see this module's own
        docstring and Chunk 02's "the domain-agnostic core never hardcodes
        what a field means" note, which this does not violate.
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
        """Which extraction-strategy shape the eval loop should propose: ``"css"`` or ``"regex"``.

        Passed straight through to ``run_eval_loop``/``run_eval_loop_multi_row(...,
        strategy_type=...)`` -- ``"css"`` (the default) preserves every
        pre-existing target's behavior (CSS-selector candidates against an
        HTML table). ``"regex"`` is for a page whose real content is prose/
        list text with no ``<table>`` structure at all (Pennsylvania's real
        WARN page, the concrete driver -- Chunk 12 rejected it outright since
        the CSS-selector candidate generator had no strategy shape to even
        attempt a candidate in). A statement about page shape, the same
        category as ``multi_row``/``wait_for`` -- the eval loop's own AI-
        driven extraction still runs unmodified, just proposing regex
        patterns instead of selectors (regex/text-block extraction,
        2026-07-14).
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
        -- see :attr:`api_results_path`.
        """
        result: str | None = self._get_raw("api_fragment_field", None)
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
        """The winning candidate's strategy; shape decided by the eval loop (Chunk 2), not this schema."""
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
        """Crossing a threshold re-triggers candidate generation (Chunk 2)."""
        return int(self._get_raw("consecutive_validation_failures", 0))


class ScrapeExtraction(BaseEntity):
    """One row per fetch — the actual output.

    Chunk 1 writes this row before the eval loop (Chunk 2) or enrichment
    pass (Chunk 3) exist, via a naive hardcoded-selector extraction that
    proves the fetch/persist path — so every eval-loop-and-later field
    defaults to its pre-eval-loop value rather than being assumed present.
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
        """Which recipe produced this row; ``None`` for Chunk 1's pre-eval-loop rows."""
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
        """The winning candidate's output; Chunk 1's naive extraction's output."""
        result: dict[str, Any] = _decode_json_field(self._get_raw("structured_fields"), {})
        return result

    @property
    def field_confidences(self) -> dict[str, Any] | None:
        """Per-field validation notes from the eval loop; ``None`` until Chunk 2."""
        result: dict[str, Any] | None = _decode_json_field(self._get_raw("field_confidences"), None)
        return result

    @property
    def enrichment_notes(self) -> dict[str, Any] | None:
        """The secondary LLM pass's free-form findings; ``None`` until Chunk 3."""
        result: dict[str, Any] | None = _decode_json_field(self._get_raw("enrichment_notes"), None)
        return result

    @property
    def validation_status(self) -> str:
        """``"validated"`` | ``"needs_review"`` | ``"failed"``; defaults to ``"needs_review"``."""
        return str(self._get_raw("validation_status", "needs_review"))


class ScrapeCollection(BaseCollection[EntityT]):
    """Three-tier collection base with an in-memory L3 fallback, built without faidh imports.

    The scrape-local mirror of ``faidh.db.collection.FaidhCollection`` —
    see this module's docstring for why the fallback is re-implemented
    here rather than inherited.
    """

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: Any = NATS_CLIENT_FROM_REGISTRY,
    ) -> None:
        """
        :param registry: the process-wide ``CollectionRegistry``; unlike
            ``FaidhCollection`` this has no faidh-side default to resolve
            from, so callers must supply it explicitly.
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

        Mirrors ``faidh.db.collection.FaidhCollection._durable_store`` exactly
        — see that docstring for why the cast is safe (every real backend the
        registry wires conforms to ``DurableStore``, by the registry's own
        ``_as_l3_backend`` design).
        """
        if self.l3_pool is None:
            return None
        return cast(DurableStore, self.l3_pool)

    def _single_pk_column(self) -> str:
        """Return ``primary_key_column`` as a plain string.

        No scrape collection declares a composite (tuple) primary key.
        Mirrors ``faidh.db.collection.FaidhCollection._single_pk_column``.

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
        """Deserialize JSON bytes from the L2 cache tier back into a row dict."""
        result: dict[str, Any] = json.loads(data)
        return result

    async def list_all(self) -> list[EntityT]:
        """Return every entity in the store (L3 scan or in-memory dict values)."""
        entity_cls = self.entity_class
        store = self._durable_store
        rows: Any = await store.scan(self.table_name) if store is not None else self._rows.values()
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

    @property
    def table_name(self) -> str:
        """Return the L3 table name for this collection."""
        return "scrape_extractions"

    @property
    def entity_class(self) -> type[ScrapeExtraction]:
        """Return the entity type this collection manages."""
        return ScrapeExtraction
