"""model capabilities registry for tracking AI model features and costs.

Defines :class:`ModelCapabilities` (the per-model metadata schema) plus a
module-level registry keyed by ``model_id`` (the public identifier used by
the provider). Each provider module registers its supported models at
import time via :func:`register_capabilities`; consumers query without
instantiating a provider via :func:`get_capabilities`.

Capability values registered by providers are the canonical, source-of-truth
shape every consumer sees by default. Deployment-scoped overrides
(:class:`CapabilityOverride`, :func:`register_capability_override`) layer on
top: when a consumer registers an override for a ``model_id``, every
subsequent :func:`get_capabilities` call for that id returns the registered
metadata with the override fields applied. Overrides exist for the cases
the canonical registry deliberately does not absorb -- testing budget
forcing, deployment-side cost throttling, A/B-testing capability shapes --
so consumers do not each invent parallel local mechanisms (e.g. metallm's
``model_overrides`` table for behavioural quirks). Overrides are
process-local and in-memory; consumers that need cross-pod coherence wrap
this layer with their own persistence + epoch broadcast (the
:mod:`threetears.epoch` pattern is the recommended way to keep multiple
pods aligned).
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from threetears.models.enums import ModelStatus, ModelTier, ModelType

__all__ = [
    "CapabilityOverride",
    "ModelCapabilities",
    "clear_capability_overrides",
    "get_capabilities",
    "get_capability_override",
    "list_capabilities",
    "register_capabilities",
    "register_capability_override",
    "unregister_capability_override",
]


class ModelCapabilities(BaseModel):
    """capabilities and metadata for registered AI model.

    tracks model features, constraints, and per-token costs for routing
    and billing decisions across chat, embedding, transcription, and
    image generation model types.

    :param model_name: unique identifier for model (e.g. claude-sonnet-4-6)
    :ptype model_name: str
    :param model_type: primary capability classification
    :ptype model_type: ModelType
    :param model_tier: relative size and cost tier
    :ptype model_tier: ModelTier
    :param model_status: lifecycle status
    :ptype model_status: ModelStatus
    :param provider_name: provider that serves this model (e.g. "anthropic")
    :ptype provider_name: str | None
    :param context_window: maximum input context length in tokens
    :ptype context_window: int | None
    :param max_output_tokens: maximum output length in tokens
    :ptype max_output_tokens: int | None
    :param supports_streaming: whether model supports streaming responses
    :ptype supports_streaming: bool | None
    :param supports_tools: whether model supports tool/function calling
    :ptype supports_tools: bool | None
    :param supports_vision: whether model supports image inputs
    :ptype supports_vision: bool | None
    :param requires_alternating_roles: whether model requires alternating user/assistant roles
    :ptype requires_alternating_roles: bool | None
    :param supports_anthropic_cache_control: whether model accepts
        ``cache_control={"type": "ephemeral"}`` on structured system content
        and honors it on the provider request (Anthropic-shape prompt caching)
    :ptype supports_anthropic_cache_control: bool | None
    :param supports_openai_auto_cache: whether model participates in OpenAI-shape
        automatic prompt caching (no opt-in markers; response surfaces
        ``prompt_tokens_details.cached_tokens``)
    :ptype supports_openai_auto_cache: bool | None
    :param min_cacheable_tokens: shortest prefix length at which the provider
        actually caches; prefixes shorter than this pay full price even when
        ``cache_control`` is attached. ``0`` means the provider has no minimum
        (auto-cache providers); ``None`` means the field has not been set for
        this model
    :ptype min_cacheable_tokens: int | None
    :param cache_ttl_seconds: provider-side ephemeral cache lifetime in seconds.
        ``0`` means the provider does not expose a TTL (auto-cache); ``None``
        means the field has not been set
    :ptype cache_ttl_seconds: int | None
    :param embedding_dimensions: dimensionality of embedding vectors
    :ptype embedding_dimensions: int | None
    :param max_embedding_tokens: maximum input length for embedding
    :ptype max_embedding_tokens: int | None
    :param supports_batch_embedding: whether model supports batch embedding
    :ptype supports_batch_embedding: bool | None
    :param supported_audio_formats: list of supported audio MIME types
    :ptype supported_audio_formats: list[str] | None
    :param max_audio_duration_seconds: maximum audio duration in seconds
    :ptype max_audio_duration_seconds: float | None
    :param supports_language_hint: whether transcription accepts language hint
    :ptype supports_language_hint: bool | None
    :param supports_img2img: whether model supports image-to-image generation
    :ptype supports_img2img: bool | None
    :param supported_sizes: list of supported output image sizes
    :ptype supported_sizes: list[str] | None
    :param supports_style_parameter: whether model accepts style parameter
    :ptype supports_style_parameter: bool | None
    :param supported_voices: available voice options for speech synthesis
    :ptype supported_voices: list[str] | None
    :param supported_output_formats: supported audio output formats (e.g. ["mp3", "wav", "opus"])
    :ptype supported_output_formats: list[str] | None
    :param max_speech_characters: maximum input text length for speech synthesis
    :ptype max_speech_characters: int | None
    :param supports_ssml: whether SSML markup is accepted
    :ptype supports_ssml: bool | None
    :param max_rerank_documents: maximum documents per reranking request
    :ptype max_rerank_documents: int | None
    :param max_rerank_tokens: maximum tokens per document for reranking
    :ptype max_rerank_tokens: int | None
    :param cost_per_input_token: cost per input token in USD
    :ptype cost_per_input_token: Decimal | None
    :param cost_per_output_token: cost per output token in USD
    :ptype cost_per_output_token: Decimal | None
    :param cost_per_request: flat cost per request in USD
    :ptype cost_per_request: Decimal | None
    """

    model_name: str
    model_type: ModelType
    model_tier: ModelTier
    model_status: ModelStatus = ModelStatus.ACTIVE
    provider_name: str | None = None

    # chat fields
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_streaming: bool | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    requires_alternating_roles: bool | None = None

    # chat caching fields. tri-state ``bool | None`` for the support flags
    # so unset (None) is distinguishable from "explicitly does not support".
    # the two int fields use the same convention: ``0`` means
    # "no minimum / no TTL" (auto-cache provider shape); ``None`` means
    # "not declared by the provider entry".
    supports_anthropic_cache_control: bool | None = None
    supports_openai_auto_cache: bool | None = None
    min_cacheable_tokens: int | None = None
    cache_ttl_seconds: int | None = None

    # embedding fields
    embedding_dimensions: int | None = None
    max_embedding_tokens: int | None = None
    supports_batch_embedding: bool | None = None

    # transcription fields
    supported_audio_formats: list[str] | None = None
    max_audio_duration_seconds: float | None = None
    supports_language_hint: bool | None = None

    # image generation fields
    supports_img2img: bool | None = None
    supported_sizes: list[str] | None = None
    supports_style_parameter: bool | None = None

    # speech fields
    supported_voices: list[str] | None = None
    supported_output_formats: list[str] | None = None
    max_speech_characters: int | None = None
    supports_ssml: bool | None = None

    # reranking fields
    max_rerank_documents: int | None = None
    max_rerank_tokens: int | None = None

    # cost fields (always Decimal, never float)
    cost_per_input_token: Decimal | None = None
    cost_per_output_token: Decimal | None = None
    # per-token prompt-cache costs (Anthropic-shape caching): the discounted
    # rate for a cache READ (reusing a cached prefix) and the surcharged rate
    # for a cache WRITE (establishing a cached prefix). ``None`` when the model
    # does not expose cache pricing. carried here so the catalog is the single
    # source of truth for every model cost a consumer (gateway billing, hub
    # bootstrap seeding) needs -- no parallel per-model cost table.
    cost_per_cache_read_token: Decimal | None = None
    cost_per_cache_write_token: Decimal | None = None
    cost_per_request: Decimal | None = None


class CapabilityOverride(BaseModel):
    """deployment-scoped override that layers on top of registered ModelCapabilities.

    Holds the subset of capability fields a consumer has chosen to override
    for a specific ``model_id``. Every field is optional: when a field is
    unset (the default), :func:`get_capabilities` returns the underlying
    registry value for that field; when set, the override value is
    returned instead. Identity fields (``model_name``, ``model_type``,
    ``model_tier``, ``provider_name``) are deliberately not overridable --
    those describe what the model IS, not how a deployment chooses to
    use it.

    Fields are limited to those that have legitimate
    deployment-scoped variation: context budgets (testing, throttling,
    cost control), output limits, the streaming / tools / vision flags
    (rare but legitimate when a deployment disables a capability the
    provider supports), and the three cost fields (deployments may
    negotiate custom pricing). Embedding / transcription / image-gen /
    speech / reranking capability shapes are left to the canonical
    registry; if a real override case for those surfaces, extend this
    schema rather than letting consumers shadow the registry locally.

    :param context_window: override max input context length in tokens
    :ptype context_window: int | None
    :param max_output_tokens: override max output length in tokens
    :ptype max_output_tokens: int | None
    :param supports_streaming: override streaming-support flag
    :ptype supports_streaming: bool | None
    :param supports_tools: override tools/function-calling flag
    :ptype supports_tools: bool | None
    :param supports_vision: override image-input flag
    :ptype supports_vision: bool | None
    :param cost_per_input_token: override per-input-token USD cost
    :ptype cost_per_input_token: Decimal | None
    :param cost_per_output_token: override per-output-token USD cost
    :ptype cost_per_output_token: Decimal | None
    :param cost_per_request: override flat per-request USD cost
    :ptype cost_per_request: Decimal | None
    """

    # ``extra="forbid"`` so callers cannot stuff identity fields
    # (``model_name``, ``model_type``, ``model_tier``, ``provider_name``)
    # or unknown fields into an override; those describe what the model
    # IS, not how a deployment chooses to use it.
    model_config = ConfigDict(extra="forbid")

    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_streaming: bool | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    cost_per_input_token: Decimal | None = None
    cost_per_output_token: Decimal | None = None
    cost_per_request: Decimal | None = None


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: dict[str, ModelCapabilities] = {}
_OVERRIDES: dict[str, CapabilityOverride] = {}


def register_capabilities(model_id: str, capabilities: ModelCapabilities) -> None:
    """registers capability metadata for ``model_id`` in the module-level registry.

    overwrites any existing entry for the same id. providers call this at
    import time so consumers can query without instantiating a provider.

    :param model_id: public identifier for the model (matches what consumers pass to the factory)
    :ptype model_id: str
    :param capabilities: capability metadata
    :ptype capabilities: ModelCapabilities
    """
    with _REGISTRY_LOCK:
        _REGISTRY[model_id] = capabilities


def register_capability_override(model_id: str, override: CapabilityOverride) -> None:
    """register a deployment-scoped override that layers on top of ``model_id``'s capabilities.

    every field set on ``override`` (i.e. fields the caller passed
    explicitly to the :class:`CapabilityOverride` constructor) wins
    over the canonical registry value at :func:`get_capabilities`
    time; unset fields fall through to the registry. registering a new
    override for an existing id replaces the previous one wholesale --
    callers that want to mutate one field on top of an existing
    override should read it via :func:`get_capability_override`,
    construct the new override, and re-register.

    overrides are process-local and in-memory. consumers that need
    multi-pod coherence (e.g. an admin UI patching a deployment-wide
    cost cap) should persist the override in their own store and use
    :mod:`threetears.epoch` to broadcast invalidation; sibling pods
    re-fetch and call this function to update their local registry.

    :param model_id: public identifier for the model the override applies to
    :ptype model_id: str
    :param override: the override values; only fields explicitly set
        are applied. fields left at their default (``None`` via the
        :class:`CapabilityOverride` constructor without specifying
        them) fall through to the canonical registry value
    :ptype override: CapabilityOverride
    """
    with _REGISTRY_LOCK:
        _OVERRIDES[model_id] = override


def unregister_capability_override(model_id: str) -> None:
    """drop the override for ``model_id``; subsequent :func:`get_capabilities`
    calls return the canonical registry value unchanged.

    no-op when no override is registered for the id, so callers can
    safely use this in unconditional teardown paths.

    :param model_id: public identifier for the model whose override should be cleared
    :ptype model_id: str
    """
    with _REGISTRY_LOCK:
        _OVERRIDES.pop(model_id, None)


def clear_capability_overrides() -> None:
    """drop every registered capability override.

    intended for test-suite cleanup; production code should use
    :func:`unregister_capability_override` for the specific id it
    owns. process-wide and irreversible -- callers that share the
    process with other consumers should NOT use this in production.
    """
    with _REGISTRY_LOCK:
        _OVERRIDES.clear()


def get_capability_override(model_id: str) -> CapabilityOverride | None:
    """return the registered override for ``model_id`` (read-only) or ``None``.

    inspects the override layer without consulting the canonical
    registry. returns ``None`` when no override is registered, which
    is distinct from "an override is registered with all fields
    unset" (that returns a :class:`CapabilityOverride` instance with
    every field ``None``, semantically equivalent to no override but
    structurally present).

    :param model_id: public identifier for the model
    :ptype model_id: str
    :return: override or ``None`` when not registered
    :rtype: CapabilityOverride | None
    """
    with _REGISTRY_LOCK:
        result = _OVERRIDES.get(model_id)
    return result


def get_capabilities(model_id: str) -> ModelCapabilities | None:
    """returns capability metadata for ``model_id`` with any registered
    override applied.

    when a :class:`CapabilityOverride` has been registered for the
    same ``model_id`` via :func:`register_capability_override`, fields
    the caller explicitly set on the override take precedence over
    the canonical registry value. fields the override left unset fall
    through to the registry. when no override is registered, the
    canonical registry value is returned unchanged.

    :param model_id: public identifier for the model
    :ptype model_id: str
    :return: capability metadata with override applied, or ``None``
        if no provider registered the id
    :rtype: ModelCapabilities | None
    """
    with _REGISTRY_LOCK:
        base = _REGISTRY.get(model_id)
        override = _OVERRIDES.get(model_id)
    if base is None:
        return None
    if override is None:
        return base
    # exclude_unset -> only fields the override explicitly set are
    # applied; fields left at their CapabilityOverride default fall
    # through to the registry value via model_copy(update=...).
    update: dict[str, Any] = override.model_dump(exclude_unset=True)
    if not update:
        return base
    return base.model_copy(update=update)


def list_capabilities() -> dict[str, ModelCapabilities]:
    """returns a snapshot of all registered model capabilities with any
    registered overrides applied.

    iterates the registry under the same lock as
    :func:`get_capabilities` so the resulting snapshot is internally
    consistent (no half-applied override on one id, registry value on
    another, mid-loop). the returned dict is a copy; mutating it does
    not affect the underlying registry.

    :return: mapping of ``model_id`` to capability metadata with
        overrides applied
    :rtype: dict[str, ModelCapabilities]
    """
    with _REGISTRY_LOCK:
        snapshot_base = dict(_REGISTRY)
        snapshot_overrides = dict(_OVERRIDES)
    result: dict[str, ModelCapabilities] = {}
    for model_id, base in snapshot_base.items():
        override = snapshot_overrides.get(model_id)
        if override is None:
            result[model_id] = base
            continue
        update: dict[str, Any] = override.model_dump(exclude_unset=True)
        result[model_id] = base.model_copy(update=update) if update else base
    return result
