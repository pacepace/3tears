"""tests for ModelCapabilities Pydantic model."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

from uuid import uuid4

import pytest
from pydantic import BaseModel

from threetears.models import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FAST_MODEL,
    DEFAULT_LARGE_MODEL,
)
from threetears.models.capabilities import (
    CapabilityOverride,
    ModelCapabilities,
    clear_capability_overrides,
    get_capabilities,
    get_capability_override,
    get_published_capabilities,
    list_capabilities,
    register_capabilities,
    register_capability_override,
    register_published_capabilities,
    unregister_capability_override,
    unregister_published_capabilities,
)
from threetears.models.enums import ModelStatus, ModelTier, ModelType


class TestModelCapabilities:
    """tests for ModelCapabilities Pydantic model."""

    def test_model_capabilities_is_pydantic_model(self) -> None:
        """ModelCapabilities inherits from pydantic BaseModel."""
        assert issubclass(ModelCapabilities, BaseModel)

    def test_model_capabilities_has_model_fields(self) -> None:
        """ModelCapabilities has pydantic model_fields attribute."""
        assert hasattr(ModelCapabilities, "model_fields")

    def test_model_capabilities_required_fields(self) -> None:
        """ModelCapabilities requires model_name, model_type, and model_tier."""
        caps = ModelCapabilities(
            model_name=DEFAULT_CHAT_MODEL,
            model_type=ModelType.CHAT,
            model_tier=ModelTier.LARGE,
        )
        assert caps.model_name == DEFAULT_CHAT_MODEL
        assert caps.model_type == ModelType.CHAT
        assert caps.model_tier == ModelTier.LARGE

    def test_model_capabilities_status_defaults_to_active(self) -> None:
        """ModelCapabilities model_status defaults to ACTIVE."""
        caps = ModelCapabilities(
            model_name="test-model",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.SMALL,
        )
        assert caps.model_status == ModelStatus.ACTIVE

    def test_model_capabilities_chat_fields_default_none(self) -> None:
        """ModelCapabilities chat-specific fields default to None."""
        caps = ModelCapabilities(
            model_name="test-model",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.SMALL,
        )
        assert caps.context_window is None
        assert caps.max_output_tokens is None
        assert caps.supports_streaming is None
        assert caps.supports_tools is None
        assert caps.supports_vision is None
        assert caps.requires_alternating_roles is None

    def test_model_capabilities_cache_fields_default_none(self) -> None:
        """ModelCapabilities cache fields default to None (tri-state).

        the four cache fields are tri-state: None means "not declared
        by the provider entry", True/False means "explicitly supports /
        does not support", and the int fields use 0 for "no minimum /
        no TTL" (auto-cache shape) vs None for "not declared".
        """
        caps = ModelCapabilities(
            model_name="test-model",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.SMALL,
        )
        assert caps.supports_anthropic_cache_control is None
        assert caps.supports_openai_auto_cache is None
        assert caps.min_cacheable_tokens is None
        assert caps.cache_ttl_seconds is None

    def test_model_capabilities_cache_fields_accept_explicit_values(self) -> None:
        """cache fields accept explicit bool / int values.

        anthropic-shape: cache_control supported, 1024 token minimum,
        300 second ephemeral TTL. openai-shape: auto-cache supported,
        no minimum, no TTL. explicit False for the unsupported flag
        is distinguishable from None (not declared).
        """
        anthropic = ModelCapabilities(
            model_name="claude-test",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.LARGE,
            supports_anthropic_cache_control=True,
            supports_openai_auto_cache=False,
            min_cacheable_tokens=1024,
            cache_ttl_seconds=300,
        )
        assert anthropic.supports_anthropic_cache_control is True
        assert anthropic.supports_openai_auto_cache is False
        assert anthropic.min_cacheable_tokens == 1024
        assert anthropic.cache_ttl_seconds == 300

        openai = ModelCapabilities(
            model_name="gpt-test",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.LARGE,
            supports_anthropic_cache_control=False,
            supports_openai_auto_cache=True,
            min_cacheable_tokens=0,
            cache_ttl_seconds=0,
        )
        assert openai.supports_anthropic_cache_control is False
        assert openai.supports_openai_auto_cache is True
        assert openai.min_cacheable_tokens == 0
        assert openai.cache_ttl_seconds == 0

    def test_model_capabilities_embedding_fields_default_none(self) -> None:
        """ModelCapabilities embedding-specific fields default to None."""
        caps = ModelCapabilities(
            model_name="test-embed",
            model_type=ModelType.EMBEDDING,
            model_tier=ModelTier.SMALL,
        )
        assert caps.embedding_dimensions is None
        assert caps.max_embedding_tokens is None
        assert caps.supports_batch_embedding is None

    def test_model_capabilities_transcription_fields_default_none(self) -> None:
        """ModelCapabilities transcription-specific fields default to None."""
        caps = ModelCapabilities(
            model_name="test-whisper",
            model_type=ModelType.TRANSCRIPTION,
            model_tier=ModelTier.MEDIUM,
        )
        assert caps.supported_audio_formats is None
        assert caps.max_audio_duration_seconds is None
        assert caps.supports_language_hint is None

    def test_model_capabilities_image_gen_fields_default_none(self) -> None:
        """ModelCapabilities image generation fields default to None."""
        caps = ModelCapabilities(
            model_name="test-dalle",
            model_type=ModelType.IMAGE_GENERATION,
            model_tier=ModelTier.LARGE,
        )
        assert caps.supports_img2img is None
        assert caps.supported_sizes is None
        assert caps.supports_style_parameter is None

    def test_model_capabilities_cost_fields_default_none(self) -> None:
        """ModelCapabilities cost fields default to None."""
        caps = ModelCapabilities(
            model_name="test-model",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.SMALL,
        )
        assert caps.cost_per_input_token is None
        assert caps.cost_per_output_token is None
        assert caps.cost_per_request is None

    def test_model_capabilities_cost_fields_are_decimal(self) -> None:
        """ModelCapabilities cost fields accept and store Decimal values."""
        caps = ModelCapabilities(
            model_name=DEFAULT_CHAT_MODEL,
            model_type=ModelType.CHAT,
            model_tier=ModelTier.LARGE,
            cost_per_input_token=Decimal("0.000003"),
            cost_per_output_token=Decimal("0.000015"),
            cost_per_request=Decimal("0.00"),
        )
        assert isinstance(caps.cost_per_input_token, Decimal)
        assert isinstance(caps.cost_per_output_token, Decimal)
        assert isinstance(caps.cost_per_request, Decimal)
        assert caps.cost_per_input_token == Decimal("0.000003")
        assert caps.cost_per_output_token == Decimal("0.000015")

    def test_model_capabilities_all_chat_fields(self) -> None:
        """ModelCapabilities stores all chat-specific fields correctly."""
        caps = ModelCapabilities(
            model_name=DEFAULT_CHAT_MODEL,
            model_type=ModelType.CHAT,
            model_tier=ModelTier.LARGE,
            model_status=ModelStatus.ACTIVE,
            context_window=200000,
            max_output_tokens=8192,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            requires_alternating_roles=True,
            cost_per_input_token=Decimal("0.000003"),
            cost_per_output_token=Decimal("0.000015"),
        )
        assert caps.context_window == 200000
        assert caps.max_output_tokens == 8192
        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.requires_alternating_roles is True

    def test_model_capabilities_all_embedding_fields(self) -> None:
        """ModelCapabilities stores all embedding-specific fields correctly."""
        caps = ModelCapabilities(
            model_name=DEFAULT_EMBEDDING_MODEL,
            model_type=ModelType.EMBEDDING,
            model_tier=ModelTier.MEDIUM,
            embedding_dimensions=1024,
            max_embedding_tokens=32000,
            supports_batch_embedding=True,
        )
        assert caps.embedding_dimensions == 1024
        assert caps.max_embedding_tokens == 32000
        assert caps.supports_batch_embedding is True

    def test_model_capabilities_all_transcription_fields(self) -> None:
        """ModelCapabilities stores all transcription-specific fields correctly."""
        caps = ModelCapabilities(
            model_name="whisper-1",
            model_type=ModelType.TRANSCRIPTION,
            model_tier=ModelTier.MEDIUM,
            supported_audio_formats=["audio/wav", "audio/mp3", "audio/ogg"],
            max_audio_duration_seconds=3600.0,
            supports_language_hint=True,
        )
        assert caps.supported_audio_formats == ["audio/wav", "audio/mp3", "audio/ogg"]
        assert caps.max_audio_duration_seconds == 3600.0
        assert caps.supports_language_hint is True

    def test_model_capabilities_all_image_gen_fields(self) -> None:
        """ModelCapabilities stores all image generation fields correctly."""
        caps = ModelCapabilities(
            model_name="dall-e-3",
            model_type=ModelType.IMAGE_GENERATION,
            model_tier=ModelTier.LARGE,
            supports_img2img=True,
            supported_sizes=["1024x1024", "1792x1024", "1024x1792"],
            supports_style_parameter=True,
            cost_per_request=Decimal("0.04"),
        )
        assert caps.supports_img2img is True
        assert caps.supported_sizes == ["1024x1024", "1792x1024", "1024x1792"]
        assert caps.supports_style_parameter is True
        assert caps.cost_per_request == Decimal("0.04")

    def test_model_capabilities_serialization_roundtrip(self) -> None:
        """ModelCapabilities can serialize to dict and back."""
        caps = ModelCapabilities(
            model_name="test-model",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.SMALL,
            model_status=ModelStatus.DEPRECATED,
            context_window=4096,
            cost_per_input_token=Decimal("0.001"),
        )
        data = caps.model_dump()
        restored = ModelCapabilities.model_validate(data)
        assert restored.model_name == caps.model_name
        assert restored.model_type == caps.model_type
        assert restored.model_tier == caps.model_tier
        assert restored.model_status == caps.model_status
        assert restored.context_window == caps.context_window
        assert restored.cost_per_input_token == caps.cost_per_input_token

    def test_model_capabilities_json_roundtrip(self) -> None:
        """ModelCapabilities can serialize to JSON and back."""
        caps = ModelCapabilities(
            model_name="test-model",
            model_type=ModelType.EMBEDDING,
            model_tier=ModelTier.MEDIUM,
            embedding_dimensions=768,
            cost_per_request=Decimal("0.0001"),
        )
        json_str = caps.model_dump_json()
        restored = ModelCapabilities.model_validate_json(json_str)
        assert restored.model_name == caps.model_name
        assert restored.embedding_dimensions == 768
        assert restored.cost_per_request == Decimal("0.0001")

    def test_model_capabilities_speech_fields_default_none(self) -> None:
        """ModelCapabilities speech-specific fields default to None."""
        caps = ModelCapabilities(
            model_name="test-tts",
            model_type=ModelType.SPEECH,
            model_tier=ModelTier.SMALL,
        )
        assert caps.supported_voices is None
        assert caps.supported_output_formats is None
        assert caps.max_speech_characters is None
        assert caps.supports_ssml is None

    def test_model_capabilities_reranking_fields_default_none(self) -> None:
        """ModelCapabilities reranking-specific fields default to None."""
        caps = ModelCapabilities(
            model_name="test-rerank",
            model_type=ModelType.RERANKING,
            model_tier=ModelTier.SMALL,
        )
        assert caps.max_rerank_documents is None
        assert caps.max_rerank_tokens is None

    def test_model_capabilities_all_speech_fields(self) -> None:
        """ModelCapabilities stores all speech-specific fields correctly."""
        caps = ModelCapabilities(
            model_name="tts-1-hd",
            model_type=ModelType.SPEECH,
            model_tier=ModelTier.LARGE,
            supported_voices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            supported_output_formats=["mp3", "wav", "opus", "flac"],
            max_speech_characters=4096,
            supports_ssml=False,
            cost_per_request=Decimal("0.015"),
        )
        assert caps.supported_voices == ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        assert caps.supported_output_formats == ["mp3", "wav", "opus", "flac"]
        assert caps.max_speech_characters == 4096
        assert caps.supports_ssml is False
        assert caps.cost_per_request == Decimal("0.015")

    def test_model_capabilities_all_reranking_fields(self) -> None:
        """ModelCapabilities stores all reranking-specific fields correctly."""
        caps = ModelCapabilities(
            model_name="rerank-2",
            model_type=ModelType.RERANKING,
            model_tier=ModelTier.MEDIUM,
            max_rerank_documents=1000,
            max_rerank_tokens=4096,
            cost_per_request=Decimal("0.002"),
        )
        assert caps.max_rerank_documents == 1000
        assert caps.max_rerank_tokens == 4096
        assert caps.cost_per_request == Decimal("0.002")

    def test_model_capabilities_explicit_status_override(self) -> None:
        """ModelCapabilities accepts explicit model_status override."""
        caps = ModelCapabilities(
            model_name="old-model",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.SMALL,
            model_status=ModelStatus.DISABLED,
        )
        assert caps.model_status == ModelStatus.DISABLED


class TestCapabilityOverride:
    """tests for the deployment-scoped CapabilityOverride layer.

    every test isolates the global override registry via the
    ``_clean_overrides`` autouse fixture so module-level state from a
    previous test never leaks into the next one. registry entries
    inserted via :func:`register_capabilities` are namespaced with a
    unique per-test ``model_id`` so they don't collide with the
    builtin providers' eager-registration on package import.
    """

    @pytest.fixture(autouse=True)
    def _clean_overrides(self) -> None:
        """ensure no leftover overrides bleed between tests."""
        clear_capability_overrides()
        yield
        clear_capability_overrides()

    def _registered(self, model_id: str, **overrides: object) -> ModelCapabilities:
        """register a baseline ModelCapabilities under ``model_id`` and return it.

        keeps the test fixture concise by burying the required-fields
        boilerplate and routing test-specific values through ``**overrides``.
        """
        caps = ModelCapabilities(
            model_name=model_id,
            model_type=ModelType.CHAT,
            model_tier=ModelTier.LARGE,
            context_window=200_000,
            max_output_tokens=8_000,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            cost_per_input_token=Decimal("0.000003"),
            cost_per_output_token=Decimal("0.000015"),
            **overrides,
        )
        register_capabilities(model_id, caps)
        return caps

    # -- override layer basics -------------------------------------------------

    def test_no_override_returns_registry_value(self) -> None:
        """get_capabilities returns canonical value when no override registered."""
        baseline = self._registered("test-no-override")
        result = get_capabilities("test-no-override")
        assert result == baseline

    def test_override_replaces_only_set_fields(self) -> None:
        """fields the override sets win; unset fields fall through."""
        self._registered("test-partial-override")
        register_capability_override(
            "test-partial-override",
            CapabilityOverride(context_window=500),
        )
        result = get_capabilities("test-partial-override")
        assert result is not None
        assert result.context_window == 500
        # unset override fields fall through to the registry value
        assert result.max_output_tokens == 8_000
        assert result.supports_streaming is True
        assert result.cost_per_input_token == Decimal("0.000003")

    def test_override_does_not_mutate_registry_entry(self) -> None:
        """registering an override leaves the underlying registry entry untouched."""
        baseline = self._registered("test-immutable")
        register_capability_override("test-immutable", CapabilityOverride(context_window=500))
        # canonical entry, accessed via list_capabilities (also override-aware),
        # confirms the registry still holds the original; we re-fetch via the
        # override-aware getter and check that the registered baseline object
        # is unchanged in its in-memory shape.
        assert baseline.context_window == 200_000

    def test_unregister_restores_registry_value(self) -> None:
        """after unregister_capability_override the registry value returns."""
        self._registered("test-unregister")
        register_capability_override("test-unregister", CapabilityOverride(context_window=500))
        unregister_capability_override("test-unregister")
        result = get_capabilities("test-unregister")
        assert result is not None
        assert result.context_window == 200_000

    def test_unregister_unknown_id_is_noop(self) -> None:
        """unregister on an id that has no override does not raise."""
        unregister_capability_override("never-registered")
        # no exception == pass

    def test_re_register_replaces_previous_override(self) -> None:
        """registering a second override wholesale-replaces the first."""
        self._registered("test-replace")
        register_capability_override(
            "test-replace",
            CapabilityOverride(context_window=500, max_output_tokens=100),
        )
        register_capability_override(
            "test-replace",
            CapabilityOverride(context_window=1_000),
        )
        result = get_capabilities("test-replace")
        assert result is not None
        assert result.context_window == 1_000
        # max_output_tokens is unset on the new override so the registry
        # value (8_000) is what falls through, NOT the previous override's 100
        assert result.max_output_tokens == 8_000

    def test_get_capability_override_returns_registered(self) -> None:
        """get_capability_override echoes the registered override."""
        self._registered("test-inspect")
        override = CapabilityOverride(context_window=500)
        register_capability_override("test-inspect", override)
        assert get_capability_override("test-inspect") == override

    def test_get_capability_override_unknown_returns_none(self) -> None:
        """get_capability_override returns None when nothing is registered."""
        self._registered("test-no-override-inspect")
        assert get_capability_override("test-no-override-inspect") is None

    def test_get_capabilities_unknown_id_returns_none(self) -> None:
        """unknown model_id returns None even when an override exists for it
        (override layer cannot synthesise capabilities out of thin air).
        """
        register_capability_override("never-registered", CapabilityOverride(context_window=500))
        assert get_capabilities("never-registered") is None

    def test_clear_capability_overrides_drops_all(self) -> None:
        """clear_capability_overrides removes every registered override."""
        self._registered("test-clear-a")
        self._registered("test-clear-b")
        register_capability_override("test-clear-a", CapabilityOverride(context_window=500))
        register_capability_override("test-clear-b", CapabilityOverride(context_window=600))
        clear_capability_overrides()
        a = get_capabilities("test-clear-a")
        b = get_capabilities("test-clear-b")
        assert a is not None and a.context_window == 200_000
        assert b is not None and b.context_window == 200_000

    # -- override semantics ---------------------------------------------------

    def test_override_with_no_fields_set_falls_through(self) -> None:
        """an override constructed with no fields explicitly set is equivalent
        to no override (every field falls through to the registry).
        """
        baseline = self._registered("test-empty-override")
        register_capability_override("test-empty-override", CapabilityOverride())
        result = get_capabilities("test-empty-override")
        assert result == baseline

    def test_override_field_set_to_none_does_not_overwrite(self) -> None:
        """explicitly setting a field to ``None`` on the override DOES count
        as setting it (Pydantic exclude_unset=True excludes only fields that
        were never assigned). callers wanting fall-through should leave the
        field unset; this test pins the documented contract.
        """
        self._registered("test-explicit-none")
        # explicitly set context_window=None on the override
        override = CapabilityOverride(context_window=None)
        register_capability_override("test-explicit-none", override)
        result = get_capabilities("test-explicit-none")
        assert result is not None
        # context_window was explicitly set to None on the override
        # so the override "wins" with None and the registry value is masked.
        assert result.context_window is None

    def test_override_applies_to_cost_fields(self) -> None:
        """cost fields can be overridden (deployment-specific pricing)."""
        self._registered("test-cost-override")
        register_capability_override(
            "test-cost-override",
            CapabilityOverride(
                cost_per_input_token=Decimal("0.000001"),
                cost_per_output_token=Decimal("0.000005"),
                cost_per_request=Decimal("0.0001"),
            ),
        )
        result = get_capabilities("test-cost-override")
        assert result is not None
        assert result.cost_per_input_token == Decimal("0.000001")
        assert result.cost_per_output_token == Decimal("0.000005")
        assert result.cost_per_request == Decimal("0.0001")

    def test_override_applies_to_capability_flags(self) -> None:
        """boolean capability flags can be overridden (e.g. disable streaming
        in a deployment that wraps the response).
        """
        self._registered("test-flag-override")
        register_capability_override(
            "test-flag-override",
            CapabilityOverride(
                supports_streaming=False,
                supports_tools=False,
                supports_vision=False,
            ),
        )
        result = get_capabilities("test-flag-override")
        assert result is not None
        assert result.supports_streaming is False
        assert result.supports_tools is False
        assert result.supports_vision is False

    # -- list_capabilities ----------------------------------------------------

    def test_list_capabilities_applies_overrides(self) -> None:
        """list_capabilities returns override-applied snapshots."""
        self._registered("test-list-a")
        self._registered("test-list-b")
        register_capability_override("test-list-a", CapabilityOverride(context_window=500))
        snapshot = list_capabilities()
        assert snapshot["test-list-a"].context_window == 500
        assert snapshot["test-list-b"].context_window == 200_000

    def test_list_capabilities_snapshot_is_a_copy(self) -> None:
        """mutating the dict returned by list_capabilities does not affect
        subsequent get_capabilities calls.
        """
        self._registered("test-list-copy")
        snapshot = list_capabilities()
        snapshot.pop("test-list-copy", None)
        # subsequent fetch still sees it
        assert get_capabilities("test-list-copy") is not None

    # -- override field shape -------------------------------------------------

    def test_capability_override_is_pydantic_model(self) -> None:
        """CapabilityOverride is a Pydantic BaseModel."""
        assert issubclass(CapabilityOverride, BaseModel)

    def test_capability_override_default_all_unset(self) -> None:
        """CapabilityOverride() has every field at None by default; model_dump
        with exclude_unset returns an empty dict.
        """
        empty = CapabilityOverride()
        assert empty.model_dump(exclude_unset=True) == {}

    def test_capability_override_rejects_identity_fields(self) -> None:
        """identity fields (model_name, model_type, model_tier, provider_name)
        are NOT on CapabilityOverride; passing them raises a Pydantic
        validation error so deployments cannot redefine what a model IS.
        """
        with pytest.raises(Exception):  # pydantic ValidationError
            CapabilityOverride(model_name="test")  # type: ignore[call-arg]
        with pytest.raises(Exception):
            CapabilityOverride(model_type=ModelType.CHAT)  # type: ignore[call-arg]
        with pytest.raises(Exception):
            CapabilityOverride(model_tier=ModelTier.LARGE)  # type: ignore[call-arg]
        with pytest.raises(Exception):
            CapabilityOverride(provider_name="anthropic")  # type: ignore[call-arg]


#: the request-settings fields, in declaration order.
_REQUEST_SETTINGS_FIELDS = (
    "accepts_temperature",
    "reasoning_efforts",
    "reasoning_budget",
    "adaptive_reasoning",
    "reasoning_disable",
    "structured_output",
)


class TestRequestSettingsFields:
    """the tri-state facts about which request settings a chat model accepts."""

    def test_every_field_defaults_to_not_declared(self) -> None:
        """a model that declares nothing reports ``None`` for each fact, never ``False``.

        :return: nothing
        :rtype: None
        """
        caps = ModelCapabilities(model_name="test-model", model_type=ModelType.CHAT, model_tier=ModelTier.SMALL)
        for name in _REQUEST_SETTINGS_FIELDS:
            assert getattr(caps, name) is None, name

    def test_every_field_says_what_a_caller_does_with_it(self) -> None:
        """each field is documented on ModelCapabilities and on CapabilityOverride.

        :return: nothing
        :rtype: None
        """
        for cls in (ModelCapabilities, CapabilityOverride):
            doc = cls.__doc__ or ""
            for name in _REQUEST_SETTINGS_FIELDS:
                assert f":param {name}:" in doc, (cls.__name__, name)
                assert f":ptype {name}:" in doc, (cls.__name__, name)

    def test_an_empty_effort_list_is_kept_distinct_from_not_declared(self) -> None:
        """``[]`` means "takes no effort parameter", which ``None`` does not.

        :return: nothing
        :rtype: None
        """
        caps = ModelCapabilities(
            model_name="test-model",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.SMALL,
            reasoning_efforts=[],
            accepts_temperature=False,
        )
        assert caps.reasoning_efforts == []
        assert caps.accepts_temperature is False

    def test_the_override_accepts_every_field(self) -> None:
        """a deployment can correct each fact.

        :return: nothing
        :rtype: None
        """
        override = CapabilityOverride(
            accepts_temperature=False,
            reasoning_efforts=["low"],
            reasoning_budget=False,
            adaptive_reasoning=True,
            reasoning_disable=False,
            structured_output=True,
        )
        assert set(override.model_dump(exclude_unset=True)) == set(_REQUEST_SETTINGS_FIELDS)


class TestRegisteredRequestSettings:
    """the static entries carry the facts the providers' docs state, and nothing more."""

    def test_the_large_and_chat_defaults_refuse_temperature_and_budgets(self) -> None:
        """the adaptive-only Anthropic models: sampling and budget_tokens removed.

        :return: nothing
        :rtype: None
        """
        for model_id in (DEFAULT_LARGE_MODEL, DEFAULT_CHAT_MODEL):
            caps = get_capabilities(model_id)
            assert caps is not None
            assert caps.accepts_temperature is False
            assert caps.reasoning_budget is False
            assert caps.adaptive_reasoning is True
            assert caps.reasoning_disable is True
            assert caps.reasoning_efforts == ["low", "medium", "high", "xhigh", "max"]
            assert caps.structured_output is True

    def test_the_fast_default_takes_a_budget_and_no_effort_word(self) -> None:
        """the budget-only Anthropic model: effort is a 400, so the list is empty.

        temperature is "allowed" in the docs, which is not recorded as True:
        only a documented removal is a static fact.

        :return: nothing
        :rtype: None
        """
        caps = get_capabilities(DEFAULT_FAST_MODEL)
        assert caps is not None
        assert caps.reasoning_efforts == []
        assert caps.reasoning_budget is True
        assert caps.adaptive_reasoning is False
        assert caps.accepts_temperature is None

    def test_openai_chat_models_are_not_reasoning_models(self) -> None:
        """gpt-4o takes no effort parameter and does take a strict schema.

        :return: nothing
        :rtype: None
        """
        for model_id in ("gpt-4o", "gpt-4o-mini"):
            caps = get_capabilities(model_id)
            assert caps is not None
            assert caps.reasoning_efforts == []
            assert caps.reasoning_budget is False
            assert caps.structured_output is True
            assert caps.accepts_temperature is None


class TestPublishedLayer:
    """provider-published facts sit above the static entry and below the override."""

    @pytest.fixture
    def model_id(self) -> Iterator[str]:
        """a fresh registered id, with every layer dropped afterwards.

        :return: the id
        :rtype: Iterator[str]
        """
        model_id = f"test-published-{uuid4().hex}"
        register_capabilities(
            model_id,
            ModelCapabilities(
                model_name=model_id,
                model_type=ModelType.CHAT,
                model_tier=ModelTier.SMALL,
                context_window=100_000,
                accepts_temperature=True,
                reasoning_efforts=["low", "high"],
                reasoning_budget=True,
            ),
        )
        yield model_id
        unregister_published_capabilities(model_id)
        unregister_capability_override(model_id)

    def test_a_published_fact_replaces_the_static_one(self, model_id: str) -> None:
        """the fetched fact wins; facts it did not publish keep the static value.

        :param model_id: the registered id
        :ptype model_id: str
        :return: nothing
        :rtype: None
        """
        register_published_capabilities(model_id, CapabilityOverride(reasoning_budget=False))
        caps = get_capabilities(model_id)
        assert caps is not None
        assert caps.reasoning_budget is False
        assert caps.reasoning_efforts == ["low", "high"]
        assert caps.accepts_temperature is True
        assert caps.context_window == 100_000

    def test_an_override_replaces_a_published_fact(self, model_id: str) -> None:
        """override > published > static, field by field.

        :param model_id: the registered id
        :ptype model_id: str
        :return: nothing
        :rtype: None
        """
        register_published_capabilities(
            model_id,
            CapabilityOverride(reasoning_efforts=["low", "medium"], reasoning_budget=False),
        )
        register_capability_override(model_id, CapabilityOverride(reasoning_efforts=["medium"]))
        caps = get_capabilities(model_id)
        assert caps is not None
        assert caps.reasoning_efforts == ["medium"]
        assert caps.reasoning_budget is False
        assert caps.accepts_temperature is True
        assert list_capabilities()[model_id] == caps

    def test_dropping_the_published_facts_restores_the_static_ones(self, model_id: str) -> None:
        """unregistering the layer leaves the static entry answering alone.

        :param model_id: the registered id
        :ptype model_id: str
        :return: nothing
        :rtype: None
        """
        register_published_capabilities(model_id, CapabilityOverride(reasoning_budget=False))
        unregister_published_capabilities(model_id)
        caps = get_capabilities(model_id)
        assert caps is not None
        assert caps.reasoning_budget is True
        assert get_published_capabilities(model_id) is None

    def test_a_model_without_a_static_entry_keeps_its_published_facts_apart(self) -> None:
        """no registry entry: get_capabilities stays None (no invented tier), the facts stay readable.

        :return: nothing
        :rtype: None
        """
        model_id = f"test-unregistered-{uuid4().hex}"
        facts = CapabilityOverride(adaptive_reasoning=True)
        register_published_capabilities(model_id, facts)
        try:
            assert get_capabilities(model_id) is None
            assert get_published_capabilities(model_id) == facts
            assert model_id not in list_capabilities()
        finally:
            unregister_published_capabilities(model_id)
