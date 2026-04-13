"""tests for ModelCapabilities Pydantic model."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from threetears.models.capabilities import ModelCapabilities
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
            model_name="claude-sonnet-4-20250514",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.LARGE,
        )
        assert caps.model_name == "claude-sonnet-4-20250514"
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
            model_name="claude-sonnet-4-20250514",
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
            model_name="claude-sonnet-4-20250514",
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
            model_name="voyage-3",
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
