"""model capabilities registry for tracking AI model features and costs.

Defines :class:`ModelCapabilities` (the per-model metadata schema) plus a
module-level registry keyed by ``model_id`` (the public identifier used by
the provider). Each provider module registers its supported models at
import time via :func:`register_capabilities`; consumers query without
instantiating a provider via :func:`get_capabilities`.
"""

from __future__ import annotations

import threading
from decimal import Decimal

from pydantic import BaseModel

from threetears.models.enums import ModelStatus, ModelTier, ModelType

__all__ = [
    "ModelCapabilities",
    "get_capabilities",
    "list_capabilities",
    "register_capabilities",
]


class ModelCapabilities(BaseModel):
    """capabilities and metadata for registered AI model.

    tracks model features, constraints, and per-token costs for routing
    and billing decisions across chat, embedding, transcription, and
    image generation model types.

    :param model_name: unique identifier for model (e.g. claude-sonnet-4-20250514)
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
    cost_per_request: Decimal | None = None


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: dict[str, ModelCapabilities] = {}


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


def get_capabilities(model_id: str) -> ModelCapabilities | None:
    """returns capability metadata for ``model_id`` if registered.

    :param model_id: public identifier for the model
    :ptype model_id: str
    :return: capability metadata, or ``None`` if no provider registered the id
    :rtype: ModelCapabilities | None
    """
    with _REGISTRY_LOCK:
        result = _REGISTRY.get(model_id)
    return result


def list_capabilities() -> dict[str, ModelCapabilities]:
    """returns a snapshot of all registered model capabilities.

    :return: mapping of ``model_id`` to capability metadata
    :rtype: dict[str, ModelCapabilities]
    """
    with _REGISTRY_LOCK:
        result = dict(_REGISTRY)
    return result
