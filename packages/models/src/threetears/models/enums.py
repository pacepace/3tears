"""enums for AI model classification and lifecycle status."""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ModelStatus",
    "ModelTier",
    "ModelType",
]


class ModelType(StrEnum):
    """classification of AI model by primary capability.

    :cvar CHAT: conversational language model
    :cvar EMBEDDING: text embedding model
    :cvar TRANSCRIPTION: audio-to-text transcription model
    :cvar IMAGE_GENERATION: text-to-image or image-to-image generation model
    :cvar SPEECH: text-to-speech synthesis model
    :cvar RERANKING: document re-ranking model for RAG pipelines
    """

    CHAT = "chat"
    EMBEDDING = "embedding"
    TRANSCRIPTION = "transcription"
    IMAGE_GENERATION = "image_generation"
    SPEECH = "speech"
    RERANKING = "reranking"


class ModelStatus(StrEnum):
    """lifecycle status of registered model.

    :cvar ACTIVE: model is available for use
    :cvar DEPRECATED: model still works but scheduled for removal
    :cvar DISABLED: model is unavailable
    """

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


class ModelTier(StrEnum):
    """relative size and cost tier of model.

    :cvar SMALL: lightweight, low-cost model
    :cvar MEDIUM: balanced performance and cost model
    :cvar LARGE: highest capability, highest cost model
    """

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
