"""image generation provider implementations.

concrete provider adapters for image generation APIs including
OpenAI DALL-E, HuggingFace Inference, Automatic1111, ModelsLab,
and ComfyUI. all providers implement ImageGenerationBackend protocol.
"""

from __future__ import annotations

from threetears.models.providers.image.a1111 import A1111ImageProvider
from threetears.models.providers.image.comfyui import ComfyUIImageProvider
from threetears.models.providers.image.huggingface import HuggingFaceImageProvider
from threetears.models.providers.image.modelslab import ModelsLabImageProvider
from threetears.models.providers.image.openai import OpenAIImageProvider

__all__ = [
    "A1111ImageProvider",
    "ComfyUIImageProvider",
    "HuggingFaceImageProvider",
    "ModelsLabImageProvider",
    "OpenAIImageProvider",
]
