"""image generation provider implementations.

concrete provider adapters for image generation APIs including
OpenAI DALL-E, HuggingFace Inference, Automatic1111, ModelsLab,
and ComfyUI. all providers implement ImageGenerationBackend protocol.
ComfyUIClient and SwarmUIClient are those servers' APIs for a caller that
brings its own graph or parameters.
"""

from __future__ import annotations

from threetears.models.providers.image.a1111 import A1111ImageProvider
from threetears.models.providers.image.comfyui import ComfyUIClient, ComfyUIError, ComfyUIImageProvider
from threetears.models.providers.image.huggingface import HuggingFaceImageProvider
from threetears.models.providers.image.modelslab import ModelsLabImageProvider
from threetears.models.providers.image.http_retry import RetryPolicy
from threetears.models.providers.image.openai import OpenAIImageProvider
from threetears.models.providers.image.swarmui import SwarmUIClient, SwarmUIError

__all__ = [
    "A1111ImageProvider",
    "ComfyUIClient",
    "ComfyUIError",
    "ComfyUIImageProvider",
    "HuggingFaceImageProvider",
    "ModelsLabImageProvider",
    "OpenAIImageProvider",
    "RetryPolicy",
    "SwarmUIClient",
    "SwarmUIError",
]
