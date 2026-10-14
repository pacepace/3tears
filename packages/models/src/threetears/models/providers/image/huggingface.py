"""HuggingFace Inference API image generation provider using httpx."""

from __future__ import annotations

from typing import Any

import httpx

from threetears.media.contracts import GeneratedImage, ImageGenerationBackend
from threetears.models.providers.image.options import dimensions_for, result_metadata

__all__ = [
    "HuggingFaceImageProvider",
]


class HuggingFaceImageProvider:
    """image generation provider for HuggingFace Inference API via httpx.

    sends text prompts to HuggingFace model endpoints and receives
    raw image bytes in response. does not support img2img; source_image
    parameter is ignored when provided, and the result says so.

    size, aspect and seed go to the text-to-image task as its ``width``,
    ``height`` and ``seed`` parameters, sent only when asked, so a model's own
    defaults stand otherwise.

    :param api_key: HuggingFace API token for authentication
    :ptype api_key: str
    :param model_id: HuggingFace model identifier
    :ptype model_id: str
    :param base_url: HuggingFace Inference API base URL
    :ptype base_url: str
    :param timeout: request timeout in seconds
    :ptype timeout: int
    """

    def __init__(
        self,
        api_key: str,
        *,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        base_url: str = "https://api-inference.huggingface.co",
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    #: a source image never shapes the result, so an edit is refused before it is paid for
    takes_source_image = False

    async def generate(
        self,
        prompt: str,
        *,
        style: str | None = None,
        source_image: bytes | None = None,
        source_mime_type: str | None = None,
        size: str | None = None,
        aspect: str | None = None,
        seed: int | None = None,
    ) -> GeneratedImage:
        """generates image from text prompt via HuggingFace Inference API.

        posts prompt to model endpoint and returns raw image bytes.
        source_image is ignored as HuggingFace Inference API does not
        support img2img via this endpoint.

        :param prompt: text description of image to generate
        :ptype prompt: str
        :param style: optional style modifier prepended to prompt
        :ptype style: str | None
        :param source_image: ignored (not supported by this provider)
        :ptype source_image: bytes | None
        :param source_mime_type: ignored (not supported by this provider)
        :ptype source_mime_type: str | None
        :param size: ``WIDTHxHEIGHT``, at most 2048 on a side
        :ptype size: str | None
        :param aspect: a shape when no size is given, on a 1024 longest side
        :ptype aspect: str | None
        :param seed: a number that repeats an image for the same prompt
        :ptype seed: int | None
        :return: generated image result with PNG data
        :rtype: GeneratedImage
        :raises ValueError: when the size or aspect cannot be read, or the size is too large
        :raises httpx.HTTPStatusError: if API returns error status code
        """
        effective_prompt = prompt
        if style is not None:
            effective_prompt = f"{style} style, {prompt}"

        parameters: dict[str, int] = {}
        made_size: str | None = None
        if size or aspect:
            width, height = dimensions_for(size, aspect, default=(1024, 1024), longest=2048)
            parameters.update(width=width, height=height)
            made_size = f"{width}x{height}"
        if seed is not None:
            parameters["seed"] = seed

        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload: dict[str, Any] = {"inputs": effective_prompt}
        if parameters:
            payload["parameters"] = parameters

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/models/{self._model_id}",
                headers=headers,
                json=payload,
            )

        response.raise_for_status()

        result = GeneratedImage(
            data=response.content,
            mime_type="image/png",
            width=parameters.get("width"),
            height=parameters.get("height"),
            metadata=result_metadata(
                {"model": self._model_id},
                size=made_size,
                seed=seed,
                source_given=source_image is not None,
                source_used=False,
                ignored={},
            ),
        )
        return result


# register protocol compliance for static analysis
_: type[ImageGenerationBackend] = HuggingFaceImageProvider
