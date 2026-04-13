"""HuggingFace Inference API image generation provider using httpx."""

from __future__ import annotations

import httpx

from threetears.agent.tools.protocols import GeneratedImage, ImageGenerationBackend


class HuggingFaceImageProvider:
    """image generation provider for HuggingFace Inference API via httpx.

    sends text prompts to HuggingFace model endpoints and receives
    raw image bytes in response. does not support img2img; source_image
    parameter is ignored when provided.

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
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def generate(
        self,
        prompt: str,
        *,
        style: str | None = None,
        source_image: bytes | None = None,
        source_mime_type: str | None = None,
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
        :return: generated image result with PNG data
        :rtype: GeneratedImage
        :raises httpx.HTTPStatusError: if API returns error status code
        """
        effective_prompt = prompt
        if style is not None:
            effective_prompt = f"{style} style, {prompt}"

        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {"inputs": effective_prompt}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/models/{self._model_id}",
                headers=headers,
                json=payload,
            )

        response.raise_for_status()

        result = GeneratedImage(
            data=response.content,
            mime_type="image/png",
        )
        return result


# register protocol compliance for static analysis
_: type[ImageGenerationBackend] = HuggingFaceImageProvider
