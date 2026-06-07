"""Automatic1111 Stable Diffusion WebUI image generation provider using httpx."""

from __future__ import annotations

import base64

import httpx

from threetears.media.contracts import GeneratedImage, ImageGenerationBackend

__all__ = [
    "A1111ImageProvider",
]


class A1111ImageProvider:
    """image generation provider for Automatic1111 Stable Diffusion WebUI API.

    supports text-to-image via /sdapi/v1/txt2img and image-to-image
    via /sdapi/v1/img2img endpoints. typically self-hosted without
    authentication.

    :param base_url: A1111 WebUI API base URL
    :ptype base_url: str
    :param timeout: request timeout in seconds
    :ptype timeout: int
    :param steps: number of sampling steps
    :ptype steps: int
    :param cfg_scale: classifier-free guidance scale
    :ptype cfg_scale: float
    :param sampler_name: sampling algorithm name
    :ptype sampler_name: str
    :param negative_prompt: negative prompt for generation
    :ptype negative_prompt: str
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:7860",
        timeout: int = 120,
        steps: int = 20,
        cfg_scale: float = 7.0,
        sampler_name: str = "Euler",
        negative_prompt: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._steps = steps
        self._cfg_scale = cfg_scale
        self._sampler_name = sampler_name
        self._negative_prompt = negative_prompt

    async def generate(
        self,
        prompt: str,
        *,
        style: str | None = None,
        source_image: bytes | None = None,
        source_mime_type: str | None = None,
    ) -> GeneratedImage:
        """generates image via Automatic1111 Stable Diffusion WebUI API.

        uses /sdapi/v1/txt2img for text-to-image or /sdapi/v1/img2img
        when source_image is provided for image-to-image generation.

        :param prompt: text description of image to generate
        :ptype prompt: str
        :param style: optional style modifier prepended to prompt
        :ptype style: str | None
        :param source_image: optional source image bytes for img2img
        :ptype source_image: bytes | None
        :param source_mime_type: MIME type of source image (unused)
        :ptype source_mime_type: str | None
        :return: generated image result with PNG data
        :rtype: GeneratedImage
        :raises httpx.HTTPStatusError: if API returns error status code
        """
        effective_prompt = prompt
        if style is not None:
            effective_prompt = f"{style} style, {prompt}"

        payload: dict[str, object] = {
            "prompt": effective_prompt,
            "negative_prompt": self._negative_prompt,
            "steps": self._steps,
            "cfg_scale": self._cfg_scale,
            "sampler_name": self._sampler_name,
            "width": 512,
            "height": 512,
            "batch_size": 1,
        }

        if source_image is not None:
            endpoint = f"{self.base_url}/sdapi/v1/img2img"
            b64_source = base64.b64encode(source_image).decode("ascii")
            payload["init_images"] = [b64_source]
        else:
            endpoint = f"{self.base_url}/sdapi/v1/txt2img"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(endpoint, json=payload)

        response.raise_for_status()

        data = response.json()
        image_bytes = base64.b64decode(data["images"][0])

        result = GeneratedImage(
            data=image_bytes,
            mime_type="image/png",
            width=512,
            height=512,
        )
        return result


# register protocol compliance for static analysis
_: type[ImageGenerationBackend] = A1111ImageProvider
