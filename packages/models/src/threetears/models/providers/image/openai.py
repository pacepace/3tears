"""OpenAI DALL-E image generation provider using httpx."""

from __future__ import annotations

import base64

import httpx

from threetears.media.contracts import GeneratedImage, ImageGenerationBackend

__all__ = [
    "OpenAIImageProvider",
]


class OpenAIImageProvider:
    """image generation provider for OpenAI DALL-E API via httpx.

    supports text-to-image generation via /images/generations and
    image-to-image editing via /images/edits endpoints. returns
    base64-decoded PNG image data.

    :param api_key: OpenAI API key for authentication
    :ptype api_key: str
    :param model_name: DALL-E model identifier
    :ptype model_name: str
    :param base_url: API base URL
    :ptype base_url: str
    :param size: output image size (e.g. "1024x1024")
    :ptype size: str
    :param quality: image quality setting
    :ptype quality: str
    :param timeout: request timeout in seconds
    :ptype timeout: int
    """

    def __init__(
        self,
        api_key: str,
        *,
        model_name: str = "dall-e-3",
        base_url: str = "https://api.openai.com/v1",
        size: str = "1024x1024",
        quality: str = "standard",
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.size = size
        self.quality = quality
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        *,
        style: str | None = None,
        source_image: bytes | None = None,
        source_mime_type: str | None = None,
    ) -> GeneratedImage:
        """generates image from text prompt via OpenAI DALL-E API.

        uses /images/generations for text-to-image or /images/edits
        when source_image is provided for image-to-image editing.

        :param prompt: text description of image to generate
        :ptype prompt: str
        :param style: optional style modifier for generation
        :ptype style: str | None
        :param source_image: optional source image bytes for img2img editing
        :ptype source_image: bytes | None
        :param source_mime_type: MIME type of source image
        :ptype source_mime_type: str | None
        :return: generated image result with PNG data
        :rtype: GeneratedImage
        :raises httpx.HTTPStatusError: if API returns error status code
        """
        headers = {"Authorization": f"Bearer {self._api_key}"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if source_image is not None:
                response = await self._img2img(
                    client,
                    headers,
                    prompt,
                    source_image,
                    source_mime_type,
                )
            else:
                response = await self._txt2img(client, headers, prompt, style)

        response.raise_for_status()

        data = response.json()
        b64_data = data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64_data)
        w, h = self.size.split("x")

        result = GeneratedImage(
            data=image_bytes,
            mime_type="image/png",
            width=int(w),
            height=int(h),
        )
        return result

    async def _txt2img(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        prompt: str,
        style: str | None,
    ) -> httpx.Response:
        """sends text-to-image request to /images/generations.

        :param client: httpx async client
        :ptype client: httpx.AsyncClient
        :param headers: request headers with authorization
        :ptype headers: dict[str, str]
        :param prompt: text description of image to generate
        :ptype prompt: str
        :param style: optional style modifier
        :ptype style: str | None
        :return: API response
        :rtype: httpx.Response
        """
        payload: dict[str, str | int] = {
            "model": self.model_name,
            "prompt": prompt,
            "size": self.size,
            "quality": self.quality,
            "response_format": "b64_json",
            "n": 1,
        }
        if style is not None:
            payload["style"] = style

        response = await client.post(
            f"{self.base_url}/images/generations",
            headers=headers,
            json=payload,
        )
        return response

    async def _img2img(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        prompt: str,
        source_image: bytes,
        source_mime_type: str | None,
    ) -> httpx.Response:
        """sends image-to-image request to /images/edits.

        :param client: httpx async client
        :ptype client: httpx.AsyncClient
        :param headers: request headers with authorization
        :ptype headers: dict[str, str]
        :param prompt: text description of desired edit
        :ptype prompt: str
        :param source_image: source image bytes
        :ptype source_image: bytes
        :param source_mime_type: MIME type of source image
        :ptype source_mime_type: str | None
        :return: API response
        :rtype: httpx.Response
        """
        mime = source_mime_type or "image/png"
        files = {"image": ("source.png", source_image, mime)}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "size": self.size,
            "response_format": "b64_json",
            "n": "1",
        }

        response = await client.post(
            f"{self.base_url}/images/edits",
            headers=headers,
            files=files,
            data=data,
        )
        return response


# register protocol compliance for static analysis
_: type[ImageGenerationBackend] = OpenAIImageProvider
