"""OpenAI DALL-E and GPT Image generation provider using httpx."""

from __future__ import annotations

import base64
import math

import httpx

from threetears.media.contracts import GeneratedImage, ImageGenerationBackend
from threetears.models.providers.image.options import dimensions_for, parse_size, result_metadata

__all__ = [
    "OpenAIImageProvider",
]

#: Models that take the GPT Image request shape: no ``style``, no ``response_format``,
#: quality ``low`` / ``medium`` / ``high`` / ``auto``.
_GPT_IMAGE_MODELS = frozenset({"gpt-image-1", "chatgpt-image-latest"})

#: The sizes each model makes; the API refuses any other.
_SIZES: dict[str, tuple[str, ...]] = {
    "dall-e-3": ("1024x1024", "1792x1024", "1024x1792"),
    "dall-e-2": ("256x256", "512x512", "1024x1024"),
    **dict.fromkeys(_GPT_IMAGE_MODELS, ("1024x1024", "1536x1024", "1024x1536")),
}

#: DALL-E quality words as GPT Image models take them.
_GPT_IMAGE_QUALITY = {"standard": "auto", "hd": "high"}


def _closest(sizes: tuple[str, ...], aspect: str) -> str:
    """The one of the model's sizes whose shape is nearest the aspect asked for.

    :param sizes: the sizes the model makes
    :ptype sizes: tuple[str, ...]
    :param aspect: a named shape or a ratio
    :ptype aspect: str
    :return: one of ``sizes``
    :rtype: str
    :raises ValueError: when the aspect cannot be read
    """
    width, height = dimensions_for(None, aspect, default=(1024, 1024), longest=4096)
    wanted = math.log(width / height)

    def _distance(size: str) -> float:
        w, h = parse_size(size)
        return abs(math.log(w / h) - wanted)

    return min(sizes, key=_distance)


class OpenAIImageProvider:
    """image generation provider for OpenAI DALL-E and GPT Image APIs via httpx.

    supports text-to-image generation via /images/generations and
    image-to-image editing via /images/edits endpoints. returns
    base64-decoded PNG image data.

    a size is checked against the sizes the model makes and refused when it is
    not one of them; an aspect picks the model's size nearest that shape. the
    service takes no seed, so a seed asked for is named as not applied.

    :param api_key: OpenAI API key for authentication
    :ptype api_key: str
    :param model_name: image model identifier
    :ptype model_name: str
    :param base_url: API base URL
    :ptype base_url: str
    :param size: output image size when none is asked for (e.g. "1024x1024")
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

    @property
    def _is_gpt_image(self) -> bool:
        """Whether the configured model takes the GPT Image request shape.

        :return: ``True`` for a GPT Image model
        :rtype: bool
        """
        return self.model_name in _GPT_IMAGE_MODELS

    def _size_for(self, size: str | None, aspect: str | None) -> tuple[str, dict[str, str]]:
        """The size to send, and what of size and aspect could not be applied.

        :param size: ``WIDTHxHEIGHT`` asked for, or ``None``
        :ptype size: str | None
        :param aspect: a shape asked for, or ``None``
        :ptype aspect: str | None
        :return: the size to send and ``{name: why}`` for an option not applied
        :rtype: tuple[str, dict[str, str]]
        :raises ValueError: when the size is not one the model makes, or cannot be read
        """
        allowed = _SIZES.get(self.model_name)
        if size:
            width, height = parse_size(size)
            asked = f"{width}x{height}"
            if allowed and asked not in allowed:
                raise ValueError(f"{self.model_name} makes only {', '.join(allowed)}; {size} is not one of them")
            return asked, {}
        if aspect:
            if allowed:
                return _closest(allowed, aspect), {}
            return self.size, {"aspect": "the sizes this model makes are not known here; give a size instead"}
        return self.size, {}

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
        """generates image from text prompt via the OpenAI image API.

        uses /images/generations for text-to-image or /images/edits
        when source_image is provided for image-to-image editing.

        :param prompt: text description of image to generate
        :ptype prompt: str
        :param style: optional style modifier; DALL-E only, a GPT Image model names it as not applied
        :ptype style: str | None
        :param source_image: optional source image bytes for img2img editing
        :ptype source_image: bytes | None
        :param source_mime_type: MIME type of source image
        :ptype source_mime_type: str | None
        :param size: ``WIDTHxHEIGHT``, one the model makes
        :ptype size: str | None
        :param aspect: a shape when no size is given; the model's nearest size is used
        :ptype aspect: str | None
        :param seed: not taken by the service; named as not applied
        :ptype seed: int | None
        :return: generated image result with PNG data
        :rtype: GeneratedImage
        :raises ValueError: when the size is not one the model makes, or the size or aspect cannot be read
        :raises httpx.HTTPStatusError: if API returns error status code
        """
        made_size, ignored = self._size_for(size, aspect)
        if seed is not None:
            ignored["seed"] = "OpenAI's image models take no seed"
        if style is not None and self._is_gpt_image:
            ignored["style"] = "GPT Image models take no style; say it in the prompt"
            style = None

        headers = {"Authorization": f"Bearer {self._api_key}"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if source_image is not None:
                response = await self._img2img(
                    client,
                    headers,
                    prompt,
                    source_image,
                    source_mime_type,
                    made_size,
                )
            else:
                response = await self._txt2img(client, headers, prompt, style, made_size)

        response.raise_for_status()

        data = response.json()
        b64_data = data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64_data)
        w, h = parse_size(made_size)

        result = GeneratedImage(
            data=image_bytes,
            mime_type="image/png",
            width=w,
            height=h,
            metadata=result_metadata(
                {"model": self.model_name},
                size=made_size,
                seed=None,
                source_given=source_image is not None,
                source_used=source_image is not None,
                ignored=ignored,
            ),
        )
        return result

    def _quality(self) -> str:
        """The quality word the configured model takes.

        :return: ``self.quality``, as a GPT Image model reads it when it is one
        :rtype: str
        """
        if self._is_gpt_image:
            return _GPT_IMAGE_QUALITY.get(self.quality, self.quality)
        return self.quality

    async def _txt2img(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        prompt: str,
        style: str | None,
        size: str,
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
        :param size: the size to make
        :ptype size: str
        :return: API response
        :rtype: httpx.Response
        """
        payload: dict[str, str | int] = {
            "model": self.model_name,
            "prompt": prompt,
            "size": size,
            "quality": self._quality(),
        }
        # GPT Image models always answer in base64 and refuse response_format
        if not self._is_gpt_image:
            payload["response_format"] = "b64_json"
        payload["n"] = 1
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
        size: str,
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
        :param size: the size to make
        :ptype size: str
        :return: API response
        :rtype: httpx.Response
        """
        mime = source_mime_type or "image/png"
        files = {"image": ("source.png", source_image, mime)}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "size": size,
        }
        if not self._is_gpt_image:
            data["response_format"] = "b64_json"
        data["n"] = "1"

        response = await client.post(
            f"{self.base_url}/images/edits",
            headers=headers,
            files=files,
            data=data,
        )
        return response


# register protocol compliance for static analysis
_: type[ImageGenerationBackend] = OpenAIImageProvider
