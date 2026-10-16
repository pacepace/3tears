"""Automatic1111 Stable Diffusion WebUI image generation provider using httpx."""

from __future__ import annotations

import base64
import json

import httpx

from threetears.media.contracts import GeneratedImage, ImageGenerationBackend
from threetears.models.providers.image.options import dimensions_for, result_metadata

__all__ = [
    "A1111ImageProvider",
]


class A1111ImageProvider:
    """image generation provider for Automatic1111 Stable Diffusion WebUI API.

    supports text-to-image via /sdapi/v1/txt2img and image-to-image
    via /sdapi/v1/img2img endpoints. typically self-hosted without
    authentication.

    size and aspect set ``width`` and ``height`` (512 by 512 unless asked, at
    most 2048 on a side); a seed is sent only when asked, and the seed the
    server reports it used comes back in the result's metadata.

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

    #: img2img takes the source image
    takes_source_image = True

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
        :param size: ``WIDTHxHEIGHT``, at most 2048 on a side
        :ptype size: str | None
        :param aspect: a shape when no size is given, on a 512 longest side
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
        width, height = dimensions_for(size, aspect, default=(512, 512), longest=2048)

        payload: dict[str, object] = {
            "prompt": effective_prompt,
            "negative_prompt": self._negative_prompt,
            "steps": self._steps,
            "cfg_scale": self._cfg_scale,
            "sampler_name": self._sampler_name,
            "width": width,
            "height": height,
            "batch_size": 1,
        }
        if seed is not None:
            payload["seed"] = seed

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
            width=width,
            height=height,
            metadata=result_metadata(
                {},
                size=f"{width}x{height}",
                seed=_seed_used(data.get("info"), seed),
                source_given=source_image is not None,
                source_used=source_image is not None,
                ignored={},
            ),
        )
        return result


def _seed_used(info: object, asked: int | None) -> int | None:
    """The seed the server says it used, from the JSON string in its ``info`` field.

    :param info: the response's ``info``: a JSON string, or absent
    :ptype info: object
    :param asked: the seed asked for, said when the server does not say one
    :ptype asked: int | None
    :return: the seed used, when known
    :rtype: int | None
    """
    if not isinstance(info, str):
        return asked
    try:
        said = json.loads(info)
    except ValueError:
        # NOSILENT: "info" is the server's own optional report; without it the seed asked for is the one said
        return asked
    seed = said.get("seed") if isinstance(said, dict) else None
    return seed if isinstance(seed, int) else asked


# register protocol compliance for static analysis
_: type[ImageGenerationBackend] = A1111ImageProvider
