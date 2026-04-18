"""ModelsLab image generation provider using httpx with fire-and-poll pattern."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from threetears.agent.tools.protocols import GeneratedImage, ImageGenerationBackend

__all__ = [
    "ModelsLabImageProvider",
]


class ModelsLabImageProvider:
    """image generation provider for ModelsLab API via httpx.

    uses fire-and-poll pattern: submits generation request, then
    polls for completion until result is ready or timeout is reached.

    :param api_key: ModelsLab API key for authentication
    :ptype api_key: str
    :param base_url: ModelsLab API base URL
    :ptype base_url: str
    :param model_id: model identifier for generation
    :ptype model_id: str
    :param timeout: request timeout in seconds
    :ptype timeout: int
    :param poll_interval: seconds between poll requests
    :ptype poll_interval: float
    :param max_polls: maximum number of poll attempts
    :ptype max_polls: int
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://modelslab.com/api/v6",
        model_id: str = "midjourney",
        timeout: int = 120,
        poll_interval: float = 3.0,
        max_polls: int = 40,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._max_polls = max_polls

    async def generate(
        self,
        prompt: str,
        *,
        style: str | None = None,
        source_image: bytes | None = None,
        source_mime_type: str | None = None,
    ) -> GeneratedImage:
        """generates image via ModelsLab API with polling for completion.

        submits text-to-image request and polls for result. if initial
        response indicates processing, polls fetch_result URL until
        completion or max_polls is exceeded.

        :param prompt: text description of image to generate
        :ptype prompt: str
        :param style: optional style modifier prepended to prompt
        :ptype style: str | None
        :param source_image: unused (not supported by this provider)
        :ptype source_image: bytes | None
        :param source_mime_type: unused (not supported by this provider)
        :ptype source_mime_type: str | None
        :return: generated image result with PNG data
        :rtype: GeneratedImage
        :raises TimeoutError: if max poll attempts exceeded
        :raises httpx.HTTPStatusError: if API returns error status code
        """
        effective_prompt = prompt
        if style is not None:
            effective_prompt = f"{style} style, {prompt}"

        payload: dict[str, Any] = {
            "key": self._api_key,
            "model_id": self._model_id,
            "prompt": effective_prompt,
            "samples": 1,
            "negative_prompt": "",
            "width": 512,
            "height": 512,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/images/text2img",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            image_url = await self._resolve_result(client, data)

            image_response = await client.get(image_url)
            image_response.raise_for_status()

        result = GeneratedImage(
            data=image_response.content,
            mime_type="image/png",
        )
        return result

    async def _resolve_result(
        self,
        client: httpx.AsyncClient,
        data: dict[str, Any],
    ) -> str:
        """resolves image URL from initial response or by polling.

        :param client: httpx async client
        :ptype client: httpx.AsyncClient
        :param data: initial API response data
        :ptype data: dict[str, Any]
        :return: URL of generated image
        :rtype: str
        :raises TimeoutError: if max poll attempts exceeded
        """
        if data.get("status") == "success":
            return str(data["output"][0])

        fetch_url = data.get("fetch_result", "")
        for _ in range(self._max_polls):
            await asyncio.sleep(self._poll_interval)

            poll_response = await client.post(
                fetch_url,
                json={"key": self._api_key},
            )
            poll_response.raise_for_status()

            poll_data = poll_response.json()
            if poll_data.get("status") == "success":
                return str(poll_data["output"][0])

        raise TimeoutError(f"ModelsLab generation did not complete after {self._max_polls} polls")


# register protocol compliance for static analysis
_: type[ImageGenerationBackend] = ModelsLabImageProvider
