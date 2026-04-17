"""ComfyUI image generation provider using httpx with workflow submission and polling."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from threetears.agent.tools.protocols import GeneratedImage, ImageGenerationBackend


# minimal default txt2img workflow for ComfyUI
# KSampler -> VAEDecode -> SaveImage pipeline
_DEFAULT_WORKFLOW: dict[str, Any] = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors",
        },
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1,
        },
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "",
            "clip": ["4", 1],
        },
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "",
            "clip": ["4", 1],
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2],
        },
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": ["8", 0],
        },
    },
}


class ComfyUIImageProvider:
    """image generation provider for ComfyUI API via httpx.

    submits workflow to ComfyUI prompt endpoint and polls history
    for completion. extracts output image filename and fetches
    generated image data. typically self-hosted without authentication.

    :param base_url: ComfyUI API base URL
    :ptype base_url: str
    :param timeout: request timeout in seconds
    :ptype timeout: int
    :param poll_interval: seconds between poll requests
    :ptype poll_interval: float
    :param max_polls: maximum number of poll attempts
    :ptype max_polls: int
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8188",
        timeout: int = 120,
        poll_interval: float = 2.0,
        max_polls: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
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
        """generates image via ComfyUI workflow submission and polling.

        builds default txt2img workflow with prompt in positive
        conditioning node, submits to ComfyUI, polls history for
        completion, and fetches resulting image.

        :param prompt: text description of image to generate
        :ptype prompt: str
        :param style: optional style modifier prepended to prompt
        :ptype style: str | None
        :param source_image: unused (not supported by default workflow)
        :ptype source_image: bytes | None
        :param source_mime_type: unused (not supported by default workflow)
        :ptype source_mime_type: str | None
        :return: generated image result with PNG data
        :rtype: GeneratedImage
        :raises TimeoutError: if max poll attempts exceeded
        :raises httpx.HTTPStatusError: if API returns error status code
        """
        effective_prompt = prompt
        if style is not None:
            effective_prompt = f"{style} style, {prompt}"

        workflow = _build_workflow(effective_prompt)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            submit_response = await client.post(
                f"{self._base_url}/prompt",
                json={"prompt": workflow},
            )
            submit_response.raise_for_status()

            prompt_id = submit_response.json()["prompt_id"]
            filename = await self._poll_for_result(client, prompt_id)

            image_response = await client.get(
                f"{self._base_url}/view",
                params={"filename": filename},
            )
            image_response.raise_for_status()

        result = GeneratedImage(
            data=image_response.content,
            mime_type="image/png",
        )
        return result

    async def _poll_for_result(
        self,
        client: httpx.AsyncClient,
        prompt_id: str,
    ) -> str:
        """polls ComfyUI history endpoint until result is available.

        :param client: httpx async client
        :ptype client: httpx.AsyncClient
        :param prompt_id: prompt ID from submission response
        :ptype prompt_id: str
        :return: output image filename
        :rtype: str
        :raises TimeoutError: if max poll attempts exceeded
        """
        for _ in range(self._max_polls):
            await asyncio.sleep(self._poll_interval)

            history_response = await client.get(
                f"{self._base_url}/history/{prompt_id}",
            )
            history_response.raise_for_status()

            history_data = history_response.json()
            if prompt_id in history_data:
                outputs = history_data[prompt_id].get("outputs", {})
                for node_output in outputs.values():
                    images = node_output.get("images", [])
                    if images:
                        return str(images[0]["filename"])

        raise TimeoutError(f"ComfyUI generation did not complete after {self._max_polls} polls")


def _build_workflow(prompt_text: str) -> dict[str, Any]:
    """builds ComfyUI workflow dict with prompt in positive conditioning node.

    creates deep copy of default workflow and sets prompt text
    in positive conditioning CLIPTextEncode node.

    :param prompt_text: text prompt for positive conditioning
    :ptype prompt_text: str
    :return: workflow dict ready for ComfyUI prompt endpoint
    :rtype: dict[str, Any]
    """
    import copy

    workflow = copy.deepcopy(_DEFAULT_WORKFLOW)
    workflow["6"]["inputs"]["text"] = prompt_text
    return workflow


# register protocol compliance for static analysis
_: type[ImageGenerationBackend] = ComfyUIImageProvider
