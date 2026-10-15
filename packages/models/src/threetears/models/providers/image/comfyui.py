"""ComfyUI image generation provider using httpx with workflow submission and polling."""

from __future__ import annotations

import asyncio
import copy
import secrets
from typing import Any

import httpx

from threetears.media.contracts import GeneratedImage, ImageGenerationBackend
from threetears.models.providers.image.options import dimensions_for, result_metadata

__all__ = [
    "ComfyUIImageProvider",
]


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


#: The default workflow's sampler (its seed), empty latent (its size) and positive prompt.
_SAMPLER_NODE_ID = "3"
_LATENT_NODE_ID = "5"
_PROMPT_NODE_ID = "6"

#: The largest seed a KSampler takes.
_MAX_SEED = 2**64 - 1

#: The range a seed is drawn from when none is asked for.
_DRAWN_SEEDS = 2**32


class ComfyUIImageProvider:
    """image generation provider for ComfyUI API via httpx.

    submits workflow to ComfyUI prompt endpoint and polls history
    for completion. extracts output image filename and fetches
    generated image data. typically self-hosted without authentication.

    every image gets its own seed (the one asked for, else a new one), so the
    same prompt does not make the same picture each time; the seed used comes
    back in the result's metadata. size and aspect set the empty latent's width
    and height (512 by 512 unless asked, at most 2048 on a side).

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
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._poll_interval = poll_interval
        self._max_polls = max_polls

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
        :param size: ``WIDTHxHEIGHT``, at most 2048 on a side
        :ptype size: str | None
        :param aspect: a shape when no size is given, on a 512 longest side
        :ptype aspect: str | None
        :param seed: a number from 0 that repeats an image for the same prompt; a new
            one is drawn when none is asked for
        :ptype seed: int | None
        :return: generated image result with PNG data
        :rtype: GeneratedImage
        :raises ValueError: when the size or aspect cannot be read, or the size is too large
        :raises TimeoutError: if max poll attempts exceeded
        :raises httpx.HTTPStatusError: if API returns error status code
        """
        effective_prompt = prompt
        if style is not None:
            effective_prompt = f"{style} style, {prompt}"

        latent = _DEFAULT_WORKFLOW[_LATENT_NODE_ID]["inputs"]
        width, height = dimensions_for(
            size,
            aspect,
            default=(int(latent["width"]), int(latent["height"])),
            longest=2048,
        )
        ignored: dict[str, str] = {}
        if seed is not None and not 0 <= seed <= _MAX_SEED:
            ignored["seed"] = f"this server takes a seed from 0 to {_MAX_SEED}; a new one was used"
            seed = None
        used_seed = seed if seed is not None else secrets.randbelow(_DRAWN_SEEDS)

        workflow = _build_workflow(effective_prompt, width=width, height=height, seed=used_seed)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            submit_response = await client.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow},
            )
            submit_response.raise_for_status()

            prompt_id = submit_response.json()["prompt_id"]
            filename = await self._poll_for_result(client, prompt_id)

            image_response = await client.get(
                f"{self.base_url}/view",
                params={"filename": filename},
            )
            image_response.raise_for_status()

        result = GeneratedImage(
            data=image_response.content,
            mime_type="image/png",
            width=width,
            height=height,
            metadata=result_metadata(
                {"prompt_id": prompt_id},
                size=f"{width}x{height}",
                seed=used_seed,
                source_given=source_image is not None,
                source_used=False,
                ignored=ignored,
            ),
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
                f"{self.base_url}/history/{prompt_id}",
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


def _build_workflow(prompt_text: str, *, width: int, height: int, seed: int) -> dict[str, Any]:
    """builds ComfyUI workflow dict with the prompt, size and seed set.

    creates deep copy of default workflow and sets prompt text in the positive
    conditioning CLIPTextEncode node, the size in the empty latent and the seed
    in the sampler.

    :param prompt_text: text prompt for positive conditioning
    :ptype prompt_text: str
    :param width: width of the image to make
    :ptype width: int
    :param height: height of the image to make
    :ptype height: int
    :param seed: the sampler's seed
    :ptype seed: int
    :return: workflow dict ready for ComfyUI prompt endpoint
    :rtype: dict[str, Any]
    """
    workflow = copy.deepcopy(_DEFAULT_WORKFLOW)
    workflow[_PROMPT_NODE_ID]["inputs"]["text"] = prompt_text
    workflow[_LATENT_NODE_ID]["inputs"]["width"] = width
    workflow[_LATENT_NODE_ID]["inputs"]["height"] = height
    workflow[_SAMPLER_NODE_ID]["inputs"]["seed"] = seed
    return workflow


# register protocol compliance for static analysis
_: type[ImageGenerationBackend] = ComfyUIImageProvider
