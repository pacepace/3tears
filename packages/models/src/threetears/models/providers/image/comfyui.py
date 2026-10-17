"""ComfyUI: a client for its HTTP API, and an image generation provider built on it.

:class:`ComfyUIClient` is one ComfyUI server's API for a caller that brings its own graph (a
workflow in ComfyUI's API format): put input pictures in its input folder, queue the graph, wait
for it, read its pictures back, and ask how busy the server is. Reads are retried with backoff
when the server is unreachable, times out or answers 5xx; queueing a graph and uploading a
picture are retried only when the request never left (a refused connection), so a slow answer
never queues the same graph twice. A graph ComfyUI rejects or fails raises :class:`ComfyUIError`
with ComfyUI's own messages.

:class:`ComfyUIImageProvider` is the ``ImageGenerationBackend`` over a built-in txt2img graph.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from types import TracebackType
from typing import Any

import httpx

from threetears.media.contracts import GeneratedImage, ImageGenerationBackend
from threetears.models.providers.image import http_retry
from threetears.models.providers.image.http_retry import RetryPolicy
from threetears.models.providers.image.options import dimensions_for, result_metadata

__all__ = [
    "ComfyUIClient",
    "ComfyUIError",
    "ComfyUIImageProvider",
]

#: the most of ComfyUI's error text kept on a :class:`ComfyUIError`
_MAX_ERROR_TEXT = 2000


class ComfyUIError(RuntimeError):
    """ComfyUI refused a request or a graph failed.

    :param message: what went wrong, with ComfyUI's own words when it gave any
    :ptype message: str
    :param status_code: the HTTP status, when the server answered one; ``None`` for a graph that
        was queued and then failed
    :ptype status_code: int | None
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """record the message and the status.

        :param message: what went wrong
        :ptype message: str
        :param status_code: the HTTP status, if any
        :ptype status_code: int | None
        :return: nothing
        :rtype: None
        """
        super().__init__(message)
        self.status_code = status_code


class ComfyUIClient:
    """one ComfyUI server's HTTP API, open for the length of an ``async with`` block.

    :param base_url: the server's address (``http://host:8188``), or any prefix that proxies its
        API (SwarmUI's ``.../ComfyBackendDirect``)
    :ptype base_url: str
    :param timeout: seconds each request may take
    :ptype timeout: float
    :param poll_interval: seconds between looks at a queued graph's history
    :ptype poll_interval: float
    :param retry: sends per request and the pauses between them; ``None`` is three sends
    :ptype retry: RetryPolicy | None
    :param client_id: the ``client_id`` sent with each queued graph (ComfyUI's per-client channel);
        ``None`` sends none
    :ptype client_id: str | None
    :param transport: an httpx transport (a test's ``httpx.MockTransport``); ``None`` is the network
    :ptype transport: httpx.AsyncBaseTransport | None
    :param sleep: how the pause between looks at a queued graph is taken; ``None`` is ``asyncio.sleep``
    :ptype sleep: Callable[[float], Awaitable[None]] | None
    :param clock: a monotonic clock for waits; ``None`` is ``time.monotonic``
    :ptype clock: Callable[[], float] | None
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        poll_interval: float = 2.0,
        retry: RetryPolicy | None = None,
        client_id: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """capture the settings; the connection opens on ``async with``.

        :param base_url: the server's address
        :ptype base_url: str
        :param timeout: seconds each request may take
        :ptype timeout: float
        :param poll_interval: seconds between looks at a queued graph
        :ptype poll_interval: float
        :param retry: sends per request and pauses, or ``None``
        :ptype retry: RetryPolicy | None
        :param client_id: sent with each queued graph, or ``None``
        :ptype client_id: str | None
        :param transport: an httpx transport, or ``None``
        :ptype transport: httpx.AsyncBaseTransport | None
        :param sleep: how a pause between looks is taken, or ``None``
        :ptype sleep: Callable[[float], Awaitable[None]] | None
        :param clock: a monotonic clock, or ``None``
        :ptype clock: Callable[[], float] | None
        :return: nothing
        :rtype: None
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.client_id = client_id
        self._retry = retry or RetryPolicy()
        self._transport = transport
        self._sleep = sleep
        self._clock = clock or time.monotonic
        self._stack: contextlib.AsyncExitStack | None = None
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> ComfyUIClient:
        """open the connection.

        :return: self
        :rtype: ComfyUIClient
        """
        stack = contextlib.AsyncExitStack()
        self._http = await stack.enter_async_context(httpx.AsyncClient(timeout=self.timeout, transport=self._transport))
        self._stack = stack
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """close the connection.

        :param exc_type: the exception's type, if the block raised
        :ptype exc_type: type[BaseException] | None
        :param exc: the exception, if the block raised
        :ptype exc: BaseException | None
        :param tb: its traceback
        :ptype tb: TracebackType | None
        :return: nothing
        :rtype: None
        """
        stack, self._stack, self._http = self._stack, None, None
        if stack is not None:
            await stack.aclose()

    # --- requests ------------------------------------------------------------------------------

    async def _send(
        self, method: str, path: str, *, repeatable: bool, timeout: float | None = None, **kwargs: Any
    ) -> httpx.Response:
        """one request, sent again as far as is safe (``http_retry.send``); a 4xx or 5xx raises.

        :param method: ``GET`` or ``POST``
        :ptype method: str
        :param path: the path under ``base_url``
        :ptype path: str
        :param repeatable: True when running it twice changes nothing (a read)
        :ptype repeatable: bool
        :param timeout: this request's timeout in seconds; ``None`` is the client's
        :ptype timeout: float | None
        :param kwargs: what httpx's ``get``/``post`` take (``json``, ``params``, ``files``)
        :ptype kwargs: Any
        :return: the answer (2xx or 3xx)
        :rtype: httpx.Response
        :raises RuntimeError: when used outside ``async with``
        :raises ComfyUIError: when the server refuses (4xx), or fails (5xx) past what may be retried
        :raises httpx.TransportError: when the server can't be reached past the attempts
        """
        if self._http is None:
            raise RuntimeError("ComfyUIClient is used inside `async with` only")
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = await http_retry.send(
            self._http, method, f"{self.base_url}{path}", repeatable=repeatable, policy=self._retry, **kwargs
        )
        if response.status_code >= 400:
            raise ComfyUIError(
                f"ComfyUI {method} {path} answered {response.status_code}: {_text(response)}",
                status_code=response.status_code,
            )
        return response

    async def queue(self, graph: Mapping[str, Any]) -> str:
        """queue a graph (ComfyUI's API format). Returns its prompt id.

        :param graph: the graph, node id -> ``{"class_type", "inputs"}``
        :ptype graph: Mapping[str, Any]
        :return: the prompt id ComfyUI gave it
        :rtype: str
        :raises ComfyUIError: when ComfyUI rejects the graph (its node errors are in the message)
        """
        body_out: dict[str, Any] = {"prompt": graph}
        if self.client_id is not None:
            body_out["client_id"] = self.client_id
        response = await self._send("POST", "/prompt", repeatable=False, json=body_out)
        body = response.json()
        if body.get("node_errors"):
            raise ComfyUIError(f"ComfyUI rejected the graph: {_short(body['node_errors'])}")
        return str(body["prompt_id"])

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        """a queued graph's history entry, or ``None`` while it hasn't finished.

        :param prompt_id: the graph's prompt id
        :ptype prompt_id: str
        :return: its entry (``outputs``, ``status``), or None
        :rtype: dict[str, Any] | None
        """
        response = await self._send("GET", f"/history/{prompt_id}", repeatable=True)
        entry = response.json().get(prompt_id)
        return entry if isinstance(entry, dict) else None

    async def wait(
        self, prompt_id: str, *, max_wait_seconds: float | None = None, max_polls: int | None = None
    ) -> dict[str, Any]:
        """wait for a queued graph to finish. Returns its history entry.

        :param prompt_id: the graph's prompt id
        :ptype prompt_id: str
        :param max_wait_seconds: how long to wait; ``None`` with no ``max_polls`` waits ten minutes
        :ptype max_wait_seconds: float | None
        :param max_polls: how many looks to take at most
        :ptype max_polls: int | None
        :return: the history entry
        :rtype: dict[str, Any]
        :raises ComfyUIError: when the graph failed (ComfyUI's messages in the text)
        :raises TimeoutError: when it hadn't finished within the bound
        """
        if max_wait_seconds is None and max_polls is None:
            max_wait_seconds = 600.0
        until = None if max_wait_seconds is None else self._clock() + max_wait_seconds
        polls = 0
        while True:
            await (self._sleep or asyncio.sleep)(self.poll_interval)
            polls += 1
            entry = await self.history(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise ComfyUIError(f"ComfyUI graph {prompt_id} failed: {_short(status.get('messages', []))}")
                return entry
            if (max_polls is not None and polls >= max_polls) or (until is not None and self._clock() >= until):
                raise TimeoutError(f"ComfyUI graph {prompt_id} did not finish (after {polls} looks)")

    @staticmethod
    def images(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
        """the pictures a finished graph saved, in node order: ``{"filename", "subfolder", "type"}``.

        :param entry: a history entry (``wait``'s answer)
        :ptype entry: Mapping[str, Any]
        :return: the pictures
        :rtype: list[dict[str, Any]]
        """
        outputs = entry.get("outputs") or {}
        return [dict(im) for node in outputs.values() for im in (node.get("images") or [])]

    async def view(self, image: Mapping[str, Any]) -> bytes:
        """one saved picture's bytes.

        :param image: one of ``images``' entries (a bare ``{"filename"}`` reads the output folder)
        :ptype image: Mapping[str, Any]
        :return: the picture
        :rtype: bytes
        """
        params = {"filename": image["filename"]}
        for key in ("subfolder", "type"):
            if image.get(key):
                params[key] = image[key]
        return (await self._send("GET", "/view", repeatable=True, params=params)).content

    async def run(self, graph: Mapping[str, Any], *, max_wait_seconds: float | None = None) -> list[dict[str, Any]]:
        """queue a graph, wait for it, and return the pictures it saved (``images``).

        :param graph: the graph
        :ptype graph: Mapping[str, Any]
        :param max_wait_seconds: how long to wait for it
        :ptype max_wait_seconds: float | None
        :return: the pictures
        :rtype: list[dict[str, Any]]
        """
        prompt_id = await self.queue(graph)
        return self.images(await self.wait(prompt_id, max_wait_seconds=max_wait_seconds))

    async def upload_image(self, data: bytes, filename: str, *, content_type: str = "image/png") -> str:
        """put a picture in the server's input folder (for a ``LoadImage`` node). Returns its name there.

        :param data: the picture
        :ptype data: bytes
        :param filename: its name; ComfyUI may give it another when the name is taken
        :ptype filename: str
        :param content_type: its media type
        :ptype content_type: str
        :return: the name a ``LoadImage`` node reads it by
        :rtype: str
        """
        response = await self._send(
            "POST", "/upload/image", repeatable=False, files={"image": (filename, data, content_type)}
        )
        body = response.json()
        sub = body.get("subfolder") or ""
        return f"{sub}/{body['name']}" if sub else str(body["name"])

    async def queue_depth(self, *, timeout: float | None = None) -> int:
        """how many graphs are running or waiting on the server.

        :param timeout: this request's timeout; ``None`` is the client's
        :ptype timeout: float | None
        :return: running plus pending
        :rtype: int
        """
        body = (await self._send("GET", "/queue", repeatable=True, timeout=timeout)).json()
        return len(body.get("queue_running") or []) + len(body.get("queue_pending") or [])


def _text(response: httpx.Response) -> str:
    """the start of an error answer's body, for a message.

    :param response: the answer
    :ptype response: httpx.Response
    :return: its text, cut short
    :rtype: str
    """
    try:
        return response.text[:_MAX_ERROR_TEXT]
    except UnicodeDecodeError, AttributeError:
        return "(no readable body)"


def _short(value: Any) -> str:
    """ComfyUI's error structure as text, cut short.

    :param value: the structure
    :ptype value: Any
    :return: its JSON, cut short
    :rtype: str
    """
    return json.dumps(value, default=str)[:_MAX_ERROR_TEXT]


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
    """image generation provider for ComfyUI API, over :class:`ComfyUIClient`.

    submits the built-in txt2img workflow, polls history for completion and
    fetches the generated image. typically self-hosted without authentication.
    a server that can't be reached, or that answers 5xx, is tried again on
    reads; the workflow itself is sent again only when it never left.

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
        :raises ComfyUIError: if ComfyUI refuses the workflow or it fails
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

        async with ComfyUIClient(self.base_url, timeout=self.timeout, poll_interval=self._poll_interval) as comfy:
            prompt_id = await comfy.queue(workflow)
            pictures = ComfyUIClient.images(await comfy.wait(prompt_id, max_polls=self._max_polls))
            if not pictures:
                raise ComfyUIError(f"ComfyUI graph {prompt_id} finished without a picture")
            data = await comfy.view({"filename": pictures[0]["filename"]})

        result = GeneratedImage(
            data=data,
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
