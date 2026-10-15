"""size, aspect and seed on the five image providers: what each sends, and what each says it could not apply.

every test drives the real provider against a recording fake service (an
``httpx.MockTransport``) and asserts on the request it recorded and on the
result's metadata. a call with none of the three options must send exactly the
body the provider sent before they existed.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from threetears.agent.tools.builtin.image_generation import create_image_generation_tool
from threetears.media.contracts import (
    GENERATION_IGNORED,
    GENERATION_SEED,
    GENERATION_SIZE,
    GENERATION_SOURCE_USED,
    GeneratedImage,
)
from threetears.models.providers.image import comfyui as comfyui_module
from threetears.models.providers.image.a1111 import A1111ImageProvider
from threetears.models.providers.image.comfyui import ComfyUIImageProvider
from threetears.models.providers.image.huggingface import HuggingFaceImageProvider
from threetears.models.providers.image.modelslab import ModelsLabImageProvider
from threetears.models.providers.image.openai import OpenAIImageProvider
from threetears.models.providers.image.options import dimensions_for

_REAL_CLIENT = httpx.AsyncClient
_PNG = b"fake-png"
_B64 = base64.b64encode(_PNG).decode()


class _Service:
    """a fake image service: answers each request with ``handler`` and keeps every request."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    def client(self, *_: Any, **kwargs: Any) -> httpx.AsyncClient:
        return _REAL_CLIENT(transport=httpx.MockTransport(self._handle), timeout=kwargs.get("timeout"))

    def json_body(self, index: int = 0) -> Any:
        return json.loads(self.requests[index].content)


@pytest.fixture
def serve() -> Iterator[Callable[[str, Callable[[httpx.Request], httpx.Response]], _Service]]:
    """patch a provider module's ``httpx.AsyncClient`` onto a recording fake service."""
    patches: list[Any] = []

    def _serve(module: str, handler: Callable[[httpx.Request], httpx.Response]) -> _Service:
        service = _Service(handler)
        p = patch(f"threetears.models.providers.image.{module}.httpx.AsyncClient", service.client)
        p.start()
        patches.append(p)
        return service

    yield _serve
    for p in patches:
        p.stop()


def _openai(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"b64_json": _B64}]})


def _bytes(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=_PNG)


def _modelslab(meta: dict[str, Any] | None = None) -> Callable[[httpx.Request], httpx.Response]:
    def _handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=_PNG)
        body: dict[str, Any] = {"status": "success", "output": ["https://img.example/out.png"]}
        if meta is not None:
            body["meta"] = meta
        return httpx.Response(200, json=body)

    return _handle


def _comfyui(request: httpx.Request) -> httpx.Response:
    if request.method == "POST":
        return httpx.Response(200, json={"prompt_id": "pid-1"})
    if request.url.path.startswith("/history/"):
        return httpx.Response(200, json={"pid-1": {"outputs": {"9": {"images": [{"filename": "out.png"}]}}}})
    return httpx.Response(200, content=_PNG)


def _a1111(info: str | None = None) -> Callable[[httpx.Request], httpx.Response]:
    def _handle(_: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = {"images": [_B64]}
        if info is not None:
            body["info"] = info
        return httpx.Response(200, json=body)

    return _handle


def _multipart_field(request: httpx.Request, name: str) -> str | None:
    """the value of one plain form field in a multipart request body, or ``None``."""
    marker = f'name="{name}"\r\n\r\n'.encode()
    content = request.content
    at = content.find(marker)
    if at < 0:
        return None
    start = at + len(marker)
    return content[start : content.index(b"\r\n", start)].decode()


# -- the shared reading of size and aspect -------------------------------------


class TestDimensionsFor:
    """dimensions_for reads a size, else an aspect, else keeps the default."""

    def test_size_wins_over_aspect(self) -> None:
        assert dimensions_for("640x480", "portrait", default=(512, 512), longest=2048) == (640, 480)

    def test_aspect_keeps_the_longest_side_on_multiples_of_64(self) -> None:
        assert dimensions_for(None, "16:9", default=(1024, 1024), longest=2048) == (1024, 576)
        assert dimensions_for(None, "portrait", default=(512, 512), longest=2048) == (320, 512)

    def test_nothing_asked_is_the_default(self) -> None:
        assert dimensions_for(None, None, default=(512, 512), longest=2048) == (512, 512)

    @pytest.mark.parametrize("size", ["big", "1024", "0x"])
    def test_unreadable_size_is_refused(self, size: str) -> None:
        with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
            dimensions_for(size, None, default=(512, 512), longest=2048)

    def test_size_past_the_longest_side_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at most 1024"):
            dimensions_for("2048x2048", None, default=(512, 512), longest=1024)

    def test_unreadable_aspect_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a shape"):
            dimensions_for(None, "sideways", default=(512, 512), longest=2048)


# -- OpenAI ---------------------------------------------------------------------


class TestOpenAIOptions:
    """OpenAI takes a size it makes and an aspect as its nearest size; it takes no seed."""

    async def test_nothing_asked_sends_what_it_sent_before(self, serve: Any) -> None:
        service = serve("openai", _openai)
        result = await OpenAIImageProvider("sk-test").generate("a cat")
        assert service.json_body() == {
            "model": "dall-e-3",
            "prompt": "a cat",
            "size": "1024x1024",
            "quality": "standard",
            "response_format": "b64_json",
            "n": 1,
        }
        assert result.metadata is not None
        assert GENERATION_IGNORED not in result.metadata
        assert result.metadata[GENERATION_SIZE] == "1024x1024"

    async def test_size_is_sent(self, serve: Any) -> None:
        service = serve("openai", _openai)
        result = await OpenAIImageProvider("sk-test").generate("a cat", size="1024x1792")
        assert service.json_body()["size"] == "1024x1792"
        assert (result.width, result.height) == (1024, 1792)
        assert result.metadata is not None
        assert result.metadata[GENERATION_SIZE] == "1024x1792"

    async def test_size_the_model_does_not_make_is_refused_and_not_sent(self, serve: Any) -> None:
        service = serve("openai", _openai)
        with pytest.raises(ValueError, match="1792x1024"):
            await OpenAIImageProvider("sk-test").generate("a cat", size="800x600")
        assert service.requests == []

    async def test_aspect_picks_the_nearest_size(self, serve: Any) -> None:
        service = serve("openai", _openai)
        await OpenAIImageProvider("sk-test").generate("a cat", aspect="landscape")
        assert service.json_body()["size"] == "1792x1024"

    async def test_aspect_on_an_unknown_model_is_named_and_not_applied(self, serve: Any) -> None:
        service = serve("openai", _openai)
        result = await OpenAIImageProvider("sk-test", model_name="some-new-model").generate("a cat", aspect="portrait")
        assert service.json_body()["size"] == "1024x1024"
        assert result.metadata is not None
        assert set(result.metadata.get(GENERATION_IGNORED) or {}) == {"aspect"}

    async def test_seed_is_named_and_not_sent(self, serve: Any) -> None:
        service = serve("openai", _openai)
        result = await OpenAIImageProvider("sk-test").generate("a cat", seed=7)
        assert "seed" not in service.json_body()
        assert result.metadata is not None
        assert result.metadata.get(GENERATION_IGNORED) == {"seed": "OpenAI's image models take no seed"}
        assert GENERATION_SEED not in result.metadata

    async def test_gpt_image_takes_no_style_and_no_response_format(self, serve: Any) -> None:
        service = serve("openai", _openai)
        result = await OpenAIImageProvider("sk-test", model_name="gpt-image-1").generate(
            "a cat", style="vivid", size="1536x1024"
        )
        body = service.json_body()
        assert "style" not in body
        assert "response_format" not in body
        assert body["quality"] == "auto"
        assert body["size"] == "1536x1024"
        assert result.metadata is not None
        assert set(result.metadata.get(GENERATION_IGNORED) or {}) == {"style"}

    async def test_edit_carries_the_size_and_says_the_source_was_used(self, serve: Any) -> None:
        service = serve("openai", _openai)
        result = await OpenAIImageProvider("sk-test", model_name="gpt-image-1").generate(
            "make it blue", source_image=b"src", source_mime_type="image/png", aspect="portrait"
        )
        request = service.requests[0]
        assert request.url.path.endswith("/images/edits")
        assert _multipart_field(request, "size") == "1024x1536"
        assert _multipart_field(request, "response_format") is None
        assert result.metadata is not None
        assert result.metadata[GENERATION_SOURCE_USED] is True


# -- HuggingFace ----------------------------------------------------------------


class TestHuggingFaceOptions:
    """HuggingFace sends width, height and seed as task parameters, only when asked."""

    async def test_nothing_asked_sends_what_it_sent_before(self, serve: Any) -> None:
        service = serve("huggingface", _bytes)
        result = await HuggingFaceImageProvider("hf-test").generate("a dog")
        assert service.json_body() == {"inputs": "a dog"}
        assert result.metadata is not None
        assert GENERATION_IGNORED not in result.metadata
        assert GENERATION_SIZE not in result.metadata

    async def test_size_and_seed_are_sent(self, serve: Any) -> None:
        service = serve("huggingface", _bytes)
        result = await HuggingFaceImageProvider("hf-test").generate("a dog", size="768x1344", seed=42)
        assert service.json_body()["parameters"] == {"width": 768, "height": 1344, "seed": 42}
        assert result.metadata is not None
        assert result.metadata[GENERATION_SIZE] == "768x1344"
        assert result.metadata[GENERATION_SEED] == 42

    async def test_aspect_is_sent_as_width_and_height(self, serve: Any) -> None:
        service = serve("huggingface", _bytes)
        await HuggingFaceImageProvider("hf-test").generate("a dog", aspect="wide")
        assert service.json_body()["parameters"] == {"width": 1024, "height": 576}

    async def test_a_source_is_said_unused(self, serve: Any) -> None:
        serve("huggingface", _bytes)
        provider = HuggingFaceImageProvider("hf-test")
        result = await provider.generate("a dog", source_image=b"src")
        assert provider.takes_source_image is False
        assert result.metadata is not None
        assert result.metadata[GENERATION_SOURCE_USED] is False


# -- ModelsLab ------------------------------------------------------------------


class TestModelsLabOptions:
    """ModelsLab sends width and height always and a seed only when asked."""

    async def test_nothing_asked_sends_what_it_sent_before(self, serve: Any) -> None:
        service = serve("modelslab", _modelslab())
        result = await ModelsLabImageProvider("ml-test").generate("a sunset")
        assert service.json_body() == {
            "key": "ml-test",
            "model_id": "midjourney",
            "prompt": "a sunset",
            "samples": 1,
            "negative_prompt": "",
            "width": 512,
            "height": 512,
        }
        assert result.metadata is not None
        assert GENERATION_IGNORED not in result.metadata
        assert GENERATION_SEED not in result.metadata

    async def test_size_and_seed_are_sent_and_the_reported_seed_is_said(self, serve: Any) -> None:
        service = serve("modelslab", _modelslab(meta={"seed": 99}))
        result = await ModelsLabImageProvider("ml-test").generate("a sunset", size="768x1024", seed=99)
        body = service.json_body()
        assert (body.get("width"), body.get("height"), body.get("seed")) == (768, 1024, 99)
        assert result.metadata is not None
        assert result.metadata[GENERATION_SIZE] == "768x1024"
        assert result.metadata[GENERATION_SEED] == 99

    async def test_the_seed_the_service_drew_is_said(self, serve: Any) -> None:
        serve("modelslab", _modelslab(meta={"seed": 123456}))
        result = await ModelsLabImageProvider("ml-test").generate("a sunset")
        assert result.metadata is not None
        assert result.metadata[GENERATION_SEED] == 123456

    async def test_aspect_is_sent_as_width_and_height(self, serve: Any) -> None:
        service = serve("modelslab", _modelslab())
        await ModelsLabImageProvider("ml-test").generate("a sunset", aspect="landscape")
        body = service.json_body()
        assert (body.get("width"), body.get("height")) == (512, 320)

    async def test_size_past_its_largest_is_refused_and_not_sent(self, serve: Any) -> None:
        service = serve("modelslab", _modelslab())
        with pytest.raises(ValueError, match="at most 1024"):
            await ModelsLabImageProvider("ml-test").generate("a sunset", size="2048x2048")
        assert service.requests == []


# -- ComfyUI --------------------------------------------------------------------


class TestComfyUIOptions:
    """ComfyUI sets the latent size and a seed per image, never seed 0 for all."""

    async def test_nothing_asked_sends_the_default_workflow_with_a_drawn_seed(self, serve: Any) -> None:
        service = serve("comfyui", _comfyui)
        with patch.object(comfyui_module.secrets, "randbelow", return_value=4242):
            result = await ComfyUIImageProvider(poll_interval=0.0).generate("a mountain")
        # the workflow sent before size and seed existed, written out, with only the seed now drawn
        expected = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 4242,
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
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a mountain", "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
        }
        assert service.json_body() == {"prompt": expected}
        assert result.metadata is not None
        assert result.metadata[GENERATION_SEED] == 4242
        assert GENERATION_IGNORED not in result.metadata

    async def test_two_images_get_two_seeds(self, serve: Any) -> None:
        service = serve("comfyui", _comfyui)
        provider = ComfyUIImageProvider(poll_interval=0.0)
        await provider.generate("a mountain")
        await provider.generate("a mountain")
        posts = [json.loads(r.content) for r in service.requests if r.method == "POST"]
        seeds = [p["prompt"]["3"]["inputs"]["seed"] for p in posts]
        assert seeds[0] != seeds[1]
        assert 0 not in seeds

    async def test_size_and_seed_are_sent(self, serve: Any) -> None:
        service = serve("comfyui", _comfyui)
        result = await ComfyUIImageProvider(poll_interval=0.0).generate("a mountain", size="1024x768", seed=7)
        workflow = service.json_body()["prompt"]
        assert (workflow["5"]["inputs"]["width"], workflow["5"]["inputs"]["height"]) == (1024, 768)
        assert workflow["3"]["inputs"]["seed"] == 7
        assert result.metadata is not None
        assert result.metadata[GENERATION_SIZE] == "1024x768"
        assert result.metadata[GENERATION_SEED] == 7

    async def test_aspect_is_sent_as_the_latent_size(self, serve: Any) -> None:
        service = serve("comfyui", _comfyui)
        await ComfyUIImageProvider(poll_interval=0.0).generate("a mountain", aspect="tall")
        latent = service.json_body()["prompt"]["5"]["inputs"]
        assert (latent["width"], latent["height"]) == (256, 512)

    async def test_a_seed_the_server_refuses_is_named_and_not_sent(self, serve: Any) -> None:
        service = serve("comfyui", _comfyui)
        with patch.object(comfyui_module.secrets, "randbelow", return_value=31):
            result = await ComfyUIImageProvider(poll_interval=0.0).generate("a mountain", seed=-5)
        assert service.json_body()["prompt"]["3"]["inputs"]["seed"] == 31
        assert result.metadata is not None
        assert set(result.metadata.get(GENERATION_IGNORED) or {}) == {"seed"}
        assert result.metadata[GENERATION_SEED] == 31


# -- A1111 ----------------------------------------------------------------------


class TestA1111Options:
    """A1111 sends width and height always and a seed only when asked, on both endpoints."""

    async def test_nothing_asked_sends_what_it_sent_before(self, serve: Any) -> None:
        service = serve("a1111", _a1111())
        result = await A1111ImageProvider().generate("a landscape")
        assert service.json_body() == {
            "prompt": "a landscape",
            "negative_prompt": "",
            "steps": 20,
            "cfg_scale": 7.0,
            "sampler_name": "Euler",
            "width": 512,
            "height": 512,
            "batch_size": 1,
        }
        assert result.metadata is not None
        assert GENERATION_IGNORED not in result.metadata
        assert GENERATION_SEED not in result.metadata

    async def test_size_and_seed_are_sent(self, serve: Any) -> None:
        service = serve("a1111", _a1111(info=json.dumps({"seed": 5})))
        result = await A1111ImageProvider().generate("a landscape", size="640x896", seed=5)
        body = service.json_body()
        assert (body.get("width"), body.get("height"), body.get("seed")) == (640, 896, 5)
        assert (result.width, result.height) == (640, 896)
        assert result.metadata is not None
        assert result.metadata[GENERATION_SIZE] == "640x896"
        assert result.metadata[GENERATION_SEED] == 5

    async def test_the_seed_the_server_drew_is_said(self, serve: Any) -> None:
        serve("a1111", _a1111(info=json.dumps({"seed": 31337})))
        result = await A1111ImageProvider().generate("a landscape")
        assert result.metadata is not None
        assert result.metadata[GENERATION_SEED] == 31337

    async def test_unreadable_info_falls_back_to_the_seed_asked(self, serve: Any) -> None:
        serve("a1111", _a1111(info="not json"))
        result = await A1111ImageProvider().generate("a landscape", seed=8)
        assert result.metadata is not None
        assert result.metadata[GENERATION_SEED] == 8

    async def test_img2img_carries_size_seed_and_uses_the_source(self, serve: Any) -> None:
        service = serve("a1111", _a1111())
        result = await A1111ImageProvider().generate("make it blue", source_image=b"src", aspect="landscape", seed=3)
        request = service.requests[0]
        body = service.json_body()
        assert request.url.path == "/sdapi/v1/img2img"
        assert (body.get("width"), body.get("height"), body.get("seed")) == (512, 320, 3)
        assert result.metadata is not None
        assert result.metadata[GENERATION_SOURCE_USED] is True


# -- through the tool -------------------------------------------------------------


class _Host:
    """a host that keeps the image and answers with one line."""

    def __init__(self) -> None:
        self.kept: list[GeneratedImage] = []

    async def load_attached_image(self) -> tuple[bytes, str] | None:
        return None

    async def persist_generated(self, image: GeneratedImage, *, prompt: str, model_name: str, style: str | None) -> str:
        self.kept.append(image)
        return f"Made an image with {model_name}."


class TestThroughTheTool:
    """the tool hands the options to a provider and reads back what it could not apply."""

    async def test_a_seed_openai_cannot_take_is_said_not_applied(self, serve: Any) -> None:
        service = serve("openai", _openai)
        tool = create_image_generation_tool({"dall-e": OpenAIImageProvider("sk-test")}, _Host(), default_model="dall-e")
        reply = await tool.ainvoke({"prompt": "a cat", "seed": 7, "size": "1024x1792"})
        assert "Not applied: seed (OpenAI's image models take no seed)." in reply
        assert service.json_body()["size"] == "1024x1792"
        assert "seed" not in service.json_body()

    async def test_options_a_provider_takes_are_applied_without_a_note(self, serve: Any) -> None:
        service = serve("huggingface", _bytes)
        host = _Host()
        tool = create_image_generation_tool({"sdxl": HuggingFaceImageProvider("hf-test")}, host, default_model="sdxl")
        reply = await tool.ainvoke({"prompt": "a dog", "seed": 7, "aspect": "square"})
        assert "Not applied" not in reply
        assert service.json_body()["parameters"] == {"width": 1024, "height": 1024, "seed": 7}
        assert host.kept[0].metadata is not None
        assert host.kept[0].metadata[GENERATION_SEED] == 7
