"""tests for image generation provider adapters."""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from threetears.agent.tools.protocols import GeneratedImage, ImageGenerationBackend
from threetears.models.providers.image.a1111 import A1111ImageProvider
from threetears.models.providers.image.comfyui import ComfyUIClient, ComfyUIImageProvider
from threetears.models.providers.image.huggingface import HuggingFaceImageProvider
from threetears.models.providers.image.modelslab import ModelsLabImageProvider
from threetears.models.providers.image.openai import OpenAIImageProvider


def _mock_json_response(
    json_data: dict[str, Any],
    status_code: int = 200,
) -> MagicMock:
    """creates mock httpx response returning JSON data.

    :param json_data: JSON data to return from response.json()
    :ptype json_data: dict[str, Any]
    :param status_code: HTTP status code
    :ptype status_code: int
    :return: mock response object
    :rtype: MagicMock
    """
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status = MagicMock()
    return response


def _mock_bytes_response(
    content: bytes,
    status_code: int = 200,
) -> MagicMock:
    """creates mock httpx response returning raw bytes.

    :param content: raw bytes to return from response.content
    :ptype content: bytes
    :param status_code: HTTP status code
    :ptype status_code: int
    :return: mock response object
    :rtype: MagicMock
    """
    response = MagicMock()
    response.status_code = status_code
    response.content = content
    response.raise_for_status = MagicMock()
    return response


# -- OpenAI ------------------------------------------------------------------


class TestOpenAIImageProvider:
    """tests for OpenAIImageProvider class."""

    def test_satisfies_image_generation_protocol(self) -> None:
        """OpenAIImageProvider instance satisfies ImageGenerationBackend protocol."""
        provider = OpenAIImageProvider("sk-test")
        assert isinstance(provider, ImageGenerationBackend)

    def test_default_config(self) -> None:
        """default configuration values are set correctly."""
        provider = OpenAIImageProvider("sk-test")
        assert provider.model_name == "dall-e-3"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.size == "1024x1024"
        assert provider.quality == "standard"
        assert provider.timeout == 120

    @pytest.mark.asyncio
    async def test_api_key_sent_in_authorization_header(self) -> None:
        """api_key is forwarded as Bearer token in Authorization header of request."""
        provider = OpenAIImageProvider("sk-custom")
        image_b64 = base64.b64encode(b"fake-png-data").decode()
        json_data = {"data": [{"b64_json": image_b64}]}

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_json_response(json_data)

        with patch("threetears.models.providers.image.openai.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.generate("a cat")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-custom"

    @pytest.mark.asyncio
    async def test_generate_returns_generated_image(self) -> None:
        """generate returns GeneratedImage from mocked txt2img API response."""
        provider = OpenAIImageProvider("sk-test")
        image_b64 = base64.b64encode(b"fake-png-data").decode()
        json_data = {"data": [{"b64_json": image_b64}]}

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_json_response(json_data)

        with patch("threetears.models.providers.image.openai.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            result = await provider.generate("a cat")

        assert isinstance(result, GeneratedImage)
        assert result.data == b"fake-png-data"
        assert result.mime_type == "image/png"
        assert result.width == 1024
        assert result.height == 1024

    @pytest.mark.asyncio
    async def test_generate_with_style(self) -> None:
        """style parameter is included in txt2img request payload."""
        provider = OpenAIImageProvider("sk-test")
        image_b64 = base64.b64encode(b"fake-png-data").decode()
        json_data = {"data": [{"b64_json": image_b64}]}

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_json_response(json_data)

        with patch("threetears.models.providers.image.openai.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.generate("a cat", style="vivid")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["json"]["style"] == "vivid"

    @pytest.mark.asyncio
    async def test_img2img_uses_edits_endpoint(self) -> None:
        """source_image triggers /images/edits endpoint instead of /images/generations."""
        provider = OpenAIImageProvider("sk-test")
        image_b64 = base64.b64encode(b"fake-png-data").decode()
        json_data = {"data": [{"b64_json": image_b64}]}

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_json_response(json_data)

        with patch("threetears.models.providers.image.openai.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.generate(
                "edit this",
                source_image=b"source-png",
                source_mime_type="image/png",
            )

        call_args = mock_client.post.call_args
        url = call_args.args[0]
        assert url.endswith("/images/edits")


# -- HuggingFace -------------------------------------------------------------


class TestHuggingFaceImageProvider:
    """tests for HuggingFaceImageProvider class."""

    def test_satisfies_image_generation_protocol(self) -> None:
        """HuggingFaceImageProvider instance satisfies ImageGenerationBackend protocol."""
        provider = HuggingFaceImageProvider("hf-test")
        assert isinstance(provider, ImageGenerationBackend)

    @pytest.mark.asyncio
    async def test_generate_returns_generated_image(self) -> None:
        """generate returns GeneratedImage with raw bytes from API."""
        provider = HuggingFaceImageProvider("hf-test")

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_bytes_response(b"fake-image-bytes")

        with patch("threetears.models.providers.image.huggingface.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            result = await provider.generate("a dog")

        assert isinstance(result, GeneratedImage)
        assert result.data == b"fake-image-bytes"
        assert result.mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_generate_with_style(self) -> None:
        """style parameter is prepended to prompt."""
        provider = HuggingFaceImageProvider("hf-test")

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_bytes_response(b"fake-image-bytes")

        with patch("threetears.models.providers.image.huggingface.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.generate("a dog", style="watercolor")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["json"]["inputs"] == "watercolor style, a dog"


# -- A1111 --------------------------------------------------------------------


class TestA1111ImageProvider:
    """tests for A1111ImageProvider class."""

    def test_satisfies_image_generation_protocol(self) -> None:
        """A1111ImageProvider instance satisfies ImageGenerationBackend protocol."""
        provider = A1111ImageProvider()
        assert isinstance(provider, ImageGenerationBackend)

    @pytest.mark.asyncio
    async def test_generate_returns_generated_image(self) -> None:
        """generate returns GeneratedImage from mocked txt2img API response."""
        provider = A1111ImageProvider()
        image_b64 = base64.b64encode(b"fake-sd-image").decode()
        json_data = {"images": [image_b64]}

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_json_response(json_data)

        with patch("threetears.models.providers.image.a1111.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            result = await provider.generate("a landscape")

        assert isinstance(result, GeneratedImage)
        assert result.data == b"fake-sd-image"
        assert result.mime_type == "image/png"
        assert result.width == 512
        assert result.height == 512

    @pytest.mark.asyncio
    async def test_generate_with_style(self) -> None:
        """style parameter is prepended to prompt."""
        provider = A1111ImageProvider()
        image_b64 = base64.b64encode(b"fake-sd-image").decode()
        json_data = {"images": [image_b64]}

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_json_response(json_data)

        with patch("threetears.models.providers.image.a1111.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.generate("a landscape", style="oil painting")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["json"]["prompt"] == "oil painting style, a landscape"

    @pytest.mark.asyncio
    async def test_img2img_uses_img2img_endpoint(self) -> None:
        """source_image triggers /sdapi/v1/img2img endpoint."""
        provider = A1111ImageProvider()
        image_b64 = base64.b64encode(b"fake-sd-image").decode()
        json_data = {"images": [image_b64]}

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_json_response(json_data)

        with patch("threetears.models.providers.image.a1111.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.generate("edit this", source_image=b"source-png")

        call_args = mock_client.post.call_args
        url = call_args.args[0]
        assert url.endswith("/sdapi/v1/img2img")
        assert "init_images" in call_args.kwargs["json"]


# -- ModelsLab ---------------------------------------------------------------


class TestModelsLabImageProvider:
    """tests for ModelsLabImageProvider class."""

    def test_satisfies_image_generation_protocol(self) -> None:
        """ModelsLabImageProvider instance satisfies ImageGenerationBackend protocol."""
        provider = ModelsLabImageProvider("ml-test")
        assert isinstance(provider, ImageGenerationBackend)

    @pytest.mark.asyncio
    async def test_generate_returns_generated_image(self) -> None:
        """generate returns GeneratedImage when API returns immediate success."""
        provider = ModelsLabImageProvider("ml-test")

        submit_response = _mock_json_response(
            {
                "status": "success",
                "output": ["https://example.com/image.png"],
            }
        )
        image_response = _mock_bytes_response(b"fake-ml-image")

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_response
        mock_client.get.return_value = image_response

        with patch("threetears.models.providers.image.modelslab.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            result = await provider.generate("a sunset")

        assert isinstance(result, GeneratedImage)
        assert result.data == b"fake-ml-image"
        assert result.mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_generate_with_style(self) -> None:
        """style parameter is prepended to prompt in request."""
        provider = ModelsLabImageProvider("ml-test")

        submit_response = _mock_json_response(
            {
                "status": "success",
                "output": ["https://example.com/image.png"],
            }
        )
        image_response = _mock_bytes_response(b"fake-ml-image")

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_response
        mock_client.get.return_value = image_response

        with patch("threetears.models.providers.image.modelslab.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.generate("a sunset", style="anime")

        call_kwargs = mock_client.post.call_args_list[0]
        assert call_kwargs.kwargs["json"]["prompt"] == "anime style, a sunset"

    @pytest.mark.asyncio
    async def test_timeout_on_max_polls(self) -> None:
        """raises TimeoutError after max poll attempts exceeded."""
        provider = ModelsLabImageProvider("ml-test", max_polls=2, poll_interval=0.01)

        submit_response = _mock_json_response(
            {
                "status": "processing",
                "id": "req-123",
                "fetch_result": "https://modelslab.com/api/v6/fetch/req-123",
            }
        )
        poll_response = _mock_json_response({"status": "processing"})

        mock_client = AsyncMock()
        mock_client.post.side_effect = [submit_response, poll_response, poll_response]
        mock_client.get.return_value = _mock_bytes_response(b"")

        with patch("threetears.models.providers.image.modelslab.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            with pytest.raises(TimeoutError):
                await provider.generate("a sunset")

    @pytest.mark.asyncio
    async def test_poll_succeeds_on_second_try(self) -> None:
        """succeeds after initial processing status followed by success on poll."""
        provider = ModelsLabImageProvider("ml-test", poll_interval=0.01)

        submit_response = _mock_json_response(
            {
                "status": "processing",
                "id": "req-123",
                "fetch_result": "https://modelslab.com/api/v6/fetch/req-123",
            }
        )
        poll_processing = _mock_json_response({"status": "processing"})
        poll_success = _mock_json_response(
            {
                "status": "success",
                "output": ["https://example.com/result.png"],
            }
        )
        image_response = _mock_bytes_response(b"final-image")

        mock_client = AsyncMock()
        mock_client.post.side_effect = [submit_response, poll_processing, poll_success]
        mock_client.get.return_value = image_response

        with patch("threetears.models.providers.image.modelslab.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            result = await provider.generate("a sunset")

        assert isinstance(result, GeneratedImage)
        assert result.data == b"final-image"


# -- ComfyUI ------------------------------------------------------------------


def _comfy_server(
    prompt_id: str, histories: list[dict[str, Any]], picture: bytes = b""
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """a scripted ComfyUI: queues as ``prompt_id``, answers history in order (the last repeats).

    :param prompt_id: the id the queued graph gets
    :ptype prompt_id: str
    :param histories: the history answers, in order
    :ptype histories: list[dict[str, Any]]
    :param picture: what ``/view`` answers
    :ptype picture: bytes
    :return: the transport and the requests it saw
    :rtype: tuple[httpx.MockTransport, list[httpx.Request]]
    """
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": prompt_id})
        if request.url.path == "/view":
            return httpx.Response(200, content=picture)
        return httpx.Response(200, json=histories.pop(0) if len(histories) > 1 else histories[0])

    return httpx.MockTransport(handle), seen


def _on(transport: httpx.MockTransport) -> Any:
    """patch the provider's ComfyUIClient onto a scripted transport.

    :param transport: the scripted server
    :ptype transport: httpx.MockTransport
    :return: the patch, for ``with``
    :rtype: Any
    """

    class _Scripted(ComfyUIClient):
        def __init__(self, base_url: str, **kwargs: Any) -> None:
            super().__init__(base_url, transport=transport, **kwargs)

    return patch("threetears.models.providers.image.comfyui.ComfyUIClient", _Scripted)


def _done(prompt_id: str, filename: str) -> dict[str, Any]:
    """a finished graph's history answer with one picture.

    :param prompt_id: the graph's id
    :ptype prompt_id: str
    :param filename: the picture's name
    :ptype filename: str
    :return: the answer
    :rtype: dict[str, Any]
    """
    return {prompt_id: {"outputs": {"9": {"images": [{"filename": filename}]}}}}


class TestComfyUIImageProvider:
    """tests for ComfyUIImageProvider class."""

    def test_satisfies_image_generation_protocol(self) -> None:
        """ComfyUIImageProvider instance satisfies ImageGenerationBackend protocol."""
        provider = ComfyUIImageProvider()
        assert isinstance(provider, ImageGenerationBackend)

    @pytest.mark.asyncio
    async def test_generate_returns_generated_image(self) -> None:
        """generate returns GeneratedImage after workflow submission and polling."""
        provider = ComfyUIImageProvider(poll_interval=0.01)
        transport, seen = _comfy_server("pid-001", [_done("pid-001", "ComfyUI_00001_.png")], b"comfy-image-data")

        with _on(transport):
            result = await provider.generate("a mountain")

        assert isinstance(result, GeneratedImage)
        assert result.data == b"comfy-image-data"
        assert result.mime_type == "image/png"
        assert dict(seen[-1].url.params) == {"filename": "ComfyUI_00001_.png"}

    @pytest.mark.asyncio
    async def test_generate_with_style(self) -> None:
        """style parameter is prepended to prompt in workflow."""
        provider = ComfyUIImageProvider(poll_interval=0.01)
        transport, seen = _comfy_server("pid-002", [_done("pid-002", "ComfyUI_00002_.png")], b"comfy-styled")

        with _on(transport):
            await provider.generate("a mountain", style="cyberpunk")

        workflow = json.loads(seen[0].content)["prompt"]
        assert workflow["6"]["inputs"]["text"] == "cyberpunk style, a mountain"

    @pytest.mark.asyncio
    async def test_timeout_on_max_polls(self) -> None:
        """raises TimeoutError after max poll attempts exceeded."""
        provider = ComfyUIImageProvider(max_polls=2, poll_interval=0.01)
        transport, _seen = _comfy_server("pid-003", [{}])

        with _on(transport), pytest.raises(TimeoutError):
            await provider.generate("a mountain")

    @pytest.mark.asyncio
    async def test_poll_succeeds_on_second_try(self) -> None:
        """succeeds after initial empty history followed by completed history."""
        provider = ComfyUIImageProvider(poll_interval=0.01)
        transport, _seen = _comfy_server("pid-004", [{}, _done("pid-004", "ComfyUI_00004_.png")], b"delayed-image")

        with _on(transport):
            result = await provider.generate("a mountain")

        assert isinstance(result, GeneratedImage)
        assert result.data == b"delayed-image"
