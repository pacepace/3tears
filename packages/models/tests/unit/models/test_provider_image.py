"""tests for image generation provider adapters."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threetears.agent.tools.protocols import GeneratedImage, ImageGenerationBackend
from threetears.models.providers.image.a1111 import A1111ImageProvider
from threetears.models.providers.image.comfyui import ComfyUIImageProvider
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
        assert provider._api_key == "sk-test"
        assert provider._model_name == "dall-e-3"
        assert provider._base_url == "https://api.openai.com/v1"
        assert provider._size == "1024x1024"
        assert provider._quality == "standard"
        assert provider._timeout == 120

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

        submit_response = _mock_json_response({
            "status": "success",
            "output": ["https://example.com/image.png"],
        })
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

        submit_response = _mock_json_response({
            "status": "success",
            "output": ["https://example.com/image.png"],
        })
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

        submit_response = _mock_json_response({
            "status": "processing",
            "id": "req-123",
            "fetch_result": "https://modelslab.com/api/v6/fetch/req-123",
        })
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

        submit_response = _mock_json_response({
            "status": "processing",
            "id": "req-123",
            "fetch_result": "https://modelslab.com/api/v6/fetch/req-123",
        })
        poll_processing = _mock_json_response({"status": "processing"})
        poll_success = _mock_json_response({
            "status": "success",
            "output": ["https://example.com/result.png"],
        })
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

        submit_response = _mock_json_response({"prompt_id": "pid-001"})
        history_response = _mock_json_response({
            "pid-001": {
                "outputs": {
                    "9": {
                        "images": [{"filename": "ComfyUI_00001_.png"}],
                    },
                },
            },
        })
        image_response = _mock_bytes_response(b"comfy-image-data")

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_response
        mock_client.get.side_effect = [history_response, image_response]

        with patch("threetears.models.providers.image.comfyui.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            result = await provider.generate("a mountain")

        assert isinstance(result, GeneratedImage)
        assert result.data == b"comfy-image-data"
        assert result.mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_generate_with_style(self) -> None:
        """style parameter is prepended to prompt in workflow."""
        provider = ComfyUIImageProvider(poll_interval=0.01)

        submit_response = _mock_json_response({"prompt_id": "pid-002"})
        history_response = _mock_json_response({
            "pid-002": {
                "outputs": {
                    "9": {
                        "images": [{"filename": "ComfyUI_00002_.png"}],
                    },
                },
            },
        })
        image_response = _mock_bytes_response(b"comfy-styled")

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_response
        mock_client.get.side_effect = [history_response, image_response]

        with patch("threetears.models.providers.image.comfyui.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.generate("a mountain", style="cyberpunk")

        call_kwargs = mock_client.post.call_args
        workflow = call_kwargs.kwargs["json"]["prompt"]
        assert workflow["6"]["inputs"]["text"] == "cyberpunk style, a mountain"

    @pytest.mark.asyncio
    async def test_timeout_on_max_polls(self) -> None:
        """raises TimeoutError after max poll attempts exceeded."""
        provider = ComfyUIImageProvider(max_polls=2, poll_interval=0.01)

        submit_response = _mock_json_response({"prompt_id": "pid-003"})
        empty_history = _mock_json_response({})

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_response
        mock_client.get.return_value = empty_history

        with patch("threetears.models.providers.image.comfyui.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            with pytest.raises(TimeoutError):
                await provider.generate("a mountain")

    @pytest.mark.asyncio
    async def test_poll_succeeds_on_second_try(self) -> None:
        """succeeds after initial empty history followed by completed history."""
        provider = ComfyUIImageProvider(poll_interval=0.01)

        submit_response = _mock_json_response({"prompt_id": "pid-004"})
        empty_history = _mock_json_response({})
        completed_history = _mock_json_response({
            "pid-004": {
                "outputs": {
                    "9": {
                        "images": [{"filename": "ComfyUI_00004_.png"}],
                    },
                },
            },
        })
        image_response = _mock_bytes_response(b"delayed-image")

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_response
        mock_client.get.side_effect = [empty_history, completed_history, image_response]

        with patch("threetears.models.providers.image.comfyui.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            result = await provider.generate("a mountain")

        assert isinstance(result, GeneratedImage)
        assert result.data == b"delayed-image"
