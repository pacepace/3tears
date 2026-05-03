"""tests for WhisperTranscriptionProvider adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from threetears.models.providers.whisper import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperTranscriptionProvider,
    _MIME_TO_EXT,
    _parse_verbose_json,
)


def _mock_response(
    json_data: dict[str, object],
    status_code: int = 200,
) -> MagicMock:
    """creates mock httpx response for testing.

    :param json_data: JSON data to return from response.json()
    :ptype json_data: dict[str, object]
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


class TestWhisperTranscriptionProvider:
    """tests for WhisperTranscriptionProvider class."""

    def test_has_transcribe_method(self) -> None:
        """WhisperTranscriptionProvider exposes the async ``transcribe`` method."""
        provider = WhisperTranscriptionProvider("sk-test")
        assert callable(provider.transcribe)

    def test_default_model_name(self) -> None:
        """default model_name is whisper-1."""
        provider = WhisperTranscriptionProvider("sk-test")
        assert provider.model_name == "whisper-1"

    def test_default_base_url(self) -> None:
        """default base_url is OpenAI API v1 endpoint."""
        provider = WhisperTranscriptionProvider("sk-test")
        assert provider.base_url == "https://api.openai.com/v1"

    def test_default_timeout(self) -> None:
        """default timeout is 120 seconds."""
        provider = WhisperTranscriptionProvider("sk-test")
        assert provider.timeout == 120

    def test_custom_config(self) -> None:
        """custom model_name, base_url, and timeout are stored correctly."""
        provider = WhisperTranscriptionProvider(
            "sk-custom",
            model_name="whisper-large-v3",
            base_url="https://custom.api.com/v1",
            timeout=60,
        )
        assert provider.model_name == "whisper-large-v3"
        assert provider.base_url == "https://custom.api.com/v1"
        assert provider.timeout == 60

    @pytest.mark.asyncio
    async def test_api_key_sent_in_authorization_header(self) -> None:
        """api_key is forwarded as Bearer token in Authorization header of request."""
        provider = WhisperTranscriptionProvider("sk-custom")

        json_data = {"text": "hello", "language": "en", "duration": 1.0}

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(json_data)

        with patch("threetears.models.providers.whisper.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.transcribe(b"audio", "audio/wav")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-custom"

    def test_base_url_trailing_slash_stripped(self) -> None:
        """trailing slash on base_url is stripped during init."""
        provider = WhisperTranscriptionProvider(
            "sk-test",
            base_url="https://api.com/v1/",
        )
        assert provider.base_url == "https://api.com/v1"

    @pytest.mark.asyncio
    async def test_transcribe_returns_result(self) -> None:
        """transcribe returns TranscriptionResult from mocked API response."""
        provider = WhisperTranscriptionProvider("sk-test")

        json_data = {
            "text": "Hello world",
            "language": "en",
            "duration": 5.0,
            "segments": [],
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(json_data)

        with patch("threetears.models.providers.whisper.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            result = await provider.transcribe(b"audio-bytes", "audio/wav")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration_seconds == 5.0

    @pytest.mark.asyncio
    async def test_transcribe_with_segments(self) -> None:
        """transcribe parses segments from API response into TranscriptionSegment list."""
        provider = WhisperTranscriptionProvider("sk-test")

        json_data = {
            "text": "Hello world",
            "language": "en",
            "duration": 5.0,
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello"},
                {"start": 2.5, "end": 5.0, "text": " world"},
            ],
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(json_data)

        with patch("threetears.models.providers.whisper.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            result = await provider.transcribe(b"audio-bytes", "audio/mp3")

        assert result.segments is not None
        assert len(result.segments) == 2
        assert isinstance(result.segments[0], TranscriptionSegment)
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 2.5
        assert result.segments[0].text == "Hello"
        assert result.segments[1].start == 2.5
        assert result.segments[1].end == 5.0
        assert result.segments[1].text == " world"

    @pytest.mark.asyncio
    async def test_transcribe_with_language_hint(self) -> None:
        """language_hint is included in request data when provided."""
        provider = WhisperTranscriptionProvider("sk-test")

        json_data = {"text": "Bonjour", "language": "fr", "duration": 1.0}

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(json_data)

        with patch("threetears.models.providers.whisper.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.transcribe(b"audio", "audio/wav", language_hint="fr")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["data"]["language"] == "fr"

    @pytest.mark.asyncio
    async def test_transcribe_without_language_hint(self) -> None:
        """language key is absent from request data when no hint provided."""
        provider = WhisperTranscriptionProvider("sk-test")

        json_data = {"text": "Hello", "language": "en", "duration": 1.0}

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(json_data)

        with patch("threetears.models.providers.whisper.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            await provider.transcribe(b"audio", "audio/wav")

        call_kwargs = mock_client.post.call_args
        assert "language" not in call_kwargs.kwargs["data"]

    @pytest.mark.asyncio
    async def test_transcribe_raises_on_error_status(self) -> None:
        """transcribe propagates HTTPStatusError from failed API call."""
        provider = WhisperTranscriptionProvider("sk-test")

        error_response = _mock_response({"error": "bad request"}, status_code=400)
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=MagicMock(),
            response=error_response,
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = error_response

        with patch("threetears.models.providers.whisper.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = False
            with pytest.raises(httpx.HTTPStatusError):
                await provider.transcribe(b"audio", "audio/wav")


class TestParseVerboseJson:
    """tests for _parse_verbose_json helper function."""

    def test_basic_response(self) -> None:
        """parses text, language, and duration from response body."""
        body = {
            "text": "Hello world",
            "language": "en",
            "duration": 10.5,
        }
        result = _parse_verbose_json(body)

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration_seconds == 10.5
        assert result.segments is None

    def test_with_segments(self) -> None:
        """parses segments list into TranscriptionSegment objects."""
        body = {
            "text": "Hello world",
            "language": "en",
            "duration": 5.0,
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 2.0, "end": 5.0, "text": " world"},
            ],
        }
        result = _parse_verbose_json(body)

        assert result.segments is not None
        assert len(result.segments) == 2
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 2.0
        assert result.segments[0].text == "Hello"
        assert result.segments[1].start == 2.0
        assert result.segments[1].end == 5.0
        assert result.segments[1].text == " world"

    def test_missing_fields(self) -> None:
        """missing fields default to None or empty string."""
        body: dict[str, object] = {}
        result = _parse_verbose_json(body)

        assert result.text == ""
        assert result.language is None
        assert result.duration_seconds is None
        assert result.segments is None

    def test_empty_segments(self) -> None:
        """empty segments list produces None segments field."""
        body = {
            "text": "Hello",
            "language": "en",
            "duration": 1.0,
            "segments": [],
        }
        result = _parse_verbose_json(body)

        assert result.segments is None


class TestMimeToExt:
    """tests for _MIME_TO_EXT mapping."""

    @pytest.mark.parametrize(
        ("mime_type", "expected_ext"),
        [
            ("audio/mpeg", "mp3"),
            ("audio/mp3", "mp3"),
            ("audio/wav", "wav"),
            ("audio/wave", "wav"),
            ("audio/x-wav", "wav"),
            ("audio/mp4", "m4a"),
            ("audio/x-m4a", "m4a"),
            ("audio/webm", "webm"),
            ("audio/ogg", "ogg"),
            ("audio/flac", "flac"),
        ],
    )
    def test_known_mime_types(self, mime_type: str, expected_ext: str) -> None:
        """known MIME type maps to correct file extension."""
        assert _MIME_TO_EXT[mime_type] == expected_ext

    def test_unknown_mime_type(self) -> None:
        """unknown MIME type falls back to bin extension."""
        assert _MIME_TO_EXT.get("audio/unknown", "bin") == "bin"
