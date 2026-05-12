"""whisper transcription provider using OpenAI-compatible audio API via httpx.

Whisper has no LangChain ``BaseChatModel`` analog (it's audio
transcription, not chat). This module keeps a regular async class that
sends multipart audio to the OpenAI-compatible ``audio/transcriptions``
endpoint and parses the ``verbose_json`` response into the result types
defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType

__all__ = [
    "TranscriptionResult",
    "TranscriptionSegment",
    "WHISPER_PROVIDER_NAME",
    "WhisperTranscriptionProvider",
]


WHISPER_PROVIDER_NAME = "whisper"


@dataclass
class TranscriptionSegment:
    """time-aligned segment within transcription result.

    :param start: segment start time in seconds
    :ptype start: float
    :param end: segment end time in seconds
    :ptype end: float
    :param text: transcribed text for segment
    :ptype text: str
    """

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """result from audio transcription provider.

    :param text: full transcription text
    :ptype text: str
    :param language: detected or specified language code
    :ptype language: str | None
    :param duration_seconds: audio duration in seconds
    :ptype duration_seconds: float | None
    :param segments: time-aligned transcript segments
    :ptype segments: list[TranscriptionSegment] | None
    """

    text: str
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[TranscriptionSegment] | None = None


# MIME type to file extension mapping for multipart upload filename
_MIME_TO_EXT: dict[str, str] = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
}


class WhisperTranscriptionProvider:
    """transcription provider for OpenAI-compatible Whisper API via httpx.

    sends multipart audio data to Whisper endpoint and parses
    verbose_json response into TranscriptionResult with segments.
    supports self-hosted Whisper endpoints via configurable base_url.

    :param api_key: API key for authentication
    :ptype api_key: str
    :param model_name: Whisper model identifier
    :ptype model_name: str
    :param base_url: API base URL (supports self-hosted endpoints)
    :ptype base_url: str
    :param timeout: request timeout in seconds
    :ptype timeout: int
    """

    def __init__(
        self,
        api_key: str,
        *,
        model_name: str = "whisper-1",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def transcribe(
        self,
        audio_data: bytes,
        mime_type: str,
        *,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        """transcribes audio data to text via Whisper API.

        sends multipart form upload to Whisper endpoint, requesting
        verbose_json response format for segment-level detail.

        :param audio_data: raw audio bytes
        :ptype audio_data: bytes
        :param mime_type: MIME type of audio data
        :ptype mime_type: str
        :param language_hint: optional language code hint for transcription
        :ptype language_hint: str | None
        :return: transcription result with text and optional segments
        :rtype: TranscriptionResult
        :raises httpx.HTTPStatusError: if API returns error status code
        """
        ext = _MIME_TO_EXT.get(mime_type, "bin")
        filename = f"audio.{ext}"

        data: dict[str, str] = {
            "model": self.model_name,
            "response_format": "verbose_json",
        }
        if language_hint is not None:
            data["language"] = language_hint

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": (filename, audio_data, mime_type)},
                data=data,
            )
            response.raise_for_status()

        result = _parse_verbose_json(response.json())
        return result


def _parse_verbose_json(body: dict[str, Any]) -> TranscriptionResult:
    """parses Whisper verbose_json response into TranscriptionResult.

    :param body: parsed JSON response from Whisper API
    :ptype body: dict[str, Any]
    :return: transcription result with text, language, duration, and segments
    :rtype: TranscriptionResult
    """
    segments = None
    raw_segments = body.get("segments")
    if raw_segments:
        segments = [
            TranscriptionSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=seg.get("text", ""),
            )
            for seg in raw_segments
        ]

    return TranscriptionResult(
        text=body.get("text", ""),
        language=body.get("language"),
        duration_seconds=body.get("duration"),
        segments=segments,
    )


# -- capability registration -------------------------------------------------

_WHISPER_CAPABILITIES: dict[str, ModelCapabilities] = {
    "whisper-1": ModelCapabilities(
        model_name="whisper-1",
        provider_name=WHISPER_PROVIDER_NAME,
        model_type=ModelType.TRANSCRIPTION,
        model_tier=ModelTier.MEDIUM,
        model_status=ModelStatus.ACTIVE,
        supported_audio_formats=[
            "audio/mpeg",
            "audio/mp3",
            "audio/wav",
            "audio/wave",
            "audio/x-wav",
            "audio/mp4",
            "audio/x-m4a",
            "audio/webm",
            "audio/ogg",
            "audio/flac",
        ],
        max_audio_duration_seconds=2_700.0,
        supports_language_hint=True,
    ),
}


for _model_id, _caps in _WHISPER_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
