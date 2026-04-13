"""whisper transcription provider using OpenAI-compatible audio API via httpx."""

from __future__ import annotations

from typing import Any

import httpx

from threetears.models.results import TranscriptionResult, TranscriptionSegment


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
    """transcription provider for OpenAI Whisper API via httpx.

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
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

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
            "model": self._model_name,
            "response_format": "verbose_json",
        }
        if language_hint is not None:
            data["language"] = language_hint

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/audio/transcriptions",
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
