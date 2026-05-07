"""Analyze media tool — vision analysis, document QA, and audio/video transcription.

This is a builtin 3tears tool that any host application can use by providing
protocol implementations for storage, vision, and transcription.

Config keys (passed via the tool registry ``config`` dict):

    storage           MediaStorage  — required
    analyzers         dict mapping analyzer display names to AnalyzerConfig
    user_id           UUID | None   — for content provenance
    media_url_fn      callable(str) -> str | None — builds a display URL
                      from a media_id string (e.g. for inline image hints)
    on_analysis       async callable(media_id_str, content_type, text)
                      — optional callback after any analysis result is stored
    response_suffix   str | None — appended to prompts sent to providers
                      (default: "Respond using markdown formatting.")
    doc_max_chars     int — max chars of extracted text sent for document QA
                      (default: 12000)
    transcript_max_chars  int — max transcript chars returned (default: 10000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import UUID

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.protocols import (
    MediaInfo,
    MediaStorage,
    TextProvider,
    TranscriptionProvider,
    VisionProvider,
)
from threetears.observe import get_logger

__all__ = [
    "AnalyzeMediaTool",
    "AnalyzerConfig",
    "MediaAnalysisInput",
    "OnAnalysisCallback",
    "create_analyze_media_tool",
]

_log = get_logger(__name__)

_DEFAULT_RESPONSE_SUFFIX = "Respond using markdown formatting."
_DEFAULT_DOC_MAX_CHARS = 12_000
_DEFAULT_TRANSCRIPT_MAX_CHARS = 10_000


@dataclass
class AnalyzerConfig:
    """Configuration for a single named analyzer.

    Each analyzer wraps a provider (vision, transcription, or both)
    and declares which media categories it can handle.
    """

    name: str
    vision: VisionProvider | None = None
    text: TextProvider | None = None
    transcription: TranscriptionProvider | None = None
    supported_categories: set[str] = field(
        default_factory=lambda: {"image"},
    )


# Type alias for the on_analysis callback: (media_id_str, content_type, text)
OnAnalysisCallback = Callable[[str, str, str], Awaitable[None]]


def _tool_error(step: str, detail: str) -> str:
    return f"[analyze_media/{step}] Error: {detail}"


class MediaAnalysisInput(BaseModel):
    """structured-args schema for the analyze_media tool.

    promoted to module level so the StructuredTool factory and any
    test fixture can reference it without going through the factory.
    """

    media_ids: list[str] = Field(
        description="List of media UUID strings to analyze",
    )
    question: str = Field(
        description=(
            "What to ask about the media, e.g. 'Describe this image' "
            "or 'What is said in this audio?'"
        ),
    )
    analyzer: str = Field(
        description=(
            "Display name of the analysis model to use. "
            "Use a vision model for images/documents, "
            "or an STT model for audio/video transcription."
        ),
    )


def create_analyze_media_tool(
    config: dict[str, Any],
    description: str,
) -> StructuredTool:
    """Factory: create an analyze_media tool from protocol implementations.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the StructuredTool path and the NATS-dispatched ToolServer
    path share :meth:`AnalyzeMediaTool._analyze` as their single
    execution body. previously the factory carried an inline
    350-line ``_analyze_media`` closure that
    :meth:`AnalyzeMediaTool.execute` called BACK into via
    ``ainvoke`` (an inverted dual-codepath that prevented the same
    unification the other 7 builtins received in v0.6.x); the
    helper closures are now methods on :class:`AnalyzeMediaTool`
    and the factory is a thin construction wrapper.

    appends the configured analyzer names to the tool description so
    the LLM sees which analyzers are available; keeps the
    description-with-analyzer-list shape that callers already depend on.

    Expected ``config`` keys:

    - ``storage`` — :class:`MediaStorage` implementation (**required**)
    - ``analyzers`` — ``dict[str, AnalyzerConfig]`` of named analyzers
    - ``user_id`` — ``UUID | None`` for content provenance
    - ``media_url_fn`` — ``(str) -> str | None``, builds display URL from media_id
    - ``on_analysis`` — ``async (str, str, str) -> None`` callback
      called as ``(media_id_str, content_type, text)`` after any result is stored
    - ``response_suffix`` — appended to provider prompts (default: markdown instruction)
    - ``doc_max_chars`` — max extracted text chars for document QA (default: 12000)
    - ``transcript_max_chars`` — max transcript chars in response (default: 10000)

    :param config: provider/storage configuration dictionary
    :ptype config: dict[str, Any]
    :param description: base description used for the LangChain tool surface
    :ptype description: str
    :return: LangChain StructuredTool wrapping the AnalyzeMediaTool instance
    :rtype: StructuredTool
    :raises TypeError: if ``storage`` is missing from *config*
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    storage: MediaStorage | None = config.get("storage")
    if storage is None:
        raise TypeError("create_analyze_media_tool requires 'storage' (MediaStorage) in config")

    analyzers_dict: dict[str, AnalyzerConfig] = config.get("analyzers", {})

    tool = AnalyzeMediaTool(
        storage=storage,
        analyzers=analyzers_dict,
        user_id=config.get("user_id"),
        media_url_fn=config.get("media_url_fn"),
        on_analysis=config.get("on_analysis"),
        response_suffix=config.get("response_suffix", _DEFAULT_RESPONSE_SUFFIX),
        doc_max_chars=config.get("doc_max_chars", _DEFAULT_DOC_MAX_CHARS),
        transcript_max_chars=config.get(
            "transcript_max_chars",
            _DEFAULT_TRANSCRIPT_MAX_CHARS,
        ),
    )

    full_description = description
    if analyzers_dict:
        analyzer_list = ", ".join(analyzers_dict.keys())
        full_description = f"{description} Available analyzers: {analyzer_list}"

    return to_langchain_tool(
        tool,
        description=full_description,
        args_schema=MediaAnalysisInput,
    )


class AnalyzeMediaTool(TearsTool):
    """TearsTool wrapper for media analysis via vision/transcription providers.

    analyzes images, documents, audio, and video using configurable
    analyzer backends. requires MediaStorage and AnalyzerConfig
    instances to be provided at construction time. the per-category
    routing logic (document QA, audio/video transcription, vision
    analysis, cached-description fast path) lives in
    :meth:`_analyze` and is invoked by both the LangChain
    StructuredTool path (via :func:`create_analyze_media_tool` →
    :func:`to_langchain_tool` → :meth:`execute`) and the
    NATS-dispatched ToolServer path (via :meth:`execute` directly).

    :param storage: media storage implementation for accessing media items
    :ptype storage: MediaStorage
    :param analyzers: mapping of analyzer display names to AnalyzerConfig
    :ptype analyzers: dict[str, AnalyzerConfig]
    :param user_id: optional user UUID for content provenance
    :ptype user_id: UUID | None
    :param media_url_fn: optional callable to build display URLs from media IDs
    :ptype media_url_fn: Callable[[str], str | None] | None
    :param on_analysis: optional async callback after analysis completes
    :ptype on_analysis: OnAnalysisCallback | None
    :param response_suffix: suffix appended to provider prompts
    :ptype response_suffix: str
    :param doc_max_chars: max chars of extracted text for document QA
    :ptype doc_max_chars: int
    :param transcript_max_chars: max transcript chars returned
    :ptype transcript_max_chars: int
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "media_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "list of media UUID strings to analyze",
            },
            "question": {
                "type": "string",
                "description": "what to ask about media (e.g. 'Describe this image')",
            },
            "analyzer": {
                "type": "string",
                "description": "display name of analysis model to use",
            },
        },
        "required": ["media_ids", "question", "analyzer"],
    }

    def __init__(
        self,
        storage: MediaStorage,
        analyzers: dict[str, AnalyzerConfig] | None = None,
        user_id: UUID | None = None,
        media_url_fn: Callable[[str], str | None] | None = None,
        on_analysis: OnAnalysisCallback | None = None,
        response_suffix: str = _DEFAULT_RESPONSE_SUFFIX,
        doc_max_chars: int = _DEFAULT_DOC_MAX_CHARS,
        transcript_max_chars: int = _DEFAULT_TRANSCRIPT_MAX_CHARS,
    ) -> None:
        """initialize analyze media tool with provider dependencies.

        :param storage: media storage implementation
        :ptype storage: MediaStorage
        :param analyzers: mapping of analyzer names to AnalyzerConfig
        :ptype analyzers: dict[str, AnalyzerConfig] | None
        :param user_id: optional user UUID for content provenance
        :ptype user_id: UUID | None
        :param media_url_fn: optional callable to build display URLs
        :ptype media_url_fn: Callable[[str], str | None] | None
        :param on_analysis: optional async callback after analysis
        :ptype on_analysis: OnAnalysisCallback | None
        :param response_suffix: suffix appended to provider prompts
        :ptype response_suffix: str
        :param doc_max_chars: max chars for document QA
        :ptype doc_max_chars: int
        :param transcript_max_chars: max transcript chars returned
        :ptype transcript_max_chars: int
        """
        self._storage = storage
        self._analyzers = analyzers or {}
        self._user_id = user_id
        self._media_url_fn = media_url_fn
        self._on_analysis = on_analysis
        self._response_suffix = response_suffix
        self._doc_max_chars = doc_max_chars
        self._transcript_max_chars = transcript_max_chars

    async def _analyze(
        self,
        media_ids: list[str],
        question: str,
        analyzer: str,
    ) -> str:
        """resolve analyzer and route media items through the per-category handlers.

        document items go through :meth:`_handle_document`,
        audio/video items through :meth:`_handle_audio_video`, and
        images through :meth:`_handle_vision`. single-media calls
        consult the cached description first to avoid redundant
        provider hits.

        :param media_ids: list of media UUID strings to analyze
        :ptype media_ids: list[str]
        :param question: prompt to send to the resolved analyzer
        :ptype question: str
        :param analyzer: display name of the analyzer to invoke
        :ptype analyzer: str
        :return: analysis text or formatted error string
        :rtype: str
        """
        _log.debug(
            "analyze_media invoked",
            extra={
                "extra_data": {
                    "media_ids": media_ids,
                    "question": question[:100],
                    "analyzer": analyzer,
                }
            },
        )

        acfg = self._analyzers.get(analyzer)
        if acfg is None:
            available = ", ".join(self._analyzers.keys())
            return _tool_error(
                "resolve analyzer",
                f"Unknown analyzer '{analyzer}'. Available: {available}",
            )

        # Pre-fetch media info for all IDs (avoids redundant storage calls)
        media_info: dict[str, MediaInfo] = {}
        for mid_str in media_ids:
            info = await self._storage.get_media(UUID(mid_str))
            if info:
                media_info[mid_str] = info

        # --- Document routing (use extracted text, not vision) ---
        for mid_str in media_ids:
            info = media_info.get(mid_str)
            if info and info.media_category == "document":
                return await self._handle_document(
                    UUID(mid_str),
                    mid_str,
                    info,
                    acfg,
                    question,
                )

        # --- Capability check ---
        for mid_str in media_ids:
            info = media_info.get(mid_str)
            if info and info.media_category not in acfg.supported_categories:
                return _tool_error(
                    "capability check",
                    f"The analyzer '{analyzer}' doesn't support "
                    f"{info.media_category} files. Capabilities: "
                    f"{', '.join(sorted(acfg.supported_categories))}.",
                )

        # --- Audio/video transcription routing ---
        if acfg.transcription:
            for mid_str in media_ids:
                info = media_info.get(mid_str)
                if info and info.media_category in ("audio", "video"):
                    return await self._handle_audio_video(
                        UUID(mid_str),
                        mid_str,
                        info,
                        acfg,
                        question,
                    )

        # --- Cached description check (single-media) ---
        if len(media_ids) == 1:
            cached = await self._storage.get_content(
                UUID(media_ids[0]),
                "description",
                model_name=analyzer,
            )
            if cached:
                _log.debug(
                    "Returning cached description",
                    extra={
                        "extra_data": {
                            "media_id": media_ids[0],
                            "model": analyzer,
                        }
                    },
                )
                return cached

        # --- Vision analysis ---
        if not acfg.vision:
            return _tool_error(
                "vision",
                f"Analyzer '{analyzer}' has no vision capability.",
            )

        return await self._handle_vision(media_ids, acfg, question, analyzer)

    async def _fire_callback(
        self,
        mid_str: str,
        content_type: str,
        text: str,
    ) -> None:
        """invoke the on_analysis callback, swallowing errors.

        :param mid_str: media UUID string
        :ptype mid_str: str
        :param content_type: callback content type ("description" or "transcript")
        :ptype content_type: str
        :param text: callback payload text
        :ptype text: str
        :return: None
        :rtype: None
        """
        if self._on_analysis:
            try:
                await self._on_analysis(mid_str, content_type, text)
            except Exception:
                pass

    async def _handle_document(
        self,
        mid: UUID,
        mid_str: str,
        info: MediaInfo,
        acfg: AnalyzerConfig,
        question: str,
    ) -> str:
        """route a document item through extracted-text question answering.

        :param mid: media UUID
        :ptype mid: UUID
        :param mid_str: media UUID string for logging/callbacks
        :ptype mid_str: str
        :param info: MediaInfo for the media item
        :ptype info: MediaInfo
        :param acfg: resolved AnalyzerConfig (must carry a text provider)
        :ptype acfg: AnalyzerConfig
        :param question: prompt to send to the text provider
        :ptype question: str
        :return: provider response or formatted error string
        :rtype: str
        """
        if info.extraction_status == "pending":
            return (
                "This document is still being processed "
                "(text extraction in progress). Please wait a moment "
                "and try again, or respond based on what you already know."
            )

        extracted = await self._storage.get_content(mid, "extracted_text")
        if not extracted:
            extracted = await self._storage.get_content(mid, "transcript")
        if not extracted:
            return _tool_error(
                "document analysis",
                "No text could be extracted from this document.",
            )

        # Need a text provider for document QA
        text_provider = acfg.text
        if text_provider is None:
            return _tool_error(
                "document analysis",
                f"Analyzer '{acfg.name}' has no text QA capability.",
            )

        truncated = extracted[: self._doc_max_chars]
        suffix = f"\n\n{self._response_suffix}" if self._response_suffix else ""
        doc_prompt = (
            f"{question}\n\n"
            f"--- DOCUMENT TEXT ---\n{truncated}"
            f"{' [truncated]' if len(extracted) > self._doc_max_chars else ''}"
            f"{suffix}"
        )

        try:
            result_text = await text_provider.answer(doc_prompt)
        except Exception as exc:
            _log.error(
                "Document analysis failed",
                extra={
                    "extra_data": {
                        "analyzer": acfg.name,
                        "error": str(exc),
                    }
                },
            )
            return _tool_error("document analysis", str(exc))

        if result_text and self._user_id:
            try:
                await self._storage.store_content(
                    mid,
                    self._user_id,
                    "description",
                    result_text,
                    metadata={"model_name": acfg.name},
                )
            except Exception as exc:
                _log.warning(
                    "Failed to persist document description",
                    extra={
                        "extra_data": {
                            "media_id": mid_str,
                            "error": str(exc),
                        }
                    },
                )

            await self._fire_callback(mid_str, "description", result_text)

        return result_text or _tool_error(
            "document analysis",
            "Model returned empty response.",
        )

    async def _handle_audio_video(
        self,
        mid: UUID,
        mid_str: str,
        info: MediaInfo,
        acfg: AnalyzerConfig,
        question: str,
    ) -> str:
        """route an audio/video item through transcription + optional description.

        :param mid: media UUID
        :ptype mid: UUID
        :param mid_str: media UUID string for logging/callbacks
        :ptype mid_str: str
        :param info: MediaInfo for the media item
        :ptype info: MediaInfo
        :param acfg: resolved AnalyzerConfig (must carry a transcription provider)
        :ptype acfg: AnalyzerConfig
        :param question: prompt (currently informational; transcript is returned)
        :ptype question: str
        :return: transcript (and any prior description) or error string
        :rtype: str
        """
        assert acfg.transcription is not None

        # Check for cached transcript
        cached_transcript = None
        if info.extraction_status == "complete":
            cached_transcript = await self._storage.get_content(mid, "transcript")

        if cached_transcript:
            transcript = cached_transcript
        else:
            # Download and transcribe
            dl = await self._storage.download_media(mid)
            if dl is None:
                return _tool_error(
                    "transcribe",
                    "Could not download media from storage.",
                )
            data, mime_type = dl

            try:
                transcript = await acfg.transcription.transcribe(
                    data,
                    mime_type,
                )
            except Exception as exc:
                _log.error(
                    "Transcription failed",
                    extra={
                        "extra_data": {
                            "media_id": mid_str,
                            "error": str(exc),
                        }
                    },
                )
                return _tool_error("transcription", str(exc))

            # Store transcript
            if self._user_id:
                try:
                    await self._storage.store_content(
                        mid,
                        self._user_id,
                        "transcript",
                        transcript,
                        metadata={"model_name": acfg.name},
                    )
                except Exception as exc:
                    _log.warning(
                        "Failed to persist transcript",
                        extra={
                            "extra_data": {
                                "media_id": mid_str,
                                "error": str(exc),
                            }
                        },
                    )

                await self._fire_callback(mid_str, "transcript", transcript)

        # Fetch any existing description
        description = await self._storage.get_content(mid, "description")

        parts = [
            f"**Transcript** ({info.media_category}):\n{transcript[: self._transcript_max_chars]}",
        ]
        if description:
            parts.append(f"\n\n**Analysis:**\n{description}")

        return "\n".join(parts)

    async def _handle_vision(
        self,
        media_ids: list[str],
        acfg: AnalyzerConfig,
        question: str,
        analyzer_name: str,
    ) -> str:
        """route image media through vision analysis.

        :param media_ids: list of media UUID strings to analyze together
        :ptype media_ids: list[str]
        :param acfg: resolved AnalyzerConfig (must carry a vision provider)
        :ptype acfg: AnalyzerConfig
        :param question: prompt for the vision model
        :ptype question: str
        :param analyzer_name: display name of the analyzer (for caching/storage)
        :ptype analyzer_name: str
        :return: vision response (with optional display-url hint) or error string
        :rtype: str
        """
        assert acfg.vision is not None

        from threetears.agent.tools.builtin.image_prep import (
            prepare_image_for_vision,
        )

        # Download and preprocess all images
        image_parts: list[tuple[bytes, str]] = []
        for mid_str in media_ids:
            mid = UUID(mid_str)
            dl = await self._storage.download_media(mid)
            if dl is None:
                _log.warning(
                    "Media not found for analysis",
                    extra={"extra_data": {"media_id": mid_str}},
                )
                continue

            data, mime_type = dl
            # Preprocess — resize large images, re-encode HEIC, etc.
            processed_data, processed_mime = prepare_image_for_vision(
                data,
                mime_type,
            )
            image_parts.append((processed_data, processed_mime))

        if not image_parts:
            return _tool_error(
                "load media",
                "No valid media found for the given media IDs.",
            )

        suffix = f"\n\n{self._response_suffix}" if self._response_suffix else ""
        prompt = f"{question}{suffix}"

        try:
            if len(image_parts) == 1:
                result_text = await acfg.vision.analyze(
                    image_parts[0][0],
                    image_parts[0][1],
                    prompt,
                )
            else:
                # For multi-image, analyze each separately
                results = []
                for img_data, img_mime in image_parts:
                    r = await acfg.vision.analyze(img_data, img_mime, prompt)
                    results.append(r)
                result_text = "\n\n---\n\n".join(results)
        except Exception as exc:
            _log.error(
                "Vision model invocation failed",
                extra={
                    "extra_data": {
                        "analyzer": analyzer_name,
                        "error": str(exc),
                    }
                },
            )
            return _tool_error("vision model invocation", str(exc))

        # Persist description for single-media analysis
        if result_text and len(media_ids) == 1 and self._user_id:
            try:
                await self._storage.store_content(
                    UUID(media_ids[0]),
                    self._user_id,
                    "description",
                    result_text,
                    metadata={"model_name": analyzer_name},
                )
            except Exception as exc:
                _log.warning(
                    "Failed to persist description",
                    extra={"extra_data": {"error": str(exc)}},
                )

            await self._fire_callback(media_ids[0], "description", result_text)

        # Include display hint if the host app provides a URL builder
        if len(media_ids) == 1 and self._media_url_fn:
            url = self._media_url_fn(media_ids[0])
            if url:
                return f"{result_text}\n\nTo display this image in your response, use: ![description]({url})"
        return result_text

    async def execute(self, **kwargs: Any) -> ToolResult:
        """analyze media items using configured providers.

        :param kwargs: must include 'media_ids', 'question', 'analyzer' keys
        :ptype kwargs: Any
        :return: result containing analysis text or error
        :rtype: ToolResult
        """
        media_ids = kwargs.get("media_ids", [])
        question = kwargs.get("question", "")
        analyzer = kwargs.get("analyzer", "")
        content = await self._analyze(media_ids, question, analyzer)
        success = not content.startswith("[analyze_media/")
        result = ToolResult(
            success=success,
            content=content,
            error=content if not success else None,
        )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for analyze media.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="analyze images, documents, audio, and video using vision/transcription providers",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.analyze_media"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
