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
    markdown          bool — ask for and label results in markdown (default: True)
    response_suffix   str | None — appended to prompts sent to providers
                      (default: the markdown ask, or plain sentences when
                      markdown is off)
    doc_max_chars     int — max chars of extracted text sent for document QA
                      (default: 12000)
    transcript_max_chars  int — max transcript chars returned (default: 10000)
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import UUID

from langchain_core.tools import StructuredTool
from threetears.media.contracts import (
    EXTRACTION_STATUS_COMPLETE,
    EXTRACTION_STATUS_PENDING,
    MediaSizeLimitExceeded,
)

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.document import DocumentParseError, can_parse_document, parse_document
from threetears.agent.tools.protocols import (
    MediaInfo,
    MediaStorage,
    ReferenceVisionProvider,
    TextProvider,
    TranscriptionProvider,
    VisionProvider,
)
from threetears.agent.tools.text_window import find_in_text, window_text
from threetears.langgraph.fence import explained_fence, mint_nonce
from threetears.observe import contain, get_logger

__all__ = [
    "DESCRIBE_PROMPT",
    "MAX_DOCUMENT_BYTES",
    "MAX_TRANSCRIPTION_BYTES",
    "MAX_VISION_IMAGE_BYTES",
    "AnalyzeMediaTool",
    "AnalyzerConfig",
    "OnAnalysisCallback",
    "create_analyze_media_tool",
]

_log = get_logger(__name__)

#: The default ask: markdown, for a host that renders it.
_DEFAULT_RESPONSE_SUFFIX = "Respond using markdown formatting."
#: The ask when a host turns markdown off: the answer goes back to a model that
#: speaks to a person, and headings and bold in what it reads come back out in
#: what it says.
_PLAIN_RESPONSE_SUFFIX = "Answer in plain sentences."
_DEFAULT_DOC_MAX_CHARS = 12_000
_DEFAULT_TRANSCRIPT_MAX_CHARS = 10_000

#: What the analyser is asked when the caller asks nothing: the describe ask. Only its
#: answer is the item's description -- served from the store when one is kept, and
#: kept when one is made. An answer to any question the caller wrote is that
#: question's alone: kept as the description, "what colour is the car?" came back
#: as the answer to the next caller's "describe this", and to every later one.
DESCRIBE_PROMPT = (
    "Describe this in full: what it shows or says, any text in it word for word, and anything notable about it."
)

#: The largest document this tool downloads to read its text: 20 MiB. A
#: document with no cached extraction is read from its own bytes, and every
#: parser needs the whole file in memory at once -- in a tool pod whose memory
#: every other call it is serving shares. The model is sent at most
#: ``doc_max_chars`` of the text anyway, so a document past this size cannot be
#: answered any better by reading all of it, and reading it is what takes the
#: pod down. 20 MiB holds any ordinary report, contract or text file and is the
#: same bound the model gateway puts on a referenced object. A document the
#: catalog records as larger is refused without a byte moving; the read itself
#: stops at this bound too, because a recorded size can be absent or wrong.
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024

#: The largest audio or video file this tool downloads to transcribe: 100 MiB.
#: A transcription backend that takes bytes is handed the whole recording in
#: one buffer, and the HTTP client sending it builds a second copy for the
#: upload, so the pod holds about twice this while the call runs -- beside
#: every other call it is serving. Recordings are legitimately larger than
#: documents, so this is five times the document bound: 100 MiB is well over
#: an hour of speech at ordinary compressed bitrates and over the 25 MB
#: upload cap the Whisper API puts on a single request, so nothing a
#: transcription backend would accept is refused here. What it stops is a
#: catalogued multi-gigabyte video, which no backend would transcribe in one
#: call, being buffered whole into a tool pod. Refused from its recorded size
#: without a byte moving; the read itself stops at this bound too.
MAX_TRANSCRIPTION_BYTES = 100 * 1024 * 1024

#: The largest image this tool downloads for a bytes-taking vision backend:
#: 20 MiB, the same bound the model gateway puts on a referenced image
#: (``aibots.gateway.media.MAX_MEDIA_BYTES``). An image is downloaded whole and
#: decoded before :func:`prepare_image_for_vision` resizes it, so the decoded
#: pixels are a multiple of the file. Keeping the bytes path at the gateway's
#: bound means an image is answerable the same way whichever vision backend
#: an analyzer uses. Refused from its recorded size without a byte moving; the
#: read itself stops at this bound too.
MAX_VISION_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass
class AnalyzerConfig:
    """Configuration for a single named analyzer.

    Each analyzer wraps a provider (vision, transcription, or both)
    and declares which media categories it can handle.
    """

    name: str
    vision: VisionProvider | ReferenceVisionProvider | None = None
    text: TextProvider | None = None
    transcription: TranscriptionProvider | None = None
    supported_categories: set[str] = field(
        default_factory=lambda: {"image"},
    )


# Type alias for the on_analysis callback: (media_id_str, content_type, text)
OnAnalysisCallback = Callable[[str, str, str], Awaitable[None]]


def _tool_error(step: str, detail: str) -> str:
    return f"[analyze_media/{step}] Error: {detail}"


def _is_uuid(value: str) -> bool:
    """whether a model-written media id parses as one.

    :param value: the id as the model wrote it
    :ptype value: str
    :return: ``True`` when it is a UUID
    :rtype: bool
    """
    try:
        UUID(str(value))
    except ValueError:
        return False
    return True


def _too_large(kind: str, size_bytes: int | None, limit_bytes: int, ask: str) -> str:
    """what the model is told about media over the size bound for its kind.

    :param kind: what the media is, as the model reads it (``"document"``, ``"image"``)
    :ptype kind: str
    :param size_bytes: the media's size when known, or ``None`` when the read passed the limit
    :ptype size_bytes: int | None
    :param limit_bytes: the bound it passed
    :ptype limit_bytes: int
    :param ask: what the model can ask for instead
    :ptype ask: str
    :return: the sentence
    :rtype: str
    """
    size = f"it is {size_bytes:,} bytes" if size_bytes is not None else "it passed that size while being read"
    return f"This {kind} is too large to read (too_large): {size}, and {kind}s over {limit_bytes:,} bytes are not read. {ask}"


def _refused_too_large(
    kind: str,
    mid_str: str,
    size_bytes: int | None,
    limit_bytes: int,
    ask: str,
) -> str:
    """log a refused read and return what the model is told about it.

    one place for the refusal, so every byte-taking path logs the same fields and
    gives the same plain answer: before a download when the recorded size is over
    the bound (``size_bytes`` set), or when a bounded read passed it (``None``).

    :param kind: what the media is, as the model reads it
    :ptype kind: str
    :param mid_str: media UUID string for logging
    :ptype mid_str: str
    :param size_bytes: the recorded size, or ``None`` when the read passed the limit
    :ptype size_bytes: int | None
    :param limit_bytes: the bound it passed
    :ptype limit_bytes: int
    :param ask: what the model can ask for instead
    :ptype ask: str
    :return: the sentence for the tool error
    :rtype: str
    """
    when = "its recorded size is over" if size_bytes is not None else "the read passed"
    _log.warning(
        f"{kind} not read: {when} the {kind} size limit",
        extra={
            "extra_data": {
                "media_id": mid_str,
                "kind": kind,
                "size_bytes": size_bytes,
                "limit_bytes": limit_bytes,
            }
        },
    )
    return _too_large(kind, size_bytes, limit_bytes, ask)


_DOCUMENT_ASK = "Ask for a smaller document or a specific excerpt."
_RECORDING_ASK = "Ask for a shorter recording or a specific excerpt."
_IMAGE_ASK = "Ask for a smaller image."


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
    - ``markdown`` — ``bool``, markdown in the ask and the result labels (default: True)
    - ``response_suffix`` — appended to provider prompts (default: follows ``markdown``)
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
        markdown=config.get("markdown", True),
        response_suffix=config.get("response_suffix"),
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
                "description": (
                    "What to ask, e.g. 'what colour is the car?'. A question gets a new look and its answer is "
                    "not kept. Leave out for the item's description."
                ),
            },
            "analyzer": {
                "type": "string",
                "description": "analysis model to use; omit it to use the first one that reads this media",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Document only: where to start reading, in characters.",
            },
            "find": {
                "type": "string",
                "description": "Document only: a word or phrase; the reading opens just before its first match.",
            },
        },
        "required": ["media_ids"],
    }

    def __init__(
        self,
        storage: MediaStorage,
        analyzers: dict[str, AnalyzerConfig] | None = None,
        user_id: UUID | None = None,
        media_url_fn: Callable[[str], str | None] | None = None,
        on_analysis: OnAnalysisCallback | None = None,
        response_suffix: str | None = None,
        doc_max_chars: int = _DEFAULT_DOC_MAX_CHARS,
        transcript_max_chars: int = _DEFAULT_TRANSCRIPT_MAX_CHARS,
        markdown: bool = True,
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
        :param response_suffix: suffix appended to provider prompts; ``None``
            asks for markdown, or for plain sentences when ``markdown`` is off
        :ptype response_suffix: str | None
        :param doc_max_chars: max chars for document QA
        :ptype doc_max_chars: int
        :param transcript_max_chars: max transcript chars returned
        :ptype transcript_max_chars: int
        :param markdown: markdown in the ask and in the result's labels
        :ptype markdown: bool
        """
        self._storage = storage
        self._analyzers = analyzers or {}
        self._user_id = user_id
        self._media_url_fn = media_url_fn
        self._on_analysis = on_analysis
        self._markdown = markdown
        if response_suffix is None:
            response_suffix = _DEFAULT_RESPONSE_SUFFIX if markdown else _PLAIN_RESPONSE_SUFFIX
        self._response_suffix = response_suffix
        self._doc_max_chars = doc_max_chars
        self._transcript_max_chars = transcript_max_chars

    async def _analyze(
        self,
        media_ids: list[str],
        question: str,
        analyzer: str | None,
        *,
        offset: int = 0,
        find: str | None = None,
    ) -> str:
        """resolve analyzer and route media items through the per-category handlers.

        a named analyzer is used exactly, and an unknown name is refused with the
        choices: it is never swapped for another. an omitted one is the first
        registered analyzer that reads every resolved item (:meth:`_default_analyzer`),
        so a model that is not told a name -- or is told the names and picks none --
        is answered on its first call.

        document items go through :meth:`_handle_document`,
        audio/video items through :meth:`_handle_audio_video`, and
        images through :meth:`_handle_vision`.

        a blank ``question`` is the describe ask: a single item's kept description is
        served, and a new one is kept. a question is always a new look, and its
        answer is never kept as the description (:data:`DESCRIBE_PROMPT`).

        :param media_ids: list of media UUID strings to analyze
        :ptype media_ids: list[str]
        :param question: what the caller asked, or ``""`` for the describe ask
        :ptype question: str
        :param analyzer: display name of the analyzer to invoke, or ``None`` for
            the first one that reads the media
        :ptype analyzer: str | None
        :param offset: for a document, where to start reading, in characters
        :ptype offset: int
        :param find: for a document, a word or phrase whose first match the read opens at
        :ptype find: str | None
        :return: analysis text or formatted error string
        :rtype: str
        """
        asked = question.strip()
        _log.debug(
            "analyze_media invoked",
            extra={
                "extra_data": {
                    "media_ids": media_ids,
                    "question": asked[:100],
                    "describe": not asked,
                    "analyzer": analyzer,
                }
            },
        )
        if not media_ids:
            return _tool_error("load media", "No media ids were given: pass the id of each item to look at.")
        unread = [m for m in media_ids if not _is_uuid(m)]
        if unread:
            return _tool_error(
                "load media",
                f"Not media ids: {', '.join(repr(m) for m in unread[:5])}. A media id is the id in a [media:<id>] "
                "line or the ID of an item in the media list.",
            )

        if analyzer is not None and analyzer not in self._analyzers:
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

        if analyzer is None:
            categories = sorted({info.media_category for info in media_info.values()})
            analyzer = self._default_analyzer(categories)
            if analyzer is None:
                return _tool_error("resolve analyzer", self._no_default_analyzer(categories))
            _log.debug(
                "analyze_media analyzer omitted; using the first that reads the media",
                extra={"extra_data": {"analyzer": analyzer, "categories": categories}},
            )
        acfg = self._analyzers[analyzer]

        # --- Document routing (use extracted text, not vision) ---
        for mid_str in media_ids:
            info = media_info.get(mid_str)
            if info and info.media_category == "document":
                return await self._handle_document(
                    UUID(mid_str),
                    mid_str,
                    info,
                    acfg,
                    asked,
                    offset=offset,
                    find=find,
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
                        asked,
                    )

        # --- The kept description: the describe ask on one item, and only that ---
        # A question is never answered from it: the description answers "what is
        # this?", and served for "what colour is the car?" it reads as a new look
        # that never happened.
        if not asked and len(media_ids) == 1 and media_ids[0] in media_info:
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
                return f"The kept description of {media_ids[0]} (no new look; ask a question for one).\n\n{cached}"

        # --- Vision analysis ---
        if not acfg.vision:
            return _tool_error(
                "vision",
                f"Analyzer '{analyzer}' has no vision capability.",
            )

        return await self._handle_vision(media_ids, media_info, acfg, asked, analyzer)

    @staticmethod
    def _reads(acfg: AnalyzerConfig, category: str) -> bool:
        """whether an analyzer can answer about media of one category.

        mirrors the routing in :meth:`_analyze`: a document is answered from its
        text whatever the analyzer's declared categories, a recording needs a
        transcriber, and anything else needs vision.

        :param acfg: the analyzer
        :ptype acfg: AnalyzerConfig
        :param category: the media category (``image``, ``document``, ``audio``, ``video``)
        :ptype category: str
        :return: ``True`` when the analyzer reads that category
        :rtype: bool
        """
        if category == "document":
            result = acfg.text is not None
        elif category in ("audio", "video"):
            result = category in acfg.supported_categories and acfg.transcription is not None
        else:
            result = category in acfg.supported_categories and acfg.vision is not None
        return result

    def _default_analyzer(self, categories: list[str]) -> str | None:
        """the first registered analyzer that reads every category, or ``None``.

        with no media resolved there is nothing to match, so the first analyzer is
        used and the routing that follows reports the missing media.

        :param categories: the media categories of the resolved items
        :ptype categories: list[str]
        :return: the analyzer's display name, or ``None`` when none reads them all
        :rtype: str | None
        """
        return next(
            (name for name, acfg in self._analyzers.items() if all(self._reads(acfg, c) for c in categories)),
            None,
        )

    def _no_default_analyzer(self, categories: list[str]) -> str:
        """what the model is told when no analyzer was named and none reads the media.

        :param categories: the media categories of the resolved items
        :ptype categories: list[str]
        :return: the sentence for the tool error
        :rtype: str
        """
        if not self._analyzers:
            result = "no analyzer is configured for this agent, so media cannot be analyzed."
        else:
            reach = "; ".join(
                f"{name} ({', '.join(sorted(acfg.supported_categories))})" for name, acfg in self._analyzers.items()
            )
            result = f"No configured analyzer reads {', '.join(categories)} media. Analyzers: {reach}."
        _log.warning(
            "analyze_media has no analyzer for the media",
            extra={"extra_data": {"categories": categories, "analyzers": list(self._analyzers)}},
        )
        return result

    def _input_schema(self) -> dict[str, Any]:
        """the input schema with the registered analyzers as the ``analyzer`` field's choices.

        built per call from :attr:`_INPUT_SCHEMA`, never by mutating it: the class
        attribute is shared by every instance. no analyzers means no ``enum`` at all,
        since an empty one admits no value.

        :return: the schema a model is bound with
        :rtype: dict[str, Any]
        """
        properties = dict(self._INPUT_SCHEMA["properties"])
        if self._analyzers:
            properties["analyzer"] = {**properties["analyzer"], "enum": list(self._analyzers)}
        return {**self._INPUT_SCHEMA, "properties": properties}

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
            except Exception as exc:
                # The analysis itself succeeded; only the caller's notification failed, so this
                # does not fail the tool. But whatever the callback was meant to do -- persist,
                # index, notify -- did not happen, and the traceback is the only diagnosis.
                contain(
                    exc,
                    log=_log,
                    what="analysis callback failed; the result was produced but not delivered",
                    extra={"media_id": mid_str, "content_type": content_type},
                    level="error",
                )

    async def _extract_from_bytes(self, mid: UUID, mid_str: str) -> tuple[str | None, str | None]:
        """read a document's text from its stored bytes, never more than :data:`MAX_DOCUMENT_BYTES`.

        the fallback for a storage that serves bytes but caches no extracted
        text. an absent download is no text (the caller answers that); a document
        past the size bound, a type no parser reads, or a parser failure, is an
        error naming why.

        :param mid: media UUID
        :ptype mid: UUID
        :param mid_str: media UUID string for logging
        :ptype mid_str: str
        :return: ``(text, None)`` on success or absence, ``(None, detail)`` on a parse failure
        :rtype: tuple[str | None, str | None]
        """
        text: str | None = None
        error: str | None = None
        try:
            dl = await self._storage.download_media(mid, max_bytes=MAX_DOCUMENT_BYTES)
        except MediaSizeLimitExceeded as exc:
            return None, _refused_too_large("document", mid_str, exc.size_bytes, exc.limit_bytes, _DOCUMENT_ASK)
        if dl is not None:
            data, mime_type = dl
            try:
                parsed = await parse_document(data, mime_type)
            except DocumentParseError as exc:
                _log.warning(
                    "document bytes could not be read as text",
                    extra={"extra_data": {"media_id": mid_str, "mime_type": mime_type, "reason": exc.reason}},
                )
                error = f"This document could not be read ({exc.reason}): {exc.detail}"
            else:
                text = parsed.text
        return text, error

    async def _handle_document(
        self,
        mid: UUID,
        mid_str: str,
        info: MediaInfo,
        acfg: AnalyzerConfig,
        question: str,
        *,
        offset: int = 0,
        find: str | None = None,
    ) -> str:
        """route a document item through extracted-text question answering.

        the text is the storage's cached extraction when it has one, otherwise
        the document's own bytes parsed by :func:`parse_document` -- a storage
        with no extraction cache (the object catalog) is still readable. those
        bytes are fetched only for a type a parser reads
        (:func:`can_parse_document`) and only up to :data:`MAX_DOCUMENT_BYTES`;
        any other type, and a document whose recorded size is over the limit,
        is answered from its metadata without downloading it.

        the part read starts at ``offset``, or just before the first match of
        ``find`` at or after it. which part was read, and every match, are said
        before the answer: a reader that keeps only the head of the result keeps
        that it covers a part. only the describe ask of the whole document's
        first part is kept as its description.

        :param mid: media UUID
        :ptype mid: UUID
        :param mid_str: media UUID string for logging/callbacks
        :ptype mid_str: str
        :param info: MediaInfo for the media item
        :ptype info: MediaInfo
        :param acfg: resolved AnalyzerConfig (must carry a text provider)
        :ptype acfg: AnalyzerConfig
        :param question: what the caller asked, or ``""`` for the describe ask
        :ptype question: str
        :param offset: where to start reading, in characters
        :ptype offset: int
        :param find: a word or phrase to open the read at
        :ptype find: str | None
        :return: provider response or formatted error string
        :rtype: str
        """
        if info.extraction_status == EXTRACTION_STATUS_PENDING:
            return "This document is still being processed and cannot be read yet. Try again in a minute."

        extracted = await self._storage.get_content(mid, "extracted_text")
        if not extracted:
            extracted = await self._storage.get_content(mid, "transcript")
        if not extracted and info.has_downloadable_data:
            # a storage with no extraction cache (the object catalog has no
            # content column) still serves the document's bytes; read the text
            # from them rather than reporting a readable document unreadable.
            # but only a type a parser reads: every type that is not image, audio
            # or video lands here, and the object store holds artifacts (packet
            # captures, database dumps) that must never be pulled whole into this
            # pod's memory only to be turned away. and a readable type is still
            # only read up to MAX_DOCUMENT_BYTES: refused here from its recorded
            # size, and bounded on the read for a size that is unknown or wrong.
            if not can_parse_document(info.mime_type):
                _log.warning(
                    "document not downloaded: no parser reads its type",
                    extra={"extra_data": {"media_id": mid_str, "mime_type": info.mime_type}},
                )
                return _tool_error(
                    "document analysis",
                    f"This document could not be read (unsupported_type): no parser reads {info.mime_type!r}",
                )
            if info.size_bytes is not None and info.size_bytes > MAX_DOCUMENT_BYTES:
                return _tool_error(
                    "document analysis",
                    _refused_too_large("document", mid_str, info.size_bytes, MAX_DOCUMENT_BYTES, _DOCUMENT_ASK),
                )
            extracted, parse_error = await self._extract_from_bytes(mid, mid_str)
            if parse_error is not None:
                return _tool_error("document analysis", parse_error)
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

        # Where to read: from offset, or from just before the first match of find.
        found_note = ""
        start = max(offset, 0)
        if find is not None and find.strip():
            found = find_in_text(extracted, find, after=start)
            found_note = found.note()
            if found.start is None:
                # Nothing to read about: the document holds no such word from there on.
                # Not a failure -- the answer to "where does it say X" is "it does not".
                return (
                    f"Looked through {mid_str} from character {start:,} of {len(extracted):,} for {find.strip()!r}: "
                    "no match, so no part was read.\n"
                    f"{found_note}"
                )
            start = found.start

        # The analyser reads a window of the document, and the answer says which
        # part it read: an analysis of the first pages, presented as an analysis
        # of the whole document, is the failure this note exists to prevent.
        window = window_text(extracted, offset=start, max_chars=self._doc_max_chars)
        suffix = f"\n\n{self._response_suffix}" if self._response_suffix else ""
        window_note = window.note(
            how=f"this analysis covers that part only; call again with offset={window.next_offset} for the next"
            if window.next_offset is not None
            else "this analysis covers that part only"
        )
        describe = not question
        # The document's words are material; a document can carry an instruction.
        # One call reads it, so nothing is cached and the nonce is minted for it.
        doc_prompt = (
            f"{question or DESCRIBE_PROMPT}\n\nThe document:\n{explained_fence(window.text, nonce=mint_nonce())}"
            f"{chr(10) + window_note if window_note else ''}{suffix}"
        )

        try:
            result_text = await text_provider.answer(doc_prompt)
        except Exception as exc:
            contain(
                exc,
                log=_log,
                what="Document analysis failed",
                extra={
                    "analyzer": acfg.name,
                    "error": str(exc),
                },
                level="error",
            )
            return _tool_error("document analysis", str(exc))

        # Kept as the document's description only when it is one: the describe ask
        # of the document from its start. An answer to a question, or a reading of
        # a later part, kept there would be served as what the whole document is.
        kept = describe and window.offset == 0 and bool(result_text) and self._user_id is not None
        if kept and self._user_id:
            try:
                await self._storage.store_content(
                    mid,
                    self._user_id,
                    "description",
                    result_text,
                    metadata={"model_name": acfg.name},
                )
            except Exception as exc:
                contain(
                    exc,
                    log=_log,
                    what="Failed to persist document description",
                    extra={
                        "media_id": mid_str,
                        "error": str(exc),
                    },
                )

            await self._fire_callback(mid_str, "description", result_text)

        if not result_text:
            return _tool_error("document analysis", "Model returned empty response.")
        end = window.offset + len(window.text)
        what = "Described" if describe else "Answered from"
        keep = (
            "; kept as its description"
            if kept
            else (
                "; not kept as its description"
                if not describe or self._user_id is None
                else "; not kept, since it covers a later part"
            )
        )
        statement = (
            f"{what} {mid_str}, characters {window.offset:,}-{end:,} of {len(extracted):,}, with {acfg.name}{keep}."
        )
        # The statement first, then what the reading covered and where find landed,
        # then the answer: the note was after the answer, where a cut drops it.
        head = "\n".join(line for line in (statement, window_note, found_note) if line)
        return f"{head}\n\n{result_text}"

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
        if info.extraction_status == EXTRACTION_STATUS_COMPLETE:
            cached_transcript = await self._storage.get_content(mid, "transcript")

        if cached_transcript:
            transcript = cached_transcript
        else:
            # download and transcribe, never more than MAX_TRANSCRIPTION_BYTES:
            # refused here from its recorded size, and bounded on the read for a
            # size that is unknown or wrong.
            kind = f"{info.media_category} file"
            if info.size_bytes is not None and info.size_bytes > MAX_TRANSCRIPTION_BYTES:
                return _tool_error(
                    "transcribe",
                    _refused_too_large(kind, mid_str, info.size_bytes, MAX_TRANSCRIPTION_BYTES, _RECORDING_ASK),
                )
            try:
                dl = await self._storage.download_media(mid, max_bytes=MAX_TRANSCRIPTION_BYTES)
            except MediaSizeLimitExceeded as exc:
                return _tool_error(
                    "transcribe",
                    _refused_too_large(kind, mid_str, exc.size_bytes, exc.limit_bytes, _RECORDING_ASK),
                )
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
                contain(
                    exc,
                    log=_log,
                    what="Transcription failed",
                    extra={
                        "media_id": mid_str,
                        "error": str(exc),
                    },
                    level="error",
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
                    contain(
                        exc,
                        log=_log,
                        what="Failed to persist transcript",
                        extra={
                            "media_id": mid_str,
                            "error": str(exc),
                        },
                    )

                await self._fire_callback(mid_str, "transcript", transcript)

        # Fetch any existing description
        description = await self._storage.get_content(mid, "description")

        # The window line leads the whole result: with a description after the
        # transcript, a note under the transcript sat in the middle of the text.
        spoken = window_text(transcript, max_chars=self._transcript_max_chars)
        heading = (
            f"**Transcript** ({info.media_category}):" if self._markdown else f"Transcript ({info.media_category}):"
        )
        window_note = spoken.note(how="the whole transcript is stored with the media")
        parts = [f"{window_note}\n{heading}\n{spoken.text}" if window_note else f"{heading}\n{spoken.text}"]
        if description:
            parts.append(f"\n\n**Analysis:**\n{description}" if self._markdown else f"\n\nDescription:\n{description}")

        return "\n".join(parts)

    async def _handle_vision(
        self,
        media_ids: list[str],
        media_info: dict[str, MediaInfo],
        acfg: AnalyzerConfig,
        question: str,
        analyzer_name: str,
    ) -> str:
        """route image media through vision analysis.

        :param media_ids: list of media UUID strings to analyze together
        :ptype media_ids: list[str]
        :param media_info: the media the caller's storage resolved, by id string;
            an id missing here was refused (not the caller's, or absent)
        :ptype media_info: dict[str, MediaInfo]
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

        # Reference path: a vision backend that resolves the image itself (a
        # gateway-backed provider) takes object ids, not bytes. all images go in
        # ONE turn, the bytes never reach this pod, and no object-store creds are
        # needed here. bytes-taking backends fall through to the download path.
        if isinstance(acfg.vision, ReferenceVisionProvider):
            # only ids the caller's storage resolved go to the backend: an id the
            # storage refused (another customer's object, or one planted in content
            # by prompt injection) must not rest on the gateway's authorization
            # alone. the bytes path gets the same guarantee from download_media.
            resolved = [mid for mid in media_ids if mid in media_info]
            if not resolved:
                return _tool_error(
                    "load media",
                    "No valid media found for the given media IDs.",
                )
            return await self._handle_vision_by_reference(resolved, acfg.vision, question, analyzer_name)

        from threetears.agent.tools.builtin.image_prep import (
            prepare_image_for_vision,
        )

        # download and preprocess all images, each never more than
        # MAX_VISION_IMAGE_BYTES. one image over the bound refuses the whole
        # analysis: an answer about the others would read as an answer about all.
        image_parts: list[tuple[bytes, str]] = []
        for mid_str in media_ids:
            mid = UUID(mid_str)
            info = media_info.get(mid_str)
            if info is not None and info.size_bytes is not None and info.size_bytes > MAX_VISION_IMAGE_BYTES:
                return _tool_error(
                    "load media",
                    f"{mid_str}: "
                    + _refused_too_large("image", mid_str, info.size_bytes, MAX_VISION_IMAGE_BYTES, _IMAGE_ASK),
                )
            try:
                dl = await self._storage.download_media(mid, max_bytes=MAX_VISION_IMAGE_BYTES)
            except MediaSizeLimitExceeded as exc:
                return _tool_error(
                    "load media",
                    f"{mid_str}: " + _refused_too_large("image", mid_str, exc.size_bytes, exc.limit_bytes, _IMAGE_ASK),
                )
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
        prompt = f"{question or DESCRIBE_PROMPT}{suffix}"

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
            contain(
                exc,
                log=_log,
                what="Vision model invocation failed",
                extra={
                    "analyzer": analyzer_name,
                    "error": str(exc),
                },
                level="error",
            )
            return _tool_error("vision model invocation", str(exc))

        # Persist description for single-media analysis
        return await self._finalize_vision(media_ids, result_text, analyzer_name, describe=not question)

    async def _handle_vision_by_reference(
        self,
        media_ids: list[str],
        vision: ReferenceVisionProvider,
        question: str,
        analyzer_name: str,
    ) -> str:
        """analyse referenced images without ever holding their bytes.

        the reference-vision backend (a gateway-backed provider) takes the
        object ids and resolves the images itself at the model boundary, so
        this pod streams no bytes and needs no object-store credentials. all
        images go in ONE turn, matching the multi-image message the platform
        supports.

        :param media_ids: media UUID strings, used verbatim as object ids
        :ptype media_ids: list[str]
        :param vision: the reference-vision backend
        :ptype vision: ReferenceVisionProvider
        :param question: prompt for the vision model
        :ptype question: str
        :param analyzer_name: display name of the analyzer
        :ptype analyzer_name: str
        :return: vision response (with optional display-url hint) or error string
        :rtype: str
        """
        suffix = f"\n\n{self._response_suffix}" if self._response_suffix else ""
        prompt = f"{question or DESCRIBE_PROMPT}{suffix}"
        object_ids = [UUID(mid) for mid in media_ids]
        try:
            result_text = await vision.analyze_ref(object_ids, prompt)
        except Exception as exc:
            contain(
                exc,
                log=_log,
                what="Reference vision invocation failed",
                extra={"analyzer": analyzer_name, "error": str(exc)},
                level="error",
            )
            return _tool_error("vision model invocation", str(exc))
        return await self._finalize_vision(media_ids, result_text, analyzer_name, describe=not question)

    async def _finalize_vision(
        self,
        media_ids: list[str],
        result_text: str,
        analyzer_name: str,
        *,
        describe: bool,
    ) -> str:
        """persist a single-image description and append a display-url hint.

        shared close-out for both vision paths (bytes and reference) so the
        persistence + display-hint behaviour has one implementation. only the
        describe ask's answer is kept as the description; the answer to a
        question is returned and kept nowhere.

        :param media_ids: the analysed media UUID strings
        :ptype media_ids: list[str]
        :param result_text: the vision model's answer
        :ptype result_text: str
        :param analyzer_name: display name of the analyzer, stored as model_name
        :ptype analyzer_name: str
        :param describe: whether this was the describe ask rather than a question
        :ptype describe: bool
        :return: the statement line, the answer, and a display hint when a URL builder exists
        :rtype: str
        """
        kept = describe and bool(result_text) and len(media_ids) == 1 and self._user_id is not None
        if kept and self._user_id:
            try:
                await self._storage.store_content(
                    UUID(media_ids[0]),
                    self._user_id,
                    "description",
                    result_text,
                    metadata={"model_name": analyzer_name},
                )
            except Exception as exc:
                contain(exc, log=_log, what="Failed to persist description", extra={"error": str(exc)})

            await self._fire_callback(media_ids[0], "description", result_text)

        items = media_ids[0] if len(media_ids) == 1 else f"{len(media_ids)} images"
        if describe:
            keep = "; kept as its description" if kept else ""
            statement = f"Described {items} with {analyzer_name}, a new look{keep}."
        else:
            statement = (
                f"Looked at {items} with {analyzer_name} for this question; the answer is not kept as a description."
            )
        result = f"{statement}\n\n{result_text}"
        if len(media_ids) == 1 and self._media_url_fn:
            url = self._media_url_fn(media_ids[0])
            if url:
                result = f"{result}\n\nTo show this image in your reply, write ![description]({url})"
        return result

    async def execute(self, **kwargs: Any) -> ToolResult:
        """analyze media items using configured providers.

        :param kwargs: must include 'media_ids'; 'question' absent or blank is the
            describe ask; 'analyzer' absent or empty means the first analyzer that
            reads the media; 'offset' and 'find' place the read in a document
        :ptype kwargs: Any
        :return: result containing analysis text or error
        :rtype: ToolResult
        """
        media_ids = kwargs.get("media_ids") or []
        question = kwargs.get("question") or ""
        analyzer = kwargs.get("analyzer") or None
        offset = int(kwargs.get("offset") or 0)
        find = kwargs.get("find") or None
        content = await self._analyze(media_ids, question, analyzer, offset=offset, find=find)
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
            input_schema=self._input_schema(),
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.media_analyze"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
