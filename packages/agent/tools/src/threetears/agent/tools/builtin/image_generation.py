"""Image generation tool with pluggable backends + host persistence.

The 3tears side owns:
- the LLM-facing input schema (``prompt`` / ``model`` / ``style`` /
  ``source_media_id`` / ``use_attached_image`` / ``size`` / ``aspect`` /
  ``seed``);
- backend routing (pick the named backend or the configured default);
- img2img source-image loading via host-provided callbacks, and the refusal
  of an edit on a model that makes images from words only;
- saying what was not done: a source the backend did not use, an option it
  could not apply;
- error-string formatting for unknown models, refused sources, backend
  exceptions.

The host owns:
- the actual ``ImageGenerationBackend`` implementations registered
  under named keys;
- finding a source image it lets this caller use
  (:class:`ImageGenerationContext.load_attached_image`, and
  :class:`SourceImageContext.load_source_image` by id);
- persistence of the produced image (storage, DB rows, token accounting,
  auto-description) and the LLM-facing return string
  (via :class:`ImageGenerationContext.persist_generated`).

This split keeps the generic routing / schema / error-shaping in 3tears
without coupling it to any specific storage / DB / observability stack.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from threetears.media.contracts import GENERATION_IGNORED, GENERATION_SOURCE_USED

from threetears.agent.tools.protocols import (
    GeneratedImage,
    ImageGenerationBackend,
)
from threetears.agent.tools.utils import tool_error, tool_no_change
from threetears.observe import contain, get_logger

__all__ = [
    "GENERATION_SOURCE_LABEL",
    "ImageGenerationContext",
    "ImageGenerationInput",
    "SourceImage",
    "SourceImageContext",
    "SourceImageRefused",
    "create_image_generation_tool",
]


_log = get_logger(__name__)

_TOOL = "image_generation"

#: ``GeneratedImage.metadata`` key the tool sets before the host persists the image:
#: what the source was, as the model reads it (a file name, or "the newest image
#: attached here"), or ``None`` when no source was asked for.
GENERATION_SOURCE_LABEL = "source_label"

_DEFAULT_DESCRIPTION = (
    "Make an image from words, or change one. Pick a model with model; leave it out for the default. "
    "To change or vary an image, pass its id as source_media_id, or set use_attached_image for the "
    "newest image attached here; the prompt then says what to change. size, aspect and seed shape the "
    "result where the model takes them, and the result says which it could not apply."
)


class ImageGenerationInput(BaseModel):
    """Input schema for the ``image_generation`` tool.

    :ivar prompt: text description of the image to generate, or of the change
        to make to the source image.
    :ivar model: optional display name of the image-generation model to
        use. When omitted, the configured default model handles the
        request. Hosts include the available model names in the tool's
        ``description`` so the calling LLM can pick one.
    :ivar style: optional style hint forwarded to the backend.
    :ivar source_media_id: id of an image the host lets this caller use as the
        source to change or vary.
    :ivar use_attached_image: use the newest image attached in this
        conversation as the source.
    :ivar size: pixel size ``WIDTHxHEIGHT``.
    :ivar aspect: shape when no size is given.
    :ivar seed: a number that repeats an image for the same prompt.
    """

    prompt: str = Field(
        description="What to make, in detail; with a source image, what to change in it.",
    )
    model: str | None = Field(
        default=None,
        description="Display name of the image model to use. If omitted, the default model is used.",
    )
    style: str | None = Field(
        default=None,
        description="Optional style hint for the image (e.g. 'natural', 'vivid', 'anime', 'photorealistic').",
    )
    source_media_id: str | None = Field(
        default=None,
        description=(
            "The id of an image in this conversation to change or vary, e.g. one you made earlier. "
            "Only an image is taken; a document or a recording is refused."
        ),
    )
    use_attached_image: bool = Field(
        default=False,
        description=(
            "Set to true to change the newest image attached in this conversation. source_media_id names a "
            "particular one instead."
        ),
    )
    size: str | None = Field(
        default=None,
        description="Size in pixels as WIDTHxHEIGHT, e.g. 1024x1536. A size the model cannot make is refused with the sizes it takes.",
    )
    aspect: str | None = Field(
        default=None,
        description="Shape when no size is given: square, landscape, portrait, or a ratio such as 16:9.",
    )
    seed: int | None = Field(
        default=None,
        description="A number that makes the same prompt give the same image again, on a model that takes one.",
    )


@dataclass(frozen=True)
class SourceImage:
    """An image the host lets the caller change, as the backend takes it.

    :ivar data: the image's bytes
    :ivar mime_type: its type, ``image/...``
    :ivar label: what it is called where the model reads about it: a file name,
        or ``image <id>``
    """

    data: bytes
    mime_type: str
    label: str


class SourceImageRefused(Exception):
    """The host will not hand over the source asked for, and says why.

    Raised by :meth:`SourceImageContext.load_source_image`. The reason reaches
    the model as a no-change result, because nothing was made: "not an image",
    "not in this conversation", "no such media".

    :param reason: why, in a sentence the model reads
    :ptype reason: str
    """

    def __init__(self, reason: str) -> None:
        """Keep the reason.

        :param reason: why, in a sentence the model reads
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        self.reason = reason
        super().__init__(reason)


@runtime_checkable
class ImageGenerationContext(Protocol):
    """Host callbacks the ``image_generation`` tool delegates to.

    Hosts plug in their per-tenant attached-image lookup and their
    storage / accounting / response-formatting orchestration via this
    contract. Implementations must be safe to call concurrently from
    different conversations.
    """

    async def load_attached_image(self) -> tuple[bytes, str] | SourceImage | None:
        """Return the newest image attached where this call is made, or ``None``.

        Called only when the invoking LLM sets ``use_attached_image=True``.
        ``None`` reaches the LLM as a no-change result asking for an image. A
        host returns a :class:`SourceImage` to name the file; a bare
        ``(bytes, mime_type)`` is read as an unnamed attached image.

        :return: the source, or None
        :rtype: tuple[bytes, str] | SourceImage | None
        """
        ...

    async def persist_generated(
        self,
        image: GeneratedImage,
        *,
        prompt: str,
        model_name: str,
        style: str | None,
    ) -> str:
        """Persist a generated image and return the LLM-facing response.

        Hosts write the image to durable storage (S3, filesystem, etc),
        record any per-conversation metadata, run any auto-describe
        pipeline they want surfaced, and return the final string the
        invoking LLM consumes. Its first line is the host's statement of
        what was made; the tool puts its own notes on the line after it.

        ``image.metadata`` carries what the backend said it did (the
        ``GENERATION_*`` keys of :mod:`threetears.media.contracts`) and
        :data:`GENERATION_SOURCE_LABEL`.

        Implementations may raise; the tool catches the exception and
        produces an LLM-facing ``[TOOL ERROR] image_generation: save``
        message.

        :param image: backend output (bytes, mime, optional dimensions)
        :ptype image: GeneratedImage
        :param prompt: original generation prompt
        :ptype prompt: str
        :param model_name: name of the backend that produced the image
            (the resolved name, after default fallback)
        :ptype model_name: str
        :param style: original style hint, if any
        :ptype style: str | None
        :return: LLM-facing response string
        :rtype: str
        """
        ...


@runtime_checkable
class SourceImageContext(Protocol):
    """A host that can hand over a source image by its id.

    Separate from :class:`ImageGenerationContext` so a host written before it
    still satisfies that protocol; a call with ``source_media_id`` on such a
    host is refused with a no-change result.
    """

    async def load_source_image(self, media_id: str) -> SourceImage:
        """Return the image ``media_id`` names, when this caller may use it.

        :param media_id: the id the LLM passed
        :ptype media_id: str
        :return: the source
        :rtype: SourceImage
        :raises SourceImageRefused: when the id names nothing this caller may use
            as a source image, with why
        """
        ...


def _takes(backend: Any, option: str) -> bool:
    """Whether a backend's ``generate`` accepts an option by name.

    :param backend: the backend
    :ptype backend: Any
    :param option: the keyword
    :ptype option: str
    :return: ``True`` when the keyword (or ``**kwargs``) is in its signature
    :rtype: bool
    """
    try:
        params = inspect.signature(backend.generate).parameters
    except TypeError, ValueError:
        return False
    return option in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _with_notes(response: str, notes: list[str]) -> str:
    """Put the tool's notes on the line after the host's statement of what it made.

    :param response: what the host returned
    :ptype response: str
    :param notes: the tool's notes, one line each
    :ptype notes: list[str]
    :return: the response with the notes as its second line
    :rtype: str
    """
    if not notes:
        return response
    first, _, rest = response.partition("\n")
    body = "\n".join(notes)
    return f"{first}\n{body}\n{rest}" if rest else f"{first}\n{body}"


def create_image_generation_tool(
    backends: dict[str, ImageGenerationBackend],
    context: ImageGenerationContext,
    *,
    default_model: str,
    description: str | None = None,
) -> StructuredTool:
    """Build the ``image_generation`` LangChain tool.

    :param backends: mapping of model display name to
        :class:`ImageGenerationBackend` implementation. The mapping is
        captured by reference; mutating it after construction is
        permitted (e.g. hot-swap during a config refresh) but the
        resolved ``default_model`` must always be a key.
    :ptype backends: dict[str, ImageGenerationBackend]
    :param context: host callbacks for img2img source loading and
        generated-image persistence; a host that also satisfies
        :class:`SourceImageContext` takes ``source_media_id``.
    :ptype context: ImageGenerationContext
    :param default_model: name (key in ``backends``) used when the
        calling LLM omits ``model``. Must be present in ``backends`` at
        call time.
    :ptype default_model: str
    :param description: optional override for the LLM-facing tool
        description. Defaults to the platform-standard wording; hosts
        typically override to embed the available model names so the
        calling LLM can pick a specific model.
    :ptype description: str | None
    :return: configured LangChain tool
    :rtype: StructuredTool
    :raises ValueError: when ``default_model`` is not a key of ``backends``
    """
    if default_model not in backends:
        raise ValueError(
            f"default_model {default_model!r} is not registered in backends; available: {sorted(backends.keys())!r}",
        )

    async def _load_source(source_media_id: str | None, use_attached_image: bool) -> SourceImage | str:
        """The source asked for, or the no-change sentence saying why there is none."""
        if source_media_id:
            if not isinstance(context, SourceImageContext):
                return (
                    "this host cannot take a source image by its id; use use_attached_image instead. Nothing was made."
                )
            try:
                return await context.load_source_image(source_media_id.strip())
            except SourceImageRefused as exc:
                return f"{exc.reason} Nothing was made."
        loaded = await context.load_attached_image()
        if loaded is None:
            return (
                "no image is attached in this conversation to change. Ask the person to attach one. Nothing was made."
            )
        if isinstance(loaded, SourceImage):
            return loaded
        data, mime_type = loaded
        return SourceImage(data=data, mime_type=mime_type, label="the newest attached image")

    async def _generate(
        prompt: str,
        model: str | None = None,
        style: str | None = None,
        source_media_id: str | None = None,
        use_attached_image: bool = False,
        size: str | None = None,
        aspect: str | None = None,
        seed: int | None = None,
    ) -> str:
        # -- Resolve backend -------------------------------------------------
        selected_model = model or default_model
        backend = backends.get(selected_model)
        if backend is None:
            return tool_error(
                _TOOL,
                "resolve model",
                f"Unknown model {selected_model!r}. Available models: {', '.join(sorted(backends.keys()))}",
            )

        wants_source = bool(source_media_id) or use_attached_image
        _log.debug(
            "image_generation invoked",
            extra={
                "extra_data": {
                    "prompt_preview": prompt[:100],
                    "model": selected_model,
                    "style": style,
                    "source_media_id": source_media_id,
                    "use_attached_image": use_attached_image,
                    "size": size,
                    "aspect": aspect,
                    "seed": seed,
                }
            },
        )

        # -- A model that cannot change an image is refused before it is paid for
        if wants_source and getattr(backend, "takes_source_image", None) is False:
            can = sorted(n for n, b in backends.items() if getattr(b, "takes_source_image", None) is True)
            others = f"Models that can: {', '.join(can)}." if can else "No model here can change an image."
            return tool_no_change(
                _TOOL,
                f"{selected_model} makes images from words only, so it cannot change an image. {others} Nothing was made.",
            )

        # -- img2img source loading ----------------------------------------
        source: SourceImage | None = None
        if wants_source:
            try:
                loaded = await _load_source(source_media_id, use_attached_image)
            except Exception as exc:
                contain(exc, log=_log, what="img2img source load failed")
                return tool_error(_TOOL, "img2img source load", str(exc))
            if isinstance(loaded, str):
                return tool_no_change(_TOOL, loaded)
            source = loaded

        # -- Options: passed only when set, and only to a backend that takes them
        asked: dict[str, Any] = {"size": size, "aspect": aspect, "seed": seed}
        options: dict[str, Any] = {k: v for k, v in asked.items() if v is not None and _takes(backend, k)}
        unread = {k: "this model does not take it" for k, v in asked.items() if v is not None and k not in options}

        # -- Backend invocation --------------------------------------------
        try:
            result = await backend.generate(
                prompt,
                style=style,
                source_image=source.data if source else None,
                source_mime_type=source.mime_type if source else None,
                **options,
            )
        except Exception as exc:
            contain(
                exc,
                log=_log,
                what="image_generation backend failed",
                extra={
                    "model": selected_model,
                    "error": str(exc),
                },
                level=logging.ERROR,
            )
            return tool_error(_TOOL, "generate", str(exc))

        # -- What was done, as the backend said, beside what was asked -----
        metadata: dict[str, Any] = dict(result.metadata or {})
        ignored: dict[str, str] = {**unread, **dict(metadata.get(GENERATION_IGNORED) or {})}
        metadata[GENERATION_IGNORED] = ignored
        metadata[GENERATION_SOURCE_LABEL] = source.label if source else None
        if source is None:
            metadata[GENERATION_SOURCE_USED] = None
        result.metadata = metadata

        notes: list[str] = []
        if source is not None:
            used = metadata.get(GENERATION_SOURCE_USED)
            if used is False:
                notes.append(
                    f"No source was used: {selected_model} made this from the words alone, not from {source.label}."
                )
            elif used is None:
                notes.append(f"{selected_model} does not say whether it used {source.label} as the source.")
        if ignored:
            notes.append("Not applied: " + "; ".join(f"{name} ({why})" for name, why in sorted(ignored.items())) + ".")

        # -- Host-side persistence + response formatting --------------------
        try:
            response = await context.persist_generated(
                result,
                prompt=prompt,
                model_name=selected_model,
                style=style,
            )
        except Exception as exc:
            contain(
                exc,
                log=_log,
                what="image_generation persistence failed",
                extra={
                    "model": selected_model,
                    "error": str(exc),
                },
                level=logging.ERROR,
            )
            return tool_error(_TOOL, "save", str(exc))

        return _with_notes(response, notes)

    return StructuredTool.from_function(
        coroutine=_generate,
        # ``threetears.image_generation`` matches the uniform 3tears
        # builtin naming (``threetears.calculator`` /
        # ``threetears.current_date`` / etc.). The dotted form is
        # canonical across the platform; provider-specific wire
        # constraints (e.g. Bedrock's ``^[a-zA-Z0-9_-]{1,128}$``
        # tool-name validator that rejects dots) are handled at the
        # OpenRouter chat-model boundary by the name-translating
        # wrapper in ``packages/models/.../providers/openrouter.py``,
        # which swaps dots for underscores on the wire and back on
        # the response so application code never sees the translated
        # form.
        name="threetears.image_generation",
        description=description or _DEFAULT_DESCRIPTION,
        args_schema=ImageGenerationInput,
    )
