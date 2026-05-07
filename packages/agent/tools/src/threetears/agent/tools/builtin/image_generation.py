"""Image generation tool with pluggable backends + host persistence.

The 3tears side owns:
- the LLM-facing input schema (``prompt`` / ``model`` / ``style`` /
  ``use_attached_image``);
- backend routing (pick the named backend or the configured default);
- img2img source-image loading via a host-provided callback when
  ``use_attached_image=True``;
- error-string formatting for unknown models, source-image-not-found,
  backend exceptions.

The host owns:
- the actual ``ImageGenerationBackend`` implementations registered
  under named keys;
- per-tenant retrieval of the user's most recent uploaded image
  (via :class:`ImageGenerationContext.load_attached_image`);
- persistence of the produced image (storage, DB rows, token accounting,
  auto-description) and the LLM-facing return string
  (via :class:`ImageGenerationContext.persist_generated`).

This split keeps the generic routing / schema / error-shaping in 3tears
without coupling it to any specific storage / DB / observability stack.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.protocols import (
    GeneratedImage,
    ImageGenerationBackend,
)
from threetears.agent.tools.utils import tool_error
from threetears.observe import get_logger

__all__ = [
    "ImageGenerationContext",
    "ImageGenerationInput",
    "create_image_generation_tool",
]


_log = get_logger(__name__)


_DEFAULT_DESCRIPTION = (
    "Generate an image from a text description. Use the ``model`` parameter "
    "to pick an available image-generation model; omit it to use the default. "
    "Set ``use_attached_image=true`` to use the user's most recently uploaded "
    "media as the source for image-to-image generation; the prompt then "
    "describes how to transform that source."
)


class ImageGenerationInput(BaseModel):
    """Input schema for the ``image_generation`` tool.

    :ivar prompt: text description of the image to generate.
    :ivar model: optional display name of the image-generation model to
        use. When omitted, the configured default model handles the
        request. Hosts include the available model names in the tool's
        ``description`` so the calling LLM can pick one.
    :ivar style: optional style hint forwarded to the backend (e.g.
        ``natural`` / ``vivid`` / ``anime`` / ``photorealistic``).
        Backends that ignore the hint return as if it were not set.
    :ivar use_attached_image: when ``True``, the host's most recent
        uploaded image is loaded and forwarded to the backend as the
        source for image-to-image generation. The backend may reject
        the request if it does not support img2img.
    """

    prompt: str = Field(
        description="A detailed text description of the image to generate",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Display name of the image-generation model to use. If omitted, "
            "the default model is used."
        ),
    )
    style: str | None = Field(
        default=None,
        description=(
            "Optional style hint for the image (e.g. 'natural', 'vivid', "
            "'anime', 'photorealistic')"
        ),
    )
    use_attached_image: bool = Field(
        default=False,
        description=(
            "Set to true to use the user's most recently uploaded media as a "
            "source for image-to-image generation. The prompt describes how "
            "to transform the image."
        ),
    )


@runtime_checkable
class ImageGenerationContext(Protocol):
    """Host callbacks the ``image_generation`` tool delegates to.

    Hosts plug in their per-tenant attached-image lookup and their
    storage / accounting / response-formatting orchestration via this
    contract. Implementations must be safe to call concurrently from
    different conversations.
    """

    async def load_attached_image(self) -> tuple[bytes, str] | None:
        """Return ``(bytes, mime_type)`` for the user's most recent
        uploaded image, or ``None`` when no eligible source exists.

        Called by :func:`create_image_generation_tool` only when the
        invoking LLM sets ``use_attached_image=True``. Returning
        ``None`` causes the tool to surface
        ``[TOOL ERROR] image_generation: ...`` to the LLM with a
        prompt to attach an image first.

        :return: source bytes + MIME type, or None
        :rtype: tuple[bytes, str] | None
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
        invoking LLM consumes — typically a markdown image tag, an
        ``[image:<id>]`` reference, or both, plus any short description.

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
        generated-image persistence.
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
            f"default_model {default_model!r} is not registered in backends; "
            f"available: {sorted(backends.keys())!r}",
        )

    async def _generate(
        prompt: str,
        model: str | None = None,
        style: str | None = None,
        use_attached_image: bool = False,
    ) -> str:
        # -- Resolve backend -------------------------------------------------
        selected_model = model or default_model
        backend = backends.get(selected_model)
        if backend is None:
            return tool_error(
                "image_generation",
                "resolve model",
                f"Unknown model {selected_model!r}. Available models: "
                f"{', '.join(sorted(backends.keys()))}",
            )

        _log.debug(
            "image_generation invoked",
            extra={"extra_data": {
                "prompt_preview": prompt[:100],
                "model": selected_model,
                "style": style,
                "use_attached_image": use_attached_image,
            }},
        )

        # -- img2img source loading ----------------------------------------
        source_image: bytes | None = None
        source_mime_type: str | None = None
        if use_attached_image:
            try:
                loaded = await context.load_attached_image()
            except Exception as exc:
                return tool_error(
                    "image_generation",
                    "img2img source load",
                    str(exc),
                )
            if loaded is None:
                return tool_error(
                    "image_generation",
                    "img2img",
                    "No downloadable source image found. Ask the user to "
                    "attach an image first.",
                )
            source_image, source_mime_type = loaded

        # -- Backend invocation --------------------------------------------
        try:
            result = await backend.generate(
                prompt,
                style=style,
                source_image=source_image,
                source_mime_type=source_mime_type,
            )
        except Exception as exc:
            _log.error(
                "image_generation backend failed",
                extra={"extra_data": {
                    "model": selected_model,
                    "error": str(exc),
                }},
            )
            return tool_error("image_generation", "generate", str(exc))

        # -- Host-side persistence + response formatting --------------------
        try:
            response = await context.persist_generated(
                result,
                prompt=prompt,
                model_name=selected_model,
                style=style,
            )
        except Exception as exc:
            _log.error(
                "image_generation persistence failed",
                extra={"extra_data": {
                    "model": selected_model,
                    "error": str(exc),
                }},
            )
            return tool_error("image_generation", "save", str(exc))

        return response

    return StructuredTool.from_function(
        coroutine=_generate,
        name="image_generation",
        description=description or _DEFAULT_DESCRIPTION,
        args_schema=ImageGenerationInput,
    )
