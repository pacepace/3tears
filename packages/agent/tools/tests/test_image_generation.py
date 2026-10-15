"""Tests for ``create_image_generation_tool``."""

from __future__ import annotations

import pytest

from threetears.agent.tools import (
    GeneratedImage,
    ImageGenerationBackend,
    ImageGenerationContext,
    ImageGenerationInput,
    create_image_generation_tool,
)


class _StubBackend:
    """Minimal :class:`ImageGenerationBackend` impl for tests."""

    def __init__(
        self,
        *,
        output: GeneratedImage | None = None,
        raise_with: Exception | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self._output = output or GeneratedImage(
            data=b"fake-image-bytes",
            mime_type="image/png",
            width=512,
            height=512,
        )
        self._raise = raise_with

    async def generate(
        self,
        prompt: str,
        *,
        style: str | None = None,
        source_image: bytes | None = None,
        source_mime_type: str | None = None,
    ) -> GeneratedImage:
        self.calls.append(
            {
                "prompt": prompt,
                "style": style,
                "source_image": source_image,
                "source_mime_type": source_mime_type,
            }
        )
        if self._raise is not None:
            raise self._raise
        return self._output


class _StubContext:
    """Minimal :class:`ImageGenerationContext` impl for tests."""

    def __init__(
        self,
        *,
        attached: tuple[bytes, str] | None = None,
        attached_raises: Exception | None = None,
        persist_returns: str = "[image:test-id]",
        persist_raises: Exception | None = None,
    ) -> None:
        self.persist_calls: list[dict] = []
        self.attached_calls: int = 0
        self._attached = attached
        self._attached_raises = attached_raises
        self._persist_returns = persist_returns
        self._persist_raises = persist_raises

    async def load_attached_image(self) -> tuple[bytes, str] | None:
        self.attached_calls += 1
        if self._attached_raises is not None:
            raise self._attached_raises
        return self._attached

    async def persist_generated(
        self,
        image: GeneratedImage,
        *,
        prompt: str,
        model_name: str,
        style: str | None,
    ) -> str:
        self.persist_calls.append(
            {
                "image": image,
                "prompt": prompt,
                "model_name": model_name,
                "style": style,
            }
        )
        if self._persist_raises is not None:
            raise self._persist_raises
        return self._persist_returns


def test_protocols_runtime_check() -> None:
    """Stub impls satisfy the protocols at runtime."""
    backend = _StubBackend()
    context = _StubContext()
    assert isinstance(backend, ImageGenerationBackend)
    assert isinstance(context, ImageGenerationContext)


def test_default_model_must_be_in_backends() -> None:
    """Constructing with a default not in backends raises ValueError."""
    backend = _StubBackend()
    context = _StubContext()
    with pytest.raises(ValueError, match="default_model"):
        create_image_generation_tool(
            {"alt": backend},
            context,
            default_model="missing",
        )


def test_returns_structured_tool() -> None:
    """Factory output is a single LangChain tool with the correct name + schema."""
    backend = _StubBackend()
    context = _StubContext()
    tool = create_image_generation_tool(
        {"sd": backend},
        context,
        default_model="sd",
    )
    # Canonical name is the dotted ``threetears.X`` form -- uniform
    # across all builtins; provider wire-format quirks live at the
    # chat-model boundary, not in the tool definition.
    assert tool.name == "threetears.image_generation"
    assert tool.args_schema is ImageGenerationInput


@pytest.mark.asyncio
async def test_default_model_is_used_when_none_specified() -> None:
    """Omitting ``model`` selects the default backend."""
    primary = _StubBackend()
    other = _StubBackend()
    context = _StubContext()
    tool = create_image_generation_tool(
        {"primary": primary, "other": other},
        context,
        default_model="primary",
    )

    result = await tool.ainvoke({"prompt": "a sunset"})
    assert result == "[image:test-id]"
    assert len(primary.calls) == 1
    assert len(other.calls) == 0


@pytest.mark.asyncio
async def test_explicit_model_routes_to_named_backend() -> None:
    """An explicit ``model`` name routes to the corresponding backend."""
    primary = _StubBackend()
    sd = _StubBackend()
    context = _StubContext()
    tool = create_image_generation_tool(
        {"primary": primary, "sd": sd},
        context,
        default_model="primary",
    )

    await tool.ainvoke({"prompt": "a city", "model": "sd"})
    assert len(primary.calls) == 0
    assert len(sd.calls) == 1


@pytest.mark.asyncio
async def test_unknown_model_returns_tool_error() -> None:
    """Routing to an unregistered model produces an LLM-facing error."""
    backend = _StubBackend()
    context = _StubContext()
    tool = create_image_generation_tool(
        {"sd": backend},
        context,
        default_model="sd",
    )
    result = await tool.ainvoke({"prompt": "x", "model": "unknown"})
    assert "[TOOL ERROR]" in result
    assert "Unknown model" in result
    assert "'unknown'" in result
    assert "sd" in result  # available model listed for the LLM
    assert len(backend.calls) == 0


@pytest.mark.asyncio
async def test_img2img_loads_attached_image() -> None:
    """``use_attached_image=True`` calls the host loader and forwards bytes."""
    backend = _StubBackend()
    context = _StubContext(attached=(b"source-bytes", "image/jpeg"))
    tool = create_image_generation_tool(
        {"sd": backend},
        context,
        default_model="sd",
    )

    await tool.ainvoke(
        {
            "prompt": "make it sepia",
            "use_attached_image": True,
        }
    )
    assert context.attached_calls == 1
    assert len(backend.calls) == 1
    assert backend.calls[0]["source_image"] == b"source-bytes"
    assert backend.calls[0]["source_mime_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_img2img_no_source_changes_nothing() -> None:
    """img2img with no attached image makes nothing, and says so as a no-change, not a failure."""
    backend = _StubBackend()
    context = _StubContext(attached=None)
    tool = create_image_generation_tool(
        {"sd": backend},
        context,
        default_model="sd",
    )

    result = await tool.ainvoke(
        {
            "prompt": "make it sepia",
            "use_attached_image": True,
        }
    )
    assert result.startswith("[NO CHANGE] image_generation:")
    assert "Ask the person to attach one" in result
    assert len(backend.calls) == 0


@pytest.mark.asyncio
async def test_img2img_loader_exception_returns_tool_error() -> None:
    """A raising loader becomes an LLM-facing tool error."""
    backend = _StubBackend()
    context = _StubContext(attached_raises=RuntimeError("S3 timeout"))
    tool = create_image_generation_tool(
        {"sd": backend},
        context,
        default_model="sd",
    )
    result = await tool.ainvoke(
        {
            "prompt": "x",
            "use_attached_image": True,
        }
    )
    assert "[TOOL ERROR]" in result
    assert "S3 timeout" in result
    assert len(backend.calls) == 0


@pytest.mark.asyncio
async def test_no_img2img_does_not_call_loader() -> None:
    """Default ``use_attached_image=False`` skips the loader entirely."""
    backend = _StubBackend()
    context = _StubContext(attached=(b"never-loaded", "image/png"))
    tool = create_image_generation_tool(
        {"sd": backend},
        context,
        default_model="sd",
    )

    await tool.ainvoke({"prompt": "from scratch"})
    assert context.attached_calls == 0
    assert backend.calls[0]["source_image"] is None
    assert backend.calls[0]["source_mime_type"] is None


@pytest.mark.asyncio
async def test_backend_exception_returns_tool_error() -> None:
    """Backend failures turn into an LLM-facing tool error."""
    backend = _StubBackend(raise_with=RuntimeError("provider rate limited"))
    context = _StubContext()
    tool = create_image_generation_tool(
        {"sd": backend},
        context,
        default_model="sd",
    )
    result = await tool.ainvoke({"prompt": "x"})
    assert "[TOOL ERROR]" in result
    assert "provider rate limited" in result
    assert context.persist_calls == []  # never reached the host


@pytest.mark.asyncio
async def test_persist_callback_receives_full_context() -> None:
    """The persistence hook receives the generated image + invocation context."""
    backend = _StubBackend()
    context = _StubContext()
    tool = create_image_generation_tool(
        {"flux": backend},
        context,
        default_model="flux",
    )
    await tool.ainvoke(
        {
            "prompt": "a city skyline",
            "model": "flux",
            "style": "vivid",
        }
    )
    assert len(context.persist_calls) == 1
    call = context.persist_calls[0]
    assert call["prompt"] == "a city skyline"
    assert call["model_name"] == "flux"
    assert call["style"] == "vivid"
    assert call["image"].mime_type == "image/png"
    assert call["image"].data == b"fake-image-bytes"


@pytest.mark.asyncio
async def test_persist_exception_returns_tool_error() -> None:
    """A persistence failure turns into an LLM-facing tool error."""
    backend = _StubBackend()
    context = _StubContext(persist_raises=RuntimeError("disk full"))
    tool = create_image_generation_tool(
        {"sd": backend},
        context,
        default_model="sd",
    )
    result = await tool.ainvoke({"prompt": "x"})
    assert "[TOOL ERROR]" in result
    assert "disk full" in result


@pytest.mark.asyncio
async def test_persist_response_is_passthrough() -> None:
    """The host's persist return string is the LLM-facing response unchanged."""
    backend = _StubBackend()
    context = _StubContext(
        persist_returns="![Generated](/api/v1/media/abc) [Image: a sunset]",
    )
    tool = create_image_generation_tool(
        {"sd": backend},
        context,
        default_model="sd",
    )
    result = await tool.ainvoke({"prompt": "a sunset"})
    assert result == "![Generated](/api/v1/media/abc) [Image: a sunset]"


@pytest.mark.asyncio
async def test_custom_description_override() -> None:
    """``description=`` kwarg overrides the platform-standard wording."""
    backend = _StubBackend()
    context = _StubContext()
    custom = "Generate images via the SDXL backend (default) or 'flux'."
    tool = create_image_generation_tool(
        {"sdxl": backend, "flux": backend},
        context,
        default_model="sdxl",
        description=custom,
    )
    assert tool.description == custom


# ---------------------------------------------------------------------------
# A source by id, a model that cannot edit, and options a model cannot apply
# ---------------------------------------------------------------------------


from threetears.agent.tools import SourceImage, SourceImageContext, SourceImageRefused  # noqa: E402
from threetears.media.contracts import GENERATION_IGNORED, GENERATION_SOURCE_USED  # noqa: E402


class _EditingBackend:
    """A backend that takes a source, a size and a seed, and says what it did with them."""

    takes_source_image = True

    def __init__(self, *, source_used: bool | None = True, ignored: dict[str, str] | None = None) -> None:
        self.calls: list[dict] = []
        self._source_used = source_used
        self._ignored = ignored or {}

    async def generate(
        self,
        prompt: str,
        *,
        style: str | None = None,
        source_image: bytes | None = None,
        source_mime_type: str | None = None,
        size: str | None = None,
        aspect: str | None = None,
        seed: int | None = None,
    ) -> GeneratedImage:
        self.calls.append({"source_image": source_image, "size": size, "aspect": aspect, "seed": seed})
        metadata: dict = {GENERATION_IGNORED: dict(self._ignored)}
        if source_image is not None and self._source_used is not None:
            metadata[GENERATION_SOURCE_USED] = self._source_used
        return GeneratedImage(data=b"img", mime_type="image/png", metadata=metadata)


class _WordsOnlyBackend(_StubBackend):
    """A backend that says it cannot change an image."""

    takes_source_image = False


class _SourceContext(_StubContext):
    """A host that hands over a source by id, or refuses it with why."""

    def __init__(self, *, sources: dict[str, SourceImage], **kw) -> None:
        super().__init__(persist_returns="Made an image [media:new-1].\n![x](/m/new-1)", **kw)
        self._sources = sources
        self.asked: list[str] = []

    async def load_source_image(self, media_id: str) -> SourceImage:
        self.asked.append(media_id)
        if media_id not in self._sources:
            raise SourceImageRefused(f"{media_id} is not an image in this conversation.")
        return self._sources[media_id]


def test_a_host_that_loads_by_id_is_a_source_image_context() -> None:
    assert isinstance(_SourceContext(sources={}), SourceImageContext)
    assert not isinstance(_StubContext(), SourceImageContext)


@pytest.mark.asyncio
async def test_a_source_by_id_reaches_the_backend_and_is_named_to_the_host() -> None:
    backend = _EditingBackend()
    context = _SourceContext(sources={"m1": SourceImage(b"cat-bytes", "image/png", "cat.png")})
    tool = create_image_generation_tool({"edit": backend}, context, default_model="edit")

    result = await tool.ainvoke({"prompt": "give the cat a hat", "source_media_id": "m1"})

    assert context.asked == ["m1"]
    assert backend.calls[0]["source_image"] == b"cat-bytes"
    image = context.persist_calls[0]["image"]
    assert image.metadata["source_label"] == "cat.png"
    assert image.metadata[GENERATION_SOURCE_USED] is True
    assert result == "Made an image [media:new-1].\n![x](/m/new-1)"


@pytest.mark.asyncio
async def test_a_refused_source_changes_nothing_and_says_why() -> None:
    backend = _EditingBackend()
    context = _SourceContext(sources={})
    tool = create_image_generation_tool({"edit": backend}, context, default_model="edit")

    result = await tool.ainvoke({"prompt": "x", "source_media_id": "pdf-9"})

    assert result.startswith("[NO CHANGE] image_generation: pdf-9 is not an image in this conversation.")
    assert backend.calls == []


@pytest.mark.asyncio
async def test_a_host_that_cannot_load_by_id_changes_nothing() -> None:
    backend = _EditingBackend()
    tool = create_image_generation_tool({"edit": backend}, _StubContext(), default_model="edit")

    result = await tool.ainvoke({"prompt": "x", "source_media_id": "m1"})

    assert result.startswith("[NO CHANGE] image_generation:")
    assert backend.calls == []


@pytest.mark.asyncio
async def test_a_model_that_makes_images_from_words_only_is_refused_an_edit_naming_one_that_can() -> None:
    words = _WordsOnlyBackend()
    edit = _EditingBackend()
    context = _SourceContext(sources={"m1": SourceImage(b"cat", "image/png", "cat.png")})
    tool = create_image_generation_tool({"words": words, "editor": edit}, context, default_model="words")

    result = await tool.ainvoke({"prompt": "give the cat a hat", "source_media_id": "m1"})

    assert result.startswith("[NO CHANGE] image_generation: words makes images from words only")
    assert "Models that can: editor." in result
    assert words.calls == [] and edit.calls == [] and context.asked == []


@pytest.mark.asyncio
async def test_a_words_only_model_still_makes_an_image_from_words() -> None:
    """the guard's good traffic: no source asked, so the words-only model makes the image."""
    words = _WordsOnlyBackend()
    context = _StubContext()
    tool = create_image_generation_tool({"words": words}, context, default_model="words")

    result = await tool.ainvoke({"prompt": "a lighthouse"})

    assert result == "[image:test-id]"
    assert len(words.calls) == 1


@pytest.mark.asyncio
async def test_an_edit_that_used_no_source_says_so_on_the_line_after_the_statement() -> None:
    backend = _EditingBackend(source_used=False)
    context = _SourceContext(sources={"m1": SourceImage(b"cat", "image/png", "cat.png")})
    tool = create_image_generation_tool({"edit": backend}, context, default_model="edit")

    result = await tool.ainvoke({"prompt": "give the cat a hat", "source_media_id": "m1"})

    lines = result.split("\n")
    assert lines[0] == "Made an image [media:new-1]."
    assert lines[1] == "No source was used: edit made this from the words alone, not from cat.png."


@pytest.mark.asyncio
async def test_an_edit_that_used_its_source_carries_no_such_note() -> None:
    backend = _EditingBackend(source_used=True)
    context = _SourceContext(sources={"m1": SourceImage(b"cat", "image/png", "cat.png")})
    tool = create_image_generation_tool({"edit": backend}, context, default_model="edit")

    result = await tool.ainvoke({"prompt": "give the cat a hat", "source_media_id": "m1"})

    assert "No source was used" not in result


@pytest.mark.asyncio
async def test_size_aspect_and_seed_reach_a_backend_that_takes_them() -> None:
    backend = _EditingBackend()
    tool = create_image_generation_tool({"edit": backend}, _StubContext(), default_model="edit")

    await tool.ainvoke({"prompt": "x", "size": "1024x1536", "aspect": "portrait", "seed": 7})

    assert backend.calls[0] == {"source_image": None, "size": "1024x1536", "aspect": "portrait", "seed": 7}


@pytest.mark.asyncio
async def test_an_option_a_backend_does_not_take_is_not_sent_and_is_named() -> None:
    old = _StubBackend()  # its generate takes no size or seed
    context = _StubContext()
    tool = create_image_generation_tool({"old": old}, context, default_model="old")

    result = await tool.ainvoke({"prompt": "x", "seed": 7, "size": "512x512"})

    assert len(old.calls) == 1
    assert (
        result.split("\n")[1] == "Not applied: seed (this model does not take it); size (this model does not take it)."
    )


@pytest.mark.asyncio
async def test_what_a_backend_says_it_ignored_is_named() -> None:
    backend = _EditingBackend(ignored={"seed": "OpenAI takes no seed"})
    tool = create_image_generation_tool({"edit": backend}, _StubContext(), default_model="edit")

    result = await tool.ainvoke({"prompt": "x", "seed": 7})

    assert "Not applied: seed (OpenAI takes no seed)." in result
