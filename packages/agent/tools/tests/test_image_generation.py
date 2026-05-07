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
        self.calls.append({
            "prompt": prompt,
            "style": style,
            "source_image": source_image,
            "source_mime_type": source_mime_type,
        })
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
        self.persist_calls.append({
            "image": image,
            "prompt": prompt,
            "model_name": model_name,
            "style": style,
        })
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

    await tool.ainvoke({
        "prompt": "make it sepia",
        "use_attached_image": True,
    })
    assert context.attached_calls == 1
    assert len(backend.calls) == 1
    assert backend.calls[0]["source_image"] == b"source-bytes"
    assert backend.calls[0]["source_mime_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_img2img_no_source_returns_tool_error() -> None:
    """img2img when no attached image is available is an LLM-facing error."""
    backend = _StubBackend()
    context = _StubContext(attached=None)
    tool = create_image_generation_tool(
        {"sd": backend},
        context,
        default_model="sd",
    )

    result = await tool.ainvoke({
        "prompt": "make it sepia",
        "use_attached_image": True,
    })
    assert "[TOOL ERROR]" in result
    assert "attach an image first" in result
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
    result = await tool.ainvoke({
        "prompt": "x",
        "use_attached_image": True,
    })
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
    await tool.ainvoke({
        "prompt": "a city skyline",
        "model": "flux",
        "style": "vivid",
    })
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
