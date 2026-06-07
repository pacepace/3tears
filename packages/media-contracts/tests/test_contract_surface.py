"""smoke tests for the media contracts public surface."""

from __future__ import annotations

import threetears.media.contracts as contracts


class TestPublicSurface:
    def test_all_symbols_importable(self) -> None:
        for name in contracts.__all__:
            assert getattr(contracts, name) is not None

    def test_generated_image_is_a_dataclass(self) -> None:
        image = contracts.GeneratedImage(data=b"png-bytes", mime_type="image/png")
        assert image.width is None
        assert image.metadata is None

    def test_backend_protocol_is_runtime_checkable(self) -> None:
        class _Backend:
            async def generate(
                self,
                prompt: str,
                *,
                style: str | None = None,
                source_image: bytes | None = None,
                source_mime_type: str | None = None,
            ) -> contracts.GeneratedImage:
                return contracts.GeneratedImage(data=b"", mime_type="image/png")

        assert isinstance(_Backend(), contracts.ImageGenerationBackend)
