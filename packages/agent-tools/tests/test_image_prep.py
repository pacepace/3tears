"""Tests for image preprocessing utility."""

from __future__ import annotations

import io

import pytest

from threetears.agent.tools.builtin.image_prep import (
    _MAX_DIMENSION,
    _SIZE_THRESHOLD,
    prepare_image_for_vision,
)


def _make_jpeg(width: int, height: int, quality: int = 95) -> bytes:
    """Create a JPEG image of the given dimensions."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_noisy_jpeg(width: int, height: int) -> bytes:
    """Create a JPEG with random noise — compresses poorly, so it's large."""
    import numpy as np
    from PIL import Image

    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(noise)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_png(width: int, height: int, *, rgba: bool = False) -> bytes:
    """Create a PNG image."""
    from PIL import Image

    mode = "RGBA" if rgba else "RGB"
    img = Image.new(mode, (width, height), color=(0, 255, 0, 128) if rgba else (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestPassthrough:
    """Small, web-safe images should pass through unchanged."""

    def test_small_jpeg_passthrough(self):
        data = _make_jpeg(100, 100)
        assert len(data) < _SIZE_THRESHOLD
        result, mime = prepare_image_for_vision(data, "image/jpeg")
        assert result is data  # exact same object
        assert mime == "image/jpeg"

    def test_small_png_passthrough(self):
        data = _make_png(100, 100)
        assert len(data) < _SIZE_THRESHOLD
        result, mime = prepare_image_for_vision(data, "image/png")
        assert result is data
        assert mime == "image/png"


class TestResize:
    """Large images should be resized and re-encoded."""

    def test_oversized_width_resized(self):
        """A large JPEG (over SIZE_THRESHOLD) with oversized dimensions gets resized."""
        data = _make_jpeg(8000, 4000, quality=100)
        # Ensure it's over threshold — pad if needed
        if len(data) <= _SIZE_THRESHOLD:
            # Solid-color compresses too well; use random noise
            data = _make_noisy_jpeg(8000, 4000)
        assert len(data) > _SIZE_THRESHOLD
        result, mime = prepare_image_for_vision(data, "image/jpeg")
        assert mime == "image/jpeg"

        from PIL import Image
        img = Image.open(io.BytesIO(result))
        assert img.size[0] <= _MAX_DIMENSION
        assert img.size[1] <= _MAX_DIMENSION

    def test_oversized_height_resized(self):
        data = _make_noisy_jpeg(2000, 6000)
        assert len(data) > _SIZE_THRESHOLD
        result, mime = prepare_image_for_vision(data, "image/jpeg")

        from PIL import Image
        img = Image.open(io.BytesIO(result))
        assert max(img.size) <= _MAX_DIMENSION

    def test_within_limits_but_large_bytes_reencoded(self):
        """Image within dimension limits but over SIZE_THRESHOLD gets re-encoded."""
        # Create a large-ish JPEG by using high quality on a moderately sized image
        data = _make_jpeg(2000, 2000, quality=100)
        if len(data) <= _SIZE_THRESHOLD:
            # Force it over threshold by padding (unlikely but defensive)
            data = data + b"\x00" * (_SIZE_THRESHOLD + 1)
        result, mime = prepare_image_for_vision(data, "image/jpeg")
        assert mime == "image/jpeg"
        # Should be re-encoded (different bytes)
        assert result != data


class TestFormatConversion:
    """Non-JPEG formats should be converted."""

    def test_rgba_png_converted_to_jpeg(self):
        data = _make_png(100, 100, rgba=True)
        # Make it larger than threshold so it triggers processing
        big_data = data + b"\x00" * (_SIZE_THRESHOLD + 1)
        result, mime = prepare_image_for_vision(big_data, "image/png")
        assert mime == "image/jpeg"

    def test_unknown_mime_processed(self):
        """HEIC or other non-web-safe types always trigger processing."""
        data = _make_jpeg(100, 100)  # valid JPEG bytes but labeled as heic
        result, mime = prepare_image_for_vision(data, "image/heic")
        assert mime == "image/jpeg"


class TestErrorHandling:
    """Invalid data should fall back to returning original."""

    def test_corrupt_data_returns_original(self):
        data = b"not an image at all"
        result, mime = prepare_image_for_vision(data, "image/jpeg")
        assert result is data
        assert mime == "image/jpeg"

    def test_empty_data_returns_original(self):
        data = b""
        result, mime = prepare_image_for_vision(data, "image/png")
        assert result is data
        assert mime == "image/png"
