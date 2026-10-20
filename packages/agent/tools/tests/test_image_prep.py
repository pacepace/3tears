"""Tests for image preprocessing utility."""

from __future__ import annotations

import io

import base64
import sys

import pytest

from threetears.agent.tools.builtin.image_prep import ImagePrepError, ImagePrepTool, prepare_image_for_vision

#: the long-edge limit the module documents for every image it re-encodes -- under
#: Anthropic's 8192px cap with room for a base64 payload under 5MB.
MAX_DIMENSION = 4096

#: the byte size the module documents as its passthrough bound: a web-safe image at
#: or under it is returned untouched.
SIZE_THRESHOLD = 512 * 1024


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
        assert len(data) < SIZE_THRESHOLD
        result, mime = prepare_image_for_vision(data, "image/jpeg")
        assert result is data  # exact same object
        assert mime == "image/jpeg"

    def test_small_png_passthrough(self):
        data = _make_png(100, 100)
        assert len(data) < SIZE_THRESHOLD
        result, mime = prepare_image_for_vision(data, "image/png")
        assert result is data
        assert mime == "image/png"


class TestResize:
    """Large images should be resized and re-encoded."""

    def test_oversized_width_resized(self):
        """A large JPEG (over SIZE_THRESHOLD) with oversized dimensions gets resized."""
        data = _make_jpeg(8000, 4000, quality=100)
        # Ensure it's over threshold — pad if needed
        if len(data) <= SIZE_THRESHOLD:
            # Solid-color compresses too well; use random noise
            data = _make_noisy_jpeg(8000, 4000)
        assert len(data) > SIZE_THRESHOLD
        result, mime = prepare_image_for_vision(data, "image/jpeg")
        assert mime == "image/jpeg"

        from PIL import Image

        img = Image.open(io.BytesIO(result))
        assert img.size[0] <= MAX_DIMENSION
        assert img.size[1] <= MAX_DIMENSION

    def test_oversized_height_resized(self):
        data = _make_noisy_jpeg(2000, 6000)
        assert len(data) > SIZE_THRESHOLD
        result, mime = prepare_image_for_vision(data, "image/jpeg")

        from PIL import Image

        img = Image.open(io.BytesIO(result))
        assert max(img.size) <= MAX_DIMENSION

    def test_within_limits_but_large_bytes_reencoded(self):
        """Image within dimension limits but over SIZE_THRESHOLD gets re-encoded."""
        # Create a large-ish JPEG by using high quality on a moderately sized image
        data = _make_jpeg(2000, 2000, quality=100)
        if len(data) <= SIZE_THRESHOLD:
            # Force it over threshold by padding (unlikely but defensive)
            data = data + b"\x00" * (SIZE_THRESHOLD + 1)
        result, mime = prepare_image_for_vision(data, "image/jpeg")
        assert mime == "image/jpeg"
        # Should be re-encoded (different bytes)
        assert result != data


class TestFormatConversion:
    """Non-JPEG formats should be converted."""

    def test_rgba_png_converted_to_jpeg(self):
        data = _make_png(100, 100, rgba=True)
        # Make it larger than threshold so it triggers processing
        big_data = data + b"\x00" * (SIZE_THRESHOLD + 1)
        result, mime = prepare_image_for_vision(big_data, "image/png")
        assert mime == "image/jpeg"

    def test_unknown_mime_processed(self):
        """HEIC or other non-web-safe types always trigger processing."""
        data = _make_jpeg(100, 100)  # valid JPEG bytes but labeled as heic
        result, mime = prepare_image_for_vision(data, "image/heic")
        assert mime == "image/jpeg"


class TestErrorHandling:
    """Small bytes labelled web-safe pass through unread, so even invalid ones come back as given."""

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


class TestAFailedResizeIsReported:
    """An image that needed processing and could not be processed is never handed back as prepared."""

    def test_corrupt_bytes_that_need_processing_raise_rather_than_come_back_as_prepared(self):
        data = b"not an image at all"
        with pytest.raises(ImagePrepError, match=r"image/heic, 19 bytes\) could not be resized") as caught:
            prepare_image_for_vision(data, "image/heic")
        assert caught.value.mime_type == "image/heic"
        assert caught.value.size_bytes == len(data)

    def test_a_missing_pillow_is_reported_rather_than_sending_the_original(self, monkeypatch):
        big = _make_jpeg(100, 100) + b"\x00" * (SIZE_THRESHOLD + 1)
        monkeypatch.setitem(sys.modules, "PIL", None)
        with pytest.raises(ImagePrepError, match="ModuleNotFoundError"):
            prepare_image_for_vision(big, "image/jpeg")

    @pytest.mark.asyncio
    async def test_no_exception_message_reaches_the_agent(self, monkeypatch):
        from PIL import Image

        def _planted(*args, **kwargs):
            raise OSError("PLANTED-1a9f cannot identify image file")

        monkeypatch.setattr(Image, "open", _planted)
        result = await ImagePrepTool().execute(
            image_data_base64=base64.b64encode(b"x" * (SIZE_THRESHOLD + 1)).decode(), mime_type="image/jpeg"
        )
        assert result.success is False
        assert result.error is not None and "for a vision model: OSError;" in result.error
        assert "PLANTED-1a9f" not in (result.error or "") + result.content

    @pytest.mark.asyncio
    async def test_the_tool_reports_the_failure_and_that_the_original_is_unchanged(self):
        result = await ImagePrepTool().execute(
            image_data_base64=base64.b64encode(b"not an image at all").decode(), mime_type="image/heic"
        )
        assert result.success is False
        assert result.error is not None
        assert "could not be resized or re-encoded for a vision model" in result.error
        assert "the original image is unchanged" in result.error


class TestInputThatIsNotBase64:
    """Input the model got wrong is refused by name; a bug in the decode propagates."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("encoded", ["abc", "ünïcode"])
    async def test_a_string_that_is_not_base64_is_refused(self, encoded: str) -> None:
        result = await ImagePrepTool().execute(image_data_base64=encoded, mime_type="image/png")
        assert result.success is False
        assert result.error is not None and result.error.startswith("failed to decode base64 image data")

    @pytest.mark.asyncio
    @pytest.mark.expect_contained(
        TypeError, why="TearsTool.run does not type-check kwargs, so a number from the model is bad input"
    )
    async def test_a_value_that_is_not_a_string_is_refused(self) -> None:
        result = await ImagePrepTool().execute(image_data_base64=12345, mime_type="image/png")
        assert result.success is False
        assert result.error is not None and result.error.startswith("failed to decode base64 image data")

    @pytest.mark.asyncio
    async def test_a_bug_in_the_decode_propagates_rather_than_reading_as_bad_input(self, monkeypatch) -> None:
        """it used to be contained and answered as undecodable base64, blaming the model."""
        from threetears.agent.tools.builtin import image_prep

        def _broken(data: object) -> bytes:
            raise AttributeError("a bug in the decode")

        monkeypatch.setattr(image_prep.base64, "b64decode", _broken)
        with pytest.raises(AttributeError, match="a bug in the decode"):
            await ImagePrepTool().execute(image_data_base64="aGVsbG8=", mime_type="image/png")
