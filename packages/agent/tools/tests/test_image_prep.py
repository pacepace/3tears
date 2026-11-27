"""Tests for image preprocessing utility."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import subprocess
import sys
import textwrap

import pytest
from threetears.agent.tools.builtin.image_prep import ImagePrepError, ImagePrepTool, prepare_image_for_vision

#: the long-edge limit the module documents for every image it re-encodes -- under
#: Anthropic's 8192px cap with room for a base64 payload under 5MB.
MAX_DIMENSION = 4096

#: the byte size the module documents as its passthrough bound: a web-safe image at
#: or under it is returned untouched.
SIZE_THRESHOLD = 512 * 1024


def _prepared(data: bytes, mime_type: str) -> tuple[bytes, str]:
    """:func:`prepare_image_for_vision` on a loop of its own: these tests build and read their images off the loop,
    in the synchronous test body, and only the call runs on one."""
    return asyncio.run(prepare_image_for_vision(data, mime_type))


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

    def test_small_jpeg_passthrough(self) -> None:
        data = _make_jpeg(100, 100)
        assert len(data) < SIZE_THRESHOLD
        result, mime = _prepared(data, "image/jpeg")
        assert result is data  # exact same object
        assert mime == "image/jpeg"

    def test_small_png_passthrough(self) -> None:
        data = _make_png(100, 100)
        assert len(data) < SIZE_THRESHOLD
        result, mime = _prepared(data, "image/png")
        assert result is data
        assert mime == "image/png"


class TestResize:
    """Large images should be resized and re-encoded."""

    def test_oversized_width_resized(self) -> None:
        """A large JPEG (over SIZE_THRESHOLD) with oversized dimensions gets resized."""
        data = _make_jpeg(8000, 4000, quality=100)
        # Ensure it's over threshold — pad if needed
        if len(data) <= SIZE_THRESHOLD:
            # Solid-color compresses too well; use random noise
            data = _make_noisy_jpeg(8000, 4000)
        assert len(data) > SIZE_THRESHOLD
        result, mime = _prepared(data, "image/jpeg")
        assert mime == "image/jpeg"

        from PIL import Image

        img = Image.open(io.BytesIO(result))
        assert img.size[0] <= MAX_DIMENSION
        assert img.size[1] <= MAX_DIMENSION

    def test_oversized_height_resized(self) -> None:
        data = _make_noisy_jpeg(2000, 6000)
        assert len(data) > SIZE_THRESHOLD
        result, mime = _prepared(data, "image/jpeg")

        from PIL import Image

        img = Image.open(io.BytesIO(result))
        assert max(img.size) <= MAX_DIMENSION

    def test_within_limits_but_large_bytes_reencoded(self) -> None:
        """Image within dimension limits but over SIZE_THRESHOLD gets re-encoded."""
        # Create a large-ish JPEG by using high quality on a moderately sized image
        data = _make_jpeg(2000, 2000, quality=100)
        if len(data) <= SIZE_THRESHOLD:
            # Force it over threshold by padding (unlikely but defensive)
            data = data + b"\x00" * (SIZE_THRESHOLD + 1)
        result, mime = _prepared(data, "image/jpeg")
        assert mime == "image/jpeg"
        # Should be re-encoded (different bytes)
        assert result != data


class TestFormatConversion:
    """Non-JPEG formats should be converted."""

    def test_rgba_png_converted_to_jpeg(self) -> None:
        data = _make_png(100, 100, rgba=True)
        # Make it larger than threshold so it triggers processing
        big_data = data + b"\x00" * (SIZE_THRESHOLD + 1)
        result, mime = _prepared(big_data, "image/png")
        assert mime == "image/jpeg"

    def test_unknown_mime_processed(self) -> None:
        """HEIC or other non-web-safe types always trigger processing."""
        data = _make_jpeg(100, 100)  # valid JPEG bytes but labeled as heic
        result, mime = _prepared(data, "image/heic")
        assert mime == "image/jpeg"


class TestErrorHandling:
    """Small bytes labelled web-safe pass through unread, so even invalid ones come back as given."""

    def test_corrupt_data_returns_original(self) -> None:
        data = b"not an image at all"
        result, mime = _prepared(data, "image/jpeg")
        assert result is data
        assert mime == "image/jpeg"

    def test_empty_data_returns_original(self) -> None:
        data = b""
        result, mime = _prepared(data, "image/png")
        assert result is data
        assert mime == "image/png"


class TestAFailedResizeIsReported:
    """An image that needed processing and could not be processed is never handed back as prepared."""

    def test_corrupt_bytes_that_need_processing_raise_rather_than_come_back_as_prepared(self) -> None:
        data = b"not an image at all"
        with pytest.raises(ImagePrepError, match=r"image/heic, 19 bytes\) could not be resized") as caught:
            _prepared(data, "image/heic")
        assert caught.value.mime_type == "image/heic"
        assert caught.value.size_bytes == len(data)

    def test_a_missing_pillow_is_reported_rather_than_sending_the_original(self) -> None:
        """Pillow is imported at the module's top, so its absence is simulated in a fresh interpreter."""
        big = _make_jpeg(100, 100) + b"\x00" * (SIZE_THRESHOLD + 1)
        probe = textwrap.dedent(
            """
            import asyncio
            import json
            import sys
            sys.modules["PIL"] = None
            from threetears.agent.tools.builtin.image_prep import ImagePrepError, prepare_image_for_vision
            big = sys.stdin.buffer.read()
            try:
                asyncio.run(prepare_image_for_vision(big, "image/jpeg"))
            except ImagePrepError as exc:
                print(json.dumps({"refused": str(exc)}))
            else:
                print(json.dumps({"answered": True}))
            """
        )
        proc = subprocess.run([sys.executable, "-c", probe], input=big, capture_output=True, timeout=120, check=False)
        assert proc.returncode == 0, proc.stderr.decode()
        outcome = json.loads(proc.stdout.decode().strip().splitlines()[-1])
        assert "refused" in outcome, outcome
        assert "ModuleNotFoundError" in outcome["refused"]

    @pytest.mark.asyncio
    async def test_no_exception_message_reaches_the_agent(self) -> None:
        """the resize runs in a worker process, which nothing patched here reaches: Pillow's own words for bytes it
        cannot read are the message that must stay out."""
        result = await ImagePrepTool().execute(
            image_data_base64=base64.b64encode(b"x" * (SIZE_THRESHOLD + 1)).decode(), mime_type="image/jpeg"
        )
        assert result.success is False
        assert result.error is not None and "for a vision model: UnidentifiedImageError;" in result.error
        assert "cannot identify image file" not in (result.error or "") + result.content

    @pytest.mark.asyncio
    async def test_the_tool_reports_the_failure_and_that_the_original_is_unchanged(self) -> None:
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
    async def test_a_bug_in_the_decode_propagates_rather_than_reading_as_bad_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """it used to be contained and answered as undecodable base64, blaming the model."""

        def _broken(data: object) -> bytes:
            raise AttributeError("a bug in the decode")

        # the tool decodes through the ``base64`` module itself, the one patched here
        monkeypatch.setattr(base64, "b64decode", _broken)
        with pytest.raises(AttributeError, match="a bug in the decode"):
            await ImagePrepTool().execute(image_data_base64="aGVsbG8=", mime_type="image/png")
