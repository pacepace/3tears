"""Image preprocessing for vision model APIs.

Resizes and re-encodes images that exceed vision model limits.
Anthropic/OpenAI vision APIs have constraints on image dimensions
and data size. iPhone camera captures (12-48MP) routinely exceed
these limits, causing opaque "Provider returned error" failures.

Limits applied:
- Max dimension: 4096px on the long edge (within Anthropic's 8192 cap)
- Output format: JPEG at quality 85
- HEIC/HEIF inputs are converted to JPEG automatically via Pillow
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.observe import get_logger

__all__ = [
    "ImagePrepTool",
    "prepare_image_for_vision",
]

_log = get_logger(__name__)

# Vision model dimension limit — Anthropic caps at 8192, but 4096 keeps
# base64 payloads well under 5MB for typical photographic content.
_MAX_DIMENSION = 4096

# JPEG quality for re-encoded images
_JPEG_QUALITY = 85

# Size threshold: only re-encode if raw bytes exceed this (512KB).
# Small images pass through untouched for exact fidelity.
_SIZE_THRESHOLD = 512 * 1024


def prepare_image_for_vision(
    data: bytes,
    mime_type: str,
) -> tuple[bytes, str]:
    """Prepare image bytes for a vision model API call.

    Returns ``(processed_bytes, effective_mime_type)``.  Small images
    that are already within limits are returned as-is.  Large images
    are resized and re-encoded as JPEG.

    :param data: raw image bytes
    :ptype data: bytes
    :param mime_type: original MIME type (e.g. ``image/jpeg``, ``image/heic``)
    :ptype mime_type: str
    :return: tuple of ``(bytes, mime_type)`` ready for base64 encoding
    """
    # Small images in web-safe formats: pass through unchanged
    if len(data) <= _SIZE_THRESHOLD and mime_type in (
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    ):
        return data, mime_type

    try:
        from PIL import Image

        opened = Image.open(io.BytesIO(data))

        # Convert to RGB for JPEG output
        img = opened.convert("RGB") if opened.mode != "RGB" else opened.copy()

        # Resize if either dimension exceeds the limit
        w, h = img.size
        if max(w, h) > _MAX_DIMENSION:
            ratio = _MAX_DIMENSION / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            _log.info(
                "Resized image for vision model",
                extra={
                    "extra_data": {
                        "original_size": f"{w}x{h}",
                        "new_size": f"{new_w}x{new_h}",
                        "original_bytes": len(data),
                    }
                },
            )

        # Re-encode as JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        result = buf.getvalue()

        _log.debug(
            "Image prepared for vision",
            extra={
                "extra_data": {
                    "original_bytes": len(data),
                    "original_mime": mime_type,
                    "output_bytes": len(result),
                    "dimensions": f"{img.size[0]}x{img.size[1]}",
                }
            },
        )

        return result, "image/jpeg"

    except Exception as exc:
        _log.warning(
            "Image preprocessing failed, sending original",
            extra={
                "extra_data": {
                    "error": str(exc),
                    "mime_type": mime_type,
                    "size_bytes": len(data),
                }
            },
        )
        return data, mime_type


class ImagePrepTool(TearsTool):
    """TearsTool wrapper for image preprocessing before vision model calls.

    accepts base64-encoded image bytes and MIME type, applies resizing
    and re-encoding as needed, and returns processed image ready for
    vision model APIs. requires Pillow for images exceeding size
    threshold.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "image_data_base64": {
                "type": "string",
                "description": "base64-encoded image bytes",
            },
            "mime_type": {
                "type": "string",
                "description": "MIME type of image (e.g. image/jpeg, image/png, image/heic)",
            },
        },
        "required": ["image_data_base64", "mime_type"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """preprocess image for vision model API call.

        :param kwargs: must include 'image_data_base64' and 'mime_type' keys
        :ptype kwargs: Any
        :return: result containing JSON with processed_base64 and effective_mime_type
        :rtype: ToolResult
        """
        image_data_base64 = kwargs.get("image_data_base64", "")
        mime_type = kwargs.get("mime_type", "")

        if not image_data_base64:
            result = ToolResult(
                success=False,
                content="",
                error="image_data_base64 is required",
            )
            return result

        if not mime_type:
            result = ToolResult(
                success=False,
                content="",
                error="mime_type is required",
            )
            return result

        try:
            raw_bytes = base64.b64decode(image_data_base64)
        except Exception as exc:
            result = ToolResult(
                success=False,
                content="",
                error=f"failed to decode base64 image data: {exc}",
            )
            return result

        processed_bytes, effective_mime = prepare_image_for_vision(raw_bytes, mime_type)
        processed_base64 = base64.b64encode(processed_bytes).decode("ascii")

        content = json.dumps(
            {
                "processed_base64": processed_base64,
                "effective_mime_type": effective_mime,
            }
        )

        result = ToolResult(
            success=True,
            content=content,
        )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for image prep.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="preprocess image for vision model API (resize, re-encode)",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.image_prep"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
