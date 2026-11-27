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
import json
from typing import Any, Final

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.image_resize import HAS_PILLOW, resize_for_vision
from threetears.core.process_workers import ProcessWorkers
from threetears.observe import contain, get_logger

__all__ = [
    "ImagePrepError",
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

#: the worker the decode, resize and encode run in. One: a tool pod prepares an image per vision call, each call then
#: waits seconds on the provider, so images queue behind one another for far less than the call they feed
_WORKERS: Final = ProcessWorkers(workers=1, name="vision image prep", process_name="3t-image-prep")


class ImagePrepError(RuntimeError):
    """an image that needed resizing or re-encoding could not be.

    raised in place of handing the original bytes back as though they were prepared: the
    original is exactly what the vision API was going to refuse, so a caller that sends it
    anyway has to know it is doing so. the cause (Pillow absent, a corrupt or unsupported
    image) is chained as ``__cause__``.

    :param mime_type: the image's declared MIME type
    :ptype mime_type: str
    :param size_bytes: the image's size
    :ptype size_bytes: int
    :param detail: why, as the exception's type name only -- this message reaches an agent,
        and an exception's message is not for it; the cause carries the message to the log
    :ptype detail: str
    """

    def __init__(self, mime_type: str, size_bytes: int, detail: str) -> None:
        self.mime_type = mime_type
        self.size_bytes = size_bytes
        self.detail = detail
        super().__init__(
            f"the image ({mime_type}, {size_bytes} bytes) could not be resized or re-encoded "
            f"for a vision model: {detail}"
        )


async def prepare_image_for_vision(
    data: bytes,
    mime_type: str,
) -> tuple[bytes, str]:
    """Prepare image bytes for a vision model API call.

    Returns ``(processed_bytes, effective_mime_type)``.  Small images
    that are already within limits are returned as-is.  Large images
    are resized and re-encoded as JPEG, in a worker process: the decode,
    resize and encode are CPU that grows with the image, never the event
    loop's (:mod:`threetears.agent.tools.image_resize`'s docstring).

    :param data: raw image bytes
    :ptype data: bytes
    :param mime_type: original MIME type (e.g. ``image/jpeg``, ``image/heic``)
    :ptype mime_type: str
    :return: tuple of ``(bytes, mime_type)`` ready for base64 encoding
    :raises ImagePrepError: the image needed processing and it failed (Pillow not
        installed, or the bytes are not an image Pillow reads)
    """
    # Small images in web-safe formats: pass through unchanged
    if len(data) <= _SIZE_THRESHOLD and mime_type in (
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    ):
        return data, mime_type

    if not HAS_PILLOW:
        # refused here, not in the worker: a worker is a fresh import, which may find a Pillow this process did not
        missing = ModuleNotFoundError("image resizing needs Pillow: install 3tears-agent-tools[vision]")
        raise ImagePrepError(mime_type, len(data), type(missing).__name__) from missing

    # always in the worker, never inline for a small image: a small one is still decoded (a 19-byte corrupt HEIC held
    # the loop 0.267 s, the first open importing Pillow's plugins), and the worker's round trip (about 0.4
    # ms, keyed/binding.py's measure) is nothing beside the vision call that follows
    try:
        result, w, h, new_w, new_h = await _WORKERS.run(resize_for_vision, data, _MAX_DIMENSION, _JPEG_QUALITY)
    except Exception as exc:  # noqa: BLE001 -- any failure is refused by name; the caller decides whether to send the original
        raise ImagePrepError(mime_type, len(data), type(exc).__name__) from exc

    if (new_w, new_h) != (w, h):
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
    _log.debug(
        "Image prepared for vision",
        extra={
            "extra_data": {
                "original_bytes": len(data),
                "original_mime": mime_type,
                "output_bytes": len(result),
                "dimensions": f"{new_w}x{new_h}",
            }
        },
    )
    return result, "image/jpeg"


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
            "image_data_base64": {"type": "string"},
            "mime_type": {
                "type": "string",
                "description": "e.g. image/jpeg, image/png, image/heic.",
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
        except (ValueError, TypeError) as exc:
            # ValueError is bad base64 (binascii.Error) or a str with non-ASCII characters. TypeError
            # is a value that is neither str nor bytes: TearsTool.run does not type-check kwargs, so
            # a model that sends a number or an object here has sent bad input, not hit a bug.
            contain(exc, log=_log, what="failed to decode base64 image data")
            result = ToolResult(
                success=False,
                content="",
                error=f"failed to decode base64 image data: {exc}",
            )
            return result

        try:
            processed_bytes, effective_mime = await prepare_image_for_vision(raw_bytes, mime_type)
        except ImagePrepError as exc:
            contain(exc, log=_log, what="image could not be prepared for a vision model")
            result = ToolResult(
                success=False,
                content="",
                error=f"{exc}; nothing was resized, and the original image is unchanged",
            )
            return result
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
