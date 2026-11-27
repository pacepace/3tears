"""An image decoded, resized and re-encoded as JPEG in a worker process: the CPU half of preparing an image for a
vision model (:func:`threetears.agent.tools.builtin.image_prep.prepare_image_for_vision`).

**Why a worker.** Decoding, resizing and re-encoding grow with the image (a 48 MP phone capture), and on the event
loop even a corrupt 19-byte one held it 0.267 s: Pillow's first ``Image.open`` in a process imports every format
plugin (``PIL.Image.init``), Python code under the interpreter lock (the loop-stall guard on the testament pool,
``test_analyze_media``, 2026-11-26). A thread would hold the lock through that import and through every pure-Python
step around Pillow's kernels, so the work runs in a worker process
(:class:`~threetears.core.process_workers.ProcessWorkers`): bytes in, bytes out.

This module imports only Pillow, so the worker that imports it stays small: never the tool's agent and storage stack.
"""

from __future__ import annotations

import io
from typing import Final

try:
    from PIL import Image

    #: whether Pillow is installed: the vision extra is optional
    HAS_PILLOW: Final[bool] = True
except ImportError:  # NOSILENT: the vision extra is optional; the caller refuses a resize naming it, sending no job
    HAS_PILLOW = False  # type: ignore[misc]

__all__ = ["HAS_PILLOW", "resize_for_vision"]


def resize_for_vision(data: bytes, max_dimension: int, quality: int) -> tuple[bytes, int, int, int, int]:
    """``data`` decoded, its long edge brought to at most ``max_dimension`` and re-encoded as JPEG at ``quality``.

    A worker process's job: plain data in and out. What fails is raised, and reaches the caller as the worker raised
    it (:meth:`~threetears.core.process_workers.ProcessWorkers.run`), its traceback chained.

    :param data: the image's bytes
    :ptype data: bytes
    :param max_dimension: the longest edge the result may have, in pixels
    :ptype max_dimension: int
    :param quality: the JPEG quality
    :ptype quality: int
    :return: ``(jpeg, width, height, new_width, new_height)``: the sizes are the decoded image's and the result's
    :rtype: tuple[bytes, int, int, int, int]
    :raises Exception: what Pillow raised for bytes it cannot read, or could not resize or encode
    """
    opened = Image.open(io.BytesIO(data))
    img = opened.convert("RGB") if opened.mode != "RGB" else opened.copy()
    width, height = img.size
    if max(width, height) > max_dimension:
        ratio = max_dimension / max(width, height)
        img = img.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), width, height, img.size[0], img.size[1]
