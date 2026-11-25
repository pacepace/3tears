"""the ONE module that reads CPython's private members: ``sys._current_frames``.

the loop watchdog (:mod:`threetears.observe.loop_watchdog`) must read the event loop thread's stack, and every other
thread's, from its own watchdog thread while the loop is held; ``sys._current_frames`` is CPython's only way to read
another thread's frames (``traceback`` and ``inspect`` reach the calling thread's alone). nothing else in the
workspace's src touches it; the paired entry is in ``tests/enforcement/_underscore_exemptions.txt``.
"""

from __future__ import annotations

import sys
from types import FrameType

__all__ = ["current_frames"]


def current_frames() -> dict[int, FrameType]:
    """every live thread's innermost frame, by thread ident, read now.

    :return: ``sys._current_frames()``: a new dict each call, the caller's own
    :rtype: dict[int, FrameType]
    """
    return sys._current_frames()
