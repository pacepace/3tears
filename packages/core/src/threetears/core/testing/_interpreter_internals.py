"""the one owner of the CPython private name the loop-stall guard depends on: ``sys._current_frames``.

:mod:`threetears.core.testing.loop_stall` names what held a test's event loop by sampling the loop thread's stack while
a long turn runs, from a thread of its own. Python has no public way to read another thread's current frame:
``sys._current_frames`` is documented ("for internal and specialized purposes only") and is what ``faulthandler``-free
samplers use. It is read with the interpreter lock held, so a thread cannot free its state under the read (the
lock-free all-threads walk of ``faulthandler`` can, which the SDK's loop watchdog avoids for that reason).

**Verified against CPython 3.14.** Moving to a new release: run ``packages/core/tests/unit/testing/test_loop_stall.py``
on it.
"""

from __future__ import annotations

import sys
from types import FrameType

__all__ = ["thread_frames"]


def thread_frames() -> dict[int, FrameType]:
    """every live thread's current frame, by thread id.

    :return: the frames, keyed by ``threading.get_ident()``
    :rtype: dict[int, FrameType]
    """
    return sys._current_frames()
