"""The one stand-in for a cross-pod pacer, shared by every suite that needs one.

Extracted because there were two. `test_robots.py` had `_FakeDelayPacer` at module level, where
the fake-parity walker enforces its declaration; `test_tool.py` grew a second copy nested inside
a helper method, where the walker never looks -- so its `# parity-with:` marker was decorative
and a drifted `claim` signature passed unnoticed. Verified by drifting it and watching the
enforcement suite stay green.

Nesting is the trap: the marker looks identical at both sites and only one of them is checked.
One module-level definition removes the choice.

What the walker enforces here, confirmed by breaking each in turn: removing the marker raises
`fake_parity.no_declaration`, and removing `claim` raises `fake_parity.method_missing`. It does
NOT check the signature -- a drifted `claim(self, key)` passes -- so the marker buys "this fake
still has the methods the protocol declares", not "it still matches them". Worth knowing before
trusting it to catch a parameter that quietly disappears.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

__all__ = ["_FakeDelayPacer"]


# parity-with: threetears.core.coordination.token_bucket.TokenBucket
class _FakeDelayPacer:
    """The one method `RobotsGate` calls on a pacer, so drift in its signature fails here."""

    def __init__(self, *, claimed: bool = True, retry_after_seconds: float = 0.0) -> None:
        self._claimed = claimed
        self._retry_after = retry_after_seconds
        self.keys: list[str] = []

    async def claim(self, key: str = "default", *, tokens: float = 1.0, max_wait_seconds: float = 0.0) -> Any:
        self.keys.append(key)
        return SimpleNamespace(claimed=self._claimed, retry_after_seconds=self._retry_after, tokens_remaining=0.0)
