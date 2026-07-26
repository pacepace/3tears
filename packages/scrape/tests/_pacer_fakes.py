"""The one stand-in for a cross-pod pacer, shared by every suite that needs one.

Extracted because there were two, and only one of them was checked. `test_robots.py` had
`_FakeDelayPacer`; `test_tool.py` grew a second copy called `_Pacer`, and the fake-parity
walker filters on the NAME alone -- `_FAKE_NAME_PREFIXES` is `("Fake", "_Fake")` and is its
only test -- so `_Pacer` was invisible and its `# parity-with:` marker checked nothing.

Confirmed in both directions rather than reasoned about: a module-level class with a non-`Fake`
name is not seen, and a `Fake`-named class nested inside a `@staticmethod` IS seen, because
`find_fakes_in_tree` uses `ast.walk` and never considers nesting. An earlier version of this
docstring blamed the nesting, which is backwards, and would have taught the next reader to move
a fake that was already fine while leaving a misnamed one unguarded.

What the marker buys, each confirmed by breaking it: removing the marker raises
`fake_parity.no_declaration`; removing `claim` raises `fake_parity.method_missing`; and a
required parameter that disappears raises `fake_parity.method_required_arg_missing`. That last
one cannot fire for THIS fake, because every parameter of `TokenBucket.claim` has a default --
so here, and only here, a drifted `claim(self, key)` passes clean.
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
