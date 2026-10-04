"""Release-contract support for test doubles standing in for an LLM client.

``threetears.evals.contracts.provider.CompletionClient`` declares ``aclose`` because the real
client owns an httpx pool and every caller that builds one per unit of work has to
release it. A double that omits it does not just fail — it fails
*late*, at whichever call site closes, with an ``AttributeError`` that reads as a
bug in the call site rather than a gap in the double.

Inheriting :class:`ReleasableClientMixin` supplies the method and, more usefully,
counts the calls: a test that asserts a call site releases needs the count, and
counting is the only way to tell "released once" from "released twice" from
"never released at all".
"""

from __future__ import annotations


class ReleasableClientMixin:
    """Gives a fake LLM client the release half of the client protocol.

    Attributes:
        aclose_calls: How many times ``aclose`` was awaited. Zero means the call
            site under test leaked; more than one is safe (the real ``aclose`` is
            idempotent) but usually means two owners think they own the client.
    """

    aclose_calls: int = 0

    async def aclose(self) -> None:
        """Record a release. Idempotent, like the real client's."""
        self.aclose_calls = self.aclose_calls + 1

    async def __aenter__(self):
        """Enter a releasing scope, mirroring the real client."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Release on the way out, raising body or not."""
        await self.aclose()


def releasable_client_mock():
    """A ``MagicMock`` LLM client whose ``aclose`` can actually be awaited.

    A bare ``MagicMock`` returns a ``MagicMock`` from ``aclose()``, which is not
    awaitable — so a call site that correctly releases its client fails against
    the mock rather than against reality. Returns a mock a test can also assert
    on: ``mock.aclose.assert_awaited_once()`` proves the site released.
    """
    from unittest.mock import AsyncMock, MagicMock

    return MagicMock(aclose=AsyncMock())
