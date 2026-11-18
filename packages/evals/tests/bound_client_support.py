"""What a test's completion-client double is besides its replies, so it is the protocol it stands in for."""

from __future__ import annotations

from typing import Any, Self

from threetears.evals.kernel.host.eval_host import CompletionRole
from threetears.evals.schema import BoundCompletionClient


class NoClient(RuntimeError):
    """A completion client was asked of a host whose test builds none."""


def no_clients(role: CompletionRole, model: str | None, *, temperature: float | None = None) -> BoundCompletionClient:
    """A host's completion clients where the test builds none: asking for one fails as a host that cannot build it
    does, so a dim that would be judged on it records its error and the run goes on.

    :raises NoClient: Always.
    """
    raise NoClient(f"a {role!r} client on {model!r} was asked for, and this test builds none")


class ClientDouble:
    """The members of :class:`~threetears.evals.schema.CompletionClient` a double answers the same way every time.

    A double subclasses it and writes its own ``generate``; there is nothing to release.
    """

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


class BoundClientDouble(ClientDouble):
    """A :class:`ClientDouble` that is a :class:`~threetears.evals.schema.BoundCompletionClient` too.

    ``price_ceiling`` answers ``None``, a client that cannot say, which the engine's ceiling reads exactly as it reads
    a client with no ``price_ceiling``; a double whose test prices its calls writes its own, and one whose test reads
    its model names it.
    """

    #: The model the double stands in for; a test reading it names its own.
    model_name: str = "test/double"

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """No ceiling: the double cannot price a call."""
        del system, user, response_format
        return None
