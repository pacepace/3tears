"""A completion client that hands back one prepared completion and records what it was asked: the model, fixtured.

Harness support, not part of the toy host. A test that drives a path calling a model through a host's completion
clients hands this in, prices every call at :data:`FIXTURED_CALL_CEILING_USD` unless it says otherwise, and reads back
the calls, the prices asked and the releases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["FIXTURED_CALL_CEILING_USD", "MODEL", "FixturedClient", "FixturedCompletion"]

#: The model the fixtured client reports itself bound to.
MODEL = "anthropic/claude-opus"


@dataclass
class FixturedCompletion:
    """The subset of an LLM result the generator reads, with a canned body."""

    content: str
    cost_usd: float | None = 0.02
    # ``None`` is a count the provider did not report, as ``CompletionResult`` allows.
    input_tokens: int | None = 1200
    output_tokens: int | None = 800
    model: str = MODEL
    stop_reason: str = "end_turn"
    temperature: float | None = None
    tool_calls: list[Any] = field(default_factory=list)
    reasoning_tokens: int | None = None


#: What every fixtured call is priced at, at most: above the completion's reported cost, well under any toy cap.
FIXTURED_CALL_CEILING_USD = 0.05


class FixturedClient:
    """A client that returns one prepared completion and records what it was asked."""

    def __init__(self, body: str) -> None:
        """Prepare the completion.

        :param body: The JSON the generator will parse.
        :ptype body: str
        """
        self.completion = FixturedCompletion(content=body)
        self.calls: list[dict[str, str]] = []
        #: The response format each call sent, in the order of :attr:`calls`.
        self.response_formats: list[Any] = []
        self.closed = 0
        #: What :meth:`price_ceiling` answers for every call; ``None`` is a client that cannot price.
        self.ceiling_usd: float | None = FIXTURED_CALL_CEILING_USD
        #: Every prompt pair priced, in order.
        self.priced: list[dict[str, str]] = []

    @property
    def model_name(self) -> str:
        """The model this client is bound to -- the port the service reads back."""
        return MODEL

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        """The fixtured ceiling of one call — what a generation is priced against its out-of-run cap by."""
        self.priced.append({"system": system, "user": user})
        return self.ceiling_usd

    async def generate(
        self, *, system: str, user: str, response_format: Any = None, tools: Any = None
    ) -> FixturedCompletion:
        """Record the call and hand back the prepared completion."""
        self.calls.append({"system": system, "user": user})
        self.response_formats.append(response_format)
        return self.completion

    async def aclose(self) -> None:
        """Release nothing: a fixtured client owns no transport. Counted, so a test can see it was released."""
        self.closed += 1

    async def __aenter__(self) -> FixturedClient:
        """Enter a scope that releases on exit, as the service enters every client it builds."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Release on the way out."""
        await self.aclose()
