"""A model's recording, made offline for a test: what :class:`~threetears.evals.quick.client.Model` replays.

A test of a program that calls ``Model(id, recording=...)`` needs the recording to exist. :func:`record_model` makes
one through the package's own :class:`~threetears.evals.testing.RecordedCompletions`, over a scripted client that
answers each prompt with what the test says, so the replay the program then makes is the replay a real recording
would give: keyed by the prompt, the directive :meth:`Model.json` sends and the call's index.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel

from threetears.evals.quick.client import json_directive
from threetears.evals.schema import StopReason
from threetears.evals.testing import RecordedCompletions

#: What each scripted call costs, as the recording keeps it.
CALL_COST_USD = 0.00001


@dataclass(frozen=True)
class ScriptedCompletion:
    """One scripted reply, in the attribute names the engine reads."""

    content: str
    model: str
    input_tokens: int | None = 40
    output_tokens: int | None = 12
    cost_usd: float | None = CALL_COST_USD
    price_source: str | None = "scripted rate"
    served_model: str | None = None
    reasoning_tokens: int | None = None
    stop_reason: StopReason = "end_turn"
    temperature: float | None = None


# parity-with: threetears.evals.schema.CompletionClient
class FakeScriptedModel:
    """A model whose every reply is what ``answer`` gives for the prompt."""

    def __init__(self, model: str, answer: Callable[[str], str]) -> None:
        self._model = model
        self._answer = answer

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> ScriptedCompletion:
        return ScriptedCompletion(content=self._answer(user), model=self._model, served_model=self._model)

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


async def record_model(
    path: Path,
    model: str,
    prompts: Sequence[str],
    answer: Callable[[str], str],
    *,
    schema: type[BaseModel] | None = None,
    repeats: int = 1,
) -> None:
    """Record ``model`` answering each prompt ``repeats`` times, as ``Model.text`` or ``Model.json`` asks it.

    Args:
        path: The recording to write, afresh.
        model: The model's id.
        prompts: Each prompt, as the program sends it.
        answer: The reply to a prompt.
        schema: The pydantic model ``Model.json`` is asked for; ``None`` for ``Model.text``.
        repeats: How many times each prompt is asked: an eval's ``k``.
    """
    recorder = RecordedCompletions.record(FakeScriptedModel(model, answer), path, settings=None, replace=True)
    directive = None if schema is None else json_directive(schema)
    for _ in range(repeats):
        for prompt in prompts:
            await recorder.generate(system="", user=prompt, response_format=directive)


__all__ = ["CALL_COST_USD", "FakeScriptedModel", "ScriptedCompletion", "record_model"]
