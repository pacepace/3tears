"""A model's recording, made offline for a test: what :class:`~threetears.evals.providers.Model` replays.

A test of a program that calls ``Model(id, recording=...)`` needs the recording to exist. :func:`record_model` makes
one with :class:`~threetears.evals.providers.Model` itself, recording live into the file over a scripted HTTP
transport that answers each prompt with what the test says, in OpenRouter's own reply shape. So the replay the
program then makes is the replay a real recording gives: keyed by the prompt, the directive ``Model.json`` sends, the
request settings the model sends and the call's index, and nothing is sent anywhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from threetears.evals.kernel import strict_response_format
from threetears.evals.providers import Catalogue, Model

#: What each scripted call costs, as OpenRouter's reply reports it and the recording keeps it.
CALL_COST_USD = 0.00001


def _scripted(model: str, answer: Callable[[str], str]) -> httpx.AsyncClient:
    """An HTTP client whose provider answers each chat request with ``answer`` of its user message, and whose
    catalogue lists no model (so nothing is priced from it: OpenRouter's reply carries the cost)."""

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        body = json.loads(request.content)
        user = next(message["content"] for message in body["messages"] if message["role"] == "user")
        reply: dict[str, Any] = {  # Any: JSON
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": answer(user)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12, "cost": CALL_COST_USD, "is_byok": False},
        }
        return httpx.Response(200, json=reply)

    return httpx.AsyncClient(transport=httpx.MockTransport(respond))


async def record_model(
    path: Path,
    model: str,
    prompts: Sequence[str],
    answer: Callable[[str], str],
    *,
    schema: type[BaseModel] | None = None,
    repeats: int = 1,
) -> None:
    """Record ``model`` (an OpenRouter ``vendor/model`` id) answering each prompt ``repeats`` times, afresh.

    Args:
        path: The recording to write; one already there is replaced.
        model: The model's id.
        prompts: Each prompt, as the program sends it.
        answer: The reply to a prompt.
        schema: The pydantic model ``Model.json`` is asked for; ``None`` for ``Model.text``.
        repeats: How many times each prompt is asked: an eval's ``k``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    http = _scripted(model, answer)
    directive = None if schema is None else strict_response_format(schema.__name__, schema.model_json_schema())
    try:
        recorder = Model(model, api_key="scripted", recording=path, http=http, catalogue=Catalogue(http=http))
        for _ in range(repeats):
            for prompt in prompts:
                await recorder.generate(system="", user=prompt, response_format=directive)
        await recorder.aclose()
    finally:
        await http.aclose()


__all__ = ["CALL_COST_USD", "record_model"]
