"""A scripted ``analyst`` client for the toy host: the report writer's turns, as ``analysis_generate``'s job runs them.

``analysis_generate`` writes a campaign's report on the client the host returns for the ``analyst`` role, which must be
a turn model. :class:`FakeAnalystClient` is one: each named session's turns are scripted replies (by default the
analyst submits :data:`TOY_DOC`, a document that passes validity and every invariant on the toy corpus campaign's
snapshot, and the first review clears it), and it is released as every role's client is. Nothing reaches a model:
every tool call the sessions make runs live on the frozen snapshot through the analyst tool, as the job mounts it.

Standalone: it imports no other test support, so ``ops_support`` builds the toy host's ``analyst`` client from it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final, Self

from threetears.evals.schema.completion import CompletionResult
from threetears.evals.schema.turns import (
    ToolResultClearing,
    ToolTurnClient,
    TurnMessage,
    TurnReply,
    TurnText,
    TurnToolSpec,
    TurnToolUse,
)

__all__ = [
    "ANALYST_MODEL",
    "RUBRIC",
    "TOY_DOC",
    "FakeAnalystClient",
    "FakeSessionClient",
    "passing_review",
    "scripted_reply",
    "submission",
]

#: The model id every scripted turn names.
ANALYST_MODEL: Final = "toy-analyst-model"

#: The rubric's items as ``editor.md`` names them, in order.
RUBRIC: Final = (
    "First screen",
    "Coherent story",
    "Honest emphasis",
    "Sound elision",
    "Confounds handled",
    "Actionable next steps",
    "Evidence fits claims",
)

#: A report document on the toy corpus campaign (two token budgets on one aspect, no control), valid on its snapshot
#: and passing every invariant: it names no figure, recommends no arm, and marks its one claim durable.
TOY_DOC: Final[dict[str, Any]] = {
    "schema": "report-doc/1",
    "story": "status",
    "headline": "The two token budgets read alike on invoice extraction.",
    "lede": "Neither arm is shown ahead on this campaign's evidence.",
    "flag": None,
    "recommends": [],
    "confidence": {
        "form": "low",
        "caveats": [
            {
                "id": "judge_uncalibrated",
                "label": None,
                "material_if": "only if the judge grades the two arms differently",
                "checked": None,
            },
            {
                "id": "no_control",
                "label": None,
                "material_if": "only if a baseline would read differently",
                "checked": None,
            },
            {
                "id": "nothing_declared",
                "label": None,
                "material_if": "only if an owner line sits between the arms",
                "checked": None,
            },
        ],
    },
    "decision": [{"slot": "recommendation", "text": "No call: run more cases before choosing a budget."}],
    "claims": [
        {
            "id": "c1",
            "text": "The two token budgets read alike on invoice extraction.",
            "weight": "lead",
            "evidence": [
                {
                    "block": "table",
                    "query": {
                        "source": "measures",
                        "aspect": "extract_invoice_fields",
                        "measures": [],
                        "arms": [],
                        "strata": [],
                        "limit": 0,
                        "transform": "none",
                        "small_multiples_by": "none",
                    },
                    "reference": None,
                    "emphasis": "",
                    "caption": "Each arm's measures.",
                }
            ],
            "caveats": [],
            "durable": True,
        }
    ],
    "explore": [],
    "omitted": [],
    "next": [],
    "method": [{"term": "Arms", "text": "Two token budgets.", "caveats": []}],
}


def scripted_reply(
    *content: TurnText | TurnToolUse, cost: float | None = 0.01, model: str = ANALYST_MODEL
) -> TurnReply:
    """A scripted reply: a tool-use stop when it calls a tool, else an end of turn."""
    calls = any(isinstance(block, TurnToolUse) for block in content)
    return TurnReply(
        content=content,
        stop_reason="tool_use" if calls else "end_turn",
        provider_stop_reason=None,
        model=model,
        served_model=f"{model}-20261001",
        input_tokens=1000,
        output_tokens=50,
        reasoning_tokens=20,
        cache_read_tokens=800,
        cache_write_tokens=100,
        cost_usd=cost,
        price_source=None if cost is None else "catalogue",
        pricing_failure="no rate for the model" if cost is None else None,
    )


def submission(doc: Mapping[str, Any] = TOY_DOC) -> TurnText:
    """An analyst's submission: one fenced JSON object."""
    return TurnText(f"Here is the report.\n```json\n{json.dumps(doc)}\n```")


def passing_review(
    scores: Sequence[int] = (2, 2, 2, 2, 2, 1, 2), findings: Sequence[Mapping[str, str]] = ()
) -> TurnText:
    """An editor's final message: a line of prose, then its review as one fenced JSON object."""
    review = {
        "items": [
            {"n": n, "name": name, "score": score, "reason": f"item {n} reads so"}
            for n, (name, score) in enumerate(zip(RUBRIC, scores, strict=True), start=1)
        ],
        "findings": list(findings),
    }
    return TurnText(f"My review.\n```json\n{json.dumps(review)}\n```")


# parity-with: threetears.evals.schema.turns.ToolTurnClient
class FakeSessionClient:
    """One session's turns: each the next scripted reply (or exception), after the gate opens when one is set."""

    def __init__(
        self, replies: Sequence[TurnReply | Exception], *, gate: asyncio.Event | None, ceiling: float | None
    ) -> None:
        self.replies = list(replies)
        self.gate = gate
        self.ceiling = ceiling
        self.sent = 0

    async def turn(
        self,
        *,
        system: str,
        messages: Sequence[TurnMessage],
        tools: Sequence[TurnToolSpec],
        max_tokens: int,
        clearing: ToolResultClearing | None = None,
    ) -> TurnReply:
        if self.gate is not None:
            await self.gate.wait()
        self.sent += 1
        if not self.replies:
            raise AssertionError(f"the session sent turn {self.sent}, which the script does not hold")
        scripted = self.replies.pop(0)
        if isinstance(scripted, Exception):
            raise scripted
        return scripted

    def turn_price_ceiling(
        self, *, system: str, messages: Sequence[TurnMessage], tools: Sequence[TurnToolSpec], max_tokens: int
    ) -> float | None:
        return self.ceiling

    async def refresh_prices(self) -> None:
        return None


def _default_script(model: str) -> dict[str, list[TurnReply | Exception]]:
    return {
        "analyst": [scripted_reply(submission(), model=model)],
        "editor-1": [scripted_reply(passing_review(), model=model)],
    }


# parity-with: threetears.evals.ops.report_writer.loop.TurnModel
class FakeAnalystClient:
    """The toy host's ``analyst`` client: a turn model whose named sessions are scripted, released as a client is.

    :ivar sessions: Each named session's client, as the writer asked for it.
    :ivar released: Whether the job released it.
    """

    def __init__(
        self,
        script: Mapping[str, Sequence[TurnReply | Exception]] | None = None,
        *,
        model_name: str = ANALYST_MODEL,
        gate: asyncio.Event | None = None,
        unpriced: str | None = None,
        ceiling: float | None = 0.01,
    ) -> None:
        self.model_name = model_name
        self.script = {name: list(replies) for name, replies in (script or _default_script(model_name)).items()}
        self.gate = gate
        self.unpriced = unpriced
        self.ceiling = ceiling
        self.sessions: dict[str, FakeSessionClient] = {}
        self.released = False

    async def turn_pricing_refusal(self) -> str | None:
        return self.unpriced

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        if name not in self.script:
            raise AssertionError(f"session {name!r} was asked for, and the script holds none")
        client = FakeSessionClient(self.script[name], gate=self.gate, ceiling=self.ceiling)
        self.sessions[name] = client
        return client

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        return self.ceiling

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        raise AssertionError("the analyst role writes in turns; nothing asks it for a completion")

    async def aclose(self) -> None:
        self.released = True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()
