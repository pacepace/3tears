"""The report writer's tests' support: scripted model turns, the Kairo fixture host, and the recorded-turn replay.

Every turn goes through a scripted :class:`FakeToolTurnClient` (one per named session, handed out by
:class:`FakeTurnModel`) or a recording (:class:`RecordedModel`); nothing reaches a model. Every tool call runs live on
the fixture snapshots through the analyst tool, as ``write_report`` mounts it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from packages.evals.tests.fixtures.report_plans.analyst_cli import seeded_host, snapshot_ids
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_doc_support import converted_doc
from threetears.evals.actions.report_writing import ReportWritten, write_report
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.loop import WriterLimits
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.schema.turns import (
    ThinkingBinding,
    ToolResultClearing,
    ToolTurnClient,
    TurnBlock,
    TurnCeiling,
    TurnMessage,
    TurnReply,
    TurnReplyRefused,
    TurnStopReason,
    TurnText,
    TurnThinking,
    TurnToolSpec,
    TurnToolUse,
)
from threetears.evals.testing import RecordedTurns

__all__ = [
    "DOC",
    "MODEL",
    "RUBRIC",
    "FakeToolTurnClient",
    "FakeTurnModel",
    "RecordedModel",
    "answer",
    "call",
    "kairo_id",
    "replay_cases",
    "reply",
    "review",
    "submitted",
    "thinking",
    "written",
]

#: The model id every scripted reply names.
MODEL: Final = "report-writer-model"

#: The converted Kairo document of the third prototype, which passes validity and every invariant on Kairo's snapshot.
DOC: Final[dict[str, Any]] = converted_doc("prototype3/kairo")

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

#: Where the hand-scripted recordings live, each a directory of ``turns.jsonl`` and ``replay.json``.
HAND_RECORDINGS: Final = Path(__file__).parent / "fixtures" / "report_writer"

#: Where the acceptance run's recordings land (``analyst_cli.py write --out``).
ACCEPTED_RECORDINGS: Final = Path(__file__).parent / "fixtures" / "report_docs" / "prototype5"


def kairo_id() -> str:
    """Kairo round 2's snapshot id, as the seeded host holds it."""
    return snapshot_ids()["kairo"]


Request = dict[str, Any]


# parity-with: threetears.evals.schema.turns.ToolTurnClient
class FakeToolTurnClient:
    """Answers each turn with the next scripted reply, and keeps every request body it was sent."""

    def __init__(
        self,
        *replies: TurnReply | TurnReplyRefused | Exception,
        ceiling: float | None = 0.01,
        estimate: bool = False,
    ) -> None:
        self.replies = list(replies)
        self.ceiling = ceiling
        self.estimate = estimate
        self.ceilings: dict[int, float | None] = {}
        self.requests: list[Request] = []
        self.refreshed = 0

    async def turn(
        self,
        *,
        system: str,
        messages: Sequence[TurnMessage],
        tools: Sequence[TurnToolSpec],
        max_tokens: int,
        clearing: ToolResultClearing | None = None,
        binding: ThinkingBinding | None = None,
    ) -> TurnReply:
        self.requests.append(
            {
                "system": system,
                "messages": [message.to_json() for message in messages],
                "tools": [tool.to_json() for tool in tools],
                "max_tokens": max_tokens,
                "context_management": clearing.to_json() if clearing is not None else None,
                "block_binding": binding.to_json() if binding is not None else None,
            }
        )
        if not self.replies:
            raise AssertionError(f"the session sent turn {len(self.requests)}, which the script does not hold")
        scripted = self.replies.pop(0)
        if isinstance(scripted, Exception):
            raise scripted
        return scripted

    def turn_price_ceiling(
        self, *, system: str, messages: Sequence[TurnMessage], tools: Sequence[TurnToolSpec], max_tokens: int
    ) -> TurnCeiling | None:
        usd = self.ceilings.get(len(self.requests), self.ceiling)
        return TurnCeiling(usd=usd, estimate=self.estimate) if usd is not None else None

    async def refresh_prices(self) -> None:
        self.refreshed += 1


# parity-with: threetears.evals.ops.report_writer.loop.TurnModel
class FakeTurnModel:
    """A model whose named sessions are scripted clients; a session the script lacks is refused by name."""

    model_name = MODEL

    def __init__(self, *, unpriced: str | None = None, **sessions: FakeToolTurnClient) -> None:
        self.sessions = {name.replace("_", "-"): client for name, client in sessions.items()}
        self.unpriced = unpriced
        self.asked: list[str] = []
        self.snapshots: list[str] = []

    async def turn_pricing_refusal(self) -> str | None:
        return self.unpriced

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        self.asked.append(name)
        self.snapshots.append(snapshot)
        if name not in self.sessions:
            raise AssertionError(f"session {name!r} was asked for, and the script holds none")
        return self.sessions[name]


@dataclass(frozen=True)
class RecordedModel:
    """A model whose named sessions are a turn recording's, replayed: :class:`RecordedTurns` with its model's id."""

    recording: RecordedTurns
    model_name: str

    async def turn_pricing_refusal(self) -> str | None:
        return None  # a replay costs nothing

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        return self.recording.turn_session(name, snapshot=snapshot)


def reply(
    *content: TurnBlock, stop: TurnStopReason | None = None, cost: float | None = 0.01, model: str = MODEL
) -> TurnReply:
    """A scripted reply: a tool-use stop when it calls a tool, else an end of turn."""
    calls = any(isinstance(block, TurnToolUse) for block in content)
    return TurnReply(
        content=content,
        stop_reason=stop or ("tool_use" if calls else "end_turn"),
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


def call(n: int, action: str, snapshot: str, **arguments: Any) -> TurnToolUse:
    """A call to the analyst tool."""
    return TurnToolUse(f"toolu_{n:03d}", "evals_analyst", {"action": action, "snapshot_id": snapshot, **arguments})


def thinking(said: str) -> TurnThinking:
    return TurnThinking({"type": "thinking", "thinking": said, "signature": f"sig-{len(said)}"})


def submitted(doc: Mapping[str, Any] = DOC) -> TurnText:
    """An analyst's submission: one fenced JSON object."""
    return TurnText(f"Here is the report.\n```json\n{json.dumps(doc)}\n```")


def answer(
    scores: Sequence[int] = (2, 2, 2, 2, 2, 1, 2), findings: Sequence[Mapping[str, str]] = (), **extra: Any
) -> dict[str, Any]:
    """An editor's review as JSON: the seven items scored, and the findings."""
    return {
        "items": [
            {"n": n, "name": name, "score": score, "reason": f"item {n} reads so"}
            for n, (name, score) in enumerate(zip(RUBRIC, scores, strict=True), start=1)
        ],
        "findings": list(findings),
        **extra,
    }


def review(
    scores: Sequence[int] = (2, 2, 2, 2, 2, 1, 2), findings: Sequence[Mapping[str, str]] = (), **extra: Any
) -> TurnText:
    """An editor's final message: a line of prose, then its review as one fenced JSON object."""
    return TurnText(f"My review.\n```json\n{json.dumps(answer(scores, findings, **extra))}\n```")


async def written(
    model: Any,  # Any: a TurnModel: FakeTurnModel or RecordedModel
    *,
    snapshot: str | None = None,
    budget_usd: float = 100.0,
    withhold: Sequence[str] = (),
    limits: WriterLimits | None = None,
    host: OpsHost | None = None,
) -> ReportWritten:
    """``write_report`` over a host seeded with every fixture snapshot (Kairo's by default)."""
    return await write_report(
        host if host is not None else off_loop(seeded_host),
        CALLER,
        model=model,
        budget=ReportBudget(usd=budget_usd),
        snapshot_id=snapshot if snapshot is not None else kairo_id(),
        withhold=withhold,
        limits=limits,
    )


def replay_cases() -> list[Path]:
    """Every recording a replay is held to: each directory, hand-scripted or accepted, holding a ``replay.json``."""
    return sorted([*HAND_RECORDINGS.glob("*/replay.json"), *ACCEPTED_RECORDINGS.glob("*/replay.json")])
