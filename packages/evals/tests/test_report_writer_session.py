"""The analyst's session: its brief, its budgets and stop rules, its bounces, its tally, and its stored form.

Every turn goes through a scripted :class:`FakeToolTurnClient` and every tool call through a :class:`FakeToolMount`,
so nothing reaches a model or a snapshot: the session's own rules are what is under test. The one turn that goes
through the shipped recorder (:class:`~threetears.evals.testing.RecordedTurns`) records onto a temporary file.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

from threetears.evals.analysis.analyst.skill import SkillText
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.ops import report_writer
from threetears.evals.ops.report_writer.analyst import AnalystLimits, AnalystSession
from threetears.evals.ops.report_writer.session import SessionLimits
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.schema.turns import (
    ToolResultClearing,
    TurnBlock,
    TurnMessage,
    TurnReply,
    TurnReplyRefused,
    TurnStopReason,
    TurnText,
    TurnThinking,
    TurnToolSpec,
    TurnToolUse,
)
from threetears.evals.testing import RecordedTurns, RecordingRefused

SNAPSHOT: Final = "snap-7f3a91c2"
MODEL: Final = "analyst-model"
SKILL: Final = SkillText(text="ANALYST SKILL TEXT", files=("SKILL.md",))
OVERVIEW: Final = "OVERVIEW: arms September, control; aspects late"

#: A converted prototype document, which the model's shape, validity and these tests' checker all pass.
DOC: Final[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "report_docs" / "prototype3" / "kairo" / "doc.json").read_text()
)
ARMS: Final = frozenset({"September", "control"})
CAVEATS: Final = frozenset(
    caveat["id"] for caveat in DOC["confidence"]["caveats"] if not caveat["id"].startswith("analyst:")
)

Request = dict[str, Any]


# parity-with: threetears.evals.schema.turns.ToolTurnClient
class FakeToolTurnClient:
    """Answers each turn with the next scripted reply, and keeps every request body it was sent."""

    def __init__(
        self, *replies: TurnReply | TurnReplyRefused, ceiling: float | None | Sequence[float | None] = 0.01
    ) -> None:
        self.replies = list(replies)
        self.ceilings = list(ceiling) if isinstance(ceiling, Sequence) else None
        self.ceiling = ceiling if not isinstance(ceiling, Sequence) else None
        self.requests: list[Request] = []
        self.ceilings_asked = 0
        self.refreshed = 0

    async def turn(
        self,
        *,
        system: str,
        messages: Sequence[TurnMessage],
        tools: Sequence[TurnToolSpec],
        max_tokens: int,
        clearing: ToolResultClearing | None = None,
    ) -> TurnReply:
        self.requests.append(
            {
                "system": system,
                "messages": [message.to_json() for message in messages],
                "tools": [tool.to_json() for tool in tools],
                "max_tokens": max_tokens,
                "context_management": clearing.to_json() if clearing is not None else None,
            }
        )
        if not self.replies:
            raise AssertionError(f"the session sent turn {len(self.requests)}, which the script does not hold")
        scripted = self.replies.pop(0)
        if isinstance(scripted, TurnReplyRefused):
            raise scripted
        return scripted

    def turn_price_ceiling(
        self, *, system: str, messages: Sequence[TurnMessage], tools: Sequence[TurnToolSpec], max_tokens: int
    ) -> float | None:
        self.ceilings_asked += 1
        if self.ceilings is not None:
            return self.ceilings.pop(0)
        return self.ceiling

    async def refresh_prices(self) -> None:
        self.refreshed += 1


@dataclass(frozen=True)
class Outcome:
    text: str
    is_error: bool = False


# parity-with: threetears.evals.ops.report_writer.session.ToolMount
class FakeToolMount:
    """The analyst tool over a frozen snapshot: each action answers a text made of its arguments."""

    def __init__(
        self,
        *,
        answers: Mapping[str, str] | None = None,
        refused: frozenset[str] = frozenset(),
        rederivable: bool = True,
        salt: str = "",
    ) -> None:
        self.answers = {"campaign_overview": OVERVIEW, **(answers or {})}
        self.refused = refused
        self.salt = salt
        self.calls: list[dict[str, Any]] = []
        self._rederivable = rederivable

    @property
    def name(self) -> str:
        return "evals_analyst"

    @property
    def description(self) -> str:
        return "Read the snapshot. PRIMER"

    @property
    def rederivable(self) -> bool:
        return self._rederivable

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}

    async def call(self, arguments: Mapping[str, Any]) -> Outcome:
        self.calls.append(dict(arguments))
        action = str(arguments.get("action"))
        text = self.answers.get(action, f"{action} reads {json.dumps(dict(arguments), sort_keys=True)}{self.salt}")
        return Outcome(text, is_error=action in self.refused)


@dataclass(frozen=True)
class Verdict:
    path: str
    fault: str
    remedy: str


class ScriptedCheck:
    """The invariants' checker: the scripted verdicts for each call in turn, then none."""

    def __init__(self, *rounds: Sequence[Verdict]) -> None:
        self.rounds = list(rounds)
        self.checked = 0

    def __call__(self, doc: ReportDoc) -> list[Verdict]:
        self.checked += 1
        return list(self.rounds.pop(0)) if self.rounds else []


def reply(
    *content: TurnBlock,
    stop: TurnStopReason | None = None,
    cost: float | None = 0.01,
    input_tokens: int | None = 1000,
) -> TurnReply:
    calls = any(isinstance(block, TurnToolUse) for block in content)
    return TurnReply(
        content=content,
        stop_reason=stop or ("tool_use" if calls else "end_turn"),
        provider_stop_reason=None,
        model=MODEL,
        served_model="analyst-model-20261001",
        input_tokens=input_tokens,
        output_tokens=50,
        reasoning_tokens=20,
        cache_read_tokens=800,
        cache_write_tokens=100,
        cost_usd=cost,
        price_source=None if cost is None else "catalogue",
        pricing_failure="no rate for the model" if cost is None else None,
    )


def call(n: int, action: str = "readings_list") -> TurnToolUse:
    return TurnToolUse(f"toolu_{n:03d}", "evals_analyst", {"action": action, "snapshot_id": SNAPSHOT, "n": n})


def thinking(n: int) -> TurnThinking:
    return TurnThinking({"type": "thinking", "thinking": f"thought {n}", "signature": f"sig-{n}"})


def submitted(doc: Mapping[str, Any] = DOC) -> TurnText:
    return TurnText(f"Here is the report.\n```json\n{json.dumps(doc)}\n```")


def start(
    client: FakeToolTurnClient,
    *,
    mount: FakeToolMount | None = None,
    check: ScriptedCheck | None = None,
    tally: Tally | None = None,
    budget: ReportBudget | None = None,
    limits: AnalystLimits | None = None,
) -> AnalystSession:
    return AnalystSession.start(
        model=MODEL,
        client=client,
        mount=mount or FakeToolMount(),
        skill=SKILL,
        snapshot=SNAPSHOT,
        arms=ARMS,
        caveats=CAVEATS,
        check=check or ScriptedCheck(),
        tally=tally or Tally(),
        budget=budget or ReportBudget(usd=100),
        limits=limits,
    )


def last_user(request: Request) -> list[dict[str, Any]]:
    message = request["messages"][-1]
    assert message["role"] == "user"
    blocks: list[dict[str, Any]] = message["content"]
    return blocks


def texts(blocks: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(block["text"] for block in blocks if block["type"] == "text")


# --- the brief and a passing submission ----------------------------------------------------------------------------


async def test_a_submission_that_passes_every_check_ends_the_draft_submitted() -> None:
    mount, check, tally = FakeToolMount(), ScriptedCheck(), Tally()
    client = FakeToolTurnClient(reply(thinking(1), call(1)), reply(thinking(2), submitted()))

    outcome = await start(client, mount=mount, check=check, tally=tally).draft()

    assert outcome.ending == "submitted" and outcome.round == 1
    assert outcome.doc == ReportDoc.model_validate(DOC)
    assert check.checked == 1
    brief = texts(client.requests[0]["messages"][0]["content"])
    assert SNAPSHOT in brief and OVERVIEW in brief and '"report-doc/1"' in brief
    assert "100 tool calls" in brief
    assert client.requests[0]["system"] == SKILL.text
    assert client.requests[0]["tools"] == [
        {"name": "evals_analyst", "description": "Read the snapshot. PRIMER", "input_schema": mount.input_schema()}
    ]
    assert mount.calls[0] == {"action": "campaign_overview", "snapshot_id": SNAPSHOT}
    assert len(mount.calls) == 2, "the overview is the brief's, and the analyst's one call is the only other"
    generation = tally.to_json()
    assert generation["by_round"]["analyst:1"]["tool_calls"] == 1, "the brief's overview is not the analyst's call"
    assert [turn["round"] for turn in generation["turns"]] == [1, 1]


def test_the_report_writer_reads_no_environment() -> None:
    """Plan §4.6: the model, its key and every limit come from the caller; no module of the loop reads the env."""
    package = Path(report_writer.__file__).parent
    for module in sorted(package.glob("*.py")):
        tree = ast.parse(module.read_text())
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in node.names
        } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        assert not imported & {"os", "environ", "getenv", "dotenv"}, f"{module.name} reaches for the environment"


# --- the call budget: warning, submit now, hard stop ---------------------------------------------------------------


async def test_the_call_budget_warns_then_says_submit_now_then_stops_hard() -> None:
    mount, tally = FakeToolMount(), Tally()
    limits = AnalystLimits(draft_calls=5, session=SessionLimits(warn_before=2))
    client = FakeToolTurnClient(
        reply(call(1), call(2), call(3)),
        reply(call(4), call(5), call(6)),
        reply(call(7)),
        reply(submitted()),
    )

    outcome = await start(client, mount=mount, tally=tally, limits=limits).draft()

    assert outcome.ending == "over_budget"
    assert "told to submit" in outcome.reason
    assert len(client.requests) == 3, "the call after submit-now stops the session without another turn"
    assert len(mount.calls) == 1 + 5, "the overview, then exactly the round's five calls"
    warned = last_user(client.requests[1])
    assert "2 left" in texts(warned)
    closing = last_user(client.requests[2])
    assert [block.get("is_error", False) for block in closing if block["type"] == "tool_result"] == [
        False,
        False,
        True,
    ]
    assert "not run" in closing[2]["content"]
    assert "used up" in texts(closing) and "Answer now" in texts(closing)
    assert tally.to_json()["by_round"]["analyst:1"]["tool_calls"] == 5


async def test_a_model_that_submits_after_submit_now_is_read() -> None:
    limits = AnalystLimits(draft_calls=1, session=SessionLimits(warn_before=0))
    client = FakeToolTurnClient(reply(call(1)), reply(submitted()))

    outcome = await start(client, limits=limits).draft()

    assert outcome.ending == "submitted"


async def test_a_revision_has_its_own_call_budget() -> None:
    limits = AnalystLimits(draft_calls=1, revision_calls=2, session=SessionLimits(warn_before=0))
    client = FakeToolTurnClient(
        reply(call(1)), reply(submitted()), reply(call(2), call(3)), reply(submitted()), reply(submitted())
    )
    session = start(client, limits=limits)
    await session.draft()

    outcome = await session.revise("The editor found claim c1 unsupported.")

    assert outcome.ending == "submitted" and outcome.round == 2
    revision = texts(last_user(client.requests[2]))
    assert revision.startswith("The editor found claim c1 unsupported.") and "2 tool calls" in revision
    assert all(not block.get("is_error") for block in last_user(client.requests[3]) if block["type"] == "tool_result")


# --- bounces --------------------------------------------------------------------------------------------------------


def _missing_headline() -> dict[str, Any]:
    doc = dict(DOC)
    del doc["headline"]
    return doc


def _unknown_arm() -> dict[str, Any]:
    return {**DOC, "recommends": ["October"]}


FENCE: Final = "```"

BOUNCES: Final = [
    pytest.param(reply(TurnText("The report is done.")), "unparsed", "no fenced block", id="no-fence"),
    pytest.param(
        reply(TurnText(f"{FENCE}json\n{{}}\n{FENCE}\n{FENCE}json\n{{}}\n{FENCE}")), "unparsed", "2 fenced", id="two"
    ),
    pytest.param(reply(TurnText(f"{FENCE}json\n{{not json\n{FENCE}")), "unparsed", "is not JSON", id="bad-json"),
    pytest.param(reply(TurnText(f"{FENCE}json\n[1]\n{FENCE}")), "unparsed", "not an object", id="not-object"),
    pytest.param(reply(submitted(_missing_headline())), "shape", "headline", id="shape"),
    pytest.param(reply(submitted(_unknown_arm())), "validity", "October", id="validity"),
    pytest.param(
        reply(thinking(9), TurnText("partial"), call(9), stop="max_tokens"), "max_tokens", "output cap", id="cut"
    ),
]


@pytest.mark.parametrize(("faulty", "kind", "said"), BOUNCES)
async def test_each_fault_bounces_the_submission_with_what_is_wrong(faulty: TurnReply, kind: str, said: str) -> None:
    tally = Tally()
    client = FakeToolTurnClient(faulty, reply(submitted()))

    outcome = await start(client, tally=tally).draft()

    assert outcome.ending == "submitted"
    bounce = texts(last_user(client.requests[1]))
    assert f"[{kind}]" in bounce and said in bounce and "bounce 1 of 3" in bounce
    assert tally.to_json()["by_round"]["analyst:1"]["bounce_kinds"] == {kind: 1}


async def test_an_invariant_verdict_bounces_with_its_path_fault_and_remedy() -> None:
    check = ScriptedCheck([Verdict("claims[0].text", "a typed digit 42", "cite the figure by its key")])
    client = FakeToolTurnClient(reply(submitted()), reply(submitted()))

    outcome = await start(client, check=check).draft()

    assert outcome.ending == "submitted" and check.checked == 2
    bounce = texts(last_user(client.requests[1]))
    assert "- [invariant] claims[0].text: a typed digit 42; cite the figure by its key" in bounce


async def test_a_reply_cut_at_the_cap_is_not_kept_and_its_calls_are_not_run() -> None:
    mount = FakeToolMount()
    cut = reply(thinking(9), TurnText("partial"), call(9), stop="max_tokens")
    client = FakeToolTurnClient(reply(call(1)), cut, reply(submitted()))

    await start(client, mount=mount).draft()

    after = client.requests[2]["messages"]
    assert all(message["role"] != "assistant" or "sig-9" not in json.dumps(message) for message in after)
    assert [message["role"] for message in after] == ["user", "assistant", "user", "user"], (
        "the bounce follows the results it was sent after, which stay as they were sent"
    )
    assert after[:3] == client.requests[1]["messages"]
    assert {block["type"] for block in after[-1]["content"]} == {"text"}
    assert len(mount.calls) == 2, "the cut reply's call is never run"


async def test_three_bounces_then_a_fault_ends_the_round_refused() -> None:
    tally = Tally()
    client = FakeToolTurnClient(*(reply(submitted(_unknown_arm())) for _ in range(4)))

    outcome = await start(client, tally=tally).draft()

    assert outcome.ending == "refused"
    assert len(client.requests) == 4, "three bounces are sent; the fourth fault ends the round"
    assert [fault.kind for fault in outcome.faults] == ["validity"]
    assert "October" in outcome.faults[0].fault
    assert tally.to_json()["by_round"]["analyst:1"]["bounces"] == 3


# --- the dollar budget and the tally --------------------------------------------------------------------------------


async def test_an_admission_refusal_sends_nothing() -> None:
    client = FakeToolTurnClient(reply(submitted()), ceiling=0.5)
    tally = Tally()

    outcome = await start(client, tally=tally, budget=ReportBudget(usd=0.25)).draft()

    assert outcome.ending == "over_budget" and "budget refused" in outcome.reason
    assert client.requests == [] and client.ceilings_asked == 1
    assert tally.turns == ()


async def test_admission_is_asked_before_every_turn_and_counts_what_was_spent() -> None:
    client = FakeToolTurnClient(reply(call(1), cost=0.1), reply(call(2), cost=0.1), reply(submitted()), ceiling=0.1)
    budget = ReportBudget(usd=0.25)

    outcome = await start(client, budget=budget).draft()

    assert outcome.ending == "over_budget"
    assert len(client.requests) == 2 and client.ceilings_asked == 3
    assert budget.spent == pytest.approx(0.2)


async def test_a_turn_the_client_cannot_bound_is_not_admitted() -> None:
    client = FakeToolTurnClient(reply(submitted()), ceiling=None)

    outcome = await start(client).draft()

    assert outcome.ending == "over_budget" and "cannot bound" in outcome.reason
    assert client.requests == []


async def test_an_unpriced_turn_is_tallied_as_unknown_never_zero() -> None:
    tally, budget = Tally(), ReportBudget(usd=10)
    client = FakeToolTurnClient(reply(call(1), cost=0.25), reply(submitted(), cost=None), ceiling=0.75)

    await start(client, tally=tally, budget=budget).draft()

    unpriced = tally.turns[1]
    assert unpriced.cost_usd is None and unpriced.price_source is None
    assert unpriced.pricing_failure == "no rate for the model"
    generation = tally.to_json()
    assert generation["turns"][1]["cost_usd"] is None
    assert generation["by_role"]["analyst"]["cost_usd"] is None
    assert generation["by_role"]["analyst"]["unpriced_turns"] == 1
    assert generation["total"]["cost_usd"] is None
    assert tally.spent_usd() is None
    assert budget.spent == pytest.approx(0.25 + 0.75), "an unpriced turn is charged its ceiling"


async def test_each_turn_is_tallied_with_its_role_round_model_and_tokens() -> None:
    tally = Tally()
    client = FakeToolTurnClient(reply(call(1)), reply(submitted()))

    await start(client, tally=tally).draft()

    first = tally.to_json()["turns"][0]
    assert {key: first[key] for key in ("session", "role", "round", "turn", "model", "served_model")} == {
        "session": "analyst",
        "role": "analyst",
        "round": 1,
        "turn": 0,
        "model": "analyst-model",
        "served_model": "analyst-model-20261001",
    }
    assert (first["input_tokens"], first["cache_read_tokens"], first["cache_write_tokens"]) == (1000, 800, 100)
    assert (first["output_tokens"], first["reasoning_tokens"]) == (50, 20)
    assert first["cost_usd"] == 0.01 and first["price_source"] == "catalogue"
    total = tally.to_json()["total"]
    assert total["cost_usd"] == pytest.approx(0.02) and total["input_tokens"] == 2000 and total["turns"] == 2
    assert tally.to_json()["transcripts"]["analyst"]


async def test_a_reply_holding_the_snapshot_placeholder_is_tallied_before_it_is_refused(tmp_path: Path) -> None:
    tally, budget = Tally(), ReportBudget(usd=10)
    inner = FakeToolTurnClient(reply(TurnText("the key reads {snapshot}"), cost=0.42), ceiling=0.5)
    recording = RecordedTurns.record(inner, tmp_path / "turns.jsonl")
    client = recording.turn_session("analyst", snapshot=SNAPSHOT)
    session = AnalystSession.start(
        model=MODEL,
        client=client,
        mount=FakeToolMount(),
        skill=SKILL,
        snapshot=SNAPSHOT,
        arms=ARMS,
        caveats=CAVEATS,
        check=ScriptedCheck(),
        tally=tally,
        budget=budget,
    )

    with pytest.raises(RecordingRefused, match=r"literal text \{snapshot\}") as refused:
        await session.draft()

    assert isinstance(refused.value, TurnReplyRefused)
    (paid,) = tally.turns
    assert paid.cost_usd == 0.42 and paid.refused is not None and "{snapshot}" in paid.refused
    assert budget.spent == pytest.approx(0.42)
    assert inner.requests[0]["context_management"] == SessionLimits().clearing().to_json(), (
        "the recording passes the clearing on to the client it records through"
    )


# --- tool results ---------------------------------------------------------------------------------------------------


async def test_refused_calls_carry_is_error_and_answered_ones_do_not() -> None:
    mount = FakeToolMount(refused=frozenset({"bogus"}))
    stray = TurnToolUse("toolu_stray", "web_search", {"q": "x"})
    client = FakeToolTurnClient(reply(call(1), call(2, action="bogus"), stray), reply(submitted()))

    await start(client, mount=mount).draft()

    results = [block for block in last_user(client.requests[1]) if block["type"] == "tool_result"]
    assert [block.get("is_error") for block in results] == [None, True, True]
    assert "no tool 'web_search'" in results[2]["content"]


async def test_a_long_result_is_cut_at_its_limit_and_says_so() -> None:
    mount = FakeToolMount(answers={"readings_list": "x" * 20_000})
    client = FakeToolTurnClient(reply(call(1)), reply(submitted()))

    await start(client, mount=mount).draft()

    (result,) = [block for block in last_user(client.requests[1]) if block["type"] == "tool_result"]
    assert result["content"].startswith("x" * 12_000 + "\n[cut at 12,000 of 20,000 characters")


@pytest.mark.parametrize("stop", ["content_filter", "truncated"])
async def test_a_refusal_or_a_reply_stopped_short_fails_the_round(stop: TurnStopReason) -> None:
    client = FakeToolTurnClient(reply(TurnText("no"), stop=stop))

    outcome = await start(client).draft()

    assert outcome.ending == "failed" and stop in outcome.reason


# --- stored and resumed ---------------------------------------------------------------------------------------------


def _script() -> list[TurnReply]:
    return [
        reply(thinking(1), call(1)),
        reply(thinking(2), call(2), call(3, action="bogus")),
        reply(thinking(3), submitted()),
        reply(thinking(4), call(4)),
        reply(thinking(5), submitted()),
    ]


REVISION: Final = "The editor found claim c1's evidence thin."


async def test_a_stored_session_resumes_with_the_request_an_uninterrupted_one_sends() -> None:
    uninterrupted = FakeToolTurnClient(*_script())
    whole = start(uninterrupted, mount=FakeToolMount(refused=frozenset({"bogus"})))
    await whole.draft()
    await whole.revise(REVISION)

    first = FakeToolTurnClient(*_script()[:3])
    stored_from = start(first, mount=FakeToolMount(refused=frozenset({"bogus"})))
    await stored_from.draft()
    record = json.loads(json.dumps(stored_from.to_json()))
    later = FakeToolTurnClient(*_script()[3:])
    mount = FakeToolMount(refused=frozenset({"bogus"}))
    resumed = await AnalystSession.resume(
        record,
        model=MODEL,
        client=later,
        mount=mount,
        skill=SKILL,
        arms=ARMS,
        caveats=CAVEATS,
        check=ScriptedCheck(),
        tally=Tally(),
        budget=ReportBudget(usd=10),
    )
    outcome = await resumed.revise(REVISION)

    assert outcome.ending == "submitted" and outcome.round == 2
    assert later.requests == uninterrupted.requests[3:], "the resumed session sends what the uninterrupted one did"
    stored = record["session"]
    assert {result["stored"] for result in stored["results"]} == {"rederive"}
    held = [block for message in stored["messages"] for block in message["content"] if block["type"] == "tool_result"]
    assert all(block["content"] == "" for block in held), "a re-derivable body is not stored"
    assert [call["n"] for call in mount.calls[:3]] == [1, 2, 3], "each dropped body is called again to resume"
    thought = [block for message in stored["messages"] for block in message["content"] if block["type"] == "thinking"]
    assert thought == [thinking(n).to_json() for n in (1, 2, 3)], "thinking blocks are stored exactly"
    assert stored["skill_digest"] == SKILL.digest


async def test_a_mount_whose_results_cannot_be_rederived_stores_them_whole() -> None:
    client = FakeToolTurnClient(reply(call(1)), reply(submitted()))
    session = start(client, mount=FakeToolMount(rederivable=False))
    await session.draft()

    stored = session.to_json()["session"]

    held = [block for message in stored["messages"] for block in message["content"] if block["type"] == "tool_result"]
    assert held[0]["content"].startswith("readings_list reads")
    assert stored["results"][0]["stored"] == "whole"


async def _stored() -> dict[str, Any]:
    session = start(FakeToolTurnClient(*_script()[:3]), mount=FakeToolMount(refused=frozenset({"bogus"})))
    await session.draft()
    return session.to_json()


async def test_a_session_is_not_resumed_with_another_skill_text() -> None:
    other = SkillText(text="ANOTHER SKILL TEXT", files=("SKILL.md",))

    with pytest.raises(ValueError, match="resume it with the skill text it started from"):
        await AnalystSession.resume(
            await _stored(),
            model=MODEL,
            client=FakeToolTurnClient(),
            mount=FakeToolMount(refused=frozenset({"bogus"})),
            skill=other,
            arms=ARMS,
            caveats=CAVEATS,
            check=ScriptedCheck(),
            tally=Tally(),
            budget=ReportBudget(usd=10),
        )


async def test_a_session_whose_tool_now_answers_otherwise_is_not_resumed() -> None:
    with pytest.raises(ValueError, match="no longer answers as it did"):
        await AnalystSession.resume(
            await _stored(),
            model=MODEL,
            client=FakeToolTurnClient(),
            mount=FakeToolMount(refused=frozenset({"bogus"}), salt=" (changed)"),
            skill=SKILL,
            arms=ARMS,
            caveats=CAVEATS,
            check=ScriptedCheck(),
            tally=Tally(),
            budget=ReportBudget(usd=10),
        )


async def _resume(record: Mapping[str, Any], client: FakeToolTurnClient, **options: Any) -> AnalystSession:
    arguments: dict[str, Any] = {
        "model": MODEL,
        "client": client,
        "mount": FakeToolMount(refused=frozenset({"bogus"})),
        "skill": SKILL,
        "arms": ARMS,
        "caveats": CAVEATS,
        "check": ScriptedCheck(),
        "tally": Tally(),
        "budget": ReportBudget(usd=10),
    }
    arguments.update(options)
    return await AnalystSession.resume(record, **arguments)


MID_ROUND: Final = [
    pytest.param(
        AnalystLimits(draft_calls=4, session=SessionLimits(warn_before=3)),
        [reply(thinking(1), call(1)), reply(thinking(2), call(2))],
        [0.01, 0.01, 100.0],
        [reply(thinking(3), call(3)), reply(thinking(4), submitted())],
        {"round": 1, "calls": 4, "calls_used": 2, "warned": True, "closing": False},
        id="the-budget-refused-a-turn-after-the-warning",
    ),
    pytest.param(
        AnalystLimits(draft_calls=1, session=SessionLimits(warn_before=0)),
        [reply(thinking(1), call(1)), reply(thinking(2), call(2))],
        [0.01, 0.01],
        [reply(thinking(3), submitted())],
        {"round": 1, "calls": 1, "calls_used": 1, "warned": False, "closing": True},
        id="a-call-after-submit-now-left-open",
    ),
]


@pytest.mark.parametrize(("limits", "draft", "ceilings", "revision", "state"), MID_ROUND)
async def test_a_session_stored_mid_round_resumes_with_the_request_an_uninterrupted_one_sends(
    limits: AnalystLimits,
    draft: list[TurnReply],
    ceilings: list[float],
    revision: list[TurnReply],
    state: dict[str, Any],
) -> None:
    later_ceilings = [0.01] * len(revision)
    uninterrupted = FakeToolTurnClient(*draft, *revision, ceiling=[*ceilings, *later_ceilings])
    whole = start(uninterrupted, limits=limits, budget=ReportBudget(usd=10))
    assert (await whole.draft()).ending == "over_budget"
    await whole.revise(REVISION)

    first = FakeToolTurnClient(*draft, ceiling=list(ceilings))
    stopped = start(first, limits=limits, budget=ReportBudget(usd=10))
    assert (await stopped.draft()).ending == "over_budget"
    record = json.loads(json.dumps(stopped.to_json()))
    later = FakeToolTurnClient(*revision, ceiling=later_ceilings)
    resumed = await _resume(record, later, mount=FakeToolMount())
    outcome = await resumed.revise(REVISION)

    assert {key: record["session"]["state"][key] for key in state} == state
    assert outcome.ending == "submitted" and outcome.round == 2
    assert later.requests == uninterrupted.requests[len(first.requests) :]


async def test_a_session_is_not_resumed_on_another_model() -> None:
    with pytest.raises(ValueError, match="ran on 'analyst-model', and it is resumed on 'other-model'"):
        await _resume(await _stored(), FakeToolTurnClient(), model="other-model")


async def test_a_reply_from_another_model_is_refused_after_it_is_counted() -> None:
    tally = Tally()
    stray = TurnReply(**{**_reply_members(reply(submitted())), "model": "other-model"})

    with pytest.raises(ValueError, match="sent the turn to 'other-model'"):
        await start(FakeToolTurnClient(stray), tally=tally).draft()

    assert [turn.model for turn in tally.turns] == ["other-model"]


def _reply_members(value: TurnReply) -> dict[str, Any]:
    return {name: getattr(value, name) for name in TurnReply.__dataclass_fields__}


def _without(path: str) -> Callable[[dict[str, Any]], Any]:
    def change(record: dict[str, Any]) -> Any:
        *parents, leaf = path.split(".")
        held = record
        for parent in parents:
            held = held[parent]
        del held[leaf]
        return record

    return change


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], Any]:
    def change(record: dict[str, Any]) -> Any:
        *parents, leaf = path.split(".")
        held = record
        for parent in parents:
            held = held[parent]
        held[leaf] = value
        return record

    return change


MALFORMED: Final = [
    pytest.param(lambda record: [record], id="not-an-object"),
    pytest.param(_without("bounces"), id="no-bounces"),
    pytest.param(_set("bounces", -1), id="negative-bounces"),
    pytest.param(_set("limits", {"draft_calls": 1, "revision_calls": 1, "bounces": 3, "extra": 1}), id="extra-limit"),
    pytest.param(_without("session.model"), id="no-model"),
    pytest.param(_without("session.state"), id="no-state"),
    pytest.param(_set("session.state", {"turns": 1}), id="state-missing-counts"),
    pytest.param(_set("session.state.calls_used", "two"), id="count-of-the-wrong-type"),
    pytest.param(_set("session.state.warned", 1), id="flag-of-the-wrong-type"),
    pytest.param(_set("session.results", [{"tool_use_id": "toolu_001"}]), id="result-missing-its-digest"),
    pytest.param(_set("session.messages", [{"role": "user"}]), id="message-missing-its-content"),
    pytest.param(_set("session.limits", {"max_tokens": 0}), id="limit-out-of-range"),
    pytest.param(_set("session.format", 1), id="an-older-session-format"),
]


@pytest.mark.parametrize("change", MALFORMED)
async def test_a_malformed_record_is_refused_with_a_value_error(change: Callable[[dict[str, Any]], Any]) -> None:
    record = change(json.loads(json.dumps(await _stored())))

    with pytest.raises(ValueError, match=r"stored|a session's max_tokens"):
        await _resume(record, FakeToolTurnClient())


async def test_consecutive_requests_share_a_byte_identical_prefix_across_warnings_bounces_and_revisions() -> None:
    """The history is append-only: each request is the one before it, then what was added since."""
    limits = AnalystLimits(draft_calls=4, revision_calls=3, session=SessionLimits(warn_before=2))
    client = FakeToolTurnClient(
        reply(thinking(1), call(1), call(2)),
        reply(thinking(2), call(3)),
        reply(thinking(3), submitted(_unknown_arm())),
        reply(thinking(4), TurnText("cut"), call(4), stop="max_tokens"),
        reply(thinking(5), submitted()),
        reply(thinking(6), call(5)),
        reply(thinking(7), call(6), call(7)),
        reply(thinking(8), submitted()),
    )
    session = start(client, limits=limits)

    assert (await session.draft()).ending == "submitted"
    assert (await session.revise(REVISION)).ending == "submitted"

    sent = client.requests
    said = "\n".join(texts(message["content"]) for message in sent[-1]["messages"] if message["role"] == "user")
    assert "2 left" in said and "[validity]" in said and "[max_tokens]" in said and REVISION in said
    assert "used up" in said, "the revision's own budget ran out and told the model to answer"
    for before, after in zip(sent, sent[1:], strict=False):
        assert (after["system"], after["tools"], after["max_tokens"]) == (
            before["system"],
            before["tools"],
            before["max_tokens"],
        )
        assert json.dumps(after["messages"][: len(before["messages"])]) == json.dumps(before["messages"]), (
            "an earlier message changed between consecutive requests"
        )
        assert len(after["messages"]) > len(before["messages"])


async def test_every_turn_asks_the_provider_to_clear_old_tool_results() -> None:
    client = FakeToolTurnClient(reply(call(1)), reply(submitted()))

    await start(client).draft()

    assert [request["context_management"] for request in client.requests] == [
        {
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": 150_000},
                    "keep": {"type": "tool_uses", "value": 12},
                    "clear_at_least": {"type": "input_tokens", "value": 36_000},
                }
            ]
        }
    ] * 2


async def test_the_brief_holds_the_overview_to_the_result_limit() -> None:
    mount = FakeToolMount(answers={"campaign_overview": "o" * 20_000})
    client = FakeToolTurnClient(reply(submitted()))

    await start(client, mount=mount).draft()

    brief = texts(client.requests[0]["messages"][0]["content"])
    assert "o" * 12_000 + "\n[cut at 12,000 of 20,000 characters" in brief
    assert "o" * 12_001 not in brief


async def test_any_client_refusing_a_paid_reply_keeps_the_turn_s_cost() -> None:
    tally, budget = Tally(), ReportBudget(usd=10)
    paid = reply(submitted(), cost=0.3)
    client = FakeToolTurnClient(TurnReplyRefused("the reply named a forbidden id", reply=paid))

    with pytest.raises(TurnReplyRefused, match="forbidden id") as refused:
        await start(client, tally=tally, budget=budget).draft()

    assert refused.value.reply is paid
    assert [(turn.cost_usd, turn.refused) for turn in tally.turns] == [(0.3, "the reply named a forbidden id")]
    assert budget.spent == pytest.approx(0.3)


async def test_a_call_left_open_by_the_hard_stop_is_answered_as_the_previous_round_s() -> None:
    limits = AnalystLimits(draft_calls=1, session=SessionLimits(warn_before=0))
    client = FakeToolTurnClient(reply(call(1)), reply(call(2)), reply(submitted()))
    session = start(client, limits=limits)
    assert (await session.draft()).ending == "over_budget"

    await session.revise(REVISION)

    (left_open, request) = last_user(client.requests[2])
    assert left_open == {
        "type": "tool_result",
        "tool_use_id": "toolu_002",
        "content": "not run: the previous round's tool calls were used up when this call was made; this round starts "
        "with a fresh budget of 30 tool calls",
        "is_error": True,
    }
    assert request["text"].startswith(REVISION)
