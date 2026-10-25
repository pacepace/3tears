"""The report planner: two calls (a structure, then its prose), each repaired at most once, held to the host's limits.

Contracts held here (``threetears.evals.analysis.planner``, ``threetears.evals.analysis.plan_assembly``):

- the requests are deterministic and say what the planner is told: the shared prompt at the head of each system prompt,
  the figure-key grammar and the call's own section after it, the report input's view and code's facts (the allowed
  modes among them) opening each user message, and the structure and its prose keys closing the prose call's;
- the kit's Kairo round 2 plan, split into its structure and prose replies, comes back as a plan that passes the plan
  checks and holds exactly the planner's part it was written from;
- a refused structure is repaired once before any prose is asked for, and a second refusal surfaces naming both; a
  refused prose reply (a key missing, a typed digit) repairs only the prose call; a call cut short is never repaired;
- the host's allowed writers refuse a model before any call, and every call, each repair included, is put to ``admit``
  first, so a spend cap refuses a call before it is sent.

The model is the one non-production part: :class:`_Replies` hands back prepared bodies in order.
"""

from __future__ import annotations

import dataclasses
import json
from copy import deepcopy
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.report_plan_support import (
    planned_reply,
    prose_reply,
    r5_input,
    r5_plan,
    structure_reply,
)
from packages.evals.tests.toyhost_memo import MODEL, FixturedClient, FixturedCompletion
from threetears.evals.analysis.errors import GenerationError, PlanRefusal, SoundnessRefusal
from threetears.evals.analysis.generator import GenerationTally
from threetears.evals.analysis.plan_assembly import ProseSlot, read_prose
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.analysis.planner import (
    FACTS_HEADING,
    PROSE_CALL,
    PROSE_KEYS_HEADING,
    REPORT_INPUT_HEADING,
    STRUCTURE_CALL,
    STRUCTURE_HEADING,
    first_plan_request,
    generate_report_plan,
    prose_plan_request,
)
from threetears.evals.analysis.report.figure_keys import GRAMMAR_HEADING
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_checks import validate_plan
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.report_plan import (
    PlanProse,
    PlanStructure,
    prose_response_format,
    structure_response_format,
)

PROMPT = EVAL_REPORT_PLAN_DEFAULT


class _Replies(FixturedClient):
    """Hands back one prepared completion per call, in order, and records what each call was asked."""

    def __init__(self, *replies: FixturedCompletion) -> None:
        super().__init__(replies[0].content)
        self.replies = list(replies)

    async def generate(self, *, system: str, user: str, response_format: Any = None, tools: Any = None) -> Any:
        self.completion = self.replies[len(self.calls)]
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


def _reply(body: dict[str, Any], *, stop_reason: str = "end_turn") -> FixturedCompletion:
    return FixturedCompletion(content=json.dumps(body), stop_reason=stop_reason)


def _input() -> ReportInput:
    return ReportInput.model_validate(r5_input())


def _structure(**changes: Any) -> dict[str, Any]:
    return {**structure_reply(r5_plan()), **changes}


def _prose(**texts: str | None) -> dict[str, Any]:
    """The Kairo prose reply with ``texts`` replacing (or, given None, leaving out) keys; dots spelled ``__``."""
    changed = {key.replace("__", "."): text for key, text in texts.items()}
    items = [
        {"key": item["key"], "text": changed.get(item["key"], item["text"])}
        for item in prose_reply(r5_plan())["prose"]
        if changed.get(item["key"], "") is not None
    ]
    return {"prose": items}


def _allowing(*models: str) -> HostProfile:
    return dataclasses.replace(toyhost_profile(), analysis_writer_models=models)


async def _generate(client: _Replies, **options: Any) -> Any:
    options.setdefault("profile", toyhost_profile())
    return await generate_report_plan(_input(), prompt=PROMPT, model=MODEL, client=client, **options)


class TestTheRequests:
    def test_the_structure_request_is_deterministic_and_says_what_the_planner_is_told(self) -> None:
        profile = toyhost_profile()
        system, user, contract = first_plan_request(_input(), PROMPT, profile)
        assert (system, user, contract) == first_plan_request(_input(), PROMPT, profile)
        assert system.startswith(PROMPT + "\n\n"), "the budgeted prompt is the head of what is sent"
        assert f"## {GRAMMAR_HEADING}" in system and system.endswith(STRUCTURE_CALL)
        assert user.startswith(REPORT_INPUT_HEADING + "\n")
        facts = json.loads(user.split(f"\n\n{FACTS_HEADING}\n", 1)[1])
        assert facts["allowed_modes"] == ["decide"], "code computes the modes when the caller passes none"
        assert facts["disclosures"]["decide"] == ["late:aspect", "late:control", "judge_uncalibrated"]
        checked = {caveat["id"]: caveat["checked"] for caveat in facts["caveats"]}
        assert checked["late:margin"] == "The late margin was checked: the call holds at every margin."
        assert checked["judge_uncalibrated"] == ""
        assert contract == structure_response_format()

    def test_the_allowed_modes_are_told(self) -> None:
        _, user, _ = first_plan_request(_input(), PROMPT, toyhost_profile(), allowed_modes=["decide"])
        assert json.loads(user.split(f"\n\n{FACTS_HEADING}\n", 1)[1])["allowed_modes"] == ["decide"]

    @pytest.mark.parametrize("modes", [[], ["decide", "pick"]])
    def test_allowed_modes_off_the_vocabulary_are_refused_before_anything(self, modes: list[str]) -> None:
        with pytest.raises(ValueError, match="allowed_modes"):
            first_plan_request(_input(), PROMPT, toyhost_profile(), allowed_modes=modes)

    def test_the_prose_request_carries_the_structure_and_exactly_its_keys(self) -> None:
        profile = toyhost_profile()
        structure = PlanStructure.model_validate(_structure())
        system, user, contract = prose_plan_request(_input(), PROMPT, profile, structure)
        assert (system, user, contract) == prose_plan_request(_input(), PROMPT, profile, structure)
        assert system.startswith(PROMPT + "\n\n") and system.endswith(PROSE_CALL)
        _, structure_user, _ = first_plan_request(_input(), PROMPT, profile)
        assert user.startswith(structure_user + "\n\n"), "both calls share their prefix, for a prompt cache"
        sent, keys = user.split(f"\n\n{STRUCTURE_HEADING}\n", 1)[1].split(f"\n\n{PROSE_KEYS_HEADING}\n", 1)
        assert PlanStructure.model_validate_json(sent) == structure
        assert [slot["key"] for slot in json.loads(keys)] == [item["key"] for item in _prose()["prose"]]
        assert contract == prose_response_format()


class TestAValidPlan:
    async def test_two_replies_become_a_checked_plan_holding_what_the_planner_wrote(self) -> None:
        client = _Replies(_reply(_structure()), _reply(_prose()))
        tally = GenerationTally()

        generated = await _generate(client, tally=tally)

        validate_plan(generated.plan, _input())
        assert planned_reply(generated.plan.model_dump(mode="json")) == planned_reply(r5_plan())
        assert (len(client.calls), tally.calls, generated.repair_attempts) == (2, 2, 0)
        assert generated.structure_refusal is None and generated.prose_refusal is None
        assert generated.token_cost == pytest.approx(0.04) and generated.generator_model == MODEL

    async def test_the_plan_carries_codes_facts(self) -> None:
        generated = await _generate(_Replies(_reply(_structure()), _reply(_prose())))
        plan, r5 = generated.plan, r5_plan()
        assert (plan.shape, plan.show_arms_key) == (r5["shape"], r5["show_arms_key"])
        assert plan.signals.model_dump(mode="json") == r5["signals"]
        assert [arm.model_dump(mode="json") for arm in plan.arms] == r5["arms"]
        assert plan.options.columns == r5["options"]["columns"]
        assert plan.full_results == r5["full_results"]
        assert plan.flag == [] == r5["flag"], "Kairo's open caveats all sit in its confidence, so no flag is left"
        assert plan.decision.confidence.model_dump(mode="json") == r5["decision"]["confidence"]


class TestCaveatLandings:
    async def test_a_landing_at_an_address_the_structure_lacks_is_refused_and_repaired(self) -> None:
        wrong = _structure()
        wrong["caveat_landings"][3]["at"] = "decision.the_trade"
        client = _Replies(_reply(wrong), _reply(_structure()), _reply(_prose()))
        generated = await _generate(client)
        assert generated.structure_refusal is not None and "caveat_landings[3].at" in generated.structure_refusal

    async def test_a_low_confidence_leaving_a_caveat_unlisted_gets_a_flag_disclosing_it(self) -> None:
        structure = _structure()
        structure["confidence"] = "low"
        structure["caveat_landings"] = [
            landing for landing in structure["caveat_landings"] if landing["caveat"] != "judge_uncalibrated"
        ]
        prose = _prose()
        prose["prose"] = [item for item in prose["prose"] if item["key"] != "confidence.caveats.judge_uncalibrated"]
        prose["prose"] += [
            {"key": "flag.title", "text": "The judge is uncalibrated."},
            {"key": "flag.text", "text": "Cost is measured and survives it."},
        ]
        generated = await _generate(_Replies(_reply(structure), _reply(prose)))
        assert generated.plan.flag[0].discloses == ["judge_uncalibrated"]


class TestTheStructureCall:
    async def test_a_refused_structure_is_repaired_once_before_any_prose(self) -> None:
        wrong = _structure(mode="tradeoff", headline_kind="trade")
        client = _Replies(_reply(wrong), _reply(_structure()), _reply(_prose()))

        generated = await _generate(client, allowed_modes=["decide"])

        assert len(client.calls) == 3 and generated.repair_attempts == 1
        refusal = generated.structure_refusal
        assert refusal is not None and "`mode` is 'tradeoff', and code allows only ['decide']" in refusal
        assert refusal in client.calls[1]["user"], "the repair is told what was refused"
        assert PROSE_KEYS_HEADING not in client.calls[1]["user"], "no prose was asked for before the structure held"

    async def test_a_second_refusal_surfaces_naming_both(self) -> None:
        wrong = _structure(mode="tradeoff", headline_kind="trade")
        client = _Replies(_reply(wrong), _reply(wrong))
        tally = GenerationTally()

        with pytest.raises(PlanRefusal) as refused:
            await _generate(client, allowed_modes=["decide"], tally=tally)

        assert len(client.calls) == 2 and len(tally.refusals) == 2
        message = str(refused.value)
        assert "structure call was refused twice" in message and "First refusal" in message and "Second" in message

    async def test_a_reply_off_the_contract_is_repaired(self) -> None:
        broken = _structure()
        del broken["method_items"]
        client = _Replies(_reply(broken), _reply(_structure()), _reply(_prose()))

        generated = await _generate(client)

        assert generated.structure_refusal is not None and "method_items" in generated.structure_refusal

    async def test_a_structure_the_checks_fault_is_refused_at_the_field(self) -> None:
        wrong = deepcopy(_structure())
        wrong["sections"][0]["chart"]["query"]["arms"] = ["nobody"]
        client = _Replies(_reply(wrong), _reply(_structure()), _reply(_prose()))

        generated = await _generate(client)

        assert "`sections[0].chart.query.arms`" in (generated.structure_refusal or "")

    async def test_a_call_cut_short_is_never_repaired(self) -> None:
        client = _Replies(_reply(_structure(), stop_reason="max_tokens"))

        with pytest.raises(GenerationError) as refused:
            await _generate(client)

        assert not isinstance(refused.value, SoundnessRefusal), "a truncation buys no repair"
        assert len(client.calls) == 1 and "billed" in str(refused.value)


class TestTheProseCall:
    async def test_a_prose_reply_missing_a_key_repairs_the_prose_call(self) -> None:
        client = _Replies(_reply(_structure()), _reply(_prose(lede=None)), _reply(_prose()))

        generated = await _generate(client)

        assert len(client.calls) == 3 and generated.structure_refusal is None
        assert generated.prose_refusal is not None and "leaves out 'lede'" in generated.prose_refusal

    async def test_a_valid_structure_with_invalid_prose_repairs_only_the_prose(self) -> None:
        typed = "Keep the control: both candidates cost 27% more."
        client = _Replies(_reply(_structure()), _reply(_prose(headline=typed)), _reply(_prose()))

        generated = await _generate(client)

        structure_calls = [call for call in client.calls if PROSE_KEYS_HEADING not in call["user"]]
        assert len(structure_calls) == 1, "the structure was asked for once"
        assert client.calls[2]["user"].startswith(client.calls[1]["user"]), "the repair re-sends the prose call"
        assert generated.prose_refusal is not None and "`headline`" in generated.prose_refusal
        assert generated.plan.headline == r5_plan()["headline"]

    async def test_a_second_prose_refusal_surfaces(self) -> None:
        client = _Replies(_reply(_structure()), _reply(_prose(lede=None)), _reply(_prose(lede=None)))

        with pytest.raises(PlanRefusal, match="prose call was refused twice"):
            await _generate(client)

        assert len(client.calls) == 3


class TestReadingProse:
    def test_a_key_twice_a_key_no_slot_has_and_an_empty_written_slot_are_refused_together(self) -> None:
        slots = [ProseSlot(key="headline", holds="h", optional=False), ProseSlot(key="t", holds="t", optional=True)]
        reply = PlanProse.model_validate(
            {
                "prose": [
                    {"key": "headline", "text": " "},
                    {"key": "headline", "text": "x"},
                    {"key": "t", "text": ""},
                    {"key": "stray", "text": "x"},
                ]
            }
        )
        with pytest.raises(PlanRefusal) as refused:
            read_prose(reply, slots)
        why = " ".join(f"{path} {text}" for path, text in refused.value.faults)
        assert "'stray', which names no prose slot" in why and "'headline' 2 times" in why

    def test_an_optional_slot_may_be_empty(self) -> None:
        slots = [ProseSlot(key="t", holds="t", optional=True)]
        assert read_prose(PlanProse.model_validate({"prose": [{"key": "t", "text": ""}]}), slots) == {"t": ""}


class TestTheHostsLimits:
    async def test_a_model_the_host_does_not_allow_is_refused_before_any_call(self) -> None:
        client = _Replies(_reply(_structure()))

        with pytest.raises(GenerationError, match="analysis_writer_models") as refused:
            await _generate(client, profile=_allowing("vendor/another-writer"))

        assert not isinstance(refused.value, SoundnessRefusal) and client.calls == []

    async def test_a_run_measuring_writers_is_not_held_to_the_list(self) -> None:
        client = _Replies(_reply(_structure()), _reply(_prose()))
        await _generate(client, profile=_allowing("vendor/another-writer"), measuring_writers=True)
        assert len(client.calls) == 2

    async def test_every_call_is_admitted_first_with_its_exact_request(self) -> None:
        asked: list[tuple[str, str, Any]] = []
        client = _Replies(_reply(_structure()), _reply(_prose()))

        await _generate(client, admit=lambda system, user, contract: asked.append((system, user, contract)))

        assert asked[0] == first_plan_request(_input(), PROMPT, toyhost_profile())
        structure = PlanStructure.model_validate(_structure())
        assert asked[1] == prose_plan_request(_input(), PROMPT, toyhost_profile(), structure)

    async def test_a_cap_refusing_a_repair_sends_nothing_more(self) -> None:
        """The cap is the caller's: ``admit`` raising on the repair call stops the generation before it is sent."""
        tally = GenerationTally()
        client = _Replies(_reply(_structure()), _reply(_prose(lede=None)), _reply(_prose()))
        admitted: list[str] = []

        def admit(system: str, user: str, contract: Any) -> None:
            if len(admitted) == 2:
                raise GenerationError("refused before sending: the cap is spent")
            admitted.append(user)

        with pytest.raises(GenerationError, match="cap is spent"):
            await _generate(client, admit=admit, tally=tally)

        assert len(client.calls) == 2 and tally.calls == 2, "the refused repair was neither sent nor counted"
