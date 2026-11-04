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
  first, so a spend cap refuses a call before it is sent;
- on the toy operations host (:func:`~packages.evals.tests.toyhost_plan.planner_fixture`), held to the host's
  out-of-run cap: the first call is priced before it is sent (over the cap, or unpriceable under it, refused with
  nothing spent and the client released); every later call, the prose and any repair, is priced when it exists and
  refused before it is sent when it would pass what is left; every call is ledgered, under no cap with enforcement off;
  the logged input size counts what the first call sends; the estimate prices the first call and makes none;
- the planner's preparation over a view past its budget refuses 422 (``ValidationFailedError``).

The model is the one non-production part: :class:`_Replies` hands back prepared bodies in order.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from copy import deepcopy
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE
from packages.evals.tests.toyhost_plan import PLAN_CALLS, planned_generation, planner_fixture
from packages.evals.tests.toyhost_memo import FIXTURED_CALL_CEILING_USD
import threetears.evals.analysis.report_input_build as report_input_build
from threetears.evals.analysis import estimate_analysis_generation, prepare_analysis_generation
from threetears.evals.analysis.service import APPROX_INPUT_TOKENS_BASIS, CHARS_PER_TOKEN
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.inner_ceilings import MAX_GENERATION_CALLS
from threetears.evals.ops import scope_out_of_run_spend
from packages.evals.tests.report_plan_support import (
    planned_reply,
    prose_reply,
    r5_input,
    r5_plan,
    structure_reply,
)
from packages.evals.tests.toyhost_memo import MODEL, FixturedClient, FixturedCompletion
from threetears.evals.analysis.errors import GenerationError, PlanRefusal, SoundnessRefusal
from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT
from threetears.evals.analysis.generator import GenerationTally
from threetears.evals.analysis.plan_assembly import prose_slots, read_prose
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
)
from threetears.evals.analysis.report.figure_keys import GRAMMAR_HEADING
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_checks import validate_plan
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.kernel.errors import RefusedInput
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
        assert facts["spine"] == {
            "kind": "forest",
            "query": {"source": "readings", "aspect": "character adherence (v2)", "measures": []},
            "reference": [{"kind": "margin"}],
        }, "code gives the spine the first section draws"
        assert facts["disclosures"]["decide"] == ["late:aspect", "late:control", "judge_uncalibrated"]
        checked = {caveat["id"]: caveat["checked"] for caveat in facts["caveats"]}
        assert checked["late:margin"] == "The late margin was checked: the call holds at every margin."
        assert checked["judge_uncalibrated"] == ""
        assert contract == structure_response_format()

    def test_the_facts_tell_the_planner_to_name_each_id_by_its_one_reader_name(self) -> None:
        _, user, _ = first_plan_request(_input(), PROMPT, toyhost_profile())
        facts = json.loads(user.split(f"\n\n{FACTS_HEADING}\n", 1)[1])
        rule = facts["reader_names_rule"]
        assert "`reader_names`" in rule and "each measure" in rule and "the same name every time" in rule
        assert list(facts).index("reader_names_rule") == list(facts).index("reader_names") - 1, "said beside them"
        assert PROMPT not in user and rule not in PROMPT, "code's, outside the prompt and its budget"

    def test_the_allowed_modes_are_told(self) -> None:
        _, user, _ = first_plan_request(_input(), PROMPT, toyhost_profile(), allowed_modes=["decide"])
        assert json.loads(user.split(f"\n\n{FACTS_HEADING}\n", 1)[1])["allowed_modes"] == ["decide"]

    @pytest.mark.parametrize("modes", [[], ["decide", "pick"]])
    def test_allowed_modes_off_the_vocabulary_are_refused_before_anything(self, modes: list[str]) -> None:
        with pytest.raises(ValueError, match="allowed_modes"):
            first_plan_request(_input(), PROMPT, toyhost_profile(), allowed_modes=modes)

    async def test_the_prose_request_carries_the_structure_and_exactly_its_keys(self) -> None:
        """Read through the front door: the prose call a generation sends, the same each time."""
        profile = toyhost_profile()
        structure = PlanStructure.model_validate(_structure())
        sent = []
        for _ in range(2):
            client = _Replies(_reply(_structure()), _reply(_prose()))
            await _generate(client, profile=profile)
            sent.append((client.calls[1]["system"], client.calls[1]["user"], client.response_formats[1]))
        system, user, contract = sent[0]
        assert sent[0] == sent[1], "the prose request is deterministic"
        assert system.startswith(PROMPT + "\n\n") and system.endswith(PROSE_CALL)
        _, structure_user, _ = first_plan_request(_input(), PROMPT, profile)
        assert user.startswith(structure_user + "\n\n"), "both calls share their prefix, for a prompt cache"
        sent, keys = user.split(f"\n\n{STRUCTURE_HEADING}\n", 1)[1].split(f"\n\n{PROSE_KEYS_HEADING}\n", 1)
        assert PlanStructure.model_validate_json(sent) == structure
        assert [slot["key"] for slot in json.loads(keys)] == [item["key"] for item in _prose()["prose"]]
        assert contract == prose_response_format()


class TestTheMemoPromptIsRefused:
    """A host whose prompt resolution still returns the memo writer's seed is told what to register, before any call."""

    def test_the_structure_request_refuses_it_naming_the_planners_prompt(self) -> None:
        with pytest.raises(RefusedInput) as refused:
            first_plan_request(_input(), EVAL_ANALYSIS_GEN_DEFAULT, toyhost_profile())
        assert "memo writer's prompt" in refused.value.fault
        assert "register the `eval_report_plan` prompt" in refused.value.remedy
        assert "EVAL_REPORT_PLAN_DEFAULT" in refused.value.remedy

    async def test_a_generation_refuses_it_before_any_call(self) -> None:
        client = _Replies(_reply(_structure()), _reply(_prose()))
        with pytest.raises(RefusedInput, match="eval_report_plan"):
            await generate_report_plan(
                _input(), prompt=EVAL_ANALYSIS_GEN_DEFAULT, model=MODEL, client=client, profile=toyhost_profile()
            )
        assert client.calls == []


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
    """Read against the slots the Kairo structure creates (``prose_slots``), as a generation asks for them."""

    def test_a_key_twice_a_key_no_slot_has_and_an_empty_written_slot_are_refused_together(self) -> None:
        slots = prose_slots(PlanStructure.model_validate(_structure()), plan_facts(_input()))
        assert any(slot.key == "headline" and not slot.optional for slot in slots)
        optional = next(slot.key for slot in slots if slot.optional)
        items = [item for item in _prose()["prose"] if item["key"] not in ("headline", optional)]
        reply = PlanProse.model_validate(
            {
                "prose": [
                    *items,
                    {"key": "headline", "text": " "},
                    {"key": "headline", "text": "x"},
                    {"key": optional, "text": ""},
                    {"key": "stray", "text": "x"},
                ]
            }
        )
        with pytest.raises(PlanRefusal) as refused:
            read_prose(reply, slots)
        why = " ".join(f"{path} {text}" for path, text in refused.value.faults)
        assert "'stray', which names no prose slot" in why and "'headline' 2 times" in why
        assert repr(optional) not in why, "an optional slot left empty is no fault"

    def test_an_optional_slot_may_be_empty(self) -> None:
        slots = prose_slots(PlanStructure.model_validate(_structure()), plan_facts(_input()))
        optional = next(slot.key for slot in slots if slot.optional)
        items = [{**item, "text": ""} if item["key"] == optional else item for item in _prose()["prose"]]
        texts = read_prose(PlanProse.model_validate({"prose": items}), slots)
        assert texts[optional] == ""

    def test_a_slot_that_must_be_written_left_empty_is_refused(self) -> None:
        slots = prose_slots(PlanStructure.model_validate(_structure()), plan_facts(_input()))
        items = [{**item, "text": " "} if item["key"] == "headline" else item for item in _prose()["prose"]]
        with pytest.raises(PlanRefusal, match="leaves 'headline' empty, and it must be written"):
            read_prose(PlanProse.model_validate({"prose": items}), slots)


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
        sent = [(call["system"], call["user"], form) for call, form in zip(client.calls, client.response_formats)]
        assert asked == sent, "each call is admitted with exactly the request then sent"
        assert asked[1][0].endswith(PROSE_CALL) and asked[1][2] == prose_response_format()

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


#: The out-of-run cap the toy host holds the planner to.
CAP = TOYHOST_LAUNCH_SETTINGS.max_out_of_run_cost_usd


async def _planner_prompt() -> str:
    return EVAL_REPORT_PLAN_DEFAULT


def _ledger(fixture: Any) -> list[Any]:
    return scope_out_of_run_spend(fixture.host.eval_host, TOYHOST_SCOPE, purpose="analysis").rows


def _attempts(fixture: Any) -> list[Any]:
    return off_loop(
        fixture.host.eval_host.storage.list_analysis_attempts_by_campaign, fixture.campaign.id, TOYHOST_SCOPE
    )


# =============================================================================
# The planner's spend on the operations host (the reporter kind's path): each call priced before it is sent
# =============================================================================


@pytest.mark.parametrize(
    ("ceiling", "said"),
    [(CAP + 1.0, "above the out-of-run cap"), (None, "cannot be priced before they are made")],
    ids=["over-the-cap", "unpriceable"],
)
async def test_a_planned_generation_over_the_cap_is_refused_before_any_call(ceiling: float | None, said: str) -> None:
    fixture = off_loop(planner_fixture, ceiling_usd=ceiling)

    with pytest.raises(ValidationFailedError, match=said):
        await planned_generation(fixture)

    (writer,) = fixture.writers
    assert writer.calls == [], "nothing was sent"
    assert writer.closed == 1, "the client it built was released"
    assert off_loop(_ledger, fixture) == []


async def test_a_planned_generation_within_the_cap_ledgers_every_call() -> None:
    fixture = off_loop(planner_fixture)
    await planned_generation(fixture)

    # The planner's two calls, its structure and its prose, each priced, admitted and ledgered.
    rows = off_loop(_ledger, fixture)
    assert len(rows) == PLAN_CALLS
    for row in rows:
        assert (row.purpose, row.campaign_id, row.subject_id, row.outcome) == (
            "analysis",
            fixture.campaign.id,
            fixture.campaign.subject_id,
            "completed",
        )
        assert (row.priced_ceiling_usd, row.cap_usd) == (FIXTURED_CALL_CEILING_USD, CAP)
        assert row.cost_usd is not None and row.cost_usd > 0


async def test_the_logged_input_size_is_a_ballpark_over_the_exact_first_request_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The size a planned generation discloses before its first call counts what it actually sends (#827): the system
    prompt, user message and response schema, at two characters a token, and the line says so.

    The old figure was ``chars/4`` over a differently rendered bundle, half or less of what a JSON-heavy call bills.
    """
    fixture = off_loop(planner_fixture)
    with caplog.at_level(logging.INFO, logger="threetears.evals"):
        await planned_generation(fixture)

    (writer,) = [each for each in fixture.writers if each.calls]
    sent, response_format = writer.calls[0], writer.response_formats[0]
    assert sent["user"].startswith(f"{REPORT_INPUT_HEADING}\n") and response_format is not None
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("eval.generate_analysis campaign=")]
    logged = int(line.split("approx_input_tokens=")[1].split(" ", 1)[0])

    sent_chars = len(sent["system"]) + len(sent["user"]) + len(json.dumps(response_format))
    assert logged == sent_chars // CHARS_PER_TOKEN
    assert f"({APPROX_INPUT_TOKENS_BASIS})" in line


async def test_with_enforcement_off_a_planned_generation_is_still_ledgered_under_no_cap() -> None:
    off = TOYHOST_LAUNCH_SETTINGS.model_copy(update={"enforcement_enabled": False})
    fixture = off_loop(planner_fixture, ceiling_usd=None, settings=off)
    await planned_generation(fixture)

    rows = off_loop(_ledger, fixture)
    assert [(row.cap_usd, row.priced_ceiling_usd) for row in rows] == [(None, None)] * PLAN_CALLS


async def test_a_prose_call_that_would_pass_the_cap_is_refused_before_it_is_sent() -> None:
    """The structure call fits and is accepted; its prose call would pass what is left, so it is never sent."""
    fixture = off_loop(planner_fixture, ceiling_usd=CAP * 0.6)
    with pytest.raises(ValidationFailedError) as refused:
        await planned_generation(fixture)

    (writer,) = fixture.writers
    assert len(writer.calls) == 1, "the prose call was refused before it was sent"
    assert "call 2 of the plan was refused before it was sent" in str(refused.value)
    assert "above the out-of-run cap" in str(refused.value)
    (attempt,) = off_loop(_attempts, fixture)
    assert (attempt.outcome, attempt.calls, attempt.analysis_id) == ("failed", 1, None)
    assert [row.outcome for row in off_loop(_ledger, fixture)] == ["completed"]
    assert off_loop(fixture.host.eval_host.storage.list_analyses_by_campaign, fixture.campaign.id, TOYHOST_SCOPE) == []


async def test_a_repair_that_would_pass_the_cap_is_refused_before_it_is_sent() -> None:
    """The first call fits; its output is refused; the repair would pass what is left, so it is never sent."""
    fixture = off_loop(planner_fixture, ceiling_usd=CAP * 0.6, first_output="not json at all")
    with pytest.raises(ValidationFailedError, match="call 2 of the plan was refused before it was sent"):
        await planned_generation(fixture)

    (writer,) = fixture.writers
    assert len(writer.calls) == 1, "the repair was refused before it was sent"
    (attempt,) = off_loop(_attempts, fixture)
    assert attempt.calls == 1 and attempt.unpriced_calls == 0, "only the call that was made is counted"
    assert [row.outcome for row in off_loop(_ledger, fixture)] == ["completed"]


async def test_a_repair_within_what_is_left_is_sent_and_every_call_is_ledgered() -> None:
    """A refused structure buys its repair; the repaired structure's prose follows: three calls, all within the cap."""
    fixture = off_loop(planner_fixture, ceiling_usd=CAP * 0.3, first_output="not json at all")
    await planned_generation(fixture)

    assert len(fixture.writers[0].calls) == PLAN_CALLS + 1
    assert [row.outcome for row in off_loop(_ledger, fixture)] == ["completed"] * (PLAN_CALLS + 1)


@pytest.mark.parametrize(("ceiling", "would_start"), [(FIXTURED_CALL_CEILING_USD, True), (CAP + 1.0, False)])
async def test_the_planners_estimate_prices_the_first_call_against_the_cap_and_makes_no_call(
    ceiling: float, would_start: bool
) -> None:
    fixture = off_loop(planner_fixture, ceiling_usd=ceiling)

    estimate = await estimate_analysis_generation(
        fixture.host.eval_host,
        fixture.campaign.id,
        TOYHOST_SCOPE,
        model=None,
        resolve_prompt=_planner_prompt,
        out_of_run_cap_usd=fixture.host.out_of_run_cap(),
    )

    assert (estimate.first_call_ceiling_usd, estimate.cap_usd, estimate.would_start) == (ceiling, CAP, would_start)
    assert (estimate.refusal is None) is would_start
    assert estimate.max_calls == MAX_GENERATION_CALLS == 4
    (writer,) = fixture.writers
    assert writer.calls == [] and writer.closed == 1, "priced, released, never called"
    assert off_loop(_ledger, fixture) == []


# =============================================================================
# The planner's view: over its budget, its preparation refuses 422
# =============================================================================


async def test_the_planners_preparation_over_its_view_budget_refuses_422(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = off_loop(planner_fixture)
    monkeypatch.setattr(report_input_build, "CAMPAIGN_MESSAGE_CHARS_PER_ASPECT", 10)

    async def prompt() -> str:
        return EVAL_REPORT_PLAN_DEFAULT

    with pytest.raises(ValidationFailedError) as refused:
        await prepare_analysis_generation(
            fixture.host.eval_host,
            fixture.campaign.id,
            TOYHOST_SCOPE,
            model=None,
            resolve_prompt=prompt,
            out_of_run_cap_usd=fixture.host.out_of_run_cap(),
        )
    assert refused.value.status_code == 422
    assert "after every trim, over its budget of 10" in refused.value.message
    assert refused.value.remedy is not None and "raise the planner's budget" in refused.value.remedy
