"""``compare(guardrails=...)``: the quick path declares a guardrail, and every arm reads held, breached or undecided.

Before it, a quick guardrail could only be a judged boundary dimension, held at zero change, which an arm alike
with the control never reads ``held``; and :class:`~threetears.evals.quick.Comparison` had no reader for the
guardrails table. A guardrail is declared with its margin and direction, never assumed; it joins no contrast; an
arm that breached one is named above the contrasts as not adopted, and the analysis writer cannot adopt it.

Mutations that turn this file red: declaring a scorer guardrail on a merit axis, or dropping its margin or its
direction; reading a judged guardrail's margin from anywhere but the campaign's ``guardrail_margins``; dropping
the breach line above the contrasts; reading ``undecided`` as safe; accepting a guardrail with no margin, on a
name nothing reports, on a core measure, beside a margin or a host of the caller's own.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, Self

import pytest

from threetears.evals.analysis import (
    EVAL_ANALYSIS_GEN_DEFAULT,
    DisclosureBlock,
    TableBlock,
    generate_analysis,
    get_campaign,
    inspect_campaign_bundle,
)
from threetears.evals.analysis.errors import SoundnessRefusal
from threetears.evals.contracts import (
    CampaignDesign,
    EvalTemplate,
    GuardrailMargin,
    RubricDim,
    StopReason,
    refuse_an_undeclarable_design,
)
from threetears.evals.quick import Comparison, Guardrail, Judge, callable_host, compare, run_eval

#: Fifty cases: an arm alike with the control on every one is shown within 0.1 of it (the bounded test's interval
#: at n=50 is ±0.081; at 40 it is ±0.1002, which does not), and twenty leaks in fifty are shown worse by more.
CASES = [{"n": index} for index in range(50)]

NO_LEAK = Guardrail(margin=0.1, direction="higher_is_better")


def correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is right."""
    return answer.startswith("right")


def no_leak(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer keeps the customer's secret."""
    return "SECRET" not in answer


def leaked(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer gave the customer's secret away."""
    return "SECRET" in answer


def politeness(case: Mapping[str, Any], answer: str) -> float:
    """How polite the answer is, from 0 to 1."""
    return 2.0 if "please" in answer else 1.0


def _answers(*, misses: int, leaks: int, tail: str = "") -> Any:
    async def candidate(case: Mapping[str, Any]) -> str:
        answer = "wrong" if case["n"] < misses else "right"
        return answer + (" SECRET" if case["n"] < leaks else "") + tail

    return candidate


#: The control misses ten cases and leaks none; ``same`` answers as it does; ``leaky`` misses none and leaks
#: twenty; ``slip`` answers as the control does and leaks three, too few to be shown either side of 0.1.
ARMS = {
    "current": _answers(misses=10, leaks=0),
    "same": _answers(misses=10, leaks=0),
    "leaky": _answers(misses=0, leaks=20),
    "slip": _answers(misses=10, leaks=3),
}


async def _compare(**overrides: Any) -> Comparison:
    arguments: dict[str, Any] = {
        "control": "current",
        "scope_id": "quick-guardrails",
        "k": 1,
        "guardrails": {"no_leak": NO_LEAK},
    }
    arguments.update(overrides)
    return await compare(CASES, arguments.pop("arms", ARMS), arguments.pop("scorers", [correct, no_leak]), **arguments)


def _lines(comparison: Comparison, source: str) -> list[str]:
    return [
        block.text
        for block in comparison.report.blocks
        if isinstance(block, DisclosureBlock) and block.source == source and block.section == "surface"
    ]


class TestAScorerGuardrail:
    async def test_each_arm_reads_held_breached_or_undecided(self) -> None:
        comparison = await _compare()
        outcomes = {row["arm"]: row["outcome"] for row in comparison.guardrails()}
        assert outcomes == {"same": "held", "leaky": "breached", "slip": "undecided"}
        leaky = comparison.guardrails("leaky")
        assert [(row["measure_id"], row["guardrail"], row["margin"]) for row in leaky] == [
            ("no_leak", "No leak score", "0.1")
        ]
        assert leaky[0]["delta"] == pytest.approx(-0.4) and leaky[0]["decision"].startswith("breached")
        (slip,) = comparison.guardrails("slip")
        assert slip["decision"].startswith("undecided: not shown held, so not known to be safe")
        assert "reaches both sides of -0.1 (the declared margin)" in slip["decision"]
        (same,) = comparison.guardrails("same")
        assert same["interval"] == "[-0.08106, 0.08106] at 95%", "a pass/fail is on 0 to 1: the bounded test's"
        assert comparison.guardrail_readings.checks[0].interval_basis == "bounded"

    async def test_the_standing_of_each_arm_and_none_for_the_control(self) -> None:
        comparison = await _compare()
        assert comparison.guardrail_standing("leaky").breached == ["no_leak"]
        assert comparison.guardrail_standing("slip").undecided == ["no_leak"]
        assert comparison.guardrail_standing("same").held == ["no_leak"]
        assert comparison.guardrail_standing("current") == ([], [], []), "no guardrail is checked against itself"
        with pytest.raises(ValueError, match="arm 'nope' names no arm"):
            comparison.guardrail_standing("nope")

    async def test_it_is_declared_a_guardrail_off_every_merit_axis_with_its_margin_and_direction(self) -> None:
        comparison = await _compare()
        measure = comparison.host.profile.measures.get("no_leak")
        assert measure is not None and measure.guardrail and measure.merit_axis is None
        assert (measure.materiality_threshold, measure.higher_is_better, measure.value_range) == (0.1, True, (0, 1))
        assert comparison.contrasts("no_leak") == [], "a guardrail joins no contrast"
        assert {row["measure_id"] for row in comparison.contrasts()} == {"correct"}

    async def test_the_printed_report_shows_the_table_and_names_the_breach_above_the_contrasts(self) -> None:
        comparison = await _compare()
        markdown = comparison.render()
        assert "**Guardrails against the control**" in markdown
        for outcome in ("held", "breached", "undecided"):
            assert f"| {outcome}: " in markdown
        breach, undecided = _lines(comparison, "guardrails")
        assert breach == (
            "Arm leaky breached the guardrail No leak score: it is shown worse than the control by more than the "
            "margin, so it is not adopted, whatever the contrasts below show it gained."
        )
        assert undecided.startswith("Arm slip is undecided on the guardrail No leak score, so it is not known to be")
        blocks = comparison.report.blocks
        table = next(i for i, b in enumerate(blocks) if isinstance(b, TableBlock) and b.name == "comparisons")
        assert [b.text for b in blocks[table - 2 : table]] == [breach, undecided], "directly above the contrasts"
        (gain,) = [row for row in comparison.contrasts("correct") if row["arm"] == "leaky"]
        assert gain["verdict"] == "improved on the control", "the breached arm gained, and is still not adopted"

    async def test_with_every_scorer_a_guardrail_the_breach_is_named_under_the_guardrails_table(self) -> None:
        comparison = await _compare(scorers=[no_leak])
        blocks = comparison.report.blocks
        assert comparison.contrasts() == [], "nothing but the guardrail is graded"
        named = [
            i
            for i, b in enumerate(blocks)
            if isinstance(b, DisclosureBlock) and b.text.startswith("Arm leaky breached")
        ]
        assert len(named) == 1 and blocks[named[0]].section == "guardrails"
        assert blocks[named[0]].text.endswith("so it is not adopted.")
        assert blocks[named[0] - 1].section == "guardrails", "directly under the guardrails section"

    async def test_a_second_control_reads_the_same_guardrail(self) -> None:
        comparison = await _compare()
        again = comparison.against("same")
        assert {row["arm"]: row["outcome"] for row in again.guardrails()}["leaky"] == "breached"

    async def test_lower_is_better_runs_the_other_way(self) -> None:
        comparison = await _compare(
            scorers=[correct, leaked], guardrails={"leaked": Guardrail(margin=0.1, direction="lower_is_better")}
        )
        outcomes = {row["arm"]: row["outcome"] for row in comparison.guardrails()}
        assert outcomes == {"same": "held", "leaky": "breached", "slip": "undecided"}
        assert comparison.host.profile.measures.get("leaked").higher_is_better is False  # type: ignore[union-attr]
        reasons = [reason for miss in comparison.misses("leaky") for reason in miss.missed_because]
        assert reasons == ["leaked gave 1"] * 20, "a leak is the miss on a lower-is-better guardrail, and none is not"

    async def test_a_bounded_scorer_holds_on_its_declared_range_and_a_value_outside_it_is_the_scorer_s_fault(
        self,
    ) -> None:
        arms = {"current": _answers(misses=0, leaks=0), "same": _answers(misses=0, leaks=0)}
        polite = {"politeness": Guardrail(margin=0.2, direction="higher_is_better")}
        ranged = {"politeness": (0.0, 1.0)}
        comparison = await _compare(arms=arms, scorers=[politeness], guardrails=polite, ranges=ranged)
        assert comparison.host.profile.measures.get("politeness").value_range == (0.0, 1.0)  # type: ignore[union-attr]
        assert comparison.arms["same"].n_excluded == 0
        (held,) = comparison.guardrails()
        assert held["outcome"] == "held" and "(t: no declared range)" not in held["interval"]

        rude = {"current": _answers(misses=0, leaks=0), "please": _answers(misses=0, leaks=0, tail=" please")}
        comparison = await _compare(arms=rude, scorers=[politeness], guardrails=polite, ranges=ranged)
        summary = comparison.arms["please"]
        assert summary.n_excluded == len(CASES) and summary.n_scored == 0
        assert "outside the range 0 to 1" in summary.errors[0]

    async def test_with_no_range_a_reading_is_never_held_and_the_reason_names_ranges(self) -> None:
        arms = {"current": _answers(misses=0, leaks=0), "same": _answers(misses=0, leaks=0)}
        polite = {"politeness": Guardrail(margin=0.2, direction="higher_is_better")}
        (row,) = (await _compare(arms=arms, scorers=[politeness], guardrails=polite)).guardrails()
        assert row["outcome"] == "undecided" and "ranges= beside the scorer" in row["decision"]
        assert "held is never read off it" in row["decision"]

    async def test_with_no_range_a_spread_inside_the_margin_is_still_not_held(self) -> None:
        """The t interval sits inside the margin, which read ``held`` until #695's rule reached guardrails."""

        def kind(case: Mapping[str, Any], answer: str) -> float:
            """How kind the answer is."""
            return 1.0 + (0.1 if answer.endswith("please") and case["n"] % 2 else 0.0)

        arms = {"current": _answers(misses=0, leaks=0), "warmer": _answers(misses=0, leaks=0, tail=" please")}
        kindness = {"kind": Guardrail(margin=0.2, direction="higher_is_better")}
        comparison = await _compare(arms=arms, scorers=[kind], guardrails=kindness)
        (check,) = comparison.guardrail_readings.checks
        assert check.interval_basis == "t" and check.interval is not None and check.interval[0] > -0.2
        assert check.decision == "undecided" and check.undecided_reason is not None
        assert "ranges= beside the scorer" in check.undecided_reason


class TestABreachedArmIsNeverRecommended:
    async def test_the_analysis_writer_cannot_adopt_it_whatever_it_gained(self) -> None:
        comparison = await _compare()
        bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id).bundle
        adopting = _adopting_writer()
        with pytest.raises(SoundnessRefusal, match="breached the guardrail no_leak"):
            await generate_analysis(
                bundle,
                prompt=EVAL_ANALYSIS_GEN_DEFAULT,
                model="offline-writer",
                client=adopting,
                prompt_id="eval_analysis_gen",
                bundle_assembled_at=datetime.now(UTC).isoformat(),
                profile=comparison.host.profile,
            )
        assert adopting.drafts >= 1


def _adopting_writer() -> Any:
    """A scripted writer whose every draft adopts the arm that improved on the control, which breached the guardrail."""
    writer = SimpleNamespace(drafts=0)

    async def generate(*, system: str, user: str, response_format: dict | None = None) -> Any:
        writer.drafts += 1
        evidence, _ = json.JSONDecoder().raw_decode(user, user.index("{"))
        tested = next(
            comparison
            for family in evidence["multiple_comparisons"]["families"]
            for comparison in family["comparisons"]
            if comparison["verdict"] == "improved"
        )
        arm, control = tested["contrast"]["cell"], tested["control"]["cell"]
        cite = {cell: "{{" + f"{cell}|{tested['name']}|measure|mean" + "}}" for cell in (arm, control)}
        finding = {
            "title": "Correct score by arm",
            "body": f"The tested arm reads {cite[arm]}, the control {cite[control]}.",
            "evidence": [{"cell": cell, "measure_id": tested["name"], "reading": "measure"} for cell in cite],
            "chart": {"type": "none", "cells": [], "measures": [], "axis": "", "note": "", "caption": ""},
            "confidence": "low",
            "axes": [],
            "caveats": [],
            "invalidates": [],
            "durable": "",
        }
        decision = {
            "proposal": "Ship the arm that answers more cases correctly.",
            "disposition": "adopted",
            "cells": [arm],
            "confidence": "medium",
            "rests_on": [0],
            "revisit_when": "",
        }
        memo = {"headline": "Adopt the more accurate arm", "summary": "- It is more accurate."}
        memo |= {"findings": [finding], "decisions": [decision], "questions": [], "next": []}
        return _Reply(content=json.dumps(memo), temperature=None, model="offline-writer")

    writer.generate = generate
    return writer


class TestARefusedGuardrail:
    @pytest.mark.parametrize(
        ("arguments", "said"),
        [
            ({"margin": None, "direction": "higher_is_better"}, "no margin is assumed"),
            ({"margin": 0, "direction": "higher_is_better"}, "a positive number in the reading's own units"),
            ({"margin": float("nan"), "direction": "higher_is_better"}, "a positive number"),
            ({"margin": True, "direction": "higher_is_better"}, "a positive number"),
            ({"margin": 0.1, "direction": "up"}, "which way is better on it"),
        ],
    )
    def test_one_no_arm_could_be_decided_against(self, arguments: dict[str, Any], said: str) -> None:
        with pytest.raises(ValueError, match=re.escape(said)):
            Guardrail(**arguments)

    def test_with_no_margin_or_direction_at_all(self) -> None:
        with pytest.raises(TypeError, match="margin"):
            Guardrail(direction="higher_is_better")  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="direction"):
            Guardrail(margin=0.1)  # type: ignore[call-arg]

    async def test_a_bare_margin_teaches_the_guardrail(self) -> None:
        with pytest.raises(ValueError, match=r"maps a name to a Guardrail, its margin and direction both declared"):
            await _compare(guardrails={"no_leak": 0.1})

    async def test_on_a_name_nothing_reports_it_names_what_does(self) -> None:
        with pytest.raises(ValueError, match=r"names 'no_leek', which no scorer reports.*'correct', 'no_leak'"):
            await _compare(guardrails={"no_leek": NO_LEAK})

    @pytest.mark.parametrize("name", ["accuracy", "match"])
    async def test_on_a_core_measure_it_teaches_the_scorer_route(self, name: str) -> None:
        with pytest.raises(ValueError, match=r"engine core measure.*declare the guardrail on that"):
            await _compare(guardrails={name: NO_LEAK})

    async def test_beside_a_margin_on_the_same_scorer(self) -> None:
        with pytest.raises(ValueError, match=r"'no_leak' is declared a guardrail and given a margin in margins= too"):
            await _compare(margins={"no_leak": 0.05})

    async def test_beside_a_host_of_your_own(self) -> None:
        with pytest.raises(ValueError, match=r"MetricDescriptor\(guardrail=True\).*pass one or the other"):
            await _compare(host=callable_host([correct, no_leak]))

    @pytest.mark.parametrize(
        ("scorers", "guardrail", "ranges"),
        [([no_leak], "no_leak", None), ([politeness], "politeness", {"politeness": (0.0, 1.0)})],
    )
    async def test_as_wide_as_every_value_the_scorer_can_return(
        self, scorers: list[Any], guardrail: str, ranges: dict[str, tuple[float, float]] | None
    ) -> None:
        wide = {guardrail: Guardrail(margin=1.0, direction="higher_is_better")}
        with pytest.raises(ValueError, match="is as wide as every value it can return"):
            await _compare(scorers=scorers, guardrails=wide, ranges=ranges)


class TestTheSingleRunSummary:
    async def test_names_a_guardrail_and_says_a_run_alone_decides_nothing(self) -> None:
        host = callable_host([correct, no_leak], guardrails={"no_leak": NO_LEAK})
        summary = await run_eval(CASES, ARMS["leaky"], [correct, no_leak], scope_id="one-guardrail", host=host, k=1)
        (measure,) = [m for m in summary.measures if m.name == "no_leak"]
        assert measure.guardrail and not next(m for m in summary.measures if m.name == "correct").guardrail
        assert (
            "  no_leak: mean 0.6 (n=50, min 0, max 1); a guardrail: held, breached or undecided is decided only "
            "against a control, in a comparison"
        ) in summary.render()


# --- a judged guardrail ----------------------------------------------------------------------------------------

JUDGE_MODEL = "scripted/judge"


@dataclass(frozen=True)
class _Reply:
    """One scripted completion, in the attribute names the engine reads."""

    content: str
    temperature: float | None
    input_tokens: int | None = 20
    output_tokens: int | None = 10
    cost_usd: float | None = 0.0001
    price_source: str | None = "scripted rate"
    model: str = JUDGE_MODEL
    served_model: str | None = JUDGE_MODEL
    reasoning_tokens: int | None = None
    stop_reason: StopReason = "end_turn"


# parity-with: threetears.evals.contracts.CompletionClient
class _FakeLeakJudge:
    """Passes an answer on every dimension unless it gives the secret away."""

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        match = re.search(r'single key "([^"]+)"', system)
        assert match is not None, "the judge's system prompt names no dimension"
        output = user.split("# Output under review\n", 1)[1]
        verdict = "fail" if "SECRET" in output else "pass"
        return _Reply(
            content=json.dumps({"reasoning": "read it", "criteria_scores": {match[1]: verdict}}), temperature=None
        )

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _judge() -> Judge:
    return Judge(
        client=_FakeLeakJudge(),
        model=JUDGE_MODEL,
        rubric={"private": "The answer gives away no customer's secret.", "clear": "The answer is clear."},
        scale="pass_fail",
    )


class TestAJudgedGuardrail:
    async def test_it_is_scored_on_the_boundary_axis_and_held_to_the_campaign_s_margin(self) -> None:
        arms = {"current": ARMS["current"], "same": ARMS["same"], "leaky": ARMS["leaky"]}
        comparison = await _compare(arms=arms, scorers=[correct], judge=_judge(), guardrails={"private": NO_LEAK})
        outcomes = {row["arm"]: (row["measure_id"], row["outcome"], row["margin"]) for row in comparison.guardrails()}
        assert outcomes == {"same": ("answer.private", "held", "0.1"), "leaky": ("answer.private", "breached", "0.1")}
        assert {row["measure_id"] for row in comparison.contrasts()} == {"correct", "answer.clear"}

        design = get_campaign(comparison.host.storage, comparison.campaign_id, comparison.scope_id).declared_design
        assert design is not None and design.guardrail_margins == [
            GuardrailMargin(dimension="answer.private", margin=0.1)
        ]
        again = comparison.against("same")
        assert {row["arm"]: row["outcome"] for row in again.guardrails()}["leaky"] == "breached"

    async def test_with_no_margin_declared_it_is_held_at_zero_change_and_never_reads_held_when_alike(self) -> None:
        dims = [
            RubricDim(name="answer.private", description="No secret given away.", scale="pass_fail", axis="boundary"),
        ]
        judge = Judge(client=_FakeLeakJudge(), model=JUDGE_MODEL, rubric=dims)
        arms = {"current": ARMS["current"], "same": ARMS["same"]}
        comparison = await _compare(arms=arms, scorers=[correct], judge=judge, guardrails=None)
        (row,) = comparison.guardrails()
        assert (row["outcome"], row["margin"]) == ("undecided", "0 (none declared)")

    @pytest.mark.parametrize(
        ("guardrail", "said"),
        [
            (Guardrail(margin=0.1, direction="lower_is_better"), "judged with higher better"),
            (Guardrail(margin=1.0, direction="higher_is_better"), "is as wide as its scale"),
        ],
    )
    async def test_refused_where_a_judged_scale_cannot_carry_it(self, guardrail: Guardrail, said: str) -> None:
        with pytest.raises(ValueError, match=re.escape(said)):
            await _compare(scorers=[correct], judge=_judge(), guardrails={"private": guardrail})


class TestTheDeclarationGate:
    """A campaign's judged guardrail margin names a boundary dimension of its template, narrower than its scale."""

    def _template(self) -> EvalTemplate:
        return EvalTemplate(
            id="t",
            scope_id="s",
            name="t",
            intent="answer",
            candidate_kind="callable-judged",
            rubric=[
                RubricDim(name="answer.private", description="No secret.", scale="pass_fail", axis="boundary"),
                RubricDim(name="answer.clear", description="Clear.", scale="ordinal"),
            ],
        )

    def _refuse(self, margins: list[GuardrailMargin], template: EvalTemplate | None) -> None:
        design = CampaignDesign(
            axes=[{"axis_id": "model", "values": [{"content": "a", "display": "a"}], "rationale": "x"}],
            held_fixed={"stimulus": "controlled", "apparatus": "commissioned"},
            guardrail_margins=margins,
        )
        refuse_an_undeclarable_design(design, behavior="score", template=template, profile=callable_host().profile)

    def test_a_boundary_dimension_passes(self) -> None:
        self._refuse([GuardrailMargin(dimension="answer.private", margin=0.1)], self._template())

    @pytest.mark.parametrize(
        ("dimension", "margin", "said"),
        [
            ("answer.clear", 0.5, "it is on the capability axis, which is compared, not held"),
            ("answer.nope", 0.5, "the template declares no such dimension"),
            ("answer.private", 1.0, "is as wide as its scale"),
        ],
    )
    def test_anything_else_is_refused(self, dimension: str, margin: float, said: str) -> None:
        with pytest.raises(ValueError, match=re.escape(said)):
            self._refuse([GuardrailMargin(dimension=dimension, margin=margin)], self._template())

    def test_with_no_template_it_says_so(self) -> None:
        with pytest.raises(ValueError, match="names no template that could be read"):
            self._refuse([GuardrailMargin(dimension="answer.private", margin=0.1)], None)

    def test_two_on_one_dimension(self) -> None:
        with pytest.raises(ValueError, match="more than one guardrail margin on: answer.private"):
            self._refuse([GuardrailMargin(dimension="answer.private", margin=m) for m in (0.1, 0.2)], self._template())
