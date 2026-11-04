"""A report planner's two replies, written by code for any report input a decide-mode plan fits — test support.

A generation is written by the report planner in two calls (``threetears.evals.analysis.planner``): a structure, then
its prose. A test that drives a generation end to end needs a planner whose replies pass the plan checks for whatever
campaign it generates over, so :func:`structure_reply` chooses the structure code's own facts permit (the dominant
option recommended, each open caveat landed once, one section drawing the spine code gives, one durable claim), and
:func:`prose_reply`
writes every prose key that structure creates, with no digit outside a figure key and, in Watch, the weakest margin
by its keys (money per 1,000 replies, the basis a page states its costs in). :class:`PlanReplies` answers each
call of a generation by the contract it was sent.

**The planner on the operations host.** :func:`planner_fixture` is the toy operations host
(:func:`~packages.evals.tests.ops_support.ops_fixture`) with a client for the planner's role, the ``analysis`` role:
:class:`MemoWriter`, which writes the plan over the campaign's bundle as the generation assembled it (or, sent the
memo writer's contract, the toy memo). :func:`planned_generation` runs the planner's path (``prepare_analysis_generation``
and ``run_analysis_generation``) on it, which the reporter kind still runs, and stores a real planned analysis.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from threetears.evals.analysis.plan_assembly import check_structure, prose_slots
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_decision import dominant_options, surviving_options
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.kernel.report_plan import PlanStructure, prose_response_format, structure_response_format
from threetears.evals.analysis import assemble_campaign_bundle, prepare_analysis_generation, run_analysis_generation
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel.campaign import EvalAnalysis, EvalCampaign, EvalInsight
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.prompts.catalogue import REPORT_PLAN_PROMPT
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import OpsFixture, ops_fixture
from packages.evals.tests.toyhost_memo import (
    FIXTURED_CALL_CEILING_USD,
    FixturedClient,
    FixturedCompletion,
    memo_payload,
)

__all__ = [
    "PLAN_CALLS",
    "PROSE_CONTRACT",
    "STRUCTURE_CONTRACT",
    "MemoWriter",
    "PlanWriter",
    "contract_name",
    "planned_generation",
    "planner_fixture",
    "prose_reply",
    "replies_for",
    "structure_reply",
]

#: The calls an unrepaired plan takes: its structure and its prose.
PLAN_CALLS = 2

#: The names the planner's two contracts are sent under, read off the directives a generation sends.
STRUCTURE_CONTRACT: str = structure_response_format()["json_schema"]["name"]
PROSE_CONTRACT: str = prose_response_format()["json_schema"]["name"]

#: Where a spine reads when code names no source: the first of these the input holds rows for.
_SPINE_SOURCES = ("tasks", "floors", "measures", "configs", "strata")


def _spine_chart(report_input: ReportInput) -> dict[str, Any]:
    """The one section's chart: the spine code gives (``PlanFacts.spine``), every query field it fixes copied, and a
    source the input holds where code names none (over the measures, the first of them, for one unit)."""
    spine = plan_facts(report_input).spine
    held = {
        "tasks": report_input.tasks,
        "floors": report_input.floors,
        "measures": report_input.measures,
        "configs": report_input.configs,
        "strata": report_input.strata,
    }
    source = spine.source or next(name for name in _SPINE_SOURCES if held[name])
    # A chart carries one unit, so a spine over the measures that code names none for reads the first of them.
    measures = list(spine.measures or ()) or (list(report_input.measures)[:1] if source == "measures" else [])
    return {
        "kind": spine.kind,
        "query": {
            "source": source,
            "aspect": spine.aspect,
            "measures": measures,
            "arms": [],
            "strata": [],
            "limit": 0,
            "transform": "none",
            "small_multiples_by": "none",
        },
        "reference": [{"kind": kind} for kind in spine.reference],
        "emphasis": "",
    }


def contract_name(response_format: Any) -> str | None:
    """The name a strict response format was sent under; None for a call sent without one.

    :param response_format: What the call was sent with.
    :return: Its schema's name.
    """
    if not isinstance(response_format, dict):
        return None
    schema = response_format.get("json_schema")
    return str(schema.get("name")) if isinstance(schema, dict) else None


def _recommended(report_input: ReportInput) -> str:
    dominant = dominant_options(report_input)
    if dominant:
        return dominant[0]
    survivors = surviving_options(report_input)
    return survivors[0] if survivors else report_input.arms[-1].id


def _claim_key(report_input: ReportInput, arm: str) -> str:
    return f"pass_rate.{arm}" if arm in report_input.scoreboard else f"arm:{arm}"


def structure_reply(report_input: ReportInput) -> dict[str, Any]:
    """A decide-mode structure the plan checks accept for ``report_input``.

    :param report_input: The report input the generation planned over.
    :return: The structure call's reply.
    :raises AssertionError: Code allows no decide mode for this input, or the checks refuse the structure.
    """
    facts = plan_facts(report_input)
    assert "decide" in facts.allowed_modes, (
        f"a decide-mode test plan needs decide allowed; code allows {facts.allowed_modes}"
    )
    recommended = _recommended(report_input)
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    rows = [{"option": recommended, "role": "recommended"}]
    rows += [
        {"option": arm.id, "role": "control" if arm.id == control else "alternative"}
        for arm in report_input.arms
        if arm.id != recommended
    ][:4]
    open_caveats = [caveat for caveat in facts.caveats if not caveat.checked]
    landings = [
        {
            "caveat": caveat.id,
            "at": "confidence"
            if caveat.reach == "report"
            else ("method.0" if caveat.reach == "method" else "decision.watch"),
        }
        for caveat in open_caveats
    ]
    structure = {
        "mode": "decide",
        "headline_kind": "decision",
        "confidence": "low",
        "caveat_landings": landings,
        "options": {"rows": rows},
        "summary": [],
        "sections": [{"id": "outcomes", "chart": _spine_chart(report_input)}],
        "explore": [],
        "method_items": 3,
        "durable_claims": [{"confidence": "low", "keys": [_claim_key(report_input, recommended)]}],
    }
    check_structure(PlanStructure.model_validate(structure), report_input, facts, facts.allowed_modes)
    return structure


def prose_reply(report_input: ReportInput, structure: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every prose key ``structure`` creates, each written, figures by key and no digit outside one.

    :param report_input: The report input the generation planned over.
    :param structure: The structure the prose is for; :func:`structure_reply`'s when None.
    :return: The prose call's reply.
    """
    chosen = PlanStructure.model_validate(structure or structure_reply(report_input))
    recommended = next((row.option for row in chosen.options.rows if row.role == "recommended"), None)
    if recommended is None:
        # A trade-off or a no-call sets no option apart: its prose names the first option the table lists.
        named = chosen.options.rows[0].option
        texts = {
            "headline": f"{{arm:{named}}} and the others trade one strength for another.",
            "lede": f"{{arm:{named}}} leads on one count and trails on another. More cases would settle it.",
        }
    else:
        key = _claim_key(report_input, recommended)
        claim = (
            f"{{arm:{recommended}}} passed {{{key}|pct}} of its trials."
            if key.startswith("pass_rate.")
            else f"{{arm:{recommended}}} was the option recommended."
        )
        texts = {
            "headline": f"Choose {{arm:{recommended}}}: it is the option the evidence favours.",
            "lede": f"{{arm:{recommended}}} is ahead where it counts. More cases would settle it.",
            "durable_claims.0": claim,
        }
    facts = plan_facts(report_input)
    if facts.weakest is not None:
        money = resolve("weakest.margin", report_input).scale == "usd"
        margin = "{weakest.margin|usd_per_1k}" if money else "{weakest.margin}"
        texts["decision.watch"] = f"The weakest margin is {{weakest.arm}} on {{weakest.criterion}}, a gap of {margin}."
    return {
        "prose": [
            {"key": slot.key, "text": texts.get(slot.key, "The evidence reads this way, as the chart shows.")}
            for slot in prose_slots(chosen, facts)
        ]
    }


def replies_for(report_input: Callable[[], ReportInput]) -> Callable[[Any], str]:
    """A reply writer for a generation: the structure or the prose, by the contract a call was sent.

    :param report_input: Reads the report input the generation planned over.
    :return: What answers a call, given its response format.
    """

    def reply(response_format: Any) -> str:
        name = contract_name(response_format)
        built = report_input()
        if name == STRUCTURE_CONTRACT:
            return json.dumps(structure_reply(built))
        if name == PROSE_CONTRACT:
            return json.dumps(prose_reply(built))
        raise AssertionError(f"a planner call was sent under {name!r}, which is neither of the planner's contracts")

    return reply


class PlanWriter(FixturedClient):
    """A fixtured planner: each call answered by the contract it was sent, over the report input it is given.

    Set :attr:`report_input` (a prepared generation's) before the generation runs.
    """

    def __init__(self) -> None:
        """Start with no report input: the generation's is set once it is prepared."""
        super().__init__("")
        self.report_input: ReportInput | None = None

    async def generate(
        self, *, system: str, user: str, response_format: Any = None, tools: Any = None
    ) -> FixturedCompletion:
        """Write the reply the call's contract asks for, then hand it back as the completion."""
        built = self.report_input
        assert built is not None, "set the prepared generation's report input before it runs"
        self.completion.content = replies_for(lambda: built)(response_format)
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


class MemoWriter(FixturedClient):
    """The fixtured generator, writing the toy plan's structure and prose (or, sent the memo writer's contract, the toy
    memo) over the bundle the generation is about to send."""

    def __init__(self, host: EvalHost, campaign: EvalCampaign) -> None:
        """Bind the writer to the campaign it analyses.

        :param host: Where the campaign's runs are read.
        :ptype host: EvalHost
        :param campaign: The campaign the generation is over.
        :ptype campaign: EvalCampaign
        """
        super().__init__("")
        self._host = host
        self._campaign = campaign
        #: Set to hold the generation open until the test releases it.
        self.gate: asyncio.Event | None = None
        #: When set, what the FIRST call returns in place of the memo — an output the generator refuses,
        #: so the generation buys its repair round-trip.
        self.first_output: str | None = None
        #: When set, what every call raises once the gate opens — a provider's failure, as its SDK raises one.
        self.raises: Exception | None = None

    async def generate(self, *, system: str, user: str, response_format: Any = None, tools: Any = None) -> Any:
        """Write the memo over the campaign's bundle, then hand it back as the completion."""
        if self.gate is not None:
            await self.gate.wait()
        if self.raises is not None:
            raise self.raises
        if self.first_output is not None and not self.calls:
            self.completion.content = self.first_output
            return await super().generate(system=system, user=user, response_format=response_format, tools=tools)
        bundle = off_loop(assemble_campaign_bundle, self._host, self._campaign.id, self._campaign.scope_id)
        if contract_name(response_format) in (STRUCTURE_CONTRACT, PROSE_CONTRACT):
            self.completion.content = replies_for(lambda: report_input(bundle))(response_format)
        else:
            self.completion.content = json.dumps(memo_payload(bundle))
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


def planner_fixture(
    *,
    gate: asyncio.Event | None = None,
    ceiling_usd: float | None = FIXTURED_CALL_CEILING_USD,
    first_output: str | None = None,
    **given: Any,
) -> OpsFixture:
    """The toy operations host with the planner's client: every role but the analyst's is a :class:`MemoWriter`.

    :param gate: Holds every generation, the planner's and the analyst's, open until set.
    :ptype gate: asyncio.Event | None
    :param ceiling_usd: What each planner call (and each ``analyst`` turn) is priced at, at most; ``None`` for a
        writer that cannot price.
    :ptype ceiling_usd: float | None
    :param first_output: What each planner client's first call returns in place of its reply, when set.
    :ptype first_output: str | None
    :param given: The rest of :func:`~packages.evals.tests.ops_support.ops_fixture`'s parameters.
    :return: The fixture; each planner client built is added to its ``writers``.
    :rtype: OpsFixture
    """
    fixture = ops_fixture(gate=gate, ceiling_usd=ceiling_usd, **given)
    eval_host = fixture.host.eval_host
    analyst = eval_host.clients
    assert analyst is not None

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> Any:
        if role == "analyst":
            return analyst(role, model, temperature=temperature)
        writer = MemoWriter(eval_host, fixture.campaign)
        writer.gate = gate
        writer.ceiling_usd = ceiling_usd
        writer.first_output = first_output
        fixture.writers.append(writer)
        return writer

    object.__setattr__(eval_host, "clients", clients)
    return fixture


async def _planner_prompt() -> str:
    # The planner's prompt, as a host registers it.
    return EVAL_REPORT_PLAN_DEFAULT


async def planned_generation(
    fixture: OpsFixture, campaign_id: str | None = None, *, model: str | None = None
) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """A planned analysis of the corpus campaign, by the report planner's path (which the reporter kind runs), inline.

    :param fixture: The toy host, as :func:`planner_fixture` builds it.
    :ptype fixture: OpsFixture
    :param campaign_id: The campaign; the corpus campaign when ``None``.
    :ptype campaign_id: str | None
    :param model: The planner model, or ``None`` for the host's default.
    :ptype model: str | None
    :return: The stored analysis and its insights; its plan reads with ``get_report_plan``.
    :rtype: tuple[EvalAnalysis, list[EvalInsight]]
    """
    eval_host = fixture.host.eval_host
    target = campaign_id or fixture.campaign.id
    prepared = await prepare_analysis_generation(
        eval_host,
        target,
        fixture.campaign.scope_id,
        model=model,
        resolve_prompt=_planner_prompt,
        out_of_run_cap_usd=fixture.host.out_of_run_cap(),
    )
    return await run_analysis_generation(eval_host, prepared, prompt_id=REPORT_PLAN_PROMPT, max_output_tokens=4000)
