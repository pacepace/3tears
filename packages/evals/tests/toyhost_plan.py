"""A report planner's two replies, written by code for any report input a decide-mode plan fits — test support.

A generation is written by the report planner in two calls (``threetears.evals.analysis.planner``): a structure, then
its prose. A test that drives a generation end to end needs a planner whose replies pass the plan checks for whatever
campaign it generates over, so :func:`structure_reply` chooses the structure code's own facts permit (the dominant
option recommended, each open caveat landed once, one section drawing the spine code gives, one durable claim), and
:func:`prose_reply`
writes every prose key that structure creates, with no digit outside a figure key and, in Watch, the weakest margin
by its keys (money per 1,000 replies, the basis a page states its costs in). :class:`PlanReplies` answers each
call of a generation by the contract it was sent.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from threetears.evals.analysis.plan_assembly import check_structure, prose_slots
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_decision import dominant_options, surviving_options
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.kernel.report_plan import PROSE_SCHEMA_NAME, STRUCTURE_SCHEMA_NAME, PlanStructure
from packages.evals.tests.toyhost_memo import FixturedClient, FixturedCompletion

__all__ = ["PLAN_CALLS", "PlanWriter", "contract_name", "prose_reply", "replies_for", "structure_reply"]

#: The calls an unrepaired plan takes: its structure and its prose.
PLAN_CALLS = 2

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
    recommended = next(row.option for row in chosen.options.rows if row.role == "recommended")
    key = _claim_key(report_input, recommended)
    claim = (
        f"{{arm:{recommended}}} passed {{{key}|pct}} of its trials."
        if key.startswith("pass_rate.")
        else f"{{arm:{recommended}}} was the option recommended."
    )
    texts: dict[str, str] = {
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
        if name == STRUCTURE_SCHEMA_NAME:
            return json.dumps(structure_reply(built))
        if name == PROSE_SCHEMA_NAME:
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
