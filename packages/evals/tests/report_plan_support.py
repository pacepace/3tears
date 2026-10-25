"""The report plan as the tests reach it: the reference Kairo round 2 plan, its input, and the planner's part of a plan.

The kit's ``references/r5-kairo-round2.plan.json`` is a stored plan: the planner's choices and prose with every field
code fills. :func:`planned_reply` strips what code fills, leaving what the planner writes; :func:`structure_reply`
and :func:`prose_reply` split that into the planner's two replies, so a test can hand them to the planner as canned
replies. What code fills is listed once here, top level, on a chart and on each unit code has a part of, and the
contract test holds the models to the same lists.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from packages.evals.tests.report_kit_support import KIT

__all__ = [
    "CODE_FILLED",
    "CODE_FILLED_ON_A_CHART",
    "CODE_FILLED_PARTS",
    "R5_INPUT",
    "R5_PLAN",
    "planned_reply",
    "prose_reply",
    "r5_input",
    "r5_plan",
    "structure_reply",
]

#: The kit's reference plan for Kairo round 2.
R5_PLAN = KIT / "references" / "r5-kairo-round2.plan.json"

#: Its report input.
R5_INPUT = Path(__file__).resolve().parent / "fixtures" / "report_plans" / "r5" / "input.json"

#: The fields code fills on the stored plan's top level.
CODE_FILLED = frozenset({"schema", "signals", "shape", "masthead", "arms", "show_arms_key", "full_results"})

#: The fields code fills on each chart.
CODE_FILLED_ON_A_CHART = frozenset({"level", "unit", "domain"})

#: The fields code fills on each unit the planner writes the rest of: the flag's disclosures and the measures it
#: weakens and leaves standing, the options table's columns, and a chart reference's key.
CODE_FILLED_PARTS: dict[str, frozenset[str]] = {
    "Flag": frozenset({"discloses", "weakens", "survives"}),
    "Options": frozenset({"columns"}),
    "ChartReference": frozenset({"key"}),
}


def r5_plan() -> dict[str, Any]:
    """The reference plan, as stored."""
    loaded: dict[str, Any] = json.loads(R5_PLAN.read_text(encoding="utf-8"))
    return loaded


def r5_input() -> dict[str, Any]:
    """The reference plan's report input."""
    loaded: dict[str, Any] = json.loads(R5_INPUT.read_text(encoding="utf-8"))
    return loaded


def planned_reply(plan: dict[str, Any], warnings: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """A stored plan without the fields code fills: what the planner wrote.

    Each answer's warning becomes its position in ``warnings``, the list code raised on the recommended arm, which is
    how the planner names it.

    :param plan: A stored plan, as JSON.
    :param warnings: The warnings code raised on the recommended arm, each as ``{kind, aspect, name}``; None when the
        plan answers none.
    :return: The planner's reply.
    """
    planned = {key: value for key, value in deepcopy(plan).items() if key not in CODE_FILLED}
    for flag in planned["flag"]:
        for key in CODE_FILLED_PARTS["Flag"]:
            del flag[key]
    for key in CODE_FILLED_PARTS["Options"]:
        del planned["options"][key]
    for chart in [*planned["summary"], *(section["chart"] for section in planned["sections"])]:
        for key in CODE_FILLED_ON_A_CHART:
            del chart[key]
        for reference in chart["reference"]:
            for key in CODE_FILLED_PARTS["ChartReference"]:
                del reference[key]
    raised = warnings or []
    planned["answers"] = [
        {"warning": raised.index(answer["warning"]), "text": answer["text"]} for answer in planned["answers"]
    ]
    return planned


def _structure_chart(chart: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in chart.items() if key != "title"}


def structure_reply(plan: dict[str, Any]) -> dict[str, Any]:
    """The planner's structure reply for a stored plan: every choice, no prose.

    :param plan: A stored plan, as JSON.
    :return: The structure call's reply.
    """
    planned = planned_reply(plan)
    return {
        "mode": planned["mode"],
        "headline_kind": planned["headline_kind"],
        "confidence": planned["decision"]["confidence"],
        "options": planned["options"],
        "summary": [_structure_chart(chart) for chart in planned["summary"]],
        "sections": [
            {"id": section["id"], "chart": _structure_chart(section["chart"])} for section in planned["sections"]
        ],
        "explore": [
            {"controls": [{k: v for k, v in control.items() if k != "title"} for control in explore["controls"]]}
            for explore in planned["explore"]
        ],
        "method_items": len(planned["method"]),
        "durable_claims": [{"confidence": c["confidence"], "keys": c["keys"]} for c in planned["durable_claims"]],
    }


def prose_reply(plan: dict[str, Any]) -> dict[str, Any]:
    """The planner's prose reply for a stored plan: every prose slot its structure creates, by key.

    :param plan: A stored plan, as JSON.
    :return: The prose call's reply.
    """
    planned = planned_reply(plan)
    prose: dict[str, str] = {"headline": planned["headline"], "lede": planned["lede"]}
    for flag in planned["flag"]:
        prose |= {"flag.title": flag["title"], "flag.text": flag["text"]}
    prose |= {f"decision.{slot['key']}": slot["text"] for slot in planned["decision"]["slots"]}
    prose |= {f"summary.{index}.title": chart["title"] for index, chart in enumerate(planned["summary"])}
    for section in planned["sections"]:
        where = f"sections.{section['id']}"
        prose |= {
            f"{where}.eyebrow": section["eyebrow"],
            f"{where}.headline": section["headline"],
            f"{where}.takeaway": section["takeaway"],
            f"{where}.title": section["chart"]["title"],
            f"{where}.caption": section["caption"],
        }
    prose |= {f"answers.{answer['warning']}": answer["text"] for answer in planned["answers"]}
    prose |= {f"durable_claims.{index}": claim["text"] for index, claim in enumerate(planned["durable_claims"])}
    for index, item in enumerate(planned["method"]):
        prose |= {f"method.{index}.term": item["term"], f"method.{index}.text": item["text"]}
    for explore in planned["explore"]:
        prose |= {"explore.title": explore["title"], "explore.recomputes": explore["recomputes"]}
        prose |= {f"explore.controls.{control['id']}.title": control["title"] for control in explore["controls"]}
    return {"prose": [{"key": key, "text": text} for key, text in prose.items()]}
