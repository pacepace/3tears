"""Build the five report-plan fixtures' report inputs from the eval report kit's reference data.

Run from the repository root: ``uv run python packages/evals/tests/fixtures/report_plans/build_inputs.py``. It writes
``r1/input.json`` to ``r5/input.json`` beside this file, each a valid
:class:`~threetears.evals.analysis.report_input.ReportInput`, and running it again writes the same bytes.

The references r1 to r4 (``docs/eval-report-kit/references/r{1..4}-*.json``) are page-shaped synthetic data: each is
what its reference page's script reads, not a report input. This script carries every figure each page prints that a
report input would hold — arms with their settings, tasks, criteria, per-arm figures with their intervals, costs,
latency, strata, examples — into the report input's fields, and marks each ``synthetic``. It copies figures; it never
recomputes one. r5 is a real report input (``references/r5-kairo-round2.json``), copied byte for byte.

What each input deliberately leaves out:

- **Everything the page derives.** A gate's pass, borderline or fail; the options table and its ordering; blended
  and monthly costs; savings; lever effects; routing plans; arm tags and short names. One gate function applies the
  ``criteria`` to the figures; the input holds the figures and the thresholds, never their outcome.
- **Method prose with no field.** How a test set was sampled, and figures stated only in a method line (r1's
  observed spread of about 11 points; r2's interval of about one point at n = 1,200; r3's annotator agreement,
  κ 0.78 and 0.91, and its smallest cell's interval). The planner writes the method section from the fields.
- **r1:** the routing explorer's speculation-gate switch is a page control, not data; the task carrying it is
  marked ``cross_cutting``, which the page's ``routable: false`` and "occurs inside every other task" say.
- **r2:** each configuration's pass-rate interval: the page computes it from n with a binomial formula, and the
  reference carries no interval to copy.
- **r3:** the overall-accuracy weighting is an explore control; the input carries both weightings as measures and
  gates on the balanced one, the page's default.
- **r4:** ``cost_eff`` (each arm's cost weighted by the depth mix, which ``share`` and ``cost_multiplier`` on the
  depth strata already give) and the hybrid options the page models from them, which were never run. The control
  is Production; the human preference rows rate Production against A, the first challenger (``f64``), as the page
  says. No reading is given per aspect: no margin was declared, so the change on each aspect against Production is
  carried as a contrast, its verdict read off its 95% interval as the pipeline reads an uncorrected one.

No reference's page flags criteria set after the results, so none carries ``late_criteria``; r4's flag (no rule set
in advance) is the empty ``merit_axes`` of its contrast family.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from threetears.evals.analysis.report_input import ReportInput

__all__ = ["REFERENCES", "build_inputs"]

#: The eval report kit's reference data.
REFERENCES = Path(__file__).resolve().parents[5] / "docs" / "eval-report-kit" / "references"

_HERE = Path(__file__).resolve().parent

#: The reference r5 is copied from verbatim: it already is a report input.
_R5 = "r5-kairo-round2.json"


def _load(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((REFERENCES / name).read_text(encoding="utf-8"))
    return loaded


def _empty_caveats() -> dict[str, Any]:
    return {"late_criteria": []}


def _meta(campaign: str, behavior: str, kind: str, design_shape: str, run_date: str, source: str) -> dict[str, Any]:
    return {
        "campaign": campaign,
        "behavior": behavior,
        "kind": kind,
        "design_shape": design_shape,
        "run_date": run_date,
        "provenance": f"Synthetic example data from docs/eval-report-kit/references/{source}",
        "synthetic": True,
    }


def _r1() -> dict[str, Any]:
    """Model selection: four candidates, no control, five tasks each gated on quality, its own behaviors and p95."""
    src = _load("r1-model-selection.json")
    arms = [
        {
            "id": c["id"],
            "name": c["name"],
            "role": "challenger",
            "served_model": c["name"],
            "settings": {"model": c["name"]},
            "description": c["kind"],
        }
        for c in src["candidates"]
    ]
    judged: dict[str, Any] = {
        "overall": {
            "label": "Overall quality",
            "scale": "ordinal",
            "calibration": "calibrated",
            "calibration_ratings": 200,
            "agreement": 0.86,
            "level": "95%",
        }
    }
    criteria: list[dict[str, Any]] = [
        {
            "id": "overall",
            "kind": "gate",
            "measure": "overall",
            "direction": "higher_is_better",
            "threshold": src["criteria"]["overall"],
            "scope": "each_task",
            "label": "Overall quality floor",
        },
        {
            "id": "p95",
            "kind": "gate",
            "measure": "latency_s",
            "statistic": "p95",
            "direction": "lower_is_better",
            "threshold": src["criteria"]["p95"],
            "scope": "each_task",
            "label": "Latency budget (p95)",
        },
    ]
    tasks = []
    for t in src["tasks"]:
        for d in t["dims"]:
            judged[d["id"]] = {"label": d["name"], "scale": "pass_fail", "level": "95%"}
            criteria.append(
                {
                    "id": f"{t['id']}.{d['id']}",
                    "kind": "gate",
                    "measure": d["id"],
                    "direction": "higher_is_better",
                    "threshold": d["floor"],
                    "task": t["id"],
                    "label": d["name"],
                }
            )
        per_arm = {}
        for arm, r in t["results"].items():
            figures = {
                "overall": {"mean": r["overall"], "half_width": r["overall_ci"]},
                "cost_per_1k": {"mean": r["cost"]},
                "latency_s": {"p50": r["p50"], "p95": r["p95"]},
            }
            figures |= {dim: {"mean": v["v"], "half_width": v["ci"]} for dim, v in r["dims"].items()}
            per_arm[arm] = figures
        tasks.append(
            {
                "id": t["id"],
                "name": t["name"],
                "description": t["desc"],
                "n_cases": t["n"],
                "share": t["share"],
                "routable": t["routable"],
                "cross_cutting": not t["routable"],
                "per_arm": per_arm,
            }
        )
    cases = sum(t["n"] for t in src["tasks"])
    return {
        "schema": "report-input/1",
        "meta": _meta(
            "Concierge model selection",
            "Concierge chat assistant",
            "choice",
            "undesignated",
            "2026-10-21",
            "r1-model-selection.json",
        ),
        "arms": arms,
        "aspects": [],
        "criteria": criteria,
        "scoreboard": {a["id"]: {"cases": cases} for a in arms},
        "measures": {
            "cost_per_1k": {
                "label": "Cost per 1k conversations",
                "unit": "usd_per_1k",
                "higher_is_better": False,
                "means": "Measured tokens times list price, per 1,000 conversations.",
            },
            "latency_s": {
                "label": "Latency",
                "unit": "s",
                "higher_is_better": False,
                "means": "End to end from user message to final token, staging environment.",
            },
        },
        "judged": judged,
        "contrasts": [],
        "caveats": _empty_caveats(),
        "tasks": tasks,
    }


def _r2() -> dict[str, Any]:
    """Config sweep: a model × reasoning budget × retry grid, gated on pass rate and p95."""
    src = _load("r2-config-sweep.json")
    model_name = {m["id"]: m["name"] for m in src["models"]}
    budget_name = {b["id"]: b["name"] for b in src["budgets"]}
    retry_name = {0: "no retry", 1: "1 retry"}
    arms, configs = [], []
    for c in src["configs"]:
        name = f"{model_name[c['model']]} · reasoning {budget_name[c['budget']].lower()} · {retry_name[c['retry']]}"
        arms.append(
            {
                "id": c["id"],
                "name": name,
                "role": "challenger",
                "served_model": model_name[c["model"]],
                "settings": {"model": model_name[c["model"]], "reasoning": c["budget"], "retry": c["retry"]},
            }
        )
        configs.append(
            {"arm": c["id"], "levels": {"model": c["model"], "reasoning": c["budget"], "retry": str(c["retry"])}}
        )
    outcome_name = {
        "wrong_args": "Wrong arguments",
        "malformed": "Malformed call",
        "missing_call": "Missing call",
        "wrong_tool": "Wrong tool",
        "timeout": "Timeout",
    }
    outcome_bins = [
        {"id": k} | ({"name": outcome_name[k]} if k in outcome_name else {}) for k in src["configs"][0]["outcomes"]
    ]
    return {
        "schema": "report-input/1",
        "meta": _meta(
            "Query planner config sweep",
            "Research query planner",
            "choice",
            "factorial",
            "2026-10-23",
            "r2-config-sweep.json",
        )
        | {"volume_per_month": 4_000_000},
        "arms": arms,
        "aspects": [],
        "criteria": [
            {
                "id": "pass_rate",
                "kind": "gate",
                "measure": "pass_rate",
                "direction": "higher_is_better",
                "threshold": src["criteria"]["pass_rate"],
                "label": "Pass rate",
            },
            {
                "id": "p95",
                "kind": "gate",
                "measure": "latency_ms",
                "statistic": "p95",
                "direction": "lower_is_better",
                "threshold": src["criteria"]["p95"],
                "label": "Latency budget (p95)",
            },
        ],
        "scoreboard": {a["id"]: {"cases": src["n"]} for a in arms},
        "measures": {
            "pass_rate": {
                "label": "Pass rate",
                "unit": "pct",
                "higher_is_better": True,
                "grade": "pass_fail",
                "means": "The executed call sequence returns the gold result, or asks for clarification where the "
                "gold answer does. Extra calls still pass.",
                "per_arm": {c["id"]: {"mean": c["pass_rate"]} for c in src["configs"]},
            },
            "calls_per_task": {
                "label": "Calls per task",
                "unit": "calls",
                "higher_is_better": False,
                "minimum": src["optimal_calls"],
                "means": "Tool calls per task, including retries; the minimum is the gold sequence's average.",
                "per_arm": {c["id"]: {"mean": c["calls"]} for c in src["configs"]},
            },
            "cost_per_1k": {
                "label": "Cost per 1k calls",
                "unit": "usd_per_1k",
                "higher_is_better": False,
                "means": "Measured tokens times list price per 1,000 planner calls, including retries.",
                "per_arm": {c["id"]: {"mean": c["cost"]} for c in src["configs"]},
            },
            "latency_ms": {
                "label": "Latency",
                "unit": "ms",
                "higher_is_better": False,
                "per_arm": {c["id"]: {"p50": c["p50"], "p95": c["p95"]} for c in src["configs"]},
            },
            "outcome": {
                "label": "Outcome",
                "unit": "pct",
                "means": "Share of tasks by outcome: passes, then each failure class.",
                "bins": outcome_bins,
                "per_arm": {c["id"]: {"shares": list(c["outcomes"].values())} for c in src["configs"]},
            },
            "extra_calls": {
                "label": "Extra calls per task",
                "unit": "pct",
                "higher_is_better": False,
                "means": "Share of passing tasks by how many calls they used beyond the minimum.",
                "bins": [{"id": "0"}, {"id": "1"}, {"id": "2"}, {"id": "3+", "name": "3 or more"}],
                "per_arm": {c["id"]: {"shares": c["excess_dist"]} for c in src["configs"]},
            },
        },
        "judged": {},
        "contrasts": [],
        "examples": [
            {
                "case": e["id"],
                "stratum": e["type"],
                "arm": e["config"],
                "outcome": e["outcome"],
                "input": e["q"],
                "expected": e["expected"],
                "excerpt": e["got"],
                "reason": e["note"],
            }
            for e in src["examples"]
        ],
        "caveats": _empty_caveats(),
        "tasks": [
            {
                "id": t["id"],
                "name": t["name"],
                "share": t["share"],
                "per_arm": {c["id"]: {"pass_rate": {"mean": c["by_type"][t["id"]]}} for c in src["configs"]},
            }
            for t in src["task_types"]
        ],
        "factors": [
            {"id": "model", "name": "Model", "levels": src["models"]},
            {"id": "reasoning", "name": "Reasoning budget", "levels": src["budgets"]},
            {
                "id": "retry",
                "name": "Retry on failure",
                "levels": [{"id": str(k), "name": v} for k, v in retry_name.items()],
            },
        ],
        "configs": configs,
    }


def _r3() -> dict[str, Any]:
    """Classifier sweep: model × prompt, graded in a complexity × domain target matrix."""
    src = _load("r3-classifier.json")
    model_name = {m["id"]: m["name"] for m in src["models"]}
    prompt_name = {p["id"]: p["name"] for p in src["prompts"]}
    by_arm = {c["id"]: c for c in src["configs"]}
    cells = {}
    for key, count in src["counts"].items():
        row, column = key.split("|")
        cells[key] = {
            "row": row,
            "column": column,
            "n_cases": count,
            "per_arm": {
                arm: {"accuracy": {"mean": c["cells"][key]["acc"], "half_width": c["cells"][key]["ci"]}}
                for arm, c in by_arm.items()
            },
        }

    def per_arm(field: str) -> dict[str, Any]:
        return {arm: {"mean": c[field]} for arm, c in by_arm.items()}

    return {
        "schema": "report-input/1",
        "meta": _meta(
            "Turn router classifier sweep",
            "Turn router: complexity × domain",
            "choice",
            "factorial",
            "2026-10-24",
            "r3-classifier.json",
        )
        | {"volume_per_month": src["volume_per_month"]},
        "arms": [
            {
                "id": c["id"],
                "name": f"{model_name[c['model']]} · {prompt_name[c['prompt']]}",
                "role": "challenger",
                "served_model": model_name[c["model"]],
                "settings": {"model": model_name[c["model"]], "prompt": prompt_name[c["prompt"]]},
            }
            for c in src["configs"]
        ],
        "aspects": [],
        "criteria": [
            {
                "id": "overall",
                "kind": "gate",
                "measure": "balanced_accuracy",
                "direction": "higher_is_better",
                "threshold": src["criteria"]["overall"],
                "label": "Overall accuracy",
            },
            {
                "id": "cell_floor",
                "kind": "gate",
                "measure": "accuracy",
                "direction": "higher_is_better",
                "threshold": src["criteria"]["cell_floor"],
                "scope": "each_cell",
                "label": "Cell floor",
            },
            {
                "id": "p95",
                "kind": "gate",
                "measure": "latency_ms",
                "statistic": "p95",
                "direction": "lower_is_better",
                "threshold": src["criteria"]["p95"],
                "label": "Latency budget (p95)",
            },
        ],
        "scoreboard": {arm: {"cases": src["n"]} for arm in by_arm},
        "measures": {
            "accuracy": {
                "label": "Accuracy",
                "unit": "pct",
                "higher_is_better": True,
                "grade": "class_label",
                "level": "95%",
                "means": "A turn is correct only if both complexity and domain match. Read per cell.",
            },
            "balanced_accuracy": {
                "label": "Balanced accuracy",
                "unit": "pct",
                "higher_is_better": True,
                "grade": "class_label",
                "means": "Accuracy with the nine target cells weighted equally.",
                "per_arm": per_arm("balanced"),
            },
            "traffic_accuracy": {
                "label": "Traffic-weighted accuracy",
                "unit": "pct",
                "higher_is_better": True,
                "grade": "class_label",
                "means": "Accuracy weighted by corpus counts, which follow production traffic.",
                "per_arm": per_arm("traffic"),
            },
            "macro_f1": {
                "label": "Macro F1",
                "unit": "pct",
                "higher_is_better": True,
                "per_arm": per_arm("macro_f1"),
            },
            "cost_per_1m": {
                "label": "Cost per million turns",
                "unit": "usd_per_1m",
                "higher_is_better": False,
                "means": "Measured tokens times list price per million turns.",
                "per_arm": per_arm("cost"),
            },
            "latency_ms": {
                "label": "Latency",
                "unit": "ms",
                "higher_is_better": False,
                "per_arm": {arm: {"p95": c["p95"]} for arm, c in by_arm.items()},
            },
        },
        "judged": {},
        "contrasts": [],
        "caveats": _empty_caveats(),
        "factors": [
            {"id": "model", "name": "Model", "levels": src["models"]},
            {"id": "prompt", "name": "Prompt", "levels": src["prompts"]},
        ],
        "configs": [{"arm": c["id"], "levels": {"model": c["model"], "prompt": c["prompt"]}} for c in src["configs"]],
        "target_matrix": {
            "rows": {"id": "complexity", "name": "Complexity", "levels": src["complexity"]},
            "columns": {"id": "domain", "name": "Domain", "levels": src["domain"]},
        },
        "cells": cells,
        "confusion": {
            axis: {
                "labels": [level["id"] for level in src[axis]],
                "unit": "row_percent",
                "per_arm": {arm: c[f"conf_{axis}"] for arm, c in by_arm.items()},
            }
            for axis in ("complexity", "domain")
        },
    }


def _verdict(lo: float, hi: float, higher_is_better: bool) -> str:
    """An uncorrected contrast's verdict, read off its interval as the pipeline reads one: separated only where the
    interval excludes zero."""
    if lo > 0:
        return "improved" if higher_is_better else "regressed"
    if hi < 0:
        return "regressed" if higher_is_better else "improved"
    return "not_separated"


def _r4() -> dict[str, Any]:
    """Tradeoff: Production against three challengers on four aspects, with no rule ranking cost against character."""
    src = _load("r4-tradeoff.json")
    control = next(a for a in src["arms"] if a["role"] == "control")
    lever_key = {"model": "model", "out": "max_output_tokens", "reasoning": "reasoning"}
    arms = []
    for a in src["arms"]:
        spec = a["spec"]
        if a is control:
            settings = {lever_key[k]: v for k, v in spec.items()}
        else:
            settings = {lever_key[k]: v for k, v in spec.items() if k == "model" or v != control["spec"][k]}
        arms.append(
            {
                "id": a["id"],
                "name": "Production" if a is control else a["id"],
                "role": a["role"],
                "served_model": spec["model"],
                "settings": settings,
            }
        )
    aspect_kind = {a["id"]: a["kind"] for a in src["aspects"]}
    judged = {
        a["id"]: {
            "label": a["name"],
            "scale": "pass_fail",
            "per_arm": {x["id"]: {"mean": x["results"][a["id"]]} for x in src["arms"]},
        }
        for a in src["aspects"]
        if a["kind"] == "judged"
    }
    judged["persona"] |= {"calibration": "calibrated", "calibration_ratings": 240, "agreement": 0.82}
    measures: dict[str, Any] = {
        a["id"]: {
            "label": a["name"],
            "unit": "pct",
            "higher_is_better": True,
            "grade": "pass_fail",
            "means": a["desc"],
            "per_arm": {x["id"]: {"mean": x["results"][a["id"]]} for x in src["arms"]},
        }
        for a in src["aspects"]
        if a["kind"] == "checked"
    }
    measures |= {
        "cost_per_1k": {
            "label": "Cost per 1k replies",
            "unit": "usd_per_1k",
            "higher_is_better": False,
            "means": "Measured tokens times list price per 1,000 replies. Later turns cost more because the context "
            "is longer.",
            "per_arm": {a["id"]: {"mean": a["cost"]} for a in src["arms"]},
        },
        "latency_s": {
            "label": "Latency",
            "unit": "s",
            "higher_is_better": False,
            "per_arm": {a["id"]: {"p50": a["p50"], "p95": a["p95"]} for a in src["arms"]},
        },
    }
    contrasts = [
        {
            "arm": a["id"],
            "control": control["id"],
            "kind": "judged" if aspect_kind[aspect] == "judged" else "measure",
            "measure": aspect,
            "verdict": _verdict(d["lo"], d["hi"], higher_is_better=True),
            "delta": d["v"],
            "interval": [d["lo"], d["hi"]],
        }
        for a in src["arms"]
        if a is not control
        for aspect, d in a["delta"].items()
    ]
    depth = [
        {
            "id": f"depth:{d['id']}",
            "name": d["name"],
            "dimension": "depth",
            "share": d["share"],
            "cost_multiplier": d["cost_mult"],
            "per_arm": {
                a["id"]: {
                    "persona": {
                        "mean": a["persona_by_depth"][i]["v"],
                        "half_width": a["persona_by_depth"][i]["ci"],
                        "n": d["n"],
                    }
                }
                for a in src["arms"]
            },
        }
        for i, d in enumerate(src["depth"])
    ]
    scenarios = [
        {
            "id": f"scenario:{c['id']}",
            "name": c["name"],
            "dimension": "scenario",
            "per_arm": {
                a["id"]: {"persona": {"change": a["cat_delta"][i]}} for a in src["arms"] if a["cat_delta"] is not None
            },
        }
        for i, c in enumerate(src["categories"])
    ]
    pref = src["human_pref"]
    return {
        "schema": "report-input/1",
        "meta": _meta(
            "Persona model tradeoff",
            "Persona chat model",
            "change",
            "multi_factor",
            "2026-10-24",
            "r4-tradeoff.json",
        )
        | {"volume_per_month": src["volume_per_month"]},
        "arms": arms,
        "aspects": [
            {
                "id": a["id"],
                "name": a["name"],
                "asked": bool(a.get("primary")),
                "description": a["desc"],
                "graded_by": a["kind"],
                "readings": [],
            }
            for a in src["aspects"]
        ],
        "criteria": [
            {
                "id": a["id"],
                "kind": "guardrail",
                "measure": a["id"],
                "direction": "higher_is_better",
                "threshold": a["guard"],
                "standing": True,
                "label": a["name"],
            }
            for a in src["aspects"]
            if a["guard"] is not None
        ],
        "scoreboard": {a["id"]: {"cases": src["n_conversations"], "ran": src["n_replies"]} for a in src["arms"]},
        "measures": measures,
        "judged": judged,
        "primary_judged": next(a["id"] for a in src["aspects"] if a.get("primary")),
        "contrasts": contrasts,
        "contrast_families": [
            {
                "merit_axes": [],
                "correction": "none",
                "disclosure": "95% intervals on paired differences, not corrected across the three challengers.",
                "level": "95%",
            }
        ],
        "strata": depth + scenarios,
        "caveats": _empty_caveats(),
        "human_pref": {
            "control": control["id"],
            "arm": "f64",
            "n": pref["n"],
            "rows": [
                {"id": r["id"], "name": r["name"], "prefers_control": r["prod_pref"], "half_width": r["ci"]}
                for r in pref["rows"]
            ],
        },
    }


_BUILDERS = {"r1": _r1, "r2": _r2, "r3": _r3, "r4": _r4}


def _serialize(document: dict[str, Any]) -> str:
    """The document as the model reads it, written the one way every build writes it."""
    report = ReportInput.model_validate(document)
    dumped = report.model_dump(mode="json", by_alias=True, exclude_unset=True)
    return json.dumps(dumped, indent=1, ensure_ascii=False) + "\n"


def build_inputs(out_dir: Path = _HERE) -> list[Path]:
    """Write ``r1/input.json`` to ``r5/input.json`` under ``out_dir``.

    :param out_dir: Where the fixture directories go; this file's own directory by default.
    :ptype out_dir: Path
    :return: The files written, r1 first.
    :rtype: list[Path]
    """
    written = []
    for name, build in _BUILDERS.items():
        path = out_dir / name / "input.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_serialize(build()), encoding="utf-8")
        written.append(path)
    r5 = out_dir / "r5" / "input.json"
    r5.parent.mkdir(parents=True, exist_ok=True)
    r5.write_bytes((REFERENCES / _R5).read_bytes())
    written.append(r5)
    return written


if __name__ == "__main__":
    for written_path in build_inputs(Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE):
        print(written_path)
