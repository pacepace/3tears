"""Write the weakened anchor plans and their labels: each anchor copy broken on one plan-check rule or one rubric item.

Run from the repository root: ``uv run python packages/evals/tests/fixtures/report_plans/weaken_anchors.py``. It reads
the anchors (``rN/anchor.plan.json``), applies each edit in :data:`WEAKENINGS` to a copy, checks the copy is what its
entry says, and writes ``anchors/weak/<name>.plan.json`` and ``anchors/LABELS.json``. Running it again writes the same
bytes. It refuses to write anything when a copy is not what its entry says.

Two kinds of copy:

- **Refused** (``refused-<rule>``): breaks one rule of the plan checks (``validate_plan``), and the checks refuse it with
  exactly the faults the entry lists, each a field path and a fragment of why: nothing more, so the copy fails that
  rule and no other.
- **Weak on an item** (``item-<id>``): its prose is weakened on one model-scored rubric item, and it still passes the
  plan checks with the same structure as its anchor: only prose fields differ.

Every entry pins the sha256 of the plan it labels (``plan_digest``). An entry starts ``pending_owner``; the owner approves
an anchor (and its expected verdict) before it counts as a human label, recorded as ``labeled_by: owner`` with the
day (``labeled_at``). An owner label belongs to the text the owner saw, so a regeneration keeps it only while the
plan's bytes are unchanged (:func:`carried_label`); a plan that moved drops back to ``pending_owner``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from threetears.evals.analysis.errors import PlanRefusal
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_checks import validate_plan
from threetears.evals.kernel.report_plan import ReportPlan

__all__ = ["ANCHORS", "LABELS_SCHEMA", "WEAKENINGS", "Weakening", "build", "carried_label", "main"]

HERE = Path(__file__).resolve().parent
WEAK = HERE / "anchors" / "weak"
LABELS = HERE / "anchors" / "LABELS.json"
#: The labels file's schema id: each entry pins its plan's digest, and an owner label its day.
LABELS_SCHEMA = "plan-anchor-labels/2"

#: The anchors, by input, with what each note says about where it came from.
ANCHORS = {
    "r1": "Hand-written from references/r1-model-selection.html.",
    "r2": "Hand-written from references/r2-config-sweep.html.",
    "r3": "Hand-written from references/r3-classifier.html.",
    "r4": "Hand-written from references/r4-tradeoff.html.",
    "r5": (
        "references/r5-kairo-round2.plan.json: a two-sentence lede, no hedge above the fold, no quoted question, a "
        "Watch citing the weakest margin code found, and a Next naming the experiment that would change the answer "
        "and which way."
    ),
}

Plan = dict[str, Any]


@dataclass(frozen=True)
class Weakening:
    """One weakened copy: its name, the anchor it copies, what it breaks, and the edit."""

    name: str
    source: str
    edit: Callable[[Plan], None]
    note: str
    rule: str | None = None
    faults: tuple[tuple[str, str], ...] = ()
    item: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _slot(plan: Plan, key: str) -> Plan:
    slot: Plan = next(slot for slot in plan["decision"]["slots"] if slot["key"] == key)
    return slot


def _set(path: tuple[Any, ...], value: Any) -> Callable[[Plan], None]:
    def edit(plan: Plan) -> None:
        node = plan
        for step in path[:-1]:
            node = node[step]
        node[path[-1]] = value

    return edit


def _swap_watch_and_next(plan: Plan) -> None:
    slots = plan["decision"]["slots"]
    keys = [slot["key"] for slot in slots]
    watch, nxt = keys.index("watch"), keys.index("next")
    slots[watch], slots[nxt] = slots[nxt], slots[watch]


def _swap_first_sections(plan: Plan) -> None:
    sections = plan["sections"]
    sections[0], sections[1] = sections[1], sections[0]


def _no_call(plan: Plan) -> None:
    plan["mode"] = "no_call"
    plan["headline_kind"] = "no_call"
    plan["options"]["rows"][0]["role"] = "control"


def _unknown_recommendation(plan: Plan) -> None:
    plan["options"]["rows"][0]["role"] = "alternative"
    plan["options"]["rows"].insert(0, {"option": "nobody", "role": "recommended"})


def _unraised_answer(plan: Plan) -> None:
    plan["answers"] = [
        {
            "warning": {"kind": "behind_on", "aspect": "character adherence (v2)", "name": "qwen3.8-flash"},
            "text": "September stands: nothing shows a candidate ahead of it.",
        }
    ]


def _decide_flag(plan: Plan) -> None:
    plan["flag"] = [
        {
            "title": "No rule set in advance.",
            "text": "The campaign set no rule for trading character against cost.",
            "discloses": ["no_rule"],
            "weakens": [],
            "survives": [],
        }
    ]


def _drop_latency_landing(plan: Plan) -> None:
    for item in plan["method"]:
        item["caveats"] = [caveat for caveat in item["caveats"] if caveat != "latency"]


def _replace(path: tuple[Any, ...], old: str, new: str) -> Callable[[Plan], None]:
    def edit(plan: Plan) -> None:
        node = plan
        for step in path[:-1]:
            node = node[step]
        text = node[path[-1]]
        if old not in text:
            raise ValueError(f"{'.'.join(map(str, path))} no longer holds {old!r}")
        node[path[-1]] = text.replace(old, new)

    return edit


def _slot_text(key: str, text: str) -> Callable[[Plan], None]:
    def edit(plan: Plan) -> None:
        _slot(plan, key)["text"] = text

    return edit


def _section_topics(plan: Plan) -> None:
    for section, topic in zip(plan["sections"], ("The asked question", "Strata", "Examples"), strict=True):
        section["headline"] = topic


def _restated_clauses(plan: Plan) -> None:
    said = {
        "judge_uncalibrated": "The judge is uncalibrated.",
        "late:aspect": "The aspect was chosen after the results.",
        "late:control": "The control was chosen after the results.",
    }
    for clause in plan["decision"]["confidence"]["caveats"]:
        clause["material_if"] = said[clause["id"]]


def _no_first_screen(plan: Plan) -> None:
    plan["headline"] = "Keep {control.name}."
    plan["lede"] = "This page reports the second round of the persona campaign. The sections below hold the evidence."
    _slot(plan, "recommendation")["text"] = "Keep {control.name}."
    _slot(plan, "why_not")["text"] = "The sections below compare the candidates."
    _slot(plan, "watch")["text"] = "Method lists what to keep in mind, {weakest.arm} first."


def _unknown_column(plan: Plan) -> None:
    plan["options"]["columns"].append("monthly_cost")


def _quoted_question(plan: Plan) -> None:
    section = next(section for section in plan["sections"] if section["id"] == "question")
    section["takeaway"] = "The question: {meta.question} The shaded region is the margin."


WEAKENINGS: tuple[Weakening, ...] = (
    Weakening(
        "refused-facts-shape",
        "r5",
        _set(("shape",), "A"),
        "The shape is not the one code looks up from the signals.",
        rule="facts",
        faults=(("shape", "the signals look up shape 'D'"),),
    ),
    Weakening(
        "refused-spine",
        "r5",
        _swap_first_sections,
        "The strata section opens the evidence, ahead of the forest that is shape D's spine.",
        rule="spine",
        faults=(("sections[0].chart.kind", "shape D's spine, code's fact, is a 'forest'"),),
    ),
    Weakening(
        "refused-mode-headline-kind",
        "r5",
        _set(("headline_kind",), "trade"),
        "A decide plan with a trade headline.",
        rule="mode",
        faults=(("headline_kind", "a decide plan's headline is a 'decision'"),),
    ),
    Weakening(
        "refused-mode-slots",
        "r5",
        _swap_watch_and_next,
        "Watch and Next out of the mode's order.",
        rule="mode",
        faults=(("decision.slots", "in that order"),),
    ),
    Weakening(
        "refused-mode-roles",
        "r5",
        _set(("options", "rows", 1, "role"), "recommended"),
        "Two options marked recommended.",
        rule="mode",
        faults=(("options.rows", "recommends exactly one option"),),
    ),
    Weakening(
        "refused-allowed-mode",
        "r5",
        _no_call,
        "A no call where code allows only decide (one survivor dominates).",
        rule="allowed_mode",
        faults=(("mode", "code allows only ['decide']"),),
    ),
    Weakening(
        "refused-ids",
        "r5",
        _set(("sections", 1, "id"), "question"),
        "Two sections share an id.",
        rule="ids",
        faults=(("sections", "name the section id 'question' 2 times"),),
    ),
    Weakening(
        "refused-recommendation",
        "r5",
        _unknown_recommendation,
        "Recommends an option that is no arm.",
        rule="recommendation",
        faults=(("options.rows[0].option", "which is no arm"),),
    ),
    Weakening(
        "refused-answers",
        "r5",
        _unraised_answer,
        "Answers a warning code never raised.",
        rule="answers",
        faults=(("answers[0].warning", "which code did not raise"),),
    ),
    Weakening(
        "refused-confidence",
        "r5",
        _set(("decision", "confidence", "form"), "high"),
        "Plain high while report-wide caveats are open.",
        rule="confidence",
        faults=(("decision.confidence.form", "open caveats remain"),),
    ),
    Weakening(
        "refused-flag-no-rule-in-decide",
        "r5",
        _decide_flag,
        "A decide plan disclosing no rule set in advance.",
        rule="flag",
        faults=(("flag[0].discloses", "holds 'no_rule'"),),
    ),
    Weakening(
        "refused-flag-no-rule-missing",
        "r4",
        _set(("flag",), []),
        "A tradeoff plan with no rule declared and no flag saying so.",
        rule="flag",
        faults=(("flag", "must disclose 'no_rule'"),),
    ),
    Weakening(
        "refused-coverage",
        "r5",
        _drop_latency_landing,
        "The latency caveat lands nowhere.",
        rule="coverage",
        faults=(("method", "drop the caveat 'latency'"),),
    ),
    Weakening(
        "refused-charts-kind",
        "r5",
        _set(("summary", 0, "kind"), "lines"),
        "A lines chart over contrasts, which it cannot draw.",
        rule="charts",
        faults=(("summary[0].kind", "cannot draw 'contrasts'"),),
    ),
    Weakening(
        "refused-charts-summary-order",
        "r5",
        _set(("summary", 1, "query", "arms"), ["1017", "qwen3.8-flash", "September"]),
        "A summary chart's arms out of the options table's order.",
        rule="charts",
        faults=(("summary[1].query.arms", "in its order"),),
    ),
    Weakening(
        "refused-keys",
        "r5",
        _replace(("lede",), "{reading.qwen3.8-flash.high|abs|pts}", "{reading.qwen3.8-flash.drop|abs|pts}"),
        "A figure key naming a statistic no reading has.",
        rule="keys",
        faults=(("lede", "reading.qwen3.8-flash.drop"),),
    ),
    Weakening(
        "refused-literal-digits",
        "r5",
        _replace(("lede",), "{reading.qwen3.8-flash.high|abs|pts} to", "3.4 to"),
        "A typed number in prose (D3).",
        rule="literal_digits",
        faults=(("lede", "types the number(s) ['3.4']"),),
    ),
    Weakening(
        "refused-weakest",
        "r5",
        _slot_text("watch", "Keep monitoring the persona metrics in production after the decision."),
        "A decide plan's Watch citing no weakest-margin key.",
        rule="weakest",
        faults=(("decision.slots[2].text", "names no weakest margin"),),
    ),
    Weakening(
        "refused-reader-ids",
        "r5",
        _replace(
            ("decision", "slots", 1, "text"),
            "in-character {contrast.qwen3.8-flash",
            "persona.holds_character {contrast.qwen3.8-flash",
        ),
        "An internal id typed above Full results where its reader name differs.",
        rule="reader_ids",
        faults=(("decision.slots[1].text", "types the judged id 'persona.holds_character'"),),
    ),
    Weakening(
        "refused-reader-ids-question",
        "r5",
        _quoted_question,
        "The campaign's question quoted above Full results, in the campaign's own ids.",
        rule="reader_ids",
        faults=(("sections[0].takeaway", "quotes the campaign's question"),),
    ),
    Weakening(
        "refused-units",
        "r2",
        _replace(
            ("lede",),
            "{measure.latency_ms.brisk-s.low.r1.p95|int} ms, inside",
            "{measure.latency_ms.brisk-s.low.r1.p95} ms, inside",
        ),
        "A unit written right after a key that prints it (the key spells 513 ms, and the prose adds ms).",
        rule="units",
        faults=(("lede", "writes 'ms' right after"),),
    ),
    Weakening(
        "refused-units-sub-cent",
        "r5",
        _replace(
            ("decision", "slots", 0, "text"),
            "({cost_per_1k.September|usd_per_1k})",
            "({scoreboard.September.reply_cost_usd} a reply)",
        ),
        "A cost per reply under one cent, on a page that states its costs per 1,000 replies.",
        rule="units",
        faults=(("decision.slots[0].text", "under one cent"),),
    ),
    Weakening(
        "refused-columns",
        "r5",
        _unknown_column,
        "An options column outside the column families code renders.",
        rule="columns",
        faults=(("options.columns[5]", "is no column family code renders"),),
    ),
    Weakening(
        "refused-small-multiples",
        "r5",
        _set(("summary", 0, "query", "small_multiples_by"), "arm"),
        "Small multiples asked of a forest, which the kit does not panel.",
        rule="small_multiples",
        faults=(("summary[0].query.small_multiples_by", "draws no small multiples of a 'forest'"),),
    ),
    Weakening(
        "item-R2",
        "r5",
        _set(("headline",), "Results of the second round of the Kairo persona comparison."),
        "The headline names a topic, not the call.",
        item="R2",
    ),
    Weakening(
        "item-R3",
        "r5",
        _set(
            ("lede",),
            "This campaign compared three persona models on character adherence. It ran under production routing. "
            "The details follow in the sections below.",
        ),
        "The lede is background in three sentences, with no supporting fact and nothing on what would change the answer.",
        item="R3",
    ),
    Weakening(
        "item-R4",
        "r5",
        _replace(
            ("lede",),
            "neither candidate beats {control.name} on anything this campaign measured",
            "the judge is uncalibrated, and the aspect and control were chosen after the results",
        ),
        "The lede repeats caveats the confidence already carries.",
        item="R4",
    ),
    Weakening("item-R5", "r5", _section_topics, "Every section headline is a topic label.", item="R5"),
    Weakening(
        "item-R6",
        "r5",
        _slot_text(
            "why_not",
            "**{arm:qwen3.8-flash}:** {cost.uplift.qwen3.8-flash|pct} more per reply, holds_character "
            "{contrast.qwen3.8-flash.persona.holds_character.delta|pts} points.\n"
            "**{arm:1017}:** {cost.uplift.1017|pct} more per reply, holds_character "
            "{contrast.1017.persona.holds_character.delta|pts} points.",
        ),
        "Why not the others names the judged measure by a fragment of its key, which no code check matches.",
        item="R6",
    ),
    Weakening(
        "item-R7",
        "r5",
        _slot_text("watch", "Keep monitoring {weakest.arm} and the persona metrics in production after the decision."),
        "Watch cites the weakest margin's arm but is generic monitoring advice and names no margin.",
        item="R7",
    ),
    Weakening(
        "item-R8",
        "r5",
        _set(
            ("lede",),
            "It could perhaps be argued that, when the results that were obtained are considered as a whole, it is worth "
            "noting that {arm:qwen3.8-flash} appears to have been found to be somewhat worse on the asked pass rate, "
            "while {arm:1017} was arguably not clearly separated from {control.name} by the analysis that was run.",
        ),
        "The lede is long, passive and hedged.",
        item="R8",
    ),
    Weakening(
        "item-R9",
        "r5",
        _no_first_screen,
        "The first screen says to keep the control but not why or what the main risk is.",
        item="R9",
    ),
    Weakening(
        "item-R10",
        "r5",
        _slot_text("next", "Keep collecting data and check the dashboards regularly."),
        "Next names no experiment that could change the answer.",
        item="R10",
    ),
    Weakening(
        "item-R11",
        "r5",
        _restated_clauses,
        "Each confidence clause restates its caveat without saying what would make it material.",
        item="R11",
    ),
    Weakening(
        "item-R12",
        "r5",
        _replace(
            ("decision", "slots", 1, "text"),
            "{cost.uplift.qwen3.8-flash|pct} more per reply",
            "{cost_per_1k.qwen3.8-flash|usd_per_1k} more than {control.name}",
        ),
        "Why not gives a candidate's own cost as what it costs more than the control.",
        item="R12",
    ),
)


def _input(name: str) -> ReportInput:
    return ReportInput.model_validate_json((HERE / name / "input.json").read_text(encoding="utf-8"))


def _faults(plan: Plan, report_input: ReportInput) -> tuple[tuple[str, str], ...]:
    try:
        validate_plan(ReportPlan.model_validate(plan), report_input)
    except PlanRefusal as refused:
        return refused.faults
    return ()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def carried_label(previous: Sequence[Mapping[str, Any]], plan: str, digest: str) -> dict[str, Any]:
    """Who labels ``plan`` now: the label it had, while its plan is the text that label was given on, else none yet.

    :param previous: The entries the labels file held before this regeneration.
    :param plan: The plan's path under this directory.
    :param digest: The sha256 of the plan's bytes as written now.
    :return: ``labeled_by`` (``owner`` or ``pending_owner``), with ``labeled_at`` for an owner label.
    """
    entry = next((entry for entry in previous if entry.get("plan") == plan), None)
    if entry is not None and entry.get("labeled_by") == "owner" and entry.get("plan_digest") == digest:
        return {"labeled_by": "owner", "labeled_at": entry["labeled_at"]}
    return {"labeled_by": "pending_owner"}


def _previous_labels() -> list[dict[str, Any]]:
    if not LABELS.is_file():
        return []
    entries: list[dict[str, Any]] = json.loads(LABELS.read_text(encoding="utf-8"))["anchors"]
    return entries


def build() -> tuple[dict[str, str], dict[str, Any]]:
    """Every weakened copy as the text written, by path under this directory, and the labels file.

    :raises ValueError: A copy is not what its entry says.
    """
    files: dict[str, str] = {}
    previous = _previous_labels()
    entries: list[dict[str, Any]] = []
    for name, note in ANCHORS.items():
        plan = f"{name}/anchor.plan.json"
        digest = _digest((HERE / plan).read_text(encoding="utf-8"))
        entries.append(
            {
                "plan": plan,
                "input": name,
                "expect": "pass",
                "plan_digest": digest,
                **carried_label(previous, plan, digest),
                "note": note,
            }
        )
    for weakening in WEAKENINGS:
        anchor: Plan = json.loads((HERE / weakening.source / "anchor.plan.json").read_text(encoding="utf-8"))
        plan = deepcopy(anchor)
        weakening.edit(plan)
        faults = _faults(plan, _input(weakening.source))
        entry: dict[str, Any] = {
            "plan": f"anchors/weak/{weakening.name}.plan.json",
            "input": weakening.source,
            "from": f"{weakening.source}/anchor.plan.json",
        }
        if weakening.rule is not None:
            wanted = weakening.faults
            matched = len(faults) == len(wanted) and all(
                path == want_path and fragment in why for (path, why), (want_path, fragment) in zip(faults, wanted)
            )
            if not matched:
                raise ValueError(f"{weakening.name}: expected faults {list(wanted)}, the checks found {list(faults)}")
            entry |= {"expect": "refused", "rule": weakening.rule, "faults": [list(fault) for fault in wanted]}
        else:
            if faults:
                raise ValueError(f"{weakening.name} weakens prose only, and the checks refuse it: {list(faults)}")
            entry |= {"expect": "fails_item", "item": weakening.item}
        text = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
        digest = _digest(text)
        entry |= {"plan_digest": digest, **carried_label(previous, entry["plan"], digest), "note": weakening.note}
        entries.append(entry)
        files[entry["plan"]] = text
    return files, {"schema": LABELS_SCHEMA, "anchors": entries}


def main() -> None:
    """Write every weakened copy and the labels file."""
    files, labels = build()
    WEAK.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        (HERE / path).write_text(text, encoding="utf-8")
    LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} weakened copies and {LABELS.relative_to(HERE)}.")


if __name__ == "__main__":
    main()
