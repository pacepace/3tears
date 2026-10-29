"""The editor's facts (``analysis/report_doc_facts.py``): each fires on a planted case, and on the converted documents
only where it holds.

Each fact is planted on a converted prototype document and its snapshot, changed as little as the fact needs. On the
eleven later converted documents (prototypes 2 and 3) each fact is pinned: most are silent, and where one fires the test
says why it is real. No fact raises on a misfit input, even beside a document written for another snapshot, and the
facts are byte-identical when only ``story`` changes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans.misfits import MISFITS
from packages.evals.tests.report_doc_support import LATER_DOCS, converted_doc, snapshot_of
from threetears.evals.analysis.report_doc_facts import EditorFact, editor_facts
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc

_KAIRO = "prototype3/kairo"
_R1 = "prototype3/r1"
_R2 = "prototype3/r2"
_R4 = "prototype3/r4"
_S1 = "prototype2/s1"
_EACH_TASK = "prototype2/r2-with-its-gates-read-in-each-task"


def _input(name: str, change: Callable[[dict[str, Any]], None] | None = None) -> ReportInput:
    """The snapshot's report input, changed as a plant needs it."""
    held = ReportInput.model_validate(snapshot_of(name).report_input)
    if change is None:
        return held
    data = held.model_dump(mode="json", by_alias=True, exclude_unset=True)
    change(data)
    return ReportInput.model_validate(data)


def _facts(
    name: str,
    *,
    recommends: list[str] | None = None,
    doc_change: Callable[[dict[str, Any]], None] | None = None,
    input_change: Callable[[dict[str, Any]], None] | None = None,
) -> list[EditorFact]:
    """The facts on one converted document and its snapshot, each changed as a plant needs."""
    doc = converted_doc(name)
    if recommends is not None:
        doc["recommends"] = recommends
    if doc_change is not None:
        doc_change(doc)
    return editor_facts(ReportDoc.model_validate(doc), _input(name, input_change))


def _of(facts: list[EditorFact], kind: str) -> list[EditorFact]:
    return [fact for fact in facts if fact.kind == kind]


# --- each fact, planted --------------------------------------------------------------------------------------------


def test_a_recommended_arm_missing_a_gate_is_a_fact_per_figure_missed() -> None:
    """r1 recommending Atlas-L, which misses the p95 latency gate in four tasks: one fact per task, each citing the
    figure and the threshold."""
    facts = _of(_facts(_R1, recommends=["atlas-l"]), "gate_failed")

    assert [fact.path for fact in facts] == ["recommends[0]"] * 4
    assert [fact.figures[1] for fact in facts] == [
        f"criteria.p95.atlas-l.{task}.value" for task in ("billing", "trouble", "research", "specul")
    ]
    assert all(fact.figures[2] == "criteria.p95.threshold" for fact in facts)
    assert "misses the declared gate Latency budget (p95) in Billing & account Q&A: 3.6 s against 3 s." in facts[0].fact


def test_the_arm_the_document_recommends_meets_every_gate_and_no_gate_fact_fires() -> None:
    assert not _of(_facts(_R1), "gate_failed")


def test_a_warning_on_the_recommended_arm_no_key_names_is_a_fact() -> None:
    """s1 recommending v3, on which code raises two unclear floors the document cites no figure of."""
    facts = _of(_facts(_S1, recommends=["v3"]), "warning_unanswered")

    assert [fact.path for fact in facts] == ["recommends[0]", "recommends[0]"]
    assert "floor_unclear" in facts[0].fact and "billing/rate_drop/*" in facts[0].fact
    assert "tone/rate_drop/reply.cites_policy" in facts[1].fact


def test_a_warning_whose_figure_the_document_cites_is_no_fact() -> None:
    """A key naming the floor's read on the arm, anywhere in the prose, names the warning."""

    def cite_the_billing_floor(doc: dict[str, Any]) -> None:
        doc["claims"][0]["text"] += " {aspects.billing.floor.billing/rate_drop/*.v3.held_above}"

    facts = _of(_facts(_S1, recommends=["v3"], doc_change=cite_the_billing_floor), "warning_unanswered")

    assert ["tone/rate_drop/reply.cites_policy" in fact.fact for fact in facts] == [True]


def test_a_call_other_than_the_one_dominant_option_is_a_fact() -> None:
    """Kairo recommending nothing, where September dominates."""
    facts = _of(_facts(_KAIRO, recommends=[]), "call_vs_dominance")

    assert [(fact.path, fact.figures) for fact in facts] == [("recommends", ["arm:September"])]
    assert facts[0].fact.endswith("as the one option dominating; the document recommends no arm.")


def test_a_call_where_code_reads_a_tradeoff_is_a_fact() -> None:
    """r4 recommending f16, where code's rule allows only a tradeoff."""
    facts = _of(_facts(_R4, recommends=["f16"]), "call_vs_dominance")

    assert [fact.figures for fact in facts] == [["arm:f16"]]
    assert "reads code's call as a tradeoff, with no option dominating" in facts[0].fact


def test_a_tradeoff_recommending_nothing_matches_code_and_no_call_fact_fires() -> None:
    assert not _of(_facts(_R4), "call_vs_dominance")


def _watch(text: str | None) -> Callable[[dict[str, Any]], None]:
    """Each Watch entry's text replaced, or every Watch entry dropped for None."""

    def change(doc: dict[str, Any]) -> None:
        if text is None:
            doc["decision"] = [entry for entry in doc["decision"] if entry["slot"] != "watch"]
            return
        for entry in doc["decision"]:
            if entry["slot"] == "watch":
                entry["text"] = text

    return change


def test_a_watch_citing_the_weakest_margins_own_figure_is_no_fact() -> None:
    """r2's Watch cites the pass-rate gate's figure on the recommended arm: the weakest margin's own figure."""
    assert not _of(_facts(_R2), "weakest_not_in_watch")


def test_a_watch_citing_no_key_naming_the_weakest_margin_is_a_fact() -> None:
    facts = _of(_facts(_R2, doc_change=_watch("Watch the pass rate.")), "weakest_not_in_watch")

    assert [fact.path for fact in facts] == ["decision[2]"]
    assert facts[0].figures == [
        "weakest.arm",
        "weakest.value",
        "weakest.criterion",
        "weakest.threshold",
        "weakest.margin",
    ]
    assert facts[0].fact.endswith(", on an arm the document recommends; no Watch entry cites a key naming it.")


def test_the_weakest_margin_on_an_arm_the_call_does_not_pick_says_so() -> None:
    """Kairo's weakest margin is qwen3.8-flash's, and the document recommends September."""
    facts = _of(_facts(_KAIRO), "weakest_not_in_watch")

    assert facts[0].fact.endswith(", on no arm the document recommends; no Watch entry cites a key naming it.")


def test_a_decision_block_with_no_watch_entry_is_a_fact_about_the_block() -> None:
    facts = _of(_facts(_R2, doc_change=_watch(None)), "weakest_not_in_watch")

    assert [fact.path for fact in facts] == ["decision"]
    assert facts[0].fact.endswith("the decision block has no Watch entry.")


def test_a_weakest_key_in_watch_names_it() -> None:
    assert not _of(_facts(_KAIRO, doc_change=_watch("It sits at {weakest.margin}.")), "weakest_not_in_watch")


def test_a_candidate_neither_shown_better_nor_shown_no_worse_is_an_adoption_fact() -> None:
    """r4 recommending f64: the asked aspect declares no floor and nothing shows f64 better on it."""
    facts = _of(_facts(_R4, recommends=["f64"]), "adoption")

    assert [(fact.path, fact.figures) for fact in facts] == [("recommends[0]", ["arm:f64", "arm:prod"])]
    assert "aspect 'persona' declares no floor" in facts[0].fact


def _unasked(*aspects: str) -> Callable[[dict[str, Any]], None]:
    def change(data: dict[str, Any]) -> None:
        for aspect in data["aspects"]:
            if aspect["id"] in aspects:
                aspect["asked"] = False

    return change


def test_a_candidate_on_a_snapshot_asking_nothing_is_an_adoption_fact() -> None:
    facts = _of(_facts(_R4, recommends=["f64"], input_change=_unasked("persona")), "adoption")

    assert ["asks no question of any aspect" in fact.fact for fact in facts] == [True]


def test_a_candidate_past_a_breached_guardrail_is_an_adoption_fact() -> None:
    def breach(data: dict[str, Any]) -> None:
        data["guardrails"] = [{"arm": "f64", "dimension": "tone", "decision": "breached"}]

    facts = _of(_facts(_R4, recommends=["f64"], input_change=breach), "adoption")

    assert ["breached the guardrail on tone" in fact.fact for fact in facts] == [True]


def test_recommending_the_control_is_no_adoption() -> None:
    assert not _of(_facts(_S1), "adoption")


def test_a_settled_loss_past_the_margin_on_an_unasked_aspect_is_a_fact() -> None:
    """s1 recommending v2 with billing unasked: v2's billing interval lies wholly past minus the margin."""
    facts = _of(_facts(_S1, recommends=["v2"], input_change=_unasked("billing")), "settled_loss_unasked")

    assert [(fact.path, fact.figures) for fact in facts] == [
        ("recommends[0]", ["arm:v2", "aspects.billing.reading.v2.interval", "aspects.billing.reading.v2.margin"])
    ]
    assert "-83.2 to -23.9 points, lies wholly at or below minus the margin of 10 points" in facts[0].fact


def test_a_settled_loss_on_an_asked_aspect_is_no_unasked_fact() -> None:
    """The same loss, billing asked: the decision reads it, and the floor warning carries it."""
    assert not _of(_facts(_S1, recommends=["v2"]), "settled_loss_unasked")


def test_an_unsettled_reading_on_an_unasked_aspect_is_no_settled_loss() -> None:
    """v4's tone interval reaches above minus the margin: #980's open caveat, not a settled loss."""
    assert not _of(_facts(_S1, recommends=["v4"], input_change=_unasked("tone")), "settled_loss_unasked")


def _uncite(*arms: str) -> Callable[[dict[str, Any]], None]:
    """Every figure key holding one of the names (an arm, or any part of a key), in any string of the document,
    replaced by a word; ids and queries are strings with no braces, so only keys change."""
    named = re.compile(r"\{[^{}]*(?:" + "|".join(re.escape(arm) for arm in arms) + r")[^{}]*\}")

    def uncited(node: Any) -> Any:
        if isinstance(node, str):
            return named.sub("it", node)
        if isinstance(node, list):
            return [uncited(item) for item in node]
        if isinstance(node, dict):
            return {key: uncited(value) for key, value in node.items()}
        return node

    def change(doc: dict[str, Any]) -> None:
        doc.update(uncited(doc))

    return change


_UNDECIDED = ("atlas-m.high.r1", "brisk-s.high.r1")


def test_an_undecided_options_implied_read_no_key_names_is_a_fact() -> None:
    """r2 with its gates read in each task, its keys on the two options taken out: they have no p95 figure in some
    task, and their p95 over every case misses the line."""
    facts = _of(_facts(_EACH_TASK, doc_change=_uncite(*_UNDECIDED)), "undecided_implied")

    assert [(fact.path, fact.figures) for fact in facts] == [
        ("", [f"arm:{arm}", f"measure.latency_ms.{arm}.p95", "criteria.p95.threshold"]) for arm in _UNDECIDED
    ]
    assert "(inferred, not stated). No figure key in the document names it." in facts[0].fact


def test_a_key_naming_the_option_but_not_the_read_does_not_answer_it() -> None:
    """Only the latency keys on the two options taken out, their monthly costs still cited: a key answers the read
    only where it names the option and the criterion or its measure, so both facts still fire."""
    latency = [f"measure.latency_ms.{arm}." for arm in _UNDECIDED]
    doc = converted_doc(_EACH_TASK)
    _uncite(*latency)(doc)
    written = json.dumps(doc)
    facts = _of(_facts(_EACH_TASK, doc_change=_uncite(*latency)), "undecided_implied")

    assert all(f"{{cost_per_month.{arm}|" in written for arm in _UNDECIDED)
    assert not any(f"latency_ms.{arm}" in written for arm in _UNDECIDED)
    assert [(fact.path, fact.figures) for fact in facts] == [
        ("", [f"arm:{arm}", f"measure.latency_ms.{arm}.p95", "criteria.p95.threshold"]) for arm in _UNDECIDED
    ]


def test_an_undecided_read_the_document_cites_is_no_fact() -> None:
    """The document as written cites each option's p95 over every case, the figure the inference rests on."""
    assert not _of(_facts(_EACH_TASK), "undecided_implied")


def test_an_undecided_option_the_document_recommends_is_named_at_its_place() -> None:
    facts = _of(
        _facts(_EACH_TASK, recommends=["brisk-s.high.r1"], doc_change=_uncite(*_UNDECIDED)), "undecided_implied"
    )

    assert [fact.path for fact in facts] == ["", "recommends[0]"]


def test_a_count_written_in_words_with_no_key_is_a_fact() -> None:
    """``four of five`` and a key of a word are counts; ``one arm``, ``the only one of`` and two keys are not."""

    def write(doc: dict[str, Any]) -> None:
        doc["claims"][0]["text"] = (
            "It holds in four of five tasks, in {n_candidates|words} of six, in Two of the three, and on one arm; it is "
            "the only one of {n_candidates|words}, and {n_candidates|words} of {n_candidates|words}."
        )

    facts = _of(_facts(_KAIRO, doc_change=write), "count_in_words")

    assert [(fact.path, fact.figures) for fact in facts] == [("claims[0].text", [])] * 3
    assert [
        '"four of five"' in facts[0].fact,
        '"{…} of six"' in facts[1].fact,
        '"Two of the three"' in facts[2].fact,
    ] == [True] * 3


_STORED_KAIRO = "prototype4/kairo"

#: The reasoning-ratio confound, which Kairo's snapshot attaches to 1017's contrasts.
_REASONING = "confound:observed:reasoning_ratio"

#: 1017's contrasts, by key (each arm and measure is held once, so unqualified, as contrasts_list cites them).
_ON_1017 = "contrast.1017"


def test_a_caveat_on_a_contrast_a_claim_reads_that_nothing_carries_is_a_fact() -> None:
    """The stored Kairo document: the cost claim's bars read every arm's production cost, and 1017's cost contrast
    carries the reasoning-ratio confound, which neither the claim nor the confidence carries; the off-lane claim reads
    1017's holds-character rate, whose contrast carries it too. The judged claim carries it, so it is no fact there."""
    facts = _of(_facts(_STORED_KAIRO), "caveat_not_carried")

    assert [(fact.path, fact.figures) for fact in facts] == [
        (
            "claims[0].caveats",
            [
                "arm:1017",
                f"{_ON_1017}.production_replicating_cost.delta",
                f"{_ON_1017}.production_replicating_cost.interval",
            ],
        ),
        (
            "claims[3].caveats",
            ["arm:1017", f"{_ON_1017}.persona.holds_character.delta", f"{_ON_1017}.persona.holds_character.interval"],
        ),
    ]
    assert facts[0].fact == (
        "The evidence of claim cost reads 1017 (B)'s contrast on Production cost, +$0.49/1000 (+$0.17/1000 to "
        f"+$0.81/1000), and code attaches caveat `{_REASONING}` to that contrast; neither the claim nor the confidence "
        "carries it."
    )


def _carried(*where: str) -> Callable[[dict[str, Any]], None]:
    def change(doc: dict[str, Any]) -> None:
        for claim in doc["claims"]:
            if claim["id"] in where:
                claim["caveats"].append(_REASONING)

    return change


def test_a_claim_carrying_the_caveat_is_no_fact() -> None:
    """The same document with the confound landed beside the cost claim: only the off-lane claim's fact is left."""
    facts = _of(_facts(_STORED_KAIRO, doc_change=_carried("cost")), "caveat_not_carried")

    assert [fact.path for fact in facts] == ["claims[3].caveats"]
    assert not _of(_facts(_STORED_KAIRO, doc_change=_carried("cost", "offlane")), "caveat_not_carried")


def test_a_caveat_the_confidence_carries_is_no_fact() -> None:
    def confided(doc: dict[str, Any]) -> None:
        doc["confidence"]["caveats"].append({"id": _REASONING, "material_if": "Only if 1017's budget moved its cost."})

    assert not _of(_facts(_STORED_KAIRO, doc_change=confided), "caveat_not_carried")


def _apparatus_on_cost(data: dict[str, Any]) -> None:
    """The simulator's settings, an apparatus confound of method reach on Kairo, named on 1017's cost contrast too."""
    for contrast in data["contrasts"]:
        if (contrast["arm"], contrast["measure"]) == ("1017", "production_replicating_cost"):
            contrast["confounds"].append("simulator_sent_settings")


def test_a_method_reach_caveat_landed_on_a_method_term_is_carried() -> None:
    """Kairo lands apparatus:simulator_sent_settings, a method-reach caveat, on its Method term: planted on 1017's cost
    contrast, it is carried there and is no fact; taken off the term, it is one, on the cost claim."""

    def unlanded(doc: dict[str, Any]) -> None:
        for item in doc["method"]:
            item["caveats"] = [caveat for caveat in item["caveats"] if not caveat.startswith("apparatus:")]

    def apparatus(facts: list[EditorFact]) -> list[str]:
        return [fact.path for fact in _of(facts, "caveat_not_carried") if "`apparatus:" in fact.fact]

    assert apparatus(_facts(_STORED_KAIRO, input_change=_apparatus_on_cost)) == []
    assert apparatus(_facts(_STORED_KAIRO, doc_change=unlanded, input_change=_apparatus_on_cost)) == [
        "claims[0].caveats"
    ]


def test_a_key_naming_a_contrast_is_evidence_that_reads_it() -> None:
    """A claim whose block reads no contrast but whose text cites 1017's cost contrast by key reads it."""

    def cited(doc: dict[str, Any]) -> None:
        examples = next(claim for claim in doc["claims"] if claim["id"] == "examples")
        examples["text"] += f" It costs {{{_ON_1017}.production_replicating_cost.delta}} more."

    facts = _of(_facts(_STORED_KAIRO, doc_change=cited), "caveat_not_carried")

    assert [fact.path for fact in facts] == ["claims[0].caveats", "claims[3].caveats", "claims[4].caveats"]


def test_a_contrast_the_input_leaves_untested_is_named_without_its_figures_and_never_refuses() -> None:
    """1017's cost contrast untested, its change and interval unstated: the fact still names the contrast and its
    caveat, cites no figure of it, and nothing raises."""

    def untested(data: dict[str, Any]) -> None:
        for contrast in data["contrasts"]:
            if (contrast["arm"], contrast["measure"]) == ("1017", "production_replicating_cost"):
                contrast.update(verdict="untested", delta=None, interval=None)

    facts = _of(_facts(_STORED_KAIRO, input_change=untested), "caveat_not_carried")

    assert [(fact.path, fact.figures) for fact in facts][0] == ("claims[0].caveats", ["arm:1017"])
    assert facts[0].fact.startswith("The evidence of claim cost reads 1017 (B)'s contrast on Production cost, and code")


# --- a high confidence over open caveats ----------------------------------------------------------------------------


def test_a_high_over_open_caveats_is_one_fact_naming_how_many_and_which() -> None:
    doc = converted_doc(_KAIRO)
    left = [caveat["id"] for caveat in doc["confidence"]["caveats"] if caveat.get("material_if") is not None]
    assert doc["confidence"]["form"] == "high_if_immaterial" and left, "the control: Kairo reads high over open caveats"

    (fact,) = _of(_facts(_KAIRO), "high_over_open")

    assert fact.path == "confidence" and fact.figures == []
    count = "1 open caveat" if len(left) == 1 else f"{len(left)} open caveats"
    assert fact.fact == (
        f"The confidence reads high, if the caveats aren't material, over {count}, none of which code proves "
        f"immaterial: {', '.join(f'`{caveat}`' for caveat in left)}."
    )


@pytest.mark.parametrize("form", ["low", "high"])
def test_a_confidence_that_does_not_read_high_over_open_caveats_is_no_fact(form: str) -> None:
    def formed(doc: dict[str, Any]) -> None:
        doc["confidence"]["form"] = form
        if form == "high":
            doc["confidence"]["caveats"] = []

    assert not _of(_facts(_KAIRO, doc_change=formed), "high_over_open")


# --- the converted documents ---------------------------------------------------------------------------------------

#: The facts each later converted document triggers, by kind. Every one that fires is real:
#:
#: - r2 with its gates read in each task fires nothing: two options are undecided on the p95 gate read in each task,
#:   their p95 over every case missing it, and the document cites that very figure for each, which answers it.
#: - s1: the document keeps production where code reads a tradeoff (no option dominates), and the weakest margin, v2's
#:   tone floor, is one Watch does not name (it discusses v2's billing floor). Both true; whether the call and Watch
#:   are argued well is the editor's.
#: - s2: the document recommends beta, at low confidence, where code reads a tradeoff. True.
#: - Kairo: the weakest margin is qwen3.8-flash's reading against the margin, a candidate the call does not pick, and
#:   Watch does not name it. True; the editor weighs whether it matters to keeping September.
#: - Every document but s2 reads high, if the caveats aren't material, over caveats it leaves open (``high_over_open``,
#:   added 2026-10-29 after a stored page printed that line above four uncleared caveats). True of each: s2 is low,
#:   and every other one's form is high_if_immaterial; whether each high stands is the editor's.
_EXPECTED: dict[str, list[str]] = {
    "prototype2/r2-with-its-gates-read-in-each-task": ["high_over_open"],
    "prototype2/r2-with-one-gate": ["high_over_open"],
    "prototype2/r3-without-its-configs": ["high_over_open"],
    "prototype2/s1": ["call_vs_dominance", "weakest_not_in_watch", "high_over_open"],
    "prototype2/s2": ["call_vs_dominance"],
    "prototype2/s3": ["high_over_open"],
    "prototype3/kairo": ["weakest_not_in_watch", "high_over_open"],
    "prototype3/r1": ["high_over_open"],
    "prototype3/r2": ["high_over_open"],
    "prototype3/r3": ["high_over_open"],
    "prototype3/r4": ["high_over_open"],
}


def test_every_later_document_is_pinned() -> None:
    assert sorted(_EXPECTED) == sorted(LATER_DOCS)


@pytest.mark.parametrize("name", LATER_DOCS)
def test_each_later_document_triggers_only_the_facts_that_hold(name: str) -> None:
    assert [fact.kind for fact in _facts(name)] == _EXPECTED[name]


# --- never a refusal, never the story ------------------------------------------------------------------------------


@pytest.mark.parametrize("misfit", list(MISFITS))
def test_no_fact_raises_on_a_misfit_input(misfit: str) -> None:
    """Every later document against the misfit input, each as written and recommending every arm the input holds at
    once (validity would refuse either; the facts read what they can)."""
    report_input = ReportInput.model_validate(MISFITS[misfit]())
    every_arm = [arm.id for arm in report_input.arms]
    for name in LATER_DOCS:
        for recommends in (None, every_arm):
            doc = converted_doc(name)
            if recommends is not None:
                doc["recommends"] = recommends
            facts = editor_facts(ReportDoc.model_validate(doc), report_input)
            assert all(isinstance(fact, EditorFact) and fact.fact for fact in facts)


def test_an_arm_the_snapshot_does_not_hold_is_written_as_the_document_wrote_it() -> None:
    facts = _facts(_KAIRO, recommends=["ghost"])

    assert [fact.kind for fact in facts] == ["call_vs_dominance", "weakest_not_in_watch", "high_over_open"]
    assert facts[0].fact.endswith("the document recommends `ghost`.")


@pytest.mark.parametrize(
    ("name", "recommends"),
    [(_EACH_TASK, ["brisk-s.high.r1"]), (_S1, ["v3"]), (_KAIRO, None), (_STORED_KAIRO, None)],
)
def test_facts_are_byte_identical_when_only_the_story_changes(name: str, recommends: list[str] | None) -> None:
    def told_as(story: str) -> Callable[[dict[str, Any]], None]:
        def change(doc: dict[str, Any]) -> None:
            _uncite(*_UNDECIDED)(doc)
            doc["story"] = story

        return change

    written = [
        [fact.model_dump_json() for fact in _facts(name, recommends=recommends, doc_change=told_as(story))]
        for story in ("decision", "tradeoff", "diagnosis", "status", "exploratory")
    ]

    assert written[0]
    assert all(each == written[0] for each in written)


def test_no_fact_tells_the_editor_what_to_conclude() -> None:
    """Facts state what holds: none says what should, must or ought to be done."""
    facts = [
        *_facts(_R1, recommends=["atlas-l"]),
        *_facts(_S1, recommends=["v3"]),
        *_facts(_R4, recommends=["f64"]),
        *_facts(_S1, recommends=["v2"], input_change=_unasked("billing")),
        *_facts(_EACH_TASK, doc_change=_uncite(*_UNDECIDED)),
    ]
    advice = ("should", "must", "ought", "consider", "needs to")

    assert len({fact.kind for fact in facts}) == 8
    assert not [fact.fact for fact in facts if any(word in fact.fact.lower() for word in advice)]


# --- a brace in a name the snapshot holds ---------------------------------------------------------------------------


def _renamed(
    path: tuple[str | int, ...], match: tuple[str, str] | None, value: str
) -> Callable[[dict[str, Any]], None]:
    """One name in the input set to ``value``: at ``path``, or in each item of the list there whose ``match`` field
    holds the given value."""

    def change(data: dict[str, Any]) -> None:
        node: Any = data
        for step in path[:-1]:
            node = node[step]
        if match is None:
            node[path[-1]] = value
            return
        field, held = match
        for item in node:
            if item.get(field) == held:
                item[path[-1]] = value

    return change


def _replaced(old: str, new: str) -> Callable[[dict[str, Any]], None]:
    """Every whole JSON string ``old`` in the input, and every ``old|`` cell id, rewritten with ``new``."""

    def change(data: dict[str, Any]) -> None:
        text = json.dumps(data).replace(f'"{old}"', f'"{new}"').replace(f'"{old}|', f'"{new}|')
        data.update(json.loads(text))

    return change


def test_a_brace_in_a_criterion_label_is_written_as_it_is() -> None:
    facts = _of(
        _facts(
            _R1, recommends=["atlas-l"], input_change=_renamed(("criteria", "label"), ("id", "p95"), "Latency {p95}")
        ),
        "gate_failed",
    )

    assert ["misses the declared gate Latency {p95} in " in fact.fact for fact in facts] == [True] * 4
    assert facts[0].figures == ["arm:atlas-l", "criteria.p95.atlas-l.billing.value", "criteria.p95.threshold"]


def test_a_brace_in_a_task_name_is_written_as_it_is() -> None:
    facts = _of(
        _facts(
            _R1, recommends=["atlas-l"], input_change=_renamed(("tasks", "name"), ("id", "billing"), "Billing {beta}")
        ),
        "gate_failed",
    )

    assert " in Billing {beta}: 3.6 s against 3 s." in facts[0].fact
    assert facts[0].figures == ["arm:atlas-l", "criteria.p95.atlas-l.billing.value", "criteria.p95.threshold"]


def test_a_brace_in_a_cell_id_is_written_as_it_is() -> None:
    """r3 with its complex row's id braced: Nano-XS few-shot misses the cell floor in that row's news cell."""
    facts = _of(
        _facts("prototype3/r3", recommends=["nano-xs.few"], input_change=_replaced("complex", "{complex}")),
        "gate_failed",
    )

    assert [fact.figures for fact in facts] == [
        ["arm:nano-xs.few", "criteria.cell_floor.nano-xs.few.{complex}.news.value", "criteria.cell_floor.threshold"]
    ]
    assert " in cell {complex}.news: " in facts[0].fact


def test_a_brace_in_an_aspect_name_and_a_floor_key_is_written_as_it_is() -> None:
    def braced(data: dict[str, Any]) -> None:
        _renamed(("aspects", "name"), ("id", "billing"), "Billing {x}")(data)
        _replaced("tone/rate_drop/reply.cites_policy", "tone/{rate}")(data)

    facts = _of(_facts(_S1, recommends=["v3"], input_change=braced), "warning_unanswered")

    assert [fact.figures for fact in facts] == [["arm:v3"], ["arm:v3"]]
    assert "floor billing/rate_drop/* on Billing {x} reads unclear" in facts[0].fact
    assert "floor tone/{rate} on tone reads unclear" in facts[1].fact


def test_a_brace_in_a_bar_measure_is_written_as_it_is() -> None:
    def missed(data: dict[str, Any]) -> None:
        data["bars"] = [
            {
                "aspect": "billing",
                "arm": "v3",
                "measure": "{m}",
                "bar": 1.0,
                "direction": "higher_is_better",
                "status": "missed",
            }
        ]

    facts = _of(_facts(_S1, recommends=["v3"], input_change=missed), "warning_unanswered")

    assert [fact.figures for fact in facts][-1] == ["arm:v3"]
    assert "it misses the bar on {m} on billing" in facts[-1].fact


def test_a_brace_in_a_guardrail_name_is_written_as_it_is() -> None:
    def breach(data: dict[str, Any]) -> None:
        data["guardrails"] = [{"arm": "f64", "dimension": "{tone}", "decision": "breached"}]

    facts = _of(_facts(_R4, recommends=["f64"], input_change=breach), "adoption")

    assert [(fact.figures, "breached the guardrail on {tone}." in fact.fact) for fact in facts] == [
        (["arm:f64", "arm:prod"], True)
    ]


def test_a_brace_in_an_arm_the_snapshot_does_not_hold_is_written_as_it_is() -> None:
    facts = _of(_facts(_KAIRO, recommends=["gh{o}st"]), "call_vs_dominance")

    assert [fact.figures for fact in facts] == [["arm:September"]]
    assert facts[0].fact.endswith("the document recommends `gh{o}st`.")


def test_a_brace_in_an_undecided_criterion_label_is_written_as_it_is() -> None:
    def braced(data: dict[str, Any]) -> None:
        _renamed(("criteria", "label"), ("id", "p95"), "Latency {p95}")(data)

    facts = _of(_facts(_EACH_TASK, doc_change=_uncite(*_UNDECIDED), input_change=braced), "undecided_implied")

    assert ["is undecided on Latency {p95}, which" in fact.fact for fact in facts] == [True, True]
    assert facts[0].figures == [
        "arm:atlas-m.high.r1",
        "measure.latency_ms.atlas-m.high.r1.p95",
        "criteria.p95.threshold",
    ]


def test_a_brace_in_an_unasked_aspect_name_is_written_as_it_is() -> None:
    def braced(data: dict[str, Any]) -> None:
        _unasked("billing")(data)
        _renamed(("aspects", "name"), ("id", "billing"), "Billing {x}")(data)

    facts = _of(_facts(_S1, recommends=["v2"], input_change=braced), "settled_loss_unasked")

    assert [fact.figures for fact in facts] == [
        ["arm:v2", "aspects.billing.reading.v2.interval", "aspects.billing.reading.v2.margin"]
    ]
    assert "reading on Billing {x}, which the campaign" in facts[0].fact
