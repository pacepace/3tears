"""The editor's facts (``analysis/report_doc_facts.py``): each fires on a planted case, and on the converted documents
only where it holds.

Each fact is planted on a converted prototype document and its snapshot, changed as little as the fact needs. On the
eleven later converted documents (prototypes 2 and 3) each fact is pinned: most are silent, and where one fires the test
says why it is real. No fact raises on a misfit input, even beside a document written for another snapshot, and the
facts are byte-identical when only ``story`` changes.
"""

from __future__ import annotations

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
    assert facts[0].figures[0] == "weakest.arm"
    assert facts[0].fact.endswith("no Watch entry cites a key naming it.")


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


def test_an_undecided_options_implied_read_is_a_fact_whatever_the_document_recommends() -> None:
    """r2 with its gates read in each task: two options have no p95 figure in some task, and their p95 over every case
    misses the line."""
    facts = _of(_facts(_EACH_TASK), "undecided_implied")

    assert [(fact.path, fact.figures[0]) for fact in facts] == [
        ("", "arm:atlas-m.high.r1"),
        ("", "arm:brisk-s.high.r1"),
    ]
    assert "(inferred, not stated)" in facts[0].fact


def test_an_undecided_option_the_document_recommends_is_named_at_its_place() -> None:
    facts = _of(_facts(_EACH_TASK, recommends=["brisk-s.high.r1"]), "undecided_implied")

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


# --- the converted documents ---------------------------------------------------------------------------------------

#: The facts each later converted document triggers, by kind. Every one that fires is real:
#:
#: - r2 with its gates read in each task: two options are undecided on the p95 gate read in each task, and their p95
#:   over every case misses it. True, and the document recommends nothing; the editor weighs whether it says why.
#: - s1: the document keeps production where code reads a tradeoff (no option dominates), and the weakest margin, v2's
#:   tone floor, is one Watch does not name (it discusses v2's billing floor). Both true; whether the call and Watch
#:   are argued well is the editor's.
#: - s2: the document recommends beta, at low confidence, where code reads a tradeoff. True.
#: - Kairo: the weakest margin is qwen3.8-flash's reading against the margin, a candidate the call does not pick, and
#:   Watch does not name it. True; the editor weighs whether it matters to keeping September.
_EXPECTED: dict[str, list[str]] = {
    "prototype2/r2-with-its-gates-read-in-each-task": ["undecided_implied", "undecided_implied"],
    "prototype2/r2-with-one-gate": [],
    "prototype2/r3-without-its-configs": [],
    "prototype2/s1": ["call_vs_dominance", "weakest_not_in_watch"],
    "prototype2/s2": ["call_vs_dominance"],
    "prototype2/s3": [],
    "prototype3/kairo": ["weakest_not_in_watch"],
    "prototype3/r1": [],
    "prototype3/r2": [],
    "prototype3/r3": [],
    "prototype3/r4": [],
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

    assert [fact.kind for fact in facts] == ["call_vs_dominance", "weakest_not_in_watch"]
    assert facts[0].fact.endswith("the document recommends `ghost`.")


@pytest.mark.parametrize("name", [_EACH_TASK, _S1, _KAIRO])
def test_facts_are_byte_identical_when_only_the_story_changes(name: str) -> None:
    written = [
        [fact.model_dump_json() for fact in _facts(name, doc_change=lambda doc, story=story: doc.update(story=story))]
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
        *_facts(_EACH_TASK),
    ]
    advice = ("should", "must", "ought", "consider", "needs to")

    assert len({fact.kind for fact in facts}) == 7
    assert not [fact.fact for fact in facts if any(word in fact.fact.lower() for word in advice)]
