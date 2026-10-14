"""An arm is named by what tells it from the other arms of its report — and by nothing else (#567).

A seven-run assistant campaign whose runs shared every lever and differed only in their rig printed every
lever it carried, ``name=value`` joined, as each arm's name: about 1,700 characters, in sixty places,
the levers of other kinds included (``classifier_prompt=(not a classifier run)``). A reader needs the
difference, so the rule is:

- an arm is named by the levers on which the report's arms differ;
- one that differs on none is named by its candidate kind and model, plus its rig where an arm was
  measured under more than one;
- a lever that does not apply to the arm's kind (the engine's "not a run of this kind" level) is never
  named;
- every level is cut alike, wherever the name is printed — and two arms the cut would name alike are
  told apart by their digests, so no two arms ever print the same name;
- no report dumps an arm's whole lever set: it names the arm by its difference, and the bundle beside it carries the
  rest.

The first half pins the rule on the naming functions themselves; the second builds reports — code-only ones through
the toy host, and analysis reports by hand — and reads their tables in both outputs, Markdown and HTML, since a name
is only as consistent as the one surface that spells it differently.
"""

from __future__ import annotations

import html
from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from threetears.evals.analysis import Report, build_report, campaign_report, report_html, report_markdown
from threetears.evals.analysis.arms import (
    ELISION,
    LABEL_LEVEL_CHARS,
    NO_LEVER_MOVED,
    arm_label,
    arm_names,
    elide_level,
    short_digest,
    writer_arms,
)
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.report import TableBlock
from threetears.evals.kernel.analysis_measures import MeasureCollection, MeasureSummary
from threetears.evals.kernel.authored import AuthoredAnalysis
from threetears.evals.kernel import EvalCampaign
from threetears.evals.kernel.campaign import EvalAnalysis, FindingResolution, VariantIndexEntry
from threetears.evals.kernel.surface import CellFacts, DecisionSurface, MeasureFacts
from threetears.evals.kernel.host import HostProfile, KindContract
from threetears.evals.schema import SweepableValue
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.schema.models import EvalRun
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_design
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    toyhost_batch,
    toyhost_measurements,
    toyhost_observation,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.factories import make_analysis
from packages.evals.tests.report_support import toy_campaign_host

#: Three rigs, by their apparatus class ids.
RIG, RIG_B, RIG_C = "a" * 64, "b" * 64, "c" * 64

# =============================================================================
# The rule, on the naming functions
# =============================================================================


def _value(content: Any, display: str | None = None) -> SweepableValue:
    return SweepableValue.of(content, display=display if display is not None else str(content))


def _entry(**levers: SweepableValue) -> VariantIndexEntry:
    """An index entry over ``levers``; a dotted lever name is spelled with ``__`` here."""
    named = {name.replace("__", "."): value for name, value in levers.items()}
    return VariantIndexEntry(variant_key=compute_variant_key(named), levers=named)


#: What an assistant arm of #567 shared with every other: a long description, directives, a model and its kind.
_SHARED: dict[str, SweepableValue] = {
    "candidate_kind": _value("assistant"),
    "model": _value("openai/gpt-5-mini"),
    "backstory": _value("A support agent who " + "walks each customer through the fix one step at a time " * 20),
    "directives": _value("Answer in the customer's language; never guess an order number."),
}


class TestAnArmIsNamedByWhatTellsItApart:
    def test_the_only_arm_of_a_report_is_named_by_its_kind_and_model(self) -> None:
        """One arm differs from nothing, so no lever names it — its kind and model say what ran."""
        (entry,) = index = [_entry(**_SHARED)]
        assert arm_names(index) == {entry.variant_key: "candidate_kind=assistant, model=openai/gpt-5-mini"}

    def test_a_described_arm_with_neither_a_difference_nor_a_model_says_it_moved_nothing(self) -> None:
        (entry,) = index = [_entry(temperature=_value(0.2))]
        assert arm_names(index) == {entry.variant_key: NO_LEVER_MOVED}

    def test_arms_that_differ_on_one_lever_are_named_by_that_lever_alone(self) -> None:
        cool, warm = index = [_entry(**_SHARED, temperature=_value(t)) for t in (0.2, 0.7)]
        assert arm_names(index) == {cool.variant_key: "temperature=0.2", warm.variant_key: "temperature=0.7"}

    def test_an_axis_one_arm_lacks_still_tells_the_arms_apart(self) -> None:
        """Absence is a level: the control of a one-knob sweep is named by the model it shares."""
        control, swept = index = [_entry(**_SHARED), _entry(**_SHARED, temperature=_value(0.7))]
        names = arm_names(index)
        assert names[swept.variant_key] == "temperature=0.7"
        assert names[control.variant_key] == "candidate_kind=assistant, model=openai/gpt-5-mini"


class TestALeverThatDoesNotApplyIsNeverNamed:
    def _mixed(self, *prompts: str) -> tuple[VariantIndexEntry, list[VariantIndexEntry]]:
        """An assistant arm and one classifier arm per prompt, each carrying the other kind's levers as the
        engine resolves them: at that kind's "not a run of this kind" level."""
        assistant = _entry(**_SHARED, classifier__prompt=SweepableValue.not_this_kind("classifier"))
        not_assistant = SweepableValue.not_this_kind("assistant")
        classifiers = [
            _entry(
                candidate_kind=_value("classifier"),
                model=_value("openai/gpt-5-mini"),
                backstory=not_assistant,
                directives=not_assistant,
                classifier__prompt=_value(prompt),
            )
            for prompt in prompts
        ]
        return assistant, classifiers

    def test_across_kinds_the_kind_names_the_arm_and_nothing_inapplicable_does(self) -> None:
        """An assistant arm and a classifier arm differ by kind; each one's own levers do not set it apart
        from an arm they do not apply to."""
        assistant, (classifier,) = self._mixed("Label the request.")
        names = arm_names([assistant, classifier])
        assert names == {
            assistant.variant_key: "candidate_kind=assistant",
            classifier.variant_key: "candidate_kind=classifier",
        }

    def test_a_lever_that_differs_among_the_arms_it_applies_to_names_only_those(self) -> None:
        assistant, classifiers = self._mixed("Label the request.", "Label the request in one word.")
        names = arm_names([assistant, *classifiers])
        assert names[assistant.variant_key] == "candidate_kind=assistant"
        assert sorted(names[arm.variant_key] for arm in classifiers) == [
            "candidate_kind=classifier, classifier.prompt=Label the request in one word.",
            "candidate_kind=classifier, classifier.prompt=Label the request.",
        ]

    def test_the_writer_is_never_handed_a_lever_that_does_not_apply(self) -> None:
        assistant, (classifier,) = self._mixed("Label the request.")
        view = writer_arms([assistant, classifier])
        assert view["shared_levels"] == {"model": "openai/gpt-5-mini"}
        levels = {arm["variant_key"]: arm["levels"] for arm in view["arms"]}  # type: ignore[attr-defined]
        assert set(levels[assistant.variant_key]) == {"backstory", "candidate_kind", "directives"}
        assert set(levels[classifier.variant_key]) == {"candidate_kind", "classifier.prompt"}
        assert "(not a" not in repr(view)

    def test_arms_that_differ_only_where_a_lever_does_not_apply_are_still_told_apart(self) -> None:
        """No kind lever here, so nothing the rule may name separates them: each is named by its model, and
        the digest says the names shown do not tell them apart."""
        bare = _entry(model=_value("m"), extractor__style=SweepableValue.not_this_kind("extractor"))
        styled = _entry(model=_value("m"), extractor__style=_value("terse"))
        assert arm_names([bare, styled]) == {
            arm.variant_key: f"model=m (arm {short_digest(arm.variant_key)})" for arm in (bare, styled)
        }

    def test_a_host_value_that_only_reads_like_the_level_is_named_like_any_value(self) -> None:
        """Recognised by its content hash: a display alone proves nothing about what the level is."""
        lookalike = _value(None, display="(not a classifier run)")
        assert lookalike.not_of_kind is None
        (entry, other) = index = [_entry(prompt=lookalike), _entry(prompt=_value("Label the request."))]
        assert arm_names(index)[entry.variant_key] == "prompt=(not a classifier run)"
        assert arm_names(index)[other.variant_key] == "prompt=Label the request."

    def test_the_level_names_its_kind_and_is_the_one_a_kind_contract_mints(self) -> None:
        level = SweepableValue.not_this_kind("classifier")
        assert (level.not_of_kind, level.display) == ("classifier", "(not a classifier run)")
        another_kinds_run = toyhost_observation().model_copy(update={"candidate_kind": "something-else"})
        minted = TOY_EXTRACTOR_CONTRACT.levels(another_kinds_run)
        assert minted and {value.not_of_kind for value in minted.values()} == {TOY_EXTRACTOR_KIND}


#: Two long values that part only in the middle — exactly the characters a cut drops.
_PARTING_IN_THE_MIDDLE = tuple(
    "Open with the station ident, then " + word + " the request line for a full hour before the news"
    for word in ("read", "skip")
)


class TestNoTwoArmsShareAName:
    def test_a_long_level_is_cut_in_the_middle_on_one_line(self) -> None:
        text = "first line\n" + "x" * 100 + " the end of it"
        cut = elide_level(text)
        assert len(cut) == LABEL_LEVEL_CHARS and ELISION in cut and "\n" not in cut
        assert cut.startswith("first line x") and cut.endswith("the end of it")
        assert elide_level("short  value\n") == "short value"

    def test_arms_the_cut_would_name_alike_are_told_apart_by_their_digests(self) -> None:
        first, second = index = [_entry(**_SHARED, opening=_value(text)) for text in _PARTING_IN_THE_MIDDLE]
        assert elide_level(_PARTING_IN_THE_MIDDLE[0]) == elide_level(_PARTING_IN_THE_MIDDLE[1])
        names = arm_names(index)
        assert names[first.variant_key] != names[second.variant_key]
        for entry in (first, second):
            assert names[entry.variant_key] == (
                f"opening={elide_level(_PARTING_IN_THE_MIDDLE[0])} (arm {short_digest(entry.variant_key)})"
            )

    def test_two_values_a_host_displayed_alike_are_told_apart_too(self) -> None:
        index = [_entry(prompt=_value(text, display="a prompt")) for text in ("one", "two")]
        names = arm_names(index)
        assert len(set(names.values())) == 2 and all(
            name.startswith("prompt=a prompt (arm ") for name in names.values()
        )

    def test_arms_the_cut_keeps_apart_carry_no_digest(self) -> None:
        """The digest marks levels that do not tell arms apart, so it appears only where they do not."""
        texts = [_PARTING_IN_THE_MIDDLE[0] + suffix for suffix in (" — rev 1", " — rev 2")]
        index = [_entry(**_SHARED, opening=_value(text)) for text in texts]
        names = arm_names(index)
        assert sorted(names.values()) == sorted(f"opening={elide_level(text)}" for text in texts)
        assert not any("(arm " in name for name in names.values())

    def test_a_cell_adds_its_rig_and_an_arm_the_index_lacks_says_so(self) -> None:
        (entry,) = index = [_entry(**_SHARED)]
        names = arm_names(index)
        assert arm_label(entry.variant_key, names, rig="abc") == f"{names[entry.variant_key]} @ rig abc"
        assert arm_label("f" * 64, names).startswith(f"unplaced ({'f' * 12})")


# =============================================================================
# Every block of a report, in all three outputs
# =============================================================================


class _Summarizer(BaseModel):
    """A second kind's one overlay, so every toy-extractor run carries a lever that does not apply to it."""

    style: str = Field(default="terse", description="how terse the summary reads")


def _profile_with_a_second_kind() -> HostProfile:
    return replace(
        toyhost_profile(), kinds=(TOY_EXTRACTOR_CONTRACT, KindContract("toy-summarizer", overlays=_Summarizer))
    )


def _code_only_report(batches: list[EvalRun], profile: HostProfile) -> Report:
    """The code-only report of a toy-host campaign over ``batches``, measured alike."""
    host = toyhost_host(profile=profile)
    for batch in batches:
        host.storage.save_eval_run(batch)
        for result in toyhost_measurements(
            batch, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8, layout_fidelity=3
        ):
            host.storage.save_eval_result(result)
    campaign = EvalCampaign(
        id="7b0e2c55-5d0f-4c8e-9a3c-1f2e3d4c5b6a",
        scope_id=TOYHOST_SCOPE,
        name="arm labels",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch in batches],
        declared_design=toyhost_design(),
        created_by="test:fixture",
    )
    host.storage.save_campaign(campaign)
    return campaign_report(host, campaign.id, campaign.scope_id)


def _batch(*, reviewer_pool: str = "pool-a", instructions: str | None = None) -> EvalRun:
    batch = toyhost_batch(
        chunk_tokens=256,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool=reviewer_pool,
        batch_label=instructions or reviewer_pool,
    )
    return batch if instructions is None else batch.model_copy(update={"overlays": {"instructions": instructions}})


def _table(report: Report, name: str) -> TableBlock:
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == name]
    return table


def _arm_column(report: Report, name: str, column: str = "arm") -> list[str]:
    """The named table's arm column, as served, without the control's marker."""
    return [str(row[column]).removeprefix("▸ ") for row in _table(report, name).values()]


def _assert_every_output_prints(report: Report, names: set[str]) -> None:
    """Each name the scoreboard prints reaches Markdown and HTML spelled alike, and no other arm name does."""
    markdown, page = report_markdown(report), report_html(report)
    assert set(_arm_column(report, "scoreboard")) == names
    for name in names:
        assert f"| {name} |" in markdown or f"| ▸ {name} |" in markdown, name
        assert f"<td>{html.escape(name)}</td>" in page or f"<td>▸ {html.escape(name)}</td>" in page, name


@pytest.fixture(scope="module")
def rig_only() -> Report:
    """#567's shape: two runs at identical levers under two rigs, with another kind's lever on each."""
    profile = _profile_with_a_second_kind()
    return _code_only_report([_batch(reviewer_pool="pool-a"), _batch(reviewer_pool="pool-b")], profile)


class TestARigOnlyCampaignIsNamedByItsModel:
    # The page reads the name's levers as words, never ``axis=level``.
    _ARM = f"candidate kind {TOY_EXTRACTOR_KIND}, model extractor-v2"

    def test_the_arm_is_named_by_its_kind_and_model(self, rig_only: Report) -> None:
        assert _arm_column(rig_only, "scoreboard") == [self._ARM]

    def test_every_output_prints_the_same_short_name(self, rig_only: Report) -> None:
        _assert_every_output_prints(rig_only, {self._ARM})

    def test_the_other_kinds_lever_is_printed_nowhere(self, rig_only: Report) -> None:
        for output in (report_markdown(rig_only), report_html(rig_only)):
            assert "toy-summarizer" not in output

    def test_no_output_dumps_the_levers_the_arms_share(self, rig_only: Report) -> None:
        for output in (report_markdown(rig_only), report_html(rig_only)):
            assert "chunk_tokens=256tok" not in output
            assert "retriever_top_k=3" not in output


@pytest.fixture(scope="module")
def one_lever() -> Report:
    """The toy campaign: two arms that differ on chunk width alone."""
    host, campaign = toy_campaign_host()
    return campaign_report(host, campaign.id, campaign.scope_id)


@pytest.fixture(scope="module")
def cut_alike() -> Report:
    """Two arms whose one difference is in the middle of a long value — the characters a cut drops."""
    return _code_only_report([_batch(instructions=text) for text in _PARTING_IN_THE_MIDDLE], toyhost_profile())


class TestArmsDifferingOnOneLeverAreNamedByIt:
    def test_every_block_names_each_arm_by_that_lever_alone(self, one_lever: Report) -> None:
        _assert_every_output_prints(one_lever, {"256tok", "1024tok"})

    def test_the_cases_table_heads_each_arm_s_column_by_the_same_name(self, one_lever: Report) -> None:
        headers = {column.header.removeprefix("▸ ") for column in _table(one_lever, "cases").columns}
        assert {"256tok", "1024tok"} <= headers


class TestTheCutNeverMergesTwoArms:
    def test_the_two_arms_are_named_apart_in_every_output(self, cut_alike: Report) -> None:
        arms = _arm_column(cut_alike, "scoreboard")
        assert len(arms) == 2 and len(set(arms)) == 2
        assert all(arm.startswith("extractor.instructions=") and ELISION in arm and " (arm " in arm for arm in arms)
        _assert_every_output_prints(cut_alike, set(arms))


# =============================================================================
# An analysis report: its recommendations and evidence name arms the same way
# =============================================================================

#: What only the shared lever set holds — a label carrying it leaked the lever dump.
_LEVER_DUMP_MARK = "A support agent who"


def _rig_only_index() -> list[VariantIndexEntry]:
    """#567's campaign: one assistant stack, carrying another kind's lever, measured under three rigs."""
    return [_entry(**_SHARED, classifier__prompt=SweepableValue.not_this_kind("classifier"))]


def _mixed_index() -> list[VariantIndexEntry]:
    """An assistant arm and a classifier arm, each carrying the other kind's levers as the engine resolves them."""
    not_assistant = SweepableValue.not_this_kind("assistant")
    return [
        _entry(**_SHARED, classifier__prompt=SweepableValue.not_this_kind("classifier")),
        _entry(
            candidate_kind=_value("classifier"),
            model=_value("openai/gpt-5-mini"),
            backstory=not_assistant,
            directives=not_assistant,
            classifier__prompt=_value("Label the request."),
        ),
    ]


def _cell(variant: str, rig: str) -> CellFacts:
    """One cell: six trials over two cases, carrying latency."""
    latency = MeasureSummary(
        population="scored", name="total_ms", attribution_scope="end_to_end", n=6, n_independent=2, mean=41250.0
    )
    return CellFacts(
        variant_key=variant,
        apparatus_class_id=rig,
        run_ids=[f"run-{variant[:6]}-{rig[:2]}"],
        n_observations=6,
        n_cases=2,
        repeats_per_case_min=3,
        repeats_per_case_max=3,
        measures=MeasureCollection(measures=[latency]),
    )


def _analysis_naming(cells: list[tuple[str, str]], index: list[VariantIndexEntry]) -> EvalAnalysis:
    """An analysis whose recommendation and finding evidence name every cell in ``cells``, over ``index``."""
    refs = [cell_ref(variant, rig) for variant, rig in cells]
    rows = [
        {"cell_ref": ref, "measure_id": "total_ms", "reading": "measure", "value": 41250.0, "n": 6, "dispersion": "sem"}
        for ref in refs
    ]
    document = AuthoredAnalysis.model_validate(
        {
            "findings": [
                {
                    "title": "No rig moved the latency.",
                    "body": "",
                    "confidence": "high",
                    "axes": [],
                    "evidence": [{"cell": ref, "measure_id": "total_ms", "reading": "measure"} for ref in refs],
                    "trials": [],
                    "caveats": [],
                    "invalidates": [],
                    "durable": "",
                }
            ],
            "decisions": [
                {
                    "proposal": "Keep the current assistant prompt.",
                    "disposition": "deferred",
                    "cells": refs,
                    "confidence": "high",
                    "rests_on": [0],
                    "revisit_when": "a second assistant is measured",
                }
            ],
            "questions": [],
            "next": [],
        }
    )
    surface = DecisionSurface(
        cells=sorted(
            (_cell(variant, rig) for variant, rig in cells), key=lambda c: (c.variant_key, c.apparatus_class_id)
        ),
        measures={"total_ms": MeasureFacts(unit="ms", merit_axis="latency", higher_is_better=False)},
    )
    return make_analysis(
        document=document,
        decision_surface=surface,
        variant_index=index,
        resolutions=[FindingResolution.model_validate({"evidence": rows})],
    )


def _rig_only_analysis() -> tuple[EvalAnalysis, list[str]]:
    (arm,) = _rig_only_index()
    rigs = [RIG, RIG_B, RIG_C]
    names = [f"candidate kind assistant, model openai/gpt-5-mini (test setup {rig[:12]})" for rig in rigs]
    return _analysis_naming([(arm.variant_key, rig) for rig in rigs], [arm]), names


def _mixed_analysis() -> tuple[EvalAnalysis, list[str]]:
    assistant, classifier = index = _mixed_index()
    return (
        _analysis_naming([(assistant.variant_key, RIG), (classifier.variant_key, RIG)], index),
        ["assistant", "classifier"],
    )


@pytest.fixture(params=[_rig_only_analysis, _mixed_analysis], ids=["one-arm-three-rigs", "two-kinds"])
def named(request: pytest.FixtureRequest) -> tuple[Report, list[str]]:
    """An analysis report, and the names its recommendation and evidence must give the cells they name, in order."""
    subject, names = request.param()
    return build_report(subject), names


class TestAnAnalysisReportNamesArmsByTheRule:
    def test_the_recommendation_names_each_arm_and_nothing_else(self, named: tuple[Report, list[str]]) -> None:
        report, names = named
        (facts,) = [
            block for block in report.blocks if isinstance(block, TableBlock) and block.name == "recommendation_facts"
        ]
        assert {"item": "Arms", "value": "; ".join(names)} in facts.values()
        assert "**Deferred:** Keep the current assistant prompt." in report_markdown(report).splitlines()
        assert f"| Arms | {'; '.join(names)} |" in report_markdown(report)
        assert f"<td>{html.escape('; '.join(names))}</td>" in report_html(report)

    def test_the_evidence_names_the_cells_alike(self, named: tuple[Report, list[str]]) -> None:
        report, names = named
        assert _arm_column(report, "evidence") == names

    def test_no_output_carries_a_lever_dump_or_an_inapplicable_lever(self, named: tuple[Report, list[str]]) -> None:
        report, _ = named
        for output in (report_markdown(report), report_html(report)):
            assert _LEVER_DUMP_MARK not in output
            assert "(not a " not in output
