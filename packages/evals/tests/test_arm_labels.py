"""An arm is named by what tells it from the other arms of its report — and by nothing else (#567).

A seven-run persona campaign whose runs shared every lever and differed only in their rig printed every
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
- the full lever set is stated once per arm, in the Arms table.

The first half pins the rule on the naming functions themselves; the second builds code-only reports
through the toy host and reads every block of them in all three outputs — Markdown, HTML and the chart
intents — since a name is only as consistent as the one surface that spells it differently.
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
    arm_settings,
    elide_level,
    short_digest,
    writer_arms,
)
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.report import ChartBlock, TableBlock, TextBlock
from threetears.evals.contracts.authored import NO_CHART, AuthoredAnalysis
from threetears.evals.contracts import EvalCampaign
from threetears.evals.contracts.campaign import EvalAnalysis, FindingResolution, VariantIndexEntry
from threetears.evals.contracts.surface import DecisionSurface, StratumFacts
from threetears.evals.contracts.host import HostProfile, KindContract, SweepableValue
from threetears.evals.contracts.identity import compute_variant_key
from threetears.evals.contracts.models import EvalRun
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
from packages.evals.tests.report_support import toy_campaign_host
from packages.evals.tests.test_surface_table import RIG, RIG_B, analysis, cell, measures

# =============================================================================
# The rule, on the naming functions
# =============================================================================


def _value(content: Any, display: str | None = None) -> SweepableValue:
    return SweepableValue.of(content, display=display if display is not None else str(content))


def _entry(**levers: SweepableValue) -> VariantIndexEntry:
    """An index entry over ``levers``; a dotted lever name is spelled with ``__`` here."""
    named = {name.replace("__", "."): value for name, value in levers.items()}
    return VariantIndexEntry(variant_key=compute_variant_key(named), levers=named)


#: What a persona arm of #567 shared with every other: a long description, directives, a model and its kind.
_SHARED: dict[str, SweepableValue] = {
    "candidate_kind": _value("persona"),
    "model": _value("openai/gpt-5-mini"),
    "backstory": _value("A late-night DJ who " + "talks between tracks about the records she loves " * 20),
    "directives": _value("Keep the room moving; never talk over a vocal."),
}


class TestAnArmIsNamedByWhatTellsItApart:
    def test_the_only_arm_of_a_report_is_named_by_its_kind_and_model(self) -> None:
        """One arm differs from nothing, so no lever names it — its kind and model say what ran."""
        (entry,) = index = [_entry(**_SHARED)]
        assert arm_names(index) == {entry.variant_key: "candidate_kind=persona, model=openai/gpt-5-mini"}

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
        assert names[control.variant_key] == "candidate_kind=persona, model=openai/gpt-5-mini"


class TestALeverThatDoesNotApplyIsNeverNamed:
    def _mixed(self, *prompts: str) -> tuple[VariantIndexEntry, list[VariantIndexEntry]]:
        """A persona arm and one classifier arm per prompt, each carrying the other kind's levers as the
        engine resolves them: at that kind's "not a run of this kind" level."""
        persona = _entry(**_SHARED, classifier__prompt=SweepableValue.not_this_kind("classifier"))
        not_persona = SweepableValue.not_this_kind("persona")
        classifiers = [
            _entry(
                candidate_kind=_value("classifier"),
                model=_value("openai/gpt-5-mini"),
                backstory=not_persona,
                directives=not_persona,
                classifier__prompt=_value(prompt),
            )
            for prompt in prompts
        ]
        return persona, classifiers

    def test_across_kinds_the_kind_names_the_arm_and_nothing_inapplicable_does(self) -> None:
        """A persona arm and a classifier arm differ by kind; each one's own levers do not set it apart
        from an arm they do not apply to."""
        persona, (classifier,) = self._mixed("Label the request.")
        names = arm_names([persona, classifier])
        assert names == {
            persona.variant_key: "candidate_kind=persona",
            classifier.variant_key: "candidate_kind=classifier",
        }

    def test_a_lever_that_differs_among_the_arms_it_applies_to_names_only_those(self) -> None:
        persona, classifiers = self._mixed("Label the request.", "Label the request in one word.")
        names = arm_names([persona, *classifiers])
        assert names[persona.variant_key] == "candidate_kind=persona"
        assert sorted(names[arm.variant_key] for arm in classifiers) == [
            "candidate_kind=classifier, classifier.prompt=Label the request in one word.",
            "candidate_kind=classifier, classifier.prompt=Label the request.",
        ]

    def test_what_an_arm_ran_leaves_out_what_does_not_apply_to_it(self) -> None:
        persona, (classifier,) = self._mixed("Label the request.")
        assert {level.axis_id for level in arm_settings(persona)} == {
            "backstory",
            "candidate_kind",
            "directives",
            "model",
        }
        assert {level.axis_id for level in arm_settings(classifier)} == {"candidate_kind", "classifier.prompt", "model"}

    def test_the_writer_is_never_handed_a_lever_that_does_not_apply(self) -> None:
        persona, (classifier,) = self._mixed("Label the request.")
        view = writer_arms([persona, classifier])
        assert view["shared_levels"] == {"model": "openai/gpt-5-mini"}
        levels = {arm["variant_key"]: arm["levels"] for arm in view["arms"]}  # type: ignore[attr-defined]
        assert set(levels[persona.variant_key]) == {"backstory", "candidate_kind", "directives"}
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


def _arm_column(report: Report, name: str) -> list[str]:
    """The Arm column of the named table, as served."""
    return [str(row["arm"]) for row in _table(report, name).rows]


def _chart_groups(report: Report) -> list[list[str]]:
    """Each chart's groups, in the order the intent draws them."""
    charts = [block.intent for block in report.blocks if isinstance(block, ChartBlock) and block.intent is not None]
    assert charts, "the report drew no chart, so nothing here was read"
    return [list(intent.identity.order) for intent in charts]


def _every_name(report: Report) -> list[str]:
    """Every arm and cell name the report prints: the two tables' Arm columns and every chart group."""
    return [
        *_arm_column(report, "arms"),
        *_arm_column(report, "surface"),
        *(group for groups in _chart_groups(report) for group in groups),
    ]


def _assert_every_output_prints(report: Report, names: set[str]) -> None:
    """Each name reaches Markdown, HTML and the chart intents spelled alike, and no other name does."""
    markdown, page = report_markdown(report), report_html(report)
    printed = set(_every_name(report))
    assert printed == names
    for name in names:
        assert f"| {name} |" in markdown, name
        assert f"<td>{html.escape(name)}</td>" in page, name
    for groups in _chart_groups(report):
        assert set(groups) <= names


@pytest.fixture(scope="module")
def rig_only() -> Report:
    """#567's shape: two runs at identical levers under two rigs, with another kind's lever on each."""
    profile = _profile_with_a_second_kind()
    return _code_only_report([_batch(reviewer_pool="pool-a"), _batch(reviewer_pool="pool-b")], profile)


class TestARigOnlyCampaignIsNamedByItsModelAndRig:
    _ARM = f"candidate_kind={TOY_EXTRACTOR_KIND}, model=extractor-v2"

    def test_the_arm_is_named_by_its_kind_and_model(self, rig_only: Report) -> None:
        assert _arm_column(rig_only, "arms") == [self._ARM]

    def test_each_cell_adds_the_rig_that_tells_it_apart(self, rig_only: Report) -> None:
        cells = _arm_column(rig_only, "surface")
        assert len(cells) == 2 and len(set(cells)) == 2
        assert all(cell.startswith(f"{self._ARM} @ rig ") for cell in cells)
        assert all(groups == cells for groups in _chart_groups(rig_only))

    def test_every_output_prints_the_same_short_names(self, rig_only: Report) -> None:
        _assert_every_output_prints(rig_only, {self._ARM, *_arm_column(rig_only, "surface")})

    def test_the_other_kinds_lever_is_printed_nowhere(self, rig_only: Report) -> None:
        for output in (report_markdown(rig_only), report_html(rig_only), rig_only.to_canonical_json()):
            assert "toy-summarizer" not in output

    def test_the_full_lever_set_is_stated_once_in_the_arms_table(self, rig_only: Report) -> None:
        (row,) = _table(rig_only, "arms").rows
        levers = str(row["levers"])
        assert levers == (
            f"candidate_kind={TOY_EXTRACTOR_KIND}; chunk_tokens=256tok; extraction_prompt=extraction prompt, rev 3; "
            "extraction_schema=field schema v1; extractor.instructions=(none); extractor.page_limit=(none); "
            "extractor.prompt_style=(none); model=extractor-v2; retriever_top_k=3"
        )
        for output in (report_markdown(rig_only), report_html(rig_only)):
            assert output.count("chunk_tokens=256tok") == 1
            assert output.count("retriever_top_k=3") == 1


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
        names = {"chunk_tokens=256tok", "chunk_tokens=1024tok"}
        assert set(_arm_column(one_lever, "arms")) == names
        _assert_every_output_prints(one_lever, names)

    def test_each_arm_still_states_everything_it_ran_once(self, one_lever: Report) -> None:
        rows = _table(one_lever, "arms").rows
        assert len(rows) == 2
        for row in rows:
            levers = str(row["levers"])
            assert f"{row['arm']}; " in levers and "model=extractor-v2" in levers and "retriever_top_k=3" in levers


class TestTheCutNeverMergesTwoArms:
    def test_the_two_arms_are_named_apart_in_every_block_and_output(self, cut_alike: Report) -> None:
        arms = _arm_column(cut_alike, "arms")
        assert len(arms) == 2 and len(set(arms)) == 2
        assert all(arm.startswith("extractor.instructions=") and ELISION in arm and " (arm " in arm for arm in arms)
        assert set(_arm_column(cut_alike, "surface")) == set(arms)
        _assert_every_output_prints(cut_alike, set(arms))

    def test_the_arms_table_still_states_each_value_as_the_host_displayed_it(self, cut_alike: Report) -> None:
        stated = [str(row["levers"]) for row in _table(cut_alike, "arms").rows]
        for text in _PARTING_IN_THE_MIDDLE:
            assert sum(text[:59] in levers for levers in stated) == 1


# =============================================================================
# An analysis report: its decisions, findings and evidence name arms the same way
# =============================================================================

_RIG_C = "c" * 64

#: What only the shared lever set holds — a reader meets it once, in the Arms table, or the label leaked it.
_LEVER_DUMP_MARK = "A late-night DJ who"


def _rig_only_index() -> list[VariantIndexEntry]:
    """#567's campaign: one persona stack, carrying another kind's lever, measured under three rigs."""
    return [_entry(**_SHARED, classifier__prompt=SweepableValue.not_this_kind("classifier"))]


def _mixed_index() -> list[VariantIndexEntry]:
    """A persona arm and a classifier arm, each carrying the other kind's levers as the engine resolves them."""
    not_persona = SweepableValue.not_this_kind("persona")
    return [
        _entry(**_SHARED, classifier__prompt=SweepableValue.not_this_kind("classifier")),
        _entry(
            candidate_kind=_value("classifier"),
            model=_value("openai/gpt-5-mini"),
            backstory=not_persona,
            directives=not_persona,
            classifier__prompt=_value("Label the request."),
        ),
    ]


def _analysis_naming(cells: list[tuple[str, str]], index: list[VariantIndexEntry]) -> EvalAnalysis:
    """An analysis whose decision and finding evidence name every cell in ``cells``, generated over ``index``."""
    refs = [cell_ref(variant, rig) for variant, rig in cells]
    rows = [
        {"cell_ref": ref, "measure_id": "total_ms", "reading": "measure", "value": 41250.0, "n": 6, "dispersion": "sem"}
        for ref in refs
    ]
    document = AuthoredAnalysis.model_validate(
        {
            "headline": "Keep the current persona prompt.",
            "summary": "",
            "findings": [
                {
                    "title": "No rig moved the latency.",
                    "body": "",
                    "confidence": "high",
                    "axes": [],
                    "evidence": [{"cell": ref, "measure_id": "total_ms", "reading": "measure"} for ref in refs],
                    "chart": {"type": NO_CHART, "cells": [], "measures": [], "axis": "", "note": "", "caption": ""},
                    "caveats": [],
                    "invalidates": [],
                    "durable": "",
                }
            ],
            "decisions": [
                {
                    "proposal": "Keep the current persona prompt.",
                    "disposition": "deferred",
                    "cells": refs,
                    "confidence": "high",
                    "rests_on": [0],
                    "revisit_when": "a second persona is measured",
                }
            ],
            "questions": [],
            "next": [],
        }
    )
    surface = DecisionSurface(
        cells=sorted(
            (
                cell(variant=variant, rig=rig, strata=[StratumFacts(stratum="easy", n_observations=6, n_cases=2)])
                for variant, rig in cells
            ),
            key=lambda c: (c.variant_key, c.apparatus_class_id),
        ),
        measures=measures(),
    )
    return analysis(
        surface,
        document=document,
        design_snapshot=None,
        variant_index=index,
        resolutions=[FindingResolution.model_validate({"evidence": rows})],
    )


def _rig_only_analysis() -> tuple[EvalAnalysis, list[str]]:
    (arm,) = _rig_only_index()
    rigs = [RIG, RIG_B, _RIG_C]
    names = [f"candidate_kind=persona, model=openai/gpt-5-mini @ rig {rig[:12]}" for rig in rigs]
    return _analysis_naming([(arm.variant_key, rig) for rig in rigs], [arm]), names


def _mixed_analysis() -> tuple[EvalAnalysis, list[str]]:
    persona, classifier = index = _mixed_index()
    return (
        _analysis_naming([(persona.variant_key, RIG), (classifier.variant_key, RIG)], index),
        ["candidate_kind=persona", "candidate_kind=classifier"],
    )


@pytest.fixture(params=[_rig_only_analysis, _mixed_analysis], ids=["one-arm-three-rigs", "two-kinds"])
def named(request: pytest.FixtureRequest) -> tuple[Report, list[str]]:
    """An analysis report, and the names its decision and evidence must give the cells they name, in order."""
    subject, names = request.param()
    return build_report(subject), names


class TestAnAnalysisReportNamesArmsByTheRule:
    def test_the_decision_line_names_each_arm_and_nothing_else(self, named: tuple[Report, list[str]]) -> None:
        report, names = named
        (decision,) = [block for block in report.blocks if isinstance(block, TextBlock) and block.role == "decision"]
        assert {fact.name: fact.value for fact in decision.facts}["Arms"] == "; ".join(names)
        line = next(line for line in report_markdown(report).splitlines() if "Disposition: deferred" in line)
        assert line == (
            "- **Keep the current persona prompt.** — Disposition: deferred · Confidence: high · Arms: "
            + "; ".join(names)
            + ". Rests on finding 1."
        )
        assert f"Arms: {html.escape('; '.join(names))}" in report_html(report)

    def test_the_evidence_the_surface_and_the_strata_name_the_cells_alike(
        self, named: tuple[Report, list[str]]
    ) -> None:
        report, names = named
        assert _arm_column(report, "evidence") == names
        assert sorted(_arm_column(report, "surface")) == sorted(names)
        assert set(_arm_column(report, "strata")) == set(names)

    def test_no_output_carries_a_lever_dump_or_an_inapplicable_lever(self, named: tuple[Report, list[str]]) -> None:
        """The shared settings are stated once — the Arms table's lever column — and never in a name."""
        report, _ = named
        for output in (report_markdown(report), report_html(report), report.to_canonical_json()):
            assert output.count(_LEVER_DUMP_MARK) == 1
            assert "(not a " not in output


# =============================================================================
# A sweep chart's configurations follow the same rule
# =============================================================================


def _sweep_configs(index: list[VariantIndexEntry]) -> list[dict[str, str]]:
    """The configurations a sweep ranking draws over one cell per arm of ``index``."""
    from threetears.evals.contracts.analysis_measures import MeasureCollection, MeasureSummary
    from packages.evals.tests.test_viz_refs import VALID, build, surface

    def reading(name: str, mean: float, higher_is_better: bool) -> MeasureSummary:
        return MeasureSummary(
            population="scored",
            name=name,
            attribution_scope="end_to_end",
            higher_is_better=higher_is_better,
            n=6,
            n_independent=2,
            mean=mean,
            sem=mean * 0.05,
        )

    readings = MeasureCollection(measures=[reading("cost_usd", 0.01, False), reading("pass_rate", 0.8, True)])
    cells = [cell(variant=entry.variant_key, measures=readings) for entry in index]
    return [row["config"] for row in build(VALID["sweep_ranking"], surface(cells, timed=False), index)["rows"]]


class TestASweepConfigurationIsCutAndNamedAlike:
    def test_a_lever_that_does_not_apply_is_an_absent_level_not_a_placeholder(self) -> None:
        from threetears.evals.analysis.viz.payloads import ABSENT_LEVEL

        configs = _sweep_configs(_mixed_index())
        assert sorted(config["classifier.prompt"] for config in configs) == sorted([ABSENT_LEVEL, "Label the request."])
        assert "(not a " not in repr(configs)

    def test_a_long_level_is_cut_as_an_arm_name_cuts_it(self) -> None:
        texts = [_PARTING_IN_THE_MIDDLE[0] + suffix for suffix in (" — rev 1", " — rev 2")]
        configs = _sweep_configs([_entry(**_SHARED, opening=_value(text)) for text in texts])
        assert sorted(config["opening"] for config in configs) == sorted(elide_level(text) for text in texts)

    def test_a_cut_that_would_merge_two_configurations_is_not_made(self) -> None:
        configs = _sweep_configs([_entry(**_SHARED, opening=_value(text)) for text in _PARTING_IN_THE_MIDDLE])
        assert sorted(config["opening"] for config in configs) == sorted(_PARTING_IN_THE_MIDDLE)
