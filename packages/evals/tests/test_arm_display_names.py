"""A name code computes from an arm's levers is printed by one rule, on every surface: ``Chunk tokens 1024tok``.

The owner's ruling (2026-10-28): "people might think Fanout 1 is different from fanout 1", so a computed lever arm name
is capitalised where the name is made, never in prose rendering, and prose, the arms key, the options table and a
chart's labels print the one name. Two pipelines make names, both downstream of ``arm_names``: the code-only page's
(``_arm_names`` and ``_short_names`` in ``analysis/report/build.py``) and the report input's ``Arm.name``
(``report_input_build``), which the document's page prints through ``printed_arm_name``. Both call
:func:`~threetears.evals.analysis.style.arm_display_name`.

"Computed" is read from data, never from the text's case: a report input arm is computed where no arm the campaign's
declaration names covers its variant key; a code-only page's name is computed where :func:`arm_name` spelled it from
the levers' reader names (``ArmName.levers``). A label a person gave, a declared short name and a model's name print
as written. The arm's id, its variant key and ``arm_names``' ``axis=level`` names do not move.

The campaign here is the toy host's, three runs at two chunk sizes and two retriever depths, its control named from
its first run, so its two candidates are derived and named by the levers they differ on.

The arms key's Configuration column names each setting by the rule measures and aspects are named by: the input's
declared name for it (a factor's), else the package's reader name (``chunk tokens``, never ``chunk_tokens``).
"""

from __future__ import annotations

import html
from collections.abc import Iterator, Mapping
from functools import cache
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.campaign import toyhost_aspect_design, toyhost_campaign_of_one_aspect
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_plan import prose_reply
from threetears.evals.analysis import build_code_only_report, report_html, report_markdown
from threetears.evals.analysis.analyst.snapshot import analyst_input
from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.campaigns import set_campaign_control
from threetears.evals.analysis.report import ChartBlock, TableBlock
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report.doc_page.text import above_appendix_text
from threetears.evals.analysis.plan_assembly import check_structure, planned_report, prose_slots, read_prose
from threetears.evals.analysis.report.build import contrast_rows
from threetears.evals.analysis.report.figure_keys import printed_arm_name, render
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.plan_page.markdown import planned_page_markdown
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown
from threetears.evals.analysis.report.words import printed_arm_labels
from threetears.evals.analysis.report.model import Report
from threetears.evals.analysis.report_decision import every_caveat
from threetears.evals.analysis.report_doc_checks import check_report_doc
from threetears.evals.analysis.report_input import Factor, Level, ReportInput
from threetears.evals.analysis.report_plan_facts import complete_plan, plan_facts
from threetears.evals.analysis.service import assemble_campaign_bundle, campaign_report, record_campaign_call
from threetears.evals.analysis.style import arm_display_name, arm_name, lever_reader_name
from threetears.evals.analysis.report.blocks import DisclosureBlock
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.plan_page.tables import arms_key
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.kernel.declaration import OverriddenBar
from packages.evals.tests.analyst_support import fixture_inputs
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.kernel.report_plan import PlanProse, PlanStructure, ReportPlan

#: Each run's chunk size and retriever depth: the control first, then a candidate moving the chunk size, then one
#: moving both.
_RUNS = ((256, 3), (1024, 3), (1024, 5))

#: The two candidates' names as the levers spell them, by the run each was addressed from.
_COMPUTED = {1: "chunk tokens 1024tok, retriever top k 3", 2: "chunk tokens 1024tok, retriever top k 5"}

#: The same names as every surface prints them.
_PRINTED = {at: name[:1].upper() + name[1:] for at, name in _COMPUTED.items()}

#: A query of every row of its source.
_WHOLE: dict[str, Any] = {
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}


def _campaign(
    *, control_label: str | None = None, control_short_name: str | None = None
) -> tuple[EvalHost, EvalCampaign]:
    """The toy host holding the campaign over :data:`_RUNS`, its control named from the first run (relabelled and
    short-named where given): its other two arms are derived."""
    profile = toyhost_profile()
    host = toyhost_host(profile=profile)
    batches = [
        toyhost_batch(
            chunk_tokens=chunk, retriever_top_k=depth, extraction_schema="v1", ocr_engine_version="tess-5.3.1"
        )
        for chunk, depth in _RUNS
    ]
    for batch in batches:
        host.storage.save_eval_run(batch)
        for result in toyhost_measurements(
            batch, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8, layout_fidelity=3
        ):
            host.storage.save_eval_result(result)
    campaign = toyhost_campaign_of_one_aspect(
        id="9c4e1b7a-3d2f-4e5a-8b6c-0d1e2f3a4b5c",
        scope_id=TOYHOST_SCOPE,
        name="arm display names",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch in batches],
        design=toyhost_aspect_design(),
        created_by="test:fixture",
    )
    host.storage.save_campaign(campaign)
    campaign = set_campaign_control(
        host.storage, campaign.id, campaign.scope_id, batches[0].id, set_by="t", profile=profile
    )
    if control_label is not None or control_short_name is not None:
        arms = [
            arm.model_copy(
                update={
                    "label": control_label or arm.label,
                    **({"short_name": control_short_name} if control_short_name else {}),
                }
            )
            if arm.role == "control"
            else arm
            for arm in campaign.arms
        ]
        campaign = campaign.model_copy(update={"arms": arms})
        host.storage.save_campaign(campaign)
    return host, campaign


def _bundle(*, control_label: str | None = None, control_short_name: str | None = None) -> CampaignBundle:
    """The campaign's bundle (:func:`_campaign`)."""
    host, campaign = _campaign(control_label=control_label, control_short_name=control_short_name)
    return assemble_campaign_bundle(host, campaign.id, campaign.scope_id)


@cache
def _derived() -> CampaignBundle:
    return _bundle()


@cache
def _input() -> ReportInput:
    return analyst_input(_derived())


def _candidate(at: int) -> str:
    """The id of the candidate addressed from run ``at``: the label the campaign derived for it."""
    return _COMPUTED[at]


def _draft(report_input: ReportInput, words: str = "") -> ReportDoc:
    """A document recommending the first candidate, naming it by ``{arm:<id>}`` in its recommendation and its claim,
    the claim drawing every measure by arm; ``words`` appended to the recommendation."""
    first = _candidate(1)
    return ReportDoc.model_validate(
        {
            "schema": "report-doc/1",
            "story": "status",
            "headline": "The evidence supports no call yet.",
            "lede": "Each block below shows what the campaign read.",
            "recommends": [first],
            "confidence": {
                "form": "low",
                "caveats": [
                    {"id": caveat.id, "material_if": "It would move the call were one made."}
                    for caveat in every_caveat(report_input)
                ],
            },
            "decision": [{"slot": "recommendation", "text": "Run {arm:" + first + "}." + words}],
            "claims": [
                {
                    "id": "c1",
                    "text": "{arm:" + first + "} reads as the others do.",
                    "weight": "lead",
                    "evidence": [
                        {"block": "dot_interval", "query": {**_WHOLE, "source": "measures"}, "caption": "Every row."}
                    ],
                }
            ],
            "explore": [],
            "omitted": [],
            "next": [],
            "method": [{"term": "Source", "text": "The snapshot the report was written from."}],
        }
    )


@cache
def _page() -> DocPage:
    return build(_draft(_input()), _input(), strict=True)


def _cells(table: TableBlock, column: str) -> list[str]:
    return [str(row[column].value) for row in table.rows]


def _labels(node: Any) -> Iterator[str]:
    """Every text a chart's data shows a reader: its labels, short names, titles and tooltips, never an ``id``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("label", "short_label", "narrow_label", "short_name", "title") and isinstance(value, str):
                yield value
            elif key == "tip" and isinstance(value, list):
                yield from (str(line) for line in value)
            else:
                yield from _labels(value)
    elif isinstance(node, list):
        for item in node:
            yield from _labels(item)


def _doc_page_surfaces(page: DocPage) -> dict[str, list[str]]:
    """Where the document's page prints an arm: its prose, the options table, the arms key and the chart's labels."""
    (claim,) = page.claims
    (evidence,) = claim.evidence
    assert evidence.chart is not None and isinstance(evidence.chart.chart, ChartBlock)
    drawn = evidence.chart.chart.kit_chart()
    assert drawn is not None
    return {
        "prose": [above_appendix_text(page)],
        "options": _cells(page.options, "option"),
        "arms key": _cells(page.arms_key.table, "arm"),
        "chart": list(_labels(drawn[1])),
    }


def _assert_printed_everywhere(page: DocPage) -> None:
    """Each computed name, capitalised, on every surface; the name as the levers spell it on none."""
    for where, texts in _doc_page_surfaces(page).items():
        joined = " | ".join(texts)
        for at, printed in _PRINTED.items():
            assert printed in joined, (where, printed, joined[:400])
            assert _COMPUTED[at] not in joined, (where, _COMPUTED[at], joined[:400])


# --- the rule ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "levers", "printed"),
    [
        ("fanout 2", ["fanout"], "Fanout 2"),
        ("fanout 2 · retry 1", ["fanout", "retry"], "Fanout 2 · retry 1"),
        ("research timeout 30s, fanout 2", ["fanout", "research_timeout"], "Research timeout 30s, fanout 2"),
        ("careful", ["candidate"], "careful"),
        ("flash · fanout 2", ["fanout"], "flash · fanout 2"),
        ("fanouts 2", ["fanout"], "fanouts 2"),
        ("deepseek-v4.1-flash", ["model"], "deepseek-v4.1-flash"),
    ],
    ids=[
        "one lever",
        "two levers",
        "a lever with an underscore",
        "a level value",
        "a model first",
        "another word",
        "a model's name",
    ],
)
def test_a_name_opening_on_a_levers_reader_name_is_capitalised_and_any_other_printed_as_written(
    name: str, levers: list[str], printed: str
) -> None:
    assert arm_display_name(name, levers) == printed


def test_arm_name_says_which_levers_spell_a_label_and_a_level_value_carries_none() -> None:
    assert arm_name("fanout=2, retry=1").levers == ("fanout", "retry")
    assert arm_name("candidate=careful").levers == ()
    assert arm_name("model=deepseek/deepseek-v4.1-flash").levers == ()


# --- the analyst's input: Arm.name --------------------------------------------------------------------------------


def test_a_derived_arm_is_named_by_the_rule_and_keeps_its_id_its_key_and_its_lever_names() -> None:
    """``Arm.name`` is the display name; ``Arm.id`` the campaign's derived label, as the levers spell it; the variant
    key the campaign arm's; and ``arm_names`` still ``axis=level``."""
    bundle = _derived()
    arms = {arm.id: arm for arm in _input().arms}
    for at, computed in _COMPUTED.items():
        arm = arms[computed]
        (campaign_arm,) = [one for one in bundle.arms if one.label == computed]
        assert (arm.name, printed_arm_name(arm)) == (_PRINTED[at], _PRINTED[at])
        assert arm.variant_key == campaign_arm.variant_key
    names = arm_names(bundle.aspects[0].variant_index)
    assert sorted(names.values()) == [
        "chunk_tokens=1024tok, retriever_top_k=3",
        "chunk_tokens=1024tok, retriever_top_k=5",
        "chunk_tokens=256tok, retriever_top_k=3",
    ]


def test_a_label_a_person_gave_and_a_declared_short_name_print_as_written() -> None:
    """The control's label ``control``, and a given label opening on a lever's reader name (``chunk tokens small``,
    which only the data says is no computed name), print as written; so does a declared short name."""
    assert {arm.id: arm.name for arm in _input().arms}["control"] == "control"

    given = analyst_input(_bundle(control_label="chunk tokens small"))
    (control,) = [arm for arm in given.arms if arm.role == "control"]
    assert (control.id, control.name, printed_arm_name(control)) == ("chunk tokens small",) * 3

    declared = analyst_input(_bundle(control_short_name="chunk tokens S"))
    (control,) = [arm for arm in declared.arms if arm.role == "control"]
    assert printed_arm_name(control) == "chunk tokens S"


def test_a_model_named_arm_prints_its_model_as_written() -> None:
    """A derived arm that differs from the control in its model alone is named by the model: printed as written."""
    from_models = analyst_input(_model_bundle())
    (candidate,) = [arm for arm in from_models.arms if arm.role != "control"]
    assert (candidate.name, printed_arm_name(candidate)) == ("extractor-v3", "extractor-v3")


def _model_bundle() -> CampaignBundle:
    """The toy campaign at one chunk size and two models, its control named: the candidate differs in its model."""
    profile = toyhost_profile()
    host = toyhost_host(profile=profile)
    batches = [
        toyhost_batch(
            chunk_tokens=256, retriever_top_k=3, extraction_schema="v1", ocr_engine_version="tess-5.3.1", model=model
        ).model_copy(update={"candidate_model": model})
        for model in ("extractor-v2", "extractor-v3")
    ]
    for batch in batches:
        host.storage.save_eval_run(batch)
        for result in toyhost_measurements(
            batch, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8, layout_fidelity=3
        ):
            host.storage.save_eval_result(result)
    campaign = toyhost_campaign_of_one_aspect(
        id="1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
        scope_id=TOYHOST_SCOPE,
        name="model names",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch in batches],
        design=toyhost_aspect_design(),
        created_by="test:fixture",
    )
    host.storage.save_campaign(campaign)
    campaign = set_campaign_control(
        host.storage, campaign.id, campaign.scope_id, batches[0].id, set_by="t", profile=profile
    )
    return assemble_campaign_bundle(host, campaign.id, campaign.scope_id)


# --- the document's page ---------------------------------------------------------------------------------------------


def test_every_surface_of_the_documents_page_prints_the_one_name() -> None:
    """Prose (``{arm:<id>}``), the options table, the arms key and the chart's labels each print ``Chunk tokens
    1024tok, retriever top k 3``, in every form, and nowhere the name as the levers spell it."""
    page = _page()
    _assert_printed_everywhere(page)
    for written in (doc_page_markdown(page), html.unescape(doc_page_html(page))):
        assert _PRINTED[1] in written
        assert "Run " + _PRINTED[1] in written


def _capitalised_as_rendered(text: str, report_input: ReportInput, arm_names: Mapping[str, str]) -> str:
    """The mutation's prose renderer: every arm's name capitalised as the prose is rendered, not where it is made."""
    return render(text, report_input, {arm: name[:1].upper() + name[1:] for arm, name in arm_names.items()})


def test_a_name_capitalised_in_prose_rendering_instead_fails_the_surfaces_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mutation: the input's name left as the levers spell it (the rule not called where the name is made) and
    the page's prose capitalising it as it renders. The prose then prints the capitalised name while the tables and
    the chart print the other, so the check above fails: the name is made in one place, never in prose rendering."""
    monkeypatch.setattr("threetears.evals.analysis.report.words.arm_display_name", lambda name, levers: name)
    monkeypatch.setattr("threetears.evals.analysis.report.doc_page.build.render", _capitalised_as_rendered)
    unmade = analyst_input(_bundle())
    assert {arm.id: arm.name for arm in unmade.arms}[_candidate(1)] == _COMPUTED[1]
    page = build(_draft(unmade), unmade, strict=True)
    surfaces = _doc_page_surfaces(page)
    assert _PRINTED[1] in surfaces["prose"][0], "the control: the mutation capitalises the prose"
    assert _COMPUTED[1] in " | ".join(surfaces["options"]), "the control: the tables print the name as made"
    with pytest.raises(AssertionError):
        _assert_printed_everywhere(page)


def test_prose_typing_a_derived_arms_id_is_its_name_in_another_case_and_stands() -> None:
    """The invariant refusing prose that types a lever arm's id the page prints by another name (220cad55b) compares
    them case-folded: the id differs from the printed name only in case, so prose typing it is no internal id."""
    held = _input()
    words = f" Read {_candidate(2)} as the control's equal."
    doc = _draft(held, words)
    verdicts = check_report_doc(doc, held, above_appendix_text=above_appendix_text(build(doc, held, strict=True)))
    assert verdicts == [], [(verdict.invariant, verdict.fault) for verdict in verdicts]


# --- the code-only page ----------------------------------------------------------------------------------------------


@cache
def _code_only() -> Report:
    """The campaign's one aspect's code-only page, naming each arm by its variant: what the page computes."""
    return build_code_only_report(_derived().aspects[0], assembled_at="2026-10-28T00:00:00+00:00", campaign_name="t")


def test_the_code_only_page_prints_a_computed_name_by_the_same_rule() -> None:
    """Its scoreboard names each arm by the rule, its outcome bar's short name too (the candidate moving one lever by
    that lever alone, ``Chunk tokens 1024tok``), in Markdown and HTML alike."""
    report = _code_only()
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "scoreboard"]
    names = {str(row["arm"]).removeprefix("▸ ") for row in table.values()}
    assert names == {"Chunk tokens 256tok, retriever top k 3", _PRINTED[1], _PRINTED[2]}
    (chart,) = [block for block in report.blocks if isinstance(block, ChartBlock) and block.chart == "outcome_bar"]
    shorts = {bar.arm: bar.short_name for bar in chart.bars}
    assert shorts[_PRINTED[1]] == "Chunk tokens 1024tok"
    for written in (report_markdown(report), html.unescape(report_html(report))):
        assert all(name in written for name in names)
        assert not any(computed in written for computed in _COMPUTED.values())


# --- the arms key's Configuration column ---------------------------------------------------------------------------


def test_the_configuration_column_names_each_setting_by_its_reader_name_and_keeps_values_as_written() -> None:
    """A setting reads by the package's reader name (``chunk tokens``, never ``chunk_tokens``), the model and every
    value as written."""
    settings = dict(zip(_cells(_page().arms_key.table, "arm"), _cells(_page().arms_key.table, "settings"), strict=True))
    assert settings["control"] == "extractor-v2 · chunk tokens 256tok · retriever top k 3"
    assert settings[_PRINTED[2]] == "extractor-v2 · chunk tokens 1024tok · retriever top k 5"
    assert not any("_" in cell for cell in settings.values())


def test_a_setting_the_input_names_reads_by_that_name() -> None:
    """A factor the input declares for a setting names it: ``Chunk size 256tok``; the others fall back."""
    held = _input()
    named = held.model_copy(
        update={"factors": [Factor(id="chunk_tokens", name="Chunk size", levels=[Level(id="256tok")])]}
    )
    page = build(_draft(named), named, strict=True)
    (control,) = [row for row in page.arms_key.table.rows if row["arm"].value == "control"]
    assert control["settings"].value == "extractor-v2 · Chunk size 256tok · retriever top k 3"


def test_on_r2_both_columns_read_the_declared_names() -> None:
    """r2 declares its arms' names and its factors' (``Reasoning budget``, ``Retry on failure``): the arm column prints
    each arm's declared name as written, and the Configuration column each setting by its factor's declared name, the
    value as written."""
    held = fixture_inputs()["r2"]
    factors = {factor.id: factor.name for factor in held.factors}
    assert (factors["reasoning"], factors["retry"]) == ("Reasoning budget", "Retry on failure"), "the control"
    rows = {
        row["arm"].value: row["settings"].value for row in arms_key(held, block_look(held, fact_arms(held))).table.rows
    }
    for arm in held.arms:
        settings = arm.settings
        assert rows[printed_arm_name(arm)] == (
            f"{settings['model']} · Reasoning budget {settings['reasoning']} · Retry on failure {settings['retry']}"
        )


def test_a_computed_name_and_a_setting_read_one_lever_reader_name() -> None:
    """The one source both columns read: a lever's declared name where one is, else its humanized id. A computed name
    opening on the declared name is capitalised by it, and one opening on the humanized id where nothing declares one."""
    declared = {"reasoning": "reasoning budget"}
    assert lever_reader_name("reasoning", declared) == "reasoning budget"
    assert lever_reader_name("research_timeout") == "research timeout"
    assert arm_display_name("reasoning budget high, retry 1", ["reasoning", "retry"], declared) == (
        "Reasoning budget high, retry 1"
    )
    assert arm_display_name("research timeout 30s", ["research_timeout"]) == "Research timeout 30s"


# --- one campaign, every surface -------------------------------------------------------------------------------------


def _plan(report_input: ReportInput) -> ReportPlan:
    """A decide-mode plan over the input, drawing code's spine (a forest of contrasts) on one measure, with prose for
    every key it creates: what a planned campaign page lays out."""
    facts = plan_facts(report_input)
    # The control is the one option code finds at least as good as every other, so the plan recommends it.
    recommended = next(arm.id for arm in report_input.arms if arm.role == "control")
    rows = [{"option": recommended, "role": "recommended"}] + [
        {"option": arm.id, "role": "control" if arm.role == "control" else "alternative"}
        for arm in report_input.arms
        if arm.id != recommended
    ]
    landings = [
        {
            "caveat": caveat.id,
            "at": "confidence"
            if caveat.reach == "report"
            else ("method.0" if caveat.reach == "method" else "decision.watch"),
        }
        for caveat in facts.caveats
        if not caveat.checked
    ]
    chart = {
        "kind": facts.spine.kind,
        "query": {**_WHOLE, "source": facts.spine.source or "contrasts", "measures": ["field_accuracy"]},
        "reference": [],
        "emphasis": "",
    }
    structure = {
        "mode": "decide",
        "headline_kind": "decision",
        "confidence": "low",
        "caveat_landings": landings,
        "options": {"rows": rows},
        "summary": [],
        "sections": [{"id": "outcomes", "chart": chart}],
        "explore": [],
        "method_items": 3,
        "durable_claims": [{"confidence": "low", "keys": [f"pass_rate.{recommended}"]}],
    }
    chosen = PlanStructure.model_validate(structure)
    check_structure(chosen, report_input, facts, facts.allowed_modes)
    prose = read_prose(PlanProse.model_validate(prose_reply(report_input, structure)), prose_slots(chosen, facts))
    return complete_plan(planned_report(chosen, prose, facts), report_input, facts)


def test_one_campaign_prints_each_derived_arm_one_way_on_every_surface() -> None:
    """The campaign's page (its call lines, its readings and every aspect's fold), the plan page, the document's page,
    the code-only page and quick's contrast rows: each prints both candidates capitalised, none the name as the levers spell it, so a
    planned campaign page, which lays a plan page over the folds, reads one name throughout."""
    host, campaign = _campaign()
    bundle = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)
    held = analyst_input(bundle)
    decided = next(arm.variant_key for arm in bundle.arms if arm.label == _COMPUTED[1])
    missed = [
        OverriddenBar(aspect=status.aspect, measure=status.measure)
        for status in bundle.readings.missed_bars()
        if status.arm == decided
    ]
    record_campaign_call(
        host, campaign.id, campaign.scope_id, arm=decided, called_by="op", reason="t", bars_overridden=missed
    )
    page = campaign_report(host, campaign.id, campaign.scope_id)
    calls = [block.text for block in page.blocks if isinstance(block, DisclosureBlock) and block.source == "call"]
    assert calls[0].startswith(f"Called: adopt {_PRINTED[1]}, by op"), calls
    assert not any(computed in line for line in calls for computed in _COMPUTED.values()), calls
    surfaces = {
        "campaign page": campaign_report_markdown(page),
        "campaign folds": " | ".join(report_markdown(fold.report) for fold in page.aspects),
        "plan page": planned_page_markdown(plan_page(_plan(held), held)),
        "document page": doc_page_markdown(build(_draft(held), held, strict=True)),
        "code-only page": report_markdown(_code_only()),
        "quick's contrasts": " | ".join(
            f"{row['contrast']} / {row['control']}" for row in contrast_rows(bundle.aspects[0])
        ),
    }
    for where, text in surfaces.items():
        assert all(printed in text for printed in _PRINTED.values()), (where, text[:400])
        assert not any(computed in text for computed in _COMPUTED.values()), where


def test_the_campaign_pages_readings_print_the_name_and_the_bundle_keeps_the_label_it_read() -> None:
    """The campaign page relabels its readings for print, in every label and every reason naming the arm; the readings
    the bundle froze, which its fingerprint reads, keep the label they were read under."""
    host, campaign = _campaign()
    bundle = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)
    labels = bundle.readings.arm_labels()
    printed = bundle.readings.relabelled(printed_arm_labels(bundle)).arm_labels()
    keys = {arm.variant_key: arm.label for arm in bundle.arms if arm.label in _COMPUTED.values()}
    assert {labels[key] for key in keys} == set(_COMPUTED.values())
    assert {printed[key] for key in keys} == set(_PRINTED.values())
    assert bundle.readings.relabelled({}) is bundle.readings
