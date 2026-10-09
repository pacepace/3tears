"""EvalAnalysis / EvalInsight — model shape, validation, and storage round-trips.

The generated, stored analysis of a campaign plus the durable insights it mints. These tests pin
the stored shape — the authored document kept verbatim beside the per-finding resolutions code
filled — every confidence a tier, never a typed probability, the
"no point estimate without n + dispersion" rule, the doc_type discriminators, the
WITHIN-doc position checks (every link names a real finding, nothing invalidates itself or cycles,
one resolution per finding), and campaign-scoped / subject-scoped retrieval.

Storage is exercised through a real :class:`~threetears.evals.contracts.storage.EvalStorage`
over the same in-memory evals-repo stand-in the campaign tests use.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.contracts.authored import (
    AuthoredAnalysis,
    Caveat,
    Chart,
    Decision,
    EvidenceRef,
    Finding,
    MeasureRef,
    NextStep,
    QuestionAnswer,
)
from threetears.evals.contracts.campaign import (
    CoverageLens,
    EvalAnalysis,
    EvalInsight,
    EvidenceRow,
    FindingResolution,
    GenerationProvenance,
    LeverCoverage,
    RunIndexEntry,
    Viz,
)
from threetears.evals.contracts.surface import DecisionSurface

# Reuse the campaign tests' in-memory evals-repo stand-in + storage wiring.
from packages.evals.tests.factories import memory_storage

_NO_CHART = Chart(type="none", cells=[], measures=[], axis="", note="", caption="")


def _finding(title: str = "v", *, invalidates: list[int] | None = None, **overrides: Any) -> Finding:
    """An authored finding with every list empty unless a caller fills it."""
    fields: dict[str, Any] = dict(
        title=title,
        body="",
        confidence="medium",
        axes=[],
        evidence=[],
        chart=_NO_CHART,
        caveats=[],
        invalidates=invalidates or [],
        durable="",
    )
    fields.update(overrides)
    return Finding(**fields)


def _decision(**overrides: Any) -> Decision:
    """An authored decision resting on nothing unless a caller says otherwise."""
    fields: dict[str, Any] = dict(
        proposal="Adopt X.", disposition="adopted", cells=[], confidence="medium", rests_on=[], revisit_when=""
    )
    fields.update(overrides)
    return Decision(**fields)


def _make_analysis(**overrides: Any) -> EvalAnalysis:
    """Build an ``EvalAnalysis`` with EVERY nested object populated.

    Two findings (the second invalidating the first, with evidence, a chart, a caveat and a
    durable claim), a decision and a question answer resting on real positions, a next step, one
    resolution per finding carrying a resolved evidence row and a compiled chart, a coverage lever
    (with the required n + dispersion), a generation-provenance block, and a run-index row — so a
    round-trip has something in every branch.

    ``findings``, ``decisions``, ``questions`` and ``next`` fold into the document; ``resolutions``
    defaults to the populated pair when the findings are the default ones, and to one empty
    resolution per finding otherwise.
    """
    default_findings = [
        _finding(
            "Alpha-2 wins on latency at equal quality.",
            body="The tail shortens with no loss on the judged dimensions.",
            confidence="high",
            axes=["planner.model"],
            evidence=[EvidenceRef(cell="vk-alpha:ac-1", measure_id="elapsed_ms", reading="measure")],
            chart=Chart(
                type="delta_table",
                cells=["vk-alpha:ac-1", "vk-beta:ac-1"],
                measures=[MeasureRef(measure_id="elapsed_ms", reading="measure")],
                axis="",
                note="",
                caption="Alpha against Beta on the tail.",
            ),
            caveats=[Caveat(kind="sampling", text="k=1 on this template is noise-dominated.")],
            durable="Alpha-2 is the planner model of record for Maple.",
        ),
        _finding("Latency is output-bound, not input-bound.", axes=["output_max_tokens"], invalidates=[0]),
    ]
    authored_findings = overrides.pop("findings", None)
    findings = list(authored_findings or default_findings)
    document = AuthoredAnalysis(
        headline=overrides.pop("headline", "Alpha-2 is the planner model of record."),
        summary=overrides.pop("summary", "Cheapest, and zero timeouts."),
        findings=findings,
        decisions=overrides.pop(
            "decisions",
            [
                _decision(
                    proposal="Keep vendor/alpha-2 as Maple's planner_model.", cells=["vk-alpha:ac-1"], rests_on=[0]
                )
            ],
        ),
        questions=overrides.pop(
            "questions",
            [QuestionAnswer(question_id="q-model", resolution="answered", answer="Alpha-2.", rests_on=[0])],
        ),
        next=overrides.pop(
            "next",
            [NextStep(title="Confirm finalists at k=5.", why="k=1 is noise-dominated.", leverage="high", lever="")],
        ),
    )
    if not authored_findings:
        resolutions = [
            FindingResolution(
                evidence=[
                    EvidenceRow(
                        cell_ref="vk-alpha:ac-1",
                        measure_id="elapsed_ms",
                        reading="measure",
                        value=41000.0,
                        n=3,
                        dispersion="±900",
                    )
                ],
                chart=Viz(
                    type="delta_table",
                    payload={
                        "a_label": "alpha-2",
                        "b_label": "beta-1",
                        "rows": [{"metric": "mean_composite", "a": 0.8, "b": 0.6, "delta": -0.2, "significant": True}],
                    },
                    ref={
                        "a_cell": "vk-alpha:ac-1",
                        "b_cell": "vk-beta:ac-1",
                        "measures": [{"measure_id": "mean_composite"}],
                    },
                ),
            ),
            FindingResolution(chart_note="the chart named a cell the surface does not have"),
        ]
    else:
        resolutions = [FindingResolution() for _ in findings]
    defaults: dict[str, Any] = dict(
        document=document,
        resolutions=resolutions,
        campaign_id="camp-1",
        subject_id="ent-maple",
        subject_kind="agent",
        behavior="planning",
        observation_refs=["run-a", "run-b"],
        model_versions={"planner_model": "vendor/alpha-2"},
        scope_id="uni-1",
        generation=GenerationProvenance(
            prompt_id="eval_analysis_gen",
            prompt_version="v1",
            generator_model="anthropic/claude-opus",
            bundle_fingerprint="sha256:abc123",
            generated_at="2026-07-25T00:00:00+00:00",
            token_cost=0.42,
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
            repair_attempts=0,
            repaired_refusal=None,
            cell_model_version=1,
            user_message_digest="sha256:message",
        ),
        coverage=CoverageLens(
            levers=[
                LeverCoverage(
                    name="planner.model",
                    cells=5,
                    k=3,
                    n=15,
                    dispersion="±0.10",
                    status="measured",
                )
            ],
        ),
        run_index=[RunIndexEntry(run_id="run-a", config={"model": "alpha-2"}, key_metrics={"latency_s": 42.0})],
        decision_surface=DecisionSurface(),
    )
    defaults.update(overrides)
    return EvalAnalysis(**defaults)


def _make_insight(**overrides: Any) -> EvalInsight:
    """Build an ``EvalInsight`` with every non-default field populated."""
    defaults: dict[str, Any] = dict(
        scope_id="uni-1",
        subject_id="ent-maple",
        subject_kind="agent",
        statement="Alpha-2 is the planner model of record for Maple.",
        scope="planning",
        confidence="high",
        evidence_run_ids=["run-a"],
        evidence_result_ids=["res-a"],
        model_versions={"planner_model": "vendor/alpha-2"},
        invalidation_trigger="A new model beats it at k>=3.",
        source_campaign_id="camp-1",
        source_analysis_id="an-1",
    )
    defaults.update(overrides)
    return EvalInsight(**defaults)


# ---------------------------------------------------------------------------
# EvalAnalysis: full round-trip
# ---------------------------------------------------------------------------


def test_analysis_model_round_trip_preserves_every_field():
    analysis = _make_analysis()
    reloaded = EvalAnalysis.from_dict(analysis.to_dict())

    assert reloaded == analysis
    # The nested objects most easily dropped by a shape change survive intact.
    assert len(reloaded.document.findings) == 2
    assert reloaded.resolutions[0].chart == Viz(
        type="delta_table",
        payload={
            "a_label": "alpha-2",
            "b_label": "beta-1",
            "rows": [{"metric": "mean_composite", "a": 0.8, "b": 0.6, "delta": -0.2, "significant": True}],
        },
        ref={"a_cell": "vk-alpha:ac-1", "b_cell": "vk-beta:ac-1", "measures": [{"measure_id": "mean_composite"}]},
    )
    assert reloaded.resolutions[0].evidence[0].value == 41000.0
    assert reloaded.resolutions[1].chart_note == "the chart named a cell the surface does not have"
    assert reloaded.document.findings[0].chart.measures == [MeasureRef(measure_id="elapsed_ms", reading="measure")]
    assert reloaded.document.findings[0].caveats == [
        Caveat(kind="sampling", text="k=1 on this template is noise-dominated.")
    ]
    assert reloaded.document.findings[1].invalidates == [0]
    assert reloaded.document.decisions[0].rests_on == [0]
    assert reloaded.document.questions[0].rests_on == [0]
    assert reloaded.document.next[0].leverage == "high"
    assert reloaded.coverage.levers[0].n == 15
    assert reloaded.coverage.levers[0].dispersion == "±0.10"
    assert reloaded.generation.token_cost == 0.42
    assert reloaded.run_index[0].key_metrics == {"latency_s": 42.0}


def test_analysis_storage_round_trip_preserves_every_field():
    storage, _ = memory_storage()
    analysis = _make_analysis()

    storage.save_analysis(analysis)
    loaded = storage.load_analysis(analysis.id, analysis.scope_id)

    assert loaded is not None
    assert loaded == analysis
    assert loaded.resolutions[0].chart == analysis.resolutions[0].chart
    assert loaded.document == analysis.document
    assert loaded.generation == analysis.generation


def test_load_missing_analysis_returns_none():
    storage, _ = memory_storage()
    assert storage.load_analysis("does-not-exist", "uni-1") is None


@pytest.mark.parametrize("blank_field", ["campaign_id", "subject_id", "behavior"])
def test_analysis_required_reference_fields_reject_empty(blank_field):
    # campaign_id/subject_id/task are min_length=1 — a blank reference (the "FK
    # present" contract) must be rejected at construction, not stored empty.
    with pytest.raises(ValidationError):
        _make_analysis(**{blank_field: ""})


# ---------------------------------------------------------------------------
# EvalInsight: round-trip
# ---------------------------------------------------------------------------


def test_insight_model_round_trip_preserves_every_field():
    insight = _make_insight()
    reloaded = EvalInsight.from_dict(insight.to_dict())
    assert reloaded == insight


def test_insight_storage_round_trip_preserves_every_field():
    storage, _ = memory_storage()
    insight = _make_insight()

    storage.save_insight(insight)
    loaded = storage.query_insights("uni-1", subject_id="ent-maple")
    assert insight in loaded


# ---------------------------------------------------------------------------
# every confidence is a tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.7, "certain", None], ids=["a-probability", "off-tier", "null"])
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda c: _finding(confidence=c), id="Finding"),
        pytest.param(lambda c: _decision(confidence=c), id="Decision"),
        pytest.param(
            lambda c: EvalInsight(scope_id="uni-1", subject_kind="", subject_id="s", statement="st", confidence=c),
            id="EvalInsight",
        ),
    ],
)
def test_an_authored_confidence_is_a_tier_and_nothing_else(build, bad):
    """What the generator authors is a tier, and so is what an insight stores; a typed probability is refused."""
    with pytest.raises(ValidationError):
        build(bad)


def test_the_stored_tiers_are_the_authored_tiers():
    """One declaration of the tiers: what a writer may author and what a stored analysis may hold cannot diverge."""
    from typing import get_args

    from threetears.evals.contracts import authored, campaign

    assert campaign.ConfidenceTier is authored.Confidence
    assert campaign.CONFIDENCE_TIERS == get_args(authored.Confidence) == ("very_high", "high", "medium", "low")


# ---------------------------------------------------------------------------
# A lever's coverage carries no confidence
# ---------------------------------------------------------------------------


def test_a_lever_carries_no_confidence_looked_up_from_its_status():
    """``confidence`` was a fixed lookup on ``status``: it is not a field, and naming it says what became of it."""
    assert "confidence" not in LeverCoverage.model_fields
    with pytest.raises(ValidationError, match="`confidence` is not a field of LeverCoverage: it was removed"):
        LeverCoverage(name="l", cells=1, k=1, n=1, dispersion="±0", status="unswept", confidence="low")


@pytest.mark.parametrize("tier", ["high", "medium", "low"])
def test_a_stored_analysis_whose_levers_carry_a_confidence_still_loads(tier: str):
    """An analysis stored before the field was retired loads, the tier discarded and every fact kept."""
    analysis = _make_analysis()
    stored = analysis.to_dict()
    stored["coverage"]["levers"][0]["confidence"] = tier

    reloaded = EvalAnalysis.from_dict(stored)

    assert reloaded == analysis
    assert "confidence" not in reloaded.to_dict()["coverage"]["levers"][0]


def test_an_old_analysis_loads_through_storage():
    storage, store = memory_storage()
    analysis = _make_analysis()
    stored = analysis.to_dict()
    stored["coverage"]["levers"][0]["confidence"] = "high"
    store.upsert(stored)

    assert storage.load_analysis(analysis.id, analysis.scope_id) == analysis


# ---------------------------------------------------------------------------
# A point estimate needs n + dispersion — both REQUIRED
# ---------------------------------------------------------------------------


def test_lever_coverage_requires_n():
    with pytest.raises(ValidationError):
        LeverCoverage(name="l", cells=1, k=1, dispersion="±0", status="measured")


def test_lever_coverage_requires_dispersion():
    with pytest.raises(ValidationError):
        LeverCoverage(name="l", cells=1, k=1, n=1, status="measured")


@pytest.mark.parametrize("missing", ["n", "dispersion"])
def test_a_resolved_evidence_row_requires_its_basis(missing):
    """A resolved number is rendered with its sample size and spread, so neither may be absent."""
    fields: dict[str, Any] = dict(cell_ref="vk:ac", measure_id="elapsed_ms", value=1.0, n=3, dispersion="±1")
    del fields[missing]
    with pytest.raises(ValidationError):
        EvidenceRow(**fields)


# ---------------------------------------------------------------------------
# doc_type discriminators reject the wrong type on both docs
# ---------------------------------------------------------------------------


def test_analysis_check_doc_type_rejects_a_wrong_doc_type():
    payload = _make_analysis().to_dict()
    payload["doc_type"] = "eval_campaign"
    with pytest.raises(ValidationError):
        EvalAnalysis.from_dict(payload)


def test_insight_check_doc_type_rejects_a_wrong_doc_type():
    payload = _make_insight().to_dict()
    payload["doc_type"] = "eval_analysis"
    with pytest.raises(ValidationError):
        EvalInsight.from_dict(payload)


# ---------------------------------------------------------------------------
# WITHIN-doc position checks: every link names a real finding
# ---------------------------------------------------------------------------


class TestPositionsMustResolve:
    """Links inside the document are finding POSITIONS, and ``check_positions`` refuses a bad one.

    Each refusal is asserted with the message naming what is wrong, beside the legal document it
    was made from, so a validator that refused everything would fail the acceptance half.
    """

    def test_a_valid_document_is_accepted(self):
        analysis = _make_analysis(
            findings=[_finding("a"), _finding("b", invalidates=[0]), _finding("c", invalidates=[0, 1])],
            decisions=[_decision(rests_on=[0, 2])],
            questions=[QuestionAnswer(question_id="q", resolution="partial", answer="a", rests_on=[1])],
        )
        assert [finding.invalidates for finding in analysis.document.findings] == [[], [0], [0, 1]]
        assert analysis.document.decisions[0].rests_on == [0, 2]

    @pytest.mark.parametrize("bad", [2, -1], ids=["past-the-end", "negative"])
    def test_a_decision_resting_on_a_missing_position_is_refused(self, bad):
        with pytest.raises(ValidationError, match=r"decisions\[0\]\.rests_on names finding position"):
            _make_analysis(findings=[_finding("a"), _finding("b")], decisions=[_decision(rests_on=[0, bad])])

    def test_a_question_answer_resting_on_a_missing_position_is_refused(self):
        with pytest.raises(ValidationError, match=r"questions\[0\]\.rests_on names finding position\(s\) \[1\]"):
            _make_analysis(
                findings=[_finding("a")],
                decisions=[],
                questions=[QuestionAnswer(question_id="q", resolution="answered", answer="a", rests_on=[1])],
            )

    def test_a_finding_invalidating_a_missing_position_is_refused(self):
        with pytest.raises(ValidationError, match=r"findings\[0\]\.invalidates names finding position\(s\) \[3\]"):
            _make_analysis(findings=[_finding("a", invalidates=[3]), _finding("b")], decisions=[], questions=[])

    def test_a_finding_invalidating_itself_is_refused(self):
        with pytest.raises(ValidationError, match=r"findings\[1\] invalidates itself"):
            _make_analysis(findings=[_finding("a"), _finding("b", invalidates=[1])], decisions=[], questions=[])

    @pytest.mark.parametrize(
        "links",
        [
            pytest.param([[1], [0]], id="two-cycle"),
            pytest.param([[1], [2], [0]], id="three-cycle"),
            pytest.param([[], [2], [1]], id="cycle-off-the-first-finding"),
        ],
    )
    def test_an_invalidates_cycle_is_refused(self, links):
        findings = [_finding(str(index), invalidates=targets) for index, targets in enumerate(links)]
        with pytest.raises(ValidationError, match="findings invalidate each other in a cycle"):
            _make_analysis(findings=findings, decisions=[], questions=[])

    def test_a_diamond_is_not_a_cycle(self):
        """Two paths into one finding is ordinary gating; only a path back to its start is a cycle."""
        findings = [
            _finding("0", invalidates=[1, 2]),
            _finding("1", invalidates=[3]),
            _finding("2", invalidates=[3]),
            _finding("3"),
        ]
        assert len(_make_analysis(findings=findings, decisions=[], questions=[]).document.findings) == 4

    @pytest.mark.parametrize("count", [1, 3])
    def test_a_resolution_count_that_does_not_match_the_findings_is_refused(self, count):
        with pytest.raises(ValidationError, match=f"resolutions has {count} entries for 2 findings"):
            _make_analysis(
                findings=[_finding("a"), _finding("b")],
                decisions=[],
                questions=[],
                resolutions=[FindingResolution() for _ in range(count)],
            )

    def test_no_resolutions_at_all_is_accepted(self):
        """Empty is the unresolved state, not a mismatch — the count is checked only once there are any."""
        document = AuthoredAnalysis(
            headline="h", summary="", findings=[_finding("a")], decisions=[], questions=[], next=[]
        )
        analysis = EvalAnalysis(
            scope_id="uni-1",
            document=document,
            campaign_id="c",
            subject_id="s",
            subject_kind="",
            behavior="b",
            generation=_make_analysis().generation,
            decision_surface=DecisionSurface(),
        )
        assert analysis.resolutions == []

    def test_the_positions_survive_the_from_dict_reload_path(self):
        analysis = _make_analysis()
        reloaded = EvalAnalysis.from_dict(analysis.to_dict())
        assert reloaded.document.decisions[0].rests_on == [0]
        assert reloaded.document.findings[1].invalidates == [0]

    def test_a_stored_document_with_a_bad_position_is_refused_on_load(self):
        stored = _make_analysis().to_dict()
        stored["document"]["decisions"][0]["rests_on"] = [5]
        with pytest.raises(ValidationError, match="names finding position"):
            EvalAnalysis.from_dict(stored)


# ---------------------------------------------------------------------------
# Literal fields reject out-of-set values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Viz(type="pie_chart", ref={}), id="Viz.type"),
        pytest.param(lambda: _decision(disposition="maybe"), id="Decision.disposition"),
        pytest.param(lambda: NextStep(title="e", why="", leverage="huge", lever=""), id="NextStep.leverage"),
        pytest.param(
            lambda: QuestionAnswer(question_id="q", resolution="mostly", answer="", rests_on=[]),
            id="QuestionAnswer.resolution",
        ),
        pytest.param(
            lambda: LeverCoverage(name="l", cells=1, k=1, n=1, dispersion="±0", status="guessed"),
            id="LeverCoverage.status",
        ),
    ],
)
def test_literal_fields_reject_out_of_set_values(build):
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize("retired", ["applied", "recommended"], ids=["applied", "recommended"])
def test_the_retired_execution_flavoured_verdicts_are_gone(retired: str):
    """`applied` and `recommended` were one verdict at two execution states.

    As sibling enum values they made the retraction — we already did it, and the evidence
    says we should not have — unspeakable. They must not survive as a second way to say it.
    """
    with pytest.raises(ValidationError):
        _decision(disposition=retired)


@pytest.mark.parametrize(
    ("model", "kwargs", "field"),
    [
        (
            EvalInsight,
            {
                "scope_id": "uni-1",
                "subject_id": "s",
                "subject_kind": "agent",
                "statement": "st",
                "confidence": "medium",
            },
            "scope",
        ),
        (
            EvalInsight,
            {
                "scope_id": "uni-1",
                "subject_id": "s",
                "subject_kind": "agent",
                "statement": "st",
                "confidence": "medium",
            },
            "invalidation_trigger",
        ),
    ],
)
def test_optional_prose_accepts_null_as_empty(model, kwargs, field):
    """A ``null`` on an optional prose field must not discard a paid generation.

    These fields are already allowed to be empty, and a model asked for an
    optional string returns ``null`` as readily as ``""``.
    """
    built = model(**kwargs, **{field: None})
    assert getattr(built, field) == ""


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (
            lambda: EvalInsight(
                scope_id="uni-1", subject_id="s", subject_kind="agent", statement=None, confidence="medium"
            ),
            "statement",
        ),
        (lambda: _finding(title=None), "title"),
        (lambda: NextStep(title=None, why="", leverage="high", lever=""), "title"),
    ],
    ids=["EvalInsight.statement", "Finding.title", "NextStep.title"],
)
def test_required_prose_still_rejects_null(build, field):
    """The null tolerance is scoped to fields that were already optional.

    Where the text is the claim itself, its absence IS the failure and must surface loudly
    rather than be normalized into an empty string — and the authored contract has no
    nullable slot at all, so a ``null`` there is off-contract wherever it appears.
    """
    with pytest.raises(ValidationError, match=field):
        build()


@pytest.mark.parametrize("field", ["evidence_run_ids", "evidence_result_ids"])
def test_optional_llm_list_accepts_null_as_empty(field):
    """The list sibling of the prose rule — a ``null`` array must not discard a paid generation.

    ``default_factory=list`` covers an ABSENT key, not an explicit ``null``.
    """
    built = EvalInsight(
        scope_id="uni-1", subject_id="s", subject_kind="agent", statement="st", confidence="medium", **{field: None}
    )
    assert getattr(built, field) == []


def test_llm_list_tolerance_did_not_erase_the_element_schema():
    """The null tolerance must not cost the element types it wraps.

    ``_LLMList`` is generic (``_LLMList[str]``) precisely so this holds. A bare ``list``
    would accept the nulls AND silently stop checking the elements.
    """
    insight = EvalInsight(
        scope_id="uni-1",
        subject_id="s",
        subject_kind="agent",
        statement="st",
        confidence="medium",
        evidence_run_ids=["run-a"],
    )
    assert insight.evidence_run_ids == ["run-a"]
    with pytest.raises(ValidationError):
        EvalInsight(
            scope_id="uni-1",
            subject_id="s",
            subject_kind="agent",
            statement="st",
            confidence="medium",
            evidence_run_ids=[{"not": "a string"}],
        )


# ---------------------------------------------------------------------------
# Storage: campaign-scoped analysis retrieval + subject/scope insight filters
# ---------------------------------------------------------------------------


def test_list_analyses_by_campaign_returns_only_that_campaign():
    storage, _ = memory_storage()
    a1 = _make_analysis(campaign_id="camp-1")
    a2 = _make_analysis(campaign_id="camp-1")
    other = _make_analysis(campaign_id="camp-2")
    for analysis in (a1, a2, other):
        storage.save_analysis(analysis)

    ids = {a.id for a in storage.list_analyses_by_campaign("camp-1", "uni-1")}
    assert ids == {a1.id, a2.id}


def test_query_insights_filters_by_subject():
    """Storage owns the ID filters. It no longer owns `scope`, and could not.

    The second half of this test asserted `query_insights(scope='planning')`
    matching by equality. That was the defect, not a contract: `scope` is declared
    free prose and a host stores whole sentences in it, so an exact match is
    not a filter an operator can type — every hand-formed argument returned empty,
    indistinguishable from a truthful one. The filter moved to
    the analysis service's `list_insights` as a case-insensitive SUBSTRING match
    with a refusal, because `by_doc_type` takes only `field_eq` and has no
    substring channel, and its coverage moved with it.
    """
    storage, _ = memory_storage()
    maple_planning = _make_insight(subject_id="ent-maple", scope="planning")
    maple_routing = _make_insight(subject_id="ent-maple", scope="routing")
    bea_planning = _make_insight(subject_id="ent-bea", scope="planning")
    for insight in (maple_planning, maple_routing, bea_planning):
        storage.save_insight(insight)

    by_subject = {i.id for i in storage.query_insights("uni-1", subject_id="ent-maple")}
    assert by_subject == {maple_planning.id, maple_routing.id}


def test_query_insights_filters_by_source_campaign():
    storage, _ = memory_storage()
    from_camp_1 = _make_insight(source_campaign_id="camp-1")
    from_camp_2 = _make_insight(source_campaign_id="camp-2")
    storage.save_insight(from_camp_1)
    storage.save_insight(from_camp_2)

    ids = {i.id for i in storage.query_insights("uni-1", source_campaign_id="camp-1")}
    assert ids == {from_camp_1.id}


def test_a_null_array_is_logged_not_silently_swallowed(caplog):
    """The tolerance must leave a trace, or it converts a loud failure into a quiet lie.

    A normalized null renders identically to a field the generator genuinely had nothing
    for, so without this line an operator reads a malfunctioning paid generation as truthful.
    """
    with caplog.at_level(logging.WARNING):
        EvalInsight(
            scope_id="uni-1",
            subject_id="s",
            subject_kind="agent",
            statement="st",
            confidence="medium",
            evidence_run_ids=None,
        )
    assert "arrived as null" in caplog.text
    assert any(r.levelname == "WARNING" for r in caplog.records)
    # A trace that does not say WHICH array was emptied cannot answer the question it exists
    # for — re-run the billed generation, or accept what it minted?
    assert "'evidence_run_ids'" in caplog.text, "the warning must name the field an operator has to distrust"


def test_the_null_coercion_log_is_pinned_to_one_exact_string(caplog):
    """The operator-facing string must not name a code path — it was wrong three times trying.

    `_coerce_null_list` is a BeforeValidator, so it fires wherever an `_LLMList` field is
    validated, on generate and on load alike. The taxonomy belongs in the comment, which has
    room to be accurate.

    This guard has two independent halves, because neither alone is sufficient:

    1. The emitted line IS the constant with the field name interpolated — this pins the call
       site to the template, so the log cannot drift away from it or drop the field name.
    2. The CONSTANT names no code path and no mechanism. Half 1 cannot check that: it compares
       the constant against the message that same constant produced, so both sides move together
       and a reworded constant passes green.

    Half 2 is itself a blocklist, so be honest about its reach: it covers the wordings that
    shipped and their near neighbours in any case, but a phrasing that avoids every token can
    still assert a path. Reviewing this string when it changes is the rest of the guard.
    """
    with caplog.at_level(logging.WARNING):
        EvalInsight(
            scope_id="uni-1",
            subject_id="s",
            subject_kind="agent",
            statement="st",
            confidence="medium",
            evidence_result_ids=None,
        )

    assert "arrived as null" in caplog.text, "the coercion stopped warning at all"
    # The template is read off the record the coercion emitted, so the two halves below are about
    # the string an operator actually receives rather than a copy of it.
    (record,) = [r for r in caplog.records if "arrived as null" in r.getMessage()]
    template = record.msg
    assert record.args == ("evidence_result_ids",) and record.getMessage() == template % "evidence_result_ids", (
        "the emitted warning is no longer one template with the field name in it — the log call has "
        "drifted from its template, or stopped naming the field an operator must distrust."
    )
    lowered = template.lower()
    for claim in ("path", "nested", "rehydrat", "read time", "_as_list", "validator", "model_validate"):
        assert claim not in lowered, (
            f"the null-list warning names {claim!r}. This string fires on BOTH the generate and load "
            "paths, so any path or mechanism claim in it is false for half its callers."
        )


class TestAResolutionIsTheReadingsItsFindingNames:
    """The tier is read off a resolution's rows, so the rows must be the finding's readings, one for one.

    Each refusal sits beside the accepted document it was made from, so a check that refused every
    resolution would fail the acceptance half.
    """

    @staticmethod
    def _pair(named: list[tuple[str, str]], resolved: list[tuple[str, str]]) -> EvalAnalysis:
        refs = [EvidenceRef(cell="v:a", measure_id=measure, reading=kind) for measure, kind in named]
        rows = [
            EvidenceRow(
                cell_ref="v:a",
                measure_id=measure,
                reading=kind,
                value=1.0,
                n=3,
                dispersion="sd 0.1",
                judged_tier="separation" if kind == "judged" else None,
            )
            for measure, kind in resolved
        ]
        return _make_analysis(findings=[_finding("a", evidence=refs)], resolutions=[FindingResolution(evidence=rows)])

    def test_the_readings_resolved_as_named_are_accepted(self):
        named = [("latency", "measure"), ("helpful", "judged")]
        assert self._pair(named, named).resolutions[0].evidence_tier == "separation"

    def test_a_judged_reading_resolved_as_a_measure_is_refused(self):
        # The tier would read mechanical over a finding that leans on a judge.
        with pytest.raises(ValidationError, match=r"resolutions\[0\]\.evidence resolves"):
            self._pair([("helpful", "judged")], [("helpful", "measure")])

    def test_a_named_reading_left_unresolved_is_refused(self):
        with pytest.raises(ValidationError, match=r"resolutions\[0\]\.evidence resolves"):
            self._pair([("latency", "measure"), ("helpful", "judged")], [("latency", "measure")])

    def test_a_resolved_row_no_reading_names_is_refused(self):
        with pytest.raises(ValidationError, match=r"resolutions\[0\]\.evidence resolves"):
            self._pair([], [("latency", "measure")])


@pytest.mark.parametrize("level", ["top", "nested"])
def test_a_stored_analysis_carrying_an_unknown_key_is_refused_at_either_level(level: str):
    """A stored `EvalAnalysis` is read as strictly as it is constructed, top-level and nested.

    The nested key rides on a `FindingResolution`, whose dump also echoes its computed
    `evidence_tier` — the one key that must NOT be refused, since it is the model's own output.
    """
    stored = _make_analysis().to_dict()
    assert "evidence_tier" in stored["resolutions"][0], "the fixture must carry the computed echo"
    assert EvalAnalysis.from_dict(stored).id == stored["id"], "the echo alone must still load"
    if level == "top":
        stored["a_retired_top_level_field"] = "old"
    else:
        stored["resolutions"][0]["a_retired_nested_field"] = 1

    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvalAnalysis.from_dict(stored)
