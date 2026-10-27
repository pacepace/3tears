"""EvalAnalysis / EvalInsight — model shape, validation, and storage round-trips.

The generated, stored analysis of a campaign plus the durable insights it mints. These tests pin
the stored shape — the authored document kept verbatim beside the per-finding resolutions code
filled — every confidence a tier, never a typed probability, the
"no point estimate without n + dispersion" rule, the doc_type discriminators, the
WITHIN-doc position checks (every link names a real finding, nothing invalidates itself or cycles,
one resolution per finding), and campaign-scoped / subject-scoped retrieval.

Storage is exercised through a real :class:`~threetears.evals.kernel.storage.EvalStorage`
over the same in-memory evals-repo stand-in the campaign tests use.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.kernel.authored import (
    AspectSummary,
    AuthoredCampaignAnalysis,
    Caveat,
    EvidenceRef,
    Finding,
    FloorOverride,
    NextStep,
    Recommendation,
)
from threetears.evals.kernel.campaign import (
    CampaignDeclaration,
    CoverageLens,
    EvalAnalysis,
    EvalInsight,
    EvidenceRow,
    FindingResolution,
    GenerationProvenance,
    LeverCoverage,
    RunIndexEntry,
)
from threetears.evals.analysis.analyst.snapshot import report_snapshot
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.declaration import Aspect
from threetears.evals.kernel.storage import DroppedJudgeProfile
from threetears.evals.schema.versioning import REGENERABLE_SCHEMA_VERSION

# Reuse the campaign tests' in-memory evals-repo stand-in + storage wiring.
from packages.evals.tests.factories import (
    CURRENT_BUNDLE_SHAPE,
    analysed_aspect,
    empty_bundle,
    make_campaign,
    memory_storage,
)
from packages.evals.tests.fixtures.report_plans.misfits import reference_data


def _finding(title: str = "v", *, invalidates: list[int] | None = None, **overrides: Any) -> Finding:
    """An authored finding with every list empty unless a caller fills it."""
    fields: dict[str, Any] = dict(
        title=title,
        body="",
        confidence="medium",
        axes=[],
        evidence=[],
        trials=[],
        caveats=[],
        invalidates=invalidates or [],
        durable="",
    )
    fields.update(overrides)
    return Finding(**fields)


def _recommendation(**overrides: Any) -> Recommendation:
    """An authored recommendation of the control, resting on nothing unless a caller says otherwise."""
    fields: dict[str, Any] = dict(
        arm="control", reasons=[], confidence="medium", rests_on=[], floors_overridden=[], behind_on=[], bars_missed=[]
    )
    fields.update(overrides)
    return Recommendation(**fields)


def _summary(aspect: str = "planning", **overrides: Any) -> AspectSummary:
    """One aspect's summary."""
    fields: dict[str, Any] = dict(aspect=aspect, what_changed="It got faster.", how_sure="Sure.", means="Switch.")
    fields.update(overrides)
    return AspectSummary(**fields)


def _make_analysis(**overrides: Any) -> EvalAnalysis:
    """Build an ``EvalAnalysis`` with EVERY nested object populated.

    Two findings (the second invalidating the first, with evidence, the trials it is drawn from, a caveat and a
    durable claim), a recommendation resting on a real position and made over a broken floor, an aspect summary, a
    next step, one resolution per finding carrying a resolved evidence row, one aspect carrying a coverage lever (with
    the required n + dispersion), a run-index row and the frozen bundle, a generation-provenance block and the
    campaign's declaration — so a round-trip has something in every branch.

    ``findings``, ``recommendation``, ``aspect_summaries`` and ``next`` fold into the document; ``resolutions``
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
            trials=["T1", "T4"],
            caveats=[Caveat(kind="sampling", text="k=1 on this template is noise-dominated.")],
            durable="Alpha-2 is the planner model of record for Maple.",
        ),
        _finding("Latency is output-bound, not input-bound.", axes=["output_max_tokens"], invalidates=[0]),
    ]
    authored_findings = overrides.pop("findings", None)
    findings = list(authored_findings or default_findings)
    document = AuthoredCampaignAnalysis(
        findings=findings,
        recommendation=[
            overrides.pop(
                "recommendation",
                _recommendation(
                    arm="alpha-2",
                    reasons=["Much faster at equal quality."],
                    rests_on=[0],
                    floors_overridden=[
                        FloorOverride(floor="planning/cost_lower/*", why="Cost is not this team's concern.")
                    ],
                ),
            )
        ],
        aspect_summaries=overrides.pop("aspect_summaries", [_summary()]),
        summary=overrides.pop("summary", "Alpha-2 is much faster at equal quality, so it is the planner to run."),
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
                        aspect="planning",
                        cell_ref="vk-alpha:ac-1",
                        measure_id="elapsed_ms",
                        reading="measure",
                        value=41000.0,
                        n=3,
                        dispersion="±900",
                    )
                ],
            ),
            FindingResolution(),
        ]
    else:
        resolutions = [FindingResolution() for _ in findings]
    aspect = analysed_aspect(
        empty_bundle(campaign_id="camp-1", aspect="planning"),
        coverage=CoverageLens(
            levers=[LeverCoverage(name="planner.model", cells=5, k=3, n=15, dispersion="±0.10", status="measured")],
        ),
        run_index=[RunIndexEntry(run_id="run-a", config={"model": "alpha-2"}, key_metrics={"latency_s": 42.0})],
    )
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
            bundle_schema_version=CURRENT_BUNDLE_SHAPE,
            host_declarations_digest="sha256:declarations",
        ),
        design_snapshot=CampaignDeclaration(
            question="Which planner model?",
            declared_at="2026-01-01T00:00:00+00:00",
            declared_by="analyst",
            arms=[],
            aspects=[Aspect(label="planning")],
            floors=[],
        ),
        recommended_variant_key=(
            None if document.recommended() is None else f"variant-of-{document.recommendation[0].arm}"
        ),
        aspects=[aspect],
        campaign_bundle={
            "schema_version": 1,
            "aspects": [{"aspect": "planning", "fingerprint": aspect.bundle_fingerprint}],
        },
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
    assert reloaded.resolutions[0].evidence[0].value == 41000.0
    assert reloaded.document.findings[0].trials == ["T1", "T4"]
    assert reloaded.aspects[0].bundle == analysis.aspects[0].bundle
    assert reloaded.aspects[0].bundle["campaign_id"] == "camp-1"
    assert reloaded.document.findings[0].caveats == [
        Caveat(kind="sampling", text="k=1 on this template is noise-dominated.")
    ]
    assert reloaded.document.findings[1].invalidates == [0]
    assert reloaded.document.recommendation[0].rests_on == [0]
    assert reloaded.document.recommendation[0].floors_overridden[0].floor == "planning/cost_lower/*"
    assert reloaded.document.aspect_summaries[0].aspect == "planning"
    assert reloaded.document.next[0].leverage == "high"
    assert reloaded.resolutions[0].evidence[0].aspect == "planning"
    assert reloaded.aspects[0].coverage.levers[0].n == 15
    assert reloaded.aspects[0].coverage.levers[0].dispersion == "±0.10"
    assert reloaded.generation.token_cost == 0.42
    assert reloaded.aspects[0].run_index[0].key_metrics == {"latency_s": 42.0}
    assert reloaded.design_snapshot.question == "Which planner model?"


def test_analysis_storage_round_trip_preserves_every_field():
    storage, _ = memory_storage()
    analysis = _make_analysis()

    storage.save_analysis(analysis)
    loaded = storage.load_analysis(analysis.id, analysis.scope_id)

    assert loaded is not None
    assert loaded == analysis
    assert loaded.resolutions == analysis.resolutions
    assert loaded.aspects == analysis.aspects
    assert loaded.campaign_bundle == analysis.campaign_bundle
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
        pytest.param(lambda c: _recommendation(confidence=c), id="Recommendation"),
        pytest.param(
            lambda c: EvalInsight(
                scope_id="uni-1",
                subject_kind="",
                subject_id="s",
                statement="st",
                confidence=c,
                invalidation_trigger="retired when its analysis is archived, or replaced by a later analysis restating it",
            ),
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

    from threetears.evals.kernel import authored, campaign

    assert campaign.ConfidenceTier is authored.Confidence
    assert campaign.CONFIDENCE_TIERS == get_args(authored.Confidence) == ("very_high", "high", "medium", "low")


# ---------------------------------------------------------------------------
# A lever's coverage carries no confidence
# ---------------------------------------------------------------------------


def test_a_lever_carries_no_confidence_looked_up_from_its_status():
    """A lever carries no confidence: how firmly a reading stands is its evidence tier, so the key is refused."""
    assert "confidence" not in LeverCoverage.model_fields
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LeverCoverage(name="l", cells=1, k=1, n=1, dispersion="±0", status="unswept", confidence="low")


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
    fields: dict[str, Any] = dict(
        aspect="planning", cell_ref="vk:ac", measure_id="elapsed_ms", value=1.0, n=3, dispersion="±1"
    )
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
            recommendation=_recommendation(rests_on=[0, 2]),
        )
        assert [finding.invalidates for finding in analysis.document.findings] == [[], [0], [0, 1]]
        assert analysis.document.recommendation[0].rests_on == [0, 2]

    @pytest.mark.parametrize("bad", [2, -1], ids=["past-the-end", "negative"])
    def test_a_recommendation_resting_on_a_missing_position_is_refused(self, bad):
        with pytest.raises(ValidationError, match=r"recommendation\.rests_on names finding position"):
            _make_analysis(findings=[_finding("a"), _finding("b")], recommendation=_recommendation(rests_on=[0, bad]))

    def test_a_finding_invalidating_a_missing_position_is_refused(self):
        with pytest.raises(ValidationError, match=r"findings\[0\]\.invalidates names finding position\(s\) \[3\]"):
            _make_analysis(findings=[_finding("a", invalidates=[3]), _finding("b")])

    def test_a_finding_invalidating_itself_is_refused(self):
        with pytest.raises(ValidationError, match=r"findings\[1\] invalidates itself"):
            _make_analysis(findings=[_finding("a"), _finding("b", invalidates=[1])])

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
            _make_analysis(findings=findings)

    def test_a_diamond_is_not_a_cycle(self):
        """Two paths into one finding is ordinary gating; only a path back to its start is a cycle."""
        findings = [
            _finding("0", invalidates=[1, 2]),
            _finding("1", invalidates=[3]),
            _finding("2", invalidates=[3]),
            _finding("3"),
        ]
        assert len(_make_analysis(findings=findings).document.findings) == 4

    @pytest.mark.parametrize("count", [1, 3])
    def test_a_resolution_count_that_does_not_match_the_findings_is_refused(self, count):
        with pytest.raises(ValidationError, match=f"resolutions has {count} entries for 2 findings"):
            _make_analysis(
                findings=[_finding("a"), _finding("b")],
                resolutions=[FindingResolution() for _ in range(count)],
            )

    def test_no_resolutions_at_all_is_accepted(self):
        """Empty is the unresolved state, not a mismatch — the count is checked only once there are any."""
        assert _make_analysis(findings=[_finding("a")], resolutions=[]).resolutions == []

    def test_the_positions_survive_the_from_dict_reload_path(self):
        analysis = _make_analysis()
        reloaded = EvalAnalysis.from_dict(analysis.to_dict())
        assert reloaded.document.recommendation[0].rests_on == [0]
        assert reloaded.document.findings[1].invalidates == [0]

    def test_a_stored_document_with_a_bad_position_is_refused_on_load(self):
        stored = _make_analysis().to_dict()
        stored["document"]["recommendation"][0]["rests_on"] = [5]
        with pytest.raises(ValidationError, match="names finding position"):
            EvalAnalysis.from_dict(stored)


class TestAspectsAreOneEach:
    """An analysis covers its campaign's aspects once each, and every row and summary names one of them."""

    def test_a_resolved_row_on_an_aspect_the_analysis_lacks_is_refused(self):
        row = EvidenceRow(
            aspect="routing", cell_ref="vk:ac", measure_id="m", reading="measure", value=1.0, n=1, dispersion="x"
        )
        finding = _finding("a", evidence=[EvidenceRef(cell="vk:ac", measure_id="m", reading="measure")])
        with pytest.raises(ValidationError, match=r"names aspect\(s\) \['routing'\], which this analysis lacks"):
            _make_analysis(findings=[finding], resolutions=[FindingResolution(evidence=[row])])

    def test_a_summary_of_an_aspect_the_analysis_lacks_is_refused(self):
        with pytest.raises(ValidationError, match=r"aspect_summaries name \['routing'\]"):
            _make_analysis(aspect_summaries=[_summary("routing")])

    def test_an_aspect_summarised_twice_is_refused(self):
        with pytest.raises(ValidationError, match="summarise \\['planning'\\] more than once"):
            _make_analysis(aspect_summaries=[_summary(), _summary()])

    def test_an_aspect_held_twice_is_refused(self):
        aspect = _make_analysis().aspects[0]
        with pytest.raises(ValidationError, match="appear more than once"):
            _make_analysis(aspects=[aspect, aspect])

    def test_an_analysis_covers_at_least_one_aspect(self):
        with pytest.raises(ValidationError, match="aspects"):
            _make_analysis(aspects=[])

    def test_an_aspect_is_found_by_its_label(self):
        analysis = _make_analysis()
        assert analysis.aspect("planning") is analysis.aspects[0]
        with pytest.raises(KeyError, match="no aspect 'routing'"):
            analysis.aspect("routing")


# ---------------------------------------------------------------------------
# Literal fields reject out-of-set values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: NextStep(title="e", why="", leverage="huge", lever=""), id="NextStep.leverage"),
        pytest.param(
            lambda: LeverCoverage(name="l", cells=1, k=1, n=1, dispersion="±0", status="guessed"),
            id="LeverCoverage.status",
        ),
    ],
)
def test_literal_fields_reject_out_of_set_values(build):
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (
            lambda: EvalInsight(
                scope_id="uni-1",
                subject_id="s",
                subject_kind="agent",
                statement=None,
                confidence="medium",
                invalidation_trigger="retired when its analysis is archived, or replaced by a later analysis restating it",
            ),
            "statement",
        ),
        (lambda: _finding(title=None), "title"),
        (lambda: NextStep(title=None, why="", leverage="high", lever=""), "title"),
    ],
    ids=["EvalInsight.statement", "Finding.title", "NextStep.title"],
)
def test_required_prose_still_rejects_null(build, field):
    """A required prose field refuses null.

    Where the text is the claim itself, its absence IS the failure and must surface loudly
    rather than be normalized into an empty string — and the authored contract has no
    nullable slot at all, so a ``null`` there is off-contract wherever it appears.
    """
    with pytest.raises(ValidationError, match=field):
        build()


def test_an_insights_evidence_lists_check_their_elements():
    """An insight's evidence id lists hold strings, and refuse anything else."""
    insight = EvalInsight(
        scope_id="uni-1",
        subject_id="s",
        subject_kind="agent",
        statement="st",
        confidence="medium",
        evidence_run_ids=["run-a"],
        invalidation_trigger="retired when its analysis is archived, or replaced by a later analysis restating it",
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
            invalidation_trigger="retired when its analysis is archived, or replaced by a later analysis restating it",
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
                aspect="planning",
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


def test_drop_derived_clears_analyses_insights_and_report_snapshots_and_keeps_the_campaign():
    """The operator step after a regenerable bump: every derived document goes, every kept and durable one stays.

    A report snapshot is derived: it freezes a report input built from evidence that stays, so it is frozen again.
    """
    storage, _ = memory_storage()
    analysis, insight = _make_analysis(), _make_insight()
    storage.save_analysis(analysis)
    storage.save_insight(insight)
    snapshot = report_snapshot(
        ReportInput.model_validate(reference_data("r1")),
        scope_id="uni-1",
        campaign_id=None,
        source="report_input",
        created_by="agent:test",
    )
    storage.save_report_snapshot(snapshot)
    campaign = make_campaign(scope_id="uni-1")
    storage.save_campaign(campaign)

    dropped = storage.drop_derived(["uni-1"])

    assert dropped.counts == {
        "eval_analysis": 1,
        "eval_insight": 1,
        "eval_judge_profile": 0,
        "eval_report_doc": 0,
        "eval_report_plan": 0,
        "eval_report_snapshot": 1,
    }
    assert dropped.judge_profiles == []
    assert storage.load_analysis(analysis.id, "uni-1") is None
    assert storage.query_insights("uni-1") == []
    assert storage.load_report_snapshot(snapshot.id, "uni-1") is None
    assert storage.load_campaign(campaign.id, "uni-1") is not None


def test_drop_derived_returns_each_dropped_judge_profile_s_runs_to_record_it_again_from():
    """A judge profile is recorded only by an operator, so the drop hands back the runs it was recorded from."""
    storage, store = memory_storage()
    old = {
        "id": "jp-1",
        "doc_type": "eval_judge_profile",
        "scope_id": "uni-1",
        "schema_version": REGENERABLE_SCHEMA_VERSION - 1,
        "run_ids": ["judge-run-a", "judge-run-b"],
    }
    store.upsert(old)

    dropped = storage.drop_derived(["uni-1"])

    assert dropped.counts["eval_judge_profile"] == 1
    assert dropped.judge_profiles == [
        DroppedJudgeProfile(scope_id="uni-1", profile_id="jp-1", run_ids=["judge-run-a", "judge-run-b"])
    ]
    assert store.get("jp-1", "uni-1") is None


def test_an_insight_from_before_the_bump_refuses_the_ledger_read_naming_the_operator_step():
    """Assembly reads the subject's ledger through this read, so one old insight refuses every generation over it."""
    storage, store = memory_storage()
    insight = _make_insight()
    store.upsert({**insight.to_dict(), "schema_version": REGENERABLE_SCHEMA_VERSION - 1})

    with pytest.raises(ValidationError, match=r"EvalStorage\.drop_derived\(scopes\)") as refused:
        storage.query_insights("uni-1", subject_id=insight.subject_id)
    assert "judge_profiles_record over that profile's run_ids" in str(refused.value)
