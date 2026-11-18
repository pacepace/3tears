"""#1187: a campaign cell holds one definition of each host measure and one answering model, and a coordinate nothing
recorded matches nothing.

* What a host measure counts is the instrument's, so it is a dimension of the rig: runs read under two definitions
  share no cell and no contrast, and a run whose results hold a host measure and that recorded no definition is
  pooled and compared with nothing, until its host states the definitions (``runs_attest``).
* Two runs whose candidate responses each named no model share no cell: two silences are no match. A run that names
  what answered it (a provider's response, a scripted client's own name, a recording) pools as it always did, and a
  host states the model that answered a silent run (``runs_attest``).
* The cell model's version moved once for it, and the frontier's sentence says what a campaign's cells keep apart.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

from threetears.evals.actions import Caller, eval_catalogue, standard_tools
from threetears.evals.analysis import (
    CELL_MODEL_VERSION,
    AnalysisContextBundle,
    BundleInspection,
    Cell,
    assemble_context_bundle,
    build_code_only_report,
    campaign_report,
)
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown
from threetears.evals.analysis.report.serialize_md import report_markdown
from threetears.evals.analysis.lenses.frontier import PASS_HAT_K_CELL_READING
from threetears.evals.kernel import ControlDeclaration, EvalStorage, SweptAxis, resolve_variant_identity
from threetears.evals.kernel.host import MeasureRegistry
from threetears.evals.ops import OpsHost
from threetears.evals import Log
from threetears.evals.run import attest_measure_definitions
from threetears.evals.run.remeasure import attest_served_model
from threetears.evals.schema import EvalResult, EvalRun, LatencyMetrics, SweepableValue
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_campaign
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.kind import FIELD_ACCURACY
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.v15_shapes import V15Design

#: The two models a launch asks for by name: the control's and the candidate arm's.
_CONTROL = "vendor/extractor-pinned"
_CANDIDATE = "vendor/extractor-next"

#: The definitions the toy host declares today, and the same measures as the host defined them before a change.
_NOW = toyhost_profile().measures.definition_digests()
_BEFORE = {name: "0" * 64 for name in _NOW}

#: The rig dimension a class is split on by its host measures' definitions, and the cell's served-model one.
_DEFINITIONS = "measure_definitions"
_SERVED = "served_model:candidate"

_CAMPAIGN_ID = "5d3f9c1e-7a42-4b6e-8f10-2c9a4e6b1d37"


def _batch(model: str, salt: str, *, definitions: dict[str, str] | None) -> EvalRun:
    """One toy batch asking for ``model``, recording ``definitions`` (None: stored before runs recorded any)."""
    batch = toyhost_batch(
        chunk_tokens=512,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
    )
    return batch.model_copy(
        update={
            "id": str(uuid.uuid5(uuid.UUID(batch.id), salt)),
            "candidate_model": model,
            "measure_definitions": definitions,
        }
    )


def _results(batch: EvalRun, served: str | None, *, host_measures: bool = True) -> list[EvalResult]:
    """The batch's toy observations, each candidate row naming ``served`` as what answered it."""
    return [
        result.model_copy(
            update={
                "usage": [
                    row.model_copy(update={"served_model": served}) if row.role == "candidate" else row
                    for row in result.usage
                ],
                **({} if host_measures else {"host_measures": {}}),
            }
        )
        for result in toyhost_measurements(
            batch, profile=toyhost_profile(), cost_usd=0.02, total_ms=900.0, field_accuracy=0.8
        )
    ]


def _design(control: EvalRun) -> V15Design:
    return V15Design(
        axes=[
            SweptAxis(
                axis_id="model",
                values=[SweepableValue.of(model, display=model) for model in (_CONTROL, _CANDIDATE)],
                rationale="which extractor model",
            )
        ],
        control=resolve_variant_identity(run=control, profile=toyhost_profile()).variant_key,
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="witnessed"),
        declared_at="2026-03-14T09:30:00+00:00",
    )


def _stored(batches: Sequence[tuple[EvalRun, list[EvalResult]]]) -> EvalStorage:
    """A real store holding the batches, their results and one campaign over them, the first batch its control."""

    def write() -> EvalStorage:
        storage = EvalStorage(InMemoryDocumentStore())
        for batch, results in batches:
            storage.save_eval_run(batch)
            for result in results:
                storage.save_eval_result(result)
        storage.save_campaign(
            make_campaign(
                id=_CAMPAIGN_ID,
                scope_id=TOYHOST_SCOPE,
                name="cell identity",
                subject_id=TOYHOST_SUBJECT.subject_id,
                subject_kind="extractor_config",
                behavior="extract_invoice_fields",
                template_id="",
                run_ids=[batch.id for batch, _ in batches],
                declared_design=_design(batches[0][0]),
                created_by="test:fixture",
            )
        )
        return storage

    return off_loop(write)


def _bundle(storage: EvalStorage) -> AnalysisContextBundle:
    def read() -> AnalysisContextBundle:
        campaign = storage.load_campaign(_CAMPAIGN_ID, TOYHOST_SCOPE)
        assert campaign is not None
        return assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())

    return off_loop(read)


def _arm(bundle: AnalysisContextBundle, *, control: bool) -> list[Cell]:
    """The control's cells, or every other arm's, in the bundle's order."""
    assert bundle.design.control_arm is not None
    key = bundle.design.control_arm.variant_key
    return [cell for cell in bundle.cells if (cell.variant_key == key) is control]


def _contrasted(bundle: AnalysisContextBundle) -> set[str]:
    """The apparatus class of every cell some comparison reads against the control."""
    return {
        comparison.contrast.apparatus_class_id
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
    }


def _named(model: str, salt: str, definitions: dict[str, str] | None) -> tuple[EvalRun, list[EvalResult]]:
    """A batch whose every response named the model it asked for."""
    batch = _batch(model, salt, definitions=definitions)
    return batch, _results(batch, model)


# =============================================================================
# (a) One definition of each host measure, or no cell and no contrast
# =============================================================================


def test_runs_recorded_under_one_definition_pool_and_are_compared_as_they_always_were() -> None:
    bundle = _bundle(
        _stored([_named(_CONTROL, "control", _NOW), _named(_CANDIDATE, "a", _NOW), _named(_CANDIDATE, "b", _NOW)])
    )
    (control,), (candidate,) = _arm(bundle, control=True), _arm(bundle, control=False)
    assert candidate.n_observations == 2 * control.n_observations, "two runs of one arm are one cell"
    assert control.apparatus_class_id == candidate.apparatus_class_id, "under one rig"
    assert _DEFINITIONS not in [*control.unknown_dimensions, *candidate.unknown_dimensions]
    assert _contrasted(bundle) == {candidate.apparatus_class_id}
    assert bundle.refused_merges == []


def test_two_runs_of_one_arm_read_under_two_definitions_never_share_a_cell_or_a_contrast() -> None:
    old, new = _named(_CANDIDATE, "old", _BEFORE), _named(_CANDIDATE, "new", _NOW)
    bundle = _bundle(_stored([_named(_CONTROL, "control", _NOW), old, new]))
    (control,) = _arm(bundle, control=True)
    cells = _arm(bundle, control=False)
    assert len(cells) == 2 and {cell.n_observations for cell in cells} == {control.n_observations}
    by_run = {tuple(cell.observation_ids): cell for cell in cells}
    current = by_run[tuple(sorted(result.id for result in new[1]))]
    stale = by_run[tuple(sorted(result.id for result in old[1]))]
    assert current.apparatus_class_id == control.apparatus_class_id, "the run read as the control was shares its rig"
    assert stale.apparatus_class_id != control.apparatus_class_id
    assert _contrasted(bundle) == {current.apparatus_class_id}, (
        "the run read under the old definition is in no contrast"
    )
    (refused,) = bundle.refused_merges
    assert (refused.reason, refused.dimensions) == ("apparatus_differs", [_DEFINITIONS])


def test_two_arms_read_under_two_definitions_are_compared_with_nothing() -> None:
    bundle = _bundle(_stored([_named(_CONTROL, "control", _BEFORE), _named(_CANDIDATE, "candidate", _NOW)]))
    (control,), (candidate,) = _arm(bundle, control=True), _arm(bundle, control=False)
    assert control.apparatus_class_id != candidate.apparatus_class_id
    assert _contrasted(bundle) == set()


def test_a_run_holding_host_measures_that_recorded_no_definition_stands_alone_and_the_page_names_the_remedy() -> None:
    """The stored-before-the-feature shape: every run unrecorded. None pools or is compared with another."""
    bundle = _bundle(
        _stored([_named(_CONTROL, "control", None), _named(_CANDIDATE, "a", None), _named(_CANDIDATE, "b", None)])
    )
    (control,) = _arm(bundle, control=True)
    cells = _arm(bundle, control=False)
    assert len(cells) == 2, "two unrecorded runs of one arm are two cells: two silences are no match"
    assert len({control.apparatus_class_id, *(cell.apparatus_class_id for cell in cells)}) == 3
    assert all(_DEFINITIONS in cell.unknown_dimensions for cell in (control, *cells))
    assert _contrasted(bundle) == set(), "an unrecorded definition is contrasted with nothing"
    (refused,) = bundle.refused_merges
    assert (refused.reason, refused.dimensions) == ("apparatus_unknown", [_DEFINITIONS])
    (experiment,) = [entry for entry in bundle.next_experiments if entry.dimension == _DEFINITIONS]
    assert experiment.n_observations_if_recorded == 2 * experiment.n_observations_now

    page = report_markdown(build_code_only_report(bundle, assembled_at="2026-11-15T00:00:00+00:00"))
    assert "hold host measures and recorded no measure definitions" in page
    assert "runs_attest, coordinate measure_definition" in page and "run_remeasure" in page


def test_an_unrecorded_run_holding_no_host_measure_pools_exactly_as_before() -> None:
    def bare(model: str, salt: str) -> tuple[EvalRun, list[EvalResult]]:
        batch = _batch(model, salt, definitions=None)
        return batch, _results(batch, model, host_measures=False)

    bundle = _bundle(_stored([bare(_CONTROL, "control"), bare(_CANDIDATE, "a"), bare(_CANDIDATE, "b")]))
    (control,), (candidate,) = _arm(bundle, control=True), _arm(bundle, control=False)
    assert candidate.n_observations == 2 * control.n_observations
    assert _contrasted(bundle) == {candidate.apparatus_class_id}


async def test_the_hosts_statement_of_the_definitions_brings_the_runs_back_into_one_cell_and_its_contrast() -> None:
    batches = [_named(_CONTROL, "control", None), _named(_CANDIDATE, "a", None), _named(_CANDIDATE, "b", None)]
    storage = _stored(batches)
    assert _contrasted(_bundle(storage)) == set()

    stated = await attest_measure_definitions(
        toyhost_host(storage=storage),
        [batch.id for batch, _ in batches],
        TOYHOST_SCOPE,
        _NOW,
        attested_by="person:host",
    )

    assert stated.run_ids == [batch.id for batch, _ in batches], "one call takes every run"
    assert (stated.coordinate, stated.stated) == ("measure_definition", _NOW)
    bundle = _bundle(storage)
    (control,), (candidate,) = _arm(bundle, control=True), _arm(bundle, control=False)
    assert candidate.n_observations == 2 * control.n_observations
    assert _contrasted(bundle) == {candidate.apparatus_class_id}
    assert _DEFINITIONS not in candidate.unknown_dimensions and bundle.refused_merges == []
    attested = {summary.run_id: summary.attested_coordinates for summary in bundle.run_summaries}
    assert all(attested[batch.id] == ["measure_definition"] for batch, _ in batches), "the bundle carries the fact"
    page = report_markdown(build_code_only_report(bundle, assembled_at="2026-11-15T00:00:00+00:00"))
    assert "Attested, not recorded: the host stated after the fact the definitions of its measures 3 run(s)" in page


# =============================================================================
# (b) One answering model, and a silence matches nothing
# =============================================================================


def _silent(salt: str) -> tuple[EvalRun, list[EvalResult]]:
    """A candidate batch none of whose responses named a model."""
    batch = _batch(_CANDIDATE, salt, definitions=_NOW)
    return batch, _results(batch, None)


def test_two_runs_whose_responses_named_no_model_do_not_pool() -> None:
    bundle = _bundle(_stored([_named(_CONTROL, "control", _NOW), _silent("a"), _silent("b")]))
    (control,) = _arm(bundle, control=True)
    cells = _arm(bundle, control=False)
    assert len(cells) == 2 and {cell.n_observations for cell in cells} == {control.n_observations}
    assert {cell.rig_class_id for cell in cells} == {control.apparatus_class_id}, "each is still read under the rig"
    assert all(cell.unknown_dimensions == [_SERVED] for cell in cells)
    (refused,) = bundle.refused_merges
    assert (refused.reason, refused.dimensions) == ("apparatus_unknown", [_SERVED])
    assert _contrasted(bundle) == {cell.apparatus_class_id for cell in cells}, "each run is a contrast of its own"


def test_one_silent_run_alone_in_its_arm_is_that_arms_one_cell_under_the_rig_itself() -> None:
    bundle = _bundle(_stored([_named(_CONTROL, "control", _NOW), _silent("only")]))
    (control,), (candidate,) = _arm(bundle, control=True), _arm(bundle, control=False)
    assert candidate.apparatus_class_id == control.apparatus_class_id and candidate.rig_class_id is None


def test_runs_that_recorded_what_answered_them_keep_pooling_whatever_it_was() -> None:
    """A scripted client names its script and a replay its recording: either is a recorded answer, so the runs pool."""
    for answered in ("scripted:toy-extractor", "recording:0123456789ab"):
        batches = [_named(_CONTROL, "control", _NOW)]
        for salt in ("a", "b"):
            batch = _batch(_CANDIDATE, salt, definitions=_NOW)
            batches.append((batch, _results(batch, answered)))
        bundle = _bundle(_stored(batches))
        (control,), (candidate,) = _arm(bundle, control=True), _arm(bundle, control=False)
        assert candidate.n_observations == 2 * control.n_observations, answered
        assert candidate.rig_class_id is None and bundle.refused_merges == []


def test_the_campaign_page_says_silent_runs_are_read_apart_and_names_the_remedy() -> None:
    storage = _stored([_named(_CONTROL, "control", _NOW), _silent("a"), _silent("b")])
    page = campaign_report_markdown(
        off_loop(campaign_report, toyhost_host(storage=storage), _CAMPAIGN_ID, TOYHOST_SCOPE)
    )
    assert "named no model in more than one run" in page
    assert "runs_attest, coordinate served_model" in page


async def test_the_hosts_statement_of_the_served_model_is_recorded_as_one_and_the_runs_pool() -> None:
    first, second = _silent("a"), _silent("b")
    storage = _stored([_named(_CONTROL, "control", _NOW), first, second])
    as_recorded = {
        run_id: [result.to_dict() for result in off_loop(storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)]
        for run_id in (first[0].id, second[0].id)
    }
    launch, _client = toyhost_launch_host(storage=storage)
    host = OpsHost(launch=launch)
    (tool, *_) = eval_catalogue().mount_all(standard_tools())
    caller = Caller(scope_id=TOYHOST_SCOPE, identity="agent:backfill")
    ask = {"action": "runs_attest", "coordinate": "served_model", "run_ids": [first[0].id, second[0].id]}

    stated = await tool.call({**ask, "served_model": "vendor/extractor-next-2026-08"}, host=host, caller=caller)

    assert not stated.is_error and stated.structured is not None, stated.text
    assert stated.structured["run_ids"] == [first[0].id, second[0].id], "one call takes many runs"
    assert stated.structured["coordinate"] == "served_model"
    assert stated.structured["stated"] == {"served_model": "vendor/extractor-next-2026-08"}
    for run_id, recorded in as_recorded.items():
        now = [result.to_dict() for result in off_loop(storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)]
        assert now == recorded, "evidence stays as recorded: no row is rewritten, a silence stays a silence"
    run = off_loop(storage.load_eval_run, first[0].id, TOYHOST_SCOPE)
    assert run is not None and run.served_model_attestation is not None
    assert run.served_model_attestation.attested_by == "agent:backfill", "stated, and by whom: never a response's word"
    assert run.served_model_attestation.served_model == "vendor/extractor-next-2026-08"
    assert run.served_model_attestation.result_ids == sorted(result.id for result in first[1]), "what it covers"
    bundle = _bundle(storage)
    (control,), (candidate,) = _arm(bundle, control=True), _arm(bundle, control=False)
    assert candidate.n_observations == 2 * control.n_observations and candidate.rig_class_id is None, (
        "the cells read the silent rows through the statement, and the runs pool"
    )
    # The bundle carries the typed fact, per arm and per run, so a caveat or a first-screen value can rank it.
    (arm,) = [reading for reading in bundle.arm_served_models if reading.variant_key == candidate.variant_key]
    assert (arm.state, arm.n_unrecorded) == ("unrecorded", 2 * control.n_observations), "what was recorded"
    assert (arm.n_attested, arm.attested_models) == (2 * control.n_observations, ["vendor/extractor-next-2026-08"])
    attested = {summary.run_id: summary.attested_coordinates for summary in bundle.run_summaries}
    assert attested[first[0].id] == ["served_model"] and attested[second[0].id] == ["served_model"]
    # The identity disclosure stays true, and says the model was stated, not recorded.
    said = bundle.model_identity_disclosure or ""
    assert "named no model" in said, said
    assert "Attested, not recorded" in said and "vendor/extractor-next-2026-08 of 2 run(s)" in said, said

    again = await tool.call({**ask, "served_model": "vendor/extractor-next-2026-08"}, host=host, caller=caller)
    assert not again.is_error and again.structured is not None, again.text
    assert again.structured["run_ids"] == []
    assert again.structured["already_attested"] == [first[0].id, second[0].id]

    other = await tool.call({**ask, "served_model": "vendor/another"}, host=host, caller=caller)
    assert other.is_error and "a statement is never replaced" in other.text


async def test_a_statement_never_replaces_a_model_a_response_named() -> None:
    named = _named(_CANDIDATE, "named", _NOW)
    storage = _stored([_named(_CONTROL, "control", _NOW), named])
    launch, _client = toyhost_launch_host(storage=storage)
    (tool, *_) = eval_catalogue().mount_all(standard_tools())

    refused = await tool.call(
        {
            "action": "runs_attest",
            "coordinate": "served_model",
            "run_ids": [named[0].id],
            "served_model": "vendor/another",
        },
        host=OpsHost(launch=launch),
        caller=Caller(scope_id=TOYHOST_SCOPE, identity="agent:backfill"),
    )

    assert refused.is_error and "a statement never replaces a record" in refused.text
    run = off_loop(storage.load_eval_run, named[0].id, TOYHOST_SCOPE)
    assert run is not None and run.served_model_attestation is None, "nothing was written"


def test_a_logged_output_carries_the_served_model_its_line_names(tmp_path: Path) -> None:
    log = Log(tmp_path / "outputs.jsonl")
    log({"id": "c1"}, "refund", model="vendor/alias-latest", served_model="vendor/model-2026-08")
    log({"id": "c2"}, "close", model="vendor/alias-latest")
    outputs = Log(tmp_path / "outputs.jsonl").outputs()
    assert outputs["c1"][0].served_model == "vendor/model-2026-08"
    assert outputs["c2"][0].served_model is None, "a line naming none stays unknown: nothing is read off the id asked"


# =============================================================================
# (c), (d) The version and the sentence
# =============================================================================


def test_the_cell_model_moved_once_and_a_bundle_carries_it() -> None:
    bundle = _bundle(_stored([_named(_CONTROL, "control", _NOW), _named(_CANDIDATE, "a", _NOW)]))
    assert CELL_MODEL_VERSION == 15 and bundle.cell_model_version == 15


def test_the_frontiers_cell_sentence_no_longer_says_a_campaigns_cells_pool_every_run_under_one_rig() -> None:
    assert "pooling every run of the variant under one rig" not in PASS_HAT_K_CELL_READING
    assert "is read in a cell of its own" in PASS_HAT_K_CELL_READING
    assert "is compared with nothing" in PASS_HAT_K_CELL_READING


def test_the_frontiers_cell_sentence_says_dominance_and_the_cheapest_pick_still_pool_definitions() -> None:
    """The frontier's dominance and cost-tie tests read contestants' headline cells whatever definitions read them:
    until they keep them apart, the cell sentence says so wherever a kept-out cell is shown (and a frontier whose
    contestants' definitions differ says it without one: ``test_definition_split_arm_level.py``)."""
    assert PASS_HAT_K_CELL_READING.endswith(
        "Dominance and the cheapest pick are still tested between contestants' headline cells whichever definitions "
        "of the host's measures read them."
    )


# =============================================================================
# Evidence: a statement moves the fingerprint, the cell model's version does not
# =============================================================================


async def test_stating_a_runs_served_model_is_evidence_and_moves_the_fingerprint_of_a_bundle_over_it() -> None:
    first, second = _silent("a"), _silent("b")
    storage = _stored([_named(_CONTROL, "control", _NOW), first, second])
    before = _bundle(storage)

    await attest_served_model(
        toyhost_host(storage=storage),
        [first[0].id, second[0].id],
        TOYHOST_SCOPE,
        "vendor/extractor-next-2026-08",
        attested_by="person:host",
    )
    after = _bundle(storage)

    assert before.evidence is not None and after.evidence is not None
    assert after.fingerprint() != before.fingerprint(), "a report over the runs as they were is no longer current"
    assert after.evidence.runs != before.evidence.runs, "who stated it, when, and what it covers is on the run"
    assert after.evidence.results == before.evidence.results, "no result was rewritten"


async def test_stating_a_runs_measure_definitions_moves_the_fingerprint_too() -> None:
    batches = [_named(_CONTROL, "control", None), _named(_CANDIDATE, "a", None)]
    storage = _stored(batches)
    before = _bundle(storage)

    await attest_measure_definitions(
        toyhost_host(storage=storage),
        [batch.id for batch, _ in batches],
        TOYHOST_SCOPE,
        _NOW,
        attested_by="person:host",
    )

    assert _bundle(storage).fingerprint() != before.fingerprint()


def test_a_stored_report_over_unmoved_evidence_reads_as_current_across_the_cell_model_move() -> None:
    """A report written under cell model 14 recorded the fingerprint of its evidence. The rule that pools cells is
    the package's, not evidence: over the same runs and results this build computes the same fingerprint, so the
    report reproduces and names no cause, whatever version its bundle is stamped with."""
    bundle = _bundle(_stored([_named(_CONTROL, "control", _NOW), _named(_CANDIDATE, "a", _NOW)]))
    stored_under_14 = bundle.model_copy(update={"cell_model_version": 14})
    assert stored_under_14.fingerprint() == bundle.fingerprint()

    inspected = BundleInspection.over(
        bundle,
        analysis_id="written-under-cell-model-14",
        recorded_fingerprint=stored_under_14.fingerprint(),
        recorded_schema_version=bundle.schema_version,
        recorded_host_declarations_digest=bundle.host_declarations_digest,
    )

    assert (inspected.reproduces_generation, inspected.mismatch_cause) == (True, None)


# =============================================================================
# Every reader of the rig sees the definitions: divergences and the confound scans (R-3)
# =============================================================================


def test_a_divergence_is_never_tested_between_runs_read_under_two_definitions_and_the_scans_name_it() -> None:
    """The divergence lens contrasts lever levels over cohorts of runs; under one definition it tests them, and where
    the cohorts were read under two, or under none recorded, it tests nothing and counts the pair untested."""

    def timed(
        model: str, salt: str, definitions: dict[str, str] | None, llm_ms: float
    ) -> tuple[EvalRun, list[EvalResult]]:
        """A batch whose wall-clock holds a model-call part, so the lens has a whole and a part to test."""
        batch, results = _named(model, salt, definitions)
        return batch, [
            result.model_copy(
                update={
                    "latency": LatencyMetrics(
                        total_ms=(result.latency.total_ms or 0.0) + llm_ms + index,
                        llm_ms=llm_ms + 7 * (index % 5),
                    )
                }
            )
            for index, result in enumerate(results)
            if result.latency is not None
        ]

    one = _bundle(_stored([timed(_CONTROL, "control", _NOW, 900.0), timed(_CANDIDATE, "a", _NOW, 4000.0)]))
    assert one.divergences_tested > 0, "under one definition the lens tests the pair, or this asserts nothing"
    assert _DEFINITIONS not in {confound.dimension for confound in one.apparatus_confounds}

    for before in (_BEFORE, None):
        two = _bundle(_stored([timed(_CONTROL, "control", before, 900.0), timed(_CANDIDATE, "a", _NOW, 4000.0)]))
        assert two.divergences_tested == 0 and two.scope_divergences == [] and two.whole_swings == [], before
        assert two.divergences_untested > 0, before
        (scanned,) = [confound for confound in two.apparatus_confounds if confound.dimension == _DEFINITIONS]
        assert scanned.status == ("undecided" if before is None else "varied"), before
        assert _DEFINITIONS in two.confound_catalog, "the reason a reader judges it by is in the bundle"


# =============================================================================
# The host procedure: a reading-code change is marked by its revision FIRST, then each era is attested (R-4)
# =============================================================================


def _digests_at(revision: int) -> dict[str, str]:
    """The toy host's definitions with ``field_accuracy`` at ``revision`` of its reading code: the descriptor's text
    otherwise exactly as it is, as a host's descriptors are across a change made in reading code alone."""
    base = toyhost_profile().measures
    descriptors = [
        descriptor.model_copy(update={"definition_revision": revision})
        if descriptor.name == FIELD_ACCURACY
        else descriptor
        for descriptor in TOYHOST_MEASURES
    ]
    return MeasureRegistry(descriptors, families=base.families).definition_digests()


async def test_two_eras_of_one_descriptor_text_attested_under_their_revisions_never_pool() -> None:
    """A host whose reading code changed behind an unchanged descriptor raises the measure's revision first, then
    attests each era under the revision it ran under: the eras stay apart. Attested under the descriptor text alone,
    as the procedure once read, both eras state one digest and pool across a real change."""
    assert _digests_at(0) != _digests_at(1), "the revision is in the digest; the text alone is not enough"
    assert _digests_at(0) == _NOW, "revision 0 is the descriptors as they stand"

    async def attested(eras: Sequence[dict[str, str]]) -> AnalysisContextBundle:
        old, new = _named(_CANDIDATE, "old-era", None), _named(_CANDIDATE, "new-era", None)
        control = _named(_CONTROL, "control", None)
        storage = _stored([control, old, new])
        host = toyhost_host(storage=storage)
        for runs, definitions in (([control[0].id, new[0].id], eras[1]), ([old[0].id], eras[0])):
            await attest_measure_definitions(host, runs, TOYHOST_SCOPE, definitions, attested_by="job:upgrade")
        return _bundle(storage)

    marked = await attested([_digests_at(0), _digests_at(1)])
    cells = _arm(marked, control=False)
    assert len(cells) == 2 and all(cell.split_on == ["measure_definition"] for cell in cells), (
        "two eras, two cells: the arm is read per definition"
    )

    unmarked = await attested([_digests_at(0), _digests_at(0)])
    (pooled,) = _arm(unmarked, control=False)
    assert pooled.split_on == [], "what following the text alone does: one digest, one cell, across a real change"


async def test_a_run_recorded_at_revision_0_before_the_revision_was_raised_pools_with_the_earliest_era_once_stated() -> (
    None
):
    """Why the host procedure re-measures such a run BEFORE any era is stated, never "leaves it apart" (#1187).

    A build carrying #1160 stamped revision-0 digests on every run it launched before the host raised the revision
    of a measure whose reading code had changed (procedure step 1), whatever code the run ran under. The earliest
    era is then stated at revision 0. A record and a statement of one digest are one definition (an attestation is
    read through, never told apart from a record), so the run joins the earliest era's cell: the cells pool on what
    is recorded or stated, never on an assumption, and an old run nothing stated matches nothing, not even that
    record. The record is what is wrong, and only a re-measure replaces it."""
    by_run: dict[str, str] = {}

    def made(
        salt: str, definitions: dict[str, str] | None, model: str = _CANDIDATE
    ) -> tuple[EvalRun, list[EvalResult]]:
        batch, results = _named(model, salt, definitions)
        by_run.update({result.id: salt for result in results})
        return batch, results

    control = made("control", _digests_at(1), _CONTROL)
    old = made("unrecorded-earliest-era", None)
    recorded_early = made("recorded-at-revision-0-under-todays-code", _digests_at(0))
    today = made("recorded-after-step-1", _digests_at(1))
    storage = _stored([control, old, recorded_early, today])

    def runs_by_cell(bundle: AnalysisContextBundle) -> list[list[str]]:
        return sorted(
            sorted({by_run[result_id] for result_id in cell.observation_ids}) for cell in _arm(bundle, control=False)
        )

    assert runs_by_cell(_bundle(storage)) == [
        ["recorded-after-step-1"],
        ["recorded-at-revision-0-under-todays-code"],
        ["unrecorded-earliest-era"],
    ], "before any statement the unrecorded run matches nothing, and the revision-0 record matches nothing recorded"

    host = toyhost_host(storage=storage)
    await attest_measure_definitions(host, [old[0].id], TOYHOST_SCOPE, _digests_at(0), attested_by="job:upgrade")

    assert runs_by_cell(_bundle(storage)) == [
        ["recorded-after-step-1"],
        ["recorded-at-revision-0-under-todays-code", "unrecorded-earliest-era"],
    ], "stated at revision 0, the earliest era pools with the run that recorded revision 0: it is not left apart"
