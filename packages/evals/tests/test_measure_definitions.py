"""Runs read under two definitions of one host measure never share a pass^k cell, and a stored run can be attested
or re-measured (#1160).

A host changes what a measure counts and keeps its key. Each run records the definition digest of every measure its
host declared at launch, and the map is an additive component of its measurement context: a run with no host
measure keeps the key it had, and no identity version moves. The frontier's pass^k cells and the comparison sets
keep two definitions apart and name the measure. A run stored before runs recorded definitions reads as not
recorded; where its results hold a host measure it pools with nothing, not even another such run (a silence is never
a match), and where they hold none it pools exactly as before. Its host may state the definitions it ran under (an
attestation, recorded as one), or re-measure it from what its cells stored; nothing is interpolated, a refused
re-measure writes nothing, and one cut short leaves the run in no cell.

Every pooling claim here is made through the front door: runs launched by the toy host into a real store, read back
by ``scope_frontier`` and ``comparison_sets``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.analysis import comparison_sets
from threetears.evals.analysis.lenses.comparison_sets import BADGE_CONTEXT_INCOMPLETE, BADGE_MEASURE_DEFINITIONS_DIFFER
from threetears.evals.analysis.lenses.frontier import PassHatKCellReading
from threetears.evals.kernel import EvalStorage, ValidationFailedError
from threetears.evals.kernel.host import HostProfile, MeasureRegistry, MeasureRereadRequest
from threetears.evals.kernel.errors import ConflictError
from threetears.evals.kernel.identity import (
    IDENTITY_VERSION,
    compute_context_key,
    derive_context_identity,
    derive_variant_identity,
    resolve_context_identity,
)
from threetears.evals.kernel.metrics import measure_definition_digest
from threetears.evals.kernel.scoring import pass_hat_k_cell
from threetears.evals.ops import LaunchArguments, OpsHost, run_launch
from threetears.evals.ops.lenses import scope_frontier
from threetears.evals.run import attest_measure_definitions, remeasure_run
from threetears.evals.run.reads import list_runs
from threetears.evals.schema import ContextComponents, EvalResult, EvalRun
from threetears.evals.schema.models import RoleUsage
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run, make_scored_result
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import FIELD_ACCURACY
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import launched_runs

_MODEL = RUN_MODELS[0]


def _strict_accuracy(request: MeasureRereadRequest) -> dict[str, bool | float | str]:
    """The toy host's reader of a stored result under its revised definition: all graded fields exact, or nothing.

    Read from what the cell stored (the fields it missed, on the result's kind payload), never from the old value.
    """
    missed = (request.result.kind_payload or {}).get("missed_fields", [])
    return {FIELD_ACCURACY: 0.0 if missed else 1.0} if FIELD_ACCURACY in request.measures else {}


def _every_held_measure(request: MeasureRereadRequest) -> dict[str, bool | float | str]:
    """A reader for a run that recorded nothing: accuracy read again, and each other measure the result holds
    answered as stored, which is the host saying that measure's definition never moved."""
    held = {
        name: request.result.host_measures[name] for name in request.measures if name in request.result.host_measures
    }
    return {**held, **_strict_accuracy(request)}


def _profile(revision: int = 0, *, rereader: Any = None, shapes_run: bool = False) -> HostProfile:
    """The toy host, declaring ``field_accuracy`` at ``revision`` of its reading code."""
    base = toyhost_profile()
    measures = [
        descriptor.model_copy(update={"definition_revision": revision, "shapes_run": shapes_run})
        if descriptor.name == FIELD_ACCURACY
        else descriptor
        for descriptor in TOYHOST_MEASURES
    ]
    return replace(base, measures=MeasureRegistry(measures, families=base.measures.families), measure_rereader=rereader)


def _store() -> tuple[InMemoryDocumentStore, EvalStorage]:
    store = InMemoryDocumentStore()
    storage = EvalStorage(store)
    storage.save_template(toyhost_template())
    return store, storage


async def _launch(storage: EvalStorage, profile: HostProfile) -> EvalRun:
    """Launch one arm of the toy host's template under ``profile``, to its end, and read the run back."""
    launch, _client = toyhost_launch_host(profile=profile, storage=storage)
    host = OpsHost(launch=launch)
    arguments = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=[_MODEL], k_runs=1
    )
    (run_id,) = await launched_runs(host, await run_launch(host, arguments, TOYHOST_SCOPE))
    run = off_loop(storage.load_eval_run, run_id, TOYHOST_SCOPE)
    assert run is not None and run.status == "completed"
    return run


def _as_stored_before_definitions(store: InMemoryDocumentStore, run_id: str) -> str:
    """Rewrite a run's record as a build before runs recorded measure definitions stored it, and answer the context
    key that build stamped: the digest of its components without the definitions."""
    document = store.get(run_id, TOYHOST_SCOPE)
    assert document is not None
    document.pop("measure_definitions")
    document["context_components"].pop("measure_definitions")
    document["context_key"] = compute_context_key(ContextComponents.model_validate(document["context_components"]))
    store.upsert(document)
    return document["context_key"]


def _cells(storage: EvalStorage, profile: HostProfile) -> list[PassHatKCellReading]:
    """The one contestant's pass^k cells as the public frontier read answers them, the headline first."""
    host, _client = toyhost_launch_host(profile=profile, storage=storage)
    frontier = off_loop(scope_frontier, host.eval_host, TOYHOST_SCOPE)
    (subject,) = frontier.subjects
    (point,) = subject.points
    assert point.pass_hat_k_cells[0].headline
    return point.pass_hat_k_cells


def _comparison(storage: EvalStorage, profile: HostProfile) -> dict[str, Any]:
    """The scope's one comparison set as a host serves it."""
    host, _client = toyhost_launch_host(profile=profile, storage=storage)

    def read() -> dict[str, Any]:
        return comparison_sets(
            storage,
            TOYHOST_SCOPE,
            list_runs=lambda scope_id, **kwargs: list_runs(host.eval_host, scope_id, **kwargs),
            profile=profile,
        )

    (grouped,) = off_loop(read)["comparison_sets"]
    return grouped


def _results(storage: EvalStorage, run_id: str) -> list[EvalResult]:
    return off_loop(storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)


# =============================================================================
# The digest, and the additive component
# =============================================================================


def test_the_digest_moves_with_what_a_value_counts_and_never_with_how_it_is_named_or_read() -> None:
    (accuracy,) = [descriptor for descriptor in TOYHOST_MEASURES if descriptor.name == FIELD_ACCURACY]
    digest = measure_definition_digest(accuracy)

    for metadata in (
        {"description": "Reworded, counting what it always counted."},
        {"nonnegative": not accuracy.nonnegative, "value_range": accuracy.value_range},
        {"reader_name": "Exactness", "reader_prose": "how exact the fields were"},
        {"materiality_threshold": 0.5},
    ):
        assert measure_definition_digest(accuracy.model_copy(update=metadata)) == digest, metadata
    for moved in ({"definition_revision": 1}, {"unit": "share"}, {"formula": "exact / graded"}, {"shapes_run": True}):
        assert measure_definition_digest(accuracy.model_copy(update=moved)) != digest, moved


def test_a_store_that_reorders_the_recorded_map_moves_neither_the_context_key_nor_the_cell() -> None:
    """A jsonb store hands a map's keys back in its own order."""
    profile = _profile()
    definitions = profile.measures.definition_digests()
    assert len(definitions) > 1
    stored = make_eval_run(template_digest="digest-1", measure_definitions=definitions, context_key="stamped")
    reordered = stored.model_copy(update={"measure_definitions": dict(reversed(list(definitions.items())))})

    assert (
        resolve_context_identity(stored, profile).context_key
        == resolve_context_identity(reordered, profile).context_key
    )
    same = stored.model_copy(update={"id": reordered.id})
    assert pass_hat_k_cell(same, profile) == pass_hat_k_cell(reordered, profile)


def test_the_component_is_additive_a_run_with_no_recorded_definitions_keeps_the_key_it_had() -> None:
    """No identity version moves: an absent map and an empty one are both left out of the pre-image."""
    profile = _profile()
    unrecorded = make_eval_run(measure_definitions=None)
    none_declared = unrecorded.model_copy(update={"measure_definitions": {}})
    recorded = unrecorded.model_copy(update={"measure_definitions": profile.measures.definition_digests()})

    without = derive_context_identity(unrecorded, profile)
    assert without.identity_version == IDENTITY_VERSION == 25
    assert without.context_components.measure_definitions is None
    assert "measure_definitions" not in without.context_components.model_dump(mode="json")
    assert "measure_definitions" not in without.missing_components
    assert derive_context_identity(none_declared, profile).context_key == without.context_key
    assert derive_context_identity(recorded, profile).context_key != without.context_key


# =============================================================================
# Recorded versus differently recorded
# =============================================================================


async def test_a_launched_run_records_each_measures_definition_in_its_measurement_context() -> None:
    _store_, storage = off_loop(_store)
    profile = _profile()

    run = await _launch(storage, profile)

    assert run.measure_definitions == profile.measures.definition_digests()
    assert run.context_components is not None
    assert run.context_components.measure_definitions == run.measure_definitions
    identity = resolve_context_identity(run, profile)
    assert (identity.source, identity.identity_version) == ("stamped", IDENTITY_VERSION)
    assert (identity.remeasured, identity.attested) == (False, False)


async def test_runs_launched_under_two_definitions_of_one_measure_never_share_a_cell_or_a_comparison() -> None:
    _store_, storage = off_loop(_store)
    before, after = _profile(revision=0), _profile(revision=1)
    first = await _launch(storage, before)
    second = await _launch(storage, before)
    redefined = await _launch(storage, after)

    headline, apart = _cells(storage, after)

    assert headline.run_ids == sorted([first.id, second.id]), "runs of one definition pool"
    assert (apart.run_ids, apart.reason) == ([redefined.id], "other_measure_definition")
    assert FIELD_ACCURACY in (apart.detail or ""), "the reason names the measure whose definition moved"
    assert first.context_key == second.context_key != redefined.context_key

    grouped = _comparison(storage, after)
    assert BADGE_MEASURE_DEFINITIONS_DIFFER in grouped["badges"]
    disclosure = grouped["measure_definition_disclosure"]
    assert FIELD_ACCURACY in disclosure and redefined.id in disclosure
    assert "pass^k keeps them in separate cells" in disclosure
    assert "still pool them" in disclosure, "the page says what is not kept apart yet"


async def test_runs_of_one_definition_carry_no_badge_and_no_disclosure() -> None:
    _store_, storage = off_loop(_store)
    profile = _profile()
    await _launch(storage, profile)
    await _launch(storage, profile)

    (only,) = _cells(storage, profile)
    grouped = _comparison(storage, profile)

    assert len(only.run_ids) == 2 and only.reason is None
    assert BADGE_MEASURE_DEFINITIONS_DIFFER not in grouped["badges"]
    assert "measure_definition_disclosure" not in grouped, "unset, it is left out of the served grouping"


# =============================================================================
# Unrecorded runs: alone where they hold a host measure, unchanged where they hold none
# =============================================================================


async def test_unrecorded_runs_that_hold_a_host_measure_pool_with_nothing_not_even_each_other() -> None:
    """The production case: two runs from either side of an unrecorded change, both stored before definitions were
    recorded, must not be averaged. A silence is never a match."""
    store, storage = off_loop(_store)
    profile = _profile()
    old_a, old_b = await _launch(storage, profile), await _launch(storage, profile)
    new_a, new_b = await _launch(storage, profile), await _launch(storage, profile)
    keys_then = {old.id: off_loop(_as_stored_before_definitions, store, old.id) for old in (old_a, old_b)}

    stored = off_loop(storage.load_eval_run, old_a.id, TOYHOST_SCOPE)
    assert stored is not None and stored.measure_definitions is None
    identity = resolve_context_identity(stored, profile)
    assert (identity.source, identity.context_key) == ("stamped", keys_then[old_a.id]), "its key is read unchanged"
    assert keys_then[old_a.id] == keys_then[old_b.id], "one key, and still two cells"

    headline, *alone = _cells(storage, profile)
    assert headline.run_ids == sorted([new_a.id, new_b.id])
    assert sorted((cell.run_ids, cell.reason) for cell in alone) == sorted(
        ([old.id], "no_recorded_measure_definitions") for old in (old_a, old_b)
    )
    assert all("recorded no definition" in (cell.detail or "") for cell in alone)

    grouped = _comparison(storage, profile)
    assert BADGE_CONTEXT_INCOMPLETE in grouped["badges"]
    assert BADGE_MEASURE_DEFINITIONS_DIFFER not in grouped["badges"], "nothing recorded evidences a difference"
    assert old_a.id in grouped["measure_definition_disclosure"]
    assert "recorded no measure definitions" in grouped["measure_definition_disclosure"]


def _plain_run(run_id: str, profile: HostProfile, **fields: Any) -> EvalRun:
    """A run of a host measure-less in what it stores, stamped as a launch before this change stamped it."""
    run = make_eval_run(
        id=run_id,
        status="completed",
        scope_id=TOYHOST_SCOPE,
        template_digest="digest-1",
        test_case_ids=["tc1"],
        **fields,
    )
    context = derive_context_identity(run, profile)
    variant = derive_variant_identity(run=run, profile=profile)
    return run.model_copy(
        update={
            "context_key": context.context_key,
            "context_components": context.context_components,
            "identity_version": IDENTITY_VERSION,
            "variant_levers": variant.levers,
        }
    )


def test_unrecorded_runs_that_hold_no_host_measure_pool_exactly_as_before() -> None:
    """Nothing a definition could have moved is on them, so nothing changes for them: one cell, the key they had."""
    profile = _profile()
    storage = EvalStorage(InMemoryDocumentStore())
    runs = [_plain_run(run_id, profile, measure_definitions=None) for run_id in ("run-a", "run-b")]
    key = derive_variant_identity(run=runs[0], profile=profile).variant_key
    for run in runs:
        storage.save_eval_run(run)
        result = make_scored_result(test_case_id="tc1", run_id=run.id, model=run.candidate_model).model_copy(
            update={
                "variant_key": key,
                "identity_version": IDENTITY_VERSION,
                "scope_id": TOYHOST_SCOPE,
                "usage": [
                    RoleUsage(
                        role="candidate", model=run.candidate_model, cost_usd=0.01, call_count=1, served_model="m"
                    )
                ],
            }
        )
        assert not result.host_measures
        storage.save_eval_result(result)
    host = toyhost_host(profile=profile, storage=storage)

    frontier = scope_frontier(host, TOYHOST_SCOPE)
    (subject,) = frontier.subjects
    (point,) = subject.points
    (only,) = point.pass_hat_k_cells

    assert only.run_ids == ["run-a", "run-b"] and only.reason is None
    stored = storage.load_eval_run("run-a", TOYHOST_SCOPE)
    assert stored is not None
    assert resolve_context_identity(stored, profile).context_key == runs[0].context_key
    (grouped,) = comparison_sets(
        storage,
        TOYHOST_SCOPE,
        list_runs=lambda scope_id, **kwargs: list_runs(host, scope_id, **kwargs),
        profile=profile,
    )["comparison_sets"]
    assert "measure_definition_disclosure" not in grouped


# =============================================================================
# The host's statement
# =============================================================================


async def test_a_host_attests_the_definitions_its_stored_runs_ran_under_and_they_pool_only_where_those_match() -> None:
    """The backfill door: the host marks its runs from before a change with the old definition and those since with
    the current one, touching no result."""
    store, storage = off_loop(_store)
    then, now = _profile(revision=0), _profile(revision=1)
    before_a, before_b = await _launch(storage, then), await _launch(storage, then)
    since = await _launch(storage, now)
    launched = await _launch(storage, now)
    for run in (before_a, before_b, since):
        off_loop(_as_stored_before_definitions, store, run.id)
    host, _client = toyhost_launch_host(profile=now, storage=storage)
    results = _results(storage, before_a.id)
    assert len(_cells(storage, now)) == 4, "three silent runs, each alone, beside the launched one"

    attested = await attest_measure_definitions(
        host.eval_host,
        [before_a.id, before_b.id],
        TOYHOST_SCOPE,
        then.measures.definition_digests(),
        attested_by="ops@toyhost",
        note="made before the 11-11 change",
    )
    await attest_measure_definitions(
        host.eval_host, [since.id], TOYHOST_SCOPE, now.measures.definition_digests(), attested_by="ops@toyhost"
    )

    assert attested.run_ids == [before_a.id, before_b.id]
    run = off_loop(storage.load_eval_run, before_a.id, TOYHOST_SCOPE)
    assert run is not None and run.measure_definitions == then.measures.definition_digests()
    mark = run.measure_attestation
    assert mark is not None and (mark.attested_by, mark.note) == ("ops@toyhost", "made before the 11-11 change")
    assert mark.attested_at and mark.definitions == run.measure_definitions
    assert _results(storage, before_a.id) == results, "no result is touched"
    identity = resolve_context_identity(run, now)
    assert (identity.source, identity.attested) == ("derived", True)

    cells = {tuple(cell.run_ids): cell.reason for cell in _cells(storage, now)}
    assert set(cells) == {tuple(sorted([before_a.id, before_b.id])), tuple(sorted([since.id, launched.id]))}
    assert "other_measure_definition" in cells.values()
    disclosure = _comparison(storage, now)["measure_definition_disclosure"]
    assert "stated by the host after the run, not recorded at launch" in disclosure and since.id in disclosure

    # A backfill job run twice changes nothing and is no error.
    marked = off_loop(storage.load_eval_run, before_a.id, TOYHOST_SCOPE)
    twice = await attest_measure_definitions(
        host.eval_host,
        [before_a.id, before_b.id],
        TOYHOST_SCOPE,
        then.measures.definition_digests(),
        attested_by="backfill-job",
    )
    assert (twice.run_ids, twice.already_attested) == ([], [before_a.id, before_b.id])
    assert off_loop(storage.load_eval_run, before_a.id, TOYHOST_SCOPE) == marked, "the first statement stands"

    # Other definitions over a launch's record, or over an attestation, are refused: a statement replaces nothing.
    for recorded in (launched.id, before_a.id):
        with pytest.raises(ValidationFailedError) as refused:
            await attest_measure_definitions(
                host.eval_host, [recorded], TOYHOST_SCOPE, now.measures.definition_digests(), attested_by="ops@toyhost"
            )
        assert "already records measure definitions" in refused.value.message
    assert off_loop(storage.load_eval_run, before_a.id, TOYHOST_SCOPE) == marked
    kept = off_loop(storage.load_eval_run, launched.id, TOYHOST_SCOPE)
    assert kept == launched and kept.measure_attestation is None, "a launch's record is never replaced"


async def test_a_run_with_no_stored_artifacts_is_attested_like_any_other() -> None:
    """The host derives an old run's definition from what the run recorded; nothing of its cells need survive."""
    storage = EvalStorage(InMemoryDocumentStore())
    profile = _profile()
    bare = make_eval_run(id="run-bare", status="completed", scope_id=TOYHOST_SCOPE, measure_definitions=None)
    off_loop(storage.save_eval_run, bare)
    host = toyhost_host(profile=profile, storage=storage)

    attested = await attest_measure_definitions(
        host, [bare.id], TOYHOST_SCOPE, profile.measures.definition_digests(), attested_by="backfill-job"
    )

    assert attested.run_ids == [bare.id]
    run = off_loop(storage.load_eval_run, bare.id, TOYHOST_SCOPE)
    assert run is not None and run.measure_definitions == profile.measures.definition_digests()


# =============================================================================
# The re-measure
# =============================================================================


async def test_a_remeasured_run_joins_the_current_definitions_cell_and_is_marked() -> None:
    _store_, storage = off_loop(_store)
    before, after = _profile(revision=0), _profile(revision=1, rereader=_strict_accuracy)
    old = await _launch(storage, before)
    new = await _launch(storage, after)
    host, _client = toyhost_launch_host(profile=after, storage=storage)
    as_written = {result.id: result.host_measures[FIELD_ACCURACY] for result in _results(storage, old.id)}

    reported = await remeasure_run(host.eval_host, old.id, TOYHOST_SCOPE)

    assert not reported.applied and reported.moved == [FIELD_ACCURACY] and reported.reread == [FIELD_ACCURACY]
    assert reported.results_reread == len(as_written)
    assert off_loop(storage.load_eval_run, old.id, TOYHOST_SCOPE) == old, "a report writes nothing"
    assert len(_cells(storage, after)) == 2

    applied = await remeasure_run(host.eval_host, old.id, TOYHOST_SCOPE, apply=True)

    assert applied.applied and applied.now == after.measures.definition_digests()
    run = off_loop(storage.load_eval_run, old.id, TOYHOST_SCOPE)
    assert run is not None and run.measure_definitions == new.measure_definitions
    (mark,) = run.measure_remeasures
    assert (mark.was, mark.now, mark.reread) == (old.measure_definitions, new.measure_definitions, [FIELD_ACCURACY])
    assert mark.completed_at is not None
    assert run.context_key == old.context_key, "the key its launch stamped stays on the document"
    identity = resolve_context_identity(run, after)
    assert (identity.source, identity.remeasured, identity.context_key) == ("derived", True, new.context_key)
    for result in _results(storage, old.id):
        (reread,) = result.measure_rereads
        assert reread.was == {FIELD_ACCURACY: as_written[result.id]}, "what the cell wrote is kept"
        assert reread.definitions == {FIELD_ACCURACY: after.measures.definition_digests()[FIELD_ACCURACY]}
        assert result.host_measures[FIELD_ACCURACY] in (0.0, 1.0), "read again under the current definition"

    (pooled,) = _cells(storage, after)
    assert pooled.run_ids == sorted([old.id, new.id])
    assert old.id in _comparison(storage, after)["measure_definition_disclosure"], "and the page says which run"
    # Read with no host to derive it, the stamped key is the launch's: it joins neither cell by assumption.
    unhosted = pass_hat_k_cell(run, None, served=("m",))
    assert unhosted != pass_hat_k_cell(old, None, served=("m",))
    assert unhosted != pass_hat_k_cell(new, None, served=("m",))

    again = await remeasure_run(host.eval_host, old.id, TOYHOST_SCOPE, apply=True)
    assert again.already_current and not again.applied
    assert len(off_loop(storage.load_eval_run, old.id, TOYHOST_SCOPE).measure_remeasures) == 1


async def test_a_remeasure_cut_short_leaves_the_run_in_no_cell_until_a_second_pass_finishes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its results are written one by one: part-way, some hold values read under the new definition, so the run is
    marked first and pools with neither the old definition's runs nor the new one's."""
    _store_, storage = off_loop(_store)
    before, after = _profile(revision=0), _profile(revision=1, rereader=_strict_accuracy)
    old, sibling = await _launch(storage, before), await _launch(storage, before)
    new = await _launch(storage, after)
    host, _client = toyhost_launch_host(profile=after, storage=storage)
    replace_result, written = storage.replace_eval_result, []

    def racing(result: EvalResult, *, if_match: str | None) -> None:
        if written:
            raise ConflictError("raced", remedy="read it again")
        written.append(result.id)
        replace_result(result, if_match=if_match)

    monkeypatch.setattr(storage, "replace_eval_result", racing)
    with pytest.raises(ConflictError):
        await remeasure_run(host.eval_host, old.id, TOYHOST_SCOPE, apply=True)
    monkeypatch.undo()

    run = off_loop(storage.load_eval_run, old.id, TOYHOST_SCOPE)
    assert run is not None and run.measure_definitions == old.measure_definitions
    assert [mark.completed_at for mark in run.measure_remeasures] == [None]
    assert len(written) == 1, "one result holds a value read under the new definition"
    cells = {tuple(cell.run_ids): cell for cell in _cells(storage, after)}
    assert set(cells) == {(old.id,), (sibling.id,), (new.id,)}, "in no cell but its own"
    assert "did not finish" in (cells[(old.id,)].detail or "")
    assert "began and did not finish" in _comparison(storage, after)["measure_definition_disclosure"]

    finished = await remeasure_run(host.eval_host, old.id, TOYHOST_SCOPE, apply=True)

    assert finished.applied and finished.results_reread == len(_results(storage, old.id)) - 1, "none is read twice"
    run = off_loop(storage.load_eval_run, old.id, TOYHOST_SCOPE)
    assert run is not None and [bool(mark.completed_at) for mark in run.measure_remeasures] == [True]
    assert all(len(result.measure_rereads) == 1 for result in _results(storage, old.id))
    assert tuple(sorted([old.id, new.id])) in {tuple(cell.run_ids) for cell in _cells(storage, after)}


async def test_a_run_that_recorded_no_definitions_is_remeasured_into_the_current_cell() -> None:
    store, storage = off_loop(_store)
    profile = _profile(rereader=_every_held_measure)
    old, new = await _launch(storage, profile), await _launch(storage, profile)
    off_loop(_as_stored_before_definitions, store, old.id)
    host, _client = toyhost_launch_host(profile=profile, storage=storage)
    assert len(_cells(storage, profile)) == 2

    applied = await remeasure_run(host.eval_host, old.id, TOYHOST_SCOPE, apply=True)

    assert applied.was is None and FIELD_ACCURACY in applied.reread
    (pooled,) = _cells(storage, profile)
    assert pooled.run_ids == sorted([old.id, new.id])


@pytest.mark.parametrize(
    ("after", "says"),
    [
        pytest.param({"revision": 1}, "wires no reader", id="no-reader"),
        pytest.param({"revision": 1, "rereader": lambda _request: {}}, "unanswered", id="value-unanswered"),
        pytest.param(
            {"revision": 1, "rereader": _strict_accuracy, "shapes_run": True}, "shaped what the run did", id="shaped"
        ),
    ],
)
async def test_a_remeasure_that_cannot_read_every_stored_value_again_refuses_and_writes_nothing(
    after: dict[str, Any], says: str
) -> None:
    """No interpolation: a value read under one definition is never kept under another's digest, nor guessed."""
    _store_, storage = off_loop(_store)
    old = await _launch(storage, _profile(revision=0))
    results = _results(storage, old.id)
    host, _client = toyhost_launch_host(profile=_profile(**after), storage=storage)

    with pytest.raises(ValidationFailedError) as refused:
        await remeasure_run(host.eval_host, old.id, TOYHOST_SCOPE, apply=True)

    assert says in refused.value.message, refused.value.message
    assert refused.value.remedy
    assert off_loop(storage.load_eval_run, old.id, TOYHOST_SCOPE) == old
    assert _results(storage, old.id) == results
