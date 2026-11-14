"""Runs read under two definitions of one host measure never pool, and a stored run can be re-measured (#1160).

A host changes what a measure counts and keeps its key. Each run records the definition digest of every measure its
host declared at launch, the map is a component of its measurement context, and the reads that pool runs (the
frontier's pass^k cells, the comparison sets) keep two definitions apart and name the measure. A run stored before
runs recorded definitions reads as not recorded and pools with nothing: a silence is never a match. A re-measure
reads a stored run again from what its cells stored, through the host's own reader, marks it, and it then joins the
cell of the definitions it was read under; nothing is interpolated, and a refused re-measure writes nothing.

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
from threetears.evals.kernel.identity import IDENTITY_VERSION, resolve_context_identity
from threetears.evals.kernel.metrics import (
    MEASURE_DEFINITION_FIELDS,
    MEASURE_READING_FIELDS,
    MetricDescriptor,
    measure_definition_digest,
)
from threetears.evals.kernel.scoring import pass_hat_k_cell
from threetears.evals.ops import LaunchArguments, OpsHost, run_launch
from threetears.evals.ops.lenses import scope_frontier
from threetears.evals.run import remeasure_run
from threetears.evals.run.reads import list_runs
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run
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


def _as_stored_before_definitions(store: InMemoryDocumentStore, run_id: str) -> None:
    """Rewrite a run's record as a build before runs recorded measure definitions stored it."""
    document = store.get(run_id, TOYHOST_SCOPE)
    assert document is not None
    document.pop("measure_definitions")
    document["context_components"].pop("measure_definitions")
    document["identity_version"] = IDENTITY_VERSION - 1
    store.upsert(document)


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
# The digest
# =============================================================================


def test_every_descriptor_field_is_either_what_the_measure_is_or_how_it_is_read() -> None:
    """A field the descriptor gains is classified before it can silently join, or silently miss, the digest."""
    assert not MEASURE_DEFINITION_FIELDS & MEASURE_READING_FIELDS
    assert MEASURE_DEFINITION_FIELDS | MEASURE_READING_FIELDS == set(MetricDescriptor.model_fields)


def test_the_digest_moves_with_what_a_value_counts_and_never_with_how_it_is_named_or_read() -> None:
    (accuracy,) = [descriptor for descriptor in TOYHOST_MEASURES if descriptor.name == FIELD_ACCURACY]
    digest = measure_definition_digest(accuracy)

    renamed = accuracy.model_copy(
        update={"reader_name": "Exactness", "description": "Reworded.", "materiality_threshold": 0.5}
    )
    assert measure_definition_digest(renamed) == digest
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


# =============================================================================
# Two definitions never share a cell or a comparison
# =============================================================================


async def test_a_launched_run_records_each_measures_definition_in_its_measurement_context() -> None:
    _store_, storage = off_loop(_store)
    profile = _profile()

    run = await _launch(storage, profile)

    assert run.measure_definitions == profile.measures.definition_digests()
    assert run.context_components is not None
    assert run.context_components.measure_definitions == run.measure_definitions
    identity = resolve_context_identity(run, profile)
    assert (identity.source, identity.identity_version, identity.remeasured) == ("stamped", IDENTITY_VERSION, False)
    assert "measure_definitions" not in identity.missing_components


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
    assert FIELD_ACCURACY in disclosure and redefined.id in disclosure and "never pool" in disclosure


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
# A run stored before the change
# =============================================================================


async def test_a_run_stored_before_runs_recorded_definitions_reads_not_recorded_and_pools_with_nothing() -> None:
    """Not with a run that records them, and not with another silent run: a silence is never a match."""
    store, storage = off_loop(_store)
    profile = _profile()
    old_a, old_b = await _launch(storage, profile), await _launch(storage, profile)
    new_a, new_b = await _launch(storage, profile), await _launch(storage, profile)
    for old in (old_a, old_b):
        off_loop(_as_stored_before_definitions, store, old.id)

    stored = off_loop(storage.load_eval_run, old_a.id, TOYHOST_SCOPE)
    assert stored is not None and stored.measure_definitions is None
    identity = resolve_context_identity(stored, profile)
    assert identity.source == "derived" and identity.partial
    assert "measure_definitions" in identity.missing_components

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
    assert "pools with nothing" in grouped["measure_definition_disclosure"]


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
