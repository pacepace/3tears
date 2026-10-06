"""The reader conformance kit passes on the toy host, and each of its cases can fail.

Three things are proved here:

* **The toy host's readers conform.** Every case of ``READER_CONFORMANCE_CASES`` runs over the toy
  host's campaign plus a batch that overlaid a retrieval knob, so the open family and its residual
  reader are read over a member rather than over an empty map.
* **No case is vacuous.** :data:`_FAULTS` is a table of broken profiles, each the toy host with one
  reader broken, and the case each must turn red. A case no fault reaches fails
  :func:`test_every_case_is_turned_red_by_some_fault`.
* **A sample with no runs is refused**, by every case: a reader checked over nothing is unchecked.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.contracts import EvalResult, EvalRun
from threetears.evals.contracts.host import HostProfile, Sweepable, SweepableRegistry, SweepableValue
from threetears.evals.testing import (
    READER_CONFORMANCE_CASES,
    ReaderConformanceCase,
    ReaderConformanceFailure,
    ReaderSample,
)
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import toyhost_batch, toyhost_measurements
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.variant import tunable_variant_levers


def _sample(profile: HostProfile | None = None) -> ReaderSample:
    """The toy host's campaign, plus one batch that overlaid a retrieval knob, checked under ``profile``.

    The runs are always recorded under the sound tunable profile, as a host's store holds them; only
    the readers checked over them change, so a broken reader is caught by the kit rather than by the
    recording.
    """
    host = toyhost_profile(tunable_retrieval=True)
    campaign, storage = toyhost_campaign(profile=host)
    runs = storage.load_eval_runs(campaign.run_ids, campaign.scope_id)
    tuned = toyhost_batch(
        chunk_tokens=512,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
        retrieval_overrides={"rerank_depth": 8},
        retrieval_config={"rerank_depth": 8, "dedupe_threshold": 0.9},
    )
    tuned_results = toyhost_measurements(tuned, profile=host, cost_usd=0.002, total_ms=900.0, field_accuracy=0.9)
    pairs = [(run, storage.query_eval_results_by_run(run.id, run.scope_id)) for run in runs]
    return ReaderSample(profile=profile if profile is not None else host, runs=[*pairs, (tuned, tuned_results)])


# --- the toy host conforms ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", READER_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_the_toy_hosts_readers_conform(case: ReaderConformanceCase) -> None:
    case.run(_sample())


def test_the_sample_exercises_an_open_familys_member() -> None:
    """The family cases mean nothing over empty maps; the tuned batch is what gives them a member."""
    sample = _sample()
    family = sample.profile.sweepables.get("retrieval_overrides")
    assert family is not None
    assert any(family.read(run, results) for run, results in sample.runs)


def test_case_names_are_unique() -> None:
    names = [case.name for case in READER_CONFORMANCE_CASES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("case", READER_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_a_sample_with_no_runs_is_refused(case: ReaderConformanceCase) -> None:
    with pytest.raises(ReaderConformanceFailure, match="holds no run"):
        case.run(ReaderSample(profile=toyhost_profile(), runs=[]))


def test_a_failure_names_the_case_the_rule_and_the_reader() -> None:
    case = next(c for c in READER_CONFORMANCE_CASES if c.name == "sweepable.json_safe")
    with pytest.raises(ReaderConformanceFailure) as raised:
        case.run(_sample(_with_reader("batch_label", lambda _run, _results: {"a", "b"})))
    message = str(raised.value)
    assert message.startswith(f"{case.name}: {case.rule} — ")
    assert "'batch_label'" in message


# --- every case can fail ------------------------------------------------------------------------------


def _with_declaration(name: str, **changes: Any) -> HostProfile:
    """The tunable toy host with one declaration changed."""
    profile = toyhost_profile(tunable_retrieval=True)
    registry = profile.host_sweepables
    rebuilt = SweepableRegistry(
        [replace(declared, **changes) if declared.name == name else declared for declared in registry.declarations],
        roles=registry.roles,
    )
    return replace(profile, host_sweepables=rebuilt)


def _with_reader(name: str, read: Callable[[EvalRun, Sequence[EvalResult]], Any]) -> HostProfile:
    return _with_declaration(name, read=read)


_COUNTER = itertools.count()


def _mutating(run: EvalRun, _results: Sequence[EvalResult]) -> str:
    run.host_payload.setdefault("toyhost", {})["touched"] = True
    return "pool-a"


def _first_result(_run: EvalRun, results: Sequence[EvalResult]) -> str | None:
    return results[0].id if results else None


def _stray_lever(run: EvalRun) -> dict[str, SweepableValue]:
    return {**tunable_variant_levers(run), "a_lever_nobody_declared": SweepableValue.of(1, display="1")}


#: ``(fault, the profile carrying it, the case it must turn red)``.
_FAULTS: tuple[tuple[str, Callable[[], HostProfile], str], ...] = (
    ("a reader returning a set", lambda: _with_reader("batch_label", lambda _r, _s: {"x"}), "sweepable.json_safe"),
    (
        "a reader counting its calls",
        lambda: _with_reader("batch_label", lambda _r, _s: next(_COUNTER)),
        "sweepable.deterministic",
    ),
    (
        "a reader keyed on the first result",
        lambda: _with_reader("batch_label", _first_result),
        "sweepable.order_independent",
    ),
    ("a reader writing to the run", lambda: _with_reader("batch_label", _mutating), "sweepable.mutates_nothing"),
    (
        "a family reading a list",
        lambda: _with_reader("retrieval_overrides", lambda _r, _s: ["retrieval.rerank_depth"]),
        "open_family.member_map",
    ),
    (
        "a family resolving a member it disowns",
        lambda: _with_reader("retrieval_overrides", lambda _r, _s: {"retrieval.colour": "blue"}),
        "open_family.member_map",
    ),
    (
        "a family resolving members with no membership test",
        lambda: _with_declaration("retrieval_overrides", owns_member=None),
        "open_family.member_map",
    ),
    (
        "a residual returning a set",
        lambda: _with_declaration("retrieval_overrides", read_residual=lambda _r, _s, _removed: {"x"}),
        "open_family.residual_json_safe",
    ),
    (
        "a lever map naming an undeclared lever",
        lambda: replace(toyhost_profile(tunable_retrieval=True), variant_levers=_stray_lever),
        "variant_levers.match_the_registry",
    ),
)


@pytest.mark.parametrize(("fault", "build", "case_name"), _FAULTS, ids=[fault for fault, _, _ in _FAULTS])
def test_each_fault_turns_its_case_red(fault: str, build: Callable[[], HostProfile], case_name: str) -> None:
    case = next(c for c in READER_CONFORMANCE_CASES if c.name == case_name)
    with pytest.raises(ReaderConformanceFailure):
        case.run(_sample(build()))


def test_every_case_is_turned_red_by_some_fault() -> None:
    reached = {case_name for _, _, case_name in _FAULTS}
    assert reached == {case.name for case in READER_CONFORMANCE_CASES}


def _red_cases(sample: ReaderSample, cases: Sequence[ReaderConformanceCase]) -> list[str]:
    """Every case run over ONE sample in the order given, and the ones that went red, in that order."""
    red = []
    for case in cases:
        try:
            case.run(sample)
        except ReaderConformanceFailure:
            red.append(case.name)
    return red


@pytest.mark.parametrize("order", ["forward", "reversed"])
@pytest.mark.parametrize(("fault", "build", "case_name"), _FAULTS, ids=[fault for fault, _, _ in _FAULTS])
def test_each_fault_turns_exactly_its_owning_case_red_over_one_shared_sample(
    fault: str, build: Callable[[], HostProfile], case_name: str, order: str
) -> None:
    """One broken promise, one red case — over one sample shared by every case, in either order.

    Shared because a host builds its sample once (a module-scoped fixture, recorded runs are expensive): a
    reader that writes to the run would otherwise do its damage in whichever case ran first, and
    ``mutates_nothing`` would compare the mutated state against itself and pass. Exactly the owner, because
    a JSON fault reported again as an ordering fault — or a list-reading family surfacing as an engine
    ``ValueError`` under the variant-key case — sends the host to the wrong reader. No case raises anything
    but its own failure: an escaping exception fails this test outright.
    """
    sample = _sample(build())
    cases = READER_CONFORMANCE_CASES if order == "forward" else tuple(reversed(READER_CONFORMANCE_CASES))

    assert _red_cases(sample, cases) == [case_name]


def test_no_case_mutates_the_sample_it_was_handed() -> None:
    """The kit reads copies: the caller's runs are untouched even by a reader that writes to every run it reads."""
    sample = _sample(_with_reader("batch_label", _mutating))
    before = [
        (run.model_dump(mode="json"), [r.model_dump(mode="json") for r in results]) for run, results in sample.runs
    ]

    _red_cases(sample, READER_CONFORMANCE_CASES)

    assert [
        (run.model_dump(mode="json"), [r.model_dump(mode="json") for r in results]) for run, results in sample.runs
    ] == before


def test_a_sweepable_declaration_is_what_the_kit_reads() -> None:
    """The kit reads the profile's merged registry — a kind contract's levers are checked too."""
    sample = _sample()
    names = {declared.name for declared in sample.profile.sweepables.declarations}
    assert any(name.startswith("extractor.") for name in names)
    assert all(isinstance(declared, Sweepable) for declared in sample.profile.sweepables.declarations)
