"""An uncapped run's spend ceiling is a recorded level, never an unknown.

``max_cost_usd`` is null for two reasons: the run was uncapped (cost enforcement off), or its writer
recorded no ceiling. ``EvalRun.max_cost_usd_origin`` says which (``uncapped`` against ``None``), and
the ``max_cost_usd`` reader now reads it: an uncapped run reports :data:`UNCAPPED_SPEND`, so two
uncapped runs agree, and only a run with no recorded ceiling stays undecidable. Pinned on both
surfaces that read the declaration: the bisection and the bundle's apparatus scan.
"""

from __future__ import annotations

from threetears.evals.analysis.reads import bisect_runs
from threetears.evals.contracts import EvalStorage
from threetears.evals.contracts.host.sweepables import UNCAPPED_SPEND
from threetears.evals.contracts.models import EvalRun
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

_UNCAPPED = {"max_cost_usd": None, "max_cost_usd_origin": "uncapped"}
_UNRECORDED = {"max_cost_usd": None, "max_cost_usd_origin": None}


def _bisect(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    storage = EvalStorage(InMemoryDocumentStore())
    run_a, run_b = make_eval_run(id="run-a", **a), make_eval_run(id="run-b", **b)
    for run in (run_a, run_b):
        storage.save_eval_run(run)
    return bisect_runs(storage, run_a.id, run_b.id, run_a.scope_id, profile=toyhost_profile())


def test_two_uncapped_runs_agree_on_their_spend_ceiling_in_the_bisection() -> None:
    bisected = _bisect(_UNCAPPED, _UNCAPPED)

    assert "max_cost_usd" in bisected["same"], bisected
    assert bisected["details"]["max_cost_usd"] == {"a": UNCAPPED_SPEND, "b": UNCAPPED_SPEND}


def test_an_uncapped_run_against_a_capped_one_is_a_difference() -> None:
    bisected = _bisect(_UNCAPPED, {"max_cost_usd": 0.5, "max_cost_usd_origin": "inherited"})

    assert "max_cost_usd" in bisected["differs"], bisected


def test_a_run_with_no_recorded_ceiling_is_still_undecidable() -> None:
    assert "max_cost_usd" in _bisect(_UNRECORDED, _UNRECORDED)["unknown"]
    assert "max_cost_usd" in _bisect(_UNCAPPED, _UNRECORDED)["unknown"]


def test_an_uncapped_origin_wins_over_a_number_that_bound_nothing() -> None:
    """Enforcement off means no number bound the run, so one carried anyway is not its ceiling."""
    bisected = _bisect(_UNCAPPED, {"max_cost_usd": 0.5, "max_cost_usd_origin": "uncapped"})

    assert "max_cost_usd" in bisected["same"], bisected


def _bundle_confounds(*updates: dict[str, object]) -> dict[str, str]:
    """The toy campaign's apparatus confounds, its runs' ceilings rewritten as ``updates`` in order."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    for run_id, update in zip(campaign.run_ids, updates, strict=True):
        run: EvalRun | None = storage.load_eval_run(run_id, TOYHOST_SCOPE)
        assert run is not None
        for field, value in update.items():
            setattr(run, field, value)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    return {confound.dimension: confound.status for confound in bundle.apparatus_confounds}


def test_two_uncapped_runs_raise_no_spend_ceiling_confound_in_the_bundle() -> None:
    assert "max_cost_usd" not in _bundle_confounds(_UNCAPPED, _UNCAPPED)


def test_a_run_with_no_recorded_ceiling_still_raises_an_undecided_confound_in_the_bundle() -> None:
    assert _bundle_confounds(_UNCAPPED, _UNRECORDED).get("max_cost_usd") == "undecided"
