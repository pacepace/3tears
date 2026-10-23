"""Runs pool across case sets and across an identity bump, and a run kept out of a pool is named with its reason.

More cases, or more repeats, of an unchanged variant and context are more samples of one cell, never a condition of
their own: the case set is no part of the context key, and the cell is read at this build's identity version, so a
run stamped before the bump pools with one stamped after it. A run whose context, template or record keeps it out
of the pool stays in its contestant's point, as cases of their own, and the point names it with why, on the analysis
bundle and in the campaign bundle the memo writer reads. No run leaves the counts unsaid.
"""

from __future__ import annotations

from typing import Any, Literal

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.campaign_bundle import writer_view
from threetears.evals.kernel.identity import IDENTITY_VERSION, derive_context_identity, derive_variant_identity
from threetears.evals.schema.models import EvalResult, EvalRun, RoleUsage
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import make_campaign, make_eval_run, make_scored_result
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

PROFILE = toyhost_profile()


def _launched(run_id: str, **fields: Any) -> EvalRun:
    """A run as the launch stamps it: its context identity and its variant's lever map, under this build."""
    run = make_eval_run(id=run_id, status="completed", **fields)
    context = derive_context_identity(run, PROFILE)
    variant = derive_variant_identity(run=run, profile=PROFILE)
    return run.model_copy(
        update={
            "context_key": context.context_key,
            "context_components": context.context_components,
            "identity_version": IDENTITY_VERSION,
            "variant_levers": variant.levers,
        }
    )


def _results(
    run: EvalRun,
    outcomes: dict[str, tuple[bool, ...]],
    *,
    identity_version: int,
    served: str | None | Literal["no call"] = "no call",
) -> list[EvalResult]:
    """One result per attempt, each case's attempts in order, stamped with the run's variant key.

    ``served`` is the model the provider's response named for each candidate call: a name, ``None`` for a response
    that named none, or ``"no call"`` for results that carry no candidate usage row at all.
    """
    key = derive_variant_identity(run=run, profile=PROFILE).variant_key
    usage = (
        []
        if served == "no call"
        else [RoleUsage(role="candidate", model=run.candidate_model, cost_usd=0.01, call_count=1, served_model=served)]
    )
    return [
        make_scored_result(test_case_id=case, run_id=run.id, k=k, goal_passes=(passed,)).model_copy(
            update={"variant_key": key, "identity_version": identity_version, "scope_id": run.scope_id, "usage": usage}
        )
        for case, attempts in outcomes.items()
        for k, passed in enumerate(attempts, start=1)
    ]


def _bundle(runs: list[EvalRun], results: dict[str, list[EvalResult]]):
    campaign = make_campaign(
        scope_id=runs[0].scope_id,
        name="pooling",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[run.id for run in runs],
        created_by="test:fixture",
    )
    return assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=PROFILE)


def test_two_runs_over_different_case_sets_pool_into_one_cell_covering_the_union_of_cases():
    first = _launched("run-a", test_case_ids=["tc1", "tc2"])
    top_up = _launched("run-b", test_case_ids=["tc2", "tc3"])
    results = {
        first.id: _results(first, {"tc1": (True,), "tc2": (True,)}, identity_version=IDENTITY_VERSION),
        top_up.id: _results(top_up, {"tc2": (False,), "tc3": (True,)}, identity_version=IDENTITY_VERSION),
    }

    [point] = _bundle([first, top_up], results).frontier.subjects[0].points

    assert point.pooled_run_ids == ["run-a", "run-b"]
    assert point.unpooled_runs == []
    # Three cases, not four: tc2's two attempts are one case's, so pass^1 averages tc1, tc2 (one of two) and tc3.
    assert point.pass_hat_k_curve[0] == {"k": 1, "pass_hat_k": 5 / 6, "n_cases": 3}


def test_a_run_stamped_before_the_bump_pools_with_one_stamped_after_it():
    before = _launched("run-before", test_case_ids=["tc1"])
    before = before.model_copy(
        update={"context_key": "an-older-predicates-key", "identity_version": IDENTITY_VERSION - 1}
    )
    after = _launched("run-after", test_case_ids=["tc1"])
    results = {
        before.id: _results(before, {"tc1": (True,)}, identity_version=IDENTITY_VERSION - 1),
        after.id: _results(after, {"tc1": (False,)}, identity_version=IDENTITY_VERSION),
    }

    frontier = _bundle([before, after], results).frontier
    [point] = frontier.subjects[0].points

    assert point.pooled_run_ids == ["run-after", "run-before"]
    assert point.variant_identity_version == IDENTITY_VERSION
    assert point.identity_version_disclosure is None
    assert frontier.identity_version_span == [IDENTITY_VERSION]
    # One case measured twice: pass^2 reads it.
    assert point.pass_hat_k_curve == [
        {"k": 1, "pass_hat_k": 0.5, "n_cases": 1},
        {"k": 2, "pass_hat_k": 0.0, "n_cases": 1},
    ]


def test_a_campaign_names_every_run_kept_out_of_a_pool_with_its_reason_and_loses_none():
    pooled = [
        _launched("run-a", test_case_ids=["tc1", "tc2"]),
        _launched("run-b", test_case_ids=["tc3"]),
        _launched("run-c", test_case_ids=["tc1"]),
    ]
    changed = _launched("run-rig", test_case_ids=["tc1"], apparatus_settings={"reviewer_pool": "pool-b"})
    other_template = _launched("run-tpl", test_case_ids=["tc1"], template_id="tpl-2")
    unstamped = make_eval_run(id="run-bare", status="completed", test_case_ids=["tc1"])
    runs = [*pooled, changed, other_template, unstamped]
    results = {
        run.id: _results(run, dict.fromkeys(run.test_case_ids, (True,)), identity_version=IDENTITY_VERSION)
        for run in runs
    }

    bundle = _bundle(runs, results)
    [point] = bundle.frontier.subjects[0].points

    assert point.pooled_run_ids == ["run-a", "run-b", "run-c"]
    reasons = {entry.run_id: entry.reason for entry in point.unpooled_runs}
    assert reasons == {
        "run-rig": "changed_context",
        "run-tpl": "other_template",
        "run-bare": "no_recorded_context",
    }
    details = {entry.run_id: entry.detail for entry in point.unpooled_runs}
    assert "apparatus_settings" in details["run-rig"]
    assert "tpl-2" in details["run-tpl"]
    # No run vanishes: every run behind the point is pooled or named.
    assert sorted([*point.pooled_run_ids, *reasons]) == sorted(run.id for run in runs)

    view = writer_view(campaign_bundle(bundle))
    named = {
        entry["run_id"]: entry["reason"]
        for aspect in view["aspects"]
        for entries in aspect["disclosures"]["pass_k_unpooled"].values()
        for entry in entries
    }
    assert named == reasons


def test_two_runs_of_one_context_answered_by_different_served_models_are_two_cells_and_the_reason_is_named():
    """The model that answered is what was measured, whatever id was asked for: a re-pointed alias splits the cell."""
    first = _launched("run-a", test_case_ids=["tc1"])
    second = _launched("run-b", test_case_ids=["tc1"])
    repointed = _launched("run-c", test_case_ids=["tc1"])
    results = {
        first.id: _results(first, {"tc1": (True,)}, identity_version=IDENTITY_VERSION, served="model-2026-09"),
        second.id: _results(second, {"tc1": (True,)}, identity_version=IDENTITY_VERSION, served="model-2026-09"),
        repointed.id: _results(repointed, {"tc1": (False,)}, identity_version=IDENTITY_VERSION, served="model-2026-10"),
    }

    [point] = _bundle([first, second, repointed], results).frontier.subjects[0].points

    assert point.pooled_run_ids == ["run-a", "run-b"]
    [kept_out] = point.unpooled_runs
    assert (kept_out.run_id, kept_out.reason) == ("run-c", "other_served_model")
    assert "model-2026-10" in kept_out.detail and "model-2026-09" in kept_out.detail
    # tc1 under the other model is a case of its own beside the cell's, never more depth of it.
    assert point.pass_hat_k_curve[0]["n_cases"] == 2


def test_a_run_whose_responses_named_no_model_pools_and_its_identity_is_disclosed_once():
    """Not evidence of another model, so it pools; the record's silence is said where model identity always is."""
    named = _launched("run-a", test_case_ids=["tc1"])
    silent = _launched("run-b", test_case_ids=["tc1"])
    results = {
        named.id: _results(named, {"tc1": (True,)}, identity_version=IDENTITY_VERSION, served="model-2026-09"),
        silent.id: _results(silent, {"tc1": (False,)}, identity_version=IDENTITY_VERSION, served=None),
    }

    bundle = _bundle([named, silent], results)
    [point] = bundle.frontier.subjects[0].points

    assert point.pooled_run_ids == ["run-a", "run-b"]
    assert point.unpooled_runs == []
    assert point.pass_hat_k_curve[-1] == {"k": 2, "pass_hat_k": 0.0, "n_cases": 1}
    assert bundle.model_identity_disclosure is not None
    assert bundle.model_identity_disclosure.startswith("Model identity unconfirmed")
    view = writer_view(campaign_bundle(bundle))
    [aspect] = view["aspects"]
    assert aspect["disclosures"]["model_identity_disclosure"] == bundle.model_identity_disclosure
    assert "pass_k_unpooled" not in aspect["disclosures"]
