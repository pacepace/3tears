"""Runs pool across case sets and across an identity bump, and a run kept out of a pool is named with its reason.

More cases, or more repeats, of an unchanged variant and context are more samples of one cell, never a condition of
their own: the case set is no part of the context key, and the cell is read at this build's identity version, so a
run stamped before the bump pools with one stamped after it. A contestant's pass^k reads its headline cell alone; a
run of another context, template revision or served model, or one that recorded no context, stands beside it in a
cell of its own, with its own reading and its reason, on the analysis bundle, in the campaign bundle the memo writer
reads and in the frontier's text. No run leaves the counts unsaid.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from threetears.evals.analysis import FrontierPoint, PassHatKCellReading, assemble_context_bundle
from threetears.evals.analysis.campaign_bundle import writer_view
from threetears.evals.analysis.lens_text import frontier_text
from threetears.evals.analysis.lenses.frontier import compute_frontier
from threetears.evals.kernel.identity import IDENTITY_VERSION, derive_context_identity, derive_variant_identity
from threetears.evals.schema.models import EvalResult, EvalRun, RoleUsage
from threetears.evals.schema.values import NominalScale, SweepableValue
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import DEFAULT_SUBJECT, make_campaign, make_eval_run, make_scored_result
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
    cost: float = 0.01,
) -> list[EvalResult]:
    """One result per attempt, each case's attempts in order, stamped with the run's variant key.

    ``served`` is the model the provider's response named for each candidate call: a name, ``None`` for a response
    that named none, or ``"no call"`` for results that carry no candidate usage row at all.
    """
    key = derive_variant_identity(run=run, profile=PROFILE).variant_key
    usage = (
        []
        if served == "no call"
        else [RoleUsage(role="candidate", model=run.candidate_model, cost_usd=cost, call_count=1, served_model=served)]
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


def _cells(point: FrontierPoint) -> list[PassHatKCellReading]:
    """The point's cells, the headline first, which every point carries at least one of."""
    assert point.pass_hat_k_cells and point.pass_hat_k_cells[0].headline
    return point.pass_hat_k_cells


def test_two_runs_over_different_case_sets_pool_into_one_cell_covering_the_union_of_cases():
    first = _launched("run-a", test_case_ids=["tc1", "tc2"])
    top_up = _launched("run-b", test_case_ids=["tc2", "tc3"])
    results = {
        first.id: _results(first, {"tc1": (True,), "tc2": (True,)}, identity_version=IDENTITY_VERSION),
        top_up.id: _results(top_up, {"tc2": (False,), "tc3": (True,)}, identity_version=IDENTITY_VERSION),
    }

    [point] = _bundle([first, top_up], results).frontier.subjects[0].points

    [headline] = _cells(point)
    assert headline.run_ids == ["run-a", "run-b"]
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

    [headline] = _cells(point)
    assert headline.run_ids == ["run-after", "run-before"]
    assert point.variant_identity_version == IDENTITY_VERSION
    assert point.identity_version_disclosure is None
    assert frontier.identity_version_span == [IDENTITY_VERSION]
    # One case measured twice: pass^2 reads it.
    assert point.pass_hat_k_curve == [
        {"k": 1, "pass_hat_k": 0.5, "n_cases": 1},
        {"k": 2, "pass_hat_k": 0.0, "n_cases": 1},
    ]


def _mixed_campaign():
    """Three runs that pool, two that pool with each other on another rig, and one each of another template and none.

    Every pooled run passes and every other run fails, so a headline counting any run kept out would read below 1.
    """
    pooled = [
        _launched("run-a", test_case_ids=["tc1", "tc2"]),
        _launched("run-b", test_case_ids=["tc3"]),
        _launched("run-c", test_case_ids=["tc1"]),
    ]
    rig = [
        _launched(run_id, test_case_ids=["tc1"], apparatus_settings={"reviewer_pool": "pool-b"})
        for run_id in ("run-rig-1", "run-rig-2")
    ]
    other_template = _launched("run-tpl", test_case_ids=["tc1"], template_id="tpl-2")
    unstamped = make_eval_run(id="run-bare", status="completed", test_case_ids=["tc1"])
    runs = [*pooled, *rig, other_template, unstamped]
    results = {
        run.id: _results(run, dict.fromkeys(run.test_case_ids, (run in pooled,)), identity_version=IDENTITY_VERSION)
        for run in runs
    }
    return runs, results


def test_a_campaign_names_every_cell_kept_out_of_a_pool_with_its_reason_and_loses_no_run():
    runs, results = _mixed_campaign()

    bundle = _bundle(runs, results)
    [point] = bundle.frontier.subjects[0].points

    headline, *beside = _cells(point)
    assert headline.run_ids == ["run-a", "run-b", "run-c"]
    by_runs = {tuple(cell.run_ids): cell for cell in beside}
    # The two runs on the other rig pooled with each other: one cell of two, never two runs called unpooled.
    assert {runs: cell.reason for runs, cell in by_runs.items()} == {
        ("run-rig-1", "run-rig-2"): "changed_context",
        ("run-tpl",): "other_template",
        ("run-bare",): "no_recorded_context",
    }
    assert "apparatus_settings" in (by_runs[("run-rig-1", "run-rig-2")].detail or "")
    assert "tpl-2" in (by_runs[("run-tpl",)].detail or "")
    # No run vanishes: every run behind the point is in exactly one cell.
    assert sorted(run_id for cell in _cells(point) for run_id in cell.run_ids) == sorted(run.id for run in runs)

    view = writer_view(campaign_bundle(bundle))
    [aspect] = view["aspects"]
    unpooled = aspect["disclosures"]["pass_k_unpooled"]
    assert "headline" in unpooled["reading"]
    lines = [line for model_lines in unpooled["by_model"].values() for line in model_lines]
    for cell in beside:
        assert any(", ".join(cell.run_ids) in line and (cell.detail or "") in line for line in lines)


def test_the_headline_pass_hat_k_reads_the_pooled_runs_alone():
    """A run kept out of the pool counts in neither the headline, its interval nor its verdict."""
    runs, results = _mixed_campaign()

    [point] = _bundle(runs, results).frontier.subjects[0].points

    assert point.pass_hat_k == 1.0
    assert point.n_pass_cases == 3
    assert point.pass_hat_k_ci_high == 1.0
    for cell in _cells(point)[1:]:
        assert cell.pass_hat_k == 0.0, "each cell kept out keeps its own reading, beside the headline"


def test_the_frontier_text_names_each_cell_kept_out_and_why():
    runs, results = _mixed_campaign()

    text = frontier_text(_bundle(runs, results).frontier)

    assert "run(s) run-rig-1, run-rig-2 kept out of this pass^k, as a cell of their own (another measurement" in text
    assert "run(s) run-tpl kept out of this pass^k, as a cell of their own (another template" in text
    assert "run(s) run-bare kept out of this pass^k, as a cell of their own (no recorded measurement context)" in text


def test_a_template_edited_in_what_it_measures_splits_the_cell_and_a_run_that_froze_no_digest_joins_it():
    """An edit in place keeps the template's id; the digest each run froze says what it measured."""
    first = _launched("run-a", test_case_ids=["tc1"], template_digest="digest-before")
    second = _launched("run-b", test_case_ids=["tc2"], template_digest="digest-before")
    older = _launched("run-old", test_case_ids=["tc3"], template_digest=None)
    edited = _launched("run-edited", test_case_ids=["tc1"], template_digest="digest-after")
    runs = [first, second, older, edited]
    results = {
        run.id: _results(run, dict.fromkeys(run.test_case_ids, (True,)), identity_version=IDENTITY_VERSION)
        for run in runs
    }

    [point] = _bundle(runs, results).frontier.subjects[0].points

    headline, beside = _cells(point)
    assert headline.run_ids == ["run-a", "run-b", "run-old"]
    assert headline.template_digest_assumed == ["run-old"]
    assert (beside.run_ids, beside.reason) == (["run-edited"], "other_template")
    assert "edited in what it measures" in (beside.detail or "")


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

    headline, beside = _cells(point)
    assert headline.run_ids == ["run-a", "run-b"]
    assert (beside.run_ids, beside.reason) == (["run-c"], "other_served_model")
    assert "model-2026-10" in (beside.detail or "") and "model-2026-09" in (beside.detail or "")
    # The other model's attempt at tc1 is not more depth of the headline's, and not part of its reading.
    assert point.pass_hat_k_curve == [
        {"k": 1, "pass_hat_k": 1.0, "n_cases": 1},
        {"k": 2, "pass_hat_k": 1.0, "n_cases": 1},
    ]


def test_a_run_whose_responses_named_no_model_pools_and_is_named_as_joining_by_assumption():
    """Not evidence of another model, so it pools; the record's silence is said where model identity always is."""
    named = _launched("run-a", test_case_ids=["tc1"])
    silent = _launched("run-b", test_case_ids=["tc1"])
    results = {
        named.id: _results(named, {"tc1": (True,)}, identity_version=IDENTITY_VERSION, served="model-2026-09"),
        silent.id: _results(silent, {"tc1": (False,)}, identity_version=IDENTITY_VERSION, served=None),
    }

    bundle = _bundle([named, silent], results)
    [point] = bundle.frontier.subjects[0].points

    [headline] = _cells(point)
    assert headline.run_ids == ["run-a", "run-b"]
    assert headline.served_model_assumed == ["run-b"]
    assert point.pass_hat_k_curve[-1] == {"k": 2, "pass_hat_k": 0.0, "n_cases": 1}
    assert bundle.model_identity_disclosure is not None
    assert bundle.model_identity_disclosure.startswith("Model identity unconfirmed")
    view = writer_view(campaign_bundle(bundle))
    [aspect] = view["aspects"]
    assert aspect["disclosures"]["model_identity_disclosure"] == bundle.model_identity_disclosure
    [line] = aspect["disclosures"]["pass_k_unpooled"]["by_model"][point.model]
    assert line == "run(s) run-b named no served model and pooled with run(s) run-a, run-b"


def test_a_run_whose_context_cannot_be_rederived_is_a_cell_of_its_own_and_the_rest_still_read():
    """A stored run naming subject state its host no longer declares stands apart, named; it fails nothing else."""
    kept = _launched("run-a", test_case_ids=["tc1", "tc2"])
    undeclared = SweepableValue(content_hash="0" * 64, display="retired", scale=NominalScale())
    stale = _launched("run-stale", test_case_ids=["tc1"]).model_copy(
        update={
            "subject_snapshot": DEFAULT_SUBJECT.model_copy(update={"state": {"retired_input": undeclared}}),
            "identity_version": IDENTITY_VERSION - 1,
        }
    )
    results = {
        kept.id: _results(kept, {"tc1": (True,), "tc2": (True,)}, identity_version=IDENTITY_VERSION),
        stale.id: _results(stale, {"tc1": (False,)}, identity_version=IDENTITY_VERSION),
    }

    frontier = compute_frontier(
        [kept, stale], [*results[kept.id], *results[stale.id]], profile=PROFILE, archived_run_ids=None
    )

    [point] = frontier.subjects[0].points
    headline, beside = _cells(point)
    assert headline.run_ids == ["run-a"]
    assert (beside.run_ids, beside.reason) == (["run-stale"], "changed_context")
    assert "retired_input" in (beside.detail or "")
    assert point.pass_hat_k == 1.0


def _silences_corpus(*others: EvalRun, served: dict[str, str | None]):
    """Two runs that froze a template digest and named their served model, beside the runs under test."""
    headline = [_launched(run_id, test_case_ids=["tc1"], template_digest="digest-1") for run_id in ("run-a", "run-b")]
    runs = [*headline, *others]
    results = {
        run.id: _results(run, {"tc1": (True,)}, identity_version=IDENTITY_VERSION, served=served.get(run.id, "model-1"))
        for run in runs
    }
    return runs, results


def test_a_run_that_froze_no_digest_on_another_rig_reads_the_rig_not_a_template_edit():
    rig = _launched("run-rig", test_case_ids=["tc1"], apparatus_settings={"reviewer_pool": "pool-b"})
    runs, results = _silences_corpus(rig, served={})

    [point] = _bundle(runs, results).frontier.subjects[0].points

    headline, beside = _cells(point)
    assert headline.run_ids == ["run-a", "run-b"]
    assert (beside.run_ids, beside.reason) == (["run-rig"], "changed_context")
    assert "apparatus_settings" in (beside.detail or "")


def test_a_run_that_froze_no_digest_answered_by_another_model_reads_the_served_model():
    other = _launched("run-other", test_case_ids=["tc1"])
    runs, results = _silences_corpus(other, served={"run-other": "model-2"})

    [point] = _bundle(runs, results).frontier.subjects[0].points

    headline, beside = _cells(point)
    assert (beside.run_ids, beside.reason) == (["run-other"], "other_served_model")


def test_a_run_silent_on_both_its_digest_and_its_served_model_joins_its_cell():
    silent = _launched("run-silent", test_case_ids=["tc2"])
    runs, results = _silences_corpus(silent, served={"run-silent": None})

    [point] = _bundle(runs, results).frontier.subjects[0].points

    [headline] = _cells(point)
    assert headline.run_ids == ["run-a", "run-b", "run-silent"]
    assert headline.template_digest_assumed == ["run-silent"]
    assert headline.served_model_assumed == ["run-silent"]


def test_the_cost_per_acceptable_outcome_reads_the_headline_cells_runs_on_both_sides():
    """pass^1 is the headline cell's, so the cost it divides is too; the point's cost still pools every run."""
    pooled = [_launched(run_id, test_case_ids=["tc1", "tc2"]) for run_id in ("run-a", "run-b")]
    rig = _launched("run-rig", test_case_ids=["tc1", "tc2"], apparatus_settings={"reviewer_pool": "pool-b"})
    results = {
        **{
            run.id: _results(
                run, {"tc1": (True,), "tc2": (False,)}, identity_version=IDENTITY_VERSION, served="m", cost=0.01
            )
            for run in pooled
        },
        rig.id: _results(
            rig, {"tc1": (True,), "tc2": (True,)}, identity_version=IDENTITY_VERSION, served="m", cost=1.0
        ),
    }

    [point] = _bundle([*pooled, rig], results).frontier.subjects[0].points

    assert point.pass_hat_k_curve[0]["pass_hat_k"] == 0.5
    assert point.cost_per_acceptable_outcome == pytest.approx(0.01 / 0.5)
    assert point.production_replicating_cost == pytest.approx((4 * 0.01 + 2 * 1.0) / 6)


def test_a_run_silent_on_both_takes_one_peers_whole_pair_never_two_majorities():
    """Three runs share digest d2 but no served model; two share (d1, s1). Silent X joins those two, not (d2, s1)."""
    pairs = {
        "run-a": ("d1", "s1"),
        "run-b": ("d1", "s1"),
        "run-c": ("d2", "s2"),
        "run-d": ("d2", "s3"),
        "run-e": ("d2", "s4"),
    }
    runs = [_launched(run_id, test_case_ids=["tc1"], template_digest=digest) for run_id, (digest, _) in pairs.items()]
    silent = _launched("run-x", test_case_ids=["tc1"])
    results = {
        run.id: _results(run, {"tc1": (True,)}, identity_version=IDENTITY_VERSION, served=pairs[run.id][1])
        for run in runs
    }
    results[silent.id] = _results(silent, {"tc1": (True,)}, identity_version=IDENTITY_VERSION, served=None)

    [point] = _bundle([*runs, silent], results).frontier.subjects[0].points

    headline = _cells(point)[0]
    assert headline.run_ids == ["run-a", "run-b", "run-x"]
    assert headline.template_digest_assumed == ["run-x"]
    assert headline.served_model_assumed == ["run-x"]
