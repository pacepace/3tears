"""An operator's own edits to a stored scenario, through the courier host: the two things only ``update_template`` does.

The courier host (``fixtures/courierhost``) is a shipped example, and these are the operator steps the host guide
promises: renaming a stratum that already has a name (the seed and a reset only fill a name a template lacks), and
retiring a scenario outright (a reset archives only a revision it supersedes). Each is driven through the fixture's
function over a campaign the courier ran, and read back the way a host reads it: the campaign's bundle and the
template listing.
"""

from __future__ import annotations

from collections import Counter

from threetears.evals.analysis import inspect_campaign_bundle
from threetears.evals.kernel import EvalCampaign
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import templates_list
from threetears.evals.run import RunnerOptions, execute_run, run_blocking

from packages.evals.tests.fixtures.courierhost import (
    COURIER_KIND,
    COURIER_MODELS,
    COURIER_SCOPE,
    COURIER_TEMPLATE_ID,
    RoutePlannerKind,
    courier_cases,
    courier_launch_host,
    courier_run,
    courier_template,
    rename_stratum,
    retire_scenario,
)
from packages.evals.tests.off_loop import off_loop

#: The names the scenario is authored with: rounds of four stops are short, the rest long.
AUTHORED_NAMES = {"short": "Short rounds", "long": "Long rounds"}


async def _stratified_campaign() -> tuple[EvalHost, EvalCampaign]:
    """The courier's bake-off over its scenario with named strata, in the launching host's store."""
    host = off_loop(courier_launch_host).eval_host
    world = host.profile.world
    assert world is not None
    kind = RoutePlannerKind(world)
    template = courier_template().model_copy(update={"stratum_names": AUTHORED_NAMES})
    cases = [
        case.model_copy(update={"stratum": "short" if case.variation_params["stops"] == "4" else "long"})
        for case in courier_cases()
    ]
    await run_blocking(host.blocking_executor, host.storage.save_template, template)
    for case in cases:
        await run_blocking(host.blocking_executor, host.storage.save_test_case, case)
    run_ids = []
    for model in COURIER_MODELS:
        run = courier_run(model, cases=cases, world=world)
        await run_blocking(host.blocking_executor, host.storage.save_eval_run, run)
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=cases,
            judge_service=None,
            options=RunnerOptions(candidate_kinds={COURIER_KIND: lambda _cell: kind}),
        )
        await run_blocking(
            host.blocking_executor, host.storage.save_eval_run, run.model_copy(update={"status": "completed"})
        )
        run_ids.append(run.id)
    campaign = EvalCampaign.model_validate(
        {
            "scope_id": COURIER_SCOPE,
            "name": "planner model bake-off, by round length",
            "subject_id": "depot-north-planner",
            "subject_kind": "route_planner_config",
            "behavior": "plan_delivery_round",
            "aspects": [{"label": "plan_delivery_round", "template_id": template.id, "run_ids": run_ids}],
            "created_by": "test:fixture",
        }
    )
    await run_blocking(host.blocking_executor, host.storage.save_campaign, campaign)
    return host, campaign


async def test_a_renamed_stratum_reads_by_its_new_name_with_every_result_where_it_was() -> None:
    host, campaign = await _stratified_campaign()
    before = off_loop(inspect_campaign_bundle, host, campaign.id, COURIER_SCOPE).bundle
    stored = off_loop(host.storage.load_template, COURIER_TEMPLATE_ID, COURIER_SCOPE)
    assert stored is not None
    assert before.stratum_names == AUTHORED_NAMES

    renamed = off_loop(rename_stratum, host, COURIER_TEMPLATE_ID, "short", "Quick rounds")

    after = off_loop(inspect_campaign_bundle, host, campaign.id, COURIER_SCOPE).bundle
    assert after.stratum_names == {"short": "Quick rounds", "long": "Long rounds"}, "the bundle reads the new name"
    # Nothing orphaned or counted twice: the same runs, every trial under the stratum it ran in.
    assert after.run_ids == before.run_ids and not after.unresolved_run_ids
    by_stratum = Counter(trial.stratum for trial in after.trials)
    assert by_stratum == Counter(trial.stratum for trial in before.trials) == {"short": 4, "long": 8}
    # A name is presentation only: no fingerprint moves, and the template is the same revision, unstamped.
    assert after.fingerprint() == before.fingerprint()
    assert (renamed.id, renamed.revision, renamed.updated_at) == (stored.id, stored.revision, stored.updated_at)
    listing = off_loop(templates_list, host, COURIER_SCOPE)
    assert [(line.id, line.revision) for line in listing.templates] == [(COURIER_TEMPLATE_ID, stored.revision)]


async def test_a_retired_scenario_leaves_the_listing_and_its_runs_still_read() -> None:
    host, campaign = await _stratified_campaign()
    before = off_loop(inspect_campaign_bundle, host, campaign.id, COURIER_SCOPE).bundle

    retired = off_loop(retire_scenario, host, COURIER_TEMPLATE_ID)

    assert retired.archived
    active = off_loop(templates_list, host, COURIER_SCOPE)
    archived = off_loop(templates_list, host, COURIER_SCOPE, archived=True)
    assert COURIER_TEMPLATE_ID not in [line.id for line in active.templates], "a retired scenario is still offered"
    assert [line.id for line in archived.templates] == [COURIER_TEMPLATE_ID]
    after = off_loop(inspect_campaign_bundle, host, campaign.id, COURIER_SCOPE).bundle
    assert after.run_ids == before.run_ids and not after.unresolved_run_ids and not after.archived_run_ids
    assert len(after.trials) == len(before.trials) == 12
    assert after.stratum_names == AUTHORED_NAMES
