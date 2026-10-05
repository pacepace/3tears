"""A launch sets a host-declared apparatus value, so one template can be compared at two of them.

An adjudicator's seat, a reviewer pool: a setup value of the measuring rig that a host declares as
apparatus and its kind's launcher builds the rig from. Before ``apparatus_settings`` it could only ride
in a template's ``kind_spec``, so comparing two values took two templates. These pin:

- **The value reaches the launcher and the run**: the kind reads it off its request, and the run records
  exactly what the launch set (``EvalRun.apparatus_settings``).
- **It is part of the measurement context**: two runs at two values are two conditions (their context
  keys differ in the ``apparatus_settings`` component) and the same candidate (their variant keys agree).
- **Every refusal fires before anything is built**: a setting the template's kind does not read, a value
  the run could not store; and at registration, a kind claiming to read a setting the host does not
  declare as its own apparatus.
- **Every launch surface carries it** — ``start_run``, the battery, ``run_launch`` (ops, action and the
  FastMCP tool) and the CLI's ``run``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest
from fastmcp import Client, FastMCP

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.contracts import IDENTITY_VERSION, EvalRun, EvalStorage, ValidationFailedError
from threetears.evals.contracts.identity import derive_context_identity, derive_variant_identity
from threetears.evals.ops import LaunchArguments, run_launch
from threetears.evals.quick import run_cli
from threetears.evals.run import (
    LaunchableKind,
    LaunchHost,
    LaunchRequest,
    settable_apparatus,
    start_run,
    start_universal_battery,
)
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.transports.fastmcp import mount_fastmcp
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.courierhost import (
    COURIER_SCOPE,
    COURIER_SUBJECT,
    COURIER_TEMPLATE_ID,
    courier_launch_host,
)
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_REVIEWER_POOL, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import CALLER, ops_fixture

#: The toy host's one launch-settable apparatus dimension: who reviews the extractions.
POOL = "reviewer_pool"


def _launching() -> tuple[LaunchHost, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    host, _client = toyhost_launch_host(storage=storage)
    return host, storage


async def _settled(host: LaunchHost, run_ids: list[str]) -> None:
    async with asyncio.timeout(10):
        while any(host.job_manager.is_active(run_id) for run_id in run_ids):
            await asyncio.sleep(0.01)


async def _launch(host: LaunchHost, **arguments: Any) -> list[EvalRun]:
    launched: dict[str, Any] = {"models": [RUN_MODELS[0]], **arguments}
    runs = await start_run(
        host,
        template_id=toyhost_template().id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        **launched,
    )
    await _settled(host, [run.id for run in runs])
    return runs


# =============================================================================
# The value reaches the launcher and the run, and is part of the measurement context
# =============================================================================


async def test_a_launchs_apparatus_setting_reaches_its_launcher_and_its_run():
    host, storage = _launching()

    (run,) = await _launch(host, apparatus_settings={POOL: "pool-b"})

    stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
    assert stored is not None and stored.status == "completed"
    assert stored.apparatus_settings == {POOL: "pool-b"}
    assert stored.host_payload["toyhost"][POOL] == "pool-b", "the launcher set its rig up from the request"
    assert host.eval_host.profile.sweepables.read_all(stored)[POOL] == "pool-b", "the host's reader reads the rig"


async def test_a_launch_setting_none_records_none_and_the_host_reviews_with_its_standing_pool():
    host, storage = _launching()

    (run,) = await _launch(host)

    stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
    assert stored is not None and stored.apparatus_settings == {}
    assert stored.host_payload["toyhost"][POOL] == TOYHOST_REVIEWER_POOL


async def test_two_values_of_one_template_are_two_conditions_of_one_candidate():
    """One template, two reviewer pools: never pooled as repeats, and still the same arm."""
    host, _storage = _launching()

    (pool_a,) = await _launch(host, apparatus_settings={POOL: "pool-a"})
    (pool_b,) = await _launch(host, apparatus_settings={POOL: "pool-b"})
    (unset,) = await _launch(host)

    profile = host.eval_host.profile
    assert len({pool_a.context_key, pool_b.context_key, unset.context_key}) == 3
    assert pool_a.context_components is not None and pool_b.context_components is not None
    differing = {
        name
        for name, value in pool_a.context_components.model_dump().items()
        if value != pool_b.context_components.model_dump()[name]
    }
    assert differing == {"apparatus_settings"}, "the component names what moved"
    assert pool_a.identity_version == IDENTITY_VERSION == 23
    variant = {derive_variant_identity(run=run, profile=profile).variant_key for run in (pool_a, pool_b, unset)}
    assert len(variant) == 1, "an apparatus value is the rig, not the candidate"


def test_the_component_hashes_the_values_and_their_types():
    """``"1"`` and ``1`` are two levels; the component is composed for a run that set none."""
    from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

    profile = toyhost_profile()
    key = {
        label: derive_context_identity(make_eval_run(apparatus_settings=settings), profile).context_components
        for label, settings in [("text", {POOL: "1"}), ("number", {POOL: 1}), ("none", {})]
    }
    assert key["text"].apparatus_settings != key["number"].apparatus_settings
    assert key["none"].apparatus_settings is not None, "no settings is a level, composed like any other"


# =============================================================================
# The refusals, each before anything is built
# =============================================================================


@pytest.mark.parametrize(
    ("settings", "said"),
    [
        ({"ocr_engine_version": "tess-6"}, "sets its rig up from no apparatus setting named 'ocr_engine_version'"),
        ({"judge_model": "j"}, "no apparatus setting named 'judge_model'"),
        ({"chunk_tokens": 512}, "no apparatus setting named 'chunk_tokens'"),
        ({POOL: ["pool-a", "pool-b"]}, "invalid apparatus_settings"),
        ({POOL: None}, "invalid apparatus_settings"),
        ({POOL: float("inf")}, "invalid apparatus_settings"),
    ],
    ids=["another-apparatus", "the-engines-apparatus", "a-lever", "a-list", "none", "not-finite"],
)
async def test_a_setting_the_kind_does_not_read_or_cannot_be_stored_is_refused_before_its_launcher_runs(settings, said):
    host, storage = _launching()
    handed: list[LaunchRequest] = []
    (launchable,) = host.kinds.values()

    async def launch(request: LaunchRequest) -> EvalRun:
        handed.append(request)
        return await launchable.launch(request)

    watched = replace(host, kinds={TOY_EXTRACTOR_KIND: replace(launchable, launch=launch)})

    with pytest.raises(ValidationFailedError, match=said):
        await _launch(watched, apparatus_settings=settings)

    assert handed == [] and storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert watched.job_manager.admitted_count == 0


@pytest.mark.parametrize(
    ("claimed", "said"),
    [
        ("no_such_dimension", "does not declare as apparatus of its own"),
        ("judge_model", "does not declare as apparatus of its own"),
        ("chunk_tokens", "does not declare as apparatus of its own"),
        ("batch_label", "does not declare as apparatus of its own"),
    ],
    ids=["undeclared", "the-engines-own", "a-lever", "a-label"],
)
def test_a_kind_claiming_a_setting_the_host_does_not_declare_as_its_own_apparatus_is_refused(claimed, said):
    host, _storage = _launching()
    (launchable,) = host.kinds.values()

    with pytest.raises(ValueError, match=said):
        replace(host, kinds={TOY_EXTRACTOR_KIND: replace(launchable, apparatus_settings=frozenset({claimed}))})


def test_the_settable_apparatus_is_the_hosts_own_and_none_of_the_engines():
    host, _storage = _launching()
    settable = settable_apparatus(host.eval_host.profile.sweepables)
    assert {"reviewer_pool", "ocr_engine_version", "grader_version"} <= settable
    assert settable.isdisjoint({"judge_model", "simulator_model", "max_cost_usd", "judge_config_ids"})
    assert "chunk_tokens" not in settable and "batch_label" not in settable


# =============================================================================
# Every launch surface carries it
# =============================================================================


async def test_a_battery_sets_every_templates_runs_up_with_it_and_refuses_one_no_kind_reads():
    host, storage = _launching()
    storage.save_template(toyhost_template().model_copy(update={"universal": True}))

    async def preflight(_subject_id: str, _models: Any) -> Any:
        async def check(_template: Any, _cassette_mode: str) -> None:
            return None

        return check

    with pytest.raises(
        ValidationFailedError, match="no apparatus setting named 'grader_version'.*Nothing was launched"
    ):
        await start_universal_battery(
            host,
            TOYHOST_SUBJECT.subject_id,
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            apparatus_settings={"grader_version": "g9"},
            preflight=preflight,
        )
    assert storage.query_eval_runs(TOYHOST_SCOPE) == []

    run_ids = await start_universal_battery(
        host,
        TOYHOST_SUBJECT.subject_id,
        scope_id=TOYHOST_SCOPE,
        models=[RUN_MODELS[0]],
        apparatus_settings={POOL: "pool-c"},
        preflight=preflight,
    )
    await _settled(host, run_ids)
    assert [storage.load_eval_run(run_id, TOYHOST_SCOPE).apparatus_settings for run_id in run_ids] == [  # type: ignore[union-attr]
        {POOL: "pool-c"}
    ]


async def test_run_launch_hands_the_launch_its_apparatus_settings():
    fixture = ops_fixture()
    launched = LaunchArguments(
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
        apparatus_settings={POOL: "pool-d"},
    )

    started = await run_launch(fixture.host, launched, TOYHOST_SCOPE)

    (job,) = started.jobs
    await _settled(fixture.host.launch, [job.target_id])
    stored = fixture.host.eval_host.storage.load_eval_run(job.target_id, TOYHOST_SCOPE)
    assert stored is not None and stored.apparatus_settings == {POOL: "pool-d"}


async def test_the_run_launch_action_hands_the_launch_its_apparatus_settings():
    fixture = ops_fixture()
    evals = eval_catalogue().mount_all(standard_tools())[0]
    call = {
        "action": "run_launch",
        "template_id": toyhost_template().id,
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "models": [RUN_MODELS[0]],
        "apparatus_settings": {"grader_version": "g9"},
    }

    outcome = await evals.call(call, host=fixture.host, caller=CALLER)

    assert outcome.is_error and "no apparatus setting named 'grader_version'" in outcome.text, outcome.text


async def test_the_fastmcp_tool_hands_the_launch_its_apparatus_settings():
    fixture = ops_fixture()
    server = FastMCP("toyhost")
    mount_fastmcp(server, eval_catalogue(), host=fixture.host, caller=lambda: CALLER)
    call = {
        "action": "run_launch",
        "template_id": toyhost_template().id,
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "models": [RUN_MODELS[0]],
        "apparatus_settings": {"grader_version": "g9"},
    }

    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        result = await client.call_tool("evals", call, raise_on_error=False)

    assert "apparatus_settings" in tools["evals"].inputSchema["properties"]
    assert result.is_error and "no apparatus setting named 'grader_version'" in result.content[0].text


def test_the_clis_run_hands_the_launch_its_apparatus_settings(capsys: pytest.CaptureFixture[str]) -> None:
    """The courier's kind reads no apparatus setting, so the launch refusing it is the proof it reached it."""
    args = ["run", "--scope", COURIER_SCOPE, "--template", COURIER_TEMPLATE_ID, "--subject", COURIER_SUBJECT.subject_id]
    given = ["--model", "planner-lite", "--apparatus-settings", json.dumps({"dispatcher": "night-shift"})]

    assert run_cli([*args, *given], host_factory=courier_launch_host) == 2

    assert "no apparatus setting named 'dispatcher'" in capsys.readouterr().err


@pytest.mark.parametrize("given", ["not json", "[1, 2]"], ids=["not-json", "not-an-object"])
def test_the_clis_apparatus_settings_must_be_a_json_object(given: str) -> None:
    args = ["run", "--scope", COURIER_SCOPE, "--template", COURIER_TEMPLATE_ID, "--subject", COURIER_SUBJECT.subject_id]
    with pytest.raises(SystemExit):
        run_cli([*args, "--apparatus-settings", given], host_factory=courier_launch_host)


def test_a_kind_reads_nothing_unless_it_says_so():
    async def launch(_request: LaunchRequest) -> EvalRun:
        raise AssertionError("never launched")

    assert LaunchableKind(launch=launch).apparatus_settings == frozenset()
