"""A host with no metered tools says so, and its runs record it rather than a ceiling that bounds nothing.

The engine required a positive ``max_metered_calls``, so a host whose tools call no paid or rationed
provider recorded a ceiling (DoW: 1) on every run that bounded nothing and read as a limit. Now
``LaunchSettings.max_metered_calls=None`` is that declaration: the run records a ceiling of ``0`` with
origin ``none_declared``, a metered call on it is refused and counted (the declaration was wrong), and a
launch naming a ceiling is refused. A host WITH metered tools is unchanged — both directions are pinned.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.contracts import EvalRun, EvalStorage, ValidationFailedError
from threetears.evals.run import KindWiring, LaunchHost, LaunchRequest, MeteredCallLedger, launch_run, start_run
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND, ScriptedExtractionClient, ToyExtractorKind
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template, toyhost_test_cases


def _host(
    *, max_metered_calls: int | None, enforcement_enabled: bool = True
) -> tuple[LaunchHost, EvalStorage, list[bool]]:
    """The toy launch host whose kind makes one metered call per cell; the admissions it got are returned."""
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    settings = TOYHOST_LAUNCH_SETTINGS.model_copy(
        update={"max_metered_calls": max_metered_calls, "enforcement_enabled": enforcement_enabled}
    )
    host, _client = toyhost_launch_host(storage=storage, settings=lambda: settings)
    world = host.eval_host.profile.world
    assert world is not None
    kind = ToyExtractorKind(
        client=ScriptedExtractionClient(), world=world, goal_checks=tuple(toyhost_template().goal_state_checks)
    )
    admitted: list[bool] = []

    def factory(context: Any) -> ToyExtractorKind:
        ledger = context.options.metered_calls
        assert ledger is not None
        admitted.append(ledger.admit(tool="web_search", action="search", spend=None))
        return kind

    async def launch(request: LaunchRequest) -> EvalRun:
        return await launch_run(
            launching,
            request,
            KindWiring(kind_factory=factory, subject=TOYHOST_SUBJECT, test_cases=toyhost_test_cases(request.template)),
        )

    (launchable,) = host.kinds.values()
    launching = replace(host, kinds={TOY_EXTRACTOR_KIND: replace(launchable, launch=launch)})
    return launching, storage, admitted


async def _launch(host: LaunchHost, storage: EvalStorage, **arguments: Any) -> EvalRun:
    (run,) = await start_run(
        host,
        template_id=toyhost_template().id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
        **arguments,
    )
    async with asyncio.timeout(10):
        while host.job_manager.is_active(run.id):
            await asyncio.sleep(0.01)
    stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
    assert stored is not None
    return stored


async def test_a_host_declaring_no_metered_tools_records_it_and_refuses_a_metered_call(caplog):
    host, storage, admitted = _host(max_metered_calls=None)

    with caplog.at_level(logging.ERROR):
        run = await _launch(host, storage)

    assert (run.max_metered_calls, run.max_metered_calls_origin) == (0, "none_declared")
    assert admitted and not any(admitted), "every metered call on a host declaring none is refused"
    assert run.metered_calls_refused == len(admitted), "and counted, so the run discloses it"
    assert "declares no metered tools" in caplog.text


async def test_a_host_with_metered_tools_records_its_ceiling_and_admits_within_it():
    host, storage, admitted = _host(max_metered_calls=100)

    run = await _launch(host, storage)

    assert (run.max_metered_calls, run.max_metered_calls_origin) == (100, "inherited")
    assert admitted and all(admitted)
    assert run.metered_calls_refused == 0


async def test_the_declaration_holds_with_enforcement_off():
    """It is a fact about the host's tools, not a ceiling enforcement switches off."""
    host, storage, admitted = _host(max_metered_calls=None, enforcement_enabled=False)

    run = await _launch(host, storage)

    assert (run.max_metered_calls, run.max_metered_calls_origin) == (0, "none_declared")
    assert not any(admitted)


async def test_with_metered_tools_and_enforcement_off_the_run_is_uncapped():
    host, storage, admitted = _host(max_metered_calls=100, enforcement_enabled=False)

    run = await _launch(host, storage)

    assert (run.max_metered_calls, run.max_metered_calls_origin) == (None, "uncapped")
    assert all(admitted)


async def test_a_launch_naming_a_ceiling_on_a_host_with_no_metered_tools_is_refused_before_anything():
    host, storage, admitted = _host(max_metered_calls=None)

    with pytest.raises(ValidationFailedError, match="declares no metered tools, so it would bound nothing"):
        await _launch(host, storage, max_metered_calls=5)

    assert admitted == [] and storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert host.job_manager.admitted_count == 0


def test_the_ledger_for_a_host_declaring_none_refuses_with_its_own_words():
    ledger = MeteredCallLedger.for_run("run-1", None, configured_max_metered_calls=None, enforcement_enabled=True)

    assert ledger.ceiling == 0
    assert not ledger.admit(tool="image_gen", action="paint", spend=None)
    assert ledger.tally().refused == 1
    message = ledger.refusal_message(tool="image_gen", action="paint")
    assert "declares no metered third-party tools" in message and "limit of" not in message


def test_the_resolution_refuses_an_override_for_a_host_declaring_none():
    with pytest.raises(ValueError, match="declares no metered tools"):
        MeteredCallLedger.resolve_effective_ceiling(5, configured_max_metered_calls=None, enforcement_enabled=True)
    assert (
        MeteredCallLedger.resolve_ceiling_origin(None, configured_max_metered_calls=None, enforcement_enabled=True)
        == "none_declared"
    )


def test_a_ledger_declaring_none_with_a_ceiling_is_refused():
    with pytest.raises(ValueError, match="allows no metered call"):
        MeteredCallLedger("run-1", 5, none_declared=True)
