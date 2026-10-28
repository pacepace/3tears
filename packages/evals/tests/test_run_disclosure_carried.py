"""What a host discloses about a launch, carried from the run to the bundle, the report input and the overview (item 3).

A host states it once (``LaunchHost.run_disclosure``): the machine a launch ran on, that machine's load as it began,
how it measures latency. The launch tail stamps it on every run it starts, whatever the kind. Contracts held here:

- **the run**: a launch through a host that discloses stamps the disclosure on its run; one through a host that
  discloses nothing stores none and no key;
- **the judge kind too**: a judge campaign's run carries the host's block, so its apparatus is never "machine absent
  by construction";
- **every hop**: the bundle's run summary (``RunApparatus.run_disclosure``), the report input's rig
  (``RunRig.disclosure``) and the overview's rig row (``RigRow.disclosed``, by run) carry it as stated;
- **a disclosure, never a coordinate** (Discodon owner ruling: machine identity is apparatus with no rig coordinate):
  two runs disclosing different machines and loads have one context key and one variant key, share one rig row, and
  leave every other part of the bundle as it reads without them: no cell, confound, coverage row or reading moves;
- **unset leaves everything put**: a bundle and an input over runs whose host disclosed nothing carry no key for it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.host import with_registry
from packages.evals.tests.fixtures.toyhost.judge import ToyJudgeCompletion
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.run_facts_support import CASES, aspect_bundle, campaign_over, result_of, run_of
from threetears.evals.actions.render import render_cited
from threetears.evals.analysis.analyst.facts import campaign_overview
from threetears.evals.analysis.report_input import Disclosure
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel import (
    JUDGE_KIND,
    JUDGE_KIND_CONTRACT,
    EvalStorage,
    JudgeCase,
    JudgeCaseSource,
    judge_case_payload,
)
from threetears.evals.kernel.identity import derive_context_identity, derive_variant_identity
from threetears.evals.run import LaunchHost, launchable_judge_kind, start_run
from threetears.evals.schema import (
    EvalResult,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    RunDisclosure,
)
from threetears.evals.schema.models import TRANSCRIPT_DIM_ID
from threetears.evals.storage import InMemoryDocumentStore

SNAPSHOT = "report-snapshot:" + "0" * 32

#: How the host says latency is measured: a statement it makes in its own words.
_LATENCY = "latency is measured in-process on a monotonic clock, the provider's round trip included"

#: The host's block on two machines.
_STUDIO = RunDisclosure(values={"machine": "studio", "load1": 0.42, "app_version": "2.9.20"}, statements=[_LATENCY])
_LAPTOP = RunDisclosure(values={"machine": "laptop", "load1": 3.1, "app_version": "2.9.20"}, statements=[_LATENCY])


def test_a_disclosure_refuses_a_blank_name_and_a_blank_statement() -> None:
    with pytest.raises(ValidationError, match="needs a name"):
        RunDisclosure(values={" ": "studio"})
    with pytest.raises(ValidationError, match="says nothing"):
        RunDisclosure(statements=[""])


# --- the run, of the host's kind and of the judge kind -----------------------------------------------------------


async def _settled(host: LaunchHost, runs: list[EvalRun]) -> None:
    async with asyncio.timeout(10):
        while any(host.job_manager.is_active(run.id) for run in runs):
            await asyncio.sleep(0.01)


def _toy(disclosure: RunDisclosure | None) -> tuple[LaunchHost, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    host, _client = toyhost_launch_host(
        storage=storage, run_disclosure=(lambda: disclosure) if disclosure is not None else None
    )
    return host, storage


async def _toy_run(disclosure: RunDisclosure | None) -> EvalRun:
    host, storage = off_loop(_toy, disclosure)
    runs = await start_run(
        host,
        template_id=toyhost_template().id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
    )
    await _settled(host, runs)
    (run,) = runs
    stored = off_loop(storage.load_eval_run, run.id, TOYHOST_SCOPE)
    assert stored is not None
    return stored


async def test_a_launch_through_a_disclosing_host_stamps_its_disclosure_on_the_run() -> None:
    stored = await _toy_run(_STUDIO)
    assert stored.run_disclosure == _STUDIO


async def test_a_launch_through_a_host_disclosing_nothing_stores_no_disclosure_and_no_key() -> None:
    stored = await _toy_run(None)
    assert stored.run_disclosure is None
    assert "run_disclosure" not in stored.to_dict()


#: The judge under test.
_JUDGE = "google/gemini-3.8-flash"

#: The template a judge campaign's cases are frozen into.
_JUDGE_TEMPLATE = "judge_calibration"


class _Judge:
    """A judge-role client: every request priced at a cent, every call answered with one score."""

    model_name = _JUDGE

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float:
        return 0.01

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        return ToyJudgeCompletion(
            content=json.dumps({"reasoning": "read it", "criteria_scores": {TRANSCRIPT_DIM_ID: 3}}),
            input_tokens=100,
            output_tokens=40,
            cost_usd=0.001,
            price_source="a rate card",
            model=_JUDGE,
        )

    async def aclose(self) -> None:
        """Nothing to release."""


def _judge_host() -> LaunchHost:
    """The toy host, disclosing its block, launching the judge kind over two frozen cases."""
    client = _Judge()
    profile = toyhost_profile()
    toy, _ = toyhost_launch_host(
        profile=dataclasses.replace(
            profile, kinds=(*profile.kinds, dataclasses.replace(JUDGE_KIND_CONTRACT, seats=None))
        ),
        clients=lambda _role, _model, *, temperature=None: client,  # type: ignore[arg-type, return-value]
        settings=lambda: TOYHOST_LAUNCH_SETTINGS,
    )
    holder: list[LaunchHost] = []
    host = dataclasses.replace(
        toy,
        eval_host=with_registry(toy.eval_host, {JUDGE_KIND: launchable_judge_kind(lambda: holder[0])}),
        run_disclosure=lambda: _STUDIO,
    )
    holder.append(host)
    storage = host.eval_host.storage
    storage.save_template(
        EvalTemplate(
            id=_JUDGE_TEMPLATE,
            scope_id=TOYHOST_SCOPE,
            name="judge calibration",
            intent="Score as a person would.",
            candidate_kind=JUDGE_KIND,
        )
    )
    for index in range(2):
        case = JudgeCase(
            dim=TRANSCRIPT_DIM_ID,
            scale="ordinal",
            intent="Answer the member's question as the bot would.",
            judged_artifact=JudgedArtifact.TRANSCRIPT,
            evidence=JudgeEvidence(case_material="The member asked about the raid.", artifact="It starts at nine."),
            source=JudgeCaseSource(run_id="source-run", result_id=f"result-{index}", test_case_id=f"case-{index}"),
        )
        storage.save_test_case(
            EvalTestCase(
                id=f"judge-case-{index}",
                scope_id=TOYHOST_SCOPE,
                template_id=_JUDGE_TEMPLATE,
                variation_params={},
                host_payload=judge_case_payload(case),
            )
        )
    return host


async def test_a_judge_launch_carries_the_hosts_block() -> None:
    host = off_loop(_judge_host)
    runs = await start_run(
        host,
        template_id=_JUDGE_TEMPLATE,
        subject_id="the flash judge",
        models=[_JUDGE],
        k_runs=1,
        max_cost_usd=1.0,
        scope_id=TOYHOST_SCOPE,
    )
    await _settled(host, runs)
    (run,) = runs
    stored = off_loop(host.eval_host.storage.load_eval_run, run.id, TOYHOST_SCOPE)
    assert stored is not None and stored.candidate_kind == JUDGE_KIND
    assert stored.run_disclosure == _STUDIO, "a judge-kind run is never machine-absent by construction"


# --- every hop -----------------------------------------------------------------------------------------------------


def _trials(base: RunDisclosure | None, fast: RunDisclosure | None) -> list[tuple[EvalRun, list[EvalResult]]]:
    held = [run_of("base", run_disclosure=base), run_of("fast", run_disclosure=fast)]
    return [(run, [result_of(run, index) for index in range(CASES)]) for run in held]


def test_each_hop_carries_the_disclosure_as_the_host_stated_it() -> None:
    trials = _trials(_STUDIO, _LAPTOP)
    _, bundle, _ = aspect_bundle(trials)
    rigs = {summary.run_id: summary.apparatus for summary in bundle.run_summaries}
    assert {run: rig.run_disclosure if rig is not None else None for run, rig in rigs.items()} == {
        "run-base": _STUDIO,
        "run-fast": _LAPTOP,
    }

    held = report_input(campaign_over(trials))
    assert {rig.run: rig.disclosure for rig in held.apparatus or ()} == {
        "run-base": Disclosure(values=_STUDIO.values, statements=[_LATENCY]),
        "run-fast": Disclosure(values=_LAPTOP.values, statements=[_LATENCY]),
    }

    (row,) = campaign_overview(held, SNAPSHOT).apparatus
    assert row.runs == {"run-base": "base", "run-fast": "fast"}, "two machines are one rig"
    assert {run: shown.values["machine"] for run, shown in row.disclosed.items()} == {
        "run-base": "studio",
        "run-fast": "laptop",
    }
    text = render_cited(campaign_overview(held, SNAPSHOT))
    assert "disclosed" in text and _LATENCY in text


# --- a disclosure, never a coordinate ------------------------------------------------------------------------------


def test_two_runs_disclosing_different_machines_have_one_context_and_one_variant() -> None:
    studio, laptop = run_of("base", run_disclosure=_STUDIO), run_of("base", run_disclosure=_LAPTOP)
    plain = run_of("base")
    profile = toyhost_profile()
    assert derive_context_identity(studio, profile) == derive_context_identity(laptop, profile)
    assert derive_context_identity(studio, profile) == derive_context_identity(plain, profile)
    assert derive_variant_identity(run=studio, profile=profile) == derive_variant_identity(run=laptop, profile=profile)


def test_a_disclosure_moves_nothing_in_the_bundle_but_the_run_summaries_that_carry_it() -> None:
    _, disclosed, _ = aspect_bundle(_trials(_STUDIO, _LAPTOP))
    _, plain, _ = aspect_bundle(_trials(None, None))
    assert disclosed.apparatus_confounds == plain.apparatus_confounds
    assert all("machine" not in confound.dimension for confound in disclosed.apparatus_confounds)
    left, right = disclosed.to_dict(), plain.to_dict()
    left.pop("run_summaries"), right.pop("run_summaries")
    assert left == right, "no cell, confound, coverage row or reading reads a disclosure"


# --- unset leaves everything put -----------------------------------------------------------------------------------


def test_runs_whose_host_disclosed_nothing_carry_no_key_in_the_bundle_or_the_input() -> None:
    trials = _trials(None, None)
    _, bundle, _ = aspect_bundle(trials)
    assert '"run_disclosure"' not in bundle.to_json()
    held = report_input(campaign_over(trials))
    assert all(rig.disclosure is None for rig in held.apparatus or ())
    assert all("disclosure" not in rig for rig in held.model_dump(mode="json", by_alias=True).get("apparatus") or ())
    assert all(not row.disclosed for row in campaign_overview(held, SNAPSHOT).apparatus)
