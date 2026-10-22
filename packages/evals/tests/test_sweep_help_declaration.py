"""A sweep's help names a way to declare its campaign's design that the freeze rule allows.

A sweep's campaign is exploratory and holds its runs, so its declaration is frozen: ``campaign_update`` refuses a
design there. The help names ``campaign_create`` over the sweep's runs, and this reads that help against the rule: the
call it names succeeds over a real sweep's runs, and the one it must not name is refused.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.kernel import EvalCampaign
from threetears.evals.ops import OpsHost, SweepArguments, SweepArm, job_poll, sweep_launch
from threetears.evals.ops import sweeps
from threetears.evals.ops.sweeps import SweepSettings
from threetears.evals.kernel import EvalStorage
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_aspect_design
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import TOY_SCRIPTS, ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.off_loop import off_loop

STYLES = ("terse", "standard", "verbose")


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _help_texts() -> list[str]:
    """Every place a sweep's help speaks of declaring a design after the runs."""
    field = SweepSettings.model_fields["campaign_name"].description or ""
    module = sweeps.__doc__ or ""
    section = module.split("**Its campaign is exploratory.**", 1)[1].split("**When it stops early.**", 1)[0]
    return [field, section]


def _sentences(text: str) -> list[str]:
    return [sentence for sentence in re.split(r"(?<=[.;])\s+", " ".join(text.split())) if sentence]


def _host() -> tuple[OpsHost, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    launch, _client = toyhost_launch_host(storage=storage, client=ScriptedExtractionClient(TOY_SCRIPTS))
    return OpsHost(launch=launch), storage


async def _swept() -> tuple[OpsHost, EvalStorage, EvalCampaign, list[str]]:
    """A real sweep run to its end: the host, its store, its campaign, and the run ids its job result names."""
    host, storage = off_loop(_host)
    arguments = SweepArguments(
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        campaign_name="prompt styles",
        campaign_behavior="extraction accuracy",
        k_runs=1,
        arms=[SweepArm(model=RUN_MODELS[0], label=style, overlays={"prompt_style": style}) for style in STYLES],
    )
    (job,) = (await sweep_launch(host, arguments, TOYHOST_SCOPE, created_by="agent:test")).jobs
    async with asyncio.timeout(20):
        while not (status := await job_poll(host, job.job_id, TOYHOST_SCOPE)).done:
            await asyncio.sleep(0.005)
    campaign = off_loop(storage.load_campaign, job.target_id, TOYHOST_SCOPE)
    assert campaign is not None
    return host, storage, campaign, status.run_ids


def test_every_sentence_naming_campaign_update_says_it_refuses_once_runs_exist() -> None:
    named = [sentence for text in _help_texts() for sentence in _sentences(text) if "campaign_update" in sentence]
    assert named, "the help says what campaign_update cannot do here"
    for sentence in named:
        assert "refuses" in sentence, sentence
    for text in _help_texts():
        assert "campaign_create" in text and "from_run_ids" in text


async def test_the_declaration_the_help_names_is_made_over_a_real_sweeps_runs(evals: MountedTool) -> None:
    host, storage, swept, run_ids = await _swept()
    assert run_ids and set(run_ids) == set(swept.run_ids), "the sweep's job result names the runs its campaign holds"
    first = off_loop(storage.load_eval_runs, run_ids[:1], TOYHOST_SCOPE)
    assert [run.overlays["prompt_style"] for run in first] == [STYLES[0]], "run_ids are in arm order"
    design = toyhost_aspect_design().model_dump(mode="json")
    aspect = {"label": swept.behavior, "run_ids": run_ids, "margin": 5.0, "design": design}

    refused = await evals.call(
        {
            "action": "campaign_update",
            "campaign_id": swept.id,
            "aspects": [{k: aspect[k] for k in ("label", "design")}],
        },
        host=host,
        caller=CALLER,
    )
    assert refused.is_error and "frozen" in refused.text

    created = await evals.call(
        {
            "action": "campaign_create",
            "name": "the sweep, declared",
            "subject_id": swept.subject_id,
            "behavior": swept.behavior,
            "question": "does a style beat terse?",
            "campaign_arms": [{"label": "terse", "role": "control", "from_run_ids": [run_ids[0]]}],
            "aspects": [aspect],
        },
        host=host,
        caller=CALLER,
    )

    assert not created.is_error, created.text
    assert created.structured["exploratory"] is False
    declared = off_loop(storage.load_campaign, created.structured["id"], TOYHOST_SCOPE)
    assert declared is not None and set(declared.run_ids) == set(run_ids)
    eval_host = host.eval_host
    bundle = off_loop(assemble_context_bundle, declared, storage=storage, profile=eval_host.profile)
    assert bundle.reading_scope.question_after_runs is not None, "a question asked after the runs says so"
