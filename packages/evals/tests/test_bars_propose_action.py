"""``bars_propose``: a baseline campaign's bar proposals through the action catalogue (#579).

A host whose bar registry is empty has one supported way to fill it from measurement: call the action on a
single-cell baseline campaign and read back a proposal per declared measure with a better end, vacuous seeds
flagged, and every reading nothing could be proposed on named. The action is read-only: it registers nothing,
and the refusals ``propose_bars`` raises come back as teaching errors.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.kernel import EvalCampaign
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.kernel.host import BarRegistry, EvalHost
from threetears.evals.ops import BarProposals, OpsHost, bars_propose
from packages.evals.tests.factories import memory_storage
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, toyhost_batch, toyhost_measurements
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.ops_support import CALLER, ops_fixture


def _batch(chunk_tokens: int, *, field_accuracy: float) -> tuple[EvalRun, list[EvalResult]]:
    batch = toyhost_batch(
        chunk_tokens=chunk_tokens,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
        grader_version="grade-2.1.0",
    )
    return batch, toyhost_measurements(
        batch, profile=toyhost_profile(), cost_usd=0.02, total_ms=900.0, field_accuracy=field_accuracy
    )


def _flat_accuracy(batch: tuple[EvalRun, list[EvalResult]]) -> tuple[EvalRun, list[EvalResult]]:
    """Field accuracy bottomed out at zero on every observation: nothing could fail a bar seeded there."""
    run, results = batch
    return run, [
        result.model_copy(update={"host_measures": {**result.host_measures, "field_accuracy": 0.0}})
        for result in results
    ]


def _host(batches: Sequence[tuple[EvalRun, list[EvalResult]]]) -> OpsHost:
    """An operations host over the toy host with an EMPTY bar registry and one baseline campaign of ``batches``."""
    storage, _ = memory_storage()
    for run, results in batches:
        storage.save_eval_run(run)
        for result in results:
            storage.save_eval_result(result)
    storage.save_campaign(
        EvalCampaign(
            id="baseline",
            scope_id=TOYHOST_SCOPE,
            name="incumbent baseline",
            subject_id=batches[0][0].subject_snapshot.subject_id,
            subject_kind="extractor_config",
            behavior="extract_invoice_fields",
            run_ids=[run.id for run, _ in batches],
            created_by="test:fixture",
        )
    )
    eval_host: EvalHost = toyhost_host(storage=storage, profile=replace(toyhost_profile(), bars=BarRegistry()))
    launch = ops_fixture().host.launch
    return OpsHost(launch=replace(launch, eval_host=eval_host))


def _tool() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _call(host: OpsHost, arguments: dict[str, Any]) -> Any:
    return asyncio.run(_tool().call({"action": "bars_propose", **arguments}, host=host, caller=CALLER))


def test_an_empty_registry_gets_a_proposal_per_declared_measure_with_vacuous_seeds_flagged() -> None:
    host = _host([_flat_accuracy(_batch(256, field_accuracy=0.5))])
    assert host.eval_host.profile.bars.bars == ()

    outcome = _call(host, {"campaign_id": "baseline"})

    assert not outcome.is_error, outcome.text
    proposals = BarProposals.model_validate(outcome.structured)
    assert proposals == bars_propose(host.eval_host, "baseline", TOYHOST_SCOPE)
    directional = {measure.name for measure in TOYHOST_MEASURES if measure.higher_is_better is not None}
    proposed = {bar.measure for bar in proposals.proposals}
    assert proposed | (directional & set(proposals.not_proposed)) == directional, "every declared measure answered"
    accuracy = next(bar for bar in proposals.proposals if bar.measure == "field_accuracy")
    assert accuracy.vacuous and accuracy.vacuous_reason
    assert "field_accuracy >= 0 — VACUOUS, do not adopt as written" in outcome.text
    assert "nothing is registered" in outcome.text
    assert host.eval_host.profile.bars.bars == (), "the action registered nothing"


def test_a_baseline_of_two_cells_is_refused_as_a_teaching_error() -> None:
    host = _host([_batch(256, field_accuracy=0.8), _batch(512, field_accuracy=0.7)])

    outcome = _call(host, {"campaign_id": "baseline"})

    assert outcome.is_error and "measured 2 cells, and a baseline is one configuration under one rig" in outcome.text


def test_a_campaign_that_is_not_there_is_refused_naming_it() -> None:
    outcome = _call(_host([_batch(256, field_accuracy=0.8)]), {"campaign_id": "nowhere"})

    assert outcome.is_error and "nowhere" in outcome.text
