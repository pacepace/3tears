"""Reports to test against: the toy host's, generated through the host, and a minimal one that passes.

Shared by ``test_report.py`` and ``test_report_html_no_script.py`` — harness support, not part of the
toy host, which reaches the engine only through its public roots.
"""

from __future__ import annotations

import json
from typing import Any

from threetears.evals.analysis import (
    Report,
    analysis_report,
    prepare_analysis_generation,
    run_analysis_generation,
)
from threetears.evals.contracts.campaign import EvalAnalysis, EvalCampaign
from threetears.evals.contracts.host import EvalHost
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.toyhost_memo import PROMPT, PROMPT_ID, FixturedClient, alias_at, memo_payload

__all__ = [
    "minimal_report",
    "toy_campaign_host",
    "toy_report",
]


def _with_a_chart(bundle: Any) -> dict[str, Any]:
    """The toy memo, its finding carrying a delta table over the two widths' wall-clock."""
    payload = memo_payload(bundle)
    payload["findings"][0]["chart"] = {
        "type": "delta_table",
        "cells": [alias_at(bundle, TOYHOST_NARROW), alias_at(bundle, TOYHOST_WIDE)],
        "measures": [{"measure_id": "total_ms", "reading": "measure"}],
        "axis": "",
        "note": "",
        "caption": "the wide chunk is the slower one",
    }
    return payload


def toy_campaign_host(clients: Any = None) -> tuple[EvalHost, EvalCampaign]:
    """The toy host holding its corpus campaign — runs, results and the campaign — and no analysis of it."""
    host = toyhost_host(clients=clients) if clients is not None else toyhost_host()
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        host.storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            host.storage.save_eval_result(result)
    host.storage.save_campaign(campaign)
    return host, campaign


async def toy_report() -> tuple[EvalHost, EvalAnalysis, Report]:
    """The toy campaign generated through the host and read back as its report."""
    built: list[FixturedClient] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> FixturedClient:
        client = FixturedClient("")
        built.append(client)
        return client

    host, campaign = toy_campaign_host(clients)

    async def prompt() -> str:
        return PROMPT

    prepared = await prepare_analysis_generation(
        host, campaign.id, campaign.scope_id, model=None, resolve_prompt=prompt
    )
    built[0].completion.content = json.dumps(_with_a_chart(prepared.bundle))
    analysis, _ = await run_analysis_generation(host, prepared, prompt_id=PROMPT_ID, max_output_tokens=4000)
    return host, analysis, analysis_report(host.storage, analysis.id, analysis.scope_id)


def minimal_report(**update: Any) -> Report:
    """A one-finding report that passes, with ``update`` applied over its fields."""
    fields: dict[str, Any] = {
        "basis": "analysis",
        "headline": "h",
        "finding_count": 1,
        "source": {
            "analysis_id": "a",
            "campaign_id": "c",
            "scope_id": "s",
            "subject_id": "x",
            "subject_kind": "",
            "behavior": "b",
            "generated_at": "2026-10-05T00:00:00+00:00",
            "generator_model": "m",
            "bundle_fingerprint": "f",
        },
        "blocks": [{"kind": "text", "section": "findings", "role": "finding_title", "finding": 0, "body": "t"}],
    }
    return Report.model_validate(fields | update)
