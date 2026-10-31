"""The report writer's actions, called through the catalogue as a host calls them, with scripted data and no model.

A host (Discodon, then metallm) creates a writer template, mints its own campaigns' report evidence into writer cases,
retires a case whose evidence moved, and reads whether a candidate should be promoted, each through
:meth:`MountedTool.call`, the one path every transport takes:

- **A template is created** of the writer kind, scored on the shipped editor rubric, and refused on a host that
  launches no writer kind.
- **A case is minted** into the template and listed in its side's set; minting it again returns it; a mint naming
  two sources is refused with its remedy; a retired case lets its source be minted again.
- **A decision is read** from a held-out campaign, on the read-only tool as on the standard one, and nothing is
  recorded.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from packages.evals.tests.analyst_support import fixture_snapshots
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.host import with_kinds
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.writer_case_support import writer_host
from packages.evals.tests.writer_promotion_support import BETTER, CANDIDATE, promotion_campaign
from threetears.evals.actions import eval_catalogue, read_only_tools, standard_tools
from threetears.evals.ops import OpsHost
from threetears.evals.ops.analyst import report_snapshot_restore
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND, writer_scoring_rubric


def _tools() -> dict[str, Any]:
    return {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}


def _writer_launching() -> OpsHost:
    """The toy host with a kind registered under the writer's name: creating a template reads only whether the
    registry has the kind, so the toy host's own launchable stands in for the writer's. Built off the loop, as its
    store is seeded."""
    host = off_loop(writer_host).host
    kinds = dict(host.launch.kinds)
    return OpsHost(launch=with_kinds(host.launch, {**kinds, WRITER_KIND: next(iter(kinds.values()))}))


async def _call(host: OpsHost, **arguments: Any) -> Any:
    return await _tools()["evals"].call(arguments, host=host, caller=CALLER)


async def test_a_host_creates_a_writer_template_scored_on_the_shipped_rubric() -> None:
    host = _writer_launching()
    created = await _call(
        host, action="writer_template_create", template_name="report writer", intent="Write a campaign's report."
    )
    assert not created.is_error, created.text
    assert created.structured["candidate_kind"] == WRITER_KIND
    stored = off_loop(host.eval_host.storage.load_template, created.structured["id"], CALLER.scope_id)
    assert stored is not None and stored.rubric == writer_scoring_rubric()


async def test_a_host_launching_no_writer_kind_is_refused_a_template() -> None:
    refused = await _call(
        off_loop(writer_host).host, action="writer_template_create", template_name="w", intent="Write."
    )
    assert refused.is_error
    assert "launches no 'report_writer' kind" in refused.text


async def test_a_host_mints_a_case_retires_it_and_mints_its_source_again() -> None:
    host = _writer_launching()
    created = await _call(host, action="writer_template_create", template_name="report writer", intent="Write.")
    template_id = created.structured["id"]
    restored = await report_snapshot_restore(host, CALLER.scope_id, fixture_snapshots(CALLER.scope_id)["r5"])

    minted = await _call(
        host, action="writer_case_mint", template_id=template_id, snapshot_id=restored.snapshot_id, group="r5"
    )
    assert not minted.is_error, minted.text
    receipt = minted.structured
    assert receipt["template_id"] == template_id and receipt["reused"] is False
    assert receipt["case_set"]["name"] == ("writer-held-out" if receipt["side"] == "held_out" else "writer-dev")

    again = await _call(
        host, action="writer_case_mint", template_id=template_id, snapshot_id=restored.snapshot_id, group="r5"
    )
    assert again.structured["test_case_id"] == receipt["test_case_id"] and again.structured["reused"] is True

    retired = await _call(
        host, action="writer_case_archive", writer_case_id=receipt["test_case_id"], archive_reason="re-run"
    )
    assert not retired.is_error, retired.text
    assert (retired.structured["archived"], retired.structured["archived_reason"]) == (True, "re-run")
    assert retired.structured["side"] == receipt["side"]

    reminted = await _call(
        host, action="writer_case_mint", template_id=template_id, snapshot_id=restored.snapshot_id, group="r5"
    )
    assert reminted.structured["test_case_id"] != receipt["test_case_id"]
    assert reminted.structured["side"] == receipt["side"], "a group keeps its recorded side"


async def test_a_mint_naming_two_sources_is_refused_with_its_remedy() -> None:
    host = _writer_launching()
    refused = await _call(
        host, action="writer_case_mint", template_id="writer-template", campaign_id="c", snapshot_id="s"
    )
    assert refused.is_error
    assert "exactly one of campaign_id, analysis_id and snapshot_id" in refused.text
    assert "campaign_id and snapshot_id were named" in refused.text


async def test_a_host_reads_a_promotion_decision_and_nothing_is_recorded() -> None:
    eval_host, campaign_id = off_loop(promotion_campaign, BETTER)
    launch = off_loop(ops_fixture, generation=False).host.launch
    host = OpsHost(launch=replace(launch, eval_host=eval_host))

    read = await _call(host, action="writer_promotion", campaign_id=campaign_id, candidate=CANDIDATE)
    assert not read.is_error, read.text
    assert read.structured["decision"] == "promote"
    assert read.structured["candidate"] == CANDIDATE and read.structured["changed"] == ["chunk_tokens"]
    stored = off_loop(eval_host.storage.load_campaign, campaign_id, TOYHOST_SCOPE)
    assert stored is not None and stored.call is None

    (read_only,) = eval_catalogue().mount_all(read_only_tools())
    assert read_only.action("writer_promotion") is not None
    assert read_only.action("writer_case_mint") is None and read_only.action("writer_template_create") is None
    from_read_only = await read_only.call(
        {"action": "writer_promotion", "campaign_id": campaign_id, "candidate": CANDIDATE}, host=host, caller=CALLER
    )
    assert from_read_only.structured["decision"] == "promote"
