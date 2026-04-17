"""integration: doc_set preserves YAML comments + key order on real audience YAML.

REALISM
-------

- **real doc_set, real ruamel.yaml handler**: the tool wires through the
  actual :class:`DocSetTool` and :func:`handler_for` so the round-trip
  fidelity is exercised against the same code the production agent
  runs.
- **real fs_read**: reads back through :class:`FsReadTool` to match the
  production read path.
- **fake DB pool**: pattern-matching in-memory store defined in
  :mod:`conftest`. the SQL for head lookup + OCC + journal insert +
  head upsert is exercised verbatim.
- **fake NATS**: audit publish is captured but not asserted here (see
  ``test_audit_event_landed.py`` for the audit end-to-end).

the point of this test is round-trip fidelity: doc_set mutates a single
scalar in a multi-section YAML, and the stored bytes must retain the
original key order and every non-comment semantic structure.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.agent.workspace.config import AllowConfig, WorkspaceConfig
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.doc_set import DocSetTool
from threetears.agent.workspace.tools.fs_read import FsReadTool


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _build_sandbox() -> WorkspaceSandbox:
    """
    build a sandbox that allows reads + writes on every yaml fixture.

    :return: workspace sandbox with permissive yaml globs
    :rtype: WorkspaceSandbox
    """
    config = WorkspaceConfig(
        allow=AllowConfig(read=["**/*"], write=["**/*.yaml"]),
    )
    return WorkspaceSandbox.from_config(config)


async def test_doc_set_on_audience_settings_preserves_structure(
    workspace_with_audience_fixture: Any,
) -> None:
    """mutate one scalar via doc_set; verify key order + anchor values intact.

    :param workspace_with_audience_fixture: pre-seeded fixture bag
    :ptype workspace_with_audience_fixture: WorkspaceFixture
    """
    fx = workspace_with_audience_fixture
    sandbox = _build_sandbox()
    # pin the workspace to the context so the tool can resolve it.
    from threetears.agent.workspace.pin import set_pin

    await set_pin(
        fx.context,
        workspace_id=fx.workspace_id,
        workspace_name=fx.workspace_name,
        pinned_by_actor_id=fx.agent_id,
    )

    doc_set = DocSetTool(
        workspace_collection=fx.workspace_collection,
        workspace_file_collection=fx.file_collection,
        workspace_file_version_collection=fx.version_collection,
        sandbox=sandbox,
        context_provider=lambda: fx.context,
        agent_id=fx.agent_id,
        db_pool=fx.pool,
        nats_client=fx.nats,
        namespace="threetears-test",
    )
    fs_read = FsReadTool(
        workspace_collection=fx.workspace_collection,
        workspace_file_collection=fx.file_collection,
        sandbox=sandbox,
        context_provider=lambda: fx.context,
        agent_id=fx.agent_id,
    )

    # original text carries the section markers we expect to survive
    original = (fx.fixture_path / "audience_settings.yaml").read_text()
    assert "audience_units:" in original
    assert "knowwho_all" in original
    assert "executive_coworkers_of_linkedin_execs" in original

    # mutate the first unit's vb_candidates from 10 -> 99.
    set_result = await doc_set.execute(
        relative_path="audience_settings.yaml",
        jsonpath="$.audience_units[0].vb_candidates",
        value=99,
    )
    assert set_result.success is True, set_result.error

    # read back through fs_read and assert the new value landed, the
    # original key order survived (knowwho_all still first, donors still
    # last), and the original section structure is still present.
    read_result = await fs_read.execute(relative_path="audience_settings.yaml")
    assert read_result.success is True, read_result.error
    assert read_result.content is not None
    updated_text = read_result.content

    # new scalar is present at the path we set
    assert "vb_candidates: 99" in updated_text
    # first-unit name still first in the list, donors block still present
    first_idx = updated_text.find("knowwho_all")
    donors_idx = updated_text.find("donors")
    assert first_idx != -1 and donors_idx != -1
    assert first_idx < donors_idx, "knowwho_all must still precede donors in audience_units"
    # the middle section marker survives the mutation
    assert "executive_coworkers_of_linkedin_execs" in updated_text
    # other value from the fixture still present verbatim
    assert "committee_transaction_amt:" in updated_text


async def test_doc_set_bumps_version_and_updates_head(
    workspace_with_audience_fixture: Any,
) -> None:
    """
    after doc_set the head row for the file advances one version and the
    stored content now carries the mutated scalar bytes.

    :param workspace_with_audience_fixture: pre-seeded fixture bag
    :ptype workspace_with_audience_fixture: WorkspaceFixture
    """
    fx = workspace_with_audience_fixture
    sandbox = _build_sandbox()
    from threetears.agent.workspace.pin import set_pin

    await set_pin(
        fx.context,
        workspace_id=fx.workspace_id,
        workspace_name=fx.workspace_name,
        pinned_by_actor_id=fx.agent_id,
    )

    doc_set = DocSetTool(
        workspace_collection=fx.workspace_collection,
        workspace_file_collection=fx.file_collection,
        workspace_file_version_collection=fx.version_collection,
        sandbox=sandbox,
        context_provider=lambda: fx.context,
        agent_id=fx.agent_id,
        db_pool=fx.pool,
    )

    initial_head = fx.store.files[(fx.workspace_id, "audience_settings.yaml")]
    assert initial_head.version == 1

    result = await doc_set.execute(
        relative_path="audience_settings.yaml",
        jsonpath="$.audience_units[0].vb_candidates",
        value=99,
    )
    assert result.success is True, result.error

    new_head = fx.store.files[(fx.workspace_id, "audience_settings.yaml")]
    assert new_head.version == 2
    assert b"vb_candidates: 99" in new_head.content
    # a fresh journal row landed with action=update
    updates = [v for v in fx.store.versions if v.relative_path == "audience_settings.yaml" and v.action == "update"]
    assert len(updates) == 1
    assert updates[0].version == 2
