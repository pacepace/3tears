"""integration: bind + stub builder against the fixture audience workspace.

REALISM
-------

- **real bind context manager + real atomic_write + real capture-back
  transaction**: the whole production bind path runs end-to-end on the
  fake DB pool, which pattern-matches the SQL that :func:`_capture_back`
  issues. the test reads the in-memory store back at the end to verify
  the head row + journal row landed.
- **real WorkspaceFileLease + real KVLease**: over the fake NATS KV.
- **fake DB pool**: shared :class:`_FakePool` from conftest.
- **fake NATS client**: used only for lease KV + publish capture; the
  audit path is verified separately in ``test_audit_event_landed.py``.

this test wires the audience fixture through the conftest seeder, runs
a stub builder inside the bind body that edits one fixture file on
disk, exits the bind cleanly, and asserts L3 now carries the mutated
bytes with an ``update``-action journal row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid7

import pytest

from threetears.agent.workspace.bind_policy import BindConflictPolicy
from threetears.agent.workspace.config import (
    AllowConfig,
    WorkspaceConfig,
)
from threetears.agent.workspace.lease import WorkspaceFileLease
from threetears.agent.workspace.materialize import bind
from threetears.agent.workspace.sandbox import WorkspaceSandbox


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_bind_captures_back_stub_builder_edit(
    tmp_path: Path,
    workspace_with_audience_fixture: Any,
) -> None:
    """stub builder edits one file in the bind body; L3 reflects the edit.

    :param tmp_path: pytest-provided scratch directory used as bind root
    :ptype tmp_path: Path
    :param workspace_with_audience_fixture: pre-seeded fixture bag
    :ptype workspace_with_audience_fixture: WorkspaceFixture
    :return: None
    :rtype: None
    """
    fx = workspace_with_audience_fixture
    bind_root = tmp_path / "bind_root"
    bind_root.mkdir()
    (bind_root / fx.workspace_name).mkdir()
    config = WorkspaceConfig(
        bind_root=bind_root,
        allow=AllowConfig(read=["**/*"], write=["**/*.yaml"]),
    )
    sandbox = WorkspaceSandbox.from_config(config)
    lease = WorkspaceFileLease(
        fx.nats, namespace="test", pod_id="test-pod"
    )
    new_payload = b"audience_units:\n  - audience_unit: test_override\n"

    async with bind(
        workspace_id=fx.workspace_id,
        sandbox=sandbox,
        lease=lease,
        workspace_collection=fx.workspace_collection,
        workspace_file_collection=fx.file_collection,
        workspace_file_version_collection=fx.version_collection,
        db_pool=fx.pool,
        actor_id=fx.agent_id,
        correlation_id=uuid7(),
        lease_ttl_seconds=30,
        lease_max_wait_seconds=10,
        nats_client=fx.nats,
        namespace="threetears-test",
        on_conflict=BindConflictPolicy.L3_WINS,
    ) as disk_root:
        # sanity: materialize placed all three fixture files.
        files_on_disk = sorted(
            p.name for p in disk_root.iterdir() if p.is_file()
        )
        assert files_on_disk == [
            "audience_settings.yaml",
            "linkedin_audience_units.yaml",
            "standard_audience_units.yaml",
        ]
        # stub builder mutates one file.
        target = disk_root / "audience_settings.yaml"
        target.write_bytes(new_payload)

    # L3 head row now carries the new bytes at version 2.
    head = fx.store.files[
        (fx.workspace_id, "audience_settings.yaml")
    ]
    assert head.content == new_payload
    assert head.version == 2

    # journal has a matching update-action row at version 2.
    update_rows = [
        v
        for v in fx.store.versions
        if v.relative_path == "audience_settings.yaml"
        and v.action == "update"
    ]
    assert len(update_rows) == 1
    assert update_rows[0].version == 2
    assert update_rows[0].content == new_payload

    # other two fixture files were not modified.
    for name in (
        "linkedin_audience_units.yaml",
        "standard_audience_units.yaml",
    ):
        still_head = fx.store.files[(fx.workspace_id, name)]
        assert still_head.version == 1, (
            f"{name} should remain at v1 (untouched on disk)"
        )
