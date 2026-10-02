"""integration: live-sync watcher during bind window.

REALISM
-------

- **real bind context manager** including import-from-disk and watcher
  spawn. the watcher task is the production task under test.
- **real WorkspaceFileLease + real KVLease** over the fake NATS KV.
- **fake DB pool**: shared :class:`_FakePool` from conftest.

SCENARIO
--------

with a live bind window open, an external process writes a new file
into ``disk_root``. the test then reads the file out of L3 via
:class:`FsReadTool` to confirm the watcher imported it before
capture-back runs on bind exit.

EVENT DELIVERY CHOICE
---------------------

:func:`watchfiles.awatch` delivers events on a cadence that depends on
OS-native watchers (Darwin FSEvents in this environment); inside a
tight-timed test that cadence is unreliable. to keep the test
deterministic the bind window is handed a scripted change source
(``bind(watch_changes=...)``) and a synthesized ``(Change.added,
abs_path)`` set is delivered through it. the window's own watcher task
and batch handler apply it, so the only simulated piece is the
event-delivery cadence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid7

import pytest
from watchfiles import Change

from threetears.agent.workspace.config import (
    AllowConfig,
    WorkspaceConfig,
)
from threetears.agent.workspace.lease import WorkspaceFileLease
from threetears.nats.subject_permissions import WORKSPACE_LOCKS_BUCKET_SUFFIX, agent_platform_bucket_suffix
from threetears.agent.workspace.materialize import bind
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.fs_read import FsReadTool
from packages.agent.workspace.tests.helpers.scripted_watch import ScriptedWatch


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_bind_live_watcher_imports_external_write(
    tmp_path: Path,
    workspace_with_audience_fixture: Any,
    permissive_acl_cache: Any,
) -> None:
    """mid-bind external write lands in L3 via the live watcher helper.

    writes a new file into ``disk_root`` inside the bind body,
    delivers the :func:`watchfiles.awatch` batch the OS would have
    delivered through the window's scripted change source, and then
    reads the file through :class:`FsReadTool` to prove the row
    reached L3 before capture-back runs.

    :param tmp_path: pytest scratch directory used as bind root
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
    # the hub declares the agent's own workspace-locks bucket; the lease only binds it
    await fx.nats.kv_bucket(name=agent_platform_bucket_suffix(fx.agent_id, WORKSPACE_LOCKS_BUCKET_SUFFIX))
    lease = WorkspaceFileLease(fx.nats, agent_id=fx.agent_id, pod_id="test-pod")
    new_payload = b"audience_units:\n  - audience_unit: external_add\n"
    new_relpath = "externally_added.yaml"
    watch = ScriptedWatch()

    async with bind(
        agent_id=fx.agent_id,
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
        watch_changes=watch,
    ) as disk_root:
        # external process writes a new file into disk_root.
        new_path = disk_root / new_relpath
        new_path.write_bytes(new_payload)

        # deliver the awatch batch the OS would; the window's watcher applies it.
        await watch.deliver({(Change.added, str(new_path))})
        assert fx.store.files[(fx.workspace_id, new_relpath)].content == new_payload

        # read the file back via fs_read INSIDE the bind window to prove
        # L3 already carries the row.
        fs_read = FsReadTool(
            workspace_collection=fx.workspace_collection,
            workspace_file_collection=fx.file_collection,
            sandbox=sandbox,
            context_provider=lambda: fx.context,
            agent_id=fx.agent_id,
            acl_cache=permissive_acl_cache,
        )
        # pin the workspace so fs_read resolves it without an explicit arg.
        from threetears.agent.workspace import pin as pin_module

        await pin_module.set_pin(
            fx.context,
            workspace_id=fx.workspace_id,
            workspace_name=fx.workspace_name,
            pinned_by_actor_id=fx.agent_id,
        )
        result = await fs_read.execute(relative_path=new_relpath)
        assert result.success is True, result.error
        assert result.content == new_payload.decode("utf-8")

    # capture-back on clean exit leaves the file in place (sha matches),
    # so the final L3 row still carries the imported bytes.
    final_row = fx.store.files[(fx.workspace_id, new_relpath)]
    assert final_row.content == new_payload
