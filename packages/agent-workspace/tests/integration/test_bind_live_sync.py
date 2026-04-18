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
deterministic we invoke the production helper :func:`_handle_watch_batch`
directly with a synthesized ``(Change.added, abs_path)`` set. the same
helper is what the :func:`_watch_loop` coroutine calls for each
``awatch`` batch, so the coverage faithfully exercises the code path
under test; the only simulated piece is the event-delivery cadence.
"""

from __future__ import annotations

from collections import deque
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
from threetears.agent.workspace.materialize import (
    _handle_watch_batch,
    bind,
)
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.fs_read import FsReadTool


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_bind_live_watcher_imports_external_write(
    tmp_path: Path,
    workspace_with_audience_fixture: Any,
    permissive_acl_cache: Any,
) -> None:
    """mid-bind external write lands in L3 via the live watcher helper.

    writes a new file into ``disk_root`` inside the bind body,
    synthesizes the :func:`watchfiles.awatch` batch the OS would have
    delivered, invokes :func:`_handle_watch_batch` directly, and then
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
    lease = WorkspaceFileLease(
        fx.nats,
        namespace="test",
        pod_id="test-pod",
    )
    new_payload = b"audience_units:\n  - audience_unit: external_add\n"
    new_relpath = "externally_added.yaml"

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
    ) as disk_root:
        # external process writes a new file into disk_root.
        new_path = disk_root / new_relpath
        new_path.write_bytes(new_payload)

        # simulate the awatch batch the OS would deliver; the production
        # helper is called verbatim.
        workspace = await fx.workspace_collection.find_by_id(fx.workspace_id)
        assert workspace is not None
        just_wrote: deque[tuple[str, str]] = deque(maxlen=256)
        changed = await _handle_watch_batch(
            batch={(Change.added, str(new_path))},
            workspace=workspace,
            disk_root=disk_root,
            resolved_root=disk_root.resolve(),
            db_pool=fx.pool,
            actor_id=fx.agent_id,
            correlation_id=uuid7(),
            just_wrote=just_wrote,
        )
        assert changed == [new_relpath]

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
