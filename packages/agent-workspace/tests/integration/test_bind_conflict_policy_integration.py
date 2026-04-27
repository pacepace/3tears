"""integration: conflict-policy governs L3 vs disk authority end-to-end.

REALISM
-------

- **real bind context manager** including seed-on-enter and watcher
  spawn. the production :func:`_handle_watch_batch` helper is driven
  directly with a synthesized batch (awatch cadence on Darwin FSEvents
  is too unreliable for tight-timed tests; the production watcher
  forwards each awatch batch into the same helper so coverage is
  faithful to the runtime path).
- **real WorkspaceFileLease + real KVLease** over the fake NATS KV.
- **real FsReadTool** exercising the L3 read surface agents use.
- **fake DB pool**: shared :class:`_FakePool` from ``conftest.py``.

SCENARIOS
---------

two tests lock in the policy contract:

- ``L3_WINS``: L3 pre-populated, external process modifies the file on
  disk mid-bind, synthesized ``Change.modified`` batch is delivered,
  and ``fs_read`` returns the L3 content (external modification
  discarded).
- ``DISK_WINS``: L3 pre-populated, external process modifies the file
  on disk mid-bind, same batch is delivered, and ``fs_read`` returns
  the NEW disk content (watcher imported the update).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid7

import pytest
from watchfiles import Change

from threetears.agent.workspace.bind_policy import BindConflictPolicy
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


_TARGET_REL = "audience_settings.yaml"
_EXTERNAL_PAYLOAD = b"audience_units:\n  - audience_unit: external_modify\n"


async def _run_policy_scenario(
    fx: Any,
    tmp_path: Path,
    on_conflict: BindConflictPolicy,
    acl_cache: Any,
) -> tuple[str, bytes]:
    """drive the shared "external modify during bind" scenario.

    constructs a fresh bind root under ``tmp_path``, enters :func:`bind`
    with the given policy, writes ``_EXTERNAL_PAYLOAD`` onto disk for
    ``_TARGET_REL``, synthesizes the awatch ``modified`` batch the OS
    would deliver, invokes :func:`_handle_watch_batch` directly, and
    reads the file back through :class:`FsReadTool`. returns the
    ``fs_read`` content observed DURING the bind window (before any
    capture-back runs) alongside the in-L3 bytes observed at the same
    moment via the file collection.

    capture-back on bind exit is out of scope for the conflict-policy
    feature; this helper therefore samples L3 state while the window
    is open, not after it closes, so the two policies' during-window
    semantics are compared cleanly.

    :param fx: populated :class:`WorkspaceFixture` bag
    :ptype fx: Any
    :param tmp_path: pytest scratch directory used as the bind root
    :ptype tmp_path: Path
    :param on_conflict: policy to pass into :func:`bind`
    :ptype on_conflict: BindConflictPolicy
    :return: tuple of ``(fs_read_content_string, in_window_l3_bytes)``
    :rtype: tuple[str, bytes]
    """
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

    result_content: str = ""
    in_window_l3_bytes: bytes = b""
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
        on_conflict=on_conflict,
    ) as disk_root:
        # external process overwrites the target file on disk.
        target = disk_root / _TARGET_REL
        target.write_bytes(_EXTERNAL_PAYLOAD)

        # synthesize the awatch batch the OS would deliver.
        workspace = await fx.workspace_collection.find_by_id(
            fx.agent_id,
            fx.workspace_id,
        )
        assert workspace is not None
        just_wrote: deque[tuple[str, str]] = deque(maxlen=256)
        await _handle_watch_batch(
            batch={(Change.modified, str(target))},
            workspace=workspace,
            disk_root=disk_root,
            resolved_root=disk_root.resolve(),
            db_pool=fx.pool,
            actor_id=fx.agent_id,
            correlation_id=uuid7(),
            just_wrote=just_wrote,
            on_conflict=on_conflict,
        )

        # pin the workspace so fs_read resolves it without explicit arg.
        from threetears.agent.workspace import pin as pin_module

        await pin_module.set_pin(
            fx.context,
            workspace_id=fx.workspace_id,
            workspace_name=fx.workspace_name,
            pinned_by_actor_id=fx.agent_id,
        )
        fs_read = FsReadTool(
            workspace_collection=fx.workspace_collection,
            workspace_file_collection=fx.file_collection,
            sandbox=sandbox,
            context_provider=lambda: fx.context,
            agent_id=fx.agent_id,
            acl_cache=acl_cache,
        )
        read_result = await fs_read.execute(relative_path=_TARGET_REL)
        assert read_result.success is True, read_result.error
        result_content = read_result.content
        in_window_l3_bytes = fx.store.files[(fx.workspace_id, _TARGET_REL)].content

    return result_content, in_window_l3_bytes


async def test_l3_wins_ignores_external_modify(
    tmp_path: Path,
    workspace_with_audience_fixture: Any,
    permissive_acl_cache: Any,
) -> None:
    """L3_WINS policy: external disk modify is discarded; fs_read returns L3.

    :param tmp_path: scratch directory used for the bind root
    :ptype tmp_path: Path
    :param workspace_with_audience_fixture: pre-seeded fixture bag
    :ptype workspace_with_audience_fixture: WorkspaceFixture
    :return: None
    :rtype: None
    """
    fx = workspace_with_audience_fixture
    original_bytes = fx.store.files[(fx.workspace_id, _TARGET_REL)].content

    read_content, in_window_bytes = await _run_policy_scenario(
        fx,
        tmp_path,
        BindConflictPolicy.L3_WINS,
        acl_cache=permissive_acl_cache,
    )

    # fs_read returned the L3 (original) content, not the disk overwrite.
    assert read_content == original_bytes.decode("utf-8")
    # L3 row sampled mid-window still carries the original bytes: the
    # external modify was discarded by the live watcher. (capture-back
    # on bind exit is out of scope for the conflict-policy feature.)
    assert in_window_bytes == original_bytes


async def test_disk_wins_imports_external_modify(
    tmp_path: Path,
    workspace_with_audience_fixture: Any,
    permissive_acl_cache: Any,
) -> None:
    """DISK_WINS policy: external disk modify is imported; fs_read returns disk.

    :param tmp_path: scratch directory used for the bind root
    :ptype tmp_path: Path
    :param workspace_with_audience_fixture: pre-seeded fixture bag
    :ptype workspace_with_audience_fixture: WorkspaceFixture
    :return: None
    :rtype: None
    """
    fx = workspace_with_audience_fixture

    read_content, in_window_bytes = await _run_policy_scenario(
        fx,
        tmp_path,
        BindConflictPolicy.DISK_WINS,
        acl_cache=permissive_acl_cache,
    )

    # fs_read returned the externally-rewritten content the watcher
    # imported during the bind window.
    assert read_content == _EXTERNAL_PAYLOAD.decode("utf-8")
    # in-window L3 row already carries the new bytes.
    assert in_window_bytes == _EXTERNAL_PAYLOAD
