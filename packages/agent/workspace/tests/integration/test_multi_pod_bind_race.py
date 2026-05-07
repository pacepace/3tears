"""integration: multi-pod bind race (shard-18 requirement WS-18-06).

REALISM
-------

- **real WorkspaceFileLease + real KVLease**: exercised end-to-end over
  the core fake NATS KV whose CAS semantics match real nats-py
  :class:`KeyValue`. the lease contract this test locks in is the
  production contract; swapping the fake for a real NATS container
  would not change the assertions.
- **real bind context manager**: from :mod:`materialize`.
- **fake DB pool**: task-14's ``_FakePool`` shim; bind's capture-back
  issues the production SQL statements and the pool records them, but
  the focus of this test is the lease serialization semantic.

the shard's WS-18-06 requirement is: "two simulated pods bind same
workspace; exactly one inside the bind body at a time". task-14's
``test_bind_lease_race.py`` already delivers that guarantee; this file
adds the ``@pytest.mark.integration`` marker formally (which the
existing tests also carry) and re-asserts the peak-concurrency == 1
property through an explicit wrapper so the shard-18 acceptance can be
traced to this file.

future graduation: when testcontainers NATS is wired into the repo, the
fake KV imports below can be replaced with a ``nats.connect(...)`` call
pointing at the container and the assertions here remain unchanged.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest

from _fake_kv import FakeNatsClient  # type: ignore[import-not-found]

from threetears.agent.workspace.lease import WorkspaceFileLease
from threetears.agent.workspace.materialize import bind
from _helpers.asyncpg_shims import FakeAsyncpgAcquireCM, FakeAsyncpgConnection, FakeAsyncpgPool, FakeAsyncpgTransaction
from _helpers.workspace_shims import (
    FakeWorkspaceCollection,
    FakeWorkspaceContext,
    FakeWorkspaceEntity,
    FakeWorkspaceFile,
    FakeWorkspaceFileCollection,
    FakeWorkspaceFileVersionCollection,
    FakeWorkspaceSandbox,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@dataclass
class _FakeWorkspace(FakeWorkspaceEntity):
    id: UUID
    name: str
    agent_id: UUID
    current_version: int = 0
    date_deleted: datetime | None = None

    @property
    def namespace_name(self) -> str:
        """canonical workspace namespace name (WS-ACL-06)."""
        return f"workspace.{self.id}"

    @property
    def owner_agent_id(self) -> UUID:
        """alias of :attr:`agent_id` matching the production entity."""
        return self.agent_id


class _FakeWorkspaceCollection(FakeWorkspaceCollection):
    def __init__(self, ws: _FakeWorkspace) -> None:
        self._ws = ws

    async def find_by_id(
        self,
        agent_id: UUID,
        workspace_id: UUID,
    ) -> _FakeWorkspace | None:
        if workspace_id != self._ws.id or agent_id != self._ws.agent_id:
            return None
        return self._ws


class _FakeFileCollection(FakeWorkspaceFileCollection):
    async def find_by_workspace(self, workspace_id: UUID) -> list[Any]:
        return []


class _FakeVersionCollection(FakeWorkspaceFileVersionCollection):
    pass


class _FakeSandbox(FakeWorkspaceSandbox):
    def __init__(self, roots: dict[str, Path]) -> None:
        self._roots = roots

    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        root = self._roots[root_name]
        candidate = (root / path).resolve()
        candidate.relative_to(root)
        return candidate


@dataclass
class _FakeTransaction(FakeAsyncpgTransaction):
    parent: Any

    async def __aenter__(self) -> _FakeTransaction:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakeConnection(FakeAsyncpgConnection):
    executions: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
        return _FakeTransaction(parent=self)

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args))
        return "INSERT 0 1"


@dataclass
class _FakeAcquireCM(FakeAsyncpgAcquireCM):
    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool(FakeAsyncpgPool):
    conn: _FakeConnection = field(default_factory=_FakeConnection)

    def acquire(self) -> _FakeAcquireCM:
        return _FakeAcquireCM(conn=self.conn)


async def test_two_pods_peak_concurrency_is_one(tmp_path: Path) -> None:
    """WS-18-06: two pods racing on bind have peak concurrent body count == 1.

    :param tmp_path: pytest-provided scratch directory
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    bind_root = tmp_path / "bind_root"
    bind_root.mkdir()
    ws_id = uuid7()
    ws_name = "racer"
    (bind_root / ws_name).mkdir()
    agent_id = uuid4()
    workspace = _FakeWorkspace(id=ws_id, name=ws_name, agent_id=agent_id)

    nats = FakeNatsClient()
    lease_a = WorkspaceFileLease(nats, namespace="test", pod_id="pod-A")
    lease_b = WorkspaceFileLease(nats, namespace="test", pod_id="pod-B")

    in_body = {"count": 0}
    peak = {"value": 0}
    observations: list[str] = []

    async def _run_pod(label: str, lease: WorkspaceFileLease) -> None:
        sandbox = _FakeSandbox({"bind": bind_root})
        async with bind(
            agent_id=agent_id,
            workspace_id=ws_id,
            sandbox=sandbox,  # type: ignore[arg-type]
            lease=lease,
            workspace_collection=_FakeWorkspaceCollection(workspace),  # type: ignore[arg-type]
            workspace_file_collection=_FakeFileCollection(),  # type: ignore[arg-type]
            workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
            db_pool=_FakePool(),
            actor_id=uuid4(),
            correlation_id=uuid7(),
            lease_ttl_seconds=30,
            lease_max_wait_seconds=10,
        ):
            in_body["count"] += 1
            peak["value"] = max(peak["value"], in_body["count"])
            observations.append(f"{label}:enter")
            for _ in range(5):
                await asyncio.sleep(0.005)
                peak["value"] = max(peak["value"], in_body["count"])
            in_body["count"] -= 1
            observations.append(f"{label}:exit")

    await asyncio.gather(
        _run_pod("A", lease_a),
        _run_pod("B", lease_b),
    )

    assert peak["value"] == 1, f"lease failed to serialize: peak = {peak['value']}; observations = {observations}"
    # each pod fully entered and exited exactly once.
    assert observations.count("A:enter") == 1
    assert observations.count("A:exit") == 1
    assert observations.count("B:enter") == 1
    assert observations.count("B:exit") == 1
    # one pod's full enter/exit completes before the other enters.
    assert observations[0].endswith(":enter")
    assert observations[1].endswith(":exit")
    assert observations[2].endswith(":enter")
    assert observations[3].endswith(":exit")
