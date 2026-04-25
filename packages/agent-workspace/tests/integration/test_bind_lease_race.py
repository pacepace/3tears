"""two-pod bind race -- verify lease serializes bodies across simulated pods.

exercises :func:`threetears.agent.workspace.materialize.bind` against the
real :class:`WorkspaceFileLease` (which wraps the real
:class:`threetears.core.coordination.KVLease`) backed by the in-memory
fake NATS KV (:class:`FakeNatsClient`). the fake KV's CAS semantics are
functionally equivalent to real NATS KV for the lease contract, so the
serialization assertions reflect what two real pods would see.

assertion mechanism: each bind body atomically increments a shared
counter on entry and decrements it on exit. the counter is observed at
multiple points during each body; a value > 1 at any observation
proves two bind bodies overlapped. with the lease in place the counter
must stay at 1 while either body is running.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest

from _fake_kv import FakeNatsClient  # type: ignore[import-not-found]
from threetears.agent.workspace.lease import WorkspaceFileLease
from threetears.agent.workspace.materialize import bind


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


# ---------------------------------------------------------------------------
# minimal fakes -- same shape as unit-test fakes but scoped to this module
# ---------------------------------------------------------------------------


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class _FakeWorkspace:
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


@dataclass
class _FakeFile:
    relative_path: str
    content: bytes
    sha256: str
    version: int


class _FakeWorkspaceCollection:
    def __init__(self, ws: _FakeWorkspace) -> None:
        self._ws = ws

    async def find_by_id(
        self, agent_id: UUID, workspace_id: UUID,
    ) -> _FakeWorkspace | None:
        if workspace_id != self._ws.id or agent_id != self._ws.agent_id:
            return None
        return self._ws


class _FakeFileCollection:
    def __init__(self, files: list[_FakeFile]) -> None:
        self._files = files

    async def find_by_workspace(self, workspace_id: UUID) -> list[_FakeFile]:
        del workspace_id
        return list(self._files)


class _FakeVersionCollection:
    pass


class _FakeSandbox:
    def __init__(self, roots: dict[str, Path]) -> None:
        self._roots = roots

    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        root = self._roots[root_name]
        candidate = (root / path).resolve()
        candidate.relative_to(root)
        return candidate


@dataclass
class _FakeTransaction:
    parent: _FakeConnection

    async def __aenter__(self) -> _FakeTransaction:
        self.parent.transaction_open = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.parent.transaction_open = False
        return None


@dataclass
class _FakeConnection:
    executions: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    transaction_open: bool = False

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
        return _FakeTransaction(parent=self)

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args))
        return "INSERT 0 1"


@dataclass
class _FakeAcquireCM:
    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool:
    conn: _FakeConnection = field(default_factory=_FakeConnection)

    def acquire(self) -> _FakeAcquireCM:
        return _FakeAcquireCM(conn=self.conn)


# ---------------------------------------------------------------------------
# race test
# ---------------------------------------------------------------------------


async def test_two_pods_binding_same_workspace_serialize_on_lease(
    tmp_path: Path,
) -> None:
    """two bind coroutines race; lease guarantees at most one body running.

    wiring:

    - one :class:`FakeNatsClient` shared across both pods (simulates the
      shared NATS cluster two real pods would talk to).
    - two distinct :class:`WorkspaceFileLease` instances with distinct
      ``pod_id`` values ("pod-A" and "pod-B") to model two pods.
    - shared counters: ``in_body`` tracks the number of bodies currently
      executing; ``peak`` records the high-water mark; ``order``
      records the acquisition order to prove the second body ran after
      the first released.

    each body pushes its label into ``order``, sleeps 50ms to widen the
    race window, and pops the label on exit. lease key is
    ``bind:bind``; the same workspace id is used so both pods race on
    the same key.
    """
    bind_root = tmp_path / "bind_root"
    bind_root.mkdir(parents=True, exist_ok=True)

    ws_id = uuid7()
    workspace_name = "racer"
    (bind_root / workspace_name).mkdir(parents=True, exist_ok=True)
    agent_id = uuid4()
    ws = _FakeWorkspace(id=ws_id, name=workspace_name, agent_id=agent_id)

    fake_nats = FakeNatsClient()
    lease_a = WorkspaceFileLease(fake_nats, namespace="test", pod_id="pod-A")
    lease_b = WorkspaceFileLease(fake_nats, namespace="test", pod_id="pod-B")

    in_body = {"count": 0}
    peak = {"value": 0}
    order: list[str] = []
    inside_order: list[str] = []

    async def _run_pod(label: str, lease: WorkspaceFileLease) -> None:
        sandbox = _FakeSandbox({"bind": bind_root})
        workspace_coll = _FakeWorkspaceCollection(ws)
        file_coll = _FakeFileCollection([])
        pool = _FakePool()
        order.append(f"{label}:enter-attempt")
        async with bind(
            agent_id=agent_id,
            workspace_id=ws_id,
            sandbox=sandbox,
            lease=lease,
            workspace_collection=workspace_coll,  # type: ignore[arg-type]
            workspace_file_collection=file_coll,  # type: ignore[arg-type]
            workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
            db_pool=pool,
            actor_id=uuid4(),
            correlation_id=uuid7(),
            lease_ttl_seconds=30,
            lease_max_wait_seconds=10,
        ):
            inside_order.append(f"{label}:enter")
            in_body["count"] += 1
            peak["value"] = max(peak["value"], in_body["count"])
            # widen the race window so an overlapping body would be obvious
            for _ in range(5):
                await asyncio.sleep(0.01)
                peak["value"] = max(peak["value"], in_body["count"])
            in_body["count"] -= 1
            inside_order.append(f"{label}:exit")

    await asyncio.gather(
        _run_pod("A", lease_a),
        _run_pod("B", lease_b),
    )

    assert peak["value"] == 1, (
        f"lease failed to serialize: peak concurrent bodies = {peak['value']}; order = {inside_order}"
    )
    # both pods ran, and entries alternate: one full enter/exit before the other
    assert len(inside_order) == 4
    assert inside_order[0].endswith(":enter")
    assert inside_order[1].endswith(":exit")
    assert inside_order[2].endswith(":enter")
    assert inside_order[3].endswith(":exit")
    # the two bodies were executed by different pods
    assert {inside_order[0].split(":")[0], inside_order[2].split(":")[0]} == {"A", "B"}


async def test_second_pod_gets_lease_after_first_releases(
    tmp_path: Path,
) -> None:
    """second pod's acquire resolves only after first pod's release fires.

    uses an :class:`asyncio.Event` the first pod waits on inside its
    body so the second pod's acquire provably blocks. once the first
    body sets the event (externally, after a short delay) and exits,
    the second acquire resolves.
    """
    bind_root = tmp_path / "bind_root"
    bind_root.mkdir(parents=True, exist_ok=True)
    ws_id = uuid7()
    ws_name = "waiter"
    (bind_root / ws_name).mkdir(parents=True, exist_ok=True)
    agent_id = uuid4()
    ws = _FakeWorkspace(id=ws_id, name=ws_name, agent_id=agent_id)

    fake_nats = FakeNatsClient()
    lease_a = WorkspaceFileLease(fake_nats, namespace="test", pod_id="pod-A")
    lease_b = WorkspaceFileLease(fake_nats, namespace="test", pod_id="pod-B")

    first_inside = asyncio.Event()
    release_first = asyncio.Event()
    timestamps: dict[str, float] = {}

    async def _first() -> None:
        sandbox = _FakeSandbox({"bind": bind_root})
        workspace_coll = _FakeWorkspaceCollection(ws)
        pool = _FakePool()
        async with bind(
            agent_id=agent_id,
            workspace_id=ws_id,
            sandbox=sandbox,
            lease=lease_a,
            workspace_collection=workspace_coll,  # type: ignore[arg-type]
            workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
            workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
            db_pool=pool,
            actor_id=uuid4(),
            correlation_id=uuid7(),
            lease_ttl_seconds=30,
            lease_max_wait_seconds=10,
        ):
            first_inside.set()
            await release_first.wait()
            timestamps["first_exit"] = asyncio.get_running_loop().time()

    async def _second() -> None:
        await first_inside.wait()
        sandbox = _FakeSandbox({"bind": bind_root})
        workspace_coll = _FakeWorkspaceCollection(ws)
        pool = _FakePool()
        async with bind(
            agent_id=agent_id,
            workspace_id=ws_id,
            sandbox=sandbox,
            lease=lease_b,
            workspace_collection=workspace_coll,  # type: ignore[arg-type]
            workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
            workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
            db_pool=pool,
            actor_id=uuid4(),
            correlation_id=uuid7(),
            lease_ttl_seconds=30,
            lease_max_wait_seconds=10,
        ):
            timestamps["second_enter"] = asyncio.get_running_loop().time()

    async def _releaser() -> None:
        await first_inside.wait()
        # let the second pod start and confirm it's blocked
        await asyncio.sleep(0.1)
        release_first.set()

    await asyncio.gather(_first(), _second(), _releaser())
    assert "first_exit" in timestamps
    assert "second_enter" in timestamps
    assert timestamps["second_enter"] >= timestamps["first_exit"], (
        "second pod entered bind body before first pod released its lease"
    )
