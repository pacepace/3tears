"""unit tests for threetears.agent.workspace.lease.WorkspaceFileLease.

covers key namespacing, length-bounded sha256 fallback, the agent's own
bind-only bucket, round-trip acquire/release through fake NATS KV,
and unwrapped propagation of :class:`LeaseUnavailable`.
"""

from __future__ import annotations

import hashlib
import re

import pytest
from uuid import UUID, uuid4

from threetears.core.testing.kv import FakeNatsClient
from threetears.agent.workspace.lease import WorkspaceFileLease
from threetears.core.coordination import LeaseHandle, LeaseUnavailable
from threetears.nats.subject_permissions import Principal, kv_key_scope_for


_SAMPLE_WORKSPACE_ID = UUID("019470a8-b5c3-7def-8123-456789abcdef")
_AGENT_ID = UUID("019470a8-b5c3-7def-8123-0000000000a1")
_OTHER_AGENT_ID = UUID("019470a8-b5c3-7def-8123-0000000000a2")

#: the bucket the hub declares for ``_AGENT_ID``, as the lease hands it to ``kv_bucket``.
_BUCKET = f"{kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_ID)}-workspace-locks"


class TestMakeKey:
    """make_key constructs namespaced keys and sha256-bounds long paths."""

    def test_short_path_produces_raw_key(self) -> None:
        """key under MAX_KEY_LEN is returned verbatim under workspace: prefix."""
        fake = FakeNatsClient(declared_buckets=(_BUCKET,))
        lease = WorkspaceFileLease(fake, agent_id=_AGENT_ID)
        result = lease.make_key(_SAMPLE_WORKSPACE_ID, "foo.yaml")
        assert result == f"workspace:{_SAMPLE_WORKSPACE_ID.hex}:foo.yaml"

    def test_long_path_produces_sha256_bounded_key(self) -> None:
        """path pushing raw key over MAX_KEY_LEN produces sha256 form."""
        fake = FakeNatsClient(declared_buckets=(_BUCKET,))
        lease = WorkspaceFileLease(fake, agent_id=_AGENT_ID)
        long_path = "A" * 500
        result = lease.make_key(_SAMPLE_WORKSPACE_ID, long_path)
        expected_digest = hashlib.sha256(long_path.encode("utf-8")).hexdigest()
        assert result == (f"workspace:{_SAMPLE_WORKSPACE_ID.hex}:sha256:{expected_digest}")
        assert re.match(r"^workspace:[0-9a-f]{32}:sha256:[0-9a-f]{64}$", result) is not None

    def test_sha256_form_is_shorter_than_raw_when_path_is_huge(self) -> None:
        """sha256 form bounds total length irrespective of input path length."""
        fake = FakeNatsClient(declared_buckets=(_BUCKET,))
        lease = WorkspaceFileLease(fake, agent_id=_AGENT_ID)
        huge_path = "x" * 10000
        result = lease.make_key(_SAMPLE_WORKSPACE_ID, huge_path)
        # workspace:{32}:sha256:{64} = 10 + 32 + 8 + 64 = 114
        assert len(result) < 200
        assert result.startswith(f"workspace:{_SAMPLE_WORKSPACE_ID.hex}:sha256:")

    def test_threshold_boundary_raw_form_still_used(self) -> None:
        """key exactly at MAX_KEY_LEN still takes the raw branch."""
        fake = FakeNatsClient(declared_buckets=(_BUCKET,))
        lease = WorkspaceFileLease(fake, agent_id=_AGENT_ID)
        # raw = "workspace:{hex}:{relative}" — len(prefix) = 10 + 32 + 1 = 43
        prefix_len = len(f"workspace:{_SAMPLE_WORKSPACE_ID.hex}:")
        filler_len = WorkspaceFileLease.MAX_KEY_LEN - prefix_len
        relative = "a" * filler_len
        result = lease.make_key(_SAMPLE_WORKSPACE_ID, relative)
        assert len(result) == WorkspaceFileLease.MAX_KEY_LEN
        assert "sha256" not in result


class TestBucketName:
    """the lock bucket is the AGENT's own, declared by the hub; the lease binds it and creates nothing.

    The lock keys name workspace ids and file paths, so a bucket every agent shared would let any
    agent list another customer's paths and create or delete any lock. The bucket is composed under
    the agent's authenticated scope exactly as the agent's grant composes it.
    """

    def test_the_bucket_is_the_agents_own_workspace_locks_bucket(self) -> None:
        """the name the lease hands kv_bucket is the one the agent pod is granted."""
        lease = WorkspaceFileLease(FakeNatsClient(), agent_id=_AGENT_ID)
        assert lease.bucket_name == _BUCKET
        assert lease.bucket_name == f"{kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT_ID)}-workspace-locks"

    def test_two_agents_never_share_a_bucket(self) -> None:
        """one agent's lease cannot name another agent's locks."""
        other = WorkspaceFileLease(FakeNatsClient(), agent_id=_OTHER_AGENT_ID)
        assert other.bucket_name != _BUCKET

    async def test_the_lease_binds_and_never_declares(self) -> None:
        """a bucket the hub never declared is refused, not created: a pod holds no stream verb."""
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, agent_id=_AGENT_ID, pod_id="pod-test")
        with pytest.raises(KeyError):
            await lease.acquire(_SAMPLE_WORKSPACE_ID, "a.yaml", max_wait_seconds=0)
        with pytest.raises(KeyError):
            await fake.kv_bucket(name=_BUCKET, create_if_missing=False)


class TestAcquireRoundTrip:
    """acquire through fake NATS returns a core LeaseHandle with expected key."""

    async def test_acquire_returns_lease_handle_with_namespaced_key(self) -> None:
        """acquire() returns handle whose key is the namespaced workspace key."""
        fake = FakeNatsClient(declared_buckets=(_BUCKET,))
        lease = WorkspaceFileLease(fake, agent_id=_AGENT_ID, pod_id="pod-test")
        handle = await lease.acquire(_SAMPLE_WORKSPACE_ID, "a/b.yaml", ttl_seconds=30)
        assert isinstance(handle, LeaseHandle)
        assert handle.holder == "pod-test"
        expected_key = f"workspace:{_SAMPLE_WORKSPACE_ID.hex}:a/b.yaml"
        assert handle.key == expected_key
        await handle.release()

    async def test_acquire_uses_namespaced_bucket_in_jetstream(self) -> None:
        """acquire writes into the agent's own workspace-locks bucket the hub declared."""
        fake = FakeNatsClient(declared_buckets=(_BUCKET,))
        lease = WorkspaceFileLease(fake, agent_id=_AGENT_ID, pod_id="pod-test")
        handle = await lease.acquire(_SAMPLE_WORKSPACE_ID, "a.yaml")
        bucket = await fake.kv_bucket(name=_BUCKET, create_if_missing=False)
        value = await bucket.get(key=handle.key)
        assert value is not None
        await handle.release()

    async def test_release_removes_entry_from_bucket(self) -> None:
        """handle.release() removes the key from the backing bucket."""
        fake = FakeNatsClient(declared_buckets=(_BUCKET,))
        lease = WorkspaceFileLease(fake, agent_id=_AGENT_ID, pod_id="pod-test")
        handle = await lease.acquire(_SAMPLE_WORKSPACE_ID, "a.yaml")
        await handle.release()
        bucket = await fake.kv_bucket(name=_BUCKET, create_if_missing=False)
        # the wrapper bucket returns ``None`` on miss instead of raising
        # KeyNotFoundError; the lease's release path leaves the entry
        # gone so ``get`` should yield ``None``.
        assert await bucket.get(key=handle.key) is None

    async def test_async_context_manager_releases_on_exit(self) -> None:
        """async with handle releases lease cleanly on context exit."""
        fake = FakeNatsClient(declared_buckets=(_BUCKET,))
        lease = WorkspaceFileLease(fake, agent_id=_AGENT_ID, pod_id="pod-test")
        handle = await lease.acquire(_SAMPLE_WORKSPACE_ID, "a.yaml")
        async with handle:
            pass
        assert handle.released is True


class TestExceptionPassthrough:
    """core lease exceptions propagate unwrapped through WorkspaceFileLease."""

    async def test_fail_fast_on_contention_raises_lease_unavailable(self) -> None:
        """second acquire with max_wait_seconds=0 on held key raises LeaseUnavailable."""
        fake = FakeNatsClient(declared_buckets=(_BUCKET,))
        first = WorkspaceFileLease(fake, agent_id=_AGENT_ID, pod_id="pod-1")
        second = WorkspaceFileLease(fake, agent_id=_AGENT_ID, pod_id="pod-2")
        held = await first.acquire(_SAMPLE_WORKSPACE_ID, "contended.yaml", ttl_seconds=30)
        try:
            with pytest.raises(LeaseUnavailable):
                await second.acquire(
                    _SAMPLE_WORKSPACE_ID,
                    "contended.yaml",
                    ttl_seconds=30,
                    max_wait_seconds=0,
                )
        finally:
            await held.release()


class TestDifferentWorkspacesDoNotCollide:
    """keys include workspace_id.hex so different workspaces never collide."""

    async def test_two_workspaces_same_relative_path_hold_independent_leases(
        self,
    ) -> None:
        """same relative_path across two workspace_ids produces distinct keys."""
        fake = FakeNatsClient(declared_buckets=(_BUCKET,))
        lease = WorkspaceFileLease(fake, agent_id=_AGENT_ID, pod_id="pod-1")
        ws_a = uuid4()
        ws_b = uuid4()
        handle_a = await lease.acquire(ws_a, "shared.yaml", ttl_seconds=30)
        handle_b = await lease.acquire(ws_b, "shared.yaml", ttl_seconds=30)
        try:
            assert handle_a.key != handle_b.key
            assert ws_a.hex in handle_a.key
            assert ws_b.hex in handle_b.key
        finally:
            await handle_a.release()
            await handle_b.release()
