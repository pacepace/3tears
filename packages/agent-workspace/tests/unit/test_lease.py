"""unit tests for threetears.agent.workspace.lease.WorkspaceFileLease.

covers key namespacing, length-bounded sha256 fallback, env-driven
default bucket name, round-trip acquire/release through fake NATS KV,
and unwrapped propagation of :class:`LeaseUnavailable`.
"""

from __future__ import annotations

import hashlib
import re

import pytest
from uuid import UUID, uuid4

from _fake_kv import FakeNatsClient  # type: ignore[import-not-found]
from threetears.agent.workspace.lease import WorkspaceFileLease
from threetears.core.coordination import LeaseHandle, LeaseUnavailable


_SAMPLE_WORKSPACE_ID = UUID("019470a8-b5c3-7def-8123-456789abcdef")


class TestMakeKey:
    """make_key constructs namespaced keys and sha256-bounds long paths."""

    def test_short_path_produces_raw_key(self) -> None:
        """key under MAX_KEY_LEN is returned verbatim under workspace: prefix."""
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, namespace="ns")
        result = lease.make_key(_SAMPLE_WORKSPACE_ID, "foo.yaml")
        assert result == f"workspace:{_SAMPLE_WORKSPACE_ID.hex}:foo.yaml"

    def test_long_path_produces_sha256_bounded_key(self) -> None:
        """path pushing raw key over MAX_KEY_LEN produces sha256 form."""
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, namespace="ns")
        long_path = "A" * 500
        result = lease.make_key(_SAMPLE_WORKSPACE_ID, long_path)
        expected_digest = hashlib.sha256(long_path.encode("utf-8")).hexdigest()
        assert result == (f"workspace:{_SAMPLE_WORKSPACE_ID.hex}:sha256:{expected_digest}")
        assert re.match(r"^workspace:[0-9a-f]{32}:sha256:[0-9a-f]{64}$", result) is not None

    def test_sha256_form_is_shorter_than_raw_when_path_is_huge(self) -> None:
        """sha256 form bounds total length irrespective of input path length."""
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, namespace="ns")
        huge_path = "x" * 10000
        result = lease.make_key(_SAMPLE_WORKSPACE_ID, huge_path)
        # workspace:{32}:sha256:{64} = 10 + 32 + 8 + 64 = 114
        assert len(result) < 200
        assert result.startswith(f"workspace:{_SAMPLE_WORKSPACE_ID.hex}:sha256:")

    def test_threshold_boundary_raw_form_still_used(self) -> None:
        """key exactly at MAX_KEY_LEN still takes the raw branch."""
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, namespace="ns")
        # raw = "workspace:{hex}:{relative}" — len(prefix) = 10 + 32 + 1 = 43
        prefix_len = len(f"workspace:{_SAMPLE_WORKSPACE_ID.hex}:")
        filler_len = WorkspaceFileLease.MAX_KEY_LEN - prefix_len
        relative = "a" * filler_len
        result = lease.make_key(_SAMPLE_WORKSPACE_ID, relative)
        assert len(result) == WorkspaceFileLease.MAX_KEY_LEN
        assert "sha256" not in result


class TestBucketName:
    """default bucket name derives from namespace arg or env fallback."""

    def test_explicit_namespace_forms_workspace_locks_bucket(self) -> None:
        """namespace='acme' -> bucket 'acme_workspace_locks'."""
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, namespace="acme")
        assert lease.bucket_name == "acme_workspace_locks"

    def test_env_namespace_forms_workspace_locks_bucket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE=prod14 -> 'prod14_workspace_locks'."""
        monkeypatch.setenv("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", "prod14")
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake)
        assert lease.bucket_name == "prod14_workspace_locks"

    def test_env_unset_falls_back_to_workspace_locks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env unset + no arg -> unscoped 'workspace_locks' (no KeyError)."""
        monkeypatch.delenv("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", raising=False)
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake)
        assert lease.bucket_name == "workspace_locks"


class TestAcquireRoundTrip:
    """acquire through fake NATS returns a core LeaseHandle with expected key."""

    async def test_acquire_returns_lease_handle_with_namespaced_key(self) -> None:
        """acquire() returns handle whose key is the namespaced workspace key."""
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, namespace="ns", pod_id="pod-test")
        handle = await lease.acquire(_SAMPLE_WORKSPACE_ID, "a/b.yaml", ttl_seconds=30)
        assert isinstance(handle, LeaseHandle)
        assert handle.holder == "pod-test"
        expected_key = f"workspace:{_SAMPLE_WORKSPACE_ID.hex}:a/b.yaml"
        assert handle.key == expected_key
        await handle.release()

    async def test_acquire_uses_namespaced_bucket_in_jetstream(self) -> None:
        """acquire opens the '{namespace}_workspace_locks' bucket on fake JS."""
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, namespace="ns", pod_id="pod-test")
        handle = await lease.acquire(_SAMPLE_WORKSPACE_ID, "a.yaml")
        bucket = await fake.jetstream().key_value("ns_workspace_locks")
        entry = await bucket.get(handle.key)
        assert entry is not None
        await handle.release()

    async def test_release_removes_entry_from_bucket(self) -> None:
        """handle.release() removes the key from the backing bucket."""
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, namespace="ns", pod_id="pod-test")
        handle = await lease.acquire(_SAMPLE_WORKSPACE_ID, "a.yaml")
        await handle.release()
        bucket = await fake.jetstream().key_value("ns_workspace_locks")
        from nats.js.errors import KeyNotFoundError

        with pytest.raises(KeyNotFoundError):
            await bucket.get(handle.key)

    async def test_async_context_manager_releases_on_exit(self) -> None:
        """async with handle releases lease cleanly on context exit."""
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, namespace="ns", pod_id="pod-test")
        handle = await lease.acquire(_SAMPLE_WORKSPACE_ID, "a.yaml")
        async with handle:
            pass
        assert handle.released is True


class TestExceptionPassthrough:
    """core lease exceptions propagate unwrapped through WorkspaceFileLease."""

    async def test_fail_fast_on_contention_raises_lease_unavailable(self) -> None:
        """second acquire with max_wait_seconds=0 on held key raises LeaseUnavailable."""
        fake = FakeNatsClient()
        first = WorkspaceFileLease(fake, namespace="ns", pod_id="pod-1")
        second = WorkspaceFileLease(fake, namespace="ns", pod_id="pod-2")
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
        fake = FakeNatsClient()
        lease = WorkspaceFileLease(fake, namespace="ns", pod_id="pod-1")
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
