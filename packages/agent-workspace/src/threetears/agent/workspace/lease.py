"""WorkspaceFileLease — thin wrapper around core :class:`KVLease`.

provides per-workspace, per-file distributed mutex semantics by
namespacing KV keys under ``workspace:{workspace_id.hex}:{relative_path}``
(or a sha256-bounded variant when the raw key would exceed the NATS KV
practical limit). all ownership-token, TTL, and CAS semantics are
inherited from core :class:`KVLease`; this wrapper only constructs
workspace-shaped keys and exposes a tighter acquire signature.

exception types from the core primitive (``LeaseUnavailable``,
``LeaseTimeout``, ``LeaseLost``) propagate unwrapped so tool callers see
the same narrow set of lease errors regardless of which wrapper minted
the handle.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any
from uuid import UUID

from threetears.core.coordination import KVLease, LeaseHandle


class WorkspaceFileLease:
    """per-workspace-file distributed lock built on core :class:`KVLease`.

    keys the underlying KV bucket with
    ``workspace:{workspace_id.hex}:{relative_path}`` so two tool calls
    targeting the same file in the same workspace serialize cleanly while
    distinct workspaces or distinct files remain concurrent.

    :cvar _MAX_KEY_LEN: practical NATS KV key length ceiling; raw keys
        longer than this fall through to the sha256-bounded form so bucket
        writes never fail on length.
    """

    _MAX_KEY_LEN = 200

    def __init__(
        self,
        nats_client: Any,
        namespace: str | None = None,
        pod_id: str | None = None,
    ) -> None:
        """configure wrapper; build bucket name and core lease factory.

        bucket name resolution precedence:

        1. ``f"{namespace}_workspace_locks"`` when ``namespace`` is given
        2. ``f"{env}_workspace_locks"`` when ``FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE`` is set
        3. ``"workspace_locks"`` as unscoped fallback

        this mirrors :meth:`KVLease._default_bucket_name` behaviour —
        deliberately using :meth:`os.environ.get` rather than subscript
        access so unit tests and local-dev runs without the platform env
        var do not blow up with :class:`KeyError` at wrapper construction.

        :param nats_client: connected NATS client exposing ``jetstream()``
        :ptype nats_client: Any
        :param namespace: NATS subject namespace override; None reads env var
        :ptype namespace: str | None
        :param pod_id: holder identifier forwarded to :class:`KVLease`;
            None delegates auto-generation to the core factory
        :ptype pod_id: str | None
        :return: None
        :rtype: None
        """
        effective_namespace = (
            namespace
            if namespace is not None
            else os.environ.get("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE")
        )
        bucket_name = (
            f"{effective_namespace}_workspace_locks"
            if effective_namespace
            else "workspace_locks"
        )
        self._kvlease = KVLease(
            nats_client, bucket_name=bucket_name, pod_id=pod_id
        )

    @property
    def bucket_name(self) -> str:
        """return bucket name used by the underlying :class:`KVLease`.

        :return: configured bucket name
        :rtype: str
        """
        return self._kvlease.bucket_name

    @property
    def pod_id(self) -> str:
        """return holder identifier used by the underlying :class:`KVLease`.

        :return: holder identifier
        :rtype: str
        """
        return self._kvlease.pod_id

    async def acquire(
        self,
        workspace_id: UUID,
        relative_path: str,
        ttl_seconds: int = 30,
        max_wait_seconds: int = 60,
    ) -> LeaseHandle:
        """acquire lease for ``(workspace_id, relative_path)``.

        delegates to :meth:`KVLease.acquire` with a namespaced key built
        by :meth:`_make_key`. returns the raw core :class:`LeaseHandle`
        so callers use the same refresh/release surface regardless of
        which wrapper created the lease.

        :param workspace_id: workspace-scope identifier
        :ptype workspace_id: UUID
        :param relative_path: relative filesystem path identifying the file
        :ptype relative_path: str
        :param ttl_seconds: lease TTL (expiry past which entry is stale)
        :ptype ttl_seconds: int
        :param max_wait_seconds: total seconds caller is willing to block;
            0 triggers fail-fast on contention
        :ptype max_wait_seconds: int
        :return: core :class:`LeaseHandle` for the acquired lease
        :rtype: LeaseHandle
        :raises LeaseUnavailable: if ``max_wait_seconds == 0`` and key is held
        :raises LeaseTimeout: if deadline elapses before lease becomes free
        """
        key = self._make_key(workspace_id, relative_path)
        return await self._kvlease.acquire(
            key,
            ttl_seconds=ttl_seconds,
            max_wait_seconds=max_wait_seconds,
        )

    def _make_key(self, workspace_id: UUID, relative_path: str) -> str:
        """build namespaced KV key for ``(workspace_id, relative_path)``.

        raw form ``workspace:{workspace_id.hex}:{relative_path}`` is used
        when total length is within :attr:`_MAX_KEY_LEN`; otherwise the
        relative path is sha256-hashed so the key length is bounded
        regardless of input path length. workspace id remains readable in
        both forms for operational debugging.

        :param workspace_id: workspace-scope identifier
        :ptype workspace_id: UUID
        :param relative_path: relative filesystem path identifying the file
        :ptype relative_path: str
        :return: namespaced KV key safe for NATS KV bucket writes
        :rtype: str
        """
        raw = f"workspace:{workspace_id.hex}:{relative_path}"
        if len(raw) <= self._MAX_KEY_LEN:
            result = raw
        else:
            digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
            result = f"workspace:{workspace_id.hex}:sha256:{digest}"
        return result
