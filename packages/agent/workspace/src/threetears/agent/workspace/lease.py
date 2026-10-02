"""WorkspaceFileLease — thin wrapper around core :class:`KVLease`.

provides per-workspace, per-file distributed mutex semantics in the
serving agent's OWN lock bucket, which the hub declares and the lease only
binds, by namespacing KV keys under ``workspace:{workspace_id.hex}:{relative_path}``
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
from typing import Any
from uuid import UUID

from threetears.core.coordination import KVLease, LeaseHandle
from threetears.nats.subject_permissions import WORKSPACE_LOCKS_BUCKET_SUFFIX, agent_platform_bucket_suffix

__all__ = [
    "WorkspaceFileLease",
]


class WorkspaceFileLease:
    """per-workspace-file distributed lock built on core :class:`KVLease`.

    keys the underlying KV bucket with
    ``workspace:{workspace_id.hex}:{relative_path}`` so two tool calls
    targeting the same file in the same workspace serialize cleanly while
    distinct workspaces or distinct files remain concurrent.

    :cvar MAX_KEY_LEN: practical NATS KV key length ceiling; raw keys
        longer than this fall through to the sha256-bounded form so bucket
        writes never fail on length.
    """

    MAX_KEY_LEN = 200

    def __init__(
        self,
        nats_client: Any,
        *,
        agent_id: UUID,
        pod_id: str | None = None,
    ) -> None:
        """configure wrapper over the agent's OWN workspace-locks bucket, bind-only.

        The bucket is ``{ns}-{scope}-workspace-locks``, where ``scope`` is the agent's L2 key scope
        (:func:`threetears.nats.subject_permissions.agent_platform_bucket_suffix` renders the name
        without the ``{ns}-`` the client layers on). It is the agent's own for a reason: the lock
        keys name workspace ids and file paths, so a bucket every agent shared would let any agent
        list another customer's paths and create or delete any lock. It is the name the agent pod's
        grant covers, and the hub declares it for every agent.

        BIND-ONLY: a pod holds no stream-management verb, so the lease never issues
        ``STREAM.CREATE``; a bucket the hub has not declared fails the first acquire loudly rather
        than costing a JetStream deadline first.

        :param nats_client: connected canonical NATS wrapper client
        :ptype nats_client: Any
        :param agent_id: the agent whose workspace files are locked -- the agent this pod serves
        :ptype agent_id: UUID
        :param pod_id: holder identifier forwarded to :class:`KVLease`;
            None delegates auto-generation to the core factory
        :ptype pod_id: str | None
        :return: None
        :rtype: None
        :raises ValueError: if ``agent_id`` is not a uuid
        """
        bucket_name = agent_platform_bucket_suffix(agent_id, WORKSPACE_LOCKS_BUCKET_SUFFIX)
        self._kvlease = KVLease(nats_client, bucket_name=bucket_name, pod_id=pod_id, create_if_missing=False)

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
        by :meth:`make_key`. returns the raw core :class:`LeaseHandle`
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
        key = self.make_key(workspace_id, relative_path)
        return await self._kvlease.acquire(
            key,
            ttl_seconds=ttl_seconds,
            max_wait_seconds=max_wait_seconds,
        )

    def make_key(self, workspace_id: UUID, relative_path: str) -> str:
        """build namespaced KV key for ``(workspace_id, relative_path)``.

        raw form ``workspace:{workspace_id.hex}:{relative_path}`` is used
        when total length is within :attr:`MAX_KEY_LEN`; otherwise the
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
        if len(raw) <= self.MAX_KEY_LEN:
            result = raw
        else:
            digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
            result = f"workspace:{workspace_id.hex}:sha256:{digest}"
        return result
