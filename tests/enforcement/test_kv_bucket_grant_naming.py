"""
enforcement: a KV grant must name the bucket its opener actually creates.

A ``JsResource.kv(...)`` entry in :mod:`threetears.nats.subject_permissions` is not
a description -- :func:`threetears.nats.user_jwt.mint_user_jwt` turns each one into
a ``$KV.{bucket}`` publish grant (whole-subtree, or narrowed to the principal's key
scope) and ``KV_{bucket}`` JetStream control grants. The string has to be the
bucket's REAL, materialised name.

**The failure is silent in both directions, which is why it needs a guard.** Grant a
name nothing creates and the principal is authorised against a bucket that does not
exist while the one it opens is ungranted; the op then blocks to its deadline rather
than raising, indistinguishable from an unreachable broker.
``test_user_jwt_scoped_grant_live.py`` exists for that shape. Remove a grant something
does need and you get the same silence.

:mod:`threetears.nats._diagnostics` now names this cause in the log -- from the
server's own refusal frame, and again at the deadline -- so the diagnosis is no
longer only in a reader's head. It does not make the grant correct, which is what
this guard is for: a log line an operator must read at the right moment is a worse
place to catch a wrong name than a test that fails before the deploy.

**Three naming conventions are live here, and all three are legitimate.** This was
learned the expensive way: an earlier version of this guard rejected the third as
malformed, which would have removed grants the hub depends on.

* Opened through :meth:`threetears.nats.kv.KvCapable.kv_bucket`, which takes a SUFFIX
  and layers the connection's ``{namespace}-`` over it. ``BaseCollection``,
  ``KVLease``, ``ReplayGuard``, ``TokenBucket``. The grant is ``{ns}-<suffix>``.
* Opened by a direct ``js.key_value(bucket=...)`` with a bare constant, which receives
  no prefix at all. ``threetears.registry.server``'s catalog. The grant is that bare
  name.
* Opened by a direct ``js.create_key_value(bucket=...)`` with a name the component
  builds itself as ``f"{namespace}_thing"``. The hub's ``AgentConfigKV`` does this and
  its own docstring calls the result platform-historical. The grant is that verbatim
  name, underscore included.

So no rule can be written over the SHAPE of a grant string -- ``{ns}_agent_config``
and a typo for ``{ns}-agent_config`` are indistinguishable by inspection, and only
one of them is wrong. What this guard pins instead is the case where both sides live
in this repository and can therefore be compared to each other.
"""

from __future__ import annotations

import pytest
from threetears.nats.subject_permissions import Principal, build_permissions, kv_bucket_names

#: Stand-in namespace, visually distinct from any bucket suffix so a mis-slice reads
#: as wrong in the failure message rather than as plausible.
_NAMESPACE = "nsprobe"

#: Pod ids are UUIDs because the L2 key scope is derived from them, and a scope derived from
#: anything else is not provably collision-free -- ``kv_key_scope_for`` refuses a slug outright.
_AGENT_ID = "019470a8-b5c3-7def-8123-0000000000a1"
_POD_ID = "01947100-0000-7000-8000-0000000000b1"


@pytest.fixture(autouse=True)
def _namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the subject namespace so grants render with a known prefix."""
    from threetears.nats import subjects

    monkeypatch.setattr(subjects, "_default_namespace", _NAMESPACE)


def test_the_lease_bucket_a_tool_pod_is_granted_is_the_one_kvlease_opens() -> None:
    """The display claim's grant and ``KVLease``'s own default must agree.

    Pinned as a PAIR rather than as two independent literals, because the defect this
    replaced was precisely the two drifting apart: ``KVLease`` read the namespace and
    returned ``{ns}_leases``, ``kv_bucket`` layered its own ``{ns}-`` over that, and
    the bucket that materialised carried the namespace twice while the grant named
    neither form. A test asserting either side alone passes throughout that.

    Both sides live in this repository, which is what makes them comparable here --
    unlike the grants whose openers are in a consumer, where no static check can
    reach.

    A pod that cannot open this bucket fails hard on its first claim rather than
    downgrading: ``KVLease.acquire`` defers the open to first use and that open raises
    ``KvError`` after a JetStream timeout. (``lease=None`` is a different path -- a
    platform passing no lease at all -- and that one does serve a display unclaimed.)
    """
    from threetears.core.coordination.lease import KVLease

    # Read through the PUBLIC property rather than the private derivation. An inline SLF001
    # noqa waiver would have bypassed this repo's exemption ledger entirely, and that
    # channel has already silently lost entries here; not needing the waiver is better than
    # recording one.
    suffix = KVLease(nats_client=object(), pod_id="probe").bucket_name  # type: ignore[arg-type]
    assert "_" not in suffix, (
        f"KVLease's default is a SUFFIX that kv_bucket prefixes; {suffix!r} looks like it has "
        f"baked in a namespace of its own, which is how the name gained one twice."
    )
    granted = kv_bucket_names(build_permissions(Principal.TOOL_POD, pod_id=_POD_ID, conn_id="conn-1"))
    assert f"{_NAMESPACE}-{suffix}" in granted, (
        f"a tool pod is not granted the bucket KVLease actually opens ('{_NAMESPACE}-{suffix}'); "
        f"it holds {list(granted)}."
    )


def test_the_registry_catalog_grant_matches_the_bucket_the_registry_opens() -> None:
    """The catalog grant is UNPREFIXED, and that is a property worth pinning.

    ``RegistryServer`` opens its catalog with a direct ``js.key_value(bucket=...)``
    rather than through ``kv_bucket``, so no namespace is ever applied. The grant is
    therefore a bare name, and "normalising" it to ``{ns}-tool_catalog`` to match the
    file's other entries would silently point the registry's authorisation at a bucket
    that does not exist.

    Read out of the server's own default rather than restated, so the two cannot drift.
    """
    import inspect

    from threetears.registry.server import RegistryServer

    default = inspect.signature(RegistryServer.__init__).parameters["kv_bucket"].default
    granted = kv_bucket_names(build_permissions(Principal.REGISTRY, conn_id="conn-1"))
    assert default in granted, (
        f"the registry opens bucket {default!r} with a direct js.key_value (no namespace prefix), "
        f"but its grants are {list(granted)}."
    )
    assert f"{_NAMESPACE}-{default}" not in granted, (
        f"the catalog grant has been prefixed to '{_NAMESPACE}-{default}', which names a bucket "
        f"the registry never opens."
    )


def test_the_epoch_bucket_the_pods_are_granted_is_the_one_epochclient_opens() -> None:
    """The epoch grant and the bucket the client actually opens must agree.

    Pinned as a PAIR for the reason this file exists: a missing or misspelled
    KV grant does not raise. The JetStream call blocks to its deadline and
    surfaces as an unreachable broker, so the symptom points at the network
    rather than at a permission the grant never carried. Asserting either side
    alone passes throughout that.

    Both halves live in this repository, which is what makes them comparable
    here: ``EpochClient`` names the bucket suffix, and ``subject_permissions``
    grants it to the three principals that bump or read an epoch.
    """
    from threetears.epoch.client import _EPOCH_BUCKET

    for principal, kwargs in (
        (Principal.AGENT_POD, {"agent_id": _AGENT_ID, "pod_id": "pod-1", "conn_id": "conn-1"}),
        (Principal.HUB, {"conn_id": "conn-1"}),
        (Principal.GATEWAY, {"conn_id": "conn-1"}),
    ):
        granted = kv_bucket_names(build_permissions(principal, **kwargs))
        assert f"{_NAMESPACE}-{_EPOCH_BUCKET}" in granted, (
            f"{principal} is not granted the epoch bucket "
            f"('{_NAMESPACE}-{_EPOCH_BUCKET}'); it holds {list(granted)}. Every epoch read "
            f"and bump from this principal would block to its deadline and read as an "
            f"unreachable broker."
        )


def test_no_pod_is_granted_the_bucket_the_checkpointer_l2_defaults_to() -> None:
    """The shared checkpoint bucket is granted to NO pod, and the saver's default names it.

    ``ThreeTierCheckpointSaver`` defaults ``l2_bucket`` to ``_DEFAULT_L2_BUCKET``, which
    ``kv_bucket`` materialises as ``{ns}-checkpoints`` -- one bucket keyed
    ``[<customer>/]<thread>[.<ns>]`` with no owner token, so a grant on it was a read of every
    agent's conversation state. Nothing on the platform reads or writes it: the agent runtime's
    checkpointer runs on L3 alone, and a host that wants a checkpoint L2 declares a coordination
    bucket of its own and passes it as ``l2_bucket`` (the survey's is ``{ns}-{scope}-checkpoints``).

    Pinned as a PAIR, read out of the saver's own default, so a pod that ever opens the default
    finds it ungranted in a test rather than as a JetStream deadline in production.
    """
    from threetears.langgraph.checkpoint import _DEFAULT_L2_BUCKET

    assert "-" not in _DEFAULT_L2_BUCKET and "_" not in _DEFAULT_L2_BUCKET, (
        f"{_DEFAULT_L2_BUCKET!r} looks like it has baked in a namespace of its own; it is a "
        f"SUFFIX that kv_bucket prefixes."
    )
    for principal in (Principal.AGENT_POD, Principal.TOOL_POD):
        granted = kv_bucket_names(_permissions_for(principal))  # type: ignore[arg-type]
        assert f"{_NAMESPACE}-{_DEFAULT_L2_BUCKET}" not in granted, (
            f"{principal} is granted the shared checkpoint bucket ('{_NAMESPACE}-{_DEFAULT_L2_BUCKET}'), "
            f"which carries every agent's thread state under keys no grant can narrow to one owner."
        )


def test_the_shared_pod_buckets_are_granted_under_the_scope_their_openers_key_with() -> None:
    """Each shared pod bucket's grant names the bucket its opener opens, narrowed to the opener's scope.

    ``KV_OWNER_KEYS`` narrows every route to ``{scope}.>``, so the opener must key with the SAME
    scope the grant was minted with, or every call on the bucket blocks to its deadline. Both
    sides live here: the tool server derives the nonce scope from its pod id, the memory extractor
    opens ``ratelimits`` by default, ``KVLease`` opens ``leases`` by default.
    """
    from threetears.agent.memory.extraction import MemoryExtractor
    from threetears.agent.tools.server import ToolServer
    from threetears.core.coordination.lease import KVLease
    from threetears.nats import Subjects
    from threetears.nats.subject_permissions import JsCapability, kv_key_scope_for

    import inspect
    from uuid import UUID

    def owner_scope_of(permissions: object, bucket: str) -> str | None:
        found = [r for r in permissions.js_resources if r.name == bucket]  # type: ignore[attr-defined]
        assert len(found) == 1, (bucket, found)
        assert found[0].capability is JsCapability.KV_OWNER_KEYS, found[0]
        return found[0].scope

    agent = _permissions_for(Principal.AGENT_POD)
    tool = _permissions_for(Principal.TOOL_POD)
    in_process = ToolServer(nats_url="nats://x", pod_id=Subjects.agent_inprocess_pod_id(UUID(_AGENT_ID), "i1"))
    tool_pod = ToolServer(nats_url="nats://x", pod_id=_POD_ID)
    assert owner_scope_of(agent, f"{_NAMESPACE}-proxy_assertion_nonces") == in_process.assertion_nonce_key_scope
    assert owner_scope_of(tool, f"{_NAMESPACE}-proxy_assertion_nonces") == tool_pod.assertion_nonce_key_scope

    ratelimits = inspect.signature(MemoryExtractor.__init__).parameters["rate_limit_bucket"].default
    assert owner_scope_of(agent, f"{_NAMESPACE}-{ratelimits}") == kv_key_scope_for(
        Principal.AGENT_POD, agent_id=_AGENT_ID
    )

    leases = KVLease(nats_client=object(), pod_id="probe").bucket_name  # type: ignore[arg-type]
    assert owner_scope_of(tool, f"{_NAMESPACE}-{leases}") == kv_key_scope_for(Principal.TOOL_POD, pod_id=_POD_ID)


def test_an_agent_pod_publishes_audit_events_and_holds_no_grant_on_the_audit_stream() -> None:
    """An agent pod EMITS audit events and consumes none, so it holds no grant on the stream.

    ``AGENT_POD`` is granted ``audit.tool.call`` as a PUBLISH subject, which is the
    emitting half, and a JetStream publish needs nothing more: it is a core publish
    acknowledged on the publisher's own inbox. The audit stream is declared and
    consumed by the hub (``start_audit_persister``), and no pod runs a consumer on it.

    Least privilege, and the reason this pin changed: a consumer grant on
    ``{ns}-audit`` let a pod create a consumer over EVERY agent's audit events --
    other customers' included -- and a stream-management grant let it create the
    stream it then read. The emitting half is pinned beside the absence so the
    narrowing cannot be mistaken for dropping the publish.
    """
    granted = build_permissions(Principal.AGENT_POD, agent_id=_AGENT_ID, pod_id=_POD_ID)
    streams = [r.stream_name for r in granted.js_resources]
    assert f"{_NAMESPACE}-audit" not in streams, (
        f"an agent pod holds a JetStream grant on the audit stream ('{_NAMESPACE}-audit'), which it only "
        f"publishes to; it holds {streams}."
    )
    assert f"KV_{_NAMESPACE}-audit" not in streams, "the audit stream is a plain stream, not a KV bucket"
    assert f"{_NAMESPACE}.audit.tool.call" in granted.publish, "the agent pod must still emit its audit events"


#: representative ids per principal, so every member of the enum resolves to a concrete
#: allow-list. Keyed by the enum itself: a member added without an entry raises a ``KeyError``
#: in :func:`_permissions_for` rather than being quietly skipped by a hand-written roster.
_IDS: dict[Principal, dict[str, str]] = {
    Principal.AGENT_POD: {"agent_id": _AGENT_ID, "pod_id": _POD_ID},
    Principal.TOOL_POD: {"pod_id": _POD_ID},
    Principal.REGISTRY: {"conn_id": "conn-1"},
    Principal.HUB: {"conn_id": "conn-1"},
    Principal.GATEWAY: {"conn_id": "conn-1"},
    Principal.CHANNEL_ADAPTER: {"conn_id": "conn-1"},
    Principal.AGENT_ROUTER: {"conn_id": "conn-1"},
    Principal.DATASET_EXECUTOR: {"conn_id": "conn-1"},
}


def _permissions_for(principal: Principal) -> object:
    """resolve one principal's allow-list from its representative ids.

    :param principal: the connection identity class
    :ptype principal: Principal
    :return: the resolved permissions
    :rtype: PrincipalPermissions
    """
    return build_permissions(principal, **_IDS[principal])


@pytest.mark.parametrize("principal", list(Principal))
def test_the_collections_bucket_every_principal_holds_is_the_one_basecollection_opens(
    principal: Principal,
) -> None:
    """The shared L2 bucket's grant and ``BaseCollection``'s own suffix must agree, for ALL.

    Two properties in one assertion, and both belong here rather than in a unit test over one
    principal.

    **The name.** ``BaseCollection`` opens its L2 through
    :meth:`~threetears.nats.kv.KvCapable.kv_bucket`, which takes a SUFFIX and layers the
    connection's ``{namespace}-`` over it, so the bucket that materialises is
    ``{ns}-collections``. Read out of ``L2_BUCKET_SUFFIX`` rather than restated, so a rename on
    either side fails here instead of becoming a JetStream call that blocks to its deadline.

    **The roster.** Every principal runs collections -- that is what stops L2 being a per-class
    privilege -- so this is parametrized over ``list(Principal)`` and never over a list somebody
    typed. A typed roster asserts "these principals" while reading as "every principal", and the
    two diverge silently the moment a member is added. Several of this platform's collections are
    ``L3 = None``, where L2 IS the store, so an ungranted principal there loses rows rather than
    missing a cache.
    """
    from threetears.core.collections.base import BaseCollection

    suffix = BaseCollection.L2_BUCKET_SUFFIX
    assert "-" not in suffix and "_" not in suffix, (
        f"{suffix!r} looks like it has baked in a namespace of its own; it is a SUFFIX that kv_bucket prefixes."
    )
    granted = kv_bucket_names(_permissions_for(principal))  # type: ignore[arg-type]
    assert f"{_NAMESPACE}-{suffix}" in granted, (
        f"{principal} is not granted the shared collections bucket "
        f"('{_NAMESPACE}-{suffix}'); it holds {list(granted)}. Every L2 read and write from this "
        f"principal would block to its deadline and read as an unreachable broker."
    )
    assert suffix not in granted, (
        f"{principal}'s collections grant is the bare name {suffix!r}, which names a bucket no "
        f"host opens through kv_bucket."
    )
