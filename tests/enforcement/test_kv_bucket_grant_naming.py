"""
enforcement: a KV grant must name the bucket its opener actually creates.

A ``kv_buckets`` entry in :mod:`threetears.nats.subject_permissions` is not a
description -- :func:`threetears.nats.user_jwt.mint_user_jwt` turns each one into
``$KV.{bucket}.>`` data grants and ``KV_{bucket}`` JetStream control grants. The
string has to be the bucket's REAL, materialised name.

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
from threetears.nats.subject_permissions import Principal, build_permissions

#: Stand-in namespace, visually distinct from any bucket suffix so a mis-slice reads
#: as wrong in the failure message rather than as plausible.
_NAMESPACE = "nsprobe"


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

    # Read through the PUBLIC property rather than the private derivation. An inline
    # ``# noqa: SLF001`` would have bypassed this repo's exemption ledger entirely, and that
    # channel has already silently lost entries here; not needing the waiver is better than
    # recording one.
    suffix = KVLease(nats_client=object(), pod_id="probe").bucket_name  # type: ignore[arg-type]
    assert "_" not in suffix, (
        f"KVLease's default is a SUFFIX that kv_bucket prefixes; {suffix!r} looks like it has "
        f"baked in a namespace of its own, which is how the name gained one twice."
    )
    granted = build_permissions(Principal.TOOL_POD, pod_id="pod-1", conn_id="conn-1").kv_buckets
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
    granted = build_permissions(Principal.REGISTRY, conn_id="conn-1").kv_buckets
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
        (Principal.AGENT_POD, {"agent_id": "agent-1", "pod_id": "pod-1", "conn_id": "conn-1"}),
        (Principal.HUB, {"conn_id": "conn-1"}),
        (Principal.GATEWAY, {"conn_id": "conn-1"}),
    ):
        granted = build_permissions(principal, **kwargs).kv_buckets
        assert f"{_NAMESPACE}-{_EPOCH_BUCKET}" in granted, (
            f"{principal} is not granted the epoch bucket "
            f"('{_NAMESPACE}-{_EPOCH_BUCKET}'); it holds {list(granted)}. Every epoch read "
            f"and bump from this principal would block to its deadline and read as an "
            f"unreachable broker."
        )
