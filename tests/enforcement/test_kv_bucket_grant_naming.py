"""
enforcement: every KV bucket grant must name a bucket some opener actually produces.

A ``kv_buckets`` entry in :mod:`threetears.nats.subject_permissions` is not a
description -- :func:`threetears.nats.user_jwt.mint_user_jwt` turns each one into
``$KV.{bucket}.>`` data grants and ``KV_{bucket}`` JetStream control grants. So the
string has to be the bucket's REAL, materialised name. Name a bucket nothing opens
and the principal is granted access to something that does not exist, while the
bucket it does open is ungranted.

**That failure is silent, which is why it needs a guard rather than care.** A missing
KV grant does not raise. The op publishes to a ``$JS.API`` subject the connection
lacks, the server drops it, and the caller blocks to its deadline -- the same symptom
as an unreachable broker. ``test_user_jwt_scoped_grant_live.py`` exists because of
that shape, and this guard is its static half.

**Two naming conventions are legitimate in this codebase, and one spelling is not.**

* Opened through :meth:`threetears.nats.kv.KvCapable.kv_bucket` -- which takes a
  SUFFIX and layers the connection's own ``{namespace}-`` over it. Almost everything
  uses this: ``BaseCollection.L2_BUCKET_SUFFIX``, ``KVLease``, ``ReplayGuard``,
  ``TokenBucket``. The grant is ``{ns}-<suffix>``.
* Opened by a direct ``js.key_value(bucket=...)``, which applies no prefix at all.
  ``threetears.registry.server`` does this for its catalog. The grant is the bare
  name.

The spelling this guard rejects is ``{ns}_<name>`` -- an underscore straight after
the namespace. It matches NEITHER convention, so a grant carrying it can only ever
name a bucket that no code path creates. Every instance found when this guard was
written was one of two mistakes: a Postgres TABLE name pasted in as though each
entity had its own bucket (they share one, ``{ns}-collections``), or a default that
read the namespace itself and let the transport prefix it a second time, so the
bucket materialised with the namespace twice.
"""

from __future__ import annotations

import re

import pytest
from threetears.nats.subject_permissions import Principal, build_permissions

#: Stand-in namespace. Chosen to be visually distinct from any bucket suffix so a
#: mis-slice shows up in the failure message rather than reading as plausible.
_NAMESPACE = "nsprobe"

#: A grant is malformed when the namespace is followed directly by ``_``. The two
#: legitimate shapes are ``{ns}-<suffix>`` and a bare name carrying no namespace.
_MALFORMED = re.compile(rf"^{re.escape(_NAMESPACE)}_")

#: Every principal that can carry KV grants, with the ids each one requires.
_PRINCIPALS: tuple[tuple[Principal, dict[str, str]], ...] = (
    (Principal.AGENT_POD, {"agent_id": "agent-1", "pod_id": "pod-1", "conn_id": "conn-1"}),
    (Principal.TOOL_POD, {"pod_id": "pod-1", "conn_id": "conn-1"}),
    (Principal.REGISTRY, {"conn_id": "conn-1"}),
    (Principal.HUB, {"conn_id": "conn-1"}),
    (Principal.GATEWAY, {"conn_id": "conn-1"}),
    (Principal.CHANNEL_ADAPTER, {"conn_id": "conn-1"}),
)


@pytest.fixture(autouse=True)
def _namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the subject namespace so grants render with a known prefix."""
    from threetears.nats import subjects

    monkeypatch.setattr(subjects, "_default_namespace", _NAMESPACE)


@pytest.mark.parametrize(("principal", "ids"), _PRINCIPALS, ids=lambda v: getattr(v, "value", ""))
def test_no_grant_uses_the_namespace_underscore_spelling(principal: Principal, ids: dict[str, str]) -> None:
    """No principal may grant a bucket named ``{ns}_...``.

    Asserts on the RENDERED grant rather than the source text, so a new principal or
    a computed name is covered without this test being edited.
    """
    granted = build_permissions(principal, **ids).kv_buckets
    malformed = [b for b in granted if _MALFORMED.match(b)]
    assert not malformed, (
        f"{principal.value} grants {malformed}, which names a bucket no opener produces. "
        f"A bucket opened via kv_bucket() materialises as '{_NAMESPACE}-<suffix>'; one opened by a "
        f"direct js.key_value() carries no namespace at all. Neither produces '{_NAMESPACE}_'."
    )


def test_the_lease_bucket_a_tool_pod_is_granted_is_the_one_kvlease_opens() -> None:
    """The display claim's grant and ``KVLease``'s own default must agree.

    Pinned as a PAIR rather than as two independent literals: the defect this replaced
    was the two drifting apart, and a test that checked either one alone would have
    passed throughout. ``KVLease`` returns a suffix and ``kv_bucket`` prefixes it, so the
    grant is that composition and nothing else.
    """
    from threetears.core.coordination.lease import KVLease

    suffix = KVLease._default_bucket_name()  # noqa: SLF001
    assert "_" not in suffix, (
        f"KVLease's default bucket is a SUFFIX that kv_bucket prefixes; {suffix!r} looks like it "
        f"has baked in a namespace of its own, which is how the name gained one twice."
    )
    granted = build_permissions(Principal.TOOL_POD, pod_id="pod-1", conn_id="conn-1").kv_buckets
    assert f"{_NAMESPACE}-{suffix}" in granted, (
        f"a tool pod is not granted the bucket KVLease actually opens "
        f"('{_NAMESPACE}-{suffix}'); it holds {list(granted)}. Without it claim_session is handed "
        f"lease=None, which does not fail -- it serves the display UNCLAIMED, so two pods can "
        f"drive one display."
    )
