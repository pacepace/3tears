"""the JetStream KV buckets the identity service opens, declared once for two readers.

**Why this is here and not in the identity service.** Two repositories have to agree on this
set and they cannot import each other: ``14-eng-ai-bot-identity`` OPENS these buckets, and
``14-eng-ai-bot``'s static-NATS-grant generator GRANTS them, because identity connects as a
static user whose permissions that generator renders. This package is already the shared
home for identity contracts both services parse (``threetears.iam.tokens``,
``threetears.iam.dpop``), so it is where a set both must read belongs.

**What it is FOR is a grant, and the failure it prevents is silent.** An ungranted KV call
does not raise. It blocks to its deadline and reports an unreachable broker, so a name
missing from this tuple surfaces as a ten-second hang inside a login -- diagnosed as a
network fault, not a permission one. That is why the identity service carries
``test_identity_kv_buckets_are_declared``, which walks its own source for every bucket
opener and fails if one names a bucket absent from here. The declaration cannot silently
fall behind the code that opens them.

**What is NOT here, and the distinction matters.** A ``purpose`` is not a bucket. The
throttles, lockouts, spray counters and the ``RedemptionLedger`` are
:class:`~threetears.core.collections.BaseCollection`-backed: their rows live in the ONE
shared ``{ns}-collections`` bucket under the writer's key scope, and are covered by that
bucket's scoped grant rather than by a bucket of their own. Only a name reaching
``kv_bucket``, ``ticket_store``, ``ReplayGuard``, ``RevocationGuard`` or ``AuthCodeStore``
is a bucket.

**The names are unprefixed, as the openers write them.** ``NatsClient.kv_bucket`` layers
``{namespace}-`` on at open time, so the live stream for ``identity-dpop-nonces`` is
``KV_aibots-identity-dpop-nonces`` on a deployment whose namespace is ``aibots``. A
consumer composing a grant applies its own namespace; nothing here assumes one.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "IDENTITY_JS_STREAMS",
    "IDENTITY_KV_BUCKETS",
    "IDENTITY_KV_KEY_SCOPES",
]

#: plain JetStream streams identity DECLARES, unprefixed, sorted.
#:
#: A stream is not a KV bucket and needs its own entry: it owns no ``$KV.<name>.>`` data subtree,
#: and declaring one needs ``STREAM.CREATE``/``STREAM.UPDATE``, which a read grant does not carry.
#:
#: ``audit`` is DECLARED BY TWO SERVICES, deliberately. identity-core and the hub both call
#: ``ensure_jetstream_stream(name="audit", ..., storage="file")`` -- the storage matches on both
#: sides on purpose, because whichever started first won the stream's storage type and the second
#: crashed outright on ``stream configuration update can not change storage type``. Identity
#: therefore needs the DECLARING capability, not merely publish: it may genuinely be the process
#: that creates the stream.
IDENTITY_JS_STREAMS: Final[tuple[str, ...]] = ("audit",)

#: every KV bucket identity-core and identity-edge open, unprefixed, sorted.
#:
#: Eleven replay-guard names that stood here are GONE, and both reasons are the "a purpose is
#: not a bucket" distinction above, arrived at from two directions. Five became a ``purpose``
#: row-key prefix in the shared collections bucket -- the spray counter, both revocation
#: ledgers, and the SAML-assertion and OAuth-client-assertion guards, the latter two now
#: watermarked against a signed issue time rather than remembering nonces at all. Six were
#: deleted outright: the artifact each guarded is a server-side record identity itself wrote,
#: so it is consumed by a revision-guarded delete of that record. A separate nonce bucket is a
#: second stream that can sit on another node, and wiping it while the guarded record survives
#: reopens the very replay window it exists to close.
#:
#: They are removed rather than left behind because an unused grant is not free. It is a bucket
#: this service may open and does not, which an operator reading the rendered conf reads as a
#: live dependency.
#:
#: ``oauth_client_assertions`` was the one name breaking the ``identity-`` convention, and it
#: left as a purpose keeping that spelling: a rename strands the keys in the old bucket, and a
#: replay guard that forgets what it has seen is a replay window rather than a cosmetic issue.
IDENTITY_KV_BUCKETS: Final[tuple[str, ...]] = (
    "identity-apikey-verify-cache",
    "identity-core-healthcheck",
    "identity-devx-bootstrap",
    "identity-dpop-nonces",
    "identity-email-change-tokens",
    "identity-flow-origin",
    "identity-github-state",
    "identity-oauth-auth-codes",
    "identity-oidc-state",
    "identity-passkey-challenges",
    "identity-recovery-tokens",
    "identity-refresh-replay-grace",
    "identity-saml-pending",
    "identity-totp-partial-auth",
)

#: the L2 key scopes identity's collections write under in the SHARED ``{ns}-collections``
#: bucket -- one per service, because a scope is the sharing boundary and identity-core and
#: identity-edge are separate deployments that must not read each other's rows.
#:
#: Both come from :func:`threetears.nats.subject_permissions.kv_key_scope_for_service` at the
#: two construction sites (``identity_core/server.py`` and ``identity_edge/app.py``), which
#: renders ``svc-{service}``. They are spelled out rather than recomposed here so a reader of
#: a rendered grant can match the subject to this tuple without running anything; the
#: identity-side test pins them against the live calls.
IDENTITY_KV_KEY_SCOPES: Final[tuple[str, ...]] = (
    "svc-identity-core",
    "svc-identity-edge",
)
