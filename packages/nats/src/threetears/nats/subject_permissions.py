"""Per-principal NATS subject-permission allow-lists for decentralized auth (platform-auth A).

When the NATS bus moves from an anonymous shared connection to authenticated per-principal
connections, the auth-callout responder mints each connecting principal a user JWT whose pub/sub
permissions come from THIS map. The map is the single authority for "what may a {principal} pub or
sub" and exists to make two properties true and testable:

- **exhaustive** — every subject a principal needs to bootstrap and run is present; a missing
  boot-critical subject bricks that principal the moment auth is enforced (it can't even handshake).
- **least-privilege isolation** — no principal is granted the bare ``>`` or the global ``_INBOX.>``;
  isolation is per-user allow-lists. Identity-bound subjects (``agents.internal.{agent}.{pod}``,
  ``tools.internal.{pod}``, heartbeats, the principal's reply inbox) are templated with the
  CONNECTING principal's OWN ids, so principal A cannot subscribe to principal B's inbox or
  impersonate B's identity-tailed subjects.

Request/reply uses a per-connection SCOPED inbox (``_INBOX_{principal}_{id}``) plus NATS
``allow_responses``: a responder may publish to the reply-subject of a message it actually received
without holding standing publish rights on every requester's inbox. So a responder never needs a
broad ``_INBOX.>`` publish grant.

The application pub/sub subjects below are built through the canonical :class:`Subjects` factory (the
namespace prefix is never hand-typed). Each principal additionally DECLARES the JetStream resources it
touches as :class:`JsResource` records -- a name, a kind (KV bucket or stream), the
:class:`JsCapability` it needs on that resource, an optional key SCOPE, and its write intent. The
minted user JWT turns each record into a ``$KV`` publish grant plus a JetStream control-plane
allow-list PINNED per declared stream (a KV bucket ``<b>`` is backed by the stream ``KV_<b>``), so a
principal can drive JS ops only against its OWN streams and is denied the cross-principal direct-read
/ destroy a bare ``$JS.API.>`` would expose (see
:func:`threetears.nats.user_jwt.js_api_grants_for_stream`).

**Pinning the stream name is not enough on a SHARED stream.** ``{ns}-collections`` is one bucket held
by many principals, so the per-stream pin admits every principal's keys. A resource that carries a
``scope`` therefore mints a NARROWED pair -- ``$KV.{b}.{scope}.>`` on publish and
``$JS.API.DIRECT.GET.KV_{b}.$KV.{b}.{scope}.>`` on read -- and drops every JetStream verb that can
carry a key in a request BODY (``STREAM.MSG.GET``, the bare ``DIRECT.GET``, consumer create with a
``filter_subject``) or export the whole stream (``SNAPSHOT`` / ``RESTORE`` / ``PURGE`` / ``UPDATE``).
Scoping is per-resource OPT-IN: no other bucket writes a scope prefix, so narrowing them all would
deny every read on all of them -- and a denied JetStream request is never answered, so it arrives as
a ten-second deadline rather than as an error.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final
from uuid import UUID

from threetears.nats.result_delivery import result_stream_name
from threetears.nats.subjects import Subjects, get_default_namespace, sanitize_subject_segment

__all__ = [
    "CROSS_PLATFORM_CACHE_INVALIDATE",
    "COORDINATION_BUCKET_SUFFIX_GRAMMAR",
    "KV_KEY_SCOPE_GRAMMAR",
    "MAX_COORDINATION_BUCKETS",
    "MAX_COORDINATION_BUCKET_SUFFIX_CHARS",
    "JsCapability",
    "JsResource",
    "JsResourceKind",
    "Principal",
    "PrincipalPermissions",
    "build_permissions",
    "capability_declares",
    "capability_is_scoped",
    "inbox_prefix_for",
    "kv_bucket_names",
    "kv_key_scope_for",
]

#: the ONE deliberately non-namespaced subject (every 3tears collection in every process listens on
#: it regardless of env prefix). mirrors :meth:`Subjects.cache_invalidate`.
CROSS_PLATFORM_CACHE_INVALIDATE = "threetears.cache.invalidate"

#: grammar every L2 key-scope segment must satisfy, matched as a whole string.
#:
#: DELIBERATELY STRICTER than the JetStream KV key grammar
#: (``^[-/_=.a-zA-Z0-9]+$``, ``threetears.core.collections.base._KV_KEY_GRAMMAR``), which
#: carries ``.`` and ``/`` INSIDE its character class. a scope is the leading NATS subject
#: TOKEN of ``$KV.{bucket}.{scope}.{table}.{body}``: a dot in it silently produces two tokens,
#: so a per-principal grant of ``$KV.{bucket}.{scope}.>`` would stop matching the keys the
#: principal writes -- a check that accepts a dot validates nothing about the property the
#: scope exists to provide. reusing the key grammar here was the mistake this constant exists
#: to make impossible.
KV_KEY_SCOPE_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"^[-_=a-zA-Z0-9]+$")

#: what one AGENT_POD-declared coordination bucket SUFFIX may contain.
#:
#: A KV bucket ``<b>`` is backed by the stream ``KV_<b>``, and a stream name is ONE
#: subject token, so a dot in the suffix would split the token and silently point every
#: emitted grant at a subject shape nothing matches. The remaining characters are what
#: nats-server accepts in a bucket name.
COORDINATION_BUCKET_SUFFIX_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"^[-_a-zA-Z0-9]+$")

#: how many coordination buckets ONE agent may declare.
#:
#: The prefix (see :func:`_coordination_resources`) is what stops a declaration reaching
#: another principal's buckets; it is not what stops a declaration being expensive. Each
#: entry materialises one JetStream stream on the platform's bus, so the count is bounded
#: as well. Generous relative to what any product needs -- the survey engine, which
#: motivated this, declares six -- and low enough that a runaway manifest is refused
#: rather than provisioned.
MAX_COORDINATION_BUCKETS: Final[int] = 16

#: longest coordination bucket suffix accepted. the composed name is
#: ``{ns}-{scope}-{suffix}`` and its stream is ``KV_`` plus that; the scope alone is 42
#: characters, so this keeps the whole stream name well inside what nats-server accepts
#: while leaving room for a name that says what the bucket is for.
MAX_COORDINATION_BUCKET_SUFFIX_CHARS: Final[int] = 64


class Principal(StrEnum):
    """a connection identity class the bus authenticates and scopes permissions for.

    Every member is REFERENCED: :data:`_RESOLVERS` carries one resolver per member, and
    :func:`kv_key_scope_for` answers a scope for each. A member with neither is a scope value that
    cannot legally be produced, which is what blocked two downstream shards from wiring a principal
    at all -- ``AGENT_ROUTER`` and ``DATASET_EXECUTOR`` are here for exactly that reason.

    Adopting a member is a GRANT-SURFACE change, not a migration. ``REGISTRY``, ``HUB``, ``GATEWAY``,
    ``CHANNEL_ADAPTER``, ``AGENT_ROUTER`` and ``DATASET_EXECUTOR`` all connect as STATIC NATS users
    today and never reach the auth callout, so their resolvers describe what the static conf should
    say rather than what is currently minted. The static half is its own piece of work.
    """

    AGENT_POD = "agent_pod"
    TOOL_POD = "tool_pod"
    REGISTRY = "registry"
    HUB = "hub"
    GATEWAY = "gateway"
    CHANNEL_ADAPTER = "channel_adapter"
    AGENT_ROUTER = "agent_router"
    DATASET_EXECUTOR = "dataset_executor"


class JsResourceKind(StrEnum):
    """what kind of JetStream resource one declaration names.

    A KV bucket ``<b>`` is backed by the stream ``KV_<b>`` and additionally owns the ``$KV.<b>.>``
    data subtree; a plain stream owns neither. The mint needs to tell them apart, so the record says
    which it is rather than guessing from the name.
    """

    KV_BUCKET = "kv_bucket"
    STREAM = "stream"


class JsCapability(StrEnum):
    """what one principal may do against one JetStream stream.

    Deliberately three values, not five, and the omission is recorded rather than accidental. A
    finer split (read-only KV / KV watch / durable consumer) would be better least-privilege, but
    every non-scoped bucket on this platform runs ``allow_direct: false``, where nats-py reads a key
    by publishing ``$JS.API.STREAM.MSG.GET`` with the key in the request BODY -- so trimming verbs
    from :attr:`FULL` on evidence nobody has gathered would deny reads that currently work, and a
    denied JetStream request arrives as a ten-second deadline rather than as an error. :attr:`FULL`
    is therefore the historical set, unchanged, and the narrowing is confined to the resources that
    actually write a key scope.

    :cvar FULL: every JS op nats-py issues against this stream. the historical grant.
    :cvar KV_SCOPED: bind (``STREAM.INFO``) plus a direct read of the principal's OWN key scope, and
        nothing else -- no body-carried read, no consumer, no destroy, no whole-stream export.
    :cvar KV_SCOPED_DECLARE: :attr:`KV_SCOPED` plus ``STREAM.CREATE`` and ``STREAM.UPDATE``. the
        DECLARING identity alone; never a pod. ``UPDATE`` is a read-all primitive on a shared stream
        (``republish`` / ``sources`` mirror every key to a subject the caller names), which is why
        this is a distinct capability rather than a softening of :attr:`KV_SCOPED`.
    """

    FULL = "full"
    KV_SCOPED = "kv_scoped"
    KV_SCOPED_DECLARE = "kv_scoped_declare"


#: capabilities whose grants are narrowed to one key scope, and therefore REQUIRE a scope.
_SCOPED_CAPABILITIES: Final[frozenset[JsCapability]] = frozenset(
    {JsCapability.KV_SCOPED, JsCapability.KV_SCOPED_DECLARE}
)

#: capabilities that may create or update the resource. never granted to a pod principal.
_DECLARING_CAPABILITIES: Final[frozenset[JsCapability]] = frozenset({JsCapability.KV_SCOPED_DECLARE})


def capability_is_scoped(capability: JsCapability) -> bool:
    """whether ``capability`` narrows its grants to one key scope.

    :param capability: the capability under test
    :ptype capability: JsCapability
    :return: whether a scope is required and the emitted grants carry it
    :rtype: bool
    """
    return capability in _SCOPED_CAPABILITIES


def capability_declares(capability: JsCapability) -> bool:
    """whether ``capability`` may CREATE or UPDATE the resource.

    ``declare`` is deliberately distinct from "admin": it is the narrow authority the identity that
    owns a shared bucket's canonical configuration needs, and no pod principal may hold it.

    :param capability: the capability under test
    :ptype capability: JsCapability
    :return: whether the capability carries ``STREAM.CREATE`` and ``STREAM.UPDATE``
    :rtype: bool
    """
    return capability in _DECLARING_CAPABILITIES


@dataclass(frozen=True, slots=True)
class JsResource:
    """one JetStream resource a principal declares, with the capability it needs on it.

    ONE record for both kinds. KV buckets and streams were previously two flat tuples of names,
    which made the capability question unaskable of either: a name alone cannot say "read my own
    scope and nothing else". The scope and the write intent live here for the same reason -- a
    bucket added to a resolver without a decision on both is the shape that reopens the hole.

    :param name: the bucket name (prefix included) or the stream name, verbatim
    :ptype name: str
    :param kind: whether ``name`` is a KV bucket or a plain stream
    :ptype kind: JsResourceKind
    :param capability: what this principal may do against the backing stream
    :ptype capability: JsCapability
    :param scope: the L2 key scope this principal writes under, from :func:`kv_key_scope_for`;
        ``None`` for a resource whose keys carry no scope prefix
    :ptype scope: str | None
    :param writable: whether the principal may publish into the bucket's ``$KV.`` data subtree.
        ``False`` yields a read-only grant -- a KV read is a ``$JS.API`` request, never a ``$KV.``
        publish, so the two are genuinely separable
    :ptype writable: bool
    """

    name: str
    kind: JsResourceKind
    capability: JsCapability
    scope: str | None
    writable: bool

    def __post_init__(self) -> None:
        """refuse a record whose scope and capability disagree.

        :return: nothing
        :rtype: None
        :raises ValueError: if a scoped capability carries no scope, if an unscoped capability
            carries one, if the scope is not a single subject token, if a stream is given a KV
            capability, or if a plain stream is declared writable
        """
        if capability_is_scoped(self.capability):
            if self.kind is not JsResourceKind.KV_BUCKET:
                raise ValueError(f"{self.capability.value} applies to a KV bucket, not to stream {self.name!r}")
            if self.scope is None:
                raise ValueError(
                    f"KV bucket {self.name!r} is declared with {self.capability.value} but this principal "
                    f"has no key scope; a scoped grant with no scope would either widen to the whole "
                    f"bucket or match nothing, and a grant that matches nothing fails as a deadline"
                )
            if not KV_KEY_SCOPE_GRAMMAR.match(self.scope):
                raise ValueError(
                    f"kv key scope {self.scope!r} does not match {KV_KEY_SCOPE_GRAMMAR.pattern}; a scope "
                    f"is ONE subject token, so a dot in it silently produces two and the grant stops "
                    f"matching the keys the principal writes"
                )
        elif self.scope is not None:
            raise ValueError(
                f"resource {self.name!r} carries scope {self.scope!r} with capability "
                f"{self.capability.value}, which emits an UNSCOPED grant; a scope that is recorded but "
                f"not enforced reads as isolation that is not there"
            )
        if self.kind is JsResourceKind.STREAM and self.writable:
            raise ValueError(
                f"stream {self.name!r} is not a KV bucket and owns no $KV. data subtree, so write "
                f"intent is not expressible for it"
            )

    @property
    def stream_name(self) -> str:
        """the JetStream stream backing this resource.

        :return: ``KV_{name}`` for a bucket, ``name`` for a stream
        :rtype: str
        """
        if self.kind is JsResourceKind.KV_BUCKET:
            return f"KV_{self.name}"
        return self.name

    @classmethod
    def kv(cls, name: str, *, scope: str | None, writable: bool, declare: bool = False) -> JsResource:
        """declare one KV bucket.

        ``scope`` and ``writable`` are keyword-only and have NO defaults on purpose: adding a bucket
        to a resolver must be a decision about both, and a default would let the next bucket inherit
        somebody else's answer.

        :param name: fully-qualified bucket name, prefix included
        :ptype name: str
        :param scope: the principal's L2 key scope, or ``None`` for a bucket whose keys carry none
        :ptype scope: str | None
        :param writable: whether the principal writes into the bucket
        :ptype writable: bool
        :param declare: whether this principal owns the bucket's canonical configuration
        :ptype declare: bool
        :return: the resource record
        :rtype: JsResource
        :raises ValueError: if ``declare`` is asked for without a scope, or the record is otherwise
            inconsistent (see :meth:`__post_init__`)
        """
        if scope is None:
            capability = JsCapability.FULL
        elif declare:
            capability = JsCapability.KV_SCOPED_DECLARE
        else:
            capability = JsCapability.KV_SCOPED
        if declare and scope is None:
            raise ValueError(
                f"KV bucket {name!r} asks to be declared without a scope; the declaring capability "
                f"exists for the SHARED scoped bucket, and granting CREATE/UPDATE on an unscoped one "
                f"buys nothing the FULL capability does not already carry"
            )
        return cls(name=name, kind=JsResourceKind.KV_BUCKET, capability=capability, scope=scope, writable=writable)

    @classmethod
    def stream(cls, name: str) -> JsResource:
        """declare one plain JetStream stream.

        Streams stay at :attr:`JsCapability.FULL`: every declared stream here is consumed with a
        durable or ephemeral consumer, and the registry additionally DECLARES the result stream at
        startup through ``STREAM.CREATE``. Narrowing them is a separate piece of work with its own
        evidence, not a side effect of scoping one KV bucket.

        :param name: the stream name, verbatim
        :ptype name: str
        :return: the resource record
        :rtype: JsResource
        """
        return cls(name=name, kind=JsResourceKind.STREAM, capability=JsCapability.FULL, scope=None, writable=False)


@dataclass(frozen=True, slots=True)
class PrincipalPermissions:
    """the resolved pub/sub allow-list + JetStream resources for one connection.

    :param publish: subjects (concrete or wildcard patterns) the principal may publish to. for
        request/reply this is the request subject (a request is a publish + an inbox subscribe).
    :ptype publish: tuple[str, ...]
    :param subscribe: subjects the principal may subscribe to, including its own scoped reply inbox.
    :ptype subscribe: tuple[str, ...]
    :param allow_responses: when true the principal may reply to the reply-subject of any message it
        received, without a standing publish grant on the requester's inbox (NATS ``allow_responses``).
        set for every principal that answers requests.
    :ptype allow_responses: bool
    :param inbox_prefix: the scoped request/reply inbox prefix this principal's client must use
        (never the global ``_INBOX``); ``{inbox_prefix}.>`` is included in ``subscribe``.
    :ptype inbox_prefix: str
    :param js_resources: every JetStream KV bucket and stream this principal touches, each with the
        capability it needs on it. ONE tuple for both kinds -- see :class:`JsResource`.
    :ptype js_resources: tuple[JsResource, ...]
    """

    publish: tuple[str, ...]
    subscribe: tuple[str, ...]
    allow_responses: bool
    inbox_prefix: str
    js_resources: tuple[JsResource, ...] = field(default_factory=tuple)


def kv_bucket_names(permissions: PrincipalPermissions) -> tuple[str, ...]:
    """the KV bucket names one principal declares, in declaration order.

    A derivation, not a second source of truth: the buckets ARE
    :attr:`PrincipalPermissions.js_resources` filtered by kind. Provided because the static-user
    configuration in the hub and several grant-naming checks need the plain name list and would
    otherwise each re-roll the same comprehension.

    :param permissions: the resolved permissions
    :ptype permissions: PrincipalPermissions
    :return: bucket names, prefix included
    :rtype: tuple[str, ...]
    """
    return tuple(r.name for r in permissions.js_resources if r.kind is JsResourceKind.KV_BUCKET)


def inbox_prefix_for(principal: Principal, *, conn_id: str) -> str:
    """the scoped request/reply inbox prefix for one connection.

    Each connection gets ``_INBOX_{principal}_{conn_id}`` instead of the shared global ``_INBOX``,
    so a responder replying via :attr:`PrincipalPermissions.allow_responses` cannot be sniffed by a
    different principal subscribing the global inbox tree. the principal's NATS client must be
    configured with this prefix.

    :param principal: the connection identity class
    :ptype principal: Principal
    :param conn_id: a value unique to this connection (the pod_id for pods; a connection uuid for
        infra principals)
    :ptype conn_id: str
    :return: the scoped inbox prefix
    :rtype: str
    """
    return f"_INBOX_{principal.value}_{_seg(conn_id)}"


def _seg(value: str) -> str:
    """subject-safe a single dynamic segment (mirror :func:`Subjects` sanitization: ``.`` -> ``-``)."""
    return sanitize_subject_segment(value)


#: the set a POD-DERIVED scope may never land on (L2S-08). infra scopes ARE these values, which
#: is why the check lives here and not in ``CollectionRegistry.configure`` -- that call site sees
#: a bare string and cannot tell an infra scope from a pod one, so the same rule applied there
#: would refuse every infra principal at startup.
_BARE_PRINCIPAL_SCOPES: Final[frozenset[str]] = frozenset(p.value for p in Principal)


def _scope_id(value: str | UUID, *, name: str, principal: Principal) -> str:
    """render one pod principal's identifying id into its scope token.

    the token is the id's 32-character UUID hex. a uuid hex is injective (two distinct
    principals can never render one token) and is already inside
    :data:`KV_KEY_SCOPE_GRAMMAR`, which is what makes it usable as an isolation boundary.
    :func:`threetears.nats.subjects.sanitize_subject_segment` is deliberately NOT used: it maps
    ``.`` to ``-`` and is therefore non-injective, so two distinct display names collapse onto
    one token -- two pods sharing one scope is the exact outcome key scoping exists to prevent.

    :param value: the principal's identifying id
    :ptype value: str | UUID
    :param name: parameter name to quote in the failure message
    :ptype name: str
    :param principal: the connection identity class, quoted in the failure message
    :ptype principal: Principal
    :return: 32-character lowercase hex token
    :rtype: str
    :raises ValueError: if value is not a uuid
    """
    if isinstance(value, UUID):
        return value.hex
    try:
        return UUID(str(value)).hex
    except ValueError as exc:
        raise ValueError(
            f"{principal.value} kv key scope requires a uuid {name}, got {value!r}; "
            f"a scope derived from anything non-uuid is not provably collision-free"
        ) from exc


def kv_key_scope_for(
    principal: Principal,
    *,
    agent_id: str | UUID | None = None,
    pod_id: str | UUID | None = None,
) -> str:
    """the L2 key scope one principal's collections write under.

    every key in the shared ``{ns}-collections`` bucket is
    ``{scope}.{table}.{body}``; this function is the ONLY producer of that leading
    segment. it sits beside :func:`inbox_prefix_for` for the same reason that one does: the
    minting side and the connecting side must derive the identical value from the identical
    inputs, and a client keyed on anything else reads and writes keys its own grant does not
    cover.

    **the scope is the sharing boundary, not the connection.** replicas of one principal
    resolve to one scope, or L2 stops being a cross-pod cache; so a per-connection value
    (``conn_id``) is never an input -- every reconnect would orphan the cache.

    per principal:

    - :attr:`Principal.AGENT_POD` -- ``agent_id``, which the identity JWT authenticates as its
      ``sub``. ``pod_id`` is REFUSED here rather than accepted as an alternative: for this
      principal the auth callout derives the pod id from the connect NAME, which is
      attacker-influenced, so a scope built from it is a scope the connecting side chooses.
    - :attr:`Principal.TOOL_POD` -- ``pod_id``, the ``tool_pods.id`` primary key. safe where the
      agent's is not, because ``_resolve_tool_pod`` ignores the connect name and pins
      ``claims.sub`` from the verified key id; configured once per deployment, so replicas share.
    - every infra principal -- the bare :class:`Principal` value. one identity per service, and
      no per-connection id that replicas would share.

    a POD principal with no identifying id RAISES. falling back to the bare enum value would
    land every tool pod that lost its id on one shared scope, which is the failure this
    function exists to make impossible.

    :param principal: the connection identity class
    :ptype principal: Principal
    :param agent_id: the connecting agent's id; required for :attr:`Principal.AGENT_POD` and
        rejected for every other principal
    :ptype agent_id: str | UUID | None
    :param pod_id: the connecting tool pod's ``tool_pods.id``; required for
        :attr:`Principal.TOOL_POD` and rejected for every other principal
    :ptype pod_id: str | UUID | None
    :return: the scope segment, matching :data:`KV_KEY_SCOPE_GRAMMAR`
    :rtype: str
    :raises ValueError: if a pod principal is missing its identifying id, if an id is supplied
        for a principal that does not take it, if an id is not a uuid, or if the rendered scope
        would collide with a bare :class:`Principal` value
    """
    if principal is Principal.AGENT_POD:
        if pod_id is not None:
            raise ValueError(
                "agent_pod kv key scope must be derived from agent_id, never pod_id: the auth "
                "callout derives an agent pod's pod id from the spoofable connect name"
            )
        if agent_id is None:
            raise ValueError("agent_pod kv key scope requires a non-empty agent_id")
        return _pod_scope(principal, _scope_id(agent_id, name="agent_id", principal=principal))
    if principal is Principal.TOOL_POD:
        if agent_id is not None:
            raise ValueError("tool_pod kv key scope must be derived from pod_id, not agent_id")
        if pod_id is None:
            raise ValueError("tool_pod kv key scope requires a non-empty pod_id")
        return _pod_scope(principal, _scope_id(pod_id, name="pod_id", principal=principal))
    if agent_id is not None or pod_id is not None:
        raise ValueError(
            f"{principal.value} kv key scope takes no id: an infra principal has one identity "
            f"per service and its scope IS its principal value"
        )
    return principal.value


def _pod_scope(principal: Principal, scope_id: str) -> str:
    """compose and validate one pod principal's scope segment.

    :param principal: the connection identity class (leads the segment, so an operator reading
        a grant can tell which principal a subject belongs to)
    :ptype principal: Principal
    :param scope_id: the principal's uuid-hex identity token
    :ptype scope_id: str
    :return: the scope segment
    :rtype: str
    :raises ValueError: if the segment falls outside :data:`KV_KEY_SCOPE_GRAMMAR` or collides
        with a bare :class:`Principal` value
    """
    scope = f"{principal.value}-{scope_id}"
    if scope in _BARE_PRINCIPAL_SCOPES:
        raise ValueError(
            f"pod-derived kv key scope {scope!r} collides with a bare principal value; "
            f"an infra principal would then share a scope with a pod"
        )
    if not KV_KEY_SCOPE_GRAMMAR.match(scope):
        raise ValueError(f"kv key scope {scope!r} does not match {KV_KEY_SCOPE_GRAMMAR.pattern}")
    return scope


def _ns() -> str:
    """current namespace prefix (the Subjects factory reads the same source)."""
    return get_default_namespace()


def _deadletter(ns: str) -> str:
    """the deadletter subtree EVERY subscribing principal must be able to publish.

    ``NatsClient.subscribe`` and ``subscribe_typed`` both default ``deadletter_on_failure=True``:
    a callback that raises -- and, on the typed path, a payload that fails validation -- is
    republished to ``{ns}.deadletter.{original_subject}``. The original subject carries its own
    dots, so this is a ``>`` subtree rather than a single wildcard token.

    Granted to every principal because every principal subscribes to something. No principal held
    it before: the registry was incidentally covered by its static ``aibots.>``, and a
    callout-minted agent pod was not, so its deadletter publish was refused and the diagnostic it
    exists to preserve was dropped at WARNING inside the very handler that was already failing.

    Hand-typed rather than built through :class:`Subjects`, which mints a CONCRETE deadletter
    subject for one original path and has no wildcard constructor; the surrounding resolvers
    hand-type their wildcard families for the same reason.

    :param ns: the subject namespace prefix
    :ptype ns: str
    :return: the deadletter publish grant
    :rtype: str
    """
    return f"{ns}.deadletter.>"


def build_permissions(
    principal: Principal,
    *,
    agent_id: str | None = None,
    pod_id: str | None = None,
    conn_id: str | None = None,
    tool_namespaces: Sequence[str] | None = None,
    coordination_buckets: Sequence[str] | None = None,
) -> PrincipalPermissions:
    """resolve the concrete allow-list for one connecting principal.

    Pod principals (:attr:`Principal.AGENT_POD`, :attr:`Principal.TOOL_POD`) require their own
    ``pod_id`` (and ``agent_id`` for the agent) so identity-bound subjects are scoped to THEM. infra
    principals use wildcards for the dynamic peer segments (any pod / any correlation id). ``conn_id``
    defaults to ``pod_id`` for pods and must be supplied for infra principals.

    :param principal: the connection identity class
    :ptype principal: Principal
    :param agent_id: the connecting agent's id (required for :attr:`Principal.AGENT_POD`)
    :ptype agent_id: str | None
    :param pod_id: the connecting pod's id (required for pod principals)
    :ptype pod_id: str | None
    :param conn_id: a connection-unique id for the scoped inbox; defaults to ``pod_id``
    :ptype conn_id: str | None
    :param tool_namespaces: the tool namespace names this pod is authorized to serve -- the
        ``allowed_namespaces`` list on its tool-pods row, the same list
        ``RegistrationHandler._authenticate_and_filter`` filters its manifest against. each entry
        becomes one EXACT per-tool subject grant, so registration filtering and subject permissions
        read from one source of truth. omitting it grants none of them (a pod that serves no
        human-in-the-loop session needs none).
    :ptype tool_namespaces: Sequence[str] | None
    :param coordination_buckets: KV bucket SUFFIXES this agent declares for its own
        coordination primitives -- the ``coordination_buckets`` column on its ``agents``
        row, resolved by the auth callout exactly as ``tool_namespaces`` is. Each becomes
        one grant composed under the agent's OWN key scope, so a declared string can never
        name a bucket outside that agent's space; see :func:`_coordination_resources`.
        Read by :attr:`Principal.AGENT_POD` alone and ignored for every other principal.
        Omitting it grants none of them, which is what every pre-existing caller does and
        therefore what every pre-existing caller still gets.
    :ptype coordination_buckets: Sequence[str] | None
    :return: the resolved permissions
    :rtype: PrincipalPermissions
    :raises ValueError: when a required id for the principal is missing, or when a declared
        coordination bucket suffix is malformed or the declaration exceeds
        :data:`MAX_COORDINATION_BUCKETS`
    """
    resolver = _RESOLVERS[principal]
    return resolver(
        agent_id=agent_id,
        pod_id=pod_id,
        conn_id=conn_id,
        tool_namespaces=tool_namespaces,
        coordination_buckets=coordination_buckets,
    )


def _require(value: str | None, *, name: str, principal: Principal) -> str:
    """fail closed when a permission resolver is missing an id it must scope on."""
    if not value:
        raise ValueError(f"{principal.value} permissions require a non-empty {name}")
    return value


def _coordination_resources(
    ns: str,
    scope: str,
    coordination_buckets: Sequence[str] | None,
) -> tuple[JsResource, ...]:
    """the KV grants for one agent's OWN declared coordination buckets.

    **The declared string is a SUFFIX and never a bucket name, and that is the whole
    security property.** Every grant is composed as ``{ns}-{scope}-{suffix}``, where
    ``scope`` is :func:`kv_key_scope_for` on the AUTHENTICATED agent id -- an id the auth
    callout pins from the verified token's ``sub``, never from the spoofable connect name.
    The prefix is therefore not an input, so no declaration can name a bucket outside the
    declaring agent's own space: an agent that asks for ``collections`` is granted its own
    ``{ns}-agent_pod-<hex>-collections`` and comes no closer to the shared bucket every
    principal writes its L2 into. That makes the declaration itself non-sensitive, which is
    what lets it come from an agent's own manifest at all.

    **Unscoped WITHIN the bucket, deliberately.** The bucket is the isolation boundary, so
    its keys need no scope segment -- and must not have one, because 3tears's own
    coordination primitives (:class:`~threetears.core.coordination.ReplayGuard`,
    :class:`~threetears.core.coordination.DistributedCounter`, the idempotency and ticket
    stores) each hash their own keys with no scope prefix. A scoped grant here would emit
    ``$KV.{bucket}.{scope}.>``, match none of their keys, and deny every call as a
    ten-second deadline that reads as an unreachable broker.

    **Never a declaring capability.** ``STREAM.UPDATE`` is a read-all primitive on any
    stream it is held against (``republish`` / ``sources`` mirror every key onto a subject
    the holder names), so a pod gets :attr:`JsCapability.FULL` on a bucket of its own and
    never :attr:`JsCapability.KV_SCOPED_DECLARE`.

    :param ns: the subject namespace prefix
    :ptype ns: str
    :param scope: this agent's own key scope, from :func:`kv_key_scope_for`
    :ptype scope: str
    :param coordination_buckets: the bucket suffixes this agent declares, or ``None``
    :ptype coordination_buckets: Sequence[str] | None
    :return: one writable KV resource per declared suffix, in declaration order
    :rtype: tuple[JsResource, ...]
    :raises ValueError: if more than :data:`MAX_COORDINATION_BUCKETS` are declared, or if
        any suffix is empty, over-long, or outside
        :data:`COORDINATION_BUCKET_SUFFIX_GRAMMAR`
    """
    declared = tuple(coordination_buckets or ())
    if len(declared) > MAX_COORDINATION_BUCKETS:
        raise ValueError(
            f"an agent declared {len(declared)} coordination buckets, over the "
            f"{MAX_COORDINATION_BUCKETS} cap; each one materialises a JetStream stream"
        )
    for suffix in declared:
        # REFUSE rather than skip. a dropped entry leaves the pod opening a bucket it was
        # never granted, and an ungranted KV call does not raise -- it blocks to its
        # deadline and reports an unreachable broker, which is the failure this file's own
        # comments repeatedly describe as the kind that costs a day.
        if not suffix or len(suffix) > MAX_COORDINATION_BUCKET_SUFFIX_CHARS:
            raise ValueError(
                f"coordination bucket suffix {suffix!r} must be 1 to {MAX_COORDINATION_BUCKET_SUFFIX_CHARS} characters"
            )
        if not COORDINATION_BUCKET_SUFFIX_GRAMMAR.match(suffix):
            raise ValueError(
                f"coordination bucket suffix {suffix!r} does not match "
                f"{COORDINATION_BUCKET_SUFFIX_GRAMMAR.pattern}; a KV bucket is backed by the "
                f"stream KV_<bucket>, and a stream name is one subject token"
            )
    return tuple(JsResource.kv(f"{ns}-{scope}-{suffix}", scope=None, writable=True) for suffix in declared)


def _agent_pod(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
    coordination_buckets: Sequence[str] | None,
) -> PrincipalPermissions:
    a = _require(agent_id, name="agent_id", principal=Principal.AGENT_POD)
    p = _require(pod_id, name="pod_id", principal=Principal.AGENT_POD)
    inbox = inbox_prefix_for(Principal.AGENT_POD, conn_id=conn_id or p)
    # derived from the AUTHENTICATED agent id, never the spoofable connect-name pod id --
    # ``kv_key_scope_for`` refuses ``pod_id`` for this principal for exactly that reason.
    scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=a)
    ns = _ns()
    publish = (
        str(Subjects.agent_register()),
        str(Subjects.agent_deregister()),
        str(Subjects.agent_heartbeat(a, p)),  # own authed agent + own pod only
        str(Subjects.hub_handshake()),
        str(Subjects.hub_secrets_request()),
        str(Subjects.hub_object_commit()),  # Path-2: catalogs a produced object under its own tenant
        str(Subjects.hub_object_resolve()),  # Path-2: resolves an object id it owns to its stored key
        # HITL: on a requires_confirmation tool pause the agent records a pending-approval marker
        # with the hub (forwarding its own identity token; the hub verifies + tenant-scopes).
        str(Subjects.hub_approval_record()),
        # engagement selection, READ ONLY: the runtime resolves the conversation channel's default
        # engagement at the tool-call stamp seam. Without this the publish is refused at the
        # connection, the resolve soft-fails to "unbound", and a scan that should have authorized
        # against an explicitly-selected engagement is refused for a missing engagement that was in
        # fact configured. The matching ``.set`` / ``.clear`` write rail is NOT granted: binding a
        # channel to an engagement is an OPERATOR action and rides the hub's authenticated admin
        # HTTP surface, so no responder subscribes to those subjects and an agent has no business
        # writing the binding it reads.
        str(Subjects.hub_channel_engagement_default_resolve()),
        str(Subjects.hub_jwks()),
        str(Subjects.gateway_completion()),
        str(Subjects.gateway_embedding()),
        str(Subjects.tools_discover()),
        str(Subjects.tools_call()),
        # in-process tool serving: an agent runs context-bound tools on ITS OWN ``AGENT_POD``
        # connection rather than as separate Tool Pods -- the devx ``DevInProcessStrategy`` standard
        # builtins, AND (in production) the ``ProdExternalPodsStrategy`` workspace + ``knowledge_drafts``
        # tools. so it must publish the registration + heartbeat subjects a tool pod publishes. the
        # heartbeat is scoped to the AUTHENTICATED ``agent_id`` subtree (``tools.heartbeat.{a}.>``),
        # NOT the spoofable connect-name pod id: the in-process server runs under the
        # ``{agent_id}.{instance}`` composite pod-id (``Subjects.agent_inprocess_pod_id``) so a tenant
        # can never publish a heartbeat under a peer agent's identity. EXTERNAL user-tool pods are a
        # separate ``TOOL_POD`` principal carrying single-token grants under their own ``_tool_pod`` JWT.
        str(Subjects.tools_register()),
        str(Subjects.tools_heartbeat_agent_subtree(a)),  # heartbeats only under its own authed agent subtree
        # STANDING result grant for the tools this agent serves IN-PROCESS. A tool that outlives one
        # connection lifetime cannot answer on the reply inbox at all: ``allow_responses`` belongs to
        # the connection that RECEIVED the call, and re-auth is a reconnect, so the right to answer is
        # destroyed by the very act of staying authenticated. This grant is derived from the
        # AUTHENTICATED ``agent_id``, so every refresh re-mints it identically and a reconnect stops
        # being an event the call has to survive. Scoped to this agent's own subtree for the same
        # reason ``tools.internal.{a}.>`` is: the in-process pod-id is the ``{agent_id}.{instance}``
        # composite, and no agent may deliver a result under a peer's identity.
        str(Subjects.tools_result_agent_subtree(a)),
        # the in-process tool server emits the baseline ``tool.call`` audit envelope on every
        # dispatch (mirrors ``_tool_pod``); audit non-repudiation is required, so the grant is
        # mandatory -- without it an agent-served tool call's actor/audit row is silently dropped.
        str(Subjects.audit_event("tool.call")),
        str(Subjects.knowledge_draft()),
        str(Subjects.workspaces_create()),
        str(Subjects.namespace_discover()),
        str(Subjects.l3_query()),
        str(Subjects.l3_batch()),
        f"{ns}.l3.tx.*",  # mirrors Subjects.l3_tx(op) over all six ops
        f"{ns}.channels.deliver.*",  # publishes finished answers (any inbound channel type)
        f"{ns}.hub.stream.{_seg(a)}.*",  # streams tokens for its own (per-request) correlation ids under its own authed agent id
        f"{ns}.agents.complete.*",  # resilience-task-07: per-turn completion signal the router awaits to ack the durable turn (keyed by correlation id)
        _deadletter(ns),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",  # scoped reply inbox
        str(Subjects.agent_internal(a, p)),  # own agent + own pod only
        str(Subjects.agent_reregister_request(a, p)),  # own authed agent + own pod only
        # in-process tool serving: receive the registry's proxied calls + reachability probe for its
        # OWN in-process tool server. scoped to the AUTHENTICATED ``agent_id`` subtree
        # (``tools.internal.{a}.>`` / ``tools.probe.{a}.>``), NOT the spoofable connect-name pod id:
        # the server subscribes ``tools.{internal,probe}.{a}.{instance}`` under the
        # ``{agent_id}.{instance}`` composite pod-id, so a tenant can NEVER be granted a subject under
        # a peer agent's identity and thus can never wiretap another agent's proxied in-process tool
        # calls. the ``{instance}`` tail keeps replicas of the same agent independently routable.
        str(Subjects.tools_internal_agent_subtree(a)),  # proxied calls to its own in-process tools
        str(Subjects.tools_probe_agent_subtree(a)),  # the registry's reachability probe for its own pods
        str(Subjects.acl_invalidate("membership")),
        str(Subjects.acl_invalidate("assignment")),
        str(Subjects.acl_invalidate("role")),
        CROSS_PLATFORM_CACHE_INVALIDATE,
        f"{ns}.gateway.stream.{_seg(a)}.*",  # token stream for its own in-flight requests under its own authed agent id
        str(Subjects.mcp_rbac_epoch()),
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,  # replies to the hub's route request
        inbox_prefix=inbox,
        js_resources=(
            # the epoch counter: every EPHEMERAL config epoch counts here.
            # a missing grant does NOT raise -- the JS call blocks to its
            # deadline, which reads as an unreachable broker rather than as a
            # permission problem, so this is the kind of omission that costs a
            # day. the epoch client opens the bucket on its first bump.
            JsResource.kv(f"{ns}-epochs", scope=None, writable=True),
            # direct js.create_key_value in the hub; no transport prefix. UNSCOPED, and
            # deliberately so: it is keyed by agent_id with no scope segment, so a scoped grant
            # would match none of its keys. cross-agent and cross-customer, recorded as an
            # accepted residual of this landing rather than closed by it.
            JsResource.kv(f"{ns}_agent_config", scope=None, writable=True),
            # THE scoped bucket. every principal shares it, and ``BaseCollection.l2_key`` writes
            # ``{scope}.{table}.{body}``, so this is the one resource where a per-principal grant
            # is expressible at all.
            JsResource.kv(f"{ns}-collections", scope=scope, writable=True),
            # UNSCOPED: ``checkpoints`` carries its OWN l2_key implementation
            # (``langgraph/checkpoint.py``), keyed by thread id with no scope segment. the same
            # accepted residual as ``{ns}_agent_config``.
            #
            # NAMESPACE-PREFIXED, unlike ``{ns}_agent_config`` beside it, and the difference is
            # in HOW each is opened rather than in what either is. ``AgentConfigKV`` builds its
            # own name and opens it with a direct ``js.create_key_value``, which applies no
            # prefix; the checkpointer's L2 is opened through ``KvCapable.kv_bucket``, which
            # takes a SUFFIX and layers ``{namespace}-`` over it. This grant read ``checkpoints``
            # bare, so it named a bucket nothing opens while the one every host actually
            # materialises -- ``{ns}-checkpoints`` -- was granted to nobody. A missing KV grant
            # does not raise: the open blocks to its deadline and reports an unreachable broker,
            # so a host that wrapped its checkpointer build in a try/except (14-eng-ai-survey
            # did) simply ran without a checkpointer until the first graph call failed.
            JsResource.kv(f"{ns}-checkpoints", scope=None, writable=True),
            # memory extraction throttle: the agent's MemoryExtractor uses a
            # per-conversation SET-NX-with-TTL key in this bucket to rate-limit
            # extraction. without the grant the JS API calls are denied and
            # time out, so the gate fails open (no throttling).
            JsResource.kv(f"{ns}-ratelimits", scope=None, writable=True),
            # in-process tool serving: the in-process tool server verifies the proxy's body-bound
            # assertion under enforce and records single-use nonces here (mirrors ``_tool_pod``). used
            # in BOTH devx (``DevInProcessStrategy`` builtins) and production
            # (``ProdExternalPodsStrategy`` workspace + ``knowledge_drafts`` tools).
            JsResource.kv(f"{ns}-proxy_assertion_nonces", scope=None, writable=True),
            # the durable answer stream carries BOTH directions this pod touches: the results its
            # in-process tool server delivers, and the replies the registry delivers back to it for
            # the long calls it makes. one stream, so one JetStream control-plane grant.
            JsResource.stream(f"{ns}-channels-deliver"),
            JsResource.stream(result_stream_name()),
            # the audit stream this pod both PUBLISHES to (``audit.tool.call``, granted
            # above) and CONSUMES from. An agent that runs its own audit consumer calls
            # ``ensure_jetstream_stream`` on it, and without this grant the create and
            # update are refused -- so the consumer never binds and every audit envelope
            # it was meant to durably record is dropped. That failure is quiet: the
            # publish side keeps succeeding, so audit looks healthy from the emitter.
            JsResource.stream(f"{ns}-audit"),
            # the agent's OWN coordination buckets, each composed under ``scope`` so a
            # declaration can only ever reach this agent's space. LAST, so a reader sees the
            # platform's fixed grants above and this agent's variable ones below.
            *_coordination_resources(ns, scope, coordination_buckets),
        ),
    )


def _tool_pod(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
    coordination_buckets: Sequence[str] | None,
) -> PrincipalPermissions:
    p = _require(pod_id, name="pod_id", principal=Principal.TOOL_POD)
    inbox = inbox_prefix_for(Principal.TOOL_POD, conn_id=conn_id or p)
    # derived from ``tool_pods.id`` -- the pod's registry primary key, which ``_resolve_tool_pod``
    # pins from the VERIFIED key id as ``claims.sub`` rather than reading off the connect name. it
    # is configured once per DEPLOYMENT, so every replica of one pod resolves to one scope and L2
    # stays a cross-pod cache; ``conn_id`` is deliberately not an input, because a per-connection
    # scope orphans the cache on every reauth. a namespace-derived scope was designed and rejected:
    # ``tool_namespace_id`` mints one row PER TOOL, is a pure function of manifest values the pod
    # itself sends, is deliberately collision-inducing across pods by its own docstring, and no such
    # row exists at connect. this needs NO new plumbing -- it is already in the claims.
    scope = kv_key_scope_for(Principal.TOOL_POD, pod_id=p)
    ns = _ns()
    # human-in-the-loop session control plane: while a pod holds a display's claim it serves the
    # owner-routed session messages (open/complete a tab, read state, close the session) forwarded
    # to it. one EXACT family literal per authorized tool, minted by hashing the SAME
    # ``allowed_namespaces`` entries that filter the pod's registration. a coarse ``{ns}.forward.>``
    # would instead let any tool pod serve any owner-routed key in the namespace, and since the key
    # segment is a digest of an arbitrary application string there is nothing else in the subject
    # left to discriminate on. SUBSCRIBE only: the owner answers on the requester's reply inbox
    # under ``allow_responses`` and never originates a forward.
    # BOTH families per tool. a session's control plane and its display stream are owner-routed
    # on the same key, so they must derive different subjects or the queue group ``serve_owner``
    # uses would split one pod's messages between the two handlers.
    hitl_sessions = tuple(
        str(Subjects.forward_scoped_wildcard(derive(name)))
        for name in (tool_namespaces or ())
        for derive in (Subjects.hitl_forward_family, Subjects.hitl_pipe_family)
    )
    # the byte pipe itself: once an attach has been answered on the control plane above, the pod
    # streams bytes on subjects that name its OWN tool digest and its OWN pod id as exact literals.
    # only the per-attach nonce is wildcarded, and it cannot be anything else -- the pod mints it
    # per stream, long after these grants were issued. the direction segment is part of each pattern
    # rather than a ``>`` tail so PUBLISH reaches only the pod's own half (``down``) and SUBSCRIBE
    # only the caller's (``up``); one ``>`` grant would let the owner publish onto both.
    pipe_down = tuple(str(Subjects.pipe_pod_wildcard(name, p, "down")) for name in (tool_namespaces or ()))
    pipe_up = tuple(str(Subjects.pipe_pod_wildcard(name, p, "up")) for name in (tool_namespaces or ()))
    publish = (
        str(Subjects.tools_register()),
        str(Subjects.tools_heartbeat(p)),  # own pod only
        str(Subjects.tools_discover()),  # polls discovery during wait_until_ready
        str(Subjects.hub_jwks()),  # fetches the JWKS to verify proxy assertions
        str(Subjects.audit_event("tool.call")),
        # Path-2 consume: a consuming tool resolves an object id -> its stored
        # key (forwarding the invoking agent's identity token; the hub verifies
        # + tenant-scopes). NOT hub_object_commit -- commit is agent-side.
        str(Subjects.hub_object_resolve()),
        # engagement scope (consumer A of the §2 keystone): a scan tool resolves
        # its call's engagement_id -> the authorized target set (same forwarded
        # identity-token auth; the hub verifies + tenant-scopes). read-only.
        str(Subjects.hub_engagement_scope()),
        # "which namespaces can my caller see". Same forwarded-token auth as the two
        # resolves above, and that is the whole reason this grant is safe to hold: the
        # hub reads the calling agent, customer and acting user off a signature it
        # verifies in-process, and the request carries no field naming any of the three,
        # so a pod holding this subject can ask about its caller and about nobody else.
        # An answer is a customer's whole namespace inventory -- every name, id and
        # owning agent -- which is why the subject must never accept a self-asserted
        # customer. Read-only; the pod asks and the hub answers.
        str(Subjects.namespace_discover()),
        # THE STANDING RESULT GRANT, and the reason this file changed at all. A scan tool runs for up
        # to 1200s while this connection is rebuilt every 60s to stay authenticated; the reply-inbox
        # right (``allow_responses``) belongs to the connection that received the call, so it is gone
        # by the time the tool finishes. Observed: exit 0, 68KB of output, permissions violation on
        # publish. This grant names the pod's OWN id, so the auth-callout re-mints it byte-identically
        # on every refresh and the answer no longer depends on which connection is current. Only this
        # pod holds it -- which is what makes it safe where a standing grant on the requester's inbox
        # tree (``_INBOX_registry_*.>``) was not: that one would have let any tool pod forge a reply
        # into any other pod's in-flight call.
        str(Subjects.tools_result_pod_wildcard(p)),
        _deadletter(ns),
        # THE GLOBAL, DELIBERATELY UN-NAMESPACED INVALIDATION SUBJECT, and granting it is a
        # DECISION rather than a detail -- ``coll-task-07c`` TP-02. A tool pod that holds an L2
        # collection must announce its own writes or every peer replica serves a value it has
        # already replaced. What the grant also confers, recorded as ACCEPTED for this landing
        # rather than left for someone to discover: a cross-customer metadata firehose (every
        # table name and entity id written by every agent, hub replica, gateway and adapter ride
        # this one subject), and a fleet-wide L1 eviction primitive, because ``_on_invalidation``
        # filters only on a SELF-ASSERTED ``origin``. The static NATS users already hold this
        # subject unprefixed, so a tool pod is not the widest holder and scoping it alone would buy
        # little. ``origin`` minted from the authenticated principal, and any move to a scoped
        # subject, are publisher-side wire-protocol changes across three repos and belong to
        # ``coll-task-08-invalidation-origin-auth``.
        CROSS_PLATFORM_CACHE_INVALIDATE,
        *pipe_down,  # own authorized tools' streams, own pod id, own half only
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.tools_internal(p)),  # own pod's proxied calls only
        str(Subjects.tools_probe(p)),  # own pod's liveness probes only
        CROSS_PLATFORM_CACHE_INVALIDATE,  # learns of a peer replica's write; see the publish half
        *hitl_sessions,  # own authorized tools' session control plane only
        *pipe_up,  # own authorized tools' streams, own pod id, caller's half only
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,  # replies to the registry's forwarded call + probe
        inbox_prefix=inbox,
        js_resources=(
            JsResource.kv(f"{ns}-proxy_assertion_nonces", scope=None, writable=True),
            # the display claim: a pod serving a human session holds a ``KVLease`` for as long as
            # it serves. every name here is the bucket that MATERIALISES: ``kv_bucket`` takes a
            # suffix and layers the connection's ``{ns}-`` over it.
            #
            # WHAT A MISSING GRANT ACTUALLY DOES, because two earlier versions of this comment got
            # it wrong in the same way. It does NOT yield ``lease=None`` and serve the display
            # unclaimed: that value is the dependency-injection path, set by a platform passing no
            # lease at all. With a lease supplied and this grant absent, ``KVLease.acquire``
            # defers opening the bucket to first use and that open raises ``KvError`` after a
            # JetStream timeout, which nothing catches. The symptom is a hard failure on the first
            # claim rather than a silent double-serve -- and because the open is deferred, a
            # platform cannot learn at construction time that it should downgrade.
            JsResource.kv(f"{ns}-leases", scope=None, writable=True),
            # the stream backing the result grant above. a result rides JetStream rather than a
            # core publish so a CONSUMER-side reconnect cannot lose an answer that took twenty
            # minutes to compute -- fixing only the publisher's half would relocate the loss.
            JsResource.stream(result_stream_name()),
            # THE SCOPED BUCKET (``coll-task-07c`` TP-01). A tool pod builds an L1+L2
            # ``BaseCollection`` stack through ``ToolServerBootstrap``, and every key it writes is
            # ``{scope}.{table}.{body}`` -- so this is the one resource on which a per-principal
            # grant is expressible at all.
            #
            # WRITABLE is the ``$KV.`` PUBLISH half only. A KV read is a ``$JS.API`` request, never
            # a ``$KV.`` subscribe, so the read authority is the scoped
            # ``$JS.API.DIRECT.GET.KV_{b}.$KV.{b}.{scope}.>`` tail that ``KV_SCOPED`` emits, and
            # ``$KV.`` appears in no subscribe allow-list anywhere.
            #
            # NEVER ``declare``: on this SHARED stream ``STREAM.UPDATE`` is a read-all primitive
            # (``republish`` / ``sources`` mirror every principal's keys onto a subject the caller
            # names). The pod BINDS the bucket the hub declared --
            # ``build_tool_pod_collection_stack`` passes ``create_if_missing=False`` for exactly
            # that reason, since a refused ``STREAM.CREATE`` is never answered and costs a
            # JetStream deadline at every startup before falling through to the bind anyway.
            JsResource.kv(f"{ns}-collections", scope=scope, writable=True),
        ),
    )


def _registry(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
    coordination_buckets: Sequence[str] | None,
) -> PrincipalPermissions:
    c = _require(conn_id, name="conn_id", principal=Principal.REGISTRY)
    inbox = inbox_prefix_for(Principal.REGISTRY, conn_id=c)
    # an infra principal's scope IS its principal value: one identity per service, and no
    # per-connection id that replicas would share.
    scope = kv_key_scope_for(Principal.REGISTRY)
    ns = _ns()
    publish = (
        # forwards calls to / probes ANY pod. the ``>`` tail (not ``.*``) spans BOTH single-token
        # Tool Pods (``tools.internal.{pod_id}``) and two-token agent in-process pods
        # (``tools.internal.{agent_id}.{instance}``); a single-token ``.*`` would silently stop
        # routing to agent in-process tool servers the moment they adopt the composite pod-id.
        str(Subjects.tools_internal_wildcard()),  # forwards calls to ANY pod (tool pod or agent in-process)
        str(Subjects.tools_probe_wildcard()),  # probes ANY pod
        str(Subjects.hub_jwks()),  # fetches the JWKS to verify identity tokens + pop
        # the registry's own half of durable delivery: it answers a long agent call here instead of
        # on the agent's reply inbox, for the same reason a pod does. the wildcard shape has the same
        # justification as ``tools.internal.>`` directly above -- one registry connection fronts EVERY
        # agent, so there is no per-connection list of agent ids to mint exact literals from, and it
        # is granted ONLY to the trusted router. what keeps that wildcard from being a redirect
        # primitive is enforced in the proxy, not here: it publishes only to a subject naming the
        # call's VERIFIED agent id (re-stamped from the identity token, never the claimed envelope).
        str(Subjects.tools_reply_wildcard()),
        _deadletter(ns),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.tools_call()),  # queue-grouped: registry
        str(Subjects.tools_discover()),  # queue-grouped: registry
        str(Subjects.tools_register()),
        str(Subjects.tools_heartbeat_wildcard()),
        str(Subjects.acl_invalidate("membership")),
        str(Subjects.acl_invalidate("assignment")),
        str(Subjects.acl_invalidate("role")),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,  # replies to agents' tools.call / tools.discover
        inbox_prefix=inbox,
        js_resources=(
            # UNPREFIXED, and deliberately so: the registry opens its catalog with a direct
            # ``js.key_value(bucket=...)`` (``threetears.registry.server``) rather than through
            # ``kv_bucket``, so no namespace prefix is ever applied to it. The other two
            # conventions in this file both prefix; this one is the exception and is the reason
            # the enforcement test allows a bare name.
            JsResource.kv("tool_catalog", scope=None, writable=True),
            # constructed at ``threetears.registry.server`` as
            # ``ReplayGuard(nc, bucket_name="pop_nonces", ...)`` -- a live call site in this
            # repository, not the usage example in ``ReplayGuard``'s own docstring -- and
            # ``ReplayGuard`` opens through ``kv_bucket``, so this one carries the prefix.
            JsResource.kv(f"{ns}-pop_nonces", scope=None, writable=True),
            # ADDED here, and it is a DATA-LOSS fix rather than a cache one. ``registry/server.py``
            # calls ``collection_registry.configure(l2_client=nc)`` and then builds a
            # ``HeartbeatCollection``, whose ``L3`` is ``None`` -- L2 IS its source of truth. So the
            # registry has been running a source-of-truth collection against a bucket it was never
            # granted, working only because the static ``registry`` NATS user carries ``$KV.>``.
            # This grant takes effect the moment that static user is narrowed
            # (``coll-task-05b``) or the registry moves onto the callout.
            JsResource.kv(f"{ns}-collections", scope=scope, writable=True),
            # the registry is on BOTH sides of durable delivery: it consumes each pod's result and
            # publishes each agent's reply, so it needs the JetStream control-plane grant for this
            # stream in both roles. it is also the process that DECLARES the stream at startup,
            # which the ``$JS.API.STREAM.*.{stream}`` entry the FULL capability mints is what
            # permits -- a live conflict with the narrowing above, and the reason capability is a
            # per-resource decision rather than a per-principal one.
            JsResource.stream(result_stream_name()),
        ),
    )


def _hub(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
    coordination_buckets: Sequence[str] | None,
) -> PrincipalPermissions:
    # the hub is the broadest principal: trust anchor + control plane + L3 broker + router. it owns
    # the whole {ns}.hub.*, {ns}.agents.*, {ns}.l3.*, and the platform-write event streams. it is
    # still NOT granted a bare `>` -- every grant below is a named family within the namespace.
    c = _require(conn_id, name="conn_id", principal=Principal.HUB)
    inbox = inbox_prefix_for(Principal.HUB, conn_id=c)
    scope = kv_key_scope_for(Principal.HUB)
    ns = _ns()
    publish = (
        f"{ns}.agents.route.*",  # routes user messages to any agent
        f"{ns}.agents.reregister_request.*.*",  # nudges any agent's pod to re-register ({agent_id}.{pod_id})
        str(Subjects.gateway_completion()),  # hub-side completions (e.g. summarization)
        str(Subjects.gateway_embedding()),
        str(Subjects.acl_invalidate("membership")),
        str(Subjects.acl_invalidate("assignment")),
        str(Subjects.acl_invalidate("role")),
        str(Subjects.gateway_catalog_epoch()),
        str(Subjects.mcp_rbac_epoch()),
        f"{ns}.hub.channel.installs.changed",  # notifies channel adapters of install changes
        # human-in-the-loop session control plane: the hub terminates the operator's WebSocket and
        # forwards session control to whichever pod holds that display. unlike the pod side this
        # cannot be an exact family literal -- one hub connection fronts EVERY tool, so there is no
        # per-connection list to mint from. a NATS wildcard matches a whole token, so the family
        # segment is either one exact literal or ``*``; there is no prefix form. the two-token
        # pattern does not match the UNSCOPED ``{ns}.forward.{key}`` family, which stays granted to
        # nobody. publish only: the hub calls, the pod serves.
        str(Subjects.forward_scoped_any_family()),
        # the byte pipe's caller half. the same reasoning as the forward family directly above: one
        # hub connection fronts EVERY tool and every pod, so there is no per-connection list to mint
        # exact literals from. what stays exact is the DIRECTION -- the hub publishes only on ``up``,
        # so no wildcard here reaches an owner's own half of any stream.
        str(Subjects.pipe_any_pod_wildcard("up")),
        _deadletter(ns),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.pipe_any_pod_wildcard("down")),  # every pod's stream to it, never a pod's own half
        str(Subjects.hub_handshake()),  # mints identity tokens
        str(Subjects.hub_jwks()),  # serves the JWKS
        str(Subjects.hub_secrets_request()),
        str(Subjects.hub_user_resolve()),
        str(Subjects.hub_object_commit()),  # Path-2: responds to object catalog commits
        str(Subjects.hub_object_resolve()),  # Path-2: responds to object id -> key resolves
        str(Subjects.hub_engagement_scope()),  # engagement scope: responds to engagement_id -> targets resolves
        # ChannelDefaultResponder answers this: the agent runtime resolves a conversation
        # channel's default engagement at the tool-call stamp seam. The agent's publish half
        # was granted and this subscribe half was not -- latent only because the hub connects
        # as a static nats.conf user holding `>`, so this table is never consulted for it.
        # On callout-minted permissions the subscription is refused, the resolve soft-fails
        # to "unbound", and a scan that SHOULD have authorized against a configured
        # engagement is refused for a missing one.
        str(Subjects.hub_channel_engagement_default_resolve()),
        # HITL approval broker: record a pending marker (agent-forwarded) + resolve an operator reply
        # (channel-router-forwarded) into an authorized approve/deny verdict on the resume rail.
        str(Subjects.hub_approval_record()),
        str(Subjects.hub_approval_resolve()),
        str(Subjects.hub_channel_installs()),
        str(Subjects.namespace_discover()),
        str(Subjects.agent_register()),
        str(Subjects.agent_deregister()),
        str(Subjects.agent_heartbeat_wildcard()),  # heartbeat monitor
        str(Subjects.l3_query()),
        str(Subjects.l3_batch()),
        f"{ns}.l3.tx.*",
        f"{ns}.datasource.*.query",
        str(Subjects.tools_register()),  # materializes tool namespace rows
        str(Subjects.workspaces_create()),
        str(Subjects.knowledge_draft()),
        str(Subjects.hub_usage_track()),
        str(Subjects.audit_wildcard()),  # unified audit consumer
        str(Subjects.acl_invalidate("membership")),
        str(Subjects.acl_invalidate("assignment")),
        str(Subjects.acl_invalidate("role")),
        f"{ns}.hub.stream.*.*",  # subscribes to agent token streams it dispatched ({agent_id}.{correlation_id})
        str(Subjects.gateway_catalog_epoch()),
        str(Subjects.mcp_rbac_epoch()),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,
        inbox_prefix=inbox,
        js_resources=(
            # the epoch counter: every EPHEMERAL config epoch counts here.
            # a missing grant does NOT raise -- the JS call blocks to its
            # deadline, which reads as an unreachable broker rather than as a
            # permission problem, so this is the kind of omission that costs a
            # day. the epoch client opens the bucket on its first bump.
            JsResource.kv(f"{ns}-epochs", scope=None, writable=True),
            # THE UNDERSCORE SPELLING IS NOT NECESSARILY A TYPO, and that is the whole reason
            # these are here. A bucket created by a DIRECT ``js.create_key_value(bucket=...)``
            # never receives the ``{ns}-`` prefix ``kv_bucket`` layers on, so a component that
            # builds its own ``f"{namespace}_thing"`` name owns that name verbatim.
            #
            # PROVEN for ``agent_config`` only: the hub's ``AgentConfigKV`` creates it that way
            # and its own docstring calls it platform-historical, and the agent router reads it
            # back. Every OTHER underscore-spelled entry below is kept on PRECAUTION rather than
            # on evidence: no opener was found for it in either repository, but "not found" is not
            # "absent", and the two failure directions are not symmetric. A grant naming a bucket
            # nothing opens costs permission surface; removing one that something does open costs
            # a JetStream operation that blocks to its deadline instead of raising, which reads as
            # an unreachable broker. Removing them on absence-of-evidence is the mistake that was
            # already made here once.
            #
            # Do not "normalise" any of these to ``{ns}-``: that renames a live bucket.
            JsResource.kv(f"{ns}_agent_config", scope=None, writable=True),
            JsResource.kv(f"{ns}_agent_sessions", scope=None, writable=True),
            JsResource.kv(f"{ns}_revoked_tokens", scope=None, writable=True),
            JsResource.kv(f"{ns}_login_lockouts", scope=None, writable=True),
            JsResource.kv(f"{ns}_rate_limits", scope=None, writable=True),
            # ADDED rather than corrected, and the distinction matters. The hub's own router rate
            # limiter creates ``f"{namespace}_ratelimit"`` -- singular, no trailing ``s`` -- with a
            # direct ``js.create_key_value``, and no grant named it, so the limiter's first
            # operation blocked to its deadline instead of raising. ``{ns}_rate_limits`` above is
            # NOT removed in the same breath: nothing proves it dead, and this file has already
            # been burned once by treating "no opener found" as "no opener".
            JsResource.kv(f"{ns}_ratelimit", scope=None, writable=True),
            # THE DECLARING IDENTITY, and the ONLY resource anywhere in this file that carries the
            # declare capability. ``coll-task-04a`` makes hub bootstrap the canonical declarer of
            # the collections bucket -- it must run ``ensure_kv_bucket`` at lifespan and on every
            # NATS reconnect to keep ``allow_direct: true``, which needs ``STREAM.CREATE`` and
            # ``STREAM.UPDATE``. No pod principal may ever hold this: on a SHARED stream ``UPDATE``
            # is a read-all primitive, since ``republish`` / ``sources`` mirror every key to a
            # subject the caller names. ``tests/enforcement/test_kv_grant_capability.py`` refuses
            # the capability anywhere but here.
            #
            # WHAT THIS RESOURCE DOES NOT COVER, recorded so ``coll-task-05b`` does not discover it
            # in production: ``hub/admin/backup_engine.py``'s ``_flush_nats_kv`` enumerates
            # ``js.key_value_store_names()`` and then WATCHES (``kv.keys()`` -> ``watchall()`` ->
            # a JetStream consumer) and per-key DELETES every bucket, collections included. Neither
            # the consumer create nor an out-of-scope ``$KV.`` delete is grantable under a scoped
            # capability. The hub reaches both today through its static ``>``; the moment that ``>``
            # is enumerated, that restore path breaks -- silently, because its ``try:`` opens before
            # ``key_value_store_names()`` and its ``except Exception: log.warning`` closes after the
            # whole nested loop, so one refused bucket aborts every REMAINING bucket mid-flush at
            # WARNING. Fix the hub, not this grant.
            JsResource.kv(f"{ns}-collections", scope=scope, writable=True, declare=True),
            JsResource.stream(f"{ns}-channels-deliver"),
        ),
    )


def _gateway(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
    coordination_buckets: Sequence[str] | None,
) -> PrincipalPermissions:
    c = _require(conn_id, name="conn_id", principal=Principal.GATEWAY)
    inbox = inbox_prefix_for(Principal.GATEWAY, conn_id=c)
    scope = kv_key_scope_for(Principal.GATEWAY)
    ns = _ns()
    publish = (
        f"{ns}.gateway.stream.*.*",  # streams tokens back for any in-flight completion ({agent_id}.{correlation_id})
        str(Subjects.hub_usage_track()),
        _deadletter(ns),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.gateway_completion()),  # queue-grouped: ai-gateway
        str(Subjects.gateway_embedding()),  # queue-grouped: ai-gateway
        str(Subjects.gateway_health()),
        str(Subjects.gateway_catalog_epoch()),
        str(Subjects.acl_invalidate("membership")),
        str(Subjects.acl_invalidate("assignment")),
        str(Subjects.acl_invalidate("role")),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,  # replies to completion / embedding / health requests
        inbox_prefix=inbox,
        js_resources=(
            # the epoch counter: every EPHEMERAL config epoch counts here.
            # a missing grant does NOT raise -- the JS call blocks to its
            # deadline, which reads as an unreachable broker rather than as a
            # permission problem, so this is the kind of omission that costs a
            # day. the epoch client opens the bucket on its first bump.
            JsResource.kv(f"{ns}-epochs", scope=None, writable=True),
            # the gateway builds two registries that carry L2 collections (``gateway/acl.py`` and
            # ``gateway/service.py``), so it is a full participant in the shared bucket.
            JsResource.kv(f"{ns}-collections", scope=scope, writable=True),
        ),
    )


def _channel_adapter(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
    coordination_buckets: Sequence[str] | None,
) -> PrincipalPermissions:
    c = _require(conn_id, name="conn_id", principal=Principal.CHANNEL_ADAPTER)
    inbox = inbox_prefix_for(Principal.CHANNEL_ADAPTER, conn_id=c)
    scope = kv_key_scope_for(Principal.CHANNEL_ADAPTER)
    ns = _ns()
    publish = (
        f"{ns}.agents.route.*",  # routes inbound channel messages to agents
        str(Subjects.hub_channel_installs()),  # fetches its bot installs
        str(Subjects.hub_user_resolve()),  # resolves a channel sender to a platform user
        # HITL: the router (in this sandboxed adapter) forwards an operator reply to the approval
        # broker, which authorizes + resolves it into a resume verdict. reply only; no marker write.
        str(Subjects.hub_approval_resolve()),
        f"{ns}.hub.channel.installs.changed",  # best-effort orphan-reload signal
        _deadletter(ns),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.channels_deliver_wildcard()),  # durable consumer of agent answers
        f"{ns}.hub.channel.installs.changed",  # reconcile live connections on install changes
        f"{ns}.channels.room.*",  # cross-pod room fanout (live socket delivery)
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    return PrincipalPermissions(
        publish=(*publish, f"{ns}.channels.room.*"),  # publishes room frames too
        subscribe=subscribe,
        allow_responses=True,
        inbox_prefix=inbox,
        js_resources=(
            JsResource.kv(f"{ns}-collections", scope=scope, writable=True),
            JsResource.stream(f"{ns}-channels-deliver"),
        ),
    )


def _agent_router(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
    coordination_buckets: Sequence[str] | None,
) -> PrincipalPermissions:
    # the sticky router: it drains the durable inbound-turn stream, forwards each turn to whichever
    # pod currently owns the conversation, and awaits the pod's completion signal before acking.
    #
    # ADOPTED HERE (``coll-task-05a`` GRANT-11) rather than in the plan chunk that also claims it,
    # because two downstream shards block on the enum value and that chunk sits behind ten unbuilt
    # ones. This process connects as the STATIC ``agent_router`` NATS user today and never reaches
    # the auth callout, so what this resolver buys immediately is a legal, pinnable
    # ``kv_key_scope_for`` value plus the source ``coll-task-05b`` generates its conf from. That
    # shard OWNS re-verifying this subject list against the running process before it narrows the
    # static user; the list below is what ``aibots/agent_router/`` demonstrably uses today.
    c = _require(conn_id, name="conn_id", principal=Principal.AGENT_ROUTER)
    inbox = inbox_prefix_for(Principal.AGENT_ROUTER, conn_id=c)
    scope = kv_key_scope_for(Principal.AGENT_ROUTER)
    ns = _ns()
    publish = (
        f"{ns}.agents.internal.*.*",  # forwards a turn to the sticky pod ({agent_id}.{pod_id})
        f"{ns}.agents.reregister_request.*.*",  # nudges a pod to re-register
        str(Subjects.agent_turn_deadletter()),  # parks a turn that exhausted its redelivery budget
        _deadletter(ns),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",
        str(Subjects.agent_register()),
        str(Subjects.agent_deregister()),
        str(Subjects.agent_heartbeat_wildcard()),  # liveness monitor over every agent + pod
        str(Subjects.agent_turn_wildcard()),  # the durable consumer's filter
        f"{ns}.agents.complete.*",  # the per-turn completion signal it awaits before acking
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,  # answers the hub's route request
        inbox_prefix=inbox,
        js_resources=(
            # UNPREFIXED: opened with a direct ``js.key_value(bucket=...)`` in
            # ``aibots/agent_router/server.py``, whose default is the bare ``agent_router_catalog``,
            # so no namespace prefix is ever applied -- the same exception the registry's
            # ``tool_catalog`` is.
            JsResource.kv("agent_router_catalog", scope=None, writable=True),
            # ALSO UNPREFIXED, and ALSO a direct bind: ``agent_router/proxy.py``'s ``start`` does
            # ``js.key_value(bucket=f"{namespace}_agent_config")`` for the per-agent turn-timeout
            # lookup, so the name is underscore-spelled verbatim and never receives the ``{ns}-``
            # that ``kv_bucket`` layers on. Matches the live ``aibots_agent_config`` bucket.
            #
            # READ-ONLY, and that is a rule rather than a trim: CLAUDE.md's Config Source-of-Truth
            # makes ``platform.agents`` the source and this bucket a hot cache over it, written
            # ONLY by the hub's admin endpoints. The router resolves a timeout from it and never
            # puts (``_config_kv`` is read at one site), and a KV read is a ``$JS.API`` request
            # rather than a ``$KV.`` publish, so withholding write authority costs it nothing.
            #
            # ADDED here (evidence-ledger bug 21) because the resolver is the canonical model and
            # the omission was already being papered over downstream: the hub's static-conf
            # generator carries this same bucket in its ``_EXTRA_RESOURCES`` table with the note
            # "``_agent_router`` in the canonical model does not declare it; reported upstream".
            # That entry must be dropped when this lands, or the rendered conf declares the bucket
            # twice.
            JsResource.kv(f"{ns}_agent_config", scope=None, writable=False),
            # ``PodAffinityCollection`` -- sticky conversation-to-pod routing, and another
            # ``L3 = None`` collection, so a key that no longer resolves is a lost route rather
            # than a cache miss.
            JsResource.kv(f"{ns}-collections", scope=scope, writable=True),
            # the durable inbound-turn stream it consumes and dead-letters into.
            JsResource.stream(f"{ns}-agents-turn"),
        ),
    )


def _dataset_executor(
    *,
    agent_id: str | None,
    pod_id: str | None,
    conn_id: str | None,
    tool_namespaces: Sequence[str] | None,
    coordination_buckets: Sequence[str] | None,
) -> PrincipalPermissions:
    # D14's separate deployment of the hub image: it drains dataset build work, holds a KVLease for
    # the run it owns, and releases the admission slots the hub took.
    #
    # ADOPTED for the same reason as :func:`_agent_router`, and with the same caveat: it connects as
    # the STATIC ``dataset_executor`` user today.
    #
    # ITS LEASE AND ADMISSION BUCKETS ARE NOT NAMED HERE, and the omission is deliberate rather than
    # an oversight. ``aibots/hub/datasets/executor/__main__.py`` reads both names out of the
    # ENVIRONMENT (``configured_admission_bucket(os.environ)`` and the configured lease bucket), so
    # there is no literal in either repository to pin them to and a guessed name would be a grant
    # for a bucket nothing opens while the real one stays ungranted -- a JetStream call that blocks
    # to its deadline and reads as an unreachable broker. ``coll-task-05b`` enumerates this user's
    # static conf and must take those two names from the deployment's own configuration.
    c = _require(conn_id, name="conn_id", principal=Principal.DATASET_EXECUTOR)
    inbox = inbox_prefix_for(Principal.DATASET_EXECUTOR, conn_id=c)
    scope = kv_key_scope_for(Principal.DATASET_EXECUTOR)
    ns = _ns()
    publish = (
        str(Subjects.hub_usage_track()),
        _deadletter(ns),
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    subscribe = (
        f"{inbox}.>",
        CROSS_PLATFORM_CACHE_INVALIDATE,
    )
    return PrincipalPermissions(
        publish=publish,
        subscribe=subscribe,
        allow_responses=True,
        inbox_prefix=inbox,
        js_resources=(JsResource.kv(f"{ns}-collections", scope=scope, writable=True),),
    )


#: one resolver per :class:`Principal` member. the mapping is total by construction --
#: ``build_permissions`` indexes it directly, so a member added without a resolver raises a
#: ``KeyError`` at the first connect rather than silently resolving to nothing.
_RESOLVERS = {
    Principal.AGENT_POD: _agent_pod,
    Principal.TOOL_POD: _tool_pod,
    Principal.REGISTRY: _registry,
    Principal.HUB: _hub,
    Principal.GATEWAY: _gateway,
    Principal.CHANNEL_ADAPTER: _channel_adapter,
    Principal.AGENT_ROUTER: _agent_router,
    Principal.DATASET_EXECUTOR: _dataset_executor,
}
