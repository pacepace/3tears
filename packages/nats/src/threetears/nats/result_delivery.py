"""the contract for delivering a tool result that outlives the connection which received the call.

NATS validates a connection's user JWT at CONNECT and offers no in-band re-auth, so refreshing a
credential IS a reconnect. ``allow_responses`` -- the right to answer a request without holding a
standing publish grant on the requester's inbox -- is scoped to the connection that RECEIVED the
request. Those two facts compose into a hard limit: **any correct credential refresh destroys the
right to answer a call that is still in flight.** Observed in production: a 92-second scan finished
with exit 0 and 68KB of results, and the publish was refused because the connection had been recycled
56 seconds earlier.

Neither obvious lever is acceptable. Raising the JWT TTL to cover the longest tool timeout means
20-minute credentials -- a security regression. Granting a standing publish right on the requester's
inbox tree (``_INBOX_registry_*.>``) lets any tool pod forge a reply into any other pod's in-flight
call, a cross-customer response-injection hole.

So a long call does not reply; it DELIVERS. The responder publishes its answer to a subject named
with its OWN identity (:meth:`threetears.nats.Subjects.tools_result` /
:meth:`~threetears.nats.Subjects.tools_reply`), on a standing grant minted from ids the auth-callout
already resolved. Re-minted identically on every refresh, so reconnects stop mattering; scoped to one
principal, so nothing can forge into a peer's call. Delivery rides JetStream rather than core publish
so a result that took twenty minutes to compute is not lost to a CONSUMER-side reconnect either.

This module owns only the pure decisions -- which calls take that path, what the stream is called, and
whether a responder is allowed to publish to the subject it was handed. It imports nothing from the
NATS client, so both responders (tool pod, registry) and both callers (registry, agent SDK) share one
implementation of each rule rather than four sympathetic copies.

Every ownership check is derived from the SUBJECT FACTORY rather than from a hand-built string. The
alternative -- reassembling ``{ns}.tools.result.{pod}.`` locally -- reintroduces the namespace as a
second source of truth, and the two only have to disagree once for a responder to refuse every
delivery subject it is legitimately given, with the failure reading as a permissions problem.
"""

from __future__ import annotations

from uuid import UUID

from threetears.nats.subjects import Subjects, get_default_namespace

__all__ = [
    "RESULT_ACK_TIMEOUT_SECONDS",
    "RESULT_RETENTION_SECONDS",
    "RESULT_STREAM_SUFFIX",
    "SYNC_REPLY_BUDGET_SECONDS",
    "reply_subject_prefix_for_agent",
    "reply_subject_is_owned_by_agent",
    "requires_async_result",
    "result_stream_name",
    "result_subject_prefix_for_pod",
    "result_subject_is_owned_by_pod",
]

#: the longest a call may run and still be answered on the request/reply inbox.
#:
#: A responder that owes a synchronous reply defers its scheduled re-auth until the reply is out
#: (``ToolServer.drain_before_reauth``), and that deferral is bounded -- waiting past the JWT's real
#: deadline trades a lost reply for a dead connection, which is strictly worse. This is that bound, so
#: a call the caller CHOSE to run synchronously always fits inside the window the responder is willing
#: to hold the connection open for. It must stay <= the re-auth drain grace
#: (``threetears.agent.tools.nats_reauth.REAUTH_BUFFER_SECONDS``); an enforcement test in the
#: agent-tools package holds the two in that relation, since they are set in different packages and
#: nothing else relates them.
SYNC_REPLY_BUDGET_SECONDS = 30.0

#: how long a caller waits for the responder to ACCEPT an asynchronous call.
#:
#: The accept is a plain request/reply, answered before any work starts, so it is bounded by
#: scheduling rather than by the tool. Keeping it short is what preserves the caller's fast
#: dead-pod signal: a pod that is gone yields "no responders" in milliseconds and the registry can
#: fail over to a sibling endpoint, instead of the whole 20-minute tool budget elapsing first.
RESULT_ACK_TIMEOUT_SECONDS = 10.0

#: stream-name suffix for the durable result/reply stream (``NatsClient.ensure_jetstream_stream``
#: layers the ``{namespace}-`` prefix on).
RESULT_STREAM_SUFFIX = "tools-results"

#: how long a delivered-but-unclaimed result is retained.
#:
#: In the ordinary case the caller is already waiting when the result lands and claims it within
#: milliseconds; retention exists for the window where the caller is mid-reconnect. Generous enough
#: to cover a reconnect storm, short enough that the memory-backed stream cannot accumulate a
#: cluster's worth of tool output.
RESULT_RETENTION_SECONDS = 900.0


def requires_async_result(timeout_seconds: float | None) -> bool:
    """whether a call with this timeout must be answered out-of-band rather than on the reply inbox.

    THE single predicate, shared by every caller (registry -> pod, agent -> registry) so the two hops
    never disagree about which path a call is on. A caller that chose the synchronous path and a
    responder that chose the asynchronous one would leave the answer on a subject nobody reads.

    an unknown timeout (``None``) is treated as long: the failure mode of guessing "short" is a
    silently discarded result, the failure mode of guessing "long" is one extra round trip.

    :param timeout_seconds: the call's effective timeout, or ``None`` when unknown
    :ptype timeout_seconds: float | None
    :return: ``True`` when the answer must be delivered on a durable, responder-owned subject
    :rtype: bool
    """
    return timeout_seconds is None or timeout_seconds > SYNC_REPLY_BUDGET_SECONDS


def result_stream_name() -> str:
    """the full JetStream stream name holding both durable answer families.

    One stream over both families (``tools.result.>`` and ``tools.reply.*.*``) rather than two: a
    subject belongs to exactly one stream on an account, both families have the same retention
    story, and every principal that touches either declares the same single name in its JetStream
    grant list.

    Reads the same process-wide namespace the subject factory does, so the stream a publisher names
    and the subjects it publishes can never end up in different namespaces.

    :return: the namespace-prefixed stream name
    :rtype: str
    :raises NamespaceNotConfiguredError: when no namespace has been configured for this process
    """
    return f"{get_default_namespace()}-{RESULT_STREAM_SUFFIX}"


def result_subject_prefix_for_pod(pod_id: str | UUID) -> str:
    """the subject prefix a tool pod may publish its own results under, ``.``-terminated.

    Derived by stripping the wildcard off the pod's own standing grant, so the prefix is by
    construction the one the subject factory builds against.

    :param pod_id: the responding pod's own routing identifier
    :ptype pod_id: str | UUID
    :return: prefix ``{ns}.tools.result.{pod_id}.``
    :rtype: str
    """
    return str(Subjects.tools_result_pod_wildcard(pod_id)).removesuffix(">")


def result_subject_is_owned_by_pod(subject: str, *, pod_id: str | UUID) -> bool:
    """whether ``subject`` lies inside the standing result grant of pod ``pod_id``.

    The responder checks the subject it was HANDED before publishing to it. The broker already
    enforces this -- the pod's JWT carries no grant outside its own subtree -- but the two failures
    read completely differently: a permissions violation surfaces as an opaque error on the publish
    of a result that has already been computed, whereas this check refuses the call up front and says
    which subject was asked for. It also stops a compromised caller from using pods as a probe for
    what the broker will and will not accept.

    :param subject: the result subject the caller asked the responder to publish to
    :ptype subject: str
    :param pod_id: the responding pod's own routing identifier
    :ptype pod_id: str | UUID
    :return: ``True`` when the subject is the responder's own to publish
    :rtype: bool
    """
    return _is_single_token_under(subject, result_subject_prefix_for_pod(pod_id))


def reply_subject_prefix_for_agent(agent_id: str | UUID) -> str:
    """the subject prefix one agent's tool replies are delivered under, ``.``-terminated.

    :param agent_id: the calling agent's authenticated identity
    :ptype agent_id: str | UUID
    :return: prefix ``{ns}.tools.reply.{agent_id}.``
    :rtype: str
    """
    return str(Subjects.tools_reply_agent_subtree(agent_id)).removesuffix("*")


def reply_subject_is_owned_by_agent(subject: str, *, agent_id: str | UUID) -> bool:
    """whether ``subject`` is the reply subject of agent ``agent_id``.

    The registry holds a two-token wildcard publish grant on the reply family, because one registry
    connection fronts every agent and there is no per-connection list of agent ids to mint exact
    literals from. This is the containment that makes that wildcard safe: the registry publishes only
    to a subject naming the call's VERIFIED agent id (the one re-stamped from the identity token,
    never the one the envelope claimed), so a caller cannot redirect its result onto a peer's
    in-flight call.

    :param subject: the reply subject the caller asked to be answered on
    :ptype subject: str
    :param agent_id: the VERIFIED identity of the calling agent
    :ptype agent_id: str | UUID
    :return: ``True`` when the subject belongs to that agent
    :rtype: bool
    """
    return _is_single_token_under(subject, reply_subject_prefix_for_agent(agent_id))


def _is_single_token_under(subject: str, prefix: str) -> bool:
    """whether ``subject`` is ``prefix`` plus exactly ONE further non-empty subject token.

    the trailing-token rule is the point, not decoration. a bare ``startswith`` would accept
    ``{prefix}a.b``, letting a caller push a responder's publish one level deeper than the family it
    owns -- and, with the wildcard grants in play, into a sibling family's routing space. it also
    rejects the empty tail and any embedded wildcard, neither of which is a legal delivery target.

    :param subject: candidate subject
    :ptype subject: str
    :param prefix: the ``.``-terminated prefix the subject must sit directly under
    :ptype prefix: str
    :return: ``True`` when the subject is exactly one token below the prefix
    :rtype: bool
    """
    result = False
    if subject.startswith(prefix):
        tail = subject[len(prefix) :]
        result = bool(tail) and "." not in tail and "*" not in tail and ">" not in tail
    return result
