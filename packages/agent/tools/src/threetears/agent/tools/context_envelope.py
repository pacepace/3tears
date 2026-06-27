"""unified call-context envelope for NATS tool dispatch.

the per-call identity dimensions (conversation_id, user_id, customer_id,
correlation_id, agent_id) used to be flat fields on
:class:`CallRequest`, :class:`ProxyCallRequest`, and
:class:`ToolCallScope`. every new dimension required a surgical edit in
five places: wire model, proxy model, scope, tool wrapper, message
handler, and every consumer. :class:`CallContext` collapses them into
one value type passed whole through the envelope; adding a new
dimension becomes a one-line addition here.

the ``trace`` field is the escape hatch for dimensions the platform has
not yet named (request_id, parent_span_id, effective_role_ids cache key,
etc.). callers can stash a correlation key without forcing a schema
bump. it is NOT a general metadata bag: only put identity-shaped
dimensions there.

pure value type. no IO, no lifecycle. :meth:`with_trace` returns a new
instance so callers can layer trace entries without mutating a shared
value.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from threetears.observe import clear_context, set_context

__all__ = [
    "CallContext",
    "bind_log_context",
]


class CallContext(BaseModel):
    """unified identity + trace envelope for a single tool dispatch.

    ride whole through the NATS envelope as a nested JSON object on
    :class:`CallRequest` / :class:`ProxyCallRequest`. each field is
    optional because different call paths populate different subsets:
    stateless utility tool calls from the hub carry none of them;
    agent-originated calls carry conversation/user/customer; internal
    workflow invocations carry correlation only. consumers read the
    dimension they need and ignore the rest.

    :param conversation_id: conversation identifier when the call
        belongs to a user-facing conversation
    :ptype conversation_id: UUID | None
    :param user_id: invoking user identifier
    :ptype user_id: UUID | None
    :param customer_id: owning customer identifier for RBAC + audit
    :ptype customer_id: UUID | None
    :param correlation_id: request correlation identifier threaded
        through logs and downstream NATS calls
    :ptype correlation_id: UUID | None
    :param agent_id: THE agent identifier for this call; the proxy's
        routing + authorization decision and the tool pod's
        origin-identity read both source from this one field. there is
        no separate top-level ``agent_id`` on
        :class:`~threetears.registry.proxy.ProxyCallRequest` -- this
        is the single source of truth
    :ptype agent_id: UUID | None
    :param identity_token: Hub-issued, EdDSA-signed identity assertion
        (a compact JWS; see
        :mod:`threetears.core.security.identity_token`) verified at the
        registry proxy before RBAC, so authorization evaluates a VERIFIED
        caller identity rather than the self-asserted envelope. rides on
        the context so it travels whole through both
        :class:`~threetears.registry.proxy.ProxyCallRequest` and
        :class:`~threetears.agent.tools.server.CallRequest`. ``None``
        until the platform-auth rollout reaches the enforce stage; nothing
        reads it yet
    :ptype identity_token: str | None
    :param trace: escape hatch for identity dimensions not yet promoted
        to first-class fields; map of short string keys to string
        values. intentionally narrow: do NOT use for arbitrary per-call
        payload
    :ptype trace: dict[str, str]
    :param user_timezone: IANA timezone name resolved per-message from
        the calling channel adapter (browser
        ``Intl.DateTimeFormat()`` for websocket, ``users.info.tz`` for
        slack, locale-derived fallback for discord). per-message rather
        than per-user-record so a user who travels mid-conversation gets
        accurate local time on the next turn. tools key off this field
        for tz-aware rendering (e.g.
        :class:`~threetears.agent.tools.builtin.current_date.CurrentDateTool`)
        without depending on the LLM to thread the value through the
        tool call args
    :ptype user_timezone: str | None
    :param user_locale: BCP 47 locale tag (``en-US``, ``ja-JP``)
        resolved from the same per-message channel-adapter source.
        consumers use it for number / currency / date formatting hints
    :ptype user_locale: str | None
    """

    model_config = ConfigDict(frozen=True)

    conversation_id: UUID | None = None
    user_id: UUID | None = None
    customer_id: UUID | None = None
    correlation_id: UUID | None = None
    agent_id: UUID | None = None
    identity_token: str | None = None
    trace: dict[str, str] = Field(default_factory=dict)
    user_timezone: str | None = None
    user_locale: str | None = None

    def with_trace(self, overlay: dict[str, str]) -> "CallContext":
        """return a new :class:`CallContext` with ``overlay`` merged over ``trace``.

        pure function. does not mutate the source instance; the source's
        ``trace`` dict is unchanged after the call. keys in ``overlay``
        win over keys already present in ``self.trace``. identity
        fields carry over untouched. callers layering multiple trace
        updates should chain: ``ctx.with_trace({'a': '1'}).with_trace({'b': '2'})``.

        :param overlay: map of trace entries to merge on top of the
            current ``trace``; may be empty
        :ptype overlay: dict[str, str]
        :return: new :class:`CallContext` with merged ``trace`` and
            identity fields copied verbatim
        :rtype: CallContext
        """
        merged_trace = dict(self.trace)
        merged_trace.update(overlay)
        result = self.model_copy(update={"trace": merged_trace})
        return result


def bind_log_context(ctx: CallContext | None) -> None:
    """bind the canonical logging context tags from a :class:`CallContext`.

    thin wrapper over :func:`threetears.observe.set_context` that
    projects a :class:`CallContext` (or ``None``) onto the platform's
    canonical log-tag keys (``cid``/``conv``/``user``/``agent``/
    ``customer``) defined in ``docs/guides/logging-contract.md``.
    callers at a NATS request boundary invoke this once per inbound
    envelope so every log line inside the handler and its callees
    renders with those tags; pair with
    :func:`threetears.observe.clear_context` in a ``finally``.
    passing ``None`` resets all tags (for paths that parsed a malformed
    envelope and have no context to bind).
    identity fields are stringified at the binding border so the
    :class:`CallContext` itself stays a pure value type with
    :class:`UUID` fields throughout the codebase.

    :param ctx: unified call context carrying the identity dimensions,
        or ``None`` to reset all tags
    :ptype ctx: CallContext | None
    :return: nothing
    :rtype: None
    """
    if ctx is None:
        clear_context()
        return
    set_context(
        cid=str(ctx.correlation_id) if ctx.correlation_id is not None else None,
        conv=str(ctx.conversation_id) if ctx.conversation_id is not None else None,
        user=str(ctx.user_id) if ctx.user_id is not None else None,
        agent=str(ctx.agent_id) if ctx.agent_id is not None else None,
        customer=str(ctx.customer_id) if ctx.customer_id is not None else None,
    )
