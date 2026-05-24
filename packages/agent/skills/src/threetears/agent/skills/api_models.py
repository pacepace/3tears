"""REST request/response DTOs for the agent-skills HTTP surface.

Per the locked PLACEMENT decision (§3.1) the REST request/response
shapes for the consumer's ``/api/v1/skills`` router are platform-defined
and live here, NOT in the consumer. metallm's router imports these and
wires them to FastAPI; this module stays FastAPI-free so the package's
dependency surface doesn't grow a web framework. Pure pydantic only.

Two model families:

- **Response models** -- serialization targets for the L3 entities
  (:class:`~threetears.agent.skills.entities.AgentSkillEntity`,
  :class:`~threetears.agent.skills.entities.AgentSkillInvocationEntity`).
  Timestamps + UUIDs cross the HTTP/JSON boundary as ``str`` so the
  router serializes ``str(entity.<uuid>)`` / ``<dt>.isoformat()`` once
  and clients never need a custom decoder. ``date`` fields are typed
  ``str`` for the same reason.
- **Request models** -- ``CreateSkillRequest`` / ``UpdateSkillRequest``
  mirror the tool input schemas (:class:`SkillCreateInput` /
  :class:`SkillUpdateInput`) MINUS the server-derived identity fields.
  ``user_id`` / ``agent_id`` are NEVER accepted from the wire -- the
  router derives them from the authenticated principal. ``extra='forbid'``
  turns an attempt to smuggle them in into a 422 rather than a silently
  ignored field, so the identity boundary is enforced structurally.

The request models subclass the existing tool input schemas so the
field definitions (defaults, ``Literal`` value sets, descriptions) stay
single-sourced: a future field addition to ``SkillCreateInput`` flows
through here automatically, and ``extra='forbid'`` is the only thing
this layer adds.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from threetears.agent.skills.tools import SkillCreateInput

__all__ = [
    "CreateSkillRequest",
    "SkillInvocationListResponse",
    "SkillInvocationResponse",
    "SkillListResponse",
    "SkillResponse",
    "SkillSummary",
    "UpdateSkillRequest",
]


# ---------------------------------------------------------------------------
# Response models -- serialization targets for the L3 entities
# ---------------------------------------------------------------------------


class SkillResponse(BaseModel):
    """Full-detail serialization of an ``agent_skills`` row.

    The router builds this from an
    :class:`~threetears.agent.skills.entities.AgentSkillEntity`,
    stringifying UUIDs and ISO-formatting timestamps. ``kind`` is always
    ``'prose'`` for an entity-backed row -- tool-skills are synthesized by
    the router from the registry, never from this table -- but the field
    is typed with both values so the same response shape covers a UNION
    listing that the router may assemble.
    """

    skill_id: str
    agent_id: str
    user_id: str
    kind: Literal["prose", "tool"]
    name: str
    summary: str
    body: str | None
    prompt_mode: Literal["additive", "replace"]
    tool_additions: list[str]
    tool_restrictions: list[str]
    trigger_keywords: str
    tags: list[str]
    source: str
    enabled: bool
    use_count: int
    last_used_at: str | None
    success_count: int
    failure_count: int
    last_failure_at: str | None
    date_created: str
    date_updated: str


class SkillSummary(BaseModel):
    """List-variant serialization of an ``agent_skills`` row (no body).

    The catalog listing omits ``body`` (and the identity + composition
    fields a list view doesn't need) to keep payloads small;
    :class:`SkillResponse` carries the full detail for a single-skill
    fetch.
    """

    skill_id: str
    kind: Literal["prose", "tool"]
    name: str
    summary: str
    prompt_mode: Literal["additive", "replace"]
    tags: list[str]
    enabled: bool
    use_count: int
    last_used_at: str | None
    success_count: int
    failure_count: int
    date_created: str


class SkillListResponse(BaseModel):
    """Envelope for ``GET /skills`` -- a page of summaries plus the total."""

    skills: list[SkillSummary]
    total_count: int


class SkillInvocationResponse(BaseModel):
    """Serialization of an ``agent_skill_invocations`` row.

    Built from an
    :class:`~threetears.agent.skills.entities.AgentSkillInvocationEntity`.
    ``message_id`` is ``None`` until the consumer's post-LLM hook attaches
    the assistant response; ``outcome`` / ``outcome_source`` are ``None``
    until a ``[SUCCESS]`` / ``[FAILED]`` marker is parsed.
    """

    invocation_id: str
    skill_id: str
    conversation_id: str
    message_id: str | None
    invocation_source: Literal["wake", "invoke"]
    invoked_at: str
    outcome: Literal["success", "failure"] | None
    outcome_source: Literal["agent_marker", "user_feedback"] | None
    notes: str | None


class SkillInvocationListResponse(BaseModel):
    """Envelope for ``GET /skills/{id}/invocations``."""

    invocations: list[SkillInvocationResponse]
    total_count: int


# ---------------------------------------------------------------------------
# Request models -- editable fields only; identity is server-derived
# ---------------------------------------------------------------------------


class CreateSkillRequest(SkillCreateInput):
    """Request body for ``POST /skills``.

    Subclasses :class:`~threetears.agent.skills.tools.SkillCreateInput` so
    the editable field set (``name`` / ``summary`` / ``body`` /
    ``prompt_mode`` / ``tool_additions`` / ``tool_restrictions`` /
    ``trigger_keywords`` / ``tags`` / ``enabled``) is single-sourced with
    the agent tool schema. ``user_id`` / ``agent_id`` are NOT fields here
    and ``extra='forbid'`` rejects any attempt to send them -- the router
    derives identity from the authenticated principal.
    """

    model_config = ConfigDict(extra="forbid")


class UpdateSkillRequest(BaseModel):
    """Request body for ``PATCH /skills/{id}``.

    Mirrors the editable field set of
    :class:`~threetears.agent.skills.tools.SkillUpdateInput` (all
    optional) but is standalone rather than a subclass: the tool schema
    carries a *required* ``skill_id`` and pydantic/mypy forbid a subclass
    from widening that to optional. On the REST surface the id is a path
    parameter, not part of the body, so it's dropped entirely. The
    remaining fields are kept in lock-step with ``SkillUpdateInput`` --
    a parity test asserts the two field sets stay aligned.
    ``extra='forbid'`` rejects ``user_id`` / ``agent_id`` (router-derived
    identity) and ``skill_id`` (path-only).
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    summary: str | None = None
    body: str | None = None
    prompt_mode: Literal["additive", "replace"] | None = None
    tool_additions: list[str] | None = None
    tool_restrictions: list[str] | None = None
    trigger_keywords: str | None = None
    tags: list[str] | None = None
    enabled: bool | None = None
