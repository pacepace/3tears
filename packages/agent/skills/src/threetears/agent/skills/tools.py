"""Skills tools -- LangChain tools for skill CRUD + invoke + introspect.

Provides seven loader factories that mint :class:`langchain_core.tools.BaseTool`
instances bound to a specific ``(agent_id, user_id)`` actor pair and the
agent-skills Collections (``AgentSkillCollection`` /
``AgentSkillInvocationCollection``). The consumer's personality
graph wires these into the per-conversation tool surface.

Factories:

- :func:`load_skill_create_tool` -- ``skill_create``
- :func:`load_skill_list_tool` -- ``skill_list`` (UNIONs prose-skill rows
  with skill-eligible tool namespaces from the registry)
- :func:`load_skill_get_tool` -- ``skill_get``
- :func:`load_skill_update_tool` -- ``skill_update``
- :func:`load_skill_delete_tool` -- ``skill_delete``
- :func:`load_skill_invoke_tool` -- ``skill_invoke``
- :func:`load_skill_introspect_tool` -- ``skill_introspect``
- :func:`load_skill_report_outcome_tool` -- ``skill_report_outcome``
- :func:`load_skill_history_tool` -- ``skill_history``

Every factory closes over ``(agent_id, user_id)`` and the Collections so
the LLM never sees identity fields in the input schema -- the runtime
provides them. ``skill_invoke``'s first-invoke-wins-per-turn rule is
enforced by a consumer-supplied state probe / setter so the package
stays decoupled from any specific LangGraph state shape.

Spec ref: ``docs/agent-skills/shard-02-agent-tools.md`` (requirements
SK-09 .. SK-17) + the skills placement spec sections 1.5 /
1.7 / 1.8 / 1.10.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from typing import Any, Protocol, get_args
from uuid import UUID

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, model_validator
from uuid_utils import uuid7

from threetears.agent.skills.collections import (
    AgentSkillCollection,
    AgentSkillInvocationCollection,
    merged_skill_shape,
    skill_shape_error,
)
from threetears.agent.skills.entities import AgentSkillEntity, AgentSkillInvocationEntity
from threetears.agent.skills.rendering import render_skill_body_block
from threetears.agent.skills.types import PromptMode, SkillKindFilter, SkillOutcome
from threetears.core.utils.local_time import render_local_time, resolve_zone
from threetears.observe.text_block import one_line, under
from threetears.observe import contain, get_logger

__all__ = [
    "ActiveSkillProbe",
    "ActiveSkillSetter",
    "BindableTools",
    "ConversationIdResolver",
    "SkillCreateInput",
    "TOOL_SKILLS_NOT_OFFERED",
    "ToolSkillCreateInput",
    "ToolSkillUpdateInput",
    "SkillDeleteInput",
    "SkillEligibleTool",
    "SkillGetInput",
    "SkillHistoryInput",
    "SkillIntrospectInput",
    "SkillInvokeInput",
    "SkillListInput",
    "SkillRegistryClient",
    "SkillReportOutcomeInput",
    "SkillToolIntrospect",
    "SkillUpdateInput",
    "load_skill_create_tool",
    "load_skill_delete_tool",
    "load_skill_get_tool",
    "load_skill_history_tool",
    "load_skill_introspect_tool",
    "load_skill_invoke_tool",
    "load_skill_list_tool",
    "load_skill_report_outcome_tool",
    "load_skill_update_tool",
    "validate_skill_arguments",
]


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Limits + validators
# ---------------------------------------------------------------------------


# Per shard-02 SK-10: name 1-128 chars (letters/digits/hyphens/spaces/
# underscores). Single source of truth referenced by both ``skill_create``
# and ``skill_update``.
NAME_MIN_LEN = 1
NAME_MAX_LEN = 128
SUMMARY_MIN_LEN = 1
SUMMARY_MAX_LEN = 256
BODY_MAX_BYTES = 32 * 1024  # 32 KB hard cap (Implementation note 2)
ARGUMENTS_MAX_BYTES = 32 * 1024  # same cap as a body: a tool skill's whole payload

# What skill_create / skill_update say when a tool-call skill is asked for and
# the consumer does not offer them: it cannot run one yet.
TOOL_SKILLS_NOT_OFFERED = "a skill cannot be one tool call here yet; give it steps in body"
TRIGGER_KEYWORDS_MAX_LEN = 512
TAGS_MAX_ENTRIES = 8
TOOL_LIST_MAX_ENTRIES = 32
DEFAULT_MAX_PROSE_SKILLS_PER_USER = 200

_NAME_RE = re.compile(r"^[A-Za-z0-9 _\-]+$")

# derived from the canonical alias, never re-spelled: a value added to
# ``PromptMode`` must reach this runtime check without a second edit.
_VALID_PROMPT_MODES: frozenset[str] = frozenset(get_args(PromptMode))


# ---------------------------------------------------------------------------
# Consumer-supplied wiring types
# ---------------------------------------------------------------------------


# Resolver hook the consumer wires so ``skill_invoke`` records the
# invocation against the active conversation rather than the conversation
# the factory was first minted under (multiplex violation otherwise).
# Mirrors ``threetears.agent.memory.tools.load_memory_add_tool``'s
# ``context_resolver`` shape.
ConversationIdResolver = Callable[[], UUID | None]


# Probe/setter pair the consumer wires for ``_active_skill_id`` state
# (PLACEMENT §1.10). The probe returns the current active skill UUID
# for the in-flight turn (``None`` when none); the setter records that
# this turn now has the given skill active. ``skill_invoke`` enforces
# first-invoke-wins by reading the probe and refusing when it returns
# a non-``None`` value.
ActiveSkillProbe = Callable[[], UUID | None]
ActiveSkillSetter = Callable[[UUID], None]

# The tools the turn can bind, by name, for ``skill_invoke`` to name only those.
# ``None`` from the probe (or no probe) means the consumer cannot say, and then
# the result names no tool at all: a skill naming a tool the agent does not hold
# sends it calling one that is not there.
BindableTools = Callable[[], Collection[str] | None]


@dataclass(frozen=True)
class SkillEligibleTool:
    """Uniform shape for a tool-skill catalog entry.

    Returned by :meth:`SkillRegistryClient.list_skill_eligible_tools`
    and consumed by ``skill_list`` (which UNIONs prose-skill rows with
    these) and ``skill_introspect`` (which renders the tool-skill
    shape from §1.8 of PLACEMENT).

    :ivar mcp_name: canonical tool name (``mcp_name`` from the
        ``TearsTool`` registration, NOT the sanitized namespace name)
    :ivar summary: one-line tool description shown in the catalog
    """

    mcp_name: str
    summary: str


@dataclass(frozen=True)
class SkillToolIntrospect:
    """Detailed introspection payload for a tool-skill.

    Returned by :meth:`SkillRegistryClient.get_tool_introspect`. Rendering
    follows the minimal-token shape from PLACEMENT §1.8: name + summary +
    args + example. No operational metadata.

    :ivar mcp_name: canonical tool name
    :ivar summary: one-line description
    :ivar args: ordered ``{arg_name: "<type>  # <description>"}`` lines
        for the introspect output. Insertion order is preserved
    :ivar example: ordered ``{arg_name: example_value}`` payload
    """

    mcp_name: str
    summary: str
    args: dict[str, str]
    example: dict[str, Any]


class SkillRegistryClient(Protocol):
    """Thin Protocol the consumer implements over its tool registry + ACL.

    Three methods, each kept minimal so the consumer's binding is a
    handful of lines over its existing ``NamespaceCollection`` + ACL
    cache + in-process tool registry. Tests substitute a fake.
    """

    async def acl_permits(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        tool_name: str,
    ) -> bool:
        """Return ``True`` iff ``(user_id, agent_id)`` may call ``tool_name``.

        Used by ``skill_create`` / ``skill_update`` to fail-fast when
        the LLM tries to put a tool into ``tool_additions`` or
        ``tool_restrictions`` that the user lacks ACL for. The
        re-validation at LOAD time (PLACEMENT §1.10) remains the source
        of truth; this check is a UX courtesy.

        :param user_id: caller's user UUID
        :ptype user_id: UUID
        :param agent_id: caller's agent UUID
        :ptype agent_id: UUID
        :param tool_name: tool ``mcp_name``
        :ptype tool_name: str
        :return: ``True`` if ACL grants ``tool.call`` on the named tool
        :rtype: bool
        """
        ...

    async def list_skill_eligible_tools(
        self,
        *,
        actor_user_id: UUID,
        actor_agent_id: UUID,
    ) -> list[SkillEligibleTool]:
        """Return the ACL-permitted subset of tools with ``skill_eligible=True``.

        The consumer's typical implementation calls
        :meth:`NamespaceCollection.list_skill_eligible_tool_namespaces`
        and maps each ``NamespaceEntity`` into a
        :class:`SkillEligibleTool` (reading ``mcp_name`` from the
        namespace metadata + the human description from the in-process
        ``TearsTool`` registry).

        :param actor_user_id: caller's user UUID
        :ptype actor_user_id: UUID
        :param actor_agent_id: caller's agent UUID
        :ptype actor_agent_id: UUID
        :return: list of skill-eligible tools (empty when none match)
        :rtype: list[SkillEligibleTool]
        """
        ...

    async def get_tool_introspect(
        self,
        *,
        actor_user_id: UUID,
        actor_agent_id: UUID,
        mcp_name: str,
    ) -> SkillToolIntrospect | None:
        """Return introspection payload for one tool, or ``None`` if absent.

        ``None`` covers two cases that look the same to the agent:
        the tool isn't registered, or the actor lacks ACL for it. Both
        collapse into "you can't see this tool"; ``skill_introspect``
        surfaces a "not found" message in either case.

        :param actor_user_id: caller's user UUID
        :ptype actor_user_id: UUID
        :param actor_agent_id: caller's agent UUID
        :ptype actor_agent_id: UUID
        :param mcp_name: tool ``mcp_name``
        :ptype mcp_name: str
        :return: introspection payload or ``None``
        :rtype: SkillToolIntrospect | None
        """
        ...


# ---------------------------------------------------------------------------
# Pydantic input schemas
# ---------------------------------------------------------------------------


class SkillCreateInput(BaseModel):
    """Input schema for the ``skill_create`` tool."""

    name: str = Field(description="A short name no other skill has, 1-128 characters.")
    summary: str = Field(description="One line saying what it is for, shown by skill_list.")
    body: str | None = Field(
        default=None,
        description="Optional. The steps to follow, in plain words, up to 32KB.",
    )
    prompt_mode: PromptMode = Field(
        default="additive",
        description="'additive' adds the steps to your instructions; 'replace' uses them instead of your instructions.",
    )
    tool_additions: list[str] = Field(
        default_factory=list,
        description="Names of tools to give you while the skill is in use.",
    )
    tool_restrictions: list[str] = Field(
        default_factory=list,
        description="Names of tools to take away while the skill is in use.",
    )
    trigger_keywords: str = Field(
        default="",
        description="Words skill_list can find it by. They do not start the skill.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Up to 8 labels to sort skills by.",
    )
    enabled: bool = Field(default=True)


class ToolSkillCreateInput(SkillCreateInput):
    """``skill_create``'s input when the consumer offers tool-call skills.

    Offered only when the consumer can run one (``offer_tool_skills``):
    a skill the agent can save but that does nothing when used is worse
    than no such skill.
    """

    tool: str | None = Field(
        default=None,
        description="Optional. Instead of steps, the name of one tool this skill calls. Not with body.",
    )
    arguments: dict[str, Any] | None = Field(
        default=None,
        description="Optional. The arguments that tool is called with, as named values. Only with tool.",
    )


class SkillListInput(BaseModel):
    """Input schema for the ``skill_list`` tool."""

    query: str | None = Field(
        default=None,
        description="Optional words to match.",
    )
    kind_filter: SkillKindFilter = Field(
        default="all",
        description="Show skills you wrote, skills that are tools, or both.",
    )
    tag_filter: str | None = Field(
        default=None,
        description="Optional label. Only skills you wrote have labels, so this leaves out skills that are tools.",
    )
    enabled_only: bool = Field(
        default=True,
        description="Leave out skills you turned off.",
    )
    limit: int = Field(default=20, ge=1, le=200, description="Most skills to list.")


class SkillGetInput(BaseModel):
    """Input schema for the ``skill_get`` tool."""

    skill_id: str = Field(description="The [skill:<id>] from skill_list or skill_create, or the skill's name.")


class SkillUpdateInput(BaseModel):
    """Input schema for the ``skill_update`` tool.

    All non-``skill_id`` fields are optional; only fields the LLM passes
    get applied. Each pass-through carries the same validation rules
    ``skill_create`` enforces (name regex, bounded sizes, ACL on tool
    lists).
    """

    skill_id: str = Field(description="The [skill:<id>] or name of the skill to change.")
    name: str | None = None
    summary: str | None = None
    body: str | None = Field(default=None, description="New steps. An empty string removes them.")
    prompt_mode: PromptMode | None = None
    tool_additions: list[str] | None = None
    tool_restrictions: list[str] | None = None
    trigger_keywords: str | None = None
    tags: list[str] | None = None
    enabled: bool | None = None


class ToolSkillUpdateInput(SkillUpdateInput):
    """``skill_update``'s input when the consumer offers tool-call skills."""

    tool: str | None = Field(
        default=None,
        description="New tool to call. An empty string removes the tool and its arguments.",
    )
    arguments: dict[str, Any] | None = Field(default=None, description="New arguments for the tool.")


class SkillDeleteInput(BaseModel):
    """Input schema for the ``skill_delete`` tool."""

    skill_id: str = Field(description="The [skill:<id>] or name of the skill to delete.")


class SkillInvokeInput(BaseModel):
    """Input schema for the ``skill_invoke`` tool."""

    skill_id: str = Field(description="The [skill:<id>] or name of the skill to use now.")
    rationale: str | None = Field(
        default=None,
        description="Optional one line on why, kept with the record of this use.",
    )


class SkillHistoryInput(BaseModel):
    """Input schema for the ``skill_history`` tool."""

    skill_id: str = Field(description="The [skill:<id>] or name of the skill.")
    limit: int = Field(default=20, ge=1, le=100, description="Most uses to list, newest first.")
    offset: int = Field(default=0, ge=0, description="Uses to skip, for the next page.")


class SkillReportOutcomeInput(BaseModel):
    """Input schema for the ``skill_report_outcome`` tool."""

    outcome: SkillOutcome = Field(
        description="Whether the skill's task worked.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional one line on why.",
    )


class SkillIntrospectInput(BaseModel):
    """Input schema for the ``skill_introspect`` tool.

    Accepts either a skill name (prose-skill or tool-skill mcp_name) or
    the ``[skill:<id>]`` UUID form. Name resolution prefers prose-skill
    matches when both kinds collide (Implementation note 7).
    """

    name_or_id: str = Field(
        description="The skill's name or its [skill:<id>], for skills you wrote and skills that are tools.",
    )

    @model_validator(mode="after")
    def _non_empty(self) -> "SkillIntrospectInput":
        if not self.name_or_id or not self.name_or_id.strip():
            raise ValueError("name_or_id must be non-empty.")
        return self


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tool_error(tool_name: str, description: str) -> str:
    """Format the canonical ``[TOOL ERROR] <tool>: <description>`` string.

    Matches existing 3tears tool conventions (memory's ``_tool_error``
    helper uses a slightly different shape with an ``action`` slot;
    skills tools use a single descriptive sentence so the LLM gets the
    full context in one read).

    :param tool_name: tool name emitting the error
    :ptype tool_name: str
    :param description: error description
    :ptype description: str
    :return: formatted error string
    :rtype: str
    """
    return f"[TOOL ERROR] {tool_name}: {description}"


def _parse_skill_id(raw: str) -> UUID | None:
    """Parse ``[skill:<id>]`` or bare UUID into a :class:`UUID`.

    Tolerates either form so the LLM can pass the rendered tag back
    directly. Returns ``None`` on parse failure -- the caller surfaces
    a tool-error rather than raising.

    :param raw: raw skill id from the LLM
    :ptype raw: str
    :return: parsed UUID or ``None`` on failure
    :rtype: UUID | None
    """
    if not raw:
        return None
    candidate = raw.strip()
    if candidate.startswith("[skill:") and candidate.endswith("]"):
        candidate = candidate[len("[skill:") : -1].strip()
    try:
        return UUID(candidate)
    except ValueError:
        # Malformed UUID literal (typo, wrong format, etc.). The caller turns None into a
        # tool-error the model sees, so this only needs to be legible to an operator asking
        # why a skill lookup keeps missing.
        log.debug("skill id did not parse as a UUID", extra={"extra_data": {"candidate": candidate}})
        return None


def _validate_name(name: str) -> str | None:
    """Return an error message when ``name`` violates the SK-10 contract.

    :param name: candidate skill name
    :ptype name: str
    :return: error string or ``None`` when valid
    :rtype: str | None
    """
    if not isinstance(name, str):
        return "name must be a string"
    if len(name) < NAME_MIN_LEN:
        return f"name must be at least {NAME_MIN_LEN} character"
    if len(name) > NAME_MAX_LEN:
        return f"name must be at most {NAME_MAX_LEN} characters"
    if not _NAME_RE.fullmatch(name):
        return "name must match [A-Za-z0-9 _-]"
    return None


def _validate_summary(summary: str) -> str | None:
    """Return an error message when ``summary`` violates SK-10 caps.

    :param summary: candidate summary
    :ptype summary: str
    :return: error string or ``None``
    :rtype: str | None
    """
    if not isinstance(summary, str):
        return "summary must be a string"
    if len(summary) < SUMMARY_MIN_LEN:
        return f"summary must be at least {SUMMARY_MIN_LEN} character"
    if len(summary) > SUMMARY_MAX_LEN:
        return f"summary must be at most {SUMMARY_MAX_LEN} characters"
    return None


def _validate_body(body: str | None) -> str | None:
    """Return an error message when ``body`` exceeds the 32 KB cap.

    :param body: candidate body (markdown)
    :ptype body: str | None
    :return: error string or ``None``
    :rtype: str | None
    """
    if body is None:
        return None
    if not isinstance(body, str):
        return "body must be a string or null"
    # cap is byte-count of the UTF-8 encoding so multibyte text is
    # measured truthfully (a 32K-char limit would let UTF-8 grow the
    # payload up to 4x in pathological cases).
    if len(body.encode("utf-8")) > BODY_MAX_BYTES:
        return f"body exceeds {BODY_MAX_BYTES // 1024} KB cap"
    return None


def validate_skill_arguments(arguments: dict[str, Any] | None) -> str | None:
    """Return an error message when ``arguments`` exceeds the 32 KB cap.

    The shape rules (a JSON object, only with a tool) are
    :func:`~threetears.agent.skills.collections.skill_shape_error`'s; this is
    the size cap the tools add on top, measured on the JSON the row stores.

    :param arguments: candidate arguments
    :ptype arguments: dict[str, Any] | None
    :return: error string or ``None``
    :rtype: str | None
    """
    error: str | None = None
    if isinstance(arguments, dict):
        try:
            encoded = json.dumps(arguments)
        except (TypeError, ValueError) as exc:
            error = f"arguments must be plain JSON: {exc}"
        else:
            if len(encoded.encode("utf-8")) > ARGUMENTS_MAX_BYTES:
                error = f"arguments exceed {ARGUMENTS_MAX_BYTES // 1024} KB cap"
    return error


def _validate_trigger_keywords(value: str) -> str | None:
    """Return an error message when ``trigger_keywords`` is too long.

    :param value: candidate keywords
    :ptype value: str
    :return: error string or ``None``
    :rtype: str | None
    """
    if not isinstance(value, str):
        return "trigger_keywords must be a string"
    if len(value) > TRIGGER_KEYWORDS_MAX_LEN:
        return f"trigger_keywords must be at most {TRIGGER_KEYWORDS_MAX_LEN} characters"
    return None


def _validate_tags(tags: list[str]) -> str | None:
    """Return an error message when ``tags`` exceeds limits.

    :param tags: candidate tags list
    :ptype tags: list[str]
    :return: error string or ``None``
    :rtype: str | None
    """
    if not isinstance(tags, list):
        return "tags must be a list of strings"
    if len(tags) > TAGS_MAX_ENTRIES:
        return f"tags must contain at most {TAGS_MAX_ENTRIES} entries"
    for entry in tags:
        if not isinstance(entry, str):
            return "tags entries must all be strings"
    return None


def _validate_tool_list(field_name: str, values: list[str]) -> str | None:
    """Return an error message when a ``tool_additions`` / ``tool_restrictions`` list is malformed.

    :param field_name: field for error messaging
    :ptype field_name: str
    :param values: candidate list
    :ptype values: list[str]
    :return: error string or ``None``
    :rtype: str | None
    """
    if not isinstance(values, list):
        return f"{field_name} must be a list of tool names"
    if len(values) > TOOL_LIST_MAX_ENTRIES:
        return f"{field_name} must contain at most {TOOL_LIST_MAX_ENTRIES} entries"
    for entry in values:
        if not isinstance(entry, str) or not entry.strip():
            return f"{field_name} entries must all be non-empty strings"
    return None


def _at_least_one_payload(
    *,
    body: str | None,
    tool_additions: list[str],
    tool_restrictions: list[str],
    tool: str | None = None,
) -> bool:
    """Mirror the L3 CHECK constraint: ≥1 of body/tool/tool_additions/tool_restrictions.

    Empty strings count as no body (the DB constraint treats NULL and
    empty-string equivalently for "no payload"). Empty lists count as
    "no list".

    :param body: skill body or ``None``
    :ptype body: str | None
    :param tool: the tool a tool skill calls, or ``None``
    :ptype tool: str | None
    :param tool_additions: list of tool names
    :ptype tool_additions: list[str]
    :param tool_restrictions: list of tool names
    :ptype tool_restrictions: list[str]
    :return: ``True`` iff at least one payload is present
    :rtype: bool
    """
    if body is not None and body.strip():
        return True
    if tool is not None:
        return True
    if tool_additions:
        return True
    if tool_restrictions:
        return True
    return False


async def _check_tool_acl(
    *,
    registry: SkillRegistryClient,
    agent_id: UUID,
    user_id: UUID,
    tool_name: str,
    label: str,
) -> str | None:
    """Probe ACL for one tool name; return an error message on rejection.

    Wraps the registry call so the failure modes (registry raises,
    grant absent) collapse into one tool-error string for the caller.

    :param registry: consumer-supplied registry client
    :ptype registry: SkillRegistryClient
    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param tool_name: tool to check
    :ptype tool_name: str
    :param label: source field for the error message (e.g. ``"tool_additions"``)
    :ptype label: str
    :return: error message string or ``None`` on success
    :rtype: str | None
    """
    try:
        permitted = await registry.acl_permits(
            user_id=user_id,
            agent_id=agent_id,
            tool_name=tool_name,
        )
    except Exception as exc:  # pragma: no cover -- registry failure modes
        contain(exc, log=log, what="skill ACL probe raised", extra={"tool_name": tool_name, "error": str(exc)})
        return f"{label} ACL probe failed for tool {tool_name!r}: {exc}"
    if not permitted:
        return f"{label} entry {tool_name!r} not authorized for this user"
    return None


async def _resolve_skill_ref(
    collection: AgentSkillCollection, *, agent_id: UUID, user_id: UUID, raw: str
) -> UUID | None:
    """The skill a model named: its ``[skill:<id>]``, its bare id, or its name.

    Models name skills by name as often as by id; refusing a name sent the
    agent hunting for an id it could not see and reporting the skill broken.

    :param collection: three-tier skills collection
    :ptype collection: AgentSkillCollection
    :param agent_id: caller's partition column
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param raw: what the model passed
    :ptype raw: str
    :return: the skill's id, or ``None`` when neither an id nor a name matches
    :rtype: UUID | None
    """
    parsed = _parse_skill_id(raw)
    if parsed is not None:
        return parsed
    name = (raw or "").strip()
    if name.startswith("[skill:") and name.endswith("]"):
        name = name[len("[skill:") : -1].strip()
    if not name:
        return None
    entity = await collection.find_by_name_for_user(agent_id=agent_id, user_id=user_id, name=name)
    return entity.skill_id if entity is not None else None


async def _load_skill_for_user(
    *,
    collection: AgentSkillCollection,
    agent_id: UUID,
    user_id: UUID,
    skill_id: UUID,
) -> AgentSkillEntity | None:
    """Fetch a skill via the Collection and enforce ``user_id`` ownership.

    The Collection's primary-key lookup partitions by ``agent_id``
    only; per-user isolation is the tool layer's responsibility. This
    helper enforces it: a skill belonging to a different user is
    indistinguishable from "not found" so existence is never leaked
    across the user boundary (SK-13).

    :param collection: three-tier skills collection
    :ptype collection: AgentSkillCollection
    :param agent_id: caller's partition column
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param skill_id: skill UUID to load
    :ptype skill_id: UUID
    :return: entity owned by ``user_id`` or ``None``
    :rtype: AgentSkillEntity | None
    """
    entity = await collection.get((agent_id, skill_id))
    if entity is None:
        return None
    if entity.user_id != user_id:
        return None
    return entity


def _format_skill_summary(entity: AgentSkillEntity) -> str:
    """Render a one-line catalog entry for a prose-skill row.

    Shared by ``skill_create``, ``skill_update``, ``skill_list`` so the
    catalog rendering is consistent.
    """
    enabled_marker = "" if entity.enabled else " [disabled]"
    summary = one_line(entity.summary or "")
    return f"[skill:{entity.skill_id}] {entity.name}{enabled_marker}" + (f" -- {summary}" if summary else "")


def _skill_tool_lines(entity: AgentSkillEntity, bindable: Collection[str] | None) -> list[str]:
    """What a skill does to the agent's tools, naming only tools the turn can bind.

    A tool the skill adds is named when the turn holds it, and is bound from the
    agent's next step; one it does not hold is counted and not named, because a
    name it cannot call reads as one it can. A tool the skill puts away is
    named when the turn held it.

    :param entity: the skill
    :ptype entity: AgentSkillEntity
    :param bindable: the tools the turn can bind, or ``None`` when the consumer cannot say
    :ptype bindable: Collection[str] | None
    :return: the lines, none when the skill changes no tool
    :rtype: list[str]
    """
    additions = list(entity.tool_additions)
    restrictions = list(entity.tool_restrictions)
    if bindable is None:
        named = len(additions) + len(restrictions)
        return [f"It changes {named} of your tools; your next step shows which you hold."] if named else []
    lines: list[str] = []
    held = [name for name in additions if name in bindable]
    if held:
        lines.append(f"Tools it adds, yours from your next step: {', '.join(held)}.")
    missing = len(additions) - len(held)
    if missing:
        lines.append(f"{missing} more tool(s) it names cannot be used on this turn.")
    put_away = [name for name in restrictions if name in bindable]
    if put_away:
        lines.append(f"Tools it puts away until the turn ends: {', '.join(put_away)}.")
    return lines


def _render_skill_invoke_block(entity: AgentSkillEntity, bindable: Collection[str] | None = None) -> str:
    """The ``skill_invoke`` result: the skill in use, its tools as the turn holds them, and its steps whole.

    The steps are rendered as a wake renders them (:func:`render_skill_body_block`),
    so the agent reads the same block however the skill arrived.

    :param entity: the skill
    :ptype entity: AgentSkillEntity
    :param bindable: the tools the turn can bind, or ``None`` when the consumer cannot say
    :ptype bindable: Collection[str] | None
    :return: the result text
    :rtype: str
    """
    parts: list[str] = [
        f"[ACTIVE SKILL: {entity.name}] in use for the rest of this turn.",
        f"prompt_mode: {entity.prompt_mode}",
        *_skill_tool_lines(entity, bindable),
        "",
        render_skill_body_block(entity) or f'The skill "{entity.name}" has no written steps.',
    ]
    return "\n".join(parts)


def _times_used(count: int) -> str:
    return "once" if count == 1 else f"{count} times"


def _render_skill_usage(entity: AgentSkillEntity, zone: tzinfo | None) -> list[str]:
    """How often a skill was used and how it went, times in the reader's zone.

    :param entity: the skill
    :ptype entity: AgentSkillEntity
    :param zone: the reader's zone, or ``None`` for UTC
    :ptype zone: tzinfo | None
    :return: the lines
    :rtype: list[str]
    """
    used = int(entity.use_count or 0)
    last = entity.last_used_at
    lines = [
        f"used: {_times_used(used)}" + (f", last {render_local_time(last, zone)}" if used and last else "")
        if used
        else "used: never",
    ]
    worked, failed = int(entity.success_count or 0), int(entity.failure_count or 0)
    outcome = f"reported: worked {worked}, failed {failed}"
    if failed and entity.last_failure_at is not None:
        outcome += f" (last failure {render_local_time(entity.last_failure_at, zone)})"
    lines.append(outcome)
    lines.append(f"written: {render_local_time(entity.date_created, zone)}")
    lines.append(f"last changed: {render_local_time(entity.date_updated, zone)}")
    if used:
        lines.append("skill_history lists each use.")
    return lines


#: The fields ``skill_update`` can change, in the order its result names them.
_EDITABLE_FIELDS: tuple[str, ...] = (
    "name",
    "summary",
    "body",
    "tool",
    "arguments",
    "prompt_mode",
    "tool_additions",
    "tool_restrictions",
    "trigger_keywords",
    "tags",
    "enabled",
)

#: A text this long or shorter is shown whole on both sides of a change; a longer
#: one by its length and the first line that differs, so a 32 KB body edit does not
#: print the body twice.
_SHOWN_WHOLE_CHARS = 200


def _skill_fields(entity: AgentSkillEntity) -> dict[str, Any]:
    """The editable fields of a skill as they stand, copied so a later edit cannot change them.

    :param entity: the skill
    :ptype entity: AgentSkillEntity
    :return: field name to value
    :rtype: dict[str, Any]
    """
    fields: dict[str, Any] = {}
    for name in _EDITABLE_FIELDS:
        value = getattr(entity, name)
        if isinstance(value, list):
            value = list(value)
        elif isinstance(value, dict):
            value = json.loads(json.dumps(value))
        fields[name] = value
    return fields


def _shown(value: Any) -> str:
    """One side of a change, as the agent reads it."""
    if value is None or value == "":
        return "(none)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]" if value else "(none)"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return json.dumps(str(value), ensure_ascii=False)


def _first_difference(before: str, after: str) -> str:
    """Where two long texts first part, by line: the line before and the line now."""
    old_lines, new_lines = before.splitlines(), after.splitlines()
    for index in range(max(len(old_lines), len(new_lines))):
        old = old_lines[index] if index < len(old_lines) else None
        new = new_lines[index] if index < len(new_lines) else None
        if old != new:
            was = _shown(old[:_SHOWN_WHOLE_CHARS] if old is not None else None)
            now = _shown(new[:_SHOWN_WHOLE_CHARS] if new is not None else None)
            return f"first change at line {index + 1}: {was} -> {now}"
    return "the lines are the same; only the spacing differs"


def _change_line(field: str, before: Any, after: Any) -> str:
    """One changed field, before -> after.

    :param field: the field
    :ptype field: str
    :param before: its value before
    :ptype before: Any
    :param after: its value now
    :ptype after: Any
    :return: the line
    :rtype: str
    """
    long_text = isinstance(before, str) and isinstance(after, str) and max(len(before), len(after)) > _SHOWN_WHOLE_CHARS
    long_text = long_text or (
        (isinstance(before, str) and len(before) > _SHOWN_WHOLE_CHARS and after is None)
        or (isinstance(after, str) and len(after) > _SHOWN_WHOLE_CHARS and before is None)
    )
    if long_text:
        old, new = before or "", after or ""
        return f"- {field}: {len(old):,} -> {len(new):,} characters; {_first_difference(old, new)}"
    return f"- {field}: {_shown(before)} -> {_shown(after)}"


def _render_prose_introspect(entity: AgentSkillEntity) -> str:
    """Render the prose-skill introspect payload per PLACEMENT §1.8.

    :param entity: prose skill entity
    :ptype entity: AgentSkillEntity
    :return: formatted introspect text block
    :rtype: str
    """
    body = entity.body or ""
    # the stored body exactly, every line indented: skill_update replaces it whole, so a body read
    # here, edited and written back must keep its blank lines and its own indentation
    body_lines = ["  " + line if line.strip() else "" for line in body.splitlines()]
    body_block = "\n".join(body_lines) if body.strip() else "  (no body)"
    lines: list[str] = [
        f"[skill:{entity.skill_id}]",
        f"name: {entity.name}",
        "kind: prose",
        f"summary: {entity.summary}",
        f"prompt_mode: {entity.prompt_mode}",
        "body: |",
        body_block,
    ]
    if entity.tool is not None:
        lines.append(f"tool: {entity.tool}")
        lines.append(f"arguments: {json.dumps(entity.arguments or {}, sort_keys=True)}")
    lines += [
        f"tool_additions: {list(entity.tool_additions)}",
        f"tool_restrictions: {list(entity.tool_restrictions)}",
        f"triggers: {entity.trigger_keywords}",
        f"tags: {list(entity.tags)}",
        f"enabled: {entity.enabled}",
    ]
    return "\n".join(lines)


def _render_tool_introspect(payload: SkillToolIntrospect) -> str:
    """Render the tool-skill introspect payload per PLACEMENT §1.8.

    :param payload: tool introspect payload from the registry client
    :ptype payload: SkillToolIntrospect
    :return: formatted introspect text block
    :rtype: str
    """
    lines: list[str] = [
        f"[skill:{payload.mcp_name}]",
        f"name: {payload.mcp_name}",
        "kind: tool",
        f"summary: {payload.summary}",
        "args:",
    ]
    if payload.args:
        for arg_name, arg_desc in payload.args.items():
            lines.extend(under(str(arg_desc), "    ", first=f"  {arg_name}: "))
    else:
        lines.append("  (no arguments)")
    lines.append("example:")
    if payload.example:
        for arg_name, value in payload.example.items():
            lines.append(f"  {arg_name}: {value!r}")
    else:
        lines.append("  (no example)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# skill_create
# ---------------------------------------------------------------------------


def load_skill_create_tool(
    *,
    agent_id: UUID,
    user_id: UUID,
    skills_collection: AgentSkillCollection,
    registry: SkillRegistryClient,
    max_prose_skills_per_user: int = DEFAULT_MAX_PROSE_SKILLS_PER_USER,
    offer_tool_skills: bool = False,
) -> list[BaseTool]:
    """Build a ``skill_create`` tool bound to ``(agent_id, user_id)``.

    ``offer_tool_skills`` adds ``tool`` and ``arguments`` to the input, so
    the agent can save a skill that is one tool call. Leave it off until
    the consumer runs such a skill: without it the agent never sees the
    fields, and a call that passes them anyway is refused.

    Validates payload (name regex, bounded sizes, ACL on tool lists),
    enforces the 200-prose-skill cap (SK-14), enforces the at-least-one-
    payload CHECK (SK-10), and persists via the Collection.

    :param agent_id: caller's agent UUID (partition column)
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param skills_collection: three-tier skills collection
    :ptype skills_collection: AgentSkillCollection
    :param registry: consumer-supplied registry client for ACL probes
    :ptype registry: SkillRegistryClient
    :param max_prose_skills_per_user: cap on prose skills per
        ``(agent_id, user_id)``; default 200 (SK-14)
    :ptype max_prose_skills_per_user: int
    :param offer_tool_skills: whether the agent may save a tool-call skill
    :ptype offer_tool_skills: bool
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("skill_create", args_schema=ToolSkillCreateInput if offer_tool_skills else SkillCreateInput)
    async def skill_create(
        name: str,
        summary: str,
        body: str | None = None,
        tool: str | None = None,
        arguments: dict[str, Any] | None = None,
        prompt_mode: PromptMode = "additive",
        tool_additions: list[str] | None = None,
        tool_restrictions: list[str] | None = None,
        trigger_keywords: str = "",
        tags: list[str] | None = None,
        enabled: bool = True,
    ) -> str:
        """Create a new skill bound to the calling actor: steps in a body, or one tool call."""
        additions = list(tool_additions or [])
        restrictions = list(tool_restrictions or [])
        tag_values = list(tags or [])
        if not offer_tool_skills and (tool is not None or arguments is not None):
            return _tool_error("skill_create", TOOL_SKILLS_NOT_OFFERED)
        # an empty body or tool is none at all, as it is when stored.
        stored_body = body if body else None
        stored_tool = tool if tool else None

        for err in (
            _validate_name(name),
            _validate_summary(summary),
            _validate_body(body),
            skill_shape_error(body=stored_body, tool=stored_tool, arguments=arguments),
            validate_skill_arguments(arguments),
            _validate_trigger_keywords(trigger_keywords),
            _validate_tags(tag_values),
            _validate_tool_list("tool_additions", additions),
            _validate_tool_list("tool_restrictions", restrictions),
        ):
            if err is not None:
                return _tool_error("skill_create", err)

        if prompt_mode not in _VALID_PROMPT_MODES:
            return _tool_error(
                "skill_create",
                f"prompt_mode must be 'additive' or 'replace'; got {prompt_mode!r}",
            )

        if not _at_least_one_payload(
            body=body,
            tool=stored_tool,
            tool_additions=additions,
            tool_restrictions=restrictions,
        ):
            return _tool_error(
                "skill_create",
                "at least one of body, tool, tool_additions, or tool_restrictions must be non-empty",
            )

        existing = await skills_collection.find_by_name_for_user(
            agent_id=agent_id,
            user_id=user_id,
            name=name,
        )
        if existing is not None:
            return _tool_error(
                "skill_create",
                f"a skill named {name!r} already exists (use skill_update to edit it)",
            )

        count = await skills_collection.count_for_user(
            agent_id=agent_id,
            user_id=user_id,
            enabled_only=False,
        )
        if count >= max_prose_skills_per_user:
            return _tool_error(
                "skill_create",
                f"max {max_prose_skills_per_user} prose skills per user; delete or disable some first",
            )

        if stored_tool is not None:
            err = await _check_tool_acl(
                registry=registry,
                agent_id=agent_id,
                user_id=user_id,
                tool_name=stored_tool,
                label="tool",
            )
            if err is not None:
                return _tool_error("skill_create", err)

        for entry in additions:
            err = await _check_tool_acl(
                registry=registry,
                agent_id=agent_id,
                user_id=user_id,
                tool_name=entry,
                label="tool_additions",
            )
            if err is not None:
                return _tool_error("skill_create", err)

        for entry in restrictions:
            err = await _check_tool_acl(
                registry=registry,
                agent_id=agent_id,
                user_id=user_id,
                tool_name=entry,
                label="tool_restrictions",
            )
            if err is not None:
                return _tool_error("skill_create", err)

        now = datetime.now(UTC)
        skill_id = UUID(str(uuid7()))
        data: dict[str, Any] = {
            "skill_id": skill_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "name": name,
            "summary": summary,
            "body": stored_body,
            "tool": stored_tool,
            "arguments": dict(arguments) if arguments is not None else None,
            "prompt_mode": prompt_mode,
            "tool_additions": additions,
            "tool_restrictions": restrictions,
            "trigger_keywords": trigger_keywords,
            "tags": tag_values,
            "source": "manual",
            "enabled": enabled,
            "use_count": 0,
            "last_used_at": None,
            "success_count": 0,
            "failure_count": 0,
            "last_failure_at": None,
            "date_created": now,
            "date_updated": now,
        }
        entity = skills_collection.create(data)
        try:
            await skills_collection.save_entity(entity)
        except Exception as exc:
            contain(exc, log=log, what="skill_create persist failed", extra={"name": name, "error": str(exc)})
            return _tool_error("skill_create", f"persist failed: {exc}")

        log.info(
            "skill_create persisted",
            extra={
                "extra_data": {
                    "skill_id": str(skill_id),
                    "name": name,
                    "prompt_mode": prompt_mode,
                    "tool": stored_tool,
                    "tool_additions_count": len(additions),
                    "tool_restrictions_count": len(restrictions),
                }
            },
        )
        return _format_skill_summary(entity)

    skill_create.description = (
        "Save a way of doing something as a skill you can use again. Give it steps in body, "
        "tools in tool_additions or tool_restrictions, or both."
        + (" Or make it one tool call: tool and arguments, with no body." if offer_tool_skills else "")
        + "\n"
        f"Returns [skill:<id>]. You can keep up to {max_prose_skills_per_user} skills you wrote."
    )

    return [skill_create]


# ---------------------------------------------------------------------------
# skill_list
# ---------------------------------------------------------------------------


def load_skill_list_tool(
    *,
    agent_id: UUID,
    user_id: UUID,
    skills_collection: AgentSkillCollection,
    registry: SkillRegistryClient,
) -> list[BaseTool]:
    """Build a ``skill_list`` tool that UNIONs prose + tool skills.

    Prose-skill rows from :class:`AgentSkillCollection` are merged with
    skill-eligible tool namespaces from the registry. Both surfaces use
    the same ``[skill:<id>]`` / ``name`` / ``summary`` / ``kind``
    discriminator (SK-16).

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param skills_collection: three-tier skills collection
    :ptype skills_collection: AgentSkillCollection
    :param registry: consumer-supplied registry client
    :ptype registry: SkillRegistryClient
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("skill_list", args_schema=SkillListInput)
    async def skill_list(
        query: str | None = None,
        kind_filter: SkillKindFilter = "all",
        tag_filter: str | None = None,
        enabled_only: bool = True,
        limit: int = 20,
    ) -> str:
        """List skills available to the caller (prose + tool UNION)."""
        # Fetch tool-skills FIRST when they're in scope so we can reserve
        # a slice of ``limit`` for them. Without this, prose alone can
        # saturate ``limit`` and silently hide every tool-skill from the
        # UNION (SK-16 discoverability bug).
        tool_entries: list[dict[str, Any]] = []
        if kind_filter in {"all", "tool"}:
            try:
                tool_rows = await registry.list_skill_eligible_tools(
                    actor_user_id=user_id,
                    actor_agent_id=agent_id,
                )
            except Exception as exc:
                contain(exc, log=log, what="skill_list: tool list failed")
                return _tool_error("skill_list", f"tool list failed: {exc}")
            needle = (query or "").lower().strip() if query else None
            for trow in tool_rows:
                # Tag filter is prose-only -- a tool-skill has no tags,
                # so passing one implicitly filters tools out unless
                # ``kind_filter='all'`` is paired with no tag filter.
                if tag_filter is not None:
                    continue
                if needle:
                    haystack = f"{trow.mcp_name} {trow.summary}".lower()
                    if needle not in haystack:
                        continue
                tool_entries.append(
                    {
                        "skill_id": trow.mcp_name,
                        "name": trow.mcp_name,
                        "summary": trow.summary,
                        "kind": "tool",
                        "enabled": True,
                    }
                )

        # Reserve up to ``limit // 2`` slots for tool-skills when any
        # exist; the rest is the prose query cap. This guarantees at
        # least some tool-skills surface when prose would otherwise
        # saturate the limit, while still respecting the caller's cap.
        tool_reserve = min(len(tool_entries), limit // 2)
        prose_cap = max(limit - tool_reserve, 1)

        prose_entries: list[dict[str, Any]] = []
        if kind_filter in {"all", "prose"}:
            try:
                rows = await skills_collection.list_for_user(
                    agent_id=agent_id,
                    user_id=user_id,
                    enabled_only=enabled_only,
                    tag_filter=[tag_filter] if tag_filter else None,
                    query=query,
                    limit=prose_cap,
                    offset=0,
                )
            except Exception as exc:
                contain(exc, log=log, what="skill_list: prose list failed")
                return _tool_error("skill_list", f"prose list failed: {exc}")
            for entity in rows:
                prose_entries.append(
                    {
                        "skill_id": str(entity.skill_id),
                        "name": entity.name,
                        "summary": entity.summary,
                        "kind": "prose",
                        "enabled": entity.enabled,
                    }
                )

        # Combine, prose-first by recency (the Collection already
        # ordered them); tool-skills sorted alphabetically so the
        # catalog has a stable shape. Final ``[:limit]`` is defensive --
        # the per-side caps above already keep the combined length
        # within ``limit``.
        combined: list[dict[str, Any]] = prose_entries + sorted(
            tool_entries,
            key=lambda e: e["name"].lower(),
        )
        combined = combined[:limit]

        if not combined:
            return "No skills available."

        lines: list[str] = [f"Found {len(combined)} skills:"]
        for entry in combined:
            marker = "" if entry["enabled"] else " [disabled]"
            summary = one_line(str(entry["summary"] or ""))
            lines.append(
                f"- [skill:{entry['skill_id']}] {entry['name']}{marker} (kind={entry['kind']})"
                + (f" -- {summary}" if summary else "")
            )
        return "\n".join(lines)

    skill_list.description = (
        "List the skills you can use: the ones you wrote ('prose') and tools that work as "
        "skills ('tool'). Returns id, name, summary and kind. skill_introspect shows one in full."
    )

    return [skill_list]


# ---------------------------------------------------------------------------
# skill_get
# ---------------------------------------------------------------------------


def load_skill_get_tool(
    *,
    agent_id: UUID,
    user_id: UUID,
    skills_collection: AgentSkillCollection,
    tz: tzinfo | str | None = None,
) -> list[BaseTool]:
    """Build a ``skill_get`` tool that reads one prose-skill by id, whole.

    Shows everything the skill is -- its steps, settings and tools -- and how
    it has gone: how often it was used and when last, how often it worked and
    failed, when it was written and last changed. ``skill_introspect`` shows
    the skill without its use, for choosing one.

    Cross-user isolation: a skill belonging to another user surfaces as
    "not found" (SK-13).

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param skills_collection: three-tier skills collection
    :ptype skills_collection: AgentSkillCollection
    :param tz: the zone the times are said in; ``None`` says them in UTC
    :ptype tz: tzinfo | str | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    zone = resolve_zone(tz)

    @tool("skill_get", args_schema=SkillGetInput)
    async def skill_get(skill_id: str) -> str:
        """Read full body + metadata for one prose-skill."""
        parsed = await _resolve_skill_ref(skills_collection, agent_id=agent_id, user_id=user_id, raw=skill_id)
        if parsed is None:
            return _tool_error("skill_get", f"no skill has the id or name {skill_id!r}")

        entity = await _load_skill_for_user(
            collection=skills_collection,
            agent_id=agent_id,
            user_id=user_id,
            skill_id=parsed,
        )
        if entity is None:
            return _tool_error("skill_get", "skill not found")
        return "\n".join([_render_prose_introspect(entity), *_render_skill_usage(entity, zone)])

    skill_get.description = (
        "Read a skill you wrote, whole: its steps, settings and tools, how often it was used and how it went. "
        "Use it before skill_update."
    )

    return [skill_get]


# ---------------------------------------------------------------------------
# skill_update
# ---------------------------------------------------------------------------


def load_skill_update_tool(
    *,
    agent_id: UUID,
    user_id: UUID,
    skills_collection: AgentSkillCollection,
    registry: SkillRegistryClient,
    offer_tool_skills: bool = False,
) -> list[BaseTool]:
    """Build a ``skill_update`` tool with partial-update semantics.

    Only fields the LLM passes get applied; the at-least-one-payload
    check + ACL re-validation run on the merged shape (SK-09 / SK-11).
    ``offer_tool_skills`` gates ``tool`` and ``arguments`` as it does on
    :func:`load_skill_create_tool`.

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param skills_collection: three-tier skills collection
    :ptype skills_collection: AgentSkillCollection
    :param registry: consumer-supplied registry client for ACL probes
    :ptype registry: SkillRegistryClient
    :param offer_tool_skills: whether the agent may make a skill a tool call
    :ptype offer_tool_skills: bool
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("skill_update", args_schema=ToolSkillUpdateInput if offer_tool_skills else SkillUpdateInput)
    async def skill_update(
        skill_id: str,
        name: str | None = None,
        summary: str | None = None,
        body: str | None = None,
        tool: str | None = None,
        arguments: dict[str, Any] | None = None,
        prompt_mode: PromptMode | None = None,
        tool_additions: list[str] | None = None,
        tool_restrictions: list[str] | None = None,
        trigger_keywords: str | None = None,
        tags: list[str] | None = None,
        enabled: bool | None = None,
    ) -> str:
        """Edit a skill in place."""
        if not offer_tool_skills and (tool is not None or arguments is not None):
            return _tool_error("skill_update", TOOL_SKILLS_NOT_OFFERED)
        parsed = await _resolve_skill_ref(skills_collection, agent_id=agent_id, user_id=user_id, raw=skill_id)
        if parsed is None:
            return _tool_error("skill_update", f"no skill has the id or name {skill_id!r}")

        entity = await _load_skill_for_user(
            collection=skills_collection,
            agent_id=agent_id,
            user_id=user_id,
            skill_id=parsed,
        )
        if entity is None:
            return _tool_error("skill_update", "skill not found")

        # Per-field validation. Partial: only validate fields the
        # caller supplied; absent fields keep their existing value.
        validation_errors: list[str | None] = []
        if name is not None:
            validation_errors.append(_validate_name(name))
        if summary is not None:
            validation_errors.append(_validate_summary(summary))
        if body is not None:
            validation_errors.append(_validate_body(body))
        if arguments is not None:
            validation_errors.append(validate_skill_arguments(arguments))
        if trigger_keywords is not None:
            validation_errors.append(_validate_trigger_keywords(trigger_keywords))
        if tags is not None:
            validation_errors.append(_validate_tags(tags))
        if tool_additions is not None:
            validation_errors.append(_validate_tool_list("tool_additions", tool_additions))
        if tool_restrictions is not None:
            validation_errors.append(_validate_tool_list("tool_restrictions", tool_restrictions))
        for err in validation_errors:
            if err is not None:
                return _tool_error("skill_update", err)

        if prompt_mode is not None and prompt_mode not in _VALID_PROMPT_MODES:
            return _tool_error(
                "skill_update",
                f"prompt_mode must be 'additive' or 'replace'; got {prompt_mode!r}",
            )

        if name is not None and name != entity.name:
            existing = await skills_collection.find_by_name_for_user(
                agent_id=agent_id,
                user_id=user_id,
                name=name,
            )
            if existing is not None and existing.skill_id != entity.skill_id:
                return _tool_error(
                    "skill_update",
                    f"a skill named {name!r} already exists",
                )

        # Compute the merged final shape (for at-least-one-payload +
        # ACL re-validation on any tool list the caller changed).
        # An empty string removes a body or a tool; removing the tool removes its arguments too.
        merged_body, merged_tool, merged_arguments = merged_skill_shape(
            entity, body=body, tool=tool, arguments=arguments
        )
        merged_additions = list(tool_additions) if tool_additions is not None else list(entity.tool_additions)
        merged_restrictions = (
            list(tool_restrictions) if tool_restrictions is not None else list(entity.tool_restrictions)
        )
        shape_err = skill_shape_error(body=merged_body, tool=merged_tool, arguments=merged_arguments)
        if shape_err is not None:
            return _tool_error("skill_update", shape_err)
        if not _at_least_one_payload(
            body=merged_body,
            tool=merged_tool,
            tool_additions=merged_additions,
            tool_restrictions=merged_restrictions,
        ):
            return _tool_error(
                "skill_update",
                "at least one of body, tool, tool_additions, or tool_restrictions must be non-empty",
            )

        if tool:
            err = await _check_tool_acl(
                registry=registry,
                agent_id=agent_id,
                user_id=user_id,
                tool_name=tool,
                label="tool",
            )
            if err is not None:
                return _tool_error("skill_update", err)

        if tool_additions is not None:
            for entry in tool_additions:
                err = await _check_tool_acl(
                    registry=registry,
                    agent_id=agent_id,
                    user_id=user_id,
                    tool_name=entry,
                    label="tool_additions",
                )
                if err is not None:
                    return _tool_error("skill_update", err)

        if tool_restrictions is not None:
            for entry in tool_restrictions:
                err = await _check_tool_acl(
                    registry=registry,
                    agent_id=agent_id,
                    user_id=user_id,
                    tool_name=entry,
                    label="tool_restrictions",
                )
                if err is not None:
                    return _tool_error("skill_update", err)

        before = _skill_fields(entity)

        # Apply changes via entity setters so the framework records
        # them on the change-tracking path.
        if name is not None:
            entity.name = name
        if summary is not None:
            entity.summary = summary
        if body is not None:
            entity.body = merged_body
        if tool is not None or arguments is not None:
            entity.tool = merged_tool
            entity.arguments = merged_arguments
        if prompt_mode is not None:
            entity.prompt_mode = prompt_mode
        if tool_additions is not None:
            entity.tool_additions = list(tool_additions)
        if tool_restrictions is not None:
            entity.tool_restrictions = list(tool_restrictions)
        if trigger_keywords is not None:
            entity.trigger_keywords = trigger_keywords
        if tags is not None:
            entity.tags = list(tags)
        if enabled is not None:
            entity.enabled = enabled
        after = _skill_fields(entity)
        changed = [field for field in _EDITABLE_FIELDS if before[field] != after[field]]
        if not changed:
            # Nothing differs: nothing is written, and the date it last changed stays true.
            return (
                f"[NO CHANGE] skill_update: [skill:{entity.skill_id}] {entity.name} already reads that way; "
                "nothing was written."
            )
        entity.date_updated = datetime.now(UTC)

        try:
            await skills_collection.save_entity(entity)
        except Exception as exc:
            contain(
                exc, log=log, what="skill_update persist failed", extra={"skill_id": str(parsed), "error": str(exc)}
            )
            return _tool_error("skill_update", f"persist failed: {exc}")

        lines = [f"Changed [skill:{entity.skill_id}] {entity.name}, was -> now:"]
        lines += [_change_line(field, before[field], after[field]) for field in changed]
        return "\n".join(lines)

    skill_update.description = (
        "Change a skill you wrote. Pass only what changes. Returns each field it changed, was -> now; "
        "a change to nothing says so and writes nothing."
    )

    return [skill_update]


# ---------------------------------------------------------------------------
# skill_delete
# ---------------------------------------------------------------------------


def load_skill_delete_tool(
    *,
    agent_id: UUID,
    user_id: UUID,
    skills_collection: AgentSkillCollection,
) -> list[BaseTool]:
    """Build a ``skill_delete`` tool that drops one prose-skill.

    Invocation history cascades server-side via the FK CASCADE -- the
    tool just deletes the parent row.

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param skills_collection: three-tier skills collection
    :ptype skills_collection: AgentSkillCollection
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("skill_delete", args_schema=SkillDeleteInput)
    async def skill_delete(skill_id: str) -> str:
        """Delete a prose-skill permanently. Invocation history cascades."""
        parsed = await _resolve_skill_ref(skills_collection, agent_id=agent_id, user_id=user_id, raw=skill_id)
        if parsed is None:
            return _tool_error("skill_delete", f"no skill has the id or name {skill_id!r}")

        entity = await _load_skill_for_user(
            collection=skills_collection,
            agent_id=agent_id,
            user_id=user_id,
            skill_id=parsed,
        )
        if entity is None:
            return _tool_error("skill_delete", "skill not found")

        # Snapshot the identity fields BEFORE delete. The entity is a
        # cache proxy that reads through the Collection's L1 store;
        # ``delete`` evicts the row from L1, after which ``entity.name`` /
        # ``entity.skill_id`` read back as ``None`` (the success message
        # would then raise on ``_as_uuid(None)``).
        deleted_name = entity.name
        deleted_skill_id = entity.skill_id

        try:
            await skills_collection.delete((agent_id, deleted_skill_id))
        except Exception as exc:
            contain(
                exc, log=log, what="skill_delete persist failed", extra={"skill_id": str(parsed), "error": str(exc)}
            )
            return _tool_error("skill_delete", f"persist failed: {exc}")

        return f"Deleted skill {deleted_name!r} ([skill:{deleted_skill_id}])."

    skill_delete.description = (
        "Delete a skill you wrote, and its history, for good. To turn it off and keep it, "
        "use skill_update with enabled=false."
    )

    return [skill_delete]


# ---------------------------------------------------------------------------
# skill_invoke
# ---------------------------------------------------------------------------


def load_skill_invoke_tool(
    *,
    agent_id: UUID,
    user_id: UUID,
    skills_collection: AgentSkillCollection,
    invocations_collection: AgentSkillInvocationCollection,
    conversation_id_resolver: ConversationIdResolver,
    active_skill_probe: ActiveSkillProbe,
    active_skill_setter: ActiveSkillSetter,
    bindable_tools: BindableTools | None = None,
) -> list[BaseTool]:
    """Build a ``skill_invoke`` tool with first-invoke-wins-per-turn semantics.

    The consumer wires four hooks:

    - ``conversation_id_resolver`` -- yields the live conversation
      UUID so the invocation row is attributed to the right
      conversation (not the conversation the factory was first minted
      under -- multiplex violation).
    - ``active_skill_probe`` -- returns the current ``_active_skill_id``
      for the turn (``None`` means no skill is active yet).
    - ``active_skill_setter`` -- records that the turn now has a skill
      active.

    Per Implementation note 4: ``prompt_mode='replace'`` skills cannot
    be ``skill_invoke``d mid-turn (only via wake-attach).

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param skills_collection: three-tier skills collection
    :ptype skills_collection: AgentSkillCollection
    :param invocations_collection: three-tier invocations collection
    :ptype invocations_collection: AgentSkillInvocationCollection
    :param conversation_id_resolver: callable returning the live conversation UUID
    :ptype conversation_id_resolver: ConversationIdResolver
    :param active_skill_probe: callable returning the active skill id
        for the in-flight turn (or ``None``)
    :ptype active_skill_probe: ActiveSkillProbe
    :param active_skill_setter: callable that marks ``skill_id`` as the
        turn's active skill
    :ptype active_skill_setter: ActiveSkillSetter
    :param bindable_tools: the tools the turn can bind, so the result names
        only those (:data:`BindableTools`); without it no tool is named
    :ptype bindable_tools: BindableTools | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    def _bindable() -> Collection[str] | None:
        if bindable_tools is None:
            return None
        try:
            return bindable_tools()
        except Exception as exc:  # the result still carries the steps; it names no tool
            contain(exc, log=log, what="skill_invoke bindable_tools raised", extra={"error": str(exc)})
            return None

    @tool("skill_invoke", args_schema=SkillInvokeInput)
    async def skill_invoke(skill_id: str, rationale: str | None = None) -> str:
        """Activate a prose-skill for the rest of THIS turn."""
        parsed = await _resolve_skill_ref(skills_collection, agent_id=agent_id, user_id=user_id, raw=skill_id)
        if parsed is None:
            return _tool_error("skill_invoke", f"no skill has the id or name {skill_id!r}")

        try:
            active = active_skill_probe()
        except Exception as exc:
            contain(exc, log=log, what="skill_invoke active_skill_probe raised", extra={"error": str(exc)})
            return _tool_error("skill_invoke", f"state probe failed: {exc}")
        if active is not None:
            return _tool_error(
                "skill_invoke",
                f"a skill is already active this turn ([skill:{active}]); only one invoke per turn",
            )

        entity = await _load_skill_for_user(
            collection=skills_collection,
            agent_id=agent_id,
            user_id=user_id,
            skill_id=parsed,
        )
        if entity is None:
            return _tool_error("skill_invoke", "skill not found")

        if not entity.enabled:
            return _tool_error(
                "skill_invoke",
                f"skill [skill:{entity.skill_id}] is disabled; re-enable via skill_update",
            )

        if entity.tool:
            # A tool-call skill has no steps to take on; activating it would
            # change nothing, and the turn would read as if the skill ran.
            return _tool_error(
                "skill_invoke",
                f"skill [skill:{entity.skill_id}] is one call to {entity.tool}; call {entity.tool} "
                "yourself, or attach the skill to a wake to run it on a schedule",
            )

        if entity.prompt_mode == "replace":
            return _tool_error(
                "skill_invoke",
                "skill has prompt_mode='replace'; replace-mode skills can only be activated by attaching to a wake schedule",
            )

        try:
            conversation_id = conversation_id_resolver()
        except Exception as exc:
            contain(exc, log=log, what="skill_invoke conversation_id_resolver raised", extra={"error": str(exc)})
            return _tool_error(
                "skill_invoke",
                f"conversation_id_resolver raised {type(exc).__name__}: {exc}",
            )
        if conversation_id is None:
            return _tool_error(
                "skill_invoke",
                "conversation_id_resolver returned None; cannot record invocation",
            )

        # Setter MUST succeed before we record the invocation so a
        # setter failure can't leak a row without corresponding
        # in-process state. With the previous order, a setter raise
        # left a persisted invocation row + unset state, letting the
        # next skill_invoke this turn pass the probe and double-record
        # (first-invoke-wins violated at the row count).
        try:
            active_skill_setter(entity.skill_id)
        except Exception as exc:
            contain(exc, log=log, what="skill_invoke active_skill_setter raised", extra={"error": str(exc)})
            return _tool_error(
                "skill_invoke",
                f"state setter failed: {exc}",
            )

        invocation_id = UUID(str(uuid7()))
        now = datetime.now(UTC)
        invocation = invocations_collection.create(
            {
                "invocation_id": invocation_id,
                "agent_id": agent_id,
                "skill_id": entity.skill_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message_id": None,
                "invocation_source": "invoke",
                "invoked_at": now,
                "outcome": None,
                "outcome_source": None,
                "notes": rationale,
            },
        )
        try:
            await invocations_collection.record(agent_id, invocation)
        except Exception as exc:
            # Setter already marked the turn active; on persist
            # failure we still return an error. The consumer's wrapper
            # treats tool errors as "this skill did NOT activate" and
            # may unset state accordingly. The in-process state will
            # be discarded with the turn regardless.
            contain(
                exc,
                log=log,
                what="skill_invoke invocation persist failed",
                extra={
                    "skill_id": str(entity.skill_id),
                    "error": str(exc),
                },
            )
            return _tool_error("skill_invoke", f"invocation persist failed: {exc}")

        # convert at border: skill_invoke-fired log extra_data fields
        log_invocation_id = str(invocation_id)
        log_conversation_id = str(conversation_id)
        log.info(
            "skill_invoke fired",
            extra={
                "extra_data": {
                    "skill_id": str(entity.skill_id),
                    "invocation_id": log_invocation_id,
                    "conversation_id": log_conversation_id,
                    "prompt_mode": entity.prompt_mode,
                }
            },
        )
        return _render_skill_invoke_block(entity, _bindable())

    skill_invoke.description = (
        "Use a skill for the rest of this turn. One skill per turn: a second call fails.\n"
        "Returns the skill's steps, whole, and the tools it adds that you can use."
    )

    return [skill_invoke]


# ---------------------------------------------------------------------------
# skill_report_outcome
# ---------------------------------------------------------------------------


def load_skill_report_outcome_tool(
    *,
    agent_id: UUID,
    conversation_id_resolver: ConversationIdResolver,
    active_skill_probe: ActiveSkillProbe,
    invocations_collection: AgentSkillInvocationCollection,
    skills_collection: AgentSkillCollection,
) -> list[BaseTool]:
    """Build a ``skill_report_outcome`` tool for the active skill's invocation.

    Replaces the retired ``[SUCCESS]``/``[FAILED]`` text-marker path: a tool
    call never enters the visible response stream, so there is nothing to
    scrub or leak-proof (unlike the marker, which required guarding every
    tool-loop runner's live stream). Resolves the active invocation the same
    way the consumer's post-LLM marker hook did -- the newest conversation
    row matching the turn's active skill (serial per-conversation execution
    makes "newest row" unambiguous within a turn) -- then records the
    outcome on both the invocation row and the skill's aggregate counters.

    :param agent_id: caller's agent UUID (partition column)
    :ptype agent_id: UUID
    :param conversation_id_resolver: callable returning the live conversation UUID
    :ptype conversation_id_resolver: ConversationIdResolver
    :param active_skill_probe: callable returning the active skill id for
        the in-flight turn (or ``None``)
    :ptype active_skill_probe: ActiveSkillProbe
    :param invocations_collection: three-tier invocations collection
    :ptype invocations_collection: AgentSkillInvocationCollection
    :param skills_collection: three-tier skills collection
    :ptype skills_collection: AgentSkillCollection
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("skill_report_outcome", args_schema=SkillReportOutcomeInput)
    async def skill_report_outcome(outcome: SkillOutcome, notes: str | None = None) -> str:
        """Report whether the currently-active skill's task succeeded or failed."""
        try:
            active_skill_id = active_skill_probe()
        except Exception as exc:
            contain(exc, log=log, what="skill_report_outcome active_skill_probe raised", extra={"error": str(exc)})
            return _tool_error("skill_report_outcome", f"state probe failed: {exc}")
        if active_skill_id is None:
            return _tool_error(
                "skill_report_outcome",
                "no skill is active this turn; call skill_invoke first",
            )

        try:
            conversation_id = conversation_id_resolver()
        except Exception as exc:
            contain(
                exc, log=log, what="skill_report_outcome conversation_id_resolver raised", extra={"error": str(exc)}
            )
            return _tool_error(
                "skill_report_outcome",
                f"conversation_id_resolver raised {type(exc).__name__}: {exc}",
            )
        if conversation_id is None:
            return _tool_error(
                "skill_report_outcome",
                "conversation_id_resolver returned None; cannot resolve invocation",
            )

        recent = await invocations_collection.list_for_conversation(
            agent_id,
            conversation_id,
            limit=1,
        )
        if not recent or recent[0].skill_id != active_skill_id:
            return _tool_error(
                "skill_report_outcome",
                "no active invocation found for this turn's skill; nothing to report",
            )
        invocation_id = recent[0].invocation_id
        kept_notes = notes.strip() if notes and notes.strip() else None

        recorded = await invocations_collection.record_outcome(
            agent_id,
            invocation_id,
            skill_id=active_skill_id,
            outcome=outcome,
            source="agent_tool",
            notes=kept_notes,
        )
        if not recorded.found:
            return _tool_error(
                "skill_report_outcome",
                "no active invocation found for this turn's skill; nothing to report",
            )

        log_invocation_id = str(invocation_id)
        log.info(
            "skill_report_outcome fired",
            extra={
                "extra_data": {
                    "skill_id": str(active_skill_id),
                    "invocation_id": log_invocation_id,
                    "outcome": outcome,
                    "previous": recorded.previous,
                    "counted": recorded.counted,
                    "notes": kept_notes,
                }
            },
        )
        if recorded.counted:
            # the counts moved in the same statement; the cached skill row is stale
            await skills_collection.invalidate_cache((agent_id, active_skill_id))
        skill = await skills_collection.get((agent_id, active_skill_id))
        tag = f"[skill:{active_skill_id}]" + (f" {skill.name}" if skill is not None else "")
        tally = (
            f" It has worked {int(skill.success_count or 0)} and failed {int(skill.failure_count or 0)} time(s)."
            if skill is not None
            else ""
        )
        if recorded.previous == outcome:
            if kept_notes is None or kept_notes == recorded.previous_notes:
                return (
                    f"[NO CHANGE] skill_report_outcome: this use of {tag} was already recorded as {outcome}; "
                    "it is counted once."
                )
            return f"Kept {outcome} for this use of {tag}, counted once. Its note now reads: {kept_notes}"
        why = f" ({json.dumps(kept_notes, ensure_ascii=False)})" if kept_notes else ""
        if recorded.previous is not None:
            return f"Changed this use of {tag} from {recorded.previous} to {outcome}{why}.{tally}"
        return f"Recorded {outcome} for this use of {tag}{why}.{tally}"

    skill_report_outcome.description = (
        "Record whether the skill in use did its job, with a line on why. Call it once you know. "
        "This is the only way the result is kept. A second report of the same use replaces the first "
        "and is not counted twice.\n"
        "Fails when no skill is in use this turn."
    )

    return [skill_report_outcome]


# ---------------------------------------------------------------------------
# skill_history
# ---------------------------------------------------------------------------


def _render_use(use: AgentSkillInvocationEntity, zone: tzinfo | None) -> list[str]:
    """One use of a skill: when, how it started, how it went, and where.

    :param use: the invocation row
    :ptype use: AgentSkillInvocationEntity
    :param zone: the reader's zone, or ``None`` for UTC
    :ptype zone: tzinfo | None
    :return: the line, and a second for the reason it was invoked when one was given
    :rtype: list[str]
    """
    how = "a wake ran it" if use.invocation_source == "wake" else "invoked"
    went = use.outcome or "no outcome reported"
    line = f"- {render_local_time(use.invoked_at, zone)}, {how}: {went}"
    outcome_notes = one_line(use.outcome_notes or "")
    if outcome_notes:
        line += f" -- {outcome_notes}"
    line += f" [conversation:{use.conversation_id}]"
    if use.message_id is not None:
        line += f" [message:{use.message_id}]"
    lines = [line]
    if use.notes:
        lines.extend(under(use.notes, "    ", first="  why it was invoked: "))
    return lines


def load_skill_history_tool(
    *,
    agent_id: UUID,
    user_id: UUID,
    skills_collection: AgentSkillCollection,
    invocations_collection: AgentSkillInvocationCollection,
    tz: tzinfo | str | None = None,
) -> list[BaseTool]:
    """Build a ``skill_history`` tool listing one skill's uses, newest first.

    Each use says when (in the reader's zone), whether a wake ran it or it was
    invoked, how it went and the note the agent left, and the conversation and
    message it was in. The head says how often the skill was used and how it
    went, and how many uses there are in all, so a page says which part it is.

    Cross-user isolation: a skill belonging to another user is "not found" (SK-13).

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param skills_collection: three-tier skills collection
    :ptype skills_collection: AgentSkillCollection
    :param invocations_collection: three-tier invocations collection
    :ptype invocations_collection: AgentSkillInvocationCollection
    :param tz: the zone the times are said in; ``None`` says them in UTC
    :ptype tz: tzinfo | str | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    zone = resolve_zone(tz)

    @tool("skill_history", args_schema=SkillHistoryInput)
    async def skill_history(skill_id: str, limit: int = 20, offset: int = 0) -> str:
        """List one skill's uses, newest first."""
        parsed = await _resolve_skill_ref(skills_collection, agent_id=agent_id, user_id=user_id, raw=skill_id)
        if parsed is None:
            return _tool_error("skill_history", f"no skill has the id or name {skill_id!r}")
        entity = await _load_skill_for_user(
            collection=skills_collection,
            agent_id=agent_id,
            user_id=user_id,
            skill_id=parsed,
        )
        if entity is None:
            return _tool_error("skill_history", "skill not found")
        total = await invocations_collection.count_for_skill(agent_id, entity.skill_id)
        tag = f"[skill:{entity.skill_id}] {entity.name}"
        if total == 0:
            return f"{tag} has not been used yet."
        uses = await invocations_collection.list_for_skill(agent_id, entity.skill_id, limit=limit, offset=offset)
        worked, failed = int(entity.success_count or 0), int(entity.failure_count or 0)
        head = f"{tag}: {total} use(s) in all; worked {worked}, failed {failed}."
        if not uses:
            return f"{head} None past use {offset:,}; skill_history with offset=0 starts at the newest."
        first, last = offset + 1, offset + len(uses)
        lines = [f"{head} Uses {first}-{last} of {total}, newest first:"]
        for use in uses:
            lines += _render_use(use, zone)
        if last < total:
            lines.append(f"skill_history with offset={last} shows the next {min(limit, total - last)}.")
        return "\n".join(lines)

    skill_history.description = (
        "List a skill's uses, newest first: when, whether a wake ran it or you invoked it, how it went "
        "and your note, and the conversation it was in."
    )

    return [skill_history]


# ---------------------------------------------------------------------------
# skill_introspect
# ---------------------------------------------------------------------------


def load_skill_introspect_tool(
    *,
    agent_id: UUID,
    user_id: UUID,
    skills_collection: AgentSkillCollection,
    registry: SkillRegistryClient,
) -> list[BaseTool]:
    """Build a ``skill_introspect`` tool returning the minimal-token shape.

    Resolves ``name_or_id`` into either a prose-skill (preferred when
    both kinds exist with the same name -- Implementation note 7) or a
    tool-skill via the registry. Returns the PLACEMENT §1.8 shape with
    NO operational metadata (SK-17).

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param skills_collection: three-tier skills collection
    :ptype skills_collection: AgentSkillCollection
    :param registry: consumer-supplied registry client
    :ptype registry: SkillRegistryClient
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("skill_introspect", args_schema=SkillIntrospectInput)
    async def skill_introspect(name_or_id: str) -> str:
        """Examine a skill before using it (prose body OR tool args+example)."""
        candidate = name_or_id.strip()

        # If the input parses as a skill UUID, try the prose-skill
        # primary-key path first.
        as_uuid = _parse_skill_id(candidate)
        if as_uuid is not None:
            entity = await _load_skill_for_user(
                collection=skills_collection,
                agent_id=agent_id,
                user_id=user_id,
                skill_id=as_uuid,
            )
            if entity is not None:
                return _render_prose_introspect(entity)

        # Otherwise treat it as a name and try prose first, then tool.
        bare_name = candidate
        if bare_name.startswith("[skill:") and bare_name.endswith("]"):
            bare_name = bare_name[len("[skill:") : -1].strip()
        try:
            prose_entity = await skills_collection.find_by_name_for_user(
                agent_id=agent_id,
                user_id=user_id,
                name=bare_name,
            )
        except Exception as exc:
            contain(
                exc, log=log, what="skill_introspect name lookup raised", extra={"name": bare_name, "error": str(exc)}
            )
            prose_entity = None
        if prose_entity is not None:
            return _render_prose_introspect(prose_entity)

        try:
            tool_payload = await registry.get_tool_introspect(
                actor_user_id=user_id,
                actor_agent_id=agent_id,
                mcp_name=bare_name,
            )
        except Exception as exc:
            contain(
                exc,
                log=log,
                what="skill_introspect tool lookup raised",
                extra={"mcp_name": bare_name, "error": str(exc)},
            )
            tool_payload = None
        if tool_payload is None:
            return _tool_error(
                "skill_introspect",
                f"no skill or tool found matching {name_or_id!r}",
            )
        return _render_tool_introspect(tool_payload)

    skill_introspect.description = (
        "Look at a skill before you use it: its steps, tools, arguments and examples. "
        "Use it to learn how to use a skill with skill_invoke or a wake."
    )

    return [skill_introspect]
