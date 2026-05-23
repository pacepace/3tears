"""Agent-skills package -- procedural memory for 3tears agents.

Shard 01 landed the schema + Collection layer. Shard 02 added the
seven agent tools (``skill_create`` / ``skill_list`` / ``skill_get`` /
``skill_update`` / ``skill_delete`` / ``skill_invoke`` /
``skill_introspect``). Shard 03 (this release) adds the per-turn
composition (``compose_turn_context``) + active-skill rendering
(``render_skill_body_block``) for the consumer's ``personality_node``.
The public surface here re-exports the entity / collection / migration
registration triad alongside the tool loader factories + their
Pydantic input schemas + the ``SkillRegistryClient`` Protocol the
consumer implements + the composition surface.

Version is sourced from the installed package metadata so a future
release that bumps ``pyproject.toml`` without touching this file
cannot drift the runtime ``__version__`` reporting.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-agent-skills")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.agent.skills.collections import (
    AgentSkillCollection,
    AgentSkillInvocationCollection,
)
from threetears.agent.skills.entities import (
    AgentSkillEntity,
    AgentSkillInvocationEntity,
)
from threetears.agent.skills.migrations import register
from threetears.agent.skills.rendering import (
    ComposedTurnContext,
    compose_turn_context,
    render_skill_body_block,
)
from threetears.agent.skills.tables import (
    agent_skill_invocations_table,
    agent_skills_table,
)
from threetears.agent.skills.tools import (
    ActiveSkillProbe,
    ActiveSkillSetter,
    ConversationIdResolver,
    SkillCreateInput,
    SkillDeleteInput,
    SkillEligibleTool,
    SkillGetInput,
    SkillIntrospectInput,
    SkillInvokeInput,
    SkillListInput,
    SkillRegistryClient,
    SkillToolIntrospect,
    SkillUpdateInput,
    load_skill_create_tool,
    load_skill_delete_tool,
    load_skill_get_tool,
    load_skill_introspect_tool,
    load_skill_invoke_tool,
    load_skill_list_tool,
    load_skill_update_tool,
)
from threetears.agent.skills.types import (
    InvocationSource,
    OutcomeSource,
    PromptMode,
    SkillOutcome,
    SkillSource,
)

__all__ = [
    "ActiveSkillProbe",
    "ActiveSkillSetter",
    "AgentSkillCollection",
    "AgentSkillEntity",
    "AgentSkillInvocationCollection",
    "AgentSkillInvocationEntity",
    "ComposedTurnContext",
    "ConversationIdResolver",
    "InvocationSource",
    "OutcomeSource",
    "PromptMode",
    "SkillCreateInput",
    "SkillDeleteInput",
    "SkillEligibleTool",
    "SkillGetInput",
    "SkillIntrospectInput",
    "SkillInvokeInput",
    "SkillListInput",
    "SkillOutcome",
    "SkillRegistryClient",
    "SkillSource",
    "SkillToolIntrospect",
    "SkillUpdateInput",
    "agent_skill_invocations_table",
    "agent_skills_table",
    "compose_turn_context",
    "load_skill_create_tool",
    "load_skill_delete_tool",
    "load_skill_get_tool",
    "load_skill_introspect_tool",
    "load_skill_invoke_tool",
    "load_skill_list_tool",
    "load_skill_update_tool",
    "register",
    "render_skill_body_block",
]
