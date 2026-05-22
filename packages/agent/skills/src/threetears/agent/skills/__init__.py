"""Agent-skills package -- procedural memory for 3tears agents.

This shard exposes only the schema + Collection layer. Agent tools
(``skill_create`` / ``skill_list`` / ``skill_get`` / ``skill_update``
/ ``skill_delete`` / ``skill_invoke`` / ``skill_introspect``) land in
shard 02; the per-turn composition function (``compose_turn_context``)
lands in shard 03. The public surface here is the entity / collection
/ migration registration triad.

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
from threetears.agent.skills.types import (
    InvocationSource,
    OutcomeSource,
    PromptMode,
    SkillOutcome,
    SkillSource,
)

__all__ = [
    "AgentSkillCollection",
    "AgentSkillEntity",
    "AgentSkillInvocationCollection",
    "AgentSkillInvocationEntity",
    "InvocationSource",
    "OutcomeSource",
    "PromptMode",
    "SkillOutcome",
    "SkillSource",
    "register",
]
