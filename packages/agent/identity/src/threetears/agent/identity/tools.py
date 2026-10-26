"""Identity tools -- the private ``identity_propose`` agent tool.

One verb, **private** (absent from any outward-facing tool set -- the LLM
uses it only on the agent-internal self-reflection path): propose a new
version of one of its own identity blocks. Consent / reject / rollback are
the human's, exposed by the host's API (they call
:mod:`threetears.agent.identity.lifecycle` directly), not as agent tools --
the agent proposes, the human disposes.

The factory binds ``user_id`` at load time (minted per-user) and never
exposes it in the Pydantic schema: user isolation is the ``user_id`` on the
collection reads + the lifecycle ownership checks, NOT RBAC (RBAC is the
agent-owner short-circuit alone). Tier-1 proposals await consent; tier-2
proposals auto-apply -- the tool's return text says which.

The factory also binds which block variants the tool may write
(``variants``; ``""`` is the block itself, and the only one by default). The
host opens more where its own rules allow them, so the model cannot write a
variant the host did not open for this run.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from threetears.observe import get_logger

from threetears.agent.identity.authorize import IdentityAuthorizerDependencies
from threetears.agent.identity.collections import IdentityVersionsCollection
from threetears.agent.identity.lifecycle import propose
from threetears.agent.identity.types import (
    IDENTITY_BLOCK_KEY_VALUES,
    IDENTITY_BLOCK_TIERS,
    IdentityBlockKey,
    IdentityTier,
    check_variant,
)

__all__ = [
    "IdentityProposeInput",
    "load_identity_propose_tool",
]

log = get_logger(__name__)


class IdentityProposeInput(BaseModel):
    """Args for ``identity_propose``."""

    block_key: str = Field(description=f"which identity block to change: one of {', '.join(IDENTITY_BLOCK_KEY_VALUES)}")
    content: str = Field(description="the full new text of the block")
    rationale: str = Field(description="why you want to change it (one or two sentences)")
    variant: str = Field(default="", description="which variant of the block; leave empty for the block itself")


async def load_identity_propose_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: IdentityAuthorizerDependencies,
    collection: IdentityVersionsCollection,
    variants: Sequence[str] = ("",),
) -> list[BaseTool]:
    """create an ``identity_propose`` tool bound to ``user_id`` and the variants it may write.

    :param user_id: owning user (isolation boundary); bound at load
    :ptype user_id: UUID
    :param agent_id: owning agent UUID (namespace owner + partition)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID (namespace scope grain)
    :ptype customer_id: UUID
    :param authorizer: identity authorizer dependency bundle
    :ptype authorizer: IdentityAuthorizerDependencies
    :param collection: three-tier identity-versions collection
    :ptype collection: IdentityVersionsCollection
    :param variants: the block variants this tool may write; ``""`` is the
        block itself
    :ptype variants: Sequence[str]
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``variants`` is empty or names a variant a block does not take
    """
    allowed = tuple(dict.fromkeys(check_variant(v) for v in variants))
    if not allowed:
        raise ValueError("identity_propose needs at least one variant it may write")

    @tool("identity_propose", args_schema=IdentityProposeInput)
    async def identity_propose(block_key: str, content: str, rationale: str, variant: str = "") -> str:
        """Propose a change to one of your own identity blocks."""
        if block_key not in IDENTITY_BLOCK_KEY_VALUES:
            return f"[TOOL ERROR] Unknown block_key '{block_key}'. Use one of: {', '.join(IDENTITY_BLOCK_KEY_VALUES)}."
        if variant not in allowed:
            return f"[TOOL ERROR] Variant '{variant}' is not open here. Use one of: {_named(allowed)}."
        name = f"{block_key}/{variant}" if variant else block_key
        version = await propose(
            collection,
            authorizer,
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            block_key=block_key,
            content=content,
            rationale=rationale,
            proposer_agent_id=agent_id,
            caller_agent_id=agent_id,
            variant=variant,
        )
        if version is None:
            return f"[TOOL ERROR] Could not propose a change to '{name}'."
        tier = IDENTITY_BLOCK_TIERS[IdentityBlockKey(block_key)]
        if tier is IdentityTier.TIER_1:
            return (
                f"Proposed a change to '{name}' [version:{version.version_id}]. "
                "It is awaiting the user's consent before it takes effect."
            )
        return f"Updated '{name}' [version:{version.version_id}]. It is live now; the user can review or roll it back."

    # Block list interpolated, never spelled out: this string is the only
    # place the model learns which blocks it may edit, so a hand-kept copy
    # going stale does not fail a test -- it silently makes a real block
    # unreachable to the agent.
    identity_propose.description = (
        "Propose a change to one of your own identity blocks "
        f"({', '.join(IDENTITY_BLOCK_KEY_VALUES)}). "
        "Identity-shaping blocks await the user's consent; routine blocks apply "
        "immediately with the user able to roll back. Give the full new text + a "
        "short rationale."
    )
    # Named only when the host opened a variant: a model told of variants it
    # cannot use would try them.
    if allowed != ("",):
        identity_propose.description += f" Variants you may write: {_named(allowed)}."
    return [identity_propose]


def _named(variants: tuple[str, ...]) -> str:
    """The variants as a model reads them: the block itself as "" and the rest by name."""
    return ", ".join(repr(v) if not v else v for v in variants)
