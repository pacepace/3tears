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
"""

from __future__ import annotations

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


async def load_identity_propose_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: IdentityAuthorizerDependencies,
    collection: IdentityVersionsCollection,
) -> list[BaseTool]:
    """create an ``identity_propose`` tool bound to ``user_id``.

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
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("identity_propose", args_schema=IdentityProposeInput)
    async def identity_propose(block_key: str, content: str, rationale: str) -> str:
        """Propose a change to one of your own identity blocks."""
        if block_key not in IDENTITY_BLOCK_KEY_VALUES:
            return f"Unknown block_key '{block_key}'. Use one of: {', '.join(IDENTITY_BLOCK_KEY_VALUES)}."
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
        )
        if version is None:
            return f"Could not propose a change to '{block_key}'."
        tier = IDENTITY_BLOCK_TIERS[IdentityBlockKey(block_key)]
        if tier is IdentityTier.TIER_1:
            return (
                f"Proposed a change to '{block_key}' [version:{version.version_id}]. "
                "It is awaiting the user's consent before it takes effect."
            )
        return (
            f"Updated '{block_key}' [version:{version.version_id}]. It is live now; "
            "the user can review or roll it back."
        )

    identity_propose.description = (
        "Propose a change to one of your own identity blocks (personality, "
        "reinforcement, anti_sycophant, self_improvement, presence). "
        "Identity-shaping blocks await the user's consent; routine blocks apply "
        "immediately with the user able to roll back. Give the full new text + a "
        "short rationale."
    )
    return [identity_propose]
