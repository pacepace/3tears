"""conversations package public surface.

owns the :class:`Conversation` entity, the
:class:`ConversationsCollection` three-tier collection, and the
agent-scope migration registering the ``conversations`` table. pulled
out of :mod:`threetears.agent.memory` because multiple packages --
memory, agent-tools context items, and upcoming workspace bindings --
key off ``conversation_id`` but none of them is the natural owner.
"""

from __future__ import annotations

__version__ = "0.7.0"

from threetears.conversations.authorize import (
    ACTION_CONVERSATION_DELETE,
    ACTION_CONVERSATION_READ,
    ACTION_CONVERSATION_WRITE,
    CONVERSATION_NAMESPACE_TYPE,
    CONVERSATION_OWNER_GROUP_PREFIX,
    CONVERSATION_OWNER_ROLE_NAME,
    ConversationAccessDenied,
    ConversationAuthorizerDependencies,
    authorize_conversation_access,
    conversation_namespace_name,
    ensure_conversation_owner_assignment,
)
from threetears.conversations.buffer import ConversationWriteBuffer
from threetears.conversations.collection import ConversationsCollection
from threetears.conversations.entity import Conversation, ConversationStatus
from threetears.conversations.events import ConversationSummarizedEvent
from threetears.conversations.migrations import register

__all__ = [
    "ACTION_CONVERSATION_DELETE",
    "ACTION_CONVERSATION_READ",
    "ACTION_CONVERSATION_WRITE",
    "CONVERSATION_NAMESPACE_TYPE",
    "CONVERSATION_OWNER_GROUP_PREFIX",
    "CONVERSATION_OWNER_ROLE_NAME",
    "Conversation",
    "ConversationAccessDenied",
    "ConversationAuthorizerDependencies",
    "ConversationStatus",
    "ConversationSummarizedEvent",
    "ConversationWriteBuffer",
    "ConversationsCollection",
    "authorize_conversation_access",
    "conversation_namespace_name",
    "ensure_conversation_owner_assignment",
    "register",
]
