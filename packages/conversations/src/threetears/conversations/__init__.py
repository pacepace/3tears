"""conversations package public surface.

owns the :class:`Conversation` entity, the
:class:`ConversationsCollection` three-tier collection, and the
agent-scope migration registering the ``conversations`` table. pulled
out of :mod:`threetears.agent.memory` because multiple packages --
memory, agent-tools context items, and upcoming workspace bindings --
key off ``conversation_id`` but none of them is the natural owner.
"""

from __future__ import annotations

__version__ = "0.5.0"

from threetears.conversations.collection import ConversationsCollection
from threetears.conversations.entity import Conversation
from threetears.conversations.migrations import register

__all__ = [
    "Conversation",
    "ConversationsCollection",
    "register",
]
