"""conversations package public surface.

owns the :class:`Conversation` entity, the
:class:`ConversationsCollection` three-tier collection, and the
agent-scope migration registering the ``conversations`` table. pulled
out of :mod:`threetears.agent.memory` because multiple packages --
memory, agent-tools context items, and upcoming workspace bindings --
key off ``conversation_id`` but none of them is the natural owner.
"""

from __future__ import annotations

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-conversations")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

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
from threetears.conversations.merge import repoint_user
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
    "repoint_user",
]
