"""large tool-result offload contract for the shared ``tool_node``.

a tool result whose serialized content exceeds a configured size
threshold blows the LLM context window -- the full dump rides on the
producing turn and persists across every later turn until lossy
whole-history summarization. this module defines the *framework*
contract for moving that dump out-of-band: the ``tool_node`` seam
(:func:`threetears.langgraph.nodes.tool_node`) stores the full content
via an injected :class:`ToolResultOffloader` and shows the model a
summary plus a recall handle instead.

the contract lives in the ``langgraph`` package on purpose: the seam
that consumes it is here, the package has no dependency on
``threetears.agent.tools`` (so the node never hard-imports a context
manager), and :class:`ToolResultOffloader` is a pure *structural*
:class:`typing.Protocol` -- a concrete implementation (e.g. the SDK's
``ContextItemOffloader`` over ``ToolContextManager``) satisfies it
without importing anything from here except the small
:class:`OffloadResult` value it returns.

opt-in by construction: when no offloader is injected on
``config["configurable"]`` the seam is byte-for-byte unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

__all__ = [
    "DEFAULT_OFFLOAD_THRESHOLD_CHARS",
    "NEVER_OFFLOAD_TOOLS",
    "OffloadResult",
    "ToolResultOffloader",
    "format_offload_handle",
    "has_offload_handle",
    "is_never_offload_tool",
]

# default inline ceiling (T1): a tool result whose serialized content
# exceeds this many characters (~2K tokens) is offloaded and the model
# sees ``summary + [ctx:<handle>]`` instead of the raw dump. the seam
# reads the effective value from ``config["configurable"]
# ["offload_threshold_chars"]`` and falls back to this when the key is
# absent, so the constant is the single source of truth shared by the
# node default and the SDK config default.
DEFAULT_OFFLOAD_THRESHOLD_CHARS = 8192

# tools whose own result must NEVER be offloaded. the recall tool is the
# canonical member: its result IS the full recalled content (by
# definition larger than the threshold), so re-offloading it would hand
# the model a fresh ``[ctx:<id>]`` instead of the bytes it asked for --
# an infinite recall loop that defeats the feature. the canonical name
# is the agent-tools ``threetears.context_recall`` builtin's ``mcp_name``;
# it lives here as a string because the ``langgraph`` package must NOT
# depend on ``agent.tools`` (the whole reason this contract lives here).
NEVER_OFFLOAD_TOOLS: frozenset[str] = frozenset({"threetears.context_recall"})


def _normalize_tool_name(name: str) -> str:
    """canonicalize a tool name for membership comparison.

    the seam matches against ``tool_call["name"]``, which is the
    LLM-facing name. that is normally the dotted canonical form
    (``threetears.context_recall``), but a provider that sanitizes dots
    to underscores on the wire could surface ``threetears_context_recall``.
    folding ``.`` -> ``_`` makes both forms compare equal so the
    never-offload guard is robust to either.

    :param name: a tool name in dotted or sanitized form.
    :ptype name: str
    :return: the underscore-folded canonical comparison key.
    :rtype: str
    """
    return name.replace(".", "_")


_NEVER_OFFLOAD_NORMALIZED: frozenset[str] = frozenset(_normalize_tool_name(n) for n in NEVER_OFFLOAD_TOOLS)

# matches the offload handle the seam appends (``[ctx:<id>]``) at the END
# of a ToolMessage's content. anchored to the tail because the seam
# always appends it after the summary; anchoring avoids flagging a tool
# result that merely mentions a bracketed token mid-text.
_OFFLOAD_HANDLE_RE = re.compile(r"\[ctx:[^\]\s]+\]\s*\Z")


def is_never_offload_tool(tool_name: str) -> bool:
    """whether a tool's result must never be offloaded.

    :param tool_name: the tool name as seen at the seam
        (``tool_call["name"]``), dotted or sanitized.
    :ptype tool_name: str
    :return: ``True`` when the tool is in :data:`NEVER_OFFLOAD_TOOLS`
        (under dot/underscore normalization).
    :rtype: bool
    """
    return _normalize_tool_name(tool_name) in _NEVER_OFFLOAD_NORMALIZED


def format_offload_handle(handle: str) -> str:
    """render the recall handle marker appended to an offloaded result.

    single source of truth for the ``[ctx:<id>]`` marker shape so the
    seam (which appends it) and any consumer that detects it (e.g. a
    post-reply context-save node skipping already-offloaded messages)
    cannot disagree on the format.

    :param handle: the stored context-item id.
    :ptype handle: str
    :return: the marker string ``"[ctx:<handle>]"``.
    :rtype: str
    """
    return f"[ctx:{handle}]"


def has_offload_handle(text: str) -> bool:
    """whether ``text`` already carries a trailing offload recall handle.

    used by consumers (e.g. the SDK post-reply context-save node) to skip
    a ToolMessage that the offload seam already replaced with
    ``summary + [ctx:<id>]`` -- the full content is already stored under
    that id, so re-saving the summary would surface a misleading second
    handle that recalls to only the summary.

    :param text: a ToolMessage's serialized content.
    :ptype text: str
    :return: ``True`` when a trailing ``[ctx:<id>]`` marker is present.
    :rtype: bool
    """
    return _OFFLOAD_HANDLE_RE.search(text) is not None


@dataclass(frozen=True)
class OffloadResult:
    """outcome of a successful :meth:`ToolResultOffloader.offload`.

    :param summary: token-efficient stand-in shown to the model in place
        of the full content. for the structural (2a) default this is a
        size/recall hint; a richer tool-authored summary can replace it
        in a later sub-cycle without changing this shape.
    :ptype summary: str
    :param handle: opaque recall handle (the stored context-item id) the
        model passes to ``context_recall`` to pull the full content back.
    :ptype handle: str
    """

    summary: str
    handle: str


class ToolResultOffloader(Protocol):
    """structural contract for moving a large tool result out-of-band.

    a concrete offloader stores the full ``content`` somewhere durable
    and conversation-scoped (the SDK implements this over the existing
    ``ToolContextManager`` three-tier store) and returns an
    :class:`OffloadResult` carrying a summary plus a recall handle.

    the implementation is responsible for tenant/conversation isolation;
    the seam passes only the verified ``conversation_id`` / ``user_id``
    it reads off ``config["configurable"]`` and never the content store
    itself, so the ``langgraph`` package stays free of any context-store
    dependency.
    """

    async def offload(
        self,
        *,
        tool_name: str,
        content: str,
        conversation_id: UUID,
        user_id: UUID | None,
        tool_summary: str | None = None,
    ) -> OffloadResult | None:
        """store ``content`` out-of-band and return a summary + handle.

        :param tool_name: name of the tool that produced the result;
            used to label the stored item and build the summary.
        :ptype tool_name: str
        :param content: full serialized tool-result content to store.
        :ptype content: str
        :param conversation_id: verified conversation scope the content
            is stored under and later recalled within.
        :ptype conversation_id: UUID
        :param user_id: verified invoking user, or ``None`` when the
            envelope carried no user identity.
        :ptype user_id: UUID | None
        :param tool_summary: a tool-authored summary (2b) the seam lifts
            from the result's ``ToolMessage.artifact``; when present the
            offloader uses it verbatim as the model-visible summary instead
            of the structural byte-count default (a good tool summary is
            what makes recall rare). ``None`` -> the structural summary.
        :ptype tool_summary: str | None
        :return: an :class:`OffloadResult` on success, or ``None`` when
            the offloader declines to store (e.g. no context manager is
            available for the conversation); ``None`` makes the seam fall
            back to the full content, preserving existing behavior.
        :rtype: OffloadResult | None
        """
        ...
