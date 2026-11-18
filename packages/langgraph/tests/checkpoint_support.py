"""the LangGraph values checkpoint-saver tests hand the saver, built whole.

:func:`make_checkpoint` is a complete :class:`~langgraph.checkpoint.base.Checkpoint` (every key
the TypedDict declares), so a test passes the saver exactly what LangGraph would.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import Checkpoint

#: the checkpoint format version LangGraph writes today
CHECKPOINT_VERSION = 4


def make_checkpoint(
    checkpoint_id: str,
    *,
    ts: str = "2026-01-01T00:00:00Z",
    channel_values: dict[str, Any] | None = None,
) -> Checkpoint:
    """a LangGraph checkpoint with id ``checkpoint_id`` holding ``channel_values``, no versions seen.

    :param checkpoint_id: the checkpoint's id
    :ptype checkpoint_id: str
    :param ts: the checkpoint's timestamp
    :ptype ts: str
    :param channel_values: the channels' values; none when omitted
    :ptype channel_values: dict[str, Any] | None
    :return: the checkpoint
    :rtype: Checkpoint
    """
    return Checkpoint(
        v=CHECKPOINT_VERSION,
        id=checkpoint_id,
        ts=ts,
        channel_values=channel_values if channel_values is not None else {},
        channel_versions={},
        versions_seen={},
        updated_channels=None,
    )
