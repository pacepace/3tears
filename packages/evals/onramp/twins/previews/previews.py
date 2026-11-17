"""Notification previews: the short form of a message that a phone shows before the message is opened.

``preview(body, channel)`` is the preview for one message on one channel. Whitespace runs (new lines, tabs, doubled
spaces) become single spaces. A message that fits the channel's limit is shown whole. A longer one is cut and ends in
an ellipsis (``…``, one character), and the preview, ellipsis included, is never longer than the channel's limit;
trailing spaces and punctuation before the ellipsis are dropped, so a cut never reads ``word,…``.
"""

from __future__ import annotations

ELLIPSIS = "…"

#: Characters each channel shows, counting the ellipsis.
LIMITS = {"lock_screen": 110, "banner": 64, "watch": 40}

_TRAILING = " ,.;:-–—"


def flatten(body: str) -> str:
    """The message on one line, every run of whitespace a single space."""
    return " ".join(body.split())


def preview(body: str, channel: str = "lock_screen") -> str:
    """The preview of ``body`` on ``channel``."""
    if channel not in LIMITS:
        raise ValueError(f"unknown channel {channel!r}; channels are {', '.join(sorted(LIMITS))}")
    limit = LIMITS[channel]
    text = flatten(body)
    if len(text) <= limit:
        return text
    return text[: limit - len(ELLIPSIS)].rstrip(_TRAILING) + ELLIPSIS


def previews(body: str) -> dict[str, str]:
    """The preview on every channel."""
    return {channel: preview(body, channel) for channel in LIMITS}
