"""Where a memory came from, kept as tags on its row.

Two facts a host can record when a memory is written, and every reader can
ask for afterwards:

- **who wrote it**: ``made_by:<writer>``, the writer in the host's own words
  (an extraction after a turn, the agent's own tool, a dream's merge).
- **whether anyone was there**: :data:`UNATTENDED_TAG`, on a memory written
  on a turn no person was present for (a scheduled wake, a webhook, a report
  from another agent).

They are tags because that is where a memory's state already lives
(:mod:`~threetears.agent.memory.revisions`): one indexed column, read by the
same key test, carried by every export of the row. A memory with neither tag
is one whose origin nobody recorded, and nothing here guesses it:
:func:`origin_of` answers ``None`` for both, and a reader that leaves
unattended memories out leaves that one in.

The platform writes one of these itself: a dream's gist is unattended when
every memory it merges is (:mod:`~threetears.agent.memory.dream`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MADE_BY_PREFIX",
    "UNATTENDED_TAG",
    "MemoryOrigin",
    "made_by_tag",
    "origin_of",
    "origin_tags",
]

#: The tag a memory written with nobody present carries.
UNATTENDED_TAG = "unattended"
#: The tag that names the writer, after the colon.
MADE_BY_PREFIX = "made_by:"


@dataclass(frozen=True)
class MemoryOrigin:
    """What a memory's tags say of where it came from.

    :ivar made_by: the writer the host named, or ``None`` when none was recorded
    :ivar unattended: ``True`` when it was written with nobody present; ``None``
        when the row records no origin at all, which is not the same as attended
    """

    made_by: str | None = None
    unattended: bool | None = None


def made_by_tag(writer: str) -> str:
    """The tag that names ``writer`` as who wrote a memory.

    :param writer: the writer, in the host's words
    :ptype writer: str
    :return: the tag
    :rtype: str
    """
    return f"{MADE_BY_PREFIX}{writer}"


def origin_tags(*, made_by: str | None = None, unattended: bool = False) -> list[str]:
    """The tags that record an origin, for a new memory's ``tags``.

    :param made_by: the writer, when the host names one
    :ptype made_by: str | None
    :param unattended: nobody was present when it was written
    :ptype unattended: bool
    :return: the tags, possibly none
    :rtype: list[str]
    """
    tags = [made_by_tag(made_by)] if made_by else []
    if unattended:
        tags.append(UNATTENDED_TAG)
    return tags


def _tags(value: Any) -> list[str]:
    """A ``tags`` value as a list: a raw read hands the column back as JSON text."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return [t for t in value if isinstance(t, str)] if isinstance(value, list) else []


def origin_of(tags: Any) -> MemoryOrigin:
    """Read an origin from a memory's tags.

    A row that names a writer and carries no :data:`UNATTENDED_TAG` was written
    with someone there, as far as its writer knew. A row with neither says
    nothing, and ``unattended`` is ``None``.

    :param tags: the row's ``tags``, as a list or as the JSON text a raw read returns
    :ptype tags: Any
    :return: the origin
    :rtype: MemoryOrigin
    """
    listed = _tags(tags)
    made_by = next((t[len(MADE_BY_PREFIX) :] for t in listed if t.startswith(MADE_BY_PREFIX)), None)
    if UNATTENDED_TAG in listed:
        return MemoryOrigin(made_by=made_by or None, unattended=True)
    return MemoryOrigin(made_by=made_by or None, unattended=False if made_by else None)
