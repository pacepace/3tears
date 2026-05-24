"""ID parsers for the agent tools.

Accept either the tagged form (``[schedule:<uuid>]`` /
``[webhook:<uuid>]``) the catalog renders OR a bare UUID. The LLM can
paste back the tag verbatim or strip the brackets; both work. Returns
``None`` on parse failure so the tool layer can surface a tool-error
rather than raise.

Mirrors :func:`threetears.agent.skills.tools._parse_skill_id` so the
shape stays consistent across the 3tears tool surface.
"""

from __future__ import annotations

from uuid import UUID

__all__ = [
    "parse_schedule_id",
    "parse_subscription_id",
]


def parse_schedule_id(raw: str) -> UUID | None:
    """Parse ``[schedule:<uuid>]`` or a bare UUID string into :class:`UUID`.

    :param raw: candidate id string supplied by the LLM
    :ptype raw: str
    :return: parsed UUID or ``None`` on failure
    :rtype: UUID | None
    """
    return _parse_tagged(raw, "schedule")


def parse_subscription_id(raw: str) -> UUID | None:
    """Parse ``[webhook:<uuid>]`` or a bare UUID string into :class:`UUID`.

    :param raw: candidate id string supplied by the LLM
    :ptype raw: str
    :return: parsed UUID or ``None`` on failure
    :rtype: UUID | None
    """
    return _parse_tagged(raw, "webhook")


def _parse_tagged(raw: str, tag: str) -> UUID | None:
    """Strip a ``[<tag>:<uuid>]`` wrapper if present, then parse to UUID.

    :param raw: candidate string
    :ptype raw: str
    :param tag: leading tag name (``schedule`` / ``webhook``)
    :ptype tag: str
    :return: parsed UUID or ``None`` on failure
    :rtype: UUID | None
    """
    if not raw or not isinstance(raw, str):
        return None
    candidate = raw.strip()
    prefix = f"[{tag}:"
    if candidate.startswith(prefix) and candidate.endswith("]"):
        candidate = candidate[len(prefix) : -1].strip()
    try:
        return UUID(candidate)
    except ValueError:
        # malformed UUID literal (typo, wrong format, etc.)
        return None
    except AttributeError:
        # defensive: handles unusual non-string candidates that
        # uuid.UUID rejects via attribute access on its input
        return None
    except TypeError:
        # defensive: non-str inputs (e.g. dict, list) that bypass the
        # earlier ``isinstance(raw, str)`` guard via duck-typing
        return None
