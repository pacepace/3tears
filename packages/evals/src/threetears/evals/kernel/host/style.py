"""Retired: a host's tone register, which only the retired writers' prompts read.

The memo writer and the report planner each took one engine-owned sentence into their prompt from
:attr:`StyleProfile.tone_register`. Both are retired, and the report writer reads no host style: its words are the skill
files. :class:`StyleProfile` and :data:`ToneRegister` stay importable from :mod:`threetears.evals.kernel.host` for one
minor, warning on every read, and ``HostProfile(style=...)`` warns too; the next minor removes them and this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Retired: the registers a host could pick from. Nothing reads one.
ToneRegister = Literal["neutral", "executive", "technical"]


@dataclass(frozen=True)
class StyleProfile:
    """Retired: one host's tone register, which nothing reads now that the writers that read it are retired."""

    tone_register: ToneRegister = "neutral"
    """Which register the retired writers wrote in. Nothing reads it."""


__all__ = [
    "StyleProfile",
    "ToneRegister",
]
