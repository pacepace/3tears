"""sender-posture enforcement domain: every NATS receiver says how it authenticates its sender.

a subject proves nothing about who published on it, so a receiver that acts on what arrives has
to declare what it checked. each repo runs :func:`sender_posture_findings` over its own source
with its own shrinking allowlist of audit rows still open; :func:`find_subscribe_sites` lets the
repo's gate prove it walked a non-empty surface. the postures and the marker grammar are in
:mod:`threetears.enforcement.sender_posture.config` and
:mod:`threetears.enforcement.sender_posture.walkers`.
"""

from threetears.enforcement.sender_posture.config import (
    DEFAULT_SITE_CALLEES,
    POSTURES,
    SenderPostureConfig,
)
from threetears.enforcement.sender_posture.walkers import (
    SubscribeSite,
    find_subscribe_sites,
    sender_posture_findings,
)

__all__ = [
    "DEFAULT_SITE_CALLEES",
    "POSTURES",
    "SenderPostureConfig",
    "SubscribeSite",
    "find_subscribe_sites",
    "sender_posture_findings",
]
