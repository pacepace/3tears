"""typed identity events dispatched through langchain custom events.

These ride the same in-process transport as the langgraph-layer events in
:mod:`threetears.langgraph.events`: they dispatch via
``adispatch_custom_event`` inside a langgraph run and surface as
``on_custom_event`` ticks in ``astream_events('v2')``. Consumers parse
them through the shared
:data:`threetears.langgraph.events.default_registry`, which this module
mutates on import (the intention/memory ``events.py`` template).

events defined here (the self-evolution lifecycle):

- :class:`IdentityProposedEvent` -- a new version was PROPOSED and awaits
  consent (tier 1). Feeds the review queue.
- :class:`IdentityConsentedEvent` -- the user CONSENTED to a tier-1
  proposal.
- :class:`IdentityAppliedEvent` -- a version became ACTIVE (tier-1 consent
  OR tier-2 auto-apply; ``auto_applied`` distinguishes). The consumer
  writes the block content back to its prompt cache + refreshes.
- :class:`IdentityRolledBackEvent` -- a prior version was restored as the
  active one.

naming follows the framework convention: ``noun_verb`` with a past-tense
verb. ``user_id`` is a string-form uuid when user-scoped and ``None`` for
an agent-internal / global block (the primitive allows a null grain).
"""

from __future__ import annotations

from typing import Any, Literal

from threetears.langgraph.events import FrameworkEvent, default_registry
from threetears.observe import get_logger

__all__ = [
    "IdentityAppliedEvent",
    "IdentityConsentedEvent",
    "IdentityProposedEvent",
    "IdentityRolledBackEvent",
]

log = get_logger(__name__)


class IdentityProposedEvent(FrameworkEvent):
    """fired when the agent proposes a new identity-block version (tier 1).

    Emitted by ``propose`` for a tier-1 (identity-shaping) block, where the
    version is born ``proposed`` and awaits user consent. Feeds the review
    queue. Tier-2 (routine) blocks auto-apply and emit
    :class:`IdentityAppliedEvent` instead.

    :ivar agent_id: string-form uuid of the proposing agent
    :ivar version_id: string-form uuid of the proposed version
    :ivar block_key: which identity block (``personality`` / ...)
    :ivar user_id: string-form uuid of the owning user, or ``None``
    :ivar rationale_preview: first ~120 chars of the change rationale
    """

    type: Literal["identity_proposed"] = "identity_proposed"
    agent_id: str
    version_id: str
    block_key: str
    user_id: str | None = None
    rationale_preview: str = ""


class IdentityConsentedEvent(FrameworkEvent):
    """fired when the user consents to a proposed tier-1 version.

    Precedes the :class:`IdentityAppliedEvent` the same consent emits.

    :ivar agent_id: string-form uuid of the owning agent
    :ivar version_id: string-form uuid of the consented version
    :ivar block_key: which identity block
    :ivar user_id: string-form uuid of the consenting user, or ``None``
    """

    type: Literal["identity_consented"] = "identity_consented"
    agent_id: str
    version_id: str
    block_key: str
    user_id: str | None = None


class IdentityAppliedEvent(FrameworkEvent):
    """fired when a version becomes the active one for its block.

    Emitted on tier-1 consent AND tier-2 auto-apply (``auto_applied``
    distinguishes). The consumer re-reads ``resolve_active(block_key)`` and
    writes the content back to its prompt cache so the next turn picks it
    up. One event per applied version.

    :ivar agent_id: string-form uuid of the owning agent
    :ivar version_id: string-form uuid of the now-active version
    :ivar block_key: which identity block
    :ivar user_id: string-form uuid of the owning user, or ``None``
    :ivar auto_applied: ``True`` for a tier-2 auto-apply, ``False`` for a
        tier-1 consent-gated apply
    """

    type: Literal["identity_applied"] = "identity_applied"
    agent_id: str
    version_id: str
    block_key: str
    user_id: str | None = None
    auto_applied: bool = False


class IdentityRolledBackEvent(FrameworkEvent):
    """fired when a prior version is restored as the active one.

    Rollback is non-destructive: a NEW version cloning the target's content
    becomes active; the prior active flips to superseded. ``target_version_id``
    is the version whose content was restored.

    :ivar agent_id: string-form uuid of the owning agent
    :ivar version_id: string-form uuid of the NEW active (clone) version
    :ivar block_key: which identity block
    :ivar user_id: string-form uuid of the owning user, or ``None``
    :ivar target_version_id: string-form uuid of the restored prior version
    """

    type: Literal["identity_rolled_back"] = "identity_rolled_back"
    agent_id: str
    version_id: str
    block_key: str
    user_id: str | None = None
    target_version_id: str = ""


def _register_identity_events(registry: Any) -> None:
    """register identity events into ``registry``.

    Invoked at import time against
    :data:`threetears.langgraph.events.default_registry` via
    :meth:`FrameworkEventRegistry.add_framework_defaults_provider`, and
    reused after a test clears the registry. Idempotent: a wire name
    already present is skipped.

    :param registry: registry to populate
    :ptype registry: FrameworkEventRegistry
    :return: nothing
    :rtype: None
    """
    for cls in (
        IdentityProposedEvent,
        IdentityConsentedEvent,
        IdentityAppliedEvent,
        IdentityRolledBackEvent,
    ):
        if cls.model_fields["type"].default in registry.names():
            continue
        registry.register(cls)


default_registry.add_framework_defaults_provider(_register_identity_events)
