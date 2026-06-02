"""agent-wake package migrations.

Single entry point :func:`register` wires the package's versioned
migration callables into a shared
:class:`~threetears.core.data.migrations.runner.MigrationRunner`.

This package owns three tables in every agent schema (per the
2026-05-19 PLACEMENT revision; the original six-table plan was
collapsed -- the junction tables and ``wake_pre_check_types`` were
dropped in favor of nullable FK columns and ordinary TearsTool
subclasses in ``3tears-agent-tools``):

- ``agent_wake_schedules`` -- one row per active wake schedule for a
  conversation. Carries nullable ``skill_id`` FK to
  ``agent_skills.skill_id``.
- ``wake_fires`` -- one row per wake fire (history). Supports both
  schedule-source and webhook-source fires via the exclusive-OR CHECK
  constraint on ``(schedule_id, webhook_subscription_id)``. The
  ``webhook_subscription_id`` FK is retro-added in v003 once
  ``webhook_subscriptions`` exists.
- ``webhook_subscriptions`` -- one row per inbound webhook
  subscription. Carries nullable ``default_skill_id`` FK to
  ``agent_skills.skill_id``.

The package declares
``depends_on=("conversations", "agent_skills")`` because:

1. Every table carries a ``conversation_id`` column (denormalised; no
   FK because ``conversations`` has composite PK
   ``(agent_id, conversation_id)`` and no standalone
   ``UNIQUE (conversation_id)`` -- a single-column FK is not legal).
   The dependency declaration ensures the conversations migrations
   apply before the wake tables are created.
2. ``agent_wake_schedules.skill_id`` and
   ``webhook_subscriptions.default_skill_id`` reference
   ``agent_skills.skill_id`` via the cross-package standalone
   ``UNIQUE (skill_id)`` constraint added in agent-skills v001. The
   dependency declaration ensures that constraint exists before the
   wake tables declare the FK.

Version history:

- v001 creates ``agent_wake_schedules`` + indexes.
- v002 creates ``wake_fires`` + indexes (the
  ``webhook_subscription_id`` FK is added by v003 because the target
  table does not yet exist at this point).
- v003 creates ``webhook_subscriptions`` + indexes + retro-adds the FK
  on ``wake_fires.webhook_subscription_id``.
- v004 extends ``wake_fires.status`` CHECK to accept the new
  ``'dispatching'`` placeholder value so ``create_dispatching`` can
  write a distinct in-flight status instead of pre-claiming the
  terminal ``'fired'``.
- v005 opens the ``webhook_subscriptions.verification_scheme`` CHECK
  constraint: replaces the hardcoded ``IN ('generic_hmac_sha256')``
  predicate (which made the receiver's pluggable verifier registry
  useless) with a slug-format guard (``^[a-z0-9_]+$``, length 1-64).
  Registered-scheme validation happens at handle time in the
  receiver.
- v006 adds ``agent_wake_schedules.include_conversation_history``
  (``BOOLEAN NOT NULL DEFAULT true``): a per-schedule switch for
  whether a fire carries the conversation's recent history into the
  wake's LLM context. Default ``true`` preserves the prior always-on
  behavior; independent of the attached skill's persona setting.
"""

from __future__ import annotations

from threetears.agent.wake.migrations.v001_create_agent_wake_schedules import (
    create_agent_wake_schedules,
)
from threetears.agent.wake.migrations.v002_create_wake_fires import (
    create_wake_fires,
)
from threetears.agent.wake.migrations.v003_create_webhook_subscriptions import (
    create_webhook_subscriptions,
)
from threetears.agent.wake.migrations.v004_add_dispatching_status import (
    add_dispatching_status,
)
from threetears.agent.wake.migrations.v005_open_verification_scheme_check import (
    open_verification_scheme_check,
)
from threetears.agent.wake.migrations.v006_add_include_conversation_history import (
    add_include_conversation_history,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "agent_wake"


def register(runner: MigrationRunner) -> PackageMigrations:
    """Register agent-wake migrations with ``runner``.

    Produces an agent-scoped :class:`PackageMigrations` declaring
    ``depends_on=("conversations", "agent_skills")`` so the upstream
    tables exist before this package's tables are created. Attaches
    every migration in version order and calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
        depends_on=("conversations", "agent_skills"),
    )
    pkg.version(1)(create_agent_wake_schedules)
    pkg.version(2)(create_wake_fires)
    pkg.version(3)(create_webhook_subscriptions)
    pkg.version(4)(add_dispatching_status)
    pkg.version(5)(open_verification_scheme_check)
    pkg.version(6)(add_include_conversation_history)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_dispatching_status",
    "add_include_conversation_history",
    "create_agent_wake_schedules",
    "create_wake_fires",
    "create_webhook_subscriptions",
    "open_verification_scheme_check",
    "register",
]
