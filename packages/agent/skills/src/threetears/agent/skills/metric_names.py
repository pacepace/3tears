"""Canonical Prometheus + Loki event-name constants for skills observability.

Consumers (the host that wires these tools into the personality
graph) register Prometheus instruments using these names with whatever
product-specific prefix they need (``myproduct_skill_load_total`` etc.).
Keeping the bare names here means one canonical source for the
instrumentation contract across every product that consumes the
``3tears-agent-skills`` package.

The Loki event-type values match the structured-log event_type field
the consumer emits at instrumentation points. Co-locating them with the
Prometheus names keeps the two halves of the observability story
single-sourced.
"""

from __future__ import annotations

__all__ = [
    "EVENT_SKILL_CREATED",
    "EVENT_SKILL_INVOKED",
    "EVENT_SKILL_LOADED",
    "EVENT_SKILL_OUTCOME_RECORDED",
    "SKILL_CREATE_TOTAL",
    "SKILL_LOAD_TOTAL",
    "SKILL_OUTCOME_RECORDED_TOTAL",
]


# Prometheus instrument names (consumer adds its own product prefix). The
# ``SKILL_LOAD_TOTAL`` instrument carries two labels:
# ``source`` (``'wake'`` | ``'invoke'``) and ``outcome``
# (``'success'`` | ``'failure'`` | ``'unknown'``). The ``SKILL_CREATE_TOTAL``
# instrument is unlabelled -- creates are uniformly successful or they
# raise; the consumer's exception handling decides whether to count a
# failed create.
SKILL_LOAD_TOTAL = "agent_skill_load_total"
SKILL_CREATE_TOTAL = "agent_skill_create_total"


# Counts outcome attributions for an active skill. Two labels:
# ``outcome``, carrying the ``SkillOutcome`` value set, and ``source``,
# carrying the ``OutcomeSource`` value set -- both defined in
# ``types.py`` and deliberately not re-spelled here. In practice
# ``source`` is ``'agent_tool'`` today: it is the only value anything
# writes, ``'agent_marker'`` is a retired path kept for historical
# rows, and ``'user_feedback'`` is reserved. Distinct from
# ``SKILL_LOAD_TOTAL``'s ``outcome`` label: load-time outcome carries
# ``'unknown'`` for "not attributed yet" whereas this instrument only
# ever records a resolved outcome.
SKILL_OUTCOME_RECORDED_TOTAL = "agent_skill_outcome_recorded_total"


# Loki structured-log event_type values. The consumer emits log lines
# with ``extra={"extra_data": {"event_type": EVENT_SKILL_LOADED, ...}}``
# so the LogQL queries on the dashboard side stay portable across
# product builds.
EVENT_SKILL_LOADED = "skill.loaded"
EVENT_SKILL_CREATED = "skill.created"
EVENT_SKILL_INVOKED = "skill.invoked"
EVENT_SKILL_OUTCOME_RECORDED = "skill.outcome_recorded"
