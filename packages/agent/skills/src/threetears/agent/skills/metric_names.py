"""Canonical Prometheus + Loki event-name constants for skills observability.

Consumers (the metallm shard that wires these tools into the personality
graph) register Prometheus instruments using these names with whatever
product-specific prefix they need (``metallm_skill_load_total`` etc.).
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


# Loki structured-log event_type values. The consumer emits log lines
# with ``extra={"extra_data": {"event_type": EVENT_SKILL_LOADED, ...}}``
# so the LogQL queries on the dashboard side stay portable across
# product builds.
EVENT_SKILL_LOADED = "skill.loaded"
EVENT_SKILL_CREATED = "skill.created"
EVENT_SKILL_INVOKED = "skill.invoked"
EVENT_SKILL_OUTCOME_RECORDED = "skill.outcome_recorded"
