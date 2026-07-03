"""Pin the canonical Prometheus + Loki event-name constants.

These names form the observability contract that consumer products
register their prefixed instruments against. A rename here is a
cross-product break, so the test freezes the bare values.
"""

from __future__ import annotations

from threetears.agent.skills import metric_names


def test_prometheus_instrument_names() -> None:
    assert metric_names.SKILL_LOAD_TOTAL == "agent_skill_load_total"
    assert metric_names.SKILL_CREATE_TOTAL == "agent_skill_create_total"
    assert metric_names.SKILL_OUTCOME_RECORDED_TOTAL == "agent_skill_outcome_recorded_total"


def test_loki_event_type_names() -> None:
    assert metric_names.EVENT_SKILL_LOADED == "skill.loaded"
    assert metric_names.EVENT_SKILL_CREATED == "skill.created"
    assert metric_names.EVENT_SKILL_INVOKED == "skill.invoked"
    assert metric_names.EVENT_SKILL_OUTCOME_RECORDED == "skill.outcome_recorded"


def test_all_exports_present() -> None:
    for name in metric_names.__all__:
        assert hasattr(metric_names, name), name
    assert "SKILL_OUTCOME_RECORDED_TOTAL" in metric_names.__all__
