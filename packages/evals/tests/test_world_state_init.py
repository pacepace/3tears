"""Unit tests for :func:`threetears.evals.contracts.world_state.init_world`.

A seed is the namespace's literal initial state. Nothing in it is resolved against the subject:
the engine never learns what a subject carries, so a string that once named a field of one host's
subject is now data like any other.
"""

from __future__ import annotations

from threetears.evals.contracts.models import WorldSeed
from threetears.evals.contracts.world_state import init_world

# =============================================================================
# Literal seed values pass through verbatim
# =============================================================================


def test_literal_dict_namespace_copied():
    seed = WorldSeed(namespaces={"calendar": {"events": [], "declined": []}, "inbox": {"messages": []}})
    state = init_world(seed)
    assert state.namespace("calendar") == {"events": [], "declined": []}
    assert state.namespace("inbox") == {"messages": []}


def test_literal_with_nested_lists_and_scalars():
    seed = WorldSeed(
        namespaces={
            "calendar": {
                "events": [],
                "next_event": None,
                "busy_ratio": 0.75,
                "tags_allowed": ["work", "travel"],
            }
        }
    )
    state = init_world(seed)
    calendar = state.namespace("calendar")
    assert calendar["events"] == []
    assert calendar["next_event"] is None
    assert calendar["busy_ratio"] == 0.75
    assert calendar["tags_allowed"] == ["work", "travel"]


def test_returned_state_is_independent_of_seed():
    """Mutations on the returned state must not reach the seed, at any depth."""
    seed = WorldSeed(namespaces={"calendar": {"events": [{"title": "standup", "attendees": ["a"]}]}})
    state = init_world(seed)
    state.namespace("calendar")["events"].append({"title": "lunch"})
    state.namespace("calendar")["events"][0]["attendees"].append("b")

    again = init_world(seed)
    assert again.namespace("calendar")["events"] == [{"title": "standup", "attendees": ["a"]}]


def test_a_string_that_once_named_a_subject_field_is_literal_data():
    """No string is a reference any more: a former seed reference stays the string it is."""
    seed = WorldSeed(namespaces={"calendar": {"events": "from_subject_catalog"}})
    state = init_world(seed)
    assert state.namespace("calendar") == {"events": "from_subject_catalog"}


# =============================================================================
# A top-level non-dict is promoted, so every namespace is dict-shaped
# =============================================================================


def test_top_level_list_promoted_into_value_sub_key():
    """A top-level list is rare; wrap into {'value': [...]} so the shape stays dict."""
    seed = WorldSeed(namespaces={"calendar": [1, 2, 3]})
    state = init_world(seed)
    assert state.namespace("calendar") == {"value": [1, 2, 3]}


def test_top_level_string_promoted_like_any_other_scalar():
    """A top-level string is data too, promoted exactly as a number is."""
    seed = WorldSeed(namespaces={"calendar": "closed", "counter": 3})
    state = init_world(seed)
    assert state.namespace("calendar") == {"value": "closed"}
    assert state.namespace("counter") == {"value": 3}


def test_a_promoted_namespace_is_dict_shaped_for_record_call():
    """After promotion, record_call must work — the namespace is a dict."""
    seed = WorldSeed(namespaces={"calendar": "closed"})
    state = init_world(seed)
    # Would raise if the namespace were the bare string.
    state.record_call("calendar", "search", {"query": "q"})
    assert state.namespace("calendar")["calls"] == [{"action": "search", "params": {"query": "q"}}]


# =============================================================================
# Empty seed
# =============================================================================


def test_empty_seed_yields_empty_state():
    state = init_world(WorldSeed())
    assert state.namespaces == {}
