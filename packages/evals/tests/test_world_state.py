"""Unit tests for the WorldState eval substrate."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.evals.contracts.world_state import WorldState

# =============================================================================
# Namespace accessors
# =============================================================================


def test_namespace_lazy_creation_returns_reference():
    """namespace() returns a mutable reference; mutations persist."""
    state = WorldState()
    ns = state.namespace("shop")
    assert ns == {}
    ns["catalog"] = [{"title": "A"}]
    # Re-accessing yields the same dict (mutation visible)
    assert state.namespace("shop")["catalog"] == [{"title": "A"}]


def test_namespace_isolates_tools():
    """Each tool name gets its own namespace."""
    state = WorldState()
    state.namespace("shop")["cart"] = []
    state.namespace("chat")["messages"] = []
    assert "cart" not in state.namespace("chat")
    assert "messages" not in state.namespace("shop")


def test_record_call_appends_to_calls_list():
    """record_call appends a {action, params} entry under the tool's calls list."""
    state = WorldState()
    state.record_call("shop", "search", {"query": "kitchen"})
    state.record_call("shop", "add_item", {"item_ref": "r1"})
    calls = state.namespace("shop")["calls"]
    assert len(calls) == 2
    assert calls[0] == {"action": "search", "params": {"query": "kitchen"}}
    assert calls[1] == {"action": "add_item", "params": {"item_ref": "r1"}}


def test_record_call_copies_params_to_prevent_post_mutation():
    """Recorded params should be insulated from caller-side mutation after the call."""
    state = WorldState()
    params = {"query": "kitchen"}
    state.record_call("shop", "search", params)
    params["query"] = "outdoor"  # caller mutates after recording
    assert state.namespace("shop")["calls"][0]["params"] == {"query": "kitchen"}


# =============================================================================
# Warn-once mechanism
# =============================================================================


# =============================================================================
# Serialization round-trip
# =============================================================================


def test_serialize_round_trip_preserves_namespaces():
    """serialize -> deserialize produces an equivalent state."""
    state = WorldState()
    state.namespace("shop")["catalog"] = [{"title": "A", "maker": "X"}]
    state.namespace("shop")["cart"] = []
    state.namespace("chat")["messages"] = [{"content": "hello"}]
    state.record_call("shop", "search", {"query": "kitchen"})

    data = state.serialize()
    restored = WorldState.deserialize(data)

    assert restored.namespace("shop")["catalog"] == [{"title": "A", "maker": "X"}]
    assert restored.namespace("shop")["cart"] == []
    assert restored.namespace("chat")["messages"] == [{"content": "hello"}]
    # Calls list survives.
    calls = restored.namespace("shop")["calls"]
    assert calls == [{"action": "search", "params": {"query": "kitchen"}}]


def test_serialize_is_json_safe():
    """The serialized dict round-trips through JSON without loss."""
    import json

    state = WorldState()
    state.namespace("shop")["cart"] = [{"title": "A", "tags": ["kitchen", "garden"]}]
    state.record_call("shop", "add_item", {"item_ref": "r1"})

    data = state.serialize()
    blob = json.dumps(data)
    restored = WorldState.deserialize(json.loads(blob))

    assert restored.namespace("shop")["cart"][0]["tags"] == ["kitchen", "garden"]


def test_deserialize_with_empty_dict_yields_empty_state():
    """An empty dict deserializes to an empty state — the runner's initial seed."""
    state = WorldState.deserialize({"namespaces": {}})
    assert state.namespace("shop") == {}


def test_deserialize_refuses_a_key_the_state_does_not_declare():
    """A serialized world is read as strictly as every stored eval shape: an unknown key is refused."""
    with pytest.raises(ValidationError, match="ledger"):
        WorldState.deserialize({"namespaces": {}, "global_calls": [], "ledger": []})


# =============================================================================
