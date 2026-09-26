"""unit tests for :mod:`threetears.agent.audit.anonymize`.

the erasure rule for audit records: an audit row is never deleted and no id
on it changes. what erasure does is anonymize the row's ``details``: every
top-level key stays, the structure stays wherever the keys are safe, values
under a SAFE key stay, and the whole value under any other key -- keys a user
chose included -- becomes :data:`ANONYMIZED_MARKER`. the safe list is
explicit, so a key nobody classified is masked -- the fail-safe property the
tests below pin by name.

declarations made through :func:`declare_safe_detail_keys` live in one
process-wide registry. every test that declares uses a prefix minted from a
fresh uuid, so no declaration can reach another test's event types.
"""

from __future__ import annotations

import copy
from typing import Any, get_type_hints
from uuid import uuid4, uuid7

import pytest
from hypothesis import given
from hypothesis import strategies as st

from threetears.agent.audit import (
    ANONYMIZED_MARKER,
    PERSONAL_DETAIL_KEYS,
    SAFE_DETAIL_KEYS,
    anonymize_details,
    anonymize_ip,
    declare_safe_detail_keys,
    is_classified_detail_key,
    safe_detail_keys_for,
)

#: an event type no declaration in this module targets, so only the platform
#: safe set applies to it.
_PLAIN_EVENT = "workspace.fs_write"

#: a key no classification names. chosen to look structural on purpose: a
#: name that merely LOOKS safe is exactly the case the fail-safe must mask.
_UNCLASSIFIED_KEY = "widget_flux_capacitance"


def _fresh_prefix() -> str:
    """
    mint an event-type prefix no other test (or production code) uses.

    :return: a dotted-safe prefix unique to the calling test
    :rtype: str
    """
    return f"anonymizetest{uuid4().hex}"


def _shape(value: Any) -> Any:
    """
    reduce a details value to its shape: keys, container types, None-ness.

    :param value: a details value
    :ptype value: Any
    :return: the same structure with every non-None leaf replaced by ``"leaf"``
    :rtype: Any
    """
    result: Any
    if isinstance(value, dict):
        result = {key: _shape(child) for key, child in value.items()}
    elif isinstance(value, list):
        result = [_shape(child) for child in value]
    elif isinstance(value, tuple):
        result = tuple(_shape(child) for child in value)
    elif value is None:
        result = None
    else:
        result = "leaf"
    return result


def _mask_to_leaf(value: Any) -> Any:
    """
    collapse every subtree under a non-safe key to one leaf, so shapes compare at safe levels.

    :param value: a value under a safe key
    :ptype value: Any
    :return: the same skeleton with each non-safe key's value replaced by ``"leaf"`` (or None)
    :rtype: Any
    """
    result: Any
    if isinstance(value, dict):
        result = {
            key: (_mask_to_leaf(child) if key in SAFE_DETAIL_KEYS else (None if child is None else "leaf"))
            for key, child in value.items()
        }
    elif isinstance(value, list):
        result = [_mask_to_leaf(child) for child in value]
    elif isinstance(value, tuple):
        result = tuple(_mask_to_leaf(child) for child in value)
    else:
        result = value
    return result


class TestMarkerAndClassification:
    """the constants consumers build on."""

    def test_marker_is_the_documented_constant(self) -> None:
        """the marker is fixed text a reader recognises as an erasure."""
        assert ANONYMIZED_MARKER == "[anonymized]"

    def test_safe_and_personal_sets_are_disjoint(self) -> None:
        """a key cannot be both kept and masked; the two records must not overlap."""
        assert SAFE_DETAIL_KEYS
        assert PERSONAL_DETAIL_KEYS
        assert not (SAFE_DETAIL_KEYS & PERSONAL_DETAIL_KEYS)

    def test_a_key_carrying_free_text_is_classified_personal(self) -> None:
        """the tool server's human-readable failure text is personal, not structural."""
        assert "failure_reason" in PERSONAL_DETAIL_KEYS
        assert "failure_reason" not in SAFE_DETAIL_KEYS

    def test_classification_lookup_answers_for_both_sets_and_nothing_else(self) -> None:
        """classified means safe OR personal; an unknown key is unclassified."""
        assert is_classified_detail_key("duration_ms", event_type=_PLAIN_EVENT)
        assert is_classified_detail_key("failure_reason", event_type=_PLAIN_EVENT)
        assert not is_classified_detail_key(_UNCLASSIFIED_KEY, event_type=_PLAIN_EVENT)


class TestAnonymizeDetails:
    """the rule itself: keep keys and the safe skeleton, keep safe values, mask the rest whole."""

    def test_every_top_level_key_and_the_safe_skeleton_are_kept(self) -> None:
        """no top-level key disappears, and structure beneath safe keys keeps its keys and types."""
        details = {
            "tool_name": "fs_write",
            "failure_reason": "could not write /home/alice/notes.md",
            "nested": {"email": "alice@example.com", "count": 3},
            "platform_repointed": {"member_count": 3, "email": "alice@example.com"},
            "tool_names": [{"version": 2, "name": "Bob"}, "fs_edit"],
            "retired_connection_ids": ("a", "b"),
        }

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {
            "tool_name": "fs_write",
            "failure_reason": ANONYMIZED_MARKER,
            "nested": ANONYMIZED_MARKER,
            "platform_repointed": {"member_count": 3, "email": ANONYMIZED_MARKER},
            "tool_names": [{"version": 2, "name": ANONYMIZED_MARKER}, "fs_edit"],
            "retired_connection_ids": ("a", "b"),
        }

    def test_values_under_safe_keys_are_untouched(self) -> None:
        """a safe key's value comes back identical, whatever its type."""
        agent_id = uuid7()
        details = {
            "tool_name": "fs_write",
            "tool_version": "1.2.0",
            "duration_ms": 12.5,
            "bytes_before": 10,
            "bytes_after": 0,
            "version": 4,
            "agent_id": agent_id,
        }

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == details
        assert anonymized["agent_id"] is agent_id

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("boom: alice@example.com", id="text"),
            pytest.param(5551234567, id="integer"),
            pytest.param(True, id="bool"),
            pytest.param(0.5, id="float"),
            pytest.param({"street": "1 Main St", "zip": 12345}, id="dict"),
            pytest.param(["name", "email"], id="list"),
            pytest.param(("name", "email"), id="tuple"),
            pytest.param({}, id="empty-dict"),
            pytest.param([], id="empty-list"),
        ],
    )
    def test_the_whole_value_under_an_unsafe_key_becomes_the_marker(self, value: Any) -> None:
        """whatever sits under an unsafe key -- a leaf or a whole subtree -- is replaced by the marker.

        :param value: the value published under an unsafe key
        :ptype value: Any
        """
        anonymized = anonymize_details({"failure_reason": value}, event_type=_PLAIN_EVENT)

        assert anonymized == {"failure_reason": ANONYMIZED_MARKER}

    def test_keys_a_user_chose_under_an_unsafe_key_do_not_survive(self) -> None:
        """``doc_set`` publishes the user's document under ``value``; its field names are the user's content."""
        details = {"value": {"alice@example.com": {"diagnosis": "flu"}, "Alice Liddell": [1, 2]}}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {"value": ANONYMIZED_MARKER}
        assert "alice" not in repr(anonymized).lower()

    def test_an_unclassified_key_fails_safe_to_the_marker(self) -> None:
        """a key on neither list is masked: nobody classified it, so it cannot leak."""
        details = {_UNCLASSIFIED_KEY: "Alice Liddell, 1 Rabbit Hole"}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {_UNCLASSIFIED_KEY: ANONYMIZED_MARKER}

    def test_an_unclassified_key_nested_under_a_safe_key_fails_safe(self) -> None:
        """a dict under a safe key is judged key by key; its unknown keys are masked."""
        details = {"platform_repointed": {"member_count": 3, _UNCLASSIFIED_KEY: {"name": "Alice"}}}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {"platform_repointed": {"member_count": 3, _UNCLASSIFIED_KEY: ANONYMIZED_MARKER}}

    def test_a_safe_spelling_inside_an_unsafe_subtree_earns_nothing(self) -> None:
        """a user document whose own field happens to be spelled ``status`` is not kept."""
        details = {"value": {"status": "Alice's diagnosis", "version": 2}}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {"value": ANONYMIZED_MARKER}

    def test_a_list_under_a_safe_key_keeps_its_leaves(self) -> None:
        """a list of structural values under a safe key survives whole."""
        ids = [str(uuid7()), str(uuid7())]
        details = {"retired_connection_ids": ids}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {"retired_connection_ids": ids}

    def test_a_list_of_dicts_under_a_safe_key_judges_each_dict_by_its_own_keys(self) -> None:
        """safety does not flow into a dict: each nested dict is judged afresh."""
        details = {"tool_names": [{"version": 2, _UNCLASSIFIED_KEY: "Alice"}]}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {"tool_names": [{"version": 2, _UNCLASSIFIED_KEY: ANONYMIZED_MARKER}]}

    def test_a_tuple_under_a_safe_key_comes_back_a_tuple(self) -> None:
        """the kept skeleton includes the container type, not just the nesting."""
        details = {"tool_names": ("fs_write", "fs_edit")}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {"tool_names": ("fs_write", "fs_edit")}
        assert isinstance(anonymized["tool_names"], tuple)

    @pytest.mark.parametrize("key", ["failure_reason", _UNCLASSIFIED_KEY, "duration_ms"])
    def test_none_stays_none(self, key: str) -> None:
        """None carries nothing to anonymize and stays None under any key, at any safe depth."""
        details = {key: None, "platform_repointed": {key: None}, "tool_names": [None]}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {key: None, "platform_repointed": {key: None}, "tool_names": [None]}

    def test_empty_containers_under_safe_keys_stay_empty(self) -> None:
        """an empty dict or list under a safe key keeps its type."""
        details: dict[str, Any] = {"platform_repointed": {}, "tool_names": [], "retired_connection_ids": ()}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {"platform_repointed": {}, "tool_names": [], "retired_connection_ids": ()}

    def test_empty_details_are_empty(self) -> None:
        """nothing in, nothing out."""
        assert anonymize_details({}, event_type=_PLAIN_EVENT) == {}

    def test_anonymizing_twice_equals_anonymizing_once(self) -> None:
        """a second pass over an erased record changes nothing."""
        details = {
            "tool_name": "fs_write",
            "failure_reason": "alice@example.com",
            "value": {"status": "x", "n": [1, None, {"k": "v"}]},
        }

        once = anonymize_details(details, event_type=_PLAIN_EVENT)
        twice = anonymize_details(once, event_type=_PLAIN_EVENT)

        assert twice == once

    def test_the_input_is_not_mutated(self) -> None:
        """the function is pure: the caller's dict comes back exactly as it went in."""
        details = {"failure_reason": "alice", "value": {"a": [1, {"b": "c"}]}, "tool_name": "t"}
        before = copy.deepcopy(details)

        anonymize_details(details, event_type=_PLAIN_EVENT)

        assert details == before

    def test_the_result_shares_no_container_with_the_input(self) -> None:
        """a caller editing the result must not reach back into the original."""
        details = {"platform_repointed": {"member_count": [1, 2]}}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)
        anonymized["platform_repointed"]["member_count"].append(3)

        assert details == {"platform_repointed": {"member_count": [1, 2]}}


class TestDeclaredExtensions:
    """a package or service widens the safe set for its own event types, and only those."""

    def test_a_declared_key_is_kept_for_its_event_type(self) -> None:
        """the declaration is honoured for the prefix's own event types."""
        prefix = _fresh_prefix()
        declare_safe_detail_keys(prefix, [_UNCLASSIFIED_KEY])

        anonymized = anonymize_details({_UNCLASSIFIED_KEY: "kept"}, event_type=f"{prefix}.response.submit")

        assert anonymized == {_UNCLASSIFIED_KEY: "kept"}

    def test_a_declared_key_applies_to_the_bare_prefix_event_type(self) -> None:
        """the prefix itself is an event type in its own family."""
        prefix = f"{_fresh_prefix()}.response"
        declare_safe_detail_keys(prefix, [_UNCLASSIFIED_KEY])

        assert _UNCLASSIFIED_KEY in safe_detail_keys_for(prefix)

    def test_a_declared_key_does_not_leak_to_other_event_types(self) -> None:
        """another family's event, and the platform set, are untouched by the declaration."""
        prefix = _fresh_prefix()
        declare_safe_detail_keys(prefix, [_UNCLASSIFIED_KEY])

        anonymized = anonymize_details({_UNCLASSIFIED_KEY: "Alice"}, event_type=_PLAIN_EVENT)

        assert anonymized == {_UNCLASSIFIED_KEY: ANONYMIZED_MARKER}
        assert _UNCLASSIFIED_KEY not in SAFE_DETAIL_KEYS

    def test_a_prefix_matches_whole_segments_only(self) -> None:
        """``survey`` covers ``survey.x`` but not ``surveyor.x``."""
        prefix = _fresh_prefix()
        declare_safe_detail_keys(prefix, [_UNCLASSIFIED_KEY])

        assert _UNCLASSIFIED_KEY not in safe_detail_keys_for(f"{prefix}or.response.submit")

    def test_a_narrower_prefix_does_not_widen_its_parent(self) -> None:
        """a declaration on ``a.b`` says nothing about ``a.c``."""
        root = _fresh_prefix()
        declare_safe_detail_keys(f"{root}.response", [_UNCLASSIFIED_KEY])

        assert _UNCLASSIFIED_KEY in safe_detail_keys_for(f"{root}.response.submit")
        assert _UNCLASSIFIED_KEY not in safe_detail_keys_for(f"{root}.template.create")

    def test_declared_keys_count_as_classified_for_their_family_only(self) -> None:
        """the gate's lookup honours the declaration exactly where anonymization does."""
        prefix = _fresh_prefix()
        declare_safe_detail_keys(prefix, [_UNCLASSIFIED_KEY])

        assert is_classified_detail_key(_UNCLASSIFIED_KEY, event_type=f"{prefix}.x")
        assert not is_classified_detail_key(_UNCLASSIFIED_KEY, event_type=_PLAIN_EVENT)

    def test_declaring_the_same_keys_twice_is_harmless(self) -> None:
        """a module imported twice, or two services declaring one key, converge."""
        prefix = _fresh_prefix()
        declare_safe_detail_keys(prefix, [_UNCLASSIFIED_KEY])
        declare_safe_detail_keys(prefix, [_UNCLASSIFIED_KEY, "other_count"])

        assert {_UNCLASSIFIED_KEY, "other_count"} <= safe_detail_keys_for(f"{prefix}.x")

    def test_a_family_may_keep_a_personal_key_for_its_own_events_only(self) -> None:
        """``reason`` is an enum in one family and free text in the rest; only the first keeps it."""
        prefix = _fresh_prefix()
        declare_safe_detail_keys(prefix, ["reason"])

        kept = anonymize_details({"reason": "target_blocked"}, event_type=f"{prefix}.stop")
        masked = anonymize_details({"reason": "SMTP refused alice@example.com"}, event_type=_PLAIN_EVENT)

        assert kept == {"reason": "target_blocked"}
        assert masked == {"reason": ANONYMIZED_MARKER}

    def test_a_refused_declaration_declares_nothing(self) -> None:
        """one bad key in a batch leaves the registry as it was."""
        prefix = _fresh_prefix()
        with pytest.raises(ValueError):
            declare_safe_detail_keys(prefix, [_UNCLASSIFIED_KEY, ""])

        assert _UNCLASSIFIED_KEY not in safe_detail_keys_for(f"{prefix}.x")

    def test_a_platform_family_is_declared_where_the_eraser_can_see_it(self) -> None:
        """the hub erases identity's events, so identity's enum ``reason`` is declared in 3tears."""
        kept = anonymize_details({"reason": "target_blocked"}, event_type="identity.impersonation.stop")
        masked = anonymize_details({"reason": "relay said: bob@example.com"}, event_type="identity.email.send_failure")

        assert kept == {"reason": "target_blocked"}
        assert masked == {"reason": ANONYMIZED_MARKER}

    @pytest.mark.parametrize("prefix", ["", ".survey", "survey.", "survey..response", "survey response"])
    def test_a_malformed_prefix_is_refused(self, prefix: str) -> None:
        """a prefix is one or more non-empty dotted segments, nothing else."""
        with pytest.raises(ValueError):
            declare_safe_detail_keys(prefix, [_UNCLASSIFIED_KEY])

    @pytest.mark.parametrize("key", ["", " "])
    def test_a_blank_key_is_refused(self, key: str) -> None:
        """an empty key names nothing and is a caller bug."""
        with pytest.raises(ValueError):
            declare_safe_detail_keys(_fresh_prefix(), [key])

    def test_a_bare_string_is_refused_as_the_key_collection(self) -> None:
        """``"user_count"`` iterates as characters; that is never what the caller meant."""
        with pytest.raises(TypeError):
            declare_safe_detail_keys(_fresh_prefix(), "user_count")


class TestAnonymizeIp:
    """the single rule for an ip_address column."""

    @pytest.mark.parametrize("address", ["203.0.113.7", "2001:db8::1", "", None])
    def test_an_address_becomes_none(self, address: str | None) -> None:
        """every address, and the absence of one, anonymizes to None."""
        assert anonymize_ip(address) is None

    def test_the_result_is_typed_as_the_column_so_a_caller_can_bind_it(self) -> None:
        """``row.ip_address = anonymize_ip(row.ip_address)`` must type-check without an ignore.

        a function annotated ``-> None`` makes binding its result a mypy
        ``func-returns-value`` error; the rule's result is a column value, so it is typed as one.
        """
        assert get_type_hints(anonymize_ip)["return"] == str | None


#: keys the property test draws from: the platform's own safe and personal
#: spellings, so generated trees exercise both branches, plus free text for the
#: unclassified case.
_KEYS = st.one_of(
    st.sampled_from(sorted(SAFE_DETAIL_KEYS)),
    st.sampled_from(sorted(PERSONAL_DETAIL_KEYS)),
    st.text(min_size=1, max_size=12),
)
_LEAVES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False),
    st.text(max_size=20),
)
_TREES = st.recursive(
    _LEAVES,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(_KEYS, children, max_size=4),
    ),
    max_leaves=25,
)
_DETAILS = st.dictionaries(_KEYS, _TREES, max_size=6)


class TestProperties:
    """the rule's promises, over arbitrary JSON-shaped details."""

    @given(details=_DETAILS)
    def test_every_top_level_key_survives(self, details: dict[str, Any]) -> None:
        """erasure never deletes a key."""
        assert anonymize_details(details, event_type=_PLAIN_EVENT).keys() == details.keys()

    @given(details=_DETAILS)
    def test_the_skeleton_under_safe_keys_is_preserved(self, details: dict[str, Any]) -> None:
        """beneath a safe key the containers, their keys and every None survive."""
        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        for key in details.keys() & SAFE_DETAIL_KEYS:
            assert _shape(_mask_to_leaf(anonymized[key])) == _shape(_mask_to_leaf(details[key]))

    @given(details=_DETAILS)
    def test_anonymization_is_idempotent(self, details: dict[str, Any]) -> None:
        """a second pass is a no-op."""
        once = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymize_details(once, event_type=_PLAIN_EVENT) == once

    @given(details=_DETAILS)
    def test_nothing_under_a_non_safe_key_survives(self, details: dict[str, Any]) -> None:
        """the whole value under any top-level key that is not safe is the marker, or None if it was None."""
        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        for key, value in anonymized.items():
            if key in SAFE_DETAIL_KEYS:
                continue
            assert value == (None if details[key] is None else ANONYMIZED_MARKER)

    @given(details=_DETAILS)
    def test_the_input_is_never_mutated(self, details: dict[str, Any]) -> None:
        """purity holds for every shape, not just the hand-written ones."""
        before = copy.deepcopy(details)

        anonymize_details(details, event_type=_PLAIN_EVENT)

        assert details == before
