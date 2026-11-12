"""unit tests for :mod:`threetears.agent.audit.anonymize`.

the erasure rule for audit records: an audit row is never deleted and no id
on it changes. Pace's ruling (2026-11-12, a contract change by owner ruling,
not a weakened test): audit rows hold ids, never a person's details; ids, the
ip address and free text stay as written. So :func:`anonymize_details` masks
the whole value under a key that names a person
(:data:`IDENTIFYING_DETAIL_KEYS`, at any depth, but a family's allowance) and
keeps every other key and value as written; :func:`refuse_identifying_details`
refuses such a key when an event is published or stored. Before the ruling the
rule masked every key not on the safe list; the tests that pinned that are
rewritten here to the ruling. The safe and personal lists remain the
classification every published key must be in (the gate
``test_audit_details_keys_are_classified``).

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
    IDENTIFYING_DETAIL_KEYS,
    PERSONAL_DETAIL_KEYS,
    SAFE_DETAIL_KEYS,
    IdentifyingAuditDetail,
    anonymize_details,
    anonymize_ip,
    declare_safe_detail_keys,
    identifying_keys_for,
    is_classified_detail_key,
    refuse_identifying_details,
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


class TestMarkerAndClassification:
    """the constants consumers build on."""

    def test_marker_is_the_documented_constant(self) -> None:
        """the marker is fixed text a reader recognises as an erasure."""
        assert ANONYMIZED_MARKER == "[anonymized]"

    def test_marker_has_one_source(self) -> None:
        """the audit package re-exports the platform marker; it does not define a second one."""
        from threetears.observe.erasure import ANONYMIZED_MARKER as platform_marker

        assert ANONYMIZED_MARKER is platform_marker

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
    """the rule itself (Pace 2026-11-12): every key and value as written, but the whole value under a key that names a
    person, at any depth."""

    def test_an_identifying_key_is_masked_at_any_depth_and_everything_else_kept(self) -> None:
        """no key disappears; free text, ids and unclassified keys stay as written; every identifying key's value,
        top-level, nested in a dict or in a list of dicts, becomes the marker."""
        member = str(uuid7())
        details = {
            "tool_name": "fs_write",
            "failure_reason": "could not write /home/alice/notes.md",
            "nested": {"email": "alice@example.com", "count": 3},
            "platform_repointed": {"member_count": 3, "member_id": member, "display_name": "Alice Liddell"},
            "tool_names": [{"version": 2, "full_name": "Bob"}, "fs_edit"],
            "retired_connection_ids": ("a", "b"),
            _UNCLASSIFIED_KEY: "kept as written",
            "subject": "alice@idp.example",
        }

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {
            "tool_name": "fs_write",
            "failure_reason": "could not write /home/alice/notes.md",
            "nested": {"email": ANONYMIZED_MARKER, "count": 3},
            "platform_repointed": {"member_count": 3, "member_id": member, "display_name": ANONYMIZED_MARKER},
            "tool_names": [{"version": 2, "full_name": ANONYMIZED_MARKER}, "fs_edit"],
            "retired_connection_ids": ("a", "b"),
            _UNCLASSIFIED_KEY: "kept as written",
            "subject": ANONYMIZED_MARKER,
        }

    def test_ids_and_values_under_safe_keys_are_untouched(self) -> None:
        """a safe key's value comes back identical, whatever its type: erasure never changes an id."""
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
            pytest.param("alice@example.com", id="text"),
            pytest.param(5551234567, id="integer"),
            pytest.param({"street": "1 Main St", "zip": 12345}, id="dict"),
            pytest.param(["alice", "bob"], id="list"),
            pytest.param(("alice", "bob"), id="tuple"),
            pytest.param({}, id="empty-dict"),
        ],
    )
    def test_the_whole_value_under_an_identifying_key_becomes_the_marker(self, value: Any) -> None:
        """whatever sits under an identifying key -- a leaf or a whole subtree -- is replaced by the marker."""
        anonymized = anonymize_details({"email": value}, event_type=_PLAIN_EVENT)

        assert anonymized == {"email": ANONYMIZED_MARKER}

    @pytest.mark.parametrize("key", sorted(IDENTIFYING_DETAIL_KEYS))
    def test_every_identifying_key_is_masked(self, key: str) -> None:
        """each key of the identifying set is masked on an event type no family allows it for."""
        assert anonymize_details({key: "Alice"}, event_type=_PLAIN_EVENT) == {key: ANONYMIZED_MARKER}

    def test_free_text_and_personal_keys_that_name_no_one_stay_as_written(self) -> None:
        """the ruling keeps free text: a failure reason, a description, a user's document value."""
        details = {
            "failure_reason": "boom",
            "description": "the quarterly group",
            "value": {"status": "x", "rows": [1, 2]},
            "reason": "SMTP refused",
        }

        assert anonymize_details(details, event_type=_PLAIN_EVENT) == details

    def test_a_familys_allowance_keeps_its_resource_name_for_its_events_alone(self) -> None:
        """a workspace's own name is a resource name on workspace.create / delete; elsewhere ``name`` is masked."""
        for event_type in ("workspace.create", "workspace.delete"):
            assert anonymize_details({"name": "taxes"}, event_type=event_type) == {"name": "taxes"}
            assert "name" not in identifying_keys_for(event_type)
        assert anonymize_details({"name": "Alice"}, event_type=_PLAIN_EVENT) == {"name": ANONYMIZED_MARKER}
        assert anonymize_details({"name": "Alice"}, event_type="workspace.fs_write") == {"name": ANONYMIZED_MARKER}

    def test_a_tuple_comes_back_a_tuple(self) -> None:
        """the kept skeleton includes the container type, not just the nesting."""
        details = {"tool_names": ("fs_write", {"login": "alice"})}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {"tool_names": ("fs_write", {"login": ANONYMIZED_MARKER})}
        assert isinstance(anonymized["tool_names"], tuple)

    @pytest.mark.parametrize("key", ["email", _UNCLASSIFIED_KEY, "duration_ms"])
    def test_none_stays_none(self, key: str) -> None:
        """None carries nothing to anonymize and stays None under any key, at any depth."""
        details = {key: None, "platform_repointed": {key: None}, "tool_names": [None]}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymized == {key: None, "platform_repointed": {key: None}, "tool_names": [None]}

    def test_empty_details_are_empty(self) -> None:
        """nothing in, nothing out."""
        assert anonymize_details({}, event_type=_PLAIN_EVENT) == {}

    def test_anonymizing_twice_equals_anonymizing_once(self) -> None:
        """a second pass over an erased record changes nothing."""
        details = {"tool_name": "fs_write", "email": "alice@example.com", "value": {"name": "x", "n": [1, None]}}

        once = anonymize_details(details, event_type=_PLAIN_EVENT)
        twice = anonymize_details(once, event_type=_PLAIN_EVENT)

        assert twice == once

    def test_the_input_is_not_mutated(self) -> None:
        """the function is pure: the caller's dict comes back exactly as it went in."""
        details = {"email": "alice", "value": {"a": [1, {"username": "c"}]}, "tool_name": "t"}
        before = copy.deepcopy(details)

        anonymize_details(details, event_type=_PLAIN_EVENT)

        assert details == before

    def test_the_result_shares_no_container_with_the_input(self) -> None:
        """a caller editing the result must not reach back into the original."""
        details = {"platform_repointed": {"member_count": [1, 2]}}

        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)
        anonymized["platform_repointed"]["member_count"].append(3)

        assert details == {"platform_repointed": {"member_count": [1, 2]}}


class TestRefuseIdentifyingDetails:
    """the write-time rule: an event whose details name a person is refused, at any depth."""

    @pytest.mark.parametrize(
        "details",
        [
            pytest.param({"email": "a@b.c"}, id="top"),
            pytest.param({"target": {"user_email": "a@b.c"}}, id="nested"),
            pytest.param({"members": [{"member_id": "x", "member_email": "a@b.c"}]}, id="list"),
            pytest.param({"pairs": ({"external_user_id": "U1"},)}, id="tuple"),
        ],
    )
    def test_an_identifying_key_anywhere_is_refused(self, details: dict[str, Any]) -> None:
        """Flips if a key naming a person passes at any depth."""
        with pytest.raises(IdentifyingAuditDetail) as refused:
            refuse_identifying_details(details, event_type=_PLAIN_EVENT)
        assert refused.value.event_type == _PLAIN_EVENT
        assert refused.value.key in IDENTIFYING_DETAIL_KEYS

    def test_ids_free_text_and_a_familys_allowance_pass(self) -> None:
        """Flips if ids, free text or an allowed family's resource name are refused."""
        refuse_identifying_details(
            {"member_id": str(uuid7()), "failure_reason": "boom alice@example.com", "count": 3}, event_type=_PLAIN_EVENT
        )
        refuse_identifying_details({"name": "taxes", "workspace_resource_id": "w"}, event_type="workspace.create")

    def test_a_value_already_anonymized_or_none_passes(self) -> None:
        """Flips if an erased row (the marker under an identifying key) or an absent value cannot be stored."""
        refuse_identifying_details({"email": ANONYMIZED_MARKER, "subject": None}, event_type=_PLAIN_EVENT)

    def test_the_identifying_keys_are_personal(self) -> None:
        """the identifying set is a subset of the personal one: every key in it was classified personal."""
        assert IDENTIFYING_DETAIL_KEYS <= PERSONAL_DETAIL_KEYS


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

        assert _UNCLASSIFIED_KEY not in safe_detail_keys_for(_PLAIN_EVENT)
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

    def test_a_family_may_classify_a_personal_key_safe_for_its_own_events_only(self) -> None:
        """``reason`` is an enum in one family and free text in the rest; only the first counts it safe."""
        prefix = _fresh_prefix()
        declare_safe_detail_keys(prefix, ["reason"])

        assert "reason" in safe_detail_keys_for(f"{prefix}.stop")
        assert "reason" not in safe_detail_keys_for(_PLAIN_EVENT)

    def test_a_refused_declaration_declares_nothing(self) -> None:
        """one bad key in a batch leaves the registry as it was."""
        prefix = _fresh_prefix()
        with pytest.raises(ValueError):
            declare_safe_detail_keys(prefix, [_UNCLASSIFIED_KEY, ""])

        assert _UNCLASSIFIED_KEY not in safe_detail_keys_for(f"{prefix}.x")

    def test_a_platform_family_is_declared_where_every_process_can_see_it(self) -> None:
        """identity's enum ``reason`` is declared in 3tears: safe on its family, free text elsewhere."""
        assert "reason" in safe_detail_keys_for("identity.impersonation.stop")
        assert "reason" not in safe_detail_keys_for("identity.email.send_failure")

    def test_an_invitations_status_is_safe_for_the_hubs_user_events_alone(self) -> None:
        """``invitation`` is emailed / not_emailed / not_sent on the hub's ``user.create`` and ``user.invite``."""
        for event_type in ("user.create", "user.invite"):
            assert "invitation" in safe_detail_keys_for(event_type)
        assert "invitation" not in safe_detail_keys_for(_PLAIN_EVENT)

    def test_a_grant_that_followed_its_renamed_node_keeps_every_key_and_value(self) -> None:
        """``rbac.assignment.move``: ids, the fixed reason and the node names (a namespace's, no person's) stay."""
        details = {
            "datasource_id": str(uuid7()),
            "group_id": str(uuid7()),
            "role_id": str(uuid7()),
            "managed_by": "manual",
            "row_scope": "platform",
            "scope_type": "subtree",
            "reason": "datasource_renamed",
            "scope_namespace_name_before": "datasources.agent_ab.reports",
            "scope_namespace_name_after": "datasources.agent_ab.summaries",
        }

        assert anonymize_details(details, event_type="rbac.assignment.move") == details
        assert all(is_classified_detail_key(key, event_type="rbac.assignment.move") for key in details)

    @pytest.mark.parametrize("event_type", ["security.exploit.approval.approved", "security.exploit.approval.denied"])
    def test_the_approver_of_a_paused_tool_call_keeps_their_id(self, event_type: str) -> None:
        """``decided_by`` is the deciding user's uuid on the hub's approval events, and erasure never changes an id."""
        approver = str(uuid7())

        kept = anonymize_details({"decided_by": approver, "decision": "approve"}, event_type=event_type)

        assert kept == {"decided_by": approver, "decision": "approve"}
        assert is_classified_detail_key("decided_by", event_type=event_type)
        assert "decided_by" not in SAFE_DETAIL_KEYS

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
    def test_the_shape_is_preserved_but_under_identifying_keys(self, details: dict[str, Any]) -> None:
        """every key at every depth survives; only an identifying key's value is replaced (by the marker)."""
        anonymized = anonymize_details(details, event_type=_PLAIN_EVENT)

        def walk(before: Any, after: Any) -> None:
            if isinstance(before, dict):
                assert after.keys() == before.keys()
                for key, value in before.items():
                    if key in IDENTIFYING_DETAIL_KEYS:
                        assert after[key] == (None if value is None else ANONYMIZED_MARKER)
                    else:
                        walk(value, after[key])
            elif isinstance(before, list | tuple):
                assert len(after) == len(before)
                for one, other in zip(before, after, strict=True):
                    walk(one, other)
            else:
                assert after == before

        walk(details, anonymized)

    @given(details=_DETAILS)
    def test_anonymization_is_idempotent(self, details: dict[str, Any]) -> None:
        """a second pass is a no-op."""
        once = anonymize_details(details, event_type=_PLAIN_EVENT)

        assert anonymize_details(once, event_type=_PLAIN_EVENT) == once

    @given(details=_DETAILS)
    def test_the_input_is_never_mutated(self, details: dict[str, Any]) -> None:
        """purity holds for every shape, not just the hand-written ones."""
        before = copy.deepcopy(details)

        anonymize_details(details, event_type=_PLAIN_EVENT)

        assert details == before
