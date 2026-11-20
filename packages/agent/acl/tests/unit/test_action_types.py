"""unit tests for which namespace type an action is an action ON: the rule caller visibility counts a grant by.

A role bound where its bucket holds an action does not make its holder a member of that namespace's customer unless
the action is the namespace type's own: prefixed ``<type>.``, or a verb naming no type (bare, ``read``; or
parameterized, ``read_file_matching:**/*.yaml``). ``{"customer": ["group.read"]}`` on a customer namespace holds an
action on groups, and opened every list of the customer before this rule (Pace's ruling 2026-11-20, option A).

The definition is reached through its front doors, the two production reads of it: :func:`permissions_act_on` (the
Python arm, :func:`threetears.agent.acl.visible_customers`) and :func:`action_acts_on_sql` (the SQL arm,
:func:`threetears.agent.acl.caller_visible_customer_clause`), which a real database holds equal in
``integration/test_query_visibility_roles.py``.
"""

from __future__ import annotations

import pytest

from threetears.agent.acl.types import (
    ACTION_ARGUMENT_SEPARATOR,
    decode_role_permissions,
    permissions_act_on,
)

#: every action shape and the type it names: None for a verb naming no type. the stem is the text before the first
#: ``:`` (the whole action when it holds none); a stem holding a ``.`` names the type before the first one
_NAMED: dict[str, str | None] = {
    "read": None,
    "call": None,
    "write": None,
    "": None,
    "read_file_matching:**/*": None,
    "read_file_matching:**/*.yaml": None,
    "read_file_matching:": None,
    "write_file_matching:docs/a.b/c.md": None,
    "customer.read": "customer",
    "customer.impersonate": "customer",
    "group.read": "group",
    "control_plane.read": "control_plane",
    "datasource.relation_read": "datasource",
    "shared_agent.invoke": "shared_agent",
    "a.b.c": "a",
    "datasource.read_matching:x.y": "datasource",
    ".read": "",
}

#: the types each action is tried against: the one it names (when it names one) and ones it does not
_TYPES: tuple[str, ...] = ("customer", "group", "datasource", "a", "read_file_matching", "")


@pytest.mark.parametrize("action", list(_NAMED))
@pytest.mark.parametrize("namespace_type", _TYPES)
def test_an_action_acts_on_the_type_its_stem_names_or_on_every_type_when_it_names_none(
    action: str, namespace_type: str
) -> None:
    """a role holding one action in the type's own bucket acts on the type exactly when the action names that type or
    names none: a stem holding a ``.`` names the text before its first one; the text after a ``:`` is an argument and
    names nothing, whatever dots it holds."""
    named = _NAMED[action]
    acts = permissions_act_on(decode_role_permissions({namespace_type: [action]}), namespace_type)
    assert acts is (named is None or named == namespace_type), (action, namespace_type, named)


def test_the_argument_separator_is_the_platforms_spelling() -> None:
    """``read_file_matching:<glob>``: the stem, then the argument (also the catalog's parameterized stem)."""
    assert ACTION_ARGUMENT_SEPARATOR == ":"


def test_a_role_acts_on_a_type_only_through_that_types_own_action() -> None:
    """``permissions_act_on`` over the type's bucket and the wildcard bucket, each action counted only on its own type."""
    narrow = decode_role_permissions({"customer": ["group.read"]})
    admin = decode_role_permissions({"customer": ["customer.read", "customer.admin", "customer.impersonate"]})
    builtin_tool = decode_role_permissions({"tool": ["call"]})
    editor = decode_role_permissions({"workspace": ["read_file_matching:**/*"]})
    wildcard_typed = decode_role_permissions({"*": ["memory.read"]})
    assert not permissions_act_on(narrow, "customer")
    assert permissions_act_on(admin, "customer")
    assert permissions_act_on(builtin_tool, "tool")
    assert permissions_act_on(editor, "workspace")
    assert permissions_act_on(wildcard_typed, "memory")
    assert not permissions_act_on(wildcard_typed, "customer")
    assert not permissions_act_on(decode_role_permissions({"system": ["control_plane.read"]}), "system")


def test_a_listing_of_no_one_type_needs_an_action_of_its_own_type_somewhere() -> None:
    """``None``: some bucket holds an action on its own type, or the wildcard bucket holds any action (each acts on
    some type: its own, or every one)."""
    assert not permissions_act_on(decode_role_permissions({"system": ["group.read"]}), None)
    assert not permissions_act_on(decode_role_permissions({"customer": ["group.read"], "tool": []}), None)
    assert permissions_act_on(decode_role_permissions({"system": ["group.read", "read"]}), None)
    assert permissions_act_on(decode_role_permissions({"customer": ["customer.read"]}), None)
    assert permissions_act_on(decode_role_permissions({"*": ["group.read"]}), None)
    assert not permissions_act_on(decode_role_permissions({"*": []}), None)
