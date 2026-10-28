"""the group-nesting depth is configuration: read from the environment once, refused when it cannot be honoured."""

from __future__ import annotations

import pytest

from threetears.agent.acl import (
    DEFAULT_MAX_GROUP_MEMBERSHIP_DEPTH,
    MAX_CONFIGURABLE_GROUP_MEMBERSHIP_DEPTH,
    MAX_GROUP_MEMBERSHIP_DEPTH_ENV,
    resolve_max_group_membership_depth,
)


def test_unset_is_the_default_of_two() -> None:
    assert DEFAULT_MAX_GROUP_MEMBERSHIP_DEPTH == 2
    assert resolve_max_group_membership_depth({}) == DEFAULT_MAX_GROUP_MEMBERSHIP_DEPTH


@pytest.mark.parametrize("depth", range(1, MAX_CONFIGURABLE_GROUP_MEMBERSHIP_DEPTH + 1))
def test_every_depth_in_range_is_taken_as_given(depth: int) -> None:
    assert resolve_max_group_membership_depth({MAX_GROUP_MEMBERSHIP_DEPTH_ENV: f" {depth} "}) == depth


@pytest.mark.parametrize("raw", ["0", "-1", str(MAX_CONFIGURABLE_GROUP_MEMBERSHIP_DEPTH + 1), "three", "", "2.5"])
def test_a_value_it_cannot_honour_is_refused_not_defaulted(raw: str) -> None:
    """a deployment that asked for a depth and silently got another would show memberships granting nothing."""
    with pytest.raises(ValueError, match=MAX_GROUP_MEMBERSHIP_DEPTH_ENV):
        resolve_max_group_membership_depth({MAX_GROUP_MEMBERSHIP_DEPTH_ENV: raw})
