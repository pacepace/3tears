"""the group nesting is configuration, in levels: read from the environment once, refused when it cannot be honoured.

levels count the chain the way a reader does -- the person plus each group in it -- and the
platform walks hops, one fewer: :data:`MAX_GROUP_MEMBERSHIP_DEPTH` is levels - 1.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from threetears.agent.acl import MAX_GROUP_MEMBERSHIP_DEPTH
from threetears.agent.acl.types import (
    DEFAULT_MAX_GROUP_NESTING_LEVELS,
    MAX_CONFIGURABLE_GROUP_NESTING_LEVELS,
    MAX_GROUP_NESTING_LEVELS,
    MAX_GROUP_NESTING_LEVELS_ENV,
    MIN_CONFIGURABLE_GROUP_NESTING_LEVELS,
    RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV,
    resolve_max_group_nesting_levels,
)


def _hops_in_a_fresh_process(env_overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """import the package in a fresh interpreter with ``env_overrides``, the way a deployment configures it.

    :param env_overrides: variables to set on top of this process's environment, with both nesting
        variables cleared first
    :ptype env_overrides: dict[str, str]
    :return: the finished interpreter; stdout is ``MAX_GROUP_MEMBERSHIP_DEPTH`` when the import succeeded
    :rtype: subprocess.CompletedProcess[str]
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in (MAX_GROUP_NESTING_LEVELS_ENV, RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV)
    }
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", "from threetears.agent.acl import MAX_GROUP_MEMBERSHIP_DEPTH as d; print(d)"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_unset_is_three_levels_which_is_two_hops() -> None:
    """the default behaves exactly like the hop setting's default: a person plus two groups."""
    assert DEFAULT_MAX_GROUP_NESTING_LEVELS == 3
    assert resolve_max_group_nesting_levels({}) == DEFAULT_MAX_GROUP_NESTING_LEVELS
    result = _hops_in_a_fresh_process({})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2"


def test_the_hops_every_surface_reads_are_levels_minus_the_person() -> None:
    assert MAX_GROUP_MEMBERSHIP_DEPTH == MAX_GROUP_NESTING_LEVELS - 1


def test_the_configurable_range_is_two_to_five_levels() -> None:
    assert (MIN_CONFIGURABLE_GROUP_NESTING_LEVELS, MAX_CONFIGURABLE_GROUP_NESTING_LEVELS) == (2, 5)


@pytest.mark.parametrize(
    "levels", range(MIN_CONFIGURABLE_GROUP_NESTING_LEVELS, MAX_CONFIGURABLE_GROUP_NESTING_LEVELS + 1)
)
def test_every_level_in_range_is_taken_as_given(levels: int) -> None:
    assert resolve_max_group_nesting_levels({MAX_GROUP_NESTING_LEVELS_ENV: f" {levels} "}) == levels


@pytest.mark.parametrize(("levels", "hops"), [(2, 1), (3, 2), (4, 3), (5, 4)])
def test_levels_set_at_startup_become_hops_one_fewer(levels: int, hops: int) -> None:
    result = _hops_in_a_fresh_process({MAX_GROUP_NESTING_LEVELS_ENV: str(levels)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(hops)


@pytest.mark.parametrize("raw", ["0", "1", "-1", str(MAX_CONFIGURABLE_GROUP_NESTING_LEVELS + 1), "three", "", "2.5"])
def test_a_value_it_cannot_honour_is_refused_not_defaulted(raw: str) -> None:
    """a deployment that asked for a nesting and silently got another would show memberships granting nothing."""
    with pytest.raises(ValueError, match=MAX_GROUP_NESTING_LEVELS_ENV):
        resolve_max_group_nesting_levels({MAX_GROUP_NESTING_LEVELS_ENV: raw})


@pytest.mark.parametrize(("old", "converted"), [("1", "2"), ("2", "3"), ("4", "5")])
def test_the_retired_hop_variable_is_refused_with_the_conversion(old: str, converted: str) -> None:
    """the old name counted hops; read as levels it would be one short, so it refuses rather than guess."""
    with pytest.raises(ValueError) as refused:
        resolve_max_group_nesting_levels({RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV: old})
    message = str(refused.value)
    assert RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV in message
    assert f"{MAX_GROUP_NESTING_LEVELS_ENV}={converted}" in message
    assert "old value + 1" in message


def test_the_retired_hop_variable_is_refused_even_beside_the_new_one() -> None:
    with pytest.raises(ValueError, match=RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV):
        resolve_max_group_nesting_levels(
            {RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV: "2", MAX_GROUP_NESTING_LEVELS_ENV: "3"}
        )


def test_the_retired_hop_variable_refuses_at_import() -> None:
    result = _hops_in_a_fresh_process({RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV: "2"})
    assert result.returncode != 0
    assert f"{MAX_GROUP_NESTING_LEVELS_ENV}=3" in result.stderr
