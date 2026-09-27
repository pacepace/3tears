"""the unsigned-agent registration concession ends with the 0.56 line, and this test is what ends it.

The 0.55 and 0.56 lines still admit an UNSIGNED agent manifest's agent-scoped copies, because agents built on an
older SDK register unsigned and refusing them outright would strip every not-yet-rebuilt agent of
its in-process tools mid-deploy. It is a security concession: an unsigned manifest under an
agent's pod id defines that agent's in-process copy, and the only fence is that the pod id has
not registered signed while the catalog holds the copy.

It was due to end at 0.56.0. It was moved to 0.57.0 because 0.56.0 became the patch release for
the Claude CLI regressions found in 0.55.0, shipped before any deployed agent image was rebuilt on
the SDK that signs; ending it there would have stripped every deployed agent of its tools on the
hub rollout.

A promise carried only by prose ships unchanged in the next release, so the promise is carried
here: once the family version reaches 0.57.0 this test fails for as long as ``admit_copy`` still
admits the unverified agent-scoped copy. There is no configuration switch for it -- a flag between
the old and the new path is exactly the dual path the platform does not ship.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from threetears.registry.ownership import CopyAudience, PublisherStanding, admit_copy

_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"

#: the first release that must refuse unsigned agent manifests
_REFUSING_RELEASE = (0, 57, 0)

_REMOVAL = (
    "the unsigned-agent concession must be removed before {version} ships: in "
    "threetears.registry.ownership.admit_copy drop ``and audience is CopyAudience.EVERYONE`` from the "
    "``not standing.verified`` condition, so an unverified publisher is refused for every audience; then "
    "delete the admitting branch of RegistrationHandler._unsigned_agent_publisher, its "
    "_unsigned_agent_pods_warned set, and the 'Rollout concession' paragraph of the CHANGELOG"
)


def _registry_version() -> tuple[int, int, int]:
    """the registry package's own version, as its pyproject declares it.

    :return: ``(major, minor, patch)``
    :rtype: tuple[int, int, int]
    """
    declared = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    major, minor, patch = (int(part) for part in declared.split(".")[:3])
    return major, minor, patch


def _concession_admits() -> bool:
    """whether an unverified publisher's agent-scoped copy is still admitted.

    :return: ``True`` while the concession is in the code
    :rtype: bool
    """
    unverified = PublisherStanding(verified=False, platform_shared=False, owned_nodes=())
    return (
        admit_copy(
            tool_name="threetears.calculator",
            audience=CopyAudience.AGENT,
            standing=unverified,
            provider_nodes=(),
        )
        is None
    )


def test_the_version_is_read() -> None:
    """non-vacuity: a version that could not be read would let the gate below pass on nothing."""
    assert _registry_version() > (0, 0, 0)


def test_the_concession_does_not_outlive_the_0_56_line() -> None:
    version = _registry_version()
    if version >= _REFUSING_RELEASE:
        assert not _concession_admits(), _REMOVAL.format(version=".".join(map(str, version)))


def test_the_concession_still_never_serves_everyone() -> None:
    """what the concession never covered: an unverified copy that would serve every caller."""
    unverified = PublisherStanding(verified=False, platform_shared=False, owned_nodes=())
    refusal = admit_copy(
        tool_name="threetears.calculator", audience=CopyAudience.EVERYONE, standing=unverified, provider_nodes=()
    )
    assert refusal is not None
