"""thin shell -- walker logic in :mod:`threetears.enforcement.invalidation_listener`.

Every ``CollectionRegistry`` holding an L2-live collection must subscribe the cross-pod
invalidation stream, and every start must be paired with a stop. Publishing an
invalidation happens automatically on every write; CONSUMING it is a separate step, and
the forgetting is silent -- the pod serves its own stale L1 copy and nothing reports it.

Scoping the keyspace per principal made the consequence worse rather than better. Before,
a peer writing the same row refreshed a SHARED key underneath a forgetful pod. Now each
principal has a private keyspace, nothing else ever writes those keys, and a row cached
before a sibling's write stays stale in BOTH tiers for the life of the process.

3tears runs this over its own source because it wires L2-live registries itself -- the
registry server holds two in one process -- and because a library that ships a gate it
does not run over itself is asking consumers to meet a bar it has not checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from threetears.enforcement.common import find_local_src_roots
from threetears.enforcement.invalidation_listener import (
    InvalidationListenerConfig,
    count_l2_live_registries,
    run_invalidation_listener_enforcement,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_EXEMPTIONS_PATH: Final[Path] = Path(__file__).resolve().parent / "_invalidation_listener_exemptions.txt"

#: floor on what the reader must SEE. Asserted from below because a scan that matches
#: nothing reports exactly what a clean repo reports. Three today: the registry server's
#: heartbeat registry, its rbac registry, and the tool pod's stack. Raise it when that
#: grows; never lower it to make a red gate green.
_MINIMUM_LIVE_REGISTRIES: Final[int] = 3

_CONFIG = InvalidationListenerConfig(
    repo_root=_REPO_ROOT,
    exemptions_path=_EXEMPTIONS_PATH,
    minimum_live_registries=_MINIMUM_LIVE_REGISTRIES,
)


class TestEveryLiveRegistryConsumesTheInvalidationStream:
    """the gate itself, over every registry construction in this repo's src trees."""

    def test_every_l2_live_registry_starts_a_listener(self) -> None:
        """an unlistened registry serves rows a peer replica has already replaced."""
        run_invalidation_listener_enforcement(_CONFIG, walker="unlistened")

    def test_every_start_is_paired_with_a_stop(self) -> None:
        """a subscription outliving its connection is the unpaired-lifecycle shape."""
        run_invalidation_listener_enforcement(_CONFIG, walker="unpaired")


class TestTheReaderIsNotVacuous:
    """a reader that matched nothing would demand nothing while reporting green."""

    def test_the_repo_really_does_wire_live_registries(self) -> None:
        """the floor is asserted here too, not only inside the runner.

        The runner's check fails the gate; this one names the number in a test whose
        name says what it is for, so a future reader lowering the floor to quiet a red
        gate has to edit something that argues back.
        """
        seen = count_l2_live_registries(find_local_src_roots(_REPO_ROOT))

        assert seen >= _MINIMUM_LIVE_REGISTRIES, (
            f"the reader saw {seen} L2-live registries, below the floor of "
            f"{_MINIMUM_LIVE_REGISTRIES}. that is a broken reader, not a clean repo"
        )


class TestTheExemptionFileIsDisciplined:
    """an exemption nobody has to justify is an off switch."""

    def test_every_exemption_carries_a_specific_rationale(self) -> None:
        """the rationale must NAME the subscriber, so the claim is checkable."""
        lines = _EXEMPTIONS_PATH.read_text().splitlines()
        entries = [(i, line) for i, line in enumerate(lines) if line.strip() and not line.startswith("#")]

        for index, entry in entries:
            preceding = [line for line in lines[:index] if line.strip()]
            assert preceding and preceding[-1].startswith("# rationale:"), (
                f"exemption {entry!r} has no immediately preceding '# rationale:' line"
            )
            rationale = preceding[-1]
            assert len(rationale) > len("# rationale: ") + 40, (
                f"exemption {entry!r} has a rationale too short to name a subscriber: {rationale!r}"
            )

    def test_every_exemption_names_a_file_that_exists(self) -> None:
        """a stale entry silences a gate over a path nothing occupies."""
        lines = _EXEMPTIONS_PATH.read_text().splitlines()
        entries = [line.strip().split(":")[0] for line in lines if line.strip() and not line.startswith("#")]

        missing = [entry for entry in entries if not (_REPO_ROOT / entry).is_file()]

        assert not missing, f"exemption entries name files that do not exist: {missing}"
