"""configuration for await-race enforcement.

The domain enforces one contract (:mod:`.walkers`): an async function never assigns a long-lived target a value derived
from a read of it taken before an ``await``. Its config has the dict rule's shape (roots, non-serving roots) and no
allowlist: a stale write-back is fixed (read after the await, or the update kept with its one owner), never waived.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["AwaitRaceConfig"]


@dataclass(frozen=True)
class AwaitRaceConfig:
    """per-repo config for await-race enforcement.

    :ivar repo_root: absolute path to the consumer repo's root
    :ivar src_roots: the trees walked; ``None`` discovers the path-dep src trees
        (:func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`)
    :ivar non_serving_roots: trees (directories or modules) that are not serving code, each declared by root as the
        dict rule's are (:attr:`threetears.enforcement.dict_state_detection.DictStateConfig.non_serving_roots`):
        nothing under one is reported
    :ivar mode_env_var: environment variable choosing strict or report mode
    """

    repo_root: Path
    src_roots: tuple[Path, ...] | None = None
    non_serving_roots: tuple[Path, ...] = ()
    mode_env_var: str = "AWAIT_RACE_ENFORCEMENT_MODE"
