"""configuration dataclass for invalidation-listener enforcement.

The contract is universal -- every ``CollectionRegistry`` holding an L2-live collection
subscribes the cross-pod invalidation stream, and every start is paired with a stop -- but
the knobs below let each repo declare its own layout and its own legitimate exemptions
without forking the walkers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["InvalidationListenerConfig"]


@dataclass(frozen=True)
class InvalidationListenerConfig:
    """per-repo config for the invalidation-listener enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root.
    :ivar src_roots: src trees to scan. ``None`` asks the runner to discover
        them with
        :func:`~threetears.enforcement.common.repo_layout.find_local_src_roots`,
        which walks ``packages/**/src`` at any depth as well as a top-level
        ``src/``. Set it to pin a single tree.
    :ivar exemptions_path: path to
        ``_invalidation_listener_exemptions.txt``; ``None`` means "no
        exemptions file".
    :ivar mode_env_var: environment variable controlling strict vs report
        mode. defaults to ``INVALIDATION_LISTENER_ENFORCEMENT_MODE``.
    :ivar minimum_live_registries: floor on how many L2-live registries the
        scan must find. **The most important knob here.** A reader that
        silently stopped matching -- a renamed constructor, a moved src root,
        a layout change -- demands nothing of anybody while still reporting
        green, and green is indistinguishable from correct. Asserting the
        count from BELOW is what makes the gate fail loudly instead. Set it to
        the number the repo actually has, and raise it when that grows.
        ``0`` disables the check, which is only right for a repo that
        genuinely wires none. **REQUIRED, with no default**, and deliberately so:
        a default of ``0`` would let a consumer adopt the minimal config and get
        exactly the silent green-over-nothing the floor exists to prevent. Making
        it positional forces the number to be a decision somebody made.
    :ivar skip_basenames: file basenames the walkers skip entirely.
    """

    repo_root: Path
    minimum_live_registries: int
    src_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "INVALIDATION_LISTENER_ENFORCEMENT_MODE"
    skip_basenames: frozenset[str] = field(default_factory=frozenset)
