"""Thin shell over the canonical nats-wrapper walker.

Every package reaches NATS through :class:`threetears.nats.NatsClient`; nothing imports
``nats`` directly. The wrapper is where reconnect policy, subject namespacing, and the KV
helpers live, so a direct import is a package quietly opting out of all three.

``packages/nats`` is absent from the scanned roots rather than exempted inside them: it
holds all 29 direct imports in the workspace, and it is supposed to. It is not an
exception to the rule, it is the thing the rule points at.

The walker has been in ``packages/enforcement`` with no caller since it was written, so
this invariant has been documented and unenforced.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.nats_wrapper_usage import (
    NatsWrapperConfig,
    run_nats_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SRC_ROOTS = tuple(
    sorted(
        {
            path.parent
            for pattern in ("packages/*/src/threetears", "packages/agent/*/src/threetears/agent")
            for path in _REPO_ROOT.glob(pattern)
            if path.is_dir() and path.parents[1].name != "nats"
        }
    )
)

_CONFIG = NatsWrapperConfig(repo_root=_REPO_ROOT, src_roots=_SRC_ROOTS)


def test_production_code_reaches_nats_only_through_the_wrapper() -> None:
    """A direct ``import nats`` skips reconnect policy, subject namespacing and the KV
    helpers all at once."""
    run_nats_enforcement(_CONFIG, walker="production")
