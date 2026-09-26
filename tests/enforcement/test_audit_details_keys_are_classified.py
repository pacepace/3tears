"""
thin shell -- walker logic in :mod:`threetears.enforcement.audit_details`.

Fails when a 3tears package publishes an audit ``details`` key that is neither in
``SAFE_DETAIL_KEYS`` (or a family declaration for its event type) nor in
``PERSONAL_DETAIL_KEYS``, or passes a ``details`` argument the walker cannot read.

Erasure anonymizes details through :func:`threetears.agent.audit.anonymize_details`,
which fails safe on its own: an unclassified key is masked. What this gate prevents is
the classification falling behind the code -- a structural key masked on erasure
without anyone having decided that, or a personal key published with no review.

What the walker reads, what it refuses, and what it cannot see (a details dict handed
to another function that adds keys; an event built by ``model_validate`` or
``model_copy``; details passed to a forwarder positionally) are documented on the
walker module. 3tears declares no forwarders: every producer here builds its
``AuditEvent`` in the function that owns the keys.
"""

from __future__ import annotations

from pathlib import Path

from threetears.agent.audit import PERSONAL_DETAIL_KEYS, SAFE_DETAIL_KEYS, safe_detail_keys_for
from threetears.enforcement.audit_details import (
    AuditDetailsConfig,
    collect_audit_details_sites,
    run_audit_details_enforcement,
)
from threetears.enforcement.common import find_local_src_roots

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFIG = AuditDetailsConfig(
    repo_root=_REPO_ROOT,
    src_roots=find_local_src_roots(_REPO_ROOT),
    safe_keys_for=safe_detail_keys_for,
    personal_keys=PERSONAL_DETAIL_KEYS,
)


class TestAuditDetailsKeysAreClassified:
    """no 3tears package publishes a details key nobody decided about."""

    def test_every_published_details_key_is_classified_and_readable(self) -> None:
        """an unclassified key or an unreadable details argument fails here, by file and line."""
        run_audit_details_enforcement(_CONFIG)

    def test_the_scan_reaches_every_known_producer(self) -> None:
        """a walker over no sites passes vacuously; pin that it found the producers it must.

        both inputs of the comparison are guarded: the sites (the tool server's baseline
        event and the workspace tools') and the classification they are judged against.
        """
        by_file = collect_audit_details_sites(_CONFIG)
        paths = [path.relative_to(_REPO_ROOT).as_posix() for path in by_file]
        keys = {key_path[-1] for sites in by_file.values() for site in sites for key_path in site.keys}

        assert any(path.startswith("packages/agent/tools/") for path in paths), paths
        assert any(path.startswith("packages/agent/workspace/") for path in paths), paths
        assert {"tool_name", "failure_reason", "workspace_resource_id"} <= keys, sorted(keys)
        assert SAFE_DETAIL_KEYS
        assert PERSONAL_DETAIL_KEYS
