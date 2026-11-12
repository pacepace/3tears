"""no 3tears producer passes an audit details key that names a person (Pace 2026-11-11: audit rows hold ids).

Thin shell -- the walk is :func:`threetears.enforcement.audit_details.identifying_detail_findings`, the rule's data
:func:`threetears.agent.audit.identifying_keys_for`. A new key that names a person under a new spelling is caught by
``test_audit_details_keys_are_classified`` (every published key is classified) and classified into
``IDENTIFYING_DETAIL_KEYS``.
"""

from __future__ import annotations

from pathlib import Path

from threetears.agent.audit import identifying_keys_for
from threetears.enforcement.audit_details import IdentifyingDetailsConfig, identifying_detail_findings
from threetears.enforcement.common import find_local_src_roots

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_producer_passes_a_key_that_names_a_person() -> None:
    """a ``details`` key naming a person, at any depth, at any audit event this repo builds fails here."""
    config = IdentifyingDetailsConfig(
        repo_root=_REPO_ROOT, src_roots=find_local_src_roots(_REPO_ROOT), identifying_keys_for=identifying_keys_for
    )
    assert identifying_detail_findings(config) == []
