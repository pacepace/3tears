"""audit-details classification enforcement domain -- every published details key is classified.

Erasure anonymizes an audit record's ``details`` under the explicit safe-key list in
``threetears.agent.audit``: a key nobody classified is masked, so the rule fails safe on its
own. This domain keeps the classification from falling behind the code, by failing when a
producer publishes a ``details`` key that is neither safe nor personal, or passes a
``details`` argument the walker cannot read.

Per-repo configuration goes through :class:`AuditDetailsConfig`;
:func:`run_audit_details_enforcement` is the pytest-friendly entry point. A consumer shell::

    from threetears.agent.audit import PERSONAL_DETAIL_KEYS, safe_detail_keys_for
    from threetears.enforcement.audit_details import AuditDetailsConfig, run_audit_details_enforcement
    from threetears.enforcement.common import find_local_src_roots

    def test_every_audit_details_key_is_classified() -> None:
        run_audit_details_enforcement(
            AuditDetailsConfig(
                repo_root=REPO_ROOT,
                src_roots=find_local_src_roots(REPO_ROOT),
                safe_keys_for=safe_detail_keys_for,
                personal_keys=PERSONAL_DETAIL_KEYS,
                forwarders=frozenset({"publish_rbac_audit"}),
            )
        )
"""

from threetears.enforcement.audit_details.config import (
    DEFAULT_AUDIT_CONSTRUCTORS,
    AuditDetailsConfig,
)
from threetears.enforcement.audit_details.runner import (
    run_audit_details_enforcement,
)
from threetears.enforcement.audit_details.walkers import (
    AuditDetailsSite,
    collect_audit_details_sites,
    find_audit_details_violations,
    read_audit_details_sites,
    unclassified_detail_paths,
)

__all__ = [
    "DEFAULT_AUDIT_CONSTRUCTORS",
    "AuditDetailsConfig",
    "AuditDetailsSite",
    "collect_audit_details_sites",
    "find_audit_details_violations",
    "read_audit_details_sites",
    "run_audit_details_enforcement",
    "unclassified_detail_paths",
]
