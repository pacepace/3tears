"""unified audit envelope + publish helper for the 3tears platform.

single :class:`AuditEvent` + :func:`publish_audit` shared by every
producing domain (workspace tools, hub rbac endpoints, agent memory,
custom tools). the hub-side ``unified_audit_consumer`` subscribes to
``{namespace}.audit.>`` and persists every event to
``platform_audit.audit_events`` without a domain-specific consumer.

erasure of an audit record goes through :func:`anonymize_details` and
:func:`anonymize_ip` (:mod:`threetears.agent.audit.anonymize`): records
are anonymized, never deleted, and no id on them changes.

produced by ``audit-task-01``; supersedes
:class:`threetears.agent.workspace.audit.WorkspaceAuditEnvelope` and
:class:`threetears.agent.acl.audit.RbacAuditEnvelope` (both deleted in
the same shard).
"""

from __future__ import annotations

from threetears.agent.audit.anonymize import (
    ANONYMIZED_MARKER,
    PERSONAL_DETAIL_KEYS,
    SAFE_DETAIL_KEYS,
    anonymize_details,
    anonymize_ip,
    declare_safe_detail_keys,
    is_classified_detail_key,
    safe_detail_keys_for,
)
from threetears.agent.audit.envelope import AuditEvent
from threetears.agent.audit.publish import publish_audit

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-agent-audit")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

__all__ = [
    "ANONYMIZED_MARKER",
    "PERSONAL_DETAIL_KEYS",
    "SAFE_DETAIL_KEYS",
    "AuditEvent",
    "anonymize_details",
    "anonymize_ip",
    "declare_safe_detail_keys",
    "is_classified_detail_key",
    "publish_audit",
    "safe_detail_keys_for",
]
