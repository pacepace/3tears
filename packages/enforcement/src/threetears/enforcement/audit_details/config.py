"""configuration dataclass for audit-details classification enforcement.

erasure anonymizes an audit record's ``details`` under an explicit safe-key
list (:func:`threetears.agent.audit.anonymize_details`): the value under a safe
key survives, the whole value under any other key becomes a marker. the rule
fails safe on its own, so this domain is not what keeps personal data out of an
erased record. it keeps the CLASSIFICATION from going stale: a producer that
publishes a key nobody classified has it masked on erasure without anyone having
decided that, and a producer that publishes a personal key gets no review at all.

the classification itself is injected rather than imported. it lives in
``threetears.agent.audit``, which carries a NATS client this scanner has no use
for, and a consumer that declares family keys at import time
(``declare_safe_detail_keys``) is only seen by a lookup made after that import --
which is the consumer's shell, not this package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DEFAULT_AUDIT_CONSTRUCTORS", "AuditDetailsConfig"]

#: the callee spellings that construct an audit event and take ``details=``.
DEFAULT_AUDIT_CONSTRUCTORS: frozenset[str] = frozenset({"AuditEvent"})


@dataclass(frozen=True)
class AuditDetailsConfig:
    """per-repo config for the audit-details classification enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root.
    :ivar src_roots: the source trees to scan, usually
        :func:`threetears.enforcement.common.find_local_src_roots` of ``repo_root``.
        Empty is itself a violation -- a shell that scans nothing reports what a
        clean repo reports.
    :ivar safe_keys_for: the safe-key lookup for an event type; pass
        :func:`threetears.agent.audit.safe_detail_keys_for`. Called with ``""`` for a
        construction whose ``event_type`` is not a literal, which yields the platform
        set with no family credit.
    :ivar personal_keys: keys classified as personal; pass
        :data:`threetears.agent.audit.PERSONAL_DETAIL_KEYS`.
    :ivar constructors: callee names (bare or attribute) that build an audit event.
    :ivar forwarders: callee names of the repo's own wrapper helpers that accept
        ``details=`` and hand it to a constructor. A call to one is read exactly like a
        constructor call, and inside one a ``details`` that is (or merely spreads,
        copies or defaults) the helper's own parameter is accepted, because its call
        sites carry the keys. Without the declaration the helper's hand-off is refused
        as unreadable, which is the point: an undeclared wrapper is a place keys pass
        through unseen.
    :ivar mode_env_var: environment variable controlling strict vs report mode.
    """

    repo_root: Path
    src_roots: tuple[Path, ...]
    safe_keys_for: Callable[[str], frozenset[str]]
    personal_keys: frozenset[str]
    constructors: frozenset[str] = DEFAULT_AUDIT_CONSTRUCTORS
    forwarders: frozenset[str] = frozenset()
    mode_env_var: str = "AUDIT_DETAILS_ENFORCEMENT_MODE"
