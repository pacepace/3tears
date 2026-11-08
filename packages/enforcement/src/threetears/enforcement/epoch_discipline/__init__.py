"""epoch-discipline enforcement domain: no code bypasses 3tears epochs or degrades quietly.

Five rules, one gate each (:data:`RULES`): no whole drops, no swallowed L2 failure, no expiry as a
safety net, no write around the epoch path, no whole re-read to recover. Each repo runs
:func:`epoch_discipline_findings` per rule over its own source, with its own shrinking allowlists
and waivers, and :func:`epoch_violation_counts` measures a repo that has not adopted the gates yet.
The rules are in :mod:`threetears.enforcement.epoch_discipline.config`, the detectors in
:mod:`threetears.enforcement.epoch_discipline.walkers`, the ledgers in
:mod:`threetears.enforcement.epoch_discipline.ratchet`.
"""

from threetears.enforcement.epoch_discipline.config import (
    RULE_EPOCH_BYPASS,
    RULE_L2_SWALLOW,
    RULE_TTL_SAFETY_NET,
    RULE_WHOLE_DROP,
    RULE_WHOLE_REREAD,
    RULES,
    EpochDisciplineConfig,
    allowlist_path_for,
)
from threetears.enforcement.epoch_discipline.ratchet import epoch_discipline_findings, epoch_violation_counts

__all__ = [
    "RULES",
    "RULE_EPOCH_BYPASS",
    "RULE_L2_SWALLOW",
    "RULE_TTL_SAFETY_NET",
    "RULE_WHOLE_DROP",
    "RULE_WHOLE_REREAD",
    "EpochDisciplineConfig",
    "allowlist_path_for",
    "epoch_discipline_findings",
    "epoch_violation_counts",
]
