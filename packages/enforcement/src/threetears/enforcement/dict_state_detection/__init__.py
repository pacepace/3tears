"""dict-state-detection enforcement domain — raw-dict persistent state walker.

flags ``self._x = {}`` / ``dict()`` / ``OrderedDict()`` / ``... or {}``
assignments inside ``__init__`` methods. shared / cached state belongs
in a 3tears L1 backend (``SQLiteBackend``) for pod-local cache or NATS
KV for cross-instance sharing. raw dicts are flagged to prevent silent
hidden-cache drift from the rest of the system.

per-repo configuration goes through :class:`DictStateConfig`;
:func:`run_dict_state_enforcement` is the pytest-friendly entry point
that orchestrates the detection / allowlist-integrity walkers, applies
the in-code allowlist + known-violations filtering, and fails in
strict mode.
"""

from threetears.enforcement.dict_state_detection.config import (
    AllowlistRationaleError,
    DictStateAllowlistEntry,
    DictStateConfig,
)
from threetears.enforcement.dict_state_detection.runner import (
    run_dict_state_enforcement,
)
from threetears.enforcement.dict_state_detection.walkers import (
    filter_against_allowlist,
    find_dict_state_violations,
    find_stale_allowlist_entries,
)

__all__ = [
    "AllowlistRationaleError",
    "DictStateAllowlistEntry",
    "DictStateConfig",
    "filter_against_allowlist",
    "find_dict_state_violations",
    "find_stale_allowlist_entries",
    "run_dict_state_enforcement",
]
