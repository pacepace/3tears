"""quiet grep under pipefail: ``cmd | grep -q`` in a script that sets ``pipefail`` (or a GitHub workflow, whose bash
steps run with it) fails when it matched, because grep exits early and the writer dies of SIGPIPE.

:func:`run_grep_q_pipefail_enforcement` is the pytest entry point; :class:`GrepQPipefailConfig` scopes it.
"""

from threetears.enforcement.grep_q_pipefail.config import DEFAULT_EXCLUDED_DIRS, DEFAULT_GLOBS, GrepQPipefailConfig
from threetears.enforcement.grep_q_pipefail.runner import run_grep_q_pipefail_enforcement
from threetears.enforcement.grep_q_pipefail.walkers import find_grep_q_pipefail_violations

__all__ = [
    "DEFAULT_EXCLUDED_DIRS",
    "DEFAULT_GLOBS",
    "GrepQPipefailConfig",
    "find_grep_q_pipefail_violations",
    "run_grep_q_pipefail_enforcement",
]
