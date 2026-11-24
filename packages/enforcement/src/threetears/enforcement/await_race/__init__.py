"""await-race enforcement domain: a long-lived value written back from a read taken before an ``await``.

The dict rule's replace-whole shape (:mod:`threetears.enforcement.dict_state_detection`) loses an update when an async
function reads the value, awaits, and assigns back a value computed from the stale read. :mod:`.walkers` finds that;
:func:`run_await_race_enforcement` is the pytest entry point. No allowlist.
"""

from threetears.enforcement.await_race.config import AwaitRaceConfig
from threetears.enforcement.await_race.runner import run_await_race_enforcement
from threetears.enforcement.await_race.walkers import find_await_race_violations

__all__ = ["AwaitRaceConfig", "find_await_race_violations", "run_await_race_enforcement"]
