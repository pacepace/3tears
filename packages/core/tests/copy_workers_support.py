"""the one worker process a test process's copy followers share, each job sent to it named.

Why one: a follower's pass sends its key work to its worker process (:mod:`threetears.core.collections.copy_key_work`),
and a pool a test module, beside the process's own, put a whole interpreter more in every test process that ran
those modules; a many-core pool host ran short of threads and processes ("can't start new thread", monad-1,
2026-11-25). Every follower a copy-follower test builds is given this one (``drift_workers``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from threetears.core.process_workers import ProcessWorkers


class CountedWorkers(ProcessWorkers):
    """the copy's worker process, each job sent to it named: a pass's key work runs there, in another process."""

    def __init__(self) -> None:
        super().__init__(workers=1, name="test copy follower")
        self.jobs: list[str] = []

    async def run(self, job: Callable[..., Any], *args: Any, inline: bool = False) -> Any:
        if not inline:
            self.jobs.append(job.__name__)
        return await super().run(job, *args, inline=inline)


#: made at its first job, joined at the interpreter's exit, never one a test
COPY_WORKERS = CountedWorkers()
