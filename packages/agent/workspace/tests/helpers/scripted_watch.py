"""a scripted disk-change source for driving a bind window's live watcher deterministically.

The production source, :func:`threetears.agent.workspace.materialize.watch_disk_changes`, delivers
the operating system's events after delays no test can predict. :class:`ScriptedWatch` stands in
for it through ``bind(watch_changes=...)``: the test hands it a batch and :meth:`ScriptedWatch.deliver`
returns once the bind's watcher has finished with that batch, so what the batch wrote to L3 can be
asserted on straight away.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from watchfiles import Change

__all__ = ["ScriptedWatch"]


# parity-with: threetears.agent.workspace.materialize.WatchChanges
class ScriptedWatch:
    """a :data:`~threetears.agent.workspace.materialize.WatchChanges` source the test feeds by hand.

    The watcher asks its iterator for the next batch only after it has handled the previous one,
    so the iterator resuming past a ``yield`` is the signal that the batch before it is done.
    """

    def __init__(self) -> None:
        """start with no batch queued and no watcher attached."""
        self.batches: asyncio.Queue[set[tuple[Change, str]]] = asyncio.Queue()
        self.handled: asyncio.Queue[None] = asyncio.Queue()
        self.watched_roots: list[Path] = []

    def __call__(self, disk_root: Path) -> AsyncIterator[set[tuple[Change, str]]]:
        """the change iterator the bind window's watcher drives.

        :param disk_root: the bound disk root
        :ptype disk_root: Path
        :return: the scripted batches, in delivery order
        :rtype: AsyncIterator[set[tuple[Change, str]]]
        """
        self.watched_roots.append(disk_root)
        return self.iterate()

    async def iterate(self) -> AsyncIterator[set[tuple[Change, str]]]:
        """yield each delivered batch, acknowledging it once the watcher asks for the next.

        :return: the scripted batches
        :rtype: AsyncIterator[set[tuple[Change, str]]]
        """
        while True:
            batch = await self.batches.get()
            yield batch
            self.handled.put_nowait(None)

    async def deliver(self, batch: set[tuple[Change, str]], *, timeout: float = 5.0) -> None:
        """hand the watcher ``batch`` and return once it has been applied.

        :param batch: the ``{(Change, absolute_path)}`` set the operating system would deliver
        :ptype batch: set[tuple[Change, str]]
        :param timeout: seconds to wait for the watcher before failing the test
        :ptype timeout: float
        :raises TimeoutError: when no bind window's watcher took the batch in time
        """
        self.batches.put_nowait(batch)
        await asyncio.wait_for(self.handled.get(), timeout=timeout)
