"""a table's keep rule: which of its folded rows a full keeps (``docs/design-pod-delta-files.md``, decision 13).

**Why.** A full on the delta-files volume (:mod:`~threetears.core.collections.volume_fulls`) holds what a pod copy
holds, never more: a table whose copy keeps only some of its rows (LSF runs thinned to every run for two hours, then
one per 15 minutes, then hourly, plus the first and newest) declares the same rule here, and the writer's fold
(:mod:`~threetears.core.collections.delta_fold`) applies it to each state it folds. Without it a full of a history
table would hold every run ever written.

**A class, not a function.** The fold runs in a worker process (:class:`~threetears.core.process_workers.
ProcessWorkers`), which is handed plain data and classes, never objects: a class crosses as its name, so the rule is
a subclass whose :meth:`DeltaKeepRule.keep` the worker calls. It is declared where the table opts in to delta files,
on its schema (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_keep`).

**Monotone.** The fold applies the rule to the old part plus the deltas since, not to every row ever written, so a
rule must keep the same rows either way: thinning what was already thinned plus what came since keeps what thinning
everything keeps. A rule that is not (one that keeps a row only while something it dropped earlier is present) makes
a full that differs from the copy.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DeltaKeepRule"]


class DeltaKeepRule:
    """the rows of one (table, state) part a full keeps after a fold; subclass it and override :meth:`keep`.

    The base keeps every row: a table that declares no rule keeps everything.
    """

    @classmethod
    def keep(cls, rows: Any) -> Any:
        """the rows the part keeps, from the folded rows: the old part with every delta since applied.

        Runs in the fold's worker process, on one state's rows at a time. Pure: the same rows in, the same rows out.

        :param rows: the folded rows, the table's columns
        :ptype rows: pyarrow.Table
        :return: the rows kept, the same columns
        :rtype: pyarrow.Table
        """
        return rows
