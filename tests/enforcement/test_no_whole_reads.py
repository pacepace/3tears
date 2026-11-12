"""enforcement: no load, cache or follower path in 3tears reads a whole table.

An epoch IS its change set: a pod reads the rows the change log and its change notes name, by key, and nothing
reads a whole table to learn what changed, re-verifies its own rows by scanning them, or falls back to a full
reload. The rule (:mod:`threetears.enforcement.no_whole_reads`) walks every package's ``src`` and names each whole
read, whole aggregate, unkeyed pager, whole-table write, whole/rebuild/warm call and every call into one.

The only way out is a read declared whole by nature where it is written (``threetears.core.declared.whole_read``:
a cold start, a schema change, a drain, ... each by Pace's ruling), with the test proving it is reached only then and,
for most kinds, its speed test at real size. There is no ledger and no allowlist of violations; this gate is red
until each one it names is fixed.

The runtime half is :class:`threetears.core.testing.plans.WholeScanGuard`, wrapped around the connection of a
load-path test.
"""

from __future__ import annotations

from tests.enforcement.no_whole_reads_support import NO_WHOLE_READS_CONFIG as _CONFIG
from threetears.enforcement.no_whole_reads import no_whole_reads_findings


def test_no_whole_reads() -> None:
    """no load, cache or follower path reads a whole table, outside a listed cold start or schema change.

    :return: nothing
    :rtype: None
    """
    findings = no_whole_reads_findings(_CONFIG)
    assert not findings, f"{len(findings)} whole read(s):\n" + "\n".join(findings)
