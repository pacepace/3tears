"""enforcement: no load, cache or follower path in 3tears reads a whole table.

An epoch IS its change set: a pod reads the rows the change log and its change notes name, by key, and nothing
reads a whole table to learn what changed, re-verifies its own rows by scanning them, or falls back to a full
reload. The rule (:mod:`threetears.enforcement.no_whole_reads`) walks every package's ``src`` and names each whole
read, whole aggregate, unkeyed pager, whole-table write, whole/rebuild/warm call and every call into one.

The only way out is ``_no_whole_reads_exemptions.txt`` beside this file: a true cold start or a schema change,
with the test proving it is reached only then. There is no allowlist of violations; this gate is red until each
one it names is fixed.

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
