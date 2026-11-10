"""enforcement: 3tears reads and writes its data tables through its helpers, never as SQL text written anywhere else.

Every keyed read has one home (:mod:`threetears.core.keyed`, :mod:`threetears.core.keyset`,
:class:`threetears.epoch.RowFollower`), and the migration and DDL runners write the schema. A statement written as
text against a data table anywhere else is a second way of reading it (:mod:`threetears.enforcement.no_whole_reads
.use_the_helper`). The walk and its declarations are the no-whole-reads gate's own (``no_whole_reads_support.NO_WHOLE_READS_CONFIG``).

The only way out is ``_use_the_helper_exemptions.txt`` beside this file: the function, its statement count and why
no helper writes it yet. There is no allowlist; where no helper fits, the helper set grows. This gate is red until
each statement it names goes through one.
"""

from __future__ import annotations

from tests.enforcement.no_whole_reads_support import NO_WHOLE_READS_CONFIG as _CONFIG
from threetears.enforcement.no_whole_reads import use_the_helper_findings


def test_use_the_helper() -> None:
    """no statement is written as text against a data table outside the helpers, outside the ledger.

    :return: nothing
    :rtype: None
    """
    findings = use_the_helper_findings(_CONFIG)
    assert not findings, f"{len(findings)} statement(s) written outside the helpers:\n" + "\n".join(findings)
