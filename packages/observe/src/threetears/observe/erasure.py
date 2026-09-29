"""the platform's one erasure marker: the value an anonymized field holds.

Person erasure keeps records and ids and replaces what identifies the person with this
marker -- audit ``details`` (:mod:`threetears.agent.audit.anonymize`), LangGraph
checkpoints (:mod:`threetears.langgraph.anonymize`), and the hub's own tables. Every
writer uses the same spelling, so a reader can recognise an erased value wherever it sits.

**Why it lives here.** ``3tears-observe`` is the one layer every package already depends
on and that depends on nothing, so homing the marker here gives it one source without
making the checkpoint saver depend on the audit package, or the audit package on the
data-layer core, for a string. :mod:`threetears.agent.audit` re-exports it as part of its
own erasure API.
"""

from __future__ import annotations

from typing import Final

__all__ = ["ANONYMIZED_MARKER"]


#: the value every anonymized field becomes. fixed text rather than ``None`` so an erased
#: value reads as erased, distinct from a value that was never there.
ANONYMIZED_MARKER: Final[str] = "[anonymized]"
