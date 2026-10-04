"""The one lock every writer of an existing campaign holds — within ONE process.

Campaign documents are written without an ETag, so every edit of one is a blind read-modify-write:
two writers that both read before either saves lose the first one's change. Every writer of an
EXISTING campaign takes this lock for its whole read-modify-write, which makes them serial within
the process holding it and nowhere else: the lock is a ``threading.RLock`` in this module's memory.

**The guarantee is single-process, and a host has to meet that premise itself.** A host meets it by
building its eval store in one process and nowhere else. A client writing campaigns from several
processes gets no serialisation from this module, and none from the campaign store port either —
``load_campaign`` returns no ETag and ``save_campaign`` is an unconditional upsert — so two of its
processes amending one campaign silently lose a write. Such a client needs its store to serialise
campaign writes on its own. The document-store write path does
carry the mechanism a port could use instead (``DocumentStore.get_with_etag`` paired with the
``if_match`` of :func:`~threetears.evals.contracts.storage.save_document`, which a host's run writes can use),
but the campaign port does not expose it.

It lives in contracts because the writers live in two packages that may not import each other: the
campaign family (:mod:`threetears.evals.analysis.campaigns` — amend, attach, detach, designate a
control) and the run-delete cascade (:mod:`threetears.evals.run.curation`), which detaches a destroyed
run from every campaign holding it. One lock has to be reachable from both, and contracts is the
only package both rows of the dependency matrix admit.

``tests/test_campaign_write_serialization.py`` derives the writers from the code — every
function calling ``save_campaign`` — and refuses one without :func:`serialized_campaign_write`.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable

# Re-entrant, so a writer that calls another writer while holding it does not deadlock on itself.
_CAMPAIGN_WRITES = threading.RLock()


def serialized_campaign_write[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Run ``fn`` — a read-modify-write of an existing campaign — holding the campaign write lock.

    Launch-path membership writes used to run on the event loop, where no other coroutine could
    interleave them; they now run on the eval I/O pool beside the MCP and REST writers, which were
    already threads. Two launches of one campaign, or a launch beside ``campaign_add_runs``, would
    otherwise each read the membership list and the later save would drop the earlier's runs.

    Args:
        fn: The writer.

    Returns:
        ``fn``, serialized.
    """

    @functools.wraps(fn)
    def serialized(*args: P.args, **kwargs: P.kwargs) -> R:
        with _CAMPAIGN_WRITES:
            return fn(*args, **kwargs)

    return serialized


__all__ = [
    "serialized_campaign_write",
]
