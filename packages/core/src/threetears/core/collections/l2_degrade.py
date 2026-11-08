"""What a collection does with an L2 (NATS KV) failure: count it, say why, and fall through or raise.

Where a durable tier stands behind L2, L2 is a cache of it: a read L2 could not answer goes to L3, and a
write L2 refused is a cache miss for the next reader. That fall-through is correct, and it must not be
quiet. :func:`degrade_l2` is the one place every such site sends its failure, and it

- names the failure's kind: ``full`` (a bucket or account bound reached),
  ``exhausted`` (the server out of storage), ``too_large`` (the value over a size limit), ``transport``
  (the broker unreachable, slow, or the connection gone) or ``other`` (the server refused for another
  cause), since each has a different remedy and a full bucket must never read as an outage;
- counts it on ``threetears_l2_degraded_total{table, op, reason}``, which is what an alert reads;
- logs a WARNING at most once a minute for each table, operation and
  reason, carrying how many it held back since the last, so a sustained failure does not bury every
  other line in the log;
- raises instead, where L2 is the collection's only store (``l2_is_the_store``): nothing stands behind
  it to fall through to, and a fall-through there would answer "absent" for a row that exists.
"""

from __future__ import annotations

import threading
import time
from typing import Final, Literal

from threetears.nats.errors import KvError, KvFullError, KvStoreExhaustedError, KvValueTooLargeError
from threetears.observe import BuildOnce, get_logger
from threetears.observe.metrics import counter

__all__ = ["degrade_l2"]

log = get_logger(__name__)

#: the kinds of L2 failure, the counter's ``reason`` label
type _Reason = Literal["full", "exhausted", "too_large", "transport", "other"]

#: the fewest seconds between two WARNINGs for one table, operation and reason
_LOG_INTERVAL_SECONDS: Final = 60.0

#: the counter's name and help. Fetched where it is incremented, never held at import: a module that a
#: test re-imports would otherwise register the same name in the process's registry a second time.
_DEGRADED_NAME: Final = "threetears_l2_degraded_total"
_DEGRADED_HELP: Final = "L2 (NATS KV) failures a collection fell through to L3 on, or raised where L2 is the store"


class _LogWindow:
    """one table, operation and reason's WARNING window: when its last line was written, and how many
    were held back since."""

    def __init__(self) -> None:
        # ABSENCE means never logged -- ``time.monotonic()`` counts from boot, so a zero would hold
        # back a fresh machine's first line.
        self._last: float | None = None
        self._held_back = 0
        self._lock = threading.Lock()

    def due(self) -> int | None:
        """whether a WARNING is due now, and if so how many were held back since the last.

        :return: how many were held back, when one is due; ``None`` when this one is held back
        :rtype: int | None
        """
        now = time.monotonic()
        with self._lock:
            if self._last is not None and now - self._last < _LOG_INTERVAL_SECONDS:
                self._held_back += 1
                return None
            self._last = now
            held_back, self._held_back = self._held_back, 0
            return held_back


#: one window per (table, op, reason)
_windows: BuildOnce[tuple[str, str, str], _LogWindow] = BuildOnce()


def _reason(exc: KvError) -> _Reason:
    """the kind of an L2 failure, as the ``reason`` it is counted under.

    :param exc: the failure
    :ptype exc: KvError
    :return: ``full``, ``exhausted``, ``too_large``, ``transport`` or ``other``
    :rtype: _Reason
    """
    if isinstance(exc, KvFullError):
        return "full"
    if isinstance(exc, KvStoreExhaustedError):
        return "exhausted"
    if isinstance(exc, KvValueTooLargeError):
        return "too_large"
    # the server answered, with a refusal that is none of the above; else it did not answer at all
    return "other" if exc.server_error_code is not None else "transport"


def degrade_l2(
    exc: KvError,
    *,
    table: str,
    op: str,
    what: str,
    l2_is_the_store: bool = False,
    refused_as: Exception | None = None,
    entity_id: object = None,
) -> None:
    """count an L2 failure, then raise it where L2 is the store, or log it where the caller falls through.

    Call it from the ``except KvError`` block of every L2 access, in place of its own log line. Where it
    returns, the caller carries on as it always has: to L3, or to a miss.

    :param exc: the failure
    :ptype exc: KvError
    :param table: the collection's table, the counter's ``table`` label
    :ptype table: str
    :param op: the L2 operation that failed (``get``, ``put``, ``delete``, ``seed`` ...), the ``op`` label
    :ptype op: str
    :param what: the WARNING's message, naming what the caller does instead
    :ptype what: str
    :param l2_is_the_store: whether L2 is this collection's only store; when it is, the failure is raised
    :ptype l2_is_the_store: bool
    :param refused_as: what to raise where L2 is the store, chained to ``exc``; ``exc`` itself when ``None``
    :ptype refused_as: Exception | None
    :param entity_id: the row's key, for the log; ``None`` when the failure is not one row's
    :ptype entity_id: object
    :return: nothing
    :rtype: None
    :raises KvError: ``exc`` itself, where L2 is the store and ``refused_as`` is ``None``
    :raises Exception: ``refused_as``, where L2 is the store
    """
    reason = _reason(exc)
    counter(_DEGRADED_NAME, description=_DEGRADED_HELP, label_names=("table", "op", "reason")).labels(
        table=table, op=op, reason=reason
    ).inc()
    if l2_is_the_store:
        if refused_as is not None:
            raise refused_as from exc
        raise exc
    held_back = _windows.get((table, op, reason), _LogWindow).due()
    if held_back is None:
        return
    log.warning(
        what,
        extra={
            "extra_data": {
                "table": table,
                "op": op,
                "reason": reason,
                "entity_id": None if entity_id is None else str(entity_id),  # convert at border: log field
                "error": f"{type(exc).__name__}: {exc}",
                "suppressed": held_back,
            }
        },
    )
