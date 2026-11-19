"""one way to build a process-wide object on first use, once, from any number of threads.

a getter that checks a module-level cache and fills it when the value is missing races the
moment two threads call it first: each thread that looked before the other stored its value
builds another. 3tears met that as ``Duplicated timeseries in CollectorRegistry`` out of
``create_chat_model``, as a second process-wide Claude CLI pool running twice the CLIs its
limits allow, as several isolation roots for one credential, and as a second sync-to-async
bridge loop stranding the work queued on the first. each was fixed by hand-copying the same
check-lock-recheck idiom, which made the fix only as complete as the sweep that found the
sites.

:class:`BuildOnce` is that idiom, once. ``tests/enforcement/test_build_once_is_the_only_lazy_fill.py``
refuses a function that fills a module-level name after checking it by hand, so a new getter
cannot quietly be written without it.

standard library only: every 3tears package already depends on this one, and this one depends
on nothing, so any package may hold its lazily built objects here without a new edge.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from typing import cast

__all__ = ["BuildOnce"]

#: marks "no value stored" apart from a stored value that happens to be ``None`` or ``False``.
_MISSING: object = object()


class BuildOnce[K: Hashable, V]:
    """values built at most once per key, however many threads ask for one first.

    :meth:`get` reads without the lock first, so every call after a key's value exists stays
    lock-free; only a caller that finds nothing takes the lock, looks again, and builds when it
    is still missing. the build runs under the lock, so a second caller for any key waits for
    it rather than building its own -- which is what makes a build that registers something
    process-wide (a prometheus instrument, an ``atexit`` hook, a background thread) happen once.

    one lock per instance, not per key: builds are rare, short and happen at startup, so the
    cost of a caller for one key waiting behind a build for another is bounded by that, and a
    lock per key would need its own build-once to create.

    a value can go stale -- a background loop that stopped, a temp directory something deleted,
    a worker process that exited. *is_current* says whether a stored value may still be handed
    out; one that is not is rebuilt, under the lock, exactly as a missing one would be.

    a cache of many values can be bounded: *cost* prices each built value and *max_cost* caps
    what the stored values may cost together. a value that would take the total past the cap is
    handed to its caller and not stored, so the next :meth:`get` for that key builds it again;
    nothing already stored is evicted for it. that admission rule, rather than evicting the least
    recently used, is what a caller walking the same keys in the same order again and again
    wants: there, eviction drops each value just before it is asked for again and nothing is ever
    found stored, while a full cache that keeps what it has still answers that share of every walk.

    :param is_current: whether a stored value may still be handed out; ``None`` means always
    :ptype is_current: Callable[[V], bool] | None
    :param cost: what storing a value costs, e.g. its size; needs *max_cost*
    :ptype cost: Callable[[V], int] | None
    :param max_cost: the most the stored values may cost together; needs *cost*
    :ptype max_cost: int | None
    :raises ValueError: when only one of *cost* and *max_cost* is given, or *max_cost* is negative
    """

    def __init__(
        self,
        *,
        is_current: Callable[[V], bool] | None = None,
        cost: Callable[[V], int] | None = None,
        max_cost: int | None = None,
    ) -> None:
        """
        prepares an empty cache.

        :param is_current: whether a stored value may still be handed out; ``None`` means always
        :ptype is_current: Callable[[V], bool] | None
        :param cost: what storing a value costs, e.g. its size; needs *max_cost*
        :ptype cost: Callable[[V], int] | None
        :param max_cost: the most the stored values may cost together; needs *cost*
        :ptype max_cost: int | None
        :raises ValueError: when only one of *cost* and *max_cost* is given, or *max_cost* is
            negative
        """
        if (cost is None) != (max_cost is None):
            raise ValueError("BuildOnce bounds by cost and max_cost together: give both or neither")
        if max_cost is not None and max_cost < 0:
            raise ValueError(f"BuildOnce max_cost must be 0 or more, got {max_cost}")
        self._values: dict[K, V] = {}
        self._costs: dict[K, int] = {}
        self._total_cost = 0
        self._lock = threading.Lock()
        self._is_current = is_current
        self._cost = cost
        self._max_cost = max_cost

    def _usable(self, value: object) -> bool:
        """whether *value*, as read from storage, may be handed out.

        :param value: a stored value, or the missing marker
        :ptype value: object
        :return: ``True`` when it is present and current
        :rtype: bool
        """
        usable = value is not _MISSING
        if usable and self._is_current is not None:
            usable = self._is_current(cast("V", value))
        return usable

    def get(self, key: K, build: Callable[[], V]) -> V:
        """the value for *key*, built by *build* when there is none or it is no longer current.

        *build* runs while this instance's lock is held, so it must not call back into this
        same instance. under a *max_cost* a value that does not fit is returned without being
        stored, and a stale value it replaces is dropped either way.

        :param key: which value
        :ptype key: K
        :param build: makes the value; called at most once per missing or stale key
        :ptype build: Callable[[], V]
        :return: the one value for *key*
        :rtype: V
        """
        found: object = self._values.get(key, _MISSING)
        if not self._usable(found):
            with self._lock:
                found = self._values.get(key, _MISSING)
                if not self._usable(found):
                    built = build()
                    self._store(key, built)
                    found = built
        return cast("V", found)

    def _store(self, key: K, value: V) -> None:
        """store *value* for *key* when it fits the cost bound; called with the lock held.

        :param key: which value
        :ptype key: K
        :param value: the value just built
        :ptype value: V
        """
        if self._cost is None or self._max_cost is None:
            self._values[key] = value
            return
        self._drop(key)
        price = self._cost(value)
        if self._total_cost + price <= self._max_cost:
            self._values[key] = value
            self._costs[key] = price
            self._total_cost += price

    def _drop(self, key: K) -> object:
        """take *key*'s value and its cost out; called with the lock held.

        :param key: which value
        :ptype key: K
        :return: the value that was stored, or the missing marker
        :rtype: object
        """
        self._total_cost -= self._costs.pop(key, 0)
        return self._values.pop(key, _MISSING)

    def peek(self, key: K) -> V | None:
        """the value stored for *key*, current or not, without building one.

        :param key: which value
        :ptype key: K
        :return: the stored value, or ``None`` when there is none
        :rtype: V | None
        """
        value: object = self._values.get(key, _MISSING)
        return None if value is _MISSING else cast("V", value)

    def pop(self, key: K) -> V | None:
        """take *key*'s value out, so the next :meth:`get` builds a new one.

        the caller owns what it took: closing a pool, stopping a loop or killing a worker is
        its job, done outside the lock, since this returns before any of that starts.

        :param key: which value
        :ptype key: K
        :return: the value that was stored, or ``None`` when there was none
        :rtype: V | None
        """
        with self._lock:
            value = self._drop(key)
        return None if value is _MISSING else cast("V", value)

    def clear(self, dispose: Callable[[V], object] | None = None) -> None:
        """drop every value, handing each to *dispose* first.

        *dispose* runs under the lock, so no :meth:`get` can build a replacement while an old
        value is still being taken apart -- a replacement prometheus emitter registering its
        instruments before the old one unregistered them would raise ``Duplicated timeseries``.
        like *build*, it must not call back into this same instance.

        :param dispose: called once per stored value before it is dropped; ``None`` drops them
        :ptype dispose: Callable[[V], object] | None
        """
        with self._lock:
            if dispose is not None:
                for value in self._values.values():
                    dispose(value)
            self._values.clear()
            self._costs.clear()
            self._total_cost = 0
