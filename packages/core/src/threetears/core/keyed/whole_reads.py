"""the whys a component may read (or write, or drop) a table whole, each counted: one registry, not one a module.

A load path reads by key. The few whole reads it may still make are named in advance, each with what it means, and
counted by table and why the moment one is made, so a replay, a soak or a dashboard sees every one. Only the
``expected`` whys (a true cold start, an operator's reset) may happen in steady state; every other is a change some
writer did not record by key, and a replay holds it at zero (:attr:`WholeReads.gated`).

One component declares one registry at module level::

    WHOLE_READS = WholeReads(
        "copy_follower_whole_reads_total",
        subject="tables a copy follower read whole from L3",
        whys={"cold_start": "a first start with no checkpoint", "reset": "an operator's reset of the table", ...},
        expected={"cold_start", "reset"},
    )
    ...
    WHOLE_READS.count("cold_start", table=table.name)

A why the registry does not name is refused where it is counted (a typo would otherwise count nothing a replay
gates on). Counting is cheap and IO-free: a counter increment and one log line, at INFO for an expected why and at
WARNING for every other.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Any

from threetears.observe import get_logger
from threetears.observe.metrics import counter

__all__ = ["WholeReads"]

log = get_logger(__name__)


class WholeReads:
    """a component's named whole reads, counted by why.

    :param metric: the counter's name (``<component>_whole_reads_total``)
    :ptype metric: str
    :param subject: what the counter counts, the start of its description
    :ptype subject: str
    :param whys: why -> what it means; every why the component may count
    :ptype whys: Mapping[str, str]
    :param expected: the whys that may happen in steady state (a cold start, an operator's reset)
    :ptype expected: Collection[str]
    :param labels: the counter's labels beside ``why``
    :ptype labels: tuple[str, ...]
    :raises ValueError: when no why is named, or an expected why is not one of them
    """

    def __init__(
        self,
        metric: str,
        *,
        subject: str,
        whys: Mapping[str, str],
        expected: Collection[str],
        labels: tuple[str, ...] = ("table",),
    ) -> None:
        if not whys:
            raise ValueError(f"{metric}: a whole-read registry names its whys")
        unknown = sorted(set(expected) - set(whys))
        if unknown:
            raise ValueError(f"{metric}: expected whys {unknown} are not among its whys")
        self._metric = metric
        self._subject = subject
        self._whys = dict(whys)
        self._expected = frozenset(expected)
        self._labels = labels

    @property
    def metric(self) -> str:
        """the counter's name.

        :return: the name
        :rtype: str
        """
        return self._metric

    @property
    def whys(self) -> Mapping[str, str]:
        """every why, and what it means.

        :return: why -> meaning
        :rtype: Mapping[str, str]
        """
        return dict(self._whys)

    @property
    def gated(self) -> frozenset[str]:
        """the whys a replay or a soak holds at zero: every one but the expected.

        :return: the whys
        :rtype: frozenset[str]
        """
        return frozenset(self._whys) - self._expected

    def _counter(self) -> Any:
        return counter(
            self._metric,
            description=f"{self._subject}, by {', '.join(self._labels)} and why ("
            + "; ".join(f"{why}: {what}" for why, what in self._whys.items())
            + f"); zero but {', '.join(sorted(self._expected)) or 'none'}",
            label_names=(*self._labels, "why"),
        )

    def count(self, why: str, **labels: str) -> None:
        """count one whole read for ``why``, and say so in the log.

        :param why: one of :attr:`whys`
        :ptype why: str
        :param labels: the counter's other labels (``table=``)
        :ptype labels: str
        :return: nothing
        :rtype: None
        :raises ValueError: when ``why`` is not one of the registry's, or the labels are not the counter's
        """
        if why not in self._whys:
            raise ValueError(f"{self._metric}: {why!r} is not one of its whys {sorted(self._whys)}")
        if set(labels) != set(self._labels):
            raise ValueError(f"{self._metric}: counts by {list(self._labels)}, got {sorted(labels)}")
        self._counter().labels(**labels, why=why).inc()
        log.log(
            logging.INFO if why in self._expected else logging.WARNING,
            "read whole",
            extra={"extra_data": {"metric": self._metric, "why": why, "what": self._whys[why], **labels}},
        )
