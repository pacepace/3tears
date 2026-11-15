"""Shared-memory autosize, the 3tears half: measure a pod's shared memory, export it, and decide "need N bytes".

**Why.** A pod's shared-memory volume (``/dev/shm``, or an ``emptyDir`` with ``medium: Memory``) is sized once, in its
deploy config, and its pages count against the container's memory limit. When the data grows past it, the store
refuses writes (cobalt prod, 2026-11-04: every change refused from 14:16Z on election day) and a pod whose
``emptyDir`` passes its ``sizeLimit`` is evicted. Sizing it by hand after the page is too late; sizing every pod for
the worst night wastes memory every other night. So the pod measures its own need and the platform raises the volume
and the memory limit together, automatically up to a per-product limit, and pages people past it.

**The split** (Pace, 2026-11-12). 3tears is the mechanism and ships no actuator: it measures, exports the metrics,
decides the need, and hands it as a platform record (:class:`SizeRequest`) to an :class:`Actuator` the consumer
supplies (for us one implementation in the hub and SDK that applies it through the deploy path, building beside and
swapping, and enforcing the product's limit, never above half a node). Products set only ``enabled``, ``threshold``
and ``limit`` (:class:`AutosizeSettings`).

**Off by default.** Disabled, it measures nothing, exports nothing and starts no thread. Enabled with no actuator, it
refuses at startup: a need decided and handed nowhere would look like a working autosize that never acts.

**The decision** (:func:`_decide`). When the volume's measured peak, the container's memory, or a store's in-use set
(``KEEP`` bytes, ``threetears.core.eviction``) crosses the threshold (default 75%) of its size, limit or budget, it asks
for the volume doubled per step until the peak sits under the threshold, and the memory limit raised by the same
growth (the volume's pages are the container's memory). One need is one record (``request_id``, the product and the
size), handed once however often it is measured; the actuator acts on an id at most once.

**The measure** (:func:`measure`) is a ``statvfs`` of the volume and two small cgroup file reads, on a thread of its own
every ``interval_seconds`` (10 s), never the event loop: about 30 us a sample on a laptop (2026-11-15), nothing at 10 s.

**Metrics** (the cache family, ``tier="shared"``, ``name`` the product): ``threetears_cache_volume_used_bytes``,
``threetears_cache_volume_size_bytes``, ``threetears_cache_rss_bytes``, ``threetears_cache_memory_limit_bytes``,
``threetears_cache_requested_bytes`` and ``threetears_cache_autosize_limit_bytes``. No alert rules here: the page past
the product limit is configured in Grafana (``requested_bytes > autosize_limit_bytes``).
"""

from __future__ import annotations

import contextlib
import os
import resource
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from threetears.core.eviction import Tier
from threetears.observe import contain, get_logger
from threetears.observe.metrics import gauge

__all__ = [
    "DEFAULT_LIMIT_BYTES",
    "DEFAULT_THRESHOLD",
    "Actuator",
    "AutosizeSettings",
    "Sample",
    "ShmAutosize",
    "SizeRequest",
    "measure",
]

log = get_logger(__name__)

#: the share of a size, limit or budget whose use asks for more (Pace, 2026-11-12: settable per product)
DEFAULT_THRESHOLD: Final = 0.75

#: a product's own limit on what is asked for automatically: past it the platform pages people instead (Pace's
#: proposed default; never above half a node, which the actuator enforces)
DEFAULT_LIMIT_BYTES: Final = 16 << 30

#: how often the pod is measured: a sample costs about 30 us (a statvfs and two cgroup reads), and a volume fills in
#: minutes (the election night's runs, about 60 an hour), never in seconds
_DEFAULT_INTERVAL_SECONDS: Final = 10.0

#: where the container's cgroup filesystem is mounted
_CGROUP_ROOT: Final = Path("/sys/fs/cgroup")

#: a cgroup v1 memory limit at or past this is "no limit" (the kernel writes the page-rounded largest signed value)
_V1_NO_LIMIT: Final = 1 << 62

_TIER: Final = Tier.SHARED.value
_LABELS: Final = ("tier", "name")

_VOLUME_USED = gauge(
    "threetears_cache_volume_used_bytes",
    description="bytes the pod's shared-memory volume holds now",
    label_names=_LABELS,
)
_VOLUME_SIZE = gauge(
    "threetears_cache_volume_size_bytes",
    description="the pod's shared-memory volume's size",
    label_names=_LABELS,
)
_RSS = gauge(
    "threetears_cache_rss_bytes",
    description="the container's memory in use (cgroup), the volume's pages included",
    label_names=_LABELS,
)
_MEMORY_LIMIT = gauge(
    "threetears_cache_memory_limit_bytes",
    description="the container's memory limit (cgroup); 0 when none",
    label_names=_LABELS,
)
_REQUESTED = gauge(
    "threetears_cache_requested_bytes",
    description="the shared-memory volume size autosize asked for last; the volume's size until it asks",
    label_names=_LABELS,
)
_LIMIT = gauge(
    "threetears_cache_autosize_limit_bytes",
    description="the product's limit on what autosize asks for automatically: past it, people are paged",
    label_names=_LABELS,
)


@dataclass(frozen=True)
class AutosizeSettings:
    """a product's autosize settings, from its deploy config.

    :ivar enabled: whether to measure and ask at all; off by default
    :ivar threshold: the share of a size, limit or budget whose use asks for more, within (0, 1)
    :ivar limit_bytes: the most asked for automatically; the actuator pages past it
    :ivar product: the deployed unit asking (hub, reports, survey, a shared pod): the record's and the metrics' name
    :ivar interval_seconds: how often the pod is measured
    """

    enabled: bool = False
    threshold: float = DEFAULT_THRESHOLD
    limit_bytes: int = DEFAULT_LIMIT_BYTES
    product: str = ""
    interval_seconds: float = _DEFAULT_INTERVAL_SECONDS


@dataclass(frozen=True)
class Sample:
    """one measure of the pod.

    :ivar volume_used_bytes: what the shared-memory volume holds
    :ivar volume_size_bytes: its size
    :ivar memory_used_bytes: the container's memory in use (the process's resident memory with no cgroup)
    :ivar memory_limit_bytes: its limit; None for none
    :ivar store_in_use_bytes: what the stores hold that they must keep (``KEEP``: held less evictable)
    :ivar store_budget_bytes: the stores' budgets together; 0 for no store
    """

    volume_used_bytes: int
    volume_size_bytes: int
    memory_used_bytes: int
    memory_limit_bytes: int | None
    store_in_use_bytes: int = 0
    store_budget_bytes: int = 0


@dataclass(frozen=True)
class SizeRequest:
    """the platform record of one need: the shared-memory volume a product's pod asks for. An actuator acts on one
    ``request_id`` at most once (building beside, then swapping); the same need measured again is the same id.

    :ivar request_id: ``{product}:{size_bytes}``: one need, one id
    :ivar product: the deployed unit
    :ivar size_bytes: the volume size asked for
    :ivar memory_limit_bytes: the memory limit asked for: today's raised by the volume's growth; None when the
        container has none
    :ivar volume_bytes: the volume's size when asked
    :ivar peak_bytes: the most the volume was measured holding
    :ivar limit_bytes: the product's limit
    :ivar past_limit: whether ``size_bytes`` passes it (the actuator pages instead of applying)
    :ivar reason: what crossed the threshold: ``"volume"``, ``"memory"`` or ``"store"``
    """

    request_id: str
    product: str
    size_bytes: int
    memory_limit_bytes: int | None
    volume_bytes: int
    peak_bytes: int
    limit_bytes: int
    past_limit: bool
    reason: str


class Actuator(Protocol):
    """what applies a need: supplied by the consumer (3tears ships none). Called on the autosize thread, never the
    event loop; it may raise, and the need is handed again at the next sample."""

    def request(self, need: SizeRequest) -> None:
        """act on ``need`` at most once per ``need.request_id``: apply it up to the product's limit, page past it.

        :param need: the record
        :ptype need: SizeRequest
        :return: nothing
        :rtype: None
        """


def _read_int(path: Path) -> int | None:
    """a cgroup file's number; None when it says ``max`` or is missing.

    :param path: the file
    :ptype path: Path
    :return: the number, or None
    :rtype: int | None
    """
    try:
        text = path.read_text().strip()
    except OSError:
        # NOSILENT: no such cgroup file here: the caller tries the next source
        return None
    return None if text == "max" else int(text)


def _process_resident() -> int:
    """this process's resident memory, where there is no cgroup to read (a laptop).

    :return: bytes
    :rtype: int
    """
    with contextlib.suppress(OSError):  # NOSILENT: no /proc: the peak below
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if os.uname().sysname == "Darwin" else peak * 1024


def measure(volume: Path | str, *, cgroup_root: Path = _CGROUP_ROOT, stores: Sequence[object] = ()) -> Sample:
    """measure the pod: the volume's use and size (``statvfs``), the container's memory and limit (cgroup v2
    ``memory.current``/``memory.max``, else v1 ``memory/memory.usage_in_bytes``/``memory.limit_in_bytes``, else this
    process's resident memory and no limit), and each store's ``KEEP`` bytes against its budget (its ``status()``,
    read from tallies, never the file system). Blocking: on a thread.

    :param volume: the shared-memory volume's mount
    :ptype volume: Path | str
    :param cgroup_root: where the cgroup filesystem is mounted
    :ptype cgroup_root: Path
    :param stores: shared Arrow stores on the volume (anything with ``status()`` giving ``kept_bytes`` and
        ``max_bytes``)
    :ptype stores: Sequence[object]
    :return: the sample
    :rtype: Sample
    """
    stat = os.statvfs(volume)
    size = stat.f_blocks * stat.f_frsize
    used = (stat.f_blocks - stat.f_bfree) * stat.f_frsize
    memory = _read_int(cgroup_root / "memory.current")
    limit = _read_int(cgroup_root / "memory.max")
    if memory is None:
        memory = _read_int(cgroup_root / "memory" / "memory.usage_in_bytes")
        limit = _read_int(cgroup_root / "memory" / "memory.limit_in_bytes")
        if limit is not None and limit >= _V1_NO_LIMIT:
            limit = None
    if memory is None:
        memory, limit = _process_resident(), None
    in_use = budget = 0
    for store in stores:
        status = store.status()  # type: ignore[attr-defined]
        in_use += int(status["kept_bytes"])
        budget += int(status["max_bytes"])
    return Sample(
        volume_used_bytes=used,
        volume_size_bytes=size,
        memory_used_bytes=memory,
        memory_limit_bytes=limit,
        store_in_use_bytes=in_use,
        store_budget_bytes=budget,
    )


def _decide(sample: Sample, peak: int, settings: AutosizeSettings) -> SizeRequest | None:
    """the need, if any: a use at or past the threshold asks for the volume doubled per step until ``peak`` sits under
    the threshold (at least one doubling), and the memory limit raised by the same growth.

    Doubling, not "peak / threshold": each step is a roll of the pod (build beside, then swap), so a step too small
    costs another roll within the hour, and a doubling's waste is at most the size before it, under the product's
    limit.

    :param sample: the measure now
    :ptype sample: Sample
    :param peak: the most the volume was measured holding
    :ptype peak: int
    :param settings: the product's settings
    :ptype settings: AutosizeSettings
    :return: the record, or None
    :rtype: SizeRequest | None
    """
    threshold = settings.threshold
    size = sample.volume_size_bytes
    reason: str | None = None
    if size > 0 and peak >= threshold * size:
        reason = "volume"
    elif sample.memory_limit_bytes and sample.memory_used_bytes >= threshold * sample.memory_limit_bytes:
        reason = "memory"
    elif sample.store_budget_bytes and sample.store_in_use_bytes >= threshold * sample.store_budget_bytes:
        reason = "store"
    if reason is None or size <= 0:
        return None
    asked = size * 2
    while peak >= threshold * asked:
        asked *= 2
    limit = sample.memory_limit_bytes
    return SizeRequest(
        request_id=f"{settings.product}:{asked}",
        product=settings.product,
        size_bytes=asked,
        memory_limit_bytes=None if limit is None else limit + (asked - size),
        volume_bytes=size,
        peak_bytes=peak,
        limit_bytes=settings.limit_bytes,
        past_limit=asked > settings.limit_bytes,
        reason=reason,
    )


class ShmAutosize:
    """the pod's autosize: off unless enabled, then a thread measuring every ``interval_seconds``, exporting the
    gauges and handing each need to the actuator once.

    :param settings: the product's settings
    :ptype settings: AutosizeSettings
    :param actuator: what applies a need; required when enabled
    :ptype actuator: Actuator | None
    :param volume: the shared-memory volume's mount
    :ptype volume: Path | str
    :param stores: the shared Arrow stores on it, whose ``KEEP`` bytes against their budgets also ask for room
    :ptype stores: Sequence[object]
    :param measure: what measures the pod (a test's fake); None for :func:`measure` over ``volume`` and ``stores``
    :ptype measure: Callable[[], Sample] | None
    :raises ValueError: when enabled with no actuator or no product, or with a threshold outside (0, 1) or a limit
        that is not positive
    """

    def __init__(
        self,
        settings: AutosizeSettings,
        actuator: Actuator | None = None,
        *,
        volume: Path | str = "/dev/shm",
        stores: Sequence[object] = (),
        measure: Callable[[], Sample] | None = None,
    ) -> None:
        if settings.enabled:
            if actuator is None:
                raise ValueError(
                    "shared-memory autosize is enabled with no actuator: a need would be decided and applied nowhere"
                )
            if not settings.product:
                raise ValueError("shared-memory autosize needs the product it sizes")
        if not 0 < settings.threshold < 1:
            raise ValueError(f"an autosize threshold is within (0, 1), not {settings.threshold}")
        if settings.limit_bytes <= 0:
            raise ValueError("an autosize limit must be positive")
        self._settings = settings
        self._actuator = actuator
        self._measure: Callable[[], Sample] = measure or (lambda: _measured(volume, stores))
        self._peak = 0
        #: the needs handed, by id: never twice
        self._handed: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """start sampling on a thread of its own; nothing when disabled.

        :return: nothing
        :rtype: None
        """
        if not self._settings.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="shm-autosize", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """stop sampling and wait for the thread.

        :return: nothing
        :rtype: None
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join()

    def sample_once(self) -> SizeRequest | None:
        """measure, export, decide, and hand a new need to the actuator: one tick. Blocking: on a thread. Nothing when
        disabled.

        :return: the need decided now, or None
        :rtype: SizeRequest | None
        """
        settings = self._settings
        if not settings.enabled or self._actuator is None:
            return None
        sample = self._measure()
        self._peak = max(self._peak, sample.volume_used_bytes)
        need = _decide(sample, self._peak, settings)
        name = settings.product
        _VOLUME_USED.labels(tier=_TIER, name=name).set(sample.volume_used_bytes)
        _VOLUME_SIZE.labels(tier=_TIER, name=name).set(sample.volume_size_bytes)
        _RSS.labels(tier=_TIER, name=name).set(sample.memory_used_bytes)
        _MEMORY_LIMIT.labels(tier=_TIER, name=name).set(sample.memory_limit_bytes or 0)
        _LIMIT.labels(tier=_TIER, name=name).set(settings.limit_bytes)
        _REQUESTED.labels(tier=_TIER, name=name).set(sample.volume_size_bytes if need is None else need.size_bytes)
        if need is not None and need.request_id not in self._handed:
            try:
                self._actuator.request(need)
            except Exception as exc:  # noqa: BLE001 -- the actuator is the consumer's; contained, handed again next tick
                contain(
                    exc, log=log, what="shared-memory autosize: the actuator refused a need; asked again next sample"
                )
            else:
                self._handed.add(need.request_id)
                log.warning(
                    "shared-memory autosize asked for a larger volume",
                    extra={
                        "extra_data": {
                            "request_id": need.request_id,
                            "size_bytes": need.size_bytes,
                            "volume_bytes": need.volume_bytes,
                            "peak_bytes": need.peak_bytes,
                            "past_limit": need.past_limit,
                            "reason": need.reason,
                        }
                    },
                )
        return need

    def _run(self) -> None:
        """sample every ``interval_seconds`` until stopped: the autosize thread.

        :return: nothing
        :rtype: None
        """
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception as exc:  # noqa: BLE001 -- a thread has no caller to raise to; contained, so it is counted
                contain(exc, log=log, what="shared-memory autosize could not sample; it samples again next tick")
            self._stop.wait(self._settings.interval_seconds)


def _measured(volume: Path | str, stores: Sequence[object]) -> Sample:
    """:func:`measure` over the volume and stores, with the default cgroup root.

    :param volume: the volume
    :ptype volume: Path | str
    :param stores: the stores
    :ptype stores: Sequence[object]
    :return: the sample
    :rtype: Sample
    """
    return measure(volume, stores=stores)
