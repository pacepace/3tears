"""@metered decorator -- zero-cost without prometheus_client.

Creates prometheus counters and a duration histogram around sync and
async functions, mirroring :func:`threetears.observe.tracing.traced`'s
design exactly: bare ``@metered`` or parameterised ``@metered(name=...)``,
works transparently on sync and async functions, and degrades to a pure
passthrough with no overhead beyond a single cached bool check when
``prometheus_client`` is not installed.

unlike :func:`traced` (OpenTelemetry spans, push/export-based), this
module's backend is ``prometheus_client`` (pull/scrape-based), matching
:class:`threetears.observe.inflight.InflightRequestsGauge`'s existing
"prometheus is OPTIONAL, lazy import, silent no-op when absent" contract
-- the two telemetry backends in this package are deliberately different
tools for different jobs (traces answer "what happened on this one
request," metrics answer "what's the aggregate rate/latency/error-count
over time"), not a stylistic inconsistency.
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, TypeVar, overload

__all__ = ["metered"]

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# prometheus_client availability check (cached after first probe)
# ---------------------------------------------------------------------------

_prometheus_available: bool | None = None


def _check_prometheus() -> bool:
    """Check if prometheus_client is importable (cached after first check)."""
    global _prometheus_available  # noqa: PLW0603
    if _prometheus_available is None:
        try:
            import prometheus_client  # noqa: F401

            _prometheus_available = True
        except ImportError:
            _prometheus_available = False
    return _prometheus_available


# ---------------------------------------------------------------------------
# metric name sanitization and lazy instrument cache
# ---------------------------------------------------------------------------

#: characters prometheus metric names commonly avoid in practice across
#: this codebase's own examples (InflightRequestsGauge's docstring uses
#: snake_case, no dots) -- module.qualname defaults contain dots (module
#: paths) and angle brackets (e.g. ``<locals>`` for nested functions),
#: neither of which reads well as a prometheus identifier even though the
#: client library itself tolerates dots.
_SANITIZE_TRANSLATION = str.maketrans({".": "_", "<": "_", ">": "_"})


def _sanitize_metric_name(name: str) -> str:
    """Translate a dotted module.qualname into a prometheus-friendly identifier.

    :param name: raw name, typically ``f"{fn.__module__}.{fn.__qualname__}"``
    :ptype name: str
    :return: sanitized name safe to use as a prometheus metric name prefix
    :rtype: str
    """
    return name.translate(_SANITIZE_TRANSLATION)


#: lazily created (Counter, Histogram) pairs, keyed by sanitized metric
#: name -- created once per distinct name, reused across every call,
#: mirroring how a real Counter/Histogram is meant to be a long-lived
#: module-level object rather than recreated per call.
_counters: dict[str, Any] = {}
_histograms: dict[str, Any] = {}


def _get_or_create_instruments(metric_name: str) -> tuple[Any, Any]:
    """Lazily create (and cache) the Counter/Histogram pair for *metric_name*.

    :param metric_name: sanitized base name for this call site's metrics
    :ptype metric_name: str
    :return: tuple of (calls counter, duration histogram)
    :rtype: tuple[Any, Any]
    """
    if metric_name not in _counters:
        from prometheus_client import Counter, Histogram

        _counters[metric_name] = Counter(
            f"{metric_name}_calls",
            f"calls to {metric_name}",
            ["status"],
        )
        _histograms[metric_name] = Histogram(
            f"{metric_name}_duration_seconds",
            f"duration of {metric_name} in seconds",
        )
    return _counters[metric_name], _histograms[metric_name]


# ---------------------------------------------------------------------------
# @metered decorator
# ---------------------------------------------------------------------------


@overload
def metered(func: F) -> F: ...


@overload
def metered(*, name: str | None = None) -> Callable[[F], F]: ...


def metered(
    func: F | None = None,
    *,
    name: str | None = None,
) -> F | Callable[[F], F]:
    """Decorator that records prometheus call-count and duration metrics for a function.

    Supports both bare ``@metered`` and parameterised ``@metered(name=...)``
    usage.  Works for sync and async functions.

    Records two instruments per decorated call site: a ``<name>_calls_total``
    counter labelled ``status`` (``"success"`` or ``"error"``), and a
    ``<name>_duration_seconds`` histogram.  Both instruments are created
    once (lazily, on first call) and reused for every subsequent call.

    When ``prometheus_client`` is not installed, the decorator is a pure
    passthrough with no overhead beyond a single bool check per call --
    identical zero-cost contract to :func:`threetears.observe.tracing.traced`.

    :param name: explicit metric name prefix; defaults to sanitized
        ``module.qualname``
    :ptype name: str | None
    :return: decorated function, or a decorator when called with arguments
    :rtype: F | Callable[[F], F]
    """

    def decorator(fn: F) -> F:
        metric_name = _sanitize_metric_name(name or f"{fn.__module__}.{fn.__qualname__}")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _check_prometheus():
                    return await fn(*args, **kwargs)

                counter, histogram = _get_or_create_instruments(metric_name)
                start = time.monotonic()
                try:
                    result = await fn(*args, **kwargs)
                except Exception:
                    counter.labels(status="error").inc()
                    raise
                else:
                    counter.labels(status="success").inc()
                    return result
                finally:
                    histogram.observe(time.monotonic() - start)

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _check_prometheus():
                    return fn(*args, **kwargs)

                counter, histogram = _get_or_create_instruments(metric_name)
                start = time.monotonic()
                try:
                    result = fn(*args, **kwargs)
                except Exception:
                    counter.labels(status="error").inc()
                    raise
                else:
                    counter.labels(status="success").inc()
                    return result
                finally:
                    histogram.observe(time.monotonic() - start)

            return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator
