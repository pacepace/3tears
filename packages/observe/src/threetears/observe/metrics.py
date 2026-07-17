"""Prometheus metrics -- zero-cost without prometheus_client.

Provides two layers:

- :func:`counter`/:func:`histogram`/:func:`gauge` -- get-or-create
  accessors for named prometheus instruments, cached after first call.
  Any code, decorated or not, can record an arbitrary custom metric this
  way (e.g. ``histogram("survey_completion_rate").observe(rate)`` for a
  value only known after a function's result is computed, which no
  decorator can see). When ``prometheus_client`` is not installed, every
  accessor returns a no-op stand-in whose ``.inc()``/``.observe()``/
  ``.set()``/``.labels()`` methods are all safe to call and do nothing.
- :func:`metered` -- a decorator built on top of the accessors above,
  mirroring :func:`threetears.observe.tracing.traced`'s design exactly:
  bare ``@metered`` or parameterised ``@metered(name=...)``, works
  transparently on sync and async functions, zero overhead beyond one
  cached bool check when the backend is absent.

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

__all__ = ["counter", "gauge", "histogram", "metered"]

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
# metric name sanitization
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


# ---------------------------------------------------------------------------
# no-op fallback returned by every accessor when prometheus_client is absent
# ---------------------------------------------------------------------------


class _NoOpMetric:
    """Safe stand-in for a Counter/Histogram/Gauge when prometheus_client is absent.

    every method accepts and ignores arbitrary arguments, and
    :meth:`labels` returns ``self`` so label-chained call sites
    (``counter("x").labels(status="ok").inc()``) work unmodified whether
    or not the real backend is installed.
    """

    def inc(self, *args: Any, **kwargs: Any) -> None:
        """No-op increment.

        :param args: ignored
        :ptype args: Any
        :param kwargs: ignored
        :ptype kwargs: Any
        :return: none
        :rtype: None
        """

    def dec(self, *args: Any, **kwargs: Any) -> None:
        """No-op decrement.

        :param args: ignored
        :ptype args: Any
        :param kwargs: ignored
        :ptype kwargs: Any
        :return: none
        :rtype: None
        """

    def observe(self, *args: Any, **kwargs: Any) -> None:
        """No-op histogram observation.

        :param args: ignored
        :ptype args: Any
        :param kwargs: ignored
        :ptype kwargs: Any
        :return: none
        :rtype: None
        """

    def set(self, *args: Any, **kwargs: Any) -> None:
        """No-op gauge set.

        :param args: ignored
        :ptype args: Any
        :param kwargs: ignored
        :ptype kwargs: Any
        :return: none
        :rtype: None
        """

    def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
        """No-op label binding, returns self for chaining.

        :param args: ignored
        :ptype args: Any
        :param kwargs: ignored
        :ptype kwargs: Any
        :return: this same no-op instance
        :rtype: _NoOpMetric
        """
        return self


_NOOP_METRIC = _NoOpMetric()


# ---------------------------------------------------------------------------
# get-or-create instrument accessors
# ---------------------------------------------------------------------------

#: lazily created instruments, keyed by (kind, sanitized name, label_names)
#: -- created once per distinct key, reused across every call, mirroring
#: how a real Counter/Histogram/Gauge is meant to be a long-lived
#: module-level object rather than recreated per call.
_instruments: dict[tuple[str, str, tuple[str, ...]], Any] = {}


def _get_or_create_instrument(
    kind: str,
    name: str,
    description: str | None,
    label_names: tuple[str, ...],
) -> Any:
    """Lazily create (and cache) a named prometheus instrument.

    :param kind: one of ``"counter"``, ``"histogram"``, ``"gauge"``
    :ptype kind: str
    :param name: raw metric name, sanitized before use
    :ptype name: str
    :param description: metric help text; a generic default is used when omitted
    :ptype description: str | None
    :param label_names: label dimension names; empty for an unlabelled instrument
    :ptype label_names: tuple[str, ...]
    :return: the cached prometheus instrument, or a no-op stand-in when
        prometheus_client is unavailable
    :rtype: Any
    """
    if not _check_prometheus():
        return _NOOP_METRIC

    sanitized = _sanitize_metric_name(name)
    key = (kind, sanitized, label_names)
    if key not in _instruments:
        from prometheus_client import Counter, Gauge, Histogram

        instrument_classes = {"counter": Counter, "histogram": Histogram, "gauge": Gauge}
        instrument_class = instrument_classes[kind]
        _instruments[key] = instrument_class(sanitized, description or f"{sanitized} {kind}", list(label_names))
    return _instruments[key]


def counter(
    name: str,
    *,
    description: str | None = None,
    label_names: tuple[str, ...] = (),
) -> Any:
    """Get or create a named prometheus Counter (cached across calls).

    Returns a no-op stand-in with an identical ``.inc()``/``.labels()``
    surface when ``prometheus_client`` is not installed, so call sites
    never need their own availability check.

    :param name: metric name; dots/angle-brackets sanitized automatically
    :ptype name: str
    :param description: metric help text; defaults to a generic description
    :ptype description: str | None
    :param label_names: label dimension names; empty for an unlabelled counter
    :ptype label_names: tuple[str, ...]
    :return: prometheus Counter, or a no-op stand-in
    :rtype: Any
    """
    return _get_or_create_instrument("counter", name, description, label_names)


def histogram(
    name: str,
    *,
    description: str | None = None,
    label_names: tuple[str, ...] = (),
) -> Any:
    """Get or create a named prometheus Histogram (cached across calls).

    Returns a no-op stand-in with an identical ``.observe()``/``.labels()``
    surface when ``prometheus_client`` is not installed, so call sites
    never need their own availability check.

    :param name: metric name; dots/angle-brackets sanitized automatically
    :ptype name: str
    :param description: metric help text; defaults to a generic description
    :ptype description: str | None
    :param label_names: label dimension names; empty for an unlabelled histogram
    :ptype label_names: tuple[str, ...]
    :return: prometheus Histogram, or a no-op stand-in
    :rtype: Any
    """
    return _get_or_create_instrument("histogram", name, description, label_names)


def gauge(
    name: str,
    *,
    description: str | None = None,
    label_names: tuple[str, ...] = (),
) -> Any:
    """Get or create a named prometheus Gauge (cached across calls).

    Returns a no-op stand-in with an identical ``.inc()``/``.dec()``/
    ``.set()``/``.labels()`` surface when ``prometheus_client`` is not
    installed, so call sites never need their own availability check.

    :param name: metric name; dots/angle-brackets sanitized automatically
    :ptype name: str
    :param description: metric help text; defaults to a generic description
    :ptype description: str | None
    :param label_names: label dimension names; empty for an unlabelled gauge
    :ptype label_names: tuple[str, ...]
    :return: prometheus Gauge, or a no-op stand-in
    :rtype: Any
    """
    return _get_or_create_instrument("gauge", name, description, label_names)


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
    usage.  Works for sync and async functions.  Built on top of
    :func:`counter`/:func:`histogram` -- functions needing additional,
    result-derived, or business-specific metrics can call those accessors
    directly from inside the wrapped function body.

    Records two instruments per decorated call site: a ``<name>_calls_total``
    counter labelled ``status`` (``"success"`` or ``"error"``), and a
    ``<name>_duration_seconds`` histogram.  Both instruments are resolved
    once, at decoration time (via :func:`counter`/:func:`histogram`'s own
    cache), and reused for every subsequent call.

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
        metric_name = name or f"{fn.__module__}.{fn.__qualname__}"
        calls_counter = counter(f"{metric_name}_calls", description=f"calls to {metric_name}", label_names=("status",))
        duration_histogram = histogram(
            f"{metric_name}_duration_seconds", description=f"duration of {metric_name} in seconds"
        )

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _check_prometheus():
                    return await fn(*args, **kwargs)

                start = time.monotonic()
                try:
                    result = await fn(*args, **kwargs)
                except Exception:
                    calls_counter.labels(status="error").inc()
                    raise
                else:
                    calls_counter.labels(status="success").inc()
                    return result
                finally:
                    duration_histogram.observe(time.monotonic() - start)

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _check_prometheus():
                    return fn(*args, **kwargs)

                start = time.monotonic()
                try:
                    result = fn(*args, **kwargs)
                except Exception:
                    calls_counter.labels(status="error").inc()
                    raise
                else:
                    calls_counter.labels(status="success").inc()
                    return result
                finally:
                    duration_histogram.observe(time.monotonic() - start)

            return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator
