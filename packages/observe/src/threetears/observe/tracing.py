"""@traced decorator -- zero-cost without OpenTelemetry.

Creates OpenTelemetry spans around sync and async functions with automatic
context correlation, argument recording, and result capture.  When the
``opentelemetry`` package is not installed, the decorator is a pure
passthrough with no overhead beyond a single bool check per call.
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, TypeVar, overload

__all__ = ["set_span_attribute", "traced"]

F = TypeVar("F", bound=Callable[..., Any])

_SENSITIVE_PARAMS = frozenset(
    {
        "password",
        "token",
        "secret",
        "key",
        "api_key",
        "encryption_key",
        "jwt",
        "auth",
        "password_hash",
    }
)

_MAX_ATTR_LENGTH = 256


# ---------------------------------------------------------------------------
# OTel availability check (cached after first probe)
# ---------------------------------------------------------------------------

_otel_available: bool | None = None


def _check_otel() -> bool:
    """Check if OpenTelemetry API is importable (cached after first check)."""
    global _otel_available  # noqa: PLW0603
    if _otel_available is None:
        try:
            import opentelemetry.trace  # noqa: F401

            _otel_available = True
        except ImportError:
            _otel_available = False
    return _otel_available


# ---------------------------------------------------------------------------
# Span attribute helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _get_param_names(fn: Callable[..., Any]) -> tuple[str, ...]:
    """Return ordered parameter names for *fn* (cached)."""
    try:
        sig = inspect.signature(fn)
        return tuple(sig.parameters.keys())
    except ValueError, TypeError:
        return ()


def _set_safe_attr(span: Any, attr_key: str, param_name: str, value: Any) -> None:
    """Set a single span attribute if the value is safe to record."""
    if param_name in _SENSITIVE_PARAMS:
        return
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > _MAX_ATTR_LENGTH:
            value = value[:_MAX_ATTR_LENGTH]
        span.set_attribute(attr_key, value)
    elif hasattr(value, "hex"):  # UUID
        span.set_attribute(attr_key, str(value))


def _record_safe_args(
    span: Any,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Record non-sensitive function arguments as span attributes."""
    param_names = _get_param_names(fn)

    for idx, value in enumerate(args):
        name = param_names[idx] if idx < len(param_names) else f"arg{idx}"
        if name in ("self", "cls"):
            continue
        _set_safe_attr(span, f"arg.{name}", name, value)

    for name, value in kwargs.items():
        _set_safe_attr(span, f"arg.{name}", name, value)


def _record_safe_result(span: Any, result: Any) -> None:
    """Record type and (for collections) length of the return value."""
    span.set_attribute("result.type", type(result).__name__)
    if hasattr(result, "__len__"):
        try:
            span.set_attribute("result.count", len(result))
        except TypeError:
            pass


def _inject_context_attrs(span: Any) -> None:
    """Inject all log context values as span attributes with ``ctx.`` prefix."""
    from threetears.observe.logging import get_context

    for key, value in get_context().items():
        span.set_attribute(f"ctx.{key}", value)


# ---------------------------------------------------------------------------
# public: attach arbitrary attributes to the currently active span
# ---------------------------------------------------------------------------


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the currently active OpenTelemetry span, if any.

    Safe to call from inside (or outside) a :func:`traced` function: a
    pure no-op when OpenTelemetry is not installed or there is no
    currently recording span, matching :func:`traced`'s own zero-cost-
    when-absent contract. Lets application code attach result-derived or
    business-specific attributes to the current span -- values only known
    after a function's body has run, which a decorator alone can never
    see -- without importing ``opentelemetry`` directly.

    Same value-safety rules as :func:`traced`'s own argument recording:
    sensitive-looking keys (``password``, ``token``, ``secret``, etc.)
    are silently dropped, and long strings are truncated.

    :param key: span attribute key
    :ptype key: str
    :param value: attribute value; strings/ints/floats/bools are recorded
        as-is (strings truncated past 256 chars), UUID-like objects
        (anything with a ``.hex`` attribute) are recorded as ``str()``,
        anything else is silently dropped
    :ptype value: Any
    :return: none
    :rtype: None
    """
    if not _check_otel():
        return

    from opentelemetry import trace

    span = trace.get_current_span()
    if not span.is_recording():
        return

    _set_safe_attr(span, key, key, value)


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------


@overload
def traced(func: F) -> F: ...


@overload
def traced(
    *,
    name: str | None = None,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[F], F]: ...


def traced(
    func: F | None = None,
    *,
    name: str | None = None,
    record_args: bool = False,
    record_result: bool = False,
) -> F | Callable[[F], F]:
    """Decorator that creates an OpenTelemetry span around a function.

    Supports both bare ``@traced`` and parameterised ``@traced(name=...)``
    usage.  Works for sync and async functions.

    When OpenTelemetry is not installed, the decorator is a pure passthrough
    with no overhead beyond a single bool check per call.

    :param name: explicit span name; defaults to ``module.qualname``.
    :param record_args: record non-sensitive arguments as span attributes.
    :param record_result: record result type/count as span attributes.
    """

    def decorator(fn: F) -> F:
        span_name = name or f"{fn.__module__}.{fn.__qualname__}"

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _check_otel():
                    return await fn(*args, **kwargs)

                from opentelemetry import trace
                from opentelemetry.trace import StatusCode

                tracer = trace.get_tracer(fn.__module__)
                with tracer.start_as_current_span(span_name) as span:
                    _inject_context_attrs(span)

                    if record_args:
                        _record_safe_args(span, fn, args, kwargs)

                    start = time.monotonic()
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(StatusCode.ERROR, str(exc))
                        raise
                    else:
                        span.set_status(StatusCode.OK)
                        if record_result:
                            _record_safe_result(span, result)
                        return result
                    finally:
                        span.set_attribute(
                            "duration_ms",
                            round((time.monotonic() - start) * 1000, 2),
                        )

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _check_otel():
                    return fn(*args, **kwargs)

                from opentelemetry import trace
                from opentelemetry.trace import StatusCode

                tracer = trace.get_tracer(fn.__module__)
                with tracer.start_as_current_span(span_name) as span:
                    _inject_context_attrs(span)

                    if record_args:
                        _record_safe_args(span, fn, args, kwargs)

                    start = time.monotonic()
                    try:
                        result = fn(*args, **kwargs)
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(StatusCode.ERROR, str(exc))
                        raise
                    else:
                        span.set_status(StatusCode.OK)
                        if record_result:
                            _record_safe_result(span, result)
                        return result
                    finally:
                        span.set_attribute(
                            "duration_ms",
                            round((time.monotonic() - start) * 1000, 2),
                        )

            return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator
