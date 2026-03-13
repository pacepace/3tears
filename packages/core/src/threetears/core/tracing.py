"""@traced decorator — zero-cost without OpenTelemetry."""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable, TypeVar, overload

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


def _truncate(value: str) -> str:
    if len(value) > _MAX_ATTR_LENGTH:
        return value[:_MAX_ATTR_LENGTH] + "..."
    return value


def _safe_attrs_from_args(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, str]:
    """Build span attributes from function arguments, filtering sensitive params."""
    import inspect

    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    attrs: dict[str, str] = {}
    for name, value in bound.arguments.items():
        if name in _SENSITIVE_PARAMS:
            attrs[f"arg.{name}"] = "[REDACTED]"
        else:
            attrs[f"arg.{name}"] = _truncate(repr(value))
    return attrs


def _result_attrs(result: Any) -> dict[str, str]:
    """Build span attributes from a return value."""
    attrs: dict[str, str] = {"result.type": type(result).__name__}
    if hasattr(result, "__len__"):
        try:
            attrs["result.count"] = str(len(result))
        except TypeError:
            pass
    return attrs


def _inject_context_attrs(span: Any) -> None:
    """Inject correlation context from threetears.core.logging into span."""
    from threetears.core.logging import _correlation_id, _conversation_id, _session_id

    cid = _correlation_id.get()
    sid = _session_id.get()
    conv = _conversation_id.get()
    if cid:
        span.set_attribute("ctx.correlation_id", cid)
    if sid:
        span.set_attribute("ctx.session_id", sid)
    if conv:
        span.set_attribute("ctx.conversation_id", conv)


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

    If OpenTelemetry is not installed, the decorator is a pure passthrough
    with no overhead beyond a single bool check per call.
    """

    def decorator(fn: F) -> F:
        span_name = name or f"{fn.__module__}.{fn.__qualname__}"

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _check_otel():
                    return await fn(*args, **kwargs)

                from opentelemetry import trace
                from opentelemetry.trace import StatusCode

                tracer = trace.get_tracer("threetears")
                with tracer.start_as_current_span(span_name) as span:
                    _inject_context_attrs(span)

                    if record_args:
                        for k, v in _safe_attrs_from_args(fn, args, kwargs).items():
                            span.set_attribute(k, v)

                    start = time.monotonic()
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception as exc:
                        span.set_status(StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise
                    finally:
                        span.set_attribute("duration_ms", round((time.monotonic() - start) * 1000, 2))

                    if record_result:
                        for k, v in _result_attrs(result).items():
                            span.set_attribute(k, v)

                    return result

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _check_otel():
                    return fn(*args, **kwargs)

                from opentelemetry import trace
                from opentelemetry.trace import StatusCode

                tracer = trace.get_tracer("threetears")
                with tracer.start_as_current_span(span_name) as span:
                    _inject_context_attrs(span)

                    if record_args:
                        for k, v in _safe_attrs_from_args(fn, args, kwargs).items():
                            span.set_attribute(k, v)

                    start = time.monotonic()
                    try:
                        result = fn(*args, **kwargs)
                    except Exception as exc:
                        span.set_status(StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise
                    finally:
                        span.set_attribute("duration_ms", round((time.monotonic() - start) * 1000, 2))

                    if record_result:
                        for k, v in _result_attrs(result).items():
                            span.set_attribute(k, v)

                    return result

            return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator  # type: ignore[return-value]
