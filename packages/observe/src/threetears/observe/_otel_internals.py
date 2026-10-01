"""the one owner of every OpenTelemetry private attribute this package touches.

OpenTelemetry's API (``opentelemetry-api`` >= 1.39) guards ``trace.set_tracer_provider`` with a
module-level set-once flag, ``trace._TRACER_PROVIDER_SET_ONCE``, and offers no public way to clear
it. A second ``set_tracer_provider`` is then silently ignored, so tracing keeps reaching whichever
provider was installed first. :func:`threetears.observe.init_telemetry` must be able to install a
provider again after :func:`threetears.observe.shutdown_telemetry` (a host that re-initializes, and
every test fixture that does), and shutdown must hand the slot back for that.

So this module is the only place in ``threetears.observe`` that reads or writes an OpenTelemetry
name with a leading underscore. ``setup.py`` calls :func:`allow_tracer_provider_reset`, never the
attribute, so an OpenTelemetry release that renames the guard is reported here, by name, instead of
as an ``AttributeError`` from wherever it was reached for.

The surface: ``opentelemetry.trace._TRACER_PROVIDER_SET_ONCE`` (an ``opentelemetry.util._once.Once``)
and its ``_done`` flag.
"""

from __future__ import annotations

from opentelemetry import trace

__all__ = ["allow_tracer_provider_reset"]


def allow_tracer_provider_reset() -> bool:
    """clear OpenTelemetry's set-once guard so the next ``set_tracer_provider`` takes effect.

    :return: ``True`` when the guard was cleared; ``False`` when this OpenTelemetry release no
        longer carries it in the shape above, in which case a later ``set_tracer_provider`` may be
        ignored and the caller must say so
    :rtype: bool
    """
    cleared = True
    try:
        trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined, unused-ignore]
    except AttributeError:
        # NOSILENT: the False return is the report; the caller logs what it costs at its own site
        cleared = False
    return cleared
