"""The toy host's :class:`~threetears.evals.kernel.host.EvalHost` — the one value it hands the engine.

This is what adopting the engine looks like for a product: build the profile, wrap a document store
in the engine's storage, name the three services that have no safe default, and pass the result to
every entrypoint. Nothing is installed and nothing is global, so a second host built the same way in
the same process is simply a second value.

The toy host has no tracing of its own by default, runs storage calls on the loop's default
executor (nothing else in a test process queues there), and has no operation registry, so it names
the engine's bare cell timeout. A caller that wants spans passes its own sink.
"""

from __future__ import annotations

from threetears.evals.kernel import EvalStorage, withhold_failure_detail
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.kernel.host import CompletionClients, EvalHost, HostProfile, default_cell_timeout
from threetears.evals.schema import TraceSink
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile


def toyhost_host(
    *,
    profile: HostProfile | None = None,
    storage: EvalStorage | None = None,
    trace_sink: TraceSink | None = None,
    clients: CompletionClients | None = None,
) -> EvalHost:
    """The toy host as the engine receives it.

    Args:
        profile: The vocabulary. ``None`` is :func:`toyhost_profile`; a caller passes its own to
            hold one registration to a different value (an apparatus map emptied, say), or to keep
            the world's read handles it built.
        storage: Where documents live. ``None`` is a fresh in-memory store, empty and scoped.
        trace_sink: The host's tracing, or ``None`` for none — a decision rather than a default.
        clients: The completion-client factory, for a drive that judges or analyses through the
            engine. The standard matrix calls no model.

    Returns:
        The host.
    """
    return EvalHost(
        profile=profile if profile is not None else toyhost_profile(),
        storage=storage if storage is not None else EvalStorage(InMemoryDocumentStore()),
        # The toy host's calls cannot be refused for an account: it has none.
        failure_describer=withhold_failure_detail,
        trace_sink=trace_sink,
        # Nothing else in a test process queues on the loop's default executor.
        blocking_executor=None,
        # The toy host has no operation registry, so the engine's bare asyncio budget is the right one.
        cell_timeout=default_cell_timeout,
        clients=clients,
    )


__all__ = ["toyhost_host"]
