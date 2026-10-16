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

from collections.abc import Mapping
from dataclasses import replace

from threetears.evals.run import LaunchableKind, LaunchHost
from threetears.evals.kernel import EvalStorage, withhold_failure_detail
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.kernel.host import (
    CompletionClients,
    EvalHost,
    HostProfile,
    TemplateChecks,
    default_cell_timeout,
)
from threetears.evals.schema import TraceSink
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile


def _admit(_checked: object) -> None:
    """A toy host check: it has nothing to refuse by."""


#: The toy host's template checks: it has no tool catalog, seed walk or kind capability table of its own, so each
#: admits, which is a decision stated here rather than a default. A host that only reads launches no kind, so its
#: registry is empty; ``toyhost_launch_host`` hands these checks the registry it launches with.
TOYHOST_TEMPLATE_CHECKS = TemplateChecks(
    require_known_tools_allowed=_admit,
    refuse_undeclared_world_seed=_admit,
    refuse_undeliverable_template=_admit,
    kinds={},
)


def with_registry(eval_host: EvalHost, kinds: Mapping[str, LaunchableKind]) -> EvalHost:
    """``eval_host`` with ``kinds`` as the launch registry, which a host states once, on its template checks.

    A host with no template checks gets the admit-everything ones (:data:`TOYHOST_TEMPLATE_CHECKS`).

    :param eval_host: The host.
    :ptype eval_host: EvalHost
    :param kinds: The launch registry, by kind name.
    :ptype kinds: Mapping[str, LaunchableKind]
    :return: The host, over ``kinds``.
    :rtype: EvalHost
    """
    checks = eval_host.template_checks if eval_host.template_checks is not None else TOYHOST_TEMPLATE_CHECKS
    return replace(eval_host, template_checks=replace(checks, kinds=kinds))


def with_kinds(launch_host: LaunchHost, kinds: Mapping[str, LaunchableKind]) -> LaunchHost:
    """``launch_host`` launching ``kinds`` instead: the registry lives on its template checks, so that is what changes.

    :param launch_host: The host.
    :ptype launch_host: LaunchHost
    :param kinds: The launch registry, by kind name.
    :ptype kinds: Mapping[str, LaunchableKind]
    :return: The host, with everything else as it was and ``kinds`` as its registry.
    :rtype: LaunchHost
    """
    return replace(launch_host, eval_host=with_registry(launch_host.eval_host, kinds))


def toyhost_host(
    *,
    profile: HostProfile | None = None,
    storage: EvalStorage | None = None,
    trace_sink: TraceSink | None = None,
    clients: CompletionClients | None = None,
    template_checks: TemplateChecks | None = TOYHOST_TEMPLATE_CHECKS,
) -> EvalHost:
    """The toy host as the engine receives it.

    :param profile: The vocabulary. ``None`` is :func:`toyhost_profile`; a caller passes its own to
        hold one registration to a different value (an apparatus map emptied, say), or to keep
        the world's read handles it built.
    :ptype profile: HostProfile | None
    :param storage: Where documents live. ``None`` is a fresh in-memory store, empty and scoped.
    :ptype storage: EvalStorage | None
    :param trace_sink: The host's tracing, or ``None`` for none — a decision rather than a default.
    :ptype trace_sink: TraceSink | None
    :param clients: The completion-client factory, for a drive that judges or analyses through the
        engine. The standard matrix calls no model.
    :ptype clients: CompletionClients | None
    :param template_checks: The checks a template is admitted under; :data:`TOYHOST_TEMPLATE_CHECKS` (each admits)
        unless a test holds a template to a check of its own, or ``None`` for a host supplying none.
    :ptype template_checks: TemplateChecks | None
    :return: The host.
    :rtype: EvalHost
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
        template_checks=template_checks,
    )


__all__ = ["TOYHOST_TEMPLATE_CHECKS", "toyhost_host", "with_kinds", "with_registry"]
