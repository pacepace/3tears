"""The one value a product builds to adopt the eval engine: its vocabulary, its store and its services.

An :class:`EvalHost` is handed to every engine entrypoint that reads what a host supplies. There is
no installed or ambient host — nothing in the engine reads a module-level profile, a context
variable or a default — so two hosts in one process, even in one event loop, are two values and
never meet. That is the property the toy host and a second host prove side by side
(``tests/test_two_hosts_one_process.py``).

**What a function takes is what it reads.** A function that reads only the vocabulary takes the
:class:`~threetears.evals.contracts.host.profile.HostProfile` (an identity derivation, a declaration
gate, a bundle lens); one that reads only documents takes its storage; one that reads both, or a
service such as the trace sink or a completion client, takes the host. The narrow forms are not a
second route around the host: they are what a host's fields are, passed on by the entrypoint that
holds it.

**Launching runs needs more than this**, and the run package says what:
:class:`~threetears.evals.run.launch.LaunchHost` composes an ``EvalHost`` with the launch registry of
kinds, the host's launch settings and the job manager it builds over this host's storage. Those are
run-package types, and this package imports nothing of that one, so they cannot be fields here; a
product that launches builds a ``LaunchHost`` around its ``EvalHost`` and hands that ``EvalHost`` to
the analysis side.

Example — the smallest host, one that reads and analyses runs with no tracing and no timeout layer
of its own::

    from threetears.evals.contracts import EvalStorage, withhold_failure_detail
    from threetears.evals.contracts.host import EvalHost, default_cell_timeout

    host = EvalHost(
        profile=my_profile(),  # the product's levers, measures, bars and world
        storage=EvalStorage(MyDocumentStore()),
        failure_describer=withhold_failure_detail,
        trace_sink=None,  # nobody is watching this host's cells
        blocking_executor=None,  # nothing else queues on the loop's default executor
        cell_timeout=default_cell_timeout,  # no operation registry of its own
    )

The three fields a host might expect defaults for — ``trace_sink``, ``blocking_executor`` and
``cell_timeout`` — have none on purpose. Each has a value that is right for a bare host and silently
wrong for one with tracing, a shared executor or an operation registry: nothing raises and nothing
logs, a run just measures less than its operator believes. So each is named where the host is built,
and a host that takes the bare answer says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    # Type-only: the storage module imports the stored models, which import this package for the
    # subject type, so a runtime import here would be a cycle. Nothing below needs them at runtime.
    from concurrent.futures import Executor

    from threetears.evals.contracts.host.profile import HostProfile
    from threetears.evals.contracts.host.timeouts import CellTimeoutFactory
    from threetears.evals.contracts.host.traces import TraceSink
    from threetears.evals.contracts.provider import BoundCompletionClient, ProviderFailureDescriber
    from threetears.evals.contracts.storage import EvalStorage

#: The apparatus roles the engine asks a host for a completion client in. ``judge`` scores a cell,
#: ``simulator`` plays the other side of a conversation, ``analysis`` writes a campaign's memo.
CompletionRole = Literal["judge", "simulator", "analysis"]


class CompletionClients(Protocol):
    """The host's completion-client factory: one client per role, model and temperature.

    The engine never builds a client and never names a provider. It asks the host for one bound to
    a model, uses it, and releases it (:meth:`~threetears.evals.contracts.provider.CompletionClient.aclose`)
    — so a client is built per unit of work, never cached here.
    """

    def __call__(
        self, role: CompletionRole, model: str | None, *, temperature: float | None = None
    ) -> BoundCompletionClient:
        """Build a client for ``role`` bound to ``model``.

        Args:
            role: Which apparatus role the client serves.
            model: The model to bind, or ``None`` for the role's default as the host resolves it.
                The built client names what it resolved to (``model_name``).
            temperature: The sampling temperature, or ``None`` for the provider default.

        Returns:
            The client, owned by the caller.
        """
        ...  # pragma: no cover — protocol


@dataclass(frozen=True, kw_only=True)
class EvalHost:
    """Everything the engine reads of one consuming product, as one frozen value.

    Built once by the product and handed to each entrypoint; see the module docstring for the
    example and for why three fields carry no default. Keyword-only, so every field is named where
    the host is built.

    A frozen dataclass rather than a Pydantic model, for the reason
    :class:`~threetears.evals.contracts.host.profile.HostProfile` is one: it holds live objects and
    callables, is a runtime registration, and is never stored or sent anywhere.

    Attributes:
        profile: The product's vocabulary — what it sweeps, what it measures, its bars, its world
            and its presentation style. Every identity, gate and lens reads it from here.
        storage: Where every eval document is read and written, over the product's own
            :class:`~threetears.evals.contracts.store_port.DocumentStore`.
        failure_describer: How a raised provider call reads — the only thing that can say a call
            was refused for the calling account, which stops a run rather than excluding its cells.
            :func:`~threetears.evals.contracts.provider.withhold_failure_detail` is the honest one
            for a product with no error types of its own.
        trace_sink: The product's tracing, or ``None`` when nobody watches its cells — which records
            no spans and no span-derived latency, and is a decision rather than a fault.
        blocking_executor: Where synchronous storage calls run off the event loop, or ``None`` for
            the loop's default executor.
        cell_timeout: What bounds each cell's wall clock;
            :func:`~threetears.evals.contracts.host.timeouts.default_cell_timeout` for a product with
            no timeout layer of its own.
        clients: Builds the completion clients the engine's own roles call — the judge on a
            re-judge, the analysis generator. ``None`` for a product whose engine work calls no
            model, and an entrypoint that needs one refuses rather than guessing.
    """

    profile: HostProfile
    storage: EvalStorage
    failure_describer: ProviderFailureDescriber
    trace_sink: TraceSink | None
    blocking_executor: Executor | None
    cell_timeout: CellTimeoutFactory
    clients: CompletionClients | None = None

    def completion_clients(self, purpose: str) -> CompletionClients:
        """The host's client factory, or a refusal naming what needed one.

        Args:
            purpose: What is asking, for the refusal — e.g. ``"a re-judge"``.

        Returns:
            The factory.

        Raises:
            ValueError: This host supplies no completion clients.
        """
        if self.clients is None:
            raise ValueError(
                f"host {self.profile.host_id!r} supplies no completion clients, and {purpose} calls a model; "
                "build the host with clients=<its client factory>"
            )
        return self.clients


__all__ = ["CompletionClients", "CompletionRole", "EvalHost"]
