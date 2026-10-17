"""The one value a product builds to adopt the eval engine: its vocabulary, its store and its services.

An :class:`EvalHost` is handed to every engine entrypoint that reads what a host supplies. There is
no installed or ambient host — nothing in the engine reads a module-level profile, a context
variable or a default — so two hosts in one process, even in one event loop, are two values and
never meet. That is the property the toy host and a second host prove side by side
(``tests/test_two_hosts_one_process.py``).

**What a function takes is what it reads.** A function that reads only the vocabulary takes the
:class:`~threetears.evals.kernel.host.profile.HostProfile` (an identity derivation, a declaration
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

    from threetears.evals.kernel import EvalStorage, withhold_failure_detail
    from threetears.evals.kernel.host import EvalHost, default_cell_timeout

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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from threetears.evals.kernel.errors import RefusedInput

if TYPE_CHECKING:
    # Type-only: the storage module imports the stored models, which import this package for the
    # subject type, so a runtime import here would be a cycle. Nothing below needs them at runtime.
    from concurrent.futures import Executor

    from threetears.evals.kernel.host.profile import HostProfile
    from threetears.evals.kernel.host.timeouts import CellTimeoutFactory
    from threetears.evals.schema.traces import TraceSink
    from threetears.evals.schema.completion import BoundCompletionClient, ProviderFailureDescriber
    from threetears.evals.schema.models import EvalTemplate
    from threetears.evals.kernel.storage import EvalStorage

#: The apparatus roles the engine asks a host for a completion client in. ``judge`` scores a cell,
#: ``simulator`` plays the other side of a conversation, ``analysis`` writes a campaign's memo, and
#: ``variation`` writes the values of a template's ``llm`` variation axes when a launch generates its
#: cases (``n_variations``) — before any run starts, so outside every run's cost cap. A role of its
#: own because it is a different job from the simulator's: a host resolves or refuses each role on
#: its own, and a kind with no simulated user still generates. ``proposer`` drafts a rubric for operator
#: review (:func:`~threetears.evals.gen.propose_draft`), outside any run, with its own output cap
#: (``Ceilings().proposer_max_tokens``, :class:`~threetears.evals.run.Ceilings`) — a role of its own so a host does not borrow
#: another role's model, cap or budget for it.
CompletionRole = Literal["judge", "simulator", "analysis", "variation", "proposer"]


class CompletionClients(Protocol):
    """The host's completion-client factory: one client per role, model and temperature.

    The engine never builds a client and never names a provider. It asks the host for one bound to
    a model, uses it, and releases it (:meth:`~threetears.evals.schema.completion.CompletionClient.aclose`)
    — so a client is built per unit of work, never cached here.
    """

    def __call__(
        self, role: CompletionRole, model: str | None, *, temperature: float | None = None
    ) -> BoundCompletionClient:
        """Build a client for ``role`` bound to ``model``.

        :param role: Which apparatus role the client serves.
        :ptype role: CompletionRole
        :param model: The model to bind, or ``None`` for the role's default as the host resolves it.
            The built client names what it resolved to (``model_name``).
        :ptype model: str | None
        :param temperature: The sampling temperature, or ``None`` for the provider default. The judge always
            names one (:data:`~threetears.evals.schema.models.DEFAULT_JUDGE_TEMPERATURE` unless a config
            says otherwise); a client whose model refuses a temperature sends none and says so on each
            completion (``CompletionResult.temperature`` is ``None``), because the score records what was
            sent, never what was asked.
        :ptype temperature: float | None
        :return: The client, owned by the caller.
        :rtype: BoundCompletionClient
        """
        ...  # pragma: no cover — protocol


class IgnoresTemplateFields(Protocol):
    """What admitting a template reads of a kind's launch registry entry: the template fields the kind never acts on.

    Structural, so a host's registry entries (``threetears.evals.run.LaunchableKind``) are handed over as they are.
    """

    @property
    def ignored_template_fields(self) -> frozenset[str]:
        """The template fields the kind never acts on; a template stating one is refused, naming it."""
        ...  # pragma: no cover — protocol


@dataclass(frozen=True, kw_only=True)
class TemplateChecks:
    """The three questions about a template that only the host can answer, and the kinds it launches, read wherever a
    template is admitted.

    ``create_template``, ``update_template``, the boot seed (``seed_eval_definitions``), a definition reset
    (``reset_definitions``) and the ``definitions_reset`` action all read them from the one host they are handed
    (:attr:`EvalHost.template_checks`), so no path admits a template under checks another path does not ask. Each check
    raises :class:`~threetears.evals.kernel.errors.ValidationFailedError` to refuse and returns ``None`` to admit. None
    has a default: a host with nothing to check passes a function that checks nothing, which is a decision its code
    states.

    :ivar require_known_tools_allowed: Whether each name in a template's ``tools_allowed`` is a tool type the host's
        catalog has; handed the list (``None`` when the template states none).
    :ivar refuse_undeclared_world_seed: Whether a template's ``world_seed`` names only paths the host's world can seed:
        the host's walk over its own namespaces.
    :ivar refuse_undeliverable_template: Whether the template declares only apparatus its candidate kind can honour: the
        host's kind capability table. The template fields a kind never acts on are not its to refuse: every path
        refuses those itself, from ``kinds``.
    :ivar kinds: The host's launch registry, by kind name: the mapping its ``LaunchHost`` is built with. Every path that
        admits a template refuses one stating a field its kind's entry ignores
        (``threetears.evals.run.authoring.refuse_ignored_template_fields``), as the launch dispatch does; a kind
        the registry does not carry is refused nothing here, and its launch is refused by name.
    """

    require_known_tools_allowed: Callable[[Sequence[str] | None], None]
    refuse_undeclared_world_seed: Callable[[EvalTemplate], None]
    refuse_undeliverable_template: Callable[[EvalTemplate], None]
    kinds: Mapping[str, IgnoresTemplateFields]


@dataclass(frozen=True, kw_only=True)
class EvalHost:
    """Everything the engine reads of one consuming product, as one frozen value.

    Built once by the product and handed to each entrypoint; see the module docstring for the
    example and for why three fields carry no default. Keyword-only, so every field is named where
    the host is built.

    A frozen dataclass rather than a Pydantic model, for the reason
    :class:`~threetears.evals.kernel.host.profile.HostProfile` is one: it holds live objects and
    callables, is a runtime registration, and is never stored or sent anywhere.

    :ivar profile: The product's vocabulary — what it sweeps, what it measures, its bars, its world
        and its presentation style. Every identity, gate and lens reads it from here.
    :ivar storage: Where every eval document is read and written, over the product's own
        :class:`~threetears.evals.schema.store_port.DocumentStore`.
    :ivar failure_describer: How a raised provider call reads — the only thing that can say a call
        was refused for the calling account, which stops a run rather than excluding its cells.
        :func:`~threetears.evals.kernel.provider.withhold_failure_detail` is the honest one
        for a product with no error types of its own.
    :ivar trace_sink: The product's tracing, or ``None`` when nobody watches its cells — which records
        no spans and no span-derived latency, and is a decision rather than a fault.
    :ivar blocking_executor: Where synchronous storage calls run off the event loop, or ``None`` for
        the loop's default executor.
    :ivar cell_timeout: What bounds each cell's wall clock;
        :func:`~threetears.evals.kernel.host.timeouts.default_cell_timeout` for a product with
        no timeout layer of its own.
    :ivar clients: Builds the completion clients the engine's own roles call — the judge on a
        re-judge, the analysis generator — and the client a kind's launcher asks for in the
        ``variation`` role to write generated cases. ``None`` for a product whose engine work calls no
        model, and an entrypoint that needs one refuses rather than guessing.
    :ivar template_checks: The host's checks on a template (:class:`TemplateChecks`), which every path that admits a
        template reads from here. ``None`` for a product that authors no template through the engine (a quick
        ``Eval`` stores its own), and an entrypoint that admits one refuses rather than admitting it unchecked.
    """

    profile: HostProfile
    storage: EvalStorage
    failure_describer: ProviderFailureDescriber
    trace_sink: TraceSink | None
    blocking_executor: Executor | None
    cell_timeout: CellTimeoutFactory
    clients: CompletionClients | None = None
    template_checks: TemplateChecks | None = None

    def completion_clients(self, purpose: str) -> CompletionClients:
        """The host's client factory, or a refusal naming what needed one.

        :param purpose: What is asking, for the refusal — e.g. ``"a re-judge"``.
        :ptype purpose: str
        :return: The factory.
        :rtype: CompletionClients
        :raises RefusedInput: This host supplies no completion clients — a :class:`ValueError`, as every caller
            documents, that a surface reports as a refusal rather than as an unanticipated error.
        """
        if self.clients is None:
            raise RefusedInput(
                f"host {self.profile.host_id!r} supplies no completion clients, and {purpose} calls a model",
                remedy="build the host with clients=<its client factory>",
            )
        return self.clients

    def checks_for(self, purpose: str) -> TemplateChecks:
        """The host's template checks, or a refusal naming what needed them.

        :param purpose: What is admitting a template, for the refusal — e.g. ``"create_template"``.
        :ptype purpose: str
        :return: The checks.
        :rtype: TemplateChecks
        :raises RefusedInput: This host supplies no template checks — a :class:`ValueError`, as every caller documents,
            that a surface reports as a refusal rather than as an unanticipated error.
        """
        if self.template_checks is None:
            raise RefusedInput(
                f"host {self.profile.host_id!r} supplies no template checks, and {purpose} admits a template",
                remedy="build the host with template_checks=<its template checks>",
            )
        return self.template_checks


__all__ = ["CompletionClients", "CompletionRole", "EvalHost", "IgnoresTemplateFields", "TemplateChecks"]
