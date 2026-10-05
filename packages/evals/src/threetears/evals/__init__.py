"""The eval engine: templates, cases, runs and results; the trial loop, its judge and simulated user; and the analysis over what runs produced.

The engine is four packages: :mod:`~threetears.evals.contracts` (the stored shapes, and the host contract under
:mod:`~threetears.evals.contracts.host`), :mod:`~threetears.evals.run` (launching, executing, judging
and storing runs), :mod:`~threetears.evals.analysis` (campaigns, bundles, memos and reports, with
charts under :mod:`~threetears.evals.analysis.viz`) and :mod:`~threetears.evals.gen` (generating
cases), plus three that sit beside the engine rather than inside it: :mod:`~threetears.evals.storage`
(the in-memory reference store), :mod:`~threetears.evals.testing` (the store conformance kit) and
:mod:`~threetears.evals.quick` (``run_eval`` and the command line). Above the engine sit the surfaces an
agent drives it through: :mod:`~threetears.evals.ops` (typed operations, and one job contract for long
work), :mod:`~threetears.evals.actions` (the action catalogue every transport mounts) and
:mod:`~threetears.evals.transports` (each transport behind its own extra — ``fastmcp``). Runs are launched
on demand; there is no scheduled-run surface.

**Import only from a public root**, and only the names its ``__all__`` declares. The roots are
:data:`PUBLIC_ROOTS`, which a consumer can read to check its own imports rather than keep a copy;
``tests/test_package_matrix.py`` holds the tuple equal to the roots it enforces.

**There is no installed host.** A product adopts the engine by building one
:class:`~threetears.evals.contracts.host.EvalHost` — its vocabulary, its storage and its services —
and handing it to every entrypoint that reads any of them; a function that reads only the vocabulary
takes the :class:`~threetears.evals.contracts.host.HostProfile` the host carries. Nothing in the
engine reads a module-level, context-variable or default host, and importing this package installs
and configures nothing, so two hosts in one process — even in one event loop — are two values that
never meet. ``tests/test_two_hosts_one_process.py`` and ``tests/test_no_process_global_state.py``
hold that.
"""

#: The public roots, as absolute module names. A consumer imports from these and from no module
#: below them, and only the names each root's ``__all__`` declares. ``contracts.host`` and
#: ``analysis.viz`` are roots of their own inside a package: the contract a host implements, and
#: the one root whose render needs ``vl_convert``.
PUBLIC_ROOTS: tuple[str, ...] = (
    "threetears.evals.contracts",
    "threetears.evals.contracts.host",
    "threetears.evals.run",
    "threetears.evals.analysis",
    "threetears.evals.analysis.viz",
    "threetears.evals.gen",
    "threetears.evals.storage",
    "threetears.evals.testing",
    "threetears.evals.quick",
    "threetears.evals.ops",
    "threetears.evals.actions",
    "threetears.evals.transports.fastmcp",
)

__all__ = ["PUBLIC_ROOTS"]
