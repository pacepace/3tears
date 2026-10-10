"""The eval engine: templates, cases, runs and results; the trial loop, its judge and simulated user; and the analysis over what runs produced.

The engine is five packages: :mod:`~threetears.evals.schema` (the stored shapes and the ports a host
implements), :mod:`~threetears.evals.kernel` (the behaviour every other package runs on, and the host
contract under :mod:`~threetears.evals.kernel.host`), :mod:`~threetears.evals.run` (launching, executing, judging
and storing runs), :mod:`~threetears.evals.analysis` (campaigns, bundles, memos and reports, with
charts under :mod:`~threetears.evals.analysis.viz`) and :mod:`~threetears.evals.gen` (generating
cases), plus three that sit beside the engine rather than inside it: :mod:`~threetears.evals.storage`
(the in-memory reference store), :mod:`~threetears.evals.testing` (the store conformance kit) and
:mod:`~threetears.evals.quick` (``run_eval`` and the command line). Above the engine sit the surfaces an
agent drives it through: :mod:`~threetears.evals.ops` (typed operations, and one job contract for long
work), :mod:`~threetears.evals.actions` (the action catalogue every transport mounts) and
:mod:`~threetears.evals.transports` (each transport behind its own extra — ``fastmcp``). One more is an
optional adapter: :mod:`~threetears.evals.vega`, the Vega-Lite chart renderer (install
``3tears-evals[vega]``), which reads the core's chart intent and which nothing in the core imports. Runs
are launched on demand; there is no scheduled-run surface.

**Import only from a public root**, and only the names its ``__all__`` declares. The roots are
:data:`PUBLIC_ROOTS`, which a consumer can read to check its own imports rather than keep a copy;
``tests/test_package_matrix.py`` holds the tuple equal to the roots it enforces.

**There is no installed host.** A product adopts the engine by building one
:class:`~threetears.evals.kernel.host.EvalHost` — its vocabulary, its storage and its services —
and handing it to every entrypoint that reads any of them; a function that reads only the vocabulary
takes the :class:`~threetears.evals.kernel.host.HostProfile` the host carries. Nothing in the
engine reads a module-level, context-variable or default host, and importing this package installs
and configures nothing, so two hosts in one process — even in one event loop — are two values that
never meet. ``tests/test_two_hosts_one_process.py`` and ``tests/test_no_process_global_state.py``
hold that.
"""

#: The public roots, as absolute module names. A consumer imports from these and from no module
#: below them, and only the names each root's ``__all__`` declares. ``kernel.host`` and
#: ``analysis.viz`` are roots of their own inside a package: the contract a host implements, and
#: the chart intent with the seam a renderer sits behind. Two roots need an extra: ``vega``, the
#: optional Vega-Lite renderer, whose rasteriser needs ``[vega]``, and ``transports.fastmcp``, which
#: needs ``[fastmcp]``.
PUBLIC_ROOTS: tuple[str, ...] = (
    "threetears.evals.schema",
    "threetears.evals.kernel",
    "threetears.evals.kernel.host",
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
    "threetears.evals.vega",
)

__all__ = ["PUBLIC_ROOTS"]
