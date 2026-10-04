"""The eval engine: templates, cases, runs and results; the trial loop, its judge and simulated user; and the analysis over what runs produced.

Four packages, each a public root: :mod:`~threetears.evals.contracts` (the stored shapes, and the
host contract under :mod:`~threetears.evals.contracts.host`), :mod:`~threetears.evals.run` (launching,
executing, judging and storing runs), :mod:`~threetears.evals.analysis` (campaigns, bundles, memos and
reports) and :mod:`~threetears.evals.gen` (generating cases). Runs are launched on demand; there is no
scheduled-run surface.

**There is no installed host.** A product adopts the engine by building one
:class:`~threetears.evals.contracts.host.EvalHost` — its vocabulary, its storage and its services —
and handing it to every entrypoint that reads any of them; a function that reads only the vocabulary
takes the :class:`~threetears.evals.contracts.host.HostProfile` the host carries. Nothing in the
engine reads a module-level, context-variable or default host, and importing this package installs
and configures nothing, so two hosts in one process — even in one event loop — are two values that
never meet. ``tests/test_two_hosts_one_process.py`` and ``tests/test_no_process_global_state.py``
hold that.
"""
