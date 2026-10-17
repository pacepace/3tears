"""The closed context bundle: its schema, the derivations that fill it, and the assembler composing them.

A host never imports from here — :mod:`threetears.evals.analysis` is the public root — and code inside the
package imports the module that owns a name directly. The modules, in dependency order (each imports only
those above it):

- ``schema`` — :class:`~threetears.evals.analysis.bundle.schema.AnalysisContextBundle` and every value object
  it carries; ``caps`` — how many entries each growing list may carry.
- ``config`` — lever and configuration resolution; ``observations`` — which arm and cell each result is;
  ``measures`` — the reflective measure walk; ``insights`` — the insight ledger as the bundle reads it.
- ``design`` — the realized design, its arms and cohorts; ``confounds`` — the confound scan;
  ``mechanisms`` — mechanism checks and served-model readings.
- ``divergence`` — the divergence lens; ``coverage`` — the coverage map and factor aliasing;
  ``telemetry`` — the descriptive telemetry rollup.
- ``cell_reads`` — per-cell populations and judged values; ``bars`` — bar adjudication;
  ``cell_notes`` — the cells a reader must be told about; ``comparisons`` — multiple comparisons and
  guardrails; ``judges`` — the judge-change reading; ``time_axis`` — the time axis; ``surface`` — the
  decision surface.
- ``assemble`` — :func:`~threetears.evals.analysis.bundle.assemble.assemble_context_bundle`, composing them all.
"""
