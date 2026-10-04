"""Structural gate: every production construction site wires the port it is handed.

Some of the eval engine's ports have a value that is correct for an anonymous host and wrong for
one with tracing, an operation registry or a shared executor, and every such default is
**silent**: ``EvalJobManager.job_timeout_factory`` defaults to a bare ``asyncio.timeout``, which
legitimately means "this host has no operation registry". It raises nothing, logs nothing and
reddens no test — an unwired site just quietly measures less than the operator believes it does.

The host's own wiring of that kind — its trace sink, cell timeout, blocking executor and failure
describer — carries no default at all on :class:`~threetears.evals.contracts.host.EvalHost`, which
is stronger than any walk: every construction names it. What remains walked here is the
constructors that still default.

That is the shape to refuse — a
net that fires silently — so it is held by a gate rather than by review attention. The gate is
an AST walk over every construction site outside ``tests/``, on the precedent that a run would
prove less at more cost: driving a real timeout means burning a real budget, and what needs
holding is one keyword at one site.

**Why this is its own module.** Both gates lived in ``test_extraction_import_boundary.py``,
which asks where the eval/host seam is — a different question, answered by walking imports
rather than calls. A canary in a file whose subject does not cover it is a canary
nobody finds when they change the thing it guards, and the two here share
:func:`_unwired_construction_sites`, so splitting them across the modules that own each *port*
would have put a second copy of that walk in the repo. The population they share is the
subject, and it is this file's name.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest


def _source_root() -> Path:
    """The package's ``src`` directory: the tree whose construction sites these gates police."""
    return Path(__file__).resolve().parents[1] / "src"


#: The trees a wiring gate walks — every tree in the package that holds non-test CODE.
#:
#: Named rather than inlined because two gates share it and a third will: a walk that quietly
#: covers less than its docstring claims is the failure mode this file keeps rediscovering, and
#: two copies of a glob drift into two populations with nothing comparing them. A host's own
#: construction sites are the host's to police; within this package, the sites are the engine's
#: own, which forward what the host injected.
_WIRING_TREES: tuple[str, ...] = ("threetears/**/*.py",)


def _unwired_construction_sites(
    symbol: str, keyword: str, exempt: dict[str, str], *, root: Path | None = None
) -> list[str]:
    """Every non-test construction of ``symbol`` that passes no ``keyword``.

    Shared by the two wiring gates below, which ask the same question of the same population
    about different constructors. They were two copies of this expression and the copy carried a
    hole: matching only ``ast.Name`` sees ``RunnerOptions(...)`` and not
    ``runner.RunnerOptions(...)``, and a module-qualified call is ordinary Python —
    ``from threetears.evals.run import runner`` then ``runner.RunnerOptions(...)``. The identical
    package-form blind spot had already been found and closed in Directions 3 and 7 of
    ``test_extraction_import_boundary.py``, where these gates used to live; a gate written after
    those fixes reproducing it is the argument for one expression rather than three.

    **Matches the last dotted segment**, so a bare name, a module-qualified call and a
    fully-qualified one all count, and **resolves import aliases**, so ``from … import X as Y``
    followed by ``Y(...)`` counts too. What it still cannot see is a construction through a
    value — a factory, a registry lookup, ``getattr`` — which no AST walk reaches and which the
    gates' messages do not claim to cover.

    Args:
        symbol: The constructor's name as declared, e.g. ``"EvalJobManager"``.
        keyword: The keyword argument a wired site must pass.
        exempt: Repo-relative file paths that legitimately pass none, keyed to the reason.
        root: The tree to walk. Defaults to the repository; the fabrication test below passes
            its own, which is the only way to watch these gates go red without writing an
            unwired construction site into the repo they police.

    Returns:
        ``"<path>:<lineno>"`` for each unwired, unexempt site, sorted.
    """
    repo_root = root or _source_root()
    sites: list[str] = []
    for pattern in _WIRING_TREES:
        for path in sorted(repo_root.glob(pattern)):
            rel = str(path.relative_to(repo_root))
            if rel in exempt:
                continue
            source = path.read_text(encoding="utf-8")
            # Cheap reject on the DECLARED name, which an aliasing file still contains: the
            # alias is introduced by an import line that spells the original.
            if symbol not in source:
                continue
            tree = ast.parse(source, filename=rel)
            names = {symbol} | {
                alias.asname
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
                if alias.name == symbol and alias.asname
            }
            sites.extend(
                f"{rel}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and ast.unparse(node.func).rsplit(".", 1)[-1] in names
                and not any(kw.arg == keyword for kw in node.keywords)
            )
    return sorted(sites)


#: The host's wiring that has a value right for a bare host and silently wrong for one with tracing,
#: a shared executor, an operation registry or provider error types — so it has no default anywhere.
_HOST_WIRING_WITHOUT_DEFAULTS: tuple[str, ...] = (
    "trace_sink",
    "blocking_executor",
    "cell_timeout",
    "failure_describer",
)


@pytest.mark.parametrize("name", _HOST_WIRING_WITHOUT_DEFAULTS)
def test_every_host_names_the_wiring_whose_default_would_be_silent(name):
    """A default that means "nobody was watching" is indistinguishable from a forgotten keyword.

    The four used to be ``RunnerOptions`` fields, defaulted, and held by an AST walk over every
    construction site outside ``tests/``. They are the host's now, carried on the one
    :class:`~threetears.evals.contracts.host.EvalHost` each entrypoint is handed, and the walk is
    replaced by something it could only approximate: the field has no default, so EVERY
    construction — a host's, a probe's, a test's — names it, and one that does not fails on
    construction rather than measuring less than its operator believes. ``None`` is still a legal
    answer for the sink and the executor, given out loud.

    Each one's silent default, for the record: a missing ``trace_sink`` records an empty trace and
    three ``None`` latency components; a bare ``cell_timeout`` runs every cell outside the host's
    operation registry, losing its budget, breach telemetry and inner-budget translation; a default
    ``blocking_executor`` queues every result write beside whatever else the loop's default executor
    serves (in a web host: the liveness probes); and a describer that cannot tell an account refusal
    keeps launching cells into a key that refuses every one of them.
    """
    from threetears.evals.contracts.host import EvalHost

    declared = {field.name: field for field in dataclasses.fields(EvalHost)}

    assert name in declared, f"EvalHost no longer carries {name!r}; this gate's list is stale"
    field = declared[name]
    assert field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING, (
        f"EvalHost.{name} has a default. A host that forgets it then runs on a value that is right for a bare "
        "host and silently wrong for this one — name it at every construction instead."
    )


def test_every_non_test_eval_job_manager_site_wires_a_timeout_factory():
    """The engine's default timeout is correct for a bare host and is not what a host with its own timeouts should run under.

    ``EvalJobManager``'s ``job_timeout_factory`` defaults to ``default_job_timeout`` —
    ``asyncio.timeout`` and nothing else — which is right for a host that has no timeout
    machinery and wrong for this one. An unwired production site still bounds its jobs, still
    fails a breached one the same way, and still logs; what it loses is silent and only visible
    to an operator: the job stops appearing in the ``eval_job`` operation budget, its breach no
    longer carries the layer's ERROR telemetry, and a nested inner operation has nothing to
    attribute the deadline to. Nothing raises, so nothing reddens.

    That is the same failure shape as an unwired ``trace_sink`` above, and it gets the same
    answer through the same walk: an AST assertion over every construction site outside
    ``tests/`` (:func:`_unwired_construction_sites`, where the population and its one blind spot
    are stated). A run would prove less at more cost — driving a real timeout means burning a
    real budget, and what needs holding is one keyword at one site.
    """
    #: Construction sites that legitimately take the engine default, with why. Empty today; an
    #: entry is a claim that this host's jobs are better off outside its own operation registry,
    #: which is an operability decision rather than an oversight.
    exempt: dict[str, str] = {}

    unwired = _unwired_construction_sites("EvalJobManager", "job_timeout_factory", exempt)

    assert not unwired, (
        f"EvalJobManager built with no job_timeout_factory at: {unwired}. Such a site runs its jobs "
        "under the engine's bare asyncio default instead of this host's eval_job operation budget, "
        "losing the registry, the ERROR-level breach telemetry and inner-op attribution — with no "
        "error and no log to say so. Pass job_timeout_factory=<the host's timeout factory>, or add the file "
        "to the exemption above with the reason its jobs belong outside the registry."
    )


@pytest.mark.parametrize("symbol", ["EvalJobManager", "EvalService"])
def test_every_non_test_site_names_where_its_blocking_storage_calls_run(symbol):
    """The engine's executor default is the loop's default executor — correct for a bare host, wrong in web.

    Both take ``blocking_executor`` and default it to ``None`` — ``EvalService`` as well, because a
    host's service forwards its own value onto the host it builds, so a service built without one
    would leave every result write falling back. Each defaults it to ``None``, which ``run_blocking`` reads
    as the loop's default executor. In the web process that is the executor the ``/healthz`` and
    ``/readyz`` probes run on, so every status write, progress tick, run save and result write of
    every eval run would queue beside them — exactly the traffic the dedicated pool exists to keep
    away. Nothing raises and nothing logs; the probes just get slow under eval load. Same answer as
    the timeout port above: name it at every site, and a site that deliberately takes the default
    says so with an explicit ``None``.
    """
    #: Sites that legitimately take no keyword, with why. Empty: an explicit ``blocking_executor=None``
    #: is how a site with nothing to protect records that, and it satisfies the gate.
    exempt: dict[str, str] = {}

    unwired = _unwired_construction_sites(symbol, "blocking_executor", exempt)

    assert not unwired, (
        f"{symbol} built with no blocking_executor at: {unwired}. Its storage calls would run on the loop's "
        "default executor, beside whatever else that executor serves (in web: the liveness probes). Pass "
        "blocking_executor=get_eval_io_pool(), or blocking_executor=None with a comment saying why nothing "
        "shares this process's default executor."
    )


@pytest.mark.parametrize("symbol", ["EvalService", "JudgeService"])
def test_every_non_test_site_names_how_a_failed_call_reads(symbol):
    """A describer that cannot tell an account refusal means an unwired site never stops.

    ``withhold_failure_detail`` is right for a host with no error types and wrong for this one: it
    answers ``account_refused=False`` for every failure, so a judge or simulated user refused for an
    exhausted account excludes its cell and the run launches the next one into the same key.
    ``JudgeService`` (like ``EvalHost``) takes no default at all, so a site that names none fails
    on construction and this gate is its second line; ``EvalService`` still defaults it, and is
    here because a host's service forwards its own onto the judge and the host it builds. Nothing raises and nothing logs for an unwired service. A site that deliberately reads
    failures the default way says so with an explicit keyword.
    """
    #: Sites that take no keyword, with why. Empty: naming the default is how a site whose calls
    #: cannot be refused for an account records that, and it satisfies the gate.
    exempt: dict[str, str] = {}

    unwired = _unwired_construction_sites(symbol, "failure_describer", exempt)

    assert not unwired, (
        f"{symbol} built with no failure_describer at: {unwired}. A call refused for the calling account "
        "then reads as any other failure and the run keeps launching cells. Pass the host's describer "
        "(describe_provider_failure, or the service's own), or name withhold_failure_detail with a "
        "comment saying why nothing that site drives can be refused for the account."
    )


def test_an_unwired_site_reddens_the_gate_and_a_wired_one_does_not(tmp_path):
    """The gates above, fabricated red — the only evidence they catch what they claim.

    Both gates are green today, and a green structural gate is evidence about the rule it
    runs rather than about the rule it is named after. The rule is only visible when
    something breaks it, and nothing may break it inside a repo these walks police, so the
    unwired site is built in a tree of its own.

    Every construction form the walk claims to reach is fabricated here, because each is a
    hole a narrower implementation would leave: the bare name, the module-qualified call, and
    the aliased import. So is the wired site, which is the half that makes the gate usable —
    a walk that flagged a correctly wired construction would be switched off within a day.
    """
    (tmp_path / "threetears").mkdir()
    (tmp_path / "threetears" / "bare.py").write_text("x = RunnerOptions(k_runs=1)\n", encoding="utf-8")
    (tmp_path / "threetears" / "qualified.py").write_text("x = runner.RunnerOptions(k_runs=1)\n", encoding="utf-8")
    (tmp_path / "threetears" / "aliased.py").write_text(
        "from threetears.evals.run.runner import RunnerOptions as Opts\n\nx = Opts(k_runs=1)\n", encoding="utf-8"
    )
    (tmp_path / "threetears" / "wired.py").write_text(
        "x = RunnerOptions(trace_sink=OtelTraceSink())\n", encoding="utf-8"
    )

    unwired = _unwired_construction_sites("RunnerOptions", "trace_sink", {}, root=tmp_path)

    assert [site.split(":")[0] for site in unwired] == [
        "threetears/aliased.py",
        "threetears/bare.py",
        "threetears/qualified.py",
    ]


def test_an_exempted_file_is_the_only_way_past_the_gate(tmp_path):
    """The exemption is a per-file argument, not a switch, and it is checked as one.

    An entry says "the cells this site drives need no trace", which is a measurement decision
    somebody has to write down. What must not work is exempting a file and having the walk
    keep finding it, or exempting nothing and having the walk go quiet — both of which a
    register consulted in the wrong place would produce.
    """
    (tmp_path / "threetears").mkdir()
    (tmp_path / "threetears" / "probe.py").write_text("x = EvalJobManager()\n", encoding="utf-8")

    assert _unwired_construction_sites("EvalJobManager", "job_timeout_factory", {}, root=tmp_path)
    assert (
        _unwired_construction_sites(
            "EvalJobManager", "job_timeout_factory", {"threetears/probe.py": "reason"}, root=tmp_path
        )
        == []
    )


def test_the_walk_reaches_the_engines_own_construction_sites():
    """Non-vacuity: the default root is the package's source, and the launch's sites are in it.

    Asking for a keyword no site passes turns every construction the walk reaches into a finding,
    so an empty answer here means the walk read the wrong tree and every gate above is green for
    nothing.
    """
    reached = _unwired_construction_sites("RunnerOptions", "no_such_keyword", {})

    assert any(site.startswith("threetears/evals/run/launch.py:") for site in reached), reached
