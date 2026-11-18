"""The statistics certificate: every large error-rate claim the evals reading makes under the null, the simulation that
counts it, and the digest of the code that decides it.

A claim such as "a comparison continued to five reads separates at most α" needs tens of thousands of simulated
newcomers to bound. Running them through ``Eval.run`` on every build took hours, so they are certified instead:

- **Restated claims** count their newcomers on the reading restated over the bet alone (``null_newcomers.py``: the
  package's own statistics functions over the same seeded answers, no ``Eval.run``), which reads tens of thousands of
  them in a minute or two. The restatement is held to the product by the ``restated-as-product`` claims: a few hundred
  of the same seeded newcomers run through ``Eval.run`` and every reading's p and decision must be the restatement's.
- **Product claims** run through the product, because what they claim depends on time: which cells a stop finds in
  flight, and so which a cut draws afresh. A drained stop draws nothing afresh, so its claims are exact rather than
  counted: every drained engine comparison reads as the uninterrupted one on its first draws
  (``engine_take_up.drained_as_uninterrupted``), and every drained quick-path watcher separates only where the bet over
  its whole committed path does (``null_newcomers.drained_held``), whose rate the ``horizon`` claims bound.

Running the certification (:data:`COMMAND`, ``packages/evals/scripts/certify_stats.py``) writes
:data:`CERTIFICATE`: each claim, its count, its Clopper-Pearson 95% bounds, its seeds, and the digest of the files that
decide it (:func:`closure`: every repository file the certification imports, derived from the code). ``test_stats_certificate.py`` fails, naming each stale claim and this command, when any of
those files has changed since; and fails on any claim whose count does not clear its target.

The restated claims run on this machine, side by side in processes; the product claims run on the testament pool as one
request (``test_certified_simulations.py``, the ``simulation`` suite), each shard recording its counts as metrics the
command reads back.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from threetears.evals.analysis.stats import EVIDENCE_STEPS, SIGNIFICANCE_ALPHA, exact_rate_interval

from packages.evals.tests import engine_take_up as engine
from packages.evals.tests import null_newcomers as quick

#: The one command that certifies, run from the repository's root.
COMMAND = "uv run python packages/evals/scripts/certify_stats.py"

#: The repository's root, which every source path is relative to.
ROOT = Path(__file__).resolve().parents[3]

#: Where the certificate is written, and read by ``test_stats_certificate.py``.
CERTIFICATE = Path(__file__).resolve().parent / "fixtures" / "stats_certificate.json"

#: Where a certification run's key is written while it runs, so the pool's shards name their metrics by it.
RUN_KEY = Path(__file__).resolve().parent / "fixtures" / "stats_certification_run.txt"

#: Where the closure starts: the modules the certification itself runs (the restated and product newcomers, the
#: engine's take-up, the claims and their shards, the command). Every module they import, at any depth and wherever
#: the import sits (inside a function, under ``TYPE_CHECKING``), that resolves to a file in this repository, is in the
#: digest: the package's reading, its runner and its engine, and every other workspace module they reach.
ENTRY_POINTS = (
    "packages/evals/tests/null_newcomers.py",
    "packages/evals/tests/engine_take_up.py",
    "packages/evals/tests/stats_certificate.py",
    "packages/evals/tests/test_certified_simulations.py",
    "packages/evals/scripts/certify_stats.py",
)


def _source_roots() -> list[Path]:
    """The directories a top-level module is found in: every workspace package's ``src``, and the repository's root
    (for ``packages.*``, the test support modules)."""
    return [*sorted(ROOT.glob("packages/*/src")), *sorted(ROOT.glob("packages/agent/*/src")), ROOT]


def _module_file(name: str, roots: Sequence[Path]) -> Path | None:
    """The repository file a dotted module name resolves to (a module, or a package's ``__init__.py``), or None for
    one outside it (the standard library, a third-party package) or a namespace directory with no file."""
    parts = name.split(".")
    for root in roots:
        base = root.joinpath(*parts)
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None


def _module_name(path: Path, roots: Sequence[Path]) -> str:
    for root in roots:
        if path.is_relative_to(root):
            relative = path.relative_to(root).with_suffix("")
            parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
            return ".".join(parts)
    raise ValueError(f"{path} is under no source root")


#: An import statement, at any indentation (inside a function or under ``TYPE_CHECKING`` too): ``from x import y`` (its
#: names in parentheses over several lines, or to the end of the line) or ``import x``. A line in a docstring that reads
#: as one only widens the closure, which costs a re-certification at most.
_FROM_IMPORT = re.compile(r"^[ \t]*from[ \t]+(\.*[\w.]*)[ \t]+import[ \t]+(\([^)]*\)|[^\n]*)", re.MULTILINE)
_IMPORT = re.compile(r"^[ \t]*import[ \t]+([^\n]+)", re.MULTILINE)
#: A module named in a string (a lazy export's table, ``import_module("threetears...")``): imported on first use.
_NAMED = re.compile(r"""["'](threetears(?:\.\w+)+)["']""")


def _names(listed: str) -> list[str]:
    """The names an import lists: parentheses, continuations, comments and ``as`` aliases left out."""
    names = []
    for line in listed.strip("()").splitlines():
        for part in line.split("#", 1)[0].replace("\\", "").split(","):
            word = part.strip().split(" as ")[0].strip()
            if word and word != "*" and all(piece.isidentifier() for piece in word.split(".")):
                names.append(word)
    return names


def _imported(path: Path, name: str) -> set[str]:
    """Every module name ``path`` (module ``name``) imports, each with its parents, and each ``from x import y`` as
    both ``x`` and ``x.y`` (``y`` may be a module). Read off the text, as :data:`_FROM_IMPORT` and :data:`_IMPORT`
    match, which is fast enough to run on every unit-suite build."""
    text = path.read_text()
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found: set[str] = set()
    for source, listed in _FROM_IMPORT.findall(text):
        level = len(source) - len(source.lstrip("."))
        if level:
            base = package.split(".")
            base = base[: len(base) - (level - 1)] if level > 1 else base
            module = ".".join([*base, *([source.lstrip(".")] if source.lstrip(".") else [])])
        else:
            module = source
        found.add(module)
        found.update(f"{module}.{imported}" for imported in _names(listed))
    for listed in _IMPORT.findall(text):
        found.update(_names(listed))
    found.update(_NAMED.findall(text))
    with_parents = set()
    for module in found:
        parts = module.split(".")
        with_parents.update(".".join(parts[: index + 1]) for index in range(len(parts)))
    return with_parents


def closure() -> list[str]:
    """Every repository file the certification imports, from :data:`ENTRY_POINTS`, as paths from the repository's
    root, in path order: what decides every claim. Computed from the code as it stands, at certify time and at check
    time, so a module the reading comes to import joins the digest without anyone listing it."""
    roots = _source_roots()
    # Every module of the workspace's ``src`` trees, indexed once by name; ``packages.*`` is looked for under the root.
    indexed: dict[str, Path] = {}
    for root in roots[:-1]:
        for found in root.rglob("*.py"):
            indexed.setdefault(_module_name(found, [root]), found)

    def module_file(module: str) -> Path | None:
        if module.startswith("packages."):
            return _module_file(module, [ROOT])
        return indexed.get(module)

    resolved: dict[str, Path | None] = {}
    seen: set[Path] = set()
    pending = [(ROOT / entry, _module_name(ROOT / entry, roots)) for entry in ENTRY_POINTS]
    while pending:
        path, name = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for module in _imported(path, name):
            if module not in resolved:
                resolved[module] = module_file(module)
            found = resolved[module]
            if found is not None and found not in seen:
                pending.append((found, module))
    return sorted(str(path.relative_to(ROOT)) for path in seen)


def file_digest(path: str) -> str:
    """One source file's sha256, read from the repository's root."""
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def sources_digest(files: Mapping[str, str]) -> str:
    """The digest of ``files`` (each path's own digest): a sha256 over each path and its digest, in path order."""
    hashed = hashlib.sha256()
    for path in sorted(files):
        hashed.update(f"{path}\0{files[path]}\n".encode())
    return hashed.hexdigest()


def current_files() -> dict[str, str]:
    """Every file of the closure with its digest, as the code stands."""
    return {path: file_digest(path) for path in closure()}


# --- the shards ------------------------------------------------------------------------------------------------------

Where = Literal["local", "pool"]

#: The counts a shard returns, each a metric the pool's shard records: ``n`` newcomers or comparisons read; ``hits`` the
#: claim's event (separated, missed, crossed); ``better`` and ``worse`` the strong evidence each way; ``held`` how many
#: held what an exact claim asserts of each; ``low`` and ``high`` the intervals whose lower end passed the true
#: difference, and whose upper end fell short of it.
FIELDS = ("n", "hits", "better", "worse", "held", "low", "high")


@dataclass(frozen=True)
class Shard:
    """One unit of a claim group's simulation: its id (a pytest id), the group it counts toward, where it runs, the job
    that runs it and that job's arguments."""

    id: str
    group: str
    where: Where
    job: str
    args: Mapping[str, Any] = field(default_factory=dict)


def _blocks(total: int, size: int) -> list[range]:
    return [range(start, min(total, start + size)) for start in range(0, total, size)]


def _shards() -> list[Shard]:
    shards: list[Shard] = []
    nulls: tuple[quick.Null, ...] = ("binary", "onramp", "skewed")
    for null in nulls:
        # Single reads at twelve cases: 20 shards of 200 (tags once-<null>-<shard>-<index>).
        for block in _blocks(20, 5):
            shards.append(
                Shard(
                    f"small-once-{null}-{block.start}",
                    f"small-once-{null}",
                    "local",
                    "small-once",
                    {"null": null, "shards": list(block), "per": 200},
                )
            )
        # Continued to five reads: 40 shards of 150.
        for block in _blocks(40, 5):
            shards.append(
                Shard(
                    f"small-continued-{null}-{block.start}",
                    f"small-continued-{null}",
                    "local",
                    "small-continued",
                    {"null": null, "shards": list(block), "per": 150},
                )
            )
    # The on-ramp's held-out plans under its null: read at twenty-four cases, then continued once with eight more: 40
    # shards of 150 (tags planned-onramp-<shard>-<index>).
    for block in _blocks(40, 5):
        shards.append(
            Shard(
                f"small-planned-onramp-{block.start}",
                "small-planned-onramp",
                "local",
                "small-planned",
                {"null": "onramp", "shards": list(block), "per": 150},
            )
        )
    for asks in ("gate", "late"):
        # A gate, and a question pointed at the leader after the read: continued from eight cases to five reads (tags
        # <asks>-<index>), and read once at sixteen cases (tags <asks>-once-<index>); 6,000 newcomers each.
        for plan in ("", "-once"):
            for block in _blocks(6000, 750):
                shards.append(
                    Shard(
                        f"small-{asks}{plan}-{block.start}",
                        f"small-{asks}{plan}",
                        "local",
                        "small-asks",
                        {"asks": asks, "plan": plan, "start": block.start, "count": len(block)},
                    )
                )
    for block in _blocks(20, 5):
        shards.append(
            Shard(
                f"small-cover-{block.start}", "small-cover", "local", "small-cover", {"shards": list(block), "per": 200}
            )
        )
    for block in _blocks(200, 10):
        shards.append(
            Shard(
                f"rereadable-continued-{block.start}",
                "rereadable-continued",
                "local",
                "rereadable",
                {"continued": True, "shards": list(block), "per": 50},
            )
        )
    for block in _blocks(40, 5):
        shards.append(
            Shard(
                f"rereadable-rerun-{block.start}",
                "rereadable-rerun",
                "local",
                "rereadable",
                {"continued": False, "shards": list(block), "per": 50},
            )
        )
    for k, launches in ((3, 5), (1, 2)):
        horizon = quick.watched_horizon(launches)
        for block in _blocks(10_000, 1000):
            shards.append(
                Shard(
                    f"horizon-k{k}-n{horizon}-{block.start}",
                    f"horizon-k{k}",
                    "local",
                    "horizon",
                    {"k": k, "horizon": horizon, "start": block.start, "count": len(block)},
                )
            )
    # The engine's bounded test on pass/fail pairs under the null, at a fixed count and over three looks: 10,000 a cell.
    for cases, plus, looks in engine.NULL_READS:
        for block in _blocks(10_000, 2_500):
            shards.append(
                Shard(
                    f"engine-null-{looks}-{cases}-{plus}-{block.start}",
                    f"engine-null-{looks}-{cases}-{plus}",
                    "local",
                    "engine-null",
                    {"cases": cases, "plus": plus, "looks": looks, "start": block.start, "count": len(block)},
                )
            )
    # The engine's unpaired bounded test on pass/fail arms under the null, at a fixed count and over three looks: 4,000 a
    # cell.
    for control, contrast, rate, looks in engine.NULL_UNPAIRED:
        for block in _blocks(4_000, 1_000):
            shards.append(
                Shard(
                    f"engine-unpaired-{looks}-{control}-{contrast}-{rate}-{block.start}",
                    f"engine-unpaired-{looks}-{control}-{contrast}-{rate}",
                    "local",
                    "engine-unpaired",
                    {
                        "cases": (control, contrast),
                        "rate": rate,
                        "looks": looks,
                        "start": block.start,
                        "count": len(block),
                    },
                )
            )
    # Through the product, on the pool: the restatement held to it, read for read, on the certified claims' own seeds.
    for null in nulls:
        shards.append(
            Shard(
                f"eq-small-once-{null}",
                "eq-small-once",
                "pool",
                "equivalence",
                {"flow": "small-once", "null": null, "tags": [f"once-{null}-0-{i}" for i in range(40)]},
            )
        )
        shards.append(
            Shard(
                f"eq-small-continued-{null}",
                "eq-small-continued",
                "pool",
                "equivalence",
                {"flow": "small-continued", "null": null, "tags": [f"continued-{null}-0-{i}" for i in range(30)]},
            )
        )
    shards.append(
        Shard(
            "eq-small-planned-onramp",
            "eq-small-planned",
            "pool",
            "equivalence",
            {"flow": "small-planned", "null": "onramp", "tags": [f"planned-onramp-0-{i}" for i in range(30)]},
        )
    )
    # A gate's page reads its margin on every read, about five seconds a continued newcomer: 15 to a shard.
    for asks, plan, count, size in (
        ("gate", "", 210, 15),
        ("gate", "-once", 120, 40),
        ("late", "", 120, 20),
        ("late", "-once", 120, 40),
    ):
        for part, block in enumerate(_blocks(count, size)):
            shards.append(
                Shard(
                    f"eq-small-{asks}{plan}-{part}",
                    f"eq-small-{asks}{plan}",
                    "pool",
                    "equivalence",
                    {"flow": f"small-{asks}{plan}", "null": "binary", "tags": [f"{asks}{plan}-{i}" for i in block]},
                )
            )
    shards.append(
        Shard(
            "eq-small-cover",
            "eq-small-cover",
            "pool",
            "equivalence",
            {"flow": "small-cover", "tags": [f"cover-0-{i}" for i in range(60)]},
        )
    )
    for part, block in enumerate(_blocks(60, 20)):
        shards.append(
            Shard(
                f"eq-rereadable-continued-{part}",
                "eq-rereadable-continued",
                "pool",
                "equivalence",
                {"flow": "rereadable-continued", "tags": [f"continued-{i // 50}-{i % 50}" for i in block]},
            )
        )
    for part, block in enumerate(_blocks(30, 15)):
        shards.append(
            Shard(
                f"eq-rereadable-rerun-{part}",
                "eq-rereadable-rerun",
                "pool",
                "equivalence",
                {"flow": "rereadable-rerun", "tags": [f"rerun-0-{i}" for i in block]},
            )
        )
    # Watchers who drain: each reading the bet over a prefix, no cell drawn twice, separated only where the path did.
    for latency, k, launches, stop, count, size in (
        ("alike-apart", 3, 5, "lead", 300, 25),
        ("alike-slow-wrong", 3, 5, "lead", 300, 25),
        ("b-slower-4x", 1, 2, "blind", 1000, 125),
        ("b-slower-20x", 1, 2, "blind", 1000, 125),
    ):
        tag = f"alike-{latency}" if stop == "lead" else f"drained-{latency}"
        for block in _blocks(count, size):
            shards.append(
                Shard(
                    f"quick-drained-{latency}-{block.start}",
                    f"quick-drained-{latency}",
                    "pool",
                    "quick-watched",
                    {
                        "tag": tag,
                        "latency": latency,
                        "stop": stop,
                        "k": k,
                        "launches": launches,
                        "cut_share": 0.0,
                        "start": block.start,
                        "count": len(block),
                    },
                )
            )
    # Watchers who cut a second time in three of ten stops: counted.
    for latency in ("b-slower-4x", "b-slower-20x"):
        for block in _blocks(1000, 125):
            shards.append(
                Shard(
                    f"quick-cut-{latency}-{block.start}",
                    f"quick-cut-{latency}",
                    "pool",
                    "quick-watched",
                    {
                        "tag": f"cut-{latency}",
                        "latency": latency,
                        "stop": "blind",
                        "k": 1,
                        "launches": 2,
                        "cut_share": 0.3,
                        "start": block.start,
                        "count": len(block),
                    },
                )
            )
    # The engine: drained comparisons held exactly; those whose interrupts cut, counted at the simulation's size.
    for reading, law, grader, stop in sorted(engine.SHOWN):
        if stop == "drain":
            # Every comparison of the simulation's full size, each held exactly: 50 to a shard.
            total = engine.FIXED_REPLICATES if reading == "fixed" else engine.LOOKS_REPLICATES
            for block in _blocks(total, 50):
                shards.append(
                    Shard(
                        f"engine-drain-{reading}-{law}-{grader}-{block.start}",
                        f"engine-drain-{reading}-{law}-{grader}",
                        "pool",
                        "engine-drained",
                        {"reading": reading, "law": law, "grader": grader, "start": block.start, "count": len(block)},
                    )
                )
            continue
        total = engine.FIXED_REPLICATES if reading == "fixed" else engine.LOOKS_REPLICATES
        size = 100 if reading == "fixed" else 50
        for block in _blocks(total, size):
            shards.append(
                Shard(
                    f"engine-{stop}-{reading}-{law}-{grader}-{block.start}",
                    f"engine-{stop}-{reading}-{law}-{grader}",
                    "pool",
                    "engine-counted",
                    {
                        "reading": reading,
                        "law": law,
                        "grader": grader,
                        "stop": stop,
                        "start": block.start,
                        "count": len(block),
                    },
                )
            )
    return shards


#: Every shard of every claim group.
SHARDS: tuple[Shard, ...] = tuple(_shards())


def _counts(n: int, hits: int = 0, **more: int) -> dict[str, int]:
    return {"n": n, "hits": hits, **more}


def run_local(shard: Shard) -> dict[str, int]:
    """Run one restated shard here: pure arithmetic over seeded answers, no ``Eval.run``."""
    args = shard.args
    if shard.job == "small-once":
        reads = [
            quick.small_once(args["null"], f"once-{args['null']}-{s}-{i}")
            for s in args["shards"]
            for i in range(args["per"])
        ]
        return _counts(len(reads), sum(read[0].decided for read in reads))
    if shard.job == "small-continued":
        came = [
            quick.came_to(quick.small_continued(args["null"], f"continued-{args['null']}-{s}-{i}"))
            for s in args["shards"]
            for i in range(args["per"])
        ]
        return _counts(
            len(came),
            sum(c.separated for c in came),
            better=sum(c.strong_better for c in came),
            worse=sum(c.strong_worse for c in came),
        )
    if shard.job == "small-planned":
        planned = {"start": quick.PLANNED_START, "reads": quick.PLANNED_READS, "add": quick.PLANNED_ADD}
        came = [
            quick.came_to(quick.small_continued(args["null"], f"planned-{args['null']}-{s}-{i}", **planned))
            for s in args["shards"]
            for i in range(args["per"])
        ]
        return _counts(len(came), sum(c.separated for c in came))
    if shard.job == "small-asks":
        plan = {"start": 16, "reads": 1} if args["plan"] == "-once" else {}
        reads = [
            quick.small_continued("binary", f"{args['asks']}{args['plan']}-{i}", args["asks"], **plan)
            for i in range(args["start"], args["start"] + args["count"])
        ]
        return _counts(len(reads), sum(read[-1].decided for read in reads))
    if shard.job == "small-cover":
        sides = [
            quick.missed_side(quick.small_covers(f"cover-{s}-{i}")) for s in args["shards"] for i in range(args["per"])
        ]
        return _counts(
            len(sides),
            sum(side is not None for side in sides),
            low=sum(side == "low" for side in sides),
            high=sum(side == "high" for side in sides),
        )
    if shard.job == "rereadable":
        kind = "continued" if args["continued"] else "rerun"
        reads = [
            quick.rereadable(f"{kind}-{s}-{i}", continued=args["continued"])
            for s in args["shards"]
            for i in range(args["per"])
        ]
        return _counts(len(reads), sum(read[-1].decided for read in reads))
    if shard.job == "horizon":
        crossed = [
            quick.horizon_crossed(f"horizon-k{args['k']}-{i}", args["k"], args["horizon"])
            for i in range(args["start"], args["start"] + args["count"])
        ]
        return _counts(len(crossed), sum(crossed))
    if shard.job == "engine-null":
        separated = [
            engine.null_reads_separate(
                f"engine-null-{args['looks']}-{args['cases']}-{args['plus']}-{i}",
                args["cases"],
                args["plus"],
                args["looks"],
            )
            for i in range(args["start"], args["start"] + args["count"])
        ]
        return _counts(len(separated), sum(separated))
    if shard.job == "engine-unpaired":
        control, contrast = args["cases"]
        separated = [
            engine.unpaired_null_separates(
                f"engine-unpaired-{args['looks']}-{control}-{contrast}-{args['rate']}-{i}",
                (control, contrast),
                args["rate"],
                args["looks"],
            )
            for i in range(args["start"], args["start"] + args["count"])
        ]
        return _counts(len(separated), sum(separated))
    raise ValueError(f"shard {shard.id} runs on the pool, not here")


async def run_pool(shard: Shard) -> dict[str, int]:
    """Run one product shard: through ``Eval.run`` or the engine's operator path. A shard holding an exact claim fails
    at the first newcomer or comparison that breaks it, saying where."""
    args = shard.args
    if shard.job == "equivalence":
        return await _equivalence(args["flow"], args.get("null"), args["tags"])
    if shard.job == "quick-watched":
        tally = _counts(0, 0, held=0, crossed=0)
        drained = args["cut_share"] == 0.0
        for index in range(args["start"], args["start"] + args["count"]):
            seen = await quick.watched(
                f"{args['tag']}-{index}",
                latency=args["latency"],
                stop=args["stop"],
                k=args["k"],
                cut_share=args["cut_share"],
                launches=args["launches"],
            )
            broke = quick.drained_held(seen) if drained else None
            assert broke is None, f"{shard.id} newcomer {index}: {broke}"
            tally["n"] += 1
            tally["hits"] += seen.product
            tally["held"] += drained
        return {key: value for key, value in tally.items() if key in FIELDS}
    if shard.job == "engine-drained":
        tally = _counts(0, 0, held=0)
        for index in range(args["start"], args["start"] + args["count"]):
            tag = engine.comparison_tag("engine", args["reading"], args["law"], args["grader"], "drain", index)
            outcome = await engine.one_comparison(tag, args["reading"], args["law"], args["grader"], "drain")
            broke = engine.drained_as_uninterrupted(outcome)
            assert broke is None, f"{shard.id} comparison {index}: {broke}"
            tally["n"] += 1
            tally["hits"] += outcome.separated
            tally["held"] += 1
        return tally
    if shard.job == "engine-counted":
        separated, _ = await engine.simulate(
            args["reading"], args["law"], args["grader"], args["stop"], args["count"], "engine", start=args["start"]
        )
        return _counts(args["count"], separated)
    raise ValueError(f"shard {shard.id} runs here, not on the pool")


async def _equivalence(flow: str, null: quick.Null | None, tags: Sequence[str]) -> dict[str, int]:
    """Run each tagged newcomer of ``flow`` through ``Eval.run`` and restated; fail at the first whose reads part."""
    held = decided = 0
    for tag in tags:
        if flow == "small-once":
            assert null is not None
            product, restated = await quick.small_once_product(null, tag), quick.small_once(null, tag)
        elif flow == "small-continued":
            assert null is not None
            product, restated = await quick.small_continued_product(null, tag), quick.small_continued(null, tag)
        elif flow == "small-planned":
            assert null is not None
            planned = {"start": quick.PLANNED_START, "reads": quick.PLANNED_READS, "add": quick.PLANNED_ADD}
            product = await quick.small_continued_product(null, tag, **planned)
            restated = quick.small_continued(null, tag, **planned)
        elif flow in ("small-gate", "small-late", "small-gate-once", "small-late-once"):
            asks = cast("quick.Asks", flow.removeprefix("small-").removesuffix("-once"))
            plan = {"start": 16, "reads": 1} if flow.endswith("-once") else {}
            product = await quick.small_continued_product("binary", tag, asks, **plan)
            restated = quick.small_continued("binary", tag, asks, **plan)
        elif flow == "small-cover":
            product, restated = await quick.small_covers_product(tag), quick.small_covers(tag)
        else:
            continued = flow == "rereadable-continued"
            product = await quick.rereadable_product(tag, continued=continued)
            restated = quick.rereadable(tag, continued=continued)
        parted = quick.same_reads(product, restated)
        assert parted is None, f"{flow} {tag}: {parted}"
        held += 1
        decided += product[-1].decided
    return _counts(len(tags), decided, held=held)


def run_shard(shard: Shard) -> dict[str, int]:
    """Run one shard in this process, wherever it belongs (the command's ``--local`` runs the pool's here too)."""
    if shard.where == "local":
        return run_local(shard)
    return asyncio.run(run_pool(shard))


# --- the claims ------------------------------------------------------------------------------------------------------

#: How a claim's count is held to its target:
#:
#: - ``at_most``: the Clopper-Pearson 95% upper bound on the rate is at most the target;
#: - ``above``: the 95% lower bound is above the target;
#: - ``share_below``: the share itself is below the target (a rate reported, failing only far past it);
#: - ``every``: every newcomer or comparison held the exact property (``held`` equals ``n``).
Test = Literal["at_most", "above", "share_below", "every"]


@dataclass(frozen=True)
class Claim:
    """One certified claim: its id, what it says, the shard group that counts it and the field it counts, how the
    count is held to the target, and its seeds. Every claim is decided by the whole closure (:func:`closure`)."""

    id: str
    says: str
    group: str
    field: str
    test: Test
    target: float
    seeds: str


def _claims() -> list[Claim]:
    alpha, strong = SIGNIFICANCE_ALPHA, 1.0 / EVIDENCE_STEPS[1]
    claims: list[Claim] = []
    for null in ("binary", "onramp", "skewed"):
        claims.append(
            Claim(
                f"small-n.once.{null}",
                f"Under the {null} null a question's primary read once at twelve cases separates at most α.",
                f"small-once-{null}",
                "hits",
                "at_most",
                alpha,
                f"once-{null}-<shard>-<index>, shards 0-19 of 200, restated",
            )
        )
        group = f"small-continued-{null}"
        seeds = f"continued-{null}-<shard>-<index>, shards 0-39 of 150, restated"
        claims.append(
            Claim(
                f"small-n.continued.{null}.separates",
                f"Under the {null} null a question's primary continued from eight cases to five reads, each add sized "
                "from the reading, separates at most α.",
                group,
                "hits",
                "at_most",
                alpha,
                seeds,
            )
        )
        for side, said in (("better", "for v2"), ("worse", "against v2")):
            claims.append(
                Claim(
                    f"small-n.continued.{null}.strong-{side}",
                    f"Under the {null} null, over those reads, the evidence {said} ever reaches 10 at most one time in "
                    "ten (Ville).",
                    group,
                    side,
                    "at_most",
                    strong,
                    seeds,
                )
            )
    claims.append(
        Claim(
            "small-n.planned.onramp.separates",
            "Under the onramp null a question's primary read at twenty-four cases and, where it does not separate, "
            "continued once with eight more (the on-ramp's held-out plans: apps as cases, three trials each), separates "
            "at most α.",
            "small-planned-onramp",
            "hits",
            "at_most",
            alpha,
            "planned-onramp-<shard>-<index>, shards 0-39 of 150, restated",
        )
    )
    for asks, said in (
        (
            "gate",
            "a gate (is v2 no worse within five points?), its bet leaning toward worse beside its equivalence test,",
        ),
        ("late", "a question set after the results and pointed at whichever arm leads, which leans no way,"),
    ):
        claims.append(
            Claim(
                f"small-n.{asks}.once",
                f"Under the binary null {said} read once at sixteen cases, separates at most α.",
                f"small-{asks}-once",
                "hits",
                "at_most",
                alpha,
                f"{asks}-once-<index>, 0-5999, restated",
            )
        )
        claims.append(
            Claim(
                f"small-n.{asks}.separates",
                f"Under the binary null {said} continued from eight cases to five reads, each add sized from the "
                "reading, separates at most α.",
                f"small-{asks}",
                "hits",
                "at_most",
                alpha,
                f"{asks}-<index>, 0-5999, restated",
            )
        )
    claims.append(
        Claim(
            "small-n.interval.misses",
            "The primary's interval (the even hedge's confidence sequence at 90%, read on the capital now), read once "
            "at sixteen cases at a true difference of 0.3, misses it at most the 10% its 90% states.",
            "small-cover",
            "hits",
            "at_most",
            1.0 - 0.90,
            "cover-<shard>-<index>, shards 0-19 of 200, restated",
        )
    )
    for side, said in (("low", "its lower end above the true difference"), ("high", "its upper end below it")):
        claims.append(
            Claim(
                f"small-n.interval.misses-{side}",
                f"Each end of that interval is a 95% one-sided bound: it misses with {said} at most α.",
                "small-cover",
                side,
                "at_most",
                alpha,
                "cover-<shard>-<index>, shards 0-19 of 200, restated",
            )
        )
    claims.append(
        Claim(
            "rereadable.continued",
            "With no question, a comparison of arms passing half the time continued from 20 cases to five runs, each add "
            "sized from the reading, separates at most α.",
            "rereadable-continued",
            "hits",
            "at_most",
            alpha,
            "continued-<shard>-<index>, shards 0-199 of 50, restated",
        )
    )
    claims.append(
        Claim(
            "rereadable.rerun",
            "Re-running every case on each run instead, read by the same bet, separates more often than α: it is not one "
            "bet and loses the bound.",
            "rereadable-rerun",
            "hits",
            "above",
            alpha,
            "rerun-<shard>-<index>, shards 0-39 of 50, restated",
        )
    )
    for flow, said in (
        ("small-once", "a question's single read at twelve cases, under each null"),
        ("small-continued", "a question's comparison continued to five reads, under each null"),
        ("small-planned", "the on-ramp's planned comparison, twenty-four cases then eight more, under its null"),
        ("small-gate", "a gate continued to five reads"),
        ("small-gate-once", "a gate read once at sixteen cases"),
        ("small-late", "a question set after each read, pointed at the leader"),
        ("small-late-once", "a question set after one read at sixteen cases, pointed at the leader"),
        ("small-cover", "a question's interval at a true difference of 0.3"),
        ("rereadable-continued", "a comparison continued with no question"),
        ("rereadable-rerun", "a comparison re-run whole with no question"),
    ):
        claims.append(
            Claim(
                f"restated-as-product.{flow}",
                f"Through Eval.run, {said} reads exactly as the restatement over the bet alone reads the same seeded "
                "answers: every reading's p and evidence to float rounding, its decision and its interval.",
                f"eq-{flow}",
                "held",
                "every",
                1.0,
                "the first newcomers of the restated claim's own seeds, through the product on the pool",
            )
        )
    for k, launches, who in (
        (3, 5, "arms alike in speed, k = 3, stopped when an arm leads, up to five runs"),
        (1, 2, "arms of different speed, k = 1, stopped blind, two runs"),
    ):
        horizon = quick.watched_horizon(launches)
        claims.append(
            Claim(
                f"watched.drained.horizon-k{k}",
                f"A drained watcher ({who}) separates at most α: the bet over any {horizon} pairs of the null, each case "
                f"the mean of {k} fair coins, ever reaches 1/α at most α, and each drained reading is a prefix of such a "
                "path (the watched.drained mechanism claims).",
                f"horizon-k{k}",
                "hits",
                "at_most",
                alpha,
                f"horizon-k{k}-<index>, 0-9999, restated",
            )
        )
    for latency in ("alike-apart", "alike-slow-wrong", "b-slower-4x", "b-slower-20x"):
        claims.append(
            Claim(
                f"watched.drained.{latency}",
                f"Watchers whose arms are {latency}, every stop drained, through Eval.run: every reading is the bet over "
                "the longest complete prefix of the committed order, no cell draws twice, and the product separates only "
                "where the bet over its whole committed path does: every newcomer of the simulation's full size held "
                "exactly.",
                f"quick-drained-{latency}",
                "held",
                "every",
                1.0,
                ("alike-" if latency.startswith("alike") else "drained-") + f"{latency}-<index>, through the product",
            )
        )
    for latency in ("b-slower-4x", "b-slower-20x"):
        claims.append(
            Claim(
                f"watched.cut.{latency}",
                f"Watchers whose arms are {latency}, three stops in ten cut by a second interrupt, separate below 2α: a "
                "continuation draws each cut cell afresh, which can lean the reading, and the print counts them.",
                f"quick-cut-{latency}",
                "hits",
                "share_below",
                2 * alpha,
                f"cut-{latency}-<index>, 0-999, through the product",
            )
        )
    for cases, plus, looks in engine.NULL_READS:
        read = (
            "at a fixed count, at α" if looks == 1 else f"over {looks} looks, each at α/{looks}, stopping at the first"
        )
        ties = {0.05: "rarely discordant", 0.25: "two fair coins", 0.5: "never tied"}[plus]
        claims.append(
            Claim(
                f"engine.null.{looks}.{cases}.{plus}",
                f"Under the null, {cases} pass/fail pairs ({ties}: each arm alone right {plus:.0%} of the time) read "
                f"{read} by the engine's bounded test, the fixed-n plug-in bet tuned to the level read, separate at "
                "most α.",
                f"engine-null-{looks}-{cases}-{plus}",
                "hits",
                "at_most",
                alpha,
                f"engine-null-{looks}-{cases}-{plus}-<index>, 0-9999, restated over the product's p",
            )
        )
        if (cases, plus, looks) == (engine.LOOK_CASES * engine.PLANNED, 0.25, engine.PLANNED):
            claims.append(
                Claim(
                    f"engine.null.{looks}.{cases}.{plus}.baseline",
                    f"The same reading is the uninterrupted engine watcher's over its plan (looks at {engine.LOOK_CASES}, "
                    f"{2 * engine.LOOK_CASES} and {cases} coin pairs): it separates at most "
                    f"{engine.UNINTERRUPTED_LOOKS:.0%}, the line a comparison whose interrupts cut is shown to lean past.",
                    f"engine-null-{looks}-{cases}-{plus}",
                    "hits",
                    "at_most",
                    engine.UNINTERRUPTED_LOOKS,
                    f"engine-null-{looks}-{cases}-{plus}-<index>, 0-9999, restated over the product's p",
                )
            )
    for control, contrast, rate, looks in engine.NULL_UNPAIRED:
        read = (
            "at a fixed count, at α" if looks == 1 else f"over {looks} looks, each at α/{looks}, stopping at the first"
        )
        claims.append(
            Claim(
                f"engine.null.unpaired.{looks}.{control}-{contrast}.{rate}",
                f"Under the null, two unpaired arms of {control} and {contrast} pass/fail cases, each passing "
                f"{rate:.0%} of the time, read {read} by the engine's unpaired bounded test (the inversion of its "
                "interval: each arm's mean bounded by the plug-in bet), separate at most α.",
                f"engine-unpaired-{looks}-{control}-{contrast}-{rate}",
                "hits",
                "at_most",
                alpha,
                f"engine-unpaired-{looks}-{control}-{contrast}-{rate}-<index>, 0-3999, restated over the product's p",
            )
        )
    for reading, law, grader, stop in sorted(engine.SHOWN):
        read = "at a fixed count of 40 at α" if reading == "fixed" else "over a plan of three looks, each at α/3"
        group = (
            f"engine-drain-{reading}-{law}-{grader}" if stop == "drain" else f"engine-{stop}-{reading}-{law}-{grader}"
        )
        seeds = f"engine-{reading}-{law}-{grader}-{stop}-<index>, through the engine"
        shown = engine.SHOWN[reading, law, grader, stop]
        if stop == "drain":
            claims.append(
                Claim(
                    f"engine.drain.{reading}.{law}.{grader}",
                    f"An engine comparison ({reading}, {law}, {grader}) whose interrupts drain draws every cell once and "
                    "separates exactly where the uninterrupted reading of its first draws does, its family read as the "
                    "product builds it. Each comparison held exactly, at the simulation's full size.",
                    group,
                    "held",
                    "every",
                    1.0,
                    seeds,
                )
            )
        claims.append(
            Claim(
                f"engine.{stop}.{reading}.{law}.{grader}.separates",
                f"Under the null, an engine comparison of coin pairs ({law}, {grader}) read {read} by the bounded test, "
                + {
                    "drain": "its interrupts draining,",
                    "second": "three of ten interrupts cut by a second,",
                    "cut": "every interrupt cutting,",
                }[stop]
                + (
                    " separates at most α."
                    if shown == "at_most"
                    else f" separates more often than the uninterrupted watcher can ({engine.UNINTERRUPTED_LOOKS:.0%}, "
                    "engine.null.3.60.0.25.baseline): the cells the cuts draw afresh lean it."
                ),
                group,
                "hits",
                shown,
                alpha if shown == "at_most" else engine.UNINTERRUPTED_LOOKS,
                seeds,
            )
        )
    return claims


#: Every certified claim.
CLAIMS: tuple[Claim, ...] = tuple(_claims())


@dataclass(frozen=True)
class Verdict:
    """A claim's count read against its target: the bounds, the figure the test reads, and whether it holds."""

    bounds: tuple[float, float]
    figure: float
    holds: bool


def judge(claim: Claim, count: int, replicates: int) -> Verdict:
    """Read ``count`` of ``replicates`` against ``claim``'s target, as :data:`Test` says."""
    interval = exact_rate_interval(count, replicates)
    if interval is None:
        return Verdict((0.0, 1.0), math.nan, False)
    low, high = interval
    if claim.test == "at_most":
        return Verdict(interval, high, high <= claim.target)
    if claim.test == "above":
        return Verdict(interval, low, low > claim.target)
    if claim.test == "share_below":
        share = count / replicates
        return Verdict(interval, share, share < claim.target)
    return Verdict(interval, count / replicates, count == replicates)


def tally(results: Mapping[str, Mapping[str, int]]) -> dict[str, tuple[int, int]]:
    """Each claim's (count, replicates), summed over its group's shards' ``results`` (by shard id)."""
    out: dict[str, tuple[int, int]] = {}
    for claim in CLAIMS:
        shards = [shard for shard in SHARDS if shard.group == claim.group]
        missing = [shard.id for shard in shards if shard.id not in results]
        if missing:
            raise KeyError(f"claim {claim.id} has no result for shards {missing}")
        out[claim.id] = (
            sum(int(results[shard.id].get(claim.field, 0)) for shard in shards),
            sum(int(results[shard.id]["n"]) for shard in shards),
        )
    return out


def certificate(results: Mapping[str, Mapping[str, int]], **about: Any) -> dict[str, Any]:
    """The certificate of ``results``: every claim with its count, bounds, verdict and seeds, and the closure's files
    with their digests and the digest over them. ``about`` adds what the run says of itself (its wall time, its pool
    request)."""
    files = current_files()
    counted = tally(results)
    claims = []
    for claim in CLAIMS:
        count, replicates = counted[claim.id]
        verdict = judge(claim, count, replicates)
        claims.append(
            {
                "id": claim.id,
                "says": claim.says,
                "test": claim.test,
                "target": claim.target,
                "count": count,
                "replicates": replicates,
                "bounds": list(verdict.bounds),
                "figure": verdict.figure,
                "holds": verdict.holds,
                "seeds": claim.seeds,
            }
        )
    return {
        "command": COMMAND,
        "digest": sources_digest(files),
        "entry_points": list(ENTRY_POINTS),
        "files": files,
        **about,
        "claims": claims,
    }


def write(document: Mapping[str, Any], path: Path = CERTIFICATE) -> None:
    path.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n")


def read(path: Path = CERTIFICATE) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text()))


def moved_files(document: Mapping[str, Any]) -> list[str]:
    """The files of the closure, now or when certified, whose digest differs from the certificate's: changed, joined
    the closure, or left it, each said so."""
    now = current_files()
    then = document.get("files", {})
    moved = []
    for path in sorted(set(now) | set(then)):
        if path not in then:
            moved.append(f"{path} (now imported)")
        elif path not in now:
            moved.append(f"{path} (no longer imported)")
        elif now[path] != then[path]:
            moved.append(path)
    return moved


def stale(document: Mapping[str, Any]) -> list[str]:
    """The certificate's claims, by id, if any file of the closure moved since it was written: every claim rests on
    the whole closure. Empty where the certificate is current."""
    if not moved_files(document) and document.get("digest") == sources_digest(document.get("files", {})):
        return []
    return [entry["id"] for entry in document.get("claims", [])]


def run_key() -> str:
    """The key the running certification named its metrics by, or ``unkeyed`` outside one."""
    return RUN_KEY.read_text().strip() if RUN_KEY.exists() else "unkeyed"


def metric_name(key: str, field_name: str) -> str:
    """The metric a pool shard records one count under."""
    return f"simcert_{key}_{field_name}"


def pool_shards() -> list[Shard]:
    """The shards the pool runs."""
    return [shard for shard in SHARDS if shard.where == "pool"]


def local_shards() -> list[Shard]:
    """The shards this machine runs."""
    return [shard for shard in SHARDS if shard.where == "local"]


def shard_by_id(shard_id: str) -> Shard:
    return next(shard for shard in SHARDS if shard.id == shard_id)


__all__ = [
    "CERTIFICATE",
    "CLAIMS",
    "COMMAND",
    "FIELDS",
    "SHARDS",
    "ENTRY_POINTS",
    "closure",
    "moved_files",
    "Claim",
    "Shard",
    "certificate",
    "judge",
    "local_shards",
    "metric_name",
    "pool_shards",
    "read",
    "run_key",
    "run_local",
    "run_pool",
    "run_shard",
    "shard_by_id",
    "stale",
    "tally",
    "write",
]
