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
decide it (:data:`SOURCES`). ``test_stats_certificate.py`` fails, naming each stale claim and this command, when any of
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
from collections.abc import Iterable, Mapping, Sequence
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

_SRC = "packages/evals/src/threetears/evals/"
_TESTS = "packages/evals/tests/"

#: The files that decide each claim, by group. A claim's digest is over every file of the groups it names
#: (:attr:`Claim.sources`); a change to any of them makes the claim stale.
SOURCES: dict[str, tuple[str, ...]] = {
    # The bet, its interval, its evidence, Holm, Clopper-Pearson, the t test the engine reads.
    "stats": (_SRC + "analysis/stats.py",),
    # The quick read path: how a comparison is filed, assembled and read, its committed order and its family.
    "read": tuple(
        _SRC + path
        for path in (
            "analysis/significance.py",
            "analysis/summary.py",
            "analysis/campaigns.py",
            "analysis/run_ending.py",
            "analysis/contention.py",
            "analysis/bundle/assemble.py",
            "analysis/bundle/comparisons.py",
            "analysis/bundle/trials.py",
            "analysis/bundle/observations.py",
            "kernel/candidate_kind.py",
            "kernel/result_condition.py",
            "kernel/result_reading.py",
            "quick/contract.py",
            "quick/result.py",
            "quick/one_call.py",
            "quick/continuation.py",
            "quick/graders.py",
            "schema/hashing.py",
        )
    ),
    # The runner: how a run launches, plays its cells, stops, drains, cuts and files.
    "runner": tuple(
        _SRC + path
        for path in (
            "run/launch.py",
            "run/executor.py",
            "run/runner.py",
            "run/jobs.py",
            "run/lifecycle.py",
            "run/arm_schedule.py",
            "run/budget.py",
            "run/reads.py",
            "kernel/offload.py",
        )
    ),
    # The engine's operator path: launch, cancel, take-up and leases.
    "engine": tuple(
        _SRC + path
        for path in (
            "ops/jobs.py",
            "ops/sweeps.py",
            "ops/resume.py",
            "ops/runs.py",
            "run/launch_job.py",
            "run/resume.py",
            "run/job_work.py",
            "kernel/job_leases.py",
        )
    ),
    # The simulations and the certification itself.
    "quick-simulation": (_TESTS + "null_newcomers.py",),
    "engine-simulation": (_TESTS + "engine_take_up.py",),
    "certification": (_TESTS + "stats_certificate.py", _TESTS + "test_certified_simulations.py"),
}

_QUICK_SOURCES = ("stats", "read", "runner", "quick-simulation", "certification")
_ENGINE_SOURCES = ("stats", "read", "runner", "engine", "engine-simulation", "certification")


def file_digest(path: str) -> str:
    """One source file's sha256, read from the repository's root."""
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def sources_digest(groups: Iterable[str], files: Mapping[str, str] | None = None) -> str:
    """The digest of every file of ``groups``: a sha256 over each path and its own digest, in path order. ``files``
    gives each file's digest where already read."""
    paths = sorted({path for group in groups for path in SOURCES[group]})
    hashed = hashlib.sha256()
    for path in paths:
        hashed.update(f"{path}\0{files[path] if files is not None else file_digest(path)}\n".encode())
    return hashed.hexdigest()


def every_source() -> list[str]:
    """Every file any claim names, once, in path order."""
    return sorted({path for paths in SOURCES.values() for path in paths})


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
        ("alike-apart", 3, 5, "lead", 40, 20),
        ("alike-slow-wrong", 3, 5, "lead", 40, 20),
        ("b-slower-4x", 1, 2, "blind", 150, 75),
        ("b-slower-20x", 1, 2, "blind", 150, 75),
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
            shards.append(
                Shard(
                    f"engine-drain-{reading}-{law}-{grader}",
                    f"engine-drain-{reading}-{law}-{grader}",
                    "pool",
                    "engine-drained",
                    {"reading": reading, "law": law, "grader": grader, "start": 0, "count": 40},
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
#: - ``every``: every newcomer or comparison held the exact property (``held`` equals ``n``);
#: - ``no_lean``: the count's one-sided binomial tail at the reading's exact rate (the target) is at least
#:   :data:`~packages.evals.tests.engine_take_up.SURE`: the interrupts leaned nothing seen;
#: - ``leans``: that tail is below it: the redrawn cells lean the reading.
Test = Literal["at_most", "above", "share_below", "every", "no_lean", "leans"]


@dataclass(frozen=True)
class Claim:
    """One certified claim: its id, what it says, the shard group that counts it and the field it counts, how the
    count is held to the target, the source groups that decide it, and its seeds."""

    id: str
    says: str
    group: str
    field: str
    test: Test
    target: float
    sources: tuple[str, ...]
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
                _QUICK_SOURCES,
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
                _QUICK_SOURCES,
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
                    _QUICK_SOURCES,
                    seeds,
                )
            )
    claims.append(
        Claim(
            "small-n.interval.misses",
            "The primary's interval, read once at sixteen cases at a true difference of 0.3, misses it at most the 10% "
            "its 90% states.",
            "small-cover",
            "hits",
            "at_most",
            1.0 - 0.90,
            _QUICK_SOURCES,
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
                _QUICK_SOURCES,
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
            _QUICK_SOURCES,
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
            _QUICK_SOURCES,
            "rerun-<shard>-<index>, shards 0-39 of 50, restated",
        )
    )
    for flow, said in (
        ("small-once", "a question's single read at twelve cases, under each null"),
        ("small-continued", "a question's comparison continued to five reads, under each null"),
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
                _QUICK_SOURCES,
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
                _QUICK_SOURCES,
                f"horizon-k{k}-<index>, 0-9999, restated",
            )
        )
    for latency in ("alike-apart", "alike-slow-wrong", "b-slower-4x", "b-slower-20x"):
        claims.append(
            Claim(
                f"watched.drained.{latency}",
                f"Watchers whose arms are {latency}, every stop drained, through Eval.run: every reading is the bet over "
                "the longest complete prefix of the committed order, no cell draws twice, and the product separates only "
                "where the bet over its whole committed path does.",
                f"quick-drained-{latency}",
                "held",
                "every",
                1.0,
                _QUICK_SOURCES,
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
                _QUICK_SOURCES,
                f"cut-{latency}-<index>, 0-999, through the product",
            )
        )
    for reading, law, grader, stop in sorted(engine.SHOWN):
        exact = engine.exact_rate(cast("engine.Reading", reading))
        group = (
            f"engine-drain-{reading}-{law}-{grader}" if stop == "drain" else f"engine-{stop}-{reading}-{law}-{grader}"
        )
        seeds = f"engine-{reading}-{law}-{grader}-{stop}-<index>, through the engine"
        if stop == "drain":
            claims.append(
                Claim(
                    f"engine.drain.{reading}.{law}.{grader}",
                    f"An engine comparison ({reading}, {law}, {grader}) whose interrupts drain draws every cell once and "
                    f"separates exactly where the uninterrupted reading of its first draws does, so it separates at the "
                    f"exact rate {exact:.5f}: no lean.",
                    group,
                    "held",
                    "every",
                    1.0,
                    _ENGINE_SOURCES,
                    seeds,
                )
            )
        else:
            shown = engine.SHOWN[reading, law, grader, stop]
            claims.append(
                Claim(
                    f"engine.{stop}.{reading}.{law}.{grader}",
                    f"An engine comparison ({reading}, {law}, {grader}) whose interrupts "
                    + ("cut a second time in three of ten" if stop == "second" else "every one cut")
                    + (
                        " leans no more than the exact rate's tail allows"
                        if shown == "no_lean"
                        else " leans the reading"
                    )
                    + f" (exact rate {exact:.5f}, one-sided at {engine.SURE}).",
                    group,
                    "hits",
                    cast("Test", shown),
                    exact,
                    _ENGINE_SOURCES,
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
    if claim.test == "every":
        return Verdict(interval, count / replicates, count == replicates)
    tail = engine.tail(count, replicates, claim.target)
    return Verdict(interval, tail, tail >= engine.SURE if claim.test == "no_lean" else tail < engine.SURE)


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
    """The certificate of ``results``: every claim with its count, bounds, verdict, seeds and digest, and every source
    file's digest. ``about`` adds what the run says of itself (its wall time, its pool request)."""
    files = {path: file_digest(path) for path in every_source()}
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
                "sources": list(claim.sources),
                "digest": sources_digest(claim.sources, files),
            }
        )
    return {
        "command": COMMAND,
        "digest": sources_digest(SOURCES, files),
        "files": files,
        **about,
        "claims": claims,
    }


def write(document: Mapping[str, Any], path: Path = CERTIFICATE) -> None:
    path.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n")


def read(path: Path = CERTIFICATE) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text()))


def stale(document: Mapping[str, Any]) -> list[str]:
    """The certificate's claims whose digest is no longer their sources', by id, each with the files that moved."""
    files = {path: file_digest(path) for path in every_source() if (ROOT / path).exists()}
    moved = sorted(path for path in every_source() if document.get("files", {}).get(path) != files.get(path))
    out = []
    for entry in document.get("claims", []):
        claim = next((claim for claim in CLAIMS if claim.id == entry["id"]), None)
        if claim is None:
            continue
        mine = sorted({path for group in claim.sources for path in SOURCES[group]} & set(moved))
        if mine or entry.get("digest") != sources_digest(
            claim.sources, files if len(files) == len(every_source()) else None
        ):
            out.append(f"{claim.id} (changed: {', '.join(mine) or 'its source list'})")
    return out


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
    "SOURCES",
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
