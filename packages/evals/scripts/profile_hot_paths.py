"""Profile the evals hot paths on the production-shaped campaign and print where the time goes, ranked.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. Run from the repo root:

    uv run python packages/evals/scripts/profile_hot_paths.py            # the full profile (about ten minutes)
    uv run python packages/evals/scripts/profile_hot_paths.py --quick    # 1x and 2x only, no json read

It builds the campaign of ``tests/hot_path_support.py`` (41 runs, about 3,100 results at 1x, each about 7 KB, its
bytes split across its fields as production's rows split theirs) and reads it as a host does, under ``cProfile``:

- **report_read**, code-only, html with the trials table, at 1x, 2x and 4x (2x and json at 1x and 2x), its time split
  into phases: the store's own calls, hydration (``_hydrate_all``), the score projection
  (``project_score_records``), each fold ``assemble_context_bundle`` makes (every function it calls directly), the
  campaign bundle (``build_campaign_bundle``, ``campaign_readings``, ``campaign_arms``), the page build and
  ``serialize_report``, and the innermost functions the read spends most in;
- **launch_estimate**: the launch's quote (``quote_launch``, the toy host's recorded pricer) against the planning read
  (``detectable_effects``) over the campaign's runs as the launched template's history;
- **analysis_estimate**: the result reads it makes (none, by design), its whole time over the toy host's recorded
  analyst client, and a catalogue-priced model's ``turn_pricing_refusal`` against a recorded catalogue (no network).

The ranked table lists each phase's share of the read's wall time under the profiler, its cumulative seconds, its
call counts at each scale (a count, unlike a time, is deterministic), and the area of the engine the file it lives in
belongs to. Wall times under ``cProfile`` run about twice the unprofiled ones; the unprofiled baseline of the code-only
html read and of the assembly alone are printed first.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import pstats
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import httpx

# Run as a script, this file's directory is first on the path and the repo root is not; the corpus is test support.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from threetears.evals.analysis import service as analysis_service  # noqa: E402
from threetears.evals.analysis.service import assemble_campaign_bundle  # noqa: E402
from threetears.evals.ops import analysis_estimate, launch_estimate, report_read  # noqa: E402
from threetears.evals.ops.lenses import LaunchArguments, detectable_effects  # noqa: E402
from threetears.evals.providers import Catalogue, Model  # noqa: E402
from threetears.evals.storage import InMemoryDocumentStore  # noqa: E402
from packages.evals.tests.fixtures.toyhost.run import toyhost_template  # noqa: E402
from packages.evals.tests.hot_path_support import (  # noqa: E402
    ARM_MODELS,
    CASES_PER_RUN,
    HOT_PATH_INSTANT,
    K_RUNS,
    HotPathCorpus,
    field_byte_shares,
    hot_path_corpus,
    report_read_counts,
)
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture  # noqa: E402

#: The package's source root, which every profiled file name is shortened against.
SOURCE = "threetears/evals/"

#: Which area of the engine each file belongs to, by path prefix under :data:`SOURCE`; first match wins.
OWNERS: tuple[tuple[str, str], ...] = (
    ("analysis/reporting.py", "score projection"),
    ("kernel/result_condition.py", "score projection"),
    ("kernel/scoring.py", "score projection"),
    ("analysis/bundle/", "bundle folds"),
    ("analysis/cells.py", "bundle folds"),
    ("analysis/agreement.py", "bundle folds"),
    ("analysis/judge_profiles.py", "bundle folds"),
    ("analysis/judge_drift.py", "bundle folds"),
    ("analysis/contention.py", "bundle folds"),
    ("analysis/campaign_bundle.py", "campaign page"),
    ("analysis/report/", "campaign page"),
    ("analysis/report_input_build.py", "campaign page"),
    ("kernel/storage.py", "store"),
    ("schema/models.py", "store"),
    ("ops/lenses.py", "planning read"),
    ("run/reads.py", "planning read"),
    ("analysis/service.py", "analysis entry"),
    ("ops/analysis.py", "analysis entry"),
    ("actions/catalogue.py", "analysis entry"),
)

#: The named phases of a report read: ``(label, file under SOURCE, function name)``. The folds are added from the
#: profile itself (every function ``assemble_context_bundle`` calls directly), and the innermost functions too.
NAMED_PHASES: tuple[tuple[str, str, str], ...] = (
    ("report_read (whole read)", "ops/analysis.py", "report_read"),
    ("assemble_context_bundle (every aspect)", "analysis/bundle/assemble.py", "assemble_context_bundle"),
    ("_hydrate_all (hydration)", "kernel/storage.py", "_hydrate_all"),
    ("project_score_records", "analysis/reporting.py", "project_score_records"),
    ("campaign_arms", "analysis/campaigns.py", "campaign_arms"),
    ("campaign_readings", "analysis/campaign_readings.py", "campaign_readings"),
    ("build_campaign_bundle", "analysis/campaign_bundle.py", "build_campaign_bundle"),
    ("page build (_campaign_page)", "analysis/service.py", "_campaign_page"),
    ("serialize_report", "analysis/report/serialize.py", "serialize_report"),
)

#: The reference store's read methods: their summed time is the store's own share.
STORE_READS = ("get", "get_many", "by_doc_type", "get_with_etag", "iter_by_doc_type")

#: Phases below this share of the read are left out of the ranked table.
FLOOR = 0.01

#: Innermost functions listed in the table: the ones with the most time of their own.
INNERMOST = 8

#: The instant a code-only page is assembled at, so the reads compared are the same read.
REPORT_NOW = "2026-10-30T09:00:00+00:00"

#: One recorded catalogue row: the model a host's default pricer does not list, so a turn is priced from it.
RECORDED_CATALOGUE = {
    "data": [
        {
            "id": "anthropic/claude-opus-9",
            "pricing": {"prompt": "0.000004", "completion": "0.00002", "input_cache_read": "0.0000004"},
        }
    ]
}

Key = tuple[str, int, str]


@dataclass
class Profiled:
    """One profiled call: its stats, its wall time under the profiler, and what it returned."""

    stats: pstats.Stats
    wall_s: float
    value: Any


def profiled(call: Callable[[], Any]) -> Profiled:
    """``call()`` under ``cProfile``."""
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    value = call()
    profiler.disable()
    return Profiled(pstats.Stats(profiler), time.perf_counter() - started, value)


def short(path: str) -> str:
    """A profiled file name, under the package's source root where it is in it."""
    return path.split(SOURCE, 1)[1] if SOURCE in path else path.rsplit("/", 1)[-1]


def owner(path: str) -> str:
    """The area ``path`` belongs to, or ``unowned`` naming the file."""
    name = short(path)
    for prefix, area in OWNERS:
        if name.startswith(prefix):
            return area
    return f"unowned ({name})"


def entries(stats: pstats.Stats) -> dict[Key, tuple[int, int, float, float, dict[Key, tuple[int, int, float, float]]]]:
    """The profile's table: ``(file, line, function) -> (primitive calls, calls, own s, cumulative s, callers)``."""
    table: dict[Key, Any] = stats.stats  # type: ignore[attr-defined]  # pstats keeps its table untyped
    return table


def find(stats: pstats.Stats, file: str, function: str) -> Key | None:
    """The profile's key for ``function`` in ``file`` (under the source root), or None when it never ran."""
    for key in entries(stats):
        if key[2] == function and short(key[0]) == file:
            return key
    return None


@dataclass
class Row:
    """One row of the ranked table at one scale."""

    label: str
    seconds: float
    calls: int
    owner: str


def phase_rows(stats: pstats.Stats) -> dict[str, Row]:
    """Every phase of one profiled report read, by label (see :data:`NAMED_PHASES`)."""
    table = entries(stats)
    rows: dict[str, Row] = {}
    for label, file, function in NAMED_PHASES:
        key = find(stats, file, function)
        if key is not None:
            _cc, calls, _own, cumulative, _callers = table[key]
            rows[label] = Row(label, cumulative, calls, owner(key[0]))
    store = [key for key in table if short(key[0]) == "storage/memory.py" and key[2] in STORE_READS]
    rows["store reads (InMemoryDocumentStore)"] = Row(
        "store reads (InMemoryDocumentStore)",
        sum(table[key][3] for key in store),
        sum(table[key][1] for key in store),
        "reference store",
    )
    assemble = find(stats, "analysis/bundle/assemble.py", "assemble_context_bundle")
    if assemble is not None:
        for key, (_cc, _calls, _own, _cumulative, callers) in table.items():
            if assemble in callers:
                _ecc, edge_calls, _eown, edge_cumulative = callers[assemble]
                label = f"  fold: {key[2]} ({short(key[0])})"
                rows[label] = Row(label, edge_cumulative, edge_calls, owner(key[0]))
    innermost = sorted(table.items(), key=lambda item: item[1][2], reverse=True)[:INNERMOST]
    for key, (_cc, calls, own, _cumulative, _callers) in innermost:
        label = f"  own time: {key[2]} ({short(key[0])}:{key[1]})"
        rows[label] = Row(label, own, calls, owner(key[0]))
    return rows


def ranked_table(title: str, by_scale: Mapping[int, Profiled]) -> str:
    """The ranked table over each scale's profile of one read, ranked by the smallest scale's seconds."""
    scales = sorted(by_scale)
    rows = {scale: phase_rows(by_scale[scale].stats) for scale in scales}
    first = rows[scales[0]]
    whole = by_scale[scales[0]].wall_s
    lines = [
        f"## {title}",
        "",
        "| Phase | Share | "
        + " | ".join(f"{scale}x s" for scale in scales)
        + " | "
        + " | ".join(f"{scale}x calls" for scale in scales)
        + " | Area |",
        "|---|---:|" + "---:|" * (2 * len(scales)) + "---|",
    ]
    for label, row in sorted(first.items(), key=lambda item: item[1].seconds, reverse=True):
        if row.seconds / whole < FLOOR:
            continue
        seconds = " | ".join(f"{rows[scale][label].seconds:.2f}" if label in rows[scale] else "-" for scale in scales)
        calls = " | ".join(f"{rows[scale][label].calls:,}" if label in rows[scale] else "-" for scale in scales)
        lines.append(f"| {label} | {row.seconds / whole:.1%} | {seconds} | {calls} | {row.owner} |")
    walls = ", ".join(f"{scale}x {by_scale[scale].wall_s:.1f}s" for scale in scales)
    lines += ["", f"Wall under the profiler: {walls}.", ""]
    return "\n".join(lines)


def pinned() -> Any:
    """The report's clock pinned, so every read of one campaign is the same read."""
    return mock.patch.object(analysis_service, "utc_now_iso", lambda: REPORT_NOW)


def read_report(corpus: HotPathCorpus, form: Any) -> Callable[[], Any]:
    """One code-only report read of ``corpus`` in ``form``, with the trials table."""

    def read() -> Any:
        with pinned():
            return report_read(corpus.host, corpus.campaign.id, corpus.scope_id, format=form)

    return read


def timed(call: Callable[[], Any]) -> float:
    """``call()``'s wall time, unprofiled."""
    started = time.perf_counter()
    call()
    return time.perf_counter() - started


class CountingStore(InMemoryDocumentStore):
    """The reference store, counting each read by method and document type."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: Counter[tuple[str, str]] = Counter()

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.reads["get_many", doc_type] += 1
        return super().get_many(doc_type, doc_ids, scope_id, **kwargs)

    def by_doc_type(
        self, doc_type: str, scope_id: str, *, field_in: Mapping[str, Sequence[Any]] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.reads["by_doc_type", doc_type] += 1
        return super().by_doc_type(doc_type, scope_id, field_in=field_in, **kwargs)

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        self.reads["get", "-"] += 1
        return super().get(doc_id, scope_id)


def launch_section(scales: Sequence[int]) -> str:
    """launch_estimate over the campaign's runs as the launched template's history, at each scale."""
    lines = ["## launch_estimate", "", "| Scale | Results | launch_estimate s | detectable_effects s | quote share |"]
    lines.append("|---:|---:|---:|---:|---:|")
    profiles: dict[int, Profiled] = {}
    for scale in scales:
        store = InMemoryDocumentStore()
        corpus = hot_path_corpus(scale=scale, scope_id=TOYHOST_SCOPE, template_id=toyhost_template().id, store=store)
        # The toy host's own template is saved over the corpus's: the launch quotes the template the host launches.
        fixture = ops_fixture(store=store)
        arguments = LaunchArguments(
            template_id=toyhost_template().id,
            subject_id=TOYHOST_SUBJECT.subject_id,
            models=list(ARM_MODELS),
            k_runs=K_RUNS,
        )
        whole = timed(lambda: asyncio.run(launch_estimate(fixture.host, arguments, TOYHOST_SCOPE)))
        planning = profiled(
            lambda: detectable_effects(
                fixture.host.eval_host,
                TOYHOST_SCOPE,
                toyhost_template().id,
                n_cases=CASES_PER_RUN,
                k_runs=K_RUNS,
                n_arms=len(ARM_MODELS),
            )
        )
        profiles[scale] = planning
        plain = timed(
            lambda: detectable_effects(
                fixture.host.eval_host,
                TOYHOST_SCOPE,
                toyhost_template().id,
                n_cases=CASES_PER_RUN,
                k_runs=K_RUNS,
                n_arms=len(ARM_MODELS),
            )
        )
        lines.append(
            f"| {scale}x | {corpus.n_results:,} | {whole:.2f} | {plain:.2f} | {max(0.0, whole - plain) / whole:.1%} |"
        )
    lines += ["", "detectable_effects under the profiler, by its heaviest callees:", ""]
    smallest = profiles[min(profiles)]
    table = entries(smallest.stats)
    top = find(smallest.stats, "ops/lenses.py", "detectable_effects")
    if top is not None:
        callees = sorted(
            ((key, callers[top]) for key, (_cc, _n, _own, _cum, callers) in table.items() if top in callers),
            key=lambda item: item[1][3],
            reverse=True,
        )[:6]
        for key, (_ecc, calls, _own, cumulative) in callees:
            lines.append(
                f"- {key[2]} ({short(key[0])}): {cumulative:.2f}s of {smallest.wall_s:.2f}s, {calls:,} calls, "
                f"{owner(key[0])}"
            )
    lines.append("")
    return "\n".join(lines)


def analysis_section() -> str:
    """analysis_estimate's result reads and time, and a catalogue-priced model's pricing check."""
    store = CountingStore()
    fixture = ops_fixture(store=store)
    store.reads.clear()
    elapsed = timed(lambda: asyncio.run(analysis_estimate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)))
    result_reads = sum(count for (_method, doc_type), count in store.reads.items() if doc_type == "eval_result")
    run_reads = sum(count for (_method, doc_type), count in store.reads.items() if doc_type == "eval_run")

    async def refusal() -> str | None:
        rows = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=RECORDED_CATALOGUE)))
        refused = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503, text="no network")))
        model = Model(
            "claude-opus-9", provider="anthropic", api_key="recorded", catalogue=Catalogue(http=rows), http=refused
        )
        return await model.turn_pricing_refusal()

    started = time.perf_counter()
    answered = asyncio.run(refusal())
    pricing = time.perf_counter() - started
    return "\n".join(
        [
            "## analysis_estimate",
            "",
            f"- store reads during the estimate: {dict(store.reads)}; eval_result reads {result_reads}, eval_run "
            f"reads {run_reads}",
            f"- analysis_estimate over the toy host's recorded analyst client: {elapsed:.3f}s",
            f"- turn_pricing_refusal on a catalogue-priced model, the catalogue recorded: {pricing:.3f}s "
            f"(answered {answered!r})",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Build, read, profile and print."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="1x and 2x only, and no json read")
    options = parser.parse_args(argv)
    scales = (1, 2) if options.quick else (1, 2, 4)

    corpora = {scale: hot_path_corpus(scale=scale) for scale in (1, 2)}
    size, shares = field_byte_shares(corpora[1])
    print(f"# Evals hot paths, profiled ({corpora[1].n_results:,} results at 1x, mean {size:,.0f} bytes stored)\n")
    print(
        "Field shares of a stored result: "
        + ", ".join(f"{name} {share:.1%}" for name, share in list(shares.items())[:10])
    )
    print()

    baseline = timed(read_report(corpora[1], "html"))
    with pinned():
        assembly = timed(lambda: assemble_campaign_bundle(corpora[1].host, corpora[1].campaign.id, corpora[1].scope_id))
    print(
        f"Baseline, unprofiled, at 1x: report_read code-only html (trials table) {baseline:.2f}s; assembly alone "
        f"(assemble_campaign_bundle) {assembly:.2f}s.\n"
    )

    counted = report_read_counts(corpora[1])
    print(
        "Counted operations at 1x (assembly / whole read): "
        + ", ".join(
            f"{name} {counted['assembly'][name]:,} / {counted['report_read'][name]:,}"
            for name in counted["report_read"]
        )
    )
    print()

    html = {scale: profiled(read_report(corpora[scale], "html")) for scale in (1, 2)}
    if 4 in scales:
        html[4] = profiled(read_report(hot_path_corpus(scale=4), "html"))
    print(ranked_table("report_read: code-only, html, trials table", html))
    if not options.quick:
        print(
            ranked_table(
                "report_read: code-only, json, trials table",
                {s: profiled(read_report(corpora[s], "json")) for s in (1, 2)},
            )
        )
    print(launch_section((1, 2)))
    print(analysis_section())
    print(f"(campaign instants are offsets of {HOT_PATH_INSTANT}; the report clock is pinned at {REPORT_NOW})")


if __name__ == "__main__":
    main()
