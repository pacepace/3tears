#!/usr/bin/env bash
# Regenerate .test_durations, the per-test seconds pytest-split balances the evals CI shards by.
#
#   ./scripts/update-test-durations.sh [--since 14d]
#
# scripts/ci-unit-tests.sh splits packages/evals across the evals-N shards with pytest-split's
# duration_based_chunks algorithm: the collection, in order, cut into runs of about equal recorded
# duration (its header says why not least_duration). Without the file it splits by COUNT, and
# alphabetical order put the slow writer and reading files in one shard (56 minutes against 22).
#
# The seconds come from the testament pool's history (this needs `testament` on PATH and a key):
#   - the 1000 evals tests that cost the pool most (the most `testament tests slowest` returns,
#     ranked by total time) each get their median over the window;
#   - every other collected test gets its file's remaining time spread evenly: the file's mean
#     seconds per test-run (`testament time --by folder`) times its collected tests, less what the
#     measured ones take.
# Pool machines are not GitHub's runners, but only the ratios between tests matter to the split.
# One distortion: the pool batches a file's tests apart, so each batch pays the file's module
# fixtures and a fixture-heavy file reads slower than it runs in a shard. It costs some balance,
# not correctness; check a refresh against the unit (evals-N) jobs' wall clocks.
#
# Stale entries are fine. A test added since the file was written gets pytest-split's average
# duration, one renamed or deleted is ignored, and either way every test still runs exactly once:
# the file moves tests between shards, it never selects them. Refresh it when a shard's wall clock
# drifts well past the others' (ci.yml's unit (evals-N) jobs), not on every change.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

since="14d"
if [ "${1:-}" = "--since" ]; then
    since="${2:?--since takes an age, e.g. 14d}"
fi

command -v testament >/dev/null 2>&1 || { echo "testament is not on PATH (docs/testing.md)" >&2; exit 2; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "collecting packages/evals (the evals shards' selection)..." >&2
# collect-only: the ids pytest-split will see, as ci-unit-tests.sh's evals session collects them
uv run pytest packages/evals -m "not integration and not loop_timing" --collect-only -q >"$work/ids.txt" || {
    echo "collection failed; see the output above" >&2
    exit 1
}

echo "reading durations from the testament pool (since ${since})..." >&2
testament tests slowest --project 3tears --area evals --limit 1000 --since "$since" --json >"$work/slowest.json"
testament time --by folder --depth 8 --area evals --project 3tears --limit 1000 --since "$since" --json >"$work/files.json"

uv run python - "$work" .test_durations <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

work, out = Path(sys.argv[1]), Path(sys.argv[2])


def result(name: str) -> dict:
    envelope = json.loads((work / name).read_text())
    if not envelope.get("ok"):
        sys.exit(f"testament refused {name}: {envelope.get('error')}")
    return envelope["result"]


ids = [line.strip() for line in (work / "ids.txt").read_text().splitlines() if "::" in line]
slowest = {row["item_id"]: row["median_s"] for row in result("slowest.json")["rows"]}
mean = {}  # file -> its mean seconds per test-run
for row in result("files.json")["rows"]:
    if row["runs"]:
        mean[row["folder"]] = row["test_s"] / row["runs"]

by_file = defaultdict(list)
for node in ids:
    by_file[node.split("::", 1)[0]].append(node)

durations = {}
estimates = []
for path, nodes in by_file.items():
    known = [n for n in nodes if n in slowest]
    unknown = [n for n in nodes if n not in slowest]
    for n in known:
        durations[n] = slowest[n]
    if unknown and path in mean:
        each = max(mean[path] * len(nodes) - sum(slowest[n] for n in known), 0.0) / len(unknown)
        estimates.append(each)
        for n in unknown:
            durations[n] = each
# a file the pool has no time for (new since the window): its file-less tests take the typical estimate
typical = sorted(estimates)[len(estimates) // 2] if estimates else 0.1
for node in ids:
    durations.setdefault(node, typical)

# the id as pytest-split sees it on an xdist worker: under --dist loadgroup, xdist suffixes each id
# with its xdist_group, which the root conftest sets to the test's package, before pytest-split
# (a trylast hook) reads it. One entry a line, sorted, so a refresh diffs line by line.
lines = [f"{json.dumps(k + '@packages/evals')}: {round(v, 3)}" for k, v in sorted(durations.items())]
out.write_text("{\n" + ",\n".join(lines) + "\n}\n")
total = sum(durations.values())
print(
    f"wrote {out}: {len(durations)} tests, {sum(n in slowest for n in ids)} measured, {total / 60:.0f} pool-minutes in all",
    file=sys.stderr,
)
PY
