#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# The on-ramp's agent test: can an agent with no prior knowledge add an eval to an unfamiliar app from the docs and
# error messages alone? An eval on the package itself (packages/evals/onramp/): its cases are three small apps with a
# planted defect, its arms are docs builds (a wheel and the llms.txt inside it), and its runner is an agent.
#
# Usage:
#   ./scripts/onramp-agent.sh --dry-run                       what would run, and what it may cost; builds nothing
#   ./scripts/onramp-agent.sh --scripted succeeds --k 1       the harness against today's wheel, offline and free
#   ... --scripted succeeds --host                            the same, its commands run on this machine (scripted only)
#   ONRAMP_LIVE=1 ./scripts/onramp-agent.sh --cap 4.50        the live baseline: cheap tier, three apps, k=3
#   ... --tier ceiling --cap 13.50                            Claude Haiku 5.5, direct to Anthropic
#   ... --arm docs@candidate=PATH.whl                         a second docs build, tested against the control
#
# The control arm, docs@control, is today's wheel: 3tears-evals built from ONRAMP_CONTROL_REF (default f9ab6f74d,
# the docs and messages before any on-ramp fix), so a later fix never leaks into the baseline. It is built once into
# packages/evals/onramp/runs/wheels/<ref>/ and reused.
#
# Every command an agent runs, and every re-run of its work, runs in a trial container (Docker, no network, not root,
# nothing of this machine but a copy of the trial's directory): packages/evals/onramp/isolation.py. The image is built
# at set-up from packages/evals/onramp/container/Dockerfile and the wheel. A run that cannot have one (no Docker daemon,
# or the image did not build) is refused before anything is sent. --host, which runs commands here, is only for
# --scripted. The canaries that hold this: ./scripts/test-integration.sh --here -k onramp -rs.
#
# A live run spends money. It is refused unless ONRAMP_LIVE=1 is set and --cap states the most it may spend; it quotes
# its cost first, holds each run to its share of the cap, and reads keys only from the environment: OPENROUTER_API_KEY
# for the cheap tiers, ANTHROPIC_API_KEY for the ceiling. Runs, the store, the report and every trial's record land in
# packages/evals/onramp/runs/<timestamp>/.

CONTROL_REF="${ONRAMP_CONTROL_REF:-f9ab6f74d}"
WHEELS="packages/evals/onramp/runs/wheels/$CONTROL_REF"

DRY_RUN=0
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then DRY_RUN=1; fi
done

wheel() {
    [ -d "$WHEELS" ] || return 0
    find "$WHEELS" -maxdepth 1 -name '3tears_evals-*.whl' | sort | head -n 1
}

if [ -z "$(wheel)" ] && [ "$DRY_RUN" = 0 ]; then
    echo "onramp-agent: building today's wheel from $CONTROL_REF" >&2
    SOURCE="$(mktemp -d)"
    trap 'rm -rf "$SOURCE"' EXIT
    git archive "$CONTROL_REF" packages/evals | tar -x -C "$SOURCE"
    mkdir -p "$WHEELS"
    uv build --quiet --wheel --out-dir "$WHEELS" "$SOURCE/packages/evals"
fi

CONTROL="$(wheel)"
CONTROL="${CONTROL:-$WHEELS/<built from $CONTROL_REF on the first run>}"
exec uv run --frozen python -m packages.evals.onramp --arm "docs@control=$CONTROL" "$@"
