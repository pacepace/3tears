#!/usr/bin/env bash
# CI's unit + enforcement suite on the testament pool: ci.yml's `route` and `unit (testament pool)` jobs.
#
#   ./scripts/ci-unit-pool.sh route   choose the route: the pool if it answers a probe, else the GitHub shards
#   ./scripts/ci-unit-pool.sh run     run the suite on the pool; say whether it passed, failed, or was lost
#
# The same tests `./scripts/ci-unit-tests.sh` runs on GitHub's runners: .testament.toml selects
# packages/ and tests/ with -m "not integration and not loop_timing", and the pool spreads them
# over its runners, so GitHub's job only waits; the loop_timing tests run apart (`-m`, below). docs/testing.md, "CI on the
# testament pool", lists every difference.
#
# THE ROUTE (`route`, writes pool=true|false and why=<text> to $GITHUB_OUTPUT). The shards run,
# and the pool does not, when any of these holds, checked in this order:
#   - the repo variable TESTAMENT_CI is "off": the shards on purpose, with the secret left in place;
#   - no TESTAMENT_KEY reached the run (a fork's pull request gets no secrets);
#   - the event is neither a push nor a pull request from a branch of this repo (ci.yml already
#     withholds the key then; this is the second check, over GitHub's own fork policy);
#   - no TESTAMENT_COORDINATOR variable;
#   - the client cannot be had: TESTAMENT_CONSOLE is not https (the key is never sent to it), the
#     download fails or fails its integrity check, or uv cannot install it;
#   - `testament run --probe` says no run is possible (coordinator_unreachable, no_runner, ...), or
#     gives no answer within TESTAMENT_PROBE_TIMEOUT seconds (60): a pool that takes the connection
#     and never answers would otherwise hold `route` to its job timeout, and `check` would fail
#     instead of falling back.
# The probe needs the client, so `route` installs it: the probe is the one check that a runner is
# connected and can take this project, which reaching the port would not say. A fallback is never
# silent: the reason goes to $GITHUB_OUTPUT, the step summary and an annotation.
#
# THE RUN (`run`, writes outcome=passed|failed|unreachable and why=<text> to $GITHUB_OUTPUT):
#   testament run --fresh --no-cache --no-fallback --json=summary -- -m "not loop_timing"
# --fresh: every batch in a container started cold from the project's image, so nothing an earlier
#   run left in a warm container reaches the required check (testament's advice for a merge or
#   release gate). It does not turn the cache off, so:
# --no-cache: every test runs; none is served from an earlier run's result on the same tree.
# --no-fallback: never testament's own local fallback, which would run the whole suite in one
#   session on this one runner (ci-unit-tests.sh's header: that OOMs its workers, and takes ~2 h).
#   ci.yml falls back to the GitHub shards instead, as below.
# -m "not loop_timing": the event loop's wall-clock tests stay off the pool. The manifest's own
#   expression already leaves them out; this says so again where CI depends on it, and testament
#   narrows, never replaces (the client sends -m "(not integration and not loop_timing) and (not
#   loop_timing)"). On a shared runner they run beside other batches,
#   and a bound on the code becomes a bound on the machine (#1140; run 43087460269: two of them
#   22 ms late). ci.yml's `unit (testament pool)` job runs them after this, serially on its own
#   runner (`ci-unit-tests.sh --loop-timing`), unless the pool was lost and the shards run them.
# The outcome, from testament's exit code and its envelope (classify, below):
#   passed       exit 0
#   failed       exit 1 (a test failed), any failed or errored test in the report whatever the
#                exit, exit 5 (nothing collected), or exit 2 with any other error code
#                (forbidden, revoked, upgrade_required, ...): CI is red
#   unreachable  exit 2 with error.code coordinator_unreachable, connection_lost, batches_lost or
#                no_runner, and no test failed: the pool is down or was lost, not the code. Or the
#                client could not be had (the console's download or uv's install failed), so
#                nothing ran. ci.yml then runs the GitHub shards, and their verdict is the check's.
# A retryable ending (error.retryable: coordinator_unreachable, connection_lost, batches_lost) runs
# once more before it counts. `run` exits 0 for passed and unreachable (the shards decide then; an
# annotation says so) and 1 for failed.
#
# Environment:
#   TESTAMENT_KEY          a client key: the owner mints it in the testament console (People, your
#                          own page, "+ New client key", named ci-3tears) and stores it as the
#                          repo's Actions secret
#   TESTAMENT_COORDINATOR  the pool's wire address (the repo's Actions variable of that name)
#   TESTAMENT_CONSOLE      the console's https URL, where the client is downloaded from
#   TESTAMENT_BRANCH       `run`: the branch the run is for (a pull request checks out a merge commit)
#   TESTAMENT_CI           `route`: "off" sends CI to the shards
#   TESTAMENT_PROBE_TIMEOUT  `route`: seconds the probe may take before it counts as no answer (default 60)
#   EVENT_NAME HEAD_REPO REPO   `route`: the event, the pull request's head repo, and this repo
#   OUT                    `run`: where the envelope and JUnit report go (default: a new temporary directory)
#
# THE CLIENT. testament is private and not on PyPI. The console serves its current release's wheel
# at /api/v1/cli/wheel to any client key, the same build its runners run, and names its sha256 in
# an X-Testament-Wheel-SHA256 header of the same response. Checking the file against that header
# is an integrity check of the download (a truncated or corrupted body is refused); it is not
# provenance, since one HTTPS response carries both. The trust is HTTPS to the console, which is
# why a non-https TESTAMENT_CONSOLE is refused before the key is sent.
set -uo pipefail

mode="${1:-}"
case "$mode" in
    route | run) ;;
    *)
        echo "usage: $0 route|run" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

# one line for $GITHUB_OUTPUT: newlines and carriage returns become spaces
oneline() { printf '%s' "$*" | tr '\r\n' '  '; }

output() { printf '%s=%s\n' "$1" "$(oneline "$2")" >>"${GITHUB_OUTPUT:-/dev/null}"; }

# an annotation on the run's page: level (notice|warning|error) and message
annotate() {
    local message
    message="$(oneline "${*:2}")"
    message="${message//%/%25}"
    echo "::$1 title=unit route::${message}"
}

summary() {
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        printf '%s\n' "$@" >>"$GITHUB_STEP_SUMMARY"
    else
        printf '%s\n' "$@"
    fi
}

# --- the client: the pool's own build, its download checked against the sha256 the response names.
# Sets `why`, and `level` (error for a refusal, warning otherwise), when it returns 1.
wheels=""
trap '[ -n "$wheels" ] && rm -rf "$wheels"' EXIT
install_client() {
    local console status headers wheel want got
    console="${TESTAMENT_CONSOLE%/}"
    case "$console" in
        https://*) ;;
        *)
            why="TESTAMENT_CONSOLE is not an https URL ('${console}'): the key is not sent to it"
            level=error
            return 1
            ;;
    esac
    wheels="$(mktemp -d)"
    # the key goes to curl on stdin (-K -), never in its argv
    status=$(printf 'header = "Authorization: Bearer %s"\n' "$TESTAMENT_KEY" |
        curl -sS --max-time 300 -K - -D "$wheels/headers" -o "$wheels/wheel.bin" -w '%{http_code}' \
            "$console/api/v1/cli/wheel") || status="unreachable"
    if [ "$status" != 200 ]; then
        why="the testament console did not hand over its client (HTTP ${status}): is the console up, TESTAMENT_KEY a current client key, and TESTAMENT_CONSOLE the console?"
        level=warning
        return 1
    fi
    headers="$(tr -d '\r' <"$wheels/headers")"
    wheel="$(printf '%s\n' "$headers" | sed -En 's/^[Cc]ontent-[Dd]isposition:.*filename="?([^";/]+)"?.*/\1/p' | tail -n 1)"
    want="$(printf '%s\n' "$headers" | sed -En 's/^[Xx]-[Tt]estament-[Ww]heel-[Ss][Hh][Aa]256: *([0-9a-f]+).*/\1/p' | tail -n 1)"
    got="$(sha256sum "$wheels/wheel.bin" | cut -d' ' -f1)"
    case "$wheel" in
        testament-*.whl) ;;
        *)
            why="the console's download named no testament wheel ('${wheel}'): not installing it"
            level=error
            return 1
            ;;
    esac
    if [ -z "$want" ] || [ "$got" != "$want" ]; then
        why="the downloaded client's sha256 is ${got}, not the '${want}' its response names: not installing it"
        level=error
        return 1
    fi
    mv "$wheels/wheel.bin" "$wheels/$wheel"
    if ! uv tool install --force --python 3.14 "$wheels/$wheel" </dev/null; then
        why="uv could not install ${wheel}"
        level=warning
        return 1
    fi
    PATH="$(uv tool dir --bin):$PATH"
    export PATH
    echo "installed ${wheel}"
}

# A push, or a pull request whose head is a branch of this repo.
same_repo() {
    case "${EVENT_NAME:-}" in
        push) return 0 ;;
        pull_request) [ -n "${HEAD_REPO:-}" ] && [ "${HEAD_REPO:-}" = "${REPO:-}" ] ;;
        *) return 1 ;;
    esac
}

# what a refusal envelope on stdin said: "<error.code>: <error.message>"
said() {
    python3 -c 'import json, sys
try:
    error = json.load(sys.stdin).get("error") or {}
except (ValueError, AttributeError):
    error = {}
print(str(error.get("code") or "no answer") + ": " + str(error.get("message") or ""))'
}

# --- route: the pool, or the shards and why
route() {
    local pool=false probe code probe_timeout="${TESTAMENT_PROBE_TIMEOUT:-60}"
    level=notice
    export TESTAMENT_KEY="${TESTAMENT_KEY:-}" TESTAMENT_COORDINATOR="${TESTAMENT_COORDINATOR:-}"
    if [ "$(printf '%s' "${TESTAMENT_CI:-}" | tr '[:upper:]' '[:lower:]')" = off ]; then
        why="the repo variable TESTAMENT_CI is off"
    elif [ -z "$TESTAMENT_KEY" ]; then
        why="no TESTAMENT_KEY secret reached this run (a fork's pull request gets none)"
    elif ! same_repo; then
        why="a ${EVENT_NAME:-unknown} event from ${HEAD_REPO:-another repository}, not a push or a pull request from a branch of ${REPO:-this repo}"
    elif [ -z "$TESTAMENT_COORDINATOR" ]; then
        why="no TESTAMENT_COORDINATOR variable names the pool"
        level=warning
    elif ! install_client; then
        why="pool unavailable at probe: ${why}"
    else
        code=0
        # to a file, not a command substitution, so the wait ends when `timeout` does; 124 is its
        # exit after stopping the probe, 137 after killing one that ignored the stop
        timeout -k 10 "$probe_timeout" testament run --probe --json >"$wheels/probe.json" || code=$?
        probe="$(cat "$wheels/probe.json")"
        if [ "$code" = 0 ]; then
            pool=true
            why="the pool answered the probe"
        elif [ "$code" = 124 ] || [ "$code" = 137 ]; then
            why="pool unavailable at probe: no answer within ${probe_timeout}s (exit ${code})"
            level=warning
        else
            why="pool unavailable at probe: $(printf '%s' "$probe" | said) (exit ${code})"
            level=warning
        fi
    fi
    output pool "$pool"
    output why "$why"
    if [ "$pool" = true ]; then
        echo "unit route: the testament pool (${why})"
        summary "### unit route: the testament pool" "" "$(oneline "$why")"
    else
        annotate "$level" "${why}: running the GitHub shards"
        summary "### unit route: the GitHub shards" "" "$(oneline "$why")"
    fi
}

# --- classify one pool run: prints "<passed|failed|unreachable> <retry yes|no> <why>"
classify() {
    python3 - "$1" "$2" <<'PY'
import json
import sys

# testament's endings that say the pool is down or was lost, not the code: the coordinator could
# not be reached or dropped the run, every runner holding a batch was lost, or no runner can take it
POOL_DOWN = {"coordinator_unreachable", "connection_lost", "batches_lost", "no_runner"}

code = int(sys.argv[1])
try:
    with open(sys.argv[2]) as f:
        envelope = json.load(f)
except (OSError, ValueError):
    envelope = None
if not isinstance(envelope, dict):
    envelope = None
error = (envelope or {}).get("error") or {}
counts = ((envelope or {}).get("report") or {}).get("counts") or {}
bad = int(counts.get("failed") or 0) + int(counts.get("error") or 0)
retry = "yes" if error.get("retryable") is True else "no"
said = f"{error.get('code')}: {error.get('message') or ''}".strip()

if envelope is None:
    print("failed", "no", f"exit {code} with no result envelope")
elif bad:
    print("failed", "no", f"{bad} test(s) failed or errored (exit {code})")
elif code == 0 and envelope.get("exit_code") == 0:
    print("passed", "no", f"counts {counts}")
elif code == 2 and error.get("code") in POOL_DOWN:
    print("unreachable", retry, said)
elif code == 5:
    print("failed", "no", "the pool collected no tests (exit 5)")
else:
    print("failed", "no", f"exit {code}: {said}" if error else f"exit {code}")
PY
}

# --- run: the suite on the pool
run() {
    : "${TESTAMENT_KEY:?set TESTAMENT_KEY, the CI client key (docs/testing.md, CI on the testament pool)}"
    : "${TESTAMENT_COORDINATOR:?set TESTAMENT_COORDINATOR, the wire address of the pool (the repo variable)}"
    : "${TESTAMENT_CONSOLE:?set TESTAMENT_CONSOLE, the https URL of the testament console}"
    export TESTAMENT_KEY TESTAMENT_COORDINATOR
    OUT="${OUT:-$(mktemp -d)}"
    mkdir -p "$OUT"

    local outcome retry verdict code attempt
    if ! install_client; then
        outcome=unreachable
        why="the client could not be had: ${why}"
    else
        for attempt in 1 2; do
            code=0
            testament run --fresh --no-cache --no-fallback --json=summary --junit-xml "$OUT/unit.xml" \
                -- -m "not loop_timing" >"$OUT/envelope.json" || code=$?
            verdict="$(classify "$code" "$OUT/envelope.json")"
            outcome="${verdict%% *}"
            verdict="${verdict#* }"
            retry="${verdict%% *}"
            why="${verdict#* }"
            if [ "$retry" = yes ] && [ "$attempt" = 1 ]; then
                annotate warning "the pool run ended retryable (${why}); running it once more"
                continue
            fi
            break
        done
        report "$OUT/envelope.json"
    fi
    output outcome "$outcome"
    output why "$why"
    case "$outcome" in
        passed)
            summary "" "**Passed on the pool.**"
            exit 0
            ;;
        unreachable)
            annotate warning "pool unreachable during the run (${why}): running the GitHub shards"
            summary "" "**Pool unreachable during the run** (${why}): the GitHub shards run instead, and their verdict is the check's."
            exit 0
            ;;
        *)
            echo "::error title=unit (testament pool)::$(oneline "$why")"
            summary "" "**Failed on the pool:** $(oneline "$why")"
            exit 1
            ;;
    esac
}

# --- what happened, on the run's page: counts, error, and each failing test
report() {
    python3 - "$1" <<'PY'
import json, os, sys

lines = ["### unit + enforcement (testament pool)", ""]
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError) as e:
    d = None
    lines.append(f"No result envelope: {e}")
if d is not None:
    r = d.get("report") or {}
    lines.append(f"exit {d.get('exit_code')}; counts {r.get('counts')}")
    if d.get("fallback"):
        lines.append(f"**Ran on GitHub's runner, not the pool:** {d['fallback'].get('reason')}")
    if d.get("error"):
        lines.append(f"**Error** `{d['error'].get('code')}`: {d['error'].get('message')}")
    if r.get("missing"):
        lines.append(f"**Never reported ({len(r['missing'])}):** " + ", ".join(r["missing"][:20]))
    bad = [i for i in r.get("items", []) if i.get("outcome") in ("failed", "error")]
    for i in bad[:50]:
        text = (i.get("message") or "").strip().splitlines() or [""]
        said = [t[1:].strip() for t in text if t.startswith("E ")]
        last = (said[0] if said else text[-1])[:300]
        lines.append(f"- `{i['id']}` {i['outcome']} on {i.get('runner')}: {last}")
        title = i["id"].replace("%", "%25").replace(":", "%3A").replace(",", "%2C")
        print(f"::error title={title}::{last.replace('%', '%25')}")
summary = os.environ.get("GITHUB_STEP_SUMMARY")
if summary:
    with open(summary, "a") as f:
        f.write("\n".join(lines) + "\n")
else:
    print("\n".join(lines))
PY
}

"$mode"
