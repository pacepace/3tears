#!/usr/bin/env bash
# CI's unit + enforcement suite on the testament pool: ci.yml's `unit (testament pool)` job.
#
#   TESTAMENT_KEY=... TESTAMENT_COORDINATOR=... ./scripts/ci-unit-pool.sh
#
# The same tests `./scripts/ci-unit-tests.sh` runs on GitHub's runners: .testament.toml selects
# packages/ and tests/ with -m "not integration", and the pool spreads them over its runners, so
# GitHub's job only waits. Used when the repo holds the TESTAMENT_KEY secret; without it (a fork's
# pull request never gets secrets) ci.yml runs the GitHub shards instead.
#
# Environment:
#   TESTAMENT_KEY          a client key: the owner mints it in the testament console (People, your
#                          own page, "+ New client key", named ci-3tears) and stores it as the
#                          repo's Actions secret
#   TESTAMENT_COORDINATOR  the pool's wire address (the repo's Actions variable of that name)
#   TESTAMENT_CONSOLE      the console's https URL, where the client is downloaded from
#   TESTAMENT_BRANCH       the branch the run is for (a pull request checks out a merge commit)
#   OUT                    where the envelope and JUnit report go (default: a new temporary directory)
#
# THE CLIENT. testament is private and not on PyPI. The console serves its current release's wheel
# at /api/v1/cli/wheel to any client key, the same build its runners run, with the sha256 its
# signed release names in X-Testament-Wheel-SHA256. So the CI key installs the client too: no
# GitHub token for the private repo, and the client is always the pool's own version.
#
# --no-fallback: an unreachable pool fails the job rather than quietly running the whole suite on
# GitHub's runner. A run that ends retryable (exit 2 with error.retryable: lost batches, a pool
# restart) runs once more, as testament's own CI does.
set -uo pipefail

: "${TESTAMENT_KEY:?set TESTAMENT_KEY, the CI client key (docs/testing.md, CI on the testament pool)}"
: "${TESTAMENT_COORDINATOR:?set TESTAMENT_COORDINATOR, the pool's wire address (the repo variable)}"
: "${TESTAMENT_CONSOLE:?set TESTAMENT_CONSOLE, the testament console's https URL}"
export TESTAMENT_KEY TESTAMENT_COORDINATOR

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="${OUT:-$(mktemp -d)}"
mkdir -p "$OUT"

# --- the client: the pool's own build, checked against the digest the pool's release names
console="${TESTAMENT_CONSOLE%/}"
wheels="$(mktemp -d)"
trap 'rm -rf "$wheels"' EXIT
status=$(printf 'header = "Authorization: Bearer %s"\n' "$TESTAMENT_KEY" |
    curl -sS --max-time 300 -K - -D "$wheels/headers" -o "$wheels/wheel.bin" -w '%{http_code}' \
        "$console/api/v1/cli/wheel") || status="unreachable"
if [ "$status" != 200 ]; then
    echo "::error::the testament console did not hand over its client (HTTP ${status}): is TESTAMENT_KEY a current client key, and TESTAMENT_CONSOLE the console?"
    exit 1
fi
headers="$(tr -d '\r' <"$wheels/headers")"
wheel="$(printf '%s\n' "$headers" | sed -En 's/^[Cc]ontent-[Dd]isposition:.*filename="?([^";/]+)"?.*/\1/p' | tail -n 1)"
want="$(printf '%s\n' "$headers" | sed -En 's/^[Xx]-[Tt]estament-[Ww]heel-[Ss][Hh][Aa]256: *([0-9a-f]+).*/\1/p' | tail -n 1)"
got="$(sha256sum "$wheels/wheel.bin" | cut -d' ' -f1)"
case "$wheel" in
    testament-*.whl) ;;
    *) echo "::error::the console's download named no testament wheel ('${wheel}')"; exit 1 ;;
esac
if [ -z "$want" ] || [ "$got" != "$want" ]; then
    echo "::error::the client's sha256 is ${got}, not the '${want}' the pool's release names: not installing it"
    exit 1
fi
mv "$wheels/wheel.bin" "$wheels/$wheel"
uv tool install --force --python 3.14 "$wheels/$wheel" </dev/null || { echo "::error::uv could not install ${wheel}"; exit 1; }
PATH="$(uv tool dir --bin):$PATH"
export PATH
echo "installed ${wheel}"

# --- the suite
code=0
for attempt in 1 2; do
    code=0
    testament run --json=summary --junit-xml "$OUT/unit.xml" --no-fallback >"$OUT/envelope.json" || code=$?
    retry=$(python3 - "$OUT/envelope.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print("no")
else:
    print("yes" if d.get("exit_code") == 2 and (d.get("error") or {}).get("retryable") else "no")
PY
    )
    if [ "$retry" = yes ] && [ "$attempt" = 1 ]; then
        echo "::warning::the pool run ended retryable; running it once more"
        continue
    fi
    break
done

# --- what happened, on the run's page: counts, fallback, error, and each failing test
python3 - "$OUT/envelope.json" <<'PY'
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

exit "$code"
