#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run the tests of every package that uses the MCP SDK against the FLOORS they declare, not the
# versions uv.lock happens to pin.
#
# Why this exists: the workspace suite only ever runs against the locked mcp and fastmcp, which are
# always recent, so code that uses a name a later release added passes every other check and breaks
# at import or at runtime for a consumer that resolves the floor. scripts/test-otel-floor.sh exists
# because exactly that happened to observe's OpenTelemetry floor.
#
# What: 3tears-mcp (`mcp>=X,<Y`), 3tears-models' claude-cli extra (the same mcp floor;
# tests/enforcement/test_mcp_floor.py keeps the two equal) and 3tears-evals' fastmcp extra
# (`fastmcp>=F,<G`, `fastmcp-slim>=F,<G`). The floors are read from the pyprojects, never restated
# here, so raising a floor is what moves this check.
#
# How: overlay exactly those versions on the locked environment and run the mcp and models unit
# tests and the evals tests that touch an MCP SDK. Overlaying (rather than an isolated environment,
# as the OpenTelemetry floor does) keeps the root conftest and the family packages the tests need.
# Before the run it checks the overlay took, or it would report a pass on the locked versions.
#
# Usage: ./scripts/test-mcp-floor.sh [extra pytest args...]

read -r MCP_FLOOR FASTMCP_FLOOR FASTMCP_SLIM_FLOOR < <(uv run --no-project --python 3.14 python - <<'PY'
import re
import sys
import tomllib


def floor(requirements, name, where):
    found = [
        match.group(1)
        for requirement in requirements
        if (match := re.fullmatch(rf"{re.escape(name)}>=([0-9][0-9A-Za-z.]*),<[0-9A-Za-z.]+", requirement.strip()))
    ]
    if len(found) != 1:
        sys.exit(f"expected one `{name}>=X,<Y` requirement in {where}, found {len(found)}; declare one")
    return found[0]


with open("packages/mcp/pyproject.toml", "rb") as handle:
    mcp = floor(tomllib.load(handle)["project"]["dependencies"], "mcp", "packages/mcp dependencies")
with open("packages/evals/pyproject.toml", "rb") as handle:
    extra = tomllib.load(handle)["project"]["optional-dependencies"]["fastmcp"]
where = "packages/evals optional-dependencies.fastmcp"
print(mcp, floor(extra, "fastmcp", where), floor(extra, "fastmcp-slim", where))
PY
)

if [ -z "${FASTMCP_SLIM_FLOOR:-}" ]; then
    echo "could not read the MCP floors from the pyprojects; refusing to report a pass" >&2
    exit 1
fi

OVERLAY=(--with "mcp==${MCP_FLOOR}" --with "fastmcp==${FASTMCP_FLOOR}" --with "fastmcp-slim==${FASTMCP_SLIM_FLOOR}")

echo "MCP floors under test: mcp==${MCP_FLOOR} fastmcp==${FASTMCP_FLOOR} fastmcp-slim==${FASTMCP_SLIM_FLOOR}"

uv run --frozen "${OVERLAY[@]}" python - "$MCP_FLOOR" "$FASTMCP_FLOOR" "$FASTMCP_SLIM_FLOOR" <<'PY'
import sys
from importlib.metadata import version

from packaging.version import Version

for name, floor in zip(("mcp", "fastmcp", "fastmcp-slim"), sys.argv[1:], strict=True):
    installed = version(name)
    if Version(installed) != Version(floor):
        sys.exit(f"the overlay resolved {name} {installed}, not the floor {floor}; refusing to report a pass")
PY

# the evals tests that touch an MCP SDK; the rest of evals never imports one
EVALS_MCP_TESTS=()
while IFS= read -r test_file; do
    EVALS_MCP_TESTS+=("$test_file")
done < <(grep -rlE 'fastmcp|mcp\.' packages/evals/tests --include='test_*.py' | sort)

exec uv run --frozen "${OVERLAY[@]}" \
    pytest packages/mcp packages/models "${EVALS_MCP_TESTS[@]}" \
    -m "not integration" -n 4 --dist loadgroup "$@"
