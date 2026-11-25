#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run 3tears-mcp's unit tests against the mcp SDK FLOOR that 3tears-mcp declares, not the version
# uv.lock happens to pin.
#
# Why this exists: the workspace suite only ever runs 3tears-mcp against the locked mcp, which is
# always recent, so code that uses a name a later 2.x added passes every other check and breaks at
# import or at runtime for a consumer that resolves the floor. scripts/test-otel-floor.sh exists
# because exactly that happened to observe's OpenTelemetry floor.
#
# How: read the `mcp>=X,<Y` requirement of packages/mcp/pyproject.toml, overlay `mcp==X` on the
# locked environment, and run the mcp unit tests. The floor is read from the pyproject, never
# restated here, so raising the floor is what moves this check. Overlaying (rather than an isolated
# environment, as the OpenTelemetry floor does) keeps the root conftest and the grants extra's
# family packages, which the mcp tests need; mcp is the only version that moves.
#
# Usage: ./scripts/test-mcp-floor.sh [extra pytest args...]

MCP_DIR="packages/mcp"

MCP_FLOOR="$(uv run --no-project --python 3.14 python - "$MCP_DIR/pyproject.toml" <<'PY'
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    dependencies = tomllib.load(handle)["project"]["dependencies"]
floors = []
for requirement in dependencies:
    match = re.fullmatch(r"mcp>=([0-9][0-9A-Za-z.]*),<[0-9A-Za-z.]+", requirement.strip())
    if match is not None:
        floors.append(match.group(1))
if len(floors) != 1:
    sys.exit(f"expected one `mcp>=X,<Y` requirement in {sys.argv[1]}, found {len(floors)}; declare one")
print(floors[0])
PY
)"

echo "3tears-mcp mcp floor under test: mcp==${MCP_FLOOR}"

# the overlay must win over the lock, or this would report a pass on the locked mcp
uv run --frozen --with "mcp==${MCP_FLOOR}" python - "$MCP_FLOOR" <<'PY'
import sys
from importlib.metadata import version

from packaging.version import Version

installed = version("mcp")
if Version(installed) != Version(sys.argv[1]):
    sys.exit(f"the overlay resolved mcp {installed}, not the floor {sys.argv[1]}; refusing to report a pass")
PY

exec uv run --frozen --with "mcp==${MCP_FLOOR}" \
    pytest "$MCP_DIR" -m "not integration" -n 4 --dist loadgroup "$@"
