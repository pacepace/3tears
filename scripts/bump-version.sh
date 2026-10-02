#!/usr/bin/env bash
# Bump, release, sync or verify the workspace version, in lockstep:
#   ./scripts/bump-version.sh patch|minor|major|release|sync
#   ./scripts/bump-version.sh verify X.Y.Z      # the release workflow's pre-flight
# The logic has one owner, threetears.enforcement.release (threetears-release); what it
# touches here is [tool.threetears-release] in the root pyproject.toml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec uv run --frozen --project "$ROOT" python -m threetears.enforcement.release --root "$ROOT" "$@"
