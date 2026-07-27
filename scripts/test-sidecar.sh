#!/usr/bin/env bash
set -euo pipefail

# The nodriver sidecar's own suite.
#
# Separate from ./scripts/test.sh because the sidecar is a separate deployable with its own
# venv: nodriver is AGPL-3.0 and is deliberately never installed into the workspace venv, so
# the workspace's pytest config carries `--ignore=packages/scrape/sidecar` and cannot run
# these.
#
# Split out, but NOT optional. Until this existed, ruff formatted the sidecar's source (it is
# inside the ruff target tree) while nothing anywhere executed the result -- and a ruff
# autofix wrote `except OSError, ProcessLookupError:` into hitl.py, a syntax error that
# passed lint, passed mypy (the sidecar is outside its file list too) and passed the whole
# whole workspace suite. Only this suite could catch it, and nothing ran this suite.
#
# Requires `uv`, which resolves the sidecar's own pyproject.toml.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIDECAR="$REPO_ROOT/packages/scrape/sidecar"

cd "$SIDECAR"
uv run pytest tests/ "$@"
