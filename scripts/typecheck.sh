#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run mypy across the typechecked packages. The agent-namespace move
# (packages/agent-memory -> packages/agent/memory, etc.) nests src/
# two levels deep which mypy cannot resolve from file paths alone, so
# we address the modules by ``-p threetears.<x>``, resolving them through
# MYPYPATH + the installed-editable layout.
#
# This is the SINGLE SOURCE OF TRUTH for the mypy target set: CI calls this
# script (see .github/workflows/ci.yml) so the two can never drift apart.
MYPYPATH=packages/core/src:packages/nats/src:packages/observe/src:packages/agent/acl/src:packages/agent/identity/src:packages/agent/intention/src:packages/agent/knowledge/src:packages/agent/memory/src:packages/agent/skills/src:packages/agent/tools/src:packages/agent/wake/src:packages/channels/src:packages/datasources/src:packages/langgraph/src:packages/media-contracts/src:packages/object-store/src:packages/backup/src \
    uv run mypy \
        --explicit-package-bases \
        -p threetears.core \
        -p threetears.agent.identity \
        -p threetears.agent.intention \
        -p threetears.agent.knowledge \
        -p threetears.agent.memory \
        -p threetears.agent.skills \
        -p threetears.agent.tools \
        -p threetears.agent.wake \
        -p threetears.channels \
        -p threetears.datasources \
        -p threetears.media.contracts \
        -p threetears.object_store \
        -p threetears.backup \
        "$@"
