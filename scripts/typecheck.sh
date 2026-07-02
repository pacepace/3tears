#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run mypy across the typechecked packages. The agent-namespace move
# (packages/agent-memory -> packages/agent/memory, etc.) nests src/
# two levels deep which mypy cannot resolve from file paths alone, so
# we follow the CI workflow's invocation: addressing the modules by
# ``-p threetears.<x>`` resolves them through MYPYPATH + the
# installed-editable layout.
MYPYPATH=packages/core/src:packages/nats/src:packages/observe/src:packages/agent/acl/src:packages/agent/memory/src:packages/agent/skills/src:packages/agent/tools/src:packages/agent/wake/src:packages/channels/src:packages/media-contracts/src:packages/object-store/src \
    uv run mypy \
        --explicit-package-bases \
        -p threetears.core \
        -p threetears.agent.memory \
        -p threetears.agent.skills \
        -p threetears.agent.tools \
        -p threetears.agent.wake \
        -p threetears.channels \
        -p threetears.media.contracts \
        -p threetears.object_store \
        "$@"
