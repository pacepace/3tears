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
#
# The target list now matches ``[tool.mypy] files``: every package in this workspace
# is strict-mypy checked here. Keep it that way -- a package added to the workspace
# without a line below is a package nobody is checking, and the gap is invisible
# because the gate still passes. Add the ``-p`` target in the same change that adds
# the package.
MYPYPATH=packages/core/src:packages/nats/src:packages/observe/src:packages/agent/acl/src:packages/agent/audit/src:packages/agent/identity/src:packages/agent/intention/src:packages/agent/knowledge/src:packages/agent/memory/src:packages/agent/skills/src:packages/agent/tools/src:packages/agent/wake/src:packages/channels/src:packages/datasources/src:packages/enforcement/src:packages/epoch/src:packages/langgraph/src:packages/media-contracts/src:packages/models/src:packages/object-store/src:packages/registry/src:packages/scheduled-jobs/src:packages/scrape/src:packages/backup/src \
    uv run mypy \
        --explicit-package-bases \
        -p threetears.core \
        -p threetears.knowledge \
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
        -p threetears.scrape \
        -p threetears.registry \
        -p threetears.epoch \
        -p threetears.scheduled_jobs \
        -p threetears.enforcement \
        -p threetears.agent.acl \
        -p threetears.agent.audit \
        -p threetears.nats \
        -p threetears.observe \
        -p threetears.models \
        -p threetears.mcp \
        -p threetears.conversations \
        -p threetears.langgraph \
        -p threetears.agent.workspace \
        "$@"
