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
# Every package in this workspace is strict-mypy checked here, and ``[tool.mypy] files``
# in the root pyproject lists the same set of source trees. Both matter and they are not
# interchangeable: this script is what CI runs, while ``files`` is what a bare
# ``uv run mypy`` uses -- an editor, or anyone invoking mypy directly. When they disagree
# the two report different answers on the same code, which is worse than either being
# wrong, because whichever one you happen to run looks authoritative.
#
# They did disagree: 28 targets here against 21 entries there, so a bare mypy silently
# skipped six packages this script checks. Keep both in step. A package added to the
# workspace without a line in each is a package nobody is checking, and the gap is
# invisible because the gate still passes.
MYPYPATH=packages/core/src:packages/nats/src:packages/observe/src:packages/agent/acl/src:packages/agent/audit/src:packages/agent/identity/src:packages/agent/intention/src:packages/agent/knowledge/src:packages/agent/memory/src:packages/agent/skills/src:packages/agent/tools/src:packages/agent/wake/src:packages/channels/src:packages/datasources/src:packages/enforcement/src:packages/epoch/src:packages/langgraph/src:packages/media-contracts/src:packages/models/src:packages/object-store/src:packages/registry/src:packages/scheduled-jobs/src:packages/scrape/src:packages/backup/src:packages/geo/src:packages/iam/src \
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
        -p threetears.geo \
        -p threetears.iam \
        "$@"
